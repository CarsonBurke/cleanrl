import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import gymnasium as gym


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl" / "lejepa" / "ppo_continuous_action_lejepa_risktrace_v8.py"
)
SPEC = importlib.util.spec_from_file_location("lejepa_risktrace_v8", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_actor_credit_horizon_is_16_while_successor_reaches_100():
    args = MODULE.Args()
    assert args.learn_trace
    assert args.gae_lambda == 0.95
    assert args.control_encoder_tau == 1.0
    assert not args.norm_adv
    assert MODULE.TD_HORIZONS[-1] == 100


def test_one_step_normalized_shell_target_matches_cumulative_definition():
    torch.manual_seed(0)
    t, b, j, d = 5, 3, len(MODULE.TD_BETAS), 7
    phi = torch.randn(t, b, d)
    next_shells = torch.randn(t, b, j, d)
    bootstrap = torch.randint(0, 2, (t, b), dtype=torch.float32)
    betas = torch.tensor(MODULE.TD_BETAS)

    widths = torch.tensor(MODULE.TD_SHELL_WIDTHS)
    targets = MODULE.build_normalized_shell_targets(
        phi, next_shells, bootstrap, betas, widths
    )
    cumulative_next = MODULE.normalized_shells_to_cumulative(
        next_shells, widths
    )
    cumulative_targets = MODULE.normalized_shells_to_cumulative(targets, widths)
    for index, beta in enumerate(betas):
        expected = (
            phi
            + beta * bootstrap.unsqueeze(-1) * cumulative_next[..., index, :]
        )
        torch.testing.assert_close(
            cumulative_targets[..., index, :], expected, rtol=1e-5, atol=1e-6
        )


def test_weighted_shell_innovations_telescope_over_time():
    torch.manual_seed(2)
    t, b, j, d = 8, 2, len(MODULE.TD_BETAS), 3
    shells = torch.randn(t + 1, b, j, d)
    phi = torch.randn(t, b, d)
    betas = torch.tensor(MODULE.TD_BETAS)
    widths = torch.tensor(MODULE.TD_SHELL_WIDTHS)
    targets = MODULE.build_normalized_shell_targets(
        phi, shells[1:], torch.ones(t, b), betas, widths
    )
    gamma = betas[-1].item()
    innovations = (
        (targets - shells[:-1]) * widths.view(1, 1, -1, 1)
    ).sum(dim=2)
    actual = MODULE.fixed_horizon_sum(
        innovations, torch.ones(t, b), gamma=gamma, horizon=4
    )
    for start in range(t):
        length = min(4, t - start)
        discounted_phi = sum(
            gamma**offset * phi[start + offset]
            for offset in range(length)
        )
        expected = (
            discounted_phi
            + gamma**length
              * (shells[start + length] * widths.view(1, -1, 1)).sum(dim=1)
            - (shells[start] * widths.view(1, -1, 1)).sum(dim=1)
        )
        torch.testing.assert_close(
            actual[start], expected, rtol=2e-5, atol=2e-6
        )


def test_full_suffix_normalized_shells_supervise_every_horizon_from_zero():
    t, b, d = 128, 2, 3
    phi = torch.ones(t, b, d)
    next_shells = torch.zeros(t, b, len(MODULE.TD_HORIZONS), d)
    bootstrap = torch.ones(t, b)
    continuation = torch.ones(t, b)
    betas = torch.tensor(MODULE.TD_BETAS)
    targets = MODULE.build_full_suffix_shell_targets(
        phi,
        next_shells,
        bootstrap,
        continuation,
        betas,
        torch.tensor(MODULE.TD_SHELL_WIDTHS),
    )
    # Every complete successor head gets a direct nonzero target immediately.
    assert (targets[:28].abs().mean(dim=(0, 1, 3)) > 0).all()


def test_full_suffix_constant_fixed_point_is_one_at_every_shell():
    t, b, d = 128, 2, 1
    phi = torch.ones(t, b, d)
    next_shells = torch.ones(t, b, len(MODULE.TD_HORIZONS), d)
    targets = MODULE.build_full_suffix_shell_targets(
        phi,
        next_shells,
        torch.ones(t, b),
        torch.ones(t, b),
        torch.tensor(MODULE.TD_BETAS),
        torch.tensor(MODULE.TD_SHELL_WIDTHS),
    )
    torch.testing.assert_close(targets, torch.ones_like(targets), rtol=2e-5, atol=2e-5)


def test_full_suffix_target_bootstraps_truncation_and_rollout_tail():
    phi = torch.ones(4, 1, 1)
    next_shell = torch.full((4, 1, 1, 1), 5.0)
    continuation = torch.tensor([[1.0], [0.0], [1.0], [1.0]])
    truncation_bootstrap = torch.ones(4, 1)
    actual = MODULE.build_full_suffix_shell_targets(
        phi,
        next_shell,
        truncation_bootstrap,
        continuation,
        torch.tensor([0.5]),
        torch.tensor([2.0]),
    )
    expected = torch.tensor([2.0, 3.0, 2.0, 3.0]).view(4, 1, 1, 1)
    torch.testing.assert_close(actual, expected)

    termination_bootstrap = truncation_bootstrap.clone()
    termination_bootstrap[1] = 0.0
    terminated = MODULE.build_full_suffix_shell_targets(
        phi,
        next_shell,
        termination_bootstrap,
        continuation,
        torch.tensor([0.5]),
        torch.tensor([2.0]),
    )
    torch.testing.assert_close(
        terminated[:2], torch.tensor([0.75, 0.5]).view(2, 1, 1, 1)
    )


def test_fixed_horizon_sum_is_not_a_lambda_mixture_and_stops_at_boundaries():
    signal = torch.arange(1, 7, dtype=torch.float32).view(6, 1, 1)
    continuation = torch.tensor([[1.0], [0.0], [1.0], [1.0], [1.0], [1.0]])

    actual = MODULE.fixed_horizon_sum(signal, continuation, gamma=0.5, horizon=3)

    expected = torch.tensor(
        [
            1.0 + 0.5 * 2.0,
            2.0,
            3.0 + 0.5 * 4.0 + 0.25 * 5.0,
            4.0 + 0.5 * 5.0 + 0.25 * 6.0,
            5.0 + 0.5 * 6.0,
            6.0,
        ]
    ).view(6, 1, 1)
    torch.testing.assert_close(actual, expected)


def test_reward_residual_makes_vector_credit_exact():
    torch.manual_seed(1)
    t, b, d = 9, 2, 4
    gamma, horizon = 0.91, 4
    phi = torch.randn(t, b, d)
    vector_delta = torch.randn(t, b, d)
    reward_weights = torch.randn(d)
    residual = torch.randn(t, b)
    reward_delta = vector_delta @ reward_weights + residual
    continuation = torch.ones(t, b)
    continuation[3, 0] = 0.0

    vector_credit = MODULE.fixed_horizon_sum(
        vector_delta, continuation, gamma, horizon
    )
    residual_credit = MODULE.fixed_horizon_sum(
        residual, continuation, gamma, horizon
    )
    scalar_credit = MODULE.fixed_horizon_sum(
        reward_delta, continuation, gamma, horizon
    )

    torch.testing.assert_close(
        vector_credit @ reward_weights + residual_credit,
        scalar_credit,
        rtol=1e-5,
        atol=1e-6,
    )


def test_truncation_bootstraps_but_cuts_credit_and_tail_shortens():
    phi = torch.ones(4, 1, 1)
    next_shells = torch.ones(4, 1, len(MODULE.TD_BETAS), 1)
    betas = torch.tensor(MODULE.TD_BETAS)
    # At t=1 this represents a time-limit truncation: use final_observation for the
    # one-step target, but do not carry credit into the reset episode at t=2.
    bootstrap = torch.ones(4, 1)
    targets = MODULE.build_normalized_shell_targets(
        phi,
        next_shells,
        bootstrap,
        betas,
        torch.tensor(MODULE.TD_SHELL_WIDTHS),
    )
    torch.testing.assert_close(targets[1], torch.ones_like(targets[1]))

    innovation = torch.ones(4, 1, 1)
    continuation = torch.tensor([[1.0], [0.0], [1.0], [1.0]])
    actual = MODULE.fixed_horizon_sum(innovation, continuation, gamma=0.5, horizon=4)
    expected = torch.tensor([1.5, 1.0, 1.5, 1.0]).view(4, 1, 1)
    torch.testing.assert_close(actual, expected)


def test_geometric_lejepa_predicts_every_horizon_and_backpropagates():
    args = SimpleNamespace(
        emb_dim=8,
        ssl_hidden=16,
        seq_len=max(MODULE.LEJEPA_HORIZONS) + 1,
        pred_depth=1,
        pred_heads=2,
        pred_mlp_dim=16,
        pred_dim_head=4,
        sigreg_num_proj=8,
        sigreg_proj_chunk=4,
    )
    model = MODULE.LeJepaSSL(obs_dim=5, act_dim=2, args=args)
    obs = torch.randn(2, args.seq_len, 5)
    actions = torch.randn(2, args.seq_len, 2)
    continuation = torch.ones(2, args.seq_len)
    continuation[1, 20] = 0.0

    loss, pred_loss, sigreg_loss, horizon_losses = model(
        obs, actions, continuation, sigreg_weight=0.01
    )
    assert torch.isfinite(torch.stack([loss, pred_loss, sigreg_loss])).all()
    assert horizon_losses.shape == (len(MODULE.LEJEPA_HORIZONS),)
    assert torch.isfinite(horizon_losses).all()
    loss.backward()
    for horizon in MODULE.LEJEPA_HORIZONS:
        grads = [
            parameter.grad
            for parameter in model.pred_projs[str(horizon)].parameters()
        ]
        assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
        trace_grads = [
            parameter.grad
            for parameter in model.action_trace_encoders[str(horizon)].parameters()
        ]
        assert all(
            grad is not None and torch.isfinite(grad).all()
            for grad in trace_grads
        )


def test_action_trace_encoder_preserves_temporal_order_and_window_alignment():
    conv = torch.nn.Conv1d(1, 1, kernel_size=2, bias=False)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([[[1.0, -1.0]]]))
    actions = torch.tensor([[[1.0, 3.0, 8.0]]])
    trace = conv(actions)
    torch.testing.assert_close(trace, torch.tensor([[[-2.0, -5.0]]]))
    reversed_trace = conv(actions.flip(-1))
    assert not torch.equal(trace, reversed_trace)


def test_unit_shell_targets_give_equal_output_gradient_at_every_horizon():
    prediction = torch.zeros(
        1, len(MODULE.TD_HORIZONS), 1, requires_grad=True
    )
    target = torch.ones_like(prediction)
    loss = (prediction - target).square().mean()
    loss.backward()
    expected = prediction.grad[:, :1].expand_as(prediction.grad)
    torch.testing.assert_close(prediction.grad, expected)
    assert MODULE.TD_HORIZONS == (1, 2, 4, 8, 16, 32, 64, 100)


def test_latent_frame_alignment_removes_rotation_and_translation():
    torch.manual_seed(4)
    reference = torch.randn(256, 8)
    q, _ = torch.linalg.qr(torch.randn(8, 8))
    source = reference @ q + torch.randn(8)
    rotation, bias, aligned = MODULE.align_latent_frame(source, reference)
    torch.testing.assert_close(rotation.T @ rotation, torch.eye(8), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(aligned, reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(source @ rotation + bias, aligned)


def test_policy_and_state_critic_consume_raw_state_plus_embedding():
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-1.0, 1.0, shape=(5,)),
        single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,)),
    )
    args = SimpleNamespace(
        emb_dim=8,
        hidden=16,
        k_blocks=1,
        n_experts=2,
        share_backbone=True,
        critic_mtp_horizon=len(MODULE.TD_HORIZONS),
        actor_dist="beta",
    )
    agent = MODULE.Agent(envs, args)
    policy_input = torch.randn(4, 5 + args.emb_dim)
    (
        action,
        latent,
        log_prob,
        log_prob_dim,
        entropy,
        successors,
        alpha,
        beta,
    ) = agent.get_action_and_value(policy_input)
    assert action.shape == latent.shape == (4, 2)
    assert log_prob.shape == entropy.shape == (4,)
    assert log_prob_dim.shape == (4, 2)
    assert alpha.shape == beta.shape == (4, 2)
    torch.testing.assert_close(log_prob_dim.sum(-1), log_prob)
    assert successors.shape == (
        4,
        len(MODULE.TD_HORIZONS),
        args.emb_dim + 5 + 2 * 2 + 1,
    )


def test_learned_vector_gae_matches_constant_lambda_recursion():
    delta = torch.arange(1, 13, dtype=torch.float32).reshape(3, 2, 2)
    continuation = torch.ones(3, 2)
    state_lambda = torch.full((3, 2), 0.5)
    actual = MODULE.learned_vector_gae(
        delta, continuation, gamma=0.9, state_lambda=state_lambda
    )
    expected = torch.zeros_like(delta)
    expected[2] = delta[2]
    expected[1] = delta[1] + 0.9 * 0.5 * expected[2]
    expected[0] = delta[0] + 0.9 * 0.5 * expected[1]
    torch.testing.assert_close(actual, expected)


def test_learned_vector_gae_cuts_reset_but_uses_next_state_lambda():
    delta = torch.ones(4, 1, 1)
    continuation = torch.tensor([[1.0], [0.0], [1.0], [1.0]])
    state_lambda = torch.tensor([[0.0], [0.25], [0.5], [1.0]])
    actual = MODULE.learned_vector_gae(
        delta, continuation, gamma=1.0, state_lambda=state_lambda
    )
    expected = torch.tensor([1.25, 1.0, 2.0, 1.0]).reshape(4, 1, 1)
    torch.testing.assert_close(actual, expected)


def test_recursive_trace_lambda_has_correct_risk_limits():
    bootstrap = torch.tensor([[2.0], [2.0], [2.0]])
    direct = torch.zeros_like(bootstrap)
    recursive = torch.tensor([[0.0], [0.0], [3.0]])
    variance = torch.tensor([[0.0], [4.0], [0.0]])
    actual = MODULE.optimal_recursive_lambda(
        bootstrap, direct, recursive, variance
    )
    torch.testing.assert_close(actual, torch.tensor([1.0, 0.5, 0.0]))


def test_vector_whitening_preserves_reward_projection():
    torch.manual_seed(8)
    advantage = torch.randn(17, 3, 5)
    covector = torch.randn(5)
    whitened, transformed, eigenvalues, _, _ = MODULE.whiten_vector_advantage(
        advantage, covector, ridge=1e-3
    )
    centered_scalar = advantage @ covector
    centered_scalar = centered_scalar - centered_scalar.mean()
    torch.testing.assert_close(
        whitened @ transformed,
        centered_scalar,
        atol=2e-5,
        rtol=2e-5,
    )
    assert torch.all(eigenvalues > 0)


def test_direct_successor_risk_is_state_only_and_positive_variance():
    model = MODULE.DirectSuccessorRisk(
        input_dim=7,
        sf_dim=11,
        hidden=16,
        variance_floor=1e-4,
    )
    embedding = torch.randn(13, 7)
    direct_mean, direct_variance, trace_mean, trace_variance = model(embedding)
    assert direct_mean.shape == direct_variance.shape == (13, 11)
    assert trace_mean.shape == trace_variance.shape == (13, 11)
    assert torch.all(direct_variance > 0)
    assert torch.all(trace_variance > 0)
    loss = (
        direct_mean.square()
        + direct_variance.log()
        + trace_mean.square()
        + trace_variance.log()
    ).mean()
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_zero_router_reduces_to_joint_scalar_policy_gradient():
    torch.manual_seed(12)
    batch, actions, dims = 19, 3, 5
    logits = (0.01 * torch.randn(batch, actions)).requires_grad_()
    common = torch.randn(batch, dims)
    covector = torch.randn(dims)
    joint_ratio = logits.sum(-1).exp()
    marginal_ratio = logits.exp()
    routed = common.unsqueeze(1).expand(-1, actions, -1)
    vector_loss = MODULE.vector_policy_surrogate(
        joint_ratio,
        marginal_ratio,
        common,
        routed,
        covector,
        clip_coef=10.0,
        clip_coef_high=10.0,
    )
    vector_gradient = torch.autograd.grad(
        vector_loss, logits, grad_outputs=covector, retain_graph=True
    )[0]
    scalar_loss = -(joint_ratio * (common @ covector)).mean()
    scalar_gradient = torch.autograd.grad(scalar_loss, logits)[0]
    torch.testing.assert_close(vector_gradient, scalar_gradient)


def test_nonzero_zero_sum_router_changes_policy_gradient_not_scalar_credit():
    torch.manual_seed(13)
    batch, actions, dims = 23, 4, 6
    logits = torch.zeros(batch, actions, requires_grad=True)
    common = torch.randn(batch, dims)
    differential = torch.randn(batch, actions, dims)
    differential = differential - differential.mean(1, keepdim=True)
    routed = common.unsqueeze(1) + differential
    covector = torch.randn(dims)
    vector_loss = MODULE.vector_policy_surrogate(
        logits.sum(-1).exp(),
        logits.exp(),
        common,
        routed,
        covector,
        clip_coef=0.2,
        clip_coef_high=0.28,
    )
    vector_gradient = torch.autograd.grad(
        vector_loss, logits, grad_outputs=covector
    )[0]
    scalar_loss = -(logits.sum(-1).exp() * (common @ covector)).mean()
    scalar_gradient = torch.autograd.grad(scalar_loss, logits)[0]
    assert not torch.allclose(vector_gradient, scalar_gradient)
    torch.testing.assert_close(
        differential.sum(1),
        torch.zeros_like(differential[:, 0]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_joint_common_term_has_sign_aware_ppo_clipping():
    covector = torch.ones(1)
    marginal_ratio = torch.ones(1, 2)
    for advantage, should_clip in ((1.0, True), (-1.0, False)):
        log_ratio = torch.tensor([1.5]).log().requires_grad_()
        joint_ratio = log_ratio.exp()
        common = torch.tensor([[advantage]])
        routed = common.unsqueeze(1).expand(-1, 2, -1)
        loss = MODULE.vector_policy_surrogate(
            joint_ratio,
            marginal_ratio,
            common,
            routed,
            covector,
            clip_coef=0.2,
            clip_coef_high=0.28,
        )
        gradient = torch.autograd.grad(loss, log_ratio)[0]
        assert (gradient.abs() < 1e-8) == should_clip


def test_action_router_is_hard_lag_compatible_and_zero_sum():
    torch.manual_seed(14)
    batch, actions, scores, features, dims = 128, 3, 2, 7, 5
    state_features = torch.randn(batch, features)
    action_score = torch.randn(batch, actions, scores)
    true_coefficients = torch.randn(actions * scores * features, dims)
    design = torch.einsum(
        "nac,nf->nacf", action_score, state_features
    ).flatten(1)
    target = design @ true_coefficients
    train_mask = torch.arange(batch).remainder(2).eq(0)
    validation_mask = ~train_mask
    fitted, r2, effective_rank, mode_gain = MODULE.fit_action_router(
        state_features,
        action_score,
        target,
        ridge=1e-8,
        train_mask=train_mask,
        validation_mask=validation_mask,
        inv_sqrt_cov=torch.eye(dims),
        sqrt_cov=torch.eye(dims),
    )
    assert r2 > 0.999
    assert effective_rank > 0
    assert mode_gain > 0.99
    common = torch.randn(batch, dims)
    routed, component, scale = MODULE.routed_vector_advantage(
        state_features,
        action_score,
        fitted,
        torch.eye(dims),
        common,
        max_fraction=0.5,
    )
    torch.testing.assert_close(
        (routed - common.unsqueeze(1)).sum(1),
        torch.zeros_like(common),
        atol=1e-5,
        rtol=1e-5,
    )
    assert component.shape == (batch, actions, dims)
    assert 0.0 < scale <= 1.0


def test_held_out_spectral_router_does_not_commute_with_scalar_projection():
    torch.manual_seed(15)
    batch, actions, scores, features, dims = 256, 1, 2, 2, 2
    state_features = torch.randn(batch, features)
    action_score = torch.randn(batch, actions, scores)
    design = torch.einsum(
        "nac,nf->nacf", action_score, state_features
    ).flatten(1)
    coefficients = torch.tensor(
        [[2.0, 0.0], [0.0, 1.5], [1.0, 0.0], [0.0, 1.0]]
    )
    target = design @ coefficients
    train_mask = torch.arange(batch).remainder(2).eq(0)
    validation_mask = ~train_mask
    target[validation_mask, 1] *= -1.0
    fitted, _, effective_rank, _ = MODULE.fit_action_router(
        state_features,
        action_score,
        target,
        ridge=1e-6,
        train_mask=train_mask,
        validation_mask=validation_mask,
        inv_sqrt_cov=torch.eye(dims),
        sqrt_cov=torch.eye(dims),
    )
    covector = torch.tensor([1.0, 1.0])
    train_design = design[train_mask].double()
    scalar_target = (target[train_mask] @ covector).double()
    gram = train_design.T @ train_design
    scalar_coefficients = torch.linalg.solve(
        gram + 1e-6 * gram.diagonal().mean() * torch.eye(gram.shape[0]),
        train_design.T @ scalar_target,
    ).float()
    assert effective_rank == 1
    assert not torch.allclose(
        fitted @ covector, scalar_coefficients, atol=1e-4, rtol=1e-4
    )
