import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import gymnasium as gym


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "ppo_continuous_action_lejepa_octavecorr_v7.py"
)
SPEC = importlib.util.spec_from_file_location("lejepa_octavecorr_v7", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_actor_credit_horizon_is_16_while_successor_reaches_100():
    args = MODULE.Args()
    assert args.advantage_horizon == 16
    assert args.octave_horizons == (16, 32, 64)
    assert args.octave_outer_grad_ratio == 0.25
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
        lejepa_cumulative_targets=True,
        gamma=0.99,
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


def test_cumulative_lejepa_target_uses_complete_discounted_future_window():
    embedding = torch.tensor([[[0.0], [1.0], [3.0], [9.0]]])
    target = MODULE.cumulative_embedding_targets(embedding, horizon=3, gamma=0.5)
    expected = torch.tensor([[[1.0 + 0.5 * 3.0 + 0.25 * 9.0]]]) / 1.75
    torch.testing.assert_close(target, expected)


def test_lejepa_window_validity_recovers_after_a_reset():
    continuation = torch.ones(1, 8)
    continuation[0, 2] = 0.0
    valid = MODULE.transition_window_validity(continuation, horizon=2)
    torch.testing.assert_close(
        valid,
        torch.tensor([[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]]),
    )


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
    action, latent, log_prob, entropy, successors = agent.get_action_and_value(
        policy_input
    )
    assert action.shape == latent.shape == (4, 2)
    assert log_prob.shape == entropy.shape == (4,)
    assert successors.shape == (
        4,
        len(MODULE.TD_HORIZONS),
        args.emb_dim + 5 + 2 * 2 + 1,
    )


def test_octave_credit_telescopes_without_subtracting_anchor_residual():
    gamma = 0.9
    delta = torch.ones(70, 1, 1)
    residual = torch.full((70, 1), 0.5)
    continuation = torch.ones(70, 1)
    credit = MODULE.build_octave_credit(
        delta, residual, torch.ones(1), continuation, gamma
    )
    p16 = sum(gamma**k for k in range(16))
    p32 = sum(gamma**k for k in range(32))
    p64 = sum(gamma**k for k in range(64))
    torch.testing.assert_close(credit[0, 0, 0], torch.tensor(1.5 * p16))
    torch.testing.assert_close(credit[0, 0, 1], torch.tensor(p32 - p16))
    torch.testing.assert_close(credit[0, 0, 2], torch.tensor(p64 - p32))
    torch.testing.assert_close(
        credit[0, 0].sum(),
        torch.tensor(p64 + 0.5 * p16),
    )


def test_octave_outer_bands_are_zero_when_boundary_precedes_anchor():
    delta = torch.randn(80, 2, 3)
    residual = torch.randn(80, 2)
    continuation = torch.ones(80, 2)
    continuation[7] = 0.0
    credit = MODULE.build_octave_credit(
        delta, residual, torch.randn(3), continuation, 0.99
    )
    torch.testing.assert_close(credit[0, :, 1:], torch.zeros_like(credit[0, :, 1:]))


def test_octave_normalization_preserves_zero_support_and_relative_scale():
    credit = torch.tensor(
        [
            [[-2.0, 0.0, 0.1], [0.0, 0.0, 0.2]],
            [[2.0, 0.0, 0.3], [4.0, 0.0, 0.4]],
        ]
    )
    normalized = MODULE.normalize_octave_credit(credit)
    torch.testing.assert_close(
        normalized[..., 1], torch.zeros_like(normalized[..., 1])
    )
    torch.testing.assert_close(normalized[..., 0].mean(), torch.tensor(0.0))
    torch.testing.assert_close(normalized[..., 0].std(), torch.tensor(1.0))
    assert normalized[..., 2].std() < 1.0


def test_cross_env_reliability_accepts_agreement_and_rejects_cancellation():
    score = torch.ones(8, 4, 2)
    agreeing_advantage = torch.ones(8, 4)
    rho_agree = MODULE.cross_env_signal_fraction(score, agreeing_advantage)
    torch.testing.assert_close(rho_agree, torch.tensor(1.0))

    cancelling_advantage = torch.tensor([1.0, -1.0, 1.0, -1.0]).expand(8, -1)
    rho_cancel = MODULE.cross_env_signal_fraction(score, cancelling_advantage)
    torch.testing.assert_close(rho_cancel, torch.tensor(0.0))


def test_head_gradient_reliability_uses_features_and_split_agreement():
    score = torch.ones(8, 4, 2)
    feature = torch.ones(8, 4, 3)
    agreeing = torch.ones(8, 4)
    torch.testing.assert_close(
        MODULE.cross_env_head_gradient_signal_fraction(score, feature, agreeing),
        torch.tensor(1.0),
    )
    split_opposed = torch.tensor([1.0, -1.0, 1.0, -1.0]).expand(8, -1)
    torch.testing.assert_close(
        MODULE.cross_env_head_gradient_signal_fraction(
            score, feature, split_opposed
        ),
        torch.tensor(0.0),
    )


def test_parameter_grad_norm_uses_parameter_device_without_gradients():
    parameter = torch.nn.Parameter(torch.ones(2))
    norm = MODULE.parameter_grad_norm([parameter])
    assert norm.device == parameter.device
    torch.testing.assert_close(norm, torch.tensor(0.0, device=parameter.device))


def test_outer_gradient_scale_enforces_anchor_budget_and_zero_anchor():
    scale = MODULE.bounded_outer_gradient_scale(
        torch.tensor(2.0), torch.tensor(10.0), ratio=0.25
    )
    torch.testing.assert_close(scale * 10.0, torch.tensor(0.5))
    zero_scale = MODULE.bounded_outer_gradient_scale(
        torch.tensor(0.0), torch.tensor(10.0), ratio=0.25
    )
    torch.testing.assert_close(zero_scale, torch.tensor(0.0))
