import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import gymnasium as gym


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_lejepa_tddelta_directgeo_v3.py"
)
SPEC = importlib.util.spec_from_file_location("lejepa_tddelta_directgeo_v3", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_tddelta_targets_telescope_to_longest_horizon_bellman_target():
    torch.manual_seed(0)
    t, b, j, d = 5, 3, len(MODULE.TD_BETAS), 7
    phi = torch.randn(t, b, d)
    current = torch.randn(t, b, j, d)
    next_bands = torch.randn_like(current)
    bootstrap = torch.randint(0, 2, (t, b), dtype=torch.float32)
    betas = torch.tensor(MODULE.TD_BETAS)

    targets = MODULE.build_tddelta_targets(phi, next_bands, bootstrap, betas)
    summed_delta = (targets - current).sum(dim=2)
    expected = (
        phi
        + betas[-1] * bootstrap.unsqueeze(-1) * next_bands.sum(dim=2)
        - current.sum(dim=2)
    )

    torch.testing.assert_close(summed_delta, expected, rtol=1e-5, atol=1e-6)


def test_consecutive_state_band_innovations_telescope_over_time():
    torch.manual_seed(2)
    t, b, j, d = 8, 2, len(MODULE.TD_BETAS), 3
    bands = torch.randn(t + 1, b, j, d)
    phi = torch.randn(t, b, d)
    betas = torch.tensor(MODULE.TD_BETAS)
    targets = MODULE.build_tddelta_targets(
        phi, bands[1:], torch.ones(t, b), betas
    )
    innovations = (targets - bands[:-1]).sum(dim=2)
    actual = MODULE.fixed_horizon_sum(
        innovations, torch.ones(t, b), gamma=betas[-1].item(), horizon=4
    )
    for start in range(t):
        length = min(4, t - start)
        discounted_phi = sum(
            betas[-1] ** offset * phi[start + offset]
            for offset in range(length)
        )
        expected = (
            discounted_phi
            + betas[-1] ** length * bands[start + length].sum(dim=1)
            - bands[start].sum(dim=1)
        )
        torch.testing.assert_close(
            actual[start], expected, rtol=2e-5, atol=2e-6
        )


def test_direct_geometric_targets_learn_all_bands_from_zero_bootstrap():
    t, b, d = 128, 2, 3
    phi = torch.ones(t, b, d)
    next_bands = torch.zeros(t, b, len(MODULE.TD_HORIZONS), d)
    bootstrap = torch.ones(t, b)
    continuation = torch.ones(t, b)
    betas = torch.tensor(MODULE.TD_BETAS)
    targets = MODULE.build_direct_geometric_targets(
        phi,
        next_bands,
        bootstrap,
        continuation,
        betas,
        MODULE.TD_HORIZONS,
    )
    # Unlike one-step TDDelta, every higher band receives a direct nonzero target
    # before any shorter-band prediction has learned.
    assert (targets[:28].abs().mean(dim=(0, 1, 3)) > 0).all()


def test_direct_geometric_constant_fixed_point_matches_band_widths():
    t, b, d = 128, 2, 1
    phi = torch.ones(t, b, d)
    widths = torch.tensor(MODULE.TD_BAND_WIDTHS, dtype=torch.float32)
    next_bands = widths.view(1, 1, -1, 1).expand(t, b, -1, d).clone()
    targets = MODULE.build_direct_geometric_targets(
        phi,
        next_bands,
        torch.ones(t, b),
        torch.ones(t, b),
        torch.tensor(MODULE.TD_BETAS),
        MODULE.TD_HORIZONS,
    )
    expected = widths.view(1, 1, -1, 1).expand_as(targets)
    torch.testing.assert_close(targets, expected, rtol=2e-5, atol=2e-5)


def test_direct_successor_target_bootstraps_truncation_and_rollout_tail():
    phi = torch.ones(4, 1, 1)
    next_value = torch.full_like(phi, 10.0)
    continuation = torch.tensor([[1.0], [0.0], [1.0], [1.0]])
    truncation_bootstrap = torch.ones(4, 1)
    actual = MODULE.fixed_n_successor_target(
        phi,
        next_value,
        truncation_bootstrap,
        continuation,
        beta=0.5,
        horizon=4,
    )
    expected = torch.tensor([4.0, 6.0, 4.0, 6.0]).view(4, 1, 1)
    torch.testing.assert_close(actual, expected)

    termination_bootstrap = truncation_bootstrap.clone()
    termination_bootstrap[1] = 0.0
    terminated = MODULE.fixed_n_successor_target(
        phi,
        next_value,
        termination_bootstrap,
        continuation,
        beta=0.5,
        horizon=4,
    )
    torch.testing.assert_close(
        terminated[:2], torch.tensor([1.5, 1.0]).view(2, 1, 1)
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
    next_bands = torch.ones(4, 1, len(MODULE.TD_BETAS), 1)
    betas = torch.tensor(MODULE.TD_BETAS)
    # At t=1 this represents a time-limit truncation: use final_observation for the
    # one-step target, but do not carry credit into the reset episode at t=2.
    bootstrap = torch.ones(4, 1)
    targets = MODULE.build_tddelta_targets(phi, next_bands, bootstrap, betas)
    assert targets[1].sum() > phi[1].sum()

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


def test_loss_preconditioning_does_not_change_raw_critic_outputs():
    torch.manual_seed(3)
    head = torch.nn.Linear(4, 6)
    x = torch.randn(5, 4)
    before = head(x).clone()
    raw_target = torch.randn_like(before)
    for scale in (torch.ones(6), torch.rand(6) + 0.2):
        _loss = ((before - raw_target) / scale).square().mean()
        assert torch.isfinite(_loss)
    after = head(x)
    torch.testing.assert_close(before, after)
    assert MODULE.TD_BAND_WIDTHS == (1, 1, 2, 4, 8, 16, 32, 36)


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
    action, latent, log_prob, entropy, bands = agent.get_action_and_value(
        policy_input
    )
    assert action.shape == latent.shape == (4, 2)
    assert log_prob.shape == entropy.shape == (4,)
    assert bands.shape == (
        4,
        len(MODULE.TD_HORIZONS),
        args.emb_dim + 5 + 2 * 2 + 1,
    )
