import importlib.util
import inspect
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


ROOT = Path(__file__).parents[1]


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load(
    "sf_vlam_v9_for_v25",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v9_leakyrelu05sq.py",
)
BASE13 = _load(
    "sf_vlam_v13_for_v25",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v13_rewardanchored.py",
)
MODULE = _load(
    "sf_vlam_v25_latent_successor_improvement",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v25_latent_successor_improvement.py",
)


class _FakeEnv:
    single_observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
    )
    single_action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(6,), dtype=np.float32
    )


def test_agent_predicts_one_rich_value_and_preserves_v9_actor_rng():
    torch.manual_seed(918)
    base = BASE.Agent(_FakeEnv(), BASE.Args())
    expected_rng = torch.get_rng_state()

    torch.manual_seed(918)
    agent = MODULE.Agent(_FakeEnv(), MODULE.Args())
    actual_rng = torch.get_rng_state()

    assert agent.sf_dim == MODULE.Args().emb_dim
    assert agent.critic_mtp_horizon == 1
    assert agent.critic_head.out_features == MODULE.Args().emb_dim
    torch.testing.assert_close(actual_rng, expected_rng)
    for actual, expected in zip(
        agent.trunk.parameters(), base.trunk.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)
    for name in ("actor_alpha_head", "actor_beta_head"):
        actual = getattr(agent, name)
        expected = getattr(base, name)
        torch.testing.assert_close(actual.weight, expected.weight)
        torch.testing.assert_close(actual.bias, expected.bias)


def test_normalized_latent_bellman_target_matches_vector_td_lambda():
    torch.manual_seed(41)
    time, batch, dim = 11, 4, 7
    gamma, gae_lambda = 0.99, 0.95
    phi = (1.0 - gamma) * torch.randn(time, batch, dim)
    psi_cur = torch.randn(time, batch, dim)
    psi_next = torch.randn(time, batch, dim)
    terminations = torch.zeros(time, batch)
    boundaries = torch.zeros(time, batch)
    valids = torch.ones(time, batch)
    terminations[5, 2] = 1.0
    boundaries[5, 2] = 1.0

    residual = MODULE.successor_lambda_residual(
        phi,
        psi_cur,
        psi_next,
        terminations,
        boundaries,
        valids,
        gamma,
        torch.full((dim,), gae_lambda),
    )

    expected = torch.zeros_like(phi)
    trace = torch.zeros_like(phi[0])
    for step in reversed(range(time)):
        bootstrap = (
            (1.0 - terminations[step]) * valids[step]
        ).unsqueeze(-1)
        continuation = (1.0 - boundaries[step]).unsqueeze(-1)
        delta = phi[step] + gamma * bootstrap * psi_next[step] - psi_cur[step]
        trace = delta + gamma * gae_lambda * continuation * trace
        expected[step] = trace
    torch.testing.assert_close(residual, expected)


def test_bridge_is_action_centered_and_control_affine():
    torch.manual_seed(4)
    bridge = MODULE.LatentSuccessorBridge(8, 3, 16)
    embedding = torch.randn(7, 8)
    behavior_mean = torch.randn(7, 3)
    action_delta = torch.randn(7, 3)
    baseline, control = bridge.components(embedding)

    centered = bridge(embedding, behavior_mean, behavior_mean)
    displaced = bridge(embedding, behavior_mean + action_delta, behavior_mean)

    torch.testing.assert_close(centered, baseline)
    torch.testing.assert_close(
        displaced - centered,
        torch.einsum("bda,ba->bd", control, action_delta),
    )


def test_bridge_fit_detaches_all_data_but_trains_bridge_parameters():
    torch.manual_seed(5)
    bridge = MODULE.LatentSuccessorBridge(8, 3, 16)
    embedding = torch.randn(20, 8, requires_grad=True)
    action = torch.randn(20, 3, requires_grad=True)
    behavior_mean = torch.randn(20, 3, requires_grad=True)
    next_value = torch.randn(20, 8, requires_grad=True)
    valid = torch.ones(20, dtype=torch.bool)

    loss, prediction = MODULE.bridge_loss(
        bridge, embedding, action, behavior_mean, next_value, valid
    )
    loss.backward()

    assert prediction.shape == next_value.shape
    assert embedding.grad is None
    assert action.grad is None
    assert behavior_mean.grad is None
    assert next_value.grad is None
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in bridge.parameters()
    )


def test_actor_gradient_flows_through_action_delta_but_not_bridge_inputs():
    torch.manual_seed(6)
    bridge = MODULE.LatentSuccessorBridge(8, 3, 16)
    for parameter in bridge.parameters():
        parameter.requires_grad_(False)
    embedding = torch.randn(12, 8, requires_grad=True)
    action_delta = torch.randn(12, 3, requires_grad=True)
    target = torch.randn(12, 8)

    loss = MODULE.latent_successor_error(
        bridge.control_delta(embedding, action_delta), target
    ).mean()
    loss.backward()

    assert embedding.grad is None
    assert action_delta.grad is not None
    assert torch.isfinite(action_delta.grad).all()
    assert action_delta.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in bridge.parameters())


def test_zero_control_matrix_produces_zero_actor_gradient():
    torch.manual_seed(7)
    bridge = MODULE.LatentSuccessorBridge(8, 3, 16)
    with torch.no_grad():
        bridge.control_head.weight.zero_()
        bridge.control_head.bias.zero_()
    for parameter in bridge.parameters():
        parameter.requires_grad_(False)
    action_delta = torch.randn(10, 3, requires_grad=True)
    prediction = bridge.control_delta(torch.randn(10, 8), action_delta)
    loss = MODULE.latent_successor_error(prediction, torch.randn(10, 8)).mean()
    gradient = torch.autograd.grad(loss, action_delta)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))


def test_fixed_bridge_actor_step_reduces_full_vector_error():
    torch.manual_seed(8)
    bridge = MODULE.LatentSuccessorBridge(6, 3, 12)
    control = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
         [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]
    )
    with torch.no_grad():
        bridge.control_head.weight.zero_()
        bridge.control_head.bias.copy_(control.flatten())
    for parameter in bridge.parameters():
        parameter.requires_grad_(False)
    embedding = torch.randn(32, 6)
    target_action_delta = 0.4 * torch.randn(32, 3)
    target = target_action_delta @ control.T
    action_delta = torch.zeros_like(target_action_delta, requires_grad=True)
    optimizer = torch.optim.SGD([action_delta], lr=4.0)

    initial = MODULE.latent_successor_error(
        bridge.control_delta(embedding, action_delta), target
    ).mean()
    for _ in range(100):
        loss = MODULE.latent_successor_error(
            bridge.control_delta(embedding, action_delta), target
        ).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final = MODULE.latent_successor_error(
        bridge.control_delta(embedding, action_delta), target
    ).mean()
    assert final < 0.2 * initial


def test_bridge_learns_action_residual_beyond_state_only_baseline():
    torch.manual_seed(9)
    bridge = MODULE.LatentSuccessorBridge(5, 2, 12)
    optimizer = torch.optim.Adam(bridge.parameters(), lr=0.03)
    embedding = torch.zeros(128, 5)
    behavior_mean = torch.zeros(128, 2)
    actions = torch.randn(128, 2)
    true_control = torch.randn(5, 2)
    next_value = 0.2 + actions @ true_control.T
    valid = torch.ones(128, dtype=torch.bool)

    for _ in range(200):
        loss, _ = MODULE.bridge_loss(
            bridge, embedding, actions, behavior_mean, next_value, valid
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        prediction = bridge(embedding, actions, behavior_mean)
        full, baseline, gain, cosine, effect_fraction = MODULE.bridge_metrics(
            bridge, prediction, embedding, next_value, valid
        )
    assert full < 0.05 * baseline
    assert gain > 0
    assert cosine > 0.95
    assert 0.8 < effect_fraction < 1.2


def test_better_successor_targets_only_losers_and_uses_no_return_magnitude():
    next_values = torch.tensor(
        [[[0.0, 1.0], [3.0, 5.0], [7.0, 11.0]]]
    )
    returns = torch.tensor([[2.0, 9.0, -1.0]])
    valid = torch.tensor([[True, True, False]])
    partner = torch.tensor([[1, 0, 1]])

    displacement, loser = MODULE.choose_better_successor_displacements(
        next_values, returns, valid, partner
    )
    torch.testing.assert_close(displacement[0, 0], torch.tensor([3.0, 4.0]))
    assert loser.tolist() == [[True, False, False]]

    scaled_displacement, scaled_loser = MODULE.choose_better_successor_displacements(
        next_values, 1000.0 * returns + 17.0, valid, partner
    )
    torch.testing.assert_close(scaled_displacement, displacement)
    torch.testing.assert_close(scaled_loser, loser)


def test_neighbor_pairs_are_cross_env_complete_and_exact_horizon_matched():
    embedding = torch.tensor(
        [[[0.0], [0.1], [3.0]], [[0.0], [0.2], [0.1]]]
    )
    horizon = torch.tensor([[100.0, 100.0, 200.0], [50.0, 100.0, 50.0]])
    complete = torch.tensor([[True, True, True], [True, False, True]])
    partner, has_partner, distance = MODULE.nearest_cross_env_pairs(
        embedding, horizon, complete, horizon_tolerance=0
    )

    assert partner[0, 0].item() == 1
    assert partner[0, 1].item() == 0
    assert not has_partner[0, 2]
    assert partner[1, 0].item() == 2
    assert partner[1, 2].item() == 0
    assert not has_partner[1, 1]
    assert torch.isinf(distance.diagonal(dim1=1, dim2=2)).all()


def test_beta_mean_action_is_bounded_and_exact_kl_is_zero_at_identity():
    torch.manual_seed(10)
    raw = 5.0 * torch.randn(128, 12)
    low = -torch.ones(6)
    high = torch.ones(6)
    mean = MODULE.beta_mean_action(raw, low, high)
    assert ((mean > low) & (mean < high)).all()
    torch.testing.assert_close(MODULE.mean_beta_kl(raw, raw, 6), torch.tensor(0.0))


def test_procrustes_alignment_pins_a_rotated_encoder_frame():
    torch.manual_seed(73)
    target = torch.randn(4096, 12)
    rotation, _ = torch.linalg.qr(torch.randn(12, 12))
    source = target @ rotation
    alignment = MODULE.orthogonal_alignment(source, target)

    torch.testing.assert_close(source @ alignment, target, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        alignment.T @ alignment, torch.eye(12), rtol=2e-5, atol=2e-5
    )


def test_genuine_lejepa_objective_has_no_reward_or_return_and_trains_all_parts():
    args = MODULE.Args(sigreg_num_proj=32, sigreg_proj_chunk=16)
    lejepa = MODULE.LeJepaSSL(17, 6, args)
    assert list(inspect.signature(lejepa.forward).parameters) == [
        "obs_seq",
        "act_seq",
        "mask_seq",
        "sigreg_weight",
    ]

    obs = torch.randn(32, args.seq_len, 17)
    action = torch.randn(32, args.seq_len, 6)
    mask = torch.ones(32, args.seq_len - 1)
    loss, prediction, sigreg = lejepa(obs, action, mask, args.sigreg_weight)
    loss.backward()

    for metric in (loss, prediction, sigreg):
        assert torch.isfinite(metric)
    for component in (
        lejepa.encoder,
        lejepa.action_encoder,
        lejepa.predictor,
        lejepa.pred_proj,
    ):
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in component.parameters()
        )


def test_lejepa_initialization_and_rng_match_v13():
    torch.manual_seed(2203)
    expected = BASE13.LeJepaSSL(17, 6, BASE13.Args())
    expected_rng = torch.get_rng_state()

    torch.manual_seed(2203)
    actual = MODULE.LeJepaSSL(17, 6, MODULE.Args())
    actual_rng = torch.get_rng_state()

    torch.testing.assert_close(actual_rng, expected_rng)
    for actual_parameter, expected_parameter in zip(
        actual.parameters(), expected.parameters(), strict=True
    ):
        torch.testing.assert_close(actual_parameter, expected_parameter)


def test_defaults_select_single_vector_value_beta_and_latent_trust_region():
    args = MODULE.Args()
    assert args.critic_mtp_horizon == 1
    assert args.emb_dim == 32
    assert args.per_dim_lambda is False
    assert args.successor_horizon_tolerance == 0
    assert args.latent_actor_target_kl == 0.03
    assert args.actor_dist == "beta"
    assert not hasattr(args, "num_bins")
    assert not hasattr(args, "transport_target_kl")


def test_training_cross_fits_bridge_and_has_no_scalar_policy_objective():
    source = (
        ROOT
        / "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v25_latent_successor_improvement.py"
    ).read_text()
    lower = source.lower()
    assert "Dreamer3BucketHLGaussSupport" not in source
    assert "decode_value(" not in source
    assert "mb_advantages" not in source
    assert "logratio" not in source
    assert "beta_raw_score" not in source
    assert "b_target_raw" not in source
    assert "return_difference" not in source
    assert "latent_successor_error(" in source
    assert "b_desired_displacement[mb_inds]" in source
    assert "critic_params if bridge_ready else list(agent.critic_head.parameters())" in source
    assert source.index("parameter.requires_grad_(False)") < source.index(
        "for epoch in range(args.update_epochs):"
    )
    assert source.index("target_alignment.copy_(next_alignment)") < source.index(
        "bridge_optimizer.step()"
    )
    assert source.index("for epoch in range(args.update_epochs):") < source.index(
        "bridge_optimizer.step()"
    )
    assert "rankgauss(" not in lower
    assert "popart_update" not in source
    assert "ema_update(" not in source
