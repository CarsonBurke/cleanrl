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
    "sf_vlam_v9_for_v22",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v9_leakyrelu05sq.py",
)
BASE13 = _load(
    "sf_vlam_v13_for_v22",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v13_rewardanchored.py",
)
MODULE = _load(
    "sf_vlam_v22_lejepa_detached_hlgauss",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v22_lejepa_detached_hlgauss.py",
)


class _FakeEnv:
    single_observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
    )
    single_action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(6,), dtype=np.float32
    )


def test_agent_predicts_exactly_one_learned_embedding_and_preserves_actor_rng():
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


def test_normalized_latent_bellman_target_has_geometric_occupancy_scale():
    torch.manual_seed(41)
    time, batch, dim = 11, 4, 7
    gamma, gae_lambda = 0.99, 0.95
    embedding = torch.randn(time, batch, dim)
    phi = (1.0 - gamma) * embedding
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

    constant_embedding = torch.ones(400, 1, 1)
    occupancy = torch.zeros_like(constant_embedding)
    running = torch.zeros(1, 1)
    for step in reversed(range(constant_embedding.shape[0])):
        running = (1.0 - gamma) * constant_embedding[step] + gamma * running
        occupancy[step] = running
    assert 0.98 < occupancy[0].item() < 1.0


def test_hlgauss_evaluator_cannot_send_gradients_into_latent():
    evaluator = MODULE.DetachedHLGaussEvaluator(8, 16, 31)
    latent = torch.randn(32, 8, requires_grad=True)
    target = torch.softmax(torch.randn(32, 31), dim=-1)
    logits = evaluator(latent)
    loss = MODULE.detached_hlgauss_ce(evaluator, latent, target)
    loss.backward()

    assert logits.shape == (32, 31)
    assert latent.grad is None
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in evaluator.parameters()
    )


def test_hlgauss_zero_logits_decode_to_zero_and_projection_is_normalized():
    args = MODULE.Args()
    support = MODULE.Dreamer3BucketHLGaussSupport(
        args.num_bins,
        args.v_min,
        args.v_max,
        args.value_sigma_to_bin_ratio,
        torch.device("cpu"),
    )
    evaluator = MODULE.DetachedHLGaussEvaluator(
        args.emb_dim, args.evaluator_hidden, args.num_bins
    )
    latent = torch.randn(23, args.emb_dim)
    logits = evaluator(latent)
    decoded = support.to_scalar(logits)
    labels = support.project(torch.tensor([-1000.0, 0.0, 1000.0]))

    torch.testing.assert_close(decoded, torch.zeros_like(decoded), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        labels.sum(-1), torch.ones(3), atol=1e-6, rtol=1e-6
    )


def test_procrustes_transport_pins_a_rotated_encoder_frame():
    torch.manual_seed(73)
    target = torch.randn(4096, 12)
    rotation, _ = torch.linalg.qr(torch.randn(12, 12))
    source = target @ rotation
    alignment = MODULE.orthogonal_alignment(source, target)

    torch.testing.assert_close(
        source @ alignment,
        target,
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        alignment.T @ alignment,
        torch.eye(12),
        rtol=2e-5,
        atol=2e-5,
    )


def test_genuine_lejepa_objective_has_no_reward_or_return_and_trains_all_parts():
    args = MODULE.Args()
    lejepa = MODULE.LeJepaSSL(17, 6, args)
    parameters = list(inspect.signature(lejepa.forward).parameters)
    assert parameters == [
        "obs_seq",
        "act_seq",
        "mask_seq",
        "sigreg_weight",
    ]

    obs = torch.randn(64, args.seq_len, 17)
    action = torch.randn(64, args.seq_len, 6)
    mask = torch.ones(64, args.seq_len - 1)
    loss, prediction, sigreg = lejepa(
        obs,
        action,
        mask,
        args.sigreg_weight,
    )
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


def test_defaults_keep_scalarization_terminal_and_single_object():
    args = MODULE.Args()
    assert args.critic_mtp_horizon == 1
    assert args.vector_adv is False
    assert args.per_dim_lambda is False
    assert args.num_bins == 511
    assert not hasattr(args, "evaluator_stat_ema")


def test_training_source_uses_target_embeddings_without_ema_or_popart():
    source = (
        ROOT
        / "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v22_lejepa_detached_hlgauss.py"
    ).read_text()
    assert "latent_target_flat[evaluator_indices]" in source
    assert "post_update_latent[evaluator_indices]" not in source
    assert "popart_update" not in source
    assert "ema_update(" not in source
