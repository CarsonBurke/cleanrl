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
    "sf_vlam_v9_for_v21",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v9_leakyrelu05sq.py",
)
MODULE = _load(
    "sf_vlam_v21_latent_successor_autoencoder",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v21_latent_successor_autoencoder.py",
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


def test_scalar_evaluator_cannot_send_gradients_into_latent():
    evaluator = MODULE.DetachedValueEvaluator(8, 16)
    latent = torch.randn(32, 8, requires_grad=True)
    target = torch.randn(32)
    loss = (evaluator(latent) - target).square().mean()
    loss.backward()

    assert latent.grad is None
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in evaluator.parameters()
    )


def test_popart_change_preserves_decoded_values_and_rescales_adam_state():
    evaluator = MODULE.DetachedValueEvaluator(5, 9)
    optimizer = torch.optim.Adam(evaluator.parameters(), lr=3e-4)
    latent = torch.randn(23, 5)
    target = torch.randn(23)
    (evaluator(latent) - target).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    old_mean = torch.tensor(3.0)
    old_std = torch.tensor(7.0)
    new_mean = torch.tensor(-2.0)
    new_std = torch.tensor(11.0)
    with torch.no_grad():
        before = old_mean + old_std * evaluator(latent)
        old_first_moment = optimizer.state[evaluator.head.weight][
            "exp_avg"
        ].clone()
    MODULE.popart_update(
        evaluator,
        optimizer,
        old_mean,
        old_std,
        new_mean,
        new_std,
    )
    with torch.no_grad():
        after = new_mean + new_std * evaluator(latent)

    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        optimizer.state[evaluator.head.weight]["exp_avg"],
        old_first_moment * (old_std / new_std),
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


def test_autoencoder_objective_has_no_reward_or_action_argument_and_trains_all_parts():
    args = MODULE.Args()
    autoencoder = MODULE.LatentStateAutoencoder(17, args)
    parameters = list(
        inspect.signature(autoencoder.forward).parameters
    )
    assert parameters == [
        "obs_seq",
        "mask_seq",
        "reconstruction_weight",
        "temporal_weight",
        "sigreg_weight",
    ]

    obs = torch.randn(64, args.seq_len, 17)
    mask = torch.ones(64, args.seq_len - 1)
    loss, reconstruction, temporal, sigreg = autoencoder(
        obs,
        mask,
        args.reconstruction_weight,
        args.temporal_weight,
        args.sigreg_weight,
    )
    loss.backward()

    for metric in (loss, reconstruction, temporal, sigreg):
        assert torch.isfinite(metric)
    for component in (
        autoencoder.encoder,
        autoencoder.predictor,
        autoencoder.decoder,
    ):
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in component.parameters()
        )


def test_defaults_keep_scalarization_terminal_and_single_object():
    args = MODULE.Args()
    assert args.critic_mtp_horizon == 1
    assert args.vector_adv is False
    assert args.per_dim_lambda is False
    assert args.reconstruction_weight > 0
    assert args.temporal_weight > 0
