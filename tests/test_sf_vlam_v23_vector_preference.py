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
    "sf_vlam_v9_for_v23",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v9_leakyrelu05sq.py",
)
BASE13 = _load(
    "sf_vlam_v13_for_v23",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v13_rewardanchored.py",
)
MODULE = _load(
    "sf_vlam_v23_vector_preference",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v23_vector_preference.py",
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


def test_preference_field_cannot_send_gradients_into_embedding_or_vector_trace():
    preference = MODULE.VectorPreferenceField(8, 16, 8)
    embedding = torch.randn(32, 8, requires_grad=True)
    vector_trace = torch.randn(32, 8, requires_grad=True)
    score, direction = preference(embedding, vector_trace)
    score.sum().backward()

    assert score.shape == (32,)
    torch.testing.assert_close(direction.norm(dim=-1), torch.ones(32))
    assert embedding.grad is None
    assert vector_trace.grad is None
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in preference.parameters()
    )


def test_local_preference_loss_uses_only_order_and_detaches_representation():
    torch.manual_seed(9)
    preference = MODULE.VectorPreferenceField(4, 16, 4)
    embedding = torch.randn(5, 3, 4, requires_grad=True)
    vector_trace = torch.randn(5, 3, 4, requires_grad=True)
    returns = torch.randn(5, 3)
    complete = torch.ones(5, 3, dtype=torch.bool)
    partner = torch.tensor([[1, 2, 0]]).expand(5, -1)

    loss, ordinal, valid, direction = MODULE.local_covector_preference_loss(
        preference,
        embedding,
        vector_trace,
        returns,
        complete,
        partner,
        temperature=0.2,
        smoothness_weight=0.01,
    )
    loss.backward()

    assert valid.all()
    assert torch.isfinite(loss) and torch.isfinite(ordinal)
    torch.testing.assert_close(direction.norm(dim=-1), torch.ones(5, 3))
    assert embedding.grad is None
    assert vector_trace.grad is None
    assert all(parameter.grad is not None for parameter in preference.parameters())

    # Scaling every return by a positive constant cannot change an ordinal loss.
    scaled_loss, _, _, _ = MODULE.local_covector_preference_loss(
        preference,
        embedding,
        vector_trace,
        returns * 1000.0,
        complete,
        partner,
        temperature=0.2,
        smoothness_weight=0.01,
    )
    torch.testing.assert_close(loss.detach(), scaled_loss.detach())


def test_neighbor_pairs_are_cross_env_complete_and_horizon_matched():
    embedding = torch.tensor(
        [[[0.0], [0.1], [3.0]], [[0.0], [0.2], [0.1]]]
    )
    horizon = torch.tensor([[100.0, 105.0, 200.0], [50.0, 100.0, 52.0]])
    complete = torch.tensor([[True, True, True], [True, False, True]])
    partner, has_partner, distance = MODULE.nearest_cross_env_pairs(
        embedding, horizon, complete, horizon_tolerance=10
    )

    assert partner[0, 0].item() == 1
    assert partner[0, 1].item() == 0
    assert not has_partner[0, 2]
    assert partner[1, 0].item() == 2
    assert partner[1, 2].item() == 0
    assert not has_partner[1, 1]
    assert torch.isinf(distance.diagonal(dim1=1, dim2=2)).all()


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


def test_defaults_use_one_vector_object_and_local_preference_field():
    args = MODULE.Args()
    assert args.critic_mtp_horizon == 1
    assert args.per_dim_lambda is False
    assert args.preference_temperature == 0.2
    assert args.preference_smoothness == 0.01
    assert not hasattr(args, "num_bins")
    assert not hasattr(args, "evaluator_lr")


def test_training_source_has_no_scalar_decoder_and_cross_fits_preference():
    source = (
        ROOT
        / "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v23_vector_preference.py"
    ).read_text()
    assert "Dreamer3BucketHLGaussSupport" not in source
    assert "decode_value(" not in source
    assert "frozen_preference_direction * vector_policy_trace" in source
    assert source.index("frozen_preference_direction * vector_policy_trace") < source.index(
        "preference_optimizer.step()"
    )
    assert "return_difference.sign() * margin" in source
    assert "critic_params if preference_ready else list(agent.critic_head.parameters())" in source
    assert "if not args.separate_grad_clip" in source
    assert "popart_update" not in source
    assert "ema_update(" not in source
