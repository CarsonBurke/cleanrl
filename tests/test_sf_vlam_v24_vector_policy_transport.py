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
    "sf_vlam_v9_for_v24",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v9_leakyrelu05sq.py",
)
BASE13 = _load(
    "sf_vlam_v13_for_v24",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v13_rewardanchored.py",
)
MODULE = _load(
    "sf_vlam_v24_vector_policy_transport",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v24_vector_policy_transport.py",
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


def test_transport_cannot_send_gradients_into_embedding_or_vector_trace():
    transport = MODULE.VectorPolicyTransport(8, 16, 6)
    embedding = torch.randn(32, 8, requires_grad=True)
    vector_trace = torch.randn(32, 8, requires_grad=True)
    covector = transport(embedding, vector_trace)
    covector.sum().backward()

    assert covector.shape == (32, 6)
    assert embedding.grad is None
    assert vector_trace.grad is None
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in transport.parameters()
    )


def test_beta_raw_score_matches_autograd():
    torch.manual_seed(91)
    raw = torch.randn(7, 6, dtype=torch.double, requires_grad=True)
    action = torch.rand(7, 3, dtype=torch.double).clamp(1e-4, 1 - 1e-4)
    dist = MODULE.beta_from_raw(raw, 3)
    expected = torch.autograd.grad(dist.log_prob(action).sum(), raw)[0]
    actual = MODULE.beta_raw_score(raw.detach(), action)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_beta_fisher_solve_inverts_each_raw_logit_block():
    torch.manual_seed(92)
    raw = torch.randn(11, 8, dtype=torch.double)
    covector = torch.randn_like(raw)
    natural = MODULE.beta_fisher_solve(raw, covector, 4)
    faa, fbb, fab = MODULE.beta_fisher_blocks(raw, 4)
    na, nb = natural.split(4, dim=-1)
    reconstructed = torch.cat([faa * na + fab * nb, fab * na + fbb * nb], dim=-1)
    torch.testing.assert_close(reconstructed, covector, rtol=1e-8, atol=1e-8)


def test_exact_beta_kl_scale_hits_rollout_target():
    torch.manual_seed(93)
    raw = torch.randn(512, 12)
    covector = 0.1 * torch.randn_like(raw)
    scale = MODULE.solve_transport_kl_scale(raw, covector, 6, 0.03, 32)
    natural = MODULE.beta_fisher_solve(raw, covector, 6)
    kl = MODULE.mean_beta_kl(raw, raw + scale * natural, 6)
    assert scale > 0
    torch.testing.assert_close(kl, torch.tensor(0.03), rtol=2e-4, atol=2e-5)


def test_exact_beta_kl_scale_remains_accurate_for_saturated_logits():
    torch.manual_seed(96)
    raw = 3.0 * torch.randn(128, 12)
    covector = 0.1 * torch.randn_like(raw)
    scale = MODULE.solve_transport_kl_scale(raw, covector, 6, 0.03, 48)
    natural = MODULE.beta_fisher_solve(raw, covector, 6)
    kl = MODULE.mean_beta_kl(raw, raw + scale * natural, 6)
    assert scale > 0
    torch.testing.assert_close(kl, torch.tensor(0.03), rtol=2e-4, atol=2e-5)


def test_actor_distribution_fit_has_finite_minimum_at_calibrated_target():
    torch.manual_seed(95)
    target = torch.randn(128, 12, dtype=torch.double)
    current = target.clone().requires_grad_(True)
    minimum = MODULE.beta_target_fit_loss(target, current, 6)
    gradient = torch.autograd.grad(minimum, current)[0]
    assert abs(float(minimum.detach())) < 1e-12
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)

    for offset in (-20.0, -0.5, 0.5, 20.0):
        displaced = MODULE.beta_target_fit_loss(target, target + offset, 6)
        assert torch.isfinite(displaced)
        assert displaced > minimum + 1e-6


def test_local_transport_loss_uses_only_order_and_detaches_inputs():
    torch.manual_seed(9)
    transport = MODULE.VectorPolicyTransport(4, 16, 6)
    embedding = torch.randn(5, 3, 4, requires_grad=True)
    vector_trace = torch.randn(5, 3, 4, requires_grad=True)
    raw_policy = torch.randn(5, 3, 6, requires_grad=True)
    native_action = torch.rand(5, 3, 3).clamp(1e-4, 1 - 1e-4)
    raw_score = MODULE.beta_raw_score(raw_policy, native_action)
    returns = torch.randn(5, 3)
    complete = torch.ones(5, 3, dtype=torch.bool)
    partner = torch.tensor([[1, 2, 0]]).expand(5, -1)

    loss, fit, valid, covector = MODULE.local_transport_loss(
        transport,
        embedding,
        vector_trace,
        raw_score,
        raw_policy,
        returns,
        complete,
        partner,
        target_fisher_rms=torch.tensor(1.7),
        zero_mean_coef=0.01,
    )
    loss.backward()

    assert valid.all()
    assert torch.isfinite(loss) and torch.isfinite(fit)
    assert covector.shape == (5, 3, 6)
    assert embedding.grad is None
    assert vector_trace.grad is None
    assert raw_policy.grad is None
    assert all(parameter.grad is not None for parameter in transport.parameters())

    # Scaling every return by a positive constant cannot change an ordinal target.
    scaled_loss, _, _, _ = MODULE.local_transport_loss(
        transport,
        embedding,
        vector_trace,
        raw_score,
        raw_policy,
        returns * 1000.0,
        complete,
        partner,
        target_fisher_rms=torch.tensor(1.7),
        zero_mean_coef=0.01,
    )
    torch.testing.assert_close(loss.detach(), scaled_loss.detach())


def test_centered_transport_pair_loss_is_invariant_to_pair_reindexing():
    torch.manual_seed(19)
    transport = MODULE.VectorPolicyTransport(4, 16, 6)
    with torch.no_grad():
        transport.head.weight.normal_(std=0.03)
    embedding = torch.randn(4, 3, 4)
    trace = torch.randn(4, 3, 4)
    raw_policy = torch.randn(4, 3, 6)
    score = MODULE.beta_raw_score(
        raw_policy, torch.rand(4, 3, 3).clamp(1e-4, 1 - 1e-4)
    )
    returns = torch.randn(4, 3)
    complete = torch.ones(4, 3, dtype=torch.bool)
    partner = torch.tensor([[1, 2, 0]]).expand(4, -1)
    loss, _, _, _ = MODULE.local_transport_loss(
        transport, embedding, trace, score, raw_policy, returns, complete, partner,
        torch.tensor(1.3), 0.01,
    )

    permutation = torch.tensor([1, 0, 2])
    inverse = torch.argsort(permutation)
    permuted_partner = inverse[partner[:, permutation]]
    permuted_loss, _, _, _ = MODULE.local_transport_loss(
        transport,
        embedding[:, permutation],
        trace[:, permutation],
        score[:, permutation],
        raw_policy[:, permutation],
        returns[:, permutation],
        complete[:, permutation],
        permuted_partner,
        torch.tensor(1.3),
        0.01,
    )
    torch.testing.assert_close(loss, permuted_loss)


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


def test_agent_exposes_beta_raw_policy_channels():
    torch.manual_seed(94)
    agent = MODULE.Agent(_FakeEnv(), MODULE.Args())
    obs = torch.randn(13, 17)
    action, native, log_prob, entropy, value, raw_policy = agent.get_action_and_value(obs)
    assert action.shape == native.shape == (13, 6)
    assert log_prob.shape == entropy.shape == (13,)
    assert value.shape == (13, 1, MODULE.Args().emb_dim)
    assert raw_policy.shape == (13, 12)
    assert ((native > 0) & (native < 1)).all()


def test_defaults_use_one_vector_object_and_exact_kl_transport():
    args = MODULE.Args()
    assert args.critic_mtp_horizon == 1
    assert args.per_dim_lambda is False
    assert args.transport_target_kl == 0.03
    assert args.transport_bisection_steps == 48
    assert args.transport_hidden == 128
    assert args.actor_dist == "beta"
    assert not hasattr(args, "num_bins")
    assert not hasattr(args, "evaluator_lr")


def test_training_source_has_no_scalar_advantage_or_rank_transform_and_cross_fits_transport():
    source = (
        ROOT
        / "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v24_vector_policy_transport.py"
    ).read_text()
    assert "Dreamer3BucketHLGaussSupport" not in source
    assert "decode_value(" not in source
    assert "rankgauss" not in source.lower()
    assert "mb_advantages" not in source
    assert "logratio" not in source
    assert "frozen_transport_covector = transport(" in source
    assert source.index("frozen_transport_covector = transport(") < source.index(
        "transport_optimizer.step()"
    )
    assert "return_difference.sign().unsqueeze(-1)" in source
    assert "critic_params if transport_ready else list(agent.critic_head.parameters())" in source
    assert "beta_target_fit_loss(" in source
    assert "b_target_raw[mb_inds]" in source
    assert "if not args.separate_grad_clip" in source
    assert "popart_update" not in source
    assert "ema_update(" not in source
