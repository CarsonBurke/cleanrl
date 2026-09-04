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
    "sf_vlam_v9_for_v26",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v9_leakyrelu05sq.py",
)
BASE13 = _load(
    "sf_vlam_v13_for_v26",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v13_rewardanchored.py",
)
MODULE = _load(
    "sf_vlam_v26_orthogonal_outcome_rlearner",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v26_orthogonal_outcome_rlearner.py",
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


def test_vector_td_lambda_target_matches_manual_recursion():
    torch.manual_seed(41)
    time, batch, dim = 11, 4, 7
    gamma, gae_lambda = 0.99, 0.95
    phi = (1.0 - gamma) * torch.randn(time, batch, dim)
    psi_cur = torch.randn(time, batch, dim)
    psi_next = torch.randn(time, batch, dim)
    terminations = torch.zeros(time, batch)
    boundaries = torch.zeros(time, batch)
    valids = torch.ones(time, batch)
    terminations[5, 2] = boundaries[5, 2] = 1.0

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
        bootstrap = ((1.0 - terminations[step]) * valids[step]).unsqueeze(-1)
        continuation = (1.0 - boundaries[step]).unsqueeze(-1)
        delta = phi[step] + gamma * bootstrap * psi_next[step] - psi_cur[step]
        trace = delta + gamma * gae_lambda * continuation * trace
        expected[step] = trace
    torch.testing.assert_close(residual, expected)


def test_realized_beta_basis_is_zero_mean_and_identity_gram():
    torch.manual_seed(1)
    action_dim = 3
    raw = torch.zeros(1, 2 * action_dim)
    distribution = MODULE.beta_from_raw(raw, action_dim)
    actions = distribution.sample((120_000,)).squeeze(1)
    basis = MODULE.realized_beta_treatment(actions, raw, raw)

    assert basis.shape == (120_000, 9)
    assert basis.mean(0).abs().max() < 0.018
    gram = basis.T @ basis / basis.shape[0]
    torch.testing.assert_close(gram, torch.eye(9), rtol=0.045, atol=0.045)


def test_expected_beta_basis_matches_candidate_monte_carlo():
    torch.manual_seed(2)
    action_dim = 3
    old_raw = torch.tensor([[0.5, -0.2, 0.9, -0.6, 0.4, 0.1]])
    new_raw = torch.tensor([[-0.3, 0.7, 0.2, 0.8, -0.5, 0.6]])
    reference_raw = torch.zeros_like(old_raw)
    actions = MODULE.beta_from_raw(new_raw, action_dim).sample((160_000,)).squeeze(1)
    empirical = (
        MODULE.beta_basis_values(actions, reference_raw)
        - MODULE.expected_beta_basis(reference_raw, old_raw, action_dim)
    ).mean(0)
    analytic = MODULE.expected_beta_treatment_shift(
        old_raw, new_raw, reference_raw, action_dim
    ).squeeze(0)
    torch.testing.assert_close(empirical, analytic, rtol=0.06, atol=0.015)
    torch.testing.assert_close(
        MODULE.expected_beta_treatment_shift(
            old_raw, old_raw, reference_raw, action_dim
        ),
        torch.zeros(1, 9),
        atol=2e-6,
        rtol=0,
    )


def test_fixed_basis_transport_exactly_reconciles_behavior_policy_shift():
    torch.manual_seed(21)
    action_dim, value_dim = 3, 5
    reference_raw = torch.zeros(1, 2 * action_dim)
    old_raw = torch.randn(64, 2 * action_dim)
    current_raw = old_raw + 0.2 * torch.randn_like(old_raw)
    actions = MODULE.beta_from_raw(current_raw, action_dim).sample()
    response = torch.randn(64, value_dim, 9)
    old_baseline = torch.randn(64, value_dim)

    old_centered_basis = (
        MODULE.beta_basis_values(actions, reference_raw)
        - MODULE.expected_beta_basis(reference_raw, old_raw, action_dim)
    )
    outcome = old_baseline + torch.einsum(
        "bdk,bk->bd", response, old_centered_basis
    )
    transported_baseline = old_baseline + torch.einsum(
        "bdk,bk->bd",
        response,
        MODULE.expected_beta_treatment_shift(
            old_raw, current_raw, reference_raw, action_dim
        ),
    )
    current_treatment = MODULE.realized_beta_treatment(
        actions, current_raw, reference_raw
    )
    reconstructed = transported_baseline + torch.einsum(
        "bdk,bk->bd", response, current_treatment
    )
    torch.testing.assert_close(reconstructed, outcome, rtol=2e-5, atol=2e-5)


def test_beta_treatment_jacobian_matches_local_policy_shift():
    torch.manual_seed(22)
    action_dim = 3
    behavior = torch.randn(7, 2 * action_dim)
    reference = torch.zeros(1, 2 * action_dim)
    direction = torch.randn_like(behavior)
    jacobian = MODULE.beta_treatment_jacobian(
        behavior, reference, action_dim
    )
    epsilon = 1e-3
    actual = MODULE.expected_beta_treatment_shift(
        behavior,
        behavior + epsilon * direction,
        reference,
        action_dim,
    )
    linearized = epsilon * torch.einsum("bkp,bp->bk", jacobian, direction)
    torch.testing.assert_close(actual, linearized, rtol=0.012, atol=2e-6)


def test_outcome_model_detaches_embeddings_and_treatment_data():
    torch.manual_seed(3)
    model = MODULE.OrthogonalOutcomeModel(8, 9, 16, 0.05)
    embedding = torch.randn(32, 8, requires_grad=True)
    basis = torch.randn(32, 9, requires_grad=True)
    outcome = torch.randn(32, 8, requires_grad=True)
    valid = torch.ones(32, dtype=torch.bool)
    residual = outcome.detach() - model.baseline(embedding).detach()
    loss, prediction, scale = MODULE.outcome_effect_loss(
        model, embedding, basis, residual, valid
    )
    loss.backward()

    assert prediction.shape == outcome.shape
    assert scale.shape == (32,)
    assert embedding.grad is None
    assert basis.grad is None
    assert outcome.grad is None
    for component in (model.effect_body, model.effect_head, model.scale_head):
        assert all(parameter.grad is not None for parameter in component.parameters())


def test_actor_gradient_flows_through_beta_moments_only():
    torch.manual_seed(4)
    model = MODULE.OrthogonalOutcomeModel(8, 9, 16, 0.05)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    embedding = torch.randn(24, 8, requires_grad=True)
    old_raw = torch.randn(24, 6)
    new_raw = old_raw.clone().requires_grad_(True)
    goal = torch.randn(24, 8)
    basis = MODULE.expected_beta_treatment_shift(
        old_raw, new_raw, torch.zeros(1, 6), 3
    )
    prediction, scale = model.effect(embedding, basis)
    loss = (MODULE.latent_vector_error(prediction, goal) / scale.square()).mean()
    loss.backward()

    assert embedding.grad is None
    assert new_raw.grad is not None and new_raw.grad.abs().sum() > 0
    assert torch.isfinite(new_raw.grad).all()
    assert all(parameter.grad is None for parameter in model.parameters())


def test_zero_response_produces_zero_actor_gradient():
    torch.manual_seed(5)
    model = MODULE.OrthogonalOutcomeModel(8, 9, 16, 0.05)
    with torch.no_grad():
        model.effect_head.weight.zero_()
        model.effect_head.bias.zero_()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    old_raw = torch.randn(12, 6)
    new_raw = old_raw.clone().requires_grad_(True)
    prediction, _ = model.effect(
        torch.randn(12, 8),
        MODULE.expected_beta_treatment_shift(
            old_raw, new_raw, torch.zeros(1, 6), 3
        ),
    )
    loss = MODULE.latent_vector_error(prediction, torch.randn(12, 8)).mean()
    gradient = torch.autograd.grad(loss, new_raw)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))


def test_fixed_response_actor_step_reduces_full_vector_goal_error():
    torch.manual_seed(6)
    action_dim, basis_dim, value_dim = 2, 5, 6
    model = MODULE.OrthogonalOutcomeModel(value_dim, basis_dim, 12, 0.05)
    response = torch.randn(value_dim, basis_dim)
    with torch.no_grad():
        model.effect_head.weight.zero_()
        model.effect_head.bias.copy_(response.flatten())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    old_raw = torch.zeros(40, 2 * action_dim)
    desired_raw = 0.35 * torch.randn_like(old_raw)
    reference_raw = torch.zeros(1, 2 * action_dim)
    desired_basis = MODULE.expected_beta_treatment_shift(
        old_raw, desired_raw, reference_raw, action_dim
    )
    goal = desired_basis @ response.T
    current_raw = old_raw.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([current_raw], lr=0.08)
    initial = MODULE.latent_vector_error(
        model.effect(
            torch.randn(40, value_dim),
            MODULE.expected_beta_treatment_shift(
                old_raw, current_raw, reference_raw, action_dim
            ),
        )[0],
        goal,
    ).mean()
    embedding = torch.randn(40, value_dim)
    for _ in range(120):
        prediction, _ = model.effect(
            embedding,
            MODULE.expected_beta_treatment_shift(
                old_raw, current_raw, reference_raw, action_dim
            ),
        )
        loss = MODULE.latent_vector_error(prediction, goal).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final = MODULE.latent_vector_error(
        model.effect(
            embedding,
            MODULE.expected_beta_treatment_shift(
                old_raw, current_raw, reference_raw, action_dim
            ),
        )[0],
        goal,
    ).mean()
    assert final < 0.1 * initial


def test_elite_goal_averages_only_local_strictly_better_innovations():
    embeddings = torch.tensor([[[0.0], [0.1], [0.2], [3.0]]])
    horizons = torch.tensor([[100.0, 100.0, 100.0, 99.0]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    returns = torch.tensor([[1.0, 5.0, 3.0, 100.0]])
    innovations = torch.tensor([[[1.0], [7.0], [5.0], [99.0]]])

    goal, loser, distances, count = MODULE.local_elite_innovation_goals(
        embeddings, horizons, valid, returns, innovations, 2, 0
    )
    torch.testing.assert_close(goal[0, 0], torch.tensor([6.0]))
    assert loser.tolist() == [[True, False, True, False]]
    assert count.tolist() == [[2, 0, 1, 0]]
    assert torch.isfinite(distances[0, 0]).all()

    scaled_goal, scaled_loser, _, _ = MODULE.local_elite_innovation_goals(
        embeddings, horizons, valid, 1000.0 * returns + 17.0, innovations, 2, 0
    )
    torch.testing.assert_close(scaled_goal, goal)
    torch.testing.assert_close(scaled_loser, loser)


def test_outcome_model_recovers_synthetic_orthogonal_action_effect():
    torch.manual_seed(7)
    n, value_dim, basis_dim = 512, 5, 5
    model = MODULE.OrthogonalOutcomeModel(value_dim, basis_dim, 20, 0.05)
    embedding = torch.zeros(n, value_dim)
    basis = torch.randn(n, basis_dim)
    true_response = torch.randn(value_dim, basis_dim)
    outcomes = 0.3 + basis @ true_response.T
    valid = torch.ones(n, dtype=torch.bool)
    baseline_optimizer = torch.optim.Adam(
        list(model.baseline_body.parameters()) + list(model.baseline_head.parameters()),
        lr=0.03,
    )
    effect_optimizer = torch.optim.Adam(
        list(model.effect_body.parameters())
        + list(model.effect_head.parameters())
        + list(model.scale_head.parameters()),
        lr=0.03,
    )
    frozen_residual = outcomes - model.baseline(embedding).detach()
    for _ in range(180):
        loss, _, _ = MODULE.outcome_effect_loss(
            model, embedding, basis, frozen_residual, valid
        )
        effect_optimizer.zero_grad()
        loss.backward()
        effect_optimizer.step()
    for _ in range(80):
        loss, _ = MODULE.outcome_baseline_loss(model, embedding, outcomes, valid)
        baseline_optimizer.zero_grad()
        loss.backward()
        baseline_optimizer.step()

    full, baseline, gain, cosine, effect_fraction, scale = MODULE.outcome_metrics(
        model, embedding, basis, outcomes, valid
    )
    assert full < 0.08 * baseline
    assert gain > 0
    assert cosine > 0.95
    assert 0.8 < effect_fraction < 1.2
    assert scale > 0


def test_beta_mean_is_bounded_and_exact_kl_is_zero_at_identity():
    torch.manual_seed(8)
    raw = 5.0 * torch.randn(128, 12)
    low, high = -torch.ones(6), torch.ones(6)
    mean = MODULE.beta_mean_action(raw, low, high)
    assert ((mean > low) & (mean < high)).all()
    torch.testing.assert_close(MODULE.mean_beta_kl(raw, raw, 6), torch.tensor(0.0))


def test_procrustes_alignment_pins_rotated_encoder_frame():
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
        "obs_seq", "act_seq", "mask_seq", "sigreg_weight"
    ]
    loss, prediction, sigreg = lejepa(
        torch.randn(32, args.seq_len, 17),
        torch.randn(32, args.seq_len, 6),
        torch.ones(32, args.seq_len - 1),
        args.sigreg_weight,
    )
    loss.backward()
    for metric in (loss, prediction, sigreg):
        assert torch.isfinite(metric)
    for component in (
        lejepa.encoder, lejepa.action_encoder, lejepa.predictor, lejepa.pred_proj
    ):
        assert all(parameter.grad is not None for parameter in component.parameters())


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


def test_defaults_select_full_32d_outcome_and_27d_beta_basis():
    args = MODULE.Args()
    assert args.critic_mtp_horizon == 1
    assert args.emb_dim == 32
    assert args.elite_size == 4
    assert args.elite_horizon_tolerance == 0
    assert args.latent_actor_target_kl == 0.03
    assert args.actor_dist == "beta"
    assert 2 * 6 + 6 * 5 // 2 == 27
    assert not hasattr(args, "transport_target_kl")


def test_training_cross_fits_nuisance_and_has_no_scalar_policy_objective():
    source = (
        ROOT
        / "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v26_orthogonal_outcome_rlearner.py"
    ).read_text()
    lower = source.lower()
    assert "Dreamer3BucketHLGaussSupport" not in source
    assert "decode_value(" not in source
    assert "mb_advantages" not in source
    assert "logratio" not in source
    assert "beta_raw_score" not in source
    assert "preference.direction" not in source
    assert "expected_beta_treatment_shift(" in source
    assert "b_desired_outcome_effect[mb_inds]" in source
    assert "critic_params if outcome_ready else list(agent.critic_head.parameters())" in source
    assert source.index("outcome_residual_targets =") < source.index(
        "outcome_effect_optimizer.step()"
    ) < source.index("outcome_baseline_optimizer.step()")
    assert source.index("target_alignment.copy_(next_alignment)") < source.index(
        "outcome_effect_optimizer.step()"
    )
    assert source.index("for epoch in range(args.update_epochs):") < source.index(
        "outcome_effect_optimizer.step()"
    )
    assert "rankgauss(" not in lower
    assert "popart_update" not in source
    assert "ema_update(" not in source
