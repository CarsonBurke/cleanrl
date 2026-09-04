import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "cleanrl" / "ebp" / "ppo_continuous_action_bpc_clamped_output_v2.py"
SPEC = importlib.util.spec_from_file_location("bpc_clamped_output_v2", MODULE_PATH)
BPC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BPC)


def small_args(**overrides):
    values = dict(
        hidden_size=4,
        num_hidden_layers=2,
        posterior_jitter=1e-9,
        prior_column_covariance=0.1,
        inference_steps=3,
        inference_lr=0.5,
        inference_backtracks=4,
        target_bisection_steps=20,
    )
    values.update(overrides)
    return BPC.Args(**values)


def random_spd(size):
    matrix = torch.randn(size, size, dtype=torch.float64)
    return matrix @ matrix.T + 0.5 * torch.eye(size, dtype=torch.float64)


def total_energy(network, observation, states, output):
    result = observation.new_zeros(observation.shape[0])
    previous = observation
    for edge, state in zip(network.edges[:-1], states):
        result = result + edge.expected_energy(previous, state)
        previous = state
    return result + network.output_edge.expected_energy(states[-1], output)


def test_mnw_roundtrip_and_discounted_evidence_weight():
    torch.manual_seed(0)
    parameters = BPC.MNWParameters(
        torch.randn(3, 5, dtype=torch.float64),
        random_spd(5),
        random_spd(3),
        torch.tensor(7.0, dtype=torch.float64),
    )
    natural = BPC.mnw_to_natural(parameters, 1e-12)
    recovered = BPC.natural_to_mnw(natural, 1e-12)
    for expected, actual in zip(parameters, recovered):
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)

    x, y = torch.randn(6, 5, dtype=torch.float64), torch.randn(6, 3, dtype=torch.float64)
    stats = BPC.sufficient_statistics(x, y)
    updated = BPC.discounted_conjugate_update(natural, natural, stats, rho=0.97, evidence_weight=0.25)
    # With current == prior, rho vanishes and evidence is added exactly once.
    torch.testing.assert_close(updated.Lambda, natural.Lambda + 0.25 * stats.Sxx)
    torch.testing.assert_close(updated.Q, natural.Q + 0.25 * stats.Syx)
    torch.testing.assert_close(updated.R, natural.R + 0.25 * stats.Syy)
    torch.testing.assert_close(updated.xi, natural.xi + 0.25 * stats.N)
    torch.linalg.cholesky(BPC.natural_to_mnw(updated).V)


def test_actor_and_critic_have_full_bayesian_output_edges_updated_conjugately():
    torch.manual_seed(1)
    args = small_args()
    actor = BPC.BPCNetwork(3, 4, args, output_std=0.01)
    critic = BPC.BPCNetwork(3, 1, args, output_std=1.0)
    assert len(actor.edges) == args.num_hidden_layers + 1
    assert actor.output_edge.output_dim == 4
    assert critic.output_edge.output_dim == 1

    observation = torch.randn(8, 3)
    actor_states, actor_output = actor.forward_states(observation)
    critic_states, critic_output = critic.forward_states(observation)
    actor_before = [edge.natural_xi.clone() for edge in actor.edges]
    critic_before = [edge.natural_xi.clone() for edge in critic.edges]
    actor.commit_naturals_(actor.posterior_candidates(observation, actor_states, actor_output + 0.2, 0.97, 0.25))
    critic.commit_naturals_(critic.posterior_candidates(observation, critic_states, critic_output + 0.2, 0.99, 1.0))
    for before, edge in zip(actor_before, actor.edges):
        torch.testing.assert_close(edge.natural_xi - before, torch.tensor(2.0, dtype=torch.float64))
    for before, edge in zip(critic_before, critic.edges):
        torch.testing.assert_close(edge.natural_xi - before, torch.tensor(8.0, dtype=torch.float64))


def test_actor_pseudo_target_follows_score_and_obeys_both_kl_bounds():
    torch.manual_seed(2)
    args = small_args(actor_target_step=100.0, actor_target_mean_kl=1e-4, actor_target_sample_kl=3e-4)
    logits = torch.randn(32, 6)
    action = torch.rand(32, 3).clamp(1e-3, 1.0 - 1e-3)
    delta = torch.linspace(-5.0, 5.0, 32)
    target, score, kl, sample_scale, global_scale = BPC.bounded_actor_pseudo_target(logits, action, delta, args)
    alignment = ((target - logits) * score).sum(1)
    assert torch.all(alignment[delta > 0] >= -1e-8)
    assert torch.all(alignment[delta < 0] <= 1e-8)
    assert kl.max().item() <= args.actor_target_sample_kl * 1.001
    assert kl.mean().item() <= args.actor_target_mean_kl * 1.001
    assert torch.all((sample_scale >= 0) & (sample_scale <= 1))
    assert 0 <= global_scale <= 1


def test_actor_pseudo_target_kl_bound_survives_extreme_fp32_logits():
    torch.manual_seed(20)
    args = small_args(actor_target_step=1e4, actor_target_mean_kl=2e-5, actor_target_sample_kl=1e-4)
    logits = (10.0 * torch.randn(64, 6)).clamp(-25.0, 25.0)
    action = torch.rand(64, 3).clamp(1e-6, 1.0 - 1e-6)
    delta = 10.0 * torch.randn(64)
    target, _, kl, _, _ = BPC.bounded_actor_pseudo_target(logits, action, delta, args)
    assert target.dtype == torch.float32
    assert kl.dtype == torch.float64
    assert kl.max().item() <= args.actor_target_sample_kl
    assert kl.mean().item() <= args.actor_target_mean_kl


def test_clamped_output_settling_changes_only_hidden_states_and_reduces_energy():
    torch.manual_seed(3)
    args = small_args(inference_steps=5, inference_lr=1.0)
    network = BPC.BPCNetwork(3, 2, args, output_std=0.2)
    observation = torch.randn(12, 3)
    initial_states, prediction = network.forward_states(observation)
    fixed_output = prediction + torch.randn_like(prediction)
    fixed_output_before = fixed_output.clone()
    before = total_energy(network, observation, initial_states, fixed_output).sum()
    settled, steps, _, accepts, rejects = network.settle_clamped_output(observation, initial_states, fixed_output, args)
    after = total_energy(network, observation, settled, fixed_output).sum()
    assert steps == args.inference_steps
    assert accepts + rejects == args.inference_steps * args.num_hidden_layers
    assert after.item() <= before.item() + 1e-4
    # The caller-owned output is a clamp, not an inferred state.
    torch.testing.assert_close(fixed_output, fixed_output_before)
    assert any(not torch.equal(old, new) for old, new in zip(initial_states, settled))


def test_inference_curvature_cache_refreshes_at_configured_interval():
    torch.manual_seed(30)
    args = small_args(inference_curvature_refresh_interval=2)
    network = BPC.BPCNetwork(3, 2, args, output_std=0.2)
    first = network.inference_curvatures(args).clone()
    network.edges[0].cached_precision.mul_(4.0)
    cached = network.inference_curvatures(args).clone()
    refreshed = network.inference_curvatures(args).clone()
    torch.testing.assert_close(cached, first)
    assert not torch.allclose(refreshed, first)


def test_recent_anchor_reservoir_and_actor_guard_limit_function_drift():
    torch.manual_seed(4)
    args = small_args()
    reservoir = BPC.RecentObservationReservoir(10, 3, torch.device("cpu"))
    reservoir.add(torch.arange(24, dtype=torch.float32).view(8, 3))
    reservoir.add(torch.arange(24, 42, dtype=torch.float32).view(6, 3))
    assert reservoir.size == 10
    anchors = reservoir.sample(7)
    assert anchors.shape == (7, 3)

    network = BPC.BPCNetwork(3, 4, args, output_std=0.01)
    observations = torch.randn(16, 3)
    states, output = network.forward_states(observations[:8])
    candidates = network.posterior_candidates(observations[:8], states, output + 50.0, 0.97, 1.0)
    accepted_kl, proposal_kl, scale, limited, _ = BPC.guarded_actor_posterior_update(
        network, candidates, observations, max_kl=1e-7, corrective_trials=2
    )
    assert limited
    assert 0 <= scale < 1
    assert accepted_kl.item() <= 1.001e-7
    assert proposal_kl.item() >= accepted_kl.item()


def test_actor_posterior_guard_checks_extreme_logits_in_float64():
    torch.manual_seed(40)
    args = small_args()
    network = BPC.BPCNetwork(3, 4, args, output_std=0.01)
    edge = network.output_edge
    posterior = edge.posterior()
    mean = posterior.M.clone()
    mean[:, -1] = torch.tensor([60.0, -60.0, 45.0, -45.0], dtype=mean.dtype)
    edge.commit_natural_(
        BPC.mnw_to_natural(BPC.MNWParameters(mean, posterior.V, posterior.Psi, posterior.nu))
    )
    observations = torch.randn(16, 3)
    old_logits = network.forward_states(observations)[1].clone()
    states, output = network.forward_states(observations[:8])
    candidates = network.posterior_candidates(observations[:8], states, output + 0.5, 0.97, 1.0)
    BPC.guarded_actor_posterior_update(
        network,
        candidates,
        observations,
        max_kl=5e-4,
        corrective_trials=2,
        max_sample_kl=2e-3,
    )
    new_logits = network.forward_states(observations)[1]
    true_kl = BPC.policy_kl(old_logits.double(), new_logits.double())
    assert torch.isfinite(true_kl).all()
    assert true_kl.mean().item() <= 5.001e-4
    assert true_kl.max().item() <= 2.001e-3


@pytest.mark.parametrize("field,bad", [("Lambda", float("inf")), ("Q", float("nan")), ("R", float("inf")), ("xi", float("nan"))])
def test_nonfinite_posterior_proposal_rolls_back_saved_naturals_and_caches_exactly(field, bad):
    torch.manual_seed(21)
    args = small_args()
    network = BPC.BPCNetwork(3, 4, args, output_std=0.01)
    observations = torch.randn(8, 3)
    before = network.snapshot_natural()
    cache_before = [
        (edge.cached_M.clone(), edge.cached_V.clone(), edge.cached_precision.clone())
        for edge in network.edges
    ]
    invalid = [BPC.NaturalParameters(*(value.clone() for value in edge)) for edge in before]
    corrupted = getattr(invalid[0], field)
    corrupted[(0,) * corrupted.ndim] = bad
    accepted, proposed, scale, limited, rolled_back = BPC.guarded_actor_posterior_update(
        network, invalid, observations, max_kl=1e-4
    )
    assert accepted.item() == 0.0
    assert torch.isinf(proposed)
    assert scale == 0.0 and limited and rolled_back
    for expected_edge, actual_edge in zip(before, network.snapshot_natural()):
        for expected, actual in zip(expected_edge, actual_edge):
            torch.testing.assert_close(actual, expected)
    for expected, edge in zip(cache_before, network.edges):
        for cached_expected, cached_actual in zip(
            expected, (edge.cached_M, edge.cached_V, edge.cached_precision)
        ):
            torch.testing.assert_close(cached_actual, cached_expected)


def test_target_critic_soft_update_moves_all_edges_without_aliasing():
    torch.manual_seed(5)
    args = small_args()
    online = BPC.BPCNetwork(3, 1, args, output_std=1.0)
    target = BPC.BPCNetwork(3, 1, args, output_std=1.0)
    target.load_state_dict(online.state_dict())
    observations = torch.randn(8, 3)
    states, values = online.forward_states(observations)
    online.commit_naturals_(online.posterior_candidates(observations, states, values + 1.0, 0.99, 1.0))
    before = target.snapshot_natural()
    source = online.snapshot_natural()
    target.soft_update_from_(online, 0.2)
    after = target.snapshot_natural()
    for initial_edge, source_edge, target_edge in zip(before, source, after):
        for initial, final, actual in zip(initial_edge, source_edge, target_edge):
            torch.testing.assert_close(actual, initial.lerp(final, 0.2))
    online.edges[0].natural_Q.add_(1.0)
    assert not torch.equal(online.edges[0].natural_Q, target.edges[0].natural_Q)


def test_default_full_critic_guard_accepts_a_td_sized_update_within_limits():
    torch.manual_seed(50)
    args = BPC.Args()
    critic = BPC.BPCNetwork(17, 1, args, output_std=1.0)
    observations = torch.randn(args.num_envs, 17)
    guard_observations = torch.randn(128, 17)
    states, values = critic.forward_states(observations)
    targets = values + torch.randn_like(values)
    candidates = critic.posterior_candidates(
        observations,
        states,
        targets,
        args.critic_posterior_discount,
        args.critic_evidence_weight,
    )
    accepted_rms, accepted_max, _, _, scale, limited, rolled_back = (
        BPC.guarded_critic_posterior_update(
            critic,
            candidates,
            guard_observations,
            args.critic_posterior_rms_limit,
            args.critic_posterior_max_abs,
            args.posterior_guard_trials,
        )
    )
    assert limited
    assert not rolled_back and scale > 0
    assert accepted_rms.item() <= args.critic_posterior_rms_limit * 1.000001
    assert accepted_max.item() <= args.critic_posterior_max_abs * 1.000001


def test_arm_is_explicitly_one_step_without_trace_or_gae_controls():
    fields = BPC.Args.__dataclass_fields__
    assert "trace_lambda" not in fields
    assert "gae_lambda" not in fields
    assert not hasattr(BPC, "EligibilityTrace")
    args = small_args()
    BPC.validate_args(args)
    with pytest.raises(ValueError, match="discount"):
        BPC.validate_args(small_args(actor_posterior_discount=1.1))
