import copy
import importlib.util
import inspect
import sys
from pathlib import Path

import gymnasium as gym
import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "cleanrl/tpo/md/ppo_continuous_action_tpomd_local_pcopt_v1.py"
SPEC = importlib.util.spec_from_file_location("tpomd_local_pcopt_v1", SCRIPT)
PC = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = PC
SPEC.loader.exec_module(PC)


class _DummyEnvs:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(5,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))


def _agent(*, actor_dist="beta", hidden=8, bins=11):
    torch.manual_seed(193)
    return PC.Agent(
        _DummyEnvs(),
        PC.Args(
            actor_dist=actor_dist,
            hidden=hidden,
            num_bins=bins,
            pc_hidden_layers=4,
            pc_inference_steps=10,
        ),
    )


def _actor_problem(agent, batch=13, candidates=5, perturb=0.15):
    generator = torch.Generator().manual_seed(277)
    observations = torch.randn(batch, 5, generator=generator)
    if agent.actor_dist == "beta":
        candidate_zs = torch.rand(batch, candidates, 2, generator=generator).clamp(0.02, 0.98)
    else:
        candidate_zs = torch.randn(batch, candidates, 2, generator=generator)
    raw = agent.actor_chain(observations)
    logits = PC.actor_candidate_logits(raw, candidate_zs, agent)
    target = torch.softmax(logits + perturb * torch.randn(logits.shape, generator=generator), dim=-1)
    return observations, candidate_zs, target


def _critic_problem(agent, batch=13, perturb=0.15):
    generator = torch.Generator().manual_seed(311)
    observations = torch.randn(batch, 5, generator=generator)
    logits = agent.critic_chain(observations)
    target = torch.softmax(logits + perturb * torch.randn(logits.shape, generator=generator), dim=-1)
    return observations, target


def _project(chain, observations, settled):
    statistics = PC.empty_chain_statistics(chain, observations.device)
    PC.accumulate_chain_statistics(chain, statistics, observations, settled)
    return statistics, PC.chain_m_step_deltas(chain, statistics)


def _flat_deltas(chain, deltas):
    return torch.cat(
        [deltas[name].reshape(-1) for name, _ in chain.named_parameters()]
    )


def test_defaults_define_one_fixed_adam_free_projection():
    args = PC.Args()
    assert args.env_id == "HalfCheetah-v4"
    assert args.total_timesteps == 8_000_000
    assert args.seed == 1 and args.cuda
    assert not args.share_backbone
    assert args.pc_hidden_layers == 4
    assert args.pc_inference_steps == 10
    assert args.pc_chunk_size == 512
    agent = _agent()
    assert len(agent.actor_chain.edges) == 5
    assert len(agent.critic_chain.edges) == 5
    assert not any(parameter.requires_grad for parameter in agent.parameters())


def test_source_has_no_experimental_import_optimizer_or_global_backward():
    source = SCRIPT.read_text()
    assert "ppo_continuous_action_iterthink" not in source
    assert "torch.optim" not in source
    assert "optim.Adam" not in source
    assert ".backward(" not in source
    assert "autograd.grad" not in source
    assert "clip_grad" not in source
    main_source = inspect.getsource(PC.apply_atomic_chain_deltas)
    assert "add_(delta)" in main_source


def test_free_phase_is_exact_standard_forward_and_uses_tanh_into_output():
    agent = _agent()
    observations = torch.randn(7, 5)
    for chain in (agent.actor_chain, agent.critic_chain):
        activities = chain.forward_activities(observations)
        torch.testing.assert_close(activities[-1], chain(observations), rtol=0, atol=0)
        parent = observations
        for edge_index, edge in enumerate(chain.edges):
            features = parent if edge_index == 0 else torch.tanh(parent)
            expected = F.linear(features, edge.weight, edge.bias)
            torch.testing.assert_close(activities[edge_index], expected, rtol=0, atol=0)
            parent = activities[edge_index]
        changed = list(activities)
        changed[-2] = changed[-2] + 0.2
        changed_prediction = chain.predictions(observations, tuple(changed))[-1]
        if torch.count_nonzero(chain.edges[-1].weight):
            assert not torch.equal(changed_prediction, activities[-1])


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_analytic_actor_score_jacobian_matches_jacrev(actor_dist):
    agent = _agent(actor_dist=actor_dist)
    raw = torch.randn(3, 4, dtype=torch.float64)
    if actor_dist == "beta":
        candidates = torch.rand(3, 5, 2, dtype=torch.float64).clamp(0.03, 0.97)
    else:
        candidates = torch.randn(3, 5, 2, dtype=torch.float64)
    actual = PC.actor_score_jacobian(raw, candidates, agent)
    expected = torch.vmap(
        torch.func.jacrev(
            lambda one_raw, one_candidates: PC.actor_candidate_logits(
                one_raw.unsqueeze(0), one_candidates.unsqueeze(0), agent
            ).squeeze(0)
        )
    )(raw, candidates)
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)


def test_actor_categorical_gn_is_psd_and_matches_dense_formula():
    probabilities = torch.softmax(torch.randn(4, 6, dtype=torch.float64), dim=-1)
    jacobian = torch.randn(4, 6, 5, dtype=torch.float64)
    actual = PC.categorical_gn_from_score_jacobian(probabilities, jacobian)
    expected = torch.stack(
        [
            j.T @ (torch.diag(p) - p[:, None] * p[None, :]) @ j
            for p, j in zip(probabilities, jacobian, strict=True)
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert torch.linalg.eigvalsh(actual).min() >= -1e-12


def test_critic_linear_time_inverse_matches_dense_solve_at_full_support():
    probabilities = torch.softmax(torch.randn(3, 511, dtype=torch.float64), dim=-1)
    gradient = torch.randn(3, 511, dtype=torch.float64)
    actual = PC.solve_identity_plus_categorical_hessian(probabilities, gradient)
    identity = torch.eye(511, dtype=torch.float64)
    expected = torch.stack(
        [
            torch.linalg.solve(identity + torch.diag(p) - p[:, None] * p[None, :], g)
            for p, g in zip(probabilities, gradient, strict=True)
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)


def test_shared_hidden_gn_matches_mean_explicit_jacobian_grams():
    weight = torch.randn(7, 5, dtype=torch.float64)
    activity = torch.randn(11, 5, dtype=torch.float64)
    actual = PC.hidden_shared_gn(weight, activity)
    explicit = []
    for row in activity:
        derivative = 1.0 - row.tanh().square()
        jacobian = weight * derivative.unsqueeze(0)
        explicit.append(jacobian.T @ jacobian)
    expected = torch.eye(5, dtype=torch.float64) + torch.stack(explicit).mean(0)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_equal_terminal_targets_leave_every_activity_and_delta_exactly_zero():
    agent = _agent()
    observations, candidates, _ = _actor_problem(agent)
    actor_free = agent.actor_chain.forward_activities(observations)
    actor_target = torch.softmax(
        PC.actor_candidate_logits(actor_free[-1], candidates, agent), dim=-1
    )
    actor_result = PC.settle_actor_chain(agent, observations, candidates, actor_target, 10)
    for actual, expected in zip(actor_result.activities, actor_free, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _, actor_delta = _project(agent.actor_chain, observations, actor_result.activities)
    assert torch.count_nonzero(_flat_deltas(agent.actor_chain, actor_delta)) == 0

    critic_free = agent.critic_chain.forward_activities(observations)
    critic_target = torch.softmax(critic_free[-1], dim=-1)
    critic_result = PC.settle_critic_chain(agent, observations, critic_target, 10)
    for actual, expected in zip(critic_result.activities, critic_free, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _, critic_delta = _project(agent.critic_chain, observations, critic_result.activities)
    assert torch.count_nonzero(_flat_deltas(agent.critic_chain, critic_delta)) == 0


def test_settling_is_detached_and_reduces_energy_and_stationarity():
    agent = _agent()
    observations, candidates, target = _actor_problem(agent)
    actor = PC.settle_actor_chain(agent, observations, candidates, target, 10)
    assert all(not activity.requires_grad and activity.grad_fn is None for activity in actor.activities)
    assert actor.energies[-1] <= actor.energies[0]
    assert torch.isfinite(actor.stationarity_rms)

    observations, target = _critic_problem(agent)
    critic = PC.settle_critic_chain(agent, observations, target, 10)
    assert all(not activity.requires_grad and activity.grad_fn is None for activity in critic.activities)
    assert critic.energies[-1] <= critic.energies[0]
    assert torch.isfinite(critic.stationarity_rms)


def test_local_m_step_matches_lstsq_and_never_increases_stopped_sse():
    torch.manual_seed(401)
    chain = PC.PCChain(3, 4, 2, 2, output_std=0.1, zero_output=False)
    for parameter in chain.parameters():
        parameter.requires_grad_(False)
    observations = torch.randn(31, 3)
    free = chain.forward_activities(observations)
    settled = tuple(activity + 0.1 * torch.randn_like(activity) for activity in free)
    statistics, deltas = _project(chain, observations, settled)
    before, after = PC.projected_chain_sse(chain, statistics, deltas)
    assert torch.all(after <= before + 2e-6 * (1.0 + before))
    predictions = chain.predictions(observations, settled)
    for edge_index, (edge, child, prediction) in enumerate(
        zip(chain.edges, settled, predictions, strict=True)
    ):
        parent = observations if edge_index == 0 else settled[edge_index - 1]
        features = chain.edge_features(edge_index, parent)
        augmented = torch.cat((features, torch.ones(features.shape[0], 1)), dim=-1).double()
        residual = (child - prediction).double()
        expected = torch.linalg.lstsq(augmented, residual).solution.T
        actual = torch.cat(
            (
                deltas[f"edges.{edge_index}.weight"].double(),
                deltas[f"edges.{edge_index}.bias"].double()[:, None],
            ),
            dim=-1,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_centered_rank_deficient_m_step_preserves_weight_nullspace():
    torch.manual_seed(419)
    chain = PC.PCChain(2, 3, 1, 2, output_std=0.1, zero_output=False)
    for parameter in chain.parameters():
        parameter.requires_grad_(False)
    coordinate = torch.linspace(-2.0, 2.0, 17)
    observations = torch.stack((coordinate + 3.0, 2.0 * coordinate - 4.0), dim=-1)
    free = chain.forward_activities(observations)
    residual = torch.stack(
        (coordinate.square(), torch.sin(coordinate), torch.cos(coordinate)), dim=-1
    )
    settled = (free[0] + residual, free[1])
    statistics, deltas = _project(chain, observations, settled)
    null_direction = torch.tensor([-2.0, 1.0])
    weight_delta = deltas["edges.0.weight"]
    torch.testing.assert_close(
        weight_delta @ null_direction,
        torch.zeros(3),
        atol=2e-6,
        rtol=0,
    )
    design = torch.cat((observations.double(), torch.ones(17, 1, dtype=torch.float64)), dim=-1)
    correction = torch.cat(
        (weight_delta.double(), deltas["edges.0.bias"].double()[:, None]), dim=-1
    )
    predicted_residual = design @ correction.T
    expected_projection = design @ torch.linalg.lstsq(design, residual.double()).solution
    torch.testing.assert_close(predicted_residual, expected_projection, rtol=2e-5, atol=2e-5)
    before, after = PC.projected_chain_sse(chain, statistics, deltas)
    assert after[0] <= before[0]


def test_chunk_order_and_sample_replication_leave_projection_invariant():
    agent = _agent(hidden=6, bins=7)
    observations, candidates, target = _actor_problem(agent, batch=24, candidates=4)
    settled = PC.settle_actor_chain(agent, observations, candidates, target, 10).activities

    whole_stats, whole = _project(agent.actor_chain, observations, settled)
    chunked_stats = PC.empty_chain_statistics(agent.actor_chain, observations.device)
    for indices in (slice(16, 24), slice(8, 16), slice(0, 8)):
        PC.accumulate_chain_statistics(
            agent.actor_chain,
            chunked_stats,
            observations[indices],
            tuple(activity[indices] for activity in settled),
        )
    chunked = PC.chain_m_step_deltas(agent.actor_chain, chunked_stats)

    replicated_stats = PC.empty_chain_statistics(agent.actor_chain, observations.device)
    PC.accumulate_chain_statistics(
        agent.actor_chain,
        replicated_stats,
        observations.repeat(3, 1),
        tuple(activity.repeat(3, 1) for activity in settled),
    )
    replicated = PC.chain_m_step_deltas(agent.actor_chain, replicated_stats)
    for name in whole:
        torch.testing.assert_close(chunked[name], whole[name], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(replicated[name], whole[name], rtol=2e-5, atol=2e-6)
    assert sum(stat.rows for stat in whole_stats) == 5 * observations.shape[0]


def test_actor_projection_is_isolated_from_critic_parameters():
    agent = _agent()
    observations, candidates, target = _actor_problem(agent)
    expected = PC.settle_actor_chain(agent, observations, candidates, target, 4)
    with torch.no_grad():
        for parameter in agent.critic_chain.parameters():
            parameter.normal_(mean=10.0, std=3.0)
    actual = PC.settle_actor_chain(agent, observations, candidates, target, 4)
    for left, right in zip(actual.activities, expected.activities, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_atomic_application_is_order_invariant():
    first = _agent()
    second = copy.deepcopy(first)
    actor = {
        name: torch.randn_like(parameter) * 1e-3
        for name, parameter in first.actor_chain.named_parameters()
    }
    critic = {
        name: torch.randn_like(parameter) * 1e-3
        for name, parameter in first.critic_chain.named_parameters()
    }
    PC.apply_atomic_chain_deltas(first, actor, critic)
    PC.apply_atomic_chain_deltas(
        second,
        dict(reversed(tuple(actor.items()))),
        dict(reversed(tuple(critic.items()))),
    )
    for actual, expected in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_zero_initialized_critic_reaches_trunk_only_after_second_projection():
    agent = _agent()
    observations = torch.randn(23, 5)
    target = torch.softmax(torch.randn(23, 11), dim=-1)
    first_settle = PC.settle_critic_chain(agent, observations, target, 10)
    _, first = _project(agent.critic_chain, observations, first_settle.activities)
    assert torch.count_nonzero(first["edges.4.weight"]) > 0
    for edge_index in range(4):
        assert torch.count_nonzero(first[f"edges.{edge_index}.weight"]) == 0
        assert torch.count_nonzero(first[f"edges.{edge_index}.bias"]) == 0
    zero_actor = {
        name: torch.zeros_like(parameter)
        for name, parameter in agent.actor_chain.named_parameters()
    }
    PC.apply_atomic_chain_deltas(agent, zero_actor, first)

    second_settle = PC.settle_critic_chain(agent, observations, target, 10)
    _, second = _project(agent.critic_chain, observations, second_settle.activities)
    for edge_index in range(4):
        assert torch.count_nonzero(second[f"edges.{edge_index}.weight"]) > 0


def test_nonfinite_settle_result_fails_loudly():
    result = PC.SettleResult(
        (torch.tensor([[float("nan")]]),),
        torch.tensor([1.0]),
        torch.tensor(0.0),
    )
    with pytest.raises(FloatingPointError, match="non-finite"):
        PC.validate_settle_result("test", result)


def test_eager_and_inductor_settle_match_and_compiled_core_reads_live_parameters():
    agent = _agent(hidden=4, bins=7)
    observations, candidates, target = _actor_problem(agent, batch=3, candidates=3)
    eager = PC._settle_actor_chain_core(agent, observations, candidates, target, 2)
    compiled_core = torch.compile(
        lambda obs, cand, q: PC._settle_actor_chain_core(agent, obs, cand, q, 2),
        backend="inductor",
        fullgraph=True,
        dynamic=False,
    )
    compiled = compiled_core(observations, candidates, target)
    for actual, expected in zip(compiled.activities, eager.activities, strict=True):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(compiled.energies, eager.energies)
    torch.testing.assert_close(compiled.stationarity_rms, eager.stationarity_rms)
    with torch.no_grad():
        agent.actor_chain.edges[0].weight.add_(0.05)
    after = compiled_core(observations, candidates, target)
    eager_after = PC._settle_actor_chain_core(agent, observations, candidates, target, 2)
    assert not torch.equal(compiled.activities[0], after.activities[0])
    for actual, expected in zip(after.activities, eager_after.activities, strict=True):
        torch.testing.assert_close(actual, expected)


def test_ten_sweeps_match_materially_longer_frozen_settlement_direction():
    agent = _agent(hidden=6, bins=7)
    observations, candidates, target = _actor_problem(agent, batch=23, candidates=4)
    ten = PC.settle_actor_chain(agent, observations, candidates, target, 10)
    fifty = PC.settle_actor_chain(agent, observations, candidates, target, 50)
    _, ten_deltas = _project(agent.actor_chain, observations, ten.activities)
    _, fifty_deltas = _project(agent.actor_chain, observations, fifty.activities)
    ten_direction = _flat_deltas(agent.actor_chain, ten_deltas)
    fifty_direction = _flat_deltas(agent.actor_chain, fifty_deltas)
    assert ten.stationarity_rms < 1e-4
    assert ten.stationarity_rms > fifty.stationarity_rms
    assert F.cosine_similarity(ten_direction, fifty_direction, dim=0) > 0.999
    assert (ten_direction - fifty_direction).norm() / fifty_direction.norm() < 0.02


def test_small_boundary_nudge_pc_projection_aligns_with_bp_diagnostic():
    agent = _agent(hidden=6, bins=7)
    observations, candidates, target = _actor_problem(
        agent, batch=47, candidates=4, perturb=1e-3
    )
    settled = PC.settle_actor_chain(agent, observations, candidates, target, 10)
    _, deltas = _project(agent.actor_chain, observations, settled.activities)
    pc_direction = _flat_deltas(agent.actor_chain, deltas)
    names, parameters = zip(*agent.actor_chain.named_parameters())

    def loss(*values):
        raw = torch.func.functional_call(
            agent.actor_chain,
            dict(zip(names, values, strict=True)),
            (observations,),
        )
        return PC.actor_boundary_energy(raw, candidates, target, agent).sum()

    gradients = torch.func.grad(loss, argnums=tuple(range(len(parameters))))(*parameters)
    bp_direction = -torch.cat([gradient.reshape(-1) for gradient in gradients])
    cosine = F.cosine_similarity(pc_direction, bp_direction, dim=0)
    assert cosine > 0.6


def test_current_rollout_tpo_scale_has_no_ema_or_score_clamp():
    args = PC.Args(tpo_dyn_trust=True, tpo_eta_base=1.0)
    anchor = torch.zeros(5, 3)
    scores = torch.tensor([[0.0, 100.0, -100.0]]).expand(5, -1)
    result = PC.build_tpo_target(anchor, scores, torch.tensor(2.0), args)
    assert result.score_scale == 2.0
    expected = torch.softmax(scores[0] / result.score_scale / result.eta, dim=-1)
    torch.testing.assert_close(result.probabilities[0], expected)


def test_td_lambda_termination_and_truncation_have_distinct_bootstrap_and_same_trace_stop():
    rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    values = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    next_values = torch.tensor([[20.0], [40.0], [50.0], [60.0]])
    terminations = torch.tensor([[0.0], [0.0], [1.0], [0.0]])
    boundaries = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    valids = torch.ones_like(rewards)
    advantages, returns = PC.gae_lambda_returns(
        rewards,
        values,
        next_values,
        terminations,
        boundaries,
        valids,
        gamma=0.9,
        gae_lambda=0.8,
    )
    delta = rewards + 0.9 * next_values * (1.0 - terminations) - values
    # Step 1 is a truncation: it bootstraps V(final_obs), but no lambda
    # credit crosses the reset boundary.  Step 2 is a true termination: it
    # neither bootstraps nor carries a trace across the boundary.
    torch.testing.assert_close(advantages[1], delta[1])
    torch.testing.assert_close(advantages[2], rewards[2] - values[2])
    torch.testing.assert_close(advantages[0], delta[0] + 0.9 * 0.8 * delta[1])
    torch.testing.assert_close(returns, advantages + values)
