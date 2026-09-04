import importlib.util
from pathlib import Path

import gymnasium as gym
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl" / "nextlat" / "ppo_continuous_action_nextlat_predadam_v9.py"
)
SPEC = importlib.util.spec_from_file_location("nextlat_predadam_v9", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _DummyEnvs:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(7,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))


def test_conflicting_predictive_update_is_removed():
    task = [torch.tensor([1.0, 0.0])]
    predictive = [torch.tensor([-3.0, 0.0])]

    admitted, stats = MODULE.admit_predictive_updates(
        task, predictive, max_ratio=0.05
    )

    assert torch.allclose(admitted[0], torch.zeros_like(admitted[0]))
    assert stats["raw_cosine"].item() == -1.0
    assert stats["accepted_fraction"].item() == 0.0


def test_orthogonal_predictive_update_is_trust_capped():
    task = [torch.tensor([3.0, 4.0])]
    predictive = [torch.tensor([-40.0, 30.0])]

    admitted, stats = MODULE.admit_predictive_updates(
        task, predictive, max_ratio=0.05
    )

    assert abs(admitted[0].norm().item() - 0.25) < 1e-6
    assert abs(torch.dot(task[0], admitted[0]).item()) < 1e-6
    assert abs(stats["accepted_fraction"].item() - 0.005) < 1e-7


def test_tensor_local_cap_prevents_cross_tensor_trust_transfer():
    task = [torch.tensor([1.0]), torch.tensor([100.0])]
    predictive = [torch.tensor([100.0]), torch.tensor([0.0])]
    # Both loss gradients see the first proposal as descent-compatible. A global-only
    # cap would allow norm 5 here by borrowing trust from the unrelated second tensor.
    actor = [torch.tensor([-1.0]), torch.tensor([0.0])]
    critic = [torch.tensor([-2.0]), torch.tensor([0.0])]

    admitted, stats = MODULE.admit_predictive_updates(
        task,
        predictive,
        max_ratio=0.05,
        actor_gradients=actor,
        critic_gradients=critic,
    )

    assert torch.allclose(admitted[0], torch.tensor([0.05]), atol=1e-7)
    assert torch.equal(admitted[1], torch.zeros_like(admitted[1]))
    assert stats["max_local_ratio"] <= 0.05 + 1e-7
    assert stats["admitted_norm"] <= 0.05 * stats["task_norm"] + 1e-7


def test_actor_and_critic_first_order_guards_are_separate():
    task = [torch.tensor([4.0, 0.0])]
    predictive = [torch.tensor([1.0, -2.0])]
    actor = [torch.tensor([1.0, 0.0])]
    critic = [torch.tensor([0.0, 1.0])]

    admitted, stats = MODULE.admit_predictive_updates(
        task,
        predictive,
        max_ratio=0.05,
        actor_gradients=actor,
        critic_gradients=critic,
    )

    # Actor-conflicting x is projected away; critic-compatible y survives at the local cap.
    assert abs(admitted[0][0].item()) < 1e-7
    assert abs(admitted[0].norm().item() - 0.2) < 1e-6
    assert torch.dot(actor[0], admitted[0]) <= 0.0
    assert torch.dot(critic[0], admitted[0]) <= 0.0
    assert stats["actor_first_order"] <= 0.0
    assert stats["critic_first_order"] <= 0.0


def test_boundary_roundoff_survives_meta_gate_and_parameter_caps():
    raw = torch.tensor(
        [-0.2298103422, -0.0073140212, -0.1305188388, 1.3700692654, -0.1109791920]
    )
    actor = torch.tensor(
        [-0.7281495929, 1.032345891, -0.5819520354, 0.3008017242, 0.1308227628]
    )
    critic = torch.tensor(
        [-2.271158934, -0.0109918527, 0.0613946542, -0.7550209761, 2.33305335]
    )
    task = torch.tensor([3.0, 4.0, 0.0, 0.0, 0.0])

    admitted, stats = MODULE.admit_predictive_updates(
        [task],
        [raw],
        max_ratio=0.05,
        actor_gradients=[actor],
        critic_gradients=[critic],
        gates=[torch.tensor(0.75)],
    )
    result = admitted[0]
    actor_tolerance = MODULE._projection_dot_tolerance(
        actor.square().sum(), result.square().sum()
    )
    critic_tolerance = MODULE._projection_dot_tolerance(
        critic.square().sum(), result.square().sum()
    )

    # This exact active actor-boundary candidate was spuriously zeroed by v9's strict
    # post-projection comparison. Its gated projection remains useful and is then capped.
    assert 0.24 < result.norm().item() <= 0.25 + 1e-6
    assert torch.dot(actor, result) <= actor_tolerance
    assert torch.dot(critic, result) <= critic_tolerance
    assert stats["numeric_valid"].item() == 1.0
    assert stats["max_local_ratio"] <= 0.05 + 1e-7


def _explicit_two_halfspace_projection(raw, actor, critic):
    """Small float64 active-set oracle independent of the vectorized implementation."""
    raw = raw.to(torch.float64)
    actor = actor.to(torch.float64)
    critic = critic.to(torch.float64)
    actor_sq = actor.square().sum()
    critic_sq = critic.square().sum()
    actor_critic = torch.dot(actor, critic)
    actor_raw = torch.dot(actor, raw)
    critic_raw = torch.dot(critic, raw)
    tolerance = 1e-11 * max(
        1.0,
        actor.norm().item() * raw.norm().item(),
        critic.norm().item() * raw.norm().item(),
    )
    candidates = [torch.zeros_like(raw)]
    if actor_raw <= tolerance and critic_raw <= tolerance:
        candidates.append(raw)

    if actor_sq > 0.0:
        actor_only = raw - actor_raw.clamp_min(0.0) / actor_sq * actor
        if torch.dot(actor, actor_only) <= tolerance and torch.dot(
            critic, actor_only
        ) <= tolerance:
            candidates.append(actor_only)
    if critic_sq > 0.0:
        critic_only = raw - critic_raw.clamp_min(0.0) / critic_sq * critic
        if torch.dot(actor, critic_only) <= tolerance and torch.dot(
            critic, critic_only
        ) <= tolerance:
            candidates.append(critic_only)

    determinant = actor_sq * critic_sq - actor_critic.square()
    if determinant > 1e-13 * actor_sq * critic_sq:
        actor_multiplier = (
            actor_raw * critic_sq - critic_raw * actor_critic
        ) / determinant
        critic_multiplier = (
            critic_raw * actor_sq - actor_raw * actor_critic
        ) / determinant
        if actor_multiplier >= -tolerance and critic_multiplier >= -tolerance:
            joint = raw - actor_multiplier * actor - critic_multiplier * critic
            if torch.dot(actor, joint) <= tolerance and torch.dot(
                critic, joint
            ) <= tolerance:
                candidates.append(joint)

    return min(candidates, key=lambda candidate: (candidate - raw).square().sum())


def _correlated_gradients(actor, generator, correlation):
    noise = torch.randn(
        actor.shape, generator=generator, dtype=actor.dtype, device=actor.device
    )
    actor_sq = actor.square().sum(-1, keepdim=True).clamp_min(
        torch.finfo(actor.dtype).tiny
    )
    noise = noise - (noise * actor).sum(-1, keepdim=True) / actor_sq * actor
    noise = noise * (
        actor_sq.sqrt()
        / noise.square().sum(-1, keepdim=True).sqrt().clamp_min(
            torch.finfo(actor.dtype).tiny
        )
    )
    return correlation * actor + (1.0 - correlation**2) ** 0.5 * noise


def test_random_projection_matches_oracle_fp32_fp64_and_correlated_gradients():
    generator = torch.Generator().manual_seed(7183)
    sample_count, dimension = 512, 7
    for dtype in (torch.float32, torch.float64):
        for correlation in (None, 0.9):
            raw = torch.randn(
                sample_count, dimension, generator=generator, dtype=dtype
            )
            actor = torch.randn(
                sample_count, dimension, generator=generator, dtype=dtype
            )
            if correlation is None:
                critic = torch.randn(
                    sample_count, dimension, generator=generator, dtype=dtype
                )
            else:
                critic = _correlated_gradients(actor, generator, correlation)
            gates = torch.rand(sample_count, generator=generator, dtype=dtype) * 0.9 + 0.1
            effective_raw = raw * gates[:, None]
            # Deliberately make both the tensor-local and global 1x caps inactive so the
            # admitted result isolates the gated active-set projection.
            task = torch.full_like(raw, 100.0)
            admitted, stats = MODULE.admit_predictive_updates(
                list(task),
                list(raw),
                max_ratio=1.0,
                actor_gradients=list(actor),
                critic_gradients=list(critic),
                gates=list(gates),
            )
            eps = torch.finfo(dtype).eps

            assert stats["numeric_valid"].item() == 1.0
            for index, result in enumerate(admitted):
                oracle = _explicit_two_halfspace_projection(
                    effective_raw[index], actor[index], critic[index]
                )
                result64 = result.to(torch.float64)
                raw64 = effective_raw[index].to(torch.float64)
                result_distance = (result64 - raw64).square().sum()
                oracle_distance = (oracle - raw64).square().sum()
                distance_tolerance = 512.0 * eps * max(
                    1.0, raw64.square().sum().item()
                )
                actor_tolerance = MODULE._projection_dot_tolerance(
                    actor[index].square().sum(), result.square().sum()
                )
                critic_tolerance = MODULE._projection_dot_tolerance(
                    critic[index].square().sum(), result.square().sum()
                )

                assert result_distance <= oracle_distance + distance_tolerance
                assert torch.dot(actor[index], result) <= actor_tolerance
                assert torch.dot(critic[index], result) <= critic_tolerance


def test_predictive_transaction_advances_adam_state_but_vetoes_unsafe_delta():
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([parameter], lr=0.1, betas=(0.0, 0.0), eps=1e-8)

    raw, admitted, ungated_safe, stats = MODULE.apply_predictive_trunk_transaction(
        [parameter],
        optimizer,
        [torch.tensor([-1.0, -1.0])],
        [torch.tensor([1.0, 1.0])],
        0.05,
        actor_gradients=[torch.tensor([1.0, 0.0])],
        critic_gradients=[torch.tensor([0.0, 1.0])],
    )

    assert raw[0].norm() > 0.0
    assert torch.equal(admitted[0], torch.zeros_like(admitted[0]))
    assert torch.equal(ungated_safe[0], admitted[0])
    assert torch.equal(parameter, torch.zeros_like(parameter))
    assert optimizer.state[parameter]["step"].item() == 1
    assert stats["actor_first_order"] <= 0.0
    assert stats["critic_first_order"] <= 0.0


def test_meta_gate_closes_on_harmful_forecast_then_reopens_from_ungated_reference():
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam(
        [parameter], lr=0.1, betas=(0.0, 0.0), eps=1e-8
    )

    raw, admitted, ungated_safe, stats = MODULE.apply_predictive_trunk_transaction(
        [parameter],
        optimizer,
        [torch.tensor([-1.0, 0.0])],
        [torch.tensor([1.0, 0.0])],
        1.0,
        actor_gradients=[torch.tensor([-1.0, 0.0])],
        critic_gradients=[torch.zeros(2)],
        gates=[torch.tensor(0.0)],
    )

    assert stats["numeric_valid"].item() == 1.0
    assert raw[0].norm().item() > 0.0
    assert torch.equal(admitted[0], torch.zeros_like(admitted[0]))
    assert ungated_safe[0].norm().item() > 0.0
    assert torch.equal(parameter, torch.zeros_like(parameter))

    previous = [ungated_safe[0].clone()]
    cross_ema = [torch.zeros(())]
    proposal_sq_ema = [torch.zeros(())]
    harmful_task = [-torch.ones_like(previous[0])]
    helpful_task = [torch.ones_like(previous[0])]

    for _ in range(6):
        gates = MODULE.update_predictive_meta_gates(
            previous,
            harmful_task,
            cross_ema,
            proposal_sq_ema,
            decay=0.8,
            warm=True,
        )
        # Mirrors training: even though the gate is closed, retain the next ungated
        # counterfactual proposal instead of overwriting the observation with zero.
        previous[0].copy_(ungated_safe[0])
    assert gates[0].item() == 0.0

    recovery = []
    for _ in range(20):
        gates = MODULE.update_predictive_meta_gates(
            previous,
            helpful_task,
            cross_ema,
            proposal_sq_ema,
            decay=0.8,
            warm=True,
        )
        previous[0].copy_(ungated_safe[0])
        recovery.append(gates[0].item())

    assert recovery[0] == 0.0
    assert any(gate > 0.0 for gate in recovery)
    assert recovery[-1] > 0.9


def test_disabled_meta_gate_matches_identity_gate_exactly():
    def run(gates):
        parameter = torch.nn.Parameter(torch.tensor([0.2, -0.3, 0.4]))
        optimizer = torch.optim.Adam(
            [parameter], lr=0.03, betas=(0.7, 0.9), eps=1e-5
        )
        result = MODULE.apply_predictive_trunk_transaction(
            [parameter],
            optimizer,
            [torch.tensor([-0.5, 0.25, -0.75])],
            [torch.tensor([0.02, -0.04, 0.03])],
            0.05,
            actor_gradients=[torch.tensor([-0.2, 0.3, -0.1])],
            critic_gradients=[torch.tensor([0.1, -0.4, -0.2])],
            gates=gates,
        )
        state = {
            name: value.detach().clone() if torch.is_tensor(value) else value
            for name, value in optimizer.state[parameter].items()
        }
        return parameter.detach().clone(), result, state

    disabled_parameter, disabled_result, disabled_state = run(None)
    identity_parameter, identity_result, identity_state = run([torch.tensor(1.0)])
    disabled_raw, disabled_admitted, disabled_safe, disabled_stats = disabled_result
    identity_raw, identity_admitted, identity_safe, identity_stats = identity_result

    assert torch.equal(disabled_parameter, identity_parameter)
    for disabled, identity in zip(
        disabled_raw + disabled_admitted + disabled_safe,
        identity_raw + identity_admitted + identity_safe,
    ):
        assert torch.equal(disabled, identity)
    for name in ("numeric_valid", "admitted_norm", "actor_first_order", "critic_first_order"):
        assert torch.equal(disabled_stats[name], identity_stats[name])
    assert disabled_state.keys() == identity_state.keys()
    for name in disabled_state:
        if torch.is_tensor(disabled_state[name]):
            assert torch.equal(disabled_state[name], identity_state[name])
        else:
            assert disabled_state[name] == identity_state[name]


def _optimizer_tensors_are_finite(optimizer):
    return all(
        not torch.is_tensor(value) or torch.isfinite(value).all()
        for state in optimizer.state.values()
        for value in state.values()
    )


def test_nonfinite_trunk_gradient_atomically_restores_parameters_and_moments():
    parameter = torch.nn.Parameter(torch.tensor([0.25, -0.5]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    parameter.grad = torch.tensor([-1.0, 0.5])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = parameter.detach().clone()
    prior_step = optimizer.state[parameter]["step"].item()

    raw, admitted, ungated_safe, stats = MODULE.apply_predictive_trunk_transaction(
        [parameter],
        optimizer,
        [torch.tensor([float("nan"), 1.0])],
        [torch.ones_like(parameter)],
        0.05,
        actor_gradients=[torch.zeros_like(parameter)],
        critic_gradients=[torch.zeros_like(parameter)],
    )

    assert torch.equal(raw[0], torch.zeros_like(raw[0]))
    assert torch.equal(admitted[0], torch.zeros_like(admitted[0]))
    assert torch.equal(ungated_safe[0], torch.zeros_like(ungated_safe[0]))
    assert torch.equal(parameter, before)
    assert optimizer.state[parameter]["step"].item() == prior_step + 1
    assert stats["numeric_valid"].item() == 0.0
    assert _optimizer_tensors_are_finite(optimizer)


def test_nonfinite_trunk_proposal_is_repaired_and_exactly_restored():
    parameter = torch.nn.Parameter(torch.tensor([0.25, -0.5]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    parameter.grad = torch.tensor([-1.0, 0.5])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    optimizer.state[parameter]["exp_avg"][0] = float("inf")
    before = parameter.detach().clone()

    raw, admitted, ungated_safe, stats = MODULE.apply_predictive_trunk_transaction(
        [parameter],
        optimizer,
        [torch.tensor([0.25, -0.75])],
        [torch.ones_like(parameter)],
        0.05,
        actor_gradients=[torch.zeros_like(parameter)],
        critic_gradients=[torch.zeros_like(parameter)],
    )

    assert torch.equal(raw[0], torch.zeros_like(raw[0]))
    assert torch.equal(admitted[0], torch.zeros_like(admitted[0]))
    assert torch.equal(ungated_safe[0], torch.zeros_like(ungated_safe[0]))
    assert torch.equal(parameter, before)
    assert stats["numeric_valid"].item() == 0.0
    assert _optimizer_tensors_are_finite(optimizer)


def test_nonfinite_private_predictor_gradient_is_atomic_across_parameters():
    first = torch.nn.Parameter(torch.tensor([0.25, -0.5]))
    second = torch.nn.Parameter(torch.tensor([0.75]))
    optimizer = torch.optim.Adam([first, second], lr=0.1)
    first.grad = torch.tensor([-1.0, 0.5])
    second.grad = torch.tensor([-0.25])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = [first.detach().clone(), second.detach().clone()]
    prior_step = optimizer.state[first]["step"].item()

    step_norm = MODULE.apply_private_optimizer_step(
        [first, second],
        optimizer,
        [torch.tensor([1.0, float("inf")]), torch.tensor([-1.0])],
    )

    assert step_norm.item() == 0.0
    assert torch.equal(first, before[0])
    assert torch.equal(second, before[1])
    assert optimizer.state[first]["step"].item() == prior_step + 1
    assert _optimizer_tensors_are_finite(optimizer)


def test_policy_space_auxiliary_freezes_decoder_weights():
    args = MODULE.Args(hidden=8, k_blocks=1, n_experts=2, critic_mtp_horizon=2, num_bins=5)
    agent = MODULE.Agent(_DummyEnvs(), args)
    source = torch.randn(4, args.hidden, requires_grad=True)
    target = torch.randn(4, args.hidden)

    with torch.no_grad():
        teacher, _, _ = agent._actor_dist(target)
    student, _, _ = agent._actor_dist_frozen_head(source)
    torch.distributions.kl_divergence(teacher, student).sum().backward()

    assert source.grad is not None
    assert source.grad.norm().item() > 0.0
    assert all(parameter.grad is None for parameter in agent.actor_alpha_head.parameters())
    assert all(parameter.grad is None for parameter in agent.actor_beta_head.parameters())


def test_optimizer_parameter_partitions_exclude_policy_head_from_auxiliary():
    args = MODULE.Args(hidden=8, k_blocks=1, n_experts=2, critic_mtp_horizon=2, num_bins=5)
    agent = MODULE.Agent(_DummyEnvs(), args)
    auxiliary_ids = {id(parameter) for parameter in agent.nextlat_parameters()}
    predictor_ids = {id(parameter) for parameter in agent.nextlat_predictor.parameters()}
    policy_head_ids = {
        id(parameter)
        for head in (agent.actor_alpha_head, agent.actor_beta_head)
        for parameter in head.parameters()
    }

    assert predictor_ids <= auxiliary_ids
    assert auxiliary_ids.isdisjoint(policy_head_ids)
    assert predictor_ids.isdisjoint({id(p) for p in agent.task_parameters()})


def test_horizon_zero_value_path_matches_full_mtp_head():
    args = MODULE.Args(hidden=8, k_blocks=1, n_experts=2, critic_mtp_horizon=3, num_bins=5)
    agent = MODULE.Agent(_DummyEnvs(), args)
    observations = torch.randn(6, 7)

    with torch.no_grad():
        full_h0 = agent.get_value(observations)[:, 0]
        direct_h0 = agent.get_value_h0(observations)

    assert torch.equal(direct_h0, full_h0)
