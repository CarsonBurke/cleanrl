import importlib.util
from pathlib import Path

import gymnasium as gym
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl" / "nextlat" / "ppo_continuous_action_nextlat_function_trust_v14.py"
)
SPEC = importlib.util.spec_from_file_location("nextlat_function_trust_v14", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _DummyEnvs:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(7,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))


def _probe_at(value, *, rows=4, actions=2, horizons=2, bins=3):
    value = torch.as_tensor(value, dtype=torch.float32)
    alpha = (2.0 + 0.35 * value).expand(rows, actions)
    beta = (2.5 - 0.2 * value).expand(rows, actions)
    logits = torch.stack((value, -0.4 * value, 0.1 * value))
    logits = logits[:bins].reshape(1, 1, bins).expand(rows, horizons, bins)
    return MODULE.BehaviorProbe(alpha, beta, logits)


def _transaction(
    parameter,
    optimizer,
    gradient,
    pre_task_probe,
    post_task_probe,
    evaluate_probe,
    *,
    trust_ratio=0.5,
    max_kl=1.0,
):
    return MODULE.apply_function_trust_transaction(
        [parameter],
        optimizer,
        [gradient],
        [torch.zeros_like(parameter)],
        actor_gradients=[torch.zeros_like(parameter)],
        critic_gradients=[torch.zeros_like(parameter)],
        pre_task_probe=pre_task_probe,
        post_task_probe=post_task_probe,
        critic_mask=torch.ones(post_task_probe.critic_logits.shape[:-1]),
        evaluate_probe=evaluate_probe,
        trust_ratio=trust_ratio,
        max_kl=max_kl,
    )


def test_beta_policy_kl_sums_actions_then_averages_rows():
    reference = MODULE.BehaviorProbe(
        torch.full((2, 3), 2.0),
        torch.full((2, 3), 3.0),
        torch.zeros(2, 1, 2),
    )
    candidate = MODULE.BehaviorProbe(
        torch.full((2, 3), 3.5),
        torch.full((2, 3), 1.75),
        torch.zeros(2, 1, 2),
    )

    actual = MODULE.beta_policy_kl(reference, candidate)
    per_action = torch.distributions.kl_divergence(
        torch.distributions.Beta(torch.tensor(2.0), torch.tensor(3.0)),
        torch.distributions.Beta(torch.tensor(3.5), torch.tensor(1.75)),
    )

    assert torch.allclose(actual, 3.0 * per_action)


def test_categorical_critic_kl_averages_only_valid_entries():
    reference_logits = torch.tensor(
        [
            [[1.0, -1.0], [0.0, 0.0], [4.0, -4.0]],
            [[-0.5, 0.5], [2.0, -2.0], [0.25, -0.25]],
        ]
    )
    candidate_logits = torch.tensor(
        [
            [[0.0, 0.0], [100.0, -100.0], [-4.0, 4.0]],
            [[0.5, -0.5], [-100.0, 100.0], [-0.25, 0.25]],
        ]
    )
    reference = MODULE.BehaviorProbe(
        torch.full((2, 1), 2.0),
        torch.full((2, 1), 2.0),
        reference_logits,
    )
    candidate = MODULE.BehaviorProbe(
        reference.actor_alpha,
        reference.actor_beta,
        candidate_logits,
    )
    mask = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    actual = MODULE.masked_categorical_critic_kl(reference, candidate, mask)
    ref_log = reference_logits.log_softmax(-1)
    cand_log = candidate_logits.log_softmax(-1)
    per_entry = (ref_log.exp() * (ref_log - cand_log)).sum(-1)
    expected = (per_entry[0, 0] + per_entry[1, 2]) / 2.0

    assert torch.allclose(actual, expected)


def test_conflict_projection_preserves_actor_and_critic_descent_signs_without_cap():
    raw = [torch.tensor([1.0, -2.0])]
    actor = [torch.tensor([1.0, 0.0])]
    critic = [torch.tensor([0.0, 1.0])]

    projected, stats = MODULE.project_predictive_updates(
        raw,
        actor_gradients=actor,
        critic_gradients=critic,
    )

    assert torch.allclose(projected[0], torch.tensor([0.0, -2.0]))
    assert torch.dot(actor[0], projected[0]) <= 0.0
    assert torch.dot(critic[0], projected[0]) <= 0.0
    assert stats["actor_first_order"] <= 0.0
    assert stats["critic_first_order"] <= 0.0
    assert torch.allclose(stats["projected_norm"], torch.tensor(2.0))


def test_boundary_roundoff_does_not_veto_valid_actor_projection():
    raw = torch.tensor(
        [-0.2298103422, -0.0073140212, -0.1305188388, 1.3700692654, -0.1109791920]
    )
    actor = torch.tensor(
        [-0.7281495929, 1.032345891, -0.5819520354, 0.3008017242, 0.1308227628]
    )
    critic = torch.tensor(
        [-2.271158934, -0.0109918527, 0.0613946542, -0.7550209761, 2.33305335]
    )

    projected, _ = MODULE.project_predictive_updates(
        [raw], actor_gradients=[actor], critic_gradients=[critic]
    )
    result = projected[0]
    actor_tolerance = MODULE._projection_dot_tolerance(
        actor.square().sum(), result.square().sum()
    )
    critic_tolerance = MODULE._projection_dot_tolerance(
        critic.square().sum(), result.square().sum()
    )

    assert result.norm().item() > 1.3
    assert torch.dot(actor, result) <= actor_tolerance
    assert torch.dot(critic, result) <= critic_tolerance


def _explicit_two_halfspace_projection(raw, actor, critic):
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


def test_random_projection_matches_explicit_convex_oracle_in_fp32_and_fp64():
    generator = torch.Generator().manual_seed(7183)
    sample_count, dimension = 1024, 7
    for dtype in (torch.float32, torch.float64):
        raw = torch.randn(sample_count, dimension, generator=generator, dtype=dtype)
        actor = torch.randn(sample_count, dimension, generator=generator, dtype=dtype)
        critic = torch.randn(sample_count, dimension, generator=generator, dtype=dtype)
        projected, _ = MODULE.project_predictive_updates(
            list(raw),
            actor_gradients=list(actor),
            critic_gradients=list(critic),
        )
        eps = torch.finfo(dtype).eps

        for index, result in enumerate(projected):
            oracle = _explicit_two_halfspace_projection(
                raw[index], actor[index], critic[index]
            )
            result64 = result.to(torch.float64)
            raw64 = raw[index].to(torch.float64)
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


def test_budget_is_squared_ratio_with_absolute_ceiling_and_scale_has_margin():
    uncapped = MODULE.function_trust_budget(torch.tensor(0.04), 0.05, 1.0)
    capped = MODULE.function_trust_budget(torch.tensor(10.0), 0.5, 1e-4)
    zero = MODULE.function_trust_budget(torch.tensor(0.0), 0.5, 1e-4)
    scale = MODULE.function_trust_scale(torch.tensor(0.04), torch.tensor(0.16))

    assert torch.allclose(uncapped, torch.tensor(0.0001))
    assert torch.allclose(capped, torch.tensor(1e-4))
    assert zero.item() == 0.0
    assert torch.allclose(scale, torch.tensor(0.45))


def test_zero_task_kl_has_no_floor_and_blocks_behavior_changing_update():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam(
        [parameter], lr=0.2, betas=(0.0, 0.0), eps=1e-8
    )
    reference = _probe_at(0.0)

    raw, projected, admitted, stats = _transaction(
        parameter,
        optimizer,
        torch.tensor([-1.0]),
        reference,
        reference,
        lambda: _probe_at(parameter[0]),
    )

    assert raw[0].norm() > 0.0
    assert projected[0].norm() > 0.0
    assert torch.equal(admitted[0], torch.zeros_like(admitted[0]))
    assert torch.equal(parameter, torch.zeros_like(parameter))
    assert stats["actor_budget"].item() == 0.0
    assert stats["critic_budget"].item() == 0.0


def test_function_null_direction_is_fully_admitted_at_zero_budget():
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam(
        [parameter], lr=0.1, betas=(0.0, 0.0), eps=1e-8
    )
    reference = _probe_at(0.0)

    _, projected, admitted, stats = _transaction(
        parameter,
        optimizer,
        torch.tensor([0.0, -1.0]),
        reference,
        reference,
        lambda: _probe_at(parameter[0]),
    )

    assert torch.allclose(admitted[0], projected[0])
    assert parameter[0].item() == 0.0
    assert parameter[1].item() > 0.0
    assert stats["scale"].item() == 1.0
    assert stats["accepted"].item() == 1.0


def test_square_root_scale_is_applied_and_exactly_verified():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam(
        [parameter], lr=0.2, betas=(0.0, 0.0), eps=1e-8
    )
    pre_task = _probe_at(-0.1)
    post_task = _probe_at(0.0)

    _, projected, admitted, stats = _transaction(
        parameter,
        optimizer,
        torch.tensor([-1.0]),
        pre_task,
        post_task,
        lambda: _probe_at(parameter[0]),
    )

    assert 0.0 < stats["scale"].item() < 1.0
    assert torch.allclose(admitted[0], projected[0] * stats["scale"])
    assert torch.allclose(parameter, admitted[0])
    assert stats["verified_actor_kl"] <= stats["actor_budget"] * 1.001 + 1e-8
    assert stats["verified_critic_kl"] <= stats["critic_budget"] * 1.001 + 1e-8
    assert stats["accepted"].item() == 1.0


def test_exact_verification_vetoes_nonlinear_scale_failure_and_rolls_back():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam(
        [parameter], lr=0.2, betas=(0.0, 0.0), eps=1e-8
    )
    pre_task = _probe_at(-0.1)
    post_task = _probe_at(0.0)

    def nonlinear_probe():
        value = parameter[0]
        if 0.0 < abs(value.item()) < 0.15:
            value = value.new_tensor(20.0)
        return _probe_at(value)

    _, _, admitted, stats = _transaction(
        parameter,
        optimizer,
        torch.tensor([-1.0]),
        pre_task,
        post_task,
        nonlinear_probe,
    )

    assert 0.0 < stats["proposal_scale"].item() < 1.0
    assert stats["accepted"].item() == 0.0
    assert torch.equal(admitted[0], torch.zeros_like(admitted[0]))
    assert torch.equal(parameter, torch.zeros_like(parameter))


def test_nonfinite_proposal_is_vetoed_but_predictive_adam_state_advances():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam(
        [parameter], lr=0.2, betas=(0.0, 0.0), eps=1e-8
    )
    pre_task = _probe_at(-0.1)
    post_task = _probe_at(0.0)

    def nonfinite_probe():
        if parameter[0].item() != 0.0:
            return _probe_at(parameter[0] * parameter[0].new_tensor(float("nan")))
        return _probe_at(0.0)

    _, _, admitted, stats = _transaction(
        parameter,
        optimizer,
        torch.tensor([-1.0]),
        pre_task,
        post_task,
        nonfinite_probe,
    )

    assert stats["accepted"].item() == 0.0
    assert torch.equal(admitted[0], torch.zeros_like(admitted[0]))
    assert torch.equal(parameter, torch.zeros_like(parameter))
    assert optimizer.state[parameter]["step"].item() == 1


def test_nonfinite_predictive_gradient_restores_task_state_and_keeps_adam_finite():
    parameter = torch.nn.Parameter(torch.tensor([0.375]))
    optimizer = torch.optim.Adam(
        [parameter], lr=0.2, betas=(0.9, 0.999), eps=1e-8
    )
    post_task_value = parameter.detach().clone()
    reference = _probe_at(post_task_value[0])

    _, _, admitted, stats = _transaction(
        parameter,
        optimizer,
        torch.tensor([float("nan")]),
        reference,
        reference,
        lambda: _probe_at(parameter[0]),
    )

    assert stats["numeric_valid"].item() == 0.0
    assert stats["accepted"].item() == 0.0
    assert torch.equal(admitted[0], torch.zeros_like(admitted[0]))
    assert torch.equal(parameter, post_task_value)
    assert optimizer.state[parameter]["step"].item() == 1
    assert all(
        not torch.is_tensor(value) or torch.isfinite(value).all()
        for value in optimizer.state[parameter].values()
    )


def test_nonfinite_private_predictor_gradient_cannot_poison_parameter_or_adam():
    parameter = torch.nn.Parameter(torch.tensor([0.25, -0.5]))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    before = parameter.detach().clone()

    step_norm = MODULE.apply_private_optimizer_step(
        [parameter],
        optimizer,
        [torch.tensor([float("nan"), float("inf")])],
    )

    assert step_norm.item() == 0.0
    assert torch.equal(parameter, before)
    assert optimizer.state[parameter]["step"].item() == 1
    assert all(
        not torch.is_tensor(value) or torch.isfinite(value).all()
        for value in optimizer.state[parameter].values()
    )


def test_rejection_restores_post_task_parameters_and_never_touches_task_adam():
    parameter = torch.nn.Parameter(torch.zeros(1))
    task_optimizer = torch.optim.Adam(
        [parameter], lr=0.1, betas=(0.0, 0.0), eps=1e-8
    )
    parameter.grad = torch.tensor([-1.0])
    task_optimizer.step()
    task_optimizer.zero_grad(set_to_none=True)
    post_task_value = parameter.detach().clone()
    task_state = {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in task_optimizer.state[parameter].items()
    }
    predictive_optimizer = torch.optim.Adam(
        [parameter], lr=0.2, betas=(0.0, 0.0), eps=1e-8
    )
    pre_task_probe = _probe_at(0.0)
    post_task_probe = _probe_at(post_task_value[0])

    def nonfinite_auxiliary_probe():
        if not torch.equal(parameter, post_task_value):
            return _probe_at(parameter[0] * parameter[0].new_tensor(float("nan")))
        return _probe_at(parameter[0])

    MODULE.apply_function_trust_transaction(
        [parameter],
        predictive_optimizer,
        [torch.tensor([-1.0])],
        [post_task_value],
        actor_gradients=[torch.zeros_like(parameter)],
        critic_gradients=[torch.zeros_like(parameter)],
        pre_task_probe=pre_task_probe,
        post_task_probe=post_task_probe,
        critic_mask=torch.ones(post_task_probe.critic_logits.shape[:-1]),
        evaluate_probe=nonfinite_auxiliary_probe,
        trust_ratio=0.5,
        max_kl=1.0,
    )

    assert torch.equal(parameter, post_task_value)
    for key, expected in task_state.items():
        actual = task_optimizer.state[parameter][key]
        if torch.is_tensor(expected):
            assert torch.equal(actual, expected)
        else:
            assert actual == expected


def test_behavior_probe_is_deterministic_and_does_not_advance_rng():
    args = MODULE.Args(
        hidden=8,
        k_blocks=1,
        n_experts=2,
        critic_mtp_horizon=2,
        num_bins=5,
    )
    agent = MODULE.Agent(_DummyEnvs(), args)
    observations = torch.randn(256, 7)
    state_before = torch.random.get_rng_state().clone()

    first = agent.get_behavior_probe(observations)
    state_after = torch.random.get_rng_state().clone()
    second = agent.get_behavior_probe(observations)

    assert torch.equal(state_before, state_after)
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert first[0].shape == (256, 3)
    assert first[2].shape == (256, 2, 5)


def test_policy_space_auxiliary_freezes_decoder_weights():
    args = MODULE.Args(
        hidden=8,
        k_blocks=1,
        n_experts=2,
        critic_mtp_horizon=2,
        num_bins=5,
    )
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


def test_optimizer_partitions_keep_predictor_private_and_decoder_task_only():
    args = MODULE.Args(
        hidden=8,
        k_blocks=1,
        n_experts=2,
        critic_mtp_horizon=2,
        num_bins=5,
    )
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
