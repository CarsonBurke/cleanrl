import importlib.util
from pathlib import Path

import gymnasium as gym
import pytest
import torch


ROOT = Path(__file__).parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "cleanrl" / "nextlat" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


V14 = _load(
    "nextlat_function_trust_v14_reference",
    "ppo_continuous_action_nextlat_function_trust_v14.py",
)
V15 = _load(
    "nextlat_function_trust_livehead_v15",
    "ppo_continuous_action_nextlat_function_trust_livehead_v15.py",
)


class _DummyEnvs:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(7,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))


def _args(module, actor_dist="beta"):
    return module.Args(
        actor_dist=actor_dist,
        hidden=8,
        k_blocks=1,
        n_experts=2,
        critic_mtp_horizon=2,
        num_bins=5,
    )


def _agents(actor_dist):
    torch.manual_seed(8713)
    reference = V14.Agent(_DummyEnvs(), _args(V14, actor_dist))
    torch.manual_seed(8713)
    candidate = V15.Agent(_DummyEnvs(), _args(V15, actor_dist))
    return reference, candidate


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_v14_v15_seeded_initialization_and_forward_are_identical(actor_dist):
    reference, candidate = _agents(actor_dist)
    assert reference.state_dict().keys() == candidate.state_dict().keys()
    for name, tensor in reference.state_dict().items():
        assert torch.equal(tensor, candidate.state_dict()[name]), name

    observations = torch.randn(6, 7)
    if actor_dist == "beta":
        native_actions = torch.full((6, 3), 0.37)
    else:
        native_actions = torch.linspace(-0.6, 0.6, 18).reshape(6, 3)
    torch.manual_seed(122)
    reference_output = reference.get_action_and_value(observations, native_actions)
    torch.manual_seed(122)
    candidate_output = candidate.get_action_and_value(observations, native_actions)
    for expected, actual in zip(reference_output, candidate_output):
        assert torch.equal(expected, actual)


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_live_student_reaches_action_heads_but_teacher_and_critic_stay_frozen(actor_dist):
    _, agent = _agents(actor_dist)
    source = torch.randn(5, 8, requires_grad=True)
    target = torch.randn(5, 8, requires_grad=True)

    with torch.no_grad():
        teacher, _, _ = agent._actor_dist(target)
    student, _, _ = agent._actor_dist_live_head(source)
    torch.distributions.kl_divergence(teacher, student).sum().backward()

    assert source.grad is not None and source.grad.norm().item() > 0.0
    assert target.grad is None
    assert all(
        parameter.grad is not None and parameter.grad.norm().item() > 0.0
        for parameter in agent.action_head_parameters()
    )
    assert all(parameter.grad is None for parameter in agent.critic_head.parameters())


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_prediction_only_loss_has_zero_action_head_gradient(actor_dist):
    _, agent = _agents(actor_dist)
    source = torch.randn(5, 8)
    actions = torch.randn(5, 3)
    target = torch.randn(5, 8)
    predicted = agent.nextlat_predictor(torch.cat([source, actions], dim=-1))

    torch.nn.functional.smooth_l1_loss(predicted, target).backward()

    assert all(parameter.grad is None for parameter in agent.action_head_parameters())


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_live_head_preserves_v14_trunk_and_predictor_gradients(actor_dist):
    reference, candidate = _agents(actor_dist)
    observations = torch.randn(7, 7)
    future_observations = torch.randn(7, 7)
    actions = torch.randn(7, 3).clamp(-1.0, 1.0)

    def gradients(agent, live):
        current = agent.get_actor_feat(observations)
        predicted = agent.nextlat_predictor(torch.cat([current, actions], dim=-1))
        with torch.no_grad():
            target = agent.get_actor_feat(future_observations)
            teacher, _, _ = agent._actor_dist(target)
        decoder = (
            agent._actor_dist_live_head
            if live
            else agent._actor_dist_frozen_head
        )
        student, _, _ = decoder(predicted)
        loss = torch.nn.functional.smooth_l1_loss(predicted, target)
        loss = loss + torch.distributions.kl_divergence(teacher, student).sum(-1).mean()
        parameters = (
            agent.nextlat_trunk_parameters()
            + agent.nextlat_predictor_parameters()
        )
        return torch.autograd.grad(loss, parameters)

    frozen_gradients = gradients(reference, live=False)
    live_gradients = gradients(candidate, live=True)
    assert len(frozen_gradients) == len(live_gradients)
    for expected, actual in zip(frozen_gradients, live_gradients):
        assert torch.allclose(expected, actual, atol=1e-7, rtol=1e-6)


def test_live_head_default_and_global_clip_are_exactly_point_zero_one_five():
    assert V15.Args().nextlat_head_grad_clip == 0.015
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(1))
    third = torch.nn.Parameter(torch.zeros(2))
    fourth = torch.nn.Parameter(torch.zeros(1))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = torch.tensor([12.0])
    third.grad = torch.tensor([0.0, 0.0])
    fourth.grad = torch.tensor([0.0])

    delivered, stats = V15.prepare_live_head_auxiliary_gradients(
        [first, second, third, fourth], 0.015
    )
    norm = torch.cat([gradient.reshape(-1) for gradient in delivered]).float().norm()

    assert stats["valid"].item() == 1.0
    assert torch.allclose(stats["raw_norm"], torch.tensor(13.0))
    assert torch.allclose(norm, torch.tensor(0.015), atol=2e-9, rtol=2e-7)
    assert torch.allclose(stats["delivered_norm"], norm)


def test_finite_live_head_gradient_merges_into_existing_task_gradient():
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(1))
    first.grad = torch.tensor([1.0, 2.0])
    second.grad = torch.tensor([-2.0])
    auxiliary = [torch.tensor([0.25, -0.5]), torch.tensor([0.75])]
    expected_first = first.grad.clone() + auxiliary[0]
    expected_second = second.grad.clone() + auxiliary[1]

    stats = V15.merge_live_head_auxiliary_gradients(
        [first, second], auxiliary, valid=True
    )

    assert torch.equal(first.grad, expected_first)
    assert torch.equal(second.grad, expected_second)
    assert stats["valid"].item() == 1.0
    assert torch.allclose(stats["task_norm"], torch.tensor(3.0))
    expected_cosine = torch.tensor(-2.25) / (
        torch.tensor(3.0) * torch.tensor(0.9354143467)
    )
    assert torch.allclose(stats["task_aux_cosine"], expected_cosine)


def test_finite_merged_gradient_drives_the_same_adam_step_as_direct_sum():
    control = torch.nn.Parameter(torch.tensor([0.4, -0.3]))
    candidate = torch.nn.Parameter(control.detach().clone())
    control_optimizer = torch.optim.Adam([control], lr=0.02)
    candidate_optimizer = torch.optim.Adam([candidate], lr=0.02)
    task_gradient = torch.tensor([0.7, -0.2])
    auxiliary_gradient = torch.tensor([-0.1, 0.05])

    control.grad = task_gradient + auxiliary_gradient
    candidate.grad = task_gradient.clone()
    V15.merge_live_head_auxiliary_gradients(
        [candidate], [auxiliary_gradient], valid=True
    )
    control_optimizer.step()
    candidate_optimizer.step()

    assert torch.equal(control, candidate)
    for key in control_optimizer.state[control]:
        assert torch.equal(
            control_optimizer.state[control][key],
            candidate_optimizer.state[candidate][key],
        )


def _clone_optimizer_state(optimizer, parameter):
    return {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in optimizer.state[parameter].items()
    }


def test_nonfinite_head_aux_is_exact_task_only_adam_and_keeps_moments_healthy():
    control = [
        torch.nn.Parameter(torch.tensor([0.3, -0.4])),
        torch.nn.Parameter(torch.tensor([0.2])),
    ]
    candidate = [torch.nn.Parameter(parameter.detach().clone()) for parameter in control]
    control_optimizer = torch.optim.Adam(control, lr=0.03)
    candidate_optimizer = torch.optim.Adam(candidate, lr=0.03)

    warmup = [torch.tensor([0.2, -0.7]), torch.tensor([0.5])]
    task = [torch.tensor([-0.6, 0.25]), torch.tensor([0.9])]
    for parameters, optimizer in (
        (control, control_optimizer),
        (candidate, candidate_optimizer),
    ):
        for parameter, gradient in zip(parameters, warmup):
            parameter.grad = gradient.clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Auxiliary backward phase: one bad tensor vetoes the entire head group.
    candidate[0].grad = torch.tensor([float("nan"), 1.0])
    candidate[1].grad = torch.tensor([2.0])
    delivered, aux_stats = V15.prepare_live_head_auxiliary_gradients(
        candidate, 0.015
    )
    assert aux_stats["valid"].item() == 0.0
    assert all(torch.equal(gradient, torch.zeros_like(gradient)) for gradient in delivered)

    for parameter, gradient in zip(control, task):
        parameter.grad = gradient.clone()
    for parameter, gradient in zip(candidate, task):
        parameter.grad = gradient.clone()
    before_candidate_grads = [parameter.grad.clone() for parameter in candidate]
    merge_stats = V15.merge_live_head_auxiliary_gradients(
        candidate, delivered, valid=aux_stats["valid"]
    )
    assert merge_stats["valid"].item() == 0.0
    assert all(
        torch.equal(parameter.grad, expected)
        for parameter, expected in zip(candidate, before_candidate_grads)
    )
    control_optimizer.step()
    candidate_optimizer.step()

    for expected, actual in zip(control, candidate):
        assert torch.equal(expected, actual)
        expected_state = _clone_optimizer_state(control_optimizer, expected)
        actual_state = _clone_optimizer_state(candidate_optimizer, actual)
        assert expected_state.keys() == actual_state.keys()
        for key in expected_state:
            if torch.is_tensor(expected_state[key]):
                assert torch.equal(expected_state[key], actual_state[key])
                assert torch.isfinite(actual_state[key]).all()
            else:
                assert expected_state[key] == actual_state[key]


def test_zero_clip_is_exact_task_only_adam_parity():
    control = torch.nn.Parameter(torch.tensor([0.3, -0.8]))
    candidate = torch.nn.Parameter(control.detach().clone())
    control_optimizer = torch.optim.Adam([control], lr=0.03)
    candidate_optimizer = torch.optim.Adam([candidate], lr=0.03)
    task_gradient = torch.tensor([0.6, -0.25])

    candidate.grad = torch.tensor([4.0, -7.0])
    delivered, stats = V15.prepare_live_head_auxiliary_gradients([candidate], 0.0)
    assert stats["valid"].item() == 1.0
    assert stats["delivered_norm"].item() == 0.0
    control.grad = task_gradient.clone()
    candidate.grad = task_gradient.clone()
    V15.merge_live_head_auxiliary_gradients(
        [candidate], delivered, valid=stats["valid"]
    )
    control_optimizer.step()
    candidate_optimizer.step()

    assert torch.equal(control, candidate)
    for key in control_optimizer.state[control]:
        assert torch.equal(
            control_optimizer.state[control][key],
            candidate_optimizer.state[candidate][key],
        )


def test_nonfinite_ppo_head_gradient_fails_loudly():
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([float("nan"), 1.0])
    with pytest.raises(RuntimeError, match="non-finite"):
        torch.nn.utils.clip_grad_norm_(
            [parameter], 0.25, error_if_nonfinite=True
        )


def _probe(module, value, *, actor_dist="beta"):
    value = torch.as_tensor(value, dtype=torch.float32)
    if actor_dist == "beta":
        first = (2.0 + 0.35 * value).expand(4, 2)
        second = (2.5 - 0.2 * value).expand(4, 2)
    else:
        first = (0.2 * value).expand(4, 2)
        second = (1.0 + 0.1 * value).expand(4, 2)
    logits = torch.stack((value, -0.4 * value, 0.1 * value))
    logits = logits.reshape(1, 1, 3).expand(4, 2, 3)
    if module is V14:
        assert actor_dist == "beta"
        return module.BehaviorProbe(first, second, logits)
    return module.BehaviorProbe(first, second, logits, actor_dist=actor_dist)


def _run_transaction(module, *, actor_dist="beta"):
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam(
        [parameter], lr=0.2, betas=(0.0, 0.0), eps=1e-8
    )
    pre_task = _probe(module, -0.1, actor_dist=actor_dist)
    post_task = _probe(module, 0.0, actor_dist=actor_dist)
    result = module.apply_function_trust_transaction(
        [parameter],
        optimizer,
        [torch.tensor([-1.0])],
        [torch.zeros_like(parameter)],
        actor_gradients=[torch.zeros_like(parameter)],
        critic_gradients=[torch.zeros_like(parameter)],
        pre_task_probe=pre_task,
        post_task_probe=post_task,
        critic_mask=torch.ones(post_task.critic_logits.shape[:-1]),
        evaluate_probe=lambda: _probe(
            module, parameter[0], actor_dist=actor_dist
        ),
        trust_ratio=0.5,
        max_kl=1.0,
    )
    return parameter.detach().clone(), optimizer, result


def test_beta_trunk_function_trust_transaction_is_unchanged_from_v14():
    reference_parameter, reference_optimizer, reference = _run_transaction(V14)
    candidate_parameter, candidate_optimizer, candidate = _run_transaction(V15)

    assert torch.equal(reference_parameter, candidate_parameter)
    for reference_group, candidate_group in zip(reference[:3], candidate[:3]):
        for expected, actual in zip(reference_group, candidate_group):
            assert torch.equal(expected, actual)
    assert reference[3].keys() == candidate[3].keys()
    for key in reference[3]:
        assert torch.equal(reference[3][key], candidate[3][key]), key
    reference_state = next(iter(reference_optimizer.state.values()))
    candidate_state = next(iter(candidate_optimizer.state.values()))
    for key in reference_state:
        assert torch.equal(reference_state[key], candidate_state[key])


def test_gaussian_policy_probe_and_trunk_transaction_are_finite():
    reference = _probe(V15, -0.2, actor_dist="gaussian")
    candidate = _probe(V15, 0.3, actor_dist="gaussian")
    policy_kl = V15.gaussian_policy_kl(reference, candidate)
    reference_dist = torch.distributions.Normal(
        reference.actor_alpha, reference.actor_beta
    )
    candidate_dist = torch.distributions.Normal(
        candidate.actor_alpha, candidate.actor_beta
    )
    expected_kl = torch.distributions.kl_divergence(
        reference_dist, candidate_dist
    ).sum(-1).mean()
    parameter, _, transaction = _run_transaction(V15, actor_dist="gaussian")

    assert policy_kl.isfinite() and policy_kl.item() > 0.0
    assert torch.equal(policy_kl, expected_kl)
    assert torch.isfinite(parameter).all()
    assert transaction[3]["numeric_valid"].item() == 1.0


def test_probe_decomposition_exactly_restores_heads_and_is_not_additive():
    head = torch.nn.Parameter(torch.tensor([0.4]))
    pre_head = [torch.tensor([0.0])]
    pre_task_probe = _probe(V15, 0.0)
    post_mixed_probe = _probe(V15, 0.7)

    _, stats = V15.measure_livehead_probe_decomposition(
        [head],
        pre_head,
        pre_task_probe=pre_task_probe,
        post_mixed_probe=post_mixed_probe,
        evaluate_probe=lambda: _probe(V15, 0.3 + head[0]),
    )

    assert torch.equal(head, torch.tensor([0.4]))
    component_sum = stats["trunk_only_actor_kl"] + stats["mixed_head_actor_kl"]
    assert not torch.allclose(stats["joint_actor_kl"], component_sum)


def test_probe_decomposition_restores_heads_when_probe_evaluation_raises():
    head = torch.nn.Parameter(torch.tensor([0.4]))

    def broken_probe():
        raise RuntimeError("synthetic probe failure")

    with pytest.raises(RuntimeError, match="synthetic probe failure"):
        V15.measure_livehead_probe_decomposition(
            [head],
            [torch.tensor([0.0])],
            pre_task_probe=_probe(V15, 0.0),
            post_mixed_probe=_probe(V15, 0.7),
            evaluate_probe=broken_probe,
        )

    assert torch.equal(head, torch.tensor([0.4]))


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_optimizer_partition_has_live_heads_in_task_but_not_private_aux(actor_dist):
    _, agent = _agents(actor_dist)
    head_ids = {id(parameter) for parameter in agent.action_head_parameters()}
    task_ids = {id(parameter) for parameter in agent.task_parameters()}
    private_ids = {id(parameter) for parameter in agent.nextlat_parameters()}
    critic_ids = {id(parameter) for parameter in agent.critic_head.parameters()}

    assert head_ids <= task_ids
    assert head_ids.isdisjoint(private_ids)
    assert head_ids.isdisjoint(critic_ids)
