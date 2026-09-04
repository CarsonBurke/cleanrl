import ast
import importlib.util
import inspect
import sys
from pathlib import Path

import gymnasium as gym
import pytest
import torch


ROOT = Path(__file__).parents[1]


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


UNION = _load(
    "tpomd_nextlat_union_v17",
    "cleanrl/tpo/md/ppo_continuous_action_tpomd_nextlat_union_v17.py",
)
V13 = _load(
    "tpomd_prednext_v13_reference_for_union",
    "cleanrl/tpo/md/ppo_continuous_action_tpomd_prednext_v13.py",
)
BASE = _load(
    "tpomd_v5_reference_for_union",
    "cleanrl/iterthink/v24_d4hlgauss/rawret/ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_idxtransfer_tpomd_v5_dyntrust.py",
)


class _DummyEnvs:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(7,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))


def _args(module, actor_dist="beta"):
    return module.Args(
        actor_dist=actor_dist,
        hidden=8,
        k_blocks=2,
        n_experts=2,
        num_bins=7,
    )


def _semantic_ast(obj):
    tree = ast.parse(inspect.getsource(obj))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]
    return ast.dump(tree, include_attributes=False)


def test_forked_predictor_preserves_exact_base_parameters_and_rng():
    torch.manual_seed(287)
    base = BASE.Agent(_DummyEnvs(), _args(BASE))
    base_rng = torch.get_rng_state().clone()

    torch.manual_seed(287)
    union = UNION.Agent(_DummyEnvs(), _args(UNION))
    union_rng = torch.get_rng_state().clone()

    for name, value in base.state_dict().items():
        torch.testing.assert_close(union.state_dict()[name], value, rtol=0.0, atol=0.0)
    assert torch.equal(union_rng, base_rng)


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_default_policy_forward_is_exact_tpomd_v5_at_initialization(actor_dist):
    torch.manual_seed(917)
    base = BASE.Agent(_DummyEnvs(), _args(BASE, actor_dist))
    torch.manual_seed(917)
    union = UNION.Agent(_DummyEnvs(), _args(UNION, actor_dist))
    observations = torch.randn(11, 7)
    native_actions = (
        torch.full((11, 3), 0.37)
        if actor_dist == "beta"
        else torch.linspace(-0.8, 0.8, 33).reshape(11, 3)
    )
    candidates = native_actions[:, None].expand(-1, 8, -1).clone()

    torch.manual_seed(121)
    expected = base.get_action_and_value(
        observations,
        native_actions,
        candidate_zs=candidates,
    )
    torch.manual_seed(121)
    actual = union.get_action_and_value(
        observations,
        native_actions,
        candidate_zs=candidates,
    )
    assert len(actual) == len(expected)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
@pytest.mark.parametrize("sample_rollout", [False, True])
def test_rng_free_model_wrapper_preserves_exact_forward_gradients_and_rng(
    actor_dist,
    sample_rollout,
):
    torch.manual_seed(6201)
    reference = UNION.Agent(_DummyEnvs(), _args(UNION, actor_dist))
    torch.manual_seed(6201)
    candidate = UNION.Agent(_DummyEnvs(), _args(UNION, actor_dist))
    observations = torch.randn(13, 7)
    native_actions = None
    if not sample_rollout:
        native_actions = (
            torch.full((13, 3), 0.37)
            if actor_dist == "beta"
            else torch.linspace(-0.8, 0.8, 39).reshape(13, 3)
        )
    candidates = (
        torch.full((13, 8, 3), 0.43)
        if actor_dist == "beta"
        else torch.linspace(-1.1, 1.1, 13 * 8 * 3).reshape(13, 8, 3)
    )

    torch.manual_seed(3389)
    expected = reference.get_action_and_value(
        observations,
        native_actions,
        candidate_zs=candidates,
        return_actor_feat=True,
    )
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(3389)
    actual, _, _, _ = UNION.action_value_from_policy_outputs(
        candidate,
        UNION.policy_model_forward(candidate, observations),
        native_actions,
        candidates,
    )
    actual_rng = torch.get_rng_state().clone()

    assert len(actual) == len(expected)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0.0, atol=0.0)
    assert torch.equal(actual_rng, expected_rng)

    sum(tensor.float().sum() for tensor in expected[2:]).backward()
    sum(tensor.float().sum() for tensor in actual[2:]).backward()
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in candidate.named_parameters():
        expected_gradient = reference_parameters[name].grad
        if expected_gradient is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(
                parameter.grad,
                expected_gradient,
                rtol=0.0,
                atol=0.0,
            )


def test_rng_free_policy_model_wrapper_is_fullgraph_compilable():
    torch.manual_seed(211)
    eager_agent = UNION.Agent(_DummyEnvs(), _args(UNION))
    torch.manual_seed(211)
    compiled_agent = UNION.Agent(_DummyEnvs(), _args(UNION))
    observations = torch.randn(16, 7)
    compiled = torch.compile(
        lambda batch: UNION.policy_model_forward(compiled_agent, batch),
        backend="eager",
        dynamic=False,
        fullgraph=True,
    )

    rng_before = torch.get_rng_state().clone()
    expected = UNION.policy_model_forward(eager_agent, observations)
    rng_after_eager = torch.get_rng_state().clone()
    actual = compiled(observations)
    rng_after_compiled = torch.get_rng_state().clone()
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0.0, atol=0.0)
    assert torch.equal(rng_before, rng_after_eager)
    assert torch.equal(rng_before, rng_after_compiled)


@pytest.mark.parametrize(
    "name",
    [
        "IndexedTransferBranch",
        "ThinkBlock",
        "ThinkTrunk",
        "tpo_restricted_target",
        "tpo_reverse_kl",
        "build_nextlat_mask",
    ],
)
def test_tpo_and_rollout_components_remain_ast_identical_to_repaired_v13(name):
    assert _semantic_ast(getattr(UNION, name)) == _semantic_ast(getattr(V13, name))


def test_tpo_target_q_and_one_sided_kl_are_unchanged():
    generator = torch.Generator().manual_seed(19)
    anchor = torch.log_softmax(torch.randn(23, 8, generator=generator), dim=-1)
    scores = torch.randn(23, 8, generator=generator)
    eta = 1.7
    expected_q = torch.softmax(anchor + scores / eta, dim=-1)
    expected_kl = (
        anchor.exp() * (anchor - torch.log_softmax(anchor + scores / eta, dim=-1))
    ).sum(-1).mean()
    torch.testing.assert_close(UNION.tpo_restricted_target(anchor, scores, eta), expected_q)
    torch.testing.assert_close(UNION.tpo_reverse_kl(anchor, scores, eta), expected_kl)


def test_predictor_preserves_v13_absolute_latent_semantics():
    agent = UNION.Agent(_DummyEnvs(), _args(UNION))
    latent = torch.randn(6, 8)
    action = torch.randn(6, 3)
    predicted = agent.nextlat_predictor(torch.cat([latent, action], dim=-1))
    reference = V13.Agent(_DummyEnvs(), _args(V13))
    reference.nextlat_predictor.load_state_dict(agent.nextlat_predictor.state_dict())
    expected = reference.nextlat_predictor(torch.cat([latent, action], dim=-1))
    torch.testing.assert_close(predicted, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_frozen_decoder_updates_predictor_and_latent_but_no_action_head(actor_dist):
    agent = UNION.Agent(_DummyEnvs(), _args(UNION, actor_dist))
    source = torch.randn(7, 8, requires_grad=True)
    action = torch.randn(7, 3).clamp(-1.0, 1.0)
    target = torch.randn(7, 8, requires_grad=True)
    prediction = agent.nextlat_predictor(torch.cat([source, action], dim=-1))
    with torch.no_grad():
        teacher, _, _ = agent._actor_dist_frozen_head(target)
    student, _, _ = agent._actor_dist_frozen_head(prediction)
    loss = torch.nn.functional.smooth_l1_loss(prediction, target.detach())
    loss = loss + torch.distributions.kl_divergence(teacher, student).sum(-1).mean()
    loss.backward()

    assert source.grad is not None and source.grad.norm() > 0.0
    assert target.grad is None
    assert all(
        parameter.grad is not None and parameter.grad.norm() > 0.0
        for parameter in agent.nextlat_predictor_parameters()
    )
    action_heads = (
        [agent.actor_head, agent.actor_logvar_head]
        if actor_dist == "gaussian"
        else [agent.actor_alpha_head, agent.actor_beta_head]
    )
    assert all(
        parameter.grad is None
        for head in action_heads
        for parameter in head.parameters()
    )
    assert all(parameter.grad is None for parameter in agent.critic_head.parameters())


def test_optimizer_and_gradient_sets_are_exact_task_adam_trunk_union():
    agent = UNION.Agent(_DummyEnvs(), _args(UNION))
    all_ids = {id(parameter) for parameter in agent.parameters()}
    task_ids = {id(parameter) for parameter in agent.task_parameters()}
    trunk_ids = {id(parameter) for parameter in agent.nextlat_trunk_parameters()}
    predictor_ids = {id(parameter) for parameter in agent.nextlat_predictor_parameters()}
    nextlat_ids = {id(parameter) for parameter in agent.nextlat_parameters()}
    critic_head_ids = {id(parameter) for parameter in agent.critic_head.parameters()}

    assert task_ids | predictor_ids == all_ids
    assert task_ids.isdisjoint(predictor_ids)
    assert nextlat_ids == trunk_ids | predictor_ids
    assert not (trunk_ids & predictor_ids)
    assert nextlat_ids.isdisjoint(critic_head_ids)
    assert UNION.Args().nextlat_trunk_grad_clip == 0.025
    assert UNION.Args().nextlat_predictor_grad_clip == 0.25


def test_defaults_match_v13_except_the_isolated_simple_union_intervention():
    expected = vars(V13.Args())
    actual = vars(UNION.Args())
    expected.pop("exp_name")
    actual.pop("exp_name")
    assert expected.pop("prednext_trust_ratio") == 0.05
    assert expected.pop("nextlat_trunk_grad_clip") == 0.25
    assert actual.pop("nextlat_trunk_grad_clip") == 0.025
    assert actual == expected


def test_task_adam_is_exact_tpomd_v5_when_auxiliary_group_is_empty():
    torch.manual_seed(411)
    base = BASE.Agent(_DummyEnvs(), _args(BASE))
    torch.manual_seed(411)
    union = UNION.Agent(_DummyEnvs(), _args(UNION))
    base_optimizer = torch.optim.Adam(base.parameters(), lr=3e-4, eps=1e-5)
    union_optimizer = torch.optim.Adam(union.task_parameters(), lr=3e-4, eps=1e-5)
    generator = torch.Generator().manual_seed(83)
    observations = torch.randn(16, 7, generator=generator)
    native_actions = torch.sigmoid(torch.randn(16, 3, generator=generator))
    candidates = torch.sigmoid(torch.randn(16, 8, 3, generator=generator))
    mirror_target = torch.softmax(torch.randn(16, 8, generator=generator), dim=-1)
    value_target = torch.softmax(torch.randn(16, 7, generator=generator), dim=-1)

    def losses(agent):
        _, _, _, entropy, value_logits, candidate_logprobs = agent.get_action_and_value(
            observations,
            native_actions,
            candidate_zs=candidates,
        )
        actor_loss = (
            -(mirror_target * torch.log_softmax(candidate_logprobs, -1)).sum(-1)
        ).mean() - 0.0 * entropy.mean()
        value_loss = (
            -(value_target * torch.log_softmax(value_logits, -1)).sum(-1)
        ).mean()
        return actor_loss, value_loss

    base_actor_loss, base_value_loss = losses(base)
    base_optimizer.zero_grad(set_to_none=True)
    base_value_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(base.critic_parameters(), 0.25)
    base_value_gradients = UNION.capture_gradients(base.critic_parameters())
    base_optimizer.zero_grad(set_to_none=True)
    base_actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(base.actor_parameters(), 0.25)
    for parameter, gradient in base_value_gradients.items():
        parameter.grad = (
            gradient if parameter.grad is None else parameter.grad + gradient
        )
    base_optimizer.step()

    union_actor_loss, union_value_loss = losses(union)
    union_optimizer.zero_grad(set_to_none=True)
    union_value_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(union.critic_parameters(), 0.25)
    union_value_gradients = UNION.capture_gradients(union.critic_parameters())
    union_optimizer.zero_grad(set_to_none=True)
    union_actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(union.actor_parameters(), 0.25)
    union_actor_gradients = UNION.capture_gradients(union.actor_parameters())
    UNION.apply_union_optimizer_step(
        union.task_parameters(),
        union_optimizer,
        actor_gradients=union_actor_gradients,
        critic_gradients=union_value_gradients,
        auxiliary_gradients={},
    )

    union_state = union.state_dict()
    for name, value in base.state_dict().items():
        torch.testing.assert_close(union_state[name], value, rtol=0.0, atol=0.0)


def test_auxiliary_trunk_uses_one_global_point_zero_two_five_clip():
    parameters = [
        torch.nn.Parameter(torch.zeros(2)),
        torch.nn.Parameter(torch.zeros(1)),
        torch.nn.Parameter(torch.zeros(2)),
    ]
    parameters[0].grad = torch.tensor([3.0, 4.0])
    parameters[1].grad = torch.tensor([12.0])
    parameters[2].grad = torch.tensor([0.0, 0.0])
    raw = torch.cat([parameter.grad.reshape(-1) for parameter in parameters]).norm()
    norm = UNION.clip_grad_norm_async_fail_loud_(parameters, 0.025)
    delivered = torch.cat([parameter.grad.reshape(-1) for parameter in parameters]).norm()

    torch.testing.assert_close(raw, torch.tensor(13.0))
    torch.testing.assert_close(norm, raw)
    torch.testing.assert_close(delivered, torch.tensor(0.025), atol=2e-8, rtol=2e-7)


def test_private_predictor_uses_independent_point_two_five_clip():
    parameters = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(1))]
    parameters[0].grad = torch.tensor([3.0, 4.0])
    parameters[1].grad = torch.tensor([12.0])
    norm = UNION.clip_grad_norm_async_fail_loud_(parameters, 0.25)
    gradients = UNION.capture_gradients(parameters)
    delivered = torch.cat([gradients[parameter].reshape(-1) for parameter in parameters]).norm()

    torch.testing.assert_close(norm, torch.tensor(13.0))
    torch.testing.assert_close(delivered, torch.tensor(0.25), atol=2e-7, rtol=2e-7)


def test_private_predictor_step_and_adam_moments_match_direct_oracle():
    control = torch.nn.Parameter(torch.tensor([0.4, -0.3]))
    candidate = torch.nn.Parameter(control.detach().clone())
    control_optimizer = torch.optim.Adam([control], lr=0.02, eps=1e-5)
    candidate_optimizer = torch.optim.Adam([candidate], lr=0.02, eps=1e-5)
    gradient = torch.tensor([0.7, -0.2])

    control.grad = gradient.clone()
    control_optimizer.step()
    UNION.apply_private_predictor_step(
        [candidate], candidate_optimizer, {candidate: gradient.clone()}
    )

    assert torch.equal(candidate, control)
    expected_state = _clone_optimizer_state(control_optimizer, control)
    actual_state = _clone_optimizer_state(candidate_optimizer, candidate)
    for key in expected_state:
        if torch.is_tensor(expected_state[key]):
            assert torch.equal(actual_state[key], expected_state[key])
        else:
            assert actual_state[key] == expected_state[key]


def _clone_optimizer_state(optimizer, parameter):
    return {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in optimizer.state[parameter].items()
    }


def test_union_sum_and_shared_adam_parameters_and_moments_match_direct_oracle():
    initial = [torch.tensor([0.4, -0.3]), torch.tensor([0.2]), torch.tensor([-0.7])]
    control = [torch.nn.Parameter(value.clone()) for value in initial]
    candidate = [torch.nn.Parameter(value.clone()) for value in initial]
    control_optimizer = torch.optim.Adam(control, lr=0.02, eps=1e-5)
    candidate_optimizer = torch.optim.Adam(candidate, lr=0.02, eps=1e-5)
    actor = [torch.tensor([0.7, -0.2]), torch.tensor([0.3])]
    critic = [torch.tensor([-0.4, 0.5]), torch.tensor([-0.8])]
    auxiliary = [torch.tensor([-0.1, 0.05]), torch.tensor([0.9])]

    control[0].grad = actor[0] + critic[0] + auxiliary[0]
    control[1].grad = actor[1]
    control[2].grad = critic[1] + auxiliary[1]
    control_optimizer.step()
    UNION.apply_union_optimizer_step(
        candidate,
        candidate_optimizer,
        actor_gradients={candidate[0]: actor[0], candidate[1]: actor[1]},
        critic_gradients={candidate[0]: critic[0], candidate[2]: critic[1]},
        auxiliary_gradients={candidate[0]: auxiliary[0], candidate[2]: auxiliary[1]},
    )

    for expected, actual in zip(control, candidate):
        assert torch.equal(actual, expected)
        expected_state = _clone_optimizer_state(control_optimizer, expected)
        actual_state = _clone_optimizer_state(candidate_optimizer, actual)
        assert expected_state.keys() == actual_state.keys()
        for key in expected_state:
            if torch.is_tensor(expected_state[key]):
                assert torch.equal(actual_state[key], expected_state[key])
            else:
                assert actual_state[key] == expected_state[key]


def test_breaker_predictor_only_capture_skips_trunk_clip_and_scan(monkeypatch):
    trunk = [torch.nn.Parameter(torch.zeros(2))]
    predictor = [torch.nn.Parameter(torch.zeros(2))]
    predictor[0].grad = torch.tensor([3.0, 4.0])
    clip_calls = []
    capture_calls = []
    real_clip = UNION.clip_grad_norm_async_fail_loud_
    real_capture = UNION.capture_gradients

    def tracking_clip(parameters, max_norm, norm_type=2.0):
        parameters = list(parameters)
        clip_calls.append((parameters, max_norm))
        return real_clip(parameters, max_norm, norm_type)

    def tracking_capture(parameters):
        parameters = list(parameters)
        capture_calls.append(parameters)
        return real_capture(parameters)

    monkeypatch.setattr(UNION, "clip_grad_norm_async_fail_loud_", tracking_clip)
    monkeypatch.setattr(UNION, "capture_gradients", tracking_capture)
    trunk_norm, predictor_norm, trunk_gradients, predictor_gradients = (
        UNION.capture_nextlat_gradient_groups(
            trunk,
            predictor,
            trunk_active=False,
            trunk_max_norm=0.025,
            predictor_max_norm=0.25,
        )
    )

    assert trunk_norm.item() == 0.0
    assert predictor_norm.item() == 5.0
    assert trunk_gradients == {}
    assert list(predictor_gradients) == predictor
    assert clip_calls == [(predictor, 0.25)]
    assert capture_calls == [predictor]


def test_breaker_predictor_only_aux_changes_no_trunk_or_head_parameter():
    agent = UNION.Agent(_DummyEnvs(), _args(UNION))
    task_optimizer = torch.optim.Adam(agent.task_parameters(), lr=0.02, eps=1e-5)
    predictor = agent.nextlat_predictor_parameters()
    predictor_optimizer = torch.optim.Adam(predictor, lr=0.02, eps=1e-5)
    heads = [agent.actor_alpha_head, agent.actor_beta_head]
    protected = agent.nextlat_trunk_parameters() + [
        parameter for head in heads for parameter in head.parameters()
    ]
    before_protected = [parameter.detach().clone() for parameter in protected]
    before_predictor = [parameter.detach().clone() for parameter in predictor]
    aux = {parameter: torch.ones_like(parameter) for parameter in predictor}

    UNION.apply_union_optimizer_step(
        agent.task_parameters(),
        task_optimizer,
        actor_gradients={},
        critic_gradients={},
        auxiliary_gradients={},
    )
    UNION.apply_private_predictor_step(predictor, predictor_optimizer, aux)

    assert all(
        torch.equal(parameter, before)
        for parameter, before in zip(protected, before_protected)
    )
    assert any(
        not torch.equal(parameter, before)
        for parameter, before in zip(predictor, before_predictor)
    )
    assert all(parameter not in task_optimizer.state for parameter in protected)


def test_breaker_detached_auxiliary_updates_predictor_without_trunk_gradient():
    agent = UNION.Agent(_DummyEnvs(), _args(UNION))
    trunk = agent.nextlat_trunk_parameters()
    predictor = agent.nextlat_predictor_parameters()
    predictor_optimizer = torch.optim.Adam(predictor, lr=0.02, eps=1e-5)
    source = agent.get_actor_feat(torch.randn(7, 7))
    action = torch.randn(7, 3)
    target = torch.randn(7, 8)
    before_predictor = [parameter.detach().clone() for parameter in predictor]

    detached_source = UNION.nextlat_source_feature(source, trunk_active=False)
    assert UNION.nextlat_source_feature(source, trunk_active=True) is source
    assert not detached_source.requires_grad
    assert detached_source.data_ptr() == source.data_ptr()
    prediction = agent.nextlat_predictor(torch.cat([detached_source, action], dim=-1))
    loss = torch.nn.functional.smooth_l1_loss(prediction, target)
    loss.backward()
    _, predictor_norm, trunk_gradients, predictor_gradients = (
        UNION.capture_nextlat_gradient_groups(
            trunk,
            predictor,
            trunk_active=False,
            trunk_max_norm=0.025,
            predictor_max_norm=0.25,
        )
    )
    UNION.apply_private_predictor_step(
        predictor,
        predictor_optimizer,
        predictor_gradients,
    )

    assert trunk_gradients == {}
    assert predictor_norm > 0.0
    assert all(parameter.grad is None for parameter in trunk)
    assert any(
        not torch.equal(parameter, before)
        for parameter, before in zip(predictor, before_predictor)
    )


def test_breaker_task_step_is_exact_critic_only_adam_oracle():
    initial = [torch.tensor([0.4, -0.3]), torch.tensor([0.2])]
    control = [torch.nn.Parameter(value.clone()) for value in initial]
    candidate = [torch.nn.Parameter(value.clone()) for value in initial]
    control_optimizer = torch.optim.Adam(control, lr=0.02, eps=1e-5)
    candidate_optimizer = torch.optim.Adam(candidate, lr=0.02, eps=1e-5)
    critic = [torch.tensor([-0.4, 0.5]), torch.tensor([-0.8])]

    for parameter, gradient in zip(control, critic):
        parameter.grad = gradient.clone()
    control_optimizer.step()
    UNION.apply_union_optimizer_step(
        candidate,
        candidate_optimizer,
        actor_gradients={},
        critic_gradients=dict(zip(candidate, critic)),
        auxiliary_gradients={},
    )

    for expected, actual in zip(control, candidate):
        assert torch.equal(actual, expected)
        expected_state = _clone_optimizer_state(control_optimizer, expected)
        actual_state = _clone_optimizer_state(candidate_optimizer, actual)
        for key in expected_state:
            if torch.is_tensor(expected_state[key]):
                assert torch.equal(actual_state[key], expected_state[key])
            else:
                assert actual_state[key] == expected_state[key]


def test_empty_gradient_delivery_does_not_advance_existing_adam_state():
    parameter = torch.nn.Parameter(torch.tensor([0.4, -0.3]))
    optimizer = torch.optim.Adam([parameter], lr=0.02, eps=1e-5)
    UNION.apply_union_optimizer_step(
        [parameter],
        optimizer,
        actor_gradients={parameter: torch.tensor([0.7, -0.2])},
        critic_gradients={},
        auxiliary_gradients={},
    )
    before = parameter.detach().clone()
    before_step = optimizer.state[parameter]["step"].detach().clone()

    UNION.apply_union_optimizer_step(
        [parameter],
        optimizer,
        actor_gradients={},
        critic_gradients={},
        auxiliary_gradients={},
    )

    assert torch.equal(parameter, before)
    assert torch.equal(optimizer.state[parameter]["step"], before_step)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_group_fails_before_shared_adam_step(bad):
    parameter = torch.nn.Parameter(torch.tensor([0.4, -0.3]))
    optimizer = torch.optim.Adam([parameter], lr=0.02)
    before = parameter.detach().clone()
    with pytest.raises(RuntimeError, match="non-finite"):
        UNION.apply_union_optimizer_step(
            [parameter],
            optimizer,
            actor_gradients={parameter: torch.tensor([bad, 0.1])},
            critic_gradients={},
            auxiliary_gradients={},
        )
    assert torch.equal(parameter, before)
    assert parameter not in optimizer.state


def test_merge_rejects_foreign_or_misaligned_gradients():
    parameter = torch.nn.Parameter(torch.zeros(2))
    foreign = torch.nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError, match="foreign"):
        UNION.merge_gradient_groups([parameter], {foreign: torch.ones(2)})
    with pytest.raises(ValueError, match="shape"):
        UNION.merge_gradient_groups([parameter], {parameter: torch.ones(3)})


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_hl_gauss_targets_and_retained_clone_stay_on_source_device(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(device_name)
    support = UNION.HLGaussSupport(
        7,
        -3.0,
        3.0,
        0.5,
        device,
        use_symlog=True,
        support_is_edges=True,
    )
    returns = torch.linspace(-5.0, 5.0, 24, device=device).reshape(2, 3, 4)
    projected = support.project(returns)
    retained = UNION.retain_graph_output(projected, compiled=True)

    assert projected.device.type == device.type
    assert retained.device.type == device.type
    assert retained.data_ptr() != projected.data_ptr()
    expected = projected.clone()
    projected.zero_()
    assert torch.equal(retained, expected)


def test_cuda_residency_compile_and_one_adam_source_wiring():
    source = Path(UNION.__file__).read_text()
    main = source[source.index('if __name__ == "__main__":') :]
    assert "hl_support_cpu" not in source
    assert "project_value_targets_fn(returns)" in main
    assert "b_target_probs[mb_inds]" in main
    assert "epoch_inds = torch.as_tensor(b_inds, device=device)" in main
    assert "torch.set_float32_matmul_precision(\"high\")" in main
    assert "fullgraph=True" in main
    assert "torch.compiler.cudagraph_mark_step_begin()" in main
    assert "retain_graph_output(" in main
    assert "dist.sample()" not in inspect.getsource(UNION.policy_model_forward)
    assert "action_value_from_policy_outputs(" in main
    assert "predictive_trunk_optimizer" not in main
    assert main.count("agent.task_parameters()") == 1
    assert "task_params = agent.task_parameters()" in main
    assert main.count("task_optimizer = optim.Adam(") == 1
    assert main.count("predictor_optimizer = optim.Adam(") == 1
    assert "apply_predictive_trunk_transaction(" not in source
    assert "apply_private_optimizer_step(" not in source


def test_training_source_uses_frozen_decoder_and_predictor_only_breaker_path():
    source = Path(UNION.__file__).read_text()
    main = source[source.index('if __name__ == "__main__":') :]
    teacher = main.index("target_dist, _, _ = agent._actor_dist_frozen_head(target_feat)")
    student = main.index("predicted_dist, _, _ = agent._actor_dist_frozen_head(h_hat)")
    detach = main.index("h_hat = nextlat_source_feature(")
    delivery = main.index("capture_nextlat_gradient_groups(", student)
    shared_step = main.index("apply_union_optimizer_step(", delivery)
    assert detach < teacher < student < delivery < shared_step
    assert "target_dist, _, _ = agent._actor_dist(target_feat)" not in main
    assert "predicted_dist, _, _ = agent._actor_dist(h_hat)" not in main
    assert "retain_graph=actor_active" in main
    assert "retain_graph=args.nextlat or actor_active" not in main
    assert "gate_nextlat_trunk_gradients(" not in source
    helper = inspect.getsource(UNION.capture_nextlat_gradient_groups)
    branch = helper[helper.index("if trunk_active:") : helper.index("predictor_norm =")]
    assert "clip_grad_norm_async_fail_loud_(" in branch
    assert "capture_gradients(trunk_parameters)" in branch
    assert "trunk_gradients = {}" in branch


def test_device_telemetry_has_one_packed_sync_and_no_minibatch_metric_item_reads():
    helper = inspect.getsource(UNION.synchronize_scalar_telemetry)
    source = Path(UNION.__file__).read_text()
    update = source[source.index("clipfrac_sum =") :]
    assert helper.count(".cpu()") == 1
    assert "clipfrac_sum.add_(" in update
    assert "clipfracs" not in update
    assert "latent_batch_std = frozen_actor_feats.std(dim=0).mean()" in update
    assert "adv_corr = (az * pz).mean()" in source
    # The sole epoch-level KL read is required for TPO's breaker control flow.
    assert update.count("epoch_mean_kl.item()") == 1
