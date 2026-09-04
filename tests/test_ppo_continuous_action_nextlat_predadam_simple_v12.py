import importlib.util
from pathlib import Path

import gymnasium as gym
import pytest
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl" / "nextlat" / "ppo_continuous_action_nextlat_predadam_simple_v12.py"
)
SPEC = importlib.util.spec_from_file_location("nextlat_predadam_simple_v12", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _DummyEnvs:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(7,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))


def test_primary_gradient_ema_update_uses_only_primary_gradients():
    ema = [torch.zeros(2), torch.zeros(1)]
    first = [torch.tensor([2.0, -4.0]), torch.tensor([6.0])]
    second = [torch.tensor([-3.0, 5.0]), torch.tensor([1.0])]

    MODULE.update_primary_gradient_ema_(ema, first, decay=0.75)
    assert torch.allclose(ema[0], 0.25 * first[0])
    assert torch.allclose(ema[1], 0.25 * first[1])

    MODULE.update_primary_gradient_ema_(ema, second, decay=0.75)
    assert torch.allclose(ema[0], 0.75 * 0.25 * first[0] + 0.25 * second[0])
    assert torch.allclose(ema[1], 0.75 * 0.25 * first[1] + 0.25 * second[1])


def test_projection_is_orthogonal_within_each_logical_block():
    # The first logical block spans two tensors, so projection must use their joint dot.
    primary = [torch.tensor([20.0, 0.0]), torch.tensor([0.0]), torch.tensor([0.0, 30.0])]
    ema = [torch.tensor([1.0, 0.0]), torch.tensor([2.0]), torch.tensor([0.0, 1.0])]
    auxiliary = [torch.tensor([3.0, 4.0]), torch.tensor([-1.0]), torch.tensor([5.0, 6.0])]
    layout = MODULE.make_gradient_block_layout(
        [[primary[0], primary[1]], [primary[2]]]
    )

    admitted, stats = MODULE.project_and_cap_auxiliary_gradients(
        primary,
        ema,
        auxiliary,
        max_raw_gradient_ratio=1.0,
        layout=layout,
    )

    first_dot = torch.dot(ema[0], admitted[0]) + torch.dot(ema[1], admitted[1])
    second_dot = torch.dot(ema[2], admitted[2])
    assert abs(first_dot.item()) < 1e-6
    assert abs(second_dot.item()) < 1e-6
    assert stats["post_ema_dot_absmax"].item() < 1e-6


def test_auxiliary_gradient_has_local_and_global_primary_relative_caps():
    primary = [torch.tensor([1.0, 0.0]), torch.tensor([100.0, 0.0])]
    ema = [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])]
    auxiliary = [torch.tensor([0.0, 100.0]), torch.tensor([0.0, 100.0])]
    layout = MODULE.make_gradient_block_layout([[primary[0]], [primary[1]]])

    admitted, stats = MODULE.project_and_cap_auxiliary_gradients(
        primary,
        ema,
        auxiliary,
        max_raw_gradient_ratio=0.05,
        layout=layout,
    )

    assert admitted[0].norm() <= 0.05 * primary[0].norm() + 1e-7
    assert admitted[1].norm() <= 0.05 * primary[1].norm() + 1e-7
    total_primary = torch.cat(primary).norm()
    total_admitted = torch.cat(admitted).norm()
    assert total_admitted <= 0.05 * total_primary + 1e-6
    assert stats["max_raw_grad_block_ratio"] <= 0.05 + 1e-7


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_projection_is_scale_invariant_and_orthogonal_at_extreme_finite_scales(scale):
    base_primary = torch.tensor([2.0, -1.0, 0.5])
    base_ema = torch.tensor([1.0, 2.0, -1.0])
    base_auxiliary = torch.tensor([3.0, -4.0, 2.0])
    primary = [scale * base_primary]
    ema = [scale * base_ema]
    auxiliary = [scale * base_auxiliary]
    layout = MODULE.make_gradient_block_layout([[primary[0]]])

    admitted, stats = MODULE.project_and_cap_auxiliary_gradients(
        primary,
        ema,
        auxiliary,
        max_raw_gradient_ratio=0.25,
        layout=layout,
    )
    reference, reference_stats = MODULE.project_and_cap_auxiliary_gradients(
        [base_primary],
        [base_ema],
        [base_auxiliary],
        max_raw_gradient_ratio=0.25,
        layout=MODULE.make_gradient_block_layout([[base_primary]]),
    )

    assert stats["numeric_valid"].item() == 1.0
    assert torch.allclose(admitted[0] / scale, reference[0], rtol=2e-5, atol=2e-6)
    relative_dot = torch.dot(ema[0], admitted[0]) / (
        ema[0].norm() * admitted[0].norm()
    )
    assert abs(relative_dot.item()) < 2e-6
    assert stats["raw_ema_cosine"].item() == pytest.approx(
        reference_stats["raw_ema_cosine"].item(), rel=2e-6, abs=2e-6
    )


@pytest.mark.parametrize("ema_scale", [1e-12, 1.0, 1e12])
def test_projection_depends_on_ema_direction_not_ema_magnitude(ema_scale):
    primary = [torch.tensor([2.0, -1.0, 0.5])]
    base_ema = torch.tensor([1.0, 2.0, -1.0])
    auxiliary = [torch.tensor([3.0, -4.0, 2.0])]

    admitted, stats = MODULE.project_and_cap_auxiliary_gradients(
        primary,
        [ema_scale * base_ema],
        auxiliary,
        max_raw_gradient_ratio=0.25,
    )
    reference, _ = MODULE.project_and_cap_auxiliary_gradients(
        primary,
        [base_ema],
        auxiliary,
        max_raw_gradient_ratio=0.25,
    )

    assert stats["numeric_valid"].item() == 1.0
    assert torch.allclose(admitted[0], reference[0], rtol=2e-5, atol=2e-6)
    relative_dot = torch.dot(ema_scale * base_ema, admitted[0]) / (
        (ema_scale * base_ema).norm() * admitted[0].norm()
    )
    assert abs(relative_dot.item()) < 2e-6


class _EMARecordingSGD(torch.optim.SGD):
    def __init__(self, parameters, *, lr, ema_state, observed_ema):
        super().__init__(parameters, lr=lr)
        self.ema_state = ema_state
        self.observed_ema = observed_ema

    @torch.no_grad()
    def step(self, closure=None):
        self.observed_ema.append(
            (
                self.ema_state.initialized,
                [gradient.detach().clone() for gradient in self.ema_state.gradients],
            )
        )
        return super().step(closure)


def test_optimizer_transaction_uses_strictly_lagged_ema_and_cold_starts_task_only():
    trunk = torch.nn.Parameter(torch.zeros(2))
    predictor = torch.nn.Parameter(torch.zeros(1))
    ema_state = MODULE.SimplePCGradientEMAState([torch.zeros_like(trunk)])
    observed_ema = []
    task_optimizer = _EMARecordingSGD(
        [trunk], lr=1.0, ema_state=ema_state, observed_ema=observed_ema
    )
    predictor_optimizer = torch.optim.SGD([predictor], lr=1.0)
    layout = MODULE.make_gradient_block_layout([[trunk]])

    first_primary = [torch.tensor([1.0, 0.0])]
    first_auxiliary = [torch.tensor([0.0, 10.0])]
    first_stats, first_predictor_step, first_trunk_step = (
        MODULE.apply_simplepc_optimizer_transaction(
            trunk_parameters=[trunk],
            predictor_parameters=[predictor],
            task_optimizer=task_optimizer,
            predictor_optimizer=predictor_optimizer,
            primary_trunk_gradients=first_primary,
            actor_trunk_gradients=first_primary,
            critic_trunk_gradients=[torch.zeros(2)],
            auxiliary_trunk_gradients=first_auxiliary,
            predictor_gradients=[torch.tensor([2.0])],
            primary_gradient_ema=ema_state,
            ema_decay=0.0,
            max_raw_gradient_ratio=1.0,
            layout=layout,
        )
    )

    # Cold start is task-only on the shared trunk, while the private predictor still learns.
    assert torch.equal(trunk, torch.tensor([-1.0, 0.0]))
    assert torch.equal(predictor, torch.tensor([-2.0]))
    assert first_stats["admitted_norm"].item() == 0.0
    assert first_stats["ema_was_initialized"].item() == 0.0
    assert first_predictor_step.item() == pytest.approx(2.0)
    assert first_trunk_step.item() == pytest.approx(1.0)
    # The task optimizer observed the uninitialized EMA; only afterward was it seeded.
    assert observed_ema[0][0] is False
    assert torch.equal(observed_ema[0][1][0], torch.zeros(2))
    assert ema_state.initialized is True
    assert torch.equal(ema_state.gradients[0], first_primary[0])

    second_primary = [torch.tensor([0.0, 2.0])]
    second_auxiliary = [torch.tensor([3.0, 4.0])]
    second_stats, _, second_trunk_step = MODULE.apply_simplepc_optimizer_transaction(
        trunk_parameters=[trunk],
        predictor_parameters=[predictor],
        task_optimizer=task_optimizer,
        predictor_optimizer=predictor_optimizer,
        primary_trunk_gradients=second_primary,
        actor_trunk_gradients=[torch.tensor([0.0, 1.0])],
        critic_trunk_gradients=[torch.tensor([0.0, 1.0])],
        auxiliary_trunk_gradients=second_auxiliary,
        predictor_gradients=[torch.zeros(1)],
        primary_gradient_ema=ema_state,
        ema_decay=0.0,
        max_raw_gradient_ratio=1.0,
        layout=layout,
    )

    # Projecting [3, 4] against the strictly lagged [1, 0] leaves [0, 4], then
    # the raw-gradient cap scales it to [0, 2]. With current primary [0, 2], SGD
    # therefore applies [0, 4]. A same-minibatch EMA leak would produce an x step.
    assert torch.equal(trunk, torch.tensor([-1.0, -4.0]))
    assert second_stats["ema_was_initialized"].item() == 1.0
    assert second_stats["actor_admitted_dot"].item() == pytest.approx(2.0)
    assert second_stats["critic_admitted_dot"].item() == pytest.approx(2.0)
    assert second_stats["actor_admitted_cosine"].item() == pytest.approx(1.0)
    assert second_stats["critic_admitted_cosine"].item() == pytest.approx(1.0)
    assert second_trunk_step.item() == pytest.approx(4.0)
    assert observed_ema[1][0] is True
    assert torch.equal(observed_ema[1][1][0], first_primary[0])
    assert torch.equal(ema_state.gradients[0], second_primary[0])


def _warm_optimizer(parameter, optimizer, gradient):
    optimizer.zero_grad(set_to_none=True)
    parameter.grad = gradient.clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _assert_floating_optimizer_state_is_finite(optimizer):
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and (
                torch.is_floating_point(value) or torch.is_complex(value)
            ):
                assert torch.isfinite(value).all()


@pytest.mark.parametrize("max_raw_gradient_ratio", [0.0, 0.05])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), 1e30])
def test_invalid_trunk_auxiliary_is_exact_task_only_with_existing_adam_momentum(
    bad_value, max_raw_gradient_ratio
):
    trunk = torch.nn.Parameter(torch.tensor([0.4, -0.2]))
    task_only_trunk = torch.nn.Parameter(trunk.detach().clone())
    predictor = torch.nn.Parameter(torch.tensor([0.3]))
    task_optimizer = torch.optim.Adam([trunk], lr=3e-3, eps=1e-5)
    task_only_optimizer = torch.optim.Adam(
        [task_only_trunk], lr=3e-3, eps=1e-5
    )
    predictor_optimizer = torch.optim.Adam([predictor], lr=3e-3, eps=1e-5)

    warm_task_gradient = torch.tensor([0.7, -0.4])
    _warm_optimizer(trunk, task_optimizer, warm_task_gradient)
    _warm_optimizer(task_only_trunk, task_only_optimizer, warm_task_gradient)
    _warm_optimizer(predictor, predictor_optimizer, torch.tensor([0.2]))

    primary = [torch.tensor([0.15, -0.35])]
    trunk.grad = primary[0].clone()
    task_only_trunk.grad = primary[0].clone()
    task_only_optimizer.step()
    task_only_optimizer.zero_grad(set_to_none=True)

    initial_ema = torch.tensor([0.25, -0.5])
    ema_state = MODULE.SimplePCGradientEMAState(
        [initial_ema.clone()], initialized=True
    )
    layout = MODULE.make_gradient_block_layout([[trunk]])
    stats, _, _ = MODULE.apply_simplepc_optimizer_transaction(
        trunk_parameters=[trunk],
        predictor_parameters=[predictor],
        task_optimizer=task_optimizer,
        predictor_optimizer=predictor_optimizer,
        primary_trunk_gradients=primary,
        actor_trunk_gradients=[primary[0].clone()],
        critic_trunk_gradients=[torch.zeros_like(primary[0])],
        auxiliary_trunk_gradients=[torch.tensor([bad_value, -bad_value])],
        predictor_gradients=[torch.tensor([0.1])],
        primary_gradient_ema=ema_state,
        ema_decay=0.5,
        max_raw_gradient_ratio=max_raw_gradient_ratio,
        layout=layout,
    )

    assert torch.equal(trunk, task_only_trunk)
    for key, expected in task_only_optimizer.state[task_only_trunk].items():
        actual = task_optimizer.state[trunk][key]
        assert torch.equal(actual, expected)
    assert stats["numeric_valid"].item() == 0.0
    assert stats["admitted_norm"].item() == 0.0
    assert all(torch.isfinite(value).all() for value in stats.values())
    assert torch.equal(ema_state.gradients[0], 0.5 * initial_ema + 0.5 * primary[0])
    _assert_floating_optimizer_state_is_finite(task_optimizer)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), 1e30])
def test_invalid_private_auxiliary_restores_parameters_and_keeps_adam_finite(
    bad_value,
):
    predictor = torch.nn.Parameter(torch.tensor([0.4, -0.2]))
    optimizer = torch.optim.Adam([predictor], lr=1e-2, eps=1e-5)
    _warm_optimizer(predictor, optimizer, torch.tensor([0.3, -0.1]))
    parameter_before = predictor.detach().clone()
    step_before = optimizer.state[predictor]["step"].item()

    step_norm = MODULE.apply_private_optimizer_step(
        [predictor], optimizer, [torch.tensor([bad_value, -bad_value])]
    )

    assert torch.equal(predictor, parameter_before)
    assert step_norm.item() == 0.0
    assert optimizer.state[predictor]["step"].item() == step_before + 1
    _assert_floating_optimizer_state_is_finite(optimizer)


def test_private_auxiliary_repairs_prepoisoned_moment_and_rejects_transaction():
    predictor = torch.nn.Parameter(torch.tensor([0.4, -0.2]))
    optimizer = torch.optim.Adam([predictor], lr=1e-2, eps=1e-5)
    _warm_optimizer(predictor, optimizer, torch.tensor([0.3, -0.1]))
    optimizer.state[predictor]["exp_avg"][0] = float("inf")
    parameter_before = predictor.detach().clone()
    step_before = optimizer.state[predictor]["step"].item()

    step_norm = MODULE.apply_private_optimizer_step(
        [predictor], optimizer, [torch.tensor([0.2, 0.1])]
    )

    assert torch.equal(predictor, parameter_before)
    assert step_norm.item() == 0.0
    assert optimizer.state[predictor]["step"].item() == step_before + 1
    _assert_floating_optimizer_state_is_finite(optimizer)


def test_normal_transaction_never_scans_or_repairs_task_optimizer_state(monkeypatch):
    trunk = torch.nn.Parameter(torch.tensor([0.4, -0.2]))
    predictor = torch.nn.Parameter(torch.tensor([0.3]))
    task_optimizer = torch.optim.Adam([trunk], lr=3e-3, eps=1e-5)
    predictor_optimizer = torch.optim.Adam([predictor], lr=3e-3, eps=1e-5)
    _warm_optimizer(trunk, task_optimizer, torch.tensor([0.7, -0.4]))
    _warm_optimizer(predictor, predictor_optimizer, torch.tensor([0.2]))
    original_repair = MODULE._repair_optimizer_state_
    repaired_optimizers = []

    def recording_repair(optimizer, reference):
        repaired_optimizers.append(optimizer)
        assert optimizer is not task_optimizer
        return original_repair(optimizer, reference)

    monkeypatch.setattr(MODULE, "_repair_optimizer_state_", recording_repair)
    primary = [torch.tensor([0.15, -0.35])]
    trunk.grad = primary[0].clone()
    MODULE.apply_simplepc_optimizer_transaction(
        trunk_parameters=[trunk],
        predictor_parameters=[predictor],
        task_optimizer=task_optimizer,
        predictor_optimizer=predictor_optimizer,
        primary_trunk_gradients=primary,
        actor_trunk_gradients=[primary[0].clone()],
        critic_trunk_gradients=[torch.zeros_like(primary[0])],
        auxiliary_trunk_gradients=[torch.tensor([0.2, 0.1])],
        predictor_gradients=[torch.tensor([0.1])],
        primary_gradient_ema=MODULE.SimplePCGradientEMAState(
            [torch.tensor([0.25, -0.5])], initialized=True
        ),
        ema_decay=0.5,
        max_raw_gradient_ratio=0.05,
        layout=MODULE.make_gradient_block_layout([[trunk]]),
    )

    assert repaired_optimizers == [predictor_optimizer, predictor_optimizer]


def test_runtime_task_clips_fail_explicitly_on_nonfinite_gradients():
    source = SCRIPT.read_text()
    critic_clip = source.index(
        "critic_params,\n                    args.critic_grad_clip,\n"
        "                    error_if_nonfinite=True,"
    )
    actor_clip = source.index(
        "actor_params,\n                    args.actor_grad_clip,\n"
        "                    error_if_nonfinite=True,"
    )
    transaction = source.index("apply_simplepc_optimizer_transaction(", actor_clip)

    assert critic_clip < actor_clip < transaction


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_policy_space_auxiliary_freezes_decoder_weights(actor_dist):
    args = MODULE.Args(
        hidden=8,
        k_blocks=1,
        n_experts=2,
        critic_mtp_horizon=2,
        num_bins=5,
        actor_dist=actor_dist,
    )
    agent = MODULE.Agent(_DummyEnvs(), args)
    source = torch.randn(4, args.hidden, requires_grad=True)
    target = torch.randn(4, args.hidden)

    with torch.no_grad():
        teacher, _, _ = agent._actor_dist(target)
    student, _, _ = agent._actor_dist_frozen_head(source)
    torch.distributions.kl_divergence(teacher, student).sum().backward()

    if actor_dist == "beta":
        heads = (agent.actor_alpha_head, agent.actor_beta_head)
    else:
        heads = (agent.actor_head, agent.actor_logvar_head)
    assert source.grad is not None
    assert source.grad.norm().item() > 0.0
    assert all(parameter.grad is None for head in heads for parameter in head.parameters())


def test_optimizer_partitions_have_one_trunk_owner_and_private_predictor():
    args = MODULE.Args(hidden=8, k_blocks=2, n_experts=2, critic_mtp_horizon=2, num_bins=5)
    agent = MODULE.Agent(_DummyEnvs(), args)
    task_ids = {id(parameter) for parameter in agent.task_parameters()}
    trunk_ids = {id(parameter) for parameter in agent.nextlat_trunk_parameters()}
    predictor_ids = {id(parameter) for parameter in agent.nextlat_predictor_parameters()}
    auxiliary_ids = {id(parameter) for parameter in agent.nextlat_parameters()}
    block_parameters = [
        parameter
        for block in agent.nextlat_trunk_parameter_blocks()
        for parameter in block
    ]
    block_ids = [id(parameter) for parameter in block_parameters]
    policy_head_ids = {
        id(parameter)
        for head in (agent.actor_alpha_head, agent.actor_beta_head)
        for parameter in head.parameters()
    }

    assert trunk_ids <= task_ids
    assert predictor_ids.isdisjoint(task_ids)
    assert predictor_ids.isdisjoint(trunk_ids)
    assert auxiliary_ids == trunk_ids | predictor_ids
    assert auxiliary_ids.isdisjoint(policy_head_ids)
    assert set(block_ids) == trunk_ids
    assert len(block_ids) == len(set(block_ids))
