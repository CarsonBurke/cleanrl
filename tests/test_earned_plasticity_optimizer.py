import copy

import torch

from cleanrl.plasticity.earned_plasticity_optimizer_v1 import (
    EarnedPlasticity,
    EarnedPlasticityState,
)


def test_only_future_gradients_fund_the_previous_position():
    stake = 0.2
    weight = torch.zeros((), device="cuda")
    state = EarnedPlasticityState(weight, initial_stake=stake)
    state.step(weight, torch.ones_like(weight))
    torch.testing.assert_close(weight, torch.full_like(weight, -stake / 2))
    torch.testing.assert_close(state.wealth, torch.zeros_like(weight))
    state.step(weight, torch.ones_like(weight))
    torch.testing.assert_close(state.wealth, torch.full_like(weight, stake / 2))
    torch.testing.assert_close(weight, torch.full_like(weight, -stake))


def test_drawdown_shortens_history_after_a_budget_exceeding_gradient():
    stake = 0.2
    weights = [torch.zeros((), device="cuda") for _ in range(2)]
    states = [EarnedPlasticityState(weight, initial_stake=stake, control=control)
              for weight, control in zip(weights, ("funded", "persistent"))]
    for grad in (1.0, 1.0, 1.0, -10.0):
        for weight, state in zip(weights, states):
            state.step(weight, torch.full_like(weight, grad))
    retention = 11.5 / 30.25
    expected = stake * (10 - 3 * retention) / (20 + 3 * retention)
    torch.testing.assert_close(weights[0], torch.full_like(weights[0], expected))
    torch.testing.assert_close(weights[1], torch.full_like(weights[1], stake * 7 / 23))
    for state in states:
        torch.testing.assert_close(state.wealth, torch.zeros_like(state.wealth))


def test_constant_coordinate_gradient_rescaling_preserves_positions():
    weights = [torch.zeros(3, device="cuda") for _ in range(2)]
    states = [EarnedPlasticityState(weight, initial_stake=0.02) for weight in weights]
    scale = torch.tensor([1e-12, 7.0, 1e12], device="cuda")
    for values in ((1, -2, 3), (-3, 1, -0.5), (0, 3, -0.1), (4, -1, -2)):
        grad = torch.tensor(values, dtype=torch.float32, device="cuda")
        states[0].step(weights[0], grad)
        states[1].step(weights[1], grad * scale)
        torch.testing.assert_close(weights[0], weights[1], rtol=3e-6, atol=1e-8)


def test_compiled_zero_warmup_and_inactive_coordinates_preserve_positions():
    weight = torch.tensor([[0.25, -1.5]], device="cuda")
    state = EarnedPlasticityState(weight)
    step = torch.compile(state.step, fullgraph=True)
    step(weight, torch.zeros_like(weight))
    torch.testing.assert_close(weight, state.anchor, rtol=0, atol=0)
    for buffer in state.buffers():
        torch.testing.assert_close(buffer, torch.zeros_like(buffer), rtol=0, atol=0)
    step(weight, torch.tensor([[2.0, 0.0]], device="cuda"))
    old_weight = weight.clone()
    old_buffers = [buffer.clone() for buffer in state.buffers()]
    step(weight, torch.tensor([[0.0, -3.0]], device="cuda"))
    torch.testing.assert_close(weight[:, 0], old_weight[:, 0], rtol=0, atol=0)
    for buffer, old in zip(state.buffers(), old_buffers):
        torch.testing.assert_close(buffer[:, 0], old[:, 0], rtol=0, atol=0)
    assert weight[0, 1] > old_weight[0, 1]


def test_shuffled_credit_changes_funding_not_the_current_gradient():
    stake = 0.2
    weights = [torch.zeros(1, 2, device="cuda") for _ in range(2)]
    states = [EarnedPlasticityState(weight, initial_stake=stake, control=control)
              for weight, control in zip(weights, ("funded", "shuffled"))]
    for values in ((1.0, 0.0), (1.0, 1.0)):
        grad = torch.tensor([values], device="cuda")
        for weight, state in zip(weights, states):
            state.step(weight, grad)
    torch.testing.assert_close(weights[0], torch.tensor([[-stake, -stake / 2]], device="cuda"))
    torch.testing.assert_close(weights[1], torch.tensor([[-2 * stake / 3, -3 * stake / 4]], device="cuda"))


def test_fixed_capital_does_not_compound_profitable_positions():
    stake = 0.02
    weights = [torch.zeros((), device="cuda") for _ in range(2)]
    states = [EarnedPlasticityState(weight, initial_stake=stake, control=control)
              for weight, control in zip(weights, ("funded", "fixed"))]
    for _ in range(4):
        for weight, state in zip(weights, states):
            state.step(weight, torch.ones_like(weight))
    torch.testing.assert_close(weights[0], torch.full_like(weights[0], -3.5 * stake))
    torch.testing.assert_close(weights[1], torch.full_like(weights[1], -0.8 * stake))


def test_checkpoint_resume_preserves_future_learning():
    weight = torch.nn.Parameter(torch.tensor([0.1, -0.2], device="cuda"))
    optimizer = EarnedPlasticity([weight], initial_stake=0.1)
    for values in ((0.5, -0.75), (0.2, -0.4), (-0.2, 0.3)):
        weight.grad = weight.detach() - torch.tensor(values, device="cuda")
        optimizer.step()
    snapshot = copy.deepcopy(optimizer.state_dict())
    restored = torch.nn.Parameter(weight.detach().clone())
    resumed = EarnedPlasticity([restored], initial_stake=1.0)
    resumed.load_state_dict(snapshot)
    for values in ((0.8, 0.2), (-0.1, 0.6), (0.0, -0.3)):
        target = torch.tensor(values, device="cuda")
        weight.grad = weight.detach() - target
        restored.grad = restored.detach() - target
        optimizer.step()
        resumed.step()
        torch.testing.assert_close(restored, weight, rtol=0, atol=0)
