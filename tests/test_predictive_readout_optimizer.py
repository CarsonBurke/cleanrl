import torch

from cleanrl.plasticity.predictive_readout_optimizer_v1 import PredictiveReadoutState


def test_first_observation_cannot_validate_its_own_memory_then_signal_writes_fully():
    weight = torch.zeros(1, 1, device="cuda")
    state = PredictiveReadoutState(weight)
    state.step(weight, weight - 1, torch.ones_like(weight))
    torch.testing.assert_close(weight, torch.zeros_like(weight), rtol=0, atol=0)
    state.step(weight, weight - 1, torch.ones_like(weight))
    torch.testing.assert_close(weight, torch.ones_like(weight))
    torch.testing.assert_close(state.allocation, torch.ones_like(weight))


def test_later_contradiction_retracts_old_live_memory_without_small_steps():
    weight = torch.zeros(1, 1, device="cuda")
    state = PredictiveReadoutState(weight)
    for target in (1.0, 1.0):
        state.step(weight, weight - target, torch.ones_like(weight))
    state.step(weight, weight + 1, torch.ones_like(weight))
    torch.testing.assert_close(weight, torch.zeros_like(weight))
    torch.testing.assert_close(state.allocation, torch.zeros_like(weight))
    assert state.memory.item() > 0


def test_joint_candidate_line_search_prevents_collinear_overshoot():
    weight = torch.zeros(1, 2, device="cuda")
    state = PredictiveReadoutState(weight)
    for _ in range(2):
        residual = weight.sum(-1, keepdim=True) - 2
        state.step(weight, residual.expand_as(weight), torch.ones_like(weight))
    torch.testing.assert_close(weight, torch.ones_like(weight))
    torch.testing.assert_close(weight.sum(-1), torch.tensor([2.0], device="cuda"))


def test_compiled_warmup_and_exact_fit_cannot_create_unjustified_update():
    weight = torch.zeros(1, 2, device="cuda")
    state = PredictiveReadoutState(weight)
    step = torch.compile(state.step, fullgraph=True)
    step(weight, torch.zeros_like(weight), torch.zeros_like(weight))
    for buffer in state.buffers():
        torch.testing.assert_close(buffer, torch.zeros_like(buffer), rtol=0, atol=0)
    for _ in range(2):
        step(weight, torch.tensor([[-1.0, 0.0]], device="cuda"), torch.tensor([[1.0, 0.0]], device="cuda"))
    before = weight.clone()
    step(weight, torch.zeros_like(weight), torch.ones_like(weight))
    torch.testing.assert_close(weight, before, rtol=0, atol=0)
