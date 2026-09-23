import torch

from cleanrl.plasticity.predictive_directions_v4 import PredictiveDirectionsState


def test_compiled_admission_writes_full_signal_without_learning_rate_tuning():
    weight = torch.zeros(1, 2, device="cuda")
    state = PredictiveDirectionsState(weight)
    step = torch.compile(state.step, fullgraph=True)
    step(weight, torch.zeros_like(weight), torch.zeros_like(weight))
    largest_change = torch.zeros((), device="cuda")
    for _ in range(10):
        before = weight.clone()
        step(weight, torch.cat((weight[:, :1] - 1, torch.zeros_like(weight[:, 1:])), dim=1),
             torch.tensor([[1.0, 0.0]], device="cuda"))
        largest_change = torch.maximum(largest_change, (weight - before).abs().max())
    torch.testing.assert_close(weight, torch.tensor([[1.0, 0.0]], device="cuda"))
    assert largest_change.item() > 0.9


def test_coordinate_evidence_can_reopen_after_long_contradictory_history():
    weight = torch.zeros(1, 1, device="cuda")
    state = PredictiveDirectionsState(weight, control="coordinate")
    step = torch.compile(state.step, fullgraph=True)
    for target in (1.0, -1.0) * 500:
        step(weight, weight - target, torch.ones_like(weight))
    torch.testing.assert_close(weight, torch.zeros_like(weight), rtol=0, atol=0)
    for _ in range(320):
        step(weight, weight - 1, torch.ones_like(weight))
    assert weight.item() > 0.8
    for _ in range(320):
        step(weight, weight, torch.ones_like(weight))
    assert weight.abs().item() < 0.1


def test_block_forecast_calibration_is_invariant_to_memory_magnitude():
    weights = [torch.zeros(1, 2, device="cuda") for _ in range(2)]
    states = [PredictiveDirectionsState(weight, control="block") for weight in weights]
    for state, scale in zip(states, (1.0, 1000.0)):
        state.slow_memory.fill_(scale)
        state.slow_target.fill_(scale)
        state.slow_curvature.fill_(1)
        state.block_target.fill_(100)
        state.block_curvature.fill_(100)
        state.block_square.fill_(100)
    for state, weight in zip(states, weights):
        state.step(weight, -torch.ones_like(weight), torch.ones_like(weight))
    torch.testing.assert_close(weights[0], weights[1], rtol=0, atol=0)
    torch.testing.assert_close(weights[0].sum(-1), torch.ones(1, device="cuda"))
