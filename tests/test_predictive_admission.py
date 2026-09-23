import torch

from cleanrl.plasticity.predictive_admission_v3 import PredictiveAdmissionState


def test_compiled_signal_admission_produces_a_unit_jump_without_early_noise_writes():
    weight = torch.zeros(1, 2, device="cuda")
    state = PredictiveAdmissionState(weight)
    step = torch.compile(state.step, fullgraph=True)
    step(weight, torch.zeros_like(weight), torch.zeros_like(weight))
    changes = []
    for _ in range(10):
        before = weight.clone()
        step(weight, torch.cat((weight[:, :1] - 1, torch.zeros_like(weight[:, 1:])), dim=1),
             torch.tensor([[1.0, 0.0]], device="cuda"))
        changes.append(weight - before)
    torch.testing.assert_close(changes[0], torch.zeros_like(weight), rtol=0, atol=0)
    torch.testing.assert_close(weight, torch.tensor([[1.0, 0.0]], device="cuda"))
    assert torch.stack(changes).abs().max().item() > 0.9


def test_block_readout_uses_joint_forecast_curvature():
    weight = torch.zeros(1, 2, device="cuda")
    state = PredictiveAdmissionState(weight, control="block")
    for _ in range(8):
        residual = weight.sum(-1, keepdim=True) - 2
        state.step(weight, residual.expand_as(weight), torch.ones_like(weight))
    torch.testing.assert_close(weight.sum(-1), torch.tensor([2.0], device="cuda"))
    torch.testing.assert_close(state.allocation, torch.full_like(weight, 0.5))


def test_consistently_contradicted_memory_is_never_admitted():
    weight = torch.zeros(1, 1, device="cuda")
    state = PredictiveAdmissionState(weight, control="coordinate")
    for target in (1.0, -1.0) * 20:
        state.step(weight, weight - target, torch.ones_like(weight))
        torch.testing.assert_close(weight, torch.zeros_like(weight), rtol=0, atol=0)
