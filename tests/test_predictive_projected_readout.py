import torch

from cleanrl.plasticity.predictive_projected_readout_v2 import ProjectedReadoutState


def test_a_current_observation_cannot_write_its_unobserved_coordinate():
    weight = torch.zeros(1, 2, device="cuda")
    state = ProjectedReadoutState(weight, control="open")
    state.step(weight, -torch.ones_like(weight), torch.ones_like(weight))
    torch.testing.assert_close(weight, torch.tensor([[0.5, 0.5]], device="cuda"))
    state.step(weight, torch.tensor([[-2.5, 0.0]], device="cuda"),
               torch.tensor([[1.0, 0.0]], device="cuda"))
    torch.testing.assert_close(weight, torch.tensor([[2.0, 0.5]], device="cuda"))


def test_compiled_validated_signal_can_enter_and_retract_in_unit_jumps():
    weight = torch.zeros(1, 1, device="cuda")
    state = ProjectedReadoutState(weight)
    step = torch.compile(state.step, fullgraph=True)
    step(weight, torch.zeros_like(weight), torch.zeros_like(weight))
    step(weight, weight - 1, torch.ones_like(weight))
    torch.testing.assert_close(weight, torch.zeros_like(weight), rtol=0, atol=0)
    step(weight, weight - 1, torch.ones_like(weight))
    torch.testing.assert_close(weight, torch.ones_like(weight))
    step(weight, weight + 1, torch.ones_like(weight))
    torch.testing.assert_close(weight, torch.zeros_like(weight))


def test_projected_write_never_increases_current_half_squared_loss():
    weight = torch.zeros(1, 3, device="cuda")
    state = ProjectedReadoutState(weight)
    examples = [([1, -2, 0], 3), ([2, 1, 1], -1), ([0, 1, -1], 2),
                ([1, 0, 0], -3), ([1, 1, 1], 1), ([0, 0, 0], 2)]
    for values, target in examples:
        x = torch.tensor([values], dtype=torch.float32, device="cuda")
        residual = (weight * x).sum(-1, keepdim=True) - target
        state.step(weight, residual * x, x.square())
        new_residual = (weight * x).sum(-1, keepdim=True) - target
        assert bool((new_residual.square() <= residual.square() + 1e-6).all())
