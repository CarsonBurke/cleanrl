import torch

from cleanrl.plasticity.predictive_writer_v6 import PredictiveWriterState


def test_predictive_write_can_disagree_with_current_noisy_label():
    weights = [torch.zeros(1, 2, device="cuda") for _ in range(2)]
    states = [PredictiveWriterState(weight, control=control)
              for weight, control in zip(weights, ("unit", "raw_line"))]
    for state in states:
        state.memory.fill_(1)
        state.target_sum.fill_(100)
        state.curvature_sum.fill_(100)
        state.validation_target.fill_(100)
        state.validation_curvature.fill_(100)
        state.validation_square.fill_(100)
    # The recorded evidence favors +1. Today's observed label is -1; only
    # feature zero participates. The writer must follow evidence, not that label.
    gradient = torch.tensor([[1.0, 0.0]], device="cuda")
    curvature = gradient.square()
    for state, weight in zip(states, weights):
        torch.compile(state.step, fullgraph=True)(weight, gradient, curvature)
        assert weight[0, 1].item() == 0
    assert weights[0][0, 0].item() > 0.9
    assert weights[1][0, 0].item() == 0
    assert (weights[0][0, 0] - 1).square().item() < 0.01
    assert (weights[0][0, 0] + 1).square().item() > 1


def test_weak_positive_evidence_does_not_preserve_false_commitment():
    weights = [torch.zeros(1, 1, device="cuda") for _ in range(2)]
    states = [PredictiveWriterState(weight, control=control)
              for weight, control in zip(weights, ("unit", "zero_retirement"))]
    for state, weight in zip(states, weights):
        weight.fill_(0.5)
        state.memory.fill_(1)
        state.target_sum.fill_(1)
        state.curvature_sum.fill_(1)
        state.validation_target.fill_(0.25)
        state.validation_curvature.fill_(1)
        state.validation_square.fill_(1)
        state.coordinate_committed.fill_(True)
        state.step(weight, weight.clone(), torch.ones_like(weight))
    torch.testing.assert_close(weights[0], torch.zeros_like(weights[0]), rtol=0, atol=0)
    assert weights[1].item() > 0.1
