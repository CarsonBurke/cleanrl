import math

import torch

from cleanrl.plasticity.predictive_commitment_v5 import PredictiveCommitmentState


def test_discovery_threshold_failure_is_not_retirement_without_counterevidence():
    weights = [torch.zeros(1, 1, device="cuda") for _ in range(2)]
    states = [PredictiveCommitmentState(weight, control=control)
              for weight, control in zip(weights, ("coordinate", "recheck"))]
    steps = [torch.compile(state.step, fullgraph=True) for state in states]
    for target in [1.0] * 10 + [-4.0]:
        for step, weight in zip(steps, weights):
            step(weight, weight - target, torch.ones_like(weight))
    assert weights[0].item() > 0.1
    torch.testing.assert_close(weights[1], torch.zeros_like(weights[1]), rtol=0, atol=0)
    for step, weight in zip(steps, weights):
        step(weight, weight + 50, torch.ones_like(weight))
        torch.testing.assert_close(weight, torch.zeros_like(weight), rtol=0, atol=0)


def test_validated_block_is_not_replaced_by_unproven_local_improvement():
    weights = [torch.zeros(1, 1, device="cuda") for _ in range(2)]
    states = [PredictiveCommitmentState(weight, control=control)
              for weight, control in zip(weights, ("coarse_first", "fine_first"))]
    for state, weight in zip(states, weights):
        state.memory.fill_(1)
        state.target_sum.fill_(1)
        state.curvature_sum.fill_(1)
        state.slow_memory.fill_(1)
        state.slow_target.fill_(1)
        state.slow_curvature.fill_(1)
        state.validation_target.fill_(80)
        state.validation_curvature.fill_(100)
        state.validation_square.fill_(100)
        state.block_target.fill_(20)
        state.block_curvature.fill_(100)
        state.block_square.fill_(1)
        state.step(weight, -torch.ones_like(weight), torch.ones_like(weight))
    torch.testing.assert_close(weights[0], torch.full_like(weights[0], 21 / 101))
    decay = math.exp(-1 / 128)
    torch.testing.assert_close(weights[1], torch.full_like(weights[1], (80 * decay + 1) / (100 * decay + 1)))
