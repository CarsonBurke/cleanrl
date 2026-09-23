"""The rollout behavior policy must match the CUDA learner, including refreshes."""
import numpy as np
import pytest
import torch

from cleanrl.ppo_continuous_action_residual_activation_v1 import ResidualHostMirror, ResidualMLP
from cleanrl.shared.runtime import configure_runtime


@pytest.mark.parametrize("activation", ["silu", "tanh", "stiglu", "relusq", "leakyrelusq"])
def test_residual_behavior_matches_cuda_before_and_after_update(activation):
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    actor = ResidualMLP(17, 12, activation, output_std=0.01).cuda()
    mirror = ResidualHostMirror(actor, 16)
    observations = np.random.default_rng(1).normal(size=(16, 17)).astype(np.float32)
    device_observations = torch.as_tensor(observations, device="cuda")
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    for _ in range(2):
        mirror.refresh()
        with torch.no_grad():
            expected = actor(device_observations).cpu().numpy()
        actual = mirror(observations).copy()
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-6)
        # A different call must not corrupt the stored skip input or future calls.
        mirror(-observations)
        np.testing.assert_allclose(mirror(observations), expected, rtol=2e-4, atol=2e-6)
        optimizer.zero_grad(set_to_none=True)
        (actor(device_observations) - 1.0).square().mean().backward()
        optimizer.step()
