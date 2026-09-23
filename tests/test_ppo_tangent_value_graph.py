"""Queue manual value-graph correctness and synchronization contracts through mlq."""

from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from cleanrl.ppo_continuous_action_tangent_joint_graph_v9 import (
    Agent, Args, GraphValueUpdate, TangentValueAdam, validate_args,
)
from test_ppo_normres_twohot import device


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

ACTUAL_GAIN, ORIGIN_LOSS, SELECTED_LOSS = 0, 7, 8


@pytest.fixture(autouse=True)
def isolated_runtime(device):
    torch._dynamo.reset()
    yield device
    torch._dynamo.reset()


def assert_optimizer_matches(critic, optimizer, reference_critic, reference, *, rtol, atol):
    for parameter, expected in zip(critic.parameters(), reference_critic.parameters()):
        torch.testing.assert_close(parameter, expected, rtol=rtol, atol=atol)
        for name in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.state[parameter][name], reference.state[expected][name],
                rtol=0 if name == "step" else rtol, atol=0 if name == "step" else atol,
            )


def test_capture_preserves_cold_and_warm_state_and_observes_each_real_batch_once(device):
    critic = nn.Linear(3, 1, device=device, dtype=torch.float64)
    reference_critic = deepcopy(critic)
    betas, vf_coef = (0.6, 0.8), 0.5
    optimizer = TangentValueAdam(critic, betas=betas, vf_coef=vf_coef)
    reference = TangentValueAdam(reference_critic, betas=betas, vf_coef=vf_coef)
    base = torch.tensor(
        [[1.0, -0.5, 0.25], [-0.75, 1.0, 0.5], [0.5, 0.25, -1.0],
         [-0.5, -1.0, 0.75], [0.25, 0.5, 1.0], [1.0, 0.75, -0.5]],
        device=device, dtype=torch.float64,
    )

    for index, shift in enumerate((0.75, -1.0, 1.5, -0.5, 0.25, -1.25)):
        observations = base.roll(index, 0) * (1 + 0.1 * index)
        targets = shift + (0.3 + 0.2 * index) * observations[:, 0] - 0.7 * observations[:, 2]
        if index in (0, 3):
            # Capture both an untouched optimizer and a genuinely trained one.
            # Check public training state exactly, not graph implementation details.
            mutable = tuple(critic.parameters()) + tuple(
                value for parameter in critic.parameters() for value in optimizer.state[parameter].values())
            before_capture = tuple(value.detach().clone() for value in mutable)
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
            graph = GraphValueUpdate(optimizer, observations, targets)
            for value, saved in zip(mutable, before_capture):
                torch.testing.assert_close(value, saved, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
            torch.testing.assert_close(torch.cuda.get_rng_state(device), cuda_rng, rtol=0, atol=0)

        if index == 0:
            loss = 0.5 * vf_coef * (critic(observations).flatten() - targets).square().mean()
            first_gradients = torch.autograd.grad(loss, tuple(critic.parameters()))
        expected = reference.update(observations, targets).clone()
        result = graph.update(observations, targets).clone()
        torch.testing.assert_close(result, expected, rtol=1e-9, atol=1e-11)
        assert_optimizer_matches(critic, optimizer, reference_critic, reference, rtol=1e-9, atol=1e-11)
        for parameter in critic.parameters():
            assert optimizer.state[parameter]["step"].item() == index + 1
        if index == 0:
            for parameter, gradient in zip(critic.parameters(), first_gradients):
                torch.testing.assert_close(
                    optimizer.state[parameter]["exp_avg"], (1 - betas[0]) * gradient,
                    rtol=1e-9, atol=1e-11,
                )
                torch.testing.assert_close(
                    optimizer.state[parameter]["exp_avg_sq"], (1 - betas[1]) * gradient.square(),
                    rtol=1e-9, atol=1e-11,
                )


def test_actual_critic_replays_changing_batches_with_meaningful_metrics_and_no_host_sync(device):
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), np.float32),
    )
    with torch.device(device):
        critic = Agent(spaces).critic
    reference_critic = deepcopy(critic)
    optimizer = TangentValueAdam(critic)
    reference = TangentValueAdam(reference_critic)
    observations = torch.randn((64, 17), device=device)
    # Real calls must replace both captured inputs, even on the first replay.
    graph = GraphValueUpdate(optimizer, torch.zeros_like(observations), torch.zeros(64, device=device))
    shifts = (0.75, -0.5, 1.25, -1.0, 0.25, -0.75)
    staged = torch.empty((len(shifts), 10), device=device)
    expected_metrics = torch.empty_like(staged)

    for index, shift in enumerate(shifts):
        current_observations = observations.roll(index, 0) * (1 + 0.03 * index) + 0.02 * index
        targets = 0.7 * current_observations[:, 0].sin() + 0.3 * current_observations[:, 1] + shift
        with torch.no_grad():
            old = critic(current_observations).flatten()
        expected_metrics[index].copy_(reference.update(current_observations, targets))
        previous_sync_mode = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            with torch._dynamo.config.patch(error_on_recompile=True):
                staged[index].copy_(graph.update(current_observations, targets))
        finally:
            torch.cuda.set_sync_debug_mode(previous_sync_mode)

        with torch.no_grad():
            new = critic(current_observations).flatten()
        result = staged[index]
        assert torch.isfinite(result).all()
        residual, change = old.double() - targets.double(), old.double() - new.double()
        actual_gain = 0.5 * (change * (residual - 0.5 * change)).mean()
        tolerance = 2e-5 * max(1.0, residual.square().mean().item())
        assert actual_gain.item() > 0
        torch.testing.assert_close(result[ACTUAL_GAIN].double(), actual_gain, rtol=3e-3, atol=tolerance)
        torch.testing.assert_close(result[ORIGIN_LOSS], 0.5 * (old - targets).square().mean(),
                                   rtol=3e-4, atol=2e-6)
        torch.testing.assert_close(result[SELECTED_LOSS], 0.5 * (new - targets).square().mean(),
                                   rtol=3e-4, atol=2e-6)
        assert_optimizer_matches(critic, optimizer, reference_critic, reference, rtol=2e-5, atol=2e-6)

    # Earlier metrics must survive subsequent replays when consumed as in training.
    torch.testing.assert_close(staged, expected_metrics, rtol=2e-5, atol=2e-6)


def test_rejects_uncompiled_value_graph_instead_of_silently_changing_execution(device):
    with pytest.raises(ValueError, match="compile"):
        validate_args(Args(compile=False))
    critic = nn.Linear(1, 1, device=device)
    optimizer = TangentValueAdam(critic, compile=False)
    with pytest.raises(ValueError, match="compiled"):
        GraphValueUpdate(optimizer, torch.zeros((4, 1), device=device), torch.zeros(4, device=device))
