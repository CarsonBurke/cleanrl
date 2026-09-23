"""CUDA equivalence contracts for v10's execution-only v9 optimization.

Run through mlq. Fixed SIGReg directions couple the stochastic realization for
numerical comparisons; production resampling and projection defaults are intact.
"""

from dataclasses import asdict
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from cleanrl import ppo_continuous_action_jepa_intact_fused_wml_v10 as fused
from cleanrl import ppo_continuous_action_jepa_intact_quotient_wml_v9 as baseline
from cleanrl.shared.lejepa import SIGReg
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(autouse=True)
def cuda_runtime():
    assert torch.cuda.is_available(), "Submit CUDA contracts through mlq"
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)


class FixedSIGReg(SIGReg):
    """Hold random directions fixed to compare the same stochastic objective."""

    def __init__(self):
        super().__init__(knots=17, num_proj=16, proj_chunk=8)
        directions = torch.randn(64, self.num_proj, device="cuda")
        self.register_buffer("directions", directions / directions.norm(dim=0))

    def forward(self, proj):
        total = proj.new_zeros(())
        for chunk in self.directions.split(self.proj_chunk, dim=1):
            total = total + self._statistic(proj, chunk).sum()
        return total / (self.num_proj * proj.size(0))


@pytest.fixture
def agents_and_args():
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1, 1, (6,), dtype=np.float32),
    )
    args = fused.Args(sigreg_num_proj=16, sigreg_proj_chunk=8)
    old = baseline.Agent(envs, args).cuda()
    new = fused.Agent(envs, args).cuda()
    old.sigreg = FixedSIGReg().cuda()
    new.sigreg = FixedSIGReg().cuda()
    new.load_state_dict(old.state_dict())
    return old, new, args


def rollout(agent, rows=64):
    observations = torch.randn(rows, agent.input_dim, device="cuda")
    native = torch.rand(rows, agent.action_dim, device="cuda") * 0.8 + 0.1
    weights, _ = fused.factual_goal_weights(torch.randn(rows, device="cuda"), 0.1, "weighted")
    returns, values, rewards = [torch.randn(rows, device="cuda") for _ in range(3)]
    following, goals = torch.randn_like(observations), torch.randn_like(observations)
    terms = (torch.arange(rows, device="cuda") % 7 == 0).float()
    return observations, native, weights, returns, values, following, goals, rewards, terms


def optimizers(agent, args):
    world, actor, critic = agent.parameter_groups()
    return (
        torch.optim.AdamW(world, lr=args.ssl_learning_rate, weight_decay=args.ssl_weight_decay, fused=True),
        torch.optim.Adam(actor, lr=args.learning_rate, eps=1e-5, fused=True),
        torch.optim.Adam(critic, lr=args.learning_rate, eps=1e-5, fused=True),
    )


def test_algorithm_schedule_and_projection_defaults_unchanged():
    expected, actual = asdict(baseline.Args()), asdict(fused.Args())
    expected.pop("exp_name")
    actual.pop("exp_name")
    assert actual == expected
    assert actual["sigreg_num_proj"] == 1024
    assert actual["sigreg_proj_chunk"] == 256


def test_indexed_loss_metrics_and_all_gradients_match(agents_and_args):
    old, new, args = agents_and_args
    batch = rollout(old)
    # Reordered and repeated indices verify true advanced-indexing semantics.
    indices = torch.tensor([47, 0, 12, 12, 3, 60, 2, 20, 35, 29, 10, 1, 44, 6, 16, 31], device="cuda")
    expected_loss, expected_metrics = baseline.training_loss(old, *(field[indices] for field in batch), args)
    actual_loss, actual_metrics = fused.indexed_training_loss(new, indices, batch, args)
    assert set(expected_metrics) == set(fused.METRIC_NAMES)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(actual_metrics, torch.stack([expected_metrics[name] for name in fused.METRIC_NAMES]), rtol=0, atol=0)
    expected_loss.backward()
    actual_loss.backward()
    for (old_name, old_parameter), (new_name, new_parameter) in zip(old.named_parameters(), new.named_parameters()):
        assert old_name == new_name
        assert old_parameter.grad is not None and new_parameter.grad is not None
        torch.testing.assert_close(new_parameter.grad, old_parameter.grad, rtol=0, atol=0)


@pytest.mark.parametrize("max_norm", [0.5, 1e-7])
def test_compiled_clipping_matches_independent_torch_groups(max_norm):
    compiled = torch.compile(fused.clip_gradient_groups, fullgraph=True, dynamic=False,
                             options={"triton.cudagraphs": False})
    # Different norms must produce independent clipping factors. A tiny bound
    # makes PyTorch's 1e-6 denominator behavior materially visible as well.
    for iteration in range(3):
        gradients = tuple(
            tuple((torch.randn(*shape, device="cuda") * scale).t()
                  for shape in ((7, 3), (4, 5)))
            for scale in (0.0, 1e-8 * (iteration + 1), 3.0 * (iteration + 1))
        )
        parameters = tuple(tuple(nn.Parameter(torch.zeros_like(g)) for g in group) for group in gradients)
        for group, gradient_group in zip(parameters, gradients):
            for parameter, gradient in zip(group, gradient_group):
                parameter.grad = gradient.clone()
        expected = torch.stack([nn.utils.clip_grad_norm_(group, max_norm) for group in parameters])
        actual = compiled(gradients, max_norm)
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=1e-12)
        for group, parameter_group in zip(gradients, parameters):
            for actual_gradient, parameter in zip(group, parameter_group):
                torch.testing.assert_close(actual_gradient, parameter.grad, rtol=3e-6, atol=1e-12)


def test_compiled_minibatches_updates_and_owned_metric_accumulation_match(agents_and_args):
    old, new, args = agents_and_args
    batch = rollout(old)
    old_optimizers, new_optimizers = optimizers(old, args), optimizers(new, args)
    compiled_loss = torch.compile(lambda indices, data: fused.indexed_training_loss(new, indices, data, args),
                                  fullgraph=True, dynamic=False, mode="reduce-overhead")
    compiled_clip = torch.compile(fused.clip_gradient_groups, fullgraph=True, dynamic=False,
                                  options={"triton.cudagraphs": False})
    value_peer = torch.compile(new.get_value, fullgraph=True, dynamic=True,
                               options={"triton.cudagraphs": False})
    expected_sums = torch.zeros(len(fused.METRIC_NAMES), device="cuda")
    actual_sums = torch.zeros_like(expected_sums)
    stored_norms = torch.empty(4, 3, device="cuda")
    expected_norms = torch.empty_like(stored_norms)
    for step in range(4):
        if step == 2:
            # A new rollout changes every tensor's storage, including the large
            # observations. Detect stale CUDA graph inputs across rollout boundaries.
            batch = rollout(old)
            batch[0].add_(1.0)
            batch[1].mul_(-1).add_(1.0)
            batch[3].add_(5.0)
            batch[7].add_(7.0)
            with torch.no_grad():
                peer_values = value_peer(batch[0]).clone()
                torch.testing.assert_close(peer_values, old.get_value(batch[0]), rtol=3e-4, atol=3e-5)
            del peer_values
        indices = (torch.arange(16, device="cuda") * 3 + step * 11) % 64
        for optimizer in (*old_optimizers, *new_optimizers):
            optimizer.zero_grad(set_to_none=True)
        expected_loss, expected_metrics = baseline.training_loss(old, *(field[indices] for field in batch), args)
        expected_loss.backward()
        for column, parameters in enumerate(old.parameter_groups()):
            expected_norms[step, column] = nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
        for optimizer in old_optimizers:
            optimizer.step()
        expected_sums.add_(torch.stack([expected_metrics[name] for name in fused.METRIC_NAMES]))

        torch.compiler.cudagraph_mark_step_begin()
        actual_loss, actual_metrics = compiled_loss(indices, batch)
        actual_loss.backward()
        gradients = tuple(tuple(parameter.grad for parameter in group if parameter.grad is not None)
                          for group in new.parameter_groups())
        stored_norms[step].copy_(compiled_clip(gradients, args.max_grad_norm))
        for optimizer in new_optimizers:
            optimizer.step()
        actual_sums.add_(actual_metrics.detach())
        del actual_loss, actual_metrics, gradients
    torch.testing.assert_close(actual_sums, expected_sums, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(stored_norms, expected_norms, rtol=4e-4, atol=3e-5)
    for actual_parameter, expected_parameter in zip(new.parameters(), old.parameters()):
        torch.testing.assert_close(actual_parameter, expected_parameter, rtol=3e-4, atol=3e-5)
    # Fused Adam/AdamW moments and step counts are part of execution equivalence.
    for actual_optimizer, expected_optimizer in zip(new_optimizers, old_optimizers):
        actual_states = list(actual_optimizer.state.values())
        expected_states = list(expected_optimizer.state.values())
        assert len(actual_states) == len(expected_states)
        for actual_state, expected_state in zip(actual_states, expected_states):
            assert actual_state.keys() == expected_state.keys()
            for key in actual_state:
                torch.testing.assert_close(actual_state[key], expected_state[key], rtol=7e-4, atol=3e-6)
