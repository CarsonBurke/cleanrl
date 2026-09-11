"""Independent Peri-actor/pre-RMS-critic contracts; CUDA checks run via mlq only."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_normres_scalar_peri_no_final_v8 as baseline
from cleanrl import ppo_continuous_action_peri_actor_precritic_v17 as trainer
from test_ppo_normres_twohot import device
from cleanrl.shared.host_actor import SiTUGLUBranch
from cleanrl.shared.norm_residual import NormResidualTrunk

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="queued CUDA test required"),
]


def _agent(module, device):
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), np.float32),
        single_action_space=gym.spaces.Box(
            np.array([-3.0, 1.0], np.float32), np.array([2.0, 5.0], np.float32)
        ),
    )
    torch.manual_seed(1)
    with torch.device(device):
        return module.Agent(spaces)


def _reference_network(network, observations, *, normalize_branch_outputs):
    """Write the scalar-network equations without calling its trunk or branches."""
    trunk, head = network
    h = F.linear(observations, trunk.in_proj.weight, trunk.in_proj.bias)
    for block, gate in zip(trunk.blocks, trunk.block_gates):
        z = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-5)
        gate_input = F.linear(z, block.gate.weight)
        up_input = F.linear(z, block.up.weight)
        hidden = (
            4.0 * torch.tanh(gate_input / 4.0) * torch.sigmoid(gate_input)
            * 25.0 * torch.tanh(up_input / 25.0)
        )
        branch = F.linear(hidden, block.down.weight)
        if normalize_branch_outputs:
            branch = branch * torch.rsqrt(branch.square().mean(-1, keepdim=True) + 1e-5)
        h = h + torch.sigmoid(gate) * branch
    return F.linear(h / 8.0, head.weight, head.bias)


def test_seed_one_preserves_actor_and_shared_critic_initialization(device):
    control = _agent(baseline, device)
    candidate = _agent(trainer, device)
    # Fixed, non-affine RMS removal must not shift the actor's RNG stream or
    # remove any trainable critic parameter: this is a topology-only ablation.
    for name in ("actor", "critic"):
        original = dict(getattr(control, name).named_parameters())
        changed = dict(getattr(candidate, name).named_parameters())
        assert original.keys() == changed.keys()
        for key, parameter in original.items():
            torch.testing.assert_close(changed[key], parameter, rtol=0, atol=0)
    observations = torch.randn(19, 17, device=device)
    with torch.no_grad():
        alpha, beta, _ = candidate.get_policy_and_value(observations)
        control_alpha, control_beta, _ = control.get_policy_and_value(observations)
        torch.testing.assert_close(alpha, control_alpha, rtol=0, atol=0)
        torch.testing.assert_close(beta, control_beta, rtol=0, atol=0)


def test_critic_keeps_branch_magnitude_while_actor_stays_unchanged(device):
    control = _agent(baseline, device).double()
    candidate = _agent(trainer, device).double()
    candidate_trunk, control_trunk = candidate.critic[0], control.critic[0]
    assert isinstance(candidate_trunk, NormResidualTrunk)
    assert isinstance(control_trunk, NormResidualTrunk)
    candidate_branch, control_branch = candidate_trunk.blocks[-1], control_trunk.blocks[-1]
    assert isinstance(candidate_branch, SiTUGLUBranch)
    assert isinstance(control_branch, SiTUGLUBranch)
    observations = torch.randn(23, 17, device=device, dtype=torch.float64)
    with torch.no_grad():
        alpha, beta, before = candidate.get_policy_and_value(observations)
        control_before = control.get_value(observations)
        # Scale the last branch so upstream inputs are identical before/after.
        # Peri's RMS is only approximately scale-invariant because eps != 0.
        candidate_branch.down.weight.mul_(3.0)
        control_branch.down.weight.mul_(3.0)
        alpha_after, beta_after, after = candidate.get_policy_and_value(observations)
        torch.testing.assert_close(alpha_after, alpha, rtol=0, atol=0)
        torch.testing.assert_close(beta_after, beta, rtol=0, atol=0)
        changed = (after - before).norm()
        control_changed = (control.get_value(observations) - control_before).norm()
        assert changed > 1e-3
        assert changed > 100.0 * control_changed
        # Moving from 1x to 5x doubles the value change of 1x to 3x, rather
        # than having RMS cancel the output weight's positive scale.
        candidate_branch.down.weight.mul_(5.0 / 3.0)
        second_after = candidate.get_value(observations)
        torch.testing.assert_close(second_after - before, 2.0 * (after - before), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("network_name", ["actor", "critic"])
def test_network_outputs_and_gradients_match_independent_definition(device, network_name):
    agent = _agent(trainer, device).double()
    network = getattr(agent, network_name)
    observations = torch.randn(11, 17, device=device, dtype=torch.float64, requires_grad=True)
    actual = network(observations)
    expected = _reference_network(
        network, observations, normalize_branch_outputs=network_name == "actor"
    )
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    probe = torch.linspace(-0.7, 1.3, actual.numel(), device=device, dtype=torch.float64).reshape_as(actual)
    differentiated = (observations, *network.parameters())
    actual_gradients = torch.autograd.grad((actual * probe).sum(), differentiated)
    expected_gradients = torch.autograd.grad((expected * probe).sum(), differentiated)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-9, atol=1e-11)


def test_compiled_clipped_ppo_loss_has_finite_nonzero_critic_gradients(device):
    agent = _agent(trainer, device)
    observations = torch.randn(24, 17, device=device)
    native_actions = torch.linspace(0.05, 0.95, 48, device=device).reshape(24, 2)
    with torch.no_grad():
        alpha, beta, old_values = agent.get_policy_and_value(observations)
        old_values = old_values.flatten()
        old_logprobs = agent.action_logprob(alpha, beta, native_actions)
        old_logprobs -= torch.tensor([0.0, 0.6, -0.6], device=device).repeat(8)
    advantages = torch.tensor([2.0, -3.0, 7.0, -1.0], device=device).repeat(6)
    targets = old_values + torch.linspace(-2.0, 3.0, 24, device=device)
    args = trainer.Args()
    args.clip_vloss = True
    torch.compiler.reset()
    try:
        def loss_fn(obs, actions, logprobs, adv, returns, values):
            return trainer.ppo_loss(agent, obs, actions, logprobs, adv, returns, values, args)

        compiled_loss = torch.compile(loss_fn, fullgraph=True, mode="reduce-overhead")
        loss, metrics = compiled_loss(observations, native_actions, old_logprobs, advantages, targets, old_values)
        assert torch.isfinite(loss)
        assert torch.isfinite(metrics).all()
        loss.backward()
        for parameter in agent.critic.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.norm() > 0
    finally:
        torch.compiler.reset()
