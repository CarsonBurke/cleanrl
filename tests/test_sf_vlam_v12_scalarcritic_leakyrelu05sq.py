import importlib.util
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).parents[1]


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V9 = _load(
    "sf_vlam_v9_for_scalarcritic_test",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v9_leakyrelu05sq.py",
)
MODULE = _load(
    "sf_vlam_v12_scalarcritic_leakyrelu05sq",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v12_scalarcritic_leakyrelu05sq.py",
)


class DummyEnv:
    single_observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
    )
    single_action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(6,), dtype=np.float32
    )


def test_critic_is_one_scalar_per_state():
    agent = MODULE.Agent(DummyEnv(), MODULE.Args()).cuda()
    values = agent.get_value(torch.randn(23, 17, device="cuda"))
    assert values.shape == (23,)
    assert agent.critic_head.weight.shape == (1, 64)


def test_scalar_gae_uses_rewards_and_cuts_trace_at_boundaries():
    rewards = torch.tensor([[1.0], [2.0], [3.0]], device="cuda")
    values = torch.tensor([[0.5], [1.0], [1.5]], device="cuda")
    next_values = torch.tensor([[1.0], [9.0], [2.0]], device="cuda")
    terminations = torch.tensor([[0.0], [0.0], [1.0]], device="cuda")
    boundaries = torch.tensor([[0.0], [1.0], [1.0]], device="cuda")
    valids = torch.ones_like(rewards)

    advantages, returns = MODULE.compute_scalar_gae(
        rewards,
        values,
        next_values,
        terminations,
        boundaries,
        valids,
        gamma=0.9,
        gae_lambda=0.8,
    )

    delta2 = 3.0 - 1.5
    delta1 = 2.0 + 0.9 * 9.0 - 1.0
    delta0 = 1.0 + 0.9 * 1.0 - 0.5
    expected = torch.tensor(
        [[delta0 + 0.9 * 0.8 * delta1], [delta1], [delta2]],
        device="cuda",
    )
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected + values)

    # Missing final observations suppress a truncation bootstrap.
    valids[1] = 0.0
    invalid_advantages, _ = MODULE.compute_scalar_gae(
        rewards,
        values,
        next_values,
        terminations,
        boundaries,
        valids,
        gamma=0.9,
        gae_lambda=0.8,
    )
    torch.testing.assert_close(invalid_advantages[1], torch.tensor([1.0], device="cuda"))


def test_trunk_and_actor_initialization_are_exactly_seed_paired_to_v9():
    torch.manual_seed(17)
    reference = V9.Agent(DummyEnv(), V9.Args())
    expected_rng_state = torch.get_rng_state()

    torch.manual_seed(17)
    scalar = MODULE.Agent(DummyEnv(), MODULE.Args())
    actual_rng_state = torch.get_rng_state()

    torch.testing.assert_close(actual_rng_state, expected_rng_state, atol=0, rtol=0)
    for name, tensor in reference.trunk.state_dict().items():
        torch.testing.assert_close(tensor, scalar.trunk.state_dict()[name], atol=0, rtol=0)
    for head_name in ("actor_alpha_head", "actor_beta_head"):
        reference_head = getattr(reference, head_name)
        scalar_head = getattr(scalar, head_name)
        for expected, actual in zip(
            reference_head.parameters(), scalar_head.parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_parameter_delta_is_only_the_smaller_critic_head():
    torch.manual_seed(3)
    reference = V9.Agent(DummyEnv(), V9.Args())
    torch.manual_seed(3)
    scalar = MODULE.Agent(DummyEnv(), MODULE.Args())
    reference_count = sum(parameter.numel() for parameter in reference.parameters())
    scalar_count = sum(parameter.numel() for parameter in scalar.parameters())
    expected_removed = (6 * (32 + 17 + 2 * 6 + 1) - 1) * 64
    assert reference_count - scalar_count == expected_removed == 23_744


def test_compiled_scalar_actor_critic_forward_and_backward_are_finite():
    agent = MODULE.Agent(DummyEnv(), MODULE.Args()).cuda()
    agent.trunk = MODULE.CompiledModule(
        agent.trunk, mode="reduce-overhead", cudagraphs=False
    )
    observations = torch.randn(128, 17, device="cuda", requires_grad=True)
    _, _, logprob, entropy, values = agent.get_action_and_value(observations)
    targets = torch.randn_like(values)
    loss = F.mse_loss(values, targets) - 0.01 * logprob.mean() - 0.01 * entropy.mean()
    loss.backward()

    assert values.shape == (128,)
    assert torch.isfinite(loss)
    assert torch.isfinite(observations.grad).all()
    assert agent.critic_head.weight.grad is not None
    assert torch.isfinite(agent.critic_head.weight.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in agent.actor_alpha_head.parameters()
    )


if __name__ == "__main__":
    test_critic_is_one_scalar_per_state()
    test_scalar_gae_uses_rewards_and_cuts_trace_at_boundaries()
    test_trunk_and_actor_initialization_are_exactly_seed_paired_to_v9()
    test_parameter_delta_is_only_the_smaller_critic_head()
    test_compiled_scalar_actor_critic_forward_and_backward_are_finite()
