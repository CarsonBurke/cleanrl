"""HL-Gauss frame transitions, CE gradients, and real PPO/host integration."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_scaled_asymclip_v7 import Agent as ScalarAgent
from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_raw_hlgauss_v11 import (
    Agent, Args, ResidualHostMirror, make_env, ppo_loss,
)
from cleanrl.shared.runtime import configure_runtime


def test_wide_support_decodes_raw_expectation_not_inverse_transformed_mean():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
    )
    agent = Agent(spaces).cuda()
    with torch.no_grad():
        # Equal mass at 0 and +20k must predict 10k, not symexp(mean(symlog)).
        agent.critic.head[0].weight.zero_()
        agent.critic.head[0].bias.fill_(-1000.0)
        agent.critic.head[0].bias[255] = 0.0
        agent.critic.head[0].bias[-1] = 0.0
        agent.critic.readout_gain.fill_(1.0)
        x = torch.zeros(4, 17, device="cuda")
        value = torch.compile(agent.get_value, fullgraph=True, options={"triton.cudagraphs": False})(x)
        torch.testing.assert_close(value, torch.full((4, 1), 10000.0, device="cuda"), rtol=2e-6, atol=0.01)
        torch.testing.assert_close(agent.get_policy_and_value(x)[2], value)


def test_public_environment_preserves_native_reward_sequence():
    raw = gym.make("HalfCheetah-v4")
    wrapped = make_env("HalfCheetah-v4", 0, False, "raw_reward_contract", 0.99)()
    try:
        raw.reset(seed=1)
        wrapped.reset(seed=1)
        generator = np.random.default_rng(1)
        for _ in range(16):
            action = generator.uniform(-1, 1, 6).astype(np.float32)
            _, expected, terminated, truncated, _ = raw.step(action)
            _, actual, actual_terminated, actual_truncated, _ = wrapped.step(action)
            assert actual == expected
            assert (actual_terminated, actual_truncated) == (terminated, truncated)
    finally:
        wrapped.close()
        raw.close()


def test_compiled_ppo_updates_preserve_actor_initialization_and_host_policy_parity():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
    )
    observations = np.random.default_rng(1).normal(size=(16, 17)).astype(np.float32)
    x = torch.as_tensor(observations, device="cuda")
    torch.manual_seed(1)
    scalar = ScalarAgent(spaces).cuda()
    scalar.normalize_matrices()
    with torch.no_grad():
        initial_policy = scalar.actor(x).clone()
    del scalar
    torch.manual_seed(1)
    args = Args()
    agent = Agent(spaces, args).cuda()
    agent.normalize_matrices()
    with torch.no_grad():
        torch.testing.assert_close(agent.actor(x), initial_policy, atol=0, rtol=0)
    mirror = ResidualHostMirror(agent.actor, 16)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.0024, eps=1e-5, fused=True)
    loss_fn = torch.compile(ppo_loss, fullgraph=True, options={"triton.cudagraphs": False})
    value_fn = torch.compile(agent.get_value, fullgraph=True, options={"triton.cudagraphs": False})
    native = torch.full((16, 6), 0.5, device="cuda")
    for shift in (0.0, 2000.0):
        with torch.no_grad():
            alpha, beta, old_values = agent.get_policy_and_value(x)
            old_logprobs = agent.action_logprob(alpha, beta, native)
            returns = old_values.flatten() + torch.linspace(-500.0, 1500.0, 16, device="cuda") + shift
            advantages = returns - old_values.flatten()
            labels = agent.histogram.project(returns)
        for _ in range(2):
            loss, metrics = loss_fn(agent, x, native, old_logprobs, advantages, labels, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for parameter in agent.parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            # A small, nonzero categorical gain must let the critic trunk learn immediately.
            assert agent.critic.first[0].gate.weight.grad.abs().sum() > 0
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            agent.normalize_matrices()
        mirror.refresh()
        with torch.no_grad():
            expected = agent.actor(x).cpu().numpy()
            # Compare against an FP64 raw expectation. A reduction over +/-20k
            # needs a raw-unit tolerance: 0.002 is about one FP32 support ULP.
            logits = agent.critic(x).double()
            reference = (logits.softmax(-1) * agent.histogram.support.double()).sum(-1, keepdim=True).float()
            torch.testing.assert_close(value_fn(x), reference, rtol=2e-4, atol=0.002)
            torch.testing.assert_close(agent.get_policy_and_value(x)[2], reference, rtol=2e-4, atol=0.002)
            assert torch.isfinite(metrics).all()
        np.testing.assert_allclose(mirror(observations), expected, rtol=2e-4, atol=2e-5)
