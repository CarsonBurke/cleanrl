"""HL-Gauss frame transitions, CE gradients, and real PPO/host integration."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_scaled_asymclip_v7 import Agent as ScalarAgent
from cleanrl.ppo_continuous_action_residual_stiglu_ngpt_kragg_hlgauss_v9 import (
    Agent, Args, KraggHistogram, ResidualHostMirror, categorical_value_loss, ppo_loss,
)
from cleanrl.shared.runtime import configure_runtime


def test_current_rollout_frame_has_no_lag_and_projection_keeps_reflected_mass():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    histogram = KraggHistogram().cuda()
    observe = torch.compile(histogram.observe, fullgraph=True, options={"triton.cudagraphs": False})
    decode = torch.compile(histogram.decode, fullgraph=True, options={"triton.cudagraphs": False})
    targets = torch.linspace(-2.0, 2.0, 65, device="cuda")
    logits = torch.linspace(-1.0, 1.0, 255, device="cuda").expand(65, -1)
    observe(targets)
    labels = histogram.project(targets)
    before = decode(logits).clone()
    observe(3.0 * targets + 10.0)
    torch.testing.assert_close(decode(logits), 3.0 * before + 10.0)
    torch.testing.assert_close(histogram.project(3.0 * targets + 10.0), labels, atol=2e-6, rtol=2e-5)
    # Endpoint clipping and finite Gaussian tails must retain normalized mass.
    extremes = histogram.mean + histogram.scale * torch.tensor([-100.0, 0.0, 100.0], device="cuda")
    probs = histogram.project(extremes)
    torch.testing.assert_close(probs.sum(-1), torch.ones(3, device="cuda"))
    torch.testing.assert_close(probs[0], probs[2].flip(0), atol=1e-7, rtol=1e-5)
    observe(torch.full_like(targets, 25.0))
    torch.testing.assert_close(decode(torch.zeros_like(logits)), torch.full_like(targets, 25.0))


def test_calibrated_ce_uses_detached_current_variance_and_detached_labels():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    histogram = KraggHistogram().cuda()
    targets = torch.linspace(-1.0, 2.0, 16, device="cuda", requires_grad=True)
    histogram.observe(targets)
    labels = histogram.project(targets)
    logits = torch.randn(16, 255, device="cuda", requires_grad=True)
    with torch.no_grad():
        probs = logits.softmax(-1)
        mean = (probs * histogram.centers).sum(-1, keepdim=True)
        variance = (probs * (histogram.centers - mean).square()).sum(-1)
        gain = histogram.scale.square() * variance.mean()
        expected_gradient = (probs - labels) * gain / 16
    loss_fn = torch.compile(categorical_value_loss, fullgraph=True, options={"triton.cudagraphs": False})
    loss_fn(histogram, logits, labels).backward()
    torch.testing.assert_close(logits.grad, expected_gradient, atol=2e-7, rtol=2e-5)
    assert targets.grad is None


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
    for shift in (0.0, 2.0):
        with torch.no_grad():
            alpha, beta, old_values = agent.get_policy_and_value(x)
            old_logprobs = agent.action_logprob(alpha, beta, native)
            returns = old_values.flatten() + torch.linspace(-0.5, 1.5, 16, device="cuda") + shift
            advantages = returns - old_values.flatten()
            agent.histogram.observe(returns)
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
            torch.testing.assert_close(value_fn(x), agent.get_policy_and_value(x)[2])
            assert torch.isfinite(metrics).all()
        np.testing.assert_allclose(mirror(observations), expected, rtol=2e-4, atol=2e-5)
