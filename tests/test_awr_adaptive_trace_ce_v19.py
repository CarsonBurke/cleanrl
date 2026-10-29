"""TraceSelector (awr v19): batched fold gradients, cross-fold signal and the view choice."""

import types

import gymnasium as gym
import numpy as np
import torch

from cleanrl.awr import ppo_continuous_action_awr_adaptive_trace_ce_v19 as v19


def make_agent():
    envs = types.SimpleNamespace(
        single_action_space=gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (5,), np.float32),
    )
    torch.manual_seed(0)
    return v19.Agent(envs).double()


def batch(agent, size):
    observations = torch.randn(size, 5, dtype=torch.float64)
    with torch.no_grad():
        alpha, beta = agent.policy(observations)
        native = torch.distributions.Beta(alpha, beta).sample().clamp(1e-4, 1 - 1e-4)
    return observations, native


def fold_gradients_by_loop(agent, observations, native, views, num_envs, folds):
    fold = (torch.arange(observations.shape[0]) % num_envs) % folds
    z = (views - views.mean(1, keepdim=True)) / views.std(1, keepdim=True)
    params = tuple(agent.actor.parameters())
    out = torch.empty(views.shape[0], folds, sum(p.numel() for p in params), dtype=torch.float64)
    for v in range(views.shape[0]):
        for f in range(folds):
            keep = fold == f
            alpha, beta = agent.policy(observations[keep])
            loss = (z[v, keep] * agent.action_logprob(alpha, beta, native[keep])).mean()
            out[v, f] = torch.cat([g.flatten() for g in torch.autograd.grad(loss, params)])
    return out


def test_signal_and_energy_match_brute_force():
    agent = make_agent()
    num_envs, folds, size = 8, 4, 64
    observations, native = batch(agent, size)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)  # empty state: P = I
    selector = v19.TraceSelector(agent, optimizer, num_envs, size, folds, decay=0.0, device="cpu")
    selector.signal, selector.energy = selector.signal.double(), selector.energy.double()
    views = torch.randn(len(v19.TRACE_LAMBDAS), size, dtype=torch.float64)
    index, diagnostics = selector(observations, native, views)

    g = fold_gradients_by_loop(agent, observations, native, views, num_envs, folds)
    signal = torch.stack([sum(g[v, f] @ g[-1, h] for f in range(folds) for h in range(folds) if f != h)
                          / (folds * (folds - 1)) for v in range(len(v19.TRACE_LAMBDAS))])
    energy = g.mean(1).pow(2).sum(-1)
    torch.testing.assert_close(selector.signal, signal)
    torch.testing.assert_close(selector.energy, energy)
    assert int(index) == int((signal / energy.sqrt()).argmax())
    assert float(diagnostics["trace/lambda"]) == v19.TRACE_LAMBDAS[int(index)]


def test_prefers_the_view_aligned_with_monte_carlo():
    """Noise-dominated views of one fold-generalizing signal: the EMA picks the least noisy view."""
    agent = make_agent()
    num_envs, folds, size = 16, 4, 4096
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    selector = v19.TraceSelector(agent, optimizer, num_envs, size, folds, decay=0.9, device="cpu")
    selector.signal, selector.energy = selector.signal.double(), selector.energy.double()
    generator = torch.Generator().manual_seed(1)
    # Views: the truth plus independent noise, least at index 2, most for Monte Carlo (the last).
    scales = (6.0, 4.0, 2.0, 4.0, 6.0, 8.0, 12.0)
    for _ in range(30):
        observations, native = batch(agent, size)
        # Signal that generalizes across folds: a fixed function of (state, action).
        truth = (native[:, 0] - 0.5) * (1.0 + observations[:, 0].tanh()) + native[:, 1]
        truth = (truth - truth.mean()) / truth.std()
        views = torch.stack([truth + s * torch.randn(size, generator=generator, dtype=torch.float64)
                             for s in scales])
        index, _ = selector(observations, native, views)
    assert int(index) == 2
