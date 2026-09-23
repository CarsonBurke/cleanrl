"""RFO adaptation invariants. CUDA-only; execute via the shared GPU queue."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from cleanrl import ppo_continuous_action_rfo_world_model_v1 as rfo
from cleanrl.shared.ppo_loop import compute_gae_from_next_values

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def deterministic_runtime():
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(91)
            yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def environments(obs_dim=1, action_dim=1):
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
        single_action_space=gym.spaces.Box(
            low=np.full(action_dim, -2.0, dtype=np.float32),
            high=np.full(action_dim, 3.0, dtype=np.float32),
        ),
    )


def make_agent(obs_dim=1, action_dim=1):
    with torch.device("cuda"):
        return rfo.Agent(environments(obs_dim, action_dim), rfo.Args(num_envs=2))


class AnalyticWorld(nn.Module):
    """Known smooth dynamics let finite differences detect any BPTT detach."""
    def __init__(self, reward_weight):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.3, device="cuda"))
        self.reward_weight = reward_weight

    def forward(self, states, actions):
        next_states = 0.7 * states + self.gain * actions
        reward = self.reward_weight * (states[:, 0] + actions[:, 0])
        # State-dependent survival also has a real gradient through earlier actions.
        termination_logits = next_states[:, 0] - 2.0
        return next_states, reward, termination_logits


@pytest.mark.parametrize("reward_weight,value_weight", [(1.0, 0.0), (0.0, 1.0)])
def test_rpg_bptt_matches_finite_difference_and_freezes_world_and_critics(reward_weight, value_weight):
    agent = make_agent()
    with torch.device("cuda"):
        agent.actor = nn.Linear(3, 1)
        agent.critics = nn.ModuleList([nn.Linear(1, 1), nn.Linear(1, 1)])
    with torch.no_grad():
        agent.actor.weight.copy_(torch.tensor([[0.2, 0.1, -0.05]], device="cuda"))
        agent.actor.bias.fill_(0.03)
        for index, critic in enumerate(agent.critics):
            critic.weight.fill_(value_weight * (1.0 + index))
            critic.bias.zero_()
    world = AnalyticWorld(reward_weight)
    raw = torch.tensor([[0.2], [-0.1]], device="cuda")
    mean, std = torch.zeros_like(raw), torch.ones_like(raw)
    reward_std = torch.ones(2, device="cuda")
    noises = torch.tensor([[[0.1], [-0.2]], [[0.0], [0.3]], [[-0.2], [0.1]]], device="cuda")
    frozen = tuple(world.parameters()) + tuple(agent.critics.parameters())
    before = [parameter.detach().clone() for parameter in frozen]
    with rfo.frozen_parameters(world, agent.critics):
        loss = -rfo.imagined_return(agent, world, raw, mean, std, reward_std, noises, 0.99).mean()
        loss.backward()
        actual = agent.actor.bias.grad.clone()
        assert actual.abs().item() > 1e-3
        assert all(parameter.grad is None and not parameter.requires_grad for parameter in frozen)
        # Perturb the actual Euler actor parameter; this reference traverses the
        # entire horizon, so a detached intermediate state fails the derivative.
        with torch.no_grad():
            original = agent.actor.bias.clone()
            agent.actor.bias.copy_(original + 1e-3)
            plus = -rfo.imagined_return(agent, world, raw, mean, std, reward_std, noises, 0.99).mean()
            agent.actor.bias.copy_(original - 1e-3)
            minus = -rfo.imagined_return(agent, world, raw, mean, std, reward_std, noises, 0.99).mean()
            agent.actor.bias.copy_(original)
        torch.testing.assert_close(actual.squeeze(), (plus - minus) / 2e-3, rtol=2e-3, atol=2e-4)
        optimizer = torch.optim.Adam(agent.actor.parameters(), lr=3e-4, eps=1e-5)
        optimizer.step()
    assert all(parameter.requires_grad for parameter in frozen)
    for parameter, expected in zip(frozen, before):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


def test_transition_targets_and_gae_distinguish_death_timeout_and_reset():
    reset_next = np.array([[90.0], [80.0], [3.0]], dtype=np.float32)
    rewards = np.array([12.5, -21.0, 0.5], dtype=np.float32)
    terms = np.array([True, False, False])
    truncs = np.array([False, True, False])
    infos = {"final_observation": [np.array([1.0]), np.array([2.0]), None],
             "_final_observation": np.array([True, True, False])}
    targets, raw_rewards, labels = rfo.transition_targets(reset_next, rewards, terms, truncs, infos)
    np.testing.assert_array_equal(targets, [[1.0], [2.0], [3.0]])
    np.testing.assert_array_equal(reset_next, [[90.0], [80.0], [3.0]])
    np.testing.assert_array_equal(raw_rewards, rewards)
    np.testing.assert_array_equal(labels, [1.0, 0.0, 0.0])
    device = "cuda"
    _, returns = compute_gae_from_next_values(
        torch.tensor(raw_rewards[None], device=device), torch.zeros(1, 3, device=device),
        torch.tensor(labels[None], device=device), torch.tensor(truncs[None], device=device),
        torch.tensor([[10.0, 20.0, 30.0]], device=device), 0.9, 0.95,
    )
    # True death has no bootstrap; a timeout bootstraps its final observation,
    # never the new episode's reset state.
    torch.testing.assert_close(returns, torch.tensor([[12.5, -3.0, 27.5]], device=device))
    with pytest.raises(RuntimeError, match="final_observation"):
        rfo.transition_targets(reset_next, rewards, terms, truncs, {})


def test_recent_cfm_buffer_keeps_exactly_two_rollouts_and_current_coordinates():
    recent = rfo.RecentRollouts()
    rows = torch.tensor([0, 1], device="cuda")
    mean = torch.tensor([[10.0], [20.0]], device="cuda")
    std = torch.tensor([[2.0], [4.0]], device="cuda")
    for iteration in range(3):
        raw = torch.tensor([[10.0 + iteration], [20.0 + iteration]], device="cuda", requires_grad=True)
        native = torch.full((2, 1), float(iteration), device="cuda", requires_grad=True)
        recent.push(raw, native)
        with torch.no_grad():
            raw.fill_(-1000.0)
            native.fill_(-1000.0)
        batch_raw, batch_native = recent.minibatch(rows)
        kept = list(range(max(0, iteration - 1), iteration + 1))
        expected_raw = torch.tensor([[base + item] for item in kept for base in (10.0, 20.0)], device="cuda")
        torch.testing.assert_close(batch_raw, expected_raw)
        torch.testing.assert_close(batch_native, torch.tensor([[float(item)] for item in kept for _ in range(2)], device="cuda"))
        assert not batch_raw.requires_grad and not batch_native.requires_grad
        normalized = rfo.normalize_states(batch_raw, mean.repeat(len(kept), 1), std.repeat(len(kept), 1))
        expected = torch.tensor([[item / scale] for item in kept for scale in (2.0, 4.0)], device="cuda")
        torch.testing.assert_close(normalized, expected)
    assert len(recent.rollouts) == 2


def test_uniform_cfm_targets_are_physical_uniform_bounded_and_endpoint_safe():
    agent = make_agent(obs_dim=2, action_dim=2)
    physical, latent = rfo.uniform_targets(agent, 65536)
    assert torch.isfinite(latent).all()
    assert (physical >= agent.action_low).all() and (physical <= agent.action_high).all()
    assert (latent.abs() <= rfo.CFM_TARGET_BOUND).all()
    torch.testing.assert_close(agent.action_bias + agent.action_scale * latent.tanh(), physical, rtol=1e-5, atol=1e-6)
    # A uniform latent target would not yield uniform physical bins after tanh.
    unit = (physical - agent.action_low) / (agent.action_high - agent.action_low)
    for index in range(4):
        fraction = ((unit >= index / 4) & (unit < (index + 1) / 4)).float().mean()
        assert abs(fraction.item() - 0.25) < 0.01


def test_cfm_uses_forward_velocity_and_detached_numerically_bounded_targets():
    agent = make_agent()
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.zero_()
        agent.actor[-1].bias.fill_(0.5)
    observations = torch.zeros(2, 1, device="cuda")
    targets = torch.tensor([[1000.0], [1.5]], device="cuda", requires_grad=True)
    noise = torch.tensor([[0.0], [1.0]], device="cuda")
    times = torch.tensor([[0.0], [1.0]], device="cuda")
    loss = rfo.cfm_loss(agent, observations, targets, noise, times)
    expected = ((0.5 - rfo.CFM_TARGET_BOUND) ** 2) / 2
    torch.testing.assert_close(loss, torch.tensor(expected, device="cuda"))
    loss.backward()
    assert targets.grad is None
    assert agent.actor[-1].bias.grad.abs().item() > 0


def test_world_uses_fixed_raw_delta_reward_scales_and_differentiable_inputs():
    agent = make_agent()
    with torch.device("cuda"):
        world = rfo.WorldModel(1, 1)
    raw = torch.tensor([[1.0], [2.0]], device="cuda")
    next_raw = torch.tensor([[3.0], [5.0]], device="cuda")
    rewards = torch.tensor([12.0, -4.0], device="cuda")
    terms = torch.tensor([1.0, 0.0], device="cuda")
    world.initialize_coordinates(raw, next_raw, rewards, agent)
    scales = (world.state_std.clone(), world.delta_std.clone(), world.reward_scale.clone())
    with torch.no_grad():
        for parameter in world.parameters():
            parameter.zero_()
    loss, metrics = rfo.world_loss(world, raw, torch.zeros_like(raw), next_raw, rewards, terms)
    torch.testing.assert_close(loss, torch.tensor(2.0 + np.log(2.0), device="cuda", dtype=torch.float32))
    torch.testing.assert_close(metrics[3], torch.tensor(6.5, device="cuda").sqrt())
    torch.testing.assert_close(metrics[4], torch.tensor(80.0, device="cuda").sqrt())
    torch.testing.assert_close(metrics[5], torch.tensor(0.25, device="cuda"))
    # Fresh distributions never silently refit the coordinate system.
    rfo.world_loss(world, raw * 100, torch.ones_like(raw), next_raw * 100, rewards * 100, terms)
    for actual, expected in zip((world.state_std, world.delta_std, world.reward_scale), scales):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The residual transition preserves the raw-state input Jacobian even with
    # frozen model weights. Actor tests above additionally check action paths.
    states = raw.clone().requires_grad_()
    with rfo.frozen_parameters(world):
        predicted_next, _, _ = world(states, torch.zeros_like(raw))
        predicted_next.sum().backward()
        torch.testing.assert_close(states.grad, torch.ones_like(states))
        assert all(parameter.grad is None for parameter in world.parameters())


def test_compiled_actor_accumulation_matches_full_batch_gradient():
    """CUDA graph replays must accumulate every minibatch, not overwrite grads."""
    args = rfo.Args(imagination_horizon=3)
    agent = make_agent()
    with torch.device("cuda"):
        world = rfo.WorldModel(1, 1)
    raw = torch.tensor([[0.1], [0.3], [-0.2], [0.4]], device="cuda")
    mean, std = torch.zeros_like(raw), torch.ones_like(raw)
    reward_std = torch.ones(4, device="cuda")
    native = torch.randn_like(raw)
    noises = torch.randn(3, 4, 1, device="cuda")
    past_noise = torch.randn_like(raw)
    past_times = torch.rand_like(raw)
    _, uniform_native = rfo.uniform_targets(agent, 4)
    uniform_noise = torch.randn_like(raw)
    uniform_times = torch.rand_like(raw)

    def objective(states, means, stds, reward_stds, targets, flow_noise,
                  past_eps, past_t, uniform_z, uniform_eps, uniform_t):
        return rfo.actor_loss(
            agent, world, states, means, stds, reward_stds, states, targets,
            means, stds, flow_noise, past_eps, past_t,
            uniform_z, uniform_eps, uniform_t, args,
        )

    compiled = torch.compile(objective, mode="reduce-overhead", fullgraph=True)
    inputs = (raw, mean, std, reward_std, native, noises,
              past_noise, past_times, uniform_native, uniform_noise, uniform_times)
    accumulated = [torch.zeros_like(p) for p in agent.actor.parameters()]
    with rfo.frozen_parameters(world, agent.critics):
        for _ in range(4):
            agent.actor.zero_grad(set_to_none=True)
            full_loss, _ = objective(*inputs)
            full_loss.backward()
            expected = [p.grad.clone() for p in agent.actor.parameters()]
            for parameter, buffer in zip(agent.actor.parameters(), accumulated):
                parameter.grad = buffer
            agent.actor.zero_grad(set_to_none=False)
            for start in (0, 2):
                rows = slice(start, start + 2)
                minibatch = tuple(value[:, rows] if index == 5 else value[rows]
                                  for index, value in enumerate(inputs))
                torch.compiler.cudagraph_mark_step_begin()
                loss, _ = compiled(*minibatch)
                (0.5 * loss).backward()
            for parameter, gradient in zip(agent.actor.parameters(), expected):
                torch.testing.assert_close(parameter.grad, gradient, rtol=3e-4, atol=3e-6)
            assert all(p.grad is None for p in world.parameters())
            assert all(p.grad is None for p in agent.critics.parameters())
            with torch.no_grad():
                for parameter in agent.actor.parameters():
                    parameter.add_(parameter.grad, alpha=-3e-4)
