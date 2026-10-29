"""Contracts for the box-bounded endpoint FPO flow actor (cleanrl/fpo/ppo_continuous_action_fpo_sepclip_endpoint_v9.py)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[1]


def _load():
    path = ROOT / "cleanrl/fpo/ppo_continuous_action_fpo_sepclip_endpoint_v9.py"
    spec = importlib.util.spec_from_file_location("fpo_sepclip_endpoint_v9", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _agent(module, obs_dim=17, act_dim=6):
    envs = SimpleNamespace(
        single_action_space=gym.spaces.Box(-1.0, 1.0, (act_dim,), np.float32),
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
    )
    args = SimpleNamespace(flow_steps=10, prior_std=0.6, time_embed_dim=8)
    torch.manual_seed(0)
    agent = module.Agent(envs, args)
    # Give the endpoint field real structure so the comparison is not trivial.
    with torch.no_grad():
        agent.actor.head[0].weight.normal_()
        agent.actor.readout_gain.fill_(0.3)
    agent.normalize_matrices()
    return agent


@pytest.mark.skipif(not torch.cuda.is_available(), reason="host mirrors copy CUDA parameters")
def test_host_sampler_matches_device_euler():
    module = _load()
    agent = _agent(module).cuda()
    sampler = module.HostFlowSampler(agent, num_rows=16)
    sampler.refresh()
    observations = np.random.default_rng(1).standard_normal((16, 17)).astype(np.float32)
    native, action = sampler(observations, np.random.default_rng(7))
    noise = np.random.default_rng(7).standard_normal((16, 6), dtype=np.float32)
    expected = agent.sample(torch.from_numpy(observations).cuda(), torch.from_numpy(noise).cuda()[:, None, :])[:, 0]
    np.testing.assert_allclose(native, expected.cpu().numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(action, native)  # unit box: the stored sample is the env action
    assert np.abs(native).max() <= 1.0
    assert np.abs(native - np.tanh(0.6 * noise)).max() > 1e-2


def test_fpo_ratio_is_one_before_any_update_and_every_pair_coordinate_is_stratified():
    module = _load()
    agent = _agent(module)
    args = SimpleNamespace(batch_size=64, mc_samples=16, clip_coef=0.05, norm_adv=False, vf_coef=0.5)
    pairs = module.RolloutPairs(args, 6, "cpu")
    pairs.draw(torch.Generator().manual_seed(3))
    uniform = torch.cat((pairs.times, torch.special.ndtr(pairs.noise.double()).float()), dim=-1)
    strata = (uniform * args.mc_samples).floor().sort(dim=1).values
    assert torch.equal(strata, torch.arange(16.0)[None, :, None].expand(64, -1, 7))
    observations, native = torch.randn(64, 17), torch.randn(64, 6)
    with torch.no_grad():
        old = module.cfm_loss(agent, agent.actor.trunk(observations), native, pairs.times, pairs.noise)
    loss, metrics = module.fpo_loss(agent, observations, native, pairs.times, pairs.noise, old,
                                    torch.randn(64), torch.randn(64), args)
    assert metrics[3].abs() < 1e-7 and metrics[5] == 0.0  # proxy KL and |log-ratio|
    loss.backward()
    assert agent.actor.velocity_trunk.first[0].gate.weight.grad.abs().sum() > 0


def test_initial_policy_matches_control_beta_spread_inside_the_box():
    module = _load()
    envs = SimpleNamespace(
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), np.float32),
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), np.float32),
    )
    torch.manual_seed(0)
    prior_std = module.Args.prior_std
    agent = module.Agent(envs, SimpleNamespace(flow_steps=10, prior_std=prior_std, time_embed_dim=8))
    agent.normalize_matrices()  # runs at startup and must not undo the near-zero init
    actions = agent.sample(torch.randn(256, 17), torch.randn(256, 512, 6))
    beta = 1.0 + np.log(2.0)  # control: alpha = beta = 1 + softplus(0)
    beta_std = 2.0 * np.sqrt(beta * beta / ((2 * beta) ** 2 * (2 * beta + 1)))
    assert abs(actions.std(dim=1).mean().item() - beta_std) < 0.02
    assert (actions.abs() > 0.99).float().mean() < 1e-3


def test_euler_stays_in_box_and_ends_on_last_endpoint_even_far_from_init():
    module = _load()
    agent = _agent(module)
    with torch.no_grad():
        agent.actor.readout_gain.fill_(25.0)  # saturating endpoints
    observations, noise = torch.randn(64, 17), 3.0 * torch.randn(64, 32, 6)
    actions = agent.sample(observations, noise)
    assert actions.abs().max() <= 1.0
    # Recompute the last endpoint from the pre-final state: the sample equals it.
    features = agent.actor.trunk(observations)[:, None, :].expand(-1, 32, -1)
    state = agent.prior_std * noise
    times = torch.empty(64, 32, 1)
    with torch.no_grad():
        for step in range(agent.flow_steps - 1):
            times.fill_(step / agent.flow_steps)
            state = state + (agent.actor.endpoint(features, state, times) - state) / (agent.flow_steps - step)
        times.fill_((agent.flow_steps - 1) / agent.flow_steps)
        last = agent.actor.endpoint(features, state, times)
    torch.testing.assert_close(actions, last, rtol=0, atol=0)


def test_endpoint_loss_is_velocity_loss_weighted_by_one_minus_tau_squared():
    module = _load()
    agent = _agent(module)
    observations, actions = torch.randn(32, 17), torch.rand(32, 6) * 2 - 1
    times, noise = torch.rand(32, 4, 1) * 0.98, torch.randn(32, 4, 6)
    with torch.no_grad():
        features = agent.actor.trunk(observations)
        source = agent.prior_std * noise
        states = (1 - times) * source + times * actions[:, None, :]
        endpoint = agent.actor.endpoint(features[:, None, :].expand(-1, 4, -1), states, times)
        velocity = (endpoint - states) / (1 - times)
        weighted = ((1 - times) ** 2 * (velocity - (actions[:, None, :] - source)).square()).mean(-1).mean(-1)
        loss = module.cfm_loss(agent, features, actions, times, noise)
    torch.testing.assert_close(loss, weighted, rtol=1e-4, atol=1e-6)


def test_owen_scramble_permutes_each_dyadic_level_and_mul32_wraps():
    module = _load()
    x = torch.randint(0, 2 ** 32, (4096,), dtype=torch.int64)
    for constant in (0x6C50B47C, 0x8D22F6E6):
        assert torch.equal(module._mul32(x, constant),
                           torch.tensor([(int(v) * constant) % 2 ** 32 for v in x]))
    assert torch.equal(module._reverse_bits32(module._reverse_bits32(x)), x)
    # Nested scrambling: inputs sharing their top k digits still share them after.
    seed = torch.tensor(0x1234ABCD)
    digits = torch.arange(256, dtype=torch.int64) << 24
    scrambled = module.owen_scramble(digits, seed)
    for k in (1, 4, 8):
        assert torch.unique(scrambled >> (32 - k)).numel() == 2 ** k
        parent = scrambled.view(2 ** k, -1) >> (32 - k)
        assert torch.equal(parent, parent[:, :1].expand_as(parent))


def test_sobol_pairs_are_unbiased_and_independent_across_actions():
    module = _load()
    args = SimpleNamespace(batch_size=4096, mc_samples=16)
    pairs = module.RolloutPairs(args, 6, "cpu")
    pairs.draw(torch.Generator().manual_seed(5))
    times, noise = pairs.times.squeeze(-1), pairs.noise
    assert abs(times.mean() - 0.5) < 5e-3 and abs(noise.mean()) < 5e-3
    assert abs(noise.square().mean() - 1.0) < 1e-2
    # Per-action sets differ, and the first-point taus are uniform across actions.
    assert not torch.equal(times[0], times[1])
    assert abs(times[:, 0].mean() - 0.5) < 2e-2 and abs(times[:, 0].var() - 1 / 12) < 5e-3
    previous = pairs.noise.clone()
    pairs.draw(torch.Generator().manual_seed(6))
    assert not torch.equal(previous, pairs.noise)
