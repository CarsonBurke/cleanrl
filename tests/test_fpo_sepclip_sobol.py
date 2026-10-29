"""Contracts for the FPO-on-sepclip flow actor with Sobol pairs (cleanrl/fpo/ppo_continuous_action_fpo_sepclip_sobol_v8.py)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[1]


def _load():
    path = ROOT / "cleanrl/fpo/ppo_continuous_action_fpo_sepclip_sobol_v8.py"
    spec = importlib.util.spec_from_file_location("fpo_sepclip_sobol_v8", path)
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
    # Give the velocity field real structure so the comparison is not trivial.
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
    np.testing.assert_allclose(action, np.tanh(native), rtol=1e-6, atol=1e-6)
    assert np.abs(native - 0.6 * noise).max() > 1e-2


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


def test_initial_policy_is_tanh_of_prior_noise():
    module = _load()
    envs = SimpleNamespace(
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), np.float32),
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), np.float32),
    )
    torch.manual_seed(0)
    agent = module.Agent(envs, SimpleNamespace(flow_steps=10, prior_std=0.6, time_embed_dim=8))
    agent.normalize_matrices()  # runs at startup and must not undo the near-zero init
    noise = torch.randn(256, 64, 6)
    native = agent.sample(torch.randn(256, 17), noise)
    assert (native - 0.6 * noise).square().mean().sqrt() < 0.1


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
