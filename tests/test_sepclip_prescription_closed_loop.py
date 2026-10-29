"""Architecture-aware compiled CUDA contracts; execution belongs to mlq."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import audit_sepclip_prescription_closed_loop as audit
from cleanrl import ppo_continuous_action_sepclip_additive_prerms_anvil_v14 as v14
from cleanrl import ppo_continuous_action_sepclip_constraint_transport_v15 as v15
from cleanrl.shared.runtime import configure_runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')]


@pytest.fixture(autouse=True)
def runtime():
    configure_runtime(matmul_precision='highest', allow_tf32=False)


def replay_fixture(module, architecture=None):
    torch.manual_seed(23)
    spaces = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,)),
                             single_action_space=gym.spaces.Box(-1., 1., (6,)))
    agent = (module.Agent(spaces, actor_architecture=architecture) if architecture else module.Agent(spaces)).cuda()
    agent.normalize_matrices()
    banks = {}
    for i, name in enumerate(audit.BANKS):
        banks[name] = {k: p.detach().clone() for k, p in agent.actor.named_parameters()}
        if i:
            for value in banks[name].values():
                value.add_(torch.randn_like(value)*(.013*i))
    def forward(parameters, x):
        z = torch.func.functional_call(agent.actor, parameters, (x,), strict=True)
        a, b = (torch.nn.functional.softplus(z)+1).chunk(2, -1)
        return z, a, b
    compiled = torch.compile(forward, fullgraph=True, dynamic=True, options={'triton.cudagraphs': False})
    value = torch.compile(lambda x: agent.get_value(x).flatten(), fullgraph=True,
                          dynamic=True, options={'triton.cudagraphs': False})
    return SimpleNamespace(agent=agent, banks=banks, compiled_forward=compiled, compiled_old_value=value)


@pytest.mark.parametrize('module,architecture', [(v14, 'original'), (v14, 'additive_prerms'), (v15, None)])
def test_actual_compiled_actor_for_every_hybrid_arm_and_future_observation(module, architecture):
    replay = replay_fixture(module, architecture)
    rng = np.random.default_rng(41)
    initial = rng.normal(size=(14, 17)).astype(np.float32)
    future = (initial*1.7 + rng.normal(size=initial.shape)).astype(np.float32)
    independent_forwards = []
    for name in audit.BANKS:
        independent = copy.deepcopy(replay.agent)
        independent.actor.load_state_dict(replay.banks[name], strict=True)
        independent_forwards.append(torch.compile(independent.get_policy_and_value,
            fullgraph=True, dynamic=True, options={'triton.cudagraphs': False}))
    for step, obs in ((0, initial), (1, future), (8, future*-.37)):
        z, a, b = audit.policy_on_arms(replay, obs, step)
        assignment = np.tile(audit.arm_indices(step), 2)
        for bank, direct in enumerate(independent_forwards):
            rows = np.flatnonzero(assignment == bank)
            with torch.no_grad():
                ea, eb, _ = direct(torch.as_tensor(obs[rows], device='cuda'))
            np.testing.assert_allclose(a[rows], ea.cpu().numpy(), atol=3e-6, rtol=3e-6)
            np.testing.assert_allclose(b[rows], eb.cpu().numpy(), atol=3e-6, rtol=3e-6)
        # Exposes stale observation/actor replacement and changed input/residual banks.
        assert z.shape == (14, 12)
        assert np.max(np.abs(a[1]-a[0])) > 1e-6


def states():
    return [dict(episode_age=20, normalized_obs=[0.]*17, raw_obs=[0.]*17,
                 obs_means=[0.]*17, obs_variances=[1.]*17, reward_std=std)
            for std in (2., 3.)]


class TraceEnv:
    def __init__(self, size, bad_step=None, bad_kind=None):
        self.size, self.step_index = size, 0
        self.bad_step, self.bad_kind = bad_step, bad_kind
        self.envs = [SimpleNamespace(unwrapped=SimpleNamespace(_ctrl_cost_weight=.1, _forward_reward_weight=1.))]
        self.rewards, self.actions = [], []
        self.closed = False
    def reset(self, seed):
        assert seed == 1
    def step(self, action):
        self.actions.append(action.copy())
        self.step_index += 1
        # Distinct arm-dependent raw rewards; independent known normalization and sums.
        reward = self.step_index*.123456789 + action.sum(-1).astype(np.float64)
        raw = np.full((self.size, 17), self.step_index*.271, dtype=np.float64)
        terms, truncs = np.zeros(self.size, dtype=bool), np.zeros(self.size, dtype=bool)
        if self.step_index == self.bad_step:
            if self.bad_kind == 'reward': reward[0] = np.nan
            elif self.bad_kind == 'obs': raw[0, 0] = np.inf
            elif self.bad_kind == 'termination': terms[0] = True
            elif self.bad_kind == 'truncation': truncs[0] = True
        self.rewards.append(reward.copy())
        cost = np.square(action).sum(-1).astype(np.float64)*.1
        reward_run = reward+cost
        if self.step_index == self.bad_step and self.bad_kind == 'reward_contract':
            reward_run = reward_run + .01
        return raw, reward, terms, truncs, dict(x_velocity=reward+cost,
                                              reward_ctrl=-cost, reward_run=reward_run)
    def close(self): self.closed = True


def test_closed_loop_returns_use_exact_gamma_per_reward_fp32_law_and_old_fixed_tail(monkeypatch):
    replay = replay_fixture(v14, 'additive_prerms')
    old_state = {k: v.clone() for k, v in replay.agent.state_dict().items()}
    monkeypatch.setattr(audit, 'restore', lambda *a: 0.)
    monkeypatch.setattr(audit, 'verify_saved_transition', lambda *a: 0.)
    draws = []
    draw = audit.native_draw
    def record_draw(alpha, beta, mode, uniforms):
        draws.append(uniforms.copy())
        return draw(alpha, beta, mode, uniforms)
    monkeypatch.setattr(audit, 'native_draw', record_draw)
    env = TraceEnv(14)
    gamma = .73
    out = audit.simulate(replay, states(), [1, 3], gamma, 'sample', 0,
                         env_factory=lambda *a, **k: env)
    std = np.repeat([2., 3.], 7)
    raw = np.asarray(env.rewards)
    normalized = np.asarray([[np.float64(np.float32(np.clip(r/s, -10, 10)))
                              for r, s in zip(row, std)] for row in raw])
    for h in (1, 3):
        row = out['horizons'][h]
        weights = gamma**np.arange(h)
        expected_raw = (weights[:, None]*raw[:h]).sum(0).reshape(2, 7)
        expected_norm = (weights[:, None]*normalized[:h]).sum(0).reshape(2, 7)
        np.testing.assert_allclose(row['raw_discounted'], expected_raw, atol=1e-12, rtol=0)
        np.testing.assert_allclose(row['normalized_discounted'], expected_norm, atol=1e-12, rtol=0)
        np.testing.assert_allclose(row['raw_fixed_old_critic_tail'], expected_raw+gamma**h*row['old_critic_terminal']*std.reshape(2, 7), atol=1e-12, rtol=0)
    assert env.closed
    for quantiles in draws:
        grid = quantiles.reshape(2, 7, 6)
        np.testing.assert_array_equal(grid, np.repeat(grid[:, :1], 7, axis=1))
        assert not np.array_equal(grid[0, 0], grid[1, 0])
    for key, value in replay.agent.state_dict().items():
        torch.testing.assert_close(value, old_state[key], atol=0, rtol=0)
    # Repeating replica preserves shared-arm quantiles; independent replica changes them.
    env2, env3 = TraceEnv(14), TraceEnv(14)
    audit.simulate(replay, states(), [3], gamma, 'sample', 0, env_factory=lambda *a, **k: env2)
    audit.simulate(replay, states(), [3], gamma, 'sample', 1, env_factory=lambda *a, **k: env3)
    np.testing.assert_array_equal(env.actions, env2.actions)
    assert not np.array_equal(env.actions, env3.actions)


@pytest.mark.parametrize('kind', ['reward', 'obs', 'termination', 'truncation', 'reward_contract'])
def test_transition_nonfinite_or_boundary_is_rejected_before_output(monkeypatch, kind):
    replay = replay_fixture(v15)
    monkeypatch.setattr(audit, 'restore', lambda *a: 0.)
    monkeypatch.setattr(audit, 'verify_saved_transition', lambda *a: 0.)
    env = TraceEnv(14, 2, kind)
    with pytest.raises(ValueError, match='nonfinite|boundary|decomposition mismatch'):
        audit.simulate(replay, states(), [3], .99, 'mean', 0, env_factory=lambda *a, **k: env)
    assert env.closed and env.step_index == 2


def test_mode_rounding_domains_and_frozen_reward_fail_closed():
    one = np.float32(1)
    above = np.nextafter(one, np.float32(2))
    alpha = np.asarray([[one, above, above, one]], dtype=np.float32)
    beta = np.asarray([[above, one, above, one]], dtype=np.float32)
    np.testing.assert_array_equal(audit.native_draw(alpha, beta, 'mode', np.full((1, 4), .5)), [[0., 1., .5, .5]])
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match='nonfinite'):
            audit.frozen_reward(np.asarray([bad]), np.asarray([1.]))
    with np.errstate(over='ignore'), pytest.raises(ValueError, match='before clipping'):
        audit.frozen_reward(np.asarray([1e308]), np.asarray([1e-300]))
    for gamma, h, age in ((np.nan, 2, 20), (.99, 0, 20), (.99, 512, 488)):
        with pytest.raises(ValueError): audit.validate_horizons([dict(episode_age=age)], [h], gamma)


def test_hybrid_interaction_is_independent_finite_return_identity():
    values = np.asarray([[2., 9., 12., 4., 5., 7., 8.]])
    row = dict(horizons={3: dict(raw_discounted=values)})
    result = audit.summarize([row], [3], 'mean')[0]['metrics']['raw_discounted']
    assert result['feedback_interaction']['first']['selected_state_average_mean'] == 2.
    assert result['feedback_interaction']['endpoint']['selected_state_average_mean'] == -1.
    assert result['paired_gains']['first_all']['selected_state_average_mean'] == 7.


def test_saved_transition_replay_checks_reward_observation_and_nonfinite_metadata():
    saved = states()
    for state in saved:
        state.update(physical_action=[0.]*6, stored_raw_next_obs=[.271]*17,
                     stored_raw_reward=.123456789)
    assert audit.verify_saved_transition(TraceEnv(14), saved) == 0.
    wrong = copy.deepcopy(saved)
    wrong[0]['stored_raw_reward'] += .01
    with pytest.raises(ValueError, match='replay mismatch'):
        audit.verify_saved_transition(TraceEnv(14), wrong)
    wrong[0]['stored_raw_reward'] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        audit.verify_saved_transition(TraceEnv(14), wrong)
