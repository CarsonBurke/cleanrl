"""Physical counterfactual contracts; execute through mlq on the CUDA runner."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.runtime import configure_runtime

spec = importlib.util.spec_from_file_location('audit_physics', Path(__file__).parents[1]/'scripts/audit_sepclip_physics.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def envs():
    configure_runtime(matmul_precision='highest', allow_tf32=False)
    assert torch.cuda.is_available(), 'These contracts must be queued on CUDA'
    envs = make_mujoco_vector_env('HalfCheetah-v4', 2, num_threads=2)
    envs.reset(seed=1)
    yield envs
    envs.close()


def record(envs):
    return [dict(state=audit.integration_state(e.unwrapped),
                 rng=copy.deepcopy(e.unwrapped.np_random.bit_generator.state),
                 age=int(envs._episode_lengths[i]), obs=e.unwrapped._get_obs().copy(), row=i)
            for i, e in enumerate(envs.envs)]


def test_complete_integration_state_replays_multistep_exactly(envs):
    rng = np.random.default_rng(7)
    actions = rng.uniform(-1, 1, (40, 2, 6)).astype(np.float32)
    for action in actions[:20]:
        envs.step(action)
    saved = record(envs)
    expected = [envs.step(action)[:4] for action in actions[20:]]
    final = record(envs)
    restored = audit.restore_states(envs, saved)
    np.testing.assert_array_equal(restored, np.stack([s['obs'] for s in saved]))
    for action, expected_step in zip(actions[20:], expected):
        actual_step = envs.step(action)[:4]
        for actual, expected_value in zip(actual_step, expected_step):
            np.testing.assert_array_equal(actual, expected_value)
    for actual, expected_state in zip(record(envs), final):
        np.testing.assert_array_equal(actual['state'], expected_state['state'])
        assert actual['age'] == expected_state['age']
        assert actual['rng'] == expected_state['rng']


def test_common_uniform_beta_forks_are_identical_and_monotone():
    alpha = np.array([[2., 4., 10.], [2., 4., 10.]], dtype=np.float32)
    beta = np.array([[3., 6., 20.], [3., 6., 20.]], dtype=np.float32)
    u = np.repeat(np.array([[.1, .5, .9]]), 2, axis=0)
    native = audit.native_action(alpha, beta, 'sample', u)
    np.testing.assert_array_equal(native[0], native[1])
    shifted = audit.native_action(alpha+1, beta, 'sample', u)
    assert np.all(shifted > native)


def test_frozen_normalization_preserves_explicit_row_metadata():
    snapshot = {'obs_norm': {'means': np.array([[1., 2.], [3., 4.]]),
                             'variances': np.array([[4., 9.], [16., 25.]]),
                             'counts': np.array([100., 200.])}}
    norm = audit.norm_from_snapshot(snapshot, [1, 0, 1])
    saved = {k: getattr(norm, k).copy() for k in ('means', 'variances', 'counts')}
    obs = np.array([[7., 9.], [3., 5.], [7., 9.]])
    expected = (obs-saved['means'])/np.sqrt(saved['variances']+1e-8)
    np.testing.assert_allclose(audit.normalized(norm, obs), expected, atol=1e-7)
    for name in saved:
        np.testing.assert_array_equal(getattr(norm, name), saved[name])


def test_beta_head_score_matches_analytic_physical_control_cost_gradient():
    # Deterministic quadrature checks the actual head score against a known
    # physical torque objective; no critic or fitted surrogate enters this test.
    from scipy.special import betaincinv, expit
    alpha, beta = 2.5, 3.2
    logits = np.log(np.expm1(np.array([[alpha-1, beta-1]])))
    uniforms = ((np.arange(65536)+.5)/65536)[:, None]
    native = betaincinv(alpha, beta, uniforms)
    score = audit.beta_logit_score(alpha, beta, logits, native)
    objective = -.1*(2*native-1)**2
    estimate = (score*objective).mean(0)
    k = alpha+beta
    denominator = k*(k+1)
    numerator = alpha*(alpha+1)
    moment_derivative_a = (2*alpha+1)/denominator-numerator*(2*k+1)/denominator**2
    moment_derivative_b = -numerator*(2*k+1)/denominator**2
    expected = -.4*np.array([moment_derivative_a-beta/k**2,
                             moment_derivative_b+alpha/k**2])*expit(logits[0])
    np.testing.assert_allclose(estimate, expected, rtol=.002, atol=2e-6)
    np.testing.assert_allclose(score.mean(0), 0, atol=3e-5)


def test_compiled_actor_supports_independent_score_gradient_forwards():
    # Actual Agent + compiled backward, matching the production conditional-PG
    # mapping. Retained compiled graphs fail with donated buffers in Torch2.11.
    import importlib
    from types import SimpleNamespace
    import gymnasium as gym

    configure_runtime(matmul_precision='highest', allow_tf32=False)
    assert torch.cuda.is_available(), 'This contract must be queued on CUDA'
    torch._dynamo.config.recompile_limit = 64
    module = importlib.import_module(audit.MODULE)
    spaces = SimpleNamespace(single_action_space=gym.spaces.Box(-1, 1, (6,), dtype=np.float32),
                             single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32))
    torch.manual_seed(1)
    agent = module.Agent(spaces).cuda().eval()
    agent.normalize_matrices()
    agent.actor.forward = torch.compile(agent.actor.forward, fullgraph=True, dynamic=False,
                                        options={'triton.cudagraphs': False})
    parameters = tuple(agent.actor.parameters())
    observations = torch.linspace(-2, 2, 16*17, device='cuda').reshape(16, 17)
    score_pg = torch.linspace(-.2, .3, 16*12, device='cuda').reshape(16, 12)
    first = torch.autograd.grad(agent.actor(observations), parameters, grad_outputs=score_pg)
    # A fresh forward is essential; no retain_graph or donated-buffer disabling.
    second = torch.autograd.grad(agent.actor(observations), parameters, grad_outputs=-score_pg)
    assert sum(float(g.square().sum()) for g in first) > 0
    for a, b in zip(first, second):
        torch.testing.assert_close(a, -b, rtol=1e-5, atol=1e-6)
