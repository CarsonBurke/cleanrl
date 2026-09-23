"""CUDA derivative contracts; execute through mlq, never as reduced training."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Beta
from torch.nn import functional as F

from cleanrl.ppo_continuous_action_latent_costate_v10 import (
    Agent, ActorNaturalGradient, Dynamics, LatentCostate, FixedAffineObsNorm,
    RecordingObsNorm, observed_beta_jacobian, differential_targets, action_jacobian,
    projected_residual, beta_geometry, beta_kl_reference, policy_gain_kl,
    incoming_projection,
)
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.vector_norm import VectorObsNorm
from cleanrl.shared.rollout_transfer import RolloutTransfer


@pytest.fixture(autouse=True)
def runtime():
    assert torch.cuda.is_available()
    configure_runtime(cudnn_deterministic=True, matmul_precision='highest', allow_tf32=False)
    torch.manual_seed(1)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_observed_beta_derivative_matches_actual_rsample_backward(dtype):
    # Reference differentiation is FP64, including when simulating FP32 host
    # samples. FP32 Dirichlet transport itself has cancellation sensitivity.
    alpha = (1.1 + 20 * torch.rand(256, 3, device='cuda', dtype=dtype)).double().requires_grad_()
    beta = (1.1 + 20 * torch.rand(256, 3, device='cuda', dtype=dtype)).double().requires_grad_()
    native = Beta(alpha, beta).rsample()
    ranges = torch.tensor([2., 3., .7], device='cuda', dtype=dtype)
    expected = torch.autograd.grad((native * ranges).sum(), (alpha, beta))
    actual = observed_beta_jacobian(native.detach().to(dtype), torch.stack((alpha, beta), -1).detach().to(dtype), ranges)
    torch.testing.assert_close(actual.double(), torch.stack(expected, -1), rtol=3e-6, atol=1e-8)
    assert (actual[..., 0] > 0).all() and (actual[..., 1] < 0).all()


class LinearPhysics(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('A', torch.tensor([[.8, .1], [-.2, .9]], device='cuda', dtype=torch.float64))
        self.register_buffer('B', torch.tensor([[.3, -.4], [.7, .2]], device='cuda', dtype=torch.float64))
        self.register_buffer('w', torch.tensor([.4, -.1], device='cuda', dtype=torch.float64))
        self.register_buffer('u', torch.tensor([.2, -.3], device='cuda', dtype=torch.float64))

    def forward(self, state, action):
        return torch.cat((state @ self.A.T + action @ self.B.T,
                          (state @ self.w + action @ self.u)[:, None]), -1)


@pytest.mark.parametrize('continuation', [0., 1.])
def test_vector_bellman_matches_full_reward_derivative_including_policy(continuation):
    model = LinearPhysics()
    actor = nn.Linear(2, 4).cuda().double()
    state = torch.randn(32, 2, device='cuda', dtype=torch.float64, requires_grad=True)
    alpha, beta = (F.softplus(actor(state)) + 1).chunk(2, -1)
    native = Beta(alpha, beta).rsample()
    action = 2 * native - 1
    output = model(state, action)
    full_return = output[:, -1] - .1 * action.square().sum(-1) + continuation * .5 * output[:, :2].square().sum(-1)
    expected, = torch.autograd.grad(full_return.sum(), state)
    jac = observed_beta_jacobian(native.detach(), torch.stack((alpha, beta), -1).detach(),
                                 torch.full((2,), 2., device='cuda', dtype=torch.float64))
    # Deliberately pass a differentiable next costate; target must stop it.
    actual, qa, eta = differential_targets(model, actor, state, action.detach(), jac,
                                          output[:, :2], torch.full((32,), continuation, device='cuda'), .1)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    expected_qa = model.u - .2 * action.detach() + continuation * output[:, :2].detach() @ model.B
    torch.testing.assert_close(qa, expected_qa)
    assert not actual.requires_grad and not eta.requires_grad


def test_action_jacobian_and_projected_error_match_chain_rule():
    model = LinearPhysics()
    state = torch.randn(32, 2, device='cuda', dtype=torch.float64)
    action = torch.randn_like(state)
    jac = action_jacobian(model, state, action)
    torch.testing.assert_close(jac, model.B.expand(32, -1, -1))
    beta_jac = torch.randn(32, 2, 2, device='cuda', dtype=torch.float64)
    error = torch.randn_like(state)
    projected = projected_residual(error, jac, beta_jac)
    torch.testing.assert_close(projected, (error @ model.B)[:, :, None] * beta_jac)


def test_incoming_projection_respects_time_order_and_episode_boundaries():
    jac = torch.arange(24, device='cuda', dtype=torch.float32).reshape(6, 2, 2)
    beta_jac = jac + 30
    continuation = torch.tensor([1., 0., 1., 1., 1., 1.], device='cuda')
    age = torch.tensor([8, 999, 9, 0, 10, 1], device='cuda')
    incoming, beta_incoming = incoming_projection(jac, beta_jac, continuation, age, 2)
    assert (incoming[:2] == 0).all() and (incoming[3] == 0).all()
    torch.testing.assert_close(incoming[2], jac[0])
    torch.testing.assert_close(incoming[4:], jac[2:4])
    torch.testing.assert_close(beta_incoming[2:], beta_jac[:-2])


def test_latent_covector_pullback_and_terminal_boundary():
    critic = LatentCostate(5, width=32, latent_features=7).cuda()
    nn.init.normal_(critic.head.weight, std=.02)
    state = torch.randn(64, 5, device='cuda')
    age = torch.arange(64, device='cuda') + 970
    value = critic(state, age)
    assert value.shape == state.shape and critic.head.out_features == 12
    assert (value[age >= 1000] == 0).all()
    value.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters())
    assert sum(p.grad.square().sum() for p in critic.encoder.parameters()) > 0


def test_frozen_affine_normalization_preserves_terminal_state_and_old_buffer():
    shared = VectorObsNorm(2, (3,))
    recorder = RecordingObsNorm(shared)
    raw = np.array([[1., 2., 3.], [4., 5., 6.]], np.float32)
    recorder.normalize(raw)
    np.testing.assert_array_equal(recorder.last_raw, raw)
    norm = FixedAffineObsNorm(shared, 2, (3,))
    old = norm.normalize(raw)
    saved = old.copy()
    final = np.array([10., 20., 30.])
    next_obs, transition = norm.normalize_step(raw + 1, np.array([False, False]),
                                             np.array([True, False]), {'final_observation': [final, None]})
    np.testing.assert_array_equal(old, saved)
    np.testing.assert_allclose(transition[0], (final - norm.mean) * norm.inverse_std, rtol=1e-6)
    np.testing.assert_array_equal(transition[1], next_obs[1])
    assert not np.array_equal(transition[0], next_obs[0])


def test_halfcheetah_progress_and_control_contract():
    env = gym.make('HalfCheetah-v4')
    try:
        env.reset(seed=1)
        action = np.array([.2, -.3, .4, -.5, .6, -.7], dtype=np.float32)
        _, reward, _, _, info = env.step(action)
        coefficient = env.unwrapped._ctrl_cost_weight
        np.testing.assert_allclose(reward + coefficient * np.square(action).sum(), info['reward_run'], rtol=1e-6)
    finally:
        env.close()


def test_shared_rollout_transfer_stages_actual_next_states():
    transfer = RolloutTransfer(2, 2, (3,), torch.device('cuda'),
                              fields={'observations': (3,), 'next_states': (3,),
                                      'native_actions': (2,), 'raw_rewards': ()})
    try:
        observed = np.zeros((2, 3), np.float32)
        following = np.zeros_like(observed)
        for step in range(2):
            observed.fill(step)
            following.fill(step + 10)
            transfer.push(step, np.ones(2, np.float32), np.zeros(2, bool), np.zeros(2, bool),
                          observations=observed, next_states=following,
                          native_actions=np.full((2, 2), .5, np.float32), raw_rewards=np.ones(2, np.float32))
        observed.fill(-99)
        following.fill(-99)
        batch = transfer.upload()
        torch.testing.assert_close(batch.fields['observations'][0], torch.zeros((2, 3), device='cuda'))
        torch.testing.assert_close(batch.fields['next_states'][0], torch.full((2, 3), 10., device='cuda'))
        torch.testing.assert_close(batch.fields['next_states'][1], torch.full((2, 3), 11., device='cuda'))
    finally:
        transfer.close()


def test_compiled_costate_learning_targets_and_two_actor_steps():
    device = 'cuda'
    env = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (5,)),
                          single_action_space=gym.spaces.Box(-1., 1., (3,)))
    actor = Agent(env).cuda().actor
    model = Dynamics(5, 3, 32).cuda()
    critic = LatentCostate(5, 32, 7).cuda()
    model_optimizer = torch.optim.Adam(model.parameters(), lr=.001, fused=True)
    optimizer = torch.optim.Adam(critic.parameters(), lr=.001, fused=True)
    obs = torch.randn(128, 5, device=device)
    physical = torch.rand(128, 3, device=device) * 1.8 - .9
    next_obs = obs + .1 * torch.randn_like(obs)
    age = torch.arange(128, device=device)
    weights = torch.ones(128, device=device)

    def model_loss():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(obs, physical)
        return (output[:, :5] - next_obs).square().mean() + output[:, -1].square().mean()

    fit_model = torch.compile(model_loss, fullgraph=True, mode='reduce-overhead')
    loss = fit_model()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model_optimizer.step()
    del loss
    for p in model.parameters():
        p.requires_grad_(False)

    predict = torch.compile(critic, fullgraph=True, mode='reduce-overhead')
    target_fn = torch.compile(lambda x, a, j, n, mask: differential_targets(model, actor, x, a, j, n, mask, .1),
                              fullgraph=True, mode='reduce-overhead')
    jac_fn = torch.compile(lambda x, a: action_jacobian(model, x, a), fullgraph=True, mode='reduce-overhead')
    beta_fn = torch.compile(observed_beta_jacobian, fullgraph=True, mode='reduce-overhead')

    def critic_loss(x, t, target, j, beta_j):
        error = critic(x, t) - target
        return error.square().mean() + projected_residual(error, j, beta_j).square().mean()

    fit_critic = torch.compile(critic_loss, fullgraph=True, mode='reduce-overhead')
    measure_fn = torch.compile(lambda x, p, ref, c: policy_gain_kl(actor(x), p, ref, c, weights),
                               fullgraph=True, mode='reduce-overhead')
    solver = ActorNaturalGradient(actor, budget=.03, cg_iterations=20, compile=True)
    for _ in range(2):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            logits = actor(obs)
            _, factor, parameters = beta_geometry(logits, (physical + 1) / 2)
            jac_beta = beta_fn((physical + 1) / 2, parameters, torch.full((3,), 2., device=device)).clone()
            next_value = predict(next_obs, age + 1).clone()
            target, qa, eta = (v.clone() for v in target_fn(obs, physical, jac_beta, next_value, weights))
            jac_action = jac_fn(obs, physical).clone()
        torch.compiler.cudagraph_mark_step_begin()
        optimizer.zero_grad(set_to_none=True)
        loss = fit_critic(obs, age, target, jac_action, jac_beta)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters())
        optimizer.step()
        del loss
        reference = beta_kl_reference(parameters)

        def measure():
            gain, kl = measure_fn(obs, parameters, reference, eta)
            return gain.clone(), kl.clone()

        torch.compiler.cudagraph_mark_step_begin()
        result = solver.step(obs, logits, factor, eta, weights, measure)
        assert 0 < result['policy/exact_joint_kl'] <= .03
        assert result['policy/accepted_gain'] > 0
