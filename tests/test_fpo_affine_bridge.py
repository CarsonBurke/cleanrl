"""Affine-flow endpoint, conditional expectation, and compiled update contracts."""
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import pytest
import torch
from cleanrl import ppo_continuous_action_fpo_standard_bridge_v5 as fpo

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')]


def agent_and_args(steps=10):
    torch.manual_seed(1)
    fpo.configure_runtime(matmul_precision='highest', allow_tf32=False)
    env = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                          single_action_space=gym.spaces.Box(-1., 1., shape=(2,), dtype=np.float32))
    args = fpo.Args(num_envs=2, num_steps=256, num_minibatches=4, flow_steps=steps)
    return fpo.Agent(env, args).cuda(), args


@pytest.mark.parametrize('steps', [1, 3, 10])
def test_zero_residual_exact_gaussian_endpoints_and_conditional_epsilon(steps):
    agent, _ = agent_and_args(steps)
    obs = torch.tensor([[.2, -.4, .7], [-.1, .5, -.6]], device='cuda')
    noise = torch.tensor([[1.2, -.7], [-.9, .6]], device='cuda')
    with torch.no_grad():
        agent.actor['affine'][-1].weight.zero_()
        # Nonunit variances and nonzero means exercise all affine terms.
        agent.actor['affine'][-1].bias.copy_(torch.tensor([.3, -.4, -1., .7], device='cuda'))
        mean, std = agent.affine_parameters(obs)
        endpoint, decoded = agent.sample(obs, noise)
        torch.testing.assert_close(endpoint, mean + std*noise, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(decoded, endpoint.tanh())
        t = torch.tensor([[.2], [.8]], device='cuda')
        x = torch.tensor([[.8, -.2], [-.5, .1]], device='cuda')
        # Bayes E[epsilon|x_t] for independent Gaussian epsilon and endpoint.
        expected_eps = t*(x-(1-t)*mean)/(t.square()+(1-t).square()*std.square())
        predicted_eps = x+(1-t)*agent.velocity(obs, x, t)
        torch.testing.assert_close(predicted_eps, expected_eps, rtol=2e-5, atol=2e-6)
    host = fpo.HostSampler(agent, 2, steps)
    z, action = host(obs.cpu().numpy(), None, noise=noise.cpu().numpy())
    np.testing.assert_allclose(z, endpoint.cpu().numpy(), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(action, decoded.cpu().numpy(), rtol=2e-5, atol=2e-6)


def test_nonzero_residual_host_refresh_and_compiled_sampling_parity():
    agent, args = agent_and_args()
    host = fpo.HostSampler(agent, 2, args.flow_steps)
    obs = torch.randn(2, 3, device='cuda')
    noise = torch.randn(2, 2, device='cuda')
    compiled = torch.compile(agent.sample, fullgraph=True, options={'triton.cudagraphs': False})
    for scale in (.02, -.05):
        with torch.no_grad():
            agent.actor['residual'][-1].weight.fill_(scale)
            agent.actor['affine'][-1].bias.add_(.1)
            z, action = compiled(obs, noise)
        host.refresh()
        hz, ha = host(obs.cpu().numpy(), None, noise=noise.cpu().numpy())
        np.testing.assert_allclose(hz, z.cpu().numpy(), rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(ha, action.cpu().numpy(), rtol=3e-5, atol=3e-6)


def test_fpo_reward_signal_can_contract_affine_noise_with_compiled_updates():
    agent, args = agent_and_args()
    args = fpo.validate_args(args)
    obs = torch.zeros(args.batch_size, 3, device='cuda')
    with torch.no_grad():
        for parameter in agent.actor['affine'].parameters():
            parameter.zero_()
        native = torch.randn(args.batch_size, 2, device='cuda')
    # Reward optimum is zero action: variance should decrease, not inflate.
    advantages = -native.square().sum(-1)
    cache = fpo.RolloutCFM(args, 2, 'cuda')
    cache.refresh(lambda s,a,t,e: (agent.get_value(s).flatten(), fpo.cfm_loss(agent,s,a,t,e,args.mse)),
                  obs, native, torch.Generator(device='cuda').manual_seed(2))
    old = cache.old_loss.clone()
    def objective(s, a, t, e, old_loss, adv, ret, old_v):
        return fpo.fpo_loss(agent,s,a,t,e,old_loss,adv,ret,old_v,args)
    compiled = torch.compile(objective, mode='reduce-overhead', fullgraph=True)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5, fused=True)
    with torch.no_grad():
        initial_std = agent.affine_parameters(obs)[1].mean().clone()
    for _ in range(4):
        torch.compiler.cudagraph_mark_step_begin()
        loss, _ = compiled(obs, native, cache.times, cache.noise, cache.old_loss,
                           advantages, cache.old_values, cache.old_values)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert agent.actor['affine'][-1].bias.grad[2:].sum() > 0
        assert agent.actor['residual'][-1].weight.grad.norm() > 0
        torch.nn.utils.clip_grad_norm_(agent.parameters(), .5)
        optimizer.step()
    with torch.no_grad():
        assert agent.affine_parameters(obs)[1].mean() < initial_std
    torch.testing.assert_close(cache.old_loss, old, rtol=0, atol=0)
