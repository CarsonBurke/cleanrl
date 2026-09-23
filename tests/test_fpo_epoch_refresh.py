"""Refreshing MC draws must not refresh the old policy or its value targets."""
import copy
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import pytest
import torch
from cleanrl import ppo_continuous_action_fpo_epoch_refresh_v6 as fpo
pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')]


def test_epoch_refresh_retains_nonunit_policy_ratio_after_update():
    torch.manual_seed(1)
    fpo.configure_runtime(matmul_precision='highest', allow_tf32=False)
    args = fpo.validate_args(fpo.Args(num_envs=2, num_steps=16, num_minibatches=2))
    env = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(3,),dtype=np.float32),
                          single_action_space=gym.spaces.Box(-1.,1.,shape=(2,),dtype=np.float32))
    agent = fpo.Agent(env,args).cuda()
    old = copy.deepcopy(agent).requires_grad_(False)
    obs = torch.randn(args.batch_size,3,device='cuda')
    actions = torch.randn(args.batch_size,2,device='cuda')
    cache = fpo.RolloutCFM(args,2,'cuda')
    statistics = fpo.graph_compile(lambda s,a,t,e: (old.get_value(s).flatten(),fpo.cfm_loss(old,s,a,t,e,args.mse)))
    rng = torch.Generator(device='cuda').manual_seed(51)
    cache.refresh(statistics,obs,actions,rng)
    old_values, initial_noise = cache.old_values.clone(), cache.noise.clone()
    with torch.no_grad():
        agent.actor[-1].bias.add_(.2)
        agent.critic[-1].bias.add_(.3)
    for _ in range(3):
        cache.refresh(statistics,obs,actions,rng)
        with torch.no_grad():
            expected_old = fpo.cfm_loss(old,obs,actions,cache.times,cache.noise,args.mse)
            new = fpo.cfm_loss(agent,obs,actions,cache.times,cache.noise,args.mse)
        torch.testing.assert_close(cache.old_loss,expected_old)
        torch.testing.assert_close(cache.old_values,old_values,rtol=0,atol=0)
        assert not torch.equal(initial_noise,cache.noise)
        assert (cache.old_loss-new).abs().max() > .01
        initial_noise.copy_(cache.noise)
    assert all(p.grad is None for p in old.parameters())
