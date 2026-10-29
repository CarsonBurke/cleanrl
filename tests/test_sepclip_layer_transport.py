"""Analytic CUDA contracts for the local audit; run through mlq."""
import torch
import pytest
from torch.nn import functional as F

from scripts.audit_sepclip_layer_transport import (
    tangent, parallel_transport, retracted_delta, fisher_gram,
    distribution_motion, finite_distribution_motion,
    mode_allocation,
    component_groups, compatible_clock_state,
    validate_audit_protocol,
)


def test_product_sphere_projection_transport_and_retraction():
    torch.manual_seed(13)
    for dim in (0,1):
        old=F.normalize(torch.randn(5,7,device='cuda',dtype=torch.float64),dim=dim)
        vector=torch.randn_like(old)
        projected=tangent(old,vector,dim)
        torch.testing.assert_close((projected*old).sum(dim),torch.zeros(7 if dim==0 else 5,device='cuda',dtype=torch.float64),atol=1e-14,rtol=0)
        torch.testing.assert_close(tangent(old,projected,dim),projected,atol=1e-14,rtol=0)
        new=F.normalize(old+.01*projected,dim=dim)
        transported=parallel_transport(old,new,projected,dim)
        torch.testing.assert_close((transported*new).sum(dim),torch.zeros_like((transported*new).sum(dim)),atol=1e-14,rtol=0)
        torch.testing.assert_close(transported.norm(dim=dim),projected.norm(dim=dim),atol=1e-14,rtol=1e-14)
        epsilon=1e-5
        actual=retracted_delta(old,epsilon*vector,dim)/epsilon
        torch.testing.assert_close(actual,projected,atol=2e-4,rtol=2e-4)
        torch.testing.assert_close(vector,projected+old*(old*vector).sum(dim,keepdim=True),atol=1e-14,rtol=0)


def test_fisher_gram_accounts_for_cancellation_and_cross_terms():
    torch.manual_seed(14)
    direction=torch.randn(11,6,device='cuda',dtype=torch.float64)
    fisher=torch.eye(2,device='cuda',dtype=torch.float64).expand(11,3,2,2)
    changes=[direction,-.7*direction,.2*direction]
    gram=fisher_gram(changes,fisher)
    expected=.5*sum(changes).square().sum(-1).mean()
    torch.testing.assert_close(gram.sum(),expected,atol=1e-14,rtol=1e-14)
    assert gram[0,1]<0
    assert gram.sum()<gram.diagonal().sum()


def test_mean_concentration_jacobian_matches_small_finite_step():
    torch.manual_seed(15)
    logits=torch.randn(17,6,device='cuda',dtype=torch.float64)
    change=torch.randn_like(logits)*1e-6
    linear=distribution_motion(logits,change)
    finite=finite_distribution_motion(logits,logits+change)
    for key in ('mean_rms','log_concentration_rms','mean_log_concentration_change'):
        assert abs(linear[key]-finite[key])<1e-12
    assert finite['exact_kl']>=-1e-12
    native=torch.rand(17,3,device='cuda',dtype=torch.float64)*.8+.1
    advantage=torch.randn(17,device='cuda',dtype=torch.float64)
    fisher=torch.eye(2,device='cuda',dtype=torch.float64).expand(17,3,2,2)
    allocation=mode_allocation(logits,change,native,advantage,fisher)
    assert allocation['additive_relative_error']<1e-14


def test_real_agent_components_and_legacy_optimizer_metadata():
    from types import SimpleNamespace
    import gymnasium as gym
    import numpy as np
    from cleanrl.ppo_continuous_action_sepclip_rollout_clock_anvil_v7 import Agent, Args, make_optimizers
    descriptor = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf,np.inf,(17,),dtype=np.float32),
                                 single_action_space=gym.spaces.Box(-1.,1.,(6,),dtype=np.float32))
    agent = Agent(descriptor).cuda()
    args = Args()
    args.num_iterations = 488
    args.batch_size = args.minibatch_size = 16384
    matrix, _ = make_optimizers(agent,args)
    names = {parameter: name for name,parameter in agent.actor.named_parameters()}
    hidden = {names[p] for _,parameters,_,_ in matrix.stacks for p in parameters}
    groups = component_groups(names.values(),hidden)
    assert all(group for _,group in groups)
    assert len(groups) == 9
    assert set(name for _,group in groups for name in group) == set(names.values())
    saved = matrix.state_dict()
    for group in saved['param_groups']:
        for key in ('slow_clock','rollout_pending','rollout_decay','rollouts'):
            group.pop(key)
    restored = compatible_clock_state(saved,'epoch',10)
    assert 'slow_clock' not in saved['param_groups'][0]
    matrix.load_state_dict(restored)
    matrix.begin_rollout(10)
    for parameter in matrix.param_groups[0]['params']:
        parameter.grad = torch.ones_like(parameter)
    matrix.step()
    assert matrix.param_groups[0]['slow_clock'] == 'epoch'


@pytest.mark.parametrize('change', [{'norm_adv': True}, {'ent_coef': .01}])
def test_reject_loss_protocols_inconsistent_with_raw_pg_reports(change):
    from types import SimpleNamespace
    args = SimpleNamespace(num_minibatches=1, normuon_network='actor', matrix_method='anvil2',
                           norm_adv=False, ent_coef=0.)
    validate_audit_protocol(args)
    for name, value in change.items():
        setattr(args, name, value)
    with pytest.raises(ValueError, match='raw policy-gradient audit'):
        validate_audit_protocol(args)


def test_tiny_float32_policy_step_kl_matches_shape_space_fisher():
    logits=torch.linspace(-12,12,132,device='cuda').reshape(11,12)
    after=logits+torch.linspace(-1,1,132,device='cuda').reshape(11,12)*.0001
    a,b=(1+F.softplus(logits)).chunk(2,-1)
    na,nb=(1+F.softplus(after)).chunk(2,-1)
    a,b,na,nb=(x.double() for x in (a,b,na,nb))
    da,db=na-a,nb-b
    common=torch.polygamma(1,a+b)
    quadratic=.5*((torch.polygamma(1,a)-common)*da.square()
                  +(torch.polygamma(1,b)-common)*db.square()-2*common*da*db).sum(-1).mean()
    result=finite_distribution_motion(logits,after)
    assert result['exact_kl_numeric_dtype']=='float64'
    assert result['policy_shape_dtype']=='torch.float32'
    assert result['exact_kl']>0
    assert abs(result['exact_kl']/float(quadratic)-1)<.01
