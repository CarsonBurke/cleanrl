"""Closed-loop V2 diagnostic contracts; execute only as queued CUDA work."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

spec=importlib.util.spec_from_file_location('closed_loop_v2',Path(__file__).parents[1]/'scripts/audit_sepclip_closed_loop_update_v2.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture(scope='module')
def captured():
    configure_runtime(matmul_precision='highest',allow_tf32=False)
    torch._dynamo.config.recompile_limit=64
    assert torch.cuda.is_available(), 'Run these contracts through mlq on CUDA'
    envs=audit.make_mujoco_vector_env('HalfCheetah-v4',2,num_threads=2)
    try:
        raw,_=envs.reset(seed=1)
        normalizer=VectorObsNorm(2,(17,))
        reward_norm=VectorRewardNorm(2,.99)
        obs=normalizer.normalize(raw)
        torch.manual_seed(1)
        agent=audit.trainer.Agent(envs).cuda().eval()
        agent.normalize_matrices()
        capture=audit.trainer.TransitionCapture(envs,6)
        observations=[];natives=[]
        rng=np.random.default_rng(19)
        for t in range(6):
            observations.append(obs.copy())
            native=rng.uniform(.2,.8,(2,6)).astype(np.float32)
            natives.append(native.copy())
            action=(2*native-1).astype(np.float32)
            capture.before(t,envs,action,normalizer)
            raw,reward,term,trunc,infos=envs.step(action)
            reward_norm.normalize(reward,term)
            obs,transition=normalizer.normalize_step(raw,term,trunc,infos)
            capture.after(t,raw,reward,obs,transition,reward_norm)
        snapshot=dict(global_step=12,args=dict(env_id='HalfCheetah-v4',gamma=.99,
            update_epochs=2,batch_size=1,minibatch_size=1),
            model={name:p.detach().cpu().clone() for name,p in agent.state_dict().items()},
            transitions=capture.arrays,
            batch=dict(obs=np.asarray(observations).reshape(-1,17),native=np.asarray(natives).reshape(-1,6),
                advantages=np.zeros(12,dtype=np.float32),values=np.zeros(12,dtype=np.float32)))
        yield snapshot,agent
    finally:
        envs.close()


def make_trace(snapshot,agent,change=0.):
    deltas={name:torch.zeros_like(p).cpu() for name,p in agent.named_parameters()}
    deltas['actor.head.0.bias'].fill_(change)
    endpoint={name:p.clone() for name,p in snapshot['model'].items()}
    for _ in range(2):
        for name,d in deltas.items():endpoint[name].add_(d)
    return dict(global_step=snapshot['global_step'],model_after_update=endpoint,
        trace=[dict(update=i,epoch=i,retracted_delta=deltas) for i in range(2)])


def test_hybrid_switches_only_intended_time():
    np.testing.assert_array_equal(audit.arm_policy_indices(0,True),[0,1,2,1,0,2,0])
    np.testing.assert_array_equal(audit.arm_policy_indices(9,True),[0,1,2,0,1,0,2])
    np.testing.assert_array_equal(audit.arm_policy_indices(0,False),[0,1,2])


def test_strict_reconstruction_and_compiled_bank_dispatch(captured):
    snapshot,agent=captured
    banks,deltas,report=audit.reconstruct_banks(agent,snapshot,make_trace(snapshot,agent,.001))
    assert report['cumulative_delta_vs_exact_endpoint_max_abs_error']==0
    assert report['unit_axis_norm_max_abs_errors']['endpoint']<1e-6
    policy=audit.compiled_policy(agent,banks,2,7)
    x=torch.as_tensor(np.repeat(snapshot['batch']['obs'][:2],7,axis=0),device='cuda')
    mapping=torch.as_tensor(audit.arm_policy_indices(0,True),device='cuda')
    with torch.no_grad():
        selected=policy(x,mapping)[0].clone().reshape(2,7,12)
        old=policy(x,torch.zeros(7,dtype=torch.int64,device='cuda'))[0].clone().reshape(2,7,12)
    shift=selected-old
    expected=np.array([0,.001,.002,.001,0,.002,0])
    np.testing.assert_allclose(shift.cpu().numpy(),np.broadcast_to(expected[None,:,None],(2,7,12)),atol=2e-6,rtol=0)
    directional=audit.compiled_directional(agent,banks,deltas)
    with torch.no_grad():
        z,first,end,value=directional(x.reshape(2,7,17)[:,0].contiguous())
    np.testing.assert_allclose(first.cpu().numpy(),.001,atol=1e-6,rtol=0)
    np.testing.assert_allclose(end.cpu().numpy(),.002,atol=1e-6,rtol=0)
    assert value.shape==(2,)
    np.testing.assert_allclose(z.cpu().numpy(),old[:,0].cpu().numpy(),atol=2e-6,rtol=0)
    broken=make_trace(snapshot,agent)
    broken['global_step']+=1
    with pytest.raises(ValueError,match='global_step mismatch'):
        audit.reconstruct_banks(agent,snapshot,broken)


def test_identical_policies_have_exact_paired_physics_and_zero_pg(captured):
    snapshot,agent=captured
    banks,deltas,_=audit.reconstruct_banks(agent,snapshot,make_trace(snapshot,agent))
    states=audit.dense.select_states(snapshot,count=2,max_age=400)
    cli=SimpleNamespace(hybrid_arms=True,seed=1,horizons=[1,16],gamma=.99,directional_pg=True,
        gae_transport=True,gae_lambda=.95,replicates=2)
    policy=audit.compiled_policy(agent,banks,2,7)
    directional=audit.compiled_directional(agent,banks,deltas)
    result=audit.simulate(agent,banks,deltas,states,cli,'sample',0,policy,directional)
    assert result['restore_max_abs_error']==0
    for horizon in cli.horizons:
        row=result['horizons'][horizon]
        for name in ('raw_discounted','raw_undiscounted','normalized_discounted','velocity_mean','control_cost_mean'):
            np.testing.assert_array_equal(row[name],np.repeat(row[name][:,:1],7,axis=1))
        np.testing.assert_array_equal(row['directional_pg']['raw'],np.zeros((2,2)))
        np.testing.assert_array_equal(row['raw_tailaug_discounted'],np.repeat(row['raw_tailaug_discounted'][:,:1],7,axis=1))
        assert row['gae_transport']['normalized']['gae_one_minus_augmented_mc_max_abs_error']==0
        assert max(abs(x) for arm in row['occupancy_kl']['old_to_endpoint'].values() for x in arm)<1e-10
    summary=audit.summarize_mode([result,result],cli,audit.HYBRID_ARMS,'sample')
    for row in summary['horizons']:
        feedback=row['metrics']['raw_discounted']['feedback_decomposition']
        assert feedback['first_all']['interaction']['selected_state_average_mean']==0
        assert feedback['endpoint_all']['interaction']['selected_state_average_mean']==0


def test_directional_pg_sums_all_times_and_discounted_future():
    rewards=np.array([[2.],[4.],[8.]])
    scores=np.array([[[1.,0.]],[[0.,2.]],[[3.,4.]]])
    baseline=np.ones_like(rewards)
    controlled,uncontrolled=audit.finite_directional_pg(rewards,baseline,scores,.5)
    # G=[6,8,8]; gamma^t=[1,.5,.25], each time has its own score.
    np.testing.assert_array_equal(uncontrolled,[[12.,16.]])
    np.testing.assert_array_equal(controlled,[[10.25,14.]])
    summary=audit.conditional_summary(np.array([[1.,3.],[3.,5.],[5.,7.],[7.,9.]]))
    assert summary['selected_state_average_mean']==5
    assert summary['effective_noise_replicates']==4
    assert summary['student_t_critical']>3


def test_matched_gae_terminal_tail_and_future_error_transport():
    rng=np.random.default_rng(9)
    rewards=rng.normal(size=(5,2));values=rng.normal(size=(5,2))
    terminal=rng.normal(size=2);scores=rng.normal(size=(5,2,2))
    gamma,lam=.99,.95
    result=audit.matched_gae_transport(rewards,values,terminal,scores,gamma,lam)
    finite,_=audit.finite_directional_pg(rewards,values,scores,gamma)
    np.testing.assert_allclose(result['augmented_mc_directional_pg']-finite,
        result['terminal_tail_directional_pg'],atol=2e-14,rtol=0)
    assert result['gae_one_minus_augmented_mc_max_abs_error']<2e-14
    # On this deterministic realized path, the augmented return is an exact
    # reference value with the SAME terminal reward. Future reference error
    # transport has gamma*(1-lambda)*(gamma*lambda)^(j-1) coefficients.
    reference=np.empty_like(rewards)
    tail=terminal.copy()
    for t in range(4,-1,-1):
        tail=rewards[t]+gamma*tail
        reference[t]=tail
    error=values-reference
    predicted=np.zeros((2,2))
    for t in range(5):
        future_error=np.zeros(2)
        for j in range(1,5-t):
            future_error+=gamma*(1-lam)*(gamma*lam)**(j-1)*error[t+j]
        predicted+=gamma**t*scores[t]*future_error[:,None]
    np.testing.assert_allclose(result['gae_minus_augmented_mc_directional_pg'],predicted,atol=2e-14,rtol=0)


@pytest.mark.parametrize('name', ['rewards','baseline','scores'])
@pytest.mark.parametrize('bad', [float('nan'),float('inf')])
def test_directional_pg_rejects_nonfinite_required_trace(name,bad):
    fields=dict(rewards=np.ones((3,2)),baseline=np.zeros((3,2)),scores=np.ones((3,2,2)))
    fields[name].flat[-1]=bad
    with pytest.raises(ValueError,match=f'Finite directional PG: nonfinite {name}'):
        audit.finite_directional_pg(**fields,gamma=.99)


@pytest.mark.parametrize('bad', [float('nan'),float('inf')])
def test_matched_gae_rejects_invalid_terminal_even_when_discount_is_zero(bad):
    with pytest.raises(ValueError,match='Matched GAE transport: nonfinite terminal_value'):
        audit.matched_gae_transport(np.ones((3,2)),np.zeros((3,2)),np.array([0.,bad]),
            np.ones((3,2,2)),gamma=0.,gae_lambda=.95)


def test_closed_loop_rejects_nonfinite_policy_before_draw_or_step(captured):
    snapshot,agent=captured
    banks,deltas,_=audit.reconstruct_banks(agent,snapshot,make_trace(snapshot,agent))
    states=audit.dense.select_states(snapshot,count=2,max_age=400)
    cli=SimpleNamespace(hybrid_arms=False,seed=1,horizons=[1],gamma=.99,
                        directional_pg=False,gae_transport=False,replicates=2)
    def invalid_policy(x,indices):
        logits=torch.full((len(x),12),float('nan'),device='cuda')
        shapes=torch.ones((len(x),6),device='cuda')*3
        return logits,shapes,shapes
    with pytest.raises(ValueError,match='Closed-loop policy step 0: nonfinite logits'):
        audit.simulate(agent,banks,deltas,states,cli,'sample',0,invalid_policy,None)


@pytest.mark.parametrize('layout', ['contiguous_views', 'strided_views'])
def test_jvp_materializes_private_primal_and_tangent_storage(captured,layout):
    snapshot,agent=captured
    original,directions,_=audit.reconstruct_banks(agent,snapshot,make_trace(snapshot,agent,.001))
    def view(value):
        if layout=='contiguous_views':
            result=torch.stack((value,value))[0]
        else:
            storage=torch.empty((*value.shape[:-1],2*value.shape[-1]),device='cuda',dtype=value.dtype)
            result=storage[...,::2]
            result.copy_(value)
        assert result._base is not None
        return result
    banks={name:view(value) for name,value in original.items()}
    deltas=tuple({name:view(value) for name,value in bank.items()} for bank in directions)
    # Bank[0] can be a contiguous VIEW: contiguous() alone need not detach it.
    if layout=='contiguous_views':assert all(value[0].is_contiguous() for value in banks.values())
    directional=audit.compiled_directional(agent,banks,deltas)
    x=torch.tensor(snapshot['batch']['obs'][-2:],device='cuda').T.contiguous().T
    assert not x.is_contiguous()
    with torch.no_grad():initial=tuple(value.clone() for value in directional(x))
    np.testing.assert_allclose(initial[1].cpu().numpy(),.001,atol=1e-6,rtol=0)
    np.testing.assert_allclose(initial[2].cpu().numpy(),.002,atol=1e-6,rtol=0)
    for value in banks.values():value[0].add_(5)
    for bank in deltas:
        for value in bank.values():value.add_(1)
    # Evaluate owns its primal/tangents; caller storage mutation cannot retarget
    # the supposedly fixed physical directional trace after construction.
    with torch.no_grad():later=tuple(value.clone() for value in directional(x))
    for before,after in zip(initial,later):assert torch.equal(before,after)


def test_compiled_body_jvp_matches_separately_compiled_centered_chord(captured):
    snapshot,agent=captured
    banks,_,_=audit.reconstruct_banks(agent,snapshot,make_trace(snapshot,agent))
    old={name:value[0].detach().clone(memory_format=torch.contiguous_format) for name,value in banks.items()}
    delta={name:torch.zeros_like(value) for name,value in old.items()}
    name='first.0.gate.weight'
    delta[name]=(.002*torch.arange(old[name].numel(),device='cuda',dtype=torch.float32).sin()).reshape(old[name].shape)
    endpoint={name:2*value for name,value in delta.items()}
    directional=audit.compiled_directional(agent,banks,(delta,endpoint))
    x=torch.tensor(snapshot['batch']['obs'][-2:],device='cuda')
    def primal(parameters,observations):
        return torch.func.functional_call(agent.actor,parameters,(observations,),strict=True)
    independent=torch.compile(primal,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    with torch.no_grad():
        _,first,end,_=tuple(value.clone() for value in directional(x))
        positive=independent({name:value+.5*delta[name] for name,value in old.items()},x).clone()
        negative=independent({name:value-.5*delta[name] for name,value in old.items()},x).clone()
    finite=(positive.double()-negative.double()) # denominator2*.5=1
    assert float(first.double().norm())>1e-7
    assert float((finite-first.double()).norm()/first.double().norm())<.05
    assert torch.allclose(end,2*first,atol=3e-6,rtol=1e-4)


@pytest.mark.parametrize('bad', [float('nan'),float('inf')])
def test_compiled_directional_rejects_nonfinite_observations_before_graph(captured,bad):
    snapshot,agent=captured
    banks,deltas,_=audit.reconstruct_banks(agent,snapshot,make_trace(snapshot,agent))
    directional=audit.compiled_directional(agent,banks,deltas)
    x=torch.tensor(snapshot['batch']['obs'][:2],device='cuda')
    x[0,0]=bad
    with pytest.raises(ValueError,match='observations must be finite'):
        directional(x)
