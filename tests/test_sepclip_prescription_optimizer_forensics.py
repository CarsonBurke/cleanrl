"""Actual-source synthetic nonzero warm continuation contracts; MLQ/CUDA only."""
import copy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_sepclip_additive_prerms_anvil_v14 as module
from cleanrl.shared.runtime import configure_runtime
from scripts.audit_sepclip_prescription_optimizer_forensics import (
    audit_optimizer,delta_contract,endpoint_contract,factory_mapping,private_warm,
    reconstruct_stages,require_factory_order,state_tree_equal,validate_clock,
)


@pytest.fixture(autouse=True)
def cuda_only():
    assert torch.cuda.is_available(),'Submit contracts through MLQ; CUDA mandatory'
    configure_runtime(matmul_precision='highest',allow_tf32=False)


def capture_fixture(method):
    """Source recording independent of audit, with nonzero rails/Adam moments.

    Synthetic clock-prefixed history is not an actual trained checkpoint. Two
    real source warm rollouts create nonzero moments, then counters are bound
    to a valid later rollout boundary to exercise the scheduled two-rail blend.
    """
    torch.manual_seed(193)
    args=module.validate_args(module.Args(num_envs=2,num_steps=2,update_epochs=10,
        actor_architecture='additive_prerms',actor_optimizer='adam' if method=='adam' else 'anvil2',
        matrix_method='anvil2' if method=='adam' else method,normuon_network='actor',
        normuon_lr=.005,normuon_momentum=.85,normuon_beta2=.95 if method=='normuon' else .9,
        matrix_precision='bf16',compile=True,staggered_starts=False,total_timesteps=2000))
    args.num_iterations=500
    env=SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf,np.inf,(17,),dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.,1.,(6,),dtype=np.float32))
    agent=module.Agent(env,'additive_prerms').cuda()
    normalize=torch.compile(agent.normalize_matrices,fullgraph=True,options={'triton.cudagraphs':False})
    normalize()
    matrix,adam=module.make_optimizers(agent,args)
    def gradients(step):
        # Actual all-parameter loss gradient from a linear scalar objective;
        # source-style actor/critic clipping is done at recording, never replay.
        agent.zero_grad(set_to_none=True)
        loss=sum((p*torch.sin(torch.arange(p.numel(),device='cuda',dtype=torch.float32).reshape(p.shape)+step)*.001).sum() for p in agent.parameters())
        loss.backward()
        for trunk in (agent.actor,agent.critic):torch.nn.utils.clip_grad_norm_(tuple(trunk.parameters()),.5,foreach=True)
    for rollout in range(2):
        if matrix is not None:matrix.begin_rollout(10)
        for index in range(10):
            gradients(rollout*10+index)
            if matrix is not None:matrix.step()
            adam.step();normalize()
    prior_steps=2500;iteration=251;fraction=1-(iteration-1)/args.num_iterations
    for group in adam.param_groups:
        group['lr']=fraction*args.learning_rate
        for p in group['params']:adam.state[p]['step'].fill_(prior_steps)
    if matrix is not None:
        for group in matrix.param_groups:group['steps']=prior_steps;group['rollouts']=250
        matrix.set_lr(fraction*args.normuon_lr)
    snapshot=dict(model=copy.deepcopy(agent.state_dict()),optimizer=copy.deepcopy(adam.state_dict()),
        matrix_optimizer=copy.deepcopy(matrix.state_dict()) if matrix is not None else None,
        global_step=iteration*args.batch_size)
    entries=[]
    if matrix is not None:matrix.begin_rollout(10)
    for index in range(10):
        gradients(31+index)
        old={name:p.detach().clone() for name,p in agent.named_parameters()}
        captured={name:p.grad.detach().clone() for name,p in agent.named_parameters()}
        if matrix is not None:matrix.step()
        adam.step()
        ambient={name:p.detach().clone()-old[name] for name,p in agent.named_parameters()}
        normalize()
        entries.append(dict(epoch=index,update=index,gradient_after_clip=captured,ambient_delta=ambient,
            retracted_delta={name:p.detach().clone()-old[name] for name,p in agent.named_parameters()}))
    endpoint=copy.deepcopy(agent.state_dict())
    agent.load_state_dict(snapshot['model'],strict=True)
    return SimpleNamespace(module=module,args=args,agent=agent,snapshot=snapshot,
        trace=dict(trace=entries,model_after_update=endpoint),validation={})


@pytest.mark.parametrize('method',['adam','anvil2','normuon'])
def test_actual_source_nonzero_warm_complete_trace_and_private_state(method):
    replay=capture_fixture(method);before=copy.deepcopy(replay.snapshot)
    result=audit_optimizer(replay)
    assert result['status']=='accepted_actual_update_reconstruction'
    assert len(result['epochs'])==10
    assert state_tree_equal(replay.snapshot,before)
    assert all(row['bitwise_equal'] for row in result['endpoint_parity'].values())
    assert all(check['bitwise_equal'] for row in result['epochs'] for check in row['ambient_parity'].values())
    assert all(check['bitwise_equal'] for row in result['epochs'] for check in row['actual_parity'].values())
    assert any(row['exp_avg_norm']>0 for row in result['epochs'][0]['auxiliary_adam'].values())
    if method!='adam':
        banks=result['factory_mapping']['matrix_banks']
        assert sorted(len(bank['parameter_names']) for bank in banks)==[1,2,4]
        assert len(result['factory_mapping']['serialized_parameter_ids']['matrix'])==7
        first=result['epochs'][0]['matrix_banks']
        assert all(bank['prior_rollout_pending'] for bank in first)
        assert all(not bank['prior_rollout_pending'] for row in result['epochs'][1:] for bank in row['matrix_banks'])
        for row in result['epochs']:
            for bank in row['matrix_banks']:
                assert bank['after_steps']==2500+row['epoch']+1
                for member in bank['members'].values():
                    if method=='normuon':assert member['slow_rail_after_norm']==0
                    elif row['epoch']>0:assert member['slow_rail_after_norm']==member['slow_rail_before_norm']


@pytest.mark.parametrize('corruption',['gradient','ambient','actual','endpoint','missing_history','matrix_order','Adam_order','unused_rail','LR','matrix_LR','clock','pending'])
def test_captured_replay_failclosed_corruptions(corruption):
    replay=capture_fixture('normuon')
    if corruption in ('gradient','ambient','actual'):
        field={'gradient':'gradient_after_clip','ambient':'ambient_delta','actual':'retracted_delta'}[corruption]
        replay.trace['trace'][1][field]['actor.trunk.in_proj.weight'][0,0]+=.01
    elif corruption=='endpoint':replay.trace['model_after_update']['actor.readout_gain'][0]+=.01
    elif corruption=='missing_history':replay.snapshot['optimizer']['state'].clear()
    elif corruption in ('matrix_order','Adam_order'):
        which='matrix_optimizer' if corruption=='matrix_order' else 'optimizer'
        group=replay.snapshot[which]['param_groups'][0]['params']
        # Actual same-shape routes, so dimensions alone cannot catch reordering.
        first,second=(1,2) if corruption=='matrix_order' else (0,2)
        group[first],group[second]=group[second],group[first]
    elif corruption=='unused_rail':next(iter(replay.snapshot['matrix_optimizer']['state'].values()))['velocity'][1].fill_(.01)
    elif corruption=='LR':replay.snapshot['optimizer']['param_groups'][0]['lr']*=2
    elif corruption=='matrix_LR':replay.snapshot['matrix_optimizer']['param_groups'][0]['lr'].mul_(2)
    elif corruption=='pending':replay.snapshot['matrix_optimizer']['param_groups'][0]['rollout_pending']=True
    else:replay.snapshot['matrix_optimizer']['param_groups'][0]['steps']+=1
    with pytest.raises(ValueError):audit_optimizer(replay)


def test_private_warm_moment_arrays_do_not_alias_original_snapshot():
    replay=capture_fixture('anvil2');before=copy.deepcopy(replay.snapshot)
    private,matrix,adam=private_warm(replay)
    for _,_,state,_ in matrix.stacks:state['velocity'].add_(1);state['second'].add_(1)
    for state in adam.state.values():state['exp_avg'].add_(1);state['exp_avg_sq'].add_(1)
    with torch.no_grad():next(private.agent.parameters()).add_(1)
    assert state_tree_equal(replay.snapshot,before)
    assert state_tree_equal(replay.agent.state_dict(),before['model'])


def test_delta_gate_rejects_small_motion_and_exact_zero_corruption():
    base=torch.ones(2,device='cuda');recorded=torch.tensor([1e-9,0.],device='cuda');bad=torch.tensor([0.,0.],device='cuda')
    with pytest.raises(ValueError):delta_contract(bad,recorded,base,'tiny')
    with pytest.raises(ValueError):delta_contract(recorded,torch.zeros_like(recorded),base,'null',require_exact=False)
    perturbed=torch.nextafter(recorded,torch.ones_like(recorded))
    with pytest.raises(ValueError):delta_contract(perturbed,recorded,base,'one_ulp')
    # Internal-stage tolerance cannot relax the actual captured-trace gate.
    assert delta_contract(recorded,recorded,base,'same')['bitwise_equal']


def test_endpoint_gate_rejects_one_ulp_recorded_endpoint_change():
    old=torch.ones(2,device='cuda');bad=old.clone();bad[0]=torch.nextafter(bad[0],torch.tensor(2.,device='cuda'))
    with pytest.raises(ValueError):endpoint_contract(old,bad,'one_ulp')


def test_factory_serialized_order_rejects_same_shape_permutation():
    fresh={'param_groups':[{'params':[0,1,2]}]};saved={'param_groups':[{'params':[0,2,1]}]}
    with pytest.raises(ValueError):require_factory_order(fresh,saved,'same_shape')


def test_stale_saved_scalar_cache_is_rebuilt_without_changing_current_lr():
    replay=capture_fixture('normuon')
    replay.snapshot['matrix_optimizer']['param_groups'][0]['scalars'].fill_(1e8)
    result=audit_optimizer(replay)
    assert result['clock']['current_lr_fraction']==.5
    assert result['epochs'][0]['matrix_banks'][0]['source_schedule_scalars'][0]==pytest.approx(.0025)
    assert all(row['bitwise_equal'] for row in result['endpoint_parity'].values())


def test_single_environment_staggered_flag_does_not_imply_source_warmup():
    args=SimpleNamespace(num_minibatches=1,update_epochs=10,target_kl=None,compile=True,
        staggered_starts=True,num_envs=1,batch_size=4,num_iterations=500,anneal_lr=True,
        learning_rate=.0024,env_id='HalfCheetah-v4')
    private=SimpleNamespace(args=args,module=module,snapshot={'global_step':4})
    assert validate_clock(private,None,SimpleNamespace(param_groups=[]))['iteration']==1


@pytest.mark.parametrize('method,precision',[('anvil2','bf16'),('normuon','bf16'),('anvil2','fp32')])
def test_reconstructed_stage_master_subtraction_against_frozen_actual_kernel(method,precision):
    torch.manual_seed(71)
    w=torch.randn(2,7,3,device='cuda');g=torch.randn_like(w)*.03
    rails=torch.randn(2,2,7,3,device='cuda')*.02;second=torch.rand(2,7,1,device='cuda')+.2
    scalars=torch.tensor([.002,.93,.85,.4385,.98**10],device='cuda')
    original=(w.clone(),rails.clone(),second.clone());beta2=.9;momentum=.85;scale=(7/3)**.5
    actual=torch.compile(module.matrix_update,fullgraph=True,dynamic=False)
    actual(w,rails,second,g,scalars,beta2,scale,method,precision,momentum,'rollout')
    if method=='normuon':scalars[1]=momentum
    def stages(w,g,v,s,c):return reconstruct_stages(module,w,g,v,s,c,beta2,scale,method,precision)
    graph=torch.compile(stages,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    values=tuple(x.clone() for x in graph(original[0],g,rails,original[2],scalars))
    delta_contract(values[-1]-original[0],w-original[0],original[0],'independent_stages',require_exact=False)
    torch.testing.assert_close(values[3],second,rtol=2e-6,atol=0)
    if method=='anvil2' and precision=='bf16':
        torch.testing.assert_close(values[6],values[5].bfloat16().float(),rtol=0,atol=0)
    else:torch.testing.assert_close(values[6],values[5],rtol=0,atol=0)
