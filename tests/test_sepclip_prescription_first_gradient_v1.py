"""CUDA source-PPO first-gradient bridge contracts; execute only through MLQ."""
import copy
import importlib
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.shared.runtime import configure_runtime
from scripts.audit_sepclip_prescription_first_gradient_v1 import (
    audit_first_gradient,exact_first_permutation,full_covector_summary,
    policy_ascent,tensor_bytes_hash,
)


@pytest.fixture(autouse=True)
def cuda_only():
    assert torch.cuda.is_available(),'Submit contracts through MLQ; CUDA mandatory'
    configure_runtime(matmul_precision='highest',allow_tf32=False)


def capture_fixture(version='v14',architecture='additive_prerms',norm_adv=False):
    name=('cleanrl.ppo_continuous_action_sepclip_additive_prerms_anvil_v14' if version=='v14'
          else 'cleanrl.ppo_continuous_action_sepclip_constraint_transport_v15')
    module=importlib.import_module(name)
    torch.manual_seed(617)
    kwargs=dict(num_envs=2,num_steps=16,num_minibatches=1,update_epochs=10,compile=True,
        norm_adv=norm_adv,ent_coef=0.,clip_vloss=False,staggered_starts=False)
    if version=='v14':kwargs['actor_architecture']=architecture
    args=module.validate_args(module.Args(**kwargs));args.num_iterations=500
    env=SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf,np.inf,(17,),dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.,1.,(6,),dtype=np.float32))
    agent=(module.Agent(env,architecture) if version=='v14' else module.Agent(env)).cuda()
    normalize=torch.compile(agent.normalize_matrices,fullgraph=True,options={'triton.cudagraphs':False});normalize()
    obs=torch.randn(args.batch_size,17,device='cuda')*1.7
    native=torch.rand(args.batch_size,6,device='cuda').clamp(.01,.99)
    with torch.no_grad():
        alpha,beta,values=agent.get_policy_and_value(obs)
        logprob=agent.action_logprob(alpha,beta,native).clone();values=values.flatten().clone()
    advantage=torch.randn(args.batch_size,device='cuda')*torch.linspace(.1,3.,args.batch_size,device='cuda')
    batch=dict(obs=obs,native=native,logprob=logprob,advantages=advantage,returns=values+advantage,values=values)
    shuffle=torch.Generator(device='cuda').manual_seed(991)
    saved=shuffle.get_state().clone()
    # Independent source recording uses its own continuing generator, not the
    # audit's exact_first_permutation or its actor-only objective.
    indices=module.device_minibatches(args.batch_size,args.minibatch_size,'cuda',shuffle)[0]
    def loss_model(obs,native,old_logprob,advantages,returns,old_values):
        return module.ppo_loss(agent,obs,native,old_logprob,advantages,returns,old_values,args)
    compiled=torch.compile(loss_model,mode=args.compile_mode,fullgraph=True,dynamic=False)
    torch.compiler.cudagraph_mark_step_begin()
    loss,_=compiled(*(batch[name][indices] for name in ('obs','native','logprob','advantages','returns','values')))
    agent.zero_grad(set_to_none=True);loss.backward()
    for trunk in (agent.actor,agent.critic):torch.nn.utils.clip_grad_norm_(tuple(trunk.parameters()),args.max_grad_norm,foreach=True)
    gradients={name:p.grad.detach().clone() for name,p in agent.named_parameters()}
    assert sum(float(value.double().square().sum()) for key,value in gradients.items() if key.startswith('actor.'))>0
    assert sum(float(value.double().square().sum()) for key,value in gradients.items() if key.startswith('critic.'))>0
    snapshot=dict(model=copy.deepcopy(agent.state_dict()),batch=batch,rollout_start_rng={'shuffle':saved})
    return SimpleNamespace(module=module,args=args,agent=agent,snapshot=snapshot,
        trace={'trace':[dict(epoch=0,update=0,gradient_after_clip=gradients)]}),indices


@pytest.mark.parametrize('version,architecture,norm_adv',[
    ('v14','additive_prerms',False),('v14','original',False),('v15','original',False),
    ('v14','additive_prerms',True)])
def test_actual_source_PPO_first_gradient_and_independent_covector_scope(version,architecture,norm_adv):
    replay,indices=capture_fixture(version,architecture,norm_adv)
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()
    model=copy.deepcopy(replay.agent.state_dict());saved=replay.snapshot['rollout_start_rng']['shuffle'].clone()
    result=audit_first_gradient(replay)
    assert result['status']=='accepted_bitwise_source_first_gradient_reconstruction'
    assert result['rng']['first_permutation_sha256']==tensor_bytes_hash(indices)
    assert all(row['bitwise_equal'] for row in result['first_gradient_parity'].values())
    assert result['conventions']['advantage_normalization']==norm_adv
    assert result['source_reconstructed_preclip_norms']['actor']>0
    assert result['source_reconstructed_preclip_norms']['critic']>0
    assert result['full_actor_covector']['common_scale_positive']
    assert result['full_actor_covector']['cosine']>0
    # No arbitrary numerical residual cutoff turns the independent graph into
    # a certificate. Inspect/export the measured FULL covector difference.
    assert result['full_actor_covector']['residual_after_common_scale_norm']>=0
    assert set(result['actor_parameters'])==set(dict(replay.agent.actor.named_parameters()))
    assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
    assert torch.equal(saved,replay.snapshot['rollout_start_rng']['shuffle'])
    assert all(torch.equal(value,replay.agent.state_dict()[name]) for name,value in model.items())


@pytest.mark.parametrize('corruption',['actor_gradient','critic_gradient','missing','dtype','layout','nan','entropy','shuffle','row_order','advantage_args','advantages','missing_shuffle'])
def test_first_gradient_reconstruction_rejects_corruptions_and_preserves_RNG(corruption):
    replay,_=capture_fixture()
    first=replay.trace['trace'][0]['gradient_after_clip']
    if corruption=='actor_gradient':first['actor.readout_gain'][0]+=.1
    elif corruption=='critic_gradient':first['critic.readout_gain'][0]+=.1
    elif corruption=='missing':del first['actor.readout_gain']
    elif corruption=='dtype':first['actor.readout_gain']=first['actor.readout_gain'].double()
    elif corruption=='layout':
        target=first['actor.readout_gain'];padded=torch.empty(2*target.numel(),device='cuda');padded[::2].copy_(target)
        first['actor.readout_gain']=padded[::2]
    elif corruption=='nan':first['actor.readout_gain'][0]=float('nan')
    elif corruption=='entropy':replay.args.ent_coef=.01
    elif corruption=='shuffle':replay.snapshot['rollout_start_rng']['shuffle']=torch.Generator(device='cuda').manual_seed(1237).get_state()
    elif corruption=='row_order':replay.snapshot['batch']['obs']=replay.snapshot['batch']['obs'].roll(1,0)
    elif corruption=='advantage_args':replay.args.norm_adv=True
    elif corruption=='advantages':replay.snapshot['batch']['advantages'][0]+=.7
    else:del replay.snapshot['rollout_start_rng']['shuffle']
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()
    with pytest.raises(ValueError):audit_first_gradient(replay)
    assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())


def test_saved_private_rng_reconstructs_source_permutation_without_global_draws():
    replay,expected=capture_fixture()
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()
    actual,_=exact_first_permutation(replay.module,replay.args,replay.snapshot['rollout_start_rng']['shuffle'],torch.device('cuda:0'))
    assert torch.equal(actual,expected)
    assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())


@pytest.mark.parametrize('invalid',['dtype','empty','minibatches'])
def test_permutation_rejects_incomplete_saved_semantics(invalid):
    replay,_=capture_fixture();state=replay.snapshot['rollout_start_rng']['shuffle']
    if invalid=='dtype':state=state.float()
    elif invalid=='empty':state=state[:0]
    else:replay.args.num_minibatches=2
    with pytest.raises(ValueError):exact_first_permutation(replay.module,replay.args,state,torch.device('cuda:0'))


def test_full_covector_clipping_algebra_independent_exact_vector():
    # Analytic reference independent of PPO/compiled gradients: ||[3,4]||=5,
    # reference ascent and negative loss covector differ by a common alpha.
    ascent={'w':torch.tensor([3.,4.],device='cuda')}
    recorded={'w':torch.tensor([.3,.4],device='cuda')}
    result=full_covector_summary(ascent,recorded)
    assert result['independent_policy_ascent_norm']==pytest.approx(5.)
    assert result['negative_recorded_clipped_loss_norm']==pytest.approx(.5)
    assert result['unconstrained_least_squares_common_scale']==pytest.approx(.1)
    assert result['cosine']==pytest.approx(1.)
    assert result['residual_after_common_scale_norm']<2e-8
    # Direction error hidden by a scalar route pairing is visible in full vector.
    wrong=full_covector_summary(ascent,{'w':torch.tensor([.5,0.],device='cuda')})
    assert wrong['cosine']==pytest.approx(.6)
    assert wrong['residual_relative_to_recorded_norm']==pytest.approx(.8)


def test_null_and_opposite_covectors_do_not_invent_positive_common_scale():
    zero={'w':torch.zeros(2,device='cuda')};one={'w':torch.ones(2,device='cuda')}
    assert full_covector_summary(zero,one)['unconstrained_least_squares_common_scale'] is None
    opposite=full_covector_summary(one,{'w':-one['w']})
    assert opposite['common_scale_positive'] is False
    assert opposite['cosine']==pytest.approx(-1.)


def test_source_clipping_is_once_and_trunk_independent():
    # Independent exact clipping witness; actor/critic do not share a norm.
    actor=torch.nn.Parameter(torch.zeros(2,device='cuda'));critic=torch.nn.Parameter(torch.zeros(2,device='cuda'))
    actor.grad=torch.tensor([3.,4.],device='cuda');critic.grad=torch.tensor([0.,12.],device='cuda')
    torch.nn.utils.clip_grad_norm_((actor,),.5,foreach=True)
    torch.nn.utils.clip_grad_norm_((critic,),.5,foreach=True)
    torch.testing.assert_close(actor.grad,torch.tensor([3.,4.],device='cuda')*(.5/(5+1e-6)),rtol=1e-6,atol=0)
    torch.testing.assert_close(critic.grad,torch.tensor([0.,12.],device='cuda')*(.5/(12+1e-6)),rtol=1e-6,atol=0)
