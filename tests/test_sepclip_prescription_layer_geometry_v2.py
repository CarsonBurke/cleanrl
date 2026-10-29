"""V2 geometry contracts; preserves v1/11905 (17 passed, uniform identity failed)."""
import importlib
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.shared.runtime import configure_runtime
from scripts.sepclip_prescription_replay import constraint_axes
from scripts.audit_sepclip_prescription_layer_geometry_v2 import (
    audit_layers, beta_kl, beta_shape_fisher, clipped_ascent, joint_gram, layer_partition,
    scalar_pairing, stage_summary, trace_geometry, trigamma_recurrence, trigamma_difference,
)


@pytest.fixture(autouse=True)
def cuda_only():
    assert torch.cuda.is_available(), 'Submit contracts through MLQ; CUDA mandatory'
    configure_runtime(matmul_precision='highest', allow_tf32=False)


def test_joint_gram_keeps_cancellation_cross_terms_and_state_denominator():
    response=torch.tensor([[[[1.,2.],[3.,4.]],[[2.,1.],[4.,3.]]]],device='cuda',dtype=torch.float64)
    response=torch.cat((response,-response),0)
    f=torch.tensor([[2.,-.5],[-.5,3.]],device='cuda',dtype=torch.float64).expand(2,2,2,2)
    gram=joint_gram(response,f)
    # Independent scalar loops: do not mirror the einsum implementation.
    energy=sum(float(x @ f[s,a] @ x) for s in range(2) for a in range(2) for x in [response[0,s,a]])/2
    torch.testing.assert_close(gram,torch.tensor([[energy,-energy],[-energy,energy]],device='cuda',dtype=torch.float64))
    assert float(gram.diagonal().sum())>0
    assert float(gram.sum())==pytest.approx(0.,abs=1e-12)


def test_shape_fisher_uniform_exact_and_finite_kl_independent_density_integral():
    ones=torch.ones((2,3),device='cuda')
    f=beta_shape_fisher(ones,ones)
    torch.testing.assert_close(f[...,0,0],torch.ones_like(f[...,0,0]),rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(f[...,0,1],torch.full_like(f[...,0,1],1-np.pi**2/6),rtol=1e-13,atol=1e-13)
    # Uniform old density against Beta(2,1): integral -log(2x) dx = 1-log2.
    assert float(beta_kl(ones,ones,2*ones,ones))==pytest.approx(3*(1-np.log(2)),abs=2e-13)
    assert float(beta_kl(ones,ones,ones,ones))==pytest.approx(0.,abs=1e-14)


def test_nearby_beta_kl_matches_half_shape_fisher_in_predeclared_range():
    a=torch.tensor([[1.5,2.],[4.,3.]],device='cuda',dtype=torch.float64)
    b=torch.tensor([[3.,4.],[2.,1.5]],device='cuda',dtype=torch.float64)
    direction=torch.tensor([[[.1,-.2],[.3,.1]],[[-.2,.4],[.1,.2]]],device='cuda',dtype=torch.float64)
    energy=float(joint_gram(direction[None],beta_shape_fisher(a,b))[0,0])
    for h in [.01,.005,.0025]:
        measured=float(beta_kl(a,b,a+h*direction[...,0],b+h*direction[...,1]))
        assert measured==pytest.approx(.5*h*h*energy,rel=.003)


@pytest.mark.parametrize('bad', ['nan','shape','below_one'])
def test_shape_fisher_rejects_invalid_shapes(bad):
    a=torch.ones((2,3),device='cuda');b=a.clone()
    if bad=='nan':a[0,0]=float('nan')
    elif bad=='shape':b=b[:1]
    else:a[0,0]=.5
    with pytest.raises(ValueError):beta_shape_fisher(a,b)


@pytest.mark.parametrize('bad', ['nan','layout','empty'])
def test_joint_gram_rejects_invalid_response(bad):
    d=torch.ones((2,2,3,2),device='cuda');f=torch.eye(2,device='cuda').expand(2,3,2,2)
    if bad=='nan':d[0,0,0,0]=float('nan')
    elif bad=='layout':d=d[...,0]
    else:d=d[:0]
    with pytest.raises(ValueError):joint_gram(d,f)


def test_partition_retains_bias_mix_gain_and_rejects_duplicates():
    names=['trunk.in_proj.weight','trunk.in_proj.bias','trunk.blocks.0.gate.weight',
           'trunk.blocks.0.gate.bias','trunk.block_gates.0','head.0.weight','head.0.bias','readout_gain']
    groups=layer_partition(names)
    assert groups['trunk.in_proj']==names[:2]
    assert groups['trunk.block_gates.0']==['trunk.block_gates.0']
    assert groups['readout_gain']==['readout_gain']
    assert sorted(sum(groups.values(),[]))==sorted(names)
    with pytest.raises(ValueError):layer_partition(names+names[:1])


def test_trace_free_radial_motion_is_retained_and_constrained_covector_split():
    weights={'free.weight':torch.tensor([[2.,0.]],device='cuda'),
             'head.weight':torch.tensor([[1.,0.]],device='cuda')}
    gradient={'actor.free.weight':torch.tensor([[3.,0.]],device='cuda'),
              'actor.head.weight':torch.tensor([[3.,4.]],device='cuda')}
    ambient={'actor.free.weight':torch.tensor([[-.2,0.]],device='cuda'),
             'actor.head.weight':torch.tensor([[-.1,-.2]],device='cuda')}
    actual={key:value.clone() for key,value in ambient.items()}
    trace=[dict(epoch=0,update=0,gradient_after_clip=gradient,ambient_delta=ambient,retracted_delta=actual)]
    result=trace_geometry(weights,trace,layer_partition(weights),{'actor.head.weight':1})[0]['layers']
    assert result['free']['geometry']=='free_euclidean'
    assert result['free']['negative_clipped_loss_gradient_dot_actual']==pytest.approx(.6)
    assert result['free']['actual_relative_weight_motion']==pytest.approx(.1)
    assert result['free']['constrained_decomposition']=={}
    split=result['head']['constrained_decomposition']['head.weight']
    assert split['clipped_gradient_normal_norm']==pytest.approx(3.)
    assert split['clipped_gradient_tangent_norm']==pytest.approx(4.)


def test_stage_ratios_and_full_situ_derivative_do_not_use_cap_proxy():
    stream=torch.tensor([[3.,4.]],device='cuda');branch=2*stream;weighted=.25*branch
    g=torch.tensor([[-1.2187557268720126,0.]],device='cuda',dtype=torch.float64)
    result=stage_summary({'block0.stream_before':stream,'block0.raw_branch':branch,
        'block0.weighted_branch':weighted,'block0.stream_after':stream+weighted,
        'block0.gate_preactivation':g})
    ratios=result['residual_ratios']['block0.']
    assert ratios['raw_branch_to_stream_rms_ratio']==pytest.approx(2.)
    assert ratios['weighted_branch_to_stream_rms_ratio']==pytest.approx(.5)
    assert ratios['stream_weighted_branch_cosine']==pytest.approx(1.)
    assert ratios['summed_stream_second_moment_identity_residual']==pytest.approx(0.)
    derivative=result['activations']['block0.gate_preactivation']['full_situ_gate_factor_derivative']
    assert derivative['absolute_quantiles'][0]<1e-14
    assert derivative['max']==pytest.approx(.5)


def test_pairing_signed_loss_credit_and_coverage_contract():
    gradient={'x':torch.tensor([2.,-3.],device='cuda')};direction={'x':torch.tensor([-1.,1.],device='cuda')}
    assert scalar_pairing(gradient,direction)==pytest.approx(-5.)
    with pytest.raises(ValueError):scalar_pairing(gradient,{})
    with pytest.raises(ValueError):scalar_pairing(gradient,{'x':torch.zeros(3,device='cuda')})


def test_clipping_sign_mask_differs_from_all_clipped_ratios():
    ratio=torch.tensor([.7,.7,1.3,1.3],device='cuda',requires_grad=True)
    advantage=torch.tensor([-1.,2.,-3.,4.],device='cuda')
    clipped_ascent(ratio,advantage,.2,.2).backward()
    torch.testing.assert_close(ratio.grad,torch.tensor([0.,.5,-.75,0.],device='cuda'))
    assert bool(((ratio.detach()<.8)|(ratio.detach()>1.2)).all())
    assert int((ratio.grad!=0).sum())==2


def test_unrecorded_minibatch_membership_is_rejected_before_model_access():
    with pytest.raises(ValueError,match='minibatch membership'):
        audit_layers(SimpleNamespace(args=SimpleNamespace(num_minibatches=2)))


@pytest.mark.parametrize('module_name,architecture', [
    ('cleanrl.ppo_continuous_action_sepclip_additive_prerms_anvil_v14','additive_prerms'),
    ('cleanrl.ppo_continuous_action_sepclip_additive_prerms_anvil_v14','original'),
    ('cleanrl.ppo_continuous_action_sepclip_constraint_transport_v15','original')])
def test_actual_architectures_joint_layers_and_independent_public_ppo_gradient(module_name,architecture):
    module=importlib.import_module(module_name)
    env=SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf,np.inf,(17,),dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.,1.,(6,),dtype=np.float32))
    torch.manual_seed(179)
    agent=(module.Agent(env,architecture) if module_name.endswith('v14') else module.Agent(env)).cuda()
    with torch.no_grad():agent.normalize_matrices()
    old={name:value.detach().clone() for name,value in agent.actor.named_parameters()}
    delta={name:(torch.arange(value.numel(),device='cuda').float().sin()*1e-5).reshape(value.shape) for name,value in old.items()}
    first={name:old[name]+delta[name] for name in old};endpoint={name:old[name]+2*delta[name] for name in old}
    obs=torch.linspace(-.7,.9,4*17,device='cuda').reshape(4,17)
    native=torch.linspace(.1,.9,24,device='cuda').reshape(4,6)
    with torch.no_grad():
        alpha,beta,values=agent.get_policy_and_value(obs);logprob=agent.action_logprob(alpha,beta,native).clone();values=values.flatten().clone()
    advantages=torch.tensor([1.,-2.,3.,-.5],device='cuda')
    batch=dict(obs=obs,native=native,logprob=logprob,advantages=advantages,values=values,returns=values+advantages)
    kwargs=dict(norm_adv=False,ent_coef=0.,clip_vloss=False)
    if module_name.endswith('v14'):kwargs['actor_architecture']=architecture
    args=module.Args(**kwargs)
    entries=[]
    for i in range(10):
        entries.append(dict(epoch=i,update=i,gradient_after_clip={'actor.'+name:torch.ones_like(value)*.01 for name,value in old.items()},
            ambient_delta={'actor.'+name:value*.2 for name,value in delta.items()},
            retracted_delta={'actor.'+name:value*.2 for name,value in delta.items()}))
    def compiled_forward(bank,x):
        z=torch.func.functional_call(agent.actor,bank,(x,),strict=True);a,b=(1+torch.nn.functional.softplus(z)).chunk(2,-1)
        return z,a,b
    forward=torch.compile(compiled_forward,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    replay=SimpleNamespace(module=module,agent=agent,args=args,snapshot={'batch':batch},trace={'trace':entries},
        banks=dict(old=old,first=first,endpoint=endpoint),deltas=(delta,{name:endpoint[name]-old[name] for name in old}),
        constraint_axes=constraint_axes(module_name,args,agent),compiled_forward=forward)
    result=audit_layers(replay)
    # Independent frozen public PPO loss/backward, not the audit's score formula.
    def public_loss(x,n,l,a,r,v):return module.ppo_loss(agent,x,n,l,a,r,v,args)[0]
    compiled_loss=torch.compile(public_loss,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    agent.zero_grad(set_to_none=True)
    compiled_loss(obs,native,logprob,advantages,batch['returns'],values).backward()
    independent={name:-value.grad.detach().clone() for name,value in agent.actor.named_parameters()}
    for label,direction in zip(('first','endpoint'),replay.deltas):
        chord=result['chords'][label]
        torch.testing.assert_close(torch.tensor(chord['old_stored_clipped_ppo_ascent_credit_full'],device='cuda'),
            torch.tensor(scalar_pairing(independent,direction),device='cuda'),rtol=3e-4,atol=2e-8)
        assert sum(chord['old_stored_clipped_ppo_ascent_credit_by_layer'].values())==pytest.approx(chord['old_stored_clipped_ppo_ascent_credit_full'],abs=1e-12)
        assert chord['layer_sum_vs_full_logit_jvp_l2_error']<max(2e-7,chord['full_logit_jvp_norm']*2e-4)
        assert chord['joint_vs_full_jvp_energy_abs_error']<max(1e-11,chord['independent_full_jvp_energy']*5e-4)
        gram=torch.tensor(chord['full_joint_shape_fisher_gram'],device='cuda',dtype=torch.float64)
        torch.testing.assert_close(gram,gram.T,rtol=1e-12,atol=1e-15)
        assert float(torch.linalg.eigvalsh(gram).min())>=-1e-12
    assert len(result['epoch_trace'])==10
    if architecture=='additive_prerms':
        assert len(agent.actor.trunk.blocks)==2
        assert len(module.hidden_matrix_parameters(agent.actor))==7
        for row in result['epoch_trace']:
            assert row['layers']['trunk.in_proj']['geometry']=='free_euclidean'
            assert row['layers']['trunk.in_proj']['constrained_decomposition']=={}
        assert len(result['stage_diagnostics']['banks']['old']['residual_ratios'])==2
    for bank in result['stage_diagnostics']['banks'].values():assert bank['actual_forward_max_abs_error']<2e-5


def integral_reference(x,separation=None):
    # Independent scaled Laplace integral, no asymptotic/recurrence/polygamma.
    # Scalar highprecision mathematical reference runs only in queued contracts;
    # all evaluated implementation tensors remain CUDA FP64.
    import mpmath as mp
    with mp.workdps(100):
        x=mp.mpf(x)
        if separation is None:
            def integrand(s):
                return x if s==0 else s*mp.exp(-s)/(-mp.expm1(-s/x))
        else:
            separation=mp.mpf(separation)
            def integrand(s):
                return mp.mpf(0) if s==0 else s*mp.exp(-s)*(-mp.expm1(-separation*s/x))/(-mp.expm1(-s/x))
        return mp.quad(integrand,[0,1,mp.inf])/(x*x)


@pytest.mark.parametrize('x',[1.,1.01,2.,3.5,32.,1e3,1e8,1e30])
def test_trigamma_recurrence_against_independent_100digit_scaled_integral(x):
    import mpmath as mp
    value,budget=trigamma_recurrence(torch.tensor(x,device='cuda',dtype=torch.float64))
    reference=integral_reference(x)
    with mp.workdps(100):
        error=abs(mp.mpf(float(value))-reference)
        assert error<=mp.mpf(float(budget))
        assert error/reference<mp.mpf('1e-13')


@pytest.mark.parametrize('x,b',[(1.,1.),(1.,30.),(30.,1.),(1e8,1.),(1e30,1.),(1e30,1e30),(1.25,3.5)])
def test_stable_diagonal_difference_against_independent_scaled_integral(x,b):
    import mpmath as mp
    value,budget=trigamma_difference(torch.tensor(x,device='cuda',dtype=torch.float64),torch.tensor(b,device='cuda',dtype=torch.float64))
    reference=integral_reference(x,b)
    with mp.workdps(100):
        error=abs(mp.mpf(float(value))-reference)
        assert error<=mp.mpf(float(budget))
        assert error/reference<mp.mpf('1e-13')
    assert float(value)>0
    if x>=1e30 and b==1.:
        assert float(value)==pytest.approx(1/(x*x),rel=1e-13)
        # Witness: straightforward binary64 argument addition already loses b.
        assert float(torch.tensor(x,device='cuda',dtype=torch.float64)+b)==x


@pytest.mark.parametrize('a,b',[(1.,1.),(1.0000001192092896,2.),(1.,1e8),(1e8,1.),(1e30,1.),(1e6,1e6)])
def test_fisher_contract_matches_independent_highprecision_and_quadratic_bound(a,b):
    import mpmath as mp
    # Actual FP32 shape storage, including the first shape above floor-one.
    alpha=torch.tensor([[a]],device='cuda',dtype=torch.float32)
    beta=torch.tensor([[b]],device='cuda',dtype=torch.float32)
    f,contract=beta_shape_fisher(alpha,beta,return_contract=True)
    with mp.workdps(100):
        a=mp.mpf(float(alpha.item()));b=mp.mpf(float(beta.item()))
        reference=mp.matrix([[mp.polygamma(1,a)-mp.polygamma(1,a+b),-mp.polygamma(1,a+b)],
                             [-mp.polygamma(1,a+b),mp.polygamma(1,b)-mp.polygamma(1,a+b)]])
        values=f[0,0].cpu().tolist()
        for i in range(2):
            for j in range(2):
                assert abs(mp.mpf(values[i][j])-reference[i,j])<=contract['maximum_entry_absolute_error_budget']
        # Balanced/asymmetric smallmode probes; normalize by diagonal scale so
        # the extreme asymmetric witness exercises its tiny diagonal too.
        for direction in [(1/mp.sqrt(reference[0,0]),1/mp.sqrt(reference[1,1])),
                          (1/mp.sqrt(reference[0,0]),-1/mp.sqrt(reference[1,1])),
                          (1/mp.sqrt(reference[0,0]),mp.mpf(0))]:
            v=mp.matrix(direction);exact=(v.T*reference*v)[0]
            measured=sum(v[i]*mp.mpf(values[i][j])*v[j] for i in range(2) for j in range(2))
            assert abs(measured-exact)/exact<=contract['maximum_conditional_relative_quadratic_error']
    assert contract['maximum_conditional_relative_quadratic_error']<=.01


@pytest.mark.parametrize('kappa',[1e14,1e30])
def test_balanced_unresolved_small_mode_is_withheld_without_ridge(kappa):
    a=torch.tensor([[kappa]],device='cuda',dtype=torch.float64)
    with pytest.raises(ValueError,match='small-mode'):
        beta_shape_fisher(a,a)


@pytest.mark.parametrize('x',[.5,float('inf'),1e40])
def test_recurrence_domain_rejects_unsupported_or_nonfinite_arguments(x):
    with pytest.raises(ValueError):trigamma_recurrence(torch.tensor(x,device='cuda',dtype=torch.float64))
