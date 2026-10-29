"""Actual-step oracle math contracts; CUDA through mlq only."""
import sys
import importlib
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cleanrl.shared.runtime import configure_runtime
from scripts.audit_sepclip_oracle_geometry import beta_geometry,tangent_jacobian
from scripts.audit_sepclip_actual_oracle_step import (
    actual_case,build_geometry,coordinate_changes,layer_motion,motion_metrics,
    paired,quadratic,replica_fit_validation,
    whitened_replica_statistics,
    retained_range_motion_metrics,
    reconstruct_banks,
    fixed_motion_objective_pairings,
)


@pytest.fixture(autouse=True)
def cuda_runtime():
    configure_runtime(matmul_precision='highest',allow_tf32=False)
    assert torch.cuda.is_available(),'CUDA is mandatory; submit contracts through mlq'


def objective_pairing_fixture():
    fields={}
    for name,values in (('finite_raw',[[1.,0.],[-2.,0.]]),
                        ('finite_normalized',[[3.,0.],[-1.,0.]]),
                        ('bootstrap_normalized',[[-3.,0.],[-1.,0.]])):
        mean=torch.tensor(values,device='cuda',dtype=torch.float64)
        fields[name]=dict(mean=mean.tolist(),replicate_values=torch.stack((.8*mean,1.2*mean)).tolist())
    horizon=dict(horizon=4,discount=.9,tail_weight=.9**4,policy_logit_pg=fields)
    motion=torch.tensor([[1.,0.],[1.,0.]],device='cuda',dtype=torch.float64)
    motions=dict(first=dict(actual_endpoint_jvp=motion,tangent_only_jvp=.5*motion,
        finite_actual_logit_change=-motion),endpoint=dict(actual_endpoint_jvp=2*motion,
        tangent_only_jvp=motion,finite_actual_logit_change=-2*motion))
    return horizon,motions


def test_fixed_objective_pairings_keep_normalization_and_tail_sign_flips():
    horizon,motions=objective_pairing_fixture()
    result=fixed_motion_objective_pairings(horizon,motions,2,True)
    # Raw is normalized times POSITIVE state-specific scales [1/3,2].
    # Pairing after a single aggregate scale conversion misses this sign flip.
    expected=dict(finite_raw=-.5,finite_normalized=1.,bootstrap_normalized=-2.)
    for name,gain in expected.items():
        first=result[name]['cases']['first']
        row=first['actual_endpoint_jvp']
        assert row['mean']==pytest.approx(gain) # Divide by two states, not four logit entries.
        assert row['replica_values']==pytest.approx([.8*gain,1.2*gain])
        assert row['mean_field_pairing']==pytest.approx(gain)
        assert first['tangent_only_jvp']['mean']==pytest.approx(.5*gain)
        assert first['finite_actual_logit_change']['mean']==pytest.approx(-gain)
        assert result[name]['cases']['endpoint']['actual_endpoint_jvp']['mean']==pytest.approx(2*gain)
        low,high=row['conditional_student_t95_interval']
        assert low<gain<high
        assert row['conditional_mc_standard_error']==pytest.approx(.2*abs(gain))


def test_fixed_tail_increment_uses_correlated_same_replica_differences():
    horizon,motions=objective_pairing_fixture()
    for name,noise in (('finite_normalized',10.),('bootstrap_normalized',10.5)):
        field=horizon['policy_logit_pg'][name]
        mean=torch.tensor(field['mean'],device='cuda',dtype=torch.float64)
        offset=torch.tensor([[noise,0.],[noise,0.]],device='cuda',dtype=torch.float64)
        field['replicate_values']=torch.stack((mean-offset,mean+offset)).tolist()
    output=fixed_motion_objective_pairings(horizon,motions,2,True)
    tail=output['paired_estimated_tail_increment']['cases']['first']['actual_endpoint_jvp']
    assert tail['replica_values']==pytest.approx([-3.5,-2.5])
    assert tail['mean']==pytest.approx(-3.)
    assert tail['mean_field_pairing']==pytest.approx(-3.)
    assert tail['conditional_mc_standard_error']==pytest.approx(.5)
    # Treating these correlated gains as independent would give SE14.5.
    finite=output['finite_normalized']['cases']['first']['actual_endpoint_jvp']
    bootstrap=output['bootstrap_normalized']['cases']['first']['actual_endpoint_jvp']
    independent_se=(finite['conditional_mc_standard_error']**2+bootstrap['conditional_mc_standard_error']**2)**.5
    assert independent_se==pytest.approx(14.5)
    assert tail['conditional_student_t95_interval'][0]<-3<tail['conditional_student_t95_interval'][1]


@pytest.mark.parametrize('corruption', ['mean','nonfinite','layout','missing','tail','count','scope'])
def test_fixed_objective_pairings_reject_inconsistent_inputs(corruption):
    horizon,motions=objective_pairing_fixture()
    field=horizon['policy_logit_pg']['finite_raw']
    if corruption=='mean':field['mean'][0][0]+=1
    elif corruption=='nonfinite':field['replicate_values'][0][0][0]=float('nan')
    elif corruption=='layout':field['replicate_values'][0].pop()
    elif corruption=='missing':del horizon['policy_logit_pg']['bootstrap_normalized']
    elif corruption=='tail':horizon['tail_weight']=.5
    elif corruption=='count':field['replicate_values'].append(field['replicate_values'][0])
    with pytest.raises((ValueError,TypeError)):
        fixed_motion_objective_pairings(horizon,motions,2,corruption!='scope')


def test_actual_mean_concentration_budget_keeps_signed_cross_term():
    logits=torch.tensor([[2.,-.8,.4,1.],[.2,1.4,-.4,2.1]],device='cuda',dtype=torch.float64)
    change=torch.tensor([[.04,-.01,.03,-.05],[-.02,.06,.01,.02]],device='cuda',dtype=torch.float64)
    _,_,_,_,fisher=beta_geometry(logits)
    mean,kappa,_,_=coordinate_changes(logits,change)
    assert torch.allclose(mean+kappa,change,atol=1e-15,rtol=1e-14)
    complete=quadratic(change,change,fisher)
    split=quadratic(mean,mean,fisher)+quadratic(kappa,kappa,fisher)+2*quadratic(mean,kappa,fisher)
    assert torch.allclose(complete,split,atol=1e-18,rtol=1e-13)
    assert abs(float(quadratic(mean,kappa,fisher)))>1e-10


def test_retained_bound_projects_motion_and_reports_discarded_energy():
    identity=torch.eye(2,device='cuda',dtype=torch.float64)
    geometry=dict(root=identity.reshape(1,1,2,2),inverse_root=identity.reshape(1,1,2,2),
        u=identity,active=torch.tensor([True,False],device='cuda'))
    score=torch.tensor([[1.,100.]],device='cuda',dtype=torch.float64)
    change=score.clone()
    result=retained_range_motion_metrics(geometry,change,score)
    assert result['retained_range_oracle_first_order_gain']==1
    assert result['retained_range_cauchy_gain_bound']==1
    assert result['efficiency_vs_retained_range_cauchy_bound']==1
    assert result['retained_range_fisher_quadratic']==.5
    assert result['discarded_motion_fisher_quadratic']==5000
    assert result['retained_motion_energy_fraction']==pytest.approx(1/10001)
    # Pairing full motion against a retained-field bound was invalid: the
    # discarded mode supplies nearly the entire numerator.
    invalid_ratio=float((score*change).sum()/change.norm())
    assert invalid_ratio>100
    for shifted in (change,-change,torch.tensor([[0.,3.]],device='cuda',dtype=torch.float64)):
        ratio=retained_range_motion_metrics(geometry,shifted,score)['efficiency_vs_retained_range_cauchy_bound']
        assert ratio is None or abs(ratio)<=1


def test_actual_trace_rejects_incomplete_sequence_and_wrong_endpoint():
    module=importlib.import_module('cleanrl.ppo_continuous_action_sepclip_rollout_clock_anvil_v7')
    gym=importlib.import_module('gymnasium')
    import numpy as np
    descriptor=SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf,np.inf,(17,),dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.,1.,(6,),dtype=np.float32))
    agent=module.Agent(descriptor).cuda().eval()
    agent.normalize_matrices()
    state={name:p.detach().clone() for name,p in agent.state_dict().items()}
    deltas={name:torch.zeros_like(p) for name,p in agent.named_parameters()}
    snapshot=dict(global_step=7,args=dict(update_epochs=2,batch_size=1,minibatch_size=1),model=state)
    trace=dict(global_step=7,model_after_update=state,
        trace=[dict(epoch=i,update=i,retracted_delta=deltas) for i in range(2)])
    assert reconstruct_banks(agent,snapshot,trace)[2]['update_count']==2
    broken=copy.deepcopy(trace);broken['trace'][0]['epoch']=1
    with pytest.raises(ValueError,match='first actual update'):reconstruct_banks(agent,snapshot,broken)
    broken=copy.deepcopy(trace);broken['trace'][1]['update']=0
    with pytest.raises(ValueError,match='sequence'):reconstruct_banks(agent,snapshot,broken)
    broken=copy.deepcopy(trace);broken['trace'].pop()
    with pytest.raises(ValueError,match='Full endpoint trace'):reconstruct_banks(agent,snapshot,broken)
    broken=copy.deepcopy(trace);broken['model_after_update']['actor.head.0.bias'].add_(.01)
    with pytest.raises(ValueError,match='reconstruction error'):reconstruct_banks(agent,snapshot,broken)


def test_natural_logit_direction_saturates_cauchy_bound():
    logits=torch.tensor([[1.,-.4,.2,2.],[-.8,.1,.6,1.5]],device='cuda',dtype=torch.float64)
    score=torch.tensor([[.2,-.1,.03,.07],[-.04,.12,.09,-.02]],device='cuda',dtype=torch.float64)
    _,_,_,_,fisher=beta_geometry(logits)
    natural=torch.linalg.solve(fisher,paired(score).unsqueeze(-1)).squeeze(-1)
    change=torch.cat(natural.unbind(-1),-1)*.003
    result=motion_metrics(logits,change,score,fisher)
    assert abs(result['efficiency_vs_unconstrained']-1)<1e-12
    assert abs(result['oracle_first_order_gain']-result['unconstrained_cauchy_gain_bound'])<1e-12


def test_replica_cross_energy_is_signed_and_whitening_basis_invariant():
    fields=torch.tensor([[1.,0.],[-1.,0.]],device='cuda',dtype=torch.float64)
    actual=torch.tensor([.2,.3],device='cuda',dtype=torch.float64)
    result=whitened_replica_statistics(fields,actual)
    assert result['mean_signed_cross_energy']==-1
    assert result['mean_half_squared_difference_noise_trace']==2
    assert result['noise_corrected_mean_field_squared_norm']==-1
    assert result['signal_unresolved_flag']
    assert result['actual_gain_replica_values']==pytest.approx([.2,-.2])
    rotation=torch.tensor([[0.,-1.],[1.,0.]],device='cuda',dtype=torch.float64)
    rotated=whitened_replica_statistics(fields@rotation.mT,rotation@actual)
    for name in ('mean_signed_cross_energy','mean_half_squared_difference_noise_trace',
                 'noise_corrected_mean_field_squared_norm','actual_gain_mean','actual_gain_mc_standard_error'):
        assert result[name]==pytest.approx(rotated[name])
    fields=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]],device='cuda',dtype=torch.float64)
    result=whitened_replica_statistics(fields)
    assert len(result['pairs'])==3
    assert result['mean_signed_cross_energy']==pytest.approx(-1/3)
    assert result['noise_corrected_mean_field_squared_norm']==pytest.approx(-1/3)


def test_actual_jvp_and_independent_replica_falsification():
    actor=torch.nn.Linear(3,4).cuda()
    with torch.no_grad():
        actor.weight.copy_(F.normalize(torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]],device='cuda'),dim=1))
        actor.bias.zero_()
    obs=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[-1.,-1.,-1.],[1.,-1.,.5]],device='cuda')
    geometry=build_geometry(SimpleNamespace(actor=actor),obs)
    generator=torch.Generator(device='cuda').manual_seed(61034)
    candidate={}
    for name,weight in geometry['base'].items():
        vector=torch.randn(weight.shape,generator=generator,device='cuda')*.001
        if weight.ndim==2:
            vector=tangent_jacobian(weight,vector,name)
            candidate[name]=F.normalize(weight+vector,dim=1)
        else:candidate[name]=weight+vector
    direction={name:value-geometry['base'][name] for name,value in candidate.items()}
    actual=actual_case(geometry,direction,candidate,'test_first_step')
    analytic=obs.double()@direction['weight'].double().mT+direction['bias'].double()
    assert torch.allclose(layer_motion(geometry,direction).sum(0),analytic,atol=1e-12,rtol=1e-12)
    assert actual['report']['finite_logit_linearization_relative_error']<.01
    assert abs(actual['report']['joint_fisher_quadratic']-float(quadratic(analytic,analytic,geometry['fisher'])))<1e-15
    score=torch.ones(5,4,device='cuda',dtype=torch.float64)
    validation=replica_fit_validation(geometry,torch.stack((score,-score)),torch.tensor(1e-7,device='cuda',dtype=torch.float64))
    assert len(validation)==2
    for entry in validation:
        assert entry['fit_replica_first_order_gain']>0
        assert entry['held_replica_first_order_gain']<0
        assert entry['held_replica_efficiency']<0
