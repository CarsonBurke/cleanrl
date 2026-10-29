"""CUDA math/functional contracts. Run via mlq, never a bare test process."""
import sys
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cleanrl.shared.runtime import configure_runtime
from scripts.audit_sepclip_oracle_geometry import (
    beta_geometry,coordinate_report,fitted_response,tangent_jacobian,threshold_sensitivity,
    validate_dense_state_mapping,validate_saved_policy_logits,oracle_cases,
)


@pytest.fixture(autouse=True)
def cuda_runtime():
    configure_runtime(matmul_precision='highest',allow_tf32=False)
    assert torch.cuda.is_available(),'CUDA is mandatory'


def test_mu_kappa_covector_matches_direct_beta_score():
    logits=torch.tensor([[1.2,-.8,.3,2.]],dtype=torch.float64,device='cuda')
    pg=torch.tensor([[.12,-.07,.05,.2]],dtype=torch.float64,device='cuda')
    result=coordinate_report(logits,pg,.001)
    a,b,sa,sb,fisher=beta_geometry(logits)
    mu=a/(a+b);k=a+b
    expected=torch.stack((k*(pg[:,:2]/sa-pg[:,2:]/sb),
                          mu*pg[:,:2]/sa+(1-mu)*pg[:,2:]/sb),-1)
    assert torch.allclose(torch.tensor(result['coordinate_pg'],device='cuda',dtype=torch.float64),expected,atol=1e-7,rtol=1e-7)
    h=1e-5
    def shapes(mean,concentration):
        aa=mean*concentration;bb=(1-mean)*concentration
        return torch.cat((torch.log(torch.expm1(aa-1)),torch.log(torch.expm1(bb-1))),-1)
    derivative=(shapes(mu+h,k)-shapes(mu-h,k))/(2*h)
    direct=(derivative[:,:2]*pg[:,:2]+derivative[:,2:]*pg[:,2:])
    assert torch.allclose(direct,expected[...,0],atol=1e-8,rtol=1e-8)
    # The declared coordinate metric reproduces the original logit Fisher.
    paired=torch.stack(derivative.chunk(2,-1),-1)
    cost=(paired.unsqueeze(-2)@fisher@paired.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    reported=torch.tensor(result['coordinate_fisher'],device='cuda',dtype=torch.float64)
    assert torch.allclose(cost,reported[...,0,0],atol=1e-7,rtol=1e-7)


def test_dense_state_mapping_rejects_corrupted_training_coordinates():
    obs=torch.tensor([[1.,2.],[3.,4.]],device='cuda')
    native=torch.tensor([[.2,.8],[.4,.6]],device='cuda')
    advantage=torch.tensor([.7,-.3],device='cuda')
    snapshot=dict(transitions=dict(episode_age=torch.tensor([[9,10]],device='cuda')),
        batch=dict(obs=obs,native=native,advantages=advantage))
    selected=dict(t=0,row=1,episode_age=10,normalized_obs=obs[1].cpu().tolist(),
        native_action=native[1].cpu().tolist(),stored_advantage=float(advantage[1]))
    row=dict(snapshot_sha256='fixture',selected_states=[selected])
    validate_dense_state_mapping(row,snapshot,'fixture')
    for key,wrong in (('t',1),('row',0),('normalized_obs',[3.,4.01]),
                      ('native_action',[.4,.7]),('stored_advantage',0.),('episode_age',9)):
        corrupted=copy.deepcopy(row);corrupted['selected_states'][0][key]=wrong
        with pytest.raises(ValueError):validate_dense_state_mapping(corrupted,snapshot,'fixture')
    corrupted=dict(row,snapshot_sha256='wrong')
    with pytest.raises(ValueError,match='hash'):validate_dense_state_mapping(corrupted,snapshot,'fixture')


def test_initial_logits_validation_and_historical_protocol_remains_unmatched():
    logits=torch.tensor([[1.,-2.]],device='cuda')
    assert validate_saved_policy_logits([[1.,-2.]],logits)==0
    with pytest.raises(ValueError,match='logits mismatch'):
        validate_saved_policy_logits([[1.,-2.01]],logits)
    with pytest.raises(ValueError,match='Invalid saved'):
        validate_saved_policy_logits([[float('nan'),-2.]],logits)
    historical=dict(fork_replicates=[dict(states=[dict(normalized_observation=[.1,.2],logits=[1.,-2.])],
        horizons=[dict(horizon=16,logit_score_pg=[[.3,.4]],normalized_reward_logit_score_pg=[[.3,.4]],
            normalized_bootstrap_logit_score_pg=[[.3,.4]])])])
    cases=list(oracle_cases(historical))
    assert len(cases)==3
    for _,metadata,_ in cases:
        assert not metadata['state_distribution_matched']
        assert metadata['_saved_initial_logits']==[[1.,-2.]]


def test_tangent_jacobian_discards_common_radial_components():
    weight=F.normalize(torch.tensor([[1.,2.,3.],[-2.,1.,.5]],device='cuda'),dim=1)
    radial=weight.expand(4,5,-1,-1).clone()
    projected=tangent_jacobian(weight,radial,'head.0.weight')
    assert projected.abs().max()<1e-6
    down=F.normalize(weight,dim=0)
    radial=down.expand(4,5,-1,-1).clone()
    assert tangent_jacobian(down,radial,'first.0.down.weight').abs().max()<1e-6


def test_row_retraction_has_correct_tangent_derivative_and_second_order_term():
    weight=F.normalize(torch.tensor([[1.,2.,3.],[-2.,1.,.5]],device='cuda',dtype=torch.float64),dim=1)
    vector=torch.tensor([[.5,-.2,.7],[.3,.4,-.6]],device='cuda',dtype=torch.float64)
    tangent=tangent_jacobian(weight,vector,'weight')
    for radius in (1e-2,1e-3):
        finite=F.normalize(weight+radius*tangent,dim=1)-weight
        remainder=finite-radius*tangent
        second=-.5*radius**2*tangent.square().sum(1,keepdim=True)*weight
        # Taylor remainder beyond the known normal second-order displacement.
        assert (remainder-second).norm() < radius**3*tangent.norm()**3
        assert (remainder.norm()/radius**2-second.norm()/radius**2).abs()<radius


def test_threshold_sensitivity_separates_arithmetic_from_resolved_modes():
    u=torch.eye(4,device='cuda',dtype=torch.float64)
    singular=torch.tensor([1.,1e-5,1e-7,1e-9],device='cuda',dtype=torch.float64)
    field=torch.tensor([1.,2.,3.,4.],device='cuda',dtype=torch.float64)
    results=threshold_sensitivity(u,singular,field,.001,1e-15,actual_white=field)
    assert results[0]['rank']==4
    assert results[-1]['rank']==1
    assert results[-1]['oracle_reachable_dual_energy_fraction']==pytest.approx(1/30)
    assert results[-1]['required_tangent_parameter_norm_at_full_budget']==pytest.approx(.002**.5)
    assert results[0]['required_tangent_parameter_norm_at_full_budget']>1e7
    assert all(r['actual_retained_range_model_bound_efficiency']==pytest.approx(1.) for r in results)


def test_ridge_free_fit_exposes_unreachable_state_pattern():
    # Five states, four outputs: 20 response coordinates. Constrained linear
    # actor has only 8 row tangents + 4 bias freedoms, so cannot fit every field.
    actor=torch.nn.Linear(3,4).cuda()
    # Explicit rows and an affinely spanning observation design avoid an
    # unseeded, potentially ill-conditioned random fit. The algorithm must
    # still report large-step failures for arbitrary real checkpoint fits.
    with torch.no_grad():
        actor.weight.copy_(F.normalize(torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]],device='cuda'),dim=1))
        actor.bias.zero_()
    generator=torch.Generator(device='cuda').manual_seed(61031)
    obs=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[-1.,-1.,-1.],[1.,-1.,.5]],device='cuda')
    pg=torch.randn(5,4,generator=generator,device='cuda',dtype=torch.float64)
    cache={}
    result=fitted_response(SimpleNamespace(actor=actor),obs,pg,.00001,cache=cache)
    assert result['rank']<=12
    assert result['projected_oracle_energy_fraction']<.999
    assert result['oracle_whitened_residual_relative_norm']>.001
    assert abs(result['joint_fisher_quadratic']-result['achieved_fisher_quadratic'])<1e-8
    assert result['finite_retraction_linearization_relative_error']<.05
    smaller=fitted_response(SimpleNamespace(actor=actor),obs,pg,.0000001,cache=cache)
    assert smaller['finite_retraction_linearization_relative_error']<.01
    assert smaller['finite_retraction_linearization_relative_error'] < .2*result['finite_retraction_linearization_relative_error']
    assert abs(smaller['required_parameter_delta_norm']/result['required_parameter_delta_norm']-.1)<1e-10
