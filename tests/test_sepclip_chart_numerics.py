"""Meaningful CUDA contracts for the arithmetic audit; queue through mlq."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from cleanrl.shared.runtime import configure_runtime

spec=importlib.util.spec_from_file_location('chart_numerics',Path(__file__).parents[1]/'scripts/audit_sepclip_chart_numerics.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture(autouse=True)
def cuda_runtime():
    configure_runtime(matmul_precision='highest',allow_tf32=False)
    assert torch.cuda.is_available(), 'Queue these compiled contracts through mlq'


def test_fp64_chart_score_chain_matches_actual_autograd():
    coords=torch.tensor([[np.log(.1/.9),np.log(8)], [0.,np.log(28)],
                         [np.log(.9/.1),np.log(98)]],device='cuda',dtype=torch.float64,requires_grad=True)
    actions=torch.tensor([[.2],[.5],[.8]],device='cuda',dtype=torch.float64)
    density,_,shapes,_=audit.compiled(audit.chart_density)(coords,actions)
    shapes=shapes.detach().clone()
    actual=torch.autograd.grad(density.sum(),coords)[0]
    a,b=shapes.unbind(-1)
    shape_score=torch.stack((actions[:,0].log()-torch.digamma(a)+torch.digamma(a+b),
        torch.log1p(-actions[:,0])-torch.digamma(b)+torch.digamma(a+b)),-1)
    expected=audit.shape_to_chart_score(shape_score,coords.detach())
    torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-10)


def test_exact_kl_same_shape_reference_and_positive_coordinate_change():
    old=torch.tensor([[2.,3.],[100.,200.],[500000.,500000.]],device='cuda',dtype=torch.float64)
    fn=audit.compiled(audit.exact_kl)
    self_kl=fn(old,old).clone()
    torch.testing.assert_close(self_kl,torch.zeros_like(self_kl),atol=1e-8,rtol=0)
    new=old*torch.tensor([1.01,.99],device='cuda',dtype=torch.float64)
    changed=fn(old,new)
    assert bool(torch.all(changed>0))
    assert bool(torch.isfinite(changed).all())


def test_real_fused_sampler_moments_and_permanent_buffer_staging():
    a,b=5.,7.
    logits=np.array([a-1+np.log(-np.expm1(-(a-1))),b-1+np.log(-np.expm1(-(b-1)))],np.float32)
    result=audit.sampler_moments(logits,a,b,batches=128,seed=1)
    assert result['count']==128*16*6
    assert abs(result['mean_error_in_theoretical_se'])<6
    assert abs(result['variance_error_in_asymptotic_se'])<6
    assert result['physical_rescale_max_abs_error']==0
    assert max(result['host_vs_cuda_shape_abs_error'])<1e-5


def test_actual_fp32_decoder_reports_domain_failures_without_hiding_them():
    result=audit.actual_decode_range()
    assert result['nonfinite_decode_cases']>0
    assert result['nonfinite_shape_backward_cases']>0
    ordinary=[r for r in result['rows'] if r['mode_logit']==0 and r['log_excess_concentration']==0]
    assert ordinary
    assert all(r['decoded_logits_finite'] and r['shape_backward_finite'] for r in ordinary)
    negative=[r for r in result['rows'] if r['log_excess_concentration']<=0]
    assert all(r['decoded_logits_finite'] and r['decoded_backward_finite'] and
               r['shape_backward_finite'] for r in negative)
    assert any(None in r['decoded_shape_logits'] for r in result['rows'])
