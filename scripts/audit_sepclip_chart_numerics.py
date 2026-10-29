"""Compiled CUDA Beta chart numerics and real host sampler moments; mlq only.

The concentration grid diagnoses arithmetic, not training hyperparameters.
FP64 likelihood references use identical actual FP32 old/new shapes and actions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import betainc
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import beta_concentrations_host,make_beta_sampler
from cleanrl import ppo_continuous_action_sepclip_anchored_precision_chart_v13 as candidate

KAPPAS=(3.,10.,30.,100.,1e3,1e4,1e6)
MODES=(.01,.1,.5,.9,.99)


def chart_shapes(coords):
    # Same stabilized exponent ordering as v13's anchored inverse chart.
    loga=coords[...,1]-F.softplus(-coords[...,0])
    logb=coords[...,1]-F.softplus(coords[...,0])
    logits=candidate.inverse_softplus_from_log(torch.stack((loga,logb),dim=-1))
    # Exact v13 inverse-softplus -> shared shape head arithmetic.
    shapes=1+F.softplus(logits)
    return shapes,logits


def core_density(shapes,actions):
    alpha,beta=shapes.unbind(-1)
    distribution=torch.distributions.Beta(alpha[:,None],beta[:,None],validate_args=False)
    density=distribution.log_prob(actions)
    entropy=torch.distributions.Beta(alpha,beta,validate_args=False).entropy()
    return density,entropy


def chart_density(coords,actions):
    shapes,logits=chart_shapes(coords)
    density,entropy=core_density(shapes,actions)
    return density,entropy,shapes,logits


def exact_kl(old,new):
    a,b=old.unbind(-1)
    c,d=new.unbind(-1)
    k,l=a+b,c+d
    # Same term ordering as sepclip's explicit exact_policy_kl.
    return (torch.lgamma(c)+torch.lgamma(d)-torch.lgamma(l)
        -torch.lgamma(a)-torch.lgamma(b)+torch.lgamma(k)
        +(a-c)*torch.digamma(a)+(b-d)*torch.digamma(b)+(l-k)*torch.digamma(k))


def compiled(fn):
    return torch.compile(fn,fullgraph=True,dynamic=True,options={'triton.cudagraphs':False})


def shape_to_chart_score(shape_score,coords):
    """Reference ideal inverse-chart chain using actual FP32 excess shapes."""
    a=(coords[:,1]-F.softplus(-coords[:,0])).exp().double()
    b=(coords[:,1]-F.softplus(coords[:,0])).exp().double()
    q=coords[:,0].sigmoid().double()
    one_minus_q=(-coords[:,0]).sigmoid().double()
    sa,sb=shape_score.unbind(-1)
    mode=a*one_minus_q*sa-b*q*sb
    precision=a*sa+b*sb
    return torch.stack((mode,precision),-1)


def relative_norm(error,reference):
    return float(torch.linalg.vector_norm(error)/(torch.linalg.vector_norm(reference)+1e-30))


def nullable(value):
    x=value.detach().cpu().numpy()
    out=x.astype(object)
    out[~np.isfinite(x)]=None
    return out.tolist()


def actual_decode_range():
    settings=[(np.log(m/(1-m)),ell) for ell in (-120.,-105.,-100.,-90.,-80.,-50.,-20.,0.,20.,50.,80.,88.,90.,100.)
              for m in (.01,.5,.99)]
    settings.extend((m,ell) for m in (-100.,-50.,-20.,-8.,0.,8.,20.,50.,100.) for ell in (-100.,0.,20.,80.,100.))
    coords=torch.tensor(settings,device='cuda',dtype=torch.float32)
    anchor=torch.zeros_like(coords)
    # Encode zero anchor once using the candidate's exact arithmetic.
    zero=F.softplus(anchor[:,0]).log()
    offset=torch.logaddexp(zero,zero)
    trainable=torch.stack((coords[:,0],coords[:,1]-offset),-1).requires_grad_()
    decoder=compiled(candidate.decode_chart)
    decoded=decoder(anchor,trainable)
    decoded_values=decoded.detach().clone()
    decode_grad=torch.autograd.grad(decoded.sum(),trainable)[0].detach().clone()
    # Independent forward: no retained compiled backward buffers.
    next_input=trainable.detach().clone().requires_grad_()
    shapes=1+F.softplus(decoder(anchor,next_input))
    shape_values=shapes.detach().clone()
    shape_grad=torch.autograd.grad(shapes.sum(),next_input)[0].detach().clone()
    rows=[]
    for index,(mode_logit,log_excess) in enumerate(settings):
        rows.append(dict(mode_logit=float(mode_logit),log_excess_concentration=float(log_excess),
            decoded_shape_logits=nullable(decoded_values[index]),shapes=nullable(shape_values[index]),
            sum_logit_gradient=nullable(decode_grad[index]),sum_shape_gradient=nullable(shape_grad[index]),
            decoded_logits_finite=bool(torch.isfinite(decoded_values[index]).all()),
            shapes_finite=bool(torch.isfinite(shape_values[index]).all()),
            decoded_backward_finite=bool(torch.isfinite(decode_grad[index]).all()),
            shape_backward_finite=bool(torch.isfinite(shape_grad[index]).all())))
    return dict(rows=rows,nonfinite_decode_cases=sum(not r['decoded_logits_finite'] for r in rows),
        nonfinite_shape_backward_cases=sum(not r['shape_backward_finite'] for r in rows),
        protocol='Actual compiled candidate.decode_chart, FP32 zero anchor, explicit mode/log-excess-concentration chart coordinate ranges; nonfinite values stored as null, not clamped.',
        candidate_source_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest())


def sampler_moments(logits,alpha,beta,batches,seed):
    low=np.full(6,-1.,np.float32)
    high=np.full(6,1.,np.float32)
    sampler=make_beta_sampler(16,6,low,high)
    head=np.concatenate((np.full((16,6),logits[0],np.float32),np.full((16,6),logits[1],np.float32)),axis=-1)
    host_a,host_b=beta_concentrations_host(head)
    a,b=float(host_a[0,0]),float(host_b[0,0])
    rng=np.random.default_rng(seed)
    count=0
    total=0.
    squared=0.
    clipped=0
    rescale_error=0.
    for _ in range(batches):
        native_buffer,physical_buffer=sampler(head,rng)
        native=native_buffer.copy()
        physical=physical_buffer.copy()
        x=native.astype(np.float64).reshape(-1)
        count+=len(x);total+=float(x.sum());squared+=float(np.square(x).sum())
        clipped+=int(np.count_nonzero((native==np.float32(1e-6))|(native==np.float32(1-1e-6))))
        rescale_error=max(rescale_error,float(np.max(np.abs(physical-(2*native-1)))))
    mean=total/count
    variance=squared/count-mean*mean
    expected_mean=a/(a+b)
    expected_variance=a*b/((a+b)**2*(a+b+1))
    mean_se=np.sqrt(expected_variance/count)
    tail=float(betainc(a,b,1e-6)+betainc(b,a,1e-6))
    k=a+b
    excess_kurtosis=6*((a-b)**2*(k+1)-a*b*(k+2))/(a*b*(k+2)*(k+3))
    variance_se=expected_variance*np.sqrt((excess_kurtosis+2)/count)
    return dict(count=count,host_alpha=a,host_beta=b,
        host_vs_cuda_shape_abs_error=[abs(a-alpha),abs(b-beta)],sample_mean=mean,sample_variance=variance,
        expected_unclipped_mean=expected_mean,expected_unclipped_variance=expected_variance,
        mean_error_in_theoretical_se=float((mean-expected_mean)/(mean_se+1e-30)),
        relative_variance_error=float(variance/expected_variance-1),
        variance_error_in_asymptotic_se=float((variance-expected_variance)/(variance_se+1e-30)),
        actual_clipped_samples=clipped,theoretical_clip_probability=tail,
        physical_rescale_max_abs_error=rescale_error,fused=bool(sampler.fused),fallback_reason=sampler.fallback_reason,
        limitations=['Moments compare to unclipped Beta; clipping probability and counts separately reported.',
                    'Native and physical permanent buffers copied before the next sampler invocation.'])


def audit(kappas=KAPPAS,modes=MODES,batches=1024,seed=1):
    settings=[(float(k),float(m)) for k in kappas for m in modes]
    coords=torch.tensor([[np.log(m/(1-m)),np.log(k-2)] for k,m in settings],device='cuda',dtype=torch.float32)
    chart=compiled(chart_density)
    core=compiled(core_density)
    shapes_fn=compiled(chart_shapes)
    kl=compiled(exact_kl)
    with torch.no_grad():
        actual_shapes,logits=shapes_fn(coords)
        actual_shapes=actual_shapes.clone()
        logits=logits.clone()
        alpha,beta=actual_shapes.unbind(-1)
        mean=alpha/(alpha+beta)
        variance=alpha*beta/((alpha+beta).square()*(alpha+beta+1))
        standardized=torch.tensor([-2.,-1.,0.,1.,2.],device='cuda')
        actions=(mean[:,None]+variance.sqrt()[:,None]*standardized).clamp(1e-6,1-1e-6)
        lp32,entropy32=core(actual_shapes,actions)
        lp32=lp32.clone();entropy32=entropy32.clone()
        lp64,entropy64=core(actual_shapes.double(),actions.double())
        lp64=lp64.clone();entropy64=entropy64.clone()
    gradients=[]
    logit_gradients=[]
    # Independent forwards avoid retained/donated compiled backward buffers.
    for sample in range(actions.shape[1]):
        current=coords.clone().requires_grad_()
        density,*_=chart(current,actions)
        g32=torch.autograd.grad(density[:,sample].sum(),current)[0].detach().clone()
        ref_shapes=actual_shapes.double().clone().requires_grad_()
        density64,_=core(ref_shapes,actions.double())
        shape_score=torch.autograd.grad(density64[:,sample].sum(),ref_shapes)[0].detach().clone()
        reference=shape_to_chart_score(shape_score,coords)
        gradients.append((g32,reference))
        fp32_shapes=actual_shapes.clone().requires_grad_()
        fp32_density,_=core(fp32_shapes,actions)
        shape_score32=torch.autograd.grad(fp32_density[:,sample].sum(),fp32_shapes)[0].detach().clone()
        logit_gradients.append((shape_score32*logits.sigmoid(),shape_score*logits.sigmoid().double()))
    g32=torch.stack([g[0] for g in gradients],dim=1)
    g64=torch.stack([g[1] for g in gradients],dim=1)
    logit32=torch.stack([g[0] for g in logit_gradients],dim=1)
    logit64=torch.stack([g[1] for g in logit_gradients],dim=1)
    changes=[]
    for name,delta in [('mode_1e-4',(1e-4,0.)),('mode_1e-3',(1e-3,0.)),('mode_1e-2',(1e-2,0.)),
                       ('precision_1e-4',(0.,1e-4)),('precision_1e-3',(0.,1e-3)),('precision_1e-2',(0.,1e-2)),
                       ('mixed_1e-3',(1e-3,1e-3))]:
        with torch.no_grad():
            changed,_=shapes_fn(coords+torch.tensor(delta,device='cuda'))
            changed=changed.clone()
            k32=kl(actual_shapes,changed).clone()
            k64=kl(actual_shapes.double(),changed.double()).clone()
        changes.append((name,delta,changed,k32,k64))
    rows=[]
    shapes_np=actual_shapes.cpu().numpy()
    logits_np=logits.cpu().numpy()
    for index,(kappa,mode) in enumerate(settings):
        errors=g32[index].double()-g64[index]
        row=dict(requested_concentration=kappa,requested_mode=mode,
            actual_fp32_shapes=shapes_np[index].tolist(),actual_fp32_concentration=float(shapes_np[index].sum()),
            actual_fp32_mode=float((shapes_np[index,0]-1)/(shapes_np[index].sum()-2)),
            actions=actions[index].cpu().tolist(),
            logprob_fp32=lp32[index].cpu().tolist(),logprob_fp64_same_shapes=lp64[index].cpu().tolist(),
            logprob_max_abs_error=float((lp32[index].double()-lp64[index]).abs().max()),
            entropy_fp32=float(entropy32[index]),entropy_fp64_same_shapes=float(entropy64[index]),
            entropy_abs_error=float((entropy32[index].double()-entropy64[index]).abs()),
            chart_score_fp32=g32[index].cpu().tolist(),chart_score_fp64_same_shapes_jacobian=g64[index].cpu().tolist(),
            chart_score_relative_norm_error=relative_norm(errors,g64[index]),
            chart_score_max_abs_error=float(errors.abs().max()),kl_tests=[])
        row.update(softplus_logit_score_fp32=logit32[index].cpu().tolist(),
            softplus_logit_score_fp64_same_shapes=logit64[index].cpu().tolist(),
            softplus_logit_score_relative_norm_error=relative_norm(logit32[index].double()-logit64[index],logit64[index]))
        for name,delta,changed,k32,k64 in changes:
            ref=float(k64[index]);observed=float(k32[index])
            row['kl_tests'].append(dict(change=name,chart_delta=list(delta),changed_fp32_shapes=changed[index].cpu().tolist(),
                fp32=observed,fp64_same_shapes=ref,abs_error=abs(observed-ref),
                relative_error=abs(observed-ref)/(abs(ref)+1e-30),fp32_negative=observed<0))
        row['sampler']=sampler_moments(logits_np[index],float(shapes_np[index,0]),float(shapes_np[index,1]),batches,seed+index)
        rows.append(row)
    finite=all(np.isfinite(row['logprob_fp32']).all() and np.isfinite(row['chart_score_fp32']).all() for row in rows)
    return dict(protocol=dict(device='cuda',compiled=True,shape_dtype='float32',reference_dtype='float64',
        reference='Cast identical actual FP32 old/new shapes and native actions to FP64; chart score uses actual FP32 excess-shape and sigmoid chain factors in FP64.',
        sampler_num_envs=16,sampler_act_dim=6,sampler_batches=batches,seed=seed,
        concentration_grid=list(kappas),mode_grid=list(modes),
        native_actions='Five labeled mean±standard-deviation probes, clamped to sampler support.'),
        rows=rows,actual_candidate_decode_range=actual_decode_range(),summary=dict(all_fp32_densities_and_chart_scores_finite=finite,
            maximum_logprob_abs_error=max(r['logprob_max_abs_error'] for r in rows),
            maximum_chart_score_relative_norm_error=max(r['chart_score_relative_norm_error'] for r in rows),
            negative_fp32_kl_cases=sum(t['fp32_negative'] for r in rows for t in r['kl_tests'])),
        limitations=['FP64 is a numerical reference on identical FP32 shapes, not a new distribution or learned policy.',
            'The grid measures likelihood/gradient/KL numerical error; it does not select a training concentration.',
            'Relative KL error is ill-scaled as the true KL approaches zero; absolute error and sign reported.',
            'No concentration cap or likelihood dtype change is implemented by this audit.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--sampler-batches',type=int,default=1024)
    parser.add_argument('--seed',type=int,default=1)
    cli=parser.parse_args()
    if cli.sampler_batches<1:parser.error('Need positive sampler batches')
    configure_runtime(matmul_precision='highest',allow_tf32=False)
    # Multiple saved policies and Jacobian closures intentionally compile here.
    # Retain fullgraph behavior rather than exhausting the default eight guards.
    torch._dynamo.config.recompile_limit = 64
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; submit through mlq')
    result=audit(batches=cli.sampler_batches,seed=cli.seed)
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out=Path(cli.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
