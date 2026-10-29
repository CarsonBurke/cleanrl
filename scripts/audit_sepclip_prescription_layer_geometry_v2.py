"""V2 captured layer geometry: explicit recurrence/asymptotic trigamma.

Preserves failed v1/11905; CUDA FP64 torch.polygamma was not accurate enough
for the exact uniform-Fisher contract. Numerical validation is pending.

CUDA/MLQ only. All layers and cross terms on the complete saved old batch;
no candidate proposals, physical-field fitting or causal optimizer attribution.
Full-chord numerical certification is a separate required diagnostic.
"""
import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sepclip_prescription_replay import MODULES, load_replay, sha256, require_pin, validate_provenance


def finite(*values):
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError('Nonfinite numerical geometry input/output')


def layer_partition(names):
    """Weight/bias share a layer; gain/mixing scalars retain their own route."""
    groups = {}
    for name in names:
        label = name.rsplit('.', 1)[0] if name.endswith(('.weight', '.bias')) else name
        groups.setdefault(label, []).append(name)
    if not groups or sum(map(len, groups.values())) != len(set(names)):
        raise ValueError('Need a nonempty unique complete actor parameter partition')
    return groups


TRIGAMMA_SHIFT = 32
# Bernoulli B2..B14 after the leading 1/x + 1/(2x²).
ASYMPTOTIC_TERMS = ((1,1.),(2,.5),(3,1/6),(5,-1/30),(7,1/42),
                    (9,-1/30),(11,5/66),(13,-691/2730),(15,7/6))
B16_ABS = 3617/510
UNIT_ROUNDOFF = torch.finfo(torch.float64).eps/2
GAMMA512 = 512*UNIT_ROUNDOFF/(1-512*UNIT_ROUNDOFF)
# Conservative propagation allowance for subnormal elementary intermediates.
UNDERFLOW_ALLOWANCE = 4096*torch.finfo(torch.float64).tiny*torch.finfo(torch.float64).eps
FISHER_MAX_RELATIVE_QUADRATIC_ERROR = .01


def _domain(x):
    finite(x)
    if x.dtype != torch.float64 or bool((x < 1).any()) or bool((x > 1e39).any()):
        raise ValueError('Trigamma numerical contract requires FP64 arguments in [1,1e39]')


def _powers(reciprocal):
    values=[torch.ones_like(reciprocal)]
    for _ in range(17): values.append(values[-1]*reciprocal)
    return values


def _error_budget(absolute_work, truncation):
    # Each arithmetic path (including rounded argument/coefficient inputs) has
    # fewer than512 elementary operations; alternating terms use absolute work.
    # This is conditional on standard RN FP64 +,*,/ and gradual underflow, NOT
    # a hardware/libdevice theorem. No polygamma/pow transcendental is used.
    return (GAMMA512*absolute_work+truncation+UNDERFLOW_ALLOWANCE)*(1+2*GAMMA512)


def trigamma_recurrence(x):
    """Value plus conditional absolute error bound, x>=1, fixed32 recurrence.

    coth partial fractions give a sign-definite first-omitted-B16 remainder
    after B14. Shifting to x+32 makes truncation tiny without backend polygamma.
    """
    _domain(x)
    recurrence=torch.zeros_like(x)
    for index in range(TRIGAMMA_SHIFT):
        reciprocal=1/(x+index)
        recurrence=recurrence+reciprocal*reciprocal
    y=x+TRIGAMMA_SHIFT; powers=_powers(1/y)
    asymptotic=torch.zeros_like(x); work=recurrence.clone()
    for power,coefficient in ASYMPTOTIC_TERMS:
        term=coefficient*powers[power]
        asymptotic=asymptotic+term;work=work+abs(coefficient)*powers[power]
    value=recurrence+asymptotic
    bound=_error_budget(work,B16_ABS*powers[17])
    finite(value,bound)
    return value,bound


def trigamma_difference(x,separation):
    """Stable T(x)-T(x+b), avoiding catastrophic asymmetric subtraction.

    All reciprocal power differences use D=(b/(x+b))/x and
    u^n-v^n = D sum_i u^(n-1-i)v^i. b is retained even when x+b rounds to x.
    Paired omitted-remainder bound is |B16|*(y^-17-(y+b)^-17), evaluated stably.
    """
    _domain(x);_domain(separation)
    if x.shape != separation.shape: raise ValueError('Trigamma difference shape mismatch')
    recurrence=torch.zeros_like(x)
    for index in range(TRIGAMMA_SHIFT):
        p=x+index;q=p+separation;u=1/p;v=1/q
        difference=(separation/q)*u
        recurrence=recurrence+difference*(u+v)
    y=x+TRIGAMMA_SHIFT;q=y+separation;u=1/y;v=1/q
    difference=(separation/q)*u;up=_powers(u);vp=_powers(v)
    def power_difference(n):
        series=torch.zeros_like(x)
        for index in range(n): series=series+up[n-1-index]*vp[index]
        return difference*series
    asymptotic=torch.zeros_like(x);work=recurrence.clone()
    for power,coefficient in ASYMPTOTIC_TERMS:
        term=power_difference(power)
        asymptotic=asymptotic+coefficient*term;work=work+abs(coefficient)*term
    value=recurrence+asymptotic
    bound=_error_budget(work,B16_ABS*power_difference(17))
    finite(value,bound)
    return value,bound


def beta_shape_fisher(alpha,beta,return_contract=False):
    """FP64 shape Fisher with error-aware normalized small-mode gate.

    SPD alone is insufficient: retain geometry only when a conditional entry
    error budget implies at most1% relative error of every quadratic pairing.
    No ridge, clipping, cancellation floor or independent latent inversion.
    """
    if alpha.shape != beta.shape or alpha.ndim != 2 or alpha.numel()==0:
        raise ValueError('Require nonempty state/action Beta shapes')
    alpha=alpha.double();beta=beta.double();_domain(alpha);_domain(beta)
    diagonal_a,error_a=trigamma_difference(alpha,beta)
    diagonal_b,error_b=trigamma_difference(beta,alpha)
    shared,error_shared=trigamma_recurrence(alpha+beta)
    f=torch.stack((torch.stack((diagonal_a,-shared),-1),torch.stack((-shared,diagonal_b),-1)),-2)
    bounds=torch.stack((torch.stack((error_a,error_shared),-1),torch.stack((error_shared,error_b),-1)),-2)
    finite(f,bounds)
    if bool((diagonal_a<=0).any()) or bool((diagonal_b<=0).any()):
        raise ValueError('Numerically nonpositive Fisher diagonal; geometry withheld')
    scale=diagonal_a.sqrt()*diagonal_b.sqrt()
    correlation=shared/scale
    normalized_gap=1-correlation.abs()
    normalized_error=torch.maximum(error_a/diagonal_a+error_shared/scale,error_b/diagonal_b+error_shared/scale)
    # Inflate comparison arithmetic and gap roundoff; congruence scaling is
    # chosen from computed positive diagonals, so it also applies to exact F.
    gap_lower=normalized_gap-8*UNIT_ROUNDOFF
    eta=normalized_error*(1+2*GAMMA512)
    true_gap_lower=gap_lower-eta
    relative_error=eta/true_gap_lower.clamp_min(torch.finfo(torch.float64).tiny)
    if bool((true_gap_lower<=0).any()) or bool((relative_error>FISHER_MAX_RELATIVE_QUADRATIC_ERROR).any()):
        raise ValueError('Fisher small-mode error budget unresolved; geometry withheld')
    _,info=torch.linalg.cholesky_ex(f)
    if bool((info!=0).any()): raise ValueError('Numerically non-SPD FP64 Fisher; geometry withheld')
    if not return_contract:return f
    return f,dict(arithmetic='CUDA FP64 fixed32 recurrence + B14 asymptotic; stable paired differences',
        shift=TRIGAMMA_SHIFT,first_omitted_coefficient='B16=-3617/510',
        maximum_entry_absolute_error_budget=float(bounds.max()),
        minimum_normalized_small_mode_gap_lower=float(gap_lower.min()),
        minimum_true_normalized_small_mode_gap_lower=float(true_gap_lower.min()),
        maximum_conditional_relative_quadratic_error=float(relative_error.max()),
        relative_quadratic_error_denominator='Exact mathematical Fisher quadratic, not computed quadratic: eta/(gap_lower-eta)',
        retained_relative_quadratic_error_limit=FISHER_MAX_RELATIVE_QUADRATIC_ERROR,
        scope='Conditional engineering bound assuming standard RN FP64 elementary arithmetic and gradual underflow; independently validate on queued CUDA references. Not a hardware proof.')


def joint_gram(shape_responses, fisher):
    """G_ij = sum_state,action d_i^T F d_j / number_of_states."""
    if shape_responses.ndim != 4 or shape_responses.shape[-1] != 2 or tuple(fisher.shape) != tuple(shape_responses.shape[1:]) + (2,):
        raise ValueError('Expected responses[layer,state,action,2], Fisher[state,action,2,2]')
    if shape_responses.shape[0] == 0 or shape_responses.shape[1] == 0:
        raise ValueError('Empty layer/state geometry')
    d = shape_responses.double(); fisher = fisher.double(); finite(d, fisher)
    gram = torch.einsum('isac,sacd,jsad->ij', d, fisher, d) / d.shape[1]
    finite(gram)
    return gram


def beta_kl(old_alpha, old_beta, new_alpha, new_beta):
    """Old||new Beta KL, sum actions then mean states, all FP64 arithmetic."""
    if any(value.shape != old_alpha.shape for value in (old_beta, new_alpha, new_beta)) or old_alpha.ndim != 2:
        raise ValueError('KL shape mismatch')
    a,b,c,d = (value.double() for value in (old_alpha,old_beta,new_alpha,new_beta)); finite(a,b,c,d)
    if any(bool((value < 1).any()) for value in (a,b,c,d)):
        raise ValueError('Actual rounded shapes must be >=1')
    logb = lambda x,y: torch.lgamma(x)+torch.lgamma(y)-torch.lgamma(x+y)
    values = logb(c,d)-logb(a,b)+(a-c)*torch.digamma(a)+(b-d)*torch.digamma(b)+(c+d-a-b)*torch.digamma(a+b)
    finite(values)
    return values.sum(-1).mean()


def scalar_pairing(covector, motion):
    if set(covector) != set(motion):
        raise ValueError('Incomplete parameter pairing')
    for key in covector:
        if covector[key].shape != motion[key].shape:
            raise ValueError('Parameter pairing shape mismatch')
        finite(covector[key], motion[key])
    return sum(float((covector[key].double()*motion[key].double()).sum()) for key in covector)


def clipped_ascent(ratio, advantage, clip_lower, clip_upper):
    """Source PPO sign-conditioned ascent; no clipfrac-based gradient mask."""
    return -torch.maximum(-advantage*ratio,-advantage*ratio.clamp(1-clip_lower,1+clip_upper)).mean()


def parameter_norm(bank):
    finite(*bank.values())
    return sum(float(value.double().square().sum()) for value in bank.values()) ** .5


class StageProbe(torch.nn.Module):
    """Source-dispatched stage observations; final output checked against actor."""
    def __init__(self, actor, additive):
        super().__init__()
        self.actor = actor
        self.additive = additive

    def forward(self, x):
        actor = self.actor
        stages = {}
        if self.additive:
            trunk = actor.trunk
            h = trunk.in_proj(x)
            stages['signed_projection'] = h
            for index, (block, gate, norm) in enumerate(zip(trunk.blocks, trunk.block_gates, trunk.block_norms)):
                inputs = norm(h) * trunk.branch_input_scale
                branch = block(inputs)
                amplitude = torch.sigmoid(gate)
                residual = amplitude * branch
                prefix = f'block{index}.'
                stages.update({prefix+'stream_before': h, prefix+'normalized_branch_input': inputs,
                    prefix+'gate_preactivation': block.gate(inputs), prefix+'up_preactivation': block.up(inputs),
                    prefix+'raw_branch': branch, prefix+'weighted_branch': residual,
                    prefix+'sigmoid_amplitude': amplitude})
                h = h + residual
                stages[prefix+'stream_after'] = h
            features = trunk.final_norm(h) * trunk.output_scale
        else:
            raw_first = actor.first(x)
            h = F.normalize(raw_first, p=2, dim=-1)
            raw_second = actor.second(8.0*h)
            branch = F.normalize(raw_second, p=2, dim=-1)
            stages.update(first_raw=raw_first, first_unit=h, second_input=8.0*h,
                second_raw=raw_second, second_unit=branch, unit_sum=h+branch)
            features = 8.0*F.normalize(h+branch, p=2, dim=-1)
        before_gain = actor.head(features)
        stages.update(head_features=features, head_before_gain=before_gain, readout_gain=actor.readout_gain)
        return actor.readout_gain*before_gain, stages


def stage_summary(stages):
    result = {}
    for name, value in stages.items():
        finite(value)
        x = value.double()
        result[name] = dict(shape=list(x.shape), rms=float(x.square().mean().sqrt()),
            min=float(x.min()), max=float(x.max()), mean=float(x.mean()))
        if x.ndim == 2:
            rms = x.square().mean(-1).sqrt()
            result[name]['state_rms_quantiles'] = torch.quantile(rms, torch.tensor([0.,.1,.5,.9,1.], device=x.device,dtype=torch.float64)).cpu().tolist()
        if name.endswith('gate_preactivation'):
            # Full SiTU gate-factor derivative, not tanh-cap derivative alone.
            t = torch.tanh(x/4); sigmoid = torch.sigmoid(x)
            derivative = (1-t.square())*sigmoid+4*t*sigmoid*(1-sigmoid)
            result[name]['full_situ_gate_factor_derivative'] = dict(min=float(derivative.min()),max=float(derivative.max()),
                mean=float(derivative.mean()),negative_fraction=float((derivative<0).double().mean()),
                absolute_quantiles=torch.quantile(derivative.abs().flatten(),torch.tensor([0.,.1,.5,.9,1.],device=x.device,dtype=torch.float64)).cpu().tolist())
    ratios = {}
    for name,stream in stages.items():
        if not name.endswith('stream_before'): continue
        prefix = name.removesuffix('stream_before')
        branch = stages[prefix+'raw_branch'].double(); weighted = stages[prefix+'weighted_branch'].double(); stream = stream.double()
        stream_norm = float(stream.norm()); branch_norm = float(branch.norm()); weighted_norm = float(weighted.norm())
        ratios[prefix] = dict(raw_branch_to_stream_rms_ratio=branch_norm/stream_norm if stream_norm>0 else None,
            weighted_branch_to_stream_rms_ratio=weighted_norm/stream_norm if stream_norm>0 else None,
            stream_weighted_branch_cosine=float((stream*weighted).sum())/(stream_norm*weighted_norm) if stream_norm>0 and weighted_norm>0 else None,
            summed_stream_second_moment_identity_residual=float((stages[prefix+'stream_after'].double().square().sum()-stream.square().sum()-weighted.square().sum()-2*(stream*weighted).sum())))
    return dict(activations=result,residual_ratios=ratios)


def stage_diagnostics(replay, observations):
    additive = hasattr(replay.module, 'AdditivePreRMSActor') and isinstance(replay.agent.actor,replay.module.AdditivePreRMSActor)
    probe = StageProbe(replay.agent.actor,additive)
    def measure(bank,x):
        return torch.func.functional_call(probe,{'actor.'+name:value for name,value in bank.items()},(x,),strict=True)
    compiled = torch.compile(measure,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    results = {}
    with torch.no_grad():
        for label,bank in replay.banks.items():
            logits,stages=compiled(bank,observations)
            logits=logits.clone(); stages={name:value.clone() for name,value in stages.items()}
            actual=replay.compiled_forward(bank,observations)[0].clone()
            finite(logits,actual)
            if not torch.allclose(logits,actual,rtol=2e-5,atol=3e-6):
                raise ValueError('Source-dispatched stage probe differs from actual compiled actor')
            results[label]=stage_summary(stages)
            results[label]['actual_forward_max_abs_error']=float((logits-actual).abs().max())
    return dict(architecture='additive_prerms' if additive else 'original_unit_residual',banks=results,
        scope='All three banks evaluated on the same saved old-policy observations and frozen per-row normalization; not their own future occupancy. RMS ratios and mix/gain changes do not alone establish a cause of return collapse.')


def trace_geometry(old, trace, groups, axes):
    """Saved gradients are clipped loss covectors at each epoch's own weights.

    Cumulative weights are reconstructions from saved deltas, not independent
    epoch checkpoints. Intrinsic v15 ambient_delta is its recorded endpoint
    before global normalization, not a hypothetical Euclidean proposal.
    """
    current = {name:value.clone() for name,value in old.items()}
    rows=[]
    for entry in trace:
        stats={}
        for label,names in groups.items():
            gradient={name:entry['gradient_after_clip']['actor.'+name].to(current[name].device) for name in names}
            ambient={name:entry['ambient_delta']['actor.'+name].to(current[name].device) for name in names}
            actual={name:entry['retracted_delta']['actor.'+name].to(current[name].device) for name in names}
            weights={name:current[name] for name in names}
            weight_norm=parameter_norm(weights); actual_norm=parameter_norm(actual)
            constraint={name:axes.get('actor.'+name) for name in names}
            radial={}
            for name,axis in constraint.items():
                if axis is None:
                    continue
                w=weights[name].double(); g=gradient[name].double()
                denominator=w.square().sum(axis,keepdim=True)
                if bool((denominator <= 0).any()): raise ValueError('Zero constrained basepoint')
                normal=w*(w*g).sum(axis,keepdim=True)/denominator
                radial[name]=dict(axis=axis, clipped_gradient_normal_norm=float(normal.norm()),
                    clipped_gradient_tangent_norm=float((g-normal).norm()),
                    reconstructed_axis_unit_max_abs_error=float((denominator.sqrt()-1).abs().max()))
            stats[label]=dict(parameter_names=names,constraint_axes=constraint,
                geometry='mixed' if len({axis is None for axis in constraint.values()})>1 else ('free_euclidean' if all(axis is None for axis in constraint.values()) else 'unit_axes'),
                reconstructed_weight_norm=weight_norm,gradient_after_clip_norm=parameter_norm(gradient),
                ambient_delta_norm=parameter_norm(ambient),retracted_delta_norm=actual_norm,
                actual_relative_weight_motion=actual_norm/weight_norm if weight_norm>0 else None,
                negative_clipped_loss_gradient_dot_ambient=-scalar_pairing(gradient,ambient),
                negative_clipped_loss_gradient_dot_actual=-scalar_pairing(gradient,actual),
                normalization_correction_norm=parameter_norm({name:actual[name]-ambient[name] for name in names}),
                matrix_weight_norms={name:dict(frobenius=float(value.double().norm()),
                    row_l2_min=float(value.double().norm(dim=1).min()),row_l2_max=float(value.double().norm(dim=1).max()),
                    column_l2_min=float(value.double().norm(dim=0).min()),column_l2_max=float(value.double().norm(dim=0).max()))
                    for name,value in weights.items() if value.ndim==2},
                constrained_decomposition=radial)
        rows.append(dict(epoch=entry['epoch'],update=entry['update'],layers=stats,
            matrix_geometry=entry.get('matrix_geometry','ambient'),
            matrix_delta_stage=entry.get('matrix_delta_stage','ambient proposal before global normalization')))
        for name in current:
            current[name].add_(entry['retracted_delta']['actor.'+name].to(current[name].device))
        finite(*current.values())
    return rows


def audit_layers(replay):
    args=replay.args
    if args.num_minibatches != 1:
        raise ValueError('Require full-batch update: trace does not independently record shuffled minibatch membership')
    actor=replay.agent.actor
    old={name:value.detach().clone(memory_format=torch.contiguous_format) for name,value in replay.banks['old'].items()}
    obs=replay.snapshot['batch']['obs'].detach().clone(memory_format=torch.contiguous_format)
    groups=layer_partition(tuple(old)); labels=list(groups)
    options=dict(fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    def primal(bank,x):
        z=torch.func.functional_call(actor,bank,(x,),strict=True)
        a,b=(1+F.softplus(z)).chunk(2,-1)
        return z,a,b
    def directional(bank,tangent,x):
        return torch.func.jvp(lambda p:primal(p,x),(bank,),(tangent,))
    forward=torch.compile(primal,**options); jvp=torch.compile(directional,**options)
    batch=replay.snapshot['batch']
    advantages=batch['advantages']
    if args.norm_adv: advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
    def policy_ascent(bank,x,native,logprob,advantage):
        _,a,b=primal(bank,x)
        ratio=(replay.agent.action_logprob(a,b,native)-logprob).exp()
        return clipped_ascent(ratio,advantage,args.clip_coef,args.clip_coef_upper)
    gradient=torch.compile(torch.func.grad(policy_ascent),**options)
    with torch.no_grad():
        reference=tuple(value.clone() for value in forward(old,obs))
        ascent={name:value.clone() for name,value in gradient(old,obs,batch['native'],batch['logprob'],advantages).items()}
    finite(*reference,*ascent.values()); _,alpha,beta=reference
    fisher,fisher_contract=beta_shape_fisher(alpha,beta,return_contract=True)
    results={}
    for label,direction,bank in zip(('first','endpoint'),replay.deltas,(replay.banks['first'],replay.banks['endpoint'])):
        direction={name:value.detach().clone(memory_format=torch.contiguous_format) for name,value in direction.items()}
        shape_rows=[]; logits_rows=[]; layer_credit={}
        with torch.no_grad():
            for layer,names in groups.items():
                isolated={name:direction[name].clone() if name in names else torch.zeros_like(value) for name,value in old.items()}
                p,d=jvp(old,isolated,obs); p=tuple(value.clone() for value in p); d=tuple(value.clone() for value in d)
                finite(*p,*d)
                if any(not torch.allclose(left,right,rtol=2e-5,atol=3e-6) for left,right in zip(reference,p)):
                    raise ValueError('Layer JVP graph primal disagrees with independent graph')
                logits_rows.append(d[0].double());shape_rows.append(torch.stack((d[1],d[2]),-1).double())
                layer_credit[layer]=scalar_pairing({name:ascent[name] for name in names},{name:direction[name] for name in names})
            _,total_jvp=jvp(old,direction,obs); total_jvp=tuple(value.clone() for value in total_jvp)
            actual=tuple(value.clone() for value in forward(bank,obs))
        finite(*actual,*total_jvp)
        responses=torch.stack(shape_rows); gram=joint_gram(responses,fisher)
        summed_shape=responses.sum(0);full_shape=torch.stack((total_jvp[1],total_jvp[2]),-1).double()
        summed_logits=torch.stack(logits_rows).sum(0); full_logits=total_jvp[0].double()
        observed=actual[0].double()-reference[0].double()
        actual_norm=float(observed.norm()); linear_norm=float(full_logits.norm())
        diagonal=float(gram.diagonal().sum()); joint=float(gram.sum())
        summed_layer_energy=float(joint_gram(summed_shape.unsqueeze(0),fisher)[0,0])
        full_energy=float(joint_gram(full_shape.unsqueeze(0),fisher)[0,0])
        results[label]=dict(layer_order=labels,full_joint_shape_fisher_gram=gram.cpu().tolist(),
            diagonal_energy_sum=diagonal,cross_term_sum=joint-diagonal,joint_energy=joint,
            summed_layer_response_energy=summed_layer_energy,
            gram_sum_vs_summed_layer_energy_abs_error=abs(joint-summed_layer_energy),
            gram_symmetry_max_abs_error=float((gram-gram.T).abs().max()),
            gram_numerical_min_eigenvalue=float(torch.linalg.eigvalsh(gram).min()),
            independent_full_jvp_energy=full_energy,joint_vs_full_jvp_energy_abs_error=abs(joint-full_energy),
            layer_sum_vs_full_shape_jvp_l2_error=float((summed_shape-full_shape).norm()),
            layer_sum_vs_full_logit_jvp_l2_error=float((summed_logits-full_logits).norm()),
            layer_logit_jvp_norms={name:float(value.norm()) for name,value in zip(labels,logits_rows)},
            old_stored_clipped_ppo_ascent_credit_by_layer=layer_credit,
            old_stored_clipped_ppo_ascent_credit_full=scalar_pairing(ascent,direction),
            finite_actual_old_to_bank_kl=float(beta_kl(alpha,beta,actual[1],actual[2])),
            first_order_half_fisher_energy=.5*joint,
            full_logit_jvp_norm=linear_norm,finite_actual_logit_delta_norm=actual_norm,
            finite_minus_linear_logit_l2_error=float((observed-full_logits).norm()),
            finite_minus_linear_relative_to_jvp=float((observed-full_logits).norm())/linear_norm if linear_norm>0 else None,
            zero_finite_logit_response_elements=int((observed==0).sum()),
            rounded_shape_floor_counts={name:dict(old=int((left==1).sum()),bank=int((right==1).sum()))
                for name,left,right in zip(('alpha','beta'),reference[1:],actual[1:])})
    return dict(fisher_numerical_contract=fisher_contract,layer_partition=groups,chords=results,epoch_trace=trace_geometry(old,replay.trace['trace'],groups,replay.constraint_axes),
        stage_diagnostics=stage_diagnostics(replay,obs),
        optimizer_metadata=dict(actor_optimizer=args.actor_optimizer,matrix_method=args.matrix_method,
            normuon_network=args.normuon_network,matrix_precision=args.matrix_precision,
            slow_clock=args.slow_clock,matrix_geometry=getattr(args,'matrix_geometry','ambient'),
            saved_auxiliary_optimizer_present=replay.snapshot.get('optimizer') is not None,
            saved_matrix_optimizer_present=replay.snapshot.get('matrix_optimizer') is not None,
            history_scope='Actual Args and checkpoint-state presence only; no optimizer moments are decoded, matched to layers, transported or used to propose new steps.'),
        scope='Full saved old batch, action-summed/state-mean Fisher; actual compiled smooth shape JVP at old weights and finite FP32 bank shapes. All cross terms retained; no physical PG or return-efficiency inference.',
        limitations=['First bank is reconstructed old+stored delta; endpoint bank is independently recorded exact model_after_update.',
            'Compiled JVP numerical certification requires the separate full captured chord audit; finite-vs-linear discrepancy mixes curvature and rounding.',
            'Stored PPO ascent credit uses exact sign-conditioned clipped surrogate on fixed old rollout and its saved GAE; it is not physical PG.',
            'Epoch pairings use each recorded clipped loss gradient at reconstructed preceding weights, not the old-policy score or a decomposition of finite returns.',
            'No hypothetical optimizer updates, tangent projection of free body weights, physical-field fitting, optimizer-state reconstruction or candidate selection.'])


def analyze(snapshot_path,trace_path,**kwargs):
    sources={str(Path(__file__).resolve()):sha256(__file__),
        str(Path(__file__).with_name('sepclip_prescription_replay.py').resolve()):sha256(Path(__file__).with_name('sepclip_prescription_replay.py'))}
    replay=load_replay(snapshot_path,trace_path,**kwargs)
    result=audit_layers(replay)
    for path,pin in ((snapshot_path,kwargs['snapshot_sha256']),(trace_path,kwargs['trace_sha256']),
                     (kwargs['manifest_path'],kwargs['manifest_sha256']),(kwargs['execution_path'],kwargs['execution_sha256'])): require_pin(path,pin)
    for path,pin in sources.items(): require_pin(path,pin)
    validate_provenance(kwargs['module_name'],kwargs['manifest_path'],kwargs['manifest_sha256'],kwargs['execution_path'],kwargs['execution_sha256'],kwargs['exp_name'])
    result.update(replay_validation=replay.validation,independently_recorded_logits=False,source_sha256=sources,full_saved_batch=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot','trace','manifest-path','execution-path','output'): parser.add_argument('--'+name,type=Path,required=True)
    for name in ('snapshot-sha256','trace-sha256','manifest-sha256','execution-sha256','exp-name'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--module-name',choices=tuple(MODULES),required=True)
    cli=vars(parser.parse_args());snapshot=cli.pop('snapshot');trace=cli.pop('trace');output=cli.pop('output')
    result=analyze(snapshot,trace,**cli);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
