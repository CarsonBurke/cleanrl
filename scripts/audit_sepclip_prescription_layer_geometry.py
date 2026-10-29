"""Actual captured layer chords, joint Fisher geometry and stored PPO credit.

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


def beta_shape_fisher(alpha, beta):
    """FP64 shape Fisher at actual rounded FP32 shapes; no latent inversion."""
    if alpha.shape != beta.shape or alpha.ndim != 2 or bool((alpha < 1).any()) or bool((beta < 1).any()):
        raise ValueError('Require state/action Beta shapes >=1')
    alpha = alpha.double(); beta = beta.double(); finite(alpha, beta)
    shared = torch.polygamma(1, alpha + beta)
    f = torch.stack((torch.stack((torch.polygamma(1, alpha) - shared, -shared), -1),
                     torch.stack((-shared, torch.polygamma(1, beta) - shared), -1)), -2)
    finite(f)
    _, info = torch.linalg.cholesky_ex(f)
    if bool((info != 0).any()):
        raise ValueError('Numerically non-SPD FP64 shape Fisher; geometry withheld')
    return f


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
    fisher=beta_shape_fisher(alpha,beta)
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
    return dict(layer_partition=groups,chords=results,epoch_trace=trace_geometry(old,replay.trace['trace'],groups,replay.constraint_axes),
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
