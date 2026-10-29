"""Actual sepclip update versus matched conditional physical PG; mlq/CUDA only.

Local Fisher/Cauchy comparisons are on explicitly saved dense states, with the
continuation policy held fixed by the physical oracle. They are not a global
optimal-control certificate or a derivative through future policy decisions.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jacrev
from scipy.stats import t as student_t

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cleanrl.shared.runtime import configure_runtime
from scripts.audit_sepclip_oracle_geometry import (
    beta_geometry,tangent_jacobian,threshold_sensitivity,
    validate_dense_state_mapping,validate_saved_policy_logits,
)
from scripts.audit_sepclip_closed_loop_update import reconstruct_banks


def paired(values):
    return torch.stack(values.chunk(2,-1),-1)


def quadratic(left,right,fisher):
    return .5*torch.einsum('sak,sakl,sal->',paired(left),fisher,paired(right))/left.shape[0]


def score_pairing(score,change):
    return (score*change).sum(-1).mean()


def fixed_motion_objective_pairings(horizon,motions,replica_count,first_action_only):
    """Pair every objective with identical saved motions; never fit a direction."""
    if first_action_only is not True:
        raise ValueError('Require explicit conditional first-action oracle scope')
    if not motions or any(not bank for bank in motions.values()) or not isinstance(replica_count,int) or replica_count<2:
        raise ValueError('Require fixed motions and >=2 declared integer replicas')
    first=next(iter(next(iter(motions.values())).values()))
    if first.ndim!=2 or first.shape[0]<2 or first.shape[1]%2:
        raise ValueError('Require state-by-paired-logit motions')
    steps=horizon.get('horizon');discount=horizon.get('discount');tail=horizon.get('tail_weight')
    if not isinstance(steps,int) or steps<1 or not isinstance(discount,(int,float)) or not 0<discount<=1:
        raise ValueError('Require finite-horizon discount metadata')
    if not isinstance(tail,(int,float)) or not np.isfinite(tail) or not np.isclose(tail,discount**steps,rtol=1e-12,atol=1e-15):
        raise ValueError('Require consistent critic-tail weight metadata')
    for bank in motions.values():
        for motion in bank.values():
            if motion.shape!=first.shape or not bool(torch.isfinite(motion).all()):
                raise ValueError('Fixed motions must have identical finite state/logit layout')
    def summarize(gains,mean_field_pairing):
        average=gains.mean();sd=gains.std(unbiased=True);se=sd/replica_count**.5
        critical=float(student_t.ppf(.975,replica_count-1))
        return dict(replica_values=gains.tolist(),mean=float(average),
            conditional_mc_standard_error=float(se),
            conditional_student_t95_interval=[float(average-critical*se),float(average+critical*se)],
            mean_field_pairing=float(mean_field_pairing))
    output={}
    for objective in ('finite_raw','finite_normalized','bootstrap_normalized'):
        field=horizon.get('policy_logit_pg',{}).get(objective)
        if not isinstance(field,dict) or 'mean' not in field or 'replicate_values' not in field:
            raise ValueError(f'Missing objective mean/replicas: {objective}')
        mean=torch.as_tensor(field['mean'],device=first.device,dtype=torch.float64)
        replicas=torch.as_tensor(field['replicate_values'],device=first.device,dtype=torch.float64)
        if mean.shape!=first.shape or replicas.shape!=(replica_count,*first.shape):
            raise ValueError(f'Objective replica/state/logit layout mismatch: {objective}')
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(replicas).all()):
            raise ValueError(f'Nonfinite objective field: {objective}')
        if not torch.allclose(mean,replicas.mean(0),rtol=1e-12,atol=1e-14):
            raise ValueError(f'Objective mean differs from replica mean: {objective}')
        cases={}
        for label,bank in motions.items():
            rows={}
            for kind,motion in bank.items():
                gains=torch.stack([score_pairing(replica,motion) for replica in replicas])
                rows[kind]=summarize(gains,score_pairing(mean,motion))
            cases[label]=rows
        output[objective]=dict(cases=cases,replica_count=replica_count,state_count=first.shape[0],
            horizon=steps,discount=discount,tail_weight=tail,
            scope='Conditional first-action covector paired with a fixed actual motion; finite-logit pairing is not a finite return or an all-time derivative.',
            interval_scope='Approximate Student-t conditional MC interval; fixed states and independent replicas, no state/seed uncertainty; unstable with two replicas.')
    finite=output['finite_normalized'];bootstrap=output['bootstrap_normalized']
    tail_cases={}
    for label,bank in motions.items():
        rows={}
        for kind in bank:
            f=finite['cases'][label][kind];b=bootstrap['cases'][label][kind]
            differences=torch.tensor(b['replica_values'],dtype=torch.float64)-torch.tensor(f['replica_values'],dtype=torch.float64)
            rows[kind]=summarize(differences,b['mean_field_pairing']-f['mean_field_pairing'])
        tail_cases[label]=rows
    output['paired_estimated_tail_increment']=dict(cases=tail_cases,replica_count=replica_count,
        state_count=first.shape[0],horizon=steps,discount=discount,tail_weight=tail,
        scope='Bootstrap-normalized minus finite-normalized conditional first-action pairing: added estimated terminal critic reward, not true critic bias.',
        replica_alignment='Identical replica index assumed from the same dense producer fields; without replica IDs this does not detect independent reordering.',
        interval_scope=finite['interval_scope'])
    return output


def coordinate_changes(logits,change):
    a,b,sa,sb,_=beta_geometry(logits.double())
    k=a+b;mu=a/k
    da,db=(logits.double().sigmoid()*change).chunk(2,-1)
    dk=da+db;dm=((1-mu)*da-mu*db)/k
    mean=torch.cat((k*dm/sa,-k*dm/sb),-1)
    concentration=torch.cat((mu*dk/sa,(1-mu)*dk/sb),-1)
    return mean,concentration,dm,dk/k


def motion_metrics(logits,change,score,fisher):
    cost=quadratic(change,change,fisher)
    gain=score_pairing(score,change)
    field=paired(score)
    inverse=torch.linalg.solve(fisher,field.unsqueeze(-1)).squeeze(-1)
    dual_squared=(field*inverse).sum((-1,-2)).mean()
    bound=torch.sqrt((2*cost*dual_squared).clamp_min(0))
    mean,kappa,dm,dlogk=coordinate_changes(logits,change)
    gram=torch.stack((torch.stack((quadratic(mean,mean,fisher),quadratic(mean,kappa,fisher))),
                      torch.stack((quadratic(kappa,mean,fisher),quadratic(kappa,kappa,fisher)))))
    result=dict(fisher_quadratic=float(cost),oracle_first_order_gain=float(gain),
        unconstrained_oracle_dual_squared=float(dual_squared),unconstrained_cauchy_gain_bound=float(bound),
        efficiency_vs_unconstrained=float(gain/bound) if float(bound)>0 else None,
        physical_mean_motion_rms=float(2*dm.square().mean().sqrt()),
        log_concentration_motion_rms=float(dlogk.square().mean().sqrt()),
        mean_concentration_fisher_gram=gram.tolist(),
        mean_only_fisher_quadratic=float(gram[0,0]),concentration_only_fisher_quadratic=float(gram[1,1]),
        mean_concentration_cross_fisher_quadratic=float(gram.sum()-gram.diagonal().sum()),
        mean_only_oracle_gain=float(score_pairing(score,mean)),
        concentration_only_oracle_gain=float(score_pairing(score,kappa)),
        coordinate_decomposition_relative_error=float((mean+kappa-change).norm()/change.norm().clamp_min(1e-300)))
    return result


def retained_range_motion_metrics(geometry,change,score):
    """Project BOTH score and motion into the same retained whitened range."""
    field=whiten_field(geometry,score)
    motion=(geometry['root']@paired(change).unsqueeze(-1)).squeeze(-1).flatten()/change.shape[0]**.5
    basis=geometry['u'][:,geometry['active']]
    retained_field=basis.mT@field
    retained_motion=basis.mT@motion
    full_energy=motion.square().sum()
    retained_energy=retained_motion.square().sum()
    discarded_energy=(motion-basis@retained_motion).square().sum()
    gain=retained_field@retained_motion
    bound=retained_field.norm()*retained_motion.norm()
    return dict(retained_range_rank=int(geometry['active'].sum()),
        retained_range_fisher_quadratic=float(retained_energy/2),
        discarded_motion_fisher_quadratic=float(discarded_energy/2),
        retained_motion_energy_fraction=float(retained_energy/full_energy) if float(full_energy)>0 else None,
        retained_range_oracle_dual_squared=float(retained_field.square().sum()),
        retained_range_oracle_first_order_gain=float(gain),
        retained_range_cauchy_gain_bound=float(bound),
        efficiency_vs_retained_range_cauchy_bound=float(gain/bound) if float(bound)>0 else None,
        interpretation='Score and whitened tangent motion both projected into the identical retained singular range; discarded motion is reported separately')


def build_geometry(agent,obs,rtol=None):
    base={name:p.detach().clone() for name,p in agent.actor.named_parameters()}
    forward=torch.compile(lambda p:functional_call(agent.actor,p,(obs,)),fullgraph=True,
                          options={'triton.cudagraphs':False})
    derivative=torch.compile(jacrev(lambda p:functional_call(agent.actor,p,(obs,))),fullgraph=True,
                             options={'triton.cudagraphs':False})
    raw={name:value.detach().double() for name,value in derivative(base).items()}
    logits=forward(base).detach()
    _,_,_,_,fisher=beta_geometry(logits.double())
    eigen,vectors=torch.linalg.eigh(fisher)
    if bool((eigen<=0).any()):raise ValueError('Non-positive diagnostic Beta Fisher')
    root=(vectors*eigen.sqrt().unsqueeze(-2))@vectors.mT
    inverse_root=(vectors*eigen.rsqrt().unsqueeze(-2))@vectors.mT
    count,outputs=logits.shape
    blocks=[];widths=[]
    for name,weight in base.items():
        jac=tangent_jacobian(weight,raw[name],name).flatten(2)
        grouped=torch.stack(jac.chunk(2,1),-2)
        blocks.append((root@grouped).reshape(count*outputs,-1)/count**.5)
        widths.append(weight.numel())
    operator=torch.cat(blocks,-1)
    u,s,vh=torch.linalg.svd(operator,full_matrices=False)
    relative_threshold=max(operator.shape)*torch.finfo(operator.dtype).eps if rtol is None else rtol
    threshold=relative_threshold*s[0]
    active=s>threshold
    return dict(base=base,forward=forward,raw=raw,logits=logits,fisher=fisher,root=root,
                inverse_root=inverse_root,operator=operator,u=u,s=s,vh=vh,active=active,
                rank_threshold=threshold,relative_threshold=relative_threshold,widths=widths)


def whiten_field(geometry,score):
    return (geometry['inverse_root']@paired(score).unsqueeze(-1)).squeeze(-1).flatten()/score.shape[0]**.5


def whitened_replica_statistics(fields,actual_white=None):
    """Independent-replica signal/noise identities, without positivity clipping.

    Pair statistics share replicas when R>2; their entries are not independent
    observations for an additional pair-count-based confidence interval.
    """
    if fields.ndim!=2 or len(fields)<2:raise ValueError('Need >=2 whitened score replicas')
    pairs=[]
    for i in range(len(fields)):
        for j in range(i+1,len(fields)):
            cross=fields[i]@fields[j]
            noise=.5*(fields[i]-fields[j]).square().sum()
            denominator=fields[i].norm()*fields[j].norm()
            pairs.append(dict(replicate_i=i,replicate_j=j,signed_cross_energy=float(cross),
                half_squared_difference_noise_trace=float(noise),
                cosine=float(cross/denominator) if float(denominator)>0 else None))
    cross_mean=sum(p['signed_cross_energy'] for p in pairs)/len(pairs)
    noise_mean=sum(p['half_squared_difference_noise_trace'] for p in pairs)/len(pairs)
    mean_energy=float(fields.mean(0).square().sum())
    result=dict(replica_count=len(fields),pairs=pairs,
        mean_signed_cross_energy=cross_mean,mean_half_squared_difference_noise_trace=noise_mean,
        mean_replica_squared_norm=float(fields.square().sum(-1).mean()),mean_score_field_squared_norm=mean_energy,
        estimated_mean_field_noise_energy=noise_mean/len(fields),
        noise_corrected_mean_field_squared_norm=mean_energy-noise_mean/len(fields),
        mean_field_noise_to_signed_signal_ratio=noise_mean/len(fields)/cross_mean if cross_mean!=0 else None,
        signal_unresolved_flag=cross_mean<=0,
        interpretation='Cross-energy estimates true conditional score-field energy; negative estimates remain signed, not clipped.',
        limitations=['Independent identical-law replicas required; all share the same selected physical states.',
            'These identities estimate Monte Carlo noise, not between-seed or state-generalization uncertainty.',
            'Pair entries share replicas and cannot be treated as independent CI samples.'])
    if actual_white is not None:
        gains=fields@actual_white
        mean=gains.mean();sd=gains.std(unbiased=True);se=sd/len(gains)**.5
        critical=float(student_t.ppf(.975,len(gains)-1))
        result.update(actual_gain_replica_values=gains.tolist(),actual_gain_mean=float(mean),
            actual_gain_replica_std=float(sd),actual_gain_mc_standard_error=float(se),
            actual_gain_student_t95_interval=[float(mean-critical*se),float(mean+critical*se)],
            actual_gain_interval_scope='Conditional scalar MC interval, assumes independent approximately normal replica gains; unstable with only two replicas.')
    return result


def layer_motion(geometry,direction,project=False):
    changes=[]
    for name,weight in geometry['base'].items():
        delta=direction[name].double()
        if project:delta=tangent_jacobian(weight,delta,name)
        jacobian=geometry['raw'][name].flatten(2)
        changes.append(jacobian@delta.flatten())
    return torch.stack(changes)


def actual_precision_kl(old_logits,new_logits):
    a,b=(1+F.softplus(old_logits)).double().chunk(2,-1)
    c,d=(1+F.softplus(new_logits)).double().chunk(2,-1)
    return torch.distributions.kl_divergence(torch.distributions.Beta(a,b),
                                            torch.distributions.Beta(c,d)).sum(-1).mean()


def actual_case(geometry,direction,candidate,label):
    changes=layer_motion(geometry,direction)
    tangent=layer_motion(geometry,direction,project=True)
    total=changes.sum(0);tangent_total=tangent.sum(0)
    new_logits=geometry['forward'](candidate).detach()
    finite=new_logits.double()-geometry['logits'].double()
    gram=torch.stack([torch.stack([quadratic(a,b,geometry['fisher']) for b in changes]) for a in changes])
    return dict(label=label,changes=changes,total=total,tangent=tangent,tangent_total=tangent_total,finite=finite,
        report=dict(timing=label,component_labels=list(geometry['base']),layer_fisher_gram=gram.tolist(),
                    joint_fisher_quadratic=float(gram.sum()),diagonal_fisher_quadratic=float(gram.diagonal().sum()),
                    cross_fisher_quadratic=float(gram.sum()-gram.diagonal().sum()),
                    finite_exact_beta_kl=float(actual_precision_kl(geometry['logits'],new_logits)),
                    finite_fisher_quadratic=float(quadratic(finite,finite,geometry['fisher'])),
                    finite_logit_linearization_relative_error=float((finite-total).norm()/total.norm().clamp_min(1e-300)),
                    normal_retraction_component_logit_relative_norm=float((total-tangent_total).norm()/total.norm().clamp_min(1e-300)),
                    parameter_delta_norm=float(torch.stack([d.double().square().sum() for d in direction.values()]).sum().sqrt())))


def replica_fit_validation(geometry,replicas,cost):
    if float(cost)<=0:return []
    results=[]
    u=geometry['u'][:,geometry['active']];s=geometry['s'][geometry['active']]
    vh=geometry['vh'][geometry['active']]
    fields=[whiten_field(geometry,score) for score in replicas]
    for train,field in enumerate(fields):
        coordinate=u.mT@field
        projected=u@coordinate
        if float(projected.norm())==0:continue
        scale=torch.sqrt(2*cost)/projected.norm()
        response=projected*scale
        delta=vh.mT@(coordinate/s)*scale
        proposed={};offset=0
        for (name,weight),width in zip(geometry['base'].items(),geometry['widths']):
            value=weight+delta[offset:offset+width].reshape(weight.shape).to(weight.dtype);offset+=width
            if weight.ndim==2:value=F.normalize(value,dim=0 if '.down.' in name else 1)
            proposed[name]=value
        new_logits=geometry['forward'](proposed).detach()
        finite=new_logits.double()-geometry['logits'].double()
        finite_white=(geometry['root']@paired(finite).unsqueeze(-1)).squeeze(-1).flatten()/finite.shape[0]**.5
        for test,held in enumerate(fields):
            if train==test:continue
            bound=torch.sqrt(2*cost*held.square().sum())
            results.append(dict(fit_replicate=train,evaluation_replicate=test,target_fisher_quadratic=float(cost),
                fit_replica_first_order_gain=float(field@response),held_replica_first_order_gain=float(held@response),
                held_replica_unconstrained_cauchy_bound=float(bound),
                held_replica_efficiency=float((held@response)/bound) if float(bound)>0 else None,
                fit_model_reachable_dual_fraction=float(projected.square().sum()/field.square().sum()),
                required_parameter_delta_norm=float(delta.norm()),
                finite_retracted_exact_beta_kl=float(actual_precision_kl(geometry['logits'],new_logits)),
                finite_retracted_fisher_quadratic=float(finite_white.square().sum()/2),
                finite_retraction_relative_error=float((finite_white-response).norm()/response.norm().clamp_min(1e-300)),
                finite_held_replica_gain=float(held@finite_white)))
    return results


def analyze(row,path,args):
    digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if row.get('snapshot_sha256')!=digest:raise ValueError('Dense oracle/snapshot hash mismatch')
    if 'selected_states' not in row:raise ValueError('Require matched dense oracle states')
    snapshot=torch.load(path,map_location='cuda',weights_only=False)
    validate_dense_state_mapping(row,snapshot,digest)
    trace_path=Path(path).with_name(f"diagnostic_updates_{snapshot['global_step']}.pt")
    trace=torch.load(trace_path,map_location='cuda',weights_only=False)
    if trace['global_step']!=snapshot['global_step']:raise ValueError('Wrong checkpoint update trace')
    state=snapshot['model'];dim=state['actor.first.0.gate.weight'].shape[-1]
    descriptor=SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf,np.inf,(dim,),dtype=np.float32),
        single_action_space=gym.spaces.Box(state['action_low'].cpu().numpy(),state['action_high'].cpu().numpy()))
    module=importlib.import_module('cleanrl.ppo_continuous_action_sepclip_rollout_clock_anvil_v7')
    agent=module.Agent(descriptor).cuda().eval();agent.load_state_dict(state)
    _,_,trace_validation=reconstruct_banks(agent,snapshot,trace)
    obs=torch.tensor([s['normalized_obs'] for s in row['selected_states']],device='cuda',dtype=torch.float32)
    geometry=build_geometry(agent,obs,args.svd_rtol)
    policy_error=validate_saved_policy_logits(row['initial_policy']['logits'],geometry['logits'])
    base=geometry['base'];first=trace['trace'][0]
    first_delta={name:first['retracted_delta']['actor.'+name] for name in base}
    first_candidate={name:weight+first_delta[name] for name,weight in base.items()}
    final_candidate={name:trace['model_after_update']['actor.'+name] for name in base}
    final_delta={name:value-base[name] for name,value in final_candidate.items()}
    cases=[actual_case(geometry,first_delta,first_candidate,'first_actual_retracted_update_at_snapshot'),
           actual_case(geometry,final_delta,final_candidate,'all_epochs_endpoint_delta_linearized_at_snapshot')]
    observed=torch.tensor(row['initial_policy']['observed_logit_score'],device='cuda',dtype=torch.float64)
    advantages=torch.tensor([s['stored_advantage'] for s in row['selected_states']],device='cuda',dtype=torch.float64)
    selected_gae=observed*advantages[:,None]
    hresults=[]
    for horizon in row['oracle']['horizons']:
        if args.horizon and horizon['horizon'] not in args.horizon:continue
        pg_field=horizon['policy_logit_pg']['finite_normalized']
        score=torch.tensor(pg_field['mean'],device='cuda',dtype=torch.float64)
        replicas=torch.tensor(pg_field['replicate_values'],device='cuda',dtype=torch.float64)
        field=whiten_field(geometry,score)
        u=geometry['u'][:,geometry['active']]
        reachable=(u@(u.mT@field)).square().sum()
        result=dict(horizon=horizon['horizon'],kind='finite_normalized_conditional_first_action_pg',cases=[])
        result['fixed_actual_motion_objective_pairings']=fixed_motion_objective_pairings(
            horizon,{case['label']:dict(actual_endpoint_jvp=case['total'],
                tangent_only_jvp=case['tangent_total'],finite_actual_logit_change=case['finite'])
                for case in cases},row['oracle']['continuation_replicates'],row['oracle'].get('first_action_only'))
        for case in cases:
            layers=[]
            for name,change,tangent in zip(base,case['changes'],case['tangent']):
                layers.append(dict(parameter=name,
                    actual_endpoint_jvp=motion_metrics(geometry['logits'],change,score,geometry['fisher']),
                    tangent_only_jvp=motion_metrics(geometry['logits'],tangent,score,geometry['fisher'])))
            info=dict(**case['report'],layers=layers,
                actual_endpoint_jvp=motion_metrics(geometry['logits'],case['total'],score,geometry['fisher']),
                tangent_only_jvp=motion_metrics(geometry['logits'],case['tangent_total'],score,geometry['fisher']),
                retained_range_tangent_motion=retained_range_motion_metrics(geometry,case['tangent_total'],score),
                finite_actual_logit_change=motion_metrics(geometry['logits'],case['finite'],score,geometry['fisher']),
                selected_observed_gae_first_order_gain=float(score_pairing(selected_gae,case['total'])),
                independent_replica_actual_first_order_gains=[float(score_pairing(replica,case['total'])) for replica in replicas])
            result['cases'].append(info)
        reference_cost=quadratic(cases[0]['tangent_total'],cases[0]['tangent_total'],geometry['fisher'])
        actual_white=(geometry['root']@paired(cases[0]['tangent_total']).unsqueeze(-1)).squeeze(-1).flatten()/score.shape[0]**.5
        result['threshold_sensitivity']=threshold_sensitivity(geometry['u'],geometry['s'],field,
            float(reference_cost),geometry['relative_threshold'],actual_white)
        result['cross_replica_fitted_direction_validation']=replica_fit_validation(geometry,replicas,reference_cost)
        replica_fields=torch.stack([whiten_field(geometry,replica) for replica in replicas])
        actual_first_white=(geometry['root']@paired(cases[0]['total']).unsqueeze(-1)).squeeze(-1).flatten()/score.shape[0]**.5
        result['cross_replica_signal_noise']=whitened_replica_statistics(replica_fields,actual_first_white)
        result['oracle_model_reachable_dual_fraction']=float(reachable/field.square().sum().clamp_min(1e-300))
        hresults.append(result)
    if not hresults:raise ValueError('No requested dense oracle horizon present')
    fraction=geometry['s'].square()/geometry['s'].square().sum()
    return dict(snapshot=str(Path(path).resolve()),snapshot_sha256=digest,global_step=snapshot['global_step'],
        trace=str(trace_path.resolve()),trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        trace_validation=trace_validation,
        state_count=len(row['selected_states']),oracle_actor_max_abs_error=policy_error,
        selected_state_coordinates=[dict(t=s['t'],row=s['row'],episode_age=s['episode_age']) for s in row['selected_states']],
        actual_update_count=len(trace['trace']),tangent_functional_singular_values=geometry['s'].tolist(),
        tangent_functional_rank=int(geometry['active'].sum()),rank_absolute_threshold=float(geometry['rank_threshold']),
        rank_interpretation='Configured exploratory arithmetic fit; FP32 derivative accuracy limits trustworthy tiny modes',
        effective_functional_rank=float(torch.exp(-(fraction*fraction.clamp_min(1e-300).log()).sum())),
        horizons=hresults,precision=dict(actor='FP32',jacobian='FP32 derivative; FP64 tangent projection and SVD',exact_KL='actual FP32 shape arithmetic, FP64 KL algebra',
            fisher='FP64 diagnostic reevaluation from actual FP32 actor logits'),
        limitations=['Same saved training states; independent conditional first-action PG, continuation policy held fixed.',
            'Later-action policy-score terms and optimal-control oracle are absent.',
            'First weights reconstructed from saved FP32 parameter deltas; full endpoint uses exact saved weights.',
            'Full ten-epoch endpoint linearization is reported separately; finite discrepancy can invalidate it.',
            'Retained-range efficiency projects both score and tangent motion; discarded motion and full unconstrained metrics are separate. Actual endpoint delta also contains second-order normal components.',
            'Cross-noise replica validation is on the same states, not held-out physical states.',
            'Cauchy bounds apply to estimated oracle score fields, not a confidence bound on the unknown true PG.',
            'Finite-state interpolation/reachability is not a generalization or global architecture-capacity guarantee.',
            'Selected observed-action GAE pairing has one sampled action per state and is not conditional mean PG.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle',action='append',required=True)
    parser.add_argument('--snapshot',action='append',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--horizon',action='append',type=int)
    parser.add_argument('--svd-rtol',type=float)
    args=parser.parse_args()
    configure_runtime(matmul_precision='highest',allow_tf32=False)
    # Multiple saved policies and Jacobian closures intentionally compile here.
    # Retain fullgraph behavior rather than exhausting the default eight guards.
    torch._dynamo.config.recompile_limit = 64
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; submit via mlq')
    if args.svd_rtol is not None and args.svd_rtol<=0:parser.error('SVD tolerance must be positive')
    snapshots={str(Path(path).resolve()):path for path in args.snapshot}
    output=dict(protocol=vars(args),results=[],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    root=Path(__file__).resolve().parents[1]
    output['imported_source_sha256']={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        root/'scripts/audit_sepclip_oracle_geometry.py',root/'scripts/audit_sepclip_closed_loop_update.py',
        root/'scripts/audit_sepclip_dense_pg.py',root/'scripts/audit_sepclip_physics.py',
        root/'cleanrl/ppo_continuous_action_sepclip_rollout_clock_anvil_v7.py',
        root/'cleanrl/ppo_continuous_action_sepclip_transition_microscopy_v12.py',root/'cleanrl/shared/runtime.py')}
    for oracle in args.oracle:
        oracle_bytes=Path(oracle).read_bytes()
        oracle_digest=hashlib.sha256(oracle_bytes).hexdigest()
        for row in json.loads(oracle_bytes)['results']:
            key=str(Path(row['snapshot']).resolve())
            if key not in snapshots:continue
            result=analyze(row,snapshots[key],args);result['oracle']=str(Path(oracle).resolve());result['oracle_sha256']=oracle_digest
            output['results'].append(result)
            target=Path(args.out);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(output,indent=2)+'\n')
    if not output['results']:raise ValueError('No matching dense oracle/snapshot pair')


if __name__=='__main__':main()
