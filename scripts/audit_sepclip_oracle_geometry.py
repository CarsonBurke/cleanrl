"""CUDA-only, ridge-free local oracle geometry; submit through mlq.

Fits an independent conditional initial-action score field at explicitly saved
states. Finite-state interpolation is not a generalization/capacity certificate.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cleanrl.shared.runtime import configure_runtime


def validate_dense_state_mapping(row,snapshot,digest):
    """Reject a relabeled oracle; snapshot identity alone does not bind rows."""
    if row.get('snapshot_sha256')!=digest:raise ValueError('Snapshot source hash does not match oracle')
    def array(x):return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
    ages=array(snapshot['transitions']['episode_age'])
    steps,envs=ages.shape
    observations=array(snapshot['batch']['obs']).reshape(steps,envs,-1)
    natives=array(snapshot['batch']['native']).reshape(steps,envs,-1)
    advantages=array(snapshot['batch']['advantages']).reshape(steps,envs)
    seen=set()
    for selected in row['selected_states']:
        t,r=selected['t'],selected['row']
        if not isinstance(t,int) or not isinstance(r,int) or not (0<=t<steps and 0<=r<envs) or (t,r) in seen:
            raise ValueError('Invalid or duplicate dense selected state coordinate')
        seen.add((t,r))
        for name,expected in (('normalized_obs',observations[t,r]),('native_action',natives[t,r])):
            supplied=np.asarray(selected[name],dtype=np.float32)
            if supplied.shape!=expected.shape or not np.array_equal(supplied,expected):
                raise ValueError(f'Dense selected state {name} mismatch at {(t,r)}')
        if selected['stored_advantage']!=float(advantages[t,r]):
            raise ValueError(f'Dense selected state stored_advantage mismatch at {(t,r)}')
        if selected['episode_age']!=int(ages[t,r]):
            raise ValueError(f'Dense selected state episode_age mismatch at {(t,r)}')
    if not seen:raise ValueError('No dense selected states')


def validate_saved_policy_logits(saved,loaded,tolerance=1e-4):
    recorded=torch.as_tensor(saved,device=loaded.device,dtype=loaded.dtype)
    if recorded.shape!=loaded.shape or not bool(torch.isfinite(recorded).all()) or not bool(torch.isfinite(loaded).all()):
        raise ValueError('Invalid saved initial policy logits')
    error=float((recorded-loaded).abs().max())
    if error>tolerance:raise ValueError(f'Oracle initial actor logits mismatch {error}')
    return error


def beta_geometry(logits):
    a, b = (1 + F.softplus(logits)).chunk(2, -1)
    sa, sb = logits.sigmoid().chunk(2, -1)
    k = a+b
    common = torch.polygamma(1, k)
    fisher = torch.stack((torch.stack(((torch.polygamma(1,a)-common)*sa.square(), -common*sa*sb),-1),
                          torch.stack((-common*sa*sb, (torch.polygamma(1,b)-common)*sb.square()),-1)), -2)
    return a, b, sa, sb, fisher


def coordinate_report(logits, pg, target_kl):
    a,b,sa,sb,fisher = beta_geometry(logits.double())
    k,mu = a+b,a/(a+b)
    paired = torch.stack(pg.double().chunk(2,-1),-1)
    # Columns are d(logits)/d(mu,kappa), not gradients in those coordinates.
    inverse = torch.stack((torch.stack((k/sa,mu/sa),-1),
                           torch.stack((-k/sb,(1-mu)/sb),-1)), -2)
    metric = inverse.mT @ fisher @ inverse
    gradient = (inverse.mT @ paired.unsqueeze(-1)).squeeze(-1)
    slope = gradient / metric.diagonal(dim1=-2,dim2=-1).sqrt()
    delta = gradient.sign() * torch.sqrt(2*target_kl/metric.diagonal(dim1=-2,dim2=-1))
    mean_slack = torch.stack((mu-1/k,1-1/k-mu),-1)
    # One coordinate changes, the other is held fixed. Finite shape feasibility
    # is separate from a Fisher radius; no clamp changes the proposed direction.
    feasible_mu = (mu+delta[...,0]>1/k)&(mu+delta[...,0]<1-1/k)
    new_k = k+delta[...,1]
    feasible_k = (mu*new_k>1)&((1-mu)*new_k>1)
    return dict(native_mean=mu.tolist(),concentration=k.tolist(),
                coordinate_pg=gradient.tolist(),coordinate_fisher=metric.tolist(),
                unit_fisher_score_slope=slope.tolist(),
                fisher_cross_correlation=(metric[...,0,1]/torch.sqrt(metric[...,0,0]*metric[...,1,1])).tolist(),
                isolated_coordinate_delta_at_target_kl=delta.tolist(),
                mean_feasible_slack=mean_slack.tolist(),
                isolated_mean_step_feasible=feasible_mu.tolist(),isolated_concentration_step_feasible=feasible_k.tolist(),
                coordinate_order=['native_mean','concentration'],
                target_kl_per_state_action_dimension=target_kl)


def tangent_jacobian(weight, jacobian, name):
    if weight.ndim != 2:
        return jacobian
    dim = -2 if '.down.' in name else -1
    # Use the exact normalized constraint in the Jacobian's precision rather
    # than treating FP32 row-norm rounding as extra representational freedoms.
    weight=F.normalize(weight.to(jacobian.dtype),dim=dim)
    return jacobian-weight*(jacobian*weight).sum(dim,keepdim=True)


def threshold_sensitivity(u,s,field,target_kl,default_rtol,actual_white=None):
    """Reuse one SVD; distinguish arithmetic rank from FP32 derivative accuracy.

    Norms are minimum tangent-parameter norms for the projected oracle direction
    at the declared Fisher cost. This is not a regularized training proposal.
    """
    coordinates=u.mT@field
    energy=field.square().sum()
    actual_coordinates=u.mT@actual_white if actual_white is not None else None
    entries=[]
    for label,relative in [('configured_arithmetic_fit',default_rtol),
                           ('fp32_sensitivity_1e-7',1e-7),('fp32_sensitivity_1e-6',1e-6),
                           ('fp32_sensitivity_1e-5',1e-5),('fp32_sensitivity_1e-4',1e-4)]:
        active=s>relative*s[0]
        projection_energy=coordinates[active].square().sum()
        fraction=projection_energy/energy.clamp_min(1e-300)
        coefficient_norm=(coordinates[active]/s[active]).norm()
        budget_scale=(2*target_kl)**.5
        entry=dict(label=label,relative_singular_cutoff=relative,rank=int(active.sum()),
            oracle_reachable_dual_energy_fraction=float(fraction),
            constrained_maximum_efficiency_vs_unconstrained=float(fraction.sqrt()),
            required_tangent_parameter_norm_at_full_budget=float(coefficient_norm*budget_scale/projection_energy.sqrt())
                if float(projection_energy)>0 else None,
            unconstrained_target_projection_parameter_norm=float(coefficient_norm*budget_scale/energy.sqrt())
                if float(energy)>0 else None)
        if actual_coordinates is not None:
            # An actual step can contain discarded modes. Project it into the
            # same retained range before comparing against that range's bound.
            gain=(coordinates[active]*actual_coordinates[active]).sum()
            actual_energy=actual_coordinates[active].square().sum()
            bound=(projection_energy*actual_energy).sqrt()
            entry.update(actual_retained_range_fisher_quadratic=float(actual_energy/2),
                actual_retained_range_motion_energy_fraction=float(actual_energy/actual_white.square().sum().clamp_min(1e-300)),
                actual_retained_range_oracle_gain=float(gain),
                actual_retained_range_model_bound_efficiency=float(gain/bound) if float(bound)>0 else None)
        entries.append(entry)
    return entries


def fitted_response(agent, obs, pg, target_kl, rtol=None, cache=None):
    actor=agent.actor
    base={name:p.detach() for name,p in actor.named_parameters()}
    forward=torch.compile(lambda p:functional_call(actor,p,(obs,)),fullgraph=True,
                          options={'triton.cudagraphs':False})
    cache={} if cache is None else cache
    if 'basis' not in cache:
        differentiate=torch.compile(jacrev(lambda p:functional_call(actor,p,(obs,))),fullgraph=True,
                                    options={'triton.cudagraphs':False})
        raw=differentiate(base)
        logits=forward(base).detach().double()
        _,_,_,_,fisher=beta_geometry(logits)
        eigen,vectors=torch.linalg.eigh(fisher)
        if bool((eigen<=0).any()):
            raise ValueError('Beta Fisher is not positive definite at oracle states')
        root=(vectors*eigen.sqrt().unsqueeze(-2))@vectors.mT
        inverse_root=(vectors*eigen.rsqrt().unsqueeze(-2))@vectors.mT
        count,outputs=logits.shape
        matrices=[]; names=[]; widths=[]
        for name,weight in base.items():
            jac=tangent_jacobian(weight,raw[name].double(),name).detach().flatten(2)
            paired=torch.stack(jac.chunk(2,1),-2)
            matrices.append((root@paired).reshape(count*outputs,-1)/count**.5)
            names.append(name);widths.append(weight.numel())
        operator=torch.cat(matrices,-1)
        u,s,vh=torch.linalg.svd(operator,full_matrices=False)
        cache['basis']=(logits,root,inverse_root,matrices,names,widths,operator,u,s,vh)
    logits,root,inverse_root,matrices,names,widths,operator,u,s,vh=cache['basis']
    count,outputs=logits.shape
    paired_pg=torch.stack(pg.double().chunk(2,-1),-1)
    target=(inverse_root@paired_pg.unsqueeze(-1)).squeeze(-1).flatten()/count**.5
    dual_norm=target.norm()
    if float(dual_norm)==0:
        raise ValueError('Oracle score field is zero; no fitted direction exists')
    target=target/dual_norm*(2*target_kl)**.5
    relative_threshold=max(operator.shape)*torch.finfo(operator.dtype).eps if rtol is None else rtol
    threshold=relative_threshold*s[0]
    active=s>threshold
    coordinates=u.mT@target
    coefficient=torch.where(active,coordinates/s.clamp_min(torch.finfo(s.dtype).tiny),0)
    delta=vh.mT@coefficient
    achieved=operator@delta
    residual=target-achieved
    energy=s.square(); fraction=energy/energy.sum()
    effective=float(torch.exp(-(fraction*fraction.clamp_min(1e-300).log()).sum()))
    directions={}; layer_changes=[];layers=[];offset=0
    for name,weight,matrix,width in zip(names,base.values(),matrices,widths):
        vector=delta[offset:offset+width];offset+=width
        directions[name]=vector.reshape(weight.shape).to(weight.dtype)
        change=matrix@vector;layer_changes.append(change)
        layer=dict(parameter=name,parameter_delta_norm=float(vector.norm()),
                   oracle_fisher_pairing=float(change@target),
                   isolated_fisher_quadratic=float(change.square().sum()/2))
        if weight.ndim==2:
            dim=0 if '.down.' in name else 1
            old=F.normalize(weight.double(),dim=dim)
            movement=vector.reshape(weight.shape)
            new=F.normalize(weight.double()+movement,dim=dim)
            cosine=(old*new).sum(dim).clamp(-1,1)
            layer.update(max_constraint_vector_delta_norm=float(movement.norm(dim=dim).max()),
                         max_constraint_angle_radians=float(torch.acos(cosine).max()))
        layers.append(layer)
    changes=torch.stack(layer_changes)
    gram=changes@changes.mT/2
    proposed={name:(weight+directions[name]) for name,weight in base.items()}
    for name,weight in proposed.items():
        if weight.ndim==2:
            proposed[name]=F.normalize(weight,dim=0 if '.down.' in name else 1)
    new_logits=forward(proposed).detach()
    finite=new_logits.double()-logits
    finite_paired=torch.stack(finite.chunk(2,-1),-1)
    finite_white=(root@finite_paired.unsqueeze(-1)).squeeze(-1).flatten()/count**.5
    # Policy shapes are formed in the learner's actual FP32 arithmetic. Only
    # the KL algebra is upgraded to FP64; re-evaluating softplus in FP64 would
    # measure a nearby, different policy when the motion is extremely small.
    aa,bb=(1+F.softplus(logits.to(new_logits.dtype))).double().chunk(2,-1)
    ca,cb=(1+F.softplus(new_logits)).double().chunk(2,-1)
    exact=torch.distributions.kl_divergence(torch.distributions.Beta(aa,bb),
                                           torch.distributions.Beta(ca,cb)).sum(-1).mean()
    return dict(state_count=count,parameter_count=operator.shape[-1],functional_coordinate_count=operator.shape[0],
                tangent_functional_singular_values=s.tolist(),rank=int(active.sum()),rank_absolute_threshold=float(threshold),
                jacobian_precision='FP32 actor and derivative; cast to FP64 before tangent projection and SVD',
                rank_interpretation='Configured arithmetic fit; singular modes below FP32 derivative accuracy may be unreliable',
                threshold_sensitivity=threshold_sensitivity(u,s,target,target_kl,relative_threshold),
                retained_functional_condition_number=float(s[0]/s[active][-1]) if bool(active.any()) else None,
                effective_rank=effective,top1_spectral_energy_fraction=float(fraction[0]),
                oracle_dual_norm=float(dual_norm),target_fisher_quadratic=target_kl,
                achieved_fisher_quadratic=float(achieved.square().sum()/2),
                projected_oracle_energy_fraction=float(achieved.square().sum()/target.square().sum()),
                oracle_whitened_pg_cosine=float((achieved@target)/(achieved.norm()*target.norm()+1e-300)),
                oracle_whitened_residual_relative_norm=float(residual.norm()/target.norm()),
                required_parameter_delta_norm=float(delta.norm()),layers=layers,
                layer_fisher_gram=gram.tolist(),joint_fisher_quadratic=float(gram.sum()),
                diagonal_fisher_quadratic=float(gram.diagonal().sum()),
                cross_fisher_quadratic=float(gram.sum()-gram.diagonal().sum()),
                finite_retracted_exact_beta_kl=float(exact),
                exact_kl_precision='FP32 policy softplus+1 shapes; FP64 KL algebra',
                fisher_precision='FP64 numerical softplus/Fisher from exact FP32 actor logits',
                finite_retracted_fisher_quadratic=float(finite_white.square().sum()/2),
                finite_retraction_linearization_relative_error=float((finite_white-achieved).norm()/achieved.norm().clamp_min(1e-300)),
                limitations=['Minimum-norm interpolation on saved states; no held-out/generalization certificate.',
                             'Rank uses a declared numerical SVD threshold, no ridge regularizer.',
                             'A large parameter step invalidates the linear fit; finite retraction checks expose that.',
                             'Conditional initial-action oracle holds continuation policy fixed; not a global optimum.'])


def oracle_cases(row):
    if 'selected_states' in row:
        states=row['selected_states']
        obs=np.asarray([s['normalized_obs'] for s in states],dtype=np.float32)
        for h in row['oracle']['horizons']:
            fields=h['policy_logit_pg']
            for label in ('finite_normalized','bootstrap_normalized','finite_raw',
                          'finite_normalized_mean','bootstrap_normalized_mean','finite_raw_mean'):
                if label in fields:
                    value=fields[label]['mean'] if isinstance(fields[label],dict) else fields[label]
                    yield obs,dict(horizon=h['horizon'],kind=label,state_distribution_matched=True,
                        _saved_initial_logits=row['initial_policy']['logits']),value
    else:
        for replicate_id,replicate in enumerate(row['fork_replicates']):
            states=replicate['states']
            if not all('normalized_observation' in state or 'normalized_obs' in state for state in states):
                raise ValueError('Historical oracle lacks exact normalized states; cannot substitute rollout states')
            obs=np.asarray([s.get('normalized_observation',s.get('normalized_obs')) for s in states],dtype=np.float32)
            for h in replicate['horizons']:
                for label in ('logit_score_pg','normalized_reward_logit_score_pg','normalized_bootstrap_logit_score_pg'):
                    yield obs,dict(horizon=h['horizon'],kind=label,replicate=replicate_id,state_distribution_matched=False,
                        _saved_initial_logits=[s['logits'] for s in states]),h[label]


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--oracle',action='append',required=True)
    cli.add_argument('--snapshot',action='append',required=True)
    cli.add_argument('--out',required=True)
    cli.add_argument('--target-kl',type=float,default=.001)
    cli.add_argument('--horizon',action='append',type=int,
                     help='Only fit these oracle horizons; repeatable, default all')
    cli.add_argument('--svd-rtol',type=float)
    cli.add_argument('--coordinates-only',action='store_true')
    args=cli.parse_args()
    if args.target_kl<=0 or (args.svd_rtol is not None and args.svd_rtol<=0):
        cli.error('KL and optional SVD tolerance must be positive')
    configure_runtime(matmul_precision='highest',allow_tf32=False)
    # Multiple saved policies and Jacobian closures intentionally compile here.
    # Retain fullgraph behavior rather than exhausting the default eight guards.
    torch._dynamo.config.recompile_limit = 64
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    snapshots={str(Path(p).resolve()):p for p in args.snapshot}
    result=[]
    for oracle in args.oracle:
        oracle_bytes=Path(oracle).read_bytes()
        oracle_digest=hashlib.sha256(oracle_bytes).hexdigest()
        document=json.loads(oracle_bytes)
        for row in document['results']:
            key=str(Path(row['snapshot']).resolve())
            if key not in snapshots:continue
            path=snapshots[key];digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if row.get('snapshot_sha256')!=digest:
                raise ValueError('Snapshot source hash does not match oracle')
            snap=torch.load(path,map_location='cuda',weights_only=False)
            if 'selected_states' in row:validate_dense_state_mapping(row,snap,digest)
            module=importlib.import_module('cleanrl.ppo_continuous_action_sepclip_rollout_clock_anvil_v7')
            state=snap['model'];dim=state['actor.first.0.gate.weight'].shape[-1]
            descriptor=SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf,np.inf,(dim,),dtype=np.float32),
                                       single_action_space=gym.spaces.Box(state['action_low'].cpu().numpy(),state['action_high'].cpu().numpy()))
            agent=module.Agent(descriptor).cuda().eval();agent.load_state_dict(state)
            actor=torch.compile(agent.actor,fullgraph=True,options={'triton.cudagraphs':False})
            geometry_cache={}
            for obs,metadata,pg in oracle_cases(row):
                if args.horizon and metadata['horizon'] not in args.horizon:continue
                x=torch.as_tensor(obs,device='cuda');gradient=torch.as_tensor(pg,device='cuda',dtype=torch.float64)
                with torch.no_grad():logits=actor(x).detach()
                policy_error=validate_saved_policy_logits(metadata.pop('_saved_initial_logits'),logits)
                item=dict(snapshot=key,snapshot_sha256=digest,oracle=str(Path(oracle).resolve()),oracle_sha256=oracle_digest,
                          oracle_actor_max_abs_error=policy_error,**metadata,
                          coordinates=coordinate_report(logits,gradient,args.target_kl))
                if not args.coordinates_only:
                    state_key=hashlib.sha256(obs.tobytes()).hexdigest()
                    item['functional_fit']=fitted_response(agent,x,gradient,args.target_kl,args.svd_rtol,
                        geometry_cache.setdefault(state_key,{}))
                result.append(item)
                output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True)
                output.write_text(json.dumps(dict(protocol=vars(args),results=result,
                    imported_source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (Path(module.__file__),Path(__file__).parents[1]/'cleanrl/shared/runtime.py')},
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    if not result:raise ValueError('No matching saved oracle states/snapshots')


if __name__=='__main__':main()
