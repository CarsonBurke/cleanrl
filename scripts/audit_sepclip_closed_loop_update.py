"""Actual v12 PPO weight updates in paired closed-loop physics; mlq/CUDA only.

Every selected old, first-update and endpoint policy controls the continuation.
Optional hybrid arms identify finite feedback interaction; directional PG sums
old-trajectory score-return terms over all times, never scales a t=0 oracle.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.special import betainc,digamma,gammaln
from scipy.stats import t as student_t
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cleanrl import ppo_continuous_action_sepclip_transition_microscopy_v12 as trainer
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.runtime import configure_runtime

_spec=importlib.util.spec_from_file_location('closed_loop_dense_helpers',ROOT/'scripts/audit_sepclip_dense_pg.py')
dense=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dense)

POLICIES=('old','first_update','endpoint')
BASIC_ARMS=('old_all','first_all','endpoint_all')
HYBRID_ARMS=BASIC_ARMS+('first_t0_only','first_after_t0','endpoint_t0_only','endpoint_after_t0')


def hash_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def arm_policy_indices(step,hybrid):
    if not hybrid:return np.array([0,1,2],dtype=np.int64)
    return np.array([0,1,2,1 if step==0 else 0,0 if step==0 else 1,
                     2 if step==0 else 0,0 if step==0 else 2],dtype=np.int64)


def reconstruct_banks(agent,snapshot,trace):
    if trace['global_step']!=snapshot['global_step']:
        raise ValueError('Snapshot and update trace global_step mismatch')
    entries=trace['trace']
    if not entries or entries[0]['update']!=0 or entries[0]['epoch']!=0:
        raise ValueError('Need complete update trace beginning at first actual update')
    expected=set(dict(agent.named_parameters()))
    state_keys=set(agent.state_dict())
    if set(snapshot['model'])!=state_keys or set(trace['model_after_update'])!=state_keys:
        raise ValueError('Strict model state key mismatch')
    args=snapshot['args']
    epochs=int(args.get('update_epochs',len(entries)))
    batch=int(args.get('batch_size',1));mini=int(args.get('minibatch_size',batch))
    expected_count=epochs*((batch+mini-1)//mini)
    if len(entries)!=expected_count:
        raise ValueError(f'Full endpoint trace needs {expected_count} updates, got {len(entries)}')
    with torch.no_grad():
        old={k:v.cuda().clone() for k,v in snapshot['model'].items()}
        first={k:v.clone() for k,v in old.items()}
        replay={k:v.clone() for k,v in old.items()}
        for index,entry in enumerate(entries):
            if entry['update']!=index or set(entry['retracted_delta'])!=expected:
                raise ValueError('Trace update sequence or parameter keys mismatch')
            for name,delta in entry['retracted_delta'].items():
                if delta.shape!=old[name].shape:
                    raise ValueError(f'Trace delta shape mismatch for {name}')
                change=delta.cuda()
                replay[name].add_(change)
                if index==0:first[name].add_(change)
        endpoint={k:v.cuda().clone() for k,v in trace['model_after_update'].items()}
        error=max(float((replay[name]-endpoint[name]).abs().max()) for name in expected)
        if error>2e-5:
            raise ValueError(f'Cumulative trace reconstruction error {error}')
        names=tuple(dict(agent.actor.named_parameters()))
        banks={name:torch.stack([state['actor.'+name] for state in (old,first,endpoint)]) for name in names}
        deltas=({name:first['actor.'+name]-old['actor.'+name] for name in names},
                {name:endpoint['actor.'+name]-old['actor.'+name] for name in names})
        norm_errors={}
        for label,state in zip(POLICIES,(old,first,endpoint)):
            errors=[]
            for stage in ('first','second'):
                for layer,axis in (('gate',1),('up',1),('down',0)):
                    w=state[f'actor.{stage}.0.{layer}.weight']
                    errors.append(float((torch.linalg.vector_norm(w,dim=axis)-1).abs().max()))
            errors.append(float((torch.linalg.vector_norm(state['actor.head.0.weight'],dim=1)-1).abs().max()))
            norm_errors[label]=max(errors)
    return banks,deltas,dict(update_count=len(entries),epochs=epochs,
        cumulative_delta_vs_exact_endpoint_max_abs_error=error,unit_axis_norm_max_abs_errors=norm_errors,
        first_policy_source='saved old FP32 weights plus first actual retracted FP32 delta; subtraction/addition roundoff remains',
        endpoint_policy_source='exact model_after_update state_dict, not cumulative deltas')


def compiled_policy(agent,banks,count,arms):
    def actor_one(parameters,x):
        return torch.func.functional_call(agent.actor,parameters,(x,))
    def policy(x,indices):
        inputs=x.reshape(count,arms,-1).transpose(0,1)
        parameters={name:value.index_select(0,indices) for name,value in banks.items()}
        logits=torch.func.vmap(actor_one)(parameters,inputs).transpose(0,1).reshape(count*arms,-1)
        alpha,beta=(1+F.softplus(logits)).chunk(2,-1)
        return logits,alpha,beta
    return torch.compile(policy,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})


def compiled_directional(agent,banks,deltas):
    old={name:value[0] for name,value in banks.items()}
    def actor(parameters,x):
        return torch.func.functional_call(agent.actor,parameters,(x,))
    def directional(x):
        z,u_first=torch.func.jvp(lambda p:actor(p,x),(old,),(deltas[0],))
        _,u_end=torch.func.jvp(lambda p:actor(p,x),(old,),(deltas[1],))
        return z,u_first,u_end,agent.get_value(x).flatten()
    return torch.compile(directional,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})


def compiled_old_value(agent):
    return torch.compile(lambda x:agent.get_value(x).flatten(),fullgraph=True,dynamic=False,
        options={'triton.cudagraphs':False})


def restore(envs,states,arms):
    errors=[]
    for i,state in enumerate(s for s in states for _ in range(arms)):
        base=envs.envs[i].unwrapped
        dense.physics.set_integration_state(base,state['integration_state'])
        mujoco.mj_forward(base.model,base.data)
        dense.physics.set_integration_state(base,state['integration_state'])
        envs._episode_lengths[i]=state['episode_age']
        envs._limits[i]._elapsed_steps=state['episode_age']
        envs._episode_returns[i]=0
        envs.observations[i]=base._get_obs()
        dense.require_finite(f'Closed-loop restored state {i}', raw_obs=envs.observations[i],
                             integration_state=dense.physics.integration_state(base))
        errors.append(max(float(np.max(np.abs(envs.observations[i]-state['raw_obs']))),
            float(np.max(np.abs(dense.physics.integration_state(base)-state['integration_state'])))))
    if max(errors)>1e-10:raise ValueError(f'Full integration-state restore mismatch {max(errors)}')
    return max(errors)


def beta_kl(a,b,c,d):
    # FP64 algebra on the actual FP32 shape values, preserving tiny-step KL.
    a,b,c,d=(np.asarray(x,dtype=np.float64) for x in (a,b,c,d))
    k,l=a+b,c+d
    return (gammaln(c)+gammaln(d)-gammaln(l)-gammaln(a)-gammaln(b)+gammaln(k)
        +(a-c)*digamma(a)+(b-d)*digamma(b)+(l-k)*digamma(k)).sum(-1)


def occupancy_kl(policy,obs,count,arms,labels):
    shapes=[]
    with torch.no_grad():
        x=torch.as_tensor(obs,device='cuda')
        for bank in range(3):
            _,a,b=policy(x,torch.full((arms,),bank,dtype=torch.int64,device='cuda'))
            shapes.append((dense.array(a).copy(),dense.array(b).copy()))
    old_a,old_b=shapes[0]
    report={}
    for bank,name in ((1,'old_to_first'),(2,'old_to_endpoint')):
        a,b=shapes[bank]
        kl=beta_kl(old_a,old_b,a,b).reshape(count,arms)
        report[name]={arm:kl[:,j].tolist() for j,arm in enumerate(labels)}
    return report


def finite_directional_pg(rewards,baseline,scores,gamma):
    """Full finite-horizon trajectory derivative, averaged over no states here."""
    dense.require_finite('Finite directional PG', rewards=rewards, baseline=baseline, scores=scores)
    h,states=rewards.shape
    continuation=np.zeros(states,dtype=np.float64)
    controlled=np.zeros((states,scores.shape[-1]),dtype=np.float64)
    uncontrolled=np.zeros_like(controlled)
    for step in range(h-1,-1,-1):
        continuation=rewards[step]+gamma*continuation
        weight=gamma**step*scores[step]
        controlled+=weight*(continuation-baseline[step])[:,None]
        uncontrolled+=weight*continuation[:,None]
    return controlled,uncontrolled


def matched_gae_transport(rewards,values,terminal_value,scores,gamma,gae_lambda=.95):
    """Same-trajectory finite augmented MC and GAE, not infinite-return truth."""
    rewards=np.asarray(rewards,dtype=np.float64)
    values=np.asarray(values,dtype=np.float64)
    terminal_value=np.asarray(terminal_value,dtype=np.float64)
    scores=np.asarray(scores,dtype=np.float64)
    dense.require_finite('Matched GAE transport', rewards=rewards, values=values,
                         terminal_value=terminal_value, scores=scores)
    horizon,states=rewards.shape
    if values.shape!=rewards.shape or terminal_value.shape!=(states,) or scores.shape[:2]!=rewards.shape:
        raise ValueError('Matched GAE transport trace shape mismatch')
    continuation=terminal_value.copy()
    gae=np.zeros(states);gae_one=np.zeros(states)
    mc_pg=np.zeros((states,scores.shape[-1]));gae_pg=np.zeros_like(mc_pg);one_pg=np.zeros_like(mc_pg)
    next_value=terminal_value
    for step in range(horizon-1,-1,-1):
        continuation=rewards[step]+gamma*continuation
        delta=rewards[step]+gamma*next_value-values[step]
        gae=delta+gamma*gae_lambda*gae
        gae_one=delta+gamma*gae_one
        weight=gamma**step*scores[step]
        mc_pg+=weight*(continuation-values[step])[:,None]
        gae_pg+=weight*gae[:,None]
        one_pg+=weight*gae_one[:,None]
        next_value=values[step]
    tail_pg=gamma**horizon*scores.sum(0)*terminal_value[:,None]
    return dict(terminal_value=terminal_value,terminal_tail_directional_pg=tail_pg,
        augmented_mc_directional_pg=mc_pg,gae_lambda_directional_pg=gae_pg,
        gae_one_directional_pg=one_pg,gae_minus_augmented_mc_directional_pg=gae_pg-mc_pg,
        gae_one_minus_augmented_mc_max_abs_error=float(np.max(np.abs(one_pg-mc_pg))))


def simulate(agent,banks,deltas,states,cli,mode,replicate,policy,directional,value_fn=None):
    labels=HYBRID_ARMS if cli.hybrid_arms else BASIC_ARMS
    arms,count=len(labels),len(states)
    if getattr(cli,'gae_transport',False) and value_fn is None:value_fn=compiled_old_value(agent)
    envs=make_mujoco_vector_env('HalfCheetah-v4',count*arms,num_threads=2)
    try:
        envs.reset(seed=cli.seed)
        restoration=restore(envs,states,arms)
        obs=np.repeat(np.asarray([s['normalized_obs'] for s in states],dtype=np.float32),arms,axis=0)
        std=np.repeat(np.asarray([s['reward_std'] for s in states]),arms)
        dense.require_finite('Closed-loop initial inputs', normalized_obs=obs, reward_std=std)
        if np.any(std <= 0):raise ValueError('Frozen closed-loop reward scale must be positive')
        rng=np.random.default_rng(cli.seed+100003*replicate)
        start=torch.as_tensor(arm_policy_indices(0,cli.hybrid_arms),device='cuda')
        later=torch.as_tensor(arm_policy_indices(1,cli.hybrid_arms),device='cuda')
        raw_discount=np.zeros((count,arms));raw_sum=np.zeros_like(raw_discount)
        norm_discount=np.zeros_like(raw_discount);norm_sum=np.zeros_like(raw_discount)
        speed_discount=np.zeros_like(raw_discount);cost_discount=np.zeros_like(raw_discount)
        speed_sum=np.zeros_like(raw_discount);cost_sum=np.zeros_like(raw_discount)
        clipped_reward_counts=np.zeros((count,arms),dtype=np.int64)
        series_velocity=[];series_cost=[]
        old_rewards=[];old_norm_rewards=[];directional_scores=[];baseline_values=[]
        clipping_atoms=0;theoretical_atom_sum=0.;theoretical_atom_max=0.;primal_error=0.
        horizon_outputs={}
        cost_weight=float(envs.envs[0].unwrapped._ctrl_cost_weight)
        forward_weight=float(envs.envs[0].unwrapped._forward_reward_weight)
        for step in range(max(cli.horizons)):
            with torch.no_grad():
                z,a,b=policy(torch.as_tensor(obs,device='cuda'),start if step==0 else later)
            logits=dense.array(z).copy();alpha=dense.array(a).copy();beta=dense.array(b).copy()
            dense.require_finite(f'Closed-loop policy step {step}', logits=logits, alpha=alpha, beta=beta)
            uniforms=np.repeat(rng.uniform(1e-7,1-1e-7,size=(count,agent.action_dim)),arms,axis=0)
            native=dense.physics.native_action(alpha,beta,mode,uniforms,logits)
            dense.require_finite(f'Closed-loop draw step {step}', native=native)
            if mode=='sample' and directional is not None:
                old_obs=obs.reshape(count,arms,-1)[:,0].copy()
                with torch.no_grad():
                    old_z,u_first,u_end,value=directional(torch.as_tensor(old_obs,device='cuda'))
                dense.require_finite(f'Closed-loop directional step {step}', old_logits=dense.array(old_z),
                    first_jvp=dense.array(u_first), endpoint_jvp=dense.array(u_end), baseline=dense.array(value))
                old_logits=logits.reshape(count,arms,-1)[:,0]
                primal_error=max(primal_error,float(np.max(np.abs(dense.array(old_z)-old_logits))))
                sa=alpha.reshape(count,arms,-1)[:,0];sb=beta.reshape(count,arms,-1)[:,0]
                old_native=native.reshape(count,arms,-1)[:,0].astype(np.float64)
                score=dense.logit_score(old_native,sa.astype(np.float64),sb.astype(np.float64),old_logits.astype(np.float64))
                directions=np.stack((dense.array(u_first),dense.array(u_end)),axis=-1)
                dense.require_finite(f'Closed-loop score step {step}', logit_score=score)
                directional_score=np.einsum('sd,sdk->sk',score,directions)
                dense.require_finite(f'Closed-loop directional score step {step}', directional_score=directional_score)
                directional_scores.append(directional_score)
                baseline_values.append(dense.array(value).copy())
                clipping_atoms+=int(np.count_nonzero((old_native==np.float32(1e-6))|(old_native==np.float32(1-1e-6))))
                mass=betainc(sa.astype(np.float64),sb.astype(np.float64),1e-6)+betainc(sb.astype(np.float64),sa.astype(np.float64),1e-6)
                theoretical_atom_sum+=float(mass.sum());theoretical_atom_max=max(theoretical_atom_max,float(mass.max()))
            action=(2*native-1).astype(np.float32)
            dense.require_finite(f'Closed-loop action step {step}', physical_action=action)
            raw,reward,terms,truncs,_=envs.step(action)
            dense.require_finite(f'Closed-loop physical transition step {step}', raw_next_obs=raw, raw_reward=reward)
            if np.any(terms|truncs):raise ValueError('Closed-loop continuation crossed reset/termination boundary')
            cost=np.square(action).sum(-1).astype(np.float64)*cost_weight
            velocity=(reward+cost)/forward_weight
            scaled_reward=reward/std
            dense.require_finite(f'Closed-loop pre-clip reward step {step}', normalized_reward=scaled_reward)
            shaped=np.clip(scaled_reward,-10,10).astype(np.float32).astype(np.float64)
            dense.require_finite(f'Closed-loop reward step {step}', normalized_reward=shaped,
                                 velocity=velocity, control_cost=cost)
            clipped_reward_counts+=(np.abs(reward/std)>10).reshape(count,arms)
            r=reward.reshape(count,arms);c=cost.reshape(count,arms);v=velocity.reshape(count,arms);nr=shaped.reshape(count,arms)
            w=cli.gamma**step
            raw_discount+=w*r;raw_sum+=r;norm_discount+=w*nr;norm_sum+=nr
            speed_discount+=w*v;cost_discount+=w*c;speed_sum+=v;cost_sum+=c
            series_velocity.append(v.mean(0));series_cost.append(c.mean(0))
            if mode=='sample' and directional is not None:
                old_rewards.append(r[:,0].copy());old_norm_rewards.append(nr[:,0].copy())
            obs=dense.normalize_frozen(raw,states,arms)
            dense.require_finite(f'Closed-loop accumulation step {step}', normalized_obs=obs,
                raw_discounted_return=raw_discount, raw_undiscounted_return=raw_sum,
                normalized_discounted_return=norm_discount, normalized_undiscounted_return=norm_sum)
            if step+1 in cli.horizons:
                output=dict(raw_discounted=raw_discount.copy(),raw_undiscounted=raw_sum.copy(),
                    normalized_discounted=norm_discount.copy(),normalized_undiscounted=norm_sum.copy(),
                    velocity_discounted=speed_discount.copy(),control_cost_discounted=cost_discount.copy(),
                    velocity_mean=speed_sum/(step+1),control_cost_mean=cost_sum/(step+1),
                    occupancy_kl=occupancy_kl(policy,obs,count,arms,labels))
                if mode=='sample' and directional is not None:
                    scores=np.asarray(directional_scores)
                    baseline=np.asarray(baseline_values,dtype=np.float64)
                    fixed_std=np.asarray([s['reward_std'] for s in states])[None]
                    raw_pg,raw_no_baseline=finite_directional_pg(np.asarray(old_rewards),baseline*fixed_std,scores,cli.gamma)
                    norm_pg,norm_no_baseline=finite_directional_pg(np.asarray(old_norm_rewards),baseline,scores,cli.gamma)
                    output['directional_pg']=dict(raw=raw_pg,raw_without_baseline=raw_no_baseline,
                        normalized=norm_pg,normalized_without_baseline=norm_no_baseline)
                    if getattr(cli,'gae_transport',False):
                        with torch.no_grad():
                            terminal_all=value_fn(torch.as_tensor(obs,device='cuda'))
                        terminal_all=dense.array(terminal_all).copy().astype(np.float64).reshape(count,arms)
                        dense.require_finite(f'Closed-loop terminal value at {step+1}', old_critic_value=terminal_all)
                        terminal=terminal_all[:,0].copy()
                        output['terminal_old_critic_value']=terminal_all
                        output['raw_tailaug_discounted']=raw_discount+cli.gamma**(step+1)*terminal_all*fixed_std.T
                        output['normalized_tailaug_discounted']=norm_discount+cli.gamma**(step+1)*terminal_all
                        output['normalized_reward_clip_counts']=clipped_reward_counts.copy()
                        output['gae_transport']=dict(
                            normalized=matched_gae_transport(np.asarray(old_norm_rewards),baseline,terminal,scores,cli.gamma,cli.gae_lambda),
                            raw=matched_gae_transport(np.asarray(old_rewards),baseline*fixed_std,
                                terminal*fixed_std.flatten(),scores,cli.gamma,cli.gae_lambda))
                horizon_outputs[step+1]=output
        trace=None
        if mode=='sample' and directional is not None:
            trace=dict(directional_score=np.asarray(directional_scores).tolist(),
                old_critic_normalized_value=np.asarray(baseline_values).tolist(),
                raw_reward=np.asarray(old_rewards).tolist(),
                frozen_normalized_reward=np.asarray(old_norm_rewards).tolist(),
                dimensions='time, selected_state, [first_update_chord, endpoint_chord] for scores; time, selected_state otherwise',
                fixed_reward_std=[s['reward_std'] for s in states])
        return dict(horizons=horizon_outputs,restore_max_abs_error=restoration,
            old_trajectory_directional_trace=trace,
            velocity_mean_time_series=np.asarray(series_velocity),control_cost_mean_time_series=np.asarray(series_cost),
            directional_score_primal_max_abs_error=primal_error,actual_old_trajectory_clipped_dimensions=clipping_atoms,
            theoretical_old_trajectory_expected_clipped_dimensions=theoretical_atom_sum,
            maximum_old_trajectory_clip_probability=theoretical_atom_max)
    finally:
        envs.close()


def conditional_summary(values,deterministic=False):
    x=np.asarray(values,dtype=np.float64) # replicas x selected states
    replicas=x.shape[0]
    mean=x.mean(0)
    averaged=x.mean(1)
    if replicas>1 and not deterministic:
        se=x.std(0,ddof=1)/np.sqrt(replicas)
        aggregate_se=float(averaged.std(ddof=1)/np.sqrt(replicas))
        critical=float(student_t.ppf(.975,replicas-1))
        intervals=np.stack((mean-critical*se,mean+critical*se),axis=-1).tolist()
        aggregate_interval=[float(averaged.mean()-critical*aggregate_se),float(averaged.mean()+critical*aggregate_se)]
    else:
        se=np.zeros_like(mean);aggregate_se=0.;critical=None;intervals=None;aggregate_interval=None
    return dict(per_state_mean=mean.tolist(),per_state_conditional_mc_se=se.tolist(),
        per_state_conditional_mc_95_interval=intervals,replica_state_values=x.tolist(),
        selected_state_average_mean=float(averaged.mean()),selected_state_average_conditional_mc_se=aggregate_se,
        selected_state_average_conditional_mc_95_interval=aggregate_interval,
        effective_noise_replicates=replicas,student_t_critical=critical,deterministic=deterministic,
        uncertainty_scope='Continuation noise conditional on these selected training states, not training seeds or state-population uncertainty')


def summarize_mode(replicates,cli,labels,mode):
    deterministic=mode!='sample'
    outputs=[]
    for horizon in cli.horizons:
        entries=[r['horizons'][horizon] for r in replicates]
        metrics={}
        metric_names=('raw_discounted','raw_undiscounted','normalized_discounted','normalized_undiscounted',
                       'velocity_discounted','control_cost_discounted','velocity_mean','control_cost_mean')
        if mode=='sample' and getattr(cli,'gae_transport',False):
            metric_names+=('raw_tailaug_discounted','normalized_tailaug_discounted')
        for metric in metric_names:
            data=np.stack([e[metric] for e in entries])
            metrics[metric]=dict(policy_returns={label:conditional_summary(data[:,:,i],deterministic) for i,label in enumerate(labels)},
                paired_gains={label:conditional_summary(data[:,:,i]-data[:,:,0],deterministic) for i,label in enumerate(labels) if i})
            if cli.hybrid_arms:
                metrics[metric]['feedback_decomposition']={}
                for full,first,later in (('first_all','first_t0_only','first_after_t0'),('endpoint_all','endpoint_t0_only','endpoint_after_t0')):
                    fi,ti,li=(labels.index(x) for x in (full,first,later))
                    metrics[metric]['feedback_decomposition'][full]=dict(
                        first_action_only=conditional_summary(data[:,:,ti]-data[:,:,0],deterministic),
                        later_actions_only=conditional_summary(data[:,:,li]-data[:,:,0],deterministic),
                        interaction=conditional_summary(data[:,:,fi]-data[:,:,ti]-data[:,:,li]+data[:,:,0],deterministic),
                        interpretation='Exact finite paired-return feedback interaction; not a gradient or critic-error decomposition')
        result=dict(horizon=horizon,discount=cli.gamma,metrics=metrics,
            occupancy_kl_replicates=[e['occupancy_kl'] for e in entries])
        if mode=='sample' and cli.directional_pg:
            result['full_finite_directional_pg']={}
            for metric in ('raw','raw_without_baseline','normalized','normalized_without_baseline'):
                data=np.stack([e['directional_pg'][metric] for e in entries])
                result['full_finite_directional_pg'][metric]={label:conditional_summary(data[:,:,i]) for i,label in enumerate(('first_all','endpoint_all'))}
            result['finite_gain_minus_directional_prediction']={}
            for actual,predicted in (('raw_discounted','raw'),('normalized_discounted','normalized')):
                actual_data=np.stack([e[actual] for e in entries])
                prediction=np.stack([e['directional_pg'][predicted] for e in entries])
                result['finite_gain_minus_directional_prediction'][actual]={label:conditional_summary(
                    actual_data[:,:,labels.index(label)]-actual_data[:,:,0]-prediction[:,:,i])
                    for i,label in enumerate(('first_all','endpoint_all'))}
            if getattr(cli,'gae_transport',False):
                result['matched_gae_transport']=dict(
                    gae_lambda=cli.gae_lambda,
                    objective='Same old-policy finite trajectory plus frozen old-critic terminal tail, not true infinite-horizon MC',
                    interpretation='Paired GAE-minus-tail-augmented-MC estimator difference for this finite augmented objective; conditional MC uncertainty remains, no global critic-accuracy claim',
                    units={},
                    all_arm_terminal_old_critic_value_replicates=[e['terminal_old_critic_value'].tolist() for e in entries],
                    normalized_reward_clip_counts_replicates=[e['normalized_reward_clip_counts'].tolist() for e in entries])
                for units in ('raw','normalized'):
                    rows=[e['gae_transport'][units] for e in entries]
                    transport=dict(terminal_value_replicas=[r['terminal_value'].tolist() for r in rows],
                        gae_one_minus_augmented_mc_max_abs_error=max(r['gae_one_minus_augmented_mc_max_abs_error'] for r in rows))
                    for metric in ('terminal_tail_directional_pg','augmented_mc_directional_pg',
                                   'gae_lambda_directional_pg','gae_one_directional_pg','gae_minus_augmented_mc_directional_pg'):
                        data=np.stack([r[metric] for r in rows])
                        transport[metric]={label:conditional_summary(data[:,:,i])
                            for i,label in enumerate(('first_all','endpoint_all'))}
                    result['matched_gae_transport']['units'][units]=transport
                    actual_metric=units+'_tailaug_discounted'
                    actual=np.stack([e[actual_metric] for e in entries])
                    derivative=np.stack([r['augmented_mc_directional_pg'] for r in rows])
                    transport['matched_tailaug_gain_minus_directional_prediction']={label:conditional_summary(
                        actual[:,:,labels.index(label)]-actual[:,:,0]-derivative[:,:,i])
                        for i,label in enumerate(('first_all','endpoint_all'))}
        outputs.append(result)
    velocity=np.stack([r['velocity_mean_time_series'] for r in replicates])
    cost=np.stack([r['control_cost_mean_time_series'] for r in replicates])
    return dict(mode=mode,requested_noise_replicates=cli.replicates,effective_noise_replicates=len(replicates),
        policy_arms=list(labels),horizons=outputs,
        old_trajectory_directional_trace_replicates=[r['old_trajectory_directional_trace'] for r in replicates],
        paired_velocity_mean_time_series={label:(velocity[:,:,i]-velocity[:,:,0]).tolist() for i,label in enumerate(labels) if i},
        paired_control_cost_mean_time_series={label:(cost[:,:,i]-cost[:,:,0]).tolist() for i,label in enumerate(labels) if i},
        maximum_restore_error=max(r['restore_max_abs_error'] for r in replicates),
        directional_score_primal_max_abs_error=max(r['directional_score_primal_max_abs_error'] for r in replicates),
        score_atom_approximation=dict(actual_clipped_dimensions=[r['actual_old_trajectory_clipped_dimensions'] for r in replicates],
            theoretical_expected_clipped_dimensions=[r['theoretical_old_trajectory_expected_clipped_dimensions'] for r in replicates],
            maximum_clip_probability=max(r['maximum_old_trajectory_clip_probability'] for r in replicates),
            caveat='Continuous Beta score at clipped actions omits clipping atoms and excludes 2e-7 uniform mass; probability is reported, gradient bias not bounded by probability alone.'))


def analyze(path,cli):
    snapshot_path=Path(path)
    snapshot_hash=hash_file(snapshot_path)
    snapshot=torch.load(snapshot_path,map_location='cpu',weights_only=False)
    trace_path=snapshot_path.with_name(f"diagnostic_updates_{snapshot['global_step']}.pt")
    trace_hash=hash_file(trace_path)
    trace=torch.load(trace_path,map_location='cpu',weights_only=False)
    if snapshot['args']['env_id']!='HalfCheetah-v4' or 'transitions' not in snapshot:
        raise ValueError('Need dense v12 HalfCheetah snapshot')
    if snapshot['args'].get('policy_chart','alpha_beta')!='alpha_beta':
        raise ValueError('This audit is for matched v12 alpha_beta policies, not chart checkpoints')
    if float(snapshot['args']['gamma'])!=cli.gamma:raise ValueError('Discount must match learner')
    if cli.gae_transport:cli.gae_lambda=float(snapshot['args']['gae_lambda'])
    states=dense.select_states(snapshot,cli.states,cli.max_age)
    if max(s['episode_age'] for s in states)+max(cli.horizons)>=1000:
        raise ValueError('Selected episode age plus horizon crosses TimeLimit')
    envs=make_mujoco_vector_env('HalfCheetah-v4',cli.states,num_threads=2)
    try:
        agent=trainer.Agent(envs).cuda().eval()
        agent.load_state_dict(snapshot['model'],strict=True)
    finally:
        envs.close()
    banks,deltas,reconstruction=reconstruct_banks(agent,snapshot,trace)
    labels=HYBRID_ARMS if cli.hybrid_arms else BASIC_ARMS
    policy=compiled_policy(agent,banks,cli.states,len(labels))
    directional=compiled_directional(agent,banks,deltas) if cli.directional_pg else None
    value_fn=compiled_old_value(agent) if cli.gae_transport else None
    results=[]
    for mode in cli.modes:
        effective=cli.replicates if mode=='sample' else 1
        rows=[simulate(agent,banks,deltas,states,cli,mode,r,policy,directional,value_fn) for r in range(effective)]
        results.append(summarize_mode(rows,cli,labels,mode))
        print(json.dumps(dict(snapshot=str(path),mode=mode,horizons=cli.horizons,status='complete')),flush=True)
    if hash_file(snapshot_path)!=snapshot_hash or hash_file(trace_path)!=trace_hash:
        raise ValueError('Snapshot or trace changed during execution')
    return dict(snapshot=str(snapshot_path.resolve()),snapshot_sha256=snapshot_hash,
        update_trace=str(trace_path.resolve()),update_trace_sha256=trace_hash,global_step=snapshot['global_step'],
        selected_states=[{k:v for k,v in s.items() if k!='integration_state'} for s in states],
        weight_reconstruction=reconstruction,closed_loop=results,
        limitations=['Finite returns from selected training states, not full episode benchmark scores or a global optimum.',
            'Per-selected-state observation and reward normalizers are frozen; the original online continuation adapted them.',
            'First-update weights reconstructed from FP32 deltas; endpoint loaded exactly from saved model_after_update.',
            'Four independent continuation replicas measure conditional Monte Carlo error; Student-t intervals are approximate, selected states are fixed.',
            'Mean and mode change the policy law; their physical effects are behavioral diagnostics, not stochastic PG validation.',
            'Hybrid return interactions quantify finite closed-loop feedback, not critic bias.',
            'Full directional PG uses a fixed actual weight chord and a local derivative at the old policy; discrepancy from finite gains includes curvature, feedback and Monte Carlo noise.',
            'Historical simulator RNG is absent; HalfCheetah is deterministic and no continuation reset is permitted.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',action='append',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--states',type=int,default=32)
    parser.add_argument('--max-age',type=int,default=400)
    parser.add_argument('--replicates',type=int,default=4)
    parser.add_argument('--horizons',type=int,nargs='+',default=[64,256,512])
    parser.add_argument('--modes',nargs='+',choices=('sample','mean','mode'),default=['sample','mean','mode'])
    parser.add_argument('--hybrid-arms',action='store_true')
    parser.add_argument('--directional-pg',action='store_true')
    parser.add_argument('--gae-transport',action='store_true')
    parser.add_argument('--gamma',type=float,default=.99)
    parser.add_argument('--seed',type=int,default=1)
    cli=parser.parse_args()
    if cli.states<2 or cli.replicates<2 or cli.max_age<0 or min(cli.horizons)<1 or cli.max_age+max(cli.horizons)>=1000:
        parser.error('Need>=2states/replicas, nonnegative ages, and horizons within no-reset range')
    if len(set(cli.horizons))!=len(cli.horizons) or len(set(cli.modes))!=len(cli.modes):parser.error('Duplicate horizons/modes')
    if cli.gae_transport and (not cli.directional_pg or 'sample' not in cli.modes):
        parser.error('--gae-transport requires --directional-pg and sample mode')
    configure_runtime(matmul_precision='highest',allow_tf32=False)
    torch._dynamo.config.recompile_limit=64
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; submit through mlq')
    result=dict(protocol=vars(cli),results=[],source_sha256=hash_file(__file__),
        helper_sha256={str(p.relative_to(ROOT)):hash_file(p) for p in (ROOT/'scripts/audit_sepclip_dense_pg.py',ROOT/'scripts/audit_sepclip_physics.py',Path(trainer.__file__))})
    for path in cli.snapshot:
        result['results'].append(analyze(path,cli))
        output=Path(cli.out);output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
