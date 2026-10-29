"""State-matched conditional physical PG for dense v12 snapshots; mlq/CUDA only.

First-action forks freeze the recorded observation/reward scales independently
at every selected training state. Continuation policy is fixed; horizons are
finite physical returns, with a separately reported estimated critic tail.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import betainc, betaincinv, digamma, expit
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cleanrl import ppo_continuous_action_sepclip_transition_microscopy_v12 as trainer
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.runtime import configure_runtime

_spec = importlib.util.spec_from_file_location('sepclip_frozen_physics', ROOT/'scripts/audit_sepclip_physics.py')
physics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(physics)


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def require_finite(context, **fields):
    """Fail closed on required diagnostic arrays; never omit bad coordinates."""
    for name, value in fields.items():
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError(f'{context}: nonfinite {name}')


def select_states(snapshot, count=32, max_age=400):
    dense = snapshot['transitions']
    age = array(dense['episode_age'])
    if age.ndim != 2 or count < 2:
        raise ValueError('Require at least two states and time-by-env episode ages')
    steps, envs = age.shape
    selections = []
    # Round-robin environment rows; stratify eligible timesteps within each row.
    allocations = [count//envs + int(row < count % envs) for row in range(envs)]
    for row, n in enumerate(allocations):
        if not n:
            continue
        eligible = np.flatnonzero((age[:, row] >= 0) & (age[:, row] <= max_age))
        if len(eligible) < n:
            raise ValueError(f'Row {row} has {len(eligible)} eligible states, need {n}')
        ranks = np.floor((np.arange(n)+.5)*len(eligible)/n).astype(int)
        selections.extend((int(eligible[k]), row) for k in ranks)
    selections.sort()
    if len(set(selections)) != count:
        raise ValueError('State selector produced duplicate states')
    flat_obs = array(snapshot['batch']['obs']).reshape(steps, envs, -1)
    native = array(snapshot['batch']['native']).reshape(steps, envs, -1)
    advantages = array(snapshot['batch']['advantages']).reshape(steps, envs)
    values = array(snapshot['batch']['values']).reshape(steps, envs)
    states = []
    for t, row in selections:
        raw = array(dense['raw_obs'])[t, row].copy()
        means = array(dense['obs_means'])[t, row].copy()
        variances = array(dense['obs_variances'])[t, row].copy()
        obs = flat_obs[t, row].copy()
        require_finite(f'Selected state ({t},{row})', raw_obs=raw, obs_means=means,
            obs_variances=variances, normalized_obs=obs,
            native_action=native[t,row], reward_std=array(dense['reward_std'])[t,row],
            physical_action=array(dense['physical_actions'])[t,row],
            stored_raw_reward=array(dense['raw_rewards'])[t,row],
            stored_raw_next_obs=array(dense['raw_next_obs'])[t,row],
            integration_state=array(dense['integration_states'])[t,row])
        if np.any(variances < 0):
            raise ValueError(f'Selected state ({t},{row}) has negative observation variance')
        normalized_preclip=(raw-means)/np.sqrt(variances+1e-8)
        require_finite(f'Selected pre-clip normalization ({t},{row})', normalized_obs=normalized_preclip)
        replay = np.clip(normalized_preclip, -10, 10).astype(np.float32)
        require_finite(f'Selected normalization ({t},{row})', replay=replay,
                       stored_advantage=advantages[t,row], stored_value=values[t,row])
        error = float(np.max(np.abs(replay-obs)))
        if error > 1e-6:
            raise ValueError(f'State ({t},{row}) normalization replay error {error}')
        states.append(dict(t=t, row=row, episode_age=int(age[t,row]),
            normalized_obs=obs.tolist(), raw_obs=raw.tolist(), obs_means=means.tolist(),
            obs_variances=variances.tolist(), normalization_replay_max_abs_error=error,
            reward_std=float(array(dense['reward_std'])[t,row]),
            native_action=native[t,row].tolist(), stored_advantage=float(advantages[t,row]),
            physical_action=array(dense['physical_actions'])[t,row].tolist(),
            stored_raw_reward=float(array(dense['raw_rewards'])[t,row]),
            stored_raw_next_obs=array(dense['raw_next_obs'])[t,row].tolist(),
            stored_value=float(values[t,row]), integration_state=array(dense['integration_states'])[t,row].copy()))
    return states


def normalize_frozen(raw, states, branches):
    means = np.repeat(np.asarray([s['obs_means'] for s in states]), branches, axis=0)
    variances = np.repeat(np.asarray([s['obs_variances'] for s in states]), branches, axis=0)
    if np.any(variances < 0):raise ValueError('Frozen normalizer has negative observation variance')
    normalized=(raw-means)/np.sqrt(variances+1e-8)
    require_finite('Frozen pre-clip normalization', normalized_obs=normalized)
    return np.clip(normalized, -10, 10).astype(np.float32)


def logit_score(native, alpha, beta, logits):
    a = np.log(native)-digamma(alpha)+digamma(alpha+beta)
    b = np.log1p(-native)-digamma(beta)+digamma(alpha+beta)
    return np.concatenate((a,b), axis=-1)*expit(logits)


def conditional_pg(scores, returns):
    """LOO action baseline is independent of each first action, conditional on noise."""
    if returns.shape[1] < 2:
        raise ValueError('Need at least two independent first actions')
    baseline = (returns.sum(1,keepdims=True)-returns)/(returns.shape[1]-1)
    return (scores*(returns-baseline)[...,None]).mean(1)


def summary(value):
    x = np.asarray(value, dtype=np.float64)
    return dict(mean=float(x.mean()), rms=float(np.sqrt(np.mean(x*x))),
                std=float(x.std()), min=float(x.min()), max=float(x.max()))


def replicate_summary(values):
    x = np.asarray(values, dtype=np.float64)
    return dict(mean=x.mean(0).tolist(), replicate_values=x.tolist(),
                replicate_sd=x.std(0,ddof=1).tolist(),
                replicate_se=(x.std(0,ddof=1)/np.sqrt(len(x))).tolist())


def compiled_functions(agent):
    def policy(x):
        z = agent.actor(x)
        alpha,beta = (1+torch.nn.functional.softplus(z)).chunk(2,-1)
        return z,alpha,beta
    options = {'triton.cudagraphs': False}
    return (torch.compile(policy, fullgraph=True, dynamic=True, options=options),
            torch.compile(agent.get_value, fullgraph=True, dynamic=True, options=options),
            torch.compile(agent.actor, fullgraph=True, dynamic=True, options=options))


def dense_boundary_audit(snapshot, value_fn, gamma, lam):
    dense, batch = snapshot['transitions'], snapshot['batch']
    require_finite('Dense historical GAE inputs',
        **{name:array(dense[name]) for name in ('rewards','terminations','truncations',
           'truncation_values','tail_value','raw_rewards','reward_std',
           'normalized_transition_obs','normalized_next_obs')},
        values=array(batch['values']), stored_advantages=array(batch['advantages']))
    if np.any(array(dense['reward_std']) <= 0):
        raise ValueError('Dense historical reward scale must be positive')
    rewards = torch.as_tensor(array(dense['rewards']), device='cuda')
    values = torch.as_tensor(array(batch['values']).reshape(rewards.shape), device='cuda')
    terms = torch.as_tensor(array(dense['terminations']), device='cuda')
    truncs = torch.as_tensor(array(dense['truncations']), device='cuda')
    truncation = torch.as_tensor(array(dense['truncation_values']), device='cuda')
    tail = torch.as_tensor(array(dense['tail_value']), device='cuda')
    fn = trainer.get_gae_fn(compiled=True, mode='reduce-overhead')
    with torch.no_grad():
        recomputed, _ = fn(rewards, values, terms, truncs, truncation, tail, gamma, lam)
        stored = torch.as_tensor(array(batch['advantages']).reshape(rewards.shape),device='cuda')
        error = float((recomputed-stored).abs().max())
        transition = torch.as_tensor(array(dense['normalized_transition_obs']),device='cuda')
        successor = value_fn(transition.flatten(0,1)).reshape(rewards.shape)
        require_finite('Dense historical critic replay', successor_value=array(successor))
        replay_truncation = float((successor[truncs.bool()]-truncation[truncs.bool()]).abs().max()) if bool(truncs.any()) else 0.
    require_finite('Dense GAE replay errors', gae_error=error, truncation_error=replay_truncation)
    if error > 2e-5 or replay_truncation > 2e-5:
        raise ValueError(f'Dense GAE/bootstrap mismatch: GAE={error}, truncation={replay_truncation}')
    raw = array(dense['raw_rewards'])
    scaled_reward=raw/array(dense['reward_std'])
    require_finite('Dense historical pre-clip reward replay', normalized_reward=scaled_reward)
    rebuilt = np.clip(scaled_reward,-10,10).astype(np.float32)
    reward_error = float(np.max(np.abs(rebuilt-array(dense['rewards']))))
    require_finite('Dense reward replay error', reward_error=reward_error)
    if reward_error > 1e-6:
        raise ValueError(f'Applied reward normalization replay mismatch {reward_error}')
    boundary = array(dense['truncations']).astype(bool)
    reset = array(dense['normalized_next_obs'])
    final = array(dense['normalized_transition_obs'])
    return dict(gae_replay_max_abs_error=error, applied_reward_replay_max_abs_error=reward_error,
        truncation_bootstrap_replay_max_abs_error=replay_truncation,
        terminations=int(array(dense['terminations']).sum()),truncations=int(boundary.sum()),
        truncated_final_vs_reset_observation_rms=float(np.sqrt(np.mean((final[boundary]-reset[boundary])**2))) if boundary.any() else None,
        final_observation_semantics='normalized_transition_obs is pre-reset; normalized_next_obs may be autoreset',
        gamma=gamma,gae_lambda=lam)


def parameter_gradients(actor_fn, agent, obs, cotangent):
    x = torch.as_tensor(obs, dtype=torch.float32, device='cuda')
    z = actor_fn(x)
    cot = torch.as_tensor(cotangent, dtype=z.dtype, device='cuda')/len(obs)
    return torch.autograd.grad(z, tuple(agent.actor.parameters()), grad_outputs=cot)


def training_gradients(agent, obs, native, advantage):
    def objective(x,y,adv):
        z=agent.actor(x)
        a,b=(1+torch.nn.functional.softplus(z)).chunk(2,-1)
        logp=(torch.distributions.Beta(a,b,validate_args=False).log_prob(y)-agent.log_action_scale).sum(-1)
        return (logp*adv).mean()
    compiled=torch.compile(objective,fullgraph=True,dynamic=True,options={'triton.cudagraphs':False})
    result=compiled(torch.as_tensor(obs,dtype=torch.float32,device='cuda'),
        torch.as_tensor(native,dtype=torch.float32,device='cuda'),
        torch.as_tensor(advantage,dtype=torch.float32,device='cuda'))
    return torch.autograd.grad(result,tuple(agent.actor.parameters()))


def compare_gradients(agent, actor_fn, states, observed_scores, horizon, stored, full):
    obs = np.asarray([s['normalized_obs'] for s in states], dtype=np.float32)
    adv = np.asarray([s['stored_advantage'] for s in states])
    output = {}
    fields = horizon['observed_action']
    for name in ('independent_advantage_finite_normalized','independent_advantage_bootstrap_normalized'):
        oracle_adv = np.asarray(fields[name]['mean'])
        oracle = parameter_gradients(actor_fn,agent,obs,observed_scores*oracle_adv[:,None])
        rows = {}
        for (parameter,_),g,o,f in zip(agent.actor.named_parameters(),stored,oracle,full):
            e = g-o
            rows[parameter] = dict(selected_training_pg_norm=float(g.norm()),oracle_observed_action_pg_norm=float(o.norm()),
                error_pg_norm=float(e.norm()),error_to_training_norm=float(e.norm()/(g.norm()+1e-30)),
                training_oracle_cosine=float((g*o).sum()/(g.norm()*o.norm()+1e-30)),
                training_oracle_pairing=float((g*o).sum()),
                full_rollout_pg_norm=float(f.norm()),selected_full_cosine=float((g*f).sum()/(g.norm()*f.norm()+1e-30)))
        output[name] = rows
    return dict(state_and_observed_action_matched=True,
        comparator='saved normalized obs/native action/GAE at selected dense (t,row)',
        limitations=['Selected observed actions supply one action per state; replicated continuations reduce continuation noise only.',
                    'Oracle advantage subtracts current estimated V; finite horizon and separately labeled critic tail are not true V*.'],
        layers=output)


def fork_replicate(agent, snapshot, states, cli, policy_fn, value_fn, replicate):
    branches = cli.pg_samples+1
    records = [s for s in states for _ in range(branches)]
    n = len(records)
    envs = make_mujoco_vector_env('HalfCheetah-v4', n,num_threads=2)
    try:
        envs.reset(seed=cli.seed)
        restoration = []
        for i,s in enumerate(records):
            base=envs.envs[i].unwrapped
            physics.set_integration_state(base,s['integration_state'])
            import mujoco
            mujoco.mj_forward(base.model,base.data)
            physics.set_integration_state(base,s['integration_state'])
            envs._episode_lengths[i]=s['episode_age']
            envs._limits[i]._elapsed_steps=s['episode_age']
            envs._episode_returns[i]=0
            envs.observations[i]=base._get_obs()
            require_finite(f'Dense restored state {i}', raw_obs=envs.observations[i])
            restoration.append(float(np.max(np.abs(envs.observations[i]-s['raw_obs']))))
        if max(restoration)>1e-10:
            raise ValueError(f'Dense physics restoration mismatch {max(restoration)}')
        start_obs=np.repeat(np.asarray([s['normalized_obs'] for s in states],np.float32),branches,axis=0)
        with torch.no_grad():
            z,a,b = policy_fn(torch.as_tensor(start_obs,device='cuda'))
        z=array(z).reshape(len(states),branches,-1)[:,0]
        a=array(a).reshape(len(states),branches,-1)[:,0]
        b=array(b).reshape(len(states),branches,-1)[:,0]
        require_finite('Initial dense policy', logits=z, alpha=a, beta=b)
        rng=np.random.default_rng(cli.seed+100003*replicate)
        u=rng.uniform(1e-7,1-1e-7,size=(len(states),cli.pg_samples,agent.action_dim))
        independent=np.clip(betaincinv(a[:,None],b[:,None],u),1e-6,1-1e-6).astype(np.float32)
        observed=np.asarray([s['native_action'] for s in states],np.float32)
        initial=np.concatenate((independent,observed[:,None]),axis=1)
        score=logit_score(independent.astype(np.float64),a[:,None],b[:,None],z[:,None])
        observed_score=logit_score(observed.astype(np.float64),a,b,z)
        require_finite('Dense first action', native=initial, logit_score=score,
                       observed_logit_score=observed_score)
        eps=1e-6
        tail_mass=betainc(a.astype(np.float64),b.astype(np.float64),eps)+1-betainc(a.astype(np.float64),b.astype(np.float64),1-eps)
        sampling=dict(theoretical_clip_tail_mass_per_state_dimension=tail_mass.tolist(),
            theoretical_clip_tail_mass_summary=summary(tail_mass),
            actual_independent_first_action_clipped_dimensions=int(np.count_nonzero(
                (independent==np.float32(eps))|(independent==np.float32(1-eps)))),
            actual_observed_first_action_clipped_dimensions=int(np.count_nonzero(
                (observed==np.float32(eps))|(observed==np.float32(1-eps)))),
            inverse_cdf_uniform_range=[1e-7,1-1e-7],excluded_uniform_probability_mass=2e-7,
            score_protocol='Continuous Beta score on clipped native values; atom probability mass reported. Gradient bias from omitted atom corrections is not bounded by probability mass alone.')
        std=np.repeat(np.asarray([s['reward_std'] for s in states]),branches)
        require_finite('Frozen dense reward scale', reward_std=std)
        if np.any(std <= 0):
            raise ValueError('Frozen dense reward scale must be positive')
        raw_total=np.zeros(n)
        normalized_total=np.zeros(n)
        clips=np.zeros(n,dtype=int)
        results={}
        first_step_error = None
        obs=start_obs
        for t in range(max(cli.horizons)):
            if t==0:
                native=initial.reshape(n,-1)
            else:
                with torch.no_grad():
                    logits,alpha,beta=policy_fn(torch.as_tensor(obs,device='cuda'))
                alpha,beta=array(alpha),array(beta)
                require_finite(f'Dense policy step {t}', logits=array(logits), alpha=alpha, beta=beta)
                uniforms=np.repeat(rng.uniform(1e-7,1-1e-7,size=(len(states),agent.action_dim)),branches,axis=0)
                native=physics.native_action(alpha,beta,'sample',uniforms)
            require_finite(f'Dense action step {t}', native=native)
            action=(2*native-1).astype(np.float32)
            if t==0:
                action.reshape(len(states),branches,-1)[:,-1]=np.asarray([s['physical_action'] for s in states],np.float32)
            require_finite(f'Dense physical action step {t}', physical_action=action)
            raw,reward,terms,truncs,_=envs.step(action)
            require_finite(f'Dense physical transition step {t}', raw_next_obs=raw, raw_reward=reward)
            if np.any(terms|truncs):
                raise ValueError('Conditional oracle crossed boundary; age/horizon contract failed')
            if t==0:
                observed_reward=reward.reshape(len(states),branches)[:,-1]
                observed_raw=raw.reshape(len(states),branches,-1)[:,-1]
                first_step_error=dict(raw_reward_max_abs_error=float(np.max(np.abs(observed_reward-[s['stored_raw_reward'] for s in states]))),
                    raw_next_obs_max_abs_error=float(np.max(np.abs(observed_raw-[s['stored_raw_next_obs'] for s in states]))))
                require_finite('Dense observed first-transition replay errors', **first_step_error)
                if max(first_step_error.values())>1e-10:
                    raise ValueError(f'Observed first action does not replay saved transition: {first_step_error}')
            raw_total+=cli.gamma**t*reward
            transformed=reward/std
            require_finite(f'Dense normalized reward step {t}', normalized_reward=transformed)
            clips+=np.abs(transformed)>10
            normalized_total+=cli.gamma**t*np.clip(transformed,-10,10)
            obs=normalize_frozen(raw,states,branches)
            require_finite(f'Dense accumulation step {t}', normalized_obs=obs,
                           raw_return=raw_total, normalized_return=normalized_total)
            if t+1 in cli.horizons:
                with torch.no_grad():
                    successor=array(value_fn(torch.as_tensor(obs,device='cuda'))).reshape(len(states),branches)
                require_finite(f'Dense terminal value at {t+1}', old_critic_value=successor)
                raw_q=raw_total.reshape(len(states),branches).copy()
                q=normalized_total.reshape(len(states),branches).copy()
                bootstrap=q+cli.gamma**(t+1)*successor
                results[t+1]=dict(finite_raw_pg=conditional_pg(score,raw_q[:,:cli.pg_samples]),
                    finite_normalized_pg=conditional_pg(score,q[:,:cli.pg_samples]),
                    bootstrap_normalized_pg=conditional_pg(score,bootstrap[:,:cli.pg_samples]),
                    observed_finite_raw_Q=raw_q[:,-1],observed_finite_normalized_Q=q[:,-1],
                    observed_bootstrap_normalized_Q=bootstrap[:,-1],
                    observed_critic_tail=cli.gamma**(t+1)*successor[:,-1],
                    reward_clip_count=clips.reshape(len(states),branches).sum(1))
        return results,observed_score,z,a,b,max(restoration),first_step_error,sampling
    finally:
        envs.close()


def clipping_sensitivity(snapshot_path,snapshot,states,horizons,agent,policy_fn):
    trace_path=Path(snapshot_path).with_name(f"diagnostic_updates_{snapshot['global_step']}.pt")
    if not trace_path.exists():
        return dict(status='unavailable',reason='No matched v12 update trace',expected_path=str(trace_path))
    trace=torch.load(trace_path,map_location='cpu',weights_only=False)
    obs=torch.as_tensor(np.asarray([s['normalized_obs'] for s in states]),dtype=torch.float32,device='cuda')
    native=np.asarray([s['native_action'] for s in states],dtype=np.float32)
    adv=np.asarray([s['stored_advantage'] for s in states])
    shape=array(snapshot['transitions']['episode_age']).shape
    old_full=array(snapshot['batch']['logprob']).reshape(shape)
    old=torch.as_tensor(np.asarray([old_full[s['t'],s['row']] for s in states]),dtype=torch.float32,device='cuda')
    parameters={k:v.detach().clone() for k,v in agent.actor.named_parameters()}
    def functional_policy(p,x):
        logits=torch.func.functional_call(agent.actor,p,(x,))
        alpha,beta=(1+torch.nn.functional.softplus(logits)).chunk(2,-1)
        return logits,alpha,beta
    functional=torch.compile(functional_policy,fullgraph=True,options={'triton.cudagraphs':False})
    lower=1-float(snapshot['args'].get('clip_coef',.2))
    upper=1+float(snapshot['args'].get('clip_coef_upper',.2))
    output=[]
    def active(a,r):
        return ~(((a>=0)&(r>upper))|((a<0)&(r<lower)))
    for entry in trace['trace']:
        with torch.no_grad():
            z,a,b=functional(parameters,obs)
            logp=(torch.distributions.Beta(a,b,validate_args=False).log_prob(torch.as_tensor(native,device='cuda'))-agent.log_action_scale).sum(-1)
            ratio=array((logp-old).exp())
        train_mask=active(adv,ratio)
        changes={}
        for h in horizons:
            oracle=np.asarray(h['observed_action']['independent_advantage_bootstrap_normalized']['mean'])
            oracle_mask=active(oracle,ratio)
            changes[str(h['horizon'])]=dict(advantage_sign_disagreement=float(np.mean((adv>=0)!=(oracle>=0))),
                active_mask_disagreement=float(np.mean(train_mask!=oracle_mask)),
                masked_advantage_error_rms=float(np.sqrt(np.mean((ratio*(adv*train_mask-oracle*oracle_mask))**2))),
                same_mask_error_rms=float(np.sqrt(np.mean((ratio*train_mask*(adv-oracle))**2))),
                mask_change_term_rms=float(np.sqrt(np.mean((ratio*oracle*(train_mask.astype(float)-oracle_mask.astype(float)))**2))))
        output.append(dict(epoch=entry['epoch'],update=entry['update'],timing='before corresponding update',
            active_fraction=float(train_mask.mean()),ratio_summary=summary(ratio),oracle_comparison=changes))
        for key,delta in entry['retracted_delta'].items():
            if key.startswith('actor.'):
                parameters[key[len('actor.'):]].add_(delta.cuda())
    after=trace['model_after_update']
    reconstruction_error=max(float((p-after['actor.'+k].cuda()).abs().max()) for k,p in parameters.items())
    return dict(status='available',trace_path=str(trace_path),updates=output,
        native_action_dtype=str(native.dtype),logprob_dtype=str(old.dtype),
        old_logprob_source='saved full-rollout FP32 physical-action logprob at exact selected (t,row)',
        final_delta_reconstruction_max_abs_error=reconstruction_error,
        limitations=['Intermediate weights reconstructed by summing FP32 deltas; final reconstruction error reported.',
                    'Mask comparison is estimated finite-H advantage sensitivity, not independent proof of true critic baseline bias.'])


def analyze(path,cli):
    snapshot=torch.load(path,map_location='cpu',weights_only=False)
    if 'transitions' not in snapshot:
        raise ValueError('Dense v12 snapshot required; do not synthesize missing boundaries/states')
    if snapshot['args']['env_id']!='HalfCheetah-v4':
        raise ValueError('Dense no-reset oracle currently supports HalfCheetah-v4 only')
    if abs(float(snapshot['args']['gamma'])-cli.gamma)>1e-12:
        raise ValueError('Oracle gamma must match learner gamma')
    states=select_states(snapshot,cli.states,cli.max_age)
    if max(s['episode_age'] for s in states)+max(cli.horizons)>=1000:
        raise ValueError('Selected state age plus horizon must stay below TimeLimit')
    envs=make_mujoco_vector_env('HalfCheetah-v4',cli.states,num_threads=2)
    try:
        agent=trainer.Agent(envs).cuda().eval()
        agent.load_state_dict(snapshot['model'])
    finally:
        envs.close()
    policy_fn,value_fn,actor_fn=compiled_functions(agent)
    boundaries=dense_boundary_audit(snapshot,value_fn,cli.gamma,float(snapshot['args']['gae_lambda']))
    replicas=[]
    restore_errors=[]
    replay_errors=[]
    sampling_protocol=[]
    for r in range(cli.continuation_replicates):
        result,score,z,a,b,restore,replay,sampling=fork_replicate(agent,snapshot,states,cli,policy_fn,value_fn,r)
        replicas.append(result);restore_errors.append(restore);replay_errors.append(replay);sampling_protocol.append(sampling)
    values=np.asarray([s['stored_value'] for s in states])
    adv=np.asarray([s['stored_advantage'] for s in states])
    selected_obs=np.asarray([s['normalized_obs'] for s in states],np.float32)
    selected_native=np.asarray([s['native_action'] for s in states],np.float32)
    stored_gradient=training_gradients(agent,selected_obs,selected_native,adv)
    full_gradient=training_gradients(agent,array(snapshot['batch']['obs']),
        array(snapshot['batch']['native']),array(snapshot['batch']['advantages']))
    analytic_gradient=parameter_gradients(actor_fn,agent,selected_obs,score*adv[:,None])
    score_replay_relative_error=float(torch.stack([(g-r).square().sum() for g,r in zip(analytic_gradient,stored_gradient)]).sum().sqrt()/
        (torch.stack([g.square().sum() for g in stored_gradient]).sum().sqrt()+1e-30))
    horizons=[]
    for horizon in cli.horizons:
        result=[r[horizon] for r in replicas]
        finite=np.stack([r['observed_finite_normalized_Q'] for r in result])-values
        bootstrap=np.stack([r['observed_bootstrap_normalized_Q'] for r in result])-values
        h=dict(horizon=horizon,discount=cli.gamma,tail_weight=cli.gamma**horizon,
            policy_logit_pg={key:replicate_summary([r[field] for r in result]) for key,field in (
                ('finite_raw','finite_raw_pg'),('finite_normalized','finite_normalized_pg'),
                ('bootstrap_normalized','bootstrap_normalized_pg'))},
            observed_action=dict(finite_raw_Q=replicate_summary([r['observed_finite_raw_Q'] for r in result]),
                finite_normalized_Q=replicate_summary([r['observed_finite_normalized_Q'] for r in result]),
                bootstrap_normalized_Q=replicate_summary([r['observed_bootstrap_normalized_Q'] for r in result]),
                critic_tail=replicate_summary([r['observed_critic_tail'] for r in result]),
                independent_advantage_finite_normalized=replicate_summary(finite),
                independent_advantage_bootstrap_normalized=replicate_summary(bootstrap),
                scoreweighted_error_finite_normalized=replicate_summary(score[None]*(adv[None,:,None]-finite[:,:,None])),
                scoreweighted_error_bootstrap_normalized=replicate_summary(score[None]*(adv[None,:,None]-bootstrap[:,:,None]))),
            reward_clip_counts=[r['reward_clip_count'].tolist() for r in result])
        h['parameter_gradient_comparison']=compare_gradients(agent,actor_fn,states,score,h,stored_gradient,full_gradient)
        horizons.append(h)
    # Original-trajectory scale drift is descriptive, never substituted onto forks.
    dense=snapshot['transitions'];time_count=array(dense['episode_age']).shape[0]
    drift={}
    for horizon in cli.horizons:
        obs_changes=[];reward_ratios=[]
        for s in states:
            end=min(s['t']+horizon,time_count-1)
            reward_ratios.append(float(array(dense['reward_std'])[end,s['row']]/s['reward_std']))
            obs_changes.append(float(np.sqrt(np.mean((array(dense['obs_means'])[end,s['row']]-s['obs_means'])**2))))
        drift[str(horizon)]=dict(original_trajectory_reward_std_ratio=summary(reward_ratios),
            original_trajectory_obs_mean_drift_rms=summary(obs_changes),
            available_future_steps=[min(horizon,time_count-1-s['t']) for s in states])
    return dict(snapshot=str(Path(path).resolve()),snapshot_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        global_step=snapshot['global_step'],selected_states=[{k:v for k,v in s.items() if k!='integration_state'} for s in states],
        initial_policy=dict(logits=z.tolist(),alpha=a.tolist(),beta=b.tolist(),observed_logit_score=score.tolist()),
        dense_boundary_replay=boundaries,physical_restore_max_abs_error=max(restore_errors),
        observed_first_transition_replay=replay_errors,
        first_action_sampling=sampling_protocol,analytic_score_pg_replay_relative_error=score_replay_relative_error,
        oracle=dict(horizons=horizons,continuation_replicates=cli.continuation_replicates,pg_samples=cli.pg_samples,
            normalization='Exact stored first normalized obs; subsequent obs use per-selected-state frozen recorded means/variances. Rewards use recorded applied reward_std, clip 10.',
            first_action_only=True),normalization_drift=drift,
        later_epoch_clipping=clipping_sensitivity(path,snapshot,states,horizons,agent,policy_fn),
        limitations=['No true value or global optimum oracle: finite discounted physical returns; critic tails are estimated and separated.',
            'Continuation policy fixed; first-action PG is conditional at selected training states.',
            'Continuation branches share inverse-CDF noise within each state; independent replicates quantify continuation sensitivity.',
            'Independent action draws differ across replicates, so replicate policy-PG uncertainty combines action and continuation noise.',
            'Frozen selected-state normalizers differ from online training continuation; original scale drift reported descriptively.',
            'Complete integration state restored; historical simulator RNG not saved. HalfCheetah dynamics are deterministic, and no resets/boundaries allowed during forks.',
            'Observed-action Q forks compare the same sampled action to saved GAE; subtract current estimated V, not independently known V*.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',action='append',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--states',type=int,default=32)
    parser.add_argument('--max-age',type=int,default=400)
    parser.add_argument('--pg-samples',type=int,default=16)
    parser.add_argument('--horizons',type=int,nargs='+',default=[1,16,64,256,512])
    parser.add_argument('--continuation-replicates',type=int,default=2)
    parser.add_argument('--gamma',type=float,default=.99)
    parser.add_argument('--seed',type=int,default=1)
    cli=parser.parse_args()
    if cli.states<2 or cli.pg_samples<2 or cli.continuation_replicates<2:
        parser.error('Need >=2 states, independent action samples and continuation replicates')
    if cli.max_age<0 or min(cli.horizons)<1 or cli.max_age+max(cli.horizons)>=1000:
        parser.error('Require nonnegative max age and positive horizons staying below 1000-step TimeLimit')
    if len(set(cli.horizons))!=len(cli.horizons):
        parser.error('Duplicate horizons')
    configure_runtime(matmul_precision='highest',allow_tf32=False)
    # Multiple saved policies and Jacobian closures intentionally compile here.
    # Retain fullgraph behavior rather than exhausting the default eight guards.
    torch._dynamo.config.recompile_limit = 64
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; submit through mlq')
    output=dict(protocol=vars(cli),results=[],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for path in cli.snapshot:
        output['results'].append(analyze(path,cli))
        target=Path(cli.out);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(snapshot=path,states=cli.states,horizons=cli.horizons,status='complete')),flush=True)


if __name__=='__main__':
    main()
