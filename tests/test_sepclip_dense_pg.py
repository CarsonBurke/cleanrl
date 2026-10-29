"""Dense-state oracle contracts; run through mlq on CUDA, never train here."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_sepclip_transition_microscopy_v12 as trainer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

spec=importlib.util.spec_from_file_location('dense_pg',Path(__file__).parents[1]/'scripts/audit_sepclip_dense_pg.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture(scope='module')
def captured():
    configure_runtime(matmul_precision='highest',allow_tf32=False)
    assert torch.cuda.is_available(), 'Queue CUDA contracts through mlq'
    envs=trainer.make_mujoco_vector_env('HalfCheetah-v4',2,num_threads=2)
    try:
        raw,_=envs.reset(seed=1)
        norm=VectorObsNorm(2,(17,))
        rew=VectorRewardNorm(2,.99)
        obs=norm.normalize(raw)
        torch.manual_seed(1)
        agent=trainer.Agent(envs).cuda().eval()
        agent.normalize_matrices()
        policy,value,actor=audit.compiled_functions(agent)
        steps=1001
        dense=trainer.TransitionCapture(envs,steps)
        observations=[];rewards=[];terms=[];truncs=[];natives=[]
        rng=np.random.default_rng(13)
        for t in range(steps):
            observations.append(obs.copy())
            native=rng.uniform(.2,.8,(2,6)).astype(np.float32)
            action=(2*native-1).astype(np.float32)
            dense.before(t,envs,action,norm)
            raw,reward,term,trunc,infos=envs.step(action)
            transformed=rew.normalize(reward,term)
            obs,transition=norm.normalize_step(raw,term,trunc,infos)
            dense.after(t,raw,reward,obs,transition,rew)
            rewards.append(transformed.copy());terms.append(term.copy());truncs.append(trunc.copy());natives.append(native)
        x=torch.as_tensor(np.asarray(observations).reshape(-1,17),device='cuda')
        native_tensor=torch.as_tensor(np.asarray(natives).reshape(-1,6),device='cuda')
        r=torch.as_tensor(np.asarray(rewards),device='cuda')
        term=torch.as_tensor(np.asarray(terms),dtype=torch.float32,device='cuda')
        trunc=torch.as_tensor(np.asarray(truncs),dtype=torch.float32,device='cuda')
        with torch.no_grad():
            def rollout_logprob(x,y):
                z=agent.actor(x)
                a,b=(1+torch.nn.functional.softplus(z)).chunk(2,-1)
                return agent.action_logprob(a,b,y)
            lp=torch.compile(rollout_logprob,fullgraph=True,options={'triton.cudagraphs':False})(x,native_tensor).clone()
            values=value(x).reshape(steps,2)
            transition_values=value(torch.as_tensor(dense.arrays['normalized_transition_obs'].reshape(-1,17),device='cuda')).reshape(steps,2)
            tv=torch.where(trunc.bool(),transition_values,0)
            tail=value(torch.as_tensor(obs,device='cuda')).flatten()
            advantage,returns=trainer.get_gae_fn(compiled=True,mode='reduce-overhead')(r,values,term,trunc,tv,tail,.99,.95)
            # get_gae_fn owns reusable CUDA-graph output storage, as in training.
            advantage=advantage.clone()
            returns=returns.clone()
        snapshot=dict(model=agent.state_dict(),args=dict(env_id='HalfCheetah-v4',gamma=.99,gae_lambda=.95),
            global_step=0,batch=dict(obs=x,native=native_tensor,logprob=lp,
                advantages=advantage.flatten(),returns=returns.flatten(),values=values.flatten()),
            transitions=dict(rewards=r,terminations=term,truncations=trunc,truncation_values=tv,tail_value=tail,**dense.arrays))
        yield snapshot,agent,policy,value,actor
    finally:
        envs.close()


def test_dense_replay_uses_truncated_final_not_autoreset(captured):
    snapshot,_,_,value,_=captured
    result=audit.dense_boundary_audit(snapshot,value,.99,.95)
    assert result['truncations']==2
    assert result['terminations']==0
    assert result['gae_replay_max_abs_error']==0
    assert result['truncated_final_vs_reset_observation_rms']>0
    # Wrong reset observation cannot reproduce saved truncation bootstrap.
    broken=dict(snapshot,transitions=dict(snapshot['transitions']))
    broken['transitions']['normalized_transition_obs']=snapshot['transitions']['normalized_next_obs']
    with pytest.raises(ValueError,match='GAE/bootstrap mismatch'):
        audit.dense_boundary_audit(broken,value,.99,.95)


def test_exact_selected_observed_action_replays_original_transition(captured):
    snapshot,agent,policy,value,_=captured
    states=audit.select_states(snapshot,count=2,max_age=400)
    assert len({(s['t'],s['row']) for s in states})==2
    assert {s['row'] for s in states}=={0,1}
    assert max(s['normalization_replay_max_abs_error'] for s in states)<=1e-6
    cli=SimpleNamespace(pg_samples=4,seed=1,gamma=.99,horizons=[1,16])
    result,score,_,_,_,restore,replay,sampling=audit.fork_replicate(agent,snapshot,states,cli,policy,value,0)
    assert restore==0
    assert replay['raw_reward_max_abs_error']==0
    assert replay['raw_next_obs_max_abs_error']==0
    for h in cli.horizons:
        assert np.isfinite(result[h]['finite_normalized_pg']).all()
        assert result[h]['finite_normalized_pg'].shape==(2,12)
    assert score.shape==(2,12)
    assert sampling['theoretical_clip_tail_mass_summary']['min']>=0
    expected=np.asarray([s['stored_raw_reward']/s['reward_std'] for s in states])
    np.testing.assert_allclose(result[1]['observed_finite_normalized_Q'],np.clip(expected,-10,10),atol=0,rtol=0)


def test_selector_rejects_mismatched_observation_metadata(captured):
    snapshot,*_=captured
    broken=dict(snapshot,transitions=dict(snapshot['transitions']))
    broken['transitions']['obs_means']=snapshot['transitions']['obs_means']+1
    with pytest.raises(ValueError,match='normalization replay error'):
        audit.select_states(broken,count=2)


def test_clipping_trace_uses_learner_fp32_actions_and_logprobs(captured,tmp_path):
    snapshot,agent,policy,_,_=captured
    states=audit.select_states(snapshot,count=2)
    deltas={name:torch.zeros_like(p).cpu() for name,p in agent.named_parameters()}
    trace=dict(trace=[dict(epoch=0,update=0,retracted_delta=deltas)],
        model_after_update={name:p.detach().cpu().clone() for name,p in agent.state_dict().items()})
    torch.save(trace,tmp_path/'diagnostic_updates_0.pt')
    horizons=[dict(horizon=16,observed_action=dict(independent_advantage_bootstrap_normalized=
        dict(mean=[-s['stored_advantage'] for s in states])))]
    result=audit.clipping_sensitivity(tmp_path/'diagnostic_snapshot_0.pt',snapshot,states,horizons,agent,policy)
    assert result['native_action_dtype']=='float32'
    assert result['logprob_dtype']=='torch.float32'
    assert result['old_logprob_source'].startswith('saved full-rollout')
    assert result['updates'][0]['active_fraction']==1
    assert result['final_delta_reconstruction_max_abs_error']==0


def test_loo_score_estimator_has_no_self_baseline_shrinkage():
    # Enumerate all equally likely pairs from Bernoulli(.5), reward=action.
    # The derivative with respect to its mean logit is exactly .25.
    actions=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    scores=(actions-.5)[...,None]
    estimate=audit.conditional_pg(scores,actions)
    np.testing.assert_allclose(estimate.mean(),.25,atol=0,rtol=0)
    naive=(scores*(actions-actions.mean(1,keepdims=True))[...,None]).mean()
    assert naive==.125


@pytest.mark.parametrize('name', ['logits','alpha','beta','native','raw_reward',
    'raw_next_obs','normalized_obs','logit_score','old_critic_value'])
@pytest.mark.parametrize('bad', [float('nan'),float('inf'),-float('inf')])
def test_required_finite_guard_rejects_any_bad_coordinate(name,bad):
    value=np.zeros((2,3),dtype=np.float64)
    value[1,2]=bad
    with pytest.raises(ValueError,match=f'contract: nonfinite {name}'):
        audit.require_finite('contract', **{name:value})
    audit.require_finite('contract', **{name:np.zeros_like(value)})


@pytest.mark.parametrize('name', ['raw_rewards','raw_next_obs','physical_actions'])
def test_selector_rejects_nonfinite_saved_physical_replay_inputs(captured,name):
    snapshot,*_=captured
    broken=dict(snapshot,transitions=dict(snapshot['transitions']))
    broken['transitions'][name]=np.full_like(audit.array(snapshot['transitions'][name]),np.nan)
    with pytest.raises(ValueError,match='Selected state .*nonfinite'):
        audit.select_states(broken,count=2)


def test_dense_boundary_rejects_bad_historical_inputs_before_critic(captured):
    snapshot,*_=captured
    broken=dict(snapshot,batch=dict(snapshot['batch']))
    broken['batch']['advantages']=np.full_like(audit.array(snapshot['batch']['advantages']),np.nan)
    def never_called(_):
        raise AssertionError('Invalid stored credit must fail before critic replay')
    with pytest.raises(ValueError,match='Dense historical GAE inputs: nonfinite stored_advantages'):
        audit.dense_boundary_audit(broken,never_called,.99,.95)


def test_frozen_normalizer_rejects_negative_variance_and_preclip_overflow():
    states=[dict(obs_means=[0.],obs_variances=[-1.])]
    with pytest.raises(ValueError,match='negative observation variance'):
        audit.normalize_frozen(np.ones((1,1)),states,1)
    states=[dict(obs_means=[0.],obs_variances=[0.])]
    with np.errstate(over='ignore'):
        with pytest.raises(ValueError,match='Frozen pre-clip normalization: nonfinite normalized_obs'):
            audit.normalize_frozen(np.array([[1e308]]),states,1)


def test_historical_reward_replay_rejects_overflow_before_clip(captured):
    snapshot,_,_,value,_=captured
    broken=dict(snapshot,transitions=dict(snapshot['transitions']))
    broken['transitions']['raw_rewards']=np.full_like(audit.array(snapshot['transitions']['raw_rewards']),1e308)
    broken['transitions']['reward_std']=np.full_like(audit.array(snapshot['transitions']['reward_std']),1e-300)
    with np.errstate(over='ignore'):
        with pytest.raises(ValueError,match='Dense historical pre-clip reward replay: nonfinite normalized_reward'):
            audit.dense_boundary_audit(broken,value,.99,.95)
