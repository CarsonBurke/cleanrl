"""Microscopic critic/credit audit. Run all CUDA work through mlq.

A GAE target residual is NOT an independent value error. Historical snapshots
omit transition rewards/boundaries/simulator states; this report refuses to
infer missing TD residuals or call bootstrapped lambda=1 targets unbiased MC.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cleanrl import ppo_continuous_action_sepclip_rollout_clock_anvil_v7 as trainer


def feature_stages(critic, observations):
    first_raw = critic.first(observations)
    first = F.normalize(first_raw, dim=-1)
    branch = F.normalize(critic.second(8 * first), dim=-1)
    final = 8 * F.normalize(first + branch, dim=-1)
    return first_raw, first, final


def spectrum_summary(matrix):
    """Gradient or feature energy participation rank, not numerical matrix rank."""
    singular = torch.linalg.svdvals(matrix.float())
    energy = singular.square()
    total = energy.sum()
    denominator = torch.where(total > 0, total, torch.ones_like(total))
    probabilities = energy / denominator
    participation = torch.where(total > 0, denominator.square() / energy.square().sum().clamp_min(1e-30), total)
    return dict(energy=total, participation_rank=participation,
                top1_energy=probabilities[0], top5_energy=probabilities[:5].sum())


def scalarize(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: scalarize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scalarize(v) for v in value]
    return value


def oracle_protocol():
    return dict(
        required_payload=['raw observations at t and transition t+1', 'raw and normalized rewards',
            'terminations and truncations', 'episode age and environment slot',
            'full MuJoCo simulator state, including hidden qpos position and time',
            'per-step normalization statistics or a frozen diagnostic transform',
            'native actions and old alpha/beta', 'policy, critic, RNG and normalizer checkpoints'],
        independent_action_credit=dict(
            procedure='Restore the same simulator state for multiple policy-distributed first actions. '
                'Follow the frozen policy with paired base uniforms transformed by the Beta inverse CDF. '
                'Same RNG seeds alone do not guarantee paired rejection-sampler draws. '
                'Accumulate actual discounted physical rewards; report finite horizons separately. '
                'At H=512, gamma=.99 leaves gamma^H about .0058. Residual tail remains a limitation.',
            transport='Compare score-weighted successor critic values with score-weighted independent '
                'continuation returns after the sampled first action. For fresh, unclipped score-policy '
                'gradients, the current-state value baseline cancels in expectation; successor value '
                'error generally does not. This cancellation does not establish invariance of later '
                'clipped PPO gradients: their active mask depends on the likelihood ratio and '
                'advantage sign, so state-baseline errors can change which samples contribute. '
                'Measure those later-epoch gradients and masks separately.',
            layers='Project both estimated policy gradients through every actor matrix layer, its '
                'unit-sphere tangent, and the actual optimizer update. Measure sign, cosine, and '
                'predicted policy-Fisher efficiency per layer; paired trajectory intervals are '
                'sampling uncertainty, not independent training seeds.'),
        gae_error_identity='Away from boundaries, V=V*+e implies advantage error '
            '-e(s_t)+gamma*(1-lambda)*sum_k (gamma*lambda)^k e(s_{t+k+1}). '
            'Finite trajectories add explicit tail/boundary terms.',
        lambda_counterfactual='Only when transition payload is available: recompute lambda=.8/.95/1 '
            'using the same frozen values and correct truncation bootstraps; compare actor gradient '
            'and optimizer tangent alignment with independent finite-horizon action-credit estimates. '
            'This is a diagnostic of credit bias, not a proposal for a generic hyperparameter sweep.')


def audit(path, writer):
    snapshot = torch.load(path, map_location='cuda', weights_only=False)
    state, batch = snapshot['model'], snapshot['batch']
    observation_dim = state['critic.first.0.gate.weight'].shape[1]
    low = state['action_low'].detach().cpu().numpy()
    high = state['action_high'].detach().cpu().numpy()
    envs = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (observation_dim,), dtype=np.float32),
                           single_action_space=gym.spaces.Box(low, high, dtype=np.float32))
    agent = trainer.Agent(envs).cuda()
    agent.load_state_dict(state)
    obs, native = batch['obs'], batch['native']
    # A time-major stride divisible by num_envs would select only slot zero.
    # An independent deterministic random subset samples across all slots/time.
    generator = torch.Generator(device='cuda').manual_seed(91039)
    indices = torch.randperm(obs.shape[0], device='cuda', generator=generator)[:256]
    selected = obs[indices].contiguous()
    num_envs = snapshot['args']['num_envs']
    env_slot_counts = torch.bincount(indices.remainder(num_envs), minlength=num_envs)
    features = torch.compile(lambda x: feature_stages(agent.critic, x), fullgraph=True,
                             dynamic=False, options={'triton.cudagraphs': False})
    with torch.no_grad():
        stages = features(obs)
        predicted = agent.critic(obs).flatten()
        values, targets, advantages = batch['values'].flatten(), batch['returns'].flatten(), batch['advantages'].flatten()
        metadata = dict(value_mean=predicted.mean(), value_std=predicted.std(),
            frozen_target_mean=targets.mean(), frozen_target_std=targets.std(),
            gae_advantage_mean=advantages.mean(), gae_advantage_std=advantages.std(),
            preupdate_prediction_replay_rmse=(predicted-values).square().mean().sqrt(),
            frozen_lambda_target_residual_rmse=(targets-predicted).square().mean().sqrt(),
            frozen_lambda_target_explained_variance=1-(targets-predicted).var()/targets.var().clamp_min(1e-30),
            gain=agent.critic.readout_gain, bias=agent.critic.head[0].bias,
            affine_bias=agent.critic.readout_gain*agent.critic.head[0].bias,
            feature_component_mean=predicted.mean()-(agent.critic.readout_gain*agent.critic.head[0].bias).sum())
        feature_rank = {name: spectrum_summary(x-x.mean(0))
                        for name,x in zip(('first_raw','first_unit','final'), stages)}
        parity=[]
        for radius in (.01, .1, 1.):
            positive = features(radius * selected)
            negative = features(-radius * selected)
            parity.append(dict(observation_scale=radius, stages={name: dict(
                odd_energy_fraction=((pos-neg)*.5).square().sum() /
                    ((pos.square().sum()+neg.square().sum())*.5).clamp_min(1e-30))
                for name,pos,neg in zip(('first_raw','first_unit','final'), positive,negative)}))

    def one_features(x):
        return feature_stages(agent.critic,x.unsqueeze(0))[1][0]
    jacobian_fn = torch.compile(torch.func.vmap(torch.func.jacrev(one_features)),
                               fullgraph=True, dynamic=False, options={'triton.cudagraphs': False})
    value_jacobian = torch.compile(torch.func.vmap(torch.func.jacrev(lambda x: agent.critic(x.unsqueeze(0)).sum())),
                                  fullgraph=True, dynamic=False, options={'triton.cudagraphs': False})
    jacobian = jacobian_fn(selected)
    input_gradient = value_jacobian(selected)
    direction = F.normalize(selected, dim=-1)
    radial = torch.einsum('nhi,ni->nh',jacobian,direction)
    tangent = jacobian-radial[:,:,None]*direction[:,None,:]
    input_metrics = dict(first_unit_radial_energy=radial.square().mean(),
        first_unit_tangent_energy=tangent.square().sum(-1).mean(),
        value_radial_sensitivity=(input_gradient*direction).sum(-1).square().mean().sqrt(),
        value_input_gradient_rms=input_gradient.square().mean().sqrt(),
        exact_zero_first_feature_jacobian=jacobian_fn(torch.zeros_like(selected[:1])).abs().max())
    if snapshot['args'].get('env_id')=='HalfCheetah-v4' and observation_dim==17:
        input_metrics['velocity_gradient_rms'] = input_gradient[:,8:].square().mean().sqrt()
        input_metrics['position_gradient_rms'] = input_gradient[:,:8].square().mean().sqrt()
    params=dict(agent.critic.named_parameters())
    parameter_jacobian = torch.compile(torch.func.vmap(torch.func.jacrev(
        lambda parameters,x: torch.func.functional_call(agent.critic,parameters,(x.unsqueeze(0),)).sum(),argnums=0),
        in_dims=(None,0)),fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    per_state_grads=parameter_jacobian(params,selected)
    layer_gradients={name:spectrum_summary(gradient.flatten(1)) for name,gradient in per_state_grads.items()
                     if name.endswith('weight') or name in {'readout_gain','head.0.bias'}}

    def actor_signal(observations,actions,weights):
        alpha,beta,_=agent.get_policy_and_value(observations)
        return -(weights*agent.action_logprob(alpha,beta,actions)).mean()
    signal_loss=torch.compile(actor_signal,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    actor_params=[(name,p) for name,p in agent.actor.named_parameters()]
    def gradient_for(weights):
        return torch.autograd.grad(signal_loss(obs,native,weights.detach()),[p for _,p in actor_params])
    true_gae_signal=gradient_for(advantages)
    # At this unchanged snapshot policy, the fresh unclipped score term for an
    # action-independent baseline is expectation-zero. Its finite-sample
    # alignment diagnoses estimator noise, not true V error. Later clipped PPO
    # gradients have ratio/advantage-dependent masks and lack this invariance.
    centered_baseline=gradient_for(predicted.detach()-predicted.detach().mean())
    signal_comparison={}
    for (name,_),g,baseline in zip(actor_params,true_gae_signal,centered_baseline):
        signal_comparison[name]=dict(gae_gradient_norm=g.norm(),
            expectation_zero_baseline_sample_norm=baseline.norm(),
            baseline_sample_to_gae_ratio=baseline.norm()/g.norm().clamp_min(1e-30),
            baseline_sample_cosine=(baseline*g).sum()/(baseline.norm()*g.norm()).clamp_min(1e-30))
    result=scalarize(dict(snapshot=str(path),step=snapshot['global_step'],critic_metadata=metadata,
        feature_spectra=feature_rank,feature_parity=parity,input_jacobian=input_metrics,
        selected_env_slot_counts=env_slot_counts,
        per_state_value_gradient_spectra=layer_gradients,finite_sample_baseline_score=signal_comparison,
        identifiability=dict(true_value_error='not_identifiable',score_weighted_successor_error='not_identifiable',
            transition_TD_residual='not_identifiable',lambda_sweep='not_identifiable',
            independent_finite_horizon_returns='not_identifiable',
            missing_fields=['transition rewards','termination/truncation masks','successor observations','simulator states'],
            reason='Stored GAE advantages and values alone do not identify transition TD residuals without boundaries. '
                'Lambda targets contain this critic bootstrap; their residuals are not independent value errors.'),
        assumptions=['Input and parameter Jacobians use up to 256 randomly selected stored states, not independent samples.',
            'Negated/scaled centered observations are algebraic representation probes, not simulator-valid counterfactuals.',
            'First SiTU is only asymptotically even near zero; real visited-state oddness is measured separately.',
            'Rank participation measures sampled value-gradient energy, not representational expressiveness bounds.',
            'Fresh unclipped baseline-score correlation is finite-sample noise; it cannot establish action-dependent bootstrap bias.',
            'Current-state baseline cancellation holds for fresh unclipped score-policy gradients. Later clipped PPO active masks depend on ratio and advantage sign; their baseline sensitivity must be measured separately.']))
    for name,value in result['critic_metadata'].items():
        if isinstance(value,(float,int)):
            writer.add_scalar('critic/'+name,value,result['step'])
    for name,value in result['input_jacobian'].items():
        writer.add_scalar('jacobian/'+name,value,result['step'])
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',type=Path,action='append',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    trainer.configure_runtime(matmul_precision='highest',allow_tf32=False)
    torch.manual_seed(1)
    torch._dynamo.config.recompile_limit=64
    report=dict(oracle_protocol=oracle_protocol(),snapshots=[])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with SummaryWriter(f'runs/HalfCheetah-v4__sepclip_critic_transport__1__{int(time.time())}') as writer:
        for path in args.snapshot:
            report['snapshots'].append(audit(path,writer))
            args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(out=str(args.out),snapshots=len(report['snapshots']))))


if __name__=='__main__':main()
