"""Paired CUDA-checkpoint physics audit; execute exclusively through mlq.

Closed-loop evaluations are paired by reset seed and normalization row. Forks
restore mjSTATE_INTEGRATION and use common inverse-CDF Beta uniforms. Finite
horizon raw returns are not infinite-horizon action values or an optimum oracle.
"""
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.special import betaincinv, digamma, expit
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.vector_norm import VectorObsNorm

MODULE = 'cleanrl.ppo_continuous_action_sepclip_rollout_clock_anvil_v7'


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    return {'mean': float(x.mean()), 'std': float(x.std()),
            'se': float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else None,
            'min': float(x.min()), 'max': float(x.max())}


def norm_from_snapshot(snapshot, rows):
    norm = VectorObsNorm(len(rows), (snapshot['obs_norm']['means'].shape[-1],))
    for name in ('means', 'variances', 'counts'):
        np.copyto(getattr(norm, name), np.asarray(snapshot['obs_norm'][name])[rows])
    return norm


def normalized(norm, obs, updating=False):
    if updating:
        return norm.normalize(obs).copy()
    return np.clip((obs - norm.means) / np.sqrt(norm.variances + norm.epsilon),
                   -norm.clip, norm.clip).astype(np.float32)


def policy(mirror, obs):
    # Mirror returns permanent storage; immediately materialize both parameters.
    logits = mirror(obs).copy()
    alpha, beta = np.split(1 + np.logaddexp(np.float32(0), logits), 2, axis=-1)
    return alpha, beta, logits


def native_action(alpha, beta, mode, uniforms, logits=None):
    if mode == 'sample':
        return np.clip(betaincinv(alpha, beta, uniforms), 1e-6, 1-1e-6).astype(np.float32)
    if mode == 'mean':
        return (alpha / (alpha + beta)).astype(np.float32)
    if logits is not None:
        a, b = np.split(np.logaddexp(np.float32(0), logits), 2, axis=-1)
        return (a/(a+b)).astype(np.float32)
    denominator = alpha+beta-2
    return np.divide(alpha-1, denominator, out=np.full_like(alpha, .5), where=denominator>0).astype(np.float32)


# MuJoCo 2.3.3 predates mj_getState/mj_setState. These are the complete
# integration inputs in that version's mjdata.h, including externally supplied
# controls/forces and solver warmstarts. Derived kinematics are reconstructed.
INTEGRATION_FIELDS = ('qpos', 'qvel', 'act', 'qacc_warmstart', 'ctrl',
                      'qfrc_applied', 'xfrc_applied', 'mocap_pos', 'mocap_quat',
                      'userdata', 'plugin_state', 'eq_active')


def beta_logit_score(alpha, beta, logits, native):
    """Exact distribution score through alpha,beta=1+softplus(logits)."""
    score_a = np.log(native)-digamma(alpha)+digamma(alpha+beta)
    score_b = np.log1p(-native)-digamma(beta)+digamma(alpha+beta)
    return np.concatenate((score_a, score_b), -1)*expit(logits)


def integration_backend():
    return 'mjSTATE_INTEGRATION' if hasattr(mujoco, 'mj_getState') else 'mjdata_complete_integration_fields'


def integration_state(base):
    if hasattr(mujoco, 'mj_getState'):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(base.model, spec), dtype=np.float64)
        mujoco.mj_getState(base.model, base.data, state, spec)
        return state
    return np.concatenate((np.array([base.data.time]),
                           *(np.asarray(getattr(base.data, name)).reshape(-1).copy()
                             for name in INTEGRATION_FIELDS if hasattr(base.data, name))))


def set_integration_state(base, state):
    if hasattr(mujoco, 'mj_setState'):
        mujoco.mj_setState(base.model, base.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        return
    base.data.time = float(state[0])
    offset = 1
    for name in INTEGRATION_FIELDS:
        if hasattr(base.data, name):
            field = np.asarray(getattr(base.data, name))
            size = field.size
            field[...] = state[offset:offset+size].reshape(field.shape)
            offset += size
    if offset != len(state):
        raise ValueError(f'Integration state size mismatch: expected {offset}, got {len(state)}')


def restore_states(envs, states):
    """Restore physics, RNG and native/wrapper episode clocks at fork boundary."""
    for i, record in enumerate(states):
        base = envs.envs[i].unwrapped
        set_integration_state(base, record['state'])
        mujoco.mj_forward(base.model, base.data)
        # mj_forward recomputes derived arrays but may update warmstart; restore
        # integration fields again so all fork branches start from exact state.
        set_integration_state(base, record['state'])
        base.np_random.bit_generator.state = copy.deepcopy(record['rng'])
        envs._episode_lengths[i] = record['age']
        envs._limits[i]._elapsed_steps = record['age']
        envs._episode_returns[i] = 0
        envs.observations[i] = base._get_obs()
    return envs.observations.copy()


def evaluate(module, agent, snapshot, cli, mode, updating, row_offset=0):
    n = cli.episodes
    envs = make_mujoco_vector_env('HalfCheetah-v4', n, num_threads=2)
    rows = (np.arange(n) + row_offset) % len(snapshot['obs_norm']['counts'])
    norm = norm_from_snapshot(snapshot, rows)
    mirror = module.ResidualHostMirror(agent.actor, n)
    rng = np.random.default_rng(cli.seed)
    raw, _ = envs.reset(seed=cli.seed)
    obs = normalized(norm, raw, updating)
    returns = np.zeros(n)
    metrics = {'velocity': [], 'control_cost': [], 'height': [], 'pitch': [],
               'contacts': [], 'joint_velocity_rms': [], 'action_abs': [],
               'action_change_squared': [], 'concentration': [], 'mean_abs': [], 'mode_abs': []}
    previous = np.zeros((n, agent.action_dim), dtype=np.float32)
    states = []
    for t in range(1000):
        alpha, beta, logits = policy(mirror, obs)
        u = rng.uniform(1e-7, 1-1e-7, size=alpha.shape)
        native = native_action(alpha, beta, mode, u, logits)
        action = (2 * native - 1).astype(np.float32)
        if mode == 'sample' and not updating and row_offset == 0 and t in (250, 500):
            for i in range(min(n, cli.states // 2)):
                base = envs.envs[i].unwrapped
                states.append({'state': integration_state(base), 'rng': copy.deepcopy(base.np_random.bit_generator.state),
                               'age': t, 'row': int(rows[i]), 'obs': raw[i].copy()})
        raw, reward, terms, truncs, infos = envs.step(action)
        returns += reward
        # The step-1000 data have autoreset; velocity uses the final observation
        # indirectly from reward + exact action cost to avoid reset displacement.
        cost = .1 * np.square(action).sum(-1)
        metrics['velocity'].extend((reward + cost).tolist())
        metrics['control_cost'].extend(cost.tolist())
        metrics['action_abs'].extend(np.abs(action).flatten().tolist())
        metrics['action_change_squared'].extend(np.square(action-previous).sum(-1).tolist())
        metrics['concentration'].extend((alpha+beta).flatten().tolist())
        metrics['mean_abs'].extend(np.abs(2*alpha/(alpha+beta)-1).flatten().tolist())
        metrics['mode_abs'].extend(np.abs(2*native_action(alpha, beta, 'mode', u, logits)-1).flatten().tolist())
        if t < 999:
            for e in envs.envs:
                data = e.unwrapped.data
                metrics['height'].append(float(data.qpos[1]))
                metrics['pitch'].append(float(data.qpos[2]))
                metrics['contacts'].append(int(data.ncon))
                metrics['joint_velocity_rms'].append(float(np.sqrt(np.mean(data.qvel[3:]**2))))
        previous = action.copy()
        obs = normalized(norm, raw, updating)
    envs.close()
    return {'mode': mode, 'normalization': 'updating' if updating else 'frozen',
            'row_offset': row_offset, 'normalization_rows': rows.tolist(),
            'returns': returns.tolist(), 'return_summary': summarize(returns),
            'metrics': {k: summarize(v) for k, v in metrics.items()},
            'torque_fraction': {str(c): float(np.mean(np.asarray(metrics['action_abs']) >= c))
                                for c in (.8, .9, .95, .98)}}, states


def fork_audit(module, agent, snapshot, cli, states):
    d, samples = agent.action_dim, cli.pg_samples
    branches = samples + 1 + 2*d
    records = [s for s in states for _ in range(branches)]
    n = len(records)
    envs = make_mujoco_vector_env('HalfCheetah-v4', n, num_threads=2)
    envs.reset(seed=cli.seed)
    raw = restore_states(envs, records)
    restoration_error = max(float(np.max(np.abs(integration_state(e.unwrapped)-r['state'])))
                            for e, r in zip(envs.envs, records))
    observation_error = max(float(np.max(np.abs(raw[i]-r['obs']))) for i, r in enumerate(records))
    if restoration_error != 0 or observation_error > 1e-10:
        raise RuntimeError(f'State restoration mismatch: integration={restoration_error}, observation={observation_error}')
    replay_action = np.zeros((n, d), dtype=np.float32)
    replay_obs, replay_reward, *_ = envs.step(replay_action)
    replay_states = [integration_state(e.unwrapped) for e in envs.envs]
    restore_states(envs, records)
    repeat_obs, repeat_reward, *_ = envs.step(replay_action)
    replay_error = max(float(np.max(np.abs(replay_obs-repeat_obs))), float(np.max(np.abs(replay_reward-repeat_reward))),
                       max(float(np.max(np.abs(integration_state(e.unwrapped)-old))) for e, old in zip(envs.envs, replay_states)))
    if replay_error != 0:
        raise RuntimeError(f'Paired full-state replay is not exact: {replay_error}')
    raw = restore_states(envs, records)
    norm = norm_from_snapshot(snapshot, [s['row'] for s in records])
    mirror = module.ResidualHostMirror(agent.actor, n)
    obs = normalized(norm, raw)
    alpha, beta, logits = policy(mirror, obs)
    initial_normalized_obs = obs.reshape(len(states), branches, -1)[:, 0].copy()
    initial_logits = logits.reshape(len(states), branches, 2*d)[:, 0].copy()
    rng = np.random.default_rng(cli.seed + 1)
    uniforms = rng.uniform(1e-7, 1-1e-7, size=(len(states), samples+1, d))
    initial = np.empty((len(states), branches, d), dtype=np.float32)
    a0 = alpha.reshape(len(states), branches, d)[:, 0]
    b0 = beta.reshape(len(states), branches, d)[:, 0]
    initial[:, :samples+1] = np.clip(betaincinv(a0[:, None], b0[:, None], uniforms), 1e-6, 1-1e-6)
    for dim in range(d):
        for sign_idx, sign in enumerate((-1, 1)):
            branch = samples+1+2*dim+sign_idx
            initial[:, branch] = initial[:, samples]
            initial[:, branch, dim] = np.clip(initial[:, branch, dim]+sign*cli.perturbation, 1e-6, 1-1e-6)
    total = np.zeros(n)
    normalized_total = np.zeros(n)
    reward_scale = 1/np.sqrt(np.asarray(snapshot['rew_norm']['variances'])[[s['row'] for s in states]]+1e-8)
    reward_scale = reward_scale.reshape(len(states), -1)[:, 0]
    fork_reward_scale = np.repeat(reward_scale, branches)
    clipped_rewards = 0
    outputs = []
    horizon_set = set(cli.horizons)
    for t in range(max(cli.horizons)):
        if t == 0:
            native = initial.reshape(n, d)
        else:
            alpha, beta, _ = policy(mirror, obs)
            # Each state's continuation uniforms shared across all action branches.
            u = np.repeat(rng.uniform(1e-7, 1-1e-7, size=(len(states), d)), branches, axis=0)
            native = native_action(alpha, beta, 'sample', u)
        raw, reward, terms, truncs, infos = envs.step((2*native-1).astype(np.float32))
        if np.any(terms | truncs):
            raise RuntimeError('Fork crossed episode boundary: reduce horizon or state age')
        total += cli.gamma**t * reward
        scaled_reward = reward*fork_reward_scale
        clipped_rewards += int(np.sum(np.abs(scaled_reward)>10))
        normalized_total += cli.gamma**t * np.clip(scaled_reward, -10, 10)
        obs = normalized(norm, raw)
        if t+1 in horizon_set:
            q = total.reshape(len(states), branches).copy()
            fd = np.stack([(q[:, samples+2+2*j]-q[:, samples+1+2*j]) /
                           (initial[:, samples+2+2*j, j]-initial[:, samples+1+2*j, j]) for j in range(d)], -1)
            # Conditional policy score estimator at restored states. Centering by
            # leave-one-out action returns avoids the 1-1/N centering shrinkage.
            values = q[:, :samples]
            baseline = (values.sum(-1, keepdims=True)-values)/(samples-1)
            actions = initial[:, :samples].astype(np.float64)
            z = logits.reshape(len(states), branches, 2*d)[:, 0]
            score = beta_logit_score(a0[:, None], b0[:, None], z[:, None], actions)
            contributions = score * (values-baseline)[..., None]
            pg = contributions.mean(1)
            sigmoid_a, sigmoid_b = np.split(expit(z), 2, axis=-1)
            pg_a, pg_b = np.split(pg, 2, axis=-1)
            concentration = a0+b0
            mean = a0/concentration
            # Exact local inverse Jacobian of (z_alpha,z_beta)->(mu,kappa).
            pg_mean = concentration*(pg_a/sigmoid_a-pg_b/sigmoid_b)
            pg_concentration = mean*pg_a/sigmoid_a+(1-mean)*pg_b/sigmoid_b
            with torch.no_grad():
                successor_value = agent.get_value(torch.as_tensor(obs, device='cuda')).flatten().cpu().numpy().reshape(len(states), branches)
            normalized_q = normalized_total.reshape(len(states), branches)[:, :samples]
            normalized_baseline = (normalized_q.sum(-1, keepdims=True)-normalized_q)/(samples-1)
            normalized_pg = (score*(normalized_q-normalized_baseline)[..., None]).mean(1)
            bootstrapped_q = normalized_q+cli.gamma**(t+1)*successor_value[:, :samples]
            bootstrap_baseline = (bootstrapped_q.sum(-1, keepdims=True)-bootstrapped_q)/(samples-1)
            bootstrap_pg = (score*(bootstrapped_q-bootstrap_baseline)[..., None]).mean(1)
            tail_pg = bootstrap_pg-normalized_pg
            parameter_comparison = compare_parameter_pg(agent, snapshot, states, norm, normalized_pg, bootstrap_pg)
            if not all(np.isfinite(x).all() for x in (pg_mean, pg_concentration, pg, bootstrap_pg)):
                raise RuntimeError('Nonfinite physics score gradient')
            outputs.append({'horizon': t+1, 'discount': cli.gamma,
                            'tail_weight': cli.gamma**(t+1), 'returns': q.tolist(),
                            'native_action_gradient': fd.tolist(), 'action_gradient_summary': summarize(fd),
                            'logit_score_pg': pg.tolist(), 'logit_score_pg_summary': summarize(pg),
                            'normalized_reward_logit_score_pg': normalized_pg.tolist(),
                            'normalized_bootstrap_logit_score_pg': bootstrap_pg.tolist(),
                            'critic_tail_logit_score_pg': tail_pg.tolist(),
                            'critic_tail_to_finite_pg_norm': float(np.linalg.norm(tail_pg)/(np.linalg.norm(normalized_pg)+1e-30)),
                            'parameter_gradient_comparison': parameter_comparison,
                            'frozen_reward_scale': reward_scale.tolist(),
                            'normalized_reward_clip_count': clipped_rewards,
                            'fixed_concentration_mean_score_pg': pg_mean.tolist(),
                            'fixed_mean_concentration_score_pg': pg_concentration.tolist(),
                            'state_mean_score_pg': pg.mean(0).tolist(),
                            'state_score_pg_se': (pg.std(0, ddof=1)/np.sqrt(len(states))).tolist(),
                            'within_state_contribution_sd': contributions.std(1, ddof=1).tolist()})
    envs.close()
    return {'integration_backend': integration_backend(), 'mujoco_version': mujoco.__version__,
            'packed_integration_fields': ['time', *INTEGRATION_FIELDS] if integration_backend() != 'mjSTATE_INTEGRATION' else None,
            'full_state_restore_max_abs_error': restoration_error,
            'observation_restore_max_abs_error': observation_error,
            'paired_step_replay_max_abs_error': replay_error,
            'states': [{'age': s['age'], 'normalization_row': s['row'],
                        'raw_observation': s['obs'].tolist(),
                        'normalized_observation': initial_normalized_obs[i].tolist(),
                        'logits': initial_logits[i].tolist(),
                        'alpha': a0[i].tolist(), 'beta': b0[i].tolist(),
                        'observation_mean': np.asarray(snapshot['obs_norm']['means'])[s['row']].tolist(),
                        'observation_variance': np.asarray(snapshot['obs_norm']['variances'])[s['row']].tolist(),
                        'integration_state': s['state'].tolist()} for i, s in enumerate(states)],
            'score_scope': 'First action only, conditional on restored evaluation states. Future actions follow the unchanged stochastic checkpoint policy.',
            'state_occupancy': 'Frozen-row sampled-policy evaluation at episode ages250/500, not the stored training-rollout occupancy.',
            'first_action_logit_score': beta_logit_score(a0[:, None], b0[:, None], initial_logits[:, None], initial[:, :samples].astype(np.float64)).tolist(),
            'samples_per_state': samples, 'branches_per_state': branches,
            'first_native_actions': initial.tolist(), 'horizons': outputs,
            'limitations': ['This replicate shares continuation noise across branches; compare independent replicates for Monte Carlo sensitivity. State standard errors alone understate conditional uncertainty.',
                            'Finite horizon has no value tail; changing horizon can reverse the gradient.',
                            'These are first-action conditional logit derivatives; parameter mapping does not include occupancy derivatives or subsequent policy-action score terms.',
                            'These are local action and logit derivatives, not a global optimal-control oracle.']}


def compare_parameter_pg(agent, snapshot, states, norm, finite_pg, bootstrap_pg):
    """Map conditional logit score estimates through the actual actor Jacobian.

    Comparator uses fresh unclipped snapshot score*GAE, not a clipped later PPO
    minibatch. Evaluation states and rollout states have different distributions.
    """
    parameters = dict(agent.actor.named_parameters())
    obs = np.stack([s['obs'] for s in states])
    # Fork normalizer has repeated rows; explicitly select the saved source row.
    means = np.asarray(snapshot['obs_norm']['means'])[[s['row'] for s in states]]
    variances = np.asarray(snapshot['obs_norm']['variances'])[[s['row'] for s in states]]
    obs = np.clip((obs-means)/np.sqrt(variances+1e-8), -10, 10).astype(np.float32)
    x = torch.as_tensor(obs, device='cuda')
    logits = agent.actor(x)
    finite = torch.autograd.grad(logits, tuple(parameters.values()),
                                 grad_outputs=torch.as_tensor(finite_pg/len(states), dtype=logits.dtype, device='cuda'))
    # Compiled backward donates intermediate buffers; a second independent
    # forward supplies a fresh graph instead of retaining the first graph.
    logits = agent.actor(x)
    bootstrap = torch.autograd.grad(logits, tuple(parameters.values()),
                                    grad_outputs=torch.as_tensor(bootstrap_pg/len(states), dtype=logits.dtype, device='cuda'))
    batch = snapshot['batch']
    # Entire stored rollout, evaluated at the snapshot's pre-update policy.
    rollout_x = batch['obs'].cuda()
    native = batch['native'].cuda()
    advantages = batch['advantages'].cuda()
    alpha, beta, _ = agent.get_policy_and_value(rollout_x)
    objective = (agent.action_logprob(alpha, beta, native)*advantages).mean()
    stored = torch.autograd.grad(objective, tuple(parameters.values()))
    report = {}
    for (name, _), f, b, g in zip(parameters.items(), finite, bootstrap, stored):
        def pairing(v):
            dot = (v*g).sum()
            return {'norm': float(v.norm()), 'pg_pairing': float(dot),
                    'cosine': float(dot/(v.norm()*g.norm()+1e-30)),
                    'projection_on_stored_pg': float(dot/(g.square().sum()+1e-30))}
        report[name] = {'fresh_rollout_pg_norm': float(g.norm()),
                        'finite_horizon': pairing(f), 'critic_bootstrapped': pairing(b)}
    return {'state_distribution_matched': False,
            'comparator': 'fresh unclipped score times stored GAE on entire training rollout',
            'layers': report}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', action='append', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--episodes', type=int, default=16)
    p.add_argument('--states', type=int, default=16)
    p.add_argument('--pg-samples', type=int, default=16)
    p.add_argument('--horizons', type=int, nargs='+', default=[1, 16, 64, 256])
    p.add_argument('--perturbation', type=float, default=.025)
    p.add_argument('--gamma', type=float, default=.99)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--normalization-sensitivity', action='store_true')
    p.add_argument('--continuation-replicates', type=int, default=2)
    cli = p.parse_args()
    if cli.continuation_replicates < 2:
        p.error('Need at least two independent continuation-noise replicates')
    if cli.pg_samples < 2 or cli.states < 2 or cli.states % 2 or cli.episodes < cli.states//2:
        p.error('Need pg-samples >=2 and even states >=2 with episodes >=states/2')
    if not cli.horizons or min(cli.horizons) < 1 or max(cli.horizons) >= 500:
        p.error('horizons must lie in [1,499]')
    configure_runtime(matmul_precision='highest', allow_tf32=False)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch._dynamo.config.recompile_limit = 64
    module = importlib.import_module(MODULE)
    results = []
    for path in cli.snapshot:
        snapshot = torch.load(path, map_location='cpu', weights_only=False)
        if 'obs_norm' not in snapshot:
            raise ValueError('Missing exact checkpoint observation normalization; cannot substitute another run')
        envs = make_mujoco_vector_env('HalfCheetah-v4', cli.episodes, num_threads=2)
        agent = module.Agent(envs).cuda().eval()
        agent.load_state_dict(snapshot['model'])
        # Compile bound forwards, keeping module/parameter names unchanged for
        # host mirrors and independent per-layer gradient comparisons. Disabled
        # CUDA graphs avoid buffer-lifetime constraints across retained graphs.
        agent.actor.forward = torch.compile(agent.actor.forward, fullgraph=True, dynamic=False,
                                            options={'triton.cudagraphs': False})
        agent.critic.forward = torch.compile(agent.critic.forward, fullgraph=True, dynamic=False,
                                             options={'triton.cudagraphs': False})
        envs.close()
        # Real CUDA forward validates the host mirror against checkpoint model.
        probe = snapshot['batch']['obs'][:cli.episodes].cuda()
        with torch.no_grad():
            expected = agent.actor(probe).cpu().numpy()
        mirror = module.ResidualHostMirror(agent.actor, cli.episodes)
        discrepancy = float(np.max(np.abs(mirror(probe.cpu().numpy())-expected)))
        if discrepancy > 1e-4:
            raise RuntimeError(f'Host/CUDA checkpoint mismatch {discrepancy}')
        evaluations = []
        states = None
        for mode in ('sample', 'mean', 'mode'):
            result, captured = evaluate(module, agent, snapshot, cli, mode, False)
            evaluations.append(result)
            if captured:
                states = captured
        if cli.normalization_sensitivity:
            for updating, offset in ((True, 0), (False, 1)):
                result, _ = evaluate(module, agent, snapshot, cli, 'sample', updating, offset)
                evaluations.append(result)
        paired = {e['mode']: np.asarray(e['returns']) for e in evaluations[:3]}
        row = {'snapshot': str(Path(path).resolve()), 'snapshot_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
               'global_step': snapshot['global_step'], 'host_cuda_max_abs_error': discrepancy,
               'evaluations': evaluations,
               'paired_return_differences': {m: summarize(paired[m]-paired['sample']) for m in ('mean', 'mode')},
               'fork_replicates': []}
        for replicate in range(cli.continuation_replicates):
            fork_cli = copy.copy(cli)
            fork_cli.seed = cli.seed+100003*replicate
            row['fork_replicates'].append(fork_audit(module, agent, snapshot, fork_cli, states))
        results.append(row)
        print(json.dumps({'snapshot': path, 'scores': {m: float(v.mean()) for m, v in paired.items()}}), flush=True)
        Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
        Path(cli.out).write_text(json.dumps({'protocol': vars(cli), 'results': results,
                                          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
