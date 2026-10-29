"""Actual v14/v15 update effects in matched physics; execute only through mlq.

The strict prescription loader owns actor/Args dispatch and saved-bank identity.
Only restoration, frozen normalization, and conditional summaries are borrowed
from older audits. Seven arms measure finite feedback interaction, not a PG or
critic accuracy certificate. No parameter projection or optimizer replay occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import betaincinv
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_sepclip_dense_pg as dense
from audit_sepclip_closed_loop_update import restore, conditional_summary
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.runtime import configure_runtime

BANKS = ('old', 'first', 'endpoint')
ARMS = ('old_all', 'first_all', 'endpoint_all', 'first_t0_only',
        'first_after_t0', 'endpoint_t0_only', 'endpoint_after_t0')


def arm_indices(step):
    return np.asarray([0, 1, 2, 1 if step == 0 else 0,
                       0 if step == 0 else 1, 2 if step == 0 else 0,
                       0 if step == 0 else 2], dtype=np.int64)


def policy_on_arms(replay, obs, step):
    """Every future observation goes through its actual source-dispatched actor."""
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim != 2 or len(obs) % len(ARMS):
        raise ValueError('Expected selected-state-major seven-arm observations')
    dense.require_finite('Arm policy inputs', observation=obs)
    indices = np.tile(arm_indices(step), len(obs) // len(ARMS))
    outputs = [None, None, None]
    for index, name in enumerate(BANKS):
        rows = np.flatnonzero(indices == index)
        with torch.no_grad():
            result = replay.compiled_forward(replay.banks[name],
                torch.as_tensor(obs[rows], device='cuda'))
        for j, tensor in enumerate(result):
            values = dense.array(tensor).copy()  # never retain permanent buffers
            if outputs[j] is None:
                outputs[j] = np.empty((len(obs), values.shape[-1]), dtype=values.dtype)
            outputs[j][rows] = values
    z, alpha, beta = outputs
    dense.require_finite('Actual arm policy', logits=z, alpha=alpha, beta=beta)
    if np.any(alpha <= 0) or np.any(beta <= 0):
        raise ValueError('Beta shapes must be positive')
    if alpha.shape != beta.shape or z.shape != (len(obs), 2 * alpha.shape[-1]):
        raise ValueError('Actual actor output shape mismatch')
    return z, alpha, beta


def native_draw(alpha, beta, mode, uniforms):
    """FP32 physical policy; mode uses actual rounded shapes, including shape=1."""
    dense.require_finite('Beta draw inputs', alpha=alpha, beta=beta, uniforms=uniforms)
    if np.any(alpha <= 0) or np.any(beta <= 0):
        raise ValueError('Beta shapes must be positive')
    if mode == 'sample':
        if np.any(uniforms <= 0) or np.any(uniforms >= 1):
            raise ValueError('Inverse-CDF quantiles must be strictly interior')
        action = betaincinv(alpha.astype(np.float64), beta.astype(np.float64), uniforms)
        dense.require_finite('Beta inverse CDF before sampler clipping', native=action)
        action = np.clip(action, 1e-6, 1-1e-6)
    elif mode == 'mean':
        total = alpha + beta
        dense.require_finite('Beta mean denominator', total=total)
        action = alpha / total
    elif mode == 'mode':
        if np.any(alpha < 1) or np.any(beta < 1):
            raise ValueError('This mode contract requires source shapes >=1')
        a, b = alpha - 1, beta - 1
        total = a + b
        dense.require_finite('Beta mode denominator', total=total)
        action = np.divide(a, total, out=np.full_like(a, .5), where=total != 0)
    else:
        raise ValueError(f'Unknown action mode {mode}')
    dense.require_finite('Beta native draw', native=action)
    return action.astype(np.float32)


def frozen_reward(raw, std):
    dense.require_finite('Reward inputs', raw_reward=raw, reward_std=std)
    if np.any(std <= 0):
        raise ValueError('Frozen reward standard deviation must be positive')
    scaled = raw / std
    dense.require_finite('Reward before clipping', scaled_reward=scaled)
    # Match trainer law: clip in FP64, then each reward rounded to FP32.
    return np.clip(scaled, -10, 10).astype(np.float32).astype(np.float64)


def validate_horizons(states, horizons, gamma):
    if not horizons or any(h <= 0 for h in horizons) or not 0 <= gamma <= 1:
        raise ValueError('Require positive horizons and discount in [0,1]')
    if any(s['episode_age'] < 0 or s['episode_age'] + max(horizons) >= 1000 for s in states):
        raise ValueError('Selected state and horizon can cross a reset boundary')


def verify_saved_transition(envs, states):
    """Check the restored integrator with its saved physical action before forks."""
    actions = np.repeat(np.asarray([s['physical_action'] for s in states], dtype=np.float32), len(ARMS), 0)
    expected_obs = np.repeat(np.asarray([s['stored_raw_next_obs'] for s in states]), len(ARMS), 0)
    expected_reward = np.repeat(np.asarray([s['stored_raw_reward'] for s in states]), len(ARMS))
    dense.require_finite('Saved transition inputs', action=actions, next_obs=expected_obs, reward=expected_reward)
    raw, reward, terms, truncs, _ = envs.step(actions)
    dense.require_finite('Saved transition replay', raw_next_obs=raw, raw_reward=reward,
                         termination=terms, truncation=truncs)
    if np.any(terms) or np.any(truncs):
        raise ValueError('Saved transition replay crossed boundary')
    error = max(float(np.max(np.abs(raw-expected_obs))), float(np.max(np.abs(reward-expected_reward))))
    dense.require_finite('Saved transition replay error', error=error)
    if error > 1e-10:
        raise ValueError(f'Saved physical transition replay mismatch {error}')
    return error


def simulate(replay, states, horizons, gamma, mode, replicate, seed=1,
             env_factory=make_mujoco_vector_env):
    validate_horizons(states, horizons, gamma)
    count, arms = len(states), len(ARMS)
    envs = env_factory('HalfCheetah-v4', count * arms, num_threads=2)
    try:
        envs.reset(seed=seed)
        restore_error = restore(envs, states, arms)
        saved_transition_error = verify_saved_transition(envs, states)
        # Restore all integration fields again after consuming the historical check.
        restore_error = max(restore_error, restore(envs, states, arms))
        obs = np.repeat(np.asarray([s['normalized_obs'] for s in states], dtype=np.float32), arms, 0)
        std = np.repeat(np.asarray([s['reward_std'] for s in states], dtype=np.float64), arms)
        dense.require_finite('Initial frozen inputs', observation=obs, reward_std=std)
        if np.any(std <= 0):
            raise ValueError('Frozen reward scale must be positive')
        rng = np.random.default_rng(seed + 100003 * replicate)
        low = dense.array(replay.agent.action_low).copy()
        scale = dense.array(replay.agent.action_scale).copy()
        if low.shape != (replay.agent.action_dim,) or scale.shape != low.shape:
            raise ValueError('Actual agent action bounds have invalid shape')
        dense.require_finite('Actual action bounds', low=low, scale=scale)
        if np.any(scale <= 0):
            raise ValueError('Action range must be positive')
        value_fn = replay.compiled_old_value
        sums = {key: np.zeros((count, arms)) for key in
                ('raw_discounted', 'raw_undiscounted', 'normalized_discounted',
                 'normalized_undiscounted', 'velocity_discounted', 'control_cost_discounted')}
        clip_counts = np.zeros((count, arms), dtype=np.int64)
        action_atoms = np.zeros((count, arms), dtype=np.int64)
        shape_one = np.zeros((count, arms), dtype=np.int64)
        concentration_sum = np.zeros((count, arms))
        outputs = {}
        raw_trace, norm_trace = [], []
        cost_weight = float(envs.envs[0].unwrapped._ctrl_cost_weight)
        forward_weight = float(envs.envs[0].unwrapped._forward_reward_weight)
        for step in range(max(horizons)):
            _, alpha, beta = policy_on_arms(replay, obs, step)
            quantiles = np.repeat(rng.uniform(1e-7, 1-1e-7,
                size=(count, replay.agent.action_dim)), arms, 0)
            native = native_draw(alpha, beta, mode, quantiles)
            action = (low + scale * native).astype(np.float32)
            dense.require_finite('Physical action', action=action)
            raw, reward, terms, truncs, info = envs.step(action)
            # Native vector returns permanent buffers: consume/copy before next call.
            raw = np.asarray(raw).copy()
            reward = np.asarray(reward, dtype=np.float64).copy()
            dense.require_finite('Physical transition', raw_next_observation=raw, raw_reward=reward,
                                 termination=terms, truncation=truncs)
            if np.any(terms) or np.any(truncs):
                raise ValueError('Physical continuation crossed reset/termination boundary')
            shaped = frozen_reward(reward, std)
            summed_cost = np.square(action).sum(-1)
            cost_dtype = np.asarray(summed_cost[0] * cost_weight).dtype
            cost = summed_cost.astype(cost_dtype) * cost_weight
            velocity = (reward + cost) / forward_weight
            dense.require_finite('Physical decomposition', cost=cost, velocity=velocity)
            required = {name: np.asarray(info[name]).copy() for name in ('x_velocity', 'reward_ctrl', 'reward_run')}
            dense.require_finite('Environment reward decomposition', **required)
            decomposition_error = max(float(np.max(np.abs(velocity-required['x_velocity']))),
                float(np.max(np.abs(cost+required['reward_ctrl']))),
                float(np.max(np.abs(reward-required['reward_run']-required['reward_ctrl']))))
            if decomposition_error > 1e-10:
                raise ValueError(f'Environment reward decomposition mismatch {decomposition_error}')
            r, nr, c, v = (x.reshape(count, arms) for x in (reward, shaped, cost, velocity))
            raw_trace.append(r.copy()); norm_trace.append(nr.copy())
            weight = gamma ** step
            for key, x in (('raw_discounted', weight*r), ('raw_undiscounted', r),
                           ('normalized_discounted', weight*nr), ('normalized_undiscounted', nr),
                           ('velocity_discounted', weight*v), ('control_cost_discounted', weight*c)):
                sums[key] += x
            dense.require_finite('Return accumulation', **sums)
            clip_counts += (np.abs(reward/std) > 10).reshape(count, arms)
            action_atoms += ((native == np.float32(1e-6)) |
                             (native == np.float32(1-1e-6))).sum(-1).reshape(count, arms)
            shape_one += ((alpha == 1) | (beta == 1)).sum(-1).reshape(count, arms)
            concentration = alpha.astype(np.float64) + beta.astype(np.float64)
            dense.require_finite('Actual shape concentration', concentration=concentration)
            concentration_sum += concentration.mean(-1).reshape(count, arms)
            obs = dense.normalize_frozen(raw, states, arms)
            if step + 1 in horizons:
                with torch.no_grad():
                    terminal = dense.array(value_fn(torch.as_tensor(obs, device='cuda'))).copy().reshape(count, arms)
                dense.require_finite('Fixed old critic terminal', value=terminal)
                output = {key: value.copy() for key, value in sums.items()}
                output.update(old_critic_terminal=terminal,
                    normalized_fixed_old_critic_tail=sums['normalized_discounted'] + gamma**(step+1)*terminal,
                    raw_fixed_old_critic_tail=sums['raw_discounted'] + gamma**(step+1)*terminal*std.reshape(count, arms),
                    reward_clip_counts=clip_counts.copy(), native_clip_atom_counts=action_atoms.copy(),
                    shape_one_dimension_counts=shape_one.copy(),
                    concentration_dimension_time_mean=concentration_sum/(step+1))
                dense.require_finite('Horizon outputs', **output)
                outputs[step+1] = output
        return dict(horizons=outputs, restore_max_abs_error=restore_error,
                    saved_transition_max_abs_error=saved_transition_error,
                    raw_reward_trace=np.asarray(raw_trace).tolist(),
                    frozen_normalized_reward_trace=np.asarray(norm_trace).tolist())
    finally:
        envs.close()


def summarize(replicates, horizons, mode):
    result = []
    for horizon in horizons:
        rows = [r['horizons'][horizon] for r in replicates]
        metrics = {}
        for key in rows[0]:
            values = np.stack([r[key] for r in rows])
            metrics[key] = dict(arms={name: conditional_summary(values[:, :, i], mode != 'sample')
                                     for i, name in enumerate(ARMS)})
            if key.endswith('discounted') or key.endswith('critic_tail'):
                metrics[key]['paired_gains'] = {name: conditional_summary(values[:, :, i]-values[:, :, 0], mode != 'sample')
                                               for i, name in enumerate(ARMS) if i}
                metrics[key]['feedback_interaction'] = {
                    name: conditional_summary(values[:, :, full]-values[:, :, first]-values[:, :, later]+values[:, :, 0], mode != 'sample')
                    for name, full, first, later in (('first', 1, 3, 4), ('endpoint', 2, 5, 6))}
        result.append(dict(horizon=horizon, metrics=metrics))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'trace', 'module-name', 'manifest', 'manifest-sha256',
                 'execution', 'execution-sha256', 'exp-name', 'snapshot-sha256', 'trace-sha256', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--states', type=int, default=32)
    parser.add_argument('--max-age', type=int, default=400)
    parser.add_argument('--horizons', type=int, nargs='+', default=[64, 256, 512])
    parser.add_argument('--replicates', type=int, default=4)
    parser.add_argument('--modes', nargs='+', choices=['sample', 'mean', 'mode'], default=['sample', 'mean', 'mode'])
    cli = parser.parse_args()
    if cli.replicates < 2:
        raise ValueError('Sample conditional uncertainty requires >=2 independent noise replicas')
    configure_runtime(matmul_precision='highest', allow_tf32=False)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; submit through mlq')
    from sepclip_prescription_replay import load_replay
    replay = load_replay(cli.snapshot, cli.trace, module_name=cli.module_name,
        manifest_path=cli.manifest, manifest_sha256=cli.manifest_sha256,
        execution_path=cli.execution, execution_sha256=cli.execution_sha256,
        exp_name=cli.exp_name, snapshot_sha256=cli.snapshot_sha256, trace_sha256=cli.trace_sha256)
    states = dense.select_states(replay.snapshot, cli.states, cli.max_age)
    gamma = float(replay.args.gamma)
    validate_horizons(states, cli.horizons, gamma)
    report = dict(validation=replay.validation, discount=gamma, seed=1, arms=ARMS,
        saved_old_logits_available=False,
        first_bank_scope='Old FP32 weights plus saved first retracted delta; original first model not independently saved',
        objective='Actual finite closed-loop returns conditional on selected saved training states',
        tail_scope='Separately reported old fixed critic tail; neither true infinite return nor updated critic',
        mode_scope='Actual rounded-shape mode; both shapes=1 uses midpoint among nonunique uniform modes',
        sampling_scope='Shared inverse-CDF quantiles across arms; independent streams across replicas; diagnostic truncated-quantile law and native clipping, not certified identical finite-precision host sampler/RNG',
        gradient_scope='No score PG, tangent projection, optimizer replay, or causal training conclusion',
        frozen_normalizer_scope='Per-state rows frozen through continuation; clip reward then FP32 rounding',
        states=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()} for s in states],
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       (Path(__file__), ROOT/'scripts/sepclip_prescription_replay.py',
                        ROOT/'scripts/audit_sepclip_physics.py', ROOT/'scripts/audit_sepclip_dense_pg.py',
                        ROOT/'scripts/audit_sepclip_closed_loop_update.py')}, modes={})
    for mode in cli.modes:
        rows = [simulate(replay, states, cli.horizons, gamma, mode, r) for r in range(cli.replicates if mode == 'sample' else 1)]
        report['modes'][mode] = dict(results=summarize(rows, cli.horizons, mode),
            replicas=[dict(restore_max_abs_error=r['restore_max_abs_error'], raw_reward_trace=r['raw_reward_trace'],
                           saved_transition_max_abs_error=r['saved_transition_max_abs_error'],
                           frozen_normalized_reward_trace=r['frozen_normalized_reward_trace']) for r in rows])
    Path(cli.output).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
