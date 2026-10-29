"""Contact-resolved gait traces and held-out local physical action interventions.

Queue through mlq. Candidate discovery and validation use independent stochastic
continuations. A first-action local gain is not a closed-loop controller gain or
a claim about the global optimum; no critic tail is substituted for raw returns.
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_sepclip_physics as physics
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.vector_norm import VectorObsNorm

CONTACT_GEOMS = ('bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot', 'torso', 'head')


def ground_contacts(base):
    floor = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    ids = [mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in CONTACT_GEOMS]
    force = np.zeros(len(ids), dtype=np.float64)
    present = np.zeros(len(ids), dtype=bool)
    contact_force = np.zeros(6)
    for j in range(base.data.ncon):
        contact = base.data.contact[j]
        other = contact.geom2 if contact.geom1 == floor else contact.geom1 if contact.geom2 == floor else -1
        if other in ids:
            k = ids.index(other)
            present[k] = True
            mujoco.mj_contactForce(base.model, base.data, j, contact_force)
            force[k] += abs(contact_force[0])
    return present, force


def contact_cycles(present, dt, rootx=None):
    """Threshold-free geometry-contact onset timing; no named gait assumption."""
    x = np.asarray(present, dtype=bool)
    onsets = np.flatnonzero(x[1:] & ~x[:-1])+1
    offsets = np.flatnonzero(~x[1:] & x[:-1])+1
    complete = []
    for begin, end in zip(onsets[:-1], onsets[1:]):
        stance_end = offsets[(offsets > begin) & (offsets < end)]
        if len(stance_end):
            complete.append({'onset_step': int(begin), 'next_onset_step': int(end),
                             'cycle_seconds': float((end-begin)*dt),
                             'stance_seconds': float((stance_end[0]-begin)*dt),
                             'stance_fraction': float((stance_end[0]-begin)/(end-begin))})
    if rootx is not None:
        for cycle in complete:
            begin, end = cycle['onset_step'], cycle['next_onset_step']
            cycle['delta_rootx'] = float(rootx[end]-rootx[begin])
            cycle['mean_cycle_velocity'] = cycle['delta_rootx']/cycle['cycle_seconds']
    return {'delta_rootx': physics.summarize([c['delta_rootx'] for c in complete]) if complete and rootx is not None else None,
            'mean_cycle_velocity': physics.summarize([c['mean_cycle_velocity'] for c in complete]) if complete and rootx is not None else None,
            'complete_cycles_total_displacement_over_time': sum(c['delta_rootx'] for c in complete)/sum(c['cycle_seconds'] for c in complete) if complete and rootx is not None else None,
            'duty_fraction': float(x.mean()), 'onsets': onsets.tolist(), 'offsets': offsets.tolist(),
            'complete_cycles': complete,
            'cycle_seconds': physics.summarize([c['cycle_seconds'] for c in complete]) if complete else None}


def phase_joint_profiles(qpos, qvel, action, present, dt, bins=20, power=None):
    cycles = contact_cycles(present, dt)['complete_cycles']
    samples = [[] for _ in range(bins)]
    power_samples = [[] for _ in range(bins)]
    for cycle in cycles:
        start, stop = cycle['onset_step'], cycle['next_onset_step']
        for t in range(start, stop):
            bucket = min(bins-1, int(bins*(t-start)/(stop-start)))
            samples[bucket].append(np.concatenate((qpos[t, 3:], qvel[t, 3:], action[t])))
            if power is not None:
                power_samples[bucket].append(power[t])
    means = np.stack([np.mean(v, axis=0) if v else np.full(18, np.nan) for v in samples])
    peak_bins = np.nanargmax(np.abs(means), axis=0).tolist() if np.any(np.isfinite(means)) else None
    return {'commanded_joint_power_mean': [np.mean(v, axis=0).tolist() if v else None for v in power_samples] if power is not None else None,
            'bin_counts': [len(v) for v in samples],
            'joint_angle_velocity_action_mean': [np.mean(v, axis=0).tolist() if v else None for v in samples],
            'peak_absolute_angle_velocity_action_phase_bin': peak_bins,
            'definition': 'Phase0 at this foot geometry ground-contact onset, phase1 at next onset; no assumed gait label.'}


def trace_checkpoint(module, agent, snapshot, cli, mode, artifact):
    n = cli.episodes
    envs = make_mujoco_vector_env('HalfCheetah-v4', n, num_threads=2)
    rows = np.arange(n) % len(snapshot['obs_norm']['counts'])
    norm = physics.norm_from_snapshot(snapshot, rows)
    mirror = module.ResidualHostMirror(agent.actor, n)
    raw, _ = envs.reset(seed=cli.seed)
    obs = physics.normalized(norm, raw)
    rng = np.random.default_rng(cli.seed)
    qpos, qvel = [], []
    contact, contact_force, torso = [], [], []
    action_trace, reward_trace, alpha_trace, beta_trace = [], [], [], []
    applied_force, commanded_torque, command_power = [], [], []
    states = []
    dt = envs.envs[0].unwrapped.dt
    for t in range(1000):
        alpha, beta, logits = physics.policy(mirror, obs)
        uniforms = rng.uniform(1e-7, 1-1e-7, alpha.shape)
        native = physics.native_action(alpha, beta, mode, uniforms, logits)
        action = (2*native-1).astype(np.float32)
        qs = np.stack([e.unwrapped.data.qpos.copy() for e in envs.envs])
        vs = np.stack([e.unwrapped.data.qvel.copy() for e in envs.envs])
        contacts = [ground_contacts(e.unwrapped) for e in envs.envs]
        qpos.append(qs); qvel.append(vs)
        contact.append(np.stack([c[0] for c in contacts])); contact_force.append(np.stack([c[1] for c in contacts]))
        torso.append(np.array([[e.unwrapped.data.xpos[mujoco.mj_name2id(e.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, 'torso'), 2],
                                e.unwrapped.data.qpos[2]] for e in envs.envs]))
        applied_force.append(np.stack([e.unwrapped.data.actuator_force.copy() for e in envs.envs]))
        gears = np.array([e.unwrapped.model.actuator_gear[:, 0] for e in envs.envs])
        torque = action*gears
        commanded_torque.append(torque); command_power.append(torque*vs[:, 3:])
        action_trace.append(action.copy()); alpha_trace.append(alpha.copy()); beta_trace.append(beta.copy())
        if mode == 'sample' and t in (200, 400, 600):
            for i in range(min(n, cli.states//3)):
                base = envs.envs[i].unwrapped
                states.append({'state': physics.integration_state(base), 'rng': copy.deepcopy(base.np_random.bit_generator.state),
                               'age': t, 'row': int(rows[i]), 'obs': raw[i].copy(),
                               'means': norm.means[i].copy(), 'variances': norm.variances[i].copy(),
                               'native': native[i].copy(), 'trace_episode': i, 'source': 'sampled_evaluation',
                               'initial_ground_contact': contacts[i][0].copy(),
                               'initial_ground_normal_force': contacts[i][1].copy(),
                               'initial_qpos': qs[i].copy(), 'initial_qvel': vs[i].copy()})
        raw, reward, terms, truncs, _ = envs.step(action)
        reward_trace.append(reward.copy())
        obs = physics.normalized(norm, raw)
    envs.close()
    arrays = {'qpos': np.asarray(qpos), 'qvel': np.asarray(qvel), 'torso_height_pitch': np.asarray(torso),
              'ground_contact': np.asarray(contact), 'ground_normal_force': np.asarray(contact_force),
              'action': np.asarray(action_trace), 'reward': np.asarray(reward_trace),
              'alpha': np.asarray(alpha_trace), 'beta': np.asarray(beta_trace),
              'actuator_force_pre_step': np.asarray(applied_force),
              'commanded_joint_torque': np.asarray(commanded_torque), 'commanded_joint_power': np.asarray(command_power)}
    arrays['step_velocity'] = arrays['reward']+.1*np.square(arrays['action']).sum(-1)
    arrays['control_cost'] = .1*np.square(arrays['action']).sum(-1)
    np.savez_compressed(artifact, **arrays, dt=dt, normalization_rows=rows)
    episodes = []
    for i in range(n):
        feet = {name: contact_cycles(arrays['ground_contact'][:, i, CONTACT_GEOMS.index(name)], dt, arrays['qpos'][:, i, 0])
                for name in ('bfoot', 'ffoot')}
        profiles = {name: phase_joint_profiles(arrays['qpos'][:, i], arrays['qvel'][:, i], arrays['action'][:, i],
                                               arrays['ground_contact'][:, i, CONTACT_GEOMS.index(name)], dt,
                                               power=arrays['commanded_joint_power'][:, i])
                    for name in feet}
        onset_delays = []
        for cycle in feet['bfoot']['complete_cycles']:
            begin, end = cycle['onset_step'], cycle['next_onset_step']
            front_onsets = [v for v in feet['ffoot']['onsets'] if begin<=v<end]
            onset_delays.extend((v-begin)/(end-begin) for v in front_onsets)
        episodes.append({'front_onset_phase_in_rear_cycles': onset_delays, 'return': float(arrays['reward'][:, i].sum()), 'feet': feet, 'phase_profiles': profiles,
                         'speed': physics.summarize(arrays['step_velocity'][:, i]),
                         'pitch': physics.summarize(arrays['torso_height_pitch'][:, i, 1]),
                         'torso_world_height': physics.summarize(arrays['torso_height_pitch'][:, i, 0]),
                         'control_cost': physics.summarize(arrays['control_cost'][:, i]),
                         'positive_command_power': np.maximum(arrays['commanded_joint_power'][:, i], 0).mean(0).tolist(),
                         'negative_command_power': np.minimum(arrays['commanded_joint_power'][:, i], 0).mean(0).tolist(),
                         'ground_geometry_duty_fraction': arrays['ground_contact'][:, i].mean(0).tolist(),
                         'ground_normal_force_mean': arrays['ground_normal_force'][:, i].mean(0).tolist(),
                         'steady_from_step200_speed': physics.summarize(arrays['step_velocity'][200:, i]),
                         'steady_from_step200_ground_duty_fraction': arrays['ground_contact'][200:, i].mean(0).tolist()})
    return {'mode': mode, 'artifact': str(artifact.resolve()), 'dt': dt,
            'contact_geometry_order': CONTACT_GEOMS, 'episodes': episodes,
            'return_summary': physics.summarize([e['return'] for e in episodes]),
            'sampling': 'Paired inverse-CDF uniforms and reset seeds; frozen original independent normalization rows.',
            'temporal_resolution': 'One sample per environment step (dt=.05s), not all five simulator substeps; forces are instantaneous, not integrated contact impulses.',
            'force_timing': 'Contact and actuator forces are sampled before current action at final solver substep of preceding step; commanded torque uses current action and gear.'}, states


def state_normalizer(states, branches):
    norm = VectorObsNorm(len(states)*branches, (len(states[0]['obs']),))
    np.copyto(norm.means, np.repeat(np.stack([s['means'] for s in states]), branches, axis=0))
    np.copyto(norm.variances, np.repeat(np.stack([s['variances'] for s in states]), branches, axis=0))
    return norm


def candidate_actions(actual, mean, mode, delta):
    actual = np.asarray(actual, dtype=np.float32)
    count, dim = actual.shape
    candidates = [actual, mean, mode]
    labels = ['actual', 'mean', 'mode']
    for j in range(dim):
        for sign in (-1, 1):
            candidate = actual.copy()
            candidate[:, j] += sign*delta
            candidates.append(np.clip(candidate, 1e-6, 1-1e-6))
            labels.append(f'joint{j}_{sign:+d}')
    return np.stack(candidates, 1).astype(np.float32), labels


def discovery_selection(candidates, discovery_q, delta):
    """Use discovery noise only; held-out returns never select a candidate."""
    q = np.asarray(discovery_q).mean(0)
    selected = np.argmax(q, axis=-1)
    dim = candidates.shape[-1]
    slopes = np.stack([(q[:, 4+2*j]-q[:, 3+2*j]) /
                       (candidates[:, 4+2*j, j]-candidates[:, 3+2*j, j]) for j in range(dim)], -1)
    direction = slopes/np.maximum(np.linalg.norm(slopes, axis=-1, keepdims=True), 1e-30)
    directed = np.clip(candidates[:, 0]+delta*np.sqrt(dim)*direction, 1e-6, 1-1e-6)
    return selected, directed.astype(np.float32), slopes


def evaluate_candidates(module, agent, states, candidates, cli, seed):
    count, branches, dim = candidates.shape
    records = [s for s in states for _ in range(branches)]
    envs = make_mujoco_vector_env('HalfCheetah-v4', len(records), num_threads=2)
    envs.reset(seed=seed)
    raw = physics.restore_states(envs, records)
    norm = state_normalizer(states, branches)
    obs = physics.normalized(norm, raw)
    mirror = module.ResidualHostMirror(agent.actor, len(records))
    rng = np.random.default_rng(seed)
    q = np.zeros(len(records))
    output = {}
    try:
        for t in range(max(cli.horizons)):
            if t == 0:
                native = candidates.reshape(len(records), dim)
            else:
                alpha, beta, _ = physics.policy(mirror, obs)
                uniforms = np.repeat(rng.uniform(1e-7, 1-1e-7, (count, dim)), branches, axis=0)
                native = physics.native_action(alpha, beta, 'sample', uniforms)
            raw, reward, terms, truncs, _ = envs.step((2*native-1).astype(np.float32))
            if np.any(terms|truncs):
                raise ValueError('Candidate fork crossed episode boundary')
            q += cli.gamma**t*reward
            obs = physics.normalized(norm, raw)
            if t+1 in cli.horizons:
                output[t+1] = q.reshape(count, branches).copy()
    finally:
        envs.close()
    return output


def local_oracle(module, agent, states, cli):
    if any('initial_ground_contact' not in s for s in states):
        envs = make_mujoco_vector_env('HalfCheetah-v4', len(states), num_threads=2)
        try:
            envs.reset(seed=cli.seed)
            physics.restore_states(envs, states)
            for s, e in zip(states, envs.envs):
                present, force = ground_contacts(e.unwrapped)
                s['initial_ground_contact'] = present
                s['initial_ground_normal_force'] = force
                s['initial_qpos'] = e.unwrapped.data.qpos.copy()
                s['initial_qvel'] = e.unwrapped.data.qvel.copy()
        finally:
            envs.close()
    norm = state_normalizer(states, 1)
    x = physics.normalized(norm, np.stack([s['obs'] for s in states]))
    mirror = module.ResidualHostMirror(agent.actor, len(states))
    alpha, beta, logits = physics.policy(mirror, x)
    mean = physics.native_action(alpha, beta, 'mean', None)
    mode = physics.native_action(alpha, beta, 'mode', None, logits)
    actual = np.stack([s['native'] for s in states])
    candidates, labels = candidate_actions(actual, mean, mode, cli.perturbation)
    discovery = [evaluate_candidates(module, agent, states, candidates, cli, cli.seed+100003*(i+1))
                 for i in range(cli.discovery_replicates)]
    selections, slopes, gradient_candidates = {}, {}, []
    for horizon in cli.horizons:
        selections[horizon], directed, slopes[horizon] = discovery_selection(
            candidates, [r[horizon] for r in discovery], cli.perturbation)
        gradient_candidates.append(directed)
    heldout_candidates = np.concatenate((candidates, np.stack(gradient_candidates, 1)), 1)
    heldout = [evaluate_candidates(module, agent, states, heldout_candidates, cli, cli.seed+2000003+100003*i)
               for i in range(cli.validation_replicates)]
    horizons = []
    for k, horizon in enumerate(cli.horizons):
        q = np.stack([r[horizon] for r in heldout])
        selected_q = q[:, np.arange(len(states)), selections[horizon]]
        directed_q = q[:, :, len(labels)+k]
        base_q = q[:, :, 0]
        difference = selected_q-base_q
        gradient_difference = directed_q-base_q
        horizons.append({'horizon': horizon, 'tail_weight': cli.gamma**horizon,
                         'discovery_selected_candidate_indices': selections[horizon].tolist(),
                         'discovery_action_gradient': slopes[horizon].tolist(),
                         'heldout_candidate_raw_Q': q.tolist(),
                         'selected_minus_actual': difference.tolist(), 'directed_minus_actual': gradient_difference.tolist(),
                         'selected_gain_state_mean': difference.mean(0).tolist(),
                         'directed_gain_state_mean': gradient_difference.mean(0).tolist(),
                         'selected_gain_summary': physics.summarize(difference.mean(0)),
                         'directed_gain_summary': physics.summarize(gradient_difference.mean(0)),
                         'selected_gain_mean_by_replicate': difference.mean(1).tolist(),
                         'directed_gain_mean_by_replicate': gradient_difference.mean(1).tolist(),
                         'selected_positive_state_fraction': float(np.mean(difference.mean(0)>0)),
                         'gradient_positive_state_fraction': float(np.mean(gradient_difference.mean(0)>0))})
    return {'labels': labels+[f'gradient_h{h}' for h in cli.horizons],
            'native_actions': heldout_candidates.tolist(), 'states': [state_metadata(s) for s in states],
            'horizons': horizons, 'discovery_replicates': cli.discovery_replicates,
            'validation_replicates': cli.validation_replicates,
            'scope': 'First action varied; continuation stochastic checkpoint policy frozen, with common uniforms per state and independent discovery/validation noise.',
            'limitations': ['Finite discounted raw-return gain includes unknown tail; not an infinite-horizon/global optimum claim.',
                            'Single-action conditional improvements cannot be multiplied by episode length or interpreted as closed-loop policy improvement.',
                            'State-based standard errors and continuation-replicate variability are separately reported; no multi-seed training inference.']}


def state_metadata(state):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in state.items() if k not in ('state', 'rng')}


def dense_states(snapshot, dense_result):
    records = []
    for selected in dense_result['selected_states']:
        t, row = selected['t'], selected['row']
        state = snapshot['transitions']['integration_states'][t, row]
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        records.append({'state': np.asarray(state).copy(), 'rng': copy.deepcopy(np.random.default_rng(1).bit_generator.state),
                        'age': selected['episode_age'], 'row': row, 'obs': np.asarray(selected['raw_obs']),
                        'means': np.asarray(selected['obs_means']), 'variances': np.asarray(selected['obs_variances']),
                        'native': np.asarray(selected['native_action']), 'source': 'matched_dense_training',
                        't': t, 'stored_advantage': selected['stored_advantage']})
    return records


def dense_contact_evidence(agent, states, dense_result):
    envs = make_mujoco_vector_env('HalfCheetah-v4', len(states), num_threads=2)
    envs.reset(seed=1)
    physics.restore_states(envs, states)
    contacts = [ground_contacts(e.unwrapped) for e in envs.envs]
    present = np.stack([c[0] for c in contacts]); force = np.stack([c[1] for c in contacts])
    joint_velocity = np.stack([e.unwrapped.data.qvel.copy() for e in envs.envs])
    envs.close()
    groups = np.stack((present[:, 2], present[:, 5]), -1).astype(int)
    phases = 2*groups[:, 0]+groups[:, 1]
    evidence = []
    for horizon in dense_result['oracle']['horizons']:
        pg = np.asarray(horizon['policy_logit_pg']['finite_raw']['mean'])
        entries = []
        for phase in range(4):
            mask = phases == phase
            entries.append({'contact_phase': phase, 'states': int(mask.sum()),
                            'finite_raw_logit_pg_mean': pg[mask].mean(0).tolist() if mask.any() else None,
                            'stored_advantage_mean': float(np.mean([s['stored_advantage'] for s, m in zip(states, mask) if m])) if mask.any() else None})
        evidence.append({'horizon': horizon['horizon'], 'phase_groups': entries})
    return {'ground_contact': present.tolist(), 'ground_normal_force': force.tolist(), 'qvel': joint_velocity.tolist(),
            'phase_code': '2*rear_foot_contact+front_foot_contact; descriptive grouping, not temporal stride phase.',
            'dense_pg_phase_groups': evidence, 'contact_geometry_order': CONTACT_GEOMS}


def match_dense_result(snapshot_path, snapshot_sha256, dense_results):
    matches = [r for r in dense_results if Path(r['snapshot']).resolve() == Path(snapshot_path).resolve()]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError('Ambiguous dense oracle: multiple results match the same checkpoint path')
    result = matches[0]
    if result.get('snapshot_sha256') != snapshot_sha256:
        raise ValueError('Dense oracle checkpoint SHA256 mismatch: cannot link states or PG evidence')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', action='append', required=True)
    p.add_argument('--dense-oracle', action='append', default=[])
    p.add_argument('--out', required=True)
    p.add_argument('--episodes', type=int, default=8)
    p.add_argument('--states', type=int, default=12)
    p.add_argument('--horizons', nargs='+', type=int, default=[64, 256])
    p.add_argument('--discovery-replicates', type=int, default=2)
    p.add_argument('--validation-replicates', type=int, default=4)
    p.add_argument('--perturbation', type=float, default=.025)
    p.add_argument('--gamma', type=float, default=.99)
    p.add_argument('--seed', type=int, default=1)
    cli = p.parse_args()
    if cli.states < 3 or cli.states % 3 or cli.episodes < cli.states//3 or min(cli.horizons)<1 or max(cli.horizons)>=400:
        p.error('Need states divisible by3, episodes>=states/3, horizons in[1,399]')
    if cli.perturbation<=0 or not 0<cli.gamma<=1:
        p.error('Need positive perturbation and gamma in(0,1]')
    if cli.discovery_replicates<2 or cli.validation_replicates<2:
        p.error('Need >=2 independent discovery and validation continuations')
    configure_runtime(matmul_precision='highest', allow_tf32=False)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; submit through mlq')
    torch._dynamo.config.recompile_limit = 64
    module = importlib.import_module(physics.MODULE)
    dense_results = [r for path in cli.dense_oracle for r in json.loads(Path(path).read_text())['results']]
    target = Path(cli.out); target.parent.mkdir(parents=True, exist_ok=True)
    output = {'protocol': vars(cli), 'results': [], 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for path in cli.snapshot:
        snapshot_digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        dense = match_dense_result(path, snapshot_digest, dense_results)
        snapshot = torch.load(path, map_location='cpu', weights_only=False)
        envs = make_mujoco_vector_env('HalfCheetah-v4', cli.episodes, num_threads=2)
        agent = module.Agent(envs).cuda().eval(); agent.load_state_dict(snapshot['model']); envs.close()
        agent.actor.forward = torch.compile(agent.actor.forward, fullgraph=True, dynamic=False, options={'triton.cudagraphs': False})
        agent.critic.forward = torch.compile(agent.critic.forward, fullgraph=True, dynamic=False, options={'triton.cudagraphs': False})
        probe = snapshot['batch']['obs'][:cli.episodes].cuda()
        with torch.no_grad():
            expected = agent.actor(probe).cpu().numpy()
        error = float(np.max(np.abs(physics.policy(module.ResidualHostMirror(agent.actor, cli.episodes), probe.cpu().numpy())[2]-expected)))
        if error>1e-4:
            raise ValueError(f'CUDA/host mismatch{error}')
        traces, states = [], []
        stem = hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:12]
        for mode in ('sample', 'mean', 'mode'):
            trace, captured = trace_checkpoint(module, agent, snapshot, cli, mode, target.parent/f'gait_{stem}_{mode}.npz')
            traces.append(trace)
            if captured:
                states = captured
        if dense is not None:
            states = dense_states(snapshot, dense)
            if max(s['age'] for s in states)+max(cli.horizons)>=1000:
                raise ValueError('Dense selected-state ages do not allow requested candidate horizon')
        result = {'snapshot': str(Path(path).resolve()), 'global_step': snapshot['global_step'],
                  'snapshot_sha256': snapshot_digest, 'host_cuda_max_abs_error': error,
                  'traces': traces, 'local_action_oracle': local_oracle(module, agent, states, cli)}
        if dense is not None:
            result['dense_contact_pg'] = dense_contact_evidence(agent, states, dense)
        output['results'].append(result)
        target.write_text(json.dumps(output, indent=2)+'\n')
        print(json.dumps({'snapshot': path, 'returns': {t['mode']: t['return_summary']['mean'] for t in traces}}), flush=True)


if __name__ == '__main__':
    main()
