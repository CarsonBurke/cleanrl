"""Strict v14/v15 saved-policy replay support; all forwards compiled CUDA via MLQ.

No legacy actor dispatch, inferred latent logits or optimizer reinterpretation.
The physical loader validates weights and saved rollout statistics; optional
warm optimizer loading uses the actual allowlisted trainer's factory.
"""
from dataclasses import dataclass, fields
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import tyro

ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    'cleanrl.ppo_continuous_action_sepclip_additive_prerms_anvil_v14': 'v14',
    'cleanrl.ppo_continuous_action_sepclip_constraint_transport_v15': 'v15',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_pin(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(f'Byte pin mismatch: {path}')


def _finite(name, value, shape=None):
    if not isinstance(value, (torch.Tensor, np.ndarray)):
        raise ValueError(f'{name} must be a recorded array')
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} shape mismatch')
    valid = torch.isfinite(value).all() if isinstance(value, torch.Tensor) else np.isfinite(value).all()
    if not bool(valid):
        raise ValueError(f'{name} contains nonfinite values')


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value


def validate_provenance(module_name, manifest_path, manifest_sha256, execution_path,
                        execution_sha256, exp_name):
    if module_name not in MODULES:
        raise ValueError('Trainer module is not explicitly allowlisted')
    require_pin(manifest_path, manifest_sha256)
    require_pin(execution_path, execution_sha256)
    manifest = json.loads(Path(manifest_path).read_text())
    execution = json.loads(Path(execution_path).read_text())
    trainer = module_name.replace('.', '/') + '.py'
    if manifest.get('group') != MODULES[module_name] or manifest.get('trainer') != trainer:
        raise ValueError('Source manifest does not name the requested actual trainer')
    sources = manifest.get('source_sha256')
    if not isinstance(sources, dict) or trainer not in sources:
        raise ValueError('Missing immutable trainer source pin')
    for relative, digest in sources.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Source pin escapes repository')
        require_pin(path, digest)
    if execution.get('name') != exp_name or execution.get('frozen_source_manifest_sha256') != manifest_sha256:
        raise ValueError('Execution name/source-manifest binding mismatch')
    executed_sources = execution.get('source_sha256', {})
    required = {k: v for k, v in sources.items() if k == trainer or k.startswith('cleanrl/shared/')}
    if any(executed_sources.get(k) != v for k, v in required.items()):
        raise ValueError('Execution source hashes do not match frozen runtime sources')
    command = execution.get('command')
    if (not isinstance(command, list) or len(command) < 3 or command[:2] != ['.venv/bin/python', '-u']
            or command[2] != trainer):
        raise ValueError('Execution command does not dispatch requested trainer')
    return manifest, execution


@dataclass
class Replay:
    module: object
    args: object
    agent: object
    snapshot: dict
    trace: dict
    banks: dict
    deltas: tuple
    constraint_axes: dict
    compiled_forward: object
    compiled_old_value: object
    validation: dict


def _strict_state(agent, state, label):
    reference = agent.state_dict()
    if not isinstance(state, dict) or set(state) != set(reference):
        raise ValueError(f'{label} strict model keys mismatch')
    for name, value in state.items():
        _finite(f'{label}.{name}', value, reference[name].shape)
        if not isinstance(value, torch.Tensor) or value.dtype != reference[name].dtype:
            raise ValueError(f'{label}.{name} dtype mismatch')


def constraint_axes(module_name, args, agent):
    axes = {'actor.head.0.weight': 1, 'critic.head.0.weight': 1}
    for trunk in ('actor', 'critic'):
        if MODULES[module_name] == 'v14' and trunk == 'actor' and args.actor_architecture == 'additive_prerms':
            continue
        for stage in ('first', 'second'):
            for layer, dim in (('gate', 1), ('up', 1), ('down', 0)):
                axes[f'{trunk}.{stage}.0.{layer}.weight'] = dim
    if not set(axes) <= set(dict(agent.named_parameters())):
        raise ValueError('Architecture constraint metadata does not match actual parameters')
    return axes


def reconstruct_banks(agent, snapshot, trace, args, axes):
    """Private first reconstruction and exact endpoint, without normalizing either."""
    if trace.get('global_step') != snapshot['global_step']:
        raise ValueError('Trace/snapshot step mismatch')
    entries = trace.get('trace')
    per_epoch = (args.batch_size + args.minibatch_size - 1) // args.minibatch_size
    if args.target_kl is not None or not isinstance(entries, list) or len(entries) != args.update_epochs * per_epoch:
        raise ValueError('Require complete fixed-epoch update trace')
    _strict_state(agent, trace.get('model_after_update'), 'endpoint')
    names = dict(agent.named_parameters())
    old = {k: v.detach().cuda().clone() for k, v in snapshot['model'].items()}
    first = {k: v.clone() for k, v in old.items()}
    summed = {k: v.clone() for k, v in old.items()}
    for index, entry in enumerate(entries):
        if entry.get('update') != index or entry.get('epoch') != index // per_epoch:
            raise ValueError('Incomplete trace update/epoch sequence')
        if hasattr(args, 'matrix_geometry'):
            expected_stage = ('intrinsic endpoint before global normalization' if args.matrix_geometry == 'intrinsic'
                              else 'ambient proposal before global normalization')
            if entry.get('matrix_geometry') != args.matrix_geometry or entry.get('matrix_delta_stage') != expected_stage:
                raise ValueError('v15 update trace geometry/stage semantics mismatch')
        for field in ('gradient_after_clip', 'ambient_delta', 'retracted_delta'):
            if not isinstance(entry.get(field), dict) or set(entry[field]) != set(names):
                raise ValueError(f'Incomplete trace parameter coverage: {field}')
            for name, value in entry[field].items():
                _finite(f'trace[{index}].{field}.{name}', value, old[name].shape)
                if not isinstance(value, torch.Tensor) or value.dtype != old[name].dtype:
                    raise ValueError('Trace tensor dtype mismatch')
        for name, delta in entry['retracted_delta'].items():
            summed[name].add_(delta.cuda())
            _finite('cumulative.' + name, summed[name])
            if index == 0:
                first[name].add_(delta.cuda())
                _finite('first.' + name, first[name])
    endpoint = {k: v.detach().cuda().clone() for k, v in trace['model_after_update'].items()}
    buffers = set(old) - set(names)
    if any(not torch.equal(old[k], endpoint[k]) for k in buffers):
        raise ValueError('Nonparameter model buffers changed across update')
    error = max(float((summed[k] - endpoint[k]).abs().max()) for k in names)
    if error > 2e-5:
        raise ValueError(f'Cumulative delta versus exact endpoint error: {error}')
    norm_errors = {}
    for label, state in (('old', old), ('first', first), ('endpoint', endpoint)):
        norm_errors[label] = max(float((torch.linalg.vector_norm(state[k], dim=axis) - 1).abs().max())
                                 for k, axis in axes.items())
        if norm_errors[label] > 1e-4:
            raise ValueError(f'Constrained axes are not unit: {label}')
    actor_names = tuple(dict(agent.actor.named_parameters()))
    banks = {label: {k: state['actor.' + k].clone() for k in actor_names}
             for label, state in (('old', old), ('first', first), ('endpoint', endpoint))}
    deltas = ({k: entries[0]['retracted_delta']['actor.' + k].cuda().clone() for k in actor_names},
              {k: endpoint['actor.' + k] - old['actor.' + k] for k in actor_names})
    return banks, deltas, dict(update_count=len(entries), cumulative_endpoint_max_abs_error=error,
        constrained_axis_unit_max_abs_errors=norm_errors,
        first_policy_source='Old FP32 state plus stored first retracted delta; original first endpoint was not independently saved; subtraction/addition roundoff remains.',
        endpoint_policy_source='Exact full model_after_update; actor banks use its exact recorded actor tensors.')


def load_replay(snapshot_path, trace_path, *, module_name, manifest_path, manifest_sha256,
                execution_path, execution_sha256, exp_name, snapshot_sha256, trace_sha256):
    """Load only pinned v14/v15 training artifacts; must execute inside MLQ."""
    manifest, execution = validate_provenance(module_name, manifest_path, manifest_sha256,
                                            execution_path, execution_sha256, exp_name)
    require_pin(snapshot_path, snapshot_sha256)
    require_pin(trace_path, trace_sha256)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is mandatory; execute through MLQ')
    module = importlib.import_module(module_name)
    if Path(module.__file__).resolve() != ROOT / manifest['trainer']:
        raise ValueError('Imported trainer path differs from pinned actual module')
    snapshot = torch.load(snapshot_path, map_location='cuda', weights_only=False)
    trace = torch.load(trace_path, map_location='cuda', weights_only=False)
    raw_args = snapshot.get('args')
    if not isinstance(raw_args, dict) or set(raw_args) != {f.name for f in fields(module.Args)}:
        raise ValueError('Require complete exact actual Args schema')
    args = module.Args(**raw_args)
    expected = module.validate_args(tyro.cli(module.Args, args=execution['command'][3:]))
    horizon = module.episode_horizon(expected.env_id) if expected.staggered_starts and expected.num_envs > 1 else 0
    expected.num_iterations = (expected.total_timesteps - horizon * expected.num_envs) // expected.batch_size
    if (vars(args) != vars(expected) or args.exp_name != exp_name
            or any(type(raw_args[k]) is not type(v) for k, v in vars(expected).items())):
        raise ValueError('Snapshot Args differ from bound execution command/runtime derivation')
    if args.env_id != 'HalfCheetah-v4' or not args.compile or not args.cuda:
        raise ValueError('Require compiled CUDA HalfCheetah prescription semantics')
    for name, value in vars(args).items():
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f'Nonfinite Args: {name}')
    if not isinstance(snapshot.get('global_step'), int) or snapshot['global_step'] <= 0:
        raise ValueError('Missing positive snapshot step')
    module.configure_runtime(cudnn_deterministic=args.torch_deterministic, matmul_precision='highest', allow_tf32=False)
    model = snapshot.get('model', {})
    low = model.get('action_low'); high = model.get('action_high')
    _finite('action_low', low, (6,)); _finite('action_high', high, (6,))
    if not np.array_equal(_numpy(low), -np.ones(6)) or not np.array_equal(_numpy(high), np.ones(6)):
        raise ValueError('HalfCheetah physical action bounds mismatch')
    descriptor = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(_numpy(low), _numpy(high)))
    agent = (module.Agent(descriptor, args.actor_architecture) if MODULES[module_name] == 'v14'
             else module.Agent(descriptor)).cuda().eval()
    _strict_state(agent, model, 'old')
    agent.load_state_dict(model, strict=True)
    if not torch.equal(agent.action_scale, agent.action_high - agent.action_low) or not torch.equal(agent.log_action_scale, agent.action_scale.log()):
        raise ValueError('Recorded action-scale buffers inconsistent with bounds')
    batch = snapshot.get('batch', {}); transitions = snapshot.get('transitions', {})
    shapes = {'obs': (args.batch_size, 17), 'native': (args.batch_size, 6),
        'alpha': (args.batch_size, 6), 'beta': (args.batch_size, 6)}
    shapes.update({k: (args.batch_size,) for k in ('logprob', 'values', 'advantages', 'returns')})
    for name, shape in shapes.items():
        _finite('batch.' + name, batch.get(name), shape)
        if not isinstance(batch[name], torch.Tensor) or batch[name].dtype != torch.float32:
            raise ValueError('Recorded batch must use actual FP32 trainer tensors')
    if not bool(((batch['alpha'] >= 1) & (batch['beta'] >= 1)).all()):
        raise ValueError('Invalid floor-one Beta shapes')
    if not bool(((batch['native'] >= module.SAMPLE_EPS) & (batch['native'] <= 1 - module.SAMPLE_EPS)).all()):
        raise ValueError('Native action clipping-law bounds mismatch')
    protocol = snapshot.get('capture_protocol', {})
    if (protocol.get('timing') != 'pre_update' or protocol.get('normalization') != 'recorded per-env rows; no pooled substitution'
            or protocol.get('integration_time_included') is not True
            or protocol.get('integration_fields') != list(module.INTEGRATION_FIELDS)
            or protocol.get('state_spec') != module.integration_backend()):
        raise ValueError('Missing actual pre-update/per-row capture semantics')
    layout = (args.num_steps, args.num_envs, 17)
    for name in ('raw_obs', 'obs_means', 'obs_variances'):
        _finite('transitions.' + name, transitions.get(name), layout)
        if _numpy(transitions[name]).dtype != np.float64:
            raise ValueError('Per-row observation capture must be FP64')
    transition_shapes = {k: layout for k in ('raw_next_obs', 'normalized_next_obs', 'normalized_transition_obs')}
    transition_shapes.update({k: (args.num_steps, args.num_envs) for k in
        ('rewards', 'terminations', 'truncations', 'truncation_values', 'episode_age', 'raw_rewards', 'reward_std')})
    transition_shapes.update(physical_actions=(args.num_steps, args.num_envs, 6), tail_value=(args.num_envs,))
    for name, shape in transition_shapes.items():
        _finite('transitions.' + name, transitions.get(name), shape)
    physical = _numpy(transitions['physical_actions']).reshape(args.batch_size, 6)
    expected_physical = _numpy(batch['native']) * _numpy(agent.action_scale) + _numpy(agent.action_low)
    physical_error = float(np.max(np.abs(physical - expected_physical)))
    if physical_error > 3e-7:
        raise ValueError('Recorded native/physical action mapping mismatch')
    integration = transitions.get('integration_states')
    _finite('transitions.integration_states', integration)
    if integration.ndim != 3 or tuple(integration.shape[:2]) != (args.num_steps, args.num_envs) or integration.shape[2] < 1:
        raise ValueError('Integration capture layout mismatch')
    if (_numpy(transitions['reward_std']) <= 0).any() or (_numpy(transitions['episode_age']) < 0).any():
        raise ValueError('Invalid reward normalization scale or episode age')
    variances = _numpy(transitions['obs_variances'])
    if (variances < 0).any():
        raise ValueError('Negative recorded observation variance')
    norm = module.VectorObsNorm(args.num_envs, (17,))
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        quotient = (_numpy(transitions['raw_obs']) - _numpy(transitions['obs_means'])) / np.sqrt(variances + norm.epsilon)
    _finite('preclip frozen observation quotient', quotient, layout)
    normalized = np.clip(quotient, -norm.clip, norm.clip).astype(np.float32)
    norm_error = float(np.max(np.abs(normalized.reshape(args.batch_size, 17) - _numpy(batch['obs']))))
    if not np.isfinite(norm_error) or norm_error > 3e-6:
        raise ValueError(f'Frozen per-row normalization/batch observation mismatch: {norm_error}')
    for label in ('obs_norm', 'rew_norm'):
        state = snapshot.get(label, {})
        names = ('means', 'variances', 'counts') if label == 'obs_norm' else ('returns', 'means', 'variances', 'counts')
        for name in names:
            shape = (args.num_envs, 17) if label == 'obs_norm' and name != 'counts' else (args.num_envs,)
            _finite(label + '.' + name, state.get(name), shape)
        if (_numpy(state['variances']) < 0).any() or (_numpy(state['counts']) <= 0).any():
            raise ValueError('Invalid saved normalizer statistics')
    def forward(obs):
        logits = agent.actor(obs)
        alpha, beta = (1 + F.softplus(logits)).chunk(2, -1)
        return logits, alpha, beta, agent.get_value(obs).flatten()
    def statistics(obs, native):
        logits, alpha, beta, values = forward(obs)
        return logits, alpha, beta, values, agent.action_logprob(alpha, beta, native)
    compiled = torch.compile(statistics, fullgraph=True, dynamic=False, options={'triton.cudagraphs': False})
    with torch.no_grad():
        replayed = compiled(batch['obs'], batch['native'])
    _finite('compiled forward logits (not independently recorded)', replayed[0], (args.batch_size, 12))
    errors = {}
    for name, actual in zip(('alpha', 'beta', 'values', 'logprob'), replayed[1:]):
        actual = actual.clone()
        if not bool(torch.isfinite(actual).all()):
            raise ValueError(f'Nonfinite compiled replay: {name}')
        errors[name] = float((actual - batch[name]).abs().max())
        if not torch.allclose(actual, batch[name], rtol=2e-5, atol=2e-5 if name == 'logprob' else 3e-6):
            raise ValueError(f'Actual compiled recorded-{name} replay mismatch: {errors[name]}')
    axes = constraint_axes(module_name, args, agent)
    banks, deltas, reconstruction = reconstruct_banks(agent, snapshot, trace, args, axes)
    for path, pin in ((snapshot_path, snapshot_sha256), (trace_path, trace_sha256),
                      (manifest_path, manifest_sha256), (execution_path, execution_sha256)):
        require_pin(path, pin)
    validate_provenance(module_name, manifest_path, manifest_sha256, execution_path, execution_sha256, exp_name)
    validation = dict(module=module_name, architecture=getattr(args, 'actor_architecture', 'original'),
        snapshot_sha256=snapshot_sha256, trace_sha256=trace_sha256,
        source_manifest_sha256=manifest_sha256, execution_sha256=execution_sha256,
        exp_name=exp_name, independently_recorded_logits=False,
        logits_source='Actual pinned compiled actor forward; snapshot has no independently saved latent logits; no inversion or floor-latent certification.',
        recorded_forward_max_abs_errors=errors, frozen_row_normalization_max_abs_error=norm_error,
        native_physical_action_max_abs_error=physical_error,
        floor_shape_count=int(((batch['alpha'] == 1) | (batch['beta'] == 1)).sum()),
        reconstruction=reconstruction)
    if tuple(agent.actor.named_buffers()):
        raise ValueError('Unspecified actor buffers require explicit replay support')
    def bank_forward(parameters, obs):
        logits = torch.func.functional_call(agent.actor, parameters, (obs,), strict=True)
        alpha, beta = (1 + F.softplus(logits)).chunk(2, -1)
        return logits, alpha, beta
    return Replay(module, args, agent, snapshot, trace, banks, deltas, axes,
        torch.compile(bank_forward, fullgraph=True, dynamic=False, options={'triton.cudagraphs': False}),
        torch.compile(lambda obs: agent.get_value(obs).flatten(), fullgraph=True, dynamic=False,
                      options={'triton.cudagraphs': False}), validation)


def load_warm_optimizers(replay):
    """Optional strict actual-module state loading; no steps or generic rail replay."""
    matrix, auxiliary = replay.module.make_optimizers(replay.agent, replay.args)
    warm_validation = dict(module=replay.module.__name__, intrinsic_stacks=[],
        execution_certification='State schema/algebra checks only; no independent warm-continuation execution certification until queued contracts and actual capture review.')
    saved = replay.snapshot.get('matrix_optimizer')
    if (matrix is None) != (saved is None):
        raise ValueError('Matrix optimizer presence differs from actual Args')
    if matrix is not None:
        semantic_keys = ('method', 'precision', 'slow_clock', 'momentum', 'beta2', 'total_steps')
        if MODULES[replay.module.__name__] == 'v15':
            semantic_keys += ('geometry', 'constraint_dims')
        expected_groups = [{k: group[k] for k in semantic_keys} for group in matrix.param_groups]
        expected_state = matrix.state_dict()['state']
        if not isinstance(saved, dict) or set(saved.get('state', {})) != set(expected_state):
            raise ValueError('Warm matrix stack coverage mismatch')
        for key, state in saved['state'].items():
            if set(state) != set(expected_state[key]):
                raise ValueError('Warm matrix state schema mismatch')
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    _finite('recorded warm matrix.' + name, value, expected_state[key][name].shape)
                    if value.dtype != expected_state[key][name].dtype:
                        raise ValueError('Recorded warm matrix state dtype mismatch')
        matrix.load_state_dict(saved)
        for group, expected in zip(matrix.param_groups, expected_groups):
            if any(group.get(k) != v for k, v in expected.items()):
                raise ValueError('Warm matrix optimizer semantic metadata mismatch')
        for group, parameters, state, _ in matrix.stacks:
            if MODULES[replay.module.__name__] == 'v15':
                if state.get('state_version') != 2:
                    raise ValueError('v15 stacked state version mismatch')
                parameter_axes = {id(p): dim for p, dim in zip(group['params'], group['constraint_dims'] or [None] * len(group['params']))}
                if any(state.get('constraint_dim') != parameter_axes[id(p)] for p in parameters):
                    raise ValueError('v15 stacked constraint dimension mismatch')
            for name in ('velocity', 'second'):
                _finite('warm.' + name, state[name])
            if bool((state['second'] < 0).any()):
                raise ValueError('Negative matrix second moment')
            if replay.args.matrix_method != group['method'] or replay.args.matrix_precision != group['precision']:
                raise ValueError('Warm matrix method/precision mismatch')
            if MODULES[replay.module.__name__] == 'v15' and group['geometry'] == 'intrinsic':
                _finite('warm.base_point', state['base_point'])
                _finite('warm.diagnostics', state['diagnostics'], (5,))
                axis = state['constraint_dim'] - 2
                point = state['base_point']; current = torch.stack(parameters)
                if (float((torch.linalg.vector_norm(point, dim=axis) - 1).abs().max()) > 1e-4
                        or bool((1 + (point * current).sum(axis) <= 0).any())
                        or float(state['diagnostics'][0]) != 0):
                    raise ValueError('Invalid intrinsic base point/pathology state')
                if float((state['velocity'] * point).sum(axis).abs().max()) > 1e-4:
                    raise ValueError('Intrinsic saved momentum is not tangent at its base point')
                warm_validation['intrinsic_stacks'].append(dict(constraint_dim=state['constraint_dim'],
                    basepoint_current_max_abs_difference=float((point - current).abs().max()),
                    renormalized_basepoint_current_max_abs_difference=float((point / torch.linalg.vector_norm(point, dim=axis, keepdim=True) - current).abs().max()),
                    minimum_transport_denominator=float((1 + (point * current).sum(axis)).min()),
                    residual_scope='Observed arithmetic residuals, no bitwise identity or capture-calibrated residual bound.'))
    adam_semantics = [{k: v for k, v in group.items() if k not in ('params', 'lr')}
                      for group in auxiliary.param_groups]
    saved_adam = replay.snapshot['optimizer']
    for state in saved_adam['state'].values():
        for name, value in state.items():
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
                raise ValueError('Recorded fused Adam state dtype/schema mismatch')
            _finite('recorded warm.Adam.' + name, value)
    auxiliary.load_state_dict(saved_adam)
    for group, expected in zip(auxiliary.param_groups, adam_semantics):
        actual = {k: v for k, v in group.items() if k not in ('params', 'lr')}
        if actual != expected:
            raise ValueError('Warm Adam non-LR semantic metadata mismatch')
    for state in auxiliary.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                _finite('warm.Adam.' + name, value)
    for group in auxiliary.param_groups:
        for parameter in group['params']:
            state = auxiliary.state.get(parameter)
            if state:
                for name in ('exp_avg', 'exp_avg_sq'):
                    _finite('warm.Adam.' + name, state.get(name), parameter.shape)
                if bool((state['exp_avg_sq'] < 0).any()):
                    raise ValueError('Negative Adam second moment')
    replay.validation['warm_optimizer'] = warm_validation
    return matrix, auxiliary
