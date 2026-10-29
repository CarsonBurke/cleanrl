"""Independent compiled CUDA replay contracts; execute through MLQ only."""
import copy
from dataclasses import asdict
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from scripts.sepclip_prescription_replay import ROOT, MODULES, load_replay, load_warm_optimizers, sha256
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(autouse=True)
def cuda_only():
    assert torch.cuda.is_available(), 'CUDA required: submit these contracts through MLQ'
    configure_runtime(matmul_precision='highest', allow_tf32=False)


def fixture_artifacts(tmp_path, version='v14', architecture='additive_prerms', floor=False):
    module_name = next(k for k, v in MODULES.items() if v == version)
    module = importlib.import_module(module_name)
    exp_name = 'sepclip_prescription_replay_contract'
    argv = ['--exp-name', exp_name, '--num-envs', '2', '--num-steps', '2', '--update-epochs', '2',
            '--num-minibatches', '1', '--compile']
    if version == 'v14':
        argv += ['--actor-architecture', architecture, '--actor-optimizer', 'adam']
    else:
        argv += ['--matrix-geometry', 'intrinsic']
    import tyro
    args = module.validate_args(tyro.cli(module.Args, args=argv))
    warmup = module.episode_horizon(args.env_id) if args.staggered_starts else 0
    args.num_iterations = (args.total_timesteps - warmup * args.num_envs) // args.batch_size
    env = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-np.ones(6, np.float32), np.ones(6, np.float32)))
    agent = (module.Agent(env, architecture) if version == 'v14' else module.Agent(env)).cuda().eval()
    with torch.no_grad():
        agent.normalize_matrices()
        if floor:
            agent.actor.head[0].bias[:6].fill_(-100 if floor in ('alpha', 'both') else 1)
            agent.actor.head[0].bias[6:].fill_(-100 if floor in ('beta', 'both') else 1)
        if version == 'v14' and architecture == 'additive_prerms':
            # Deliberately non-unit free-body matrix; loader must not retract it.
            agent.actor.trunk.in_proj.weight.mul_(2)
    rng = np.random.default_rng(713)
    raw = rng.normal(size=(2, 2, 17)).astype(np.float64)
    means = rng.normal(size=(2, 2, 17)).astype(np.float64)
    variances = rng.uniform(.2, 2, size=(2, 2, 17)).astype(np.float64)
    obs = torch.tensor(np.clip((raw - means) / np.sqrt(variances + 1e-8), -10, 10).astype(np.float32).reshape(4, 17), device='cuda')
    native = torch.full((4, 6), .25, device='cuda')
    native[0, 0] = module.SAMPLE_EPS
    native[1, 0] = 1 - module.SAMPLE_EPS
    # Independently record using actual public policy/value API, not loader helper.
    def record(x, a):
        alpha, beta, values = agent.get_policy_and_value(x)
        return alpha, beta, values.flatten(), agent.action_logprob(alpha, beta, a)
    with torch.no_grad():
        alpha, beta, values, logprob = [x.clone() for x in torch.compile(record, fullgraph=True,
            options={'triton.cudagraphs': False})(obs, native)]
    model = copy.deepcopy(agent.state_dict())
    matrix, auxiliary = module.make_optimizers(agent, args)
    transitions = dict(raw_obs=raw, obs_means=means, obs_variances=variances,
        raw_next_obs=raw.copy(), normalized_next_obs=obs.cpu().numpy().reshape(2, 2, 17),
        normalized_transition_obs=obs.cpu().numpy().reshape(2, 2, 17),
        integration_states=np.zeros((2, 2, 32), np.float64), episode_age=np.zeros((2, 2), np.int32),
        physical_actions=(2 * native - 1).cpu().numpy().reshape(2, 2, 6),
        raw_rewards=np.zeros((2, 2), np.float64), reward_std=np.ones((2, 2), np.float64),
        rewards=torch.zeros((2, 2), device='cuda'), terminations=torch.zeros((2, 2), device='cuda', dtype=torch.bool),
        truncations=torch.zeros((2, 2), device='cuda', dtype=torch.bool),
        truncation_values=torch.zeros((2, 2), device='cuda'), tail_value=torch.zeros(2, device='cuda'))
    snapshot = dict(model=model, args=asdict(args), global_step=warmup * 2 + 4,
        optimizer=auxiliary.state_dict(), matrix_optimizer=matrix.state_dict() if matrix is not None else None,
        batch=dict(obs=obs, native=native, alpha=alpha, beta=beta, values=values,
            logprob=logprob, advantages=torch.ones(4, device='cuda'), returns=values.clone()),
        transitions=transitions, capture_protocol=dict(timing='pre_update', state_spec=module.integration_backend(),
            integration_fields=list(module.INTEGRATION_FIELDS), integration_time_included=True,
            normalization='recorded per-env rows; no pooled substitution'),
        obs_norm=dict(means=means[-1].copy(), variances=variances[-1].copy(), counts=np.ones(2)),
        rew_norm=dict(returns=np.zeros(2), means=np.zeros(2), variances=np.ones(2), counts=np.ones(2)))
    entries = []
    for index in range(2):
        zeros = {k: torch.zeros_like(v) for k, v in agent.named_parameters()}
        zeros['actor.head.0.bias'].fill_(.002)
        entries.append(dict(update=index, epoch=index, gradient_after_clip=copy.deepcopy(zeros),
            ambient_delta=copy.deepcopy(zeros), retracted_delta=zeros))
        if version == 'v15':
            entries[-1].update(matrix_geometry=args.matrix_geometry,
                matrix_delta_stage='intrinsic endpoint before global normalization')
    endpoint = copy.deepcopy(model)
    for entry in entries:
        for name, delta in entry['retracted_delta'].items():
            endpoint[name].add_(delta)
    trace = dict(global_step=snapshot['global_step'], trace=entries, model_after_update=endpoint)
    manifest_path = ROOT / f'benchmarks/sepclip_prescription_v14/source_manifest_{version}.json'
    manifest = json.loads(manifest_path.read_text())
    execution = dict(name=exp_name, frozen_source_manifest_sha256=sha256(manifest_path),
        source_sha256=manifest['source_sha256'], command=['.venv/bin/python', '-u', manifest['trainer'], *argv])
    snapshot_path = tmp_path / 'snapshot.pt'; trace_path = tmp_path / 'trace.pt'; execution_path = tmp_path / 'execution.json'
    torch.save(snapshot, snapshot_path); torch.save(trace, trace_path)
    execution_path.write_text(json.dumps(execution))
    kwargs = dict(module_name=module_name, manifest_path=manifest_path, manifest_sha256=sha256(manifest_path),
        execution_path=execution_path, execution_sha256=sha256(execution_path), exp_name=exp_name,
        snapshot_sha256=sha256(snapshot_path), trace_sha256=sha256(trace_path))
    return snapshot, trace, snapshot_path, trace_path, kwargs


@pytest.mark.parametrize('version,architecture', [('v14', 'additive_prerms'), ('v14', 'original'), ('v15', 'original')])
def test_actual_architecture_compiled_recorded_shapes_values_and_exact_banks(tmp_path, version, architecture):
    snapshot, trace, sp, tp, kwargs = fixture_artifacts(tmp_path, version, architecture)
    before = copy.deepcopy(snapshot['model'])
    replay = load_replay(sp, tp, **kwargs)
    assert replay.validation['independently_recorded_logits'] is False
    assert set(replay.banks) == {'old', 'first', 'endpoint'}
    for name, parameter in replay.agent.actor.named_parameters():
        assert torch.equal(replay.banks['endpoint'][name], trace['model_after_update']['actor.' + name])
        assert torch.equal(replay.banks['first'][name], before['actor.' + name] + trace['trace'][0]['retracted_delta']['actor.' + name])
        assert torch.equal(parameter, before['actor.' + name])
    with torch.no_grad():
        logits, alpha, beta = replay.compiled_forward(replay.banks['old'], replay.snapshot['batch']['obs'])
        values = replay.compiled_old_value(replay.snapshot['batch']['obs'])
    assert torch.allclose(alpha, snapshot['batch']['alpha'], atol=3e-6, rtol=2e-5)
    assert torch.allclose(beta, snapshot['batch']['beta'], atol=3e-6, rtol=2e-5)
    assert torch.allclose(values, snapshot['batch']['values'], atol=3e-6, rtol=2e-5)
    assert logits.shape == (4, 12)
    if architecture == 'additive_prerms':
        assert not any('actor.trunk.in_proj' in k for k in replay.constraint_axes)
        assert not torch.allclose(torch.linalg.vector_norm(replay.agent.actor.trunk.in_proj.weight, dim=1), torch.ones(64, device='cuda'))


@pytest.mark.parametrize('floor', ['alpha', 'beta', 'both'])
def test_floor_shapes_have_forward_logits_but_no_independent_latent_record(tmp_path, floor):
    _, _, sp, tp, kwargs = fixture_artifacts(tmp_path, floor=floor)
    replay = load_replay(sp, tp, **kwargs)
    assert replay.validation['floor_shape_count'] > 0
    assert replay.validation['independently_recorded_logits'] is False
    with torch.no_grad():
        logits, alpha, beta = replay.compiled_forward(replay.banks['old'], replay.snapshot['batch']['obs'])
    assert torch.isfinite(logits).all()
    if floor in ('alpha', 'both'): assert (alpha == 1).all()
    if floor in ('beta', 'both'): assert (beta == 1).all()
    if floor == 'alpha': assert (beta > 1).all()
    if floor == 'beta': assert (alpha > 1).all()


@pytest.mark.parametrize('version,architecture', [('v14', 'additive_prerms'), ('v14', 'original'), ('v15', 'original')])
def test_saved_first_head_chord_compiled_jvp_matches_independent_affine_derivative(tmp_path, version, architecture):
    _, _, sp, tp, kwargs = fixture_artifacts(tmp_path, version, architecture)
    replay = load_replay(sp, tp, **kwargs)
    old = replay.banks['old']; direction = replay.deltas[0]; obs = replay.snapshot['batch']['obs']
    def derivative(parameters, observations):
        return torch.func.jvp(lambda p: torch.func.functional_call(replay.agent.actor, p, (observations,), strict=True),
                              (parameters,), (direction,))[1]
    jvp = torch.compile(derivative, fullgraph=True, options={'triton.cudagraphs': False})(old, obs).clone()
    # The only saved motion is head bias+.002, so derivative is gain*.002.
    expected = .002 * old['readout_gain'].expand_as(jvp)
    assert torch.allclose(jvp, expected, atol=1e-7, rtol=1e-5)
    with torch.no_grad():
        initial = replay.compiled_forward(old, obs)[0].clone()
        first = replay.compiled_forward(replay.banks['first'], obs)[0].clone()
    assert torch.allclose(first - initial, expected, atol=3e-6, rtol=2e-3)


def test_additive_free_projection_radial_jvp_is_not_legacy_sphere_tangent(tmp_path):
    _, _, sp, tp, kwargs = fixture_artifacts(tmp_path)
    replay = load_replay(sp, tp, **kwargs)
    old = replay.banks['old']; obs = replay.snapshot['batch']['obs']
    name = 'trunk.in_proj.weight'; weight = old[name]
    direction = {k: torch.zeros_like(v) for k, v in old.items()}
    direction[name] = .01 * weight
    wrong = {k: value.clone() for k, value in direction.items()}
    wrong[name] -= weight * ((weight * direction[name]).sum(1, keepdim=True) /
                             weight.square().sum(1, keepdim=True))
    def derivative(parameters, vector, observations):
        return torch.func.jvp(lambda p: torch.func.functional_call(replay.agent.actor, p, (observations,), strict=True),
                              (parameters,), (vector,))[1]
    compiled = torch.compile(derivative, fullgraph=True, options={'triton.cudagraphs': False})
    actual = compiled(old, direction, obs).clone()
    legacy = compiled(old, wrong, obs).clone()
    assert float(actual.norm()) > 1e-7
    assert float(legacy.norm()) < 1e-4 * float(actual.norm())
    with torch.no_grad():
        plus = replay.compiled_forward({k: v + direction[k] for k, v in old.items()}, obs)[0].clone()
        minus = replay.compiled_forward({k: v - direction[k] for k, v in old.items()}, obs)[0].clone()
    finite = (plus - minus) / 2
    assert float((finite - actual).norm() / actual.norm()) < .05


@pytest.mark.parametrize('corruption', ['args', 'derived_args', 'normalizer', 'normalizer_overflow', 'shapes', 'values', 'logprob', 'native_bounds', 'endpoint', 'coverage', 'epoch', 'nonfinite', 'missing_args', 'bytepin'])
def test_replay_rejects_corrupt_recording_or_bytepin(tmp_path, corruption):
    snapshot, trace, sp, tp, kwargs = fixture_artifacts(tmp_path)
    if corruption == 'args': snapshot['args']['matrix_method'] = 'normuon'
    elif corruption == 'derived_args': snapshot['args']['num_iterations'] += 1
    elif corruption == 'normalizer': snapshot['transitions']['obs_means'][0, 0, 0] += 1
    elif corruption == 'normalizer_overflow':
        snapshot['transitions']['raw_obs'][0, 0, 0] = 1e308
        snapshot['transitions']['obs_means'][0, 0, 0] = -1e308
    elif corruption == 'shapes': snapshot['batch']['alpha'][0, 0] += .1
    elif corruption == 'values': snapshot['batch']['values'][0] += .1
    elif corruption == 'logprob': snapshot['batch']['logprob'][0] += .1
    elif corruption == 'native_bounds': snapshot['batch']['native'][0, 0] = 0
    elif corruption == 'endpoint': trace['model_after_update']['actor.readout_gain'][0] += .1
    elif corruption == 'coverage': del trace['trace'][0]['ambient_delta']['actor.readout_gain']
    elif corruption == 'epoch': trace['trace'][1]['epoch'] = 0
    elif corruption == 'nonfinite': trace['trace'][0]['gradient_after_clip']['actor.readout_gain'][0] = float('nan')
    elif corruption == 'missing_args': del snapshot['args']['actor_architecture']
    torch.save(snapshot, sp); torch.save(trace, tp)
    if corruption != 'bytepin':
        kwargs.update(snapshot_sha256=sha256(sp), trace_sha256=sha256(tp))
    else:
        kwargs['snapshot_sha256'] = '0' * 64
    with pytest.raises((ValueError, RuntimeError)):
        load_replay(sp, tp, **kwargs)


@pytest.mark.parametrize('corruption', ['wrong_module', 'source', 'name'])
def test_identical_shape_trainer_and_execution_provenance_are_not_interchangeable(tmp_path, corruption):
    _, _, sp, tp, kwargs = fixture_artifacts(tmp_path, architecture='original')
    if corruption == 'wrong_module':
        kwargs['module_name'] = next(k for k, v in MODULES.items() if v == 'v15')
    else:
        execution = json.loads(kwargs['execution_path'].read_text())
        if corruption == 'name': execution['name'] = 'another_run'
        else: execution['source_sha256'][kwargs['module_name'].replace('.', '/') + '.py'] = '0' * 64
        kwargs['execution_path'].write_text(json.dumps(execution))
        kwargs['execution_sha256'] = sha256(kwargs['execution_path'])
    with pytest.raises(ValueError):
        load_replay(sp, tp, **kwargs)


@pytest.mark.parametrize('corruption', ['missing_geometry', 'geometry', 'stage'])
def test_v15_trace_requires_actual_geometry_and_delta_stage(tmp_path, corruption):
    _, trace, sp, tp, kwargs = fixture_artifacts(tmp_path, 'v15', 'original')
    if corruption == 'missing_geometry': del trace['trace'][0]['matrix_geometry']
    elif corruption == 'geometry': trace['trace'][0]['matrix_geometry'] = 'ambient'
    else: trace['trace'][0]['matrix_delta_stage'] = 'ambient proposal before global normalization'
    torch.save(trace, tp); kwargs['trace_sha256'] = sha256(tp)
    with pytest.raises(ValueError): load_replay(sp, tp, **kwargs)


@pytest.mark.parametrize('corruption', [None, 'geometry', 'axis', 'state_axis', 'version', 'basepoint', 'Adam_semantics'])
def test_v15_warm_optimizer_uses_actual_intrinsic_semantics(tmp_path, corruption):
    _, _, sp, tp, kwargs = fixture_artifacts(tmp_path, 'v15', 'original')
    replay = load_replay(sp, tp, **kwargs)
    saved = replay.snapshot['matrix_optimizer']
    if corruption == 'geometry': saved['param_groups'][0]['geometry'] = 'ambient'
    elif corruption == 'axis': saved['param_groups'][0]['constraint_dims'][0] = 0
    elif corruption == 'Adam_semantics': replay.snapshot['optimizer']['param_groups'][0]['eps'] = 1e-3
    elif corruption in ('version', 'basepoint', 'state_axis'):
        state = next(iter(saved['state'].values()))
        if corruption == 'version': state['state_version'] = 1
        elif corruption == 'state_axis': state['constraint_dim'] = 1 - state['constraint_dim']
        else: state['base_point'].add_(.1)
    if corruption is not None:
        with pytest.raises(ValueError): load_warm_optimizers(replay)
    else:
        matrix, _ = load_warm_optimizers(replay)
        assert matrix.__class__ is replay.module.MatrixOptimizer
        assert all(group['geometry'] == 'intrinsic' for group in matrix.param_groups)
