"""Conditional residual stacking on the unchanged, chronological full-file SPY stream.

All six hyperparameters lock on real observations [2000,294912), before the
first suffix label. The null inherits those locks but initializes every learner
fresh. The suffix extends prior actual consumption, not all historical exposure.
"""

import hashlib
import inspect
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.plasticity import covariance_stock_eval_v1 as stock
from cleanrl.plasticity import predictive_dynamic_stock_eval_v3 as dynamic
from cleanrl.plasticity import stock_stream
from cleanrl.plasticity.predictive_conditional_stack_v7 import StateConditionedStack
from cleanrl.plasticity.predictive_stock_transfer_eval_v1 import AdamBank, StockRunner, summarize_predictions
from cleanrl.shared import runtime
from cleanrl.shared.autocull import PRUNED_EXIT_CODE, ProxyCull, ProxyPruned, prune_proxy

SELECTION_END = 294912
COLD_START = 2000
PRIORS = (.01, .1, 1.)
ADAM_GRID = stock.Args.adam_lrs
MODEL_FAMILIES = ('calibration', 'stacking', 'context')
ADAM_FAMILIES = ('linear', 'mlp', 'context')
MODEL_NAMES = (*(f'{family}_{prior:g}' for family in MODEL_FAMILIES for prior in PRIORS),
               *(f'base_{name}' for name in dynamic.MODEL_NAMES),
               *(f'context_adam_{lr:g}' for lr in ADAM_GRID))


@dataclass
class Args:
    output_dir: str
    view: Literal['real', 'random_sign'] = 'real'
    bars: str = stock_stream.Args.bars
    seed: int = 1
    selection_end: int = SELECTION_END
    log_every: int = 8192
    autocull: bool = True
    real_result: str = ''


def phase_ranges(samples, selection_end=SELECTION_END):
    if type(selection_end) is not int or selection_end != SELECTION_END:
        raise ValueError('selection_end must remain the prospectively fixed 294912')
    if type(samples) is not int or samples <= selection_end:
        raise ValueError('stream must extend beyond the fixed selection prefix')
    return {'cold_start': (0, COLD_START), 'selection_prefix': (COLD_START, selection_end),
            'prefix_all': (0, selection_end), 'suffix_all': (selection_end, samples)}


def output_names(model_names=MODEL_NAMES, adam_grid=ADAM_GRID):
    return (*model_names, *(f'adam_linear_{lr:g}' for lr in adam_grid),
            *(f'adam_mlp_{lr:g}' for lr in adam_grid))


def source_fingerprint():
    # The frozen evaluator already enumerates all transitive learner/stream/runtime
    # sources. Include it too because its semantic base names are used above.
    hashes = dynamic.source_fingerprint()
    for path in (__file__, inspect.getfile(StateConditionedStack)):
        hashes[Path(path).name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return hashes


def candidate_columns(names, family, grid, *, adam=False):
    if len(set(names)) != len(names):
        raise ValueError('output names must be unique')
    prefix = ('context_adam' if family == 'context' else f'adam_{family}') if adam else family
    return [names.index(f'{prefix}_{value:g}') for value in grid]


def select_prefix_locks(predictions, target, names, consumed, selection_end=SELECTION_END,
                        priors=PRIORS, adam_grid=ADAM_GRID):
    """Select all six families once, using only the exact real-prefix barrier."""
    if (type(selection_end) is not int or selection_end != SELECTION_END
            or type(consumed) is not int or consumed != selection_end
            or len(predictions) != selection_end or len(target) != selection_end):
        raise ValueError('selection requires the exact 294912 barrier before any suffix label')
    if predictions.ndim != 2 or predictions.shape[1] != len(names):
        raise ValueError('one prediction column is required per output name')
    locks = {'model_locks': {}, 'adam_locks': {}}
    for adam, families, grid, key in ((False, MODEL_FAMILIES, priors, 'model_locks'),
                                      (True, ADAM_FAMILIES, adam_grid, 'adam_locks')):
        for family in families:
            columns = candidate_columns(names, family, grid, adam=adam)
            lock = stock.select_prefix(predictions[:, columns], target, grid,
                                       COLD_START, selection_end, consumed)
            if not all(math.isfinite(row['error_ratio']) for row in lock['candidates']):
                raise FloatingPointError('nonfinite prefix candidate; no selective omission of failed arms')
            if not adam:
                lock['selected_prior'] = lock.pop('selected_lr')
                lock['tie_break'] = 'first (smallest) prior'
                for candidate in lock['candidates']:
                    candidate['prior'] = candidate.pop('lr')
            locks[key][family] = lock
    return locks


def load_real_locks(path, fingerprint, source_sha256, selection_end=SELECTION_END,
                    adam_grid=ADAM_GRID, priors=PRIORS):
    """Validate the six real-prefix locks, including their evidence and argmins."""
    try:
        source = json.loads(Path(path).read_bytes())
        if (source['args']['view'] != 'real' or source['args']['seed'] != 1
                or source['args']['selection_end'] != selection_end):
            raise ValueError('locks require matching real seed1 selection provenance')
        if source['status'] not in ('running', 'completed', 'pruned'):
            raise ValueError('real artifact must be running, completed, or intentionally pruned')
        for name in ('features', 'target'):
            value = fingerprint[name]
            if (not isinstance(value, str) or len(value) != 64
                    or any(c not in '0123456789abcdef' for c in value)
                    or source['data_sha256'][name] != value):
                raise ValueError(f'real data SHA256 mismatch: {name}')
        if not source_sha256 or source['source_sha256'] != source_sha256:
            raise ValueError('algorithm source SHA256 mismatch')
        samples = source['samples']
        phases = phase_ranges(samples, selection_end)
        if source['phases'] != {name: list(bounds) for name, bounds in phases.items()}:
            raise ValueError('real phases do not match fixed selection boundary')
        consumed = source['processed_observations']
        if type(consumed) is not int or not selection_end <= consumed <= samples:
            raise ValueError('real artifact has not consumed the exact selection prefix')
        if source['processed'] != consumed or source['maximum_observations'] != samples:
            raise ValueError('inconsistent processed count or maximum horizon')
        if source['status'] == 'completed' and consumed != samples:
            raise ValueError('completed artifact has an unconsumed suffix')
        if (tuple(adam_grid) != ADAM_GRID or source['adam_grid'] != list(adam_grid)
                or tuple(priors) != PRIORS or source['model_priors'] != list(priors)
                or source['output_names'] != list(output_names())):
            raise ValueError('incompatible fixed model priors, output order, or original Adam grid')
        shared_target_sum = None
        for key, families, grid, value_key in (('model_locks', MODEL_FAMILIES, priors, 'prior'),
                                              ('adam_locks', ADAM_FAMILIES, adam_grid, 'lr')):
            locks = source[key]
            if set(locks) != set(families):
                raise ValueError(f'all three {key} are required')
            for family, lock in locks.items():
                for field, expected in (('selection_start_inclusive', COLD_START),
                                        ('selection_end_exclusive', selection_end),
                                        ('optimizer_updates_at_lock', selection_end),
                                        ('suffix_observations_used', 0)):
                    if type(lock[field]) is not int or lock[field] != expected:
                        raise ValueError(f'{family} lock is not causal: {field}')
                tie_break = 'first (smallest) prior' if value_key == 'prior' else 'first (smallest) learning rate'
                if (lock['criterion'] != 'minimum prequential squared error / zero-predictor squared error'
                        or lock['tie_break'] != tie_break):
                    raise ValueError('incompatible prefix selection rule')
                index = lock['selected_index']
                if (type(index) is not int or not 0 <= index < len(grid)
                        or lock[f'selected_{value_key}'] != grid[index]):
                    raise ValueError('selected index/value does not match grid')
                if len(lock['candidates']) != len(grid):
                    raise ValueError('candidate count does not match grid')
                scores = []
                for candidate, value in zip(lock['candidates'], grid):
                    if (candidate[value_key] != value or type(candidate['count']) is not int
                            or candidate['count'] != selection_end - COLD_START):
                        raise ValueError('candidate value/count does not match prefix')
                    keys = ('target_squared_sum', 'prediction_squared_sum', 'target_prediction_sum',
                            'error_squared_sum', 'error_ratio', 'prediction_energy_ratio',
                            'signed_cross_term_ratio', 'decomposition_residual')
                    if any(not math.isfinite(candidate[name]) for name in keys):
                        raise ValueError('nonfinite prefix candidate evidence')
                    yy, pp, yp, ee = (candidate[name] for name in keys[:4])
                    if yy <= 0 or pp < 0 or ee < 0:
                        raise ValueError('invalid squared sums')
                    if shared_target_sum is None:
                        shared_target_sum = yy
                    if yy != shared_target_sum:
                        raise ValueError('all six families must share identical real-prefix targets')
                    expected = {'error_ratio': ee / yy, 'prediction_energy_ratio': pp / yy,
                                'signed_cross_term_ratio': -2 * yp / yy,
                                'decomposition_residual': (ee - yy - pp + 2 * yp) / yy}
                    if (not math.isclose(ee, yy + pp - 2 * yp, rel_tol=1e-10, abs_tol=1e-10 * yy)
                            or yp * yp > yy * pp + 1e-10 * yy * max(yy, pp)
                            or any(not math.isclose(candidate[name], expected_value, rel_tol=1e-10, abs_tol=1e-12)
                                   for name, expected_value in expected.items())):
                        raise ValueError('candidate sums and normalized metrics disagree')
                    scores.append(candidate['error_ratio'])
                if index != min(range(len(scores)), key=scores.__getitem__):
                    raise ValueError('fabricated winner or invalid first-candidate tie break')
        return {key: source[key] for key in ('model_locks', 'adam_locks')}
    except (KeyError, TypeError, IndexError, OverflowError, json.JSONDecodeError) as error:
        raise ValueError(f'invalid real lock provenance: {error}') from error


def selected_columns(names, model_locks, adam_locks, priors=PRIORS, adam_grid=ADAM_GRID):
    """Resolve controls semantically, independently of model width or column order."""
    columns = {name: names.index(name) for name in
               ('base_static_mixture', 'base_adaptive_mixture', 'base_dynamic_mixture', 'base_zero')}
    if model_locks:
        for family in MODEL_FAMILIES:
            columns[family] = candidate_columns(names, family, priors)[model_locks[family]['selected_index']]
    if adam_locks:
        for family in ADAM_FAMILIES:
            columns[f'adam_{family}'] = candidate_columns(names, family, adam_grid, adam=True)[
                adam_locks[family]['selected_index']]
    return columns


def paired_comparisons(summary, names, columns):
    comparisons = {}
    primary = columns.get('context')
    for phase, row in summary.items():
        metrics = row['metrics']
        comparisons[phase] = {'count': row['count'], 'complete': row['complete'],
                              'observed_range': row['observed_range'], 'comparisons': [
            {'name': names[primary], 'role': 'context', 'baseline': baseline,
             'baseline_name': names[column],
             'error_ratio_difference': metrics[primary]['error_ratio'] - metrics[column]['error_ratio'],
             'mse_difference': (metrics[primary]['error_squared_sum']
                                - metrics[column]['error_squared_sum']) / row['count']}
            for baseline, column in columns.items() if baseline != 'context'
        ] if metrics and primary is not None else []}
    return comparisons


def checkpoint_health(runner):
    model, adam = runner.model, runner.adam
    clocks = {'runner': int(runner.index.item()), 'model': int(model.observations.item()),
              'base': int(model.base.observations.item()),
              'meta_updates': float((2 * (model.meta.alpha - 2)).item()),
              'context_adam_steps': int(model.context_adam.steps.item()),
              'linear_index': int(adam.linear.index.item()), 'linear_steps': int(adam.linear.steps.item()),
              'mlp_index': int(adam.mlp.index.item()), 'mlp_steps': int(adam.mlp.steps.item())}
    covariance = {}
    for name, cov in (('base', model.base.cov), ('meta', model.meta.cov)):
        _, info = torch.linalg.cholesky_ex(cov, check_errors=False)
        covariance[name] = {'cholesky_info': info.cpu().tolist(),
                            'max_asymmetry': float((cov - cov.mT).abs().amax().item())}
    return clocks, covariance


def save_artifacts(root, runner, target, names, phases, consumed, policy, result):
    if type(consumed) is not int or not 0 <= consumed <= len(target):
        raise ValueError('checkpoint consumed count is outside the stream')
    prediction = runner.predictions[:consumed].cpu().numpy()
    # Never sanitize failed evidence or persist unwritten future trajectory slots.
    np.save(root / 'predictions.npy', prediction)
    np.save(root / 'targets.npy', target[:consumed])
    model = runner.model
    adam_state = [tensor.cpu() for tensor in runner.adam.mutable
                  if tensor is not runner.adam.mlp.predictions]
    temporary = root / 'checkpoint.pt.tmp'
    torch.save({'step': consumed, 'runner_index': runner.index.cpu(),
                'model_state': [tensor.cpu() for tensor in model.state_tensors()],
                'base_state': [tensor.cpu() for tensor in model.base.state_tensors()],
                'meta_state': [tensor.cpu() for tensor in model.meta.state_tensors()],
                'context_adam_state': [tensor.cpu() for tensor in model.context_adam.state_tensors()],
                'model_configs': model.configs, 'adam_state': adam_state,
                'adam_configs': runner.adam.configs, 'autocull': policy.state_dict(),
                'note': 'Complete mutable learner state; consumed trajectories stored separately. No resume CLI.'},
               temporary)
    temporary.replace(root / 'checkpoint.pt')
    # Normal checkpoints already have current diagnostics from the replay loop.
    # Capture/exception exits may not; preserve raw data/state before attempting
    # diagnostics so even an unhealthy covariance cannot erase consumed evidence.
    if result.get('clocks', {}).get('runner') != consumed:
        try:
            clocks, covariance = checkpoint_health(runner)
            result.update(clocks=clocks, covariance_health=covariance)
        except Exception as error:
            result['checkpoint_diagnostic_failure'] = {'type': type(error).__name__, 'message': str(error)}
    summary = summarize_predictions(target, prediction, names, phases, consumed)
    columns = selected_columns(names, result.get('model_locks'), result.get('adam_locks'),
                               result['model_priors'], result['adam_grid'])
    result.update(processed=consumed, processed_observations=consumed, phase_metrics=summary,
                  selected_columns=columns, paired_comp=paired_comparisons(summary, names, columns),
                  prediction_artifact='predictions.npy', target_artifact='targets.npy',
                  checkpoint_artifact='checkpoint.pt',
                  nonfinite_predictions=int(np.count_nonzero(~np.isfinite(prediction))),
                  autocull_state=policy.state_dict())
    stock.save_result(root, result)


@torch.no_grad()
def main():
    args = tyro.cli(Args)
    if not args.output_dir.strip():
        raise ValueError('explicit nonempty --output-dir is required')
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError('output directory must be fresh and empty')
    root.mkdir(parents=True, exist_ok=True)
    result = {'args': asdict(args), 'run_dir': str(root), 'status': 'preparing',
              'processed': 0, 'processed_observations': 0, 'curves': [],
              'model_locks': None, 'adam_locks': None}
    stock.save_result(root, result)
    print(f'RESULTS {root / "results.json"}', flush=True)
    runner, writer, target, policy = None, None, None, None
    consumed = 0
    started = time.perf_counter()
    try:
        if args.seed != 1 or args.log_every <= 0 or args.selection_end != SELECTION_END:
            raise ValueError('seed1, fixed selection_end=294912, and positive log cadence required')
        if (args.view == 'random_sign') != bool(args.real_result):
            raise ValueError('random_sign requires --real-result; real selects its own six prefix locks')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU learner fallback')
        runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
        helper_args = stock_stream.Args(seed=1, steps=0)
        bars = stock_stream.read_bars(args.bars)
        features, real_target = stock_stream.build_stream(bars, helper_args)
        samples = len(real_target)
        phases = phase_ranges(samples, args.selection_end)
        prefix, grid = args.selection_end, ADAM_GRID
        fingerprint = {'features': hashlib.sha256(memoryview(features).cast('B')).hexdigest(),
                       'target': hashlib.sha256(memoryview(real_target).cast('B')).hexdigest()}
        sources = source_fingerprint()
        target = real_target
        if args.view == 'random_sign':
            payload = Path(args.real_result).read_bytes()
            locks = load_real_locks(args.real_result, fingerprint, sources, prefix, grid)
            if payload != Path(args.real_result).read_bytes():
                raise ValueError('real lock artifact changed while loading')
            if json.loads(payload)['samples'] != samples:
                raise ValueError('real lock horizon differs from current stream')
            result.update(**locks, transferred_lock_sha256=hashlib.sha256(payload).hexdigest())
            signs = np.random.default_rng(1).integers(0, 2, size=samples, dtype=np.int8) * 2 - 1
            target = real_target * signs
        result.update(samples=samples, maximum_observations=samples, bars=len(bars),
                      phases=phases, data_sha256=fingerprint, source_sha256=sources,
                      view_target_sha256=hashlib.sha256(memoryview(target).cast('B')).hexdigest(),
                      stock_helper_args=asdict(helper_args), model_priors=PRIORS, adam_grid=grid,
                      protocol={
                          'primary_output': 'selected context',
                          'controls': 'selected stacking, selected calibration, frozen base static/adaptive/dynamic, zero, original linear/MLP Adam, matched-feature context Adam',
                          'selection': 'Three NIG priors and three Adam learning rates separately minimize real prequential error on [2000,294912); save all six locks before label294912. No suffix selection.',
                          'architecture': 'Residual stacking of same-step prelabel frozen-v6 forecasts, anchored on static mixture. Calibration vs unconstrained linear expert combination vs state-conditioned expert combination; same 128-coordinate embedding and isotropic priors. No new density selection.',
                          'context': 'g=[static,(15 raw forecasts-static)/sqrt(15)]; z=tanh(mean(latest min(8,lags) original seven-feature rows))/sqrt(7); h=[g,outer(z,g)]. No intercept; zero expert forecasts imply zero meta forecasts.',
                          'matched_adam': 'All original eleven learning rates; vanilla Adam on the identical h_context and y-base_static residual, zero coefficients and base_static anchor. Structural feature change, not a drop-in Adam replacement.',
                          'random_sign': 'Fresh base, meta, context Adam, and original Adam states; seed1 signs times real target, identical features, all six real locks inherited, exact source/data checks.',
                          'primary_segment': '[294912,N) extends previous real consumption294912 and null278528; older aggregate market benchmarks covered the file, so this is not pristine never-seen data.',
                          'suffix': 'Continuous predict-before-update learning without resets or suffix choices. Retain all61 columns, including every actual prelabel frozen base forecast for independent reconstruction.',
                          'pruning': 'Only selected context is protected; suffix-local default ProxyCull warmup/patience, error_ratio min_delta1e-4, plus null prediction_energy_ratio1e-4. Censored observed-only evidence, exit75.',
                          'precision': 'CUDA-only compiled blocks16/singletons; FP32 coefficients/covariances and FP64 NIG scalars. TF32 disabled. Both base and meta covariance Cholesky diagnostics; never clamp or repair.',
                          'alignment': 'Unchanged stock helper: row t uses bars t..t+31, target ret[t+33]; newest return two bars old, target normalization includes intervening bar t+32.',
                          'normalization': 'Original trailing scaling/clipping and scaled targets, raw_target=False, vol_feature=False. No stream feature changes.',
                          'limitations': 'One SPY stream and seed; null diagnoses zero conditional mean, not significance or universal optimality.'})
        stock.save_result(root, result)
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise FloatingPointError('stock features or targets are nonfinite')
        device = torch.device('cuda')
        xs, ys = torch.as_tensor(features, device=device), torch.as_tensor(target, device=device)
        model = StateConditionedStack(features.shape[1], device, priors=PRIORS)
        adam = AdamBank(xs, ys, grid)
        runner = StockRunner(xs, ys, model, adam)
        names = output_names(model.output_names, grid)
        if tuple(model.output_names) != MODEL_NAMES:
            raise ValueError('conditional model differs from fixed output contract')
        result.update(output_names=names, input_dim=features.shape[1], hidden=64, model_configs=model.configs)
        policy = ProxyCull(1, {'error_ratio': 1e-4, **(
            {'prediction_energy_ratio': 1e-4} if args.view == 'random_sign' else {})})
        writer = SummaryWriter(str(root))
        result['tensorboard_dir'] = str(root)
        writer.add_text('protocol', json.dumps(result['protocol'], indent=2))
        capture_start = time.perf_counter()
        graphs = runner.capture(16)
        result.update(capture_seconds=time.perf_counter() - capture_start, status='running',
                      capture_steps=[1, 16], capture_complete_state_verified=True)
        stock.save_result(root, result)
        endpoints = sorted(set(range(args.log_every, samples, args.log_every))
                           | {bound for bounds in phases.values() for bound in bounds if bound})
        previous, replay_seconds = 0, 0.0
        for step in endpoints:
            replay_start = time.perf_counter()
            blocks, remainder = divmod(step - previous, 16)
            for _ in range(blocks):
                graphs[16].replay()
            for _ in range(remainder):
                graphs[1].replay()
            torch.cuda.synchronize()
            replay_seconds += time.perf_counter() - replay_start
            consumed = int(runner.index.item())
            result.update(processed=consumed, processed_observations=consumed, replay_seconds=replay_seconds)
            clocks, covariance_health = checkpoint_health(runner)
            result.update(clocks=clocks, covariance_health=covariance_health)
            if any(value != step for value in clocks.values()):
                raise RuntimeError(f'prequential clocks diverged at {step}: {clocks}')
            pred = runner.predictions[previous:step].cpu().numpy()
            scales = torch.cat((model.noise, adam.scale, adam.linear.noise, adam.mlp.noise))
            finite_state = torch.stack([torch.isfinite(tensor).all() for tensor in
                                        (*model.state_tensors(), *adam.linear.mutable,
                                         *(value for value in adam.mlp.mutable if value is not adam.mlp.predictions))]).all()
            if (not np.isfinite(pred).all() or not bool(finite_state)
                    or not bool((torch.isfinite(scales) & (scales > 0)).all())):
                result['nonfinite_failure_interval'] = [previous, step]
                raise FloatingPointError('nonfinite forecasts/state or nonfinite/nonpositive learner scales')
            if any(any(info != 0 for info in row['cholesky_info']) for row in covariance_health.values()):
                raise FloatingPointError('base or meta posterior covariance lost positive definiteness')
            metrics = stock.metric_sums(target[previous:step], pred)
            if any(not math.isfinite(value) for row in metrics for value in row.values()):
                raise FloatingPointError('nonfinite metric; zero target energy is not success')
            if step == prefix:
                if args.view == 'real':
                    result.update(**select_prefix_locks(runner.predictions[:prefix].cpu().numpy(),
                                                         target[:prefix], names, consumed, prefix))
                # Persist all six locks before any replay can consume label294912.
                save_artifacts(root, runner, target, names, phases, consumed, policy, result)
            columns = selected_columns(names, result['model_locks'], result['adam_locks'])
            primary = columns.get('context')
            decision = None
            if args.autocull and previous >= prefix:
                if primary is None:
                    raise RuntimeError('suffix cannot run without its real-prefix locks')
                decision = policy.observe(step, {key: [metrics[primary][key]] for key in policy.metrics},
                                          phase='suffix', phase_start=prefix)
            result['curves'].append({'step': step, 'interval_start': previous,
                                     'prequential': [{'name': name, **row} for name, row in zip(names, metrics)],
                                     'diagnostics': model.diagnostics(), 'covariance_health': covariance_health,
                                     'clocks': clocks, 'autocull': policy.state_dict()})
            result.update(autocull_state=policy.state_dict(), wall_seconds=time.perf_counter() - started)
            for name, row in zip(names, metrics):
                for metric in ('error_ratio', 'prediction_energy_ratio', 'signed_cross_term_ratio'):
                    writer.add_scalar(f'{args.view}/{name}/{metric}', row[metric], step)
            if primary is not None:
                for baseline, column in columns.items():
                    if baseline != 'context':
                        writer.add_scalar(f'{args.view}/context_minus_{baseline}/error_ratio',
                                          metrics[primary]['error_ratio'] - metrics[column]['error_ratio'], step)
            writer.add_scalar('updates_per_second', step / replay_seconds, step)
            writer.flush()
            stock.save_result(root, result)
            primary_ratio = metrics[primary]['error_ratio'] if primary is not None else 'unlocked'
            print(f'PROGRESS view={args.view} consumed={step}/{samples} context_error_ratio={primary_ratio} replay_seconds={replay_seconds:.1f}', flush=True)
            if decision:
                result.update(status='pruned', pruning=decision, exit_code=PRUNED_EXIT_CODE)
                save_artifacts(root, runner, target, names, phases, consumed, policy, result)
                print(f'RESULTS {root / "results.json"} status=pruned consumed={consumed}', flush=True)
                prune_proxy(root, args.view, decision)
            previous = step
        result.update(status='completed', wall_seconds=time.perf_counter() - started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save_artifacts(root, runner, target, names, phases, consumed, policy, result)
        print(f'RESULTS {root / "results.json"} status=completed consumed={consumed}', flush=True)
    except ProxyPruned:
        raise SystemExit(PRUNED_EXIT_CODE) from None
    except Exception as error:
        result.update(status='failed', failure={'type': type(error).__name__, 'message': str(error)},
                      wall_seconds=time.perf_counter() - started)
        if isinstance(error, FloatingPointError):
            result['exit_code'] = PRUNED_EXIT_CODE
        if runner is not None and policy is not None:
            try:
                consumed = int(runner.index.item())
                save_artifacts(root, runner, target, names, phases, consumed, policy, result)
            except Exception as save_error:
                result['artifact_failure'] = {'type': type(save_error).__name__, 'message': str(save_error)}
        stock.save_result(root, result)
        print(f'RESULTS {root / "results.json"} status=failed consumed={consumed}', flush=True)
        if isinstance(error, FloatingPointError):
            raise SystemExit(PRUNED_EXIT_CODE) from None
        raise
    finally:
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    main()
