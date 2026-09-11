"""Mean-preserving dynamic NIG on the unchanged chronological SPY stream.

[237568, N) extends beyond the previous experiment's consumed observations; it
is not pristine data because older aggregate benchmarks covered the full file.
Only the two original Adam learning rates are prefix-selected. All three NIG
mixtures use causal Bayesian evidence, with fixed priors and drift half-lives.
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

from cleanrl.plasticity import covariance_sparse_eval_v1 as sparse
from cleanrl.plasticity import covariance_stock_eval_v1 as stock
from cleanrl.plasticity import network_bayes_stream_v2 as bayes
from cleanrl.plasticity import stock_stream
from cleanrl.plasticity.predictive_correlated_nig_v5 import student_t_log_prob
from cleanrl.plasticity.predictive_dynamic_nig_v6 import DynamicNIG
from cleanrl.plasticity.predictive_stock_transfer_eval_v1 import AdamBank, StockRunner, summarize_predictions
from cleanrl.shared import runtime
from cleanrl.shared.autocull import PRUNED_EXIT_CODE, ProxyCull, ProxyPruned, prune_proxy

SELECTION_END = 237568
COLD_START = 2000
PRIORS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
HALFLIVES = (65536, 262144)
ADAM_GRID = stock.Args.adam_lrs
MODEL_CONFIGS = tuple({'prior': prior, 'halflife': half} for half in (0, *HALFLIVES) for prior in PRIORS)
MODEL_NAMES = ('adaptive_mixture', 'static_mixture', 'dynamic_mixture', 'zero',
               *(f'static_{prior:g}' for prior in PRIORS),
               *(f'discount_{half:g}_{prior:g}' for half in HALFLIVES for prior in PRIORS))


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
        raise ValueError('selection_end must remain the prospectively fixed 237568')
    if type(samples) is not int or samples <= selection_end:
        raise ValueError('stream must extend beyond the fixed selection prefix')
    return {'cold_start': (0, COLD_START), 'selection_prefix': (COLD_START, selection_end),
            'prefix_all': (0, selection_end), 'suffix_all': (selection_end, samples)}


def output_names(model_names=MODEL_NAMES, adam_grid=ADAM_GRID):
    return (*model_names, *(f'adam_linear_{lr:g}' for lr in adam_grid),
            *(f'adam_mlp_{lr:g}' for lr in adam_grid))


def source_fingerprint():
    paths = {__file__, inspect.getfile(DynamicNIG), inspect.getfile(student_t_log_prob),
             inspect.getfile(StockRunner), inspect.getfile(StockRunner.__mro__[1]),
             stock.__file__, sparse.__file__, bayes.__file__, stock_stream.__file__,
             runtime.__file__, inspect.getfile(ProxyCull)}
    return {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sorted(paths)}


def load_real_locks(path, fingerprint, source_sha256, selection_end=SELECTION_END, adam_grid=ADAM_GRID):
    """Transfer only verified real-prefix Adam locks, never choose on null labels."""
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
                or source['model_configs'] != list(MODEL_CONFIGS)
                or source['output_names'] != list(output_names())):
            raise ValueError('incompatible fixed model configuration or original Adam grid')
        locks = source['adam_locks']
        if set(locks) != {'linear', 'mlp'}:
            raise ValueError('both original Adam locks are required')
        shared_target_sum = None
        for family, lock in locks.items():
            for field, expected in (('selection_start_inclusive', COLD_START),
                                    ('selection_end_exclusive', selection_end),
                                    ('optimizer_updates_at_lock', selection_end),
                                    ('suffix_observations_used', 0)):
                if type(lock[field]) is not int or lock[field] != expected:
                    raise ValueError(f'{family} lock is not causal: {field}')
            if (lock['criterion'] != 'minimum prequential squared error / zero-predictor squared error'
                    or lock['tie_break'] != 'first (smallest) learning rate'):
                raise ValueError('incompatible prefix selection rule')
            index = lock['selected_index']
            if (type(index) is not int or not 0 <= index < len(adam_grid)
                    or lock['selected_lr'] != adam_grid[index]):
                raise ValueError('selected index/rate does not match grid')
            if len(lock['candidates']) != len(adam_grid):
                raise ValueError('candidate count does not match grid')
            scores = []
            for candidate, rate in zip(lock['candidates'], adam_grid):
                if (candidate['lr'] != rate or type(candidate['count']) is not int
                        or candidate['count'] != selection_end - COLD_START):
                    raise ValueError('candidate rate/count does not match prefix')
                keys = ('target_squared_sum', 'prediction_squared_sum', 'target_prediction_sum',
                        'error_squared_sum', 'error_ratio', 'prediction_energy_ratio',
                        'signed_cross_term_ratio', 'decomposition_residual')
                if any(not math.isfinite(candidate[key]) for key in keys):
                    raise ValueError('nonfinite prefix candidate evidence')
                yy, pp, yp, ee = (candidate[key] for key in keys[:4])
                if yy <= 0 or pp < 0 or ee < 0:
                    raise ValueError('invalid squared sums')
                if shared_target_sum is None:
                    shared_target_sum = yy
                if yy != shared_target_sum:
                    raise ValueError('all candidates must share identical real-prefix targets')
                expected = {'error_ratio': ee / yy, 'prediction_energy_ratio': pp / yy,
                            'signed_cross_term_ratio': -2 * yp / yy,
                            'decomposition_residual': (ee - yy - pp + 2 * yp) / yy}
                if (not math.isclose(ee, yy + pp - 2 * yp, rel_tol=1e-10, abs_tol=1e-10 * yy)
                        or yp * yp > yy * pp + 1e-10 * yy * max(yy, pp)
                        or any(not math.isclose(candidate[key], value, rel_tol=1e-10, abs_tol=1e-12)
                               for key, value in expected.items())):
                    raise ValueError('candidate sums and normalized metrics disagree')
                scores.append(candidate['error_ratio'])
            if index != min(range(len(scores)), key=scores.__getitem__):
                raise ValueError('fabricated winner or invalid first-candidate tie break')
        return {'adam_locks': locks}
    except (KeyError, TypeError, IndexError, OverflowError, json.JSONDecodeError) as error:
        raise ValueError(f'invalid real lock provenance: {error}') from error


def select_adam_prefix(predictions, target, names, consumed, selection_end=SELECTION_END,
                       adam_grid=ADAM_GRID):
    """The only selector: exact real-prefix barrier, with model-width-aware slices."""
    if (type(selection_end) is not int or selection_end != SELECTION_END
            or type(consumed) is not int or consumed != selection_end
            or len(predictions) != selection_end or len(target) != selection_end):
        raise ValueError('Adam selection requires the exact 237568 barrier before any suffix label')
    width = len(names) - 2 * len(adam_grid)
    return {
        family: stock.select_prefix(predictions[:, offset:offset + len(adam_grid)],
                                    target, adam_grid, COLD_START, selection_end, consumed)
        for family, offset in (('linear', width), ('mlp', width + len(adam_grid)))}


def selected_columns(names, adam_locks, adam_grid=ADAM_GRID):
    """Fixed mixtures by semantic name; Adam offset follows actual model width."""
    columns = {name: names.index(name) for name in MODEL_NAMES[:4]}
    if adam_locks:
        width = len(names) - 2 * len(adam_grid)
        columns.update(adam_linear=width + adam_locks['linear']['selected_index'],
                       adam_mlp=width + len(adam_grid) + adam_locks['mlp']['selected_index'])
    return columns


def paired_comparisons(summary, names, columns):
    comparisons = {}
    for phase, row in summary.items():
        metrics = row['metrics']
        primary = columns['adaptive_mixture']
        comparisons[phase] = {'count': row['count'], 'complete': row['complete'],
                              'observed_range': row['observed_range'], 'comparisons': [
            {'name': names[primary], 'role': 'adaptive_mixture', 'baseline': baseline,
             'baseline_name': names[column],
             'error_ratio_difference': metrics[primary]['error_ratio'] - metrics[column]['error_ratio'],
             'mse_difference': (metrics[primary]['error_squared_sum']
                                - metrics[column]['error_squared_sum']) / row['count']}
            for baseline, column in columns.items() if baseline != 'adaptive_mixture'] if metrics else []}
    return comparisons


def save_artifacts(root, runner, target, names, phases, consumed, policy, result):
    prediction = runner.predictions[:consumed].cpu().numpy()
    # Preserve nonfinite observed evidence, rather than sanitizing stored forecasts.
    np.save(root / 'predictions.npy', prediction)
    np.save(root / 'targets.npy', target[:consumed])
    adam_state = [tensor.cpu() for tensor in runner.adam.mutable
                  if tensor is not runner.adam.mlp.predictions]
    temporary = root / 'checkpoint.pt.tmp'
    torch.save({'step': consumed, 'runner_index': runner.index.cpu(),
                'model_state': [tensor.cpu() for tensor in runner.model.state_tensors()],
                'model_configs': runner.model.configs, 'adam_state': adam_state,
                'adam_configs': runner.adam.configs, 'autocull': policy.state_dict(),
                'note': 'Complete learner state including covariance; observed trajectories saved separately. No resume CLI.'},
               temporary)
    temporary.replace(root / 'checkpoint.pt')
    summary = summarize_predictions(target, prediction, names, phases, consumed)
    columns = selected_columns(names, result.get('adam_locks'), result['adam_grid'])
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
              'processed': 0, 'processed_observations': 0, 'curves': [], 'adam_locks': None}
    stock.save_result(root, result)
    print(f'RESULTS {root / "results.json"}', flush=True)
    runner, writer, target, policy = None, None, None, None
    consumed = 0
    started = time.perf_counter()
    try:
        if args.seed != 1 or args.log_every <= 0 or args.selection_end != SELECTION_END:
            raise ValueError('seed1, fixed selection_end=237568, and positive log cadence required')
        if (args.view == 'random_sign') != bool(args.real_result):
            raise ValueError('random_sign requires --real-result; real selects its own Adam prefix')
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
                      stock_helper_args=asdict(helper_args), model_configs=MODEL_CONFIGS, adam_grid=grid,
                      protocol={
                          'primary_output': 'adaptive_mixture', 'control_output': 'static_mixture',
                          'secondary_output': 'dynamic_mixture',
                          'selection': 'Only original linear and hidden64 MLP Adam LRs minimize real prequential error on [2000,237568); saved at exact barrier before label237568. No prior, half-life, or mixture row selection.',
                          'mixtures': 'Fixed causal Bayesian weights: adaptive null/static/dynamic mass .5/.25/.25; static and dynamic-only each .5 null/.5 experts, uniform within families. Shared expert trajectories, independent evidence vectors.',
                          'drift': 'Conditional-scale DLM: mean unchanged before observing label; Pminus=P/delta, delta=2**(-1/H). Process covariance (delta^-1-1)*P depends on past feature geometry. Not a fixed-process-Q posterior. Global sigma evidence accumulates innovations, not discounted residual sums.',
                          'random_sign': 'Seed1 Rademacher signs times real targets; identical features and real-target hashes; same fixed model/source; only two real Adam locks transferred.',
                          'primary_segment': '[237568,N) extends prior actual experiment consumption; older aggregate market benchmarks covered the file, so this is not pristine never-seen data.',
                          'suffix': 'Continuous predict-before-update learning, no resets and no suffix choices. Report every mixture and raw expert plus both full original Adam grids.',
                          'pruning': 'One protected candidate: adaptive_mixture. Suffix-local default ProxyCull warmup/patience, error_ratio min_delta=1e-4, plus null prediction_energy_ratio=1e-4. Censored partial evidence, exit75.',
                          'precision': 'CUDA-only compiled blocks16/singletons; FP32 matrices, FP64 scalar scales/evidence, TF32 disabled; checkpoint covariance Cholesky diagnostics.',
                          'alignment': 'Unchanged stock helper: row t uses bars t..t+31, target ret[t+33]; newest return two bars old, target normalization includes intervening bar t+32.',
                          'normalization': 'Original trailing scaling/clipping and scaled targets; raw_target=False, vol_feature=False. No feature modifications.',
                          'limitations': 'One SPY stream and seed; null diagnoses zero conditional mean, not significance or universal optimality.'})
        stock.save_result(root, result)
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise FloatingPointError('stock features or targets are nonfinite')
        device = torch.device('cuda')
        xs, ys = torch.as_tensor(features, device=device), torch.as_tensor(target, device=device)
        model = DynamicNIG(features.shape[1], device, priors=PRIORS, halflives=HALFLIVES)
        adam = AdamBank(xs, ys, grid)
        runner = StockRunner(xs, ys, model, adam)
        names = output_names(model.output_names, grid)
        if tuple(model.configs) != MODEL_CONFIGS or tuple(model.output_names) != MODEL_NAMES:
            raise ValueError('dynamic model differs from fixed experiment contract')
        result.update(output_names=names, input_dim=features.shape[1], hidden=64)
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
            clocks = {'runner': consumed, 'model': int(model.observations.item()),
                      'linear_index': int(adam.linear.index.item()), 'linear_steps': int(adam.linear.steps.item()),
                      'mlp_index': int(adam.mlp.index.item()), 'mlp_steps': int(adam.mlp.steps.item())}
            result.update(processed=consumed, processed_observations=consumed, clocks=clocks,
                          replay_seconds=replay_seconds)
            if any(value != step for value in clocks.values()):
                raise RuntimeError(f'prequential clocks diverged at {step}: {clocks}')
            pred = runner.predictions[previous:step].cpu().numpy()
            scales = torch.cat((model.noise, adam.scale, adam.linear.noise, adam.mlp.noise))
            finite_state = torch.stack([torch.isfinite(tensor).all() for tensor in model.state_tensors()]).all()
            if (not np.isfinite(pred).all() or not bool(finite_state)
                    or not bool((torch.isfinite(scales) & (scales > 0)).all())):
                result['nonfinite_failure_interval'] = [previous, step]
                raise FloatingPointError('nonfinite forecasts/state or nonfinite/nonpositive learner scales')
            _, cholesky_info = torch.linalg.cholesky_ex(model.cov, check_errors=False)
            covariance_health = {'cholesky_info': cholesky_info.cpu().tolist(),
                                 'max_asymmetry': float((model.cov - model.cov.mT).abs().amax().item())}
            result['covariance_health'] = covariance_health
            if bool((cholesky_info != 0).any()):
                raise FloatingPointError('posterior covariance lost positive definiteness')
            metrics = stock.metric_sums(target[previous:step], pred)
            if any(not math.isfinite(value) for row in metrics for value in row.values()):
                raise FloatingPointError('nonfinite metric; zero target energy is not success')
            if step == prefix:
                if args.view == 'real':
                    prefix_predictions = runner.predictions[:prefix].cpu().numpy()
                    result['adam_locks'] = select_adam_prefix(
                        prefix_predictions, target[:prefix], names, consumed, prefix, grid)
                # Persist exact prefix evidence BEFORE any graph can consume label237568.
                save_artifacts(root, runner, target, names, phases, consumed, policy, result)
            columns = selected_columns(names, result['adam_locks'], grid)
            decision = None
            if args.autocull and previous >= prefix:
                primary = columns['adaptive_mixture']
                decision = policy.observe(step, {key: [metrics[primary][key]] for key in policy.metrics},
                                          phase='suffix', phase_start=prefix)
            result['curves'].append({'step': step, 'interval_start': previous,
                                     'prequential': [{'name': name, **row} for name, row in zip(names, metrics)],
                                     'diagnostics': model.diagnostics(), 'covariance_health': covariance_health,
                                     'autocull': policy.state_dict()})
            result.update(autocull_state=policy.state_dict(), wall_seconds=time.perf_counter() - started)
            for name, row in zip(names, metrics):
                for metric in ('error_ratio', 'prediction_energy_ratio', 'signed_cross_term_ratio'):
                    writer.add_scalar(f'{args.view}/{name}/{metric}', row[metric], step)
            for baseline, column in columns.items():
                if baseline != 'adaptive_mixture':
                    writer.add_scalar(f'{args.view}/adaptive_minus_{baseline}/error_ratio',
                                      metrics[columns['adaptive_mixture']]['error_ratio']
                                      - metrics[column]['error_ratio'], step)
            writer.add_scalar('updates_per_second', step / replay_seconds, step)
            writer.flush()
            stock.save_result(root, result)
            primary_ratio = metrics[columns['adaptive_mixture']]['error_ratio']
            print(f'PROGRESS view={args.view} consumed={step}/{samples} adaptive_error_ratio={primary_ratio} replay_seconds={replay_seconds:.1f}', flush=True)
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
