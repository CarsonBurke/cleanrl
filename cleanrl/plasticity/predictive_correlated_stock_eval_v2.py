"""Dense NIG stock experiment with one fixed, real-prefix hyperparameter lock.

[163840, N) is a forward extension beyond the preceding v3/v4 transfer's
consumption, not a pristine dataset: older Adam/covariance summaries covered the
full file. Every reported forecast is predict-before-update. Only the real
[2000, 163840) prefix selects priors and Adam rates; random signs inherit locks.
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
from cleanrl.plasticity.predictive_correlated_nig_v5 import CorrelatedNIG
from cleanrl.plasticity.predictive_stock_transfer_eval_v1 import AdamBank, StockRunner, summarize_predictions
from cleanrl.shared import runtime
from cleanrl.shared.autocull import PRUNED_EXIT_CODE, ProxyCull, ProxyPruned, prune_proxy

SELECTION_END = 163840
COLD_START = 2000
PRIORS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
ADAM_GRID = stock.Args.adam_lrs


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
        raise ValueError('selection_end must remain the prospectively fixed 163840')
    if type(samples) is not int or samples <= selection_end:
        raise ValueError('stream must extend beyond the fixed selection prefix')
    return {'cold_start': (0, COLD_START), 'selection_prefix': (COLD_START, selection_end),
            'prefix_all': (0, selection_end), 'suffix_all': (selection_end, samples)}


def select_model_prefix(predictions, target, priors, family, selection_end, consumed):
    """Use the stock risk selector, exposing prior variance rather than a fake LR.

    Model locks retain the stock timing, criterion, candidate metric sums, and
    selected_index fields. They rename selected_lr to selected_prior and each
    candidate's lr to prior, add candidate_type, and use a prior-specific tie rule.
    """
    if family not in ('dense', 'diagonal') or selection_end != SELECTION_END:
        raise ValueError('model selection requires dense/diagonal at the fixed prefix')
    lock = stock.select_prefix(predictions, target, priors, COLD_START, selection_end, consumed)
    lock['selected_prior'] = lock.pop('selected_lr')
    lock['candidate_type'] = family
    lock['tie_break'] = 'first (smallest) prior'
    for candidate in lock['candidates']:
        candidate['prior'] = candidate.pop('lr')
    return lock


def load_real_locks(path, fingerprint, selection_end, priors, adam_grid):
    """Validate and return model_locks{dense,diagonal}, adam_locks{linear,mlp}.

    Accept a completed, prefix-barrier running, or intentionally pruned real
    artifact. Failed runs and any boundary/grid/hash/score/winner mismatch fail
    closed. No target labels from the receiving random-sign view are consulted.
    """
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
        samples = source['samples']
        phases = phase_ranges(samples, selection_end)
        if source['phases'] != {name: list(bounds) for name, bounds in phases.items()}:
            raise ValueError('real phases do not match fixed selection boundary')
        consumed = source['processed_observations']
        if type(consumed) is not int or not selection_end <= consumed <= samples:
            raise ValueError('real artifact has not consumed the exact selection prefix')
        if source['status'] == 'completed' and consumed != samples:
            raise ValueError('completed artifact has an unconsumed suffix')
        if source['maximum_observations'] != samples:
            raise ValueError('inconsistent maximum horizon')
        if (tuple(priors) != PRIORS or tuple(adam_grid) != ADAM_GRID
                or source['model_priors'] != list(priors) or source['adam_grid'] != list(adam_grid)):
            raise ValueError('incompatible model prior or original Adam grid')
        model_locks, adam_locks = source['model_locks'], source['adam_locks']
        if set(model_locks) != {'dense', 'diagonal'} or set(adam_locks) != {'linear', 'mlp'}:
            raise ValueError('both model and both original Adam locks are required')
        shared_target_sum = None
        for family, lock in (*model_locks.items(), *adam_locks.items()):
            is_model = family in model_locks
            grid = priors if is_model else adam_grid
            parameter = 'prior' if is_model else 'lr'
            if is_model and lock['candidate_type'] != family:
                raise ValueError('incorrect model candidate type')
            for field, expected in (('selection_start_inclusive', COLD_START),
                                    ('selection_end_exclusive', selection_end),
                                    ('optimizer_updates_at_lock', selection_end),
                                    ('suffix_observations_used', 0)):
                if type(lock[field]) is not int or lock[field] != expected:
                    raise ValueError(f'{family} lock is not causal: {field}')
            if (lock['criterion'] != 'minimum prequential squared error / zero-predictor squared error'
                    or lock['tie_break'] != ('first (smallest) prior' if is_model
                                             else 'first (smallest) learning rate')):
                raise ValueError('incompatible prefix selection rule')
            index = lock['selected_index']
            if (type(index) is not int or not 0 <= index < len(grid)
                    or lock[f'selected_{parameter}'] != grid[index]):
                raise ValueError('selected index/parameter does not match grid')
            if len(lock['candidates']) != len(grid):
                raise ValueError('candidate count does not match grid')
            scores = []
            for candidate, value in zip(lock['candidates'], grid):
                if (candidate[parameter] != value or type(candidate['count']) is not int
                        or candidate['count'] != selection_end - COLD_START):
                    raise ValueError('candidate parameter/count does not match prefix')
                metric_keys = ('target_squared_sum', 'prediction_squared_sum', 'target_prediction_sum',
                               'error_squared_sum', 'error_ratio', 'prediction_energy_ratio',
                               'signed_cross_term_ratio', 'decomposition_residual')
                if any(not math.isfinite(candidate[key]) for key in metric_keys):
                    raise ValueError('nonfinite prefix candidate evidence')
                yy, pp, yp, ee = (candidate[key] for key in metric_keys[:4])
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
                        or any(not math.isclose(candidate[key], val, rel_tol=1e-10, abs_tol=1e-12)
                               for key, val in expected.items())):
                    raise ValueError('candidate sums and normalized metrics disagree')
                scores.append(candidate['error_ratio'])
            if index != min(range(len(scores)), key=scores.__getitem__):
                raise ValueError('fabricated winner or invalid first-candidate tie break')
        return {'model_locks': model_locks, 'adam_locks': adam_locks}
    except (KeyError, TypeError, IndexError, OverflowError, json.JSONDecodeError) as error:
        raise ValueError(f'invalid real lock provenance: {error}') from error


def selected_columns(names, model_locks, adam_locks, priors, adam_grid):
    """Resolve model by semantic name and Adam offsets by actual model width."""
    model_width = len(names) - 2 * len(adam_grid)
    return {'dense': names.index(f'dense_{priors[model_locks["dense"]["selected_index"]]:g}'),
            'diagonal': names.index(f'diagonal_{priors[model_locks["diagonal"]["selected_index"]]:g}'),
            'dense_bayes_mixture': names.index('dense_bayes_mixture'),
            'zero': names.index('zero'),
            'adam_linear': model_width + adam_locks['linear']['selected_index'],
            'adam_mlp': model_width + len(adam_grid) + adam_locks['mlp']['selected_index']}


def paired_comparisons(summary, names, columns):
    comparisons = {}
    for phase, row in summary.items():
        metrics = row['metrics']
        comparisons[phase] = {'count': row['count'], 'complete': row['complete'], 'comparisons': [
            {'name': names[columns[method]], 'role': method,
             'baseline': baseline, 'baseline_name': names[columns[baseline]],
             'error_ratio_difference': (metrics[columns[method]]['error_ratio']
                                        - metrics[columns[baseline]]['error_ratio']),
             'mse_difference': (metrics[columns[method]]['error_squared_sum']
                                - metrics[columns[baseline]]['error_squared_sum']) / row['count']}
            for method in ('dense', 'diagonal', 'dense_bayes_mixture')
            for baseline in ('zero', 'adam_linear', 'adam_mlp')] if metrics else []}
    return comparisons


def save_artifacts(root, runner, target, names, phases, consumed, policy, result):
    prediction = runner.predictions[:consumed].cpu().numpy()
    # Preserve raw nonfinite evidence on failure; JSON's finite conversion is only
    # presentation, never a replacement forecast or a successful metric.
    np.save(root / 'predictions.npy', prediction)
    np.save(root / 'targets.npy', target[:consumed])
    adam_state = [tensor.cpu() for tensor in runner.adam.mutable
                  if tensor is not runner.adam.mlp.predictions]
    temporary = root / 'checkpoint.pt.tmp'
    torch.save({'step': consumed, 'runner_index': runner.index.cpu(),
                'model_state': [tensor.cpu() for tensor in runner.model.state_tensors()],
                'model_configs': runner.model.configs, 'adam_state': adam_state,
                'adam_configs': runner.adam.configs, 'autocull': policy.state_dict(),
                'note': 'Complete learner state; full covariance included; trajectories saved separately. No resume CLI.'},
               temporary)
    temporary.replace(root / 'checkpoint.pt')
    summary = summarize_predictions(target, prediction, names, phases, consumed)
    result.update(processed_observations=consumed, phase_metrics=summary,
                  prediction_artifact='predictions.npy', target_artifact='targets.npy',
                  checkpoint_artifact='checkpoint.pt',
                  nonfinite_predictions=int(np.count_nonzero(~np.isfinite(prediction))),
                  autocull_state=policy.state_dict())
    if result.get('model_locks') and result.get('adam_locks'):
        columns = selected_columns(names, result['model_locks'], result['adam_locks'],
                                   result['model_priors'], result['adam_grid'])
        result['selected_columns'] = columns
        result['paired_comparisons'] = paired_comparisons(summary, names, columns)
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
              'processed_observations': 0, 'curves': [], 'model_locks': None, 'adam_locks': None}
    stock.save_result(root, result)
    print(f'RESULTS {root / "results.json"}', flush=True)
    runner, writer, target, policy = None, None, None, None
    consumed = 0
    started = time.perf_counter()
    try:
        if args.seed != 1 or args.log_every <= 0 or args.selection_end != SELECTION_END:
            raise ValueError('seed1, fixed selection_end=163840, and positive log cadence required')
        if (args.view == 'random_sign') != bool(args.real_result):
            raise ValueError('random_sign requires --real-result; real selects its own prefix')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU learner fallback')
        runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
        helper_args = stock_stream.Args(seed=1, steps=0)
        bars = stock_stream.read_bars(args.bars)
        features, real_target = stock_stream.build_stream(bars, helper_args)
        samples = len(real_target)
        phases = phase_ranges(samples, args.selection_end)
        prefix = args.selection_end
        grid, priors = ADAM_GRID, PRIORS
        fingerprint = {'features': hashlib.sha256(memoryview(features).cast('B')).hexdigest(),
                       'target': hashlib.sha256(memoryview(real_target).cast('B')).hexdigest()}
        target = real_target
        if args.view == 'random_sign':
            payload = Path(args.real_result).read_bytes()
            locks = load_real_locks(args.real_result, fingerprint, prefix, priors, grid)
            if payload != Path(args.real_result).read_bytes():
                raise ValueError('real lock artifact changed while loading')
            if json.loads(payload)['samples'] != samples:
                raise ValueError('real lock horizon differs from current stream')
            result.update(**locks, transferred_lock_sha256=hashlib.sha256(payload).hexdigest())
            signs = np.random.default_rng(1).integers(0, 2, size=samples, dtype=np.int8) * 2 - 1
            target = real_target * signs
        source_paths = {__file__, inspect.getfile(CorrelatedNIG), inspect.getfile(StockRunner),
                        inspect.getfile(StockRunner.__mro__[1]), stock.__file__, sparse.__file__,
                        bayes.__file__, stock_stream.__file__, runtime.__file__, inspect.getfile(ProxyCull)}
        result.update(samples=samples, maximum_observations=samples, bars=len(bars),
                      phases=phases, data_sha256=fingerprint,
                      view_target_sha256=hashlib.sha256(memoryview(target).cast('B')).hexdigest(),
                      stock_helper_args=asdict(helper_args), model_priors=priors, adam_grid=grid,
                      source_sha256={Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                     for path in sorted(source_paths)},
                      protocol={
                          'primary_output': 'real-prefix-selected dense prior',
                          'ablation': 'separately real-prefix-selected diagonal prior; approximate factorized posterior, not exact dense regression',
                          'secondary_output': 'dense_bayes_mixture; static Bayesian dense-plus-null evidence, not mean-risk selection',
                          'selection': 'All candidates learn on the full real prefix; minimum prequential noisy MSE on [2000,163840) locks dense prior, diagonal prior, linear Adam LR, MLP Adam LR before any suffix replay.',
                          'random_sign': 'Seed1 independent Rademacher signs times real targets; unchanged features; all real locks transferred, no null selection.',
                          'primary_segment': '[163840,N), a forward extension beyond preceding v3/v4 transfer consumption; older Adam/covariance summary covered the full file, so not pristine never-seen data.',
                          'suffix': 'Continued chronological predict-before-update learning; no suffix hyperparameter selection. Compare selected dense against zero AND both selected original Adam families.',
                          'pruning': 'Only selected dense and fixed dense Bayesian mixture protect the run; suffix-local default ProxyCull, error_ratio min_delta=1e-4 plus null prediction_energy_ratio=1e-4. Pruned evidence is censored; exit75.',
                          'precision': 'FP32 state matrices, float64 NIG scales/evidence, highest matmul precision and TF32 off; compiled CUDA blocks16 plus singleton tails.',
                          'alignment': 'Original stock helper unchanged: row t bars t..t+31, target ret[t+33]; newest return two bars old; target centering/volatility includes intervening bar t+32.',
                          'normalization': 'Original trailing channel scaling/clipping and scaled targets, raw_target=False, vol_feature=False; no feature changes.',
                          'limitations': 'One SPY stream and seed; random signs diagnose zero conditional mean, not financial significance or universal optimality.'})
        stock.save_result(root, result)
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise FloatingPointError('stock features or targets are nonfinite')
        device = torch.device('cuda')
        xs, ys = torch.as_tensor(features, device=device), torch.as_tensor(target, device=device)
        model = CorrelatedNIG(features.shape[1], device, priors=priors)
        adam = AdamBank(xs, ys, grid)
        runner = StockRunner(xs, ys, model, adam)
        model_width = len(model.output_names)
        names = (*model.output_names, *(f'adam_linear_{lr:g}' for lr in grid),
                 *(f'adam_mlp_{lr:g}' for lr in grid))
        result.update(output_names=names, expert_configs=model.configs, input_dim=features.shape[1], hidden=64)
        policy = ProxyCull(2, {'error_ratio': 1e-4, **(
            {'prediction_energy_ratio': 1e-4} if args.view == 'random_sign' else {})})
        log_dir = root
        writer = SummaryWriter(str(log_dir))
        result['tensorboard_dir'] = str(log_dir)
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
            result.update(processed_observations=consumed, clocks=clocks, replay_seconds=replay_seconds)
            if any(value != step for value in clocks.values()):
                raise RuntimeError(f'prequential clocks diverged at {step}: {clocks}')
            pred = runner.predictions[previous:step].cpu().numpy()
            scales = torch.cat((model.noise, adam.scale, adam.linear.noise, adam.mlp.noise))
            finite_state = torch.stack([torch.isfinite(tensor).all() for tensor in model.state_tensors()]).all()
            if (not np.isfinite(pred).all() or not bool(finite_state)
                    or not bool((torch.isfinite(scales) & (scales > 0)).all())):
                result['nonfinite_failure_interval'] = [previous, step]
                raise FloatingPointError('nonfinite forecasts/state or nonfinite/nonpositive learner scales')
            metrics = stock.metric_sums(target[previous:step], pred)
            if any(not math.isfinite(value) for row in metrics for value in row.values()):
                raise FloatingPointError('nonfinite metric; zero target energy is not success')
            if step == prefix:
                if args.view == 'real':
                    prefix_predictions = runner.predictions[:prefix].cpu().numpy()
                    result['model_locks'] = {
                        family: select_model_prefix(prefix_predictions[:, offset:offset + len(priors)],
                                                    target[:prefix], priors, family, prefix, consumed)
                        for family, offset in (('dense', 0), ('diagonal', len(priors)))}
                    result['adam_locks'] = {
                        family: stock.select_prefix(prefix_predictions[:, offset:offset + len(grid)],
                                                    target[:prefix], grid, COLD_START, prefix, consumed)
                        for family, offset in (('linear', model_width), ('mlp', model_width + len(grid)))}
                # Atomic results replacement plus observed arrays/checkpoint precede
                # ANY new-segment replay, including in the lock-inheriting null run.
                save_artifacts(root, runner, target, names, phases, consumed, policy, result)
            decision = None
            columns = None
            if step >= prefix:
                columns = selected_columns(names, result['model_locks'], result['adam_locks'], priors, grid)
            if args.autocull and previous >= prefix:
                protected = (columns['dense'], columns['dense_bayes_mixture'])
                decision = policy.observe(step, {key: [metrics[i][key] for i in protected]
                                                 for key in policy.metrics},
                                          phase='suffix', phase_start=prefix)
            result['curves'].append({'step': step, 'interval_start': previous,
                                     'prequential': [{'name': name, **row} for name, row in zip(names, metrics)],
                                     'diagnostics': model.diagnostics(), 'autocull': policy.state_dict()})
            result.update(autocull_state=policy.state_dict(), wall_seconds=time.perf_counter() - started)
            for name, row in zip(names, metrics):
                for metric in ('error_ratio', 'prediction_energy_ratio', 'signed_cross_term_ratio'):
                    writer.add_scalar(f'{args.view}/{name}/{metric}', row[metric], step)
            if columns:
                for baseline in ('zero', 'adam_linear', 'adam_mlp'):
                    writer.add_scalar(f'{args.view}/selected_dense_minus_{baseline}/error_ratio',
                                      metrics[columns['dense']]['error_ratio']
                                      - metrics[columns[baseline]]['error_ratio'], step)
            writer.add_scalar('updates_per_second', step / replay_seconds, step)
            writer.flush()
            stock.save_result(root, result)
            primary = metrics[columns['dense']]['error_ratio'] if columns else None
            print(f'PROGRESS view={args.view} consumed={step}/{samples} selected_dense_error_ratio={primary} replay_seconds={replay_seconds:.1f}', flush=True)
            if decision:
                result.update(status='pruned', pruning=decision)
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
        if runner is not None and policy is not None:
            try:
                consumed = int(runner.index.item())
                save_artifacts(root, runner, target, names, phases, consumed, policy, result)
            except Exception as save_error:
                result['artifact_failure'] = {'type': type(save_error).__name__, 'message': str(save_error)}
        stock.save_result(root, result)
        print(f'RESULTS {root / "results.json"} status=failed', flush=True)
        raise
    finally:
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    main()
