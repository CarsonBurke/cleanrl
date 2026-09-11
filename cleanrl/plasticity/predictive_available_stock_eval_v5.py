"""One additional available bar, unchanged targets, frozen paired NIG and Adam.

The forecast timestamp is after bar t+32: original inputs end at t+31,
latest inputs end at t+32, and BOTH predict the original normalized ret[t+33].
This tests information availability, not a new target or an optimizer-only win.
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
from cleanrl.plasticity.predictive_available_state_v8 import (
    AvailableStateComparison, PairedAdamBank, build_paired_stream,
)
from cleanrl.plasticity.predictive_correlated_nig_v5 import CorrelatedNIG
from cleanrl.plasticity.predictive_stock_transfer_eval_v1 import StockRunner, summarize_predictions
from cleanrl.shared import runtime
from cleanrl.shared.autocull import PRUNED_EXIT_CODE, ProxyCull, ProxyPruned, prune_proxy

SELECTION_END = 335872
COLD_START = 2000
ADAM_GRID = stock.Args.adam_lrs
ADAM_FAMILIES = ('original_linear', 'original_mlp', 'latest_linear', 'latest_mlp')
PRIORS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
BASE_NAMES = (*(f'{family}_{prior:g}' for family in ('dense', 'diagonal') for prior in PRIORS),
              'dense_bayes_mixture', 'zero')
MODEL_NAMES = tuple(f'{frame}_{name}' for frame in ('original', 'latest') for name in BASE_NAMES)
PRIMARY = 'latest_dense_bayes_mixture'
DATA_KEYS = ('features', 'original_features', 'latest_features', 'target')


@dataclass
class Args:
    output_dir: str
    reference_result: str
    view: Literal['real', 'random_sign'] = 'real'
    bars: str = stock_stream.Args.bars
    seed: int = 1
    selection_end: int = SELECTION_END
    log_every: int = 8192
    autocull: bool = True
    real_result: str = ''


def phase_ranges(samples, selection_end=SELECTION_END):
    if type(selection_end) is not int or selection_end != SELECTION_END:
        raise ValueError('selection_end must remain the prospectively fixed 335872')
    if type(samples) is not int or samples <= selection_end:
        raise ValueError('stream must extend beyond the fixed selection prefix')
    return {'cold_start': (0, COLD_START), 'selection_prefix': (COLD_START, selection_end),
            'prefix_all': (0, selection_end), 'suffix_all': (selection_end, samples)}


def output_names(model_names=MODEL_NAMES, adam_grid=ADAM_GRID, *, include_diagnostic=True):
    learned = (*model_names, *(f'{frame}_adam_{family}_{lr:g}'
                               for frame in ('original', 'latest')
                               for family in ('linear', 'mlp') for lr in adam_grid))
    return (*learned, 'raw_zero_transformed') if include_diagnostic else learned


def array_sha256(value):
    return hashlib.sha256(memoryview(value).cast('B')).hexdigest()


def source_fingerprint():
    paths = {__file__, inspect.getfile(AvailableStateComparison), inspect.getfile(CorrelatedNIG),
             inspect.getfile(StockRunner), inspect.getfile(StockRunner.__mro__[1]),
             stock.__file__, sparse.__file__, bayes.__file__, stock_stream.__file__,
             runtime.__file__, inspect.getfile(ProxyCull)}
    return {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sorted(paths)}


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def verify_reference(path, original_features, target, stock_stream_sha256):
    """Verify the full frozen input/target horizon before any CUDA initialization."""
    try:
        payload = Path(path).read_bytes()
        reference = json.loads(payload)
        if (reference['args']['view'] != 'real' or reference['args']['seed'] != 1
                or reference['status'] not in ('completed', 'pruned')
                or reference['processed_observations'] != SELECTION_END
                or reference['processed'] != SELECTION_END
                or 'predictive_conditional_stack_v7.py' not in reference['source_sha256']):
            raise ValueError('reference must be the real seed1 v7 artifact consumed through 335872')
        samples = len(target)
        if (original_features.dtype != np.float32 or target.dtype != np.float32
                or original_features.shape != (samples, 224) or target.ndim != 1
                or type(reference['samples']) is not int or reference['samples'] != samples
                or reference['maximum_observations'] != samples):
            raise ValueError('reference full-file sample count or FP32 original feature shape differs')
        phase_ranges(samples)
        for key, value in (('features', original_features), ('target', target)):
            if (not valid_digest(reference['data_sha256'][key])
                    or array_sha256(value) != reference['data_sha256'][key]):
                raise ValueError(f'reference full-file {key} SHA256 mismatch; target/task changes forbidden')
        if (not valid_digest(stock_stream_sha256)
                or reference['source_sha256']['stock_stream.py'] != stock_stream_sha256):
            raise ValueError('frozen stock helper source SHA256 mismatch')
        for key, value in (('lags', 32), ('steps', 0), ('center', True),
                           ('raw_target', False), ('vol_feature', False), ('vol_span', .01)):
            if reference['stock_helper_args'][key] != value:
                raise ValueError(f'reference stock helper configuration differs: {key}')
        return {'artifact_sha256': hashlib.sha256(payload).hexdigest(), 'samples': samples,
                'processed_observations': reference['processed_observations'],
                'data_sha256': reference['data_sha256'], 'source_sha256': reference['source_sha256'],
                'original_features_bitwise_verified': True, 'target_bitwise_verified': True}
    except (KeyError, TypeError, IndexError, json.JSONDecodeError) as error:
        raise ValueError(f'invalid reference provenance: {error}') from error


def candidate_columns(names, family, grid=ADAM_GRID):
    if len(set(names)) != len(names):
        raise ValueError('output names must be unique')
    frame, kind = family.split('_')
    return [names.index(f'{frame}_adam_{kind}_{lr:g}') for lr in grid]


def select_prefix_locks(predictions, target, names, consumed, selection_end=SELECTION_END,
                        adam_grid=ADAM_GRID):
    """Four Adam-only locks; the primary Bayes mixture is never selected."""
    if (type(selection_end) is not int or selection_end != SELECTION_END
            or type(consumed) is not int or consumed != selection_end
            or len(predictions) != selection_end or len(target) != selection_end):
        raise ValueError('selection requires the exact 335872 barrier before any suffix label')
    if predictions.ndim != 2 or predictions.shape[1] != len(names):
        raise ValueError('one prediction column is required per output name')
    locks = {}
    for family in ADAM_FAMILIES:
        columns = candidate_columns(names, family, adam_grid)
        lock = stock.select_prefix(predictions[:, columns], target, adam_grid,
                                   COLD_START, selection_end, consumed)
        if not all(math.isfinite(row['error_ratio']) for row in lock['candidates']):
            raise FloatingPointError('nonfinite prefix candidate; failed arms cannot be omitted')
        locks[family] = lock
    return locks


def load_real_locks(path, fingerprint, source_sha256, reference, samples, raw_zero_sha256,
                    selection_end=SELECTION_END, adam_grid=ADAM_GRID):
    """Null uses all four real locks, never reselects against randomized labels."""
    try:
        source = json.loads(Path(path).read_bytes())
        if (source['args']['view'] != 'real' or source['args']['seed'] != 1
                or source['args']['selection_end'] != selection_end):
            raise ValueError('locks require matching real seed1 selection provenance')
        if source['status'] not in ('running', 'completed', 'pruned'):
            raise ValueError('real artifact must be running, completed, or intentionally pruned')
        if any(not valid_digest(fingerprint[key]) or source['data_sha256'][key] != fingerprint[key]
               for key in DATA_KEYS):
            raise ValueError('real packed/original/latest/target SHA256 mismatch')
        if not valid_digest(raw_zero_sha256) or source['raw_zero_sha256'] != raw_zero_sha256:
            raise ValueError('causal raw-zero diagnostic SHA256 mismatch')
        if not source_sha256 or source['source_sha256'] != source_sha256:
            raise ValueError('algorithm source SHA256 mismatch')
        if source['reference'] != reference:
            raise ValueError('reference provenance mismatch')
        if source['view_target_sha256'] != fingerprint['target']:
            raise ValueError('real labels differ from verified reference target')
        phases = phase_ranges(samples, selection_end)
        if (source['samples'] != samples or source['maximum_observations'] != samples
                or source['phases'] != {key: list(bounds) for key, bounds in phases.items()}):
            raise ValueError('real horizon or phases differ')
        consumed = source['processed_observations']
        if (type(consumed) is not int or not selection_end <= consumed <= samples
                or source['processed'] != consumed
                or (source['status'] == 'completed' and consumed != samples)):
            raise ValueError('inconsistent real consumed horizon')
        if (tuple(adam_grid) != ADAM_GRID or source['adam_grid'] != list(adam_grid)
                or source['output_names'] != list(output_names())
                or source['protocol']['primary_output'] != PRIMARY):
            raise ValueError('incompatible Adam grid, output order, or fixed primary')
        locks = source['adam_locks']
        if set(locks) != set(ADAM_FAMILIES):
            raise ValueError('all four Adam locks are required')
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
                    or lock['selected_lr'] != adam_grid[index]
                    or len(lock['candidates']) != len(adam_grid)):
                raise ValueError('selected index/LR or candidate count differs from grid')
            scores = []
            for candidate, lr in zip(lock['candidates'], adam_grid):
                if (candidate['lr'] != lr or type(candidate['count']) is not int
                        or candidate['count'] != selection_end - COLD_START):
                    raise ValueError('candidate LR/count does not match prefix')
                keys = ('target_squared_sum', 'prediction_squared_sum', 'target_prediction_sum',
                        'error_squared_sum', 'error_ratio', 'prediction_energy_ratio',
                        'signed_cross_term_ratio', 'decomposition_residual')
                if any(not math.isfinite(candidate[key]) for key in keys):
                    raise ValueError('nonfinite prefix evidence')
                yy, pp, yp, ee = (candidate[key] for key in keys[:4])
                if yy <= 0 or pp < 0 or ee < 0:
                    raise ValueError('invalid squared sums')
                if shared_target_sum is None:
                    shared_target_sum = yy
                if yy != shared_target_sum:
                    raise ValueError('all four families must use identical real-prefix targets')
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
        return locks
    except (KeyError, TypeError, IndexError, OverflowError, json.JSONDecodeError) as error:
        raise ValueError(f'invalid real lock provenance: {error}') from error


def selected_columns(names, adam_locks, adam_grid=ADAM_GRID):
    columns = {name: names.index(name) for name in
               (PRIMARY, 'original_dense_bayes_mixture', 'original_zero', 'latest_zero',
                'raw_zero_transformed')}
    if adam_locks:
        for family in ADAM_FAMILIES:
            columns[f'adam_{family}'] = candidate_columns(names, family, adam_grid)[
                adam_locks[family]['selected_index']]
    return columns


def paired_comparisons(summary, names, columns):
    primary = columns[PRIMARY]
    return {phase: {'count': row['count'], 'complete': row['complete'],
                    'observed_range': row['observed_range'], 'comparisons': [
                        {'name': names[primary], 'role': PRIMARY, 'baseline': baseline,
                         'baseline_name': names[column],
                         'error_ratio_difference': row['metrics'][primary]['error_ratio']
                                                   - row['metrics'][column]['error_ratio'],
                         'mse_difference': (row['metrics'][primary]['error_squared_sum']
                                            - row['metrics'][column]['error_squared_sum']) / row['count']}
                        for baseline, column in columns.items() if baseline != PRIMARY
                    ] if row['metrics'] else []} for phase, row in summary.items()}


def checkpoint_health(runner):
    clocks = {'runner': int(runner.index.item()), 'model': int(runner.model.observations.item())}
    covariance = {}
    for frame in ('original', 'latest'):
        model, adam = getattr(runner.model, frame), getattr(runner.adam, frame)
        clocks[f'{frame}_model'] = int(model.observations.item())
        clocks[f'{frame}_nig_updates'] = float((2 * (model.alpha - 2)).item())
        for family in ('linear', 'mlp'):
            learner = getattr(adam, family)
            clocks[f'{frame}_{family}_index'] = int(learner.index.item())
            clocks[f'{frame}_{family}_steps'] = int(learner.steps.item())
        cov = model.dense_cov
        _, info = torch.linalg.cholesky_ex(cov, check_errors=False)
        covariance[frame] = {'cholesky_info': info.cpu().tolist(),
                             'max_asymmetry': float((cov - cov.mT).abs().amax().item()),
                             'diagonal_min': float(model.diag_cov.amin().item()),
                             'diagonal_finite': bool(torch.isfinite(model.diag_cov).all())}
    return clocks, covariance


def trajectory_tensors(adam):
    return (adam.original.mlp.predictions, adam.latest.mlp.predictions)


def cheap_adam_state(adam):
    trajectories = trajectory_tensors(adam)
    return [tensor for tensor in adam.mutable if all(tensor is not value for value in trajectories)]


def reported_predictions(runner, causal_raw_zero, start, stop):
    """Append the deterministic pretarget diagnostic only to consumed host rows."""
    learned = runner.predictions[start:stop].cpu().numpy()
    return np.concatenate((learned, causal_raw_zero[start:stop, None]), axis=1)


def save_artifacts(root, runner, target, names, phases, consumed, policy, result, causal_raw_zero):
    if type(consumed) is not int or not 0 <= consumed <= len(target):
        raise ValueError('checkpoint consumed count is outside the stream')
    prediction = reported_predictions(runner, causal_raw_zero, 0, consumed)
    np.save(root / 'predictions.npy', prediction)
    np.save(root / 'targets.npy', target[:consumed])
    temporary = root / 'checkpoint.pt.tmp'
    torch.save({'step': consumed, 'runner_index': runner.index.cpu(),
                'model_state': [tensor.cpu() for tensor in runner.model.state_tensors()],
                'original_state': [tensor.cpu() for tensor in runner.model.original.state_tensors()],
                'latest_state': [tensor.cpu() for tensor in runner.model.latest.state_tensors()],
                'model_configs': runner.model.configs,
                'adam_state': [tensor.cpu() for tensor in cheap_adam_state(runner.adam)],
                'adam_configs': runner.adam.configs, 'autocull': policy.state_dict(),
                'trajectory_columns': {frame: candidate_columns(names, f'{frame}_mlp', result['adam_grid'])
                                       for frame in ('original', 'latest')},
                'note': 'Complete mutable learner state except allocated trajectory buffers; consumed MLP trajectories are exact columns of predictions.npy. No resume CLI.'}, temporary)
    temporary.replace(root / 'checkpoint.pt')
    if result.get('clocks', {}).get('runner') != consumed:
        try:
            clocks, covariance = checkpoint_health(runner)
            result.update(clocks=clocks, covariance_health=covariance)
        except Exception as error:
            result['checkpoint_diagnostic_failure'] = {'type': type(error).__name__, 'message': str(error)}
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
            raise ValueError('seed1, fixed selection_end=335872, and positive log cadence required')
        if not args.reference_result.strip():
            raise ValueError('--reference-result must identify the frozen v7 real artifact')
        if (args.view == 'random_sign') != bool(args.real_result):
            raise ValueError('random_sign requires --real-result; real selects only its four Adam locks')
        helper_args = stock_stream.Args(seed=1, steps=0)
        bars = stock_stream.read_bars(args.bars)
        original, latest, real_target, causal_raw_zero = build_paired_stream(bars, helper_args)
        samples = len(real_target)
        phases = phase_ranges(samples, args.selection_end)
        prefix, grid = args.selection_end, ADAM_GRID
        sources = source_fingerprint()
        reference = verify_reference(args.reference_result, original, real_target, sources['stock_stream.py'])
        if latest.shape != original.shape or latest.dtype != np.float32 or samples != len(bars) - 33:
            raise ValueError('paired windows must retain every original full-file target')
        features = np.concatenate((original, latest), axis=1)
        fingerprint = {'features': array_sha256(features), 'original_features': array_sha256(original),
                       'latest_features': array_sha256(latest), 'target': array_sha256(real_target)}
        raw_zero_sha256 = array_sha256(causal_raw_zero)
        target = real_target
        if args.view == 'random_sign':
            payload = Path(args.real_result).read_bytes()
            locks = load_real_locks(args.real_result, fingerprint, sources, reference, samples,
                                    raw_zero_sha256, prefix, grid)
            if payload != Path(args.real_result).read_bytes():
                raise ValueError('real lock artifact changed while loading')
            result.update(adam_locks=locks, transferred_lock_sha256=hashlib.sha256(payload).hexdigest())
            signs = np.random.default_rng(1).integers(0, 2, size=samples, dtype=np.int8) * 2 - 1
            target = real_target * signs
        result.update(samples=samples, maximum_observations=samples, bars=len(bars),
                      phases=phases, data_sha256=fingerprint, source_sha256=sources, reference=reference,
                      view_target_sha256=array_sha256(target), stock_helper_args=asdict(helper_args),
                      adam_grid=grid, original_feature_reference_sha256=reference['data_sha256']['features'],
                      target_reference_sha256=reference['data_sha256']['target'],
                      raw_zero_sha256=raw_zero_sha256,
                      protocol={
                          'primary_output': PRIMARY,
                          'controls': 'Fixed original dense Bayes mixture, both zero rows, causal raw_zero_transformed normalization diagnostic, and separately real-prefix-locked original/latest linear/MLP Adam.',
                          'selection': 'Only four Adam LR locks on [2000,335872); checkpoint at exactly335872 before label335872. Same complete eleven-LR grids and frozen optimizers/initialization. No prior or model-row selection.',
                          'architecture': 'Two independent frozen five-prior CorrelatedNIG learners; identical current targets. No drift, meta-learning, or added normalization.',
                          'information_change': 'Forecast declared after bar t+32. Original window bars t..t+31; latest window bars t+1..t+32. Both predict unchanged ret[t+33] centered/scaled using t+32 and clipped +/-10. Additional available information, not an optimizer-only claim.',
                          'normalization': 'Frozen helper channel construction, EWMA, centering, trailing scale and clipping computed once in paired helper. Original full-file feature and target bytes must match the required v7 real reference.',
                          'raw_zero_transformed': 'Causal clip(-drift[t+32]/vol[t+32],-10,10), computed from pretarget states only and unchanged on random_sign (never multiplied by the current sign). Transform of a raw-zero prior, not the exact conditional mean of clipped noise or proof of market predictability; zero remains the conditional-mean null baseline.',
                          'horizon': 'Every original B-33 target retained, including final row; no append, fabrication, suffix truncation, or label shift.',
                          'random_sign': 'Fresh paired learners and banks, seed1 signs times verified real target; identical packed inputs/reference/source hashes and all four real Adam locks inherited without null selection.',
                          'primary_segment': '[335872,N), beyond v7 real consumption335872; earlier aggregate experiments covered the file, so not pristine never-seen data.',
                          'suffix': 'Continuous prelabel predict/update without resets or suffix choices; graph68 learned outputs unchanged, plus host-only deterministic raw-zero diagnostic as column69. Save consumed rows only.',
                          'pruning': 'Only fixed latest_dense_bayes_mixture; suffix-local default ProxyCull warmup/patience, error_ratio min_delta1e-4 plus null prediction_energy_ratio1e-4. Censored observed-only evidence, exit75.',
                          'precision': 'CUDA compiled blocks16/singletons, TF32 disabled; frozen FP32 covariance/weights and FP64 NIG scalars. Both dense covariance Cholesky and diagonal positivity, all nested clocks; never repair.',
                          'limitations': 'One SPY stream and seed; null diagnoses conditional-mean behavior, not significance or universal optimality. Compare latest against original Bayes and matched latest Adam before attributing any gain.'})
        stock.save_result(root, result)
        if (not np.isfinite(features).all() or not np.isfinite(target).all()
                or causal_raw_zero.shape != target.shape or causal_raw_zero.dtype != np.float32
                or not np.isfinite(causal_raw_zero).all()):
            raise FloatingPointError('stock features, targets, or causal diagnostic are invalid/nonfinite')
        # Reference, target, horizon, and null provenance checks deliberately precede CUDA.
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU learner fallback')
        runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
        device = torch.device('cuda')
        xs, ys = torch.as_tensor(features, device=device), torch.as_tensor(target, device=device)
        model = AvailableStateComparison(features.shape[1], device)
        adam = PairedAdamBank(xs, ys, grid)
        runner = StockRunner(xs, ys, model, adam)
        names = output_names(model.output_names, grid)
        if tuple(model.output_names) != MODEL_NAMES:
            raise ValueError('paired model differs from fixed output contract')
        result.update(output_names=names, input_dim=features.shape[1], frame_input_dim=original.shape[1],
                      learned_output_names=output_names(model.output_names, grid, include_diagnostic=False),
                      hidden=64, model_configs=model.configs)
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
            pred = reported_predictions(runner, causal_raw_zero, previous, step)
            scales = torch.cat((model.noise, adam.scale, *(getattr(getattr(adam, frame), family).noise
                                for frame in ('original', 'latest') for family in ('linear', 'mlp'))))
            finite_state = torch.stack([torch.isfinite(tensor).all() for tensor in
                                        (*model.state_tensors(), *cheap_adam_state(adam))]).all()
            if (not np.isfinite(pred).all() or not bool(finite_state)
                    or not bool((torch.isfinite(scales) & (scales > 0)).all())):
                result['nonfinite_failure_interval'] = [previous, step]
                raise FloatingPointError('nonfinite forecasts/state or nonfinite/nonpositive learner scales')
            if any(any(info != 0 for info in row['cholesky_info']) or not row['diagonal_finite']
                   or row['diagonal_min'] <= 0 for row in covariance_health.values()):
                raise FloatingPointError('original/latest posterior covariance lost positive definiteness')
            metrics = stock.metric_sums(target[previous:step], pred)
            if any(not math.isfinite(value) for row in metrics for value in row.values()):
                raise FloatingPointError('nonfinite metric; zero target energy is not success')
            if step == prefix:
                if args.view == 'real':
                    result['adam_locks'] = select_prefix_locks(
                        reported_predictions(runner, causal_raw_zero, 0, prefix),
                        target[:prefix], names, consumed, prefix)
                # Persist all four locks before replay can consume label335872.
                save_artifacts(root, runner, target, names, phases, consumed, policy, result, causal_raw_zero)
            columns = selected_columns(names, result['adam_locks'])
            primary = columns[PRIMARY]
            decision = None
            if previous >= prefix:
                if not result['adam_locks']:
                    raise RuntimeError('suffix cannot run without all four real-prefix Adam locks')
                if args.autocull:
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
            for baseline, column in columns.items():
                if baseline != PRIMARY:
                    writer.add_scalar(f'{args.view}/{PRIMARY}_minus_{baseline}/error_ratio',
                                      metrics[primary]['error_ratio'] - metrics[column]['error_ratio'], step)
            writer.add_scalar('updates_per_second', step / replay_seconds, step)
            writer.flush()
            stock.save_result(root, result)
            print(f'PROGRESS view={args.view} consumed={step}/{samples} latest_bayes_error_ratio={metrics[primary]["error_ratio"]} replay_seconds={replay_seconds:.1f}', flush=True)
            if decision:
                result.update(status='pruned', pruning=decision, exit_code=PRUNED_EXIT_CODE)
                save_artifacts(root, runner, target, names, phases, consumed, policy, result, causal_raw_zero)
                print(f'RESULTS {root / "results.json"} status=pruned consumed={consumed}', flush=True)
                prune_proxy(root, args.view, decision)
            previous = step
        result.update(status='completed', wall_seconds=time.perf_counter() - started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save_artifacts(root, runner, target, names, phases, consumed, policy, result, causal_raw_zero)
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
                save_artifacts(root, runner, target, names, phases, consumed, policy, result, causal_raw_zero)
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
