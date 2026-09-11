"""Causal full-file SPY transfer of frozen v3/v4 against paired original Adam grids.

The primary row is fixed before evaluation; suffix scores never select a posterior
variant or an Adam rate. Random-sign runs inherit both real first-quarter locks.
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
from cleanrl.plasticity.predictive_mean_risk_conjugate_eval_v3 import Runner
from cleanrl.plasticity.predictive_mean_risk_conjugate_v3 import PredictiveMeanRisk
from cleanrl.plasticity.predictive_persistent_support_v4 import PersistentSupport
from cleanrl.plasticity.predictive_structure_filter_v1 import SparsePosterior
from cleanrl.shared import runtime
from cleanrl.shared.autocull import PRUNED_EXIT_CODE, ProxyCull, ProxyPruned, prune_proxy


@dataclass
class Args:
    output_dir: str
    view: Literal['real', 'random_sign'] = 'real'
    bars: str = stock_stream.Args.bars
    seed: int = 1
    log_every: int = 8192
    autocull: bool = True
    real_result: str = ''


class TransferModel:
    output_names = ('v3_likelihood_static', 'v3_likelihood_switching',
                    'v4_likelihood_birth_only', 'v4_likelihood_switching')

    def __init__(self, input_dim, device):
        self.v3 = PredictiveMeanRisk(input_dim, device)
        self.v4 = PersistentSupport(input_dim, device)
        self.observations = torch.zeros((), dtype=torch.int64, device=device)
        self.configs = {'v3': self.v3.configs, 'v4': self.v4.configs}

    @property
    def noise(self):
        return torch.cat((self.v3.noise, self.v4.noise))

    def state_tensors(self):
        return [*self.v3.state_tensors(), *self.v4.state_tensors(), self.observations]

    @torch.no_grad()
    def update(self, x, y):
        prediction = torch.cat((self.v3.update(x, y)[:2], self.v4.update(x, y)[:2]))
        self.observations.add_(1)
        return prediction


class AdamBank:
    """Independent, unchanged linear and hidden64 MLP Adam learners."""

    def __init__(self, xs, ys, grid):
        self.configs = tuple(grid)
        linear_args = sparse.Args(seed=1, steps=len(xs), input_dim=xs.shape[1], graph_steps=16)
        mlp_args = bayes.Args(seed=1, samples=len(xs), input_dim=xs.shape[1], hidden=64,
                              graph_steps=16, known_noise=False)
        initial = bayes.init_weights(mlp_args, torch.Generator(device=xs.device).manual_seed(1), xs.device)
        self.linear = sparse.LinearLearner('adam', grid, linear_args, xs, ys)
        self.mlp = stock.MeasuredLearner('adam', grid, initial, mlp_args, xs, ys)
        self.scale = torch.cat((self.linear.scale, self.mlp.scale))
        self.prediction = torch.zeros(2 * len(grid), device=xs.device)
        self.mutable = [*self.linear.mutable, *self.mlp.mutable, self.prediction]

    @property
    def index(self):
        return self.linear.index

    @torch.no_grad()
    def update(self):
        self.linear.update()
        self.mlp.update()
        self.prediction.copy_(torch.cat((self.linear.prediction,
                                        self.mlp.predictions.index_select(
                                            0, (self.mlp.index - 1).reshape(1)).squeeze(0))))


class StockRunner(Runner):
    @torch.no_grad()
    def capture(self, count=16):
        """Frozen runner update with complete restoration, including on capture failure."""
        if type(count) is not int or not 1 <= count <= len(self.xs):
            raise ValueError('graph count must fit the stream')
        initial = [tensor.clone() for tensor in self.mutable]

        def reset():
            for tensor, saved in zip(self.mutable, initial):
                tensor.copy_(saved)

        compiled = torch.compile(self.update, fullgraph=True, mode='max-autotune-no-cudagraphs')
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graphs = {}
        try:
            with torch.cuda.stream(stream):
                compiled()
                reset()
                compiled()
            stream.synchronize()
            reset()
            torch.cuda.synchronize()
            for size in sorted({1, count}):
                for _ in range(size):
                    compiled()
                expected = [tensor.clone() for tensor in self.mutable]
                reset()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(size):
                        compiled()
                reset()
                graph.replay()
                torch.cuda.synchronize()
                for actual, reference in zip(self.mutable, expected):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                reset()
                graphs[size] = graph
            return graphs
        finally:
            stream.synchronize()
            reset()
            torch.cuda.synchronize()


def summarize_predictions(target, predictions, names, phases, consumed):
    """Reduce only written forecasts; retain explicit empty/censored phase records."""
    target, predictions = np.asarray(target), np.asarray(predictions)
    if (target.ndim != 1 or predictions.ndim != 2 or predictions.shape[1] != len(names)
            or len(set(names)) != len(names)):
        raise ValueError('expected one target vector and one named column per output')
    if (type(consumed) is not int or not 0 <= consumed <= len(target)
            or consumed > len(predictions)):
        raise ValueError('consumed must be within both stored arrays')
    summary = {}
    for phase, bounds in phases.items():
        if (len(bounds) != 2 or any(type(value) is not int for value in bounds)
                or not 0 <= bounds[0] <= bounds[1] <= len(target)):
            raise ValueError(f'invalid phase bounds: {phase}')
        start, stop = bounds
        end = max(start, min(stop, consumed))
        count = end - start
        metrics = stock.metric_sums(target[start:end], predictions[start:end]) if count else []
        summary[phase] = {'count': count, 'expected_count': stop - start,
                          'complete': consumed >= stop, 'observed_range': [start, end],
                          'metrics': [{'name': name, **row} for name, row in zip(names, metrics)]}
    return summary


def load_real_locks(path, fingerprint, grid):
    """Accept only matching, internally consistent real-prefix evidence, even if pruned."""
    source = json.loads(Path(path).read_bytes())
    try:
        if source['args']['view'] != 'real' or source['args']['seed'] != 1:
            raise ValueError('Adam locks require real seed1 provenance')
        if source.get('status') not in ('running', 'completed', 'pruned'):
            raise ValueError('real artifact must be running, completed or intentionally pruned')
        for name in ('features', 'target'):
            value = fingerprint[name]
            if (not isinstance(value, str) or len(value) != 64
                    or any(c not in '0123456789abcdef' for c in value)
                    or source['data_sha256'][name] != value):
                raise ValueError(f'Adam lock data SHA256 mismatch: {name}')
        samples = source['samples']
        if type(samples) is not int:
            raise ValueError('samples must be an integer')
        phases = stock.phase_ranges(samples, 2000)
        if {name: list(bounds) for name, bounds in phases.items()} != source['phases']:
            raise ValueError('Adam lock phases must use original first-quarter selection')
        consumed, prefix = source['processed_observations'], samples // 4
        if type(consumed) is not int or not prefix <= consumed <= samples:
            raise ValueError('Adam locks require an actually consumed first quarter')
        if source['status'] == 'completed' and consumed != samples:
            raise ValueError('completed real artifact has an unconsumed suffix')
        if source.get('maximum_observations', samples) != samples:
            raise ValueError('inconsistent real maximum horizon')
        grid = list(grid)
        if (not grid or grid != sorted(set(grid))
                or any(not math.isfinite(lr) or lr <= 0 for lr in grid)
                or source.get('adam_grid', grid) != grid):
            raise ValueError('incompatible Adam grid')
        locks = source['adam_locks']
        if set(locks) != {'linear', 'mlp'}:
            raise ValueError('both original Adam locks are required')
        shared_target_sum = None
        for family, lock in locks.items():
            for field, expected in (('selection_start_inclusive', 2000),
                                    ('selection_end_exclusive', prefix),
                                    ('optimizer_updates_at_lock', prefix),
                                    ('suffix_observations_used', 0)):
                if type(lock[field]) is not int or lock[field] != expected:
                    raise ValueError(f'{family} lock is not exact causal prefix selection: {field}')
            if (lock['criterion'] != 'minimum prequential squared error / zero-predictor squared error'
                    or lock['tie_break'] != 'first (smallest) learning rate'):
                raise ValueError('incompatible Adam selection rule')
            index = lock['selected_index']
            if type(index) is not int or not 0 <= index < len(grid) or lock['selected_lr'] != grid[index]:
                raise ValueError('Adam selected index/rate does not match grid')
            candidates = lock['candidates']
            if len(candidates) != len(grid):
                raise ValueError('Adam candidate count does not match grid')
            scores = []
            for candidate, lr in zip(candidates, grid):
                if candidate['lr'] != lr or candidate['count'] != prefix - 2000:
                    raise ValueError('Adam candidate rate/count does not match selection interval')
                target_sum = candidate['target_squared_sum']
                if not math.isfinite(target_sum) or target_sum <= 0:
                    raise ValueError('Adam prefix requires finite positive target energy')
                if shared_target_sum is None:
                    shared_target_sum = target_sum
                if target_sum != shared_target_sum:
                    raise ValueError('Adam candidates must share identical prefix targets')
                error, score = candidate['error_squared_sum'], candidate['error_ratio']
                if (not math.isfinite(error) or error < 0 or not math.isfinite(score) or score < 0
                        or not math.isclose(score, error / target_sum, rel_tol=1e-12, abs_tol=0)):
                    raise ValueError('Adam candidate error sums disagree with selection score')
                scores.append(score)
            if index != min(range(len(scores)), key=scores.__getitem__):
                raise ValueError('Adam winner violates prefix minimum or first-rate tie break')
        return locks
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError(f'invalid real Adam lock provenance: {error}') from error


def save_artifacts(root, runner, target, names, phases, consumed, policy, result):
    prediction = runner.predictions[:consumed].cpu().numpy()
    np.save(root / 'predictions.npy', prediction)
    np.save(root / 'targets.npy', target[:consumed])
    # MeasuredLearner retains its own trajectory; it is evidence, not optimizer state.
    adam_state = [tensor.cpu() for tensor in runner.adam.mutable
                  if tensor is not runner.adam.mlp.predictions]
    temporary = root / 'checkpoint.pt.tmp'
    torch.save({'step': consumed, 'runner_index': runner.index.cpu(),
                'model_state': [tensor.cpu() for tensor in runner.model.state_tensors()],
                'adam_state': adam_state, 'adam_configs': runner.adam.configs,
                'autocull': policy.state_dict(),
                'note': 'Complete learner state; prediction trajectories stored separately. No resume CLI.'}, temporary)
    temporary.replace(root / 'checkpoint.pt')
    result.update(processed_observations=consumed,
                  phase_metrics=summarize_predictions(target, prediction, names, phases, consumed),
                  prediction_artifact='predictions.npy', target_artifact='targets.npy',
                  checkpoint_artifact='checkpoint.pt',
                  nonfinite_predictions=int(np.count_nonzero(~np.isfinite(prediction))),
                  autocull_state=policy.state_dict())
    locks = result.get('adam_locks')
    if locks:
        width = len(runner.adam.configs)
        columns = {'linear': 4 + locks['linear']['selected_index'],
                   'mlp': 4 + width + locks['mlp']['selected_index']}
        result['paired_comparisons'] = {}
        for phase, row in result['phase_metrics'].items():
            metrics = row['metrics']
            result['paired_comparisons'][phase] = {
                'count': row['count'], 'complete': row['complete'],
                'comparisons': [{
                    'name': names[i], 'adam_family': family, 'adam_name': names[column],
                    'error_ratio_difference': metrics[i]['error_ratio'] - metrics[column]['error_ratio'],
                    'mse_difference': ((metrics[i]['error_squared_sum'] - metrics[column]['error_squared_sum'])
                                       / row['count']),
                } for i in range(4) for family, column in columns.items()] if metrics else []}
    stock.save_result(root, result)


@torch.no_grad()
def main():
    args = tyro.cli(Args)
    if not args.output_dir.strip():
        raise ValueError('explicit nonempty --output-dir is required')
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'results.json').exists():
        raise ValueError('output directory already contains results; use a fresh directory')
    result = {'args': asdict(args), 'run_dir': str(root), 'status': 'preparing',
              'processed_observations': 0, 'curves': [], 'adam_locks': None}
    stock.save_result(root, result)
    print(f'RESULTS {root / "results.json"}', flush=True)
    runner, writer, target, policy = None, None, None, None
    consumed = 0
    started = time.perf_counter()
    try:
        if args.seed != 1 or args.log_every <= 0:
            raise ValueError('seed1 and positive logging cadence are required')
        if (args.view == 'random_sign') != bool(args.real_result):
            raise ValueError('random_sign requires --real-result; real must select its own prefix')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU learner fallback')
        runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
        helper_args = stock_stream.Args(seed=1, steps=0)
        bars = stock_stream.read_bars(args.bars)
        features, real_target = stock_stream.build_stream(bars, helper_args)
        samples = len(real_target)
        phases = stock.phase_ranges(samples, 2000)
        prefix = phases['prefix_all'][1]
        grid = stock.Args.adam_lrs
        fingerprint = {'features': hashlib.sha256(memoryview(features).cast('B')).hexdigest(),
                       'target': hashlib.sha256(memoryview(real_target).cast('B')).hexdigest()}
        target = real_target
        if args.view == 'random_sign':
            payload = Path(args.real_result).read_bytes()
            locks = load_real_locks(args.real_result, fingerprint, grid)
            if payload != Path(args.real_result).read_bytes():
                raise ValueError('real lock artifact changed while loading')
            source = json.loads(payload)
            if source['samples'] != samples or source['phases'] != {k: list(v) for k, v in phases.items()}:
                raise ValueError('real lock horizon differs from the actual stream')
            result.update(adam_locks=locks, transferred_lock_sha256=hashlib.sha256(payload).hexdigest())
            signs = np.random.default_rng(1).integers(0, 2, size=samples, dtype=np.int8) * 2 - 1
            target = real_target * signs
        result.update(samples=samples, maximum_observations=samples, bars=len(bars),
                      phases=phases, data_sha256=fingerprint,
                      view_target_sha256=hashlib.sha256(memoryview(target).cast('B')).hexdigest(),
                      stock_helper_args=asdict(helper_args), adam_grid=grid,
                      source_sha256={Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                     for path in (__file__, inspect.getfile(PredictiveMeanRisk),
                                                  inspect.getfile(PersistentSupport), inspect.getfile(SparsePosterior),
                                                  inspect.getfile(Runner), stock.__file__, sparse.__file__,
                                                  bayes.__file__, stock_stream.__file__, runtime.__file__,
                                                  inspect.getfile(ProxyCull))},
                      protocol={
                          'primary_output': TransferModel.output_names[0],
                          'secondary_output': TransferModel.output_names[2],
                          'controls': [TransferModel.output_names[1], TransferModel.output_names[3]],
                          'selection': 'Both Adam grids lock real first-quarter error ratio excluding 2000 cold-start observations, before any suffix update; no posterior selection.',
                          'random_sign': 'Independent seed1 Rademacher signs times original targets; same features, full grids, original real locked indices only.',
                          'suffix': 'Chronological predict-before-update; continued learning, not frozen weights; report each paired row relative to zero and both locked Adam forecasts.',
                          'phases': 'Original stock evaluator phase boundaries retained; positive/reversed names are chronological halves only, with no planted signal or sign reversal.',
                          'pruning': 'Four posterior outputs only; suffix-local phase_start=prefix, default warmup/patience; error_ratio plus null prediction_energy_ratio; partial evidence exits75.',
                          'precision': 'FP32 frozen algorithms; highest matmul precision, TF32 disabled; compiled fullgraph CUDA blocks16 and exact singleton tails.',
                          'alignment': 'Original helper unchanged: row t uses bars t..t+31, target ret[t+33]; newest return is two bars old; target centering/volatility use intervening bar t+32.',
                          'normalization': 'Original trailing channel scaling/clipping and default scaled targets; raw_target=False, vol_feature=False; no projection or feature changes.',
                          'limitations': 'One SPY stream and seed; random signs diagnose zero conditional mean, not financial significance. Pruned suffix evidence is censored, never a full-horizon result.'})
        stock.save_result(root, result)
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise FloatingPointError('stock features or targets are nonfinite')
        device = torch.device('cuda')
        xs, ys = torch.as_tensor(features, device=device), torch.as_tensor(target, device=device)
        model = TransferModel(features.shape[1], device)
        adam = AdamBank(xs, ys, grid)
        runner = StockRunner(xs, ys, model, adam)
        names = (*model.output_names, *(f'adam_linear_{lr:g}' for lr in grid),
                 *(f'adam_mlp_{lr:g}' for lr in grid))
        result.update(output_names=names, expert_configs=model.configs, input_dim=features.shape[1], hidden=64)
        policy = ProxyCull(4, {'error_ratio': 1e-4, **(
            {'prediction_energy_ratio': 1e-4} if args.view == 'random_sign' else {})})
        writer = SummaryWriter(str(root))
        writer.add_text('protocol', json.dumps(result['protocol'], indent=2))
        capture_start = time.perf_counter()
        graphs = runner.capture(16)
        result.update(capture_seconds=time.perf_counter() - capture_start, status='running',
                      capture_steps=[1, 16], capture_complete_state_verified=True)
        stock.save_result(root, result)
        endpoints = sorted(set(range(args.log_every, samples, args.log_every))
                           | {bound for bounds in phases.values() for bound in bounds if bound})
        previous = 0
        replay_seconds = 0.0
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
            clocks = {'runner': consumed, 'linear_index': int(adam.linear.index.item()),
                      'linear_steps': int(adam.linear.steps.item()), 'mlp_index': int(adam.mlp.index.item()),
                      'mlp_steps': int(adam.mlp.steps.item()), 'model': int(model.observations.item()),
                      'v4': int(model.v4.observations.item())}
            result.update(processed_observations=consumed, clocks=clocks, replay_seconds=replay_seconds)
            if any(value != step for value in clocks.values()):
                raise RuntimeError(f'prequential clocks diverged at {step}: {clocks}')
            pred = runner.predictions[previous:step].cpu().numpy()
            scales = torch.cat((model.noise, model.v3.energy.reshape(1), adam.scale,
                                adam.linear.noise, adam.mlp.noise))
            if not np.isfinite(pred).all() or not bool((torch.isfinite(scales) & (scales > 0)).all()):
                result['nonfinite_failure_interval'] = [previous, step]
                raise FloatingPointError('nonfinite forecasts or nonfinite/nonpositive learner scales')
            metrics = stock.metric_sums(target[previous:step], pred)
            if any(not math.isfinite(value) for row in metrics for value in row.values()):
                raise FloatingPointError('nonfinite prequential metric; zero target energy is not scored as success')
            if step == prefix and args.view == 'real':
                prefix_predictions = runner.predictions[:prefix].cpu().numpy()
                result['adam_locks'] = {
                    family: stock.select_prefix(prefix_predictions[:, offset:offset + len(grid)],
                                                target[:prefix], grid, 2000, prefix, consumed)
                    for family, offset in (('linear', 4), ('mlp', 4 + len(grid)))}
                # Atomic results replacement is the lock barrier before ANY suffix replay.
                save_artifacts(root, runner, target, names, phases, consumed, policy, result)
            decision = None
            if args.autocull and previous >= prefix:
                decision = policy.observe(step, {key: [row[key] for row in metrics[:4]]
                                                 for key in policy.metrics},
                                          phase='suffix', phase_start=prefix)
            result['curves'].append({'step': step, 'interval_start': previous,
                                     'prequential': [{'name': name, **row} for name, row in zip(names, metrics)],
                                     'residual_scales': model.noise.cpu().tolist(),
                                     'autocull': policy.state_dict()})
            result.update(autocull_state=policy.state_dict(), wall_seconds=time.perf_counter() - started)
            for name, row in zip(names, metrics):
                for metric in ('error_ratio', 'prediction_energy_ratio', 'signed_cross_term_ratio'):
                    writer.add_scalar(f'{args.view}/{name}/{metric}', row[metric], step)
            writer.add_scalar('updates_per_second', step / replay_seconds, step)
            writer.flush()
            stock.save_result(root, result)
            print(f'PROGRESS view={args.view} consumed={step}/{samples} primary_error_ratio={metrics[0]["error_ratio"]:.8g} replay_seconds={replay_seconds:.1f}', flush=True)
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
                # Capture rehearsals are restored; only actual replayed observations count.
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
