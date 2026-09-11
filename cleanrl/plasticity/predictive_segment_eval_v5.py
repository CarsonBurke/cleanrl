"""Full CUDA-graph segment research: frozen families, prefix-only AdamW selection.

Development reuses consumed seed1 data. Confirmation is namespace100, not a new
independent seed; its development lock is read before generating any labels.
All views run 60000 observations without culling. Clean labels are reporting
only. A global support prior is not a generic optimizer advance or financial
validation. Run this module through mlq; it never submits jobs itself.
"""

import hashlib
import inspect
import itertools
import json
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
from cleanrl.plasticity.predictive_mean_risk_conjugate_eval_v3 import Runner
from cleanrl.plasticity.predictive_mean_risk_conjugate_v3 import PredictiveMeanRisk
from cleanrl.plasticity.predictive_persistent_support_v4 import PersistentSupport
from cleanrl.plasticity.predictive_segment_posterior_v5 import SegmentPosterior
from cleanrl.plasticity.predictive_structure_filter_v1 import SparsePosterior
from cleanrl.shared import runtime


@dataclass
class Args:
    view: Literal['stationary', 'change', 'null', 'recurrent', 'two_support'] = 'stationary'
    phase: Literal['development', 'confirmation'] = 'development'
    seed: int = 1
    lock: str | None = None
    graph_steps: int = 100
    log_every: int = 1000
    output: str = 'runs'


def adam_grid():
    return [dict(lr=lr, beta1=b1, beta2=b2, decay=decay, epsilon=1e-8)
            for lr, b1, b2, decay in itertools.product(
                (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4), (0., .9, .99), (.9, .999), (0., .01))]


def family_contract():
    return {'segment': [{'budget': 9, 'hazard': 1e-4}, {'budget': 9, 'hazard': 0.},
                        {'budget': 32, 'hazard': 1e-4}],
            'prior': {'inclusion': '1/D', 'slab_variance': 1., 'coordinate_hazard': 0., 'null_mass': .5},
            'noise_rate': .001, 'ig_prior': [2., 1.], 'scoring': 'Gaussian predictive moments',
            'controls': ['v3_likelihood_static', 'v4_birth_only', 'v4_continuous', 'zero', 'legacy_adam_1e-5'],
            'adamw_grid': adam_grid(), 'adamw_objective': 'half squared noisy residual, decoupled weight decay',
            'selection': 'stationary development observations [0,15000), noisy prequential MSE only',
            'stream': {'steps': 60000, 'input_dim': 4096, 'feature_prob': .01, 'noise_sigma': 1., 'seed': 1},
            'namespaces': {'development': 0, 'confirmation': 100, 'stride': 10000019,
                           'feature': 'seed + namespace*stride', 'noise': 'feature + 1000003'},
            'views': {'stationary': 'coordinate0', 'change': '0 -> 1 at30000', 'null': 'zero',
                      'recurrent': '0 -> 1 -> 0 at20000/40000', 'two_support': 'coordinates0+1 throughout'},
            'primary': 'segment_b9', 'approximation_control': 'segment_b32',
            'no_change_ablation': 'segment_h0', 'culling': 'disabled for every full view',
            'confirmation_selection': 'none; every family and AdamW setting prelocked on development',
            'legacy_adam': 'frozen LinearLearner kernel at historical prefix-selected lr1e-5; no reselection'}


def source_hashes():
    objects = {'evaluator': __file__, 'segment': inspect.getfile(SegmentPosterior),
               'kernel': inspect.getfile(SparsePosterior), 'v3': inspect.getfile(PredictiveMeanRisk),
               'v4': inspect.getfile(PersistentSupport), 'runner': inspect.getfile(Runner),
               'stream_legacy_adam': inspect.getfile(sparse.LinearLearner),
               'legacy_adam_base': inspect.getfile(sparse.v2.Learner),
               'metrics': inspect.getfile(stock.metric_sums), 'runtime': inspect.getfile(runtime)}
    return {key: hashlib.sha256(Path(path).read_bytes()).hexdigest() for key, path in objects.items()}


def load_lock(path, sources):
    payload = Path(path).read_bytes()
    lock = json.loads(payload)
    if lock.get('version') != 'segment_v5_lock' or lock.get('families') != family_contract():
        raise ValueError('lock does not match the frozen v5 scientific contract')
    if lock.get('source_sha256') != sources:
        raise ValueError('source files changed since development lock')
    if lock.get('phase') != 'development' or lock.get('view') != 'stationary' or lock.get('locked_at') != 15000:
        raise ValueError('only a stationary development prefix lock may transfer')
    selected = lock.get('selected_adamw')
    scores = lock.get('prefix_noisy_mse')
    if (selected not in adam_grid() or not isinstance(scores, list) or len(scores) != 72
            or not np.isfinite(scores).all() or selected != adam_grid()[int(np.argmin(scores))]):
        raise ValueError('invalid AdamW selection provenance')
    return lock, hashlib.sha256(payload).hexdigest()


class ComparisonModel:
    output_names = ('segment_b9', 'segment_h0', 'segment_b32',
                    'v3_likelihood_static', 'v4_birth_only', 'v4_continuous', 'zero')

    def __init__(self, dimension, device):
        self.segments = [SegmentPosterior(dimension, device, **config) for config in family_contract()['segment']]
        self.v3 = PredictiveMeanRisk(dimension, device)
        self.v4 = PersistentSupport(dimension, device)
        self._zero = torch.zeros(1, device=device)

    def state_tensors(self):
        return [t for model in (*self.segments, self.v3, self.v4) for t in model.state_tensors()]

    def mean_weights(self):
        return torch.cat((*[model.mean_weights() for model in self.segments],
                          self.v3.mean_weights()[:1], self.v4.mean_weights()[:2],
                          torch.zeros_like(self.v3.filter.slab_mean[:1])))

    def update(self, x, y):
        return torch.cat((*[model.update(x, y) for model in self.segments],
                          self.v3.update(x, y)[:1], self.v4.update(x, y)[:2], self._zero))

    def diagnostics(self):
        return {**{name: model.diagnostics() for name, model in zip(self.output_names, self.segments)},
                'v3_aggregation': self.v3.aggregation_weights().cpu().tolist(),
                'v3_inclusion_0_1': self.v3.filter.log_odds[:, :2].sigmoid().cpu().tolist(),
                'v3_slab_mean_0_1': self.v3.filter.slab_mean[:, :2].cpu().tolist(),
                'v4': self.v4.diagnostics()}


class AdamWBank:
    """Parallel online AdamW grid plus the actual frozen historical Adam kernel."""

    def __init__(self, configs, cfg, xs, ys):
        self.configs, self.xs, self.ys = configs, xs, ys
        device = xs.device
        self.w = torch.zeros((len(configs), xs.shape[1]), device=device)
        self.m, self.v = torch.zeros_like(self.w), torch.zeros_like(self.w)
        self.index = torch.zeros((), device=device, dtype=torch.int64)
        self.steps = torch.zeros((), device=device)
        self.lr, self.b1, self.b2, self.decay, self.eps = [
            torch.tensor([c[key] for c in configs], device=device).unsqueeze(-1)
            for key in ('lr', 'beta1', 'beta2', 'decay', 'epsilon')]
        self.legacy = sparse.LinearLearner('adam', (1e-5,), cfg, xs, ys)
        self.scale = torch.zeros(len(configs) + 1, device=device)
        self.prediction = torch.zeros_like(self.scale)
        self.mutable = [self.w, self.m, self.v, self.index, self.steps, self.prediction, *self.legacy.mutable]

    def mean_weights(self):
        return torch.cat((self.w, self.legacy.weights[0][:, 0, :]))

    def update(self):
        ix = self.index.reshape(1)
        x = self.xs.index_select(0, ix).squeeze(0).float()
        y = self.ys.index_select(0, ix).squeeze(0)
        prediction = (self.w * x).sum(-1)
        gradient = (prediction - y).unsqueeze(-1) * x
        self.steps.add_(1)
        self.m.mul_(self.b1).add_((1 - self.b1) * gradient)
        self.v.mul_(self.b2).add_((1 - self.b2) * gradient.square())
        delta = (self.m / (1 - self.b1 ** self.steps)) / ((self.v / (1 - self.b2 ** self.steps)).sqrt() + self.eps)
        self.w.mul_(1 - self.lr * self.decay).sub_(self.lr * delta)
        self.legacy.update()
        self.prediction.copy_(torch.cat((prediction, self.legacy.prediction)))
        self.index.add_(1)


class TracedRunner(Runner):
    """Reuse complete-state graph capture; retain every local truncation and branch."""

    def __init__(self, xs, ys, model, adam):
        super().__init__(xs, ys, model, adam)
        self.traces = [torch.zeros((len(xs), segment.budget + 1, 13), device=xs.device)
                       for segment in model.segments]
        self.discards = torch.zeros((len(xs), len(model.segments)), device=xs.device)
        self.mutable.extend((*self.traces, self.discards))

    def update(self):
        super().update()
        ix = (self.index - 1).reshape(1)
        for trace, segment in zip(self.traces, self.model.segments):
            trace.index_copy_(0, ix, segment.trace().unsqueeze(0))
        self.discards.index_copy_(0, ix, torch.stack([s.discarded_mass for s in self.model.segments]).unsqueeze(0))


def labels(xs, noise, view):
    if view in ('stationary', 'change', 'null'):
        return sparse.teacher_labels(xs, noise, view)
    if view == 'recurrent':
        clean = torch.cat((xs[:20000, 0], xs[20000:40000, 1], xs[40000:, 0])).float()
    else:
        clean = xs[:, :2].float().sum(-1)
    return clean + noise, clean


def support_at(view, step, dimension, device):
    support = torch.zeros(dimension, device=device)
    if view == 'two_support':
        support[:2] = 1
    elif view != 'null':
        moved = (view == 'change' and step >= 30000) or (view == 'recurrent' and 20000 <= step < 40000)
        support[int(moved)] = 1
    return support


def phase_boundaries(view):
    return [0, 20000, 40000, 60000] if view == 'recurrent' else ([0, 30000, 60000] if view == 'change' else [0, 60000])


def tensor_hash(tensor):
    return hashlib.sha256(memoryview(tensor.cpu().numpy())).hexdigest()


@torch.no_grad()
def main():
    args = tyro.cli(Args)
    cfg = sparse.Args(seed=1, graph_steps=args.graph_steps)
    if args.seed != 1 or not 1 <= args.graph_steps <= 15000 or args.log_every <= 0:
        raise ValueError('seed1, graph_steps in [1,15000], and positive logging cadence required')
    selecting = args.phase == 'development' and args.view == 'stationary'
    if selecting != (args.lock is None):
        raise ValueError('only development stationary selects; every other run requires --lock prefix_lock.json')
    sources = source_hashes()
    lock, lock_hash = (None, None) if selecting else load_lock(args.lock, sources)
    configs = adam_grid() if selecting else [lock['selected_adamw']]
    namespace = 0 if args.phase == 'development' else 100
    root = Path(args.output) / f'SparseStream__segment_v5_{args.phase}_{args.view}__1__{time.time_ns()}'
    root.mkdir(parents=True, exist_ok=True)
    result = {'args': asdict(args), 'status': 'preparing', 'families': family_contract(),
              'source_sha256_before': sources, 'transferred_lock_sha256': lock_hash,
              'namespace': namespace, 'curves': [], 'processed_observations': 0,
              'branch_trace_columns': ['retained_mass', 'birth_observation', 'ig_count', 'ig_Q', 'ig_variance',
                                       'inclusion_0', 'inclusion_1', 'slab_mean_0', 'slab_mean_1',
                                       'effective_0', 'effective_1', 'sum_inclusion', 'effective_squared_norm'],
              'branch_trace_timing': 'post-label retained posterior; index t has consumed label t; row0=null',
              'scientific_limitations': [
                  'bounded segment filtering, factorized support, Gaussian IG-moment scoring; not exact joint Bayes',
                  'null histories share one past-only scale rather than a mixture of segment-specific null scales',
                  'new segment hypothesis forgets support; recurrence does not retrieve a stored old segment',
                  'global changes and expected one active coordinate are misspecified for asynchronous or denser tasks',
                  'same seed1, named RNG namespaces, not independent-seed replication',
                  '72-way noisy-prefix selection may be noisy; no post-prefix or clean-target selection',
                  'sparse methods have informative priors unavailable to dense AdamW; not optimizer superiority',
                  'reported replay time is joint including diagnostics, not matched per-method compute',
                  'local discarded-mass sums are diagnostics, not a global truncation-error bound']}
    sparse.save_json(root / 'results.json', result)
    writer = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU model fallback')
        runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
        result['runtime'] = {'torch': torch.__version__, 'numpy': np.__version__,
                             'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(),
                             'dtype': 'float32', 'compile': 'fullgraph', 'tf32': False}
        writer = SummaryWriter(str(root))
        # Namespace0 exactly retains the original generator/chunking contract.
        draw_cfg = sparse.Args(seed=1 + namespace * 10000019)
        xs, noise = sparse.draw_stream(draw_cfg, torch.device('cuda'))
        result['feature_sha256'] = tensor_hash(xs)
        result['noise_sha256'] = tensor_hash(noise)
        if args.phase == 'development' and lock is not None and result['feature_sha256'] != lock['feature_sha256']:
            raise ValueError('development transfer must reproduce the paired feature stream')
        ys, clean = labels(xs, noise, args.view)
        model, adam = ComparisonModel(cfg.input_dim, xs.device), AdamWBank(configs, cfg, xs, ys)
        runner = TracedRunner(xs, ys, model, adam)
        names = (*model.output_names, *(f'adamw_{adam_grid().index(c):02d}' for c in configs), 'legacy_adam_1e-5')
        result.update(output_names=names, adamw_configs=configs,
                      mutable_model_bytes={name: sum(t.numel() * t.element_size() for t in m.state_tensors())
                                           for name, m in zip(('segment_b9', 'segment_h0', 'segment_b32', 'v3_all_rows', 'v4_all_rows'),
                                                              (*model.segments, model.v3, model.v4))})
        start = time.perf_counter()
        graphs = runner.capture(args.graph_steps)
        torch.cuda.synchronize()
        result.update(capture_seconds=time.perf_counter() - start, status='running')
        prefix = 15000
        boundaries = phase_boundaries(args.view)
        endpoints = sorted(set(range(args.log_every, cfg.steps, args.log_every)) | set(boundaries[1:]) | {prefix})
        previous = 0
        start = time.perf_counter()
        for step in endpoints:
            blocks, tail = divmod(step - previous, args.graph_steps)
            for _ in range(blocks):
                graphs[args.graph_steps].replay()
            for _ in range(tail):
                graphs[1].replay()
            torch.cuda.synchronize()
            if int(runner.index) != step or int(adam.index) != step or any(int(s.observations) != step for s in model.segments):
                raise RuntimeError('stream clocks diverged')
            if step == prefix and selecting:
                # Lock on noisy prefix BEFORE any suffix replay or clean scoring.
                prefix_prediction = runner.predictions[:prefix, len(model.output_names):-1].cpu().numpy().astype(np.float64)
                prefix_target = ys[:prefix].cpu().numpy().astype(np.float64)
                scores = np.square(prefix_prediction - prefix_target[:, None]).mean(0)
                if not np.isfinite(scores).all():
                    raise RuntimeError('nonfinite AdamW prefix candidate; no selection lock written')
                lock = {'version': 'segment_v5_lock', 'phase': 'development', 'view': 'stationary',
                        'locked_at': prefix, 'families': family_contract(), 'source_sha256': sources,
                        'selected_adamw': configs[int(scores.argmin())], 'prefix_noisy_mse': scores.tolist(),
                        'feature_sha256': result['feature_sha256'], 'noise_sha256': result['noise_sha256'],
                        'prefix_target_sha256': tensor_hash(ys[:prefix])}
                if source_hashes() != sources:
                    raise RuntimeError('sources changed before prefix lock')
                sparse.save_json(root / 'prefix_lock.json', lock)
                result['prefix_lock_sha256'] = hashlib.sha256((root / 'prefix_lock.json').read_bytes()).hexdigest()
            predictions = runner.predictions[previous:step].cpu().numpy().astype(np.float64)
            target = ys[previous:step].cpu().numpy().astype(np.float64)
            truth = clean[previous:step].cpu().numpy().astype(np.float64)
            weights = torch.cat((model.mean_weights(), adam.mean_weights()))
            if not np.isfinite(predictions).all() or not bool(torch.isfinite(weights).all()):
                raise RuntimeError(f'nonfinite prediction or next-forecast coefficients at observation {step}')
            stale_index = None
            if args.view == 'change' and step >= 30000:
                stale_index = 0
            elif args.view == 'recurrent' and step >= 20000:
                stale_index = 0 if step < 40000 else 1
            risk = sparse.exact_risk(weights, support_at(args.view, step, cfg.input_dim, xs.device),
                                     cfg.feature_prob, stale_index=stale_index)
            row = {'step': step, 'interval_start': previous, 'interval_count': step - previous,
                   'noisy_mse': np.square(predictions - target[:, None]).mean(0).tolist(),
                   'clean_mse': np.square(predictions - truth[:, None]).mean(0).tolist(),
                   'exact_risk': {key: value.cpu().tolist() for key, value in risk.items()},
                   'coefficient_0_1': weights[:, :2].cpu().tolist(), 'diagnostics': model.diagnostics()}
            result['curves'].append(row)
            result.update(processed_observations=step, replay_seconds=time.perf_counter() - start,
                          selected_adamw=lock['selected_adamw'] if lock else None)
            for i, name in enumerate(names):
                for metric in ('noisy_mse', 'clean_mse'):
                    writer.add_scalar(f'{name}/{metric}', row[metric][i], step)
                writer.add_scalar(f'{name}/next_forecast_risk', row['exact_risk']['clean_mse'][i], step)
            for name, segment in zip(model.output_names, model.segments):
                writer.add_scalar(f'{name}/discarded_mass', float(segment.discarded_mass), step)
                writer.add_scalar(f'{name}/cumulative_discarded_mass', float(segment.cumulative_discarded_mass), step)
            writer.add_scalar('joint_updates_per_second', step / result['replay_seconds'], step)
            writer.flush()
            sparse.save_json(root / 'results.json', result)
            previous = step
        pred = runner.predictions.cpu().numpy()
        noisy, truth = ys.cpu().numpy(), clean.cpu().numpy()
        noisy_error = np.square(pred.astype(np.float64) - noisy[:, None])
        clean_error = np.square(pred.astype(np.float64) - truth[:, None])
        np.savez(root / 'prequential.npz', predictions=pred, noisy_target=noisy, clean_target=truth,
                 noisy_squared_error=noisy_error, clean_squared_error=clean_error)
        np.savez(root / 'branch_traces.npz', **{name: trace.cpu().numpy() for name, trace in zip(model.output_names, runner.traces)},
                 discarded_mass=runner.discards.cpu().numpy())
        torch.save({'model_state': [t.cpu() for t in model.state_tensors()],
                    'adam_state': [t.cpu() for t in adam.mutable], 'next_mean_weights': weights.cpu(),
                    'note': 'analysis state; no resume implementation'}, root / 'checkpoint.pt')
        result.update(full_noisy_mse=noisy_error.mean(0).tolist(), full_clean_mse=clean_error.mean(0).tolist(),
                      phase_errors=[{'start': lo, 'end': hi, 'noisy_mse': noisy_error[lo:hi].mean(0).tolist(),
                                     'clean_mse': clean_error[lo:hi].mean(0).tolist()} for lo, hi in zip(boundaries, boundaries[1:])],
                      post_change_clean_mse=clean_error[boundaries[1]:].mean(0).tolist() if len(boundaries) > 2 else None,
                      post_change_noisy_mse=noisy_error[boundaries[1]:].mean(0).tolist() if len(boundaries) > 2 else None,
                      finalrisk=result['curves'][-1]['exact_risk'], target_sha256=tensor_hash(ys),
                      clean_sha256=tensor_hash(clean), source_sha256_after=source_hashes(),
                      feature_sha256_after=tensor_hash(xs), noise_sha256_after=tensor_hash(noise),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if (result['source_sha256_after'] != sources or result['feature_sha256_after'] != result['feature_sha256']
                or result['noise_sha256_after'] != result['noise_sha256']):
            raise RuntimeError('source/data mutation during experiment')
        result['status'] = 'completed'
        sparse.save_json(root / 'results.json', result)
        print(f'RESULTS {root / "results.json"}', flush=True)
    except Exception as error:
        result.update(status='failed', failure=repr(error), source_sha256_after=source_hashes())
        sparse.save_json(root / 'results.json', result)
        raise
    finally:
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    main()
