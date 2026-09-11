"""Predictive model-space inference, v1: structure, nonlinear functions, and change.

A coefficient covariance assumes its model is worth fitting. Instead retain a
null, sparse spike/slab models, linear Gaussian models, and Gaussian models in a
fixed sketched neural-tangent function space. Predict with posterior model means.
Compare static Bayes aggregation, explicit switching-model aggregation, uniform
aggregation, and each model family using IDENTICAL independently trained experts.
This avoids point optimization of the nonlinear representation; it does not
promise a global optimum outside these approximate probabilistic model classes.

Expert residual scales are causal EMAs, NOT identified irreducible noise. Sparse
posteriors factorize; reset mixtures are moment-matched. The aggregate is exact
Bayesian filtering over these causal expert densities, not exact neural Bayes.
CUDA, seed1; configured streams are maximum horizons, not full-run evidence.
Automatic proxy culling is on by default (--no-autocull explicitly disables it).
All six outputs must stagnate on every applicable interval TRAIN metric: noisy
prequential error ratio, plus prediction energy for null/random-sign controls or
prequential clean MSE for non-null synthetic views. Heldout risk never gates
culling. The shared policy uses an 8192-observation EMA half-life, 16384-observation
warmup, at least 4096 observations between evaluations, patience three, and
absolute improvement 1e-4. Actual label-regime changes reset warmup starting with
the first wholly post-switch interval. A surviving output protects the view.
Pruning saves consumed predictions and lightweight posterior state, marks the
whole serial job pruned, and exits 75 before later views; partial results are not
complete-horizon or paired-response evidence. No current clean label enters an
optimizer. Queue with --max-attempts 1; exit 75 must not trigger success-dependent
jobs or automatic retry.
"""

import hashlib
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
from cleanrl.plasticity import network_bayes_stream_v2 as original
from cleanrl.plasticity.kernel_posterior_v1 import GaussianPosterior, TangentFeatures
from cleanrl.plasticity.predictive_structure_filter_v1 import SparsePosterior
from cleanrl.shared import runtime
from cleanrl.shared.autocull import PRUNED_EXIT_CODE, ProxyCull, ProxyPruned, prune_proxy


@dataclass
class Args:
    task: Literal['sparse', 'stock', 'dense'] = 'sparse'
    seed: int = 1
    projected_dim: int = 256
    mixture_hazard: float = 1e-4
    noise_rate: float = .001
    log_every: int = 4096
    feature_chunk: int = 256
    test_samples: int = 8192
    hetero: float = 1.0
    bars: str = stock.Args.bars
    output: str = 'runs'
    autocull: bool = True


OUTPUT_NAMES = ('switching', 'static', 'uniform', 'sparse_family', 'linear_family', 'kernel_family')


class ModelMixture:
    def __init__(self, input_dim, feature_dim, device, hazard=1e-4, noise_rate=.001):
        self.hazard, self.noise_rate = hazard, noise_rate
        self.configs = {
            'sparse': [dict(prior=p, inclusion=r, hazard=h)
                       for p in (.01, 1.) for r in (1 / input_dim, .1) for h in (0., 1e-4)],
            'linear': [dict(prior=p, hazard=h)
                       for p in (.001, .1, 10.) for h in (0., 1e-4)],
            'kernel': [dict(prior=p, hazard=h)
                       for p in (.01, .1, 1., 10.) for h in (0., 1e-4)]}
        self.filters = (SparsePosterior(input_dim, self.configs['sparse'], device),
                        GaussianPosterior(input_dim, self.configs['linear'], device),
                        GaussianPosterior(feature_dim, self.configs['kernel'], device))
        self.raw_scale = math.sqrt(input_dim)
        counts = [len(values) for values in self.configs.values()]
        self.slices = []
        start = 1
        prior = [0.25]
        masks = torch.zeros((3, 1 + sum(counts)), device=device, dtype=torch.bool)
        for i, count in enumerate(counts):
            self.slices.append(slice(start, start + count))
            masks[i, start:start + count] = True
            prior.extend([.25 / count] * count)
            start += count
        self.family_masks = masks
        self.log_prior = torch.tensor(prior, device=device).log()
        self.log_weights = self.log_prior.expand(2, -1).clone()
        self.noise = torch.ones(len(prior), device=device)

    def state_tensors(self):
        return [self.log_weights, self.noise,
                *(t for model in self.filters for t in model.state_tensors())]

    def aggregation_weights(self):
        static = self.log_weights[1] - self.log_weights[1].logsumexp(-1)
        switching = self.log_weights[0] - self.log_weights[0].logsumexp(-1)
        if self.hazard:
            switching = torch.logaddexp(switching + math.log1p(-self.hazard),
                                         self.log_prior + math.log(self.hazard))
        family = torch.softmax(torch.where(self.family_masks, static.unsqueeze(0), -torch.inf), -1)
        return torch.cat((switching.exp().unsqueeze(0), static.exp().unsqueeze(0),
                          self.log_prior.exp().unsqueeze(0), family), 0)

    @torch.no_grad()
    def update(self, raw, features, y):
        weights = self.aggregation_weights()
        means, variances = [torch.zeros_like(self.noise[:1])], [self.noise[:1].clone()]
        for model, x, section in zip(self.filters, (raw, raw / self.raw_scale, features), self.slices):
            mean, variance = model.update(x, y, self.noise[section])
            means.append(mean)
            variances.append(variance)
        mean, variance = torch.cat(means), torch.cat(variances)
        # Every returned expert mean was formed before its current label update.
        prediction = weights @ mean
        log_likelihood = -.5 * (math.log(2 * math.pi) + variance.log() + (y - mean).square() / variance)
        # Recover finite log priors directly, not log(probability) after underflow.
        prior_log = self.log_weights - self.log_weights.logsumexp(-1, keepdim=True)
        if self.hazard:
            switched = torch.logaddexp(prior_log[0] + math.log1p(-self.hazard),
                                       self.log_prior + math.log(self.hazard))
            prior_log = torch.stack((switched, prior_log[1]))
        posterior_log = prior_log + log_likelihood.unsqueeze(0)
        self.log_weights.copy_(posterior_log - posterior_log.logsumexp(-1, keepdim=True))
        self.noise.lerp_((y - mean).square(), self.noise_rate)
        return prediction

    @torch.no_grad()
    def frozen_predictions(self, raw, features):
        means = [torch.zeros((len(raw), 1), device=raw.device)]
        for model, x in zip(self.filters, (raw.float(), raw.float() / self.raw_scale, features)):
            means.append(x @ model.mean_weights().T)
        return torch.cat(means, -1) @ self.aggregation_weights().T


class StreamRunner:
    def __init__(self, xs, features, ys, model, baseline):
        self.xs, self.features, self.ys = xs, features, ys
        self.model, self.baseline = model, baseline
        self.index = torch.zeros((), device=xs.device, dtype=torch.int64)
        self.predictions = torch.zeros((len(xs), len(OUTPUT_NAMES)), device=xs.device)
        self.mutable = [self.index, self.predictions, *model.state_tensors(), *baseline.mutable]
        if hasattr(baseline, 'predictions'):
            self.baseline_predictions = baseline.predictions
        else:
            self.baseline_predictions = torch.zeros((len(xs), len(baseline.scale)), device=xs.device)
            self.mutable.append(self.baseline_predictions)

    @torch.no_grad()
    def update(self):
        ix = self.index.reshape(1)
        raw = self.xs.index_select(0, ix).squeeze(0).float()
        features = self.features.index_select(0, ix).squeeze(0)
        y = self.ys.index_select(0, ix).squeeze(0)
        prediction = self.model.update(raw, features, y)
        self.predictions.index_copy_(0, ix, prediction.unsqueeze(0))
        self.baseline.update()
        if hasattr(self.baseline, 'prediction'):
            self.baseline_predictions.index_copy_(0, ix, self.baseline.prediction.unsqueeze(0))
        self.index.add_(1)

    @torch.no_grad()
    def capture(self, count):
        initial = [t.clone() for t in self.mutable]
        def reset():
            for t, v in zip(self.mutable, initial):
                t.copy_(v)
        compiled = torch.compile(self.update, fullgraph=True, mode='max-autotune-no-cudagraphs')
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            compiled()
            compiled()
        stream.synchronize()
        reset()
        # Verify replay against the SAME compiled recurrence, not a different FP order.
        compiled()
        expected = [t.clone() for t in self.mutable]
        reset()
        torch.cuda.synchronize()
        graphs = {}
        for n in sorted({1, count}):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(n):
                    compiled()
            reset()
            graphs[n] = g
        graphs[1].replay()
        torch.cuda.synchronize()
        for actual, reference in zip(self.mutable, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        reset()
        return graphs


@torch.no_grad()
def dataset(a, device):
    if a.task == 'sparse':
        cfg = sparse.Args()
        xs, noise = sparse.draw_stream(cfg, device)
        views, clean = {}, {}
        for name in ('stationary', 'null', 'change'):
            views[name], clean[name] = sparse.teacher_labels(xs, noise, name)
        gen = torch.Generator(device=device).manual_seed(10_000_019)
        xt = torch.rand((a.test_samples, cfg.input_dim), generator=gen, device=device) < cfg.feature_prob
        test_y = [xt[:, 0].float(), xt[:, 1].float()]
        meta = {'source': 'covariance_sparse_eval_v1.draw_stream/teacher_labels', 'settings': asdict(cfg)}
        return xs, views, clean, xt, test_y, None, meta
    if a.task == 'stock':
        cfg = stock.stock_stream.Args(seed=1, steps=0)
        bars = stock.stock_stream.read_bars(a.bars)
        xx, yy = stock.stock_stream.build_stream(bars, cfg)
        phases = stock.phase_ranges(len(yy), 2000)
        views, signal, plant = stock.build_views(xx, yy, phases['prefix_all'][1], phases['suffix_positive'][1], 1)
        meta = {'source': 'covariance_stock_eval_v1 shared views', 'settings': asdict(cfg), 'planting': plant,
                'data_sha256': {'features': hashlib.sha256(memoryview(xx).cast('B')).hexdigest(),
                                'target': hashlib.sha256(memoryview(yy).cast('B')).hexdigest()},
                'planted_signal': signal}
        return torch.as_tensor(xx, device=device), {k: torch.as_tensor(v, device=device) for k, v in views.items()}, {}, None, None, None, meta
    # Identical task-generation RNG order/precision to the frozen v2 dense benchmark.
    runtime.configure_runtime()
    cfg = original.Args(hetero=a.hetero, switch_at=.5)
    gen = torch.Generator(device=device).manual_seed(1)
    t1, t2 = original.draw_teacher(cfg, gen, device), original.draw_teacher(cfg, gen, device)
    initial = original.init_weights(cfg, gen, device)
    xs = torch.randn(cfg.samples, cfg.input_dim, generator=gen, device=device)
    direction = torch.randn(cfg.input_dim, generator=gen, device=device)
    direction /= direction.norm()
    sigma = (a.hetero * torch.tanh(xs @ direction)).exp()
    first = original.teach(t1, xs)
    changed = first.clone()
    changed[cfg.samples // 2:] = original.teach(t2, xs[cfg.samples // 2:])
    noise = sigma * torch.randn(cfg.samples, generator=gen, device=device)
    torch.randn(cfg.validation, cfg.input_dim, generator=gen, device=device)
    xt = torch.randn(a.test_samples, cfg.input_dim, generator=gen, device=device)
    test_y = [original.teach(t1, xt), original.teach(t2, xt)]
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    clean = {'stationary': first, 'null': torch.zeros_like(first), 'change': changed}
    return xs, {k: y + noise for k, y in clean.items()}, clean, xt, test_y, initial, {'settings': asdict(cfg), 'source': 'frozen v2 RNG order'}


def save(root, result):
    temporary = root / 'results.json.tmp'
    temporary.write_text(json.dumps(original.finite_json(result), indent=2, allow_nan=False) + '\n')
    temporary.replace(root / 'results.json')


def save_partial(root, view, runner, model, consumed, policy):
    """Persist consumed outputs and predictive state, never dense covariance."""
    np.save(root / f'{view}_predictions.npy', runner.predictions[:consumed].cpu().numpy())
    np.save(root / f'{view}_adam_predictions.npy', runner.baseline_predictions[:consumed].cpu().numpy())
    posterior_means = {
        family: (expert.log_odds.sigmoid() * expert.slab_mean
                 if isinstance(expert, SparsePosterior) else expert.mean).detach().cpu()
        for family, expert in zip(model.configs, model.filters)}
    state = {
        'processed_observations': consumed, 'autocull': policy,
        'posterior_mean_weights': posterior_means,
        'next_prior_mean_weights': {family: expert.mean_weights().detach().cpu()
                                    for family, expert in zip(model.configs, model.filters)},
        'log_weights': model.log_weights.detach().cpu(),
        'aggregation_weights': model.aggregation_weights().detach().cpu(),
        'log_prior': model.log_prior.detach().cpu(), 'noise': model.noise.detach().cpu(),
        'hazard': model.hazard, 'noise_rate': model.noise_rate, 'raw_scale': model.raw_scale,
        'expert_configs': model.configs,
        'adam_weights': [weight.detach().cpu() for weight in runner.baseline.weights],
        'note': 'Predictive snapshot only; covariance and optimizer state omitted; not resumable.'}
    temporary = root / f'{view}_partial_state.pt.tmp'
    torch.save(state, temporary)
    temporary.replace(root / f'{view}_partial_state.pt')


@torch.no_grad()
def main():
    a = tyro.cli(Args)
    if a.seed != 1 or min(a.projected_dim, a.log_every, a.feature_chunk, a.test_samples) <= 0:
        raise ValueError('seed1 and positive dimensions/cadences required')
    if not 0 <= a.mixture_hazard < 1 or not 0 <= a.noise_rate < 1:
        raise ValueError('hazard and noise rate must be probabilities below one')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    device = torch.device('cuda')
    root = Path(a.output) / f'{a.task}__posterior_model_v1__1__{time.time_ns()}'
    root.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(root))
    result = {'args': asdict(a), 'run_dir': str(root), 'status': 'preparing', 'views': {},
              'protocol': {'horizons': 'maximum 60000 sparse/65536 dense/all stock bars; pruned runs are partial evidence',
                           'autocull': {'enabled': a.autocull, 'candidates': OUTPUT_NAMES,
                                        'evidence': 'interval prequential TRAIN metrics only; no heldout risk',
                                        'pruned_exit_code': PRUNED_EXIT_CODE},
                           'selection': 'Adam real/stationary first-quarter lock; posterior hyperparameters fixed ex ante, model weights learn only from past observations',
                           'scale': 'each expert has its own previous-residual-square EMA; not identified irreducible noise',
                           'aggregation': 'switching HMM/static Bayes/uniform/family controls share identical independent expert states',
                           'mean': 'zero initial functional mean, including nonlinear tangent models; original dense Adam initializer is retained',
                           'limitations': 'one seed; approximate expert posteriors; no global optimality or financial significance claim'}}
    save(root, result)
    try:
        xs, views, clean, xt, test_y, initial, meta = dataset(a, device)
        signal = meta.pop('planted_signal', None)
        if signal is not None:
            np.save(root / 'planted_signal.npy', signal)
        result['dataset'] = meta
        n, d = xs.shape
        prefix = n // 4
        result['maximum_observations_per_view'] = n
        result['maximum_total_observations'] = n * len(views)
        result['processed_observations'] = 0
        switch = prefix + (n - prefix) // 2 if a.task == 'stock' else n // 2
        phases = {'prefix': (0, prefix), 'suffix_before_change': (prefix, switch),
                  'suffix_after_change': (switch, n), 'suffix': (prefix, n)}
        result['phases'] = phases
        encoder = TangentFeatures(d, projected_dim=a.projected_dim, hidden=64, seed=1, device=str(device))
        features = torch.empty((n, encoder.output_dim), device=device)
        for start in range(0, n, a.feature_chunk):
            end = min(start + a.feature_chunk, n)
            features[start:end].copy_(encoder.transform(xs[start:end].float()))
        test_features = None
        if xt is not None:
            test_features = torch.empty((len(xt), encoder.output_dim), device=device)
            for start in range(0, len(xt), a.feature_chunk):
                end = min(start + a.feature_chunk, len(xt))
                test_features[start:end].copy_(encoder.transform(xt[start:end].float()))
        result['feature_map'] = encoder.metadata()
        del encoder
        result['feature_dim'] = features.shape[1]
        result['output_names'] = OUTPUT_NAMES
        grid = stock.Args.adam_lrs
        if a.task == 'dense':
            grid = original.Args.adam_lrs
        lock = None
        real_predictions = None
        for view, ys in views.items():
            torch.cuda.reset_peak_memory_stats()
            model = ModelMixture(d, features.shape[1], device, a.mixture_hazard, a.noise_rate)
            result['expert_configs'] = model.configs
            if initial is None:
                cfg = sparse.Args(input_dim=d, steps=n, noise_rate=a.noise_rate)
                baseline = sparse.LinearLearner('adam', grid, cfg, xs, ys)
            else:
                cfg = original.Args(samples=n, input_dim=d, graph_steps=1)
                baseline = stock.MeasuredLearner('adam', grid, initial, cfg, xs, ys)
            runner = StreamRunner(xs, features, ys, model, baseline)
            cull_metrics = {'error_ratio': 1e-4}
            if view in ('null', 'random_sign'):
                cull_metrics['prediction_energy_ratio'] = 1e-4
            elif view in clean:
                cull_metrics['prequential_clean_mse'] = 1e-4
            policy = ProxyCull(candidates=len(OUTPUT_NAMES), metrics=cull_metrics) if a.autocull else None
            changes_regime = view == 'change' or (a.task == 'stock' and view == 'planted')
            graph_steps = 1 if d >= 1024 else 16
            started = time.perf_counter()
            graphs = runner.capture(graph_steps)
            startup = time.perf_counter() - started
            row = {'status': 'running', 'startup_seconds': startup, 'curves': [],
                   'maximum_observations': n, 'processed_observations': 0,
                   'autocull': policy.state_dict() if policy is not None else {'enabled': False}}
            result['views'][view] = row
            result['status'] = 'running'
            save(root, result)
            previous = 0
            started = time.perf_counter()
            endpoints = sorted(set(range(a.log_every, n, a.log_every)) | {prefix, switch, n})
            for end in endpoints:
                blocks, tail = divmod(end - previous, graph_steps)
                for _ in range(blocks):
                    graphs[graph_steps].replay()
                for _ in range(tail):
                    graphs[1].replay()
                torch.cuda.synchronize()
                pred = runner.predictions[previous:end].cpu().numpy()
                bp = runner.baseline_predictions[previous:end].cpu().numpy()
                target = ys[previous:end].cpu().numpy()
                metrics = stock.metric_sums(target, pred)
                curve = {'step': end, 'interval_start': previous, 'prequential': metrics,
                         'adam_grid_prequential': stock.metric_sums(target, bp),
                         'model_weights': model.aggregation_weights()[:2].cpu().tolist()}
                if not np.isfinite(pred).all() or not torch.isfinite(model.noise).all():
                    raise RuntimeError(f'nonfinite predictive model at {view}/{end}; no expert silently discarded')
                if view in clean:
                    cy = clean[view][previous:end].cpu().numpy()
                    curve['prequential_clean_mse'] = np.square(pred.astype(np.float64) - cy[:, None]).mean(0).tolist()
                if end == prefix and lock is None:
                    lock = stock.select_prefix(runner.baseline_predictions[:prefix].cpu().numpy(),
                                               ys[:prefix].cpu().numpy(), grid, 0, prefix, end)
                    result['adam_prefix_lock'] = lock
                    save(root, result)
                if xt is not None:
                    assert test_y is not None and test_features is not None
                    truth = torch.zeros_like(test_y[0]) if view == 'null' else test_y[1 if view == 'change' and end >= switch else 0]
                    frozen = model.frozen_predictions(xt, test_features)
                    curve['heldout_clean_mse'] = (frozen.double() - truth[:, None]).square().mean(0).cpu().tolist()
                    if initial is None:
                        bp_test = xt.float() @ baseline.weights[0][:, 0, :].T
                        support = torch.zeros(d, device=device)
                        if view != 'null':
                            support[1 if view == 'change' and end >= switch else 0] = 1
                        curve['adam_exact_clean_risk'] = {k: v.cpu().tolist() for k, v in sparse.exact_risk(
                            baseline.weights[0][:, 0, :], support, .01).items()}
                    else:
                        bp_test = original.forward(baseline.weights, xt)[2].T
                    curve['adam_grid_heldout_mse'] = (bp_test.double() - truth[:, None]).square().mean(0).cpu().tolist()
                row['curves'].append(curve)
                for name, metric in zip(OUTPUT_NAMES, metrics):
                    writer.add_scalar(f'{view}/{name}/error_ratio', metric['error_ratio'], end)
                    writer.add_scalar(f'{view}/{name}/prediction_energy_ratio', metric['prediction_energy_ratio'], end)
                if 'heldout_clean_mse' in curve:
                    for name, score in zip(OUTPUT_NAMES, curve['heldout_clean_mse']):
                        writer.add_scalar(f'{view}/{name}/heldout_clean_mse', score, end)
                writer.flush()
                row['processed_observations'] = int(runner.index)
                if row['processed_observations'] != end or int(baseline.index) != end:
                    raise RuntimeError('stream/optimizer clocks diverged')
                record = None
                if policy is not None:
                    evidence = {'error_ratio': [metric['error_ratio'] for metric in metrics]}
                    if 'prediction_energy_ratio' in cull_metrics:
                        evidence['prediction_energy_ratio'] = [metric['prediction_energy_ratio'] for metric in metrics]
                    if 'prequential_clean_mse' in cull_metrics:
                        evidence['prequential_clean_mse'] = curve['prequential_clean_mse']
                    # The checkpoint ending at the switch still measures the old regime.
                    post_switch = changes_regime and previous >= switch
                    record = policy.observe(end, evidence,
                                            phase='after_change' if post_switch else 'initial',
                                            phase_start=switch if post_switch else 0)
                    if changes_regime and end == switch:
                        # An old-regime plateau cannot cancel the imminent reset.
                        record = None
                    row['autocull'] = policy.state_dict()
                curve['autocull'] = row['autocull']
                row['seconds'] = time.perf_counter() - started
                result['processed_observations'] = sum(v['processed_observations'] for v in result['views'].values())
                if record is not None:
                    save_partial(root, view, runner, model, end, row['autocull'])
                    row['status'] = 'pruned'
                    row['pruning'] = record
                    row['partial_state'] = f'{view}_partial_state.pt'
                    row['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                    row['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                    result['status'] = 'pruned'
                    result['pruned_view'] = view
                    save(root, result)
                    prune_proxy(root, view, record)
                save(root, result)
                previous = end
            prediction = runner.predictions.cpu().numpy()
            bp = runner.baseline_predictions.cpu().numpy()
            target = ys.cpu().numpy()
            row['phase_metrics'] = {name: stock.metric_sums(target[s:e], prediction[s:e]) for name, (s, e) in phases.items()}
            row['adam_phase_metrics'] = {name: stock.metric_sums(target[s:e], bp[s:e]) for name, (s, e) in phases.items()}
            row['seconds'] = time.perf_counter() - started
            row['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            row['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            row['status'] = 'completed'
            np.save(root / f'{view}_predictions.npy', prediction)
            np.save(root / f'{view}_adam_predictions.npy', bp)
            if view == 'real':
                real_predictions = prediction.copy()
            if view == 'planted':
                row['paired_injection_response'] = stock.paired_response(real_predictions, prediction, signal, phases)
            save(root, result)
            print(json.dumps({'view': view, 'observations': n, 'seconds': row['seconds'],
                              'suffix_ratios': [r['error_ratio'] for r in row['phase_metrics']['suffix']]}), flush=True)
            del graphs, runner, baseline, model
            torch.compiler.reset()
            torch.cuda.empty_cache()
        result['status'] = 'completed'
        save(root, result)
        print(f'RESULTS {root / "results.json"}', flush=True)
    except ProxyPruned:
        raise SystemExit(PRUNED_EXIT_CODE) from None
    except Exception as exc:
        result['status'] = 'failed'
        result['failure'] = repr(exc)
        save(root, result)
        raise
    finally:
        writer.close()


if __name__ == '__main__':
    main()
