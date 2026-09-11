"""Frozen global-hazard uncertainty, with v5's prefix-selected AdamW control.

Run full CUDA views through mlq. Development is the reused namespace0 stream;
confirmation is namespace200, not an independent seed. Clean targets only score.
SIGTERM requests a stop at the next reporting boundary, not resumable training.
"""

import hashlib
import inspect
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.plasticity import predictive_segment_eval_v5 as v5
from cleanrl.plasticity.predictive_hazard_mixture_v6 import HazardMixture
from cleanrl.shared import runtime


@dataclass
class Args:
    lock: str
    view: Literal['stationary', 'change', 'null', 'recurrent', 'two_support'] = 'stationary'
    phase: Literal['development', 'confirmation'] = 'development'
    seed: int = 1
    graph_steps: int = 100
    log_every: int = 1000
    output: str = 'runs'


def family_contract():
    return {'hazards': [0., 1e-5, 1e-4, 1e-3], 'hyperprior': [.25] * 4, 'budget': 32,
            'live_segment_branches': 128,
            'compute_comparison': 'four budget32 children; not budget matched to a single v5 model',
            'noise_rate': .001, 'primary': 'hazard_mixture',
            'hyperupdate': 'cumulative proper child Gaussian mixture predictive log density; no forgetting',
            'parameter_approximation': 'v5 bounded segment and factorized coordinate beliefs within each hazard',
            'selection': 'none; original v5 stationary development prefix AdamW lock imported before draws',
            'stream': v5.family_contract()['stream'], 'views': v5.family_contract()['views'],
            'namespaces': {**v5.family_contract()['namespaces'], 'confirmation': 200},
            'controls': ['hazard_0', 'hazard_1e-5', 'hazard_1e-4', 'hazard_1e-3',
                         'v3_likelihood_static', 'v4_birth_only', 'v4_continuous', 'zero',
                         'selected_adamw', 'legacy_adam_1e-5'],
            'culling': 'disabled; all views fixed60000 unless interrupted or failed'}


def source_hashes():
    return {**{f'v5_{key}': value for key, value in v5.source_hashes().items()},
            'v6_evaluator': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'v6_model': hashlib.sha256(Path(inspect.getfile(HazardMixture)).read_bytes()).hexdigest()}


def load_protocol(args):
    """Validate the original lock before any feature/noise/label generation."""
    if args.seed != 1 or not 1 <= args.graph_steps <= 15000 or args.log_every <= 0:
        raise ValueError('seed1, graph_steps in [1,15000], and positive logging cadence required')
    if args.phase not in ('development', 'confirmation') or args.view not in v5.family_contract()['views']:
        raise ValueError('unknown phase or view')
    lock, digest = v5.load_lock(args.lock, v5.source_hashes())
    namespace = 0 if args.phase == 'development' else 200
    return lock, digest, namespace


class ComparisonModel:
    output_names = (*HazardMixture.output_names, 'v3_likelihood_static',
                    'v4_birth_only', 'v4_continuous', 'zero')
    child_names = HazardMixture.output_names[1:]

    def __init__(self, dimension, device):
        self.mixture = HazardMixture(dimension, device, budget=32)
        self.segments = self.mixture.segments
        # Construct controls directly: no redundant v5 segment bank.
        self.v3 = v5.PredictiveMeanRisk(dimension, device)
        self.v4 = v5.PersistentSupport(dimension, device)
        self._zero = torch.zeros(1, device=device)

    def state_tensors(self):
        return [tensor for model in (self.mixture, self.v3, self.v4) for tensor in model.state_tensors()]

    def mean_weights(self):
        return torch.cat((self.mixture.mean_weights(), self.v3.mean_weights()[:1],
                          self.v4.mean_weights()[:2], torch.zeros_like(self.v3.filter.slab_mean[:1])))

    def update(self, x, y):
        return torch.cat((self.mixture.update(x, y), self.v3.update(x, y)[:1],
                          self.v4.update(x, y)[:2], self._zero))

    def diagnostics(self):
        return {'mixture': self.mixture.diagnostics(),
                'v3_aggregation': self.v3.aggregation_weights().cpu().tolist(),
                'v3_inclusion_0_1': self.v3.filter.log_odds[:, :2].sigmoid().cpu().tolist(),
                'v3_slab_mean_0_1': self.v3.filter.slab_mean[:, :2].cpu().tolist(),
                'v4': self.v4.diagnostics()}


class TracedRunner(v5.Runner):
    """Inherited complete-state capture/reset; trace row t consumes exactly y[t]."""

    def __init__(self, xs, ys, model, adam):
        super().__init__(xs, ys, model, adam)
        self.traces = [torch.zeros((len(xs), segment.budget + 1, 13), device=xs.device)
                       for segment in model.segments]
        self.discards = torch.zeros((len(xs), len(model.segments)), device=xs.device)
        self.hazard_weights_before = torch.zeros_like(self.discards)
        self.hazard_weights_after = torch.zeros_like(self.discards)
        self.predictive_log_prob = torch.zeros_like(self.discards)
        self.mutable.extend((*self.traces, self.discards, self.hazard_weights_before,
                             self.hazard_weights_after, self.predictive_log_prob))

    def update(self):
        # reshape alone aliases the parent's incremented clock; clone BEFORE update.
        ix = self.index.clone().reshape(1)
        self.hazard_weights_before.index_copy_(0, ix, self.model.mixture.log_hazard_weights.exp().unsqueeze(0))
        super().update()
        self.hazard_weights_after.index_copy_(0, ix, self.model.mixture.log_hazard_weights.exp().unsqueeze(0))
        for trace, segment in zip(self.traces, self.model.segments):
            trace.index_copy_(0, ix, segment.trace().unsqueeze(0))
        self.discards.index_copy_(0, ix, torch.stack([s.discarded_mass for s in self.model.segments]).unsqueeze(0))
        self.predictive_log_prob.index_copy_(
            0, ix, torch.stack([s.predictive_log_prob for s in self.model.segments]).unsqueeze(0))


def next_forecast_risk(weights, view, step, probability):
    stale_index = None
    if view == 'change' and step >= 30000:
        stale_index = 0
    elif view == 'recurrent' and step >= 20000:
        stale_index = 0 if step < 40000 else 1
    return v5.sparse.exact_risk(weights, v5.support_at(view, step, weights.shape[-1], weights.device),
                                probability, stale_index=stale_index)


def error_summary(prediction, noisy, clean, view):
    noisy_error = np.square(prediction.astype(np.float64) - noisy.astype(np.float64)[:, None])
    clean_error = np.square(prediction.astype(np.float64) - clean.astype(np.float64)[:, None])
    count = len(prediction)
    def interval(lo, hi):
        return {'start': lo, 'end': hi, 'count': hi - lo,
                'noisy_squared_error_sum': noisy_error[lo:hi].sum(0).tolist(),
                'clean_squared_error_sum': clean_error[lo:hi].sum(0).tolist(),
                'noisy_mse': noisy_error[lo:hi].mean(0).tolist(),
                'clean_mse': clean_error[lo:hi].mean(0).tolist()}
    bounds = v5.phase_boundaries(view)
    summary = {'consumed_interval': [0, count], 'full_noisy_mse': None, 'full_clean_mse': None,
               'phase_errors': [], 'post_change_errors': None,
               'post_change_clean_mse': None, 'post_change_noisy_mse': None}
    if count:
        whole = interval(0, count)
        summary.update(full_noisy_mse=whole['noisy_mse'], full_clean_mse=whole['clean_mse'],
                       noisy_squared_error_sum=whole['noisy_squared_error_sum'],
                       clean_squared_error_sum=whole['clean_squared_error_sum'],
                       phase_errors=[interval(lo, min(hi, count)) for lo, hi in zip(bounds, bounds[1:]) if lo < count])
        if len(bounds) > 2 and count > bounds[1]:
            summary['post_change_errors'] = interval(bounds[1], count)
            summary.update(post_change_clean_mse=summary['post_change_errors']['clean_mse'],
                           post_change_noisy_mse=summary['post_change_errors']['noisy_mse'])
    return summary, noisy_error, clean_error


def save_predictions(root, runner, ys, clean, step, view):
    pred = runner.predictions[:step].cpu().numpy()
    noisy, truth = ys[:step].cpu().numpy(), clean[:step].cpu().numpy()
    summary, noisy_error, clean_error = error_summary(pred, noisy, truth, view)
    temporary = root / 'prequential.tmp.npz'
    np.savez(temporary, predictions=pred, noisy_target=noisy, clean_target=truth,
             noisy_squared_error=noisy_error, clean_squared_error=clean_error)
    temporary.replace(root / 'prequential.npz')
    return summary


def save_analysis_state(root, runner, step):
    temporary = root / 'branch_traces.tmp.npz'
    np.savez(temporary, **{name: trace[:step].cpu().numpy()
                         for name, trace in zip(runner.model.child_names, runner.traces)},
             discarded_mass=runner.discards[:step].cpu().numpy(),
             hazard_weights_before=runner.hazard_weights_before[:step].cpu().numpy(),
             hazard_weights_after=runner.hazard_weights_after[:step].cpu().numpy(),
             predictive_log_prob=runner.predictive_log_prob[:step].cpu().numpy())
    temporary.replace(root / 'branch_traces.npz')
    temporary = root / 'checkpoint.pt.tmp'
    torch.save({'step': step, 'model_state': [t.cpu() for t in runner.model.state_tensors()],
                'adam_state': [t.cpu() for t in runner.adam.mutable],
                'next_mean_weights': torch.cat((runner.model.mean_weights(), runner.adam.mean_weights())).cpu(),
                'note': 'analysis state only; no resume implementation'}, temporary)
    temporary.replace(root / 'checkpoint.pt')


@torch.no_grad()
def main():
    args = tyro.cli(Args)
    lock, lock_hash, namespace = load_protocol(args)
    sources = source_hashes()
    cfg = v5.sparse.Args(seed=1, graph_steps=args.graph_steps, autocull=False)
    configs = [lock['selected_adamw']]
    root = Path(args.output) / f'SparseStream__hazard_v6_{args.phase}_{args.view}__1__{time.time_ns()}'
    root.mkdir(parents=True, exist_ok=True)
    result = {'args': asdict(args), 'status': 'preparing', 'families': family_contract(),
              'source_sha256_before': sources, 'transferred_lock_sha256': lock_hash,
              'transferred_lock_source_contract': lock['source_sha256'],
              'transferred_lock_families': lock['families'], 'selected_adamw': configs[0],
              'namespace': namespace, 'curves': [], 'processed_observations': 0,
              'consumed_observations': 0, 'validated_observations': 0,
              'child_names': ComparisonModel.child_names,
              'branch_trace_columns': ['retained_mass', 'birth_observation', 'ig_count', 'ig_Q', 'ig_variance',
                                       'inclusion_0', 'inclusion_1', 'slab_mean_0', 'slab_mean_1',
                                       'effective_0', 'effective_1', 'sum_inclusion', 'effective_squared_norm'],
              'trace_timing': {'branches': 'post-label; row t consumed y[t]; branch0=null',
                               'hazard_weights_before': 'pre-label weights used to forecast y[t]',
                               'hazard_weights_after': 'posterior after y[t]',
                               'predictive_log_prob': 'child pre-label Gaussian mixture log density at y[t]'},
              'scientific_limitations': [
                  'Bayes over four global hazards only; child parameter beliefs remain bounded/factorized approximations',
                  'proper Gaussian mixture scoring still uses approximate segment IG moments, not an exact joint posterior',
                  'null histories share one past-only scale; recurrence does not retrieve stored old segments',
                  'fixed global hazards and expected-one-coordinate prior are misspecified for denser/asynchronous tasks',
                  'namespace0 development reuses consumed data; namespace200 is named confirmation, not a new seed',
                  'v5 noisy prefix selected AdamW; no new configuration or hazard-prior selection',
                  'informative sparse priors prevent generic optimizer superiority claims; brain analogy is hypothesis only',
                  'joint replay timing includes controls and traces; local discarded mass is not a global error bound',
                  'SIGTERM stops at reporting boundaries; SIGKILL/device failure may leave only last durable raw boundary',
                  'checkpoint is analysis-only, not resumable; partial prefixes are not full-horizon evidence']}
    v5.sparse.save_json(root / 'results.json', result)
    writer = runner = xs = noise = ys = clean = None
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    previous_handler = signal.signal(signal.SIGTERM, request_stop)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU model fallback')
        runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
        result['runtime'] = {'torch': torch.__version__, 'numpy': np.__version__, 'cuda': torch.version.cuda,
                             'device': torch.cuda.get_device_name(), 'dtype': 'float32',
                             'compile': 'fullgraph', 'tf32': False}
        writer = SummaryWriter(str(root))
        # This exact v5 draw function/chunking uses independent feature/noise generators.
        draw_cfg = v5.sparse.Args(seed=1 + namespace * 10000019)
        xs, noise = v5.sparse.draw_stream(draw_cfg, torch.device('cuda'))
        result.update(feature_sha256=v5.tensor_hash(xs), noise_sha256=v5.tensor_hash(noise))
        if args.phase == 'development' and any(result[key] != lock[key] for key in ('feature_sha256', 'noise_sha256')):
            raise ValueError('development transfer must reproduce both paired v5 feature and noise streams')
        ys, clean = v5.labels(xs, noise, args.view)
        result.update(target_sha256=v5.tensor_hash(ys), clean_sha256=v5.tensor_hash(clean))
        model = ComparisonModel(cfg.input_dim, xs.device)
        adam = v5.AdamWBank(configs, cfg, xs, ys)
        runner = TracedRunner(xs, ys, model, adam)
        names = (*model.output_names, f'adamw_{v5.adam_grid().index(configs[0]):02d}', 'legacy_adam_1e-5')
        result.update(output_names=names, adamw_configs=configs,
                      mutable_model_bytes={name: sum(t.numel() * t.element_size() for t in m.state_tensors())
                                           for name, m in (('hazard_mixture', model.mixture),
                                                           ('v3_all_rows', model.v3), ('v4_all_rows', model.v4))})
        start = time.perf_counter()
        graphs = runner.capture(args.graph_steps)
        torch.cuda.synchronize()
        result.update(capture_seconds=time.perf_counter() - start, status='running')
        boundaries = v5.phase_boundaries(args.view)
        endpoints = sorted(set(range(args.log_every, cfg.steps, args.log_every)) | set(boundaries[1:]))
        previous = 0
        start = time.perf_counter()
        for step in endpoints:
            if stop_requested:
                break
            blocks, tail = divmod(step - previous, args.graph_steps)
            for _ in range(blocks):
                graphs[args.graph_steps].replay()
            for _ in range(tail):
                graphs[1].replay()
            torch.cuda.synchronize()
            result['consumed_observations'] = int(runner.index)
            if int(runner.index) != step or int(adam.index) != step or any(int(s.observations) != step for s in model.segments):
                raise RuntimeError('stream clocks diverged')
            predictions = runner.predictions[previous:step].cpu().numpy().astype(np.float64)
            target = ys[previous:step].cpu().numpy().astype(np.float64)
            truth = clean[previous:step].cpu().numpy().astype(np.float64)
            weights = torch.cat((model.mean_weights(), adam.mean_weights()))
            finite_traces = (runner.discards, runner.hazard_weights_before,
                             runner.hazard_weights_after, runner.predictive_log_prob, *runner.traces)
            if (not np.isfinite(predictions).all() or not bool(torch.isfinite(weights).all())
                    or any(not bool(torch.isfinite(t[previous:step]).all()) for t in finite_traces)):
                raise RuntimeError(f'nonfinite forecast or trace at observation {step}')
            risk = next_forecast_risk(weights, args.view, step, cfg.feature_prob)
            noisy_error = np.square(predictions - target[:, None])
            clean_error = np.square(predictions - truth[:, None])
            row = {'step': step, 'interval_start': previous, 'interval_count': step - previous,
                   'noisy_mse': noisy_error.mean(0).tolist(),
                   'clean_mse': clean_error.mean(0).tolist(),
                   'noisy_squared_error_sum': noisy_error.sum(0).tolist(),
                   'clean_squared_error_sum': clean_error.sum(0).tolist(),
                   'exact_risk': {key: value.cpu().tolist() for key, value in risk.items()},
                   'coefficient_0_1': weights[:, :2].cpu().tolist(), 'diagnostics': model.diagnostics()}
            result['curves'].append(row)
            result.update(processed_observations=step, validated_observations=step,
                          replay_seconds=time.perf_counter() - start, finalrisk=row['exact_risk'])
            for i, name in enumerate(names):
                for metric in ('noisy_mse', 'clean_mse'):
                    writer.add_scalar(f'{name}/{metric}', row[metric][i], step)
                writer.add_scalar(f'{name}/next_forecast_risk', row['exact_risk']['clean_mse'][i], step)
            for i, (name, segment) in enumerate(zip(model.child_names, model.segments)):
                writer.add_scalar(f'{name}/discarded_mass', float(segment.discarded_mass), step)
                writer.add_scalar(f'{name}/cumulative_discarded_mass', float(segment.cumulative_discarded_mass), step)
                writer.add_scalar(f'{name}/hazard_posterior', float(model.mixture.log_hazard_weights[i].exp()), step)
            writer.add_scalar('joint_updates_per_second', step / result['replay_seconds'], step)
            writer.flush()
            result.update(save_predictions(root, runner, ys, clean, step, args.view))
            result['durable_raw_observations'] = step
            v5.sparse.save_json(root / 'results.json', result)
            previous = step
        result['status'] = 'completed' if previous == cfg.steps else 'interrupted'
    except Exception as error:
        result.update(status='failed', failure=repr(error))
        raise
    finally:
        # Best-effort failure evidence must not disguise the original exception.
        try:
            result['source_sha256_after'] = source_hashes()
            for key, tensor in (('feature', xs), ('noise', noise), ('target', ys), ('clean', clean)):
                if tensor is not None:
                    result[f'{key}_sha256_after'] = v5.tensor_hash(tensor)
            if runner is not None:
                step = int(runner.index)
                result.update(consumed_observations=step, adam_consumed_observations=int(runner.adam.index),
                              child_consumed_observations=[int(s.observations) for s in runner.model.segments])
                if not 0 <= step <= len(xs):
                    raise RuntimeError('runner consumed bounds are invalid; refusing misleading snapshot')
                result.update(save_predictions(root, runner, ys, clean, step, args.view))
                result['durable_raw_observations'] = step
                save_analysis_state(root, runner, step)
                result['analysis_checkpoint_observations'] = step
                result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved())
            mutated = result['source_sha256_after'] != sources or any(
                result.get(f'{key}_sha256_after') != result.get(f'{key}_sha256')
                for key in ('feature', 'noise', 'target', 'clean') if f'{key}_sha256' in result)
            if mutated:
                result.update(status='failed', integrity_failure='source/data mutation during experiment')
        except Exception as error:
            result.update(status='failed', evidence_failure=repr(error))
        finally:
            v5.sparse.save_json(root / 'results.json', result)
            signal.signal(signal.SIGTERM, previous_handler)
            if writer is not None:
                writer.close()
    print(f'RESULTS {root / "results.json"}', flush=True)
    if result['status'] == 'failed':
        raise RuntimeError(result.get('evidence_failure', result.get('integrity_failure', 'experiment failed')))
    if result['status'] == 'interrupted':
        raise SystemExit(128 + signal.SIGTERM)


if __name__ == '__main__':
    main()
