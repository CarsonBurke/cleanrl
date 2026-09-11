"""Full60000 causal structural prediction experiment; enqueue through mlq.

Development reuses namespace0. Confirmation uses new namespace300, not a new
seed. Original v5 prefix AdamW selection is authenticated before any draws.
No v6 confirmation results are read. Numerical failures remain failed evidence.
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

from cleanrl.plasticity import predictive_hazard_eval_v6 as v6
from cleanrl.plasticity import predictive_segment_eval_v5 as v5
from cleanrl.plasticity.predictive_structure_mixture_v7 import HAZARDS, SingletonSegment, StructuralMixture
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
    return {'primary': 'structural_mixture', 'hazards': list(HAZARDS), 'hazard_hyperprior': [.25] * 4,
            'structure_hyperprior': [.5, .5], 'budget': 32, 'top_mass_hazard': 1e-4,
            'within_segment': {'null_prior': .5, 'singleton_prior': '1/(2D)',
                               'alpha0': 2., 'beta0': 1., 'conditional_slab_variance': 'sigma_squared'},
            'evidence': 'exact NIG Student-t support mixture; frozen v6 Gaussian moment mixture for hedge',
            'compression': 'floor(log(age)/log(1.5)); posterior categorical representative; retain full bucket mass',
            'approximation': 'exact within-segment specialized prior; stochastic approximate change-history posterior',
            'stream': v5.family_contract()['stream'], 'views': v5.family_contract()['views'],
            'namespaces': {'development': 0, 'confirmation': 300},
            'resampling': {'namespace_base': 700, 'seed_rule': '1+(700+data_namespace)*10000019',
                           'generator': 'independent CUDA torch.Generator', 'shape': [60000, 4, 32],
                           'dtype': 'float64', 'consumption': '128 uniforms/label including empty buckets and zero hazard'},
            'outputs': [*ComparisonModel.output_names, 'selected_adamw', 'legacy_adam_1e-5'],
            'selection': 'none; original v5 stationary development prefix lock imported before draws',
            'capacity': {'singleton_retained_slots': 160, 'factorized_retained_slots': 128,
                         'reachable_age_buckets_at_60000': 28,
                         'note': 'slots are capacity, not occupied branches; occupancy measured from finite log mass; unequal compute'},
            'promotion': {'change_and_recurrent_gain_vs_v3': .20,
                          'stationary_and_two_support_max_regression_vs_v3': .05, 'null_clean_mse_max': 1e-5},
            'culling': 'disabled; full60000 in every view or honest failed/interrupted result'}


def source_hashes():
    return {**v6.source_hashes(),
            'v7_evaluator': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'v7_model': hashlib.sha256(Path(inspect.getfile(StructuralMixture)).read_bytes()).hexdigest(),
            'v7_plan': hashlib.sha256(Path('benchmarks/plasticity/structural_inference_v7_plan.json').read_bytes()).hexdigest()}


def load_protocol(args):
    if args.seed != 1 or not 1 <= args.graph_steps <= 15000 or args.log_every <= 0:
        raise ValueError('seed1, graph_steps in [1,15000], and positive logging cadence required')
    if args.phase not in ('development', 'confirmation') or args.view not in v5.family_contract()['views']:
        raise ValueError('unknown phase or view')
    lock, digest = v5.load_lock(args.lock, v5.source_hashes())
    return lock, digest, 0 if args.phase == 'development' else 300


def draw_resampling(steps, namespace, device):
    generator = torch.Generator(device=device).manual_seed(1 + (700 + namespace) * 10000019)
    return torch.rand((steps, 4, 32), device=device, dtype=torch.float64, generator=generator)


class ComparisonModel:
    output_names = (*StructuralMixture.output_names, 'v3_likelihood_static', 'zero')
    child_names = ('singleton_hazard_0', 'singleton_hazard_1e-5', 'singleton_hazard_1e-4',
                   'singleton_hazard_1e-3', 'singleton_top_mass_1e-4',
                   'v6_hazard_0', 'v6_hazard_1e-5', 'v6_hazard_1e-4', 'v6_hazard_1e-3')

    def __init__(self, dimension, device, uniforms):
        self.mixture = StructuralMixture(dimension, device, uniforms)
        self.segments = self.mixture.segments
        self.v3 = v5.PredictiveMeanRisk(dimension, device)
        self._zero = torch.zeros(1, device=device, dtype=torch.float64)

    def state_tensors(self):
        return [*self.mixture.state_tensors(), *self.v3.state_tensors()]

    def mean_weights(self):
        means = self.v3.mean_weights()[:1].double()
        return torch.cat((self.mixture.mean_weights(), means, torch.zeros_like(means)))

    def update(self, x, y):
        return torch.cat((self.mixture.update(x, y), self.v3.update(x, y)[:1].double(), self._zero))

    def diagnostics(self):
        return {'mixture': self.mixture.diagnostics(),
                'v3_aggregation': self.v3.aggregation_weights().cpu().tolist(),
                'v3_inclusion_0_1': self.v3.filter.log_odds[:, :2].sigmoid().cpu().tolist(),
                'v3_slab_mean_0_1': self.v3.filter.slab_mean[:, :2].cpu().tolist()}


class TracedRunner(v5.Runner):
    """Complete mutable state includes clocks, RNG consumption and every trace."""
    def __init__(self, xs, ys, model, adam):
        super().__init__(xs, ys, model, adam)
        self.predictions = self.predictions.double()
        self.mutable[1] = self.predictions
        self.traces = [torch.zeros((len(xs), s.budget + (0 if isinstance(s, SingletonSegment) else 1), 13),
                                   device=xs.device, dtype=torch.float64) for s in model.segments]
        self.branch_log_masses = [torch.zeros((len(xs), len(s.log_weights)), device=xs.device, dtype=torch.float64)
                                  for s in model.segments]
        self.discards = torch.zeros((len(xs), 9), device=xs.device, dtype=torch.float64)
        self.hazard_weights_before = torch.zeros((len(xs), 2, 4), device=xs.device, dtype=torch.float64)
        self.hazard_weights_after = torch.zeros_like(self.hazard_weights_before)
        self.structure_weights_before = torch.zeros((len(xs), 2), device=xs.device, dtype=torch.float64)
        self.structure_weights_after = torch.zeros_like(self.structure_weights_before)
        self.predictive_log_prob = torch.zeros_like(self.discards)
        self.structure_log_prob = torch.zeros_like(self.structure_weights_before)
        self.mass_errors = torch.zeros((len(xs), 5), device=xs.device, dtype=torch.float64)
        self.randomness_consumed = torch.zeros(len(xs), device=xs.device, dtype=torch.int64)
        self.trace_fields = ('discards', 'hazard_weights_before', 'hazard_weights_after',
                             'structure_weights_before', 'structure_weights_after', 'predictive_log_prob',
                             'structure_log_prob', 'mass_errors', 'randomness_consumed')
        self.mutable.extend((*self.traces, *self.branch_log_masses, *(getattr(self, key) for key in self.trace_fields)))

    def update(self):
        ix = self.index.clone().reshape(1)
        mixture = self.model.mixture
        self.hazard_weights_before.index_copy_(0, ix, torch.stack((mixture.singleton.log_hazard_weights.exp(),
                                                                 mixture.factorized.log_hazard_weights.exp().double())).unsqueeze(0))
        self.structure_weights_before.index_copy_(0, ix, mixture.log_structure_weights.exp().unsqueeze(0))
        x = self.xs.index_select(0, ix).squeeze(0).float()
        y = self.ys.index_select(0, ix).squeeze(0)
        prediction = self.model.update(x, y)
        self.adam.update()
        self.predictions.index_copy_(0, ix, torch.cat((prediction, self.adam.prediction.double())).unsqueeze(0))
        self.index.add_(1)
        self.hazard_weights_after.index_copy_(0, ix, torch.stack((mixture.singleton.log_hazard_weights.exp(),
                                                                mixture.factorized.log_hazard_weights.exp().double())).unsqueeze(0))
        self.structure_weights_after.index_copy_(0, ix, mixture.log_structure_weights.exp().unsqueeze(0))
        for trace, segment in zip(self.traces, self.model.segments):
            trace.index_copy_(0, ix, segment.trace().double().unsqueeze(0))
        for trace, segment in zip(self.branch_log_masses, self.model.segments):
            trace.index_copy_(0, ix, segment.log_weights.double().unsqueeze(0))
        self.discards.index_copy_(0, ix, torch.stack([s.discarded_mass.double() for s in self.model.segments]).unsqueeze(0))
        self.predictive_log_prob.index_copy_(0, ix, torch.stack([s.predictive_log_prob.double() for s in self.model.segments]).unsqueeze(0))
        self.structure_log_prob.index_copy_(0, ix, mixture.structure_log_prob.unsqueeze(0))
        self.mass_errors.index_copy_(0, ix, torch.stack([s.mass_error for s in self.model.segments[:5]]).unsqueeze(0))
        self.randomness_consumed.index_copy_(0, ix, mixture.singleton.uniforms_consumed.reshape(1))


def next_forecast_risk(weights, view, step, probability):
    return v6.next_forecast_risk(weights, view, step, probability)


def error_summary(prediction, noisy, clean, view):
    if prediction.ndim != 2 or len(prediction) != len(noisy) or len(prediction) != len(clean) or len(prediction) > 60000:
        raise ValueError('unaligned or out-of-horizon raw evidence')
    return v6.error_summary(prediction, noisy, clean, view)


def export_bound(runner, step):
    if not 0 <= step <= len(runner.xs) or step != int(runner.index):
        raise ValueError('export must end at the exact consumed observation')


def save_predictions(root, runner, ys, clean, step, view):
    export_bound(runner, step)
    if len(ys) != len(runner.xs) or len(clean) != len(runner.xs):
        raise ValueError('targets must align with the complete pregenerated stream')
    return v6.save_predictions(root, runner, ys, clean, step, view)


def save_analysis_state(root, runner, step):
    export_bound(runner, step)
    temporary = root / 'branch_traces.tmp.npz'
    np.savez(temporary, **{name: trace[:step].cpu().numpy() for name, trace in zip(runner.model.child_names, runner.traces)},
             **{name + '_log_mass': trace[:step].cpu().numpy()
                for name, trace in zip(runner.model.child_names, runner.branch_log_masses)},
             **{key: getattr(runner, key)[:step].cpu().numpy() for key in runner.trace_fields})
    temporary.replace(root / 'branch_traces.npz')
    temporary = root / 'checkpoint.pt.tmp'
    torch.save({'step': step, 'model_state': [t.cpu() for t in runner.model.state_tensors()],
                'adam_state': [t.cpu() for t in runner.adam.mutable],
                'runner_index': runner.index.cpu(),
                'resampling_sha256': v5.tensor_hash(runner.model.mixture.singleton.uniforms),
                'uniforms_consumed': int(runner.model.mixture.singleton.uniforms_consumed),
                'next_mean_weights': torch.cat((runner.model.mean_weights(), runner.adam.mean_weights().double())).cpu(),
                'note': 'analysis state only; RNG tape is separately authenticated; no resume CLI'}, temporary)
    temporary.replace(root / 'checkpoint.pt')


def storage_cost(model):
    groups = [('singleton_hazard_mixture', model.mixture.singleton), ('singleton_top_mass', model.mixture.top_mass),
              ('factorized_hazard_mixture', model.mixture.factorized), ('v3_all_rows', model.v3)]
    result = {}
    for name, child in groups:
        tensors = child.state_tensors()
        storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}
        result[name] = {'logical_mutable_bytes': sum(t.numel() * t.element_size() for t in tensors),
                        'unique_mutable_storage_bytes': sum(storages.values())}
    all_tensors = model.state_tensors()
    result['total_unique_mutable_storage_bytes'] = sum(
        {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in all_tensors}.values())
    result['shared_support_prior_bytes'] = model.mixture.singleton.segments[0].log_support_prior.untyped_storage().nbytes()
    result['sharing_note'] = 'One immutable resampling tape and one singleton support-prior tensor; independent sampled history banks.'
    return result


@torch.no_grad()
def main():
    args = tyro.cli(Args)
    lock, lock_hash, namespace = load_protocol(args)
    sources = source_hashes()
    cfg = v5.sparse.Args(seed=1, graph_steps=args.graph_steps, autocull=False)
    configs = [lock['selected_adamw']]
    root = Path(args.output) / f'SparseStream__structure_v7_{args.phase}_{args.view}__1__{time.time_ns()}'
    root.mkdir(parents=True, exist_ok=True)
    result = {'args': asdict(args), 'status': 'preparing', 'families': family_contract(),
              'source_sha256_before': sources, 'transferred_lock_sha256': lock_hash,
              'transferred_lock_source_contract': lock['source_sha256'],
              'transferred_lock_families': lock['families'], 'selected_adamw': configs[0],
              'namespace': namespace, 'resampling_namespace': 700 + namespace,
              'curves': [], 'processed_observations': 0, 'consumed_observations': 0, 'validated_observations': 0,
              'child_names': ComparisonModel.child_names,
              'singleton_trace_columns': SingletonSegment.trace_columns,
              'factorized_trace_columns': ['retained_mass', 'birth_observation', 'ig_count', 'ig_Q', 'ig_variance',
                                          'inclusion_0', 'inclusion_1', 'slab_mean_0', 'slab_mean_1',
                                          'effective_0', 'effective_1', 'sum_inclusion', 'effective_squared_norm'],
              'trace_timing': 'row t consumes y[t]; before weights predict y[t]; branch states/after weights include y[t]',
              'scientific_limitations': [
                  'Exact within-segment NIG posterior only under the mutually exclusive null/singleton specialized prior.',
                  'Log-age resampling preserves mass but approximates histories; no exact global posterior claim.',
                  'Frozen v6 density uses approximate factorized moments; structural hedge does not make it exact.',
                  'Global segment changes cannot represent asynchronous changes; recurrence has no saved segment retrieval.',
                  'Development reuses namespace0; namespace300 confirmation is not a new seed or iid significance claim.',
                  'Brain analogies supply hypotheses, not evidence; no generic optimizer, PPO, global-optimality or equal-compute claim.',
                  'SIGTERM stops at reporting boundaries; hard device failure can leave only the last durable boundary.',
                  'Checkpoints are analysis-only; partial prefixes are not full-horizon learning evidence.']}
    v5.sparse.save_json(root / 'results.json', result)
    writer = runner = xs = noise = ys = clean = uniforms = None
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    previous_handler = signal.signal(signal.SIGTERM, request_stop)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU fallback')
        runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
        result['runtime'] = {'torch': torch.__version__, 'numpy': np.__version__, 'cuda': torch.version.cuda,
                             'device': torch.cuda.get_device_name(), 'singleton_dtype': 'float64',
                             'frozen_controls_dtype': 'float32', 'compile': 'fullgraph', 'tf32': False}
        writer = SummaryWriter(str(root))
        draw_cfg = v5.sparse.Args(seed=1 + namespace * 10000019)
        xs, noise = v5.sparse.draw_stream(draw_cfg, torch.device('cuda'))
        result.update(feature_sha256=v5.tensor_hash(xs), noise_sha256=v5.tensor_hash(noise))
        if args.phase == 'development' and any(result[key] != lock[key] for key in ('feature_sha256', 'noise_sha256')):
            raise ValueError('development must reproduce both paired v5 streams')
        ys, clean = v5.labels(xs, noise, args.view)
        uniforms = draw_resampling(len(xs), namespace, xs.device)
        result.update(target_sha256=v5.tensor_hash(ys), clean_sha256=v5.tensor_hash(clean),
                      resampling_sha256=v5.tensor_hash(uniforms),
                      pregenerated_feature_elements=xs.numel(), pregenerated_noise_elements=noise.numel(),
                      pregenerated_uniforms=uniforms.numel(), resampling_bytes=uniforms.numel() * uniforms.element_size())
        model = ComparisonModel(cfg.input_dim, xs.device, uniforms)
        adam = v5.AdamWBank(configs, cfg, xs, ys)
        runner = TracedRunner(xs, ys, model, adam)
        names = (*model.output_names, f'adamw_{v5.adam_grid().index(configs[0]):02d}', 'legacy_adam_1e-5')
        result.update(output_names=names, adamw_configs=configs, storage_cost=storage_cost(model))
        start = time.perf_counter()
        graphs = runner.capture(args.graph_steps)
        torch.cuda.synchronize()
        result.update(capture_seconds=time.perf_counter() - start, status='running')
        endpoints = sorted(set(range(args.log_every, cfg.steps, args.log_every)) | set(v5.phase_boundaries(args.view)[1:]))
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
            if (int(runner.index) != step or int(adam.index) != step or int(adam.legacy.index) != step
                    or any(int(s.observations) != step for s in model.segments)
                    or int(model.mixture.singleton.index) != step
                    or int(model.mixture.singleton.uniforms_consumed) != 128 * step):
                raise RuntimeError('stream or randomness clocks diverged')
            predictions = runner.predictions[previous:step].cpu().numpy()
            weights = torch.cat((model.mean_weights(), adam.mean_weights().double()))
            finite_traces = (*runner.traces, *(getattr(runner, key) for key in runner.trace_fields))
            if (not np.isfinite(predictions).all() or not bool(torch.isfinite(weights).all())
                    or any(not bool(torch.isfinite(t[previous:step]).all()) for t in finite_traces)
                    or any(bool((torch.isnan(t[previous:step]) | torch.isposinf(t[previous:step])).any())
                           for t in runner.branch_log_masses)):
                raise RuntimeError(f'nonfinite forecast or trace at observation {step}')
            if bool((runner.mass_errors[previous:step].abs() > 1e-10).any()):
                raise RuntimeError('posterior mass not conserved')
            risk = next_forecast_risk(weights, args.view, step, cfg.feature_prob)
            noisy_error = np.square(predictions - ys[previous:step].cpu().numpy().astype(np.float64)[:, None])
            clean_error = np.square(predictions - clean[previous:step].cpu().numpy().astype(np.float64)[:, None])
            row = {'step': step, 'interval_start': previous, 'interval_count': step - previous,
                   'noisy_mse': noisy_error.mean(0).tolist(), 'clean_mse': clean_error.mean(0).tolist(),
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
            for name, segment in zip(model.child_names, model.segments):
                writer.add_scalar(f'{name}/discarded_mass', float(segment.discarded_mass), step)
                writer.add_scalar(f'{name}/cumulative_discarded_mass', float(segment.cumulative_discarded_mass), step)
            for family, child in (('singleton', model.mixture.singleton), ('factorized', model.mixture.factorized)):
                for i, hazard in enumerate(HAZARDS):
                    writer.add_scalar(f'{family}/hazard_{hazard}_posterior', float(child.log_hazard_weights[i].exp()), step)
            for i, family in enumerate(('singleton', 'factorized')):
                writer.add_scalar(f'structure/{family}_posterior', float(model.mixture.log_structure_weights[i].exp()), step)
            writer.add_scalar('joint_updates_per_second', step / result['replay_seconds'], step)
            writer.flush()
            result.update(save_predictions(root, runner, ys, clean, step, args.view))
            save_analysis_state(root, runner, step)
            result.update(durable_raw_observations=step, analysis_checkpoint_observations=step)
            v5.sparse.save_json(root / 'results.json', result)
            previous = step
        result['status'] = 'completed' if previous == cfg.steps else 'interrupted'
        if result['status'] == 'completed':
            primary, reference = names.index('structural_mixture'), names.index('v3_likelihood_static')
            mse = result['full_clean_mse']
            threshold = (1e-5 if args.view == 'null' else mse[reference] *
                         (.8 if args.view in ('change', 'recurrent') else 1.05))
            result['promotion'] = {'eligible_confirmation': args.phase == 'confirmation',
                                   'view_pass': mse[primary] <= threshold,
                                   'primary_clean_mse': mse[primary], 'v3_clean_mse': mse[reference],
                                   'threshold': threshold, 'all_views_required': True}
    except Exception as error:
        result.update(status='failed', failure=repr(error))
        raise
    finally:
        try:
            result['source_sha256_after'] = source_hashes()
            result['transferred_lock_sha256_after'] = hashlib.sha256(Path(args.lock).read_bytes()).hexdigest()
            for key, tensor in (('feature', xs), ('noise', noise), ('target', ys), ('clean', clean), ('resampling', uniforms)):
                if tensor is not None:
                    result[f'{key}_sha256_after'] = v5.tensor_hash(tensor)
            if runner is not None:
                step = int(runner.index)
                export_bound(runner, step)
                result.update(consumed_observations=step, consumed_feature_elements=step * xs.shape[1],
                              consumed_noise_elements=step, consumed_targets=step,
                              uniforms_consumed=int(runner.model.mixture.singleton.uniforms_consumed),
                              consumed_resampling_sha256=v5.tensor_hash(uniforms[:step]),
                              adam_consumed_observations=int(runner.adam.index),
                              legacy_consumed_observations=int(runner.adam.legacy.index),
                              child_consumed_observations=[int(s.observations) for s in runner.model.segments])
                result.update(save_predictions(root, runner, ys, clean, step, args.view))
                save_analysis_state(root, runner, step)
                result.update(durable_raw_observations=step, analysis_checkpoint_observations=step,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved())
            mutated = (result['source_sha256_after'] != sources
                       or result['transferred_lock_sha256_after'] != lock_hash or any(
                           result.get(f'{key}_sha256_after') != result.get(f'{key}_sha256')
                           for key in ('feature', 'noise', 'target', 'clean', 'resampling') if f'{key}_sha256' in result))
            if mutated:
                result.update(status='failed', integrity_failure='source/data/randomness/lock mutation during experiment')
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
