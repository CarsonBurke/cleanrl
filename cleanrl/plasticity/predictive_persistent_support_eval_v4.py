"""Paired full-stream deterministic-restart specialists versus frozen v3 controls.

The primary comparison is birth-only evidence refresh against v3 static
likelihood; continuous fixed-share versus v3 switching is secondary. Both v4
rows share identical persistent-support cohorts. This is deterministic
specialist aggregation, not Bayesian-global optimality. Clean support is evaluation-only.
"""
import hashlib
import inspect
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
from cleanrl.plasticity.predictive_mean_risk_conjugate_eval_v3 import (
    Runner, load_stationary_lock, save_checkpoint,
)
from cleanrl.plasticity.predictive_mean_risk_conjugate_v3 import PredictiveMeanRisk
from cleanrl.plasticity.predictive_persistent_support_v4 import PersistentSupport
from cleanrl.plasticity.predictive_structure_filter_v1 import SparsePosterior
from cleanrl.shared import runtime
from cleanrl.shared.autocull import ProxyCull, ProxyPruned, PRUNED_EXIT_CODE, prune_proxy


@dataclass
class Args:
    view: Literal['stationary', 'null', 'change'] = 'stationary'
    seed: int = 1
    graph_steps: int = 64
    log_every: int = 4096
    adam_lock: str | None = None
    autocull: bool = True
    output: str = 'runs'


class ComparisonModel:
    """Independent learners on identical labels; selected controls retain real state."""

    control_indices = (0, 1, 10, 11)
    control_names = ('v3_likelihood_static', 'v3_likelihood_switching',
                     'v3_sparse_static', 'v3_sparse_reset')

    def __init__(self, input_dim, device, periods=(8192, 16384, 32768, 65536),
                 share_rate=1e-4, noise_rate=.001):
        self.v4 = PersistentSupport(input_dim, device, periods=periods,
                                    share_rate=share_rate, noise_rate=noise_rate)
        # Match the chosen prior and null mass without any restart bank.
        self.single = PersistentSupport(input_dim, device, periods=(),
                                        share_rate=share_rate, noise_rate=noise_rate)
        self.v3 = PredictiveMeanRisk(input_dim, device)
        self.candidate_count = len(self.v4.output_names)
        self.output_names = (*(f'v4_{name}' for name in self.v4.output_names),
                             'single_likelihood', 'single_cohort', *self.control_names)
        self.configs = {'v4': self.v4.configs, 'single': self.single.configs, 'v3': self.v3.configs,
                        'v3_control_indices': self.control_indices}
        self._control_indices = torch.tensor(self.control_indices, device=device, dtype=torch.int64)
        self._single_indices = torch.tensor((0, 4), device=device, dtype=torch.int64)

    def state_tensors(self):
        return [*self.v4.state_tensors(), *self.single.state_tensors(), *self.v3.state_tensors()]

    @torch.no_grad()
    def mean_weights(self):
        return torch.cat((self.v4.mean_weights(),
                          self.single.mean_weights().index_select(0, self._single_indices),
                          self.v3.mean_weights().index_select(0, self._control_indices)))

    @torch.no_grad()
    def aggregation_weights(self):
        # Different expert sets: never pad these into a fictitious common simplex.
        return {'v4': self.v4.aggregation_weights(), 'single': self.single.aggregation_weights(),
                'v3': self.v3.aggregation_weights()}

    @property
    def noise(self):
        return torch.cat((self.v4.noise, self.single.noise, self.v3.noise))

    @property
    def energy(self):
        return self.v4.energy

    @torch.no_grad()
    def diagnostics(self):
        return {'v4': self.v4.diagnostics(),
                'single': self.single.diagnostics(),
                'v3': {'aggregation_weights': self.v3.aggregation_weights().cpu().tolist(),
                       'control_coefficients_0_1': self.v3.mean_weights().index_select(
                           0, self._control_indices)[:, :2].cpu().tolist(),
                       'inclusion_0_1': self.v3.filter.log_odds[:, :2].sigmoid().cpu().tolist(),
                       'slab_mean_0_1': self.v3.filter.slab_mean[:, :2].cpu().tolist()}}

    @torch.no_grad()
    def update(self, x, y):
        v4 = self.v4.update(x, y)
        single = self.single.update(x, y).index_select(0, self._single_indices)
        v3 = self.v3.update(x, y).index_select(0, self._control_indices)
        return torch.cat((v4, single, v3))


def summarize_comparison(curves, names, view, maximum_observations, selected_adam=None):
    """Observation-weighted clean errors, with no old/new phase mixing or censoring fiction.

    Intervals must be contiguous and split at the prefix and switch. Relapse is
    explicitly checkpoint-sampled exact-risk deterioration, not an unobserved
    per-observation recovery claim. The risk at the switch describes the NEXT
    forecast; the interval ending there contains only old-phase predictions.
    """
    prefix, switch = maximum_observations // 4, maximum_observations // 2
    previous = 0
    for row in curves:
        start, end = row['interval_start'], row['step']
        if start != previous or not start < end <= maximum_observations:
            raise ValueError('curve intervals must be contiguous, positive, and inside the stream')
        if any(start < boundary < end for boundary in (prefix, switch)):
            raise ValueError('curve intervals must split exactly at prefix and switch')
        if len(row['prequential_clean_mse']) != len(names):
            raise ValueError('clean metrics must cover every output')
        previous = end

    phases = {'full': (0, maximum_observations), 'prefix': (0, prefix),
              'suffix': (prefix, maximum_observations), 'suffix_before_change': (prefix, switch),
              'suffix_after_change': (switch, maximum_observations)}
    clean_metrics, phase_metrics = {}, {}
    for phase, (start, end) in phases.items():
        rows = [row for row in curves if start <= row['interval_start'] and row['step'] <= end]
        count = sum(row['step'] - row['interval_start'] for row in rows)
        squared_sum = np.zeros(len(names), dtype=np.float64)
        for row in rows:
            squared_sum += np.asarray(row['prequential_clean_mse'], dtype=np.float64) * (
                row['step'] - row['interval_start'])
        clean_metrics[phase] = {'count': count, 'expected_count': end - start,
                                'complete': count == end - start,
                                'clean_squared_error_sum': squared_sum.tolist(),
                                'clean_mse': (squared_sum / count).tolist() if count else None}
        if rows:
            sums = [{key: sum(row['prequential'][i][key] for row in rows)
                     for key in ('count', 'target_squared_sum', 'prediction_squared_sum',
                                 'target_prediction_sum', 'error_squared_sum')}
                    for i in range(len(names))]
            for metric in sums:
                energy = metric['target_squared_sum']
                metric['error_ratio'] = metric['error_squared_sum'] / energy if energy > 0 else None
                metric['prediction_energy_ratio'] = metric['prediction_squared_sum'] / energy if energy > 0 else None
            phase_metrics[phase] = sums
        else:
            phase_metrics[phase] = None

    primary = names.index('v4_likelihood_birth_only')
    controls = ['single_likelihood', 'v3_likelihood_static', 'v3_likelihood_switching']
    if selected_adam is not None:
        if selected_adam not in names:
            raise ValueError('selected Adam output is not present')
        controls.append(selected_adam)
    comparisons = {}
    for control in controls:
        index = names.index(control)
        comparisons[control] = {
            phase: None if metric['clean_mse'] is None else
            metric['clean_mse'][primary] - metric['clean_mse'][index]
            for phase, metric in clean_metrics.items()}
    switching = names.index('v4_likelihood_switching')
    switching_control = names.index('v3_likelihood_switching')
    secondary = {
        phase: None if metric['clean_mse'] is None else
        metric['clean_mse'][switching] - metric['clean_mse'][switching_control]
        for phase, metric in clean_metrics.items()}

    relapse = None
    if view == 'change':
        changed = [row for row in curves if row['step'] >= switch]
        if changed:
            risks = np.asarray([row['exact_risk']['clean_mse'] for row in changed], dtype=np.float64)
            best = np.minimum.accumulate(risks, axis=0)
            relapse = {'sampling': 'checkpoint next-forecast exact clean risk; switch checkpoint is new phase',
                       'checkpoint_steps': [row['step'] for row in changed],
                       'best_clean_mse': best[-1].tolist(),
                       'final_minus_best_clean_mse': (risks[-1] - best[-1]).tolist(),
                       'maximum_rise_from_running_best': (risks - best).max(axis=0).tolist()}
    return {'processed_observations': previous, 'full_horizon': previous == maximum_observations,
            'clean_metrics': clean_metrics, 'phase_metrics': phase_metrics,
            'primary_output': 'v4_likelihood_birth_only',
            'primary_minus_control_clean_mse': comparisons,
            'secondary_switching_minus_v3_switching_clean_mse': secondary,
            'selected_adam_output': selected_adam,
            'final_risk': curves[-1]['exact_risk'] if curves else None,
            'postchange_relapse': relapse}


@torch.no_grad()
def main():
    a = tyro.cli(Args)
    cfg = sparse.Args(seed=1, graph_steps=a.graph_steps)
    grid, selection, lock_hash = stock.Args.adam_lrs, None, None
    root = Path(a.output) / f'SparseStream__persistent_support_v4_{a.view}__1__{time.time_ns()}'
    root.mkdir(parents=True, exist_ok=True)
    result = {'args': asdict(a), 'stream': asdict(cfg), 'status': 'preparing', 'curves': [],
              'processed_observations': 0, 'maximum_observations': cfg.steps,
              'adam_prefix_lock': None, 'transferred_lock_sha256': None,
              'protocol': {
                  'version': 'persistent_support_v4',
                  'scope': 'full-size sparse linear conditional-mean diagnostic, not neural/market proof',
                  'interpretation': 'deterministic restarting specialists with birth-only or continuous fixed-share aggregation; no Bayesian-global guarantee',
                  'periods': [8192, 16384, 32768, 65536],
                  'cohorts': 'one never-reset cohort plus offsets 0 and P//2 for each period P',
                  'support_prior': {'slab_variance': 1., 'inclusion': 1 / cfg.input_dim, 'coordinate_hazard': 0.},
                  'aggregation_prior': 'null .5; remaining .5 uniformly split among cohorts',
                  'share_rate': 1e-4, 'noise_rate': .001,
                  'restart_due': '(P>0) & (t>0) & (t>=offset) & ((t-offset)%safe_P==0), before prediction t',
                  'retirement': 'discard due cohort log mass to -inf; renormalize survivors; then fixed-share prior injection',
                  'birth_only_aggregation': 'apply fixed-share only when any scheduled cohort is born/reset; cumulative likelihood between births',
                  'switching_aggregation': 'apply fixed-share every observation, including between births',
                  'shared_cohorts': 'birth-only and continuous rows score identical independent cohort trajectories; only aggregation forgetting differs',
                  'restart_state': 'original support odds, slab mean0/variance1, residual sum0/count0/noise1; null never resets',
                  'residual_scale': 'retained IG(2,1): count=(1-r)*count+1; sum=(1-r)*sum+prelabel_residual^2; noise=(2+sum)/(2+count)',
                  'likelihood': 'Gaussian integrated predictive moments with previous residual scale',
                  'primary_output': 'v4_likelihood_birth_only',
                  'primary_controls': ['single_likelihood', 'v3_likelihood_static'],
                  'matched_prior_control': 'one never-reset cohort, same slab/inclusion/null prior and birth-only aggregation; isolates cohort restarts from prior-grid changes',
                  'secondary_output': 'v4_likelihood_switching',
                  'secondary_control': 'v3_likelihood_switching',
                  'paired_controls': 'frozen v3 independent learning, identical features/noisy labels, indices 0,1,10,11',
                  'primary_metrics': 'full prequential clean MSE weighted by observation count; postchange clean error and checkpoint relapse; endpoint risk alone insufficient',
                  'selection': 'Adam stationary first-quarter noisy prequential MSE; transfer unchanged',
                  'oracle': 'clean labels/support score predictions and exact risk only; never enter learning',
                  'phase_causality': 'interval ending at30000 is old phase; next-forecast exact risk at30000 is changed phase; skip culling at switch',
                  'pruning': 'default ProxyCull, v4 outputs only; v3 and Adam cannot keep a dead candidate set alive',
                  'censoring': 'pruned/failed observations are partial evidence, never completed full-horizon comparisons',
                  'queue_contract': 'max-attempts1; prune exits75 and after-success children must not run'},
              'code_sha256': {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in (
                  ('evaluator', __file__), ('v4_model', inspect.getfile(PersistentSupport)),
                  ('v3_model', inspect.getfile(PredictiveMeanRisk)), ('filter_kernel', inspect.getfile(SparsePosterior)),
                  ('runner_and_lock', inspect.getfile(Runner)), ('stream_and_adam', inspect.getfile(sparse.LinearLearner)),
                  ('metrics_and_selection', inspect.getfile(stock.select_prefix)),
                  ('policy', inspect.getfile(ProxyCull)), ('runtime', inspect.getfile(runtime)))}}
    sparse.save_json(root / 'results.json', result)
    writer = None
    try:
        if a.seed != 1 or not 1 <= a.graph_steps <= cfg.steps or a.log_every <= 0:
            raise ValueError('seed1, valid graph block, and positive logging cadence required')
        if (a.view == 'stationary') != (a.adam_lock is None):
            raise ValueError('stationary selects its own prefix; null/change require --adam-lock stationary/results.json')
        if a.adam_lock is not None:
            selection, grid, lock_hash = load_stationary_lock(a.adam_lock, cfg)
            payload = Path(a.adam_lock).read_bytes()
            if hashlib.sha256(payload).hexdigest() != lock_hash:
                raise ValueError('Adam lock artifact changed while loading')
            result['transferred_feature_sha256'] = json.loads(payload)['feature_sha256']
        result.update(adam_prefix_lock=selection, adam_grid=grid, transferred_lock_sha256=lock_hash)
        sparse.save_json(root / 'results.json', result)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU model fallback')
        runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
        writer = SummaryWriter(str(root))
        xs, noise = sparse.draw_stream(cfg, torch.device('cuda'))
        result['feature_sha256'] = hashlib.sha256(memoryview(xs.cpu().numpy())).hexdigest()
        if a.adam_lock is not None and result['feature_sha256'] != result['transferred_feature_sha256']:
            raise ValueError('Generated feature SHA256 does not match the stationary Adam lock')
        ys, clean = sparse.teacher_labels(xs, noise, a.view)
        result['target_sha256'] = hashlib.sha256(memoryview(ys.cpu().numpy())).hexdigest()
        model = ComparisonModel(cfg.input_dim, 'cuda')
        adam = sparse.LinearLearner('adam', grid, cfg, xs, ys)
        runner = Runner(xs, ys, model, adam)
        names = (*model.output_names, *(f'adam_{lr:g}' for lr in grid))
        result.update(output_names=names, expert_configs=model.configs,
                      culling_output_names=model.output_names[:model.candidate_count])
        policy = ProxyCull(model.candidate_count, {'error_ratio': 1e-4, 'clean_mse': 1e-4})
        result['policy'] = policy.state_dict()
        start = time.perf_counter()
        graphs = runner.capture(a.graph_steps)
        torch.cuda.synchronize()
        result['capture_seconds'] = time.perf_counter() - start
        result['status'] = 'running'
        sparse.save_json(root / 'results.json', result)
        writer.add_text('protocol', json.dumps(result['protocol'], indent=2))
        prefix, switch = cfg.steps // 4, cfg.steps // 2
        endpoints = sorted(set(range(a.log_every, cfg.steps, a.log_every)) | {prefix, switch, cfg.steps})
        previous, history = 0, []
        start = time.perf_counter()
        for step in endpoints:
            blocks, tail = divmod(step - previous, a.graph_steps)
            for _ in range(blocks):
                graphs[a.graph_steps].replay()
            for _ in range(tail):
                graphs[1].replay()
            torch.cuda.synchronize()
            if (int(runner.index) != step or int(adam.index) != step or int(adam.steps) != step
                    or int(model.v4.observations) != step):
                raise RuntimeError('prequential stream and optimizer clocks diverged')
            pred = runner.predictions[previous:step].cpu().numpy()
            target = ys[previous:step].cpu().numpy()
            metrics = stock.metric_sums(target, pred)
            weights = torch.cat((model.mean_weights(), adam.weights[0][:, 0, :]))
            support = torch.zeros(cfg.input_dim, device=xs.device)
            moved = a.view == 'change' and step >= switch
            if a.view != 'null':
                support[1 if moved else 0] = 1
            risk = {key: value.cpu().tolist() for key, value in sparse.exact_risk(
                weights, support, cfg.feature_prob, stale_index=0 if moved else None).items()}
            row = {'step': step, 'interval_start': previous, 'interval_count': step - previous,
                   'interval_phase': 'changed' if a.view == 'change' and previous >= switch else a.view,
                   'next_forecast_phase': 'changed' if moved else a.view,
                   'prequential': metrics, 'exact_risk': risk,
                   'prequential_clean_mse': np.square(pred.astype(np.float64) - clean[previous:step].cpu().numpy()[:, None]).mean(0).tolist(),
                   'coefficient_0': weights[:, 0].cpu().tolist(), 'coefficient_1': weights[:, 1].cpu().tolist(),
                   'aggregation_weights': {key: value.cpu().tolist() for key, value in model.aggregation_weights().items()},
                   'diagnostics': model.diagnostics(), 'residual_variance_estimates': model.noise.cpu().tolist(),
                   'common_target_energy': float(model.energy)}
            if step == prefix and selection is None:
                selection = stock.select_prefix(runner.predictions[:prefix, len(model.output_names):].cpu().numpy(),
                                                ys[:prefix].cpu().numpy(), grid, 0, prefix, step)
                result.update(adam_prefix_lock=selection, processed_observations=step)
                # Persist before any suffix replay; no later selection or retuning.
                sparse.save_json(root / 'results.json', result)
            decision = None
            if a.autocull and not (a.view == 'change' and step == switch):
                count = model.candidate_count
                decision = policy.observe(step, {'error_ratio': [m['error_ratio'] for m in metrics[:count]],
                                                 'clean_mse': risk['clean_mse'][:count]},
                                          phase='changed' if moved else a.view,
                                          phase_start=switch if moved else 0)
            row['autocull'] = policy.state_dict()
            result['curves'].append(row)
            history.append(weights.cpu())
            selected_adam = f"adam_{selection['selected_lr']:g}" if selection is not None else None
            result.update(processed_observations=step, replay_seconds=time.perf_counter() - start,
                          comparison=summarize_comparison(result['curves'], names, a.view, cfg.steps, selected_adam))
            for i, name in enumerate(names):
                for metric in ('clean_mse', 'distractor_leakage_mse', 'prediction_energy'):
                    writer.add_scalar(f'{a.view}/{name}/{metric}', risk[metric][i], step)
                writer.add_scalar(f'{a.view}/{name}/error_ratio', metrics[i]['error_ratio'], step)
                writer.add_scalar(f'{a.view}/{name}/prequential_clean_mse', row['prequential_clean_mse'][i], step)
            writer.add_scalar('updates_per_second', step / result['replay_seconds'], step)
            writer.flush()
            save_checkpoint(root, runner, history, step, policy)
            sparse.save_json(root / 'results.json', result)
            if decision:
                result.update(status='pruned', pruning=decision)
                sparse.save_json(root / 'results.json', result)
                prune_proxy(root, a.view, decision)
            previous = step
        result.update(status='completed', peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        sparse.save_json(root / 'results.json', result)
        print(f'RESULTS {root / "results.json"}', flush=True)
    except ProxyPruned:
        raise SystemExit(PRUNED_EXIT_CODE) from None
    except Exception as error:
        result.update(status='failed', failure=repr(error))
        sparse.save_json(root / 'results.json', result)
        raise
    finally:
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    main()
