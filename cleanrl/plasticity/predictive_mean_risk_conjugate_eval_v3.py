"""Full-size sparse proxy: isolate predictive-density versus mean-risk selection.

Same eight O(D) spike/slab experts and forecasts feed both selection objectives.
No dense covariance, feature sketch, shortened dataset, or clean-label updates.
Stationary/null/change use the existing paired 60000x4096 Bernoulli(.01) stream.
Null/change must transfer an Adam LR locked on the stationary first quarter.
Each view is a separate scientific decision, not a blindly chained sweep.
Default ProxyCull persists partial evidence and exits75; mlq max-attempts=1.
This is a controlled conditional-mean diagnostic, not neural global optimality.
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
from cleanrl.plasticity.predictive_mean_risk_conjugate_v3 import PredictiveMeanRisk
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


class Runner:
    def __init__(self, xs, ys, model, adam):
        self.xs, self.ys, self.model, self.adam = xs, ys, model, adam
        self.index = torch.zeros((), dtype=torch.int64, device=xs.device)
        self.predictions = torch.zeros((len(xs), len(model.output_names) + len(adam.scale)), device=xs.device)
        self.mutable = [self.index, self.predictions, *model.state_tensors(), *adam.mutable]

    @torch.no_grad()
    def update(self):
        ix = self.index.reshape(1)
        x = self.xs.index_select(0, ix).squeeze(0).float()
        y = self.ys.index_select(0, ix).squeeze(0)
        prediction = self.model.update(x, y)
        self.adam.update()
        self.predictions.index_copy_(0, ix, torch.cat((prediction, self.adam.prediction)).unsqueeze(0))
        self.index.add_(1)

    @torch.no_grad()
    def capture(self, count):
        # Same complete-state reset/compiled replay contract as the v1 evaluator.
        initial = [tensor.clone() for tensor in self.mutable]
        def reset():
            for tensor, saved in zip(self.mutable, initial):
                tensor.copy_(saved)
        compiled = torch.compile(self.update, fullgraph=True, mode='max-autotune-no-cudagraphs')
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            compiled()
            compiled()
        stream.synchronize()
        reset()
        compiled()
        expected = [tensor.clone() for tensor in self.mutable]
        reset()
        torch.cuda.synchronize()
        graphs = {}
        for n in sorted({1, count}):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(n):
                    compiled()
            reset()
            graphs[n] = graph
        graphs[1].replay()
        torch.cuda.synchronize()
        for actual, reference in zip(self.mutable, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        reset()
        return graphs


def save_checkpoint(root, runner, history, step, policy):
    # Sparse posterior state is cheap enough to retain in full, unlike dense P.
    temporary = root / 'checkpoint.pt.tmp'
    torch.save({'step': step, 'model_state': [t.cpu() for t in runner.model.state_tensors()],
                'adam_state': [t.cpu() for t in runner.adam.mutable],
                'weight_history': torch.stack(history), 'autocull': policy.state_dict(),
                'note': 'State for analysis; this evaluator does not implement resume.'}, temporary)
    temporary.replace(root / 'checkpoint.pt')
    np.save(root / 'predictions.npy', runner.predictions[:step].cpu().numpy())


def load_stationary_lock(path, cfg):
    """Validate a stationary prefix artifact on the host; never select on new labels."""
    payload = Path(path).read_bytes()
    try:
        source = json.loads(payload)
        if source['args']['view'] != 'stationary' or source['args']['seed'] != 1 or cfg.seed != 1:
            raise ValueError('Adam lock requires stationary seed1 provenance')
        for field in ('seed', 'steps', 'input_dim', 'feature_prob', 'noise_sigma'):
            if source['stream'][field] != getattr(cfg, field):
                raise ValueError(f'Adam lock stream mismatch: {field}')
        prefix = 15000
        if cfg.steps // 4 != prefix or source['processed_observations'] < prefix:
            raise ValueError('Adam lock requires a consumed 15000-observation first quarter')
        selection = source['adam_prefix_lock']
        for field, expected in (('selection_start_inclusive', 0),
                                ('selection_end_exclusive', prefix),
                                ('optimizer_updates_at_lock', prefix),
                                ('suffix_observations_used', 0)):
            if selection[field] != expected:
                raise ValueError(f'Adam lock is not prefix-only: {field}')
        if selection['criterion'] != 'minimum prequential squared error / zero-predictor squared error':
            raise ValueError('Adam lock has an incompatible selection criterion')
        if selection['tie_break'] != 'first (smallest) learning rate':
            raise ValueError('Adam lock has an incompatible tie break')
        grid = source['adam_grid']
        if not grid or any(not math.isfinite(lr) or lr <= 0 for lr in grid) or grid != sorted(set(grid)):
            raise ValueError('Adam lock grid must be finite, positive and strictly increasing')
        candidates = selection['candidates']
        if len(candidates) != len(grid):
            raise ValueError('Adam lock candidate count does not match its grid')
        index, lr = selection['selected_index'], selection['selected_lr']
        if type(index) is not int or not 0 <= index < len(grid):
            raise ValueError('Adam lock selected index is outside its grid')
        if not math.isfinite(lr) or lr <= 0 or lr != grid[index]:
            raise ValueError('Adam lock selected LR does not match its grid')
        scores = []
        target_sum = candidates[0]['target_squared_sum']
        if not math.isfinite(target_sum) or target_sum <= 0:
            raise ValueError('Adam lock requires a finite positive prefix target energy')
        for candidate, candidate_lr in zip(candidates, grid):
            if candidate['lr'] != candidate_lr or candidate['count'] != prefix:
                raise ValueError('Adam lock candidate LR/count does not match the prefix grid')
            if candidate['target_squared_sum'] != target_sum:
                raise ValueError('Adam lock candidates were not scored on the same targets')
            score, squared_error = candidate['error_ratio'], candidate['error_squared_sum']
            # Nonfinite losing candidates are serialized as null by save_json.
            if score is None or squared_error is None:
                if score is not None or squared_error is not None:
                    raise ValueError('Adam lock has incomplete candidate score evidence')
                scores.append(math.inf)
                continue
            if (not math.isfinite(score) or score < 0 or not math.isfinite(squared_error)
                    or squared_error < 0 or not math.isclose(score, squared_error / target_sum,
                                                           rel_tol=1e-12, abs_tol=0.0)):
                raise ValueError('Adam lock candidate score disagrees with its criterion')
            scores.append(score)
        if not math.isfinite(scores[index]) or index != min(range(len(scores)), key=scores.__getitem__):
            raise ValueError('Adam lock winner disagrees with the recorded prefix criterion')
        feature_hash = source['feature_sha256']
        if (not isinstance(feature_hash, str) or len(feature_hash) != 64
                or any(c not in '0123456789abcdef' for c in feature_hash)):
            raise ValueError('Adam lock is missing a valid feature SHA256')
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError(f'Invalid Adam lock provenance: {error}') from error
    return selection, (lr,), hashlib.sha256(payload).hexdigest()


@torch.no_grad()
def main():
    a = tyro.cli(Args)
    cfg = sparse.Args(seed=1, graph_steps=a.graph_steps)
    grid = stock.Args.adam_lrs
    selection = None
    lock_hash = None
    root = Path(a.output) / f'SparseStream__mean_risk_conjugate_v3_{a.view}__1__{time.time_ns()}'
    root.mkdir(parents=True, exist_ok=True)
    result = {'args': asdict(a), 'stream': asdict(cfg), 'status': 'preparing', 'curves': [],
              'processed_observations': 0, 'maximum_observations': cfg.steps,
              'adam_prefix_lock': selection, 'transferred_lock_sha256': lock_hash,
              'protocol': {'scope': 'full-size sparse linear conditional-mean diagnostic, not neural/market proof',
                           'version': 'mean_risk_conjugate_v3',
                           'ablation': 'within v3, identical independent experts feed both aggregation objectives',
                           'cross_version': 'conjugate scales change expert trajectories versus v2; not a matched-trajectory comparison',
                           'scale_estimator': {'family': 'retained inverse-Gamma prior with discounted observed sufficient statistics',
                                               'alpha0': 2, 'beta0': 1, 'prior_mean': 1,
                                               'initial_count': 0, 'initial_sums': 0,
                                               'count_update': '(1-noise_rate)*count + 1',
                                               'residual_sum_update': '(1-noise_rate)*residual_sum + prelabel_error**2',
                                               'target_sum_update': '(1-noise_rate)*target_sum + y**2',
                                               'noise_readout': '(2+residual_sum)/(2+count)',
                                               'energy_readout': '(2+target_sum)/(2+count)',
                                               'timing': 'previous readouts score current observation; updated readouts apply next observation',
                                               'discount': 'rate0 accumulates evidence; positive rate discounts observations only, retaining the prior',
                                               'numerical_floor': None},
                           'primary_output': 'mean_risk_switching',
                           'selection': 'Adam stationary first-quarter noisy prequential MSE; transfer unchanged',
                           'oracle': 'known support only scores exact risk; never enters optimizer',
                           'pruning': 'all model outputs; training noisy error and exact synthetic clean risk; phase-local',
                           'interpretation_gates': {'stationary_clean_mse_max': .001,
                                                    'stationary_coefficient_min': .9,
                                                    'null_prediction_energy_max': .0001,
                                                    'meaning': 'ex-ante engineering gates, not statistical significance'},
                           'censoring': 'pruned rows are partial evidence, never completed full-horizon comparisons'},
              'code_sha256': {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in (
                  ('evaluator', __file__), ('model', inspect.getfile(PredictiveMeanRisk)),
                  ('filter', inspect.getfile(SparsePosterior)), ('stream', inspect.getfile(sparse.draw_stream)))}}
    sparse.save_json(root / 'results.json', result)
    writer = None
    try:
        if a.seed != 1 or not 1 <= a.graph_steps <= 60000 or a.log_every <= 0:
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
        model = PredictiveMeanRisk(cfg.input_dim, 'cuda')
        adam = sparse.LinearLearner('adam', grid, cfg, xs, ys)
        runner = Runner(xs, ys, model, adam)
        names = (*model.output_names, *(f'adam_{lr:g}' for lr in grid))
        result.update(output_names=names, expert_configs=model.configs, adam_grid=grid)
        policy = ProxyCull(len(model.output_names), {'error_ratio': 1e-4, 'clean_mse': 1e-4})
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
            if int(runner.index) != step or int(adam.index) != step or int(adam.steps) != step:
                raise RuntimeError('prequential stream and optimizer clocks diverged')
            pred = runner.predictions[previous:step].cpu().numpy()
            target = ys[previous:step].cpu().numpy()
            metrics = stock.metric_sums(target, pred)
            weights = torch.cat((model.mean_weights(), adam.weights[0][:, 0, :]))
            support = torch.zeros(cfg.input_dim, device=xs.device)
            moved = a.view == 'change' and step >= switch
            if a.view != 'null':
                support[1 if moved else 0] = 1
            risk = {k: v.cpu().tolist() for k, v in sparse.exact_risk(
                weights, support, cfg.feature_prob, stale_index=0 if moved else None).items()}
            row = {'step': step, 'interval_start': previous, 'prequential': metrics, 'exact_risk': risk,
                   'prequential_clean_mse': np.square(pred.astype(np.float64) - clean[previous:step].cpu().numpy()[:, None]).mean(0).tolist(),
                   'coefficient_0': weights[:, 0].cpu().tolist(), 'coefficient_1': weights[:, 1].cpu().tolist(),
                   'aggregation_weights': model.aggregation_weights().cpu().tolist(),
                   'residual_variance_estimates': model.noise.cpu().tolist(), 'common_target_energy': float(model.energy)}
            if step == prefix and selection is None:
                model_count = len(model.output_names)
                locked = stock.select_prefix(runner.predictions[:prefix, model_count:].cpu().numpy(),
                                             ys[:prefix].cpu().numpy(), grid, 0, prefix, step)
                selection = {**locked, 'selected_lr': grid[locked['selected_index']]}
                result.update(adam_prefix_lock=selection, processed_observations=step)
                sparse.save_json(root / 'results.json', result)
            decision = None
            # Frozen risk switches immediately; the interval at switch is OLD.
            if a.autocull and not (a.view == 'change' and step == switch):
                count = len(model.output_names)
                decision = policy.observe(step, {'error_ratio': [m['error_ratio'] for m in metrics[:count]],
                                                 'clean_mse': risk['clean_mse'][:count]},
                                          phase='changed' if moved else a.view,
                                          phase_start=switch if moved else 0)
            row['autocull'] = policy.state_dict()
            result['curves'].append(row)
            history.append(weights.cpu())
            result.update(processed_observations=step, replay_seconds=time.perf_counter() - start)
            for i, name in enumerate(names):
                for metric in ('clean_mse', 'distractor_leakage_mse', 'prediction_energy'):
                    writer.add_scalar(f'{a.view}/{name}/{metric}', risk[metric][i], step)
                writer.add_scalar(f'{a.view}/{name}/error_ratio', metrics[i]['error_ratio'], step)
            writer.add_scalar('updates_per_second', step / result['replay_seconds'], step)
            writer.flush()
            sparse.save_json(root / 'results.json', result)
            if decision:
                save_checkpoint(root, runner, history, step, policy)
                result.update(status='pruned', pruning=decision)
                sparse.save_json(root / 'results.json', result)
                prune_proxy(root, a.view, decision)
            previous = step
        save_checkpoint(root, runner, history, cfg.steps, policy)
        prediction = runner.predictions.cpu().numpy()
        target = ys.cpu().numpy()
        result['phase_metrics'] = {name: stock.metric_sums(target[s:e], prediction[s:e])
                                   for name, (s, e) in {'prefix': (0, prefix), 'suffix': (prefix, cfg.steps),
                                                       'suffix_before_change': (prefix, switch),
                                                       'suffix_after_change': (switch, cfg.steps)}.items()}
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
