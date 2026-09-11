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
from cleanrl.plasticity.predictive_mean_risk_v2 import PredictiveMeanRisk
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


@torch.no_grad()
def main():
    a = tyro.cli(Args)
    if a.seed != 1 or not 1 <= a.graph_steps <= 60000 or a.log_every <= 0:
        raise ValueError('seed1, valid graph block, and positive logging cadence required')
    if (a.view == 'stationary') != (a.adam_lock is None):
        raise ValueError('stationary selects its own prefix; null/change require --adam-lock stationary/results.json')
    cfg = sparse.Args(seed=1, graph_steps=a.graph_steps)
    grid = stock.Args.adam_lrs
    selection = None
    lock_hash = None
    if a.adam_lock is not None:
        payload = Path(a.adam_lock).read_bytes()
        source = json.loads(payload)
        if source['args']['view'] != 'stationary' or source['processed_observations'] < cfg.steps // 4:
            raise ValueError('Adam lock must come from a consumed stationary first quarter')
        selection = source['adam_prefix_lock']
        grid = (selection['selected_lr'],)
        lock_hash = hashlib.sha256(payload).hexdigest()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU model fallback')
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    root = Path(a.output) / f'SparseStream__mean_risk_v2_{a.view}__1__{time.time_ns()}'
    root.mkdir(parents=True, exist_ok=True)
    result = {'args': asdict(a), 'stream': asdict(cfg), 'status': 'preparing', 'curves': [],
              'processed_observations': 0, 'maximum_observations': cfg.steps,
              'adam_prefix_lock': selection, 'transferred_lock_sha256': lock_hash,
              'protocol': {'scope': 'full-size sparse linear conditional-mean diagnostic, not neural/market proof',
                           'ablation': 'identical independent experts; only aggregation objective changes',
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
    writer = SummaryWriter(str(root))
    try:
        xs, noise = sparse.draw_stream(cfg, torch.device('cuda'))
        ys, clean = sparse.teacher_labels(xs, noise, a.view)
        result['feature_sha256'] = hashlib.sha256(memoryview(xs.cpu().numpy())).hexdigest()
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
                result['adam_prefix_lock'] = selection
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
        writer.close()


if __name__ == '__main__':
    main()
