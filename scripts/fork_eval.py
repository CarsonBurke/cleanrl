#!/usr/bin/env python3
"""Score the fork arms of a REPPO noise-fork experiment (cleanrl/reppo/*noisefork*) against their control.

    python scripts/fork_eval.py --trainer cleanrl/reppo/ppo_continuous_action_reppo_noisefork_v60.py \\
        --runs 'runs/HalfCheetah-v4__noisefork_v60_4M_*' --episodes 32

Every fork run's fork_<k>.pt checkpoints are rolled out deterministically (noise scale 0) and with the arm's
own noise (scale 1) on the same episode seeds and base noise (common random numbers). Arms are paired with the
control (fork multiplier equal to the source run's --ent-target-mult) of the same fork seed; the effect of an arm
is the mean over fork seeds of that paired difference, and its standard error is the spread across fork seeds,
which is what carries the chaotic divergence of training. The training metrics of each window (fork episodes'
return, entropy at the end of the window) come from the runs' event files, to confirm every arm reached its
target and returned to control. Queue with mlq.
"""
import argparse
import glob
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cleanrl.shared import landscape as ls  # noqa: E402
from cleanrl.shared.runtime import configure_runtime  # noqa: E402


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def window_metrics(run, fork_iterations, batch):
    """Mean training return of the episodes that ended in each window (perturbation: fork iterations
    1..fork_iterations; washout: the rest), and the mean entropy over each window's last 5 iterations."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(run), size_guidance={"scalars": 0})
    events.Reload()
    tags = events.Tags()["scalars"]
    returns = [(e.step, e.value) for e in events.Scalars("charts/episodic_return")] if "charts/episodic_return" in tags else []
    entropy = [(e.step, e.value) for e in events.Scalars("policy/entropy")] if "policy/entropy" in tags else []
    if not entropy:
        return {}
    steps = np.array([s for s, _ in entropy])
    first = steps.min()  # policy metrics are logged once per iteration, at its last step
    boundary = first + (fork_iterations - 1) * batch
    out = {}
    for name, (low, high) in {"perturb": (first - batch, boundary), "washout": (boundary, steps.max())}.items():
        values = [v for s, v in returns if low < s <= high]
        out[f"{name}_train_return"] = float(np.mean(values)) if values else float("nan")
        window = [v for s, v in entropy if low < s <= high]
        out[f"{name}_final_entropy"] = float(np.mean(window[-5:])) if window else float("nan")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--runs", required=True, help="glob of fork run directories")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--max-envs", type=int, default=512)
    parser.add_argument("--env-threads", type=int, default=8)
    parser.add_argument("--out", required=True, help="JSON output path")
    cli = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    device = torch.device("cuda")
    landscape = load_script("landscape")

    rows = []
    for run in sorted(Path(p) for p in glob.glob(cli.runs)):
        checkpoints = sorted(run.glob("fork_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if not checkpoints:
            print(f"skip {run.name}: no fork checkpoints")
            continue
        for path in checkpoints:
            snapshot = torch.load(path, map_location="cpu", weights_only=False)
            args = snapshot["args"]
            adapter, env_id = landscape.reppo_adapter(snapshot, cli.trainer, device)
            base = {k: v[None].expand(2, *v.shape) for k, v in ls.base_parameters(adapter.module).items()}
            make_env = landscape.make_env_factory(env_id, cli.env_threads)
            returns = ls.episode_returns(make_env, adapter, base, cli.episodes, [0.0, 1.0], cli.seed,
                                         cli.max_envs).returns
            arm = args["fork_ent_target_mult"] if args["fork_ent_target_mult"] >= 0 else args["ent_target_mult"]
            row = {"run": run.name, "checkpoint": path.stem, "fork_index": int(path.stem.split("_")[1]),
                   "resume": args["resume"], "arm": arm, "control": args["ent_target_mult"],
                   "fork_seed": args["seed"], "deterministic": returns[0].tolist(), "stochastic": returns[1].tolist()}
            print(f"{run.name} {path.stem} arm={arm} seed={args['seed']} "
                  f"det={returns[0].mean():.0f} stoch={returns[1].mean():.0f}", flush=True)
            rows.append(row)
        last = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
        metrics = window_metrics(run, last["args"]["fork_iterations"],
                                 last["args"]["num_envs"] * last["args"]["num_steps"])
        for row in rows:
            if row["run"] == run.name:
                row.update(metrics)

    # Paired effects: arm minus the same-seed control, per source, checkpoint and noise scale.
    table = defaultdict(dict)
    for row in rows:
        table[(row["resume"], row["checkpoint"], row["fork_seed"])][row["arm"]] = row
    effects = defaultdict(lambda: defaultdict(list))
    for (resume, checkpoint, seed), arms in table.items():
        control = next((r for a, r in arms.items() if a == r["control"]), None)
        if control is None:
            continue
        for arm, row in arms.items():
            for scale in ("deterministic", "stochastic"):
                diff = np.mean(row[scale]) - np.mean(control[scale])
                effects[(resume, checkpoint, arm)][scale].append(diff)
            for key in ("perturb_train_return", "washout_train_return"):
                if key in row and key in control:
                    effects[(resume, checkpoint, arm)][key].append(row[key] - control[key])
            for key in ("perturb_final_entropy", "washout_final_entropy"):
                if key in row:
                    effects[(resume, checkpoint, arm)][key].append(row[key])
    summary = []
    print(f"\n{'source':>28} {'ckpt':>8} {'arm':>5} {'n':>2} {'det diff':>14} {'stoch diff':>14} "
          f"{'train diff P/W':>16} {'entropy P/W':>13}")
    for (resume, checkpoint, arm), values in sorted(effects.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        entry = {"resume": resume, "checkpoint": checkpoint, "arm": arm}
        for key, series in values.items():
            series = np.asarray(series, dtype=float)
            entry[key] = {"mean": float(np.nanmean(series)), "values": series.tolist(),
                          "se": float(np.nanstd(series, ddof=1) / np.sqrt(np.sum(~np.isnan(series))))
                          if np.sum(~np.isnan(series)) > 1 else float("nan")}
        summary.append(entry)
        fmt = lambda k: f"{entry[k]['mean']:+6.0f}±{entry[k]['se']:<5.0f}" if k in entry else "-"
        print(f"{Path(resume).parent.name[-28:]:>28} {checkpoint:>8} {arm:5.2f} "
              f"{len(values['deterministic']):2d} {fmt('deterministic'):>14} {fmt('stochastic'):>14} "
              f"{entry.get('perturb_train_return', {}).get('mean', float('nan')):+7.0f}/"
              f"{entry.get('washout_train_return', {}).get('mean', float('nan')):+7.0f} "
              f"{entry.get('perturb_final_entropy', {}).get('mean', float('nan')):5.2f}/"
              f"{entry.get('washout_final_entropy', {}).get('mean', float('nan')):5.2f}")
    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.out).write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
    print(f"wrote {cli.out}")


if __name__ == "__main__":
    main()
