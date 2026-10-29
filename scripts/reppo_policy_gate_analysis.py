"""Postprocess saved paired simulator outcomes against the actual behavior-policy KL gate.

CUDA, through mlq; no simulator calls. States were sampled under the paired old
policy, not drawn from the training minibatch. Gate strata are descriptive, not
causal subgroups: a shared-network update also changes states excluded by the gate.
"""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import reppo_latentroll_oracle as oracle


@torch.no_grad()
def gate_kl(trainer, snapshot, paired, device):
    args = SimpleNamespace(**snapshot["args"])
    rows = torch.as_tensor(paired["rows"], device=device)
    means = torch.as_tensor(snapshot["obs_means"], dtype=torch.float64, device=device)
    scale = (torch.as_tensor(snapshot["obs_variances"], dtype=torch.float64, device=device)
             + snapshot["obs_epsilon"]).sqrt()
    obs_dim = means.shape[-1]
    raw = torch.as_tensor(paired["states"][:, 1:1 + obs_dim], device=device)
    obs = ((raw - means[rows]) / scale[rows]).clamp(-args.obs_clip, args.obs_clip).float()
    # Last linear output is [mean, log_std], independent of hidden width.
    action_dim = list(snapshot["actor_net"].values())[-1].numel() // 2
    moments = {}
    for name, key in (("behavior", "actor_net_before_update"),
                      ("prelast", "actor_net_before_last_actor_update"), ("after", "actor_net")):
        actor = trainer.Actor(obs_dim, action_dim, args).to(device).eval().requires_grad_(False)
        actor.net.load_state_dict(snapshot[key])
        actor.net = torch.compile(actor.net, dynamic=True)
        moments[name] = actor(obs)
    result = {}
    for name in ("prelast", "after"):
        result[name] = trainer.gaussian_kl(*moments["behavior"], *moments[name]).double().cpu().numpy()
    return result


def reweight_soft_gain(scaled_raw, soft_gain, before_temperature, final_temperature):
    """Evaluate both fixed policies at a common new temperature without resimulation."""
    if before_temperature <= 0:
        raise ValueError("A positive source temperature is needed to recover entropy returns")
    return scaled_raw + (final_temperature / before_temperature) * (soft_gain - scaled_raw)


def summarize_outcomes(paired, kl, snapshot):
    count = len(paired["rows"])
    bound = snapshot["args"]["kl_bound"]
    last = paired["comparison"] == "last_actor_update"
    before_temperature = snapshot.get(f"temperature_before_{paired['comparison']}", snapshot["temperature"])
    final_temperature = snapshot["temperature"]
    inside = kl["prelast"] < bound
    strata = {"all": np.ones(count, dtype=bool)}
    if last:
        strata.update(prelast_inside_gate=inside, prelast_outside_gate=~inside)
    report = {"states": count, "kl_bound": bound, "before_temperature": before_temperature,
              "final_temperature": final_temperature, "behavior_to_after_kl_mean": float(kl["after"].mean()),
              "after_outside_gate_fraction": float((kl["after"] >= bound).mean()), "strata": {}}
    if last:
        report.update(behavior_to_prelast_kl_mean=float(kl["prelast"].mean()),
                      prelast_outside_gate_fraction=float((~inside).mean()),
                      behavior_kl_change_mean=float((kl["after"] - kl["prelast"]).mean()),
                      outside_to_inside_fraction=float(((~inside) & (kl["after"] < bound)).mean()),
                      inside_to_outside_fraction=float((inside & (kl["after"] >= bound)).mean()))
    pred = paired["predictions"]
    for name, mask in strata.items():
        result = {"states": int(mask.sum())}
        report["strata"][name] = result
        if mask.sum() < 2:
            result["skipped"] = "Fewer than two states; bootstrap interval undefined"
            continue
        clusters = np.asarray(paired["rows"])[mask]
        def stats(values, prediction=None):
            return oracle.paired_statistics(values[mask], None if prediction is None else prediction[mask], clusters)
        result["behavior_to_after_kl_mean"] = float(kl["after"][mask].mean())
        result["after_outside_gate_fraction"] = float((kl["after"][mask] >= bound).mean())
        if last:
            result["behavior_to_prelast_kl_mean"] = float(kl["prelast"][mask].mean())
            result["behavior_kl_change_mean"] = float((kl["after"][mask] - kl["prelast"][mask]).mean())
            result["behavior_kl_decreased_fraction"] = float((kl["after"][mask] < kl["prelast"][mask]).mean())
        result["predicted_first_step_q_gain"] = stats(pred["q_delta"])
        result["predicted_first_step_entropy_bonus_gain"] = stats(pred["soft_objective_delta"] - pred["q_delta"])
        result["predicted_first_step_soft_gain"] = stats(pred["soft_objective_delta"])
        result["horizons"] = {}
        for horizon, outcomes in paired["returns"].items():
            summary = {}
            result["horizons"][str(horizon)] = summary
            for branch, label in ((1, "first_action"), (2, "full_policy")):
                raw = outcomes["raw"][:, branch] - outcomes["raw"][:, 0]
                soft = outcomes["soft"][:, branch] - outcomes["soft"][:, 0]
                scaled_raw = snapshot["args"]["reward_scale"] * raw
                summary[label] = {"raw_discounted_gain": stats(raw), "soft_gain": stats(soft,
                    pred["soft_objective_delta"] if branch == 1 else None),
                    "soft_gain_at_final_temperature": stats(reweight_soft_gain(
                        scaled_raw, soft, before_temperature, final_temperature)),
                    "scaled_raw_gain": stats(scaled_raw),
                    "discounted_entropy_bonus_gain_all_steps": stats(soft - scaled_raw)}
                if "raw_undiscounted" in outcomes:
                    summary[label]["raw_undiscounted_gain"] = stats(
                        outcomes["raw_undiscounted"][:, branch] - outcomes["raw_undiscounted"][:, 0])
    return report


def select_snapshots(cli):
    roots = sorted((oracle.REPO / "runs").glob(cli.run_pattern), key=lambda p: p.stat().st_mtime)
    if not roots:
        raise ValueError(f"No run matches {cli.run_pattern}")
    paths = sorted(roots[-1].glob("snapshot_*.pt"))
    if not paths:
        raise ValueError("No snapshots in selected run")
    chosen = []
    for step in cli.snapshot_steps:
        path = min(paths, key=lambda p: abs(int(p.stem.split("_")[-1]) - step))
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        tolerance = snapshot["args"]["num_envs"] * snapshot["args"]["num_steps"]
        if abs(snapshot["global_step"] - step) > tolerance:
            raise ValueError(f"Requested {step} has no snapshot within one rollout")
        if path not in chosen:
            chosen.append(path)
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-pattern", required=True)
    p.add_argument("--snapshot-steps", nargs="+", type=int, default=[8000000, 12000000, 16000000])
    p.add_argument("--oracle-dir", type=Path, default=oracle.REPO / "runs/reppo_latentroll_oracle")
    cli = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; submit through mlq")
    for path in select_snapshots(cli):
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        if "actor_net_before_last_actor_update" not in snapshot:
            print(f"skip {path}: no saved paired actors", flush=True)
            continue
        trainer, _ = oracle.load_trainer(snapshot)
        trainer.configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
        report = {"snapshot": str(path), "global_step": snapshot["global_step"], "comparisons": {},
                  "limitations": ["States sampled under each paired old policy, not actual training minibatch.",
                    "Outside-gate states receive KL pullback; their negative predicted gain is not evidence of a bad Q direction.",
                    "Gate strata are descriptive; shared parameters couple all states.",
                    "Prediction is infinite-horizon soft Q; simulator is finite-horizon without bootstrap.",
                    "Measured entropy component includes every future step; predicted entropy component is first-step only.",
                    "Final-temperature soft gain reevaluates both fixed policies at the same final temperature; no old-critic prediction is attached."]}
        for comparison in ("last_actor_update", "update"):
            source = cli.oracle_dir / f"{path.parent.name}_{snapshot['global_step']}_paired_{comparison}.pt"
            paired = torch.load(source, map_location="cpu", weights_only=False)
            kl = gate_kl(trainer, snapshot, paired, torch.device("cuda"))
            report["comparisons"][comparison] = summarize_outcomes(paired, kl, snapshot)
            report["comparisons"][comparison]["source"] = str(source)
        target = cli.oracle_dir / f"{path.parent.name}_{snapshot['global_step']}_policy_gates.json"
        target.write_text(json.dumps(report, indent=2))
        exp = f"policy_gates_{path.parent.name.split('__')[1]}_{snapshot['global_step']}"
        with SummaryWriter(str(oracle.REPO / "runs" / f"HalfCheetah-v4__{exp}__1__{int(time.time())}")) as writer:
            for tag, value in oracle.flatten_scalars(report):
                writer.add_scalar(tag, value, snapshot["global_step"])
            writer.add_text("source", str(target))
        print(f"saved {target}", flush=True)


if __name__ == "__main__":
    main()
