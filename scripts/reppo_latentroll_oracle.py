"""Simulator diagnostics for latent-roll REPPO snapshots (run through mlq, CUDA only).

Reuses the validated reppo_oracle physics, CRN sampler, finite differences and reports.
Adds three paired interventions: old policy, new first action then old policy, and
new policy throughout. Raw and soft discounted finite-horizon returns have NO critic
bootstrap. MuJoCo continues across the artificial time limit; this matches the
continuing objective bootstrapped by training, while avoiding a circular critic tail.
Report multiple horizons and gamma**H rather than calling finite-horizon labels exact Q.

The single last actor update is preferred when captured: its pre-update critic was
actually used to calculate that update. A whole-block pair is also reported, explicitly
labelled because its initial critic did not generate all updates in the block.
"""
import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reppo_oracle as legacy

REPO = Path(__file__).resolve().parents[1]


def load_trainer(snapshot, explicit=None):
    path = explicit or snapshot.get("trainer_file")
    if path is None:
        # Only unambiguous state-dict architectures are inferred. v25 has predictor;
        # v27 has dynamics/reward_head. Instrumented snapshots name their source.
        keys = snapshot["critic"]
        if any(k.startswith("dynamics.") for k in keys):
            path = REPO / "cleanrl/reppo/ppo_continuous_action_reppo_latentroll_v27.py"
        elif any(k.startswith("predictor.") for k in keys):
            path = REPO / "cleanrl/reppo/ppo_continuous_action_reppo_refresh_v25.py"
        else:
            raise ValueError("Cannot infer trainer architecture; supply --trainer")
    path = Path(path)
    if not path.is_absolute():
        path = REPO / path
    spec = importlib.util.spec_from_file_location("latentroll_oracle_trainer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, str(path)


# The old oracle's collection is valid physics, but its v23 target formula is not
# the expected-soft v25/v27 formula. Replace only target reconstruction.
_collect_legacy = legacy.collect


@torch.no_grad()
def collect(oracle, trainer, num_envs, num_steps, seed):
    data = _collect_legacy(oracle, trainer, num_envs, num_steps, seed)
    device = oracle.device
    rows = torch.arange(num_envs, device=device).repeat(num_steps)
    raw = data["raw_next_obs"].to(device)
    obs = ((raw - oracle.obs_mean[rows]) / oracle.obs_scale[rows]).clamp(-oracle.obs_clip, oracle.obs_clip).float()
    ids = torch.arange(len(rows), device=device)
    k = int(oracle.args.bootstrap_samples)
    keys = legacy.base.stream_key(oracle.seed_hash, legacy.TARGET, ids[:, None], torch.arange(k, device=device))
    action, logp = oracle.sample(obs[:, None].expand(-1, k, -1), keys)
    q = oracle.critic.value(obs[:, None].expand(-1, k, -1), action).mean(1)
    bonus = -oracle.temperature * logp.mean(1)
    shape = (num_steps, num_envs)
    cuts = torch.maximum(data["truncations"], data["ends"]).to(device).view(shape)
    reward = data["rewards"].to(device).view(shape) + oracle.gamma * oracle.lam * (1 - cuts) * bonus.view(shape)
    target = trainer.lambda_returns(legacy.get_gae_fn(compiled=False, explicit_next_values=True), reward,
                                   (q + bonus).view(shape), torch.zeros_like(cuts), cuts, oracle.gamma, oracle.lam)
    data.update(targets=target.flatten().cpu(), next_q=q.cpu(), next_logprob=logp.mean(1).cpu(),
                target_formula="expected-soft-lambda-v25-v27", bootstrap_samples=k)
    return data


class CompiledOracle(legacy.Oracle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor.net = torch.compile(self.actor.net, dynamic=True)
        self.critic.value = torch.compile(self.critic.value, dynamic=True)


def paired_statistics(delta, prediction=None, cluster_ids=None):
    """State-cluster bootstrap; rollouts and interventions are paired within states.

    Intervals describe the sampled state distribution with rollout noise retained.
    States come from one trajectory chunk, so residual serial dependence can make
    these intervals optimistic; report that limitation, do not call them seed CIs.
    """
    per_state = np.asarray(delta, dtype=np.float64).mean(-1)
    rng = np.random.default_rng(1)
    draws = rng.integers(len(per_state), size=(2000, len(per_state)))
    means = per_state[draws].mean(-1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    out = {"mean": float(per_state.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
           "state_sd": float(per_state.std(ddof=1)), "states": len(per_state),
           "rollouts": delta.shape[-1],
           "within_state_mc_se_rms": float(np.sqrt(np.mean(np.var(delta, axis=-1, ddof=1) / delta.shape[-1])))}
    if cluster_ids is not None:
        groups = np.unique(cluster_ids)
        sums = np.array([per_state[cluster_ids == group].sum() for group in groups])
        counts = np.array([(cluster_ids == group).sum() for group in groups])
        group_draws = rng.integers(len(groups), size=(2000, len(groups)))
        cluster_means = sums[group_draws].sum(1) / counts[group_draws].sum(1)
        clo, chi = np.quantile(cluster_means, [0.025, 0.975])
        out.update(trajectory_clusters=len(groups), trajectory_ci95_low=float(clo), trajectory_ci95_high=float(chi))
    if prediction is not None:
        pred = np.asarray(prediction).mean(-1)
        out["predicted_mean"] = float(pred.mean())
        error = per_state - pred
        out["prediction_error_mean"] = float(error.mean())
        elo, ehi = np.quantile(error[draws].mean(-1), [0.025, 0.975])
        out.update(prediction_error_ci95_low=float(elo), prediction_error_ci95_high=float(ehi))
        if cluster_ids is not None:
            error_sums = np.array([error[cluster_ids == group].sum() for group in groups])
            cluster_errors = error_sums[group_draws].sum(1) / counts[group_draws].sum(1)
            eclo, echi = np.quantile(cluster_errors, [0.025, 0.975])
            out.update(prediction_error_trajectory_ci95_low=float(eclo),
                       prediction_error_trajectory_ci95_high=float(echi))
        out["prediction_correlation"] = (float(np.corrcoef(per_state, pred)[0, 1])
                                          if pred.std() > 0 and per_state.std() > 0 else None)
    return out


def make_pair(snapshot, key):
    before = dict(snapshot)
    before["actor_net"] = snapshot[f"actor_net_before_{key}"]
    before["critic"] = snapshot.get(f"critic_before_{key}", snapshot["critic"])
    before["temperature"] = snapshot.get(f"temperature_before_{key}", snapshot["temperature"])
    after = dict(snapshot)
    # Hold objective temperature fixed; retain both critics for separate predictions.
    after["critic"] = snapshot["critic"]
    after["temperature"] = before["temperature"]
    return before, after


@torch.no_grad()
def paired_returns(before, after, cli, data, index):
    """(state, branch, rollout) CRN samples; branch 0 old, 1 first action, 2 full new."""
    n, m, device = len(index), cli.rollouts, before.device
    shape = (n, 3, m)
    state = np.ascontiguousarray(np.repeat(data["state"][index], 3 * m, axis=0))
    rows = torch.as_tensor(data["row"][index], device=device).repeat_interleave(3 * m)
    sid = torch.as_tensor(index, device=device).repeat_interleave(3 * m)
    rollout = torch.arange(m, device=device).repeat(n * 3)
    branch = torch.arange(3, device=device).repeat_interleave(m).repeat(n)
    raw_total = torch.zeros(n * 3 * m, device=device, dtype=torch.float64)
    soft_total = torch.zeros_like(raw_total)
    benchmark_total = torch.zeros_like(raw_total)
    checkpoints = sorted(set(cli.policy_checkpoints) | {cli.policy_horizon})
    result, predictions = {}, {}
    for t in range(cli.policy_horizon):
        current = torch.from_numpy(state).to(device)
        obs = before.observe(current, rows)
        key = legacy.base.stream_key(before.seed_hash, 91, sid, rollout, t)
        old_action, old_logp = before.sample(obs, key)
        new_action, new_logp = after.sample(obs, key)
        use_new = branch > 0 if t == 0 else branch == 2
        action = torch.where(use_new[:, None], new_action, old_action)
        logp = torch.where(use_new, new_logp, old_logp)
        if t == 0:
            # First-action policy objective difference includes entropy at t=0;
            # old future actions cancel in expectation, unlike raw Q's soft tail.
            q_old = before.critic.value(obs, old_action)
            q_new = before.critic.value(obs, new_action)
            post_q_delta = after.critic.value(obs, new_action) - after.critic.value(obs, old_action)
            old_mean, old_std = before.moments(obs)
            new_mean, new_std = after.moments(obs)
            kl = (new_std.log() - old_std.log() + (old_std.square() + (old_mean - new_mean).square()) / (2 * new_std.square()) - 0.5).sum(-1)
            grad = before.critic_grad(obs, old_action)
            linear = (grad * (new_action - old_action)).sum(-1)
            entropy_delta = -before.temperature * (new_logp - old_logp)
            for name, value in {"q_delta": q_new - q_old, "soft_objective_delta": q_new - q_old + entropy_delta,
                                "post_critic_soft_delta": post_q_delta + entropy_delta, "kl_old_to_new": kl,
                                "linear_q_delta": linear, "linear_soft_delta": linear + entropy_delta,
                                "action_delta_rms": (new_action - old_action).square().mean(-1).sqrt()}.items():
                predictions[name] = value.view(shape)[:, 0].double().cpu().numpy()
        before.physics.step(state, action.cpu().numpy())
        reward = before.raw_reward(torch.from_numpy(state[:, 0]).to(device), current[:, 0], action)
        benchmark_total += reward
        raw_total += before.gamma ** t * reward
        soft_total += before.gamma ** t * (before.reward_scale * reward - before.temperature * logp)
        if t + 1 in checkpoints:
            result[t + 1] = {"raw": raw_total.view(shape).cpu().numpy(),
                             "raw_undiscounted": benchmark_total.view(shape).cpu().numpy(),
                             "soft": soft_total.view(shape).cpu().numpy()}
            print(f"paired policy horizon {t + 1}/{cli.policy_horizon}", flush=True)
    return result, predictions


def flatten_scalars(value, prefix=""):
    for key, child in value.items():
        tag = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(child, dict):
            yield from flatten_scalars(child, tag)
        elif isinstance(child, (float, int)) and np.isfinite(child):
            yield tag, child


def policy_diagnostics(path, snapshot, trainer, cli, device):
    reports = {}
    for key in ("last_actor_update", "update"):
        if f"actor_net_before_{key}" not in snapshot:
            continue
        left, right = make_pair(snapshot, key)
        physics = legacy.base.Physics(cli.threads, cli.seed)
        before = legacy.Oracle(trainer, left, physics, cli, device)
        after = legacy.Oracle(trainer, right, physics, cli, device)
        data = legacy.collect(before, trainer, before.args.num_envs, before.args.num_steps, cli.seed)
        rng = np.random.default_rng(cli.seed)
        index = np.sort(rng.choice(len(data["age"]), cli.states, replace=False))
        returns, predictions = paired_returns(before, after, cli, data, index)
        results = {}
        for horizon, values in returns.items():
            result = {"gamma_power_horizon": before.gamma ** horizon}
            for branch, name in ((1, "first_action"), (2, "full_policy")):
                result[name] = {}
                for objective in ("raw", "raw_undiscounted", "soft"):
                    delta = values[objective][:, branch] - values[objective][:, 0]
                    # A soft Q cannot predict a raw-return change. Do not conflate them.
                    pred = predictions["soft_objective_delta"] if objective == "soft" and branch == 1 else None
                    result[name][objective] = paired_statistics(delta, pred, data["row"][index])
            results[horizon] = result
        stem = cli.out_dir / f"{path.parent.name}_{snapshot['global_step']}_paired_{key}"
        torch.save({"returns": returns, "predictions": predictions, "states": data["state"][index],
                    "rows": data["row"][index], "ages": data["age"][index], "indices": index,
                    "snapshot": str(path), "comparison": key}, stem.with_suffix(".pt"))
        reports[key] = {"results": results, "prediction_means": {k: float(v.mean()) for k, v in predictions.items()},
                        "critic": "before this update" if f"critic_before_{key}" in snapshot else "after update (fallback)",
                        "temperature": before.temperature}
    if not reports:
        return {"skipped": "Snapshot lacks before-update policy; no paired check can be reconstructed"}
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("snapshots", type=Path, nargs="*")
    p.add_argument("--run-pattern", help="Glob over runs/ directories; selects newest matching run")
    p.add_argument("--snapshot-steps", nargs="+", type=int, help="Select snapshots nearest these steps")
    p.add_argument("--trainer", type=Path)
    p.add_argument("--states", type=int, default=256)
    p.add_argument("--rollouts", type=int, default=16)
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--checkpoints", nargs="+", type=int, default=[100, 200, 300])
    p.add_argument("--policy-horizon", type=int, default=1000)
    p.add_argument("--policy-checkpoints", nargs="+", type=int, default=[100, 300, 1000])
    p.add_argument("--fresh", type=int, default=8)
    p.add_argument("--value-samples", type=int, default=16)
    p.add_argument("--baseline-samples", type=int, default=64)
    p.add_argument("--fd-steps", nargs="+", type=float, default=[0.2, 0.05])
    p.add_argument("--group-states", type=int, default=32)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--validate-episodes", type=int, default=32)
    p.add_argument("--gym-episodes", type=int, default=32)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--mode", choices=("all", "finite-difference", "policy"), default="all")
    p.add_argument("--out-dir", type=Path, default=REPO / "runs/reppo_latentroll_oracle")
    cli = p.parse_args()
    if cli.states < 2 or cli.rollouts < 2 or cli.fresh < 2 or any(h <= 0 for h in cli.fd_steps):
        p.error("states, rollouts, fresh require >=2; finite difference steps must be positive")
    if any(h < 1 or h > cli.horizon for h in cli.checkpoints):
        p.error("finite difference checkpoints must fit horizon")
    if any(h < 1 or h > cli.policy_horizon for h in cli.policy_checkpoints):
        p.error("policy checkpoints must fit policy horizon")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; submit this evaluation through mlq")
    if cli.run_pattern:
        matches = sorted((REPO / "runs").glob(cli.run_pattern), key=lambda p: p.stat().st_mtime)
        if not matches:
            p.error("No run directory matches --run-pattern")
        cli.snapshots.extend(sorted(matches[-1].glob("snapshot_*.pt")))
    if cli.snapshot_steps and cli.snapshots:
        chosen = []
        for step in cli.snapshot_steps:
            candidate = min(cli.snapshots, key=lambda p: abs(int(p.stem.split("_")[-1]) - step))
            metadata = torch.load(candidate, map_location="cpu", weights_only=False)
            batch = metadata["args"]["num_envs"] * metadata["args"]["num_steps"]
            if abs(int(metadata["global_step"]) - step) > batch:
                p.error(f"No snapshot within one rollout ({batch} steps) of requested {step}; nearest is {candidate}")
            chosen.append(candidate)
        cli.snapshots = list(dict.fromkeys(chosen))
    if not cli.snapshots:
        p.error("No snapshots found")
    legacy.collect = collect
    legacy.Oracle = CompiledOracle
    cli.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    for path in cli.snapshots:
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        if snapshot["args"]["env_id"] != legacy.ENV_ID:
            raise ValueError("Only HalfCheetah-v4 physics is supported")
        trainer, trainer_path = load_trainer(snapshot, cli.trainer)
        trainer.configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
        torch.manual_seed(cli.seed)
        report = {"snapshot": str(path), "trainer": trainer_path, "global_step": snapshot["global_step"],
                  "limitations": ["State CIs ignore serial dependence; trajectory CIs cluster entire independent environment rows (usually16). Neither includes training-seed variance.",
                                  "Finite difference labels have critic-bootstrap tails; paired policy returns do not.",
                                  "Raw returns are unscaled; soft returns use reward_scale and frozen temperature."]}
        start = time.perf_counter()
        if cli.mode != "policy":
            with ThreadPoolExecutor(max_workers=1) as pool:
                report["finite_difference"] = legacy.run_snapshot(path, cli, trainer, device, pool)
        if cli.mode != "finite-difference":
            if cli.mode == "policy":
                oracle = legacy.Oracle(trainer, snapshot, legacy.base.Physics(cli.threads, cli.seed), cli, device)
                report["validation"] = legacy.validate(oracle, trainer, cli)
            report["policy"] = policy_diagnostics(path, snapshot, trainer, cli, device)
        report["seconds"] = time.perf_counter() - start
        stem = cli.out_dir / f"{path.parent.name}_{snapshot['global_step']}_combined"
        stem.with_suffix(".json").write_text(json.dumps(report, indent=2, default=float))
        parent_parts = path.parent.name.split("__")
        parent_exp = parent_parts[1] if len(parent_parts) == 4 else path.parent.name.replace("__", "_")
        exp = f"latentroll_oracle_{parent_exp}_{snapshot['global_step']}_{cli.mode}"
        run_name = f"HalfCheetah-v4__{exp}__{cli.seed}__{int(time.time())}"
        with SummaryWriter(str(REPO / "runs" / run_name)) as writer:
            for tag, value in flatten_scalars(report):
                writer.add_scalar(f"oracle/{tag}", value, snapshot["global_step"])
            writer.add_text("oracle/source", json.dumps({"snapshot": str(path), "report": str(stem.with_suffix('.json'))}))
        print(f"saved {stem}.json", flush=True)


if __name__ == "__main__":
    main()
