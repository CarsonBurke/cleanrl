#!/usr/bin/env python3
"""Return landscape of a frozen policy snapshot, with the critic's surrogate on the same slices.

    # true noise-cost curve J(c): the policy's own noise scaled by c (0 = deterministic, 1 = trained)
    python scripts/landscape.py SNAP.pt --trainer cleanrl/reppo/<trainer>.py --slice spread
    # 1-D along the actual update (needs a later snapshot of the same run; 1 = the step taken)
    python scripts/landscape.py SNAP.pt --to SNAP_LATER.pt --trainer ... --slice update --coefficients=-1:4:21
    # 1-D along the critic's ascent direction, restricted to the mean head's parameters
    python scripts/landscape.py SNAP.pt --trainer ... --slice ascent --subset mean
    # 2-D plane (the second direction is orthogonalized against the first)
    python scripts/landscape.py SNAP.pt --trainer ... --plane ascent,random --coefficients=-0.1:0.1:11

Families (--family, inferred from the snapshot's keys):
    reppo  REPPO-style snapshots (actor_net, critic, obs_means; saved by --snapshot-steps): tanh-Gaussian actor
           rebuilt from --trainer's Actor/policy_moments, Q critic from its Critic. Subsets: mean, spread.
    beta   PPO/AWR snapshots from scripts/awr_capture_policy.py (trainer_module, model, obs_norm): Beta actor
           from the trainer's Agent; exact common-noise sampling by inverse CDF. No Q critic.

Directions: update (to --to), ascent (gradient of the actor's own surrogate E_s E_eps [Q - T log pi], no trust
region; held-out states and noise score the critic curve), random
(filter-normalized, Li et al. 2018). Unit (--unit): weights = a norm equal to a filter-normalized random
direction's, so c is a fraction of the weights' scale; update = the norm of the actual step (needs --to). Points
share the base snapshot's observation normalization, so c = 1 on the update slice is the later weights under
the earlier normalization; J(theta_to) under its own normalization is reported alongside as a reference.
Every point runs --episodes seeds with common random numbers (same resets and base noise at every point),
so paired differences to the center point are reported with their standard errors. Queue with mlq.
"""
import argparse
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cleanrl.shared import landscape as ls  # noqa: E402
from cleanrl.shared.mujoco_env import make_mujoco_vector_env  # noqa: E402
from cleanrl.shared.runtime import configure_runtime  # noqa: E402

NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))


def load_module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- families

class Moments(nn.Module):
    """[mean, std] of a tanh-Gaussian from the trainer's own policy_moments."""

    def __init__(self, net, moments):
        super().__init__()
        self.net = net
        self.moments = moments

    def forward(self, observations):
        return torch.cat(self.moments(self.net, observations), dim=-1)


def reppo_adapter(snapshot, trainer_path, device):
    if trainer_path is None:
        raise ValueError("--trainer is required for REPPO snapshots (the file that saved them)")
    trainer = load_module(trainer_path)
    args = SimpleNamespace(**snapshot["args"])
    means = np.asarray(snapshot["obs_means"])
    observation_dim = means.shape[-1]
    last = [value for name, value in snapshot["actor_net"].items() if name.endswith("weight")][-1]
    action_dim = last.shape[0] // 2
    actor = trainer.Actor(observation_dim, action_dim, args).to(device)
    actor.net.load_state_dict(snapshot["actor_net"])
    critic = trainer.Critic(observation_dim, action_dim, args, device).to(device)
    critic.load_state_dict(snapshot["critic"])
    critic.requires_grad_(False)
    module = Moments(actor.net, trainer.policy_moments).requires_grad_(True)
    last_name = "net." + [name for name in snapshot["actor_net"] if name.endswith("weight")][-1][: -len(".weight")]

    def act(outputs, noise, scale):
        mean, std = outputs.chunk(2, dim=-1)
        return torch.tanh(mean + scale * std * noise)

    temperature = torch.as_tensor(snapshot.get("temperature", 0.0), dtype=torch.float32,
                                  device=device).expand(action_dim)
    log_det = getattr(trainer, "tanh_log_det", lambda pre: 2.0 * (np.log(2.0) - pre - F.softplus(-2.0 * pre)))

    def entropy_bonus(outputs, noise, scale):
        """-T log pi of the sample (the trainer's actor loss is -Q + T log pi); pi's std is scale * std."""
        mean, std = outputs.chunk(2, dim=-1)
        pre = mean + scale * std * noise
        logprob = -0.5 * noise.square() - torch.log(scale * std) - 0.5 * np.log(2 * np.pi) - log_det(pre)
        return -(temperature * logprob).sum(-1)

    def spread_rows(name, value):
        mask = torch.zeros_like(value)
        if name.startswith(last_name + "."):
            mask[action_dim:] = 1.0
        return mask

    variances = np.asarray(snapshot["obs_variances"])
    epsilon = float(snapshot.get("obs_epsilon", 1e-8))
    return ls.Adapter(
        module=module,
        obs_mean=torch.as_tensor(means.reshape(-1, observation_dim)[0], dtype=torch.float32, device=device),
        obs_inv_std=torch.as_tensor(1.0 / np.sqrt(variances.reshape(-1, observation_dim)[0] + epsilon),
                                    dtype=torch.float32, device=device),
        obs_clip=float(getattr(args, "obs_clip", 10.0)), action_dim=action_dim, act=act, base_noise="normal",
        subsets={"spread": spread_rows, "mean": lambda name, value: 1.0 - spread_rows(name, value)},
        critic=critic.value, regularizer=entropy_bonus), args.env_id


class BetaHead(nn.Module):
    """[alpha, beta] of the trainer's Agent.get_policy_and_value, from the actor alone."""

    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, observations):
        return F.softplus(self.actor(observations)) + 1.0


def beta_adapter(snapshot, device):
    from scipy.special import betaincinv

    trainer = importlib.import_module(snapshot["trainer_module"])
    args = argparse.Namespace(**snapshot["args"])
    envs = trainer.make_training_env(args, "landscape")
    agent = trainer.Agent(envs).to(device)
    envs.close()
    agent.load_state_dict(snapshot["model"])
    module = BetaHead(agent.actor)
    low, span = agent.action_low, agent.action_high - agent.action_low
    action_dim = int(low.numel())

    def act(outputs, noise, scale):
        alpha, beta = outputs.chunk(2, dim=-1)
        alpha, beta = torch.broadcast_tensors(alpha, beta)
        u = noise.expand_as(alpha)
        x = torch.as_tensor(betaincinv(alpha.double().cpu().numpy(), beta.double().cpu().numpy(),
                                       u.double().cpu().numpy()), dtype=torch.float32, device=alpha.device)
        center = alpha / (alpha + beta)
        return low + span * (center + scale * (x - center)).clamp(0.0, 1.0)

    obs = snapshot["obs_norm"]
    means, variances = np.asarray(obs["means"]), np.asarray(obs["variances"])
    dim = means.shape[-1]
    adapter = ls.Adapter(
        module=module,
        obs_mean=torch.as_tensor(means.reshape(-1, dim)[0], dtype=torch.float32, device=device),
        obs_inv_std=torch.as_tensor(1.0 / np.sqrt(variances.reshape(-1, dim)[0] + 1e-8), dtype=torch.float32,
                                    device=device),
        obs_clip=float(getattr(args, "obs_clip", 10.0)), action_dim=action_dim, act=act, base_noise="uniform",
        subsets={"head": lambda name, value: torch.full_like(value, float(name.startswith("actor.head.")))})
    return adapter, args.env_id


def build(snapshot, trainer, device, family):
    family = family or ("reppo" if "actor_net" in snapshot else "beta" if "trainer_module" in snapshot else None)
    if family == "reppo":
        return reppo_adapter(snapshot, trainer, device)
    if family == "beta":
        return beta_adapter(snapshot, device)
    raise ValueError("cannot infer the snapshot's family; pass --family")


# ---------------------------------------------------------------- CLI

def parse_coefficients(text):
    if ":" in text:
        low, high, count = text.split(":")
        return np.linspace(float(low), float(high), int(count))
    return np.array([float(value) for value in text.split(",")])


def make_env_factory(env_id, threads):
    backend = "native" if env_id in NATIVE_TASKS else "sync"
    return lambda count: make_mujoco_vector_env(env_id, count, backend=backend, num_threads=min(threads, count))


def load_target(cli, device):
    """The later snapshot's adapter: its weights and its own observation normalization."""
    if cli.to is None:
        raise ValueError("the update direction and --unit update need --to")
    target, _ = build(torch.load(cli.to, map_location="cpu", weights_only=False), cli.trainer, device, cli.family)
    return target


def resolve_direction(kind, adapter, base, mask, update, states, samples, seed):
    if kind == "update":
        return ls.restrict(update(), mask)
    if kind == "ascent":
        if states is None:
            raise ValueError("the ascent direction needs states (run with a Q critic)")
        return ls.critic_ascent_direction(adapter, base, states, samples, 1.0, seed, mask)
    if kind == "random":
        return ls.random_direction(base, seed, mask)
    raise ValueError(f"unknown direction {kind}")


def collect_states(adapter, base, make_env, cli):
    """Normalized states the base policy visits (for the critic surrogate), from a short CRN run."""
    rollouts = ls.episode_returns(make_env, adapter, {k: v[None] for k, v in base.items()}, cli.state_episodes,
                                  1.0, cli.seed + 7919, cli.max_envs, record_point=0, record_every=4)
    states = rollouts.states
    generator = torch.Generator(device=states.device).manual_seed(cli.seed)
    return states[torch.randperm(states.shape[0], generator=generator, device=states.device)[: cli.states]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshot")
    parser.add_argument("--trainer", help="trainer file that saved a REPPO snapshot")
    parser.add_argument("--family", choices=("reppo", "beta"))
    parser.add_argument("--to", help="a later snapshot of the same run (the update direction)")
    parser.add_argument("--slice", help="spread, update, ascent or random (1-D)")
    parser.add_argument("--plane", help="two directions, e.g. update,ascent (2-D)")
    parser.add_argument("--subset", default="all", help="parameter subset every direction is restricted to")
    parser.add_argument("--unit", choices=("weights", "update"), default="weights")
    parser.add_argument("--coefficients", default=None, help="low:high:count or a comma list")
    parser.add_argument("--scales", default="1,0", help="noise amplitudes evaluated at every point")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-envs", type=int, default=512)
    parser.add_argument("--env-threads", type=int, default=8)
    parser.add_argument("--states", type=int, default=4096, help="states in the critic surrogate")
    parser.add_argument("--state-episodes", type=int, default=8)
    parser.add_argument("--critic-samples", type=int, default=8)
    parser.add_argument("--out", default=None, help="output directory (default runs/landscape/<snapshot>)")
    cli = parser.parse_args()
    if (cli.slice is None) == (cli.plane is None):
        parser.error("pass exactly one of --slice and --plane")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    device = torch.device("cuda")
    snapshot_path = Path(cli.snapshot)
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    adapter, env_id = build(snapshot, cli.trainer, device, cli.family)
    base = ls.base_parameters(adapter.module)
    cached = {}

    def target():
        if "target" not in cached:
            cached["target"] = load_target(cli, device)
        return cached["target"]

    def update():
        """theta_to - theta: the step the trainer actually took between the two snapshots."""
        return ls.difference(ls.base_parameters(target().module), base)
    mask = ls.subset_mask(adapter, base, cli.subset)
    make_env = make_env_factory(env_id, cli.env_threads)
    scales = [float(value) for value in cli.scales.split(",")]
    started = time.perf_counter()
    states = held_out = None
    if adapter.critic is not None:
        # One half fits the ascent direction, the other half scores the critic along it (no selection bias).
        states = collect_states(adapter, base, make_env, cli)
        states, held_out = states[: states.shape[0] // 2], states[states.shape[0] // 2:]
    name = cli.slice or cli.plane.replace(",", "-")
    out = Path(cli.out or REPO / "runs" / "landscape" / f"{snapshot_path.parent.name}__{snapshot_path.stem}")
    out.mkdir(parents=True, exist_ok=True)
    result = {"snapshot": str(snapshot_path), "to": cli.to, "env_id": env_id, "slice": name,
              "subset": cli.subset, "episodes": cli.episodes, "seed": cli.seed, "scales": scales,
              "critic_states": 0 if states is None else 2 * states.shape[0]}
    critic_seed = cli.seed + 7777

    if cli.slice == "spread":
        coefficients = parse_coefficients(cli.coefficients or "0:2:9")
        single = {k: v[None] for k, v in base.items()}
        repeated = {k: v.expand(len(coefficients), *v.shape[1:]) for k, v in single.items()}
        returns = ls.episode_returns(make_env, adapter, repeated, cli.episodes, coefficients, cli.seed,
                                     cli.max_envs).returns  # [C, episodes]: points are noise amplitudes
        center = int(np.argmin(np.abs(coefficients - 1.0)))
        summary = ls.paired_summary(returns, center)
        result.update(coefficients=coefficients.tolist(), returns=returns.tolist(),
                      summary={k: v.tolist() for k, v in summary.items()},
                      fit=ls.paired_quadratic(coefficients, summary["diff"], summary["diff_se"], center))
        if states is not None:
            critic = [float(ls.critic_objective(adapter, single, held_out, cli.critic_samples, float(c),
                                                critic_seed)) for c in coefficients]
            result.update(critic=critic, critic_fit=ls.quadratic_peak(coefficients, np.array(critic)))
        write(result, out, name, cli.subset)
        report_1d(result, coefficients, {"J(c)": summary}, result.get("critic"), center, "noise amplitude c")
    else:
        kinds = [cli.slice] if cli.slice else cli.plane.split(",")
        directions = [resolve_direction(kind, adapter, base, mask, update, states, cli.critic_samples,
                                        cli.seed + 101 * i) for i, kind in enumerate(kinds)]
        if len(directions) == 2:
            directions[1] = ls.orthogonalize(directions[1], directions[0])
        if cli.unit == "update":
            unit = ls.norm(ls.restrict(update(), mask))
        else:
            unit = ls.norm(ls.random_direction(base, cli.seed + 12345, mask))
        raw_norms = [float(ls.norm(d)) for d in directions]
        directions = [ls.scaled(d, unit / ls.norm(d).clamp_min(1e-30)) for d in directions]
        coefficients = parse_coefficients(cli.coefficients or ("-1:3:17" if cli.unit == "update" else "-0.1:0.1:11"))
        if len(directions) == 1:
            grid = coefficients[:, None]
        else:
            grid = np.stack(np.meshgrid(coefficients, coefficients, indexing="ij"), -1).reshape(-1, 2)
        points = ls.grid_points(base, directions, torch.as_tensor(grid))
        center = int(np.argmin(np.abs(grid).sum(1)))
        per_scale = {}
        for scale in scales:
            returns = ls.episode_returns(make_env, adapter, points, cli.episodes, scale, cli.seed, cli.max_envs).returns
            per_scale[scale] = {"returns": returns, "summary": ls.paired_summary(returns, center)}
        if cli.to is not None:
            # J(theta_to) under its own normalization, paired with the center (same resets and noise).
            other = target()
            reference = {}
            for scale in scales:
                returns = ls.episode_returns(make_env, other, {k: v[None] for k, v in
                                                               ls.base_parameters(other.module).items()},
                                             cli.episodes, scale, cli.seed, cli.max_envs).returns[0]
                diff = returns - per_scale[scale]["returns"][center]
                reference[str(scale)] = {"mean": float(returns.mean()), "diff": float(diff.mean()),
                                         "diff_se": float(diff.std(ddof=1) / np.sqrt(len(diff)))}
            shift = ((other.obs_mean - adapter.obs_mean) * adapter.obs_inv_std).norm()
            result.update(to_reference=reference, normalization_shift=float(shift),
                          normalization_scale_ratio=float((other.obs_inv_std / adapter.obs_inv_std).log().abs().max()))
        result.update(directions=kinds, direction_raw_norms=raw_norms, unit_norm=float(unit),
                      coefficients=coefficients.tolist(),
                      per_scale={str(s): {"returns": v["returns"].tolist(),
                                          "summary": {k: x.tolist() for k, x in v["summary"].items()}}
                                 for s, v in per_scale.items()})
        if cli.to is not None:
            # Cosine of each direction with the actual step: how aligned the critic's ascent is with it.
            step = ls.restrict(update(), mask)
            result["update_norm"] = float(ls.norm(step))
            result["update_cosine"] = [float(ls.dot(d, step) / (ls.norm(d) * ls.norm(step)).clamp_min(1e-30))
                                       for d in directions]
        critic = None
        if states is not None:
            with torch.no_grad():
                critic = ls.critic_objective(adapter, points, held_out, cli.critic_samples, 1.0,
                                             critic_seed).cpu().numpy()
                regularized = ls.critic_objective(adapter, points, held_out, cli.critic_samples, 1.0, critic_seed,
                                                  regularized=True).cpu().numpy()
            result.update(critic=critic.tolist(), critic_regularized=regularized.tolist())
        if len(directions) == 1:
            result["fit"] = {str(s): ls.paired_quadratic(coefficients, v["summary"]["diff"], v["summary"]["diff_se"],
                                                         center) for s, v in per_scale.items()}
            if critic is not None:
                result["critic_fit"] = ls.quadratic_peak(coefficients, critic)
                result["critic_regularized_fit"] = ls.quadratic_peak(coefficients, regularized)
                result["critic_true_corr"] = {str(s): float(np.corrcoef(critic, v["summary"]["mean"])[0, 1])
                                              for s, v in per_scale.items()}
            write(result, out, name, cli.subset)
            report_1d(result, coefficients, {f"J scale={s:g}": v["summary"] for s, v in per_scale.items()},
                      critic, center, f"{kinds[0]} coefficient ({cli.unit} units)")
        else:
            write(result, out, name, cli.subset)
            report_2d(result, coefficients, per_scale, critic, kinds, cli.unit, center)
    result["elapsed_s"] = time.perf_counter() - started
    write(result, out, name, cli.subset)
    plot(result, out / f"{name}_{cli.subset}.png")


def write(result, out, name, subset):
    """Written before any reporting, so a reporting error cannot lose the rollouts."""
    path = out / f"{name}_{subset}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}")


# ---------------------------------------------------------------- reporting

def report_1d(result, coefficients, summaries, critic, center, label):
    print(f"\n{result['snapshot']}  slice={result['slice']} subset={result['subset']}  x = {label}")
    header = f"{'x':>8}" + "".join(f"  {name:>26}" for name in summaries) + ("  critic dE[Q]" if critic is not None else "")
    print(header)
    for i, c in enumerate(coefficients):
        row = f"{c:8.3f}" + "".join(f"  {s['mean'][i]:9.0f} {s['diff'][i]:+8.0f}±{s['diff_se'][i]:<6.0f}"
                                    for s in summaries.values())
        if critic is not None:
            row += f"  {critic[i] - critic[center]:+.4f}"
        print(row)
    for key in ("fit", "critic_fit", "critic_regularized_fit", "critic_true_corr", "update_cosine",
                "to_reference", "normalization_shift", "critic_states"):
        if key in result:
            print(f"{key}: {result[key]}")


def report_2d(result, coefficients, per_scale, critic, kinds, unit, center):
    print(f"\n{result['snapshot']}  plane={kinds} ({unit} units)")
    n = len(coefficients)
    for scale, value in per_scale.items():
        grid = value["summary"]["mean"].reshape(n, n)
        i, j = np.unravel_index(np.argmax(grid), grid.shape)
        print(f"scale {scale:g}: center {grid[np.unravel_index(center, grid.shape)]:.0f}, max {grid[i, j]:.0f} at "
              f"({coefficients[i]:.3f}, {coefficients[j]:.3f})")
    if critic is not None:
        grid = np.asarray(critic).reshape(n, n)
        i, j = np.unravel_index(np.argmax(grid), grid.shape)
        print(f"critic: max at ({coefficients[i]:.3f}, {coefficients[j]:.3f})")


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coefficients = np.asarray(result["coefficients"])
    critic = result.get("critic")
    if result["slice"] == "spread" or "per_scale" in result and len(result.get("directions", [])) == 1:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        series = ({"J(c)": result["summary"]} if result["slice"] == "spread"
                  else {f"scale {s}": v["summary"] for s, v in result["per_scale"].items()})
        for name, summary in series.items():
            mean, se = np.asarray(summary["mean"]), np.asarray(summary["diff_se"])
            ax.plot(coefficients, mean, marker="o", label=name)
            ax.fill_between(coefficients, mean - se, mean + se, alpha=0.2)
        ax.set_xlabel("noise amplitude c" if result["slice"] == "spread" else f"{result['slice']} coefficient")
        ax.set_ylabel("episode return (paired se band)")
        if critic is not None:
            twin = ax.twinx()
            twin.plot(coefficients, critic, color="black", linestyle="--", label="critic E[Q]")
            twin.set_ylabel("critic surrogate (scaled)")
            twin.legend(loc="lower right")
        ax.legend(loc="upper left")
        ax.set_title(Path(result["snapshot"]).parent.name[:60], fontsize=8)
    else:
        n = len(coefficients)
        panels = [(f"J scale {s}", np.asarray(v["summary"]["mean"]).reshape(n, n))
                  for s, v in result["per_scale"].items()]
        if critic is not None:
            panels.append(("critic E[Q]", np.asarray(critic).reshape(n, n)))
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2))
        for ax, (title, grid) in zip(np.atleast_1d(axes), panels):
            contour = ax.contourf(coefficients, coefficients, grid.T, levels=20)
            fig.colorbar(contour, ax=ax)
            ax.plot(0, 0, "w+", markersize=12)
            ax.set_xlabel(result["directions"][0])
            ax.set_ylabel(result["directions"][1] + " (orthogonalized)")
            ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
