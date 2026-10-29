"""Per-state action-noise structure of trained FPO flow policies (vs a Beta PPO reference).

Question: did the flow learn coordinated (cross-dimension correlated) exploration noise,
which a factorized Gaussian/Beta head cannot express, or is each per-state action
distribution essentially axis-aligned and independent?

For each checkpoint:
1. Roll out the stochastic policy on N_ENVS HalfCheetah episodes using the checkpoint's own
   Agent class (loaded from the run's saved source.py) and its saved per-env observation
   moments (frozen; row i normalizes env i, as in training). States are recorded every
   `stride` steps after `skip` steps.
2. Draw K actions per recorded state with fresh initial noise through Agent.sample: the
   same Euler integration and decoder used by the training HostSampler, run on CUDA in FP32
   with TF32 disabled. Statistics use pre-decoder ("native") and env actions.
3. References: "shuffle" permutes each dimension independently across the K samples (same
   marginals, dependence destroyed, giving the K-sample noise floor for |corr| and TC);
   "gauss" draws an independent Gaussian with the flow's per-state, per-dimension native
   mean and std, decoded the same way (an independence-and-Gaussianity reference whose
   post-decoder shape comes only from the squash).
"""

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_CHECKPOINTS = [
    "fpo_v1_velocity=runs/HalfCheetah-v4__fpo_v1_velocity__1__1789275435859923150/fpo_v1_velocity.cleanrl_model",
    "fpo_v2_tanh_velocity=runs/HalfCheetah-v4__fpo_v2_tanh_velocity__1__1789275581221096172/fpo_v2_tanh_velocity.cleanrl_model",
    "fpo_v1_eps=runs/HalfCheetah-v4__fpo_v1__1__1789254081580996253/fpo_v1.cleanrl_model",
]
DEFAULT_BETA = ("beta_ppo_unclipped_adam=runs/HalfCheetah-v4__ppo_unclipped_adam_control_8M_v1__1__1789772099/"
                "ppo_unclipped_adam_control_8M_v1.cleanrl_model")
BETA_SOURCE = REPO / "cleanrl" / "ppo_continuous_action_unclipped_adam_v1.py"
DEGENERATE_STD = 1e-4  # env-action std below this counts as "no exploration" in that dim
BC_THRESHOLD = 5.0 / 9.0  # Sarle bimodality coefficient of a uniform distribution


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def env_spaces(env_id):
    env = gym.make(env_id)
    spaces = SimpleNamespace(single_action_space=env.action_space,
                             single_observation_space=env.observation_space)
    env.close()
    return spaces


class FlowPolicy:
    kind = "flow"

    def __init__(self, path, device, env_id):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        run_dir = Path(path).parent
        module = load_module(run_dir / "source.py", f"src_{run_dir.name.replace('-', '_').replace('.', '_')}")
        self.args = SimpleNamespace(**ckpt["args"])
        self.agent = module.Agent(env_spaces(env_id), self.args).to(device)
        self.agent.load_state_dict(ckpt["state_dict"])
        self.agent.eval()
        self.decoded = getattr(self.args, "action_transform", "tanh") == "tanh"
        obs = ckpt["normalization"]["observation"]
        self.obs_norm = {k: obs[k].numpy() for k in ("means", "variances")}
        self.obs_norm.update(epsilon=obs["epsilon"], clip=obs["clip"])
        self.device = device
        self.action_dim = self.agent.action_dim

    @torch.no_grad()
    def sample(self, observations, generator):
        noise = torch.randn((observations.shape[0], self.action_dim), device=self.device, generator=generator)
        return self.agent.sample(observations, noise)

    def decode(self, native):
        return self.agent.decode(native) if self.decoded else native


class BetaPolicy:
    kind = "beta"

    def __init__(self, path, device, env_id):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        module = load_module(BETA_SOURCE, "beta_src")
        self.agent = module.Agent(env_spaces(env_id)).to(device)
        self.agent.load_state_dict(ckpt["model"])
        self.agent.eval()
        self.args = SimpleNamespace(**ckpt["args"])
        obs = ckpt["obs_norm"]
        self.obs_norm = {k: obs[k].numpy() for k in ("means", "variances")}
        self.obs_norm.update(epsilon=obs["epsilon"], clip=obs["clip"])
        self.device = device
        self.action_dim = self.agent.action_dim

    @torch.no_grad()
    def sample(self, observations, generator):
        alpha, beta = (torch.nn.functional.softplus(self.agent.actor(observations)) + 1.0).chunk(2, dim=-1)
        # Beta(a,b) = Ga/(Ga+Gb); torch's Gamma sampler takes no generator, so seed globally.
        native = torch.distributions.Beta(alpha, beta).sample()
        return native, self.decode(native)

    def decode(self, native):
        return self.agent.action_low + self.agent.action_scale * native


def normalize(policy, raw, rows):
    n = policy.obs_norm
    out = (raw - n["means"][rows]) / np.sqrt(n["variances"][rows] + n["epsilon"])
    return np.clip(out, -n["clip"], n["clip"]).astype(np.float32)


@torch.no_grad()
def rollout(policy, env_id, num_envs, skip, stride, seed, generator):
    envs = [gym.wrappers.ClipAction(gym.wrappers.FlattenObservation(gym.make(env_id))) for _ in range(num_envs)]
    raw = np.stack([env.reset(seed=seed + i)[0] for i, env in enumerate(envs)])
    rows = np.arange(num_envs)
    returns, done = np.zeros(num_envs), np.zeros(num_envs, dtype=bool)
    states, t = [], 0
    while not done.all():
        obs = normalize(policy, raw, rows)
        if t >= skip and (t - skip) % stride == 0:
            states.append(obs[~done])
        _, action = policy.sample(torch.from_numpy(obs).to(policy.device), generator)
        action = action.cpu().numpy()
        for i, env in enumerate(envs):
            if done[i]:
                continue
            raw[i], reward, term, trunc, _ = env.step(action[i])
            returns[i] += reward
            done[i] = term or trunc
        t += 1
    for env in envs:
        env.close()
    return np.concatenate(states), returns


def moments(x):
    """x: (S,K,D) float64 -> mean, std, skew, excess kurtosis over K (biased estimators)."""
    mean = x.mean(1, keepdim=True)
    c = x - mean
    var = c.pow(2).mean(1)
    std = var.sqrt()
    z = c / std.clamp_min(1e-300).unsqueeze(1)
    return mean.squeeze(1), std, z.pow(3).mean(1), z.pow(4).mean(1) - 3.0


def bimodality(skew, kurt, n):
    return (skew ** 2 + 1.0) / (kurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def structure(x):
    """Per-state dependence/shape statistics for samples x of shape (S,K,D)."""
    S, K, D = x.shape
    _, std, skew, kurt = moments(x)
    deg = std < DEGENERATE_STD
    c = x - x.mean(1, keepdim=True)
    cov = c.transpose(1, 2) @ c / K
    inv = torch.where(deg, torch.zeros_like(std), 1.0 / std.clamp_min(1e-300))
    corr = cov * inv.unsqueeze(2) * inv.unsqueeze(1)
    eye = torch.eye(D, dtype=x.dtype, device=x.device)
    corr = torch.where(deg.unsqueeze(2) | deg.unsqueeze(1), eye.expand(S, D, D), corr)
    corr = 0.5 * (corr + corr.transpose(1, 2))
    corr.diagonal(dim1=1, dim2=2).fill_(1.0)
    off = ~torch.eye(D, dtype=torch.bool, device=x.device)
    offv = corr[:, off]  # (S, D*(D-1)) each pair twice
    iu = torch.triu_indices(D, D, 1, device=x.device)
    pairs = corr[:, iu[0], iu[1]]  # (S, P)

    tc = -0.5 * torch.linalg.slogdet(corr)[1]
    ev_corr, vec_corr = torch.linalg.eigh(corr)
    ev_cov, vec_cov = torch.linalg.eigh(cov)
    ev_cov = ev_cov.clamp_min(0)
    pr_cov = ev_cov.sum(1) ** 2 / ev_cov.pow(2).sum(1).clamp_min(1e-300)
    pr_corr = ev_corr.sum(1) ** 2 / ev_corr.pow(2).sum(1)
    top_frac_cov = ev_cov[:, -1] / ev_cov.sum(1).clamp_min(1e-300)

    # Shape along the top principal component of the covariance.
    proj = (c @ vec_cov[:, :, -1:]).squeeze(-1).unsqueeze(-1)
    _, _, pc_skew, pc_kurt = moments(proj)
    pc_bc = bimodality(pc_skew, pc_kurt, K).squeeze(-1)

    # Nonlinear dependence linear corr misses: corr of squared z-scores (variance coupling).
    zz = (c * inv.unsqueeze(1)).pow(2)
    zz = zz - zz.mean(1, keepdim=True)
    zstd = zz.pow(2).mean(1).sqrt()
    zcorr = (zz.transpose(1, 2) @ zz / K) / (zstd.unsqueeze(2) * zstd.unsqueeze(1)).clamp_min(1e-300)
    valid = ~(deg.unsqueeze(2) | deg.unsqueeze(1))
    zpairs = zcorr[:, iu[0], iu[1]][valid[:, iu[0], iu[1]]]

    # State dependence: decompose per-state off-diagonal corr into global mean + per-state deviation.
    mean_pairs = pairs.mean(0)
    within = (pairs - mean_pairs).pow(2).sum(1).mean()
    mean_corr = corr.mean(0)
    gvec = torch.linalg.eigh(mean_corr)[1][:, -1]
    top_align = (vec_corr[:, :, -1] @ gvec).abs()
    perm = torch.randperm(S, device=x.device)
    pair_align = (vec_corr[:, :, -1] * vec_corr[perm, :, -1]).sum(1).abs()

    bc = bimodality(skew, kurt, K)
    live = ~deg
    return {
        "std": std, "deg_frac": deg.double().mean().item(),
        "mean_abs_corr": offv.abs().mean(1), "max_abs_corr": offv.abs().amax(1), "abs_pairs": pairs.abs(),
        "tc": tc, "pr_cov": pr_cov, "pr_corr": pr_corr, "top_frac_cov": top_frac_cov,
        "top_eig_corr": ev_corr[:, -1],
        "kurt": kurt[live], "abs_skew": skew[live].abs(), "bc_frac": (bc[live] > BC_THRESHOLD).double().mean().item(),
        "pc_kurt": pc_kurt.squeeze(-1), "pc_bc_frac": (pc_bc > BC_THRESHOLD).double().mean().item(),
        "sqcorr_abs": zpairs.abs(),
        "global_mean_abs_corr": mean_pairs.abs().mean().item(),
        "global_max_abs_corr": mean_pairs.abs().max().item(),
        "within_state_dev": within.item(), "total_sq": pairs.pow(2).sum(1).mean().item(),
        "top_align_global": top_align.mean().item(), "top_align_pairs": pair_align.mean().item(),
        "mean_pairs": mean_pairs,
    }


def shuffle_columns(x, generator):
    S, K, D = x.shape
    idx = torch.rand((S, K, D), device=x.device, generator=generator).argsort(1)
    return x.gather(1, idx)


def q(t, p):
    return torch.quantile(t.flatten().double().cpu(), p).item()


def summarize(name, r, null99):
    return {
        "mean|corr| (med/p90)": f"{q(r['mean_abs_corr'], .5):.3f}/{q(r['mean_abs_corr'], .9):.3f}",
        "max|corr| (med/p90)": f"{q(r['max_abs_corr'], .5):.3f}/{q(r['max_abs_corr'], .9):.3f}",
        "pairs>|null p99|": f"{(r['abs_pairs'] > null99).double().mean().item():.3f}",
        "TC nats (med/p90)": f"{q(r['tc'], .5):.3f}/{q(r['tc'], .9):.3f}",
        "PR corr (med)": f"{q(r['pr_corr'], .5):.2f}",
        "PR cov (med)": f"{q(r['pr_cov'], .5):.2f}",
        "top-eig frac cov": f"{q(r['top_frac_cov'], .5):.3f}",
        "exkurt (med/p10/p90)": f"{q(r['kurt'], .5):+.2f}/{q(r['kurt'], .1):+.2f}/{q(r['kurt'], .9):+.2f}",
        "|skew| med": f"{q(r['abs_skew'], .5):.2f}",
        "BC>5/9 dims": f"{r['bc_frac']:.3f}",
        "BC>5/9 topPC": f"{r['pc_bc_frac']:.3f}",
        "|corr z^2| mean": f"{r['sqcorr_abs'].mean().item():.3f}",
        "global mean|corr|": f"{r['global_mean_abs_corr']:.3f} (max {r['global_max_abs_corr']:.3f})",
        "state-dev / total": f"{r['within_state_dev']:.3f}/{r['total_sq']:.3f}",
        "topvec |cos| glob/pairs": f"{r['top_align_global']:.2f}/{r['top_align_pairs']:.2f}",
        "degenerate dims": f"{r['deg_frac']:.3f}",
    }


@torch.no_grad()
def analyze(label, policy, cfg, device):
    g = torch.Generator(device=device).manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    t0 = time.perf_counter()
    states, returns = rollout(policy, cfg.env_id, cfg.num_envs, cfg.skip, cfg.stride, cfg.seed, g)
    obs = torch.from_numpy(states).to(device)
    S, K = obs.shape[0], cfg.samples
    natives = []
    for chunk in obs.split(max(1, cfg.chunk_rows // K)):
        native, _ = policy.sample(chunk.repeat_interleave(K, 0), g)
        natives.append(native.view(chunk.shape[0], K, -1))
    native = torch.cat(natives).double()
    if policy.kind == "flow" and policy.decoded:
        env = policy.agent.action_bias.double() + policy.agent.action_scale.double() * native.tanh()
    else:
        env = policy.decode(native).double()
    env = env.clamp(-1.0, 1.0)  # ClipAction: what the environment actually receives

    out = {"label": label, "kind": policy.kind, "states": S, "K": K,
           "rollout_returns": returns.round(1).tolist(), "views": {}}
    views = {"env": env} if policy.kind == "beta" else {"native": native, "env": env}
    mu_n, sd_n = native.mean(1, keepdim=True), native.std(1, keepdim=True)
    gauss_native = mu_n + sd_n * torch.randn(native.shape, device=device, generator=g, dtype=native.dtype)
    gauss_env = (policy.agent.action_bias.double() + policy.agent.action_scale.double() * gauss_native.tanh()
                 if policy.kind == "flow" and policy.decoded else gauss_native).clamp(-1, 1)
    gauss = {"native": gauss_native, "env": gauss_env}
    for view, x in views.items():
        res = structure(x)
        nul = structure(shuffle_columns(x, g))
        gau = structure(gauss[view] if policy.kind == "flow" else x)
        null99 = q(nul["abs_pairs"], 0.99)
        out["views"][view] = {
            "flow": summarize("flow", res, null99), "shuffle": summarize("shuffle", nul, null99),
            "gauss_diag": summarize("gauss", gau, null99), "null_abs_corr_p99": null99,
            "global_mean_pairs": [round(v, 3) for v in res["mean_pairs"].tolist()],
        }
        if view == "env":
            std = res["std"]
            out["exploration"] = {
                "env_std_per_dim_mean": [round(v, 3) for v in std.mean(0).tolist()],
                "env_std_mean": std.mean().item(), "env_std_over_range": (std / 2.0).mean().item(),
                "saturated_frac(|a|>0.99)": (x.abs() > 0.99).double().mean().item(),
                "across_state_std_of_mean": x.mean(1).std(0).mean().item(),
            }
            if policy.kind == "flow":
                out["exploration"]["native_std_mean"] = native.std(1).mean().item()
    out["seconds"] = time.perf_counter() - t0
    return out


def print_report(out):
    print(f"\n=== {out['label']} ({out['kind']}) states={out['states']} K={out['K']} "
          f"rollout_return mean={np.mean(out['rollout_returns']):.0f} min={np.min(out['rollout_returns']):.0f} "
          f"max={np.max(out['rollout_returns']):.0f} [{out['seconds']:.0f}s]")
    print("  exploration:", json.dumps(out["exploration"]))
    for view, v in out["views"].items():
        cols = ("flow", "shuffle") if out["kind"] == "beta" else ("flow", "shuffle", "gauss_diag")
        print(f"  [{view}] null |corr| p99={v['null_abs_corr_p99']:.3f}; global mean offdiag corr={v['global_mean_pairs']}")
        print(f"    {'stat':26s}" + "".join(f"{c:>24s}" for c in cols))
        for key in v["flow"]:
            print(f"    {key:26s}" + "".join(f"{v[c][key]:>24s}" for c in cols))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", action="append", help="label=path (flow); default: the three FPO runs")
    p.add_argument("--beta", default=DEFAULT_BETA, help="label=path Beta PPO reference ('' to skip)")
    p.add_argument("--env-id", default="HalfCheetah-v4")
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument("--skip", type=int, default=50)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--chunk-rows", type=int, default=262144)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="benchmarks/fpo_noise_structure/results.json")
    cfg = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    specs = [(s.split("=", 1)[0], s.split("=", 1)[1], FlowPolicy) for s in (cfg.checkpoint or DEFAULT_CHECKPOINTS)]
    if cfg.beta:
        specs.append((cfg.beta.split("=", 1)[0], cfg.beta.split("=", 1)[1], BetaPolicy))
    results = []
    for label, path, cls in specs:
        out = analyze(label, cls(REPO / path, device, cfg.env_id), cfg, device)
        out["checkpoint"] = path
        print_report(out)
        results.append(out)
        sys.stdout.flush()
    dest = REPO / cfg.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"config": vars(cfg), "results": results}, indent=1))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
