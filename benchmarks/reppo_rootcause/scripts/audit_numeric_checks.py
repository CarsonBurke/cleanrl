"""CPU-only numeric checks for the REPPO code audit (no training).

1 HL-Gauss encode/decode round trip and its target-derivative at the scales in play.
2 Soft lambda-return recursion of v60 (reference GAE path) against a brute-force definition on a toy chain.
3 FP32 vs FP64 critic values, within-state contrasts and dQ/da on a trained checkpoint.
4 Per-env observation-statistic spread, first-layer action/observation weights.
"""
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import torch

torch.set_num_threads(2)
REPO = "/home/marvin/Documents/repositories/cleanrl"
sys.path.insert(0, REPO)
spec = importlib.util.spec_from_file_location("v60", f"{REPO}/cleanrl/reppo/ppo_continuous_action_reppo_noisefork_v60.py")
v60 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v60)
from cleanrl.shared.ppo_loop import compute_gae_from_next_values  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{REPO}/runs/HalfCheetah-v4__reppo_noisefork_v60_base__1__1790713932/checkpoint_1015424.pt"

print("== 1 HL-Gauss round trip (151 bins on [-50, 250], width 2, sigma 1.5)")
hl = v60.HLGaussConfig(v_min=-50.0, v_max=250.0, num_bins=151).build("cpu")
for dtype in (torch.float32, torch.float64):
    x = torch.linspace(-50, 250, 30001, dtype=torch.float64)
    probs = hl.project(x.to(dtype))
    back = (probs.double() * hl.support.double()).sum(-1)
    err = (back - x)
    interior = (x > -40) & (x < 240)
    print(f"  {dtype}: max |decode(project(x)) - x| interior {err[interior].abs().max():.2e}, at edges {err.abs().max():.3f}; "
          f"slope d decode / d x interior min {(np.gradient(back[interior].numpy(), x[interior].numpy())).min():.5f} "
          f"max {(np.gradient(back[interior].numpy(), x[interior].numpy())).max():.5f}")
# resolution of a 0.03 contrast at Q = 100 in float32
a = hl.project(torch.tensor([100.0, 100.03]))
print("  float32 decode of labels at 100.00 and 100.03:", (a * hl.support).sum(-1).tolist())
# noisy targets: CE-optimal histogram is the label mixture; its decoded mean vs the target mean
g = torch.Generator().manual_seed(0)
for mean, sd in ((100.0, 0.4), (100.0, 3.0), (245.0, 3.0)):
    t = mean + sd * torch.randn(200000, generator=g)
    print(f"  targets N({mean}, {sd}): mean {t.mean():.4f}, decoded CE-optimum {(hl.project(t).mean(0) * hl.support).sum():.4f}")

print("== 2 soft lambda-return recursion on a toy chain")
T, N, gamma, lam = 40, 3, 0.99, 0.8
g = torch.Generator().manual_seed(1)
rewards = torch.randn(T, N, generator=g, dtype=torch.float64)
vnext = torch.randn(T, N, generator=g, dtype=torch.float64)  # V'(s_{t+1}) incl. bonus
bonus = torch.rand(T, N, generator=g, dtype=torch.float64)
term = torch.zeros(T, N, dtype=torch.float64)
trunc = torch.zeros(T, N, dtype=torch.float64)
term[7, 0] = term[20, 1] = 1.0
trunc[13, 0] = trunc[20, 2] = trunc[38, 1] = 1.0
ends = torch.zeros(T, N, dtype=torch.float64)
ends[-1] = 1.0
cuts = torch.maximum(trunc, ends)
continuing = 1.0 - term
soft_rewards = rewards + gamma * lam * continuing * (1.0 - cuts) * bonus
got = v60.lambda_returns(compute_gae_from_next_values, soft_rewards, vnext, term, cuts, gamma, lam)
want = torch.zeros(T, N, dtype=torch.float64)
for n in range(N):
    for t in range(T - 1, -1, -1):
        if term[t, n]:
            want[t, n] = rewards[t, n]  # nothing after a termination
        elif cuts[t, n]:
            want[t, n] = rewards[t, n] + gamma * vnext[t, n]  # truncation / rollout end: bootstrap V' alone
        else:  # G_t = r + gamma [(1 - lam) V'_{t+1} + lam (G_{t+1} + T H(s_{t+1}))]
            want[t, n] = rewards[t, n] + gamma * ((1 - lam) * vnext[t, n] + lam * (want[t + 1, n] + bonus[t, n]))
print(f"  max |v60 lambda_returns - brute force| = {(got - want).abs().max():.2e}")
weights = [(1 - lam) * lam ** (k - 1) for k in range(1, 6)]
print(f"  n-step mixture weights (1-lam) lam^(n-1), n=1..5: {np.round(weights, 3).tolist()}, sum {sum(weights):.3f}; "
      f"all of the target beyond the rollout's rewards is bootstrap")

print("== 3 FP32 vs FP64 on the trained critic")
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
args = SimpleNamespace(**ck["args"])
obs_dim, act_dim = ck["obs_means"].shape[1], len(ck["temperature"])
critic = v60.Critic(obs_dim, act_dim, args, "cpu")
critic.load_state_dict(ck["critic"])
actor = v60.Actor(obs_dim, act_dim, args)
actor.load_state_dict(ck["actor_state"])
critic64 = v60.Critic(obs_dim, act_dim, args, "cpu").double()
critic64.load_state_dict(ck["critic"])
critic64.hl = critic64.hl.double()
g = torch.Generator().manual_seed(2)
obs = torch.randn(2048, obs_dim, generator=g)
with torch.no_grad():
    mean, std = actor(obs)
    acts = torch.tanh(mean[None] + std[None] * torch.randn(8, 2048, act_dim, generator=g))
    q32 = critic.value(obs.expand(8, -1, -1), acts)
    q64 = critic64.value(obs.double().expand(8, -1, -1), acts.double())
c32, c64 = q32 - q32.mean(0), q64 - q64.mean(0)
print(f"  Q mean {q64.mean():.2f}; |Q32 - Q64| max {(q32 - q64).abs().max():.2e}; within-state contrast sd {c64.std():.4f}, "
      f"contrast error max {(c32 - c64).abs().max():.2e} (rms {(c32 - c64).square().mean().sqrt():.2e})")
a32 = acts[0].clone().requires_grad_(True)
(g32,) = torch.autograd.grad(critic.value(obs, a32).sum(), a32)
a64 = acts[0].double().clone().requires_grad_(True)
(g64,) = torch.autograd.grad(critic64.value(obs.double(), a64).sum(), a64)
rel = (g32 - g64).norm(dim=-1) / g64.norm(dim=-1)
print(f"  dQ/da: |g| median {g64.norm(dim=-1).median():.3f}; relative FP32 error median {rel.median():.2e}, max {rel.max():.2e}")

print("== 4 normalization and first layer")
means, variances = ck["obs_means"], ck["obs_variances"]
sd = np.sqrt(variances + ck["obs_epsilon"])
spread = means.std(0) / sd.mean(0)
print(f"  per-env obs mean spread across the 16 lanes, in pooled sd units: median {np.median(spread):.3f}, max {spread.max():.3f}; "
      f"per-env sd ratio max/min per dim: median {np.median(sd.max(0) / sd.min(0)):.3f}, max {(sd.max(0) / sd.min(0)).max():.3f}; "
      f"counts {ck['obs_counts'].min():.0f}..{ck['obs_counts'].max():.0f}")
w = ck["critic"]["encoder.0.weight"]
print(f"  critic first-layer column norms: obs dims mean {w[:, :obs_dim].norm(dim=0).mean():.3f} "
      f"(min {w[:, :obs_dim].norm(dim=0).min():.3f}, max {w[:, :obs_dim].norm(dim=0).max():.3f}), "
      f"action dims {np.round(w[:, obs_dim:].norm(dim=0).numpy(), 3).tolist()}")
print(f"  temperature {ck['temperature'][0]:.5f}, log_kl_multiplier {float(ck['actor_state']['log_kl_multiplier']):.3f}, "
      f"zero_dist norm {ck['critic']['zero_dist'].norm():.3f}")
