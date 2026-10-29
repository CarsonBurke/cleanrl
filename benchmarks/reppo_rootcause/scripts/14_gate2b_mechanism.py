"""Gate 2, mechanism: (i) is a spike the action-sensitivity of a second mode of the HL-Gauss value distribution?
(ii) is the truth as large as the critic says where the critic's gradient is large (oracle datasets, v23 / v29)?

    CUDA_VISIBLE_DEVICES= .venv/bin/python benchmarks/reppo_rootcause/scripts/14_gate2b_mechanism.py

(i)  With k* the modal bin and s_k the bin centres, Q - s_k* = sum_k p_k (s_k - s_k*) splits exactly into a core part
     (bins within CORE of the mode: the label kernel's own width), a low tail and a high tail; so does dQ/da.
(ii) At the 512 oracle states the finite-difference truth at the same action a0 is projected on the critic's own unit
     gradient direction: an unbiased estimate of the true slope along the direction the actor is pushed in.
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("common14", Path(__file__).resolve().parent / "14_common.py")
c14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c14)
common = c14.common
spec = importlib.util.spec_from_file_location("d1", Path(__file__).resolve().parent / "07_D_headtohead.py")
d1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d1)
torch.set_num_threads(1)
CORE = 6.0  # value units; bins are 2.0 wide and the label kernel's sd is 1.5


def modes(step):
    ck, s = c14.load_checkpoint(step), c14.load_states(step)
    critic = ck.critic
    obs = torch.from_numpy(c14.normalise(s["raw"], s["lane"], ck))
    action = torch.from_numpy(s["action"]).clone().requires_grad_(True)
    probs = critic.logits(critic.features(obs, action)).softmax(-1)
    support = critic.hl.support
    centre = support[probs.argmax(-1)].detach()
    offset = support[None] - centre[:, None]
    parts = {"low tail": offset < -CORE, "core": offset.abs() <= CORE, "high tail": offset > CORE}
    grads = {k: torch.autograd.grad((probs * offset * m).sum(), action, retain_graph=True)[0].double().numpy() for k, m in parts.items()}
    mass = {k: (probs * m).sum(-1).detach().double().numpy() for k, m in parts.items()}
    mean_offset = {k: ((probs * offset * m).sum(-1) / (probs * m).sum(-1).clamp_min(1e-12)).detach().double().numpy() for k, m in parts.items()}
    total = sum(grads.values())
    jac = 1 - s["action"].astype(np.float64) ** 2
    norm_pre = np.linalg.norm(total * jac, axis=1)
    spike = norm_pre >= np.quantile(norm_pre, 0.99)
    print(f"\n===== checkpoint {step}: value distribution at the {spike.sum()} top-1% states (pre-tanh norm) against the rest; bin width "
          f"{float(support[1] - support[0]):.2f}, core = mode +- {CORE}")
    for name in parts:
        share = (grads[name] * total).sum(1) / (total ** 2).sum(1)
        pooled = lambda m: (grads[name][m] * total[m]).sum() / (total[m] ** 2).sum()
        print(f"   {name:9s} mass: top-1% mean {mass[name][spike].mean():.4f} (median {np.median(mass[name][spike]):.4f}), rest mean {mass[name][~spike].mean():.4f} "
              f"(median {np.median(mass[name][~spike]):.5f}); mean distance from the mode top-1% {mean_offset[name][spike].mean():+.1f}, rest {mean_offset[name][~spike].mean():+.1f}; "
              f"share of dQ/da (projection on the total): top-1% energy-weighted {pooled(spike):+.2f}, median {np.median(share[spike]):+.2f}; rest energy-weighted {pooled(~spike):+.2f}, median {np.median(share[~spike]):+.2f}")
    tail = mass["low tail"] + mass["high tail"]
    order = np.argsort(-tail)
    energy = norm_pre ** 2
    print(f"   states ranked by tail mass: the top 1% / 5% / 20% hold {energy[order[:len(order) // 100]].sum() / energy.sum():.2f} / "
          f"{energy[order[:len(order) // 20]].sum() / energy.sum():.2f} / {energy[order[:len(order) // 5]].sum() / energy.sum():.2f} of the gradient energy; "
          f"Spearman(tail mass, |g_pre|) {np.corrcoef(np.argsort(np.argsort(tail)), np.argsort(np.argsort(norm_pre)))[0, 1]:+.2f}; "
          f"states with tail mass > 5%: {np.mean(tail > .05):.3f} of rows, {energy[tail > .05].sum() / energy.sum():.2f} of energy")
    # Does the low mode correspond to something that happens? Realised discounted reward over the next 100 steps.
    reward, t = s["reward"] * ck.args.reward_scale, s["t"]
    horizon, gamma = 100, ck.args.gamma
    realised = np.full(len(reward), np.nan)
    for i in range(len(reward)):
        if t[i] + horizon <= 1000:
            realised[i] = (reward[i:i + horizon] * gamma ** np.arange(horizon)).sum()
    usable = np.isfinite(realised)
    typical = np.nanmedian(realised)
    for label, m in (("top-1%", spike & usable), ("rest", ~spike & usable)):
        print(f"   realised discounted reward over the next {horizon} steps, {label}: mean {realised[m].mean():.2f}, p10 {np.quantile(realised[m], .1):.2f}, "
              f"share more than 5 below the median ({typical:.1f}) {np.mean(realised[m] < typical - 5):.3f}  (n {m.sum()})")


def truth_at_spikes(name):
    d = common.load(name, threads=1)
    truth, _ = d1.truth_fields(d, 100)
    jac = 1.0 - d.action[d.index] ** 2
    g_c = d.grad[d.index] * jac
    norm = np.linalg.norm(g_c, axis=1)
    unit = g_c / norm[:, None]
    chunk_norm = np.linalg.norm(d.grad * (1 - d.action ** 2), axis=1)
    rng = np.random.default_rng(0)
    print(f"\n===== {name}: critic vs branching truth by size of the critic's gradient, 512 oracle states (chunk top-1% energy share "
          f"{np.sort(chunk_norm ** 2)[-164:].sum() / (chunk_norm ** 2).sum():.2f}; chunk p99 of |g| {np.quantile(chunk_norm, .99):.2f})")
    fd = truth["fd_a0_h0.2"]  # (n, 6, M), same action as the critic gradient
    m = fd.shape[-1]
    projection = np.einsum("ndm,nd->nm", fd, unit)  # true slope along the critic's direction, per rollout
    energy_true = np.einsum("nd,nd->n", fd[..., ::2].mean(-1), fd[..., 1::2].mean(-1))  # unbiased |g_true|^2
    edges = np.quantile(norm, [0, .5, .9, .95, .99, 1.0])
    for lo, hi, label in zip(edges[:-1], edges[1:], ("bottom 50%", "50-90%", "90-95%", "95-99%", "top 1%")):
        keep = (norm >= lo) & ((norm < hi) | (hi == edges[-1]))
        boots = []
        for _ in range(1000):
            pick = rng.choice(np.flatnonzero(keep), keep.sum())
            boots.append(projection[pick].mean() / norm[pick].mean())
        print(f"   {label:10s} n {keep.sum():3d}  mean |g_critic| {norm[keep].mean():7.3f}  true slope along it {projection[keep].mean():+7.3f} "
              f"(se {projection[keep].mean(1).std(ddof=1) / np.sqrt(keep.sum()):.3f})  ratio true / claimed {projection[keep].mean() / norm[keep].mean():+.2f} "
              f"[{np.percentile(boots, 2.5):+.2f}, {np.percentile(boots, 97.5):+.2f}]  true |g|^2 {energy_true[keep].mean():7.3f}  critic |g|^2 {(norm[keep] ** 2).mean():8.3f}")
    # Is the true field itself heavy-tailed? Rank on one third of the rollouts, measure energy on the other two.
    thirds = [fd[..., i::3].mean(-1) for i in range(3)]
    rank, energy = (thirds[0] ** 2).sum(1), (thirds[1] * thirds[2]).sum(1)
    order = np.argsort(-rank)
    print(f"   truth: states ranked by |g_true|^2 on one third of the rollouts, energy from the other two: top 1% / 5% / 20% hold "
          f"{energy[order[:5]].sum() / energy.sum():.2f} / {energy[order[:26]].sum() / energy.sum():.2f} / {energy[order[:102]].sum() / energy.sum():.2f} "
          f"(a lower bound on the truth's concentration); critic on the same 512 states: {np.sort(norm ** 2)[-5:].sum() / (norm ** 2).sum():.2f} / "
          f"{np.sort(norm ** 2)[-26:].sum() / (norm ** 2).sum():.2f} / {np.sort(norm ** 2)[-102:].sum() / (norm ** 2).sum():.2f}; "
          f"Spearman(|g_critic|, split-half true energy) {np.corrcoef(np.argsort(np.argsort(norm)), np.argsort(np.argsort(energy_true)))[0, 1]:+.2f}")
    print(f"   pooled calibration: slope of the true projection on |g_critic| through the origin {(projection.mean(1) * norm).sum() / (norm ** 2).sum():+.2f}; "
          f"without the top 1% {(projection.mean(1) * norm)[norm < edges[-2]].sum() / (norm[norm < edges[-2]] ** 2).sum():+.2f}")


for checkpoint_step in c14.STEPS:
    modes(checkpoint_step)
for dataset in ("v23_4M", "v29h8_8M", "v23_12M", "v23_30M"):
    truth_at_spikes(dataset)
