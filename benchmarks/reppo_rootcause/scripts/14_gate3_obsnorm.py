"""Gate 3: what v60's per-lane cumulative observation statistics do to the networks' inputs.

    CUDA_VISIBLE_DEVICES= .venv/bin/python benchmarks/reppo_rootcause/scripts/14_gate3_obsnorm.py

v60 keeps one independent running mean / variance per env lane (cleanrl/shared/vector_norm.py, the NormalizeObservation
contract) and feeds every row through the statistics of the lane that produced it; actor and critic are shared by all
lanes. Per checkpoint: the lane statistics, the statistics of the upright gait (states of 14_states.py), the range the
gait occupies in normalised units, and what happens to one physical state when it is read through each lane in turn
(network inputs, policy, Q, dQ/da). Then the same lane table for the v23 / v29 oracle chunks.
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("common14", Path(__file__).resolve().parent / "14_common.py")
c14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c14)
common = c14.common
torch.set_num_threads(1)
np.set_printoptions(precision=3, suppress=True, linewidth=250)
FOCUS = (c14.PITCH, c14.PITCH_VELOCITY)


def lane_table(means, sds, gait_mean, gait_sd, label):
    print(f"   {label}: per dimension, lanes' cumulative sd (min / median / max), max/min, lanes' mean spread (max - min) in units of the median "
          f"cumulative sd and of the gait sd, gait sd in normalised units (gait sd / median cumulative sd), gait mean offset in gait sds")
    for d, name in enumerate(c14.OBS_NAMES):
        sd, mu = sds[:, d], means[:, d]
        print(f"      {d:2d} {name:11s} sd {sd.min():7.3f} / {np.median(sd):7.3f} / {sd.max():7.3f}  ratio {sd.max() / sd.min():5.2f}   mean spread {np.ptp(mu) / np.median(sd):5.3f} cum-sd, "
              f"{np.ptp(mu) / gait_sd[d]:5.2f} gait-sd   gait sd {gait_sd[d]:7.3f} = {gait_sd[d] / np.median(sd):5.2f} normalised units   "
              f"offset {(gait_mean[d] - np.median(mu)) / gait_sd[d]:+5.2f}")


def run(step):
    ck, s = c14.load_checkpoint(step), c14.load_states(step)
    raw, upright = s["raw"], np.cos(s["pitch"]) > 0
    gait = raw[upright]
    gait_mean, gait_sd = gait.mean(0), gait.std(0)
    print(f"\n================ v60 checkpoint {step}: {len(raw)} sampled states, {upright.sum()} upright; lane sample counts "
          f"{ck.checkpoint['obs_counts'].min():.0f}-{ck.checkpoint['obs_counts'].max():.0f} (one more iteration moves a lane's statistics by ~{1024 / ck.checkpoint['obs_counts'].mean():.4f} of the gap)")
    lane_table(ck.means, ck.sds, gait_mean, gait_sd, "lane statistics")
    np.savetxt(c14.OUT / f"14_gate3_lane_stats_{step}.tsv", np.concatenate([ck.means, ck.sds], 1), delimiter="\t", fmt="%.6f",
               header="\t".join([f"mean_{n}" for n in c14.OBS_NAMES] + [f"sd_{n}" for n in c14.OBS_NAMES]))
    for d in FOCUS:
        print(f"   {c14.OBS_NAMES[d]}: lane cumulative sd {np.round(ck.sds[:, d], 3).tolist()}; lane cumulative mean {np.round(ck.means[:, d], 3).tolist()}")
        lo, hi = np.quantile(gait[:, d], [.01, .99])
        through = (np.array([lo, hi])[None] - ck.means[:, d, None]) / ck.sds[:, d, None]
        print(f"      gait: mean {gait_mean[d]:+.3f}, sd {gait_sd[d]:.3f}, p1..p99 [{lo:+.3f}, {hi:+.3f}]; that interval in normalised units, by lane: "
              f"narrowest [{through[np.argmin(np.ptp(through, 1))][0]:+.2f}, {through[np.argmin(np.ptp(through, 1))][1]:+.2f}], "
              f"median width {np.median(np.ptp(through, 1)):.2f}, widest [{through[np.argmax(np.ptp(through, 1))][0]:+.2f}, {through[np.argmax(np.ptp(through, 1))][1]:+.2f}]; "
              f"a variable normalised by its own gait statistics would span {(hi - lo) / gait_sd[d]:.2f}")
    ratio = gait_sd[None] / ck.sds
    print(f"   gait sd in normalised units over all 17 dims and 16 lanes: min {ratio.min():.2f} ({c14.OBS_NAMES[np.unravel_index(ratio.argmin(), ratio.shape)[1]]}), "
          f"median {np.median(ratio):.2f}, max {ratio.max():.2f}; dims whose gait sd is under 0.5 normalised units in the median lane: "
          f"{[c14.OBS_NAMES[d] for d in np.flatnonzero(np.median(ratio, 0) < 0.5)]}")

    # ---- one physical state through each lane's statistics
    lanes = np.arange(16)
    x = np.clip((gait[None] - ck.means[:, None]) / ck.sds[:, None], -ck.args.obs_clip, ck.args.obs_clip)  # (lane, state, dim)
    within = x.std(1).mean(0)  # within-gait sd of each normalised input, averaged over lanes
    across = x.std(0)  # (state, dim): sd across lanes of the input for one physical state
    print("   the same physical gait state read through the 16 lanes: sd across lanes of each network input, in units of that input's within-gait sd")
    print("      " + "  ".join(f"{n[:9]:>9s}" for n in c14.OBS_NAMES))
    print("      " + "  ".join(f"{v:9.3f}" for v in np.sqrt((across ** 2).mean(0)) / within) + "   (rms over states)")
    print("      " + "  ".join(f"{v:9.3f}" for v in np.quantile(across, .99, axis=0) / within) + "   (99th percentile state)")
    pair = (x.max(0) - x.min(0)).max(0)
    print("      " + "  ".join(f"{v:9.3f}" for v in pair / within) + "   (largest difference between any two lanes at any state)")
    total = np.sqrt(((x - x.mean(0)) ** 2).sum(-1).mean())  # rms distance of a lane's input from the lane average
    episode = np.flatnonzero((s["t"][1:] > s["t"][:-1]) & upright[1:] & upright[:-1])
    own = c14.normalise(raw, s["lane"], ck).astype(np.float64)
    step_distance = np.sqrt(((own[episode + 1] - own[episode]) ** 2).sum(-1))
    print(f"   17-D input: rms distance of one lane's reading from the lane-average reading {total:.3f}; one environment step moves the input by a median "
          f"{np.median(step_distance):.3f} (p10 {np.quantile(step_distance, .1):.3f}); within-gait rms radius {np.sqrt((within ** 2).sum()):.3f}")

    # ---- what the shared networks output for that one state under each lane
    pick = np.random.default_rng(0).choice(len(gait), 1000, replace=False)
    obs = torch.from_numpy(x[:, pick].astype(np.float32))  # (lane, n, 17)
    with torch.no_grad():
        mean, std = ck.actor(obs)
    mean, std = mean.double(), std.double()
    kl = (torch.log(std[None] / std[:, None]) + (std[:, None] ** 2 + (mean[:, None] - mean[None]) ** 2) / (2 * std[None] ** 2) - 0.5).sum(-1)
    off = ~torch.eye(16, dtype=torch.bool)
    shift = ((mean - mean.mean(0)) / std.mean(0)).pow(2).sum(-1).sqrt()  # (lane, n): mean displacement in policy sds
    print(f"   policy at the same physical state under two different lanes: KL median {kl[off].median():.4f}, mean {kl[off].mean():.4f}, p90 {kl[off].flatten().quantile(.9):.4f}, "
          f"worst lane pair (mean over states) {kl.mean(-1)[off].max():.4f}; v60's per-iteration KL bound is {ck.args.kl_bound}. Mean-action displacement from the "
          f"lane-average policy: rms {shift.pow(2).mean().sqrt():.3f} policy sds (6-D norm), worst lane {shift.pow(2).mean(1).sqrt().max():.3f}; log-std rms spread {std.log().std(0).pow(2).mean().sqrt():.3f}")
    gen = torch.Generator().manual_seed(0)
    fresh = torch.tanh(mean[0].float()[None] + std[0].float()[None] * torch.randn((8,) + mean[0].shape, generator=gen))  # (8, n, 6) actions
    with torch.no_grad():
        q = torch.stack([ck.critic.value(obs[:, :, None].expand(-1, -1, 8, -1), fresh.permute(1, 0, 2)[None].expand(16, -1, -1, -1))]).squeeze(0).double()  # (lane, n, 8)
    advantage = q - q.mean(-1, keepdim=True)
    print(f"   critic at the same physical (state, action) under the 16 lanes: sd of Q across lanes {q.std(0).pow(2).mean().sqrt():.3f} (median {q.std(0).median():.3f}); "
          f"sd of Q across states {q.mean(-1).std(1).mean():.2f}; within-state sd of Q over 8 policy actions {q.std(-1).pow(2).mean().sqrt():.3f} (median {q.std(-1).median():.3f}); "
          f"sd across lanes of the within-state advantage {advantage.std(0).pow(2).mean().sqrt():.3f} "
          f"= {advantage.std(0).pow(2).mean().sqrt() / advantage.mean(0).pow(2).mean().sqrt():.2f} of the lane-average advantage's rms {advantage.mean(0).pow(2).mean().sqrt():.3f}")
    corr = []
    adv = advantage.numpy()
    for i in range(adv.shape[1]):
        c = np.corrcoef(adv[:, i])
        corr.append(c[np.triu_indices(16, 1)].mean())
    print(f"      correlation between two lanes' rankings of the same 8 actions at the same state: median over states {np.median(corr):.3f}, p10 {np.quantile(corr, .1):.3f}")
    action = fresh[0]
    grads = torch.stack([common.critic_grad(ck.critic, obs[lane], action)[1].double() for lane in lanes])  # (lane, n, 6)
    unit = grads / grads.norm(dim=-1, keepdim=True)
    cosine = torch.einsum("lnd,mnd->lmn", unit, unit)[off]
    norms = grads.norm(dim=-1)
    print(f"      dQ/da at the same (state, action) under two lanes: cosine median {cosine.median():.3f}, p10 {cosine.flatten().quantile(.1):.3f}; "
          f"norm max/min over lanes: median state {(norms.max(0).values / norms.min(0).values).median():.2f}, p90 {(norms.max(0).values / norms.min(0).values).quantile(.9):.2f}, "
          f"top-1% states by norm {(norms.max(0).values / norms.min(0).values)[norms.mean(0) >= norms.mean(0).quantile(.99)].median():.2f}")
    # Which lane and which input drive the disagreement?
    reference = np.median(ck.sds, 0)
    for d in FOCUS:
        swapped = x[:, pick].copy()
        swapped[:, :, d] = np.clip((gait[pick, d][None] - np.median(ck.means[:, d])) / reference[d], -ck.args.obs_clip, ck.args.obs_clip)
        with torch.no_grad():
            m2, s2 = ck.actor(torch.from_numpy(swapped.astype(np.float32)))
        m2, s2 = m2.double(), s2.double()
        kl2 = (torch.log(s2[None] / s2[:, None]) + (s2[:, None] ** 2 + (m2[:, None] - m2[None]) ** 2) / (2 * s2[None] ** 2) - 0.5).sum(-1)
        with torch.no_grad():
            q2 = ck.critic.value(torch.from_numpy(swapped.astype(np.float32)), action[None].expand(16, -1, -1)).double()
        print(f"      with {c14.OBS_NAMES[d]} alone read through the median lane's statistics in every lane: between-lane policy KL mean {kl2[off].mean():.4f}, "
              f"median {kl2[off].median():.4f}; sd of Q across lanes {q2.std(0).pow(2).mean().sqrt():.3f} (median {q2.std(0).median():.3f})")
    worst = int(((kl.mean(-1).sum(1) + kl.mean(-1).sum(0)) / 30).argmax())
    keep = off.clone()
    keep[worst], keep[:, worst] = False, False
    typical = torch.arange(16) != worst
    print(f"      without the most deviant lane ({worst}): policy KL between two lanes median {kl[keep].median():.4f}, mean {kl[keep].mean():.4f}; sd of Q across lanes "
          f"{q[typical].std(0).pow(2).mean().sqrt():.3f} (median {q[typical].std(0).median():.3f}); advantage sd across lanes {advantage[typical].std(0).pow(2).mean().sqrt():.3f}")
    probe = obs[0].clone()
    for d in FOCUS + (0, c14.X_VELOCITY, 2):
        moved = probe.clone()
        moved[:, d] += 0.05
        with torch.no_grad():
            m0, s0 = ck.actor(probe)
            m1, _ = ck.actor(moved)
            dq = (ck.critic.value(moved, action) - ck.critic.value(probe, action)).double()
        print(f"      sensitivity to +0.05 normalised units of {c14.OBS_NAMES[d]:9s}: policy mean moves {((m1 - m0) / s0).double().pow(2).sum(-1).sqrt().median():.3f} policy sds (median), "
              f"Q moves by {dq.abs().median():.3f} (median |dQ|)")
    print(f"      between-lane policy KL by lane (mean against the other 15): {np.round(((kl.mean(-1).sum(1) + kl.mean(-1).sum(0)) / 30).numpy(), 4).tolist()}")


def oracle_chunks(name):
    data = torch.load(common.REPO / common.DATASETS[name][0], map_location="cpu", weights_only=False)
    means = data["norm"]["obs_means"].numpy()
    sds = np.sqrt(data["norm"]["obs_variances"].numpy() + data["norm"]["obs_epsilon"])
    raw, state = data["chunk"]["raw_obs"].numpy(), np.asarray(data["chunk"]["state"])
    upright = np.cos(state[:, 2]) > 0
    gait = raw[upright]
    print(f"\n===== {name} (on-policy chunk, {upright.mean():.3f} upright)")
    for d in FOCUS:
        print(f"   {c14.OBS_NAMES[d]:10s} lane cumulative sd min / median / max {sds[:, d].min():.3f} / {np.median(sds[:, d]):.3f} / {sds[:, d].max():.3f} (ratio {sds[:, d].max() / sds[:, d].min():.2f}); "
              f"chunk upright sd {gait[:, d].std():.3f} = {gait[:, d].std() / np.median(sds[:, d]):.2f} normalised units; lane mean spread {np.ptp(means[:, d]) / gait[:, d].std():.2f} gait sds")
    ratio = sds.max(0) / sds.min(0)
    print(f"   all dims: lane sd ratio max/min {np.round(ratio, 2).tolist()}")


for checkpoint_step in c14.STEPS:
    run(checkpoint_step)
for dataset in common.DATASETS:
    oracle_chunks(dataset)
