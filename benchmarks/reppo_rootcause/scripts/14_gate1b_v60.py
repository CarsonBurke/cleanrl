"""Gate 1 on the v60 recipe itself, as far as it can be taken without an oracle: the chunk-only arm of 07 D2.

    CUDA_VISIBLE_DEVICES= .venv/bin/python benchmarks/reppo_rootcause/scripts/14_gate1b_v60.py

Data: the five stochastic episodes per v60 checkpoint from 14_states.py (5000 on-policy rows, a third of a rollout).
Truth reference: the same rows' score function with a near-Monte-Carlo return (GAE 0.99 on soft rewards, baseline
U_c = E_pi Q_c), used only in products between DIFFERENT episodes. Estimators and bootstrap as in 14_gate1_hybrid.py
(blocks = episode x time block, episode labels kept). No oracle arm exists for v60, so |V*|^2 and every alignment come
from five independent trajectories: wide intervals, read the signs and the noise shares.
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c14, g1 = _load("common14", "14_common.py"), _load("gate1", "14_gate1_hybrid.py")
common, d1 = c14.common, g1.d1
torch.set_num_threads(2)
BOOTS, SF, REF = 2000, "sf_gae0.95", "sf_gae0.99"


def gae(delta, episode_start, discount):
    out, running = np.zeros_like(delta), 0.0
    for t in range(len(delta) - 1, -1, -1):
        running = delta[t] + discount * (0.0 if (t + 1 == len(delta) or episode_start[t + 1]) else running)
        out[t] = running
    return out


def run(step):
    ck, s = c14.load_checkpoint(step), c14.load_states(step)
    actor, critic, trainer, args = ck.actor, ck.critic, ck.trainer, ck.args
    obs = torch.from_numpy(c14.normalise(s["raw"], s["lane"], ck))
    noise = torch.from_numpy(s["noise"])
    with torch.no_grad():
        mean, std = actor(obs)
        pre = mean + std * noise
        action = torch.tanh(pre)
        logprob = (-0.5 * noise.square() - std.log() - trainer.LOG_SQRT_2PI - trainer.tanh_log_det(pre)).sum(-1).double().numpy()
    temperature = float(ck.checkpoint["temperature"][0])
    u = common.expected_q(critic, obs, mean, std, 16, 7).numpy()
    grad = common.critic_grad(critic, obs, action)[1].double().numpy()
    a, sigma = action.double().numpy(), std.double().numpy()
    start = s["t"] == 0
    last = np.append(start[1:], True)  # final row of each episode: no successor stored, dropped below
    reward = s["reward"] * args.reward_scale
    delta = np.zeros(len(reward))
    delta[:-1] = reward[:-1] - args.gamma * temperature * logprob[1:] + args.gamma * u[1:] - u[:-1]
    keep = ~last
    episode = np.cumsum(start) - 1
    fields = {"critic": grad * (1 - a ** 2)}
    for lam in (0.8, 0.95, 0.99):
        adv = np.zeros(len(reward))
        adv[keep] = gae(delta[keep], start[keep], args.gamma * lam)
        fields[f"sf_gae{lam:g}"] = adv[:, None] * s["noise"].astype(np.float64) / sigma
    fields["crit_norm_pre"], fields["crit_clip_pre"] = g1.reweighted(fields["critic"], "norm"), g1.reweighted(fields["critic"], "clip")
    fields["crit_drop1"] = g1.reweighted(fields["critic"], "drop")
    fields["critic_perm"] = fields["critic"][np.random.default_rng(0).permutation(len(a))]
    fields = {k: v * keep[:, None] for k, v in fields.items()}
    hidden = actor.net[:-1](obs).double().numpy()
    ones = np.ones((len(a), 1))
    spaces = {"linear(108)": (np.concatenate([ones, obs.double().numpy()], 1), 8), "lastlayer(3078)": (np.concatenate([ones, hidden], 1), 8),
              "actor_params(full)": (None, 4)}
    print(f"\n===== v60 checkpoint {step}: {keep.sum()} rows, 5 episodes; advantage sd GAE0.95 {np.std(fields[SF] * sigma / np.maximum(np.abs(s['noise']), 1e-9)):.2f}")
    for name, (features, tb) in spaces.items():
        block = episode * tb + s["t"] * tb // 1000
        lanes = np.repeat(np.arange(5), tb)
        count = np.bincount(block[keep], minlength=len(lanes)).astype(float)
        if features is None:
            shim = type("Shim", (), {"net": actor.net, "__call__": lambda self, x: actor(x)})()
            c = {k: d1.param_contributions(shim, obs, v, block, len(lanes)).astype(np.float64) for k, v in fields.items()}
        else:
            c = {k: d1.contributions(v, features, block, len(lanes)) for k, v in fields.items()}
        rng = np.random.default_rng(11)
        v = np.concatenate([np.ones((1, len(lanes))), rng.multinomial(len(lanes), np.ones(len(lanes)) / len(lanes), size=BOOTS)]).astype(float)
        different = (lanes[:, None] != lanes[None, :]).astype(float)
        vn = v * count
        den, total = g1.quad(vn, different, vn), vn.sum(1)

        def cross(x, y):
            kernel = c[x] @ c[y].T
            return g1.quad(v, 0.5 * (kernel + kernel.T) * different, v) / den

        def gram(x, y):
            kernel = c[x] @ c[y].T
            return (g1.quad(v, kernel, v) - (v ** 2 - v) @ np.diag(kernel)) / total ** 2
        b = cross(REF, REF)
        names = [k for k in fields if k != REF]
        a_ = {k: cross(k, REF) for k in names}
        g = {(k, k): gram(k, k) for k in names} | {(x, SF): gram(x, SF) for x in ("critic", "crit_clip_pre", "crit_norm_pre", "critic_perm")}
        gains = {k: a_[k] / np.sqrt(g[k, k]) for k in names}
        for label, x in (("unit_sum", "critic"), ("unit_sum_clippre", "crit_clip_pre"), ("unit_sum_critperm", "critic_perm")):
            gains[label] = g1.mixed(1 / np.sqrt(g[x, x]), 1 / np.sqrt(g[SF, SF]), a_, g, (x, SF))
        print(f"[{step} | {name}] |V*|^2 {b[0]:.3e} [{np.percentile(b[1:], 2.5):.2e}, {np.percentile(b[1:], 97.5):.2e}] P(>0) {np.mean(b[1:] > 0):.3f}")
        for label, value in gains.items():
            r = g1.summary(value, gains["critic"])
            print(f"    {label:18s} gain {r['point']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] P(>0) {r['p_pos']:.3f} | vs critic {r['diff']:+.4f} "
                  f"[{r['diff_ci'][0]:+.4f}, {r['diff_ci'][1]:+.4f}] P(better) {r['p_better']:.3f}")
        x_cc, x_ss, x_cs = cross("critic", "critic"), cross(SF, SF), cross("critic", SF)
        noise_c, noise_s = g["critic", "critic"] - x_cc, g[SF, SF] - x_ss
        rows = {"noise_share_critic": noise_c / g["critic", "critic"], "noise_share_sf": noise_s / g[SF, SF],
                "rho_noise": (g["critic", SF] - x_cs) / np.sqrt(np.where((noise_c > 0) & (noise_s > 0), noise_c * noise_s, np.nan)),
                "expectation_cos_critic_sf": x_cs / np.sqrt(np.where((x_cc > 0) & (x_ss > 0), x_cc * x_ss, np.nan)),
                "cos_critic_truth (S only)": np.where(b > 0, gains["critic"] / np.sqrt(np.abs(b)), np.nan),
                "cos_sf_truth (S only)": np.where(b > 0, gains[SF] / np.sqrt(np.abs(b)), np.nan)}
        for label, value in rows.items():
            r = g1.summary(value)
            print(f"    {label:28s} {r['point']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  (valid draws {np.mean(np.isfinite(value[1:])):.2f})")


for checkpoint_step in c14.STEPS:
    run(checkpoint_step)
