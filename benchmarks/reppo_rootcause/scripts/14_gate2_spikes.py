"""Gate 2: the spikes of the v60 critic's action gradient, on states sampled from each checkpoint (14_states.py).

    CUDA_VISIBLE_DEVICES= .venv/bin/python benchmarks/reppo_rootcause/scripts/14_gate2_spikes.py

g_act = dQ/da at the sampled action (post-tanh coordinates), g_pre = g_act (1 - a^2) (what reaches the pre-tanh mean;
v60's actor differentiates Q(s, tanh(mu + sigma eps)) through the tanh). Per checkpoint:
  1. distribution of |g|: top-1% / 5% energy share, kurtosis, in both coordinate systems; state vs action share
  2. what the top-1% states are: posture, speed, saturation, policy std, gait phase, contacts, distance from the bulk
  3. dimensions: is a spike one action coordinate?
  4. where in the critic the amplitude is made: decode gain |dQ/dlogits|, head gain, encoder gain, RMSNorm scale;
     finite-difference width of the spike along its own direction
  5. the actor's parameter gradient (v60's actor loss at the first step of an iteration, 2048-row minibatches) without
     the top 1% and with per-state normalisation / p99 clipping: cosine with the unmodified gradient, norm against 0.5
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch
from scipy import stats

spec = importlib.util.spec_from_file_location("common14", Path(__file__).resolve().parent / "14_common.py")
c14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c14)
common = c14.common
torch.set_num_threads(1)
MINIBATCHES, MINIBATCH, FRESH = 48, 2048, 8


def top_share(energy, fraction):
    return float(np.sort(energy)[-max(1, int(round(len(energy) * fraction))):].sum() / energy.sum())


def mahalanobis(x, reference):
    centre, cov = reference.mean(0), np.cov(reference.T) + 1e-6 * np.eye(reference.shape[1])
    delta = x - centre
    return np.sqrt(np.einsum("nd,de,ne->n", delta, np.linalg.inv(cov), delta))


def knn_distance(x, k=10):
    z = (x - x.mean(0)) / x.std(0)
    out = np.empty(len(z))
    for i in range(0, len(z), 500):
        d = ((z[i:i + 500, None] - z[None]) ** 2).sum(-1)
        out[i:i + 500] = np.sqrt(np.partition(d, k, axis=1)[:, k])
    return out


def critic_stages(critic, obs, action, scale):
    """Q, dQ/da and the gradient's amplitude at each stage of the critic."""
    action = action.clone().requires_grad_(True)
    pre1 = critic.encoder[0](torch.cat((obs, action), -1))
    features = critic.encoder[3](critic.encoder[2](critic.encoder[1](pre1)))
    features.retain_grad()
    pre2 = critic.value_head[1](critic.value_head[0](features))
    logits = critic.value_head[4](critic.value_head[3](critic.value_head[2](pre2))) + scale * critic.zero_dist
    logits.retain_grad()
    q = critic.hl.to_scalar(logits)
    q.sum().backward()
    probs = logits.detach().softmax(-1).double()
    support = critic.hl.support.double()
    sd = (probs * support ** 2).sum(-1).sub((probs * support).sum(-1) ** 2).clamp_min(0).sqrt()
    return {"q": q.detach().double().numpy(), "g_act": action.grad.double().numpy(), "dist_sd": sd.numpy(),
            "decode_gain": logits.grad.norm(dim=-1).double().numpy(), "feature_grad": features.grad.norm(dim=-1).double().numpy(),
            "rms1": pre1.detach().square().mean(-1).sqrt().double().numpy(), "rms2": pre2.detach().square().mean(-1).sqrt().double().numpy()}


def gait_phase(raw, upright):
    """Phase of the gait cycle from the two leading principal components of the six joint angles."""
    joints = raw[:, 2:8]
    z = (joints - joints[upright].mean(0)) / joints[upright].std(0)
    values, vectors = np.linalg.eigh(np.cov(z[upright].T))
    pcs = z @ vectors[:, ::-1][:, :2]
    return np.arctan2(pcs[:, 1], pcs[:, 0]), values[::-1][:2].sum() / values.sum()


def run(step):
    ck, s = c14.load_checkpoint(step), c14.load_states(step)
    actor, critic, trainer = ck.actor, ck.critic, ck.trainer
    raw, lane, n = s["raw"], s["lane"], len(s["raw"])
    obs = torch.from_numpy(c14.normalise(raw, lane, ck))
    noise = torch.from_numpy(s["noise"])
    with torch.no_grad():
        mean, std = actor(obs)
        action = torch.tanh(mean + std * noise)
    assert float((action - torch.from_numpy(s["action"])).abs().max()) < 1e-4
    stages = critic_stages(critic, obs, action, trainer.ZERO_DIST_SCALE)
    q_check, g_check = common.critic_grad(critic, obs, action)
    assert np.allclose(stages["g_act"], g_check.double().numpy(), atol=1e-5) and np.allclose(stages["q"], q_check.double().numpy(), atol=1e-3)
    a = action.double().numpy()
    g_act = stages["g_act"]
    g_pre = g_act * (1 - a ** 2)
    sigma = std.double().numpy()
    n_act, n_pre = np.linalg.norm(g_act, axis=1), np.linalg.norm(g_pre, axis=1)
    upright = np.cos(s["pitch"]) > 0
    print(f"\n================ checkpoint {step}: {n} states, upright rows {upright.mean():.3f}, Q mean {stages['q'].mean():.1f}, "
          f"policy std mean {sigma.mean():.3f}, temperature {ck.checkpoint['temperature'][0]:.4f}")

    # ---- 1. distribution
    print("1. distribution of the per-state gradient norm")
    for label, norm in (("post-tanh |dQ/da|", n_act), ("pre-tanh |dQ/da (1-a^2)|", n_pre), ("noise units |sigma dQ/du|", np.linalg.norm(g_pre * sigma, axis=1))):
        e = norm ** 2
        print(f"   {label:28s} p50 {np.median(norm):.3f}  p90 {np.quantile(norm, .9):.3f}  p99 {np.quantile(norm, .99):.3f}  max {norm.max():.2f}  "
              f"max/p50 {norm.max() / np.median(norm):.0f}  top-1% energy {top_share(e, .01):.3f}  top-5% {top_share(e, .05):.3f}  top-1% upright only "
              f"{top_share(e[upright], .01):.3f}  excess kurtosis of |g| {stats.kurtosis(norm):.0f}, of log|g| {stats.kurtosis(np.log(norm)):.2f}")
    print(f"   excess kurtosis of the pooled components: post-tanh {stats.kurtosis(g_act.ravel()):.0f}, pre-tanh {stats.kurtosis(g_pre.ravel()):.0f} (Gaussian 0)")
    gen = torch.Generator().manual_seed(1)
    fresh_pre, fresh_act = [], []
    for _ in range(FRESH):
        fresh = torch.tanh(mean + std * torch.randn(mean.shape, generator=gen))
        grad = common.critic_grad(critic, obs, fresh)[1].double().numpy()
        fresh_act.append(np.linalg.norm(grad, axis=1))
        fresh_pre.append(np.linalg.norm(grad * (1 - fresh.double().numpy() ** 2), axis=1))
    for label, values in (("post-tanh", np.log(np.array(fresh_act))), ("pre-tanh", np.log(np.array(fresh_pre)))):
        between, within = values.mean(0).var(ddof=1), values.var(0, ddof=1).mean()
        print(f"   {label}: share of var(log|g|) that belongs to the state rather than the sampled action (ICC, {FRESH} fresh actions) "
              f"{(between - within / FRESH) / (between - within / FRESH + within):.2f}")
    log_pre = np.log(n_pre)
    same_episode = s["t"][1:] > s["t"][:-1]
    print(f"   lag-1 autocorrelation of log|g_pre| along the trajectory {np.corrcoef(log_pre[1:][same_episode], log_pre[:-1][same_episode])[0, 1]:+.2f}")

    # ---- 2. what the spike states are
    spike = n_pre >= np.quantile(n_pre, 0.99)
    spike_act = n_act >= np.quantile(n_act, 0.99)
    tanh_mean = np.tanh(mean.double().numpy())
    critic_in = np.concatenate([obs.double().numpy(), a], 1)
    phase, phase_share = gait_phase(raw, upright)
    contact = s["contact"]
    geoms = list(s["geoms"])
    feet = contact[:, [geoms.index("bfoot"), geoms.index("ffoot")]]
    other = contact[:, [geoms.index(g) for g in ("torso", "head", "bthigh", "bshin", "fthigh", "fshin")]].any(1)
    features = {
        "cos(pitch)": np.cos(raw[:, c14.PITCH]), "|pitch| (rad)": np.abs(raw[:, c14.PITCH]), "|pitch velocity|": np.abs(raw[:, c14.PITCH_VELOCITY]),
        "x velocity": raw[:, c14.X_VELOCITY], "reward": s["reward"], "step in episode": s["t"].astype(float),
        "max_d |tanh(mean)|": np.abs(tanh_mean).max(1), "mean_d |tanh(mean)|": np.abs(tanh_mean).mean(1),
        "dims with |a| > 0.99": (np.abs(a) > 0.99).sum(1).astype(float), "dims with |a| > 0.9": (np.abs(a) > 0.9).sum(1).astype(float),
        "policy std (mean over dims)": sigma.mean(1), "|noise| of the sample": np.linalg.norm(s["noise"], axis=1),
        "Mahalanobis of normalised obs": mahalanobis(obs.double().numpy(), obs.double().numpy()[upright]),
        "Mahalanobis of critic input (s, a)": mahalanobis(critic_in, critic_in[upright]),
        "10-NN distance of critic input": knn_distance(critic_in),
        "Q": stages["q"], "sd of the HL-Gauss distribution": stages["dist_sd"],
        "feet on floor": feet.sum(1).astype(float), "non-foot geom on floor": other.astype(float),
        "|dQ/da| post-tanh": n_act,
    }
    print(f"2. top-1% states by pre-tanh norm ({spike.sum()} states; {spike[~upright].sum()} inverted; overlap with the post-tanh top-1% "
          f"{(spike & spike_act).sum()}); medians (means for counts), and Spearman of log|g_pre| with each feature over all states")
    for name, value in features.items():
        centre = np.mean if ("dims with" in name or "on floor" in name) else np.median
        rho = stats.spearmanr(log_pre, value)[0]
        print(f"   {name:36s} top-1% {centre(value[spike]):9.3f}   rest {centre(value[~spike]):9.3f}   p-value (Mann-Whitney) "
              f"{stats.mannwhitneyu(value[spike], value[~spike]).pvalue:.1e}   Spearman {rho:+.2f}")
    bins = np.floor((phase + np.pi) / (2 * np.pi) * 8).clip(0, 7).astype(int)
    energy = n_pre ** 2
    resultant = lambda p: float(np.abs(np.exp(1j * p).mean()))
    print(f"   gait phase (PC1/PC2 of joint angles hold {phase_share:.2f} of their variance): resultant length of the top-1% phases "
          f"{resultant(phase[spike]):.2f} (all rows {resultant(phase):.2f}; 0 = uniform, 1 = one phase)")
    print("   phase octant:            " + "  ".join(f"{b:5d}" for b in range(8)))
    print("   share of rows            " + "  ".join(f"{np.mean(bins == b):5.2f}" for b in range(8)))
    print("   share of top-1% states   " + "  ".join(f"{np.mean(bins[spike] == b):5.2f}" for b in range(8)))
    print("   share of gradient energy " + "  ".join(f"{energy[bins == b].sum() / energy.sum():5.2f}" for b in range(8)))
    print("   median |g_pre|           " + "  ".join(f"{np.median(n_pre[bins == b]):5.2f}" for b in range(8)))
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], spike.astype(int), [0]])))[:2 * (len(np.flatnonzero(np.diff(np.concatenate([[0], spike.astype(int), [0]])))) // 2)].reshape(-1, 2), axis=1).ravel()
    print(f"   top-1% states come in runs of consecutive steps: {len(runs)} runs, lengths median {np.median(runs):.0f}, max {runs.max()}; "
          f"episodes hit {sorted(set(lane[spike].tolist()))} (lanes), steps {np.sort(s['t'][spike])[:12].tolist()} ...")

    # ---- 3. dimensions
    print("3. action dimensions (bthigh, bshin, bfoot, fthigh, fshin, ffoot)")
    for label, g in (("pre-tanh", g_pre), ("post-tanh", g_act)):
        e = g ** 2
        participation = e.sum(1) ** 2 / (e ** 2).sum(1)
        which = np.bincount(e[spike].argmax(1), minlength=6) / spike.sum()
        print(f"   {label:9s} largest coordinate's share of |g|^2: top-1% {np.median(e[spike].max(1) / e[spike].sum(1)):.2f}, rest {np.median(e[~spike].max(1) / e[~spike].sum(1)):.2f}; "
              f"participation ratio (1-6) top-1% {np.median(participation[spike]):.2f}, rest {np.median(participation[~spike]):.2f}; largest coordinate among top-1%: "
              f"{np.round(which, 2).tolist()}; share of all energy per coordinate {np.round(e.sum(0) / e.sum(), 2).tolist()}")
    pooled = (g_act[spike] ** 2).sum(0) / (g_act[spike] ** 2).sum()
    print(f"   per-coordinate top-1% energy share (each coordinate ranked on its own), pre-tanh: {[round(top_share(g_pre[:, d] ** 2, .01), 2) for d in range(6)]}; "
          f"post-tanh energy of the spike states by coordinate {np.round(pooled, 2).tolist()}")

    # ---- 3b. pre- versus post-tanh, saturation
    print("3b. tanh coordinates and saturation")
    abs_a = np.abs(a)
    for lo, hi in ((0, .5), (.5, .9), (.9, .99), (.99, 1.01)):
        m = (abs_a >= lo) & (abs_a < hi)
        print(f"   components with |a| in [{lo}, {hi}): share {m.mean():.3f}; rms dQ/da {np.sqrt((g_act[m] ** 2).mean()):.3f}; share of post-tanh energy "
              f"{(g_act[m] ** 2).sum() / (g_act ** 2).sum():.3f}; share of pre-tanh energy {(g_pre[m] ** 2).sum() / (g_pre ** 2).sum():.3f}; "
              f"median |dQ/da| {np.median(np.abs(g_act[m])):.3f}")
    print(f"   Spearman(|dQ/da_d|, |a_d|) over components {stats.spearmanr(np.abs(g_act).ravel(), abs_a.ravel())[0]:+.2f}; "
          f"Spearman(|g_act|, |g_pre|) over states {stats.spearmanr(n_act, n_pre)[0]:+.2f}; median |g_pre| / |g_act| {np.median(n_pre / n_act):.2f} "
          f"(top-1% by post-tanh norm: {np.median((n_pre / n_act)[spike_act]):.2f})")
    star = torch.tanh(mean)
    g_star = common.critic_grad(critic, obs, star)[1].double().numpy()
    n_star = np.linalg.norm(g_star * (1 - star.double().numpy() ** 2), axis=1)
    print(f"   at the mean action a* = tanh(mu): top-1% energy share pre-tanh {top_share(n_star ** 2, .01):.3f}; Spearman with the sampled-action norm {stats.spearmanr(n_star, n_pre)[0]:+.2f}")

    # ---- 4. where the amplitude is made
    print("4. where in the critic the amplitude comes from (Spearman with log|g_act| over states; medians top-1% by post-tanh norm vs rest)")
    encoder_gain = n_act / stages["feature_grad"]
    head_gain = stages["feature_grad"] / stages["decode_gain"]
    for name, value in (("decode gain |dQ/dlogits|", stages["decode_gain"]), ("head gain |dQ/dfeat| / |dQ/dlogits|", head_gain),
                        ("encoder gain |dQ/da| / |dQ/dfeat|", encoder_gain), ("sd of HL-Gauss distribution", stages["dist_sd"]),
                        ("1 / rms of encoder pre-norm", 1 / stages["rms1"]), ("1 / rms of head pre-norm", 1 / stages["rms2"])):
        print(f"   {name:38s} Spearman {stats.spearmanr(np.log(n_act), value)[0]:+.2f}   top-1% {np.median(value[spike_act]):9.4f}   rest {np.median(value[~spike_act]):9.4f}   "
              f"ratio {np.median(value[spike_act]) / np.median(value[~spike_act]):6.2f}   p1/p99 over states {np.quantile(value, .01):.4f} / {np.quantile(value, .99):.4f}")
    direction = torch.from_numpy(g_act / n_act[:, None]).float()
    print("   finite-difference slope along the gradient's own direction, relative to the analytic norm (1 = locally linear):")
    for h in (0.01, 0.05, 0.2, 0.5):
        with torch.no_grad():
            hi_a, lo_a = (action + h * direction).clamp(-1, 1), (action - h * direction).clamp(-1, 1)
            span = (hi_a - lo_a).norm(dim=-1).double().numpy()
            fd = (critic.value(obs, hi_a) - critic.value(obs, lo_a)).double().numpy() / span
        ratio = fd / n_act
        print(f"      h {h:4.2f}: top-1% median {np.median(ratio[spike_act]):.2f} (p10 {np.quantile(ratio[spike_act], .1):.2f}); rest median {np.median(ratio[~spike_act]):.2f}; "
              f"top-1% energy share of the FD slope {top_share(fd ** 2, .01):.3f}")

    # ---- 5. the actor's parameter gradient
    print("5. actor parameter gradient (v60 actor loss, first step of an iteration: every state inside the KL gate)")
    params = list(actor.net.parameters())
    for parameter in params:
        parameter.requires_grad_(True)
    temperature = actor.temperature().detach()
    g_act_t = torch.from_numpy(g_act).float()

    def gradient(rows, weights=None, entropy=True, true_critic=False):
        mu, sd = actor(obs[rows])
        pre = mu + sd * noise[rows]
        act = torch.tanh(pre)
        logprob = -0.5 * noise[rows].square() - sd.log() - trainer.LOG_SQRT_2PI - trainer.tanh_log_det(pre)
        if true_critic:
            q = critic.value(obs[rows], act)
        else:  # first-order surrogate: identical parameter gradient to Q for weights = 1
            q = (g_act_t[rows] * weights[rows, None] * act).sum(-1)
        objective = -q + (temperature * logprob).sum(-1) * float(entropy)
        return torch.cat([g.flatten() for g in torch.autograd.grad(objective.mean(), params)]).double()

    def scale(norm, mode):
        if mode == "norm":
            return np.sqrt(np.mean(norm ** 2)) / norm
        if mode == "clip":
            return np.minimum(1.0, np.quantile(norm, 0.99) / norm)
        return (norm < np.quantile(norm, 0.99)).astype(float)
    variants = {"unmodified": np.ones(n), "without top 1% (pre)": scale(n_pre, "drop"), "top 1% only (pre)": 1 - scale(n_pre, "drop"),
                "(e) unit per state, pre": scale(n_pre, "norm"), "(e) unit per state, post": scale(n_act, "norm"),
                "(f) clip p99, pre": scale(n_pre, "clip"), "(f) clip p99, post": scale(n_act, "clip"),
                "clip p90, pre": np.minimum(1.0, np.quantile(n_pre, .9) / n_pre)}
    variants = {k: torch.from_numpy(v).float() for k, v in variants.items()}
    everything = torch.arange(n)
    exact = gradient(everything, true_critic=True)
    full = {k: gradient(everything, w) for k, w in variants.items()}
    pathwise = gradient(everything, variants["unmodified"], entropy=False)
    cosine = lambda x, y: float(x @ y / (x.norm() * y.norm()))
    print(f"   surrogate check: cos(first-order surrogate, exact loss gradient) {cosine(full['unmodified'], exact):.6f}; full-batch norm {exact.norm():.3f}; "
          f"pathwise part alone {pathwise.norm():.3f} (cos with total {cosine(pathwise, exact):.3f}), entropy part {(exact - pathwise).norm():.3f}")
    rng = np.random.default_rng(0)
    batches = [torch.from_numpy(rng.integers(0, n, MINIBATCH)) for _ in range(MINIBATCHES)]
    per = {k: torch.stack([gradient(rows, w) for rows in batches]) for k, w in variants.items()}
    reference = per["unmodified"]

    def pairwise(x):
        unit = x / x.norm(dim=1, keepdim=True)
        gram = unit @ unit.T
        return float((gram.sum() - gram.diag().sum()) / (len(x) * (len(x) - 1)))
    print(f"   {MINIBATCHES} minibatches of {MINIBATCH} rows drawn with replacement, as the trainer does; clip threshold {ck.args.max_grad_norm}")
    print(f"   {'variant':28s} {'full-batch norm':>15s} {'cos w/ unmod (full)':>20s} {'minibatch norm p50 [p10, p90]':>31s} {'norm / 0.5':>10s} {'clipped':>8s} "
          f"{'cos w/ unmod (minibatch)':>25s} {'cos between minibatches':>24s}")
    for name, grads in per.items():
        norms = grads.norm(dim=1).numpy()
        cos_mb = np.array([cosine(grads[i], reference[i]) for i in range(MINIBATCHES)])
        print(f"   {name:28s} {full[name].norm():15.3f} {cosine(full[name], full['unmodified']):20.3f} "
              f"{np.median(norms):12.3f} [{np.quantile(norms, .1):.3f}, {np.quantile(norms, .9):.3f}]".ljust(100)
              + f" {np.median(norms) / ck.args.max_grad_norm:10.2f} {np.mean(norms > ck.args.max_grad_norm):8.2f} {np.median(cos_mb):18.3f} (p10 {np.quantile(cos_mb, .1):.2f}) {pairwise(grads):24.3f}")
    for parameter in params:
        parameter.requires_grad_(False)


for checkpoint_step in c14.STEPS:
    run(checkpoint_step)
