"""Rank policy-update rules by held-out improvement per KL on real frozen-policy data.

    python scripts/awr_update_bench.py SNAPSHOT.pt [SNAPSHOT.pt ...] [--out results.json]

Each snapshot comes from scripts/awr_capture_policy.py: a trainer's policy, critic and
normalizer statistics, taken as an update began. For each snapshot the bench:
1. Collects `--chunks` rollouts of 16 envs x 1024 steps with that frozen policy, using the
   trainer's own env, host actor, sampler, normalizers and truncation bootstraps. GAE is
   computed per chunk exactly as the trainer does, so every chunk is a genuine training batch.
2. Uses the first half of the chunks as independent training batches, and the second
   half (disjoint in time) as the held-out evaluation set.
3. Replays the snapshot's own rule exactly as its trainer would (real Adam moments, lr, clip);
   its KL must match the trainer's logged KL, which validates the bench.
4. Runs every rule's real update on each training batch: 10 full-batch Adam epochs in the
   trainer's real per-parameter geometry (exp_avg_sq rescaled to the rule's gradient power, no
   inherited momentum), the trainer's nGPT matrix normalization and the rule's gradient clip,
   swept over lr multipliers so each rule traces a dJ-vs-KL frontier.
5. Reports, on the held-out set:
   - dJ = E[A (r - 1)]: the importance-sampled improvement that every surrogate targets, with
     GAE advantages and with lambda=1 Monte Carlo ones (critic bias only through bootstraps);
   - dJ_train: the same quantity in-sample;
   - the exact Beta KL(pi_old || pi_new);
   - the share of that KL spent on concentration rather than location;
   - the entropy change.

Rules are compared on the dJ-vs-KL frontier, so step-size choices don't confound the direction.
The dJ_train - dJ gap measures how much of an update fits noise. Run it through mlq (CUDA, compiled).
"""

import argparse
import copy
import importlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_MODULE = "cleanrl.awr.ppo_continuous_action_awr_chi2_ce_v10"


# ---------------------------------------------------------------- rules
# Each rule maps (policy stats, batch, prepared targets, config) to an actor loss.
# Targets are prepared per training batch (z-scoring is per batch, as in the trainers).

def logratio_of(agent, batch):
    alpha, beta = agent.policy(batch["obs"])
    newlogprob = (Beta(alpha, beta, validate_args=False).log_prob(batch["native"])
                  - agent.log_action_scale).sum(-1)
    return alpha, beta, newlogprob - batch["logprob"]


def zscore(advantages):
    return (advantages - advantages.mean()) / advantages.std()


def shape(z, kind, c):
    """Per-sample influence psi(z): how far one noisy advantage may pull its own ratio."""
    if kind == "lin":
        return z
    if kind == "huber":
        return z.clamp(-c, c)
    if kind == "neg_huber":
        return z.clamp_min(-c)
    if kind == "tanh":
        return c * torch.tanh(z / c)
    if kind == "pos_tanh":
        return torch.where(z > 0, c * torch.tanh(z / c), z)
    if kind == "pos_huber":
        return z.clamp_max(c)
    if kind == "rank":
        # Gaussianized batch ranks: keeps the ordering, discards the heavy-tailed magnitudes.
        ranks = z.argsort().argsort().to(z.dtype)
        return math.sqrt(2.0) * torch.special.erfinv(2.0 * (ranks + 0.5) / z.numel() - 1.0)
    raise ValueError(kind)


def chi2_targets(psi, eta):
    """(1 + (psi - lambda)/eta)_+ with lambda bisected so the mean target is 1 (v10 when psi = z)."""
    psi = psi - psi.mean()
    low, high = torch.zeros((), device=psi.device), psi.max() + eta
    for _ in range(40):
        middle = 0.5 * (low + high)
        above = F.relu(1.0 + (psi - middle) / eta).mean() > 1.0
        low, high = torch.where(above, middle, low), torch.where(above, high, middle)
    return F.relu(1.0 + (psi - 0.5 * (low + high)) / eta)


def ppo_loss(agent, batch, targets, config):
    _, _, logratio = logratio_of(agent, batch)
    ratio = logratio.exp()
    clipped = torch.clamp(ratio, 1 - config["clip"], 1 + config["clip_upper"])
    return torch.max(-targets * ratio, -targets * clipped).mean()


def ce_loss(agent, batch, targets, config):
    # Power-divergence variants (d/dlogratio = (r - r*) r^-beta) were tried here: any beta > 0 trades
    # v10's overshoot runaway for an undershoot one (r* r^-beta at r -> 0) and went NaN at lr x1.5.
    _, _, logratio = logratio_of(agent, batch)
    return (logratio.exp() - targets * logratio).mean()


def logreg_loss(agent, batch, targets, config):
    # v6's M-step: least squares in log-ratio space, per-state centered by the exact Beta KL
    # (E_old[logratio + KL | s] = 0), so every sample has a finite fixed point and a two-sided,
    # residual-proportional pull, unlike the generalized-KL fit's (r - r*) (<= 1 down, r* - 1 up).
    alpha, beta, logratio = logratio_of(agent, batch)
    centered = logratio + beta_kl(batch["alpha"], batch["beta"], alpha, beta).sum(-1)
    residual = centered - targets
    delta = config.get("delta")
    if delta is None:
        return 0.5 * residual.square().mean()
    # Huber in log space: residual-proportional pull within delta of the target, constant beyond,
    # so no heavy-tailed advantage outweighs another by more than its target's reach.
    return F.huber_loss(centered, targets, delta=delta)


def reach_loss(agent, batch, targets, config):
    # |z|-weighted log-space regression onto a reachable target eps * sign(z): the initial gradient is
    # eps * E[z grad log pi] (the linear policy gradient, negative magnitudes intact), and each sample's
    # pull ends, and reverses, once its per-state-centered log-ratio reaches +-eps.
    alpha, beta, logratio = logratio_of(agent, batch)
    centered = logratio + beta_kl(batch["alpha"], batch["beta"], alpha, beta).sum(-1)
    weight, target = targets
    return 0.5 * (weight * (centered - target).square()).mean()


def prepare_reach(advantages, config, batch):
    z = zscore(advantages)
    weight = z.abs()
    return weight / weight.mean(), config["eps"] * torch.sign(z)


def wce_loss(agent, batch, targets, config):
    # Weighted generalized-KL projection onto the KL-optimal target r* = exp(z/eta): fixed point r = r*
    # per sample, finite and positive for every advantage.
    _, _, logratio = logratio_of(agent, batch)
    weight, target = targets
    if config.get("flat") == "down":
        # Only negative-advantage samples switch off past their target: removes the |z|-weighted anchor
        # on the bad tail while positives keep their restoring force.
        logratio = torch.where(target < 1.0, torch.maximum(logratio, target.log()), logratio)
    elif config.get("flat"):
        # One-sided: once a sample passes its target in its own direction its force is zero, as past
        # PPO's clip, instead of restoring against the samples still pulling the shared parameters.
        # The CE force w (r* - r) already decays to zero at r*, so the switch-off is continuous.
        bound = target.log()
        logratio = torch.where(target >= 1.0, torch.minimum(logratio, bound), torch.maximum(logratio, bound))
    return (weight * (logratio.exp() - target * logratio)).mean()


def prepare_wce(advantages, config, batch):
    # w = x / expm1(x) makes the gradient at theta_old exactly -(z/eta) grad log pi: the linear policy
    # gradient with every negative magnitude intact, which r* >= 0 alone caps at 1 per sample.
    x = zscore(advantages) / config["eta"]
    weight = torch.where(x.abs() < 1e-4, 1.0 - 0.5 * x, x / torch.expm1(x))
    return weight, x.exp()


def prepare_kce(advantages, config, batch):
    # The weighted CE sum w (r - r* log r) with w = k - psi, r* = k / (k - psi) equals, up to a constant,
    # -E[psi (r - 1)] + k E_old[r - 1 - log r]: the importance-sampled improvement under a uniform
    # reverse-KL penalty, and the only CE weighting whose implied penalty is sample-independent.
    # Negative psi: force psi at theta_old, reach log(1 + |psi|/k). Positive psi needs psi < k (pole).
    psi = shape(zscore(advantages), config["psi"], config.get("c", 0.0))
    psi = psi - psi.mean()
    # Optional |psi|-proportional penalty: k_i = k + |psi|/eps is PPO's structure in CE form (reach
    # log r* -> -log(1 -+ eps) independent of |A|, force still psi); no pole for eps < 1.
    k = config["k"] + psi.abs() / config["eps"] if "eps" in config else config["k"]
    weight = k - psi
    if bool((weight <= 0).any()):
        raise ValueError(f"kce needs psi < k: max psi {psi.max().item():.3g} >= k {config['k']}")
    return weight, k / weight


def prepare_sce(advantages, config, batch):
    # The eps -> 1, k -> 0 edge of kce with separate reaches: |z|-weighted CE onto a sign target,
    # r* = e^up for z > 0 and e^-down for z < 0, w = z / (r* - 1). Force z at theta_old; every sample's
    # fixed point is independent of |A|, as under PPO's clip, but restoring rather than flat.
    z = zscore(advantages)
    up = z > 0
    target = torch.where(up, math.exp(config["up"]), math.exp(-config["down"]))
    weight = z.abs() / torch.where(up, math.expm1(config["up"]), -math.expm1(-config["down"]))
    return weight, target


def prepare_gce(advantages, config, batch):
    # Separable (weight, reach) map: force sign(z) |z|^q at theta_old, log-ratio fixed point
    # +-R tanh(|z|/c) (c = 0: sce's constant reach). One noisy sample cannot justify an unbounded move,
    # and a near-zero z cannot justify a full one.
    z = zscore(advantages)
    up = z > 0
    size = z.abs()
    c = config.get("c", 0.0)
    reach = torch.where(up, config["up"], -config["down"]) * (torch.tanh(size / c) if c > 0 else 1.0)
    weight = size.pow(config.get("q", 1.0)) / torch.expm1(reach).abs().clamp_min(1e-6)
    return weight, reach.exp()


def prepare_hce(advantages, config, batch):
    # Sign reach on the good side (fixed point independent of |A|: positive outliers cannot carve noise),
    # exp reach on the bad side (log r* = z/eta: the anchor k = w r* -> 0 on the catastrophic tail, as in
    # v14/v15, so the bad tail stops braking concentration). Force z on both sides.
    z = zscore(advantages)
    up = z > 0
    x = torch.where(up, config["up"], z / config["eta"])
    # Negatives: z / expm1(z/eta) = eta * x / expm1(x), in its stable form near x = 0.
    down_weight = config["eta"] * torch.where(x.abs() < 1e-4, 1.0 - 0.5 * x, x / torch.expm1(x))
    return torch.where(up, z / math.expm1(config["up"]), down_weight), x.exp()


def prepare_rwce(advantages, config, batch):
    # v15's exp target on Gaussianized ranks: the order survives, the heavy-tailed magnitudes do not.
    x = shape(zscore(advantages), "rank", 0.0) / config["eta"]
    weight = torch.where(x.abs() < 1e-4, 1.0 - 0.5 * x, x / torch.expm1(x))
    return weight, x.exp()


def beta_leverage(alpha, beta, native):
    """s' F^-1 s summed over action dimensions: the score's squared norm in the Beta's own Fisher metric.
    Moving log pi(a|s) by delta costs at least delta^2 / (2 lev) of KL; E_old[lev] = 2 per dimension."""
    total = alpha + beta
    common = torch.digamma(total)
    score_a = torch.log(native) - torch.digamma(alpha) + common
    score_b = torch.log1p(-native) - torch.digamma(beta) + common
    cross = -torch.special.polygamma(1, total)
    fisher_a = torch.special.polygamma(1, alpha) + cross
    fisher_b = torch.special.polygamma(1, beta) + cross
    det = fisher_a * fisher_b - cross.square()
    return ((fisher_b * score_a.square() - 2.0 * cross * score_a * score_b + fisher_a * score_b.square())
            / det).sum(-1)


def prepare_lce(advantages, config, batch):
    # sce with each sample's reach scaled by (lev / mean lev)^(p/2): p = 1 gives every destination the
    # same minimum KL cost (tail actions, cheap to move, go further); p = -1 discounts them instead.
    _, target = prepare_sce(advantages, config, batch)
    leverage = beta_leverage(batch["alpha"], batch["beta"], batch["native"])
    reach = target.log() * (leverage / leverage.mean()).pow(0.5 * config["p"])
    return zscore(advantages).abs() / torch.expm1(reach).abs().clamp_min(1e-6), reach.exp()


def prepare_logreg(advantages, config, batch):
    psi = shape(zscore(advantages), config["psi"], config.get("c", 0.0))
    return (psi - psi.mean()) / config["eta"]


def prepare_ppo(advantages, config, batch):
    if config["norm_adv"]:
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages


def prepare_ppo_chi2(advantages, config, batch):
    # The chi^2 target's per-sample pull (r* - 1), in advantage units, under PPO's clipped fit:
    # separates what the target discards (the truncated negative tail) from how CE fits it.
    # eta * (r* - 1) * std = std * max(z - lambda, -eta): slope 1 in the bulk, as ppo_chi2_weights_v1 trains.
    return (chi2_targets(zscore(advantages), config["eta"]) - 1.0) * (config["eta"] * advantages.std())


def prepare_ce(advantages, config, batch):
    return chi2_targets(shape(zscore(advantages), config["psi"], config.get("c", 0.0)), config["eta"])


@dataclass
class Rule:
    name: str
    loss: callable
    prepare: callable
    config: dict
    lr_mult: float = 1.0
    max_grad_norm: float = float("inf")
    label: str = field(default="")

    @property
    def family(self):
        return (self.name, tuple(sorted(self.config.items())))

    @property
    def advantage_key(self):
        return f"advantages_{self.config.get('lam', 0.95):g}"

    def __post_init__(self):
        if not self.label:
            knobs = ",".join(f"{key}={value}" for key, value in self.config.items())
            self.label = f"{self.name}[{knobs}] lr{self.lr_mult:g}"


# The 50M PPO reference (--no-norm-adv --clip-coef-upper 0.2); a PPO snapshot's own args override it.
PPO_REFERENCE = {"clip": 0.2, "clip_upper": 0.2, "norm_adv": False, "max_grad_norm": 0.5}
# Training-target GAE lambdas collected per chunk; evaluation always uses 0.95 and 1.0 (Monte Carlo).
TARGET_LAMBDAS = (0.8, 0.9, 0.95)
# CE families: (psi, c, eta[, lam]). v10 is ("lin", -, eta). "reach" etas make the target's own
# chi^2 ~ what one update realizes, so the fit saturates per sample instead of pushing one direction.
CE_FAMILIES = [
    {"psi": "lin", "eta": 1.0}, {"psi": "lin", "eta": 0.6},
    {"psi": "lin", "eta": 2.0}, {"psi": "lin", "eta": 4.0},
    {"psi": "neg_huber", "c": 1.0, "eta": 2.0}, {"psi": "huber", "c": 2.0, "eta": 2.0},
    {"psi": "huber", "c": 1.0, "eta": 1.0}, {"psi": "huber", "c": 1.0, "eta": 2.0},
    {"psi": "tanh", "c": 1.0, "eta": 1.0},
    {"psi": "rank", "eta": 1.0}, {"psi": "rank", "eta": 2.0},
    {"psi": "lin", "eta": 1.0, "lam": 0.9}, {"psi": "lin", "eta": 1.0, "lam": 0.8},
    # M-step conditioning: Adam's second moment tracking within the update (b2) instead of lagging.
    *({"psi": "lin", "eta": 1.0, "b2": b2} for b2 in (0.99, 0.9)),
    {"psi": "lin", "eta": 0.6, "b2": 0.9}, {"psi": "lin", "eta": 1.0, "lam": 0.8, "b2": 0.9},
    # Bounded per-sample pull: r* capped near 1 + (c - lambda)/eta instead of v10's ~10-17.
    *({"psi": psi, "c": c, "eta": 0.6, "b2": 0.9} for psi, c in (("tanh", 1.0), ("tanh", 2.0), ("huber", 1.0))),
    {"psi": "rank", "eta": 0.6, "b2": 0.9},
]


# Log-space fits: v6 (lin, eta 2) and bounded-influence scores at several target scales.
LOGREG_FAMILIES = [
    {"psi": "lin", "eta": 2.0}, {"psi": "lin", "eta": 1.0},
    *({"psi": "tanh", "c": c, "eta": eta} for c, eta in ((2.0, 1.0), (2.0, 0.5), (1.0, 0.5), (1.0, 0.25))),
    # The KL-optimal log target A/eta - log Z(s), untruncated, under a bounded-influence fit.
    *({"psi": "lin", "eta": eta, "delta": delta} for eta in (2.0, 4.0, 8.0) for delta in (0.02, 0.1)),
]


# Uniform reverse-KL penalty k; positive advantages soft-capped at c < k so no sample hits the pole.
KCE_FAMILIES = [
    *({"psi": "pos_tanh", "c": c, "k": k, "b2": 0.9} for c, k in ((2.0, 3.0), (2.0, 4.0), (2.0, 6.0),
                                                              (3.0, 4.0), (3.0, 6.0), (3.0, 9.0))),
    {"psi": "tanh", "c": 3.0, "k": 6.0, "b2": 0.9}, {"psi": "pos_huber", "c": 3.0, "k": 6.0, "b2": 0.9},
    {"psi": "pos_tanh", "c": 3.0, "k": 6.0, "b2": 0.999},
    *({"psi": "lin", "k": k, "eps": eps, "b2": 0.9} for eps in (0.1, 0.2, 0.4) for k in (0.01, 1.0, 3.0)),
]


def default_rules(args, lr_mults):
    ppo = dict(PPO_REFERENCE)
    if hasattr(args, "clip_coef_upper"):
        ppo = {"clip": args.clip_coef, "clip_upper": args.clip_coef_upper,
               "norm_adv": args.norm_adv, "max_grad_norm": args.max_grad_norm}
    ppo_clip = ppo.pop("max_grad_norm")
    awr_clip = float("inf") if hasattr(args, "clip_coef_upper") else float(args.max_grad_norm)
    families = [("ppo", ppo_loss, prepare_ppo, ppo, ppo_clip),
                ("ppo", ppo_loss, prepare_ppo, {**ppo, "lam": 0.9}, ppo_clip),
                ("ppo", ppo_loss, prepare_ppo, {**ppo, "b2": 0.9}, ppo_clip)]
    families += [("ppo_chi2", ppo_loss, prepare_ppo_chi2, {**ppo, "eta": eta}, ppo_clip) for eta in (1.0, 0.6)]
    families += [("ce", ce_loss, prepare_ce, config, awr_clip) for config in CE_FAMILIES]
    families += [("logreg", logreg_loss, prepare_logreg, config, awr_clip) for config in LOGREG_FAMILIES]
    families += [("reach", reach_loss, prepare_reach, {"eps": eps}, awr_clip) for eps in (0.1, 0.2, 0.3)]
    families += [("wce", wce_loss, prepare_wce, {"eta": eta, "b2": b2}, awr_clip)
                 for eta in (2.0, 4.0, 8.0) for b2 in (0.999, 0.9)]
    families += [("kce", wce_loss, prepare_kce, config, awr_clip) for config in KCE_FAMILIES]
    families += [("sce", wce_loss, prepare_sce, {"up": up, "down": down, "b2": 0.9}, awr_clip)
                 for up in (0.4, 0.7, 1.0) for down in (0.25, 0.4, 0.6)]
    families += [("gce", wce_loss, prepare_gce, {"up": up, "down": down, "c": c, "q": q, "b2": 0.9}, awr_clip)
                 for up, down in ((1.0, 0.6), (1.5, 0.9)) for c, q in ((0.5, 1.0), (1.0, 1.0), (2.0, 1.0),
                                                                    (0.0, 0.75), (0.0, 0.5), (1.0, 0.75))]
    families += [("lce", wce_loss, prepare_lce, {"up": 1.0, "down": 0.6, "p": p, "b2": 0.9}, awr_clip)
                 for p in (1.0, 0.5, -0.5, -1.0)]
    families += [("sce", wce_loss, prepare_sce, {"up": 1.0, "down": 0.6, "flat": "down", "b2": 0.9}, awr_clip)]
    families += [("hce", wce_loss, prepare_hce, {"up": up, "eta": eta, "b2": 0.9}, awr_clip)
                 for up in (0.7, 1.0) for eta in (1.0, 2.0)]
    families += [("rwce", wce_loss, prepare_rwce, {"eta": eta, "b2": 0.9}, awr_clip) for eta in (1.0, 2.0)]
    # Past the first grid's edge, where larger reach kept helping.
    families += [("sce", wce_loss, prepare_sce, {"up": up, "down": down, "b2": 0.9}, awr_clip)
                 for up, down in ((1.0, 0.9), (1.5, 0.6), (1.5, 0.9), (2.0, 0.9), (1.5, 1.2), (2.0, 1.5))]
    families += [("sce", wce_loss, prepare_sce, {"up": up, "down": down, "flat": True, "b2": 0.9}, awr_clip)
                 for up, down in ((0.2, 0.2), (0.3, 0.3), (0.4, 0.25), (0.4, 0.4), (0.7, 0.4))]
    families += [("wce", wce_loss, prepare_wce, {"eta": eta, "flat": True, "b2": 0.9}, awr_clip) for eta in (2.0, 4.0)]
    return [Rule(name, loss, prepare, config, lr, clip)
            for name, loss, prepare, config, clip in families for lr in lr_mults]


def native_family(snapshot, rules):
    """The (name, config) of the rule that trained this snapshot: its replay validates the bench."""
    args = snapshot_args(snapshot)
    if snapshot["trainer_module"].endswith("ppo_sepclip_control_v1"):
        return next(rule.family for rule in rules if rule.name == "ppo")
    if snapshot["trainer_module"].endswith("awr_chi2_ce_v10"):
        family = ("ce", tuple(sorted({"psi": "lin", "eta": float(args.advantage_temperature)}.items())))
        if all(rule.family != family for rule in rules):
            raise ValueError(f"advantage_temperature={args.advantage_temperature} is not among the bench's etas")
        return family
    if snapshot["trainer_module"].endswith("awr_sign_reach_ce_v16"):
        family = ("sce", tuple(sorted({"up": float(args.up_reach), "down": float(args.down_reach),
                                       "b2": float(args.actor_adam_beta2)}.items())))
        if all(rule.family != family for rule in rules):
            raise ValueError(f"v16 reach ({args.up_reach}, {args.down_reach}) is not among the bench's sce rules")
        return family
    raise ValueError(f"no native rule for {snapshot['trainer_module']}")


# ---------------------------------------------------------------- data

def restore_normalizers(obs_norm, rew_norm, snapshot, statistics_only=False):
    # The native kernels hold these arrays' addresses: copy in place, never rebind.
    for name, value in snapshot["obs_norm"].items():
        getattr(obs_norm, name)[...] = value
    for name, value in snapshot["rew_norm"].items():
        if not (statistics_only and name == "returns"):
            getattr(rew_norm, name)[...] = value


@torch.no_grad()
def collect(model_module, agent, snapshot, chunks, device, seed):
    """Frozen-policy rollouts, one trainer-identical GAE batch per chunk, kept on the GPU."""
    m = model_module
    args = copy.copy(snapshot_args(snapshot))
    num_envs, num_steps = args.num_envs, args.num_steps
    envs = m.make_training_env(args, f"bench_{seed}")
    try:
        obs_shape = envs.single_observation_space.shape
        host_actor = m.ResidualHostMirror(agent.actor, num_envs)
        low, high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(seed)
        sample_actions = m.make_beta_sampler(num_envs, agent.action_dim, low, high)

        def act(observations):
            native, action = sample_actions(host_actor(observations), sampler)
            return native, action.reshape((num_envs,) + agent.action_shape)

        transfer = m.RolloutTransfer(num_steps, num_envs, obs_shape, device, non_blocking=False,
                                     fields={"observations": obs_shape, "native_actions": (agent.action_dim,)})
        bootstraps = m.TruncationBootstrapCache(num_steps, num_envs, obs_shape)
        obs_norm = m.VectorObsNorm(num_envs, obs_shape)
        rew_norm = m.VectorRewardNorm(num_envs, args.gamma)
        restore_normalizers(obs_norm, rew_norm, snapshot)
        horizon = m.episode_horizon(args.env_id)
        phases = m.compute_phase_offsets(num_envs, horizon, seed)
        warm = m.run_phase_warmup(envs, obs_norm=obs_norm, rew_norm=rew_norm,
                                  act_fn=lambda observations: act(observations)[1],
                                  horizon=horizon, phase_offsets=phases, seed=seed)
        next_obs_np, suppress = warm.next_obs, warm.suppress_mask
        gae_fn = m.get_gae_fn(compiled=False)
        batches, returns = [], []
        for _ in range(chunks):
            # Freeze the snapshot's normalization so every chunk sees the same policy.
            restore_normalizers(obs_norm, rew_norm, snapshot, statistics_only=True)
            bootstraps.reset()
            for step in range(num_steps):
                obs_step = next_obs_np
                native, host_action = act(obs_step)
                raw_obs, raw_reward, terms, truncs, infos = envs.step(host_action)
                reward = rew_norm.normalize(raw_reward, terms)
                next_obs_np, transition_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                bootstraps.push_normalized(step, truncs, transition_obs)
                transfer.push(step, reward, terms, truncs, observations=obs_step, native_actions=native)
                for index, info in enumerate(infos.get("final_info", ())):
                    if info and "episode" in info:
                        # The first episode after phase warmup is partial; the trainer drops it too.
                        if suppress[index]:
                            suppress[index] = False
                            continue
                        returns.append(float(info["episode"]["r"]))
            batch = transfer.upload()
            b_obs = batch.fields["observations"].flatten(0, 1).clone()
            b_native = batch.fields["native_actions"].flatten(0, 1).clone()
            alpha, beta, value = agent.get_policy_and_value(b_obs)
            logprob = agent.action_logprob(alpha, beta, b_native)
            tail_value = agent.get_value(transfer.observation(next_obs_np)).flatten()
            truncation_values = bootstraps.resolve(agent.get_value, device)
            advantages, _ = gae_fn(batch.rewards, value.view(num_steps, num_envs), batch.terminations,
                                   batch.truncations, truncation_values, tail_value,
                                   args.gamma, args.gae_lambda)
            # lambda = 1: Monte Carlo advantages, critic-biased only through the bootstraps.
            advantages_mc, _ = gae_fn(batch.rewards, value.view(num_steps, num_envs), batch.terminations,
                                      batch.truncations, truncation_values, tail_value, args.gamma, 1.0)
            entry = {"obs": b_obs, "native": b_native, "logprob": logprob, "alpha": alpha, "beta": beta,
                     "advantages": advantages.flatten().clone(), "advantages_mc": advantages_mc.flatten().clone()}
            for lam in TARGET_LAMBDAS:
                # Training-target variants: lower lambda trades critic bias for return noise.
                entry[f"advantages_{lam:g}"] = gae_fn(batch.rewards, value.view(num_steps, num_envs),
                                                      batch.terminations, batch.truncations, truncation_values,
                                                      tail_value, args.gamma, lam)[0].flatten().clone()
            batches.append(entry)
        transfer.close()
        return batches, returns
    finally:
        envs.close()


def snapshot_args(snapshot):
    return argparse.Namespace(**snapshot["args"])


def concatenate(batches):
    return {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}


# ---------------------------------------------------------------- update + evaluation

def beta_kl(old_alpha, old_beta, alpha, beta):
    old_sum, new_sum = old_alpha + old_beta, alpha + beta
    return (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(new_sum)
            - torch.lgamma(old_alpha) - torch.lgamma(old_beta) + torch.lgamma(old_sum)
            + (old_alpha - alpha) * torch.digamma(old_alpha)
            + (old_beta - beta) * torch.digamma(old_beta)
            + (new_sum - old_sum) * torch.digamma(old_sum))


# Held-out advantage bins (z over the whole held-out set): where each update moves the policy.
Z_EDGES = (-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0)
NUM_BINS = len(Z_EDGES) + 1


def evaluate_part(agent, part):
    """Per-chunk means of improvement, KL, location-only KL and entropy change, then per-z-bin sums of
    count, log-ratio and A(r - 1), then the top-1% and top-10% states' shares of the chunk's KL."""
    alpha, beta = agent.policy(part["obs"])
    new = Beta(alpha, beta, validate_args=False)
    old = Beta(part["alpha"], part["beta"], validate_args=False)
    logratio = (new.log_prob(part["native"]) - agent.log_action_scale).sum(-1) - part["logprob"]
    # Location-only move: the new mean at the old concentration.
    concentration = part["alpha"] + part["beta"]
    mean = alpha / (alpha + beta)
    kl = beta_kl(part["alpha"], part["beta"], alpha, beta).sum(-1)
    kl_loc = beta_kl(part["alpha"], part["beta"], mean * concentration, (1 - mean) * concentration).sum(-1)
    # Concentration-only move: the old mean at the new concentration.
    old_mean, new_concentration = part["alpha"] / concentration, alpha + beta
    kl_conc = beta_kl(part["alpha"], part["beta"], old_mean * new_concentration,
                      (1 - old_mean) * new_concentration).sum(-1)
    excess = logratio.expm1()
    scalars = torch.stack(((part["advantages"] * excess).mean(), (part["advantages_mc"] * excess).mean(),
                           kl.mean(), kl_loc.mean(), kl_conc.mean(),
                           (new.entropy() - old.entropy()).sum(-1).mean()))
    bins = part["zbin"]
    binned = torch.stack([torch.zeros(NUM_BINS, device=kl.device).index_add_(0, bins, value)
                          for value in (torch.ones_like(kl), logratio, part["advantages"] * excess, kl)]).flatten()
    ordered = kl.sort(descending=True).values
    count = kl.numel()
    top = torch.stack((ordered[: count // 100].sum(), ordered[: count // 10].sum())) / kl.sum().clamp_min(1e-12)
    return torch.cat((scalars, binned, top))


@torch.no_grad()
def evaluate(evaluator, data, chunk_size):
    """Held-out improvement, KL, concentration share and entropy change, with a chunk-mean SE."""
    parts = torch.stack([evaluator({key: value[start:start + chunk_size] for key, value in data.items()})
                         for start in range(0, data["obs"].shape[0], chunk_size)])
    stacked = dict(zip(("dj", "dj_mc", "kl", "kl_loc", "kl_conc", "dh"), parts[:, :6].unbind(1)))
    out = {key: float(values.mean()) for key, values in stacked.items()}
    counts, logratios, gains, kls = parts[:, 6:6 + 4 * NUM_BINS].sum(0).view(4, NUM_BINS)
    total = counts.sum()
    # Per bin: mean log-ratio change, and that bin's contribution to dJ (sums to dJ).
    out["bin_logratio"] = (logratios / counts.clamp_min(1)).tolist()
    out["bin_dj"] = (gains / total).tolist()
    out["bin_share"] = (counts / total).tolist()
    out["bin_kl_share"] = (kls / kls.sum().clamp_min(1e-12)).tolist()
    out["kl_top1"], out["kl_top10"] = parts[:, -2:].mean(0).tolist()
    # Contiguous chunks are correlated in time, so this SE is optimistic.
    out["dj_se"] = float(stacked["dj"].std() / parts.shape[0] ** 0.5) if parts.shape[0] > 1 else float("nan")
    # Heuristic: Beta KL is not additive over mean and concentration (not Fisher-orthogonal).
    out["conc_share"] = 1.0 - out["kl_loc"] / max(out["kl"], 1e-12)
    return out


def actor_parameters(agent):
    return tuple(agent.actor.parameters())


def run_update(agent, base_state, rule, batch, targets, lr, moments, epochs, loss_fn, after_epoch=None):
    """One real update from the given Adam moments (step, exp_avg, exp_avg_sq)."""
    agent.load_state_dict(base_state)
    params = actor_parameters(agent)
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, rule.config.get("b2", 0.999)), eps=1e-5, fused=True)
    step, first, second = moments
    for param, exp_avg, exp_avg_sq in zip(params, first, second):
        optimizer.state[param] = {"step": step.clone(), "exp_avg": exp_avg.clone(), "exp_avg_sq": exp_avg_sq.clone()}
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(batch, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(params, rule.max_grad_norm, foreach=True)
        optimizer.step()
        agent.normalize_matrices()
        if after_epoch is not None:
            after_epoch()


def gram_shares(gram, keep):
    count = len(keep)
    sub = gram[keep][:, keep]
    cross = (sub.sum() - sub.diagonal().sum()) / (count * (count - 1))
    return cross, sub.diagonal().mean()


def reproducibility(fields, null_fields=None):
    """Signal share of the update: mean cross-batch inner product of held-out log-ratio changes over
    their mean energy. Independent batches share only what the data systematically says, so this is
    ~1 for a pure-signal update and ~0 for one driven by each batch's own noise. Advantage-independent
    drift (e.g. finite-sample concentration) is shared too, so the same statistic on updates with
    within-batch-permuted advantages is subtracted when given. Returns (share, jackknife SE, null share)."""
    fields = torch.stack(fields)
    gram = fields @ fields.T
    null_gram = None if null_fields is None else (lambda f: f @ f.T)(torch.stack(null_fields))
    count = fields.shape[0]

    def share(keep):
        cross, energy = gram_shares(gram, keep)
        if null_gram is not None:
            cross = cross - gram_shares(null_gram, keep)[0]
        return float(cross / energy)

    full = share(list(range(count)))
    leave = np.array([share([j for j in range(count) if j != i]) for i in range(count)])
    se = float(np.sqrt((count - 1) / count * np.square(leave - leave.mean()).sum()))
    null = None if null_gram is None else float(gram_shares(null_gram, list(range(count)))[0]
                                                / gram_shares(gram, list(range(count)))[1])
    return full, se, null


def gradient_power(agent, base_state, rule, batches, targets, loss_fn):
    """Mean squared norm of the rule's clipped first-epoch gradient over the training batches."""
    agent.load_state_dict(base_state)
    params = actor_parameters(agent)
    total = 0.0
    for batch, target in zip(batches, targets):
        agent.zero_grad(set_to_none=True)
        loss_fn(batch, target).backward()
        nn.utils.clip_grad_norm_(params, rule.max_grad_norm, foreach=True)
        total += float(sum(param.grad.square().sum() for param in params))
    agent.zero_grad(set_to_none=True)
    return total / len(batches)


def matched(points, kl_target):
    """dJ at a given KL, interpolated linearly in log KL along one rule's step-size sweep."""
    points = sorted(points)
    for (kl_a, dj_a), (kl_b, dj_b) in zip(points, points[1:]):
        if kl_a <= kl_target <= kl_b and kl_b > kl_a:
            weight = (np.log(kl_target) - np.log(kl_a)) / (np.log(kl_b) - np.log(kl_a))
            return dj_a + weight * (dj_b - dj_a)
    return None


def summarize(rows):
    summary = {key: np.mean([row[key] for row in rows], axis=0).tolist() for key in rows[0]}
    summary["dj_batch_sd"] = float(np.std([row["dj"] for row in rows]))
    return summary


def bench_snapshot(path, cli, device, model_module):
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    args = snapshot_args(snapshot)
    if args.num_minibatches != 1 or args.update_epochs != cli.epochs:
        raise ValueError(f"{path}: the bench replays 1 minibatch x {cli.epochs} epochs; the trainer used "
                         f"{args.num_minibatches} x {args.update_epochs}")
    if getattr(args, "actor_epochs", cli.epochs) != cli.epochs:
        raise ValueError(f"{path}: actor_epochs={args.actor_epochs} differs from --epochs")
    rules = [rule for rule in default_rules(args, cli.lr_mults) if cli.rules.search(rule.label)]
    if "actor_adam" not in snapshot:
        raise ValueError(f"{path}: no actor Adam state; recapture with scripts/awr_capture_policy.py")
    if args.gae_lambda != 0.95:
        raise ValueError(f"{path}: the bench's default training targets assume gae_lambda=0.95")
    adam = snapshot["adam_hyperparameters"]
    # Group 0 is the actor's; v14+ trainers give it their own beta2 (rules replay theirs via config "b2").
    betas = (0.9, float(getattr(args, "actor_adam_beta2", 0.999)))
    if (adam["eps"], tuple(adam["betas"]), adam["weight_decay"], adam["amsgrad"]) != (1e-5, betas, 0, False):
        raise ValueError(f"{path}: the bench replays Adam(eps=1e-5, betas={betas}); the trainer used {adam}")
    if getattr(args, "ent_coef", 0.0) != 0.0:
        raise ValueError(f"{path}: the bench's PPO loss has no entropy bonus; ent_coef={args.ent_coef}")
    envs = model_module.make_training_env(args, "bench_probe")
    agent = model_module.Agent(envs).to(device)
    envs.close()
    agent.load_state_dict(snapshot["model"])
    base_state = copy.deepcopy(agent.state_dict())
    start = time.perf_counter()
    batches, episode_returns = collect(model_module, agent, snapshot, cli.chunks, device, cli.seed)
    collect_s = time.perf_counter() - start
    half = cli.chunks // 2
    train_batches = batches[:half][:: max(1, half // cli.train_batches)][: cli.train_batches]
    eval_data = concatenate(batches[half:])
    edges = torch.tensor(Z_EDGES, device=device)
    # Held-out bins use the held-out set's own z; each training batch its own (as its targets do).
    for part in (*train_batches, eval_data):
        part["zbin"] = torch.bucketize(zscore(part["advantages"]), edges)
    header = {
        "snapshot": str(path), "trainer": snapshot["trainer_module"], "global_step": snapshot["global_step"],
        "learning_rate": snapshot["learning_rate"], "collect_s": collect_s,
        "episode_return_mean": float(np.mean(episode_returns)) if episode_returns else None,
        "eval_samples": int(eval_data["obs"].shape[0]), "train_batches": len(train_batches),
    }
    print(json.dumps(header), flush=True)
    all_rules = default_rules(args, (1.0,))
    native = native_family(snapshot, all_rules)
    native_rule = next(rule for rule in all_rules if rule.family == native)
    # Every loss closure shares one code object; give its guard cache room for all families.
    torch._dynamo.reset()
    torch._dynamo.config.recompile_limit = 4 * len(all_rules) + 16
    torch._dynamo.config.accumulated_recompile_limit = max(torch._dynamo.config.accumulated_recompile_limit,
                                                          8 * len(all_rules) + 64)
    evaluator = torch.compile(lambda part: evaluate_part(agent, part), fullgraph=True, dynamic=False)
    prepared = {}

    def prepare(rule):
        if rule.family not in prepared:
            def loss(batch, targets, rule=rule):
                return rule.loss(agent, batch, targets, rule.config)
            loss_fn = torch.compile(loss, fullgraph=True, dynamic=False)
            targets = [rule.prepare(batch[rule.advantage_key], rule.config, batch) for batch in train_batches]
            prepared[rule.family] = (loss_fn, targets,
                                     gradient_power(agent, base_state, rule, train_batches, targets, loss_fn))
        return prepared[rule.family]

    # The trainer's real actor moments as this update begins.
    names = [name for name, _ in agent.actor.named_parameters()]
    real = snapshot["actor_adam"]
    real_step = real[names[0]]["step"].to(device=device, dtype=torch.float32)
    real_first = [real[name]["exp_avg"].to(device) for name in names]
    real_second = [real[name]["exp_avg_sq"].to(device) for name in names]

    # Validation: the native rule replayed exactly as the trainer runs it must reproduce the logged KL.
    loss_fn, targets, native_power = prepare(native_rule)
    replay_rows = []
    for batch, target in zip(train_batches, targets):
        run_update(agent, base_state, native_rule, batch, target, snapshot["learning_rate"],
                   (real_step, real_first, real_second), cli.epochs, loss_fn)
        in_sample = evaluate(evaluator, batch, cli.eval_chunk)
        replay_rows.append({**evaluate(evaluator, eval_data, cli.eval_chunk),
                            "dj_train": in_sample["dj"], "kl_train": in_sample["kl"]})
    replay = summarize(replay_rows)
    replay["label"] = native_rule.label
    # The trainer logs KL at the start of its last epoch (9 steps, in-sample): expect a match within ~30%.
    print(f"native replay {native_rule.label}: KL={replay['kl']:.4f} KLtrain={replay['kl_train']:.4f} dJ={replay['dj']*1e3:.3f}e-3 "
          f"dJmc={replay['dj_mc']*1e3:.3f}e-3 dJtr={replay['dj_train']*1e3:.3f}e-3 dH={replay['dh']:+.4f}",
          flush=True)

    # Every rule steps in the trainer's real per-parameter geometry, with the second moment rescaled to its
    # own gradient power and no inherited momentum; the lr sweep then traces its dJ-vs-KL frontier.
    scales = {}

    def moments_of(rule):
        scale = scales[rule.family]
        return real_step, [torch.zeros_like(first) for first in real_first], [second * scale for second in real_second]

    # Strided (time-spread) held-out subsets: a probe for update fields, a trace set for epoch paths.
    probe = {key: value[:: max(1, value.shape[0] // cli.probe)][: cli.probe] for key, value in eval_data.items()}
    trace_set = {key: value[:: max(1, value.shape[0] // cli.trace)][: cli.trace] for key, value in eval_data.items()}
    probe_logratio = torch.compile(lambda part: logratio_of(agent, part)[2], fullgraph=True, dynamic=False)

    results = []
    for rule in rules:
        loss_fn, targets, power = prepare(rule)
        scales[rule.family] = power / native_power
        moments = moments_of(rule)
        lr = snapshot["learning_rate"] * rule.lr_mult
        rows, fields, paths = [], [], []
        for batch, target in zip(train_batches, targets):
            path_points = []

            def trace():
                with torch.no_grad():
                    point = evaluate(evaluator, trace_set, cli.eval_chunk)
                path_points.append((point["dj"], point["dj_mc"], point["kl"]))

            run_update(agent, base_state, rule, batch, target, lr, moments, cli.epochs, loss_fn,
                       after_epoch=trace)
            held_out = evaluate(evaluator, eval_data, cli.eval_chunk)
            in_sample = evaluate(evaluator, batch, cli.eval_chunk)
            rows.append({**held_out, "dj_train": in_sample["dj"], "train_bin_logratio": in_sample["bin_logratio"],
                         "train_bin_kl_share": in_sample["bin_kl_share"], "train_kl_top1": in_sample["kl_top1"]})
            with torch.no_grad():
                fields.append(probe_logratio(probe).float())
            if path_points:
                paths.append(path_points)
        # Null: the same rule and target distribution with advantages permuted within each batch.
        null_fields = []
        generator = torch.Generator(device=device).manual_seed(cli.seed)
        for batch in train_batches:
            shuffled = batch[rule.advantage_key][torch.randperm(batch["obs"].shape[0], device=device,
                                                                generator=generator)]
            run_update(agent, base_state, rule, batch, rule.prepare(shuffled, rule.config, batch), lr, moments,
                       cli.epochs, loss_fn)
            with torch.no_grad():
                null_fields.append(probe_logratio(probe).float())
        summary = summarize(rows)
        summary["dj_rows"] = [row["dj"] for row in rows]
        summary["dh_rows"] = [row["dh"] for row in rows]
        raw_share, raw_se, _ = reproducibility(fields)
        summary["reproducibility"], summary["reproducibility_se"] = raw_share, raw_se
        if null_fields is not None:
            share, se, null = reproducibility(fields, null_fields)
            summary["signal_share"], summary["signal_share_se"], summary["null_share"] = share, se, null
        reference = next((result for result in results if result["family"] == results[0]["family"]
                          and result["lr_mult"] == rule.lr_mult), None) if results else None
        if reference is not None:
            difference = np.asarray(summary["dj_rows"]) - np.asarray(reference["dj_rows"])
            summary["dj_vs_ref"] = float(difference.mean())
            summary["dj_vs_ref_se"] = float(difference.std(ddof=1) / np.sqrt(len(difference)))
            # Same batches, so the batch's own entropy drift cancels: resolves the ~1e-3/update
            # concentration drift that separates a plateauing rule from PPO over thousands of updates.
            difference = np.asarray(summary["dh_rows"]) - np.asarray(reference["dh_rows"])
            summary["dh_vs_ref"] = float(difference.mean())
            summary["dh_vs_ref_se"] = float(difference.std(ddof=1) / np.sqrt(len(difference)))
        if paths:
            # Mean held-out (dJ, dJ_mc, KL) after each epoch: where does the gain stop?
            summary["epoch_path"] = np.mean(np.asarray(paths), axis=0).tolist()
        summary["label"], summary["lr_mult"] = rule.label, rule.lr_mult
        summary["family"] = f"{rule.name}[{','.join(f'{k}={v}' for k, v in rule.config.items())}]"
        # dJ is linear and KL quadratic in a small step, so dJ/sqrt(KL) is step-scale invariant.
        summary["efficiency"] = summary["dj"] / max(summary["kl"], 1e-12) ** 0.5
        summary["efficiency_mc"] = summary["dj_mc"] / max(summary["kl"], 1e-12) ** 0.5
        summary["v_scale"] = scales[rule.family]
        results.append(summary)
        print(f"{rule.label:34s} dJ={summary['dj']*1e3:8.3f}e-3 (sd {summary['dj_batch_sd']*1e3:6.3f}) "
              f"dJmc={summary['dj_mc']*1e3:8.3f}e-3 dJtr={summary['dj_train']*1e3:8.3f}e-3 "
              f"KL={summary['kl']:.4f} eff={summary['efficiency']:7.4f} effmc={summary['efficiency_mc']:7.4f} "
              f"kl_loc={summary['kl_loc']:.4f} kl_conc={summary['kl_conc']:.4f} dH={summary['dh']:+.4f} "
              f"repro={summary['reproducibility']:.3f}±{summary['reproducibility_se']:.3f}"
              + (f" signal={summary['signal_share']:.3f}±{summary['signal_share_se']:.3f} "
                 f"null={summary['null_share']:.3f}" if "signal_share" in summary else "")
              + (f" dJ-ref={summary['dj_vs_ref']*1e3:+.3f}±{summary['dj_vs_ref_se']*1e3:.3f}e-3"
                 f" dH-ref={summary['dh_vs_ref']*1e3:+.2f}±{summary['dh_vs_ref_se']*1e3:.2f}e-3"
                 if "dj_vs_ref" in summary else ""), flush=True)
        print("    held-out by z bin " + " ".join(f"{edge:+g}" for edge in Z_EDGES) + " | mean dlogpi (e-3): "
              + " ".join(f"{value*1e3:.1f}" for value in summary["bin_logratio"])
              + " | dJ (e-3): " + " ".join(f"{value*1e3:+.3f}" for value in summary["bin_dj"])
              + f" | KL top1% {summary['kl_top1']:.3f} top10% {summary['kl_top10']:.3f}", flush=True)
        print("    in-sample by z bin | mean dlogpi (e-3): " + " ".join(f"{value*1e3:.1f}" for value in summary["train_bin_logratio"])
              + " | KL share: " + " ".join(f"{value:.3f}" for value in summary["train_bin_kl_share"])
              + f" | bin share: " + " ".join(f"{value:.3f}" for value in summary["bin_share"])
              + f" | train KL top1% {summary['train_kl_top1']:.3f}", flush=True)
        if "epoch_path" in summary:
            print("    epoch path dJ(e-3)@KL: " + " ".join(
                f"{index + 1}:{point[0]*1e3:.3f}@{point[2]:.4f}" for index, point in enumerate(summary["epoch_path"])),
                flush=True)
    frontier = {}
    for family in dict.fromkeys(result["family"] for result in results):
        sweep = [result for result in results if result["family"] == family]
        frontier[family] = {f"{target:g}": {key: matched([(result["kl"], result[key]) for result in sweep], target)
                                             for key in ("dj", "dj_mc", "dj_train", "dh", "reproducibility", "signal_share")}
                            for target in cli.kl_targets}
    print("dJ (x1e-3, GAE / MC) and reproducibility at matched KL:", flush=True)
    for family, values in frontier.items():
        cells = "  ".join(f"KL {target}: " + ("-" if cell["dj"] is None else
                          f"{cell['dj']*1e3:7.3f} / {cell['dj_mc']*1e3:7.3f} s{cell['signal_share']:.2f}")
                          for target, cell in values.items())
        print(f"  {family:40s} {cells}", flush=True)
    agent.load_state_dict(base_state)
    return {"header": header, "native_replay": replay, "results": results, "frontier": frontier}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshots", nargs="+", type=Path)
    parser.add_argument("--chunks", type=int, default=128, help="16x1024 rollouts; the second half is held out")
    parser.add_argument("--train-batches", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-chunk", type=int, default=65536)
    # Cold momentum (m=0) travels ~4 of 10 steps' worth, so the realistic KL range sits above lr x1.
    parser.add_argument("--lr-mults", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.5, 6.5])
    parser.add_argument("--probe", type=int, default=65536, help="held-out samples for update fields")
    parser.add_argument("--trace", type=int, default=131072, help="held-out samples for per-epoch paths")
    parser.add_argument("--kl-targets", type=float, nargs="+", default=[0.01, 0.02, 0.03])
    parser.add_argument("--rules", type=re.compile, default=re.compile(""), help="regex over rule labels")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    cli = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model_module = importlib.import_module(MODEL_MODULE)
    model_module.configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(cli.seed)
    device = torch.device("cuda")
    report = [bench_snapshot(path, cli, device, model_module) for path in cli.snapshots]
    if cli.out is not None:
        cli.out.parent.mkdir(parents=True, exist_ok=True)
        cli.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
