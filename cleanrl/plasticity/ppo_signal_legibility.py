"""How does a unit's scalar teaching signal vary with its own state in PPO?

THE QUESTION. This diagnostic measures one candidate source of state-dependent
plasticity: the conditional mean and variation of a hidden unit's scalar
backpropagated signal across bins of its own preactivation. It does not test all
state-dependent learning rules or measure their optimizer performance.

WHAT IS PAIRED. For every hidden unit of the actor and critic trunks, and every
sample in every minibatch, we read the exact pair

    state   z_bh = the unit's own preactivation on that sample
    signal  d_bh = dL/dz_bh, the scalar teaching signal at that unit

`d` comes from `retain_grad()` on the preactivation of the real clipped PPO loss,
including its 1/B reduction. A common nonzero scale cancels from these ratios.
For an incoming weight the sample gradient is d_bh * x_bj, not d_bh alone:
E[d | z] does not determine E[d * x | z] or the full incoming gradient vector.
No measured statistic is used to change the optimizer.

THE STATISTICS. Each update splits each unit's samples into K equal-count rank
bins of its own preactivation. For bin counts n_k, means m_k and grand mean m:

    SSB = sum_k n_k (m_k - m)^2
    SSW = SST - SSB
    F   = (SSB/(K-1)) / (SSW/(n-K))
    eta2_adj = max(0, (SSB - (K-1)*MSW) / SST)

The nonnegative adjusted effect estimate fluctuates under pure noise; it is not
deterministically zero. Its classical bias adjustment assumes an independent,
homoskedastic model. PPO samples share trajectories, advantages and repeated
rollout reuse, and can be heteroskedastic, so F and the adjustment are descriptive
here, not calibrated hypothesis tests. Even in the classical null model F is a
random variable, not identically 1.

We also compute ANOVA after scaling each bin by its empirical standard deviation.
This tests variation in signed bin SNR, not only reliability magnitude. Scaling
cancels a positive gain that is constant within a bin, but not the tanh slope
when it varies within a coarse bin. Pooled statistics mix changing rank-bin
boundaries and policies across updates; more samples are not independent repeats.

THE CONTROL. Each unit's signal column is independently permuted across samples
within the same update. This preserves both marginals and update identity while
breaking their observed pairing. It also disrupts within-update dependence, so
the control is descriptive, not an exchangeability-valid PPO null distribution.
One control value per unit is not a per-unit permutation test, and its across-unit
95th percentile does not imply a calibrated 5% false-positive rate.

A tie or small effect only limits this scalar, coarse-state diagnostic under the
sampled policy and measurement window. It cannot falsify rules using incoming
inputs, richer state, history, covariance, or different learning dynamics.
The reported squared-SNR weighting references optimize a plug-in standardized
scalar aggregation model, not optimizer improvement or return.

Diagnostic only: this runs the baseline PPO learner to collect measurements.
Queue execution through mlq, for example with this command as its payload:

    .venv/bin/python cleanrl/plasticity/ppo_signal_legibility.py
    .venv/bin/python cleanrl/plasticity/ppo_signal_legibility.py --total-steps 4000000
"""
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import Beta

from cleanrl.plasticity.ppo_continuous_action_precision_v1 import Agent
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import sample_beta_actions
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm


@dataclass
class Args:
    env_id: str = "HalfCheetah-v4"
    """target environment; the domain whose legibility is in question"""
    total_steps: int = 2_000_000
    """environment steps to harvest over"""
    num_envs: int = 16
    """parallel environments"""
    num_steps: int = 256
    """rollout length per environment"""
    minibatch_size: int = 1024
    """PPO minibatch, i.e. how many (unit, sample) pairs per unit per update"""
    update_epochs: int = 10
    """PPO epochs per rollout"""
    learning_rate: float = 8.1e-4
    """the LR-tuned baseline learning rate, so the learner being measured is the
    reference learner and not a detuned one"""
    bins: int = 8
    """equal-count rank bins of a unit's own preactivation"""
    stages: int = 8
    """report windows across the harvest, to expose any late-training onset"""
    seed: int = 1
    """paired seed: the same stream and init feed the true and permuted arms,
    because they are literally the same forward and backward pass"""
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True


class Harvest:
    """Accumulates the ANOVA sufficient statistics for the true and permuted arms."""

    def __init__(self, units, bins, device):
        self.units, self.bins, self.device = units, bins, device
        self.reset()

    def reset(self):
        shape = (self.bins, self.units)
        zeros = lambda: torch.zeros(shape, device=self.device)  # noqa: E731
        self.count = torch.zeros(shape, device=self.device)
        self.slope = torch.zeros(shape, device=self.device)
        self.sums = {"true": zeros(), "perm": zeros()}
        self.squares = {"true": zeros(), "perm": zeros()}
        # per-update statistics, averaged over updates
        self.keys = ("f", "eta", "f_snr", "eta_snr")
        self.per_update = {arm: {key: torch.zeros(self.units, device=self.device)
                                 for key in self.keys}
                           for arm in ("true", "perm")}
        self.updates = 0

    @torch.no_grad()
    def observe(self, state, signal):
        """`state`, `signal`: (B, U) preactivations and their per-sample gradients."""
        samples, units = state.shape
        bins = self.bins
        if samples % bins:
            raise ValueError("minibatch size must be divisible by the bin count")
        per_bin = samples // bins
        # exact equal-count bins of each unit's OWN preactivation
        order = state.argsort(0).argsort(0)
        index = order.mul_(bins).div_(samples, rounding_mode="floor")
        # permuted arm: each unit's signal column independently shuffled across
        # samples of THIS update. Marginals and update identity preserved.
        permutation = torch.rand_like(signal).argsort(0)
        columns = {"true": signal, "perm": signal.gather(0, permutation)}
        self.count.scatter_add_(0, index, torch.ones_like(signal))
        # Describe the mean own-state gain by bin; this does not identify how
        # much of the raw signal variation is attributable to the tanh slope.
        self.slope.scatter_add_(0, index, 1.0 - state.tanh().square())
        self.updates += 1
        for arm, values in columns.items():
            totals = torch.zeros((bins, units), device=values.device)
            squares = torch.zeros((bins, units), device=values.device)
            totals.scatter_add_(0, index, values)
            squares.scatter_add_(0, index, values.square())
            self.sums[arm] += totals
            self.squares[arm] += squares
            statistic = anova(totals, squares, per_bin, bins)
            for key in self.keys:
                self.per_update[arm][key] += statistic[key]

    def report(self):
        out = {}
        for arm in ("true", "perm"):
            pooled = anova(self.sums[arm], self.squares[arm], self.count[0], self.bins)
            entry = {f"{key}_update": self.per_update[arm][key] / max(self.updates, 1)
                     for key in self.keys}
            entry.update({f"{key}_pooled": value for key, value in pooled.items()})
            out[arm] = entry
        mean_slope = self.slope / self.count.clamp_min(1.0)
        out["slope_spread"] = mean_slope.clamp_min(1e-30).log().std(0)
        out["cells"] = {arm: self.cell_snr(arm) for arm in ("true", "perm")}
        return out

    def cell_snr(self, arm):
        """Per (state bin, unit) SNR of the teaching signal, for persistence tests."""
        counts = self.count.clamp_min(1.0)
        means = self.sums[arm] / counts
        variance = (self.squares[arm] / counts - means.square()).clamp_min(0.0)
        return means / variance.sqrt().clamp_min(1e-30)


def pearson(left, right):
    """Correlation between two flattened cell patterns."""
    left = left - left.mean()
    right = right - right.mean()
    return (left * right).sum() / (left.norm() * right.norm()).clamp_min(1e-30)


def anova(totals, squares, per_bin, bins):
    """One-way ANOVA of a unit's teaching signal on its own state bins.

    ``totals``/``squares``: (K, U) per-bin sums of ``d`` and ``d^2``.
    ``per_bin``: samples per bin, scalar or (U,). Returns per-unit statistics.

    Raw signal is ``dL/dz = (dL/da) * (1 - tanh(z)^2)``. The own-state
    slope can contribute to raw conditional moments, but need not create a
    nonzero conditional mean. Bin-SD normalization cancels a positive gain only
    when it is constant within each bin; varying slope remains a confound.

    ``f_snr`` and ``eta_snr`` are ANOVA summaries of the empirically standardized
    samples, whose bin means are signed SNRs. Neither they nor the raw ANOVA
    provide calibrated inference under PPO dependence and heteroskedasticity.
    """
    per_bin = torch.as_tensor(per_bin, device=totals.device, dtype=totals.dtype)
    samples = per_bin * bins
    means = totals / per_bin.clamp_min(1.0)
    grand = totals.sum(0) / samples.clamp_min(1.0)
    between = (per_bin * (means - grand).square()).sum(0)
    total = squares.sum(0) - samples * grand.square()
    within = (total - between).clamp_min(0.0)
    degrees = (samples - bins).clamp_min(1.0)
    mean_square_within = within / degrees
    f = (between / (bins - 1)) / mean_square_within.clamp_min(1e-30)
    eta = ((between - (bins - 1) * mean_square_within) / total.clamp_min(1e-30)).clamp_min(0.0)
    variance = (squares / per_bin.clamp_min(1.0) - means.square()).clamp_min(0.0)
    deviation = variance.clamp_min(1e-30).sqrt()
    # Empirical mean / population SD per state cell. This is invariant to a
    # positive bin-constant gain, not to general sample-varying own-state gains.
    snr = means / deviation.clamp_min(1e-30)
    # Squared empirical mean / empirical second moment, not a calibrated
    # probability that the teaching signal is correct.
    reliability = snr.square() / (1.0 + snr.square())
    # Apply the same ANOVA to the bin-standardized samples. Population-SD
    # scaling gives n residual SS for nondegenerate bins, not MSW == 1:
    # the residual degrees of freedom are n-K. Retain the actual variance ratio
    # for zero-variance bins and the numerical floor.
    snr_mean = snr.mean(0)
    between_snr = per_bin * (snr - snr_mean).square().sum(0)
    within_snr = (per_bin * variance / deviation.square()).sum(0)
    mean_square_within_snr = within_snr / degrees
    f_snr = (between_snr / (bins - 1)) / mean_square_within_snr.clamp_min(1e-30)
    total_snr = between_snr + within_snr
    eta_snr = ((between_snr - (bins - 1) * mean_square_within_snr)
               / total_snr.clamp_min(1e-30)).clamp_min(0.0)
    # Plug-in references for J(w) = (sum_k w_k snr_k)^2 / (K sum_k w_k^2).
    # This models equally represented independent, unit-variance scalar cells;
    # it omits actual input-weighted gradients, covariance and optimizer dynamics.
    # Uniform refers to weights on standardized signals, not on raw gradients.
    # For w >= 0, the squared objective chooses the larger same-sign half.
    # Signed weights attain mean(snr^2). These are in-sample fitted references,
    # not bounds on optimizer improvement, and the permutation is descriptive.
    # No universal 1/n bias subtraction is valid for the selected-sign gate
    # (or under PPO dependence); keep every reference and ratio unadjusted.
    uniform = snr_mean.square()
    positive = snr.clamp_min(0.0).square().mean(0)
    negative = snr.clamp_max(0.0).square().mean(0)
    gate = torch.maximum(positive, negative)
    oracle = snr.square().mean(0)
    return {"f": f, "eta": eta, "f_snr": f_snr, "eta_snr": eta_snr,
            "snr_abs": snr.abs().median(0).values,
            "snr_max": snr.abs().max(0).values,
            "attainable": oracle,
            "uniform": uniform,
            "gain": oracle / uniform.clamp_min(1e-30),
            "gate_abs": gate,
            "gate_gain": gate / uniform.clamp_min(1e-30),
            "oracle_gain": oracle / uniform.clamp_min(1e-30),
            "sign_balance": (snr > 0).to(snr.dtype).mean(0),
            "reliability": reliability.max(0).values - reliability.min(0).values,
            "noise_spread": deviation.clamp_min(1e-30).log().std(0)}


class Instrumented(nn.Module):
    """The baseline Agent, with every hidden preactivation exposed for reading."""

    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.trunks = []
        for trunk in (agent.actor, agent.critic):
            modules = list(trunk)
            linears = [i for i, module in enumerate(modules) if isinstance(module, nn.Linear)]
            self.trunks.append((modules, set(linears[:-1])))
        self.units = sum(modules[i].weight.shape[0]
                         for modules, tracked in self.trunks for i in tracked)

    def forward(self, observations):
        states, outputs = [], []
        for modules, tracked in self.trunks:
            activations = observations
            for index, module in enumerate(modules):
                activations = module(activations)
                if index in tracked:
                    activations.retain_grad()
                    states.append(activations)
            outputs.append(activations)
        return states, outputs

    @staticmethod
    def gradients(states):
        return [state.grad for state in states]


def ppo_loss(head, value, native_actions, old_logprobs, advantages, returns, old_values,
             log_scale, args):
    alpha, beta = (F.softplus(head) + 1.0).chunk(2, dim=-1)
    distribution = Beta(alpha, beta, validate_args=False)
    newlogprob = (distribution.log_prob(native_actions) - log_scale).sum(-1)
    ratio = (newlogprob - old_logprobs).exp()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss = torch.max(-advantages * ratio,
                        -advantages * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
    newvalue = value.view(-1)
    clipped = old_values + (newvalue - old_values).clamp(-args.clip_coef, args.clip_coef)
    v_loss = 0.5 * torch.max((newvalue - returns).square(), (clipped - returns).square()).mean()
    return pg_loss + args.vf_coef * v_loss


@torch.no_grad()
def collect(agent, envs, obs_norm, rew_norm, next_obs, args, obs_dim, device):
    steps, envs_count = args.num_steps, args.num_envs
    buffers = {name: torch.zeros((steps, envs_count) + shape, device=device)
               for name, shape in (("obs", (obs_dim,)), ("act", (agent.action_dim,)),
                                   ("logp", ()), ("val", ()), ("rew", ()), ("done", ()))}
    for step in range(steps):
        obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
        alpha, beta, value = agent.get_policy_and_value(obs_t)
        native, physical = sample_beta_actions(alpha, beta, agent.action_low, agent.action_high)
        buffers["obs"][step] = obs_t
        buffers["act"][step] = native
        buffers["logp"][step] = agent.action_logprob(alpha, beta, native)
        buffers["val"][step] = value.flatten()
        raw_obs, raw_reward, terms, truncs, infos = envs.step(
            physical.cpu().numpy().reshape((envs_count,) + agent.action_shape))
        buffers["rew"][step] = torch.as_tensor(rew_norm.normalize(raw_reward, terms),
                                               dtype=torch.float32, device=device)
        next_obs, _ = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
        buffers["done"][step] = torch.as_tensor(np.maximum(terms, truncs).astype(np.float32),
                                                device=device)
    tail = agent.get_value(torch.as_tensor(next_obs, dtype=torch.float32,
                                           device=device)).flatten()
    advantages = torch.zeros_like(buffers["rew"])
    running = torch.zeros_like(tail)
    for step in reversed(range(steps)):
        following = tail if step == steps - 1 else buffers["val"][step + 1]
        nonterminal = 1.0 - buffers["done"][step]
        delta = buffers["rew"][step] + args.gamma * nonterminal * following - buffers["val"][step]
        running = delta + args.gamma * args.gae_lambda * nonterminal * running
        advantages[step] = running
    returns = advantages + buffers["val"]
    flat = (buffers["obs"].flatten(0, 1), buffers["act"].flatten(0, 1),
            buffers["logp"].flatten(0, 1), advantages.flatten(), returns.flatten(),
            buffers["val"].flatten())
    return flat, next_obs


def summarize(label, report, units):
    true, perm = report["true"], report["perm"]
    lines = []

    def compare(name, key, digits=4):
        observed, control = true[key], perm[key]
        threshold = torch.quantile(control, 0.95)
        excess = (observed > threshold).float().mean()
        lines.append(f"  {name:<26} true {observed.median():9.{digits}f}   "
                     f"permuted {control.median():9.{digits}f}   "
                     f"units > perm p95 {excess * 100:5.1f}% (descriptive)   "
                     f"max true {observed.max():9.{digits}f}")

    lines.append("  BIN-STANDARDIZED: does signed scalar SNR vary with own-state rank?")
    lines.append("  Bin-varying tanh slope remains; F and perm p95 are not calibrated tests.")
    compare("F of SNR, per update", "f_snr_update")
    compare("F of SNR, pooled", "f_snr_pooled")
    compare("adjusted eta2, SNR", "eta_snr_pooled", digits=6)
    compare("reliability range [0,1]", "reliability_pooled", digits=6)
    compare("|SNR| median over bins", "snr_abs_pooled", digits=5)
    lines.append("  RAW: scalar moments can reflect both upstream signal and tanh slope.")
    compare("F of raw signal, pooled", "f_pooled")
    compare("adjusted eta2, raw", "eta_pooled", digits=6)
    lines.append(f"  {'log-sd of within-bin SD':<26} true "
                 f"{true['noise_spread_pooled'].median():9.4f}   permuted "
                 f"{perm['noise_spread_pooled'].median():9.4f}   "
                 f"log-sd of tanh slope by state {report['slope_spread'].median():7.4f}")
    lines.append("  PLUG-IN squared-SNR references for standardized scalar aggregation.")
    lines.append("  In-sample fitted weights; no bias subtraction or optimizer-return bound.")
    lines.append("  Per-unit RATIOS are heavy-tailed (near-zero denominators), so the")
    lines.append("  headline is the ratio of MEDIANS; per-unit ratios follow for shape.")
    for name, key in (("signed reference", "attainable_pooled"),
                      ("gate (w >= 0)", "gate_abs_pooled")):
        numerator, control = true[key].median(), perm[key].median()
        denominator, control_denominator = true["uniform_pooled"].median(), \
            perm["uniform_pooled"].median()
        lines.append(f"  {name:<26} true {numerator:.3e} / {denominator:.3e} = "
                     f"{numerator / denominator.clamp_min(1e-30):7.2f}x   "
                     f"permuted {control:.3e} / {control_denominator:.3e} = "
                     f"{control / control_denominator.clamp_min(1e-30):6.2f}x")
    compare("gate gain, per-unit median", "gate_gain_pooled")
    compare("signed gain, per-unit med", "oracle_gain_pooled")
    compare("fraction of cells snr>0", "sign_balance_pooled")
    print(f"\n{label}  ({units} units)")
    print("\n".join(lines))
    return {"f_snr_true": float(true["f_snr_pooled"].median()),
            "f_snr_perm": float(perm["f_snr_pooled"].median()),
            "gain_true": float(true["gain_pooled"].median()),
            "gain_perm": float(perm["gain_pooled"].median()),
            "gate_true": float(true["gate_gain_pooled"].median()),
            "gate_perm": float(perm["gate_gain_pooled"].median()),
            "eta_snr_true": float(true["eta_snr_pooled"].median()),
            "eta_snr_perm": float(perm["eta_snr_pooled"].median()),
            "rel_true": float(true["reliability_pooled"].median()),
            "rel_perm": float(perm["reliability_pooled"].median()),
            "f_raw_true": float(true["f_pooled"].median()),
            "f_raw_perm": float(perm["f_pooled"].median()),
            "noise_true": float(true["noise_spread_pooled"].median()),
            "slope_spread": float(report["slope_spread"].median())}


def main():
    args = tyro.cli(Args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    envs = make_mujoco_vector_env(args.env_id, args.num_envs, backend="native",
                                  num_threads=min(4, args.num_envs))
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    agent = Agent(envs).to(device)
    probe = Instrumented(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    obs_norm = VectorObsNorm(args.num_envs, (obs_dim,))
    rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
    raw_obs, _ = envs.reset(seed=args.seed)
    next_obs = obs_norm.normalize(raw_obs)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    harvest = Harvest(probe.units, args.bins, device)

    batch = args.num_envs * args.num_steps
    rollouts = max(args.total_steps // batch, args.stages)
    per_stage = max(rollouts // args.stages, 1)
    print(f"# {args.env_id}: {rollouts} rollouts x {batch} steps = "
          f"{rollouts * batch} env steps, minibatch {args.minibatch_size}, "
          f"{probe.units} hidden units, {args.bins} own-state bins, lr {args.learning_rate}")
    print(f"# pairs per unit per stage: "
          f"{per_stage * args.update_epochs * (batch // args.minibatch_size) * args.minibatch_size}")
    stage_rows, returns_seen, start = [], [], time.perf_counter()
    previous = None
    for rollout in range(rollouts):
        data, next_obs = collect(agent, envs, obs_norm, rew_norm, next_obs, args,
                                 obs_dim, device)
        returns_seen.append(float(data[4].mean()))
        for _ in range(args.update_epochs):
            order = torch.randperm(batch, device=device, generator=generator)
            for start_index in range(0, batch, args.minibatch_size):
                indices = order[start_index:start_index + args.minibatch_size]
                obs, act, logp, adv, ret, val = (tensor[indices] for tensor in data)
                states, outputs = probe(obs)
                loss = ppo_loss(outputs[0], outputs[1], act, logp, adv, ret, val,
                                agent.log_action_scale, args)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                harvest.observe(torch.cat([state.detach() for state in states], dim=1),
                                torch.cat(probe.gradients(states), dim=1))
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
        if (rollout + 1) % per_stage == 0 or rollout == rollouts - 1:
            steps = (rollout + 1) * batch
            report = harvest.report()
            row = summarize(f"=== steps {steps / 1e6:.2f}M  "
                            f"(mean return proxy {np.mean(returns_seen[-per_stage:]):.1f}) ===",
                            report, probe.units)
            row["steps"] = steps
            # Adjacent-stage correlation of within-unit centered bin SNRs is
            # descriptive persistence, not held-out validation of an optimizer.
            # Centering removes each unit's unconditional SNR level; state-varying
            # slope and changing rank boundaries can still affect the pattern.
            # Permuted persistence is a comparison, not a guaranteed zero null.
            row["persist"], row["persist_perm"] = float("nan"), float("nan")
            if previous is not None:
                for key, arm in (("persist", "true"), ("persist_perm", "perm")):
                    current = report["cells"][arm]
                    current = current - current.mean(0, keepdim=True)
                    earlier = previous[arm]
                    row[key] = float(pearson(current.flatten(), earlier.flatten()))
            previous = {arm: (report["cells"][arm]
                              - report["cells"][arm].mean(0, keepdim=True)).clone()
                        for arm in ("true", "perm")}
            print(f"  {'centred persistence':<26} true {row['persist']:9.4f}   "
                  f"permuted {row['persist_perm']:9.4f}")
            stage_rows.append(row)
            harvest.reset()
    envs.close()

    print(f"\n# harvest wall time {time.perf_counter() - start:.1f}s")
    print("\n=== SUMMARY: own-state scalar-signal diagnostic, not an optimizer verdict ===")
    print("# F and adjusted eta2 summarize signed bin-SNR variation; neither has")
    print("# calibrated null inference here. Noise can yield positive adjusted eta2.")
    print("# `perm` preserves update marginals but disrupts dependence as well as pairing.")
    print("# Small effects do not rule out richer state/input/history-dependent learning.")
    print(f"{'steps':>10} {'F_snr true':>11} {'F_snr perm':>11} {'ratio':>7} "
          f"{'eta2 true':>10} {'gate true':>10} {'gate perm':>10} "
          f"{'persist':>9} {'persist perm':>13} "
          f"{'rel range':>10} {'SD spread':>9} {'slope sd':>9}")
    for row in stage_rows:
        print(f"{row['steps'] / 1e6:9.2f}M {row['f_snr_true']:11.4f} "
              f"{row['f_snr_perm']:11.4f} "
              f"{row['f_snr_true'] / max(row['f_snr_perm'], 1e-9):7.3f} "
              f"{row['eta_snr_true']:10.6f} "
              f"{row['gate_true']:10.4f} {row['gate_perm']:10.4f} "
              f"{row['persist']:9.4f} {row['persist_perm']:13.4f} "
              f"{row['rel_true']:10.6f} {row['noise_true']:9.4f} "
              f"{row['slope_spread']:9.4f}")


if __name__ == "__main__":
    main()
