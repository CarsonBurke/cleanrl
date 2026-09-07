"""Is a unit's teaching signal legible from that unit's OWN state, in real PPO?

THE QUESTION. State-dependent plasticity requires that, for a single perceptron
`h` on a single sample `b`, the reliability of the teaching signal it receives
depends on ITS OWN state on that sample. Everything in this family so far has
assumed that; nothing has measured it in the target domain. This harness
measures it directly from real PPO updates on real HalfCheetah-v4 rollouts.

WHAT IS PAIRED. For every hidden unit of the actor and critic trunks, and every
sample in every minibatch, we read the exact pair

    state   z_bh = the unit's own preactivation on that sample
    signal  d_bh = dL/dz_bh, the teaching signal that sample sends to that unit

`d` is obtained from `retain_grad()` on the preactivation tensor, so it is the
true per-sample per-unit backprop signal of the real clipped PPO loss (scaled by
1/B, a constant that cancels out of every statistic below). No batch statistic is
used as a mechanism: the batch is only a measurement device.

THE STATISTIC. Within each update, each unit's samples are split into K equal
octile bins of ITS OWN preactivation (exact rank bins, so n_k is identical
across bins and no threshold is tuned). Per unit we then form a one-way analysis
of variance of its teaching signal on its own state:

    SSB = sum_k n_k (m_k - m)^2          between-bin, i.e. state-explained
    SSW = SST - SSB                      within-bin, i.e. sampling noise
    F   = (SSB/(K-1)) / (SSW/(n-K))      1.0 under "signal independent of state"
    eta2_adj = (SSB - (K-1)*MSW) / SST    bias-corrected fraction in [0,1]

`eta2_adj` is the fraction of a unit's teaching-signal energy that is legible
from its own state, and it maps pure noise to 0 by construction. `F` answers
Main's exact question: does E[d | own state] have magnitude above its own
sampling error. Both are also accumulated ACROSS updates at fixed rank bins,
which multiplies the sample count per bin by the number of updates and is the
high-power version of the same test.

THE CONTROL, which is the whole point. Every statistic is computed twice: once on
the true pairing, and once with each unit's signal column independently permuted
across the samples WITHIN THE SAME UPDATE. That destroys the state -> signal
correspondence and nothing else: both marginals are preserved exactly, the
update identity is preserved exactly, so slow drift over training (a unit's state
and its signal both moving with the policy) cannot masquerade as state
dependence. If the true pairing ties its permuted control, state dependence
contributed exactly zero and the premise is not expressible in this domain.

The harness reports per training stage, because the one place this family has
ever seen a positive effect was late in a long PPO run.

Diagnostic only: this trains nothing that is scored, submits nothing, and takes
minutes. Usage:

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
        # the deterministic own-state gain, so its share of any raw effect is
        # measurable rather than assumed
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

    THE GAIN CONFOUND. A unit's raw signal is ``dL/dz = (dL/da) * (1 - a^2)``,
    and the tanh slope ``1 - a^2`` is a DETERMINISTIC function of the unit's own
    preactivation. So the raw ``f`` and ``eta`` below are guaranteed to be large
    for a trivial reason that carries no information: the unit's own nonlinearity
    rescales everything it receives. Every ratio statistic here -- ``snr``,
    ``reliability``, ``f_snr``, ``eta_snr`` -- divides that factor out exactly,
    because it multiplies the per-bin mean and the per-bin sd identically. Those
    are the statistics that answer the question; the raw pair is reported only so
    the size of the confound is visible.
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
    # Per (unit, state bin) signal-to-noise of the teaching signal, invariant to
    # any deterministic own-state gain. This IS the reliability, per state cell.
    snr = means / deviation.clamp_min(1e-30)
    # per (unit, bin) reliability: fraction of the squared teaching signal in
    # that state cell that is systematic rather than per-sample noise, in [0, 1].
    reliability = snr.square() / (1.0 + snr.square())
    # Standardising each bin by its own sd makes the within-bin variance exactly
    # one, so the between-bin sum of squares of the SNR is itself the F ratio:
    # "does this unit's signal reliability vary across its own state?"
    snr_mean = snr.mean(0)
    between_snr = per_bin * (snr - snr_mean).square().sum(0)
    f_snr = between_snr / (bins - 1)
    eta_snr = ((between_snr - (bins - 1)) / samples.clamp_min(1.0)).clamp_min(0.0)
    # THE CEILING, in squared SNR of a unit's aggregated teaching signal.
    #   uniform  -- any rule that treats a unit's own states alike: (mean_k snr)^2
    #   gate     -- the best NON-NEGATIVE per-state weighting, which is what a
    #               plasticity rule in [0, 1] is: it can suppress a state cell
    #               but never flip it, so it keeps only the same-sign cells
    #   oracle   -- the best signed weighting, mean_k snr_k^2; not a plasticity
    #               rule at all, quoted only as an absolute bound
    # No analytic bias correction is applied to these three: the permuted arm is
    # computed identically and IS the empirical null, which is the honest control.
    uniform = snr.mean(0).square()
    # A gate w >= 0 can suppress a state cell but never flip its sign, so its
    # optimum keeps only the same-sign cells: mean_k max(0, snr_k)^2.
    gate = snr.clamp_min(0.0).square().mean(0)
    oracle = snr.square().mean(0)
    return {"f": f, "eta": eta, "f_snr": f_snr, "eta_snr": eta_snr,
            "snr_abs": snr.abs().median(0).values,
            "snr_max": snr.abs().max(0).values,
            "attainable": (oracle - 1.0 / per_bin.clamp_min(1.0)).clamp_min(0.0),
            "uniform": uniform,
            "gain": (oracle - 1.0 / per_bin.clamp_min(1.0)).clamp_min(0.0)
            / uniform.clamp_min(1e-30),
            "gate_abs": (gate - 1.0 / per_bin.clamp_min(1.0)).clamp_min(0.0),
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
                     f"units > perm p95 {excess * 100:5.1f}% (chance 5)   "
                     f"max true {observed.max():9.{digits}f}")

    lines.append("  GAIN-INVARIANT (the question): does reliability vary with own state?")
    compare("F of SNR, per update", "f_snr_update")
    compare("F of SNR, pooled", "f_snr_pooled")
    compare("eta2 of SNR, pooled", "eta_snr_pooled", digits=6)
    compare("reliability range [0,1]", "reliability_pooled", digits=6)
    compare("|SNR| median over bins", "snr_abs_pooled", digits=5)
    lines.append("  RAW (confounded by the unit's own tanh slope; size of confound shown)")
    compare("F of raw signal, pooled", "f_pooled")
    compare("eta2 of raw, pooled", "eta_pooled", digits=6)
    lines.append(f"  {'log-sd of noise by state':<26} true "
                 f"{true['noise_spread_pooled'].median():9.4f}   permuted "
                 f"{perm['noise_spread_pooled'].median():9.4f}   "
                 f"log-sd of tanh slope by state {report['slope_spread'].median():7.4f}")
    lines.append("  CEILING in squared SNR of a unit's aggregated teaching signal.")
    lines.append("  Per-unit RATIOS are heavy-tailed (near-zero denominators), so the")
    lines.append("  headline is the ratio of MEDIANS; per-unit ratios follow for shape.")
    for name, key in (("oracle (signed)", "attainable_pooled"),
                      ("gate (w >= 0)", "gate_abs_pooled")):
        numerator, control = true[key].median(), perm[key].median()
        denominator, control_denominator = true["uniform_pooled"].median(), \
            perm["uniform_pooled"].median()
        lines.append(f"  {name:<26} true {numerator:.3e} / {denominator:.3e} = "
                     f"{numerator / denominator.clamp_min(1e-30):7.2f}x   "
                     f"permuted {control:.3e} / {control_denominator:.3e} = "
                     f"{control / control_denominator.clamp_min(1e-30):6.2f}x")
    compare("gate gain, per-unit median", "gate_gain_pooled")
    compare("oracle gain, per-unit med", "oracle_gain_pooled")
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
            # PERSISTENCE. A gate has to be ESTIMATED before it can be used, so
            # the pattern must outlive the window needed to measure it. Correlate
            # this stage's per-(state cell, unit) SNR with the previous stage's,
            # CENTRED WITHIN EACH UNIT: without centring, a unit's unconditional
            # signal level and its tanh-slope profile persist on their own and
            # inflate even the permuted arm. Centred, only the state-conditional
            # pattern remains, and the permuted arm is the zero reference.
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
    print("\n=== VERDICT: is a unit's signal reliability legible from its own state? ===")
    print("# F of SNR: 1.0 == no relation. eta2 of SNR: fraction of the reliability")
    print("# variation explained by own state, 0 == none. `perm` columns are the")
    print("# matched control that destroys ONLY the state->signal correspondence.")
    print(f"{'steps':>10} {'F_snr true':>11} {'F_snr perm':>11} {'ratio':>7} "
          f"{'eta2 true':>10} {'gate true':>10} {'gate perm':>10} "
          f"{'persist':>9} {'persist perm':>13} "
          f"{'rel range':>10} {'noise sd':>9} {'slope sd':>9}")
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
