# Sparse linear feature-selection diagnostic, not a per-neuron plasticity test.
#
# The linear task has iid Bernoulli inputs and homoscedastic target noise.
# The regime task exposes a global noise flag, allowing a conditional-reliability
# sanity check, but still has one linear output and no unit-specific conflicts.
# Neither establishes the family's state-conditioned per-perceptron premise.
#
# All methods receive the same training samples. Randomized evidence uses a
# separate RNG; gates read past evidence before ingesting the current residual.
# Final clean-target risk is also computed exactly from the Bernoulli moments,
# decomposed into signal reconstruction, distractor leakage and their mean cross
# term. Selectivity alone can reward a learner that recovers almost no signal.
# `oracle` is support-informed Adam, not a mathematical performance bound.
import math
from dataclasses import dataclass
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.runtime import configure_runtime

METHODS = ("sgd", "adam", "adamw", "energy", "snr", "statewiener", "mirror",
           "smoothgate", "softhinge", "softhinge_amp", "bayes",
           "softveto", "oracle")


@dataclass
class Args:
    task: str = "linear"
    """linear | regime | hidden"""
    method: str = "all"
    """one of sgd/adam/energy/snr/statewiener/oracle, or `all`"""
    input_dim: int = 4096
    signal_inputs: int = 1
    hidden_dim: int = 256
    regime_index: int = 1
    feature_prob: float = 0.01
    target_noise_std: float = math.sqrt(5.0)
    quiet_std: float = 0.5
    noisy_std: float = 5.0
    spike_prob: float = 0.01
    steps: int = 20_000
    seeds: int = 4
    """batched replicas; spread is descriptive, not a confidence interval"""
    lr: float = 1e-3
    weight_decay: float = 0.01
    """decoupled weight decay for the `adamw` baseline -- a real competitor
    here, since shrinking unused weights is itself a form of noise rejection"""
    plot: str = ""
    """if set, write a prediction figure to this path"""
    plot_window: int = 400
    """steps of prediction trace to draw"""
    eval_steps: int = 8192
    """independent frozen-model CLEAN-target evaluation samples"""
    gate_lr: float = 3e-2
    stat_beta: float = 0.999
    """EMA horizon for running statistics, counted in OBSERVATIONS not steps"""
    anchor: str = "quantile"
    """mean | geomean | quantile -- how the level's reference is set"""
    anchor_q: float = 0.99
    exponent: float = 1.0
    soft_exponent: float = 2.0
    """sharpness p of the graded gate (t^2/(t^2+z^2))^p"""
    adaptive_z: bool = False
    """estimate a threshold from past sign-randomized twin evidence, per replica.
    This is an empirical null heuristic, not calibrated false-discovery control;
    adaptive_q and the gate shape remain hyperparameters."""
    adaptive_q: float = 0.999
    evidence_decay: float = 0.0
    """per-active-observation exponential forgetting. Squared-evidence sums use
    squared decay factors, matching the variance of the weighted signed sum."""
    gls_evidence: bool = False
    """weight mirror evidence using the PREVIOUS residual-energy EWMA.
    This is an estimated weighting heuristic, not an oracle GLS guarantee."""
    gate_every: int = 50
    """refresh cadence for the mirror level (the ensemble sort is amortized)"""
    switch_back: float = 0.0
    """fraction of training after which the signal RETURNS to its first home.
    Tests whether retained evidence on a once-predictive coordinate is a
    liability or a prior that buys instant re-adaptation."""
    switch_at: float = 0.0
    """fraction of training after which the signal MOVES to `switch_to`"""
    switch_to: int = 2
    bayes_prior: float = 1.0
    """`bayes`: prior variance per connection."""
    bayes_q: float = 1e-4
    """`bayes`: process noise as a fraction of the prior; the rate of world change."""
    bayes_logit0: float = -8.3
    """`bayes`: prior inclusion log-odds."""
    bayes_logit_cap: float = 12.0
    """`bayes`: log-odds clamp; keeps inclusion reversible."""
    level_floor: float = 0.01
    """floor in the global factor of `softhinge_amp`; caps amplification at 1/floor."""
    hinge_sharpness: float = 24.0
    """softplus sharpness. This implementation can underflow to exact zero for
    sufficiently weak evidence; smooth in real arithmetic does not imply a
    nonzero numerical floor or guaranteed recovery after a switch."""
    gate_power: float = 4.0
    """`smoothgate`: steepness exponent. The gate is (t^2/(t^2+z^2))^p, which is
    differentiable EVERYWHERE -- no threshold, no hinge, no floor -- while still
    suppressing far below the level a hard zero achieves in effect. That matters
    because what the task requires is not an exact zero but a MAGNITUDE: with D
    distractors the output noise a learner absorbs scales with the rms gate level
    over them, so anything much below 1/sqrt(D) is indistinguishable from zero in
    consequence. At p=1 a null coordinate (t^2 ~ 1, z^2 = 25) still holds 0.04,
    which is above 1/sqrt(4095) = 0.016 and shows up as a residual noise floor.
    At p=4 it holds 2e-6. Same family of curve, differentiable, and it can be
    made state-conditional and meta-learned, which a hinge cannot."""
    veto_z: float = 4.0
    """fixed self-normalized evidence threshold; not an exact Gaussian z-score"""
    veto_floor: float = 0.0
    """level granted to a NON-admitted coordinate (0 = full veto)"""
    debias: bool = True
    """subtract the estimator's own variance floor via a sign-randomized twin"""
    weight_suppress: float = 8.0
    weight_inflate: float = 2.0
    seed: int = 1
    cuda: bool = True
    chunk_steps: int = 100
    """sequential observations per CUDA graph replay; must divide by gate_every"""
    eval_batch_size: int = 256
    """evaluation samples per vectorized forward pass"""
    output_dir: str = ""
    """optional run directory; default uses the repository runs/ convention"""


class Stream:
    """Block sampler; optimizer randomization never consumes these RNGs."""

    def __init__(self, args, device, regime, seed_offset=0):
        self.args = args
        self.regime = regime
        self.device = device
        seed = args.seed + seed_offset
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.noise_generator = torch.Generator(device=device).manual_seed(seed + 1_000_003)
        self.flag_generator = torch.Generator(device=device).manual_seed(seed + 2_000_003)
        self.initial = torch.zeros(args.input_dim, device=device)
        self.initial[:args.signal_inputs] = 1.0
        self.moved = torch.zeros_like(self.initial)
        self.moved[args.switch_to:args.switch_to + args.signal_inputs] = 1.0

    def support(self, steps):
        args = self.args
        moved = torch.zeros_like(steps, dtype=torch.bool)
        if args.switch_at:
            moved = steps >= int(args.steps * args.switch_at)
        if args.switch_back:
            moved = moved & (steps < int(args.steps * args.switch_back))
        return torch.where(moved.unsqueeze(1), self.moved, self.initial)

    def draw(self, count, start, *, frozen=False):
        args = self.args
        shape = (count, args.seeds, args.input_dim)
        x = (torch.rand(shape, device=self.device, generator=self.generator)
             < args.feature_prob).float()
        if self.regime:
            noisy = (torch.rand((count, args.seeds), device=self.device,
                                generator=self.flag_generator) < 0.5).float()
            x[:, :, args.regime_index] = noisy
            std = args.quiet_std + (args.noisy_std - args.quiet_std) * noisy
            spike = 0.0
        else:
            noisy = torch.zeros((count, args.seeds), device=self.device)
            std = args.target_noise_std
            u = torch.rand((count, args.seeds), device=self.device,
                           generator=self.flag_generator)
            spike = (u < args.spike_prob / 2).float() - (
                (u >= args.spike_prob / 2) & (u < args.spike_prob)).float()
        steps = torch.full((count,), args.steps, device=self.device) if frozen else (
            torch.arange(start, start + count, device=self.device))
        support = self.support(steps)
        clean = (x * support.unsqueeze(1)).sum(-1)
        noise = torch.randn((count, args.seeds), device=self.device,
                            generator=self.noise_generator) * std
        return x, clean + spike + noise, clean, noisy, support


def row_quantile(values, q):
    """Fixed-shape quantile without torch.quantile's CUDA scalar checks."""
    position = (values.shape[1] - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    ordered = values.sort(dim=1).values
    return ordered[:, lower:lower + 1].lerp(
        ordered[:, upper:upper + 1], position - lower)


class Gate:
    """Past-gradient evidence per input coordinate, not per hidden neuron.

    Energy does not encode gradient sign; conditional residual energies can
    nonetheless differ across active features. The randomized twin is a noisy
    null reference, not an exact unbiased correction after clipping/feedback.
    """

    def __init__(self, kind, shape, args, device, state_dim=2):
        self.kind = kind
        self.args = args
        self.mu = torch.zeros(shape, device=device)
        self.energy = torch.zeros(shape, device=device)
        self.mu_ctrl = torch.zeros(shape, device=device)
        self.mean_w = torch.zeros((*shape, state_dim), device=device)
        self.logn_w = torch.zeros((*shape, state_dim), device=device)
        self.mean_ctrl = torch.zeros((*shape, state_dim), device=device)
        self.ref = torch.ones((shape[0], 1), device=device)
        # [raw_signal, n_signal, raw_distract, n_distract,
        #  lvl_signal, lvl_distract, quiet_raw, n_quiet, noisy_raw, n_noisy,
        #  uniform_log, n_uniform]
        self.acc = torch.zeros(12, device=device)

    def statistic(self, grad, active, state, sign):
        args = self.args
        rate = (1.0 - args.stat_beta) * active
        keep = 1.0 - rate
        if self.kind == "statewiener":
            mu = (self.mean_w * state).sum(-1)
            log_n = (self.logn_w * state).sum(-1).clamp(-12.0, 12.0)
            var = log_n.exp()
            residual = grad - mu
            d_mu = (-residual / var) * active
            d_log_n = 0.5 * (1.0 - residual.square() / var) * active
            self.mean_w -= args.gate_lr * d_mu.unsqueeze(-1) * state
            self.logn_w -= args.gate_lr * d_log_n.unsqueeze(-1) * state
            signal = mu.square()
            if args.debias:
                # the twin sees the SAME state, noise, horizon and step size,
                # but a target whose true mean is exactly zero
                mu_c = (self.mean_ctrl * state).sum(-1)
                d_mu_c = (-(sign * grad - mu_c) / var) * active
                self.mean_ctrl -= args.gate_lr * d_mu_c.unsqueeze(-1) * state
                signal = (signal - mu_c.square()).clamp_min(0.0)
            return signal / (signal + var + 1e-12)
        signal = self.mu.square()
        if args.debias:
            signal = (signal - self.mu_ctrl.square()).clamp_min(0.0)
        raw = signal / (self.energy + 1e-12)
        if self.kind == "energy":
            raw = 1.0 / (1.0 + self.energy / (
                self.energy.mean(dim=1, keepdim=True) + 1e-12))
        # Return the PRE-observation statistic, consistently with statewiener.
        self.mu.mul_(keep).add_(rate * grad)
        self.energy.mul_(keep).add_(rate * grad.square())
        if args.debias:
            self.mu_ctrl.mul_(keep).add_(rate * sign * grad)
        return raw

    def level(self, raw, active):
        args = self.args
        seen = active
        if args.anchor == "mean":
            reference = (raw * seen).sum(1, keepdim=True) / seen.sum(1, keepdim=True).clamp_min(1.0)
        elif args.anchor == "geomean":
            logs = (raw.clamp_min(1e-30).log() * seen).sum(1, keepdim=True) / seen.sum(1, keepdim=True).clamp_min(1.0)
            reference = logs.exp()
        else:
            # anchor near the TOP of the distribution: the most reliable
            # coordinate keeps its step and everything else is suppressed
            # relative to it. With 4095/4096 coordinates being distractors, a
            # central anchor pins the reference at the distractor value.
            reference = row_quantile(raw, args.anchor_q)
        self.ref.mul_(args.stat_beta).add_((1.0 - args.stat_beta) * reference)
        level = (raw / self.ref.clamp_min(1e-12)).pow(args.exponent)
        return level.clamp(1.0 / args.weight_suppress, args.weight_inflate)

    def record(self, raw, level, active, noisy, support):
        sig_mask = active * support
        dis_mask = active * (1.0 - support)
        quiet = (1.0 - noisy).unsqueeze(1) * sig_mask
        loud = noisy.unsqueeze(1) * sig_mask
        self.acc.add_(torch.stack([
            (raw * sig_mask).sum(), sig_mask.sum(),
            (raw * dis_mask).sum(), dis_mask.sum(),
            (level * sig_mask).sum(), (level * dis_mask).sum(),
            (level * quiet).sum(), quiet.sum(),
            (level * loud).sum(), loud.sum(),
            (level.clamp_min(1e-30).log() * active).sum(), active.sum(),
        ]))


def adam_ratio(m, v, t):
    return (m / (1.0 - 0.9 ** t)) / ((v / (1.0 - 0.999 ** t)).sqrt() + 1e-5)


@torch.no_grad()
def run(args, method, device):
    if device.type != "cuda":
        raise ValueError("this diagnostic uses compiled CUDA graphs; CPU execution is unsupported")
    stream = Stream(args, device, args.task == "regime")
    control_rng = torch.Generator(device=device).manual_seed(args.seed + 3_000_017)
    shape = (args.seeds, args.input_dim)
    weight = torch.zeros(shape, device=device)
    m, v = torch.zeros_like(weight), torch.zeros_like(weight)
    running_sum, running_sq = torch.zeros_like(weight), torch.zeros_like(weight)
    mirror_sum, level_buf = torch.zeros_like(weight), torch.zeros_like(weight)
    resid_var = torch.ones((args.seeds, 1), device=device)
    post_w = torch.zeros_like(weight)
    post_var = torch.full_like(weight, args.bayes_prior)
    logit = torch.full_like(weight, args.bayes_logit0)
    noise_sum = torch.zeros((args.seeds, 1), device=device)
    noise_n = torch.zeros((), device=device)
    step_count = torch.zeros((), device=device)
    train_error = torch.zeros(args.seeds, device=device)
    gate = Gate(method, shape, args, device) if method in {
        "energy", "snr", "statewiener"} else None
    evidence_method = method in {
        "mirror", "softveto", "smoothgate", "softhinge", "softhinge_amp"}

    def train_step(x, y, noisy, support, sign, refresh):
        step_count.add_(1)
        if method == "oracle":
            # Support-informed Adam: no stale output or momentum survives a move.
            # This reference knows the support, not coefficients or target noise.
            weight.mul_(support)
            m.mul_(support)
            v.mul_(support)
        prediction = (weight * x).sum(1)
        clean = (x * support).sum(1)
        train_error.add_((prediction - clean).square())
        delta = (y - prediction).unsqueeze(1)
        grad = -delta * x
        active = (x != 0).float()
        if method == "sgd":
            weight.add_(grad, alpha=-args.lr)
            return
        if method == "bayes":
            # Diagonal approximate filter, NOT an exact multivariate posterior.
            # All likelihoods use the pre-observation residual-noise estimate.
            pi = logit.sigmoid()
            contribution = post_w * x
            err_out = delta + pi * contribution
            err_in = err_out - contribution
            logit.add_((err_out.square() - err_in.square()) / (2 * resid_var))
            logit.clamp_(-args.bayes_logit_cap, args.bayes_logit_cap)
            gain = post_var * x / ((post_var * x.square()).sum(1, keepdim=True) + resid_var)
            post_w.add_(gain * delta)
            post_var.sub_(gain * x * post_var).clamp_min_(1e-8)
            post_var.add_(args.bayes_q * args.bayes_prior)
            weight.copy_(logit.sigmoid() * post_w)
            noise_sum.add_(delta.square())
            noise_n.add_(1)
            resid_var.copy_((noise_sum / noise_n).clamp_min(1e-12))
            return
        m.mul_(0.9).add_(grad, alpha=0.1)
        v.mul_(0.999).addcmul_(grad, grad, value=0.001)
        update = args.lr * adam_ratio(m, v, step_count)
        if method == "oracle":
            update = update * support
        elif evidence_method:
            # Predict the gate from PAST evidence; ingest this residual only
            # after selecting its level. In particular GLS cannot see delta^2.
            denominator = running_sq.clamp_min(1e-30)
            t_sq = running_sum.square() / denominator
            twin_sq = mirror_sum.square() / denominator
            if method == "mirror":
                if refresh:
                    observed = t_sq.sqrt()
                    null_sorted = twin_sq.sqrt().sort(dim=1).values
                    obs_sorted = observed.sort(dim=1).values
                    false_ge = args.input_dim - torch.searchsorted(null_sorted, observed)
                    total_ge = (args.input_dim - torch.searchsorted(
                        obs_sorted, observed)).clamp_min(1)
                    # Empirical tail ratio; dependent coordinates and adaptive
                    # residuals preclude a formal FDP-control interpretation.
                    level_buf.copy_((1 - false_ge.float() / total_ge).clamp(0, 1))
            else:
                z_sq = row_quantile(twin_sq, args.adaptive_q).clamp_min(1.0) \
                    if args.adaptive_z else args.veto_z ** 2
                if method in {"softveto", "smoothgate"}:
                    power = args.soft_exponent if method == "softveto" else args.gate_power
                    level_buf.copy_((t_sq / (t_sq + z_sq)).pow(power))
                else:
                    raw = 1 - z_sq / t_sq.clamp_min(1e-30)
                    certainty = F.softplus(args.hinge_sharpness * raw) / args.hinge_sharpness
                    if method == "softhinge_amp":
                        certainty = certainty / (
                            certainty.mean(1, keepdim=True) + args.level_floor)
                    level_buf.copy_(certainty)
            update = update * level_buf
            weighted = grad / resid_var.clamp_min(1e-8) \
                if method == "mirror" and args.gls_evidence else grad
            keep = 1 - args.evidence_decay * active
            running_sum.mul_(keep).add_(weighted)
            running_sq.mul_(keep.square()).add_(weighted.square())
            mirror_sum.mul_(keep).add_(weighted * sign)
            if method == "mirror" and args.gls_evidence:
                resid_var.mul_(0.99).add_(delta.square(), alpha=0.01)
        elif gate is not None:
            state = torch.stack([torch.ones_like(x), noisy.unsqueeze(1).expand_as(x)], -1)
            raw = gate.statistic(grad, active, state, sign)
            level = gate.level(raw, active)
            update = update * level
            gate.record(raw, level, active, noisy, support)
        if method == "adamw":
            weight.mul_(1 - args.lr * args.weight_decay)
        weight.sub_(update)

    # Fixed-shape fused updates inside a manual multi-step graph. Sampling stays
    # outside capture, with separate generators shared identically across arms.
    # Warmup/capture mutate scratch state only; every tensor is restored before
    # the first real observation. No training samples are used for compilation.
    step_fn = torch.compile(train_step, fullgraph=True, dynamic=False,
                            options={"triton.cudagraphs": False})
    count = args.chunk_steps
    inputs = (
        torch.zeros((count, *shape), device=device),
        torch.zeros((count, args.seeds), device=device),
        torch.zeros((count, args.seeds), device=device),
        torch.zeros((count, args.input_dim), device=device),
        torch.ones((count, args.seeds, 1), device=device),
    )
    states = [weight, m, v, running_sum, running_sq, mirror_sum, level_buf,
              resid_var, post_w, post_var, logit, noise_sum, noise_n,
              step_count, train_error]
    if gate is not None:
        states += [value for value in vars(gate).values() if isinstance(value, torch.Tensor)]
    snapshots = [value.clone() for value in states]

    def block():
        for offset in range(count):
            step_fn(*(value[offset] for value in inputs),
                    offset % args.gate_every == 0)

    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    started = time.perf_counter()
    try:
        with torch.cuda.stream(capture_stream):
            step_fn(*(value[0] for value in inputs), True)
            step_fn(*(value[0] for value in inputs), False)
        capture_stream.synchronize()
        with torch.cuda.graph(graph, stream=capture_stream):
            block()
        capture_stream.synchronize()
    finally:
        capture_stream.synchronize()
        for value, saved in zip(states, snapshots):
            value.copy_(saved)
        torch.cuda.current_stream(device).synchronize()
    compile_seconds = time.perf_counter() - started
    del snapshots

    started = time.perf_counter()
    for start in range(1, args.steps + 1, count):
        length = min(count, args.steps - start + 1)
        x, y, _clean, noisy, support = stream.draw(length, start)
        signs = torch.where(torch.rand((length, args.seeds, 1), device=device,
                                      generator=control_rng) < 0.5, -1.0, 1.0)
        for destination, source in zip(inputs, (x, y, noisy, support, signs)):
            destination[:length].copy_(source)
        if length == count:
            graph.replay()
        else:
            # The final partial block is exact: never train on padded samples.
            for offset in range(length):
                step_fn(*(value[offset] for value in inputs),
                        offset % args.gate_every == 0)
    torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started

    support = stream.support(torch.tensor([args.steps], device=device))[0]
    distractor = 1 - support
    probabilities = torch.full((args.input_dim,), args.feature_prob, device=device)
    if args.task == "regime":
        probabilities[args.regime_index] = 0.5
    # Independent Bernoulli features: E[(a.x)^2] = Var(a.x) + E[a.x]^2.
    # FP64 here only, for stable small-risk and cancellation diagnostics.
    p = probabilities.double()

    def power(coefficients):
        a = coefficients.double()
        return (a.square() * p * (1 - p)).sum(-1) + (a * p).sum(-1).square()

    signal_error = (weight - 1) * support
    junk_weight = weight * distractor
    signal_reconstruction = power(signal_error)
    distractor_leakage = power(junk_weight)
    cross = 2 * (signal_error.double() * p).sum(-1) * (junk_weight.double() * p).sum(-1)
    exact_mse = power(weight - support)
    zero_mse = power(support).expand(args.seeds)
    mean_mse = (support.double() * p * (1 - p)).sum().expand(args.seeds)
    signal = (weight * support).sum(1) / args.signal_inputs
    distract = junk_weight.square().sum(1).div(
        max(args.input_dim - args.signal_inputs, 1)).sqrt()
    out = {
        "signal": signal, "distract": distract,
        "ratio": signal / distract.clamp_min(1e-30),
        "exact_mse": exact_mse, "exact_trivial": zero_mse,
        "exact_mean_predictor_mse": mean_mse,
        "signal_reconstruction_mse": signal_reconstruction,
        "distractor_leakage_mse": distractor_leakage,
        "signal_distractor_cross": cross,
        "signal_coefficient_rmse": signal_error.square().sum(1).div(args.signal_inputs).sqrt(),
        "prequential_clean_mse": train_error / args.steps,
        "compile_seconds": compile_seconds, "train_seconds": train_seconds,
        "samples_per_second": args.steps * args.seeds / train_seconds,
    }
    stale = ((stream.initial + stream.moved).clamp_max(1) - support).clamp_min(0) \
        if args.switch_at else torch.zeros_like(support)
    pruned = weight * (1 - stale)
    if args.switch_at:
        out["stale"] = (weight.abs() * stale).sum(1) / stale.sum().clamp_min(1)
        out["exact_ablated_mse"] = power(pruned - support)

    # Fresh, common frozen-model samples; never select an LR/method on this
    # stream. CLI runs fixed hyperparameters only; any external sweep is
    # exploratory unless selected on separate training realizations first.
    evaluation = Stream(args, device, args.task == "regime", seed_offset=10_000_019)
    squared = torch.zeros(args.seeds, device=device, dtype=torch.float64)
    trivial = torch.zeros_like(squared)
    ablated = torch.zeros_like(squared)
    active_error, active_count = torch.zeros_like(squared), torch.zeros_like(squared)
    inactive_error, inactive_count = torch.zeros_like(squared), torch.zeros_like(squared)
    window = min(args.plot_window, args.eval_steps) if args.plot else 0
    trace = torch.zeros((3, window), device=device)
    for start in range(0, args.eval_steps, args.eval_batch_size):
        length = min(args.eval_batch_size, args.eval_steps - start)
        x, y, clean, _noisy, _support = evaluation.draw(length, 0, frozen=True)
        prediction = (x * weight).sum(-1)
        error = (prediction.double() - clean).square()
        squared.add_(error.sum(0))
        trivial.add_(clean.double().square().sum(0))
        ablated.add_(((x * pruned).sum(-1).double() - clean).square().sum(0))
        active = clean != 0
        active_error.add_((error * active).sum(0))
        active_count.add_(active.sum(0))
        inactive_error.add_((error * ~active).sum(0))
        inactive_count.add_((~active).sum(0))
        if start < window:
            take = min(length, window - start)
            trace[:, start:start + take].copy_(
                torch.stack([prediction[:take, 0], clean[:take, 0], y[:take, 0]]))
    out.update({
        "test_mse": squared / args.eval_steps, "trivial": trivial / args.eval_steps,
        "signal_active_mse": torch.where(active_count > 0, active_error / active_count, float("nan")),
        "signal_inactive_mse": torch.where(inactive_count > 0, inactive_error / inactive_count, float("nan")),
        "signal_active_samples": active_count, "signal_inactive_samples": inactive_count,
    })
    if args.switch_at:
        out["ablated_mse"] = ablated / args.eval_steps
    if evidence_method:
        out["lvl_sig"] = (level_buf * support).sum(1) / args.signal_inputs
        out["lvl_dis"] = (level_buf * distractor).square().sum(1).div(
            max(args.input_dim - args.signal_inputs, 1)).sqrt()
    if gate is not None:
        out["acc"] = gate.acc
    if window:
        out["trace"] = trace
    return out


def report(args, method, out):
    signal = out["signal"].mean().item()
    distract = out["distract"].mean().item()
    ratio = out["ratio"]
    spread = ratio.std().item() if args.seeds > 1 else 0.0
    line = (f"  {method:>12s}  signal={signal:>8.4f}  distractor={distract:>9.5f}  "
            f"selectivity={ratio.mean().item():>7.2f} +-{spread:>5.2f}")
    # The trivial predictor (always zero) has mse = E[clean^2]. Reporting the
    # ratio to it is the honest scale: the blog's whole point is that SGD-family
    # learners end up WORSE than useless on an unfiltered stream.
    trivial = max(out["trivial"].mean().item(), 1e-12)
    mse = out["test_mse"].mean().item()
    line += (f"\n{'':16s}TEST vs clean target: mse={mse:.5f}  "
             f"= {mse / trivial:>6.2f}x the zero-predictor "
             f"({'BEATS' if mse < trivial else 'WORSE THAN'} predicting nothing)")
    if "stale" in out:
        abl = out["ablated_mse"].mean().item()
        line += (f"\n{'':16s}signal MOVED: current coord w={signal:.4f}  "
                 f"stale coord |w|={out['stale'].mean().item():.4f}"
                 f"\n{'':16s}stale-pruned mse={abl:.5f} "
                 f"(retaining it costs {mse - abl:+.5f}, "
                 f"{100.0 * (mse - abl) / mse:+.1f}% of error)")
    if "lvl_sig" in out:
        line += (f"\n{'':16s}mirror level: signal={out['lvl_sig'].mean().item():.4f}  "
                 f"distractor rms={out['lvl_dis'].mean().item():.4f}")
    if "acc" in out:
        a = out["acc"].tolist()
        raw_sig = a[0] / max(a[1], 1.0)
        raw_dis = a[2] / max(a[3], 1.0)
        lvl_sig = a[4] / max(a[1], 1.0)
        lvl_dis = a[5] / max(a[3], 1.0)
        uniform = math.exp(a[10] / max(a[11], 1.0))
        line += (f"\n{'':16s}raw statistic: signal={raw_sig:.3e} "
                 f"distractor={raw_dis:.3e}  separation={raw_sig / max(raw_dis, 1e-30):.2f}x")
        line += (f"\n{'':16s}realized level: signal={lvl_sig:.3f} "
                 f"distractor={lvl_dis:.3f}  separation={lvl_sig / max(lvl_dis, 1e-30):.2f}x"
                 f"  uniform={uniform:.3f}")
        if a[7] > 0 and a[9] > 0:
            quiet = a[6] / a[7]
            loud = a[8] / a[9]
            line += (f"\n{'':16s}by REGIME (signal-coord level): "
                     f"quiet={quiet:.3f} noisy={loud:.3f}"
                     f"  separation={quiet / max(loud, 1e-30):.2f}x")
    print(line, flush=True)


LABELS = {"veto": "admission\nveto", "softveto": "graded\nplasticity",
          "gradedveto": "graded\ncertainty\n(hinge)",
          "smoothgate": "graded\ncertainty\n(smooth)",
          "sgd": "SGD", "adam": "Adam", "adamw": "AdamW", "energy": "energy gate\n(v8/v9 rule)",
          "snr": "state\nplasticity", "statewiener": "state\nplasticity\n(conditioned)",
          "oracle": "Oracle"}


def draw_figure(args, traces):
    """Prediction traces in the style of the Oak blog's figures.

    One panel per learner, predictions only, on a fixed +-1.5 axis. The
    learnable target is drawn once at the top for reference: it is zero except
    for isolated spikes to 1.0, so a learner that has absorbed noise shows
    visible activity everywhere and a learner that has not is flat between
    spikes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [m for m in METHODS if m in traces]
    width = traces[order[0]].shape[1]
    rows = len(order) + 1
    figure, axes = plt.subplots(rows, 1, sharex=True, figsize=(9.5, 1.35 * rows))
    steps = range(width)

    def style(axis, label, color):
        axis.set_ylim(-1.5, 1.5)
        axis.set_yticks([-1.5, 0.0, 1.5])
        axis.tick_params(labelsize=8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center",
                        labelpad=14)

    style(axes[0], "Learnable\ntarget", None)
    axes[0].plot(steps, traces[order[0]][1], color="#2E7D32", linewidth=1.0)
    for axis, method in zip(axes[1:], order):
        style(axis, f"Predictions\nlearned by\n{LABELS.get(method, method)}", None)
        axis.plot(steps, traces[method][0], color="#F4511E", linewidth=0.8)

    axes[-1].set_xticks([0, width // 2, width])
    axes[-1].set_xticklabels(["t", f"t+{width // 2}", f"t+{width}"])
    axes[-1].set_xlabel("Time", fontsize=9)
    figure.tight_layout()
    figure.savefig(args.plot, dpi=150, facecolor="white")
    print(f"  figure written to {args.plot}")


def main():
    args = tyro.cli(Args)
    if args.task not in {"linear", "regime", "hidden"}:
        raise ValueError("task must be linear, regime or hidden")
    if args.task == "hidden":
        raise NotImplementedError("hidden stream is served by the dense variant")
    if args.regime_index < args.signal_inputs:
        raise ValueError("the regime flag must not overlap the signal inputs")
    chosen = METHODS if args.method == "all" else tuple(args.method.split(","))
    for name in chosen:
        if name not in METHODS:
            raise ValueError(f"method must be `all` or a comma-separated subset of {METHODS}")
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest",
                      allow_tf32=False)
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda" if args.cuda else "cpu")

    print(f"task={args.task} steps={args.steps} seeds={args.seeds} "
          f"input_dim={args.input_dim} anchor={args.anchor}")
    print("selectivity = w[signal] / rms w[distractors]; +- is the spread over seeds")
    traces = {}
    for method in chosen:
        out = run(args, method, device)
        report(args, method, out)
        if "trace" in out:
            traces[method] = out["trace"].cpu().numpy()
    if args.plot and traces:
        draw_figure(args, traces)


if __name__ == "__main__":
    main()
