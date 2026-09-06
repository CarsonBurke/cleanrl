# Noisy-stream credit-assignment diagnostic. Seconds per method, not minutes.
#
# PURPOSE. A benchmark score cannot say WHICH capability a mechanism has, so
# these streams make one capability the whole task. Nothing here is RL-specific:
# the same question ("may this unit move on this sample?") is what an LLM or a
# time-series learner faces on unfiltered data.
#
# STREAMS
#   linear  -- the Oak/Sutton stream. 4096 Bernoulli(p) inputs, only index 0
#              predictive, target = x[0] + (+-1 spike) + N(0, 5). Tests CREDIT
#              ASSIGNMENT ACROSS INPUTS. Its noise is homoscedastic and its only
#              state is active/inactive, so it CANNOT test state-dependence.
#   regime  -- index `regime_index` is an observable Bernoulli(0.5) flag that
#              does not shift the target mean but multiplies its noise
#              (quiet_std vs noisy_std). Tests STATE-DEPENDENT reliability: the
#              "this bar is unreliable" case. A running per-coordinate statistic
#              cannot express it because it averages over regimes.
#   hidden  -- 4096 inputs -> ReLU hidden -> scalar, leading inputs predictive.
#
# MEASUREMENT DISCIPLINE (learned the hard way, twice)
#   * The RAW statistic is reported separately from the REALIZED level. With
#     4095/4096 coordinates being distractors, anchoring a level on its
#     arithmetic mean pins the reference AT the distractor value, so the signal
#     saturates the envelope and the measured separation is the envelope's
#     ceiling no matter how good the statistic is. Conflating the two produced a
#     bogus "2.49x" here before it was caught.
#   * The realized geometric mean of the level is reported as `uniform`. A level
#     with a drifting uniform component is a learning-rate change wearing a
#     mechanism's clothes; this family has manufactured that fake win four
#     times.
#   * Every gated method applies its level POST-optimizer, because a
#     pre-optimizer magnitude is cancelled by Adam (measured: 0.897 realized for
#     a requested 0.125x on one batch, 1.00 sustained).
#
# THE IDENTITY THAT KILLS A WHOLE CLASS OF IDEA. With mu and sigma^2 estimated
# as running moments of one gradient stream, the Wiener gain
#     mu^2 / (mu^2 + sigma^2) = mu^2 / (mu^2 + (E[g^2] - mu^2)) = mu^2 / E[g^2]
# so "fit both moments" is IDENTICALLY the uncentered SNR -- verified
# bit-identical. Only a state-CONDITIONED conditional mean breaks it, and only
# on a stream whose predictability varies with state.
import math
from dataclasses import dataclass

import torch
import tyro

from cleanrl.shared.runtime import configure_runtime

METHODS = ("sgd", "adam", "adamw", "energy", "snr", "statewiener", "veto",
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
    """independent replicas run as a batch dimension -- free error bars"""
    lr: float = 1e-3
    weight_decay: float = 0.01
    """decoupled weight decay for the `adamw` baseline -- a real competitor
    here, since shrinking unused weights is itself a form of noise rejection"""
    plot: str = ""
    """if set, write a prediction figure to this path"""
    plot_window: int = 400
    """steps of prediction trace to draw"""
    eval_steps: int = 8192
    """fresh samples used for the CLEAN-target test error"""
    gate_lr: float = 3e-2
    stat_beta: float = 0.999
    """EMA horizon for running statistics, counted in OBSERVATIONS not steps"""
    anchor: str = "quantile"
    """mean | geomean | quantile -- how the level's reference is set"""
    anchor_q: float = 0.99
    exponent: float = 1.0
    veto_z: float = 4.0
    """admission threshold in sigmas; sets the false-admission rate directly"""
    veto_floor: float = 0.0
    """level granted to a NON-admitted coordinate (0 = full veto)"""
    debias: bool = True
    """subtract the estimator's own variance floor via a sign-randomized twin"""
    weight_suppress: float = 8.0
    weight_inflate: float = 2.0
    seed: int = 1
    cuda: bool = True


class Stream:
    """Fully tensorised sampler. No host sync, so the loop stays on device."""

    def __init__(self, args, device, regime):
        self.args = args
        self.regime = regime
        self.generator = torch.Generator(device=device).manual_seed(args.seed)
        self.shape = (args.seeds, args.input_dim)
        self.device = device

    def draw(self):
        args = self.args
        x = (torch.rand(self.shape, device=self.device, generator=self.generator)
             < args.feature_prob).to(torch.float32)
        clean = x[:, : args.signal_inputs].sum(1)
        if self.regime:
            noisy = (torch.rand((args.seeds,), device=self.device,
                                generator=self.generator) < 0.5).to(torch.float32)
            x[:, args.regime_index] = noisy
            std = args.quiet_std + (args.noisy_std - args.quiet_std) * noisy
            spike = torch.zeros_like(clean)
        else:
            noisy = torch.zeros((args.seeds,), device=self.device)
            std = torch.full((args.seeds,), args.target_noise_std, device=self.device)
            hit = (torch.rand((args.seeds,), device=self.device,
                              generator=self.generator) < args.spike_prob)
            sign = torch.where(
                torch.rand((args.seeds,), device=self.device,
                           generator=self.generator) < 0.5, -1.0, 1.0)
            spike = hit.to(torch.float32) * sign
        noise = torch.randn((args.seeds,), device=self.device,
                            generator=self.generator) * std
        return x, clean + spike + noise, clean, noisy


class Gate:
    """Per-coordinate plasticity statistic, applied to the REALIZED step.

    Three statistics share one code path so they are compared under identical
    anchoring, envelope and application point:
      energy      -- inverse predicted energy, 1/E[g^2]. The v8/v9 rule. Blind
                     to signal: a distractor's energy is delta^2 and so is the
                     signal's.
      snr         -- mu^2/E[g^2] from running moments, per OBSERVATION. Matched
                     horizons matter: with beta1 != beta2 the ratio measures
                     burstiness, and a 1%-sparse feature is otherwise averaged
                     almost entirely over its own absence.
      statewiener -- mu(s)^2/(mu(s)^2 + n(s)) from a per-coordinate affine
                     readout over the STATE, both moments fitted by the
                     heteroscedastic Gaussian NLL with analytic gradients.

    PAIRED SIGN-RANDOMIZED CONTROL (`--debias`). Squaring an ESTIMATED mean is
    biased: E[mu_hat^2] = mu^2 + Var(mu_hat), and Var(mu_hat) grows with the
    noise, so mu_hat^2 is inflated exactly where the data is least reliable.
    Measured consequence: on the regime stream the undebiased statistic gets
    reliability BACKWARDS (0.79x, i.e. MORE plasticity in the noisy regime).
    The fix runs a second, identical estimator on the sign-randomized stream
    eps_t * g_t, eps_t in {-1,+1} shared across coordinates. Its true mean is
    exactly zero while its energy, sparsity, horizon and step size are
    identical, so its square IS this estimator's own variance floor, measured
    rather than assumed. The debiased signal is max(mu_hat^2 - mu_ctrl^2, 0),
    which is not expressible as any ratio of moments of g.
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
        self.ref = torch.ones((), device=device)
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
        self.mu.mul_(keep).add_(rate * grad)
        self.energy.mul_(keep).add_(rate * grad.square())
        if self.kind == "energy":
            # inverse energy, rescaled to a comparable [0, 1] range
            return 1.0 / (1.0 + self.energy / (self.energy.mean() + 1e-12))
        signal = self.mu.square()
        if args.debias:
            self.mu_ctrl.mul_(keep).add_(rate * sign * grad)
            signal = (signal - self.mu_ctrl.square()).clamp_min(0.0)
        return signal / (self.energy + 1e-12)

    def level(self, raw, active):
        args = self.args
        seen = active
        if args.anchor == "mean":
            reference = (raw * seen).sum() / seen.sum().clamp_min(1.0)
        elif args.anchor == "geomean":
            logs = (raw.clamp_min(1e-30).log() * seen).sum() / seen.sum().clamp_min(1.0)
            reference = logs.exp()
        else:
            # anchor near the TOP of the distribution: the most reliable
            # coordinate keeps its step and everything else is suppressed
            # relative to it. With 4095/4096 coordinates being distractors, a
            # central anchor pins the reference at the distractor value.
            reference = torch.quantile(raw.flatten().float(), args.anchor_q)
        self.ref.mul_(args.stat_beta).add_((1.0 - args.stat_beta) * reference)
        level = (raw / self.ref.clamp_min(1e-12)).pow(args.exponent)
        return level.clamp(1.0 / args.weight_suppress, args.weight_inflate)

    def record(self, raw, level, active, noisy, signal_inputs):
        sig_mask = active[:, :signal_inputs]
        dis_mask = active[:, signal_inputs:]
        self.acc[0] += (raw[:, :signal_inputs] * sig_mask).sum()
        self.acc[1] += sig_mask.sum()
        self.acc[2] += (raw[:, signal_inputs:] * dis_mask).sum()
        self.acc[3] += dis_mask.sum()
        self.acc[4] += (level[:, :signal_inputs] * sig_mask).sum()
        self.acc[5] += (level[:, signal_inputs:] * dis_mask).sum()
        # restricted to the SIGNAL coordinates: averaged over all 4096 the
        # ratio is dominated by distractors whose debiased signal is clamped to
        # exactly zero in BOTH regimes, so it reported numerical noise
        quiet = (1.0 - noisy).unsqueeze(1) * sig_mask
        loud = noisy.unsqueeze(1) * sig_mask
        self.acc[6] += (level[:, :signal_inputs] * quiet).sum()
        self.acc[7] += quiet.sum()
        self.acc[8] += (level[:, :signal_inputs] * loud).sum()
        self.acc[9] += loud.sum()
        self.acc[10] += (level.clamp_min(1e-30).log() * active).sum()
        self.acc[11] += active.sum()


def adam_ratio(m, v, t):
    return (m / (1.0 - 0.9 ** t)) / ((v / (1.0 - 0.999 ** t)).sqrt() + 1e-5)


def run(args, method, device):
    regime = args.task == "regime"
    stream = Stream(args, device, regime)
    shape = (args.seeds, args.input_dim)
    weight = torch.zeros(shape, device=device)
    m = torch.zeros(shape, device=device)
    v = torch.zeros(shape, device=device)
    oracle_mask = torch.zeros(args.input_dim, device=device)
    oracle_mask[: args.signal_inputs] = 1.0
    gate = Gate(method, shape, args, device) if method in {"energy", "snr", "statewiener"} \
        else None
    # A soft multiplier cannot suppress: with 4095 distractors the rms level is
    # set by the luckiest few, so a right tail of any size leaks. The veto is
    # therefore BINARY, and its threshold is a calibrated false-admission rate
    # rather than a tuned scale. Self-normalised sum: under the null "this
    # coordinate does not predict the residual", P/sqrt(Q) is asymptotically
    # N(0,1) with no horizon, no reference and no learned estimator.
    running_sum = torch.zeros(shape, device=device)
    running_sq = torch.zeros(shape, device=device)
    window = min(args.plot_window, args.steps) if args.plot else 0
    trace_start = args.steps - window
    trace = torch.zeros((3, max(window, 1)), device=device)

    for step in range(1, args.steps + 1):
        x, y, _clean, noisy = stream.draw()
        prediction = (weight * x).sum(1)
        delta = y - prediction
        grad = -delta.unsqueeze(1) * x
        if window and step > trace_start:
            index = step - trace_start - 1
            trace[0, index] = prediction[0]
            trace[1, index] = _clean[0]
            trace[2, index] = y[0]

        if method == "sgd":
            weight -= args.lr * grad
            continue
        m.mul_(0.9).add_(grad, alpha=0.1)
        v.mul_(0.999).addcmul_(grad, grad, value=0.001)
        update = args.lr * adam_ratio(m, v, step)
        if method == "oracle":
            update = update * oracle_mask
        elif method == "softveto":
            # SMOOTH and everywhere-differentiable in the evidence, with NO
            # floor: level = (t^2/(t^2+z^2))^p. This is a graded plasticity
            # rule, not an admission test -- it isolates whether the veto's win
            # came from being BINARY or merely from having enough dynamic range.
            running_sum += grad
            running_sq += grad.square()
            t_sq = running_sum.square() / running_sq.clamp_min(1e-30)
            update = update * (t_sq / (t_sq + args.veto_z ** 2)).pow(args.exponent)
        elif method == "veto":
            running_sum += grad
            running_sq += grad.square()
            t_stat = running_sum.abs() / running_sq.sqrt().clamp_min(1e-30)
            admitted = (t_stat > args.veto_z).to(torch.float32)
            update = update * (args.veto_floor + (1.0 - args.veto_floor) * admitted)
        elif gate is not None:
            active = (x != 0).to(torch.float32)
            # the coordinate's own state: is it active, and what regime is this
            state = torch.stack([torch.ones_like(x),
                                 noisy.unsqueeze(1).expand_as(x)], dim=-1)
            sign = torch.where(
                torch.rand((args.seeds, 1), device=device,
                           generator=stream.generator) < 0.5, -1.0, 1.0)
            raw = gate.statistic(grad, active, state, sign)
            level = gate.level(raw, active)
            update = update * level
            gate.record(raw, level, active, noisy, args.signal_inputs)
        weight -= update
        if method == "adamw":
            weight -= args.lr * args.weight_decay * weight

    # THE TEST: error against the NOISE-FREE target on fresh samples. This is
    # the blog's actual claim -- learn only the learnable part -- whereas
    # selectivity is a proxy read off the weights.
    with torch.no_grad():
        squared = torch.zeros((args.seeds,), device=device)
        variance = torch.zeros((args.seeds,), device=device)
        for _ in range(args.eval_steps):
            x, _y, clean, _n = stream.draw()
            squared += ((weight * x).sum(1) - clean).square()
            variance += clean.square()
        test_mse = squared / args.eval_steps
        clean_power = variance / args.eval_steps

    signal = weight[:, : args.signal_inputs].mean(1)
    distract = weight[:, args.signal_inputs:].square().mean(1).sqrt()
    out = {"signal": signal, "distract": distract, "ratio": signal / distract,
           "test_mse": test_mse, "trivial": clean_power}
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
    trivial = out["trivial"].mean().item()
    mse = out["test_mse"].mean().item()
    line += (f"\n{'':16s}TEST vs clean target: mse={mse:.5f}  "
             f"= {mse / trivial:>6.2f}x the zero-predictor "
             f"({'BEATS' if mse < trivial else 'WORSE THAN'} predicting nothing)")
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
    if args.method != "all" and args.method not in METHODS:
        raise ValueError(f"method must be `all` or one of {METHODS}")
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest",
                      allow_tf32=False)
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda" if args.cuda else "cpu")

    print(f"task={args.task} steps={args.steps} seeds={args.seeds} "
          f"input_dim={args.input_dim} anchor={args.anchor}")
    print("selectivity = w[signal] / rms w[distractors]; +- is the spread over seeds")
    traces = {}
    for method in (METHODS if args.method == "all" else (args.method,)):
        out = run(args, method, device)
        report(args, method, out)
        if "trace" in out:
            traces[method] = out["trace"].cpu().numpy()
    if args.plot and traces:
        draw_figure(args, traces)


if __name__ == "__main__":
    main()
