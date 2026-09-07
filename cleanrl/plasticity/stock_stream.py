"""Mirror plasticity on real market data: one-pass online regression, batch 1.

WHY THIS TASK. The synthetic diagnostic is honest but planted: I know which
coordinate predicts. Market data has the property the mechanism was built for --
mostly-unpredictable targets, mostly-useless features, non-stationarity -- but
the noise is intrinsic and unknown, so a raw error number cannot by itself say
whether a learner found signal or absorbed noise.

THE VERIFICATION THAT WORKS ANYWAY. Run every learner twice: once on the real
stream, once with the targets PERMUTED in time. The permutation destroys all
predictability while preserving the marginal distribution of the target, so it is
a ground-truth null. A learner that only learns the learnable part must land at
the zero-predictor on permuted data (prequential ratio 1.0) and below it on real
data. A learner that absorbs noise beats nothing and is worse than 1.0 on BOTH.
The real-minus-permuted gap is the only claim this data can support, and it is
exactly the claim at issue.

PROTOCOL. Chronological single pass, batch size 1, prequential (predict-then-
update) error, so every prediction is out-of-sample by construction and there is
no train/test split to leak through. Features are causal: each is a function of
strictly past bars, scaled by a trailing EWMA so the input distribution is
stationary without look-ahead.

LEARNING-RATE CONTROL. Every method is swept over the same LR grid and reported
at ITS OWN best LR. A plasticity rule that merely rescales the effective step
would otherwise show up as a win; this family has already produced four such
false positives (see `FAMILY.md`).

WHY THE NULL WAS NOT NULL, AND THE TWO FIXES. With the weights pinned at zero
(`--lr-grid 0.0`) the only gradient is `-y x`, so on a PERMUTED target the level
must collapse to zero. It did not: 0.696. Two causes, both measured here.

  1. NO MEAN CENTRING. `E[y x_i] = mu_y m_i` is nonzero whenever neither the
     target nor the channel is mean-zero, and it survives any permutation of the
     target. `--center` subtracts a TRAILING EWMA mean from both. Alone it takes
     the permuted level 0.696 -> 0.344.
  2. SERIAL DEPENDENCE -- FALSIFIED, MEASURED, KEPT AS A CONTROL. The prime
     suspect was that overlapping 32-lag windows and volatility clustering make
     consecutive gradients dependent, inflating `Var(A)` above `Q` while a
     per-step sign flip destroys that dependence in the twin. `--sign-block B`
     draws one Rademacher sign per block of `B` CONSECUTIVE steps, carrying
     within-block dependence into the twin (the block-bootstrap correction).
     It changes nothing, because there is nothing to correct: the energy of the
     block sums over `Q` is 1.0000/0.9969/0.9944/0.9865 at B=1/8/64/512 on the
     permuted stream and 1.0000/0.9750/0.9149/0.9183 on the real one. The
     increments are already effectively uncorrelated. Permuted mean level over
     B=1..32768 stays in 0.14-0.32 with no trend.
  3. THE ACTUAL CAUSE: the level's own null is not zero. The 224 columns are
     strongly cross-correlated, so the observed `t` vector moves as a BLOCK from
     realisation to realisation, while `level = 1 - #{twin>=t}/#{obs>=t}` is
     clamped into [0, 1]. The clamp makes the error one-sided -- a twin draw
     that lands low inflates the level, one that lands high is clipped at 0
     instead of going negative -- so E[level] > 0 under a TRUE null. Measured
     with `--null-reference`: on 200 exact null realisations carrying the real
     column covariance the mean level is 0.099 +/- 0.140 (p95 0.419), NOT 0.
     Against that reference the permuted stream (0.27) is p = 0.13, i.e. it was
     never significant, and the real stream (0.84) exceeds all 200 draws.
     The level is usable as a RANKING and as a gate; its absolute value is not a
     false-discovery rate and must be read against this empirical null.

POSITIVE CONTROL. `--inject-alpha` adds `alpha * x[:, k]` to the target, so a
known edge of known size is planted in the REAL feature stream (and destroyed by
the permutation, exactly as a real edge would be). Sweeping alpha down until the
learners stop finding it converts a null result into a bound: it says how large
an edge would have had to be for this suite to have seen it. Alphas are a config
dimension too, so the whole sensitivity curve is one pass.

MEASURED, SPY.300.bars, 468021 bars, 224 features, prequential batch 1, every
method at ITS OWN best LR from the shared grid {3e-7, 1e-6, 3e-6, 1e-5, 3e-5},
`--null-streams 8` independent permutations giving the null's mean and sd.

                        best ratio    at lr    null mean +- sd    perm p
    uncentred  sgd        0.99998     3e-7     1.00002  0.00002   0.000
    uncentred  adam       0.99997     3e-7     0.99999  0.00001   0.000
    uncentred  mirror     0.99997     1e-6     0.99999  0.00001   0.000
      centred  sgd        0.99936     1e-6     1.00017  0.00002   0.000
      centred  adam       0.99936     1e-6     1.00005  0.00001   0.000
      centred  mirror     0.99941     3e-6     1.00000  0.00002   0.000

Three conclusions, none of which is a mechanism win.

* THE SIGNAL IS REAL AND IT WAS THE CENTRING THAT EXPOSED IT. Every method now
  beats the zero-predictor, by 0.00064 against a permutation null whose sd is
  0.00002 -- zero of 24 permutations came within reach. The earlier "no method
  beats the zero-predictor, no real-vs-permuted gap anywhere" was an artifact of
  the uncentred stream: at the same LR grid, uncentred tops out at 0.99997.
  Widening the LR grid alone does NOT produce it; centring does.
* MIRROR DOES NOT WIN. At the truncated grid that bottoms out at 1e-5 mirror
  looks like the only sub-1.0 method (0.99941 vs adam 0.99995). That is purely a
  step-size effect: mirror's mean level ~0.13 makes its effective LR ~10x
  smaller, and once adam and sgd are given the 1e-6 they actually want they
  reach 0.99936 and mirror is 5e-5 WORSE. Replicated on QQQ.300.bars: sgd/adam
  0.99947, mirror 0.99951. Same sign, same size, second instrument.
* WHAT MIRROR DOES BUY IS LR INSENSITIVITY. Over 3e-7..3e-5 mirror spans
  0.99939-0.99977 (5e-4) while sgd spans 0.99934-1.00680 (7.5e-3), a 16x tighter
  envelope, and sgd diverges at 1e-3 where mirror is still 1.032. That is a real
  and reproducible property. It is not an accuracy win.

SOFTHINGE: A CALIBRATION WIN THAT IS NOT AN ACCURACY WIN. A second level mode,
`softhinge`, replaces the rank-based false-discovery proportion with a soft
hinge on the FAMILY-WISE twin null,

    level = softplus(k * (1 - z^2/t^2)) / k    z^2 = max_i t_null_i^2, k = 24

(`state_plasticity._softhinge_level`; `_softhinge_rows` below for the batched
sweep, equal to `hidden_stream.softhinge_level` to float32 rounding). It is the
best rule on both synthetic harnesses in this family. Here it is not, and the
two claims below are separate.

  ACCURACY. Grid 1e-8..1e-2 in half decades, EVERY winner interior, 4 paired
  seeds (identical bars, identical zero init; only the twin signs and the
  permutation are resampled). Best real ratio: sgd/adam/adamw 0.999358 at 1e-6,
  mirror 0.999421 at 1e-5, softhinge 0.999726 at 3e-5. Paired difference
  softhinge - adam = +0.000368 +- 0.000086 (sd of the DIFFERENCE, n = 4);
  mirror - adam = +0.000063 +- 0.000015. Both gates are WORSE than the bare
  optimizer at its own optimum, softhinge by 6x mirror's margin. Note where
  softhinge's optimum sits: 3e-5, 30x above adam's, which is exactly the
  effective-step shrinkage its own gate imposes -- on a grid that stopped at
  1e-5 it would have read as a win. Uncentred (`--no-center`), same grid:
  nothing beats the zero-predictor by more than 3e-5 and no method shows a
  real-minus-permuted gap, which reproduces the original null on this data.

  CALIBRATION -- the reason to run this at all. With the weights pinned
  (`--lr-grid 0.0`) the level is a pure detector and the permuted stream is an
  exact zero-signal ground truth. 4 twin seeds x 4 independent permutations,
  224 connections apiece:

                      REAL              PERMUTED (zero signal, ground truth)
                  mean   open>.5     mean     open>.5   >0.1     p95
    mirror       0.846    0.875     0.156      0.099    0.395   7.0e-1
    softhinge    0.115    0.108     0.00226    0.000    0.008   1.1e-10

  Mirror holds a mean level of 0.156 and 9.9% of connections fully open on a
  stream that cannot be learned -- individual realisations reach 0.604 mean
  level -- which is the clamp-induced one-sided bias documented above, and it
  is anti-conservative. Softhinge reads 0.00226, opens ZERO of 3584 connection
  levels, and its 95th permuted percentile is 1.1e-10, on the
  softplus(-24)/24 = 1.6e-12 floor. Real-over-permuted mean-level ratio: 51
  for softhinge against 5.4 for mirror. Softhinge is calibrated on a
  zero-signal stream to a residual 0.2% of the step budget; mirror is not
  calibrated at all. Softhinge is also much more conservative on the REAL
  stream (11% open against 87%) -- the same conservatism, and what costs it the
  accuracy against a 6e-4 edge.

SENSITIVITY (what edge would we have had to have?). `--inject-alpha a` on the
most recent return column plants a feature with `corr = a` almost exactly
(`R^2 = a^2 var(x)/(var(y) + a^2 var(x))`, and here `var(x) = var(y) = 2.2565`).
At each method's own best LR the injected edge only becomes visible above the
alpha=0 baseline at a = 0.03 (corr 3.0%, R^2 0.090%), and mirror only overtakes
adam and sgd at a = 0.05 (corr 5.0%, R^2 0.249%; mirror 0.99808 vs 0.99831).
For scale, SPY's own best single feature carries corr 1.14% and all 224 together
carry an in-sample OLS R^2 of 0.165%. The real edge is a THIRD of the smallest
injected edge this protocol can resolve marginally, which is why the learners
capture only ~0.064% of 0.165% and why no accuracy ranking between methods on
this data should be trusted below about 5e-5.
"""

import os
import time
import struct
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import tyro

from cleanrl.shared.state_plasticity import MirrorPlasticity

BAR_DTYPE = np.dtype([("t", "<i8"), ("open", "<f4"), ("high", "<f4"),
                      ("low", "<f4"), ("close", "<f4"), ("volume", "<f4"),
                      ("vwap", "<f4"), ("count", "<u4")])
HEADER = struct.Struct("<8sIIQqq24s")
MAGIC = b"TBBARS01"
METHODS = ("sgd", "adam", "adamw", "mirror", "softhinge", "hetvar")
CHANNELS = ("ret", "high-close", "close-low", "close-vwap", "range",
            "volume-surprise", "count-surprise")


@dataclass
class Args:
    bars: str = "/home/marvin/Documents/repositories/trading_bot_0/long_data/bars/SPY.300.bars"
    """packed 5-minute bar file (header `TBBARS01`, 36-byte records)"""
    method: str = "all"
    lags: int = 32
    """how many past bars each feature block reaches back over"""
    steps: int = 0
    """0 = the whole file"""
    lr_grid: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3)
    refresh_every: int = 50
    hinge_sharpness: float = 24.0
    """`k` in the `softhinge` level `softplus(k*(1 - z^2/t^2))/k`. 24 puts the
    fully-closed level at `softplus(-24)/24 = 1.6e-12`, far below the 1/sqrt(224)
    at which absorbed noise would still be visible, and leaves the open end
    saturating at 1 so a certified coordinate is not taxed."""
    score_after: int = 2000
    """bars excluded from scoring, so no learner is judged on its cold start.
    The plasticity rule itself engages from step 1: a warmup during which the
    bare optimizer runs would bank the noise it exists to refuse, and freezing
    the weights afterwards cannot undo it (measured: ratio 1.079 with 99.6% of
    levels already shut)."""
    vol_span: float = 0.01
    """EWMA rate for the causal volatility scale"""
    weight_decay: float = 1e-2
    center: bool = True
    """subtract a TRAILING EWMA mean from every channel and from the target.
    Without it neither is mean-zero, so `E[y x_i] = mu_y m_i != 0` survives any
    permutation of the target: the null is not null, and a correct detector
    fires on the drift-times-feature-mean product."""
    sign_mode: str = "coordinate"
    """`step` = one Rademacher sign per step, shared by every coordinate;
    `coordinate` = an independent sign per coordinate. Sharing makes the twin's
    coordinates co-signed, which collapses its effective degrees of freedom when
    the real features are cross-correlated and leaves the false-discovery
    estimate anti-conservative."""
    sign_block: tuple[int, ...] = (1,)
    """block length for the Rademacher flip: one sign per `B` CONSECUTIVE steps.
    B=1 is the original i.i.d. flip, which assumes independent gradient
    increments. Swept as a config dimension -- every B is a row of the same
    single pass."""
    twin_reps: int = 1
    """independent sign-randomised twins per config. 1 reproduces the mechanism
    exactly (the null is the twin's 224 coordinates). More reps only sharpen the
    null used by the DIAGNOSTIC; they also pool into the level's null."""
    inject_alpha: tuple[float, ...] = (0.0,)
    """positive control: add `alpha * x[:, inject_column]` to the target before
    the permutation, so the planted edge is real on stream 0 and absent on the
    null. Swept as a config dimension."""
    inject_column: int = -1
    """feature column to inject; -1 = the most recent bar's return channel"""
    feature_report: bool = False
    """per-feature `t = |A|/sqrt(Q)` against the twin, for the frozen-weight rows"""
    null_reference: bool = False
    """calibrate the level against its OWN null. Each twin replicate is an exact
    null realisation of the accumulator, so scoring replicate r against the pool
    of the others gives the distribution of `mean level` when nothing is real.
    Needs `--sign-mode step --twin-reps 64` or more: the shared per-step sign is
    what carries the columns' cross-correlation into the null, and independent
    per-coordinate signs would destroy the very structure being calibrated."""
    null_streams: int = 1
    """independent time-permutations of the target carried alongside the real
    stream. One gives the original single null; several turn the real-minus-
    permuted gap into a permutation test with a p-value instead of an
    unreplicated difference that nobody can size."""
    seed: int = 1
    cuda: bool = True
    readout_lr: float = 0.002
    """per-sample normalized-LMS rate of `hetvar`'s state readouts (E[r|x], log Var(r|x),
    and the mispaired twin whose deviation variance shrinks the field)"""
    raw_target: bool = False
    """predict the raw (centred) return in fixed units instead of the trailing-vol-scaled
    one: volatility clustering stays in the target, so per-SAMPLE precision has something
    to read from the lagged range/return channels"""
    vol_feature: bool = False
    """append one causal feature: log of the trailing return volatility in the target's
    units. Every channel is divided by its own trailing scale, so without this the state
    carries NO volatility level and target heteroscedasticity is invisible to any
    state-conditional rule"""


def read_bars(path):
    with open(path, "rb") as handle:
        head = HEADER.unpack(handle.read(HEADER.size))
    if head[0] != MAGIC:
        raise ValueError(f"{path}: not a TBBARS01 file")
    bars = np.fromfile(path, dtype=BAR_DTYPE, offset=HEADER.size)
    if bars.size != head[3]:
        raise ValueError(f"{path}: header claims {head[3]} records, found {bars.size}")
    return bars


def build_stream(bars, args):
    """Causal features and a scaled next-bar return. Nothing reads the future.

    Seven per-bar channels, each lagged `lags` times: the return, the two wick
    ranges, the close-vwap dislocation, the range, and volume/trade-count
    surprise. All are divided by a TRAILING EWMA scale, so the network sees a
    roughly stationary input without any look-ahead.
    """
    close = bars["close"].astype(np.float64)
    if not np.all(close > 0):
        raise ValueError("non-positive closes present")
    ret = np.zeros_like(close)
    ret[1:] = np.log(close[1:] / close[:-1])
    volume = np.log1p(bars["volume"].astype(np.float64))
    count = np.log1p(bars["count"].astype(np.float64))
    channels = np.stack([
        ret,
        (bars["high"] - bars["close"]) / close,
        (bars["close"] - bars["low"]) / close,
        (bars["close"] - bars["vwap"]) / close,
        (bars["high"] - bars["low"]) / close,
        volume - _ewma(volume, args.vol_span),
        count - _ewma(count, args.vol_span),
    ], axis=1)
    # trailing scale per channel: EWMA of |x|, shifted so step t uses < t only
    if args.center:
        mean = _ewma(channels, args.vol_span, axis=0)
        channels = channels - np.concatenate([np.zeros_like(mean[:1]), mean[:-1]], axis=0)
    scale = _ewma(np.abs(channels), args.vol_span, axis=0)
    scale = np.concatenate([scale[:1], scale[:-1]], axis=0)
    channels = channels / np.maximum(scale, 1e-12)
    np.clip(channels, -10.0, 10.0, out=channels)

    lags, n_chan = args.lags, channels.shape[1]
    total = channels.shape[0] - lags - 1
    if total <= 0:
        raise ValueError("not enough bars for the requested lag depth")
    features = np.empty((total, lags * n_chan), dtype=np.float32)
    for lag in range(lags):
        # row t holds bars t..t+lags-1, and the target is bar t+lags
        features[:, lag * n_chan:(lag + 1) * n_chan] = channels[lag:lag + total]
    drift = _ewma(ret, args.vol_span)
    centred = ret - np.concatenate([[0.0], drift[:-1]]) if args.center else ret
    vol = np.maximum(_ewma(np.abs(centred), args.vol_span), 1e-12)
    if args.raw_target:
        # keep volatility clustering IN the target: one causal constant (trailing vol at
        # the last unscored bar) sets the units, nothing tracks the level afterwards
        vol = np.full_like(vol, vol[min(args.score_after, len(vol) - 1)])
    target = (centred[lags + 1:lags + 1 + total] / vol[lags:lags + total]).astype(np.float32)
    if args.vol_feature:
        trailing = np.maximum(_ewma(np.abs(centred), args.vol_span), 1e-12)
        # row t predicts ret[t+lags+1]; trailing[t+lags] uses |ret| up to index t+lags: causal
        logvol = np.log(trailing[lags:lags + total] / vol[lags:lags + total])
        features = np.concatenate([features, logvol[:, None].astype(np.float32)], axis=1)
    clip = 50.0 if args.raw_target else 10.0
    np.clip(target, -clip, clip, out=target)
    return features, target


def feature_name(column, lags, n_chan=len(CHANNELS)):
    """Column -> (channel name, bars back from the predicted bar).

    Column `lag*n_chan + c` holds bar `t+lag` of a row whose target is bar
    `t+lags`, so the column is `lags - lag` bars old.
    """
    lag_slot, chan = divmod(column, n_chan)
    return CHANNELS[chan], lags - lag_slot


def _ewma(values, rate, axis=0):
    """Causal EWMA: entry t depends on entries <= t only."""
    out = np.empty_like(values, dtype=np.float64)
    acc = np.zeros(values.shape[1:], dtype=np.float64) if values.ndim > 1 else 0.0
    keep = 1.0 - rate
    flat = np.moveaxis(values, axis, 0)
    result = np.moveaxis(out, axis, 0)
    for index in range(flat.shape[0]):
        acc = keep * acc + rate * flat[index] if index else flat[index]
        result[index] = acc
    return out


def run_all(args, features, targets, configs, device):
    """Every (method, lr, target-stream, sign-block) config advanced in ONE pass.

    The stream is sequential and cannot be parallelised over time, but the
    configs are independent, so they become a leading batch dimension: one pass
    over 468k bars scores the whole sweep instead of one cell of it. Everything
    below is a handful of fused kernels per bar over a (K, D) weight tensor.
    """
    n_cfg = len(configs)
    dim = features.shape[1]
    reps = max(args.twin_reps, 1)
    lr = torch.tensor([c["lr"] for c in configs], device=device).unsqueeze(1)
    is_adam = torch.tensor([c["method"] != "sgd" for c in configs],
                           device=device).unsqueeze(1).float()
    decay = torch.tensor([args.weight_decay if c["method"] == "adamw" else 0.0
                          for c in configs], device=device).unsqueeze(1)
    is_mirror = torch.tensor([c["method"] == "mirror" for c in configs],
                             device=device).unsqueeze(1).float()
    is_hinge = torch.tensor([c["method"] == "softhinge" for c in configs],
                            device=device).unsqueeze(1).float()
    is_hetvar = torch.tensor([c["method"] == "hetvar" for c in configs],
                             device=device).unsqueeze(1).float()
    # per-SAMPLE precision from the sample's own state (sample_stream.py `hetvar_ta`):
    # c_t = (1/Var(r|x_t)) * (nu+1)/(nu+z_t^2), field shrunk by a mispaired twin,
    # nu from the standardized residual's kurtosis, level normalized by a running mean
    u_mu = torch.zeros((n_cfg, dim + 1), device=device)
    u_lv = torch.zeros((n_cfg, dim + 1), device=device)
    u_tw = torch.zeros((n_cfg, dim + 1), device=device)
    v_pred = torch.zeros((n_cfg,), device=device)
    v_twin = torch.zeros((n_cfg,), device=device)
    var_ema = torch.ones((n_cfg,), device=device)
    c_ema = torch.ones((n_cfg,), device=device)
    # the twin's mispaired residual^2 must come from a sample whose noise is independent of
    # this state. The PREVIOUS bar is not: volatility clusters, so a lag-1 twin learns the
    # real field and shrinks it to zero. Draw from a long ring of past residuals instead.
    ring = torch.ones((65536, n_cfg), device=device)   # ~3 years of bars: past vol memory
    z2_ema = torch.ones((n_cfg,), device=device)
    z4_ema = 3.0 * torch.ones((n_cfg,), device=device)
    c_sum = torch.zeros((n_cfg,), device=device)
    c_sq = torch.zeros((n_cfg,), device=device)
    beta = 0.999
    stream = torch.tensor([c["stream"] for c in configs], device=device)

    weight = torch.zeros((n_cfg, dim), device=device)
    bias = torch.zeros((n_cfg, 1), device=device)
    mom = torch.zeros((n_cfg, dim + 1), device=device)
    vel = torch.zeros((n_cfg, dim + 1), device=device)
    total = torch.zeros((n_cfg, dim), device=device)
    square = torch.zeros((n_cfg, dim), device=device)
    twin = torch.zeros((n_cfg, reps, dim), device=device)
    level = torch.ones((n_cfg, dim), device=device)
    hinge = torch.ones((n_cfg, dim), device=device)
    open_gate = torch.ones((n_cfg, dim), device=device)
    gate = torch.ones((n_cfg, dim), device=device)
    squared = torch.zeros((n_cfg,), device=device)
    trivial = torch.zeros((n_cfg,), device=device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    per_coord = args.sign_mode == "coordinate"

    # one flip vector per config row; rows sharing a block length refresh together
    flip = torch.ones((n_cfg, reps, dim if per_coord else 1), device=device)
    groups = {}
    for index, cfg in enumerate(configs):
        groups.setdefault(cfg["block"], []).append(index)
    flip_groups = []
    for block, rows in sorted(groups.items()):
        shape = (len(rows), reps, dim if per_coord else 1)
        index = None if len(rows) == n_cfg else torch.tensor(rows, device=device)
        flip_groups.append((block, index, shape))
    scored = 0

    for step in range(features.shape[0]):
        x = features[step]
        y = targets[stream, step]
        prediction = weight @ x + bias.squeeze(1)
        residual = prediction - y
        if step >= args.score_after:
            squared += residual.detach().square()
            trivial += y.square()
            scored += 1
        if is_hetvar.any():
            phi = torch.cat([x, x.new_ones(1)])                       # (dim+1,) shared
            p_mu = u_mu @ phi
            dev_lv = u_lv[:, :dim] @ x
            dev_tw = u_tw[:, :dim] @ x
            v_pred = beta * v_pred + (1 - beta) * dev_lv.square()
            v_twin = beta * v_twin + (1 - beta) * dev_tw.square()
            shrink = (1.0 - v_twin / v_pred.clamp_min(1e-12)).clamp(0.0, 1.0)
            p_lv = (u_lv[:, dim] + shrink * dev_lv).clamp(-12.0, 12.0)
            centred = residual - p_mu
            var_hat = p_lv.exp()
            var_ema = beta * var_ema + (1 - beta) * var_hat
            c_raw = 1.0 / var_hat.clamp_min(0.02 * var_ema)
            z2 = (centred.square() / var_hat).clamp_max(1e4)
            kappa = z4_ema / z2_ema.square().clamp_min(1e-12)
            nu = torch.where(kappa > 3.05, (4.0 * kappa - 6.0) / (kappa - 3.0).clamp_min(1e-3),
                             torch.full_like(kappa, 1e6))
            c_raw = c_raw * (nu + 1.0) / (nu + z2)
            z2_ema = beta * z2_ema + (1 - beta) * z2
            z4_ema = beta * z4_ema + (1 - beta) * z2.square()
            c_ema = beta * c_ema + (1 - beta) * c_raw
            c = (c_raw / c_ema.clamp_min(1e-12)).clamp_max(20.0)
            c = torch.where(is_hetvar.squeeze(1).bool(), c, torch.ones_like(c))
            if step >= args.score_after:
                c_sum += c
                c_sq += c * c
            # readouts: normalized LMS on the sample's own state
            phin = phi / phi.square().sum()
            c2 = centred.square()
            e_lv = (1.0 - c2 / (u_lv @ phi).clamp(-12.0, 12.0).exp()).clamp(-20.0, 1.0)
            mis = ring[torch.randint(0, ring.shape[0], (1,), device=device, generator=gen)].squeeze(0)
            e_tw = (1.0 - mis / (u_tw @ phi).clamp(-12.0, 12.0).exp()).clamp(-20.0, 1.0)
            ring[step % ring.shape[0]] = c2
            u_mu -= args.readout_lr * (p_mu - residual).unsqueeze(1) * phin
            u_lv -= args.readout_lr * e_lv.unsqueeze(1) * phin
            u_tw -= args.readout_lr * e_tw.unsqueeze(1) * phin
            residual = residual * c
        grad_w = residual.unsqueeze(1) * x.unsqueeze(0)
        grad_b = residual.unsqueeze(1)

        # evidence on the RAW gradient, before any step is taken or scaled
        total += grad_w
        square += grad_w * grad_w
        for block, index, shape in flip_groups:
            if step % block:
                continue
            draw = torch.randint(0, 2, shape, device=device, generator=gen,
                                 dtype=torch.float32).mul_(2.0).sub_(1.0)
            if index is None:
                flip.copy_(draw)
            else:
                flip[index] = draw
        twin.addcmul_(grad_w.unsqueeze(1), flip)
        if step % args.refresh_every == 0:
            scale = square.sqrt().clamp_min(1e-30)
            t_obs = total.abs() / scale
            t_null = (twin.abs() / scale.unsqueeze(1)).reshape(n_cfg, -1)
            level = _fdp_rows(t_obs, t_null)
            hinge = _softhinge_rows(t_obs, t_null, args.hinge_sharpness)
            # select once per refresh rather than once per bar; a row running a
            # bare optimizer keeps its ones and is exactly ungated
            gate = torch.where(is_mirror.bool(), level,
                               torch.where(is_hinge.bool(), hinge, open_gate))

        grad = torch.cat([grad_w, grad_b], dim=1)
        mom.mul_(0.9).add_(grad, alpha=0.1)
        vel.mul_(0.999).addcmul_(grad, grad, value=0.001)
        adam_step = ((mom / (1.0 - 0.9 ** (step + 1)))
                     / ((vel / (1.0 - 0.999 ** (step + 1))).sqrt() + 1e-8))
        update = lr * torch.where(is_adam.bool(), adam_step, grad)
        # the level scales the REALIZED step, so Adam cannot divide it back out
        weight -= update[:, :dim] * gate
        bias -= update[:, dim:]
        weight -= lr * decay * weight

    mse = (squared / max(scored, 1)).cpu().numpy()
    base = (trivial / max(scored, 1)).cpu().numpy()
    scale = square.sqrt().clamp_min(1e-30)
    return {"mse": mse, "trivial": base, "scored": scored,
            "weight_rms": weight.square().mean(1).sqrt().cpu().numpy(),
            "level_mean": gate.mean(1).cpu().numpy(),
            "c_mean": (c_sum / max(scored, 1)).cpu().numpy(),
            "c_sd": (c_sq / max(scored, 1) - (c_sum / max(scored, 1)) ** 2).clamp_min(0).sqrt().cpu().numpy(),
            "level_open": (gate > 0.5).float().mean(1).cpu().numpy(),
            "levels": gate.cpu().numpy(),
            "t_obs": (total.abs() / scale).cpu().numpy(),
            "t_null": (twin.abs() / scale.unsqueeze(1)).cpu().numpy()}


def _fdp_rows(t_obs, t_null):
    """One minus the false-discovery proportion, independently per config row.

    `t_null` may pool several twin replicates, so the expected count of false
    discoveries at threshold `t` is `dim * P_null(T >= t)`, not a raw count.
    """
    width = t_obs.shape[1]
    null_sorted = t_null.sort(dim=1).values
    obs_sorted = t_obs.sort(dim=1).values
    draws = t_null.shape[1]
    false_ge = draws - torch.searchsorted(null_sorted, t_obs, right=False)
    expected = false_ge.float() * (width / draws)
    total_ge = (width - torch.searchsorted(obs_sorted, t_obs, right=False)).clamp_min(1)
    return (1.0 - expected / total_ge.float()).clamp_(0.0, 1.0)


def _softhinge_rows(t_obs, t_null, sharpness):
    """The soft-hinge level, independently per config row.

    `z^2 = max_i t_null_i^2` is the FAMILY-WISE null: the largest squared
    evidence this row's provably-null twin reached anywhere among the 224
    coordinates (pooled over replicates) over the identical window. One
    reduction, no sort, no threshold, and no knowledge of the noise scale. The
    row form of `state_plasticity._softhinge_level`, which takes the max over a
    whole parameter tensor instead of over a config row.
    """
    z_sq = t_null.square().amax(dim=1, keepdim=True)
    return functional.softplus(1.0 - z_sq / t_obs.square().clamp_min(1e-30),
                               beta=sharpness)


def build_targets(target_np, features_np, args, orders):
    """(alpha, real | null_1..null_N) -> a row of the target matrix.

    The injection is applied BEFORE the permutation, so the planted edge lives on
    the real stream and every null destroys it, exactly as a genuine edge would
    be. `--null-streams N` carries N INDEPENDENT permutations, which turns the
    real-minus-permuted gap from a single unreplicated difference into a
    permutation test: the N null ratios are the exact reference distribution of
    the gap under "nothing is predictable".
    """
    column = args.inject_column
    if column < 0:
        column = (args.lags - 1) * len(CHANNELS)
    rows = []
    for alpha in args.inject_alpha:
        y = target_np if alpha == 0.0 else target_np + alpha * features_np[:, column]
        rows.append(y)
        rows.extend(y[order] for order in orders)
    return np.stack(rows), column


def _report_features(args, out, configs):
    """Per-feature evidence with the weights frozen: is ANY column predictive?"""
    lags = args.lags
    for index, cfg in enumerate(configs):
        if cfg["method"] != "mirror" or cfg["lr"] != 0.0 or cfg["kind"] > 1:
            continue
        t_obs = out["t_obs"][index]
        t_null = out["t_null"][index]
        maxima = t_null.max(axis=1)
        kind = "PERMUTED" if cfg["kind"] else "REAL"
        print(f"\n-- {kind} stream, sign-block {cfg['block']}, alpha {cfg['alpha']:g}: "
              f"per-feature t = |A|/sqrt(Q), {t_null.size} twin draws")
        print(f"   twin: mean {t_null.mean():.3f}  p95 {np.quantile(t_null, 0.95):.3f}  "
              f"max {t_null.max():.3f} | family-wise (max over 224 cols per twin): "
              f"median {np.median(maxima):.3f} p95 {np.quantile(maxima, 0.95):.3f}")
        above = int((t_obs > t_null.max()).sum())
        print(f"   observed: mean {t_obs.mean():.3f}  max {t_obs.max():.3f} | "
              f"{above}/{t_obs.size} columns above the twin's max | "
              f"mean level {out['level_mean'][index]:.4f}")
        for rank, column in enumerate(np.argsort(-t_obs)[:10]):
            name, back = feature_name(int(column), lags)
            tail = float((t_null >= t_obs[column]).mean())
            print(f"   {rank + 1:2d}. {name:>15s} lag {back:2d} bars  "
                  f"t {t_obs[column]:7.3f}  twin tail {tail:.5f}  "
                  f"family-wise p {float((maxima >= t_obs[column]).mean()):.4f}")


def _level_np(t_obs, null_flat):
    """The mechanism's level, per column, against a pooled null sample."""
    width, draws = t_obs.size, null_flat.size
    false_ge = (draws - np.searchsorted(np.sort(null_flat), t_obs)) * (width / draws)
    total_ge = np.maximum(width - np.searchsorted(np.sort(t_obs), t_obs), 1)
    return np.clip(1.0 - false_ge / total_ge, 0.0, 1.0)


def _null_reference(args, out, configs):
    """What does `mean level` read when the null is EXACTLY true?

    Every twin replicate is a bona fide null realisation of the accumulator that
    carries the real columns' cross-correlation (with a shared per-step sign), so
    leave-one-out scoring of the replicates traces the level's own null
    distribution. Without this reference a nonzero mean level cannot be read as
    evidence of anything, because its null is not zero.
    """
    for index, cfg in enumerate(configs):
        if cfg["method"] != "mirror" or cfg["lr"] != 0.0 or cfg["kind"] > 1:
            continue
        t_null = out["t_null"][index]
        reps = t_null.shape[0]
        if reps < 8:
            raise ValueError("--null-reference needs --twin-reps 8 or more")
        draws = np.stack([_level_np(t_null[r], np.delete(t_null, r, axis=0).ravel()).mean()
                          for r in range(reps)])
        observed = _level_np(out["t_obs"][index], t_null.ravel()).mean()
        kind = "PERMUTED" if cfg["kind"] else "REAL"
        print(f"\n-- {kind} stream, sign-block {cfg['block']}, alpha {cfg['alpha']:g}: "
              f"level calibrated against {reps} exact null realisations")
        print(f"   NULL mean level: mean {draws.mean():.3f} sd {draws.std():.3f} "
              f"p95 {np.quantile(draws, 0.95):.3f} max {draws.max():.3f}   <- not zero")
        print(f"   observed mean level {observed:.3f}  ->  p = "
              f"{float((draws >= observed).mean()):.4f}")


def _report_levels(out, configs):
    """Mean, dispersion and open fraction of the level, real versus permuted.

    This is the calibration test, and it does not need a win to be informative.
    The permuted stream has EXACTLY zero predictability, so a correctly
    calibrated level must close there: `mean` at the floor, `open` at zero. Any
    level a rule holds open on the permuted stream is manufactured certainty.
    """
    print(f"\n{'method':>9s} {'lr':>8s} {'stream':>8s} | {'mean':>10s} "
          f"{'sd':>10s} {'open>.5':>8s} {'p95':>10s} {'max':>8s}")
    for index, cfg in enumerate(configs):
        if cfg["method"] not in ("mirror", "softhinge") or cfg["kind"] > 1:
            continue
        lv = out["levels"][index]
        kind = "PERMUTED" if cfg["kind"] else "REAL"
        print(f"{cfg['method']:>9s} {cfg['lr']:>8g} {kind:>8s} | {lv.mean():10.3e} "
              f"{lv.std():10.3e} {(lv > 0.5).mean():8.3f} "
              f"{np.quantile(lv, 0.95):10.3e} {lv.max():8.4f}")


def main():
    args = tyro.cli(Args)
    methods = METHODS if args.method == "all" else (args.method,)
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"method must be `all` or one of {METHODS}")
    if not torch.cuda.is_available() and args.cuda:
        raise RuntimeError("CUDA required")
    device = torch.device("cuda" if args.cuda else "cpu")
    bars = read_bars(args.bars)
    features_np, target_np = build_stream(bars, args)
    if args.steps:
        features_np, target_np = features_np[:args.steps], target_np[:args.steps]
    rng = np.random.default_rng(args.seed)
    orders = [rng.permutation(target_np.shape[0]) for _ in range(max(args.null_streams, 1))]
    target_rows, column = build_targets(target_np, features_np, args, orders)
    name, back = feature_name(column, args.lags)
    print(f"{os.path.basename(args.bars)}: {features_np.shape[0]} bars, "
          f"{features_np.shape[1]} features, {features_np.shape[0] - args.score_after} scored, "
          f"center={args.center} sign={args.sign_mode} reps={args.twin_reps} "
          f"inject={name}@{back}bars", flush=True)
    features = torch.as_tensor(features_np, device=device)
    # row ai*(1+N) = real at alpha ai, the N rows after it are its permutation nulls
    targets = torch.as_tensor(target_rows, device=device)

    nulls = len(orders)
    configs = [{"method": m, "lr": lr, "block": b, "alpha": a,
                "kind": kind, "stream": ai * (1 + nulls) + kind}
               for ai, a in enumerate(args.inject_alpha)
               for m in methods for lr in args.lr_grid
               for b in args.sign_block for kind in range(1 + nulls)]
    started = time.perf_counter()
    out = run_all(args, features, targets, configs, device)
    elapsed = time.perf_counter() - started
    ratio = out["mse"] / out["trivial"]
    print(f"{len(configs)} configs in one pass, {elapsed:.1f}s "
          f"({1e6 * elapsed / max(out['scored'], 1) / len(configs):.1f}us per config-bar)\n")
    print(f"{'alpha':>7s} {'blk':>5s} {'method':>9s} {'lr':>8s} | {'REAL':>9s} "
          f"{'NULL mean':>9s} {'NULL sd':>8s} | {'gap':>8s} {'perm p':>7s} | "
          f"{'lvl REAL':>8s} {'lvl NULL':>8s}")
    seen = {}
    for index, cfg in enumerate(configs):
        key = (cfg["alpha"], cfg["block"], cfg["method"], cfg["lr"])
        seen.setdefault(key, {})[cfg["kind"]] = index
    for (alpha, block, method, lr), cells in seen.items():
        real = ratio[cells[0]]
        null = np.array([ratio[cells[k]] for k in range(1, 1 + nulls)])
        # p = how often a target with NO predictability scores at least as well
        pval = float((null <= real).mean())
        print(f"{alpha:>7g} {block:>5d} {method:>9s} {lr:>8g} | {real:9.5f} "
              f"{null.mean():9.5f} {null.std():8.5f} | {null.mean() - real:8.5f} "
              f"{pval:7.3f} | {out['level_mean'][cells[0]]:8.4f} "
              f"{out['level_mean'][cells[1]]:8.4f}"
              + (f" | c {out['c_mean'][cells[0]]:.3f}±{out['c_sd'][cells[0]]:.3f}"
                 if method == "hetvar" else ""))
    _report_levels(out, configs)
    if args.feature_report:
        _report_features(args, out, configs)
    if args.null_reference:
        _null_reference(args, out, configs)


if __name__ == "__main__":
    main()
