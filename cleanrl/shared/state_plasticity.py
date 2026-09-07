"""Mirror plasticity: graded, self-calibrating, per-connection step-size control.

PREMISE. Each perceptron (each synapse, in the default granularity) decides,
from its own accumulated experience, how much of the current step it is allowed
to take. The decision is graded in [0, 1], smooth in the evidence, and carries
no tuned threshold.

WHAT IS MEASURED (`cleanrl/plasticity/noisy_stream_diagnostic.py`, batch size 1,
single pass, 4096 inputs of which 1 predicts and 4095 are 1%-sparse distractors,
targets carrying N(0, 5) noise; test MSE against the NOISE-FREE target, where
the zero-predictor scores 0.0112):

    oracle (told the answer)   0.00219
    mirror (this module)       0.00397   level 1.0000 on signal, 0.0398 on noise
    binary admission veto      0.00604
    graded (t^2/(t^2+z^2))^2   0.01088
    sgd                        0.04176
    adam                       0.36029   (32x WORSE than predicting nothing)

THE STATISTIC. For a parameter entry, accumulate over updates

    A = sum_t g_t        Q = sum_t g_t^2        R = sum_t eps_t g_t

with ``eps_t`` a single Rademacher sign per step, shared by every entry and
independent of the gradients. Then ``t_A = |A| / sqrt(Q)`` is the exact
self-normalised cumulative: it grows as ``sqrt(n) * mu/sigma`` for an entry whose
gradient has a consistent sign and stays ``O(1)`` forever for one that does not.
Separation therefore IMPROVES with evidence -- unlike any ratio of EMA moments,
whose separation is capped by its horizon no matter how much evidence arrives.

THE CALIBRATION. ``R`` is a sign-randomized twin of ``A``: because ``eps`` is
independent of ``g``, ``E[R] = 0`` exactly while ``E[R^2] = E[A^2] - (E A)^2``,
so the twin shares the energy of the observed accumulator but is provably null
over the identical window. Its order statistics ARE the null distribution of the
evidence, which yields a false-discovery proportion with no knowledge of the
noise scale, the number of observations, or the sparsity:

    level = 1 - #{twin >= t_A} / #{observed >= t_A}      clamped to [0, 1]

The level REACHES exactly 1 once evidence is overwhelming. That matters: a
sigmoid-like gate such as ``(t^2/(t^2+z^2))^p`` throttles the very entry it has
certified, which measured as a 2.7x MSE penalty and half the signal weight.

PROPERTIES THAT SURVIVED MEASUREMENT.

* Batch-size invariant. At batch ``B`` the update gradient has mean ``mu`` and sd
  ``sigma/sqrt(B)``, so after ``n`` samples (``n/B`` updates)
  ``t = sqrt(n/B) * mu sqrt(B)/sigma = sqrt(n) mu/sigma``: the evidence depends
  on samples seen, not on how they are batched.
* Works at small fan-in. The null is pooled across the whole tensor, so a layer
  of ``H x D`` supplies ``H*D`` null draws. Measured to beat SGD 2.8x at fan-in
  64 with the level still exactly 1.0 on the signal.
* Applied POST-optimizer, to the realized step. Adam is invariant to a per-row
  gradient rescale, so a pre-optimizer multiplier is divided straight back out
  (measured: a 0.125x pre-Adam weighting delivers 0.897x for one batch and
  1.00x when sustained). The accumulators read the RAW gradient, which is never
  modified, so there is no path by which ``v`` can cancel the level.
* Never amplifies. ``level <= 1``, so the rule can only return budget to
  ``clip_grad_norm_``, never steal it from another unit.
* Retained evidence is a feature, not a leak. When the target relationship
  moves, pruning the now-obsolete weights recovers 1.4% of the error, while a
  coordinate that becomes predictive AGAIN is re-admitted immediately: measured
  7x the weight from 34% less post-change time than a fresh coordinate gets.
  Forgetting is a false fix -- an evidence decay ``d`` caps attainable ``t`` at
  ``sqrt(1/d)`` and strictly hurt at every rate tested.

LEVEL MODES. ``level_mode="mirror"`` is the false-discovery proportion above.
``level_mode="softhinge"`` keeps the three accumulators and the application
point and replaces the rank statistic with a soft hinge on the FAMILY-WISE twin
null ``z^2 = max_i t_null_i^2`` -- one number per tensor instead of two sorts:

    level = softplus(k * (1 - z^2 / t^2)) / k          k = 24

Measured elsewhere in this family, not here: on the noisy synthetic stream
(`cleanrl/plasticity/noisy_stream_diagnostic.py`, 4096 features of which 1
predicts, 100k steps, 4 seeds) softhinge scores 0.00039 test MSE against the
noise-free target = 0.04x the zero-predictor, versus oracle 0.00043, hard veto
0.00037, mirror 0.00227, sgd 0.09194 and adam 0.44816; on a batch-1 hidden layer
(`cleanrl/plasticity/hidden_stream.py`, 1024 inputs of which 4 are useful) the
pooled reallocated form scores 0.18292 against mirror 0.20703, oracle 0.28707
and adam 0.64374. Both ends of the shape are load-bearing. The closed level must
sit far below ``1/sqrt(D)`` because absorbed output noise scales with the rms
level over ``D`` distractors -- a floor of 0.125 fails outright, 0.04 leaves a
visible noise floor, ``softplus(-24)/24 = 1.6e-12`` does not -- and the open
level must saturate at 1, because a gate that keeps taxing an entry it has
already certified (``(t^2/(t^2+z^2))^p``) tripled the error and halved the
signal weight.

COST. Three buffers per parameter (1.5x Adam's optimizer state) plus two sorts
of the tensor every ``refresh_every`` updates.

NOT RL-SPECIFIC. The mechanism sees only parameter gradients, so PPO, time
series and language pretraining are all just consumers.
"""

import random

import torch
import torch.nn.functional as functional

__all__ = ["MirrorPlasticity"]


def _fdp_level(t_obs, t_null):
    """One minus the false-discovery proportion, per entry.

    ``t_null`` supplies the empirical null. Both are flattened, so the null is
    pooled across the whole tensor: an ``H x D`` layer contributes ``H*D`` draws
    and small fan-in is not a barrier.
    """
    flat = t_obs.reshape(-1)
    null_sorted = t_null.reshape(-1).sort().values
    obs_sorted = flat.sort().values
    n_null = null_sorted.numel()
    n_obs = obs_sorted.numel()
    # counts at-or-above each observed value; `right=False` counts ties as above
    false_ge = n_null - torch.searchsorted(null_sorted, flat, right=False)
    total_ge = (n_obs - torch.searchsorted(obs_sorted, flat, right=False)).clamp_min(1)
    fdp = false_ge.to(t_obs.dtype) / total_ge.to(t_obs.dtype)
    return (1.0 - fdp).clamp_(0.0, 1.0).view_as(t_obs)


def _softhinge_level(obs_sq, null_sq, sharpness):
    """Soft hinge against the family-wise twin null, on SQUARED statistics.

    ``z^2 = max_i t_null_i^2`` is the largest evidence the provably-null twin
    managed anywhere in the tensor over the identical window, so it is a
    family-wise null level that costs one reduction. An entry opens to the
    extent its own ``t^2`` clears it, smoothly and with no tuned threshold.
    """
    z_sq = null_sq.reshape(-1).max()
    return functional.softplus(1.0 - z_sq / obs_sq.clamp_min(1e-30),
                               beta=sharpness)


class MirrorPlasticity:
    """Graded per-entry plasticity applied to the realized optimizer step.

    Usage, around an existing optimizer::

        loss.backward()
        plasticity.before_step()        # reads .grad, snapshots weights
        torch.nn.utils.clip_grad_norm_(params, max_norm)
        optimizer.step()
        plasticity.after_step()         # scales the realized step per entry

    ``granularity``:
      ``"connection"`` -- every entry decides for itself (measured default).
      ``"unit"``       -- one level per output unit, the mean over its fan-in,
                          i.e. literally "each perceptron decides".
      ``"input"``      -- one level per INCOMING signal, pooling evidence across
                          every unit that receives it. Admission is latency-bound
                          and latency falls as ``1/sqrt(n)``, so pooling a layer
                          of width ``H`` detects a useful input ``sqrt(H)``
                          sooner. Per-unit signs differ, so the pooled statistic
                          is the energy ``sum_j t_ji^2`` -- which the same twin
                          calibrates exactly, no distributional assumption.

    ``level_mode``:
      ``"mirror"``    -- the false-discovery proportion (the measured default).
      ``"softhinge"`` -- the family-wise soft hinge, ``sharpness`` = ``k``.
    """

    def __init__(self, params, refresh_every=50, granularity="connection",
                 warmup=0, seed=0, reallocate=0.0, level_mode="mirror",
                 sharpness=24.0):
        if granularity not in ("connection", "unit", "input"):
            raise ValueError("granularity must be `connection`, `unit` or `input`")
        if level_mode not in ("mirror", "softhinge"):
            raise ValueError("level_mode must be `mirror` or `softhinge`")
        if refresh_every < 1:
            raise ValueError("refresh_every must be positive")
        self.params = [p for p in params if p.requires_grad]
        if not self.params:
            raise ValueError("MirrorPlasticity needs at least one trainable parameter")
        if reallocate and reallocate <= 1.0:
            raise ValueError("reallocate is an amplification cap and must exceed one")
        self.reallocate = float(reallocate)
        self.refresh_every = int(refresh_every)
        self.granularity = granularity
        self.level_mode = level_mode
        self.sharpness = float(sharpness)
        self.warmup = int(warmup)
        self.updates = 0
        # The Rademacher sign needs no device randomness, and drawing it on the
        # host keeps the update path free of a per-step D2H sync.
        self.signs = random.Random(seed)
        self.sums = [torch.zeros_like(p) for p in self.params]
        self.sqs = [torch.zeros_like(p) for p in self.params]
        self.mirrors = [torch.zeros_like(p) for p in self.params]
        # Level 1 during warmup: the rule starts as an exact no-op, so any
        # divergence from the unmodified baseline is attributable to it.
        self.levels = [torch.ones_like(p) for p in self.params]
        self.snapshots = [torch.empty_like(p) for p in self.params]

    @torch.no_grad()
    def before_step(self, evidence_weight=1.0):
        """Accumulate evidence from the RAW gradients and snapshot the weights.

        ``evidence_weight`` is an optional per-update reliability weight, applied
        to the ACCUMULATORS only -- never to the step. Weighting every one of
        ``A``, ``Q`` and ``R`` identically keeps the twin exactly null (``eps`` is
        still independent of everything else) while turning ``t`` into the
        GLS-weighted score, which is the maximum-power statistic when the
        observation noise varies. Pass ``1/sigma_hat^2`` from a causal estimate
        of the residual scale. Only useful under heteroscedasticity: on a
        homoscedastic stream a constant weight cancels out of ``t`` exactly.
        """
        self.updates += 1
        sign = 1.0 if self.signs.random() < 0.5 else -1.0
        for param, total, square, mirror, snapshot in zip(
                self.params, self.sums, self.sqs, self.mirrors, self.snapshots):
            snapshot.copy_(param)
            grad = param.grad
            if grad is None:
                continue
            if evidence_weight != 1.0:
                grad = grad * evidence_weight
            total.add_(grad)
            square.addcmul_(grad, grad)
            mirror.add_(grad, alpha=sign)
        if self.updates > self.warmup and (self.updates - self.warmup - 1) % self.refresh_every == 0:
            self.refresh()

    def _level(self, obs, null, squared=False):
        """The configured level, from an observed statistic and its twin.

        Both modes are invariant to a monotone transform of the statistic --
        `_fdp_level` is rank-based and the hinge works on squares -- so the
        pooled `input` granularity can hand over energies directly by setting
        ``squared``.
        """
        if self.level_mode == "mirror":
            return _fdp_level(obs, null)
        return _softhinge_level(obs if squared else obs.square(),
                                null if squared else null.square(),
                                self.sharpness)

    @torch.no_grad()
    def refresh(self):
        """Recompute the levels from the twin-calibrated null."""
        for total, square, mirror, level in zip(
                self.sums, self.sqs, self.mirrors, self.levels):
            scale = square.sqrt().clamp_min(1e-30)
            t_obs, t_null = total.abs() / scale, mirror.abs() / scale
            if self.granularity == "input" and t_obs.dim() > 1:
                # pool over output units: energy per incoming signal, and the
                # twin's pooled energy is that statistic's exact null
                pooled = t_obs.square().sum(dim=0)
                pooled_null = t_null.square().sum(dim=0)
                new = self._level(pooled, pooled_null, squared=True
                                  ).unsqueeze(0).expand_as(level)
            else:
                new = self._level(t_obs, t_null)
                if self.granularity == "unit" and new.dim() > 1:
                    new = new.mean(dim=tuple(range(1, new.dim())),
                                   keepdim=True).expand_as(level)
            if self.reallocate:
                # Conserve the tensor's step budget instead of shrinking it: shut
                # entries FUND confident ones. A level bounded by 1 can only ever
                # take smaller steps, so it cannot raise the step on the entries
                # that carry signal -- and their noise, not the junk's, is what
                # sets the usable learning rate (measured: the ORACLE's optimum LR
                # equals Adam's). Reallocation beat a told-the-answer oracle mask
                # by 1.40x. It gives up the never-amplifies property, so it can
                # take budget from clip_grad_norm_ and is not unconditionally safe.
                new = (new * (new.numel() / new.sum().clamp_min(1e-12))
                       ).clamp_(0.0, self.reallocate)
            level.copy_(new)

    @torch.no_grad()
    def after_step(self):
        """Rescale the step the optimizer actually took, per entry."""
        if self.updates <= self.warmup:
            return
        for param, level, snapshot in zip(self.params, self.levels, self.snapshots):
            # param <- w_before + level * (w_after - w_before), allocation-free
            param.sub_(snapshot).mul_(level).add_(snapshot)

    @torch.no_grad()
    def level_stats(self):
        """Mean and dispersion of the levels, for logging. One D2H sync."""
        flat = torch.cat([level.reshape(-1) for level in self.levels])
        return {"level_mean": flat.mean().item(), "level_std": flat.std().item(),
                "level_open": (flat > 0.5).to(torch.float32).mean().item()}
