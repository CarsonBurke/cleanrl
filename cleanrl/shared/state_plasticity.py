"""State-dependent plasticity as a general mechanism, not an RL trick.

PREMISE. Each perceptron decides, from ITS OWN STATE on the current sample, how
much of that sample's gradient is allowed to move its incoming weights. Nothing
here is RL-specific: the mechanism sees only ``(x, z, delta)`` -- a layer's
input, its pre-activation, and the incoming pre-activation gradient -- which
exist identically in LLM pretraining, time-series forecasting, and PPO. The
same object is therefore usable for any of them.

WHY BOTH MOMENTS. Write a unit's per-sample gradient contribution as
``g = mu + eps``. Earlier versions of this family fitted only ``E[g^2]``, i.e.
they forced ``mu = 0``. That is blind to signal by construction: for a useless
input the residual energy is ``delta^2`` and for a predictive input it is ALSO
``delta^2``. Only the first moment differs. So the predictor here has TWO heads:

    mu_hat(s)     predicted mean of delta given the unit's state
    n_hat(s)      predicted variance of delta given the unit's state

trained jointly by the heteroscedastic Gaussian NLL

    L_gate = mean[ (delta - mu_hat)^2 / n_hat + log n_hat ]

which is minimised at the true conditional mean and variance. Two products come
out of one predictor, and they are used in the two different places the
optimizer permits:

  DIRECTION (per sample, pre-optimizer). GLS weighting ``w_t = n_ref / n_hat_t``
  reweights samples within the batch. This changes the summed gradient's
  direction, which survives Adam. Note the correct GLS weight involves ONLY the
  conditional variance -- but it must be the CENTERED variance, which is exactly
  what needs the mean head to estimate honestly.

  MAGNITUDE (per unit, post-optimizer). A per-sample magnitude cannot survive
  Adam: beta1 attenuates one batch to ~10% of the intended effect and beta2
  divides out anything persistent. Measured, asking for a 0.125x step:
  pre-optimizer 0.897 for one batch and 1.00 sustained, versus post-optimizer
  0.124 at any duration. So magnitude is applied to the REALIZED step, and the
  finest granularity available there is per unit per update.

SQUARING AN ESTIMATED MEAN IS BIASED, AND THE BIAS POINTS THE WRONG WAY.
``E[|rbar|^2] = |mu|^2 + Var(rbar)``, and ``Var(rbar)`` grows with the noise, so
a naive signal estimate is inflated exactly where the data is least reliable.
Measured on the regime stream, the undebiased statistic gets reliability
BACKWARDS. For a batch estimator the correction is free and exact, because the
variance of the mean is already computed for the denominator:

    signal = max(|rbar|^2 - Var(rbar), 0)

For a streaming EMA estimator, where there is no within-batch replication to
estimate ``Var`` from, use ``sign_control=True``: a twin estimator is run on the
sign-randomized stream ``eps_t * g_t`` with ``eps_t`` in {-1,+1} shared across
units. Its true mean is exactly zero while its energy, sparsity, horizon and
step size are identical, so its square IS the estimator's own variance floor.

ANCHORING. The level's reference must NOT be the cross-unit arithmetic mean.
When most units are unreliable the mean sits AT the unreliable value, so the few
good units saturate the envelope and the measured dispersion is the envelope's
ceiling rather than the statistic's resolution. Anchoring near the top of the
distribution instead means the most reliable unit keeps its step and everything
else is suppressed relative to it, which also matches the asymmetry of the
envelope: gain lives in [0, 1] and noise is heavy-tailed upward.

THE LEVEL IS MEASURED, NOT PREDICTED. The per-unit level is the realized
signal-to-noise of the reweighted mean estimator,

    gain_i = |rbar_i|^2 / (|rbar_i|^2 + var_i)   in [0, 1]

with ``rbar`` the weighted mean contribution and ``var`` the variance of that
mean. This is the Wiener/MMSE shrinkage of the unit's own gradient estimate, so
better weights raise the gain and the level is exactly how much the estimate
improved. Nothing predicts the level, so it cannot be self-fulfilling.

SEPARATING THE TWO FACTORS. Any level decomposes into a uniform part (a
learning-rate change) and a dispersion part (WHICH units get the plasticity).
Top-anchoring makes the uniform part systematically less than one, so with
``conserve=True`` -- the default -- the level is divided by its own geometric
mean, forcing the uniform part to exactly one and leaving only dispersion. Then
no score change can be bought with an LR change, which is the trap this family
fell into four times. The pre-normalisation uniform factor is still exported so
the size of what was removed is visible, and ``conserve=False`` deliberately
re-admits it for the "this entire batch is unreliable" capability -- which must
then be judged against a matched learning-rate control, never against the
default LR.

UNIFORM COMPONENT. A level whose cross-unit mean drifts is a learning-rate
change wearing a mechanism's clothes; this family has manufactured that fake
win four separate times. Earlier versions removed it by forcing the geometric
mean to exactly one, which also made "this entire batch is noise"
inexpressible. Here the uniform component is PERMITTED but measured: the
realized geometric mean is exported every step as ``uniform`` so any score
change can be checked against a matched learning-rate control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def state_features(z):
    """A unit's own state, bounded and scale-free.

    Scale-free matters: a unit must not be able to earn plasticity merely by
    being loud, so the pre-activation is normalised by its own batch scale
    before being squashed. Returns a signed feature and a magnitude feature.
    """
    zn = z * torch.rsqrt(z.square().mean(0, keepdim=True) + 1e-8)
    zs = zn.square()
    return zn / (1.0 + zn.abs()), zs / (1.0 + zs) - 0.5


class TwoMomentPredictor(nn.Module):
    """Per-unit conditional mean and variance of ``delta``, read off the state.

    ``ctx_dim`` gives a low-rank view of the layer input so a unit can condition
    on WHICH region of input space it is in, not only on its own activation.
    Every parameter is zero at init except the fixed random projection, so
    ``mu_hat == 0``, ``log n_hat == 0`` and every multiplier is exactly one:
    a model wrapped in this is bit-identical to the unwrapped model at step 0.
    """

    def __init__(self, in_features, out_features, ctx_dim=8, noise_cap=2.0,
                 mean_cap=4.0):
        super().__init__()
        self.noise_cap = float(noise_cap)
        self.mean_cap = float(mean_cap)
        self.ctx_proj = nn.Parameter(torch.empty(int(ctx_dim), in_features))
        nn.init.orthogonal_(self.ctx_proj, 1.0)
        self.ctx_proj.requires_grad_(False)          # a fixed random view
        self.noise_read = nn.Parameter(torch.zeros(out_features, int(ctx_dim)))
        self.noise_state = nn.Parameter(torch.zeros(out_features))
        self.noise_mag = nn.Parameter(torch.zeros(out_features))
        self.noise_bias = nn.Parameter(torch.zeros(out_features))
        # Free ABSOLUTE level. The state readout is capped and cannot track a
        # level that drifts by tens of nats as gradients shrink; this can.
        self.noise_level = nn.Parameter(torch.zeros(out_features))
        self.mean_read = nn.Parameter(torch.zeros(out_features, int(ctx_dim)))
        self.mean_state = nn.Parameter(torch.zeros(out_features))
        self.mean_mag = nn.Parameter(torch.zeros(out_features))
        self.mean_scale = nn.Parameter(torch.zeros(out_features))

    def context(self, x):
        return torch.tanh(F.linear(x, self.ctx_proj))

    def forward(self, x, z):
        """Returns ``(mu_hat, log_n_state, log_n_absolute)``.

        The state part of the noise is returned separately because the GLS
        weight uses only that part -- the free absolute level must cancel out of
        a weight, or the weight would track the global gradient scale instead of
        the unit's state.
        """
        f_state, f_mag = state_features(z)
        ctx = self.context(x)
        raw_noise = (self.noise_bias + self.noise_state * f_state
                     + self.noise_mag * f_mag + F.linear(ctx, self.noise_read))
        log_n_state = self.noise_cap * torch.tanh(raw_noise / self.noise_cap)
        raw_mean = (self.mean_state * f_state + self.mean_mag * f_mag
                    + F.linear(ctx, self.mean_read))
        # scaled by a free per-unit magnitude: delta's scale is unknown a priori
        mu_hat = self.mean_scale * torch.tanh(raw_mean / self.mean_cap)
        return mu_hat, log_n_state, self.noise_level + log_n_state

    def nll(self, delta, mu_hat, log_n):
        """Heteroscedastic Gaussian NLL against the UNNORMALIZED residual.

        Unnormalized on purpose: the optimum is then the ABSOLUTE conditional
        variance, which is what lets ``noise_level`` learn the drifting scale
        and frees the state part to report an entire batch as unreliable.
        """
        residual = (delta - mu_hat).square()
        return (log_n + residual * torch.exp(-log_n)).mean()


class PlasticityState:
    """Buffers and reductions shared by the direction and magnitude paths."""

    def __init__(self, out_features, device, ref_beta=0.99,
                 weight_suppress=8.0, weight_inflate=2.0, anchor_q=0.9,
                 conserve=True):
        self.ref_beta = float(ref_beta)
        self.anchor_q = float(anchor_q)
        self.conserve = bool(conserve)
        # asymmetric on purpose: gain lives in [0, 1] and noise is heavy-tailed
        # upward, so suppression has room to run while confidence has a ceiling
        self.lo = 1.0 / float(weight_suppress)
        self.hi = float(weight_inflate)
        self.noise_ref = torch.zeros(out_features, device=device)
        self.gain_ref = torch.ones((), device=device)
        self.level = torch.ones(out_features, device=device)
        # exported telemetry: [uniform, dispersion, mean_r2, mean_gain]
        self.stats = torch.zeros(4, device=device)

    @torch.no_grad()
    def sample_weights(self, log_n_state):
        """GLS direction weights: inverse conditional variance, referenced."""
        return torch.exp(self.noise_ref - log_n_state).clamp(self.lo, self.hi)

    @torch.no_grad()
    def observe(self, log_n_state, delta, mu_hat):
        """Track the slow noise reference and the explained variance of the mean."""
        self.noise_ref.mul_(self.ref_beta).add_(
            log_n_state.mean(0), alpha=1.0 - self.ref_beta)
        total = delta.var(0, unbiased=False) + 1e-12
        residual = (delta - mu_hat).var(0, unbiased=False)
        self.stats[2] = (1.0 - residual / total).clamp(-1.0, 1.0).mean()

    @torch.no_grad()
    def update_level(self, x, delta, weights, exponent=1.0):
        """Wiener gain of the reweighted per-unit mean estimator.

        For unit i the augmented contribution is ``r_t = delta_{t,i} [x_t, 1]``.
        ``|rbar|^2`` is the signal and the weighted variance of the mean is the
        noise, so ``gain = signal / (signal + noise)`` is the MMSE shrinkage of
        this unit's own gradient estimate. Measured from data, never predicted.
        """
        rows = delta.shape[0]
        norm = weights.sum(0).clamp_min(1e-12)
        wd = weights * delta
        weight_mean = (wd.transpose(0, 1) @ x) / norm.unsqueeze(1)
        bias_mean = wd.sum(0) / norm
        signal = weight_mean.square().sum(1) + bias_mean.square()
        energy_scale = x.square().sum(1, keepdim=True) + 1.0
        cross = F.linear(x, weight_mean, bias_mean)
        centered = (delta.square() * energy_scale - 2.0 * delta * cross
                    + signal).clamp_min(0.0)
        # variance of a weighted mean of `rows` samples
        variance = (weights.square() * centered).sum(0) / norm.square()
        # debias: E[|rbar|^2] = |mu|^2 + Var(rbar), and the inflation is largest
        # exactly where the noise is worst, which inverts the statistic
        signal = (signal - variance).clamp_min(0.0)
        gain = signal / (signal + variance + 1e-12)
        # anchor near the TOP, not the mean: with most units unreliable a
        # central anchor pins the reference at the unreliable value and the
        # good units merely saturate the envelope
        reference = torch.quantile(gain.float(), self.anchor_q)
        self.gain_ref.mul_(self.ref_beta).add_(reference, alpha=1.0 - self.ref_beta)
        level = (gain / self.gain_ref.clamp_min(1e-12)).pow(exponent)
        level = level.clamp(self.lo, self.hi)
        uniform = level.log().mean().exp()
        if self.conserve:
            # strip the LR component: dispersion only, geometric mean exactly 1
            level = (level / uniform).clamp(self.lo, self.hi)
        self.level.copy_(level)
        self.stats[0] = uniform          # what was removed, reported either way
        self.stats[1] = self.level.std()
        self.stats[3] = gain.mean()
        return self.level


class PostStepLevel:
    """Applies per-unit levels to the REALIZED optimizer step.

    Adam is invariant to a per-row gradient rescale, so the level cannot be
    applied to the gradient. Snapshot before ``optimizer.step()``, then correct

        w_i += (level_i - 1) * (w_i_after - w_i_before)

    which leaves the optimizer's moments untouched and is bit-exact for a
    neutral unit. Works with any optimizer, not just Adam.
    """

    def __init__(self, plan):
        """``plan``: list of ``(param, level_source, is_matrix)``."""
        self.plan = list(plan)
        self.snapshots = [torch.empty_like(param) for param, _, _ in self.plan]

    @torch.no_grad()
    def stash(self):
        torch._foreach_copy_(self.snapshots, [param for param, _, _ in self.plan])

    @torch.no_grad()
    def apply(self):
        for (param, source, is_matrix), snapshot in zip(self.plan, self.snapshots):
            offset = source() - 1.0
            gain = offset.unsqueeze(1) if is_matrix else offset
            snapshot.sub_(param).mul_(gain)
            param.sub_(snapshot)
