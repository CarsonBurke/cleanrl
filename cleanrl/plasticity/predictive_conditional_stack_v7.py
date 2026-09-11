"""Conditional residual regression on genuinely pre-label frozen-v6 forecasts.

The prior forecast is the frozen static mixture s, not zero: y = s + h'b + e,
with b | sigma² ~ N(0, tau I sigma²), sigma² ~ IG(2, 1). The NIG posterior is
exact conditional regression for the realized predictable designs. It does not
integrate uncertainty in the base learners, assert stationary financial noise,
or turn adaptive, label-history-dependent designs into an exogenous batch model.

Calibration learns only the amplitude of s. Plain stacking jointly learns an
unconstrained combination of s and all 15 centered expert forecasts; it does not
select a density winner. Context adds interactions with seven bounded summaries
of the latest eight available lag rows. These summaries use the original SPY
channel order and chronology (oldest row first), without changing its targets or
normalization. This is a structural contextual transformation, not a drop-in
optimizer replacement. A matched-feature Adam bank isolates optimizer effects.

Every family starts at the same static anchor, uses the same isotropic prior on
its active coordinates, and is embedded in the same feature dimension. There is
no intercept: zero base forecasts imply zero features and zero final forecasts,
even after learning. There are no gates, forgetting, clamps, or density-based
meta weights. Frozen base outputs are retained verbatim for independent audits.

CUDA only; callers disable TF32 and own compilation/capture. Matrices and
coefficients are FP32, NIG scalar statistics FP64. Covariance drift must be
reported, not repaired. update performs no host synchronization.
"""

import math

import torch

from cleanrl.plasticity import covariance_stock_eval_v1 as stock
from cleanrl.plasticity.predictive_dynamic_nig_v6 import DynamicNIG


def composition_features(x, base_predictions):
    """Return calibration/plain/context designs using only x and owned forecasts.

    base_predictions is the returned pre-label DynamicNIG output, never its
    updated coefficient state. The static mixture is coordinate zero; centered
    raw-expert contrasts are divided by sqrt(15). Context coordinates follow
    channel-major outer-product order. No current target is an input.
    """
    anchor = base_predictions[1]
    raw = base_predictions[4:]
    g = torch.cat((anchor.reshape(1), (raw - anchor) / math.sqrt(raw.numel())))
    rows = x.reshape(-1, 7)
    z = rows[-min(8, rows.shape[0]):].mean(0).tanh() / math.sqrt(7)
    interactions = (z[:, None] * g[None, :]).flatten()
    context = torch.cat((g, interactions))
    plain = torch.cat((g, torch.zeros_like(interactions)))
    calibration = torch.cat((anchor.reshape(1), torch.zeros_like(context[1:])))
    return torch.stack((calibration, plain, context))


def _validated_grid(values, name):
    values = tuple(float(value) for value in values)
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f'{name} must be nonempty, finite, and positive')
    if len({f'{value:g}' for value in values}) != len(values):
        raise ValueError(f'{name} must have distinct output names')
    return values


def _cuda_dimension(dimension, device):
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError('dimension must be a positive integer')
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('conditional stacking requires CUDA')
    return device


class ResidualNIGBank:
    """Three conditional Gaussian regressions, each with an independent tau grid."""

    families = ('calibration', 'stacking', 'context')

    def __init__(self, feature_dim, device, priors):
        device = _cuda_dimension(feature_dim, device)
        self.feature_dim = feature_dim
        self.priors = _validated_grid(priors, 'priors')
        self.configs = [{'family': family, 'prior': prior}
                        for family in self.families for prior in self.priors]
        self.output_names = tuple(f'{cfg["family"]}_{cfg["prior"]:g}' for cfg in self.configs)
        k = len(self.configs)
        self.mean = torch.zeros((k, feature_dim), device=device, dtype=torch.float32)
        self.cov = torch.zeros((k, feature_dim, feature_dim), device=device, dtype=torch.float32)
        prior_tensor = torch.tensor([cfg['prior'] for cfg in self.configs],
                                    device=device, dtype=torch.float32)
        self.cov.diagonal(dim1=-2, dim2=-1).copy_(prior_tensor[:, None])
        self.alpha = torch.tensor(2., device=device, dtype=torch.float64)
        self.beta = torch.ones(k, device=device, dtype=torch.float64)

    def state_tensors(self):
        """Complete mutable posterior, in stable checkpoint/capture order."""
        return [self.mean, self.cov, self.alpha, self.beta]

    @property
    def noise(self):
        return self.beta / (self.alpha - 1)

    @torch.no_grad()
    def update(self, features, residual_target):
        """Return pre-label correction means, then exact NIG conditioning."""
        h = features.repeat_interleave(len(self.priors), dim=0)
        u = torch.bmm(self.cov, h.unsqueeze(-1)).squeeze(-1)
        leverage = (h * u).sum(-1).double() + 1
        prediction = (self.mean * h).sum(-1)
        innovation = residual_target.double() - prediction.double()
        rank = u * leverage.rsqrt().float()[:, None]
        self.cov.sub_(rank[:, :, None] * rank[:, None, :])
        self.mean.add_(u * (innovation / leverage).float()[:, None])
        self.beta.add_(.5 * innovation.square() / leverage)
        self.alpha.add_(.5)
        return prediction


class ContextAdam:
    """Vanilla Adam residual regression on exactly the context NIG design.

    Update algebra is the frozen covariance_sparse_eval_v1.LinearLearner Adam
    branch: gradient of half squared error, beta=(.9,.999), epsilon=1e-5,
    zero initialization, one step per observation, without clipping or decay.
    """

    def __init__(self, feature_dim, device, lrs):
        device = _cuda_dimension(feature_dim, device)
        self.feature_dim = feature_dim
        self.lrs = _validated_grid(lrs, 'lrs')
        self.output_names = tuple(f'context_adam_{lr:g}' for lr in self.lrs)
        self.configs = {'lrs': self.lrs, 'betas': (.9, .999), 'epsilon': 1e-5,
                        'loss': 'half_squared_residual_error', 'weight_decay': 0.}
        self.weights = torch.zeros((len(self.lrs), feature_dim), device=device, dtype=torch.float32)
        self.m = torch.zeros_like(self.weights)
        self.v = torch.zeros_like(self.weights)
        self.steps = torch.zeros((), device=device, dtype=torch.float32)
        self._scale = torch.tensor(self.lrs, device=device, dtype=torch.float32)

    def state_tensors(self):
        return [self.weights, self.m, self.v, self.steps]

    @torch.no_grad()
    def update(self, h, residual_target):
        prediction = (self.weights * h).sum(-1)
        grad = (prediction - residual_target)[:, None] * h
        self.steps.add_(1)
        self.m.lerp_(grad, .1)
        self.v.lerp_(grad.square(), .001)
        update = self._scale[:, None] * (self.m / (1 - .9 ** self.steps)) / (
            (self.v / (1 - .999 ** self.steps)).sqrt() + 1e-5)
        self.weights.sub_(update)
        return prediction


class StateConditionedStack:
    """Frozen experts plus three residual NIG families and matched-context Adam."""

    def __init__(self, input_dim, device, priors=(.01, .1, 1.)):
        device = _cuda_dimension(input_dim, device)
        if input_dim % 7:
            raise ValueError('input_dim must contain complete seven-channel lag rows')
        self.input_dim = input_dim
        self.base = DynamicNIG(input_dim, device)
        self.feature_dim = 8 * (1 + len(self.base.configs))
        self.meta = ResidualNIGBank(self.feature_dim, device, priors)
        self.priors = self.meta.priors
        self.context_adam = ContextAdam(self.feature_dim, device, stock.Args.adam_lrs)
        self.observations = torch.zeros((), device=device, dtype=torch.int64)
        self.output_names = (*self.meta.output_names,
                             *(f'base_{name}' for name in self.base.output_names),
                             *self.context_adam.output_names)
        self.configs = {
            'base': self.base.configs, 'meta': self.meta.configs,
            'context_adam': self.context_adam.configs,
            'input_dim': input_dim, 'feature_dim': self.feature_dim,
            'composition_dim': 1 + len(self.base.configs), 'context_dim': 7,
            'context_lags': min(8, input_dim // 7),
            'context_transform': 'tanh(mean(latest_lag_rows, axis=0))/sqrt(7)',
            'composition_transform': '[static, (raw_experts-static)/sqrt(raw_count)]',
            'context_design': '[composition, outer(context,composition).flatten()]',
            'controls': 'zero-padded calibration and plain stacking in the same feature space',
            'anchor': 'prelabel frozen static mixture', 'intercept': False,
            'initial_alpha': 2., 'initial_beta': 1.,
            'coefficient_dtype': 'float32', 'scalar_statistics_dtype': 'float64',
        }

    def state_tensors(self):
        return [*self.base.state_tensors(), *self.meta.state_tensors(),
                *self.context_adam.state_tensors(), self.observations]

    @property
    def noise(self):
        return torch.cat((self.base.noise, self.meta.noise))

    @torch.no_grad()
    def diagnostics(self):
        """Host-only posterior scales and interpretable context coefficient blocks."""
        t = len(self.priors)
        context = self.meta.mean[2 * t:]
        g_dim = 1 + len(self.base.configs)
        return {
            'base': self.base.diagnostics(),
            'meta': {'alpha': float(self.meta.alpha.cpu()),
                     'beta': self.meta.beta.cpu().tolist(),
                     'noise': self.meta.noise.cpu().tolist(),
                     'noise_std': self.meta.noise.sqrt().cpu().tolist(),
                     'coefficient_norms': self.meta.mean.norm(dim=-1).cpu().tolist(),
                     'context_composition_coefficients': context[:, :g_dim].cpu().tolist(),
                     'context_interaction_coefficients': context[:, g_dim:].reshape(t, 7, g_dim).cpu().tolist(),
                     'context_interaction_norms': context[:, g_dim:].norm(dim=-1).cpu().tolist()},
            'context_adam': {'coefficient_norms': self.context_adam.weights.norm(dim=-1).cpu().tolist()},
        }

    @torch.no_grad()
    def update(self, x, y):
        # DynamicNIG mutates itself, but its owned return contains ONLY pre-label
        # forecasts. Neither feature construction nor meta prediction reads the
        # now-updated base means, covariances, probabilities, or noise.
        base_prediction = self.base.update(x, y)
        features = composition_features(x, base_prediction)
        anchor = base_prediction[1]
        residual = y - anchor
        correction = self.meta.update(features, residual)
        adam_correction = self.context_adam.update(features[2], residual)
        prediction = torch.cat((correction + anchor, base_prediction, adam_correction + anchor))
        self.observations.add_(1)
        return prediction
