"""One-bar-later available inputs, with the frozen stock target unchanged.

For row t and lag depth L, the original frame contains bars t..t+L-1;
the latest frame contains bars t+1..t+L. Both predict the ORIGINAL return at
bar t+L+1, centered and scaled using information through bar t+L. Forecasts
are therefore made after bar t+L. This is an information-set comparison,
not an optimizer-only improvement or a shifted-target experiment.

Channel arithmetic is copied from stock_stream.build_stream, retaining its
operation order and shared _ewma. Neither normalization nor target clipping
is changed. Both frames retain all B-L-1 original examples, including the
last available window that cannot be obtained by shifting the old matrix.

The fourth stream output transforms a raw-return-zero forecast with the known
trailing drift/scale. It is diagnostic only, not a model input or new target.
Clipping that transform is not the exact conditional expectation of clipped
return noise; transformed-target zero is also not raw-return zero.

The CUDA wrappers only route inputs to two independent frozen CorrelatedNIG
and AdamBank instances. No prior selection, meta learner, or new recurrence
is introduced; update paths never synchronize with the host.
"""

import numpy as np
import torch

from cleanrl.plasticity import stock_stream
from cleanrl.plasticity.predictive_correlated_nig_v5 import CorrelatedNIG
from cleanrl.plasticity.predictive_stock_transfer_eval_v1 import AdamBank


def build_paired_stream(bars, args):
    """Return FP32 (original, latest, target, causal_raw_zero), preserving labels.

    The supported stock configuration is center=True, raw_target=False,
    vol_feature=False, steps=0. Positive lag depths are supported for temporal
    checks; production uses 32. Other training-only Args fields are irrelevant
    to the frozen helper as well.
    """
    if isinstance(args.lags, bool) or not isinstance(args.lags, int) or args.lags <= 0:
        raise ValueError('lags must be a positive integer')
    if not args.center or args.raw_target or args.vol_feature or args.steps != 0:
        raise ValueError('paired stream requires center=True, raw_target=False, vol_feature=False, steps=0')

    # Keep the frozen helper's dtype choices and arithmetic ordering verbatim.
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
        volume - stock_stream._ewma(volume, args.vol_span),
        count - stock_stream._ewma(count, args.vol_span),
    ], axis=1)
    if args.center:
        mean = stock_stream._ewma(channels, args.vol_span, axis=0)
        channels = channels - np.concatenate([np.zeros_like(mean[:1]), mean[:-1]], axis=0)
    scale = stock_stream._ewma(np.abs(channels), args.vol_span, axis=0)
    scale = np.concatenate([scale[:1], scale[:-1]], axis=0)
    channels = channels / np.maximum(scale, 1e-12)
    np.clip(channels, -10.0, 10.0, out=channels)

    lags, n_chan = args.lags, channels.shape[1]
    total = channels.shape[0] - lags - 1
    if total <= 0:
        raise ValueError("not enough bars for the requested lag depth")
    original = np.empty((total, lags * n_chan), dtype=np.float32)
    latest = np.empty_like(original)
    for lag in range(lags):
        original[:, lag * n_chan:(lag + 1) * n_chan] = channels[lag:lag + total]
        latest[:, lag * n_chan:(lag + 1) * n_chan] = channels[lag + 1:lag + 1 + total]
    drift = stock_stream._ewma(ret, args.vol_span)
    centred = ret - np.concatenate([[0.0], drift[:-1]]) if args.center else ret
    vol = np.maximum(stock_stream._ewma(np.abs(centred), args.vol_span), 1e-12)
    target = (centred[lags + 1:lags + 1 + total] / vol[lags:lags + total]).astype(np.float32)
    np.clip(target, -10.0, 10.0, out=target)
    causal_raw_zero = (-drift[lags:lags + total] / vol[lags:lags + total]).astype(np.float32)
    np.clip(causal_raw_zero, -10.0, 10.0, out=causal_raw_zero)
    return original, latest, target, causal_raw_zero


class AvailableStateComparison:
    """Independent old-information control and latest-information frozen NIGs."""

    # Host-only evaluator schema for the frozen v5 default five-prior bank.
    # Instances derive their names from the actual frozen learners below.
    output_names = tuple(
        f'{frame}_{name}'
        for frame in ('original', 'latest')
        for name in (
            *(f'dense_{prior:g}' for prior in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)),
            *(f'diagonal_{prior:g}' for prior in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)),
            'dense_bayes_mixture', 'zero',
        )
    )

    def __init__(self, input_dim, device):
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0 or input_dim % 2:
            raise ValueError('input_dim must be a positive even packed dimension')
        self.input_dim = input_dim
        self.frame_dim = input_dim // 2
        self.original = CorrelatedNIG(self.frame_dim, device)
        self.latest = CorrelatedNIG(self.frame_dim, device)
        self.observations = torch.zeros((), device=device, dtype=torch.int64)
        self.output_names = (*(f'original_{name}' for name in self.original.output_names),
                             *(f'latest_{name}' for name in self.latest.output_names))
        self.configs = {
            'original': self.original.configs, 'latest': self.latest.configs,
            'input_dim': input_dim, 'frame_dim': self.frame_dim,
            'original_window': 'bars t..t+L-1', 'latest_window': 'bars t+1..t+L',
            'forecast_time': 'after bar t+L',
            'target': 'frozen ret[t+L+1], centered/scaled through bar t+L, clipped to +/-10',
            'primary': 'latest_dense_bayes_mixture', 'control': 'original_dense_bayes_mixture',
        }

    def state_tensors(self):
        return [*self.original.state_tensors(), *self.latest.state_tensors(), self.observations]

    @property
    def noise(self):
        return torch.cat((self.original.noise, self.latest.noise))

    @torch.no_grad()
    def diagnostics(self):
        """Host-only frozen diagnostics and all mixture probabilities, including null."""
        return {
            frame: {**model.diagnostics(),
                    'mixture_probabilities': model.log_weights.exp().cpu().tolist(),
                    'mixture_components': [*(f'dense_{prior:g}' for prior in model.priors), 'null']}
            for frame, model in (('original', self.original), ('latest', self.latest))
        }

    @torch.no_grad()
    def update(self, x, y):
        prediction = torch.cat((self.original.update(x[:self.frame_dim], y),
                                self.latest.update(x[self.frame_dim:], y)))
        self.observations.add_(1)
        return prediction


class PairedAdamBank:
    """Matched frozen linear/MLP Adam grids on the two information sets.

    Order is original linear, original MLP, latest linear, latest MLP. All
    nested mutable tensors, including the two full MLP forecast trajectories,
    belong to the capture/checkpoint state. Input slices remain views.
    """

    def __init__(self, xs, ys, grid):
        if xs.ndim != 2 or xs.shape[1] <= 0 or xs.shape[1] % 2:
            raise ValueError('xs must have a positive even packed feature dimension')
        self.configs = tuple(grid)
        dimension = xs.shape[1] // 2
        self.original = AdamBank(xs[:, :dimension], ys, self.configs)
        self.latest = AdamBank(xs[:, dimension:], ys, self.configs)
        self.scale = torch.cat((self.original.scale, self.latest.scale))
        self.prediction = torch.zeros(4 * len(self.configs), device=xs.device)
        self.mutable = [*self.original.mutable, *self.latest.mutable, self.prediction]
        self.output_names = tuple(f'{frame}_adam_{family}_{lr:g}'
                                  for frame in ('original', 'latest')
                                  for family in ('linear', 'mlp')
                                  for lr in self.configs)

    @property
    def index(self):
        return self.original.index

    @torch.no_grad()
    def update(self):
        self.original.update()
        self.latest.update()
        self.prediction.copy_(torch.cat((self.original.prediction, self.latest.prediction)))
