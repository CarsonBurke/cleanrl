"""
Dreamer4 World Model — Core Utilities
======================================
Shared helpers, data structures, tensor ops, loss normalization, SymExp TwoHot encoding.
Self-contained: only depends on torch and standard library.
"""
from __future__ import annotations
from typing import Callable
from math import ceil, log2
from random import random
from functools import partial, wraps
from dataclasses import dataclass, asdict
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor, cat, stack, tensor, arange, randn, randn_like, full, linspace, ones, zeros, rand, randint, empty
from torch.nn import Module, Linear, RMSNorm, Parameter, Sequential, Embedding
from torch.distributions import Normal, Beta, kl_divergence

# ──────────────────────────────────────────────
# Named tuples for structured returns
# ──────────────────────────────────────────────

WorldModelLosses = namedtuple('WorldModelLosses', (
    'flow', 'shortcut', 'rewards', 'continuous_actions',
    'state_pred', 'agent_state_pred',
))
TokenizerLosses = namedtuple('TokenizerLosses', ('recon', 'time_decorr', 'space_decorr'))

Predictions = namedtuple('Predictions', ['flow', 'proprioception', 'state'])
Embeds = namedtuple('Embeds', ['agent', 'state_pred'])
Actions = namedtuple('Actions', ['continuous'])

MaybeTensor = Tensor | None

# ──────────────────────────────────────────────
# Experience dataclass
# ──────────────────────────────────────────────

@dataclass
class Experience:
    latents: Tensor
    proprio: MaybeTensor = None
    agent_embed: MaybeTensor = None
    rewards: Tensor | None = None
    actions: Actions | None = None
    log_probs: MaybeTensor = None
    old_action_params: tuple[Tensor, Tensor] | None = None  # (mean, log_var)
    values: MaybeTensor = None
    step_size: int | None = None
    lens: MaybeTensor = None
    is_truncated: MaybeTensor = None
    is_from_world_model: bool | Tensor = True

    def cpu(self):
        return self.to(torch.device('cpu'))

    def to(self, device):
        from torch.utils._pytree import tree_map
        d = asdict(self)
        d = tree_map(lambda t: t.to(device) if isinstance(t, Tensor) else t, d)
        return Experience(**d)


# ──────────────────────────────────────────────
# Basic helpers
# ──────────────────────────────────────────────

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def is_power_two(num):
    return log2(num).is_integer()

def first(arr):
    return arr[0]

def xnor(x, y):
    return x == y

def has_at_least_one(*bools):
    return sum([*map(int, bools)]) > 0


# ──────────────────────────────────────────────
# Tensor helpers
# ──────────────────────────────────────────────

def l2norm(t):
    return F.normalize(t, dim=-1, p=2)

def softclamp(t, value=50.):
    return (t / value).tanh() * value

def log_safe(t, eps=1e-20):
    return t.clamp(min=eps).log()

def straight_through(src, tgt):
    return tgt + src - src.detach()

def frac_gradient(t, frac=1.):
    t_grad = t * frac
    return straight_through(t_grad, t.detach())

def safe_cat(tensors, dim=0):
    """Cat tensors filtering out None values."""
    if isinstance(tensors, (list, tuple)):
        tensors = [t for t in tensors if exists(t)]
        if len(tensors) == 0:
            return None
        return cat(tensors, dim=dim)
    return tensors

def safe_stack(tensors, dim=0):
    tensors = [t for t in tensors if exists(t)]
    if len(tensors) == 0:
        return None
    return stack(tensors, dim=dim)

def lens_to_mask(lens, max_len):
    """(b,) -> (b, max_len) boolean mask."""
    arange_t = arange(max_len, device=lens.device)
    return arange_t.unsqueeze(0) < lens.unsqueeze(1)

def masked_mean(t, mask=None, dim=None):
    if not exists(mask):
        return t.mean() if dim is None else t.mean(dim=dim)
    mask = mask.float()
    if dim is None:
        return (t * mask).sum() / mask.sum().clamp(min=1.)
    return (t * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1.)

def pad_at_dim(t, padding, dim, value=0.):
    """Pad tensor at a specific dimension."""
    ndim = t.ndim
    pad_list = [0] * (2 * ndim)
    # F.pad pads from last dim backwards
    idx = 2 * (ndim - 1 - dim)
    pad_list[idx] = padding[0]
    pad_list[idx + 1] = padding[1]
    return F.pad(t, pad_list, value=value)

def pad_right_at_dim_to(t, target_len, dim=1, value=0.):
    """Pad a tensor to target_len along dim."""
    curr_len = t.shape[dim]
    if curr_len >= target_len:
        return t
    return pad_at_dim(t, (0, target_len - curr_len), dim, value)

def align_dims_left(pair):
    """Expand first tensor to match ndim of second by adding trailing dims."""
    t, ref = pair
    while t.ndim < ref.ndim:
        t = t.unsqueeze(-1)
    return t, ref


# ──────────────────────────────────────────────
# Multi-token prediction targets
# ──────────────────────────────────────────────

def create_multi_token_prediction_targets(t, steps_future):
    """
    t: (b, seq_len, ...)
    Returns: (b, seq_len, steps_future, ...) targets and (b, seq_len, steps_future) mask
    """
    batch, seq_len = t.shape[:2]
    device = t.device

    seq_arange = arange(seq_len, device=device)
    steps_arange = arange(steps_future, device=device)

    # indices: (seq_len, steps_future)
    indices = seq_arange.unsqueeze(1) + steps_arange.unsqueeze(0)
    mask = indices < seq_len
    indices = indices.clamp(max=seq_len - 1)

    # gather
    batch_arange = arange(batch, device=device).view(batch, 1, 1)
    out = t[batch_arange, indices.unsqueeze(0).expand(batch, -1, -1)]

    mask = mask.unsqueeze(0).expand(batch, -1, -1)
    return out, mask


# ──────────────────────────────────────────────
# Loss Normalizer (from Dreamer4 paper)
# ──────────────────────────────────────────────

class LossNormalizer(Module):
    def __init__(self, num_losses=1, beta=0.95, eps=1e-6):
        super().__init__()
        self.register_buffer('exp_avg_sq', torch.ones(num_losses))
        self.beta = beta
        self.eps = eps

    def forward(self, losses: Tensor, update_ema=None):
        update_ema = default(update_ema, self.training)
        rms = self.exp_avg_sq.sqrt()

        if update_ema:
            decay = 1. - self.beta
            self.exp_avg_sq.lerp_(losses.detach().square(), decay)

        assert losses.numel() == rms.numel()
        return losses / rms.clamp(min=self.eps)


# ──────────────────────────────────────────────
# SymExp TwoHot Encoding (DreamerV3 style)
# ──────────────────────────────────────────────

class SymExpTwoHot(Module):
    """Symmetric exponential two-hot encoding for reward/value prediction."""

    def __init__(self, reward_range=(-20., 20.), num_bins=255, learned_embedding=False, dim_embed=None):
        super().__init__()
        min_value, max_value = reward_range
        values = linspace(min_value, max_value, num_bins)
        values = values.sign() * (torch.exp(values.abs()) - 1.)
        self.reward_range = reward_range
        self.num_bins = num_bins
        self.learned_embedding = learned_embedding
        self.register_buffer('bin_values', values)

        if learned_embedding:
            assert exists(dim_embed)
            self.bin_embeds = nn.Embedding(num_bins, dim_embed)

    @property
    def device(self):
        return self.bin_values.device

    def bins_to_scalar_value(self, logits, normalize=True):
        weights = logits.softmax(dim=-1) if normalize else logits
        return (weights * self.bin_values).sum(dim=-1)

    def embed(self, two_hot_encoding):
        assert self.learned_embedding, 'can only embed if learned_embedding=True'
        weights, bin_indices = two_hot_encoding.topk(k=2, dim=-1)
        two_embeds = self.bin_embeds(bin_indices)
        return (two_embeds * weights.unsqueeze(-1)).sum(dim=-2)

    def forward(self, values):
        """Encode scalar values to two-hot vectors."""
        bin_values = self.bin_values
        min_val, max_val = bin_values[0], bin_values[-1]

        orig_shape = values.shape
        values = values.reshape(-1)
        num_values = values.shape[0]

        values = values.clamp(min=min_val, max=max_val)

        indices = torch.searchsorted(bin_values, values)
        left_indices = (indices - 1).clamp(min=0)
        right_indices = (left_indices + 1).clamp(max=self.num_bins - 1)

        left_values = bin_values[left_indices]
        right_values = bin_values[right_indices]

        total_distance = (right_values - left_values).clamp(min=1e-8)
        left_weight = (right_values - values) / total_distance
        right_weight = 1. - left_weight

        encoded = torch.zeros(num_values, self.num_bins, device=self.device)
        encoded.scatter_(-1, left_indices.unsqueeze(-1), left_weight.unsqueeze(-1))
        encoded.scatter_(-1, right_indices.unsqueeze(-1), right_weight.unsqueeze(-1))

        return encoded.reshape(*orig_shape, self.num_bins)


# ──────────────────────────────────────────────
# GAE with associative scan
# ──────────────────────────────────────────────

@torch.no_grad()
def calc_gae(rewards, values, masks=None, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation.

    Args:
        rewards: (batch, time)
        values: (batch, time)
        masks: (batch, time) — 1 where episode continues, 0 at terminal
        gamma: discount factor
        lam: GAE lambda
    Returns:
        returns: (batch, time)
    """
    if not exists(masks):
        masks = torch.ones_like(values)

    values_padded = F.pad(values, (0, 1), value=0.)
    values_curr, values_next = values_padded[..., :-1], values_padded[..., 1:]

    delta = rewards + gamma * values_next * masks - values_curr
    gates = gamma * lam * masks

    # reverse cumulative scan
    batch, time = delta.shape
    gae = torch.zeros_like(delta)
    last_gae = torch.zeros(batch, device=delta.device)

    for t in reversed(range(time)):
        last_gae = delta[:, t] + gates[:, t] * last_gae
        gae[:, t] = last_gae

    returns = gae + values_curr
    return returns


# ──────────────────────────────────────────────
# Ramp weight (eq 8 in Dreamer4 paper)
# ──────────────────────────────────────────────

def ramp_weight(times, slope=0.9, intercept=0.1):
    return slope * times + intercept


# ──────────────────────────────────────────────
# BetaDist helper for state prediction
# ──────────────────────────────────────────────

class BetaDist(Module):
    def __init__(self, unimodal=True):
        super().__init__()
        self.unimodal = unimodal

    def forward(self, params):
        alpha, beta = params.unbind(dim=-1)
        offset = 1. if self.unimodal else 0.
        alpha = F.softplus(alpha) + offset
        beta = F.softplus(beta) + offset
        return Beta(alpha, beta)


# ──────────────────────────────────────────────
# MLP builder
# ──────────────────────────────────────────────

def build_mlp(dim_in, dim_hidden, dim_out, depth=3, activation=nn.ReLU):
    """Local equivalent of the reference create_mlp(..., dim=dim_hidden, depth=depth)."""
    hidden_dims = (dim_hidden,) * (depth + 1)
    dims = (dim_in, *hidden_dims, dim_out)

    layers = []
    for i, (layer_in, layer_out) in enumerate(zip(dims[:-1], dims[1:]), start=1):
        is_last = i == (len(dims) - 1)
        modules = [Linear(layer_in, layer_out)]
        if not is_last:
            modules.append(nn.LayerNorm(layer_out))
            modules.append(activation())
        layers.append(nn.Sequential(*modules))

    return Sequential(*layers)


def covariance_off_diagonal_loss(x: Tensor, eps: float = 1e-5) -> Tensor:
    """Penalize feature covariance off-diagonals on a 2D tensor shaped (samples, dim)."""
    assert x.ndim == 2
    if x.shape[0] < 2 or x.shape[1] < 2:
        return x.new_zeros(())

    x = x - x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp(min=eps)
    x = x / std
    cov = x.t() @ x / x.shape[0]
    off_diag = cov - torch.diag_embed(torch.diagonal(cov))
    return off_diag.square().mean()


# ──────────────────────────────────────────────
# Ensemble module for multi-token reward prediction
# ──────────────────────────────────────────────

class Ensemble(Module):
    """Create N independent copies of a module for ensemble prediction."""

    def __init__(self, module_fn, n_copies):
        super().__init__()
        self.copies = nn.ModuleList([module_fn() for _ in range(n_copies)])

    def forward(self, x):
        """Returns stacked predictions: (n_copies, batch, ..., out_dim)."""
        return stack([copy(x) for copy in self.copies], dim=0)

    def forward_one(self, x, id=0):
        """Forward through a single copy."""
        return self.copies[id](x)


# ──────────────────────────────────────────────
# State tokenizer for MuJoCo (no images, just state vectors)
# ──────────────────────────────────────────────

class StateTokenizer(Module):
    """
    For MuJoCo environments where observations are state vectors (not images).
    Encodes observation vectors into latent tokens compatible with the dynamics model.

    Unlike the VideoTokenizer which uses patches + transformer, this uses a simple
    MLP encoder/decoder since observations are low-dimensional vectors.
    """

    def __init__(
        self,
        dim_obs: int,
        dim_latent: int = 32,
        num_latent_tokens: int = 4,
        dim_hidden: int = 256,
        encoder_depth: int = 2,
        decoder_depth: int = 2,
        encoder_add_decorr_aux_loss: bool = True,
        time_decorr_loss_weight: float = 1e-3,
        space_decorr_loss_weight: float = 1e-3,
    ):
        super().__init__()
        self.dim_obs = dim_obs
        self.dim_latent = dim_latent
        self.num_latent_tokens = num_latent_tokens
        self.encoder_add_decorr_aux_loss = encoder_add_decorr_aux_loss
        self.time_decorr_loss_weight = time_decorr_loss_weight
        self.space_decorr_loss_weight = space_decorr_loss_weight

        # Encoder: obs -> latent tokens
        self.encoder = Sequential(
            Linear(dim_obs, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.SiLU(),
            *[
                nn.Sequential(
                    Linear(dim_hidden, dim_hidden),
                    nn.LayerNorm(dim_hidden),
                    nn.SiLU(),
                )
                for _ in range(encoder_depth - 1)
            ],
            Linear(dim_hidden, num_latent_tokens * dim_latent),
            nn.Tanh(),
        )

        # Decoder: latent tokens -> obs reconstruction
        self.decoder = Sequential(
            Linear(num_latent_tokens * dim_latent, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.SiLU(),
            *[
                nn.Sequential(
                    Linear(dim_hidden, dim_hidden),
                    nn.LayerNorm(dim_hidden),
                    nn.SiLU(),
                )
                for _ in range(decoder_depth - 1)
            ],
            Linear(dim_hidden, dim_obs),
        )

        self.register_buffer('zero', tensor(0.), persistent=False)

    @property
    def device(self):
        return self.zero.device

    @torch.no_grad()
    def tokenize(self, obs):
        """
        obs: (batch, time, dim_obs)
        Returns: (batch, time, num_latent_tokens, dim_latent)
        """
        self.eval()
        return self.forward(obs, return_latents=True)

    def decode(self, latents):
        """
        latents: (batch, time, num_latent_tokens, dim_latent)
        Returns: (batch, time, dim_obs)
        """
        b, t, n, d = latents.shape
        assert (n, d) == (self.num_latent_tokens, self.dim_latent)
        flat = latents.reshape(b * t, n * d)
        recon = self.decoder(flat)
        return recon.reshape(b, t, self.dim_obs)

    def forward(self, obs, return_latents: bool = False, return_intermediates: bool = False):
        """
        Training forward: encode, decode, return reconstruction loss.
        obs: (batch, time, dim_obs)
        """
        b, t, d = obs.shape
        flat = obs.reshape(b * t, d)

        latent_flat = self.encoder(flat)
        latents = latent_flat.reshape(b, t, self.num_latent_tokens, self.dim_latent)

        if return_latents:
            return latents

        recon = self.decode(latents)

        recon_loss = F.mse_loss(recon, obs)

        time_decorr_loss = self.zero
        space_decorr_loss = self.zero

        if self.encoder_add_decorr_aux_loss:
            if t > 1:
                time_tokens = latents.mean(dim=2).reshape(b * t, self.dim_latent)
                time_decorr_loss = covariance_off_diagonal_loss(time_tokens)

            if self.num_latent_tokens > 1:
                space_tokens = latents.mean(dim=1).reshape(b * self.num_latent_tokens, self.dim_latent)
                space_decorr_loss = covariance_off_diagonal_loss(space_tokens)

        total_loss = (
            recon_loss +
            time_decorr_loss * self.time_decorr_loss_weight +
            space_decorr_loss * self.space_decorr_loss_weight
        )

        if not return_intermediates:
            return total_loss, latents

        losses = TokenizerLosses(recon_loss, time_decorr_loss, space_decorr_loss)
        return total_loss, latents, losses, recon


# ──────────────────────────────────────────────
# Seeded RNG helpers
# ──────────────────────────────────────────────

def with_seed(seed):
    def decorator(fn):
        if not exists(seed):
            return fn

        @wraps(fn)
        def inner(*args, **kwargs):
            has_cuda = torch.cuda.is_available()
            orig_torch_state = torch.get_rng_state()
            orig_cuda_states = torch.cuda.get_rng_state_all() if has_cuda else None

            torch.manual_seed(seed)
            if has_cuda:
                torch.cuda.manual_seed_all(seed)

            try:
                out = fn(*args, **kwargs)
            finally:
                torch.set_rng_state(orig_torch_state)
                if has_cuda and orig_cuda_states:
                    torch.cuda.set_rng_state_all(orig_cuda_states)

            return out
        return inner
    return decorator

def sample_prob(prob):
    return rand(1).item() < prob
