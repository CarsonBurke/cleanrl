"""
Dreamer4 World Model — Action Embedder & Policy Components
============================================================
Continuous-only action embedding, unembedding, sampling, and log-prob computation
for MuJoCo continuous control tasks.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, Tensor, cat, arange, tensor
from torch.nn import Module, Linear, Parameter, Embedding, Sequential
from torch.distributions import Normal, kl_divergence

from .utils import exists, default

MaybeTensor = Tensor | None


class ContinuousActionEmbedder(Module):
    """
    Embeds and unembeds continuous actions for the Dreamer4 world model.

    Embedding: per-action-dim learned embedding scaled by action value.
    Unembedding: project from policy head dim to (mean, log_var) per action dim.
    """

    def __init__(
        self,
        dim: int,
        num_actions: int,
        unembed_dim: int | None = None,
        num_unembed_preds: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.num_actions = num_actions
        self.num_unembed_preds = num_unembed_preds

        self.register_buffer('dummy', tensor(0), persistent=False)

        # Embedding: each action dimension gets a learned vector
        self.action_embed = Embedding(num_actions, dim)

        # Unembedding: project to (mean, log_var) for each action dim
        unembed_dim = default(unembed_dim, dim)
        # (num_actions, num_preds, unembed_dim, 2)
        self.action_unembed = Parameter(torch.randn(num_actions, num_unembed_preds, unembed_dim, 2) * 1e-2)

        self.register_buffer('action_types', arange(num_actions), persistent=False)

    @property
    def device(self):
        return self.dummy.device

    @property
    def has_actions(self):
        return self.num_actions > 0

    def embed_parameters(self):
        return set(self.action_embed.parameters())

    def unembed_parameters(self):
        return {self.action_unembed}

    def forward(self, actions):
        """
        actions: (..., num_actions)
        Returns: (..., dim) — sum-pooled action embedding
        """
        action_embed = self.action_embed(self.action_types)  # (num_actions, dim)
        # Scale each embedding by the action value
        # actions: (..., na), embed: (na, d) -> (..., na, d)
        scaled = actions.unsqueeze(-1) * action_embed
        return scaled.sum(dim=-2)  # (..., d)

    def unembed(self, embeds, pred_head_index=None):
        """
        embeds: (..., unembed_dim)
        Returns: (mtp, ..., num_actions, 2) for (mean, log_var)
        If pred_head_index given or mtp==1, squeezes mtp dim -> (..., num_actions, 2)
        """
        unembed_w = self.action_unembed  # (na, mtp, unembed_dim, 2)

        if exists(pred_head_index):
            if isinstance(pred_head_index, int):
                pred_head_index = tensor(pred_head_index, device=self.device)
            unembed_w = unembed_w[:, pred_head_index:pred_head_index+1]

        na, mtp, d, two = unembed_w.shape
        lead_shape = embeds.shape[:-1]
        embeds_flat = embeds.reshape(-1, d)  # (N, d)
        N = embeds_flat.shape[0]

        # Reshape weight: (na, mtp, d, 2) -> (mtp * na * 2, d)
        w_flat = unembed_w.permute(1, 0, 3, 2).reshape(mtp * na * two, d)

        # (N, d) @ (d, mtp*na*2) -> (N, mtp*na*2)
        out_flat = embeds_flat @ w_flat.t()

        # Reshape to (N, mtp, na, 2)
        out = out_flat.reshape(N, mtp, na, two)

        # Reshape N back to lead_shape: (*lead_shape, mtp, na, 2)
        out = out.reshape(*lead_shape, mtp, na, two)

        # Move mtp to front: (mtp, ..., na, 2)
        # Number of lead dims
        n_lead = len(lead_shape)
        # Permute mtp (at index n_lead) to position 0
        perm = [n_lead] + list(range(n_lead)) + [n_lead + 1, n_lead + 2]
        out = out.permute(*perm)

        # squeeze single prediction head
        if exists(pred_head_index) or self.num_unembed_preds == 1:
            out = out.squeeze(0)

        return out

    def sample(self, embed, temperature=1., pred_head_index=None):
        """
        embed: (..., unembed_dim)
        Returns: (..., num_actions) sampled continuous actions
        """
        mean_log_var = self.unembed(embed, pred_head_index=pred_head_index)
        mean, log_var = mean_log_var.unbind(dim=-1)
        std = (0.5 * log_var).exp()
        return mean + std * torch.randn_like(mean) * temperature

    def log_probs(self, embeds, targets, pred_head_index=None, return_entropies=False):
        """
        embeds: (..., unembed_dim)
        targets: (..., num_actions) or (mtp, ..., num_actions)
        Returns: log_probs (..., num_actions)
        """
        mean_log_var = self.unembed(embeds, pred_head_index=pred_head_index)
        mean, log_var = mean_log_var.unbind(dim=-1)
        std = (0.5 * log_var).exp()
        dist = Normal(mean, std)

        # broadcast targets to mtp dim if needed
        if targets.ndim < mean.ndim:
            targets = targets.unsqueeze(0)

        log_p = dist.log_prob(targets)

        if not return_entropies:
            return log_p

        return log_p, dist.entropy()

    def kl_div(self, src_params, tgt_params):
        """
        src_params, tgt_params: (..., num_actions, 2) — (mean, log_var)
        Returns: (...,) KL divergence summed over action dims
        """
        def to_dist(params):
            mean, log_var = params.unbind(dim=-1)
            return Normal(mean, (0.5 * log_var).exp())

        src_dist = to_dist(src_params)
        tgt_dist = to_dist(tgt_params)
        return kl_divergence(src_dist, tgt_dist).sum(dim=-1)
