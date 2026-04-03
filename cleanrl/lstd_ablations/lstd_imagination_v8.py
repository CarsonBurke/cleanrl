"""LSTD + Imagination v8: Combines the LSTD actor architecture (RMSNorm + SiLU backbone,
shared weight matrix for mean and noise modulation, SDE noise with learned log_std_param)
with the imagination v8 world-model and dreamed-rollout framework.

The behavior actor uses LSTD-style state-dependent exploration on encoder latents.
The imagination actor remains a standard diagonal Gaussian (deepcopy of actor_mean).
"""
import copy
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from cleanrl.imagination import latent_imagination_v8_core as v8

LOG_STD_INIT = -2.0
LOG_STD_MIN = -3.0
LOG_STD_MAX = -0.5
SDE_EPS = 1e-6
SDE_PRESCALE = 1.5


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class LSTDAgent(v8.Agent):
    """v8 Agent with LSTD-style actor operating on encoder latents."""

    def __init__(
        self,
        envs,
        latent_dim: int = 64,
        model_hidden_dim: int = 128,
        model_min_std: float = 0.05,
        model_max_std: float = 1.0,
        use_done_model: bool = False,
        context_hidden_dim: int = 64,
        imagination_num_bins: int = 255,
        imagination_bin_range: float = 3.0,
    ):
        super().__init__(
            envs,
            latent_dim=latent_dim,
            model_hidden_dim=model_hidden_dim,
            model_min_std=model_min_std,
            model_max_std=model_max_std,
            use_done_model=use_done_model,
            context_hidden_dim=context_hidden_dim,
            imagination_num_bins=imagination_num_bins,
            imagination_bin_range=imagination_bin_range,
        )
        action_dim = int(np.prod(envs.single_action_space.shape))
        hidden_dim = 64

        # Replace v5's actor_mean Sequential with LSTD backbone
        # Delete the inherited actor_mean and actor_logstd
        del self.actor_mean
        del self.actor_logstd

        # LSTD actor backbone: latent -> hidden features
        self.actor_fc1 = v8.v5.layer_init(nn.Linear(latent_dim, hidden_dim))
        self.actor_norm1 = RMSNorm(hidden_dim)
        self.actor_fc2 = v8.v5.layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.actor_norm2 = RMSNorm(hidden_dim)

        # Shared output head for mean AND noise modulation
        self.actor_out = v8.v5.layer_init(nn.Linear(hidden_dim, action_dim), std=1.0)
        self.mean_scale = nn.Parameter(torch.tensor(0.01))

        # SDE noise with learned log_std_param
        self.sde_fc = v8.v5.layer_init(nn.Linear(hidden_dim, hidden_dim), std=1.0)
        self.sde_norm = RMSNorm(hidden_dim)
        self.sde_fc2 = v8.v5.layer_init(nn.Linear(hidden_dim, hidden_dim), std=1.0)
        self.log_std_param = nn.Parameter(torch.zeros(hidden_dim, action_dim))

        # Replace v5's critic with LSTD-style critic
        del self.critic
        self.critic_fc1 = v8.v5.layer_init(nn.Linear(latent_dim, hidden_dim))
        self.critic_norm1 = RMSNorm(hidden_dim)
        self.critic_fc2 = v8.v5.layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.critic_norm2 = RMSNorm(hidden_dim)
        self.value_out = v8.v5.layer_init(nn.Linear(hidden_dim, 1), std=1.0)

        # Rebuild imagination actor as a standard diagonal Gaussian (simple MLP)
        # since we don't need SDE for imagined rollouts
        del self.imagination_actor_mean
        del self.imagination_actor_logstd
        self.imagination_actor_mean = nn.Sequential(
            v8.v5.layer_init(nn.Linear(latent_dim, 64)),
            nn.Tanh(),
            v8.v5.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            v8.v5.layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.imagination_actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def _actor_features(self, latent):
        h = F.silu(self.actor_norm1(self.actor_fc1(latent)))
        h = F.silu(self.actor_norm2(self.actor_fc2(h)))
        return h

    def _get_action_std(self, h):
        sde_raw = self.sde_fc(h)
        sde_latent = (self.sde_fc2(self.sde_norm(sde_raw)) / SDE_PRESCALE).tanh()
        log_std = (self.log_std_param + LOG_STD_INIT).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std_sq = log_std.exp().pow(2)
        action_var = (sde_latent.pow(2)) @ std_sq
        action_std = (action_var + SDE_EPS).sqrt()
        return action_std

    def _critic_features(self, latent):
        h = F.silu(self.critic_norm1(self.critic_fc1(latent)))
        h = F.silu(self.critic_norm2(self.critic_fc2(h)))
        return h

    def get_dist_from_latent(self, latent):
        h = self._actor_features(latent)
        action_mean = self.actor_out(h) * self.mean_scale
        action_std = self._get_action_std(h)
        return Normal(action_mean, action_std)

    def get_value_from_latent(self, latent):
        h = self._critic_features(latent)
        return self.value_out(h)

    def get_value(self, obs):
        return self.get_value_from_latent(self.encode(obs))

    def get_action_and_value(self, obs, action=None):
        latent = self.encode(obs)
        probs = self.get_dist_from_latent(latent)
        if action is None:
            action = probs.sample()
        value = self.get_value_from_latent(latent)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

    def behavior_parameters(self):
        params = []
        params.extend(self.encoder.parameters())
        # LSTD actor params
        params.extend(self.actor_fc1.parameters())
        params.extend(self.actor_norm1.parameters())
        params.extend(self.actor_fc2.parameters())
        params.extend(self.actor_norm2.parameters())
        params.extend(self.actor_out.parameters())
        params.append(self.mean_scale)
        params.extend(self.sde_fc.parameters())
        params.extend(self.sde_norm.parameters())
        params.extend(self.sde_fc2.parameters())
        params.append(self.log_std_param)
        # LSTD critic params
        params.extend(self.critic_fc1.parameters())
        params.extend(self.critic_norm1.parameters())
        params.extend(self.critic_fc2.parameters())
        params.extend(self.critic_norm2.parameters())
        params.extend(self.value_out.parameters())
        return params


@dataclass
class Args(v8.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    clip_coef_low: float = 0.2
    """the lower surrogate clipping coefficient (ratio floor = 1 - this)"""
    clip_coef_high: float = 0.28
    """the upper surrogate clipping coefficient (ratio ceiling = 1 + this)"""


if __name__ == "__main__":
    v8.main(Args, agent_class=LSTDAgent)
