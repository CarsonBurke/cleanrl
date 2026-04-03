"""
Dreamer4 World Model — Dynamics World Model for MuJoCo
=======================================================
Core dynamics model with:
- Shortcut flow matching for latent prediction
- Token packing: [flow_token | space_tokens | proprio_token | state_pred | registers | action | reward | agent]
- AxialSpaceTimeTransformer backbone
- Multi-token reward prediction (ensemble)
- Value head (MLP + SymExp TwoHot)
- Policy head (MLP + continuous action unembedding)
- PPO / PMPO policy optimization from dreams
- Generation (denoising loop)
- State prediction head with entropy bonus

Adapted for MuJoCo: no video/images, state-based observations, continuous actions only.
"""
from __future__ import annotations
from typing import Callable
from math import log2, ceil
from functools import partial, wraps
from contextlib import nullcontext
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor, cat, stack, arange, tensor, randn, randn_like, full, empty, rand, randint
from torch.nn import Module, ModuleList, Linear, Parameter, Sequential, Embedding, RMSNorm
from torch.optim import Optimizer

from .utils import (
    exists, default, divisible_by, is_power_two, first,
    l2norm, softclamp, safe_cat, safe_stack,
    lens_to_mask, masked_mean, pad_at_dim, pad_right_at_dim_to, align_dims_left,
    create_multi_token_prediction_targets,
    LossNormalizer, SymExpTwoHot, BetaDist, Ensemble, build_mlp,
    calc_gae, ramp_weight, frac_gradient,
    Experience, Actions, Predictions, Embeds, WorldModelLosses,
    with_seed, sample_prob,
    StateTokenizer,
)
from .transformer import (
    AxialSpaceTimeTransformer, TransformerIntermediates,
)
from .actions import ContinuousActionEmbedder

LinearNoBias = partial(Linear, bias=False)

MaybeTensor = Tensor | None


class DynamicsWorldModel(Module):
    """
    Dreamer4-style dynamics world model for continuous control (MuJoCo).

    The model operates on latent state tokens produced by a StateTokenizer.
    It uses shortcut flow matching to predict next latent states, and includes
    policy/value/reward heads for reinforcement learning.
    """

    def __init__(
        self,
        dim: int = 256,
        dim_latent: int = 32,
        state_tokenizer: StateTokenizer | None = None,
        dim_obs: int | None = None,
        max_steps: int = 64,
        num_register_tokens: int = 8,
        num_spatial_tokens: int = 2,
        num_latent_tokens: int | None = None,
        dim_proprio: int | None = None,
        reward_encoder_kwargs: dict = dict(),
        depth: int = 4,
        pred_orig_latent: bool = True,
        time_block_every: int = 4,
        attn_kwargs: dict = dict(),
        attn_heads: int = 8,
        attn_dim_head: int = 64,
        attn_softclamp_value: float = 50.,
        ff_kwargs: dict = dict(),
        use_time_rnn: bool = True,
        loss_weight_fn: Callable = ramp_weight,
        prob_shortcut_train: float | None = None,
        add_state_pred_head: bool = False,
        state_pred_loss_weight: float = 0.1,
        state_entropy_bonus_weight: float = 0.05,
        agent_predicts_state: bool = False,
        agent_state_pred_loss_weight: float = 0.1,
        eps_latent_pred: float = 1e-6,
        num_continuous_actions: int = 0,
        multi_token_pred_len: int = 8,
        value_head_mlp_depth: int = 3,
        policy_head_mlp_depth: int = 3,
        add_reward_embed_to_agent_token: bool = False,
        add_reward_embed_dropout: float = 0.1,
        latent_flow_loss_weight: float = 1.,
        shortcut_loss_weight: float = 1.,
        reward_loss_weight: float = 1.,
        continuous_action_loss_weight: float = 1.,
        keep_reward_ema_stats: bool = False,
        reward_ema_decay: float = 0.998,
        reward_quantile_filter: tuple = (0.05, 0.95),
        gae_discount_factor: float = 0.997,
        gae_lambda: float = 0.95,
        ppo_eps_clip: float = 0.2,
        pmpo_pos_to_neg_weight: float = 0.5,
        pmpo_reverse_kl: bool = True,
        pmpo_kl_div_loss_weight: float = 0.3,
        use_delight_gating: bool = True,
        value_clip: float = 0.4,
        policy_entropy_weight: float = 0.01,
        use_loss_normalization: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # State tokenizer
        self.state_tokenizer = state_tokenizer
        if exists(state_tokenizer):
            num_latent_tokens = default(num_latent_tokens, state_tokenizer.num_latent_tokens)
            assert state_tokenizer.num_latent_tokens == num_latent_tokens

        assert exists(num_latent_tokens), '`num_latent_tokens` must be set'

        # Spatial tokens
        self.num_latent_tokens = num_latent_tokens
        self.dim_latent = dim_latent
        self.latent_shape = (num_latent_tokens, dim_latent)

        if num_spatial_tokens >= num_latent_tokens:
            assert divisible_by(num_spatial_tokens, num_latent_tokens)
            expand_factor = num_spatial_tokens // num_latent_tokens

            self.latents_to_spatial_tokens = Sequential(
                Linear(dim_latent, dim * expand_factor),
            )
            self.spatial_expand_factor = expand_factor

            self.to_latent_pred = Sequential(
                RMSNorm(dim),
                LinearNoBias(dim, dim_latent),
            )
            self.latent_pred_needs_pool = True
        else:
            assert divisible_by(num_latent_tokens, num_spatial_tokens)
            tokens_per_space = num_latent_tokens // num_spatial_tokens

            self.latents_to_spatial_tokens = Sequential(
                Linear(num_latent_tokens * dim_latent, dim * num_spatial_tokens),
            )
            self.spatial_expand_factor = num_spatial_tokens

            self.to_latent_pred = Sequential(
                RMSNorm(dim),
                LinearNoBias(dim, dim_latent * tokens_per_space),
            )
            self.latent_pred_needs_pool = False
            self._latent_tokens_per_space = tokens_per_space

        self.num_spatial_tokens = max(num_spatial_tokens, num_latent_tokens) if num_spatial_tokens >= num_latent_tokens else num_spatial_tokens

        # Proprioception
        self.has_proprio = exists(dim_proprio) and dim_proprio > 0
        self.dim_proprio = dim_proprio

        if self.has_proprio:
            self.to_proprio_token = Linear(dim_proprio, dim)
            self.to_proprio_pred = Sequential(
                RMSNorm(dim),
                Linear(dim, dim_proprio),
            )

        # Register tokens
        self.num_register_tokens = num_register_tokens
        self.register_tokens = Parameter(torch.randn(num_register_tokens, dim) * 1e-2)

        # Signal and step size embeddings
        assert divisible_by(dim, 2)
        dim_half = dim // 2

        assert is_power_two(max_steps)
        self.max_steps = max_steps
        self.num_step_sizes_log2 = int(log2(max_steps))

        self.signal_levels_embed = Embedding(max_steps, dim_half)
        self.step_size_embed = Embedding(self.num_step_sizes_log2, dim_half)

        self.prob_shortcut_train = default(prob_shortcut_train, 1. - self.num_step_sizes_log2 ** -1.)

        # Loss
        self.pred_orig_latent = pred_orig_latent
        self.loss_weight_fn = loss_weight_fn

        # State prediction
        self.add_state_pred_head = add_state_pred_head
        self.state_pred_loss_weight = state_pred_loss_weight
        self.should_pred_state = add_state_pred_head and state_pred_loss_weight > 0.
        self.eps_latent_pred = eps_latent_pred

        if self.should_pred_state:
            self.state_pred_token = Parameter(torch.randn(dim) * 1e-2)
            self.to_state_pred = Sequential(
                RMSNorm(dim),
                Linear(dim, num_latent_tokens * dim_latent * 2),
            )
            self.state_beta_dist = BetaDist(unimodal=True)

        self.state_entropy_bonus_weight = state_entropy_bonus_weight
        self.add_state_entropy_bonus = self.should_pred_state and state_entropy_bonus_weight > 0.

        # Agent tokens
        self.agent_learned_embed = Parameter(randn(1, dim) * 1e-2)
        self.action_learned_embed = Parameter(randn(1, dim) * 1e-2)
        self.reward_learned_embed = Parameter(randn(1, dim) * 1e-2)

        # Policy head
        self.policy_head = build_mlp(
            dim_in=dim,
            dim_hidden=dim * 4,
            dim_out=dim * 4,
            depth=policy_head_mlp_depth,
        )

        # Action embedder
        self.action_embedder = ContinuousActionEmbedder(
            dim=dim,
            num_actions=num_continuous_actions,
            unembed_dim=dim * 4,
            num_unembed_preds=multi_token_pred_len,
        )

        # Agent state prediction
        self.agent_predicts_state = agent_predicts_state
        self.agent_state_pred_loss_weight = agent_state_pred_loss_weight

        if self.agent_predicts_state:
            dim_in = dim * 2 if self.action_embedder.has_actions else dim
            self.to_agent_state_pred = Sequential(
                Linear(dim_in, dim_in),
                nn.SiLU(),
                Linear(dim_in, dim),
                RMSNorm(dim),
                Linear(dim, num_latent_tokens * dim_latent * 2),
            )
            self.agent_state_beta_dist = BetaDist(unimodal=True)

        # Multi-token prediction
        self.multi_token_pred_len = multi_token_pred_len

        # Reward encoder
        self.add_reward_embed_to_agent_token = add_reward_embed_to_agent_token
        self.add_reward_embed_dropout = add_reward_embed_dropout

        self.reward_encoder = SymExpTwoHot(
            **reward_encoder_kwargs,
            dim_embed=dim,
            learned_embedding=add_reward_embed_to_agent_token,
        )

        def make_reward_pred():
            return Sequential(
                RMSNorm(dim),
                LinearNoBias(dim, self.reward_encoder.num_bins),
            )

        self.to_reward_pred = Ensemble(make_reward_pred, multi_token_pred_len)

        # Value head
        self.value_head = build_mlp(
            dim_in=dim,
            dim_hidden=dim * 4,
            dim_out=self.reward_encoder.num_bins,
            depth=value_head_mlp_depth,
        )

        # Transformer backbone
        self.transformer = AxialSpaceTimeTransformer(
            dim=dim,
            depth=depth,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            attn_softclamp_value=attn_softclamp_value,
            attn_kwargs=attn_kwargs,
            ff_kwargs=ff_kwargs,
            num_special_spatial_tokens=1,  # one agent token
            time_block_every=time_block_every,
            final_norm=False,
            rnn_time=use_time_rnn,
        )

        # PPO
        self.gae_discount_factor = gae_discount_factor
        self.gae_lambda = gae_lambda
        self.ppo_eps_clip = ppo_eps_clip
        self.value_clip = value_clip
        self.policy_entropy_weight = policy_entropy_weight

        # PMPO
        self.pmpo_pos_to_neg_weight = pmpo_pos_to_neg_weight
        self.pmpo_kl_div_loss_weight = pmpo_kl_div_loss_weight
        self.pmpo_reverse_kl = pmpo_reverse_kl

        # Delight gating
        self.use_delight_gating = use_delight_gating

        # Reward stats
        self.keep_reward_ema_stats = keep_reward_ema_stats
        self.reward_ema_decay = reward_ema_decay
        self.register_buffer('reward_quantile_filter', tensor(reward_quantile_filter), persistent=False)
        self.register_buffer('ema_returns_mean', tensor(0.))
        self.register_buffer('ema_returns_var', tensor(1.))

        # Loss normalization
        self.use_loss_normalization = use_loss_normalization
        self.flow_loss_normalizer = LossNormalizer() if use_loss_normalization else None
        self.shortcut_flow_loss_normalizer = LossNormalizer() if use_loss_normalization else None
        self.reward_loss_normalizer = LossNormalizer(multi_token_pred_len) if use_loss_normalization else None
        self.continuous_actions_loss_normalizer = LossNormalizer(multi_token_pred_len) if (num_continuous_actions > 0 and use_loss_normalization) else None

        # Loss weights
        self.latent_flow_loss_weight = latent_flow_loss_weight
        self.shortcut_loss_weight = shortcut_loss_weight
        self.register_buffer('reward_loss_weight', tensor(reward_loss_weight))
        self.register_buffer('continuous_action_loss_weight', tensor(continuous_action_loss_weight))

        self.register_buffer('zero', tensor(0.), persistent=False)

    @property
    def device(self):
        return self.zero.device

    def policy_head_parameters(self):
        return [
            *self.policy_head.parameters(),
            *self.action_embedder.unembed_parameters(),
        ]

    def value_head_parameters(self):
        return list(self.value_head.parameters())

    def get_times_from_signal_level(self, signal_levels, ref=None):
        times = signal_levels.float() / self.max_steps
        if exists(ref):
            times, _ = align_dims_left((times, ref))
        return times

    # ──────────────────────────────────────────
    # Token packing helper
    # ──────────────────────────────────────────

    def _pack_tokens(self, flow_token, space_tokens, proprio_token, state_pred_token,
                     registers, action_tokens, reward_tokens, agent_tokens):
        """
        Pack all token types along spatial dimension.
        All inputs: (b, t, ?, d) where ? varies by token type.
        Returns: (b, t, total_spatial, d) and list of sizes for unpacking.
        """
        parts = [flow_token, space_tokens, proprio_token, state_pred_token,
                 registers, action_tokens, reward_tokens, agent_tokens]
        sizes = [p.shape[2] for p in parts]
        tokens = cat(parts, dim=2)
        return tokens, sizes

    def _unpack_tokens(self, tokens, sizes):
        """Unpack tokens along spatial dimension."""
        parts = []
        idx = 0
        for s in sizes:
            parts.append(tokens[:, :, idx:idx+s])
            idx += s
        return parts

    # ──────────────────────────────────────────
    # Spatial token conversion
    # ──────────────────────────────────────────

    def _latents_to_space(self, latents):
        """(b, t, n, d_latent) -> (b, t, num_spatial, dim)"""
        b, t, n, d = latents.shape
        flat = latents.reshape(b, t, n * d) if not self.latent_pred_needs_pool else latents
        if self.latent_pred_needs_pool:
            # (b, t, n, dim * expand)
            projected = self.latents_to_spatial_tokens(flat)
            # reshape to (b, t, n * expand, dim)
            return projected.reshape(b, t, n * self.spatial_expand_factor, self.dim)
        else:
            projected = self.latents_to_spatial_tokens(flat)
            return projected.reshape(b, t, self.num_spatial_tokens, self.dim)

    def _space_to_latent_pred(self, space_tokens):
        """(b, t, num_spatial, dim) -> (b, t, n, d_latent)"""
        if self.latent_pred_needs_pool:
            # mean pool over spatial expand factor
            b, t, s, d = space_tokens.shape
            n = self.num_latent_tokens
            ef = s // n
            pooled = space_tokens.reshape(b, t, n, ef, d).mean(dim=3)
            pred = self.to_latent_pred(pooled)
            return pred
        else:
            b, t, s, d = space_tokens.shape
            pred = self.to_latent_pred(space_tokens)
            return pred.reshape(b, t, self.num_latent_tokens, self.dim_latent)

    # ──────────────────────────────────────────
    # Core forward
    # ──────────────────────────────────────────

    def _get_prediction(
        self,
        noised_latents,
        noised_proprio,
        signal_levels,
        step_sizes_log2,
        state_pred_token,
        action_tokens,
        reward_tokens,
        agent_tokens,
        time_cache=None,
        return_agent_tokens=False,
        return_time_cache=False,
        return_intermediates=False,
    ):
        batch, time = noised_latents.shape[:2]

        # Latents to spatial tokens
        space_tokens = self._latents_to_space(noised_latents)

        # Register tokens
        registers = self.register_tokens.unsqueeze(0).unsqueeze(0).expand(batch, time, -1, -1)

        # Proprio token
        if exists(noised_proprio):
            proprio_token = self.to_proprio_token(noised_proprio).unsqueeze(2)
        else:
            proprio_token = agent_tokens[:, :, 0:0]  # empty

        # Signal + step size embedding -> flow token
        signal_embed = self.signal_levels_embed(signal_levels)
        step_size_embed = self.step_size_embed(step_sizes_log2)
        step_size_embed = step_size_embed.unsqueeze(1).expand(-1, time, -1)

        flow_token = cat((signal_embed, step_size_embed), dim=-1)
        flow_token = flow_token.unsqueeze(2)  # (b, t, 1, d)

        # Pack tokens
        tokens, packed_sizes = self._pack_tokens(
            flow_token, space_tokens, proprio_token, state_pred_token,
            registers, action_tokens, reward_tokens, agent_tokens
        )

        # Transformer
        tokens, intermediates = self.transformer(tokens, cache=time_cache, return_intermediates=True)

        # Unpack
        parts = self._unpack_tokens(tokens, packed_sizes)
        flow_out, space_out, proprio_out, state_pred_out, reg_out, action_out, reward_out, agent_out = parts

        # Latent prediction from space tokens
        pred_latent = self._space_to_latent_pred(space_out)

        # Proprio prediction
        pred_proprio = None
        if self.has_proprio:
            pred_proprio = self.to_proprio_pred(proprio_out.squeeze(2))

        # State prediction
        pred_state = None
        if self.should_pred_state:
            pred_state = self.to_state_pred(state_pred_out.squeeze(2))
            pred_state = pred_state.reshape(batch, time, self.num_latent_tokens, self.dim_latent, 2)

        predictions = Predictions(pred_latent, pred_proprio, pred_state)
        embeds = Embeds(agent_out, state_pred_out)

        if not return_agent_tokens:
            return predictions

        if not return_time_cache:
            return predictions, embeds

        return predictions, (embeds, intermediates, packed_sizes)

    # ──────────────────────────────────────────
    # Generation (denoising loop)
    # ──────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        time_steps: int,
        num_steps: int = 4,
        batch_size: int = 1,
        initial_latents: Tensor | None = None,
        initial_proprio: Tensor | None = None,
        initial_actions: Tensor | None = None,
        initial_rewards: Tensor | None = None,
        context_signal_noise: float = 0.1,
        use_time_cache: bool = True,
        return_for_policy_optimization: bool = False,
        store_agent_embed: bool = True,
        store_old_action_params: bool = True,
        continuous_temperature: float = 1.,
    ):
        """Generate dream trajectories for policy optimization."""
        was_training = self.training
        self.eval()

        assert is_power_two(num_steps)
        assert 0 < num_steps <= self.max_steps

        latent_shape = self.latent_shape
        step_size = self.max_steps // num_steps

        # Initialize
        if exists(initial_latents):
            latents = initial_latents.clone()
        else:
            latents = empty((batch_size, 0, *latent_shape), device=self.device)

        past_context_noise = latents.clone()

        # Proprio
        has_proprio = self.has_proprio
        if has_proprio:
            proprio = initial_proprio.clone() if exists(initial_proprio) else empty((batch_size, 0, self.dim_proprio), device=self.device)
            past_proprio_noise = proprio.clone()

        # Actions/rewards
        decoded_actions = initial_actions.clone() if exists(initial_actions) else empty((batch_size, 0, self.action_embedder.num_actions), device=self.device)
        decoded_rewards = initial_rewards.clone() if exists(initial_rewards) else None

        decoded_log_probs = None
        decoded_values = None
        acc_agent_embed = None
        acc_policy_embed = None

        time_cache = None

        while latents.shape[1] < time_steps:
            curr_time = latents.shape[1]

            take_extra_step = (
                use_time_cache or store_agent_embed or
                return_for_policy_optimization
            )

            # New noised latent
            noised_latent = randn((batch_size, 1, *latent_shape), device=self.device)
            noised_proprio = randn((batch_size, 1, self.dim_proprio), device=self.device) if has_proprio else None

            num_iterations = num_steps + int(take_extra_step)

            for step in range(num_iterations):
                is_last = (step + 1) == num_iterations

                signal_val = min(step * step_size, self.max_steps - 1)
                signal_levels = full((batch_size, 1), signal_val, dtype=torch.long, device=self.device)

                # Context noising
                noised_context = latents.lerp(past_context_noise, context_signal_noise)
                noised_with_context = cat((noised_context, noised_latent), dim=1)

                noised_proprio_with_context = None
                if has_proprio:
                    noised_pc = proprio.lerp(past_proprio_noise, context_signal_noise)
                    noised_proprio_with_context = cat((noised_pc, noised_proprio), dim=1)

                signal_with_context = F.pad(signal_levels, (curr_time, 0), value=self.max_steps - 1)

                # Actions for context
                curr_actions = None
                if decoded_actions.shape[1] > 0:
                    curr_actions = decoded_actions[:, :curr_time]
                    curr_actions = pad_right_at_dim_to(curr_actions, curr_time, dim=1)

                # Prepare state pred / action / reward / agent tokens
                batch = batch_size
                time_len = noised_with_context.shape[1]

                agent_tokens = self.agent_learned_embed.unsqueeze(0).unsqueeze(0).expand(batch, time_len, -1, -1)

                # Action tokens
                if exists(curr_actions) and curr_actions.shape[1] > 0:
                    act_embed = self.action_embedder(curr_actions)
                    act_embed = act_embed + self.action_learned_embed.squeeze(0)
                    action_tokens_input = pad_at_dim(act_embed, (1, 0), dim=1, value=0.)[:, :time_len]
                    action_tokens_input = action_tokens_input.unsqueeze(2)
                else:
                    action_tokens_input = torch.zeros(batch, time_len, 1, self.dim, device=self.device)

                # Reward tokens
                reward_tokens = agent_tokens[:, :, 0:0]
                if exists(decoded_rewards) and self.add_reward_embed_to_agent_token:
                    reward_tokens = self.reward_encoder.embed(self.reward_encoder(decoded_rewards))
                    pop_last_reward = int(reward_tokens.shape[1] == agent_tokens.shape[1])
                    reward_tokens = pad_at_dim(reward_tokens, (1, -pop_last_reward), dim=1, value=0.)
                    reward_tokens = pad_right_at_dim_to(reward_tokens, time_len, dim=1)
                    reward_tokens = reward_tokens + self.reward_learned_embed
                    reward_tokens = reward_tokens.unsqueeze(2)

                # State pred token
                if self.should_pred_state:
                    state_pred_token = self.state_pred_token.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch, time_len, 1, -1)
                else:
                    state_pred_token = agent_tokens[:, :, 0:0]

                step_sizes_log2 = torch.zeros(batch, dtype=torch.long, device=self.device)
                for ss in range(self.num_step_sizes_log2):
                    if 2 ** ss == step_size:
                        step_sizes_log2[:] = ss
                        break

                pred, (embeds, intermediates, packed_sizes) = self._get_prediction(
                    noised_with_context,
                    noised_proprio_with_context,
                    signal_with_context,
                    step_sizes_log2,
                    state_pred_token,
                    action_tokens_input,
                    reward_tokens,
                    agent_tokens,
                    time_cache=time_cache,
                    return_agent_tokens=True,
                    return_time_cache=True,
                    return_intermediates=True,
                )

                if use_time_cache and is_last:
                    time_cache = intermediates

                if take_extra_step and is_last:
                    break

                # Denoise step
                pred_latent = pred.flow[:, curr_time:]

                if self.pred_orig_latent:
                    times = self.get_times_from_signal_level(signal_levels, pred_latent)
                    flow = (pred_latent - noised_latent) / (1. - times).clamp(min=1e-6)
                else:
                    flow = pred_latent

                noised_latent = noised_latent + flow * (step_size / self.max_steps)

                if has_proprio:
                    pred_proprio = pred.proprioception[:, curr_time:]
                    if self.pred_orig_latent:
                        flow_p = (pred_proprio - noised_proprio) / (1. - times.squeeze(-1).squeeze(-1)).clamp(min=1e-6).unsqueeze(-1)
                    else:
                        flow_p = pred_proprio
                    noised_proprio = noised_proprio + flow_p * (step_size / self.max_steps)

            denoised_latent = noised_latent

            # Decode reward from agent token
            agent_embed = embeds.agent
            one_agent_embed = agent_embed[:, -1:, 0]

            reward_logits = self.to_reward_pred.forward_one(one_agent_embed, id=0)
            pred_reward = self.reward_encoder.bins_to_scalar_value(reward_logits, normalize=True)
            decoded_rewards = safe_cat((decoded_rewards, pred_reward), dim=1)

            # Store agent embed
            if store_agent_embed:
                acc_agent_embed = safe_cat((acc_agent_embed, one_agent_embed), dim=1)

            # Sample action
            if self.action_embedder.has_actions:
                policy_embed = self.policy_head(one_agent_embed)

                if store_old_action_params:
                    acc_policy_embed = safe_cat((acc_policy_embed, policy_embed), dim=1)

                sampled_action = self.action_embedder.sample(
                    policy_embed, pred_head_index=0, temperature=continuous_temperature
                )
                decoded_actions = cat((decoded_actions, sampled_action), dim=1)

                if return_for_policy_optimization:
                    log_p = self.action_embedder.log_probs(
                        policy_embed, pred_head_index=0, targets=sampled_action
                    )
                    decoded_log_probs = safe_cat((decoded_log_probs, log_p), dim=1)

                    value_bins = self.value_head(one_agent_embed)
                    value = self.reward_encoder.bins_to_scalar_value(value_bins)
                    decoded_values = safe_cat((decoded_values, value), dim=1)

            # Append denoised
            latents = cat((latents, denoised_latent), dim=1)
            past_context_noise = cat((past_context_noise, randn_like(denoised_latent)), dim=1)

            if has_proprio:
                denoised_proprio = noised_proprio
                proprio = cat((proprio, denoised_proprio), dim=1)
                past_proprio_noise = cat((past_proprio_noise, randn_like(denoised_proprio)), dim=1)

        self.train(was_training)

        # Build experience
        experience_lens = full((batch_size,), time_steps, device=self.device)

        gen = Experience(
            latents=latents,
            proprio=proprio if has_proprio else None,
            agent_embed=acc_agent_embed if store_agent_embed else None,
            old_action_params=self.action_embedder.unembed(acc_policy_embed, pred_head_index=0) if exists(acc_policy_embed) else None,
            step_size=step_size,
            lens=experience_lens,
            is_from_world_model=True,
        )

        gen.rewards = decoded_rewards

        if self.action_embedder.has_actions:
            gen.actions = Actions(decoded_actions)

        if return_for_policy_optimization:
            gen.log_probs = decoded_log_probs
            gen.values = decoded_values

        return gen

    # ──────────────────────────────────────────
    # Learn from experience (PPO/PMPO)
    # ──────────────────────────────────────────

    def learn_from_experience(
        self,
        experience: Experience,
        only_learn_policy_value_heads: bool = True,
        use_pmpo: bool = True,
        use_delight_gating: bool | None = None,
        normalize_advantages: bool | None = None,
        eps: float = 1e-6,
    ):
        use_delight_gating = default(use_delight_gating, self.use_delight_gating)
        experience = experience.to(self.device)

        latents = experience.latents
        actions = experience.actions
        proprio = experience.proprio
        old_log_probs = experience.log_probs
        old_values = experience.values
        rewards = experience.rewards
        agent_embeds = experience.agent_embed
        old_action_params = experience.old_action_params

        step_size = experience.step_size

        assert all([exists(x) for x in (old_log_probs, actions, old_values, rewards, step_size)])

        batch = latents.shape[0]
        time = latents.shape[1]
        reward_time = rewards.shape[1]

        # Masks
        if not exists(experience.is_truncated):
            experience.is_truncated = full((batch,), True, device=self.device)

        if exists(experience.lens):
            effective_lens = torch.clamp(experience.lens, max=reward_time)
            mask_for_gae = lens_to_mask(effective_lens, reward_time)
            rewards = rewards.masked_fill(~mask_for_gae, 0.)
            old_values = old_values.masked_fill(~mask_for_gae, 0.)

        # GAE
        returns = calc_gae(rewards, old_values, gamma=self.gae_discount_factor, lam=self.gae_lambda)

        # Variable length mask (for policy/value learning, use reward_time)
        is_var_len = exists(experience.lens)
        mask = None
        if is_var_len:
            learnable_lens = torch.clamp(experience.lens - experience.is_truncated.long(), max=reward_time)
            mask = lens_to_mask(learnable_lens, reward_time)

        world_model_context = torch.no_grad if only_learn_policy_value_heads else nullcontext

        # Reward normalization
        if self.keep_reward_ema_stats:
            decay = 1. - self.reward_ema_decay
            lo, hi = torch.quantile(returns, self.reward_quantile_filter).tolist()
            returns_filtered = returns.clamp(lo, hi)
            self.ema_returns_mean.lerp_(returns_filtered.mean(), decay)
            self.ema_returns_var.lerp_(returns_filtered.var(), decay)
            std = self.ema_returns_var.clamp(min=1e-5).sqrt()
            advantage = (returns - self.ema_returns_mean) / std - (old_values - self.ema_returns_mean) / std
        else:
            advantage = returns - old_values

        normalize_advantages = default(normalize_advantages, not use_pmpo)
        if normalize_advantages:
            advantage = F.layer_norm(advantage, advantage.shape, eps=eps)

        if use_pmpo:
            pos_mask = advantage >= 0.
            neg_mask = ~pos_mask

        # Truncate to reward_time (latents may have +1 for bootstrap)
        latents = latents[:, :reward_time]
        if exists(agent_embeds) and agent_embeds.shape[1] > reward_time:
            agent_embeds = agent_embeds[:, :reward_time]

        # Get continuous actions
        continuous_actions = actions.continuous
        if continuous_actions.shape[1] > reward_time:
            continuous_actions = continuous_actions[:, :reward_time]
        if old_log_probs.shape[1] > reward_time:
            old_log_probs = old_log_probs[:, :reward_time]

        # Replay if needed
        if not only_learn_policy_value_heads or not exists(agent_embeds):
            with world_model_context():
                _, (embeds, _) = self.forward(
                    latents=latents,
                    signal_levels=self.max_steps - 1,
                    step_sizes=step_size,
                    rewards=rewards,
                    continuous_actions=continuous_actions,
                    proprio=proprio,
                    latent_is_noised=True,
                    return_pred_only=True,
                    return_intermediates=True,
                )

            agent_embeds = embeds.agent[:, :, 0]

        if only_learn_policy_value_heads:
            agent_embeds = agent_embeds.detach()

        # Policy
        policy_embed = self.policy_head(agent_embeds)
        log_probs, entropies = self.action_embedder.log_probs(
            policy_embed, pred_head_index=0,
            targets=continuous_actions,
            return_entropies=True,
        )

        advantage_expanded = advantage.unsqueeze(-1)

        # Delight gating
        if use_delight_gating:
            delight_gate = (-log_probs * advantage_expanded).sigmoid().detach()

        if use_pmpo:
            maybe_gated = log_probs * delight_gate if use_delight_gating else log_probs
            scaled = maybe_gated * advantage_expanded.tanh().abs()

            na = scaled.shape[-1]
            pos_m = pos_mask.unsqueeze(-1).expand_as(scaled)
            neg_m = neg_mask.unsqueeze(-1).expand_as(scaled)

            if exists(mask):
                mask_e = mask.unsqueeze(-1).expand_as(scaled)
                pos_m = pos_m & mask_e
                neg_m = neg_m & mask_e

            pos_loss = scaled[pos_m].sum() if pos_m.any() else 0.
            neg_loss = scaled[neg_m].sum() if neg_m.any() else 0.

            num_adv = max(1, len(advantage))
            alpha = self.pmpo_pos_to_neg_weight
            policy_loss = -alpha * (pos_loss - neg_loss) / num_adv

            # KL regularization
            if self.pmpo_kl_div_loss_weight > 0. and exists(old_action_params):
                new_params = self.action_embedder.unembed(policy_embed, pred_head_index=0)
                src, tgt = new_params, old_action_params
                if self.pmpo_reverse_kl:
                    src, tgt = tgt, src
                kl_loss = self.action_embedder.kl_div(src, tgt)
                kl_loss = masked_mean(kl_loss, mask)
                policy_loss = policy_loss + kl_loss * self.pmpo_kl_div_loss_weight
        else:
            maybe_weighted_adv = advantage_expanded * delight_gate if use_delight_gating else advantage_expanded
            ratio = (log_probs - old_log_probs).exp()
            clipped = ratio.clamp(1. - self.ppo_eps_clip, 1. + self.ppo_eps_clip)
            policy_loss = -torch.min(ratio * maybe_weighted_adv, clipped * maybe_weighted_adv)
            policy_loss = policy_loss.sum(dim=-1)
            policy_loss = masked_mean(policy_loss, mask)

        # Entropy
        entropy_loss = -entropies.sum(dim=-1)
        entropy_loss = masked_mean(entropy_loss, mask)

        total_policy_loss = policy_loss + entropy_loss * self.policy_entropy_weight

        # Value loss (no clipping)
        value_bins = self.value_head(agent_embeds)

        return_bins = self.reward_encoder(returns)

        vb_t = value_bins.permute(0, 2, 1)
        rb_t = return_bins.permute(0, 2, 1)

        value_loss = -(rb_t * vb_t.log_softmax(dim=1)).sum(dim=1)

        if is_var_len:
            value_loss = value_loss[mask].mean()
        else:
            value_loss = value_loss.mean()

        return total_policy_loss, value_loss

    # ──────────────────────────────────────────
    # Training forward
    # ──────────────────────────────────────────

    def forward(
        self,
        *,
        obs: Tensor | None = None,
        latents: Tensor | None = None,
        lens: Tensor | None = None,
        signal_levels: Tensor | None = None,
        step_sizes: int | Tensor | None = None,
        step_sizes_log2: Tensor | None = None,
        rewards: Tensor | None = None,
        continuous_actions: Tensor | None = None,
        shift_action_tokens: bool = True,
        proprio: Tensor | None = None,
        time_cache=None,
        return_pred_only: bool = False,
        latent_is_noised: bool = False,
        return_all_losses: bool = False,
        return_intermediates: bool = False,
        add_autoregressive_action_loss: bool = True,
        update_loss_ema=None,
        seed=None,
    ):
        # Handle obs -> latents
        assert exists(obs) or exists(latents), 'must pass obs or latents'
        if exists(obs) and not exists(latents):
            assert exists(self.state_tokenizer)
            latents = self.state_tokenizer.tokenize(obs)

        assert latents.shape[-2:] == tuple(self.latent_shape), \
            f'latents shape mismatch: {latents.shape[-2:]} vs {self.latent_shape}'

        batch, time, device = *latents.shape[:2], latents.device

        if exists(rewards):
            rewards_len = rewards.shape[1]
            assert rewards_len in {time, time - 1}, \
                f'rewards length must be {time} or {time - 1}, got {rewards_len}'
            if not return_pred_only:
                assert rewards_len == time, \
                    f'training rewards length must be {time}, got {rewards_len}'

        # Conform signal/step sizes
        if exists(signal_levels):
            if isinstance(signal_levels, int):
                signal_levels = tensor(signal_levels, device=device)
            if signal_levels.ndim == 0:
                signal_levels = signal_levels.unsqueeze(0).expand(batch)
            if signal_levels.ndim == 1:
                signal_levels = signal_levels.unsqueeze(1).expand(-1, time)

        if exists(step_sizes):
            if isinstance(step_sizes, int):
                step_sizes = tensor(step_sizes, device=device)
            if step_sizes.ndim == 0:
                step_sizes = step_sizes.unsqueeze(0).expand(batch)

        if exists(step_sizes_log2):
            if isinstance(step_sizes_log2, int):
                step_sizes_log2 = tensor(step_sizes_log2, device=device)
            if step_sizes_log2.ndim == 0:
                step_sizes_log2 = step_sizes_log2.unsqueeze(0).expand(batch)

        assert not (exists(step_sizes) and exists(step_sizes_log2))
        if exists(step_sizes):
            step_sizes_log2 = torch.log2(step_sizes.float()).long()

        assert not (exists(signal_levels) ^ exists(step_sizes_log2))
        is_inference = exists(signal_levels)
        return_pred_only = return_pred_only or latent_is_noised

        # Seeded RNG
        _sample_prob = with_seed(seed)(sample_prob)
        _randint = with_seed(seed)(randint)
        _randn_like = with_seed(seed)(randn_like)

        # Training: generate random signal levels and step sizes
        if not is_inference:
            shortcut_train = _sample_prob(self.prob_shortcut_train)

            if shortcut_train:
                step_sizes_log2 = _randint(1, self.num_step_sizes_log2, (batch,), device=device)
                num_step_sizes = 2 ** step_sizes_log2
                signal_levels = _randint(0, self.max_steps, (batch, time), device=device) // num_step_sizes[:, None] * num_step_sizes[:, None]
            else:
                step_sizes_log2 = torch.zeros((batch,), device=device).long()
                signal_levels = _randint(0, self.max_steps, (batch, time), device=device)

        times = self.get_times_from_signal_level(signal_levels)

        # Noise latents
        if not latent_is_noised:
            noise = _randn_like(latents)
            times_aligned, _ = align_dims_left((times, latents))
            noised_latents = noise.lerp(latents, times_aligned)
        else:
            noised_latents = latents
            noise = None

        # Build tokens
        agent_tokens = self.agent_learned_embed.unsqueeze(0).unsqueeze(0).expand(batch, time, -1, -1)

        # Action tokens
        next_action_tokens = None
        if exists(continuous_actions):
            act_embed = self.action_embedder(continuous_actions)
            act_embed = act_embed + self.action_learned_embed.squeeze(0)

            action_len = act_embed.shape[1]

            if action_len == time and shift_action_tokens:
                next_action_tokens = act_embed[:, 1:]
                act_embed = pad_at_dim(act_embed[:, :-1], (1, 0), value=0., dim=1)
            elif action_len == (time - 1):
                next_action_tokens = act_embed
                act_embed = pad_at_dim(act_embed, (1, 0), value=0., dim=1)
            else:
                next_action_tokens = act_embed

            action_tokens = act_embed.unsqueeze(2)
        elif self.action_embedder.has_actions:
            action_tokens = torch.zeros(batch, time, 1, self.dim, device=device)
        else:
            action_tokens = agent_tokens[:, :, 0:0]

        # Reward tokens
        reward_tokens = agent_tokens[:, :, 0:0]
        if exists(rewards):
            two_hot_encoding = self.reward_encoder(rewards)

            if (
                self.add_reward_embed_to_agent_token and
                (not self.training or not _sample_prob(self.add_reward_embed_dropout))
            ):
                reward_tokens = self.reward_encoder.embed(two_hot_encoding)
                pop_last_reward = int(reward_tokens.shape[1] == agent_tokens.shape[1])
                reward_tokens = pad_at_dim(reward_tokens, (1, -pop_last_reward), dim=1, value=0.)
                reward_tokens = reward_tokens + self.reward_learned_embed
                reward_tokens = reward_tokens.unsqueeze(2)

        # State pred token
        if self.should_pred_state:
            state_pred_token = self.state_pred_token.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch, time, 1, -1)
        else:
            state_pred_token = agent_tokens[:, :, 0:0]

        # Proprio
        noised_proprio = None
        proprio_noise = None
        if self.has_proprio and exists(proprio):
            if not latent_is_noised:
                proprio_noise = _randn_like(proprio)
                times_p, _ = align_dims_left((times, proprio))
                noised_proprio = proprio_noise.lerp(proprio, times_p)
            else:
                noised_proprio = proprio

        # Forward
        get_pred = partial(
            self._get_prediction,
            state_pred_token=state_pred_token,
            action_tokens=action_tokens,
            reward_tokens=reward_tokens,
            agent_tokens=agent_tokens,
        )

        pred, (embeds, intermediates, packed_sizes) = get_pred(
            noised_latents, noised_proprio, signal_levels, step_sizes_log2,
            return_agent_tokens=True, return_time_cache=True, return_intermediates=return_intermediates,
        )

        if return_pred_only:
            if not return_intermediates:
                return pred
            return pred, (embeds, intermediates)

        # ─── Losses ───

        is_x_space = self.pred_orig_latent
        packed_pred = pred.flow

        if self.has_proprio and exists(proprio):
            pred_full = cat([packed_pred.reshape(batch, time, -1), pred.proprioception], dim=-1)
            noised_full = cat([noised_latents.reshape(batch, time, -1), noised_proprio], dim=-1)
            data_full = cat([latents.reshape(batch, time, -1), proprio], dim=-1)
            noise_full = cat([noise.reshape(batch, time, -1), proprio_noise], dim=-1) if exists(noise) else None
        else:
            pred_full = packed_pred.reshape(batch, time, -1)
            noised_full = noised_latents.reshape(batch, time, -1)
            data_full = latents.reshape(batch, time, -1)
            noise_full = noise.reshape(batch, time, -1) if exists(noise) else None

        # Flow loss
        if not is_x_space:
            pred_target = data_full - noise_full
        else:
            pred_target = data_full

        flow_loss_weight = 1.
        if is_x_space:
            flow_loss_weight = (1. - times) ** 2

        flow_losses = F.mse_loss(pred_full, pred_target, reduction='none')

        if isinstance(flow_loss_weight, Tensor):
            flw, _ = align_dims_left((flow_loss_weight, flow_losses))
            flow_losses = flow_losses * flw

        # Shortcut loss
        should_compute_shortcut = not is_inference and shortcut_train

        if should_compute_shortcut:
            step_sizes_log2_minus_one = step_sizes_log2 - 1
            half_step_size = 2 ** step_sizes_log2_minus_one

            # First prediction at half step
            with torch.no_grad():
                first_pred = get_pred(noised_latents, noised_proprio, signal_levels, step_sizes_log2_minus_one)
                first_pred_full = first_pred.flow.reshape(batch, time, -1)
                if self.has_proprio and exists(pred.proprioception):
                    first_pred_full = cat([first_pred_full, first_pred.proprioception], dim=-1)

                if is_x_space:
                    first_times = self.get_times_from_signal_level(signal_levels)
                    ft_aligned, _ = align_dims_left((first_times, first_pred_full))
                    first_flow = (first_pred_full - noised_full) / (1. - ft_aligned).clamp(min=1e-6)
                else:
                    first_flow = first_pred_full

                # Half step
                hs_aligned, _ = align_dims_left((half_step_size, noised_full))
                denoised_half = noised_full + first_flow * (hs_aligned.float() / self.max_steps)

                # Second prediction
                sig_plus_half = signal_levels + half_step_size.unsqueeze(1)
                # Need to split back for the prediction function
                n_lat = self.num_latent_tokens * self.dim_latent
                denoised_lat = denoised_half[..., :n_lat].reshape(batch, time, self.num_latent_tokens, self.dim_latent)
                denoised_proprio_half = denoised_half[..., n_lat:] if self.has_proprio else None

                second_pred = get_pred(denoised_lat, denoised_proprio_half, sig_plus_half, step_sizes_log2_minus_one)
                second_pred_full = second_pred.flow.reshape(batch, time, -1)
                if self.has_proprio and exists(second_pred.proprioception):
                    second_pred_full = cat([second_pred_full, second_pred.proprioception], dim=-1)

                if is_x_space:
                    second_times = self.get_times_from_signal_level(sig_plus_half)
                    st_aligned, _ = align_dims_left((second_times, second_pred_full))
                    second_flow = (second_pred_full - denoised_half) / (1. - st_aligned).clamp(min=1e-6)
                else:
                    second_flow = second_pred_full

                shortcut_target = (first_flow + second_flow).detach() / 2

            shortcut_pred = pred_full
            shortcut_loss_w = 1.
            if is_x_space:
                ft_aligned2, _ = align_dims_left((first_times, pred_full))
                shortcut_pred = (shortcut_pred - noised_full) / (1. - ft_aligned2).clamp(min=1e-6)
                shortcut_loss_w = (1. - first_times) ** 2

            shortcut_losses = F.mse_loss(shortcut_pred, shortcut_target, reduction='none')
            if isinstance(shortcut_loss_w, Tensor):
                slw, _ = align_dims_left((shortcut_loss_w, shortcut_losses))
                shortcut_losses = shortcut_losses * slw
        else:
            shortcut_losses = torch.zeros_like(flow_losses)

        # Ramp weighting
        if exists(self.loss_weight_fn):
            lw = self.loss_weight_fn(times)
            lw, _ = align_dims_left((lw, flow_losses))
            flow_losses = flow_losses * lw

        # Mask for variable length
        is_var_len = exists(lens)
        loss_mask = loss_mask_without_last = None
        if is_var_len:
            loss_mask = lens_to_mask(lens, time)
            loss_mask_without_last = loss_mask[:, :-1]
            flow_loss = flow_losses[loss_mask.unsqueeze(-1).expand_as(flow_losses)].mean()
            shortcut_flow_loss = shortcut_losses[loss_mask.unsqueeze(-1).expand_as(shortcut_losses)].mean() if should_compute_shortcut else self.zero
        else:
            flow_loss = flow_losses.mean()
            shortcut_flow_loss = shortcut_losses.mean() if should_compute_shortcut else self.zero

        # Reward loss
        reward_loss = self.zero
        if exists(rewards):
            two_hot = self.reward_encoder(rewards)
            encoded_agent = embeds.agent[:, :, 0]  # (b, t, d)

            reward_pred = self.to_reward_pred(encoded_agent[:, :-1])
            # reward_pred: (mtp, b, t-1, bins)
            reward_pred = reward_pred.permute(0, 1, 3, 2)  # (mtp, b, bins, t-1)

            reward_targets, reward_loss_mask = create_multi_token_prediction_targets(
                two_hot[:, :-1], self.multi_token_pred_len
            )
            # (b, t-1, mtp, bins) -> (mtp, b, bins, t-1)
            reward_targets = reward_targets.permute(2, 0, 3, 1)

            rew_losses = -(reward_targets * reward_pred.log_softmax(dim=2)).sum(dim=2)
            # (mtp, b, t-1)
            reward_loss_mask = reward_loss_mask.permute(2, 0, 1)
            rew_losses = rew_losses.masked_fill(~reward_loss_mask, 0.)

            if is_var_len:
                # rew_losses: (mtp, b, t-1) — mask and reduce properly
                mask_expanded = loss_mask_without_last.unsqueeze(0).expand_as(rew_losses)  # (mtp, b, t-1)
                rew_losses = rew_losses.masked_fill(~mask_expanded, 0.)
                reward_loss = rew_losses.sum(dim=(1, 2)) / mask_expanded.float().sum(dim=(1, 2)).clamp(min=1.)
            else:
                reward_loss = rew_losses.mean(dim=(1, 2))

        # State prediction loss
        state_pred_loss = self.zero
        if self.should_pred_state and exists(pred.state):
            state_logits = pred.state
            pred_s, target_s = state_logits[:, :-1], latents[:, 1:]
            dist = self.state_beta_dist(pred_s)
            target_s_scaled = (target_s + 1.) / 2.
            target_s_scaled = target_s_scaled.clamp(self.eps_latent_pred, 1. - self.eps_latent_pred)
            state_losses = -dist.log_prob(target_s_scaled)
            if is_var_len:
                state_pred_loss = state_losses[loss_mask_without_last.unsqueeze(-1).unsqueeze(-1).expand_as(state_losses)].mean()
            else:
                state_pred_loss = state_losses.mean()

        # Action autoregressive loss
        continuous_action_loss = self.zero
        if (
            add_autoregressive_action_loss and
            time > 1 and
            exists(continuous_actions)
        ):
            has_continuous = True

            if shift_action_tokens:
                ca_padded = pad_at_dim(continuous_actions, (1, 0), value=0., dim=1)
            else:
                ca_padded = continuous_actions

            pred_len = ca_padded.shape[1]
            num_targets = pred_len - 1 if shift_action_tokens else pred_len

            encoded_agent = embeds.agent[:, :, 0]
            policy_embed = self.policy_head(encoded_agent[:, :num_targets])

            targets, mask_mtp = create_multi_token_prediction_targets(ca_padded, self.multi_token_pred_len)
            if shift_action_tokens:
                targets, mask_mtp = targets[:, 1:], mask_mtp[:, 1:]

            targets = targets.permute(2, 0, 1, 3)  # (mtp, b, t, na)
            mask_mtp = mask_mtp.permute(2, 0, 1)  # (mtp, b, t)

            cont_lp = self.action_embedder.log_probs(policy_embed, targets=targets)
            cont_lp = cont_lp.masked_fill(~mask_mtp.unsqueeze(-1), 0.)

            if is_var_len:
                action_loss_mask = loss_mask_without_last if pred_len == (time - 1) else loss_mask
                cont_losses = -cont_lp.permute(1, 2, 3, 0)  # (b, t, na, mtp)
                continuous_action_loss = cont_losses[action_loss_mask].mean(dim=0).mean(dim=0)
            else:
                continuous_action_loss = (-cont_lp).mean(dim=(1, 2, 3))

        # Agent state prediction loss
        agent_state_pred_loss = self.zero
        if self.agent_predicts_state and exists(next_action_tokens) and next_action_tokens.shape[1] > 0:
            agent_embeds_s = embeds.agent[:, :-1, 0]
            next_act = next_action_tokens
            seq_len = min(agent_embeds_s.shape[1], next_act.shape[1])
            pred_input = cat((agent_embeds_s[:, :seq_len], next_act[:, :seq_len]), dim=-1)
            pred_s2 = self.to_agent_state_pred(pred_input)
            pred_s2 = pred_s2.reshape(batch, seq_len, self.num_latent_tokens, self.dim_latent, 2)
            target_s2 = latents[:, 1:1+seq_len]
            target_s2 = (target_s2 + 1.) / 2.
            target_s2 = target_s2.clamp(self.eps_latent_pred, 1. - self.eps_latent_pred)
            dist2 = self.agent_state_beta_dist(pred_s2)
            as_losses = -dist2.log_prob(target_s2)
            agent_state_pred_loss = as_losses.mean()

        # Loss normalization
        if exists(self.flow_loss_normalizer):
            flow_loss = self.flow_loss_normalizer(flow_loss, update_ema=update_loss_ema)
        if exists(self.shortcut_flow_loss_normalizer) and should_compute_shortcut:
            shortcut_flow_loss = self.shortcut_flow_loss_normalizer(shortcut_flow_loss, update_ema=update_loss_ema)
        if exists(rewards) and exists(self.reward_loss_normalizer):
            reward_loss = self.reward_loss_normalizer(reward_loss, update_ema=update_loss_ema)
        if exists(continuous_actions) and exists(self.continuous_actions_loss_normalizer):
            continuous_action_loss = self.continuous_actions_loss_normalizer(
                continuous_action_loss,
                update_ema=update_loss_ema,
            )

        # Total loss
        total_loss = (
            flow_loss * self.latent_flow_loss_weight +
            shortcut_flow_loss * self.shortcut_loss_weight +
            (reward_loss * self.reward_loss_weight).sum() +
            (continuous_action_loss * self.continuous_action_loss_weight).sum() +
            state_pred_loss * self.state_pred_loss_weight +
            agent_state_pred_loss * self.agent_state_pred_loss_weight
        )

        losses = WorldModelLosses(
            flow_loss, shortcut_flow_loss, reward_loss,
            continuous_action_loss, state_pred_loss, agent_state_pred_loss,
        )

        if not (return_all_losses or return_intermediates):
            return total_loss

        ret = (total_loss,)
        if return_all_losses:
            ret = (*ret, losses)
        if return_intermediates:
            ret = (*ret, intermediates)
        return ret
