"""Tests for d4_wm.world_model — DynamicsWorldModel end-to-end."""
import pytest
import torch

from cleanrl.d4_wm.utils import StateTokenizer, Experience, Actions
from cleanrl.d4_wm.world_model import DynamicsWorldModel


@pytest.fixture
def small_wm():
    """A small world model for fast testing."""
    tok = StateTokenizer(dim_obs=17, dim_latent=16, num_latent_tokens=4, dim_hidden=64)
    wm = DynamicsWorldModel(
        dim=64, dim_latent=16, state_tokenizer=tok,
        max_steps=16, num_register_tokens=2, num_spatial_tokens=4,
        num_latent_tokens=4, depth=4, time_block_every=4,
        attn_heads=4, attn_dim_head=16, num_continuous_actions=6,
        multi_token_pred_len=2, value_head_mlp_depth=2, policy_head_mlp_depth=2,
    )
    return wm


@pytest.fixture
def latents(small_wm):
    """Pre-tokenized latents for testing."""
    obs = torch.randn(2, 8, 17)
    return small_wm.state_tokenizer.tokenize(obs)


# ─── Construction ───

class TestConstruction:
    def test_param_count(self, small_wm):
        n = sum(p.numel() for p in small_wm.parameters())
        assert n > 0

    def test_device(self, small_wm):
        assert small_wm.device.type == 'cpu'

    def test_policy_head_params(self, small_wm):
        params = small_wm.policy_head_parameters()
        assert len(params) > 0

    def test_value_head_params(self, small_wm):
        params = list(small_wm.value_head_parameters())
        assert len(params) > 0


# ─── Forward pass ───

class TestForward:
    def test_basic(self, small_wm, latents):
        loss = small_wm(
            latents=latents,
            rewards=torch.randn(2, 8),
            continuous_actions=torch.randn(2, 8, 6),
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_backward(self, small_wm, latents):
        loss = small_wm(
            latents=latents,
            rewards=torch.randn(2, 8),
            continuous_actions=torch.randn(2, 8, 6),
        )
        loss.backward()
        grads = [p.grad for p in small_wm.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_return_all_losses(self, small_wm, latents):
        total, losses = small_wm(
            latents=latents,
            rewards=torch.randn(2, 8),
            continuous_actions=torch.randn(2, 8, 6),
            return_all_losses=True,
        )
        assert torch.isfinite(total)
        assert hasattr(losses, 'flow')
        assert hasattr(losses, 'shortcut')
        assert hasattr(losses, 'rewards')
        assert hasattr(losses, 'continuous_actions')

    def test_no_actions(self, small_wm, latents):
        """Forward with only latents and rewards (no actions)."""
        loss = small_wm(
            latents=latents,
            rewards=torch.randn(2, 8),
        )
        assert torch.isfinite(loss)

    def test_no_rewards(self, small_wm, latents):
        """Forward with only latents (no rewards or actions)."""
        loss = small_wm(latents=latents)
        assert torch.isfinite(loss)

    def test_from_obs(self, small_wm):
        """Forward from raw observations."""
        obs = torch.randn(2, 8, 17)
        loss = small_wm(
            obs=obs,
            rewards=torch.randn(2, 8),
            continuous_actions=torch.randn(2, 8, 6),
        )
        assert torch.isfinite(loss)

    def test_reward_embed_to_agent_token_path(self):
        tok = StateTokenizer(dim_obs=17, dim_latent=16, num_latent_tokens=4, dim_hidden=64)
        wm = DynamicsWorldModel(
            dim=64, dim_latent=16, state_tokenizer=tok,
            max_steps=16, num_register_tokens=2, num_spatial_tokens=4,
            num_latent_tokens=4, depth=4, time_block_every=4,
            attn_heads=4, attn_dim_head=16, num_continuous_actions=6,
            multi_token_pred_len=2, value_head_mlp_depth=2, policy_head_mlp_depth=2,
            add_reward_embed_to_agent_token=True,
        )
        obs = torch.randn(2, 8, 17)
        latents = wm.state_tokenizer.tokenize(obs)
        loss = wm(
            latents=latents,
            rewards=torch.randn(2, 8),
            continuous_actions=torch.randn(2, 8, 6),
        )
        assert torch.isfinite(loss)

    def test_different_batch_sizes(self, small_wm):
        for batch in [1, 4]:
            lat = small_wm.state_tokenizer.tokenize(torch.randn(batch, 6, 17))
            loss = small_wm(latents=lat, rewards=torch.randn(batch, 6))
            assert torch.isfinite(loss)

    def test_return_intermediates(self, small_wm, latents):
        out = small_wm(
            latents=latents,
            rewards=torch.randn(2, 8),
            continuous_actions=torch.randn(2, 8, 6),
            return_all_losses=True,
            return_intermediates=True,
        )
        total, losses, intermediates = out
        assert torch.isfinite(total)
        assert intermediates is not None

    def test_return_pred_only(self, small_wm, latents):
        pred = small_wm(
            latents=latents,
            signal_levels=small_wm.max_steps - 1,
            step_sizes=4,
            latent_is_noised=True,
            return_pred_only=True,
        )
        assert pred.flow.shape == latents.shape

    def test_reward_length_validation(self, small_wm, latents):
        with pytest.raises(AssertionError, match='rewards length must be'):
            small_wm(
                latents=latents,
                rewards=torch.randn(2, 5),
                continuous_actions=torch.randn(2, 8, 6),
            )

    def test_return_pred_only_allows_time_minus_one_rewards(self, small_wm, latents):
        pred = small_wm(
            latents=latents,
            rewards=torch.randn(2, 7),
            signal_levels=small_wm.max_steps - 1,
            step_sizes=4,
            latent_is_noised=True,
            return_pred_only=True,
        )
        assert pred.flow.shape == latents.shape


# ─── Token packing ───

class TestTokenPacking:
    def test_pack_unpack_roundtrip(self, small_wm):
        batch, time, dim = 2, 4, 64
        parts = [
            torch.randn(batch, time, 1, dim),    # flow
            torch.randn(batch, time, 4, dim),    # space
            torch.randn(batch, time, 0, dim),    # proprio (empty)
            torch.randn(batch, time, 0, dim),    # state_pred (empty)
            torch.randn(batch, time, 2, dim),    # registers
            torch.randn(batch, time, 1, dim),    # action
            torch.randn(batch, time, 0, dim),    # reward (empty)
            torch.randn(batch, time, 1, dim),    # agent
        ]
        packed, sizes = small_wm._pack_tokens(*parts)
        unpacked = small_wm._unpack_tokens(packed, sizes)
        for orig, recon in zip(parts, unpacked):
            assert torch.allclose(orig, recon)


# ─── Generation ───

class TestGenerate:
    def test_basic(self, small_wm):
        start_lat = small_wm.state_tokenizer.tokenize(torch.randn(2, 1, 17))
        dreams = small_wm.generate(
            time_steps=4, num_steps=4, batch_size=2,
            initial_latents=start_lat,
            return_for_policy_optimization=True,
            use_time_cache=False,
        )
        assert dreams.latents.shape[0] == 2
        assert dreams.latents.shape[1] == 4
        assert dreams.rewards is not None
        assert dreams.actions is not None
        assert dreams.log_probs is not None
        assert dreams.values is not None

    def test_longer_horizon(self, small_wm):
        start_lat = small_wm.state_tokenizer.tokenize(torch.randn(1, 1, 17))
        dreams = small_wm.generate(
            time_steps=8, num_steps=4, batch_size=1,
            initial_latents=start_lat, use_time_cache=False,
        )
        assert dreams.latents.shape == (1, 8, 4, 16)
        assert dreams.rewards.shape == (1, 7)

    def test_no_policy_opt(self, small_wm):
        start_lat = small_wm.state_tokenizer.tokenize(torch.randn(2, 1, 17))
        dreams = small_wm.generate(
            time_steps=4, num_steps=4, batch_size=2,
            initial_latents=start_lat,
            return_for_policy_optimization=False,
            use_time_cache=False,
        )
        assert dreams.log_probs is None
        assert dreams.values is None

    def test_temperature(self, small_wm):
        start_lat = small_wm.state_tokenizer.tokenize(torch.randn(2, 1, 17))
        dreams = small_wm.generate(
            time_steps=3, num_steps=4, batch_size=2,
            initial_latents=start_lat,
            continuous_temperature=0.5,
            use_time_cache=False,
        )
        assert dreams.actions is not None

    def test_reward_embed_generation_without_initial_rewards(self):
        tok = StateTokenizer(dim_obs=17, dim_latent=16, num_latent_tokens=4, dim_hidden=64)
        wm = DynamicsWorldModel(
            dim=64, dim_latent=16, state_tokenizer=tok,
            max_steps=16, num_register_tokens=2, num_spatial_tokens=4,
            num_latent_tokens=4, depth=4, time_block_every=4,
            attn_heads=4, attn_dim_head=16, num_continuous_actions=6,
            multi_token_pred_len=2, value_head_mlp_depth=2, policy_head_mlp_depth=2,
            add_reward_embed_to_agent_token=True,
        )
        start_lat = wm.state_tokenizer.tokenize(torch.randn(2, 1, 17))
        dreams = wm.generate(
            time_steps=4, num_steps=4, batch_size=2,
            initial_latents=start_lat,
            use_time_cache=False,
        )
        assert dreams.rewards.shape == (2, 3)


# ─── Learn from experience ───

class TestLearnFromExperience:
    def _make_dreams(self, small_wm, batch=2, horizon=5):
        start_lat = small_wm.state_tokenizer.tokenize(torch.randn(batch, 1, 17))
        return small_wm.generate(
            time_steps=horizon, num_steps=4, batch_size=batch,
            initial_latents=start_lat,
            return_for_policy_optimization=True,
            use_time_cache=False,
        )

    def test_ppo(self, small_wm):
        dreams = self._make_dreams(small_wm)
        pi_loss, v_loss = small_wm.learn_from_experience(dreams, use_pmpo=False)
        assert torch.isfinite(pi_loss)
        assert torch.isfinite(v_loss)

    def test_pmpo(self, small_wm):
        dreams = self._make_dreams(small_wm)
        pi_loss, v_loss = small_wm.learn_from_experience(dreams, use_pmpo=True)
        assert torch.isfinite(pi_loss)
        assert torch.isfinite(v_loss)

    def test_backward_policy(self, small_wm):
        dreams = self._make_dreams(small_wm)
        pi_loss, v_loss = small_wm.learn_from_experience(dreams, use_pmpo=False)
        pi_loss.backward()
        has_grad = any(p.grad is not None for p in small_wm.policy_head_parameters())
        assert has_grad

    def test_backward_value(self, small_wm):
        dreams = self._make_dreams(small_wm)
        pi_loss, v_loss = small_wm.learn_from_experience(dreams, use_pmpo=False)
        v_loss.backward()
        has_grad = any(p.grad is not None for p in small_wm.value_head_parameters())
        assert has_grad

    def test_separate_backwards(self, small_wm):
        """Policy and value losses should have independent computation graphs."""
        dreams = self._make_dreams(small_wm)
        pi_loss, v_loss = small_wm.learn_from_experience(dreams, use_pmpo=False)
        pi_loss.backward()
        v_loss.backward()  # should not fail

    def test_delight_gating(self, small_wm):
        dreams = self._make_dreams(small_wm)
        pi_loss, v_loss = small_wm.learn_from_experience(
            dreams, use_pmpo=False, use_delight_gating=True
        )
        assert torch.isfinite(pi_loss)

    def test_no_delight_gating(self, small_wm):
        dreams = self._make_dreams(small_wm)
        pi_loss, v_loss = small_wm.learn_from_experience(
            dreams, use_pmpo=False, use_delight_gating=False
        )
        assert torch.isfinite(pi_loss)


# ─── Training step (optimizer integration) ───

class TestTrainingStep:
    def test_wm_optim_step(self, small_wm, latents):
        opt = torch.optim.Adam(small_wm.parameters(), lr=1e-3)
        rewards = torch.randn(2, 8)
        actions = torch.randn(2, 8, 6)

        loss0 = small_wm(latents=latents, rewards=rewards, continuous_actions=actions)
        opt.zero_grad()
        loss0.backward()
        opt.step()

        loss1 = small_wm(latents=latents, rewards=rewards, continuous_actions=actions)
        # Loss should change after a step
        assert loss0.item() != loss1.item()

    def test_multi_step_convergence(self, small_wm, latents):
        """Loss should generally decrease over multiple steps."""
        opt = torch.optim.Adam(small_wm.parameters(), lr=1e-3)
        rewards = torch.randn(2, 8)
        actions = torch.randn(2, 8, 6)

        losses = []
        for _ in range(10):
            loss = small_wm(latents=latents, rewards=rewards, continuous_actions=actions)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(small_wm.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        # Loss at end should be less than at start (on average)
        assert losses[-1] < losses[0]


# ─── Edge cases ───

class TestEdgeCases:
    def test_single_timestep(self, small_wm):
        lat = small_wm.state_tokenizer.tokenize(torch.randn(1, 1, 17))
        loss = small_wm(latents=lat)
        assert torch.isfinite(loss)

    def test_large_batch(self, small_wm):
        lat = small_wm.state_tokenizer.tokenize(torch.randn(8, 4, 17))
        loss = small_wm(latents=lat, rewards=torch.randn(8, 4))
        assert torch.isfinite(loss)

    def test_latent_shape_mismatch(self, small_wm):
        """Should raise on wrong latent shape."""
        bad_latents = torch.randn(2, 4, 3, 16)  # wrong num_latent_tokens
        with pytest.raises(AssertionError):
            small_wm(latents=bad_latents)
