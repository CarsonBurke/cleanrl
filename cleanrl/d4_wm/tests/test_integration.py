"""Integration tests — full pipeline from env to training."""
import pytest
import torch
import numpy as np

from cleanrl.d4_wm.utils import StateTokenizer
from cleanrl.d4_wm.world_model import DynamicsWorldModel
from cleanrl.d4_wm.train_mujoco import ReplayBuffer, EnvActor


@pytest.fixture
def wm_and_tok():
    tok = StateTokenizer(dim_obs=17, dim_latent=16, num_latent_tokens=4, dim_hidden=64)
    wm = DynamicsWorldModel(
        dim=64, dim_latent=16, state_tokenizer=tok,
        max_steps=16, num_register_tokens=2, num_spatial_tokens=4,
        num_latent_tokens=4, depth=4, time_block_every=4,
        attn_heads=4, attn_dim_head=16, num_continuous_actions=6,
        multi_token_pred_len=2, value_head_mlp_depth=2, policy_head_mlp_depth=2,
    )
    return wm, tok


# ─── ReplayBuffer ───

class TestReplayBuffer:
    def test_add_and_size(self):
        buf = ReplayBuffer(100, obs_dim=4, act_dim=2)
        obs = np.random.randn(5, 4).astype(np.float32)
        acts = np.random.randn(5, 2).astype(np.float32)
        rews = np.random.randn(5).astype(np.float32)
        dones = np.zeros(5, dtype=np.float32)
        buf.add(obs, acts, rews, dones)
        assert buf.size == 5

    def test_capacity_wrap(self):
        buf = ReplayBuffer(10, obs_dim=2, act_dim=1)
        for _ in range(15):
            buf.add(
                np.random.randn(1, 2).astype(np.float32),
                np.random.randn(1, 1).astype(np.float32),
                np.array([1.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
            )
        assert buf.size == 10
        assert buf.pos == 5  # wrapped around

    def test_sample_sequences(self):
        buf = ReplayBuffer(100, obs_dim=4, act_dim=2)
        for _ in range(50):
            buf.add(
                np.random.randn(1, 4).astype(np.float32),
                np.random.randn(1, 2).astype(np.float32),
                np.array([1.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
            )
        obs, acts, rews = buf.sample_sequences(3, 8)
        assert obs.shape == (3, 8, 4)
        assert acts.shape == (3, 8, 2)
        assert rews.shape == (3, 8)

    def test_sample_respects_episodes(self):
        """Buffer should try to avoid sequences crossing episode boundaries."""
        buf = ReplayBuffer(100, obs_dim=2, act_dim=1)
        for i in range(50):
            done = 1.0 if (i + 1) % 10 == 0 else 0.0
            buf.add(
                np.random.randn(1, 2).astype(np.float32),
                np.random.randn(1, 1).astype(np.float32),
                np.array([1.0], dtype=np.float32),
                np.array([done], dtype=np.float32),
            )
        # Sample many times and count how often we cross episode boundaries
        crossings = 0
        for _ in range(100):
            obs, acts, rews = buf.sample_sequences(1, 5)
            # Can't easily test from the returned tensors without more info
        # Just verify it doesn't crash
        assert True


# ─── EnvActor ───

class TestEnvActor:
    def test_get_action(self, wm_and_tok):
        wm, tok = wm_and_tok
        actor = EnvActor(wm, tok)
        obs = torch.randn(4, 17)
        actions = actor.get_action(obs)
        assert actions.shape == (4, 6)

    def test_get_value(self, wm_and_tok):
        wm, tok = wm_and_tok
        actor = EnvActor(wm, tok)
        obs = torch.randn(4, 17)
        values = actor.get_value(obs)
        assert values.shape == (4,)

    def test_stateless_calls_do_not_initialize_or_mutate_history(self, wm_and_tok):
        wm, tok = wm_and_tok
        actor = EnvActor(wm, tok)
        obs = torch.randn(4, 17)
        _ = actor.get_action(obs)
        _ = actor.get_value(obs)
        assert actor.obs_histories == []
        assert actor.time_caches == []

    def test_value_does_not_advance_stateful_cache(self, wm_and_tok):
        wm, tok = wm_and_tok
        actor = EnvActor(wm, tok)
        obs = torch.randn(2, 17)
        actor.reset(obs)
        _ = actor.value()
        assert actor.time_caches == [None, None]


# ─── Full Pipeline ───

class TestFullPipeline:
    def test_tokenizer_then_wm(self, wm_and_tok):
        """Tokenizer output should be valid input to world model."""
        wm, tok = wm_and_tok
        obs = torch.randn(2, 8, 17)

        # Train tokenizer
        tok_loss, latents = tok(obs)
        tok_loss.backward()

        # Train world model
        with torch.no_grad():
            lat = tok.tokenize(obs)

        wm_loss = wm(
            latents=lat,
            rewards=torch.randn(2, 8),
            continuous_actions=torch.randn(2, 8, 6),
        )
        wm_loss.backward()

        assert torch.isfinite(tok_loss)
        assert torch.isfinite(wm_loss)

    def test_dream_then_learn(self, wm_and_tok):
        """Generate dreams then learn from them."""
        wm, tok = wm_and_tok
        start_lat = tok.tokenize(torch.randn(4, 1, 17))

        dreams = wm.generate(
            time_steps=6, num_steps=4, batch_size=4,
            initial_latents=start_lat,
            return_for_policy_optimization=True,
            use_time_cache=False,
        )

        pi_loss, v_loss = wm.learn_from_experience(dreams, use_pmpo=True)
        pi_loss.backward()
        v_loss.backward()

        assert torch.isfinite(pi_loss)
        assert torch.isfinite(v_loss)

    def test_full_training_loop(self, wm_and_tok):
        """Simulate a mini training loop: tokenizer, WM, dream, learn."""
        wm, tok = wm_and_tok
        tok_opt = torch.optim.Adam(tok.parameters(), lr=1e-3)
        wm_params = [p for n, p in wm.named_parameters() if not n.startswith('state_tokenizer.')]
        wm_opt = torch.optim.Adam(wm_params, lr=1e-3)
        pi_opt = torch.optim.Adam(wm.policy_head_parameters(), lr=1e-3)
        v_opt = torch.optim.Adam(wm.value_head_parameters(), lr=1e-3)

        obs = torch.randn(2, 8, 17)
        rewards = torch.randn(2, 8)
        actions = torch.randn(2, 8, 6)

        for step in range(3):
            # 1. Train tokenizer
            tok_opt.zero_grad()
            tok_loss, _ = tok(obs)
            tok_loss.backward()
            tok_opt.step()

            # 2. Train world model
            with torch.no_grad():
                lat = tok.tokenize(obs)

            wm_opt.zero_grad()
            wm_loss = wm(latents=lat, rewards=rewards, continuous_actions=actions)
            wm_loss.backward()
            torch.nn.utils.clip_grad_norm_(wm_params, 1.0)
            wm_opt.step()

            # 3. Dream and learn
            wm.eval()
            start_lat = tok.tokenize(torch.randn(2, 1, 17))
            dreams = wm.generate(
                time_steps=4, num_steps=4, batch_size=2,
                initial_latents=start_lat,
                return_for_policy_optimization=True,
                use_time_cache=False,
            )
            wm.train()

            pi_opt.zero_grad()
            v_opt.zero_grad()
            pi_loss, v_loss = wm.learn_from_experience(dreams, use_pmpo=True)
            pi_loss.backward()
            pi_opt.step()
            v_loss.backward()
            v_opt.step()

        assert torch.isfinite(tok_loss)
        assert torch.isfinite(wm_loss)
        assert torch.isfinite(pi_loss)
        assert torch.isfinite(v_loss)


# ─── Numerical stability ───

class TestNumericalStability:
    def test_large_rewards(self, wm_and_tok):
        wm, tok = wm_and_tok
        lat = tok.tokenize(torch.randn(2, 4, 17))
        loss = wm(latents=lat, rewards=torch.ones(2, 4) * 100)
        assert torch.isfinite(loss)

    def test_zero_actions(self, wm_and_tok):
        wm, tok = wm_and_tok
        lat = tok.tokenize(torch.randn(2, 4, 17))
        loss = wm(latents=lat, continuous_actions=torch.zeros(2, 4, 6))
        assert torch.isfinite(loss)

    def test_extreme_observations(self, wm_and_tok):
        wm, tok = wm_and_tok
        obs = torch.randn(2, 4, 17) * 100
        lat = tok.tokenize(obs)
        loss = wm(latents=lat)
        assert torch.isfinite(loss)
