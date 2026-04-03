"""Tests for d4_wm.utils — helpers, data structures, encodings, GAE."""
import pytest
import torch
import torch.nn.functional as F
from torch import tensor

from cleanrl.d4_wm.utils import (
    exists, default, divisible_by, is_power_two, l2norm, softclamp,
    safe_cat, safe_stack, lens_to_mask, masked_mean,
    pad_at_dim, pad_right_at_dim_to, align_dims_left,
    create_multi_token_prediction_targets,
    LossNormalizer, SymExpTwoHot, BetaDist, Ensemble, build_mlp,
    calc_gae, ramp_weight, StateTokenizer, Experience,
)


# ─── Basic helpers ───

class TestHelpers:
    def test_exists(self):
        assert exists(1)
        assert not exists(None)

    def test_default(self):
        assert default(None, 5) == 5
        assert default(3, 5) == 3

    def test_divisible_by(self):
        assert divisible_by(6, 3)
        assert not divisible_by(7, 3)

    def test_is_power_two(self):
        assert is_power_two(1)
        assert is_power_two(16)
        assert is_power_two(64)
        assert not is_power_two(6)


# ─── Tensor helpers ───

class TestTensorHelpers:
    def test_l2norm(self):
        t = torch.randn(3, 4)
        normed = l2norm(t)
        norms = normed.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)

    def test_softclamp(self):
        t = torch.tensor([100., -100., 0.])
        clamped = softclamp(t, value=50.)
        assert clamped.abs().max() <= 50.0 + 1e-5

    def test_safe_cat_filters_none(self):
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        result = safe_cat((a, None, b), dim=0)
        assert result.shape == (4, 3)

    def test_safe_cat_all_none(self):
        assert safe_cat((None, None)) is None

    def test_safe_stack(self):
        a = torch.randn(3)
        b = torch.randn(3)
        result = safe_stack((a, None, b))
        assert result.shape == (2, 3)

    def test_lens_to_mask(self):
        lens = tensor([2, 4, 1])
        mask = lens_to_mask(lens, 5)
        assert mask.shape == (3, 5)
        assert mask[0].tolist() == [True, True, False, False, False]
        assert mask[1].tolist() == [True, True, True, True, False]
        assert mask[2].tolist() == [True, False, False, False, False]

    def test_masked_mean_no_mask(self):
        t = torch.ones(2, 3)
        assert masked_mean(t).item() == pytest.approx(1.0)

    def test_masked_mean_with_mask(self):
        t = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        mask = torch.tensor([[True, True, False], [True, False, False]])
        result = masked_mean(t, mask)
        expected = (1 + 2 + 4) / 3
        assert result.item() == pytest.approx(expected)

    def test_pad_at_dim(self):
        t = torch.ones(2, 3, 4)
        padded = pad_at_dim(t, (1, 2), dim=1, value=0.)
        assert padded.shape == (2, 6, 4)
        assert padded[:, 0].sum().item() == 0.
        assert padded[:, -2:].sum().item() == 0.

    def test_pad_right_at_dim_to(self):
        t = torch.ones(2, 3)
        padded = pad_right_at_dim_to(t, 5, dim=1)
        assert padded.shape == (2, 5)
        assert padded[:, 3:].sum().item() == 0.

    def test_pad_right_at_dim_to_no_op(self):
        t = torch.ones(2, 5)
        padded = pad_right_at_dim_to(t, 3, dim=1)
        assert padded.shape == (2, 5)  # no padding needed

    def test_align_dims_left(self):
        t = torch.randn(2, 3)
        ref = torch.randn(2, 3, 4, 5)
        aligned, _ = align_dims_left((t, ref))
        assert aligned.shape == (2, 3, 1, 1)


# ─── Multi-token prediction targets ───

class TestMTP:
    def test_shapes(self):
        t = torch.randn(2, 8, 4)
        targets, mask = create_multi_token_prediction_targets(t, 3)
        assert targets.shape == (2, 8, 3, 4)
        assert mask.shape == (2, 8, 3)

    def test_values(self):
        # Simple sequential data
        t = torch.arange(10).float().unsqueeze(0).unsqueeze(-1)  # (1, 10, 1)
        targets, mask = create_multi_token_prediction_targets(t, 3)
        # targets[0, 0, :, 0] should be [0, 1, 2]
        assert targets[0, 0, 0, 0].item() == 0.
        assert targets[0, 0, 1, 0].item() == 1.
        assert targets[0, 0, 2, 0].item() == 2.

    def test_mask_end(self):
        t = torch.randn(1, 5, 2)
        _, mask = create_multi_token_prediction_targets(t, 4)
        # At position 3 (4th), only 1 future step is valid
        assert mask[0, 3].tolist() == [True, True, False, False]
        # At position 4 (last), only current is valid
        assert mask[0, 4].tolist() == [True, False, False, False]


# ─── LossNormalizer ───

class TestLossNormalizer:
    def test_basic(self):
        ln = LossNormalizer(num_losses=1)
        loss = tensor(2.0)
        normalized = ln(loss)
        # Initial exp_avg_sq is 1.0, so rms=1.0, output = loss / 1.0
        assert normalized.item() == pytest.approx(2.0)

    def test_ema_update(self):
        ln = LossNormalizer(num_losses=1, beta=0.9)
        ln.train()
        for _ in range(50):
            ln(tensor(4.0))
        # After many steps, exp_avg_sq should converge toward 16.0
        assert ln.exp_avg_sq.item() == pytest.approx(16.0, rel=0.1)


# ─── SymExpTwoHot ───

class TestSymExpTwoHot:
    def test_output_shape(self):
        enc = SymExpTwoHot(num_bins=255)
        vals = torch.randn(3, 4)
        encoded = enc(vals)
        assert encoded.shape == (3, 4, 255)

    def test_sums_to_one(self):
        enc = SymExpTwoHot(num_bins=255)
        vals = torch.randn(10)
        encoded = enc(vals)
        sums = encoded.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(10), atol=1e-5)

    def test_roundtrip(self):
        enc = SymExpTwoHot(num_bins=255)
        vals = torch.tensor([0., 5., -5., 10., -10.])
        encoded = enc(vals)
        decoded = enc.bins_to_scalar_value(encoded, normalize=False)
        assert torch.allclose(vals, decoded, atol=0.5)

    def test_boundary_values(self):
        """Bug fix: right_indices should not go out of bounds."""
        enc = SymExpTwoHot(num_bins=255)
        max_val = enc.bin_values[-1].item()
        min_val = enc.bin_values[0].item()
        vals = torch.tensor([max_val, min_val, max_val + 100, min_val - 100])
        encoded = enc(vals)
        assert encoded.shape == (4, 255)
        assert torch.allclose(encoded.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_scalar_input(self):
        enc = SymExpTwoHot(num_bins=255)
        val = torch.tensor(3.0)
        encoded = enc(val)
        assert encoded.shape == (255,)


# ─── GAE ───

class TestGAE:
    def test_shape(self):
        rewards = torch.randn(2, 10)
        values = torch.randn(2, 10)
        returns = calc_gae(rewards, values)
        assert returns.shape == (2, 10)

    def test_no_discount(self):
        """With gamma=0, returns = rewards + 0 * next_values - values + values = rewards."""
        rewards = torch.tensor([[1., 2., 3.]])
        values = torch.zeros(1, 3)
        returns = calc_gae(rewards, values, gamma=0., lam=0.)
        assert torch.allclose(returns, rewards, atol=1e-5)

    def test_with_masks(self):
        rewards = torch.tensor([[1., 2., 3.]])
        values = torch.ones(1, 3)
        masks = torch.tensor([[1., 0., 1.]])  # episode ends after step 1
        returns = calc_gae(rewards, values, masks=masks, gamma=0.99, lam=0.95)
        assert returns.shape == (1, 3)
        # After terminal (mask=0), bootstrap is cut off
        # Step 1 (index 1): delta = 2 + 0.99 * 0 * 1 - 1 = 1
        assert returns[0, 1].item() != returns[0, 0].item()

    def test_monotonic_gamma(self):
        """Higher gamma should give higher returns for positive rewards."""
        rewards = torch.ones(1, 10)
        values = torch.zeros(1, 10)
        r_low = calc_gae(rewards, values, gamma=0.5, lam=0.95)
        r_high = calc_gae(rewards, values, gamma=0.99, lam=0.95)
        assert r_high.sum() > r_low.sum()


# ─── Ramp weight ───

class TestRampWeight:
    def test_values(self):
        times = torch.tensor([0., 0.5, 1.])
        weights = ramp_weight(times)
        assert weights[0].item() == pytest.approx(0.1)
        assert weights[1].item() == pytest.approx(0.55)
        assert weights[2].item() == pytest.approx(1.0)


# ─── BetaDist ───

class TestBetaDist:
    def test_unimodal(self):
        bd = BetaDist(unimodal=True)
        params = torch.randn(3, 4, 2)
        dist = bd(params)
        samples = dist.sample()
        assert samples.shape == (3, 4)
        assert (samples >= 0).all() and (samples <= 1).all()


# ─── build_mlp ───

class TestBuildMLP:
    def test_output_shape(self):
        mlp = build_mlp(16, 64, 8, depth=3)
        x = torch.randn(2, 16)
        out = mlp(x)
        assert out.shape == (2, 8)


# ─── Ensemble ───

class TestEnsemble:
    def test_forward(self):
        def make_linear():
            return torch.nn.Linear(4, 3)
        ens = Ensemble(make_linear, 5)
        x = torch.randn(2, 4)
        out = ens(x)
        assert out.shape == (5, 2, 3)

    def test_forward_one(self):
        def make_linear():
            return torch.nn.Linear(4, 3)
        ens = Ensemble(make_linear, 5)
        x = torch.randn(2, 4)
        out = ens.forward_one(x, id=2)
        assert out.shape == (2, 3)


# ─── StateTokenizer ───

class TestStateTokenizer:
    @pytest.fixture
    def tokenizer(self):
        return StateTokenizer(dim_obs=17, dim_latent=16, num_latent_tokens=4, dim_hidden=64)

    def test_forward_loss(self, tokenizer):
        obs = torch.randn(2, 8, 17)
        loss, latents = tokenizer(obs)
        assert loss.shape == ()
        assert loss.item() > 0
        assert latents.shape == (2, 8, 4, 16)

    def test_tokenize(self, tokenizer):
        obs = torch.randn(3, 5, 17)
        latents = tokenizer.tokenize(obs)
        assert latents.shape == (3, 5, 4, 16)

    def test_tokenize_matches_forward_return_latents(self, tokenizer):
        obs = torch.randn(3, 5, 17)
        latents_a = tokenizer.tokenize(obs)
        latents_b = tokenizer(obs, return_latents=True)
        assert torch.allclose(latents_a, latents_b)

    def test_decode(self, tokenizer):
        latents = torch.randn(2, 4, 4, 16).tanh()
        recon = tokenizer.decode(latents)
        assert recon.shape == (2, 4, 17)

    def test_tanh_bounded(self, tokenizer):
        obs = torch.randn(2, 4, 17)
        _, latents = tokenizer(obs)
        assert latents.abs().max() <= 1.0

    def test_return_intermediates(self, tokenizer):
        obs = torch.randn(2, 4, 17)
        total, latents, losses, recon = tokenizer(obs, return_intermediates=True)
        assert torch.isfinite(total)
        assert latents.shape == (2, 4, 4, 16)
        assert torch.isfinite(losses.recon)
        assert torch.isfinite(losses.time_decorr)
        assert torch.isfinite(losses.space_decorr)
        assert recon.shape == obs.shape

    def test_tokens_have_nontrivial_spread(self, tokenizer):
        obs = torch.randn(8, 8, 17)
        latents = tokenizer(obs, return_latents=True)
        assert latents.std().item() > 1e-3

    def test_gradient_flow(self, tokenizer):
        obs = torch.randn(2, 4, 17)
        loss, _ = tokenizer(obs)
        loss.backward()
        for p in tokenizer.parameters():
            assert p.grad is not None


# ─── Experience ───

class TestExperience:
    def test_to_device(self):
        exp = Experience(
            latents=torch.randn(2, 3, 4, 16),
            rewards=torch.randn(2, 3),
        )
        exp_cpu = exp.to(torch.device('cpu'))
        assert exp_cpu.latents.device.type == 'cpu'

    def test_cpu(self):
        exp = Experience(latents=torch.randn(2, 3, 4, 16))
        exp_cpu = exp.cpu()
        assert exp_cpu.latents.device.type == 'cpu'
