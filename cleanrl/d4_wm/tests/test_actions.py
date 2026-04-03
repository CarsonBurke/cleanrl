"""Tests for d4_wm.actions — ContinuousActionEmbedder."""
import pytest
import torch

from cleanrl.d4_wm.actions import ContinuousActionEmbedder


@pytest.fixture
def embedder():
    return ContinuousActionEmbedder(
        dim=64, num_actions=6, unembed_dim=256, num_unembed_preds=4,
    )


@pytest.fixture
def embedder_single_pred():
    return ContinuousActionEmbedder(
        dim=64, num_actions=6, unembed_dim=256, num_unembed_preds=1,
    )


# ─── Forward (embed) ───

class TestEmbed:
    def test_shape(self, embedder):
        actions = torch.randn(2, 8, 6)
        out = embedder(actions)
        assert out.shape == (2, 8, 64)

    def test_zero_actions(self, embedder):
        actions = torch.zeros(2, 8, 6)
        out = embedder(actions)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_gradient(self, embedder):
        actions = torch.randn(2, 4, 6, requires_grad=True)
        out = embedder(actions)
        out.sum().backward()
        assert actions.grad is not None


# ─── Unembed ───

class TestUnembed:
    def test_shape_all_heads(self, embedder):
        embeds = torch.randn(2, 8, 256)
        out = embedder.unembed(embeds)
        # (mtp=4, batch=2, time=8, na=6, 2)
        assert out.shape == (4, 2, 8, 6, 2)

    def test_shape_single_head(self, embedder):
        embeds = torch.randn(2, 8, 256)
        out = embedder.unembed(embeds, pred_head_index=0)
        # squeezed mtp: (batch=2, time=8, na=6, 2)
        assert out.shape == (2, 8, 6, 2)

    def test_shape_single_pred_model(self, embedder_single_pred):
        embeds = torch.randn(2, 8, 256)
        out = embedder_single_pred.unembed(embeds)
        # mtp=1, auto-squeezed: (2, 8, 6, 2)
        assert out.shape == (2, 8, 6, 2)

    def test_1d_input(self, embedder):
        embeds = torch.randn(256)
        out = embedder.unembed(embeds, pred_head_index=0)
        assert out.shape == (6, 2)

    def test_2d_input(self, embedder):
        embeds = torch.randn(3, 256)
        out = embedder.unembed(embeds, pred_head_index=0)
        assert out.shape == (3, 6, 2)


# ─── Sample ───

class TestSample:
    def test_shape(self, embedder):
        embeds = torch.randn(2, 8, 256)
        action = embedder.sample(embeds, pred_head_index=0)
        assert action.shape == (2, 8, 6)

    def test_temperature_zero(self, embedder):
        """Temperature 0 should give deterministic (mean) actions."""
        embeds = torch.randn(1, 4, 256)
        a1 = embedder.sample(embeds, pred_head_index=0, temperature=0.)
        a2 = embedder.sample(embeds, pred_head_index=0, temperature=0.)
        assert torch.allclose(a1, a2)

    def test_temperature_high(self, embedder):
        """High temperature should give more variance."""
        torch.manual_seed(42)
        embeds = torch.randn(1, 100, 256)
        a_low = embedder.sample(embeds, pred_head_index=0, temperature=0.1)
        torch.manual_seed(42)
        a_high = embedder.sample(embeds, pred_head_index=0, temperature=10.)
        # High temp should have more variance
        assert a_high.std() > a_low.std()


# ─── Log probs ───

class TestLogProbs:
    def test_shape_single_head(self, embedder):
        embeds = torch.randn(2, 8, 256)
        targets = torch.randn(2, 8, 6)
        lp = embedder.log_probs(embeds, targets=targets, pred_head_index=0)
        assert lp.shape == (2, 8, 6)

    def test_shape_all_heads(self, embedder):
        embeds = torch.randn(2, 8, 256)
        targets = torch.randn(4, 2, 8, 6)  # (mtp, b, t, na)
        lp = embedder.log_probs(embeds, targets=targets)
        assert lp.shape == (4, 2, 8, 6)

    def test_negative(self, embedder):
        """Log probs should be negative."""
        embeds = torch.randn(2, 4, 256)
        targets = torch.randn(2, 4, 6)
        lp = embedder.log_probs(embeds, targets=targets, pred_head_index=0)
        assert (lp <= 0).all()

    def test_return_entropies(self, embedder):
        embeds = torch.randn(2, 4, 256)
        targets = torch.randn(2, 4, 6)
        lp, ent = embedder.log_probs(
            embeds, targets=targets, pred_head_index=0, return_entropies=True
        )
        assert lp.shape == (2, 4, 6)
        assert ent.shape == (2, 4, 6)
        assert (ent > 0).all()  # entropy should be positive

    def test_gradient_flow(self, embedder):
        embeds = torch.randn(2, 4, 256, requires_grad=True)
        targets = torch.randn(2, 4, 6)
        lp = embedder.log_probs(embeds, targets=targets, pred_head_index=0)
        lp.sum().backward()
        assert embeds.grad is not None

    def test_consistency_with_sample(self, embedder):
        """Sampled actions should have reasonable log probs."""
        embeds = torch.randn(2, 4, 256)
        action = embedder.sample(embeds, pred_head_index=0, temperature=1.)
        lp = embedder.log_probs(embeds, targets=action, pred_head_index=0)
        # Should be finite
        assert torch.isfinite(lp).all()


# ─── KL divergence ───

class TestKLDiv:
    def test_zero_kl_same_dist(self, embedder):
        params = torch.randn(2, 4, 6, 2)
        kl = embedder.kl_div(params, params)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

    def test_positive_kl(self, embedder):
        p1 = torch.randn(2, 4, 6, 2)
        p2 = torch.randn(2, 4, 6, 2) + 2.  # shift means
        kl = embedder.kl_div(p1, p2)
        assert (kl > 0).all()

    def test_shape(self, embedder):
        p1 = torch.randn(3, 5, 6, 2)
        p2 = torch.randn(3, 5, 6, 2)
        kl = embedder.kl_div(p1, p2)
        assert kl.shape == (3, 5)


# ─── Properties ───

class TestProperties:
    def test_has_actions(self, embedder):
        assert embedder.has_actions

    def test_no_actions(self):
        e = ContinuousActionEmbedder(dim=32, num_actions=0)
        assert not e.has_actions

    def test_embed_unembed_parameters(self, embedder):
        assert len(embedder.embed_parameters()) > 0
        assert len(embedder.unembed_parameters()) > 0
