"""Tests for d4_wm.transformer — attention, feedforward, axial transformer."""
import pytest
import torch

from cleanrl.d4_wm.transformer import (
    Rotary1D, apply_rotations, MultiHeadRMSNorm,
    Attention, SwiGLUFeedforward, GRULayer, AttentionResidual,
    AxialSpaceTimeTransformer, _naive_attend, get_attend_fn,
)


# ─── Rotary Embeddings ───

class TestRotary1D:
    def test_shape(self):
        rot = Rotary1D(dim_head=32)
        freqs = rot(seq_len=10)
        assert freqs.shape == (10, 32)

    def test_offset(self):
        rot = Rotary1D(dim_head=32)
        f0 = rot(seq_len=5, offset=0)
        f5 = rot(seq_len=5, offset=5)
        f10 = rot(seq_len=10)
        # f5 should match f10[5:10]
        assert torch.allclose(f5, f10[5:10], atol=1e-5)


class TestApplyRotations:
    def test_preserves_norm(self):
        """Rotation should preserve vector norms."""
        rot = Rotary1D(dim_head=32)
        freqs = rot(seq_len=8)
        t = torch.randn(2, 4, 8, 32)
        rotated = apply_rotations(freqs, t)
        orig_norms = t.norm(dim=-1)
        rot_norms = rotated.norm(dim=-1)
        assert torch.allclose(orig_norms, rot_norms, atol=1e-4)


# ─── MultiHeadRMSNorm ───

class TestMultiHeadRMSNorm:
    def test_shape(self):
        norm = MultiHeadRMSNorm(dim_head=32, heads=4)
        x = torch.randn(2, 4, 8, 32)
        out = norm(x)
        assert out.shape == (2, 4, 8, 32)

    def test_normalization(self):
        norm = MultiHeadRMSNorm(dim_head=16, heads=2)
        x = torch.randn(1, 2, 4, 16) * 100  # large values
        out = norm(x)
        # Output should be scaled but not as extreme
        assert out.abs().max() < 1000


# ─── Naive Attend ───

class TestNaiveAttend:
    def test_basic_shape(self):
        q = torch.randn(2, 4, 8, 32)
        k = torch.randn(2, 4, 8, 32)
        v = torch.randn(2, 4, 8, 32)
        out = _naive_attend(q, k, v)
        assert out.shape == (2, 4, 8, 32)

    def test_causal_masking(self):
        """Causal attention: position i should not attend to j > i."""
        dim_head = 8
        q = torch.randn(1, 1, 4, dim_head)
        k = torch.randn(1, 1, 4, dim_head)
        v = torch.randn(1, 1, 4, dim_head)

        out = _naive_attend(q, k, v, causal=True, softclamp_value=None)
        assert out.shape == (1, 1, 4, dim_head)

    def test_gqa(self):
        """Grouped query attention: more query heads than KV heads."""
        q = torch.randn(2, 8, 4, 32)   # 8 query heads
        k = torch.randn(2, 4, 4, 32)   # 4 KV heads
        v = torch.randn(2, 4, 4, 32)
        out = _naive_attend(q, k, v)
        assert out.shape == (2, 8, 4, 32)

    def test_mask(self):
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        mask = torch.ones(4, 4, dtype=torch.bool)
        mask[0, 3] = False  # position 0 cannot attend to position 3
        out = _naive_attend(q, k, v, mask=mask, softclamp_value=None)
        assert out.shape == (1, 1, 4, 8)


class TestGetAttendFn:
    def test_causal(self):
        fn = get_attend_fn(causal=True, seq_len=8, device='cpu')
        q = torch.randn(1, 2, 8, 16)
        k = torch.randn(1, 2, 8, 16)
        v = torch.randn(1, 2, 8, 16)
        out = fn(q, k, v)
        assert out.shape == (1, 2, 8, 16)

    def test_special_tokens(self):
        fn = get_attend_fn(
            causal=False, seq_len=10, num_special_tokens=2, device='cpu'
        )
        q = torch.randn(1, 2, 10, 16)
        k = torch.randn(1, 2, 10, 16)
        v = torch.randn(1, 2, 10, 16)
        out = fn(q, k, v)
        assert out.shape == (1, 2, 10, 16)


# ─── Attention ───

class TestAttention:
    @pytest.fixture
    def attn(self):
        return Attention(dim=64, heads=4, dim_head=16, value_residual=False)

    def test_basic(self, attn):
        x = torch.randn(2, 8, 64)
        out = attn(x)
        assert out.shape == (2, 8, 64)

    def test_residual(self):
        """Output should be different from input (attention modifies)."""
        attn = Attention(dim=64, heads=4, dim_head=16, value_residual=False)
        x = torch.randn(2, 8, 64)
        out = attn(x)
        assert not torch.allclose(out, x)

    def test_with_context(self, attn):
        x = torch.randn(2, 4, 64)
        ctx = torch.randn(2, 8, 64)
        out = attn(x, context=ctx)
        assert out.shape == (2, 4, 64)

    def test_return_intermediates(self, attn):
        x = torch.randn(2, 8, 64)
        out, inter = attn(x, return_intermediates=True)
        assert out.shape == (2, 8, 64)
        assert inter.next_kv_cache is not None
        assert inter.normed_inputs is not None

    def test_gradient_flow(self, attn):
        x = torch.randn(2, 4, 64, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None

    def test_4d_input(self):
        """Attention should handle (batch, time, space, dim) by flattening."""
        attn = Attention(dim=64, heads=4, dim_head=16, value_residual=False)
        x = torch.randn(2, 3, 8, 64)
        out = attn(x)
        assert out.shape == (2, 3, 8, 64)

    def test_with_rotary(self):
        attn = Attention(dim=64, heads=4, dim_head=16, value_residual=False)
        rot = Rotary1D(dim_head=16)
        x = torch.randn(2, 8, 64)
        freqs = rot(8)
        out = attn(x, rotary_pos_emb=freqs)
        assert out.shape == (2, 8, 64)


# ─── SwiGLU Feedforward ───

class TestSwiGLUFeedforward:
    def test_shape(self):
        ff = SwiGLUFeedforward(dim=64)
        x = torch.randn(2, 8, 64)
        out = ff(x)
        assert out.shape == (2, 8, 64)

    def test_gradient(self):
        ff = SwiGLUFeedforward(dim=32)
        x = torch.randn(1, 4, 32, requires_grad=True)
        out = ff(x)
        out.sum().backward()
        assert x.grad is not None


# ─── GRULayer ───

class TestGRULayer:
    def test_shape(self):
        gru = GRULayer(dim=64, dim_out=64)
        x = torch.randn(2, 8, 64)
        out, hiddens = gru(x)
        assert out.shape == (2, 8, 64)
        assert hiddens.shape == (1, 2, 64)

    def test_stateful(self):
        gru = GRULayer(dim=32, dim_out=32)
        x1 = torch.randn(2, 4, 32)
        out1, h1 = gru(x1)
        x2 = torch.randn(2, 4, 32)
        out2, h2 = gru(x2, prev_hiddens=h1)
        # Outputs should differ due to hidden state
        assert not torch.allclose(out1, out2)


# ─── AttentionResidual ───

class TestAttentionResidual:
    def test_shape(self):
        ar = AttentionResidual(dim=64, heads=2, dim_head=16)
        x = torch.randn(2, 8, 64)
        hiddens = [torch.randn(2, 8, 64) for _ in range(3)]
        out = ar(x, hiddens=hiddens)
        assert out.shape == (2, 8, 64)


# ─── AxialSpaceTimeTransformer ───

class TestAxialSpaceTimeTransformer:
    @pytest.fixture
    def small_transformer(self):
        return AxialSpaceTimeTransformer(
            dim=64, depth=4, attn_heads=4, attn_dim_head=16,
            time_block_every=4, num_special_spatial_tokens=1,
            value_residual=True, rnn_time=True,
        )

    def test_output_shape(self, small_transformer):
        tokens = torch.randn(2, 4, 8, 64)
        out = small_transformer(tokens)
        assert out.shape == (2, 4, 8, 64)

    def test_return_intermediates(self, small_transformer):
        tokens = torch.randn(2, 4, 8, 64)
        out, inter = small_transformer(tokens, return_intermediates=True)
        assert out.shape == (2, 4, 8, 64)
        assert inter.layer_hiddens is not None
        assert len(inter.layer_hiddens) > 0

    def test_gradient_flow(self, small_transformer):
        tokens = torch.randn(2, 3, 6, 64, requires_grad=True)
        out = small_transformer(tokens)
        out.sum().backward()
        assert tokens.grad is not None

    def test_single_timestep(self, small_transformer):
        tokens = torch.randn(2, 1, 6, 64)
        out = small_transformer(tokens)
        assert out.shape == (2, 1, 6, 64)

    def test_depth_8(self):
        """Test with deeper transformer (2 temporal blocks)."""
        transformer = AxialSpaceTimeTransformer(
            dim=32, depth=8, attn_heads=2, attn_dim_head=16,
            time_block_every=4, num_special_spatial_tokens=1,
        )
        tokens = torch.randn(1, 4, 5, 32)
        out = transformer(tokens)
        assert out.shape == (1, 4, 5, 32)

    def test_no_rnn(self):
        transformer = AxialSpaceTimeTransformer(
            dim=32, depth=4, attn_heads=2, attn_dim_head=16,
            time_block_every=4, rnn_time=False,
        )
        tokens = torch.randn(1, 4, 5, 32)
        out = transformer(tokens)
        assert out.shape == (1, 4, 5, 32)
