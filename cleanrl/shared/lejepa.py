"""LeJEPA / LeWM components: SIGReg + a causal action-conditioned latent predictor.

Ported from the LeWorldModel reference implementation (Maes, Le Lidec, Scieur, LeCun,
Balestriero) at ../le-wm/module.py, adapted from pixels to low-dimensional state vectors.

The point of LeJEPA is that a JEPA can be trained with exactly TWO loss terms --
a next-embedding prediction loss and SIGReg, a regularizer pushing the embedding
distribution toward an isotropic Gaussian. SIGReg is what prevents representation
collapse, which is why there is deliberately NO stop-gradient, NO EMA/momentum
teacher, and NO asymmetric architecture anywhere in here. Adding any of them would
re-introduce a second, unanchored timescale -- exactly what this is built to avoid.

Reference details that differ from the paper write-ups, and that matter:
  * The Epps-Pulley quadrature grid is t in [0, 3] with 17 knots, NOT [-5, 5]. The
    integrand is even in t, so the reference folds the negative half into the
    trapezoid weights (2*dt interior, dt at the endpoints).
  * exp(-t^2/2) serves as BOTH the target characteristic function and the quadrature
    weighting window.
  * Projections are resampled on every call.
  * Projections are NOT standardized before the test. Classical Epps-Pulley
    standardizes by the sample mean/sd, which makes it a shape-only test; comparing
    raw projections against N(0,1) is what constrains mean, scale AND shape.
"""

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# SIGReg
# ---------------------------------------------------------------------------


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer (single-GPU).

    Cramer-Wold: a multivariate distribution is determined by its 1-D marginals, so
    testing many random 1-D projections against N(0,1) suffices. Per direction we
    compute the Epps-Pulley statistic n * integral |phi_n(t) - exp(-t^2/2)|^2 w(t) dt
    by quadrature -- the quadrature form is what makes this O(N*K); the closed form
    carries an O(N^2) pairwise term.

    INPUT LAYOUT IS LOAD-BEARING. `forward` expects (T, B, D) and reduces the
    empirical characteristic function over dim -3, scaling by `proj.size(-2)`. Both
    resolve to the BATCH only in that layout. Passing (B, T, D) silently averages the
    CF over T samples (typically 4), which makes the statistic noise, disables the
    collapse protection this whole design rests on, and still logs a plausible
    decreasing number. Callers holding (B, T, D) must `.transpose(0, 1)` first.

    `proj_chunk` splits the projection directions. The statistic is a mean over
    directions, so chunking is numerically exact -- it only bounds the (T, B, num_proj,
    knots) intermediate, which is 2.3 GB at B=8192 before autograd saves.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024, proj_chunk: int = 256):
        super().__init__()
        self.num_proj = num_proj
        self.proj_chunk = proj_chunk or num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt  # fold the negative half of the even integrand
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def _statistic(self, proj: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        """proj: (T, B, D); directions: (D, m) -> (T, m)."""
        x_t = (proj @ directions).unsqueeze(-1) * self.t  # (T, B, m, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return (err @ self.weights) * proj.size(-2)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """proj: (T, B, D) -> scalar, averaged over projections and time."""
        directions = torch.randn(
            proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype
        )
        directions = directions / directions.norm(p=2, dim=0)
        total = proj.new_zeros(())
        for chunk in directions.split(self.proj_chunk, dim=1):
            total = total + self._statistic(proj, chunk).sum()
        return total / (self.num_proj * proj.size(0))


# ---------------------------------------------------------------------------
# Transformer stack (causal, AdaLN-zero conditioned)
# ---------------------------------------------------------------------------


def modulate(x, shift, scale):
    """AdaLN-zero modulation."""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking.

    `is_causal` is kept a module-level constant rather than a forward argument: a
    Python bool argument becomes a dynamo guard and costs an extra compile.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        """x: (B, T, D)"""
        b, t, _ = x.shape
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        q, k, v = (
            proj.view(b, t, self.heads, -1).transpose(1, 2)
            for proj in self.to_qkv(x).chunk(3, dim=-1)
        )
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
        return self.to_out(out.transpose(1, 2).reshape(b, t, -1))


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning (actions modulate, not concat)."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )

    def forward(self, x, c):
        x = self.input_proj(x)
        c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)
        return self.output_proj(self.norm(x))


class ARPredictor(nn.Module):
    """Causal autoregressive next-embedding predictor, conditioned on actions."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,   # reference config uses 0.1; zero here so the prediction TARGET
        #                    is deterministic (it is attached, not a stop-grad teacher) and
        #                    so no RNG enters the compiled graph. It also makes it safe that
        #                    train()/eval() is never toggled around the inference-time
        #                    encoder calls that build phi.
    ):
        super().__init__()
        # DELIBERATE DEVIATION from the reference, which uses randn(...) at std 1. That is
        # tuned for a ViT-tiny at embed_dim=192; here input_dim=32 and SIGReg is actively
        # driving the embedding toward N(0, I), so a std-1 positional offset would be as
        # large as the entire signal it is supposed to tag. Scaled to 0.02 (the standard
        # ViT init) so position is a nudge, not the dominant component.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
        )

    def forward(self, x, c, pos_offset=0):
        """x: (B, T, d) embeddings; c: (B, T, d) action embeddings.

        pos_offset shifts which positional embeddings the tokens receive, for
        autoregressive rollout: at round k the sequence stands for frames k..L-1, so its
        tokens must be tagged k..L-1 rather than restarting at 0. Without it a latent
        standing for frame f is tagged f on the first round and f-k on round k, so rolled
        predictions are re-injected at positions their teacher-forced counterparts never
        occupied. Defaults to 0, which is the single-pass behaviour.
        """
        x = x + self.pos_embedding[:, pos_offset : pos_offset + x.size(1)]
        return self.transformer(self.dropout(x), c)


# ---------------------------------------------------------------------------
# MLPs
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Simple MLP.

    Note the norm is LayerNorm, not the reference's BatchNorm1d: BN mutates running
    stats in place (a CUDA-graph hazard under reduce-overhead) and imposes a
    batch-level constraint that is redundant-to-conflicting with SIGReg, which is
    already doing exactly that job.
    """

    def __init__(self, input_dim, hidden_dim, output_dim=None, act_fn=nn.GELU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        return self.net(x)


class StateEncoder(nn.Module):
    """Observation -> embedding. A couple of MLPs (backbone + projector)."""

    def __init__(self, obs_dim, emb_dim, hidden_dim):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_dim, hidden_dim)
        self.projector = MLP(hidden_dim, hidden_dim, emb_dim)

    def forward(self, x):
        return self.projector(F.gelu(self.backbone(x)))


class CausalTransformer(nn.Module):
    """Unconditional causal transformer: a plain pre-norm residual stack.

    Deliberately NOT `Transformer`, which is built from `ConditionalBlock` (AdaLN-zero).
    Driving that with c=0 would be a trap: every gate is `gate * branch(...)` with the
    modulation Linear zero-initialised in BOTH weight and bias, so with a constant-zero
    conditioning signal the block is an EXACT identity at init and can only escape through
    the modulation biases. It would train -- slowly, through a path that exists to carry
    conditioning that is not there -- and log an entirely plausible loss. An encoder that
    must not be conditioned needs a block with no conditioning pathway at all.
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = x + attn(x)      # Attention is causal (is_causal=True) and pre-norms internally
            x = x + ff(x)
        return self.norm(x)


class HistoryEncoder(nn.Module):
    """s_{t-H+1..t} -> e_t via a causal transformer over the history window.

    Input is TIME-ORDERED: position 0 is the oldest frame, position H-1 is s_t, and e_t is
    read from the LAST position, so causal masking means e_t sees exactly the H frames up
    to and including s_t and nothing after.

    NO ACTION CONDITIONING, on purpose. If e_t carried a_t then phi_t = [e, s, a, a*a, 1]
    would hold the action twice, and worse, psi -- which reads state features only -- would
    be regressing a target that depends on the action actually taken, turning V = w_r.psi
    into an action-dependent baseline and BIASING the policy gradient. Past actions would
    be safe (they are part of the history), but omitting actions entirely is strictly safer
    and costs nothing here: the observation already determines the dynamics.
    """

    def __init__(self, obs_dim, emb_dim, hidden_dim, hist_len, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.frame = MLP(obs_dim, hidden_dim, emb_dim)
        # 0.02 for the same reason ARPredictor uses it: SIGReg drives e toward N(0, I), so a
        # std-1 positional offset would be as large as the signal it is meant to tag.
        self.pos_embedding = nn.Parameter(torch.randn(1, hist_len, emb_dim) * 0.02)
        self.transformer = CausalTransformer(emb_dim, depth, heads, dim_head, mlp_dim)
        self.head = MLP(emb_dim, hidden_dim, emb_dim)

    def forward(self, x):
        """x: (N, H, obs) -> (N, emb). Position H-1 must be the CURRENT observation."""
        z = self.frame(x) + self.pos_embedding[:, : x.size(1)]
        return self.head(self.transformer(z)[:, -1])


class ActionEncoder(nn.Module):
    """Action vector -> embedding, matching the reference Embedder's role."""

    def __init__(self, action_dim, emb_dim, mlp_scale=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# torch.compile helpers (pattern from ppo_continuous_action_ebp_bpc_v1_2.py)
# ---------------------------------------------------------------------------


def clone_graph_output(output):
    """Clone tensors returned by a cudagraph-replayed module.

    CUDA graph trees reuse the output buffers on the next replay, so anything
    retained across calls (rollout buffers, values feeding GAE) must be copied out.
    """
    if isinstance(output, torch.Tensor):
        return output.clone()
    if isinstance(output, tuple):
        return tuple(clone_graph_output(item) for item in output)
    if isinstance(output, list):
        return [clone_graph_output(item) for item in output]
    if isinstance(output, dict):
        return {key: clone_graph_output(value) for key, value in output.items()}
    return output


class CompiledModule(nn.Module):
    """Wrap a module with torch.compile, optionally with CUDA graphs.

    cudagraphs=False is mandatory in two situations, both hit by this codebase:

    1. Anything whose backward runs twice under retain_graph (the actor/critic path):
       the graph pool outputs get mutated in place by clip_grad_norm_ and the second
       backward replays over live refs, which either raises or forces a re-record every
       minibatch.
    2. TWO OR MORE cudagraph-wrapped modules chained inside ONE forward whose backward
       runs later. Every forward calls cudagraph_mark_step_begin(), which advances the
       graph-tree generation and invalidates the still-pending backward of the modules
       called earlier in that same forward. clone_graph_output() does NOT save you here:
       it clones the forward OUTPUTS, but the intermediates saved for backward still
       live in the invalidated pool. This raises "accessing tensor output of CUDAGraphs
       that has been overwritten by a subsequent run" on the FIRST backward, and
       torch._dynamo.config.suppress_errors does NOT catch it -- that only suppresses
       dynamo compile-time errors, and this is an inductor runtime error surfaced
       through Tensor.backward(). Wrap the OUTERMOST module instead, or use
       cudagraphs=False.
    """

    def __init__(self, module, mode="reduce-overhead", cudagraphs=True):
        super().__init__()
        self._orig_mod = module
        self.cudagraphs = cudagraphs
        if cudagraphs:
            self._compiled_forward = torch.compile(module.forward, mode=mode, dynamic=False)
        else:
            self._compiled_forward = torch.compile(
                module.forward, dynamic=False, options={"triton.cudagraphs": False}
            )

    def forward(self, *args, **kwargs):
        if self.cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        output = self._compiled_forward(*args, **kwargs)
        if self.cudagraphs:
            output = clone_graph_output(output)
        return output
