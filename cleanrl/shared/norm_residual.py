"""Identity residual trunks with moment norms or Peri-only sphere/Derf branches."""

import math

import torch
from torch import nn

from cleanrl.shared.host_actor import LReluSqPair, SiTUGLUBranch, init_situglu_branch, justnorm


class SphereNorm(nn.Module):
    """Unit-L2 branch projection using the shared justnorm epsilon."""

    def forward(self, x):
        return justnorm(x)


class Derf(nn.Module):
    """Channels-last Dynamic erf: erf(alpha*x + shift)*weight + bias.

    Formula and initialization match zlab-princeton/Derf, ViT/dynamic_erf.py:
    https://github.com/zlab-princeton/Derf/blob/2d068e910da7cd1938b3371f7132c42c6ea24e3d/ViT/dynamic_erf.py
    Alpha and shift are learnable scalars, with channel-wise affine parameters;
    this is pointwise, with neither a moment reduction nor an epsilon.
    """

    def __init__(self, width):
        super().__init__()
        self.normalized_shape = (int(width),)
        self.alpha = nn.Parameter(torch.full((1,), 0.5))
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.erf(self.alpha * x + self.shift) * self.weight + self.bias


class NormResidualTrunk(nn.Module):
    """Three blocks by default, with per-channel sigmoid gates and no dense skips.

    Write s = branch_input_scale, r = output_scale and a = sigmoid(g).
    Pre-norm: h = in_proj(x); h += a * B(N(h) * s); return N_final(h) * r.
    Post-norm: h = N_in(in_proj(x)); h = N(h + a * B(h * s)); return h * r.
    Peri-norm: h = in_proj(x); h += a * N_out(B(N(h) * s));
    return N_final(h) * r. Each update repeats once per block. Peri keeps the
    per-channel gate outside its branch output transform and omits the optional
    embedding norm; moment-norm variants use fixed gamma.
    LayerNorm/RMSNorm act over the last axis, with no learned affine and
    eps=1e-5. LayerNorm centers and uses population variance; RMSNorm uses
    mean square. Sphere and Derf are Peri-only alternatives at both branch
    sites, with Identity final norm and no residual-stream projection.
    A fixed readout multiplier defaults to width**-0.5, matching sphere head
    inputs without shrinking head initialization (which would change Adam's
    relative update scale). This calibration applies only after the stack;
    set output_scale=1.0 to omit it.
    The constructor leaves linear initialization at module defaults; use the factory for training.
    """

    def __init__(self, in_dim, width=64, n_blocks=3, *, placement="pre",
                 norm_kind="layer", activation="lrelusq", output_scale=None,
                 branch_input_scale=1.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.width = int(width)
        self.n_blocks = int(n_blocks)
        if self.in_dim < 1 or self.width < 1 or self.n_blocks < 1:
            raise ValueError("NormResidualTrunk needs positive dimensions and at least one block")
        if placement not in ("pre", "post", "peri"):
            raise ValueError("placement must be 'pre', 'post' or 'peri'")
        if norm_kind not in ("layer", "rms", "sphere", "derf"):
            raise ValueError("norm_kind must be 'layer', 'rms', 'sphere' or 'derf'")
        if norm_kind in ("sphere", "derf") and placement != "peri":
            raise ValueError("sphere and derf norms require peri placement")
        if activation not in ("lrelusq", "stiglu"):
            raise ValueError("activation must be 'lrelusq' or 'stiglu'")
        self.placement = placement
        self.norm_kind = norm_kind
        self.activation = activation
        self.output_scale = self.width ** -0.5 if output_scale is None else float(output_scale)
        self.branch_input_scale = float(branch_input_scale)
        if not math.isfinite(self.branch_input_scale) or self.branch_input_scale <= 0:
            raise ValueError("branch_input_scale must be finite and positive")
        self.in_proj = nn.Linear(self.in_dim, self.width)
        self.blocks = nn.ModuleList(
            LReluSqPair(self.width) if activation == "lrelusq"
            else SiTUGLUBranch(self.width, self.width)
            for _ in range(self.n_blocks)
        )
        self.block_gates = nn.ParameterList(
            nn.Parameter(torch.full((self.width,), -1.5)) for _ in range(self.n_blocks)
        )
        def make_norm():
            if norm_kind == "sphere":
                return SphereNorm()
            if norm_kind == "derf":
                return Derf(self.width)
            norm_cls = nn.LayerNorm if norm_kind == "layer" else nn.RMSNorm
            return norm_cls(self.width, eps=1e-5, elementwise_affine=False)

        self.block_norms = nn.ModuleList(make_norm() for _ in range(self.n_blocks))
        if placement == "peri":
            self.block_output_norms = nn.ModuleList(make_norm() for _ in range(self.n_blocks))
        self.input_norm = make_norm() if placement == "post" else nn.Identity()
        self.final_norm = (make_norm()
                           if placement in ("pre", "peri") and norm_kind in ("layer", "rms")
                           else nn.Identity())

    def forward(self, x):
        h = self.in_proj(x)
        if self.placement == "pre":
            for block, gate, norm in zip(self.blocks, self.block_gates, self.block_norms):
                h = h + torch.sigmoid(gate) * block(norm(h) * self.branch_input_scale)
            h = self.final_norm(h)
        elif self.placement == "peri":
            for block, gate, norm, output_norm in zip(
                self.blocks, self.block_gates, self.block_norms, self.block_output_norms
            ):
                branch = block(norm(h) * self.branch_input_scale)
                h = h + torch.sigmoid(gate) * output_norm(branch)
            h = self.final_norm(h)
        else:
            h = self.input_norm(h)
            for block, gate, norm in zip(self.blocks, self.block_gates, self.block_norms):
                h = norm(h + torch.sigmoid(gate) * block(h * self.branch_input_scale))
        return h * self.output_scale


def make_norm_residual_trunk(in_dim, width=64, n_blocks=3, *, placement="pre",
                             norm_kind="layer", activation="lrelusq", output_scale=None,
                             branch_input_scale=1.0):
    """Initialize projection gain sqrt(width/in_dim) and matched branch scales.

    At isotropic unit-second-moment input, branch preactivations have variance
    two and raw branch outputs target second moment 0.5. The LeakyReluSq moment
    6.375 is E[leaky_relu(z, 0.5)^4] for Gaussian z of variance two; SiTU uses
    the existing shape-aware initializer. These are distributional targets,
    not guarantees for an arbitrary normalized observation distribution.
    Peri transforms each raw branch output before gating. Norm construction
    consumes no initialization RNG; only Derf adds learned norm parameters.
    Readout calibration defaults to width**-0.5, as in the constructor.
    Dividing branch inputs by a constant and multiplying their input weights
    by that constant preserves the initial function, but changes Adam updates.
    branch_input_scale=width**-0.5 matches sphere input-matrix parameter scales.
    """
    trunk = NormResidualTrunk(in_dim, width, n_blocks, placement=placement,
                              norm_kind=norm_kind, activation=activation,
                              output_scale=output_scale,
                              branch_input_scale=branch_input_scale)
    nn.init.orthogonal_(trunk.in_proj.weight, math.sqrt(trunk.width / trunk.in_dim))
    nn.init.zeros_(trunk.in_proj.bias)
    for block in trunk.blocks:
        if isinstance(block, SiTUGLUBranch):
            init_situglu_branch(block, target_out_var=0.5)
            with torch.no_grad():
                block.gate.weight.div_(trunk.branch_input_scale)
                block.up.weight.div_(trunk.branch_input_scale)
        elif isinstance(block, LReluSqPair):
            nn.init.orthogonal_(block.lin1.weight, math.sqrt(2.0) / trunk.branch_input_scale)
            nn.init.zeros_(block.lin1.bias)
            nn.init.orthogonal_(block.lin2.weight, math.sqrt(0.5 / 6.375))
            nn.init.zeros_(block.lin2.bias)
    return trunk
