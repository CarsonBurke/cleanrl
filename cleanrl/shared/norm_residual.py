"""Identity residual trunks with non-affine LayerNorm or RMSNorm, not a sphere."""

import math

import torch
from torch import nn

from cleanrl.shared.host_actor import LReluSqPair, SiTUGLUBranch, init_situglu_branch


class NormResidualTrunk(nn.Module):
    """Three blocks by default, with per-channel sigmoid gates and no dense skips.

    Pre-norm: h = in_proj(x); h += sigmoid(g) * B(N(h)); return N(h).
    Post-norm: h = N(in_proj(x)); h = N(h + sigmoid(g) * B(h)).
    Each N acts over the last axis, with no learned affine and eps=1e-5.
    LayerNorm centers and uses population variance; RMSNorm uses mean square.
    Normalized streams have RMS approximately one. A fixed readout multiplier
    defaults to width**-0.5, matching sphere head inputs without shrinking
    head initialization (which would change Adam's relative update scale).
    This calibration is applied only after the residual stack; it is not a
    branch/stream sphere projection. Set output_scale=1.0 for unit-RMS output.
    The constructor is init-free apart from gates; use the factory for training.
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
        if placement not in ("pre", "post"):
            raise ValueError("placement must be 'pre' or 'post'")
        if norm_kind not in ("layer", "rms"):
            raise ValueError("norm_kind must be 'layer' or 'rms'")
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
        norm_cls = nn.LayerNorm if norm_kind == "layer" else nn.RMSNorm
        self.block_norms = nn.ModuleList(
            norm_cls(self.width, eps=1e-5, elementwise_affine=False)
            for _ in range(self.n_blocks)
        )
        self.input_norm = (norm_cls(self.width, eps=1e-5, elementwise_affine=False)
                           if placement == "post" else nn.Identity())
        self.final_norm = (norm_cls(self.width, eps=1e-5, elementwise_affine=False)
                           if placement == "pre" else nn.Identity())

    def forward(self, x):
        h = self.in_proj(x)
        if self.placement == "pre":
            for block, gate, norm in zip(self.blocks, self.block_gates, self.block_norms):
                h = h + torch.sigmoid(gate) * block(norm(h) * self.branch_input_scale)
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
    two and branch outputs target second moment 0.5. The LeakyReluSq moment
    6.375 is E[leaky_relu(z, 0.5)^4] for Gaussian z of variance two; SiTU uses
    the existing shape-aware initializer. These are distributional targets,
    not guarantees for an arbitrary normalized observation distribution.
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
