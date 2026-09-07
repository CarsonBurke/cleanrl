"""Single-linear pre-RMS stages with optional gated identity residuals."""

import math

import torch
from torch import nn

from cleanrl.shared.host_actor import LeakyReluSq
from cleanrl.shared.norm_residual import NormResidualTrunk


class PreRMSStage(nn.Module):
    """B(u) = LeakyReluSq(Wu + b) * sqrt(0.25 / 6.375).

    At Gaussian preactivation variance two, 6.375 is the activation's second
    moment. Six stages target the same aggregate branch second-moment budget
    as three two-linear branches of second moment 0.5. These are initialization
    targets, not guarantees for arbitrary normalized observation distributions.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)
        self.lin = nn.Linear(self.dim, self.dim)
        self.act = LeakyReluSq()
        self.output_scale = math.sqrt(0.25 / 6.375)

    def forward(self, x):
        return self.act(self.lin(x)) * self.output_scale


class PreRMSStageTrunk(NormResidualTrunk):
    """RMS -> Linear -> activation, with no hidden sphere projection.

    Each stage consumes RMS(h) * branch_input_scale. Residual stages add
    sigmoid(g) * B(u), with per-channel g initialized to -1.5; plain stages
    replace h with B(u) and have no gate parameters. Optional learned input RMS
    gains start at one. Final RMS is optional; readout always scales by
    width**-0.5. All RMS norms use eps=1e-5; the final norm is non-affine.
    Use the factory for calibrated orthogonal initialization.
    """

    def __init__(self, in_dim, width=64, n_blocks=6, *, residual=True,
                 learned_input_gain=False, final_norm=True, branch_input_scale=1.0):
        # Do not construct and discard the parent's paired branches.
        nn.Module.__init__(self)
        self.in_dim = int(in_dim)
        self.width = int(width)
        self.n_blocks = int(n_blocks)
        if self.in_dim < 1 or self.width < 1 or self.n_blocks < 1:
            raise ValueError("PreRMSStageTrunk needs positive dimensions and at least one stage")
        self.placement = "pre"
        self.norm_kind = "rms"
        self.activation = "lrelusq"
        self.residual = bool(residual)
        self.learned_input_gain = bool(learned_input_gain)
        self.output_scale = self.width ** -0.5
        self.branch_input_scale = float(branch_input_scale)
        if not math.isfinite(self.branch_input_scale) or self.branch_input_scale <= 0:
            raise ValueError("branch_input_scale must be finite and positive")
        self.in_proj = nn.Linear(self.in_dim, self.width)
        self.blocks = nn.ModuleList(PreRMSStage(self.width) for _ in range(self.n_blocks))
        self.block_gates = nn.ParameterList(
            nn.Parameter(torch.full((self.width,), -1.5))
            for _ in range(self.n_blocks if self.residual else 0)
        )
        self.block_norms = nn.ModuleList(
            nn.RMSNorm(self.width, eps=1e-5, elementwise_affine=self.learned_input_gain)
            for _ in range(self.n_blocks)
        )
        self.input_norm = nn.Identity()
        self.final_norm = (nn.RMSNorm(self.width, eps=1e-5, elementwise_affine=False)
                           if final_norm else nn.Identity())

    def forward(self, x):
        if self.residual:
            return super().forward(x)
        h = self.in_proj(x)
        for block, norm in zip(self.blocks, self.block_norms):
            h = block(norm(h) * self.branch_input_scale)
        return self.final_norm(h) * self.output_scale


def make_pre_rms_stage_trunk(in_dim, width=64, n_blocks=6, *, residual=True,
                             learned_input_gain=False, final_norm=True,
                             branch_input_scale=1.0):
    """Initialize both arms identically: projection sqrt(width/in_dim),
    stage weight sqrt(2)/branch_input_scale, and all linear biases zero.
    """
    trunk = PreRMSStageTrunk(in_dim, width, n_blocks, residual=residual,
                             learned_input_gain=learned_input_gain, final_norm=final_norm,
                             branch_input_scale=branch_input_scale)
    nn.init.orthogonal_(trunk.in_proj.weight, math.sqrt(trunk.width / trunk.in_dim))
    nn.init.zeros_(trunk.in_proj.bias)
    for block in trunk.blocks:
        nn.init.orthogonal_(block.lin.weight, math.sqrt(2.0) / trunk.branch_input_scale)
        nn.init.zeros_(block.lin.bias)
    return trunk
