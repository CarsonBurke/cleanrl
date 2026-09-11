"""Capacity-matched SiTU pre-RMS trunks with relocated norms or residuals."""

import math

import torch
from torch import nn

from cleanrl.shared.host_actor import SITU_GLU_MEAN_SQUARE, init_situglu_branch, situ_glu
from cleanrl.shared.norm_residual import NormResidualTrunk


class SiTUReorgTrunk(NormResidualTrunk):
    """Reuse the baseline's modules, parameters and construction RNG sequence.

    With N = non-affine RMS(eps=1e-5), S = situ_glu, a = sigmoid(gate),
    B(u) = down(S(gate(u), up(u))) and x0 = in_proj(x), each block is:

    baseline:     h = h + a * B(N(h))
    gate_only:    h = h + a * down(S(gate(N(h)), up(h)))
    up_only:      h = h + a * down(S(gate(h), up(N(h))))
    product_norm: h = h + a * down(sqrt(E[S^2]) * N(S(gate(h), up(h))))
    branch_norm:  h = h + a * sqrt(0.5) * N(B(h))
    blend:        h = h + a * (B(N(h)) - h)
    anchor:       h = x0 + a * B(N(h))

    Every variant returns N(h) / sqrt(width), calls exactly one norm per
    block plus the final norm, and preserves all SiTUGLUBranch linears.
    Product normalization changes only the existing non-affine RMS shape.
    Use the factory for the baseline's calibrated orthogonal initialization.
    """

    def __init__(self, in_dim, width=64, n_blocks=3, *, variant="baseline"):
        if variant not in ("baseline", "gate_only", "up_only", "product_norm",
                           "branch_norm", "blend", "anchor"):
            raise ValueError(
                "variant must be baseline, gate_only, up_only, product_norm, "
                "branch_norm, blend or anchor")
        super().__init__(in_dim, width, n_blocks, placement="pre", norm_kind="rms",
                         activation="stiglu")
        self.variant = variant
        if variant == "product_norm":
            for block, norm in zip(self.blocks, self.block_norms):
                norm.normalized_shape = (block.hidden_dim,)

    def forward(self, x):
        if self.variant == "baseline":
            return super().forward(x)
        x0 = self.in_proj(x)
        h = x0
        for block, gate, norm in zip(self.blocks, self.block_gates, self.block_norms):
            if self.variant == "gate_only":
                branch = block.down(situ_glu(block.gate(norm(h)), block.up(h)))
            elif self.variant == "up_only":
                branch = block.down(situ_glu(block.gate(h), block.up(norm(h))))
            elif self.variant == "product_norm":
                product = situ_glu(block.gate(h), block.up(h))
                branch = block.down(norm(product) * math.sqrt(SITU_GLU_MEAN_SQUARE))
            elif self.variant == "branch_norm":
                branch = norm(block(h)) * math.sqrt(0.5)
            else:
                branch = block(norm(h))
            if self.variant == "blend":
                h = h + torch.sigmoid(gate) * (branch - h)
            elif self.variant == "anchor":
                h = x0 + torch.sigmoid(gate) * branch
            else:
                h = h + torch.sigmoid(gate) * branch
        return self.final_norm(h) * self.output_scale


def make_situ_reorg_trunk(in_dim, width=64, n_blocks=3, *, variant="baseline"):
    """Preserve the pre/RMS/SiTU factory's parameter values and subsequent RNG.

    All variants construct the same modules in the same order, then apply
    the same initializer in the same order. No throwaway modules, additional
    random draws, learned gains or weight rescalings are introduced.
    """
    trunk = SiTUReorgTrunk(in_dim, width, n_blocks, variant=variant)
    nn.init.orthogonal_(trunk.in_proj.weight, math.sqrt(trunk.width / trunk.in_dim))
    nn.init.zeros_(trunk.in_proj.bias)
    for block in trunk.blocks:
        init_situglu_branch(block, target_out_var=0.5)
    return trunk
