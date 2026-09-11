"""SiTU pre-RMS residuals with separate down direction and positive amplitude."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from cleanrl.shared.host_actor import init_situglu_branch, situ_glu
from cleanrl.shared.norm_residual import NormResidualTrunk


class SiTUFactorizedTrunk(NormResidualTrunk):
    """Reuse baseline parameters, interpreting each gate as a raw amplitude.

    The effective down matrix is diag(softplus(g)) * row_normalize(Wdown).
    Positive down-row rescaling leaves the function unchanged outside the
    normalization epsilon region. No new layers, parameters or buffers are
    introduced. Use the factory to preserve the baseline's initial function.
    """

    def __init__(self, in_dim, width=64, n_blocks=3):
        super().__init__(in_dim, width, n_blocks, placement="pre", norm_kind="rms",
                         activation="stiglu")

    def forward(self, x):
        h = self.in_proj(x)
        for block, raw_amplitude, norm in zip(self.blocks, self.block_gates, self.block_norms):
            normalized = norm(h)
            product = situ_glu(block.gate(normalized), block.up(normalized))
            directional = F.linear(
                product, F.normalize(block.down.weight, p=2, dim=1, eps=1e-12))
            h = h + F.softplus(raw_amplitude) * directional
        return self.final_norm(h) * self.output_scale


def make_situ_factorized_trunk(in_dim, width=64, n_blocks=3):
    """Match baseline construction, RNG and weights; convert existing gates.

    For each initialized down row, softplus(new_g) = sigmoid(old_g) * ||row||.
    Thus the initial effective matrix is unchanged without rescaling weights
    or drawing additional random values. Inverse softplus uses expm1 for
    accuracy at small amplitudes and avoids exponentiating large amplitudes.
    """
    trunk = SiTUFactorizedTrunk(in_dim, width, n_blocks)
    nn.init.orthogonal_(trunk.in_proj.weight, math.sqrt(trunk.width / trunk.in_dim))
    nn.init.zeros_(trunk.in_proj.bias)
    for block in trunk.blocks:
        init_situglu_branch(block, target_out_var=0.5)
    with torch.no_grad():
        for block, raw_amplitude in zip(trunk.blocks, trunk.block_gates):
            amplitude = torch.sigmoid(raw_amplitude) * block.down.weight.norm(p=2, dim=1)
            raw_amplitude.copy_(amplitude + torch.log(-torch.expm1(-amplitude)))
    return trunk
