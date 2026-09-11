"""nGPT geometry adapted to observation MLPs, without attention.

Each block uses a 4*width SwiGLU branch, not the narrower SiTU-GLU branches
used by other PPO variants. The homogeneous input coordinate gives zero
observations a learnable direction. Weight projection follows the donor's
post-optimizer normalization; optimizer moments and learned gains are not
projected. Call project_ngpt_weights after EVERY optimizer step.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class NGPTBlock(nn.Module):
    """Unit-sphere SwiGLU update with an absolute, uncapped channel step."""

    def __init__(self, width):
        super().__init__()
        self.width = int(width)
        if self.width < 1:
            raise ValueError("NGPTBlock width must be positive")
        self.base_scale = self.width ** -0.5
        self.gate = nn.Linear(self.width, 4 * self.width, bias=False)
        self.up = nn.Linear(self.width, 4 * self.width, bias=False)
        self.down = nn.Linear(4 * self.width, self.width, bias=False)
        self.suv = nn.Parameter(torch.ones(2, 4 * self.width))
        self.alpha = nn.Parameter(torch.full((self.width,), self.base_scale))
        for projection in (self.gate, self.up, self.down):
            nn.init.normal_(projection.weight, std=self.base_scale)
        project_ngpt_weights(self)

    def forward(self, h):
        h = F.normalize(h, p=2, dim=-1, eps=1e-12)
        up = self.up(h) * self.suv[0] * math.sqrt(self.width)
        gate = self.gate(h) * self.suv[1] * math.sqrt(self.width)
        branch = F.normalize(self.down(up * F.silu(gate)), p=2, dim=-1, eps=1e-12)
        alpha = (self.alpha * (0.05 / self.base_scale)).abs()
        return F.normalize(h + alpha * (branch - h), p=2, dim=-1, eps=1e-12)


class NGPTTrunk(nn.Module):
    """Homogeneous observation projection followed by spherical MLP blocks."""

    def __init__(self, in_dim, width=64, n_blocks=3):
        super().__init__()
        self.in_dim = int(in_dim)
        self.width = int(width)
        self.n_blocks = int(n_blocks)
        if min(self.in_dim, self.width, self.n_blocks) < 1:
            raise ValueError("NGPTTrunk dimensions and block count must be positive")
        self.in_proj = nn.Linear(self.in_dim + 1, self.width, bias=False)
        # Initialize the homogeneous coordinate along with every other column.
        nn.init.normal_(self.in_proj.weight, std=self.width ** -0.5)
        with torch.no_grad():
            self.in_proj.weight.copy_(F.normalize(self.in_proj.weight, p=2, dim=1, eps=1e-12))
        self.blocks = nn.ModuleList(NGPTBlock(self.width) for _ in range(self.n_blocks))

    def forward(self, x):
        homogeneous = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        h = F.normalize(self.in_proj(homogeneous), p=2, dim=-1, eps=1e-12)
        for block in self.blocks:
            h = block(h)
        return h


class NGPTHead(nn.Linear):
    """Unit-row readout with a learned, signed output scale."""

    def __init__(self, width, out_features):
        super().__init__(width, out_features, bias=False)
        self.base_scale = width ** -0.5
        self.scale = nn.Parameter(torch.full((out_features,), self.base_scale))
        nn.init.normal_(self.weight, std=self.base_scale)
        project_ngpt_weights(self)

    def forward(self, input):
        return F.linear(input, self.weight) * (self.scale / self.base_scale)


@torch.no_grad()
def project_ngpt_weights(module):
    """Project only nGPT matrices in-place; leave gains and moments untouched.

    Input, gate, up, and head matrices have unit rows. Down projections have
    unit columns, including their rectangular 4*width expanded dimension.
    Traversal is static under torch.compile and performs no host tensor reads.
    """
    for child in module.modules():
        if isinstance(child, NGPTTrunk):
            child.in_proj.weight.copy_(F.normalize(child.in_proj.weight, p=2, dim=1, eps=1e-12))
        elif isinstance(child, NGPTBlock):
            child.gate.weight.copy_(F.normalize(child.gate.weight, p=2, dim=1, eps=1e-12))
            child.up.weight.copy_(F.normalize(child.up.weight, p=2, dim=1, eps=1e-12))
            child.down.weight.copy_(F.normalize(child.down.weight, p=2, dim=0, eps=1e-12))
        elif isinstance(child, NGPTHead):
            child.weight.copy_(F.normalize(child.weight, p=2, dim=1, eps=1e-12))
