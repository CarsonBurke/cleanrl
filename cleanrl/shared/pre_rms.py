"""Pre-RMS paired trunks with optional residuals and branch normalization."""

import math

from torch import nn

from cleanrl.shared.host_actor import LReluSqPair
from cleanrl.shared.norm_residual import NormResidualTrunk


class PreRMSPair(LReluSqPair):
    """Reuse a pair's linears, inserting one optional non-affine inner RMS.

    Wrapping the already-created pair consumes no random draws, preserving the
    baseline trunk's parameter construction and orthogonal initialization order.
    """

    def __init__(self, pair, norm_position):
        nn.Module.__init__(self)
        self.dim = pair.dim
        self.lin1 = pair.lin1
        self.act = pair.act
        self.lin2 = pair.lin2
        self.norm_position = norm_position
        self.norm_scale = {
            "input": 1.0,
            "preact": math.sqrt(2.0),
            "postact": math.sqrt(6.375),
            "branch": math.sqrt(0.5),
        }[norm_position]
        self.inner_norm = (nn.Identity() if norm_position == "input" else
                           nn.RMSNorm(self.dim, eps=1e-5, elementwise_affine=False))

    def forward(self, x):
        h = self.lin1(x)
        if self.norm_position == "preact":
            h = self.inner_norm(h) * self.norm_scale
        h = self.act(h)
        if self.norm_position == "postact":
            h = self.inner_norm(h) * self.norm_scale
        h = self.lin2(h)
        if self.norm_position == "branch":
            h = self.inner_norm(h) * self.norm_scale
        return h


class PreRMSTrunk(NormResidualTrunk):
    """Pre-RMS paired blocks, with optional gated residuals and no dense skips.

    Every block uses u = RMS(h) * branch_input_scale. With residual=True,
    h += sigmoid(g) * B(u), with gates initialized to -1.5; otherwise h = B(u)
    and there are no gates. Only the input RMS may learn a channel gain,
    initialized to one. For f = squared leaky ReLU (negative slope 0.5):

    input:   B(u) = W2 f(W1 u + b1) + b2
    preact:  B(u) = W2 f(sqrt(2) RMS(W1 u + b1)) + b2
    postact: B(u) = W2 [sqrt(6.375) RMS(f(W1 u + b1))] + b2
    branch:  B(u) = sqrt(0.5) RMS(W2 f(W1 u + b1) + b2)

    Inner and final RMS norms are non-affine with eps=1e-5. The output is
    RMS(h) / sqrt(width), or h / sqrt(width) when final_norm=False. Use the
    factory for calibrated orthogonal initialization.
    """

    def __init__(self, in_dim, width=64, n_blocks=3, *, norm_position="input",
                 learned_input_gain=False, final_norm=True, branch_input_scale=1.0,
                 residual=True):
        if norm_position not in ("input", "preact", "postact", "branch"):
            raise ValueError("norm_position must be 'input', 'preact', 'postact' or 'branch'")
        if residual:
            super().__init__(in_dim, width, n_blocks, placement="pre", norm_kind="rms",
                             activation="lrelusq", branch_input_scale=branch_input_scale)
        else:
            nn.Module.__init__(self)
            self.in_dim = int(in_dim)
            self.width = int(width)
            self.n_blocks = int(n_blocks)
            if self.in_dim < 1 or self.width < 1 or self.n_blocks < 1:
                raise ValueError("PreRMSTrunk needs positive dimensions and at least one block")
            self.placement = "pre"
            self.norm_kind = "rms"
            self.activation = "lrelusq"
            self.output_scale = self.width ** -0.5
            self.branch_input_scale = float(branch_input_scale)
            if not math.isfinite(self.branch_input_scale) or self.branch_input_scale <= 0:
                raise ValueError("branch_input_scale must be finite and positive")
            self.in_proj = nn.Linear(self.in_dim, self.width)
            self.blocks = nn.ModuleList(LReluSqPair(self.width) for _ in range(self.n_blocks))
            self.block_gates = nn.ParameterList()
            self.block_norms = nn.ModuleList(
                nn.RMSNorm(self.width, eps=1e-5, elementwise_affine=False)
                for _ in range(self.n_blocks)
            )
            self.input_norm = nn.Identity()
            self.final_norm = nn.RMSNorm(self.width, eps=1e-5, elementwise_affine=False)
        self.residual = bool(residual)
        self.norm_position = norm_position
        self.learned_input_gain = bool(learned_input_gain)
        self.blocks = nn.ModuleList(PreRMSPair(pair, norm_position) for pair in self.blocks)
        if self.learned_input_gain:
            self.block_norms = nn.ModuleList(
                nn.RMSNorm(self.width, eps=1e-5, elementwise_affine=True)
                for _ in range(self.n_blocks)
            )
        if not final_norm:
            self.final_norm = nn.Identity()

    def forward(self, x):
        if self.residual:
            return super().forward(x)
        h = self.in_proj(x)
        for block, norm in zip(self.blocks, self.block_norms):
            h = block(norm(h) * self.branch_input_scale)
        return self.final_norm(h) * self.output_scale


def make_pre_rms_trunk(in_dim, width=64, n_blocks=3, *, norm_position="input",
                       learned_input_gain=False, final_norm=True, branch_input_scale=1.0,
                       residual=True):
    """Initialize exactly like the squared-pair pre-RMS NormResidualTrunk.

    Projection gain is sqrt(width/in_dim); lin1 gain is sqrt(2)/input_scale;
    lin2 gain is sqrt(0.5/6.375); all linear biases are zero. The default
    configuration preserves the baseline's random draws and forward operations.
    """
    trunk = PreRMSTrunk(in_dim, width, n_blocks, norm_position=norm_position,
                        learned_input_gain=learned_input_gain, final_norm=final_norm,
                        branch_input_scale=branch_input_scale, residual=residual)
    nn.init.orthogonal_(trunk.in_proj.weight, math.sqrt(trunk.width / trunk.in_dim))
    nn.init.zeros_(trunk.in_proj.bias)
    for block in trunk.blocks:
        nn.init.orthogonal_(block.lin1.weight, math.sqrt(2.0) / trunk.branch_input_scale)
        nn.init.zeros_(block.lin1.bias)
        nn.init.orthogonal_(block.lin2.weight, math.sqrt(0.5 / 6.375))
        nn.init.zeros_(block.lin2.bias)
    return trunk
