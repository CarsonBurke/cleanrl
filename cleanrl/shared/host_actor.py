"""Host-side FP32 mirror of a small MLP policy: rollouts with zero GPU round trips.

STANDARD: new PPO-family versions whose behavior policy is a plain FP32
``nn.Sequential`` of ``Linear``/``Tanh``/``ReLU``/``LeakyReluSq``/``SiTUGLUBranch``
layers MUST act from this mirror during rollouts. Values and old
log-probabilities are then produced by ONE batched device forward over the
uploaded rollout (see ``ppo_continuous_action.py``), which is the same network
and numerics the loss uses, so the first-minibatch ratio is still exactly one.
Policies that cannot be mirrored (autocast/bf16, large nets) use
``rollout_graph.RolloutStepGraph``.

Why
---
A 64-unit MLP over 16 rows costs ~5us in NumPy and ~20us with Beta sampling.
The cheapest possible device path -- one captured graph, one sync -- costs
~50us on an idle GPU but ~400us whenever another process shares the GPU
(context time-slicing), which is the normal state under the mlq queue. With
CPU physics every step needs the action before the next step, so the only way
to make the rollout independent of GPU contention is to not touch the GPU.

Contracts
---------
- ``refresh`` copies the current device parameters into pinned host storage
  (one small D2H per tensor, one event wait). Call it after every optimizer
  update and before the first rollout; the mirror is otherwise stale.
- ``__call__`` takes a float32 ``(num_rows, in)`` array and returns a
  ``(num_rows, out)`` float32 view of internal scratch, overwritten by the
  next call. Row count is fixed at construction (rollout batch = num_envs).
- Host arithmetic is true FP32; the device may use TF32 matmuls. Sampling
  parameters therefore differ from the learner's view by TF32 rounding
  (~1e-3 relative), the same order as the batch-size-dependent kernel
  differences already present between rollout and minibatch forwards.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class LeakyReluSq(nn.Module):
    """f(x) = leaky_relu(x, negative_slope=0.5)^2 = (0.5 x + 0.5 relu(x))^2."""

    def forward(self, x):
        return F.leaky_relu(x, negative_slope=0.5).square()


# E[SiTU(g,u)^2] at sqrt(2) gate/up gain and isotropic unit-variance input
# (Gauss-Hermite quadrature; matches the encoder-value SiTU-GLU branch).
SITU_GLU_MEAN_SQUARE = 1.2630450818573506


def situ_glu(gate, up):
    """Kimi K3 SiTU-GLU product before the down projection."""
    capped_gate = 4.0 * torch.tanh(gate / 4.0)
    capped_up = 25.0 * torch.tanh(up / 25.0)
    return (capped_gate * torch.sigmoid(gate)) * capped_up


class SiTUGLUBranch(nn.Module):
    """SiTU-GLU FFN block: down(situ_glu(gate(x), up(x))).

    Follows the ``tpo_intra_beta_situglu_v5`` trunk layout: stack two branches
    per trunk with the first consuming observations directly, replacing each
    Linear+activation stage. The gated width M=round(2(out_dim+1)/3) is the
    parameter-matched width against two biased out_dim->out_dim linears
    (H=64: 8,256 vs 8,320). The module is init-free; callers apply
    ``layer_init``-style orthogonal init to ``gate``/``up`` (sqrt(2)) and scale
    ``down`` so the block's initial output scale matches the replaced stage.
    """

    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = (max(1, round(2.0 * (out_dim + 1) / 3.0))
                           if hidden_dim is None else int(hidden_dim))
        self.gate = nn.Linear(in_dim, self.hidden_dim, bias=False)
        self.up = nn.Linear(in_dim, self.hidden_dim, bias=False)
        self.down = nn.Linear(self.hidden_dim, out_dim, bias=False)

    def forward(self, x):
        return self.down(situ_glu(self.gate(x), self.up(x)))


def init_situglu_branch(branch, target_out_var=0.5):
    """Shape-aware init putting SiTU at its design point.

    The E[SiTU^2]=SITU_GLU_MEAN_SQUARE constant assumes gate/up preactivation
    variance 2. For a wide layer (hidden_dim <= in_features) orthonormal rows
    give that at gain sqrt(2); for a tall layer the norm is preserved instead,
    so the gain is corrected to sqrt(2*hidden_dim/in_features). The down
    projection then targets ``target_out_var`` output second moment.
    """
    in_features = branch.gate.in_features
    if branch.hidden_dim > in_features:
        gain = np.sqrt(2.0 * branch.hidden_dim / in_features)
    else:
        gain = np.sqrt(2.0)
    torch.nn.init.orthogonal_(branch.gate.weight, gain)
    torch.nn.init.orthogonal_(branch.up.weight, gain)
    down_std = np.sqrt(target_out_var * branch.out_dim / (branch.hidden_dim * SITU_GLU_MEAN_SQUARE))
    torch.nn.init.orthogonal_(branch.down.weight, down_std)
    return branch


class SiTUResTrunk(nn.Module):
    """Two gated SiTU blocks with per-block residuals and a U-net long skip.

    x0 = in_proj(x); h1 = x0 + s(lam1)*B1(x0);
    h2 = h1 + s(lam2)*B2(h1) + s(lam_skip)*x0, with s = sigmoid. The long skip
    re-injects the early-stream snapshot x0 into the late stage, as in
    modded-nanogpt (skip_lambda init -1.5 -> ~0.18); per-block gates use the
    same init so the trunk starts stream-dominated and branches fade in. No
    normalization: plain additive skips only.
    """

    def __init__(self, in_dim, width=64):
        super().__init__()
        self.in_dim = int(in_dim)
        self.width = int(width)
        self.in_proj = nn.Linear(in_dim, width)
        self.block1 = SiTUGLUBranch(width, width)
        self.block2 = SiTUGLUBranch(width, width)
        self.lam1 = nn.Parameter(torch.tensor(-1.5))
        self.lam2 = nn.Parameter(torch.tensor(-1.5))
        self.lam_skip = nn.Parameter(torch.tensor(-1.5))

    def forward(self, x):
        x0 = self.in_proj(x)
        h1 = x0 + torch.sigmoid(self.lam1) * self.block1(x0)
        h2 = h1 + torch.sigmoid(self.lam2) * self.block2(h1) + torch.sigmoid(self.lam_skip) * x0
        return h2


def make_situ_res_trunk(in_dim, width=64, branch_out_var=0.5):
    """Build an init-complete SiTUResTrunk with a unit-variance stream.

    The input projection uses gain sqrt(width/in_dim) so the residual stream
    starts at unit variance; both blocks' gate/up projections then sit exactly
    at SiTU's v=2 design point (gain sqrt(2) suffices, see
    ``init_situglu_branch``), and each block targets ``branch_out_var``
    output second moment. With gates at sigmoid(-1.5) ~= 0.18 the trunk starts
    stream-dominated (output E[x^2] ~= 1.1-1.5, the tanh/lrelusq regime) and
    branch contributions fade in through the learned gates.
    """
    trunk = SiTUResTrunk(in_dim, width)
    torch.nn.init.orthogonal_(trunk.in_proj.weight, np.sqrt(width / in_dim))
    torch.nn.init.zeros_(trunk.in_proj.bias)
    init_situglu_branch(trunk.block1, target_out_var=branch_out_var)
    init_situglu_branch(trunk.block2, target_out_var=branch_out_var)
    return trunk


class SiTUDenseTrunk(nn.Module):
    """SiTU blocks with dense per-channel gated residuals.

    s_0 = in_proj(x); s_k = s_{k-1} + g_k * B_k(s_{k-1}) + sum_{j<k-1} a_{kj} * s_j,
    where every gate is a learned per-channel (per-perceptron) vector passed
    through a sigmoid -- each unit routes its own effective depth instead of
    sharing one layer-wide scalar. Every previous stream feeds every later
    stage (N(N-1)/2 skip gates for N blocks). All gates init at -1.5 (~0.18)
    so the trunk starts stream-dominated; no normalization, plain adds only.
    """

    def __init__(self, in_dim, width=64, n_blocks=3):
        super().__init__()
        self.in_dim = int(in_dim)
        self.width = int(width)
        self.n_blocks = int(n_blocks)
        if self.n_blocks < 1:
            raise ValueError("SiTUDenseTrunk needs at least one block")
        self.in_proj = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList(SiTUGLUBranch(width, width) for _ in range(n_blocks))
        self.block_gates = nn.ParameterList(
            nn.Parameter(torch.full((width,), -1.5)) for _ in range(n_blocks)
        )
        self.skip_gates = nn.ParameterList()
        self.skip_index = []
        for k in range(1, n_blocks + 1):
            for j in range(k - 1):
                self.skip_index.append((k, j))
                self.skip_gates.append(nn.Parameter(torch.full((width,), -1.5)))

    def forward(self, x):
        streams = [self.in_proj(x)]
        skip_pos = 0
        for k, (block, gate) in enumerate(zip(self.blocks, self.block_gates), start=1):
            h = streams[-1] + torch.sigmoid(gate) * block(streams[-1])
            for j in range(k - 1):
                h = h + torch.sigmoid(self.skip_gates[skip_pos]) * streams[j]
                skip_pos += 1
            streams.append(h)
        return streams[-1]


def make_situ_dense_trunk(in_dim, width=64, n_blocks=3, branch_out_var=0.5):
    """Build an init-complete SiTUDenseTrunk with a unit-variance stream.

    Same contract as ``make_situ_res_trunk``: the input projection sets stream
    variance 1 so every block's gate/up preacts sit at SiTU's v=2 design
    point, and each block targets ``branch_out_var`` output second moment.
    """
    trunk = SiTUDenseTrunk(in_dim, width, n_blocks)
    torch.nn.init.orthogonal_(trunk.in_proj.weight, np.sqrt(width / in_dim))
    torch.nn.init.zeros_(trunk.in_proj.bias)
    for block in trunk.blocks:
        init_situglu_branch(block, target_out_var=branch_out_var)
    return trunk


def justnorm(x, eps=1e-12):
    """Project rows onto the unit hypersphere (nGPT-style, no affine)."""
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


class SiTUSphereTrunk(nn.Module):
    """SiTU blocks on a hypersphere (nGPT port, arXiv:2412.13587).

    Every stream lives on the unit sphere: s_0 = justnorm(in_proj(x)), and
    each stage mixes unit-normed branch outputs into the unit-normed stream
    with learned per-channel step sizes, then re-projects:
    s_k = justnorm(s_{k-1} + g_k*(Bh_k - s_{k-1}) + sum_{j<k-1} a_{kj}*(s_j - s_{k-1})),
    with Bh_k = justnorm(B_k(s_{k-1})). Normalizing branch outputs before
    mixing bounds every contribution by 2|g| per channel no matter how hot
    the SiTU activations run, so the trunk output scale is architecture-
    guaranteed instead of init-tuned. All gates init at -1.5 (~0.18 step).
    No RMSNorm or other affine norms: justnorm only.
    """

    def __init__(self, in_dim, width=64, n_blocks=3):
        super().__init__()
        self.in_dim = int(in_dim)
        self.width = int(width)
        self.n_blocks = int(n_blocks)
        if self.n_blocks < 1:
            raise ValueError("SiTUSphereTrunk needs at least one block")
        self.in_proj = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList(SiTUGLUBranch(width, width) for _ in range(n_blocks))
        self.block_gates = nn.ParameterList(
            nn.Parameter(torch.full((width,), -1.5)) for _ in range(n_blocks)
        )
        self.skip_gates = nn.ParameterList()
        self.skip_index = []
        for k in range(1, n_blocks + 1):
            for j in range(k - 1):
                self.skip_index.append((k, j))
                self.skip_gates.append(nn.Parameter(torch.full((width,), -1.5)))

    def forward(self, x):
        streams = [justnorm(self.in_proj(x))]
        skip_pos = 0
        for k, (block, gate) in enumerate(zip(self.blocks, self.block_gates), start=1):
            h = streams[-1] + torch.sigmoid(gate) * (justnorm(block(streams[-1])) - streams[-1])
            for j in range(k - 1):
                h = h + torch.sigmoid(self.skip_gates[skip_pos]) * (streams[j] - streams[-1])
                skip_pos += 1
            streams.append(justnorm(h))
        return streams[-1]


class LReluSqPair(nn.Module):
    """Two biased Linear+LeakyReluSq stages (the v1 lrelusq hidden pair).

    Init-free; ``make_lrelu_res_trunk`` applies orthogonal sqrt(2) to the
    first linear and a contribution-matched gain to the second.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)
        self.lin1 = nn.Linear(dim, dim)
        self.act = LeakyReluSq()
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


class LReluResTrunk(nn.Module):
    """res_v1 layout with LeakyReluSq pairs instead of SiTU branches.

    Identical scaffolding to ``SiTUResTrunk`` (in-proj, two gated blocks,
    U-net long skip, layer-wide scalar gates init -1.5, no normalization) so
    SiTU-vs-LeakyReluSq differs only in block nonlinearity. Pair output scale
    is matched to the SiTU block target (0.5) by the factory, keeping branch
    contributions comparable at init.
    """

    def __init__(self, in_dim, width=64):
        super().__init__()
        self.in_dim = int(in_dim)
        self.width = int(width)
        self.in_proj = nn.Linear(in_dim, width)
        self.pair1 = LReluSqPair(width)
        self.pair2 = LReluSqPair(width)
        self.lam1 = nn.Parameter(torch.tensor(-1.5))
        self.lam2 = nn.Parameter(torch.tensor(-1.5))
        self.lam_skip = nn.Parameter(torch.tensor(-1.5))

    def forward(self, x):
        x0 = self.in_proj(x)
        h1 = x0 + torch.sigmoid(self.lam1) * self.pair1(x0)
        h2 = h1 + torch.sigmoid(self.lam2) * self.pair2(h1) + torch.sigmoid(self.lam_skip) * x0
        return h2


def make_lrelu_res_trunk(in_dim, width=64, pair_out_var=0.5):
    """Build an init-complete LReluResTrunk with a unit-variance stream.

    The input projection sets stream variance 1 (as in ``make_situ_res_trunk``).
    On a unit stream, lin1 (orthogonal sqrt(2)) yields preact variance 2 --
    exactly the regime where E[LeakyReluSq^2] = 6.375 -- so lin2 targets
    ``pair_out_var`` via gain sqrt(pair_out_var/6.375), matching the SiTU
    block contribution scale for a clean activation ablation.
    """
    trunk = LReluResTrunk(in_dim, width)
    torch.nn.init.orthogonal_(trunk.in_proj.weight, np.sqrt(width / in_dim))
    torch.nn.init.zeros_(trunk.in_proj.bias)
    for pair in (trunk.pair1, trunk.pair2):
        torch.nn.init.orthogonal_(pair.lin1.weight, np.sqrt(2.0))
        torch.nn.init.zeros_(pair.lin1.bias)
        torch.nn.init.orthogonal_(pair.lin2.weight, np.sqrt(pair_out_var / 6.375))
        torch.nn.init.zeros_(pair.lin2.bias)
    return trunk


class LReluSphereTrunk(nn.Module):
    """Sphere geometry with LeakyReluSq pairs instead of SiTU branches.

    Same nGPT mixing as ``SiTUSphereTrunk`` (unit-sphere streams, justnormed
    branch outputs, A + g*(B - A) steps, dense per-channel skips, gates init
    -1.5). Only the block nonlinearity changes: each SiTU-GLU is replaced by
    the 0.5-matched LeakyReluSq pair from ``LReluResTrunk``.
    """

    def __init__(self, in_dim, width=64, n_blocks=3):
        super().__init__()
        self.in_dim = int(in_dim)
        self.width = int(width)
        self.n_blocks = int(n_blocks)
        if self.n_blocks < 1:
            raise ValueError("LReluSphereTrunk needs at least one block")
        self.in_proj = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList(LReluSqPair(width) for _ in range(n_blocks))
        self.block_gates = nn.ParameterList(
            nn.Parameter(torch.full((width,), -1.5)) for _ in range(n_blocks)
        )
        self.skip_gates = nn.ParameterList()
        self.skip_index = []
        for k in range(1, n_blocks + 1):
            for j in range(k - 1):
                self.skip_index.append((k, j))
                self.skip_gates.append(nn.Parameter(torch.full((width,), -1.5)))

    def forward(self, x):
        streams = [justnorm(self.in_proj(x))]
        skip_pos = 0
        for k, (block, gate) in enumerate(zip(self.blocks, self.block_gates), start=1):
            h = streams[-1] + torch.sigmoid(gate) * (justnorm(block(streams[-1])) - streams[-1])
            for j in range(k - 1):
                h = h + torch.sigmoid(self.skip_gates[skip_pos]) * (streams[j] - streams[-1])
                skip_pos += 1
            streams.append(justnorm(h))
        return streams[-1]


def make_lrelu_sphere_trunk(in_dim, width=64, n_blocks=3, pair_out_var=0.5):
    """Build an init-complete LReluSphereTrunk.

    The stream is unit-norm (per-row RMS 1/sqrt(width)), so lin1 uses gain
    sqrt(2*width) to put LeakyReluSq at the v=2 / E[x^2]=6.375 design point.
    lin2 then targets ``pair_out_var`` (default 0.5) before justnorm, matching
    the residual LReluSq pair scale and the SiTU sphere down projection.
    Exact output magnitudes don't survive justnorm; the match keeps init
    directions comparable to stiglu_sphere.
    """
    trunk = LReluSphereTrunk(in_dim, width, n_blocks)
    torch.nn.init.orthogonal_(trunk.in_proj.weight, np.sqrt(width / in_dim))
    torch.nn.init.zeros_(trunk.in_proj.bias)
    lin1_gain = np.sqrt(2.0 * width)
    lin2_gain = np.sqrt(pair_out_var / 6.375)
    for pair in trunk.blocks:
        torch.nn.init.orthogonal_(pair.lin1.weight, lin1_gain)
        torch.nn.init.zeros_(pair.lin1.bias)
        torch.nn.init.orthogonal_(pair.lin2.weight, lin2_gain)
        torch.nn.init.zeros_(pair.lin2.bias)
    return trunk


def make_situ_sphere_trunk(in_dim, width=64, n_blocks=3):
    """Build an init-complete SiTUSphereTrunk.

    Scales need only keep SiTU gate/up preacts near their v=2 design point at
    init: the stream is unit-norm (per-row RMS 1/sqrt(width)), so gate/up use
    gain sqrt(2*width) and down targets branch_out_var 0.5. Exact output
    scales don't matter -- justnorm re-projects every stream and every branch
    output, so drift only changes directions, never magnitudes.
    """
    trunk = SiTUSphereTrunk(in_dim, width, n_blocks)
    torch.nn.init.orthogonal_(trunk.in_proj.weight, np.sqrt(width / in_dim))
    torch.nn.init.zeros_(trunk.in_proj.bias)
    for block in trunk.blocks:
        torch.nn.init.orthogonal_(block.gate.weight, np.sqrt(2.0 * width))
        torch.nn.init.orthogonal_(block.up.weight, np.sqrt(2.0 * width))
        down_std = np.sqrt(0.5 * trunk.blocks[0].out_dim
                           / (block.hidden_dim * SITU_GLU_MEAN_SQUARE))
        torch.nn.init.orthogonal_(block.down.weight, down_std)
    return trunk


class HostMLP:
    def __init__(self, sequential, num_rows):
        if not isinstance(sequential, nn.Sequential):
            raise TypeError("HostMLP mirrors an nn.Sequential")
        if num_rows <= 0:
            raise ValueError("num_rows must be positive")
        self.num_rows = int(num_rows)
        self._layers = []
        self._sources = []
        self._hosts = []
        self._scratch = []
        width = None
        for module in sequential:
            if isinstance(module, nn.Linear):
                if module.weight.dtype != torch.float32 or not module.weight.is_cuda:
                    raise ValueError("HostMLP mirrors FP32 CUDA parameters only")
                if width is not None and module.in_features != width:
                    raise ValueError("inconsistent layer widths")
                width = module.out_features
                weight = torch.empty_like(module.weight, device="cpu", pin_memory=True)
                bias = None
                self._sources.append(module.weight)
                self._hosts.append(weight)
                if module.bias is not None:
                    bias = torch.empty_like(module.bias, device="cpu", pin_memory=True)
                    self._sources.append(module.bias)
                    self._hosts.append(bias)
                # BLAS consumes the transposed view directly; no copy per call.
                self._layers.append(("linear", weight.numpy().T, None if bias is None else bias.numpy()))
                self._scratch.append(np.empty((self.num_rows, width), dtype=np.float32))
            elif isinstance(module, nn.Tanh):
                self._layers.append(("tanh", None, None))
            elif isinstance(module, nn.ReLU):
                self._layers.append(("relu", None, None))
            elif isinstance(module, LeakyReluSq):
                self._layers.append(("leakyrelusq", None, None))
            elif isinstance(module, SiTUGLUBranch):
                linears = (module.gate, module.up, module.down)
                if width is not None and module.in_dim != width:
                    raise ValueError("inconsistent layer widths")
                if (module.gate.in_features != module.in_dim or module.up.in_features != module.in_dim
                        or module.down.in_features != module.hidden_dim
                        or module.gate.out_features != module.hidden_dim
                        or module.up.out_features != module.hidden_dim
                        or module.down.out_features != module.out_dim):
                    raise ValueError("SiTUGLUBranch has inconsistent gate/up/down shapes")
                mirrors = []
                for linear in linears:
                    if linear.weight.dtype != torch.float32 or not linear.weight.is_cuda:
                        raise ValueError("HostMLP mirrors FP32 CUDA parameters only")
                    if linear.bias is not None:
                        raise ValueError("HostMLP mirrors bias-free SiTUGLUBranch blocks only")
                    host = torch.empty_like(linear.weight, device="cpu", pin_memory=True)
                    self._sources.append(linear.weight)
                    self._hosts.append(host)
                    mirrors.append(host.numpy().T)
                width = module.out_dim
                gate_tmp = np.empty((self.num_rows, module.hidden_dim), dtype=np.float32)
                up_tmp = np.empty((self.num_rows, module.hidden_dim), dtype=np.float32)
                sig_tmp = np.empty((self.num_rows, module.hidden_dim), dtype=np.float32)
                self._layers.append(("situglu", mirrors[0], mirrors[1], mirrors[2],
                                     gate_tmp, up_tmp, sig_tmp))
                self._scratch.append(np.empty((self.num_rows, width), dtype=np.float32))
            else:
                raise TypeError(
                    "HostMLP supports Linear/Tanh/ReLU/LeakyReluSq/SiTUGLUBranch layers, "
                    f"not {type(module).__name__}"
                )
        if width is None:
            raise ValueError("HostMLP needs at least one Linear layer")
        first = sequential[0]
        if isinstance(first, nn.Linear):
            self.in_features = first.in_features
        elif isinstance(first, SiTUGLUBranch):
            self.in_features = first.in_dim
        else:
            self.in_features = None
        if self.in_features is None:
            raise ValueError("HostMLP requires the first layer to be Linear or SiTUGLUBranch")
        self.out_features = width
        self.device = self._sources[0].device
        self._event = torch.cuda.Event()
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        """Pull the current device parameters; ordered after prior stream work."""
        for host, source in zip(self._hosts, self._sources):
            host.copy_(source, non_blocking=True)
        self._event.record(torch.cuda.current_stream(self.device))
        self._event.synchronize()

    def __call__(self, x):
        if x.shape != (self.num_rows, self.in_features) or x.dtype != np.float32:
            raise ValueError(f"expected float32 input of shape {(self.num_rows, self.in_features)}")
        h = x
        index = 0
        for entry in self._layers:
            kind = entry[0]
            if kind == "linear":
                _, weight_t, bias = entry
                out = self._scratch[index]
                index += 1
                np.matmul(h, weight_t, out=out)
                if bias is not None:
                    np.add(out, bias, out=out)
                h = out
            elif kind == "tanh":
                np.tanh(h, out=h)
            elif kind == "leakyrelusq":
                np.square(np.maximum(h, 0.5 * h), out=h)
            elif kind == "situglu":
                _, gate_t, up_t, down_t, gate_tmp, up_tmp, sig_tmp = entry
                out = self._scratch[index]
                index += 1
                np.matmul(h, gate_t, out=gate_tmp)
                np.matmul(h, up_t, out=up_tmp)
                # sig_tmp = sigmoid(gate); gate_tmp = 4*tanh(g/4)*sigmoid(g).
                np.negative(gate_tmp, out=sig_tmp)
                np.exp(sig_tmp, out=sig_tmp)
                np.add(sig_tmp, 1.0, out=sig_tmp)
                np.reciprocal(sig_tmp, out=sig_tmp)
                np.divide(gate_tmp, 4.0, out=gate_tmp)
                np.tanh(gate_tmp, out=gate_tmp)
                np.multiply(gate_tmp, 4.0, out=gate_tmp)
                np.multiply(gate_tmp, sig_tmp, out=gate_tmp)
                # up_tmp = 25*tanh(u/25); gate_tmp holds the SiTU-GLU product.
                np.divide(up_tmp, 25.0, out=up_tmp)
                np.tanh(up_tmp, out=up_tmp)
                np.multiply(up_tmp, 25.0, out=up_tmp)
                np.multiply(gate_tmp, up_tmp, out=gate_tmp)
                np.matmul(gate_tmp, down_t, out=out)
                h = out
            else:
                np.maximum(h, 0.0, out=h)
        return h


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class HostSiTUResActor:
    """Host mirror for ``nn.Sequential(SiTUResTrunk, Linear)`` actors.

    Same contract as ``HostMLP`` (fixed row count, ``refresh()`` after every
    optimizer update, ``__call__`` returns a logits view overwritten by the
    next call) but hardwired to the residual trunk layout, whose adds and
    learned gate scalars a flat layer list cannot express.
    """

    def __init__(self, sequential, num_rows):
        if not isinstance(sequential, nn.Sequential) or len(sequential) != 2:
            raise TypeError("HostSiTUResActor mirrors Sequential(SiTUResTrunk, Linear)")
        trunk, head = sequential
        if not isinstance(trunk, SiTUResTrunk) or not isinstance(head, nn.Linear):
            raise TypeError("HostSiTUResActor mirrors Sequential(SiTUResTrunk, Linear)")
        if num_rows <= 0:
            raise ValueError("num_rows must be positive")
        self.num_rows = int(num_rows)
        self.width = int(trunk.width)
        self.hidden_dim = int(trunk.block1.hidden_dim)
        if trunk.block2.hidden_dim != self.hidden_dim or trunk.in_dim != trunk.in_proj.in_features:
            raise ValueError("SiTUResTrunk has inconsistent block shapes")
        for linear in (trunk.in_proj, trunk.block1.gate, trunk.block1.up, trunk.block1.down,
                       trunk.block2.gate, trunk.block2.up, trunk.block2.down, head):
            if linear.weight.dtype != torch.float32 or not linear.weight.is_cuda:
                raise ValueError("HostSiTUResActor mirrors FP32 CUDA parameters only")
        for scalar in (trunk.lam1, trunk.lam2, trunk.lam_skip):
            if scalar.dtype != torch.float32 or not scalar.is_cuda or scalar.numel() != 1:
                raise ValueError("HostSiTUResActor mirrors scalar FP32 CUDA gates only")
        self.in_features = int(trunk.in_dim)
        self.out_features = int(head.out_features)
        self._sources = [trunk.in_proj.weight, trunk.in_proj.bias,
                         trunk.block1.gate.weight, trunk.block1.up.weight, trunk.block1.down.weight,
                         trunk.block2.gate.weight, trunk.block2.up.weight, trunk.block2.down.weight,
                         head.weight, head.bias,
                         trunk.lam1, trunk.lam2, trunk.lam_skip]
        if any(t is None for t in self._sources):
            raise ValueError("SiTUResActor trunk/head must all be biased linears with scalar gates")
        self._hosts = [torch.empty_like(t, device="cpu", pin_memory=True) for t in self._sources]
        (self._w_in, self._b_in, self._g1, self._u1, self._d1,
         self._g2, self._u2, self._d2, self._w_head, self._b_head,
         self._lam1, self._lam2, self._lam_skip) = [h.numpy() for h in self._hosts]
        self._w_in = self._w_in.T
        self._g1, self._u1, self._d1 = self._g1.T, self._u1.T, self._d1.T
        self._g2, self._u2, self._d2 = self._g2.T, self._u2.T, self._d2.T
        self._w_head = self._w_head.T
        rows, width, hidden = self.num_rows, self.width, self.hidden_dim
        self._h0 = np.empty((rows, width), dtype=np.float32)
        self._h1 = np.empty((rows, width), dtype=np.float32)
        self._h2 = np.empty((rows, width), dtype=np.float32)
        self._ga = np.empty((rows, hidden), dtype=np.float32)
        self._ua = np.empty((rows, hidden), dtype=np.float32)
        self._sa = np.empty((rows, hidden), dtype=np.float32)
        self._logits = np.empty((rows, self.out_features), dtype=np.float32)
        self.device = self._sources[0].device
        self._event = torch.cuda.Event()
        self._gate_values = (0.0, 0.0, 0.0)
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        """Pull the current device parameters; ordered after prior stream work."""
        for host, source in zip(self._hosts, self._sources):
            host.copy_(source, non_blocking=True)
        self._event.record(torch.cuda.current_stream(self.device))
        self._event.synchronize()
        self._gate_values = (float(_sigmoid(self._lam1)),
                             float(_sigmoid(self._lam2)),
                             float(_sigmoid(self._lam_skip)))

    def _branch(self, h, gate_t, up_t, down_t, out):
        np.matmul(h, gate_t, out=self._ga)
        np.matmul(h, up_t, out=self._ua)
        np.negative(self._ga, out=self._sa)
        np.exp(self._sa, out=self._sa)
        np.add(self._sa, 1.0, out=self._sa)
        np.reciprocal(self._sa, out=self._sa)
        np.divide(self._ga, 4.0, out=self._ga)
        np.tanh(self._ga, out=self._ga)
        np.multiply(self._ga, 4.0, out=self._ga)
        np.multiply(self._ga, self._sa, out=self._ga)
        np.divide(self._ua, 25.0, out=self._ua)
        np.tanh(self._ua, out=self._ua)
        np.multiply(self._ua, 25.0, out=self._ua)
        np.multiply(self._ga, self._ua, out=self._ga)
        np.matmul(self._ga, down_t, out=out)

    def __call__(self, x):
        if x.shape != (self.num_rows, self.in_features) or x.dtype != np.float32:
            raise ValueError(f"expected float32 input of shape {(self.num_rows, self.in_features)}")
        g1, g2, gs = self._gate_values
        np.matmul(x, self._w_in, out=self._h0)
        np.add(self._h0, self._b_in, out=self._h0)
        self._branch(self._h0, self._g1, self._u1, self._d1, self._h1)
        self._h1 *= g1
        self._h1 += self._h0
        self._branch(self._h1, self._g2, self._u2, self._d2, self._h2)
        self._h2 *= g2
        self._h2 += self._h1
        self._h2 += gs * self._h0
        np.matmul(self._h2, self._w_head, out=self._logits)
        np.add(self._logits, self._b_head, out=self._logits)
        return self._logits


class HostSiTUDenseActor:
    """Host mirror for ``nn.Sequential(SiTUDenseTrunk, Linear)`` actors.

    Same contract as ``HostSiTUResActor`` but loops over an arbitrary block
    count with dense per-channel gated skips, mirroring ``SiTUDenseTrunk``.
    """

    def __init__(self, sequential, num_rows, trunk_cls=SiTUDenseTrunk):
        if not isinstance(sequential, nn.Sequential) or len(sequential) != 2:
            raise TypeError(f"HostSiTUDenseActor mirrors Sequential({trunk_cls.__name__}, Linear)")
        trunk, head = sequential
        if not isinstance(trunk, trunk_cls) or not isinstance(head, nn.Linear):
            raise TypeError(f"HostSiTUDenseActor mirrors Sequential({trunk_cls.__name__}, Linear)")
        if num_rows <= 0:
            raise ValueError("num_rows must be positive")
        self.num_rows = int(num_rows)
        self.width = int(trunk.width)
        self.n_blocks = int(trunk.n_blocks)
        self.hidden_dim = int(trunk.blocks[0].hidden_dim)
        if trunk.in_dim != trunk.in_proj.in_features:
            raise ValueError("SiTUDenseTrunk has inconsistent shapes")
        linears = [trunk.in_proj]
        for block in trunk.blocks:
            if (block.in_dim != self.width or block.out_dim != self.width
                    or block.hidden_dim != self.hidden_dim):
                raise ValueError("SiTUDenseTrunk has inconsistent block shapes")
            if block.gate.bias is not None or block.up.bias is not None or block.down.bias is not None:
                raise ValueError("HostSiTUDenseActor mirrors bias-free SiTUGLUBranch blocks only")
            linears.extend((block.gate, block.up, block.down))
        linears.append(head)
        for linear in linears:
            if linear.weight.dtype != torch.float32 or not linear.weight.is_cuda:
                raise ValueError("HostSiTUDenseActor mirrors FP32 CUDA parameters only")
        if trunk.in_proj.bias is None or head.bias is None:
            raise ValueError("HostSiTUDenseActor requires biased in_proj and head linears")
        gate_params = list(trunk.block_gates) + list(trunk.skip_gates)
        if len(gate_params) != self.n_blocks + self.n_blocks * (self.n_blocks - 1) // 2:
            raise ValueError("SiTUDenseTrunk has inconsistent gate counts")
        for scalar in gate_params:
            if scalar.dtype != torch.float32 or not scalar.is_cuda or tuple(scalar.shape) != (self.width,):
                raise ValueError("HostSiTUDenseActor mirrors per-channel FP32 CUDA gates only")
        self.in_features = int(trunk.in_dim)
        self.out_features = int(head.out_features)
        self._sources = []
        for linear in linears:
            self._sources.extend((linear.weight, linear.bias))  # branch biases are None
        self._sources.extend(gate_params)
        self._hosts = [torch.empty_like(t, device="cpu", pin_memory=True) if t is not None else None
                       for t in self._sources]
        views = [h.numpy() if h is not None else None for h in self._hosts]
        n_lin = len(linears)
        self._lin_t = [views[2 * i].T for i in range(n_lin)]
        self._lin_b = [views[2 * i + 1] for i in range(n_lin)]
        self._gate_raw = views[2 * n_lin:]
        self._gate_values = [np.zeros(self.width, dtype=np.float32) for _ in self._gate_raw]
        rows, width, hidden = self.num_rows, self.width, self.hidden_dim
        self._streams = [np.empty((rows, width), dtype=np.float32) for _ in range(self.n_blocks + 1)]
        self._tmp = np.empty((rows, width), dtype=np.float32)
        self._ga = np.empty((rows, hidden), dtype=np.float32)
        self._ua = np.empty((rows, hidden), dtype=np.float32)
        self._sa = np.empty((rows, hidden), dtype=np.float32)
        self._logits = np.empty((rows, self.out_features), dtype=np.float32)
        self._skip_index = list(trunk.skip_index)
        self.device = self._sources[0].device
        self._event = torch.cuda.Event()
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        """Pull the current device parameters; ordered after prior stream work."""
        for host, source in zip(self._hosts, self._sources):
            if host is not None:
                host.copy_(source, non_blocking=True)
        self._event.record(torch.cuda.current_stream(self.device))
        self._event.synchronize()
        for cached, raw in zip(self._gate_values, self._gate_raw):
            np.negative(raw, out=cached)
            with np.errstate(over="ignore"):
                np.exp(cached, out=cached)
            np.add(cached, 1.0, out=cached)
            np.reciprocal(cached, out=cached)

    def _branch(self, h, block_idx, out):
        base = 1 + 3 * block_idx
        np.matmul(h, self._lin_t[base], out=self._ga)
        np.matmul(h, self._lin_t[base + 1], out=self._ua)
        np.negative(self._ga, out=self._sa)
        # Saturated gates overflow exp toward inf; the following reciprocal
        # maps that to the correct 0 limit, so only the warning is silenced.
        with np.errstate(over="ignore"):
            np.exp(self._sa, out=self._sa)
        np.add(self._sa, 1.0, out=self._sa)
        np.reciprocal(self._sa, out=self._sa)
        np.divide(self._ga, 4.0, out=self._ga)
        np.tanh(self._ga, out=self._ga)
        np.multiply(self._ga, 4.0, out=self._ga)
        np.multiply(self._ga, self._sa, out=self._ga)
        np.divide(self._ua, 25.0, out=self._ua)
        np.tanh(self._ua, out=self._ua)
        np.multiply(self._ua, 25.0, out=self._ua)
        np.multiply(self._ga, self._ua, out=self._ga)
        np.matmul(self._ga, self._lin_t[base + 2], out=out)

    def __call__(self, x):
        if x.shape != (self.num_rows, self.in_features) or x.dtype != np.float32:
            raise ValueError(f"expected float32 input of shape {(self.num_rows, self.in_features)}")
        np.matmul(x, self._lin_t[0], out=self._streams[0])
        np.add(self._streams[0], self._lin_b[0], out=self._streams[0])
        skip_pos = self.n_blocks  # block gates come first in _gate_values
        for k in range(1, self.n_blocks + 1):
            prev, cur = self._streams[k - 1], self._streams[k]
            self._branch(prev, k - 1, self._tmp)
            np.multiply(self._tmp, self._gate_values[k - 1], out=self._tmp)
            np.add(prev, self._tmp, out=cur)
            for _ in range(k - 1):
                _, sj = self._skip_index[skip_pos - self.n_blocks]
                np.multiply(self._streams[sj], self._gate_values[skip_pos], out=self._tmp)
                cur += self._tmp
                skip_pos += 1
        np.matmul(self._streams[self.n_blocks], self._lin_t[-1], out=self._logits)
        np.add(self._logits, self._lin_b[-1], out=self._logits)
        return self._logits


class HostSiTUSphereActor(HostSiTUDenseActor):
    """Host mirror for ``nn.Sequential(SiTUSphereTrunk, Linear)`` actors.

    Same dense per-channel gated layout as ``HostSiTUDenseActor``, but every
    stream and every branch output is L2-normalized (nGPT ``justnorm``) and
    mixes take the ``A + g*(B - A)`` step form. Only the trunk type check
    and the stream update differ; linear/gate mirroring is inherited.
    """

    def __init__(self, sequential, num_rows):
        super().__init__(sequential, num_rows, trunk_cls=SiTUSphereTrunk)
        self._sq = np.empty((self.num_rows, self.width), dtype=np.float32)
        self._nrm = np.empty((self.num_rows, 1), dtype=np.float32)

    def _justnorm(self, a):
        np.square(a, out=self._sq)
        np.sum(self._sq, axis=1, keepdims=True, out=self._nrm)
        np.sqrt(self._nrm, out=self._nrm)
        np.maximum(self._nrm, 1e-12, out=self._nrm)
        a /= self._nrm
        return a

    def __call__(self, x):
        if x.shape != (self.num_rows, self.in_features) or x.dtype != np.float32:
            raise ValueError(f"expected float32 input of shape {(self.num_rows, self.in_features)}")
        np.matmul(x, self._lin_t[0], out=self._streams[0])
        np.add(self._streams[0], self._lin_b[0], out=self._streams[0])
        self._justnorm(self._streams[0])
        skip_pos = self.n_blocks  # block gates come first in _gate_values
        for k in range(1, self.n_blocks + 1):
            prev, cur = self._streams[k - 1], self._streams[k]
            self._branch(prev, k - 1, self._tmp)
            self._justnorm(self._tmp)
            self._tmp -= prev
            np.multiply(self._tmp, self._gate_values[k - 1], out=self._tmp)
            np.add(prev, self._tmp, out=cur)
            for _ in range(k - 1):
                _, sj = self._skip_index[skip_pos - self.n_blocks]
                np.subtract(self._streams[sj], prev, out=self._tmp)
                np.multiply(self._tmp, self._gate_values[skip_pos], out=self._tmp)
                cur += self._tmp
                skip_pos += 1
            self._justnorm(cur)
        np.matmul(self._streams[self.n_blocks], self._lin_t[-1], out=self._logits)
        np.add(self._logits, self._lin_b[-1], out=self._logits)
        return self._logits


class HostLReluResActor:
    """Host mirror for ``nn.Sequential(LReluResTrunk, Linear)`` actors.

    Same contract and layout as ``HostSiTUResActor`` (in-proj, two gated
    blocks, U-net long skip, scalar gates) with the SiTU product replaced by
    the LeakyReluSq pair: h @ W1 + b -> leaky_relu(., 0.5)^2 -> @ W2 + b.
    """

    def __init__(self, sequential, num_rows):
        if not isinstance(sequential, nn.Sequential) or len(sequential) != 2:
            raise TypeError("HostLReluResActor mirrors Sequential(LReluResTrunk, Linear)")
        trunk, head = sequential
        if not isinstance(trunk, LReluResTrunk) or not isinstance(head, nn.Linear):
            raise TypeError("HostLReluResActor mirrors Sequential(LReluResTrunk, Linear)")
        if num_rows <= 0:
            raise ValueError("num_rows must be positive")
        self.num_rows = int(num_rows)
        self.width = int(trunk.width)
        if (trunk.pair1.dim != self.width or trunk.pair2.dim != self.width
                or trunk.in_dim != trunk.in_proj.in_features):
            raise ValueError("LReluResTrunk has inconsistent block shapes")
        for linear in (trunk.in_proj, trunk.pair1.lin1, trunk.pair1.lin2,
                       trunk.pair2.lin1, trunk.pair2.lin2, head):
            if linear.weight.dtype != torch.float32 or not linear.weight.is_cuda:
                raise ValueError("HostLReluResActor mirrors FP32 CUDA parameters only")
        for scalar in (trunk.lam1, trunk.lam2, trunk.lam_skip):
            if scalar.dtype != torch.float32 or not scalar.is_cuda or scalar.numel() != 1:
                raise ValueError("HostLReluResActor mirrors scalar FP32 CUDA gates only")
        self.in_features = int(trunk.in_dim)
        self.out_features = int(head.out_features)
        self._sources = [trunk.in_proj.weight, trunk.in_proj.bias,
                         trunk.pair1.lin1.weight, trunk.pair1.lin1.bias,
                         trunk.pair1.lin2.weight, trunk.pair1.lin2.bias,
                         trunk.pair2.lin1.weight, trunk.pair2.lin1.bias,
                         trunk.pair2.lin2.weight, trunk.pair2.lin2.bias,
                         head.weight, head.bias,
                         trunk.lam1, trunk.lam2, trunk.lam_skip]
        if any(t is None for t in self._sources):
            raise ValueError("LReluResActor trunk/head must all be biased linears with scalar gates")
        self._hosts = [torch.empty_like(t, device="cpu", pin_memory=True) for t in self._sources]
        (self._w_in, self._b_in,
         self._w1a, self._b1a, self._w1b, self._b1b,
         self._w2a, self._b2a, self._w2b, self._b2b,
         self._w_head, self._b_head,
         self._lam1, self._lam2, self._lam_skip) = [h.numpy() for h in self._hosts]
        self._w_in = self._w_in.T
        self._w1a, self._w1b = self._w1a.T, self._w1b.T
        self._w2a, self._w2b = self._w2a.T, self._w2b.T
        self._w_head = self._w_head.T
        rows, width = self.num_rows, self.width
        self._h0 = np.empty((rows, width), dtype=np.float32)
        self._h1 = np.empty((rows, width), dtype=np.float32)
        self._h2 = np.empty((rows, width), dtype=np.float32)
        self._pa = np.empty((rows, width), dtype=np.float32)
        self._logits = np.empty((rows, self.out_features), dtype=np.float32)
        self.device = self._sources[0].device
        self._event = torch.cuda.Event()
        self._gate_values = (0.0, 0.0, 0.0)
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        """Pull the current device parameters; ordered after prior stream work."""
        for host, source in zip(self._hosts, self._sources):
            host.copy_(source, non_blocking=True)
        self._event.record(torch.cuda.current_stream(self.device))
        self._event.synchronize()
        self._gate_values = (float(_sigmoid(self._lam1)),
                             float(_sigmoid(self._lam2)),
                             float(_sigmoid(self._lam_skip)))

    def _pair(self, h, w_a, b_a, w_b, b_b, out):
        np.matmul(h, w_a, out=self._pa)
        np.add(self._pa, b_a, out=self._pa)
        np.square(np.maximum(self._pa, 0.5 * self._pa), out=self._pa)
        np.matmul(self._pa, w_b, out=out)
        np.add(out, b_b, out=out)

    def __call__(self, x):
        if x.shape != (self.num_rows, self.in_features) or x.dtype != np.float32:
            raise ValueError(f"expected float32 input of shape {(self.num_rows, self.in_features)}")
        g1, g2, gs = self._gate_values
        np.matmul(x, self._w_in, out=self._h0)
        np.add(self._h0, self._b_in, out=self._h0)
        self._pair(self._h0, self._w1a, self._b1a, self._w1b, self._b1b, self._h1)
        self._h1 *= g1
        self._h1 += self._h0
        self._pair(self._h1, self._w2a, self._b2a, self._w2b, self._b2b, self._h2)
        self._h2 *= g2
        self._h2 += self._h1
        self._h2 += gs * self._h0
        np.matmul(self._h2, self._w_head, out=self._logits)
        np.add(self._logits, self._b_head, out=self._logits)
        return self._logits


class HostLReluSphereActor:
    """Host mirror for ``nn.Sequential(LReluSphereTrunk, Linear)`` actors.

    Dense per-channel gated sphere mixing as ``HostSiTUSphereActor``, with
    each SiTU product replaced by the LeakyReluSq pair.
    """

    def __init__(self, sequential, num_rows):
        if not isinstance(sequential, nn.Sequential) or len(sequential) != 2:
            raise TypeError("HostLReluSphereActor mirrors Sequential(LReluSphereTrunk, Linear)")
        trunk, head = sequential
        if not isinstance(trunk, LReluSphereTrunk) or not isinstance(head, nn.Linear):
            raise TypeError("HostLReluSphereActor mirrors Sequential(LReluSphereTrunk, Linear)")
        if num_rows <= 0:
            raise ValueError("num_rows must be positive")
        self.num_rows = int(num_rows)
        self.width = int(trunk.width)
        self.n_blocks = int(trunk.n_blocks)
        if trunk.in_dim != trunk.in_proj.in_features:
            raise ValueError("LReluSphereTrunk has inconsistent shapes")
        linears = [trunk.in_proj]
        for block in trunk.blocks:
            if block.dim != self.width:
                raise ValueError("LReluSphereTrunk has inconsistent block shapes")
            linears.extend((block.lin1, block.lin2))
        linears.append(head)
        for linear in linears:
            if linear.weight.dtype != torch.float32 or not linear.weight.is_cuda:
                raise ValueError("HostLReluSphereActor mirrors FP32 CUDA parameters only")
            if linear.bias is None:
                raise ValueError("HostLReluSphereActor requires biased linears")
        gate_params = list(trunk.block_gates) + list(trunk.skip_gates)
        if len(gate_params) != self.n_blocks + self.n_blocks * (self.n_blocks - 1) // 2:
            raise ValueError("LReluSphereTrunk has inconsistent gate counts")
        for scalar in gate_params:
            if scalar.dtype != torch.float32 or not scalar.is_cuda or tuple(scalar.shape) != (self.width,):
                raise ValueError("HostLReluSphereActor mirrors per-channel FP32 CUDA gates only")
        self.in_features = int(trunk.in_dim)
        self.out_features = int(head.out_features)
        self._sources = []
        for linear in linears:
            self._sources.extend((linear.weight, linear.bias))
        self._sources.extend(gate_params)
        self._hosts = [torch.empty_like(t, device="cpu", pin_memory=True) for t in self._sources]
        views = [h.numpy() for h in self._hosts]
        n_lin = len(linears)
        self._lin_t = [views[2 * i].T for i in range(n_lin)]
        self._lin_b = [views[2 * i + 1] for i in range(n_lin)]
        self._gate_raw = views[2 * n_lin:]
        self._gate_values = [np.zeros(self.width, dtype=np.float32) for _ in self._gate_raw]
        rows, width = self.num_rows, self.width
        self._streams = [np.empty((rows, width), dtype=np.float32) for _ in range(self.n_blocks + 1)]
        self._tmp = np.empty((rows, width), dtype=np.float32)
        self._pa = np.empty((rows, width), dtype=np.float32)
        self._sq = np.empty((rows, width), dtype=np.float32)
        self._nrm = np.empty((rows, 1), dtype=np.float32)
        self._logits = np.empty((rows, self.out_features), dtype=np.float32)
        self._skip_index = list(trunk.skip_index)
        self.device = self._sources[0].device
        self._event = torch.cuda.Event()
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        """Pull the current device parameters; ordered after prior stream work."""
        for host, source in zip(self._hosts, self._sources):
            host.copy_(source, non_blocking=True)
        self._event.record(torch.cuda.current_stream(self.device))
        self._event.synchronize()
        for cached, raw in zip(self._gate_values, self._gate_raw):
            np.negative(raw, out=cached)
            with np.errstate(over="ignore"):
                np.exp(cached, out=cached)
            np.add(cached, 1.0, out=cached)
            np.reciprocal(cached, out=cached)

    def _justnorm(self, a):
        np.square(a, out=self._sq)
        np.sum(self._sq, axis=1, keepdims=True, out=self._nrm)
        np.sqrt(self._nrm, out=self._nrm)
        np.maximum(self._nrm, 1e-12, out=self._nrm)
        a /= self._nrm
        return a

    def _pair(self, h, block_idx, out):
        base = 1 + 2 * block_idx
        np.matmul(h, self._lin_t[base], out=self._pa)
        np.add(self._pa, self._lin_b[base], out=self._pa)
        np.square(np.maximum(self._pa, 0.5 * self._pa), out=self._pa)
        np.matmul(self._pa, self._lin_t[base + 1], out=out)
        np.add(out, self._lin_b[base + 1], out=out)

    def __call__(self, x):
        if x.shape != (self.num_rows, self.in_features) or x.dtype != np.float32:
            raise ValueError(f"expected float32 input of shape {(self.num_rows, self.in_features)}")
        np.matmul(x, self._lin_t[0], out=self._streams[0])
        np.add(self._streams[0], self._lin_b[0], out=self._streams[0])
        self._justnorm(self._streams[0])
        skip_pos = self.n_blocks
        for k in range(1, self.n_blocks + 1):
            prev, cur = self._streams[k - 1], self._streams[k]
            self._pair(prev, k - 1, self._tmp)
            self._justnorm(self._tmp)
            self._tmp -= prev
            np.multiply(self._tmp, self._gate_values[k - 1], out=self._tmp)
            np.add(prev, self._tmp, out=cur)
            for _ in range(k - 1):
                _, sj = self._skip_index[skip_pos - self.n_blocks]
                np.subtract(self._streams[sj], prev, out=self._tmp)
                np.multiply(self._tmp, self._gate_values[skip_pos], out=self._tmp)
                cur += self._tmp
                skip_pos += 1
            self._justnorm(cur)
        np.matmul(self._streams[self.n_blocks], self._lin_t[-1], out=self._logits)
        np.add(self._logits, self._lin_b[-1], out=self._logits)
        return self._logits
