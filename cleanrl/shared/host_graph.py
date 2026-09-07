"""Fused native forward for the host-side policy mirrors of ``host_actor``.

Why
---
``host_actor.HostSiTUSphereActor`` is the behavior policy of the live runs. Its
forward is ~1.2 MFLOP for 16 rows of a width-64, 3-block trunk, but NumPy
spreads it over ~85 ufunc calls on (16, 64) / (16, 43) arrays, so per-call
dispatch -- not arithmetic -- sets the ~90us cost. ``HostGraphActor`` mirrors
the same network as a preassembled op graph (op codes plus integer operands in
one int32 array, buffer addresses in one pointer table) and evaluates it with a
single ctypes call into ``host_kernel.c``. The graph is marshalled once at
construction; a step only re-binds the input and output pointers.

Contracts (identical to the NumPy mirrors, so this is a drop-in replacement)
---------------------------------------------------------------------------
- ``refresh`` copies the current device parameters into pinned host storage
  (one small D2H per tensor, one event wait) and re-derives the per-channel
  ``sigmoid(gate)`` values. Call it after every optimizer update and before the
  first rollout; the mirror is otherwise stale. In-place parameter updates need
  no reallocation: the pinned mirrors and their C buffers are permanent.
- ``__call__`` takes a C-contiguous float32 ``(num_rows, in_features)`` array
  and returns a ``(num_rows, out_features)`` float32 view of internal scratch,
  overwritten by the next call. Row count is fixed at construction.
- Supported architectures: ``Sequential(<trunk>, Linear)`` for every trunk in
  ``host_actor`` (``SiTUSphereTrunk``, ``SiTUDenseTrunk``, ``SiTUResTrunk``,
  ``LReluSphereTrunk``, ``LReluResTrunk``), ``NormResidualTrunk`` from
  ``norm_residual`` and its ``PreRMSTrunk``/``PreRMSStageTrunk`` subclasses, and plain ``nn.Sequential`` stacks of
  ``Linear``/``Tanh``/``ReLU``/``LeakyReluSq``/``SiTUGLUBranch`` (what
  ``HostMLP`` accepts). Unsupported networks may use a NumPy mirror, except
  ``NormResidualTrunk`` and its subclasses, which require fused execution.
- Host arithmetic is true FP32 with no relaxed-IEEE compiler flags. It is not
  bit-identical to the NumPy mirrors: BLAS reassociates its dot products, the
  row sums in ``justnorm`` use a different (explicit, 16-way) partial-sum
  order, and tanh/sigmoid use the kernel's own <=3 ulp polynomials instead of
  NumPy's. Deviation from the FP32 device forward stays at the same ~1e-6
  order the NumPy mirrors already have.
  LayerNorm/RMSNorm use eps=1e-5 and population variance/mean square,
  respectively; their FP32 reduction order can differ from PyTorch. LayerNorm
  must be non-affine; learned RMS gains use the same pinned refresh contract
  as linear parameters. Fixed scales and RMS gains lower to native gated-add
  with a zero base, without a sigmoid or per-call NumPy arithmetic.

Weight layout
-------------
Weights are stored transposed relative to ``nn.Linear``, as ``(in, out)``
row-major, so the kernel's innermost loop is contiguous over output columns:
one broadcast source element feeds independent FMAs into register-resident
accumulators, with no horizontal reduction and no vectorization over the
reduction axis (17 for the input projection, which would need masking). The
transpose is a per-``refresh`` copy of a few KB, i.e. once per optimizer step
rather than once per env step.
"""

from collections import namedtuple
import ctypes
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import warnings

import numpy as np
import torch
from torch import nn

from cleanrl.shared.host_actor import (
    SQ_PAIR_CAP, CappedLeakyReluSq, CappedSignedSquare, HostLReluResActor,
    HostLReluSphereActor, HostMLP, HostSiTUDenseActor, HostSiTUResActor,
    HostSiTUSphereActor, LeakyReluSq, LReluResTrunk, LReluSphereTrunk,
    LReluSqPair, SignedSquare, SiTUDenseTrunk, SiTUGLUBranch, SiTUResTrunk,
    SiTUSphereTrunk,
)
from cleanrl.shared.norm_residual import NormResidualTrunk
from cleanrl.shared.pre_rms import PreRMSPair, PreRMSTrunk
from cleanrl.shared.pre_rms_stage import PreRMSStage, PreRMSStageTrunk

# Must match the enum in host_kernel.c. Op codes are append-only: a graph
# marshalled by an older process must never mean something else.
_OP_LINEAR = 0
_OP_TANH = 1
_OP_RELU = 2
_OP_LRELUSQ = 3
_OP_SITU_GLU = 4
_OP_JUSTNORM = 5
_OP_GATED_MIX = 6
_OP_GATED_ADD = 7
_OP_GATED_MIX_ACC = 8
_OP_BETA_CONC = 9
_OP_BETA_RESCALE = 10
_OP_SIGNSQ = 11
_OP_CAPSIGNSQ = 12
_OP_CAPLRELUSQ = 13
_OP_LAYERNORM = 14
_OP_RMSNORM = 15
_OP_STRIDE = 8

# -march=native output is CPU specific and no flag here relaxes IEEE semantics.
# -ffp-contract=off is what GCC already does in ISO C mode (verified: identical
# codegen with and without it), stated explicitly because the Beta-head ops are
# bit-identical to NumPy only if `low + span * v` keeps its two roundings.
_BUILD_FLAGS = ("-O3", "-march=native", "-std=c11", "-ffp-contract=off", "-fPIC", "-shared")

_LIBRARY = None


class _HostGraph(ctypes.Structure):
    _fields_ = [
        ("ops", ctypes.c_void_p),
        ("bufs", ctypes.c_void_p),
        ("n_ops", ctypes.c_int),
        ("x_slot", ctypes.c_int),
        ("out_slot", ctypes.c_int),
    ]


def _build_tag():
    """Identify the machine `-march=native` was resolved against."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.machine()


def _kernel_library():
    """Build once into the user cache, publish atomically, load with ctypes."""
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    source = Path(__file__).with_name("host_kernel.c")
    fingerprint = hashlib.sha256(
        source.read_bytes() + "\x00".join(_BUILD_FLAGS).encode() + _build_tag().encode()
    ).hexdigest()[:20]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    directory = cache / "cleanrl" / "host-kernel" / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "host_kernel.so"
    if not output.exists():
        # Concurrent processes compile separate files and atomically publish.
        fd, temporary = tempfile.mkstemp(suffix=".so", dir=directory)
        os.close(fd)
        try:
            subprocess.run(
                ["cc", *_BUILD_FLAGS, str(source), "-o", temporary],
                check=True, capture_output=True, text=True,
            )
            os.replace(temporary, output)
        except (OSError, subprocess.CalledProcessError) as error:
            detail = getattr(error, "stderr", str(error))
            raise RuntimeError(f"Unable to build the host graph kernel: {detail}") from error
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    lib = ctypes.CDLL(str(output))
    lib.cleanrl_host_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.cleanrl_host_forward.restype = None
    _LIBRARY = lib
    return lib


class _Marshalled:
    """One kernel-ready graph, and everything the kernel dereferences.

    The op stream, the pointer table and the struct are all reachable from
    here for exactly as long as ``address`` is: the kernel holds raw addresses
    and nothing else keeps these alive.
    """

    __slots__ = ("ops", "table", "struct", "address")

    def __init__(self, arrays, ops, x_slot, out_slot):
        self.ops = np.asarray(ops, dtype=np.int32)
        self.table = (ctypes.c_void_p * len(arrays))()
        for slot, array in enumerate(arrays):
            self.table[slot] = array.ctypes.data
        self.struct = _HostGraph(
            ops=self.ops.ctypes.data,
            bufs=ctypes.addressof(self.table),
            n_ops=len(self.ops) // _OP_STRIDE,
            x_slot=x_slot,
            out_slot=out_slot,
        )
        self.address = ctypes.addressof(self.struct)


# How one trunk family maps onto the graph: its block module, whether streams
# live on the unit sphere, and -- for the res layout, whose two blocks and
# three layer-wide scalar gates are plain attributes rather than the
# ModuleList/ParameterList pair of the dense and sphere trunks -- the attribute
# names to read them from. ``blocks is None`` means "dense/sphere layout".
_TrunkKind = namedtuple("_TrunkKind", "block sphere blocks gates")

_TRUNK_KINDS = (
    (SiTUSphereTrunk, _TrunkKind(SiTUGLUBranch, True, None, None)),
    (SiTUDenseTrunk, _TrunkKind(SiTUGLUBranch, False, None, None)),
    (LReluSphereTrunk, _TrunkKind(LReluSqPair, True, None, None)),
    (SiTUResTrunk, _TrunkKind(SiTUGLUBranch, False, ("block1", "block2"),
                              ("lam1", "lam2", "lam_skip"))),
    (LReluResTrunk, _TrunkKind(LReluSqPair, False, ("pair1", "pair2"),
                               ("lam1", "lam2", "lam_skip"))),
)


def _trunk_kind(module):
    """The graph layout for ``module``, or ``None`` if it is not a known trunk."""
    if isinstance(module, PreRMSStageTrunk):
        return _TrunkKind(PreRMSStage, False, None, None)
    if isinstance(module, NormResidualTrunk):
        block = SiTUGLUBranch if module.activation == "stiglu" else LReluSqPair
        return _TrunkKind(block, False, None, None)
    for trunk_cls, kind in _TRUNK_KINDS:
        if isinstance(module, trunk_cls):
            return kind
    return None


def _branch_shape(block):
    """``(in_dim, out_dim, hidden_dim)`` of a trunk block of any family."""
    if isinstance(block, SiTUGLUBranch):
        return int(block.in_dim), int(block.out_dim), int(block.hidden_dim)
    return int(block.dim), int(block.dim), int(block.dim)


class HostGraphActor:
    """Native op-graph mirror of ``nn.Sequential(<trunk>, nn.Linear)`` actors.

    Drop-in for ``HostMLP`` and every ``Host*Actor`` mirror in ``host_actor``:
    same constructor signature, same ``refresh()`` / ``__call__`` contract, and
    the same ``num_rows``, ``in_features``, ``out_features``, ``device``
    attributes. See the module docstring for the supported architectures.
    """

    def __init__(self, sequential, num_rows):
        if not isinstance(sequential, nn.Sequential):
            raise TypeError("HostGraphActor mirrors an nn.Sequential")
        if num_rows <= 0:
            raise ValueError("num_rows must be positive")
        self.num_rows = int(num_rows)
        self.in_features = None
        self._sources = []       # device parameters, in mirror order
        self._hosts = []         # pinned host tensors, same order
        self._weight_jobs = []   # (pinned (out, in) view, transposed C buffer)
        self._gate_jobs = []     # (pinned raw gate view, sigmoid(gate) buffer)
        self._arrays = []        # every C buffer, indexed by graph slot
        self._ops = []           # flat int32 op stream
        self._scale_zeros = {}   # shared all-row zero bases for channel scaling
        self._max_cols = 1
        # Slot 0 is re-bound to the caller's input on every call; slot 1 is the
        # shared zero vector that stands in for absent linear biases.
        self._x_slot = self._add(np.zeros(1, dtype=np.float32))
        self._zero_slot = self._add(None)
        kind = _trunk_kind(sequential[0]) if len(sequential) == 2 else None
        if kind is not None:
            self._out_slot, self.out_features = self._build_trunk(sequential, kind)
        else:
            self._out_slot, self.out_features = self._build_mlp(sequential)
        self._arrays[self._zero_slot] = np.zeros(self._max_cols, dtype=np.float32)
        self._logits = self._arrays[self._out_slot]
        self._graph = _Marshalled(self._arrays, self._ops, self._x_slot, self._out_slot)
        self._graph_address = self._graph.address
        self._out_address = self._logits.ctypes.data
        self._forward = _kernel_library().cleanrl_host_forward
        self.device = self._sources[0].device
        self._event = torch.cuda.Event()
        self.refresh()

    # -- graph assembly ----------------------------------------------------

    def _add(self, array):
        self._arrays.append(array)
        return len(self._arrays) - 1

    def _scratch(self, cols):
        return self._add(np.empty((self.num_rows, cols), dtype=np.float32))

    def _op(self, code, *operands):
        self._ops.extend((code, *operands))
        self._ops.extend([0] * (_OP_STRIDE - 1 - len(operands)))

    def _mirror(self, tensor):
        """Register a pinned host mirror of a device parameter."""
        host = torch.empty_like(tensor, device="cpu", pin_memory=True)
        self._sources.append(tensor)
        self._hosts.append(host)
        return host.numpy()

    @staticmethod
    def _require_fp32(tensor):
        if tensor.dtype != torch.float32 or not tensor.is_cuda:
            raise ValueError("HostGraphActor mirrors FP32 CUDA parameters only")

    def _linear(self, dst, src, linear):
        """Emit ``dst = src @ linear.weight.T + linear.bias``.

        ``dst`` is always a freshly allocated scratch buffer, so it never
        aliases ``src``/weight/bias -- which is what lets the kernel declare
        its gemm pointers ``restrict``.
        """
        self._max_cols = max(self._max_cols, linear.out_features)
        weight = self._mirror(linear.weight)
        transposed = np.empty((linear.in_features, linear.out_features), dtype=np.float32)
        self._weight_jobs.append((weight, transposed))
        bias = self._zero_slot if linear.bias is None else self._add(self._mirror(linear.bias))
        self._op(_OP_LINEAR, dst, src, self._add(transposed), bias,
                 self.num_rows, linear.in_features, linear.out_features)

    def _situ_glu_branch(self, src, block):
        """Emit a bias-free SiTU-GLU branch, returning the output slot."""
        if (block.gate.in_features != block.in_dim or block.up.in_features != block.in_dim
                or block.down.in_features != block.hidden_dim
                or block.gate.out_features != block.hidden_dim
                or block.up.out_features != block.hidden_dim
                or block.down.out_features != block.out_dim):
            raise ValueError("SiTUGLUBranch has inconsistent gate/up/down shapes")
        for linear in (block.gate, block.up, block.down):
            self._require_fp32(linear.weight)
            if linear.bias is not None:
                raise ValueError("HostGraphActor mirrors bias-free SiTUGLUBranch blocks only")
        gate = self._scratch(block.hidden_dim)
        up = self._scratch(block.hidden_dim)
        self._linear(gate, src, block.gate)
        self._linear(up, src, block.up)
        self._op(_OP_SITU_GLU, gate, gate, up, 0, self.num_rows * block.hidden_dim)
        out = self._scratch(block.out_dim)
        self._linear(out, gate, block.down)
        return out

    def _lrelu_pair_branch(self, src, block):
        """Emit a squared-activation pair, including PreRMSPair inner norms.

        Returns the output slot. The pair activation is resolved here --
        ``LeakyReluSq``, ``SignedSquare`` and their tanh-capped variants are
        the mirrored ones -- so an unmirrored activation raises instead of
        being silently replaced. The capped ops hardcode ``SQ_PAIR_CAP``, so a
        capped activation with any other cap raises ``ValueError``.
        Both linears are biased in every shipped ``LReluSqPair``; ``_linear``
        emits whatever bias is present, so a bias is never dropped.
        """
        if isinstance(block.act, LeakyReluSq):
            act_code = _OP_LRELUSQ
        elif isinstance(block.act, SignedSquare):
            act_code = _OP_SIGNSQ
        elif isinstance(block.act, (CappedSignedSquare, CappedLeakyReluSq)):
            if block.act.cap != SQ_PAIR_CAP:
                raise ValueError(
                    f"the fused capped squared ops hardcode SQ_PAIR_CAP={SQ_PAIR_CAP}, "
                    f"not cap={block.act.cap}")
            act_code = (_OP_CAPSIGNSQ if isinstance(block.act, CappedSignedSquare)
                        else _OP_CAPLRELUSQ)
        else:
            raise TypeError(
                "LReluSqPair activation must be LeakyReluSq, SignedSquare, "
                "CappedSignedSquare or CappedLeakyReluSq, "
                f"not {type(block.act).__name__}")
        if (block.lin1.in_features != block.dim or block.lin1.out_features != block.dim
                or block.lin2.in_features != block.dim or block.lin2.out_features != block.dim):
            raise ValueError("LReluSqPair has inconsistent lin1/lin2 shapes")
        for linear in (block.lin1, block.lin2):
            self._require_fp32(linear.weight)
            if linear.bias is not None:
                self._require_fp32(linear.bias)
        hidden = self._scratch(block.dim)
        self._linear(hidden, src, block.lin1)
        if isinstance(block, PreRMSPair) and block.norm_position == "preact":
            self._norm(hidden, hidden, block.inner_norm, block.dim)
            self._channel_scale(hidden, hidden, block.norm_scale, block.dim)
        self._op(act_code, hidden, hidden, 0, 0, self.num_rows * block.dim)
        if isinstance(block, PreRMSPair) and block.norm_position == "postact":
            self._norm(hidden, hidden, block.inner_norm, block.dim)
            self._channel_scale(hidden, hidden, block.norm_scale, block.dim)
        out = self._scratch(block.dim)
        self._linear(out, hidden, block.lin2)
        if isinstance(block, PreRMSPair) and block.norm_position == "branch":
            self._norm(out, out, block.inner_norm, block.dim)
            self._channel_scale(out, out, block.norm_scale, block.dim)
        return out

    def _pre_rms_stage_branch(self, src, block):
        """Emit one Linear, squared leaky activation, and calibrated scale."""
        if type(block.lin) is not nn.Linear:
            raise TypeError("PreRMSStage linear must be Linear")
        if type(block.act) is not LeakyReluSq:
            raise TypeError("PreRMSStage activation must be LeakyReluSq")
        if block.lin.in_features != block.dim or block.lin.out_features != block.dim:
            raise ValueError("PreRMSStage has inconsistent linear shapes")
        self._require_fp32(block.lin.weight)
        if block.lin.bias is not None:
            self._require_fp32(block.lin.bias)
        out = self._scratch(block.dim)
        self._linear(out, src, block.lin)
        self._op(_OP_LRELUSQ, out, out, 0, 0, self.num_rows * block.dim)
        self._channel_scale(out, out, block.output_scale, block.dim)
        return out

    def _branch(self, src, block):
        """Emit one trunk block, returning its output slot."""
        if isinstance(block, PreRMSStage):
            return self._pre_rms_stage_branch(src, block)
        if isinstance(block, SiTUGLUBranch):
            return self._situ_glu_branch(src, block)
        return self._lrelu_pair_branch(src, block)

    def _channel_scale(self, dst, src, scale, width):
        """Emit a fixed or learned channel multiplier, not a sigmoid gate."""
        if isinstance(scale, torch.Tensor):
            self._require_fp32(scale)
            if tuple(scale.shape) != (width,):
                raise ValueError("HostGraphActor requires width-wise channel scales")
            values = self._mirror(scale)
        else:
            values = np.full(width, scale, dtype=np.float32)
        if width not in self._scale_zeros:
            self._scale_zeros[width] = self._add(
                np.zeros((self.num_rows, width), dtype=np.float32))
        self._op(_OP_GATED_ADD, dst, self._scale_zeros[width], src, self._add(values),
                 self.num_rows, width)

    def _norm(self, dst, src, norm, width):
        """Emit a last-axis norm, including learned RMS gain; dst may alias src."""
        if type(norm) is nn.Identity:
            if dst != src:
                self._channel_scale(dst, src, 1.0, width)
            return
        if type(norm) not in (nn.LayerNorm, nn.RMSNorm):
            raise TypeError("HostGraphActor requires LayerNorm, RMSNorm or Identity")
        if tuple(norm.normalized_shape) != (width,) or norm.eps != 1e-5:
            raise ValueError("HostGraphActor requires width-wise norms with eps=1e-5")
        if isinstance(norm, nn.LayerNorm) and (norm.elementwise_affine or norm.bias is not None):
            raise ValueError("HostGraphActor requires non-affine LayerNorm")
        code = _OP_LAYERNORM if isinstance(norm, nn.LayerNorm) else _OP_RMSNORM
        self._op(code, dst, src, 0, 0, self.num_rows, width)
        if norm.weight is not None:
            self._channel_scale(dst, dst, norm.weight, width)

    def _build_mlp(self, sequential):
        current, cols = self._x_slot, None
        for module in sequential:
            if isinstance(module, nn.Linear):
                self._require_fp32(module.weight)
                if cols is not None and module.in_features != cols:
                    raise ValueError("inconsistent layer widths")
                if self.in_features is None:
                    self.in_features = int(module.in_features)
                cols = int(module.out_features)
                dst = self._scratch(cols)
                self._linear(dst, current, module)
                current = dst
            elif isinstance(module, SiTUGLUBranch):
                if cols is not None and module.in_dim != cols:
                    raise ValueError("inconsistent layer widths")
                if self.in_features is None:
                    self.in_features = int(module.in_dim)
                current = self._situ_glu_branch(current, module)
                cols = int(module.out_dim)
            elif isinstance(module, (nn.Tanh, nn.ReLU, LeakyReluSq)):
                if cols is None:
                    raise ValueError(
                        "HostGraphActor requires the first layer to be Linear or SiTUGLUBranch")
                if isinstance(module, nn.Tanh):
                    code = _OP_TANH
                elif isinstance(module, nn.ReLU):
                    code = _OP_RELU
                else:
                    code = _OP_LRELUSQ
                self._op(code, current, current, 0, 0, self.num_rows * cols)
            else:
                raise TypeError(
                    "HostGraphActor supports Sequential(<host_actor trunk>, Linear) and "
                    "Linear/Tanh/ReLU/LeakyReluSq/SiTUGLUBranch stacks, not "
                    f"{type(module).__name__}"
                )
        if cols is None:
            raise ValueError("HostGraphActor needs at least one Linear or SiTUGLUBranch layer")
        return current, cols

    def _build_trunk(self, sequential, kind):
        trunk, head = sequential
        if not isinstance(head, nn.Linear):
            raise TypeError(
                f"HostGraphActor mirrors Sequential({type(trunk).__name__}, Linear)")
        width = int(trunk.width)
        normalized = isinstance(trunk, NormResidualTrunk)
        plain = isinstance(trunk, (PreRMSTrunk, PreRMSStageTrunk)) and not trunk.residual
        if plain and trunk.placement != "pre":
            raise ValueError("Plain pre-RMS trunks require pre normalization")
        if normalized:
            if trunk.placement not in ("pre", "post"):
                raise ValueError("NormResidualTrunk placement must be pre or post")
            if len(trunk.block_norms) != trunk.n_blocks:
                raise ValueError("NormResidualTrunk has an inconsistent norm count")
        if kind.blocks is None:
            blocks = list(trunk.blocks)
            gate_params = list(trunk.block_gates)
            if not normalized:
                gate_params += list(trunk.skip_gates)
        else:
            blocks = [getattr(trunk, name) for name in kind.blocks]
            gate_params = [getattr(trunk, name) for name in kind.gates]
        n_blocks = len(blocks)
        if n_blocks < 1 or int(getattr(trunk, "n_blocks", n_blocks)) != n_blocks:
            raise ValueError(f"{type(trunk).__name__} has an inconsistent block count")
        # Stage k mixes in every earlier stream j < k-1, in constructor order:
        # this is what every trunk's ``forward`` iterates, and the res layout's
        # x0 long skip is exactly the single skip of a two-block dense trunk.
        skip_index = ([] if normalized else
                      [(k, j) for k in range(1, n_blocks + 1) for j in range(k - 1)])
        declared = getattr(trunk, "skip_index", None)
        if declared is not None and [tuple(pair) for pair in declared] != skip_index:
            raise ValueError(f"{type(trunk).__name__} has an unexpected skip ordering")
        for block in blocks:
            if not isinstance(block, kind.block):
                raise TypeError(
                    f"{type(trunk).__name__} blocks must be {kind.block.__name__}, "
                    f"not {type(block).__name__}")
        hidden = _branch_shape(blocks[0])[2]
        if trunk.in_dim != trunk.in_proj.in_features or trunk.in_proj.out_features != width:
            raise ValueError(f"{type(trunk).__name__} has inconsistent shapes")
        if head.in_features != width:
            raise ValueError("head does not consume the trunk width")
        if trunk.in_proj.bias is None or head.bias is None:
            raise ValueError("HostGraphActor requires biased in_proj and head linears")
        self._require_fp32(trunk.in_proj.weight)
        self._require_fp32(head.weight)
        for block in blocks:
            if _branch_shape(block) != (width, width, hidden):
                raise ValueError(f"{type(trunk).__name__} has inconsistent block shapes")
        if len(gate_params) != (0 if plain else n_blocks + len(skip_index)):
            raise ValueError(f"{type(trunk).__name__} has inconsistent gate counts")
        per_channel = kind.blocks is None
        for scalar in gate_params:
            shaped = tuple(scalar.shape) == (width,) if per_channel else scalar.numel() == 1
            if scalar.dtype != torch.float32 or not scalar.is_cuda or not shaped:
                raise ValueError(
                    f"HostGraphActor mirrors {'per-channel' if per_channel else 'scalar'} "
                    "FP32 CUDA gates only")
        gates = []
        for scalar in gate_params:
            # A layer-wide scalar gate broadcasts into the per-channel buffer at
            # refresh, so the mixing ops need no scalar variants.
            values = np.empty(width, dtype=np.float32)
            self._gate_jobs.append((self._mirror(scalar), values))
            gates.append(self._add(values))

        rows = self.num_rows
        streams = [self._scratch(width)]
        self._linear(streams[0], self._x_slot, trunk.in_proj)
        if kind.sphere:
            self._op(_OP_JUSTNORM, streams[0], streams[0], 0, 0, rows, width)
        if normalized and trunk.placement == "post":
            self._norm(streams[0], streams[0], trunk.input_norm, width)
        skip = n_blocks  # block gates come first in gate_params
        for k in range(1, n_blocks + 1):
            previous = streams[-1]
            branch_input = previous
            if normalized and trunk.placement == "pre":
                branch_input = self._scratch(width)
                self._norm(branch_input, previous, trunk.block_norms[k - 1], width)
            if normalized and trunk.branch_input_scale != 1.0:
                scaled_input = self._scratch(width)
                self._channel_scale(scaled_input, branch_input, trunk.branch_input_scale, width)
                branch_input = scaled_input
            branch = self._branch(branch_input, blocks[k - 1])
            current = branch if plain else self._scratch(width)
            if kind.sphere:
                self._op(_OP_JUSTNORM, branch, branch, 0, 0, rows, width)
                self._op(_OP_GATED_MIX, current, previous, branch, gates[k - 1], rows, width)
            elif not plain:
                self._op(_OP_GATED_ADD, current, previous, branch, gates[k - 1], rows, width)
            for _ in range(0 if normalized else k - 1):
                _, j = skip_index[skip - n_blocks]
                if kind.sphere:
                    self._op(_OP_GATED_MIX_ACC, current, previous, streams[j], gates[skip],
                             rows, width)
                else:
                    self._op(_OP_GATED_ADD, current, current, streams[j], gates[skip],
                             rows, width)
                skip += 1
            if kind.sphere:
                self._op(_OP_JUSTNORM, current, current, 0, 0, rows, width)
            if normalized and trunk.placement == "post":
                self._norm(current, current, trunk.block_norms[k - 1], width)
            streams.append(current)
        if normalized and trunk.placement == "pre":
            self._norm(streams[-1], streams[-1], trunk.final_norm, width)
        if normalized:
            # Fixed readout calibration, not a learned sigmoid gate, even
            # when the final stream norm is disabled.
            self._channel_scale(streams[-1], streams[-1], trunk.output_scale, width)
        logits = self._scratch(head.out_features)
        self._linear(logits, streams[-1], head)
        self.in_features = int(trunk.in_dim)
        return logits, int(head.out_features)

    # -- runtime -----------------------------------------------------------

    @torch.no_grad()
    def refresh(self):
        """Pull the current device parameters; ordered after prior stream work."""
        for host, source in zip(self._hosts, self._sources):
            host.copy_(source, non_blocking=True)
        self._event.record(torch.cuda.current_stream(self.device))
        self._event.synchronize()
        for weight, transposed in self._weight_jobs:
            np.copyto(transposed, weight.T)
        for raw, values in self._gate_jobs:
            np.negative(raw, out=values)
            # Saturated gates overflow exp toward inf; the following reciprocal
            # maps that to the correct 0 limit, so only the warning is silenced.
            with np.errstate(over="ignore"):
                np.exp(values, out=values)
            np.add(values, 1.0, out=values)
            np.reciprocal(values, out=values)

    def __call__(self, x):
        if x.shape != (self.num_rows, self.in_features) or x.dtype != np.float32:
            raise ValueError(f"expected float32 input of shape {(self.num_rows, self.in_features)}")
        if not x.flags.c_contiguous:
            raise ValueError("expected a C-contiguous input")
        self._forward(self._graph_address, x.ctypes.data, self._out_address)
        return self._logits


_FALLBACK_MIRRORS = (
    (SiTUSphereTrunk, HostSiTUSphereActor),
    (SiTUDenseTrunk, HostSiTUDenseActor),
    (SiTUResTrunk, HostSiTUResActor),
    (LReluSphereTrunk, HostLReluSphereActor),
    (LReluResTrunk, HostLReluResActor),
)


def make_host_mirror(sequential, num_rows, *, fused=True):
    """Return the fastest available host mirror of ``sequential``.

    Prefers :class:`HostGraphActor` (one native call) and falls back to the
    architecture-matched NumPy mirror in ``host_actor`` when the kernel does
    not support the network. Every mirror exposes the same ``refresh()`` /
    ``__call__(obs)`` contract, so callers never branch on the choice.

    Falling back costs ~4-10x on the policy forward, which is silent and easy
    to ship by accident, so a successful fallback warns once with the kernel's
    own reason. If the fallback mirror rejects the network too, its error
    propagates instead: the network is unsupported, not merely unfused. The
    result carries ``fused`` and ``fallback_reason`` for logging.
    ``NormResidualTrunk`` has no NumPy fallback: disabling fused execution or
    an unsupported normalized graph raises instead.
    """
    if (isinstance(sequential, nn.Sequential) and len(sequential)
            and isinstance(sequential[0], NormResidualTrunk)):
        if not fused:
            raise ValueError("NormResidualTrunk requires the fused host mirror")
        mirror = HostGraphActor(sequential, num_rows)
        mirror.fused = True
        mirror.fallback_reason = None
        return mirror
    reason = None
    if fused:
        try:
            mirror = HostGraphActor(sequential, num_rows)
        except (TypeError, ValueError) as error:
            reason = str(error)
        else:
            mirror.fused = True
            mirror.fallback_reason = None
            return mirror
    for trunk_cls, mirror_cls in _FALLBACK_MIRRORS:
        if isinstance(sequential[0], trunk_cls):
            break
    else:
        mirror_cls = HostMLP
    mirror = mirror_cls(sequential, num_rows)
    if reason is not None:
        warnings.warn(
            f"HostGraphActor cannot mirror this policy ({reason}); falling back to "
            f"{mirror_cls.__name__}, which costs roughly 4-10x on the policy forward",
            RuntimeWarning, stacklevel=2,
        )
    mirror.fused = False
    mirror.fallback_reason = reason
    return mirror


def _beta_bounds(low, high, act_dim):
    """``(low, high - low)`` as float32 ``(act_dim,)`` buffers.

    Only float32 bounds are accepted. ``high - low`` is hoisted out of the step
    path, which is free of consequence for float32 bounds -- rounding their
    FP64 difference lands on the same float32 as subtracting them in FP32 --
    but NOT for FP64 bounds, where NumPy's own promotion decides whether the
    subtraction rounds once or twice. Rather than guess at that, the graph
    declines those bounds and the caller keeps the NumPy path.
    """
    buffers = []
    for name, bound in (("low", low), ("high", high)):
        array = np.asarray(bound)
        if array.dtype != np.float32:
            raise ValueError(
                f"BetaHeadGraph mirrors float32 action bounds only, not {array.dtype} {name}")
        if array.shape not in ((), (act_dim,)):
            raise ValueError(
                f"expected scalar or ({act_dim},) action bounds, not {name} of shape {array.shape}")
        buffers.append(np.ascontiguousarray(np.broadcast_to(array, (act_dim,))))
    return buffers[0], buffers[1] - buffers[0]


class BetaHeadGraph:
    """The arithmetic around ``rng.beta`` in ``sampling.sample_beta_actions_host``.

    Two single-op graphs through the same kernel, sharing nothing with each
    other but the pointer-table machinery:

    - ``concentration(logits)`` -> ``(alpha, beta)``: ``1 + softplus(logits)``
      split on the last axis into two C-contiguous ``(num_rows, act_dim)``
      arrays, which is what ``numpy.random.Generator.beta`` wants (the NumPy
      path hands it two strided views of one ``(num_rows, 2 * act_dim)``
      buffer instead).
    - ``rescale(draw)`` -> ``(native, action)``: the FP64 draw cast to float32,
      clipped to ``[epsilon, 1 - epsilon]``, then ``low + (high - low) * it``.

    The generator itself is untouched: it is called by the caller, with the
    same values, shapes, dtype and call count as before, so the random stream
    is bit-identical. Both halves are bit-identical to the NumPy expressions
    they replace as well -- see ``host_kernel.c``'s numerics notes and
    ``tests/test_host_graph.py``.

    ``alpha``, ``beta``, ``native`` and ``action`` are permanent buffers
    overwritten by the next call, exactly like ``HostGraphActor``'s logits:
    nothing on this path allocates per step.
    """

    def __init__(self, num_rows, act_dim, low, high, *, epsilon=1e-6):
        if num_rows <= 0 or act_dim <= 0:
            raise ValueError("num_rows and act_dim must be positive")
        self.num_rows, self.act_dim = int(num_rows), int(act_dim)
        self.epsilon = float(epsilon)
        rows, cols = self.num_rows, self.act_dim
        low_buffer, span_buffer = _beta_bounds(low, high, cols)
        self.alpha = np.empty((rows, cols), dtype=np.float32)
        self.beta = np.empty((rows, cols), dtype=np.float32)
        self.native = np.empty((rows, cols), dtype=np.float32)
        self.action = np.empty((rows, cols), dtype=np.float32)
        # low and span share one buffer; the op reads span at affine + cols.
        self._affine = np.concatenate((low_buffer, span_buffer))
        self._bounds = np.array([self.epsilon, 1.0 - self.epsilon], dtype=np.float32)
        # Slot 0 of either graph is re-bound to the caller's array per call.
        stub = np.zeros(1, dtype=np.float32)
        self._concentration = _Marshalled(
            (stub, self.alpha, self.beta),
            (_OP_BETA_CONC, 1, 0, 2, 0, rows, cols, 0),
            x_slot=0, out_slot=1,
        )
        self._rescale = _Marshalled(
            (stub, self.action, self.native, self._affine, self._bounds),
            (_OP_BETA_RESCALE, 1, 0, 2, 3, rows, cols, 4),
            x_slot=0, out_slot=1,
        )
        self._alpha_address = self.alpha.ctypes.data
        self._action_address = self.action.ctypes.data
        self._bound_logits = None
        self._logits_address = 0
        self._forward = _kernel_library().cleanrl_host_forward

    def _bind(self, logits):
        """Validate a new logits array and cache its address.

        ``ndarray.ctypes.data`` costs ~0.6us -- half again the native call it
        serves -- and the rollout hands over the same buffer (the mirror's
        logits) every step, so the address is resolved per array object rather
        than per step. The reference kept here is what keeps that address
        valid. Reshaping an already-bound array in place would go unnoticed;
        passing a different array is validated afresh.
        """
        if logits.shape != (self.num_rows, 2 * self.act_dim) or logits.dtype != np.float32:
            raise ValueError(
                f"expected float32 logits of shape {(self.num_rows, 2 * self.act_dim)}")
        if not logits.flags.c_contiguous:
            raise ValueError("expected C-contiguous logits")
        self._bound_logits = logits
        self._logits_address = logits.ctypes.data

    def concentration(self, logits):
        """``(alpha, beta)`` for float32 C-contiguous ``(num_rows, 2 * act_dim)`` logits."""
        if logits is not self._bound_logits:
            self._bind(logits)
        self._forward(self._concentration.address, self._logits_address, self._alpha_address)
        return self.alpha, self.beta

    def rescale(self, draw):
        """``(native, action)`` for the FP64 ``(num_rows, act_dim)`` Beta draw."""
        if draw.shape != (self.num_rows, self.act_dim) or draw.dtype != np.float64:
            raise ValueError(
                f"expected float64 draws of shape {(self.num_rows, self.act_dim)}")
        if not draw.flags.c_contiguous:
            raise ValueError("expected C-contiguous draws")
        self._forward(self._rescale.address, draw.ctypes.data, self._action_address)
        return self.native, self.action
