"""Process-level performance defaults for single-file GPU trainers.

STANDARD: every new training version MUST call :func:`configure_runtime`
before building networks. Thread-pool sizing (``OMP_NUM_THREADS=1``,
``MKL_NUM_THREADS=1``) is NOT set here: those variables only take effect when
exported before the process starts, so pass them via ``mlq --env`` (see the
submit template in CLAUDE.md). What this module does set takes effect
post-import and is therefore safe to centralize:
- FP32 matmul precision ``"high"`` and TF32 for CUDA matmuls/cuDNN. These are
  DEFAULTS, not the production setting: the plasticity and 32xlr families all
  call ``configure_runtime(matmul_precision="highest", allow_tf32=False)`` and
  therefore run with TF32 OFF, for fidelity. Only the iterthink/d3bucket
  family and the benchmark scripts take the TF32 defaults;
- ``torch.set_num_threads(1)``: these trainers run all math on CUDA, so CPU
  intra-op parallelism is pure oversubscription overhead on tiny ops.
"""

import os
from pathlib import Path

import torch


def configure_compile_cache():
    """Keep reusable compiler artifacts on disk, respecting explicit locations."""
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "cleanrl"
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "torchinductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(root / "triton"))


def configure_runtime(
    *,
    cudnn_deterministic=True,
    matmul_precision="high",
    allow_tf32=True,
    cpu_threads=1,
):
    configure_compile_cache()
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.set_float32_matmul_precision(matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
