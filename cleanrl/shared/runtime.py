"""Process-level performance defaults for single-file GPU trainers.

STANDARD: every new training version MUST call :func:`configure_runtime`
before building networks. Thread-pool sizing (``OMP_NUM_THREADS=1``,
``MKL_NUM_THREADS=1``) is NOT set here: those variables only take effect when
exported before the process starts, so pass them via ``mlq --env`` (see the
submit template in CLAUDE.md). What this module does set takes effect
post-import and is therefore safe to centralize:

- FP32 matmul precision ``"high"`` (TensorFloat32-capable GEMMs on Ampere+);
- TF32 allowed for CUDA matmuls and cuDNN (V-MPO contract; master weights,
  PopArt state, and distribution special functions stay FP32 in caller code);
- ``torch.set_num_threads(1)``: these trainers run all math on CUDA, so CPU
  intra-op parallelism is pure oversubscription overhead on tiny ops.
"""

import torch


def configure_runtime(
    *,
    cudnn_deterministic=True,
    matmul_precision="high",
    allow_tf32=True,
    cpu_threads=1,
):
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.set_float32_matmul_precision(matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
