"""Capture fixed-shape, multi-optimizer training steps as CUDA graphs.

Why
---
Off-policy learners (TD3/TD7/SAC) take one small update per environment step:
a few hundred kernels on 256-row batches, each far shorter than its launch. An
eager step is therefore launch- and sync-bound. ``capture_steps`` records each
step function once and replays it with a single launch, so a whole update --
replay sampling, several losses, several optimizers, priority writes -- costs
one ``CUDAGraph.replay``.

Contracts
---------
- A step function takes no arguments and does device-only work on tensors
  that outlive it (module parameters, optimizer state, persistent buffers).
  No host reads (``.item()``, ``float(t)``, ``.cpu()``), no data-dependent
  Python control flow, no shape changes. Results must be written in place
  into persistent tensors; values only returned are lost.
- Randomness must come from the default CUDA generator (``torch.rand`` and
  friends), which is graph-safe: every replay draws fresh numbers.
- Each step must zero its own gradients with ``zero_grad(set_to_none=False)``;
  gradient tensors are allocated during warmup and then owned by the graphs.
- Optimizers must be ``torch.optim.Adam``/``AdamW`` built with ``fused=True``;
  they are switched to ``capturable=True`` here. After capture, do not call
  their ``step``/``zero_grad`` outside the graphs.
- Warmup is invisible to training: every optimized parameter, every
  optimizer-state tensor (absent state is zeroed, including ``step``), every
  tensor in ``state_tensors`` and the CUDA RNG state are restored before
  capture. Pass every other tensor the steps mutate in ``state_tensors``.
- Replays must be issued from the caller's current stream.
"""

import torch

from cleanrl.shared.runtime import configure_compile_cache


def capture_steps(step_fns, *, optimizers, state_tensors=(), warmup=3):
    """Warm up, restore, then capture each of ``step_fns``; returns their graphs."""
    configure_compile_cache()
    if warmup < 1:
        raise ValueError("warmup must be positive")
    for optimizer in optimizers:
        if type(optimizer) not in (torch.optim.Adam, torch.optim.AdamW):
            raise TypeError("capture_steps supports torch Adam and AdamW")
        for group in optimizer.param_groups:
            if not group.get("fused", False):
                raise ValueError("construct Adam/AdamW with fused=True")
            group["capturable"] = True
    parameters = list({id(p): p for o in optimizers for g in o.param_groups for p in g["params"]}.values())
    tensors = list({id(t): t for t in [*parameters, *state_tensors]}.values())
    device = tensors[0].device
    snapshots = [t.detach().clone() for t in tensors]
    optimizer_snapshots = [
        {p: {k: v.clone() for k, v in s.items() if isinstance(v, torch.Tensor)} for p, s in o.state.items()}
        for o in optimizers
    ]
    rng = torch.cuda.get_rng_state(device)
    caller = torch.cuda.current_stream(device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(caller)
    graphs = []
    try:
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                for fn in step_fns:
                    fn()
            # Own plain gradient buffers: a compiled backward may have installed
            # its own output tensors as ``p.grad``.
            for p in parameters:
                if p.grad is not None:
                    p.grad = p.grad.detach().clone()
        stream.synchronize()
        with torch.no_grad():
            for tensor, saved in zip(tensors, snapshots):
                tensor.copy_(saved)
            for optimizer, saved in zip(optimizers, optimizer_snapshots):
                for p, state in optimizer.state.items():
                    previous = saved.get(p, {})
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            value.copy_(previous[key]) if key in previous else value.zero_()
        torch.cuda.set_rng_state(rng, device)
        stream.synchronize()
        for fn in step_fns:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                fn()
            graphs.append(graph)
    finally:
        stream.synchronize()
        caller.wait_stream(stream)
    return graphs
