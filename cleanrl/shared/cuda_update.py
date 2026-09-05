"""Capture a complete Adam update without applying capture warmup to training.

This is experimental infrastructure for isolated fixed-shape CUDA learners.
Integration with live compiled peer learners/Inductor CUDA graph trees is
unsupported: the regression diagnostic reproduces an illegal memory access.
Do not use this helper in production training; use the original compiled update.
The raw loss must
return ``(scalar_loss, metrics)`` and perform device-only work. Compilation,
forward, backward, optional gradient clipping, optimizer and tensor-only
post-step constraints are captured together. Input shape, parameter membership
and optimizer options must remain fixed; learning rate has a device setter.
"""

import torch
from torch.utils._pytree import tree_flatten, tree_map

from cleanrl.shared.runtime import configure_compile_cache


class CudaGraphUpdate:
    """A reusable full learner update; returned metric buffers are overwritten.

    Pass every module whose buffers the loss may mutate in ``modules``.
    Snapshot returned tensors before another call if they must survive replay.
    Sampling inside the loss uses CUDA's graph-aware default RNG. Custom RNG
    objects and host-dependent control flow are not supported by this helper.
    The optimizer is owned by this object after construction; do not call its
    step/zero_grad or mutate its groups externally while using the graph.
    Replays and learning-rate changes must use the constructor's caller stream.
    Captured gradients are internal scratch storage; preexisting ``p.grad`` is
    not preserved. Differentiable inputs and FP64 parameters are unsupported.
    Release prior autograd graphs before construction (parameter snapshots
    must be detached), so gradient accumulation can bind to the capture stream.
    """

    def __init__(self, loss_fn, optimizer, example_inputs, *, modules=(),
                 max_grad_norm=None, post_step=None, compile_loss=True, warmup=3):
        configure_compile_cache()
        if type(optimizer) not in (torch.optim.Adam, torch.optim.AdamW):
            raise TypeError("CudaGraphUpdate supports torch Adam and AdamW")
        if warmup < 1:
            raise ValueError("warmup must be positive")
        if not isinstance(example_inputs, (tuple, list)):
            raise ValueError("example_inputs must be a tuple or list of positional inputs")
        example_inputs = tuple(example_inputs)
        leaves, self._input_spec = tree_flatten(example_inputs)
        if not leaves or any(not isinstance(x, torch.Tensor) or not x.is_cuda for x in leaves):
            raise ValueError("example inputs must contain only CUDA tensors")
        self.device = leaves[0].device
        if any(x.device != self.device for x in leaves):
            raise ValueError("all inputs must share one CUDA device")
        if any(x.requires_grad for x in leaves):
            raise ValueError("rollout inputs must not require gradients")
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        if any(p.device != self.device for p in parameters):
            raise ValueError("all optimized parameters must share the input device")
        if any(p.dtype not in (torch.float32, torch.float16, torch.bfloat16) for p in parameters):
            raise ValueError("capture supports FP32/BF16/FP16 parameters, not FP64")
        if any(not group.get("fused", False) for group in optimizer.param_groups):
            raise ValueError("use fused=True when constructing Adam/AdamW")
        self.optimizer = optimizer
        self._inputs = tree_map(lambda x: x.detach().clone(), example_inputs)
        self._input_leaves, _ = tree_flatten(self._inputs)
        self._rates = []
        for group in optimizer.param_groups:
            group["capturable"] = True
            rate = torch.as_tensor(group["lr"], dtype=torch.float32, device=self.device).clone()
            group["lr"] = rate
            self._rates.append(rate)
            for p in group["params"]:
                state = optimizer.state.get(p, {})
                if "step" in state:
                    state["step"] = state["step"].to(self.device)
        tensors = parameters + [b for module in modules for b in module.buffers()]
        # Deduplicate tied parameters/buffers without retaining duplicate copies.
        tensors = list({id(t): t for t in tensors}.values())
        snapshots = [t.detach().clone() for t in tensors]
        state_snapshot = {
            p: {key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in state.items()}
            for p, state in optimizer.state.items()
        }
        # The captured graph contains device pointers, not Python ownership of
        # compiled-code constants. Keep the callable and its closed-over state
        # alive for every replay, including after the constructor returns.
        self._loss = torch.compile(loss_fn, fullgraph=True, options={"triton.cudagraphs": False}) if compile_loss else loss_fn

        def update():
            optimizer.zero_grad(set_to_none=False)
            result = self._loss(*self._inputs)
            if not isinstance(result, tuple) or len(result) != 2 or result[0].numel() != 1:
                raise ValueError("loss_fn must return (scalar_loss, metrics)")
            result[0].backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, foreach=True)
            optimizer.step()
            if post_step is not None:
                with torch.no_grad():
                    post_step()
            return result

        with torch.cuda.device(self.device):
            rng = torch.cuda.get_rng_state(self.device)
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            try:
                with torch.enable_grad(), torch.cuda.stream(stream):
                    for _ in range(warmup):
                        update()
                    # A compiled backward can install its output tensors as
                    # p.grad. Those may belong to an Inductor graph-tree pool.
                    # Own separate accumulation buffers before manual capture,
                    # so later inference generations cannot invalidate them.
                    # Keep unused gradients None (Adam must still skip them).
                    for parameter in parameters:
                        if parameter.grad is not None:
                            parameter.grad = parameter.grad.detach().clone()
                stream.synchronize()
                self._graph = torch.cuda.CUDAGraph()
                with torch.enable_grad(), torch.cuda.graph(self._graph, stream=stream):
                    outputs = update()
                stream.synchronize()
                self._outputs = tree_map(lambda x: x.detach() if isinstance(x, torch.Tensor) else x, outputs)
            finally:
                stream.synchronize()
                with torch.no_grad():
                    for tensor, saved in zip(tensors, snapshots):
                        tensor.copy_(saved)
                    for p, state in optimizer.state.items():
                        previous = state_snapshot.get(p, {})
                        for key, value in state.items():
                            if isinstance(value, torch.Tensor):
                                if key in previous:
                                    value.copy_(previous[key])
                                else:
                                    value.zero_()
                torch.cuda.set_rng_state(rng, self.device)
                # Finish restoration before returning graph ownership to caller.
                torch.cuda.current_stream(self.device).synchronize()
        self._caller_stream = torch.cuda.current_stream(self.device).cuda_stream

    def _check_stream(self):
        if torch.cuda.current_stream(self.device).cuda_stream != self._caller_stream:
            raise RuntimeError("use the constructor's caller CUDA stream for this update graph")

    def set_learning_rate(self, value, group=None):
        """Update graph-visible learning rate; no graph recapture is needed."""
        self._check_stream()
        rates = self._rates if group is None else (self._rates[group],)
        for rate in rates:
            rate.fill_(value)

    def __call__(self, *inputs):
        self._check_stream()
        leaves, spec = tree_flatten(inputs)
        if spec != self._input_spec:
            raise ValueError("input structure differs from captured example")
        for source, target in zip(leaves, self._input_leaves):
            if not isinstance(source, torch.Tensor) or (source.shape, source.dtype, source.device) != (target.shape, target.dtype, target.device):
                raise ValueError("input shape, dtype and device must match capture")
            if source.requires_grad:
                raise ValueError("rollout inputs must not require gradients")
        for source, target in zip(leaves, self._input_leaves):
            target.copy_(source)
        self._graph.replay()
        return self._outputs
