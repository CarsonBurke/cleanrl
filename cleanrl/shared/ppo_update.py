"""Minibatch indexing fused into a trainer-owned PPO loss graph."""

import torch


def make_minibatch_loss(loss_fn, *, compiled=True, mode="reduce-overhead"):
    """Return ``(indices, *rollout_tensors) -> loss_fn(*(x[indices] ...))``.

    ``loss_fn`` retains all algorithm-specific policy/value/metric semantics.
    Every tensor is indexed along its first dimension, in the supplied order;
    no shuffle, RNG, backward, gradient clipping or optimizer work happens here.

    The compiled wrapper owns the CUDA-graph step boundary. Finish backward and
    consume/copy metrics before calling it again or invoking a compiled peer.
    Returned tensors may alias graph storage; copy metrics into persistent
    logging storage rather than retaining graph outputs across minibatches.

    Full-rollout inputs must outlive all minibatches. If a graph-enabled peer
    produced them, clone/copy them OUTSIDE that peer's compiled boundary once
    per rollout, before this wrapper advances the graph generation. Do not put
    a step marker between this forward and its backward, or chain this helper
    into a larger forward with other pending graph-enabled backwards.
    """
    def indexed_loss(indices, *rollout_tensors):
        return loss_fn(*(tensor[indices] for tensor in rollout_tensors))

    if not compiled:
        return indexed_loss

    compiled_loss = torch.compile(indexed_loss, mode=mode, fullgraph=True, dynamic=False)

    def minibatch_loss(indices, *rollout_tensors):
        torch.compiler.cudagraph_mark_step_begin()
        return compiled_loss(indices, *rollout_tensors)

    return minibatch_loss
