"""Matched production-size learner benchmark; submit through mlq, CUDA only.

Measures execution cost, not learning quality or end-to-end environment SPS.
Each variant runs in a separate process to avoid compiled graph interference.
"""
import argparse
import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.runtime import configure_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('v9', 'v10'), required=True)
    parser.add_argument('--steps', type=int, default=320)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    assert args.steps > 0 and args.repeats > 0
    assert torch.cuda.is_available(), 'Submit through mlq on CUDA'
    configure_runtime(cudnn_deterministic=True, matmul_precision='highest', allow_tf32=False)
    torch.manual_seed(1)
    suffix = 'quotient_wml_v9' if args.variant == 'v9' else 'fused_wml_v10'
    model = importlib.import_module('cleanrl.ppo_continuous_action_jepa_intact_' + suffix)
    config = model.Args()
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1, 1, (6,), dtype=np.float32),
    )
    agent = model.Agent(envs, config).cuda()
    groups = agent.parameter_groups()
    optimizers = (
        torch.optim.AdamW(groups[0], lr=config.ssl_learning_rate, weight_decay=config.ssl_weight_decay, fused=True),
        torch.optim.Adam(groups[1], lr=config.learning_rate, eps=1e-5, fused=True),
        torch.optim.Adam(groups[2], lr=config.learning_rate, eps=1e-5, fused=True),
    )
    rows = 16384
    obs = torch.randn(rows, agent.input_dim, device='cuda')
    batch = (obs, torch.rand(rows, 6, device='cuda') * .8 + .1,
             torch.ones(rows, device='cuda'), torch.randn(rows, device='cuda'),
             torch.randn(rows, device='cuda'), torch.randn_like(obs),
             torch.randn_like(obs), torch.randn(rows, device='cuda'),
             torch.zeros(rows, device='cuda'))
    indices = torch.randperm(rows, device='cuda').split(512)
    if args.variant == 'v10':
        loss_fn = torch.compile(lambda ix, data: model.indexed_training_loss(agent, ix, data, config),
                                fullgraph=True, mode='reduce-overhead')
        clip = torch.compile(model.clip_gradient_groups, fullgraph=True, dynamic=False,
                             options={'triton.cudagraphs': False})
        metric_sum = torch.zeros(len(model.METRIC_NAMES), device='cuda')
    else:
        loss_fn = torch.compile(lambda *data: model.training_loss(agent, *data, config),
                                fullgraph=True, mode='reduce-overhead')
        metric_sum = None
    norms = torch.empty((args.steps, 3), device='cuda')

    def update(step):
        nonlocal metric_sum
        torch.compiler.cudagraph_mark_step_begin()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        ix = indices[step % len(indices)]
        loss, metrics = loss_fn(ix, batch) if args.variant == 'v10' else loss_fn(*(x[ix] for x in batch))
        loss.backward()
        if args.variant == 'v10':
            gradients = tuple(tuple(p.grad for p in group if p.grad is not None) for group in groups)
            norms[step % args.steps].copy_(clip(gradients, config.max_grad_norm))
        else:
            for column, group in enumerate(groups):
                norms[step % args.steps, column] = torch.nn.utils.clip_grad_norm_(group, config.max_grad_norm)
        for optimizer in optimizers:
            optimizer.step()
        if args.variant == 'v10':
            metric_sum.add_(metrics.detach())
        else:
            if metric_sum is None:
                metric_sum = {name: torch.zeros_like(value) for name, value in metrics.items()}
            for name, value in metrics.items():
                metric_sum[name].add_(value.detach())

    # Compilation and allocator warmup are excluded; this is a component benchmark.
    for step in range(32):
        update(step)
    torch.cuda.synchronize()
    durations = []
    run = f'HalfCheetah-v4__intact_update_benchmark_{args.variant}__1__{int(time.time())}'
    with SummaryWriter('runs/' + run) as writer:
        writer.add_text('benchmark', 'Synthetic production-size learner; no environment return evidence.')
        for repeat in range(args.repeats):
            start = time.perf_counter()
            for step in range(args.steps):
                update(step)
            torch.cuda.synchronize()
            duration = time.perf_counter() - start
            durations.append(duration)
            writer.add_scalar('benchmark/update_seconds', duration, repeat)
        assert torch.isfinite(norms).all(), 'Nonfinite gradients invalidate timing'
    result = dict(variant=args.variant, steps=args.steps, durations_s=durations,
                  median_s=float(np.median(durations)), device=torch.cuda.get_device_name(),
                  torch_version=torch.__version__, batch_size=rows, minibatch_size=512,
                  scope='learner only; not environment SPS or learning quality')
    path = Path('benchmarks') / f'intact_update_{args.variant}.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    # Separate attribution pass: profiling overhead never enters timing above.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
        for step in range(16):
            update(step)
        torch.cuda.synchronize()
    profile_path = path.with_suffix('.profile.txt')
    profile_path.write_text(profile.key_averages().table(sort_by='self_cuda_time_total', row_limit=40))


if __name__ == '__main__':
    main()
