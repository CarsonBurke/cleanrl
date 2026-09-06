"""Per-component latency attribution for one vectorized rollout step.

Measures, for a fixed ``num_envs``, the isolated cost of each link in the
synchronous rollout chain so the next optimization target is chosen from data:
native physics (per thread count), host normalization, the compiled policy
chain as written in ``ppo_continuous_action.py``, and a hand-captured CUDA graph
of the same GPU work. Submit through mlq; there is no learning signal here.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanrl.shared.host_actor import HostMLP
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.rollout_transfer import ActionTransfer, RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import sample_beta_actions, sample_beta_actions_host
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm


def timeit(fn, iters, warmup=50):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    return statistics.median(samples) * 1e6, min(samples) * 1e6


def bench_env(env_id, num_envs, threads, iters):
    envs = make_mujoco_vector_env(env_id, num_envs, backend="native", num_threads=threads)
    envs.reset(seed=1)
    rng = np.random.default_rng(0)
    action = rng.uniform(-1, 1, size=envs.action_space.shape).astype(np.float32)
    med, best = timeit(lambda: envs.step(action), iters)
    envs.close()
    return med, best


def bench_norm(env_id, num_envs, iters):
    envs = make_mujoco_vector_env(env_id, num_envs, backend="native", num_threads=1)
    raw, _ = envs.reset(seed=1)
    rng = np.random.default_rng(0)
    action = rng.uniform(-1, 1, size=envs.action_space.shape).astype(np.float32)
    raw, rew, terms, truncs, infos = envs.step(action)
    envs.close()
    obs_norm = VectorObsNorm(num_envs, raw.shape[1:])
    rew_norm = VectorRewardNorm(num_envs, 0.99)

    def step():
        rew_norm.normalize(rew, terms)
        obs_norm.normalize_step(raw, terms, truncs, infos)

    return timeit(step, iters)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                                   nn.Linear(64, 2 * act_dim))
        self.critic = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                                    nn.Linear(64, 1))
        self.register_buffer("low", torch.full((act_dim,), -1.0))
        self.register_buffer("high", torch.full((act_dim,), 1.0))
        self.register_buffer("log_scale", torch.full((act_dim,), float(np.log(2.0))))

    def policy(self, x):
        alpha, beta = (F.softplus(self.actor(x)) + 1.0).chunk(2, dim=-1)
        return alpha, beta, self.critic(x)

    def logprob(self, alpha, beta, native):
        return (Beta(alpha, beta, validate_args=False).log_prob(native) - self.log_scale).sum(-1)


def bench_gpu(num_envs, obs_dim, act_dim, num_steps, iters, mode):
    device = torch.device("cuda")
    agent = Actor(obs_dim, act_dim).to(device)
    policy = torch.compile(agent.policy, mode=mode, fullgraph=True, dynamic=False)
    logprob = torch.compile(agent.logprob, mode=mode, fullgraph=True, dynamic=False)
    obs = torch.empty((num_steps, num_envs, obs_dim), device=device)
    native_buf = torch.empty((num_steps, num_envs, act_dim), device=device)
    logp_buf = torch.empty((num_steps, num_envs), device=device)
    val_buf = torch.empty((num_steps, num_envs), device=device)
    transfer = RolloutTransfer(num_steps, num_envs, (obs_dim,), device)
    action_transfer = ActionTransfer((num_envs, act_dim), device)
    host_obs = np.random.default_rng(0).standard_normal((num_envs, obs_dim)).astype(np.float32)
    next_obs = transfer.observation(host_obs)
    counter = [0]

    @torch.no_grad()
    def baseline_step():
        step = counter[0] % num_steps
        counter[0] += 1
        torch.compiler.cudagraph_mark_step_begin()
        alpha, beta, value = policy(next_obs)
        native, action = sample_beta_actions(alpha, beta, agent.low, agent.high)
        lp = logprob(alpha, beta, native)
        action_transfer.submit(action)
        obs[step].copy_(next_obs)
        native_buf[step].copy_(native)
        val_buf[step].copy_(value.flatten())
        logp_buf[step].copy_(lp)
        host = action_transfer.wait()
        transfer.observation(host_obs)
        return host

    results = {"baseline_chain": timeit(baseline_step, iters)}

    @torch.no_grad()
    def policy_only():
        torch.compiler.cudagraph_mark_step_begin()
        policy(next_obs)
        torch.cuda.synchronize()

    results["compiled_policy_sync"] = timeit(policy_only, iters)

    @torch.no_grad()
    def sample_only():
        alpha, beta, _ = agent.policy(next_obs)
        sample_beta_actions(alpha, beta, agent.low, agent.high)

    results["eager_policy_sample_nosync"] = timeit(sample_only, iters)

    # Hand-captured CUDA graph: H2D obs, policy, sample, logprob, scatter into
    # rollout storage at a device step index, D2H action to pinned host.
    pinned_obs = torch.empty((num_envs, obs_dim), dtype=torch.float32, pin_memory=True)
    pinned_act = torch.empty((num_envs, act_dim), dtype=torch.float32, pin_memory=True)
    static_obs = torch.empty((num_envs, obs_dim), device=device)
    step_index = torch.zeros((), dtype=torch.long, device=device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    compiled_policy = torch.compile(agent.policy, fullgraph=True, dynamic=False,
                                    options={"triton.cudagraphs": False})
    compiled_logprob = torch.compile(agent.logprob, fullgraph=True, dynamic=False,
                                     options={"triton.cudagraphs": False})

    def make_body(policy_fn, logprob_fn):
        @torch.no_grad()
        def graph_body():
            static_obs.copy_(pinned_obs, non_blocking=True)
            alpha, beta, value = policy_fn(static_obs)
            native, action = sample_beta_actions(alpha, beta, agent.low, agent.high)
            lp = logprob_fn(alpha, beta, native)
            idx = step_index.view(1)
            obs.index_copy_(0, idx, static_obs.unsqueeze(0))
            native_buf.index_copy_(0, idx, native.unsqueeze(0))
            val_buf.index_copy_(0, idx, value.flatten().unsqueeze(0))
            logp_buf.index_copy_(0, idx, lp.unsqueeze(0))
            pinned_act.copy_(action, non_blocking=True)
            step_index.add_(1).remainder_(num_steps)
        return graph_body

    event = torch.cuda.Event()
    pinned_arr = pinned_obs.numpy()
    act_arr = pinned_act.numpy()
    for label, body in (("eager", make_body(agent.policy, agent.logprob)),
                        ("inductor", make_body(compiled_policy, compiled_logprob))):
        with torch.cuda.stream(stream):
            for _ in range(3):
                body()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            body()
        stream.synchronize()

        def graph_step(graph=graph):
            np.copyto(pinned_arr, host_obs)
            graph.replay()
            event.record(stream)
            event.synchronize()
            return act_arr

        with torch.cuda.stream(stream):
            results[f"cuda_graph_chain[{label}]"] = timeit(graph_step, iters)
            results[f"cuda_graph_replay_cpu[{label}]"] = timeit(graph.replay, iters)
            stream.synchronize()
    return results


def bench_host_actor(num_envs, obs_dim, act_dim, iters):
    agent = Actor(obs_dim, act_dim).cuda()
    mirror = HostMLP(agent.actor, num_envs)
    rng = np.random.default_rng(0)
    host_obs = rng.standard_normal((num_envs, obs_dim)).astype(np.float32)
    low, high = np.full(act_dim, -1.0, np.float32), np.full(act_dim, 1.0, np.float32)
    results = {"host_actor_forward": timeit(lambda: mirror(host_obs), iters)}
    results["host_actor_chain"] = timeit(lambda: sample_beta_actions_host(mirror(host_obs), low, high, rng), iters)
    results["host_actor_refresh"] = timeit(mirror.refresh, iters)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="HalfCheetah-v4")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    configure_runtime()
    print(f"num_envs={args.num_envs} env={args.env_id} (median / min, microseconds per vector step)")
    for threads in args.threads:
        med, best = bench_env(args.env_id, args.num_envs, threads, args.iters)
        print(f"  env_step threads={threads:<2d} {med:8.1f} / {best:8.1f}")
    med, best = bench_norm(args.env_id, args.num_envs, args.iters)
    print(f"  host_normalize       {med:8.1f} / {best:8.1f}")
    if args.cpu_only:
        return
    envs = make_mujoco_vector_env(args.env_id, args.num_envs, backend="native", num_threads=1)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    envs.close()
    for bench in (bench_host_actor(args.num_envs, obs_dim, act_dim, args.iters),
                  bench_gpu(args.num_envs, obs_dim, act_dim, 2048, args.iters, args.compile_mode)):
        for name, (med, best) in bench.items():
            print(f"  {name:<28s} {med:8.1f} / {best:8.1f}")


if __name__ == "__main__":
    main()
