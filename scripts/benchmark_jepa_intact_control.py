"""Queue through mlq: untrained synthetic INTACT direct-control computational cost.

No environment training or control-quality comparison is performed. The CEM
reference uses this exact world model, not a LeWM package or a paper latency.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cleanrl import ppo_continuous_action_jepa_intact_model_control_v5 as model
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler


def measure(function, warmup, repeats, batch_size, *, cuda):
    def synchronize():
        if cuda:
            torch.cuda.synchronize()

    synchronize()
    started = time.perf_counter()
    for _ in range(warmup):
        if cuda:
            torch.compiler.cudagraph_mark_step_begin()
        function()
    synchronize()
    warmup_seconds = time.perf_counter() - started
    milliseconds = []
    for _ in range(repeats):
        if cuda:
            torch.compiler.cudagraph_mark_step_begin()
        synchronize()
        started = time.perf_counter_ns()
        function()
        synchronize()
        milliseconds.append((time.perf_counter_ns() - started) / 1e6)
    values = np.asarray(milliseconds)
    return {
        "warmup_calls": warmup,
        "warmup_seconds_including_compilation": warmup_seconds,
        "timed_calls": repeats,
        "batch_size": batch_size,
        "latency_ms_mean": float(values.mean()),
        "latency_ms_median": float(np.median(values)),
        "latency_ms_p95": float(np.quantile(values, 0.95)),
        "actions_per_second": float(batch_size * 1000 / values.mean()),
        "latency_ms_samples": milliseconds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--elites", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--host-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if min(args.batch_size, args.horizon, args.candidates, args.iterations,
           args.elites, args.warmup, args.repeats, args.host_repeats) <= 0:
        parser.error("batch sizes, search counts, warmup and timing counts must be positive")
    if args.elites > args.candidates:
        parser.error("--elites cannot exceed --candidates")
    if not torch.cuda.is_available():
        parser.error("CUDA is required; submit this benchmark through mlq")
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
    )
    model_args = model.Args(imagination_horizon=args.horizon)
    agent = model.Agent(envs, model_args).cuda().eval()
    physical = torch.randn(args.batch_size, agent.observation_dim, device="cuda")
    previous = agent.action_low + agent.action_scale * torch.rand(args.batch_size, agent.action_dim, device="cuda")
    observations = torch.cat((physical, previous), dim=-1)

    def direct_control(observations):
        alpha, beta = agent.direct_policy(observations)
        native = Beta(alpha, beta, validate_args=False).rsample().clamp(model.SAMPLE_EPS, 1 - model.SAMPLE_EPS)
        return agent.action_low + agent.action_scale * native

    def cem_iteration(initial_z, initial_previous, mean, std):
        # Independent searches per state. Exactly candidates sequences are scored.
        native = (mean[:, None] + std[:, None] * torch.randn(
            args.batch_size, args.candidates, args.horizon, agent.action_dim,
            device=mean.device, dtype=mean.dtype,
        )).clamp(model.SAMPLE_EPS, 1 - model.SAMPLE_EPS)
        actions = agent.action_low + agent.action_scale * native
        z = initial_z[:, None].expand(-1, args.candidates, -1).reshape(-1, initial_z.shape[-1])
        prev = initial_previous[:, None].expand(-1, args.candidates, -1).reshape(-1, agent.action_dim)
        score = z.new_zeros(args.batch_size * args.candidates)
        discount = torch.ones_like(score)
        for step in range(args.horizon):
            action = actions[:, :, step].reshape(-1, agent.action_dim)
            score = score + discount * agent.predict_reward(z, action, frozen=True)
            discount = discount * model_args.gamma * agent.predict_continuation(z, action, frozen=True)
            z = agent.predict_next(z, action, frozen=True)
            prev = action
        score = score + discount * agent.value_from_latent(z, prev, frozen=True).flatten()
        indices = score.reshape(args.batch_size, args.candidates).topk(args.elites, dim=1).indices
        elite = native.gather(1, indices[:, :, None, None].expand(-1, -1, args.horizon, agent.action_dim))
        return elite.mean(1), elite.std(1, unbiased=False), agent.action_low + agent.action_scale * elite[:, 0, 0]

    compiled_direct = torch.compile(direct_control, mode="reduce-overhead", fullgraph=True)
    compiled_iteration = torch.compile(cem_iteration, mode="reduce-overhead", fullgraph=True)
    compiled_encode = torch.compile(agent.encode, mode="reduce-overhead", fullgraph=True)

    def cem_control():
        z = compiled_encode(observations)
        mean = observations.new_full((args.batch_size, args.horizon, agent.action_dim), 0.5)
        std = torch.full_like(mean, 0.5)
        for _ in range(args.iterations):
            mean, std, action = compiled_iteration(z, previous, mean, std)
        return action

    with torch.inference_mode():
        direct = measure(lambda: compiled_direct(observations), args.warmup, args.repeats, args.batch_size, cuda=True)
        cem = measure(cem_control, args.warmup, args.repeats, args.batch_size, cuda=True)

    host_rows = 16
    host_observations = np.concatenate((
        rng.standard_normal((host_rows, agent.observation_dim)).astype(np.float32),
        rng.uniform(-1, 1, (host_rows, agent.action_dim)).astype(np.float32),
    ), axis=-1)
    host = model.HostIntactActor(agent, host_rows)
    sampler = make_beta_sampler(host_rows, agent.action_dim, envs.single_action_space.low,
                                envs.single_action_space.high, epsilon=model.SAMPLE_EPS)
    host_policy = measure(lambda: host(host_observations), args.warmup, args.host_repeats, host_rows, cuda=False)
    host_control = measure(lambda: sampler(host(host_observations), rng), args.warmup, args.host_repeats, host_rows, cuda=False)
    report = {
        "kind": "untrained_synthetic_model_computational_cost",
        "interpretation": "Computational cost only: no training, control quality, or exact paper latency claim. Same initialized model for direct actor and CEM.",
        "timing_protocol": "Separate warmup (including compilation), synchronized end-to-end wall latency for every CUDA call; host timing excludes GPU work. CEM encodes once per search; no CPU action transfers for either CUDA path.",
        "metadata": {
            "seed": args.seed, "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "dtype": "float32", "tf32": False,
            "compile_mode": "reduce-overhead", "fullgraph": True,
            "observation_dim": agent.observation_dim, "action_dim": agent.action_dim,
            "input_dim": agent.input_dim, "parameter_counts": agent.parameter_counts(),
        },
        "cem_reference": {
            "distribution": "Gaussian over native action sequences, clipped to native Beta support; elite moment refit without smoothing",
            "score": "discounted model reward with predicted nontermination plus terminal critic value",
            "horizon": args.horizon, "candidates_per_iteration_per_state": args.candidates,
            "iterations": args.iterations, "elites": args.elites,
            "scored_sequences_per_state": args.candidates * args.iterations,
            "scored_sequences_per_call": args.batch_size * args.candidates * args.iterations,
            "model_transitions_per_call": args.batch_size * args.candidates * args.iterations * args.horizon,
            "compilation": "one fullgraph compiled scoring/refit iteration repeated by Python; encoder separately compiled",
        },
        "cuda_direct_stochastic_actor": direct,
        "cuda_cem": cem,
        "native_host_logits_16_env": host_policy,
        "native_host_stochastic_actor_16_env": host_control,
        "cem_to_direct_latency_ratio": cem["latency_ms_mean"] / direct["latency_ms_mean"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
