"""Fixed-work CUDA measurements of the maintained Beta PPO implementation.

Submit through mlq with --max-parallel-runs 1. This isolates inference, the
actual PPO loss/backward/clipping, and Adam minibatches; it does not simulate an
environment, train an agent to convergence, or report learning scores. Full
FP32/highest precision matches the base trainer. Compiler CUDA graphs use the
production reduce-overhead mode by default; --no-cuda-graphs provides an ablation.
No manual capture is used. Each compiled call begins an independent replay;
parity outputs are cloned before the next replay. GAE separately
covers the default (2048, 1) and equal-batch (128, 16) rollout shapes.
"""

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Beta
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import sample_beta_actions
from cleanrl.shared.ppo_loop import compute_gae as shared_compute_gae
from scripts.benchmark_mujoco_throughput import (
    json_compatible, tensor_difference, timing_summary, write_scalars,
)


METRICS = ("policy_loss", "value_loss", "entropy", "old_approx_kl", "approx_kl", "clip_fraction")
TOLERANCES = {
    "inference": {"atol": 1e-6, "rtol": 1e-5},
    "loss_metrics": {"atol": 2e-6, "rtol": 1e-5},
    "gradients": {"atol": 2e-6, "rtol": 1e-5},
    "parameters": {"atol": 2e-7, "rtol": 2e-6},
    "optimizer_state": {"atol": 2e-7, "rtol": 1e-5},
    # Compiled FP32 recurrence fusion can round differently from eager kernels.
    # Record errors and reject drift beyond this absolute/relative contract.
    "gae": {"atol": 1e-5, "rtol": 1e-5},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--minibatch-sizes", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True,
                        help="use production reduce-overhead compiler graphs (no manual capture)")
    parser.add_argument("--gae-iterations", type=int, default=5,
                        help="calls per GAE timing repetition; separate from model timings")
    parser.add_argument("--gae-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exp-name", default="ppo_base_beta_components_v1")
    args = parser.parse_args()
    if min(*args.num_envs, *args.minibatch_sizes, args.iterations, args.repeats,
           args.gae_iterations, args.gae_repeats) <= 0:
        parser.error("sizes, iterations and repeats must be positive")
    return args


def compile_configuration(cuda_graphs):
    configuration = {"fullgraph": True, "dynamic": False}
    if cuda_graphs:
        configuration["mode"] = "reduce-overhead"
    else:
        configuration["options"] = {"triton.cudagraphs": False}
    return configuration


def compile_fn(fn, *, cuda_graphs):
    compiled = torch.compile(fn, **compile_configuration(cuda_graphs))

    def replay(*args):
        if cuda_graphs:
            # Every call is independent: parity consumers clone outputs, while
            # timed forward-only outputs are immediately discarded. A training
            # caller finishes backward before beginning the next invocation.
            torch.compiler.cudagraph_mark_step_begin()
        return compiled(*args)

    return replay


def measure_cuda(fn, iterations, repeats):
    """Time independent calls; compiled wrappers alone own replay markers.

    Forward-only results are discarded immediately, including their autograd
    graphs. Only repetition boundaries synchronize, never individual calls.
    """
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    wall, stream = [], []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        wall.append(time.perf_counter() - started)
        stream.append(start.elapsed_time(end) / 1000)
    result = timing_summary(wall, iterations)
    result["cuda_stream_seconds"] = stream
    result["cuda_stream_microseconds_per_call"] = statistics.median(stream) * 1e6 / iterations
    return result


def flat_parameters(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()


def flat_gradients(model):
    parameters = list(model.parameters())
    if any(p.grad is None for p in parameters):
        raise AssertionError("PPO fixture must exercise every actor and critic parameter")
    return torch.cat([p.grad.detach().reshape(-1) for p in parameters]).clone()


def difference(reference, candidate, contract):
    return tensor_difference(reference, candidate, **TOLERANCES[contract])


def require_checks(report, save):
    """Save measured error magnitudes before rejecting any numerical result."""
    failures = []

    def visit(value, path=""):
        if not isinstance(value, dict):
            return
        for name, child in value.items():
            key = f"{path}/{name}"
            if name == "checks":
                failures.extend(f"{key}/{label}" for label, passed in child.items() if passed is not True)
            else:
                visit(child, key)

    visit(report)
    report["numerical_gate"] = {"passed": not failures, "failures": failures}
    save()
    if failures:
        raise AssertionError("PPO component numerical gate failed: " + "; ".join(failures[:8]))


def inference_measurements(base, args, model, result, save):
    for n in args.num_envs:
        output = result[str(n)] = {"startup_seconds": {}, "timing": {}, "parity": {},
                                  "compile_configuration": compile_configuration(args.cuda_graphs)}
        obs = torch.randn(n, 17, device="cuda")
        eager = model.get_policy_and_value
        compiled = compile_fn(eager, cuda_graphs=args.cuda_graphs)
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.perf_counter()
            compiled(obs)
            torch.cuda.synchronize()
            output["startup_seconds"]["compiled_inference"] = time.perf_counter() - start
            rng = torch.cuda.get_rng_state()
            expected = tuple(t.clone() for t in eager(obs))
            actual = tuple(t.clone() for t in compiled(obs))
            output["checks"] = {"inference_rng_unchanged": torch.equal(rng, torch.cuda.get_rng_state())}
            for name, a, b in zip(("alpha", "beta", "value"), expected, actual):
                output["parity"][name] = difference(a, b, "inference")
            require_checks(output, save)
            # Sampling remains outside compilation. Inference is numerically
            # checked above, not advertised as an identical stochastic trajectory.
            for label, fn in (("eager", eager), ("compiled", compiled)):
                output["timing"][label] = measure_cuda(lambda: fn(obs), args.iterations, args.repeats)
                save()
            output["speedup"] = (output["timing"]["eager"]["median_seconds"] /
                                  output["timing"]["compiled"]["median_seconds"])


def minibatch_measurements(base, args, initial_model, result, save):
    loss_args = base.Args()
    for size in args.minibatch_sizes:
        row = result[str(size)] = {"variants": {}, "fixture": {}}
        obs = torch.randn(size, 17, device="cuda")
        with torch.no_grad():
            alpha, beta, values = initial_model.get_policy_and_value(obs)
            native, _ = sample_beta_actions(alpha, beta, initial_model.action_low, initial_model.action_high)
            logp = Beta(alpha, beta, validate_args=False).log_prob(native).sum(-1)
            logp = logp - initial_model.action_scale.log().sum()
            # Ratios straddle the clipping threshold; returns are deliberately
            # offset enough to activate clipping of nontrivial critic gradients.
            old_logp = logp + torch.linspace(-0.4, 0.4, size, device="cuda")
            advantages = torch.randn(size, device="cuda") * 2
            returns = values.flatten() + 8 + torch.randn(size, device="cuda")
            old_values = values.flatten() + torch.linspace(-0.5, 0.5, size, device="cuda")
            inputs = tuple(x.detach().clone() for x in (obs, native, old_logp, advantages, returns, old_values))
        row["fixture"].update(batch_size=size, observation_dim=17, action_dim=6,
                              max_grad_norm=loss_args.max_grad_norm, adam_epsilon=1e-5,
                              sampling="Original Beta sampler outside compilation", matched_update_checks=3)
        reference_updates = []
        modes = (("eager_adam_auto", False, None),
                 ("compiled_adam_auto", True, None),
                 ("compiled_adam_fused", True, True))
        for label, use_compile, fused in modes:
            model = copy.deepcopy(initial_model)
            optimizer = torch.optim.Adam(model.parameters(), lr=loss_args.learning_rate,
                                          eps=1e-5, foreach=None, fused=fused)

            def raw_loss(*batch):
                return base.ppo_loss(model, *batch, loss_args)

            loss_fn = compile_fn(raw_loss, cuda_graphs=args.cuda_graphs) if use_compile else raw_loss
            measured = row["variants"][label] = {"compiled": use_compile, "adam_fused": fused,
                                                "adam_foreach": None, "updates": {},
                                                "compile_configuration": (
                                                    compile_configuration(args.cuda_graphs) if use_compile else None)}

            def backward():
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = loss_fn(*inputs)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), loss_args.max_grad_norm, foreach=True)
                return loss.detach(), metrics.detach(), norm.detach()

            def step():
                output = backward()
                optimizer.step()
                return output

            torch.cuda.synchronize()
            start = time.perf_counter()
            for index in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = loss_fn(*inputs)
                loss.backward()
                gradients = flat_gradients(model)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), loss_args.max_grad_norm, foreach=True)
                clipped = flat_gradients(model)
                optimizer.step()
                snapshot = {"loss": loss.detach().clone(), "metrics": metrics.detach().clone(),
                            "gradients": gradients, "clipped_gradients": clipped,
                            "preclip_norm": norm.detach().clone(), "parameters": flat_parameters(model),
                            "optimizer_state": {key: torch.cat([
                                # Noncapturable ordinary Adam keeps its step
                                # counters on CPU; fused Adam uses CUDA. Compare
                                # values on one device outside measured work.
                                optimizer.state[p][key].detach().to(obs.device).reshape(-1)
                                for p in model.parameters()
                            ]).clone() for key in ("step", "exp_avg", "exp_avg_sq")}}
                comparison = measured["updates"][str(index + 1)] = {"checks": {
                    "finite_loss": bool(snapshot["loss"].isfinite()),
                    "finite_parameters": bool(snapshot["parameters"].isfinite().all()),
                    "gradient_clipping_active": bool(norm > loss_args.max_grad_norm),
                }}
                if label == modes[0][0]:
                    reference_updates.append(snapshot)
                else:
                    expected = reference_updates[index]
                    for key, contract in (("loss", "loss_metrics"), ("gradients", "gradients"),
                                           ("clipped_gradients", "gradients"), ("preclip_norm", "gradients"),
                                           ("parameters", "parameters")):
                        comparison[key] = difference(expected[key], snapshot[key], contract)
                    comparison["metrics"] = {name: difference(a, b, "loss_metrics")
                                              for name, a, b in zip(METRICS, expected["metrics"], snapshot["metrics"])}
                    comparison["optimizer_state"] = {
                        key: difference(expected["optimizer_state"][key], value, "optimizer_state")
                        for key, value in snapshot["optimizer_state"].items()}
                if index == 0:
                    torch.cuda.synchronize()
                    measured["startup_first_update_seconds"] = time.perf_counter() - start
                require_checks(measured, save)
                del loss, metrics, snapshot
            measured["loss_forward"] = measure_cuda(lambda: loss_fn(*inputs), args.iterations, args.repeats)
            measured["forward_backward_clipping"] = measure_cuda(backward, args.iterations, args.repeats)
            measured["full_minibatch_update"] = measure_cuda(step, args.iterations, args.repeats)
            measured["checks"] = {"finite_after_fixed_work": bool(flat_parameters(model).isfinite().all())}
            require_checks(measured, save)
            del step, backward, loss_fn, raw_loss, optimizer, model
        baseline = row["variants"][modes[0][0]]["full_minibatch_update"]["median_seconds"]
        for measured in row["variants"].values():
            measured["full_update_speedup_vs_eager_adam"] = baseline / measured["full_minibatch_update"]["median_seconds"]
        save()


@torch.no_grad()
def gae_measurements(base, args, result, save):
    """Compare the public reference recurrence against the shared implementation."""
    settings = base.Args()
    for steps, n in ((2048, 1), (128, 16)):
        row = result[f"T{steps}_N{n}"] = {
            "fixture": {"num_steps": steps, "num_envs": n, "batch_size": steps * n,
                        "gamma": settings.gamma, "gae_lambda": settings.gae_lambda,
                        "timing_iterations": args.gae_iterations,
                        "timing_repeats": args.gae_repeats, "timing_warmup_calls": 10},
            "compile_configuration": compile_configuration(args.cuda_graphs),
            "parity": {}, "timing": {},
        }
        rewards = torch.randn(steps, n, device="cuda")
        values = torch.randn_like(rewards)
        terms = torch.zeros_like(rewards)
        truncs = torch.zeros_like(rewards)
        # Cover continuing transitions, terminal-only, truncated-only, and
        # simultaneous flags (termination must override truncation bootstrap).
        terms[23::37] = 1
        truncs[17::53] = 1
        terms[0] = truncs[0] = 1
        # Keep the single-environment tail continuing; the parallel case
        # exercises both ordinary tail values and final-observation overrides.
        if n > 1:
            truncs[-1, ::2] = 1
        finals = 3 * torch.randn_like(rewards)
        tail = torch.randn(n, device="cuda")
        inputs = (rewards, values, terms, truncs, finals, tail, settings.gamma, settings.gae_lambda)
        row["fixture"].update(
            terminal_only_count=int(((terms == 1) & (truncs == 0)).sum()),
            truncated_only_count=int(((truncs == 1) & (terms == 0)).sum()),
            simultaneous_count=int(((terms == 1) & (truncs == 1)).sum()),
            continuing_count=int(((terms == 0) & (truncs == 0)).sum()),
            continuing_tail_count=int(((terms[-1] == 0) & (truncs[-1] == 0)).sum()),
        )
        row["checks"] = {"finite_inputs": all(bool(t.isfinite().all()) for t in inputs[:6])}
        expected = tuple(t.clone() for t in base.compute_gae(*inputs))
        compiled = compile_fn(shared_compute_gae, cuda_graphs=args.cuda_graphs)
        save()  # Retain the fixture and completed model timings if compilation fails.
        rng = torch.cuda.get_rng_state()
        torch.cuda.synchronize()
        started = time.perf_counter()
        actual = tuple(t.clone() for t in compiled(*inputs))
        torch.cuda.synchronize()
        row["startup_compiled_gae_seconds"] = time.perf_counter() - started
        row["checks"]["gae_rng_unchanged"] = torch.equal(rng, torch.cuda.get_rng_state())
        for name, reference, candidate in zip(("advantages", "returns"), expected, actual):
            row["parity"][name] = difference(reference, candidate, "gae")
        require_checks(row, save)
        for label, fn in (("eager_public_reference", base.compute_gae), ("compiled_shared", compiled)):
            row["timing"][label] = measure_cuda(lambda: fn(*inputs), args.gae_iterations, args.gae_repeats)
            save()
        row["speedup"] = (row["timing"]["eager_public_reference"]["median_seconds"] /
                          row["timing"]["compiled_shared"]["median_seconds"])
        save()


def main():
    args = parse_args()
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    if not torch.cuda.is_available():
        raise RuntimeError("queued CUDA execution required; no CPU model fallback")
    torch.manual_seed(args.seed)
    base = importlib.import_module("cleanrl.ppo_continuous_action")
    run_dir = Path("runs") / f"HalfCheetah-v4__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"kind": "fixed_work_base_ppo_components_not_training", "status": "running", "args": vars(args),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "precision": "FP32/highest/TF32 disabled",
              "compile_configuration": compile_configuration(args.cuda_graphs), "numerical_tolerances": TOLERANCES,
              "source_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
              "shared_gae_source_sha256": hashlib.sha256(
                  Path(importlib.import_module("cleanrl.shared.ppo_loop").__file__).read_bytes()).hexdigest(),
              "inference": {}, "minibatches": {}, "gae": {}}
    writer = SummaryWriter(str(run_dir))

    def save():
        (run_dir / "benchmark.json").write_text(json.dumps(json_compatible(report), indent=2, allow_nan=False) + "\n")
        write_scalars(writer, report["inference"], "benchmark/inference")
        write_scalars(writer, report["minibatches"], "benchmark/minibatches")
        write_scalars(writer, report["gae"], "benchmark/gae")
        writer.flush()

    try:
        writer.add_text("hyperparameters", json.dumps(vars(args), indent=2))
        save()
        spaces = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float64),
                                 single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32))
        model = base.Agent(spaces).cuda()
        inference_measurements(base, args, model, report["inference"], save)
        minibatch_measurements(base, args, model, report["minibatches"], save)
        gae_measurements(base, args, report["gae"], save)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        writer.close()


if __name__ == "__main__":
    main()
