"""Exact TD7 CUDA graph throughput and mathematical-equivalence benchmark.

Submit through mlq; this file constructs CUDA models and performs GPU work.
Synthetic transitions exercise realistic shapes and the full replay scan. They
are not environment trajectories, and no learning-performance claim follows.
Module paths are benchmarked sequentially to avoid concurrent-model timing.
"""

import argparse
import inspect
import gc
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass


def load_module(path, index):
    name = f"td7_exact_benchmark_{index}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses may look up their owning module
    spec.loader.exec_module(module)
    return module


def make_batch(agent, count, generator):
    """Bounded poses, moderate velocities, and coherent behavior latents."""
    batch = torch.zeros(count, agent.row_dim, device=agent.device)
    fields = agent._fields
    state = torch.randn(count, agent.state_dim, device=agent.device, generator=generator)
    state[:, : min(8, agent.state_dim)] *= 0.2
    state[:, min(8, agent.state_dim) :] *= 1.5
    mu = 0.6 * torch.tanh(torch.randn(count, agent.action_dim, device=agent.device,
                                     generator=generator))
    sigma = 0.05 + 0.15 * torch.rand(count, agent.action_dim, device=agent.device,
                                    generator=generator)
    eps = torch.randn(count, agent.action_dim, device=agent.device, generator=generator)
    u = mu + sigma * eps
    action = u.clamp(-1, 1)
    batch[:, fields["state"]] = state
    batch[:, fields["next_state"]] = state + 0.02 * torch.randn(
        state.shape, device=agent.device, generator=generator)
    for name, value in (("mu", mu), ("sigma", sigma), ("eps", eps),
                        ("pre_action", u), ("action", action)):
        batch[:, fields[name]] = value
    velocity_index = min(8, agent.state_dim - 1)
    batch[:, fields["reward"]] = (0.25 * state[:, velocity_index]
                                    - 0.1 * action.square().sum(-1)).unsqueeze(-1)
    batch[:, fields["not_done"]] = (
        torch.rand(count, 1, device=agent.device, generator=generator) > 0.01).float()
    return batch


def populate(agent):
    generator = torch.Generator(device=agent.device).manual_seed(913)
    with torch.no_grad():
        for start in range(0, agent.args.buffer_size, 32768):
            count = min(32768, agent.args.buffer_size - start)
            agent.storage[start : start + count].copy_(make_batch(agent, count, generator))
        agent.priority.zero_()  # a blocked sampler may pad its priority array
        for start in range(0, agent.args.buffer_size, 32768):
            count = min(32768, agent.args.buffer_size - start)
            # LAP priorities have a unit lower bound and a nonuniform upper tail.
            agent.priority[start : start + count].copy_((0.4 * torch.randn(
                count, device=agent.device, generator=generator).abs()).exp())
        agent.max_priority.copy_(agent.priority.max())
        agent.last.fill_(agent.args.buffer_size - 1)
        agent.size = agent.args.buffer_size
        agent.ptr = 0
        agent.min_target.fill_(-100)
        agent.max_target.fill_(100)
        agent.running_min.fill_(-100)
        agent.running_max.fill_(100)
        if hasattr(agent, "rebuild_priority_blocks"):
            agent.rebuild_priority_blocks()
    agent.current_global_step = 0


def cpu_states(agent):
    names = ("actor", "actor_target", "critic", "critic_target", "encoder",
             "fixed_encoder", "fixed_encoder_target")
    return {name: {key: value.detach().cpu().clone()
                   for key, value in getattr(agent, name).state_dict().items()}
            for name in names}


def effective_alpha(agent):
    module = sys.modules[type(agent).__module__]
    parameter = agent.entropy_alpha if hasattr(agent, "entropy_alpha") else agent.log_alpha
    if hasattr(module, "effective_entropy_alpha"):
        return module.effective_entropy_alpha(parameter, agent.args.entropy_floor)
    return parameter.detach().exp() if getattr(agent.args, "entropy_floor", True) else torch.zeros_like(parameter)


def math_evidence(agent, actor_entropy_coefficient=None):
    """Exercise arbitrary nonzero JVP directions and the exact training loss."""
    gen = torch.Generator(device=agent.device).manual_seed(619)
    batch = make_batch(agent, agent.args.batch_size, gen)
    f = agent._fields
    state, action = batch[:, f["state"]], batch[:, f["action"]]
    with torch.no_grad():
        zs = agent.fixed_encoder.zs(state)
        zsa = agent.fixed_encoder.zsa(zs, action)
        latent = torch.cat([batch[:, f["mu"]], batch[:, f["sigma"]]], -1)
        t_latent = 0.1 * torch.randn(latent.shape, device=agent.device, generator=gen)
        t_pre = 0.1 * torch.randn(action.shape, device=agent.device, generator=gen)
        t_zsa = 0.1 * torch.randn(zsa.shape, device=agent.device, generator=gen)
    critic_args = (state, latent, batch[:, f["pre_action"]], zsa, zs)
    with torch.no_grad():
        values = agent.critic(*critic_args)
    q, tangent = agent.critic(*critic_args, t_latent, t_pre, t_zsa)
    direct_loss = q.square().mean() + tangent.square().mean()
    parameters = tuple(agent.critic.parameters())
    direct_grad = torch.autograd.grad(direct_loss, parameters)
    loss, td, diagnostics, fixed_zs = agent._critic_loss_fn(
        batch, torch.tensor(-100.0, device=agent.device),
        torch.tensor(100.0, device=agent.device))
    training_grad = torch.autograd.grad(loss, parameters)
    out = {"critic_forward": values, "critic_jvp_value": q,
           "critic_jvp_tangent": tangent, "critic_loss": loss,
           "td_errors": td, "diagnostics": diagnostics, "fixed_zs": fixed_zs}
    for name, grad, train_grad in zip(dict(agent.critic.named_parameters()),
                                     direct_grad, training_grad):
        out[f"jvp_gradient/{name}"] = grad
        out[f"training_gradient/{name}"] = train_grad
    torch.manual_seed(197)
    entropy_parameter = hasattr(agent, "log_alpha") or hasattr(agent, "entropy_alpha")
    coefficient = (effective_alpha(agent) if actor_entropy_coefficient is None else torch.tensor(actor_entropy_coefficient, device=agent.device)) if entropy_parameter else None
    actor_args = (state, fixed_zs, coefficient) if entropy_parameter else (state, fixed_zs)
    # Native Hermite versions retain the previous point actor as an explicit
    # equivalence control. Captured training still uses their configured actor.
    actor_advantage = getattr(agent.args, "actor_advantage", None)
    try:
        if actor_advantage is not None:
            agent.args.actor_advantage = "point"
        if "eps" in inspect.signature(agent._actor_loss_fn).parameters:
            mu_shape = (agent.args.actor_samples // 2, state.shape[0], agent.action_dim)
            eps = torch.randn(mu_shape, device=state.device)
            actor_args += (torch.cat([eps, -eps]),)
        actor_loss, log_std, actor_diagnostics = agent._actor_loss_fn(*actor_args)
    finally:
        if actor_advantage is not None:
            agent.args.actor_advantage = actor_advantage
    actor_grads = torch.autograd.grad(actor_loss, tuple(agent.actor.parameters()))
    diagnostic_names = getattr(sys.modules[type(agent).__module__], "DELIGHT_DIAGNOSTICS", None)
    if diagnostic_names is not None:
        indices = {name: index for index, name in enumerate(diagnostic_names)}
        common = [actor_diagnostics[indices[name]] for name in (
            "gate_mean", "gate_suppressed", "gate_saturated", "advantage_mean", "advantage_rms", "surprisal_mean")]
        common += [actor_diagnostics[indices["negative_surprisal_fraction"]] if "negative_surprisal_fraction" in indices else torch.zeros_like(actor_diagnostics[0]), actor_diagnostics[indices["clipped_action_fraction"]]]
        actor_diagnostics = torch.stack(common)
    out.update(actor_loss=actor_loss, actor_log_std=log_std,
               actor_diagnostics=actor_diagnostics[:8])  # shared gate metrics; reference diagnostics vary by method
    for name, grad in zip(dict(agent.actor.named_parameters()), actor_grads):
        out[f"actor_gradient/{name}"] = grad
    return {name: value.detach().cpu().clone() for name, value in out.items()}


def compare(reference, candidate, atol, rtol):
    if reference.keys() != candidate.keys():
        raise AssertionError("Equivalence evidence keys differ")
    errors = {}
    for name, expected in reference.items():
        actual = candidate[name]
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol,
                                   msg=lambda msg: f"{name}: {msg}")
        errors[name] = float((actual - expected).abs().max())
    return {"passed": True, "atol": atol, "rtol": rtol,
            "max_absolute_errors": errors}


def hermite_evidence(module, agent, atol, rtol):
    """Validate native geometry against the critic's trained JVP and score boundary."""
    if not hasattr(agent, "_hermite_actor_loss_fn"):
        return None
    gen = torch.Generator(device=agent.device).manual_seed(927)
    state = make_batch(agent, agent.args.batch_size, gen)[:, agent._fields["state"]]
    with torch.no_grad():
        zs = agent.fixed_encoder.zs(state)
        mu, log_std = agent.actor(state, zs)
        sigma = log_std.exp()
        (grad_mu, grad_sigma), center = torch.func.grad(
            agent._center_value_fn, argnums=(0, 1), has_aux=True,
        )(mu, sigma, state, zs)
        eps = torch.randn(mu.shape, device=agent.device, generator=gen)
        native, _, _ = module.hermite_advantage(eps, sigma, grad_mu, grad_sigma)
        t_mu = sigma * eps
        t_sigma = 0.5 * sigma * (eps.square() - 1)
        zsa, t_zsa = agent.fixed_encoder.zsa_jvp(zs, mu, t_mu)
        value, tangent = agent.critic(
            state, torch.cat([mu, sigma], -1), mu, zsa, zs,
            torch.cat([t_mu, t_sigma], -1), t_mu, t_zsa,
        )
        torch.testing.assert_close(center, value.mean(-1), atol=atol, rtol=rtol)
        torch.testing.assert_close(native, tangent.mean(-1), atol=atol, rtol=rtol)
        tangent_error = float((native - tangent.mean(-1)).abs().max())

    actor_parameters = tuple(agent.actor.parameters())
    other_parameters = tuple(agent.critic.parameters()) + tuple(agent.fixed_encoder.parameters())
    # Fixed encoder parameters are frozen by the trainer. Check every trainable
    # parameter outside the actor without asking autograd about frozen tensors.
    other_parameters = tuple(p for p in other_parameters if p.requires_grad)
    torch.manual_seed(457)
    loss, _, diagnostics = agent._hermite_actor_loss_fn(state, zs, effective_alpha(agent))
    gradients = torch.autograd.grad(loss, actor_parameters + other_parameters, allow_unused=True)
    if any(g is None for g in gradients[:len(actor_parameters)]):
        raise AssertionError("Native score actor has an unused actor parameter")
    if any(g is not None for g in gradients[len(actor_parameters):]):
        raise AssertionError("Native score actor differentiates critic/encoder geometry")
    original_bias = agent.critic.q3.bias.detach().clone()
    try:
        with torch.no_grad():
            agent.critic.q3.bias.add_(100)
        torch.manual_seed(457)
        shifted_loss, _, shifted_diagnostics = agent._hermite_actor_loss_fn(
            state, zs, effective_alpha(agent))
        shifted_gradients = torch.autograd.grad(shifted_loss, actor_parameters)
        torch.testing.assert_close(shifted_loss, loss, atol=0, rtol=0)
        torch.testing.assert_close(shifted_diagnostics[:8], diagnostics[:8], atol=0, rtol=0)
        for actual, expected in zip(shifted_gradients, gradients[:len(actor_parameters)]):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        with torch.no_grad():
            agent.critic.q3.bias.copy_(original_bias)
    return {"passed": True, "max_native_vs_trained_jvp_error": tangent_error,
            "score_only_actor_gradient": True, "constant_offset_invariance": True,
            "actor_samples": agent.args.actor_samples,
            "scope": "Native center derivatives, including preaction and SALE paths; not true-Q accuracy"}


def measure(agent, steps, repeats, use_train):
    for i in range(100):
        if use_train:
            agent.train()
        else:
            agent.graphs[int((i + 1) % agent.args.policy_freq == 0)].replay()
    torch.cuda.synchronize()
    samples = []
    for repeat in range(repeats):
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start_event.record()
        for i in range(steps):
            if use_train:
                agent.train()
            else:
                step = repeat * steps + i + 1
                agent.graphs[int(step % agent.args.policy_freq == 0)].replay()
        end_event.record()
        end_event.synchronize()
        wall_ms = 1000 * (time.perf_counter() - wall_start)
        device_ms = start_event.elapsed_time(end_event)
        samples.append({"wall_us_per_step": wall_ms * 1000 / steps,
                        "cuda_interval_us_per_step": device_ms * 1000 / steps})
    wall_median = statistics.median(sample["wall_us_per_step"] for sample in samples)
    return {"steps_per_repeat": steps, "repeats": repeats, "samples": samples,
            "median_wall_us_per_step": wall_median,
            "median_steps_per_second": 1e6 / wall_median}


def reset_optimizer_state(agent):
    # Preserve captured tensor addresses. load_state_dict would replace storage
    # that the graphs still reference. Adam's initial moments/step are all zero.
    with torch.no_grad():
        for name in ("encoder_optimizer", "critic_optimizer", "actor_optimizer", "alpha_optimizer"):
            if not hasattr(agent, name):
                continue
            for state in getattr(agent, name).state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        value.zero_()


def point_probe_evidence(module, agent):
    """Validate the finite-batch actor-gradient probe without changing learner state."""
    if not hasattr(agent, "gradient_probe"):
        return None
    models = [getattr(agent, name) for name in module.CHECKPOINT_MODELS]
    parameters = [p for model in models for p in model.parameters()]
    weights = [p.detach().clone() for p in parameters]
    gradients = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    host_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(agent.device)
    metrics = agent.gradient_probe()
    if not torch.isfinite(metrics).all():
        raise AssertionError("Point-DG gradient probe produced nonfinite metrics")
    for before, parameter, gradient in zip(weights, parameters, gradients):
        torch.testing.assert_close(parameter, before, atol=0, rtol=0)
        if gradient is None:
            if parameter.grad is not None:
                raise AssertionError("Gradient probe populated a model .grad")
        else:
            torch.testing.assert_close(parameter.grad, gradient, atol=0, rtol=0)
    if not torch.equal(host_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state(agent.device)):
        raise AssertionError("Gradient probe changed the training random stream")
    names = dict(zip(module.GRADIENT_PROBE, metrics.tolist()))
    return {"passed": True, "no_weight_gradient_or_rng_mutation": True, "metrics": names,
            "scope": "Same saved training directions, current weights; finite-batch gradients, not population"}


def checkpoint_evidence(module, agent, output):
    """Serialized learner restore must reproduce the next captured update exactly."""
    if not hasattr(agent, "checkpoint_state"):
        return None
    path = (output.parent if output else ROOT / "benchmarks/td7_delight_cubature_v1") / "checkpoint_validation.pt"
    rng = np.random.default_rng(71)
    agent.training_steps = agent.args.target_update_rate - 1

    def differences(left, right, name):
        failures = {}
        if isinstance(left, torch.Tensor):
            if left.shape != right.shape or left.dtype != right.dtype:
                failures[name] = {"left_shape": list(left.shape), "right_shape": list(right.shape),
                                  "left_dtype": str(left.dtype), "right_dtype": str(right.dtype)}
            elif not torch.equal(left, right):
                delta = (left.to(torch.float64) - right.to(torch.float64)).abs()
                failures[name] = {"max_absolute_error": float(delta.max()),
                                  "mismatched_elements": int((left != right).sum()),
                                  "elements": left.numel()}
        elif isinstance(left, dict):
            if left.keys() != right.keys():
                failures[name] = {"left_keys": list(left), "right_keys": list(right)}
            else:
                for key in left:
                    failures.update(differences(left[key], right[key], f"{name}/{key}"))
        elif isinstance(left, (tuple, list)):
            if len(left) != len(right):
                failures[name] = {"left_length": len(left), "right_length": len(right)}
            else:
                for index, (a, b) in enumerate(zip(left, right)):
                    failures.update(differences(a, b, f"{name}/{index}"))
        elif isinstance(left, np.ndarray):
            if not np.array_equal(left, right):
                failures[name] = "numpy array differs"
        elif left != right:
            failures[name] = {"left": left, "right": right}
        return failures

    def addresses():
        result = {}
        for name in module.CHECKPOINT_MODELS:
            for key, value in getattr(agent, name).state_dict().items():
                result[f"models/{name}/{key}"] = value.data_ptr()
        for name in module.CHECKPOINT_OPTIMIZERS:
            for index, state in enumerate(getattr(agent, name).state.values()):
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        result[f"optimizers/{name}/{index}/{key}"] = value.data_ptr()
        multiplier_name = "entropy_alpha" if hasattr(agent, "entropy_alpha") else "log_alpha"
        for name in (*module.CHECKPOINT_BUFFERS, "storage", "priority", multiplier_name):
            result[name] = getattr(agent, name).data_ptr()
        return result

    def priority_fingerprint(before, after):
        changed = torch.nonzero(before != after).flatten()
        return changed, {"changed_rows": changed.numel(), "first_changed_indices": changed[:32].tolist()}

    def roundtrip(label, require_exact_update):
        module.save_checkpoint(path, agent, rng, 163840, 10)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        module.validate_checkpoint_args(checkpoint, agent.args)
        captured_addresses = addresses()
        agent.train()
        expected = agent.checkpoint_state()
        expected_rng = module.capture_rng_state(rng)
        agent.restore_checkpoint_state(checkpoint["agent"])
        module.restore_rng_state(checkpoint["rng"], rng)
        restored = agent.checkpoint_state()
        restored_rng = module.capture_rng_state(rng)
        pre_state_errors = differences(checkpoint["agent"], restored, "restored-state")
        pre_rng_errors = differences(checkpoint["rng"], restored_rng, "restored-rng")
        address_errors = differences(captured_addresses, addresses(), "captured-addresses")
        agent.train()
        actual = agent.checkpoint_state()
        actual_rng = module.capture_rng_state(rng)
        next_errors = differences(expected, actual, "next-update")
        next_rng_errors = differences(expected_rng, actual_rng, "next-rng")
        expected_indices, expected_fp = priority_fingerprint(checkpoint["agent"]["priority"], expected["priority"])
        actual_indices, actual_fp = priority_fingerprint(checkpoint["agent"]["priority"], actual["priority"])
        diagnostics = {"restored_state_errors": pre_state_errors, "restored_rng_errors": pre_rng_errors,
                       "captured_address_errors": address_errors, "next_update_errors": next_errors,
                       "next_rng_errors": next_rng_errors,
                       "priority_sampler": {"expected": expected_fp, "actual": actual_fp,
                                            "same_changed_indices": torch.equal(expected_indices, actual_indices),
                                            "expected_only_indices": sorted(set(expected_indices.tolist()) - set(actual_indices.tolist())),
                                            "actual_only_indices": sorted(set(actual_indices.tolist()) - set(expected_indices.tolist()))},
                       "models_max_absolute_errors": {
                           name: max((float((expected["models"][name][key].to(torch.float64) -
                                             actual["models"][name][key].to(torch.float64)).abs().max())
                                      for key in expected["models"][name]), default=0.0)
                           for name in module.CHECKPOINT_MODELS}}
        diagnostics["phase"] = label
        print("checkpoint_diagnostics=" + json.dumps(diagnostics, sort_keys=True), flush=True)
        if pre_state_errors or pre_rng_errors or address_errors or next_rng_errors:
            raise AssertionError("Checkpoint learner/RNG/address identity failed; see checkpoint_diagnostics above")
        if require_exact_update and next_errors:
            raise AssertionError("Checkpoint exact-next-step identity failed; see checkpoint_diagnostics above")
        return checkpoint, diagnostics

    def sampler_probe():
        """Separate the original CDF scan from the saved CUDA random draws."""
        saved_rng = torch.cuda.get_rng_state_all()
        cdf = torch.empty_like(agent.priority)
        uniforms = torch.empty(agent.args.batch_size, device=agent.device)
        indices = torch.empty(agent.args.batch_size, device=agent.device, dtype=torch.long)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                cdf.copy_(torch.cumsum(agent.priority, 0))
                uniforms.copy_(torch.rand(agent.args.batch_size, device=agent.device))
                indices.copy_(torch.minimum(torch.searchsorted(cdf, uniforms * cdf.gather(0, agent.last)), agent.last))
            torch.cuda.set_rng_state_all(saved_rng)
            graph.replay()
            first = (cdf.cpu().clone(), uniforms.cpu().clone(), indices.cpu().clone())
            torch.cuda.set_rng_state_all(saved_rng)
            graph.replay()
            second = (cdf.cpu().clone(), uniforms.cpu().clone(), indices.cpu().clone())
            changed = torch.nonzero(first[2] != second[2]).flatten()
            return {"uniform_draws_identical": torch.equal(first[1], second[1]),
                    "cdf_identical": torch.equal(first[0], second[0]),
                    "cdf_max_absolute_error": float((first[0] - second[0]).abs().max()),
                    "sample_indices_identical": torch.equal(first[2], second[2]),
                    "changed_sample_positions": changed.tolist(),
                    "first_indices_at_changes": first[2][changed].tolist(),
                    "second_indices_at_changes": second[2][changed].tolist()}
        finally:
            torch.cuda.synchronize()
            torch.cuda.set_rng_state_all(saved_rng)

    try:
        sampler = sampler_probe()
        print("checkpoint_sampler_probe=" + json.dumps(sampler, sort_keys=True), flush=True)
        base, nonuniform = roundtrip("nonuniform-original-priority", False)
        # The original CUDA FP32 parallel prefix scan is nondeterministic. Keep
        # its weighted state restore proof separate from next-update identity.
        # Integers <= 1M sum exactly in FP32, independent of scan association;
        # this fixture exercises the SAME captured graph and complete update.
        agent.restore_checkpoint_state(base["agent"])
        module.restore_rng_state(base["rng"], rng)
        del base
        with torch.no_grad():
            agent.priority.zero_()
            agent.priority[:agent.size].fill_(1)
            agent.max_priority.fill_(1)
        if hasattr(agent, "rebuild_priority_blocks"):
            agent.rebuild_priority_blocks()
        uniform_base, uniform = roundtrip("uniform-exact-prefix-priority", True)
        del uniform_base
        return {"passed": True, "exact_serialized_nonuniform_state": True,
                "exact_next_update_uniform_prefix": True,
                "nonuniform_next_update_exact": not bool(nonuniform["next_update_errors"]),
                "sampler_probe": sampler, "nonuniform_diagnostics": nonuniform, "uniform_diagnostics": uniform,
                "scope": "Exact serialized learner/replay/Adam/alpha/RNG/addresses; exact next captured update with uniform FP32 prefix sums, including hard targets; excludes environment trajectory",
                "limitation": "Original nonuniform CUDA FP32 parallel CDF may vary by scan association; its next update is reported separately without a bitwise requirement"}

    finally:
        path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modules", nargs="+", type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=17)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--actor-objective", choices=("pg", "delight", "pathwise"), default="pg")
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-entropy-floor", action="store_true")
    parser.add_argument("--target-smoothing-sigma", type=float)
    parser.add_argument("--gate-scale", choices=("rms", "raw"))
    parser.add_argument("--gate-scope", choices=("mean", "all"))
    parser.add_argument("--actor-baseline", choices=("latent", "modal", "cubature"))
    parser.add_argument("--actor-advantage", choices=("point", "hermite"))
    parser.add_argument("--surprisal", choices=("relative", "density"))
    parser.add_argument("--math-actor-entropy-coefficient", type=float,
                        help="Fixed coefficient for actor math comparison only; captured controller remains configured")
    cli = parser.parse_args()
    if min(cli.steps, cli.repeats, cli.batch_size, cli.buffer_size,
           cli.state_dim, cli.action_dim) < 1:
        parser.error("sizes, steps and repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; submit this benchmark through mlq")
    result = {"scope": "Synthetic fixed-shape GPU benchmark; not a training evaluation",
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "configuration": vars(cli).copy(),
              "modules": []}
    result["configuration"]["modules"] = list(map(str, cli.modules))
    result["configuration"]["output"] = str(cli.output) if cli.output else None
    reference_states = reference_math = None
    for index, path in enumerate(cli.modules):
        print(f"Building {path}", flush=True)
        module = load_module(path, index)
        module.configure_runtime()
        torch.manual_seed(1)
        torch.cuda.reset_peak_memory_stats()
        args = module.Args(compile=True, buffer_size=cli.buffer_size,
                           batch_size=cli.batch_size, actor_objective=cli.actor_objective)
        if cli.no_entropy_floor:
            if not hasattr(args, "entropy_floor"):
                raise ValueError("This trainer does not expose an entropy-floor ablation")
            args.entropy_floor = False
        if cli.target_smoothing_sigma is not None:
            args.target_smoothing_sigma = cli.target_smoothing_sigma
        if cli.gate_scale is not None:
            args.gate_scale = cli.gate_scale
        if cli.gate_scope is not None:
            args.gate_scope = cli.gate_scope
        if cli.surprisal is not None:
            if not hasattr(args, "surprisal") and cli.surprisal != "relative":
                raise ValueError("This trainer does not expose density surprisal")
            if hasattr(args, "surprisal"):
                args.surprisal = cli.surprisal
        if cli.actor_advantage is not None and hasattr(args, "actor_advantage"):
            args.actor_advantage = cli.actor_advantage
        # A candidate may additionally offer a new scientific objective. Timing
        # and equivalence here compare the original point-Q objective instead.
        if hasattr(args, "actor_query"):
            args.actor_query = "point"
        if hasattr(args, "actor_baseline"):
            args.actor_baseline = cli.actor_baseline or "cubature"
        if hasattr(args, "target_std"):
            args.target_std = "fixed"
        build_start = time.perf_counter()
        agent = module.TD7Agent(cli.state_dim, cli.action_dim, args,
                                torch.device("cuda"), NullWriter())
        torch.cuda.synchronize()
        build_seconds = time.perf_counter() - build_start
        if reference_states is None:
            reference_states = cpu_states(agent)
        else:
            for name, state in reference_states.items():
                getattr(agent, name).load_state_dict(state)
        evidence = math_evidence(agent, cli.math_actor_entropy_coefficient)
        if reference_math is None:
            reference_math = evidence
            equivalence = {"reference": True}
        else:
            equivalence = compare(reference_math, evidence, cli.atol, cli.rtol)
        native = hermite_evidence(module, agent, cli.atol, cli.rtol)
        populate(agent)
        graph = measure(agent, cli.steps, cli.repeats, use_train=False)
        # Reset mutable training data/parameters before measuring the API cadence.
        for name, state in reference_states.items():
            getattr(agent, name).load_state_dict(state)
        reset_optimizer_state(agent)
        if hasattr(agent, "log_alpha"):
            with torch.no_grad():
                agent.log_alpha.zero_()
        elif hasattr(agent, "entropy_alpha"):
            with torch.no_grad():
                agent.entropy_alpha.fill_(args.initial_alpha)
        populate(agent)
        train = measure(agent, cli.steps, cli.repeats, use_train=True)
        point_probe = point_probe_evidence(module, agent)
        checkpoint = checkpoint_evidence(module, agent, cli.output)
        entry = {"path": str(path), "trainer_args": vars(args),
                 "build_capture_seconds": build_seconds,
                 "equivalence": equivalence, "cuda_graph_mixed_cadence": graph,
                 "native_hermite_validation": native,
                 "point_gradient_probe_validation": point_probe,
                 "train_api_with_targets_and_logging_cadence": train,
                 "checkpoint_restore": checkpoint,
                 "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved()}
        result["modules"].append(entry)
        print(json.dumps(entry), flush=True)
        del agent, evidence, module
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
    output = json.dumps(result, indent=2)
    if cli.output:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        cli.output.write_text(output + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
