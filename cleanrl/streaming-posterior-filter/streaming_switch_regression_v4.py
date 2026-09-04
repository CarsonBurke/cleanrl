# Bayesian process-noise mixture v4.
# A robust assumed-density diagonal filter bank maintains low/mid/high process
# noise experts. A hazard-mixed Student-t evidence posterior selects stability
# or plasticity per output online; no gradient or hand-coded switch detector.

import json
import math
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter


PERIODS = (17, 19, 23, 29, 31, 37, 43, 47, 53, 59, 67, 73, 83, 97, 109, 127)
FILTER_METHODS = (
    "bayes_mixture",
    "robust_adaptive",
    "robust_fixed",
    "robust_online_r",
    "gaussian_filter",
)
CONDITIONS = (
    "stationary_clean",
    "stationary_outlier",
    "switching_clean",
    "switching_outlier",
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    method: str = "bayes_mixture"
    condition: str = "switching_outlier"
    seed: int = 1
    total_steps: int = 500_000
    input_dim: int = 32
    output_dim: int = 8
    min_regime_steps: int = 20_000
    max_regime_steps: int = 30_000
    noise_std: float = 0.1
    outlier_probability: float = 0.002
    outlier_scale: float = 1.0
    log_interval: int = 8_192
    compile: bool = True
    compile_mode: str = "reduce-overhead"

    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    idbd_initial_rate: float = 0.03
    idbd_meta_rate: float = 1e-3

    filter_student_df: float = 5.0
    filter_initial_variance: float = 1.0
    filter_process_variance: float = 1e-5
    filter_q_rate: float = 1.0
    filter_q_prior: float = 1e-5
    filter_q_min: float = 1e-8
    filter_q_max: float = 1e-3
    filter_r_half_life: float = 20_000.0
    mixture_q_low: float = 1e-8
    mixture_q_mid: float = 1e-5
    mixture_q_high: float = 1e-3
    mixture_hazard: float = 4e-5


def regime_starts(args):
    if args.condition.startswith("stationary"):
        return [0]
    generator = torch.Generator().manual_seed(args.seed + 10_000)
    starts = [0]
    while True:
        dwell = int(
            torch.randint(
                args.min_regime_steps,
                args.max_regime_steps + 1,
                (),
                generator=generator,
            ).item()
        )
        next_start = starts[-1] + dwell
        if next_start >= args.total_steps:
            break
        starts.append(next_start)
    return starts


def make_stream(args, device):
    if args.input_dim != 2 * len(PERIODS):
        raise ValueError(f"input_dim must be {2 * len(PERIODS)} for exact sin/cos excitation")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    steps = args.total_steps
    outputs = args.output_dim
    starts = regime_starts(args)

    t = torch.arange(steps, device=device, dtype=torch.float32).unsqueeze(1)
    periods = torch.tensor(PERIODS, device=device, dtype=torch.float32).unsqueeze(0)
    phases = torch.empty((1, len(PERIODS)), device=device).uniform_(
        0.0,
        2.0 * torch.pi,
        generator=generator,
    )
    angles = 2.0 * torch.pi * t / periods + phases
    pair_scale = (2.0 / args.input_dim) ** 0.5
    x = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).reshape(steps, args.input_dim)
    x.mul_(pair_scale)

    true_weights = torch.randn(
        (len(starts), outputs, args.input_dim),
        device=device,
        generator=generator,
    )
    latent_targets = torch.empty((steps, outputs), device=device)
    gram_min, gram_max = [], []
    for regime, start in enumerate(starts):
        end = starts[regime + 1] if regime + 1 < len(starts) else steps
        segment = x[start:end]
        latent_targets[start:end] = F.linear(segment, true_weights[regime])
        gram = segment.T @ segment / max(1, end - start)
        eigenvalues = torch.linalg.eigvalsh(gram)
        gram_min.append(eigenvalues[0])
        gram_max.append(eigenvalues[-1])

    gaussian_targets = latent_targets + args.noise_std * torch.randn(
        latent_targets.shape,
        device=device,
        generator=generator,
    )
    outlier_mask = torch.zeros_like(latent_targets, dtype=torch.bool)
    if args.condition.endswith("outlier"):
        raw_mask = torch.rand(latent_targets.shape, device=device, generator=generator) < args.outlier_probability
        previous_event = torch.cat(
            (
                torch.zeros(1, device=device, dtype=torch.bool),
                raw_mask[:-1].any(dim=1),
            )
        )
        outlier_mask.copy_(raw_mask & ~previous_event.unsqueeze(1))
        for start in starts[1:]:
            outlier_mask[start] = False
        numerator = torch.randn(latent_targets.shape, device=device, generator=generator)
        chi = torch.randn((*latent_targets.shape, 3), device=device, generator=generator).square().sum(-1)
        corruption = args.outlier_scale * numerator / (
            (chi / 3.0).clamp_min(torch.finfo(chi.dtype).tiny).sqrt()
        )
        observed_targets = gaussian_targets + corruption * outlier_mask
    else:
        observed_targets = gaussian_targets.clone()

    switch_mask = torch.zeros(steps, device=device, dtype=torch.bool)
    if len(starts) > 1:
        switch_mask[torch.tensor(starts[1:], device=device)] = True
    return {
        "x": x,
        "latent": latent_targets,
        "gaussian": gaussian_targets,
        "observed": observed_targets,
        "outlier_mask": outlier_mask,
        "switch_mask": switch_mask,
        "starts": starts,
        "gram_min": torch.stack(gram_min),
        "gram_max": torch.stack(gram_max),
    }


def make_adamw_step(args, device):
    weight = torch.nn.Parameter(torch.zeros((args.output_dim, args.input_dim), device=device))
    optimizer = optim.AdamW(
        [weight],
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    empty = torch.zeros(8, device=device)

    def step(x, observed, latent):
        optimizer.zero_grad(set_to_none=True)
        prediction = F.linear(x, weight)
        loss = 0.5 * (prediction - observed).square().sum()
        loss.backward()
        optimizer.step()
        return (prediction.detach() - latent).square().mean(), prediction.detach(), empty

    return step, weight, {}


def make_idbd_step(args, device):
    weight = torch.zeros((args.output_dim, args.input_dim), device=device)
    log_rate = torch.full_like(weight, math.log(args.idbd_initial_rate))
    trace = torch.zeros_like(weight)
    empty = torch.zeros(8, device=device)

    def step(x, observed, latent):
        prediction = F.linear(x, weight)
        gradient = (prediction - observed).unsqueeze(1) * x.unsqueeze(0)
        rate = log_rate.exp()
        log_rate.add_(-(gradient * trace), alpha=args.idbd_meta_rate)
        rate = log_rate.exp()
        weight.addcmul_(rate, gradient, value=-1.0)
        trace.mul_((1.0 - rate * x.square().unsqueeze(0)).clamp_min(0.0))
        trace.addcmul_(rate, gradient, value=-1.0)
        return (prediction - latent).square().mean(), prediction, empty

    return step, weight, {"log_rate": log_rate}


def make_filter_step(args, device):
    weight = torch.zeros((args.output_dim, args.input_dim), device=device)
    posterior_var = torch.full_like(weight, args.filter_initial_variance)
    observation_var = torch.full((args.output_dim,), args.noise_std**2, device=device)
    eta0 = math.log(args.filter_process_variance)
    log_q = torch.full((args.output_dim,), eta0, device=device)
    log_q_min = math.log(args.filter_q_min)
    log_q_max = math.log(args.filter_q_max)
    nu = args.filter_student_df
    online_r = args.method == "robust_online_r"
    adaptive_q = args.method in ("robust_adaptive", "robust_online_r")
    robust = args.method != "gaussian_filter"
    r_rate = math.log(2.0) / args.filter_r_half_life

    def step(x, observed, latent):
        process_var = log_q.exp()
        predicted_var = posterior_var + process_var.unsqueeze(1)
        prediction = F.linear(x, weight)
        residual = observed - prediction
        feature_square = x.square()
        a_i = predicted_var * feature_square.unsqueeze(0)
        projected_var = a_i.sum(1)
        innovation_var = observation_var + projected_var
        delta = residual.square() / innovation_var
        raw_weight = (nu + 1.0) / (nu + delta)
        student_weight = raw_weight.clamp_max(1.0) if robust else torch.ones_like(raw_weight)
        effective_noise = torch.maximum(
            observation_var,
            observation_var * (nu + delta) / (nu + 1.0),
        ) if robust else observation_var
        denominator = projected_var + effective_noise
        gain = predicted_var * x.unsqueeze(0) / denominator.unsqueeze(1)
        update = gain * residual.unsqueeze(1)
        weight.add_(update)

        # Joseph diagonal. B is the uncertainty in every coordinate except i,
        # plus measurement noise; all terms remain nonnegative.
        b_i = effective_noise.unsqueeze(1) + (projected_var.unsqueeze(1) - a_i).clamp_min(0.0)
        one_minus_kx = 1.0 - gain * x.unsqueeze(0)
        posterior_var.copy_(predicted_var * one_minus_kx.square() + gain.square() * b_i)

        if online_r:
            post_residual = residual * effective_noise / denominator
            post_along_x = projected_var * effective_noise / denominator
            r_target = student_weight * (post_residual.square() + post_along_x)
            observation_var.lerp_(r_target, r_rate)
            observation_var.lerp_(torch.full_like(observation_var, args.noise_std**2), r_rate * 0.01)
            observation_var.clamp_(1e-8, 1.0)

        q_score = 0.5 * (1.0 - raw_weight * delta) * (
            process_var * feature_square.sum() / innovation_var
        ) + args.filter_q_prior * (log_q - eta0)
        if adaptive_q:
            log_q.add_(q_score, alpha=-args.filter_q_rate)
            log_q.clamp_(log_q_min, log_q_max)

        diagnostics = torch.stack(
            (
                student_weight.mean(),
                delta.mean(),
                posterior_var.mean(),
                process_var.mean(),
                (projected_var / denominator).mean(),
                q_score.mean(),
                observation_var.mean(),
                update.square().mean().sqrt(),
            )
        )
        return (prediction - latent).square().mean(), prediction, diagnostics

    state = {
        "posterior_var": posterior_var,
        "observation_var": observation_var,
        "log_q": log_q,
    }
    return step, weight, state


def make_mixture_filter_step(args, device):
    process_vars = torch.tensor(
        (args.mixture_q_low, args.mixture_q_mid, args.mixture_q_high),
        device=device,
    )
    if not torch.all(process_vars > 0):
        raise ValueError("mixture process variances must be positive")
    if not torch.all(process_vars[1:] > process_vars[:-1]):
        raise ValueError("mixture process variances must be strictly increasing")
    if not 0.0 < args.mixture_hazard < 1.0:
        raise ValueError("mixture_hazard must be in (0, 1)")

    experts = process_vars.numel()
    expert_weight = torch.zeros(
        (experts, args.output_dim, args.input_dim),
        device=device,
    )
    posterior_var = torch.full_like(expert_weight, args.filter_initial_variance)
    observation_var = torch.full((args.output_dim,), args.noise_std**2, device=device)
    log_model_prob = torch.full(
        (experts, args.output_dim),
        -math.log(experts),
        device=device,
    )
    mixture_weight = torch.zeros((args.output_dim, args.input_dim), device=device)
    nu = args.filter_student_df

    def step(x, observed, latent):
        posterior_prob = log_model_prob.exp()
        prior_prob = (
            (1.0 - args.mixture_hazard) * posterior_prob
            + args.mixture_hazard / experts
        )
        log_prior_prob = prior_prob.log()
        predicted_var = posterior_var + process_vars.view(experts, 1, 1)
        expert_prediction = torch.einsum("kod,d->ko", expert_weight, x)
        prediction = (prior_prob * expert_prediction).sum(0)
        expert_residual = observed.unsqueeze(0) - expert_prediction
        feature_square = x.square()
        a_i = predicted_var * feature_square.view(1, 1, -1)
        projected_var = a_i.sum(2)
        innovation_var = observation_var.unsqueeze(0) + projected_var
        delta = expert_residual.square() / innovation_var
        log_likelihood = (
            -0.5 * innovation_var.log()
            - 0.5 * (nu + 1.0) * torch.log1p(delta / nu)
        )
        next_log_model_prob = log_prior_prob + log_likelihood
        next_log_model_prob.sub_(torch.logsumexp(next_log_model_prob, dim=0, keepdim=True))
        next_model_prob = next_log_model_prob.exp()

        raw_weight = (nu + 1.0) / (nu + delta)
        student_weight = raw_weight.clamp_max(1.0)
        effective_noise = torch.maximum(
            observation_var.unsqueeze(0),
            observation_var.unsqueeze(0) * (nu + delta) / (nu + 1.0),
        )
        denominator = projected_var + effective_noise
        gain = predicted_var * x.view(1, 1, -1) / denominator.unsqueeze(2)
        update = gain * expert_residual.unsqueeze(2)
        expert_weight.add_(update)

        b_i = effective_noise.unsqueeze(2) + (
            projected_var.unsqueeze(2) - a_i
        ).clamp_min(0.0)
        one_minus_kx = 1.0 - gain * x.view(1, 1, -1)
        posterior_var.copy_(
            predicted_var * one_minus_kx.square() + gain.square() * b_i
        )
        prior_mean_weight = (
            prior_prob.unsqueeze(2) * (expert_weight - update)
        ).sum(0)
        mixture_weight.copy_(
            (next_model_prob.unsqueeze(2) * expert_weight).sum(0)
        )
        log_model_prob.copy_(next_log_model_prob)

        diagnostics = torch.stack(
            (
                (next_model_prob * student_weight).sum(0).mean(),
                (next_model_prob * delta).sum(0).mean(),
                (next_model_prob.unsqueeze(2) * posterior_var).sum(0).mean(),
                (next_model_prob * process_vars.unsqueeze(1)).sum(0).mean(),
                (next_model_prob * projected_var / denominator).sum(0).mean(),
                (
                    -(next_model_prob * next_log_model_prob).sum(0)
                    / math.log(experts)
                ).mean(),
                next_model_prob[-1].mean(),
                (mixture_weight - prior_mean_weight).square().mean().sqrt(),
            )
        )
        return (prediction - latent).square().mean(), prediction, diagnostics

    state = {
        "expert_weight": expert_weight,
        "posterior_var": posterior_var,
        "observation_var": observation_var,
        "log_model_prob": log_model_prob,
        "process_vars": process_vars,
    }
    return step, mixture_weight, state


def summarize_errors(errors, stream):
    summary = {"latent_mse": errors.mean().item()}
    latent_energy = stream["latent"].square().mean().item()
    summary["explained_energy"] = 1.0 - summary["latent_mse"] / latent_energy
    outlier_steps = stream["outlier_mask"].any(dim=1)
    summary["ordinary_mse"] = errors[~outlier_steps].mean().item()
    if outlier_steps.any():
        summary["outlier_step_mse"] = errors[outlier_steps].mean().item()
        indices = torch.nonzero(outlier_steps).squeeze(1)
        for lag in (1, 10, 100, 1000):
            valid = indices + lag < errors.numel()
            if valid.any():
                summary[f"post_outlier_{lag}_mse"] = errors[indices[valid] + lag].mean().item()
    starts = stream["starts"]
    if len(starts) > 1:
        post_switch = []
        for start in starts[1:]:
            post_switch.append(errors[start : min(errors.numel(), start + 2000)])
        summary["post_switch_2k_mse"] = torch.cat(post_switch).mean().item()
    return summary


def main():
    args = tyro.cli(Args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.method not in (*FILTER_METHODS, "adamw", "idbd"):
        raise ValueError(f"unknown method {args.method}")
    if args.condition not in CONDITIONS:
        raise ValueError(f"unknown condition {args.condition}")
    if args.total_steps <= 0 or args.log_interval <= 0:
        raise ValueError("step counts must be positive")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    run_name = (
        f"StreamingSwitch-v4__{args.exp_name}_{args.method}_{args.condition}"
        f"__{args.seed}__{int(time.time())}"
    )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join(f"|{key}|{value}|" for key, value in asdict(args).items()),
    )

    stream = make_stream(args, device)
    writer.add_scalar("stream/gram_eigen_min", stream["gram_min"].min().item(), 0)
    writer.add_scalar("stream/gram_eigen_max", stream["gram_max"].max().item(), 0)
    if args.method == "bayes_mixture":
        step, weight, state = make_mixture_filter_step(args, device)
    elif args.method in FILTER_METHODS:
        step, weight, state = make_filter_step(args, device)
    elif args.method == "adamw":
        step, weight, state = make_adamw_step(args, device)
    else:
        step, weight, state = make_idbd_step(args, device)
    if args.compile:
        step = torch.compile(step, mode=args.compile_mode)

    errors = torch.empty(args.total_steps, device=device)
    latent_sum = torch.zeros((), device=device)
    latent_energy_sum = torch.zeros((), device=device)
    gaussian_sum = torch.zeros((), device=device)
    observed_sum = torch.zeros((), device=device)
    diagnostics_sum = torch.zeros(8, device=device)
    block_count = 0

    torch.cuda.synchronize()
    total_start = time.perf_counter()
    steady_start = None
    for index in range(args.total_steps):
        error, prediction, diagnostics = step(
            stream["x"][index],
            stream["observed"][index],
            stream["latent"][index],
        )
        errors[index] = error
        latent_sum.add_(error)
        latent_energy_sum.add_(stream["latent"][index].square().mean())
        gaussian_sum.add_((prediction - stream["gaussian"][index]).square().mean())
        observed_sum.add_((prediction - stream["observed"][index]).square().mean())
        if args.method in FILTER_METHODS:
            diagnostics_sum.add_(diagnostics)
        block_count += 1

        if index + 1 == args.log_interval:
            torch.cuda.synchronize()
            steady_start = time.perf_counter()
        if (index + 1) % args.log_interval == 0:
            torch.cuda.synchronize()
            step_count = index + 1
            elapsed = time.perf_counter() - total_start
            # Fixed-window prequential explained energy: zero predictor = 0,
            # perfect prediction = 1, and negative means worse than zero.
            block_latent_mse = (latent_sum / block_count).item()
            block_latent_energy = (latent_energy_sum / block_count).item()
            explained_energy = 1.0 - block_latent_mse / max(
                block_latent_energy,
                torch.finfo(torch.float32).tiny,
            )
            writer.add_scalar("stream/explained_energy", explained_energy, step_count)
            writer.add_scalar("stream/latent_prequential_mse", block_latent_mse, step_count)
            writer.add_scalar("stream/latent_energy", block_latent_energy, step_count)
            writer.add_scalar("stream/gaussian_target_mse", (gaussian_sum / block_count).item(), step_count)
            writer.add_scalar("stream/observed_target_mse", (observed_sum / block_count).item(), step_count)
            writer.add_scalar("stream/weight_rms", weight.square().mean().sqrt().item(), step_count)
            writer.add_scalar("charts/SPS_total", step_count / elapsed, step_count)
            if steady_start is not None and step_count > args.log_interval:
                writer.add_scalar(
                    "charts/SPS_steady",
                    (step_count - args.log_interval) / (time.perf_counter() - steady_start),
                    step_count,
                )
            if args.method in FILTER_METHODS:
                values = diagnostics_sum / block_count
                names = (
                    (
                        "student_weight",
                        "delta",
                        "posterior_var",
                        "process_var",
                        "effective_gain",
                        "model_entropy",
                        "high_q_probability",
                        "update_rms",
                    )
                    if args.method == "bayes_mixture"
                    else (
                        "student_weight",
                        "delta",
                        "posterior_var",
                        "process_var",
                        "effective_gain",
                        "q_score",
                        "observation_var",
                        "update_rms",
                    )
                )
                for name, value in zip(names, values):
                    writer.add_scalar(f"filter/{name}", value.item(), step_count)
                if args.method == "bayes_mixture":
                    model_prob = state["log_model_prob"].exp().mean(1)
                    for name, probability in zip(("low", "mid", "high"), model_prob):
                        writer.add_scalar(
                            f"mixture/{name}_q_probability",
                            probability.item(),
                            step_count,
                        )
                else:
                    writer.add_scalar("filter/q_bound_fraction", (
                        (state["log_q"] <= math.log(args.filter_q_min))
                        | (state["log_q"] >= math.log(args.filter_q_max))
                    ).float().mean().item(), step_count)
                diagnostics_sum.zero_()
            elif args.method == "idbd":
                rate = state["log_rate"].exp()
                writer.add_scalar("idbd/rate_mean", rate.mean().item(), step_count)
                writer.add_scalar("idbd/rate_std", rate.std().item(), step_count)

            latent_sum.zero_()
            latent_energy_sum.zero_()
            gaussian_sum.zero_()
            observed_sum.zero_()
            block_count = 0

    torch.cuda.synchronize()
    seconds = time.perf_counter() - total_start
    summary: dict[str, object] = dict(summarize_errors(errors, stream))
    summary.update(
        {
            "method": args.method,
            "condition": args.condition,
            "seed": args.seed,
            "steps": args.total_steps,
            "seconds": seconds,
            "sps_total": args.total_steps / seconds,
            "gram_min": stream["gram_min"].min().item(),
            "gram_max": stream["gram_max"].max().item(),
            "run_name": run_name,
        }
    )
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"summary/{key}", value, args.total_steps)
    print(json.dumps(summary, sort_keys=True))
    writer.close()


if __name__ == "__main__":
    main()
