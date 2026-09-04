# Fully streaming falsification benchmark for the Streaming Posterior Filter family.
# One temporally ordered sample is predicted and then consumed exactly once.
# No learner reads batch moments, future samples, replay, or shuffled data.

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


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    method: str = "filter"  # filter | adamw | idbd
    seed: int = 1
    total_steps: int = 500_000
    input_dim: int = 64
    output_dim: int = 16
    regime_steps: int = 100_000
    noise_std: float = 0.1
    outlier_probability: float = 0.01
    outlier_scale: float = 30.0
    temporal_noise: float = 0.15
    log_interval: int = 5_000
    compile: bool = True
    compile_mode: str = "reduce-overhead"

    # AdamW control.
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    # IDBD control.
    idbd_initial_rate: float = 1e-3
    idbd_meta_rate: float = 1e-3

    # Robust diagonal weight filter.
    filter_student_df: float = 5.0
    filter_initial_variance: float = 1.0
    filter_process_variance: float = 1e-6
    filter_noise_rate: float = 1e-3
    filter_q_rate: float = 1e-3
    filter_q_prior: float = 1e-2


def make_stream(args, device):
    generator = torch.Generator(device=device).manual_seed(args.seed)
    steps = args.total_steps
    dims = args.input_dim
    outputs = args.output_dim
    num_regimes = (steps + args.regime_steps - 1) // args.regime_steps

    # A causal temporal signal represented in closed form. Frequencies span slow
    # and fast components; the additive innovation prevents deterministic lookup.
    t = torch.arange(steps, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.empty(dims, device=device).uniform_(
            math.log(2e-4),
            math.log(5e-2),
            generator=generator,
        )
    )
    phases = torch.empty(dims, device=device).uniform_(0.0, 2.0 * torch.pi, generator=generator)
    secondary_phase = torch.empty(dims, device=device).uniform_(
        0.0, 2.0 * torch.pi, generator=generator
    )
    x = torch.sin(t * frequencies + phases)
    x.add_(0.5 * torch.sin(t * (frequencies * 0.173) + secondary_phase))
    x.add_(
        torch.randn((steps, dims), device=device, generator=generator),
        alpha=args.temporal_noise,
    )
    x.mul_(1.0 / (0.5 * (1.0 + 0.25) + args.temporal_noise**2) ** 0.5)

    true_weights = torch.empty((num_regimes, outputs, dims), device=device)
    true_weights[0] = torch.randn((outputs, dims), device=device, generator=generator) / dims**0.5
    for regime in range(1, num_regimes):
        innovation = torch.randn((outputs, dims), device=device, generator=generator) / dims**0.5
        # Abrupt but related task: half of the old mapping persists. This is
        # tracking, not complete relearning from an unrelated random target.
        true_weights[regime] = 0.5 * true_weights[regime - 1] + (0.75**0.5) * innovation

    clean_targets = torch.empty((steps, outputs), device=device)
    for regime in range(num_regimes):
        start = regime * args.regime_steps
        end = min(steps, start + args.regime_steps)
        clean_targets[start:end] = F.linear(x[start:end], true_weights[regime])

    observed_targets = clean_targets + args.noise_std * torch.randn(
        clean_targets.shape,
        device=device,
        generator=generator,
    )
    outlier_mask = torch.rand(
        clean_targets.shape,
        device=device,
        generator=generator,
    ) < args.outlier_probability
    # Student-t_3 corruption, isolated in time and output coordinate.
    numerator = torch.randn(clean_targets.shape, device=device, generator=generator)
    chi = torch.randn((*clean_targets.shape, 3), device=device, generator=generator).square().sum(-1)
    outlier = args.outlier_scale * args.noise_std * numerator / (
        (chi / 3.0).clamp_min(torch.finfo(chi.dtype).tiny).sqrt()
    )
    observed_targets.add_(outlier * outlier_mask)
    return x, clean_targets, observed_targets, outlier_mask


def make_adamw_step(args, device):
    weight = torch.nn.Parameter(torch.zeros((args.output_dim, args.input_dim), device=device))
    optimizer = optim.AdamW(
        [weight],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    empty_diagnostics = torch.zeros(5, device=device)

    def step(x, observed, clean):
        optimizer.zero_grad(set_to_none=True)
        prediction = F.linear(x, weight)
        loss = 0.5 * (prediction - observed).square().sum()
        loss.backward()
        optimizer.step()
        return (prediction.detach() - clean).square().mean(), loss.detach(), empty_diagnostics

    return step, weight, {}


def make_idbd_step(args, device):
    weight = torch.zeros((args.output_dim, args.input_dim), device=device)
    log_rate = torch.full_like(weight, math.log(args.idbd_initial_rate))
    trace = torch.zeros_like(weight)
    empty_diagnostics = torch.zeros(5, device=device)

    def step(x, observed, clean):
        prediction = F.linear(x, weight)
        residual = prediction - observed
        gradient = residual.unsqueeze(1) * x.unsqueeze(0)
        rate = log_rate.exp()
        meta_signal = -(gradient * trace)
        log_rate.add_(meta_signal, alpha=args.idbd_meta_rate)
        rate = log_rate.exp()
        weight.addcmul_(rate, gradient, value=-1.0)
        trace.mul_((1.0 - rate * x.square().unsqueeze(0)).clamp_min(0.0))
        trace.addcmul_(rate, gradient, value=-1.0)
        loss = 0.5 * residual.square().sum()
        return (prediction - clean).square().mean(), loss, empty_diagnostics

    return step, weight, {"log_rate": log_rate, "trace": trace}


def make_filter_step(args, device):
    weight = torch.zeros((args.output_dim, args.input_dim), device=device)
    posterior_var = torch.full_like(weight, args.filter_initial_variance)
    observation_var = torch.full((args.output_dim,), args.noise_std**2, device=device)
    log_q_ratio = torch.zeros((args.output_dim,), device=device)
    nu = args.filter_student_df

    def step(x, observed, clean):
        process_var = args.filter_process_variance * log_q_ratio.exp()
        predicted_var = posterior_var + process_var.unsqueeze(1)
        prediction = F.linear(x, weight)
        residual = observed - prediction
        projected_var = (predicted_var * x.square().unsqueeze(0)).sum(1)
        innovation_var = observation_var + projected_var
        z2 = residual.square() / innovation_var
        student_weight = (nu + 1.0) / (nu + z2)

        effective_noise = observation_var / student_weight
        denominator = effective_noise + projected_var
        gain = predicted_var * x.unsqueeze(0) / denominator.unsqueeze(1)
        weight.add_(gain * residual.unsqueeze(1))
        posterior_var.copy_(
            predicted_var
            - predicted_var.square() * x.square().unsqueeze(0) / denominator.unsqueeze(1)
        )

        post_residual = observed - F.linear(x, weight)
        post_projected_var = (posterior_var * x.square().unsqueeze(0)).sum(1)
        noise_target = student_weight * (post_residual.square() + post_projected_var)
        observation_var.lerp_(noise_target, args.filter_noise_rate)
        observation_var.clamp_min_(torch.finfo(observation_var.dtype).tiny)

        # Predictive Student-t likelihood hypergradient for row-scalar Q. Its
        # influence is slow and bounded: an isolated extreme cannot move Q much,
        # while persistent innovations accumulate across consecutive samples.
        d_nll_d_log_s = 0.5 * (
            1.0 - (nu + 1.0) * residual.square() / (nu * innovation_var + residual.square())
        )
        d_log_s_d_log_q = process_var * x.square().sum() / innovation_var
        q_gradient = d_nll_d_log_s * d_log_s_d_log_q + args.filter_q_prior * log_q_ratio
        log_q_ratio.add_(q_gradient, alpha=-args.filter_q_rate)

        loss = 0.5 * ((nu + 1.0) * torch.log1p(z2 / nu) + innovation_var.log()).sum()
        diagnostics = torch.stack(
            (
                student_weight.mean(),
                z2.mean(),
                posterior_var.mean(),
                observation_var.mean(),
                process_var.mean(),
            )
        )
        return (prediction - clean).square().mean(), loss, diagnostics

    state = {
        "posterior_var": posterior_var,
        "observation_var": observation_var,
        "log_q_ratio": log_q_ratio,
    }
    return step, weight, state


def main():
    args = tyro.cli(Args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.method not in ("filter", "adamw", "idbd"):
        raise ValueError(f"unknown method {args.method}")
    if args.total_steps <= 0 or args.regime_steps <= 0 or args.log_interval <= 0:
        raise ValueError("step counts must be positive")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    run_name = f"StreamingSwitch-v0__{args.exp_name}_{args.method}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join(f"|{key}|{value}|" for key, value in asdict(args).items()),
    )

    x, clean_targets, observed_targets, outlier_mask = make_stream(args, device)
    if args.method == "filter":
        step, weight, state = make_filter_step(args, device)
    elif args.method == "adamw":
        step, weight, state = make_adamw_step(args, device)
    else:
        step, weight, state = make_idbd_step(args, device)
    if args.compile:
        step = torch.compile(step, mode=args.compile_mode)

    error_sum = torch.zeros((), device=device)
    target_energy_sum = torch.zeros((), device=device)
    loss_sum = torch.zeros((), device=device)
    outlier_error_sum = torch.zeros((), device=device)
    clean_error_sum = torch.zeros((), device=device)
    diagnostics_sum = torch.zeros(5, device=device)
    block_count = 0
    block_outliers = torch.zeros((), device=device)

    torch.cuda.synchronize()
    total_start = time.perf_counter()
    steady_start = None
    for index in range(args.total_steps):
        result = step(x[index], observed_targets[index], clean_targets[index])
        clean_error, loss, diagnostics = result
        error_sum.add_(clean_error)
        target_energy_sum.add_(clean_targets[index].square().mean())
        loss_sum.add_(loss)
        is_outlier = outlier_mask[index].any().to(dtype=clean_error.dtype)
        outlier_error_sum.add_(clean_error * is_outlier)
        clean_error_sum.add_(clean_error * (1.0 - is_outlier))
        block_outliers.add_(is_outlier)
        if args.method == "filter":
            diagnostics_sum.add_(diagnostics)
        block_count += 1

        if index + 1 == args.log_interval:
            torch.cuda.synchronize()
            steady_start = time.perf_counter()
        if (index + 1) % args.log_interval == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - total_start
            step_count = index + 1
            block_clean_mse = (error_sum / block_count).item()
            block_target_energy = (target_energy_sum / block_count).item()
            explained_energy = 1.0 - block_clean_mse / max(
                block_target_energy,
                torch.finfo(torch.float32).tiny,
            )
            writer.add_scalar("stream/explained_energy", explained_energy, step_count)
            writer.add_scalar("stream/clean_prequential_mse", block_clean_mse, step_count)
            writer.add_scalar("stream/target_energy", block_target_energy, step_count)
            writer.add_scalar("stream/training_loss", (loss_sum / block_count).item(), step_count)
            nonoutlier_count = (block_count - block_outliers).clamp_min(1)
            outlier_count = block_outliers.clamp_min(1)
            writer.add_scalar(
                "stream/nonoutlier_prequential_mse",
                (clean_error_sum / nonoutlier_count).item(),
                step_count,
            )
            writer.add_scalar(
                "stream/outlier_step_prequential_mse",
                (outlier_error_sum / outlier_count).item(),
                step_count,
            )
            writer.add_scalar("charts/SPS_total", step_count / elapsed, step_count)
            if steady_start is not None and step_count > args.log_interval:
                writer.add_scalar(
                    "charts/SPS_steady",
                    (step_count - args.log_interval) / (time.perf_counter() - steady_start),
                    step_count,
                )
            writer.add_scalar("stream/weight_rms", weight.square().mean().sqrt().item(), step_count)
            writer.add_scalar("stream/outlier_fraction", (block_outliers / block_count).item(), step_count)
            if args.method == "filter":
                diagnostics = diagnostics_sum / block_count
                for name, value in zip(
                    ("student_weight", "z2", "posterior_var", "observation_var", "process_var"),
                    diagnostics,
                ):
                    writer.add_scalar(f"filter/{name}", value.item(), step_count)
                writer.add_scalar("filter/q_ratio", state["log_q_ratio"].exp().mean().item(), step_count)
                diagnostics_sum.zero_()
            elif args.method == "idbd":
                writer.add_scalar("idbd/rate_mean", state["log_rate"].exp().mean().item(), step_count)
                writer.add_scalar("idbd/rate_std", state["log_rate"].exp().std().item(), step_count)

            error_sum.zero_()
            target_energy_sum.zero_()
            loss_sum.zero_()
            outlier_error_sum.zero_()
            clean_error_sum.zero_()
            block_count = 0
            block_outliers.zero_()

    torch.cuda.synchronize()
    total_seconds = time.perf_counter() - total_start
    result = {
        "method": args.method,
        "steps": args.total_steps,
        "seconds": total_seconds,
        "sps_total": args.total_steps / total_seconds,
        "run_name": run_name,
    }
    print(json.dumps(result, sort_keys=True))
    writer.close()


if __name__ == "__main__":
    main()
