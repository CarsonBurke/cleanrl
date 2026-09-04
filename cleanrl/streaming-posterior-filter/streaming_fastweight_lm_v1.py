# Fully streaming byte-language carrier for the Streaming Posterior Filter family.
# One byte is predicted, scored, and consumed once. Delta fast weights update in
# the forward path; no token batching, replay, future context, or BPTT is used.

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter


METHODS = ("fast_robust", "slow_robust", "fast_ce", "slow_ce")
CONDITIONS = ("clean", "corrupted")


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    corpus_path: str = str(Path(__file__).with_name("shakespeare_corpus.txt"))
    method: str = "fast_robust"
    condition: str = "corrupted"
    seed: int = 1
    total_steps: int = 300_000
    dim: int = 64
    hidden_dim: int = 128
    fast_heads: int = 4
    corruption_probability: float = 0.002
    validation_fraction: float = 0.2
    validation_steps: int = 4_096
    inference_rows: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    contamination_probability: float = 0.002
    student_df: float = 5.0
    fast_eta_max: float = 0.5
    scale_rate: float = 1e-3
    latent_coef: float = 0.1
    write_coef: float = 0.1
    log_interval: int = 4_096
    compile: bool = True
    compile_mode: str = "default"


class StreamingFastWeightLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.dim % args.fast_heads != 0:
            raise ValueError("dim must be divisible by fast_heads")
        self.dim = args.dim
        self.heads = args.fast_heads
        self.head_dim = args.dim // args.fast_heads
        self.eta_max = args.fast_eta_max
        self.embedding = nn.Embedding(256, args.dim)
        self.in_norm = nn.RMSNorm(args.dim)
        self.recurrent = nn.GRUCell(args.dim, args.dim)
        self.slow = nn.Sequential(
            nn.Linear(args.dim, args.hidden_dim),
            nn.SiLU(),
            nn.Linear(args.hidden_dim, args.dim),
        )
        self.slow_norm = nn.RMSNorm(args.dim)
        self.query = nn.Linear(args.dim, args.dim, bias=False)
        self.key = nn.Linear(args.dim, args.dim, bias=False)
        self.value = nn.Linear(args.dim, args.dim, bias=False)
        self.eta = nn.Linear(args.dim, args.fast_heads)
        self.fast_out = nn.Linear(args.dim, args.dim, bias=False)
        self.gate = nn.Linear(args.dim, args.dim)
        self.latent_predictor = nn.Linear(args.dim, args.dim)
        self.logit_bias = nn.Parameter(torch.zeros(256))

    def initial_memory(self, device):
        return torch.zeros(
            (self.heads, self.head_dim, self.head_dim),
            device=device,
        )

    def initial_hidden(self, device):
        return torch.zeros(self.dim, device=device)

    def forward_step(
        self,
        token,
        observed_target,
        memory,
        hidden,
        fast_scale,
        use_fast,
        robust,
        student_df,
    ):
        embedding = self.embedding(token)
        recurrent = self.recurrent(self.in_norm(embedding), hidden)
        slow = self.slow_norm(recurrent + self.slow(recurrent))
        query = F.normalize(self.query(slow).view(self.heads, self.head_dim), dim=-1)
        if use_fast:
            fast_read = torch.einsum("hde,he->hd", memory, query).reshape(self.dim)
            combined = slow + torch.sigmoid(self.gate(slow)) * self.fast_out(fast_read)
        else:
            combined = slow
        combined = F.rms_norm(combined, (self.dim,))
        logits = F.linear(combined, self.embedding.weight, self.logit_bias)
        latent_prediction = self.latent_predictor(combined)

        if not use_fast:
            zero = logits.new_zeros(())
            return logits, latent_prediction, memory, recurrent, zero, zero, zero, zero

        key = F.normalize(self.key(slow).view(self.heads, self.head_dim), dim=-1)
        target_embedding = self.embedding(observed_target)
        value = self.value(target_embedding).view(self.heads, self.head_dim)
        predicted_value = torch.einsum("hde,he->hd", memory, key)
        innovation = value - predicted_value
        z2 = innovation.square().mean(dim=1) / fast_scale
        student_weight = (
            ((student_df + 1.0) / (student_df + z2)).clamp_max(1.0)
            if robust
            else torch.ones_like(z2)
        )
        fast_scale_target = (student_weight * innovation.square().mean(dim=1)).mean()
        eta = self.eta_max * torch.sigmoid(self.eta(slow))
        update_scale = (eta * student_weight).view(self.heads, 1, 1)
        candidate = memory + update_scale * innovation.unsqueeze(2) * key.unsqueeze(1)
        post_innovation = value - torch.einsum("hde,he->hd", candidate, key)
        return (
            logits,
            latent_prediction,
            candidate,
            recurrent,
            fast_scale_target,
            post_innovation.square().mean(),
            student_weight.mean(),
            eta.mean(),
        )


def make_stream(args, device):
    corpus = Path(args.corpus_path).read_bytes()
    if len(corpus) < 1024:
        raise ValueError("corpus is too small")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    split = int(len(corpus) * (1.0 - args.validation_fraction))
    training_base = torch.tensor(bytearray(corpus[:split]), device=device, dtype=torch.long)
    validation = torch.tensor(bytearray(corpus[split:]), device=device, dtype=torch.long)
    if validation.numel() < args.validation_steps + 1:
        raise ValueError("validation split is shorter than validation_steps + 1")
    repeats = math.ceil((args.total_steps + 1) / training_base.numel())
    clean = training_base.repeat(repeats)[: args.total_steps + 1]
    observed = clean.clone()
    corruption = torch.zeros_like(clean, dtype=torch.bool)
    if args.condition == "corrupted":
        generator = torch.Generator(device=device).manual_seed(args.seed + 1_000)
        raw = torch.rand(clean.shape, device=device, generator=generator) < args.corruption_probability
        previous = torch.cat((torch.zeros(1, device=device, dtype=torch.bool), raw[:-1]))
        corruption.copy_(raw & ~previous)
        random_bytes = torch.randint(0, 256, clean.shape, device=device, generator=generator)
        observed.copy_(torch.where(corruption, random_bytes, clean))
    return clean, observed, corruption, validation, training_base.numel()


@torch.no_grad()
def evaluate_heldout(model, validation, args, device, calibrated_fast_scale):
    use_fast = args.method.startswith("fast")
    robust = args.method.endswith("robust")
    steps = args.validation_steps
    rows = min(args.inference_rows, steps)
    memory = model.initial_memory(device)
    hidden = model.initial_hidden(device)
    fast_scale = calibrated_fast_scale.detach().clone()
    nll_sum = torch.zeros((), device=device)
    top1_correct = torch.zeros((), device=device)
    top5_correct = torch.zeros((), device=device)
    truth_ids = torch.empty(rows, device=device, dtype=torch.long)
    top_ids = torch.empty((rows, 5), device=device, dtype=torch.long)
    top_probs = torch.empty((rows, 5), device=device)

    model.eval()
    for index in range(steps):
        target = validation[index + 1]
        (
            logits,
            _,
            candidate,
            next_hidden,
            fast_scale_target,
            _,
            _,
            _,
        ) = model.forward_step(
            validation[index],
            target,
            memory,
            hidden,
            fast_scale,
            use_fast,
            robust,
            args.student_df,
        )
        log_probs = F.log_softmax(logits, dim=-1)
        probabilities = log_probs.exp()
        probabilities_top, predictions_top = probabilities.topk(5)
        nll_sum.add_(-log_probs[target])
        top1_correct.add_((predictions_top[0] == target).to(dtype=nll_sum.dtype))
        top5_correct.add_((predictions_top == target).any().to(dtype=nll_sum.dtype))
        if index < rows:
            truth_ids[index] = target
            top_ids[index] = predictions_top
            top_probs[index] = probabilities_top

        if use_fast:
            fast_scale = torch.lerp(
                fast_scale,
                fast_scale_target,
                args.scale_rate,
            ).clamp_min(torch.finfo(fast_scale.dtype).tiny)
        memory = candidate
        hidden = next_hidden

    nll = (nll_sum / steps).item()
    return {
        "metrics": {
            "nll": nll,
            "perplexity": math.exp(nll),
            "top1_accuracy": (top1_correct / steps).item(),
            "top5_accuracy": (top5_correct / steps).item(),
            "tokens": steps,
        },
        "sequence_ids": validation[: rows + 1].cpu().tolist(),
        "truth_ids": truth_ids.cpu().tolist(),
        "top_ids": top_ids.cpu().tolist(),
        "top_probs": top_probs.cpu().tolist(),
    }


def main():
    args = tyro.cli(Args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.method not in METHODS:
        raise ValueError(f"unknown method {args.method}")
    if args.condition not in CONDITIONS:
        raise ValueError(f"unknown condition {args.condition}")
    if args.total_steps <= 0 or args.log_interval <= 0:
        raise ValueError("step counts must be positive")

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    use_fast = args.method.startswith("fast")
    robust = args.method.endswith("robust")
    model = StreamingFastWeightLM(args).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    memory = model.initial_memory(device)
    hidden = model.initial_hidden(device)
    latent_scale = torch.ones((), device=device)
    fast_scale = torch.ones((), device=device)
    clean_tokens, observed_tokens, corruption, validation_tokens, training_corpus_bytes = (
        make_stream(args, device)
    )

    def train_step(token, observed_target, clean_target, memory, hidden, latent_scale, fast_scale):
        optimizer.zero_grad(set_to_none=True)
        (
            logits,
            latent_prediction,
            candidate,
            next_hidden,
            fast_scale_target,
            write_error,
            fast_weight,
            eta,
        ) = model.forward_step(
            token,
            observed_target,
            memory,
            hidden,
            fast_scale,
            use_fast,
            robust,
            args.student_df,
        )
        log_probs = F.log_softmax(logits, dim=-1)
        observed_log_prob = log_probs[observed_target]
        if robust:
            contamination = args.contamination_probability
            categorical_loss = -torch.logaddexp(
                math.log1p(-contamination) + observed_log_prob,
                logits.new_tensor(math.log(contamination / 256.0)),
            )
        else:
            categorical_loss = -observed_log_prob

        target_embedding = model.embedding(observed_target).detach()
        latent_residual = latent_prediction - target_embedding
        latent_z2 = latent_residual.square() / latent_scale
        if robust:
            latent_loss = 0.5 * (
                (args.student_df + 1.0) * torch.log1p(latent_z2 / args.student_df)
                + latent_scale.log()
            ).mean()
            write_z2 = write_error / fast_scale
            write_loss = 0.5 * (
                (args.student_df + 1.0) * torch.log1p(write_z2 / args.student_df)
                + fast_scale.log()
            )
        else:
            latent_loss = 0.5 * latent_residual.square().mean()
            write_loss = 0.5 * write_error

        loss = categorical_loss + args.latent_coef * latent_loss
        if use_fast:
            loss = loss + args.write_coef * write_loss
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            latent_weight = (
                (args.student_df + 1.0) / (args.student_df + latent_z2.mean())
                if robust
                else latent_scale.new_ones(())
            )
            next_latent_scale = torch.lerp(
                latent_scale,
                latent_weight * latent_residual.detach().square().mean(),
                args.scale_rate,
            )
            if use_fast:
                next_fast_scale = torch.lerp(
                    fast_scale,
                    fast_scale_target.detach(),
                    args.scale_rate,
                )
            else:
                next_fast_scale = fast_scale
            tiny = torch.finfo(latent_scale.dtype).tiny
            next_latent_scale = next_latent_scale.clamp_min(tiny)
            next_fast_scale = next_fast_scale.clamp_min(tiny)

        clean_nll = -log_probs[clean_target]
        clean_probability = observed_log_prob.exp()
        mixture = (
            (1.0 - args.contamination_probability) * clean_probability
            + args.contamination_probability / 256.0
        )
        clean_responsibility = (
            (1.0 - args.contamination_probability) * clean_probability / mixture
            if robust
            else clean_probability.new_ones(())
        )
        metrics = torch.stack(
            (
                clean_nll.detach(),
                loss.detach(),
                categorical_loss.detach(),
                latent_loss.detach(),
                write_loss.detach(),
                fast_weight.detach(),
                eta.detach(),
                candidate.square().mean().sqrt(),
                clean_responsibility.detach(),
                grad_norm.detach(),
            )
        )
        return (
            candidate.detach(),
            next_hidden.detach(),
            next_latent_scale,
            next_fast_scale,
            metrics,
        )

    if args.compile:
        train_step = torch.compile(train_step, mode=args.compile_mode)

    run_name = (
        f"StreamingLanguage-v1__{args.exp_name}_{args.method}_{args.condition}"
        f"__{args.seed}__{int(time.time())}"
    )
    run_dir = Path("runs") / run_name
    writer = SummaryWriter(str(run_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join(f"|{key}|{value}|" for key, value in asdict(args).items()),
    )

    metrics_sum = torch.zeros(10, device=device)
    clean_metrics_sum = torch.zeros((), device=device)
    corrupt_metrics_sum = torch.zeros((), device=device)
    corrupt_count = torch.zeros((), device=device)
    block_count = 0
    cumulative_clean_nll = torch.zeros((), device=device)
    clean_nlls = torch.empty(args.total_steps, device=device)
    torch.cuda.synchronize()
    total_start = time.perf_counter()
    steady_start = None

    for index in range(args.total_steps):
        memory, hidden, latent_scale, fast_scale, metrics = train_step(
            observed_tokens[index],
            observed_tokens[index + 1],
            clean_tokens[index + 1],
            memory,
            hidden,
            latent_scale,
            fast_scale,
        )
        metrics_sum.add_(metrics)
        cumulative_clean_nll.add_(metrics[0])
        clean_nlls[index] = metrics[0]
        corrupted_target = corruption[index + 1].to(dtype=metrics.dtype)
        corrupt_metrics_sum.add_(metrics[0] * corrupted_target)
        clean_metrics_sum.add_(metrics[0] * (1.0 - corrupted_target))
        corrupt_count.add_(corrupted_target)
        block_count += 1

        if index + 1 == args.log_interval:
            torch.cuda.synchronize()
            steady_start = time.perf_counter()
        if (index + 1) % args.log_interval == 0:
            torch.cuda.synchronize()
            step_count = index + 1
            elapsed = time.perf_counter() - total_start
            values = metrics_sum / block_count
            # Fraction of uniform-byte code length removed: random = 0,
            # perfect prediction = 1, and negative means worse than uniform.
            code_length_skill = 1.0 - values[0].item() / math.log(256.0)
            writer.add_scalar("language/code_length_skill", code_length_skill, step_count)
            for name, value in zip(
                (
                    "clean_nll",
                    "training_loss",
                    "categorical_loss",
                    "latent_loss",
                    "write_loss",
                    "student_weight",
                    "eta",
                    "memory_rms",
                    "clean_responsibility",
                    "grad_norm",
                ),
                values,
            ):
                writer.add_scalar(f"language/{name}", value.item(), step_count)
            writer.add_scalar(
                "language/noncorrupt_clean_nll",
                (clean_metrics_sum / (block_count - corrupt_count).clamp_min(1)).item(),
                step_count,
            )
            writer.add_scalar(
                "language/corrupt_step_clean_nll",
                (corrupt_metrics_sum / corrupt_count.clamp_min(1)).item(),
                step_count,
            )
            writer.add_scalar("language/latent_scale", latent_scale.item(), step_count)
            writer.add_scalar("language/fast_scale", fast_scale.item(), step_count)
            writer.add_scalar("charts/SPS_total", step_count / elapsed, step_count)
            if steady_start is not None and step_count > args.log_interval:
                writer.add_scalar(
                    "charts/SPS_steady",
                    (step_count - args.log_interval) / (time.perf_counter() - steady_start),
                    step_count,
                )
            metrics_sum.zero_()
            clean_metrics_sum.zero_()
            corrupt_metrics_sum.zero_()
            corrupt_count.zero_()
            block_count = 0

    torch.cuda.synchronize()
    seconds = time.perf_counter() - total_start
    clean_nll = (cumulative_clean_nll / args.total_steps).item()
    evaluation = evaluate_heldout(
        model,
        validation_tokens,
        args,
        device,
        fast_scale,
    )
    for name, value in evaluation["metrics"].items():
        writer.add_scalar(f"heldout/{name}", value, args.total_steps)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    recurrent_state_floats = (
        args.dim
        + args.fast_heads * (args.dim // args.fast_heads) ** 2
        + 2
    )
    torch.save(
        {
            "args": asdict(args),
            "model": model.state_dict(),
            "fast_scale": fast_scale.detach(),
            "latent_scale": latent_scale.detach(),
        },
        run_dir / "checkpoint.pt",
    )
    (run_dir / "inference.json").write_text(
        json.dumps(evaluation),
        encoding="utf-8",
    )
    result = {
        "method": args.method,
        "condition": args.condition,
        "seed": args.seed,
        "steps": args.total_steps,
        "seconds": seconds,
        "clean_nll": clean_nll,
        "code_length_skill": 1.0 - clean_nll / math.log(256.0),
        "sps_total": args.total_steps / seconds,
        "run_name": run_name,
        "parameter_count": parameter_count,
        "parameter_bytes_fp32": parameter_count * 4,
        "recurrent_state_floats": recurrent_state_floats,
        "recurrent_state_bytes_fp32": recurrent_state_floats * 4,
        "training_corpus_bytes": training_corpus_bytes,
        "corpus_passes": args.total_steps / training_corpus_bytes,
        "heldout_nll": evaluation["metrics"]["nll"],
        "heldout_perplexity": evaluation["metrics"]["perplexity"],
        "heldout_top1_accuracy": evaluation["metrics"]["top1_accuracy"],
        "heldout_top5_accuracy": evaluation["metrics"]["top5_accuracy"],
        "checkpoint": str(run_dir / "checkpoint.pt"),
        "inference_artifact": str(run_dir / "inference.json"),
    }
    corrupt_indices = torch.nonzero(corruption[1 : args.total_steps + 1]).squeeze(1)
    if corrupt_indices.numel() > 0:
        for lag in (1, 10, 100):
            valid = corrupt_indices + lag < args.total_steps
            if valid.any():
                post_corrupt_nll = clean_nlls[corrupt_indices[valid] + lag].mean().item()
                result[f"post_corrupt_{lag}_nll"] = post_corrupt_nll
                writer.add_scalar(
                    f"summary/post_corrupt_{lag}_nll",
                    post_corrupt_nll,
                    args.total_steps,
                )
    writer.close()
    if args.compile:
        del train_step
        torch.compiler.reset()
    torch.cuda.synchronize()

    print(json.dumps(result, sort_keys=True), flush=True)
    if args.compile:
        # PyTorch 2.8 can segfault while destroying this per-token compiled
        # optimizer graph after all outputs are flushed. Avoid that buggy
        # interpreter teardown; the OS releases the already-synchronized CUDA
        # context and the writer/checkpoint are closed above.
        os._exit(0)
if __name__ == "__main__":
    main()
