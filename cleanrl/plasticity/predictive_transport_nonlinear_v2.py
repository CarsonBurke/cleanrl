"""Full batch-one nonlinear parity stream for a task-agnostic paired-gradient rule.

32 inputs, 32 tanh hidden units, scalar output; only two inputs define parity.
The remaining inputs are rare distractors. Training labels add Gaussian variance
5. Every candidate gets autograd gradients at two parameter vectors on the SAME
single observation. Clean targets are evaluation-only; no curvature is supplied.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time

import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.plasticity.predictive_transport_v2 import CONTROLS, PredictiveTransportState
from cleanrl.plasticity.predictive_transport_benchmark_v2 import (
    best_row, comparisons, configurations, log_result, save_json, score_metrics,
)

CHUNK = 100
INPUTS = 32
HIDDEN = 32
SIZES = (INPUTS * HIDDEN, HIDDEN, HIDDEN, 1)
PARAMETERS = sum(SIZES)


@dataclass
class Args:
    seed: int = 1
    observations: int = 50000
    evaluation_observations: int = 16384
    log_every: int = 10000


def forward(weight, x):
    w1 = weight[:INPUTS * HIDDEN].reshape(HIDDEN, INPUTS)
    b1 = weight[INPUTS * HIDDEN:INPUTS * HIDDEN + HIDDEN]
    w2 = weight[-HIDDEN - 1:-1]
    return (torch.tanh(x @ w1.T + b1) * w2).sum(-1) + weight[-1]


def loss_with_prediction(weight, x, target):
    prediction = forward(weight, x)
    return 0.5 * (prediction - target).square(), prediction


batched_grad = torch.vmap(torch.func.grad_and_value(loss_with_prediction, has_aux=True), in_dims=(0, None, 0))
batched_forward = torch.vmap(forward, in_dims=(0, None))


def draw_inputs(count, generator):
    x = (torch.rand((count, INPUTS), device="cuda", generator=generator) < 0.01).float()
    x[:, :2] = 2 * (torch.rand((count, 2), device="cuda", generator=generator) < 0.5).float() - 1
    return x


def credit_audits():
    theta = torch.ones(2, device="cuda", dtype=torch.float64)
    target = theta.new_tensor((0.0, 9.95))
    def curved_loss(w):
        return 0.5 * (torch.stack((w[0], 10 * w[1].square())) - target).square().sum()
    exact = torch.func.grad(curved_loss)(theta)
    less_accurate = theta.new_tensor((1.0, 0.0))
    a = theta.new_tensor(0.5)
    def delayed_loss(gain):
        first = 1 - gain
        second = (1 - 1.5) * first
        return 0.5 * second.square()
    exact_historical = torch.func.grad(delayed_loss)(a)
    final_theta = (1 - 1.5) * (1 - a)
    untransported_credit = -final_theta
    anchor = theta.new_tensor(1.0)
    displacement = final_theta - anchor
    current_credit = final_theta * displacement
    def rescaled_loss(scale):
        return 0.5 * (anchor + scale * displacement).square()
    epsilon = 1e-5
    finite_difference = (rescaled_loss(1 + epsilon) - rescaled_loss(1 - epsilon)) / (2 * epsilon)
    return {
        "nonlinear_forecast_falsifier": {
            "initial_loss": float(curved_loss(theta)),
            "exact_forecast_loss_after_step": float(curved_loss(theta - 0.02 * exact)),
            "inaccurate_forecast_loss_after_step": float(curved_loss(theta - 0.02 * less_accurate)),
            "interpretation": "Gradient forecast accuracy does not imply a useful finite update.",
        },
        "credit_scope_falsifier": {
            "exact_historical_gain_derivative": float(exact_historical),
            "naive_untransported_credit": float(untransported_credit),
            "current_pending_displacement_derivative": float(current_credit),
            "current_derivative_finite_difference": float(finite_difference),
            "interpretation": "Current displacement credit is NOT the causal derivative of an earlier gain through intervening updates.",
        },
    }


class Experiment:
    def __init__(self, args, initial, reference, configs, groups):
        self.args, self.configs, self.groups = args, configs, groups
        n = len(configs)
        self.weight = initial.expand(n, -1).clone()
        self.snapshot = self.weight.clone()
        self.m = torch.zeros_like(self.weight[groups["adamw"]])
        self.v = torch.zeros_like(self.m)
        self.t = torch.zeros((), device="cuda")
        self.count = torch.zeros(2, dtype=torch.float64, device="cuda")
        self.error = torch.zeros((2, n), dtype=torch.float64, device="cuda")
        self.excess = torch.zeros_like(self.error)
        self.power = torch.zeros_like(self.error)
        self.clean_error = torch.zeros_like(self.error)
        self.zero = torch.zeros((2, 2), dtype=torch.float64, device="cuda")
        self.forecast_error = torch.zeros(n, dtype=torch.float64, device="cuda")
        self.zero_forecast_error = torch.zeros_like(self.forecast_error)
        self.write_effect = torch.zeros_like(self.forecast_error)
        self.update_energy = torch.zeros_like(self.forecast_error)
        self.peak_update = torch.zeros_like(self.forecast_error)
        self.ever_bad = torch.zeros(n, dtype=torch.bool, device="cuda")
        self.first_bad = torch.full((n,), -1.0, device="cuda")
        self.index = torch.tensor([r["stream"] for r in configs], device="cuda")
        self.lr = torch.tensor([r.get("lr", 0.0) for r in configs], device="cuda")[:, None]
        self.controllers = {name: PredictiveTransportState(self.weight[groups[name]], control=name,
                                                          reference=reference, groups=SIZES)
                            for name in CONTROLS}
        self.states = [self.weight, self.snapshot, self.m, self.v, self.t, self.count, self.error, self.excess,
                       self.power, self.clean_error, self.zero, self.forecast_error,
                       self.zero_forecast_error, self.write_effect, self.update_energy,
                       self.peak_update, self.ever_bad, self.first_bad]
        self.states.extend(v for state in self.controllers.values() for v in state.buffers().values())
        self.inputs = (torch.zeros((CHUNK, INPUTS), device="cuda"),
                       torch.zeros((CHUNK, 2), device="cuda"),
                       torch.zeros((CHUNK, 2), device="cuda"))
        candidate_start = groups[CONTROLS[0]].start

        def step(x, target_rows, clean_rows):
            before = self.snapshot
            targets = target_rows[self.index]
            gradient, (_, prediction) = batched_grad(self.weight, x, targets)
            bad = ~torch.isfinite(prediction)
            self.first_bad.copy_(torch.where(bad & ~self.ever_bad, self.t + 1, self.first_bad))
            self.ever_bad.logical_or_(bad)
            phase = torch.stack((self.t < args.observations // 2, self.t >= args.observations // 2))
            self.count.add_(phase.double())
            pred64, target64 = prediction.double(), targets.double()
            self.error.add_(torch.where(phase[:, None], (pred64 - target64).square()[None, :], 0.0))
            self.excess.add_(torch.where(phase[:, None], (pred64.square() - 2 * pred64 * target64)[None, :], 0.0))
            self.power.add_(torch.where(phase[:, None], pred64.square()[None, :], 0.0))
            self.clean_error.add_(torch.where(phase[:, None], (pred64 - clean_rows[self.index].double()).square()[None, :], 0.0))
            self.zero.add_(torch.where(phase[:, None], target_rows.double().square()[None, :], 0.0))
            self.t.add_(1)
            old_weights = torch.cat(tuple(state.previous_weight for state in self.controllers.values()))
            old_gradient, (_, old_prediction) = batched_grad(old_weights, x, targets[candidate_start:])
            for name, state in self.controllers.items():
                rows = groups[name]
                previous_rows = slice(rows.start - candidate_start, rows.stop - candidate_start)
                g_old = old_gradient[previous_rows]
                self.forecast_error[rows].add_((g_old - state.forecast).double().square().sum(1))
                self.zero_forecast_error[rows].add_(g_old.double().square().sum(1))
                old_residual = old_prediction[previous_rows].double() - target64[rows]
                self.write_effect[rows].add_(0.5 * ((pred64[rows] - target64[rows]).square() - old_residual.square()))
                candidate_weight = self.snapshot[rows].clone()
                state.step(candidate_weight, gradient[rows], g_old)
                self.weight[rows].copy_(candidate_weight)
            sgd, adam = groups["sgd"], groups["adamw"]
            self.weight[sgd].sub_(self.lr[sgd] * gradient[sgd])
            self.m.mul_(0.9).add_(gradient[adam], alpha=0.1)
            self.v.mul_(0.999).addcmul_(gradient[adam], gradient[adam], value=0.001)
            direction = (self.m / (1 - 0.9 ** self.t)) / ((self.v / (1 - 0.999 ** self.t)).sqrt() + 1e-8)
            self.weight[adam].mul_(1 - self.lr[adam] * 0.01)
            self.weight[adam].sub_(self.lr[adam] * direction)
            movement = (self.weight - before).double().square().sum(1)
            self.update_energy.add_(movement)
            self.peak_update.copy_(torch.maximum(self.peak_update, movement.sqrt()))

        kernel = torch.compile(step, fullgraph=True, dynamic=False,
                               options={"triton.cudagraphs": False})

        def invoke(fn, x, targets, clean):
            # Separate snapshot from compiled parameter mutations; see the
            # repeated-write regression in test_predictive_transport.py.
            self.snapshot.copy_(self.weight)
            fn(x, targets, clean)

        self.compiled = lambda x, targets, clean: invoke(kernel, x, targets, clean)
        self.eager_step = lambda x, targets, clean: invoke(step, x, targets, clean)

    def capture(self):
        snapshots = [value.clone() for value in self.states]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        self.graph = torch.cuda.CUDAGraph()
        started = time.perf_counter()
        try:
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.compiled(*(v[0] for v in self.inputs))
            stream.synchronize()
            with torch.cuda.graph(self.graph, stream=stream):
                for index in range(CHUNK):
                    self.compiled(*(v[index] for v in self.inputs))
            stream.synchronize()
        finally:
            stream.synchronize()
            for value, saved in zip(self.states, snapshots):
                value.copy_(saved)
            torch.cuda.synchronize()
        self.compile_seconds = time.perf_counter() - started
        self.capture_restored = all(torch.equal(value, saved) for value, saved in zip(self.states, snapshots))
        if not self.capture_restored:
            raise RuntimeError("capture changed learner state")

    def advance(self, x, targets, clean):
        for buffer, value in zip(self.inputs, (x, targets, clean)):
            buffer.copy_(value)
        self.graph.replay()

    def report(self, observed, elapsed, evaluation_inputs):
        n = len(self.configs)
        clean_sum = torch.zeros(n, dtype=torch.float64, device="cuda")
        clean_square_sum = torch.zeros_like(clean_sum)
        for x in evaluation_inputs.split(256):
            prediction = batched_forward(self.weight, x).double()
            truth = x[:, 0] * x[:, 1]
            targets = torch.where(self.index[:, None] == 0, truth[None, :], 0.0).double()
            error = (prediction - targets).square()
            clean_sum.add_(error.sum(1))
            clean_square_sum.add_(error.square().sum(1))
        count_eval = len(evaluation_inputs)
        risk = clean_sum / count_eval
        stderr = ((clean_square_sum / count_eval - risk.square()).clamp_min(0) / count_eval).sqrt()
        finite = torch.isfinite(self.weight).all(1)
        adam = self.groups["adamw"]
        finite[adam] &= torch.isfinite(self.m).all(1) & torch.isfinite(self.v).all(1)
        diagnostics = {}
        for name, state in self.controllers.items():
            rows = self.groups[name]
            size = rows.stop - rows.start
            for value in state.buffers().values():
                valid = torch.isfinite(value)
                if value.ndim:
                    finite[rows] &= valid.reshape(size, -1).all(1)
                else:
                    finite[rows] &= valid
            values = {"assimilation_mean": state.trust_logit.sigmoid().mean(1).cpu().tolist(),
                      "local_write_gain_mean": state.write_log_gain.exp().mean(1).cpu().tolist(),
                      "block_write_gain_mean": state.group_log_gain.exp().mean(1).cpu().tolist()}
            for offset, index in enumerate(range(rows.start, rows.stop)):
                diagnostics[index] = {key: value[offset] for key, value in values.items()}
        arrays = {name: value.cpu().tolist() for name, value in (
            ("count", self.count), ("error", self.error), ("excess", self.excess), ("power", self.power),
            ("clean", self.clean_error), ("zero", self.zero), ("forecast", self.forecast_error),
            ("zero_forecast", self.zero_forecast_error), ("write_effect", self.write_effect),
            ("update_energy", self.update_energy), ("peak_update", self.peak_update),
            ("bad", self.ever_bad), ("first_bad", self.first_bad), ("finite", finite),
            ("risk", risk), ("stderr", stderr))}
        rows = []
        for index, config in enumerate(self.configs):
            stream = config["stream"]
            row = {**config, "finite_optimizer_state": arrays["finite"][index],
                   "finite_predictions_entire_pass": not arrays["bad"][index],
                   "first_nonfinite_prediction_step": arrays["first_bad"][index],
                   "final_clean_mse": arrays["risk"][index], "final_clean_mse_MC_standard_error": arrays["stderr"][index],
                   "final_clean_R2": 1 - arrays["risk"][index] if stream == 0 else None,
                   "parameter_update_energy": arrays["update_energy"][index],
                   "largest_parameter_update_l2": arrays["peak_update"][index], **diagnostics.get(index, {})}
            row.update(score_metrics([v[index] for v in arrays["error"]], [v[index] for v in arrays["excess"]],
                                     [v[stream] for v in arrays["zero"]], [v[index] for v in arrays["power"]],
                                     [v[index] for v in arrays["clean"]], arrays["count"]))
            if config["method"] in CONTROLS:
                row["gradient_forecast_MSE_ratio_to_zero"] = (arrays["forecast"][index] / arrays["zero_forecast"][index] if arrays["zero_forecast"][index] else None)
                row["mean_next_example_write_loss_change"] = arrays["write_effect"][index] / observed if observed else None
            rows.append(row)
        return {"observations": observed, "total_observations": self.args.observations,
                "reported_epoch_seconds": time.time(), "compile_capture_seconds": self.compile_seconds,
                "capture_mutable_state_restored": self.capture_restored,
                "pass_wall_seconds_including_reporting": elapsed, "observations_per_second": observed / elapsed,
                "config_observations_per_second": observed * n / elapsed,
                "finite_state_config_count": int(finite.sum()), "configs": rows}


@torch.no_grad()
def main():
    args = tyro.cli(Args)
    if args.seed != 1 or args.observations != 50000 or args.evaluation_observations != 16384:
        raise ValueError("This registered protocol uses seed 1, 50000 observations, 16384 independent clean evaluation cases")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    initial = torch.cat((torch.randn((INPUTS * HIDDEN,), generator=generator, device="cuda") / math.sqrt(INPUTS),
                         0.1 * torch.randn((HIDDEN,), generator=generator, device="cuda"),
                         torch.randn((HIDDEN,), generator=generator, device="cuda") / math.sqrt(HIDDEN),
                         torch.zeros(1, device="cuda")))
    reference = torch.cat(tuple(torch.full((size,), 1 / math.sqrt(fan_in), device="cuda")
                                for size, fan_in in zip(SIZES, (INPUTS, INPUTS, HIDDEN, HIDDEN))))
    input_generator = torch.Generator(device="cuda").manual_seed(args.seed + 10000)
    noise_generator = torch.Generator(device="cuda").manual_seed(args.seed + 1000000)
    evaluation_generator = torch.Generator(device="cuda").manual_seed(args.seed + 2000000)
    evaluation_inputs = draw_inputs(args.evaluation_observations, evaluation_generator)
    streams = [{"kind": "signal", "alpha": 1.0}, {"kind": "pure_noise", "alpha": 0.0}]
    configs, groups = configurations(streams)
    stamp = f"{time.time():.6f}"
    directory = Path("runs") / f"nonlinear__predictive_transport_v2__1__{stamp}"
    directory.mkdir(parents=True)
    experiment = Experiment(args, initial, reference, configs, groups)
    source_names = ("predictive_transport_v2.py", "predictive_transport_nonlinear_v2.py", "predictive_transport_benchmark_v2.py")
    metadata = {"arguments": vars(args), "run_stamp": stamp, "task": "nonlinear parity",
                "network": {"inputs": INPUTS, "hidden_tanh": HIDDEN, "parameters": PARAMETERS},
                "device": torch.cuda.get_device_name(), "torch_version": torch.__version__,
                "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in source_names},
                "prediction_registry_sha256": hashlib.sha256(Path("runs/predictive_transport_v1_preregistration.json").read_bytes()).hexdigest(),
                "batch_size": 1, "observations": args.observations, "noise_variance": 5,
                "data": "Two independent +/-1 inputs define parity; thirty Bernoulli(.01) distractors. Targets are parity+Gaussian noise or matched noise alone.",
                "evaluation": "Independent, fixed 16384-case clean Monte Carlo evaluation; not exact population risk. Never used to update learners or select baseline LR.",
                "credit_audits": credit_audits(),
                "candidate_state_bytes_per_model_including_reference": {name: (sum(v.numel() * v.element_size() for v in state.buffers().values()) + state.reference.numel() * state.reference.element_size()) / (groups[name].stop - groups[name].start) for name, state in experiment.controllers.items()},
                "cost": "Two autograd gradient evaluations per candidate observation, one per baseline. Same deterministic input/target within each pair. No exact Hessian or linear-regression identities supplied.",
                "baseline": "Same initialization and stream; fifteen-rate SGD/AdamW grids, AdamW eps1e-8 decay.01. First-half observed noisy MSE selection frozen before second half.",
                "scope": "A nonlinear mechanism falsifier, not an LLM, RL, delayed-credit, or universal-generalization benchmark."}
    print(f"run={directory} parameters={PARAMETERS} observations={args.observations} configs={len(configs)}", flush=True)
    writer = SummaryWriter(str(directory))
    selection = {}
    try:
        experiment.capture()
        print(f"compile/capture={experiment.compile_seconds:.3f}s restored={experiment.capture_restored}", flush=True)
        started = time.perf_counter()
        result = experiment.report(0, max(time.perf_counter() - started, 1e-12), evaluation_inputs)
        result["comparisons"] = comparisons(result, "nonlinear", selection)
        save_json(directory / "metadata.json", {"metadata": metadata, **result})
        log_result(writer, result, 0)
        for start in range(0, args.observations, CHUNK):
            x = draw_inputs(CHUNK, input_generator)
            clean = x[:, 0] * x[:, 1]
            noise = math.sqrt(5) * torch.randn(CHUNK, generator=noise_generator, device="cuda")
            experiment.advance(x, torch.stack((clean + noise, noise), -1),
                               torch.stack((clean, torch.zeros_like(clean)), -1))
            observed = start + CHUNK
            if observed % args.log_every == 0 or observed in (args.observations // 2, args.observations):
                torch.cuda.synchronize()
                result = experiment.report(observed, time.perf_counter() - started, evaluation_inputs)
                if observed == args.observations // 2:
                    for stream in range(2):
                        for method in groups:
                            best = best_row([r for r in result["configs"] if r["stream"] == stream and r["method"] == method], "first_half_mse")
                            selection[f"{stream}:{method}"] = None if best is None else best["id"]
                    save_json(directory / "frozen_first_half_selection.json", {"metadata": metadata, "frozen_epoch_seconds": time.time(), "selection": selection, **result})
                result["frozen_selection"] = selection.copy()
                result["comparisons"] = comparisons(result, "nonlinear", selection)
                save_json(directory / "partial.json", {"metadata": metadata, **result})
                log_result(writer, result, observed)
                print(f"progress {observed}/{args.observations} elapsed={result['pass_wall_seconds_including_reporting']:.1f}s finite={result['finite_state_config_count']}/{len(configs)}", flush=True)
        result["finished_epoch_seconds"] = time.time()
        save_json(directory / "results.json", {"metadata": metadata, **result})
        torch.save({"weight": experiment.weight.cpu(), "configs": configs, "metadata": metadata,
                    "controllers": {name: {key: value.cpu() for key, value in state.buffers().items()} for name, state in experiment.controllers.items()}}, directory / "final_state.pt")
        print(f"saved {directory / 'results.json'}", flush=True)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
