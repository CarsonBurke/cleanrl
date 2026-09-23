"""Full batch-one nonlinear parity experiment for contextual predictive belief.

The ordinary 32 -> 32 tanh -> scalar predictor is unchanged. Optimizer-only
contexts use target-free local activations, and prediction Jacobians are derived
directly, including at zero residual. Conditional residual calibration concerns
the downstream scalar prediction error, not hidden-neuron supervision.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time

import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.runtime import configure_runtime

from cleanrl.plasticity.contextual_belief_v2 import CONTROLS, ContextualBeliefState
from cleanrl.plasticity.contextual_belief_benchmark_v2 import (
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


def local_context(weight, x, prediction):
    """Separate target-free pre/post contexts in the predictor's weight order."""
    w1 = weight[:, :INPUTS * HIDDEN].reshape(-1, HIDDEN, INPUTS)
    b1 = weight[:, INPUTS * HIDDEN:INPUTS * HIDDEN + HIDDEN]
    hidden_preactivation = (w1 * x[None, None, :]).sum(-1) + b1
    hidden_activation = hidden_preactivation.tanh()
    output_context = prediction.tanh()[:, None]
    presynaptic = torch.cat((
        x.tanh().expand(weight.shape[0], HIDDEN, INPUTS).flatten(1),
        torch.ones_like(hidden_activation),
        hidden_activation.tanh(),
        torch.ones_like(output_context),
    ), dim=1)
    postsynaptic = torch.cat((
        hidden_activation[:, :, None].expand(-1, -1, INPUTS).flatten(1),
        hidden_activation,
        output_context.expand(-1, HIDDEN),
        output_context,
    ), dim=1)
    return torch.stack((presynaptic, postsynaptic), dim=-1)


batched_grad = torch.vmap(torch.func.grad_and_value(loss_with_prediction, has_aux=True), in_dims=(0, None, 0))
batched_forward = torch.vmap(forward, in_dims=(0, None))
batched_jacobian = torch.vmap(torch.func.grad(forward), in_dims=(0, None))


def draw_inputs(count, generator):
    x = (torch.rand((count, INPUTS), device="cuda", generator=generator) < 0.01).float()
    x[:, :2] = 2 * (torch.rand((count, 2), device="cuda", generator=generator) < 0.5).float() - 1
    return x


class Experiment:
    def __init__(self, args, initial, configs, groups):
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
        self.update_energy = torch.zeros(n, dtype=torch.float64, device="cuda")
        self.peak_update = torch.zeros_like(self.update_energy)
        self.ever_bad = torch.zeros(n, dtype=torch.bool, device="cuda")
        self.first_bad = torch.full((n,), -1.0, device="cuda")
        self.index = torch.tensor([r["stream"] for r in configs], device="cuda")
        self.lr = torch.tensor([r.get("lr", 0.0) for r in configs], device="cuda")[:, None]
        self.controllers = {
            name: ContextualBeliefState(self.weight[groups[name]], control=name)
            for name in CONTROLS
        }
        self.mutable_buffers = {
            name: getattr(self, name) for name in (
                "weight", "snapshot", "m", "v", "t", "count", "error", "excess",
                "power", "clean_error", "zero", "update_energy", "peak_update", "ever_bad", "first_bad",
            )
        }
        self.states = list(self.mutable_buffers.values())
        self.states.extend(v for state in self.controllers.values() for v in state.buffers().values())
        self.inputs = (torch.zeros((CHUNK, INPUTS), device="cuda"),
                       torch.zeros((CHUNK, 2), device="cuda"),
                       torch.zeros((CHUNK, 2), device="cuda"))
        candidate_start = groups[CONTROLS[0]].start

        def step(x, target_rows, clean_rows):
            before = self.snapshot
            targets = target_rows[self.index]
            # Preserve the original autograd loss-gradient path for baselines.
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
            # Candidate-only direct prediction derivatives remain informative
            # when prediction == target; never divide a loss gradient by error.
            jacobian = batched_jacobian(self.weight[candidate_start:], x)
            context = local_context(self.weight[candidate_start:], x, prediction[candidate_start:])
            residual = prediction - targets
            for name, state in self.controllers.items():
                rows = groups[name]
                candidate_rows = slice(rows.start - candidate_start, rows.stop - candidate_start)
                candidate_weight = self.snapshot[rows].clone()
                state.step(candidate_weight, gradient[rows], jacobian[candidate_rows],
                           context[candidate_rows], residual[rows])
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
            # The old-state snapshot must remain outside the fused mutation
            # kernel so repeated writes cannot alias their own reference state.
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
            values = {
                "calibration_observation_count": state.observation_count.cpu().tolist(),
                "calibration_nll": state.nll_sum.cpu().tolist(),
                "calibration_standardized_square": state.standardized_square_sum.cpu().tolist(),
                "mean_predicted_variance": state.predicted_variance_sum.cpu().tolist(),
                "mean_effective_gain": state.gain_sum.cpu().tolist(),
                "mean_bias_probability": state.bias_probability_sum.cpu().tolist(),
                "mean_change_probability": state.change_probability_sum.cpu().tolist(),
            }
            for offset, index in enumerate(range(rows.start, rows.stop)):
                diagnostic = {key: value[offset] for key, value in values.items()}
                count = diagnostic["calibration_observation_count"]
                for key in ("calibration_nll", "calibration_standardized_square",
                            "mean_predicted_variance", "mean_effective_gain",
                            "mean_bias_probability", "mean_change_probability"):
                    diagnostic[key] = diagnostic[key] / count if count else None
                diagnostics[index] = diagnostic
        arrays = {name: value.cpu().tolist() for name, value in (
            ("count", self.count), ("error", self.error), ("excess", self.excess), ("power", self.power),
            ("clean", self.clean_error), ("zero", self.zero),
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
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    initial = torch.cat((torch.randn((INPUTS * HIDDEN,), generator=generator, device="cuda") / math.sqrt(INPUTS),
                         0.1 * torch.randn((HIDDEN,), generator=generator, device="cuda"),
                         torch.randn((HIDDEN,), generator=generator, device="cuda") / math.sqrt(HIDDEN),
                         torch.zeros(1, device="cuda")))
    input_generator = torch.Generator(device="cuda").manual_seed(args.seed + 10000)
    noise_generator = torch.Generator(device="cuda").manual_seed(args.seed + 1000000)
    evaluation_generator = torch.Generator(device="cuda").manual_seed(args.seed + 2000000)
    evaluation_inputs = draw_inputs(args.evaluation_observations, evaluation_generator)
    streams = [{"kind": "signal", "alpha": 1.0}, {"kind": "pure_noise", "alpha": 0.0}]
    configs, groups = configurations(streams)
    stamp = f"{time.time():.6f}"
    directory = Path("runs") / f"nonlinear__contextual_belief_v2__1__{stamp}"
    directory.mkdir(parents=True)
    experiment = Experiment(args, initial, configs, groups)
    source_names = ("contextual_belief_v2.py", "contextual_belief_nonlinear_v2.py",
                    "contextual_belief_benchmark_v2.py", "predictive_transport_benchmark_v2.py",
                    "noisy_stream_diagnostic.py", "stock_stream.py")
    metadata = {
        "arguments": vars(args), "run_stamp": stamp, "run_label": "contextual_belief_v2", "task": "nonlinear parity",
        "network": {"inputs": INPUTS, "hidden_tanh": HIDDEN, "parameters": PARAMETERS},
        "device": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in source_names},
        "preregistration_sha256": hashlib.sha256(Path("runs/contextual_belief_v2_preregistration.json").read_bytes()).hexdigest(),
        "batch_size": 1, "observations": args.observations, "noise_variance": 5,
        "data": "Two independent +/-1 inputs define parity; thirty Bernoulli(.01) distractors. Targets are parity+Gaussian noise or matched noise alone.",
        "evaluation": "Independent, fixed 16384-case clean Monte Carlo evaluation; not exact population risk. Never used to update learners or select baseline LR.",
        "candidate_state_bytes_per_model": {
            name: sum(v.numel() * v.element_size() for v in state.buffers().values())
                  / (groups[name].stop - groups[name].start)
            for name, state in experiment.controllers.items()
        },
        "candidate_state_scalars_per_parameter": 21,
        "candidate_state_bytes_per_parameter": 168,
        "candidate_diagnostic_bytes_per_model": 56,
        "cost": "Experimental, not LLM-ready: 21 FP64 optimizer scalars (168 bytes) per parameter plus seven FP64 cumulative counters (56 bytes) per model. Network weights and baseline optimizer state remain FP32; score sums are FP64. One autograd loss-gradient evaluation for every learner; an extra direct prediction-Jacobian pass and local-context activation reconstruction only for candidates. No Hessian supplied. Compiled CUDA-graph execution.",
        "baseline": "Same initialization and stream; fifteen-rate SGD/AdamW grids, AdamW eps1e-8 decay.01. First-half observed noisy MSE selection frozen before second half. Independent null selections are exploratory only; primary null refusal/power uses matched_signal_selected_config at signal-selected hyperparameters.",
        "contexts": "Separate (presynaptic, postsynaptic) components: input weights (tanh(input), tanh(hidden preactivation)); hidden biases (1, tanh(hidden preactivation)); output weights (tanh(hidden activation), tanh(output)); output bias (1, tanh(output)). Bias 1 is an explicit intercept indicator. No target enters context.",
        "calibration": "Per-parameter predictive beliefs over calibrated-zero, persistent conditional error, and changed-error hypotheses with Gaussian coefficient uncertainty and a learned revision hazard. The modeled outcome is the downstream scalar predictive residual, not a hidden-neuron target. Uncertainty concerns predictive-error coefficients, not network weights. Beliefs are optimizer metadata, not model layers or outputs. All gains precede the current outcome's calibration update. Cumulative diagnostic sums are divided by observation_count, or reported as null at zero observations; comparisons are on each learner's own trajectory, not common-state rankings.",
        "update": "Bounded Bayes-risk shrinkage gain M/(M+R), not inverse variance or a learned learning rate; normalized LMS update -gain_i*g_i/sum_j j_j^2, with zero writes for a zero Jacobian row. Linearized output correction averages per-parameter gains by Jacobian energy; redundant parameters do not amplify confidence. No exact historical credit or arbitrary nonlinear safety guarantee.",
        "scope": "A nonlinear mechanism falsifier, not an LLM, RL, delayed-credit, or universal-generalization benchmark.",
    }
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
        result["completed_full_horizon"] = observed == args.observations
        save_json(directory / "results.json", {"metadata": metadata, **result})
        torch.save({
            "weight": experiment.weight.cpu(), "configs": configs, "metadata": metadata,
            "frozen_selection": selection,
            "mutable_buffers": {name: value.cpu() for name, value in experiment.mutable_buffers.items()},
            "controllers": {name: {key: value.cpu() for key, value in state.buffers().items()}
                            for name, state in experiment.controllers.items()},
            "input_generator_state": input_generator.get_state(),
            "noise_generator_state": noise_generator.get_state(),
            "evaluation_inputs": evaluation_inputs.cpu(),
        }, directory / "final_state.pt")
        print(f"saved {directory / 'results.json'}", flush=True)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
