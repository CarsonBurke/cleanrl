"""Frozen-half utility identification; deliberately not a streaming optimizer.

Learn one counterfactual function, with U(c, 0)=0 and its own exact action
Jacobian. No independent block targets, curvature model, or confidence gate.
"""
import math
import time

import torch
from torch import nn


class UtilityModel(nn.Module):
    def __init__(self, context_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(context_dim + 4, 64), nn.SiLU(),
                                 nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, context, action):
        zeros = torch.zeros_like(action)
        return (self.net(torch.cat((context, action), -1))
                - self.net(torch.cat((context, zeros), -1))).squeeze(-1)

    def values_and_derivatives(self, context, actions):
        count, probes = context.shape[0], actions.shape[0]
        values = self(context[:, None, :].expand(-1, probes, -1),
                      actions[None, :, :].expand(count, -1, -1))
        # At action=1, dU/daction equals dU/dlog(action). This differentiates
        # the SAME finite-utility function, not an unrelated derivative head.
        reference = context.new_ones((count, 4))
        derivative = torch.func.grad(lambda a: self(context, a).sum())(reference)
        return values, derivative


def signed_log(value):
    return value.sign() * torch.log1p(value.abs())


def block_summary(values):
    """Descriptive temporal uncertainty, not an IID significance guarantee."""
    means = torch.stack([part.double().mean() for part in torch.tensor_split(values, 50)])
    mean = float(values.double().mean())
    se = float(means.std(unbiased=True) / math.sqrt(len(means)))
    return {"mean": mean, "block_mean_se": se, "mean_minus_two_block_se": mean - 2 * se,
            "chronological_block_means": means.cpu().tolist()}


def identify(contexts, utilities, derivatives, actions, midpoint, score_after, streams,
             writer, directory, save_json, task):
    """Inputs indexed by originating observation; final observation has no target."""
    started = time.perf_counter()
    # origin midpoint-1 would use the first second-half target: never train on it.
    training_context = contexts[score_after:midpoint - 1]
    training_utility = utilities[score_after:midpoint - 1]
    training_derivative = derivatives[score_after:midpoint - 1]
    mean_by_stream = training_utility.double().mean(0)
    raw = signed_log(training_context.reshape(-1, contexts.shape[-1]))
    center = raw.mean(0)
    spread = raw.std(0).where(raw.std(0) > 0, 1.0)
    train_x = (raw - center) / spread
    scale = training_utility.double().square().mean().sqrt().float()
    if not bool(torch.isfinite(scale) & (scale > 0)):
        raise RuntimeError("Degenerate/nonfinite utility supervision")
    train_y = training_utility.reshape(-1, actions.shape[0]) / scale
    train_d = training_derivative.reshape(-1, 4) / scale
    if not all(bool(torch.isfinite(v).all()) for v in (train_x, train_y, train_d)):
        raise RuntimeError("Nonfinite first-half identification records")
    torch.manual_seed(1)
    model = UtilityModel(contexts.shape[-1]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, foreach=True)

    def objective(x, y, d):
        predicted, slope = model.values_and_derivatives(x, actions)
        return (predicted - y).square().mean() + (slope - d).square().mean()

    objective = torch.compile(objective, fullgraph=True, dynamic=False)
    generator = torch.Generator(device="cuda").manual_seed(2000001)
    history = []
    for epoch in range(10):
        order = torch.randperm(len(train_x), generator=generator, device="cuda")
        accumulated = torch.zeros((), device="cuda")
        for indices in order.split(2048):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(train_x[indices], train_y[indices], train_d[indices])
            loss.backward()
            optimizer.step()
            accumulated.add_(loss.detach() * len(indices))
        value = float(accumulated / len(train_x))
        if not math.isfinite(value):
            raise RuntimeError("Utility model fitting became nonfinite")
        history.append(value)
        writer.add_scalar("identification/training_objective", value, epoch + 1)
        writer.flush()
        print(f"utility epoch={epoch + 1}/10 objective={value:.7g}", flush=True)
    torch.cuda.synchronize()
    frozen = {"training_objective": history, "target_scale": float(scale),
              "mean_utility_by_stream": mean_by_stream.cpu().tolist(),
              "training_origins": [score_after, midpoint - 2],
              "excluded_cross_boundary_origin": midpoint - 1,
              "frozen_epoch_seconds": time.time(), "fit_seconds": time.perf_counter() - started,
              "model_parameters": sum(p.numel() for p in model.parameters())}
    save_json(directory / "frozen_utility_model.json", frozen)
    torch.save({"state_dict": model.state_dict(), "center": center, "spread": spread,
                "scale": scale, "frozen": frozen}, directory / "utility_model.pt")
    del raw, train_x, train_y, train_d, optimizer

    @torch.no_grad()
    def predict(x):
        x = (signed_log(x) - center) / spread
        values, slopes = model.values_and_derivatives(x, actions)
        return values * scale, slopes * scale

    compiled = torch.compile(predict, fullgraph=True, dynamic=False)
    output = []
    with torch.no_grad():
        for stream_index, stream in enumerate(streams):
            actual = utilities[midpoint:, stream_index]
            n = len(actual)
            chunks = [compiled(chunk) for chunk in contexts[midpoint:, stream_index].split(8192)]
            prediction = torch.cat([pair[0] for pair in chunks])
            slope = torch.cat([pair[1] for pair in chunks])
            actual_slope = derivatives[midpoint:, stream_index]
            if not bool(torch.isfinite(prediction).all() & torch.isfinite(slope).all()):
                raise RuntimeError("Nonfinite frozen utility prediction")
            # A zero-action candidate is exact and participates in argmax without
            # a threshold. This is a counterfactual choice, never a model write.
            choices = torch.cat((prediction.new_zeros((n, 1)), prediction), 1).argmax(1)
            with_zero = torch.cat((actual.new_zeros((n, 1)), actual), 1)
            selected = with_zero.gather(1, choices[:, None]).squeeze(1)
            signal_index = stream_index - stream_index % 5 if len(streams) == 15 else 0
            fixed = int(torch.cat((mean_by_stream.new_zeros(1), mean_by_stream[signal_index])).argmax())
            fixed_utility = with_zero[:, fixed]
            one = actual[:, 2]
            residual = prediction.double() - actual.double()
            constant_residual = mean_by_stream[stream_index] - actual.double()
            zero_error = float(actual.double().square().mean())
            model_error = float(residual.square().mean())
            constant_error = float(constant_residual.square().mean())
            interaction = actual[:, 2] - actual[:, 5:9].sum(1)
            selected_summary = block_summary(selected)
            advantage_summary = block_summary(selected - fixed_utility)
            phases = {}
            if task == "switch":
                for label, part in (("moved", slice(0, 25000)), ("returned", slice(25000, None))):
                    phases[label] = {"selected": block_summary(selected[part]),
                                     "advantage_over_fixed": block_summary((selected - fixed_utility)[part])}
            row = {"stream": stream_index, **stream, "observations": n,
                   "prediction_mse": model_error, "zero_prediction_mse": zero_error,
                   "context_free_prediction_mse": constant_error,
                   "mse_over_zero": model_error / zero_error if zero_error else None,
                   "mse_over_context_free": model_error / constant_error if constant_error else None,
                   "selected_utility": selected_summary, "advantage_over_fixed": advantage_summary,
                   "reference_one_utility": block_summary(one),
                   "fixed_action_index_including_zero": fixed,
                   "fixed_action_utility": block_summary(fixed_utility),
                   "hindsight_noisy_oracle_utility": block_summary(with_zero.max(1).values),
                   "chosen_action_counts_including_zero": torch.bincount(choices, minlength=16).cpu().tolist(),
                   "block_interaction_rms": float(interaction.double().square().mean().sqrt()),
                   "joint_utility_rms": float(one.double().square().mean().sqrt()),
                   "utility_sign_accuracy_nonzero_targets": float(
                       ((prediction > 0) == (actual > 0))[actual != 0].float().mean()),
                   "doubling_improvement_sign_accuracy": float(
                       ((prediction[:, 3] > prediction[:, 2]) == (actual[:, 3] > actual[:, 2])).float().mean()),
                   "derivative_mse": float((slope.double() - actual_slope.double()).square().mean()),
                   "zero_derivative_mse": float(actual_slope.double().square().mean()),
                   "derivative_sign_accuracy_nonzero_targets": float(
                       ((slope > 0) == (actual_slope > 0))[actual_slope != 0].float().mean()),
                   "phase_details": phases,
                   "passes_registered_gate": (model_error < constant_error
                        and selected_summary["mean_minus_two_block_se"] > 0
                        and advantage_summary["mean_minus_two_block_se"] > 0)}
            output.append(row)
            writer.add_scalar(f"identification/stream_{stream_index}/mse_over_context_free",
                              row["mse_over_context_free"], len(contexts))
            writer.add_scalar(f"identification/stream_{stream_index}/selected_utility",
                              selected_summary["mean"], len(contexts))
            writer.flush()
    return {"frozen": frozen, "streams": output,
            "identification_seconds": time.perf_counter() - started}
