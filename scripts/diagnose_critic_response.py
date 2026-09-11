"""Inspect trusted captured PPO critic updates on CUDA; queue through mlq, not a benchmark."""

import argparse
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.runtime import configure_runtime

SCALAR_TRAINER = "cleanrl.ppo_continuous_action_32xlr_1mb_noadvnorm_normres_indclip_v3"
TWOHOT_TRAINER = "cleanrl.ppo_continuous_action_32xlr_1mb_noadvnorm_normres_twohot_indclip_v4"


def emit(label, **metrics):
    """Reduce on CUDA first, then transfer one packed metrics group."""
    values = torch.stack(tuple(value.double() for value in metrics.values())).cpu().tolist()
    print(label + " " + " ".join(f"{key}={value:.9g}" for key, value in zip(metrics, values)))


def divide(numerator, denominator):
    return torch.where(denominator != 0, numerator / denominator, torch.full_like(numerator, float("nan")))


def rmse(error):
    return error.square().mean().sqrt()


def make_agent(trainer, args, snapshot, device):
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            -np.inf, np.inf, shape=tuple(snapshot["batch"]["observations"].shape[1:]), dtype=np.float32
        ),
        single_action_space=gym.spaces.Box(
            snapshot["model"]["action_low"].numpy().copy(),
            snapshot["model"]["action_high"].numpy().copy(),
            dtype=np.float32,
        ),
    )
    options = dict(placement=args.placement, norm_kind=args.norm_kind, activation=args.activation)
    if snapshot["trainer_module"] == TWOHOT_TRAINER:
        options.update(
            value_loss=args.value_loss,
            value_num_bins=args.value_num_bins,
            value_max_abs=args.value_max_abs,
            value_spacing=args.value_spacing,
        )
    # Match the existing replay: saved bounds/observation shape suffice for critic forwards.
    # Initialization stays on CUDA and its RNG effects are isolated by the caller.
    with torch.device(device):
        agent = trainer.Agent(spaces, **options)
    agent.load_state_dict(snapshot["model"])
    return agent


def direction_stats(label, correction, residual):
    nonzero = residual != 0
    wrong = correction * residual < 0
    emit(
        label,
        correction_mean=correction.mean(),
        correction_rms=rmse(correction),
        correction_abs_mean=correction.abs().mean(),
        wrong_fraction_nonzero=divide(wrong.double().sum(), nonzero.double().sum()),
        wrong_abs_mean=divide(torch.where(wrong, correction.abs(), 0).sum(), wrong.double().sum()),
        zero_fraction=(correction == 0).double().mean(),
        residual_projection=divide((correction * residual).sum(), residual.square().sum()),
    )


def target_geometry(support, returns, q):
    # Match project()'s right insertion, including duplicate zeros and clipped endpoints.
    raw = returns.float().contiguous()
    insertion = torch.searchsorted(support.support, raw, right=True)
    below = (insertion - 1).clamp(0, support.num_bins - 1)
    above = insertion.clamp(0, support.num_bins - 1)
    centers = support.support.double()
    width = centers[above] - centers[below]
    valid = width > 0
    phase = divide(returns - centers[below], width)
    entropy = -torch.special.xlogy(q, q).sum(-1)
    # A ten-bin sample histogram is not the entropy of a batch-averaged target.
    phase_bin = torch.where(valid, phase, 0).mul(10).long().clamp(0, 9)
    phase_counts = torch.zeros(10, dtype=torch.float64, device=returns.device)
    phase_counts.scatter_add_(0, phase_bin, valid.double())
    phase_mean = divide(torch.where(valid, phase, 0).sum(), valid.double().sum())
    emit(
        "target_geometry",
        width_mean=width.mean(),
        width_std=width.std(correction=0),
        target_entropy_mean=entropy.mean(),
        target_entropy_std=entropy.std(correction=0),
        phase_mean=phase_mean,
        phase_std=divide(
            torch.where(valid, (phase - phase_mean).square(), 0).sum(),
            valid.double().sum(),
        ).sqrt(),
        degenerate_interval_fraction=(~valid).double().mean(),
        outside_support_fraction=((returns < centers[0]) | (returns > centers[-1])).double().mean(),
        projected_target_bias=((q * centers).sum(-1) - returns).mean(),
        projected_target_rmse=rmse((q * centers).sum(-1) - returns),
        target_mass_max_error=(q.sum(-1) - 1).abs().max(),
    )
    emit(
        "phase_fraction_[0,.1)...[.9,1]_nondegenerate",
        **{f"bin{i}": divide(phase_counts[i], valid.double().sum()) for i in range(10)},
    )
    return insertion, width, entropy


def categorical_stats(stage, logits, values, returns, support, q, entropy, width):
    # Softmax/log-softmax retain trainer FP32 math; diagnostic reductions use FP64.
    p = logits.float().softmax(-1).double()
    logp = logits.float().log_softmax(-1).double()
    centers = support.support.double()
    ce = -(q * logp).sum(-1)
    kl = (torch.special.xlogy(q, q) - q * logp).sum(-1)
    predicted_entropy = -torch.special.xlogy(p, p).sum(-1)
    residual = returns - values
    local = width > 0
    emit(
        f"{stage}/categorical",
        CE=ce.mean(),
        target_H=entropy.mean(),
        KL_q_p=kl.mean(),
        decomposition_max_abs=(ce - entropy - kl).abs().max(),
        predicted_H=predicted_entropy.mean(),
        abs_error_le_2_local_width_fraction=divide(
            ((residual.abs() <= 2 * width) & local).double().sum(), local.double().sum()
        ),
    )
    jacobian = p * (centers - values.unsqueeze(-1))
    ce_gradient = p - q
    jacobian_norm = torch.linalg.vector_norm(jacobian, dim=-1)
    gradient_norm = torch.linalg.vector_norm(ce_gradient, dim=-1)
    correction = -(jacobian * ce_gradient).sum(-1)
    emit(
        f"{stage}/head_geometry",
        value_jacobian_norm_mean=jacobian_norm.mean(),
        value_jacobian_norm_rms=rmse(jacobian_norm),
        value_jacobian_norm_max=jacobian_norm.max(),
        CE_gradient_norm_mean=gradient_norm.mean(),
        CE_gradient_norm_rms=rmse(gradient_norm),
    )
    direction_stats(f"{stage}/independent_logit_GD_per_unit_step", correction, residual)
    del jacobian, ce_gradient, logp
    for threshold in (10000, 1000000):
        tail = centers.abs() > threshold
        tail_mass = (p * tail).sum(-1)
        signed = (p * (centers * tail)).sum(-1)
        absolute = (p * (centers.abs() * tail)).sum(-1)
        kept = p * (~tail)
        kept_mass = kept.sum(-1)
        masked_value = divide((kept * centers).sum(-1), kept_mass)
        emit(
            f"{stage}/abs_atom_gt_{threshold}",
            probability_mean=tail_mass.mean(),
            probability_max=tail_mass.max(),
            signed_decoded_mean=signed.mean(),
            signed_decoded_rms=rmse(signed),
            absolute_decoded_mean=absolute.mean(),
            abs_net_decoded_mean=signed.abs().mean(),
            masked_renorm_value_mean=masked_value.mean(),
            masked_renorm_RMSE=rmse(masked_value - returns),
            masked_value_delta_rms=rmse(masked_value - values),
            zero_retained_mass_fraction=(kept_mass == 0).double().mean(),
        )
    return ce


def interval_groups(support, insertion, width, entropy, returns, pre, post, pre_ce, post_ce):
    # Group by insertion rather than lower index: overflow endpoint labels stay separate.
    size = support.num_bins + 1
    counts = torch.bincount(insertion, minlength=size)
    totals = torch.zeros((size, 7), dtype=torch.float64, device=returns.device)
    rows = torch.stack(
        (returns, width, entropy, pre_ce, post_ce, (pre - returns).square(), (post - returns).square()), dim=-1
    )
    totals.index_add_(0, insertion, rows)
    largest = counts.argsort(descending=True, stable=True)[:5]
    average = divide(totals[largest], counts[largest, None].double())
    centers = support.support.double()
    lower = centers[(largest - 1).clamp(0, support.num_bins - 1)]
    upper = centers[largest.clamp(0, support.num_bins - 1)]
    packed = (
        torch.cat(
            (largest[:, None].double(), counts[largest, None].double(), lower[:, None], upper[:, None], average), dim=-1
        )
        .cpu()
        .tolist()
    )
    for index, count, low, high, target, local_width, h, ce0, ce1, mse0, mse1 in packed:
        if count:
            print(
                f"interval insertion={int(index)} count={int(count)} bounds=[{low:.9g},{high:.9g}] "
                f"target_mean={target:.9g} width={local_width:.9g} H={h:.9g} "
                f"CE_pre={ce0:.9g} CE_post={ce1:.9g} RMSE_pre={mse0 ** .5:.9g} RMSE_post={mse1 ** .5:.9g}"
            )


@torch.no_grad()
def analyze(snapshot, path, trainer, args):
    device = torch.device("cuda")
    batch = snapshot["batch"]
    agent = make_agent(trainer, args, snapshot, device)
    support = getattr(agent, "value_support", None)
    observations = batch["observations"].to(device, copy=True)
    returns = batch["returns"].to(device).double().flatten()
    old_values = batch["old_values"].to(device).double().flatten()
    advantages = batch["advantages"].to(device).double().flatten()
    rewards = batch["rewards"].to(device).double().flatten()
    reward_baseline = divide(rewards.mean(), rewards.new_tensor(1 - args.gamma))
    print(f"\nsnapshot={path} global_step={snapshot['global_step']} trainer={snapshot['trainer_module']}")
    print(
        f"objective={'twohot' if support is not None else 'scalar_half_MSE'} reward_norm={args.reward_norm} "
        f"gamma={args.gamma} schedule={args.total_timesteps} seed={args.seed} batch={len(returns)} "
        f"optimizer_steps={len(snapshot['minibatch_indices'])} "
        f"lr={snapshot['optimizer']['param_groups'][0]['lr']}"
    )
    if support is not None:
        print(f"support={args.value_num_bins}/{args.value_spacing}/+/-{args.value_max_abs}")
    emit(
        "rollout_fixed",
        return_mean=returns.mean(),
        return_std=returns.std(correction=0),
        reward_mean=rewards.mean(),
        reward_mean_over_1_minus_gamma=reward_baseline,
        stored_actor_GAE_mean=advantages.mean(),
        stored_actor_GAE_std=advantages.std(correction=0),
        stored_critic_mean=old_values.mean(),
    )

    def forward(obs):
        output = agent.critic(obs)
        value = support.to_scalar(output) if support is not None else output.flatten()
        return output, value

    compiled_forward = graph_compile(forward)
    compiled_hidden = graph_compile(agent.critic[0]) if support is None else None
    geometry = None
    if support is not None:
        q = batch["targets"].to(device).double()
        if q.shape != (len(returns), support.num_bins):
            raise ValueError("Captured categorical targets have the wrong shape")
        insertion, width, entropy = target_geometry(support, returns, q)
        geometry = q, insertion, width, entropy
    values = []
    cross_entropies = []
    for stage in ("pre", "post"):
        if stage == "post":
            agent.load_state_dict(snapshot["post_model"])
        logits, raw_values = compiled_forward(observations)
        value = raw_values.double().flatten()
        values.append(value)
        residual = returns - value
        emit(
            f"{stage}/raw_value",
            value_mean=value.mean(),
            value_std=value.std(correction=0),
            value_minus_reward_baseline=value.mean() - reward_baseline,
            RMSE=rmse(residual),
            bias=-residual.mean(),
            residual_mean=residual.mean(),
            residual_std=residual.std(correction=0),
            EV=1 - divide(residual.var(correction=0), returns.var(correction=0)),
        )
        if geometry is not None:
            q, insertion, width, entropy = geometry
            cross_entropies.append(categorical_stats(stage, logits, value, returns, support, q, entropy, width))
        else:
            emit(
                f"{stage}/head_geometry",
                value_jacobian_norm_mean=value.new_tensor(1),
                half_MSE_gradient_abs_mean=residual.abs().mean(),
                half_MSE_gradient_rms=rmse(residual),
            )
            direction_stats(f"{stage}/independent_scalar_GD_per_unit_step", residual, residual)
            assert compiled_hidden is not None
            hidden = compiled_hidden(observations).double()
            weight = agent.critic[-1].weight.detach().double().flatten()
            head_bias = agent.critic[-1].bias.detach().double().squeeze()
            weight_norm = torch.linalg.vector_norm(weight)
            hidden_norm = torch.linalg.vector_norm(hidden, dim=-1)
            cosine = divide((hidden * weight).sum(-1), hidden_norm * weight_norm)
            upper_bound = weight_norm + head_bias
            emit(
                f"{stage}/scalar_readout_saturation",
                head_weight_norm=weight_norm,
                head_bias=head_bias,
                unit_hidden_upper_bound=upper_bound,
                fraction_within_1pct_abs_upper_bound=((value - upper_bound).abs() <= 0.01 * upper_bound.abs())
                .double()
                .mean(),
                value_mean=value.mean(),
                value_max=value.max(),
                hidden_norm_mean=hidden_norm.mean(),
                hidden_norm_max=hidden_norm.max(),
                hidden_weight_cosine_mean=cosine.mean(),
                hidden_weight_cosine_min=cosine.min(),
                hidden_weight_cosine_std=cosine.std(correction=0),
            )
            del hidden, weight, hidden_norm, cosine
        del logits, raw_values
    pre, post = values
    delta = post - pre
    residual = returns - pre
    emit(
        "actual_update",
        delta_mean=delta.mean(),
        delta_RMS=rmse(delta),
        residual_projection=divide((delta * residual).sum(), residual.square().sum()),
        RMSE_change=rmse(post - returns) - rmse(residual),
        pre_vs_stored_value_RMS=rmse(pre - old_values),
        pre_vs_stored_value_max_abs=(pre - old_values).abs().max(),
    )
    direction_stats("actual_update/direction", delta, residual)
    if geometry is not None:
        _, insertion, width, entropy = geometry
        interval_groups(support, insertion, width, entropy, returns, pre, post, *cross_entropies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", type=Path, nargs="+", help="Trusted own critic_probe_*.pt (pickle)")
    paths = parser.parse_args().snapshots
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    print(
        "Fixed recorded rollout, pre/post actual optimizer update; no env steps or training. "
        "Head GD is an unweighted independent-output counterfactual, not Adam/network dynamics. "
        "Stored actor GAE is not recomputed post-update; reward/(1-gamma) is only a descriptive stationary reference. "
        "Errors use raw returns, bias=value-return, population std; undefined ratios are nan."
    )
    for path in paths:
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        if snapshot["format_version"] != 1 or snapshot["trainer_module"] not in (SCALAR_TRAINER, TWOHOT_TRAINER):
            raise ValueError(f"Unsupported snapshot: {path}")
        trainer = importlib.import_module(snapshot["trainer_module"])
        args = trainer.Args(**snapshot["args"])
        if not args.cuda or not args.compile or args.compile_mode != "reduce-overhead":
            raise ValueError("Capture must use CUDA and compiled reduce-overhead updates")
        if len(snapshot["batch"]["observations"]) != args.batch_size:
            raise ValueError("Captured rollout does not match Args batch size")
        if (
            not snapshot["minibatch_indices"]
            or not snapshot["immutable_inputs"]
            or not all(snapshot["immutable_inputs"].values())
        ):
            raise ValueError("Snapshot lacks completed update evidence or has mutated inputs")
        configure_runtime(cudnn_deterministic=args.torch_deterministic, matmul_precision="highest", allow_tf32=False)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            analyze(snapshot, path, trainer, args)


if __name__ == "__main__":
    main()
