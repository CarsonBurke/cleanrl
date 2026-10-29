"""Queued CUDA diagnosis on an atomic checkpoint's existing four probe draws.

No new action samples or modification to the live trainer. Optimizer steps use
independent checkpoint clones, including saved Adam moments and gradient clipping.
These counterfactuals do not reconstruct the historical optimizer update.
"""
import json
import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cleanrl.td7.delight import td7_latentq_delight_reference_v10 as t


def clipped_gaussian_log_probability(pre_action, mu, log_std):
    """Executed clipped-action law: unit Lebesgue interior and unit boundary atoms."""
    action = pre_action.detach().clamp(-1, 1)
    inverse_sigma = (-log_std).exp()
    interior = -.5 * ((action - mu) * inverse_sigma).square() - log_std - .5 * math.log(2 * math.pi)
    lower = torch.special.log_ndtr((-1 - mu) * inverse_sigma)
    upper = torch.special.log_ndtr((mu - 1) * inverse_sigma)
    return torch.where(action <= -1, lower, torch.where(action >= 1, upper, interior)).sum(-1)


def main():
    t.configure_runtime()
    saved = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    args = t.Args(**saved["args"])
    data = saved["agent"]
    state = data["buffers"]["probe_state"].cuda()
    zs = data["buffers"]["probe_zs"].cuda()
    eps = data["buffers"]["probe_eps"].cuda()
    state_dim, action_dim = state.shape[-1], eps.shape[-1]
    actor = t.Actor(state_dim, action_dim, args.zs_dim if args.actor_sale else 0, args.hidden_dim).cuda()
    critic = t.PreActionLatentTwinCritic(state_dim, action_dim, args.zs_dim, args.hidden_dim).cuda()
    encoder = t.Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).cuda()
    for name, model in [("actor", actor), ("critic", critic), ("fixed_encoder", encoder)]:
        model.load_state_dict(data["models"][name])
    agent = SimpleNamespace(args=args, actor=actor, critic=critic, fixed_encoder=encoder, state_dim=state_dim)
    mu, log_std = actor(state, zs)
    with torch.no_grad():
        normalizer = log_std.sum(-1) + action_dim / 2 * math.log(2 * math.pi)
        quantiles = torch.tensor([0., .1, .25, .5, .75, .9, 1.], device=state.device)
        sigma_quantiles = torch.quantile(log_std.exp().flatten(), quantiles).tolist()
        normalizer_quantiles = torch.quantile(normalizer, quantiles).tolist()
    results = {"checkpoint_step": saved["global_step"], "scope": "current-weight saved finite-batch draws; cloned one-step Adam counterfactuals", "quantile_probabilities": quantiles.tolist(), "sigma_quantiles": sigma_quantiles, "density_normalizer_quantiles": normalizer_quantiles, "references": {}}
    parameters = list(actor.parameters())
    sigma_mean_gradient = torch.autograd.grad(log_std.exp().mean(), parameters, retain_graph=True, allow_unused=True, materialize_grads=True)
    for reference in ["latent", "modal"]:
        with torch.no_grad():
            u, advantage = t.TD7Agent._advantage(agent, state, zs, mu, log_std.exp(), eps, reference)
            weight = t.delight_weight(advantage, eps, temperature=args.delight_temperature, cap=args.surprisal_cap, scale=args.gate_scale)[0]
            scale = advantage[..., None] * (eps.square() - 1)
            gated_scale = weight[..., None] * scale
            positive = advantage[..., None] > 0
            outer = eps.square() >= 1
            clipped = u.abs() > 1
            partitions = {}
            for sign, sign_mask in [("positive", positive), ("nonpositive", ~positive)]:
                for region, region_mask in [("inner", ~outer), ("outer", outer)]:
                    for clip, clip_mask in [("clipped", clipped), ("unclipped", ~clipped)]:
                        mask = sign_mask & region_mask & clip_mask
                        partitions[f"{sign}_{region}_{clip}"] = {"fraction": mask.float().mean().item(), "pg_scale": (scale * mask).mean().item(), "dg_scale": (gated_scale * mask).mean().item()}
        item = {"advantage_mean": advantage.mean().item(), "sigma_mean": log_std.exp().mean().item(), "sigma_per_coordinate": log_std.exp().mean(0).detach().tolist(), "ceiling_per_coordinate": (log_std > t.LOG_STD_MAX - .05).float().mean(0).tolist(), "clipped_per_coordinate": clipped.float().mean((0, 1)).tolist(), "scale_pg": scale.mean().item(), "scale_dg": gated_scale.mean().item(), "partitions": partitions, "gate_variants": {}}
        with torch.no_grad():
            item["advantage_quantiles"] = torch.quantile(advantage.abs().flatten(), quantiles).tolist()
            item["all_positive_state_fraction"] = (advantage > 0).all(0).float().mean().item()
            item["all_nonpositive_state_fraction"] = (advantage <= 0).all(0).float().mean().item()
        log_probability = t.gaussian_log_probability(u, mu, log_std)
        with torch.no_grad():
            density_surprise = (-log_probability).clamp(-args.surprisal_cap, args.surprisal_cap)
            units = advantage.square().mean().sqrt().clamp(min=1e-8) if args.gate_scale == "rms" else 1.0
            density_weight = 2 * torch.sigmoid(advantage / units * density_surprise / args.delight_temperature)
            rms_units = advantage.square().mean().sqrt().clamp(min=1e-8)
            relative_rms_weight = t.delight_weight(advantage, eps, temperature=args.delight_temperature, cap=args.surprisal_cap, scale="rms")[0]
            density_rms_weight = 2 * torch.sigmoid(advantage / rms_units * density_surprise / args.delight_temperature)
        physical_log_probability = clipped_gaussian_log_probability(u, mu, log_std)
        with torch.no_grad():
            physical_surprise = (-physical_log_probability).clamp(-args.surprisal_cap, args.surprisal_cap)
            physical_weight = 2 * torch.sigmoid(advantage / units * physical_surprise / args.delight_temperature)
        variants = [("pg", torch.ones_like(weight), log_probability),
                    ("delight_relative", weight, log_probability),
                    ("delight_relative_rms", relative_rms_weight, log_probability),
                    ("delight_density_doubled", density_weight, log_probability),
                    ("delight_density_rms_doubled", density_rms_weight, log_probability),
                    ("delight_physical_doubled", physical_weight, physical_log_probability)]
        for objective, gate, score in variants:
            loss = -(gate.detach() * advantage.detach() * score).mean()
            gradient = torch.autograd.grad(loss, parameters, retain_graph=True)
            # First-order change in mean sigma under raw gradient descent. Adam
            # and global clipping can change this; report no historical causality.
            item[f"{objective}_raw_descent_sigma_projection"] = -sum((a * b).sum() for a, b in zip(gradient, sigma_mean_gradient)).item()
            with torch.no_grad():
                gate_metrics = {"scale_per_coordinate": (gate[..., None] * scale).mean((0, 1)).tolist() if objective != "delight_physical_doubled" else None, "gate_mean": gate.mean().item(), "gate_suppressed": (gate < .02).float().mean().item(), "negative_surprise_fraction": (density_surprise < 0).float().mean().item() if objective == "delight_density_doubled" else (physical_surprise < 0).float().mean().item() if objective == "delight_physical_doubled" else None}
                if objective != "delight_physical_doubled":
                    # Pair averaging is essential: eps/sigma alone exaggerates
                    # floor-coordinate noise because even components cancel.
                    mean_coefficient = (gate[..., None] * advantage[..., None] * eps / log_std.exp()).mean(0)
                    floor, ceiling = log_std < t.LOG_STD_MIN + .05, log_std > t.LOG_STD_MAX - .05
                    gate_metrics["paired_mean_coefficient_bins"] = {
                        name: {"fraction": mask.float().mean().item(), "rms": mean_coefficient[mask].square().mean().sqrt().item() if mask.any() else None}
                        for name, mask in [("floor", floor), ("ceiling", ceiling), ("interior", ~(floor | ceiling))]
                    }
                    gate_metrics["scale_coefficient_bins"] = {
                        name: (gate[..., None] * scale)[:, mask].mean().item() if mask.any() else None
                        for name, mask in [("floor", floor), ("ceiling", ceiling), ("interior", ~(floor | ceiling))]
                    }
                    # Derivative of the bounded log_std mapping with respect
                    # to its unconstrained output; tiny values imply saturation.
                    tanh_output = 2 * (log_std - t.LOG_STD_MIN) / (t.LOG_STD_MAX - t.LOG_STD_MIN) - 1
                    slope = .5 * (t.LOG_STD_MAX - t.LOG_STD_MIN) * (1 - tanh_output.square())
                    gate_metrics["log_std_mapping_slope_bins"] = {
                        name: slope[mask].mean().item() if mask.any() else None
                        for name, mask in [("floor", floor), ("ceiling", ceiling), ("interior", ~(floor | ceiling))]
                    }
            # Identical original weights and saved Adam history for every arm.
            candidate = copy.deepcopy(actor)
            optimizer = torch.optim.Adam(candidate.parameters(), lr=args.actor_lr, fused=True)
            optimizer.load_state_dict(copy.deepcopy(data["optimizers"]["actor_optimizer"]))
            alpha = data["log_alpha"].cuda().exp() if args.entropy_floor else torch.zeros((), device="cuda")
            total = loss - alpha.detach() * log_std.sum(-1).mean()
            total_gradient = torch.autograd.grad(total, parameters, retain_graph=True)
            for parameter, grad in zip(candidate.parameters(), total_gradient):
                parameter.grad = grad.detach().clone()
            norm = torch.nn.utils.clip_grad_norm_(candidate.parameters(), args.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                next_mu, next_log_std = candidate(state, zs)
                gate_metrics.update(preclip_norm=norm.item(), sigma_mean_delta=(next_log_std.exp() - log_std.exp()).mean().item(), sigma_per_coordinate_delta=(next_log_std.exp() - log_std.exp()).mean(0).tolist(), geometric_sigma_delta=(next_log_std.mean().exp() - log_std.mean().exp()).item(), mean_action_delta=(next_mu - mu).square().mean().sqrt().item())
            item["gate_variants"][objective] = gate_metrics
        results["references"][reference] = item
    output = Path(sys.argv[2])
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
