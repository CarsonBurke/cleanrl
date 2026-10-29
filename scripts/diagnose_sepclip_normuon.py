"""Counterfactual PPO updates on identical frozen-policy rollouts; queue via mlq.

Not training evidence: cold NorMuon state differs from the ongoing Adam state.
Adam-reset controls and actor/critic fresh-training ablations address that limit.
"""
import argparse
import copy
import functools
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.distributions import Beta
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import awr_update_bench as bench
from cleanrl import ppo_continuous_action_sepclip_polar_express_normuon_v1 as m


def hidden(agent, network):
    return [(name, p) for name, p in agent.named_parameters()
            if name.startswith(network + ".") and ".head." not in name and p.ndim == 2]


def spectral_update(p, buf, second, grad, lr, momentum, beta2, scale, rank, power):
    """Change only singular-value weighting; preserve the reference polar norm."""
    buf.lerp_(grad, 1 - momentum)
    g = grad.lerp(buf, momentum)
    reference = m.polar_express(g).float()
    u, s, vh = torch.linalg.svd(g, full_matrices=False)
    weights = s.pow(power)
    # Do not invent a full-polar update in numerical null-space directions.
    tolerance = torch.finfo(s.dtype).eps * max(g.shape[-2:]) * s.amax(dim=-1, keepdim=True)
    weights = weights * (s > tolerance)
    if rank:
        weights = weights * (torch.arange(s.shape[-1], device=s.device) < rank)
    v = (u * weights.unsqueeze(-2)) @ vh
    v = v * reference.norm(dim=(-2, -1), keepdim=True) / v.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-10)
    second.lerp_(v.square().mean(dim=-1, keepdim=True), 1 - beta2)
    scaled = v * second.clamp_min(1e-10).rsqrt()
    v = scaled * v.norm(dim=(-2, -1), keepdim=True) / scaled.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-10)
    p.sub_(lr * scale * v)


@torch.no_grad()
def evaluate(agent, batch):
    a, b, values = agent.get_policy_and_value(batch["obs"])
    old = Beta(batch["alpha"], batch["beta"], validate_args=False)
    new = Beta(a, b, validate_args=False)
    delta = agent.action_logprob(a, b, batch["native"]) - batch["logprob"]
    scalars = torch.stack((
        torch.distributions.kl_divergence(old, new).sum(-1).mean(),
        (batch["advantages"] * delta.expm1()).mean(),
        (batch["advantages_mc"] * delta.expm1()).mean(),
        ((delta.exp() < .8) | (delta.exp() > 1.2)).float().mean(),
        (values.flatten() - batch["returns"]).square().mean(),
        (new.entropy() - old.entropy()).sum(-1).mean(),
        (torch.minimum(batch["advantages"] * delta.exp(),
                       batch["advantages"] * delta.exp().clamp(.8, 1.2)) - batch["advantages"]).mean(),
    )).cpu().tolist()
    return dict(zip(("exact_kl", "surrogate_gain", "mc_gain", "clipfrac", "value_mse", "entropy_delta",
                     "clipped_surrogate_gain"), scalars))


def restore_adam(opt, agent, snapshot, reset):
    source = snapshot["optimizer_state"]
    selected = {id(p) for group in opt.param_groups for p in group["params"]}
    for name, p in agent.named_parameters():
        reset_here = reset == "all" or (reset in ("actor", "critic", "both")
                     and name.split(".")[0] in (("actor", "critic") if reset == "both" else (reset,))
                     and ".head." not in name and p.ndim == 2)
        if id(p) in selected and not reset_here:
            opt.state[p] = {key: value.to(p.device).clone() if isinstance(value, torch.Tensor) else value
                            for key, value in source[name].items()}


def gradients(agent, batch, loss_fn):
    agent.zero_grad(set_to_none=True)
    loss, _ = loss_fn(batch["obs"], batch["native"], batch["logprob"], batch["advantages"],
                      batch["returns"], batch["values"])
    loss.backward()
    return {name: p.grad.detach().clone() for name, p in agent.named_parameters()}


@torch.no_grad()
def spectral_measurements(agent, train_grads, hold_grads):
    rows = []
    for network in ("actor", "critic"):
        network_norm = torch.sqrt(sum(g.square().sum() for name, g in train_grads.items()
                                      if name.startswith(network + ".")))
        clip_scale = (.5 / (network_norm + 1e-6)).clamp(max=1)
        for name, p in hidden(agent, network):
            g, h = train_grads[name] * clip_scale, hold_grads[name]
            u, singular, vh = torch.linalg.svd(g, full_matrices=False)
            exact = u @ vh
            approx = m.polar_express(g).float()
            fp32 = g.mT if g.shape[0] > g.shape[1] else g
            fp32 = fp32 / (fp32.norm() * 1.02 + 1e-6)
            for a, b, c in m.POLAR_EXPRESS:
                gram = fp32 @ fp32.mT
                fp32 = a * fp32 + (b * gram + c * (gram @ gram)) @ fp32
            if g.shape[0] > g.shape[1]:
                fp32 = fp32.mT
            # Remove radial components using the same row/column axes as nGPT.
            dim = 0 if ".down." in name else 1
            tangent = lambda d: d - p * (p * d).sum(dim=dim, keepdim=True)
            cosine = lambda x, y: (x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-20)
            projected = tangent(approx)
            top5 = tangent(u[:, :5] @ vh[:5])
            vals = torch.stack((singular[0] / singular[-1].clamp_min(1e-20),
                                singular[:5].sum() / singular.sum(),
                                singular[:5].square().sum() / singular.square().sum(),
                                cosine(g, h), cosine(projected, h), cosine(tangent(exact), h),
                                (approx - exact).norm() / exact.norm(),
                                projected.norm() / approx.norm(), cosine(approx, fp32),
                                (approx - fp32).norm() / fp32.norm(), cosine(top5, h)))
            rows.append(dict(parameter=name, **dict(zip((
                "condition_number", "top5_singular_mass", "top5_gradient_energy",
                "train_hold_gradient_cosine", "polar_hold_cosine", "exact_polar_hold_cosine",
                "bf16_polar_relative_error", "tangent_fraction", "bf16_fp32_cosine",
                "bf16_fp32_relative_error", "top5_polar_hold_cosine"), vals.cpu().tolist()))))
    return rows


def diagnose(snapshot_path, chunks, writer, step_counter, controls_only=False, spectral_only=False):
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    if "optimizer_state" not in snapshot:
        raise ValueError("format-3 snapshot with full Adam state is required")
    args = SimpleNamespace(**snapshot["args"])
    envs = m.make_training_env(args, "normuon_diagnostic")
    try:
        agent = m.Agent(envs).cuda()
    finally:
        envs.close()
    agent.load_state_dict(snapshot["model"])
    base = copy.deepcopy(agent.state_dict())
    with torch.no_grad():
        batches, episodic_returns = bench.collect(m, agent, snapshot, chunks, "cuda", 1)
        for batch in batches:
            batch["values"] = agent.get_value(batch["obs"]).flatten().clone()
            batch["returns"] = batch["values"] + batch["advantages"]
    train, hold = batches[0], bench.concatenate(batches[1:])
    loss_fn = torch.compile(lambda obs, native, old, adv, returns, values:
                            m.ppo_loss(agent, obs, native, old, adv, returns, values, args),
                            fullgraph=True, options={"triton.cudagraphs": False})
    train_grads = gradients(agent, train, loss_fn)
    hold_grads = gradients(agent, hold, loss_fn)
    spectral = spectral_measurements(agent, train_grads, hold_grads)
    fraction = snapshot["learning_rate"] / args.learning_rate
    experiments = [("adam_native", None, 0.0, .95, None),
                   ("adam_reset", None, 0.0, .95, "all")]
    experiments += [(f"adam_{network}_hidden_reset", None, 0.0, .95, network)
                    for network in ("actor", "critic", "both")]
    experiments += [] if controls_only else [(f"{network}_muon_lr{lr:g}_m{momentum:g}", network, lr, momentum, None)
                    for network in ("actor", "critic", "both")
                    for lr, momentum in ((.0025, .95), (.005, .95), (.01, .95), (.02, .95),
                                         (.005, .9), (.005, 0.0))]
    if spectral_only:
        experiments = [("adam_native", None, 0., .95, None)]
        experiments += [(f"actor_{kind}_lr{lr:g}", "actor", lr, .95, None)
                        for kind in ("muon", "svd_full", "svd_top5", "svd_sqrt")
                        for lr in (.00125, .0025, .005)]
    results = []
    for label, network, lr, momentum, reset in experiments:
        agent.load_state_dict(base)
        selected = [(name, p) for trunk in ("actor", "critic") if network in (trunk, "both")
                    for name, p in hidden(agent, trunk)]
        ids = {id(p) for _, p in selected}
        auxiliary = [p for p in agent.parameters() if id(p) not in ids]
        adam = torch.optim.Adam(auxiliary, lr=snapshot["learning_rate"], eps=1e-5, fused=True)
        restore_adam(adam, agent, snapshot, reset)
        muon = m.PolarExpressNorMuon([p for _, p in selected], lr=lr * fraction,
                                    momentum=momentum) if selected else None
        if "svd_" in label:
            rank = 5 if "svd_top5" in label else 0
            power = .5 if "svd_sqrt" in label else 0.
            muon.update = torch.compile(functools.partial(spectral_update, rank=rank, power=power),
                                        fullgraph=True)
        epochs = []
        for epoch in range(args.update_epochs):
            before = {name: p.detach().clone() for name, p in agent.named_parameters()}
            current = gradients(agent, train, loss_fn)
            torch.nn.utils.clip_grad_norm_(tuple(agent.actor.parameters()), args.max_grad_norm, foreach=True)
            torch.nn.utils.clip_grad_norm_(tuple(agent.critic.parameters()), args.max_grad_norm, foreach=True)
            if muon:
                muon.step()
            adam.step()
            agent.normalize_matrices()
            # Descent alignment is measured with the actual projected displacement.
            alignment = {}
            for trunk in ("actor", "critic"):
                params = hidden(agent, trunk)
                update = torch.cat([(p - before[name]).flatten() for name, p in params])
                g = torch.cat([current[name].flatten() for name, _ in params])
                hg = torch.cat([hold_grads[name].flatten() for name, _ in params])
                vals = torch.stack((update.norm(), -(update * g).sum(), -(update * hg).sum(),
                                    -(update * g).sum() / (update.norm() * g.norm()).clamp_min(1e-20)))
                alignment[trunk] = dict(zip(("step_norm", "train_linear_gain", "hold_initial_linear_gain",
                                             "descent_cosine"), vals.detach().cpu().tolist()))
            train_metrics, hold_metrics = evaluate(agent, train), evaluate(agent, hold)
            epochs.append(dict(epoch=epoch+1, train=train_metrics, hold=hold_metrics, alignment=alignment))
            step_counter += 1
            for tag, value in hold_metrics.items():
                writer.add_scalar(f"{snapshot['global_step']}/{label}/{tag}", value, step_counter)
        results.append(dict(label=label, epochs=epochs))
        results[-1]["heldout_chunks_final"] = [evaluate(agent, chunk) for chunk in batches[1:]]
        print(json.dumps(dict(snapshot_step=snapshot["global_step"], label=label, final=epochs[-1])), flush=True)
    return dict(snapshot=str(snapshot_path), step=snapshot["global_step"],
                frozen_policy_returns=episodic_returns, spectral=spectral, results=results), step_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshots", nargs="+", type=Path)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument("--spectral-only", action="store_true")
    cli = parser.parse_args()
    if cli.chunks < 3:
        raise ValueError("at least two independent held-out chunks required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    m.configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    # This diagnostic intentionally crosses network membership, shape and
    # momentum settings; the ordinary eight-graph cache cannot hold the grid.
    torch._dynamo.config.recompile_limit = 64
    torch.manual_seed(1)
    output, counter = [], 0
    with SummaryWriter(f"runs/HalfCheetah-v4__sepclip_normuon_causal_bench__1__{int(time.time())}") as writer:
        for snapshot in cli.snapshots:
            row, counter = diagnose(snapshot, cli.chunks, writer, counter, cli.controls_only, cli.spectral_only)
            output.append(row)
            cli.out.parent.mkdir(parents=True, exist_ok=True)
            cli.out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
