"""CUDA checkpoint audit of latent/physical-point internal consistency.

This is an evaluation, not training: fresh states, two independent antithetic
quadrature panels, and full input derivatives. Agreement does not establish
correct environmental values or policy gradients. Run only through mlq.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.runtime import configure_runtime
from cleanrl.td7.latentq.td7_latentq_preaction_sparse_critic_v13 import (
    Actor, Encoder, PreActionLatentTwinCritic,
)


@dataclass
class Args:
    checkpoint: Path
    out: Path
    exp_name: str = "td7_latentq_checkpoint_consistency_audit"


TRANSITIONS = 32768
STATES = 256
NUM_ENVS = 16
CHUNK = 16
DRAWS = 128  # total actions: 64 independent normal draws and their negatives
SEED = 1
MAX_SECONDS = 19 * 60  # queue's external cap should be 20 minutes


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comparison(candidate, reference):
    """Per-state derivative comparison; undefined zero-vector cosines are excluded."""
    candidate = candidate.flatten(1)
    reference = reference.flatten(1)
    cn = candidate.norm(dim=1)
    rn = reference.norm(dim=1)
    valid = (cn > 1e-10) & (rn > 1e-10)
    cosine = (candidate * reference).sum(1) / (cn * rn).clamp_min(1e-10)
    return {
        "rms_mismatch": (candidate - reference).square().mean().sqrt().item(),
        "relative_rms_mismatch": ((candidate - reference).square().mean().sqrt()
                                  / reference.square().mean().sqrt().clamp_min(1e-10)).item(),
        "candidate_rms": candidate.square().mean().sqrt().item(),
        "reference_rms": reference.square().mean().sqrt().item(),
        "mean_state_cosine": cosine[valid].mean().item() if valid.any() else None,
        "valid_cosine_states": valid.sum().item(),
        "global_norm_ratio": (candidate.norm() / reference.norm().clamp_min(1e-10)).item(),
    }


def value_comparison(candidate, reference):
    residual = candidate - reference
    return {
        "rms_mismatch": residual.square().mean().sqrt().item(),
        "relative_rms_mismatch": (residual.square().mean().sqrt()
                                  / reference.square().mean().sqrt().clamp_min(1e-10)).item(),
        "bias": residual.mean().item(),
        "candidate_mean": candidate.mean().item(),
        "reference_mean": reference.mean().item(),
    }


def derivatives(forward, state, mu, sigma, *extra):
    mu = mu.detach().clone().requires_grad_(True)
    sigma = sigma.detach().clone().requires_grad_(True)
    values = forward(state, mu, sigma, *extra)
    grads = []
    for twin in range(2):
        gm, gs = torch.autograd.grad(values[:, twin].sum(), (mu, sigma), retain_graph=twin == 0)
        grads.append(torch.stack((gm.detach(), gs.detach()), dim=1))
    # values: [state,twin], gradients: [state,twin,mu_or_sigma,action]
    return values.detach(), torch.stack(grads, dim=1)


def mechanism_report(center, latent_grad, panel_values, panel_grads):
    reference = (panel_values[0] + panel_values[1]) / 2
    ref_grad = (panel_grads[0] + panel_grads[1]) / 2
    values = {}
    gradients = {}
    reliability = {}
    for twin in range(2):
        name = f"twin{twin}"
        values[name] = value_comparison(center[:, twin], reference[:, twin])
        reliability[f"value_{name}"] = value_comparison(panel_values[0][:, twin], panel_values[1][:, twin])
    values["conservative_backup"] = value_comparison(center.min(1).values, reference.min(1).values)
    reliability["value_conservative_backup"] = value_comparison(
        panel_values[0].min(1).values, panel_values[1].min(1).values)
    for name, lg, rg, p0, p1 in [
        ("twin0", latent_grad[:, 0], ref_grad[:, 0], panel_grads[0][:, 0], panel_grads[1][:, 0]),
        ("twin1", latent_grad[:, 1], ref_grad[:, 1], panel_grads[0][:, 1], panel_grads[1][:, 1]),
        ("twin_mean", latent_grad.mean(1), ref_grad.mean(1), panel_grads[0].mean(1), panel_grads[1].mean(1)),
    ]:
        for component, index in (("mu", 0), ("sigma", 1), ("joint", None)):
            a, b, c, d = (x if index is None else x[:, index] for x in (lg, rg, p0, p1))
            key = f"{name}_{component}"
            gradients[key] = comparison(a, b)
            reliability[f"gradient_{key}"] = comparison(c, d)
            # Full independent-panel difference is conservative relative to pooled-panel error.
            denominator = gradients[key]["rms_mismatch"]
            reliability[f"gradient_{key}"]["panel_disagreement_over_latent_mismatch"] = (
                reliability[f"gradient_{key}"]["rms_mismatch"] / max(denominator, 1e-10))
    return {"center_vs_point_convolution": values, "latent_vs_reparameterized_gradients": gradients,
            "independent_panel_reliability": reliability}


def main(args: Args):
    started = time.monotonic()
    configure_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError("This audit requires CUDA; CPU fallback is forbidden")
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_hash = sha256(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_args = saved["args"]
    saved_step = saved["global_step"]
    env_id = source_args["env_id"]
    run_name = f"{env_id}__{args.exp_name}__{SEED}__{time.time_ns() // 1000000000}"
    writer = SummaryWriter(f"runs/{run_name}")
    envs = None

    def deadline():
        if time.monotonic() - started > MAX_SECONDS:
            raise TimeoutError("Checkpoint audit exceeded its 19 minute internal budget")

    try:
        writer.add_text("hyperparameters", json.dumps({**asdict(args), "checkpoint": str(checkpoint_path),
                        "out": str(args.out), "seed": SEED, "env_id": env_id,
                        "total_timesteps": TRANSITIONS, "evaluation_only": True}, default=str))
        envs = make_mujoco_vector_env(env_id, NUM_ENVS, num_threads=2)
        state_dim = int(np.prod(envs.single_observation_space.shape))
        action_dim = int(np.prod(envs.single_action_space.shape))
        zs_dim, hidden_dim = source_args["zs_dim"], source_args["hidden_dim"]
        # Construct parameters on CUDA from the first allocation, never execute a CPU model.
        with torch.device(device):
            actor = Actor(state_dim, action_dim, zs_dim, hidden_dim)
            encoder = Encoder(state_dim, action_dim, zs_dim, hidden_dim)
            critic = PreActionLatentTwinCritic(state_dim, action_dim, zs_dim, hidden_dim)
        actor.load_state_dict(saved["actor"])
        encoder.load_state_dict(saved["fixed_encoder"])
        critic.load_state_dict(saved["critic"])
        for model in (actor, encoder, critic):
            model.eval().requires_grad_(False)
        del saved

        def policy(state):
            mu, log_std = actor(state, encoder.zs(state))
            return mu, log_std.exp()

        def center_forward(state, mu, sigma):
            zs = encoder.zs(state)
            return critic(state, torch.cat((mu, sigma), 1), mu, encoder.zsa(zs, mu.clamp(-1, 1)), zs)

        def convolution_forward(state, mu, sigma, eps):
            n = state.shape[0]
            actions = (mu[:, None] + sigma[:, None] * eps).clamp(-1, 1).reshape(n * DRAWS, action_dim)
            repeated_state = state[:, None].expand(-1, DRAWS, -1).reshape(n * DRAWS, state_dim)
            zs = encoder.zs(state)[:, None].expand(-1, DRAWS, -1).reshape(n * DRAWS, zs_dim)
            point = critic(repeated_state, torch.cat((actions, torch.zeros_like(actions)), 1), actions,
                           encoder.zsa(zs, actions), zs)
            return point.reshape(n, DRAWS, 2).mean(1)

        # Fullgraph forbids hidden eager fallbacks. Default mode avoids CUDA graph
        # lifetime aliasing across the two input-gradient calls on each twin.
        policy_compiled = torch.compile(policy, fullgraph=True)
        center_compiled = torch.compile(center_forward, fullgraph=True)
        convolution_compiled = torch.compile(convolution_forward, fullgraph=True)
        observations = torch.empty(TRANSITIONS, state_dim, device=device)
        means = torch.empty(TRANSITIONS, action_dim, device=device)
        stds = torch.empty_like(means)
        obs, _ = envs.reset(seed=SEED)
        collection_rng = torch.Generator(device=device).manual_seed(SEED)
        low = torch.as_tensor(envs.single_action_space.low, device=device)
        high = torch.as_tensor(envs.single_action_space.high, device=device)
        if not torch.allclose(high, -low):
            raise ValueError("Frozen v13 assumes a symmetric action space")
        # v13's critic sees normalized [-1,1] actions; env receives max_action-scaled actions.
        max_action = float(envs.single_action_space.high[0])
        if not np.all(envs.single_action_space.high == max_action):
            raise ValueError("Frozen v13 assumes identical action bounds")
        with torch.no_grad():
            for offset in range(0, TRANSITIONS, NUM_ENVS):
                deadline()
                state = torch.as_tensor(obs, dtype=torch.float32, device=device)
                mu, sigma = policy_compiled(state)
                observations[offset:offset + NUM_ENVS].copy_(state)
                means[offset:offset + NUM_ENVS].copy_(mu)
                stds[offset:offset + NUM_ENVS].copy_(sigma)
                eps = torch.randn(mu.shape, device=device, generator=collection_rng)
                action = (mu + sigma * eps).clamp(-1, 1) * max_action
                obs, _, _, _, _ = envs.step(action.cpu().numpy())
                if offset % 4096 == 0:
                    print(f"audit collected={offset + NUM_ENVS}/{TRANSITIONS}", flush=True)
        # 16 evenly spaced time strata x all 16 lanes: exact lane balance.
        selected_times = torch.linspace(0, TRANSITIONS // NUM_ENVS - 1, STATES // NUM_ENVS,
                                        device=device).round().long()
        indices = (selected_times[:, None] * NUM_ENVS + torch.arange(NUM_ENVS, device=device)).flatten()
        states, mus, behavior_sigma = observations[indices], means[indices], stds[indices]
        del observations, means, stds
        # Two local independent streams; quadrature never advances collection RNG.
        panel_generators = [torch.Generator(device=device).manual_seed(SEED + 10001 + p) for p in range(2)]
        panels = []
        for generator in panel_generators:
            eps = torch.randn(STATES, DRAWS // 2, action_dim, generator=generator, device=device)
            panels.append(torch.cat((eps, -eps), dim=1))
        mechanisms = {}
        for name, sigmas in (("behavior_sigma", behavior_sigma),
                             ("constant_sigma_0.1", torch.full_like(behavior_sigma, 0.1)),
                             ("constant_sigma_0.2", torch.full_like(behavior_sigma, 0.2))):
            center_values, center_gradients = [], []
            values, gradients = [[], []], [[], []]
            for offset in range(0, STATES, CHUNK):
                deadline()
                sl = slice(offset, offset + CHUNK)
                value, gradient = derivatives(center_compiled, states[sl], mus[sl], sigmas[sl])
                center_values.append(value)
                center_gradients.append(gradient)
                for p in range(2):
                    value, gradient = derivatives(convolution_compiled, states[sl], mus[sl], sigmas[sl], panels[p][sl])
                    values[p].append(value)
                    gradients[p].append(gradient)
            mechanisms[name] = mechanism_report(torch.cat(center_values), torch.cat(center_gradients),
                                                [torch.cat(x) for x in values], [torch.cat(x) for x in gradients])
            print(f"audit query={name} complete", flush=True)
        elapsed = time.monotonic() - started
        result = {
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
            "source_args": source_args, "saved_step": saved_step,
            "audit_source_sha256": sha256(__file__), "run_name": run_name,
            "device": str(device), "elapsed_seconds": elapsed, "seed": SEED,
            "collection_transitions": TRANSITIONS, "sampled_states": STATES,
            "sampling": "16 evenly spaced collection times, all 16 lanes per time",
            "quadrature": {"independent_panels": 2, "actions_per_panel": DRAWS,
                           "antithetic_pairs_per_panel": DRAWS // 2, "seed_offsets": [10001, 10002]},
            "query_definition": "center F(s,mu,sigma,mu); point F(s,a,0,a), a=clip(mu+sigma*eps)",
            "backup_definition": "min over twins after each twin's quadrature expectation; no reward/discount/cap",
            "mechanisms": mechanisms,
            "caveats": [
                "Internal consistency only: physical point-Q is not an oracle for environmental value or gradient accuracy.",
                "Fresh policy states were not fitting rows; no environmental return target is computed.",
                "Both views share critic and fixed SALE encoder, so correlated errors can agree.",
                "Independent panel disagreement measures quadrature noise, not model error; two panels give limited precision.",
                "Antithetic clipping includes its zero derivative outside [-1,1]; samples exactly at kinks have measure zero.",
                "Constant sigma queries hold behavior mu fixed and may lie outside training distribution coverage.",
                "Gradients include all latent, pre-action and action-conditioned SALE paths; encoder state features are fixed.",
                "These are online checkpoint networks, not saved target networks; backup comparison excludes TD7 target clipping.",
            ],
        }

        def emit(prefix, node):
            for key, value in node.items():
                tag = f"{prefix}/{key}"
                if isinstance(value, dict):
                    emit(tag, value)
                elif isinstance(value, (int, float)) and value is not None:
                    writer.add_scalar(tag, value, TRANSITIONS)
        emit("checkpoint_audit", mechanisms)
        writer.add_scalar("charts/SPS", TRANSITIONS / elapsed, TRANSITIONS)
        writer.add_scalar("checkpoint_audit/elapsed_seconds", elapsed, TRANSITIONS)
        writer.flush()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(f".{args.out.name}.{os.getpid()}.partial")
        try:
            with temporary.open("w") as handle:
                json.dump(result, handle, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, args.out)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"audit output={args.out} elapsed={elapsed:.1f}s", flush=True)
    finally:
        if envs is not None:
            envs.close()
        writer.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
