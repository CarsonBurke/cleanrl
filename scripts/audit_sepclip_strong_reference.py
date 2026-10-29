"""Frozen-row physical replay of a saved 50M pre-RMS Beta reference.

CUDA/mlq only. This unmatched-budget historical policy is a physical reference,
not a causal optimizer comparison, a matched benchmark, or an optimum oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cleanrl import ppo_continuous_action_prerms_adamh_v22 as trainer
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.runtime import configure_runtime
from scripts import audit_sepclip_physics as physics
from scripts.audit_sepclip_gait import CONTACT_GEOMS, ground_contacts

DEFAULT_CHECKPOINT = ROOT / "runs/HalfCheetah-v4__prerms_adamh_v22_control_50M__1__1788906876/prerms_adamh_v22_control_50M.cleanrl_model"
EXPECTED_HASH = "1f09a49740f656482f4e01a71f1070cfe13ebd1e44a52210eca8db73f0089532"


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_metadata(snapshot):
    args = trainer.Args(**snapshot["args"])
    if (args.env_id, args.num_envs, args.placement, args.norm_kind, args.activation) != (
            "HalfCheetah-v4", 16, "pre", "rms", "stiglu"):
        raise ValueError("Reference architecture/environment/row count differs from recorded policy")
    norm = snapshot["obs_norm"]
    if float(norm["epsilon"]) != 1e-8 or float(norm["clip"]) != 10:
        raise ValueError("Unsupported recorded observation epsilon/clip")
    arrays = {}
    for name, shape in (("means", (16, 17)), ("variances", (16, 17)), ("counts", (16,))):
        value = norm[name]
        x = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        if x.shape != shape or x.dtype != np.float64 or not np.isfinite(x).all():
            raise ValueError(f"Invalid reference observation normalizer {name}")
        if name != "means" and np.any(x < 0 if name == "variances" else x <= 0):
            raise ValueError(f"Invalid reference observation normalizer {name}")
        arrays[name] = dict(shape=list(x.shape), dtype=str(x.dtype),
                            sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest())
    return args, dict(arrays=arrays, epsilon=1e-8, clip=10., rows=list(range(16)))


def validate_bounds(agent, envs):
    actual = {"action_low": envs.single_action_space.low,
              "action_high": envs.single_action_space.high}
    for name, bound in actual.items():
        if not np.array_equal(getattr(agent, name).detach().cpu().numpy(), bound):
            raise ValueError(f"Checkpoint {name} disagrees with environment")
    low = agent.action_low.detach().cpu().numpy()
    high = agent.action_high.detach().cpu().numpy()
    if not np.array_equal(low, np.full(6, -1, np.float32)) or not np.array_equal(high, np.ones(6, np.float32)):
        raise ValueError("Reference requires six [-1,1] physical action bounds")
    np.testing.assert_array_equal(agent.action_scale.detach().cpu().numpy(), high-low)
    np.testing.assert_allclose(agent.log_action_scale.detach().cpu().numpy(), np.log(high-low), atol=0, rtol=0)


def load_agent(snapshot, envs):
    args, metadata = validate_metadata(snapshot)
    agent = trainer.Agent(envs, placement=args.placement, norm_kind=args.norm_kind,
                          activation=args.activation).cuda().eval()
    agent.load_state_dict(snapshot["model"], strict=True)
    validate_bounds(agent, envs)
    if not all(bool(torch.isfinite(p).all()) for p in agent.parameters()):
        raise ValueError("Nonfinite reference parameters")
    return agent, metadata


def compiled_policy(agent):
    def policy(x):
        z = agent.actor(x)
        a, b = (1 + F.softplus(z)).chunk(2, -1)
        return z, a, b
    return torch.compile(policy, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})


def checked_host_policy(mirror, cuda_policy, obs):
    # Both paths may own permanent buffers; materialize before the next call.
    a, b, z = physics.policy(mirror, obs)
    with torch.no_grad():
        expected = [v.detach().cpu().numpy().copy() for v in cuda_policy(torch.as_tensor(obs, device="cuda"))]
    errors = {}
    for label, actual, target in zip(("logits", "alpha", "beta"), (z, a, b), expected):
        if not np.isfinite(actual).all() or not np.isfinite(target).all():
            raise ValueError(f"Nonfinite {label}")
        error = float(np.max(np.abs(actual-target)))
        scale = max(1., float(np.max(np.abs(target))))
        if error > 2e-5 + 2e-6 * scale:
            raise ValueError(f"Visited-observation host/CUDA {label} mismatch {error}, scale {scale}")
        errors[label] = error
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("Invalid nonpositive FP32 Beta shape")
    return a.copy(), b.copy(), z.copy(), errors


def reference_native_action(alpha, beta, mode, uniforms):
    # Mode of the actual rounded FP32 Beta law. Alpha=beta=1 is uniform;
    # choose its midpoint as an explicit deterministic tie convention.
    if mode == "mode":
        excess_a = alpha-1
        excess_b = beta-1
        denominator = excess_a+excess_b
        return np.divide(excess_a,denominator,out=np.full_like(alpha,.5),
                         where=denominator>0).astype(np.float32)
    return physics.native_action(alpha, beta, mode, uniforms, logits=None)


def evaluate(agent, snapshot, mode, seed=1, trace_path=None):
    envs = make_mujoco_vector_env("HalfCheetah-v4", 16, num_threads=2)
    try:
        norm = physics.norm_from_snapshot(snapshot, np.arange(16))
        norm.epsilon = float(snapshot["obs_norm"]["epsilon"])
        norm.clip = float(snapshot["obs_norm"]["clip"])
        mirror = make_host_mirror(agent.actor, 16)
        policy = compiled_policy(agent)
        raw, _ = envs.reset(seed=seed)
        obs = physics.normalized(norm, raw)
        rng = np.random.default_rng(seed)
        returns = np.zeros(16)
        data = {name: [] for name in ("velocity", "control_cost", "action", "concentration",
                "mean", "mode", "ground_contact", "ground_normal_force", "qpos", "qvel")}
        parity = dict(logits=0., alpha=0., beta=0.)
        rounded = dict(alpha_equal_one=0,beta_equal_one=0,both_equal_one=0,
                       intended_mode_nonfinite_dimensions=0,
                       intended_vs_rounded_native_mode_max_abs_error=0.)
        for t in range(1000):
            a, b, z, errors = checked_host_policy(mirror, policy, obs)
            parity = {k: max(parity[k], errors[k]) for k in parity}
            uniforms = rng.uniform(1e-7, 1-1e-7, a.shape)
            native = reference_native_action(a, b, mode, uniforms)
            actual_mode = reference_native_action(a,b,"mode",uniforms)
            with np.errstate(divide="ignore",invalid="ignore"):
                intended_mode = physics.native_action(a,b,"mode",uniforms,z)
            rounded["alpha_equal_one"] += int(np.count_nonzero(a==1))
            rounded["beta_equal_one"] += int(np.count_nonzero(b==1))
            rounded["both_equal_one"] += int(np.count_nonzero((a==1)&(b==1)))
            finite_intended = np.isfinite(intended_mode)
            rounded["intended_mode_nonfinite_dimensions"] += int(np.count_nonzero(~finite_intended))
            if finite_intended.any():
                rounded["intended_vs_rounded_native_mode_max_abs_error"] = max(
                    rounded["intended_vs_rounded_native_mode_max_abs_error"],
                    float(np.max(np.abs(intended_mode[finite_intended]-actual_mode[finite_intended]))))
            action = (2*native-1).astype(np.float32)
            contacts = [ground_contacts(e.unwrapped) for e in envs.envs]
            data["ground_contact"].append(np.stack([c[0] for c in contacts]))
            data["ground_normal_force"].append(np.stack([c[1] for c in contacts]))
            for k in ("qpos", "qvel"):
                data[k].append(np.stack([getattr(e.unwrapped.data, k).copy() for e in envs.envs]))
            raw, reward, terms, truncs, _ = envs.step(action)
            if np.any(terms) or (t < 999 and np.any(truncs)) or (t == 999 and not np.all(truncs)):
                raise ValueError("Unexpected reference evaluation episode boundary")
            cost = envs.envs[0].unwrapped._ctrl_cost_weight*np.square(action).sum(-1)
            velocity = (reward+cost)/envs.envs[0].unwrapped._forward_reward_weight
            for name, value in (("velocity", velocity), ("control_cost", cost), ("action", action),
                    ("concentration", a+b), ("mean", 2*a/(a+b)-1),
                    ("mode", 2*actual_mode-1)):
                data[name].append(value.copy())
            returns += reward
            obs = physics.normalized(norm, raw)
        arrays = {k: np.asarray(v) for k, v in data.items()}
        if trace_path is not None:
            np.savez_compressed(trace_path, **arrays, normalization_rows=np.arange(16),
                                dt=envs.envs[0].unwrapped.dt)
        return dict(mode=mode, normalization="frozen exact row0..15", returns=returns.tolist(),
            return_summary=physics.summarize(returns), visited_host_cuda_max_abs_error=parity,
            rounded_shape_diagnostics=rounded,
            mode_law="Actual FP32 Beta shapes; alpha=beta=1 uniform midpoint tie=.5. Intended logit-excess mode difference reported separately",
            metrics={k: physics.summarize(arrays[k]) for k in ("velocity", "control_cost", "concentration")},
            action_abs=physics.summarize(np.abs(arrays["action"])),
            action_change_squared=physics.summarize(np.square(np.diff(arrays["action"],axis=0)).sum(-1)),
            per_joint_mean_abs_action=np.abs(arrays["action"]).mean((0,1)).tolist(),
            torque_fraction={str(c): float(np.mean(np.abs(arrays["action"])>=c)) for c in (.8,.9,.95,.98)},
            ground_geometry_order=list(CONTACT_GEOMS),
            ground_duty_fraction=arrays["ground_contact"].mean((0,1)).tolist(),
            ground_normal_force_mean=arrays["ground_normal_force"].mean((0,1)).tolist(),
            trace_path=str(trace_path) if trace_path else None,
            metric_uncertainty="Step metric SEs are descriptive and ignore temporal correlation; episode paired-return differences are primary",
            contact_timing="Pre-action contact/force at final solver substep of prior step, not contact impulse")
    finally:
        envs.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--expected-checkpoint-sha256", default=EXPECTED_HASH)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-traces", action="store_true")
    cli = parser.parse_args()
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; submit through mlq")
    digest = file_hash(cli.checkpoint)
    if digest != cli.expected_checkpoint_sha256:
        raise ValueError("Reference checkpoint SHA256 differs from selected inventory")
    snapshot = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    envs = make_mujoco_vector_env("HalfCheetah-v4", 16, num_threads=2)
    try:
        agent, norm = load_agent(snapshot, envs)
    finally:
        envs.close()
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(trainer.__file__), ROOT/"scripts/audit_sepclip_physics.py",
        ROOT/"scripts/audit_sepclip_gait.py"] + [ROOT/"cleanrl/shared"/n for n in
        ("host_graph.py", "host_actor.py", "host_kernel.c", "norm_residual.py", "pre_rms.py",
         "vector_norm.py", "mujoco_env.py", "runtime.py", "sampling.py")]
    result = dict(checkpoint=str(cli.checkpoint.resolve()), checkpoint_sha256=digest,
        source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in sources},
        historical_execution_source_pinned=False, observation_normalizer=norm,
        checkpoint_args=snapshot["args"], protocol=dict(episodes=16,seed=1,steps_per_episode=1000,
        modes=["sample","mean","mode"],paired_reset_seeds=True,normalization="frozen exact rows"),
        environment=dict(mujoco_version=mujoco.__version__), results=[],
        limitations=["50M training budget is unmatched to sepclip8M; architecture also differs.",
            "Current source strictly loads the saved state but historical execution source was not pinned.",
            "Frozen-normalizer inverse-CDF evaluation differs from historical live gamma sampler/online normalization.",
            "Sample/mean/mode are different control laws; this is physical comparison, not causal PG/optimizer attribution or optimum certification.",
            "Mode uses actual FP32 Beta shapes with uniform midpoint tie, while historical sepclip mode helper used intended softplus-excess logits; rounding discrepancies reported."])
    for mode in ("sample","mean","mode"):
        trace = cli.out.parent/f"strong_reference_{digest[:12]}_{mode}.npz" if cli.save_traces else None
        row = evaluate(agent,snapshot,mode,trace_path=trace)
        result["results"].append(row)
        cli.out.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(dict(mode=mode,return_summary=row["return_summary"])),flush=True)
    old = np.asarray(result["results"][0]["returns"])
    result["paired_return_differences"]={r["mode"]:physics.summarize(np.asarray(r["returns"])-old)
                                          for r in result["results"][1:]}
    if file_hash(cli.checkpoint)!=digest:
        raise ValueError("Reference checkpoint changed during evaluation")
    cli.out.write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
