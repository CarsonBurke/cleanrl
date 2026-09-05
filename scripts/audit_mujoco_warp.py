#!/usr/bin/env python3
"""Audit GPU MuJoCo against the repository's actual v4 physics, without training.

Run BOTH stages through mlq. Export uses the repository .venv; audit uses an
isolated modern MuJoCo/MJWarp environment. Neither stage changes engine pins.
The GPU workload uses full original episode trajectories, CUDA graph capture,
and resident state/action fixtures. Its throughput is physics replay throughput,
not a claim about end-to-end policy training or learned-policy equivalence.

Example stages (wrap each command in mlq submit --max-parallel-runs 1):
  .venv/bin/python scripts/audit_mujoco_warp.py export --dataset-dir artifacts/mjwarp-v1
  /path/to/isolated/bin/python scripts/audit_mujoco_warp.py audit \
      --dataset-dir artifacts/mjwarp-v1 --output artifacts/mjwarp-v1/audit.json
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DOCS_URL = "https://mujoco.readthedocs.io/en/stable/mjwarp/index.html"
ENV_IDS = ("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4")
STATE_FIELDS = ("qpos", "qvel", "qacc_warmstart")
MODEL_FIELDS = (
    "qpos0", "body_mass", "body_inertia", "body_ipos", "body_iquat", "dof_armature",
    "dof_damping", "geom_pos", "geom_size", "geom_friction", "geom_solref", "geom_solimp",
    "jnt_pos", "jnt_axis", "actuator_gear", "actuator_gainprm", "actuator_biasprm",
)
OPTION_FIELDS = (
    "timestep", "integrator", "solver", "iterations", "tolerance", "disableflags",
    "enableflags", "cone", "jacobian", "noslip_iterations", "noslip_tolerance",
    "impratio", "gravity", "density", "viscosity", "wind", "magnetic",
)


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return int(value)


def json_safe(value):
    """Keep nonfinite diagnostics explicit without emitting invalid JSON numbers."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return str(float(value))
    return json_value(value)


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(json_safe(document), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(packages):
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "script_sha256": sha256(__file__),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


class Scalars:
    """TensorBoard scalar events without introducing a torch dependency."""

    def __init__(self, env_id, run_name, seed, runs_dir):
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        self.path = Path(runs_dir) / f"{env_id}__{run_name}__{seed}__{time.time_ns() / 1e9:.6f}"
        self.writer = EventFileWriter(str(self.path))

    def add(self, name, value, step):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary

        self.writer.add_event(Event(
            wall_time=time.time(), step=int(step),
            summary=Summary(value=[Summary.Value(tag=name, simple_value=float(value))]),
        ))

    def close(self):
        self.writer.flush()
        self.writer.close()


@contextmanager
def scalar_run(env_id, run_name, seed, runs_dir):
    writer = Scalars(env_id, run_name, seed, runs_dir)
    try:
        yield writer
    finally:
        writer.close()


def task_outputs(metadata, before_qpos, after_qpos, after_qvel, actions, ages):
    """Vectorized exact v4 reward/observation/boundary definitions from export."""
    spec = metadata["task"]
    env_id = metadata["env_id"]
    healthy = np.ones(after_qpos.shape[0], dtype=bool)
    if env_id != "HalfCheetah-v4":
        min_z, max_z = map(float, spec["healthy_z_range"])
        min_angle, max_angle = map(float, spec["healthy_angle_range"])
        healthy &= (min_z < after_qpos[:, 1]) & (after_qpos[:, 1] < max_z)
        healthy &= (min_angle < after_qpos[:, 2]) & (after_qpos[:, 2] < max_angle)
        if env_id == "Hopper-v4":
            state = np.concatenate((after_qpos, after_qvel), axis=-1)[:, 2:]
            low, high = map(float, spec["healthy_state_range"])
            healthy &= np.all((low < state) & (state < high), axis=-1)
    terminated = ~healthy if spec["terminate_when_unhealthy"] else np.zeros_like(healthy)
    healthy_reward = (
        (healthy | spec["terminate_when_unhealthy"]).astype(np.float64) * spec["healthy_reward"]
    )
    velocity = (after_qpos[:, 0] - before_qpos[:, 0]) / metadata["dt"]
    # Gym reduces each float32 action row, then multiplies its scalar by a Python
    # float. Promote AFTER the reduction to retain precisely that arithmetic.
    control_cost = np.square(actions).sum(axis=-1).astype(np.float64) * spec["ctrl_cost_weight"]
    reward = spec["forward_reward_weight"] * velocity + healthy_reward - control_cost
    positions = after_qpos[:, 1:] if spec["exclude_current_positions"] else after_qpos
    velocities = after_qvel if env_id == "HalfCheetah-v4" else np.clip(after_qvel, -10, 10)
    observations = np.concatenate((positions, velocities), axis=-1)
    truncations = ages + 1 >= metadata["horizon"]
    return observations, reward, terminated, truncations


def task_metadata(core, env_id):
    return {
        "forward_reward_weight": float(core._forward_reward_weight),
        "ctrl_cost_weight": float(core._ctrl_cost_weight),
        "healthy_reward": float(getattr(core, "_healthy_reward", 0.0)),
        "terminate_when_unhealthy": bool(getattr(core, "_terminate_when_unhealthy", False)),
        "healthy_z_range": list(getattr(core, "_healthy_z_range", (0.0, 0.0))),
        "healthy_angle_range": list(getattr(core, "_healthy_angle_range", (0.0, 0.0))),
        "healthy_state_range": list(getattr(core, "_healthy_state_range", (0.0, 0.0))),
        "exclude_current_positions": bool(core._exclude_current_positions_from_observation),
    }


def export_one(args, env_id, num_envs):
    import gymnasium as gym
    import mujoco
    from cleanrl.shared.vector_norm import make_raw_continuous_env

    destination = Path(args.dataset_dir) / f"{env_id}__n{num_envs}"
    destination.mkdir(parents=True, exist_ok=False)
    envs = []
    try:
        envs = [make_raw_continuous_env(env_id, index, False, args.run_name)() for index in range(num_envs)]
        cores = [env.unwrapped for env in envs]
        core, model = cores[0], cores[0].model
        # All three v4 tasks have no extra stateful actuators, mocap, userdata,
        # equality switches, or plugins. Refuse silently incomplete fixtures.
        for field in ("na", "nmocap", "nuserdata", "neq", "nplugin"):
            if getattr(model, field, 0):
                raise ValueError(f"fixture exporter must be extended for nonzero model.{field}")
        source_xml = Path(core.fullpath)
        xml = source_xml.read_text()
        root = ET.fromstring(xml)
        if any(element.tag == "include" or "file" in element.attrib for element in root.iter()):
            raise ValueError("fixture exporter needs explicit asset copying for external XML assets")
        (destination / "model.xml").write_text(xml)
        metadata = {
            "format_version": 1, "env_id": env_id, "num_envs": num_envs,
            "steps": args.steps, "seed": args.seed, "frame_skip": int(core.frame_skip),
            "dt": float(core.dt), "horizon": int(envs[0].spec.max_episode_steps),
            "source_xml": str(source_xml), "model_xml_sha256": sha256(destination / "model.xml"),
            "core_source_sha256": sha256(sys.modules[type(core).__module__].__file__),
            "task": task_metadata(core, env_id),
            "model_options": {field: json_value(getattr(model.opt, field)) for field in OPTION_FIELDS},
            "provenance": provenance(("mujoco", "gymnasium", "numpy")),
            "action_distribution": "seeded standard normal, float32, clipped to [-1, 1]",
            "state_scope": "time, qpos, qvel, qacc_warmstart; all other integration inputs verified zero",
        }
        # JSON has no infinity literal; unbounded healthy ranges are encoded as
        # strings and reconstructed into float values when loading the audit.
        for field in ("healthy_z_range", "healthy_state_range", "healthy_angle_range"):
            metadata["task"][field] = [float(value) if np.isfinite(value) else str(value)
                                      for value in metadata["task"][field]]
        numeric_metadata = load_task_numbers(metadata)
        arrays = {f"state_{field}": np.empty((args.steps, num_envs, getattr(core.data, field).size))
                  for field in STATE_FIELDS}
        arrays.update({
            "state_time": np.empty((args.steps, num_envs)),
            "next_qpos": np.empty((args.steps, num_envs, model.nq)),
            "next_qvel": np.empty((args.steps, num_envs, model.nv)),
            "next_qacc_warmstart": np.empty((args.steps, num_envs, model.nv)),
            "observations": np.empty((args.steps, num_envs) + core.observation_space.shape),
            "rewards": np.empty((args.steps, num_envs)),
            "terminations": np.empty((args.steps, num_envs), dtype=bool),
            "truncations": np.empty((args.steps, num_envs), dtype=bool),
            "ages": np.empty((args.steps, num_envs), dtype=np.int32),
        })
        rng = np.random.default_rng(args.seed)
        actions = np.clip(rng.standard_normal((args.steps, num_envs, model.nu)).astype(np.float32), -1, 1)
        arrays["actions"] = actions
        for field in MODEL_FIELDS:
            arrays[f"model_{field}"] = np.array(getattr(model, field), copy=True)
        for index, env in enumerate(envs):
            env.reset(seed=args.seed + index)
        ages = np.zeros(num_envs, dtype=np.int32)
        step_seconds = 0.0
        with scalar_run(env_id, f"{args.run_name}_export_n{num_envs}", args.seed, args.runs_dir) as log:
            for step in range(args.steps):
                arrays["ages"][step] = ages
                for index, (env, core) in enumerate(zip(envs, cores)):
                    data = core.data
                    if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
                        raise ValueError("fixture exporter must include nonzero external forces")
                    arrays["state_time"][step, index] = data.time
                    for field in STATE_FIELDS:
                        arrays[f"state_{field}"][step, index] = getattr(data, field)
                    start = time.perf_counter()
                    obs, reward, terminated, truncated, info = env.step(actions[step, index])
                    step_seconds += time.perf_counter() - start
                    for field in STATE_FIELDS:
                        arrays[f"next_{field}"][step, index] = getattr(data, field)
                    arrays["observations"][step, index] = obs
                    arrays["rewards"][step, index] = reward
                    arrays["terminations"][step, index] = terminated
                    arrays["truncations"][step, index] = truncated
                    ages[index] += 1
                    if terminated or truncated:
                        env.reset()
                        ages[index] = 0
                reconstructed = task_outputs(
                    numeric_metadata, arrays["state_qpos"][step], arrays["next_qpos"][step],
                    arrays["next_qvel"][step], actions[step], arrays["ages"][step],
                )
                for actual, field in zip(reconstructed, ("observations", "rewards", "terminations", "truncations")):
                    np.testing.assert_array_equal(actual, arrays[field][step], err_msg=f"v4 {field} definition differs")
                if (step + 1) % 100 == 0 or step == args.steps - 1:
                    log.add("charts/SPS", (step + 1) * num_envs / step_seconds, (step + 1) * num_envs)
                    log.add("audit/export_steps", step + 1, (step + 1) * num_envs)
            metadata["tensorboard_dir"] = str(log.path)
        metadata["original_wrapped_step_seconds"] = step_seconds
        metadata["original_wrapped_step_sps"] = args.steps * num_envs / step_seconds
        metadata["termination_count"] = int(arrays["terminations"].sum())
        metadata["truncation_count"] = int(arrays["truncations"].sum())
        np.savez(destination / "trajectory.npz", **arrays)
        metadata["trajectory_sha256"] = sha256(destination / "trajectory.npz")
        write_json(destination / "metadata.json", metadata)
        print(json.dumps({"exported": str(destination), "transitions": args.steps * num_envs,
                          "sps": metadata["original_wrapped_step_sps"]}), flush=True)
        return metadata
    finally:
        for env in envs:
            env.close()


def load_task_numbers(metadata):
    metadata = dict(metadata)
    metadata["task"] = dict(metadata["task"])
    for field in ("healthy_z_range", "healthy_state_range", "healthy_angle_range"):
        metadata["task"][field] = [float(value) for value in metadata["task"][field]]
    return metadata


def warp_scalar_option_values(option, num_worlds):
    """Read MJWarp's broadcast/per-world scalar options for setup diagnostics.

    MJWarp 3.12 stores tolerance and timestep in Warp arrays even for a shared
    model. An explicit host read belongs outside all measured stepping paths.
    """
    values = np.asarray(option.numpy(), dtype=np.float64)
    if values.ndim != 1 or values.size not in (1, num_worlds):
        raise ValueError(f"expected one broadcast option or {num_worlds} world options, got {values.shape}")
    return np.broadcast_to(values, (num_worlds,)).copy()


class Comparison:
    def __init__(self):
        self.fields = {}
        self.termination_disagreements = 0
        self.truncation_disagreements = 0
        self.transitions = 0

    def add(self, name, actual, expected):
        difference = np.abs(np.asarray(actual, dtype=np.float64) - expected)
        item = self.fields.setdefault(name, {"max_abs_error": 0.0, "squared_error_sum": 0.0, "count": 0,
                                             "nonfinite": 0})
        finite = np.isfinite(difference)
        item["nonfinite"] += int((~finite).sum())
        item["max_abs_error"] = max(item["max_abs_error"], float(np.max(difference[finite], initial=0)))
        item["squared_error_sum"] += float(np.square(difference[finite]).sum())
        item["count"] += difference.size

    def add_step(self, metadata, arrays, step, qpos, qvel, warmstart):
        self.add("qpos", qpos, arrays["next_qpos"][step])
        self.add("qvel", qvel, arrays["next_qvel"][step])
        self.add("qacc_warmstart", warmstart, arrays["next_qacc_warmstart"][step])
        observation, reward, terms, truncs = task_outputs(
            metadata, arrays["state_qpos"][step], qpos, qvel, arrays["actions"][step], arrays["ages"][step],
        )
        self.add("observations", observation, arrays["observations"][step])
        self.add("rewards", reward, arrays["rewards"][step])
        self.termination_disagreements += int(np.count_nonzero(terms != arrays["terminations"][step]))
        self.truncation_disagreements += int(np.count_nonzero(truncs != arrays["truncations"][step]))
        self.transitions += qpos.shape[0]

    def report(self, atol):
        fields = {name: {
            "max_abs_error": value["max_abs_error"] if not value["nonfinite"] else float("inf"),
            "rmse": (value["squared_error_sum"] / max(1, value["count"])) ** 0.5
                    if not value["nonfinite"] else float("inf"),
            "nonfinite": value["nonfinite"],
        } for name, value in self.fields.items()}
        passed = bool(fields) and all(item["nonfinite"] == 0 and item["max_abs_error"] <= atol
                                      for item in fields.values())
        passed &= self.termination_disagreements == self.truncation_disagreements == 0
        return {"fields": fields, "termination_disagreements": self.termination_disagreements,
                "truncation_disagreements": self.truncation_disagreements,
                "transitions": self.transitions, "strict_numeric_parity": bool(passed), "atol": atol}


def install_fixture(mujoco, model, data, arrays, step, world):
    mujoco.mj_resetData(model, data)
    data.time = arrays["state_time"][step, world]
    for field in STATE_FIELDS:
        getattr(data, field)[:] = arrays[f"state_{field}"][step, world]
    data.ctrl[:] = arrays["actions"][step, world]


def audit_native(mujoco, model, metadata, arrays, args):
    data = [mujoco.MjData(model) for _ in range(metadata["num_envs"])]
    comparison = Comparison()
    qpos = np.empty_like(arrays["next_qpos"][0])
    qvel = np.empty_like(arrays["next_qvel"][0])
    warmstart = np.empty_like(qvel)
    physics_seconds = 0.0
    start = time.perf_counter()
    for step in range(metadata["steps"]):
        for world, instance in enumerate(data):
            install_fixture(mujoco, model, instance, arrays, step, world)
            begin = time.perf_counter()
            mujoco.mj_step(model, instance, nstep=metadata["frame_skip"])
            mujoco.mj_rnePostConstraint(model, instance)
            physics_seconds += time.perf_counter() - begin
            qpos[world] = instance.qpos
            qvel[world] = instance.qvel
            warmstart[world] = instance.qacc_warmstart
        comparison.add_step(metadata, arrays, step, qpos, qvel, warmstart)
    report = comparison.report(args.atol)
    report.update({"physics_seconds": physics_seconds,
                   "physics_sps": metadata["steps"] * metadata["num_envs"] / physics_seconds,
                   "audit_wall_seconds": time.perf_counter() - start,
                   "timing_scope": "sequential native mj_step(frame_skip) + mj_rnePostConstraint; state restore excluded"})
    return report


def audit_warp(mujoco, model, metadata, arrays, args, log):
    global wp
    import warp as wp
    import mujoco_warp as mjw

    @wp.kernel
    def load_fixture(
        fixtures: wp.array3d(dtype=wp.float32), index: wp.array(dtype=wp.int32),
        nq: int, nv: int, nu: int, time_out: wp.array(dtype=wp.float32),
        qpos: wp.array2d(dtype=wp.float32), qvel: wp.array2d(dtype=wp.float32),
        warmstart: wp.array2d(dtype=wp.float32), ctrl: wp.array2d(dtype=wp.float32),
    ):
        world, coordinate = wp.tid()
        step = index[0]
        if coordinate == 0:
            time_out[world] = fixtures[step, world, 0]
        if coordinate < nq:
            qpos[world, coordinate] = fixtures[step, world, 1 + coordinate]
        if coordinate < nv:
            qvel[world, coordinate] = fixtures[step, world, 1 + nq + coordinate]
            warmstart[world, coordinate] = fixtures[step, world, 1 + nq + nv + coordinate]
        if coordinate < nu:
            ctrl[world, coordinate] = fixtures[step, world, 1 + nq + 2 * nv + coordinate]

    @wp.kernel
    def advance_fixture(index: wp.array(dtype=wp.int32), steps: int):
        index[0] = (index[0] + 1) % steps

    wp.config.quiet = True
    wp.init()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise ValueError("GPU audit requires a CUDA device")
    nworld, steps = metadata["num_envs"], metadata["steps"]
    packed = np.concatenate((arrays["state_time"][..., None], arrays["state_qpos"],
                             arrays["state_qvel"], arrays["state_qacc_warmstart"], arrays["actions"]), axis=-1)
    started = time.perf_counter()
    with wp.ScopedDevice(device):
        gpu_model = mjw.put_model(model)
        gpu_model.opt.warn_overflow = False  # The audit checks every overflow bit explicitly.
        effective_tolerances = warp_scalar_option_values(gpu_model.opt.tolerance, nworld)
        initial = mujoco.MjData(model)
        install_fixture(mujoco, model, initial, arrays, 0, 0)
        mujoco.mj_forward(model, initial)
        data = mjw.put_data(model, initial, nworld=nworld, nconmax=args.nconmax, njmax=args.njmax)
        fixtures = wp.array(packed.astype(np.float32), dtype=wp.float32, device=device)
        index = wp.zeros(1, dtype=wp.int32, device=device)

        def one_step():
            wp.launch(load_fixture, dim=(nworld, max(model.nq, model.nv, model.nu)), inputs=[
                fixtures, index, model.nq, model.nv, model.nu, data.time, data.qpos, data.qvel,
                data.qacc_warmstart, data.ctrl,
            ])
            for _ in range(metadata["frame_skip"]):
                mjw.step(gpu_model, data)
            mjw.rne_postconstraint(gpu_model, data)
            wp.launch(advance_fixture, dim=1, inputs=[index, steps])

        # Compile once before capture. These are setup calls on the full fixture,
        # not reduced training or an empirical learning-validation run.
        one_step()
        wp.synchronize_device(device)
        index.zero_()
        with wp.ScopedCapture(device=device) as capture:
            one_step()
        wp.synchronize_device(device)
        setup_seconds = time.perf_counter() - started
        comparison = Comparison()
        parity_start = time.perf_counter()
        for step in range(steps):
            wp.capture_launch(capture.graph)
            overflows = data.overflow.numpy()
            if np.any(overflows):
                raise RuntimeError(f"GPU contact/constraint overflow at fixture step {step}: {overflows.tolist()}")
            # Cast to float64 BEFORE task reward arithmetic: isolate physics
            # differences from a separate change in Gym reward evaluation dtype.
            comparison.add_step(metadata, arrays, step, data.qpos.numpy().astype(np.float64),
                                data.qvel.numpy().astype(np.float64), data.qacc_warmstart.numpy().astype(np.float64))
            if (step + 1) % 100 == 0 or step == steps - 1:
                log.add("audit/gpu_parity_steps", step + 1, (step + 1) * nworld)
                log.add("parity/gpu_max_qpos_error", comparison.fields["qpos"]["max_abs_error"],
                        (step + 1) * nworld)
        parity_seconds = time.perf_counter() - parity_start
        report = comparison.report(args.atol)
        timings = []
        for _ in range(args.repeats):
            index.zero_()
            data.overflow.zero_()
            wp.synchronize_device(device)
            begin_event = wp.Event(device=device, enable_timing=True)
            end_event = wp.Event(device=device, enable_timing=True)
            start = time.perf_counter()
            wp.record_event(begin_event)
            for _ in range(steps):
                wp.capture_launch(capture.graph)
            wp.record_event(end_event)
            wp.synchronize_device(device)
            wall_seconds = time.perf_counter() - start
            cuda_seconds = wp.get_event_elapsed_time(begin_event, end_event) / 1000.0
            overflow = data.overflow.numpy()
            if np.any(overflow):
                raise RuntimeError(f"GPU throughput replay overflow: {overflow.tolist()}")
            timings.append({"wall_seconds": wall_seconds, "cuda_seconds": cuda_seconds,
                            "wall_sps": steps * nworld / wall_seconds, "cuda_sps": steps * nworld / cuda_seconds})
        report.update({
            "device": str(device), "device_name": device.name,
            "physics_dtype": str(data.qpos.dtype),
            "effective_tolerance": (float(effective_tolerances[0])
                                    if np.all(effective_tolerances == effective_tolerances[0]) else None),
            "effective_tolerance_per_world": effective_tolerances.tolist(),
            "setup_seconds": setup_seconds, "parity_audit_wall_seconds": parity_seconds,
            "nconmax": args.nconmax, "njmax": args.njmax, "overflow_count": 0,
            "throughput_repeats": timings,
            "median_resident_replay_sps": float(np.median([item["wall_sps"] for item in timings])),
            "timing_scope": "CUDA graph: resident fixture state/action load, frame_skip physics, rnePostConstraint, fixture-index advance; no policy, H2D/D2H, reward normalization, or reset sampling",
        })
        return report


def audit_one(args, destination, report, flush):
    import mujoco

    source_metadata = json.loads((destination / "metadata.json").read_text())
    metadata = load_task_numbers(source_metadata)
    if sha256(destination / "model.xml") != metadata["model_xml_sha256"]:
        raise ValueError("exported model XML checksum mismatch")
    if sha256(destination / "trajectory.npz") != metadata["trajectory_sha256"]:
        raise ValueError("exported trajectory checksum mismatch")
    model = mujoco.MjModel.from_xml_path(str(destination / "model.xml"))
    with np.load(destination / "trajectory.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    modern_options = {field: json_value(getattr(model.opt, field)) for field in OPTION_FIELDS}
    option_changes = {field: {"original": metadata["model_options"][field], "modern": value}
                      for field, value in modern_options.items() if value != metadata["model_options"][field]}
    model_differences = {}
    for field in MODEL_FIELDS:
        current, source = np.asarray(getattr(model, field)), arrays[f"model_{field}"]
        model_differences[field] = (float(np.max(np.abs(current - source), initial=0))
                                    if current.shape == source.shape else "shape changed")
    report.update({"env_id": metadata["env_id"], "num_envs": metadata["num_envs"], "steps": metadata["steps"],
                   "source": source_metadata, "modern_model_option_changes": option_changes,
                   "modern_model_array_max_errors": model_differences})
    with scalar_run(metadata["env_id"], f"{args.run_name}_n{metadata['num_envs']}",
                    metadata["seed"], args.runs_dir) as log:
        log.add("audit/started", 1, 0)
        report["tensorboard_dir"] = str(log.path)
        flush()
        report["modern_native"] = audit_native(mujoco, model, metadata, arrays, args)
        # Native engine drift remains useful evidence if GPU compilation or
        # feature support fails later. Persist it before initializing MJWarp.
        flush()
        report["warp"] = audit_warp(mujoco, model, metadata, arrays, args, log)
        reasons = []
        if metadata["provenance"]["packages"]["mujoco"] != mujoco.__version__:
            reasons.append("GPU backend requires a different MuJoCo engine version from the v4 reference")
        if not report["warp"]["strict_numeric_parity"]:
            reasons.append("GPU one-step state/observation/reward/termination parity failed the requested tolerance")
        if any(value != metadata["model_options"]["tolerance"]
               for value in report["warp"]["effective_tolerance_per_world"]):
            reasons.append("MJWarp automatically changed the constraint-solver tolerance")
        reasons.append("MJWarp integrates in float32 while the original native simulator integrates in float64")
        report["eligible_as_no_compromise_replacement"] = len(reasons) == 0
        report["ineligibility_reasons"] = reasons
        report["docs"] = DOCS_URL
        report["limits"] = [
            "Stochastic reference actions provide physics coverage, not evidence about a trained policy's scores.",
            "Same-state replay isolates one-step errors; long-horizon free-running dynamics can diverge more.",
            "Resident replay throughput excludes policy inference/update, observation/reward computation, normalization, and reset RNG.",
            "Native physics timing is sequential; compare the shared native parallel benchmark separately.",
        ]
        total = metadata["steps"] * metadata["num_envs"]
        log.add("charts/SPS", report["warp"]["median_resident_replay_sps"], total)
        log.add("audit/eligible_as_no_compromise_replacement", report["eligible_as_no_compromise_replacement"], total)
        log.add("throughput/original_wrapped_step_sps", metadata["original_wrapped_step_sps"], total)
        log.add("throughput/modern_native_physics_sps", report["modern_native"]["physics_sps"], total)
        log.add("throughput/warp_resident_replay_sps", report["warp"]["median_resident_replay_sps"], total)
        for engine in ("modern_native", "warp"):
            log.add(f"parity/{engine}_termination_disagreements", report[engine]["termination_disagreements"], total)
            for field, values in report[engine]["fields"].items():
                log.add(f"parity/{engine}_{field}_max_error", values["max_abs_error"], total)
        log.add("audit/completed", 1, total)
        report["status"] = "complete"
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in ("export", "audit"):
        sub = subparsers.add_parser(stage)
        sub.add_argument("--dataset-dir", type=Path, required=True)
        sub.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
        sub.add_argument("--run-name", default="mujoco_warp_audit_v1")
        if stage == "export":
            sub.add_argument("--env-ids", nargs="+", choices=ENV_IDS, default=list(ENV_IDS))
            sub.add_argument("--num-envs", nargs="+", type=int, default=[16, 64])
            sub.add_argument("--steps", type=int, default=1000)
            sub.add_argument("--seed", type=int, default=1)
        else:
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--device", default="cuda:0")
            sub.add_argument("--atol", type=float, default=1e-10)
            sub.add_argument("--nconmax", type=int, default=64)
            sub.add_argument("--njmax", type=int, default=256)
            sub.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.stage == "export" and (args.steps < 1000 or min(args.num_envs) < 1):
        parser.error("export requires at least one full 1000-step horizon and positive environment counts")
    if args.stage == "audit" and (args.atol < 0 or args.repeats < 1 or args.nconmax < 1 or args.njmax < 1):
        parser.error("audit requires nonnegative atol and positive repeats/contact/constraint capacity")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "export":
        for env_id in args.env_ids:
            for num_envs in args.num_envs:
                export_one(args, env_id, num_envs)
        return 0
    datasets = sorted(path.parent for path in args.dataset_dir.glob("*/metadata.json"))
    if not datasets:
        raise ValueError("no complete exported datasets were found")
    result = {"provenance": provenance(("mujoco", "mujoco-warp", "warp-lang", "numpy")),
              "datasets": [], "status": "running", "docs": DOCS_URL}
    write_json(args.output, result)
    failed = False
    for destination in datasets:
        print(json.dumps({"auditing": str(destination)}), flush=True)
        report = {"dataset": str(destination), "status": "running"}
        result["datasets"].append(report)
        try:
            audit_one(args, destination, report, lambda: write_json(args.output, result))
        except Exception as error:
            traceback.print_exc()
            report.update({"error": str(error), "status": "error",
                           "eligible_as_no_compromise_replacement": False})
            failed = True
        write_json(args.output, result)
    result["status"] = "incomplete" if failed else "complete"
    result["eligible_as_no_compromise_replacement"] = (
        not failed and all(report["eligible_as_no_compromise_replacement"] for report in result["datasets"])
    )
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "eligible_as_no_compromise_replacement": result["eligible_as_no_compromise_replacement"]}), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
