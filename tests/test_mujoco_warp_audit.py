"""Pure CPU metadata/arithmetic checks; no environment or GPU construction."""

import json

import numpy as np
import pytest

from scripts.audit_mujoco_warp import (
    Comparison, json_safe, load_task_numbers, task_outputs, warp_scalar_option_values, write_json,
)


def metadata(env_id):
    return {
        "env_id": env_id, "dt": 0.05 if env_id == "HalfCheetah-v4" else 0.008,
        "horizon": 1000,
        "task": {
            "healthy_z_range": [0.7, "inf"] if env_id == "Hopper-v4" else [0.8, 2.0],
            "healthy_angle_range": [-0.2, 0.2] if env_id == "Hopper-v4" else [-1.0, 1.0],
            "healthy_state_range": [-100.0, 100.0], "terminate_when_unhealthy": env_id != "HalfCheetah-v4",
            "forward_reward_weight": 1.0, "ctrl_cost_weight": 0.1 if env_id == "HalfCheetah-v4" else 0.001,
            "healthy_reward": 0.0 if env_id == "HalfCheetah-v4" else 1.0,
            "exclude_current_positions": True,
        },
    }


@pytest.mark.parametrize("env_id", ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"])
def test_vectorized_reward_and_boundaries_match_v4_scalar_arithmetic(env_id):
    spec = metadata(env_id)
    rng = np.random.default_rng(1)
    before = rng.normal(size=(5, 6))
    after = before + rng.normal(scale=0.01, size=(5, 6))
    after[:, 1:3] = [[1.0, 0.0], [0.7, 0.0], [1.0, 0.2], [2.0, -1.0], [1.0, 0.0]]
    qvel = rng.normal(size=(5, 6))
    qvel[4, -1] = 100.0
    actions = rng.normal(size=(5, 3)).astype(np.float32)
    ages = np.array([0, 999, 998, 999, 100])
    observations, rewards, terms, truncs = task_outputs(spec, before, after, qvel, actions, ages)
    task = spec["task"]
    for row in range(5):
        if env_id == "HalfCheetah-v4":
            healthy = True
            healthy_reward = 0.0
        else:
            low_z, high_z = map(float, task["healthy_z_range"])
            low_angle, high_angle = task["healthy_angle_range"]
            healthy = low_z < after[row, 1] < high_z and low_angle < after[row, 2] < high_angle
            if env_id == "Hopper-v4":
                state = np.concatenate((after[row], qvel[row]))[2:]
                healthy = healthy and np.all((-100 < state) & (state < 100))
            healthy_reward = float(healthy or task["terminate_when_unhealthy"]) * task["healthy_reward"]
        forward_reward = task["forward_reward_weight"] * ((after[row, 0] - before[row, 0]) / spec["dt"])
        control_cost = task["ctrl_cost_weight"] * np.sum(np.square(actions[row]))
        assert rewards[row] == forward_reward + healthy_reward - control_cost
        assert terms[row] == (not healthy if task["terminate_when_unhealthy"] else False)
        assert truncs[row] == (ages[row] + 1 >= 1000)
        velocity = qvel[row] if env_id == "HalfCheetah-v4" else np.clip(qvel[row], -10, 10)
        np.testing.assert_array_equal(observations[row], np.concatenate((after[row, 1:], velocity)))


def test_metadata_and_failed_numerics_serialize_as_valid_json(tmp_path):
    # Include NumPy arrays/scalars and nested options: serialization must never
    # discard audit evidence just because a bad simulator emitted NaN/Inf.
    document = {"task": {"healthy_z_range": [0.7, np.inf]},
                "options": np.array([np.nan, -np.inf]), "count": np.int32(4)}
    destination = tmp_path / "audit.json"
    write_json(destination, document)
    def reject_nonstandard_constant(value):
        raise AssertionError(value)
    got = json.loads(destination.read_text(), parse_constant=reject_nonstandard_constant)
    assert got == {"task": {"healthy_z_range": [0.7, "inf"]}, "options": ["nan", "-inf"], "count": 4}
    assert not destination.with_suffix(".json.pending").exists()
    hopper = load_task_numbers(metadata("Hopper-v4"))
    assert hopper["task"]["healthy_z_range"][-1] == np.inf


def test_nonfinite_comparison_is_ineligible_and_reports_explicit_failure():
    comparison = Comparison()
    comparison.add("qpos", np.array([np.nan, np.inf]), np.zeros(2))
    result = comparison.report(1e-10)
    assert not result["strict_numeric_parity"]
    assert result["fields"]["qpos"]["nonfinite"] == 2
    assert json_safe(result)["fields"]["qpos"]["max_abs_error"] == "inf"


class FakeWarpArray:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def numpy(self):
        return self.values

    def __float__(self):
        raise TypeError("Warp option arrays require an explicit host read")


@pytest.mark.parametrize("values", [[1e-6], [1e-6, 2e-6, 3e-6]])
def test_warp_options_extract_broadcast_and_per_world_values(values):
    option = FakeWarpArray(values)
    result = warp_scalar_option_values(option, 3)
    np.testing.assert_array_equal(result, np.broadcast_to(np.asarray(values, dtype=np.float32), (3,)))
    assert result.dtype == np.float64
    option.values[:] = 0
    assert np.all(result > 0)


@pytest.mark.parametrize("values", [[], [1e-6, 2e-6], [[1e-6]]])
def test_warp_options_reject_unexpected_array_shapes(values):
    with pytest.raises(ValueError, match="broadcast option"):
        warp_scalar_option_values(FakeWarpArray(values), 3)
