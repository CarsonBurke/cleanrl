"""Checkpoint plumbing on raw tensors; no network, environment or GPU execution."""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cleanrl.td7.delight import td7_latentq_value_floor_v5 as trainer


@pytest.fixture(autouse=True, params=["td7_latentq_value_floor_v5", "td7_latentq_delight_hermite_v6", "td7_latentq_delight_hermite_ablation_v7", "td7_latentq_delight_hermite_integrated_v8", "td7_latentq_delight_reference_v10"])
def checkpoint_implementation(request, monkeypatch):
    module = importlib.import_module(f"cleanrl.td7.delight.{request.param}")
    monkeypatch.setitem(globals(), "trainer", module)


class TensorState:
    def __init__(self, value):
        self.value = torch.tensor([float(value)])

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state, *, strict, assign):
        assert strict and not assign
        self.value.copy_(state["value"])


class TensorOptimizer:
    def __init__(self, parameter):
        self.param_groups = [{"params": [parameter], "lr": 0.001, "betas": (0.9, 0.999)}]
        self.state = {parameter: {"step": torch.tensor(7.0),
                                 "exp_avg": torch.ones_like(parameter),
                                 "exp_avg_sq": torch.full_like(parameter, 2.0)}}

    def state_dict(self):
        return {"param_groups": [dict(self.param_groups[0], params=[0])],
                "state": {0: self.state[self.param_groups[0]["params"][0]]}}


def tensor_agent():
    # Bypass TD7Agent construction entirely: these are storage fixtures, not networks.
    agent = object.__new__(trainer.TD7Agent)
    agent.args = trainer.Args(buffer_size=8, num_envs=1, num_steps=2)
    agent.row_dim = 5
    for index, name in enumerate(trainer.CHECKPOINT_MODELS):
        setattr(agent, name, TensorState(index + 1))
    agent.log_alpha = torch.tensor(0.7)
    parameters = [agent.encoder.value, agent.critic.value, agent.actor.value, agent.log_alpha]
    for name, parameter in zip(trainer.CHECKPOINT_OPTIMIZERS, parameters):
        setattr(agent, name, TensorOptimizer(parameter))
    agent.size, agent.ptr, agent.training_steps = 3, 3, 7
    agent.current_global_step = 120
    agent.storage = torch.arange(40, dtype=torch.float32).reshape(8, 5)
    agent.priority = torch.arange(8, dtype=torch.float32)
    agent.last = torch.tensor([2])
    for name in trainer.CHECKPOINT_BUFFERS:
        if name == "last":
            continue
        setattr(agent, name, torch.ones(2, 5) if name == "fresh" else torch.tensor(1.0))
    agent.policy = SimpleNamespace(refresh=lambda: None)
    return agent


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def test_learner_checkpoint_only_saves_filled_replay_and_restores_addresses():
    agent = tensor_agent()
    saved = agent.checkpoint_state()
    assert saved["storage"].shape == (3, 5)
    assert saved["priority"].shape == (3,)
    tracked = [getattr(agent, name).value for name in trainer.CHECKPOINT_MODELS]
    tracked += [agent.storage, agent.priority, agent.log_alpha]
    tracked += [getattr(agent, name) for name in trainer.CHECKPOINT_BUFFERS]
    for name in trainer.CHECKPOINT_OPTIMIZERS:
        for state in getattr(agent, name).state.values():
            tracked.extend(state.values())
    addresses = [tensor.data_ptr() for tensor in tracked]
    for tensor in tracked:
        tensor.fill_(99)
    agent.ptr, agent.size, agent.training_steps = 0, 8, 999
    agent.restore_checkpoint_state(saved)
    assert [tensor.data_ptr() for tensor in tracked] == addresses
    assert_state_equal(agent.checkpoint_state(), saved)
    assert torch.count_nonzero(agent.storage[3:]) == 0
    assert torch.count_nonzero(agent.priority[3:]) == 0


@pytest.mark.parametrize("mutation", ["setting", "shape", "missing"])
def test_optimizer_restore_rejects_incompatible_state(mutation):
    optimizer = TensorOptimizer(torch.zeros(3))
    saved = trainer.checkpoint_to_cpu(optimizer.state_dict())
    if mutation == "setting":
        saved["param_groups"][0]["lr"] = 0.2
    elif mutation == "shape":
        saved["state"][0]["exp_avg"] = torch.zeros(4)
    else:
        del saved["state"][0]["step"]
    with pytest.raises(ValueError, match="checkpoint optimizer"):
        trainer.restore_optimizer_in_place(optimizer, saved)


def test_algorithm_configuration_is_strict_but_resume_budget_and_paths_may_change():
    args = trainer.Args()
    checkpoint = {"format": trainer.__name__.rsplit(".", 1)[-1], "version": 1, "args": vars(args).copy()}
    trainer.validate_checkpoint_args(checkpoint, trainer.Args(
        total_timesteps=12_000_000, exp_name="continued", resume="prior.pt"))
    with pytest.raises(ValueError, match="actor_baseline"):
        trainer.validate_checkpoint_args(checkpoint, trainer.Args(actor_baseline="cubature"))
    with pytest.raises(ValueError, match="target_sigma"):
        trainer.validate_checkpoint_args(checkpoint, trainer.Args(target_sigma=0.2))


def test_rng_snapshot_does_not_draw_and_restore_recovers_all_host_streams(monkeypatch):
    # CUDA calls are replaced with raw state fixtures, never executed on hardware.
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda state: None)
    rng = np.random.default_rng(71)
    state = trainer.capture_rng_state(rng)
    assert_state_equal(state["torch"], torch.get_rng_state())
    assert state["host"] == rng.bit_generator.state
    expected = (trainer.random.random(), np.random.random(), rng.random(), torch.rand(4))
    trainer.restore_rng_state(state, rng)
    actual = (trainer.random.random(), np.random.random(), rng.random(), torch.rand(4))
    assert expected[:3] == actual[:3]
    torch.testing.assert_close(expected[3], actual[3], rtol=0, atol=0)


def test_atomic_serialization_roundtrip_preserves_state_and_failed_save_is_visible(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    agent = tensor_agent()
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(path, agent, np.random.default_rng(71), 120, 5)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 120
    assert checkpoint["completed_rollouts"] == 5
    assert_state_equal(checkpoint["agent"], agent.checkpoint_state())
    previous_bytes = path.read_bytes()

    def fail(*args):
        raise OSError("checkpoint disk unavailable")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="checkpoint disk unavailable"):
        trainer.save_checkpoint(path, agent, np.random.default_rng(71), 120, 5)
    assert path.read_bytes() == previous_bytes
    assert not list(tmp_path.glob("*.tmp"))
