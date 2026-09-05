"""Packed rollout staging preserves values, layouts, and input lifetimes."""

import numpy as np
import pytest
import torch

from cleanrl.shared.rollout_transfer import ActionTransfer, RolloutTransfer


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"
)])])
def device(request):
    return request.param


@pytest.mark.parametrize("store_transitions", [False, True])
@pytest.mark.parametrize("non_blocking", [False, True])
def test_rollout_upload_preserves_all_fields_and_reuses_storage(device, store_transitions, non_blocking):
    steps, envs, obs_shape = 4, 3, (2, 2)
    staging = RolloutTransfer(
        steps, envs, obs_shape, device,
        store_transition_observations=store_transitions,
        non_blocking=non_blocking,
    )
    rng = np.random.default_rng(4)
    rewards = rng.normal(size=(steps, envs))
    terms = rng.random((steps, envs)) < 0.3
    truncs = rng.random((steps, envs)) < 0.3
    transitions = rng.normal(size=(steps, envs) + obs_shape)
    for step in range(steps):
        # Reusing mutable environment outputs after push must be safe.
        inputs = [array[step].copy() for array in (rewards, terms, truncs)]
        transition = transitions[step].copy() if store_transitions else None
        staging.push(step, *inputs, transition)
        for array in inputs:
            array[...] = 0
        if transition is not None:
            transition[...] = 0
    got = staging.upload()
    for result, expected in zip(got[:3], (rewards, terms, truncs)):
        assert result.is_contiguous()
        torch.testing.assert_close(result.cpu(), torch.tensor(expected, dtype=torch.float32))
    if store_transitions:
        assert got.transition_observations.is_contiguous()
        torch.testing.assert_close(
            got.transition_observations.cpu(), torch.tensor(transitions, dtype=torch.float32)
        )
    else:
        assert got.transition_observations is None
    assert staging.upload().rewards.data_ptr() == got.rewards.data_ptr()
    assert staging._host.is_pinned() == (device == "cuda")


@pytest.mark.parametrize("non_blocking", [False, True])
def test_policy_observation_snapshot_and_reuse(device, non_blocking):
    staging = RolloutTransfer(2, 2, (3,), device, non_blocking=non_blocking)
    observation = np.arange(6, dtype=np.float64).reshape(2, 3)
    got = staging.observation(observation)
    stored = got.clone()
    observation[...] = 100
    torch.testing.assert_close(stored.cpu(), torch.arange(6, dtype=torch.float32).reshape(2, 3))
    torch.testing.assert_close(got, stored)
    assert staging.observation(observation).data_ptr() == got.data_ptr()
    torch.testing.assert_close(got.cpu(), torch.full((2, 3), 100.0))


def test_rollout_transfer_requires_transition_observations_when_enabled():
    staging = RolloutTransfer(2, 2, (3,), "cpu", store_transition_observations=True)
    with pytest.raises(ValueError, match="required"):
        staging.push(0, np.ones(2), np.zeros(2), np.zeros(2))


def test_rollout_transfer_rejects_unconfigured_transition_observations():
    staging = RolloutTransfer(2, 2, (3,), "cpu")
    with pytest.raises(ValueError, match="enable"):
        staging.push(0, np.ones(2), np.zeros(2), np.zeros(2), np.ones((2, 3)))


def test_action_transfer_owns_storage_and_enforces_pending_state(device):
    actions = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
    transfer = ActionTransfer(actions.shape, device)
    with pytest.raises(RuntimeError, match="no action transfer"):
        transfer.wait()
    transfer.submit(actions)
    with pytest.raises(RuntimeError, match="pending"):
        transfer.submit(actions)
    # Stream ordering must protect the copy from a subsequent source mutation.
    actions.fill_(-1)
    got = transfer.wait()
    np.testing.assert_array_equal(got, np.arange(6, dtype=np.float32).reshape(2, 3))
    transfer.submit(actions)
    assert transfer.wait() is got
    np.testing.assert_array_equal(got, np.full((2, 3), -1, dtype=np.float32))
    transfer.submit(actions)
    transfer.close()
    with pytest.raises(RuntimeError, match="no action transfer"):
        transfer.wait()


def test_action_transfer_rejects_silent_dtype_or_shape_changes():
    transfer = ActionTransfer((2, 3), "cpu")
    with pytest.raises(ValueError, match="dtype"):
        transfer.submit(torch.zeros((2, 3), dtype=torch.float64))
    with pytest.raises(ValueError, match="shape"):
        transfer.submit(torch.zeros((3, 2)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.cuda
@pytest.mark.parametrize("slots", [1, 2, 3])
def test_async_cuda_host_slots_survive_dma_delay_and_immediate_input_reuse(slots):
    staging = RolloutTransfer(2, 256, (1024,), "cuda", non_blocking=True, staging_slots=slots)
    observation = np.zeros((256, 1024), dtype=np.float32)
    snapshots = []
    for index in range(9):
        # Make DMA lag behind the CPU to expose reuse without event protection.
        torch.cuda._sleep(2_000_000)
        observation.fill(index)
        snapshots.append(staging.observation(observation).clone())
        observation.fill(-100)
    staging.close()
    for index, snapshot in enumerate(snapshots):
        torch.testing.assert_close(snapshot.cpu(), torch.full((256, 1024), float(index)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.cuda
def test_async_cuda_rollout_upload_protects_host_on_next_push():
    staging = RolloutTransfer(2, 256, (1024,), "cuda", non_blocking=True)
    ones, zeros = np.ones(256), np.zeros(256)
    for step in range(2):
        staging.push(step, ones, zeros, ones)
    torch.cuda._sleep(2_000_000)
    first = tuple(field.clone() for field in staging.upload()[:3])
    upload_event = staging._upload_event
    for step in range(2):
        staging.push(step, -ones, ones, zeros)
    second = staging.upload()
    assert staging._upload_event is upload_event
    staging.close()
    assert not staging._upload_pending
    for field, expected in zip(first, (1, 0, 1)):
        torch.testing.assert_close(field.cpu(), torch.full((2, 256), float(expected)))
    for field, expected in zip(second[:3], (-1, 1, 0)):
        torch.testing.assert_close(field.cpu(), torch.full((2, 256), float(expected)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.cuda
def test_cuda_action_transfer_keeps_temporary_source_alive_until_copy_finishes():
    transfer = ActionTransfer((256, 1024), "cuda")
    torch.cuda._sleep(2_000_000)
    transfer.submit(torch.full((256, 1024), 7.0, device="cuda"))
    # Exercise allocator reuse while the source tensor has no Python owner.
    replacements = [torch.full((256, 1024), -2.0, device="cuda") for _ in range(4)]
    np.testing.assert_array_equal(transfer.wait(), np.full((256, 1024), 7.0, dtype=np.float32))
    assert len(replacements) == 4
