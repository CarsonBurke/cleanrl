"""Gait/contact and held-out candidate-discovery contracts; queue through mlq."""
import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import torch

from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.runtime import configure_runtime

spec = importlib.util.spec_from_file_location('audit_gait', Path(__file__).parents[1]/'scripts/audit_sepclip_gait.py')
gait = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gait)


def test_contact_cycles_distinguish_stance_and_swing():
    trace = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1], bool)
    result = gait.contact_cycles(trace, .05)
    assert result['onsets'] == [1, 5, 9]
    assert result['offsets'] == [3, 7]
    assert len(result['complete_cycles']) == 2
    for cycle in result['complete_cycles']:
        assert cycle['cycle_seconds'] == .2
        assert cycle['stance_seconds'] == .1
        assert cycle['stance_fraction'] == .5


def test_candidate_selection_uses_discovery_and_respects_feasible_gradient_budget():
    actual = np.array([[.5, .5], [.999, .001]], np.float32)
    candidates, labels = gait.candidate_actions(actual, actual, actual, .025)
    slopes = np.array([[2., -1.], [-3., 4.]])
    discovery_q = np.einsum('scd,sd->sc', candidates, slopes)
    selection, directed, estimate = gait.discovery_selection(candidates, [discovery_q, discovery_q], .025)
    np.testing.assert_allclose(estimate, slopes, atol=2e-5)
    np.testing.assert_array_equal(selection, np.argmax(discovery_q, axis=1))
    assert np.all(directed>=1e-6) and np.all(directed<=1-1e-6)
    assert np.all(np.linalg.norm(directed-actual, axis=-1)<=.025*np.sqrt(2)+1e-7)
    assert np.all(np.einsum('sd,sd->s', directed-actual, slopes)>0)
    # Held-out candidates may lose despite winning discovery; their ranking is
    # never used by discovery_selection, preventing an oracle winner's bias.
    heldout_q = -discovery_q
    assert np.any(heldout_q[np.arange(2), selection]-heldout_q[:, 0]<0)


def test_real_floor_contacts_report_named_geometries_and_nonnegative_normal_force():
    configure_runtime(matmul_precision='highest', allow_tf32=False)
    assert torch.cuda.is_available(), 'Queue physical contracts on CUDA'
    envs = make_mujoco_vector_env('HalfCheetah-v4', 1, num_threads=2)
    try:
        envs.reset(seed=1)
        base = envs.envs[0].unwrapped
        base.data.qpos[:] = 0
        base.data.qvel[:] = 0
        base.data.qpos[1] = -.3
        mujoco.mj_forward(base.model, base.data)
        present, force = gait.ground_contacts(base)
        assert present.shape == force.shape == (len(gait.CONTACT_GEOMS),)
        assert present.any()
        assert np.all(force>=0)
        assert np.all(force[~present]==0)
        assert present[gait.CONTACT_GEOMS.index('bfoot')]
        assert present[gait.CONTACT_GEOMS.index('ffoot')]
    finally:
        envs.close()


def test_phase_profiles_recover_known_action_peak_timing():
    n = 50
    contact = np.zeros(n, bool)
    contact[[1, 11, 21, 31, 41]] = True
    qpos = np.zeros((n, 9)); qvel = np.zeros_like(qpos)
    actions = np.zeros((n, 6))
    for start in (1, 11, 21, 31):
        actions[start+5, :] = 1
    profile = gait.phase_joint_profiles(qpos, qvel, actions, contact, .05, bins=10)
    assert profile['peak_absolute_angle_velocity_action_phase_bin'][12:] == [5]*6


def test_dense_link_rejects_same_path_checkpoint_hash_mismatch(tmp_path):
    import pytest
    snapshot = tmp_path/'checkpoint.pt'
    result = {'snapshot': str(snapshot), 'snapshot_sha256': 'original_digest'}
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        gait.match_dense_result(snapshot, 'overwritten_digest', [result])
    assert gait.match_dense_result(snapshot, 'original_digest', [result]) is result
    assert gait.match_dense_result(tmp_path/'other.pt', 'original_digest', [result]) is None
    with pytest.raises(ValueError, match='Ambiguous'):
        gait.match_dense_result(snapshot, 'original_digest', [result, result])


def test_cycle_displacement_velocity_and_phase_power_use_same_timing():
    n = 50
    contact = np.zeros(n, bool)
    contact[[1, 11, 21, 31, 41]] = True
    qpos = np.zeros((n, 9)); qpos[:, 0] = np.arange(n)*.4
    qvel = np.zeros_like(qpos); actions = np.zeros((n, 6))
    power = np.zeros_like(actions)
    for start in (1, 11, 21, 31):
        power[start+5, :] = 2
    cycles = gait.contact_cycles(contact, .05, qpos[:, 0])
    for cycle in cycles['complete_cycles']:
        np.testing.assert_allclose(cycle['delta_rootx'], 4)
        np.testing.assert_allclose(cycle['mean_cycle_velocity'], 8)
    np.testing.assert_allclose(cycles['complete_cycles_total_displacement_over_time'], 8)
    profile = gait.phase_joint_profiles(qpos, qvel, actions, contact, .05, bins=10, power=power)
    np.testing.assert_array_equal(profile['commanded_joint_power_mean'][5], np.full(6, 2))
    np.testing.assert_array_equal(profile['commanded_joint_power_mean'][0], np.zeros(6))
