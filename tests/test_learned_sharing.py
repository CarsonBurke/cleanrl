import torch

from cleanrl.plasticity.learned_sharing_v9 import LearnedSharingState


def test_shared_signal_learns_negative_nonunit_amplitude_one_observation_at_a_time():
    weight = torch.zeros(1, 8, device="cuda")
    state = LearnedSharingState(weight)
    step = torch.compile(state.step, fullgraph=True)
    for index in range(120):
        x = (torch.arange(8, device="cuda") % 3 != index % 3).float()[None, :]
        target = -1.7 * x.sum()
        gradient = ((weight * x).sum() - target) * x
        step(weight, gradient, x.square())
    torch.testing.assert_close(weight, torch.full_like(weight, -1.7), rtol=0.01, atol=0.01)


def test_balanced_noise_is_not_written_as_shared_signal():
    weight = torch.zeros(1, 8, device="cuda")
    state = LearnedSharingState(weight)
    x = torch.ones_like(weight)
    for index in range(100):
        target = 1.0 if index % 2 == 0 else -1.0
        state.step(weight, ((weight * x).sum() - target) * x, x.square())
    torch.testing.assert_close(weight, torch.zeros_like(weight), rtol=0, atol=0)


def test_scalar_fit_does_not_snap_positive_signal_to_one():
    weight = torch.zeros(1, 1, device="cuda")
    state = LearnedSharingState(weight, control="no_dense")
    for index in range(100):
        target = 2.3 + (0.2 if index % 2 == 0 else -0.2)
        state.step(weight, weight - target, torch.ones_like(weight))
    torch.testing.assert_close(weight, torch.full_like(weight, 2.3), rtol=0.02, atol=0.02)


def test_three_nearly_equal_observations_do_not_certify_a_direction():
    weight = torch.zeros(1, 8, device="cuda")
    state = LearnedSharingState(weight)
    x = torch.ones_like(weight)
    for target in (1.0, 1.0001, 0.9999):
        state.step(weight, ((weight * x).sum() - target) * x, x.square())
    torch.testing.assert_close(weight, torch.zeros_like(weight), rtol=0, atol=0)
