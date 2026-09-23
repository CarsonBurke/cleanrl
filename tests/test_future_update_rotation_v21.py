"""Contracts for direction-only updates and credit from genuinely future batches.

CUDA tests: execute through mlq. These are estimator/optimizer checks, not
shortened learning runs or evidence of benchmark performance.
"""

import copy
import math

import pytest
import torch

from cleanrl.ppo_continuous_action_future_update_rotation_v21 import (
    Args,
    FutureUpdateController,
    ResidualMLP,
    UpdateRotation,
    rotate_displacement,
)
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(scope="module", autouse=True)
def cuda_runtime():
    if not torch.cuda.is_available():
        pytest.skip("The update learner requires CUDA")
    configure_runtime(matmul_precision="highest", allow_tf32=False)


@pytest.fixture(scope="module")
def rotate():
    return torch.compile(rotate_displacement, fullgraph=True,
                         options={"triton.cudagraphs": False})


def test_rotation_changes_direction_without_shrinking_step(rotate):
    displacement = torch.tensor([3.0, 0.0, 0.0], device="cuda")
    descent = torch.tensor([1.0, 2.0, 0.0], device="cuda")
    angle = torch.tensor(math.pi / 3, device="cuda")
    rotated = rotate(displacement, descent, angle)
    expected = torch.tensor([1.5, 3 * math.sqrt(3) / 2, 0.0], device="cuda")
    torch.testing.assert_close(rotated, expected)
    torch.testing.assert_close(rotated.norm(), displacement.norm())
    assert not torch.allclose(rotated, displacement)


def test_zero_and_degenerate_rotation_preserve_adam_step(rotate):
    displacement = torch.tensor([2.0, -1.0, 3.0], device="cuda")
    zero = torch.zeros((), device="cuda")
    torch.testing.assert_close(rotate(displacement, displacement.flip(0), zero),
                               displacement, rtol=0, atol=0)
    angle = torch.tensor(0.8, device="cuda")
    for descent in (displacement * 2, -displacement, torch.zeros_like(displacement)):
        torch.testing.assert_close(rotate(displacement, descent, angle),
                                   displacement, rtol=0, atol=0)
    torch.testing.assert_close(rotate(torch.zeros_like(displacement), displacement, angle),
                               torch.zeros_like(displacement), rtol=0, atol=0)


def test_rotation_preserves_norm_for_matrix_and_nearly_parallel_directions(rotate):
    generator = torch.Generator(device="cuda").manual_seed(41)
    displacement = torch.randn(32, 17, generator=generator, device="cuda")
    noise = torch.randn(32, 17, generator=generator, device="cuda")
    angle = torch.tensor(-0.37, device="cuda")
    for descent in (noise, displacement + 1e-5 * noise):
        rotated = rotate(displacement, descent, angle)
        assert torch.isfinite(rotated).all()
        torch.testing.assert_close(rotated.norm(), displacement.norm(), rtol=2e-5, atol=1e-6)


def test_identity_wrapper_preserves_adam_parameters_and_moment_history():
    torch.manual_seed(9)
    actor = ResidualMLP(17, 12, output_std=0.01).cuda()
    reference = copy.deepcopy(actor)
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.0024, eps=1e-5, fused=True)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.0024, eps=1e-5, fused=True)
    rotation = UpdateRotation(actor, compile=True)
    angles = torch.zeros(3, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(23)
    for _ in range(3):
        for parameter, expected in zip(actor.parameters(), reference.parameters(), strict=True):
            gradient = torch.randn(parameter.shape, generator=generator, device="cuda")
            parameter.grad = gradient.clone()
            expected.grad = gradient.clone()
        rotation.capture()
        optimizer.step()
        reference_optimizer.step()
        rotation.apply(angles)
        for parameter, expected in zip(actor.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
            for key in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(optimizer.state[parameter][key],
                                           reference_optimizer.state[expected][key], rtol=0, atol=0)


def test_directional_wrapper_keeps_adam_step_norms_and_does_not_rewrite_moments():
    torch.manual_seed(19)
    actor = ResidualMLP(17, 12, output_std=0.01).cuda()
    reference = copy.deepcopy(actor)
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.0024, eps=1e-5, fused=True)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.0024, eps=1e-5, fused=True)
    rotation = UpdateRotation(actor, compile=True)
    generator = torch.Generator(device="cuda").manual_seed(37)
    old_parameters = [parameter.detach().clone() for parameter in actor.parameters()]
    for parameter, expected in zip(actor.parameters(), reference.parameters(), strict=True):
        gradient = torch.randn(parameter.shape, generator=generator, device="cuda")
        parameter.grad = gradient.clone()
        expected.grad = gradient.clone()
    rotation.capture()
    optimizer.step()
    reference_optimizer.step()
    rotation.apply(torch.tensor([0.2, -0.3, 0.4], device="cuda"))
    changed = False
    for before, parameter, expected in zip(old_parameters, actor.parameters(), reference.parameters(), strict=True):
        actual_step = parameter.detach() - before
        adam_step = expected.detach() - before
        torch.testing.assert_close(actual_step.norm(), adam_step.norm(), rtol=2e-4, atol=1e-7)
        changed |= not torch.equal(parameter, expected)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(optimizer.state[parameter][key],
                                       reference_optimizer.state[expected][key], rtol=0, atol=0)
    assert changed, "The learned control must change direction, not just record angles"


def test_meta_credit_waits_for_future_rollouts_and_uses_frozen_baseline():
    args = Args(meta_mode="learn", meta_horizon=4, meta_credit_horizon=4)
    controller = FutureUpdateController(args, torch.device("cuda"))
    controller.observe(10.0, 0.1)
    sampled_angles = controller.angles.clone()
    initial_weights = controller.weights.detach().clone()
    for reward in (11.0, 12.0, 13.0):
        controller.observe(reward, 0.2)
        torch.testing.assert_close(controller.angles, sampled_angles, rtol=0, atol=0)
        torch.testing.assert_close(controller.weights, initial_weights, rtol=0, atol=0)
    controller.observe(14.0, 0.2)
    # Mean future reward is 12.5; the baseline remains pre-intervention 10.
    # At zero initial mean the Gaussian location score is a / sigma^2.
    expected_gradient = -0.25 * sampled_angles / args.meta_std**2
    torch.testing.assert_close(controller.weights.grad[:, 0], expected_gradient)
    assert torch.any(controller.weights != initial_weights)


def test_credit_horizon_changes_objective_not_action_hold_or_decision_frequency():
    short_args = Args(meta_mode="learn", meta_horizon=4, meta_credit_horizon=1)
    long_args = Args(meta_mode="learn", meta_horizon=4, meta_credit_horizon=4)
    short = FutureUpdateController(short_args, torch.device("cuda"))
    long = FutureUpdateController(long_args, torch.device("cuda"))
    short.observe(10.0, 0.1)
    long.observe(10.0, 0.1)
    sampled_angles = short.angles.clone()
    torch.testing.assert_close(long.angles, sampled_angles, rtol=0, atol=0)
    # Immediate outcome improves, subsequent learning makes outcomes worse.
    for reward in (12.0, 4.0, 4.0):
        short.observe(reward, 0.2)
        long.observe(reward, 0.2)
        torch.testing.assert_close(short.angles, sampled_angles, rtol=0, atol=0)
        torch.testing.assert_close(long.angles, sampled_angles, rtol=0, atol=0)
    short.observe(4.0, 0.2)
    long.observe(4.0, 0.2)
    score = sampled_angles / short_args.meta_std**2
    torch.testing.assert_close(short.weights.grad[:, 0], -0.2 * score)
    torch.testing.assert_close(long.weights.grad[:, 0], 0.4 * score)
    # The two objectives should reward opposite directional interventions.
    assert torch.all(short.weights[:, 0] * long.weights[:, 0] < 0)


def test_midwindow_restore_preserves_delayed_credit_and_independent_rng():
    args = Args(meta_mode="learn", meta_horizon=4, meta_credit_horizon=4)
    original = FutureUpdateController(args, torch.device("cuda"))
    original.observe(5.0, 0.1)
    original.observe(6.0, 0.2)
    original.observe(7.0, 0.3)
    state = copy.deepcopy(original.state_dict())
    restored = FutureUpdateController(args, torch.device("cuda"))
    restored.load_state_dict(state)
    for index, reward in enumerate((8.0, 9.0, 12.0, 8.0, 7.0, 6.0)):
        # Unrelated RNG consumption must not change controller exploration.
        torch.randn(11, device="cuda")
        original.observe(reward, 0.4 + index / 100)
        torch.randn(37, device="cuda")
        restored.observe(reward, 0.4 + index / 100)
        torch.testing.assert_close(restored.angles, original.angles, rtol=0, atol=0)
        torch.testing.assert_close(restored.weights, original.weights, rtol=0, atol=0)


def test_frozen_random_control_does_not_learn_from_delayed_rewards():
    args = Args(meta_mode="random", meta_horizon=4, meta_credit_horizon=4)
    one = FutureUpdateController(args, torch.device("cuda"))
    other = FutureUpdateController(args, torch.device("cuda"))
    one.observe(1.0, 0.0)
    other.observe(10.0, 0.0)
    initial_weights = one.weights.detach().clone()
    for reward in (2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0, 11.0):
        one.observe(reward, 0.5)
        other.observe(-reward, 0.5)
        torch.testing.assert_close(one.angles, other.angles, rtol=0, atol=0)
        torch.testing.assert_close(one.weights, initial_weights, rtol=0, atol=0)
