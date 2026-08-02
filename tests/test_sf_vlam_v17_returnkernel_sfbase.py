import importlib.util
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


ROOT = Path(__file__).parents[1]


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load(
    "sf_vlam_v9_for_v17",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v9_leakyrelu05sq.py",
)
MODULE = _load(
    "sf_vlam_v17_returnkernel_sfbase",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v17_returnkernel_sfbase.py",
)


class _FakeEnv:
    single_observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
    )
    single_action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(6,), dtype=np.float32
    )


def test_agent_parameters_rng_and_first_five_outputs_match_v9():
    torch.manual_seed(918)
    base = BASE.Agent(_FakeEnv(), BASE.Args()).cuda()
    expected_rng = torch.get_rng_state()

    torch.manual_seed(918)
    treatment = MODULE.Agent(_FakeEnv(), MODULE.Args()).cuda()
    actual_rng = torch.get_rng_state()

    torch.testing.assert_close(actual_rng, expected_rng)
    assert list(base.state_dict()) == list(treatment.state_dict())
    for actual, expected in zip(
        treatment.parameters(), base.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)

    observation = torch.randn(127, 17, device="cuda")
    native_action = torch.rand(127, 6, device="cuda").clamp(1e-4, 1.0 - 1e-4)
    base_outputs = base.get_action_and_value(observation, native_action)
    treatment_outputs = treatment.get_action_and_value(
        observation, native_action
    )
    for actual, expected in zip(
        treatment_outputs[:5], base_outputs, strict=True
    ):
        torch.testing.assert_close(actual, expected)


def test_sampled_return_respects_all_boundary_semantics():
    rewards = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]],
        device="cuda",
    )
    next_values = torch.tensor(
        [[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0]],
        device="cuda",
    )
    terminations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        device="cuda",
    )
    boundaries = torch.tensor(
        [[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        device="cuda",
    )
    valids = torch.tensor(
        [[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        device="cuda",
    )
    returns, valid = MODULE.compute_sampled_returns(
        rewards,
        next_values,
        terminations,
        boundaries,
        valids,
        gamma=0.5,
    )

    expected = torch.tensor(
        [
            [1.0, 6.0, 7.5, 27.5],
            [15.5, 26.0, 36.5, 47.0],
        ],
        device="cuda",
    )
    torch.testing.assert_close(returns, expected)
    assert valid.tolist() == [[True, True, False, True], [True, True, True, True]]


def test_correlation_regularizer_has_no_false_radial_shrink_gradient():
    scalar = torch.randn(4096, device="cuda")
    coords = torch.randn(4096, 7, device="cuda", requires_grad=True)
    _, _, correlation, scalar_correlation, _ = (
        MODULE.return_code_regularization(coords, scalar)
    )
    decorrelation = correlation + scalar_correlation
    gradient = torch.autograd.grad(decorrelation, coords)[0]
    radial_derivative = (gradient * coords).sum().abs()

    assert radial_derivative < 2e-4


def test_auxiliary_gradient_is_orthogonalized_and_capped():
    parameter = torch.nn.Parameter(torch.zeros(3, device="cuda"))
    critic = {parameter: torch.tensor([1.0, 0.0, 0.0], device="cuda")}
    actor = {parameter: torch.tensor([1.0, 1.0, 0.0], device="cuda")}
    primary = {parameter: critic[parameter] + actor[parameter]}
    auxiliary = {parameter: torch.tensor([-1.0, 2.0, 4.0], device="cuda")}
    MODULE.protect_auxiliary_trunk_gradient(
        [critic, actor], auxiliary, [parameter], primary, max_ratio=0.25
    )

    assert torch.dot(critic[parameter], auxiliary[parameter]).abs() < 1e-7
    assert torch.dot(actor[parameter], auxiliary[parameter]).abs() < 1e-7
    assert auxiliary[parameter].norm() <= 0.25 * primary[parameter].norm() + 1e-7


def test_gate_rejects_even_one_collapsed_coordinate():
    args = MODULE.Args()
    live_target_std = torch.ones(args.return_code_dim, device="cuda")
    live_prediction_std = torch.full(
        (args.return_code_dim,), 0.5, device="cuda"
    )
    assert MODULE.passes_rich_gate(
        0.5, 5.0, 4.0, live_target_std, live_prediction_std, args
    )

    live_target_std[-1] = 0.0
    assert not MODULE.passes_rich_gate(
        0.5, 5.0, 4.0, live_target_std, live_prediction_std, args
    )
    live_target_std[-1] = 1.0
    live_prediction_std[-1] = 0.0
    assert not MODULE.passes_rich_gate(
        0.5, 5.0, 4.0, live_target_std, live_prediction_std, args
    )


def test_auxiliary_shuffle_does_not_advance_global_cuda_rng():
    global_before = torch.cuda.get_rng_state()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(170_018)
    torch.randperm(4096, device="cuda", generator=generator)
    global_after = torch.cuda.get_rng_state()

    torch.testing.assert_close(global_after, global_before)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test")
    test_agent_parameters_rng_and_first_five_outputs_match_v9()
    test_sampled_return_respects_all_boundary_semantics()
    test_correlation_regularizer_has_no_false_radial_shrink_gradient()
    test_auxiliary_gradient_is_orthogonalized_and_capped()
    test_gate_rejects_even_one_collapsed_coordinate()
    test_auxiliary_shuffle_does_not_advance_global_cuda_rng()
