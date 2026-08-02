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
    "sf_vlam_v9_for_v18",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v9_leakyrelu05sq.py",
)
MODULE = _load(
    "sf_vlam_v18_fullrank_valueembed",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v18_fullrank_valueembed.py",
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


def test_rank_one_code_pays_a_large_decorrelation_penalty():
    scalar = torch.randn(4096, device="cuda")
    rank_one = scalar.unsqueeze(-1).repeat(1, 7)
    _, _, correlation, scalar_correlation, _ = (
        MODULE.return_code_regularization(rank_one, scalar)
    )

    assert MODULE.effective_rank(rank_one) < 1.01
    assert correlation > 0.8
    assert scalar_correlation > 0.8


def test_nonzero_orthogonal_rich_head_starts_full_rank():
    features = torch.randn(4096, 64, device="cuda")
    head = MODULE.layer_init(
        torch.nn.Linear(64, 7, bias=False), std=0.1
    ).cuda()
    code = head(features)

    assert MODULE.effective_rank(code) > 6.5
    assert code.std(0).min() > 0.08


def test_return_encoder_outputs_full_scale_coordinates_immediately():
    args = MODULE.Args()
    encoder = MODULE.ReturnKernelEncoder(
        args.return_code_dim, args.return_hidden
    ).cuda()
    sampled_return = torch.randn(8192, device="cuda")
    code = encoder(sampled_return)

    torch.testing.assert_close(
        code.mean(0), torch.zeros(args.return_code_dim, device="cuda"), atol=2e-5, rtol=0
    )
    torch.testing.assert_close(
        code.std(0, unbiased=False),
        torch.ones(args.return_code_dim, device="cuda"),
        atol=2e-4,
        rtol=0,
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
    test_rank_one_code_pays_a_large_decorrelation_penalty()
    test_nonzero_orthogonal_rich_head_starts_full_rank()
    test_return_encoder_outputs_full_scale_coordinates_immediately()
    test_auxiliary_shuffle_does_not_advance_global_cuda_rng()
