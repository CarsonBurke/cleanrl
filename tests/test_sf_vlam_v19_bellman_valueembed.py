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
    "sf_vlam_v9_for_v19",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v9_leakyrelu05sq.py",
)
MODULE = _load(
    "sf_vlam_v19_bellman_valueembed",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v19_bellman_valueembed.py",
)


class _FakeEnv:
    single_observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
    )
    single_action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(6,), dtype=np.float32
    )


class _SimpleComposer(torch.nn.Module):
    def forward(self, reward_offset, next_full_value):
        scalar = next_full_value[..., :1]
        rich = next_full_value[..., 1:]
        return 0.25 * rich + 2.0 * scalar + 3.0 * reward_offset.unsqueeze(-1)


class _ExactlyClosedAffineCode(torch.nn.Module):
    def forward(self, reward_offset, next_full_value):
        next_scalar = next_full_value[..., :1]
        return 2.0 * reward_offset.unsqueeze(-1) + next_scalar + 1.0


def test_agent_parameters_rng_and_first_five_outputs_match_v9():
    torch.manual_seed(919)
    base = BASE.Agent(_FakeEnv(), BASE.Args()).cuda()
    expected_rng = torch.get_rng_state()

    torch.manual_seed(919)
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

    _, recursive_scalar, recursive_valid = (
        MODULE.compute_recursive_embedding_targets(
            rewards,
            next_values,
            torch.zeros(2, 4, 1, device="cuda"),
            torch.zeros(1, device="cuda"),
            terminations,
            boundaries,
            valids,
            _SimpleComposer(),
            gamma=0.5,
        )
    )
    torch.testing.assert_close(recursive_scalar, returns)
    torch.testing.assert_close(recursive_valid, valid)


def test_recursive_target_carries_exact_scalar_and_bootstraps_rich_value():
    reward_offsets = torch.tensor([[0.1], [1.0]], device="cuda")
    next_scalar = torch.tensor([[0.7], [1.1]], device="cuda")
    next_rich = torch.tensor([[[9.0]], [[2.0]]], device="cuda")
    flags = torch.zeros(2, 1, device="cuda")
    valids = torch.ones(2, 1, device="cuda")

    rich, scalar, valid = MODULE.compute_recursive_embedding_targets(
        reward_offsets,
        next_scalar,
        next_rich,
        torch.zeros(1, device="cuda"),
        flags,
        flags,
        valids,
        _SimpleComposer(),
        gamma=0.5,
    )

    torch.testing.assert_close(
        scalar, torch.tensor([[0.875], [1.55]], device="cuda")
    )
    torch.testing.assert_close(
        rich, torch.tensor([[[4.825]], [[5.7]]], device="cuda")
    )
    assert valid.all()


def test_recursive_target_matches_an_exactly_representable_event_code():
    reward_offsets = torch.tensor([[0.1], [1.0]], device="cuda")
    next_scalar = torch.tensor([[0.7], [1.1]], device="cuda")
    next_rich = 2.0 * next_scalar.unsqueeze(-1) + 1.0
    flags = torch.zeros(2, 1, device="cuda")
    valids = torch.ones(2, 1, device="cuda")

    rich, scalar, _ = MODULE.compute_recursive_embedding_targets(
        reward_offsets,
        next_scalar,
        next_rich,
        torch.ones(1, device="cuda"),
        flags,
        flags,
        valids,
        _ExactlyClosedAffineCode(),
        gamma=0.5,
    )

    torch.testing.assert_close(rich, 2.0 * scalar.unsqueeze(-1) + 1.0)


def test_composer_is_affine_in_full_next_value():
    composer = MODULE.AffineReturnComposer(
        rich_dim=7,
        hidden=32,
        gamma=0.99,
        matrix_scale=0.1,
    ).cuda()
    reward_offset = torch.randn(128, device="cuda")
    first = torch.randn(128, 8, device="cuda")
    second = torch.randn(128, 8, device="cuda")
    mixing = 0.37

    mixed_output = composer(
        reward_offset, mixing * first + (1.0 - mixing) * second
    )
    expected = (
        mixing * composer(reward_offset, first)
        + (1.0 - mixing) * composer(reward_offset, second)
    )
    torch.testing.assert_close(mixed_output, expected, atol=2e-6, rtol=2e-6)


def test_composer_rich_dynamics_are_row_contracting_and_trainable():
    composer = MODULE.AffineReturnComposer(
        rich_dim=7,
        hidden=32,
        gamma=0.99,
        matrix_scale=0.1,
    ).cuda()
    reward_offset = torch.linspace(-10.0, 10.0, 31, device="cuda")
    zero = torch.zeros(31, 8, device="cuda")
    baseline = composer(reward_offset, zero)
    columns = []
    for coordinate in range(7):
        basis = zero.clone()
        basis[:, coordinate + 1] = 1.0
        columns.append(composer(reward_offset, basis) - baseline)
    rich_matrix = torch.stack(columns, dim=-1)

    assert rich_matrix.abs().sum(-1).max() <= 0.99901
    rich_matrix.sum().backward()
    gradient = composer.diagonal_logits.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().min() > 0


def test_return_encoder_is_not_batch_relative():
    encoder = MODULE.ReturnKernelEncoder(7, 64).cuda()
    sampled_return = torch.randn(1024, device="cuda")

    alone = encoder(sampled_return[:1])
    in_batch = encoder(sampled_return)[:1]
    torch.testing.assert_close(alone, in_batch)


def test_return_encoder_stays_finite_at_extreme_returns():
    encoder = MODULE.ReturnKernelEncoder(7, 64).cuda()
    extreme_return = torch.tensor(
        [-1e20, -1e10, 0.0, 1e10, 1e20], device="cuda"
    )
    code = encoder(extreme_return)
    assert torch.isfinite(code).all()

    code.square().mean().backward()
    for parameter in encoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_cross_correlation_alignment_penalizes_duplicate_coordinates():
    target = torch.randn(8192, 7, device="cuda")
    exact, _, _ = MODULE.cross_correlation_alignment(target, target)
    duplicate = target[:, :1].expand_as(target)
    collapsed, _, _ = MODULE.cross_correlation_alignment(duplicate, target)

    assert exact < 0.01
    assert collapsed > exact + 0.5


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
    test_recursive_target_carries_exact_scalar_and_bootstraps_rich_value()
    test_recursive_target_matches_an_exactly_representable_event_code()
    test_composer_is_affine_in_full_next_value()
    test_composer_rich_dynamics_are_row_contracting_and_trainable()
    test_return_encoder_is_not_batch_relative()
    test_return_encoder_stays_finite_at_extreme_returns()
    test_cross_correlation_alignment_penalizes_duplicate_coordinates()
    test_correlation_regularizer_has_no_false_radial_shrink_gradient()
    test_rank_one_code_pays_a_large_decorrelation_penalty()
    test_auxiliary_shuffle_does_not_advance_global_cuda_rng()
