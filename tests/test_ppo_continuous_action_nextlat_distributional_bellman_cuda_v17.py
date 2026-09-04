import importlib.util
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch


ROOT = Path(__file__).parents[1]
MODULE_PATH = (
    ROOT / "cleanrl/nextlat/ppo_continuous_action_nextlat_distributional_bellman_cuda_v17.py"
)
LEGACY_MODULE_PATH = (
    ROOT / "cleanrl/nextlat/ppo_continuous_action_nextlat_distributional_bellman_v11.py"
)


def _load_module():
    name = "ppo_continuous_action_nextlat_distributional_bellman_cuda_v17_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _load_legacy_module():
    name = "ppo_continuous_action_nextlat_distributional_bellman_v11_oracle"
    spec = importlib.util.spec_from_file_location(name, LEGACY_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LEGACY_MODULE = _load_legacy_module()


class _FakeEnv:
    single_observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
    )
    single_action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(6,), dtype=np.float32
    )


def _decoded_expectation(distribution, support):
    return (distribution * support).sum(-1)


def test_defaults_and_horizon_schedule_match_the_benchmark_design():
    args = MODULE.Args()
    assert args.env_id == "HalfCheetah-v4"
    assert args.total_timesteps == 8_000_000
    assert args.seed == 1
    assert args.cuda is True
    assert args.num_bins == 511
    assert MODULE.parse_horizons(args.nextlat_horizons) == (1, 2, 4, 8)
    assert MODULE.parse_horizons(args.bellman_horizons) == (1, 2, 4)
    assert args.predadam_trust_ratio == 0.05
    critic_bytes = (
        args.num_steps
        * args.num_envs
        * args.critic_mtp_horizon
        * args.num_bins
        * torch.tensor([], dtype=torch.float32).element_size()
    )
    bellman_bytes = (
        args.num_steps
        * args.num_envs
        * len(MODULE.parse_horizons(args.bellman_horizons))
        * args.num_bins
        * torch.tensor([], dtype=torch.float16).element_size()
    )
    # Regression guard for the deliberate GPU-resident label design: ~383 MiB +
    # ~96 MiB at defaults, small relative to the required 32 GiB single-run budget.
    assert 478.0 < (critic_bytes + bellman_bytes) / (1024**2) < 480.0


def test_cuda_residency_eliminates_over_639_mib_of_legacy_transfer_per_iteration():
    args = MODULE.Args()
    rows = args.num_steps * args.num_envs
    scalar_bytes = torch.tensor([], dtype=torch.float32).element_size()
    legacy_transfer_bytes = (
        # return_mtp CUDA -> CPU, then its dense HL-Gauss labels CPU -> CUDA
        rows * args.critic_mtp_horizon * scalar_bytes
        + rows * args.critic_mtp_horizon * args.num_bins * scalar_bytes
        # frozen bootstrap distribution CUDA -> CPU and Bellman labels CPU -> CUDA
        + rows * args.num_bins * scalar_bytes
        + rows
        * len(MODULE.parse_horizons(args.bellman_horizons))
        * args.num_bins
        * scalar_bytes
        # rewards, three transition flags, and decoded bootstrap values CUDA -> CPU
        + rows * 5 * scalar_bytes
    )
    assert legacy_transfer_bytes / (1024**2) > 639.0

    source = MODULE_PATH.read_text()
    label_region = source[
        source.index("# Project the ~383 MiB table directly on CUDA") :
        source.index("# A predicted latent may not cross an episode reset")
    ]
    assert ".cpu(" not in label_region
    assert "hl_support_cpu" not in source
    assert "critic_label_fn(return_mtp.detach())" in label_region
    assert "bellman_label_fn(" in label_region


def test_performance_paths_use_tf32_cuda_graphs_stable_clones_and_async_checks():
    source = MODULE_PATH.read_text()
    assert 'torch.set_float32_matmul_precision("high")' in source
    assert "mode=args.compile_mode, dynamic=False, fullgraph=True" in source
    assert "target_probs_graph.clone() if args.compile" in source
    assert "torch.compiler.cudagraph_mark_step_begin()" in source
    assert "torch._assert_async(" in source
    assert source.count("nn.utils.clip_grad_norm_(") == 1

    update_loop = source[
        source.index("for epoch in range(args.update_epochs):") :
        source.index("if args.target_kl is not None and approx_kl")
    ]
    assert ".item()" not in update_loop
    assert "clipfrac_sum = clipfrac_sum +" in update_loop
    assert "ret_perc_scale_log = mb_perc_scale.detach()" in update_loop

    bellman_impl = source[
        source.index("def _build_nstep_distributional_targets_impl(") :
        source.index("def build_nstep_distributional_targets(")
    ]
    assert ".cpu(" not in bellman_impl
    assert ".any()" not in bellman_impl
    assert ".nonzero(" not in bellman_impl


def test_nonuniform_categorical_projection_preserves_mass_and_expectation():
    support = torch.tensor([-5.0, -1.0, 0.0, 2.0, 9.0])
    probabilities = torch.tensor(
        [[0.05, 0.15, 0.30, 0.40, 0.10], [0.0, 0.0, 1.0, 0.0, 0.0]]
    )
    shift = torch.tensor([0.5, -0.75])
    scale = torch.tensor([0.5, 1.25])
    projected = MODULE.categorical_affine_projection(
        probabilities, shift, scale, support
    )

    transformed_expectation = shift + scale * _decoded_expectation(
        probabilities, support
    )
    torch.testing.assert_close(projected.sum(-1), torch.ones(2))
    torch.testing.assert_close(
        _decoded_expectation(projected, support), transformed_expectation
    )
    assert torch.all(projected >= 0.0)


def test_nstep_targets_handle_terminal_truncation_missing_final_obs_and_tail():
    gamma = 0.9
    time_steps, num_envs = 6, 4
    support = torch.linspace(-20.0, 20.0, 81)
    rewards = torch.ones(time_steps, num_envs)
    terminations = torch.zeros_like(rewards)
    boundaries = torch.zeros_like(rewards)
    valids = torch.ones_like(rewards)

    # env 1 terminates after transition 1; env 2 truncates there with a final
    # observation; env 3 truncates there without one and must be censored.
    terminations[1, 1] = 1.0
    boundaries[1, 1:] = 1.0
    valids[1, 3] = 0.0
    # A terminal on the final stored transition makes even n=4 complete at the tail.
    terminations[-1, 1] = 1.0
    boundaries[-1, 1] = 1.0

    bootstrap_value = 4.0
    bootstrap_index = int((bootstrap_value - support[0]).item() / 0.5)
    next_probabilities = torch.zeros(time_steps, num_envs, support.numel())
    next_probabilities[..., bootstrap_index] = 1.0
    next_values = torch.full_like(rewards, bootstrap_value)

    targets, masks = MODULE.build_nstep_distributional_targets(
        rewards,
        terminations,
        boundaries,
        valids,
        next_probabilities,
        next_values,
        support,
        gamma,
        1.0,
        horizons=(1, 2, 4),
        projection_chunk=7,
    )
    decoded = _decoded_expectation(targets, support)

    # No boundary: two observed rewards and the distribution at s_{t+2}.
    torch.testing.assert_close(
        decoded[0, 0, 1], torch.tensor(1.0 + gamma + gamma**2 * bootstrap_value)
    )
    # Termination includes its reward and drops all bootstrap mass.
    torch.testing.assert_close(decoded[0, 1, 2], torch.tensor(1.0 + gamma))
    # Truncation bootstraps the supplied final observation at its actual stopping time.
    torch.testing.assert_close(
        decoded[0, 2, 2], torch.tensor(1.0 + gamma + gamma**2 * bootstrap_value)
    )
    assert masks[0, 1, 2]
    assert masks[0, 2, 2]
    assert not masks[0, 3, 2]

    # A full n=2 target may bootstrap from next_obs of the last transition; n=4
    # beginning one step later runs off the buffer and is invalid without a boundary.
    assert masks[4, 0, 1]
    assert not masks[3, 0, 2]
    # An observed terminal completes a long target even at the rollout tail.
    assert masks[5, 1, 2]
    torch.testing.assert_close(decoded[5, 1, 2], torch.tensor(1.0))

    valid_targets = targets[masks]
    torch.testing.assert_close(
        valid_targets.sum(-1), torch.ones(valid_targets.shape[0]), atol=1e-6, rtol=1e-6
    )


def test_nstep_targets_match_recursive_lambda_return_semantics():
    gamma, gae_lambda = 0.9, 0.6
    support = torch.linspace(-20.0, 20.0, 401)
    rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    terminations = torch.zeros_like(rewards)
    boundaries = torch.zeros_like(rewards)
    valids = torch.ones_like(rewards)
    next_values = torch.tensor([[5.0], [6.0], [7.0], [8.0]])
    bootstrap_values = torch.tensor([[5.0], [6.0], [7.0], [8.0]])
    next_probabilities = torch.zeros(4, 1, support.numel())
    indices = ((bootstrap_values - support[0]) / (support[1] - support[0])).round().long()
    next_probabilities.scatter_(-1, indices.unsqueeze(-1), 1.0)

    targets, masks = MODULE.build_nstep_distributional_targets(
        rewards,
        terminations,
        boundaries,
        valids,
        next_probabilities,
        next_values,
        support,
        gamma,
        gae_lambda,
        horizons=(1, 3),
    )
    decoded = _decoded_expectation(targets, support)
    trace = gamma * gae_lambda
    expected_n1 = 1.0 + gamma * (1.0 - gae_lambda) * 5.0 + trace * 5.0
    expected_n3 = (
        1.0
        + gamma * (1.0 - gae_lambda) * 5.0
        + trace * (2.0 + gamma * (1.0 - gae_lambda) * 6.0)
        + trace**2 * (3.0 + gamma * (1.0 - gae_lambda) * 7.0)
        + trace**3 * 7.0
    )
    assert masks[0, 0].all()
    torch.testing.assert_close(decoded[0, 0, 0], torch.tensor(expected_n1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(decoded[0, 0, 1], torch.tensor(expected_n3), atol=1e-5, rtol=1e-5)


def test_lambda_targets_handle_terminal_and_truncation_without_lambda_leakage():
    gamma, gae_lambda = 0.9, 0.4
    support = torch.linspace(-20.0, 20.0, 401)
    rewards = torch.ones(3, 3)
    terminations = torch.zeros_like(rewards)
    boundaries = torch.zeros_like(rewards)
    valids = torch.ones_like(rewards)
    terminations[1, 0] = 1.0
    boundaries[1, 0:] = 1.0
    valids[1, 2] = 0.0
    next_values = torch.full_like(rewards, 5.0)
    next_probabilities = torch.zeros(3, 3, support.numel())
    bootstrap_index = int(round((5.0 - support[0].item()) / (support[1] - support[0]).item()))
    next_probabilities[..., bootstrap_index] = 1.0

    targets, masks = MODULE.build_nstep_distributional_targets(
        rewards,
        terminations,
        boundaries,
        valids,
        next_probabilities,
        next_values,
        support,
        gamma,
        gae_lambda,
        horizons=(3,),
    )
    decoded = _decoded_expectation(targets, support)
    trace = gamma * gae_lambda
    first_step = 1.0 + gamma * (1.0 - gae_lambda) * 5.0
    # At the second transition, termination gets only its reward. Truncation gets
    # the full gamma*V(final), independent of lambda. A missing final obs is censored.
    torch.testing.assert_close(decoded[0, 0, 0], torch.tensor(first_step + trace * 1.0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        decoded[0, 1, 0], torch.tensor(first_step + trace * (1.0 + gamma * 5.0)), atol=1e-5, rtol=1e-5
    )
    assert masks[0, 0, 0]
    assert masks[0, 1, 0]
    assert not masks[0, 2, 0]


def test_zero_lambda_horizons_reduce_to_valid_one_step_targets_at_tail():
    support = torch.linspace(-10.0, 10.0, 201)
    rewards = torch.tensor([[1.0], [2.0]])
    flags = torch.zeros_like(rewards)
    valids = torch.ones_like(rewards)
    next_values = torch.tensor([[3.0], [4.0]])
    next_probabilities = torch.zeros(2, 1, support.numel())
    next_probabilities[..., support.numel() // 2] = 1.0
    targets, masks = MODULE.build_nstep_distributional_targets(
        rewards,
        flags,
        flags,
        valids,
        next_probabilities,
        next_values,
        support,
        gamma=0.9,
        gae_lambda=0.0,
        horizons=(1, 8),
    )
    decoded = _decoded_expectation(targets, support)
    assert masks[:, :, 1].all()
    torch.testing.assert_close(decoded[:, :, 0], decoded[:, :, 1])
    torch.testing.assert_close(decoded[-1, 0, 1], torch.tensor(2.0 + 0.9 * 4.0))


def _randomized_bellman_case(dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    time_steps = 3 + seed % 9
    num_envs = 1 + (seed * 3) % 5
    num_bins = 17
    coord = torch.linspace(-3.0, 3.0, num_bins, dtype=dtype)
    support = coord.sign() * (coord.abs().exp() - 1.0)
    rewards = torch.randn(
        time_steps, num_envs, generator=generator, dtype=dtype
    )
    terminations = torch.rand(time_steps, num_envs, generator=generator) < 0.17
    truncations = torch.rand(time_steps, num_envs, generator=generator) < 0.19
    boundaries = terminations | truncations
    # Boundary rows may or may not supply final_observation. Also exercise missing
    # transitions without a reset, which the recurrence must censor.
    transition_valids = (
        torch.rand(time_steps, num_envs, generator=generator) < 0.82
    )
    logits = torch.randn(
        time_steps, num_envs, num_bins, generator=generator, dtype=dtype
    )
    next_probabilities = torch.softmax(logits, dim=-1)
    next_values = (next_probabilities * support).sum(-1)
    gae_lambda = (0.0, 0.4, 0.95, 1.0)[seed % 4]
    return (
        rewards,
        terminations,
        boundaries,
        transition_valids,
        next_probabilities,
        next_values,
        support,
        gae_lambda,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tensor_only_builder_matches_v11_randomized_oracle(dtype):
    atol = 3e-7 if dtype == torch.float32 else 2e-15
    rtol = 2e-6 if dtype == torch.float32 else 2e-14
    for seed in range(12):
        case = _randomized_bellman_case(dtype, seed)
        *inputs, gae_lambda = case
        legacy_targets, legacy_masks = LEGACY_MODULE.build_nstep_distributional_targets(
            *inputs,
            gamma=0.99,
            gae_lambda=gae_lambda,
            horizons=(1, 2, 4, 8),
            projection_chunk=3,
        )
        targets, masks = MODULE.build_nstep_distributional_targets(
            *inputs,
            gamma=0.99,
            gae_lambda=gae_lambda,
            horizons=(1, 2, 4, 8),
            projection_chunk=3,
        )
        assert torch.equal(masks, legacy_masks)
        torch.testing.assert_close(targets, legacy_targets, atol=atol, rtol=rtol)
        assert torch.count_nonzero(targets[~masks]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA oracle requires a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_targets_match_cpu_across_randomized_masks_and_precision(dtype):
    atol = 5e-6 if dtype == torch.float32 else 2e-12
    rtol = 2e-5 if dtype == torch.float32 else 2e-11
    for seed in range(8):
        cpu_case = _randomized_bellman_case(dtype, seed + 100)
        *cpu_inputs, gae_lambda = cpu_case
        cpu_targets, cpu_masks = MODULE.build_nstep_distributional_targets(
            *cpu_inputs,
            gamma=0.99,
            gae_lambda=gae_lambda,
            horizons=(1, 2, 4, 8),
            projection_chunk=5,
        )
        cuda_inputs = [tensor.cuda() for tensor in cpu_inputs]
        cuda_targets, cuda_masks = MODULE.build_nstep_distributional_targets(
            *cuda_inputs,
            gamma=0.99,
            gae_lambda=gae_lambda,
            horizons=(1, 2, 4, 8),
            projection_chunk=5,
        )
        assert torch.equal(cuda_masks.cpu(), cpu_masks)
        torch.testing.assert_close(
            cuda_targets.cpu(), cpu_targets, atol=atol, rtol=rtol
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA oracle requires a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_hl_gauss_projection_matches_cpu(dtype):
    targets = torch.randn(9, 7, 6, generator=torch.Generator().manual_seed(99)).to(
        dtype
    ) * 100.0
    cpu_support = MODULE.Dreamer3BucketHLGaussSupport(
        511, -9.90353755128617, 9.90353755128617, 0.75, torch.device("cpu")
    )
    cuda_support = MODULE.Dreamer3BucketHLGaussSupport(
        511, -9.90353755128617, 9.90353755128617, 0.75, torch.device("cuda")
    )
    # Supports are FP32 in training; float64 exercises target input promotion without
    # changing that production label geometry.
    expected = cpu_support.project(targets)
    actual = cuda_support.project(targets.cuda()).cpu()
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)


def test_flat_index_shift_preserves_environment_at_the_rollout_tail():
    # t-major indices for (t=3, env=0) and (t=3, env=2), shifted beyond T=4.
    shifted = MODULE.shifted_flat_indices(
        np.array([9, 11]), shift=8, num_envs=3, num_steps=4
    )
    np.testing.assert_array_equal(shifted, np.array([9, 11]))

    tensor_indices = torch.tensor([9, 11], device="cpu")
    tensor_shifted = MODULE.shifted_flat_indices(
        tensor_indices, shift=8, num_envs=3, num_steps=4
    )
    torch.testing.assert_close(tensor_shifted, torch.tensor([9, 11]))


def test_block_local_admission_protects_both_task_losses_and_caps_each_block():
    task_updates = [torch.tensor([100.0]), torch.tensor([1.0, 0.0])]
    predictive_updates = [torch.tensor([0.0]), torch.tensor([100.0, 100.0])]
    actor_gradients = [torch.tensor([0.0]), torch.tensor([1.0, 0.0])]
    critic_gradients = [torch.tensor([0.0]), torch.tensor([0.0, 1.0])]

    admitted, stats = MODULE.admit_predictive_updates(
        task_updates,
        predictive_updates,
        max_ratio=0.05,
        actor_gradients=actor_gradients,
        critic_gradients=critic_gradients,
    )

    # The second proposal increases both losses and is projected to zero. In
    # particular, it cannot borrow the first block's large trust budget.
    torch.testing.assert_close(admitted[0], torch.zeros_like(admitted[0]))
    torch.testing.assert_close(admitted[1], torch.zeros_like(admitted[1]))
    assert stats["actor_first_order"] <= 0.0
    assert stats["critic_first_order"] <= 0.0
    assert stats["max_block_ratio"] <= 0.05 + 1e-7
    assert stats["actor_conflict_fraction"] == 0.5
    assert stats["critic_conflict_fraction"] == 0.5


def test_block_local_admission_retains_safe_direction_at_local_trust_limit():
    task_updates = [torch.tensor([100.0]), torch.tensor([1.0, 0.0])]
    predictive_updates = [torch.tensor([0.0]), torch.tensor([-30.0, -40.0])]
    actor_gradients = [torch.tensor([0.0]), torch.tensor([1.0, 0.0])]
    critic_gradients = [torch.tensor([0.0]), torch.tensor([0.0, 1.0])]

    admitted, stats = MODULE.admit_predictive_updates(
        task_updates,
        predictive_updates,
        max_ratio=0.05,
        actor_gradients=actor_gradients,
        critic_gradients=critic_gradients,
    )

    torch.testing.assert_close(admitted[1].norm(), torch.tensor(0.05))
    assert torch.dot(actor_gradients[1], admitted[1]) <= 0.0
    assert torch.dot(critic_gradients[1], admitted[1]) <= 0.0
    assert stats["max_block_ratio"] <= 0.05 + 1e-7


def test_boundary_roundoff_retains_the_valid_actor_boundary_projection():
    # This fixed FP32 case lands a few ulps on the positive side of the actor
    # boundary after subtraction. Exact ``dot <= 0`` used to reject it and return zero.
    raw = torch.tensor(
        [-0.2298103422, -0.0073140212, -0.1305188388, 1.3700692654, -0.1109791920]
    )
    actor = torch.tensor(
        [-0.7281495929, 1.032345891, -0.5819520354, 0.3008017242, 0.1308227628]
    )
    critic = torch.tensor(
        [-2.271158934, -0.0109918527, 0.0613946542, -0.7550209761, 2.33305335]
    )

    projected = MODULE._project_block_to_joint_descent(raw, actor, critic)
    actor_tolerance = MODULE._projection_dot_tolerance(
        actor.square().sum(), projected.square().sum()
    )
    critic_tolerance = MODULE._projection_dot_tolerance(
        critic.square().sum(), projected.square().sum()
    )

    assert projected.norm().item() > 1.3
    assert torch.dot(actor, projected) > 0.0
    assert torch.dot(actor, projected) <= actor_tolerance
    assert torch.dot(critic, projected) <= critic_tolerance


def test_near_opposed_constraints_preserve_the_unconstrained_coordinate():
    for dtype, deltas in (
        (torch.float32, (1e-4, 3e-4)),
        (torch.float64, (1e-9, 1e-8)),
    ):
        proposal = torch.tensor([0.0, 1.0, 1.0], dtype=dtype)
        actor = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
        for delta in deltas:
            critic = torch.tensor([-1.0, delta, 0.0], dtype=dtype)
            projected = MODULE._project_block_to_joint_descent(
                proposal, actor, critic
            )
            torch.testing.assert_close(
                projected,
                torch.tensor([0.0, 0.0, 1.0], dtype=dtype),
                rtol=1e-4 if dtype == torch.float32 else 1e-7,
                atol=1e-6 if dtype == torch.float32 else 1e-10,
            )


def _explicit_two_halfspace_projection(raw, actor, critic):
    """Enumerate the convex projection's active sets in FP64."""
    raw = raw.to(torch.float64)
    actor = actor.to(torch.float64)
    critic = critic.to(torch.float64)
    actor_sq = actor.square().sum()
    critic_sq = critic.square().sum()
    actor_critic = torch.dot(actor, critic)
    actor_raw = torch.dot(actor, raw)
    critic_raw = torch.dot(critic, raw)

    def tolerance(gradient, candidate):
        return 1024.0 * torch.finfo(torch.float64).eps * max(
            1.0, gradient.norm().item() * candidate.norm().item()
        )

    def feasible(candidate):
        return (
            torch.dot(actor, candidate) <= tolerance(actor, candidate)
            and torch.dot(critic, candidate) <= tolerance(critic, candidate)
        )

    candidates = [torch.zeros_like(raw)]
    if feasible(raw):
        candidates.append(raw)
    if actor_sq > 0.0:
        actor_only = raw - actor_raw.clamp_min(0.0) / actor_sq * actor
        if feasible(actor_only):
            candidates.append(actor_only)
    if critic_sq > 0.0:
        critic_only = raw - critic_raw.clamp_min(0.0) / critic_sq * critic
        if feasible(critic_only):
            candidates.append(critic_only)

    determinant = actor_sq * critic_sq - actor_critic.square()
    if determinant > 1e-14 * actor_sq * critic_sq:
        actor_multiplier = (
            actor_raw * critic_sq - critic_raw * actor_critic
        ) / determinant
        critic_multiplier = (
            critic_raw * actor_sq - actor_raw * actor_critic
        ) / determinant
        if actor_multiplier >= -1e-11 and critic_multiplier >= -1e-11:
            joint = raw - actor_multiplier * actor - critic_multiplier * critic
            if feasible(joint):
                candidates.append(joint)

    return min(candidates, key=lambda candidate: (candidate - raw).square().sum())


def test_random_projection_matches_convex_oracle_across_dtypes_and_correlations():
    generator = torch.Generator().manual_seed(7183)
    sample_count, dimension = 96, 7
    for dtype in (torch.float32, torch.float64):
        actor = torch.randn(sample_count, dimension, generator=generator, dtype=dtype)
        actor = actor / actor.norm(dim=-1, keepdim=True)
        orthogonal = torch.randn(
            sample_count, dimension, generator=generator, dtype=dtype
        )
        orthogonal = orthogonal - (orthogonal * actor).sum(-1, keepdim=True) * actor
        orthogonal = orthogonal / orthogonal.norm(dim=-1, keepdim=True)
        raw = torch.randn(sample_count, dimension, generator=generator, dtype=dtype)
        near_delta = 2e-3 if dtype == torch.float32 else 1e-4
        gradient_pairs = (
            torch.randn(sample_count, dimension, generator=generator, dtype=dtype),
            0.95 * actor + 0.3122499 * orthogonal,
            actor + near_delta * orthogonal,
            -actor + near_delta * orthogonal,
        )
        row = torch.arange(sample_count).unsqueeze(-1)
        actor_scale = torch.where(row.remainder(2) == 0, 1e4, 1e-4).to(dtype)
        critic_scale = actor_scale.reciprocal()
        scaled_actor = actor * actor_scale
        eps = torch.finfo(dtype).eps
        ill_conditioned_slack = 3e-4 if dtype == torch.float32 else 3e-8

        for case_index, unscaled_critic in enumerate(gradient_pairs):
            cosine = torch.nn.functional.cosine_similarity(actor, unscaled_critic)
            if case_index >= 2:
                assert cosine.abs().min() > 0.99999
            critic = unscaled_critic * critic_scale

            for index in range(sample_count):
                result = MODULE._project_block_to_joint_descent(
                    raw[index], scaled_actor[index], critic[index]
                )
                oracle = _explicit_two_halfspace_projection(
                    raw[index], scaled_actor[index], critic[index]
                )
                result64 = result.to(torch.float64)
                raw64 = raw[index].to(torch.float64)
                result_distance = (result64 - raw64).square().sum()
                oracle_distance = (oracle - raw64).square().sum()
                distance_tolerance = max(8192.0 * eps, ill_conditioned_slack) * max(
                    1.0, raw64.square().sum().item()
                )
                actor_tolerance = MODULE._projection_dot_tolerance(
                    scaled_actor[index].square().sum(), result.square().sum()
                )
                critic_tolerance = MODULE._projection_dot_tolerance(
                    critic[index].square().sum(), result.square().sum()
                )

                assert result_distance <= oracle_distance + distance_tolerance
                assert torch.dot(scaled_actor[index], result) <= actor_tolerance
                assert torch.dot(critic[index], result) <= critic_tolerance


def test_nonfinite_predictive_transaction_restores_every_parameter_exactly():
    trunk = torch.nn.Parameter(torch.tensor([0.25, -0.75]))
    predictor = torch.nn.Parameter(torch.tensor([1.5, -2.0]))
    optimizer = torch.optim.Adam(
        [trunk, predictor], lr=0.1, betas=(0.0, 0.0), eps=1e-8
    )
    trunk_before = trunk.detach().clone()
    predictor_before = predictor.detach().clone()

    raw, admitted, predictor_norm, stats = (
        MODULE.apply_predictive_optimizer_transaction(
            [[trunk]],
            [predictor],
            optimizer,
            [torch.tensor([float("nan"), 1.0]), torch.tensor([1.0, -1.0])],
            [torch.tensor([0.2, -0.1])],
            0.05,
            actor_gradients=[torch.tensor([1.0, 0.0])],
            critic_gradients=[torch.tensor([0.0, 1.0])],
        )
    )

    assert torch.equal(trunk, trunk_before)
    assert torch.equal(predictor, predictor_before)
    assert torch.equal(raw[0], torch.zeros_like(raw[0]))
    assert torch.equal(admitted[0], torch.zeros_like(admitted[0]))
    assert predictor_norm.item() == 0.0
    assert stats["numeric_valid"].item() == 0.0
    for parameter in (trunk, predictor):
        assert optimizer.state[parameter]["step"].item() == 1
        assert all(
            torch.isfinite(value).all()
            for value in optimizer.state[parameter].values()
            if isinstance(value, torch.Tensor)
        )


def test_async_grad_clip_matches_torch_and_fails_loudly_on_nonfinite():
    actual = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    expected = torch.nn.Parameter(actual.detach().clone())
    actual.grad = torch.tensor([6.0, 8.0])
    expected.grad = actual.grad.clone()

    actual_norm = MODULE.clip_grad_norm_async_([actual], 0.25, "test")
    expected_norm = torch.nn.utils.clip_grad_norm_([expected], 0.25)
    torch.testing.assert_close(actual_norm, expected_norm)
    torch.testing.assert_close(actual.grad, expected.grad)

    invalid = torch.nn.Parameter(torch.ones(2))
    invalid.grad = torch.tensor([float("nan"), 1.0])
    with pytest.raises(RuntimeError, match="non-finite test gradient norm"):
        MODULE.clip_grad_norm_async_([invalid], 0.25, "test")


def test_finite_gradient_whose_square_overflows_cannot_poison_adam_state():
    trunk = torch.nn.Parameter(torch.tensor([0.25, -0.75]))
    predictor = torch.nn.Parameter(torch.tensor([1.5, -2.0]))
    optimizer = torch.optim.Adam([trunk, predictor], lr=0.1)
    trunk_before = trunk.detach().clone()
    predictor_before = predictor.detach().clone()
    huge = torch.tensor([1e30, -1e30])
    assert torch.isfinite(huge).all()
    assert not torch.isfinite(huge.square()).all()

    raw, admitted, predictor_norm, stats = (
        MODULE.apply_predictive_optimizer_transaction(
            [[trunk]],
            [predictor],
            optimizer,
            [huge, -huge],
            [torch.tensor([0.2, -0.1])],
            0.05,
            actor_gradients=[torch.tensor([1.0, 0.0])],
            critic_gradients=[torch.tensor([0.0, 1.0])],
        )
    )

    assert torch.equal(trunk, trunk_before)
    assert torch.equal(predictor, predictor_before)
    assert torch.equal(raw[0], torch.zeros_like(raw[0]))
    assert torch.equal(admitted[0], torch.zeros_like(admitted[0]))
    assert predictor_norm.item() == 0.0
    assert stats["numeric_valid"].item() == 0.0
    for parameter in (trunk, predictor):
        assert optimizer.state[parameter]["step"].item() == 1
        assert all(
            torch.isfinite(value).all()
            for value in optimizer.state[parameter].values()
            if isinstance(value, torch.Tensor)
        )


def test_flat_block_views_round_trip_without_parameter_rebinding():
    first = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    second = torch.nn.Parameter(torch.tensor([-1.0, -2.0, -3.0]))
    original_storage = (first.data_ptr(), second.data_ptr())
    flat = MODULE._flatten_tensors([first, second]) + 10.0
    with torch.no_grad():
        torch._foreach_copy_([first, second], MODULE._flat_views(flat, [first, second]))
    torch.testing.assert_close(first, torch.tensor([[11.0, 12.0], [13.0, 14.0]]))
    torch.testing.assert_close(second, torch.tensor([9.0, 8.0, 7.0]))
    assert (first.data_ptr(), second.data_ptr()) == original_storage


def test_residual_dynamics_starts_as_exact_identity():
    predictor = MODULE.ResidualDynamics(hidden=13, action_dim=4)
    latent = torch.randn(19, 13)
    action = torch.randn(19, 4)
    prediction, linear_delta = predictor(latent, action)
    torch.testing.assert_close(prediction, latent)
    torch.testing.assert_close(linear_delta, torch.zeros_like(linear_delta))


def test_auxiliary_gradient_reaches_only_live_trunk_and_predictor():
    torch.manual_seed(73)
    args = MODULE.Args()
    agent = MODULE.Agent(_FakeEnv(), args)
    with torch.no_grad():
        agent.critic_head.weight.normal_(mean=0.0, std=0.05)
    target = MODULE.FrozenValueTarget(agent)

    observations = torch.randn(32, 17)
    actions = torch.randn(32, 6).clamp(-1.0, 1.0)
    source_latent = agent.get_actor_feat(observations)
    imagined, _ = agent.nextlat_predictor(source_latent, actions)
    with torch.no_grad():
        target_latent = target.encode(torch.randn(32, 17))
    logits = target.decode_imagined(imagined)
    labels = torch.softmax(torch.randn_like(logits), dim=-1)
    bellman_loss = -(labels * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
    latent_loss = (imagined - target_latent).square().mean()
    (bellman_loss + 0.1 * latent_loss).backward()

    trunk_parameters = list(agent.trunk.parameters())
    predictor_parameters = list(agent.nextlat_predictor.parameters())
    assert any(parameter.grad is not None for parameter in trunk_parameters)
    assert any(parameter.grad is not None for parameter in predictor_parameters)
    assert all(parameter.grad is None for parameter in agent.critic_head.parameters())
    assert all(
        parameter.grad is None
        for parameter in list(agent.actor_alpha_head.parameters())
        + list(agent.actor_beta_head.parameters())
    )
    assert all(parameter.grad is None for parameter in target.parameters())
    assert not any(parameter.requires_grad for parameter in target.parameters())

    auxiliary_ids = {id(parameter) for parameter in agent.nextlat_parameters()}
    expected_ids = {
        id(parameter) for parameter in trunk_parameters + predictor_parameters
    }
    assert auxiliary_ids == expected_ids
    block_ids = [
        id(parameter)
        for block in agent.nextlat_trunk_parameter_blocks()
        for parameter in block
    ]
    assert block_ids == [id(parameter) for parameter in trunk_parameters]

    frozen_weight = target.decoder.weight.detach().clone()
    with torch.no_grad():
        agent.critic_head.weight.add_(1.0)
    torch.testing.assert_close(target.decoder.weight, frozen_weight)
    target.sync_from(agent)
    torch.testing.assert_close(target.decoder.weight, agent.critic_head.weight)
    assert not any(parameter.requires_grad for parameter in target.parameters())
    probe = torch.randn(7, 17)
    frozen_probabilities = target.value_probabilities(probe)
    live_probabilities = torch.softmax(agent.get_value(probe)[:, 0], dim=-1)
    torch.testing.assert_close(frozen_probabilities, live_probabilities)
    arbitrary_support = torch.linspace(-3.0, 5.0, args.num_bins)
    torch.testing.assert_close(
        (frozen_probabilities * arbitrary_support).sum(-1),
        (live_probabilities * arbitrary_support).sum(-1),
    )
