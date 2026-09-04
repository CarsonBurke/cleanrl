import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

from cleanrl.recurrent_search.grounded_recurrent_search_v1 import (
    Args,
    DistinctRootPlanner,
    EnsembleLinear,
    GPUVectorReplay,
    MCValueEnsemble,
    Proposal,
    WorldEnsemble,
    evaluate_search,
    factual_next_observations,
    gaussian_nll,
    is_evaluation_step,
    map_normalized_action,
    proposal_elite_nll,
    update_episode_members,
    value_mse_loss,
    world_nll_loss,
)


def tiny_args(**overrides):
    values = dict(
        ensemble_size=2,
        hidden_dim=16,
        bottleneck_dim=4,
        world_depth=1,
        value_depth=1,
        proposal_depth=1,
        root_candidates=4,
        beam_width=2,
        branch_factor=2,
        search_depth=3,
        elite_roots=2,
    )
    values.update(overrides)
    return Args(**values)


def replay(elite_roots=2, capacity=32, num_envs=1, ensemble_size=2, seed=7):
    return GPUVectorReplay(
        total_capacity=capacity,
        num_envs=num_envs,
        obs_dim=2,
        action_dim=1,
        ensemble_size=ensemble_size,
        elite_roots=elite_roots,
        max_episode_steps=10,
        device="cpu",
        seed=seed,
        heldout_fraction=0.2,
    )


def add_step(buffer, reward, *, terminated=False, truncated=False, member=0, obs=0.0):
    buffer.add(
        obs=np.array([[obs, -obs]], np.float32),
        action=np.array([[0.25]], np.float32),
        reward=np.array([reward], np.float32),
        next_obs=np.array([[obs + 1.0, 1.0 - obs]], np.float32),
        terminated=np.array([terminated]),
        truncated=np.array([truncated]),
        remaining=np.array([1.0 - obs / 10.0], np.float32),
        member=np.array([member]),
        elite_pre_tanh=np.zeros((1, buffer.elite_pre_tanh.shape[2], 1), np.float32),
        predicted_score=np.array([reward + 0.5], np.float32),
        planner_valid=np.array([True]),
    )


def test_args_lock_the_full_benchmark_and_search_shape():
    args = Args()
    assert args.env_id == "HalfCheetah-v4"
    assert args.total_timesteps == 8_000_000
    assert args.seed == 1
    assert args.num_envs == 16
    assert args.ensemble_size == 4
    assert (args.root_candidates, args.beam_width, args.branch_factor) == (256, 4, 4)
    assert (args.search_depth, args.elite_roots) == (16, 8)
    assert args.model_learning_starts == 4_096
    assert args.planning_starts == 32_768
    assert (args.batch_size, args.updates_per_vector_step) == (256, 16)
    assert args.compile_mode == "reduce-overhead"


def test_factual_next_observation_replaces_both_terminal_kinds_only():
    autoreset = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], np.float32)
    infos = {
        "final_observation": np.array(
            [np.array([9.0, 8.0]), np.array([7.0, 6.0]), None], dtype=object
        ),
        "_final_observation": np.array([True, True, False]),
    }
    factual = factual_next_observations(
        autoreset,
        terminated=np.array([True, False, False]),
        truncated=np.array([False, True, False]),
        infos=infos,
    )
    np.testing.assert_array_equal(factual[0], [9.0, 8.0])
    np.testing.assert_array_equal(factual[1], [7.0, 6.0])
    np.testing.assert_array_equal(factual[2], autoreset[2])


def test_finished_transition_without_final_observation_is_rejected():
    with pytest.raises(RuntimeError, match="final_observation"):
        factual_next_observations(
            np.zeros((1, 2), np.float32),
            np.array([False]),
            np.array([True]),
            {},
        )


def test_action_map_uses_every_box_low_and_high_coordinate():
    normalized = torch.tensor([[-1.0, 0.0, 1.0], [0.5, -0.5, 0.25]])
    low = torch.tensor([-2.0, 1.0, -7.0])
    high = torch.tensor([4.0, 5.0, 9.0])
    mapped = map_normalized_action(normalized, low, high)
    torch.testing.assert_close(mapped[0], torch.tensor([-2.0, 3.0, 9.0]))
    torch.testing.assert_close(mapped[1], torch.tensor([2.5, 2.0, 3.0]))


def test_ring_chronology_survives_wrap():
    buffer = replay(capacity=4)
    for step in range(7):
        add_step(buffer, float(step), truncated=True, obs=float(step))
    torch.testing.assert_close(buffer.chronological(0, "reward"), torch.tensor([3.0, 4.0, 5.0, 6.0]))
    torch.testing.assert_close(
        buffer.chronological(0, "obs")[:, 0], torch.tensor([3.0, 4.0, 5.0, 6.0])
    )


def test_mc_suffix_returns_are_raw_exact_and_never_cross_reset():
    buffer = replay(capacity=16)
    add_step(buffer, 1.0, obs=0.0)
    add_step(buffer, -2.0, obs=1.0)
    add_step(buffer, 4.0, truncated=True, obs=2.0)
    add_step(buffer, 10.0, obs=0.0)
    add_step(buffer, 20.0, terminated=True, obs=1.0)

    torch.testing.assert_close(buffer.value_return[:3, 0], torch.tensor([3.0, 2.0, 4.0]))
    torch.testing.assert_close(buffer.value_return[3:5, 0], torch.tensor([30.0, 20.0]))
    assert buffer.value_valid[:5, 0].all()
    assert buffer.episode[0, 0] == buffer.episode[2, 0]
    assert buffer.episode[2, 0] != buffer.episode[3, 0]


def test_unfinished_episode_has_no_value_supervision():
    buffer = replay()
    add_step(buffer, 5.0)
    add_step(buffer, 7.0)
    assert not buffer.value_valid[:2, 0].any()


def test_member_specific_value_sampler_routes_each_episode_only_to_its_generator():
    buffer = replay(capacity=32, ensemble_size=2)
    add_step(buffer, 2.0, truncated=True, member=0)
    add_step(buffer, 9.0, truncated=True, member=1)
    buffer.holdout.zero_()
    generator = torch.Generator().manual_seed(3)
    batch = buffer.sample_value(128, generator)
    for model_index, expected in enumerate((2.0, 9.0)):
        mask = batch["supervision_mask"][model_index].bool()
        assert mask.any()
        torch.testing.assert_close(
            batch["value_return"][model_index, mask],
            torch.full((int(mask.sum()),), expected),
        )


def test_episode_poisson_weights_are_constant_within_episode():
    buffer = replay(capacity=32, ensemble_size=4)
    add_step(buffer, 1.0)
    add_step(buffer, 1.0)
    add_step(buffer, 1.0, truncated=True)
    first_episode = buffer.bootstrap_weight[:3, 0]
    torch.testing.assert_close(first_episode, first_episode[:1].expand_as(first_episode))
    assert torch.all(first_episode == first_episode.round())
    assert torch.all(first_episode >= 0)


def test_ensemble_linear_all_and_selected_paths_are_identical():
    torch.manual_seed(5)
    layer = EnsembleLinear(3, 4, 2)
    inputs = torch.randn(3, 7, 4)
    all_outputs = layer.forward_all(inputs)
    flat_inputs = inputs.flatten(0, 1)
    members = torch.arange(3).repeat_interleave(7)
    selected = layer.forward_selected(flat_inputs, members).view(3, 7, 2)
    torch.testing.assert_close(selected, all_outputs)


def test_selected_ensemble_gradient_reaches_only_selected_member():
    torch.manual_seed(6)
    layer = EnsembleLinear(3, 2, 1)
    output = layer.forward_selected(torch.ones(4, 2), torch.full((4,), 1, dtype=torch.long))
    output.square().sum().backward()
    assert torch.count_nonzero(layer.weight.grad[0]) == 0
    assert torch.count_nonzero(layer.weight.grad[1]) > 0
    assert torch.count_nonzero(layer.weight.grad[2]) == 0


def test_gaussian_nll_matches_manual_density():
    target = torch.tensor([2.0, -1.0], dtype=torch.float64)
    mean = torch.tensor([0.5, -0.5], dtype=torch.float64)
    std = torch.tensor([2.0, 0.25], dtype=torch.float64)
    expected = torch.log(std) + (target - mean).square() / (2.0 * std.square()) + 0.5 * math.log(
        2.0 * math.pi
    )
    torch.testing.assert_close(gaussian_nll(target, mean, std), expected)


def test_world_loss_is_joint_nll_and_routes_every_head_gradient():
    torch.manual_seed(7)
    args = tiny_args()
    world = WorldEnsemble(3, 2, args)
    shape = (args.ensemble_size, 5)
    obs = torch.randn(shape + (3,))
    next_obs = obs + torch.randn_like(obs)
    action = torch.randn(shape + (2,))
    reward = torch.randn(shape)
    terminated = torch.randint(2, shape).float()
    weights = torch.ones(shape)
    total, obs_nll, reward_nll, term_bce, obs_floor, reward_floor = world_nll_loss(
        world, obs, action, next_obs, reward, terminated, weights
    )
    torch.testing.assert_close(total, obs_nll + reward_nll + term_bce)
    total.backward()
    assert world.network.head.weight.grad is not None
    assert torch.count_nonzero(world.network.head.weight.grad) > 0
    decoded = world.predict_all(obs, action)
    assert decoded[0].shape == next_obs.shape
    assert decoded[2].shape == reward.shape
    assert torch.all(decoded[1] > args.std_floor)
    assert torch.all(decoded[3] > args.std_floor)
    assert 0 <= obs_floor <= 1
    assert 0 <= reward_floor <= 1


def test_zero_poisson_batch_is_finite_and_has_zero_data_gradient():
    torch.manual_seed(71)
    args = tiny_args()
    world = WorldEnsemble(2, 1, args)
    shape = (args.ensemble_size, 4)
    loss = world_nll_loss(
        world,
        torch.randn(shape + (2,)),
        torch.randn(shape + (1,)),
        torch.randn(shape + (2,)),
        torch.randn(shape),
        torch.zeros(shape),
        torch.zeros(shape),
    )[0]
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in world.parameters()
    )


def test_world_physics_has_no_episode_clock_input():
    args = tiny_args()
    world = WorldEnsemble(3, 2, args)
    assert world.network.stem.in_features == 5
    obs = torch.randn(args.ensemble_size, 4, 3)
    action = torch.randn(args.ensemble_size, 4, 2)
    first = world.predict_all(obs, action)
    second = world.predict_all(obs, action)
    for left, right in zip(first, second, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_value_rate_is_exact_fixed_reparameterization_of_raw_return():
    class FixedValue:
        def predict_all(self, obs, remaining):
            return torch.tensor([[0.25, -0.5]], dtype=obs.dtype)

    target_return = torch.tensor([[25.0, -50.0]])
    loss, rate = value_mse_loss(
        FixedValue(),
        torch.zeros(1, 2, 1),
        torch.ones(1, 2),
        target_return,
        torch.ones_like(target_return),
        100,
    )
    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(100 * rate, target_return)


def test_value_mask_excludes_non_generating_rows_exactly():
    class FixedValue:
        def predict_all(self, obs, remaining):
            return torch.tensor([[1.0, 100.0]], dtype=obs.dtype)

    loss, _ = value_mse_loss(
        FixedValue(),
        torch.zeros(1, 2, 1),
        torch.ones(1, 2),
        torch.tensor([[10.0, float("nan")]]),
        torch.tensor([[1.0, 0.0]]),
        10,
    )
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.tensor(0.0))


class ConstantProposal(nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor([mean], dtype=torch.float32))
        self.std = nn.Parameter(torch.tensor([std], dtype=torch.float32))
        self.std_floor = 1e-4

    def forward(self, obs, remaining, members):
        shape = obs.shape[:-1] + (1,)
        return self.mean.expand(shape), self.std.expand(shape)


def test_uniform_elite_nll_has_empirical_gaussian_optimum():
    elite = torch.tensor([[[-2.0], [0.0], [2.0], [0.0]]])
    optimum_std = elite.std(dim=1, correction=0).item()
    proposal = ConstantProposal(mean=0.0, std=optimum_std)
    loss, _, _, _, _ = proposal_elite_nll(
        proposal, torch.zeros(1, 1), torch.ones(1), torch.zeros(1, dtype=torch.long), elite
    )
    loss.backward()
    torch.testing.assert_close(proposal.mean.grad, torch.zeros_like(proposal.mean), atol=1e-6, rtol=0)
    torch.testing.assert_close(proposal.std.grad, torch.zeros_like(proposal.std), atol=1e-6, rtol=0)


class ToyWorld(nn.Module):
    def __init__(self, reward_mode="action", termination_logit=-100.0):
        super().__init__()
        self.reward_mode = reward_mode
        self.termination_logit = termination_logit

    def predict_selected(self, obs, action, members):
        delta = torch.zeros_like(obs)
        std = torch.ones_like(obs)
        if self.reward_mode == "action":
            reward = action[..., 0]
        elif self.reward_mode == "constant":
            reward = torch.full_like(action[..., 0], 2.0)
        else:
            reward = -(action[..., 0] - obs[..., 0]).square()
        reward_std = torch.ones_like(reward)
        termination = torch.full_like(reward, self.termination_logit)
        return delta, std, reward, reward_std, termination


class ToyValue(nn.Module):
    def __init__(self, rate=0.0):
        super().__init__()
        self.rate = rate

    def predict_selected(self, obs, remaining, members):
        return torch.full_like(remaining, self.rate)


class ToyProposal(nn.Module):
    def __init__(self, std=1.0):
        super().__init__()
        self.std = std

    def forward(self, obs, remaining, members):
        return torch.zeros(obs.shape[:-1] + (1,)), torch.full(
            obs.shape[:-1] + (1,), self.std, dtype=obs.dtype, device=obs.device
        )


def make_toy_planner(
    *, roots=4, beam=2, branch=2, depth=3, elites=2, reward_mode="action", term=-100.0, rate=0.0
):
    return DistinctRootPlanner(
        ToyWorld(reward_mode, term),
        ToyValue(rate),
        ToyProposal(),
        max_episode_steps=10,
        root_candidates=roots,
        beam_width=beam,
        branch_factor=branch,
        depth=depth,
        elite_roots=elites,
    )


def test_planner_includes_reward_before_survival_and_zeros_tail_at_time_limit():
    planner = make_toy_planner(roots=1, beam=1, branch=1, depth=2, elites=1, reward_mode="constant", term=0.0, rate=0.7)
    zeros_root = torch.zeros(1, 1, 1)
    zeros_branch = torch.zeros(1, 1, 1, 1, 1, 1)
    full = planner(torch.zeros(1, 1), torch.ones(1), torch.zeros(1, dtype=torch.long), zeros_root, zeros_branch)
    # 2 now + 0.5*2 next + 0.25*(10*0.7) at the leaf.
    torch.testing.assert_close(full[2], torch.tensor([4.75]))
    final_step = planner(
        torch.zeros(1, 1),
        torch.tensor([0.1]),
        torch.zeros(1, dtype=torch.long),
        zeros_root,
        zeros_branch,
    )
    torch.testing.assert_close(final_step[2], torch.tensor([2.0]))


def test_search_executes_a_root_that_beats_the_proposal_mean():
    planner = make_toy_planner(roots=4, beam=1, branch=1, depth=1, elites=2)
    root_noise = torch.tensor([[[-2.0], [-0.5], [0.5], [2.0]]])
    branch_noise = torch.empty(1, 0, 4, 1, 1, 1)
    result = planner(
        torch.zeros(1, 1), torch.ones(1), torch.zeros(1, dtype=torch.long), root_noise, branch_noise
    )
    assert result[0].item() > 0.9
    assert result[2].item() > result[3].item()


def test_root_score_decomposition_and_survival_are_exposed_without_changing_ranking():
    planner = make_toy_planner(roots=3, beam=2, branch=2, depth=3, elites=2, term=0.0, rate=0.4)
    root_noise = torch.tensor([[[-1.0], [0.0], [1.0]]])
    branch_noise = torch.zeros(1, 2, 3, 2, 2, 1)
    result = planner(
        torch.zeros(1, 1), torch.ones(1), torch.zeros(1, dtype=torch.long), root_noise, branch_noise
    )
    root_score, prefix, tail = result[16], result[17], result[18]
    torch.testing.assert_close(root_score, prefix + tail)
    torch.testing.assert_close(result[12], torch.tensor([0.125]))
    assert 0 <= result[8].item() <= 1
    assert 0 <= result[11].item() <= 1


def slow_distinct_root_scores(planner, obs, remaining, member, root_noise, branch_noise):
    root_mean, root_std = planner.proposal(obs, remaining, member)
    root_pre = root_mean[:, None] + root_std[:, None] * root_noise
    results = []
    for root_index in range(planner.root_candidates):
        action = torch.tanh(root_pre[:, root_index])
        state, time_left, reward, continuation = planner._step(obs, remaining, action, member)
        beams = [(state, time_left, reward, continuation)]
        for depth_index in range(1, planner.depth):
            candidates = []
            for beam_index, (state, time_left, cumulative, survival) in enumerate(beams):
                mean, std = planner.proposal(state, time_left, member)
                for branch_index in range(planner.branch_factor):
                    noise = branch_noise[
                        :, depth_index - 1, root_index, beam_index, branch_index
                    ]
                    action = torch.tanh(mean + std * noise)
                    child_state, child_time, child_reward, child_continue = planner._step(
                        state, time_left, action, member
                    )
                    child_cumulative = cumulative + survival * child_reward
                    child_survival = survival * child_continue
                    child_score = child_cumulative + child_survival * planner.max_episode_steps * planner.value.predict_selected(
                        child_state, child_time, member
                    )
                    candidates.append(
                        (child_score, child_state, child_time, child_cumulative, child_survival)
                    )
            score_tensor = torch.stack([entry[0] for entry in candidates], dim=1)
            _, best = torch.topk(
                score_tensor, min(planner.beam_width, len(candidates)), dim=1, sorted=False
            )
            beams = [
                (
                    torch.cat([candidates[int(i)][1][row : row + 1] for row, i in enumerate(index)]),
                    torch.cat([candidates[int(i)][2][row : row + 1] for row, i in enumerate(index)]),
                    torch.cat([candidates[int(i)][3][row : row + 1] for row, i in enumerate(index)]),
                    torch.cat([candidates[int(i)][4][row : row + 1] for row, i in enumerate(index)]),
                )
                for index in best.T
            ]
        scores = [
            cumulative
            + survival
            * planner.max_episode_steps
            * planner.value.predict_selected(state, time_left, member)
            for state, time_left, cumulative, survival in beams
        ]
        results.append(torch.stack(scores, dim=1).max(dim=1).values)
    return torch.stack(results, dim=1)


def test_vectorized_per_root_beam_matches_slow_exhaustive_oracle():
    torch.manual_seed(13)
    planner = make_toy_planner(roots=3, beam=2, branch=2, depth=3, elites=2, reward_mode="state")
    obs = torch.tensor([[0.4]])
    remaining = torch.ones(1)
    member = torch.zeros(1, dtype=torch.long)
    root_noise = torch.randn(1, 3, 1)
    branch_noise = torch.randn(1, 2, 3, 2, 2, 1)
    vectorized = planner(obs, remaining, member, root_noise, branch_noise)[16]
    slow = slow_distinct_root_scores(planner, obs, remaining, member, root_noise, branch_noise)
    torch.testing.assert_close(vectorized, slow)


def test_member_persists_until_done_and_behavior_rng_is_training_rng_independent():
    behavior = torch.Generator().manual_seed(17)
    control = torch.Generator().manual_seed(17)
    training = torch.Generator().manual_seed(99)
    members = torch.tensor([0, 1, 0, 1])
    unchanged = update_episode_members(
        members, torch.zeros(4, dtype=torch.bool), behavior, 2
    )
    control_unchanged = update_episode_members(
        members, torch.zeros(4, dtype=torch.bool), control, 2
    )
    torch.testing.assert_close(unchanged, members)
    torch.testing.assert_close(unchanged, control_unchanged)
    _ = torch.randn(10_000, generator=training)
    done = torch.tensor([True, False, True, False])
    after_training = update_episode_members(members, done, behavior, 2)
    control_after = update_episode_members(members, done, control, 2)
    torch.testing.assert_close(after_training, control_after)
    torch.testing.assert_close(after_training[~done], members[~done])


def test_planner_eager_and_compiled_fullgraph_match():
    torch.manual_seed(19)
    args = tiny_args(root_candidates=3, beam_width=2, branch_factor=2, search_depth=2, elite_roots=2)
    world = WorldEnsemble(2, 1, args)
    value = MCValueEnsemble(2, args)
    proposal = Proposal(2, 1, args)
    planner = DistinctRootPlanner(world, value, proposal, 10, 3, 2, 2, 2, 2)
    obs = torch.randn(2, 2)
    remaining = torch.tensor([1.0, 0.6])
    members = torch.tensor([0, 1])
    root_noise = torch.randn(2, 3, 1)
    branch_noise = torch.randn(2, 1, 3, 2, 2, 1)
    eager = planner(obs, remaining, members, root_noise, branch_noise)
    compiled = torch.compile(planner, backend="eager", dynamic=False, fullgraph=True)
    actual = compiled(obs, remaining, members, root_noise, branch_noise)
    for eager_tensor, actual_tensor in zip(eager, actual, strict=True):
        torch.testing.assert_close(eager_tensor, actual_tensor)


def test_all_three_static_losses_match_compiled_fullgraph():
    torch.manual_seed(191)
    args = tiny_args()
    world = WorldEnsemble(2, 1, args)
    value = MCValueEnsemble(2, args)
    proposal = Proposal(2, 1, args)
    member_batch = (args.ensemble_size, 4)
    obs = torch.randn(member_batch + (2,))
    action = torch.randn(member_batch + (1,))
    next_obs = torch.randn(member_batch + (2,))
    reward = torch.randn(member_batch)
    terminated = torch.zeros(member_batch)
    weights = torch.ones(member_batch)
    compiled_world = torch.compile(world_nll_loss, backend="eager", dynamic=False, fullgraph=True)
    eager_world = world_nll_loss(world, obs, action, next_obs, reward, terminated, weights)
    actual_world = compiled_world(world, obs, action, next_obs, reward, terminated, weights)
    for eager_tensor, actual_tensor in zip(eager_world, actual_world, strict=True):
        torch.testing.assert_close(eager_tensor, actual_tensor)

    remaining = torch.rand(member_batch)
    factual_return = torch.randn(member_batch)
    mask = torch.ones(member_batch)
    compiled_value = torch.compile(value_mse_loss, backend="eager", dynamic=False, fullgraph=True)
    eager_value = value_mse_loss(value, obs, remaining, factual_return, mask, 10)
    actual_value = compiled_value(value, obs, remaining, factual_return, mask, 10)
    for eager_tensor, actual_tensor in zip(eager_value, actual_value, strict=True):
        torch.testing.assert_close(eager_tensor, actual_tensor)

    flat_obs = torch.randn(4, 2)
    flat_remaining = torch.rand(4)
    members = torch.tensor([0, 1, 0, 1])
    elites = torch.randn(4, 2, 1)
    compiled_proposal = torch.compile(
        proposal_elite_nll, backend="eager", dynamic=False, fullgraph=True
    )
    eager_proposal = proposal_elite_nll(
        proposal, flat_obs, flat_remaining, members, elites
    )
    actual_proposal = compiled_proposal(
        proposal, flat_obs, flat_remaining, members, elites
    )
    for eager_tensor, actual_tensor in zip(eager_proposal, actual_proposal, strict=True):
        torch.testing.assert_close(eager_tensor, actual_tensor)


class TinyEvalVectorEnv:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.steps = np.zeros(num_envs, dtype=np.int64)

    def reset(self, seed=None):
        self.steps.fill(0)
        return np.zeros((self.num_envs, 1), np.float32), {}

    def step(self, action):
        self.steps += 1
        reward = np.asarray(action)[:, 0].astype(np.float64)
        terminated = self.steps == 2
        truncated = np.zeros(self.num_envs, dtype=bool)
        observation = self.steps[:, None].astype(np.float32)
        return observation, reward, terminated, truncated, {}


def test_deterministic_evaluation_runs_search_and_reseeds_its_own_rng():
    args = tiny_args(num_envs=2, root_candidates=4, beam_width=1, branch_factor=1, search_depth=1, elite_roots=2)
    planner = make_toy_planner(roots=4, beam=1, branch=1, depth=1, elites=2)
    first = evaluate_search(
        planner,
        TinyEvalVectorEnv(2),
        args,
        obs_dim=1,
        action_dim=1,
        max_episode_steps=2,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        device=torch.device("cpu"),
    )
    second = evaluate_search(
        planner,
        TinyEvalVectorEnv(2),
        args,
        obs_dim=1,
        action_dim=1,
        max_episode_steps=2,
        action_low=torch.tensor([-1.0]),
        action_high=torch.tensor([1.0]),
        device=torch.device("cpu"),
    )
    np.testing.assert_array_equal(first, second)
    assert np.all(first > 0)


def test_evaluation_schedule_has_early_gates_then_one_million_cadence():
    assert is_evaluation_step(100_000)
    assert is_evaluation_step(250_000)
    assert is_evaluation_step(500_000)
    assert is_evaluation_step(1_000_000)
    assert is_evaluation_step(8_000_000)
    assert not is_evaluation_step(750_000)
    assert not is_evaluation_step(1_500_000)


def test_source_has_no_shadow_teacher_or_self_latent_objective():
    source_path = pathlib.Path(__file__).parents[1] / "cleanrl/recurrent_search/grounded_recurrent_search_v1.py"
    source = source_path.read_text().lower()
    assert "copy.deepcopy" not in source
    assert "target_network" not in source
    assert "target_model" not in source
    assert "polyak" not in source
    assert "exponential_moving_average" not in source
    assert "latent_loss" not in source
    assert "bootstrap_observation" not in source
    assert "normalizeobservation" not in source
    assert "value_replay_window" not in source
    assert "world_nll_loss" in source
    assert "value_mse_loss" in source
    assert "root_candidates: int = 256" in source
    assert "error_if_nonfinite=true" in source
    assert "torch._assert_async" in source


def test_script_pycompiles_and_cli_help_does_not_initialize_cuda():
    source_path = pathlib.Path(__file__).parents[1] / "cleanrl/recurrent_search/grounded_recurrent_search_v1.py"
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    help_result = subprocess.run(
        [sys.executable, str(source_path), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--total-timesteps" in help_result.stdout
    assert "--compile" in help_result.stdout
