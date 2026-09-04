import ast
import importlib.util
import inspect
import random
import sys
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "cleanrl/tpo/md/ppo_continuous_action_tpomd_horizon_td_tape_v23.py"


def _load():
    spec = importlib.util.spec_from_file_location("tpomd_horizon_td_tape_v23", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TAPE = _load()


def _load_reference():
    path = ROOT / "cleanrl/tpo/md/ppo_continuous_action_tpomd_multiscale_nextlat_v20.py"
    spec = importlib.util.spec_from_file_location("tpomd_v20_reference_for_tape", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V20 = _load_reference()


class _DummyEnvs:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(7,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))


def _args(actor_dist="beta", tape=True):
    return TAPE.Args(
        actor_dist=actor_dist,
        hidden=8,
        k_blocks=2,
        n_experts=2,
        num_bins=7,
        tape=tape,
        tape_horizon=4,
        tape_horizon_embed_dim=3,
        tape_predictor_hidden=11,
    )


def test_defaults_are_one_dense_graph_free_intervention():
    args = TAPE.Args()
    assert args.env_id == "HalfCheetah-v4"
    assert args.total_timesteps == 8_000_000
    assert args.seed == 1 and args.cuda
    assert args.tape_horizon == 16
    assert args.tape_trunk_grad_clip == 0.025
    assert args.tape_predictor_grad_clip == 0.25
    assert args.compile_mode == "reduce-overhead"
    source = SCRIPT.read_text()
    assert "tape_gammas" not in source
    assert "grounded_cumulant" not in source


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_disabled_mode_is_exact_v20_off_task_initialization_rng_and_adam(actor_dist):
    common = dict(hidden=8, k_blocks=2, n_experts=2, num_bins=7, actor_dist=actor_dist)
    torch.manual_seed(491)
    reference = V20.Agent(_DummyEnvs(), V20.Args(pc_mode="off", **common))
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(491)
    candidate = TAPE.Agent(_DummyEnvs(), TAPE.Args(tape=False, **common))
    candidate_rng = torch.get_rng_state().clone()
    assert torch.equal(candidate_rng, reference_rng)
    assert tuple(candidate.state_dict()) == tuple(reference.state_dict())
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)
    assert not candidate.tape_parameters()
    assert len(candidate.task_parameters()) == len(list(candidate.parameters()))

    observations = torch.randn(6, 7)
    reference_outputs = V20.policy_model_forward(reference, observations)
    candidate_outputs = TAPE.policy_model_forward(candidate, observations)
    for actual, expected in zip(candidate_outputs, reference_outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reference_optimizer = torch.optim.Adam(reference.task_parameters(), lr=3e-4, eps=1e-5)
    candidate_optimizer = torch.optim.Adam(candidate.task_parameters(), lr=3e-4, eps=1e-5)
    sum(output.square().mean() for output in reference_outputs).backward()
    sum(output.square().mean() for output in candidate_outputs).backward()
    reference_optimizer.step()
    candidate_optimizer.step()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)


def test_exact_tape_target_oracle_and_cumulative_equivalence():
    eta = torch.tensor([[[2.0, -1.0]], [[5.0, 3.0]]])
    next_prediction = torch.tensor(
        [
            [[[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]]],
            [[[40.0, 4.0], [50.0, 5.0], [60.0, 6.0]]],
        ]
    )
    result = TAPE.build_horizon_td_tape_targets(
        eta,
        next_prediction.requires_grad_(),
        torch.tensor([[0.0], [1.0]]),
        torch.ones(2, 1),
    )
    expected = torch.tensor(
        [
            [[[2.0, -1.0], [10.0, 1.0], [20.0, 2.0]]],
            [[[5.0, 3.0], [0.0, 0.0], [0.0, 0.0]]],
        ]
    )
    torch.testing.assert_close(result.values, expected)
    torch.testing.assert_close(
        TAPE.tape_to_cumulative(result.values)[0, 0],
        torch.tensor([[2.0, -1.0], [12.0, 0.0], [32.0, 2.0]]),
    )
    assert not result.values.requires_grad
    assert result.masks.all()


def test_repeated_lower_shift_propagates_and_is_exactly_nilpotent():
    tape = torch.arange(1.0, 13.0).reshape(1, 4, 3)
    shifted = tape
    for power in range(1, 5):
        shifted = TAPE.lower_shift_tape(shifted)
        assert torch.count_nonzero(shifted[..., :power, :]) == 0
    assert torch.count_nonzero(shifted) == 0


def test_ordinary_sarsa_action_timing_and_boundary_truth_table():
    behavior = torch.arange(1.0, 9.0).reshape(4, 1, 2)
    sampled = torch.full_like(behavior, 99.0)
    boundaries = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
    terminals = torch.tensor([[0.0], [0.0], [0.0], [0.0]])
    valids = torch.ones(4, 1)
    actions, sampled_rows = TAPE.build_next_action_table(
        behavior, sampled, terminals, boundaries, valids
    )
    torch.testing.assert_close(actions[0], behavior[1])
    torch.testing.assert_close(actions[1], sampled[1])
    torch.testing.assert_close(actions[2], behavior[3])
    torch.testing.assert_close(actions[3], sampled[3])
    assert torch.equal(sampled_rows[:, 0], torch.tensor([False, True, False, True]))

    terminals[1] = 1.0
    valids[2] = 0.0
    poisoned = sampled.clone()
    poisoned[1] = float("nan")
    poisoned[2] = float("nan")
    actions, sampled_rows = TAPE.build_next_action_table(
        behavior, poisoned, terminals, boundaries, valids
    )
    assert torch.count_nonzero(actions[1]) == 0
    assert torch.count_nonzero(actions[2]) == 0
    assert torch.isfinite(actions).all()
    assert not sampled_rows[1, 0]


def test_terminal_truncation_rollout_edge_and_missing_final_target_truth_table():
    eta = torch.tensor([1.0, 2.0, 3.0, float("nan")]).view(4, 1, 1)
    next_prediction = torch.arange(1.0, 13.0).view(4, 1, 3, 1)
    terminals = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
    valids = torch.tensor([[1.0], [1.0], [1.0], [0.0]])
    result = TAPE.build_horizon_td_tape_targets(
        eta, next_prediction, terminals, valids
    )
    torch.testing.assert_close(result.values[0, 0, :, 0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(result.values[1, 0, :, 0], torch.tensor([2.0, 4.0, 5.0]))
    torch.testing.assert_close(result.values[2, 0, :, 0], torch.tensor([3.0, 7.0, 8.0]))
    assert torch.count_nonzero(result.values[3]) == 0
    assert not result.masks[3].any()
    assert torch.isfinite(result.values).all()


def test_invalid_poison_cannot_change_direct_mc_or_td_targets():
    eta = torch.arange(1.0, 6.0).view(5, 1, 1)
    valids = torch.tensor([[1.0], [1.0], [0.0], [1.0], [1.0]])
    boundaries = torch.zeros(5, 1)
    prediction = torch.randn(5, 1, 4, 1)
    baseline = TAPE.build_horizon_td_tape_targets(
        eta, prediction, torch.zeros(5, 1), valids
    )
    direct, mask = TAPE.build_direct_delayed_innovation_targets(
        eta, boundaries, valids, 4
    )
    eta[2] = float("nan")
    prediction[2] = float("nan")
    poisoned = TAPE.build_horizon_td_tape_targets(
        eta, prediction, torch.zeros(5, 1), valids
    )
    direct_poisoned, mask_poisoned = TAPE.build_direct_delayed_innovation_targets(
        eta, boundaries, valids, 4
    )
    torch.testing.assert_close(poisoned.values, baseline.values)
    torch.testing.assert_close(direct_poisoned, direct)
    assert torch.equal(mask_poisoned, mask)


def test_direct_delayed_innovation_oracle_stops_after_boundary():
    eta = torch.arange(1.0, 6.0).view(5, 1, 1)
    boundaries = torch.tensor([[0.0], [0.0], [1.0], [0.0], [0.0]])
    values, masks = TAPE.build_direct_delayed_innovation_targets(
        eta, boundaries, torch.ones(5, 1), 4
    )
    torch.testing.assert_close(
        values[:, 0, :, 0],
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0],
                [2.0, 3.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0],
                [4.0, 5.0, 0.0, 0.0],
                [5.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    assert torch.equal(
        masks[:, 0],
        torch.tensor(
            [
                [True, True, True, False],
                [True, True, False, False],
                [True, False, False, False],
                [True, True, False, False],
                [True, False, False, False],
            ]
        ),
    )


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_isolated_persistent_rng_preserves_main_stream(actor_dist):
    agent = TAPE.Agent(_DummyEnvs(), _args(actor_dist))
    features = torch.randn(9, 8)
    before = torch.get_rng_state().clone()
    first, state, device_state = TAPE.frozen_policy_sample_action_isolated(
        agent, features, seed=991
    )
    assert torch.equal(before, torch.get_rng_state())
    second, state2, device_state2 = TAPE.frozen_policy_sample_action_isolated(
        agent, features, seed=991, cpu_rng_state=state, device_rng_state=device_state
    )
    replay, replay_state, _ = TAPE.frozen_policy_sample_action_isolated(
        agent, features, seed=991
    )
    torch.testing.assert_close(first, replay, rtol=0, atol=0)
    assert torch.equal(state, replay_state)
    assert not torch.equal(state, state2)
    assert not torch.equal(first, second)
    assert device_state is device_state2 is None
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_hard_snapshot_is_exact_frozen_and_coherent(actor_dist):
    agent = TAPE.Agent(_DummyEnvs(), _args(actor_dist))
    observations = torch.randn(6, 7)
    with torch.no_grad():
        for parameter in agent.tape_predictor.parameters():
            parameter.add_(0.25)
        for parameter in agent.actor_parameters():
            parameter.add_(0.1)
    assert agent.target_snapshot_lag() > 0
    absolute_drift, relative_drift = TAPE.target_encoder_functional_drift(
        agent, observations
    )
    assert absolute_drift > 0 and relative_drift > 0
    agent.snapshot_tape_target()
    assert agent.target_snapshot_lag().item() == 0.0
    absolute_drift, relative_drift = TAPE.target_encoder_functional_drift(
        agent, observations
    )
    assert absolute_drift.item() == 0.0 and relative_drift.item() == 0.0
    assert all(
        not parameter.requires_grad
        for module in agent.target_bundle_modules()
        for parameter in module.parameters()
    )
    frozen = [p.clone() for m in agent.target_bundle_modules() for p in m.parameters()]
    optimizer = torch.optim.Adam(agent.task_parameters(), lr=1e-3)
    loss = agent.get_actor_feat(torch.randn(5, 7)).square().mean()
    loss.backward()
    optimizer.step()
    for before, after in zip(
        frozen,
        [p for m in agent.target_bundle_modules() for p in m.parameters()],
        strict=True,
    ):
        torch.testing.assert_close(before, after)


def test_shared_predictor_shape_zero_init_and_parameter_partitions():
    agent = TAPE.Agent(_DummyEnvs(), _args())
    prediction = agent.predict_tapes(torch.randn(6, 8), torch.randn(6, 3))
    assert prediction.shape == (6, 4, 8)
    assert torch.count_nonzero(prediction) == 0
    assert isinstance(agent.tape_predictor.horizon_embedding, torch.nn.Embedding)
    assert not isinstance(agent.tape_predictor, torch.nn.ModuleList)
    task = set(agent.task_parameters())
    predictor = set(agent.tape_predictor_parameters())
    trainable = {p for p in agent.parameters() if p.requires_grad}
    assert task.isdisjoint(predictor)
    assert task | predictor == trainable


def test_shared_predictor_compiles_fullgraph_with_static_output():
    agent = TAPE.Agent(_DummyEnvs(), _args())
    with torch.no_grad():
        agent.tape_predictor.output.weight.normal_(0, 0.01)
    features = torch.randn(6, 8)
    actions = torch.randn(6, 3)
    compiled = torch.compile(
        lambda feature, action: TAPE.tape_predictor_forward(
            agent, feature, action
        ),
        backend="inductor",
        dynamic=False,
        fullgraph=True,
    )
    expected = agent.predict_tapes(features, actions)
    actual = compiled(features, actions)
    torch.testing.assert_close(actual, expected)
    retained = TAPE.retain_graph_output(actual, compiled=True)
    actual.zero_()
    torch.testing.assert_close(retained, expected)


def test_common_scalar_loss_scale_equal_head_weighting_and_zero_mask():
    prediction = torch.zeros(2, 3, 2, requires_grad=True)
    target = torch.tensor(
        [[[1.0, -1.0], [10.0, 0.0], [100.0, 0.0]],
         [[3.0, -3.0], [20.0, 0.0], [200.0, 0.0]]]
    )
    mask = torch.tensor([[True, True, True], [True, False, False]])
    loss, scale, per_head = TAPE.horizon_td_tape_loss(
        prediction, target, mask, latent_rms=torch.tensor(2.0)
    )
    expected_scale = target[:, 0].square().mean().sqrt()
    torch.testing.assert_close(scale, expected_scale)
    manual = F.smooth_l1_loss(
        prediction / scale, target / scale, reduction="none"
    )
    expected = torch.stack(
        [manual[:, h][mask[:, h]].mean() if mask[:, h].any() else manual.new_zeros(())
         for h in range(3)]
    )
    torch.testing.assert_close(per_head, expected)
    torch.testing.assert_close(loss, expected.mean())
    empty_loss, empty_scale, empty_heads = TAPE.horizon_td_tape_loss(
        prediction, target, torch.zeros_like(mask), latent_rms=torch.tensor(2.0)
    )
    assert empty_loss == 0 and torch.count_nonzero(empty_heads) == 0
    torch.testing.assert_close(empty_scale, torch.tensor(0.002))

    poisoned_prediction = prediction.detach().clone()
    poisoned_target = target.clone()
    poisoned_prediction[~mask] = float("nan")
    poisoned_target[~mask] = float("nan")
    poison_loss, _, poison_heads = TAPE.horizon_td_tape_loss(
        poisoned_prediction, poisoned_target, mask, scale=scale
    )
    torch.testing.assert_close(poison_loss, loss)
    torch.testing.assert_close(poison_heads, per_head)
    poison_rms = TAPE.normalized_tape_rms(
        poisoned_prediction - poisoned_target, mask, scale
    )
    assert torch.isfinite(poison_rms).all()
    poison_loss_auto, poison_scale_auto, _ = TAPE.horizon_td_tape_loss(
        poisoned_prediction,
        poisoned_target,
        mask,
        latent_rms=torch.tensor(2.0),
    )
    torch.testing.assert_close(poison_loss_auto, loss)
    torch.testing.assert_close(poison_scale_auto, scale)


def test_targets_are_stopped_but_loss_routes_trunk_and_predictor_gradients():
    agent = TAPE.Agent(_DummyEnvs(), _args())
    source = agent.get_actor_feat(torch.randn(7, 7))
    prediction = agent.predict_tapes(source, torch.randn(7, 3))
    # Zero output head blocks hidden layers at initialization but output parameters and
    # trunk receive gradients once the tape depends on source; seed output first.
    with torch.no_grad():
        agent.tape_predictor.output.weight.normal_(0, 0.01)
    prediction = agent.predict_tapes(source, torch.randn(7, 3))
    target_source = torch.randn_like(prediction, requires_grad=True)
    target = TAPE.build_horizon_td_tape_targets(
        torch.randn(7, 1, 8),
        target_source.reshape(7, 1, 4, 8),
        torch.zeros(7, 1),
        torch.ones(7, 1),
    )
    loss, _, _ = TAPE.horizon_td_tape_loss(
        prediction, target.values[:, 0], target.masks[:, 0], scale=torch.tensor(1.0)
    )
    loss.backward()
    assert target_source.grad is None
    assert any(p.grad is not None and p.grad.norm() > 0 for p in agent.tape_trunk_parameters())
    assert any(p.grad is not None and p.grad.norm() > 0 for p in agent.tape_predictor_parameters())


def test_breaker_is_predictor_only_and_union_caps_are_exact():
    agent = TAPE.Agent(_DummyEnvs(), _args())
    with torch.no_grad():
        agent.tape_predictor.output.weight.normal_(0, 0.01)
    source = agent.get_actor_feat(torch.randn(7, 7))
    prediction = agent.predict_tapes(
        TAPE.tape_source_feature(source, trunk_active=False), torch.randn(7, 3)
    )
    prediction.square().mean().backward()
    assert all(p.grad is None for p in agent.tape_trunk_parameters())
    assert any(p.grad is not None for p in agent.tape_predictor_parameters())
    assert TAPE.Args().tape_trunk_grad_clip == 0.025
    assert TAPE.Args().tape_predictor_grad_clip == 0.25


def test_union_adam_receives_one_sum_and_private_predictor_has_separate_moments():
    task_parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    predictor_parameter = torch.nn.Parameter(torch.tensor([0.5, -0.5]))
    task_optimizer = torch.optim.Adam([task_parameter], lr=0.01, eps=1e-5)
    predictor_optimizer = torch.optim.Adam([predictor_parameter], lr=0.01, eps=1e-5)
    actor = {task_parameter: torch.tensor([0.1, 0.2])}
    critic = {task_parameter: torch.tensor([0.3, -0.1])}
    auxiliary = {task_parameter: torch.tensor([-0.2, 0.4])}
    merged = TAPE.apply_union_optimizer_step(
        [task_parameter],
        task_optimizer,
        actor_gradients=actor,
        critic_gradients=critic,
        auxiliary_gradients=auxiliary,
    )
    torch.testing.assert_close(merged[task_parameter], torch.tensor([0.2, 0.5]))
    assert task_optimizer.state[task_parameter]["step"] == 1
    assert predictor_parameter not in task_optimizer.state

    TAPE.apply_private_predictor_step(
        [predictor_parameter],
        predictor_optimizer,
        {predictor_parameter: torch.tensor([0.25, -0.25])},
    )
    assert predictor_optimizer.state[predictor_parameter]["step"] == 1
    assert task_parameter not in predictor_optimizer.state


def test_compiled_output_ownership_and_one_mark_per_call_are_present():
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cudagraph_mark_step_begin"
    ]
    assert calls
    assert "target_tape_fn(next_feature_chunk, next_action_chunk)" in source
    assert "target_next_prediction_chunks.append(\n                        retain_graph_output" in source
    compiled = TAPE.retain_graph_output(torch.tensor([1.0]), compiled=True)
    original = torch.tensor([1.0])
    retained = TAPE.retain_graph_output(original, compiled=True)
    original.add_(2.0)
    torch.testing.assert_close(retained, torch.tensor([1.0]))
    assert not compiled.requires_grad


def test_replacement_diagnostic_retains_first_mutable_graph_output():
    source = SCRIPT.read_text()
    action_call = source.index("action_replacement_prediction = retain_graph_output")
    source_call = source.index("source_replacement_prediction = retain_graph_output")
    assert action_call < source_call
    between = source[action_call:source_call]
    assert "compiled=args.compile" in between

    # Model the same replay-owned buffer contract: retaining call one must protect it
    # from call two mutating the shared graph output.
    shared = torch.tensor([1.0, 2.0])
    first = TAPE.retain_graph_output(shared, compiled=True)
    shared.copy_(torch.tensor([9.0, 8.0]))
    second = TAPE.retain_graph_output(shared, compiled=True)
    torch.testing.assert_close(first, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(second, torch.tensor([9.0, 8.0]))


def test_no_temporal_autograd_or_learned_unroll_in_training_source():
    source = SCRIPT.read_text()
    assert "for _ in range(agent.tape_horizon" not in source
    assert "retain_graph=True" not in source
    target_source = inspect.getsource(TAPE.build_horizon_td_tape_targets)
    assert ".detach()" in target_source
    assert "next_predictions.detach()" in target_source
    predictor_source = inspect.getsource(TAPE.HorizonTapePredictor.forward)
    assert "for " not in predictor_source


def test_head_and_cross_head_telemetry_is_scalar_and_finite():
    prediction = torch.randn(5, 4, 3)
    mask = torch.ones(5, 4, dtype=torch.bool)
    cosine, energy, phase = TAPE.tape_cross_head_statistics(
        prediction, mask, torch.tensor(2.0)
    )
    stats = TAPE.build_tape_head_telemetry(
        head_losses=torch.arange(4.0),
        bellman_errors=torch.ones(4),
        target_rms=torch.ones(4),
        target_rms_over_eta=torch.ones(4),
        prediction_rms=torch.ones(4),
        direct_mc_errors=torch.ones(4),
        direct_coverage=torch.ones(4),
        direct_zero_baseline=torch.ones(4),
        direct_nmse=torch.ones(4),
        direct_cosine=torch.ones(4),
        endpoint_errors=torch.ones(4),
        endpoint_persistence_baseline=torch.ones(4),
        endpoint_nmse=torch.ones(4),
        rollout_zero_baseline=torch.ones(4),
        rollout_persistence_baseline=torch.ones(4),
    )
    assert len(stats) == 60
    assert all(value.numel() == 1 for value in stats.values())
    assert torch.isfinite(torch.stack((cosine, energy, phase))).all()


def test_relative_tail_risk_is_exact_and_rng_free():
    returns = deque([100.0, 2000.0, 6000.0, -100.0], maxlen=512)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    stats = TAPE.summarize_episode_tail_risk(returns, (1500.0, 5000.5))
    assert stats["below_half_window_median_count"] == 2.0
    assert stats["below_half_window_median_fraction"] == 0.5
    assert stats["median"] == 1050.0
    assert stats["bottom_5pct_mean"] == -100.0
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.get_rng_state())


def test_paired_direct_and_endpoint_nmse_use_identical_rows():
    direct = torch.tensor(
        [[[1.0], [2.0]], [[3.0], [1000.0]], [[5.0], [2000.0]]]
    )
    prediction = direct.clone()
    prediction[0, 0] += 1.0
    prediction[1, 0] -= 1.0
    mask = torch.tensor([[True, True], [True, False], [False, False]])
    scale = torch.tensor(1.0)
    error = TAPE.normalized_tape_rms(prediction - direct, mask, scale)
    zero = TAPE.normalized_tape_rms(direct, mask, scale)
    nmse = (error / zero.clamp_min(1e-12)).square()
    torch.testing.assert_close(nmse[0], torch.tensor(0.2))
    torch.testing.assert_close(nmse[1], torch.tensor(0.0))

    endpoints = TAPE.tape_to_cumulative(direct)
    predicted_endpoints = TAPE.tape_to_cumulative(prediction)
    endpoint_error = TAPE.normalized_tape_rms(
        predicted_endpoints - endpoints, mask, scale
    )
    persistence = TAPE.normalized_tape_rms(endpoints, mask, scale)
    endpoint_nmse = (endpoint_error / persistence.clamp_min(1e-12)).square()
    torch.testing.assert_close(endpoint_nmse[0], torch.tensor(0.2))
    # Head two has one valid paired row: direct endpoint=3, predicted endpoint=4.
    torch.testing.assert_close(endpoint_nmse[1], torch.tensor(1.0 / 9.0))


def test_per_head_direct_cosine_tracks_phase_not_cross_head_similarity():
    target = torch.tensor(
        [[[1.0], [1.0]], [[-1.0], [1.0]], [[1.0], [-1.0]], [[-1.0], [-1.0]]]
    )
    mask = torch.ones(4, 2, dtype=torch.bool)
    prediction = target.clone()
    prediction[:, 1].neg_()
    cosine = TAPE.masked_head_cosine(prediction, target, mask)
    torch.testing.assert_close(cosine, torch.tensor([1.0, -1.0]))
