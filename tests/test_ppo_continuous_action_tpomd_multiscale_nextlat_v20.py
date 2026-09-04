import ast
import importlib.util
import inspect
import sys
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).parents[1]


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V20 = _load(
    "tpomd_multiscale_nextlat_v20",
    "cleanrl/tpo/md/ppo_continuous_action_tpomd_multiscale_nextlat_v20.py",
)
UNION = _load(
    "tpomd_nextlat_union_v17_reference_for_v20",
    "cleanrl/tpo/md/ppo_continuous_action_tpomd_nextlat_union_v17.py",
)
BASE = _load(
    "tpomd_v5_reference_for_v20",
    "cleanrl/iterthink/v24_d4hlgauss/rawret/ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_idxtransfer_tpomd_v5_dyntrust.py",
)


class _DummyEnvs:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(7,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))


def _args(module, **overrides):
    values = dict(hidden=8, k_blocks=2, n_experts=2, num_bins=7)
    values.update(overrides)
    return module.Args(**values)


def _semantic_ast(obj):
    tree = ast.parse(inspect.getsource(obj))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]
    return ast.dump(tree, include_attributes=False)


def test_default_and_mode_resolution_are_causal_controls():
    args = V20.Args()
    assert args.pc_mode == "direct"
    assert args.nextlat_horizons == "1,2,3,4"
    assert args.tail_risk_thresholds == (1500.0, 5000.0)
    assert args.nextlat_loss_scale == "raw"
    assert V20.resolve_nextlat_spec("off", "1,2,3,4") == V20.NextLatSpec(
        "off", (), (), ()
    )
    assert V20.resolve_nextlat_spec("recursive", "1,2,3,4") == V20.NextLatSpec(
        "recursive", (1, 2, 3, 4), (1, 2, 3, 4), ()
    )
    assert V20.resolve_nextlat_spec("direct", "1,4,16,32") == V20.NextLatSpec(
        "direct", (1, 4, 16, 32), (), (1, 4, 16, 32)
    )
    assert V20.resolve_nextlat_spec("hybrid", "1,4,16,32", 0, 4) == V20.NextLatSpec(
        "hybrid", (1, 4, 16, 32), (1, 4), (16, 32)
    )
    assert V20.parse_nextlat_horizons("9", recursive_depth=4) == (1, 2, 3, 4)


@pytest.mark.parametrize(
    "call",
    [
        lambda: V20.resolve_nextlat_spec("unknown", "1"),
        lambda: V20.parse_nextlat_horizons(""),
        lambda: V20.parse_nextlat_horizons("1,1"),
        lambda: V20.parse_nextlat_horizons("2,1"),
        lambda: V20.parse_nextlat_horizons("0,1"),
        lambda: V20.resolve_nextlat_spec("hybrid", "1,2,4", 0, 4),
        lambda: V20.resolve_nextlat_spec("hybrid", "8,16", 0, 4),
        lambda: V20.resolve_nextlat_spec("recursive", "1,4,8"),
        lambda: V20.resolve_nextlat_spec("hybrid", "1,4,8,16", 0, 8),
        lambda: V20.parse_tail_thresholds("1500,nan"),
    ],
)
def test_configuration_guards(call):
    with pytest.raises(ValueError):
        call()


def test_randomized_multiscale_mask_matches_transition_oracle():
    generator = torch.Generator().manual_seed(787)
    for num_steps, num_envs, horizons in [(13, 3, (1, 4, 7)), (8, 5, (2, 3, 6))]:
        boundaries = (torch.rand(num_steps, num_envs, generator=generator) < 0.2).float()
        actual = V20.build_multiscale_nextlat_mask(boundaries, horizons)
        expected = torch.zeros_like(actual)
        for step in range(num_steps):
            for env in range(num_envs):
                for index, horizon in enumerate(horizons):
                    expected[step, env, index] = float(
                        step + horizon < num_steps
                        and not boundaries[step : step + horizon, env].bool().any()
                    )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_multiscale_indices_match_t_major_oracle():
    sources = np.array([0, 1, 5, 14, 15])
    action, target = V20.make_multiscale_nextlat_indices(sources, 2, 16, (1, 3, 5))
    expected_action = np.clip(sources[None] + 2 * np.arange(5)[:, None], 0, 15)
    expected_target = np.clip(sources[None] + 2 * np.array([1, 3, 5])[:, None], 0, 15)
    np.testing.assert_array_equal(action, expected_action)
    np.testing.assert_array_equal(target, expected_target)


def test_recursive_depth4_is_exact_union_prediction_semantics():
    torch.manual_seed(113)
    candidate = V20.Agent(
        _DummyEnvs(), _args(V20, pc_mode="recursive", nextlat_recursive_depth=4)
    )
    torch.manual_seed(113)
    reference = UNION.Agent(_DummyEnvs(), _args(UNION))
    candidate_predictor = candidate.nextlat_predictor.state_dict()
    for name, tensor in reference.nextlat_predictor.state_dict().items():
        torch.testing.assert_close(candidate_predictor[name], tensor, rtol=0.0, atol=0.0)
    source = torch.randn(9, 8)
    actions = torch.randn(4, 9, 3)
    expected = []
    predicted = source
    for action in actions:
        predicted = reference.nextlat_predictor(torch.cat((predicted, action), dim=-1))
        expected.append(predicted)
    actual = V20.multiscale_prediction_forward(candidate, source, actions)
    torch.testing.assert_close(actual, torch.stack(expected), rtol=0.0, atol=0.0)

    targets = torch.randn(4, 9, 8)
    masks = torch.tensor(
        [[1, 1, 0, 1, 0, 1, 1, 1, 0]] * 4, dtype=torch.float32
    )
    expected_prediction_losses, expected_kls = [], []
    for prediction, target, mask in zip(expected, targets, masks):
        denominator = mask.sum().clamp_min(1.0)
        per_example = F.smooth_l1_loss(prediction, target, reduction="none").mean(-1)
        expected_prediction_losses.append((per_example * mask).sum() / denominator)
        with torch.no_grad():
            target_dist, _, _ = candidate._actor_dist_frozen_head(target)
        prediction_dist, _, _ = candidate._actor_dist_frozen_head(prediction)
        kl = torch.distributions.kl_divergence(target_dist, prediction_dist).sum(-1)
        expected_kls.append((kl * mask).sum() / denominator)
    actual_losses = V20.compute_nextlat_loss(
        candidate,
        actual,
        targets,
        masks,
        torch.ones(4),
        "raw",
    )
    torch.testing.assert_close(
        actual_losses["prediction"], torch.stack(expected_prediction_losses).mean(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_losses["policy_kl"], torch.stack(expected_kls).mean(), rtol=0, atol=0
    )


def test_direct_prefix_has_no_future_leakage_and_endpoints_are_independent():
    torch.manual_seed(91)
    predictor = V20.DirectHorizonPredictor(8, 3, (1, 2, 4), 4)
    source = torch.randn(6, 8)
    actions = torch.randn(4, 6, 3, requires_grad=True)
    baseline = predictor(source, actions)
    changed = actions.detach().clone()
    changed[1:] += 1000.0
    torch.testing.assert_close(predictor(source, changed)[0], baseline[0], rtol=0.0, atol=0.0)
    baseline[0].sum().backward()
    torch.testing.assert_close(actions.grad[1:], torch.zeros_like(actions.grad[1:]))

    single = V20.DirectHorizonPredictor(8, 3, (4,), 4)
    single.load_state_dict(predictor.state_dict())
    torch.testing.assert_close(single(source, actions.detach())[0], baseline[2], rtol=0.0, atol=0.0)


def test_rollout_target_spread_uses_weighted_global_moments():
    targets = torch.tensor(
        [
            [[1.0, 3.0], [5.0, 7.0], [100.0, 100.0]],
            [[2.0, 2.0], [4.0, 8.0], [10.0, 12.0]],
        ]
    )
    masks = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    actual = V20.compute_rollout_target_spreads(targets, masks)
    expected = torch.stack(
        [targets[0, :2].reshape(-1).std(correction=0), targets[1, 1:].reshape(-1).std(correction=0)]
    )
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad


def test_loss_is_equal_horizon_average_valid_normalized_and_scaled():
    agent = V20.Agent(_DummyEnvs(), _args(V20, pc_mode="direct", nextlat_horizons="1,2"))
    source = torch.zeros(3, 8)
    targets = torch.zeros(2, 3, 8)
    predictions = torch.zeros_like(targets)
    predictions[0, 0] = 1.0
    predictions[1, :2] = 2.0
    masks = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    spreads = torch.tensor([1.0, 2.0])
    raw = V20.compute_nextlat_loss(
        agent, predictions, targets, masks, spreads, "raw"
    )
    scaled = V20.compute_nextlat_loss(
        agent, predictions, targets, masks, spreads, "target_std"
    )
    expected_raw = (
        F.smooth_l1_loss(torch.tensor(1.0), torch.tensor(0.0))
        + F.smooth_l1_loss(torch.tensor(2.0), torch.tensor(0.0))
    ) / 2
    expected_scaled = (
        F.smooth_l1_loss(torch.tensor(1.0), torch.tensor(0.0))
        + F.smooth_l1_loss(torch.tensor(2.0), torch.tensor(0.0)) / 2
    ) / 2
    torch.testing.assert_close(raw["prediction"], expected_raw)
    torch.testing.assert_close(scaled["prediction"], expected_scaled)


def test_all_invalid_horizons_are_finite_zero():
    agent = V20.Agent(_DummyEnvs(), _args(V20, pc_mode="direct", nextlat_horizons="1"))
    source = torch.randn(4, 8)
    target = torch.randn(1, 4, 8)
    prediction = torch.randn(1, 4, 8, requires_grad=True)
    result = V20.compute_nextlat_loss(
        agent, prediction, target, torch.zeros(1, 4), torch.ones(1), "raw"
    )
    assert result["prediction"].item() == 0.0
    assert result["policy_kl"].item() == 0.0
    assert torch.isfinite(torch.stack(tuple(value.mean() for value in result.values()))).all()


def test_normalized_endpoint_error_is_separate_no_grad_diagnostic():
    source = torch.zeros(3, 2, requires_grad=True)
    targets = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [100.0, 100.0]],
            [[2.0, 2.0], [4.0, 4.0], [8.0, 8.0]],
        ]
    )
    predictions = targets.detach().clone()
    predictions[0, :2] = torch.tensor([[0.5, 0.5], [1.0, 1.0]])
    predictions[1] = 0.0
    predictions.requires_grad_(True)
    masks = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    actual = V20.normalized_endpoint_errors_vs_persistence(
        predictions, source, targets, masks
    )
    torch.testing.assert_close(actual, torch.tensor([0.25, 1.0]))
    assert not actual.requires_grad


def test_latent_participation_rank_uses_trace_ratio_without_eigendecomposition():
    isotropic = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    )
    scale, participation_rank = V20.latent_scale_and_participation_rank(isotropic)
    torch.testing.assert_close(scale, torch.sqrt(torch.tensor(1.0 / 3.0)))
    torch.testing.assert_close(participation_rank, torch.tensor(2.0))

    rank_one = torch.tensor([[-1.0, -2.0], [0.0, 0.0], [1.0, 2.0]])
    _, rank_one_participation = V20.latent_scale_and_participation_rank(rank_one)
    torch.testing.assert_close(rank_one_participation, torch.tensor(1.0))
    helper_source = inspect.getsource(V20.latent_scale_and_participation_rank)
    assert "eigvalsh" not in helper_source
    assert "trace.square() / trace_of_square.clamp_min" in helper_source


def test_direct_parameter_budget_is_closely_matched_and_override_is_exact():
    agent = V20.Agent(_DummyEnvs(), _args(V20, pc_mode="direct", nextlat_horizons="1,2,3,4"))
    actual, reference = agent.nextlat_parameter_budget()
    assert abs(actual / reference - 1.0) < 0.06
    overridden = V20.Agent(
        _DummyEnvs(),
        _args(V20, pc_mode="direct", nextlat_horizons="1,4,16,32", nextlat_direct_hidden=11),
    )
    assert overridden.nextlat_direct_predictor.network[0].out_features == 11


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_off_is_exact_base_initialization_forward_rng_and_task_adam(actor_dist):
    torch.manual_seed(417)
    base = BASE.Agent(_DummyEnvs(), _args(BASE, actor_dist=actor_dist))
    base_rng = torch.get_rng_state().clone()
    torch.manual_seed(417)
    off = V20.Agent(_DummyEnvs(), _args(V20, actor_dist=actor_dist, pc_mode="off"))
    off_rng = torch.get_rng_state().clone()
    assert tuple(off.state_dict()) == tuple(base.state_dict())
    for name, value in base.state_dict().items():
        torch.testing.assert_close(off.state_dict()[name], value, rtol=0.0, atol=0.0)
    assert torch.equal(off_rng, base_rng)
    assert not off.nextlat_parameters()

    observations = torch.randn(7, 7)
    base_optimizer = torch.optim.Adam(base.parameters(), lr=3e-4, eps=1e-5)
    off_optimizer = torch.optim.Adam(off.task_parameters(), lr=3e-4, eps=1e-5)
    base_loss = sum(tensor.float().sum() for tensor in V20.policy_model_forward(base, observations))
    off_loss = sum(tensor.float().sum() for tensor in V20.policy_model_forward(off, observations))
    base_loss.backward()
    off_loss.backward()
    base_optimizer.step()
    off_optimizer.step()
    for name, value in base.state_dict().items():
        torch.testing.assert_close(off.state_dict()[name], value, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_frozen_decoder_routes_gradients_only_to_source_and_predictor(actor_dist):
    agent = V20.Agent(
        _DummyEnvs(), _args(V20, actor_dist=actor_dist, pc_mode="direct", nextlat_horizons="1,2")
    )
    source = torch.randn(5, 8, requires_grad=True)
    actions = torch.randn(2, 5, 3)
    targets = torch.randn(2, 5, 8)
    predictions = V20.multiscale_prediction_forward(agent, source, actions)
    losses = V20.compute_nextlat_loss(
        agent, predictions, targets, torch.ones(2, 5), torch.ones(2), "raw"
    )
    (losses["prediction"] + losses["policy_kl"]).backward()
    assert source.grad is not None and source.grad.norm() > 0
    assert any(parameter.grad is not None for parameter in agent.nextlat_predictor_parameters())
    head_names = ("actor_head", "actor_logvar_head") if actor_dist == "gaussian" else (
        "actor_alpha_head", "actor_beta_head"
    )
    for name in head_names:
        assert all(parameter.grad is None for parameter in getattr(agent, name).parameters())
    task_ids = {id(parameter) for parameter in agent.task_parameters()}
    predictor_ids = {id(parameter) for parameter in agent.nextlat_predictor_parameters()}
    assert task_ids.isdisjoint(predictor_ids)


def test_one_global_auxiliary_clip_bounds_total_trunk_budget_for_any_horizon_count():
    for multiplier in (1.0, 100.0):
        trunk = [torch.nn.Parameter(torch.ones(5)), torch.nn.Parameter(torch.ones(7))]
        predictor = [torch.nn.Parameter(torch.ones(3))]
        for parameter in trunk + predictor:
            parameter.grad = torch.full_like(parameter, multiplier)
        _, _, trunk_gradients, _ = V20.capture_nextlat_gradient_groups(
            trunk,
            predictor,
            trunk_active=True,
            trunk_max_norm=0.025,
            predictor_max_norm=0.25,
        )
        delivered = torch.stack(
            [gradient.square().sum() for gradient in trunk_gradients.values()]
        ).sum().sqrt()
        assert delivered <= 0.025001


def test_final_minibatch_gradient_mechanism_statistics():
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(1))
    parameters = [first, second]
    auxiliary = {first: torch.tensor([1.0, 0.0]), second: torch.tensor([2.0])}
    actor = {first: torch.tensor([1.0, 0.0]), second: torch.tensor([0.0])}
    critic = {first: torch.tensor([0.0, 1.0]), second: torch.tensor([2.0])}
    torch.testing.assert_close(
        V20.gradient_group_norm(parameters, auxiliary), torch.sqrt(torch.tensor(5.0))
    )
    torch.testing.assert_close(
        V20.gradient_group_cosine(parameters, auxiliary, actor),
        torch.tensor(1.0 / np.sqrt(5.0), dtype=torch.float32),
    )
    expected_task = torch.tensor(5.0 / np.sqrt(5.0 * 6.0), dtype=torch.float32)
    torch.testing.assert_close(
        V20.gradient_group_cosine(parameters, auxiliary, actor, critic), expected_task
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_group_fails_before_adam_step(bad):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.02)
    with pytest.raises(RuntimeError):
        V20.apply_union_optimizer_step(
            [parameter],
            optimizer,
            actor_gradients={parameter: torch.tensor([bad])},
            critic_gradients={},
            auxiliary_gradients={},
        )
    assert parameter not in optimizer.state


@pytest.mark.parametrize("mode,horizons", [("recursive", "1,2,3,4"), ("direct", "1,2,3,4")])
def test_static_multiscale_prediction_wrapper_compiles_fullgraph(mode, horizons):
    torch.manual_seed(10)
    eager_agent = V20.Agent(_DummyEnvs(), _args(V20, pc_mode=mode, nextlat_horizons=horizons))
    torch.manual_seed(10)
    compiled_agent = V20.Agent(_DummyEnvs(), _args(V20, pc_mode=mode, nextlat_horizons=horizons))
    source = torch.randn(6, 8)
    actions = torch.randn(4, 6, 3)
    compiled = torch.compile(
        lambda source_, actions_: V20.multiscale_prediction_forward(
            compiled_agent, source_, actions_
        ),
        backend="eager",
        dynamic=False,
        fullgraph=True,
    )
    expected = V20.multiscale_prediction_forward(eager_agent, source, actions)
    actual = compiled(source, actions)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_tail_risk_telemetry_is_deterministic_and_configurable():
    returns = deque([100.0, 2000.0, 6000.0, -100.0], maxlen=512)
    stats = V20.tail_risk_statistics(returns, (1500.0, 5000.5))
    assert stats["tail_risk/window_size"] == 4
    assert stats["tail_risk/median"] == 1050.0
    assert stats["tail_risk/cvar_05"] == -100.0
    assert stats["tail_risk/bottom_5pct_mean"] == -100.0
    assert stats["tail_risk/below_1500_count"] == 2
    assert stats["tail_risk/below_5000p5_fraction"] == 0.75
    assert stats["tail_risk/below_half_window_median_count"] == 2
    assert stats["tail_risk/below_half_window_median_fraction"] == 0.5


def test_scalar_max_telemetry_accumulates_exactly_without_host_materialization():
    maximum = torch.full((), float("-inf"), dtype=torch.float64)
    for value in (-0.2, -0.7, 0.125, 0.03125):
        returned = V20.update_scalar_max_(maximum, torch.tensor(value))
        assert returned is maximum
    assert maximum.item() == 0.125
    helper_source = inspect.getsource(V20.update_scalar_max_)
    for host_operation in (".item(", ".cpu(", ".numpy(", ".tolist("):
        assert host_operation not in helper_source


def test_tail_threshold_metric_names_round_trip_distinct_float_values():
    first = 1.0
    second = np.nextafter(first, 2.0)
    assert format(first, "g") == format(second, "g")
    assert V20.metric_number(first) != V20.metric_number(second)
    assert V20.metric_number(second) == "1p0000000000000002"
    stats = V20.tail_risk_statistics([0.0, 2.0], (first, second))
    assert f"tail_risk/below_{V20.metric_number(first)}_count" in stats
    assert f"tail_risk/below_{V20.metric_number(second)}_count" in stats


def test_counterfactual_diagnostics_are_cross_sample_replacements_not_time_shuffles():
    source = Path(V20.__file__).read_text()
    assert "diagnostic_actions.roll(1, dims=1)" in source
    assert "action_prefix_replacement" in source
    assert "source_replacement" in source
    assert "shuffle_action_prediction" not in source
    assert "permute temporal order within a prefix" in source


def test_source_preserves_tpo_chassis_and_compiled_predictor_perf_contract():
    source = Path(V20.__file__).read_text()
    main = source[source.index('if __name__ == "__main__":') :]
    for name in (
        "IndexedTransferBranch",
        "ThinkBlock",
        "ThinkTrunk",
        "tpo_restricted_target",
        "tpo_reverse_kl",
    ):
        assert _semantic_ast(getattr(V20, name)) == _semantic_ast(getattr(UNION, name))
    assert main.count("task_optimizer = optim.Adam(") == 1
    assert main.count("predictor_optimizer = optim.Adam(") == 1
    assert "nextlat_prediction_fn = torch.compile(" in main
    assert "fullgraph=True" in main
    assert "predicted_features = nextlat_prediction_fn(h_hat, future_actions)" in main
    predictor_call = main.index(
        "predicted_features = nextlat_prediction_fn(h_hat, future_actions)"
    )
    assert "cudagraph_mark_step_begin" not in main[predictor_call - 180 : predictor_call]
    assert "agent.nextlat_predictor(" not in main
    assert inspect.getsource(V20.multiscale_prediction_forward).count(
        "agent.nextlat_direct_predictor(source, future_actions)"
    ) == 1
    loss_source = inspect.getsource(V20.compute_nextlat_loss)
    assert "prediction_losses = torch.stack(prediction_losses)" in loss_source
    assert '"prediction": prediction_losses.mean()' in loss_source
    assert "persistence" not in loss_source
    assert "square()" not in loss_source
    assert "source" not in inspect.signature(V20.compute_nextlat_loss).parameters
    assert main.count("normalized_endpoint_errors_vs_persistence(") == 1
    assert main.index("normalized_endpoint_errors_vs_persistence(") > main.index(
        "mechanism_telemetry = {}"
    )
    assert "nextlat_latent_effective_rank" not in source
    assert "nextlat_latent_participation_rank" in source
    assert 'format(float(value), ".17g")' in inspect.getsource(V20.metric_number)
    assert "torch.random" not in inspect.getsource(V20.tail_risk_statistics)
    assert 'if predictor_optimizer is not None:' in main
    assert main.index('if predictor_optimizer is not None:') < main.index(
        'predictor_optimizer.param_groups[0]["lr"] = lrnow'
    )
    assert main.count("synchronize_scalar_telemetry(device_telemetry)") == 1
    assert main.count("update_scalar_max_(") == 3
    assert '"losses/old_approx_kl": old_approx_kl' in main
    assert '"losses/approx_kl": approx_kl' in main
    for tag in (
        "kl_diagnostics/rollout_max_minibatch_approx_kl",
        "kl_diagnostics/rollout_max_epoch_mean_approx_kl",
        "kl_diagnostics/rollout_max_minibatch_old_approx_kl",
    ):
        assert f'"{tag}"' in main
        assert main.index(f'"{tag}"') < main.index(
            "host_telemetry = synchronize_scalar_telemetry(device_telemetry)"
        )
    epoch_mean_update = main.index(
        "update_scalar_max_(rollout_max_epoch_mean_approx_kl, epoch_mean_kl)"
    )
    assert epoch_mean_update < main.index("if actor_active:", epoch_mean_update)
