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


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "cleanrl/tpo/md/ppo_continuous_action_tpomd_grounded_gvf_v21.py"


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GVF = _load(
    "tpomd_grounded_gvf_v21",
    "cleanrl/tpo/md/ppo_continuous_action_tpomd_grounded_gvf_v21.py",
)
UNION = _load(
    "tpomd_nextlat_union_reference_for_grounded_gvf",
    "cleanrl/tpo/md/ppo_continuous_action_tpomd_nextlat_union_v17.py",
)
BASE = _load(
    "tpomd_v5_reference_for_grounded_gvf",
    "cleanrl/iterthink/v24_d4hlgauss/rawret/ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_idxtransfer_tpomd_v5_dyntrust.py",
)


class _DummyEnvs:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(7,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))


def _args(module, actor_dist="beta"):
    return module.Args(
        actor_dist=actor_dist,
        hidden=8,
        k_blocks=2,
        n_experts=2,
        num_bins=7,
    )


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


def _scalar_gvf_oracle(
    cumulants,
    tails,
    next_tails,
    terminations,
    boundaries,
    valids,
    gammas,
    horizons,
):
    """Independent slow reference for randomized boundary tests."""
    time_dim, num_envs, _ = cumulants.shape
    num_bands = len(gammas)
    values = torch.zeros_like(tails)
    masks = valids.bool().unsqueeze(-1).expand(time_dim, num_envs, num_bands).clone()
    bootstraps = torch.zeros_like(masks)
    observed_mass = cumulants.new_zeros(masks.shape)
    bootstrap_mass = cumulants.new_zeros(masks.shape)
    boundaries = boundaries.bool()
    terminations = terminations.bool()
    valids = valids.bool()
    for source in range(time_dim):
        for env in range(num_envs):
            if not valids[source, env]:
                continue
            for band, (gamma, horizon) in enumerate(zip(gammas, horizons)):
                discount = 1.0
                for offset in range(horizon):
                    row = source + offset
                    if not valids[row, env]:
                        values[source, env, band] += discount * tails[row, env, band]
                        bootstraps[source, env, band] = True
                        bootstrap_mass[source, env, band] += discount
                        break
                    weight = (1.0 - gamma) * discount
                    values[source, env, band] += weight * cumulants[row, env]
                    observed_mass[source, env, band] += weight
                    next_discount = discount * gamma
                    if boundaries[row, env]:
                        if not terminations[row, env]:
                            values[source, env, band] += (
                                next_discount * next_tails[row, env, band]
                            )
                            bootstraps[source, env, band] = True
                            bootstrap_mass[source, env, band] += next_discount
                        break
                    if offset + 1 == horizon and row + 1 < time_dim:
                        values[source, env, band] += (
                            next_discount * tails[row + 1, env, band]
                        )
                        bootstraps[source, env, band] = True
                        bootstrap_mass[source, env, band] += next_discount
                        break
                    if row + 1 == time_dim:
                        values[source, env, band] += (
                            next_discount * next_tails[row, env, band]
                        )
                        bootstraps[source, env, band] = True
                        bootstrap_mass[source, env, band] += next_discount
                        break
                    discount = next_discount
    return values, masks, bootstraps, observed_mass, bootstrap_mass


def test_defaults_define_four_distinct_questions_with_bounded_tail_mass():
    args = GVF.Args()
    assert args.env_id == "HalfCheetah-v4"
    assert args.total_timesteps == 8_000_000
    assert args.seed == 1
    assert args.cuda
    assert args.gvf_gammas == (0.50, 0.90, 0.97, 0.99)
    assert args.gvf_trace_lambda == 1.0
    horizons = GVF.gvf_prefix_horizons(
        args.gvf_gammas, args.gvf_residual_tail_mass
    )
    assert horizons == (4, 22, 76, 230)
    for gamma, horizon in zip(args.gvf_gammas, horizons):
        assert gamma**horizon <= args.gvf_residual_tail_mass
        assert gamma ** (horizon - 1) > args.gvf_residual_tail_mass
    assert args.tail_risk_window == 512
    assert args.tail_risk_thresholds == (1500.0, 5000.0)


@pytest.mark.parametrize(
    "gammas,tail",
    [((0.0,), 0.1), ((1.0,), 0.1), ((0.9,), 0.0), ((0.9,), 1.0), ((), 0.1)],
)
def test_gamma_and_residual_mass_endpoints_are_rejected(gammas, tail):
    with pytest.raises(ValueError):
        GVF.gvf_prefix_horizons(gammas, tail)


def test_grounded_cumulants_are_fixed_observables_and_stop_gradient():
    observations = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=True)
    next_observations = torch.tensor(
        [[[2.0, 0.0]], [[100.0, 100.0]]], requires_grad=True
    )
    rewards = torch.tensor([[5.0], [9.0]], requires_grad=True)
    actions = torch.tensor([[[1.0, -1.0]], [[0.5, 0.5]]], requires_grad=True)
    terminations = torch.tensor([[0.0], [1.0]])
    valids = torch.tensor([[1.0], [0.0]])

    result = GVF.build_grounded_cumulants(
        observations,
        next_observations,
        rewards,
        actions,
        terminations,
        valids,
    )

    expected = torch.tensor([[[1.0, -2.0, 5.0, 1.0, 1.0]], [[0.0] * 5]])
    torch.testing.assert_close(result.values, expected)
    assert result.obs_dim == 2
    assert (result.reward_index, result.control_index, result.continuation_index) == (
        2,
        3,
        4,
    )
    assert not result.values.requires_grad
    assert "encoder" not in inspect.signature(GVF.build_grounded_cumulants).parameters


def test_forward_view_is_exact_normalized_bellman_prefix_plus_one_tail():
    cumulants = torch.arange(1.0, 6.0).view(5, 1, 1)
    tails = (10.0 * torch.arange(1.0, 6.0)).view(5, 1, 1, 1)
    next_tails = torch.full_like(tails, 10.0)
    result = GVF.build_grounded_gvf_targets(
        cumulants,
        tails,
        next_tails,
        torch.zeros(5, 1),
        torch.zeros(5, 1),
        torch.ones(5, 1),
        gammas=(0.5,),
        prefix_horizons=(2,),
    )

    # (1-g)*c0 + (1-g)*g*c1 + g^2*Psi(s2,a2)
    torch.testing.assert_close(result.values[0, 0, 0, 0], torch.tensor(8.5))
    torch.testing.assert_close(result.observed_mass[0, 0, 0], torch.tensor(0.75))
    torch.testing.assert_close(result.bootstrap_mass[0, 0, 0], torch.tensor(0.25))
    # Artificial-edge rows bootstrap once from the actual final next state.
    torch.testing.assert_close(result.values[4, 0, 0, 0], torch.tensor(7.5))
    assert result.bootstrap_masks[:, 0, 0].all()


def test_terminal_truncation_tail_and_missing_final_semantics():
    cumulants = torch.tensor([1.0, 2.0, 100.0, 200.0]).view(4, 1, 1)
    tails = torch.full((4, 1, 1, 1), 1000.0)
    next_tails = torch.full_like(tails, 20.0)
    boundaries = torch.tensor([[0.0], [1.0], [0.0], [0.0]])

    terminal = GVF.build_grounded_gvf_targets(
        cumulants,
        tails,
        next_tails,
        boundaries,
        boundaries,
        torch.ones_like(boundaries),
        (0.5,),
        (3,),
    )
    torch.testing.assert_close(terminal.values[0, 0, 0, 0], torch.tensor(1.0))
    assert not terminal.bootstrap_masks[0, 0, 0]

    truncation = GVF.build_grounded_gvf_targets(
        cumulants,
        tails,
        next_tails,
        torch.zeros_like(boundaries),
        boundaries,
        torch.ones_like(boundaries),
        (0.5,),
        (3,),
    )
    torch.testing.assert_close(truncation.values[0, 0, 0, 0], torch.tensor(6.0))
    assert truncation.bootstrap_masks[0, 0, 0]

    missing_valids = torch.tensor([[1.0], [0.0], [1.0], [1.0]])
    missing = GVF.build_grounded_gvf_targets(
        cumulants,
        tails,
        next_tails,
        boundaries,
        boundaries,
        missing_valids,
        (0.5,),
        (3,),
    )
    torch.testing.assert_close(missing.values[0, 0, 0, 0], torch.tensor(500.5))
    assert not missing.masks[1, 0, 0]
    assert missing.bootstrap_masks[0, 0, 0]


def test_vectorized_targets_match_randomized_scalar_oracle():
    for seed in range(25):
        generator = torch.Generator().manual_seed(seed)
        time_dim, num_envs, bands, features = 9, 3, 3, 4
        cumulants = torch.randn(time_dim, num_envs, features, generator=generator)
        tails = torch.randn(
            time_dim, num_envs, bands, features, generator=generator
        )
        next_tails = torch.randn_like(tails, generator=generator)
        boundaries = torch.rand(time_dim, num_envs, generator=generator) < 0.20
        terminations = boundaries & (
            torch.rand(time_dim, num_envs, generator=generator) < 0.55
        )
        valids = torch.ones_like(boundaries)
        missing = boundaries & (
            torch.rand(time_dim, num_envs, generator=generator) < 0.20
        )
        valids[missing] = False
        gammas = (0.4, 0.8, 0.95)
        horizons = (1, 3, 6)
        actual = GVF.build_grounded_gvf_targets(
            cumulants,
            tails,
            next_tails,
            terminations,
            boundaries,
            valids,
            gammas,
            horizons,
        )
        expected = _scalar_gvf_oracle(
            cumulants,
            tails,
            next_tails,
            terminations,
            boundaries,
            valids,
            gammas,
            horizons,
        )
        for field, expected_tensor in zip((
            "values",
            "masks",
            "bootstrap_masks",
            "observed_mass",
            "bootstrap_mass",
        ), expected):
            torch.testing.assert_close(getattr(actual, field), expected_tensor)


def test_targets_and_loss_scales_are_stop_gradient_and_keep_raw_coordinates():
    cumulants = torch.randn(6, 2, 5, requires_grad=True)
    tails = torch.randn(6, 2, 2, 5, requires_grad=True)
    next_tails = torch.randn(6, 2, 2, 5, requires_grad=True)
    targets = GVF.build_grounded_gvf_targets(
        cumulants,
        tails,
        next_tails,
        torch.zeros(6, 2),
        torch.zeros(6, 2),
        torch.ones(6, 2),
        (0.6, 0.95),
        (2, 4),
    )
    assert not targets.values.requires_grad
    prediction = torch.randn_like(targets.values, requires_grad=True)
    loss, scale, per_band, _ = GVF.grounded_gvf_loss(
        prediction,
        targets.values,
        targets.masks,
        obs_dim=2,
        reward_weight=0.25,
        control_weight=0.1,
        continuation_weight=0.1,
    )
    loss.backward()
    assert prediction.grad is not None
    assert scale.shape == (2, 5) and not scale.requires_grad
    assert per_band.shape == (2,)
    # Scaling raw Bellman coordinates and their predictions changes only detached units.
    scaled_loss, scaled_scale, _, _ = GVF.grounded_gvf_loss(
        13.0 * prediction.detach(),
        13.0 * targets.values,
        targets.masks,
        obs_dim=2,
        reward_weight=0.25,
        control_weight=0.1,
        continuation_weight=0.1,
    )
    torch.testing.assert_close(scaled_loss, loss.detach())
    torch.testing.assert_close(scaled_scale, 13.0 * scale)


def test_equal_band_average_does_not_downweight_slow_questions():
    target = torch.zeros(10, 2, 4)
    prediction = target.clone()
    prediction[:, 1] = 2.0
    mask = torch.ones(10, 2, dtype=torch.bool)
    scale = torch.ones(2, 4)
    loss, _, per_band, _ = GVF.grounded_gvf_loss(
        prediction,
        target,
        mask,
        obs_dim=1,
        reward_weight=1.0,
        control_weight=1.0,
        continuation_weight=1.0,
        scale=scale,
    )
    assert per_band[0] == 0.0
    assert per_band[1] > 0.0
    torch.testing.assert_close(loss, per_band.mean())


def test_normalized_gvf_rms_reduces_samples_and_channels_per_band():
    residual = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
            [[2.0, 4.0, 6.0], [10.0, 10.0, 10.0]],
            [[20.0, 20.0, 20.0], [1.0, 2.0, 3.0]],
        ]
    )
    mask = torch.tensor([[True, True], [True, False], [False, True]])
    scale = torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])

    actual = GVF.normalized_gvf_rms(residual, mask, scale)
    expected = []
    for band in range(2):
        normalized_squares = []
        for row in range(3):
            if mask[row, band]:
                for channel in range(3):
                    normalized_squares.append(
                        (residual[row, band, channel] / scale[band, channel]) ** 2
                    )
        expected.append(torch.stack(normalized_squares).mean().sqrt())

    assert actual.shape == (2,)
    torch.testing.assert_close(actual, torch.stack(expected))


def test_full_gvf_band_telemetry_is_scalar_and_packed_sync_accepts_it():
    target_channel_rms = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0]]
    )
    vector = torch.tensor([0.25, 0.75])
    telemetry = GVF.build_gvf_band_telemetry(
        (0.5, 0.9),
        band_losses=vector,
        band_errors=vector + 1.0,
        target_channel_rms=target_channel_rms,
        zero_baseline=vector + 2.0,
        mean_baseline=vector + 3.0,
        signflip_action_sensitivity=vector + 4.0,
        observed_target_mass=vector + 5.0,
        frozen_tail_bootstrap_mass=vector + 6.0,
        bootstrap_fraction=vector + 7.0,
        obs_dim=2,
        reward_index=2,
        control_index=3,
        continuation_index=4,
    )

    assert telemetry
    assert all(value.shape == torch.Size([]) for value in telemetry.values())
    expected_pooled = target_channel_rms[0].square().mean().sqrt()
    torch.testing.assert_close(
        telemetry["gvf/band_0p5_pooled_target_rms"], expected_pooled
    )
    assert "gvf/band_0p5_frozen_tail_bootstrap_mass" in telemetry
    assert not any("ema_bootstrap_mass" in tag for tag in telemetry)
    assert not any("residual_bootstrap_mass" in tag for tag in telemetry)
    host = GVF.synchronize_scalar_telemetry(telemetry)
    assert host.keys() == telemetry.keys()
    assert all(isinstance(value, float) for value in host.values())

    with pytest.raises(ValueError, match="shape \\[num_bands\\]"):
        GVF.build_gvf_band_telemetry(
            (0.5, 0.9),
            band_losses=torch.ones(2, 5),
            band_errors=vector,
            target_channel_rms=target_channel_rms,
            zero_baseline=vector,
            mean_baseline=vector,
            signflip_action_sensitivity=vector,
            observed_target_mass=vector,
            frozen_tail_bootstrap_mass=vector,
            bootstrap_fraction=vector,
            obs_dim=2,
            reward_index=2,
            control_index=3,
            continuation_index=4,
        )


def test_forked_auxiliary_preserves_exact_base_task_parameters_and_rng():
    torch.manual_seed(287)
    base = BASE.Agent(_DummyEnvs(), _args(BASE))
    base_rng = torch.get_rng_state().clone()
    torch.manual_seed(287)
    gvf = GVF.Agent(_DummyEnvs(), _args(GVF))
    gvf_rng = torch.get_rng_state().clone()

    gvf_state = gvf.state_dict()
    for name, value in base.state_dict().items():
        torch.testing.assert_close(gvf_state[name], value, rtol=0.0, atol=0.0)
    assert torch.equal(gvf_rng, base_rng)


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_default_policy_forward_is_exact_tpomd_base_at_initialization(actor_dist):
    torch.manual_seed(917)
    base = BASE.Agent(_DummyEnvs(), _args(BASE, actor_dist))
    torch.manual_seed(917)
    gvf = GVF.Agent(_DummyEnvs(), _args(GVF, actor_dist))
    observations = torch.randn(11, 7)
    native_actions = (
        torch.full((11, 3), 0.37)
        if actor_dist == "beta"
        else torch.linspace(-0.8, 0.8, 33).reshape(11, 3)
    )
    candidates = native_actions[:, None].expand(-1, 8, -1).clone()
    torch.manual_seed(121)
    expected = base.get_action_and_value(
        observations, native_actions, candidate_zs=candidates
    )
    torch.manual_seed(121)
    actual = gvf.get_action_and_value(
        observations, native_actions, candidate_zs=candidates
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0.0, atol=0.0)


def test_target_snapshot_is_frozen_and_tracks_only_online_predictor():
    agent = GVF.Agent(_DummyEnvs(), _args(GVF))
    target_before = [
        parameter.detach().clone()
        for parameter in agent.target_gvf_predictor.parameters()
    ]
    with torch.no_grad():
        for parameter in agent.gvf_predictor.parameters():
            parameter.add_(0.25)
    assert any(
        not torch.equal(target, online)
        for target, online in zip(
            target_before, agent.gvf_predictor.parameters()
        )
    )
    agent.snapshot_gvf_target()
    for target, online in zip(
        agent.target_gvf_predictor.parameters(), agent.gvf_predictor.parameters()
    ):
        torch.testing.assert_close(target, online)
        assert not target.requires_grad


def test_task_and_private_optimizer_partitions_exclude_frozen_target():
    agent = GVF.Agent(_DummyEnvs(), _args(GVF))
    task_ids = {id(parameter) for parameter in agent.task_parameters()}
    predictor_ids = {id(parameter) for parameter in agent.gvf_predictor_parameters()}
    target_ids = {
        id(parameter) for parameter in agent.target_gvf_predictor.parameters()
    }
    trainable_ids = {
        id(parameter) for parameter in agent.parameters() if parameter.requires_grad
    }
    assert task_ids.isdisjoint(predictor_ids)
    assert task_ids | predictor_ids == trainable_ids
    assert target_ids.isdisjoint(task_ids | predictor_ids)
    assert all(
        not parameter.requires_grad
        for parameter in agent.target_gvf_predictor.parameters()
    )
    assert set(agent.gvf_parameters()) == set(agent.gvf_trunk_parameters()) | set(
        agent.gvf_predictor_parameters()
    )
    assert GVF.Args().gvf_trunk_grad_clip == 0.025
    assert GVF.Args().gvf_predictor_grad_clip == 0.25


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_policy_and_value_heads_are_diagnostic_only_for_gvf(actor_dist):
    agent = GVF.Agent(_DummyEnvs(), _args(GVF, actor_dist))
    source = agent.get_actor_feat(torch.randn(7, 7))
    action = torch.randn(7, 3).clamp(-1, 1)
    prediction = agent.predict_gvfs(source, action)
    (prediction - 1.0).square().mean().backward()

    heads = [agent.critic_head]
    if actor_dist == "gaussian":
        heads.extend((agent.actor_head, agent.actor_logvar_head))
    else:
        heads.extend((agent.actor_alpha_head, agent.actor_beta_head))
    assert all(
        parameter.grad is None for head in heads for parameter in head.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.norm() > 0
        for parameter in agent.gvf_predictor_parameters()
    )


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_frozen_policy_sample_is_persistent_replayable_and_rng_isolated(actor_dist):
    agent = GVF.Agent(_DummyEnvs(), _args(GVF, actor_dist))
    features = torch.randn(9, 8)
    rng_before = torch.get_rng_state().clone()
    first, first_cpu_state, first_device_state = (
        GVF.frozen_policy_sample_action_isolated(agent, features, seed=991)
    )
    assert torch.equal(rng_before, torch.get_rng_state())
    second, second_cpu_state, second_device_state = (
        GVF.frozen_policy_sample_action_isolated(
            agent,
            features,
            seed=991,
            cpu_rng_state=first_cpu_state,
            device_rng_state=first_device_state,
        )
    )
    assert torch.equal(rng_before, torch.get_rng_state())
    replay, replay_cpu_state, replay_device_state = (
        GVF.frozen_policy_sample_action_isolated(agent, features, seed=991)
    )

    torch.testing.assert_close(first, replay, rtol=0.0, atol=0.0)
    assert torch.equal(first_cpu_state, replay_cpu_state)
    assert first_device_state is replay_device_state is None
    assert second_device_state is None
    assert not torch.equal(first_cpu_state, second_cpu_state)
    assert not torch.equal(first, second)
    assert ((first >= -1.0) & (first <= 1.0)).all()
    assert ((second >= -1.0) & (second <= 1.0)).all()


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_sparse_next_state_tails_encode_predict_and_sample_only_selected_rows(
    monkeypatch, actor_dist
):
    agent = GVF.Agent(_DummyEnvs(), _args(GVF, actor_dist))
    with torch.no_grad():
        output_dim = agent.gvf_num_bands * agent.gvf_feature_dim
        agent.target_gvf_predictor[-1].bias.copy_(
            torch.linspace(0.1, 1.0, output_dim)
        )
    next_observations = torch.randn(4, 3, 7)
    bootstrap_rows = torch.tensor(
        [
            [False, True, False],
            [True, False, False],
            [False, False, True],
            [True, False, True],
        ]
    )
    selected_count = int(bootstrap_rows.sum())
    calls = {"encoder_rows": [], "predictor_rows": []}
    original_encoder = GVF.target_actor_feat_forward
    original_predictor = GVF.target_gvf_predictor_forward

    def tracked_encoder(model, observations):
        calls["encoder_rows"].append(observations.shape[0])
        return original_encoder(model, observations)

    def tracked_predictor(model, actor_features, actions):
        calls["predictor_rows"].append(actor_features.shape[0])
        return original_predictor(model, actor_features, actions)

    monkeypatch.setattr(GVF, "target_actor_feat_forward", tracked_encoder)
    monkeypatch.setattr(GVF, "target_gvf_predictor_forward", tracked_predictor)
    rng_before = torch.get_rng_state().clone()
    table, cpu_state, device_state = GVF.build_sparse_next_state_tail_table(
        agent,
        next_observations,
        bootstrap_rows,
        seed=811,
    )

    assert calls == {
        "encoder_rows": [selected_count],
        "predictor_rows": [selected_count],
    }
    assert torch.equal(rng_before, torch.get_rng_state())
    assert cpu_state is not None and device_state is None
    assert table.shape == (
        4,
        3,
        agent.gvf_num_bands,
        agent.gvf_feature_dim,
    )
    assert torch.count_nonzero(table[~bootstrap_rows]) == 0
    assert torch.count_nonzero(table[bootstrap_rows]) > 0


def test_sparse_next_state_tails_do_no_work_and_consume_no_rng_without_rows(
    monkeypatch,
):
    agent = GVF.Agent(_DummyEnvs(), _args(GVF))
    next_observations = torch.randn(2, 3, 7)
    bootstrap_rows = torch.zeros(2, 3, dtype=torch.bool)
    persistent_state = torch.get_rng_state().clone()

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("empty sparse table must not run a model")

    monkeypatch.setattr(GVF, "target_actor_feat_forward", unexpected_call)
    monkeypatch.setattr(GVF, "target_gvf_predictor_forward", unexpected_call)
    table, cpu_state, device_state = GVF.build_sparse_next_state_tail_table(
        agent,
        next_observations,
        bootstrap_rows,
        seed=12,
        cpu_rng_state=persistent_state,
    )
    assert torch.count_nonzero(table) == 0
    assert torch.equal(cpu_state, persistent_state)
    assert device_state is None


def test_breaker_detaches_only_trunk_and_predictor_still_updates():
    agent = GVF.Agent(_DummyEnvs(), _args(GVF))
    source = agent.get_actor_feat(torch.randn(7, 7))
    prediction = agent.predict_gvfs(
        GVF.gvf_source_feature(source, trunk_active=False), torch.randn(7, 3)
    )
    prediction.square().mean().backward()
    assert all(parameter.grad is None for parameter in agent.gvf_trunk_parameters())
    assert any(
        parameter.grad is not None for parameter in agent.gvf_predictor_parameters()
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_private_gradient_is_rejected_before_adam(bad):
    parameter = torch.nn.Parameter(torch.tensor([0.4, -0.3]))
    optimizer = torch.optim.Adam([parameter], lr=0.02)
    before = parameter.detach().clone()
    with pytest.raises(RuntimeError, match="non-finite"):
        GVF.apply_private_predictor_step(
            [parameter], optimizer, {parameter: torch.tensor([bad, 0.1])}
        )
    assert torch.equal(parameter, before)
    assert parameter not in optimizer.state


def test_tail_risk_summary_is_exact_and_does_not_touch_rng():
    returns = deque([100.0, 1000.0, 2000.0, 6000.0, 8000.0], maxlen=512)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    summary = GVF.summarize_episode_tail_risk(returns, (1500.0, 5000.0))

    assert summary == {
        "window_size": 5.0,
        "median": 2000.0,
        "bottom_5pct_mean": 100.0,
        "cvar_5pct": 100.0,
        "below_1500_count": 2.0,
        "below_1500_fraction": 0.4,
        "below_5000_count": 3.0,
        "below_5000_fraction": 0.6,
    }
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.get_rng_state())


def test_empty_tail_risk_window_still_emits_all_configured_metrics():
    summary = GVF.summarize_episode_tail_risk((), (100.0, 200.0))
    assert summary == {
        "window_size": 0.0,
        "median": 0.0,
        "bottom_5pct_mean": 0.0,
        "cvar_5pct": 0.0,
        "below_100_count": 0.0,
        "below_100_fraction": 0.0,
        "below_200_count": 0.0,
        "below_200_fraction": 0.0,
    }


@pytest.mark.parametrize(
    "name",
    [
        "IndexedTransferBranch",
        "ThinkBlock",
        "ThinkTrunk",
        "tpo_restricted_target",
        "tpo_reverse_kl",
        "policy_model_forward",
        "action_value_from_policy_outputs",
        "apply_union_optimizer_step",
    ],
)
def test_tpo_task_and_optimizer_chassis_remain_ast_identical_to_union_v17(name):
    assert _semantic_ast(getattr(GVF, name)) == _semantic_ast(getattr(UNION, name))


def test_cuda_compile_and_source_performance_guards():
    source = SCRIPT.read_text()
    main = source[source.index('if __name__ == "__main__":') :]
    assert "torch.set_float32_matmul_precision(\"high\")" in main
    assert "torch.compiler.cudagraph_mark_step_begin()" in main
    assert "fullgraph=True" in main
    assert "b_target_probs[mb_inds]" in main
    assert "epoch_inds = torch.as_tensor(b_inds, device=device)" in main
    assert "target_gvf_fn = torch.compile(" in main
    assert "gvf_update_fn = torch.compile(" in main
    assert main.count("gvf_update_fn(") == 3  # definition + live + one diagnostic
    assert main.count("gvf_gradient_telemetry(") == 1
    assert main.count("build_gvf_band_telemetry(") == 1
    telemetry_call = main.index("gvf_grad_telemetry = gvf_gradient_telemetry(")
    assert "epoch + 1 == args.update_epochs" in main[telemetry_call - 240 : telemetry_call]
    policy_call = main.index("update_outputs = policy_update_fn(")
    live_call = main.index("gvf_prediction = gvf_update_fn(")
    signflip_call = main.index("signflip_prediction = gvf_update_fn(")
    actor_backward = main.index(
        "(actor_loss - ent_coef_eff * entropy_loss).backward()"
    )
    assert "torch.compiler.cudagraph_mark_step_begin()" in main[policy_call - 160 : policy_call]
    # A minibatch is one graph-tree iteration. Starting another iteration while the
    # compiled policy output still has pending actor/critic backwards can invalidate it.
    assert "cudagraph_mark_step_begin" not in main[live_call - 180 : live_call]
    assert "cudagraph_mark_step_begin" not in main[signflip_call - 260 : signflip_call]
    assert live_call < actor_backward < signflip_call
    assert "retain_graph_output(" in main
    assert "predictive_trunk_optimizer" not in source
    assert "frozen_policy_deterministic_action" not in source
    assert "frozen_policy_sample_action_isolated(" in inspect.getsource(
        GVF.build_sparse_next_state_tail_table
    )
    assert (
        inspect.getsource(GVF.frozen_policy_sample_action_isolated).count(".sample()")
        == 1
    )
    assert "torch.random.fork_rng" in inspect.getsource(
        GVF.frozen_policy_sample_action_isolated
    )
    assert "gvf_aux_cpu_rng_state" in main
    assert "gvf_aux_cuda_rng_state" in main
    assert "gvf_next_state_rows = next_state_bootstrap_rows(" in main
    assert "build_sparse_next_state_tail_table(" in main
    assert "frozen_next_actor_feats" not in main
    assert "next_feature_chunks" not in main
    assert "ema_bootstrap_mass" not in source
    assert "residual_bootstrap_mass" not in source
    assert main.count("frozen_tail_bootstrap_mass=") == 1
    assert 'assert not args.clip_reward, "grounded reward cumulants require unclipped rewards"' in main
    assert "episode_return_window = deque(maxlen=args.tail_risk_window)" in main
    assert "tail_risk/bottom_5pct_mean" not in main  # dynamically namespaced, no training dependency
    assert "summarize_episode_tail_risk(" in main
