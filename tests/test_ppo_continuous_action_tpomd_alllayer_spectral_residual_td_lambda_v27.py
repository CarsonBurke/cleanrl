import importlib.util
import inspect
import sys
from pathlib import Path

import gymnasium as gym
import pytest
import torch
from torch.distributions import Normal


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "cleanrl/tpo/md/ppo_continuous_action_tpomd_alllayer_spectral_residual_td_lambda_v27.py"
)
V25_SCRIPT = ROOT / "cleanrl/tpo/md/ppo_continuous_action_tpomd_alllayer_residual_td_v25.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TAPE = _load("tpomd_alllayer_spectral_residual_td_lambda_v27", SCRIPT)
V25 = _load("tpomd_alllayer_residual_td_v25_reference", V25_SCRIPT)


class _DummyEnvs:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(7,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))


def _args(*, tape=True, actor_dist="beta"):
    return TAPE.Args(
        actor_dist=actor_dist,
        hidden=8,
        k_blocks=2,
        n_experts=2,
        num_bins=7,
        tape=tape,
        tape_period=16,
        tape_slot_embed_dim=3,
        tape_predictor_hidden=11,
    )


def _complex_rotate(value, omega):
    return value * torch.exp(-1j * omega)


def _independent_forward_view(
    innovations,
    predictions,
    edge_predictions,
    terminations,
    boundaries,
    valids,
    lam,
    rho,
    period=16,
):
    """Small scalar complex oracle intentionally independent of production packing."""
    time_dim, num_envs, num_layers, latent_dim = innovations.shape
    frequencies = 2 * torch.pi * torch.arange(period // 2 + 1) / period
    pred_modes = predictions[..., 1:, :].reshape(
        time_dim, num_envs, num_layers, period // 2 + 1, 2, latent_dim
    )
    pred_complex = pred_modes[..., 0, :].double() + 1j * pred_modes[..., 1, :].double()
    edge_modes = edge_predictions[..., 1:, :].reshape(
        -1, num_layers, period // 2 + 1, 2, latent_dim
    )
    edge_complex = edge_modes[..., 0, :].double() + 1j * edge_modes[..., 1, :].double()
    eta = innovations.double()
    valid = valids.bool()
    boundary = boundaries.bool()
    terminal = terminations.bool()
    edge_rows = TAPE.next_state_bootstrap_rows(terminations, boundaries, valids)
    edge_lookup = {}
    for compact, flat in enumerate(edge_rows.reshape(-1).nonzero().flatten().tolist()):
        edge_lookup[(flat // num_envs, flat % num_envs)] = edge_complex[compact]

    alpha = rho * torch.exp(-1j * frequencies.double())
    cache = {}

    def g(time, env, horizon):
        key = (time, env, horizon)
        if key in cache:
            return cache[key]
        injection = eta[time, env].unsqueeze(-2).expand(
            num_layers, frequencies.numel(), latent_dim
        ).to(torch.complex128)
        if not valid[time, env]:
            answer = torch.zeros_like(injection)
        elif terminal[time, env]:
            answer = injection
        else:
            ordinary = time + 1 < time_dim and not boundary[time, env]
            if ordinary:
                one = injection + alpha.reshape(1, -1, 1) * pred_complex[time + 1, env]
            else:
                one = injection + alpha.reshape(1, -1, 1) * edge_lookup[(time, env)]
            if horizon == 1 or not ordinary or not valid[time + 1, env]:
                answer = one
            else:
                answer = injection + alpha.reshape(1, -1, 1) * g(
                    time + 1, env, horizon - 1
                )
        cache[key] = answer
        return answer

    result = torch.zeros(
        time_dim,
        num_envs,
        num_layers,
        frequencies.numel(),
        latent_dim,
        dtype=torch.complex128,
    )
    for time in range(time_dim):
        for env in range(num_envs):
            if not valid[time, env]:
                continue
            for horizon in range(1, period + 1):
                weight = (
                    lam ** (period - 1)
                    if horizon == period
                    else (1 - lam) * lam ** (horizon - 1)
                )
                result[time, env] += weight * g(time, env, horizon)
    result[..., 0, :].imag.zero_()
    result[..., -1, :].imag.zero_()
    return result


def test_defaults_and_disabled_task_path_match_v25_exactly():
    defaults = TAPE.Args()
    assert defaults.env_id == "HalfCheetah-v4"
    assert defaults.total_timesteps == 8_000_000
    assert defaults.seed == 1 and defaults.cuda
    assert defaults.tape_period == 16
    assert defaults.tape_rho == 0.99
    assert defaults.tape_lambda == 0.95
    assert defaults.tape_trunk_grad_clip == 0.025
    assert defaults.tape_predictor_grad_clip == 0.25

    common = dict(hidden=8, k_blocks=2, n_experts=2, num_bins=7, actor_dist="beta")
    torch.manual_seed(913)
    reference = V25.Agent(_DummyEnvs(), V25.Args(tape=False, **common))
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(913)
    candidate = TAPE.Agent(_DummyEnvs(), TAPE.Args(tape=False, **common))
    candidate_rng = torch.get_rng_state().clone()
    assert torch.equal(candidate_rng, reference_rng)
    assert tuple(candidate.state_dict()) == tuple(reference.state_dict())
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], expected, rtol=0, atol=0)

    observations = torch.randn(6, 7)
    expected_outputs = V25.policy_model_forward(reference, observations)
    actual_outputs = TAPE.policy_model_forward(candidate, observations)
    for actual, expected in zip(actual_outputs, expected_outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_enabled_auxiliary_preserves_v25_task_initialization_and_rng_stream():
    common = dict(hidden=8, k_blocks=2, n_experts=2, num_bins=7, actor_dist="beta")
    torch.manual_seed(1771)
    reference = V25.Agent(
        _DummyEnvs(),
        V25.Args(
            tape=True,
            tape_horizon=16,
            tape_horizon_embed_dim=3,
            tape_predictor_hidden=11,
            **common,
        ),
    )
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(1771)
    candidate = TAPE.Agent(_DummyEnvs(), TAPE.Args(tape=True, **common))
    candidate_rng = torch.get_rng_state().clone()
    assert torch.equal(candidate_rng, reference_rng)
    for actual, expected in zip(
        candidate.task_parameters(), reference.task_parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    observations = torch.randn(6, 7)
    expected_outputs = V25.policy_model_forward(reference, observations)
    actual_outputs = TAPE.policy_model_forward(candidate, observations)
    for actual, expected in zip(actual_outputs, expected_outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_predictor_is_independent_per_layer_zero_initialized_and_endpoint_real():
    agent = TAPE.Agent(_DummyEnvs(), _args())
    observations = torch.randn(6, 7)
    features = agent.get_actor_feat(observations)
    assert agent.tape_layer_names == ("entry", "block_1", "block_2", "output")
    assert features.shape == (6, 4, 8)
    prediction = agent.predict_tapes(features, torch.randn(6, 3))
    assert prediction.shape == (6, 4, 19, 8)
    assert torch.count_nonzero(prediction) == 0
    assert len({id(module) for module in agent.tape_predictor.predictors}) == 4

    with torch.no_grad():
        for predictor in agent.tape_predictor.predictors:
            predictor.output.weight.normal_()
            predictor.output.bias.normal_()
    prediction = agent.predict_tapes(features, torch.randn(6, 3))
    assert torch.count_nonzero(prediction[..., 2, :]) == 0  # imag(k=0)
    assert torch.count_nonzero(prediction[..., 18, :]) == 0  # imag(k=8)
    assert torch.count_nonzero(prediction[..., 1, :]) > 0
    assert torch.count_nonzero(prediction[..., 3, :]) > 0


def test_complex_rotation_matches_exp_minus_i_omega_closed_form():
    generator = torch.Generator().manual_seed(41)
    modes = torch.randn(3, 9, 2, 5, generator=generator, dtype=torch.float64)
    frequencies = TAPE.spectral_frequencies(reference=modes)
    actual = TAPE.rotate_complex_modes(modes, frequencies)
    complex_modes = modes[..., 0, :] + 1j * modes[..., 1, :]
    expected_complex = complex_modes * torch.exp(
        -1j * frequencies.reshape(1, -1, 1)
    )
    expected = torch.stack((expected_complex.real, expected_complex.imag), dim=-2)
    expected[..., 0, 1, :] = 0
    expected[..., -1, 1, :] = 0
    torch.testing.assert_close(actual, expected)


def test_spectral_td_lambda_matches_independent_complex_forward_view():
    generator = torch.Generator().manual_seed(71)
    time, envs, layers, latent = 20, 2, 2, 2
    slots = TAPE.SPECTRAL_NUM_SLOTS
    innovations = torch.randn(time, envs, layers, latent, generator=generator)
    predictions = torch.randn(time, envs, layers, slots, latent, generator=generator)
    terminations = torch.zeros(time, envs)
    boundaries = torch.zeros(time, envs)
    boundaries[7, 0] = terminations[7, 0] = 1
    boundaries[11, 1] = 1
    valids = torch.ones(time, envs)
    edge_rows = TAPE.next_state_bootstrap_rows(terminations, boundaries, valids)
    edge = torch.randn(int(edge_rows.sum()), layers, slots, latent, generator=generator)
    actual = TAPE.build_spectral_td_lambda_targets(
        innovations,
        predictions,
        edge,
        terminations,
        boundaries,
        valids,
        0.63,
        0.91,
    )
    expected = _independent_forward_view(
        innovations,
        predictions,
        edge,
        terminations,
        boundaries,
        valids,
        0.63,
        0.91,
    )
    h1, modes = TAPE.split_spectral_slots(actual.values)
    actual_complex = modes[..., 0, :].double() + 1j * modes[..., 1, :].double()
    torch.testing.assert_close(h1, innovations)
    torch.testing.assert_close(actual_complex, expected, rtol=2e-5, atol=2e-5)
    assert not actual.values.requires_grad


def test_terminal_truncation_rollout_edge_and_missing_final_semantics():
    slots = TAPE.SPECTRAL_NUM_SLOTS
    eta = torch.full((1, 1, 1, 1), 2.0)
    prediction = torch.full((1, 1, 1, slots, 1), 17.0)
    valid = torch.ones(1, 1)

    terminal = TAPE.build_spectral_td_lambda_targets(
        eta,
        prediction,
        prediction.new_zeros((0, 1, slots, 1)),
        torch.ones(1, 1),
        torch.ones(1, 1),
        valid,
        0.95,
        0.99,
    )
    terminal_h1, terminal_modes = TAPE.split_spectral_slots(terminal.values)
    torch.testing.assert_close(terminal_h1, eta)
    torch.testing.assert_close(terminal_modes[..., 0, :], eta.unsqueeze(-2).expand_as(terminal_modes[..., 0, :]))
    assert torch.count_nonzero(terminal_modes[..., 1, :]) == 0

    edge = torch.zeros(1, 1, slots, 1)
    edge[..., 1, :] = 3.0
    rollout_edge = TAPE.build_spectral_td_lambda_targets(
        eta,
        prediction,
        edge,
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        valid,
        0.95,
        0.5,
    )
    _, rollout_modes = TAPE.split_spectral_slots(rollout_edge.values)
    torch.testing.assert_close(rollout_modes[..., 0, 0, :], torch.full((1, 1, 1, 1), 3.5))

    truncation = TAPE.build_spectral_td_lambda_targets(
        eta,
        prediction,
        edge,
        torch.zeros(1, 1),
        torch.ones(1, 1),
        valid,
        0.95,
        0.5,
    )
    torch.testing.assert_close(truncation.values, rollout_edge.values)

    eta2 = torch.tensor([1.0, 99.0]).reshape(2, 1, 1, 1)
    pred2 = torch.zeros(2, 1, 1, slots, 1)
    pred2[1, ..., 1, :] = 4.0
    missing = TAPE.build_spectral_td_lambda_targets(
        eta2,
        pred2,
        pred2.new_zeros((0, 1, slots, 1)),
        torch.tensor([[0.0], [1.0]]),
        torch.tensor([[0.0], [1.0]]),
        torch.tensor([[1.0], [0.0]]),
        1.0,
        0.5,
    )
    _, missing_modes = TAPE.split_spectral_slots(missing.values)
    torch.testing.assert_close(missing_modes[0, ..., 0, 0, :], torch.full((1, 1, 1), 3.0))
    assert torch.count_nonzero(missing.values[1]) == 0
    assert not missing.masks[1].any()


def test_targets_detach_every_temporal_and_snapshot_input():
    time, envs, layers, latent = 4, 1, 2, 3
    slots = TAPE.SPECTRAL_NUM_SLOTS
    innovations = torch.randn(time, envs, layers, latent, requires_grad=True)
    predictions = torch.randn(time, envs, layers, slots, latent, requires_grad=True)
    edge = torch.randn(envs, layers, slots, latent, requires_grad=True)
    targets = TAPE.build_spectral_td_lambda_targets(
        innovations,
        predictions,
        edge,
        torch.zeros(time, envs),
        torch.zeros(time, envs),
        torch.ones(time, envs),
        0.95,
        0.99,
    )
    assert targets.values.grad_fn is None
    assert not targets.values.requires_grad
    assert not targets.masks.requires_grad


def test_loss_is_scale_invariant_equal_family_and_masks_endpoint_imaginary_slots():
    generator = torch.Generator().manual_seed(103)
    batch, layers, latent = 7, 3, 4
    slots = TAPE.SPECTRAL_NUM_SLOTS
    target = torch.randn(batch, layers, slots, latent, generator=generator)
    prediction = torch.randn(batch, layers, slots, latent, generator=generator)
    active = TAPE.spectral_slot_mask().reshape(1, 1, slots)
    mask = active.expand(batch, layers, slots)
    latent_rms = torch.tensor([0.5, 2.0, 10.0])
    loss, h1_scale, mode_scale, h1_losses, mode_losses = TAPE.spectral_td_lambda_loss(
        prediction,
        target,
        mask,
        latent_rms=latent_rms,
        scale_floor_ratio=1e-3,
    )
    scaled = TAPE.spectral_td_lambda_loss(
        prediction * 100,
        target * 100,
        mask,
        latent_rms=latent_rms * 100,
        scale_floor_ratio=1e-3,
    )
    torch.testing.assert_close(scaled[0], loss)
    torch.testing.assert_close(scaled[1], h1_scale * 100)
    torch.testing.assert_close(scaled[2], mode_scale * 100)
    torch.testing.assert_close(loss, 0.5 * (h1_losses.mean() + mode_losses.mean()))

    poisoned_target = target.clone()
    poisoned_prediction = prediction.clone()
    poisoned_target[..., 2, :] = float("nan")
    poisoned_target[..., 18, :] = float("nan")
    poisoned_prediction[..., 2, :] = float("nan")
    poisoned_prediction[..., 18, :] = float("nan")
    poisoned = TAPE.spectral_td_lambda_loss(
        poisoned_prediction,
        poisoned_target,
        mask,
        latent_rms=latent_rms,
        scale_floor_ratio=1e-3,
    )
    torch.testing.assert_close(poisoned[0], loss)


def test_target_only_scales_match_loss_and_compile_as_one_fullgraph():
    batch, layers, latent = 9, 3, 4
    target = torch.randn(batch, layers, TAPE.SPECTRAL_NUM_SLOTS, latent)
    prediction = torch.randn_like(target)
    mask = TAPE.spectral_slot_mask().reshape(1, 1, -1).expand(*target.shape[:-1])
    latent_rms = torch.tensor([0.5, 1.0, 2.0])
    expected_scales = TAPE.spectral_target_scales(
        target,
        mask,
        latent_rms=latent_rms,
        scale_floor_ratio=1e-3,
    )
    implicit = TAPE.spectral_td_lambda_loss(
        prediction,
        target,
        mask,
        latent_rms=latent_rms,
        scale_floor_ratio=1e-3,
    )
    explicit = TAPE.spectral_td_lambda_loss(
        prediction,
        target,
        mask,
        latent_rms=latent_rms,
        scale_floor_ratio=1e-3,
        h1_scale=expected_scales[0],
        mode_scale=expected_scales[1],
    )
    torch.testing.assert_close(implicit[0], explicit[0])
    torch.testing.assert_close(implicit[1], expected_scales[0])
    torch.testing.assert_close(implicit[2], expected_scales[1])

    compiled = torch.compile(
        lambda target_, mask_, latent_: TAPE.spectral_target_scales(
            target_,
            mask_,
            latent_rms=latent_,
            scale_floor_ratio=1e-3,
        ),
        backend="inductor",
        dynamic=False,
        fullgraph=True,
    )
    actual_scales = compiled(target, mask, latent_rms)
    for actual, expected in zip(actual_scales, expected_scales, strict=True):
        torch.testing.assert_close(actual, expected)


def test_m_step_bootstrap_oracle_includes_rotated_frozen_tail_and_masks_partial_paths():
    time, envs, layers, latent = 7, 1, 1, 1
    innovations = torch.arange(1.0, time + 1).reshape(time, envs, layers, latent)
    modes = torch.zeros(time, envs, layers, 9, 2, latent)
    modes[3, ..., 0, 0, :] = 10.0
    modes[3, ..., 1, 0, :] = 4.0
    boundaries = torch.zeros(time, envs)
    valids = torch.ones(time, envs)
    rows = torch.tensor([0, 5], dtype=torch.long)
    rho = 0.8
    actual, mask, steps = TAPE.build_m_step_bootstrapped_spectral_returns(
        innovations,
        modes,
        boundaries,
        valids,
        rows,
        max_steps=3,
        rho=rho,
    )
    frequencies = TAPE.spectral_frequencies(reference=innovations)
    for frequency in range(9):
        omega = frequencies[frequency]
        prefix = sum(
            rho**delay
            * _complex_rotate(innovations[delay, 0, 0, 0].to(torch.complex64), omega * delay)
            for delay in range(3)
        )
        tail = modes[3, 0, 0, frequency, 0, 0].to(torch.complex64)
        tail += 1j * modes[3, 0, 0, frequency, 1, 0].to(torch.complex64)
        expected = prefix + rho**3 * _complex_rotate(tail, omega * 3)
        if frequency in (0, 8):
            expected = expected.real.to(torch.complex64)
        actual_complex = actual[0, 0, frequency, 0, 0] + 1j * actual[0, 0, frequency, 1, 0]
        torch.testing.assert_close(actual_complex, expected)
    assert mask.tolist() == [True, False]
    assert steps.tolist() == [3, 2]

    boundaries[1] = 1
    _, boundary_mask, boundary_steps = TAPE.build_m_step_bootstrapped_spectral_returns(
        innovations,
        modes,
        boundaries,
        valids,
        torch.tensor([0]),
        max_steps=3,
        rho=rho,
    )
    assert not boundary_mask.item()
    assert boundary_steps.item() == 2


def test_diagnostic_rows_are_deterministic_and_stratify_all_default_envs():
    rows = TAPE.deterministic_diagnostic_rows(32_768, 1_024, 1, torch.device("cpu"))
    repeated = TAPE.deterministic_diagnostic_rows(
        32_768, 1_024, 1, torch.device("cpu")
    )
    rotated = TAPE.deterministic_diagnostic_rows(
        32_768, 1_024, 2, torch.device("cpu")
    )
    assert torch.equal(rows, repeated)
    assert not torch.equal(rows, rotated)
    assert rows.unique().numel() == rows.numel()
    assert torch.equal(rows.remainder(16).unique().sort().values, torch.arange(16))


def test_output_layer_alone_drives_frozen_policy_edge_actions_without_rng_drift():
    class EdgeAgent:
        tape_num_layers = 3
        tape_feature_dim = 2
        tape_action_dim = 1
        actor_dist = "gaussian"

        def __init__(self):
            self.seen = None

        def _target_actor_dist(self, features):
            self.seen = features.detach().clone()
            return Normal(features[:, :1], torch.ones_like(features[:, :1]) * 1e-6), lambda x: x

    agent = EdgeAgent()
    features = torch.arange(1.0, 4 * 2 * 3 * 2 + 1).reshape(4, 2, 3, 2)
    rows = torch.tensor([[False, True], [True, False], [False, False], [True, True]])
    before = torch.get_rng_state().clone()
    actions, _, _ = TAPE.build_sparse_frozen_next_action_table(agent, features, rows, seed=17)
    assert torch.equal(before, torch.get_rng_state())
    torch.testing.assert_close(agent.seen, features[..., -1, :][rows])
    assert torch.count_nonzero(actions[~rows]) == 0


def test_loss_reaches_every_trunk_stage_and_breaker_is_predictor_only():
    agent = TAPE.Agent(_DummyEnvs(), _args())
    with torch.no_grad():
        for predictor in agent.tape_predictor.predictors:
            predictor.output.weight.normal_(0.0, 0.01)
    observations = torch.randn(7, 7)
    features = agent.get_actor_feat(observations)
    prediction = agent.predict_tapes(features, torch.randn(7, 3))
    target = torch.randn_like(prediction)
    mask = TAPE.spectral_slot_mask().reshape(1, 1, -1).expand(*prediction.shape[:-1])
    loss = TAPE.spectral_td_lambda_loss(
        prediction,
        target,
        mask,
        latent_rms=torch.ones(agent.tape_num_layers),
        scale_floor_ratio=1e-3,
    )[0]
    loss.backward()
    for block in agent.tape_trunk_blocks():
        assert any(
            parameter.grad is not None and parameter.grad.norm() > 0 for parameter in block
        )
    assert all(
        any(parameter.grad is not None for parameter in predictor.parameters())
        for predictor in agent.tape_predictor.predictors
    )

    agent.zero_grad(set_to_none=True)
    detached = TAPE.tape_source_feature(agent.get_actor_feat(torch.randn(7, 7)), trunk_active=False)
    agent.predict_tapes(detached, torch.randn(7, 3)).square().mean().backward()
    assert all(parameter.grad is None for parameter in agent.tape_trunk_parameters())
    assert any(parameter.grad is not None for parameter in agent.tape_predictor_parameters())


def test_optimizer_partition_global_caps_and_private_predictor_moments():
    agent = TAPE.Agent(_DummyEnvs(), _args())
    task = set(agent.task_parameters())
    predictor = set(agent.tape_predictor_parameters())
    trainable = {parameter for parameter in agent.parameters() if parameter.requires_grad}
    assert task.isdisjoint(predictor)
    assert task | predictor == trainable
    assert set(agent.tape_trunk_parameters()) == set(agent.trunk.parameters())

    blocks = agent.tape_trunk_blocks()
    for parameter in agent.tape_trunk_parameters():
        parameter.grad = torch.ones_like(parameter)
    for parameter in predictor:
        parameter.grad = torch.ones_like(parameter)
    raw, predictor_raw, block_raw, block_delivered, trunk_grads, predictor_grads = (
        TAPE.capture_tape_gradient_groups(
            blocks,
            list(predictor),
            trunk_active=True,
            trunk_max_norm=0.025,
            predictor_max_norm=0.25,
        )
    )
    assert raw > 0.025 and predictor_raw > 0.25
    assert torch.all(block_delivered <= block_raw)
    delivered = torch.stack([value.square().sum() for value in trunk_grads.values()]).sum().sqrt()
    private = torch.stack([value.square().sum() for value in predictor_grads.values()]).sum().sqrt()
    assert delivered <= 0.025001
    assert private <= 0.25001


def test_hard_snapshot_and_compiled_output_retention_are_exact():
    agent = TAPE.Agent(_DummyEnvs(), _args())
    with torch.no_grad():
        for parameter in agent.tape_predictor.parameters():
            parameter.add_(0.25)
        for parameter in agent.actor_parameters():
            parameter.add_(0.1)
    assert agent.target_snapshot_lag() > 0
    agent.snapshot_tape_target()
    assert agent.target_snapshot_lag().item() == 0.0
    assert all(
        not parameter.requires_grad
        for module in agent.target_bundle_modules()
        for parameter in module.parameters()
    )

    features = torch.randn(6, agent.tape_num_layers, agent.tape_feature_dim)
    actions = torch.randn(6, agent.tape_action_dim)
    compiled = torch.compile(
        lambda feature, action: TAPE.tape_predictor_forward(agent, feature, action),
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


def test_combined_predictor_and_loss_is_one_real_cpu_fullgraph():
    agent = TAPE.Agent(_DummyEnvs(), _args())
    with torch.no_grad():
        for predictor in agent.tape_predictor.predictors:
            predictor.output.weight.normal_(0.0, 0.01)
    batch = 5
    features = torch.randn(batch, agent.tape_num_layers, agent.tape_feature_dim)
    actions = torch.randn(batch, agent.tape_action_dim)
    target = torch.randn(batch, agent.tape_num_layers, agent.tape_num_slots, agent.tape_feature_dim)
    mask = TAPE.spectral_slot_mask().reshape(1, 1, -1).expand(*target.shape[:-1])
    latent = torch.ones(agent.tape_num_layers)
    h1_scale = torch.ones(agent.tape_num_layers)
    mode_scale = torch.ones(agent.tape_num_layers, TAPE.SPECTRAL_NUM_FREQUENCIES)

    def combined(feature, action, target_, mask_, latent_, h1_scale_, mode_scale_):
        return TAPE.tape_auxiliary_update_forward(
            agent,
            feature,
            action,
            target_,
            mask_,
            latent_,
            h1_scale_,
            mode_scale_,
            scale_floor_ratio=1e-3,
            period=16,
        )

    expected = combined(
        features, actions, target, mask, latent, h1_scale, mode_scale
    )
    compiled = torch.compile(combined, backend="inductor", dynamic=False, fullgraph=True)
    actual = compiled(
        features, actions, target, mask, latent, h1_scale, mode_scale
    )
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value)
    actual[0].backward()
    assert any(parameter.grad is not None for parameter in agent.tape_predictor_parameters())


def test_source_contains_no_counterfactual_target_or_temporal_hot_loop():
    target_source = inspect.getsource(TAPE.build_spectral_td_lambda_targets).lower()
    m_step_source = inspect.getsource(TAPE.build_m_step_bootstrapped_spectral_returns).lower()
    full_source = SCRIPT.read_text()
    assert "counterfactual" not in full_source.lower()
    assert "intervention" not in full_source.lower()
    assert "tpo_" not in target_source
    assert "probe" not in target_source
    assert ".item()" not in target_source
    assert ".item()" not in m_step_source
    assert "range(2, period + 1)" in target_source
    assert "torch.lerp(one_step_tail, shifted, lam)" in target_source
    assert "mixed_modes = mixed_modes +" not in target_source
    assert "reversed(range" not in target_source
    assert "build_all_layer_horizon_td_targets" not in full_source
    assert "dynamic=False" in full_source
    assert 'f"{prefix}_nmse"' in full_source
    assert 'f"{prefix}_cosine"' in full_source
    for target_alias in (
        "tape_targets",
        "flat_tape_targets",
        "flat_tape_mask",
        "b_tape_targets",
        "b_tape_mask",
    ):
        assert target_alias in full_source.split("# These names alias", 1)[1]
    assert "torch.zeros_like(tape_targets.values)" not in full_source
    assert "diagnostic_target = flat_tape_targets.index_select" in full_source
    assert "tape_auxiliary_update_forward(" in full_source
    assert "tape_target_scale_fn = torch.compile(" in full_source


@pytest.mark.parametrize("slot", [2, 18])
def test_inactive_endpoint_slot_never_changes_loss(slot):
    batch, layers, latent = 3, 2, 4
    target = torch.randn(batch, layers, TAPE.SPECTRAL_NUM_SLOTS, latent)
    prediction = torch.randn_like(target)
    mask = TAPE.spectral_slot_mask().reshape(1, 1, -1).expand(*target.shape[:-1])
    kwargs = dict(
        latent_rms=torch.ones(layers),
        scale_floor_ratio=1e-3,
    )
    reference = TAPE.spectral_td_lambda_loss(prediction, target, mask, **kwargs)[0]
    target[..., slot, :] = 1e20
    prediction[..., slot, :] = -1e20
    actual = TAPE.spectral_td_lambda_loss(prediction, target, mask, **kwargs)[0]
    torch.testing.assert_close(actual, reference)
