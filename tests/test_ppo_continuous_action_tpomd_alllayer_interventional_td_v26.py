import importlib.util
import sys
from pathlib import Path

import gymnasium as gym
import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Normal


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "cleanrl/tpo/md/ppo_continuous_action_tpomd_alllayer_interventional_td_v26.py"
V25_SCRIPT = ROOT / "cleanrl/tpo/md/ppo_continuous_action_tpomd_alllayer_residual_td_v25.py"
V24_SCRIPT = ROOT / "cleanrl/tpo/md/ppo_continuous_action_tpomd_horizon_td_lambda_tape_v24.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TAPE = _load("tpomd_alllayer_interventional_td_v26", SCRIPT)
V25 = _load("tpomd_alllayer_residual_td_v25_reference", V25_SCRIPT)
V24 = _load("tpomd_horizon_td_lambda_tape_v24_reference", V24_SCRIPT)


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
        tape_horizon=4,
        tape_horizon_embed_dim=3,
        tape_predictor_hidden=11,
    )


def test_defaults_and_disabled_task_path_match_v24_exactly():
    defaults = TAPE.Args()
    assert defaults.env_id == "HalfCheetah-v4"
    assert defaults.total_timesteps == 8_000_000
    assert defaults.seed == 1 and defaults.cuda
    assert defaults.tape_horizon == 16
    assert defaults.tape_lambda == 0.95
    assert defaults.tape_trunk_grad_clip == 0.025
    direct, bootstrap = TAPE.td_lambda_mixture_weights(
        defaults.tape_horizon, defaults.tape_lambda
    )
    torch.testing.assert_close(direct + bootstrap, torch.ones(16))
    assert direct[-1] < bootstrap[-1]

    common = dict(hidden=8, k_blocks=2, n_experts=2, num_bins=7, actor_dist="beta")
    torch.manual_seed(913)
    reference = V24.Agent(_DummyEnvs(), V24.Args(tape=False, **common))
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(913)
    candidate = TAPE.Agent(_DummyEnvs(), TAPE.Args(tape=False, **common))
    candidate_rng = torch.get_rng_state().clone()
    assert torch.equal(candidate_rng, reference_rng)
    assert tuple(candidate.state_dict()) == tuple(reference.state_dict())
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], expected, rtol=0, atol=0)

    observations = torch.randn(6, 7)
    expected_outputs = V24.policy_model_forward(reference, observations)
    actual_outputs = TAPE.policy_model_forward(candidate, observations)
    for actual, expected in zip(actual_outputs, expected_outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_every_trunk_representation_has_an_independent_zero_residual_predictor():
    agent = TAPE.Agent(_DummyEnvs(), _args())
    observations = torch.randn(6, 7)
    features = agent.get_actor_feat(observations)
    assert agent.tape_layer_names == ("entry", "block_1", "block_2", "output")
    assert features.shape == (6, 4, 8)
    torch.testing.assert_close(features[:, -1], agent.trunk(observations))

    prediction = agent.predict_tapes(features, torch.randn(6, 3))
    assert prediction.shape == (6, 4, 4, 8)
    assert torch.count_nonzero(prediction) == 0
    assert len(agent.tape_predictor.predictors) == 4
    assert len({id(predictor) for predictor in agent.tape_predictor.predictors}) == 4
    task = set(agent.task_parameters())
    predictor = set(agent.tape_predictor_parameters())
    trainable = {parameter for parameter in agent.parameters() if parameter.requires_grad}
    assert task.isdisjoint(predictor)
    assert task | predictor == trainable
    assert set(agent.tape_trunk_parameters()) == set(agent.trunk.parameters())


def test_all_layer_predictor_is_one_static_fullgraph_program():
    agent = TAPE.Agent(_DummyEnvs(), _args())
    with torch.no_grad():
        for predictor in agent.tape_predictor.predictors:
            predictor.output.weight.normal_(0.0, 0.01)
    features = torch.randn(6, agent.tape_num_layers, agent.tape_feature_dim)
    actions = torch.randn(6, agent.tape_action_dim)
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


def test_all_layer_target_layout_matches_independent_td_recurrences():
    generator = torch.Generator().manual_seed(71)
    time, envs, layers, horizon, latent = 6, 2, 3, 4, 2
    innovations = torch.randn(time, envs, layers, latent, generator=generator)
    predictions = torch.randn(time, envs, layers, horizon, latent, generator=generator)
    terminations = torch.zeros(time, envs)
    boundaries = torch.zeros(time, envs)
    boundaries[2, 0] = 1
    terminations[2, 0] = 1
    valids = torch.ones(time, envs)
    edge_rows = TAPE.next_state_bootstrap_rows(terminations, boundaries, valids)
    edge = torch.randn(
        int(edge_rows.sum()), layers, horizon, latent, generator=generator
    )

    actual = TAPE.build_all_layer_horizon_td_targets(
        innovations,
        predictions,
        edge,
        terminations,
        boundaries,
        valids,
        0.63,
    )
    assert actual.values.shape == (time, envs, layers, horizon, latent)
    assert actual.masks.shape == (time, envs, layers, horizon)
    for layer in range(layers):
        expected = V24.build_horizon_td_lambda_tape_targets(
            innovations[:, :, layer],
            predictions[:, :, layer],
            edge[:, layer],
            terminations,
            boundaries,
            valids,
            0.63,
        )
        torch.testing.assert_close(actual.values[:, :, layer], expected.values)
        assert torch.equal(actual.masks[:, :, layer], expected.masks)
    torch.testing.assert_close(actual.values[..., 0, :], innovations)
    assert not actual.values.requires_grad


def test_layerwise_units_prevent_easy_or_large_layers_from_diluting_loss():
    target = torch.tensor(
        [
            [
                [[1.0, -1.0], [2.0, -2.0]],
                [[100.0, -100.0], [200.0, -200.0]],
            ],
            [
                [[3.0, -3.0], [4.0, -4.0]],
                [[300.0, -300.0], [400.0, -400.0]],
            ],
        ]
    )
    prediction = torch.zeros_like(target, requires_grad=True)
    mask = torch.tensor(
        [
            [[True, True], [True, True]],
            [[True, False], [True, False]],
        ]
    )
    loss, scale, per_layer_head = TAPE.horizon_td_tape_loss(
        prediction,
        target,
        mask,
        latent_rms=torch.tensor([10.0, 1_000.0]),
    )
    expected_scale = torch.stack(
        [target[:, layer, 0][mask[:, layer, 0]].square().mean().sqrt() for layer in range(2)]
    )
    torch.testing.assert_close(scale, expected_scale)
    manual = F.smooth_l1_loss(
        prediction / scale.view(1, 2, 1, 1),
        target / scale.view(1, 2, 1, 1),
        reduction="none",
    )
    expected_terms = torch.stack(
        [
            manual[:, layer, head][mask[:, layer, head]].mean()
            for layer in range(2)
            for head in range(2)
        ]
    ).reshape(2, 2)
    torch.testing.assert_close(per_layer_head, expected_terms)
    torch.testing.assert_close(loss, expected_terms.mean())

    poisoned_prediction = prediction.detach().clone()
    poisoned_target = target.clone()
    poisoned_prediction[~mask] = float("nan")
    poisoned_target[~mask] = float("nan")
    poisoned_loss, _, poisoned_terms = TAPE.horizon_td_tape_loss(
        poisoned_prediction, poisoned_target, mask, scale=scale
    )
    torch.testing.assert_close(poisoned_loss, loss)
    torch.testing.assert_close(poisoned_terms, per_layer_head)


def test_output_layer_alone_drives_frozen_policy_edge_actions():
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
    rows = torch.tensor(
        [[False, True], [True, False], [False, False], [True, True]]
    )
    before = torch.get_rng_state().clone()
    actions, _, _ = TAPE.build_sparse_frozen_next_action_table(
        agent, features, rows, seed=17
    )
    assert torch.equal(before, torch.get_rng_state())
    torch.testing.assert_close(agent.seen, features[..., -1, :][rows])
    assert torch.count_nonzero(actions[~rows]) == 0
    torch.testing.assert_close(actions[rows], features[..., -1, :][rows, :1], atol=1e-5, rtol=0)


def test_loss_reaches_every_trunk_stage_and_breaker_detaches_all_sources():
    agent = TAPE.Agent(_DummyEnvs(), _args())
    with torch.no_grad():
        for predictor in agent.tape_predictor.predictors:
            predictor.output.weight.normal_(0.0, 0.01)
    observations = torch.randn(7, 7)
    features = agent.get_actor_feat(observations)
    prediction = agent.predict_tapes(features, torch.randn(7, 3))
    target = torch.randn_like(prediction)
    mask = torch.ones(prediction.shape[:-1], dtype=torch.bool)
    loss, _, _ = TAPE.horizon_td_tape_loss(
        prediction, target, mask, scale=torch.ones(agent.tape_num_layers)
    )
    loss.backward()
    for block in agent.tape_trunk_blocks():
        assert any(
            parameter.grad is not None and parameter.grad.norm() > 0
            for parameter in block
        )
    for predictor in agent.tape_predictor.predictors:
        assert any(
            parameter.grad is not None and parameter.grad.norm() > 0
            for parameter in predictor.parameters()
        )

    agent.zero_grad(set_to_none=True)
    detached = TAPE.tape_source_feature(
        agent.get_actor_feat(torch.randn(7, 7)), trunk_active=False
    )
    agent.predict_tapes(detached, torch.randn(7, 3)).square().mean().backward()
    assert all(parameter.grad is None for parameter in agent.tape_trunk_parameters())
    assert any(parameter.grad is not None for parameter in agent.tape_predictor_parameters())


def test_one_global_trunk_cap_preserves_block_diagnostics_without_multiplying_budget():
    agent = TAPE.Agent(_DummyEnvs(), _args())
    blocks = agent.tape_trunk_blocks()
    predictor_parameters = agent.tape_predictor_parameters()
    for parameter in agent.tape_trunk_parameters():
        parameter.grad = torch.ones_like(parameter)
    for parameter in predictor_parameters:
        parameter.grad = torch.ones_like(parameter)

    (
        raw_trunk_norm,
        _,
        block_raw,
        block_delivered,
        trunk_gradients,
        _,
    ) = TAPE.capture_tape_gradient_groups(
        blocks,
        predictor_parameters,
        trunk_active=True,
        trunk_max_norm=0.025,
        predictor_max_norm=0.25,
    )
    assert raw_trunk_norm > 0.025
    assert torch.all(block_delivered <= block_raw)
    delivered_norm = torch.stack(
        [gradient.float().square().sum() for gradient in trunk_gradients.values()]
    ).sum().sqrt()
    torch.testing.assert_close(delivered_norm, block_delivered.square().sum().sqrt())
    assert delivered_norm <= 0.025001


def test_hard_snapshot_covers_all_layer_predictors_and_stays_frozen():
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
    observations = torch.randn(5, 7)
    frozen_features = agent.target_actor_feature(observations).clone()
    frozen_predictions = agent.predict_target_tapes(
        frozen_features, torch.randn(5, 3)
    ).clone()
    optimizer = torch.optim.Adam(agent.task_parameters(), lr=1e-3)
    agent.get_actor_feat(torch.randn(5, 7)).square().mean().backward()
    optimizer.step()
    torch.testing.assert_close(agent.target_actor_feature(observations), frozen_features)
    assert frozen_predictions.shape == (5, 4, 4, 8)


def test_intervention_contrast_and_rotating_query_oracle():
    features = torch.tensor(
        [
            [[[1.0], [2.0]], [[4.0], [8.0]], [[7.0], [14.0]]],
            [[[2.0], [3.0]], [[5.0], [9.0]], [[11.0], [15.0]]],
        ]
    )
    query = torch.tensor([0, 2])
    expected = torch.tensor(
        [[[(-4.5)], [(-9.0)]], [[7.5], [9.0]]]
    )
    actual = TAPE.reduce_candidate_contrast(features, query)
    torch.testing.assert_close(actual, expected)

    rows = torch.arange(12)
    q = TAPE.intervention_query_indices(rows, 4, iteration=3)
    assert torch.equal(q, torch.tensor([2, 3, 0, 1] * 3))
    assert torch.bincount(q, minlength=4).tolist() == [3, 3, 3, 3]
    before = torch.get_rng_state().clone()
    TAPE.intervention_query_indices(rows, 4, iteration=3)
    assert torch.equal(before, torch.get_rng_state())

    values = torch.arange(5.0)
    grouped = values.view(1, 5, 1, 1).expand(10, -1, 1, 1)
    queries = TAPE.intervention_query_indices(10, 5, iteration=1)
    contrast = TAPE.reduce_candidate_contrast(grouped, queries).flatten()
    assert torch.equal(queries, torch.tensor([0, 1, 2, 3, 4] * 2))
    # Query-minus-other is K/(K-1) times the centered candidate value.
    centered = values - values.mean()
    expected_variance = (5.0 / 4.0) ** 2 * centered.square().mean()
    torch.testing.assert_close(contrast.square().mean(), expected_variance)

    offset = torch.randn(10, 1, 1, 1)
    torch.testing.assert_close(
        TAPE.reduce_candidate_contrast(grouped + offset, queries),
        TAPE.reduce_candidate_contrast(grouped, queries),
    )


def test_intervention_chunk_offset_and_candidate_permutation_invariance():
    torch.manual_seed(19)
    flat = torch.randn(12, 3, 5)
    full, q_full = TAPE.build_intervention_target_contrast(flat, 3, 4)
    first, q_first = TAPE.build_intervention_target_contrast(flat[:6], 3, 4)
    second, q_second = TAPE.build_intervention_target_contrast(
        flat[6:], 3, 4, global_row_offset=2
    )
    torch.testing.assert_close(torch.cat((first, second)), full)
    assert torch.equal(torch.cat((q_first, q_second)), q_full)

    permutation = torch.tensor([2, 0, 1])
    permuted = flat.reshape(4, 3, 3, 5)[:, permutation]
    inverse = torch.argsort(permutation)
    permuted_target = TAPE.reduce_candidate_contrast(
        permuted,
        inverse[q_full],
    )
    torch.testing.assert_close(permuted_target, full)
    assert inverse.tolist() == [1, 2, 0]


def test_intervention_scale_and_nonfinite_guards():
    target = torch.tensor(
        [[[0.0, 0.0], [2.0, -2.0]], [[0.0, 0.0], [4.0, -4.0]]]
    )
    scale = TAPE.intervention_contrast_scale(
        target, latent_rms=torch.tensor([5.0, 3.0]), floor_ratio=0.1
    )
    torch.testing.assert_close(scale, torch.tensor([0.5, 3.1622777]))
    with pytest.raises((FloatingPointError, RuntimeError)):
        TAPE.intervention_contrast_scale(target.clone().fill_(float("nan")))
    with pytest.raises(ValueError):
        TAPE.build_intervention_target_contrast(torch.randn(5, 3, 2), 3, 1)


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_specialized_h1_value_and_gradient_parity(actor_dist):
    predictor = TAPE.HorizonTapePredictor(8, 3, 4, 5, 11)
    with torch.no_grad():
        predictor.output.weight.normal_(0.0, 0.01)
        predictor.output.bias.normal_(0.0, 0.01)
    feature = torch.randn(4, 8, requires_grad=True)
    candidate_actions = torch.randn(4, 6, 3)
    specialized = predictor.forward_candidate_h1(feature, candidate_actions)
    flattened_feature = feature[:, None, :].expand(-1, 6, -1).reshape(-1, 8)
    flattened_actions = candidate_actions.reshape(-1, 3)
    ordinary = predictor(flattened_feature, flattened_actions)[:, 0].reshape(4, 6, 8)
    torch.testing.assert_close(specialized, ordinary, rtol=2e-5, atol=2e-6)
    grad_specialized = torch.autograd.grad(specialized.square().mean(), feature)[0]
    grad_ordinary = torch.autograd.grad(ordinary.square().mean(), feature)[0]
    torch.testing.assert_close(grad_specialized, grad_ordinary, rtol=2e-5, atol=2e-6)


def test_specialized_h1_is_one_compile_safe_fullgraph_program():
    predictor = TAPE.AllLayerHorizonTapePredictor(4, 8, 3, 4, 5, 11)
    features = torch.randn(6, 4, 8)
    actions = torch.randn(6, 7, 3)
    compiled = torch.compile(
        lambda feature, candidate_actions: predictor.forward_candidate_h1(
            feature, candidate_actions
        ),
        backend="inductor",
        dynamic=False,
        fullgraph=True,
    )
    expected = predictor.forward_candidate_h1(features, actions)
    actual = compiled(features, actions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_combined_auxiliary_forward_is_compile_safe_and_replayable():
    agent = TAPE.Agent(_DummyEnvs(), _args())
    with torch.no_grad():
        for predictor in agent.tape_predictor.predictors:
            predictor.output.weight.normal_(0.0, 0.01)
    features = torch.randn(12, agent.tape_num_layers, agent.tape_feature_dim)
    actions = torch.randn(12, agent.tape_action_dim)
    candidate_actions = torch.randn(12, 5, agent.tape_action_dim)
    compiled = torch.compile(
        lambda feature, action, candidates: TAPE.tape_auxiliary_forward(
            agent, feature, action, candidates
        ),
        backend="inductor",
        dynamic=False,
        fullgraph=True,
    )
    expected = TAPE.tape_auxiliary_forward(
        agent, features, actions, candidate_actions
    )
    actual = compiled(features, actions, candidate_actions)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    replay = compiled(features + 0.1, actions + 0.2, candidate_actions + 0.3)
    assert not torch.equal(actual[0], replay[0])


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_candidate_zero_action_is_exact_policy_action(actor_dist):
    agent = TAPE.Agent(_DummyEnvs(), _args(actor_dist=actor_dist))
    feature = torch.randn(5, 8)
    dist, to_action, _ = agent._actor_dist(feature)
    z = dist.sample((4,)).permute(1, 0, 2).contiguous()
    if actor_dist == "beta":
        z = z.clamp(TAPE.SAMPLE_EPS, 1.0 - TAPE.SAMPLE_EPS)
    executed = to_action(z[:, 0])
    candidates = to_action(z)
    torch.testing.assert_close(candidates[:, 0], executed, rtol=0, atol=0)


def test_single_combined_auxiliary_backward_and_breaker_detaches_trunk():
    agent = TAPE.Agent(_DummyEnvs(), _args())
    with torch.no_grad():
        for predictor in agent.tape_predictor.predictors:
            predictor.output.weight.normal_(0.0, 0.01)
    feature = agent.get_actor_feat(torch.randn(7, 7))
    actions = torch.randn(7, 3)
    candidate_actions = torch.randn(7, 5, 3)
    h16, h1 = TAPE.tape_auxiliary_forward(agent, feature, actions, candidate_actions)
    target = torch.randn_like(h16)
    mask = torch.ones(h16.shape[:-1], dtype=torch.bool)
    target_contrast = torch.randn_like(h1[:, 0])
    q = TAPE.intervention_query_indices(7, 5, 1)
    tape_loss, tape_scale, _ = TAPE.horizon_td_tape_loss(
        h16, target, mask, scale=torch.ones(agent.tape_num_layers)
    )
    intervention_loss, _, _ = TAPE.intervention_contrast_loss(
        h1, target_contrast, q, scale=torch.ones(agent.tape_num_layers)
    )
    (tape_loss + intervention_loss).backward()
    assert all(p.grad is not None for p in agent.tape_predictor_parameters())
    assert any(p.grad is not None for p in agent.tape_trunk_parameters())
    if hasattr(agent, "actor_head"):
        assert all(p.grad is None for p in agent.actor_head.parameters())
    assert all(p.grad is None for p in agent.actor_alpha_head.parameters())
    assert all(p.grad is None for p in agent.actor_beta_head.parameters())
    assert all(
        p.grad is None
        for module in agent.target_bundle_modules()
        for p in module.parameters()
    )

    agent.zero_grad(set_to_none=True)
    detached = TAPE.tape_source_feature(
        agent.get_actor_feat(torch.randn(7, 7)), trunk_active=False
    )
    _, detached_h1 = TAPE.tape_auxiliary_forward(
        agent, detached, actions, candidate_actions
    )
    detached_h1.square().mean().backward()
    assert all(p.grad is None for p in agent.tape_trunk_parameters())
    assert any(p.grad is not None for p in agent.tape_predictor_parameters())


def test_intervention_metrics_have_per_layer_scalar_vectors_and_snapshot_lifetime():
    torch.manual_seed(7)
    predictions = torch.randn(9, 4, 4, 8)
    target = torch.randn(9, 4, 8)
    q = TAPE.intervention_query_indices(9, 4, 2)
    metrics = TAPE.intervention_contrast_metrics(predictions, target, q)
    assert set(metrics) == {"mse", "nmse", "cosine", "shuffled_nmse", "shuffled_ratio"}
    assert all(value.shape == (4,) and torch.isfinite(value).all() for value in metrics.values())
    retained = TAPE.retain_graph_output(predictions, compiled=True)
    predictions.zero_()
    assert torch.count_nonzero(retained) > 0


def test_compiled_probe_feature_reduction_owns_small_clone_across_replay():
    agent = TAPE.Agent(_DummyEnvs(), _args())
    compiled = torch.compile(
        lambda observations: TAPE.probe_value_and_actor_features_forward(
            agent, observations
        ),
        backend="inductor",
        dynamic=False,
        fullgraph=True,
    )
    first_value, first_features = compiled(torch.randn(6, 7))
    retained_value = TAPE.retain_graph_output(first_value, compiled=True)
    retained_contrast, _ = TAPE.build_intervention_target_contrast(
        first_features, 3, iteration=1
    )
    contrast_before_replay = retained_contrast.clone()
    second_value, second_features = compiled(torch.randn(6, 7) + 2.0)
    torch.testing.assert_close(retained_value, torch.zeros_like(retained_value))
    torch.testing.assert_close(retained_contrast, contrast_before_replay)
    assert retained_contrast.shape == (2, agent.tape_num_layers, agent.tape_feature_dim)
    assert torch.isfinite(retained_contrast).all()
    assert torch.isfinite(second_features).all()


def test_intervention_source_has_one_combined_compiled_path_and_v25_off_switch():
    source = SCRIPT.read_text()
    assert "tape_auxiliary_update_fn = torch.compile" in source
    assert "tape_intervention_update_fn = torch.compile" not in source
    assert "tape_auxiliary_update_fn(" in source
    assert "tape_intervention: bool = True" in source
    assert "assert args.tape" in source
    helper_source = source[
        source.index("def intervention_query_indices") : source.index("class Agent")
    ]
    assert ".item()" not in helper_source
    assert "if torch.any" not in helper_source
    common = dict(hidden=8, k_blocks=2, n_experts=2, num_bins=7, actor_dist="beta")
    torch.manual_seed(913)
    reference = V24.Agent(_DummyEnvs(), V24.Args(tape=False, **common))
    torch.manual_seed(913)
    candidate = TAPE.Agent(
        _DummyEnvs(), TAPE.Args(tape=False, tape_intervention=False, **common)
    )
    assert tuple(candidate.state_dict()) == tuple(reference.state_dict())
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], expected, rtol=0, atol=0)


def test_tape_enabled_intervention_off_preserves_v25_agent_and_partitions():
    common = dict(
        hidden=8,
        k_blocks=2,
        n_experts=2,
        num_bins=7,
        actor_dist="beta",
        tape=True,
        tape_intervention=False,
        tape_horizon=4,
        tape_horizon_embed_dim=3,
        tape_predictor_hidden=11,
    )
    torch.manual_seed(177)
    reference = TAPE.Agent(_DummyEnvs(), TAPE.Args(**common))
    torch.manual_seed(177)
    copied = V25.Agent(
        _DummyEnvs(),
        V25.Args(
            hidden=8,
            k_blocks=2,
            n_experts=2,
            num_bins=7,
            actor_dist="beta",
            tape=True,
            tape_horizon=4,
            tape_horizon_embed_dim=3,
            tape_predictor_hidden=11,
        ),
    )
    assert tuple(reference.state_dict()) == tuple(copied.state_dict())
    for name, expected in copied.state_dict().items():
        torch.testing.assert_close(reference.state_dict()[name], expected, rtol=0, atol=0)
    observations = torch.randn(6, 7)
    actions = torch.randn(6, 3)
    torch.testing.assert_close(
        reference.predict_tapes(reference.get_actor_feat(observations), actions),
        copied.predict_tapes(copied.get_actor_feat(observations), actions),
        rtol=0,
        atol=0,
    )
    assert {id(p) for p in reference.task_parameters()} | {
        id(p) for p in reference.tape_predictor_parameters()
    } == {id(p) for p in reference.parameters() if p.requires_grad}
