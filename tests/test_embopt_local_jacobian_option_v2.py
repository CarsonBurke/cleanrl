import ast
import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "embedding-optimization"
    / "ppo_continuous_action_embopt_local_jacobian_option_v2.py"
)
SPEC = importlib.util.spec_from_file_location(
    "embopt_local_jacobian_option_v2", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def small_args():
    return MODULE.Args(
        history_length=3,
        latent_dim=4,
        hidden_dim=16,
        transformer_layers=2,
        transformer_heads=2,
        option_heads=3,
        minibatch_size=8,
    )


def world():
    args = small_args()
    return MODULE.LocalJacobianWorldModel(
        obs_dim=5,
        action_dim=2,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        history_length=args.history_length,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
    )


def test_factorization_is_exact_and_base_cannot_receive_probe_delta():
    model = world().eval()
    observations = torch.randn(7, 3, 5)
    nominal = torch.randn(7, 3, 2)
    delta = torch.randn(7, 3, 2)
    base, jacobian = model.predict_components(observations, nominal)
    factual = model.factual_prediction(observations, nominal, delta)
    expected = base + torch.einsum("btza,bta->btz", jacobian, delta)
    torch.testing.assert_close(factual, expected)

    changed = model.factual_prediction(observations, nominal, delta + 10)
    base_again, _ = model.predict_components(observations, nominal)
    torch.testing.assert_close(base, base_again)
    assert not torch.allclose(factual, changed)
    assert "action_deltas" not in inspect.signature(
        model.predict_components
    ).parameters


def test_jacobian_head_requires_nonzero_probe_for_prediction_gradient():
    model = world()
    sigreg = MODULE.SIGReg(projections=8, knots=5, reference_samples=8)
    observations = torch.randn(8, 4, 5)
    nominal = torch.randn(8, 3, 2)
    valid = torch.ones(8, 4, dtype=torch.bool)
    zero_delta = torch.zeros(8, 3, 2)
    loss, _, _ = MODULE.masked_world_objective(
        model, sigreg, observations, nominal, zero_delta, valid, 0.0
    )
    loss.backward()
    zero_probe_gradient = sum(
        0.0 if parameter.grad is None else float(parameter.grad.abs().sum())
        for parameter in model.jacobian_head.parameters()
    )
    assert zero_probe_gradient == 0.0

    model.zero_grad(set_to_none=True)
    nonzero_delta = torch.randn(8, 3, 2)
    loss, _, _ = MODULE.masked_world_objective(
        model, sigreg, observations, nominal, nonzero_delta, valid, 0.0
    )
    loss.backward()
    probe_gradient = sum(
        0.0 if parameter.grad is None else float(parameter.grad.abs().sum())
        for parameter in model.jacobian_head.parameters()
    )
    assert probe_gradient > 0


def test_world_target_is_attached_and_invalid_padding_is_excluded():
    model = world()
    sigreg = MODULE.SIGReg(projections=8, knots=5, reference_samples=8)
    observations = torch.randn(4, 4, 5, requires_grad=True)
    nominal = torch.randn(4, 3, 2)
    delta = torch.randn(4, 3, 2)
    valid = torch.tensor([[False, False, True, True]] * 4)
    loss, _, _ = MODULE.masked_world_objective(
        model, sigreg, observations, nominal, delta, valid, 0.0
    )
    loss.backward()
    assert observations.grad[:, -1].abs().sum() > 0
    assert observations.grad[:, :2].abs().sum() == 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.observation_encoder.parameters()
    )


def test_padding_mask_prevents_invalid_tokens_from_changing_valid_prediction():
    model = world().eval()
    valid = torch.tensor([[False, False, True]])
    observations = torch.randn(1, 3, 5)
    nominal = torch.randn(1, 3, 2)
    base, jacobian = model.predict_components(observations, nominal, valid)
    changed_observations = observations.clone()
    changed_nominal = nominal.clone()
    changed_observations[:, :2] += 1000
    changed_nominal[:, :2] -= 1000
    changed_base, changed_jacobian = model.predict_components(
        changed_observations, changed_nominal, valid
    )
    torch.testing.assert_close(base[:, -1], changed_base[:, -1])
    torch.testing.assert_close(jacobian[:, -1], changed_jacobian[:, -1])


class DummyWorld(nn.Module):
    def __init__(self, jacobian):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("jacobian", jacobian)

    def encode(self, observations):
        return observations[..., :2] * self.scale

    def final_jacobian(
        self, observation_context, nominal_actions, source_valid=None
    ):
        return self.jacobian.expand(observation_context.shape[0], -1, -1)


class DummyPolicy(nn.Module):
    def __init__(self, action):
        super().__init__()
        self.raw = nn.Parameter(torch.atanh(action))

    def forward(self, z, goal):
        return torch.tanh(self.raw).expand(z.shape[0], -1)


def test_actor_is_factually_anchored_and_uses_only_local_jacobian():
    jacobian = torch.eye(2)
    stored_nominal = torch.tensor([[0.1, 0.2]])
    stored_delta = torch.tensor([[2.0, 3.0]])
    actual_next_obs = torch.tensor([[10.0, 20.0, 99.0]])
    nominal_successor = torch.tensor([[8.0, 17.0]])
    agent = SimpleNamespace(
        world=DummyWorld(jacobian),
        policy=DummyPolicy(stored_nominal[0]),
    )
    loss, fresh, metrics = MODULE.factual_local_actor_loss(
        agent,
        torch.zeros(1, 3, 3),
        torch.ones(1, 3, dtype=torch.bool),
        torch.zeros(1, 2, 2),
        actual_next_obs,
        nominal_successor,
        stored_nominal,
        stored_delta,
        control_cost_weight=0.0,
        reward_scale=0.1,
    )
    torch.testing.assert_close(fresh, stored_nominal)
    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(metrics["factual_goal_mse"], torch.tensor(6.5))


def test_actor_gradient_ownership_is_policy_only():
    agent = SimpleNamespace(
        world=DummyWorld(torch.eye(2)),
        policy=DummyPolicy(torch.tensor([0.0, 0.0])),
    )
    loss, _, _ = MODULE.factual_local_actor_loss(
        agent,
        torch.zeros(2, 3, 3),
        torch.ones(2, 3, dtype=torch.bool),
        torch.zeros(2, 2, 2),
        torch.ones(2, 3),
        torch.full((2, 2), 2.0),
        torch.zeros(2, 2),
        torch.full((2, 2), 0.1),
        control_cost_weight=0.1,
        reward_scale=0.1,
    )
    loss.backward()
    assert agent.policy.raw.grad is not None
    assert agent.policy.raw.grad.abs().sum() > 0
    assert agent.world.scale.grad is None


def test_goal_proposer_is_free_latent_and_only_goal_receives_gradient():
    args = small_args()
    agent = MODULE.Agent(5, 2, args)
    with torch.no_grad():
        for head in agent.c.heads:
            torch.nn.init.normal_(head[-1].weight, std=0.1)
    observations = torch.randn(8, 5)
    probes = torch.randn(8, 4)
    mask = torch.ones(8, dtype=torch.bool)
    loss, metrics = MODULE.goal_proposal_loss(
        agent, observations, probes, mask, args.pessimism_coef
    )
    loss.backward()
    assert metrics["latent_displacement_norm"] > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in agent.goal.parameters()
    )
    assert all(parameter.grad is None for parameter in agent.h.parameters())
    assert all(parameter.grad is None for parameter in agent.c.parameters())
    assert all(parameter.grad is None for parameter in agent.world.parameters())


def test_persistent_latent_goal_survives_and_resets_only_at_boundary():
    goals = MODULE.PersistentLatentGoals(3, 4)
    proposed = np.arange(12, dtype=np.float32).reshape(3, 4)
    used = goals.apply(proposed, np.asarray([True, True, False]))
    previous = used.copy()
    used = goals.apply(proposed + 100, np.asarray([False, False, True]))
    np.testing.assert_array_equal(used[:2], previous[:2])
    goals.reset(np.asarray([False, True, True]))
    assert goals.has_goal.tolist() == [True, False, False]


def test_goal_decision_stores_exact_latent_proposal_and_probe():
    args = small_args()
    agent = MODULE.Agent(5, 2, args)
    persistent = MODULE.PersistentLatentGoals(3, 4)
    observations = torch.randn(3, 5)
    probes = torch.randn(3, 4)
    with torch.no_grad():
        z = agent.world.encode(observations)
        expected = z + agent.goal(z) + probes
    used, proposal = MODULE.option_decision(
        agent, observations, persistent, probes, args
    )
    assert proposal.all()
    np.testing.assert_allclose(used, expected.numpy(), rtol=1e-5, atol=1e-5)


def test_transition_sequence_tracks_nominal_and_delta_with_correct_arity():
    context, past_nominal, past_delta, valid = MODULE.initialize_context(
        np.ones((2, 5), dtype=np.float32), 3, 2
    )
    result = MODULE.transition_sequence(
        context,
        past_nominal,
        past_delta,
        valid,
        np.ones((2, 2), dtype=np.float32),
        np.full((2, 2), 0.25, dtype=np.float32),
        np.full((2, 5), 2.0, dtype=np.float32),
    )
    assert len(result) == 4
    observations, nominal, delta, sequence_valid = result
    assert observations.shape == (2, 4, 5)
    assert nominal.shape == delta.shape == (2, 3, 2)
    np.testing.assert_array_equal(delta[:, -1], 0.25)
    assert sequence_valid[:, -1].all()


def test_replay_contains_only_factual_sequences_nominal_delta_and_valid():
    replay = MODULE.SequenceReplay(16, 3, 5, 2)
    assert set(vars(replay)) == {
        "capacity",
        "observations",
        "nominal_actions",
        "action_deltas",
        "valid",
        "pointer",
        "size",
    }
    assert not hasattr(replay, "goals")
    assert not hasattr(replay, "rewards")


def test_deterministic_probes_are_small_balanced_and_full_rank_each_step():
    probe = MODULE.deterministic_probe(17, 16, 6, 0.12, 257)
    torch.testing.assert_close(
        probe.mean(dim=0), torch.zeros(6), atol=1e-6, rtol=0
    )
    assert probe.abs().max() <= 0.12 + 1e-6
    covariance = probe.T @ probe / len(probe)
    assert torch.linalg.matrix_rank(covariance) == 6
    torch.testing.assert_close(
        covariance,
        torch.eye(6) * 0.12**2,
        atol=1e-6,
        rtol=1e-6,
    )


def test_cumulative_rate_is_exact_telemetry_not_ema():
    rate = MODULE.CumulativeRewardRate()
    assert rate.update([1.0, 3.0]) == 2.0
    assert rate.update([9.0]) == 13.0 / 3.0
    assert rate.reward_sum == 13.0
    assert rate.step_count == 3


def test_learned_reward_rate_follows_detached_bellman_residual():
    rate = MODULE.LearnedRewardRate()
    loss = MODULE.reward_rate_loss(
        torch.ones(8),
        torch.zeros(3, 8),
        torch.zeros(3, 8),
        torch.ones(8),
        rate,
    )
    loss.backward()
    assert rate.value.grad < 0


def test_average_boundary_is_regenerative_and_discounted_is_episodic():
    shape = (2, 3)
    next_h = torch.ones(shape)
    next_c = torch.full(shape, 2.0)
    live_h = torch.full(shape, 7.0)
    factual_h = torch.full(shape, 11.0)
    terminated = torch.tensor([True, False, False])
    truncated = torch.tensor([False, True, False])
    selected, bootstrap, _ = MODULE.boundary_next_option_values(
        next_h,
        next_c,
        live_h,
        factual_h,
        terminated,
        truncated,
        True,
        0.0,
    )
    torch.testing.assert_close(selected[:, :2], live_h[:, :2])
    assert bootstrap.tolist() == [1.0, 1.0, 1.0]
    selected, bootstrap, _ = MODULE.boundary_next_option_values(
        next_h,
        next_c,
        live_h,
        factual_h,
        terminated,
        truncated,
        False,
        0.0,
    )
    torch.testing.assert_close(selected[:, 1], factual_h[:, 1])
    assert bootstrap.tolist() == [0.0, 1.0, 1.0]


def test_h_optimizer_does_not_step_without_proposals():
    args = small_args()
    h = MODULE.StateBiasEnsemble(4, 16, 3)

    class CountingOptimizer:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

    optimizer = CountingOptimizer()
    loss, gradient = MODULE.update_proposal_h(
        h,
        torch.randn(5, 4),
        torch.randn(3, 5),
        torch.zeros(5, dtype=torch.bool),
        optimizer,
        args.bootstrap_probability,
        args.max_grad_norm,
    )
    assert loss == gradient == 0
    assert optimizer.steps == 0


def test_optimizer_ownership_is_exact_and_disjoint():
    agent = MODULE.Agent(5, 2, small_args())
    groups = MODULE.optimizer_parameter_groups(agent)
    names = list(groups)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            assert groups[name].isdisjoint(groups[other])
    assert set().union(*groups.values()) == {
        id(parameter) for parameter in agent.parameters()
    }
    assert groups["world"] == MODULE.world_parameter_ids(agent)


def test_procrustes_alignment_restores_global_chart():
    raw_new = torch.randn(64, 4)
    q, _ = torch.linalg.qr(torch.randn(4, 4))
    global_old = raw_new @ q
    alignment = MODULE.procrustes_alignment(raw_new, global_old)
    torch.testing.assert_close(
        raw_new @ alignment, global_old, atol=1e-5, rtol=1e-5
    )


def test_update_world_uses_only_valid_observations_as_chart_anchors():
    source = inspect.getsource(MODULE.update_world)
    assert "anchor_observations = observations[valid]" in source
    assert "observations.reshape" not in source


def test_training_update_order_and_world_objective_are_explicit():
    source = SCRIPT.read_text()
    actor = source.index("policy_optimizer.step()", source.index("def main"))
    goal = source.index("goal_optimizer.step()", actor)
    critic = source.index("c_optimizer.step()", goal)
    world = source.index("update_world(", critic)
    assert actor < goal < critic < world
    update_source = inspect.getsource(MODULE.update_world)
    assert "masked_world_objective" in update_source
    assert "loss.backward()" in update_source
    assert "optimizer.step()" in update_source


def test_source_has_no_policy_gradient_q_learning_or_planning_mechanism():
    tree = ast.parse(SCRIPT.read_text())
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    } | {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    forbidden = {
        "categorical",
        "normal",
        "log_prob",
        "entropy",
        "advantage",
        "ppo",
        "q_network",
        "target_network",
        "cem",
        "planner",
    }
    assert forbidden.isdisjoint({name.lower() for name in names})
