import ast
import copy
import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_lejepa_geometric_dr_actor_v5.py"
)
SPEC = importlib.util.spec_from_file_location("lejepa_geometric_dr_actor_v5", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_default_gamma_has_long_geometric_horizon():
    args = MODULE.Args()
    assert args.gamma == 0.9970087504549047
    assert abs(args.gamma**1000 - 0.05) < 1e-12


def test_base_phi_contains_normalized_state_action_terms_and_constant():
    embedding = torch.tensor([[2.0, -3.0]])
    next_obs = torch.tensor([[3.0, 4.0, 0.0]])
    action = torch.tensor([[0.25, -0.5]])
    phi = MODULE.base_outcome_features(embedding, next_obs, action)

    assert phi.shape == (1, 2 + 3 + 2 + 2 + 1)
    torch.testing.assert_close(phi[0, :2], embedding[0])
    # Observations are already coordinate-normalized by the environment wrapper;
    # preserving them here retains linear forward-velocity reward structure.
    torch.testing.assert_close(phi[0, 2:5], next_obs[0])
    torch.testing.assert_close(phi[0, 5:7], action[0])
    torch.testing.assert_close(phi[0, 7:9], action[0].square())
    assert phi[0, -1].item() == 1.0


def test_ridge_probe_defines_an_exact_reward_residual():
    generator = torch.Generator().manual_seed(5)
    phi = torch.randn(4, 3, 7, generator=generator)
    reward = torch.randn(4, 3, generator=generator)
    weight, residual = MODULE.fit_reward_covector(phi, reward, ridge=0.2)

    reconstruction = torch.einsum("...d,d->...", phi, weight) + residual
    torch.testing.assert_close(reconstruction, reward, rtol=0.0, atol=2e-7)
    augmented = MODULE.augment_reward_residual(phi, reward, weight)
    task = torch.cat([weight, torch.ones(1)])
    torch.testing.assert_close(
        torch.einsum("...d,d->...", augmented, task),
        reward,
        rtol=0.0,
        atol=2e-7,
    )


def test_full_suffix_uses_all_observations_and_frozen_rollout_tail():
    gamma = 0.8
    outcomes = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    frozen_tail = torch.tensor([[[10.0]], [[20.0]], [[30.0]]])
    zeros = torch.zeros(3, 1)
    ones = torch.ones(3, 1)

    target, valid = MODULE.full_suffix_rb_targets(
        outcomes, frozen_tail, zeros, zeros, ones, gamma
    )
    expected_2 = 0.2 * 3.0 + 0.8 * 30.0
    expected_1 = 0.2 * 2.0 + 0.8 * expected_2
    expected_0 = 0.2 * 1.0 + 0.8 * expected_1
    torch.testing.assert_close(
        target[:, 0, 0], torch.tensor([expected_0, expected_1, expected_2])
    )
    assert valid.all()


def test_termination_zeros_tail_while_truncation_bootstraps_tail():
    gamma = 0.9
    outcomes = torch.tensor([[[2.0, 4.0]], [[3.0, 5.0]]])
    tails = torch.tensor([[[7.0, 11.0]], [[13.0, 17.0]]])
    boundaries = torch.ones(2, 1)
    valids = torch.ones(2, 1)

    truncation_target, _ = MODULE.full_suffix_rb_targets(
        outcomes, tails, torch.zeros(2, 1), boundaries, valids, gamma
    )
    terminal_target, _ = MODULE.full_suffix_rb_targets(
        outcomes, tails, torch.ones(2, 1), boundaries, valids, gamma
    )
    torch.testing.assert_close(
        truncation_target, (1.0 - gamma) * outcomes + gamma * tails
    )
    torch.testing.assert_close(terminal_target, (1.0 - gamma) * outcomes)


def test_invalid_suffix_does_not_leak_across_an_episode():
    outcomes = torch.ones(3, 1, 2)
    tails = torch.zeros_like(outcomes)
    terminations = torch.zeros(3, 1)
    boundaries = torch.zeros(3, 1)
    valids = torch.tensor([[1.0], [0.0], [1.0]])
    _, target_valid = MODULE.full_suffix_rb_targets(
        outcomes, tails, terminations, boundaries, valids, 0.9
    )
    assert target_valid[:, 0].tolist() == [False, False, True]


def test_dr_alpha_zero_is_state_baseline_likelihood_ratio_estimator():
    ratio = torch.tensor([0.7, 1.2])
    target = torch.tensor([[3.0, -1.0], [2.0, 5.0]])
    baseline = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
    data_mean = torch.randn_like(target)
    auxiliary_ratio = torch.randn(2, 3).exp()
    auxiliary_mean = torch.randn(2, 3, 2)

    actual = MODULE.doubly_robust_vector_surrogate(
        ratio,
        target,
        baseline,
        data_mean,
        auxiliary_ratio,
        auxiliary_mean,
        torch.tensor(0.0),
    )
    torch.testing.assert_close(actual, ratio[:, None] * (target - baseline))


def test_dr_addback_is_centered_by_independent_state_baseline():
    ratio = torch.ones(1)
    target = torch.tensor([[4.0, 7.0]])
    data_mean = torch.tensor([[2.0, 5.0]])
    auxiliary_mean = torch.tensor([[[1.0, 3.0], [3.0, 7.0]]])
    baseline = torch.tensor([[0.5, -1.0]])
    alpha = torch.tensor(1.0)

    actual = MODULE.doubly_robust_vector_surrogate(
        ratio,
        target,
        baseline,
        data_mean,
        torch.ones(1, 2),
        auxiliary_mean,
        alpha,
    )
    expected = (
        target
        - baseline
        - (data_mean - baseline)
        + (auxiliary_mean - baseline[:, None]).mean(1)
    )
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, target - data_mean)


def test_dr_nonuniform_coordinate_alpha_broadcasts_per_outcome():
    ratio = torch.tensor([1.2, 0.8])
    target = torch.tensor([[4.0, 2.0], [1.0, -3.0]])
    baseline = torch.tensor([[0.5, -1.0], [0.2, 0.7]])
    data_mean = torch.tensor([[2.0, 1.0], [-1.0, -2.0]])
    auxiliary_ratio = torch.tensor([[0.9, 1.1], [1.3, 0.7]])
    auxiliary_mean = torch.tensor(
        [
            [[1.0, 3.0], [3.0, 7.0]],
            [[-0.5, 1.5], [2.5, -1.0]],
        ]
    )
    alpha = torch.tensor([0.25, 1.75])

    actual = MODULE.doubly_robust_vector_surrogate(
        ratio,
        target,
        baseline,
        data_mean,
        auxiliary_ratio,
        auxiliary_mean,
        alpha,
    )
    expected = ratio[:, None] * (
        target - baseline - alpha * (data_mean - baseline)
    ) + alpha * (
        auxiliary_ratio[:, :, None]
        * (auxiliary_mean - baseline[:, None])
    ).mean(1)
    torch.testing.assert_close(actual, expected)


def test_dr_score_expectation_equals_target_score_when_model_is_exact():
    # A discrete quadrature of the estimator's score identity. Scores sum to zero
    # under the behavior law; data and independent auxiliary expectations match.
    score = torch.tensor([[-1.0, 0.5], [0.25, -1.0], [0.75, 0.5]])
    conditional_target = torch.tensor([1.0, 4.0, -2.0])
    model = conditional_target.clone()
    baseline = model.mean()
    alpha = 0.63
    data_term = (
        score
        * (conditional_target - baseline - alpha * (model - baseline)).unsqueeze(-1)
    ).mean(0)
    addback = alpha * (
        score * (model - baseline).unsqueeze(-1)
    ).mean(0)
    desired = (score * conditional_target.unsqueeze(-1)).mean(0)
    torch.testing.assert_close(data_term + addback, desired)


def test_variance_optimal_coordinate_alpha_recovers_joint_cancellation():
    generator = torch.Generator().manual_seed(2)
    direction = torch.randn(512, 3, 5, generator=generator)
    expected = torch.tensor([2.5, -0.7, 1.2])
    noise = 0.02 * torch.randn(512, 5, generator=generator)
    g0 = -torch.einsum("ndp,d->np", direction, expected) + noise
    alpha = MODULE.variance_optimal_coordinate_alpha(g0, direction)
    torch.testing.assert_close(alpha, expected, atol=0.01, rtol=0.0)


def test_unidentified_coordinate_alpha_falls_back_to_zero():
    g0 = torch.randn(8, 3)
    direction = torch.ones(8, 4, 3)
    torch.testing.assert_close(
        MODULE.variance_optimal_coordinate_alpha(g0, direction),
        torch.zeros(4),
    )


def test_policy_rollback_restores_parameters_and_optimizer_moments():
    torch.manual_seed(11)
    module = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(module.parameters(), lr=0.1)
    x = torch.randn(5, 3)
    optimizer.zero_grad()
    module(x).square().mean().backward()
    optimizer.step()
    module_snapshot = copy.deepcopy(module.state_dict())
    optimizer_snapshot = copy.deepcopy(optimizer.state_dict())

    optimizer.zero_grad()
    (-module(x).square().mean()).backward()
    optimizer.step()
    assert any(
        not torch.equal(value, module_snapshot[name])
        for name, value in module.state_dict().items()
    )
    MODULE.restore_training_state(
        module, optimizer, module_snapshot, optimizer_snapshot
    )
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, module_snapshot[name])
    restored_optimizer = optimizer.state_dict()
    for parameter_id, state in optimizer_snapshot["state"].items():
        for key, expected in state.items():
            actual = restored_optimizer["state"][parameter_id][key]
            if torch.is_tensor(expected):
                torch.testing.assert_close(actual, expected)
            else:
                assert actual == expected


def test_residual_gauge_resets_only_last_output_adam_moments():
    torch.manual_seed(17)
    linear = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=0.01, amsgrad=True)
    linear(torch.randn(5, 4)).square().mean().backward()
    optimizer.step()
    weight_first_rows = {
        key: value[:-1].clone()
        for key, value in optimizer.state[linear.weight].items()
        if torch.is_tensor(value) and value.ndim > 0
    }
    MODULE.reset_optimizer_output_row(optimizer, linear)
    for parameter in (linear.weight, linear.bias):
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            assert torch.count_nonzero(optimizer.state[parameter][key][-1]) == 0
    for key, expected in weight_first_rows.items():
        torch.testing.assert_close(
            optimizer.state[linear.weight][key][:-1], expected
        )


def test_alpha_holdout_rotates_whole_environment_streams():
    valid = torch.ones(5, 4, dtype=torch.bool)
    heldout_0, actor_0 = MODULE.heldout_env_split(valid, 4, 0.25, offset=0)
    heldout_1, actor_1 = MODULE.heldout_env_split(valid, 4, 0.25, offset=1)
    assert set(heldout_0.remainder(4).tolist()) == {0}
    assert set(heldout_1.remainder(4).tolist()) == {1}
    assert set(actor_0.remainder(4).tolist()) == {1, 2, 3}
    assert set(actor_1.remainder(4).tolist()) == {0, 2, 3}


def test_residual_gauge_transport_preserves_reward_and_changes_coordinates():
    torch.manual_seed(9)
    model = MODULE.StateGeometricOutcomeMean(3, 6, 8)
    obs = torch.randn(7, 3)
    old_probe = torch.tensor([0.2, -0.7, 0.4, 1.1, -0.3])
    new_probe = torch.tensor([-0.1, 0.5, 0.8, -0.2, 0.6])
    before = model(obs).detach()
    old_task = torch.cat([old_probe, torch.ones(1)])
    represented_reward = before @ old_task

    MODULE.transport_residual_gauge(model, old_probe, new_probe)
    after = model(obs).detach()
    new_task = torch.cat([new_probe, torch.ones(1)])
    torch.testing.assert_close(after[:, :-1], before[:, :-1])
    torch.testing.assert_close(after @ new_task, represented_reward, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(
        after[:, -1],
        before[:, -1] + before[:, :-1] @ (old_probe - new_probe),
        atol=2e-6,
        rtol=1e-5,
    )


def test_cold_factorization_is_identity_dynamics_with_zero_scalar_control():
    class IdentityEncoder(torch.nn.Module):
        def forward(self, obs):
            return obs

    obs = torch.tensor([[0.25, -0.75]])
    action = torch.tensor([[0.4]])
    gamma = 0.8
    # emb(2) + obs(2) + action(1) + action^2(1) + constant(1) + residual(1)
    state_model = MODULE.StateGeometricOutcomeMean(2, 8, 8)
    dynamics = MODULE.OneStepDynamics(2, 1, 8)
    rotation, offset = torch.eye(2), torch.zeros(2)

    control = MODULE.factorized_action_control(
        state_model,
        dynamics,
        IdentityEncoder(),
        obs,
        action,
        rotation,
        offset,
        torch.zeros(7),
        gamma,
    )
    expected_immediate = torch.tensor(
        [[0.25, -0.75, 0.25, -0.75, 0.4, 0.16, 1.0, 0.0]]
    )
    torch.testing.assert_close(control, (1.0 - gamma) * expected_immediate)
    # Before the first reward-probe fit, task=[0,...,0,1], and the explicitly
    # zero counterfactual residual makes fixed alpha=1 exactly neutral.
    initial_task = torch.tensor([0.0] * 7 + [1.0])
    torch.testing.assert_close(control @ initial_task, torch.zeros(1))


def test_factorization_uses_dynamics_prediction_in_immediate_and_tail():
    class IdentityEncoder(torch.nn.Module):
        def forward(self, obs):
            return obs

    class ConstantStateMean(torch.nn.Module):
        def forward(self, obs):
            return torch.cat([obs, obs, obs[:, :1], obs[:, :1], obs[:, :1], obs[:, :1]], -1)

    class ActionDynamics(torch.nn.Module):
        def forward(self, obs, action):
            delta = torch.cat([action, -action], -1)
            return delta, action[:, 0] + 2.0

    obs = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    action = torch.tensor([[0.5], [-0.5]])
    control = MODULE.factorized_action_control(
        ConstantStateMean(),
        ActionDynamics(),
        IdentityEncoder(),
        obs,
        action,
        torch.eye(2),
        torch.zeros(2),
        torch.zeros(7),
        0.7,
    )
    assert control.shape == (2, 8)
    assert not torch.allclose(control[0], control[1])
    predicted_delta, predicted_reward = ActionDynamics()(obs, action)
    predicted_next = obs + predicted_delta
    expected_tail = ConstantStateMean()(predicted_next)
    torch.testing.assert_close(
        control[:, -1],
        0.3 * predicted_reward + 0.7 * expected_tail[:, -1],
    )


def test_counterfactual_immediate_outcome_contracts_to_predicted_reward():
    class IdentityEncoder(torch.nn.Module):
        def forward(self, obs):
            return obs

    class ZeroStateMean(torch.nn.Module):
        def forward(self, obs):
            return torch.zeros(obs.shape[0], 8)

    class RewardDynamics(torch.nn.Module):
        def forward(self, obs, action):
            return torch.zeros_like(obs), 1.5 + action[:, 0]

    obs = torch.tensor([[0.2, -0.4], [0.8, 0.1]])
    action = torch.tensor([[0.3], [-0.2]])
    reward_covector = torch.tensor([0.4, -0.1, 0.7, 0.2, -0.5, 0.3, 0.9])
    gamma = 0.6
    control = MODULE.factorized_action_control(
        ZeroStateMean(),
        RewardDynamics(),
        IdentityEncoder(),
        obs,
        action,
        torch.eye(2),
        torch.zeros(2),
        reward_covector,
        gamma,
    )
    reward_task = torch.cat([reward_covector, torch.ones(1)])
    expected_reward = 1.5 + action[:, 0]
    torch.testing.assert_close(
        control @ reward_task,
        (1.0 - gamma) * expected_reward,
    )


def test_no_forbidden_algorithmic_mechanisms_and_lejepa_target_is_attached():
    source = SCRIPT.read_text()
    tree = ast.parse(source)
    args_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Args"
    )
    fields = {
        node.target.id
        for node in args_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "gae_lambda" not in fields
    assert "clip_coef" not in fields
    assert "norm_adv" not in fields
    assert not any(
        isinstance(node, ast.ClassDef)
        and ("critic" in node.name.lower() or node.name.lower() == "qnetwork")
        for node in ast.walk(tree)
    )
    assert "torch.maximum" not in source
    ssl_forward = source[
        source.index("class LeJepaSSL") : source.index(
            "class StateGeometricOutcomeMean"
        )
    ]
    assert "embedding[:, 1:].detach()" not in ssl_forward
    assert "class StateGeometricOutcomeMean" in source
    assert "class OneStepDynamics" in source
    assert "model_target(b_obs, b_action)" not in source


def test_models_are_hard_updated_only_after_actor_consumption_and_fitting():
    source = SCRIPT.read_text()
    control_position = source.index("auxiliary_mean = factorized_action_control")
    actor_position = source.index("for epoch in range(args.update_epochs)")
    state_fit_position = source.index("prediction = state_model(b_obs[mb])")
    dynamics_fit_position = source.index(
        "predicted_delta, predicted_reward = dynamics(",
        state_fit_position,
    )
    hard_state_position = source.index("hard_update(state_model_target, state_model)")
    hard_dynamics_position = source.index("hard_update(dynamics_target, dynamics)")
    assert control_position < actor_position < state_fit_position < dynamics_fit_position
    assert dynamics_fit_position < hard_state_position
    assert dynamics_fit_position < hard_dynamics_position
