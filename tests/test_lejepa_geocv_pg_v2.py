import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "ppo_continuous_action_lejepa_geocv_pg_v2.py"
)
SPEC = importlib.util.spec_from_file_location("lejepa_geocv_pg_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_longest_horizon_td_returns_bootstrap_only_unfinished_tail():
    rewards = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
    )
    boundaries = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    )
    returns = MODULE.longest_horizon_td_returns(
        rewards,
        boundaries,
        last_value=torch.tensor([100.0, 200.0]),
        gamma=0.5,
    )
    expected_returns = torch.tensor(
        [[2.0, 27.5], [2.0, 35.0], [30.0, 30.0], [54.0, 140.0]]
    )
    torch.testing.assert_close(returns, expected_returns)


def test_successor_td_target_stops_bootstrap_at_boundary():
    next_embedding = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    next_successor = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    target = MODULE.successor_td_target(
        next_embedding,
        next_successor,
        boundaries=torch.tensor([0.0, 1.0]),
        gamma=0.75,
    )
    expected = torch.tensor([[8.0, 16.0], [1.5, 2.0]])
    torch.testing.assert_close(target, expected)


def test_beta_linear_control_expectation_has_exact_analytic_gradient():
    torch.manual_seed(3)
    n, action_dim = 200_000, 2
    alpha = torch.tensor([[2.3, 1.7]]).expand(n, -1)
    beta = torch.tensor([[1.8, 2.6]]).expand(n, -1)
    distribution = torch.distributions.Beta(alpha, beta)
    z = distribution.sample()
    action = 2.0 * z - 1.0
    mean, d_alpha, d_beta = MODULE.beta_action_mean_and_derivatives(
        alpha, beta
    )
    coefficient = torch.tensor([0.7, -1.2])
    control = ((action - mean) * coefficient).sum(-1)
    sampled_gradient = (
        MODULE.beta_parameter_score(z, alpha, beta)
        * control.unsqueeze(-1)
    ).mean(0)
    analytic_gradient = torch.cat(
        [coefficient * d_alpha[0], coefficient * d_beta[0]]
    )
    torch.testing.assert_close(
        sampled_gradient, analytic_gradient, atol=1.2e-2, rtol=3e-2
    )


def test_control_addback_cancels_off_old_policy():
    points = 200_000
    z = (
        torch.arange(points, dtype=torch.float64) + 0.5
    ) / points
    old_alpha = torch.tensor(2.3, dtype=torch.float64)
    old_beta = torch.tensor(1.8, dtype=torch.float64)
    new_alpha = torch.tensor(
        2.05, dtype=torch.float64, requires_grad=True
    )
    new_beta = torch.tensor(
        2.15, dtype=torch.float64, requires_grad=True
    )
    old_distribution = torch.distributions.Beta(
        old_alpha, old_beta
    )
    new_distribution = torch.distributions.Beta(
        new_alpha, new_beta
    )
    weights = old_distribution.log_prob(z).exp()
    weights = weights / weights.sum()
    ratio = (
        new_distribution.log_prob(z)
        - old_distribution.log_prob(z)
    ).exp()
    reward_signal = torch.sin(3.0 * z) + 0.4 * z.square()
    old_mean = 2.0 * old_alpha / (old_alpha + old_beta) - 1.0
    new_mean = 2.0 * new_alpha / (new_alpha + new_beta) - 1.0
    action = 2.0 * z - 1.0
    coefficient = 0.73
    control = coefficient * (action - old_mean)
    anchor = (weights * ratio * reward_signal).sum()
    corrected = (
        weights * ratio * (reward_signal - control)
    ).sum() + coefficient * (new_mean - old_mean)
    anchor_gradient = torch.autograd.grad(
        anchor, (new_alpha, new_beta), retain_graph=True
    )
    corrected_gradient = torch.autograd.grad(
        corrected, (new_alpha, new_beta)
    )
    torch.testing.assert_close(
        torch.stack(anchor_gradient),
        torch.stack(corrected_gradient),
        atol=2e-5,
        rtol=2e-5,
    )


def test_crossfit_control_variate_reduces_synthetic_gradient_variance():
    torch.manual_seed(7)
    timesteps, envs, feature_dim, action_dim = 256, 8, 12, 2
    n = timesteps * envs
    alpha = torch.full((n, action_dim), 2.0)
    beta = torch.full((n, action_dim), 2.0)
    z = torch.distributions.Beta(alpha, beta).sample()
    action = 2.0 * z - 1.0
    mean, _, _ = MODULE.beta_action_mean_and_derivatives(alpha, beta)
    jacobian = torch.randn(n, feature_dim, action_dim) * 0.2
    features = torch.einsum("nfa,na->nf", jacobian, action - mean)
    true_coefficient = torch.randn(feature_dim)
    advantage = features @ true_coefficient + 0.1 * torch.randn(n)
    valid = torch.ones(n, dtype=torch.bool)
    env_index = torch.arange(n) % envs

    control, action_gradient, ratio, rank = (
        MODULE.fit_crossfit_control_variate(
            features,
            jacobian,
            advantage,
            z,
            alpha,
            beta,
            valid,
            env_index,
            rank=8,
            ridge=1e-3,
            min_variance_gain=0.0,
        )
    )
    assert rank > 0
    assert ratio < 1.0
    assert control.std() > 0
    assert action_gradient.std() > 0


def test_crossfit_gate_disables_action_insensitive_model():
    n, feature_dim, action_dim = 64, 6, 2
    alpha = torch.full((n, action_dim), 2.0)
    beta = torch.full((n, action_dim), 2.0)
    z = torch.distributions.Beta(alpha, beta).sample()
    control, action_gradient, ratio, rank = (
        MODULE.fit_crossfit_control_variate(
            torch.zeros(n, feature_dim),
            torch.zeros(n, feature_dim, action_dim),
            torch.randn(n),
            z,
            alpha,
            beta,
            torch.ones(n, dtype=torch.bool),
            torch.arange(n) % 8,
            rank=4,
            ridge=1e-2,
            min_variance_gain=0.02,
        )
    )
    assert rank == 0
    assert ratio == 1.0
    assert control.count_nonzero() == 0
    assert action_gradient.count_nonzero() == 0


def test_crossfit_never_fits_a_row_from_its_own_environment_fold():
    torch.manual_seed(9)
    timesteps, envs, feature_dim, action_dim = 64, 8, 8, 2
    n = timesteps * envs
    alpha = torch.full((n, action_dim), 2.0)
    beta = torch.full((n, action_dim), 2.0)
    z = torch.distributions.Beta(alpha, beta).sample()
    action = 2.0 * z - 1.0
    mean, _, _ = MODULE.beta_action_mean_and_derivatives(alpha, beta)
    jacobian = torch.randn(n, feature_dim, action_dim) * 0.2
    features = torch.einsum("nfa,na->nf", jacobian, action - mean)
    env_index = torch.arange(n) % envs
    valid = torch.ones(n, dtype=torch.bool)
    advantage = features @ torch.randn(feature_dim) + 0.2 * torch.randn(n)
    common = dict(
        raw_features=features,
        raw_jacobian=jacobian,
        z=z,
        alpha=alpha,
        beta=beta,
        valid=valid,
        env_index=env_index,
        rank=6,
        ridge=1e-2,
        min_variance_gain=-10.0,
    )
    control_a, gradient_a, _, _ = MODULE.fit_crossfit_control_variate(
        advantage=advantage, **common
    )
    modified = advantage.clone()
    even_fold = (env_index % 2) == 0
    modified[even_fold] += 1000.0 * torch.randn_like(modified[even_fold])
    control_b, gradient_b, _, _ = MODULE.fit_crossfit_control_variate(
        advantage=modified, **common
    )
    torch.testing.assert_close(control_a[even_fold], control_b[even_fold])
    torch.testing.assert_close(gradient_a[even_fold], gradient_b[even_fold])


def test_local_and_successor_predictors_both_receive_gradients():
    torch.manual_seed(11)
    args = MODULE.Args(
        emb_dim=8,
        ssl_hidden=16,
        sigreg_num_proj=8,
        sigreg_proj_chunk=4,
    )
    model = MODULE.LeJepaSSL(obs_dim=5, act_dim=2, args=args)
    batch = 12
    obs = torch.randn(batch, 5)
    actions = torch.randn(batch, 2).tanh()
    target_obs = torch.randn(batch, 5)
    successor_target = torch.randn(batch, 8)
    loss, next_loss, successor_loss, sigreg_loss = model(
        obs,
        actions,
        target_obs,
        successor_target,
        sigreg_weight=0.01,
        successor_weight=1.0,
    )
    assert torch.isfinite(
        torch.stack([loss, next_loss, successor_loss, sigreg_loss])
    ).all()
    loss.backward()
    for predictor in (model.next_predictor, model.successor_predictor):
        assert all(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            for parameter in predictor.parameters()
        )
