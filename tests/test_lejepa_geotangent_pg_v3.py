import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "ppo_continuous_action_lejepa_geotangent_pg_v3.py"
)
SPEC = importlib.util.spec_from_file_location("lejepa_geotangent_pg_v3", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_fixed_horizon_targets_bootstrap_truncation_but_not_termination():
    rewards = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    boundaries = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    terminations = torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    next_values = torch.tensor([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]])
    targets = MODULE.fixed_horizon_td_returns(
        rewards,
        boundaries,
        terminations,
        next_values,
        gamma=0.5,
        horizons=(1, 2),
    )
    expected_h1 = torch.tensor([[6.0, 60.0], [12.0, 20.0], [18.0, 180.0]])
    expected_h2 = torch.tensor([[7.0, 20.0], [12.0, 20.0], [18.0, 180.0]])
    torch.testing.assert_close(targets[0], expected_h1)
    torch.testing.assert_close(targets[1], expected_h2)


def test_horizon_portfolio_is_normalized_and_keeps_primary_share():
    torch.manual_seed(3)
    horizons, timesteps, envs, action_dim = 7, 64, 8, 2
    advantages = torch.randn(horizons, timesteps, envs)
    alpha = torch.full((timesteps, envs, action_dim), 2.0)
    beta = torch.full_like(alpha, 2.0)
    z = torch.distributions.Beta(alpha, beta).sample()
    portfolio, weights, agreement = MODULE.horizon_portfolio(
        advantages, z, alpha, beta
    )
    assert portfolio.shape == (timesteps, envs)
    assert agreement.shape == (horizons,)
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
    assert weights.max() >= 0.5
    assert torch.isfinite(portfolio).all()


def test_conflict_projection_preserves_reward_descent_component():
    reward = [torch.tensor([1.0, 0.0])]
    embedding = [torch.tensor([-1.0, 2.0])]
    combined, cosine, scale = MODULE.compose_projected_gradients(
        reward, embedding, fraction=0.2
    )
    assert cosine < 0
    assert scale > 0
    assert torch.dot(combined[0], reward[0]) > 0
    orthogonal_addition = combined[0] - reward[0]
    torch.testing.assert_close(
        torch.dot(orthogonal_addition, reward[0]),
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        orthogonal_addition.norm(),
        torch.tensor(0.2),
        atol=1e-6,
        rtol=0,
    )


def test_all_direct_horizon_predictors_receive_gradients():
    torch.manual_seed(5)
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
    targets = torch.randn(batch, len(MODULE.TD_HORIZONS), 5)
    masks = torch.ones(batch, len(MODULE.TD_HORIZONS))
    loss, prediction_loss, sigreg_loss, per_horizon = model(
        obs,
        actions,
        targets,
        masks,
        sigreg_weight=0.01,
    )
    assert per_horizon.shape == (len(MODULE.TD_HORIZONS),)
    assert torch.isfinite(
        torch.stack([loss, prediction_loss, sigreg_loss])
    ).all()
    loss.backward()
    for predictor in model.predictors:
        assert all(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            for parameter in predictor.parameters()
        )


def test_actor_retains_factor_log_prob_axis():
    args = MODULE.Args(hidden=16, k_blocks=1, n_experts=2)
    actor = MODULE.Actor(obs_dim=5, act_dim=3, args=args)
    _, _, joint, factors, _, _, _ = actor(torch.randn(7, 5))
    assert factors.shape == (7, 3)
    torch.testing.assert_close(joint, factors.sum(-1))
