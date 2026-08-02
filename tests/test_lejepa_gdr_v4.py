import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "ppo_continuous_action_lejepa_gdr_v4.py"
)
SPEC = importlib.util.spec_from_file_location("lejepa_gdr_v4", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_gamma_is_the_declared_long_geometric_horizon():
    args = MODULE.Args()
    assert args.gamma == 0.9970087504549047
    assert abs(1.0 / (1.0 - args.gamma) - 334.308619) < 1e-3


def test_longest_suffix_uses_rollout_and_boundary_bootstraps():
    features = torch.tensor(
        [[[1.0], [10.0]], [[2.0], [20.0]], [[3.0], [30.0]]]
    )
    successor = torch.tensor(
        [[[100.0], [1000.0]], [[200.0], [2000.0]], [[300.0], [3000.0]]]
    )
    continuation = torch.tensor(
        [[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]]
    )
    bootstrap = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    )
    target = MODULE.longest_suffix_targets(
        features, successor, continuation, bootstrap, gamma=0.5
    )
    expected = torch.tensor(
        [[[50.75], [10.0]], [[101.0], [10.0]], [[151.5], [1515.0]]]
    )
    torch.testing.assert_close(target, expected)


def test_vector_dr_correction_is_exact_for_any_alpha_in_expectation():
    # Two independent behavior samples enumerate the same finite action law. The
    # subtraction and addback cancel before any reward contraction.
    target = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    model = torch.tensor([[5.0, 7.0], [-1.0, 2.0]])
    rows_target = target.repeat_interleave(2, dim=0)
    observed_model = model.repeat_interleave(2, dim=0)
    auxiliary_model = model.repeat(2, 1)
    ratio_observed = torch.tensor([0.7, 0.7, 1.3, 1.3])
    ratio_auxiliary = torch.tensor([0.7, 1.3, 0.7, 1.3])
    alpha = torch.full((4,), 8.3)
    corrected = MODULE.vector_doubly_robust_surrogate(
        ratio_observed,
        ratio_auxiliary,
        rows_target,
        observed_model,
        auxiliary_model,
        alpha,
    )
    anchor = (ratio_observed.unsqueeze(-1) * rows_target).mean(0)
    torch.testing.assert_close(corrected, anchor)


def test_crossfit_alpha_does_not_use_its_own_environment_fold():
    torch.manual_seed(4)
    rows, feature_dim, score_dim = 80, 5, 4
    env_index = torch.arange(rows) % 8
    target = torch.randn(rows, feature_dim)
    observed = torch.randn(rows, feature_dim)
    auxiliary = torch.randn(rows, feature_dim)
    score = torch.randn(rows, score_dim)
    auxiliary_score = torch.randn(rows, score_dim)
    alpha_a, _ = MODULE.crossfit_alpha(
        target,
        observed,
        auxiliary,
        score,
        auxiliary_score,
        env_index,
        mode="heldout",
    )
    even = (env_index % 2) == 0
    modified = target.clone()
    modified[even] += 1000.0 * torch.randn_like(modified[even])
    alpha_b, _ = MODULE.crossfit_alpha(
        modified,
        observed,
        auxiliary,
        score,
        auxiliary_score,
        env_index,
        mode="heldout",
    )
    torch.testing.assert_close(alpha_a[even], alpha_b[even])


def test_alpha_ablation_modes_are_exact():
    tensors = [torch.randn(6, 3), torch.randn(6, 3), torch.randn(6, 3)]
    scores = [torch.randn(6, 4), torch.randn(6, 4)]
    env_index = torch.arange(6)
    zero, _ = MODULE.crossfit_alpha(
        *tensors, *scores, env_index, mode="zero"
    )
    one, _ = MODULE.crossfit_alpha(
        *tensors, *scores, env_index, mode="one"
    )
    torch.testing.assert_close(zero, torch.zeros_like(zero))
    torch.testing.assert_close(one, torch.ones_like(one))


def test_attached_target_encoder_and_predictor_receive_gradients():
    torch.manual_seed(8)
    args = MODULE.Args(
        emb_dim=6,
        model_hidden=16,
        sigreg_num_proj=8,
        sigreg_proj_chunk=4,
        sigreg_batch=12,
    )
    model = MODULE.GeometricSuccessor(obs_dim=5, act_dim=2, args=args)
    time_steps, envs = 4, 3
    obs = torch.randn(time_steps, envs, 5)
    next_obs = torch.randn_like(obs)
    action = torch.randn(time_steps, envs, 2).tanh()
    continuation = torch.ones(time_steps, envs)
    bootstrap_mask = torch.zeros_like(continuation)
    bootstrap_mask[-1] = 1.0
    bootstrap = torch.randn(time_steps, envs, model.feature_dim)
    loss, prediction, sigreg, targets, features = model.attached_loss(
        obs,
        next_obs,
        action,
        continuation,
        bootstrap_mask,
        bootstrap,
        args.gamma,
        args.sigreg_weight,
        args.sigreg_batch,
    )
    assert targets.requires_grad
    assert features.requires_grad
    assert torch.isfinite(torch.stack([loss, prediction, sigreg])).all()
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.encoder.parameters()
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.predictor.parameters()
    )
