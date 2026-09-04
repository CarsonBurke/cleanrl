import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "streaming-posterior-filter"
    / "ppo_continuous_action_spf_online_posteriorinfo_v13.py"
)
spec = importlib.util.spec_from_file_location(
    "ppo_continuous_action_spf_online_posteriorinfo_v13",
    SCRIPT,
)
assert spec is not None
module = importlib.util.module_from_spec(spec)
loader = spec.loader
assert loader is not None
loader.exec_module(module)
Args = module.Args
RobustPosteriorDynamics = module.RobustPosteriorDynamics
validate_online_contract = module.validate_online_contract


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "posterior-information tests require CUDA"
    return torch.device("cuda")


def test_default_configuration_is_single_use_v_only():
    validate_online_contract(Args())


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("num_steps", 2),
        ("update_epochs", 2),
        ("num_minibatches", 2),
        ("norm_adv", True),
        ("ret_percnorm", True),
        ("adv_transform", "rankgauss"),
        ("critic_mtp_horizon", 2),
        ("target_kl", 0.03),
        ("ent_alpha", 0.1),
        ("ent_coef", 0.01),
        ("auto_entropy", True),
        ("posterior_info_coef", 0.0),
        ("posterior_prior_variance", 0.0),
        ("posterior_reset_rate", 0.0),
        ("posterior_reset_rate", 1.01),
        ("posterior_student_df", 0.0),
        ("posterior_scale_rate", 0.0),
        ("posterior_scale_rate", 1.01),
        ("posterior_baseline_rate", 0.0),
        ("posterior_baseline_rate", 1.01),
        ("posterior_bonus_clip", 0.0),
    ),
)
def test_contract_rejects_nonstreaming_or_confounded_settings(name, value):
    args = Args()
    setattr(args, name, value)

    with pytest.raises(ValueError, match=name):
        validate_online_contract(args)


def test_information_is_read_before_update_and_falls_on_visited_key(device):
    args = Args(
        num_envs=1,
        posterior_reset_rate=1e-8,
        posterior_scale_rate=1e-6,
    )
    posterior = RobustPosteriorDynamics(1, 2, 1, args, device)
    obs = torch.tensor(((1.0, 0.0),), device=device)
    action = torch.tensor(((0.75,),), device=device)
    next_obs = torch.tensor(((1.1, -0.1),), device=device)
    valid = torch.ones(1, device=device)

    _, raw_before = posterior.information_gain(obs, action)
    posterior.update(obs, action, next_obs, valid, raw_before)
    _, raw_after = posterior.information_gain(obs, action)

    assert raw_after.item() < raw_before.item()


def test_posterior_becomes_action_direction_selective(device):
    args = Args(
        num_envs=1,
        posterior_reset_rate=1e-8,
        posterior_scale_rate=1e-6,
    )
    posterior = RobustPosteriorDynamics(1, 2, 1, args, device)
    obs = torch.tensor(((1.0, 0.0),), device=device)
    visited_action = torch.tensor(((1.0,),), device=device)
    novel_action = torch.tensor(((-1.0,),), device=device)
    next_obs = torch.tensor(((1.0, 0.0),), device=device)
    valid = torch.ones(1, device=device)

    for _ in range(8):
        _, raw = posterior.information_gain(obs, visited_action)
        posterior.update(obs, visited_action, next_obs, valid, raw)

    _, visited_info = posterior.information_gain(obs, visited_action)
    _, novel_info = posterior.information_gain(obs, novel_action)

    assert novel_info.item() > visited_info.item()


def test_joseph_update_preserves_positive_covariance_and_stream_isolation(device):
    args = Args(num_envs=2, posterior_reset_rate=1e-3)
    posterior = RobustPosteriorDynamics(2, 3, 2, args, device)
    obs = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), device=device)
    action = torch.tensor(((0.5, -0.5), (-0.5, 0.5)), device=device)
    next_obs = obs + 0.1
    valid = torch.tensor((1.0, 0.0), device=device)
    before_weight = posterior.weight.clone()
    posterior.covariance[1].mul_(0.5)
    before_covariance = posterior.covariance.clone()
    _, raw = posterior.information_gain(obs, action)

    posterior.update(obs, action, next_obs, valid, raw)

    assert not torch.equal(posterior.weight[0], before_weight[0])
    torch.testing.assert_close(posterior.weight[1], before_weight[1], rtol=0, atol=0)
    eigenvalues = torch.linalg.eigvalsh(posterior.covariance)
    assert torch.all(eigenvalues > 0)
    expected_invalid = torch.lerp(
        before_covariance[1],
        posterior.prior_covariance[1],
        args.posterior_reset_rate,
    )
    torch.testing.assert_close(posterior.covariance[1], expected_invalid)


def test_student_weight_bounds_extreme_dynamics_write(device):
    args = Args(
        num_envs=1,
        posterior_reset_rate=1e-8,
        posterior_scale_rate=1e-6,
    )
    moderate = RobustPosteriorDynamics(1, 2, 1, args, device)
    extreme = RobustPosteriorDynamics(1, 2, 1, args, device)
    obs = torch.zeros((1, 2), device=device)
    action = torch.zeros((1, 1), device=device)
    valid = torch.ones(1, device=device)
    _, moderate_raw = moderate.information_gain(obs, action)
    _, extreme_raw = extreme.information_gain(obs, action)

    moderate.update(obs, action, torch.ones_like(obs), valid, moderate_raw)
    extreme.update(obs, action, torch.full_like(obs, 1_000.0), valid, extreme_raw)

    assert extreme.weight.norm().item() < moderate.weight.norm().item()
    assert torch.isfinite(extreme.weight).all()
