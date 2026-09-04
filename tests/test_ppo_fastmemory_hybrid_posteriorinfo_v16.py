import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "streaming-posterior-filter"
    / "ppo_continuous_action_spf_hybrid_posteriorinfo_v16.py"
)
spec = importlib.util.spec_from_file_location(
    "ppo_continuous_action_spf_hybrid_posteriorinfo_v16", SCRIPT
)
assert spec is not None
module = importlib.util.module_from_spec(spec)
loader = spec.loader
assert loader is not None
loader.exec_module(module)
Args = module.Args
RobustDiagonalPosteriorDynamics = module.RobustDiagonalPosteriorDynamics
propagate_information_advantage = module.propagate_information_advantage
validate_posterior_contract = module.validate_posterior_contract


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "hybrid-posterior tests require CUDA"
    return torch.device("cuda")


def test_default_hybrid_contract_is_valid():
    validate_posterior_contract(Args())


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("ent_alpha", 0.1),
        ("ent_coef", 0.01),
        ("auto_entropy", True),
        ("normalize_reward", True),
        ("clip_reward", True),
        ("posterior_info_coef", 0.0),
        ("posterior_prior_variance", 0.0),
        ("posterior_reset_rate", 0.0),
        ("posterior_reset_rate", 1.01),
        ("posterior_reset_rate", 1.0),
        ("posterior_student_df", 0.0),
        ("posterior_scale_rate", 0.0),
        ("posterior_scale_rate", 1.01),
        ("posterior_scale_rate", 1.0),
        ("posterior_baseline_rate", 0.0),
        ("posterior_baseline_rate", 1.01),
        ("posterior_baseline_rate", 1.0),
        ("posterior_bonus_clip", 0.0),
    ),
)
def test_contract_rejects_confounds(name, value):
    args = Args()
    setattr(args, name, value)
    with pytest.raises(ValueError, match=name):
        validate_posterior_contract(args)


def test_information_advantage_stops_at_episode_boundary(device):
    bonuses = torch.tensor(((1.0,), (2.0,), (3.0,)), device=device)
    boundaries = torch.tensor(((0.0,), (1.0,), (0.0,)), device=device)

    result = propagate_information_advantage(bonuses, boundaries, 0.9, 0.5)

    torch.testing.assert_close(
        result, torch.tensor(((1.9,), (2.0,), (3.0,)), device=device)
    )

def test_diagonal_posterior_prefers_unvisited_action_dimension(device):
    args = Args(
        num_envs=1,
        posterior_reset_rate=1e-8,
        posterior_scale_rate=1e-6,
    )
    posterior = RobustDiagonalPosteriorDynamics(1, 2, 2, args, device)
    obs = torch.tensor(((1.0, 0.0),), device=device)
    visited = torch.tensor(((1.0, 0.0),), device=device)
    novel = torch.tensor(((0.0, 1.0),), device=device)
    next_obs = obs.clone()
    valid = torch.ones(1, device=device)

    for _ in range(16):
        _, raw = posterior.information_gain(obs, visited)
        posterior.update(obs, visited, next_obs, valid, raw)

    _, visited_info = posterior.information_gain(obs, visited)
    _, novel_info = posterior.information_gain(obs, novel)
    assert novel_info.item() > visited_info.item()


def test_student_gate_bounds_extreme_write(device):
    args = Args(
        num_envs=1,
        posterior_reset_rate=1e-8,
        posterior_scale_rate=1e-6,
    )
    moderate = RobustDiagonalPosteriorDynamics(1, 2, 1, args, device)
    extreme = RobustDiagonalPosteriorDynamics(1, 2, 1, args, device)
    obs = torch.zeros((1, 2), device=device)
    action = torch.zeros((1, 1), device=device)
    valid = torch.ones(1, device=device)
    _, moderate_raw = moderate.information_gain(obs, action)
    _, extreme_raw = extreme.information_gain(obs, action)

    moderate.update(obs, action, torch.ones_like(obs), valid, moderate_raw)
    extreme.update(obs, action, torch.full_like(obs, 1_000.0), valid, extreme_raw)

    assert extreme.weight.norm().item() < moderate.weight.norm().item()
    assert torch.isfinite(extreme.weight).all()
