import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "streaming-posterior-filter"
    / "ppo_continuous_action_spf_hybrid_robustsurprise_v17.py"
)
spec = importlib.util.spec_from_file_location(
    "ppo_continuous_action_spf_hybrid_robustsurprise_v17", SCRIPT
)
assert spec is not None
module = importlib.util.module_from_spec(spec)
loader = spec.loader
assert loader is not None
loader.exec_module(module)
Args = module.Args
RobustFastDynamics = module.RobustFastDynamics
propagate_surprise_advantage = module.propagate_surprise_advantage
validate_surprise_contract = module.validate_surprise_contract


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "robust-surprise tests require CUDA"
    return torch.device("cuda")


def test_default_surprise_contract_is_valid():
    validate_surprise_contract(Args())


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("ent_alpha", 0.1),
        ("ent_coef", 0.01),
        ("auto_entropy", True),
        ("normalize_reward", True),
        ("clip_reward", True),
        ("surprise_coef", 0.0),
        ("surprise_baseline_rate", 0.0),
        ("surprise_baseline_rate", 1.0),
        ("surprise_bonus_clip", 0.0),
    ),
)
def test_contract_rejects_confounds(name, value):
    args = Args()
    setattr(args, name, value)
    with pytest.raises(ValueError, match=name):
        validate_surprise_contract(args)


def test_surprise_credit_stops_at_boundary(device):
    bonuses = torch.tensor(((1.0,), (2.0,), (3.0,)), device=device)
    boundaries = torch.tensor(((0.0,), (1.0,), (0.0,)), device=device)

    result = propagate_surprise_advantage(bonuses, boundaries, 0.9, 0.5)

    torch.testing.assert_close(
        result, torch.tensor(((1.9,), (2.0,), (3.0,)), device=device)
    )


def test_student_gate_rejects_extreme_surprise(device):
    args = Args(num_envs=1, surprise_baseline_rate=1e-6)
    moderate = RobustFastDynamics(1, 2, args, device)
    extreme = RobustFastDynamics(1, 2, args, device)
    obs = torch.zeros((1, 2), device=device)
    valid = torch.ones(1, device=device)

    moderate_bonus = moderate.update(obs, torch.ones_like(obs), valid)
    extreme_bonus = extreme.update(obs, torch.full_like(obs, 1_000.0), valid)

    assert moderate_bonus.item() > 0
    assert extreme_bonus.item() < moderate_bonus.item()
    assert torch.isfinite(extreme_bonus).all()


def test_invalid_transition_neither_writes_nor_rewards(device):
    args = Args(num_envs=2)
    memory = RobustFastDynamics(2, 2, args, device)
    obs = torch.tensor(((1.0, 0.0), (0.0, 1.0)), device=device)
    next_obs = obs + 1.0
    valid = torch.tensor((1.0, 0.0), device=device)

    bonus = memory.update(obs, next_obs, valid)

    assert bonus[0] != 0
    torch.testing.assert_close(bonus[1], torch.zeros_like(bonus[1]), rtol=0, atol=0)
    torch.testing.assert_close(
        memory.memory[1], torch.zeros_like(memory.memory[1]), rtol=0, atol=0
    )
