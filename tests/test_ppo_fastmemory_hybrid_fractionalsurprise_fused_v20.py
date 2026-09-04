import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "streaming-posterior-filter"
    / "ppo_continuous_action_spf_hybrid_fractionalsurprise_fused_v20.py"
)
spec = importlib.util.spec_from_file_location(
    "ppo_continuous_action_spf_hybrid_fractionalsurprise_fused_v20", SCRIPT
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

INCUMBENT_SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "streaming-posterior-filter"
    / "ppo_continuous_action_spf_hybrid_fractionalsurprise_v19.py"
)
incumbent_spec = importlib.util.spec_from_file_location(
    "ppo_continuous_action_spf_hybrid_fractionalsurprise_v19",
    INCUMBENT_SCRIPT,
)
assert incumbent_spec is not None
incumbent_module = importlib.util.module_from_spec(incumbent_spec)
incumbent_loader = incumbent_spec.loader
assert incumbent_loader is not None
incumbent_loader.exec_module(incumbent_module)
IncumbentArgs = incumbent_module.Args
IncumbentRobustFastDynamics = incumbent_module.RobustFastDynamics


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "fused fractional-surprise tests require CUDA"
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

def test_contract_rejects_unknown_compile_mode():
    args = Args(fast_memory_compile_mode="reduce-overhead")
    with pytest.raises(ValueError, match="fast_memory_compile_mode"):
        validate_surprise_contract(args)


def test_surprise_credit_stops_at_boundary(device):
    bonuses = torch.tensor(((1.0,), (2.0,), (3.0,)), device=device)
    boundaries = torch.tensor(((0.0,), (1.0,), (0.0,)), device=device)

    result = propagate_surprise_advantage(bonuses, boundaries, 0.9, 0.5)

    torch.testing.assert_close(
        result, torch.tensor(((1.9,), (2.0,), (3.0,)), device=device)
    )


def test_bonus_uses_fractional_variance_normalization(device):
    args = Args(num_envs=1, compile_fast_memory=False)
    memory = RobustFastDynamics(1, 2, args, device)
    memory.surprise_mean.fill_(0.2)
    memory.surprise_var.fill_(0.0625)
    obs = torch.zeros((1, 2), device=device)
    valid = torch.ones(1, device=device)

    bonus = memory.update(obs, torch.ones_like(obs), valid)

    expected_error = torch.log(torch.tensor(2.0, device=device)) - 0.2
    surprise_std = torch.tensor(
        0.0625 + (0.05 * 0.2) ** 2, device=device
    ).sqrt()
    expected = expected_error / surprise_std.sqrt()
    torch.testing.assert_close(bonus.squeeze(0), expected)
    assert expected_error < bonus < expected_error / surprise_std


def test_repeated_predictable_surprise_self_anneals(device):
    args = Args(num_envs=1, surprise_baseline_rate=0.1, compile_fast_memory=False)
    memory = RobustFastDynamics(1, 2, args, device)
    obs = torch.zeros((1, 2), device=device)
    next_obs = torch.ones_like(obs)
    valid = torch.ones(1, device=device)

    first_bonus = memory.update(obs, next_obs, valid)
    final_bonus = first_bonus
    for _ in range(100):
        final_bonus = memory.update(obs, next_obs, valid)

    assert first_bonus.item() > 0
    assert 0 <= final_bonus.item() < 1e-3 * first_bonus.item()


def test_student_gate_rejects_extreme_surprise(device):
    args = Args(num_envs=1, surprise_baseline_rate=1e-6, compile_fast_memory=False)
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
    args = Args(num_envs=2, compile_fast_memory=False)
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


def test_eager_recurrence_matches_v19(device):
    incumbent = IncumbentRobustFastDynamics(
        2, 3, IncumbentArgs(num_envs=2), device
    )
    fused = RobustFastDynamics(
        2, 3, Args(num_envs=2, compile_fast_memory=False), device
    )
    generator = torch.Generator(device=device).manual_seed(20260822)

    for step in range(16):
        obs = torch.randn((2, 3), device=device, generator=generator)
        next_obs = torch.randn((2, 3), device=device, generator=generator)
        valid = torch.tensor(
            (1.0, float(step % 3 != 0)), device=device
        )
        torch.testing.assert_close(fused.read(obs), incumbent.read(obs))
        incumbent_bonus = incumbent.update(obs, next_obs, valid)
        fused_bonus = fused.update(obs, next_obs, valid)

        torch.testing.assert_close(fused_bonus, incumbent_bonus)
        torch.testing.assert_close(fused.memory, incumbent.memory)
        torch.testing.assert_close(fused.scale, incumbent.scale)
        torch.testing.assert_close(
            fused.surprise_mean, incumbent.surprise_mean
        )
        torch.testing.assert_close(
            fused.surprise_var, incumbent.surprise_var
        )
        torch.testing.assert_close(
            fused.diagnostic_sum, incumbent.diagnostic_sum
        )


def test_compiled_recurrence_matches_eager(device):
    eager = RobustFastDynamics(
        2, 3, Args(num_envs=2, compile_fast_memory=False), device
    )
    compiled = RobustFastDynamics(
        2, 3, Args(num_envs=2, compile_fast_memory=True), device
    )
    generator = torch.Generator(device=device).manual_seed(20260823)

    for step in range(8):
        obs = torch.randn((2, 3), device=device, generator=generator)
        next_obs = torch.randn((2, 3), device=device, generator=generator)
        valid = torch.tensor(
            (1.0, float(step % 2 == 0)), device=device
        )
        torch.testing.assert_close(compiled.read(obs), eager.read(obs))
        eager_bonus = eager.update(obs, next_obs, valid)
        compiled_bonus = compiled.update(obs, next_obs, valid)

        torch.testing.assert_close(compiled_bonus, eager_bonus)
        torch.testing.assert_close(compiled.memory, eager.memory)
        torch.testing.assert_close(compiled.scale, eager.scale)
        torch.testing.assert_close(
            compiled.surprise_mean, eager.surprise_mean
        )
        torch.testing.assert_close(
            compiled.surprise_var, eager.surprise_var
        )
        torch.testing.assert_close(
            compiled.diagnostic_sum, eager.diagnostic_sum
        )

    compiled.release_compiled_graphs()
    assert not compiled.compiled
    torch.compiler.reset()
