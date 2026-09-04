import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "streaming-posterior-filter"
    / "ppo_continuous_action_spf_fastmemory_v2.py"
)
spec = importlib.util.spec_from_file_location("ppo_continuous_action_spf_fastmemory_v2", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
loader = spec.loader
assert loader is not None
loader.exec_module(module)
Agent = module.Agent
Args = module.Args
RobustFastDynamics = module.RobustFastDynamics
gym = module.gym
ONLINE_SCRIPT = SCRIPT.with_name("ppo_continuous_action_spf_online_v3.py")
online_spec = importlib.util.spec_from_file_location(
    "ppo_continuous_action_spf_online_v3",
    ONLINE_SCRIPT,
)
assert online_spec is not None
online_module = importlib.util.module_from_spec(online_spec)
online_loader = online_spec.loader
assert online_loader is not None
online_loader.exec_module(online_module)
OnlineArgs = online_module.Args
OnlineRobustFastDynamics = online_module.RobustFastDynamics
validate_online_contract = online_module.validate_online_contract


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "fast-memory PPO tests require CUDA"
    return torch.device("cuda")


def test_fast_memory_read_precedes_transition_write(device):
    args = Args(num_envs=2)
    memory = RobustFastDynamics(2, 3, args, device)
    obs = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), device=device)
    next_obs = torch.tensor(((2.0, 1.0, 0.0), (0.0, 3.0, 1.0)), device=device)
    valid = torch.ones(2, device=device)

    before = memory.read(obs)
    memory.update(obs, next_obs, valid)
    after = memory.read(obs)

    torch.testing.assert_close(before, torch.zeros_like(before), rtol=0, atol=0)
    assert after.square().sum() > 0
    assert torch.isfinite(after).all()


def test_invalid_transition_never_mutates_memory(device):
    args = Args(num_envs=2)
    memory = RobustFastDynamics(2, 3, args, device)
    obs = torch.randn((2, 3), device=device)
    next_obs = torch.randn((2, 3), device=device)
    before_memory = memory.memory.clone()
    before_scale = memory.scale.clone()

    memory.update(obs, next_obs, torch.zeros(2, device=device))

    torch.testing.assert_close(memory.memory, before_memory, rtol=0, atol=0)
    torch.testing.assert_close(memory.scale, before_scale, rtol=0, atol=0)


def test_student_gate_limits_extreme_transition(device):
    args = Args(num_envs=1)
    memory = RobustFastDynamics(1, 3, args, device)
    obs = torch.tensor(((1.0, 0.0, 0.0),), device=device)
    next_obs = torch.full((1, 3), 1_000.0, device=device)

    memory.update(obs, next_obs, torch.ones(1, device=device))
    diagnostics = memory.diagnostics()

    assert diagnostics["fast_memory/student_weight"] < 0.1
    assert diagnostics["fast_memory/memory_rms"] < 1.0


def test_agent_replays_stored_context_exactly(device):
    torch.manual_seed(7)
    args = Args(hidden=8, k_blocks=1, n_experts=2, critic_mtp_horizon=2, num_bins=11)
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-1.0, 1.0, shape=(3,)),
        single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,)),
    )
    agent = Agent(envs, args).to(device)
    obs = torch.randn((4, 3), device=device)
    context = torch.randn((4, 3), device=device)

    first = agent.get_action_and_value(obs, context)
    replay = agent.get_action_and_value(obs, context, first[1])
    changed = agent.get_action_and_value(obs, context + 1.0, first[1])

    torch.testing.assert_close(first[2], replay[2], rtol=0, atol=0)
    torch.testing.assert_close(first[4], replay[4], rtol=0, atol=0)
    assert not torch.equal(replay[2], changed[2])


def test_online_defaults_consume_each_transition_once():
    args = OnlineArgs()
    validate_online_contract(args)

    assert args.num_steps == 1
    assert args.update_epochs == 1
    assert args.num_minibatches == 1
    assert not args.norm_adv
    assert not args.ret_percnorm
    assert args.critic_mtp_horizon == 1


def test_online_contract_rejects_sample_reuse():
    for name, value in (
        ("num_steps", 2),
        ("num_minibatches", 2),
        ("update_epochs", 2),
        ("norm_adv", True),
        ("ret_percnorm", True),
        ("critic_mtp_horizon", 2),
        ("target_kl", 0.03),
        ("adv_transform", "rankgauss"),
    ):
        args = OnlineArgs()
        setattr(args, name, value)
        with pytest.raises(ValueError, match=name):
            validate_online_contract(args)


def test_online_fast_memory_reads_before_writing(device):
    args = OnlineArgs(num_envs=2)
    memory = OnlineRobustFastDynamics(2, 3, args, device)
    obs = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), device=device)
    next_obs = torch.tensor(((2.0, 1.0, 0.0), (0.0, 3.0, 1.0)), device=device)

    before = memory.read(obs)
    memory.update(obs, next_obs, torch.ones(2, device=device))
    after = memory.read(obs)

    torch.testing.assert_close(before, torch.zeros_like(before), rtol=0, atol=0)
    assert after.square().sum() > 0
