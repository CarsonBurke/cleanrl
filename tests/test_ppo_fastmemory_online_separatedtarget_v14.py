import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "streaming-posterior-filter"
    / "ppo_continuous_action_spf_online_separatedtarget_v14.py"
)
spec = importlib.util.spec_from_file_location(
    "ppo_continuous_action_spf_online_separatedtarget_v14",
    SCRIPT,
)
assert spec is not None
module = importlib.util.module_from_spec(spec)
loader = spec.loader
assert loader is not None
loader.exec_module(module)
Agent = module.Agent
Args = module.Args
gym = module.gym
one_step_target = module.one_step_target
polyak_update_value = module.polyak_update_value
validate_online_contract = module.validate_online_contract


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "separated-target tests require CUDA"
    return torch.device("cuda")


@pytest.fixture(scope="module")
def envs():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-1.0, 1.0, shape=(3,)),
        single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,)),
    )


def test_default_contract_requires_disjoint_single_use_v_critic():
    validate_online_contract(Args())


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("share_backbone", True),
        ("num_steps", 2),
        ("update_epochs", 2),
        ("num_minibatches", 2),
        ("norm_adv", True),
        ("ret_percnorm", True),
        ("critic_mtp_horizon", 2),
        ("ent_alpha", 0.1),
        ("ent_coef", 0.01),
        ("auto_entropy", True),
        ("target_value_tau", 0.0),
        ("target_value_tau", 1.01),
    ),
)
def test_contract_rejects_confounded_settings(name, value):
    args = Args()
    setattr(args, name, value)

    with pytest.raises(ValueError, match=name):
        validate_online_contract(args)


def test_actor_and_value_parameter_sets_are_disjoint(envs, device):
    agent = Agent(envs, Args(hidden=8, k_blocks=1, n_experts=2)).to(device)

    actor_ids = {id(parameter) for parameter in agent.actor_parameters()}
    critic_ids = {id(parameter) for parameter in agent.critic_parameters()}

    assert actor_ids
    assert critic_ids
    assert actor_ids.isdisjoint(critic_ids)


def test_polyak_update_leaves_target_actor_exactly_unchanged(envs, device):
    source = Agent(envs, Args(hidden=8, k_blocks=1, n_experts=2)).to(device)
    target = copy.deepcopy(source).to(device)
    target_actor_before = [parameter.clone() for parameter in target.actor_parameters()]
    target_critic_before = [
        parameter.clone() for parameter in target.critic_parameters()
    ]
    with torch.no_grad():
        for parameter in source.actor_parameters():
            parameter.add_(2.0)
        for parameter in source.critic_parameters():
            parameter.add_(1.0)

    polyak_update_value(target, source, 0.5)

    for before, after in zip(target_actor_before, target.actor_parameters()):
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    for before, target_parameter, source_parameter in zip(
        target_critic_before,
        target.critic_parameters(),
        source.critic_parameters(),
    ):
        torch.testing.assert_close(
            target_parameter, torch.lerp(before, source_parameter, 0.5)
        )


def test_one_step_target_bootstraps_truncation_but_not_termination(device):
    reward = torch.tensor((2.0, 2.0, 2.0), device=device)
    current = torch.tensor((1.0, 1.0, 1.0), device=device)
    target_next = torch.tensor((10.0, 10.0, 10.0), device=device)
    termination = torch.tensor((0.0, 1.0, 0.0), device=device)
    valid = torch.tensor((1.0, 1.0, 0.0), device=device)

    advantage, target = one_step_target(
        reward, current, target_next, termination, valid, 0.9
    )

    torch.testing.assert_close(
        advantage, torch.tensor((10.0, 1.0, 1.0), device=device)
    )
    torch.testing.assert_close(
        target, torch.tensor((11.0, 2.0, 2.0), device=device)
    )
