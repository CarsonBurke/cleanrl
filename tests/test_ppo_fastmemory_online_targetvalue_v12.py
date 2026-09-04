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
    / "ppo_continuous_action_spf_online_targetvalue_v12.py"
)
spec = importlib.util.spec_from_file_location(
    "ppo_continuous_action_spf_online_targetvalue_v12",
    SCRIPT,
)
assert spec is not None
module = importlib.util.module_from_spec(spec)
loader = spec.loader
assert loader is not None
loader.exec_module(module)
Args = module.Args
Agent = module.Agent
gym = module.gym
one_step_target = module.one_step_target
polyak_update_value = module.polyak_update_value
validate_online_contract = module.validate_online_contract


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "target-value tests require CUDA"
    return torch.device("cuda")


def test_default_configuration_is_single_use_v_critic():
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
        ("target_value_tau", 0.0),
        ("target_value_tau", 1.01),
    ),
)
def test_contract_rejects_nonstreaming_or_confounded_settings(name, value):
    args = Args()
    setattr(args, name, value)
    with pytest.raises(ValueError, match=name):
        validate_online_contract(args)



def test_one_step_target_uses_frozen_successor_and_raw_reward(device):
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


def test_frozen_target_is_separate_until_polyak_update(device):
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-1.0, 1.0, shape=(3,)),
        single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,)),
    )
    source = Agent(envs, Args(hidden=8, k_blocks=1, n_experts=2)).to(device)
    target = copy.deepcopy(source).to(device)
    target.eval()
    target.requires_grad_(False)
    obs = torch.randn(4, 3, device=device)
    context = torch.randn(4, 3, device=device)

    with torch.no_grad():
        before = target.get_value(obs, context).clone()
        for parameter in source.critic_parameters():
            parameter.add_(0.1)
        still_frozen = target.get_value(obs, context)
        polyak_update_value(target, source, 0.5)
        after = target.get_value(obs, context)

    torch.testing.assert_close(still_frozen, before, rtol=0, atol=0)
    assert not torch.equal(after, before)
    assert all(not parameter.requires_grad for parameter in target.parameters())


class _ValueCarrier:
    def __init__(self, critic, actor):
        self.critic = critic
        self.actor = actor

    def critic_parameters(self):
        return list(self.critic.parameters())


def test_polyak_update_moves_only_value_parameters(device):
    source = _ValueCarrier(
        torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 1)).to(device),
        torch.nn.Linear(3, 2).to(device),
    )
    target = _ValueCarrier(
        torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 1)).to(device),
        torch.nn.Linear(3, 2).to(device),
    )
    with torch.no_grad():
        for parameter in source.critic.parameters():
            parameter.fill_(3.0)
        for parameter in target.critic.parameters():
            parameter.fill_(-1.0)
        for parameter in target.actor.parameters():
            parameter.fill_(7.0)

    polyak_update_value(target, source, 0.25)

    for parameter in target.critic.parameters():
        torch.testing.assert_close(parameter, torch.zeros_like(parameter), rtol=0, atol=0)
    for parameter in target.actor.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 7.0), rtol=0, atol=0)


def test_polyak_tau_one_is_exact_hard_copy(device):
    source = _ValueCarrier(torch.nn.Linear(2, 1).to(device), torch.nn.Identity())
    target = _ValueCarrier(torch.nn.Linear(2, 1).to(device), torch.nn.Identity())
    with torch.no_grad():
        source.critic.weight.copy_(torch.tensor(((2.0, -4.0),), device=device))
        source.critic.bias.fill_(1.5)

    polyak_update_value(target, source, 1.0)

    for target_parameter, source_parameter in zip(
        target.critic_parameters(), source.critic_parameters()
    ):
        torch.testing.assert_close(target_parameter, source_parameter, rtol=0, atol=0)


def test_polyak_update_rejects_mismatched_value_structures(device):
    source = SimpleNamespace(
        critic_parameters=lambda: list(torch.nn.Linear(2, 1).to(device).parameters())
    )
    target = SimpleNamespace(
        critic_parameters=lambda: list(torch.nn.Linear(1, 2).to(device).parameters())
    )

    with pytest.raises(ValueError, match="structures differ"):
        polyak_update_value(target, source, 0.5)
