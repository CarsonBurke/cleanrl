import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "streaming-posterior-filter"
    / "ppo_continuous_action_spf_online_epistemictoken_v15.py"
)
spec = importlib.util.spec_from_file_location(
    "ppo_continuous_action_spf_online_epistemictoken_v15",
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
resample_episode_tokens = module.resample_episode_tokens
validate_online_contract = module.validate_online_contract


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "epistemic-token tests require CUDA"
    return torch.device("cuda")


@pytest.fixture(scope="module")
def envs():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-1.0, 1.0, shape=(3,)),
        single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,)),
    )


def test_default_contract_is_single_use_beta_token_policy():
    validate_online_contract(Args())


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("num_steps", 2),
        ("update_epochs", 2),
        ("num_minibatches", 2),
        ("norm_adv", True),
        ("ret_percnorm", True),
        ("critic_mtp_horizon", 2),
        ("ent_alpha", 0.1),
        ("ent_coef", 0.01),
        ("auto_entropy", True),
        ("actor_dist", "gaussian"),
        ("normalize_reward", True),
        ("clip_reward", True),
        ("policy_token_rank", 0),
        ("policy_token_scale", 0.0),
        ("policy_token_offset_clip", 0.0),
    ),
)
def test_contract_rejects_confounded_settings(name, value):
    args = Args()
    setattr(args, name, value)

    with pytest.raises(ValueError, match=name):
        validate_online_contract(args)


def test_token_is_episode_persistent_and_only_boundaries_resample(device):
    torch.manual_seed(7)
    tokens = torch.randn(3, 4, device=device)
    boundaries = torch.tensor((0.0, 1.0, 0.0), device=device)

    updated = resample_episode_tokens(tokens, boundaries)

    torch.testing.assert_close(updated[0], tokens[0], rtol=0, atol=0)
    assert not torch.equal(updated[1], tokens[1])
    torch.testing.assert_close(updated[2], tokens[2], rtol=0, atol=0)


def test_different_tokens_change_beta_policy_but_not_value(envs, device):
    args = Args(hidden=8, k_blocks=1, n_experts=2, policy_token_rank=4)
    agent = Agent(envs, args).to(device)
    obs = torch.randn(5, 3, device=device)
    context = torch.randn(5, 3, device=device)
    native_action = torch.full((5, 2), 0.5, device=device)
    first_token = torch.ones(5, 4, device=device)
    second_token = -first_token

    _, _, first_logprob, _, first_value, first_dist = agent.get_action_and_value(
        obs, context, native_action, first_token
    )
    _, _, second_logprob, _, second_value, second_dist = agent.get_action_and_value(
        obs, context, native_action, second_token
    )

    assert not torch.equal(first_dist.concentration1, second_dist.concentration1)
    assert not torch.equal(first_logprob, second_logprob)
    torch.testing.assert_close(first_value, second_value, rtol=0, atol=0)


def test_stored_token_and_native_action_reproduce_behavior_logprob(envs, device):
    args = Args(hidden=8, k_blocks=1, n_experts=2, policy_token_rank=4)
    agent = Agent(envs, args).to(device)
    obs = torch.randn(6, 3, device=device)
    context = torch.randn(6, 3, device=device)
    token = torch.randn(6, 4, device=device)

    _, native_action, behavior_logprob, _, _, behavior_dist = agent.get_action_and_value(
        obs, context, actor_token=token
    )
    _, _, replay_logprob, _, _, replay_dist = agent.get_action_and_value(
        obs, context, native_action, token
    )

    torch.testing.assert_close(replay_logprob, behavior_logprob, rtol=0, atol=0)
    torch.testing.assert_close(
        replay_dist.concentration1, behavior_dist.concentration1, rtol=0, atol=0
    )
    torch.testing.assert_close(
        replay_dist.concentration0, behavior_dist.concentration0, rtol=0, atol=0
    )


def test_token_logit_effect_is_smoothly_bounded(envs, device):
    args = Args(
        hidden=8,
        k_blocks=1,
        n_experts=2,
        policy_token_rank=4,
        policy_token_offset_clip=0.2,
    )
    agent = Agent(envs, args).to(device)
    actor_feature = torch.randn(8, 8, device=device)
    huge_token = torch.full((8, 4), 1e6, device=device)

    agent._actor_dist(actor_feature, huge_token)

    assert agent.last_token_offset_rms <= args.policy_token_offset_clip
    assert torch.isfinite(agent.last_token_offset_rms)
