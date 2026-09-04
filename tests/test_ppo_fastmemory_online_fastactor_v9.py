import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "streaming-posterior-filter"
    / "ppo_continuous_action_spf_online_fastactor_v9.py"
)
spec = importlib.util.spec_from_file_location(
    "ppo_continuous_action_spf_online_fastactor_v9",
    SCRIPT,
)
assert spec is not None
module = importlib.util.module_from_spec(spec)
loader = spec.loader
assert loader is not None
loader.exec_module(module)
Agent = module.Agent
Args = module.Args
LocalFastActor = module.LocalFastActor
gym = module.gym
validate_online_contract = module.validate_online_contract


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "fast-actor tests require CUDA"
    return torch.device("cuda")


def test_default_configuration_is_streaming_v_critic():
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
        ("actor_dist", "gaussian"),
    ),
)
def test_contract_rejects_nonstreaming_or_unsupported_settings(name, value):
    args = Args()
    setattr(args, name, value)

    with pytest.raises(ValueError, match=name):
        validate_online_contract(args)


def test_local_update_matches_beta_logprob_autograd(device):
    args = Args(
        num_envs=1,
        fast_actor_eta=0.01,
        fast_actor_leak=1.0,
        fast_actor_signal_clip=100.0,
        fast_actor_score_clip=100.0,
    )
    fast_actor = LocalFastActor(1, 1, 2, args, device)
    obs = torch.tensor(((1.0, 0.0),), device=device)
    z = torch.tensor(((0.8,),), device=device)
    alpha = torch.tensor(((2.0,),), device=device)
    beta = torch.tensor(((3.0,),), device=device)
    td_error = torch.tensor((1.5,), device=device)
    alpha_excess = alpha - 1.0
    beta_excess = beta - 1.0
    raw_alpha = (alpha_excess + torch.log(-torch.expm1(-alpha_excess))).requires_grad_()
    raw_beta = (beta_excess + torch.log(-torch.expm1(-beta_excess))).requires_grad_()
    distribution = torch.distributions.Beta(
        1.0 + F.softplus(raw_alpha),
        1.0 + F.softplus(raw_beta),
    )
    alpha_score, beta_score = torch.autograd.grad(
        distribution.log_prob(z).sum(),
        (raw_alpha, raw_beta),
    )

    fast_actor.update(obs, z, alpha, beta, td_error)

    expected_alpha = args.fast_actor_eta * td_error * alpha_score[:, 0]
    expected_beta = args.fast_actor_eta * td_error * beta_score[:, 0]
    torch.testing.assert_close(fast_actor.alpha_weight[0, 0, 0], expected_alpha[0])
    torch.testing.assert_close(fast_actor.beta_weight[0, 0, 0], expected_beta[0])
    torch.testing.assert_close(fast_actor.alpha_weight[0, 0, 1], torch.zeros((), device=device))


def test_fast_actor_reads_before_update_and_isolates_streams(device):
    args = Args(num_envs=2, fast_actor_leak=1.0)
    fast_actor = LocalFastActor(2, 1, 2, args, device)
    obs = torch.tensor(((1.0, 0.0), (0.0, 1.0)), device=device)
    z = torch.full((2, 1), 0.8, device=device)
    alpha = torch.full((2, 1), 2.0, device=device)
    beta = torch.full((2, 1), 3.0, device=device)

    before = fast_actor.read(obs)
    fast_actor.update(obs, z, alpha, beta, torch.tensor((1.0, 0.0), device=device))
    after = fast_actor.read(obs)

    torch.testing.assert_close(before[0], torch.zeros_like(before[0]), rtol=0, atol=0)
    assert after[0][0].abs().sum() > 0
    torch.testing.assert_close(after[0][1], torch.zeros_like(after[0][1]), rtol=0, atol=0)
    torch.testing.assert_close(after[1][1], torch.zeros_like(after[1][1]), rtol=0, atol=0)


def test_agent_replays_stored_fast_offsets_exactly(device):
    torch.manual_seed(11)
    args = Args(hidden=8, k_blocks=1, n_experts=2, num_bins=11)
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-1.0, 1.0, shape=(3,)),
        single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,)),
    )
    agent = Agent(envs, args).to(device)
    obs = torch.randn((4, 3), device=device)
    context = torch.randn((4, 3), device=device)
    offsets = (
        torch.randn((4, 2), device=device),
        torch.randn((4, 2), device=device),
    )

    first = agent.get_action_and_value(obs, context, fast_offsets=offsets)
    replay = agent.get_action_and_value(
        obs,
        context,
        first[1],
        fast_offsets=offsets,
    )
    changed = agent.get_action_and_value(
        obs,
        context,
        first[1],
        fast_offsets=(offsets[0] + 1.0, offsets[1]),
    )

    torch.testing.assert_close(first[2], replay[2], rtol=0, atol=0)
    assert not torch.equal(replay[2], changed[2])
