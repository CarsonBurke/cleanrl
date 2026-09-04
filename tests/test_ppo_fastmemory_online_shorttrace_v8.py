import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "streaming-posterior-filter"
    / "ppo_continuous_action_spf_online_shorttrace_v8.py"
)
spec = importlib.util.spec_from_file_location(
    "ppo_continuous_action_spf_online_shorttrace_v8",
    SCRIPT,
)
assert spec is not None
module = importlib.util.module_from_spec(spec)
loader = spec.loader
assert loader is not None
loader.exec_module(module)
Args = module.Args
BetaHeadEligibility = module.BetaHeadEligibility
validate_online_contract = module.validate_online_contract


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "eligibility tests require CUDA"
    return torch.device("cuda")


def test_default_configuration_is_streaming_v_critic():
    args = Args()
    validate_online_contract(args)
    assert args.actor_trace_lambda == 0.5


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
        ("separate_grad_clip", False),
    ),
)
def test_contract_rejects_nonstreaming_or_unsupported_settings(name, value):
    args = Args()
    setattr(args, name, value)

    with pytest.raises(ValueError, match=name):
        validate_online_contract(args)


def test_lambda_zero_matches_autograd_beta_head_gradient(device):
    torch.manual_seed(3)
    batch, act_dim, hidden = 4, 2, 3
    alpha_head = nn.Linear(hidden, act_dim).to(device)
    beta_head = nn.Linear(hidden, act_dim).to(device)
    actor_feat = torch.randn(batch, hidden, device=device)
    z = torch.tensor(
        ((0.2, 0.7), (0.4, 0.8), (0.6, 0.3), (0.9, 0.5)),
        device=device,
    )
    advantage = torch.tensor((1.0, -0.5, 2.0, -1.5), device=device)
    raw_alpha = alpha_head(actor_feat)
    raw_beta = beta_head(actor_feat)
    alpha = 1.0 + F.softplus(raw_alpha)
    beta = 1.0 + F.softplus(raw_beta)
    loss = -(advantage * torch.distributions.Beta(alpha, beta).log_prob(z).sum(1)).mean()
    expected = torch.autograd.grad(
        loss,
        (
            alpha_head.weight,
            alpha_head.bias,
            beta_head.weight,
            beta_head.bias,
        ),
    )
    trace = BetaHeadEligibility(
        batch,
        act_dim,
        hidden,
        Args(actor_trace_lambda=0.0),
        device,
    )

    actual = trace.gradients(
        actor_feat.detach(),
        raw_alpha.detach(),
        raw_beta.detach(),
        z,
        advantage,
        torch.zeros(batch, device=device),
    )

    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_future_td_error_uses_exact_decayed_past_head_score(device):
    args = Args(num_envs=1, actor_trace_lambda=0.9)
    trace = BetaHeadEligibility(1, 1, 2, args, device)
    raw = torch.zeros((1, 1), device=device)
    boundary = torch.zeros(1, device=device)

    first = trace.gradients(
        torch.ones((1, 2), device=device),
        raw,
        raw,
        torch.full((1, 1), 0.25, device=device),
        torch.zeros(1, device=device),
        boundary,
    )
    past_alpha_weight = trace.alpha_weight.clone()
    second = trace.gradients(
        torch.zeros((1, 2), device=device),
        raw,
        raw,
        torch.full((1, 1), 0.5, device=device),
        torch.ones(1, device=device),
        boundary,
    )

    torch.testing.assert_close(first[0], torch.zeros_like(first[0]), rtol=0, atol=0)
    torch.testing.assert_close(second[0], -trace.decay * past_alpha_weight[0])


def test_episode_boundary_credits_then_clears_only_its_stream(device):
    args = Args(num_envs=2, actor_trace_lambda=0.9)
    trace = BetaHeadEligibility(2, 1, 2, args, device)
    raw = torch.zeros((2, 1), device=device)
    features = torch.ones((2, 2), device=device)
    z = torch.full((2, 1), 0.25, device=device)

    trace.gradients(
        features,
        raw,
        raw,
        z,
        torch.zeros(2, device=device),
        torch.zeros(2, device=device),
    )
    past_alpha_weight = trace.alpha_weight.clone()
    terminal_gradient = trace.gradients(
        features,
        raw,
        raw,
        z,
        torch.tensor((1.0, 0.0), device=device),
        torch.tensor((1.0, 0.0), device=device),
    )

    expected = -(1.0 + trace.decay) * past_alpha_weight[0] / 2.0
    torch.testing.assert_close(terminal_gradient[0], expected)
    torch.testing.assert_close(trace.alpha_weight[0], torch.zeros_like(trace.alpha_weight[0]))
    assert trace.alpha_weight[1].abs().sum() > 0
