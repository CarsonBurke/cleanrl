import torch
import torch.nn.functional as F

from cleanrl.td7.lesale.td7_lesale_v1 import (
    Args,
    EncoderLossCore,
    IsometricOutcomeTokens,
    StockEncoder,
)


def _tokens(latent_dim=32, action_dim=6, direct_reward=True):
    torch.manual_seed(12)
    return IsometricOutcomeTokens(
        latent_dim=latent_dim,
        action_dim=action_dim,
        token_dim=64,
        reward_num_bins=51,
        reward_raw_min=-40.0,
        reward_raw_max=40.0,
        reward_sigma_ratio=0.75,
        policy_beta_nll=True,
        policy_beta_nll_eps=1e-5,
        policy_beta_max_precision=30.0,
        reward_hlgauss_ce=direct_reward,
    )


def test_direct_reward_uses_hlgauss_logits_without_a_target_encoder():
    tokens = _tokens()
    transition = torch.randn(8, 32, requires_grad=True)
    action = torch.rand(8, 6) * 2.0 - 1.0
    reward = torch.linspace(-50.0, 50.0, 8).unsqueeze(-1)
    policy_moments = torch.rand(8, 12) * 2.0 - 1.0
    reward_logits, reward_target, policy_raw, policy_target = tokens(
        None, action, reward, policy_moments, transition
    )[:4]
    reward_ce = -(reward_target * F.log_softmax(reward_logits, dim=-1)).sum(-1).mean()
    reward_ce.backward()

    assert reward_logits.shape == (8, 51)
    assert reward_target.shape == (8, 51)
    assert torch.allclose(reward_target.sum(-1), torch.ones(8))
    assert policy_raw.shape == (8, 24)
    assert policy_target is policy_moments
    assert not hasattr(tokens, "reward_tokenizer")
    assert tokens.gradient_parameter_groups()[0] == []
    zero_prior_logits = tokens.project_reward(torch.zeros(1)).clamp_min(1e-20).log()
    assert torch.allclose(reward_logits, zero_prior_logits.expand_as(reward_logits))
    assert float(tokens.reward_readout[-1].weight.grad.abs().sum()) > 0.0

    # v215-style prior calibration deliberately firewalls the transition on step one. Once the
    # zero-initialized readout has moved, direct reward gradients must reach the world transition.
    with torch.no_grad():
        tokens.reward_readout[-1].weight.add_(
            tokens.reward_readout[-1].weight.grad, alpha=-1e-3
        )
    tokens.zero_grad(set_to_none=True)
    transition.grad = None
    updated_logits = tokens(None, action, reward, policy_moments, transition)[0]
    updated_ce = -(
        reward_target * F.log_softmax(updated_logits, dim=-1)
    ).sum(-1).mean()
    updated_ce.backward()
    assert float(transition.grad.abs().sum()) > 0.0


def test_direct_reward_keeps_v8_policy_branch_identical():
    torch.manual_seed(13)
    v8 = _tokens(direct_reward=False)
    torch.manual_seed(13)
    v9 = _tokens(direct_reward=True)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            v8.policy_predictor.state_dict().values(),
            v9.policy_predictor.state_dict().values(),
        )
    )


def test_encoder_combines_scaled_hlgauss_and_bounded_beta_and_compiles():
    torch.manual_seed(14)
    encoder = StockEncoder(state_dim=5, action_dim=2, zs_dim=8, hdim=16)
    tokens = _tokens(latent_dim=8, action_dim=2)
    args = Args(
        zs_dim=8,
        hidden_dim=16,
        prediction_from_lap=False,
        outcome_from_transition=True,
        isometric_outcome_tokens=True,
        policy_beta_nll=True,
        policy_beta_nll_coef=0.05,
        policy_beta_max_precision=30.0,
        reward_hlgauss_ce=True,
        outcome_token_coef=0.05,
        sd_noise=True,
    )
    core = EncoderLossCore(encoder, None, None, args, outcome_tokens=tokens)
    inputs = (
        torch.randn(3, 5),
        torch.rand(3, 2) * 2.0 - 1.0,
        torch.randn(3, 5),
        torch.randn(3, 5),
        torch.rand(3, 2) * 2.0 - 1.0,
        torch.randn(3, 5),
        torch.randn(3, 1),
        torch.rand(3, 4) * 2.0 - 1.0,
        torch.ones(3, 1),
    )
    outputs = core(*inputs)
    expected_total = (
        outputs[1]
        + args.subsig_coef * outputs[2]
        + args.outcome_token_coef * outputs[13]
        + args.policy_beta_nll_coef * outputs[15]
    )
    assert torch.allclose(outputs[0], expected_total)
    assert torch.isfinite(outputs[13])

    compiled = torch.compile(core, backend="eager", fullgraph=True)
    compiled_outputs = compiled(*inputs)
    assert torch.allclose(compiled_outputs[0], outputs[0])
