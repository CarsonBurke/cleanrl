import torch

from cleanrl.td7_lesale_v1 import (
    Args,
    DreamerLossNormalizer,
    EncoderLossCore,
    IsometricOutcomeTokens,
    StockEncoder,
)


def _tokens(latent_dim=8, action_dim=2):
    torch.manual_seed(16)
    return IsometricOutcomeTokens(
        latent_dim,
        action_dim,
        64,
        51,
        -40.0,
        40.0,
        0.75,
        True,
        1e-5,
        30.0,
        True,
    )


def test_loss_normalizer_matches_dreamer4_lagged_rms_update():
    normalizer = DreamerLossNormalizer(beta=0.0)
    losses = [torch.tensor(4.0), torch.tensor(-3.0)]
    first = normalizer(losses[0])
    second = normalizer(losses[1])
    third = normalizer(losses[1])
    assert torch.equal(first, torch.tensor(4.0))
    assert torch.equal(second, torch.tensor(-0.75))
    assert torch.equal(third, torch.tensor(-1.0))


def test_v10_equalizes_main_loss_magnitudes_and_compiles_with_ema_updates():
    torch.manual_seed(17)
    encoder = StockEncoder(state_dim=5, action_dim=2, zs_dim=8, hdim=16)
    tokens = _tokens()
    args = Args(
        zs_dim=8,
        hidden_dim=16,
        prediction_from_lap=False,
        outcome_from_transition=True,
        isometric_outcome_tokens=True,
        policy_beta_nll=True,
        policy_beta_nll_coef=1.0,
        policy_beta_max_precision=30.0,
        reward_hlgauss_ce=True,
        outcome_token_coef=1.0,
        dreamer_loss_normalization=True,
        loss_normalization_beta=0.0,
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
    first = core(*inputs)
    second = core(*inputs)
    expected = (
        second[1] / first[1].abs().clamp_min(args.loss_normalization_eps)
        + second[13] / first[13].abs().clamp_min(args.loss_normalization_eps)
        + second[15] / first[15].abs().clamp_min(args.loss_normalization_eps)
    )
    assert torch.allclose(second[0], expected)
    assert torch.allclose(second[1] / first[1].abs(), torch.ones(()))
    assert torch.allclose(second[13] / first[13].abs(), torch.ones(()))
    assert torch.allclose(
        (second[15] / first[15].abs()).abs(), torch.ones(())
    )

    compiled = torch.compile(core, backend="eager", fullgraph=True)
    compiled_outputs = compiled(*inputs)
    assert torch.isfinite(compiled_outputs[0])
