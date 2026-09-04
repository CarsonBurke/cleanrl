import torch

from cleanrl.iterthink.dg.ppo_continuous_action_iterthink_v24_beta_dg_v1 import (
    Args,
    delight_gate,
    dg_raw_surprisal,
)


def test_dg_defaults_are_faithful_score_function():
    args = Args()
    assert args.dg_enable is True
    assert args.dg_mode == "score"
    assert args.dg_surprisal == "raw_clip"
    assert args.dg_eta == 1.0
    assert args.dg_clip == 10.0
    assert args.dg_surprisal_norm is False
    assert args.dg_whiten_chi is False
    assert args.dg_renorm is False


def test_raw_surprisal_entref_subtracts_entropy():
    args = Args(dg_surprisal="entref")
    logprob = torch.randn(32)
    entropy = torch.rand(32) * 2.0
    assert torch.allclose(dg_raw_surprisal(logprob, entropy, args), -logprob - entropy)


def test_raw_surprisal_raw_clip_ignores_entropy():
    args = Args(dg_surprisal="raw_clip")
    logprob = torch.randn(32)
    entropy = torch.rand(32) * 2.0
    assert torch.allclose(dg_raw_surprisal(logprob, entropy, args), -logprob)


def test_delight_gate_in_unit_interval_with_right_shape():
    args = Args()
    torch.manual_seed(0)
    adv = torch.randn(256)
    surp = torch.randn(256)
    gate, surprisal, chi = delight_gate(adv, surp, args)
    assert gate.shape == adv.shape == surprisal.shape == chi.shape
    assert torch.all(gate > 0.0) and torch.all(gate < 1.0)
    assert torch.all(surprisal.abs() <= args.dg_clip + 1e-6)
    assert torch.allclose(chi, adv * surprisal)


def test_entref_typical_action_gives_neutral_half_gate():
    # ell~ = -logp - H. A typical action has -logp ~ H => raw surprisal ~ 0 => gate ~ 0.5,
    # independent of advantage. The Beta-aware anchor.
    args = Args(dg_surprisal="entref")
    entropy = torch.full((128,), 2.0)
    logprob = -entropy  # -logp == entropy => raw surprisal == 0
    surp = dg_raw_surprisal(logprob, entropy, args)
    gate, _, chi = delight_gate(torch.randn(128) * 5.0, surp, args)
    assert torch.allclose(surp, torch.zeros_like(surp), atol=1e-6)
    assert torch.allclose(chi, torch.zeros_like(chi), atol=1e-6)
    assert torch.allclose(gate, torch.full_like(gate, 0.5), atol=1e-6)


def test_entref_amplifies_rare_success_over_rare_failure():
    # Same tail-ness (surprisal>0): positive advantage gets a strictly larger gate.
    args = Args()
    entropy = torch.tensor([2.0, 2.0])
    logprob = torch.tensor([-4.0, -4.0])  # -logp - H = 2 > 0 => tail action
    surp = dg_raw_surprisal(logprob, entropy, args)
    gate, s, _ = delight_gate(torch.tensor([1.0, -1.0]), surp, args)
    assert torch.all(s > 0)
    assert gate[0] > 0.5 > gate[1]


def test_absolute_anchor_preserved_without_whitening():
    # No batch-whitening: the gate reflects ABSOLUTE chi, not rank-within-batch. An all-
    # positive-advantage, all-tail batch must keep EVERY gate above 0.5 (no sample pushed
    # below neutral just because others share the batch).
    args = Args()  # dg_whiten_chi defaults False
    surp = torch.full((64,), 1.5)  # all rarer-than-typical
    adv = torch.rand(64) + 0.1  # all strictly positive
    gate, _, _ = delight_gate(adv, surp, args)
    assert torch.all(gate > 0.5)


def test_whitening_breaks_anchor_pushes_half_below_neutral():
    # Contrast: with batch-whitening the same all-positive batch gets ~half its samples
    # below 0.5 -- the failure mode the expert flagged.
    args = Args(dg_whiten_chi=True)
    surp = torch.full((64,), 1.5)
    adv = torch.rand(64) + 0.1
    gate, _, _ = delight_gate(adv, surp, args)
    assert (gate < 0.5).float().mean() > 0.3


def test_surprisal_norm_scales_by_ell_scale_not_chi():
    # dg_surprisal_norm divides the SURPRISAL by ell_scale (chi still = U * scaled surprisal),
    # so chi -> 0 as U -> 0 is preserved.
    args = Args(dg_surprisal_norm=True)
    adv = torch.randn(128)
    surp = torch.randn(128) * 0.1  # tiny scale, like a near-uniform Beta
    _, s_unit, chi = delight_gate(adv, surp, args, ell_scale=0.1)
    # surprisal scaled up ~10x (then clipped); chi tracks U * scaled surprisal
    assert torch.allclose(chi, adv * surp.div(0.1 + 1e-8).clamp(-args.dg_clip, args.dg_clip), atol=1e-5)
    # zero advantage still gives exactly zero chi (relaxation to PG near optimum)
    _, _, chi0 = delight_gate(torch.zeros(8), torch.randn(8), args, ell_scale=0.05)
    assert torch.allclose(chi0, torch.zeros(8))


def test_score_loss_gradient_flows_through_logprob_only_not_gate():
    args = Args()
    adv = torch.randn(64)
    logprob = torch.randn(64, requires_grad=True)
    surp = torch.rand(64)
    gate, _, _ = delight_gate(adv, surp, args)
    w = gate.detach()
    loss = -(w * adv * logprob).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.allclose(logprob.grad, -(w * adv) / logprob.numel())


def test_gate_ppo_mode_only_downweights():
    args = Args(dg_mode="gate_ppo")
    adv = torch.randn(64)
    ratio = torch.rand(64) * 0.6 + 0.7
    gate, _, _ = delight_gate(adv, torch.randn(64), args)
    w = gate.detach()
    clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
    s = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi))
    assert torch.isfinite((w * s).mean())
    assert torch.all((w * s).abs() <= s.abs() + 1e-6)
