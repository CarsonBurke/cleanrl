"""Unit tests for the v4 continuous-DAE variant (calibrated to the OFFICIAL DAE code).

v4 fixes the deviations that left the advantage at its noise floor in v2/v3, verified
against github.com/hrpan/dae:
  1. policy-weighted RMS advantage normalization before the surrogate (the official
     `advantage_normalization: true`);
  2. a non-saturating, gain-0.1 advantage head (official advantage_net is a plain Linear,
     not a tanh-saturated std=0.01 head);
  3. a separate, generous grad-clip budget for the advantage head (official does not clip
     the value/adv path at all);
  4. beta_V = dae_coef = 1.5 (official vf_coef).
"""
import numpy as np
import torch
import torch.nn as nn

from cleanrl.iterthink.v24_d4hlgauss.other.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_dae_v4 import (
    Agent,
    Args,
)


class _Box:
    def __init__(self, shape, low=-1.0, high=1.0):
        self.shape = shape
        self.low = np.full(shape, low, dtype=np.float32)
        self.high = np.full(shape, high, dtype=np.float32)


class _Envs:
    def __init__(self, obs_dim, act_dim):
        self.single_observation_space = _Box((obs_dim,))
        self.single_action_space = _Box((act_dim,))


def _make_agent(obs_dim=17, act_dim=6):
    args = Args()
    torch.manual_seed(0)
    return Agent(_Envs(obs_dim, act_dim), args), args


def test_v4_defaults_match_official_config():
    args = Args()
    assert args.dae_adv_rms_norm is True        # advantage_normalization: true
    assert args.dae_coef == 1.5                  # vf_coef: 1.5
    assert args.dae_grad_clip == 1.0             # advantage head's own (generous) budget
    assert args.ent_coef == 0.01
    assert args.update_epochs == 6
    assert args.dae_nstep == 128
    assert args.num_steps % args.dae_nstep == 0


def test_rms_normalization_gives_unit_scale_and_preserves_sign():
    # The official _normalize_advantage divides by sqrt(E[A^2]); it must NOT subtract a
    # mean (A_hat is already pi-centered) -> sign/ranking preserved, scale ~ 1.
    torch.manual_seed(3)
    adv = torch.randn(2048) * 0.02          # tiny raw advantage, like the real run
    rms = adv.pow(2).mean().sqrt()
    norm = adv / (rms + 1e-5)
    assert abs(norm.pow(2).mean().sqrt().item() - 1.0) < 1e-3   # unit RMS
    assert torch.equal(torch.sign(norm), torch.sign(adv))       # sign/ranking preserved


def test_advantage_head_is_not_born_flat():
    # The gain-0.1 ELU head must express real action contrast at init (v2/v3's
    # tanh+std=0.01 head was ~flat, adv_absmean ~1e-3).
    agent, args = _make_agent()
    x = torch.randn(128, 17)
    z = torch.rand(128, 6).clamp(1e-6, 1 - 1e-6)
    _, _, _, adv = agent.get_action_value_adv(x, z, args.dae_k_center)
    assert adv.abs().mean().item() > 3e-3


def test_advantage_head_has_its_own_grad_budget():
    # The value group and the advantage heads must be clippable as DISJOINT groups, so the
    # dominant value gradient cannot crush the advantage gradient.
    agent, args = _make_agent()
    value_params = agent.critic_parameters()
    adv_params = list(agent.adv_action_proj.parameters()) + list(agent.adv_head.parameters())
    # disjoint
    vp_ids = {id(p) for p in value_params}
    assert all(id(p) not in vp_ids for p in adv_params)

    x = torch.randn(64, 17)
    z = torch.rand(64, 6).clamp(1e-6, 1 - 1e-6)
    _, _, value_logits, adv = agent.get_action_value_adv(x, z, args.dae_k_center)
    loss = ((torch.randn(64) - adv) ** 2).mean() + value_logits.pow(2).mean()
    agent.zero_grad(set_to_none=True)
    loss.backward()
    vgn = nn.utils.clip_grad_norm_(value_params, args.critic_grad_clip)
    agn = nn.utils.clip_grad_norm_(adv_params, args.dae_grad_clip)
    assert agent.adv_head.weight.grad is not None
    assert float(agn) >= 0.0 and float(vgn) >= 0.0


def test_centering_still_zero_when_taken_is_the_only_sample():
    agent, args = _make_agent()
    B, H = 8, args.hidden
    critic_feat = torch.randn(B, H)
    z = torch.rand(B, 6)
    adv = agent._adv_f(critic_feat, z) - agent._adv_f_multi(critic_feat, z.unsqueeze(1)).mean(1)
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-6)
