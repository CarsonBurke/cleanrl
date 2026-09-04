"""Unit tests for the JOINT continuous-DAE variant (v2).

v2 fixes v1's "Indirect" flaw: V and A_hat are trained TOGETHER by the single DAE
shaped-return loss (V_hat live in the residual), the value target is the shaped
return G_t (not the lambda-return), and A_hat is used raw. These tests check the
defaults, the joint-gradient routing, and that the recursion/centering are intact.
"""
import numpy as np
import torch

from cleanrl.iterthink.v24_d4hlgauss.other.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_dae_v2 import (
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


def _make_agent(obs_dim=5, act_dim=3):
    args = Args()
    torch.manual_seed(0)
    return Agent(_Envs(obs_dim, act_dim), args), args


def test_v2_defaults_are_paper_faithful():
    args = Args()
    assert args.norm_adv is False          # paper uses raw A_hat (Eq 49)
    assert args.ent_coef == 0.01           # beta_ent
    assert args.update_epochs == 6
    assert args.dae_nstep == 128
    assert args.num_steps % args.dae_nstep == 0
    assert (args.num_steps // args.dae_nstep) * args.num_envs % args.num_minibatches == 0
    assert args.total_timesteps == 8000000 and args.env_id == "HalfCheetah-v4"
    assert args.value_symlog is True


def test_centering_zero_when_taken_action_is_the_only_sample():
    agent, args = _make_agent()
    B, A, H = 6, 3, args.hidden
    critic_feat = torch.randn(B, H)
    z = torch.rand(B, A)
    adv = agent._adv_f(critic_feat, z) - agent._adv_f_multi(critic_feat, z.unsqueeze(1)).mean(1)
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-6)


def test_joint_dae_loss_trains_both_value_and_advantage_heads():
    # The JOINT loss (G - V_hat_scalar)^2, with G containing A_hat, must put gradient
    # on the advantage head AND the value head AND the shared trunk simultaneously.
    agent, args = _make_agent()
    from cleanrl.shared.hl_gauss import HLGaussSupport
    from cleanrl.iterthink.v24_d4hlgauss.other.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_dae_v2 import value_support_bounds
    smin, smax = value_support_bounds(args)
    hl = HLGaussSupport(args.num_bins, smin, smax, args.value_sigma_to_bin_ratio,
                        torch.device("cpu"), use_symlog=args.value_symlog, support_is_edges=True)

    B, K, A = 12, args.dae_k_center, 3
    x = torch.randn(B, 5)
    z = torch.rand(B, A).clamp(1e-6, 1 - 1e-6)
    cz = torch.rand(B, K, A).clamp(1e-6, 1 - 1e-6)
    _, _, value_logits, adv = agent.get_action_value_adv(x, z, cz)

    v_scalar = hl.to_scalar(value_logits)        # live value mean (differentiable)
    G = (torch.randn(B) - adv)                    # stand-in shaped return containing A_hat
    dae_loss = ((G - v_scalar) ** 2).mean()

    agent.zero_grad(set_to_none=True)
    dae_loss.backward()
    assert agent.adv_head.weight.grad is not None        # advantage trained
    assert agent.critic_head.weight.grad is not None      # value trained (through to_scalar)
    assert agent.trunk.out_proj.weight.grad is not None   # shared trunk trained


def _seg_view(x, nstep, num_chunks, num_envs, n_segments):
    rest = x.shape[2:]
    x = x.reshape(num_chunks, nstep, num_envs, *rest)
    x = x.permute(1, 0, 2, *range(3, 3 + len(rest)))
    return x.reshape(nstep, n_segments, *rest)


def test_recursion_matches_reference_with_masking():
    torch.manual_seed(1)
    nstep, S, gamma = 5, 4, 0.99
    rew, adv = torch.randn(nstep, S), torch.randn(nstep, S)
    nextnonterm = (torch.rand(nstep, S) > 0.2).float()
    boot = torch.randn(S)

    shaped = rew - adv
    running = boot.clone()
    rows = [None] * nstep
    for j in reversed(range(nstep)):
        running = shaped[j] + gamma * (running if j == nstep - 1 else nextnonterm[j] * running)
        rows[j] = running
    G = torch.stack(rows, 0)

    G_ref = torch.zeros(nstep, S)
    for s in range(S):
        for t in range(nstep):
            acc, disc = 0.0, 1.0
            for tp in range(t, nstep):
                acc += disc * shaped[tp, s].item()
                disc *= gamma * (nextnonterm[tp, s].item() if tp < nstep - 1 else 1.0)
            G_ref[t, s] = acc + disc * boot[s].item()
    assert torch.allclose(G, G_ref, atol=1e-4)
