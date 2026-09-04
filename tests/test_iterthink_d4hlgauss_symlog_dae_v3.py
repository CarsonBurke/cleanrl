"""Unit tests for the v3 continuous-DAE variant (train-time high-K pi-centering).

v3 fixes v2's centering NOISE: instead of K=4 centering actions frozen at rollout
time, E_pi[f(s,a')] is estimated with K (default 64) FRESH samples from the current
policy at train time. A_hat = f(s,a) - mean_k f(s,a'_k) is zero-mean under pi by
construction for any K; larger K only shrinks the centering variance. These tests
check the new signature, the by-construction centering, the variance reduction with
K, the joint-gradient routing, and the recursion.
"""
import numpy as np
import torch

from cleanrl.iterthink.v24_d4hlgauss.other.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_dae_v3 import (
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


def test_v3_defaults_are_paper_faithful_with_highk_centering():
    args = Args()
    assert args.norm_adv is False          # paper uses raw A_hat (Eq 49)
    assert args.ent_coef == 0.01
    assert args.update_epochs == 6
    assert args.dae_nstep == 128
    assert args.dae_k_center >= 32         # v3: high-K centering (was 4 in v2)
    assert args.num_steps % args.dae_nstep == 0
    assert (args.num_steps // args.dae_nstep) * args.num_envs % args.num_minibatches == 0
    assert args.total_timesteps == 8000000 and args.env_id == "HalfCheetah-v4"
    assert args.value_symlog is True


def test_act_returns_four_values_no_centering_stored():
    agent, args = _make_agent()
    x = torch.randn(7, 5)
    out = agent.act(x)
    assert len(out) == 4
    action, z, log_prob, value_logits = out
    assert action.shape == (7, 3)
    assert z.shape == (7, 3)
    assert log_prob.shape == (7,)
    assert value_logits.shape == (7, args.num_bins)


def test_get_action_value_adv_signature_and_shapes():
    agent, args = _make_agent()
    x = torch.randn(11, 5)
    z = torch.rand(11, 3).clamp(1e-6, 1 - 1e-6)
    logp, ent, vlogits, adv = agent.get_action_value_adv(x, z, args.dae_k_center)
    assert logp.shape == (11,)
    assert ent.shape == (11,)
    assert vlogits.shape == (11, args.num_bins)
    assert adv.shape == (11,)
    assert adv.requires_grad


def test_centering_variance_shrinks_with_k():
    # With a NON-trivial f (give the adv head real weights), the centered advantage's
    # variability across independent centering draws must fall as K grows (~1/sqrt(K)).
    agent, args = _make_agent()
    with torch.no_grad():
        agent.adv_head.weight.mul_(50.0)            # make f(s,a) vary with a
        agent.adv_action_proj.weight.mul_(3.0)
    x = torch.randn(64, 5)
    z = torch.rand(64, 3).clamp(1e-6, 1 - 1e-6)

    def spread(K, trials=8):
        torch.manual_seed(123)
        cols = [agent.get_action_value_adv(x, z, K)[3].detach() for _ in range(trials)]
        return torch.stack(cols, 0).std(0).mean().item()   # mean over states of across-draw std

    s4, s64 = spread(4), spread(64)
    assert s64 < s4 * 0.7   # more samples -> the centering estimate is markedly less noisy


def test_centering_zero_mean_by_construction():
    # E_a~pi[A_hat] ~= 0: averaging the taken-action advantage over many taken actions
    # from the same state collapses to ~0 because A_hat subtracts the policy mean of f.
    agent, args = _make_agent()
    with torch.no_grad():
        agent.adv_head.weight.mul_(50.0)
    x = torch.randn(1, 5).expand(2048, 5).contiguous()
    # taken actions drawn from the policy itself:
    _, z, _, _ = agent.act(x)
    adv = agent.get_action_value_adv(x, z, args.dae_k_center)[3].detach()
    assert abs(adv.mean().item()) < 0.05 * (adv.abs().mean().item() + 1e-6) + 1e-3


def test_joint_dae_loss_trains_value_and_advantage_and_trunk():
    agent, args = _make_agent()
    from cleanrl.shared.hl_gauss import HLGaussSupport
    from cleanrl.iterthink.v24_d4hlgauss.other.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_dae_v3 import value_support_bounds
    smin, smax = value_support_bounds(args)
    hl = HLGaussSupport(args.num_bins, smin, smax, args.value_sigma_to_bin_ratio,
                        torch.device("cpu"), use_symlog=args.value_symlog, support_is_edges=True)
    x = torch.randn(12, 5)
    z = torch.rand(12, 3).clamp(1e-6, 1 - 1e-6)
    _, _, value_logits, adv = agent.get_action_value_adv(x, z, args.dae_k_center)
    v_scalar = hl.to_scalar(value_logits)
    G = (torch.randn(12) - adv)
    dae_loss = ((G - v_scalar) ** 2).mean()
    agent.zero_grad(set_to_none=True)
    dae_loss.backward()
    assert agent.adv_head.weight.grad is not None
    assert agent.critic_head.weight.grad is not None
    assert agent.trunk.out_proj.weight.grad is not None


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
