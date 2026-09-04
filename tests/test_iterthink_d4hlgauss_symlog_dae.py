"""Unit tests for the continuous DAE (Direct Advantage Estimation) variant.

These validate the DAE-specific pieces in isolation (no training): the advantage
head + pi-centering shapes, the trajectory-segment view round-trip and ordering,
and the backward n-step shaped-return recursion against a reference implementation.
"""
import numpy as np
import torch

from cleanrl.iterthink.v24_d4hlgauss.other.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_dae_v1 import (
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
    agent = Agent(_Envs(obs_dim, act_dim), args)
    return agent, args


def test_dae_defaults():
    args = Args()
    assert args.norm_adv is True           # affine standardize: preserves ranking/relative magnitude, fixes scale
    assert args.total_timesteps == 8000000
    assert args.env_id == "HalfCheetah-v4"
    assert args.num_steps % args.dae_nstep == 0
    n_seg = (args.num_steps // args.dae_nstep) * args.num_envs
    assert n_seg % args.num_minibatches == 0
    assert args.value_symlog is True and args.v_min == -10.0 and args.v_max == 10.0


def test_adv_head_shapes_and_centering():
    agent, args = _make_agent()
    B, K, A, H = 7, args.dae_k_center, 3, args.hidden
    critic_feat = torch.randn(B, H)
    z = torch.rand(B, A)
    cz = torch.rand(B, K, A)

    f_taken = agent._adv_f(critic_feat, z)
    f_multi = agent._adv_f_multi(critic_feat, cz)
    assert f_taken.shape == (B,)
    assert f_multi.shape == (B, K)

    # pi-centered advantage = f(s,a) - mean_k f(s,a'_k): when the taken action IS one
    # of the centering samples and K=1, the centered advantage is exactly zero.
    cz1 = z.unsqueeze(1)  # (B,1,A) == taken action
    adv = agent._adv_f(critic_feat, z) - agent._adv_f_multi(critic_feat, cz1).mean(1)
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-6)


def test_get_action_value_adv_shapes():
    agent, args = _make_agent()
    B, K, A = 11, args.dae_k_center, 3
    x = torch.randn(B, 5)
    z = torch.rand(B, A).clamp(1e-6, 1 - 1e-6)
    cz = torch.rand(B, K, A).clamp(1e-6, 1 - 1e-6)
    logp, ent, vlogits, adv = agent.get_action_value_adv(x, z, cz)
    assert logp.shape == (B,)
    assert ent.shape == (B,)
    assert vlogits.shape == (B, args.num_bins)
    assert adv.shape == (B,)
    assert adv.requires_grad  # advantage carries grad for L_A


def _seg_view(x, nstep, num_chunks, num_envs, n_segments):
    rest = x.shape[2:]
    x = x.reshape(num_chunks, nstep, num_envs, *rest)
    x = x.permute(1, 0, 2, *range(3, 3 + len(rest)))
    return x.reshape(nstep, n_segments, *rest)


def test_seg_view_ordering_roundtrip():
    # A segment column must hold a CONTIGUOUS, time-ordered slice of one env.
    num_steps, num_envs, nstep = 8, 2, 4
    num_chunks = num_steps // nstep
    n_segments = num_chunks * num_envs
    # tag[t, e] = t*100 + e so we can read back time/env from the value.
    t = torch.arange(num_steps).view(-1, 1).expand(num_steps, num_envs)
    e = torch.arange(num_envs).view(1, -1).expand(num_steps, num_envs)
    tag = (t * 100 + e).float()
    v = _seg_view(tag, nstep, num_chunks, num_envs, n_segments)  # (nstep, n_segments)
    assert v.shape == (nstep, n_segments)
    for seg in range(n_segments):
        col = v[:, seg]
        env = int(col[0].item()) % 100
        times = [int(col[j].item()) // 100 for j in range(nstep)]
        # all same env, strictly consecutive times
        assert all((int(col[j].item()) % 100) == env for j in range(nstep))
        assert times == list(range(times[0], times[0] + nstep))


def test_dae_recursion_matches_reference():
    # The vectorized backward recursion must equal an explicit per-step n-step
    # shaped-return-to-go with episode masking + bootstrap.
    torch.manual_seed(1)
    nstep, S = 5, 6
    gamma = 0.99
    rew = torch.randn(nstep, S)
    adv = torch.randn(nstep, S)
    nextnonterm = (torch.rand(nstep, S) > 0.2).float()  # some episode boundaries
    boot = torch.randn(S)

    shaped = rew - adv
    running = boot.clone()
    R_rows = [None] * nstep
    for j in reversed(range(nstep)):
        if j == nstep - 1:
            running = shaped[j] + gamma * running
        else:
            running = shaped[j] + gamma * nextnonterm[j] * running
        R_rows[j] = running
    R = torch.stack(R_rows, dim=0)

    # Reference: R[t] = sum_{t'=t}^{end} prod(gamma * masks) (r-adv)_{t'} + tail bootstrap.
    R_ref = torch.zeros(nstep, S)
    for s in range(S):
        for t in range(nstep):
            acc = 0.0
            disc = 1.0
            broke = False
            for tp in range(t, nstep):
                acc = acc + disc * shaped[tp, s].item()
                if tp < nstep - 1:
                    disc = disc * gamma * nextnonterm[tp, s].item()
                else:
                    disc = disc * gamma  # last step -> bootstrap weight
            acc = acc + disc * boot[s].item()
            R_ref[t, s] = acc
    assert torch.allclose(R, R_ref, atol=1e-4)


def test_triple_backward_grad_routing():
    # value-CE grads must NOT touch the adv head; DAE grads must NOT touch the
    # critic head; both must reach the shared trunk.
    agent, args = _make_agent()
    B, K, A = 16, args.dae_k_center, 3
    x = torch.randn(B, 5)
    z = torch.rand(B, A).clamp(1e-6, 1 - 1e-6)
    cz = torch.rand(B, K, A).clamp(1e-6, 1 - 1e-6)
    _, _, vlogits, adv = agent.get_action_value_adv(x, z, cz)

    # value loss grad
    agent.zero_grad(set_to_none=True)
    vlogits.sum().backward(retain_graph=True)
    assert agent.adv_head.weight.grad is None  # value path does not feed adv head
    assert agent.critic_head.weight.grad is not None
    trunk_w = agent.trunk.out_proj.weight
    assert trunk_w.grad is not None

    # dae loss grad
    agent.zero_grad(set_to_none=True)
    adv.sum().backward()
    assert agent.critic_head.weight.grad is None  # adv path does not feed critic head
    assert agent.adv_head.weight.grad is not None
    assert agent.trunk.out_proj.weight.grad is not None
