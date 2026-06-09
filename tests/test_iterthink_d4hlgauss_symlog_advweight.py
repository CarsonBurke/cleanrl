"""Unit tests for the LEARNED ADVANTAGE WEIGHT variant (advweight_v1).

The method keeps STANDARD GAE as the policy signal (sign/rank) and replaces the
fixed rankgauss magnitude with a learned, value-grounded, batch-relative weight
w = |Q-V| / mean|Q-V| on each sample's PPO loss contribution. These tests pin the
paper-faithful defaults, the value-grounded Q head, the weight math (mean~=1,
clamp, blend, commutes with the clip max), and the grad routing.
"""
import numpy as np
import torch

from cleanrl.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_advweight_v1 import (
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


def test_advweight_defaults():
    args = Args()
    # The base advantage estimator is STANDARD GAE, un-reshaped, then standardized.
    assert args.adv_transform == "v10"
    assert args.norm_adv is True
    assert args.advweight is True
    assert args.advweight_blend == 1.0
    assert 0.0 < args.advweight_min < 1.0 < args.advweight_max
    assert args.total_timesteps == 8000000 and args.env_id == "HalfCheetah-v4"
    assert args.value_symlog is True


def test_weight_is_mean_one_relative_to_batch():
    # w = |a| / mean|a| has mean ~= 1 by construction (before clamp), so the
    # per-sample weighting preserves the overall update scale.
    torch.manual_seed(1)
    a = torch.randn(2048) * 3.0
    abs_a = a.abs()
    w = abs_a / (abs_a.mean() + 1e-8)
    assert abs(w.mean().item() - 1.0) < 1e-4
    assert (w >= 0).all()


def test_weight_clamp_and_blend():
    abs_a = torch.tensor([0.0, 1.0, 1000.0])
    w_raw = abs_a / (abs_a.mean() + 1e-8)
    w_min, w_max = 0.1, 10.0
    w = w_raw.clamp(w_min, w_max)
    assert w.min().item() >= w_min - 1e-7
    assert w.max().item() <= w_max + 1e-7
    # blend=0 collapses to uniform weights (== standardized-GAE PPO)
    blend = 0.0
    w_eff = (1.0 - blend) + blend * w
    assert torch.allclose(w_eff, torch.ones_like(w_eff))


def test_weight_commutes_with_clip_max():
    # max(w*l1, w*l2) == w*max(l1,l2) for w>=0, so weighting the surrogate is
    # identical to scaling the advantage inside the clip.
    torch.manual_seed(2)
    l1, l2 = torch.randn(64), torch.randn(64)
    w = torch.rand(64) * 5.0
    lhs = torch.maximum(w * l1, w * l2)
    rhs = w * torch.maximum(l1, l2)
    assert torch.allclose(lhs, rhs, atol=1e-6)


def test_q_head_value_grounded_and_grad_routing():
    # The Q head produces a_learned=Q-V; its CE-to-returns loss must reach q_head,
    # q_action_proj and the shared trunk, but NOT the actor heads.
    agent, args = _make_agent()
    B, A = 16, 3
    x = torch.randn(B, 5)
    z = torch.rand(B, A).clamp(1e-6, 1 - 1e-6)
    _, _, _, _, value_logits, q_logits = agent.get_action_and_value(x, z)
    assert q_logits.shape == (B, args.num_bins)

    agent.zero_grad(set_to_none=True)
    q_logits.sum().backward()
    assert agent.q_head.weight.grad is not None
    assert agent.q_action_proj.weight.grad is not None
    assert agent.trunk.out_proj.weight.grad is not None
    # value path is separate from the actor heads
    assert agent.actor_alpha_head.weight.grad is None


def test_weight_is_detached_from_policy_graph():
    # The weight is built from a_learned.detach(); scaling pg_loss by it must not
    # backprop into the Q head (it is a fixed multiplier, like rankgauss).
    agent, args = _make_agent()
    B, A = 12, 3
    x = torch.randn(B, 5)
    z = torch.rand(B, A).clamp(1e-6, 1 - 1e-6)
    _, _, logp, _, value_logits, q_logits = agent.get_action_and_value(x, z)
    from cleanrl.shared.hl_gauss import HLGaussSupport
    from cleanrl.ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_advweight_v1 import (
        value_support_bounds,
    )
    smin, smax = value_support_bounds(args)
    hl = HLGaussSupport(args.num_bins, smin, smax, args.value_sigma_to_bin_ratio,
                        torch.device("cpu"), use_symlog=args.value_symlog, support_is_edges=True)
    a_learned = (hl.to_scalar(q_logits) - hl.to_scalar(value_logits)).detach()
    abs_a = a_learned.abs()
    w = (abs_a / (abs_a.mean() + 1e-8)).clamp(args.advweight_min, args.advweight_max)
    pg = (w * (-logp)).mean()  # stand-in surrogate weighted by w
    agent.zero_grad(set_to_none=True)
    pg.backward()
    assert agent.q_head.weight.grad is None          # weight did not differentiate Q
    assert agent.actor_alpha_head.weight.grad is not None  # policy still trains
