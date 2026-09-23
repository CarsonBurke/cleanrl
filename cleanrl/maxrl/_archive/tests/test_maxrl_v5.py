"""Contract tests for MaxRL v5 (inverse-improvability weighting)."""
import importlib.util
import math
from pathlib import Path

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "maxrl_v5", Path(__file__).resolve().parents[1] / "cleanrl" / "ppo_continuous_action_maxrl_v5.py")
v5 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v5)


def make_args(**overrides):
    args = v5.Args()
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise AttributeError(key)
        setattr(args, key, value)
    return args


class Envs:
    import gymnasium as gym
    import numpy as np
    single_action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
    single_observation_space = gym.spaces.Box(-np.inf, np.inf, (17,), np.float32)


# --- the pass rate is improvability, and it is non-negative by construction ------------

def test_pass_rate_needs_no_floor_because_the_positive_part_is_non_negative():
    """v4 needed a reward shift to make its denominator positive; (A)_+ already is."""
    torch.manual_seed(0)
    advantages = torch.randn(4096) * 7.0 - 3.0
    targets = v5.upside_targets(advantages)
    assert float(targets.min()) == 0.0
    assert torch.all(targets >= 0.0)


def test_upside_target_is_scale_and_offset_free():
    """The reward normaliser's units drift all run; the head's job must not."""
    torch.manual_seed(1)
    advantages = torch.randn(2048)
    base = v5.upside_targets(advantages)
    for transformed in (advantages * 31.0, advantages * 0.02, advantages * 5.0 - 11.0):
        torch.testing.assert_close(v5.upside_targets(transformed), base, rtol=1e-4, atol=1e-4)


def test_upside_target_is_the_positive_part_of_the_standardised_advantage():
    advantages = torch.tensor([-2.0, 0.0, 1.0, 5.0])
    reference = (advantages - advantages.mean()) / advantages.std()
    torch.testing.assert_close(v5.upside_targets(advantages), reference.clamp_min(0.0))


def test_head_output_is_non_negative_so_the_denominator_cannot_flip_sign():
    torch.manual_seed(2)
    agent = v5.Agent(Envs())
    with torch.no_grad():
        got = agent.get_upside(torch.randn(256, 17) * 12.0)
    assert torch.all(got >= 0.0) and got.shape == (256,)


# --- the weight -----------------------------------------------------------------------

def weights_for(pass_rate, lam=0.3):
    return v5.maxrl_weights(pass_rate, make_args(maxrl_lambda=lam))


def test_weights_have_unit_mean():
    torch.testing.assert_close(weights_for(torch.rand(512)).mean(), torch.tensor(1.0))


def test_weights_are_strictly_decreasing_in_the_pass_rate():
    weights = weights_for(torch.tensor([0.0, 0.1, 0.5, 2.0, 9.0]))
    assert torch.all(weights[1:] < weights[:-1])


@pytest.mark.parametrize("lam", [0.1, 0.3, 1.0, 4.0])
def test_weight_ratio_matches_the_derived_dispersion_bound(lam):
    torch.manual_seed(3)
    pass_rate = torch.rand(4096) * 3.0
    weights = weights_for(pass_rate, lam=lam)
    kappa = pass_rate.max() / pass_rate.mean()
    torch.testing.assert_close(
        weights.max() / weights.min(),
        (pass_rate.max() + lam * pass_rate.mean()) / (pass_rate.min() + lam * pass_rate.mean()),
        rtol=1e-4, atol=1e-4)
    assert weights.max() / weights.min() <= 1.0 + kappa / lam + 1e-3


def test_large_lambda_approaches_uniform_weighting():
    weights = weights_for(torch.tensor([0.0, 0.3, 1.0, 6.0]), lam=1e6)
    torch.testing.assert_close(weights, torch.ones(4), rtol=1e-4, atol=1e-4)


def test_denominator_stays_positive_for_an_all_zero_pass_rate():
    """A converged batch with no reachable improvement must not divide by zero."""
    weights = weights_for(torch.zeros(64))
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights, torch.ones(64))


def test_weight_is_stateless_across_batches():
    args = make_args(maxrl_lambda=0.3)
    normal = torch.tensor([0.1, 0.2, 0.3, 0.4])
    before = v5.maxrl_weights(normal, args)
    v5.maxrl_weights(torch.tensor([0.0, 0.0, 99.0]), args)
    torch.testing.assert_close(v5.maxrl_weights(normal, args), before)


# --- falsification controls -----------------------------------------------------------

def test_shuffle_preserves_the_weight_multiset():
    torch.manual_seed(4)
    pass_rate = torch.rand(256)
    weights = weights_for(pass_rate)
    shuffled = v5.maxrl_controls(weights, pass_rate, "shuffle", torch.Generator().manual_seed(0))
    torch.testing.assert_close(shuffled.sort().values, weights.sort().values)
    assert not torch.allclose(shuffled, weights)


def test_invert_reverses_the_pass_rate_correspondence_exactly():
    pass_rate = torch.tensor([0.4, 0.05, 1.1, 0.2])
    weights = weights_for(pass_rate)
    inverted = v5.maxrl_controls(weights, pass_rate, "invert", None)
    torch.testing.assert_close(inverted.sort().values, weights.sort().values)
    assert inverted[pass_rate.argmax()] == weights[pass_rate.argmin()]
    assert torch.all(inverted[pass_rate.argsort()][1:] > inverted[pass_rate.argsort()][:-1])


def test_off_mode_leaves_weights_untouched():
    weights = weights_for(torch.rand(32))
    torch.testing.assert_close(v5.maxrl_controls(weights, torch.rand(32), "off", None), weights)


# --- the confound v4 had to fix -------------------------------------------------------

@pytest.mark.parametrize("rho", [-0.3, 0.0, 0.25])
@pytest.mark.parametrize("mode", ["ml", "shuffle", "invert"])
def test_weighted_advantages_are_centred_for_every_arm_and_correlation(mode, rho):
    """mean(A*w) = Cov(A, w) orders the arms exactly as the hypothesis predicts unless
    it is removed; drawing A and p independently would hide it."""
    torch.manual_seed(5)
    pass_rate = torch.rand(8192) * 2.0
    standard = (pass_rate - pass_rate.mean()) / pass_rate.std()
    advantages = rho * standard + math.sqrt(1.0 - rho ** 2) * torch.randn(8192)
    weights = v5.maxrl_controls(weights_for(pass_rate, lam=0.1), pass_rate, mode,
                                torch.Generator().manual_seed(6))
    out = v5.maxrl_advantages(advantages, weights, make_args(maxrl_mode=mode, adv_norm_scope="batch"))
    assert abs(float(out.mean())) < 1e-5


# --- `off` must remain a valid control ------------------------------------------------

def test_upside_head_is_constructed_last_so_actor_and_critic_init_is_unchanged():
    """Any earlier construction would shift the initialisation RNG and make `off` a
    different run from plain PPO."""
    import importlib.util as iu
    spec = iu.spec_from_file_location(
        "base", Path(__file__).resolve().parents[1] / "cleanrl" / "ppo_continuous_action.py")
    base = iu.module_from_spec(spec)
    spec.loader.exec_module(base)
    torch.manual_seed(7)
    reference = base.Agent(Envs())
    torch.manual_seed(7)
    candidate = v5.Agent(Envs())
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(parameter, dict(candidate.named_parameters())[name])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_off_mode_gives_the_upside_head_no_gradient():
    torch.manual_seed(8)
    device = torch.device("cuda")
    agent = v5.Agent(Envs()).to(device)
    args = v5.validate_args(make_args(maxrl_mode="off", adv_norm_scope="minibatch"))
    n = 64
    observations = torch.randn(n, 17, device=device)
    native = torch.rand(n, 6, device=device).clamp(1e-3, 1 - 1e-3)
    with torch.no_grad():
        alpha, beta, _ = agent.get_policy_and_value(observations)
        old_logprobs = agent.action_logprob(alpha, beta, native)
    loss, _ = v5.ppo_loss(agent, observations, native, old_logprobs,
                          torch.randn(n, device=device), torch.randn(n, device=device),
                          torch.randn(n, device=device), torch.rand(n, device=device), args)
    loss.backward()
    assert all(p.grad is None for p in agent.upside.parameters())
    assert any(p.grad is not None for p in agent.critic.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ml_mode_trains_the_upside_head():
    torch.manual_seed(9)
    device = torch.device("cuda")
    agent = v5.Agent(Envs()).to(device)
    args = v5.validate_args(make_args(maxrl_mode="ml"))
    n = 64
    observations = torch.randn(n, 17, device=device)
    native = torch.rand(n, 6, device=device).clamp(1e-3, 1 - 1e-3)
    with torch.no_grad():
        alpha, beta, _ = agent.get_policy_and_value(observations)
        old_logprobs = agent.action_logprob(alpha, beta, native)
    loss, metrics = v5.ppo_loss(agent, observations, native, old_logprobs,
                                torch.randn(n, device=device), torch.randn(n, device=device),
                                torch.randn(n, device=device), torch.rand(n, device=device), args)
    assert torch.isfinite(loss) and torch.isfinite(metrics).all() and metrics.numel() == 7
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in agent.upside.parameters())
    assert float(metrics[6]) > 0.0


def test_upside_head_can_learn_a_state_dependent_positive_part():
    """If (A)_+ were unlearnable from the state the weight would be pure noise; this
    pins that the head has the capacity, so a null upside_ev at run time is a fact
    about the environment rather than about the architecture."""
    torch.manual_seed(10)
    agent = v5.Agent(Envs())
    optimizer = torch.optim.Adam(agent.upside.parameters(), lr=3e-3)
    observations = torch.randn(1024, 17)
    target = observations[:, 0].abs() * 0.7
    for _ in range(300):
        loss = ((agent.get_upside(observations) - target) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        explained = 1.0 - (target - agent.get_upside(observations)).var() / target.var()
    assert float(explained) > 0.9


# --- validation -----------------------------------------------------------------------

@pytest.mark.parametrize("overrides,message", [
    ({"maxrl_mode": "bogus"}, "unknown maxrl_mode"),
    ({"maxrl_lambda": 0.0}, "maxrl_lambda must be positive"),
    ({"upside_coef": -0.1}, "upside_coef must be non-negative"),
    ({"maxrl_mode": "ml", "adv_norm_scope": "minibatch"}, "cancels MaxRL weighting"),
])
def test_validate_args_rejects(overrides, message):
    with pytest.raises(ValueError, match=message):
        v5.validate_args(make_args(**overrides))


def test_defaults_are_accepted():
    args = v5.validate_args(make_args())
    assert args.maxrl_mode == "ml" and args.adv_norm_scope == "batch" and args.norm_adv is False
