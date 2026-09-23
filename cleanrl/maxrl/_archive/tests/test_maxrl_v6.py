"""Contract tests for MaxRL v6 (improvability-allocated exploration)."""
import importlib.util
from pathlib import Path

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "maxrl_v6", Path(__file__).resolve().parents[1] / "cleanrl" / "ppo_continuous_action_maxrl_v6.py")
v6 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v6)


def make_args(**overrides):
    args = v6.Args()
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


def loss_inputs(n, device, seed=0):
    torch.manual_seed(seed)
    agent = v6.Agent(Envs()).to(device)
    observations = torch.randn(n, 17, device=device)
    native = torch.rand(n, 6, device=device).clamp(1e-3, 1 - 1e-3)
    with torch.no_grad():
        alpha, beta, _ = agent.get_policy_and_value(observations)
        old_logprobs = agent.action_logprob(alpha, beta, native)
    return agent, observations, native, old_logprobs


# --- the advantages are no longer touched by the method -------------------------------

def test_advantages_are_never_reweighted():
    """The variance price of reweighting is exactly what v6 stops paying."""
    torch.manual_seed(0)
    advantages = torch.randn(4096)
    out = v6.maxrl_advantages(advantages, make_args(adv_norm_scope="batch"))
    reference = (advantages - advantages.mean()) / advantages.std()
    torch.testing.assert_close(out, reference)


def test_advantage_composition_is_identical_across_every_mode():
    torch.manual_seed(1)
    advantages = torch.randn(2048)
    outputs = [v6.maxrl_advantages(advantages, make_args(maxrl_mode=mode))
               for mode in ("off", "flat", "ml", "shuffle", "invert")]
    for other in outputs[1:]:
        torch.testing.assert_close(outputs[0], other)


def test_none_scope_passes_advantages_through():
    advantages = torch.randn(64)
    assert v6.maxrl_advantages(advantages, make_args(adv_norm_scope="none")) is advantages


# --- allocation has mean one, so only its distribution varies -------------------------

def test_entropy_weights_have_unit_mean_for_every_arm():
    """ent_coef alone fixes average exploration pressure; the arms move allocation."""
    torch.manual_seed(2)
    pass_rate = torch.rand(4096) * 2.0
    base = v6.maxrl_weights(pass_rate, make_args(maxrl_lambda=0.3))
    generator = torch.Generator().manual_seed(3)
    for mode in ("ml", "shuffle", "invert"):
        weights = v6.maxrl_controls(base, pass_rate, mode, generator)
        torch.testing.assert_close(weights.mean(), torch.tensor(1.0), rtol=1e-5, atol=1e-5)


def test_ml_widens_where_improvement_is_rare_and_invert_reverses_it():
    pass_rate = torch.tensor([0.05, 0.3, 0.9, 2.5])
    weights = v6.maxrl_weights(pass_rate, make_args(maxrl_lambda=0.3))
    assert torch.all(weights[1:] < weights[:-1])
    inverted = v6.maxrl_controls(weights, pass_rate, "invert", None)
    assert torch.all(inverted[1:] > inverted[:-1])
    torch.testing.assert_close(inverted.sort().values, weights.sort().values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_uniform_allocation_reproduces_the_plain_entropy_bonus():
    device = torch.device("cuda")
    agent, observations, native, old_logprobs = loss_inputs(96, device, seed=4)
    args = v6.validate_args(make_args(maxrl_mode="flat"))
    n = observations.shape[0]
    common = (torch.randn(n, device=device), torch.randn(n, device=device),
              torch.randn(n, device=device), torch.rand(n, device=device))
    loss_a, _ = v6.ppo_loss(agent, observations, native, old_logprobs, *common,
                            torch.ones(n, device=device), args)
    alpha, beta, _ = agent.get_policy_and_value(observations)
    from torch.distributions import Beta
    entropy = (Beta(alpha, beta, validate_args=False).entropy() + agent.log_action_scale).sum(-1)
    with torch.no_grad():
        plain = entropy.mean()
        weighted = (entropy * torch.ones(n, device=device)).mean()
    torch.testing.assert_close(plain, weighted)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_allocation_changes_the_loss_but_not_the_policy_gradient_term():
    """Two arms with the same mean allocation must differ only through entropy."""
    device = torch.device("cuda")
    agent, observations, native, old_logprobs = loss_inputs(128, device, seed=5)
    # A freshly initialised actor has std=0.01 on its output layer, so entropy is very
    # nearly state-independent and ANY mean-one allocation reproduces the flat bonus.
    # Heterogeneous states are what make the allocation observable at all.
    with torch.no_grad():
        agent.actor[-1].weight.mul_(80.0)
        alpha, beta, _ = agent.get_policy_and_value(observations)
        old_logprobs = agent.action_logprob(alpha, beta, native)
    args = v6.validate_args(make_args(maxrl_mode="flat"))
    n = observations.shape[0]
    common = (torch.randn(n, device=device), torch.randn(n, device=device),
              torch.randn(n, device=device), torch.rand(n, device=device))
    skewed = torch.rand(n, device=device) + 0.5
    skewed = skewed / skewed.mean()
    from torch.distributions import Beta
    flat_loss, flat_metrics = v6.ppo_loss(agent, observations, native, old_logprobs, *common,
                                          torch.ones(n, device=device), args)
    skew_loss, skew_metrics = v6.ppo_loss(agent, observations, native, old_logprobs, *common,
                                          skewed, args)
    torch.testing.assert_close(flat_metrics[0], skew_metrics[0])   # policy loss identical
    torch.testing.assert_close(flat_metrics[1], skew_metrics[1])   # value loss identical
    assert not torch.isclose(flat_metrics[2], skew_metrics[2])     # entropy term differs
    assert not torch.isclose(flat_loss, skew_loss)
    with torch.no_grad():
        entropy = (Beta(*agent.get_policy_and_value(observations)[:2], validate_args=False
                        ).entropy() + agent.log_action_scale).sum(-1)
    torch.testing.assert_close(skew_metrics[2], (entropy * skewed).mean(), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_off_and_flat_leave_the_upside_head_untrained():
    device = torch.device("cuda")
    n = 64
    for mode in ("off", "flat"):
        agent, observations, native, old_logprobs = loss_inputs(n, device, seed=6)
        args = v6.validate_args(make_args(maxrl_mode=mode))
        loss, _ = v6.ppo_loss(agent, observations, native, old_logprobs,
                              torch.randn(n, device=device), torch.randn(n, device=device),
                              torch.randn(n, device=device), torch.rand(n, device=device),
                              torch.ones(n, device=device), args)
        loss.backward()
        assert all(p.grad is None for p in agent.upside.parameters()), mode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ml_trains_the_upside_head_and_stays_finite():
    device = torch.device("cuda")
    n = 64
    agent, observations, native, old_logprobs = loss_inputs(n, device, seed=7)
    args = v6.validate_args(make_args(maxrl_mode="ml"))
    weights = v6.maxrl_weights(torch.rand(n, device=device), args)
    loss, metrics = v6.ppo_loss(agent, observations, native, old_logprobs,
                                torch.randn(n, device=device), torch.randn(n, device=device),
                                torch.randn(n, device=device), torch.rand(n, device=device),
                                weights, args)
    assert torch.isfinite(loss) and torch.isfinite(metrics).all()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in agent.upside.parameters())


def test_entropy_bonus_is_on_by_default_or_the_allocation_arms_are_inert():
    assert make_args().ent_coef > 0.0


@pytest.mark.parametrize("overrides,message", [
    ({"maxrl_mode": "bogus"}, "unknown maxrl_mode"),
    ({"ent_coef": -0.1}, "ent_coef must be non-negative"),
    ({"maxrl_lambda": 0.0}, "maxrl_lambda must be positive"),
])
def test_validate_args_rejects(overrides, message):
    with pytest.raises(ValueError, match=message):
        v6.validate_args(make_args(**overrides))


def test_minibatch_scope_is_now_allowed_because_nothing_reweights_advantages():
    args = v6.validate_args(make_args(maxrl_mode="ml", adv_norm_scope="minibatch"))
    assert args.norm_adv is True
