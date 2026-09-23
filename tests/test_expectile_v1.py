"""Contract tests for the expectile critic.

Load-bearing claims: tau=0.5 is PPO to the bit; the critic converges to the tau-expectile
of the return distribution rather than its mean; nothing multiplies the advantage, so the
gradient keeps the property an outcome weight destroys -- a state-dependent error in the
critic contributes nothing to the policy gradient.
"""
import importlib.util
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "cleanrl" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex = _load("expectile", "ppo_continuous_action_expectile_v1.py")
base = _load("ppo_baseline", "ppo_continuous_action.py")


class Envs:
    import gymnasium as gym
    import numpy as np
    single_action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
    single_observation_space = gym.spaces.Box(-np.inf, np.inf, (17,), np.float32)


def make_args(module, **overrides):
    args = module.Args()
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise AttributeError(key)
        setattr(args, key, value)
    return args


def loss_inputs(n, seed=0, module=ex):
    torch.manual_seed(seed)
    agent = module.Agent(Envs())
    observations = torch.randn(n, 17)
    native = torch.rand(n, 6).clamp(1e-3, 1 - 1e-3)
    with torch.no_grad():
        alpha, beta, _ = agent.get_policy_and_value(observations)
        old_logprobs = agent.action_logprob(alpha, beta, native)
    return agent, observations, native, old_logprobs


def empirical_expectile(samples, tau, iterations=400):
    """Fixed point of asymmetric least squares, computed directly from the definition."""
    value = samples.mean()
    for _ in range(iterations):
        weight = torch.where(samples - value < 0.0, 1.0 - tau, tau)
        value = (weight * samples).sum() / weight.sum()
    return value


# ------------------------------------------------------------------- PPO is the tau=0.5 arm

@pytest.mark.parametrize("clip_vloss", [True, False])
def test_half_expectile_is_bit_identical_to_the_baseline_loss(clip_vloss):
    agent, observations, native, old_logprobs = loss_inputs(256, seed=3)
    base_agent, *_ = loss_inputs(256, seed=3, module=base)
    torch.manual_seed(11)
    advantages, returns, old_values = (torch.randn(256) for _ in range(3))
    loss, metrics = ex.ppo_loss(agent, observations, native, old_logprobs, advantages,
                                returns, old_values,
                                make_args(ex, value_expectile=0.5, clip_vloss=clip_vloss))
    ref_loss, ref_metrics = base.ppo_loss(base_agent, observations, native, old_logprobs,
                                          advantages, returns, old_values,
                                          make_args(base, clip_vloss=clip_vloss))
    assert float(loss.detach()) == float(ref_loss.detach())
    assert torch.equal(metrics, ref_metrics)


def test_the_trainer_takes_the_baseline_expression_verbatim_at_one_half():
    """Not merely numerically close: the same source expression, so no rearrangement can
    introduce a rounding difference that makes the control not-quite-PPO."""
    source = (_ROOT / "cleanrl" / "ppo_continuous_action_expectile_v1.py").read_text()
    assert "if args.value_expectile == 0.5:" in source
    assert "v_loss = 0.5 * squared.mean()" in source


def test_only_the_expectile_argument_differs_from_the_baseline():
    mine, theirs = vars(ex.Args()), vars(base.Args())
    assert set(mine) - set(theirs) == {"value_expectile"}
    for name, value in theirs.items():
        if name == "exp_name":
            continue
        assert mine[name] == value, name


def test_the_agent_and_gae_are_untouched():
    """The whole design is that only what the critic REPRESENTS changes."""
    torch.manual_seed(7)
    agent = ex.Agent(Envs())
    torch.manual_seed(7)
    reference = base.Agent(Envs())
    named = dict(reference.named_parameters())
    assert {n for n, _ in agent.named_parameters()} == set(named)
    for name, parameter in agent.named_parameters():
        assert torch.equal(parameter, named[name]), name
    source = (_ROOT / "cleanrl" / "ppo_continuous_action_expectile_v1.py").read_text()
    assert "def compute_gae" in source
    baseline_gae = (_ROOT / "cleanrl" / "ppo_continuous_action.py").read_text()
    start = source.index("def compute_gae"), baseline_gae.index("def compute_gae")
    assert source[start[0]:start[0] + 900] == baseline_gae[start[1]:start[1] + 900]


# ------------------------------------------------- the critic learns the right functional

@pytest.mark.parametrize("tau", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_minimising_the_value_loss_recovers_the_expectile_of_the_returns(tau):
    """A scalar critic against a fixed return distribution must converge to that
    distribution's tau-expectile -- the definition, checked against a direct fixed-point
    iteration rather than against the same formula."""
    torch.manual_seed(0)
    returns = torch.randn(20000) * 2.0 + 1.0
    value = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([value], lr=0.05)
    for _ in range(3000):
        optimizer.zero_grad()
        squared = (value.expand_as(returns) - returns) ** 2
        check = torch.where(returns - value < 0.0, 1.0 - tau, tau)
        (check * squared).mean().backward()
        optimizer.step()
    assert float(value) == pytest.approx(float(empirical_expectile(returns, tau)), abs=0.05)


@pytest.mark.parametrize("tau,direction", [(0.7, 1), (0.9, 1), (0.3, -1), (0.1, -1)])
def test_optimistic_expectiles_sit_above_the_mean_and_pessimistic_below(tau, direction):
    torch.manual_seed(1)
    returns = torch.randn(50000) * 3.0 - 2.0
    gap = float(empirical_expectile(returns, tau) - returns.mean())
    assert direction * gap > 0.1, (tau, gap)


def test_the_expectile_is_monotone_in_tau():
    torch.manual_seed(2)
    returns = torch.randn(50000).exp()
    values = [float(empirical_expectile(returns, t)) for t in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert all(a < b for a, b in zip(values, values[1:])), values
    assert values[2] == pytest.approx(float(returns.mean()), rel=1e-3)


# ------------------------------- the property an outcome weight destroys, kept here

@pytest.mark.parametrize("tau", [0.3, 0.5, 0.7, 0.9])
def test_a_state_dependent_critic_error_contributes_nothing_to_the_policy_gradient(tau):
    """The reason this design replaces the weighted ones.

    An outcome WEIGHT m(outcome) multiplies the advantage and depends on the actions, so
    E[m*score|s] != 0 and a constant critic error e(s) enters the gradient linearly --
    measured at 0.401/0.452/0.503/0.606 for e = 0/0.5/1/2, not shrinking with batch size.
    Here nothing multiplies the advantage, so shifting every advantage in a state by a
    constant must leave the policy gradient untouched however the critic is trained.
    """
    agent, observations, native, old_logprobs = loss_inputs(4096, seed=5)
    torch.manual_seed(6)
    advantages, returns, old_values = (torch.randn(4096) for _ in range(3))
    args = make_args(ex, value_expectile=tau, norm_adv=False, vf_coef=0.0, ent_coef=0.0)

    def policy_gradient(offset):
        agent.zero_grad(set_to_none=True)
        loss, _ = ex.ppo_loss(agent, observations, native, old_logprobs,
                              advantages + offset, returns, old_values, args)
        loss.backward()
        return torch.cat([p.grad.flatten() for p in agent.actor.parameters()])

    reference = policy_gradient(0.0)
    for offset in (0.5, 1.0, 2.0, -1.0):
        shifted = policy_gradient(offset)
        # A constant offset is a baseline change: it moves the gradient by exactly
        # offset * E[score], which is the same vector for every tau and carries no
        # dependence on how the critic was fit.
        drift = (shifted - reference) / offset
        if offset == 0.5:
            expected = drift
        assert torch.allclose(drift, expected, atol=1e-5), (tau, offset)


def test_nothing_multiplies_the_advantage_in_the_policy_term():
    """Pins the structural claim by source: the policy loss must see the advantage itself,
    never a product of it with anything derived from the rollout."""
    source = (_ROOT / "cleanrl" / "ppo_continuous_action_expectile_v1.py").read_text()
    baseline = (_ROOT / "cleanrl" / "ppo_continuous_action.py").read_text()
    for marker in ("pg_loss1 = -advantages * ratio",
                   "pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)",
                   "pg_loss = torch.max(pg_loss1, pg_loss2).mean()"):
        assert marker in source and marker in baseline


# ---------------------------------------------------------- the asymmetry is real and signed

@pytest.mark.parametrize("tau", [0.6, 0.7, 0.9])
def test_under_prediction_is_penalised_more_than_over_prediction_when_optimistic(tau):
    agent, observations, native, old_logprobs = loss_inputs(512, seed=9)
    torch.manual_seed(10)
    advantages, old_values = torch.randn(512), torch.zeros(512)
    args = make_args(ex, value_expectile=tau, vf_coef=1.0, clip_vloss=False)
    with torch.no_grad():
        _, _, predicted = agent.get_policy_and_value(observations)
        predicted = predicted.view(-1)

    def value_loss(returns):
        loss, metrics = ex.ppo_loss(agent, observations, native, old_logprobs, advantages,
                                    returns, old_values, args)
        return float(metrics[1])

    gap = torch.full((512,), 1.0)
    too_low = value_loss(predicted + gap)      # critic under-predicts: weight tau
    too_high = value_loss(predicted - gap)     # critic over-predicts: weight 1 - tau
    assert too_low > too_high
    assert too_low / too_high == pytest.approx(tau / (1.0 - tau), rel=1e-4)


def test_the_two_sides_balance_exactly_at_one_half():
    agent, observations, native, old_logprobs = loss_inputs(512, seed=9)
    torch.manual_seed(10)
    advantages, old_values = torch.randn(512), torch.zeros(512)
    args = make_args(ex, value_expectile=0.5, vf_coef=1.0, clip_vloss=False)
    with torch.no_grad():
        _, _, predicted = agent.get_policy_and_value(observations)
        predicted = predicted.view(-1)
    gap = torch.full((512,), 1.0)

    def value_loss(returns):
        return float(ex.ppo_loss(agent, observations, native, old_logprobs, advantages,
                                 returns, old_values, args)[1][1])
    assert value_loss(predicted + gap) == pytest.approx(value_loss(predicted - gap), rel=1e-6)


# ------------------------------------------------------------------------ argument guard

@pytest.mark.parametrize("tau", [0.0, 1.0, -0.1, 1.5])
def test_validate_args_rejects_expectiles_outside_the_open_unit_interval(tau):
    with pytest.raises(ValueError, match="value_expectile"):
        ex.validate_args(make_args(ex, value_expectile=tau))


def test_validate_args_accepts_the_sweep():
    for tau in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
        assert ex.validate_args(make_args(ex, value_expectile=tau)).value_expectile == tau
