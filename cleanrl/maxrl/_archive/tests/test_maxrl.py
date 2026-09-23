"""Contracts for the MaxRL maximum-likelihood reweighting in ppo_continuous_action_maxrl_v1.

The transplant rests on three claims that are cheap to pin down here:
- w_T is the order-T truncation of the ML weight, bounded in [1, T], and finite at
  the probability clamps where the naive (1-(1-p)^T)/p form is 0/0.
- the advantage composition applies the paper's ordering (centre, then weight) and
  leaves the gradient scale alone, so T and beta are not disguised learning rates.
- --maxrl-mode off is an exact control: every shared parameter must draw the same
  RNG stream as the unmodified baseline, or an ablation against it means nothing.
"""

import importlib

import numpy as np
import pytest
import torch

maxrl = importlib.import_module("cleanrl.ppo_continuous_action_maxrl_v1")
baseline = importlib.import_module("cleanrl.ppo_continuous_action")

ORDERS = (1, 2, 4, 8, 16, 64)


def make_args(**overrides):
    args = maxrl.Args()
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize("order", ORDERS)
def test_weight_matches_the_truncated_maclaurin_series(order):
    """w_T(p) must equal sum_{k=1..T} (1-p)^(k-1), the paper's Proposition 6."""
    probability = torch.linspace(0.02, 0.98, 97, dtype=torch.float64)
    series = sum((1.0 - probability) ** (k - 1) for k in range(1, order + 1))
    torch.testing.assert_close(maxrl.maxrl_weight(probability, order), series)


@pytest.mark.parametrize("order", ORDERS)
def test_weight_is_bounded_and_decreasing(order):
    """w_T(1)=1 and w_T(0)=T bracket the reallocation; monotonicity gives it its sign."""
    probability = torch.linspace(maxrl.PASS_FLOOR, maxrl.PASS_CEIL, 4096, dtype=torch.float64)
    weight = maxrl.maxrl_weight(probability, order)
    assert torch.isfinite(weight).all()
    assert weight.min() >= 1.0 - 1e-9 and weight.max() <= order + 1e-6
    assert (weight.diff() <= 1e-12).all()
    torch.testing.assert_close(maxrl.maxrl_weight(torch.ones(1, dtype=torch.float64), order),
                               torch.ones(1, dtype=torch.float64))
    # w_T(p) -> T only as p -> 0; at the clamp it is T(1 - (T-1)p/2) + O(p^2).
    at_floor = maxrl.maxrl_weight(torch.full((1,), maxrl.PASS_FLOOR, dtype=torch.float64), order)
    assert at_floor.item() == pytest.approx(order, rel=order * maxrl.PASS_FLOOR)
    tiny = torch.full((1,), 1e-12, dtype=torch.float64)
    assert maxrl.maxrl_weight(tiny, order).item() == pytest.approx(order, rel=1e-9)


def test_weight_recovers_plain_ppo_at_order_one():
    """T=1 is REINFORCE: the paper's family must contain the method it generalises."""
    probability = torch.linspace(maxrl.PASS_FLOOR, maxrl.PASS_CEIL, 1000, dtype=torch.float64)
    torch.testing.assert_close(maxrl.maxrl_weight(probability, 1), torch.ones_like(probability))


def test_weight_mode_reallocates_without_rescaling():
    """Mean-1 weighting moves gradient between states but not the overall scale."""
    generator = torch.Generator().manual_seed(0)
    advantages = torch.randn(4096, generator=generator)
    returns = torch.randn(4096, generator=generator)
    logits = torch.randn(4096, generator=generator) * 2.0
    args = make_args(maxrl_mode="weight", maxrl_order=8)
    weighted, success, _ = maxrl.maxrl_advantages(advantages, returns, logits,
                                                  torch.tensor(0.0), args)

    probability = torch.sigmoid(logits).clamp(maxrl.PASS_FLOOR, maxrl.PASS_CEIL)
    weight = maxrl.maxrl_weight(probability, args.maxrl_order)
    torch.testing.assert_close(weighted, advantages * weight / weight.mean())
    # Reallocation, not rescaling: the per-sample weights average to one.
    assert (weight / weight.mean()).mean().item() == pytest.approx(1.0, rel=1e-6)
    # Hard states (low phat) must receive strictly more weight than easy ones.
    assert weight[probability.argmin()] > weight[probability.argmax()]
    torch.testing.assert_close(success, (returns > 0.0).float())


def test_tail_mode_is_algorithm_one_with_the_truncated_weight():
    """The added term is w_T(phat) * (success - phat): centred first, then weighted."""
    generator = torch.Generator().manual_seed(1)
    advantages = torch.randn(4096, generator=generator)
    returns = torch.randn(4096, generator=generator)
    logits = torch.randn(4096, generator=generator)
    bar = torch.tensor(0.3)
    args = make_args(maxrl_mode="tail", maxrl_order=8, maxrl_beta=0.5)
    composed, success, _ = maxrl.maxrl_advantages(advantages, returns, logits, bar, args)

    probability = torch.sigmoid(logits).clamp(maxrl.PASS_FLOOR, maxrl.PASS_CEIL)
    weight = maxrl.maxrl_weight(probability, args.maxrl_order)
    tail = weight * (success - probability)
    tail = tail * (advantages.std() / (tail.std() + 1e-8))
    expected = (advantages + args.maxrl_beta * tail) / np.sqrt(1.0 + args.maxrl_beta ** 2)
    torch.testing.assert_close(composed, expected)
    # Asymmetry is the point: a rare success outweighs a routine failure.
    assert tail.max() > tail.min().abs()


def test_beta_does_not_smuggle_in_a_learning_rate_change():
    """Gradient scale must stay put across beta, or the ablation is confounded."""
    generator = torch.Generator().manual_seed(2)
    advantages = torch.randn(65536, generator=generator)
    returns = torch.randn(65536, generator=generator)
    logits = torch.randn(65536, generator=generator)
    scales = []
    for beta in (0.0, 0.5, 1.0, 2.0):
        args = make_args(maxrl_mode="tail", maxrl_beta=beta)
        composed, _, _ = maxrl.maxrl_advantages(advantages, returns, logits,
                                                torch.tensor(0.0), args)
        scales.append(composed.std().item())
    assert max(scales) / min(scales) < 1.15


def test_success_event_tracks_the_bar():
    """The success indicator is exactly 1{return target > bar}."""
    returns = torch.tensor([-2.0, 0.0, 0.5, 1.0, 3.0])
    args = make_args(maxrl_mode="weight")
    _, success, _ = maxrl.maxrl_advantages(torch.ones(5), returns, torch.zeros(5),
                                           torch.tensor(0.5), args)
    torch.testing.assert_close(success, torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))


def test_diagnostics_report_the_spread_that_makes_or_breaks_the_method():
    """phat_std near zero means w_T is flat and MaxRL has degenerated into PPO."""
    logits = torch.full((1024,), 0.7)
    args = make_args(maxrl_mode="weight")
    _, _, diagnostics = maxrl.maxrl_advantages(torch.ones(1024), torch.ones(1024), logits,
                                               torch.tensor(0.0), args)
    bar, mean, std, below, above, weight_mean, weight_max, rate, calibration = diagnostics
    assert std.item() == pytest.approx(0.0, abs=1e-6)
    assert mean.item() == pytest.approx(torch.sigmoid(torch.tensor(0.7)).item(), rel=1e-5)
    assert weight_mean.item() == pytest.approx(weight_max.item(), rel=1e-6)
    assert below.item() == 0.0 and above.item() == 0.0
    assert rate.item() == 1.0 and bar.item() == 0.0
    assert calibration.item() == pytest.approx(1.0 - mean.item(), rel=1e-5)


def test_off_mode_agent_is_the_untouched_baseline():
    """Shared parameters must be bit-identical, so mode=off is a true control."""
    class Space:
        def __init__(self, shape, low, high):
            self.shape, self.low, self.high = shape, low, high

    class Envs:
        single_observation_space = Space((17,), None, None)
        single_action_space = None

    import gymnasium as gym
    Envs.single_action_space = gym.spaces.Box(low=-np.ones(6, dtype=np.float32),
                                              high=np.ones(6, dtype=np.float32))
    Envs.single_observation_space = gym.spaces.Box(low=-np.ones(17, dtype=np.float32) * np.inf,
                                                   high=np.ones(17, dtype=np.float32) * np.inf)
    torch.manual_seed(7)
    reference = baseline.Agent(Envs())
    torch.manual_seed(7)
    candidate = maxrl.Agent(Envs())

    reference_critic = list(reference.critic.parameters())
    candidate_critic = list(candidate.critic_trunk.parameters()) + list(candidate.value_head.parameters())
    assert len(reference_critic) == len(candidate_critic)
    for expected, actual in zip(reference_critic, candidate_critic):
        torch.testing.assert_close(actual, expected)
    for expected, actual in zip(reference.actor.parameters(), candidate.actor.parameters()):
        torch.testing.assert_close(actual, expected)


def test_detached_pass_head_cannot_reshape_the_value_function():
    """With the default detach, BCE gradients must not reach the critic trunk."""
    import gymnasium as gym

    class Envs:
        single_action_space = gym.spaces.Box(low=-np.ones(6, dtype=np.float32),
                                             high=np.ones(6, dtype=np.float32))
        single_observation_space = gym.spaces.Box(low=-np.ones(17, dtype=np.float32) * np.inf,
                                                  high=np.ones(17, dtype=np.float32) * np.inf)

    torch.manual_seed(3)
    agent = maxrl.Agent(Envs())
    observations = torch.randn(64, 17)
    _, _, _, logits = agent.get_policy_value_pass(observations, True)
    logits.square().mean().backward()
    assert all(parameter.grad is None for parameter in agent.critic_trunk.parameters())
    assert agent.pass_head.weight.grad is not None

    agent.zero_grad(set_to_none=True)
    _, _, _, logits = agent.get_policy_value_pass(observations, False)
    logits.square().mean().backward()
    assert any(parameter.grad is not None for parameter in agent.critic_trunk.parameters())


@pytest.mark.parametrize("bad", [
    {"maxrl_order": 0}, {"maxrl_quantile": 1.0}, {"maxrl_quantile": 0.0},
    {"maxrl_tau_ema": 1.0}, {"maxrl_beta": -0.1}, {"maxrl_pass_coef": -1.0},
])
def test_invalid_maxrl_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        maxrl.validate_args(make_args(**bad))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the trainer is CUDA-only")
@pytest.mark.parametrize("mode", ["off", "weight", "tail", "both"])
def test_compiled_loss_path_runs_on_cuda(mode):
    """The real compiled loss must accept the extra success input in every mode.

    Shape or graph errors here would otherwise only surface once a queued 8M-step run
    reaches its first update, long after submission.
    """
    import gymnasium as gym

    class Envs:
        single_action_space = gym.spaces.Box(low=-np.ones(6, dtype=np.float32),
                                             high=np.ones(6, dtype=np.float32))
        single_observation_space = gym.spaces.Box(low=-np.ones(17, dtype=np.float32) * np.inf,
                                                  high=np.ones(17, dtype=np.float32) * np.inf)

    device = torch.device("cuda")
    torch.manual_seed(11)
    agent = maxrl.Agent(Envs()).to(device)
    args = make_args(maxrl_mode=mode)
    batch = 512

    observations = torch.randn(batch, 17, device=device)
    native = torch.rand(batch, 6, device=device).clamp(maxrl.SAMPLE_EPS, 1 - maxrl.SAMPLE_EPS)
    with torch.no_grad():
        alpha, beta, values, pass_logits = agent.get_policy_value_pass(observations, True)
        old_logprobs = agent.action_logprob(alpha, beta, native)
        values = values.flatten()
    advantages = torch.randn(batch, device=device)
    returns = advantages + values
    success = torch.zeros_like(returns)
    if mode != "off":
        bar = torch.quantile(returns, args.maxrl_quantile)
        advantages, success, diagnostics = maxrl.maxrl_advantages(
            advantages, returns, pass_logits, bar, args
        )
        assert torch.isfinite(diagnostics).all()

    compiled = torch.compile(
        lambda *inputs: maxrl.ppo_loss(agent, *inputs, args),
        mode=args.compile_mode, fullgraph=True, dynamic=False,
    )
    torch.compiler.cudagraph_mark_step_begin()
    loss, metrics = compiled(observations, native, old_logprobs, advantages, returns,
                             values, success)
    assert torch.isfinite(loss) and metrics.shape == (7,)
    loss.backward()
    assert torch.isfinite(agent.actor[0].weight.grad).all()
    # mode=off must leave the pass head untouched, making it an exact PPO control.
    assert (agent.pass_head.weight.grad is None) == (mode == "off")
