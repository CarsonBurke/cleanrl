"""Contracts for the critic-free MaxRL transplant in ppo_continuous_action_maxrl_pure_v1.

The value of this variant is that it is the AUTHORS' estimator, so the tests check it
against the reference implementation in the MaxRL repo rather than against my reading of
the paper. maclaurin_weights is compared element-for-element with their maclaurin.py, and
the closed-form estimators against the formulas in their core_algos.py.
"""

import importlib
import importlib.util
import pathlib

import numpy as np
import pytest
import torch

pure = importlib.import_module("cleanrl.ppo_continuous_action_maxrl_pure_v1")

REFERENCE = pathlib.Path(__file__).resolve().parents[2] / "maxrl" / "verl" / "trainer" / "ppo" / "maclaurin.py"


def load_reference():
    if not REFERENCE.exists():
        pytest.skip(f"reference MaxRL checkout not present at {REFERENCE}")
    spec = importlib.util.spec_from_file_location("reference_maclaurin", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_args(**overrides):
    args = pure.Args()
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize("group_size", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("order", [1, 2, 4, 8, 64])
def test_maclaurin_weights_match_the_reference_implementation(group_size, order):
    """Our running-ratio form must reproduce their Kahan-summed weights exactly."""
    reference = load_reference()
    successes = torch.arange(0, group_size + 1, dtype=torch.float64)
    ours = pure.maclaurin_weights(successes, group_size, order)
    theirs = reference.maclaurin_weights(successes, group_size, order)
    for actual, expected in zip(ours, theirs):
        torch.testing.assert_close(actual.to(torch.float64), expected.to(torch.float64),
                                   rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("group_size", [4, 16])
def test_order_one_maclaurin_is_reinforce_on_the_success_indicator(group_size):
    """T=1 must drop every failure term, leaving the pass@1 gradient."""
    successes = torch.arange(0, group_size + 1, dtype=torch.float64)
    w_succ, w_fail = pure.maclaurin_weights(successes, group_size, 1)
    torch.testing.assert_close(w_succ, torch.full_like(w_succ, 1.0 / group_size))
    torch.testing.assert_close(w_fail, torch.zeros_like(w_fail))


def test_maclaurin_failure_weight_grows_with_the_truncation_order():
    """Higher T pulls in more failure terms, so |w_fail| must increase monotonically."""
    successes = torch.tensor([2.0])
    magnitudes = [pure.maclaurin_weights(successes, 16, order)[1].abs().item()
                  for order in (1, 2, 4, 8, 16)]
    assert magnitudes == sorted(magnitudes)
    assert magnitudes[0] == 0.0 and magnitudes[-1] > 0.0


def test_maxrl_estimator_is_algorithm_one():
    """A_i = (r_i - mean)/(mean + eps): centred by the group mean, divided by it, no std."""
    returns = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    # A pass rate away from 1/2, where a Bernoulli's mean and std coincide and the two
    # estimators would agree for reasons that have nothing to do with the objective.
    bar = torch.tensor(5.5)
    advantages, _ = pure.outcome_advantages(returns, bar, make_args(maxrl_estimator="maxrl"))
    rewards = (returns > bar).float()
    assert rewards.mean().item() == pytest.approx(0.25)
    expected = (rewards - rewards.mean()) / (rewards.mean() + pure.GROUP_EPS)
    torch.testing.assert_close(advantages, expected)
    # The distinguishing property against GRPO: division by the mean, not the deviation.
    grpo, _ = pure.outcome_advantages(returns, bar, make_args(maxrl_estimator="grpo"))
    assert not torch.allclose(advantages, grpo)
    # Both recover the raw centred reward once their own denominator is undone.
    torch.testing.assert_close(advantages * rewards.mean(), grpo * rewards.std(unbiased=False))


def test_maxrl_upweights_the_group_the_harder_its_prompt_is():
    """1/mean is the whole mechanism: it must amplify hard prompts FASTER than RLOO does.

    RLOO already gives a success the advantage 1-p, which grows as a prompt gets harder.
    MaxRL gives (1-p)/p, so the extra 1/p is what separates the two: the hard-to-easy
    ratio picks up exactly the pass-rate ratio that the paper's w(p)=1/p predicts.
    """
    bar = torch.tensor(0.5)
    easy = torch.tensor([1.0] * 7 + [0.0])      # pass rate 7/8
    hard = torch.tensor([1.0] + [0.0] * 7)      # pass rate 1/8

    def peak(estimator, group):
        return pure.outcome_advantages(group, bar, make_args(maxrl_estimator=estimator))[0].max().item()

    maxrl_ratio = peak("maxrl", hard) / peak("maxrl", easy)
    rloo_ratio = peak("rloo", hard) / peak("rloo", easy)
    assert maxrl_ratio > rloo_ratio
    # The excess is the pass-rate ratio itself: w(p) = 1/p on top of RLOO's w(p) = 1.
    assert maxrl_ratio / rloo_ratio == pytest.approx((7 / 8) / (1 / 8), rel=1e-4)


def test_dense_estimator_shifts_returns_into_the_positive_orthant():
    """Appendix M.4 needs r >= 0 with E[r] > 0; MuJoCo returns are signed."""
    returns = torch.tensor([-300.0, -50.0, 10.0, 900.0])
    advantages, _ = pure.outcome_advantages(returns, torch.tensor(0.0),
                                            make_args(maxrl_estimator="dense_maxrl"))
    shifted = returns - returns.min() + 1.0
    torch.testing.assert_close(advantages, (shifted - shifted.mean()) / (shifted.mean() + pure.GROUP_EPS))
    assert torch.isfinite(advantages).all()
    assert advantages.argmax().item() == returns.argmax().item()


@pytest.mark.parametrize("estimator", pure.ESTIMATORS)
def test_every_estimator_is_finite_at_the_degenerate_pass_rates(estimator):
    """All-fail and all-pass groups must not produce nan, even where 1/mean blows up."""
    args = make_args(maxrl_estimator=estimator)
    for returns in (torch.zeros(8), torch.ones(8), torch.full((8,), -5.0)):
        advantages, diagnostics = pure.outcome_advantages(returns, torch.tensor(0.5), args)
        assert torch.isfinite(advantages).all(), estimator
        assert torch.isfinite(diagnostics).all(), estimator


def test_group_quantile_bar_keeps_the_pass_rate_off_the_degenerate_ends():
    """With one group per iteration, C=0 would zero the whole gradient; the quantile bar
    is what prevents that."""
    generator = torch.Generator().manual_seed(0)
    args = make_args(maxrl_estimator="maxrl", maxrl_quantile=0.75)
    for _ in range(20):
        returns = torch.randn(16, generator=generator) * 100.0 + 500.0
        bar = torch.quantile(returns, args.maxrl_quantile)
        advantages, diagnostics = pure.outcome_advantages(returns, bar, args)
        successes = diagnostics[2].item()
        assert 0 < successes < 16
        assert advantages.abs().sum() > 0


def test_segment_must_be_a_whole_episode():
    """A partial segment would silently break the one-response-per-prompt correspondence."""
    with pytest.raises(ValueError, match="horizon"):
        pure.validate_args(make_args(num_steps=1024))
    with pytest.raises(ValueError, match="group"):
        pure.validate_args(make_args(num_steps=1000, num_envs=1))
    assert pure.validate_args(make_args(num_steps=1000, num_envs=16)).batch_size == 16000


def test_agent_has_no_critic():
    """Critic-free is the defining property of this variant, so pin it."""
    import gymnasium as gym

    class Envs:
        single_action_space = gym.spaces.Box(low=-np.ones(6, dtype=np.float32),
                                             high=np.ones(6, dtype=np.float32))
        single_observation_space = gym.spaces.Box(low=-np.ones(17, dtype=np.float32) * np.inf,
                                                  high=np.ones(17, dtype=np.float32) * np.inf)

    agent = pure.Agent(Envs())
    names = [name for name, _ in agent.named_parameters()]
    assert all(name.startswith("actor.") for name in names), names
    assert not hasattr(agent, "critic") and not hasattr(agent, "value_head")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the trainer is CUDA-only")
def test_compiled_loss_path_runs_on_cuda():
    """Catch shape and graph errors before an 8M-step run reaches its first update."""
    import gymnasium as gym

    class Envs:
        single_action_space = gym.spaces.Box(low=-np.ones(6, dtype=np.float32),
                                             high=np.ones(6, dtype=np.float32))
        single_observation_space = gym.spaces.Box(low=-np.ones(17, dtype=np.float32) * np.inf,
                                                  high=np.ones(17, dtype=np.float32) * np.inf)

    device = torch.device("cuda")
    torch.manual_seed(5)
    agent = pure.Agent(Envs()).to(device)
    args = make_args()
    observations = torch.randn(512, 17, device=device)
    native = torch.rand(512, 6, device=device).clamp(pure.SAMPLE_EPS, 1 - pure.SAMPLE_EPS)
    with torch.no_grad():
        alpha, beta = agent.policy(observations)
        old_logprobs = agent.action_logprob(alpha, beta, native)
    advantages = torch.randn(512, device=device)

    compiled = torch.compile(lambda *inputs: pure.ppo_loss(agent, *inputs, args),
                             mode=args.compile_mode, fullgraph=True, dynamic=False)
    torch.compiler.cudagraph_mark_step_begin()
    loss, metrics = compiled(observations, native, old_logprobs, advantages)
    assert torch.isfinite(loss) and metrics.shape == (5,)
    loss.backward()
    assert torch.isfinite(agent.actor[0].weight.grad).all()
