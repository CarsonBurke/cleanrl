"""Contract tests for MaxRL v4 (inverse-value weighting of the policy gradient)."""
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from cleanrl.shared.ppo_loop import compute_gae

_SPEC = importlib.util.spec_from_file_location(
    "maxrl_v4", Path(__file__).resolve().parents[1] / "cleanrl" / "ppo_continuous_action_maxrl_v4.py")
v4 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v4)


def make_args(**overrides):
    args = v4.Args()
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise AttributeError(key)
        setattr(args, key, value)
    return args


def weights_for(values, lam=0.3):
    return v4.maxrl_weights(values, make_args(maxrl_lambda=lam))


# --- the objective: A/V is exactly Appendix M.4's advantage with mu = V ---------------

def test_maxrl_advantage_matches_appendix_m4_form():
    """(G - V)/V, the weighted advantage, is M.4's (r - mu)/mu with mu the critic."""
    returns = torch.tensor([3.0, 5.0, 11.0, 1.0])
    value = returns.mean()
    m4 = (returns - value) / value
    ours = (returns - value) * (1.0 / value)
    torch.testing.assert_close(ours, m4)


def test_log_expected_return_gradient_is_advantage_over_value():
    """d/dp log E[R] equals E[A grad log pi]/V on a two-outcome discrete model."""
    logit = torch.tensor(0.4, requires_grad=True)
    rewards = torch.tensor([2.0, 7.0])
    probability = torch.sigmoid(logit)
    mixture = torch.stack((1.0 - probability, probability))
    objective = (mixture * rewards).sum().log()
    objective.backward()
    exact = logit.grad.clone()

    with torch.no_grad():
        probability = torch.sigmoid(torch.tensor(0.4))
        mixture = torch.stack((1.0 - probability, probability))
        value = (mixture * rewards).sum()
        # score of a Bernoulli logit for outcomes 0 and 1
        scores = torch.stack((-probability, 1.0 - probability))
        reconstructed = (mixture * (rewards - value) / value * scores).sum()
    torch.testing.assert_close(exact, reconstructed, rtol=1e-5, atol=1e-6)


# --- the reward shift that makes log E[R] well defined leaves GAE untouched ------------

def _gae(rewards, values, terminations, tail, gamma=0.99, lam=0.95):
    truncations = torch.zeros_like(terminations)
    return compute_gae(rewards, values, terminations, truncations,
                       torch.zeros_like(values), tail, gamma, lam)[0]


def test_constant_reward_shift_leaves_advantages_exactly_unchanged():
    gamma = 0.99
    torch.manual_seed(0)
    rewards = torch.randn(64, 4)
    values = torch.randn(64, 4) * 5.0
    tail = torch.randn(4) * 5.0
    zeros = torch.zeros(64, 4)
    base = _gae(rewards, values, zeros, tail, gamma)

    shift = 1.75
    offset = shift / (1.0 - gamma)
    moved = _gae(rewards + shift, values + offset, zeros, tail + offset, gamma)
    torch.testing.assert_close(base, moved, rtol=0, atol=2e-4)


def test_reward_shift_invariance_needs_the_absence_of_termination():
    """Honest scope: with terminations the shift becomes a survival bonus."""
    gamma = 0.99
    torch.manual_seed(1)
    rewards, values = torch.randn(32, 2), torch.randn(32, 2)
    tail = torch.randn(2)
    terms = torch.zeros(32, 2)
    terms[10, 0] = 1.0
    shift, offset = 2.0, 2.0 / (1.0 - gamma)
    base = _gae(rewards, values, terms, tail, gamma)
    moved = _gae(rewards + shift, values + offset, terms, tail + offset, gamma)
    assert not torch.allclose(base, moved, atol=1e-2)


# --- the weight function --------------------------------------------------------------

def test_shifted_values_are_non_negative_without_clamping():
    values = torch.tensor([-40.0, -3.0, 0.0, 12.0])
    _, floor = weights_for(values)
    assert torch.all(values - floor >= 0.0)


def test_weights_have_unit_mean():
    values = torch.tensor([-8.0, 1.0, 4.0, 30.0, 100.0])
    weights, _ = weights_for(values)
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))


def test_weights_are_strictly_decreasing_in_value():
    values = torch.tensor([-5.0, 0.0, 3.0, 9.0, 50.0])
    weights, _ = weights_for(values)
    assert torch.all(weights[1:] < weights[:-1])


@pytest.mark.parametrize("lam", [0.1, 0.3, 1.0, 4.0])
def test_weight_ratio_matches_the_derived_dispersion_bound(lam):
    """max/min == 1 + kappa/lambda with kappa = max(shifted)/mean(shifted).

    The ratio is NOT a function of lambda alone: the batch's own value dispersion
    kappa enters, which is why the run logs the realised ratio.
    """
    torch.manual_seed(2)
    values = torch.randn(4096) * 37.0
    weights, floor = weights_for(values, lam=lam)
    shifted = values - floor
    kappa = shifted.max() / shifted.mean()
    torch.testing.assert_close(weights.max() / weights.min(), 1.0 + kappa / lam,
                               rtol=1e-4, atol=1e-4)


def test_weight_ratio_shrinks_monotonically_as_lambda_grows():
    torch.manual_seed(2)
    values = torch.randn(4096) * 37.0
    ratios = [float(weights_for(values, lam=lam)[0].max() / weights_for(values, lam=lam)[0].min())
              for lam in (0.1, 0.3, 1.0, 4.0)]
    assert ratios == sorted(ratios, reverse=True)
    assert ratios[0] > 10.0 and ratios[-1] < 2.0


def test_denominator_is_positive_for_every_state_without_clipping():
    torch.manual_seed(9)
    values = torch.randn(4096) * 37.0
    args = make_args(maxrl_lambda=0.1)
    weights, floor = weights_for(values, lam=0.1)
    shifted = values - floor
    denominator = shifted + args.maxrl_lambda * shifted.mean()
    assert float(denominator.min()) > 0.0
    assert torch.isfinite(weights).all()


def test_large_lambda_approaches_uniform_weighting():
    values = torch.tensor([-10.0, 0.0, 5.0, 60.0])
    weights, _ = weights_for(values, lam=1e6)
    torch.testing.assert_close(weights, torch.ones(4), rtol=1e-4, atol=1e-4)


def test_small_lambda_approaches_inverse_value_weighting():
    values = torch.tensor([0.0, 1.0, 3.0, 8.0])
    weights, floor = weights_for(values, lam=1e-6)
    exact = 1.0 / (values - floor).clamp_min(1e-12)
    finite = exact[1:] / exact[1:].mean()
    torch.testing.assert_close(weights[1:] / weights[1:].mean(), finite, rtol=1e-3, atol=1e-3)


def test_degenerate_equal_values_give_uniform_weights():
    values = torch.full((16,), 7.5)
    weights, _ = weights_for(values)
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights, torch.ones(16))


def test_the_floor_is_stateless_so_no_history_can_shrink_the_dose():
    """A prior revision smoothed the floor with an EMA; one outlier batch then halved
    the dose for ~100 iterations (a fifth of a run). With a batch-minimum floor, an
    outlier affects only its own iteration."""
    args = make_args(maxrl_lambda=0.3)
    normal = torch.tensor([1.0, 2.0, 3.0, 4.0])
    before, _ = v4.maxrl_weights(normal, args)
    v4.maxrl_weights(torch.tensor([-500.0, 1.0, 2.0, 3.0, 4.0]), args)
    after, _ = v4.maxrl_weights(normal, args)
    torch.testing.assert_close(before, after)


def test_shifted_minimum_is_exactly_zero_so_the_ratio_identity_holds():
    torch.manual_seed(11)
    for _ in range(5):
        values = torch.randn(2048) * 13.0 + 400.0
        weights, floor = weights_for(values, lam=0.3)
        shifted = values - floor
        assert float(shifted.min()) == 0.0
        torch.testing.assert_close(weights.max() / weights.min(),
                                   1.0 + shifted.max() / shifted.mean() / 0.3,
                                   rtol=1e-4, atol=1e-4)


def test_weights_are_invariant_to_the_reward_normaliser_scale_and_offset():
    """VectorRewardNorm rescales by a running return std, so value units drift all run;
    a stateless floor makes the weights exactly invariant to that."""
    values = torch.tensor([-3.0, 2.0, 9.0, 40.0])
    base, _ = weights_for(values)
    for transformed in (values * 17.0, values * 0.05, values + 900.0, values * 3.0 - 50.0):
        got, _ = weights_for(transformed)
        torch.testing.assert_close(base, got, rtol=1e-4, atol=1e-4)


# --- falsification controls -----------------------------------------------------------

def test_shuffle_preserves_the_weight_multiset():
    generator = torch.Generator().manual_seed(0)
    values = torch.randn(256) * 10.0
    weights, _ = weights_for(values)
    shuffled = v4.maxrl_controls(weights, values, "shuffle", generator)
    torch.testing.assert_close(shuffled.sort().values, weights.sort().values)
    assert not torch.allclose(shuffled, weights)


def test_invert_reverses_the_value_correspondence_exactly():
    values = torch.tensor([4.0, -2.0, 11.0, 0.5])
    weights, _ = weights_for(values)
    inverted = v4.maxrl_controls(weights, values, "invert", None)
    torch.testing.assert_close(inverted.sort().values, weights.sort().values)
    # highest value now carries the weight the lowest value had
    assert inverted[values.argmax()] == weights[values.argmin()]
    assert inverted[values.argmin()] == weights[values.argmax()]
    assert torch.all(inverted[values.argsort()][1:] > inverted[values.argsort()][:-1])


def test_off_mode_leaves_weights_untouched():
    values = torch.randn(32)
    weights, _ = weights_for(values)
    torch.testing.assert_close(v4.maxrl_controls(weights, values, "off", None), weights)


def test_controls_and_treatment_share_a_mean_of_one():
    values = torch.randn(128) * 6.0
    weights, _ = weights_for(values)
    generator = torch.Generator().manual_seed(3)
    for mode in ("ml", "shuffle", "invert"):
        got = v4.maxrl_controls(weights, values, mode, generator)
        torch.testing.assert_close(got.mean(), torch.tensor(1.0), rtol=1e-5, atol=1e-5)


# --- advantage composition ------------------------------------------------------------

def test_off_with_minibatch_scope_is_the_identity():
    advantages = torch.randn(64)
    args = make_args(maxrl_mode="off", adv_norm_scope="minibatch")
    out = v4.maxrl_advantages(advantages, torch.ones(64), args)
    assert out is advantages


def test_off_with_batch_scope_is_plain_standardisation():
    advantages = torch.randn(512) * 3.0 + 1.0
    args = make_args(maxrl_mode="off", adv_norm_scope="batch")
    out = v4.maxrl_advantages(advantages, torch.ones(512), args)
    torch.testing.assert_close(out.mean(), torch.tensor(0.0), atol=1e-5, rtol=0)
    torch.testing.assert_close(out.std(), torch.tensor(1.0), atol=1e-4, rtol=0)


def test_rescale_restores_the_batch_advantage_scale():
    torch.manual_seed(4)
    advantages = torch.randn(4096)
    values = torch.randn(4096) * 20.0
    weights, _ = weights_for(values, lam=0.1)
    args = make_args(maxrl_mode="ml", adv_norm_scope="batch", maxrl_rescale=True)
    out = v4.maxrl_advantages(advantages, weights, args)
    torch.testing.assert_close(out.std(), torch.tensor(1.0), rtol=1e-3, atol=1e-3)


def test_weighting_actually_changes_the_advantages():
    torch.manual_seed(5)
    advantages = torch.randn(1024)
    values = torch.randn(1024) * 15.0
    weights, _ = weights_for(values, lam=0.1)
    args = make_args(maxrl_mode="ml", adv_norm_scope="batch")
    out = v4.maxrl_advantages(advantages, weights, args)
    reference = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    assert not torch.allclose(out, reference, atol=1e-3)


@pytest.mark.parametrize("rho", [-0.3, -0.1, 0.0, 0.2])
@pytest.mark.parametrize("mode", ["ml", "shuffle", "invert"])
def test_weighted_advantages_are_centred_for_every_arm_and_correlation(mode, rho):
    """Regression for the confound that broke the falsification design.

    mean(A*w) = Cov(A, w): positive for `ml`, ~0 for `shuffle`, negative for `invert`
    whenever A and V correlate -- the exact ordering the hypothesis predicts, so an
    uncentred offset would masquerade as the effect. Drawing A and V INDEPENDENTLY (as
    an earlier version of this test did) forces Cov to zero and hides it entirely.
    """
    torch.manual_seed(6)
    values = torch.randn(8192) * 25.0
    standard = (values - values.mean()) / values.std()
    advantages = rho * standard + math.sqrt(1.0 - rho ** 2) * torch.randn(8192)
    weights, _ = weights_for(values, lam=0.1)
    weights = v4.maxrl_controls(weights, values, mode, torch.Generator().manual_seed(12))
    args = make_args(maxrl_mode=mode, adv_norm_scope="batch")
    out = v4.maxrl_advantages(advantages, weights, args)
    assert abs(float(out.mean())) < 1e-5


def test_uncentred_weighting_would_have_ordered_the_arms_by_the_nuisance_offset():
    """Shows the defect the re-centring removes, so the guard cannot be quietly dropped."""
    torch.manual_seed(6)
    values = torch.randn(8192) * 25.0
    standard = (values - values.mean()) / values.std()
    advantages = -0.3 * standard + math.sqrt(1.0 - 0.09) * torch.randn(8192)
    reference = (advantages - advantages.mean()) / advantages.std()
    base, _ = weights_for(values, lam=0.1)
    generator = torch.Generator().manual_seed(12)
    offsets = {mode: float((reference * v4.maxrl_controls(base, values, mode, generator)).mean())
               for mode in ("ml", "shuffle", "invert")}
    assert offsets["ml"] > offsets["shuffle"] > offsets["invert"]
    assert offsets["ml"] - offsets["invert"] > 0.01


def test_participation_ratio_is_one_for_uniform_weights_and_lower_otherwise():
    def participation(weights):
        return float(weights.sum() ** 2 / (weights.numel() * (weights ** 2).sum()))
    assert participation(torch.ones(100)) == pytest.approx(1.0)
    torch.manual_seed(7)
    values = torch.randn(4096) * 30.0
    assert participation(weights_for(values, lam=0.1)[0]) < 0.95
    assert participation(weights_for(values, lam=4.0)[0]) > participation(weights_for(values, lam=0.1)[0])


# --- argument validation --------------------------------------------------------------

@pytest.mark.parametrize("overrides,message", [
    ({"maxrl_mode": "bogus"}, "unknown maxrl_mode"),
    ({"adv_norm_scope": "bogus"}, "unknown adv_norm_scope"),
    ({"maxrl_lambda": 0.0}, "maxrl_lambda must be positive"),
    ({"maxrl_lambda": -1.0}, "maxrl_lambda must be positive"),
    ({"maxrl_mode": "ml", "adv_norm_scope": "minibatch"}, "cancels MaxRL weighting"),
    ({"maxrl_mode": "invert", "adv_norm_scope": "minibatch"}, "cancels MaxRL weighting"),
])
def test_validate_args_rejects(overrides, message):
    with pytest.raises(ValueError, match=message):
        v4.validate_args(make_args(**overrides))


@pytest.mark.parametrize("scope,expected", [("minibatch", True), ("batch", False), ("none", False)])
def test_norm_adv_is_derived_from_the_scope(scope, expected):
    args = v4.validate_args(make_args(maxrl_mode="off", adv_norm_scope=scope))
    assert args.norm_adv is expected


def test_default_configuration_is_accepted_and_disables_minibatch_norm():
    args = v4.validate_args(make_args())
    assert args.maxrl_mode == "ml" and args.adv_norm_scope == "batch" and args.norm_adv is False


# --- end to end -----------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ppo_loss_consumes_weighted_advantages_on_cuda():
    import gymnasium as gym
    import numpy as np

    class Envs:
        single_action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
        single_observation_space = gym.spaces.Box(-np.inf, np.inf, (17,), np.float32)

    torch.manual_seed(8)
    device = torch.device("cuda")
    agent = v4.Agent(Envs()).to(device)
    args = v4.validate_args(make_args(maxrl_mode="ml", adv_norm_scope="batch"))
    n = 128
    observations = torch.randn(n, 17, device=device)
    native = torch.rand(n, 6, device=device).clamp(1e-3, 1 - 1e-3)
    values = torch.randn(n, device=device) * 10.0
    weights, _ = v4.maxrl_weights(values, args)
    advantages = v4.maxrl_advantages(torch.randn(n, device=device), weights, args)
    with torch.no_grad():
        alpha, beta, _ = agent.get_policy_and_value(observations)
        old_logprobs = agent.action_logprob(alpha, beta, native)
    loss, metrics = v4.ppo_loss(agent, observations, native, old_logprobs, advantages,
                                torch.randn(n, device=device), values, args)
    assert torch.isfinite(loss) and torch.isfinite(metrics).all()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in agent.parameters() if p.grad is not None)
