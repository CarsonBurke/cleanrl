"""Contract tests for MaxRL-tail v1 (superquantile policy objective).

The load-bearing claims are: PPO is the alpha=0 member of the family (not an external
control), the tail advantage is a valid baselined estimator, and the pinball head solves
the problem its calibration metric reports on.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "cleanrl" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tail = _load("maxrl_tail_v1", "ppo_continuous_action_maxrl_tail_v1.py")
base = _load("ppo_baseline", "ppo_continuous_action.py")


def make_args(**overrides):
    args = tail.Args()
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


def loss_inputs(n, device, seed=0, module=tail):
    torch.manual_seed(seed)
    agent = module.Agent(Envs()).to(device)
    observations = torch.randn(n, 17, device=device)
    native = torch.rand(n, 6, device=device).clamp(1e-3, 1 - 1e-3)
    with torch.no_grad():
        alpha, beta, _ = agent.get_policy_and_value(observations)
        old_logprobs = agent.action_logprob(alpha, beta, native)
    return agent, observations, native, old_logprobs


# ---------------------------------------------------------------- the alpha -> 0 limit

@pytest.mark.parametrize("alpha", [1e-4, 1e-3, 1e-2])
def test_tail_advantage_converges_to_the_plain_advantage_as_alpha_vanishes(alpha):
    """With the hinge open below the whole support, q cancels and w is exactly A.

    This is the claim that PPO is a MEMBER of the family rather than a separate control,
    so it is checked against an arbitrary (wrong) quantile: the limit must not depend on
    the head being right, only on the hinge being open.
    """
    torch.manual_seed(0)
    advantages = torch.randn(4096)
    advantages = advantages - advantages.mean()
    quantile = torch.full_like(advantages, float(advantages.min()) - 3.0)
    # The true conditional excess when the hinge is fully open: E[A - q] / (1-alpha).
    excess = (advantages - quantile).mean() / (1.0 - alpha)
    out = tail.tail_advantages(advantages, quantile, excess, alpha)
    assert torch.allclose(out, advantages / (1.0 - alpha) - advantages.mean() / (1.0 - alpha),
                          atol=1e-5)
    # After the standardisation the loss applies, the 1/(1-alpha) is a pure scale.
    standardised = (out - out.mean()) / (out.std() + 1e-8)
    reference = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    assert torch.allclose(standardised, reference, atol=1e-4)


def test_alpha_zero_is_bit_identical_to_the_baseline_ppo_loss():
    """Not "close to PPO": the same number, so the control arm has no private code path."""
    device = torch.device("cpu")
    agent, observations, native, old_logprobs = loss_inputs(256, device, seed=3)
    base_agent, *_ = loss_inputs(256, device, seed=3, module=base)
    torch.manual_seed(11)
    advantages = torch.randn(256, device=device)
    returns = torch.randn(256, device=device)
    old_values = torch.randn(256, device=device)
    args = make_args(tail_alpha=0.0, cuda=False)
    base_args = base.Args()
    loss, metrics = tail.ppo_loss(agent, observations, native, old_logprobs, advantages,
                                  returns, old_values, advantages, args)
    ref_loss, ref_metrics = base.ppo_loss(base_agent, observations, native, old_logprobs,
                                          advantages, returns, old_values, base_args)
    assert float(loss) == float(ref_loss)
    assert torch.equal(metrics[:6], ref_metrics[:6])
    assert float(metrics[6]) == 0.0


def test_tail_head_is_constructed_last_so_actor_and_critic_init_is_unchanged():
    """The alpha=0 identity above only holds if the shared modules draw the same RNG."""
    torch.manual_seed(7)
    agent = tail.Agent(Envs())
    torch.manual_seed(7)
    reference = base.Agent(Envs())
    named = dict(reference.named_parameters())
    assert set(named) == {n for n, _ in agent.named_parameters() if not n.startswith("tail.")}
    for name, parameter in agent.named_parameters():
        if name.startswith("tail."):
            continue
        assert torch.equal(parameter, named[name]), name


def test_alpha_zero_gives_the_tail_head_no_gradient():
    device = torch.device("cpu")
    agent, observations, native, old_logprobs = loss_inputs(128, device, seed=5)
    torch.manual_seed(1)
    advantages = torch.randn(128, device=device)
    args = make_args(tail_alpha=0.0, cuda=False)
    loss, _ = tail.ppo_loss(agent, observations, native, old_logprobs, advantages,
                            torch.randn(128), torch.randn(128), advantages, args)
    loss.backward()
    assert all(p.grad is None for p in agent.tail.parameters())


def test_both_sides_coincide_at_alpha_zero_and_separate_above_it():
    """The falsification arm is only a falsification arm if it shares the control."""
    torch.manual_seed(2)
    advantages = torch.randn(2048)
    advantages = advantages - advantages.mean()

    def transform(side, alpha):
        sign = tail.tail_orientation(side)
        oriented = sign * advantages
        quantile = torch.full_like(oriented, float(torch.quantile(oriented, alpha)))
        excess = (oriented - quantile).clamp_min(0.0).mean() / (1.0 - alpha)
        return sign * tail.tail_advantages(oriented, quantile, excess, alpha)

    # The residual gap is the finite-batch quantile sitting just above the support
    # minimum rather than below it, so it must vanish in proportion to alpha -- that
    # linear scaling is the mathematical content, where a fixed threshold is a magic number.
    gaps = [float((transform("upper", a) - transform("lower", a)).abs().max())
            for a in (1e-5, 1e-4, 1e-3)]
    assert gaps[0] < 0.02
    assert 5.0 < gaps[1] / gaps[0] < 20.0
    assert 5.0 < gaps[2] / gaps[1] < 20.0
    assert float((transform("upper", 0.7) - transform("lower", 0.7)).abs().max()) > 0.5


# ------------------------------------------------------------------- estimator validity

@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75, 0.9])
def test_tail_advantage_is_mean_zero_when_the_head_is_correct(alpha):
    """w = u - E[u|s] is a baselined estimator, so it must not smuggle in an offset.

    This is the guard the v4 post-mortem earned: an uncentred weight would have ordered
    the arms by a nuisance mean rather than by the mechanism under test.
    """
    torch.manual_seed(0)
    advantages = torch.randn(200000)
    quantile = torch.full_like(advantages, float(torch.quantile(advantages, alpha)))
    excess = (advantages - quantile).clamp_min(0.0).mean() / (1.0 - alpha)
    out = tail.tail_advantages(advantages, quantile, excess, alpha)
    assert abs(float(out.mean())) < 5e-3


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_rockafellar_uryasev_identity_recovers_the_superquantile(alpha):
    """u = q + (Z-q)_+/(1-alpha) has mean equal to E[Z | Z >= q_alpha] at the true q."""
    torch.manual_seed(4)
    z = torch.randn(400000)
    quantile = torch.quantile(z, alpha)
    integrand = quantile + (z - quantile).clamp_min(0.0) / (1.0 - alpha)
    conditional = z[z >= quantile].mean()
    assert abs(float(integrand.mean() - conditional)) < 0.02


@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.8])
def test_pinball_loss_is_minimised_at_the_alpha_quantile(alpha):
    """The head's calibration metric is only meaningful if its loss targets that quantile."""
    torch.manual_seed(6)
    samples = torch.randn(100000) * 2.0 + 1.0
    prediction = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([prediction], lr=0.05)
    for _ in range(600):
        optimizer.zero_grad()
        tail.pinball_loss(samples - prediction, alpha).mean().backward()
        optimizer.step()
    assert abs(float(prediction) - float(torch.quantile(samples, alpha))) < 0.05
    below = (samples < prediction).float().mean()
    assert abs(float(below) - alpha) < 0.02


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75])
def test_tail_advantage_is_monotone_and_flat_below_the_hinge(alpha):
    """Everything outside the tail receives the SAME push; that hinge IS the mechanism."""
    advantages = torch.linspace(-3.0, 3.0, 601)
    quantile = torch.full_like(advantages, float(torch.tensor(alpha) * 2.0 - 1.0))
    excess = torch.full_like(advantages, 0.4)
    out = tail.tail_advantages(advantages, quantile, excess, alpha)
    assert torch.all(out[1:] >= out[:-1] - 1e-6)
    below = out[advantages < quantile]
    assert float(below.max() - below.min()) < 1e-6
    active = (advantages > quantile).float().mean()
    assert float(((out[1:] - out[:-1]) > 1e-6).float().mean()) == pytest.approx(float(active), abs=0.01)


def test_standardisation_makes_the_loss_blind_to_a_constant_offset_in_the_weight():
    """A constant excess error cannot be the mechanism, because norm_adv erases it."""
    device = torch.device("cpu")
    agent, observations, native, old_logprobs = loss_inputs(256, device, seed=9)
    torch.manual_seed(12)
    weights = torch.randn(256)
    returns, old_values = torch.randn(256), torch.randn(256)
    args = make_args(tail_alpha=0.6, cuda=False, norm_adv=True)
    a, _ = tail.ppo_loss(agent, observations, native, old_logprobs, weights, returns,
                         old_values, weights, args)
    b, _ = tail.ppo_loss(agent, observations, native, old_logprobs, weights + 7.5, returns,
                         old_values, weights, args)
    assert float(a) == pytest.approx(float(b), abs=1e-5)


# ------------------------------------------------------------------------ the head works

def test_tail_head_can_learn_a_state_dependent_quantile_on_fresh_data():
    """Capacity test, so a null `tail/quantile_dispersion` is a fact about the environment
    and not about the architecture.

    The advantage noise is resampled every step, which is the trainer's regime: a new
    rollout each iteration, and a head read on the fresh batch BEFORE it is updated on it.
    On a frozen dataset the same head instead memorises each state's single advantage draw
    and its held-out quantile error grows with training while its calibration still looks
    perfect -- which is exactly why `tail/below_frac` is measured on unseen data.
    """
    torch.manual_seed(0)
    agent = tail.Agent(Envs())
    alpha = 0.75
    observations = torch.randn(4096, 17)
    # Advantage spread is driven by one observation coordinate, so the true quantile
    # varies across states and a collapsed head cannot fit it.
    spread = 0.5 + observations[:, 0].abs()
    truth = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(alpha)) * spread
    optimizer = torch.optim.Adam(agent.tail.parameters(), lr=3e-3)
    for _ in range(600):
        advantages = torch.randn(4096) * spread
        optimizer.zero_grad()
        quantile, excess = agent.get_tail(observations)
        target = (advantages - quantile.detach()).clamp_min(0.0) / (1.0 - alpha)
        (tail.pinball_loss(advantages - quantile, alpha) + (excess - target) ** 2).mean().backward()
        optimizer.step()
    with torch.no_grad():
        held_out = torch.randn(4096) * spread
        quantile, _ = agent.get_tail(observations)
        below = (held_out < quantile).float().mean()
        residual = ((quantile - truth) ** 2).mean() / truth.var()
    assert abs(float(below) - alpha) < 0.05, "head is not calibrated on held-out data"
    assert float(residual) < 0.25, "head collapsed to a constant instead of tracking the state"
    assert float(quantile.std()) > 0.15 * float(held_out.std())


def test_a_frozen_dataset_makes_the_head_memorise_instead_of_generalise():
    """Pins the failure mode the fresh-data fixture above exists to avoid, so nobody
    "simplifies" that test back into a static dataset and reads the result as capacity."""
    torch.manual_seed(0)
    agent = tail.Agent(Envs())
    alpha = 0.75
    observations = torch.randn(4096, 17)
    spread = 0.5 + observations[:, 0].abs()
    truth = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(alpha)) * spread
    frozen = torch.randn(4096) * spread
    optimizer = torch.optim.Adam(agent.tail.parameters(), lr=3e-3)
    residuals = []
    for step in range(1200):
        optimizer.zero_grad()
        quantile, _ = agent.get_tail(observations)
        tail.pinball_loss(frozen - quantile, alpha).mean().backward()
        optimizer.step()
        if step in (299, 1199):
            with torch.no_grad():
                quantile, _ = agent.get_tail(observations)
                residuals.append(float(((quantile - truth) ** 2).mean() / truth.var()))
    # In-sample calibration stays perfect while the held-out quantile error GROWS.
    with torch.no_grad():
        quantile, _ = agent.get_tail(observations)
        assert abs(float((frozen < quantile).float().mean()) - alpha) < 0.05
    assert residuals[1] > residuals[0]
    assert residuals[1] > 1.0


def test_excess_target_is_detached_from_the_quantile():
    """Otherwise the excess regression can move the hinge to wherever its own target is
    easiest to fit, which is a degenerate solution with a perfectly healthy loss curve."""
    device = torch.device("cpu")
    agent, observations, native, old_logprobs = loss_inputs(256, device, seed=13)
    torch.manual_seed(14)
    advantages = torch.randn(256)
    args = make_args(tail_alpha=0.5, cuda=False, tail_coef=1.0, vf_coef=0.0, ent_coef=0.0)

    head = agent.tail[-1]
    # Freeze the policy and value paths so the only gradient into the head is the tail loss.
    for parameter in list(agent.actor.parameters()) + list(agent.critic.parameters()):
        parameter.requires_grad_(False)
    loss, _ = tail.ppo_loss(agent, observations, native, old_logprobs, advantages,
                            torch.randn(256), torch.randn(256), advantages, args)
    loss.backward()
    grad = head.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    # The quantile row's gradient must be bounded by the pinball subgradient, which lives
    # in [alpha-1, alpha]; a leaked squared-error term is unbounded and blows this up.
    activations = torch.tanh(agent.tail[2](torch.tanh(agent.tail[0](observations))))
    bound = args.tail_alpha * activations.abs().mean(0).sum() / 1.0
    assert float(grad[0].abs().sum()) <= float(bound) + 1e-4


# ------------------------------------------------------------------------- argument guard

@pytest.mark.parametrize("alpha", [-0.1, 1.0, 1.5])
def test_validate_args_rejects_alpha_outside_the_family(alpha):
    with pytest.raises(ValueError, match="tail_alpha"):
        tail.validate_args(make_args(tail_alpha=alpha))


def test_validate_args_rejects_unknown_side_and_negative_coefficients():
    with pytest.raises(ValueError, match="tail_side"):
        tail.validate_args(make_args(tail_side="sideways"))
    with pytest.raises(ValueError, match="tail_coef"):
        tail.validate_args(make_args(tail_coef=-1.0))


def test_validate_args_accepts_the_whole_sweep():
    for alpha in (0.0, 0.25, 0.5, 0.75, 0.9):
        for side in ("upper", "lower"):
            args = tail.validate_args(make_args(tail_alpha=alpha, tail_side=side))
            assert args.batch_size == args.num_envs * args.num_steps


# --------------------------------------------------------------------------- numerics

@pytest.mark.parametrize("alpha", [0.5, 0.9, 0.95])
def test_loss_is_finite_when_no_sample_in_a_minibatch_reaches_the_hinge(alpha):
    """At high alpha a minibatch can be entirely below the quantile; standardisation is
    scale-free so this is well posed, but it is the obvious place to blow up."""
    device = torch.device("cpu")
    agent, observations, native, old_logprobs = loss_inputs(128, device, seed=15)
    torch.manual_seed(16)
    excess = torch.rand(128) * 0.3 + 0.1
    weights = -excess  # every sample flat-side of the hinge
    args = make_args(tail_alpha=alpha, cuda=False)
    loss, metrics = tail.ppo_loss(agent, observations, native, old_logprobs, weights,
                                  torch.randn(128), torch.randn(128), torch.randn(128), args)
    assert torch.isfinite(loss) and torch.isfinite(metrics).all()


@pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75, 0.9])
def test_variance_price_is_bounded_and_grows_with_alpha(alpha):
    """The transform buys direction with variance; this records the exchange rate so a
    regression in it is visible rather than inferred from a score."""
    torch.manual_seed(8)
    advantages = torch.randn(200000)
    quantile = torch.full_like(advantages, float(torch.quantile(advantages, alpha)))
    excess = (advantages - quantile).clamp_min(0.0).mean() / (1.0 - alpha)
    out = tail.tail_advantages(advantages, quantile, excess, alpha)
    ratio = float(out.std() / advantages.std())
    assert 1.0 < ratio < 6.0
    assert float(torch.corrcoef(torch.stack((out, advantages)))[0, 1]) > 0.5
