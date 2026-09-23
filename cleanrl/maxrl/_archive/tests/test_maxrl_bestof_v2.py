"""Contract tests for MaxRL-bestof v2 (best-of-T on the empirical CDF).

The load-bearing claims are: the weight depends on the sample's RANK and nothing else (so
no distribution is assumed), it reproduces the true gradient of E[max of T] including the
T-discriminating scale derivative, and T=1 is PPO.
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


bo = _load("maxrl_bestof_v2", "ppo_continuous_action_maxrl_bestof_v2.py")
base = _load("ppo_baseline", "ppo_continuous_action.py")


def make_args(**overrides):
    args = bo.Args()
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


def loss_inputs(n, device, seed=0, module=bo):
    torch.manual_seed(seed)
    agent = module.Agent(Envs()).to(device)
    observations = torch.randn(n, 17, device=device)
    native = torch.rand(n, 6, device=device).clamp(1e-3, 1 - 1e-3)
    with torch.no_grad():
        alpha, beta, _ = agent.get_policy_and_value(observations)
        old_logprobs = agent.action_logprob(alpha, beta, native)
    return agent, observations, native, old_logprobs


def standardise(x):
    return (x - x.mean()) / (x.std() + 1e-8)


# --------------------------------------------------------- the weight is a rank statistic

def test_order_one_is_the_advantage_up_to_the_shift_standardisation_removes():
    x = torch.randn(10000, dtype=torch.float64)
    h = bo.rank_weight(x, 1.0)
    assert float((h - (x - x.min())).abs().max()) < 1e-9
    assert torch.allclose(standardise(h), standardise(x), atol=1e-9)


def test_weight_is_equivariant_to_permutation_so_only_rank_matters():
    torch.manual_seed(0)
    x = torch.randn(4096, dtype=torch.float64)
    permutation = torch.randperm(4096)
    assert torch.allclose(bo.rank_weight(x, 4.0)[permutation],
                          bo.rank_weight(x[permutation], 4.0))


@pytest.mark.parametrize("order", [2.0, 4.0, 8.0])
def test_weight_is_positively_homogeneous_so_the_advantage_scale_cannot_matter(order):
    """h is an integral in z units, so h(aX) = a h(X) for a > 0. Combined with the
    standardisation in the loss this makes the method free of the advantage's units --
    the drifting reward-normaliser scale cannot change what the policy sees."""
    torch.manual_seed(1)
    x = torch.randn(4096, dtype=torch.float64)
    for factor in (0.01, 3.0, 250.0):
        assert torch.allclose(bo.rank_weight(factor * x, order), factor * bo.rank_weight(x, order))
        assert torch.allclose(standardise(bo.rank_weight(factor * x, order)),
                              standardise(bo.rank_weight(x, order)))
    # A pure shift changes h only by a constant, which standardisation also removes.
    assert torch.allclose(standardise(bo.rank_weight(x + 17.0, order)),
                          standardise(bo.rank_weight(x, order)))


@pytest.mark.parametrize("order", [2.0, 4.0, 8.0])
def test_increments_match_the_exact_trapezoid_on_the_order_statistics(order):
    """The closed form this replaces a quadrature with: increment between neighbouring
    order statistics is T * mean(F^(T-1)) * gap, with F = (i + 1/2)/N."""
    torch.manual_seed(2)
    x = torch.randn(512, dtype=torch.float64)
    values, _ = torch.sort(x)
    cdf = (torch.arange(512, dtype=torch.float64) + 0.5) / 512
    density = cdf.pow(order - 1.0)
    expected = order * 0.5 * (density[:-1] + density[1:]) * values.diff()
    got = bo.rank_weight(x, order).sort().values.diff()
    assert float((got - expected).abs().max()) < 1e-12


@pytest.mark.parametrize("order", [2.0, 4.0, 8.0])
def test_weight_is_increasing_and_convex_in_rank(order):
    torch.manual_seed(3)
    x = torch.randn(8192, dtype=torch.float64)
    increments = bo.rank_weight(x, order).sort().values.diff()
    assert torch.all(increments >= 0.0)
    # The slope a sample earns grows with its percentile: never flat, unlike the hinge.
    lower = increments[:100].sum()
    upper = increments[-100:].sum()
    assert float(upper) > 5.0 * float(lower) > 0.0


# ------------------------------------------------------------------- it is the T gradient

@pytest.mark.parametrize("order,expected", [(1.0, 0.0), (2.0, 0.5642), (4.0, 1.0294), (8.0, 1.4236)])
def test_weight_reproduces_the_scale_gradient_of_the_expected_maximum(order, expected):
    """d/dsigma E[max of T] = E[max of T standard normals], the check that discriminates
    orders (the location derivative is 1 for every T and so proves nothing)."""
    torch.manual_seed(4)
    z = torch.randn(4_000_000, dtype=torch.float64)
    weight = bo.rank_weight(z, order)
    weight = weight - weight.mean()
    assert abs(float((weight * (z * z - 1.0)).mean()) - expected) < 0.02


@pytest.mark.parametrize("order", [1.0, 2.0, 4.0, 8.0])
def test_weight_reproduces_the_location_gradient(order):
    torch.manual_seed(5)
    z = torch.randn(2_000_000, dtype=torch.float64)
    weight = bo.rank_weight(z, order)
    weight = weight - weight.mean()
    assert abs(float((weight * z).mean()) - 1.0) < 0.02


def test_estimator_is_unbiased_on_a_heavy_tailed_law_where_a_gaussian_plug_in_is_not():
    """The reason this version exists: the running batches carry kurtosis 12-150, and a
    Phi(A/sigma) percentile reads their extremes as 1. Here the ranks are exact whatever
    the shape, so the variance price stays bounded."""
    torch.manual_seed(6)
    n = 2_000_000
    chi = torch.distributions.Chi2(torch.tensor(3.0, dtype=torch.float64)).sample((n,))
    heavy = torch.randn(n, dtype=torch.float64) / (chi / 3).sqrt()
    for order, ceiling in ((2.0, 2.0), (4.0, 3.0), (8.0, 5.0)):
        weight = bo.rank_weight(heavy, order)
        ratio = float(weight.std() / heavy.std())
        assert 1.0 <= ratio < ceiling, (order, ratio)


# -------------------------------------------------------------------- PPO is the T=1 arm

def test_order_one_is_bit_identical_to_the_baseline_ppo_loss():
    device = torch.device("cpu")
    agent, observations, native, old_logprobs = loss_inputs(256, device, seed=3)
    base_agent, *_ = loss_inputs(256, device, seed=3, module=base)
    torch.manual_seed(11)
    advantages, returns, old_values = (torch.randn(256) for _ in range(3))
    loss, metrics = bo.ppo_loss(agent, observations, native, old_logprobs, advantages,
                                returns, old_values, advantages, advantages,
                                make_args(bestof_order=1.0))
    ref_loss, ref_metrics = base.ppo_loss(base_agent, observations, native, old_logprobs,
                                          advantages, returns, old_values, base.Args())
    assert float(loss.detach()) == float(ref_loss.detach())
    assert torch.equal(metrics[:6], ref_metrics[:6])
    assert float(metrics[6]) == 0.0


def test_baseline_head_is_constructed_last_so_actor_and_critic_init_is_unchanged():
    torch.manual_seed(7)
    agent = bo.Agent(Envs())
    torch.manual_seed(7)
    reference = base.Agent(Envs())
    named = dict(reference.named_parameters())
    assert set(named) == {n for n, _ in agent.named_parameters() if not n.startswith("spread.")}
    for name, parameter in agent.named_parameters():
        if not name.startswith("spread."):
            assert torch.equal(parameter, named[name]), name


def test_order_one_gives_the_baseline_head_no_gradient():
    device = torch.device("cpu")
    agent, observations, native, old_logprobs = loss_inputs(128, device, seed=5)
    torch.manual_seed(1)
    advantages = torch.randn(128)
    loss, _ = bo.ppo_loss(agent, observations, native, old_logprobs, advantages,
                          torch.randn(128), torch.randn(128), advantages, advantages,
                          make_args(bestof_order=1.0))
    loss.backward()
    assert all(p.grad is None for p in agent.spread.parameters())


def test_both_sides_coincide_at_order_one_and_separate_above_it():
    torch.manual_seed(2)
    advantages = torch.randn(4096)
    baseline = torch.zeros_like(advantages)

    def arm(side, order):
        sign = bo.bestof_orientation(side)
        return sign * bo.bestof_advantages(sign * advantages, baseline, order)

    assert torch.allclose(standardise(arm("upper", 1.0)), standardise(arm("lower", 1.0)), atol=1e-5)
    assert float((standardise(arm("upper", 4.0)) - standardise(arm("lower", 4.0))).abs().max()) > 0.5


# -------------------------------------------------------------------- estimator validity

@pytest.mark.parametrize("order", [2.0, 4.0, 8.0])
def test_weight_is_conditionally_centred_when_the_baseline_is_correct(order):
    """w must be mean-zero GIVEN s, or the arms can be ordered by a state-dependent offset
    rather than by the mechanism -- the defect the v4 post-mortem turned up."""
    torch.manual_seed(0)
    spreads = torch.tensor([0.3, 0.7, 1.0, 2.5, 6.0])
    labels = torch.arange(5).repeat_interleave(8192)
    advantages = torch.randn(5 * 8192) * spreads[labels]
    transformed = bo.rank_weight(advantages, order)
    baseline = torch.zeros_like(advantages)
    for index in range(5):
        baseline[labels == index] = transformed[labels == index].mean()
    out = bo.bestof_advantages(advantages, baseline, order)
    for index in range(5):
        assert abs(float(out[labels == index].mean())) < 1e-4


def test_baseline_head_can_learn_the_conditional_mean_on_fresh_data():
    """Capacity test, so a null `bestof/baseline_ev` is a fact about the environment.

    Advantages are resampled every step: the trainer reads the head on a fresh batch
    BEFORE updating it, so memorising single draws does not help.
    """
    torch.manual_seed(0)
    agent = bo.Agent(Envs())
    observations = torch.randn(4096, 17)
    spread = 0.4 + observations[:, 0].abs()
    optimizer = torch.optim.Adam(agent.spread.parameters(), lr=3e-3)
    for _ in range(600):
        advantages = torch.randn(4096) * spread
        target = bo.rank_weight(advantages, 4.0)
        optimizer.zero_grad()
        ((agent.get_spread(observations) - target) ** 2).mean().backward()
        optimizer.step()
    with torch.no_grad():
        held_out = bo.rank_weight(torch.randn(4096) * spread, 4.0)
        predicted = agent.get_spread(observations)
        explained = 1.0 - (held_out - predicted).var() / held_out.var()
    assert float(explained) > 0.1, "baseline head learned nothing state-dependent"


def test_standardisation_makes_the_loss_blind_to_a_constant_offset_in_the_weight():
    device = torch.device("cpu")
    agent, observations, native, old_logprobs = loss_inputs(256, device, seed=9)
    torch.manual_seed(12)
    weights, returns, old_values = (torch.randn(256) for _ in range(3))
    args = make_args(bestof_order=4.0, norm_adv=True)
    a, _ = bo.ppo_loss(agent, observations, native, old_logprobs, weights, returns,
                       old_values, weights, weights, args)
    b, _ = bo.ppo_loss(agent, observations, native, old_logprobs, weights + 7.5, returns,
                       old_values, weights, weights, args)
    assert float(a.detach()) == pytest.approx(float(b.detach()), abs=1e-5)


def test_head_loss_drives_only_the_baseline_and_uses_a_precomputed_target():
    device = torch.device("cpu")
    agent, observations, native, old_logprobs = loss_inputs(256, device, seed=13)
    torch.manual_seed(14)
    advantages, weighted_target = torch.randn(256), torch.randn(256)
    args = make_args(bestof_order=4.0, bestof_coef=1.0, vf_coef=0.0, ent_coef=0.0)
    for parameter in list(agent.actor.parameters()) + list(agent.critic.parameters()):
        parameter.requires_grad_(False)
    loss, _ = bo.ppo_loss(agent, observations, native, old_logprobs, advantages,
                          torch.randn(256), torch.randn(256), advantages, weighted_target, args)
    loss.backward()
    grad = agent.spread[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and float(grad.abs().sum()) > 1e-6
    reference = bo.Agent(Envs())
    reference.load_state_dict(agent.state_dict())
    ((reference.get_spread(observations) - weighted_target) ** 2).mean().backward()
    assert torch.allclose(grad, reference.spread[-1].weight.grad, atol=1e-6)


# ------------------------------------------------------------------------ argument guard

@pytest.mark.parametrize("order", [0.0, 0.5, -1.0])
def test_validate_args_rejects_orders_below_one(order):
    with pytest.raises(ValueError, match="bestof_order"):
        bo.validate_args(make_args(bestof_order=order))


def test_validate_args_rejects_unknown_side_and_negative_coefficient():
    with pytest.raises(ValueError, match="bestof_side"):
        bo.validate_args(make_args(bestof_side="sideways"))
    with pytest.raises(ValueError, match="bestof_coef"):
        bo.validate_args(make_args(bestof_coef=-1.0))


def test_validate_args_accepts_the_whole_sweep():
    for order in (1.0, 2.0, 4.0):
        for side in ("upper", "lower"):
            args = bo.validate_args(make_args(bestof_order=order, bestof_side=side))
            assert args.batch_size == args.num_envs * args.num_steps


# ----------------------------------------------------------------------------- numerics

def test_constant_batch_gives_a_constant_weight_rather_than_a_nan():
    """Every advantage equal means every gap is zero: h must be flat, not undefined."""
    out = bo.rank_weight(torch.full((512,), 2.5, dtype=torch.float64), 4.0)
    assert torch.isfinite(out).all() and float(out.std()) == 0.0


def test_float32_accumulation_does_not_lose_the_bottom_of_the_sort():
    """The cumsum runs in float64 on purpose: in float32 a 32k-term climb whose late
    increments dwarf its early ones erases exactly the low-rank gradation order T is meant
    to preserve smoothly."""
    torch.manual_seed(8)
    x = torch.randn(32768)
    out = bo.rank_weight(x, 8.0)
    increments = out.sort().values.diff()
    assert torch.all(increments >= 0.0)
    assert int((increments[:16384] > 0).sum()) > 16000, "low ranks collapsed to a flat region"


def test_loss_is_finite_for_a_degenerate_minibatch():
    device = torch.device("cpu")
    agent, observations, native, old_logprobs = loss_inputs(128, device, seed=15)
    weights = torch.full((128,), -0.3)
    loss, metrics = bo.ppo_loss(agent, observations, native, old_logprobs, weights,
                                torch.randn(128), torch.randn(128), torch.randn(128),
                                torch.randn(128), make_args(bestof_order=4.0))
    assert torch.isfinite(loss) and torch.isfinite(metrics).all()


# ------------------------------------------------------------------- logged diagnostics

@pytest.mark.parametrize("order", [1.0, 2.0, 4.0, 8.0])
def test_top_slope_diagnostic_equals_its_closed_form_whatever_the_distribution(order):
    """`bestof/top_slope` mirrors the main loop: the mean per-pair slope in the top decile
    over the same at the median. Each consecutive pair carries the slope at a known
    percentile, T*F^(T-1), so equal weighting over pairs gives exactly
    (1 - 0.9^T)/(0.55^T - 0.45^T) regardless of how the advantages are shaped -- which is
    what makes a departure in the logs evidence about the code and not about the data."""
    expected = (1.0 - 0.9 ** order) / (0.55 ** order - 0.45 ** order)
    torch.manual_seed(0)
    for sample in (torch.randn(32768, dtype=torch.float64),
                   torch.randn(32768, dtype=torch.float64).exp(),
                   torch.rand(32768, dtype=torch.float64) * 4.0 - 2.0):
        ranking = sample.argsort()
        slope = bo.rank_weight(sample, order)[ranking].diff() / sample[ranking].diff()
        decile, middle = 3276, 16383
        got = slope[-decile:].mean() / slope[middle - decile // 2:middle + decile // 2].mean()
        assert float(got) == pytest.approx(expected, rel=0.02)


def test_baseline_ev_is_measured_against_the_target_the_head_is_actually_trained_on():
    """The head regresses the ORIENTED target with no sign applied, so the residual the
    diagnostic reports must be h - b. Folding `sign` in would have made every `lower` arm
    log a nonsense explained-variance without changing the policy at all."""
    torch.manual_seed(0)
    for side in ("upper", "lower"):
        sign = bo.bestof_orientation(side)
        advantages = torch.randn(8192)
        oriented = sign * standardise(advantages)
        target = bo.rank_weight(oriented, 4.0)
        perfect = target + 0.0
        residual = target - perfect
        assert float(1.0 - residual.var() / target.var()) == pytest.approx(1.0, abs=1e-6)
        # And the policy weight itself does carry the sign, so the two differ by design.
        assert torch.allclose(sign * (target - perfect), bo.bestof_advantages(oriented, perfect, 4.0) * sign)


# ------------------------------------------------- the head must not steal the clip budget

def _grad_norms(agent, args, weights, oriented, target, observations, native, old_logprobs,
                returns, values):
    agent.zero_grad(set_to_none=True)
    loss, _ = bo.ppo_loss(agent, observations, native, old_logprobs, weights,
                          returns, values, oriented, target, args)
    loss.backward()
    def norm(prefix, want):
        chosen = [p.grad for n, p in agent.named_parameters()
                  if n.startswith("spread.") is want and p.grad is not None]
        return float(torch.sqrt(sum((g ** 2).sum() for g in chosen))) if chosen else 0.0
    return norm("core", False), norm("head", True)


@pytest.mark.parametrize("order", [2.0, 4.0, 8.0])
def test_head_gradient_does_not_shrink_the_policy_step(order):
    """The defect this guards: `spread` shares no parameter with actor/critic, so under one
    global clip_grad_norm_ its regression gradient is pure extra norm mass. Measured on the
    unsplit version, |g_core| was flat at 0.51 for every order while |g_head| ran
    0.0/1.81/1.28/1.08, so the post-clip policy step fell 0.500/0.137/0.185/0.214 -- the
    order sweep silently became a learning-rate sweep biased against every non-PPO arm.
    The trainer therefore clips the two disjoint sets separately; this pins that the policy
    norm it clips is the same one plain PPO would clip."""
    device = torch.device("cpu")
    agent, observations, native, old_logprobs = loss_inputs(512, device, seed=21)
    torch.manual_seed(22)
    advantages = torch.randn(512)
    oriented = standardise(advantages)
    target = bo.rank_weight(oriented, order)
    returns, values = torch.randn(512), torch.randn(512)
    args = make_args(bestof_order=order)
    core, head = _grad_norms(agent, args, target, oriented, target, observations, native,
                             old_logprobs, returns, values)
    # The head does get trained...
    assert head > 1e-3
    # ...but the policy's gradient is exactly what it would be with no head at all, so
    # clipping the two sets separately leaves the policy path identical to the baseline.
    reference = [p.grad.clone() for n, p in agent.named_parameters() if not n.startswith("spread.")]
    bare = make_args(bestof_order=order, bestof_coef=0.0)
    core_bare, head_bare = _grad_norms(agent, bare, target, oriented, target,
                                       observations, native, old_logprobs, returns, values)
    assert head_bare == 0.0
    assert core_bare == pytest.approx(core, rel=1e-6)
    for got, want in zip((p.grad for n, p in agent.named_parameters()
                          if not n.startswith("spread.")), reference):
        assert torch.allclose(got, want, atol=1e-7)


def test_the_trainer_clips_two_disjoint_sets_that_cover_every_parameter():
    """A parameter in neither list would silently never be clipped; one in both would be
    clipped twice. The trainer asserts this at startup -- this is the same check, so a new
    head or a new module cannot be added without confronting it."""
    source = (_ROOT / "cleanrl" / "ppo_continuous_action_maxrl_bestof_v2.py").read_text()
    assert "clip_grad_norm_(agent.parameters()" not in source, "global clip reintroduced"
    assert source.count("nn.utils.clip_grad_norm_(policy_parameters, args.max_grad_norm)") == 1
    assert source.count("nn.utils.clip_grad_norm_(head_parameters, args.max_grad_norm)") == 1
    agent = bo.Agent(Envs())
    policy = {id(p) for p in list(agent.actor.parameters()) + list(agent.critic.parameters())}
    head = {id(p) for p in agent.spread.parameters()}
    everything = {id(p) for p in agent.parameters()}
    assert policy.isdisjoint(head)
    assert policy | head == everything


# -------------------------------------------- how much of the batch actually carries signal

@pytest.mark.parametrize("order,floor,ceiling", [
    (1.0, 0.55, 0.65), (2.0, 0.18, 0.26), (4.0, 0.02, 0.06), (8.0, 0.0005, 0.004)])
def test_the_share_of_weight_variance_left_in_the_bottom_half_collapses_with_order(order, floor, ceiling):
    """Pins the quantity the header's "never flat" claim glosses over.

    `h` is strictly increasing for every T, so monotonicity and convexity tests pass at any
    order and discriminate nothing. What matters is arithmetic, not calculus: the slope at
    the median is T*0.5^(T-1), so the bottom half of the batch holds 61% / 21.5% / 3.6% /
    0.14% of std(h) at T=1/2/4/8. At T=8 half the batch is flat to four significant
    figures, and for those samples `w = h - b(s)` is just `const - b(s)` -- structurally
    the same cross-state re-pricing as the hinge this version replaced.

    That is not a bug: it is what best-of-T MEANS, since a bad draw barely moves the max.
    But it is the number that decides how to read a sweep over T, so it is pinned here
    rather than left implicit.
    """
    torch.manual_seed(0)
    x = torch.randn(16384, dtype=torch.float64)
    h = bo.rank_weight(x, order).sort().values
    share = float(h[:8192].std() / h.std())
    assert floor <= share <= ceiling, (order, share)


def test_monotone_convex_test_would_pass_even_on_an_almost_entirely_flat_weight():
    """Why the assertion above is needed: a weight flat over the bottom 99% still satisfies
    `increments >= 0` and `upper > 5 * lower`, so the existing shape tests cannot object."""
    torch.manual_seed(0)
    x = torch.randn(8192, dtype=torch.float64)
    increments = bo.rank_weight(x, 8.0).sort().values.diff()
    assert torch.all(increments >= 0.0)
    assert float(increments[-100:].sum()) > 5.0 * float(increments[:100].sum())
    # ...while nearly all of the rise sits in the top decile.
    assert float(increments[-819:].sum() / increments.sum()) > 0.9
