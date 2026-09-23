"""Contract tests for MaxRL-bestof v1 (exact best-of-T policy objective).

The load-bearing claims are: psi_1 is the identity so PPO is the T=1 member exactly, the
weight reproduces the true gradient of E[max of T] (including the T-DISCRIMINATING scale
derivative, which d/dmu cannot see), and the transform is smooth where tail_v1's was flat.
"""
import importlib.util
import math
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "cleanrl" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bo = _load("maxrl_bestof_v1", "ppo_continuous_action_maxrl_bestof_v1.py")
base = _load("ppo_baseline", "ppo_continuous_action.py")
NODES = 128
# psi_T is monotone and convex in exact arithmetic; the midpoint rule reproduces that only
# to its own O(h^2) error, which is ~2e-7 at this node count and concentrated in the flat
# left tail where the true slope is itself below 1e-15. The trainer computes in float32 and
# standardises the weight to unit variance, so this sits far under the noise floor.
QUAD_TOL = 1e-5


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


def normal_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# ------------------------------------------------------------------ psi_T is the family

def test_psi_at_order_one_is_the_identity_in_floating_point():
    """Not "approximately PPO": the same tensor, which is what makes T=1 a member."""
    x = torch.linspace(-8.0, 8.0, 401, dtype=torch.float64)
    assert torch.equal(bo.rank_weight(x, 1.0, NODES), x)
    # Outside the quadrature range the linear extension carries it, exactly at order 1
    # because Phi^0 = 1 makes the asymptotic slope the true slope everywhere.
    far = torch.tensor([-1e4, -50.0, 50.0, 1e4], dtype=torch.float64)
    assert torch.allclose(bo.rank_weight(far, 1.0, NODES), far, rtol=1e-12)


@pytest.mark.parametrize("order", [2.0, 4.0, 8.0, 16.0])
def test_psi_slope_is_the_order_t_rank_density(order):
    """d psi_T/dx = T Phi(x)^(T-1) is the claim the whole construction rests on."""
    x = torch.linspace(-3.0, 3.0, 13, dtype=torch.float64).requires_grad_(True)
    bo.rank_weight(x, order, NODES).sum().backward()
    expected = order * normal_cdf(x.detach()).pow(order - 1.0)
    assert float((x.grad - expected).abs().max()) < 1e-3


@pytest.mark.parametrize("order", [1.0, 2.0, 4.0, 8.0])
def test_weight_reproduces_the_location_gradient_of_the_expected_maximum(order):
    """E[w * score] must equal d/dmu E[max of T] = 1, by translation invariance."""
    torch.manual_seed(0)
    z = torch.randn(2_000_000, dtype=torch.float64)
    constant = bo.rank_weight(torch.randn(1_000_000, dtype=torch.float64), order, NODES).mean()
    weight = bo.rank_weight(z, order, NODES) - constant
    assert abs(float((weight * z).mean()) - 1.0) < 0.01


@pytest.mark.parametrize("order,expected", [(1.0, 0.0), (2.0, 0.5642), (4.0, 1.0294), (8.0, 1.4236)])
def test_weight_reproduces_the_scale_gradient_which_actually_discriminates_t(order, expected):
    """d/dsigma E[max of T] = E[max of T standard normals], which DOES depend on T.

    The location check above returns 1 for every order, so it cannot tell a correct psi_T
    from a wrong one; this is the test that can.
    """
    torch.manual_seed(1)
    z = torch.randn(4_000_000, dtype=torch.float64)
    constant = bo.rank_weight(torch.randn(2_000_000, dtype=torch.float64), order, NODES).mean()
    weight = bo.rank_weight(z, order, NODES) - constant
    assert abs(float((weight * (z * z - 1.0)).mean()) - expected) < 0.01


@pytest.mark.parametrize("order", [2.0, 4.0, 8.0])
def test_psi_is_monotone_and_never_flat_unlike_the_hinge_it_replaces(order):
    """The defect this version exists to fix: tail_v1 gave a whole flat region the same
    push, which is the gradation dense reward supplies and the hinge discarded."""
    x = torch.linspace(-3.0, 3.0, 2001, dtype=torch.float64)
    psi = bo.rank_weight(x, order, NODES)
    steps = psi[1:] - psi[:-1]
    assert torch.all(steps > -QUAD_TOL), "psi_T must be non-decreasing"
    # Convex: a sample's slope grows with its percentile, so the top is favoured smoothly.
    assert torch.all(steps[1:] >= steps[:-1] - QUAD_TOL)
    # And genuinely NOT flat where the hinge was: a sample at the 84th percentile must
    # earn a visibly steeper slope than one at the 16th, where the hinge gave them either
    # the same slope or none at all. Both points sit well above the quadrature noise.
    step_at = lambda point: float(steps[int((point + 3.0) / 6.0 * (len(steps) - 1))])
    assert step_at(1.0) > 3.0 * step_at(-1.0) > 0.0


def test_quadrature_node_count_is_adequate_at_the_default():
    """Pins the `bestof_nodes` default against the exact slope, so lowering it is a visible
    change rather than a silent accuracy loss."""
    x = torch.linspace(-4.0, 4.0, 33, dtype=torch.float64).requires_grad_(True)
    bo.rank_weight(x, 8.0, bo.Args().bestof_nodes).sum().backward()
    expected = 8.0 * normal_cdf(x.detach()).pow(7.0)
    assert float((x.grad - expected).abs().max()) < 1e-3


# ------------------------------------------------------------------- PPO is the T=1 arm

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


def test_spread_head_is_constructed_last_so_actor_and_critic_init_is_unchanged():
    torch.manual_seed(7)
    agent = bo.Agent(Envs())
    torch.manual_seed(7)
    reference = base.Agent(Envs())
    named = dict(reference.named_parameters())
    assert set(named) == {n for n, _ in agent.named_parameters() if not n.startswith("spread.")}
    for name, parameter in agent.named_parameters():
        if not name.startswith("spread."):
            assert torch.equal(parameter, named[name]), name


def test_order_one_gives_the_spread_head_no_gradient():
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
    """The falsification arm must share the control exactly, or it is not a control."""
    torch.manual_seed(2)
    advantages = torch.randn(4096)
    advantages = advantages - advantages.mean()
    scale = torch.full_like(advantages, 1.0)
    baseline = torch.zeros_like(advantages)

    def arm(side, order):
        sign = bo.bestof_orientation(side)
        return sign * bo.bestof_advantages(sign * advantages, scale, baseline, order, NODES)

    assert torch.allclose(arm("upper", 1.0), arm("lower", 1.0), atol=1e-6)
    assert float((arm("upper", 4.0) - arm("lower", 4.0)).abs().max()) > 1.0


def test_lower_arm_is_the_exact_reflection_of_the_upper_arm():
    torch.manual_seed(3)
    advantages = torch.randn(2048)
    scale, baseline = torch.ones_like(advantages), torch.zeros_like(advantages)
    upper_on_negated = bo.bestof_advantages(-advantages, scale, baseline, 4.0, NODES)
    lower = -bo.bestof_advantages(-advantages, scale, baseline, 4.0, NODES)
    assert torch.allclose(lower, -upper_on_negated)
    # Pessimism must rank samples in the OPPOSITE order to optimism.
    upper = bo.bestof_advantages(advantages, scale, baseline, 4.0, NODES)
    rho = float(torch.corrcoef(torch.stack((upper, lower)))[0, 1])
    assert rho > 0.0, "both arms still increase with A; they differ in curvature, not sign"
    assert rho < 0.95, "the arms must not be near-identical reweightings"


# -------------------------------------------------------------------- estimator validity

@pytest.mark.parametrize("order", [2.0, 4.0, 8.0])
def test_weight_is_conditionally_centred_when_the_baseline_head_is_correct(order):
    """w = sigma psi_T(A/sigma) - b(s) must be mean-zero GIVEN s, or the arms can be
    ordered by a state-dependent offset rather than by the mechanism -- the exact defect
    the v4 post-mortem turned up."""
    torch.manual_seed(0)
    states = 64
    per_state = 4096
    offsets = []
    for index in range(states):
        scale_value = 0.3 + index / states
        advantages = torch.randn(per_state) * scale_value
        scale = torch.full_like(advantages, scale_value)
        weighted = scale * bo.rank_weight(advantages / scale, order, NODES)
        baseline = torch.full_like(advantages, float(weighted.mean()))
        out = bo.bestof_advantages(advantages, scale, baseline, order, NODES)
        offsets.append(float(out.mean()))
    assert max(abs(o) for o in offsets) < 0.05


def test_scale_nll_is_minimised_at_the_conditional_standard_deviation():
    torch.manual_seed(4)
    truth = 1.7
    samples = torch.randn(200000) * truth
    scale = torch.full((1,), 0.5, requires_grad=True)
    optimizer = torch.optim.Adam([scale], lr=0.02)
    for _ in range(800):
        optimizer.zero_grad()
        bo.scale_nll(samples, scale.abs() + 1e-6).mean().backward()
        optimizer.step()
    assert abs(float(scale.abs()) - truth) < 0.02


def test_spread_head_can_learn_a_state_dependent_scale_on_fresh_data():
    """Capacity test, so a null `bestof/scale_dispersion` is a fact about the environment.

    Advantages are resampled every step, which is the trainer's regime: the head is read
    on a fresh batch BEFORE being updated on it, so memorising single draws does not help.
    """
    torch.manual_seed(0)
    agent = bo.Agent(Envs())
    observations = torch.randn(4096, 17)
    truth = 0.4 + observations[:, 0].abs()
    optimizer = torch.optim.Adam(agent.spread.parameters(), lr=3e-3)
    for _ in range(600):
        advantages = torch.randn(4096) * truth
        optimizer.zero_grad()
        scale, baseline = agent.get_spread(observations)
        weighted = scale.detach() * bo.rank_weight(advantages / scale.detach(), 4.0, NODES)
        (bo.scale_nll(advantages, scale) + (baseline - weighted) ** 2).mean().backward()
        optimizer.step()
    with torch.no_grad():
        scale, _ = agent.get_spread(observations)
        residual = float(((scale - truth) ** 2).mean() / truth.var())
    assert residual < 0.25, "scale head collapsed instead of tracking the state"
    assert float(scale.std() / scale.mean()) > 0.15


def test_unit_scale_arm_removes_all_state_dependence_from_the_weight():
    """The `unit` arm must differ from `state` by exactly one thing: whether sigma(s) is
    consumed. Both still train the same head, so the arms share head dynamics."""
    torch.manual_seed(5)
    advantages = torch.randn(4096)
    varying = 0.5 + torch.rand(4096)
    baseline = torch.zeros_like(advantages)
    state_arm = bo.bestof_advantages(advantages, varying, baseline, 4.0, NODES)
    unit_arm = bo.bestof_advantages(advantages, torch.ones_like(varying), baseline, 4.0, NODES)
    assert not torch.allclose(state_arm, unit_arm)
    # The unit arm is a fixed monotone function of A alone: rank in A must be rank in w.
    order = advantages.argsort()
    assert torch.all(unit_arm[order][1:] >= unit_arm[order][:-1] - QUAD_TOL)


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


def test_baseline_target_carries_no_gradient_into_the_scale_head():
    """The baseline regression must estimate the conditional mean GIVEN the scale, never
    drive the scale toward whatever value makes its own target easy to fit. The target is
    now an input tensor precomputed once per batch, so the only gradient reaching the scale
    row is the NLL's -- this asserts that equality rather than trusting the structure."""
    device = torch.device("cpu")
    agent, observations, native, old_logprobs = loss_inputs(256, device, seed=13)
    torch.manual_seed(14)
    advantages = torch.randn(256)
    weighted_target = torch.randn(256)
    args = make_args(bestof_order=4.0, bestof_coef=1.0, vf_coef=0.0, ent_coef=0.0)
    for parameter in list(agent.actor.parameters()) + list(agent.critic.parameters()):
        parameter.requires_grad_(False)
    loss, _ = bo.ppo_loss(agent, observations, native, old_logprobs, advantages,
                          torch.randn(256), torch.randn(256), advantages, weighted_target, args)
    loss.backward()
    grad = agent.spread[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all()

    reference = bo.Agent(Envs())
    reference.load_state_dict(agent.state_dict())
    scale, _ = reference.get_spread(observations)
    bo.scale_nll(advantages, scale).mean().backward()
    assert torch.allclose(grad[0], reference.spread[-1].weight.grad[0], atol=1e-6)
    # And the baseline row must be driven, or the head is not learning its own target.
    assert float(grad[1].abs().sum()) > 1e-6


# ------------------------------------------------------------------------ argument guard

@pytest.mark.parametrize("order", [0.0, 0.5, -1.0])
def test_validate_args_rejects_orders_below_one(order):
    with pytest.raises(ValueError, match="bestof_order"):
        bo.validate_args(make_args(bestof_order=order))


def test_validate_args_rejects_unknown_side_scale_and_bad_coefficients():
    with pytest.raises(ValueError, match="bestof_side"):
        bo.validate_args(make_args(bestof_side="sideways"))
    with pytest.raises(ValueError, match="bestof_scale"):
        bo.validate_args(make_args(bestof_scale="adaptive"))
    with pytest.raises(ValueError, match="bestof_nodes"):
        bo.validate_args(make_args(bestof_nodes=4))
    with pytest.raises(ValueError, match="bestof_coef"):
        bo.validate_args(make_args(bestof_coef=-1.0))


def test_validate_args_accepts_the_whole_sweep():
    for order in (1.0, 2.0, 4.0):
        for side in ("upper", "lower"):
            for scale in ("state", "unit"):
                args = bo.validate_args(make_args(bestof_order=order, bestof_side=side,
                                                  bestof_scale=scale))
                assert args.batch_size == args.num_envs * args.num_steps


# ----------------------------------------------------------------------------- numerics

@pytest.mark.parametrize("order", [2.0, 8.0, 32.0])
def test_transform_is_finite_at_the_scale_floor_and_extreme_advantages(order):
    """The scale floor is the guard that keeps A/sigma bounded; check it actually is."""
    advantages = torch.tensor([-50.0, -8.0, 0.0, 8.0, 50.0])
    scale = torch.full_like(advantages, bo.SPREAD_FLOOR)
    out = bo.bestof_advantages(advantages, scale, torch.zeros_like(advantages), order, NODES)
    assert torch.isfinite(out).all()
    assert torch.all(out[1:] >= out[:-1] - QUAD_TOL)


def test_loss_is_finite_for_a_degenerate_minibatch():
    device = torch.device("cpu")
    agent, observations, native, old_logprobs = loss_inputs(128, device, seed=15)
    weights = torch.full((128,), -0.3)  # zero variance: standardisation must not explode
    loss, metrics = bo.ppo_loss(agent, observations, native, old_logprobs, weights,
                                torch.randn(128), torch.randn(128), torch.randn(128),
                                torch.randn(128), make_args(bestof_order=4.0))
    assert torch.isfinite(loss) and torch.isfinite(metrics).all()


@pytest.mark.parametrize("order", [2.0, 4.0, 8.0, 32.0])
def test_psi_is_accurate_far_outside_the_quadrature_range(order):
    """Regression for a real defect: substituting v = x t spaces nodes |x|/n apart, so a
    single unclamped rule steps over the whole region near zero. At x = -1000, T = 4 it
    returned -2e-30 for a true value of -0.167 -- a weight that should saturate instead
    vanished. The clamp plus asymptotic extension is what fixes it."""
    far = torch.tensor([-1e4, -1e3, -50.0], dtype=torch.float64)
    edge = bo.rank_weight(torch.tensor([-bo.RANK_LIMIT], dtype=torch.float64), order, NODES)
    got = bo.rank_weight(far, order, NODES)
    # Below the clamp psi_T saturates: it must equal its edge value to high RELATIVE
    # precision, which an absolute tolerance would not catch at large T where the edge
    # value is itself tiny.
    assert torch.allclose(got, edge.expand(3), rtol=1e-6, atol=0.0)


def test_psi_saturates_to_the_known_value_rather_than_underflowing():
    """The concrete number from the defect: an unclamped rule gave psi_4(-1000) = -2e-30."""
    got = float(bo.rank_weight(torch.tensor([-1000.0], dtype=torch.float64), 4.0, NODES))
    assert got == pytest.approx(-0.16745, abs=1e-3)
