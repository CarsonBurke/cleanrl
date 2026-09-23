"""Contracts for MaxRL-PPO v2's mirrored weight and its direction-vs-variance control.

v1 measured that MaxRL's w_T(p) loses monotonically in the dose on HalfCheetah. v2's
claim is that the sign is the problem, not the magnitude, so these tests pin (a) that
`sharpen` really is the exact mirror of `weight` inside the same bounded family, (b)
that the three arms differ ONLY in how the multiplier correlates with the state, and
(c) that `reallocation_rho` reads that correlation out with the sign the arm intends.
"""
import importlib.util
import pathlib

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "cleanrl" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


maxrl = _load("ppo_continuous_action_maxrl_v2")


class Args:
    """Only the fields maxrl_advantages reads."""
    def __init__(self, mode, order=8, quantile=0.8):
        self.maxrl_mode = mode
        self.maxrl_order = order
        self.maxrl_quantile = quantile


ORDERS = [1, 2, 4, 8, 16, 64]


# --- the mirror identity ----------------------------------------------------

@pytest.mark.parametrize("order", ORDERS)
def test_sharpen_is_the_exact_mirror_of_the_maxrl_weight(order):
    p = torch.linspace(maxrl.PASS_FLOOR, maxrl.PASS_CEIL, 4096, dtype=torch.float64)
    assert torch.allclose(
        maxrl.sharpen_weight(p, order), maxrl.maxrl_weight(1.0 - p, order), rtol=0, atol=0
    ), "the mirror must be the identity w^s(p) = w(1-p), not an approximation of it"


@pytest.mark.parametrize("order", ORDERS)
def test_sharpen_equals_its_geometric_series(order):
    """w_T^s(p) = (1-p^T)/(1-p) = sum_{k=1..T} p^(k-1), the partial geometric sum."""
    p = torch.linspace(0.02, 0.98, 512, dtype=torch.float64)
    series = sum(p ** (k - 1) for k in range(1, order + 1))
    assert torch.allclose(maxrl.sharpen_weight(p, order), series, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("order", ORDERS)
def test_sharpen_is_bounded_and_increasing(order):
    p = torch.linspace(maxrl.PASS_FLOOR, maxrl.PASS_CEIL, 8192, dtype=torch.float64)
    w = maxrl.sharpen_weight(p, order)
    assert torch.all(w >= 1.0 - 1e-12) and torch.all(w <= order + 1e-9)
    assert torch.all(w.diff() >= -1e-12), "w^s must increase in p, the opposite of w_T"
    # Endpoints: w^s(0)=1 and w^s(1)=T, mirroring w_T(0)=T and w_T(1)=1. At the CEIL
    # clamp the true value is T(1 - (T-1)(1-p)/2), so the tolerance scales with T.
    assert maxrl.sharpen_weight(torch.tensor(1e-12, dtype=torch.float64), order).item() == pytest.approx(1.0, rel=1e-9)
    assert maxrl.sharpen_weight(torch.tensor(1.0 - 1e-12, dtype=torch.float64), order).item() == pytest.approx(order, rel=1e-9)


@pytest.mark.parametrize("mode", ["weight", "sharpen", "shuffle"])
def test_order_one_collapses_every_arm_onto_ppo(mode):
    """T=1 truncates the Maclaurin series at REINFORCE, so w == 1 and v2 is PPO."""
    torch.manual_seed(0)
    advantages = torch.randn(1024)
    returns = torch.randn(1024)
    logits = torch.randn(1024)
    bar = torch.tensor(0.0)
    out, _, _ = maxrl.maxrl_advantages(
        advantages, returns, logits, bar, Args(mode, order=1),
        torch.Generator().manual_seed(0),
    )
    assert torch.allclose(out, advantages, rtol=1e-6, atol=1e-6)


# --- what the arms share, and what they do not ------------------------------

@pytest.fixture
def batch():
    torch.manual_seed(7)
    n = 4096
    # Correlate the pass logit with the advantage, as a trained head would be: states
    # that are doing well are states whose return target clears the bar.
    advantages = torch.randn(n)
    logits = 2.0 * advantages + 0.5 * torch.randn(n)
    returns = advantages + 1.0
    return advantages, returns, logits, torch.tensor(1.0)


@pytest.mark.parametrize("mode", ["weight", "sharpen", "shuffle"])
def test_every_arm_preserves_gradient_scale(batch, mode):
    """Mean-1 normalisation: the arms reallocate gradient, they do not add or remove it.

    Without this an arm would smuggle in an effective learning-rate change and the
    comparison against v1's weight arm would be confounded.
    """
    advantages, returns, logits, bar = batch
    out, _, _ = maxrl.maxrl_advantages(
        advantages, returns, logits, bar, Args(mode), torch.Generator().manual_seed(0)
    )
    # The multiplier averages to exactly 1, so it cannot shift the gradient's scale.
    ratio = out / advantages
    assert ratio.mean().item() == pytest.approx(1.0, rel=1e-5)


def test_shuffle_permutes_the_weights_it_does_not_change_them(batch):
    """The control must hold the multiplier DISTRIBUTION fixed; only its pairing moves."""
    advantages, returns, logits, bar = batch
    weighted, _, _ = maxrl.maxrl_advantages(
        advantages, returns, logits, bar, Args("weight"), torch.Generator().manual_seed(0)
    )
    shuffled, _, _ = maxrl.maxrl_advantages(
        advantages, returns, logits, bar, Args("shuffle"), torch.Generator().manual_seed(0)
    )
    assert torch.allclose(
        (weighted / advantages).sort().values, (shuffled / advantages).sort().values,
        rtol=1e-5, atol=1e-6,
    )
    assert not torch.allclose(weighted, shuffled), "the permutation must actually move"


def test_sharpen_and_weight_order_states_oppositely(batch):
    """The two arms are one sign flip apart: their multipliers must rank-invert."""
    advantages, returns, logits, bar = batch
    w, _, _ = maxrl.maxrl_advantages(advantages, returns, logits, bar, Args("weight"), None)
    s, _, _ = maxrl.maxrl_advantages(advantages, returns, logits, bar, Args("sharpen"), None)
    w_mult, s_mult = w / advantages, s / advantages
    rho = torch.corrcoef(torch.stack((
        w_mult.argsort().argsort().float(), s_mult.argsort().argsort().float()
    )))[0, 1]
    assert rho.item() == pytest.approx(-1.0, abs=1e-6)


# --- the diagnostic that reads the direction out ----------------------------

REALLOCATION = 9  # index of maxrl/reallocation_rho in the diagnostics tensor


def test_reallocation_rho_signs_the_three_arms(batch):
    """The single number that says which way gradient moved, per arm.

    Negative means gradient is being pushed toward states that are doing badly (MaxRL's
    intent, and v1's measured failure); positive means toward states doing well
    (sharpen's claim); ~0 means the multiplier carries no state information at all,
    which is exactly what makes shuffle a variance-only control.
    """
    advantages, returns, logits, bar = batch
    rhos = {}
    for mode in ("weight", "sharpen", "shuffle"):
        _, _, diagnostics = maxrl.maxrl_advantages(
            advantages, returns, logits, bar, Args(mode), torch.Generator().manual_seed(0)
        )
        rhos[mode] = diagnostics[REALLOCATION].item()
    assert rhos["weight"] < -0.9
    assert rhos["sharpen"] > 0.9
    assert abs(rhos["shuffle"]) < 0.05


def test_off_mode_is_a_no_op_inside_the_estimator_too(batch):
    """The control arm must survive a refactor that hoists the call out of main()'s guard.

    main() skips maxrl_advantages entirely for `off`, so this path is unreachable today.
    It is pinned anyway because `off` is the BASELINE: if it ever silently started
    reweighting, it would become the v1 arm that lost 1900 return, and the whole
    comparison would read as noise rather than as a broken control.
    """
    advantages, returns, logits, bar = batch
    out, _, _ = maxrl.maxrl_advantages(advantages, returns, logits, bar, Args("off"), None)
    assert torch.equal(out, advantages)


def test_an_unknown_mode_raises_rather_than_silently_reweighting(batch):
    advantages, returns, logits, bar = batch
    with pytest.raises(ValueError, match="unknown maxrl_mode"):
        maxrl.maxrl_advantages(advantages, returns, logits, bar, Args("typo"), None)


def test_defaults():
    assert "off" in maxrl.Args.__annotations__["maxrl_mode"].__args__
    assert maxrl.Args().maxrl_mode == "sharpen"
    assert maxrl.Args().maxrl_pass_detach is True


# --- the loss still compiles and runs in every arm --------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["off", "weight", "sharpen", "shuffle"])
@pytest.mark.parametrize("detach", [True, False])
def test_compiled_loss_backward_reaches_the_heads_it_should(mode, detach):
    import gymnasium as gym

    device = torch.device("cuda")

    class Envs:
        single_action_space = gym.spaces.Box(-1.0, 1.0, (6,))
        single_observation_space = gym.spaces.Box(-float("inf"), float("inf"), (17,))

    torch.manual_seed(0)
    agent = maxrl.Agent(Envs()).to(device)
    args = maxrl.Args()
    args.maxrl_mode, args.maxrl_pass_detach = mode, detach
    n = 64
    obs = torch.randn(n, 17, device=device)
    # Beta's support is (0,1): these are the NATIVE pre-scale samples the loss takes,
    # not the [-1,1] action the environment sees.
    actions = torch.rand(n, 6, device=device).clamp(1e-3, 1 - 1e-3)
    loss, metrics = maxrl.ppo_loss(
        agent, obs, actions, torch.randn(n, device=device), torch.randn(n, device=device),
        torch.randn(n, device=device), torch.randn(n, device=device),
        torch.randint(0, 2, (n,), device=device).float(), args,
    )
    assert torch.isfinite(loss) and torch.isfinite(metrics).all()
    loss.backward()
    trunk_grad = agent.critic_trunk[0].weight.grad
    pass_grad = agent.pass_head.weight.grad
    if mode == "off":
        # The pass head is outside the loss, so it must receive nothing -- that is what
        # makes `off` a bit-identical control against plain PPO.
        assert pass_grad is None or torch.count_nonzero(pass_grad) == 0
    else:
        assert pass_grad is not None and torch.count_nonzero(pass_grad) > 0
    assert trunk_grad is not None and torch.count_nonzero(trunk_grad) > 0
