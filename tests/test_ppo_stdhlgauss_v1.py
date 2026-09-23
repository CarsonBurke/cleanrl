"""Contracts for the target-standardized HL-Gauss PPO trainer.

Queue through mlq: CUDA contracts, not shortened environment training runs.
These pin the properties the ablation rests on -- that the arms differ in
exactly the value readout, that the scalar arm is still the frozen baseline,
and that the categorical trust region and the normalizer reframe do what the
scalar formulas do.
"""

from dataclasses import replace
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_normres_indclip_headscope_v5 as baseline
from cleanrl import ppo_continuous_action_normres_stdhlgauss_v1 as trainer

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="queued CUDA test required"),
]


def _spaces():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), np.float32),
        single_action_space=gym.spaces.Box(
            np.array([-3, -2, 1, -4, 0, -1], np.float32),
            np.array([2, 4, 5, -1, 2, 8], np.float32),
        ),
    )


def _args(**overrides):
    args = trainer.Args(**overrides)
    return trainer.validate_args(args)


@pytest.fixture
def device():
    with torch.random.fork_rng(devices=["cuda"]):
        torch.manual_seed(1)
        yield torch.device("cuda")


def _agent(device, **overrides):
    torch.manual_seed(1)
    return trainer.Agent(_spaces(), _args(**overrides)).to(device)


def _batch(device, count=512):
    generator = torch.Generator(device="cpu").manual_seed(7)
    observations = torch.randn((count, 17), generator=generator).to(device)
    native = torch.rand((count, 6), generator=generator).clamp(0.05, 0.95).to(device)
    advantages = torch.randn((count,), generator=generator).to(device)
    targets = (torch.randn((count,), generator=generator) * 0.4 + 3.8).to(device)
    return observations, native, advantages, targets


def test_every_arm_shares_one_actor_and_critic_trunk():
    """The ablation claims a single-flag difference; initialization must honor it."""
    reference = None
    for value_loss in ("mse", "mse_popart", "hlgauss", "mse_softmax"):
        torch.manual_seed(1)
        agent = trainer.Agent(_spaces(), _args(value_loss=value_loss))
        state = {name: tensor.clone() for name, tensor in agent.state_dict().items()}
        if reference is None:
            reference = state
            continue
        for name, tensor in state.items():
            if name.startswith("actor") or name.startswith("critic.0"):
                torch.testing.assert_close(tensor, reference[name], rtol=0, atol=0, msg=name)


def test_scalar_arm_reproduces_the_frozen_baseline_loss(device):
    args = _args(value_loss="mse")
    agent = _agent(device, value_loss="mse")
    reference = baseline.Agent(_spaces(), placement="pre", norm_kind="rms", activation="stiglu").to(device)
    reference.load_state_dict(
        {name: value for name, value in agent.state_dict().items() if not name.startswith("histogram")}
    )
    observations, native, advantages, targets = _batch(device)
    with torch.no_grad():
        alpha, beta, values = agent.get_policy_and_readout(observations)
        old_logprobs = agent.action_logprob(alpha, beta, native)
        old_values = values.view(-1)
    reference_args = baseline.Args(
        clip_heads=args.clip_heads, clip_vloss=args.clip_vloss, norm_adv=args.norm_adv, vf_coef=args.vf_coef
    )
    expected, _ = baseline.ppo_loss(
        reference, observations, native, old_logprobs, advantages, targets, old_values, reference_args
    )
    actual, _ = trainer.scalar_ppo_loss(
        agent, observations, native, old_logprobs, advantages, targets, old_values, args
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_categorical_trust_region_drops_exactly_the_scalar_zero_gradient_samples(device):
    """PPO's value clip works by zeroing a gradient; the mask must match that set."""
    args = _args(value_loss="hlgauss")
    count = 4096
    generator = torch.Generator(device="cpu").manual_seed(3)
    old_values = (torch.randn(count, generator=generator) * 0.4 + 3.8).to(device)
    newvalue = old_values + (torch.randn(count, generator=generator) * 0.4).to(device)
    targets = old_values + (torch.randn(count, generator=generator) * 0.6).to(device)
    probe = newvalue.clone().requires_grad_(True)
    scalar, scalar_clipfrac = trainer.scalar_value_loss(probe, targets, old_values, args)
    (gradient,) = torch.autograd.grad(scalar, probe)
    clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
    frozen = (clipped - targets).square() > (newvalue - targets).square()
    assert torch.equal(frozen, gradient == 0)
    assert scalar_clipfrac.item() == pytest.approx(frozen.float().mean().item())
    assert 0.05 < frozen.float().mean().item() < 0.95


def test_matched_ce_gradient_equals_the_mse_gradient_in_value_space(device):
    """Raw CE runs a ~100x value-space step at the same vf_coef; matched does not.

    The gain Var_p(z)*scale^2 is the exact first-order factor for a *continuous*
    Gaussian label. Discretizing the label onto bins inflates Var_p(z) by the
    bin's own spread, so the residual mismatch must fall monotonically as sigma
    grows relative to the bin width, and vanish in that limit. Pinning the
    trend states the claim; pinning one number would only record fp noise.
    """
    _, _, _, targets = _batch(device, count=1024)

    def mismatch(sigma_bins):
        agent = _agent(device, value_loss="hlgauss", value_sigma_bins=sigma_bins)
        histogram = _histogram(agent)
        histogram.observe(targets)
        # A head already in the label family, displaced slightly: the regime
        # where the first-order CE/MSE correspondence is meant to hold.
        values = targets - 0.01 * histogram.scale
        base = histogram.project(values).clamp_min(1e-30).log()
        probs = histogram.project(targets)

        def value_gradient(ce_scale, loss_kind):
            readout = base.clone().requires_grad_(True)
            newvalue = histogram.decode(readout)
            args = _args(value_loss="hlgauss", clip_vloss=False, ce_scale=ce_scale)
            if loss_kind == "mse":
                loss, _ = trainer.scalar_value_loss(newvalue, targets, values, args)
            else:
                loss, _ = trainer.categorical_value_loss(
                    agent, readout, newvalue, targets, values, probs, args
                )
            return torch.autograd.grad(loss, readout)[0]

        mse = value_gradient("raw", "mse")
        raw = value_gradient("raw", "ce")
        matched = value_gradient("matched", "ce")
        # Scale by the gradient that exists; most bins carry ~0 probability.
        return (raw.norm() / mse.norm()).item(), ((matched - mse).abs().max() / mse.abs().max()).item()

    gains, residuals = zip(*(mismatch(sigma) for sigma in (0.75, 2.0, 4.0)))
    assert min(gains) > 20.0
    assert residuals[0] > residuals[1] > residuals[2]
    assert residuals[0] < 0.2 and residuals[-1] < 0.05


def _critic_modules(agent):
    trunk, head = agent.critic[0], agent.critic[1]
    assert isinstance(head, torch.nn.Linear)
    return trunk, head


def _histogram(agent):
    histogram = agent.histogram
    assert histogram is not None
    return histogram


@pytest.mark.parametrize("value_loss", ["hlgauss", "mse_popart"])
def test_reframing_preserves_the_unchanged_head_prediction(device, value_loss):
    """Updating the normalizer must not look like a policy-driven value move."""
    agent = _agent(device, value_loss=value_loss)
    histogram = _histogram(agent)
    observations, _, _, targets = _batch(device)
    histogram.observe(targets)
    with torch.no_grad():
        before = agent.get_value(observations).view(-1)
        previous_mean, previous_scale = histogram.mean.clone(), histogram.scale.clone()
        histogram.observe(targets + 1.7)
        reframed = histogram.mean + histogram.scale * (before - previous_mean) / previous_scale
        after = agent.get_value(observations).view(-1)
    torch.testing.assert_close(reframed, after, rtol=1e-5, atol=1e-5)
    assert (after - before).abs().mean() > 0.5


def test_popart_absorbs_a_global_target_shift_without_touching_the_head(device):
    """The proxy's decisive control: standardization, not classification.

    A pure shift of the return distribution must be tracked exactly by the
    normalizer, leaving the head's standardized output untouched, so the
    critic spends its capacity on cross-state structure instead of re-fitting
    the drifting level every iteration.
    """
    agent = _agent(device, value_loss="mse_popart")
    histogram = _histogram(agent)
    observations, _, _, targets = _batch(device)
    histogram.observe(targets)
    with torch.no_grad():
        standardized = agent.critic(observations).view(-1)
        before = agent.get_value(observations).view(-1)
        histogram.observe(targets + 2.5)
        after = agent.get_value(observations).view(-1)
        unchanged = agent.critic(observations).view(-1)
    torch.testing.assert_close(unchanged, standardized, rtol=0, atol=0)
    torch.testing.assert_close(after - before, torch.full_like(before, 2.5), rtol=0, atol=2e-4)


def test_mse_softmax_trains_the_categorical_head_through_the_decoded_mean(device):
    agent = _agent(device, value_loss="mse_softmax")
    args = _args(value_loss="mse_softmax")
    histogram = _histogram(agent)
    observations, native, advantages, targets = _batch(device)
    histogram.observe(targets)
    with torch.no_grad():
        alpha, beta, readout = agent.get_policy_and_readout(observations)
        old_logprobs = agent.action_logprob(alpha, beta, native)
        old_values = agent.decode(readout).view(-1)
    loss, metrics = trainer.categorical_ppo_loss(
        agent,
        observations,
        native,
        old_logprobs,
        advantages,
        targets,
        old_values,
        histogram.project(targets),
        args,
    )
    agent.zero_grad(set_to_none=True)
    loss.backward()
    trunk, head = _critic_modules(agent)
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert trunk.in_proj.weight.grad.abs().sum() > 0
    assert metrics.shape == (7,)


def test_small_gain_head_starts_at_the_support_center_with_a_live_trunk(device):
    agent = _agent(device, value_loss="hlgauss")
    histogram = _histogram(agent)
    observations, _, _, targets = _batch(device)
    histogram.observe(targets)
    value = agent.get_value(observations).view(-1)
    assert (value - histogram.mean).abs().max() < 0.25 * histogram.scale
    loss = -(histogram.project(targets) * agent.critic(observations).log_softmax(-1)).sum(-1).mean()
    agent.zero_grad(set_to_none=True)
    loss.backward()
    trunk, _ = _critic_modules(agent)
    assert trunk.in_proj.weight.grad.abs().sum() > 0


def test_rejects_cross_entropy_knobs_on_non_hlgauss_arms():
    with pytest.raises(ValueError):
        trainer.validate_args(replace(trainer.Args(), value_loss="mse", ce_scale="matched"))
    with pytest.raises(ValueError):
        trainer.validate_args(replace(trainer.Args(), value_loss="mse_softmax", ce_scale="matched"))


@pytest.mark.parametrize("value_loss", ["mse", "mse_popart", "hlgauss", "mse_softmax"])
def test_compiled_update_path_runs_and_sees_normalizer_updates(device, value_loss):
    """The main loop compiles the loss and CUDA-graphs the rollout forward.

    Buffers mutated between iterations must be visible to the captured graph,
    which is the whole premise of a normalizer that moves during training.
    """
    agent = _agent(device, value_loss=value_loss)
    args = _args(value_loss=value_loss)
    observations, native, advantages, targets = _batch(device, count=4096)
    histogram = agent.histogram
    if histogram is not None:
        histogram.observe(targets)

    def rollout_statistics(inputs, actions):
        alpha, beta, readout = agent.get_policy_and_readout(inputs)
        return agent.decode(readout).flatten(), agent.action_logprob(alpha, beta, actions)

    if value_loss == "mse":

        def loss_model(inputs, actions, old_logprobs, adv, target, old_values, probs):
            return trainer.scalar_ppo_loss(agent, inputs, actions, old_logprobs, adv, target, old_values, args)

    else:

        def loss_model(inputs, actions, old_logprobs, adv, target, old_values, probs):
            return trainer.categorical_ppo_loss(
                agent, inputs, actions, old_logprobs, adv, target, old_values, probs, args
            )

    compiled_statistics = trainer.graph_compile(rollout_statistics)
    compiled_value = torch.compile(
        agent.get_value, fullgraph=True, dynamic=True, options={"triton.cudagraphs": False}
    )
    compiled_loss = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4, eps=1e-5, fused=True)
    probs = (
        histogram.project(targets)
        if histogram is not None
        else torch.zeros((observations.shape[0], 1), device=device)
    )
    with torch.no_grad():
        old_values, old_logprobs = compiled_statistics(observations, native)
        assert torch.isfinite(compiled_value(observations[:37])).all()
    for _ in range(2):
        torch.compiler.cudagraph_mark_step_begin()
        loss, metrics = compiled_loss(
            observations, native, old_logprobs, advantages, targets, old_values, probs
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(metrics).all()
    if histogram is None:
        return
    with torch.no_grad():
        before = compiled_value(observations[:64]).view(-1).clone()
        histogram.observe(targets + 5.0)
        after = compiled_value(observations[:64]).view(-1)
    assert (after - before).abs().mean() > 0.1
