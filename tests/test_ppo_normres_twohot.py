"""CUDA contracts for the selected pre/RMS/SiTUGLU base; run through mlq."""

import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_normres_v2 as baseline
from cleanrl import ppo_continuous_action_32xlr_1mb_noadvnorm_normres_dreamer_twohot_v3 as trainer

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="queued CUDA test required"),
]

SELECTED_TRUNK = dict(placement="pre", norm_kind="rms", activation="stiglu")


@pytest.fixture
def device():
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    with torch.random.fork_rng():
        torch.manual_seed(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            yield torch.device("cuda")
        finally:
            torch.set_float32_matmul_precision(precision)
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _spaces():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), np.float32),
        # Asymmetric physical bounds expose native/physical density mistakes.
        single_action_space=gym.spaces.Box(
            np.array([-3, -2, 1, -4, 0, -1], np.float32),
            np.array([2, 4, 5, -1, 2, 8], np.float32),
        ),
    )


def _agent(module, device, **kwargs):
    torch.manual_seed(1)
    with torch.device(device):
        return module.Agent(_spaces(), **SELECTED_TRUNK, **kwargs)


def _observations(count, device):
    rng = np.random.default_rng(37)
    return torch.as_tensor(rng.standard_normal((count, 17)).astype(np.float32), device=device)


def _policy_inputs(agent, observations):
    count = observations.shape[0]
    native = torch.linspace(0.03, 0.97, count * 6, device=observations.device).reshape(count, 6)
    with torch.no_grad():
        alpha, beta, values = agent.get_policy_and_value(observations)
        old_logprobs = agent.action_logprob(alpha, beta, native)
        # Both advantage signs cross both sides of PPO's ratio clipping window.
        offsets = observations.new_tensor([0.0, 0.6, -0.6, 0.6, -0.6, 0.0])
        old_logprobs = old_logprobs - offsets.repeat((count + 5) // 6)[:count]
    # Deliberately nonzero mean and non-unit variance: never standardized GAE.
    advantages = observations.new_tensor([2.0, 7.0, 11.0, -3.0, -5.0, -1.0])
    return native, old_logprobs, advantages.repeat((count + 5) // 6)[:count], values.flatten()


def _assert_parameters(actual, expected, *, gradients=False, exact=False):
    left, right = dict(actual.named_parameters()), dict(expected.named_parameters())
    assert left.keys() == right.keys()
    for name in left:
        a = left[name].grad if gradients else left[name]
        b = right[name].grad if gradients else right[name]
        assert a is not None and b is not None, name
        torch.testing.assert_close(
            a,
            b,
            rtol=0 if exact else 2e-6,
            atol=0 if exact else 2e-7,
            msg=name,
        )


def _dreamer_bins(num_bins, extent, device):
    # Independent transcription of Dreamer3 heads.symexp_twohot, including the
    # duplicated central zero for even K, rather than a uniform symlog grid.
    half_count = (num_bins - 1) // 2 + 1 if num_bins % 2 else num_bins // 2
    half = torch.linspace(-math.log1p(extent), 0, half_count, device=device, dtype=torch.float32)
    half = half.sign() * half.abs().expm1()
    reflected = -half[:-1].flip(0) if num_bins % 2 else -half.flip(0)
    return torch.cat((half, reflected))


def _dreamer_labels(targets, bins):
    # Deliberately use the dense comparison/one-hot formulation in D3 outs.py,
    # independent of the production searchsorted/scatter implementation.
    targets = targets.detach().float()
    below = ((bins <= targets[..., None]).sum(-1) - 1).clamp(0, len(bins) - 1)
    above = (len(bins) - (bins > targets[..., None]).sum(-1)).clamp(0, len(bins) - 1)
    equal = below == above
    to_below = torch.where(equal, 1.0, (bins[below] - targets).abs())
    to_above = torch.where(equal, 1.0, (bins[above] - targets).abs())
    total = to_below + to_above
    return (
        F.one_hot(below, len(bins)) * (to_above / total)[..., None]
        + F.one_hot(above, len(bins)) * (to_below / total)[..., None]
    )


def _dreamer_decode(logits, bins):
    probs = logits.float().softmax(-1)
    middle = len(bins) // 2
    if len(bins) % 2:
        center = (probs[..., middle : middle + 1] * bins[middle : middle + 1]).sum(-1)
        positive = probs[..., middle + 1 :] * bins[middle + 1 :]
    else:
        center = 0
        positive = probs[..., middle:] * bins[middle:]
    return center + ((probs[..., :middle] * bins[:middle]).flip(-1) + positive).sum(-1)


def test_head_changes_preserve_selected_base_initialization(device):
    reference = _agent(baseline, device)
    observations = _observations(32, device)
    with torch.no_grad():
        expected_policy = reference.actor(observations)
        expected_features = reference.critic[0](observations)
        expected_values = reference.get_value(observations)
        for value_loss, bins, extent, spacing in [
            ("mse", 255, 10.0, "symexp"),
            ("mse_unclipped", 255, 20000.0, "linear"),
            ("twohot", 255, 10.0, "symexp"),
            ("twohot", 255, 20000.0, "linear"),
            ("twohot", 64, math.expm1(20), "symexp"),
            ("twohot", 254, 10.0, "linear"),
        ]:
            candidate = _agent(
                trainer,
                device,
                value_loss=value_loss,
                value_num_bins=bins,
                value_max_abs=extent,
                value_spacing=spacing,
            )
            _assert_parameters(candidate.actor, reference.actor, exact=True)
            _assert_parameters(candidate.critic[0], reference.critic[0], exact=True)
            torch.testing.assert_close(candidate.actor(observations), expected_policy, rtol=0, atol=0)
            features = candidate.critic[0](observations)
            torch.testing.assert_close(features, expected_features, rtol=0, atol=0)
            # Preserves the actual head-input calibration, not just parameters.
            torch.testing.assert_close(features.square().sum(-1), torch.ones(32, device=device), rtol=2e-5, atol=2e-5)
            if value_loss != "twohot":
                _assert_parameters(candidate.critic, reference.critic, exact=True)
                torch.testing.assert_close(candidate.get_value(observations), expected_values, rtol=0, atol=0)
            else:
                _, _, logits = candidate.get_policy_and_value_output(observations)
                assert logits.shape == (32, bins)
                centers = (
                    _dreamer_bins(bins, extent, device)
                    if spacing == "symexp"
                    else torch.linspace(-extent, extent, bins, device=device)
                )
                expected = _dreamer_decode(logits, centers)
                torch.testing.assert_close(candidate.get_value(observations).flatten(), expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "value_loss,clip_vloss,effective_clip",
    [("mse", True, True), ("mse", False, False), ("mse_unclipped", True, False)],
    ids=["clipped-mse", "unclipped-mse", "forced-unclipped-mse"],
)
def test_mse_joint_adam_steps_match_selected_frozen_base(device, value_loss, clip_vloss, effective_clip):
    reference = _agent(baseline, device)
    candidate = _agent(trainer, device, value_loss=value_loss)
    common = dict(
        **SELECTED_TRUNK,
        norm_adv=False,
        learning_rate=0.0096,
        max_grad_norm=0.5,
        ent_coef=0.0,
        vf_coef=0.5,
        clip_coef=0.2,
    )
    expected_args = baseline.Args(**common, clip_vloss=effective_clip)
    actual_args = trainer.Args(**common, value_loss=value_loss, clip_vloss=clip_vloss)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.0096, eps=1e-5, fused=True)
    candidate_optimizer = torch.optim.Adam(candidate.parameters(), lr=0.0096, eps=1e-5, fused=True)
    observations = _observations(256, device)
    native, logprobs, advantages, initial_values = _policy_inputs(reference, observations)
    # Exercise actual global clipping, rather than merely comparing two
    # implementations when their norm remains below the clipping threshold.
    advantages = advantages * 10.0
    direction = torch.where(torch.arange(256, device=device) % 2 == 0, 1.0, -1.0)
    clip_active = torch.arange(256, device=device) % 4 < 2
    old_values = initial_values + torch.where(clip_active, direction, 0.0)
    returns = initial_values - 2.0 * direction
    saved_advantages = advantages.clone()
    # The fixture must activate value clipping, not accidentally make both
    # control branches identical and permit a silently unclipped MSE control.
    clipped = old_values + (initial_values - old_values).clamp(-0.2, 0.2)
    assert torch.all((clipped - returns).square()[clip_active] > (initial_values - returns).square()[clip_active])
    torch.testing.assert_close(clipped[~clip_active], initial_values[~clip_active], rtol=0, atol=0)
    for epoch in range(4):
        indices = torch.arange(256, device=device).roll(31 * epoch)
        batch = tuple(x[indices] for x in (observations, native, logprobs, advantages, returns, old_values))
        reference_optimizer.zero_grad(set_to_none=True)
        candidate_optimizer.zero_grad(set_to_none=True)
        expected, expected_metrics = baseline.ppo_loss(reference, *batch, expected_args)
        actual, actual_metrics = trainer.ppo_loss(candidate, *batch, actual_args)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(actual_metrics, expected_metrics, rtol=2e-6, atol=2e-7)
        expected.backward()
        actual.backward()
        _assert_parameters(candidate.actor, reference.actor, gradients=True)
        _assert_parameters(candidate.critic, reference.critic, gradients=True)
        if epoch == 0:
            for network in (candidate.actor, candidate.critic):
                assert sum(p.grad.square().sum() for p in network.parameters()) > 0
        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.5, foreach=True)
        actual_norm = torch.nn.utils.clip_grad_norm_(candidate.parameters(), 0.5, foreach=True)
        torch.testing.assert_close(actual_norm, expected_norm, rtol=2e-6, atol=2e-7)
        if epoch == 0:
            assert actual_norm > 0.5
        reference_optimizer.step()
        candidate_optimizer.step()
        _assert_parameters(candidate.actor, reference.actor)
        _assert_parameters(candidate.critic, reference.critic)
    torch.testing.assert_close(advantages, saved_advantages, rtol=0, atol=0)
    assert actual_args.clip_vloss is clip_vloss


@pytest.mark.parametrize("num_bins,extent", [(255, 10.0), (254, 2500.0)], ids=["normalized-odd", "raw-even"])
def test_twohot_joint_loss_has_dreamer_ce_and_unchanged_ppo_gradients(device, num_bins, extent):
    candidate = _agent(trainer, device, value_num_bins=num_bins, value_max_abs=extent)
    reference = _agent(baseline, device)
    observations = _observations(96, device)
    native, logprobs, advantages, old_values = _policy_inputs(reference, observations)
    returns = torch.linspace(-1.2 * extent, 1.2 * extent, 96, device=device, requires_grad=True)
    bins = _dreamer_bins(num_bins, extent, device)
    oracle_labels = _dreamer_labels(returns, bins)
    labels = candidate.value_support.project(returns)
    assert labels.dtype == torch.float32 and not labels.requires_grad
    torch.testing.assert_close(labels, oracle_labels, rtol=2e-5, atol=2e-5)
    # Nonzero readout makes this defend trunk as well as head gradients even
    # when the categorical head's canonical initialization is all zeros.
    with torch.no_grad():
        candidate.critic[-1].weight.uniform_(-0.05, 0.05)
        candidate.critic[-1].bias.uniform_(-0.02, 0.02)
    common = dict(**SELECTED_TRUNK, norm_adv=False, ent_coef=0.0, vf_coef=0.5)
    args = trainer.Args(**common, value_num_bins=num_bins, value_max_abs=extent)
    base_args = baseline.Args(**common)
    expected_actor_loss, expected_metrics = baseline.ppo_loss(
        reference,
        observations,
        native,
        logprobs,
        advantages,
        returns.detach(),
        old_values,
        base_args,
    )
    expected_actor_gradients = torch.autograd.grad(expected_actor_loss, tuple(reference.actor.parameters()))
    logits = candidate.critic(observations)
    log_pred = logits.float() - torch.logsumexp(logits.float(), -1, keepdim=True)
    expected_ce = -(oracle_labels * log_pred).sum(-1).mean()
    expected_critic_gradients = torch.autograd.grad(args.vf_coef * expected_ce, tuple(candidate.critic.parameters()))
    differentiable_labels = labels.clone().requires_grad_()
    actual, metrics = trainer.ppo_loss(
        candidate,
        observations,
        native,
        logprobs,
        advantages,
        differentiable_labels,
        old_values,
        args,
    )
    torch.testing.assert_close(metrics[1], expected_ce.detach(), rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(metrics[[0, 2, 3, 4, 5]], expected_metrics[[0, 2, 3, 4, 5]], rtol=0, atol=0)
    torch.testing.assert_close(actual, expected_metrics[0] + args.vf_coef * expected_ce.detach(), rtol=2e-6, atol=2e-6)
    actual.backward()
    assert differentiable_labels.grad is None and returns.grad is None
    for parameter, expected in zip(candidate.actor.parameters(), expected_actor_gradients):
        torch.testing.assert_close(parameter.grad, expected, rtol=2e-6, atol=2e-7)
    for parameter, expected in zip(candidate.critic.parameters(), expected_critic_gradients):
        torch.testing.assert_close(parameter.grad, expected, rtol=2e-5, atol=2e-7)
    with torch.no_grad():
        expected_values = _dreamer_decode(candidate.critic(observations), bins)
        torch.testing.assert_close(candidate.get_value(observations).flatten(), expected_values, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "reward_norm,extent,spacing",
    [(True, 10.0, "symexp"), (True, 10.0, "linear"), (False, 20000.0, "symexp")],
    ids=["normalized-symexp", "normalized-linear", "raw-symexp"],
)
def test_compiled_full_batch_joint_updates_keep_labels_and_refresh_host_policy(device, reward_norm, extent, spacing):
    candidate = _agent(trainer, device, value_num_bins=255, value_max_abs=extent, value_spacing=spacing)
    args = trainer.Args(
        **SELECTED_TRUNK,
        value_num_bins=255,
        value_max_abs=extent,
        value_spacing=spacing,
        reward_norm=reward_norm,
        norm_adv=False,
        learning_rate=0.0096,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=0.5,
        update_epochs=10,
    )
    prototypes = _observations(2, device)
    observations = prototypes.repeat(16384, 1)
    returns = observations.new_tensor([-0.3 * extent, 0.3 * extent]).repeat(16384)
    labels = candidate.value_support.project(returns)
    saved_labels, saved_returns = labels.clone(), returns.clone()
    native, logprobs, advantages, old_values = _policy_inputs(candidate, observations)
    saved_advantages = advantages.clone()
    optimizer = torch.optim.Adam(candidate.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
    compiled_loss = torch.compile(
        lambda obs, actions, old_logp, adv, targets, values: trainer.ppo_loss(
            candidate,
            obs,
            actions,
            old_logp,
            adv,
            targets,
            values,
            args,
        ),
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
    )
    host_actor = baseline.make_host_mirror(candidate.actor, 16)
    host_observations = observations[:16].cpu().numpy()
    with torch.no_grad():
        initial_policy = candidate.actor(prototypes).clone()
        initial_logits = candidate.critic(prototypes).clone()
        initial_ce = -(labels[:2] * initial_logits.log_softmax(-1)).sum(-1).mean()
    for _ in range(args.update_epochs):
        torch.compiler.cudagraph_mark_step_begin()
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = compiled_loss(observations, native, logprobs, advantages, labels, old_values)
        assert torch.isfinite(loss) and torch.isfinite(metrics).all()
        loss.backward()
        for network in (candidate.actor, candidate.critic):
            gradients = [p.grad for p in network.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in gradients)
            assert sum(g.square().sum() for g in gradients) > 0
        norm = torch.nn.utils.clip_grad_norm_(candidate.parameters(), args.max_grad_norm, foreach=True)
        assert torch.isfinite(norm)
        optimizer.step()
        del loss, metrics, norm
        host_actor.refresh()
        with torch.no_grad():
            expected_policy = candidate.actor(observations[:16]).cpu().numpy()
        np.testing.assert_allclose(host_actor(host_observations), expected_policy, rtol=2e-4, atol=2e-5)
    with torch.no_grad():
        final_logits = candidate.critic(prototypes)
        final_ce = -(labels[:2] * final_logits.log_softmax(-1)).sum(-1).mean()
        assert final_ce < initial_ce
        assert not torch.equal(candidate.actor(prototypes), initial_policy)
        values = candidate.get_value(observations).flatten()
        assert torch.isfinite(values).all()
        assert values.abs().max() <= extent * (1 + 2e-6)
        assert not torch.equal(values[0], values[1])
        centers = (
            _dreamer_bins(255, extent, device)
            if spacing == "symexp"
            else torch.linspace(-extent, extent, 255, device=device)
        )
        expected_values = _dreamer_decode(final_logits, centers)
        torch.testing.assert_close(values[:2], expected_values, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(labels, saved_labels, rtol=0, atol=0)
    torch.testing.assert_close(returns, saved_returns, rtol=0, atol=0)
    torch.testing.assert_close(advantages, saved_advantages, rtol=0, atol=0)
