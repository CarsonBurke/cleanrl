"""CUDA algebra/optimization contracts, not fixed-policy or held-out ML gates."""
import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cleanrl.ppo_continuous_action_return_field_v8 import (
    ActorTrustRegion, ReturnField, TEMPORAL_LOWER, beta_geometry, beta_kl_reference,
    complete_episode_layout, leave_environment_out_rewards, policy_gain_kl,
    temporal_returns, unwhiten,
)
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(autouse=True)
def cuda_runtime():
    assert torch.cuda.is_available(), 'Run through mlq on CUDA'
    configure_runtime(cudnn_deterministic=True, matmul_precision='highest', allow_tf32=False)
    torch.manual_seed(1)


@pytest.mark.parametrize('gamma', [1., .99])
def test_temporal_bands_match_actual_rewards_and_telescope(gamma):
    rewards = torch.randn(3, 1000, device='cuda')
    fn = torch.compile(temporal_returns, fullgraph=True, dynamic=True, options={'triton.cudagraphs': False})
    actual = fn(rewards, gamma)
    assert actual.shape == (3, 1000, len(TEMPORAL_LOWER))
    upper = TEMPORAL_LOWER[1:] + (1000,)
    for start in (0, 33, 511, 997, 999):
        for band, (lo, hi) in enumerate(zip(TEMPORAL_LOWER, upper)):
            offsets = torch.arange(lo, min(hi, 1000 - start), device='cuda') if lo < 1000 - start else torch.empty(0, device='cuda', dtype=torch.long)
            expected = (rewards[:, start + offsets].double() * gamma ** offsets.double()).sum(-1)
            torch.testing.assert_close(actual[:, start, band].double(), expected, rtol=2e-6, atol=2e-6)
        offsets = torch.arange(1000 - start, device='cuda')
        total = (rewards[:, start + offsets].double() * gamma ** offsets.double()).sum(-1)
        torch.testing.assert_close(actual[:, start].double().sum(-1), total, rtol=3e-6, atol=3e-6)


def test_baseline_excludes_all_episodes_of_source_environment():
    rewards = torch.randn(4, 1000, device='cuda')
    environment = torch.tensor([0, 1, 0, 2], device='cuda')
    centered = leave_environment_out_rewards(rewards, environment, 3)
    baseline = rewards - centered
    torch.testing.assert_close(baseline[0], (rewards[1] + rewards[3]) / 2, rtol=1e-5, atol=1e-6)
    changed = rewards.clone()
    changed[[0, 2]] += 100 * torch.randn(2, 1000, device='cuda')
    changed_baseline = changed - leave_environment_out_rewards(changed, environment, 3)
    torch.testing.assert_close(changed_baseline[[0, 2]], baseline[[0, 2]], rtol=1e-4, atol=3e-5)


def field_inputs():
    observation = torch.randn(256, 5, device='cuda')
    parameters = torch.rand(256, 3, 2, device='cuda') + 1
    age = torch.arange(256, device='cuda')
    environment = age % 4
    return observation, parameters, age, environment


def test_popart_preserves_raw_tensor_and_rescales_head_moments():
    model = ReturnField(5, 3, 4, width=32).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, fused=True)
    inputs = field_inputs()
    model(*inputs).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = model(*inputs).detach()
    old_first = optimizer.state[model.head.weight]['exp_avg'].clone()
    old_second = optimizer.state[model.head.weight]['exp_avg_sq'].clone()
    old_scale = model.scale.clone()
    target = 300 + 1000 * torch.randn_like(before)
    model.update_statistics(target, torch.ones(256, device='cuda'), optimizer, .05)
    after = model(*inputs).detach()
    torch.testing.assert_close(after, before, rtol=.01, atol=1e-4)
    ratio = (old_scale / model.scale).flatten()[:, None]
    torch.testing.assert_close(optimizer.state[model.head.weight]['exp_avg'], old_first * ratio)
    torch.testing.assert_close(optimizer.state[model.head.weight]['exp_avg_sq'], old_second * ratio.square())
    before = after
    model.update_statistics(-target * 2, torch.ones(256, device='cuda'), optimizer, .05)
    torch.testing.assert_close(model(*inputs).detach(), before, rtol=.01, atol=1e-4)


def test_bf16_trunk_keeps_popart_readout_fp32():
    model = ReturnField(5, 3, 4, width=32).cuda()
    optimizer = torch.optim.Adam(model.parameters(), fused=True)
    inputs = field_inputs()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        before = model(*inputs).detach()
        normalized = model.normalized(*inputs)
    assert normalized.dtype == torch.float32
    model.update_statistics(300 + 1000 * torch.randn_like(before), torch.ones(256, device='cuda'), optimizer, .05)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        after = model(*inputs).detach()
    torch.testing.assert_close(after, before, rtol=.02, atol=1e-4)


def test_joint_kl_matches_distribution_and_is_zero_at_collection_policy():
    old_logits = torch.randn(512, 6, device='cuda')
    alpha, beta = (F.softplus(old_logits) + 1).chunk(2, -1)
    parameters = torch.stack((alpha, beta), -1)
    reference = beta_kl_reference(parameters)
    credit = torch.randn_like(parameters)
    weights = torch.rand(512, device='cuda')
    gain, kl = policy_gain_kl(old_logits, parameters, reference, credit, weights)
    assert gain == 0 and abs(float(kl)) < 1e-12
    proposed = old_logits + .2 * torch.randn_like(old_logits)
    a, b = (F.softplus(proposed) + 1).double().chunk(2, -1)
    _, actual = policy_gain_kl(proposed, parameters, reference, credit, weights)
    expected = torch.distributions.kl_divergence(torch.distributions.Beta(alpha.double(), beta.double()),
                                                torch.distributions.Beta(a, b)).sum(-1)
    torch.testing.assert_close(actual, (expected * weights).sum() / weights.sum(), rtol=1e-10, atol=1e-10)


def test_frozen_vector_gain_has_the_original_return_score_gradient():
    logits = torch.randn(512, 6, device='cuda', requires_grad=True)
    actions = torch.rand(512, 3, device='cuda').clamp(.01, .99)
    reward = torch.randn(512, device='cuda')
    z, factor, parameters = beta_geometry(logits.detach(), actions)
    natural_credit = unwhiten(z * reward[:, None, None], factor)
    gain, _ = policy_gain_kl(logits, parameters, beta_kl_reference(parameters), natural_credit, torch.ones(512, device='cuda'))
    actual, = torch.autograd.grad(gain, logits)
    alpha, beta = (F.softplus(logits) + 1).chunk(2, -1)
    expected, = torch.autograd.grad((torch.distributions.Beta(alpha, beta).log_prob(actions).sum(-1) * reward).mean(), logits)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-8)


def test_reward_cross_moments_distinguish_mean_and_concentration_credit():
    logits = torch.full((300000, 4), .6, device='cuda', dtype=torch.float64)
    alpha, beta = (F.softplus(logits) + 1).chunk(2, -1)
    actions = torch.distributions.Beta(alpha, beta).sample()
    z, factor, _ = beta_geometry(logits, actions)
    # Different actuators have different objectives: direction versus spread.
    reward = actions[:, 0] + 3 * (actions[:, 1] - .5).square()
    credit = ((reward - reward.mean())[:, None, None] * z).mean(0)
    score_gradient = unwhiten(credit, factor[0])
    assert score_gradient[0, 0] > 0 and score_gradient[0, 1] < 0
    assert score_gradient[1, 0] < 0 and score_gradient[1, 1] < 0
    assert torch.linalg.matrix_rank(credit) == 2
    independent = logits[0].detach().clone().requires_grad_(True)
    a, b = (F.softplus(independent) + 1).chunk(2, -1)
    expected_reward = a[0] / (a[0] + b[0]) + 3 * (a[1] * (a[1] + 1) / ((a[1] + b[1]) * (a[1] + b[1] + 1)) - a[1] / (a[1] + b[1]) + .25)
    expected, = torch.autograd.grad(expected_reward, independent)
    actual = torch.cat((score_gradient[:, 0], score_gradient[:, 1])) * independent.sigmoid()
    torch.testing.assert_close(actual, expected, rtol=.05, atol=.0004)


def test_temporal_cancellation_uses_one_joint_kl_budget():
    logits = torch.randn(64, 4, device='cuda')
    _, factor, parameters = beta_geometry(logits, torch.rand(64, 2, device='cuda').clamp(.01, .99))
    first = torch.randn_like(parameters)
    credit = torch.stack((first, -first), 1).sum(1)
    assert credit.square().sum() == 0
    proposed = logits + .1
    gain, kl = policy_gain_kl(proposed, parameters, beta_kl_reference(parameters), unwhiten(credit, factor), torch.ones(64, device='cuda'))
    assert gain == 0 and kl > 0


def test_complete_rejection_restores_actor_and_adam_state():
    actor = torch.nn.Linear(3, 4).cuda()
    optimizer = torch.optim.Adam(actor.parameters(), lr=.1, fused=True)
    actor(torch.randn(16, 3, device='cuda')).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    trust = ActorTrustRegion(actor, optimizer, .01, 3)
    trust.snapshot()
    before = copy.deepcopy(actor.state_dict())
    state = copy.deepcopy(optimizer.state_dict())
    actor(torch.randn(16, 3, device='cuda')).square().mean().backward()
    optimizer.step()
    gain, kl, fraction, _ = trust.accept(lambda: (torch.tensor(-1., device='cuda'), torch.tensor(.1, device='cuda')))
    assert fraction == 0
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for key, values in optimizer.state_dict()['state'].items():
        for name, value in values.items():
            torch.testing.assert_close(value, state['state'][key][name], rtol=0, atol=0)


def test_compiled_vector_field_and_actor_optimizer_paths():
    field = ReturnField(5, 3, 4, width=32).cuda()
    inputs = field_inputs()
    target = torch.randn(256, len(TEMPORAL_LOWER), 3, 2, device='cuda')
    optimizer = torch.optim.Adam(field.parameters(), lr=1e-3, fused=True)

    def loss(*inputs):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = field.normalized(*inputs)
        return (prediction - (target - field.mean) / field.scale).square().mean()

    compiled = torch.compile(loss, fullgraph=True, mode='reduce-overhead')
    field.update_statistics(target, torch.ones(256, device='cuda'), optimizer, .05)
    compiled(*inputs).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in field.parameters())
    optimizer.step()
    with torch.no_grad():
        credit = field(*inputs).sum(1).detach()
    actor = torch.nn.Linear(5, 6).cuda()
    logits = actor(inputs[0]).detach()
    z, factor, parameters = beta_geometry(logits, torch.rand(256, 3, device='cuda').clamp(.01, .99))
    reference = beta_kl_reference(parameters)
    natural = unwhiten(credit, factor)

    def policy_loss(observation, parameters, reference, natural):
        gain, kl = policy_gain_kl(actor(observation), parameters, reference, natural, torch.ones(256, device='cuda'))
        return kl - gain

    compiled_policy = torch.compile(policy_loss, fullgraph=True, mode='reduce-overhead')
    compiled_policy(inputs[0], parameters, reference, natural).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())


def test_complete_episode_layout_keeps_fixed_length_objective():
    ages = np.tile(np.arange(1000), 2)[:, None].repeat(2, axis=1)
    ids, count = complete_episode_layout(ages, np.zeros_like(ages, dtype=bool), ages == 999, 1000)
    assert count == 4 and (ids >= 0).all()


def test_field_cannot_assign_credit_after_episode_end():
    model = ReturnField(5, 3, 4, width=32).cuda()
    observation, parameters, age, environment = field_inputs()
    age = torch.arange(744, 1000, device='cuda')
    raw = model(observation, parameters, age, environment)
    support = model.support(age).expand_as(raw)
    assert (raw[~support] == 0).all()
    # The last physical action retains its immediate-reward field.
    assert support[-1, 0].all() and not support[-1, 1:].any()
    # A partly observed final band is NOT discounted by an arbitrary fraction.
    unmasked = model.normalized(observation, parameters, age, environment).float() * model.scale + model.mean
    torch.testing.assert_close(raw[support], unmasked[support])
