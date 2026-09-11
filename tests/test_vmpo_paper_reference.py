"""Numerical reference contracts; run CUDA cases through the machine-wide mlq."""

import math

import numpy as np
import pytest
import torch
from torch.distributions import Normal, kl_divergence

from cleanrl.shared.runtime import configure_runtime
from cleanrl.vmpo.ppo_continuous_action_vmpo_paper_v59 import (
    Agent,
    Args,
    GaussianSampler,
    gaussian_kls,
    gaussian_log_prob,
    replica_estep,
    replica_samples,
    validate_args,
    vmpo_loss,
)

pytestmark = pytest.mark.cuda


@pytest.fixture(autouse=True)
def cuda_runtime():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)


def test_gaussian_channels_match_distribution_distances_not_their_sum():
    old_mean = torch.tensor([[0.2, -0.4], [1.0, 2.0]], device="cuda", dtype=torch.float64)
    old_std = torch.tensor([[0.3, 0.8], [1.0, 0.2]], device="cuda", dtype=torch.float64)
    mean = old_mean + torch.tensor([[0.01, 0.3], [-0.1, 0.03]], device="cuda")
    std = old_std * torch.tensor([[1.0001, 1.2], [0.8, 0.9999]], device="cuda")
    mean_kl, covariance_kl, full_kl = gaussian_kls(old_mean, old_std, mean, std)
    old = Normal(old_mean, old_std)
    torch.testing.assert_close(mean_kl, kl_divergence(old, Normal(mean, old_std)).sum(-1))
    torch.testing.assert_close(covariance_kl, kl_divergence(old, Normal(old_mean, std)).sum(-1))
    torch.testing.assert_close(full_kl, kl_divergence(old, Normal(mean, std)).sum(-1))
    assert not torch.allclose(mean_kl + covariance_kl, full_kl)
    actions = mean + 0.7 * std
    torch.testing.assert_close(gaussian_log_prob(mean, std, actions), Normal(mean, std).log_prob(actions).sum(-1))


def test_replica_selection_keeps_trajectory_groups_and_one_shared_temperature():
    # Large differences between replicas must not exclude the lower-return
    # replica, and both timesteps of each environment belong to the same shard.
    advantages = torch.tensor([[0.0, 10.0, 100.0, 110.0], [1.0, 11.0, 101.0, 111.0]],
                              device="cuda", dtype=torch.float64)
    shards = replica_samples(advantages.flatten(), 2, 4, 2)
    torch.testing.assert_close(shards, torch.tensor([[0.0, 10.0, 1.0, 11.0],
                                                    [100.0, 110.0, 101.0, 111.0]], device="cuda", dtype=torch.float64))
    eta = torch.tensor(2.0, device="cuda", dtype=torch.float64, requires_grad=True)
    weights, eta_loss, kl, ess, selected_count = replica_estep(shards, eta, 0.01)
    eta_loss.backward()
    torch.testing.assert_close(eta.grad, 0.01 - kl.mean())
    torch.testing.assert_close(weights.sum(1), torch.ones(2, device="cuda", dtype=torch.float64))
    torch.testing.assert_close(weights[0], weights[1])
    assert not weights.requires_grad
    assert (weights[:, [0, 2]] == 0).all()
    assert (ess <= selected_count).all()
    shifted = replica_estep(shards + 1234.0, eta.detach(), 0.01)
    torch.testing.assert_close(shifted[0], weights)
    torch.testing.assert_close(shifted[2], kl)


def test_tied_elites_have_uniform_weights_and_temperature_moves_down():
    advantages = torch.ones((8, 312), device="cuda")
    eta = torch.tensor(1.0, device="cuda", requires_grad=True)
    weights, loss, kl, ess, selected = replica_estep(advantages, eta, 0.01)
    loss.backward()
    torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-6, rtol=0)
    torch.testing.assert_close(ess, selected.float(), atol=1e-3, rtol=0)
    torch.testing.assert_close(eta.grad, torch.tensor(0.01, device="cuda"), atol=1e-6, rtol=0)
    # Averaging replica losses gives weights / R, not eight times the pressure.
    torch.testing.assert_close((weights / 8).square().sum().reciprocal(),
                               torch.tensor(2496.0, device="cuda"), atol=0, rtol=1e-6)


def test_popart_rescaling_preserves_nonconstant_raw_value_predictions():
    agent = Agent(17, 6).cuda()
    with torch.no_grad():
        agent.value_head.weight.normal_(0, 0.1)
        agent.value_head.bias.fill_(0.7)
        observations = torch.randn(64, 17, device="cuda")
        original = agent.value(observations).clone()
        assert original.std() > 0.01
        for center, scale in [(100.0, 10.0), (-300.0, 150.0), (0.0, 0.001)]:
            targets = center + scale * torch.randn(64, device="cuda")
            normalized = agent.update_popart(targets, 0.1, 0.01, 1e6)
            torch.testing.assert_close(agent.value(observations), original, rtol=3e-4, atol=2e-5)
            torch.testing.assert_close(normalized * agent.popart_std + agent.popart_mean, targets,
                                       rtol=1e-4, atol=2e-5)


def test_raw_gaussian_samples_are_not_replaced_by_executed_clipped_actions():
    sampler = GaussianSampler(16, [-1.0, -1.0], [1.0, 1.0], 1)
    logits = np.zeros((16, 4), dtype=np.float32)
    logits[:, :2] = 10.0
    logits[:, 2:] = math.log(math.expm1(1.0 - 1e-6))
    raw, executed, mean, std = sampler(logits)
    assert np.all(raw > 1)
    np.testing.assert_array_equal(executed, np.ones_like(executed))
    mean_t, std_t, raw_t, executed_t = (torch.tensor(x, device="cuda") for x in (mean, std, raw, executed))
    raw_logp = gaussian_log_prob(mean_t, std_t, raw_t)
    clipped_logp = gaussian_log_prob(mean_t, std_t, executed_t)
    assert (raw_logp > clipped_logp + 20).all()


def test_compiled_joint_loss_replays_with_correct_dual_gradients():
    args = validate_args(Args(num_envs=8, num_steps=3, num_replicas=2))
    agent = Agent(17, 6).cuda()
    duals = torch.nn.Parameter(torch.ones(3, device="cuda"))
    observations = torch.randn(args.batch_size, 17, device="cuda")
    with torch.no_grad():
        mean, std = agent.policy(observations)
        old_mean = mean + 0.4
        old_std = std * 1.2
        actions = old_mean + old_std * torch.randn_like(old_mean)
    advantages = torch.linspace(-2, 2, args.batch_size, device="cuda")
    returns = torch.linspace(-1, 1, args.batch_size, device="cuda")

    def loss_fn(obs, act, old_m, old_s, adv, targets):
        return vmpo_loss(agent, duals, obs, act, old_m, old_s, adv, targets, args)

    compiled = torch.compile(loss_fn, mode="reduce-overhead", fullgraph=True)
    parameters = [*agent.parameters(), duals]
    for scale in (1.0, 2.0):
        inputs = (observations, actions, old_mean, old_std, advantages * scale, returns)
        loss, metrics = loss_fn(*inputs)
        eager_grads = torch.autograd.grad(loss, parameters)
        # KL violations must raise the corresponding multiplier under descent.
        assert eager_grads[-1][1] < 0 and eager_grads[-1][2] < 0
        torch.testing.assert_close(eager_grads[-1][0], args.epsilon_eta - metrics[6], atol=1e-6, rtol=1e-5)
        torch.compiler.cudagraph_mark_step_begin()
        compiled_loss, compiled_metrics = compiled(*inputs)
        compiled_grads = torch.autograd.grad(compiled_loss, parameters)
        torch.testing.assert_close(compiled_loss, loss, rtol=2e-5, atol=1e-5)
        torch.testing.assert_close(compiled_metrics, metrics, rtol=2e-5, atol=1e-5)
        for actual, expected in zip(compiled_grads, eager_grads):
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
