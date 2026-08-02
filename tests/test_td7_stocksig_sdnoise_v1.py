import math

import torch
import torch.nn as nn

from cleanrl.td7_lesale_v1 import (
    Actor,
    ActorLossCore,
    Args,
    CriticLossCore,
    SDNoiseActor,
    sdnoise_alpha_loss,
)


class _IdentityEncoder(nn.Module):
    def zsa(self, zs, action):
        return torch.cat([zs[:, : action.size(1)], action], dim=1)


class _ActionCritic(nn.Module):
    def forward(self, state, action, zsa, zs):
        value = action.sum(dim=1, keepdim=True) + 0.1 * zsa.sum(dim=1, keepdim=True)
        return torch.cat([value, value], dim=1)


class _ZeroEncoder(nn.Module):
    def __init__(self, zs_dim):
        super().__init__()
        self.zs_dim = zs_dim

    def zs(self, state):
        return state.new_zeros((state.size(0), self.zs_dim))

    def zsa(self, zs, action):
        return torch.zeros_like(zs)


class _ZeroCritic(nn.Module):
    def forward(self, state, action, zsa, zs):
        return state.new_zeros((state.size(0), 2))


def test_sdnoise_head_does_not_perturb_stocksig_initialization_stream():
    torch.manual_seed(123)
    baseline = Actor(5, 2, zs_dim=7, hdim=11)
    expected_next_random = torch.randn(8)

    torch.manual_seed(123)
    sdnoise = SDNoiseActor(5, 2, zs_dim=7, hdim=11, head_seed=456)
    actual_next_random = torch.randn(8)

    for name, value in baseline.state_dict().items():
        assert torch.equal(value, sdnoise.state_dict()[name])
    assert torch.equal(actual_next_random, expected_next_random)


def test_sdnoise_additive_sample_and_scale_bounds():
    actor = SDNoiseActor(
        5,
        2,
        zs_dim=7,
        hdim=11,
        log_std_min=-5.0,
        log_std_max=0.0,
        head_seed=456,
    )
    state = torch.randn(6, 5)
    zs = torch.randn(6, 7)
    epsilon = torch.randn(6, 2)
    epsilon[0, 0] = 100.0

    mean, log_std = actor.policy_stats(state, zs)
    action, entropy_proxy = actor.sample_additive(state, zs, epsilon)

    assert torch.equal(actor(state, zs), mean)
    assert torch.allclose(action, (mean + log_std.exp() * epsilon).clamp(-1, 1))
    assert action[0, 0] == 1.0
    assert torch.allclose(entropy_proxy, log_std.sum(dim=1, keepdim=True))
    assert float(log_std.detach().min()) >= -5.0
    assert float(log_std.detach().max()) <= 0.0


def test_sdnoise_actor_loss_updates_mean_and_scale_without_sac_targets():
    torch.manual_seed(7)
    actor = SDNoiseActor(5, 2, zs_dim=7, hdim=11, head_seed=456)
    args = Args(sd_noise=True)
    core = ActorLossCore(actor, _ActionCritic(), _IdentityEncoder(), args)
    state = torch.randn(6, 5)
    zs = torch.randn(6, 7)
    epsilon = torch.randn(6, 2)
    alpha = torch.tensor(0.3)

    action, entropy_proxy = actor.sample_additive(state, zs, epsilon)
    zsa = core.fixed_encoder.zsa(zs, action)
    expected = -core.critic(state, action, zsa, zs).mean() - alpha * entropy_proxy.mean()
    loss, returned_entropy = core(state, zs, epsilon, alpha)

    assert torch.allclose(loss, expected)
    assert torch.allclose(returned_entropy, entropy_proxy.mean())
    loss.backward()
    assert actor.l3.weight.grad is not None
    assert actor.log_std_head.weight.grad is not None
    assert float(actor.l3.weight.grad.abs().sum()) > 0.0
    assert float(actor.log_std_head.weight.grad.abs().sum()) > 0.0


def test_sdnoise_temperature_moves_toward_target_scale():
    target = 2.0 * math.log(0.1)

    below_target_log_alpha = torch.zeros((), requires_grad=True)
    below_target_entropy = torch.tensor([[target - 1.0]])
    sdnoise_alpha_loss(
        below_target_log_alpha, below_target_entropy, target
    ).backward()
    assert float(below_target_log_alpha.grad) < 0.0

    above_target_log_alpha = torch.zeros((), requires_grad=True)
    above_target_entropy = torch.tensor([[target + 1.0]])
    sdnoise_alpha_loss(
        above_target_log_alpha, above_target_entropy, target
    ).backward()
    assert float(above_target_log_alpha.grad) > 0.0


def test_sdnoise_critic_target_uses_deterministic_mean_only():
    actor = SDNoiseActor(5, 2, zs_dim=7, hdim=11, head_seed=456)
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.l3.bias.fill_(torch.atanh(torch.tensor(0.25)))
        actor.log_std_head.bias.fill_(10.0)

    args = Args(sd_noise=True, gamma=0.99)
    encoder = _ZeroEncoder(zs_dim=7)
    core = CriticLossCore(
        _ZeroCritic(), _ActionCritic(), actor, encoder, encoder, args
    )
    batch_size = 4
    state = torch.zeros(batch_size, 5)
    action = torch.zeros(batch_size, 2)
    outputs = core(
        state,
        action,
        state,
        torch.zeros(batch_size, 1),
        torch.ones(batch_size, 1),
        torch.zeros_like(action),
        torch.tensor(1.0),
        torch.tensor(-100.0),
        torch.tensor(100.0),
    )
    q_target_min, q_target_max = outputs[2], outputs[3]

    expected = torch.tensor(0.99 * (0.25 + 0.25))
    assert torch.allclose(q_target_min, expected)
    assert torch.allclose(q_target_max, expected)
