"""Contracts of v65's arm pieces; small CPU networks, no environments."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cleanrl.reppo.ppo_continuous_action_reppo_arms_v65 import (
    Actor, Args, Critic, action_slope, clip_slope, ema_decay, init_uniform_sigma, unit_slope,
    upper_order_statistic, validate_args, withheld_gradients,
)

OBSERVATION_DIM, ACTION_DIM = 17, 6


def spiked_slopes(rows=200, spikes=5):
    generator = torch.Generator().manual_seed(0)
    slope = torch.randn((rows, ACTION_DIM), generator=generator, dtype=torch.float64)
    slope[:spikes] *= 50.0
    return slope


def small_args():
    return Args(actor_hidden_dim=32, critic_hidden_dim=32, num_bins=31)


def test_upper_order_statistic_matches_a_sort():
    values = torch.randperm(200, generator=torch.Generator().manual_seed(0)).double()
    for quantile, expected in ((0.99, 197.0), (0.5, 99.0), (1.0, 199.0)):
        assert upper_order_statistic(values, quantile).item() == expected
    assert upper_order_statistic(torch.tensor([2.0, 2.0, 2.0, 7.0]), 0.75).item() == 2.0


def test_clip_caps_only_the_rows_above_the_percentile():
    slope = spiked_slopes()
    norm = slope.norm(dim=-1)
    limit = norm.sort().values[197]  # ceil(0.99 * 200) entries at or below it
    clipped = clip_slope(slope)
    torch.testing.assert_close(clipped.norm(dim=-1), norm.clamp_max(limit))
    torch.testing.assert_close(F.cosine_similarity(clipped, slope, dim=-1), torch.ones(200, dtype=torch.float64))
    unchanged = norm <= limit
    assert unchanged.sum() == 198
    assert torch.equal(clipped[unchanged], slope[unchanged])


def test_unit_equalizes_amplitude_and_keeps_direction_and_energy():
    slope = spiked_slopes()
    rms = slope.norm(dim=-1).square().mean().sqrt()
    conditioned = unit_slope(slope)
    torch.testing.assert_close(conditioned.norm(dim=-1), rms.expand(200))
    torch.testing.assert_close(conditioned.norm(dim=-1).square().mean().sqrt(), rms)
    torch.testing.assert_close(F.cosine_similarity(conditioned, slope, dim=-1), torch.ones(200, dtype=torch.float64))
    # The spikes no longer own the field: every row holds the same share of the energy.
    energy = conditioned.square().sum(-1)
    torch.testing.assert_close(energy / energy.sum(), torch.full((200,), 1 / 200, dtype=torch.float64))


def test_ema_decay_spans_the_horizon():
    steps = 128
    for horizon in (0.5, 4.0, 8.0):
        decay = ema_decay(horizon, steps)
        assert 1.0 / (1.0 - decay) == pytest.approx(horizon * steps)
    # After one horizon of steps toward a constant, 1 / e of the start is left.
    average, critic, weight = torch.zeros(()), torch.ones(()), 1.0 - ema_decay(4.0, steps)
    for _ in range(4 * steps):
        average.lerp_(critic, weight)
    assert 1.0 - average.item() == pytest.approx(np.exp(-1.0), rel=2e-3)
    assert ema_decay(1.0 / steps, steps) == 0.0  # a one-step horizon tracks the critic exactly


@pytest.mark.parametrize("sigma", [0.55, 1.0])
def test_init_sigma_gives_a_uniform_quiet_start(sigma):
    torch.manual_seed(0)
    actor = Actor(OBSERVATION_DIM, ACTION_DIM, small_args())
    trunk = [parameter.clone() for parameter in actor.net[:-1].parameters()]
    default_stream = torch.get_rng_state()
    init_uniform_sigma(actor.net[-1], sigma, seed=1)
    assert torch.equal(torch.get_rng_state(), default_stream)  # the run's init stream is not read
    for before, after in zip(trunk, actor.net[:-1].parameters()):
        assert torch.equal(before, after)
    observations = torch.randn((4096, OBSERVATION_DIM), generator=torch.Generator().manual_seed(3)).clamp(-10, 10)
    mean, std = actor(observations)
    torch.testing.assert_close(std, torch.full_like(std, sigma), rtol=0.05, atol=0.0)
    torch.testing.assert_close(std.mean(), torch.tensor(sigma), rtol=5e-3, atol=0.0)
    assert mean.abs().max() < 0.05
    # Same seed, same head.
    other = Actor(OBSERVATION_DIM, ACTION_DIM, small_args())
    init_uniform_sigma(other.net[-1], sigma, seed=1)
    assert torch.equal(other.net[-1].weight, actor.net[-1].weight)


def test_default_init_is_not_uniform():
    torch.manual_seed(0)
    _, std = Actor(OBSERVATION_DIM, ACTION_DIM, small_args())(torch.randn(4096, OBSERVATION_DIM))
    assert std.max() / std.min() > 1.5  # what --init-sigma removes


def test_action_slope_is_the_critics_action_gradient():
    torch.manual_seed(0)
    critic = Critic(OBSERVATION_DIM, ACTION_DIM, small_args(), torch.device("cpu"))
    observations = torch.randn(64, OBSERVATION_DIM)
    actions = torch.tanh(torch.randn(64, ACTION_DIM))
    slope, q = action_slope(critic, observations, actions)
    leaf = actions.clone().requires_grad_(True)
    value = critic.value(observations, leaf)
    (expected,) = torch.autograd.grad(value.sum(), leaf)
    torch.testing.assert_close(slope, expected)
    torch.testing.assert_close(q, value.detach())
    assert not slope.requires_grad and not q.requires_grad


def test_unconditioned_surrogate_has_the_pathwise_gradient():
    torch.manual_seed(0)
    args = small_args()
    actor = Actor(OBSERVATION_DIM, ACTION_DIM, args)
    critic = Critic(OBSERVATION_DIM, ACTION_DIM, args, torch.device("cpu"))
    observations = torch.randn(64, OBSERVATION_DIM)
    noise = torch.randn(64, ACTION_DIM)
    parameters = list(actor.net.parameters())

    def sampled_actions():
        mean, std = actor(observations)
        return torch.tanh(mean + std * noise)

    actions = sampled_actions()
    pathwise = torch.autograd.grad(-critic.value(observations, actions).mean(), parameters)
    actions = sampled_actions()
    slope, _ = action_slope(critic, observations, actions.detach())
    surrogate = torch.autograd.grad(-(actions * slope).sum(-1).mean(), parameters)
    for first, second in zip(pathwise, surrogate):
        torch.testing.assert_close(first, second, rtol=1e-4, atol=1e-7)


def test_conditioners_leave_a_zero_slope_row_at_zero():
    slope = spiked_slopes()
    slope[7] = 0.0
    for conditioner in (clip_slope, unit_slope):
        conditioned = conditioner(slope)
        assert torch.isfinite(conditioned).all()
        assert not conditioned[7].any()


def critic_with_optimizer():
    torch.manual_seed(0)
    critic = Critic(OBSERVATION_DIM, ACTION_DIM, small_args(), torch.device("cpu"))
    optimizer = torch.optim.AdamW(critic.parameters(), lr=3e-4, fused=True)
    observations, actions = torch.randn(64, OBSERVATION_DIM), torch.tanh(torch.randn(64, ACTION_DIM))

    def full_loss():
        features = critic.features(observations, actions)
        return (critic.logits(features).square().mean()
                + critic.reward(critic.transition(features, actions)).square().mean())

    def categorical_loss():
        return critic.logits(critic.features(observations, actions)).square().mean()

    aux_only = [*critic.dynamics.parameters(), *critic.reward_head.parameters()]
    return critic, optimizer, full_loss, categorical_loss, aux_only


def optimizer_state(optimizer, parameters):
    return [{key: value.clone() for key, value in optimizer.state[parameter].items()
             if isinstance(value, torch.Tensor)} for parameter in parameters]


def test_categorical_loss_misses_exactly_the_aux_only_parameters():
    critic, _, _, categorical_loss, aux_only = critic_with_optimizer()
    categorical_loss().backward()
    unreached = {id(parameter) for parameter in critic.parameters() if parameter.grad is None}
    assert unreached == {id(parameter) for parameter in aux_only}


def test_withheld_gradients_freeze_parameters_and_adam_state():
    critic, optimizer, full_loss, categorical_loss, aux_only = critic_with_optimizer()
    full_loss().backward()
    optimizer.step()  # every parameter now has momentum
    shared = [parameter for parameter in critic.parameters() if all(parameter is not other for other in aux_only)]
    before = [parameter.detach().clone() for parameter in aux_only]
    shared_before = [parameter.detach().clone() for parameter in shared]
    state_before = optimizer_state(optimizer, aux_only)
    gradients = [parameter.grad for parameter in aux_only]
    optimizer.zero_grad(set_to_none=False)
    categorical_loss().backward()
    with withheld_gradients(aux_only):
        assert all(parameter.grad is None for parameter in aux_only)
        optimizer.step()
    for parameter, saved, saved_state, gradient in zip(aux_only, before, state_before, gradients):
        assert torch.equal(parameter, saved)
        for key, value in optimizer_state(optimizer, [parameter])[0].items():
            assert torch.equal(value, saved_state[key]), key
        assert parameter.grad is gradient  # the very tensor a captured graph writes into
    assert all(not torch.equal(parameter, saved) for parameter, saved in zip(shared, shared_before))
    # The next full step accumulates into that same storage.
    pointers = [parameter.grad.data_ptr() for parameter in aux_only]
    optimizer.zero_grad(set_to_none=False)
    full_loss().backward()
    assert pointers == [parameter.grad.data_ptr() for parameter in aux_only]
    assert all(parameter.grad.any() for parameter in aux_only if parameter.ndim > 1)


def test_a_zero_gradient_step_would_have_moved_them():
    # What withholding prevents: AdamW coasts on momentum (and decays both moments) under a zero gradient.
    critic, optimizer, full_loss, categorical_loss, aux_only = critic_with_optimizer()
    full_loss().backward()
    optimizer.step()
    before = [parameter.detach().clone() for parameter in aux_only]
    optimizer.zero_grad(set_to_none=False)
    categorical_loss().backward()
    optimizer.step()
    assert any(not torch.equal(parameter, saved) for parameter, saved in zip(aux_only, before))


def test_arm_flags_are_validated():
    assert validate_args(Args()).batch_size == 16 * 1024
    for bad in (dict(dqda_norm="rms"), dict(actor_critic="target"), dict(aux_every=0), dict(reward_latents=9),
                dict(init_sigma=-0.1), dict(init_sigma=0.55, resume="checkpoint.pt"),
                dict(actor_critic_horizon=0.0), dict(rng_stream=-1)):
        with pytest.raises(ValueError):
            validate_args(Args(**bad))
