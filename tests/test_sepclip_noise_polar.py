"""CUDA contracts for trajectory-noise filtering; execute through mlq."""
import copy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.ppo_continuous_action_sepclip_noise_polar_v6 import (
    Agent, Args, MatrixOptimizer, environment_folds, fold_gradients,
    matrix_update, noise_covariance, ppo_loss, soft_polar,
)


@pytest.mark.parametrize('shape', [(43, 17), (43, 64), (64, 43)])
def test_orthogonal_spectral_noise_curve_and_zero_noise(shape):
    torch.manual_seed(1)
    rank = min(shape)
    left, _ = torch.linalg.qr(torch.randn(shape[0], rank, device='cuda'))
    right, _ = torch.linalg.qr(torch.randn(shape[1], rank, device='cuda'))
    singular = torch.linspace(.1, 1., rank, device='cuda')
    matrix = (left * singular) @ right.T
    zero_covariance = torch.zeros(rank, rank, device='cuda')
    torch.testing.assert_close(soft_polar(matrix, zero_covariance),
                               left @ right.T, atol=2e-5, rtol=2e-5)
    noise_modes = torch.linspace(.01, .2, rank, device='cuda')
    noise = (left * noise_modes) @ right.T
    covariance = noise_covariance(noise)
    expected = (left * (singular / (singular.square() + noise_modes.square()).sqrt())) @ right.T
    actual = soft_polar(matrix, covariance)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert soft_polar(matrix, covariance * 4).norm() < actual.norm()
    torch.testing.assert_close(soft_polar(matrix, covariance, 0.), left @ right.T, atol=2e-5, rtol=2e-5)
    assert torch.count_nonzero(soft_polar(torch.zeros_like(matrix), covariance)) == 0
    assert torch.count_nonzero(soft_polar(torch.zeros_like(matrix), zero_covariance)) == 0


def test_correlated_noise_alignment_and_cold_start_scale_cancellation():
    gradient = torch.eye(2, device='cuda')[None]
    noise = torch.diag(torch.tensor([4., 0.], device='cuda'))[None]
    expected = torch.diag(torch.tensor([1 / 17**.5, 1.], device='cuda'))[None]
    torch.testing.assert_close(soft_polar(gradient, noise_covariance(noise)), expected)
    for momentum in (.85, .95):
        parameter, velocity = torch.zeros_like(gradient), torch.zeros_like(gradient)
        second = torch.zeros(1, 2, 1, device='cuda')
        moment = torch.zeros(1, 2, 2, device='cuda')
        prior = torch.zeros_like(moment)
        coefficients = torch.tensor([momentum**2, (1-momentum)**2,
                                     momentum**4, (1-momentum**2)**2], device='cuda')
        matrix_update(parameter, velocity, second, moment, prior, gradient, noise, coefficients,
                      torch.tensor(1., device='cuda'), momentum, .9, 1., 1., False)
        torch.testing.assert_close(-parameter, expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(moment, noise_covariance(noise) * (1-momentum)**2)
        assert torch.count_nonzero(second) == 0


@pytest.mark.parametrize('lane_equalize', [False, True])
def test_equalizer_preserves_soft_norm_and_inputs(lane_equalize):
    torch.manual_seed(1)
    gradient = torch.randn(2, 43, 64, device='cuda')
    noise = torch.randn_like(gradient) * .5
    original_gradient, original_noise = gradient.clone(), noise.clone()
    velocity = torch.zeros_like(gradient)
    second = torch.zeros(2, 1, 64, device='cuda')
    parameter = torch.zeros_like(gradient)
    prior, moment = [torch.zeros(2, 43, 43, device='cuda') for _ in range(2)]
    lr = torch.tensor(.005, device='cuda')
    beta = .85
    coefficients = torch.tensor([beta**2, (1-beta)**2, beta**4, (1-beta**2)**2], device='cuda')
    expected_polar = soft_polar(gradient * (1-beta**2), noise_covariance(noise) * (1-beta**2)**2)
    metrics = matrix_update(parameter, velocity, second, moment, prior, gradient, noise, coefficients,
                            lr, beta, .9, 1., 1., lane_equalize)
    torch.testing.assert_close(metrics[:, 1], expected_polar.norm(dim=(-2, -1)))
    torch.testing.assert_close(metrics[:, 2], metrics[:, 1])
    torch.testing.assert_close(parameter.norm(dim=(-2, -1)) / lr, metrics[:, 1])
    torch.testing.assert_close(gradient, original_gradient, atol=0, rtol=0)
    torch.testing.assert_close(noise, original_noise, atol=0, rtol=0)


def test_environment_folds_preserve_trajectories_and_policy_rng():
    generator = torch.Generator(device='cuda').manual_seed(17042)
    policy_rng = torch.cuda.get_rng_state().clone()
    first, second = environment_folds(7, 16, generator, 'cuda')
    assert first.numel() == second.numel() == 56
    assert torch.unique(torch.cat((first, second))).numel() == 112
    env_first, env_second = first.remainder(16), second.remainder(16)
    assert torch.unique(env_first).numel() == torch.unique(env_second).numel() == 8
    assert not torch.isin(env_first, env_second).any()
    torch.testing.assert_close(torch.cuda.get_rng_state(), policy_rng, atol=0, rtol=0)


def test_fold_gradient_mean_matches_original_ppo_and_noise_algebra():
    torch.manual_seed(1)
    envs = SimpleNamespace(single_action_space=gym.spaces.Box(-1., 1., (6,), dtype=np.float32),
                           single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32))
    agent = Agent(envs).cuda()
    args = Args()
    obs = torch.randn(64, 17, device='cuda')
    native = .2 + .6 * torch.rand(64, 6, device='cuda')
    with torch.no_grad():
        alpha, beta, values = agent.get_policy_and_value(obs)
        old = agent.action_logprob(alpha, beta, native)
    batch = (obs, native, old, torch.randn(64, device='cuda'),
             torch.randn(64, device='cuda'), values.flatten())
    loss_fn = lambda *fields: ppo_loss(agent, *fields, args)
    parameters = tuple(agent.parameters())
    loss, full_metrics = loss_fn(*batch)
    loss.backward()
    expected = [parameter.grad.clone() for parameter in parameters]
    folds = environment_folds(4, 16, torch.Generator(device='cuda').manual_seed(17042), 'cuda')
    individual = []
    for fold in folds:
        agent.zero_grad(set_to_none=True)
        loss_fn(*(field[fold] for field in batch))[0].backward()
        individual.append([parameter.grad.clone() for parameter in parameters])
    metrics, noise = fold_gradients(loss_fn, batch, folds, parameters,
                                     {id(p) for p in parameters if p.ndim == 2})
    torch.testing.assert_close(metrics, full_metrics, atol=2e-6, rtol=2e-5)
    for parameter, gradient, a, b in zip(parameters, expected, *individual):
        torch.testing.assert_close(parameter.grad, gradient, atol=3e-6, rtol=3e-5)
        if parameter.ndim == 2:
            torch.testing.assert_close(noise[parameter], (a - b) * .5, atol=0, rtol=0)
            torch.testing.assert_close(parameter.grad + noise[parameter], a, atol=3e-6, rtol=3e-5)


def test_compiled_checkpoint_continuation_preserves_banks_and_ownership():
    torch.manual_seed(1)
    parameters = [torch.nn.Parameter(torch.randn(*shape, device='cuda'))
                  for shape in [(43, 17), (43, 17), (43, 64), (64, 43)]]
    optimizer = MatrixOptimizer(parameters)
    optimizer.begin_rollout()
    for _ in range(3):
        noises = {}
        for parameter in parameters:
            parameter.grad = torch.randn_like(parameter)
            noises[parameter] = torch.randn_like(parameter) * .05
        optimizer.step(noises)
    clones = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    other = MatrixOptimizer(clones)
    other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for lr in (.0025, .001, .0005):
        if lr == .0005:
            optimizer.begin_rollout()
            other.begin_rollout()
        optimizer.set_lr(lr)
        other.set_lr(lr)
        noises, clone_noises, original_gradients, original_noises = {}, {}, [], []
        for parameter, clone in zip(parameters, clones):
            gradient = torch.randn_like(parameter)
            noise = torch.randn_like(parameter) * .05
            parameter.grad, clone.grad = gradient.clone(), gradient.clone()
            noises[parameter], clone_noises[clone] = noise.clone(), noise.clone()
            original_gradients.append(gradient)
            original_noises.append(noise)
        optimizer.step(noises)
        other.step(clone_noises)
        for parameter, clone, gradient, noise in zip(parameters, clones, original_gradients, original_noises):
            torch.testing.assert_close(parameter, clone, atol=0, rtol=0)
            torch.testing.assert_close(parameter.grad, gradient, atol=0, rtol=0)
            torch.testing.assert_close(noises[parameter], noise, atol=0, rtol=0)
        for old, new in zip(optimizer.stacks, other.stacks):
            assert new[2] is other.state[new[1][0]]
            for key in ('velocity', 'second', 'covariance_prior', 'covariance_moment'):
                torch.testing.assert_close(old[2][key], new[2][key], atol=0, rtol=0)


def test_rollout_covariance_clock_and_repeated_noise():
    beta = .85
    parameter = torch.nn.Parameter(torch.ones(4, 6, device='cuda'))
    optimizer = MatrixOptimizer([parameter], momentum=beta)
    noise = torch.arange(24, device='cuda', dtype=torch.float32).reshape(4, 6) / 100
    covariance = noise_covariance(noise)[None]
    optimizer.begin_rollout()
    for k in (1, 2, 3):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step({parameter: noise})
        torch.testing.assert_close(optimizer.state[parameter]['covariance_moment'],
                                   covariance * (1-beta**k)**2)
    previous = optimizer.state[parameter]['covariance_moment'].clone()
    optimizer.begin_rollout()
    torch.testing.assert_close(optimizer.state[parameter]['covariance_prior'], previous)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step({parameter: noise})
    torch.testing.assert_close(optimizer.state[parameter]['covariance_moment'],
                               beta**2 * previous + (1-beta)**2 * covariance)
