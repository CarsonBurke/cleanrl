"""CUDA numerical contracts, not learning benchmarks; execute only through mlq."""
from copy import deepcopy
from functools import partial
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action_predictive_gradient_owned_v6 as m


def _reference_state(gradients, mode="adaptive"):
    state = m.PredictiveGradientControl(gradients[0], mode=mode)
    for gradient in gradients:
        state.accumulate_reference(gradient)
    state.finish_reference()
    return state


def _agent(activation="relu", dtype=torch.float32):
    m.configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(713)
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32),
        single_action_space=gym.spaces.Box(
            np.array([-2., .5], dtype=np.float32),
            np.array([1., 4.], dtype=np.float32),
        ),
    )
    args = m.Args(activation=activation, norm_adv=True, clip_vloss=True, ent_coef=.01)
    return m.Agent(envs, args).to(device="cuda", dtype=dtype), args


def _components(agent):
    """Fixed equal-size components with distinct local advantage statistics."""
    dtype = next(agent.parameters()).dtype
    grid = torch.arange(24 * 5, device="cuda", dtype=dtype).reshape(24, 5)
    observations = .6 * torch.sin(.37 * grid) + .3 * torch.cos(.13 * grid)
    row = torch.arange(24, device="cuda", dtype=dtype)
    actions = torch.stack((.15 + .7 * (row % 7) / 6, .2 + .6 * (row % 5) / 4), dim=-1)
    with torch.no_grad():
        alpha, beta, value = agent.get_policy_and_value(observations)
        logprobs = agent.action_logprob(alpha, beta, actions)
        values = value.flatten()
        returns = values + .4 * torch.cos(.73 * row) + .2
    advantages = torch.sin(.91 * row) * (1 + row // 8) + .7 * (row // 8)
    batch = (observations, actions, logprobs, advantages, returns, values)
    return [tuple(value[start:start + 8].clone() for value in batch) for start in (0, 8, 16)]


def _direct_gradient(agent, batch, args):
    loss, metrics = m.ppo_loss(agent, *batch, args)
    gradient = torch.autograd.grad(loss, tuple(agent.parameters()))
    return torch.cat([value.flatten() for value in gradient]), metrics.detach().clone()


def test_reference_population_moments_match_independent_batch_computation():
    gradients = torch.tensor(
        [[1000001., -3., .25, 7.], [999998., 5., -.5, 7.],
         [1000004., 1., .75, 7.], [999997., -7., -.25, 7.],
         [1000002., 4., .125, 7.]],
        dtype=torch.float32, device="cuda",
    )
    state = _reference_state(gradients)
    # Independent two-pass population moments, including a constant coordinate
    # and an offset large enough to expose accumulation in the input dtype.
    expected_mean = gradients.double().mean(dim=0)
    expected_variance = (gradients.double() - expected_mean).square().mean(dim=0)
    torch.testing.assert_close(state.mean, expected_mean, rtol=0, atol=2e-10)
    torch.testing.assert_close(state.variance, expected_variance, rtol=2e-10, atol=2e-10)


def test_frozen_ppo_component_mean_matches_finite_objective_before_clipping_or_adam():
    reference, args = _agent(dtype=torch.float64)
    components = _components(reference)
    current = deepcopy(reference)
    with torch.no_grad():
        for index, parameter in enumerate(current.parameters()):
            direction = torch.arange(parameter.numel(), device="cuda", dtype=parameter.dtype)
            parameter.add_(.003 * torch.sin(direction + index).reshape_as(parameter))
    reference_loss = partial(m.ppo_loss, reference, args=args)
    current_loss = partial(m.ppo_loss, current, args=args)
    references = torch.stack([
        m.loss_and_flat_gradient(reference, reference_loss, batch, compiled=False)[0]
        for batch in components
    ])
    state = _reference_state(references)
    # An arbitrary coordinatewise coefficient fixed BEFORE evaluating the draw.
    # Preparing this history is not a claim that calibration finds these values.
    chosen = torch.linspace(0., 1., references.shape[1], device="cuda", dtype=torch.float64)
    state.pair_count.fill_(1)
    state.cross_sum.copy_((chosen - 1) * state.variance)
    coefficient = torch.where(state.variance > 0, chosen, 0.)
    raw, corrected = [], []
    for batch, reference_gradient in zip(components, references):
        gradient, _ = m.loss_and_flat_gradient(current, current_loss, batch, compiled=False)
        result, actual_coefficient = state.correct(gradient, reference_gradient)
        torch.testing.assert_close(actual_coefficient, coefficient, atol=2e-15, rtol=2e-15)
        torch.testing.assert_close(
            result, gradient - coefficient * (reference_gradient - references.mean(dim=0)),
            atol=2e-13, rtol=2e-12,
        )
        raw.append(gradient)
        corrected.append(result)
    # Average the component LOSSES, not a concatenated batch loss: PPO's
    # advantage normalization is component-local and is part of the objective.
    objective = torch.stack([m.ppo_loss(current, *batch, args)[0] for batch in components]).mean()
    objective_gradient = torch.cat([
        gradient.flatten() for gradient in torch.autograd.grad(objective, tuple(current.parameters()))
    ])
    torch.testing.assert_close(torch.stack(raw).mean(0), objective_gradient, atol=2e-12, rtol=2e-10)
    torch.testing.assert_close(torch.stack(corrected).mean(0), objective_gradient, atol=2e-12, rtol=2e-10)


def test_same_component_snapshot_cancels_actor_and_critic_sampling_variance():
    agent, args = _agent(dtype=torch.float64)
    snapshot = deepcopy(agent)
    components = _components(agent)
    gradients = torch.stack([_direct_gradient(snapshot, batch, args)[0] for batch in components])
    state = _reference_state(gradients, mode="fixed")
    corrected = []
    for batch in components:
        current, _ = _direct_gradient(agent, batch, args)
        reference, _ = _direct_gradient(snapshot, batch, args)
        result, _ = state.correct(current, reference)
        corrected.append(result)
    corrected = torch.stack(corrected)
    offset = 0
    for name, parameter in agent.named_parameters():
        section = slice(offset, offset + parameter.numel())
        torch.testing.assert_close(
            corrected[:, section], gradients.mean(0)[section].expand(3, -1),
            atol=2e-15, rtol=2e-13,
        )
        offset += parameter.numel()
    for prefix in ("actor", "critic"):
        mask = torch.cat([
            torch.full((parameter.numel(),), name.startswith(prefix), device="cuda", dtype=torch.bool)
            for name, parameter in agent.named_parameters()
        ])
        assert gradients[:, mask].var(dim=0, correction=0).sum() > 1e-8
    # Wrong pairing still preserves the full-draw mean, so mean identity alone
    # cannot detect this error. Its snapshot variance must remain nonzero.
    mismatched = torch.stack([
        state.correct(gradient, gradients[(index + 1) % 3])[0]
        for index, gradient in enumerate(gradients)
    ])
    assert mismatched.var(dim=0, correction=0).sum() > 1e-8


def test_partial_calibration_beats_raw_and_fixed_on_analytic_quadratic():
    q = torch.tensor([-1., -1., 1., 1.], device="cuda", dtype=torch.float64)
    r = torch.tensor([-1., 1., -1., 1.], device="cuda", dtype=torch.float64)
    # Four strictly convex losses L_i(theta)=h_i theta^2/2+q_i theta.
    # At theta_ref=0, g_ref=q. At theta=1, g=2+.25q+.2r.
    # Orthogonal q,r have unit variance: lambda*=.25 and residual variance=.04.
    hessian = 2 - .75 * q + .2 * r
    gradients = []
    for location in (0., 1.):
        theta = torch.tensor(location, device="cuda", dtype=torch.float64, requires_grad=True)
        losses = .5 * hessian * theta.square() + q * theta
        gradients.append(torch.stack([
            torch.autograd.grad(loss, theta, retain_graph=True)[0] for loss in losses
        ])[:, None])
    references, current = gradients
    torch.testing.assert_close(references[:, 0], q)
    torch.testing.assert_close(current[:, 0], 2 + .25 * q + .2 * r)
    adaptive = _reference_state(references)
    fixed = _reference_state(references, mode="fixed")
    # Deterministic balanced calibration history; evaluation freezes the fitted
    # coefficient rather than adapting it to the component being evaluated.
    for _ in range(128):
        for gradient, reference in zip(current, references):
            adaptive.step(gradient, reference)
    coefficient = adaptive.coefficients()
    torch.testing.assert_close(coefficient, torch.full_like(coefficient, .25), atol=.025, rtol=0)
    corrected = torch.stack([adaptive.correct(g, ref)[0] for g, ref in zip(current, references)])
    fixed_corrected = torch.stack([fixed.correct(g, ref)[0] for g, ref in zip(current, references)])
    raw_variance = current.var(dim=0, correction=0)
    fixed_variance = fixed_corrected.var(dim=0, correction=0)
    reduced_variance = corrected.var(dim=0, correction=0)
    torch.testing.assert_close(raw_variance, torch.full_like(raw_variance, .1025), atol=1e-14, rtol=0)
    torch.testing.assert_close(fixed_variance, torch.full_like(fixed_variance, .6025), atol=1e-14, rtol=0)
    torch.testing.assert_close(reduced_variance, .04 + (coefficient - .25).square(), atol=1e-14, rtol=0)
    assert (reduced_variance < .5 * raw_variance).all()
    assert (reduced_variance < .1 * fixed_variance).all()
    torch.testing.assert_close(corrected.mean(0), current.mean(0), atol=1e-14, rtol=0)


def test_correction_precedes_assimilation_and_anticorrelation_rejects_reference():
    references = torch.tensor([[-1., 2.], [1., -2.]], device="cuda", dtype=torch.float64)
    state = _reference_state(references)
    reference = references[1]
    gradient = .25 * reference
    corrected, coefficient = state.step(gradient, reference)
    torch.testing.assert_close(coefficient, torch.ones_like(coefficient), atol=0, rtol=0)
    torch.testing.assert_close(corrected, -.75 * reference, atol=0, rtol=0)
    # Only the NEXT prediction can use this pair's evidence. Repeated correct
    # calls are read-only, not additional calibration observations.
    for _ in range(3):
        corrected, coefficient = state.correct(gradient, reference)
        torch.testing.assert_close(coefficient, torch.full_like(coefficient, .25), atol=0, rtol=0)
        torch.testing.assert_close(corrected, torch.zeros_like(corrected), atol=0, rtol=0)
    anticorrelated = _reference_state(references)
    anticorrelated.step(-reference, reference)
    corrected, coefficient = anticorrelated.correct(-references[0], references[0])
    torch.testing.assert_close(coefficient, torch.zeros_like(coefficient), atol=0, rtol=0)
    torch.testing.assert_close(corrected, -references[0], atol=0, rtol=0)


def test_reset_discards_old_rollout_and_zero_variance_preserves_ordinary_gradient():
    old = torch.tensor([[-3., 8., 2.], [3., -8., 2.]], device="cuda", dtype=torch.float32)
    state = _reference_state(old)
    state.step(-old[1], old[1])
    state.step(2 * old[0], old[0])
    state.reset()
    new = torch.tensor([[12., 7., -4.], [10., 7., 2.], [8., 7., 5.]], device="cuda")
    for reference in new:
        state.accumulate_reference(reference)
    state.finish_reference()
    fresh = _reference_state(new)
    for reference, gradient in zip(new, new.square() * .125):
        actual, coefficient = state.step(gradient, reference)
        expected, expected_coefficient = fresh.step(gradient, reference)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(coefficient, expected_coefficient, atol=0, rtol=0)
        torch.testing.assert_close(actual[1], gradient[1], atol=0, rtol=0)
        torch.testing.assert_close(coefficient[1], coefficient.new_zeros(()), atol=0, rtol=0)
    # A single reference component has no sampling noise in either mode.
    for mode in ("adaptive", "fixed"):
        singleton = _reference_state(new[:1], mode=mode)
        gradient = new[0] + 3
        corrected, _ = singleton.step(gradient, new[0])
        torch.testing.assert_close(corrected, gradient, atol=0, rtol=0)


@pytest.mark.parametrize("activation", ["relu", "situglu"])
def test_compiled_and_eager_ppo_updates_match_independent_next_predictions(activation):
    eager, args = _agent(activation)
    compiled = deepcopy(eager)
    oracle = deepcopy(eager)
    eager_reference, compiled_reference = deepcopy(eager), deepcopy(eager)
    components = _components(eager)
    eager_loss = partial(m.ppo_loss, eager, args=args)
    eager_reference_loss = partial(m.ppo_loss, eager_reference, args=args)
    compiled_loss = torch.compile(
        partial(m.ppo_loss, compiled, args=args), fullgraph=True, mode="reduce-overhead",
    )
    compiled_reference_loss = torch.compile(
        partial(m.ppo_loss, compiled_reference, args=args), fullgraph=True, mode="reduce-overhead",
    )
    references = torch.stack([_direct_gradient(eager_reference, batch, args)[0] for batch in components])
    eager_state, compiled_state, oracle_state = [_reference_state(references) for _ in range(3)]
    compiled_step = torch.compile(
        compiled_state.step, fullgraph=True, options={"triton.cudagraphs": False},
    )
    optimizers = [torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5, fused=True)
                  for model in (eager, compiled, oracle)]
    query = torch.cat([batch[0] for batch in components])
    held_snapshots = []
    # Revisit components after graph warmup and parameter updates. Retaining
    # snapshots also detects graph-output aliasing across current/reference calls.
    for index in (2, 0, 2, 1, 0, 1):
        batch = components[index]
        expected_gradient, expected_metrics = _direct_gradient(oracle, batch, args)
        expected_corrected, expected_coefficient = oracle_state.step(expected_gradient, references[index])
        eager_gradient, eager_metrics = m.loss_and_flat_gradient(eager, eager_loss, batch, compiled=False)
        eager_ref, _ = m.loss_and_flat_gradient(eager_reference, eager_reference_loss, batch, compiled=False)
        eager_corrected, eager_coefficient = eager_state.step(eager_gradient, eager_ref)
        actual_gradient, actual_metrics = m.loss_and_flat_gradient(compiled, compiled_loss, batch)
        actual_ref, _ = m.loss_and_flat_gradient(compiled_reference, compiled_reference_loss, batch)
        actual_corrected, actual_coefficient = compiled_step(actual_gradient, actual_ref)
        for gradient in (eager_gradient, actual_gradient):
            torch.testing.assert_close(gradient, expected_gradient, atol=3e-6, rtol=5e-4)
        for metrics in (eager_metrics, actual_metrics):
            torch.testing.assert_close(metrics, expected_metrics, atol=3e-6, rtol=5e-4)
        for corrected in (eager_corrected, actual_corrected):
            torch.testing.assert_close(corrected, expected_corrected, atol=4e-6, rtol=8e-4)
        # Tiny near-zero reference coordinates need not have numerically stable
        # coefficients; the corrected gradient and next prediction are the contract.
        held_snapshots.append((actual_gradient, expected_gradient.clone(), actual_metrics, expected_metrics.clone()))
        m.assign_flat_gradient(eager, eager_corrected)
        m.assign_flat_gradient(compiled, actual_corrected)
        # Independent assignment oracle: bypass BOTH exported gradient helpers.
        for parameter, gradient in zip(oracle.parameters(), expected_corrected.split(
                [parameter.numel() for parameter in oracle.parameters()])):
            parameter.grad = gradient.reshape_as(parameter).clone()
        for model, optimizer in zip((eager, compiled, oracle), optimizers):
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
        with torch.no_grad():
            expected_predictions = oracle.get_policy_and_value(query)
            for model in (eager, compiled):
                for actual, expected in zip(model.get_policy_and_value(query), expected_predictions):
                    torch.testing.assert_close(actual, expected, atol=8e-6, rtol=8e-4)
    for gradient, expected_gradient, metrics, expected_metrics in held_snapshots:
        torch.testing.assert_close(gradient, expected_gradient, atol=3e-6, rtol=5e-4)
        torch.testing.assert_close(metrics, expected_metrics, atol=3e-6, rtol=5e-4)
