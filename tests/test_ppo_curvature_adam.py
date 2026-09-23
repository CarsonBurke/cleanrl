"""Queue these CUDA directional-curvature optimizer checks through mlq."""

import pytest
import torch

from cleanrl.ppo_continuous_action_curvature_adam_v2 import CurvatureAdam
from test_ppo_normres_twohot import device


@pytest.fixture(autouse=True)
def isolated_runtime(device):
    torch._dynamo.reset()
    yield device
    torch._dynamo.reset()


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def parameters(actor=(2.0,), critic=(5.0,)):
    return [torch.nn.Parameter(torch.tensor(value, device="cuda")) for value in (actor, critic)]


def quadratic(params, curvatures=(1.0, 1.0), targets=(0.0, 0.0)):
    return torch.stack(
        [0.5 * (curvature * (parameter - target).square()).sum()
         for parameter, curvature, target in zip(params, curvatures, targets)]
    )


def observe_gradient(optimizer, params, curvatures=(1.0, 1.0), targets=(0.0, 0.0)):
    optimizer.zero_grad(set_to_none=True)
    losses = quadratic(params, curvatures, targets)
    losses.sum().backward()
    return losses.detach()


def test_probe_matches_default_adam_but_predicts_with_unmodified_raw_gradients():
    params = parameters((3.0, -4.0, 1e-4), (2.0,))
    refs = [torch.nn.Parameter(parameter.detach().clone()) for parameter in params]
    optimizer = CurvatureAdam([params[0]], [params[1]], compile=False)
    reference = torch.optim.Adam(refs, lr=3e-4, eps=1e-5, fused=True)
    observe_gradient(optimizer, params, (100.0, 200.0))
    origins = [parameter.detach().clone() for parameter in params]
    gradients = [parameter.grad.clone() for parameter in params]
    for ref, gradient in zip(refs, gradients):
        ref.grad = gradient.clone()
    # Clipping is deliberately strong and joint, not one norm per objective.
    torch.nn.utils.clip_grad_norm_(refs, 0.5)
    reference.step()
    optimizer.probe()

    prediction = torch.stack(
        [(gradient * (origin - parameter.detach())).sum()
         for gradient, origin, parameter in zip(gradients, origins, params)]
    )
    torch.testing.assert_close(optimizer.probe_prediction, prediction, rtol=2e-6, atol=1e-8)
    for parameter, ref, gradient in zip(params, refs, gradients):
        torch.testing.assert_close(parameter, ref, rtol=0, atol=5e-7)
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)


def test_quadratic_candidates_are_independent_exact_directional_optima_without_more_adam_updates():
    params = parameters((2.0, -1.0), (0.5, 3.0))
    curvatures = [torch.tensor(value, device="cuda") for value in ((1.0, 4.0), (9.0, 2.0))]
    optimizer = CurvatureAdam([params[0]], [params[1]], lr=0.4, max_grad_norm=100.0, compile=False)
    before = observe_gradient(optimizer, params, curvatures)
    origins = [parameter.detach().clone() for parameter in params]
    gradients = [parameter.grad.clone() for parameter in params]
    optimizer.probe()
    probe = quadratic(params, curvatures).detach()
    directions = [(origin - parameter.detach()) / 0.4 for origin, parameter in zip(origins, params)]
    optimal_lrs = torch.stack(
        [(gradient * direction).sum() / (curvature * direction.square()).sum()
         for gradient, direction, curvature in zip(gradients, directions, curvatures)]
    )
    history = [{key: optimizer.state[parameter][key].clone() for key in ("step", "exp_avg", "exp_avg_sq")}
               for parameter in params]
    for state in history:
        assert state["step"].item() == 1

    optimizer.propose(before, probe)
    for parameter, origin, direction, length in zip(params, origins, directions, optimal_lrs):
        torch.testing.assert_close(parameter, origin - length * direction, rtol=2e-5, atol=2e-5)
    candidate = quadratic(params, curvatures).detach()
    candidate_weights = [parameter.detach().clone() for parameter in params]
    candidate_prediction = torch.stack(
        [(gradient * (origin - parameter.detach())).sum()
         for gradient, origin, parameter in zip(gradients, origins, params)]
    )
    torch.testing.assert_close(optimizer.candidate_prediction, candidate_prediction, rtol=2e-5, atol=1e-6)
    for parameter, state in zip(params, history):
        for key, expected in state.items():
            torch.testing.assert_close(optimizer.state[parameter][key], expected, rtol=0, atol=0)

    metrics = optimizer.finish(before, probe, candidate)
    assert metrics.shape == (2, 10)
    torch.testing.assert_close(metrics[:, 7], torch.full_like(metrics[:, 7], 2))
    torch.testing.assert_close(metrics[:, 9], before - candidate)
    torch.testing.assert_close(optimizer.lrs, optimal_lrs, rtol=2e-5, atol=2e-5)
    for parameter, expected_weights, state in zip(params, candidate_weights, history):
        torch.testing.assert_close(parameter, expected_weights, rtol=0, atol=0)
        for key, expected in state.items():
            torch.testing.assert_close(optimizer.state[parameter][key], expected, rtol=0, atol=0)


def test_negative_curvature_expands_the_checked_step_without_a_learning_rate_ceiling():
    params = parameters()
    optimizer = CurvatureAdam([params[0]], [params[1]], lr=3.0, compile=False)
    curvatures = (-1.0, -4.0)
    before = observe_gradient(optimizer, params, curvatures)
    origins = [parameter.detach().clone() for parameter in params]
    optimizer.probe()
    probe = quadratic(params, curvatures).detach()
    probe_weights = [parameter.detach().clone() for parameter in params]
    optimizer.propose(before, probe)
    for parameter, origin, checked in zip(params, origins, probe_weights):
        torch.testing.assert_close(parameter, origin + 2 * (checked - origin), rtol=2e-6, atol=1e-6)
    candidate = quadratic(params, curvatures).detach()
    metrics = optimizer.finish(before, probe, candidate)
    torch.testing.assert_close(optimizer.lrs, torch.full_like(optimizer.lrs, 6.0))
    torch.testing.assert_close(metrics[:, 7], torch.full_like(metrics[:, 7], 2))
    torch.testing.assert_close(quadratic(params, curvatures).detach(), candidate)


def test_actor_candidate_can_succeed_while_nonfinite_critic_trials_restore_origin():
    params = parameters()
    optimizer = CurvatureAdam([params[0]], [params[1]], lr=0.4, compile=False)
    before = observe_gradient(optimizer, params)
    critic_origin = params[1].detach().clone()
    optimizer.probe()
    critic_history = {key: optimizer.state[params[1]][key].clone() for key in ("step", "exp_avg", "exp_avg_sq")}
    probe = quadratic(params).detach()
    probe[1] = float("inf")
    optimizer.propose(before, probe)
    torch.testing.assert_close(optimizer.candidate_lrs[1], optimizer.probe_lrs[1] / 2)
    candidate = quadratic(params).detach()
    candidate[1] = float("inf")
    actor_candidate = params[0].detach().clone()
    metrics = optimizer.finish(before, probe, candidate)

    torch.testing.assert_close(params[0], actor_candidate, rtol=0, atol=0)
    torch.testing.assert_close(params[1], critic_origin, rtol=0, atol=0)
    torch.testing.assert_close(metrics[:, 7], torch.tensor([2.0, 0.0], device="cuda"))
    torch.testing.assert_close(metrics[:, 9], torch.stack((before[0] - candidate[0], before.new_zeros(()))))
    torch.testing.assert_close(optimizer.lrs[1], before.new_tensor(0.1))
    for key, expected in critic_history.items():
        torch.testing.assert_close(optimizer.state[params[1]][key], expected, rtol=0, atol=0)


def test_nonfinite_candidates_retain_the_improving_probe_weights_and_lengths():
    params = parameters()
    optimizer = CurvatureAdam([params[0]], [params[1]], lr=0.4, compile=False)
    before = observe_gradient(optimizer, params)
    optimizer.probe()
    probe = quadratic(params).detach()
    probe_weights = [parameter.detach().clone() for parameter in params]
    optimizer.propose(before, probe)
    # NaN and infinity must not displace an already checked, improving probe.
    candidate = before.new_tensor([float("nan"), float("inf")])
    metrics = optimizer.finish(before, probe, candidate)

    for parameter, expected in zip(params, probe_weights):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    torch.testing.assert_close(metrics[:, 7], torch.ones_like(metrics[:, 7]))
    torch.testing.assert_close(metrics[:, 9], before - probe)
    torch.testing.assert_close(optimizer.lrs, torch.full_like(optimizer.lrs, 0.4))


def test_uphill_momentum_uses_current_gradient_with_the_same_adam_variance():
    params = parameters((1.0,), (1.0,))
    optimizer = CurvatureAdam([params[0]], [params[1]], lr=0.1, max_grad_norm=100.0, compile=False)
    before = observe_gradient(optimizer, params)
    optimizer.probe()
    probe = quadratic(params).detach()
    optimizer.propose(before, probe)
    optimizer.finish(before, probe, quadratic(params).detach())

    targets = (params[0].detach().clone() + 0.01, params[1].detach().clone() - 1.0)
    before = observe_gradient(optimizer, params, targets=targets)
    actor_origin = params[0].detach().clone()
    gradient = params[0].grad.clone()
    optimizer.lrs.fill_(0.1)
    optimizer.probe()
    state = optimizer.state[params[0]]
    assert (gradient * state["exp_avg"]).sum().item() < 0
    expected_direction = gradient / ((state["exp_avg_sq"] / (1 - 0.999**2)).sqrt() + 1e-5)
    torch.testing.assert_close(params[0], actor_origin - 0.1 * expected_direction, rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(optimizer.fallback, torch.tensor([1.0, 0.0], device="cuda"), check_dtype=False)
    probe = quadratic(params, targets=targets).detach()
    assert probe[0].item() < before[0].item()
    optimizer.propose(before, probe)
    candidate = quadratic(params, targets=targets).detach()
    optimizer.finish(before, probe, candidate)
    assert quadratic(params, targets=targets)[0].item() <= probe[0].item()


def test_zero_current_gradient_preserves_origin_despite_nonzero_momentum():
    params = parameters()
    optimizer = CurvatureAdam([params[0]], [params[1]], lr=0.4, compile=False)
    before = observe_gradient(optimizer, params)
    optimizer.probe()
    probe = quadratic(params).detach()
    optimizer.propose(before, probe)
    optimizer.finish(before, probe, quadratic(params).detach())

    origins = tuple(parameter.detach().clone() for parameter in params)
    before = observe_gradient(optimizer, params, targets=origins)
    lengths = optimizer.lrs.clone()
    optimizer.probe()
    probe = quadratic(params, targets=origins).detach()
    optimizer.propose(before, probe)
    candidate = quadratic(params, targets=origins).detach()
    metrics = optimizer.finish(before, probe, candidate)

    for parameter, origin in zip(params, origins):
        torch.testing.assert_close(parameter, origin, rtol=0, atol=0)
    torch.testing.assert_close(metrics[:, (0, 2, 7, 9)], torch.zeros_like(metrics[:, (0, 2, 7, 9)]), rtol=0, atol=0)
    torch.testing.assert_close(optimizer.lrs, lengths, rtol=0, atol=0)


def test_compiled_cycles_allow_new_values_and_lengths_without_recompile_or_cuda_sync():
    from torch._inductor.compile_fx import compile_fx

    compilations = 0

    def backend(graph, inputs, **kwargs):
        nonlocal compilations
        compilations += 1
        return compile_fx(graph, inputs, config_patches=kwargs.pop("options", {}), **kwargs)

    real_compile = torch.compile
    params = parameters((1.0, -0.5), (2.0, 3.0))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "compile", lambda fn, **kwargs: real_compile(fn, backend=backend, **kwargs))
        optimizer = CurvatureAdam([params[0]], [params[1]])

    with torch._dynamo.config.patch(error_on_recompile=True):
        for index in range(8):
            targets = (0.1 * index, -0.2 * index)
            before = observe_gradient(optimizer, params, (1.0, 7.0), targets)
            optimizer.lrs.copy_(before.new_tensor([0.1 + 0.02 * index, 0.3 - 0.01 * index]))
            previous = torch.cuda.get_sync_debug_mode()
            try:
                if index:
                    torch.cuda.set_sync_debug_mode("error")
                optimizer.probe()
                probe = quadratic(params, (1.0, 7.0), targets).detach()
                optimizer.propose(before, probe)
                candidate = quadratic(params, (1.0, 7.0), targets).detach()
                metrics = optimizer.finish(before, probe, candidate)
            finally:
                torch.cuda.set_sync_debug_mode(previous)
            if index == 0:
                warm_compilations = compilations
                assert warm_compilations > 0
            assert compilations == warm_compilations
            expected = torch.stack((before, probe, candidate)).min(dim=0).values
            torch.testing.assert_close(quadratic(params, (1.0, 7.0), targets).detach(), expected, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(metrics[:, 9], before - expected, rtol=2e-5, atol=2e-6)
