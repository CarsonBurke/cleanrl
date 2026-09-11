"""CUDA contracts for scheduled support retirement, causal forecasts, and capture.

Small dimensions and periods here exercise boundaries, not learning performance.
"""

import json
import math

import pytest
import torch

from cleanrl.plasticity.predictive_persistent_support_v4 import PersistentSupport
from cleanrl.shared import runtime

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def setup_model(periods=(4,), share_rate=.1, noise_rate=.5, input_dim=5):
    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    return PersistentSupport(input_dim, 'cuda', periods=periods,
                             share_rate=share_rate, noise_rate=noise_rate)


def compiled_update(model):
    return torch.compile(model.update, fullgraph=True, mode='max-autotune-no-cudagraphs')


def warmup(*updates):
    x = torch.tensor([1., -.5, .25, .7, -.3], device='cuda')
    for value in (.7, -.2, 1.3, .5):
        y = torch.tensor(value, device='cuda')
        for update in updates:
            update(x, y)
    return x


@torch.no_grad()
def test_current_label_cannot_change_reset_boundary_forecast():
    left, right = setup_model(), setup_model()
    update_left, update_right = compiled_update(left), compiled_update(right)
    x = warmup(update_left, update_right)
    saved_x = x.clone()
    low, high = torch.tensor(-4., device='cuda'), torch.tensor(4., device='cuda')
    # Index four retires the offset-zero cohort, before either label is seen.
    a, b = update_left(x, low), update_right(x, high)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.allclose(left.mean_weights() @ x, right.mean_weights() @ x)
    torch.testing.assert_close(x, saved_x, rtol=0, atol=0)
    assert float(low) == -4. and float(high) == 4.


@torch.no_grad()
def test_pure_coefficients_predict_next_output_across_both_restart_phases():
    model = setup_model()
    update = compiled_update(model)
    generator = torch.Generator(device='cuda').manual_seed(17)
    xs = torch.randn(11, 5, device='cuda', generator=generator)
    ys = torch.randn(11, device='cuda', generator=generator)
    for step, (x, y) in enumerate(zip(xs, ys)):
        before = [value.clone() for value in model.state_tensors()]
        coefficients = model.mean_weights()
        weights = model.aggregation_weights()
        if step == 4:
            diagnostic = model.diagnostics()
            json.dumps(diagnostic, allow_nan=False)
            assert diagnostic['cohort_ages'] == [4, 0, 2]
            assert diagnostic['next_restart'] == [False, True, False]
            assert diagnostic['inclusion'][1] == pytest.approx([.2, .2])
            assert diagnostic['effective_weights'][1] == [0., 0.]
        for actual, saved in zip(model.state_tensors(), before):
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)
        torch.testing.assert_close(coefficients[:3], weights @ coefficients[3:])
        expected = coefficients @ x
        actual = update(x, y)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-7)


@torch.no_grad()
def test_restarting_cohort_relearns_from_original_prior_while_persistent_retains_evidence():
    model, fresh = setup_model(), setup_model()
    update, fresh_update = compiled_update(model), compiled_update(fresh)
    x = warmup(update)
    stored_persistent = model.filter.mean_weights()[0].clone()
    assert stored_persistent.norm() > 0
    assert model.filter.mean_weights()[1].norm() > 0
    next_coefficients = model.mean_weights()
    torch.testing.assert_close(next_coefficients[4], stored_persistent, rtol=0, atol=0)
    torch.testing.assert_close(next_coefficients[5], torch.zeros_like(stored_persistent), rtol=0, atol=0)
    y = torch.tensor(.9, device='cuda')
    actual, newborn = update(x, y), fresh_update(x, y)
    torch.testing.assert_close(actual[5], newborn[4], rtol=0, atol=0)
    torch.testing.assert_close(actual[4], stored_persistent @ x, rtol=3e-5, atol=2e-7)
    # Matching a fresh cohort after conditioning also verifies odds, slab
    # uncertainty, and residual-scale sufficient statistics were all reset.
    for state, initial in zip(model.filter.state_tensors(), fresh.filter.state_tensors()):
        torch.testing.assert_close(state[1], initial[0], rtol=3e-5, atol=2e-7)
    for state, initial in ((model.count, fresh.count), (model.residual_sum, fresh.residual_sum),
                           (model.noise, fresh.noise)):
        torch.testing.assert_close(state[2], initial[1], rtol=3e-5, atol=2e-7)
    assert model.count[1] > model.count[2]


@torch.no_grad()
def test_retired_dominant_cohort_gets_only_share_prior_even_after_survivor_underflow():
    model = setup_model(share_rate=.2)
    # Cohort one is due at t=4. Its dominant mature mass must not survive its
    # replacement, even when every surviving probability would underflow FP32.
    model.observations.fill_(4)
    model.log_weights.copy_(torch.tensor([-1000., -1001., 0., -1002.], device='cuda'))
    before = [state.clone() for state in model.state_tensors()]
    prior = torch.tensor([.5, 1 / 6, 1 / 6, 1 / 6], device='cuda', dtype=torch.float64)
    survivor = torch.tensor([-1000., -1001., -math.inf, -1002.], device='cuda', dtype=torch.float64).softmax(0)
    expected = .8 * survivor + .2 * prior
    weights = model.aggregation_weights()
    torch.testing.assert_close(weights[:2], expected.float().expand(2, -1), rtol=5e-5, atol=2e-7)
    torch.testing.assert_close(weights[2], prior.float())
    assert float(weights[0, 2]) == pytest.approx(.2 / 6)
    for actual, saved in zip(model.state_tensors(), before):
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)
    # With zero covariates and equal initial scales all scores are identical,
    # so the posterior must retain exactly the retired-and-shared prior.
    compiled_update(model)(torch.zeros(5, device='cuda'), torch.zeros((), device='cuda'))
    torch.testing.assert_close(model.log_weights.exp(), expected.float().expand(2, -1), rtol=5e-5, atol=2e-7)


@torch.no_grad()
def test_birth_only_preserves_cumulative_evidence_between_scheduled_births():
    model = setup_model(share_rate=.2)
    update = compiled_update(model)
    posterior = torch.tensor([.7, .1, .15, .05], device='cuda')
    model.log_weights.copy_(posterior.log())
    model.observations.fill_(1)
    prior = model.log_prior.exp()
    # Index one has no births: the primary row retains its evidence exactly,
    # while the switching control refreshes toward the mixture prior.
    expected_switching = .8 * posterior + .2 * prior
    weights = model.aggregation_weights()
    torch.testing.assert_close(weights[0], posterior)
    torch.testing.assert_close(weights[1], expected_switching)
    x, y = torch.zeros(5, device='cuda'), torch.zeros((), device='cuda')
    update(x, y)
    torch.testing.assert_close(model.log_weights[0].exp(), posterior)
    torch.testing.assert_close(model.log_weights[1].exp(), expected_switching)
    # The following forecast is at the staggered birth t=2. Both rows now
    # retire cohort two, renormalize their own survivors, and admit the newborn.
    birth_weights = model.aggregation_weights()
    for row, previous in enumerate((posterior, expected_switching)):
        survivors = previous.clone()
        survivors[3] = 0
        expected = .8 * survivors / survivors.sum() + .2 * prior
        torch.testing.assert_close(birth_weights[row], expected)


@pytest.mark.parametrize('noise_rate', [0., .5])
@torch.no_grad()
def test_integrated_scoring_and_retained_scale_statistics_restart_causally(noise_rate):
    model = setup_model(noise_rate=noise_rate)
    update = compiled_update(model)
    x = torch.tensor([1., -.5, .25, .7, -.3], device='cuda')
    labels = torch.tensor([.7, -1.2, 2., -.4, 1.3, -.8], device='cuda')
    count = torch.zeros(4, device='cuda', dtype=torch.float64)
    residual_sum = torch.zeros_like(count)
    noise = torch.ones_like(count)
    for step, y in enumerate(labels):
        # An independent host-side schedule oracle for period four's two phases.
        retired = torch.tensor([False, False, step > 0 and step % 4 == 0,
                                step >= 2 and (step - 2) % 4 == 0], device='cuda')
        count[retired] = 0
        residual_sum[retired] = 0
        noise[retired] = 1
        _, stored_variance = model.filter.predict(x, noise[1:].float())
        variance = torch.cat((noise[:1].float(), stored_variance))
        # A newborn has marginal coefficient variance 1/D and noise prior one.
        variance[retired] = 1 + x.square().sum() / 5
        mean = model.mean_weights()[3:] @ x
        likelihood = -.5 * (math.log(2 * math.pi) + variance.log() + (y - mean).square() / variance)
        posterior = model.aggregation_weights()[:2].log() + likelihood
        expected_log = posterior - posterior.logsumexp(-1, keepdim=True)
        prediction = update(x, y)
        torch.testing.assert_close(prediction[3:], mean, rtol=3e-5, atol=2e-7)
        torch.testing.assert_close(model.log_weights, expected_log, rtol=3e-5, atol=2e-6)
        count = (1 - noise_rate) * count + 1
        residual_sum = (1 - noise_rate) * residual_sum + (y.double() - prediction[3:].double()).square()
        noise = (2 + residual_sum) / (2 + count)
        torch.testing.assert_close(model.count, count.float())
        torch.testing.assert_close(model.residual_sum, residual_sum.float(), rtol=3e-5, atol=2e-7)
        torch.testing.assert_close(model.noise, noise.float(), rtol=3e-5, atol=2e-7)
        torch.testing.assert_close(model.energy, noise[0].float(), rtol=3e-5, atol=2e-7)


@torch.no_grad()
def test_capture_restores_all_state_and_replays_restart_crossing_nondivisible_tail():
    from cleanrl.plasticity import covariance_sparse_eval_v1 as sparse
    from cleanrl.plasticity.predictive_mean_risk_conjugate_eval_v3 import Runner

    runtime.configure_runtime(matmul_precision='highest', allow_tf32=False)
    generator = torch.Generator(device='cuda').manual_seed(1)
    xs = torch.randn(11, 5, device='cuda', generator=generator)
    ys = torch.randn(11, device='cuda', generator=generator)
    args = sparse.Args(input_dim=5, steps=11, graph_steps=4)

    def make():
        return Runner(xs, ys, setup_model(periods=(6,)),
                      sparse.LinearLearner('adam', (.001, .003), args, xs, ys))

    actual, reference = make(), make()
    initial = [state.clone() for state in actual.mutable]
    graphs = actual.capture(4)
    for state, saved in zip(actual.mutable, initial):
        torch.testing.assert_close(state, saved, rtol=0, atol=0)
    graphs[4].replay()
    graphs[1].replay()
    # Save at five; both subsequent graph and scalar replay cross resets at six
    # and nine. Restoring the observation counter is essential to repeat them.
    checkpoint = [state.clone() for state in actual.mutable]

    def finish():
        graphs[4].replay()
        graphs[1].replay()
        graphs[1].replay()

    finish()
    final = [state.clone() for state in actual.mutable]
    for state, saved in zip(actual.mutable, checkpoint):
        state.copy_(saved)
    finish()
    torch.cuda.synchronize()
    for state, saved in zip(actual.mutable, final):
        torch.testing.assert_close(state, saved, rtol=0, atol=0)
    reference_update = compiled_update(reference)
    for _ in range(11):
        reference_update()
    torch.cuda.synchronize()
    assert int(actual.index) == int(actual.model.observations) == int(actual.adam.index) == 11
    for state, expected in zip(actual.mutable, reference.mutable):
        torch.testing.assert_close(state, expected, rtol=3e-4, atol=3e-6)
    torch.testing.assert_close(actual.model.mean_weights(), reference.model.mean_weights(), rtol=3e-4, atol=3e-6)
