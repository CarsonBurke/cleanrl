"""CUDA numerical contracts; execution is queued through mlq, never on CPU."""
import math

import torch

from cleanrl.plasticity.predictive_evidence_v3 import (
    PredictiveEvidenceState,
    fit_inclusion_probability,
)
from cleanrl.shared.runtime import configure_runtime


def _examples():
    feature = torch.tensor(
        [[-.5, 0., .25], [-.25, .5, 0.], [0., -.25, -.5],
         [.25, 0., .5], [.5, .25, -.25], [-.5, -.5, .25],
         [.25, .5, -.5], [.5, -.25, 0.]],
        dtype=torch.float64, device="cuda",
    )[:, None, :]
    target = 3. + 2.5 * feature[..., 0] + .1 * feature[..., 1] - .05 * feature[..., 2]
    return feature, target


def _observe_without_writes(state, weight, feature, target):
    absent = torch.zeros_like(weight)
    for x, y in zip(feature, target):
        # Zero Jacobians imply valid zero gradients even with nonzero residuals.
        state.step(weight, absent, absent, x, y, -y)


@torch.no_grad()
def test_welford_centered_moments_match_independent_batch_computation():
    feature, target = _examples()
    feature = torch.cat((feature, .5 * feature + .125), dim=1)
    feature = torch.cat((feature, torch.ones_like(feature[..., :1]) * .25), dim=-1)
    target = torch.cat((target, 100000. - 2. * target), dim=1)
    weight = torch.zeros_like(feature[0])
    state = PredictiveEvidenceState(weight)
    _observe_without_writes(state, weight, feature, target)

    centered_x = feature - feature.mean(dim=0)
    centered_y = target - target.mean(dim=0)
    torch.testing.assert_close(state.mean_x, feature.mean(dim=0), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(state.mean_y, target.mean(dim=0), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(state.m2_x, centered_x.square().sum(dim=0), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(state.m2_y, centered_y.square().sum(dim=0), rtol=1e-10, atol=1e-9)
    torch.testing.assert_close(
        state.cross_xy, (centered_x * centered_y[..., None]).sum(dim=0),
        rtol=1e-10, atol=1e-9,
    )


@torch.no_grad()
def test_unit_information_bayes_factor_matches_centered_regression_reference():
    feature, target = _examples()
    feature = torch.cat((feature, torch.ones_like(feature[..., :1]) * .25), dim=-1)
    weight = torch.zeros_like(feature[0])
    state = PredictiveEvidenceState(weight, control="fixed_prior")
    _observe_without_writes(state, weight, feature, target)
    statistics = state.statistics(feature[-1])

    # Integrate the slope via the observation-space covariance I + n * H_x.
    # This uses a dense solve and determinant, not the implementation's R-squared formula.
    n = feature.shape[0]
    centered_y = target[:, 0] - target[:, 0].mean()
    null_energy = centered_y @ centered_y
    expected_bf = torch.zeros_like(weight)
    alternative_energy = null_energy.expand_as(weight).clone()
    for coordinate in range(feature.shape[-1] - 1):
        x = feature[:, 0, coordinate]
        x = x - x.mean()
        covariance = torch.eye(n, dtype=x.dtype, device=x.device) + n * torch.outer(x, x) / (x @ x)
        alternative_energy[0, coordinate] = centered_y @ torch.linalg.solve(covariance, centered_y)
        expected_bf[0, coordinate] = (
            -.5 * torch.linalg.slogdet(covariance).logabsdet
            - (n - 1) / 2 * torch.log(alternative_energy[0, coordinate] / null_energy)
        )
    torch.testing.assert_close(statistics["log_bf"], expected_bf, rtol=1e-11, atol=1e-12)
    probability = expected_bf.sigmoid()
    expected_noise = ((1. - probability) * null_energy + probability * alternative_energy) / (n - 3)
    torch.testing.assert_close(statistics["noise"], expected_noise, rtol=1e-11, atol=1e-12)
    assert statistics["slope_location"][0, -1].item() == 0.


@torch.no_grad()
def test_target_and_input_rescaling_preserve_evidence_and_signal_fraction():
    feature, target = _examples()
    scale = feature.new_tensor([-.25, 1.5, -1.])
    offset = feature.new_tensor([.1, -.2, .05])
    query = feature.new_tensor([[.4, -.3, .2]])
    statistics = []
    for x, y, probe in (
        (feature, target, query),
        (feature * scale + offset, -7. * target + 9., query * scale + offset),
    ):
        weight = torch.zeros_like(x[0])
        state = PredictiveEvidenceState(weight)
        _observe_without_writes(state, weight, x, y)
        statistics.append(state.statistics(probe))
    original, rescaled = statistics
    for key in ("log_bf", "prior_probability", "participation", "signal_fraction"):
        torch.testing.assert_close(rescaled[key], original[key], rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(rescaled["effect"], -7. * original["effect"], rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(rescaled["noise"], 49. * original["noise"], rtol=1e-11, atol=1e-12)


@torch.no_grad()
def test_inclusion_mle_resolves_interior_boundary_flat_and_extreme_evidence():
    log_bf = torch.tensor(
        [[math.log(4.), math.log(.25)],
         [math.log(9.), math.log(1. / 3.)],
         [math.log(2.), math.log(4.)],
         [math.log(.5), math.log(.25)],
         [0., 0.],
         [1000., -1000.]],
        dtype=torch.float64, device="cuda",
    )
    probability, posterior = fit_inclusion_probability(log_bf)
    expected_probability = log_bf.new_tensor([[.5], [11. / 16.], [1.], [0.], [0.], [.5]])
    expected_posterior = log_bf.new_tensor(
        [[.8, .2], [99. / 104., 11. / 26.], [1., 1.], [0., 0.], [0., 0.], [1., 0.]]
    )
    torch.testing.assert_close(probability, expected_probability, rtol=1e-9, atol=3e-10)
    torch.testing.assert_close(posterior, expected_posterior, rtol=1e-9, atol=3e-10)


@torch.no_grad()
def test_raw_target_evidence_accumulates_without_writes_and_allows_later_activation():
    weight = torch.zeros((1, 1), dtype=torch.float64, device="cuda")
    state = PredictiveEvidenceState(weight)
    feature = weight.new_tensor([-.5, .5] * 6).reshape(-1, 1, 1)
    target = 2. + 3. * feature[..., 0]
    absent = torch.zeros_like(weight)
    query = weight.new_tensor([[.75]])
    initial = state.statistics(query)
    assert initial["participation"].item() == 0.
    assert initial["signal_fraction"].item() == 0.
    for index, (x, y) in enumerate(zip(feature, target)):
        # Residuals have the opposite slope; the observer must learn raw targets.
        state.step(weight, absent, absent, x, y, -y)
        if index < 3:
            prior = state.statistics(query)
            assert bool(torch.isinf(prior["noise"]).all())
            assert prior["signal_fraction"].item() == 0.
    torch.testing.assert_close(weight, torch.zeros_like(weight), rtol=0, atol=0)
    learned = state.statistics(query)
    assert learned["log_bf"].item() > initial["log_bf"].item()
    assert learned["slope_location"].item() > 0.
    assert learned["participation"].item() > .99
    assert learned["signal_fraction"].item() > .9

    jacobian = torch.ones_like(weight)
    next_target = weight.new_tensor([4.25])
    residual = -next_target
    state.step(weight, residual[:, None] * jacobian, jacobian, query, next_target, residual)
    assert abs((weight * jacobian).sum().item() - next_target.item()) < abs(residual.item())


@torch.no_grad()
def test_current_target_cannot_change_its_own_prior_gain():
    feature, target = _examples()
    weight = torch.zeros_like(feature[0])
    state = PredictiveEvidenceState(weight)
    _observe_without_writes(state, weight, feature, target)
    query = weight.new_tensor([[.4, -.3, .2]])
    jacobian = weight.new_tensor([[1., -.5, .25]])
    prior = state.statistics(query)
    gain = prior["participation"] * prior["signal_fraction"] / (
        1. + (prior["participation"] * jacobian.square()).sum(dim=1, keepdim=True)
    )
    assert gain.sum().item() > 0.
    for current_target in (-100., 100.):
        y = weight.new_tensor([current_target])
        residual = (weight * jacobian).sum(dim=1) - y
        gradient = residual[:, None] * jacobian
        updated_weight, _ = state.transition(weight, gradient, jacobian, query, y, residual)
        torch.testing.assert_close(updated_weight, weight - gain * gradient, rtol=1e-11, atol=1e-12)


@torch.no_grad()
def test_zero_jacobian_coordinates_never_write_with_valid_gradients():
    feature, target = _examples()
    weight = feature.new_tensor([[1., -2., .5]])
    state = PredictiveEvidenceState(weight, control="no_evidence")
    _observe_without_writes(state, weight, feature, target)
    query = weight.new_tensor([[.4, -.3, .2]])
    jacobian = weight.new_tensor([[0., 1., -.5]])
    y = weight.new_tensor([7.])
    residual = (weight * jacobian).sum(dim=1) - y
    updated_weight, _ = state.transition(weight, residual[:, None] * jacobian, jacobian, query, y, residual)
    assert updated_weight[0, 0].item() == weight[0, 0].item()
    assert bool((updated_weight[:, 1:] != weight[:, 1:]).any())
    absent = torch.zeros_like(weight)
    unchanged, _ = state.transition(weight, absent, absent, query, y, -y)
    torch.testing.assert_close(unchanged, weight, rtol=0, atol=0)


@torch.no_grad()
def test_many_participating_coordinates_cannot_amplify_signal_noise_gain():
    weight = torch.zeros((2, 1024), dtype=torch.float64, device="cuda")
    state = PredictiveEvidenceState(weight, control="no_evidence")
    x = weight.new_tensor([-.5, .5] * 4)
    noise = weight.new_tensor([-1., -1., 1., 1.] * 2)
    feature = x[:, None, None].expand(-1, *weight.shape)
    target = (.2 * x + noise)[:, None].expand(-1, weight.shape[0])
    _observe_without_writes(state, weight, feature, target)
    query = torch.full_like(weight, .5)
    jacobian = torch.ones_like(weight)
    jacobian[0, 1:] = 0.
    prior = state.statistics(query)
    signal_fraction = prior["signal_fraction"].max(dim=1).values
    assert bool(((signal_fraction > 0.) & (signal_fraction < .01)).all())
    residual = weight.new_tensor([1., 1.])
    updated_weight, _ = state.transition(
        weight, residual[:, None] * jacobian, jacobian, query, -residual, residual,
    )
    output_gain = -((updated_weight - weight) * jacobian).sum(dim=1) / residual
    expected = (prior["participation"] * prior["signal_fraction"] * jacobian.square()).sum(dim=1) / (
        1. + (prior["participation"] * jacobian.square()).sum(dim=1)
    )
    torch.testing.assert_close(output_gain, expected, rtol=1e-11, atol=1e-12)
    assert bool((output_gain <= signal_fraction + 1e-12).all())
    assert output_gain[1].item() > .9 * signal_fraction[1].item()


@torch.no_grad()
def test_linear_cuda_graph_matches_eager_next_example_predictions():
    from cleanrl.plasticity.predictive_evidence_benchmark_v3 import Args, Runner, configurations

    configure_runtime(matmul_precision="highest", allow_tf32=False)
    args = Args(task="sparse")
    streams = [{"kind": "signal", "alpha": 1.}, {"kind": "pure_noise", "alpha": 0.}]
    configs, groups = configurations(streams)
    eager = Runner(args, 16, streams, 20000, 0, configs, groups)
    compiled = Runner(args, 16, streams, 20000, 0, configs, groups)
    compiled.capture()
    rng = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.rand((100, 16), generator=rng, device="cuda") < .2).float()
    clean = torch.zeros((100, 2), device="cuda")
    clean[:, 0] = x[:, 0]
    targets = clean + torch.randn((100, 2), generator=rng, device="cuda") * .1
    compiled.advance(x, targets, clean)
    for offset in range(100):
        eager.eager_step(x[offset], targets[offset], clean[offset])
    probes = (torch.rand((17, 16), generator=rng, device="cuda") < .2).float()
    probes[0].fill_(1.)
    rows = slice(groups["evidence"].start, None)
    eager_predictions = eager.weight[rows] @ probes.T
    compiled_predictions = compiled.weight[rows] @ probes.T
    assert bool((eager.weight[groups["evidence"]] @ probes.T != 0.).any())
    torch.testing.assert_close(compiled_predictions, eager_predictions, rtol=5e-4, atol=2e-5)
