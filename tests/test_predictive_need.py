"""CUDA numerical contracts; execution is queued through mlq, never on CPU."""
import math

import pytest
import torch

from cleanrl.plasticity.predictive_need_v4 import (
    PredictiveNeedState,
    fit_inclusion_probability,
)
from cleanrl.shared.runtime import configure_runtime


CONTROLS = ("predictive_need", "signal_strength", "no_epistemic", "fixed_prior", "residual_statistics")


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
    # An inactive predictor has both zero forward output and zero Jacobian.
    prediction = (weight * absent).sum(dim=1)
    for x, y in zip(feature, target):
        residual = prediction - y
        state.step(weight, residual[:, None] * absent, absent, x, y, residual, prediction)


def _dense_posterior(feature, target, query):
    """Integrate a g=n slope in observation space, with a flat intercept.

    Each coordinate defines a separate local H1. This reference uses dense
    covariance solves, not the controller's centered R-squared identities.
    """
    n, models, coordinates = feature.shape
    result = {key: torch.empty_like(query) for key in ("log_bf", "location", "noise", "uncertainty")}
    identity = torch.eye(n, dtype=feature.dtype, device=feature.device)
    for model in range(models):
        y = target[:, model]
        centered_y = y - y.mean()
        null_energy = centered_y @ centered_y
        for coordinate in range(coordinates):
            x = feature[:, model, coordinate]
            centered_x = x - x.mean()
            contrast = query[model, coordinate] - x.mean()
            prior_slope_variance = n / (centered_x @ centered_x)
            covariance = identity + prior_slope_variance * torch.outer(centered_x, centered_x)
            cross_covariance = prior_slope_variance * centered_x * contrast
            solved_y = torch.linalg.solve(covariance, centered_y)
            energy = centered_y @ solved_y
            noise = energy / (n - 3)
            latent_variance = (
                1. / n + prior_slope_variance * contrast.square()
                - cross_covariance @ torch.linalg.solve(covariance, cross_covariance)
            )
            result["log_bf"][model, coordinate] = (
                -.5 * torch.linalg.slogdet(covariance).logabsdet
                - (n - 1) / 2 * torch.log(energy / null_energy)
            )
            result["location"][model, coordinate] = y.mean() + cross_covariance @ solved_y
            result["noise"][model, coordinate] = noise
            result["uncertainty"][model, coordinate] = noise * latent_variance
    return result


@torch.no_grad()
def test_welford_centered_moments_match_independent_batch_computation():
    feature, target = _examples()
    feature = torch.cat((feature, .5 * feature + .125), dim=1)
    feature = torch.cat((feature, torch.ones_like(feature[..., :1]) * .25), dim=-1)
    target = torch.cat((target, 100000. - 2. * target), dim=1)
    weight = torch.zeros_like(feature[0])
    state = PredictiveNeedState(weight)
    _observe_without_writes(state, weight, feature, target)
    centered_x = feature - feature.mean(dim=0)
    centered_y = target - target.mean(dim=0)
    torch.testing.assert_close(state.mean_x, feature.mean(dim=0), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(state.mean_y, target.mean(dim=0), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(state.m2_x, centered_x.square().sum(dim=0), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(state.m2_y, centered_y.square().sum(dim=0), rtol=1e-10, atol=1e-9)
    torch.testing.assert_close(
        state.cross_xy, (centered_x * centered_y[..., None]).sum(dim=0), rtol=1e-10, atol=1e-9,
    )


@torch.no_grad()
def test_dense_h1_posterior_and_expected_current_error_shrinkage():
    feature, target = _examples()
    weight = torch.zeros_like(feature[0])
    state = PredictiveNeedState(weight, control="fixed_prior")
    _observe_without_writes(state, weight, feature, target)
    query = feature.new_tensor([[.4, -.3, .2]])
    prediction = feature.new_tensor([2.75])
    actual = state.statistics(query, prediction)
    reference = _dense_posterior(feature, target, query)
    for key, expected in (
        ("log_bf", reference["log_bf"]),
        ("participation", reference["log_bf"].sigmoid()),
        ("predictive_location", reference["location"]),
        ("noise", reference["noise"]),
        ("mean_uncertainty", reference["uncertainty"]),
        ("error_location", prediction[:, None] - reference["location"]),
    ):
        torch.testing.assert_close(actual[key], expected, rtol=1e-10, atol=1e-12)

    mean_error = prediction[:, None] - reference["location"]
    second_moment = mean_error.square() + reference["uncertainty"]
    expected_fraction = second_moment / (second_moment + reference["noise"])
    torch.testing.assert_close(actual["write_fraction"], expected_fraction, rtol=1e-10, atol=1e-12)

    # Four nodes match the joint first/second moments needed by this quadratic
    # risk; marginal independence after integrating the scale is not required.
    signs = feature.new_tensor([-1., 1.])
    latent_mean = reference["location"][None, None] + signs[:, None, None, None] * reference["uncertainty"].sqrt()
    observed_target = latent_mean + signs[None, :, None, None] * reference["noise"].sqrt()

    def risk(fraction):
        updated_prediction = prediction[None, None, :, None] - fraction * (
            prediction[None, None, :, None] - observed_target
        )
        return (updated_prediction - latent_mean).square().mean(dim=(0, 1))

    optimum = risk(actual["write_fraction"])
    torch.testing.assert_close(
        optimum, second_moment * reference["noise"] / (second_moment + reference["noise"]),
        rtol=1e-10, atol=1e-12,
    )
    for candidate in (0., 1., expected_fraction - .1, expected_fraction + .1):
        assert bool((optimum < risk(candidate)).all())


@torch.no_grad()
def test_same_association_requires_less_plasticity_at_matching_prediction():
    feature = torch.tensor([-.5, .5] * 32, dtype=torch.float64, device="cuda").reshape(-1, 1, 1)
    noise = feature.new_tensor([-.1, -.1, .1, .1] * 16).reshape(-1, 1)
    target = 2. + 3. * feature[..., 0] + noise
    query = feature.new_tensor([[.5]])
    reference = _dense_posterior(feature, target, query)
    location = reference["location"][:, 0]
    for control in ("predictive_need", "no_epistemic", "signal_strength"):
        weight = torch.zeros_like(query)
        state = PredictiveNeedState(weight, control=control)
        _observe_without_writes(state, weight, feature, target)
        # Use the learned location for exact matching, after checking the dense oracle.
        learned = state.statistics(query, location)
        torch.testing.assert_close(learned["predictive_location"], reference["location"], rtol=1e-11, atol=1e-12)
        matching_weight = learned["predictive_location"] / query
        far_weight = matching_weight + 10. / query
        matching_prediction = (matching_weight * query).sum(dim=1)
        far_prediction = (far_weight * query).sum(dim=1)
        matching = state.statistics(query, matching_prediction)
        far = state.statistics(query, far_prediction)
        for key in ("log_bf", "participation", "predictive_location", "mean_uncertainty", "noise"):
            torch.testing.assert_close(matching[key], far[key], rtol=0, atol=0)
        assert matching["participation"].item() > .99
        if control == "signal_strength":
            torch.testing.assert_close(matching["write_fraction"], far["write_fraction"], rtol=0, atol=0)
            assert matching["write_fraction"].item() > .9
        else:
            assert matching["write_fraction"].item() < .05 * far["write_fraction"].item()
            assert far["write_fraction"].item() > .99
            if control == "no_epistemic":
                assert matching["write_fraction"].item() == 0.
            else:
                assert 0. < matching["mean_uncertainty"].item() < float("inf")
                assert matching["write_fraction"].item() > 0.
        y = location + reference["noise"][:, 0].sqrt()
        for before, prediction, statistics in (
            (matching_weight, matching_prediction, matching), (far_weight, far_prediction, far),
        ):
            residual = prediction - y
            gradient = residual[:, None] * query
            after, _ = state.transition(before, gradient, query, query, y, residual, prediction)
            next_prediction = (after * query).sum(dim=1)
            output_gain = (prediction - next_prediction) / residual
            expected_gain = (
                statistics["participation"] * statistics["write_fraction"] * query.square()
                / (1. + statistics["participation"] * query.square())
            )[:, 0]
            torch.testing.assert_close(output_gain, expected_gain, rtol=1e-10, atol=1e-12)


@torch.no_grad()
def test_signal_strength_control_matches_dense_model_averaged_noise_reference():
    feature, target = _examples()
    weight = torch.zeros_like(feature[0])
    state = PredictiveNeedState(weight, control="signal_strength")
    _observe_without_writes(state, weight, feature, target)
    query = feature.new_tensor([[.4, -.3, .2]])
    actual = state.statistics(query, feature.new_tensor([7.]))
    reference = _dense_posterior(feature, target, query)
    null_noise = (target - target.mean(dim=0)).square().sum(dim=0)[:, None] / (feature.shape[0] - 3)
    mixture_noise = (1. - actual["participation"]) * null_noise + actual["participation"] * reference["noise"]
    effect = reference["location"] - target.mean(dim=0)[:, None]
    torch.testing.assert_close(actual["noise"], mixture_noise, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(
        actual["write_fraction"], effect.square() / (effect.square() + mixture_noise), rtol=1e-10, atol=1e-12,
    )


@torch.no_grad()
def test_target_and_input_rescaling_preserve_evidence_and_predictive_need():
    feature, target = _examples()
    scale = feature.new_tensor([-.25, 1.5, -1.])
    offset = feature.new_tensor([.1, -.2, .05])
    query = feature.new_tensor([[.4, -.3, .2]])
    prediction = feature.new_tensor([2.75])
    statistics = []
    for x, y, probe, output in (
        (feature, target, query, prediction),
        (feature * scale + offset, -7. * target + 9., query * scale + offset, -7. * prediction + 9.),
    ):
        weight = torch.zeros_like(x[0])
        state = PredictiveNeedState(weight)
        _observe_without_writes(state, weight, x, y)
        statistics.append(state.statistics(probe, output))
    original, rescaled = statistics
    for key in ("log_bf", "prior_probability", "participation", "write_fraction"):
        torch.testing.assert_close(rescaled[key], original[key], rtol=1e-9, atol=1e-10)
    for key in ("effect", "error_location"):
        torch.testing.assert_close(rescaled[key], -7. * original[key], rtol=1e-11, atol=1e-12)
    for key in ("noise", "mean_uncertainty"):
        torch.testing.assert_close(rescaled[key], 49. * original[key], rtol=1e-11, atol=1e-12)


@torch.no_grad()
def test_inclusion_mle_resolves_interior_boundary_flat_and_extreme_evidence():
    log_bf = torch.tensor(
        [[math.log(4.), math.log(.25)], [math.log(9.), math.log(1. / 3.)],
         [math.log(2.), math.log(4.)], [math.log(.5), math.log(.25)], [0., 0.], [1000., -1000.]],
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
def test_current_target_cannot_change_its_own_prior_gain_or_erase_actual_prediction():
    feature, target = _examples()
    weight = feature.new_tensor([[1., -2., .5]])
    state = PredictiveNeedState(weight)
    _observe_without_writes(state, weight, feature, target)
    query = weight.new_tensor([[.4, -.3, .2]])
    jacobian = weight.new_tensor([[1., -.5, .25]])
    prediction = (weight * jacobian).sum(dim=1)
    prior = state.statistics(query, prediction)
    gain = prior["participation"] * prior["write_fraction"] / (
        1. + (prior["participation"] * jacobian.square()).sum(dim=1, keepdim=True)
    )
    assert gain.sum().item() > 0.
    for current_target in (-100., 100., 2. ** 60):
        y = weight.new_tensor([current_target])
        residual = prediction - y
        gradient = residual[:, None] * jacobian
        if current_target == 2. ** 60:
            # Reconstructing the forward value as y + residual loses it entirely.
            assert (y + residual).item() == 0.
            assert prediction.item() != 0.
        updated_weight, _ = state.transition(weight, gradient, jacobian, query, y, residual, prediction)
        torch.testing.assert_close(updated_weight, weight - gain * gradient, rtol=1e-11, atol=1e-12)


@torch.no_grad()
def test_residual_statistics_models_error_not_a_second_target_prediction():
    feature, target = _examples()
    weight = torch.zeros_like(feature[0])
    state = PredictiveNeedState(weight, control="residual_statistics")
    _observe_without_writes(state, weight, feature, target)
    query = weight.new_tensor([[.4, -.3, .2]])
    reference = _dense_posterior(feature, -target, query)
    for prediction in (weight.new_tensor([0.]), weight.new_tensor([50.])):
        actual = state.statistics(query, prediction)
        torch.testing.assert_close(actual["error_location"], reference["location"], rtol=1e-10, atol=1e-12)
        second_moment = reference["location"].square() + reference["uncertainty"]
        torch.testing.assert_close(
            actual["write_fraction"], second_moment / (second_moment + reference["noise"]),
            rtol=1e-10, atol=1e-12,
        )


@torch.no_grad()
def test_zero_jacobian_coordinates_never_write_with_valid_gradients():
    feature, target = _examples()
    weight = feature.new_tensor([[1., -2., .5]])
    state = PredictiveNeedState(weight, control="fixed_prior")
    _observe_without_writes(state, weight, feature, target)
    query = weight.new_tensor([[.4, -.3, .2]])
    jacobian = weight.new_tensor([[0., 1., -.5]])
    y = weight.new_tensor([7.])
    prediction = (weight * jacobian).sum(dim=1)
    residual = prediction - y
    updated_weight, _ = state.transition(weight, residual[:, None] * jacobian, jacobian, query, y, residual, prediction)
    assert updated_weight[0, 0].item() == weight[0, 0].item()
    assert bool((updated_weight[:, 1:] != weight[:, 1:]).any())
    absent = torch.zeros_like(weight)
    prediction = (weight * absent).sum(dim=1)
    residual = prediction - y
    unchanged, _ = state.transition(weight, residual[:, None] * absent, absent, query, y, residual, prediction)
    torch.testing.assert_close(unchanged, weight, rtol=0, atol=0)


@pytest.mark.parametrize(("control", "history"), [
    ("predictive_need", "empty"),
    ("fixed_prior", "three"),
    ("signal_strength", "three"),
    ("no_epistemic", "constant_feature"),
    ("signal_strength", "constant_feature"),
    ("fixed_prior", "constant_target"),
    ("signal_strength", "constant_target"),
    ("residual_statistics", "constant_target"),
])
@torch.no_grad()
def test_unavailable_or_degenerate_moments_cannot_create_writes_or_nans(control, history):
    feature, target = _examples()
    if history in ("empty", "three"):
        count = {"empty": 0, "three": 3}[history]
        feature, target = feature[:count], target[:count]
    elif history == "constant_feature":
        feature = torch.full_like(feature, .25)
    else:
        target = torch.full_like(target, 3.)
    weight = torch.tensor([[1., -2., .5]], dtype=torch.float64, device="cuda")
    state = PredictiveNeedState(weight, control=control)
    _observe_without_writes(state, weight, feature, target)
    query = weight.new_tensor([[.4, -.3, .2]])
    prediction = (weight * query).sum(dim=1)
    prior = state.statistics(query, prediction)
    assert bool(torch.isinf(prior["mean_uncertainty"]).all())
    torch.testing.assert_close(prior["write_fraction"], torch.zeros_like(weight), rtol=0, atol=0)
    assert all(not bool(torch.isnan(value).any()) for value in prior.values())
    y = weight.new_tensor([7.])
    residual = prediction - y
    after, updated = state.transition(weight, residual[:, None] * query, query, query, y, residual, prediction)
    torch.testing.assert_close(after, weight, rtol=0, atol=0)
    assert all(bool(torch.isfinite(value).all()) for value in updated.values())


@torch.no_grad()
def test_many_participating_coordinates_cannot_amplify_predictive_need_gain():
    weight = torch.zeros((2, 1024), dtype=torch.float64, device="cuda")
    state = PredictiveNeedState(weight, control="fixed_prior")
    x = weight.new_tensor([-.5, .5] * 4)
    noise = weight.new_tensor([-1., -1., 1., 1.] * 2)
    feature = x[:, None, None].expand(-1, *weight.shape)
    target = (.2 * x + noise)[:, None].expand(-1, weight.shape[0])
    _observe_without_writes(state, weight, feature, target)
    query = torch.full_like(weight, .5)
    jacobian = torch.ones_like(weight)
    jacobian[0, 1:] = 0.
    prediction = (weight * jacobian).sum(dim=1)
    prior = state.statistics(query, prediction)
    fraction = prior["write_fraction"].max(dim=1).values
    assert bool(((fraction > 0.) & (fraction < 1.)).all())
    y = weight.new_tensor([-1., -1.])
    residual = prediction - y
    updated_weight, _ = state.transition(weight, residual[:, None] * jacobian, jacobian, query, y, residual, prediction)
    output_gain = -((updated_weight - weight) * jacobian).sum(dim=1) / residual
    expected = (prior["participation"] * prior["write_fraction"] * jacobian.square()).sum(dim=1) / (
        1. + (prior["participation"] * jacobian.square()).sum(dim=1)
    )
    torch.testing.assert_close(output_gain, expected, rtol=1e-11, atol=1e-12)
    assert bool((output_gain <= fraction + 1e-12).all())
    assert output_gain[1].item() > .9 * fraction[1].item()


@torch.no_grad()
def test_linear_cuda_graph_and_eager_use_preupdate_forward_for_next_predictions():
    from cleanrl.plasticity.predictive_need_benchmark_v4 import Args, Runner, configurations

    configure_runtime(matmul_precision="highest", allow_tf32=False)
    args = Args(task="sparse")
    streams = [{"kind": "signal", "alpha": 1.}, {"kind": "pure_noise", "alpha": 0.}]
    configs, groups = configurations(streams)
    eager = Runner(args, 16, streams, 20000, 0, configs, groups)
    compiled = Runner(args, 16, streams, 20000, 0, configs, groups)
    compiled.capture()
    reference_weights = {control: torch.zeros_like(eager.weight[groups[control]]) for control in CONTROLS}
    reference_states = {control: PredictiveNeedState(weight, control=control) for control, weight in reference_weights.items()}
    rng = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.rand((100, 16), generator=rng, device="cuda") < .2).float()
    clean = torch.zeros((100, 2), device="cuda")
    clean[:, 0] = x[:, 0]
    targets = clean + torch.randn((100, 2), generator=rng, device="cuda") * .1
    # The final target exposes lossy target+residual reconstruction in float32.
    x[-1].fill_(1.)
    clean[-1, 0] = 1.
    targets[-1].fill_(2. ** 30)
    compiled.advance(x, targets, clean)
    for offset in range(100):
        eager.eager_step(x[offset], targets[offset], clean[offset])
        for control, weight in reference_weights.items():
            rows = groups[control]
            prediction = (weight * x[offset]).sum(dim=1)
            y = targets[offset][eager.stream_index[rows]]
            residual = prediction - y
            jacobian = x[offset].expand_as(weight)
            reference_states[control].step(
                weight, residual[:, None] * jacobian, jacobian,
                x[offset].tanh().expand_as(weight), y, residual, prediction,
            )
            if offset + 1 < len(x):
                torch.testing.assert_close(
                    eager.weight[rows] @ x[offset + 1], weight @ x[offset + 1], rtol=5e-4, atol=2e-5,
                )
    probes = (torch.rand((17, 16), generator=rng, device="cuda") < .2).float()
    probes[0].fill_(1.)
    for control, weight in reference_weights.items():
        expected_predictions = weight @ probes.T
        for runner in (eager, compiled):
            torch.testing.assert_close(
                runner.weight[groups[control]] @ probes.T, expected_predictions, rtol=5e-4, atol=2e-5,
            )
    assert bool((reference_weights["predictive_need"] @ probes.T != 0.).any())
