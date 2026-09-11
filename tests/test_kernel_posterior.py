"""CUDA mathematical contracts for a NEW fixed-basis model; execute through mlq."""

import json
import math

import pytest
import torch

from cleanrl.plasticity import kernel_posterior_v1 as kernel
from cleanrl.plasticity import network_bayes_stream_v2 as v2
from cleanrl.shared import runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def full_precision():
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)


@torch.no_grad()
def test_zero_hazard_matches_batch_bayesian_regression_and_predictive_variance():
    configs = [{"prior": .4, "hazard": 0.0}, {"prior": 2.0, "hazard": 0.0}]
    model = kernel.GaussianPosterior(3, configs, "cuda")
    xs = torch.tensor([[1., .3, -.2], [0., 1., 1.], [1., -1., .4], [.2, .5, 1.]], device="cuda")
    ys = torch.tensor([1.2, -.5, 2., .3], device="cuda")
    noise = torch.tensor([.3, 1.2, .7, .2], device="cuda")
    eye = torch.eye(3, device="cuda", dtype=torch.float64)
    for step, (x, y, r) in enumerate(zip(xs, ys, noise)):
        past_x, past_y, past_r = xs[:step].double(), ys[:step].double(), noise[:step].double()
        expected_means, expected_vars = [], []
        for config in configs:
            precision = eye / config["prior"] + past_x.T @ (past_x / past_r[:, None])
            cov = torch.linalg.inv(precision)
            mean = torch.linalg.solve(precision, past_x.T @ (past_y / past_r))
            expected_means.append(x.double() @ mean)
            expected_vars.append(r.double() + x.double() @ cov @ x.double())
        forecast = model.update(x, y, r)
        torch.testing.assert_close(forecast[0].double(), torch.stack(expected_means), rtol=3e-6, atol=3e-7)
        torch.testing.assert_close(forecast[1].double(), torch.stack(expected_vars), rtol=3e-6, atol=3e-7)
        seen_x, seen_y, seen_r = xs[:step + 1].double(), ys[:step + 1].double(), noise[:step + 1].double()
        for row, config in enumerate(configs):
            precision = eye / config["prior"] + seen_x.T @ (seen_x / seen_r[:, None])
            expected_mean = torch.linalg.solve(precision, seen_x.T @ (seen_y / seen_r))
            torch.testing.assert_close(model.mean[row].double(), expected_mean, rtol=4e-6, atol=3e-7)
            torch.testing.assert_close(model.cov[row].double(), torch.linalg.inv(precision), rtol=4e-6, atol=3e-7)
        assert torch.all(forecast[1] > r)


@torch.no_grad()
def test_redraw_transition_matches_explicit_mixture_moments_before_conditioning():
    configs = [{"prior": 1.7, "hazard": h} for h in (0., .3, 1.)]
    model = kernel.GaussianPosterior(2, configs, "cuda")
    old_mean = torch.tensor([1.3, -.8], device="cuda", dtype=torch.float64)
    old_cov = torch.tensor([[.7, -.2], [-.2, .5]], device="cuda", dtype=torch.float64)
    model.mean.copy_(old_mean.float())
    model.cov.copy_(old_cov.float())
    x = torch.tensor([.4, -1.2], device="cuda")
    y, r = torch.tensor(1.1, device="cuda"), torch.tensor(.6, device="cuda")
    next_means, next_covs = [], []
    predictions, variances = [], []
    conditioned_means, conditioned_covs = [], []
    for config in configs:
        h = config["hazard"]
        component_means = torch.stack((old_mean, torch.zeros_like(old_mean)))
        component_covs = torch.stack((old_cov, config["prior"] * torch.eye(2, device="cuda", dtype=torch.float64)))
        probability = torch.tensor([1 - h, h], device="cuda", dtype=torch.float64)
        mean = (probability[:, None] * component_means).sum(0)
        centered = component_means - mean
        cov = (probability[:, None, None]
               * (component_covs + centered[:, :, None] * centered[:, None, :])).sum(0)
        next_means.append(mean)
        next_covs.append(cov)
        u = cov @ x.double()
        prediction = mean @ x.double()
        variance = r.double() + x.double() @ u
        predictions.append(prediction)
        variances.append(variance)
        conditioned_means.append(mean + u * (y.double() - prediction) / variance)
        conditioned_covs.append(cov - torch.outer(u, u) / variance)
    torch.testing.assert_close(model.mean_weights().double(), torch.stack(next_means), rtol=2e-6, atol=2e-7)
    # Zero-direction conditioning changes no coefficient: expose the full prior
    # covariance rather than inferring it from only one quadratic form.
    transition_only = kernel.GaussianPosterior(2, configs, "cuda")
    transition_only.mean.copy_(model.mean)
    transition_only.cov.copy_(model.cov)
    transition_only.update(torch.zeros_like(x), y, r)
    torch.testing.assert_close(transition_only.mean.double(), torch.stack(next_means), rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(transition_only.cov.double(), torch.stack(next_covs), rtol=2e-6, atol=2e-7)
    actual = model.update(x, y, r)
    torch.testing.assert_close(actual[0].double(), torch.stack(predictions), rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(actual[1].double(), torch.stack(variances), rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(model.mean.double(), torch.stack(conditioned_means), rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(model.cov.double(), torch.stack(conditioned_covs), rtol=3e-6, atol=3e-7)


@torch.no_grad()
def test_predict_is_non_mutating_and_update_forecast_cannot_see_current_label():
    configs = [{"prior": .8, "hazard": .2}]
    left = kernel.GaussianPosterior(2, configs, "cuda")
    right = kernel.GaussianPosterior(2, configs, "cuda")
    x, r = torch.tensor([1., -.4], device="cuda"), torch.tensor(.3, device="cuda")
    for model in (left, right):
        model.update(x, torch.tensor(2., device="cuda"), r)
    before = [tensor.clone() for tensor in left.state_tensors()]
    predicted = left.predict(x, r)
    for _ in range(3):
        left.predict(x, r)
    for actual, expected in zip(left.state_tensors(), before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    left_forecast = left.update(x, torch.tensor(-5., device="cuda"), r)
    right_forecast = right.update(x, torch.tensor(9., device="cuda"), r)
    for left_value, right_value, expected in zip(left_forecast, right_forecast, predicted):
        torch.testing.assert_close(left_value, expected, rtol=0, atol=0)
        torch.testing.assert_close(right_value, expected, rtol=0, atol=0)
    assert not torch.allclose(left.mean, right.mean)
    torch.testing.assert_close(left.cov, right.cov, rtol=0, atol=0)


@torch.no_grad()
def test_joint_evidence_transfers_credit_to_an_unobserved_coordinate():
    model = kernel.GaussianPosterior(2, [{"prior": 1., "hazard": 0.}], "cuda")
    r = torch.tensor(1., device="cuda")
    model.update(torch.tensor([1., 1.], device="cuda"), torch.tensor(0., device="cuda"), r)
    # Joint evidence makes the coefficients anticorrelated. Positive evidence
    # about coordinate 0 must then lower coordinate 1 even when x[1] is zero.
    model.update(torch.tensor([1., 0.], device="cuda"), torch.tensor(1., device="cuda"), r)
    torch.testing.assert_close(model.mean[0], torch.tensor([.4, -.2], device="cuda"), rtol=2e-6, atol=2e-7)
    unseen_mean, unseen_variance = model.predict(torch.tensor([0., 1.], device="cuda"), r)
    torch.testing.assert_close(unseen_mean, torch.tensor([-.2], device="cuda"), rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(unseen_variance, torch.tensor([1.6], device="cuda"), rtol=2e-6, atol=2e-7)


@torch.no_grad()
def test_tangent_features_are_chunk_invariant_and_retain_every_raw_coordinate():
    features = kernel.TangentFeatures(5, projected_dim=7, hidden=3, seed=4, device="cuda")
    x = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.], [0., 0., 1., 0., 0.],
                      [0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.], [1., -.3, .8, 2., -1.]], device="cuda")
    whole = features.transform(x)
    chunks = torch.cat([features.transform(chunk) for chunk in x.split(2)])
    torch.testing.assert_close(chunks, whole, rtol=3e-6, atol=3e-7)
    torch.testing.assert_close(whole[:, :5], x / math.sqrt(5), rtol=0, atol=0)
    repeat = kernel.TangentFeatures(5, projected_dim=7, hidden=3, seed=4, device="cuda")
    torch.testing.assert_close(repeat.transform(x), whole, rtol=0, atol=0)
    # Neither observed suffix values nor suffix batch length can fit/modify phi.
    features.transform(x * 100)
    torch.testing.assert_close(features.transform(x), whole, rtol=0, atol=0)
    metadata = features.metadata()
    assert json.loads(json.dumps(metadata)) == metadata
    assert repeat.metadata()["tensor_sha256"] == metadata["tensor_sha256"]
    different = kernel.TangentFeatures(5, projected_dim=7, hidden=3, seed=5, device="cuda")
    assert different.metadata()["tensor_sha256"] != metadata["tensor_sha256"]


def test_tangent_projection_matches_explicit_autograd_jacobian_including_biases():
    features = kernel.TangentFeatures(3, projected_dim=5, hidden=2, seed=7, device="cuda")
    x = torch.tensor([[.2, -.7, 1.1], [0., 0., 0.], [-.3, .4, .8]], device="cuda")
    weights = tuple(weight.detach().double().requires_grad_() for weight in features.weights)
    explicit_rows = []
    for sample in x.double():
        _, _, out = v2.forward(tuple(weight.unsqueeze(0) for weight in weights), sample[None])
        jacobian = torch.autograd.grad(out.sum(), weights)
        explicit_rows.append(sum((j[:, :, None] * direction.double()).sum((0, 1))
                                 for j, direction in zip(jacobian, features.directions)))
    expected = torch.cat((x.double() / math.sqrt(3), torch.stack(explicit_rows) / math.sqrt(5)), dim=-1)
    actual = features.transform(x)
    torch.testing.assert_close(actual.double(), expected, rtol=4e-6, atol=4e-7)
    posterior = kernel.GaussianPosterior(features.output_dim, [{"prior": 1., "hazard": 0.}], "cuda")
    # Frozen network output is generally nonzero, but it is NOT the prior mean.
    assert v2.forward(tuple(weight.unsqueeze(0) for weight in features.weights), x)[2].abs().max() > .01
    for row in actual:
        mean, variance = posterior.predict(row, torch.tensor(.4, device="cuda"))
        torch.testing.assert_close(mean, torch.zeros_like(mean), rtol=0, atol=0)
        torch.testing.assert_close(variance, (.4 + row.square().sum()).reshape(1), rtol=2e-6, atol=2e-7)


@torch.no_grad()
def test_compiled_feature_and_posterior_graph_replay_resets_all_mutable_state():
    features = kernel.TangentFeatures(2, projected_dim=3, hidden=2, seed=8, device="cuda")
    configs = [{"prior": .7, "hazard": 0.}, {"prior": 1.4, "hazard": .2}]
    model = kernel.GaussianPosterior(features.output_dim, configs, "cuda")
    reference = kernel.GaussianPosterior(features.output_dim, configs, "cuda")
    xs = torch.tensor([[1., -.2], [.3, .7], [-.4, 1.2]], device="cuda")
    ys = torch.tensor([1.3, -.5, .8], device="cuda")
    rs = torch.tensor([.3, .7, .4], device="cuda")
    x, y, r = xs[:1].clone(), ys[0].clone(), rs[0].clone()
    snapshot = [tensor.clone() for tensor in model.state_tensors()]
    pointers = [tensor.data_ptr() for tensor in model.state_tensors()]

    def step():
        phi = features.transform(x)[0]
        return model.update(phi, y, r)

    compiled = torch.compile(step, fullgraph=True, mode="max-autotune-no-cudagraphs")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            compiled()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_mean, captured_variance = compiled()
    for actual, initial in zip(model.state_tensors(), snapshot):
        actual.copy_(initial)
    for xi, yi, ri in zip(xs, ys, rs):
        x.copy_(xi[None])
        y.copy_(yi)
        r.copy_(ri)
        expected_mean, expected_variance = reference.update(features.transform(xi[None])[0], yi, ri)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured_mean, expected_mean, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(captured_variance, expected_variance, rtol=1e-5, atol=1e-6)
        for actual, expected in zip(model.state_tensors(), reference.state_tensors()):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert [tensor.data_ptr() for tensor in model.state_tensors()] == pointers


@torch.no_grad()
def test_per_expert_observation_variance_matches_independent_scalar_filters():
    configs = [{"prior": .8, "hazard": 0.}, {"prior": 1.5, "hazard": .3}]
    batched = kernel.GaussianPosterior(2, configs, "cuda")
    separate = [kernel.GaussianPosterior(2, [config], "cuda") for config in configs]
    xs = torch.tensor([[1., -.3], [.5, 1.2], [-.7, .4]], device="cuda")
    ys = torch.tensor([2., -.8, .3], device="cuda")
    variances = torch.tensor([[.2, 1.3], [.6, .4], [.1, 2.]], device="cuda")
    for x, y, r in zip(xs, ys, variances):
        before = batched.predict(x, r)
        actual = batched.update(x, y, r)
        expected = [model.update(x, y, row_r) for model, row_r in zip(separate, r)]
        for index in range(2):
            torch.testing.assert_close(actual[index], before[index], rtol=0, atol=0)
            torch.testing.assert_close(actual[index], torch.cat([pair[index] for pair in expected]),
                                       rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(batched.mean, torch.cat([model.mean for model in separate]),
                                   rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(batched.cov, torch.cat([model.cov for model in separate]),
                                   rtol=2e-6, atol=2e-7)
