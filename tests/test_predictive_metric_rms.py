"""CUDA numerical contracts for standalone predictive updates; run via mlq."""
import pytest
import torch

from cleanrl.plasticity.predictive_metric_rms_v8 import PredictiveMetricState
from cleanrl.plasticity.predictive_need_v4 import PredictiveNeedState
from cleanrl.shared.runtime import configure_runtime


def _batch_history(state, features, targets):
    n = features.shape[0]
    state.observation_count.fill_(n)
    if n:
        x, y = features.double(), targets.double()
        mx, my = x.mean(0), y.mean(0)
        state.mean_x.copy_(mx)
        state.mean_y.copy_(my)
        state.m2_x.copy_((x - mx).square().sum(0))
        state.m2_y.copy_((y - my).square().sum(0))
        state.cross_xy.copy_(((x - mx) * (y - my)[..., None]).sum(0))


def _fixture():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    t = torch.arange(128, device="cuda", dtype=torch.float64)
    x = torch.stack((torch.sin(t * .7), torch.cos(t * .43), torch.sin(t * 1.13)), 1)[:, None, :]
    y = 2 * x[..., 0] + .15 * torch.cos(t * .23)[:, None]
    weight = torch.tensor([[.2, -.1, .05]], device="cuda", dtype=torch.float64)
    state = PredictiveMetricState(weight)
    _batch_history(state, x, y)
    state.rms_jacobian.copy_(torch.tensor([[.1, .7, 2.]], device="cuda").sqrt())
    return state, weight


def _dense_update(state, weight, jacobian, feature, target, curvature, mode="metric"):
    """Independent dense proximal solve, not the implementation's rank-one formula."""
    prediction = (weight * jacobian).sum(1)
    stats = state.statistics(feature, prediction)
    valid = stats["noise_ready"] & (state.m2_x > 0)
    p = torch.where(valid, stats["participation"], 0.)
    d = p / torch.where(curvature > 0, curvature, 1.)
    root = d.sqrt()
    u = root * jacobian.double()
    identity = torch.eye(weight.shape[1], device="cuda", dtype=torch.float64)
    matrix = identity + u[..., :, None] * u[..., None, :]
    unconstrained = -(prediction - target).double()[:, None] * root * torch.linalg.solve(matrix, u[..., None]).squeeze(-1)
    energy = u.square()
    m = torch.where(valid, stats["error_location"].square() + stats["mean_uncertainty"], 0.)
    r = torch.where(valid, stats["noise"], 0.)
    denominator = (energy * (m + r)).sum(1, keepdim=True)
    fraction = (energy * m).sum(1, keepdim=True) / torch.where(denominator > 0, denominator, 1.)
    confidence = torch.where(jacobian != 0, p, 0.).amax(1, keepdim=True)
    if mode == "local_need":
        fraction = stats["write_fraction"]
    elif mode == "uncapped":
        confidence = torch.ones_like(confidence)
    elif mode == "no_need":
        fraction = torch.ones_like(fraction)
    return weight + (confidence * fraction * unconstrained).to(weight.dtype)


@torch.no_grad()
def test_next_predictions_match_independent_dense_proximal_risk_solution():
    state, weight = _fixture()
    j = torch.tensor([[1., -.3, .2]], device="cuda", dtype=torch.float64)
    feature = j.tanh()
    target = torch.tensor([1.7], device="cuda", dtype=torch.float64)
    prediction = (weight * j).sum(1)
    residual = prediction - target
    curvature = (128 * state.rms_jacobian.square() + j.square()) / 129
    expected = _dense_update(state, weight, j, feature, target, curvature)
    actual, _ = state.transition(weight, residual[:, None] * j, j, feature, target, residual, prediction)
    probes = torch.tensor([[1., 1., 1.], [-.4, 1., .2], [.7, -.2, 2.]], device="cuda", dtype=torch.float64)
    torch.testing.assert_close(actual @ probes.T, expected @ probes.T, rtol=1e-11, atol=1e-12)


@torch.no_grad()
def test_small_current_jacobian_does_not_create_large_future_prediction():
    state, weight = _fixture()
    weight.zero_()
    state.rms_jacobian.fill_(1.)
    j = torch.tensor([[1e-9, 0., 0.]], device="cuda", dtype=torch.float64)
    feature = torch.tensor([[.8, 0., 0.]], device="cuda", dtype=torch.float64)
    target = torch.tensor([2.], device="cuda", dtype=torch.float64)
    prediction = (weight * j).sum(1)
    residual = prediction - target
    actual, _ = state.transition(weight, residual[:, None] * j, j, feature, target, residual, prediction)
    # On a later unit-Jacobian query the new prediction must still be tiny;
    # bare inverse-current-J normalization would create an O(1e9) change.
    assert abs(actual[0, 0].item()) < 1e-8
    torch.testing.assert_close(actual[0, 1:], weight[0, 1:], rtol=0, atol=0)


@pytest.mark.parametrize("compiled", (False, True))
@torch.no_grad()
def test_consistent_coordinate_rescaling_preserves_future_predictions(compiled):
    state, weight = _fixture()
    scaled, _ = _fixture()
    scale = torch.tensor([[1e-155, 1e155, 4.]], device="cuda", dtype=torch.float64)
    scaled.rms_jacobian.mul_(scale.abs())
    j = torch.tensor([[1., -.3, .2]], device="cuda", dtype=torch.float64)
    feature, target = j.tanh(), torch.tensor([1.7], device="cuda", dtype=torch.float64)
    prediction = (weight * j).sum(1)
    residual = prediction - target
    base_transition = state.transition
    scaled_transition = scaled.transition
    if compiled:
        base_transition = torch.compile(base_transition, fullgraph=True, options={"triton.cudagraphs": False})
        scaled_transition = torch.compile(scaled_transition, fullgraph=True, options={"triton.cudagraphs": False})
    original, _ = base_transition(weight, residual[:, None] * j, j, feature, target, residual, prediction)
    transformed, _ = scaled_transition(weight / scale, residual[:, None] * j * scale, j * scale,
                                       feature, target, residual, prediction)
    probes = torch.tensor([[1., 1., 1.], [-.4, 1., .2]], device="cuda", dtype=torch.float64)
    torch.testing.assert_close(original @ probes.T, transformed @ (probes * scale).T, rtol=1e-10, atol=1e-12)


@torch.no_grad()
def test_current_target_cannot_change_its_own_linear_response():
    state, weight = _fixture()
    j = torch.tensor([[1., -.3, .2]], device="cuda", dtype=torch.float64)
    prediction = (weight * j).sum(1)
    responses = []
    for value in (-5., 8.):
        target = torch.tensor([value], device="cuda", dtype=torch.float64)
        residual = prediction - target
        new, _ = state.transition(weight, residual[:, None] * j, j, j.tanh(), target, residual, prediction)
        responses.append(((new - weight) * j).sum(1) / residual)
    torch.testing.assert_close(responses[0], responses[1], rtol=1e-11, atol=1e-12)
    assert -1 < responses[0].item() < 0


@torch.no_grad()
def test_cuda_graph_next_predictions_match_independent_sequential_updates():
    from cleanrl.plasticity.predictive_metric_rms_benchmark_v8 import Args, CHUNK, Runner, configurations

    configure_runtime(matmul_precision="highest", allow_tf32=False)
    args = Args(task="sparse")
    streams = [{"kind": "signal", "alpha": 1.}, {"kind": "pure_noise", "alpha": 0.}]
    configs, groups = configurations(streams)
    runner = Runner(args, 3, streams, CHUNK, 0, configs, groups)
    runner.capture()
    t = torch.arange(CHUNK, device="cuda")
    x = torch.stack(((t % 2) == 0, (t % 3) == 0, (t % 7) == 0), 1).float()
    targets = torch.stack((1.25 * x[:, 0] + .07 * torch.sin(t.float()), .3 * torch.cos(t.float() * .71)), 1)
    expected = {mode: torch.zeros((2, 3), device="cuda") for mode in groups if mode not in ("sgd", "adamw")}
    states = {mode: PredictiveNeedState(w) for mode, w in expected.items()}
    for index in range(CHUNK):
        j = x[index].expand(2, -1)
        feature = j.tanh()
        curvature = x[:index + 1].double().square().mean(0).expand(2, -1)
        for mode, w in expected.items():
            state = states[mode]
            history = x[:index].tanh()[:, None, :].expand(-1, 2, -1)
            _batch_history(state, history, targets[:index])
            if mode == "v4_reference":
                prediction = (w * j).sum(1)
                residual = prediction - targets[index]
                state.step(w, residual[:, None] * j, j, feature, targets[index], residual, prediction)
            else:
                w.copy_(_dense_update(state, w, j, feature, targets[index], curvature, mode))
    runner.advance(x, targets, torch.zeros_like(targets))
    probes = torch.tensor([[1., 0., 0.], [0., 1., 1.], [1., 1., 1.]], device="cuda")
    for mode, weight in expected.items():
        torch.testing.assert_close(runner.weight[groups[mode]] @ probes.T, weight @ probes.T,
                                   rtol=3e-4, atol=2e-5)
