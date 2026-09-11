"""CUDA contract checks; run through mlq, never as a shortened learning run."""

import bisect

import pytest
import torch
from torch.nn import functional as F

from cleanrl.plasticity.panel_distributional_model_v1 import Config, FAMILIES, Learner
from cleanrl.plasticity.panel_hd_gate import Net
from cleanrl.shared import runtime


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make(configs=None, *, samples=9, bins=9):
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    if configs is None:
        configs = [Config(family, lr) for family in FAMILIES for lr in (1e-3, 3e-3)]
    return Learner(5, 7, 1.3, configs, "cuda", bins=bins, num_samples=samples)


def independent_labels(target, support):
    """CPU reference interpolation, independent of the shared torch projector."""
    centers = support.detach().cpu().tolist()
    rows = []
    for value in target.detach().cpu().tolist():
        row = [0.0] * len(centers)
        upper = bisect.bisect_right(centers, value)
        if upper == 0:
            row[0] = 1.0
        elif upper == len(centers):
            row[-1] = 1.0
        else:
            lower = upper - 1
            fraction = (value - centers[lower]) / (centers[upper] - centers[lower])
            row[lower], row[upper] = 1.0 - fraction, fraction
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32, device=target.device)


class AutogradReference:
    """Independent dense autograd and torch.optim.Adam, with post-Adam gating."""

    def __init__(self, learner, index):
        self.config = learner.configs[index]
        self.parameters = [torch.nn.Parameter(p[index].clone()) for p in learner.parameters]
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.config.lr, foreach=False, fused=False)
        self.support = learner.support.clone()
        self.kappa = learner.kappa
        self.s1 = [torch.zeros_like(p) for p in self.parameters]
        self.s2 = [torch.zeros_like(p) for p in self.parameters]

    def step(self, x, y, mask):
        w1, b1, w2, b2, w3, b3 = self.parameters
        h = torch.tanh(F.linear(torch.tanh(F.linear(x, w1, b1)), w2, b2))
        logits = F.linear(h, w3, b3)
        if self.config.family.startswith("scalar"):
            prediction = logits[:, 0]
        else:
            prediction = (logits.softmax(-1) * self.support).sum(-1)
        if not bool(mask.any()):
            return prediction.detach()
        if self.config.family in ("categorical_ce", "categorical_ce_js"):
            labels = independent_labels(y[mask], self.support)
            loss = -(labels * logits[mask].log_softmax(-1)).sum(-1).mean()
        else:
            loss = (prediction[mask] - y[mask]).square().mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        old = [p.detach().clone() for p in self.parameters]
        self.optimizer.step()
        with torch.no_grad():
            for p, before, s1, s2 in zip(self.parameters, old, self.s1, self.s2):
                if self.config.family.endswith("_js"):
                    # Independent equivalent statistic; gate observes strictly
                    # earlier gradients, not the current optimizer gradient.
                    t_squared = s1.square() / s2.clamp_min(torch.finfo(torch.float32).tiny)
                    gate = (1.0 - self.kappa / t_squared.clamp_min(torch.finfo(torch.float32).tiny)).clamp_min(0.0)
                    p.copy_(before + gate * (p - before))
                s1.add_(p.grad)
                s2.add_(p.grad.square())
        return prediction.detach()


def stream():
    generator = torch.Generator(device="cuda").manual_seed(19)
    for index in range(12):
        x = torch.randn((9, 5), generator=generator, device="cuda")
        y = torch.rand(9, generator=generator, device="cuda") * 25.0 - 1.3
        mask = torch.rand(9, generator=generator, device="cuda") > 0.3
        mask[0] = True
        if index == 4:
            mask.zero_()
        elif index == 7:
            mask.zero_()
            mask[3] = True
        yield x, y, mask


def test_initialization_matches_frozen_scalar_and_shared_trunks():
    learner = make()
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(1)
        baseline = Net(5, 7)
    baseline = baseline.cuda()
    for index, config in enumerate(learner.configs):
        for actual, expected in zip(learner.parameters[:4], list(baseline.parameters())[:4]):
            torch.testing.assert_close(actual[index], expected, rtol=0, atol=0)
        if config.family.startswith("scalar"):
            torch.testing.assert_close(learner.weights[2][index, :1], baseline.l3.weight, rtol=0, atol=0)
            torch.testing.assert_close(learner.biases[2][index, :1], baseline.l3.bias, rtol=0, atol=0)
            assert torch.count_nonzero(learner.weights[2][index, 1:]) == 0
            assert torch.count_nonzero(learner.biases[2][index, 1:]) == 0
        else:
            torch.testing.assert_close(learner.weights[2][index], learner.weights[2][4], rtol=0, atol=0)
            raw_support = learner.support + 1.3
            torch.testing.assert_close(learner.biases[2][index], -raw_support, rtol=0, atol=2e-6)
    x, y, mask = next(stream())
    prediction = learner.step(x, y, mask)
    torch.testing.assert_close(prediction[0], baseline(x)[0], rtol=2e-6, atol=2e-7)


def test_manual_trajectories_match_autograd_adam_and_prior_history_js():
    learner = make()
    references = [AutogradReference(learner, i) for i in range(len(learner.configs))]
    for x, y, mask in stream():
        actual = learner.step(x, y, mask)
        for index, reference in enumerate(references):
            expected = reference.step(x, y, mask)
            torch.testing.assert_close(actual[index], expected, rtol=1e-4, atol=5e-6)
            for parameter, first, second, s1, s2, ref_parameter, ref_s1, ref_s2 in zip(
                learner.parameters, learner.first_moments, learner.second_moments,
                learner.s1, learner.s2, reference.parameters, reference.s1, reference.s2
            ):
                state = reference.optimizer.state[ref_parameter]
                torch.testing.assert_close(parameter[index], ref_parameter, rtol=2e-4, atol=3e-6)
                torch.testing.assert_close(first[index], state["exp_avg"], rtol=2e-4, atol=5e-6)
                torch.testing.assert_close(second[index], state["exp_avg_sq"], rtol=3e-4, atol=5e-7)
                torch.testing.assert_close(s1[index], ref_s1, rtol=2e-4, atol=5e-5)
                torch.testing.assert_close(s2[index], ref_s2, rtol=3e-4, atol=5e-4)
    assert int(learner.steps) == 12
    assert int(learner.adam_steps) == 11


def test_projection_preserves_probability_and_target_expectation():
    learner = make(bins=33)
    centers = learner.support
    target = torch.cat((centers, 0.17 * centers[:-1] + 0.83 * centers[1:]))
    labels = learner.projector.project(target)
    torch.testing.assert_close(labels, independent_labels(target, centers), rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(labels.sum(-1), torch.ones_like(target), rtol=0, atol=1e-7)
    torch.testing.assert_close(labels @ centers, target, rtol=1e-6, atol=2e-6)
    assert bool((labels >= 0).all())
    assert bool(((labels > 0).sum(-1) <= 2).all())
    torch.testing.assert_close(labels[:33], torch.eye(33, device="cuda"), rtol=0, atol=0)
    torch.testing.assert_close(centers[[0, -1]], torch.tensor([-1.3, 23.7], device="cuda"), rtol=0, atol=1e-6)


def test_invalid_samples_do_not_affect_updates_and_empty_bar_skips_optimizer():
    left, right = make(), make()
    x, y, _ = next(stream())
    mask = torch.tensor([True, False, True, False, False, True, True, False, True], device="cuda")
    alternate_x, alternate_y = x.clone(), y.clone()
    alternate_x[~mask] = 100.0
    alternate_y[~mask] = float("nan")
    for _ in range(3):
        left_prediction = left.step(x, y, mask)
        right_prediction = right.step(alternate_x, alternate_y, mask)
        torch.testing.assert_close(left_prediction[:, mask], right_prediction[:, mask], rtol=0, atol=0)
        for a, b in zip(left.state_tensors()[:-1], right.state_tensors()[:-1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    before = [t.clone() for t in left.state_tensors()]
    left.step(x, torch.full_like(y, float("nan")), torch.zeros_like(mask))
    # Parameters, moments and evidence are unchanged; only consumed-bar clock and
    # current predictions may change. Nonempty-bar Adam clock stays unchanged.
    for actual, expected in zip(left.state_tensors()[:-3], before[:-3]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(left.adam_steps, before[-2], rtol=0, atol=0)
    assert int(left.steps) == int(before[-3]) + 1


def test_current_label_cannot_affect_current_forecast_but_changes_future():
    left, right = make(), make()
    x, y, mask = next(stream())
    for _ in range(3):
        left.step(x, y, mask)
        right.step(x, y, mask)
    a = left.step(x, torch.zeros_like(y), mask).clone()
    b = right.step(x, torch.full_like(y, 10.0), mask).clone()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    future_a = left.step(x, y, mask).clone()
    future_b = right.step(x, y, mask).clone()
    assert bool(((future_a - future_b).abs().max(-1).values > 1e-7).all())


def test_js_uses_previous_evidence_and_does_not_gate_adam_moments():
    learner = make([Config("scalar_adam", 1e-3), Config("scalar_js", 1e-3)])
    initial = [p[1].clone() for p in learner.parameters]
    x, y, mask = next(stream())
    for _ in range(2):
        learner.step(x, y, mask)
        for actual, expected in zip(learner.parameters, initial):
            torch.testing.assert_close(actual[1], expected, rtol=0, atol=0)
    assert any(bool(moment[1].ne(0).any()) for moment in learner.first_moments)
    learner.step(x, y, mask)
    assert any(bool((actual[1] - expected).abs().max() > 1e-7) for actual, expected in zip(learner.parameters, initial))


def test_fullgraph_capture_restores_all_state_and_replays_bitwise():
    learner = make()
    x, y, mask = next(stream())
    compiled = torch.compile(learner.step, fullgraph=True, options={"triton.cudagraphs": False})
    mutable = learner.state_tensors()
    assert len({t.data_ptr() for t in mutable}) == len(mutable)
    initial = [t.clone() for t in mutable]

    def restore(saved):
        for tensor, value in zip(mutable, saved):
            tensor.copy_(value)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            compiled(x, y, mask)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    restore(initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = compiled(x, y, mask)
    restore(initial)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    expected = [t.clone() for t in mutable]
    restore(initial)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    for actual, value in zip(mutable, expected):
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
    torch.testing.assert_close(result, learner.prediction, rtol=0, atol=0)
    reference = make()
    reference.step(x, y, mask)
    reference.step(x, y, mask)
    for actual, value in zip(mutable, reference.state_tensors()):
        torch.testing.assert_close(actual, value, rtol=2e-4, atol=3e-6)
    assert int(learner.steps) == 2
