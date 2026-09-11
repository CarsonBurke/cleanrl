"""Frontier contracts; CUDA execution belongs to the parent's queued validation."""

import pytest
import torch
from torch.nn import functional as F

from cleanrl.plasticity import panel_distributional_model_v1 as frozen
from cleanrl.plasticity.panel_frontier_model_v3 import (
    Config,
    FAMILIES,
    Learner,
    capacity_matched_width,
    categorical_parameter_count,
    scalar_parameter_count,
)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make(configs=None, *, bins=9):
    if configs is None:
        # Deliberately interleave families and reverse their canonical order.
        configs = [Config(family, lr) for lr in (3e-3, 1e-3) for family in reversed(FAMILIES)]
    return Learner(5, 7, 1.3, configs, "cuda", bins=bins, num_samples=9)


def stream():
    generator = torch.Generator(device="cuda").manual_seed(19)
    for index in range(12):
        x = torch.randn((9, 5), generator=generator, device="cuda")
        y = torch.rand(9, generator=generator, device="cuda") * 25.0 - 1.3
        mask = torch.rand(9, generator=generator, device="cuda") > 0.3
        mask[0] = True
        if index == 4:
            mask.zero_()
            y.fill_(float("nan"))
        elif index == 7:
            mask.zero_()
            mask[3] = True
        yield x, y, mask


class MSEJSReference:
    """Autograd mean-MSE + torch Adam, with independent post-update JS shrinkage."""

    def __init__(self, bank, index):
        self.parameters = [torch.nn.Parameter(p[index].clone()) for p in bank.parameters]
        self.optimizer = torch.optim.Adam(self.parameters, lr=bank.configs[index].lr, foreach=False, fused=False)
        self.support = bank.support.clone()
        self.kappa = bank.kappa
        self.s1 = [torch.zeros_like(p) for p in self.parameters]
        self.s2 = [torch.zeros_like(p) for p in self.parameters]

    def step(self, x, y, mask):
        w1, b1, w2, b2, w3, b3 = self.parameters
        hidden = torch.tanh(F.linear(torch.tanh(F.linear(x, w1, b1)), w2, b2))
        prediction = (F.linear(hidden, w3, b3).softmax(-1) * self.support).sum(-1)
        if not bool(mask.any()):
            return prediction.detach()
        self.optimizer.zero_grad(set_to_none=True)
        (prediction[mask] - y[mask]).square().mean().backward()
        old = [p.detach().clone() for p in self.parameters]
        self.optimizer.step()
        with torch.no_grad():
            for parameter, before, s1, s2 in zip(self.parameters, old, self.s1, self.s2):
                # The signal-to-noise statistic uses prior gradients only;
                # torch Adam's moments must remain completely ungated.
                tiny = torch.finfo(torch.float32).tiny
                signal_to_noise = s1.square() / s2.clamp_min(tiny)
                gate = (1.0 - self.kappa / signal_to_noise.clamp_min(tiny)).clamp_min(0.0)
                gate = torch.where(signal_to_noise == 0, 0.0, gate)
                parameter.copy_(before + gate * (parameter - before))
                s1.add_(parameter.grad)
                s2.add_(parameter.grad.square())
        return prediction.detach()


def test_capacity_width_is_minimal_integer_and_matches_production_budget():
    assert categorical_parameter_count(257, 256, 33) == 140321
    assert capacity_matched_width(257, 256, 33) == 267
    assert scalar_parameter_count(257, 267) == 140710
    # Include exact equality, small dimensions and an unusually large head.
    for input_dim, width, bins in ((1, 1, 1), (5, 7, 9), (257, 256, 33), (2, 1, 1000)):
        result = capacity_matched_width(input_dim, width, bins)
        budget = categorical_parameter_count(input_dim, width, bins)
        assert scalar_parameter_count(input_dim, result) >= budget
        assert result == 1 or scalar_parameter_count(input_dim, result - 1) < budget


@cuda
def test_reported_capacity_matches_active_and_allocated_tensors():
    learner = Learner(257, 256, 1.3, [Config(family, 1e-3) for family in FAMILIES], "cuda")
    assert learner.widths == [256, 267, 256, 256]
    assert learner.effective_parameter_counts == [132097, 140710, 140321, 140321]
    actual_allocated = 0
    for bank, metadata in zip(learner.groups, learner.group_metadata):
        allocated = sum(parameter.numel() for parameter in bank.parameters)
        actual_allocated += allocated
        active = sum(parameter[0].numel() for parameter in bank.parameters[:4])
        if metadata["family"].startswith("scalar"):
            active += bank.weights[2][0, :1].numel() + bank.biases[2][0, :1].numel()
        else:
            active += bank.weights[2][0].numel() + bank.biases[2][0].numel()
        assert metadata["effective_parameters_per_config"] == active
        assert metadata["allocated_parameters_per_config"] == allocated
    assert learner.allocated_parameter_count == actual_allocated
    assert learner.allocated_parameter_count == sum(p.numel() for p in learner.parameters)
    assert actual_allocated > sum(learner.effective_parameter_counts)


@cuda
def test_categorical_loss_ablation_has_identical_initialization_and_gate():
    learner = make([Config("categorical_mse_js", 1e-3), Config("categorical_ce_js", 1e-3)])
    mse, ce = learner.groups
    for left, right in zip(mse.state_tensors(), ce.state_tensors()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert left.data_ptr() != right.data_ptr()
    torch.testing.assert_close(mse.support, ce.support, rtol=0, atol=0)
    torch.testing.assert_close(mse.js, ce.js, rtol=0, atol=0)
    assert bool(mse.js.all()) and not bool(mse.cross_entropy.any()) and bool(ce.cross_entropy.all())
    assert mse.configs[0].family == "categorical_mse_js"
    assert mse.output_names == ("categorical_mse_js_0.001",)
    assert learner.group_metadata[0]["raw_family"] == "categorical_mse"
    # The override must not leak into the frozen constructor or family registry.
    raw = frozen.Learner(5, 7, 1.3, [Config("categorical_mse", 1e-3)], "cuda", bins=9, num_samples=9)
    assert not bool(raw.js.any())
    assert "categorical_mse_js" not in frozen.FAMILIES
    x, y, mask = next(stream())
    learner.step(x, y, mask)
    assert any(not torch.equal(a, b) for a, b in zip(mse.first_moments, ce.first_moments))


@cuda
@pytest.mark.parametrize("kappa", [0.0, 1.0, 2.0])
def test_categorical_mse_js_matches_independent_autograd_adam_trajectory(kappa):
    learner = Learner(5, 7, 1.3, [Config("categorical_mse_js", lr) for lr in (1e-3, 3e-3)],
                      "cuda", bins=9, num_samples=9, kappa=kappa)
    bank = learner.groups[0]
    references = [MSEJSReference(bank, index) for index in range(2)]
    for x, y, mask in stream():
        actual = learner.step(x, y, mask)
        for index, reference in enumerate(references):
            expected = reference.step(x, y, mask)
            torch.testing.assert_close(actual[index], expected, rtol=1e-4, atol=5e-6)
            for parameter, first, second, s1, s2, ref_parameter, ref_s1, ref_s2 in zip(
                bank.parameters, bank.first_moments, bank.second_moments, bank.s1, bank.s2,
                reference.parameters, reference.s1, reference.s2,
            ):
                state = reference.optimizer.state[ref_parameter]
                torch.testing.assert_close(parameter[index], ref_parameter, rtol=2e-4, atol=3e-6)
                torch.testing.assert_close(first[index], state["exp_avg"], rtol=2e-4, atol=5e-6)
                torch.testing.assert_close(second[index], state["exp_avg_sq"], rtol=3e-4, atol=5e-7)
                torch.testing.assert_close(s1[index], ref_s1, rtol=2e-4, atol=5e-5)
                torch.testing.assert_close(s2[index], ref_s2, rtol=3e-4, atol=5e-4)
    assert int(learner.steps) == 12
    assert int(learner.adam_steps) == 11


@cuda
def test_interleaved_frozen_baselines_keep_exact_trajectories():
    configs = [Config("categorical_ce_js", 3e-3), Config("scalar_budget_js", 1e-3),
               Config("scalar_js", 3e-3), Config("categorical_ce_js", 1e-3),
               Config("scalar_budget_js", 3e-3)]
    learner = make(configs)
    references = [frozen.Learner(
        5, metadata["width"], 1.3,
        [Config(metadata["raw_family"], configs[i].lr) for i in metadata["config_indices"]],
        "cuda", bins=9, num_samples=9,
    ) for metadata in learner.group_metadata]
    for x, y, mask in stream():
        actual = learner.step(x, y, mask)
        for bank, metadata, reference in zip(learner.groups, learner.group_metadata, references):
            expected = reference.step(x, y, mask)
            for local, position in enumerate(metadata["config_indices"]):
                torch.testing.assert_close(actual[position], expected[local], rtol=0, atol=0)
            for left, right in zip(bank.state_tensors(), reference.state_tensors()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert learner.configs == tuple(configs)
    assert learner.steps is learner.clocks[0][0]
    assert learner.adam_steps is learner.clocks[0][1]
    assert [(int(consumed), int(optimizer)) for consumed, optimizer in learner.clocks] == [(12, 11)] * 3


@cuda
def test_labels_are_causal_and_eventually_affect_all_closed_gate_forecasts():
    left, right = make(), make()
    x, y, mask = next(stream())
    current_left = left.step(x, torch.zeros_like(y), mask).clone()
    current_right = right.step(x, torch.full_like(y, 10.0), mask).clone()
    torch.testing.assert_close(current_left, current_right, rtol=0, atol=0)
    changed = torch.zeros(len(left.configs), dtype=torch.bool, device="cuda")
    # All gates start closed: label history must eventually change forecasts,
    # not necessarily the immediately following forecast.
    for _ in range(8):
        future_left = left.step(x, y, mask).clone()
        future_right = right.step(x, y, mask).clone()
        changed |= (future_left - future_right).abs().max(-1).values > 1e-7
    assert bool(changed.all())


@cuda
def test_new_mse_gate_observes_prior_evidence_without_gating_moments():
    learner = make([Config("categorical_mse_js", 3e-3)])
    bank = learner.groups[0]
    initial = [p.clone() for p in bank.parameters]
    x, y, mask = next(stream())
    for _ in range(2):
        learner.step(x, y, mask)
        for actual, expected in zip(bank.parameters, initial):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert any(bool(moment.ne(0).any()) for moment in bank.first_moments)
    learner.step(x, y, mask)
    assert any(bool((actual - expected).abs().max() > 1e-7) for actual, expected in zip(bank.parameters, initial))


@cuda
def test_candidate_health_maps_interleaved_group_state_and_owned_output():
    learner = make()
    assert bool(learner.candidate_finite().all())
    expected = torch.ones(len(learner.configs), dtype=torch.bool, device="cuda")
    for bank, metadata in zip(learner.groups, learner.group_metadata):
        bank.s2[-1][1, 0] = float("inf")
        expected[metadata["config_indices"][1]] = False
    learner.prediction[0, 0] = float("nan")
    expected[0] = False
    torch.testing.assert_close(learner.candidate_finite(), expected, rtol=0, atol=0)


@cuda
def test_compiled_graph_restores_every_group_and_replays_bitwise():
    learner = make()
    x, y, mask = next(stream())
    compiled = torch.compile(learner.step, fullgraph=True, options={"triton.cudagraphs": False})
    mutable = learner.state_tensors()
    assert len({tensor.data_ptr() for tensor in mutable}) == len(mutable)
    initial = [tensor.clone() for tensor in mutable]

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
    for _ in range(8):
        graph.replay()
    torch.cuda.synchronize()
    expected = [tensor.clone() for tensor in mutable]
    restore(initial)
    for _ in range(8):
        graph.replay()
    torch.cuda.synchronize()
    for actual, value in zip(mutable, expected):
        assert torch.equal(actual.reshape(-1).view(torch.uint8), value.reshape(-1).view(torch.uint8))
    torch.testing.assert_close(result, learner.prediction, rtol=0, atol=0)
    reference = make()
    for _ in range(8):
        reference.step(x, y, mask)
    for actual, value in zip(mutable, reference.state_tensors()):
        torch.testing.assert_close(actual, value, rtol=3e-4, atol=5e-5)
    assert [(int(consumed), int(optimizer)) for consumed, optimizer in learner.clocks] == [(8, 8)] * 4
    restore(initial)
    mask.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert [(int(consumed), int(optimizer)) for consumed, optimizer in learner.clocks] == [(1, 0)] * 4
    changed_storage = {bank.steps.data_ptr() for bank in learner.groups}
    changed_storage.update(bank.prediction.data_ptr() for bank in learner.groups)
    changed_storage.add(learner.prediction.data_ptr())
    for actual, value in zip(mutable, initial):
        if actual.data_ptr() not in changed_storage:
            torch.testing.assert_close(actual, value, rtol=0, atol=0)


@cuda
def test_invalid_labels_do_not_change_any_group_update():
    left, right = make(), make()
    x, y, _ = next(stream())
    mask = torch.tensor([True, False, True, False, False, True, True, False, True], device="cuda")
    alternate_y = y.clone()
    alternate_y[~mask] = float("nan")
    for _ in range(8):
        torch.testing.assert_close(left.step(x, y, mask), right.step(x, alternate_y, mask), rtol=0, atol=0)
        for actual, expected in zip(left.state_tensors(), right.state_tensors()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
