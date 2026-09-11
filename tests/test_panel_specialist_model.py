"""CUDA behavioral checks; Main runs these through mlq, not as training evidence."""

import pytest
import torch
from torch import nn

from cleanrl.plasticity.panel_distributional_model_v1 import Learner as DenseLearner
from cleanrl.plasticity.panel_hd_gate import Net
from cleanrl.plasticity.panel_specialist_model_v2 import Config, DENSE_FAMILIES, FAMILIES, Learner
from cleanrl.shared import runtime


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make(configs=None):
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    if configs is None:
        configs = [Config(family, lr) for family in FAMILIES for lr in (1e-3, 3e-3)]
    return Learner(5, 8, 1.3, configs, "cuda", bins=5, num_samples=9)


def stream():
    generator = torch.Generator(device="cuda").manual_seed(71)
    for index in range(12):
        x = torch.randn((9, 5), generator=generator, device="cuda")
        y = torch.randn(9, generator=generator, device="cuda") * 2.0
        mask = torch.rand(9, generator=generator, device="cuda") > 0.3
        mask[0] = True
        if index == 4:
            mask.zero_()
        elif index == 7:
            mask.zero_()
            mask[3] = True
        x[~mask] = float("nan")
        y[~mask] = float("nan")
        yield x, y, mask


class MixtureReference(nn.Module):
    """Independent per-expert modules, autograd MSE, and torch.optim.Adam."""

    def __init__(self, config):
        super().__init__()
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            experts = []
            for index in range(4):
                torch.random.default_generator.manual_seed(1 + index * 1009)
                experts.append(Net(5, 2))
        self.experts = nn.ModuleList(experts).cuda()
        self.weight = nn.Parameter(torch.zeros((4, 5), device="cuda")) if config.family.startswith("state_") else None
        self.bias = nn.Parameter(torch.zeros(4, device="cuda")) if config.family != "uniform_moe_js" else None
        self.config = config
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr, foreach=False, fused=False)
        self.evidence = {p: (torch.zeros_like(p), torch.zeros_like(p)) for p in self.parameters()}

    def forward(self, x):
        output = torch.stack([expert(x)[0] for expert in self.experts], -1)
        logits = torch.zeros_like(output)
        if self.weight is not None:
            logits = logits + x @ self.weight.T
        if self.bias is not None:
            logits = logits + self.bias
        return (logits.softmax(-1) * output).sum(-1)

    def step(self, x, y, mask):
        prediction = self(torch.where(mask[:, None], x, 0.0))
        if not bool(mask.any()):
            return prediction.detach()
        self.optimizer.zero_grad(set_to_none=True)
        (prediction[mask] - y[mask]).square().mean().backward()
        before = {p: p.detach().clone() for p in self.parameters()}
        self.optimizer.step()
        with torch.no_grad():
            for p in self.parameters():
                s1, s2 = self.evidence[p]
                if self.config.family.endswith("_js"):
                    # Equivalent statistic, independently expressed. Evidence is
                    # read before adding this bar's raw autograd gradient.
                    tiny = torch.finfo(torch.float32).tiny
                    statistic = s1.square() / s2.clamp_min(tiny)
                    factor = (1.0 - 1.0 / statistic.clamp_min(tiny)).clamp_min(0.0)
                    p.copy_(before[p] + factor * (p - before[p]))
                s1.add_(p.grad)
                s2.add_(p.grad.square())
        return prediction.detach()


def reference_pairs(learner, moe_index, reference):
    for parameter_index in range(6):
        refs = [tuple(expert.parameters())[parameter_index] for expert in reference.experts]
        yield learner.experts, moe_index, parameter_index, refs
    if reference.bias is not None:
        index = learner.router_bias_indices.tolist().index(moe_index)
        yield learner.router_bias, index, 0, [reference.bias]
    if reference.weight is not None:
        index = learner.router_weight_indices.tolist().index(moe_index)
        yield learner.router_weight, index, 0, [reference.weight]


def test_manual_moe_trajectories_match_independent_autograd_adam_and_js():
    configs = [Config(family, lr) for family in FAMILIES if family not in DENSE_FAMILIES for lr in (1e-3, 3e-3)]
    learner = make(configs)
    references = [MixtureReference(config) for config in configs]
    for x, y, mask in stream():
        actual = learner.step(x, y, mask)
        for index, reference in enumerate(references):
            torch.testing.assert_close(actual[index], reference.step(x, y, mask), rtol=2e-4, atol=5e-6)
            for group, row, parameter_index, refs in reference_pairs(learner, index, reference):
                def packed(values):
                    return torch.stack(values) if group is learner.experts else values[0]

                torch.testing.assert_close(group.parameters[parameter_index][row], packed([p.detach() for p in refs]), rtol=3e-4, atol=5e-6)
                for actual_state, expected_state in (
                    (group.first_moments, [reference.optimizer.state[p]["exp_avg"] for p in refs]),
                    (group.second_moments, [reference.optimizer.state[p]["exp_avg_sq"] for p in refs]),
                    (group.s1, [reference.evidence[p][0] for p in refs]),
                    (group.s2, [reference.evidence[p][1] for p in refs]),
                ):
                    torch.testing.assert_close(actual_state[parameter_index][row], packed(expected_state), rtol=5e-4, atol=1e-5)
    assert int(learner.steps) == 12
    assert int(learner.adam_steps) == 11


def test_dense_controls_preserve_frozen_trajectories_and_arbitrary_output_order():
    configs = [Config("state_moe_js", 1e-3), Config("categorical_ce_js", 3e-3), Config("uniform_moe_js", 1e-3), Config("scalar_adam", 1e-3), Config("scalar_js", 3e-3)]
    learner = make(configs)
    dense_configs = [config for config in configs if config.family in DENSE_FAMILIES]
    reference = DenseLearner(5, 8, 1.3, dense_configs, "cuda", bins=5, num_samples=9)
    for x, y, mask in stream():
        prediction = learner.step(x, y, mask)
        expected = reference.step(torch.where(mask[:, None], x, 0.0), torch.where(mask, y, 0.0), mask)
        torch.testing.assert_close(prediction[[1, 3, 4]], expected, rtol=0, atol=0)
        for actual, wanted in zip(learner.dense.state_tensors(), reference.state_tensors()):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        assert all(int(clock) == int(learner.steps) for clock in (learner.dense.steps, learner.moe_steps))
        assert int(learner.moe_adam_steps) == int(reference.adam_steps)


def test_initial_mixture_matches_independent_experts_and_truthful_capacity():
    configs = [Config(family, 1e-3) for family in FAMILIES if family not in DENSE_FAMILIES]
    learner = make(configs)
    x, y, mask = next(stream())
    references = [MixtureReference(config) for config in configs]
    actual = learner.step(x, y, mask)
    for index, reference in enumerate(references):
        expected = reference(torch.where(mask[:, None], x, 0.0))
        torch.testing.assert_close(actual[index], expected, rtol=2e-6, atol=2e-7)
        assert learner.effective_parameter_counts[index] == sum(p.numel() for p in reference.parameters())
    torch.testing.assert_close(actual, actual[:1].expand_as(actual), rtol=0, atol=0)
    assert learner.allocated_parameter_count == sum(sum(p.numel() for p in ref.parameters()) for ref in references)
    # Independent expert storage: changing one arm/expert cannot change another.
    initial = learner.experts.parameters[0].clone()
    learner.experts.parameters[0][0, 0].add_(1.0)
    torch.testing.assert_close(learner.experts.parameters[0][1:], initial[1:], rtol=0, atol=0)
    torch.testing.assert_close(learner.experts.parameters[0][0, 1:], initial[0, 1:], rtol=0, atol=0)


def set_constant_experts(learner):
    for parameter in learner.experts.parameters:
        parameter.zero_()
    learner.experts.parameters[5].copy_(torch.tensor([-2.0, -1.0, 1.0, 2.0], device="cuda")[None, :, None])


def test_router_credit_sign_increases_weight_of_experts_toward_target():
    learner = make([Config("state_moe_adam", 1e-3)])
    set_constant_experts(learner)
    x = torch.zeros((9, 5), device="cuda")
    x[:, 0] = 1.0
    y = torch.full((9,), 2.0, device="cuda")
    mask = torch.ones(9, dtype=torch.bool, device="cuda")
    torch.testing.assert_close(learner.step(x, y, mask), torch.zeros((1, 9), device="cuda"), rtol=0, atol=0)
    # dL/dlogit = 2*(0-2)*(1/4)*[-2,-1,1,2]; summed over the bar.
    expected_gradient = torch.tensor([[2.0, 1.0, -1.0, -2.0]], device="cuda")
    torch.testing.assert_close(learner.router_bias.s1[0], expected_gradient, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(learner.router_weight.s1[0][:, :, 0], expected_gradient, rtol=1e-6, atol=1e-7)
    assert bool((learner.router_bias.parameters[0][0, 2:] > 0).all())
    assert bool((learner.router_bias.parameters[0][0, :2] < 0).all())
    assert bool((learner.step(x, y, mask) > 0).all())


def test_uniform_constant_and_state_controls_separate_conditional_routing():
    learner = make([Config(family, 3e-3) for family in ("uniform_moe_js", "constant_moe_js", "state_moe_js")])
    set_constant_experts(learner)
    x = torch.zeros((9, 5), device="cuda")
    x[:, 0] = 1.0
    y = torch.full((9,), 2.0, device="cuda")
    mask = torch.ones(9, dtype=torch.bool, device="cuda")
    initial = [p.clone() for p in learner.parameters]
    for _ in range(2):
        learner.step(x, y, mask)
        for actual, expected in zip(learner.parameters, initial):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    learner.step(x, y, mask)
    x[::2, 0] = -1.0
    prediction = learner.step(x, y, mask).clone()
    torch.testing.assert_close(prediction[0], prediction[0, :1].expand(9), rtol=0, atol=0)
    torch.testing.assert_close(prediction[1], prediction[1, :1].expand(9), rtol=0, atol=0)
    assert prediction[1, 0] > prediction[0, 0]
    assert prediction[2, 1] > prediction[2, 0]


def test_mask_sanitization_and_empty_bar_leave_optimizer_state_unchanged():
    left, right = make(), make()
    x, y, mask = next(stream())
    alternate_x, alternate_y = x.clone(), y.clone()
    alternate_x[~mask] = float("inf")
    alternate_y[~mask] = -float("inf")
    for _ in range(4):
        left.step(x, y, mask)
        right.step(alternate_x, alternate_y, mask)
        for a, b in zip(left.state_tensors(), right.state_tensors()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    before = [tensor.clone() for tensor in left.state_tensors()]
    left.step(torch.full_like(x, float("nan")), torch.full_like(y, float("nan")), torch.zeros_like(mask))
    mutable_forecasts = {id(left.prediction), id(left.dense.prediction)}
    consumed_clocks = {id(left.steps), id(left.moe_steps)}
    for tensor, saved in zip(left.state_tensors(), before):
        if id(tensor) in consumed_clocks:
            torch.testing.assert_close(tensor, saved + 1, rtol=0, atol=0)
        elif id(tensor) not in mutable_forecasts:
            torch.testing.assert_close(tensor, saved, rtol=0, atol=0)
    assert bool(left.candidate_finite().all())


def test_current_label_causality_and_future_response_in_all_families():
    left, right = make(), make()
    x, y, mask = next(stream())
    for _ in range(4):
        left.step(x, y, mask)
        right.step(x, y, mask)
    current_a = left.step(x, torch.zeros_like(y), mask).clone()
    current_b = right.step(x, torch.full_like(y, 10.0), mask).clone()
    torch.testing.assert_close(current_a, current_b, rtol=0, atol=0)
    next_a = left.step(x, y, mask).clone()
    next_b = right.step(x, y, mask).clone()
    assert bool(((next_a[:, mask] - next_b[:, mask]).abs().amax(-1) > 1e-7).all())


def test_candidate_health_contains_failure_to_own_configuration():
    learner = make([Config("state_moe_js", 1e-3), Config("scalar_adam", 1e-3), Config("constant_moe_js", 1e-3), Config("uniform_moe_js", 1e-3)])
    learner.router_bias.s2[0][1, 0] = float("nan")
    torch.testing.assert_close(learner.candidate_finite(), torch.tensor([True, True, False, True], device="cuda"))
    learner.dense.first_moments[0][0, 0, 0] = float("inf")
    torch.testing.assert_close(learner.candidate_finite(), torch.tensor([True, False, False, True], device="cuda"))


def test_fullgraph_capture_restores_complete_state_and_replays_exactly():
    learner = make([Config(family, 1e-3) for family in FAMILIES])
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
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize()
    expected = [tensor.clone() for tensor in mutable]
    restore(initial)
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize()
    for actual, wanted in zip(mutable, expected):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    torch.testing.assert_close(result, learner.prediction, rtol=0, atol=0)
    reference = make([Config(family, 1e-3) for family in FAMILIES])
    for _ in range(4):
        reference.step(x, y, mask)
    for actual, wanted in zip(mutable, reference.state_tensors()):
        torch.testing.assert_close(actual, wanted, rtol=5e-4, atol=1e-5)
    assert int(learner.steps) == int(learner.moe_steps) == 4
    assert int(learner.adam_steps) == int(learner.moe_adam_steps) == 4
