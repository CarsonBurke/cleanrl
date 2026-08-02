import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V3 = _load(
    "sf_vlam_v3_for_v11_test",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v3_situglu.py",
)
MODULE = _load(
    "sf_vlam_v11_scaleisolated_learnedbeta",
    "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v11_scaleisolated_learnedbeta.py",
)


def test_block_beta_allocation_initializes_exactly_and_is_compact():
    block = MODULE.ThinkBlock(64, 64, 16)
    delta, beta_gate, beta_up = block.beta_allocation()
    torch.testing.assert_close(delta, torch.tensor(0.0), atol=0.0, rtol=0.0)
    torch.testing.assert_close(beta_gate, torch.tensor(4.0), atol=0.0, rtol=0.0)
    torch.testing.assert_close(beta_up, torch.tensor(25.0), atol=0.0, rtol=0.0)

    for raw in (-1e6, -2.0, 0.0, 2.0, 1e6):
        with torch.no_grad():
            block.beta_raw.fill_(raw)
        delta, beta_gate, beta_up = block.beta_allocation()
        assert -torch.log(torch.tensor(4.0)) <= delta <= torch.log(torch.tensor(4.0))
        assert 1.0 <= beta_gate <= 16.0
        assert 6.25 <= beta_up <= 100.0
        torch.testing.assert_close(
            beta_gate * beta_up, torch.tensor(100.0), atol=2e-5, rtol=2e-7
        )


def test_three_block_trunk_has_only_three_shared_beta_parameters():
    trunk = MODULE.ThinkTrunk(17, 64, 3, 16)
    beta_parameters = [
        parameter
        for name, parameter in trunk.named_parameters()
        if name.endswith("beta_raw")
    ]
    assert len(beta_parameters) == 3
    assert sum(parameter.numel() for parameter in beta_parameters) == 3
    assert not any(
        name.endswith("beta_raw")
        for branch in [
            module
            for module in trunk.modules()
            if isinstance(module, MODULE.SiTUGLUBranch)
        ]
        for name, _ in branch.named_parameters()
    )


def test_initial_function_and_ordinary_gradients_match_v3():
    inputs_v3 = torch.randn(256, 64, device="cuda", requires_grad=True)
    inputs_v11 = inputs_v3.detach().clone().requires_grad_(True)

    torch.manual_seed(11)
    reference = V3.SiTUGLUBranch(64).cuda()
    torch.manual_seed(11)
    learned = MODULE.SiTUGLUBranch(64).cuda()

    output_v3 = reference(inputs_v3)
    output_v11 = learned(
        inputs_v11,
        torch.tensor(4.0, device="cuda"),
        torch.tensor(25.0, device="cuda"),
    )
    torch.testing.assert_close(output_v11, output_v3, atol=2e-6, rtol=2e-6)

    direction = torch.randn_like(output_v3)
    (output_v3 * direction).sum().backward()
    (output_v11 * direction).sum().backward()
    torch.testing.assert_close(inputs_v11.grad, inputs_v3.grad, atol=3e-5, rtol=3e-5)
    for learned_parameter, reference_parameter in zip(
        learned.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(
            learned_parameter.grad,
            reference_parameter.grad,
            atol=3e-5,
            rtol=3e-5,
        )


def test_adaptive_output_matches_reference_rms_for_every_sample():
    branch = MODULE.SiTUGLUBranch(64).cuda()
    inputs = torch.randn(257, 64, device="cuda")
    gate = branch.gate(inputs)
    up = branch.up(inputs)
    reference = branch.down(MODULE.situ_glu(gate, up))
    reference_rms = reference.square().mean(dim=-1).sqrt()

    for delta in (-1.0, -0.5, 0.5, 1.0):
        beta_gate = torch.tensor(4.0 * torch.exp(torch.tensor(delta)), device="cuda")
        beta_up = torch.tensor(25.0 * torch.exp(torch.tensor(-delta)), device="cuda")
        adaptive = branch(inputs, beta_gate, beta_up)
        adaptive_rms = adaptive.square().mean(dim=-1).sqrt()
        torch.testing.assert_close(
            adaptive_rms, reference_rms, atol=2e-5, rtol=2e-5
        )


def test_beta_tangent_is_radial_free_but_task_direction_is_learnable():
    block = MODULE.ThinkBlock(64, 64, 2).cuda()
    inputs = torch.randn(64, 64, device="cuda")
    x0 = torch.randn(64, 64, device="cuda")

    # A branch's squared norm is beta-invariant, so its beta derivative is radial-free.
    _, beta_gate, beta_up = block.beta_allocation()
    branch_output = block.dense(block.dense_norm(inputs), beta_gate, beta_up)
    energy_grad, = torch.autograd.grad(
        branch_output.square().sum(), block.beta_raw, retain_graph=True
    )
    assert energy_grad.abs() < 2e-4

    # A directional task objective still supplies a finite, nonzero beta gradient.
    output = block(inputs, x0)
    direction = torch.randn_like(output)
    task_grad, = torch.autograd.grad((output * direction).mean(), block.beta_raw)
    assert torch.isfinite(task_grad)
    assert task_grad.abs() > 1e-8


def test_beta_gradient_accumulator_applies_clip_factor_and_clears_live_grad():
    parameters = [
        torch.nn.Parameter(torch.tensor(0.0, device="cuda")),
        torch.nn.Parameter(torch.tensor(0.0, device="cuda")),
    ]
    parameters[0].grad = torch.tensor(4.0, device="cuda")
    parameters[1].grad = torch.tensor(-2.0, device="cuda")
    accumulators = [torch.zeros_like(parameter) for parameter in parameters]
    MODULE.accumulate_scaled_grads(
        parameters, accumulators, torch.tensor(0.25, device="cuda")
    )
    torch.testing.assert_close(accumulators[0], torch.tensor(1.0, device="cuda"))
    torch.testing.assert_close(accumulators[1], torch.tensor(-0.5, device="cuda"))
    assert all(parameter.grad is None for parameter in parameters)
    assert MODULE.Args().beta_learning_rate == 3e-3


def test_trunk_preserves_v3_rng_and_parameter_budget():
    torch.manual_seed(7)
    reference = V3.ThinkTrunk(17, 64, 3, 16)
    expected_rng_state = torch.get_rng_state()

    torch.manual_seed(7)
    learned = MODULE.ThinkTrunk(17, 64, 3, 16)
    actual_rng_state = torch.get_rng_state()

    torch.testing.assert_close(actual_rng_state, expected_rng_state)
    reference_count = sum(parameter.numel() for parameter in reference.parameters())
    learned_count = sum(parameter.numel() for parameter in learned.parameters())
    assert learned_count == reference_count + 3


def test_actual_compiled_trunk_forward_and_backward_are_finite():
    trunk = MODULE.ThinkTrunk(17, 64, 3, 16).cuda()
    compiled = MODULE.CompiledModule(
        trunk, mode="reduce-overhead", cudagraphs=False
    )
    inputs = torch.randn(128, 17, device="cuda", requires_grad=True)
    output = compiled(inputs)
    output.square().mean().backward()
    assert output.shape == (128, 64)
    assert torch.isfinite(output).all()
    assert torch.isfinite(inputs.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in trunk.parameters()
    )


if __name__ == "__main__":
    test_block_beta_allocation_initializes_exactly_and_is_compact()
    test_three_block_trunk_has_only_three_shared_beta_parameters()
    test_initial_function_and_ordinary_gradients_match_v3()
    test_adaptive_output_matches_reference_rms_for_every_sample()
    test_beta_tangent_is_radial_free_but_task_direction_is_learnable()
    test_beta_gradient_accumulator_applies_clip_factor_and_clears_live_grad()
    test_trunk_preserves_v3_rng_and_parameter_budget()
    test_actual_compiled_trunk_forward_and_backward_are_finite()
