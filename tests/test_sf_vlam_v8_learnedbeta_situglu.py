import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
V3_SCRIPT = (
    ROOT
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_sf_vlam_v3_situglu.py"
)
V8_SCRIPT = (
    ROOT
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_sf_vlam_v8_learnedbeta_situglu.py"
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V3 = _load_module("sf_vlam_v3_situglu_reference", V3_SCRIPT)
MODULE = _load_module("sf_vlam_v8_learnedbeta_situglu", V8_SCRIPT)


def test_learned_betas_initialize_to_official_allocation():
    branch = MODULE.SiTUGLUBranch(64)
    torch.testing.assert_close(branch.beta_gate(), torch.tensor(4.0), atol=0.0, rtol=0.0)
    torch.testing.assert_close(branch.beta_up(), torch.tensor(25.0), atol=0.0, rtol=0.0)


def test_learned_betas_preserve_cap_product_when_ratio_changes():
    branch = MODULE.SiTUGLUBranch(64)
    for log_ratio in (-10.0, -2.0, 0.0, 2.0, 10.0):
        with torch.no_grad():
            branch.log_beta_delta.fill_(log_ratio)
        product = branch.beta_gate() * branch.beta_up()
        torch.testing.assert_close(product, torch.tensor(100.0), atol=2e-5, rtol=2e-7)


def test_learned_situ_glu_matches_equation_and_product_bound():
    branch = MODULE.SiTUGLUBranch(64).cuda()
    gate = torch.linspace(-10_000.0, 10_000.0, 100_001, device="cuda")
    up = gate.flip(0)

    for log_ratio in (-10.0, -2.0, 0.0, 2.0, 10.0):
        with torch.no_grad():
            branch.log_beta_delta.fill_(log_ratio)
        beta_gate = branch.beta_gate()
        beta_up = branch.beta_up()
        expected = (
            100.0
            * torch.tanh(gate / beta_gate)
            * torch.sigmoid(gate)
            * torch.tanh(up / beta_up)
        )
        actual = MODULE.situ_glu(gate, up, beta_gate, beta_up)
        torch.testing.assert_close(actual, expected)
        assert actual.abs().max() <= 100.0


def test_learned_branch_is_honestly_parameter_matched_and_bias_free():
    branch = MODULE.SiTUGLUBranch(64)
    parameter_count = sum(parameter.numel() for parameter in branch.parameters())
    old_parameter_count = 2 * (64 * 64 + 64)
    assert branch.hidden_dim == 43
    assert parameter_count == 8_257
    assert abs(parameter_count - old_parameter_count) / old_parameter_count < 0.01
    assert branch.log_beta_delta.numel() == 1
    assert branch.gate.bias is None
    assert branch.up.bias is None
    assert branch.down.bias is None


def test_learned_branch_preserves_v2_rng_position():
    torch.manual_seed(7)
    MODULE.layer_init(torch.nn.Linear(64, 64))
    MODULE.layer_init(torch.nn.Linear(64, 64))
    expected_rng_state = torch.get_rng_state()

    torch.manual_seed(7)
    MODULE.SiTUGLUBranch(64)
    actual_rng_state = torch.get_rng_state()
    torch.testing.assert_close(actual_rng_state, expected_rng_state)


def test_learned_branch_recovers_v3_initial_function_within_float32_rounding():
    inputs = torch.randn(256, 64, generator=torch.Generator().manual_seed(13))

    torch.manual_seed(11)
    reference = V3.SiTUGLUBranch(64)
    torch.manual_seed(11)
    learned = MODULE.SiTUGLUBranch(64)

    torch.testing.assert_close(learned.gate.weight, reference.gate.weight)
    torch.testing.assert_close(learned.up.weight, reference.up.weight)
    torch.testing.assert_close(learned.down.weight, reference.down.weight)
    torch.testing.assert_close(learned(inputs), reference(inputs), atol=2e-6, rtol=2e-6)


def test_beta_ratio_and_all_branch_parameters_receive_finite_gradients():
    branch = torch.compile(MODULE.SiTUGLUBranch(64).cuda(), mode="reduce-overhead")
    inputs = torch.randn(128, 64, device="cuda", requires_grad=True)
    output = branch(inputs)
    output.square().mean().backward()

    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
    assert torch.isfinite(inputs.grad).all()
    assert branch.log_beta_delta.grad is not None
    assert torch.isfinite(branch.log_beta_delta.grad)
    assert branch.log_beta_delta.grad.abs() > 0.0
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in branch.parameters()
    )


if __name__ == "__main__":
    test_learned_betas_initialize_to_official_allocation()
    test_learned_betas_preserve_cap_product_when_ratio_changes()
    test_learned_situ_glu_matches_equation_and_product_bound()
    test_learned_branch_is_honestly_parameter_matched_and_bias_free()
    test_learned_branch_preserves_v2_rng_position()
    test_learned_branch_recovers_v3_initial_function_within_float32_rounding()
    test_beta_ratio_and_all_branch_parameters_receive_finite_gradients()
