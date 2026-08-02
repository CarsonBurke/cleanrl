import importlib.util
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import torch


ROOT = Path(__file__).parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1 = load_module(
    "spk_v1_for_coordinate_test",
    ROOT / "cleanrl" / "sparse-nn" / "ppo_continuous_action_spk_v1.py",
)
v4 = load_module(
    "spk_packed_coordinate_v4",
    ROOT
    / "cleanrl"
    / "sparse-nn"
    / "ppo_continuous_action_spk_packed_coordinate_v4.py",
)


def make_v1_layer(in_features=73, out_features=11, k=64):
    return v1.SparseKLinear(
        in_features,
        out_features,
        k,
        rewire_mode="none",
    )


def make_v4_layer(coordinate, in_features=73, out_features=11, k=64):
    return v4.SparseKLinear(
        in_features,
        out_features,
        k,
        rewire_mode="none",
        weight_coordinate=coordinate,
    )


def test_effective_packed_layer_matches_v1_forward_backward_and_adam_step():
    torch.manual_seed(10)
    reference = make_v1_layer()
    torch.manual_seed(10)
    packed = make_v4_layer("effective")

    assert torch.equal(packed.indices, reference.indices)
    assert torch.equal(packed.weight, reference.weight)
    assert torch.equal(packed.bias, reference.bias)

    reference_input = torch.randn(19, 73, requires_grad=True)
    packed_input = reference_input.detach().clone().requires_grad_(True)
    reference_output = reference(reference_input)
    packed_output = packed(packed_input)
    assert torch.allclose(packed_output, reference_output, atol=1e-6, rtol=1e-6)

    reference_loss = reference_output.square().mean()
    packed_loss = packed_output.square().mean()
    reference_loss.backward()
    packed_loss.backward()
    assert torch.allclose(packed_input.grad, reference_input.grad, atol=1e-7, rtol=1e-6)
    assert torch.allclose(packed.weight.grad, reference.weight.grad, atol=1e-7, rtol=1e-6)
    assert torch.allclose(packed.bias.grad, reference.bias.grad, atol=1e-7, rtol=1e-6)

    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=3e-4, eps=1e-5)
    packed_optimizer = torch.optim.Adam(packed.parameters(), lr=3e-4, eps=1e-5)
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.5)
    torch.nn.utils.clip_grad_norm_(packed.parameters(), 0.5)
    reference_optimizer.step()
    packed_optimizer.step()

    assert torch.allclose(packed.weight, reference.weight, atol=1e-7, rtol=1e-6)
    assert torch.allclose(packed.bias, reference.bias, atol=1e-7, rtol=1e-6)


def test_raw_and_effective_arms_start_with_identical_functions():
    torch.manual_seed(11)
    effective = make_v4_layer("effective")
    torch.manual_seed(11)
    raw = make_v4_layer("raw")
    inputs = torch.randn(17, 73)

    assert torch.equal(raw.indices, effective.indices)
    assert torch.allclose(raw.effective_weight(), effective.effective_weight())
    assert torch.allclose(raw(inputs), effective(inputs), atol=1e-6, rtol=1e-6)


def test_raw_coordinate_has_eight_times_smaller_effective_adam_step_at_k64():
    torch.manual_seed(12)
    effective = make_v4_layer("effective")
    torch.manual_seed(12)
    raw = make_v4_layer("raw")
    inputs = torch.randn(23, 73)
    effective_before = effective.effective_weight().detach().clone()
    raw_before = raw.effective_weight().detach().clone()
    effective_optimizer = torch.optim.Adam(effective.parameters(), lr=3e-4, eps=1e-5)
    raw_optimizer = torch.optim.Adam(raw.parameters(), lr=3e-4, eps=1e-5)

    effective(inputs).square().mean().backward()
    raw(inputs).square().mean().backward()
    effective_optimizer.step()
    raw_optimizer.step()

    effective_update = (effective.effective_weight() - effective_before).abs().mean()
    raw_update = (raw.effective_weight() - raw_before).abs().mean()
    ratio = (effective_update / raw_update).detach()

    assert 7.5 < float(ratio) < 8.5


def test_raw_coordinate_has_sqrt_k_smaller_effective_adam_step_at_k17():
    torch.manual_seed(121)
    effective = make_v4_layer("effective", in_features=17, k=17)
    torch.manual_seed(121)
    raw = make_v4_layer("raw", in_features=17, k=17)
    inputs = torch.randn(23, 17)
    effective_before = effective.effective_weight().detach().clone()
    raw_before = raw.effective_weight().detach().clone()
    effective_optimizer = torch.optim.Adam(effective.parameters(), lr=3e-4, eps=1e-5)
    raw_optimizer = torch.optim.Adam(raw.parameters(), lr=3e-4, eps=1e-5)

    effective(inputs).square().mean().backward()
    raw(inputs).square().mean().backward()
    effective_optimizer.step()
    raw_optimizer.step()

    effective_update = (effective.effective_weight() - effective_before).abs().mean()
    raw_update = (raw.effective_weight() - raw_before).abs().mean()
    ratio = (effective_update / raw_update).detach()

    assert 3.8 < float(ratio) < 4.4


def test_effective_full_agent_matches_v1_initialization_and_outputs():
    env = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-1.0, 1.0, shape=(17,)),
        single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(6,)),
    )
    reference_args = v1.Args(pool="prior", rewire="none")
    packed_args = v4.Args(pool="prior", rewire="none", weight_coordinate="effective")
    torch.manual_seed(122)
    reference = v1.Agent(env, reference_args)
    torch.manual_seed(122)
    packed = v4.Agent(env, packed_args)

    for reference_trunk, packed_trunk in (
        (reference.actor_trunk, packed.actor_trunk),
        (reference.critic_trunk, packed.critic_trunk),
    ):
        for reference_layer, packed_layer in zip(
            reference_trunk.sparse_layers(), packed_trunk.sparse_layers()
        ):
            assert torch.equal(packed_layer.indices, reference_layer.indices)
            assert torch.equal(packed_layer.weight, reference_layer.weight)
            assert torch.equal(packed_layer.bias, reference_layer.bias)

    observations = torch.randn(13, 17)
    latent_actions = torch.rand(13, 6).clamp(v1.SAMPLE_EPS, 1.0 - v1.SAMPLE_EPS)
    reference_outputs = reference.get_beta_action_and_value(observations, latent_actions)
    packed_outputs = packed.get_beta_action_and_value(observations, latent_actions)
    for reference_output, packed_output in zip(reference_outputs, packed_outputs):
        assert torch.allclose(packed_output, reference_output, atol=1e-6, rtol=1e-6)


def test_control_rejects_unknown_weight_coordinates():
    try:
        make_v4_layer("unknown")
    except ValueError as error:
        assert "effective|raw" in str(error)
    else:
        raise AssertionError("unknown coordinate should be rejected")


def run_cuda_parity_check():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA parity check requires CUDA")
    for k, in_features in ((17, 17), (64, 73)):
        torch.manual_seed(200 + k)
        reference = make_v1_layer(in_features=in_features, k=k).cuda()
        torch.manual_seed(200 + k)
        packed = make_v4_layer("effective", in_features=in_features, k=k).cuda()
        reference_input = torch.randn(1024, in_features, device="cuda", requires_grad=True)
        packed_input = reference_input.detach().clone().requires_grad_(True)
        reference_output = reference(reference_input)
        packed_output = packed(packed_input)
        forward_error = (packed_output - reference_output).abs().max()
        reference_output.square().mean().backward()
        packed_output.square().mean().backward()
        input_grad_error = (packed_input.grad - reference_input.grad).abs().max()
        weight_grad_error = (packed.weight.grad - reference.weight.grad).abs().max()
        repeated_error = max(
            float((packed(packed_input.detach()) - packed_output.detach()).abs().max())
            for _ in range(5)
        )
        print(
            f"k={k} forward_error={float(forward_error):.3g} "
            f"input_grad_error={float(input_grad_error):.3g} "
            f"weight_grad_error={float(weight_grad_error):.3g} "
            f"repeated_error={repeated_error:.3g}"
        )
        assert float(forward_error) < 1e-5
        assert float(input_grad_error) < 1e-5
        assert float(weight_grad_error) < 1e-5
        assert repeated_error < 1e-5


if __name__ == "__main__":
    run_cuda_parity_check()
