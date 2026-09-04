import ast
import importlib.util
import inspect
import sys
from pathlib import Path

import gymnasium as gym
import pytest
import torch


ROOT = Path(__file__).parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "cleanrl" / "nextlat" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V15 = _load(
    "nextlat_function_trust_livehead_v15_reference_for_cuda_v16",
    "ppo_continuous_action_nextlat_function_trust_livehead_v15.py",
)
V16 = _load(
    "nextlat_function_trust_livehead_cuda_v16",
    "ppo_continuous_action_nextlat_function_trust_livehead_cuda_v16.py",
)


class _DummyEnvs:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(7,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))


def _args(module, actor_dist="beta"):
    return module.Args(
        actor_dist=actor_dist,
        hidden=8,
        k_blocks=1,
        n_experts=2,
        critic_mtp_horizon=2,
        num_bins=5,
    )


def _semantic_ast(obj):
    tree = ast.parse(inspect.getsource(obj))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]
    return ast.dump(tree, include_attributes=False)


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
def test_v16_preserves_v15_seeded_parameters_and_forward_semantics(actor_dist):
    torch.manual_seed(93017)
    reference = V15.Agent(_DummyEnvs(), _args(V15, actor_dist))
    torch.manual_seed(93017)
    candidate = V16.Agent(_DummyEnvs(), _args(V16, actor_dist))

    assert reference.state_dict().keys() == candidate.state_dict().keys()
    for name, expected in reference.state_dict().items():
        assert torch.equal(expected, candidate.state_dict()[name]), name

    observations = torch.randn(11, 7)
    native_actions = (
        torch.full((11, 3), 0.37)
        if actor_dist == "beta"
        else torch.linspace(-0.8, 0.8, 33).reshape(11, 3)
    )
    torch.manual_seed(122)
    expected = reference.get_action_and_value(observations, native_actions)
    torch.manual_seed(122)
    actual = candidate.get_action_and_value(observations, native_actions)
    for expected_tensor, actual_tensor in zip(expected, actual):
        assert torch.equal(expected_tensor, actual_tensor)


def test_v16_defaults_match_v15_except_filename_derived_experiment_name():
    reference = vars(V15.Args())
    candidate = vars(V16.Args())
    reference.pop("exp_name")
    candidate.pop("exp_name")
    assert candidate == reference
    assert V16.Args().compile_mode == "reduce-overhead"


@pytest.mark.parametrize(
    "name",
    [
        "Agent",
        "project_predictive_updates",
        "apply_function_trust_transaction",
        "apply_private_optimizer_step",
        "prepare_live_head_auxiliary_gradients",
        "merge_live_head_auxiliary_gradients",
        "measure_livehead_probe_decomposition",
        "behavioral_kls",
        "policy_kl",
        "shape_advantage",
    ],
)
def test_v15_optimizer_and_model_components_are_ast_identical(name):
    assert _semantic_ast(getattr(V16, name)) == _semantic_ast(getattr(V15, name))


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_hl_gauss_projection_and_retention_stay_on_the_source_device(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable to this test process")
    device = torch.device(device_name)
    support = V16.Dreamer3BucketHLGaussSupport(
        5,
        -3.0,
        3.0,
        0.75,
        device,
    )
    returns = torch.linspace(-5.0, 5.0, 24, device=device).reshape(2, 3, 4)
    projected = support.project(returns)
    retained = V16.retain_projected_targets(projected, compiled=True)

    assert projected.device.type == device.type
    assert retained.device.type == device.type
    if device.type == "cuda":
        assert projected.device.index == torch.cuda.current_device()
        assert retained.device.index == torch.cuda.current_device()
    assert retained.data_ptr() != projected.data_ptr()
    expected = projected.clone()
    projected.zero_()
    assert torch.equal(retained, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compiled_projection_retention_survives_later_cuda_graph_replays():
    device = torch.device("cuda", torch.cuda.current_device())
    support = V16.Dreamer3BucketHLGaussSupport(5, -3.0, 3.0, 0.75, device)

    def project(targets):
        return support.project(targets)

    project = torch.compile(
        project,
        mode="reduce-overhead",
        dynamic=False,
        fullgraph=True,
    )
    inputs = [
        torch.full((2, 3, 4), value, device=device)
        for value in (-2.0, -0.5, 1.0, 2.5)
    ]
    retained = []
    expected = []
    for targets in inputs:
        expected.append(support.project(targets).clone())
        torch.compiler.cudagraph_mark_step_begin()
        graph_output = project(targets)
        retained.append(V16.retain_projected_targets(graph_output, compiled=True))
    torch.cuda.synchronize()

    for actual, reference in zip(retained, expected):
        assert torch.allclose(actual, reference, atol=3e-7, rtol=2e-6)


def test_training_wiring_keeps_targets_and_masks_cuda_resident_across_epochs():
    source = Path(V16.__file__).read_text()
    assert "hl_support_cpu" not in source
    assert "hl_support.project(return_mtp.detach().cpu())" not in source
    assert "b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon)" in source
    assert "target_probs_mb = b_target_probs[mb_inds]" in source
    assert "value_mask = b_target_mask[mb_inds]" in source
    assert "target_probs_mb = b_target_probs[mb_inds].to" not in source
    assert "b_target_mask[mb_inds].to(device=" not in source

    # Default v15 shape: this was copied once per epoch from pageable host memory.
    label_bytes = 2048 * 16 * 6 * 511 * torch.tensor([], dtype=torch.float32).element_size()
    assert label_bytes / 2**20 == pytest.approx(383.25)
    assert 10 * label_bytes / 2**30 == pytest.approx(3.74267578125)


def test_compiled_target_retention_clones_only_the_graph_backed_path():
    tensor = torch.randn(3, 4)
    eager = V16.retain_projected_targets(tensor, compiled=False)
    compiled = V16.retain_projected_targets(tensor, compiled=True)

    assert eager.data_ptr() == tensor.data_ptr()
    assert compiled.data_ptr() != tensor.data_ptr()
    tensor.add_(10.0)
    assert torch.equal(eager, tensor)
    assert not torch.equal(compiled, tensor)


def test_scalar_telemetry_preserves_order_values_and_has_one_sync_boundary():
    values = {
        "first": torch.tensor(1.25),
        "second": torch.tensor(-3.5, dtype=torch.float64),
        "third": torch.tensor(float("inf")),
    }
    result = V16.synchronize_scalar_telemetry(values)

    assert list(result) == list(values)
    assert result["first"] == pytest.approx(1.25)
    assert result["second"] == pytest.approx(-3.5)
    assert result["third"] == float("inf")
    helper_source = inspect.getsource(V16.synchronize_scalar_telemetry)
    assert helper_source.count(".cpu()") == 1
    assert ".item()" not in helper_source
    assert "float(value)" not in helper_source


def test_scalar_telemetry_rejects_non_scalars_and_host_values():
    with pytest.raises(ValueError, match="scalar"):
        V16.synchronize_scalar_telemetry({"bad": torch.ones(2)})
    with pytest.raises(TypeError, match="tensors"):
        V16.synchronize_scalar_telemetry({"bad": 1.0})


def test_clip_grad_norm_async_finite_path_is_exactly_torch_clipping():
    reference = [torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(2))]
    candidate = [torch.nn.Parameter(parameter.detach().clone()) for parameter in reference]
    gradients = [torch.tensor([3.0, 4.0, -2.0]), torch.tensor([0.5, -1.5])]
    for parameter, gradient in zip(reference, gradients):
        parameter.grad = gradient.clone()
    for parameter, gradient in zip(candidate, gradients):
        parameter.grad = gradient.clone()

    expected_norm = torch.nn.utils.clip_grad_norm_(
        reference,
        0.25,
        error_if_nonfinite=False,
    )
    actual_norm = V16.clip_grad_norm_async_fail_loud_(candidate, 0.25)

    assert torch.equal(actual_norm, expected_norm)
    for expected, actual in zip(reference, candidate):
        assert torch.equal(actual.grad, expected.grad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_clip_grad_norm_async_still_fails_loudly_on_nonfinite_gradient(bad):
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([bad, 1.0])
    with pytest.raises(RuntimeError, match="non-finite"):
        V16.clip_grad_norm_async_fail_loud_([parameter], 0.25)


def test_async_clip_implementation_has_no_scalar_host_read():
    source = inspect.getsource(V16.clip_grad_norm_async_fail_loud_)
    assert "error_if_nonfinite=False" in source
    assert "torch._assert_async" in source
    implementation = source[source.index("    total_norm =") :]
    assert ".item()" not in implementation
    assert ".tolist()" not in implementation


def test_minibatch_telemetry_and_indices_do_not_force_host_syncs():
    source = Path(V16.__file__).read_text()
    training = source[source.index('if __name__ == "__main__":') :]
    assert "clipfracs" not in training
    assert "clipfrac_sum.add_(" in training
    assert "mb_perc_scale.item()" not in training
    assert "ret_perc_scale.copy_(mb_perc_scale.detach())" in training
    assert "epoch_inds = torch.as_tensor(b_inds, device=device)" in training
    assert "mb_inds = epoch_inds[start:end]" in training
    assert ".item()" not in training
    assert ".tolist()" not in training


def test_reduce_overhead_cuda_graph_and_tf32_wiring_is_explicit_and_ordered():
    source = Path(V16.__file__).read_text()
    precision = source.index('torch.set_float32_matmul_precision("high")')
    compile_block = source.index("if args.compile:", precision)
    assert precision < compile_block
    assert 'compile_mode: str = "reduce-overhead"' in source
    assert "project_value_targets_fn = torch.compile(" in source
    assert "fullgraph=True" in source
    assert "torch.compiler.cudagraph_mark_step_begin()" in source
    assert "retain_projected_targets(" in source
    assert "Float32 matmul precision" in source
    assert "last-bit numerics" in source
