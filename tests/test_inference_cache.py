"""Metadata checks are CPU-only; numerical/model tests must run through mlq."""

import pytest
import torch
from torch import nn

from cleanrl.shared.inference_cache import InferenceParameterCache, _MethodCall, _unique_parameter_slots, linear_parameter_names


def test_linear_selection_includes_aliases_and_excludes_custom_subclasses():
    class CustomLinear(nn.Linear):
        pass

    linear = nn.Linear(3, 2, device="meta")
    model = nn.ModuleDict({"first": linear, "tied": linear, "custom": CustomLinear(3, 2, device="meta")})
    assert linear_parameter_names(model) == ("first.weight", "first.bias", "tied.weight", "tied.bias")
    assert linear_parameter_names(nn.Linear(3, 2, bias=False, device="meta")) == ("weight",)


def test_method_adapter_preserves_module_hooks_and_custom_method_semantics():
    events = []

    class MetadataOnly(nn.Module):
        def forward(self, label):
            events.append("forward")
            return label + " output"

        def custom(self, label):
            events.append("custom")
            return label + " custom"

    model = MetadataOnly()

    def before(module, args):
        events.append("pre")
        return (args[0] + " input",)

    def after(module, args, output):
        events.append("post")
        return output + " hooked"

    model.register_forward_pre_hook(before)
    model.register_forward_hook(after)
    assert _MethodCall(model, "forward")("test") == "test input output hooked"
    assert events == ["pre", "forward", "post"]
    events.clear()
    assert _MethodCall(model, "custom")("test") == "test custom"
    assert events == ["custom"]


@pytest.mark.parametrize("names,match", [([], "at least one"), (["missing"], "unknown"), (["weight", "weight"], "duplicates")])
def test_invalid_selection_fails_before_cuda_allocation(names, match):
    with pytest.raises(ValueError, match=match):
        InferenceParameterCache(nn.Linear(3, 2, device="meta"), names)


def test_invalid_device_dtype_method_and_partial_ties_fail_before_cuda_allocation():
    model = nn.Linear(3, 2, device="meta")
    with pytest.raises(ValueError, match="FP32 CUDA"):
        InferenceParameterCache(model, ["weight"])
    with pytest.raises(ValueError, match="cache dtype"):
        InferenceParameterCache(model, ["weight"], dtype=torch.float32)
    with pytest.raises(ValueError, match="callable"):
        InferenceParameterCache(model, ["weight"], method="missing")
    with pytest.raises(TypeError, match="iterable"):
        InferenceParameterCache(model, "weight")
    model.tied = model.weight
    with pytest.raises(ValueError, match="every tied"):
        InferenceParameterCache(model, ["weight"])


@pytest.mark.parametrize("fail", [False, True])
def test_shared_module_slots_restore_master_parameters_and_replacements_on_meta(fail):
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(3, 2, device="meta")
            self.tied = self.first

        def forward(self):
            # Metadata-only: no tensor arithmetic or CPU model execution.
            assert self.first.weight is self.tied.weight
            assert self.first.weight.dtype == torch.bfloat16
            if fail:
                raise ValueError("intentional")
            return self.first.weight

    model = Shared()
    originals = dict(model.named_parameters(remove_duplicate=False))
    copies = {id(p): torch.empty_like(p, dtype=torch.bfloat16) for p in originals.values()}
    cached = {name: copies[id(p)] for name, p in originals.items()}
    replacements = _unique_parameter_slots(model, cached)
    assert tuple(replacements) == ("first.weight", "first.bias")
    before = replacements.copy()
    if fail:
        with pytest.raises(ValueError, match="intentional"):
            torch.func.functional_call(model, replacements, (), tie_weights=False)
    else:
        assert torch.func.functional_call(model, replacements, (), tie_weights=False) is cached["first.weight"]
    for name, parameter in originals.items():
        assert model.get_parameter(name) is parameter
        assert parameter.dtype == torch.float32
    for name, value in before.items():
        assert replacements[name] is value
        assert value.dtype == torch.bfloat16


class _MixedPrecisionModel(nn.Module):
    """Exercise the v30 Linear/GEMM/FP32 expert-bias/residual precision boundary."""

    def __init__(self):
        super().__init__()
        self.entry = nn.Linear(17, 32, device="cuda")
        self.weight1 = nn.Parameter(torch.randn(4, 32, 32, device="cuda") / 8)
        self.weight2 = nn.Parameter(torch.randn(4, 32, 32, device="cuda") / 8)
        self.bias1 = nn.Parameter(torch.randn(4, 32, device="cuda") / 8)
        self.bias2 = nn.Parameter(torch.randn(4, 32, device="cuda") / 8)
        self.resid_gate = nn.Parameter(torch.randn(32, device="cuda"))
        self.register_buffer("offset", torch.randn(32, device="cuda") / 8)
        self.head = nn.Linear(32, 6, device="cuda")

    def features(self, x, *, scale=1.0):
        entry = self.entry(x)
        hidden = torch.einsum("bi,eoi->beo", entry, self.weight1) + self.bias1
        hidden = torch.relu(hidden).square()
        expert = torch.einsum("bei,eoi->beo", hidden, self.weight2) + self.bias2
        features = entry + expert.mean(1) * torch.sigmoid(self.resid_gate) + self.offset
        return self.head(features) * scale, hidden, features

    def forward(self, x, *, scale=1.0):
        return self.features(x, scale=scale)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("compiled", [False, True])
def test_exact_outputs_refresh_live_fp32_state_and_rng(compiled):
    torch.manual_seed(1)
    model = _MixedPrecisionModel()
    names = linear_parameter_names(model) + ("weight1", "weight2")
    original = {name: (p, p.detach().clone()) for name, p in model.named_parameters()}
    rng = torch.cuda.get_rng_state().clone()
    cpu_rng = torch.get_rng_state().clone()
    cache = InferenceParameterCache(model, names, method="features")
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    addresses = {name: value.data_ptr() for name, value in cache.cached_parameters.items()}
    assert set(cache.cached_parameters) == set(names)
    for name, (parameter, snapshot) in original.items():
        assert model.get_parameter(name) is parameter
        assert parameter.dtype == torch.float32
        torch.testing.assert_close(parameter, snapshot, rtol=0, atol=0)
    for name, value in cache.cached_parameters.items():
        assert not value.requires_grad
        assert value.stride() == model.get_parameter(name).stride()
    reference = torch.compile(model.features, fullgraph=True) if compiled else model.features
    candidate = torch.compile(cache, fullgraph=True) if compiled else cache
    x = torch.randn(16, 17, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Compile/warm both before testing RNG neutrality.
        reference(x, scale=0.75)
        candidate(x, scale=0.75)
    for iteration in range(3):
        # A new autocast scope also invalidates eager PyTorch's own cast cache.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            rng = torch.cuda.get_rng_state().clone()
            model.bias1.add_(0.125)  # Unselected state is live without refresh.
            model.offset.add_(0.0625)
            if iteration:
                model.entry.weight.add_(0.25)
                model.weight2.mul_(0.75)
                cache.refresh()
            expected = tuple(value.clone() for value in reference(x, scale=0.75))
            actual = candidate(x, scale=0.75)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert actual[1].dtype == actual[2].dtype == torch.float32
            assert torch.equal(torch.cuda.get_rng_state(), rng)
            assert {name: value.data_ptr() for name, value in cache.cached_parameters.items()} == addresses
    for name, (parameter, _) in original.items():
        assert model.get_parameter(name) is parameter
        assert parameter.dtype == torch.float32
        assert parameter.grad is None


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cache_requires_explicit_refresh_and_matching_inference_context():
    model = nn.Linear(3, 2, device="cuda")
    cache = InferenceParameterCache(model, linear_parameter_names(model))
    x = torch.ones(2, 3, device="cuda")
    with pytest.raises(RuntimeError, match="no_grad"):
        cache(x)
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="matching CUDA autocast"):
            cache(x)
        with torch.autocast("cuda", dtype=torch.float16):
            with pytest.raises(RuntimeError, match="matching CUDA autocast"):
                cache(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            before = cache(x).clone()
            model.weight.add_(1)
            torch.testing.assert_close(cache(x), before, rtol=0, atol=0)
            cache.refresh()
            torch.testing.assert_close(cache(x), model(x), rtol=0, atol=0)
            assert not torch.equal(cache(x), before)
    model.weight = nn.Parameter(model.weight.detach().clone())
    with pytest.raises(RuntimeError, match="identity or metadata"):
        cache.refresh()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tied_cache_strides_and_exception_restore_original_parameters():
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 3, device="cuda").T)
            self.alias = self.weight

        def forward(self, x, *, fail=False):
            if fail:
                raise ValueError("intentional forward failure")
            return x @ self.weight + x @ self.alias

    model = Tied()
    weight = model.weight
    cache = InferenceParameterCache(model, ["weight", "alias"])
    assert cache.cached_parameters["weight"] is cache.cached_parameters["alias"]
    assert cache.cached_parameters["weight"].stride() == weight.stride()
    x = torch.randn(2, 3, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        torch.testing.assert_close(cache(x), model(x), rtol=0, atol=0)
        with pytest.raises(ValueError, match="intentional"):
            cache(x, fail=True)
    assert model.weight is model.alias is weight


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_submodule_cache_restores_originals_after_repeated_calls_and_refresh():
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(3, 2, device="cuda")
            self.tied = self.first

        def forward(self, x):
            return self.first(x) + self.tied(x)

    model = Shared()
    originals = dict(model.named_parameters(remove_duplicate=False))
    cache = InferenceParameterCache(model, linear_parameter_names(model))
    x = torch.randn(2, 3, device="cuda")
    for _ in range(3):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.first.weight.add_(0.25)
            cache.refresh()
            torch.testing.assert_close(cache(x), model(x), rtol=0, atol=0)
        for name, parameter in originals.items():
            assert model.get_parameter(name) is parameter
            assert parameter.dtype == torch.float32
        assert all(value.dtype == torch.bfloat16 for value in cache.cached_parameters.values())
