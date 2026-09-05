"""Explicit autocast parameter caching for repeated, inference-only forwards.

Only select parameters whose every use already casts to the chosen autocast dtype.
For example, an expert GEMM weight is suitable; its separately added FP32 bias or
residual gate is not. This is not general-purpose model quantization.

Selection also needs numerical validation for every deployed compiled batch
shape: pre-casting can change compiler padding/layout choices and thus GEMM
rounding, even when the parameter and arithmetic dtypes appear identical.
This experimental utility is not automatically enabled by shared collectors.
"""

from collections.abc import Iterable, Mapping
from types import MappingProxyType

import torch
from torch import nn


def linear_parameter_names(module: nn.Module) -> tuple[str, ...]:
    """Names of weights/biases on exact nn.Linear modules, including tied aliases.

    Subclasses are deliberately excluded: their forward may use parameters in
    additional FP32 operations. Custom GEMM weights must be selected explicitly.
    """
    names = []
    for prefix, child in module.named_modules(remove_duplicate=False):
        if type(child) is nn.Linear:
            for name in ("weight", "bias"):
                if isinstance(child._parameters.get(name), nn.Parameter):
                    names.append(f"{prefix}.{name}" if prefix else name)
    return tuple(names)


def _unique_parameter_slots(module: nn.Module, parameters: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Swap each owner attribute once, even when submodules have several names.

    Distinct attributes holding a tied Parameter still each need a replacement;
    two paths to the *same* attribute must not be swapped/restored twice.
    """
    slots = set()
    replacements = {}
    for name, value in parameters.items():
        prefix, _, local_name = name.rpartition(".")
        slot = (id(module.get_submodule(prefix)), local_name)
        if slot not in slots:
            slots.add(slot)
            replacements[name] = value
    return replacements


class _MethodCall(nn.Module):
    def __init__(self, module: nn.Module, method: str):
        super().__init__()
        self.module = module
        self.method = method

    def forward(self, *args, **kwargs):
        if self.method == "forward":
            return self.module(*args, **kwargs)
        return getattr(self.module, self.method)(*args, **kwargs)


class InferenceParameterCache:
    """Persistent reduced-precision copies; all other model state stays live.

    Construct after moving an FP32 model to CUDA, then call under no_grad and
    matching CUDA autocast. Compile this callable with the surrounding inference
    function to eliminate Python functional_call overhead. Refresh after each
    optimizer update or target promotion, before the next inference, including
    captured optimizer updates that may not increment tensor version counters.
    Cache freshness is deliberately never inferred from tensor version counters.
    Refresh uses
    existing allocations and performs no host synchronization or RNG draws.

    Inference, refresh, and model updates must be ordered on the same CUDA stream
    (or explicitly synchronized by the caller). Do not concurrently call/train the
    original module from another host thread: functional_call temporarily swaps
    selected attributes in eager execution/tracing. Parameters must not be mutated
    by forward. Buffers and unselected parameters retain original dtype, identity,
    and normal module semantics, including buffer mutation and training mode.

    Parameter identities/shapes/devices must stay fixed for compiled graphs;
    replacing parameters requires rebuilding the cache and compiled callable.
    No optimizer should receive the inference-only cached tensors.
    """

    def __init__(
        self,
        module: nn.Module,
        parameter_names: Iterable[str],
        *,
        dtype: torch.dtype = torch.bfloat16,
        method: str = "forward",
    ):
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("cache dtype must be bfloat16 or float16")
        if not isinstance(module, nn.Module):
            raise TypeError("module must be an nn.Module")
        if not isinstance(method, str) or not callable(getattr(module, method, None)):
            raise ValueError("method must name a callable on the module")
        if isinstance(parameter_names, str):
            raise TypeError("parameter_names must be an iterable of full parameter names")
        names = tuple(parameter_names)
        if not names or any(not isinstance(name, str) for name in names):
            raise ValueError("select at least one parameter by its full string name")
        if len(set(names)) != len(names):
            raise ValueError("parameter_names must not contain duplicates")
        parameters = dict(module.named_parameters(remove_duplicate=False))
        unknown = set(names) - parameters.keys()
        if unknown:
            raise ValueError(f"unknown parameter names: {sorted(unknown)}")
        selected_ids = {id(parameters[name]) for name in names}
        unselected_aliases = [name for name, p in parameters.items() if id(p) in selected_ids and name not in names]
        buffer_aliases = [name for name, b in module.named_buffers(remove_duplicate=False) if id(b) in selected_ids]
        if unselected_aliases or buffer_aliases:
            raise ValueError("select every tied parameter alias; selected parameters cannot alias buffers")
        sources = tuple(dict.fromkeys(parameters[name] for name in names))
        if any(p.device.type != "cuda" or p.dtype != torch.float32 or p.layout != torch.strided for p in sources):
            raise ValueError("selected parameters must be strided FP32 CUDA tensors")
        if len({p.device for p in sources}) != 1:
            raise ValueError("selected parameters must share one CUDA device")
        self.module = module
        self.parameter_names = names
        self.dtype = dtype
        self._sources = sources
        copies = {
            id(p): torch.empty_strided(p.shape, p.stride(), dtype=dtype, device=p.device)
            for p in sources
        }
        # CUDA graphs must consume these persistent tensors in place, like
        # parameters. Otherwise every replay copies all cached weights into
        # graph input buffers, replacing cheap casts with dozens of copies.
        for cached in copies.values():
            torch._dynamo.mark_static_address(cached, guard=True)
        self._destinations = tuple(copies[id(p)] for p in sources)
        self._cached = {name: copies[id(parameters[name])] for name in names}
        self._replacements = {
            f"module.{name}": value for name, value in _unique_parameter_slots(module, self._cached).items()
        }
        self._call_module = _MethodCall(module, method)
        self._source_by_name = {name: parameters[name] for name in names}
        self.refresh()

    @property
    def cached_parameters(self) -> Mapping[str, torch.Tensor]:
        """Read-only mapping for diagnostics; do not modify its tensor contents."""
        return MappingProxyType(self._cached)

    @torch.no_grad()
    def refresh(self) -> None:
        """Copy current selected FP32 weights into persistent autocast storage."""
        for name, source in self._source_by_name.items():
            current = self.module.get_parameter(name)
            cached = self._cached[name]
            if (
                current is not source
                or current.shape != cached.shape
                or current.stride() != cached.stride()
                or current.device != cached.device
                or current.dtype != torch.float32
            ):
                raise RuntimeError("selected parameter identity or metadata changed; rebuild the inference cache")
        torch._foreach_copy_(self._destinations, self._sources)

    def __call__(self, *args, **kwargs):
        if torch.is_grad_enabled():
            raise RuntimeError("inference parameter cache requires no_grad or inference_mode")
        if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != self.dtype:
            raise RuntimeError("inference parameter cache requires matching CUDA autocast dtype")
        return torch.func.functional_call(
            self._call_module, self._replacements, args, kwargs, tie_weights=False, strict=False
        )
