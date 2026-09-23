"""Prospective Memory Control, v1: learn which parameter changes to keep.

A write is an action, not evidence of a true coefficient. Later loss gradients
credit earlier writes/retention through eligibility traces. Both write rate and
retention are learned; refusing future writes alone cannot remove stored noise.
No posterior, covariance fit, support detector, replay, or expert mixture.

The main update is row-normalized implicit-gradient motion plus learned retention
of displacement from initialization. The most recent write receives its full
normalization-Jacobian credit; older forward sensitivities propagate diagonally.
This is not exact RTRL: older cross-coordinate sensitivities and derivatives
through adapting controls/curvature are omitted. Current actions use OLD controls;
the current outcome can change only the NEXT observation's controls.
"""

import math

import torch
from torch.optim import Optimizer


def _configuration(value, weight, name, *, allow_zero=False, upper=None):
    tensor = torch.as_tensor(value, device=weight.device, dtype=weight.dtype)
    lower_ok = tensor >= 0 if allow_zero else tensor > 0
    valid = torch.isfinite(tensor) & lower_ok
    if upper is not None:
        valid = valid & (tensor < upper)
    if not bool(valid.all()):
        raise ValueError(f"invalid {name}")
    if torch.broadcast_shapes(tensor.shape, weight.shape) != weight.shape:
        raise ValueError(f"{name} must broadcast into the parameter shape")
    return tensor


@torch.no_grad()
def _step(weight, grad, curvature, *, anchor, log_write, retain_logit,
          write_trace, retain_trace, write_square, retain_square,
          write_mass, retain_mass, last_write, last_curvature,
          meta_lr, meta_beta, control):
    # A_t is chosen before seeing this gradient. In log space the normalized
    # write remains finite even when a learned raw write rate is enormous.
    log_terms = log_write + curvature.log()
    largest, largest_index = log_terms.max(dim=-1, keepdim=True)
    shift = largest.clamp_min(0)
    scaled = torch.exp(log_terms - shift)
    leading = torch.exp(largest - shift)
    is_largest = torch.arange(weight.shape[-1], device=weight.device) == largest_index
    remainder = torch.where(is_largest, 0.0, scaled).sum(-1, keepdim=True)
    base = torch.exp(-shift)
    denominator = base + leading + remainder
    log_denom = shift + denominator.log()
    write = grad.sign() * torch.exp(log_write + grad.abs().log() - log_denom)
    own_curvature = scaled / denominator
    # Computing 1-own_curvature loses its derivative at a saturated write.
    # Sum the complement without first adding/subtracting its dominant entry.
    complement = (base + remainder + torch.where(
        is_largest, 0.0, leading - scaled)) / denominator
    active = (curvature > 0) | (grad != 0)
    forget_fraction = 0.0 if control == "write" else torch.sigmoid(-retain_logit)
    if control not in {"write", "wall"}:
        forget_fraction = torch.where(active, forget_fraction, 0.0)
    retention = 1.0 if control == "write" else (
        torch.sigmoid(retain_logit) if control == "wall" else
        torch.where(active, torch.sigmoid(retain_logit), 1.0))
    displacement = weight - anchor
    normalized_write = normalized_retain = 0.0

    if control != "fixed":
        # Score old eligibilities BEFORE replacing them. Current label noise
        # cannot choose the write/retention action used on that same observation.
        credit_write = write_trace
        credit_retain = retain_trace
        credit_last_write = last_write
        credit_last_curvature = last_curvature
        if control == "shuffle":
            credit_write = torch.roll(credit_write, 1, dims=-1)
            credit_retain = torch.roll(credit_retain, 1, dims=-1)
            credit_last_write = torch.roll(credit_last_write, 1, dims=-1)
            credit_last_curvature = torch.roll(credit_last_curvature, 1, dims=-1)
        local_previous = grad * credit_last_write
        hyper_write = grad * credit_write + credit_last_curvature * (
            local_previous.sum(-1, keepdim=True) - local_previous)
        hyper_retain = grad * credit_retain
        retain_activity = active.to(weight.dtype) * (1.0 - meta_beta)
        write_activity = (active | ((credit_last_curvature > 0)
                                   & active.any(-1, keepdim=True))).to(weight.dtype) * (1.0 - meta_beta)
        tiny = torch.finfo(weight.dtype).tiny
        jacobian_diagonal = complement - forget_fraction

        if control != "retain":
            write_mass.add_(write_activity * (1.0 - write_mass))
            write_square.add_(write_activity * (hyper_write.square() - write_square))
            normalized_write = hyper_write / (
                write_square / write_mass.clamp_min(tiny)).clamp_min(tiny).sqrt()
            write_trace.mul_(jacobian_diagonal).sub_(write * complement)
        if control != "write":
            retain_mass.add_(retain_activity * (1.0 - retain_mass))
            retain_square.add_(retain_activity * (hyper_retain.square() - retain_square))
            normalized_retain = hyper_retain / (
                retain_square / retain_mass.clamp_min(tiny)).clamp_min(tiny).sqrt()
            # Avoid 1-sigmoid(z): it rounds to zero while the derivative is
            # still representable. The complementary sigmoid retains it.
            retain_derivative = retention * forget_fraction
            retain_trace.mul_(jacobian_diagonal).add_(retain_derivative * displacement)

    weight.copy_(anchor + retention * displacement - write)
    if control not in {"fixed", "retain"}:
        last_write.copy_(write)
        last_curvature.copy_(own_curvature)
    if control != "fixed":
        if control != "retain":
            log_write.sub_(meta_lr * normalized_write)
        if control != "write":
            retain_logit.sub_(meta_lr * normalized_retain)


class ProspectiveMemoryState:
    """O(parameters) streaming controller, with no access to features or targets.

    Parameters represent independent scalar-output rows (..., fan_in). `grad`
    and nonnegative `curvature` must refer to the SAME preupdate model and loss
    scaling. For half-squared linear loss these are residual*x and x*x.
    Curvature is required, so this is not advertised as a gradient-only drop-in.

    `retain_rate` is the initial fraction forgotten per participating observation.
    No curvature/gradient means no evidence about that coordinate, so absence does
    not erase it. The `wall` ablation instead forgets on every wall-time step.
    Write RMS also counts prior normalization actions affecting observed peers.
    Controls parameterize a learning policy,
    not a belief about coefficient sparsity or a known support count.
    """

    def __init__(self, weight, *, initial_lr=0.01, meta_lr=0.01,
                 retain_rate=0.001, control="full", meta_beta=0.99):
        if not weight.is_cuda or weight.dtype != torch.float32:
            raise ValueError("CUDA FP32 master parameters are required")
        if weight.ndim < 1 or min(weight.shape) < 1:
            raise ValueError("parameters need a nonempty final coordinate axis")
        if control not in {"full", "write", "retain", "fixed", "shuffle", "wall"}:
            raise ValueError("unknown prospective control")
        if not math.isfinite(meta_beta) or not 0 <= meta_beta < 1:
            raise ValueError("meta_beta must lie in [0, 1)")
        rate = _configuration(initial_lr, weight, "initial_lr")
        retain = _configuration(retain_rate, weight, "retain_rate", upper=1.0)
        self.meta_lr = _configuration(meta_lr, weight, "meta_lr", allow_zero=True)
        self.meta_beta, self.control = meta_beta, control
        self.anchor = weight.detach().clone()
        self.log_write = rate.log().expand_as(weight).clone()
        self.retain_logit = (torch.log1p(-retain) - retain.log()).expand_as(weight).clone()
        self.write_trace = torch.zeros_like(weight)
        self.retain_trace = torch.zeros_like(weight)
        self.write_square = torch.zeros_like(weight)
        self.retain_square = torch.zeros_like(weight)
        self.write_mass = torch.zeros_like(weight)
        self.retain_mass = torch.zeros_like(weight)
        self.last_write = torch.zeros_like(weight)
        self.last_curvature = torch.zeros_like(weight)

    def buffers(self):
        return (self.log_write, self.retain_logit, self.write_trace,
                self.retain_trace, self.write_square, self.retain_square,
                self.write_mass, self.retain_mass, self.last_write,
                self.last_curvature)

    @torch.no_grad()
    def step(self, weight, grad, curvature):
        _step(weight, grad, curvature, anchor=self.anchor,
              log_write=self.log_write, retain_logit=self.retain_logit,
              write_trace=self.write_trace, retain_trace=self.retain_trace,
              write_square=self.write_square, retain_square=self.retain_square,
              write_mass=self.write_mass, retain_mass=self.retain_mass,
              last_write=self.last_write, last_curvature=self.last_curvature,
              meta_lr=self.meta_lr,
              meta_beta=self.meta_beta, control=self.control)


class ProspectiveMemory(Optimizer):
    """Torch interface: step(curvatures=[...]) in parameter-group order.

    Controls and eligibility state survive state_dict/load_state_dict. Curvature
    metadata must use the same preupdate parameters as each gradient. The diagonal
    eligibility approximation is local to the tensor's final parameter axis.
    """

    def __init__(self, params, *, initial_lr=0.01, meta_lr=0.01,
                 retain_rate=0.001, control="full", meta_beta=0.99):
        super().__init__(params, dict(initial_lr=initial_lr, meta_lr=meta_lr,
                                     retain_rate=retain_rate, control=control,
                                     meta_beta=meta_beta))
        for group in self.param_groups:
            for parameter in group["params"]:
                self._initialize(parameter, group)

    def _initialize(self, parameter, group):
        controller = ProspectiveMemoryState(
            parameter, **{key: group[key] for key in (
                "initial_lr", "meta_lr", "retain_rate", "control", "meta_beta")})
        self.state[parameter].update({key: value for key, value in vars(controller).items()
                                      if isinstance(value, torch.Tensor)})

    @torch.no_grad()
    def step(self, closure=None, *, curvatures=None):
        if curvatures is None:
            raise ValueError("curvatures are required for prospective memory")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        parameters = [(group, parameter) for group in self.param_groups
                      for parameter in group["params"]]
        curvatures = list(curvatures)
        if len(curvatures) != len(parameters):
            raise ValueError("one curvature entry required per parameter")
        for (group, parameter), curvature in zip(parameters, curvatures):
            if parameter.grad is None:
                continue
            if parameter.grad.is_sparse:
                raise ValueError("sparse gradient layout is unsupported")
            if not self.state[parameter]:
                self._initialize(parameter, group)
            _step(parameter, parameter.grad, curvature, **self.state[parameter],
                  meta_beta=group["meta_beta"], control=group["control"])
        return loss
