"""Contracts for the shared GatePolar optimizer: it reproduces the plasticity proxy model v12 `gate_polar`
rule in float64, degenerates to polar / Adam at weight_decay 0, has exact gate limits, compiles without
drift, keeps device LR/step tensors, and validates its groups. Queue through mlq, not direct pytest."""

import copy

import pytest
import torch

from cleanrl.plasticity import optimizer_proxy_model_v12 as proxy
from cleanrl.shared import gate_polar as module
from cleanrl.shared.gate_polar import GatePolar, mlp_groups
from tests.test_optimizer_proxy_v9 import state as proxy_state

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

SHAPES = [(4, 3), (4, 4), (1, 4)]
HYPER = dict(lr=0.003, beta1=0.9, beta2=0.999, weight_decay=0.1, head_lr_scale=2.0, gate_beta=0.99)


def build_mlp(dtype=torch.float64, seed=101):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    layers = []
    for index, (o, i) in enumerate(SHAPES):
        linear = torch.nn.Linear(i, o, dtype=dtype, device="cuda")
        with torch.no_grad():
            linear.weight.copy_(torch.randn(o, i, generator=gen, device="cuda", dtype=dtype) * .1)
            linear.bias.copy_(torch.randn(o, generator=gen, device="cuda", dtype=dtype) * .1)
        layers.append(linear)
        if index < len(SHAPES) - 1:
            layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers)


def proxy_weights(mlp):
    """[1, O, I+1] layers with the bias as the last incoming coordinate."""
    linears = [layer for layer in mlp if isinstance(layer, torch.nn.Linear)]
    return [torch.cat((layer.weight.detach(), layer.bias.detach().unsqueeze(-1)), dim=-1).unsqueeze(0).clone()
            for layer in linears]


def stream(mlp, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    linears = [layer for layer in mlp if isinstance(layer, torch.nn.Linear)]
    return [(torch.randn(layer.weight.shape, generator=gen, device="cuda", dtype=layer.weight.dtype) * .2,
             torch.randn(layer.bias.shape, generator=gen, device="cuda", dtype=layer.bias.dtype) * .2) for layer in linears]


def apply_grads(mlp, grads):
    for layer, (gw, gb) in zip([l for l in mlp if isinstance(l, torch.nn.Linear)], grads):
        layer.weight.grad = gw.clone()
        layer.bias.grad = gb.clone()


def roll_proxy(mlp, grads_per_step, method, hyper, dtype=torch.float64):
    """Proxy v12 transition on the same start and gradients; returns [O, I+1] layers per step."""
    w = proxy_weights(mlp)
    _, _, _, _, base_hyper = proxy_state(dtype, 1)
    hyper = {**base_hyper, **{key: w[0].new_tensor(value).reshape(1, 1, 1) for key, value in hyper.items()}}
    m = [torch.zeros_like(x) for x in w]
    v = [torch.zeros_like(x) for x in w]
    aux = proxy.initial_aux(w)
    step = torch.tensor(0, device="cuda", dtype=torch.int64)
    history = []
    for grads in grads_per_step:
        g = [torch.cat((gw, gb.unsqueeze(-1)), dim=-1).unsqueeze(0) for gw, gb in grads]
        w, m, v, aux, step = proxy.transition(w, [x.clone() for x in w], m, v, aux, step, g,
                                              [torch.zeros_like(x) for x in g], hyper, method, 1, 32)
        history.append([x[0].clone() for x in w])
    return history


def roll_torch(mlp, grads_per_step, optimizer):
    history = []
    for grads in grads_per_step:
        apply_grads(mlp, grads)
        optimizer.step()
        history.append(proxy_weights(mlp))
    return [[x[0] for x in step] for step in history]


def make_optimizer(mlp, hyper, compile=False, **overrides):
    return GatePolar(mlp_groups(mlp, head_lr_scale=hyper["head_lr_scale"]), lr=hyper["lr"],
                     betas=(hyper["beta1"], hyper["beta2"]), eps=1e-8, weight_decay=hyper["weight_decay"],
                     gate_beta=hyper["gate_beta"], compile=compile, **overrides)


@pytest.mark.parametrize("weight_decay,method", [(0.1, "gate_polar"), (0.0, "polar")])
def test_reproduces_proxy_v12_rule_in_float64(weight_decay, method):
    hyper = {**HYPER, "weight_decay": weight_decay}
    mlp = build_mlp()
    grads = [stream(mlp, 300 + t) for t in range(8)]
    expected = roll_proxy(mlp, grads, method, hyper)
    actual = roll_torch(mlp, grads, make_optimizer(mlp, hyper))
    for step_expected, step_actual in zip(expected, actual):
        for a, b in zip(step_actual, step_expected):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)


def test_adam_groups_match_torch_adam_at_zero_decay():
    mlp = build_mlp()
    twin = copy.deepcopy(mlp)
    reference = torch.optim.Adam(twin.parameters(), lr=0.003, betas=(0.9, 0.999), eps=1e-5)
    candidate = GatePolar([{"params": list(mlp.parameters()), "polar": False}], lr=0.003, betas=(0.9, 0.999),
                          eps=1e-5, weight_decay=0.0, compile=False)
    for t in range(6):
        grads = stream(mlp, 500 + t)
        apply_grads(mlp, grads)
        apply_grads(twin, grads)
        candidate.step()
        reference.step()
    for a, b in zip(mlp.parameters(), twin.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)


def test_gate_limits_are_exact():
    """Zero-variance stream agreeing with shrinking: gate 1 (plain decoupled decay). Resisting: gate 0."""
    hyper = {**HYPER, "gate_beta": 0.9}
    for sign, decayed in ((1.0, True), (-1.0, False)):
        mlp = build_mlp()
        with torch.no_grad():
            for p in mlp.parameters():
                p.copy_(torch.sign(p) * (p.abs() + 0.05))
        # gradient sign(w) shrinks toward zero (agrees with decay); -sign(w) grows (resists).
        grads = [[(sign * torch.sign(l.weight.detach()) * 0.3, torch.zeros_like(l.bias)) for l in mlp
                  if isinstance(l, torch.nn.Linear)] for _ in range(4)]
        control = copy.deepcopy(mlp)
        control_opt = make_optimizer(control, {**hyper, "weight_decay": 0.0})
        candidate = make_optimizer(mlp, hyper)
        linears = [l for l in mlp if isinstance(l, torch.nn.Linear)]
        control_linears = [l for l in control if isinstance(l, torch.nn.Linear)]
        for step_grads in grads:
            apply_grads(control, step_grads)
            apply_grads(mlp, step_grads)
            before = [l.weight.detach().clone() for l in linears]
            control_before = [l.weight.detach().clone() for l in control_linears]
            control_opt.step()
            candidate.step()
            # The direction depends on the gradient stream only, so the candidate's step is the
            # control's displacement plus the gated pull on the candidate's own pre-step weights.
            for index, (layer, ref, w0, c0) in enumerate(zip(linears, control_linears, before, control_before)):
                lr = hyper["lr"] * (hyper["head_lr_scale"] if index == len(linears) - 1 else 1.0)
                pull = lr * hyper["weight_decay"] * w0 if decayed else torch.zeros_like(w0)
                torch.testing.assert_close(layer.weight, w0 + (ref.weight - c0) - pull, rtol=1e-12, atol=1e-14)
                torch.testing.assert_close(layer.bias, ref.bias, rtol=0, atol=0)


def test_pure_noise_gate_is_near_one_half_on_average():
    gen = torch.Generator(device="cuda").manual_seed(9)
    s = torch.randn(2048, 64, generator=gen, device="cuda", dtype=torch.float64) * 0.05
    q = s.square() + 1.0
    w = torch.randn_like(s)
    gate = module.resistance_gate(s, q, w, torch.tensor(0.99, device="cuda", dtype=torch.float64),
                                  torch.tensor(5000.0, device="cuda", dtype=torch.float64))
    assert 0.48 < gate.mean().item() < 0.52
    assert gate.min().item() >= 0 and gate.max().item() <= 1


@pytest.mark.parametrize("dtype,rtol,atol", [(torch.float64, 1e-10, 1e-12), (torch.float32, 1e-3, 1e-5)])
def test_compiled_update_matches_eager(dtype, rtol, atol):
    """Five Newton-Schulz steps amplify reassociation rounding by about 3.4^5, so float32 gets a loose bound."""
    eager_mlp = build_mlp(dtype)
    compiled_mlp = copy.deepcopy(eager_mlp)
    eager = make_optimizer(eager_mlp, HYPER, compile=False)
    compiled = make_optimizer(compiled_mlp, HYPER, compile=True)
    for t in range(6):
        grads = stream(eager_mlp, 700 + t)
        apply_grads(eager_mlp, grads)
        apply_grads(compiled_mlp, grads)
        eager.step()
        compiled.step()
        eager.set_lr(HYPER["lr"] * (1 - t / 10))
        compiled.set_lr(HYPER["lr"] * (1 - t / 10))
    for a, b in zip(compiled_mlp.parameters(), eager_mlp.parameters()):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def test_set_lr_scales_groups_in_place_and_state_dict_round_trips():
    mlp = build_mlp()
    optimizer = make_optimizer(mlp, HYPER)
    tensors = [group["lr"] for group in optimizer.param_groups]
    optimizer.set_lr(0.001)
    assert [t.item() for t in tensors] == pytest.approx([0.001, 0.001, 0.002])
    apply_grads(mlp, stream(mlp, 1))
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    restored = make_optimizer(mlp, HYPER)
    restored.load_state_dict(saved)
    for group, twin in zip(optimizer.param_groups, restored.param_groups):
        assert twin["lr"].device.type == "cuda" and twin["step"].device.type == "cuda"
        assert twin["lr"].item() == group["lr"].item() and twin["step"].item() == 1
    for p in mlp.parameters():
        for key, value in optimizer.state[p].items():
            torch.testing.assert_close(restored.state[p][key], value, rtol=0, atol=0)


def test_missing_gradient_leaves_parameter_untouched_while_clock_advances():
    mlp = build_mlp()
    optimizer = make_optimizer(mlp, HYPER)
    linears = [l for l in mlp if isinstance(l, torch.nn.Linear)]
    apply_grads(mlp, stream(mlp, 2))
    frozen = linears[0].weight
    frozen.grad = None
    before = frozen.detach().clone()
    optimizer.step()
    torch.testing.assert_close(frozen, before, rtol=0, atol=0)
    assert frozen not in optimizer.state
    assert optimizer.param_groups[0]["step"].item() == 1


def test_group_validation():
    mlp = build_mlp()
    bias = [l for l in mlp if isinstance(l, torch.nn.Linear)][0].bias
    with pytest.raises(ValueError, match="matrices only"):
        GatePolar([{"params": [bias], "polar": True}], compile=False)
    with pytest.raises(ValueError, match="gate_beta"):
        GatePolar(mlp.parameters(), gate_beta=1.0, compile=False)
    with pytest.raises(ValueError, match="weight_decay"):
        GatePolar(mlp.parameters(), weight_decay=-0.1, compile=False)
    groups = mlp_groups(mlp)
    assert [len(g["params"]) for g in groups] == [2, 2, 2]
    assert groups[0]["polar"] and not groups[1]["polar"] and not groups[2]["polar"]


def test_float32_rejects_poles_that_round_to_one_and_float64_accepts_them():
    mlp32 = build_mlp(torch.float32)
    with pytest.raises(ValueError, match="below one"):
        GatePolar(mlp_groups(mlp32), gate_beta=1 - 1e-9, compile=False)
    GatePolar(mlp_groups(build_mlp(torch.float64)), gate_beta=1 - 1e-9, compile=False)


def test_decay_raised_after_the_first_step_allocates_the_gate_state():
    mlp = build_mlp()
    optimizer = make_optimizer(mlp, {**HYPER, "weight_decay": 0.0})
    apply_grads(mlp, stream(mlp, 3))
    optimizer.step()
    weight = [l for l in mlp if isinstance(l, torch.nn.Linear)][0].weight
    assert "s" not in optimizer.state[weight]
    for group in optimizer.param_groups:
        group["weight_decay"] = 0.1
    before = weight.detach().clone()
    apply_grads(mlp, stream(mlp, 4))
    optimizer.step()
    assert optimizer.state[weight]["s"].shape == weight.shape
    assert not torch.equal(weight, before)
