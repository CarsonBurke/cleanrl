"""v10's batched ``PlasticityStepper.levels`` must be v9's loop, byte for byte.

v10 fuses each width group's ``snr``/``lam``/``stats`` into one 2D buffer whose
rows are the layers' own buffers, so ``levels`` costs O(1) launches per group
instead of ~10 per layer. The dangerous failure mode is silent: the two
centerings are PER-LAYER cross-unit means, and a reduction over the wrong axis
turns them into a global mean across every layer. That still trains, still
produces plausible ``lam_mean``/``lam_std`` telemetry, and destroys the
mechanism -- the uniform direction becomes earnable and the per-layer
``prod(lam)**(1/n) == 1`` invariant is gone.

So every check here is exact-byte (``torch.equal``) against a local, verbatim
copy of v9's per-layer loop, and the sensitivity of that comparison is proven,
not assumed: the wrong-axis, permuted-row and shifted-buffer variants are all
asserted to give DIFFERENT answers on the same planted inputs.
"""

import importlib

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

V10 = importlib.import_module("cleanrl.plasticity.ppo_continuous_action_sphere_sdplast_v10")

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
              pytest.mark.cuda]


def make_args(**overrides):
    args = V10.Args()
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class FakeSite(nn.Module):
    """Exactly the attributes ``PlasticityStepper`` touches on a plastic layer."""

    def __init__(self, width, in_features=5, bias=True, device="cuda"):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(width, in_features, device=device))
        self.bias = nn.Parameter(torch.zeros(width, device=device)) if bias else None
        self.probe = nn.Parameter(torch.zeros(3, width, device=device))
        self.register_buffer("lam", torch.ones(width, device=device))
        self.register_buffer("snr", torch.zeros(width, device=device))
        self.register_buffer("stats", torch.zeros(4, device=device))


def build_stepper(widths, args, device="cuda"):
    """One stepper over ``widths`` (a list of unit counts, ragged on purpose)."""
    sites = [(f"site{index}", FakeSite(width, bias=index % 2 == 0, device=device))
             for index, width in enumerate(widths)]
    stepper = V10.PlasticityStepper(sites, args)
    return stepper, [module for _, module in sites]


def plant(layers, seed):
    """Per-layer SNR draws whose per-layer means differ from each other.

    A global reduction cannot reproduce a per-layer one on this input: each
    layer's log draws carry a distinct offset, and the offsets are deliberately
    NOT an arithmetic progression -- an evenly spaced ladder puts the middle
    layer exactly on the group mean, where a wrong-axis reduction would agree
    with the right one. Also plants sentinels in ``stats[0]`` and ``stats[3]``,
    which ``levels`` must never touch.
    """
    for index, layer in enumerate(layers):
        generator = torch.Generator(device=layer.snr.device).manual_seed(seed * 101 + index)
        draw = torch.rand(layer.snr.shape, device=layer.snr.device, generator=generator)
        offset = 0.7 * index + 1.5 * float(np.sin(index + 1.0))
        layer.snr.copy_(((draw * 8.0 - 4.0) + offset).exp())
        layer.stats[0] = -3.5 - index
        layer.stats[3] = 17.25 + index


@torch.no_grad()
def reference_levels(layers, args):
    """v9's ``levels`` body, verbatim, on standalone per-layer tensors.

    Standalone (freshly allocated, offset-zero) tensors are v9's exact buffer
    layout, so this is the arithmetic AND the memory layout the batched path
    must reproduce. Returns ``(lam, lam_mean, lam_std)`` per layer.
    """
    span = float(np.log(args.lam_span))
    expected = []
    for layer in layers:
        snr = layer.snr.clone()
        level = args.snr_exponent * snr.clamp_min(1e-30).log()
        level = level - level.mean()
        bounded = torch.tanh(args.lam_gain * level / span)
        bounded = bounded - bounded.mean()
        lam = torch.empty_like(snr)
        lam.copy_(torch.exp(span * bounded))
        expected.append((lam, lam.mean(), lam.std()))
    return expected


@torch.no_grad()
def wrong_axis_levels(layers, args, first_global, second_global):
    """The bug: one or both centerings reduced across every layer at once."""
    span = float(np.log(args.lam_span))
    widths = {layer.snr.numel() for layer in layers}
    assert len(widths) == 1, "the global variant needs one stackable width"
    snr = torch.stack([layer.snr for layer in layers])
    level = args.snr_exponent * snr.clamp_min(1e-30).log()
    level = level - (level.mean() if first_global else level.mean(1, keepdim=True))
    bounded = torch.tanh(args.lam_gain * level / span)
    bounded = bounded - (bounded.mean() if second_global else bounded.mean(1, keepdim=True))
    return torch.exp(span * bounded)


def assert_planting_is_adversarial(layers):
    """A wrong-axis reduction reduces over a whole WIDTH GROUP, so the planting
    must separate the layers inside each group. One-layer groups cannot be
    separated and are exempt: there the two axes coincide by definition.
    """
    by_width = {}
    for layer in layers:
        by_width.setdefault(layer.snr.numel(), []).append(layer)
    for width, members in by_width.items():
        if len(members) < 2:
            continue
        logs = torch.stack([layer.snr.clamp_min(1e-30).log() for layer in members])
        means = [float(value) for value in logs.mean(1)]
        assert len(set(means)) == len(means), f"width {width}: layers share an SNR mean"
        group_mean = float(logs.mean())
        assert all(abs(mean - group_mean) > 1e-3 for mean in means), \
            f"width {width}: a layer sits on the group mean, so the bug would pass"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("widths", [
    [64] * 8 + [43] * 12,              # the production HalfCheetah-v4 topology
    [64, 43, 43, 64, 32, 43, 64],      # ragged, interleaved widths
    [17] * 3 + [64] * 5 + [96] * 2,    # three groups, ragged sizes
    [7],                               # a single one-layer group
])
def test_batched_levels_is_bitwise_the_per_layer_loop(seed, widths):
    args = make_args(snr_warmup=1)
    stepper, layers = build_stepper(widths, args)
    plant(layers, seed)
    assert_planting_is_adversarial(layers)
    expected = reference_levels(layers, args)

    stepper.updates = args.snr_warmup  # next call is the first live one
    stepper.levels()

    for layer, (lam, lam_mean, lam_std) in zip(layers, expected):
        assert torch.equal(layer.lam, lam)
        assert torch.equal(layer.stats[1], lam_mean)
        assert torch.equal(layer.stats[2], lam_std)
        assert (layer.lam.double() - lam.double()).abs().max().item() == 0.0
    for index, layer in enumerate(layers):
        assert float(layer.stats[0]) == -3.5 - index   # untouched by levels
        assert float(layer.stats[3]) == 17.25 + index
        assert not torch.equal(layer.stats[1], layer.stats[2]), \
            "lam_mean and lam_std must differ, or a stats stride error would pass"


def test_production_topology_uses_the_batched_path():
    """A silent permanent fallback would keep the numerics and lose the win."""
    class StubEnvs:
        single_observation_space = gym.spaces.Box(-np.inf, np.inf, (17,), np.float64)
        single_action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)

    args = make_args(num_envs=16, num_steps=2048, num_minibatches=1)
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    agent = V10.Agent(StubEnvs(), args).to("cuda")
    stepper = V10.PlasticityStepper(agent.plastic_sites, args)

    assert len(stepper.layers) == 20
    assert all(group.batched for group in stepper.groups)
    assert sorted(tuple(group.lam.shape) for group in stepper.groups) == [(8, 64), (12, 43)]
    for group in stepper.groups:
        for index, layer in enumerate(group.layers):
            assert layer.lam.data_ptr() == group.lam[index].data_ptr()
            assert layer.stats.data_ptr() == group.stats[index].data_ptr()
            assert "lam" in layer._buffers and "stats" in layer._buffers
            # snr stays standalone: mutated views alias inside the compiled gate
            # model and cost it its cudagraphs.
            assert layer.snr._base is None
            assert group.snrs[index] is layer.snr

    plant(stepper.layers, 5)
    expected = reference_levels(stepper.layers, args)
    stepper.updates = args.snr_warmup
    stepper.levels()
    for layer, (lam, lam_mean, lam_std) in zip(stepper.layers, expected):
        assert torch.equal(layer.lam, lam)
        assert torch.equal(layer.stats[1], lam_mean)
        assert torch.equal(layer.stats[2], lam_std)


def test_wide_group_is_bitwise_or_falls_back_to_standalone_buffers():
    """Row reductions do diverge from full ones at large widths; refuse those.

    1024-wide groups are exactly where a CUDA row reduction picks a different
    accumulation split than the (width,) full reduction, so this asserts the
    construction-time check catches it and that the fallback is still bitwise.
    """
    args = make_args(snr_warmup=1)
    stepper, layers = build_stepper([1024] * 20, args)
    plant(layers, 9)
    expected = reference_levels(layers, args)
    stepper.updates = args.snr_warmup
    stepper.levels()
    for layer, (lam, lam_mean, lam_std) in zip(layers, expected):
        assert torch.equal(layer.lam, lam)
        assert torch.equal(layer.stats[1], lam_mean)
        assert torch.equal(layer.stats[2], lam_std)
    fell_back = [group for group in stepper.groups if not group.batched]
    assert fell_back, "a 1024-wide group must not pass the bitwise check on this device"
    for group in fell_back:
        assert group.lam is None and group.stats is None
        for layer in group.layers:
            assert layer.lam._base is None, "fallback layers keep standalone buffers"
            assert layer.stats._base is None


@pytest.mark.parametrize("first_global,second_global", [(True, False), (False, True), (True, True)])
def test_a_global_reduction_axis_is_caught(first_global, second_global):
    """Proof the equality checks bind: the wrong axis gives other numbers."""
    args = make_args(snr_warmup=1)
    stepper, layers = build_stepper([64] * 8, args)
    plant(layers, 4)
    assert_planting_is_adversarial(layers)
    wrong = wrong_axis_levels(layers, args, first_global, second_global)

    stepper.updates = args.snr_warmup
    stepper.levels()
    correct = torch.stack([layer.lam for layer in layers])
    assert not torch.equal(correct, wrong)
    assert (correct - wrong).abs().max().item() > 1e-4
    if second_global:
        # The signature of the destroyed mechanism: per-layer geometric means
        # stop being one, so a uniform level is earnable again.
        geometric = wrong.double().log().mean(1).exp()
        assert (geometric - 1.0).abs().max().item() > 1e-3


def test_permuted_or_shifted_rows_are_caught():
    """A gather/scatter off by one row must not be able to pass the identity."""
    args = make_args(snr_warmup=1)
    stepper, layers = build_stepper([64] * 6, args)
    plant(layers, 6)
    stepper.updates = args.snr_warmup
    stepper.levels()
    correct = torch.stack([layer.lam for layer in layers])
    for shift in (1, 2, 5):
        rolled = torch.roll(correct, shift, dims=0)
        assert not torch.equal(correct, rolled)
        assert (correct - rolled).abs().max().item() > 1e-4
    stats = torch.stack([layer.stats for layer in layers])
    assert not torch.equal(stats[:, 1], torch.roll(stats[:, 1], 1))


def test_warmup_and_disabled_regimes_keep_the_counter_semantics():
    args = make_args(snr_warmup=5)
    stepper, layers = build_stepper([64] * 4 + [43] * 3, args)
    plant(layers, 7)
    for update in range(args.snr_warmup):
        stepper.levels()
        assert stepper.updates == update + 1
        for layer in layers:
            assert torch.equal(layer.lam, torch.ones_like(layer.lam))
            assert float(layer.stats[1]) == 0.0 and float(layer.stats[2]) == 0.0

    expected = reference_levels(layers, args)
    stepper.levels()  # updates == snr_warmup + 1: the first live update
    assert stepper.updates == args.snr_warmup + 1
    for layer, (lam, lam_mean, lam_std) in zip(layers, expected):
        assert torch.equal(layer.lam, lam)
        assert torch.equal(layer.stats[1], lam_mean)
        assert torch.equal(layer.stats[2], lam_std)

    disabled_args = make_args(snr_level=False, snr_warmup=2)
    disabled, disabled_layers = build_stepper([64] * 3, disabled_args)
    plant(disabled_layers, 8)
    assert disabled.groups == []
    for update in range(7):
        disabled.levels()
        assert disabled.updates == update + 1  # counter runs even when disabled
    for layer in disabled_layers:
        assert torch.equal(layer.lam, torch.ones_like(layer.lam))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_geometric_mean_of_lam_is_one_per_layer(seed):
    args = make_args(snr_warmup=1)
    stepper, layers = build_stepper([64] * 5 + [43] * 4, args)
    plant(layers, seed)
    stepper.updates = args.snr_warmup
    stepper.levels()
    for layer in layers:
        geometric = float(layer.lam.double().log().mean().exp())
        assert abs(geometric - 1.0) < 1e-6
        assert float(layer.lam.std()) > 1e-3, "a flat lam would satisfy this trivially"


def test_apply_levels_is_bit_exact_for_a_neutral_unit():
    args = make_args(snr_warmup=1)
    stepper, layers = build_stepper([8, 8], args)
    plant(layers, 3)
    stepper.updates = args.snr_warmup
    stepper.levels()
    layers[1].lam.fill_(1.0)  # a fully neutral layer

    before = [param.clone() for param in stepper.params]
    stepper.stash()
    with torch.no_grad():
        for index, param in enumerate(stepper.params):
            param.add_(torch.full_like(param, 0.125 * (index + 1)))
    stepped = [param.clone() for param in stepper.params]
    stepper.apply_levels()

    for (layer, param, is_matrix), start, plain in zip(stepper.plan, before, stepped):
        if layer is layers[1]:
            assert torch.equal(param, plain)  # lam == 1: bit-exact untouched
            continue
        offset = layer.lam - 1.0
        gain = offset.unsqueeze(1) if is_matrix else offset
        assert torch.equal(param, plain + gain * (plain - start))
        assert not torch.equal(param, plain)


def test_clear_probes_drops_every_gradient():
    args = make_args()
    stepper, layers = build_stepper([8, 16], args)
    for layer in layers:
        layer.probe.grad = torch.ones_like(layer.probe)
    stepper.clear_probes()
    assert all(layer.probe.grad is None for layer in layers)


@pytest.mark.parametrize("value", [0, -1, -10])
def test_sdp_log_every_must_be_positive(value):
    with pytest.raises(ValueError, match="sdp_log_every"):
        V10.validate_args(make_args(sdp_log_every=value))


@pytest.mark.parametrize("value", [1, 2, 10])
def test_sdp_log_every_accepts_positive_cadences(value):
    assert V10.validate_args(make_args(sdp_log_every=value)).sdp_log_every == value
