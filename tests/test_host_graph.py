"""Fused native host graph: FP32 parity with CUDA and with the NumPy mirrors."""

import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

try:  # the clip ufunc the frozen NumPy Beta head uses
    from numpy._core.umath import clip as ufunc_clip
except ImportError:  # NumPy < 2
    from numpy.core.umath import clip as ufunc_clip

from cleanrl.shared.host_actor import (
    CappedLeakyReluSq, CappedSignedSquare, HostLReluResActor,
    HostLReluSphereActor, HostMLP, HostSiTUDenseActor, HostSiTUResActor,
    HostSiTUSphereActor, LeakyReluSq, SiTUGLUBranch,
    make_caplrelusq_sphere_trunk, make_capsignsq_sphere_trunk,
    make_lrelu_res_trunk, make_lrelu_sphere_trunk, make_signsq_sphere_trunk,
    make_situ_dense_trunk, make_situ_res_trunk, make_situ_sphere_trunk,
)
from cleanrl.shared import host_graph
from cleanrl.shared.host_graph import HostGraphActor, make_host_mirror
from cleanrl.shared.runtime import configure_runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def device_forward(sequential, x):
    with torch.no_grad():
        return sequential(torch.as_tensor(x, device="cuda")).cpu().numpy()


def test_host_graph_matches_situsphere_device_forward_and_numpy_mirror():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(5)
    seq = nn.Sequential(make_situ_sphere_trunk(17, 64, 3), nn.Linear(64, 12)).cuda()
    x = np.random.default_rng(5).standard_normal((16, 17)).astype(np.float32)
    graph, mirror = HostGraphActor(seq, 16), HostSiTUSphereActor(seq, 16)
    assert (graph.num_rows, graph.in_features, graph.out_features) == (16, 17, 12)
    assert graph.device == mirror.device
    # Same tolerance as test_host_actor's sphere parity leg (max abs 6.0e-8).
    np.testing.assert_allclose(graph(x), device_forward(seq, x), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(graph(x), mirror(x), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        graph(x[:8])
    with pytest.raises(ValueError):
        graph(np.asfortranarray(x))


def test_host_graph_situsphere_refresh_tracks_in_place_updates():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(5)
    trunk = make_situ_sphere_trunk(17, 64, 3)
    seq = nn.Sequential(trunk, nn.Linear(64, 12)).cuda()
    x = np.random.default_rng(15).standard_normal((16, 17)).astype(np.float32)
    graph = HostGraphActor(seq, 16)
    stale = graph(x).copy()
    with torch.no_grad():
        trunk.block_gates[1].fill_(1.0)
        for parameter in seq.parameters():
            parameter.mul_(-1.5)
        expected = device_forward(seq, x)
    np.testing.assert_array_equal(graph(x), stale)  # not refreshed yet
    graph.refresh()
    # Perturbed weights inflate activations; looser leg proves tracking, and
    # matches the tolerance test_host_actor uses for the same perturbation.
    np.testing.assert_allclose(graph(x), expected, rtol=1e-3, atol=1e-4)


def test_host_graph_matches_situdense_device_forward_and_numpy_mirror():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(4)
    seq = nn.Sequential(make_situ_dense_trunk(17, 64, 3), nn.Linear(64, 12)).cuda()
    x = np.random.default_rng(4).standard_normal((16, 17)).astype(np.float32)
    graph, mirror = HostGraphActor(seq, 16), HostSiTUDenseActor(seq, 16)
    # Same tolerance as test_host_actor's dense parity leg (max abs 7.2e-7).
    np.testing.assert_allclose(graph(x), device_forward(seq, x), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(graph(x), mirror(x), rtol=1e-5, atol=1e-6)


def test_host_graph_matches_lrelusphere_device_forward_and_numpy_mirror():
    """Sphere geometry with LeakyReluSq pairs, at a width that is not a
    multiple of the 16-column gemm tile."""
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    for seed, (rows, obs, width, blocks) in enumerate(((16, 17, 64, 3), (13, 11, 43, 2))):
        torch.manual_seed(seed)
        seq = nn.Sequential(make_lrelu_sphere_trunk(obs, width, blocks),
                            nn.Linear(width, 12)).cuda()
        x = np.random.default_rng(seed).standard_normal((rows, obs)).astype(np.float32)
        graph, mirror = HostGraphActor(seq, rows), HostLReluSphereActor(seq, rows)
        assert (graph.num_rows, graph.in_features, graph.out_features) == (rows, obs, 12)
        # justnorm bounds the trunk output, so deviation stays ~1e-7 (mirror 6e-8).
        np.testing.assert_allclose(graph(x), device_forward(seq, x), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(graph(x), mirror(x), rtol=1e-5, atol=1e-6)


def test_host_graph_matches_signsqsphere_device_forward_and_numpy_mirror():
    """Same sphere trunk with the zero-mean SignedSquare pair: the fused
    CLEANRL_OP_SIGNSQ path and the NumPy fallback must agree with the device."""
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    for seed, (rows, obs, width, blocks) in enumerate(((16, 17, 64, 3), (13, 11, 43, 2))):
        torch.manual_seed(seed)
        seq = nn.Sequential(make_signsq_sphere_trunk(obs, width, blocks),
                            nn.Linear(width, 12)).cuda()
        x = np.random.default_rng(seed).standard_normal((rows, obs)).astype(np.float32)
        graph, mirror = HostGraphActor(seq, rows), HostLReluSphereActor(seq, rows)
        assert (graph.num_rows, graph.in_features, graph.out_features) == (rows, obs, 12)
        np.testing.assert_allclose(graph(x), device_forward(seq, x), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(graph(x), mirror(x), rtol=1e-5, atol=1e-6)


def test_host_graph_matches_capped_sphere_device_forward_and_numpy_mirror():
    """The tanh-capped squared pairs: the fused CLEANRL_OP_CAPSIGNSQ /
    CLEANRL_OP_CAPLRELUSQ paths (in-file tanh polynomial) and the NumPy
    fallback (np.tanh) must both agree with the device."""
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    shapes = ((16, 17, 64, 3), (13, 11, 43, 2))
    for factory in (make_capsignsq_sphere_trunk, make_caplrelusq_sphere_trunk):
        for seed, (rows, obs, width, blocks) in enumerate(shapes, start=41):
            torch.manual_seed(seed)
            seq = nn.Sequential(factory(obs, width, blocks), nn.Linear(width, 12)).cuda()
            x = np.random.default_rng(seed).standard_normal((rows, obs)).astype(np.float32)
            graph, mirror = HostGraphActor(seq, rows), HostLReluSphereActor(seq, rows)
            assert (graph.num_rows, graph.in_features, graph.out_features) == (rows, obs, 12)
            np.testing.assert_allclose(graph(x), device_forward(seq, x), rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(graph(x), mirror(x), rtol=1e-5, atol=1e-6)


def test_host_graph_rejects_a_nondefault_squared_pair_cap():
    """The fused ops bake in 4.0f, so any other cap must fail at construction."""
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(43)
    trunk = make_capsignsq_sphere_trunk(17, 64, 2)
    trunk.blocks[1].act = CappedSignedSquare(cap=8.0)
    with pytest.raises(ValueError):
        HostGraphActor(nn.Sequential(trunk, nn.Linear(64, 12)).cuda(), 16)
    trunk = make_caplrelusq_sphere_trunk(17, 64, 2)
    trunk.blocks[0].act = CappedLeakyReluSq(cap=2.0)
    with pytest.raises(ValueError):
        HostGraphActor(nn.Sequential(trunk, nn.Linear(64, 12)).cuda(), 16)


def test_host_graph_matches_situres_device_forward_and_numpy_mirror():
    """Layer-wide scalar gates broadcast into the per-channel mixing ops."""
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(11)
    trunk = make_situ_res_trunk(17, 64)
    seq = nn.Sequential(trunk, nn.Linear(64, 12)).cuda()
    x = np.random.default_rng(11).standard_normal((16, 17)).astype(np.float32)
    graph, mirror = HostGraphActor(seq, 16), HostSiTUResActor(seq, 16)
    # Unnormalized stream, same tolerance as the dense leg (max abs 7.2e-7).
    np.testing.assert_allclose(graph(x), device_forward(seq, x), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(graph(x), mirror(x), rtol=1e-5, atol=1e-6)
    with torch.no_grad():  # the long skip gate must be tracked, not baked in
        trunk.lam_skip.fill_(2.0)
        expected = device_forward(seq, x)
    graph.refresh()
    np.testing.assert_allclose(graph(x), expected, rtol=1e-5, atol=1e-6)


def test_host_graph_matches_lrelures_device_forward_and_numpy_mirror():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    for seed, (rows, obs, width) in enumerate(((16, 17, 64), (13, 11, 37)), start=20):
        torch.manual_seed(seed)
        seq = nn.Sequential(make_lrelu_res_trunk(obs, width), nn.Linear(width, 12)).cuda()
        x = np.random.default_rng(seed).standard_normal((rows, obs)).astype(np.float32)
        graph, mirror = HostGraphActor(seq, rows), HostLReluResActor(seq, rows)
        np.testing.assert_allclose(graph(x), device_forward(seq, x), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(graph(x), mirror(x), rtol=1e-5, atol=1e-6)


def test_host_graph_rejects_unmirrorable_new_trunk_variants():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(12)
    trunk = make_lrelu_sphere_trunk(17, 64, 2)
    trunk.blocks[1].act = nn.ReLU()  # only LeakyReluSq/SignedSquare pairs are mirrored
    with pytest.raises(TypeError):
        HostGraphActor(nn.Sequential(trunk, nn.Linear(64, 12)).cuda(), 16)
    trunk = make_lrelu_sphere_trunk(17, 64, 2)
    trunk.blocks[0] = SiTUGLUBranch(64, 64)  # wrong block family for this trunk
    with pytest.raises(TypeError):
        HostGraphActor(nn.Sequential(trunk, nn.Linear(64, 12)).cuda(), 16)
    trunk = make_lrelu_sphere_trunk(17, 64, 2)
    trunk.block_gates[1] = nn.Parameter(torch.full((1,), -1.5))  # layer-wide gate
    with pytest.raises(ValueError):
        HostGraphActor(nn.Sequential(trunk, nn.Linear(64, 12)).cuda(), 16)
    trunk = make_situ_res_trunk(17, 64)
    trunk.lam2 = nn.Parameter(torch.full((64,), -1.5))  # per-channel gate
    with pytest.raises(ValueError):
        HostGraphActor(nn.Sequential(trunk, nn.Linear(64, 12)).cuda(), 16)
    trunk = make_situ_res_trunk(17, 64)
    with pytest.raises(ValueError):  # the head must be biased
        HostGraphActor(nn.Sequential(trunk, nn.Linear(64, 12, bias=False)).cuda(), 16)
    trunk = make_lrelu_res_trunk(17, 64)
    trunk.pair2.lin1 = nn.Linear(64, 32)  # pair is width-preserving
    with pytest.raises(ValueError):
        HostGraphActor(nn.Sequential(trunk, nn.Linear(64, 12)).cuda(), 16)
    trunk = make_lrelu_res_trunk(17, 64)
    trunk.in_proj = nn.Linear(17, 64, bias=False)  # in_proj must be biased
    with pytest.raises(ValueError):
        HostGraphActor(nn.Sequential(trunk, nn.Linear(64, 12)).cuda(), 16)


def test_host_graph_matches_plain_mlp_device_forward_and_numpy_mirror():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(17, 64), nn.Tanh(), nn.Linear(64, 32), nn.ReLU(),
                        nn.Linear(32, 12)).cuda()
    x = np.random.default_rng(0).standard_normal((16, 17)).astype(np.float32)
    graph, mirror = HostGraphActor(net, 16), HostMLP(net, 16)
    np.testing.assert_allclose(graph(x), device_forward(net, x), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(graph(x), mirror(x), rtol=1e-5, atol=1e-6)
    stale = graph(x).copy()
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.mul_(-1.5)
        expected = device_forward(net, x)
    np.testing.assert_array_equal(graph(x), stale)  # not refreshed yet
    graph.refresh()
    np.testing.assert_allclose(graph(x), expected, rtol=1e-5, atol=1e-6)


def test_host_graph_matches_leakyrelusq_and_situglu_device_forwards():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    lrelu = nn.Sequential(nn.Linear(17, 64), LeakyReluSq(), nn.Linear(64, 12)).cuda()
    x = np.random.default_rng(1).standard_normal((16, 17)).astype(np.float32)
    np.testing.assert_allclose(HostGraphActor(lrelu, 16)(x), device_forward(lrelu, x),
                               rtol=1e-5, atol=1e-6)
    torch.manual_seed(2)
    glu = nn.Sequential(SiTUGLUBranch(17, 64), SiTUGLUBranch(64, 64), nn.Linear(64, 12)).cuda()
    x = np.random.default_rng(2).standard_normal((16, 17)).astype(np.float32)
    np.testing.assert_allclose(HostGraphActor(glu, 16)(x), device_forward(glu, x),
                               rtol=1e-5, atol=1e-6)


def test_host_graph_returns_one_reused_output_buffer():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(6)
    seq = nn.Sequential(make_situ_sphere_trunk(17, 64, 2), nn.Linear(64, 12)).cuda()
    graph = HostGraphActor(seq, 16)
    rng = np.random.default_rng(6)
    x = rng.standard_normal((16, 17)).astype(np.float32)
    first = graph(x)
    assert first.shape == (16, 12) and first.dtype == np.float32
    snapshot = first.copy()
    second = graph(rng.standard_normal((16, 17)).astype(np.float32))
    assert second is first
    assert not np.array_equal(second, snapshot)


def test_host_graph_matches_device_forward_on_ragged_shapes():
    """Row/column counts that hit every gemm remainder tile (8/4/1 and 16/8/4/1)."""
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(9)
    seq = nn.Sequential(make_situ_sphere_trunk(11, 37, 3), nn.Linear(37, 5)).cuda()
    x = np.random.default_rng(9).standard_normal((13, 11)).astype(np.float32)
    graph, mirror = HostGraphActor(seq, 13), HostSiTUSphereActor(seq, 13)
    np.testing.assert_allclose(graph(x), device_forward(seq, x), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(graph(x), mirror(x), rtol=1e-5, atol=1e-6)


def test_host_graph_rejects_unsupported_modules():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(7)
    with pytest.raises(TypeError):
        HostGraphActor(nn.Linear(3, 3).cuda(), 2)
    with pytest.raises(TypeError):
        HostGraphActor(nn.Sequential(nn.Linear(3, 3), nn.Sigmoid()).cuda(), 2)
    with pytest.raises(ValueError):
        HostGraphActor(nn.Sequential(nn.Linear(3, 3)).cuda(), 0)
    with pytest.raises(ValueError):  # host parameters are not mirrored
        HostGraphActor(nn.Sequential(nn.Linear(3, 3)), 2)
    with pytest.raises(ValueError):
        HostGraphActor(nn.Sequential(nn.Linear(3, 3), nn.Linear(4, 3)).cuda(), 2)


def test_host_graph_rejects_unmirrorable_trunk_variants():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(8)
    trunk = make_situ_sphere_trunk(17, 64, 2)
    biased = nn.Sequential(trunk, nn.Linear(64, 12, bias=False)).cuda()
    with pytest.raises(ValueError):
        HostGraphActor(biased, 16)
    trunk = make_situ_sphere_trunk(17, 64, 2)
    trunk.blocks[0].gate.bias = nn.Parameter(torch.zeros(trunk.blocks[0].hidden_dim))
    with pytest.raises(ValueError):
        HostGraphActor(nn.Sequential(trunk, nn.Linear(64, 12)).cuda(), 16)
    trunk = make_situ_sphere_trunk(17, 64, 2)
    trunk.block_gates[0] = nn.Parameter(torch.full((1,), -1.5))  # layer-wide scalar gate
    with pytest.raises(ValueError):
        HostGraphActor(nn.Sequential(trunk, nn.Linear(64, 12)).cuda(), 16)


def test_make_host_mirror_fuses_every_shipped_architecture():
    """No trainer architecture may silently take the 4-10x slower NumPy path."""
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(3)
    builders = (
        lambda: nn.Sequential(make_situ_sphere_trunk(17, 64, 3), nn.Linear(64, 12)),
        lambda: nn.Sequential(make_situ_dense_trunk(17, 64, 3), nn.Linear(64, 12)),
        lambda: nn.Sequential(make_situ_res_trunk(17, 64), nn.Linear(64, 12)),
        lambda: nn.Sequential(make_lrelu_sphere_trunk(17, 64, 3), nn.Linear(64, 12)),
        lambda: nn.Sequential(make_signsq_sphere_trunk(17, 64, 3), nn.Linear(64, 12)),
        lambda: nn.Sequential(make_capsignsq_sphere_trunk(17, 64, 3), nn.Linear(64, 12)),
        lambda: nn.Sequential(make_caplrelusq_sphere_trunk(17, 64, 3), nn.Linear(64, 12)),
        lambda: nn.Sequential(make_lrelu_res_trunk(17, 64), nn.Linear(64, 12)),
        lambda: nn.Sequential(nn.Linear(17, 64), LeakyReluSq(), nn.Linear(64, 12)),
    )
    for build in builders:
        sequential = build().cuda()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mirror = make_host_mirror(sequential, 16)
        assert isinstance(mirror, HostGraphActor), type(sequential[0]).__name__
        assert mirror.fused is True and mirror.fallback_reason is None
        assert [str(entry.message) for entry in caught] == []
    # A net neither builder can express stays a hard error, not a silent mirror
    # (make_host_mirror warns about the fallback first, then HostMLP rejects it).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(TypeError):
            make_host_mirror(
                nn.Sequential(nn.Linear(17, 8), nn.Sigmoid(), nn.Linear(8, 4)).cuda(), 4)


# -- Beta head ops -------------------------------------------------------------
#
# These replace NumPy arithmetic whose output feeds numpy.random.Generator.beta
# and the learner, so "a few ulp" is a failure: every assertion below is on raw
# bit patterns, and each is built so that a plausible bug (a half read at the
# wrong offset, a wrong row stride, a per-dimension bound collapsed to a single
# global one, one branch of logaddexp dropped) changes the bits.


def reference_concentration(logits):
    """The exact expression ``sample_beta_actions_host`` uses, frozen."""
    concentration = np.logaddexp(0.0, logits, dtype=np.float32)
    concentration += 1.0
    half = concentration.shape[-1] // 2
    return concentration[..., :half], concentration[..., half:]


def reference_rescale(draw, low, high, epsilon=1e-6):
    native = draw.astype(np.float32)
    ufunc_clip(native, epsilon, 1.0 - epsilon, out=native)
    return native, np.asarray(low + (high - low) * native, dtype=np.float32)


def bits(array):
    return np.ascontiguousarray(array).view(np.uint32)


def test_beta_op_codes_match_the_kernel_enum_and_stay_append_only():
    """The op stream is integers: a renumbered enum silently reinterprets graphs."""
    source = Path(host_graph.__file__).with_name("host_kernel.c")
    declared = dict(re.findall(r"CLEANRL_OP_([A-Z_0-9]+) = (\d+)", source.read_text()))
    mirrored = {name[len("_OP_"):]: value for name, value in vars(host_graph).items()
                if name.startswith("_OP_") and name != "_OP_STRIDE"}
    assert mirrored == {name: int(code) for name, code in declared.items()}
    # Append-only: codes are exactly 0..n-1 and the two Beta ops are the tail.
    assert sorted(mirrored.values()) == list(range(len(mirrored)))
    assert mirrored["BETA_CONC"] == 9 and mirrored["BETA_RESCALE"] == 10


@pytest.mark.parametrize("rows,act_dim", [(16, 6), (1, 1), (3, 17), (17, 3), (64, 24)])
def test_beta_concentration_op_splits_the_last_axis_bitwise(rows, act_dim):
    """alpha and beta come from disjoint, non-saturating value ranges, so an
    offset or a wrong row stride cannot coincide with the right answer."""
    rng = np.random.default_rng(rows * 100 + act_dim)
    logits = np.empty((rows, 2 * act_dim), dtype=np.float32)
    logits[:, :act_dim] = rng.uniform(-8.0, -2.0, size=(rows, act_dim))
    logits[:, act_dim:] = rng.uniform(2.0, 8.0, size=(rows, act_dim))
    graph = host_graph.BetaHeadGraph(rows, act_dim, np.float32(-1.0), np.float32(1.0))
    alpha, beta = graph.concentration(logits)
    expected_alpha, expected_beta = reference_concentration(logits)
    assert np.array_equal(bits(alpha), bits(expected_alpha))
    assert np.array_equal(bits(beta), bits(expected_beta))
    assert np.max(np.abs(alpha.astype(np.float64) - expected_alpha.astype(np.float64))) == 0.0
    assert np.max(np.abs(beta.astype(np.float64) - expected_beta.astype(np.float64))) == 0.0
    assert alpha.flags.c_contiguous and beta.flags.c_contiguous
    # Sensitivity: the halves differ everywhere (an offset bug reads beta's),
    # and the source row stride is 2 * act_dim, so walking it with the
    # destination's stride would read a different set of logits entirely.
    assert np.all(alpha < beta)
    if rows > 1:
        wrong_stride = logits.reshape(-1)[: rows * act_dim].reshape(rows, act_dim)
        assert not np.array_equal(wrong_stride, logits[:, :act_dim])


def test_beta_concentration_op_matches_logaddexp_over_swept_float32_inputs():
    """Denormals, signed zeros, the x == y branch, both exp branches, infinities
    and NaN payloads -- 1 ulp anywhere here moves the Beta draw."""
    windows = [
        np.arange(0, 1 << 18, dtype=np.uint32),                      # +0 .. denormals
        np.arange(1 << 31, (1 << 31) + (1 << 18), dtype=np.uint32),  # -0 .. -denormals
        np.arange(0x3F000000, 0x3F040000, dtype=np.uint32),          # around 0.5-1.0
        np.arange(0x42AE0000, 0x42B20000, dtype=np.uint32),          # around +87
        np.arange(0xC2AE0000, 0xC2B20000, dtype=np.uint32),          # around -87
        np.arange(0x7F7F0000, 0x7F810000, dtype=np.uint32),          # +max .. +inf, NaNs
        np.random.default_rng(7).integers(0, 1 << 32, size=1 << 20, dtype=np.uint64
                                          ).astype(np.uint32),
    ]
    for window in windows:
        logits = window.view(np.float32).reshape(1, -1)
        rows, act_dim = 1, logits.shape[1] // 2
        graph = host_graph.BetaHeadGraph(rows, act_dim, np.float32(0.0), np.float32(1.0))
        with np.errstate(invalid="ignore"):
            expected_alpha, expected_beta = reference_concentration(logits)
        alpha, beta = graph.concentration(logits)
        assert np.array_equal(bits(alpha), bits(expected_alpha))
        assert np.array_equal(bits(beta), bits(expected_beta))


@pytest.mark.parametrize("bounds", ["unit", "per_dimension", "scalar"])
def test_beta_rescale_op_matches_numpy_clip_and_rescale_bitwise(bounds):
    rows, act_dim = 16, 6
    rng = np.random.default_rng(11)
    if bounds == "unit":
        low, high = np.full(act_dim, -1.0, np.float32), np.full(act_dim, 1.0, np.float32)
    elif bounds == "per_dimension":
        # Per-dimension bounds that differ in every entry: collapsing them to
        # one global pair (a plausible fusion bug) changes every action.
        low = np.linspace(-9.0, -0.5, act_dim).astype(np.float32)
        high = np.linspace(0.25, 7.5, act_dim).astype(np.float32)
    else:
        low, high = np.float32(-2.5), np.float32(0.75)
    graph = host_graph.BetaHeadGraph(rows, act_dim, low, high)
    for _ in range(20):
        draw = rng.beta(rng.uniform(0.02, 4.0, size=(rows, act_dim)),
                        rng.uniform(0.02, 4.0, size=(rows, act_dim)))
        # Values on and outside both clip edges, including exact 0 and 1.
        draw.reshape(-1)[:8] = [0.0, 1.0, 1e-6, 1.0 - 1e-6, 5e-7, 0.9999999, 1e-300, 1.0 - 1e-16]
        native, action = graph.rescale(draw)
        expected_native, expected_action = reference_rescale(draw, low, high)
        assert np.array_equal(bits(native), bits(expected_native))
        assert np.array_equal(bits(action), bits(expected_action))
        assert np.max(np.abs(action.astype(np.float64) - expected_action.astype(np.float64))) == 0.0
    if bounds == "per_dimension":
        # A fusion that collapsed the per-dimension bounds to one pair would
        # produce different actions for the same draw, so this test sees it.
        collapsed = host_graph.BetaHeadGraph(rows, act_dim, low[0], high[0])
        assert not np.array_equal(collapsed.rescale(draw)[1], action)


def test_beta_head_graph_rejects_what_it_cannot_reproduce_bitwise():
    rows, act_dim = 4, 3
    with pytest.raises(ValueError, match="float32 action bounds"):
        host_graph.BetaHeadGraph(rows, act_dim, np.zeros(act_dim), np.ones(act_dim))
    with pytest.raises(ValueError, match="action bounds"):
        host_graph.BetaHeadGraph(rows, act_dim, np.zeros(act_dim + 1, np.float32),
                                 np.ones(act_dim + 1, np.float32))
    with pytest.raises(ValueError, match="positive"):
        host_graph.BetaHeadGraph(0, act_dim, np.float32(0.0), np.float32(1.0))
    graph = host_graph.BetaHeadGraph(rows, act_dim, np.float32(0.0), np.float32(1.0))
    with pytest.raises(ValueError, match="float32 logits"):
        graph.concentration(np.zeros((rows, 2 * act_dim), dtype=np.float64))
    with pytest.raises(ValueError, match="float32 logits"):
        graph.concentration(np.zeros((rows, act_dim), dtype=np.float32))
    with pytest.raises(ValueError, match="C-contiguous"):
        graph.concentration(np.asfortranarray(np.zeros((rows, 2 * act_dim), np.float32)))
    with pytest.raises(ValueError, match="float64 draws"):
        graph.rescale(np.zeros((rows, act_dim), dtype=np.float32))
    with pytest.raises(ValueError, match="C-contiguous"):
        graph.rescale(np.asfortranarray(np.zeros((rows, act_dim), np.float64)))


def test_beta_head_graph_reuses_its_output_buffers():
    """Allocation-free per step, which is why callers must copy what they keep."""
    rows, act_dim = 8, 5
    graph = host_graph.BetaHeadGraph(rows, act_dim, np.float32(-1.0), np.float32(1.0))
    rng = np.random.default_rng(3)
    alpha, beta = graph.concentration(rng.normal(size=(rows, 2 * act_dim)).astype(np.float32))
    native, action = graph.rescale(rng.beta(2.0, 3.0, size=(rows, act_dim)))
    snapshot = action.copy()
    again = graph.concentration(rng.normal(size=(rows, 2 * act_dim)).astype(np.float32))
    assert again[0] is alpha and again[1] is beta
    again = graph.rescale(rng.beta(2.0, 3.0, size=(rows, act_dim)))
    assert again[0] is native and again[1] is action
    assert not np.array_equal(action, snapshot)
