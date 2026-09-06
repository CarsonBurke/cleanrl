"""Fused native host graph: FP32 parity with CUDA and with the NumPy mirrors."""

import warnings

import numpy as np
import pytest
import torch
from torch import nn

from cleanrl.shared.host_actor import (
    HostLReluResActor, HostLReluSphereActor, HostMLP, HostSiTUDenseActor,
    HostSiTUResActor, HostSiTUSphereActor, LeakyReluSq, SiTUGLUBranch,
    make_lrelu_res_trunk, make_lrelu_sphere_trunk, make_situ_dense_trunk,
    make_situ_res_trunk, make_situ_sphere_trunk,
)
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
    trunk.blocks[1].act = nn.ReLU()  # pair must be Linear -> LeakyReluSq -> Linear
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
