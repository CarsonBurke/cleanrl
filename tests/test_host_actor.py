"""Host actor mirror: FP32 parity, refresh after updates, and the NumPy Beta head."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Beta

from cleanrl.shared.host_actor import (
    SQ_PAIR_CAP, CappedLReluSqPair, CappedLeakyReluSq, CappedSignSqPair,
    CappedSignedSquare, HostLReluResActor, HostLReluSphereActor, HostMLP,
    HostSiTUDenseActor, HostSiTUResActor, HostSiTUSphereActor, LReluResTrunk,
    LReluSphereTrunk, LeakyReluSq, SiTUGLUBranch, SiTUDenseTrunk, SiTUResTrunk,
    SiTUSphereTrunk, SignSqPair, SignedSquare, justnorm,
    make_caplrelusq_sphere_trunk, make_capsignsq_sphere_trunk,
    make_lrelu_res_trunk, make_lrelu_sphere_trunk, make_signsq_sphere_trunk,
    make_situ_dense_trunk, make_situ_res_trunk, make_situ_sphere_trunk,
)
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import sample_beta_actions_host

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def test_host_mlp_matches_device_forward_and_tracks_in_place_updates():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(17, 64), nn.Tanh(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 12)).cuda()
    mirror = HostMLP(net, 16)
    x = np.random.default_rng(0).standard_normal((16, 17)).astype(np.float32)
    with torch.no_grad():
        expected = net(torch.as_tensor(x, device="cuda")).cpu().numpy()
    np.testing.assert_allclose(mirror(x), expected, rtol=1e-5, atol=1e-6)
    stale = mirror(x).copy()
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.mul_(-1.5)
        expected = net(torch.as_tensor(x, device="cuda")).cpu().numpy()
    np.testing.assert_array_equal(mirror(x), stale)  # not refreshed yet
    mirror.refresh()
    np.testing.assert_allclose(mirror(x), expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        mirror(x[:8])


def test_host_mlp_matches_leakyrelusq_device_forward():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    net = nn.Sequential(
        nn.Linear(17, 64), LeakyReluSq(), nn.Linear(64, 12)
    ).cuda()
    mirror = HostMLP(net, 16)
    x = np.random.default_rng(1).standard_normal((16, 17)).astype(np.float32)
    with torch.no_grad():
        expected = net(torch.as_tensor(x, device="cuda")).cpu().numpy()
    np.testing.assert_allclose(mirror(x), expected, rtol=1e-5, atol=1e-6)


def test_host_mlp_matches_situglu_device_forward():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(2)
    net = nn.Sequential(
        SiTUGLUBranch(17, 64), SiTUGLUBranch(64, 64), nn.Linear(64, 12)
    ).cuda()
    mirror = HostMLP(net, 16)
    x = np.random.default_rng(2).standard_normal((16, 17)).astype(np.float32)
    with torch.no_grad():
        expected = net(torch.as_tensor(x, device="cuda")).cpu().numpy()
    np.testing.assert_allclose(mirror(x), expected, rtol=1e-5, atol=1e-6)


def test_host_situres_actor_matches_device_forward_and_tracks_gates():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(3)
    trunk = make_situ_res_trunk(17, 64)
    assert isinstance(trunk, SiTUResTrunk)
    head = nn.Linear(64, 12)
    seq = nn.Sequential(trunk, head).cuda()
    mirror = HostSiTUResActor(seq, 16)
    x = np.random.default_rng(3).standard_normal((16, 17)).astype(np.float32)
    with torch.no_grad():
        expected = seq(torch.as_tensor(x, device="cuda")).cpu().numpy()
    np.testing.assert_allclose(mirror(x), expected, rtol=1e-5, atol=1e-6)
    with torch.no_grad():
        trunk.lam1.fill_(2.0)
        for parameter in seq.parameters():
            parameter.mul_(-1.5)
        expected = seq(torch.as_tensor(x, device="cuda")).cpu().numpy()
    mirror.refresh()
    # Perturbed weights inflate activations, so FP32 accumulation-order noise
    # grows too; the looser leg still proves refresh tracking, not parity.
    np.testing.assert_allclose(mirror(x), expected, rtol=1e-4, atol=1e-5)
    with pytest.raises(TypeError):
        HostSiTUResActor(nn.Sequential(nn.Linear(17, 64).cuda(), nn.Linear(64, 12).cuda()), 2)
    with pytest.raises(ValueError):
        mirror(x[:8])


def test_host_situdense_actor_matches_device_forward_and_tracks_gates():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(4)
    trunk = make_situ_dense_trunk(17, 64, 3)
    assert isinstance(trunk, SiTUDenseTrunk)
    seq = nn.Sequential(trunk, nn.Linear(64, 12)).cuda()
    mirror = HostSiTUDenseActor(seq, 16)
    x = np.random.default_rng(4).standard_normal((16, 17)).astype(np.float32)
    with torch.no_grad():
        expected = seq(torch.as_tensor(x, device="cuda")).cpu().numpy()
    np.testing.assert_allclose(mirror(x), expected, rtol=1e-5, atol=1e-6)
    with torch.no_grad():
        trunk.skip_gates[0].fill_(1.0)
        for parameter in seq.parameters():
            parameter.mul_(-1.5)
        expected = seq(torch.as_tensor(x, device="cuda")).cpu().numpy()
    mirror.refresh()
    # Perturbed weights inflate activations ~100x, so FP32 accumulation-order
    # noise grows proportionally; this leg proves refresh wiring, not parity.
    np.testing.assert_allclose(mirror(x), expected, rtol=1e-3, atol=1e-4)
    with pytest.raises(TypeError):
        HostSiTUDenseActor(nn.Sequential(nn.Linear(17, 64).cuda(), nn.Linear(64, 12).cuda()), 2)
    with pytest.raises(ValueError):
        mirror(x[:8])


def test_host_situsphere_actor_matches_device_forward_and_tracks_gates():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(5)
    trunk = make_situ_sphere_trunk(17, 64, 3)
    assert isinstance(trunk, SiTUSphereTrunk)
    seq = nn.Sequential(trunk, nn.Linear(64, 12)).cuda()
    mirror = HostSiTUSphereActor(seq, 16)
    x = np.random.default_rng(5).standard_normal((16, 17)).astype(np.float32)
    with torch.no_grad():
        expected = seq(torch.as_tensor(x, device="cuda")).cpu().numpy()
    np.testing.assert_allclose(mirror(x), expected, rtol=1e-5, atol=1e-6)
    with torch.no_grad():
        trunk.block_gates[1].fill_(1.0)
        for parameter in seq.parameters():
            parameter.mul_(-1.5)
        expected = seq(torch.as_tensor(x, device="cuda")).cpu().numpy()
    mirror.refresh()
    # Perturbed weights inflate activations; looser leg proves tracking.
    np.testing.assert_allclose(mirror(x), expected, rtol=1e-3, atol=1e-4)
    with pytest.raises(TypeError):
        HostSiTUSphereActor(nn.Sequential(nn.Linear(17, 64).cuda(), nn.Linear(64, 12).cuda()), 2)
    with pytest.raises(ValueError):
        mirror(x[:8])


def test_host_lrelures_actor_matches_device_forward_and_tracks_gates():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(6)
    trunk = make_lrelu_res_trunk(17, 64)
    assert isinstance(trunk, LReluResTrunk)
    seq = nn.Sequential(trunk, nn.Linear(64, 12)).cuda()
    mirror = HostLReluResActor(seq, 16)
    x = np.random.default_rng(6).standard_normal((16, 17)).astype(np.float32)
    with torch.no_grad():
        expected = seq(torch.as_tensor(x, device="cuda")).cpu().numpy()
    np.testing.assert_allclose(mirror(x), expected, rtol=1e-5, atol=1e-6)
    with torch.no_grad():
        trunk.lam2.fill_(1.0)
        for parameter in seq.parameters():
            parameter.mul_(-1.5)
        expected = seq(torch.as_tensor(x, device="cuda")).cpu().numpy()
    mirror.refresh()
    # Perturbed weights inflate activations; looser leg proves tracking.
    np.testing.assert_allclose(mirror(x), expected, rtol=1e-3, atol=1e-4)
    with pytest.raises(TypeError):
        HostLReluResActor(nn.Sequential(nn.Linear(17, 64).cuda(), nn.Linear(64, 12).cuda()), 2)
    with pytest.raises(ValueError):
        mirror(x[:8])


def test_host_lrelusphere_actor_matches_device_forward_and_tracks_gates():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(7)
    trunk = make_lrelu_sphere_trunk(17, 64, 3)
    assert isinstance(trunk, LReluSphereTrunk)
    seq = nn.Sequential(trunk, nn.Linear(64, 12)).cuda()
    mirror = HostLReluSphereActor(seq, 16)
    x = np.random.default_rng(7).standard_normal((16, 17)).astype(np.float32)
    with torch.no_grad():
        expected = seq(torch.as_tensor(x, device="cuda")).cpu().numpy()
    np.testing.assert_allclose(mirror(x), expected, rtol=1e-5, atol=1e-6)
    with torch.no_grad():
        trunk.block_gates[1].fill_(1.0)
        for parameter in seq.parameters():
            parameter.mul_(-1.5)
        expected = seq(torch.as_tensor(x, device="cuda")).cpu().numpy()
    mirror.refresh()
    # Perturbed weights inflate activations; looser leg proves tracking.
    np.testing.assert_allclose(mirror(x), expected, rtol=1e-3, atol=1e-4)
    with pytest.raises(TypeError):
        HostLReluSphereActor(nn.Sequential(nn.Linear(17, 64).cuda(), nn.Linear(64, 12).cuda()), 2)
    with pytest.raises(ValueError):
        mirror(x[:8])


def test_host_signsqsphere_actor_matches_device_forward_and_tracks_gates():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(21)
    trunk = make_signsq_sphere_trunk(17, 64, 3)
    assert isinstance(trunk, LReluSphereTrunk)
    assert all(isinstance(block, SignSqPair) for block in trunk.blocks)
    assert all(isinstance(block.act, SignedSquare) for block in trunk.blocks)
    seq = nn.Sequential(trunk, nn.Linear(64, 12)).cuda()
    mirror = HostLReluSphereActor(seq, 16)
    x = np.random.default_rng(21).standard_normal((16, 17)).astype(np.float32)
    with torch.no_grad():
        expected = seq(torch.as_tensor(x, device="cuda")).cpu().numpy()
    np.testing.assert_allclose(mirror(x), expected, rtol=1e-5, atol=1e-6)
    with torch.no_grad():
        trunk.block_gates[1].fill_(1.0)
        for parameter in seq.parameters():
            parameter.mul_(-1.5)
        expected = seq(torch.as_tensor(x, device="cuda")).cpu().numpy()
    mirror.refresh()
    # Perturbed weights inflate activations; looser leg proves tracking.
    np.testing.assert_allclose(mirror(x), expected, rtol=1e-3, atol=1e-4)


def test_host_lrelusphere_actor_rejects_unmirrored_pair_activation():
    """The dispatch must fail at construction, not mirror the wrong f() silently."""
    torch.manual_seed(22)
    trunk = make_signsq_sphere_trunk(17, 64, 2)
    trunk.blocks[1].act = nn.ReLU()
    with pytest.raises(TypeError):
        HostLReluSphereActor(nn.Sequential(trunk, nn.Linear(64, 12)).cuda(), 16)
    trunk = make_lrelu_sphere_trunk(17, 64, 2)
    trunk.blocks[0].act = nn.Tanh()
    with pytest.raises(TypeError):
        HostLReluSphereActor(nn.Sequential(trunk, nn.Linear(64, 12)).cuda(), 16)


def test_make_signsq_sphere_trunk_sits_at_its_design_point():
    """lin1 puts the preactivation at variance 2, where E[f^2] = 3v^2 = 12, so
    the sqrt(pair_out_var/12) lin2 gain lands the block output on pair_out_var."""
    torch.manual_seed(23)
    trunk = make_signsq_sphere_trunk(17, 64, 3, pair_out_var=0.5)
    u = justnorm(torch.randn(4096, 64, generator=torch.Generator().manual_seed(23)))
    with torch.no_grad():
        preact = trunk.blocks[0].lin1(u)
        out = trunk.blocks[0](u)
    assert preact.var().item() == pytest.approx(2.0, rel=1e-3)
    # 3% under 0.5: unit-vector coordinates are marginally lighter-tailed than
    # the Gaussian the E[x^4] = 3v^2 design point assumes (3/(d(d+2)) vs 3/d^2).
    assert out.pow(2).mean().item() == pytest.approx(0.5, rel=0.06)


def test_signed_square_removes_the_dc_direction_justnorm_cannot():
    """The measured mechanism: LeakyReluSq is non-negative, so a fixed share of
    its justnormed branch output is an input-independent constant direction."""
    torch.manual_seed(24)
    signsq, lrelusq = make_signsq_sphere_trunk(17, 64, 3), make_lrelu_sphere_trunk(17, 64, 3)
    u = justnorm(torch.randn(4096, 64, generator=torch.Generator().manual_seed(24)))

    def dc_fraction(block):
        with torch.no_grad():
            y = justnorm(block(u))
        return (y.mean(0).pow(2).sum() / y.pow(2).sum(1).mean()).item()

    assert dc_fraction(signsq.blocks[0]) < 0.02
    assert dc_fraction(lrelusq.blocks[0]) > 0.2


def test_host_capped_sphere_actors_match_device_forward_and_track_gates():
    """The tanh cap is inside the mirrored activation, so a mismatch in the
    cap stage (wrong constant, cap applied after the square) shows up here."""
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    legs = ((make_capsignsq_sphere_trunk, CappedSignSqPair, CappedSignedSquare),
            (make_caplrelusq_sphere_trunk, CappedLReluSqPair, CappedLeakyReluSq))
    for seed, (factory, pair_cls, act_cls) in enumerate(legs, start=31):
        torch.manual_seed(seed)
        trunk = factory(17, 64, 3)
        assert isinstance(trunk, LReluSphereTrunk)
        assert all(isinstance(block, pair_cls) for block in trunk.blocks)
        assert all(isinstance(block.act, act_cls) for block in trunk.blocks)
        seq = nn.Sequential(trunk, nn.Linear(64, 12)).cuda()
        mirror = HostLReluSphereActor(seq, 16)
        x = np.random.default_rng(seed).standard_normal((16, 17)).astype(np.float32)
        with torch.no_grad():
            expected = seq(torch.as_tensor(x, device="cuda")).cpu().numpy()
        np.testing.assert_allclose(mirror(x), expected, rtol=1e-5, atol=1e-6)
        with torch.no_grad():
            trunk.block_gates[1].fill_(1.0)
            for parameter in seq.parameters():
                parameter.mul_(-1.5)
            expected = seq(torch.as_tensor(x, device="cuda")).cpu().numpy()
        mirror.refresh()
        # Perturbed weights push the activation into saturation; the looser
        # leg proves tracking.
        np.testing.assert_allclose(mirror(x), expected, rtol=1e-3, atol=1e-4)


def test_host_lrelusphere_actor_rejects_a_nondefault_activation_cap():
    """Both mirrors hardcode SQ_PAIR_CAP (the fused op has 4.0f baked in), so
    a pair carrying any other cap must be refused, not silently rescaled."""
    torch.manual_seed(32)
    trunk = make_capsignsq_sphere_trunk(17, 64, 2)
    trunk.blocks[1].act = CappedSignedSquare(cap=8.0)
    with pytest.raises(ValueError):
        HostLReluSphereActor(nn.Sequential(trunk, nn.Linear(64, 12)).cuda(), 16)
    trunk = make_caplrelusq_sphere_trunk(17, 64, 2)
    trunk.blocks[0].act = CappedLeakyReluSq(cap=2.0)
    with pytest.raises(ValueError):
        HostLReluSphereActor(nn.Sequential(trunk, nn.Linear(64, 12)).cuda(), 16)


def test_capped_sphere_trunks_sit_at_their_design_points():
    """lin1 still targets preactivation variance 2, but the cap removes ~47% of
    each activation's second moment (E[f^2] = 6.35657 vs the uncapped 12 for
    t|t|, 3.38142 vs 6.375 for the leaky-relu square), so only the capped
    constants land the block output on pair_out_var."""
    for seed, factory in enumerate((make_capsignsq_sphere_trunk,
                                    make_caplrelusq_sphere_trunk), start=33):
        torch.manual_seed(seed)
        trunk = factory(17, 64, 3, pair_out_var=0.5)
        u = justnorm(torch.randn(4096, 64, generator=torch.Generator().manual_seed(seed)))
        with torch.no_grad():
            preact = trunk.blocks[0].lin1(u)
            out = trunk.blocks[0](u)
        assert preact.var().item() == pytest.approx(2.0, rel=1e-3)
        # Same ~6% band as the signsq leg: unit-vector coordinates are
        # marginally lighter-tailed than the Gaussian the design point assumes.
        assert out.pow(2).mean().item() == pytest.approx(0.5, rel=0.06)


def test_capped_activations_are_bounded_and_keep_the_dc_signature_of_their_shape():
    """The property under test: saturation supplies the restoring force the
    uncapped degree-2 homogeneous pairs lack, and capping does not by itself
    remove the non-negative shape's DC direction."""
    x = torch.linspace(-200.0, 200.0, 200_001, dtype=torch.float64)
    assert CappedSignedSquare()(x).abs().max().item() <= SQ_PAIR_CAP ** 2
    assert CappedLeakyReluSq()(x).abs().max().item() <= SQ_PAIR_CAP ** 2
    assert SignedSquare()(x).abs().max().item() > 1e4  # the uncapped contrast
    torch.manual_seed(34)
    capsignsq, caplrelusq = make_capsignsq_sphere_trunk(17, 64, 3), make_caplrelusq_sphere_trunk(17, 64, 3)
    u = justnorm(torch.randn(4096, 64, generator=torch.Generator().manual_seed(34)))

    def dc_fraction(block):
        with torch.no_grad():
            y = justnorm(block(u))
        return (y.mean(0).pow(2).sum() / y.pow(2).sum(1).mean()).item()

    assert dc_fraction(capsignsq.blocks[0]) < 0.02
    assert dc_fraction(caplrelusq.blocks[0]) > 0.2



def test_host_mlp_rejects_unsupported_layers():
    with pytest.raises(TypeError):
        HostMLP(nn.Sequential(nn.Linear(3, 3), nn.Sigmoid()).cuda(), 2)
    with pytest.raises(TypeError):
        HostMLP(nn.Linear(3, 3).cuda(), 2)


def test_host_beta_head_matches_device_parameterization_and_is_seeded():
    logits = np.random.default_rng(3).standard_normal((4096, 6)).astype(np.float32) * 3
    low, high = np.array([-3.0, -0.25, 1.0], np.float32), np.array([-1.0, 4.0, 9.0], np.float32)
    native, action = sample_beta_actions_host(logits, low, high, np.random.default_rng(1))
    assert native.dtype == action.dtype == np.float32
    assert native.min() >= 1e-6 and native.max() <= 1 - 1e-6
    np.testing.assert_allclose(action, low + (high - low) * native, rtol=0, atol=0)
    concentration = torch.nn.functional.softplus(torch.as_tensor(logits)) + 1
    alpha, beta = concentration.chunk(2, dim=-1)
    distribution = Beta(alpha, beta)
    z = (native.mean(0) - distribution.mean.mean(0).numpy()) / (distribution.variance.mean(0).sqrt().numpy() / np.sqrt(4096))
    assert np.abs(z).max() < 5
    again, _ = sample_beta_actions_host(logits, low, high, np.random.default_rng(1))
    np.testing.assert_array_equal(native, again)
