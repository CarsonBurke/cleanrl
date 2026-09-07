"""Sphere trunk diagnostics: DC-energy split, norm drift, and the radial pin."""

import numpy as np
import pytest
import torch

from cleanrl.shared.host_actor import make_lrelu_sphere_trunk, make_situ_sphere_trunk
from cleanrl.shared.ppo_loop import gather_metrics
from cleanrl.shared.sphere_diagnostics import TrunkProbe, WeightNormPin
from cleanrl.shared.runtime import configure_runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]

METRIC_KEYS = (
    "w_norm_ratio", "w_norm_ratio_max", "block_gate", "skip_gate",
    "preact_var", "branch_dc_frac", "out_dc_frac", "out_erank",
)


def _obs(rows=4096, in_dim=17, seed=0):
    x = np.random.default_rng(seed).standard_normal((rows, in_dim)).astype(np.float32)
    return torch.as_tensor(x, device="cuda")


def test_lrelusq_branch_output_is_dc_heavy_at_its_design_point():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(0)
    trunk = make_lrelu_sphere_trunk(17, 64).cuda()
    stats = TrunkProbe(trunk).metrics(_obs())
    assert stats["preact_var"].item() == pytest.approx(2.0, rel=0.15)
    assert stats["w_norm_ratio"].item() == pytest.approx(1.0, rel=1e-5)
    assert stats["w_norm_ratio_max"].item() == pytest.approx(1.0, rel=1e-5)
    assert stats["branch_dc_frac"].item() > 0.2
    assert stats["block_gate"].item() == pytest.approx(torch.sigmoid(torch.tensor(-1.5)).item(), rel=1e-5)


def test_situglu_branch_output_is_nearly_zero_mean():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(0)
    trunk = make_situ_sphere_trunk(17, 64).cuda()
    stats = TrunkProbe(trunk).metrics(_obs())
    assert stats["preact_var"].item() == pytest.approx(2.0, rel=0.15)
    assert stats["branch_dc_frac"].item() < 0.05


def test_pin_restores_norms_without_changing_the_scale_invariant_output():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    trunk = make_lrelu_sphere_trunk(17, 64).cuda()
    probe, pin = TrunkProbe(trunk), WeightNormPin(trunk)
    obs = _obs(256)
    with torch.no_grad():
        reference = trunk(obs).clone()

    with torch.no_grad():
        trunk.blocks[1].lin1.weight.mul_(1.5)
    drifted = probe.metrics(obs)
    assert drifted["w_norm_ratio_max"].item() == pytest.approx(1.5, rel=1e-4)
    assert drifted["w_norm_ratio"].item() > 1.0

    pin.apply()
    pinned = probe.metrics(obs)
    assert pinned["w_norm_ratio_max"].item() == pytest.approx(1.0, rel=1e-5)
    assert pinned["w_norm_ratio"].item() == pytest.approx(1.0, rel=1e-5)
    with torch.no_grad():
        cosine = torch.nn.functional.cosine_similarity(trunk(obs), reference, dim=-1)
    assert cosine.min().item() > 1.0 - 1e-4


def test_pin_is_idempotent_and_holds_the_activation_design_point():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(2)
    trunk = make_lrelu_sphere_trunk(17, 64).cuda()
    probe, pin = TrunkProbe(trunk), WeightNormPin(trunk)
    obs = _obs(256)
    design_var = probe.metrics(obs)["preact_var"].item()
    with torch.no_grad():
        for weight in (trunk.in_proj.weight, trunk.blocks[0].lin1.weight, trunk.blocks[2].lin2.weight):
            weight.mul_(0.4)
    assert probe.metrics(obs)["preact_var"].item() < 0.5 * design_var
    pin.apply()
    pin.apply()
    assert probe.metrics(obs)["preact_var"].item() == pytest.approx(design_var, rel=1e-4)


def test_effective_rank_collapses_on_a_degenerate_batch():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(3)
    trunk = make_lrelu_sphere_trunk(17, 64).cuda()
    probe = TrunkProbe(trunk)
    assert probe.metrics(_obs(1024))["out_erank"].item() > 4.0
    obs = _obs(1024)
    obs[1:] = obs[0]
    assert probe.metrics(obs)["out_erank"].item() < 2.0


def test_metrics_are_zero_dim_device_tensors_that_survive_gather_metrics():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(4)
    obs = _obs(512)
    for trunk in (make_lrelu_sphere_trunk(17, 64, n_blocks=1).cuda(),
                  make_situ_sphere_trunk(17, 64, n_blocks=3).cuda()):
        stats = TrunkProbe(trunk).metrics(obs)
        assert tuple(stats) == METRIC_KEYS
        for key, value in stats.items():
            assert value.shape == () and value.device == obs.device, key
        if trunk.n_blocks == 1:
            assert len(trunk.skip_gates) == 0
            assert stats["skip_gate"].item() == 0.0
        else:
            assert stats["skip_gate"].item() > 0.0
        host = gather_metrics(stats)
        assert sorted(host) == sorted(METRIC_KEYS)
        assert all(np.isfinite(v) for v in host.values())
