"""Analytical gradient attribution contract; CUDA tests must run through mlq."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('snapshot_gradients_test_module', SCRIPTS / 'reppo_snapshot_gradients.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required; queue through mlq')
def test_group_norms_include_offset_and_weighted_cancellation():
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(1, 1, bias=False, device='cuda')
    model.zero_dist = torch.nn.Parameter(torch.tensor(2., device='cuda'))
    offset_gradient = 4 * 40.9  # same learned-logit offset multiplier as REPPO
    losses = {'ce': 3 * model.encoder.weight.sum() + offset_gradient * model.zero_dist,
              'latent': -model.encoder.weight.sum()}
    result = module.group_gradients(model, losses, {'ce': 1., 'latent': 2.})
    assert result['group/zero_dist/ce_raw_norm'] == pytest.approx(offset_gradient)
    assert result['group/zero_dist/latent_raw_norm'] == 0.
    assert result['group/encoder/ce_latent_cosine'] == -1.
    assert result['group/encoder/total_norm'] == 1.
    assert result['group/zero_dist/total_norm'] == pytest.approx(offset_gradient)
    assert result['total/grad_norm'] == pytest.approx((1 + offset_gradient ** 2) ** .5)
    assert result['group/zero_dist/squared_norm_share'] == pytest.approx(offset_gradient ** 2 / (1 + offset_gradient ** 2))
    assert result['total/ce_latent_cosine'] == pytest.approx(-3 / (9 + offset_gradient ** 2) ** .5)
    assert all(parameter.grad is None for parameter in model.parameters())
