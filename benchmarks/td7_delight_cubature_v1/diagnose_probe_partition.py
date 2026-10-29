"""Queued CUDA check separating score reconstruction from VJP arithmetic precision."""
import json
from pathlib import Path
import runpy
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cleanrl.td7.delight import td7_latentq_delight_reference_v10 as trainer

fixtures = runpy.run_path(str(ROOT / 'tests/test_td7_latentq_delight_reference_v10.py'))
results = []
for precision in ('production', 'float32_precise', 'float64'):
    for baseline in ('modal', 'latent'):
        actor, critic, encoder, state, sizes = fixtures['networks'](10)
        if precision != 'production':
            trainer.configure_runtime(matmul_precision='highest', allow_tf32=False)
        if precision == 'float64':
            actor, critic, encoder = [module.to(dtype=torch.float64) for module in (actor, critic, encoder)]
            state = state.double()
        zs = encoder.zs(state).detach()
        eps = fixtures['antithetic'](4, state).to(state.dtype)
        agent = fixtures['stub'](trainer.Args(actor_baseline=baseline, **sizes), actor, critic, encoder, state)
        mu, log_std = actor(state, zs)
        with torch.no_grad():
            u, advantage = agent._advantage(state, zs, mu, log_std.exp(), eps, baseline)
            effective_eps = (u - mu) * (-log_std).exp()
        loss = trainer.delight_score_loss(u, mu, log_std, advantage)[0]
        full = fixtures['flat_gradient'](loss, actor, retain_graph=True)
        location = fixtures['flat_gradient'](
            trainer.delight_score_loss(u, mu, log_std.detach(), advantage)[0], actor, retain_graph=True)
        for source, directions in (('supplied', eps), ('reconstructed', effective_eps)):
            with torch.no_grad():
                weight = trainer.delight_weight(advantage, directions)[0]
                coefficient = -(weight * advantage)[..., None] * (directions.square() - 1) / advantage.numel()
            parts = [fixtures['flat_gradient'](log_std, actor,
                        grad_outputs=(coefficient * mask).sum(0), retain_graph=True)
                     for mask in (directions.square() < 1, directions.square() >= 1)]
            error = parts[0] + parts[1] + location - full
            results.append({'precision': precision, 'reference': baseline, 'eps': source,
                            'max_absolute_error': error.abs().max().item(),
                            'relative_norm_error': (error.norm() / full.norm()).item(),
                            'full_gradient_norm': full.norm().item()})
print(json.dumps(results, indent=2))
(ROOT / 'benchmarks/td7_delight_cubature_v1/reference_v10_partition_diagnosis.json').write_text(json.dumps(results, indent=2)+'\n')
assert max(r['relative_norm_error'] for r in results if r['precision'] == 'float64' and r['eps'] == 'reconstructed') < 1e-10
