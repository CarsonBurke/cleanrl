"""The original cubature experiment with raw gate units and passive clipping telemetry."""
import ast
from dataclasses import asdict
import inspect
import textwrap

import torch

from cleanrl.td7.delight import td7_latentq_delight_cubature_v1 as original
from cleanrl.td7.delight import td7_latentq_delight_cubature_raw_v9 as trainer
from cleanrl.td7.delight import td7_latentq_delight_hermite_integrated_v8 as telemetry_reference


def syntax(value):
    return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(value))))


def test_original_scientific_methods_are_unchanged():
    for name in ('Actor', 'Encoder', 'EnsembleLinear', 'PreActionLatentTwinCritic',
                 'HostPolicy', 'gaussian_log_probability', 'delight_score_loss'):
        assert syntax(getattr(original, name)) == syntax(getattr(trainer, name))
    for name in ('_encoder_loss_fn', '_critic_loss_fn', '_actor_loss_fn',
                 '_entropy_fn', 'hard_update', 'learn', 'stage', 'flush'):
        assert syntax(getattr(original.TD7Agent, name)) == syntax(getattr(trainer.TD7Agent, name))


def test_only_default_gate_scale_changes():
    before, after = asdict(original.Args()), asdict(trainer.Args())
    assert after.pop('gate_scale') == 'raw'
    assert before.pop('gate_scale') == 'rms'
    before.pop('exp_name'); after.pop('exp_name')
    assert before == after
    assert trainer.Args().target_smoothing_sigma == 0.2
    assert trainer.Args().target_sigma == 0.1


def test_statistics_match_reviewed_instrumentation_and_remain_detached():
    assert syntax(trainer.accumulate_clipping_statistics) == syntax(telemetry_reference.accumulate_clipping_statistics)
    stats = torch.zeros(5, dtype=torch.float64)
    norms = torch.tensor([0., .25, 1., 2.], dtype=torch.float64, requires_grad=True)
    for norm in norms:
        trainer.accumulate_clipping_statistics(stats, norm, .5)
    torch.testing.assert_close(stats[:3], torch.tensor([3.25, 4., 2.], dtype=torch.float64))
    torch.testing.assert_close(stats[3], (.5 / (norms.detach() + 1e-6)).clamp(max=1).sum())
    assert stats[4].item() == 2
    assert not stats.requires_grad
