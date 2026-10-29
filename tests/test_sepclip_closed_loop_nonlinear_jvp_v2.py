"""V2 independent finite-difference contract for saved nonlinear actor chords.

Run through mlq on CUDA. Uses full saved Adam/Anvil update traces, never a
synthetic bias-only direction. Differences follow the ambient parameter chord
used by the audit; retraction would define a different derivative.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "nonlinear_closed_loop_v2", ROOT / "scripts/audit_sepclip_closed_loop_update_v2.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

SNAPSHOTS = (
    "HalfCheetah-v4__sepclip_microscopy_v12_adam_control__1__1791260590",
    "HalfCheetah-v4__sepclip_microscopy_v12_anvil_control__1__1791260697",
)
# Artifact bytes reconciled on supervisor restart; checked before loading.
ARTIFACT_SHA256 = {'runs/HalfCheetah-v4__sepclip_microscopy_v12_adam_control__1__1791260590/diagnostic_snapshot_4013696.pt': '8b17da17c1a4d6ed31e62076f76debfc01763e6f0a8b95f401674e55415d6197',
 'runs/HalfCheetah-v4__sepclip_microscopy_v12_adam_control__1__1791260590/diagnostic_snapshot_7995008.pt': '5d7979136fb15329c9bd6d0e6340cde6206d8b0956c21f273f223bbb851b3f30',
 'runs/HalfCheetah-v4__sepclip_microscopy_v12_adam_control__1__1791260590/diagnostic_updates_4013696.pt': '762e1629a4d2b547fa3d5442c081c2ff99ea62c05c3f97a46477304b14f9fa6b',
 'runs/HalfCheetah-v4__sepclip_microscopy_v12_adam_control__1__1791260590/diagnostic_updates_7995008.pt': '7cdb678b6a24b260f543cca4771c234e6338bdf2de8d6fd4c428d7a18cd38926',
 'runs/HalfCheetah-v4__sepclip_microscopy_v12_anvil_control__1__1791260697/diagnostic_snapshot_4013696.pt': '18d6879248fb472d9d84c4254b5052918f2432bf731c398f0f87ad8138ae56cb',
 'runs/HalfCheetah-v4__sepclip_microscopy_v12_anvil_control__1__1791260697/diagnostic_snapshot_7995008.pt': '06819047d869a87baa62ea4d4f4e4a18044f0c52c71fa3beea2b9a66cab7c587',
 'runs/HalfCheetah-v4__sepclip_microscopy_v12_anvil_control__1__1791260697/diagnostic_updates_4013696.pt': '25709629234dc4f8bd43f9c6597247d25557be071702d8bfdb66895c2c2ffbde',
 'runs/HalfCheetah-v4__sepclip_microscopy_v12_anvil_control__1__1791260697/diagnostic_updates_7995008.pt': '1bb42d0cc7faa006422f055f5f43264a0782356d8703b6b340618b5a455efe1b'}

STEPS = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0)


@pytest.mark.parametrize("run", SNAPSHOTS)
@pytest.mark.parametrize("checkpoint_step", (4013696, 7995008))
def test_actual_nonlinear_chords_agree_with_centered_parameter_differences(run, checkpoint_step):
    audit.configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch._dynamo.config.recompile_limit = 64
    assert torch.cuda.is_available(), "Submit this contract through mlq on CUDA"
    path = ROOT / "runs" / run / f"diagnostic_snapshot_{checkpoint_step}.pt"
    trace_path = path.with_name(f"diagnostic_updates_{checkpoint_step}.pt")
    provenance = dict(snapshot_sha256=audit.hash_file(path),
                      update_trace_sha256=audit.hash_file(trace_path),
                      source_sha256={str(p.relative_to(ROOT)): audit.hash_file(p)
                          for p in (Path(__file__),
                                    ROOT / "scripts/audit_sepclip_closed_loop_update_v2.py",
                                    ROOT / "scripts/audit_sepclip_dense_pg.py",
                                    ROOT / "scripts/audit_sepclip_physics.py",
                                    Path(audit.trainer.__file__))})
    assert provenance["snapshot_sha256"] == ARTIFACT_SHA256[str(path.relative_to(ROOT))]
    assert provenance["update_trace_sha256"] == ARTIFACT_SHA256[str(trace_path.relative_to(ROOT))]
    # Missing evidence is a failure, never a skipped numerical contract.
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    trace = torch.load(trace_path,
                       map_location="cpu", weights_only=False)
    envs = audit.make_mujoco_vector_env("HalfCheetah-v4", 16, num_threads=2)
    try:
        agent = audit.trainer.Agent(envs).cuda().eval()
        agent.load_state_dict(snapshot["model"], strict=True)
    finally:
        envs.close()
    banks, chords, reconstruction = audit.reconstruct_banks(agent, snapshot, trace)
    assert reconstruction["cumulative_delta_vs_exact_endpoint_max_abs_error"] <= 2e-5
    # Tie both returned policies and chords directly to stored evidence, apart
    # from the helper used by the derivative and finite-difference paths.
    assert set(banks) == {name.removeprefix("actor.") for name in snapshot["model"]
                          if name.startswith("actor.")}
    assert len(chords) == 2
    for name, bank in banks.items():
        assert bank.shape[0] == 3
        key = "actor." + name
        saved_old = snapshot["model"][key].cuda()
        saved_first = saved_old + trace["trace"][0]["retracted_delta"][key].cuda()
        saved_endpoint = trace["model_after_update"][key].cuda()
        for actual, saved in zip(bank, (saved_old, saved_first, saved_endpoint)):
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)
        torch.testing.assert_close(chords[0][name], saved_first - saved_old, rtol=0, atol=0)
        torch.testing.assert_close(chords[1][name], saved_endpoint - saved_old, rtol=0, atol=0)
    states = audit.dense.select_states(snapshot, count=32, max_age=400)
    x = torch.as_tensor(np.asarray([s["normalized_obs"] for s in states]),
                        dtype=torch.float32, device="cuda")
    old = {name: bank[0] for name, bank in banks.items()}

    # Independent primal evaluations: no JVP, backward, or analytical Jacobian.
    def forward(parameters, observations):
        return torch.func.functional_call(agent.actor, parameters, (observations,))

    finite_forward = torch.compile(forward, fullgraph=True, dynamic=False,
                                   options={"triton.cudagraphs": False})
    directional = audit.compiled_directional(agent, banks, chords)
    with torch.no_grad():
        primal, first, endpoint, _ = directional(x)
        independent_primal = finite_forward(old, x).clone()
    torch.testing.assert_close(primal, independent_primal, atol=3e-5, rtol=3e-6)

    for label, chord, expected in zip(("first", "endpoint"), chords, (first, endpoint)):
        # Require genuine hidden nonlinear and readout motion in the saved run.
        for prefix in ("first.0.gate", "first.0.up", "first.0.down",
                       "second.0.gate", "second.0.up", "second.0.down",
                       "head.0", "readout_gain"):
            assert any(bool(d.abs().max() > 0) for name, d in chord.items()
                       if name.startswith(prefix)), (run, label, prefix)
        derivative = expected.double()
        signal = float(derivative.norm())
        assert np.isfinite(signal) and signal > 0
        chord_norm = torch.stack([d.double().square().sum() for d in chord.values()]).sum().sqrt()
        rows = []
        for h in STEPS:
            plus = {name: old[name] + h * chord[name] for name in old}
            minus = {name: old[name] - h * chord[name] for name in old}
            with torch.no_grad():
                zp = finite_forward(plus, x).clone()
                zm = finite_forward(minus, x).clone()
            observed = (zp.double() - zm.double()) / (2 * h)
            # Measure FP32 parameter quantization rather than assuming h*d
            # survived addition. This also detects accidentally dropped banks.
            rounded_error = torch.stack([
                ((plus[name].double() - minus[name].double()) / (2 * h)
                 - chord[name].double()).square().sum() for name in old
            ]).sum().sqrt() / chord_norm
            relative_error = float((observed - derivative).norm() / signal)
            cosine = float((observed * derivative).sum() / (observed.norm() * signal))
            # Heuristic output-rounding diagnostic, not a mathematical error
            # bound or an allowance for noisy differences to pass accuracy.
            rounding_floor = float((16 * torch.finfo(torch.float32).eps
                * torch.maximum(zp.abs(), zm.abs()).double() / h).norm() / signal)
            rows.append(dict(h=h, relative_error=relative_error, cosine=cosine,
                             rounded_chord_relative_error=float(rounded_error),
                             estimated_output_rounding_relative_floor=rounding_floor))
        print(json.dumps(dict(run=run, checkpoint_step=checkpoint_step,
                              chord=label, directional_norm=signal,
                              provenance=provenance,
                              rounding_floor_scope="Conservative heuristic; not a mathematical error bound",
                              finite_difference_sweep=rows)), flush=True)
        # Two adjacent useful sizes must agree: a lucky isolated crossing of
        # curvature and roundoff is insufficient. Large-h curvature and tiny-h
        # quantization are retained in output, rather than hidden or clamped.
        good = [r["relative_error"] < .03 and r["cosine"] > .999
                and r["rounded_chord_relative_error"] < .02
                and r["estimated_output_rounding_relative_floor"] < .03
                for r in rows]
        assert any(a and b for a, b in zip(good, good[1:])), (run, label, rows)
    assert audit.hash_file(path) == provenance["snapshot_sha256"]
    assert audit.hash_file(trace_path) == provenance["update_trace_sha256"]
