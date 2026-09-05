"""Queue CUDA graph correctness through mlq; these are not training runs."""

import copy
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from cleanrl.shared.cuda_update import CudaGraphUpdate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.cuda
@pytest.mark.parametrize("initialized", [False, True])
def test_capture_preserves_initial_state_and_matches_adam_updates(initialized):
    torch.manual_seed(1)
    reference = torch.nn.Linear(7, 3, device="cuda")
    candidate = copy.deepcopy(reference)
    opts = [torch.optim.Adam(m.parameters(), lr=3e-4, fused=True, capturable=True) for m in (reference, candidate)]
    x = torch.randn(32, 7, device="cuda")
    y = torch.randn(32, 3, device="cuda")

    def loss_for(model):
        def fn(a, b):
            loss = (model(a) - b).square().mean()
            return loss, {"loss": loss.detach()}
        return fn

    fns = [loss_for(m) for m in (reference, candidate)]
    if initialized:
        for fn, opt in zip(fns, opts):
            fn(x, y)[0].backward()
            opt.step()
    before = [p.detach().clone() for p in candidate.parameters()]
    graph = CudaGraphUpdate(fns[1], opts[1], (x, y), modules=(candidate,), compile_loss=False)
    for a, b in zip(before, candidate.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for index in range(5):
        rate = 3e-4 / (index + 1)
        opts[0].param_groups[0]["lr"] = torch.tensor(rate, device="cuda")
        graph.set_learning_rate(rate)
        opts[0].zero_grad(set_to_none=False)
        expected, _ = fns[0](x + index, y)
        expected.backward()
        opts[0].step()
        actual, metrics = graph(x + index, y)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(metrics["loss"], actual)
        for a, b in zip(reference.parameters(), candidate.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


def test_requires_cuda_inputs_before_initializing_cuda():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter], fused=True)
    with pytest.raises(ValueError, match="CUDA tensors"):
        CudaGraphUpdate(lambda x: x, optimizer, (torch.zeros(1),))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_rejects_stream_changes_and_differentiable_inputs():
    parameter = torch.nn.Parameter(torch.ones(1, device="cuda"))
    optimizer = torch.optim.Adam([parameter], fused=True)
    x = torch.ones(1, device="cuda")
    fn = lambda value: ((parameter * value).sum(), parameter.detach())
    with pytest.raises(ValueError, match="require gradients"):
        CudaGraphUpdate(fn, optimizer, (x.clone().requires_grad_(),), compile_loss=False)
    graph = CudaGraphUpdate(fn, optimizer, [x], compile_loss=False)
    with pytest.raises(ValueError, match="require gradients"):
        graph(x.clone().requires_grad_())
    with torch.cuda.stream(torch.cuda.Stream()):
        with pytest.raises(RuntimeError, match="caller CUDA stream"):
            graph(x)
        with pytest.raises(RuntimeError, match="caller CUDA stream"):
            graph.set_learning_rate(0.01)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_adamw_clipping_group_rates_and_post_step_match_reference():
    torch.manual_seed(1)
    reference = torch.nn.Linear(5, 3, device="cuda")
    with torch.no_grad():
        reference.bias.fill_(0.2)
    candidate = copy.deepcopy(reference)

    def optimizer_for(model):
        return torch.optim.AdamW(
            [{"params": [model.weight], "lr": 0.03},
             {"params": [model.bias], "lr": 0.02}],
            weight_decay=0.1, fused=True, capturable=True,
        )

    reference_optimizer, candidate_optimizer = (optimizer_for(model) for model in (reference, candidate))
    x = torch.full((16, 5), 8.0, device="cuda")
    y = torch.full((16, 3), -2.0, device="cuda")

    def loss_for(model):
        def loss_fn(inputs, targets):
            error = model(inputs) - targets
            loss = error.square().mean()
            return loss, {"mean_error": error.detach().mean()}
        return loss_fn

    reference_loss = loss_for(reference)
    before = tuple(parameter.detach().clone() for parameter in candidate.parameters())
    graph = CudaGraphUpdate(
        loss_for(candidate), candidate_optimizer, (x, y), modules=(candidate,),
        max_grad_norm=0.5, post_step=lambda: candidate.bias.clamp_(-0.05, 0.05),
        compile_loss=True,
    )
    for actual, expected in zip(candidate.parameters(), before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for index in range(3):
        weight_rate = 0.03 / (index + 1)
        graph.set_learning_rate(weight_rate, group=0)
        reference_optimizer.param_groups[0]["lr"] = torch.tensor(weight_rate, device="cuda")
        reference_optimizer.param_groups[1]["lr"] = torch.tensor(0.02, device="cuda")
        torch.testing.assert_close(candidate_optimizer.param_groups[1]["lr"],
                                   torch.tensor(0.02, device="cuda"), rtol=0, atol=0)
        reference_optimizer.zero_grad(set_to_none=False)
        expected_loss, expected_metrics = reference_loss(x + index, y)
        expected_loss.backward()
        unclipped_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.5, foreach=True)
        assert unclipped_norm > 0.5  # This case must exercise actual clipping.
        reference_optimizer.step()
        with torch.no_grad():
            reference.bias.clamp_(-0.05, 0.05)
        actual_loss, actual_metrics = graph(x + index, y)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_metrics, expected_metrics, rtol=1e-5, atol=1e-6)
        assert (candidate.bias.abs() <= 0.05).all()
        clipped_norm = torch.stack([parameter.grad.square().sum() for parameter in candidate.parameters()]).sum().sqrt()
        assert clipped_norm <= 0.5 + 1e-6
        for expected, actual in zip(reference.parameters(), candidate.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
            for name, expected_state in reference_optimizer.state[expected].items():
                torch.testing.assert_close(candidate_optimizer.state[actual][name], expected_state,
                                           rtol=1e-5, atol=2e-7)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_capture_restores_rng_and_module_buffers_then_replays_stochastic_updates():
    class StatefulLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.7, device="cuda"))
            self.register_buffer("calls", torch.zeros((), dtype=torch.int64, device="cuda"))
            self.register_buffer("running", torch.tensor(0.25, device="cuda"))

        def forward(self, inputs):
            draw = torch.rand_like(inputs)
            with torch.no_grad():
                self.calls.add_(1)
                self.running.mul_(0.8).add_(draw.mean() * 0.2)
            loss = (self.scale * (inputs + draw)).square().mean() + self.running
            return loss, {"draw": draw, "running": self.running.detach().clone(),
                          "calls": self.calls.detach().clone()}

    torch.manual_seed(1)
    reference = StatefulLoss()
    candidate = copy.deepcopy(reference)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.001, fused=True, capturable=True)
    candidate_optimizer = torch.optim.Adam(candidate.parameters(), lr=0.001, fused=True, capturable=True)
    inputs = torch.ones(23, device="cuda")
    before = tuple(tensor.detach().clone() for tensor in (*candidate.parameters(), *candidate.buffers()))
    rng_before = torch.cuda.get_rng_state()
    graph = CudaGraphUpdate(candidate, candidate_optimizer, (inputs,), modules=(candidate,), compile_loss=False)
    assert torch.equal(torch.cuda.get_rng_state(), rng_before)
    for actual, expected in zip((*candidate.parameters(), *candidate.buffers()), before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for state in candidate_optimizer.state.values():
        for value in state.values():
            assert torch.count_nonzero(value) == 0

    for index in range(3):
        rng = torch.cuda.get_rng_state()
        reference_optimizer.zero_grad(set_to_none=False)
        expected_loss, expected_metrics = reference(inputs + index)
        expected_loss.backward()
        reference_optimizer.step()
        expected_rng = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(rng)
        actual_loss, actual_metrics = graph(inputs + index)
        assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
        torch.testing.assert_close(actual_metrics["draw"], expected_metrics["draw"], rtol=0, atol=0)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-6, atol=1e-7)
        for name in ("running", "calls"):
            torch.testing.assert_close(actual_metrics[name], expected_metrics[name], rtol=0, atol=0)
        for actual, expected in zip(candidate.buffers(), reference.buffers()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for expected, actual in zip(reference.parameters(), candidate.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            for name, expected_state in reference_optimizer.state[expected].items():
                torch.testing.assert_close(candidate_optimizer.state[actual][name], expected_state,
                                           rtol=1e-6, atol=1e-7)


class _CompiledPeerIllegalAccess(RuntimeError):
    """Only the reproduced manual-replay/compiled-peer CUDA fault."""


_KNOWN_PEER_FAULT = "KNOWN_MANUAL_CAPTURE_COMPILED_PEER_ILLEGAL_ACCESS"
_PEER_DIAGNOSTIC = """
import runpy
import sys

namespace = runpy.run_path(sys.argv[1])
try:
    namespace['_manual_update_capture_after_compiled_inference'](True)
except namespace['_CompiledPeerIllegalAccess'] as error:
    print(namespace['_KNOWN_PEER_FAULT'], flush=True)
    print(str(error), file=sys.stderr, flush=True)
    sys.exit(86)
"""


def _run_compiled_peer_subprocess():
    # Remain in the mlq runner's foreground process group. A fresh interpreter
    # contains a poisoned CUDA context without contaminating other GPU tests.
    result = subprocess.run(
        [sys.executable, "-c", _PEER_DIAGNOSTIC, str(Path(__file__).resolve())],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        timeout=300,
    )
    if result.returncode == 86 and _KNOWN_PEER_FAULT in result.stdout.splitlines():
        raise _CompiledPeerIllegalAccess(result.stderr)
    assert result.returncode == 0, (
        f"compiled-peer diagnostic exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("interleave_compiled_reference", [
    False,
    pytest.param(True, marks=pytest.mark.xfail(
        strict=True, raises=_CompiledPeerIllegalAccess,
        reason="manual update replay with a live compiled peer has a known CUDA illegal access; unsupported",
    )),
])
def test_manual_update_capture_after_compiled_inference_on_same_parameters(interleave_compiled_reference):
    if interleave_compiled_reference:
        _run_compiled_peer_subprocess()
    else:
        _manual_update_capture_after_compiled_inference(False)


@pytest.mark.parametrize("returncode, stdout, error", [
    (0, "", None),
    (86, _KNOWN_PEER_FAULT + "\n", _CompiledPeerIllegalAccess),
    (86, "unrelated failure", AssertionError),
    (1, _KNOWN_PEER_FAULT + "\n", AssertionError),
    (-11, "", AssertionError),
])
def test_compiled_peer_subprocess_failure_classification(monkeypatch, returncode, stdout, error):
    """CPU-only: never mistake other failures or a crash for the known fault."""
    def run(command, **kwargs):
        assert command[:2] == [sys.executable, "-c"]
        assert Path(command[-1]) == Path(__file__).resolve()
        assert kwargs["timeout"] == 300
        assert "start_new_session" not in kwargs
        return subprocess.CompletedProcess(command, returncode, stdout, "diagnostic stderr")

    monkeypatch.setattr(subprocess, "run", run)
    if error is None:
        _run_compiled_peer_subprocess()
    else:
        with pytest.raises(error):
            _run_compiled_peer_subprocess()


def _manual_update_capture_after_compiled_inference(interleave_compiled_reference):
    """Exercise Inductor inference-graph ownership before manual learner capture.

    The synthetic v30 update benchmark captures a fresh model. Real collectors
    first use those SAME parameters in compiled no-grad inference and GAE. An
    independent compiled learner can also be live during equivalence checks.
    Keep both graph types alive and alternate inference/updates across replays.
    """
    torch.manual_seed(1)
    reference = torch.nn.Sequential(
        torch.nn.Linear(7, 32, device="cuda"), torch.nn.ReLU(),
        torch.nn.Linear(32, 3, device="cuda"),
    )
    candidate = copy.deepcopy(reference)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=3e-4, fused=True, capturable=True)
    candidate_optimizer = torch.optim.Adam(candidate.parameters(), lr=3e-4, fused=True, capturable=True)
    x = torch.randn(32, 7, device="cuda")
    y = torch.randn(32, 3, device="cuda")

    def inference(inputs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return candidate(inputs).float()

    def loss_for(model):
        def loss_fn(inputs, targets):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(inputs).float()
            loss = (prediction - targets).square().mean()
            return loss, {"prediction_mean": prediction.detach().mean()}
        return loss_fn

    compiled_inference = torch.compile(inference, mode="reduce-overhead", fullgraph=True)
    # Both learners must use the same compiled BF16 arithmetic. This switch
    # isolates graph-tree interleaving, not eager-vs-compiled gradient rounding.
    reference_loss = torch.compile(loss_for(reference), fullgraph=True,
                                   options={"triton.cudagraphs": interleave_compiled_reference})

    # Three full-shape invocations warm, record and replay the inference graph.
    # Snapshots are allocated outside capture; no saved autograd graph survives.
    with torch.no_grad():
        for _ in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            observed = compiled_inference(x).clone()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected_prediction = reference(x).float()
        torch.testing.assert_close(observed, expected_prediction, rtol=0, atol=0)
    graph = None
    for iteration in range(3):
        inputs = x + iteration
        reference_optimizer.zero_grad(set_to_none=True)
        if interleave_compiled_reference:
            torch.compiler.cudagraph_mark_step_begin()
        original_loss, original_metrics = reference_loss(inputs, y)
        original_loss.backward()
        reference_optimizer.step()
        expected_loss = original_loss.detach().clone()
        expected_metrics = {key: value.detach().clone() for key, value in original_metrics.items()}
        del original_loss, original_metrics
        if graph is None:
            graph = CudaGraphUpdate(loss_for(candidate), candidate_optimizer, (inputs, y),
                                    modules=(candidate,), compile_loss=True)
        try:
            actual_loss, actual_metrics = graph(inputs, y)
            # Attribute invalid pointers to replay, not an unrelated cleanup.
            torch.cuda.synchronize()
        except RuntimeError as error:
            if "CUDA error: an illegal memory access was encountered" in str(error):
                raise _CompiledPeerIllegalAccess(str(error)) from error
            raise
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_metrics, expected_metrics, rtol=1e-5, atol=1e-6)
        for expected, actual in zip(reference.parameters(), candidate.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        # The manual graph remains alive while Inductor advances its own graph
        # generation, matching collection between learner updates.
        with torch.no_grad():
            torch.compiler.cudagraph_mark_step_begin()
            actual_prediction = compiled_inference(inputs).clone()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                expected_prediction = reference(inputs).float()
            torch.testing.assert_close(actual_prediction, expected_prediction, rtol=1e-5, atol=1e-6)
