"""Tests for shared runtime defaults and zero-sync phase timing."""

import time

import pytest
import torch

from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.timing import PhaseTimer


@pytest.fixture
def restore_torch_flags():
    state = {
        "deterministic": torch.backends.cudnn.deterministic,
        "precision": torch.get_float32_matmul_precision(),
        "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "threads": torch.get_num_threads(),
    }
    yield
    torch.backends.cudnn.deterministic = state["deterministic"]
    torch.set_float32_matmul_precision(state["precision"])
    torch.backends.cuda.matmul.allow_tf32 = state["cuda_tf32"]
    torch.backends.cudnn.allow_tf32 = state["cudnn_tf32"]
    torch.set_num_threads(state["threads"])


def test_configure_runtime_defaults(restore_torch_flags):
    configure_runtime()
    assert torch.backends.cudnn.deterministic is True
    assert torch.get_float32_matmul_precision() == "high"
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.get_num_threads() == 1


def test_configure_runtime_respects_overrides(restore_torch_flags):
    configure_runtime(
        cudnn_deterministic=False,
        matmul_precision="highest",
        allow_tf32=False,
        cpu_threads=None,  # leaves thread count alone
    )
    assert torch.backends.cudnn.deterministic is False
    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


def test_phase_timer_accumulates_cpu_spans():
    timer = PhaseTimer()
    with timer.span("env", use_cuda=False):
        time.sleep(0.01)
    with timer.span("env", use_cuda=False):
        time.sleep(0.01)
    with timer.span("update", use_cuda=False):
        pass
    stats = timer.summary()
    assert stats["env"]["calls"] == 2
    assert stats["update"]["calls"] == 1
    assert stats["env"]["total_s"] >= 0.02
    assert stats["env"]["total_s"] < 5.0
    # Summary is repeatable; reset starts a fresh window.
    assert timer.summary()["env"]["calls"] == 2
    timer.reset()
    assert timer.summary() == {}


def test_phase_timer_manual_start_stop():
    timer = PhaseTimer()
    timer.start("rollout", use_cuda=False)
    time.sleep(0.005)
    timer.stop()
    timer.start("rollout", use_cuda=False)
    timer.stop()
    stats = timer.summary()
    assert stats["rollout"]["calls"] == 2
    assert stats["rollout"]["total_s"] >= 0.005
    with pytest.raises(RuntimeError, match="without a matching start"):
        timer.stop()


def test_phase_timer_rejects_interleaving():
    timer = PhaseTimer()
    with pytest.raises(RuntimeError, match="must not interleave"):
        with timer.span("a", use_cuda=False):
            with timer.span("b", use_cuda=False):
                pass


def test_phase_timer_cuda_flag_falls_back_without_gpu():
    timer = PhaseTimer()
    with timer.span("rollout", use_cuda=True):
        time.sleep(0.005)
    stats = timer.summary()
    assert stats["rollout"]["calls"] == 1
    assert stats["rollout"]["total_s"] >= 0.005
