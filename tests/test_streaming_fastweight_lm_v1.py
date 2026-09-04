import importlib.util
import math
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "streaming-posterior-filter"
    / "streaming_fastweight_lm_v1.py"
)
spec = importlib.util.spec_from_file_location("streaming_fastweight_lm_v1", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
loader = spec.loader
assert loader is not None
loader.exec_module(module)
Args = module.Args
StreamingFastWeightLM = module.StreamingFastWeightLM
evaluate_heldout = module.evaluate_heldout
make_stream = module.make_stream


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "fast-weight tests require CUDA"
    return torch.device("cuda")


@pytest.fixture
def model(device):
    torch.manual_seed(1)
    args = Args(dim=8, hidden_dim=16, fast_heads=2, fast_eta_max=0.5)
    return StreamingFastWeightLM(args).to(device)


def run_step(model, device, target, memory=None, hidden=None, fast_scale=1.0, use_fast=True):
    if memory is None:
        memory = model.initial_memory(device)
    if hidden is None:
        hidden = model.initial_hidden(device)
    return model.forward_step(
        torch.tensor(65, device=device),
        torch.tensor(target, device=device),
        memory,
        hidden,
        torch.tensor(fast_scale, device=device),
        use_fast,
        True,
        5.0,
    )


def test_target_is_not_visible_to_current_prediction(model, device):
    first = run_step(model, device, 66)
    second = run_step(model, device, 90)

    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
    torch.testing.assert_close(first[3], second[3], rtol=0, atol=0)
    assert not torch.equal(first[2], second[2])


def test_fast_memory_is_forward_state_not_slow_weight(model, device):
    memory = model.initial_memory(device)
    result = run_step(model, device, 66, memory=memory)

    torch.testing.assert_close(memory, torch.zeros_like(memory), rtol=0, atol=0)
    assert result[2].square().sum() > 0
    assert result[4] >= 0
    assert result[5] >= 0
    assert 0 < result[6] <= 1
    assert 0 < result[7] < 0.5


def test_student_weight_rejects_large_normalized_write(model, device):
    ordinary = run_step(model, device, 66, fast_scale=1.0)
    surprising = run_step(model, device, 66, fast_scale=1e-6)

    assert surprising[6] < ordinary[6]
    assert surprising[4] < ordinary[4]
    assert surprising[2].square().sum() < ordinary[2].square().sum()


def test_slow_control_never_mutates_fast_memory(model, device):
    memory = torch.randn_like(model.initial_memory(device))
    result = run_step(model, device, 66, memory=memory, use_fast=False)

    torch.testing.assert_close(result[2], memory, rtol=0, atol=0)
    assert result[4] == 0
    assert result[5] == 0
    assert result[6] == 0
    assert result[7] == 0


def test_training_and_validation_corpus_are_disjoint(device):
    args = Args(
        total_steps=100,
        validation_fraction=0.2,
        validation_steps=16,
    )
    clean, _, _, validation, training_bytes = make_stream(args, device)
    corpus = Path(args.corpus_path).read_bytes()
    expected_split = int(len(corpus) * 0.8)

    assert training_bytes == expected_split
    assert clean.cpu().tolist() == list(corpus[:101])
    assert validation[:17].cpu().tolist() == list(corpus[expected_split : expected_split + 17])


def test_heldout_inference_is_frozen_and_reports_ground_truth(model, device):
    args = Args(
        dim=8,
        hidden_dim=16,
        fast_heads=2,
        validation_steps=16,
        inference_rows=8,
    )
    validation = torch.arange(17, device=device, dtype=torch.long)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    result = evaluate_heldout(
        model,
        validation,
        args,
        device,
        torch.ones((), device=device),
    )

    assert result["metrics"]["tokens"] == 16
    assert math.isfinite(result["metrics"]["nll"])
    assert math.isfinite(result["metrics"]["perplexity"])
    assert 0.0 <= result["metrics"]["top1_accuracy"] <= 1.0
    assert 0.0 <= result["metrics"]["top5_accuracy"] <= 1.0
    assert result["truth_ids"] == list(range(1, 9))
    assert len(result["top_ids"]) == 8
    assert len(result["top_probs"]) == 8
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
