"""Archived failing numerical probe for the retired full-reuse optimizer family."""

import json
from pathlib import Path

import pytest
import torch

from cleanrl.plasticity import network_bayes_stream_v2 as reference
from cleanrl.plasticity.optimizer_proxy_eval_v7 import Args, RoundLearner, generator, training_task
from cleanrl.shared import runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@torch.no_grad()
def test_full_reuse_high_update_grid_capture_preserves_transition_and_replay():
    runtime.configure_runtime()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    args = Args()
    plan = json.loads(Path(args.plan).read_text())
    scenario = next(s for s in plan["suite"] if s["name"] == "drifting_reuse")
    initial = reference.init_weights(args, generator(1, 1), "cuda")
    teacher_gen = generator(1, 2)
    teachers = [reference.draw_teacher(args, teacher_gen, "cuda") for _ in range(9)]
    train_gen = generator(1, 3)
    count, batch = scenario["samples"], scenario["batch_size"]
    xs = torch.randn(count, args.input_dim, generator=train_gen, device="cuda")
    noise = torch.randn(count, generator=train_gen, device="cuda")
    policy_noise = torch.randn(count, generator=train_gen, device="cuda")
    xs, targets = training_task(teachers, xs, batch, True)
    learner = RoundLearner(initial, plan["configurations"], scenario, "round_full")
    learner.x.copy_(xs[:batch])
    learner.target.copy_(targets[:batch] + noise[:batch])
    learner.action_noise.copy_(policy_noise[:batch])
    learner.reward_noise.copy_(noise[:batch])
    before = [x.clone() for x in learner.mutable]
    # Internally checks every local transition against eager at exactly that
    # state, finite masks, and exact repeated graph replay. Failed in v6.
    graph = learner.capture()
    for actual, expected in zip(learner.mutable, before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    graph.replay()
    first = [x.clone() for x in learner.mutable]
    learner.restore(before)
    graph.replay()
    for actual, expected in zip(learner.mutable, first):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert int(learner.step) == scenario["epochs"]
