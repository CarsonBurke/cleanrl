"""Ordered optimizer and CUDA-graph lifetime contracts for indexed PPO loss."""

import copy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action as ppo
from cleanrl.shared.ppo_update import make_minibatch_loss
from cleanrl.shared.runtime import configure_runtime


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available(), reason="queued CUDA execution required")]


@pytest.mark.parametrize("compiled", [False, True])
def test_indexed_loss_preserves_repeated_indices_and_order_in_backward(compiled):
    # Repeated rows must accumulate gradients, not become a set or a shuffle.
    weights = torch.tensor([0.5, -2.0, 3.0], device="cuda", requires_grad=True)
    features = torch.arange(15, device="cuda", dtype=torch.float32).reshape(5, 3)
    targets = torch.tensor([1.0, 2.0, -1.0, 0.5, 4.0], device="cuda")
    indices = torch.tensor([4, 1, 4, 0], device="cuda")

    def loss_fn(x, y):
        errors = x @ weights - y
        return errors.square().mean(), errors.detach()

    loss, metrics = make_minibatch_loss(loss_fn, compiled=compiled)(indices, features, targets)
    loss.backward()
    errors = features[indices] @ weights.detach() - targets[indices]
    torch.testing.assert_close(metrics, errors, rtol=0, atol=0)
    torch.testing.assert_close(weights.grad, (2 * errors[:, None] * features[indices]).mean(0))


def test_indexed_compiled_loss_matches_adam_across_peer_inference_and_reused_rollouts():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float64),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
    )
    initial = ppo.Agent(spaces).cuda()
    args = ppo.Args()
    batch_size, minibatch_size = 2048, 64
    observations = torch.randn(3, batch_size, 17, device="cuda")
    native = torch.rand(3, batch_size, 6, device="cuda") * 0.8 + 0.1
    advantages = torch.randn(3, batch_size, device="cuda")
    indices = torch.stack([torch.randperm(batch_size, device="cuda")[:minibatch_size] for _ in range(4)])
    expected = []
    for indexed in (False, True):
        model = copy.deepcopy(initial)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)

        def raw_loss(*batch):
            return ppo.ppo_loss(model, *batch, args)

        if indexed:
            loss_fn = make_minibatch_loss(raw_loss)
        else:
            compiled_loss = torch.compile(raw_loss, mode="reduce-overhead", fullgraph=True, dynamic=False)

            def loss_fn(selected, *batch):
                torch.compiler.cudagraph_mark_step_begin()
                return compiled_loss(*(tensor[selected] for tensor in batch))

        def statistics(obs, actions):
            alpha, beta, value = model.get_policy_and_value(obs)
            return value.flatten(), model.action_logprob(alpha, beta, actions)

        peer = torch.compile(statistics, mode="reduce-overhead", fullgraph=True, dynamic=False)
        buffers = (torch.empty_like(observations[0]), torch.empty_like(native[0]),
                   *(torch.empty(batch_size, device="cuda") for _ in range(4)))
        retained_metrics = torch.empty(12, 6, device="cuda")
        sequence = 0
        for rollout in range(3):
            with torch.no_grad():
                # Same allocations are overwritten for the next rollout, like upload().
                buffers[0].copy_(observations[rollout])
                buffers[1].copy_(native[rollout])
                torch.compiler.cudagraph_mark_step_begin()
                values, logprobs = peer(buffers[0], buffers[1])
                # Consume graph-owned outputs before the learner's step marker.
                buffers[2].copy_(logprobs)
                buffers[5].copy_(values)
                del values, logprobs
                buffers[2].add_(torch.linspace(-0.4, 0.4, batch_size, device="cuda"))
                buffers[3].copy_(advantages[rollout])
                buffers[4].copy_(buffers[5]).add_(8)
                buffers[5].add_(torch.linspace(-0.5, 0.5, batch_size, device="cuda"))
            rng = torch.cuda.get_rng_state()
            for selected in indices:
                loss, metrics = loss_fn(selected, *buffers)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradients = tuple(p.grad.detach().clone() for p in model.parameters())
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                clipped = tuple(p.grad.detach().clone() for p in model.parameters())
                optimizer.step()
                retained_metrics[sequence].copy_(metrics)
                snapshot = (loss.detach().clone(), gradients, clipped, norm.detach().clone(),
                            tuple(p.detach().clone() for p in model.parameters()),
                            tuple({key: value.detach().clone() for key, value in optimizer.state[p].items()}
                                  for p in model.parameters()))
                if indexed:
                    reference = expected[sequence]
                    torch.testing.assert_close(snapshot[0], reference[0], atol=2e-6, rtol=1e-5)
                    for position in (1, 2, 3):
                        torch.testing.assert_close(snapshot[position], reference[position], atol=2e-6, rtol=1e-5)
                    torch.testing.assert_close(snapshot[4], reference[4], atol=2e-7, rtol=2e-6)
                    torch.testing.assert_close(snapshot[5], reference[5], atol=2e-7, rtol=1e-5)
                else:
                    expected.append(snapshot)
                sequence += 1
                del loss, metrics, snapshot
            assert torch.equal(torch.cuda.get_rng_state(), rng)
        # Includes snapshots surviving both a subsequent update and peer replay.
        if indexed:
            torch.testing.assert_close(retained_metrics, expected_metrics, atol=2e-6, rtol=1e-5)
        else:
            expected_metrics = retained_metrics.clone()
        torch.cuda.synchronize()
