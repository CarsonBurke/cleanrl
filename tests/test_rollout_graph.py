"""Captured per-step rollout graph: storage slots, ring wrap, host action, RNG."""

import numpy as np
import pytest
import torch

from cleanrl.shared.rollout_graph import RolloutStepGraph, graph_compile
from cleanrl.shared.runtime import configure_runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def deterministic_policy(observations):
    return {"action": torch.tanh(observations[:, :2]) * 3.0, "value": observations.sum(-1),
            "flag": (observations[:, 0] > 0)}


def test_steps_store_inputs_and_outputs_at_slots_and_return_host_actions():
    graph = RolloutStepGraph(deterministic_policy, 4, 3, (5,), "cuda")
    rng = np.random.default_rng(0)
    seen = []
    for step in range(6):  # wraps past num_steps once
        host = rng.standard_normal((3, 5))  # float64 input, cast on copy
        action = graph.step(host).copy()
        expected = deterministic_policy(torch.as_tensor(host, dtype=torch.float32, device="cuda"))
        np.testing.assert_array_equal(action, expected["action"].cpu().numpy())
        seen.append((host.astype(np.float32), expected))
    for step, (host, expected) in enumerate(seen[2:], start=2):  # slots 2,3 then 0,1 after wrap
        slot = step % 4
        np.testing.assert_array_equal(graph.observations[slot].cpu().numpy(), host)
        torch.testing.assert_close(graph.outputs["value"][slot], expected["value"], rtol=0, atol=0)
        assert graph.outputs["flag"].dtype == torch.bool
        torch.testing.assert_close(graph.outputs["flag"][slot], expected["flag"])
    graph.reset()
    host = rng.standard_normal((3, 5))
    graph.step(host)
    np.testing.assert_array_equal(graph.observations[0].cpu().numpy(), host.astype(np.float32))
    staged = graph.stage_observation(host * 2)
    np.testing.assert_array_equal(staged.cpu().numpy(), (host * 2).astype(np.float32))
    assert staged is graph.observation


def test_sampling_inside_graph_is_seeded_and_advances_per_replay():
    def stochastic(observations):
        return {"action": observations + torch.randn_like(observations)}

    def draws(seed):
        torch.manual_seed(seed)
        graph = RolloutStepGraph(stochastic, 2, 4, (3,), "cuda")
        return [graph.step(np.zeros((4, 3), np.float32)).copy() for _ in range(3)]

    first, second = draws(7), draws(7)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[0], first[1])
    assert not np.array_equal(draws(8)[0], first[0])


def test_compiled_policy_captures_and_matches_eager():
    torch.manual_seed(0)
    linear = torch.nn.Linear(6, 2).cuda()

    def policy(observations):
        return {"action": torch.tanh(linear(observations)), "value": linear(observations).sum(-1)}

    eager = RolloutStepGraph(policy, 3, 8, (6,), "cuda")
    compiled = RolloutStepGraph(graph_compile(policy), 3, 8, (6,), "cuda")
    host = np.random.default_rng(1).standard_normal((8, 6)).astype(np.float32)
    np.testing.assert_allclose(eager.step(host), compiled.step(host), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(eager.outputs["value"][0], compiled.outputs["value"][0], rtol=1e-5, atol=1e-6)


def test_rejects_outputs_without_action_or_wrong_batch():
    with pytest.raises(ValueError, match="action"):
        RolloutStepGraph(lambda o: {"value": o.sum(-1)}, 2, 2, (3,), "cuda")
    with pytest.raises(ValueError, match="leading dimension"):
        RolloutStepGraph(lambda o: {"action": o[:1]}, 2, 2, (3,), "cuda")


def test_replays_observe_in_place_updates_beside_cudagraph_tree_learner():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(6, 32), torch.nn.Tanh(), torch.nn.Linear(32, 3)).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, fused=True)
    graph = RolloutStepGraph(lambda o: {"action": torch.tanh(net(o)), "value": net(o).sum(-1)},
                             3, 8, (6,), "cuda")
    loss_model = torch.compile(lambda o: net(o).square().mean(), mode="reduce-overhead", fullgraph=True, dynamic=False)
    host = np.random.default_rng(1).standard_normal((3, 8, 6)).astype(np.float32)
    for _ in range(3):
        graph.reset()
        for step in range(3):
            graph.step(host[step])
        with torch.no_grad():
            expected = net(graph.observations.flatten(0, 1)).sum(-1).view(3, 8)
        torch.testing.assert_close(graph.outputs["value"], expected, rtol=1e-5, atol=1e-6)
        for _ in range(2):
            torch.compiler.cudagraph_mark_step_begin()
            loss = loss_model(graph.observations.flatten(0, 1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
