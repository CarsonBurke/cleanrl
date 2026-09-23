"""Numerical contracts for the projected Beta gradient control; run through mlq."""
import numpy as np
import pytest
import torch
from torch import nn
from cleanrl import ppo_continuous_action_normres_beta_gradient_v2 as m


def test_projected_score_and_correction_match_autograd():
    torch.manual_seed(1)
    logits = torch.randn(7, 4, device="cuda", dtype=torch.float64, requires_grad=True)
    directions = torch.randn(3, 7, 4, device="cuda", dtype=torch.float64)
    alpha, beta = (torch.nn.functional.softplus(logits) + 1).chunk(2, -1)
    reference = tuple(t.detach() for t in m.beta_score_reference(alpha, beta))
    actions = torch.linspace(.1, .9, 14, device="cuda", dtype=torch.float64).reshape(7, 2)
    scores, derivatives = m.projected_beta_statistics(logits, directions, actions, reference)
    for k in range(3):
        def logprob(z):
            a, b = (torch.nn.functional.softplus(z) + 1).chunk(2, -1)
            return torch.distributions.Beta(a, b).log_prob(actions).sum(-1)
        def means(z):
            a, b = (torch.nn.functional.softplus(z) + 1).chunk(2, -1)
            return m.beta_score_means(a, b, reference)
        torch.testing.assert_close(scores[:, k], torch.func.jvp(logprob, (logits,), (directions[k],))[1])
        torch.testing.assert_close(derivatives[:, k], torch.func.jvp(means, (logits,), (directions[k],))[1])


def test_blocks_preserve_environment_trajectories():
    data = torch.arange(48, device="cuda").reshape(8, 2, 3).float()
    expected = torch.stack([data[t:t+4, e].mean(0) for t in (0, 4) for e in range(2)])
    torch.testing.assert_close(m.block_gradients(data.flatten(0, 1), 2, 4), expected)


def test_compiled_projection_and_control_gradient():
    torch.manual_seed(1)
    actor = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 4)).cuda()
    observations = torch.randn(32, 3, device="cuda")
    actions = torch.rand(32, 2, device="cuda") * .8 + .1
    with torch.no_grad():
        a, b = (torch.nn.functional.softplus(actor(observations)) + 1).chunk(2, -1)
        reference = m.beta_score_reference(a, b)
        features = m.beta_score_features(actions, reference)
        directions = m.sample_parameter_directions(actor, 3, torch.Generator(device="cuda").manual_seed(1))
        project = m.make_projection_function(actor)
        expected = project(observations, actions, reference, directions)
        compiled = torch.compile(project, fullgraph=True, mode="reduce-overhead")
        actual = tuple(t.clone() for t in compiled(observations, actions, reference, directions))
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-4)
    control = nn.Linear(3, 5).cuda()
    advantages = torch.randn(32, device="cuda", requires_grad=True)
    def loss():
        return m.control_loss(control, observations, advantages, features, actual[0], actual[1], torch.ones((), device="cuda"), 2, 4)
    eager = loss()
    expected_grad = torch.autograd.grad(eager, tuple(control.parameters()))
    torch.compiler.cudagraph_mark_step_begin()
    compiled_loss = torch.compile(loss, fullgraph=True, mode="reduce-overhead")
    compiled_loss().backward()
    for p, g in zip(control.parameters(), expected_grad):
        torch.testing.assert_close(p.grad, g, atol=2e-5, rtol=2e-4)
    assert advantages.grad is None
    assert all(p.grad is None for p in actor.parameters())


def test_frozen_observations_use_terminal_not_reset_state():
    state = dict(means=torch.tensor([[1., 2.], [3., 4.]]), variances=torch.ones(2, 2), counts=torch.ones(2), epsilon=0., clip=10.)
    norm = m.FrozenObsNorm(state, 2, (2,))
    old = norm.normalize(np.array([[2., 3.], [4., 5.]]))
    saved = old.copy()
    nxt, terminal = norm.normalize_step(np.zeros((2, 2)), np.array([False, False]), np.array([True, False]), {"final_observation": [np.array([5., 6.]), None]})
    np.testing.assert_array_equal(old, saved)
    np.testing.assert_array_equal(terminal[0], [4., 4.])
    np.testing.assert_array_equal(nxt[0], [-1., -2.])


@pytest.mark.parametrize("ratio,passes", [(0.8, True), (1.1, False), (.98, False)])
def test_gate_requires_meaningful_heldout_reduction(ratio, passes):
    rows = [{"gradient/ordinary_block_variance": float(i), "gradient/corrected_block_variance": ratio*i,
             "gradient/ordinary_variance_trace": float(i), "gradient/corrected_variance_trace": ratio*i} for i in range(1, 17)]
    assert m.summarize_gate(rows)["passed"] is passes
