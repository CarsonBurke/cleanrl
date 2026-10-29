"""In-context improvement operator contracts; run CUDA work exclusively through mlq."""
import torch

from cleanrl.shared.ppo_loop import compute_gae
from cleanrl.tilt.ppo_continuous_action_sepclip_incontext_operator_v2 import (
    ImprovementOperator, trace_advantages,
)

LAMBDAS = (0.0, 0.6, 0.9, 0.95, 0.99, 1.0)


def test_trace_advantages_match_gae_at_every_lambda_across_episode_boundaries():
    torch.manual_seed(1)
    steps, envs = 96, 5
    rewards, values, bootstrap = (torch.randn(steps, envs, device="cuda") for _ in range(3))
    tail = torch.randn(envs, device="cuda")
    terminations = (torch.rand(steps, envs, device="cuda") < 0.05).float()
    truncations = ((torch.rand(steps, envs, device="cuda") < 0.05).float() * (1.0 - terminations))
    fields = (rewards, values, terminations, truncations, bootstrap, tail, 0.99)
    traces = trace_advantages(compute_gae(*fields, 0.95)[0], terminations, truncations, 0.99, 0.95, LAMBDAS)
    assert traces.shape == (steps, envs, len(LAMBDAS))
    for index, trace_lambda in enumerate(LAMBDAS):
        torch.testing.assert_close(traces[..., index], compute_gae(*fields, trace_lambda)[0], atol=2e-4, rtol=2e-4)


def fixture(queries=64, context=256, envs=4, observation_dim=17, action_dim=6):
    torch.manual_seed(1)
    operator = ImprovementOperator(observation_dim, action_dim, len(LAMBDAS), heads=4, key_dim=16).cuda()
    # Exercise every path: the memory gain and the evidence readout start at zero.
    torch.nn.init.ones_(operator.memory.readout_gain)
    torch.nn.init.normal_(operator.evidence[-1].weight)
    draw = lambda *shape: torch.randn(*shape, device="cuda")
    environments = torch.arange(context, device="cuda") % envs
    return operator, (draw(queries, observation_dim), torch.arange(queries, device="cuda") % envs), [
        draw(context, observation_dim), environments, draw(context, len(LAMBDAS)), draw(context),
        draw(context, 2 * action_dim)]


@torch.no_grad()
def test_tilt_ignores_every_sample_from_the_query_environment():
    operator, (query, query_environments), context = fixture()
    expected = operator(query, query_environments, *context)[0]
    own = context[1] == 0
    for index in (0, 2, 3, 4):
        context[index] = torch.where(own.view(-1, *[1] * (context[index].ndim - 1)),
                                     torch.randn_like(context[index]), context[index])
    changed = operator(query, query_environments, *context)[0]
    asked = query_environments == 0
    torch.testing.assert_close(changed[asked], expected[asked])
    assert not torch.allclose(changed[~asked], expected[~asked])


@torch.no_grad()
def test_initial_operator_is_kernel_smoothed_policy_gradient():
    torch.manual_seed(1)
    operator = ImprovementOperator(17, 6, len(LAMBDAS), heads=4, key_dim=16).cuda()
    _, (query, query_environments), (observations, environments, evidence, reference, scores) = fixture()
    tilt, memory, context = operator(query, query_environments, observations, environments, evidence, reference, scores)
    assert memory.abs().max() == 0.0
    logits = 8.0 * torch.einsum("qhd,chd->hqc", operator.keys(query), operator.keys(observations))
    logits = logits.masked_fill(query_environments[:, None] == environments[None, :], -torch.inf)
    expected = (logits.softmax(-1) @ (reference[:, None] * scores)).mean(0)
    torch.testing.assert_close(tilt, expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(context, tilt)
