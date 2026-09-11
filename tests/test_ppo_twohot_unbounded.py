"""Raw residual readouts must agree in the CUDA learner and fused rollout."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.ppo_continuous_action_32xlr_1mb_noadvnorm_normres_twohot_unbounded_v6 import (
    Agent, Args, gradient_norms, ppo_loss,
)
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.runtime import configure_runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def raw_readout(network, observations):
    trunk, head = network
    stream = trunk.in_proj(observations)
    for block, gate, norm in zip(trunk.blocks, trunk.block_gates, trunk.block_norms):
        stream = stream + gate.sigmoid() * block(norm(stream))
    return head(stream / 8.0)


def test_compiled_raw_readouts_and_native_rollout_after_update():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
    )
    agent = Agent(envs).cuda()
    obs = torch.randn(16, 17, device="cuda")
    compiled = torch.compile(agent.get_policy_and_value_output, fullgraph=True,
                             options={"triton.cudagraphs": False})
    alpha, beta, logits = compiled(obs)
    torch.testing.assert_close(logits, raw_readout(agent.critic, obs))
    actor_logits = raw_readout(agent.actor, obs)
    expected_a, expected_b = (torch.nn.functional.softplus(actor_logits) + 1).chunk(2, -1)
    torch.testing.assert_close(alpha, expected_a)
    torch.testing.assert_close(beta, expected_b)
    actions = alpha.detach() / (alpha.detach() + beta.detach())
    old_logprobs = agent.action_logprob(alpha, beta, actions).detach()
    targets = agent.value_support.project(torch.linspace(-1000, 2000, 16, device="cuda"))
    args = Args()
    loss_fn = torch.compile(lambda: ppo_loss(agent, obs, actions, old_logprobs,
                                            torch.linspace(-2, 2, 16, device="cuda"), targets, args)[0],
                            fullgraph=True, options={"triton.cudagraphs": False})
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
    optimizer.zero_grad(set_to_none=True)
    (1000 * loss_fn()).backward()
    gradients = [p.grad.clone() for p in agent.parameters() if p.grad is not None]
    gradient_norms(tuple(agent.actor.parameters()), tuple(agent.critic.parameters()))
    for before, parameter in zip(gradients, [p for p in agent.parameters() if p.grad is not None]):
        torch.testing.assert_close(parameter.grad, before, rtol=0, atol=0)
    optimizer.step()
    _, _, trained_logits = compiled(obs)
    torch.testing.assert_close(trained_logits, raw_readout(agent.critic, obs), rtol=1e-5, atol=1e-6)
    mirror = make_host_mirror(agent.actor, 16)
    assert mirror.fused, "The ablation must not fall back to a slower rollout implementation"
    host = mirror(obs.cpu().numpy()).copy()
    expected = raw_readout(agent.actor, obs).detach().cpu().numpy()
    np.testing.assert_allclose(host, expected, rtol=2e-4, atol=2e-6)
