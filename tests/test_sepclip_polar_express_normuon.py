"""Numerical and rollout parity checks; execute CUDA tests through mlq."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.ppo_continuous_action_sepclip_polar_express_normuon_v1 import (
    Agent, Args, PolarExpressNorMuon, ResidualHostMirror, make_optimizers,
    polar_express,
)


@pytest.mark.parametrize("shape", [(64, 17), (64, 64)])
def test_compiled_multistep_matches_neuron_normalization(shape):
    torch.manual_seed(1)
    weights = [torch.nn.Parameter(torch.randn(*shape, device="cuda")) for _ in range(4)]
    opt = PolarExpressNorMuon(weights)
    moment = torch.zeros(4, *shape, device="cuda")
    second = torch.zeros(4, shape[0], 1, device="cuda")
    scale = max(1, shape[0] / shape[1]) ** 0.5
    for lr in (0.005, 0.0025, 0.0):
        old = torch.stack(weights).detach().clone()
        grads = torch.randn_like(old)
        for p, g in zip(weights, grads.unbind()):
            p.grad = g.clone()
        moment = 0.95 * moment + 0.05 * grads
        direction = polar_express(0.05 * grads + 0.95 * moment).float()
        second = 0.95 * second + 0.05 * direction.square().mean(-1, keepdim=True)
        normalized = direction / second.clamp_min(1e-10).sqrt()
        normalized *= direction.norm(dim=(-2, -1), keepdim=True) / normalized.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-10)
        opt.set_lr(lr)
        opt.step()
        actual = torch.stack(weights).detach() - old
        expected = -lr * scale * normalized
        if lr:
            # Compiled bf16 polynomial fusion differs slightly from eager.
            assert (actual - expected).norm() < 0.08 * expected.norm()
        else:
            torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)
        torch.testing.assert_close(weights[0].grad, grads[0], atol=0, rtol=0)
    state = opt.state[weights[0]]
    torch.testing.assert_close(state["momentum_buffer"], moment)


def test_optimizer_partition_and_host_parity_after_update():
    envs = SimpleNamespace(single_action_space=gym.spaces.Box(-1, 1, (6,)),
                           single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,)))
    agent = Agent(envs).cuda()
    agent.normalize_matrices()
    matrices, adam = make_optimizers(agent, Args())
    matrix_ids = {id(p) for g in matrices.param_groups for p in g["params"]}
    adam_ids = {id(p) for g in adam.param_groups for p in g["params"]}
    assert len(matrix_ids) == 12 and not matrix_ids & adam_ids
    assert matrix_ids | adam_ids == {id(p) for p in agent.parameters()}
    assert id(agent.actor.head[0].weight) in adam_ids
    obs = torch.randn(16, 17, device="cuda")
    alpha, beta, value = agent.get_policy_and_value(obs)
    (alpha.square().mean() + beta.square().mean() + value.square().mean()).backward()
    matrices.step()
    adam.step()
    agent.normalize_matrices()
    mirror = ResidualHostMirror(agent.actor, 16)
    np.testing.assert_allclose(mirror(obs.cpu().numpy()), agent.actor(obs).detach().cpu().numpy(),
                               atol=3e-5, rtol=3e-5)
