"""CUDA PPO transport contracts; execute through the machine-wide mlq queue."""

import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Beta

from cleanrl import ppo_continuous_action_prerms_adamh_v22 as frozen
from cleanrl.shared import predictive_ppo as transport
from cleanrl.shared.runtime import configure_runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


class TinyAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 4))
        self.critic = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1))
        self.register_buffer("log_action_scale", torch.tensor([0.7, 0.4]))

    def forward(self, obs):
        return self.actor(obs), self.critic(obs).flatten()

    def get_policy_and_value(self, obs):
        logits, value = self(obs)
        alpha, beta = (nn.functional.softplus(logits) + 1).chunk(2, -1)
        return alpha, beta, value


class OutputAgent:
    def __init__(self, logits, values, scale):
        self.logits, self.values, self.log_action_scale = logits, values, scale

    def get_policy_and_value(self, obs):
        alpha, beta = (nn.functional.softplus(self.logits) + 1).chunk(2, -1)
        return alpha, beta, self.values


def setup(mode="both", *, norm_adv=False):
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    agent = TinyAgent().cuda()
    args = SimpleNamespace(transport_mode=mode, clip_coef=0.2, clip_vloss=True,
                           ent_coef=0.03, vf_coef=0.5, norm_adv=norm_adv)
    obs = torch.randn(8, 3, device="cuda")
    native = torch.rand(8, 2, device="cuda") * 0.8 + 0.1
    with torch.no_grad():
        alpha, beta, values = agent.get_policy_and_value(obs)
        logprob = (Beta(alpha, beta).log_prob(native) - agent.log_action_scale).sum(-1)
        ratios = torch.tensor([0.5, 0.8, 1.0, 1.2, 1.5, 0.9, 1.3, 0.7], device="cuda")
        oldlp = logprob - ratios.log()
        advantage = torch.tensor([-2., 1., 3., -1., 2., -3., 0.5, -0.7], device="cuda")
        targets = values + torch.tensor([1., -1., 0.1, -0.1, 0.5, -0.5, 2., -2.], device="cuda")
        oldvalues = values + torch.tensor([-0.6, 0.5, 0.1, -0.1, -0.3, 0.3, 0.8, -0.8], device="cuda")
    batch = (obs, native, oldlp, advantage, targets, oldvalues)
    params = {name: p.detach() for name, p in agent.named_parameters()}
    previous = {name: p.detach().clone() + 0.04 * torch.randn_like(p) for name, p in agent.named_parameters()}
    return agent, args, params, previous, batch


def expected_scores(outputs, batch, scale, args):
    leaves = tuple(output.detach().requires_grad_(True) for output in outputs)
    loss, _ = frozen.ppo_loss(OutputAgent(*leaves, scale), *batch, args)
    return tuple(score.detach() for score in torch.autograd.grad(loss, leaves))


@pytest.mark.parametrize("mode", ["adam", "critic", "both"])
@pytest.mark.parametrize("norm_adv", [False, True])
def test_raw_gradients_and_output_transport_match_frozen_ppo_autograd(mode, norm_adv):
    agent, args, params, previous, batch = setup(mode, norm_adv=norm_adv)
    grads, corrections, metrics = transport.make_gradient_function(agent, args)(params, previous, *batch)
    loss, expected_metrics = frozen.ppo_loss(agent, *batch, args)
    expected_gradients = torch.autograd.grad(loss, tuple(agent.parameters()))
    for name, expected in zip(params, expected_gradients):
        torch.testing.assert_close(grads[name], expected, rtol=3e-4, atol=5e-6)
    torch.testing.assert_close(metrics, expected_metrics, rtol=3e-4, atol=5e-6)
    outputs = agent(batch[0])
    previous_outputs = torch.func.functional_call(agent, previous, (batch[0],))
    current_scores = expected_scores(outputs, batch, agent.log_action_scale, args)
    old_scores = expected_scores(previous_outputs, batch, agent.log_action_scale, args)
    delta = tuple(a - b for a, b in zip(current_scores, old_scores))
    if mode != "both":
        delta = (torch.zeros_like(delta[0]), delta[1])
    if mode == "adam":
        delta = tuple(torch.zeros_like(value) for value in delta)
    expected_correction = torch.autograd.grad(outputs, tuple(agent.parameters()), grad_outputs=delta)
    for name, expected in zip(params, expected_correction):
        torch.testing.assert_close(corrections[name], expected, rtol=3e-4, atol=5e-6)
    if mode == "critic":
        assert all(torch.count_nonzero(value) == 0 for name, value in corrections.items() if name.startswith("actor."))
        assert sum(value.square().sum() for name, value in corrections.items() if name.startswith("critic.")) > 0


@pytest.mark.parametrize("mode", ["critic", "both"])
def test_identical_current_and_previous_models_produce_no_transport(mode):
    agent, args, params, _, batch = setup(mode)
    previous = {name: value.clone() for name, value in params.items()}
    _, corrections, _ = transport.make_gradient_function(agent, args)(params, previous, *batch)
    assert all(torch.count_nonzero(value) == 0 for value in corrections.values())


def test_unclipped_critic_transport_cancels_current_target_noise():
    agent, args, params, previous, batch = setup("critic")
    args.clip_vloss = False
    fn = transport.make_gradient_function(agent, args)
    left_g, left_c, _ = fn(params, previous, *batch)
    changed = (*batch[:4], batch[4] + torch.linspace(-3, 4, 8, device="cuda"), batch[5])
    right_g, right_c, _ = fn(params, previous, *changed)
    for name in params:
        torch.testing.assert_close(left_c[name], right_c[name], rtol=3e-4, atol=5e-6)
    assert any(not torch.equal(left_g[name], right_g[name]) for name in params if name.startswith("critic."))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("beta", [0.0, 0.9, 0.99999])
@torch.no_grad()
def test_zero_correction_matches_adam_and_previous_is_owned_preupdate(compiled, beta):
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    values = torch.tensor([[0.2, -0.4], [0.7, 0.1]], device="cuda")
    actual, expected = nn.Parameter(values.clone()), nn.Parameter(values.clone())
    optimizer = transport.TransportAdam({"weight": actual}, lr=0.007, beta1=beta, compile=compiled)
    adam = torch.optim.Adam([expected], lr=0.007, betas=(beta, 0.999), eps=1e-5, fused=False)
    for step in range(1, 9):
        if step == 4:
            optimizer.set_lr(0.002)
            adam.param_groups[0]["lr"] = 0.002
        gradient = torch.sin(values * step) + 0.1 * step
        before = actual.clone()
        expected.grad = gradient.clone()
        adam.step()
        optimizer.step({"weight": gradient}, {"weight": torch.zeros_like(gradient)})
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=8e-6)
        torch.testing.assert_close(optimizer.previous["weight"], before, rtol=0, atol=0)
        assert optimizer.previous["weight"].data_ptr() != actual.data_ptr()


@pytest.mark.parametrize("beta", [0.9, 0.99, 0.99999])
@torch.no_grad()
def test_exact_gradient_changes_transport_only_existing_momentum_mass(beta):
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    param = nn.Parameter(torch.tensor([0.2, -0.3], device="cuda"))
    optimizer = transport.TransportAdam({"weight": param}, lr=0.001, beta1=beta, compile=True)
    previous_g = torch.zeros_like(param)
    expected = param.double().clone()
    variance = torch.zeros_like(expected)
    for step in range(1, 9):
        gradient = torch.tensor([step * 0.03, -0.2 + step * 0.01], device="cuda")
        variance = 0.999 * variance + 0.001 * gradient.double().square()
        expected -= 0.001 * gradient.double() / ((variance / -math.expm1(math.log(0.999) * step)).sqrt() + 1e-5)
        optimizer.step({"weight": gradient}, {"weight": gradient - previous_g})
        torch.testing.assert_close(param, expected.float(), rtol=3e-4, atol=8e-6)
        previous_g = gradient


@pytest.mark.parametrize("mode", ["adam", "critic", "both"])
def test_compiled_gradients_and_optimizer_match_actual_residual_agent(mode):
    from cleanrl import ppo_continuous_action_predictive_transport_v23 as trainer

    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    spaces = SimpleNamespace(single_action_space=gym.spaces.Box(-1, 1, (6,), dtype=np.float32),
                             single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32))
    agent = trainer.Agent(spaces).cuda()
    args = trainer.Args(transport_mode=mode, beta1=0.99)
    params = {name: p.detach() for name, p in agent.named_parameters()}
    optimizer = transport.TransportAdam(dict(agent.named_parameters()), lr=0.001, beta1=0.99, compile=True)
    eager = transport.make_gradient_function(agent, args)
    compiled = torch.compile(eager, fullgraph=True, mode="reduce-overhead")
    obs = torch.randn(8, 17, device="cuda")
    native = torch.rand(8, 6, device="cuda") * 0.8 + 0.1
    batch = (obs, native, torch.randn(8, device="cuda"), torch.randn(8, device="cuda"),
             torch.randn(8, device="cuda"), torch.randn(8, device="cuda"))
    for _ in range(3):
        torch.compiler.cudagraph_mark_step_begin()
        expected_g, expected_c, expected_metrics = eager(params, optimizer.previous, *batch)
        actual_g, actual_c, actual_metrics = compiled(params, optimizer.previous, *batch)
        for name in params:
            torch.testing.assert_close(actual_g[name], expected_g[name], rtol=3e-3, atol=3e-5)
            torch.testing.assert_close(actual_c[name], expected_c[name], rtol=3e-3, atol=3e-5)
        torch.testing.assert_close(actual_metrics, expected_metrics, rtol=3e-3, atol=3e-5)
        before = {name: p.clone() for name, p in params.items()}
        optimizer.step(actual_g, actual_c)
        for name in params:
            torch.testing.assert_close(optimizer.previous[name], before[name], rtol=0, atol=0)
            assert torch.isfinite(params[name]).all()
