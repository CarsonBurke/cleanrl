"""Variational actor (mdl v9): v6 untouched without it, the dual temperature, the reverse-KL prior term."""

import math
import types

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Beta

from cleanrl.mdl import ppo_continuous_action_mdl_critic_predictive_v6 as v6
from cleanrl.mdl import ppo_continuous_action_mdl_variational_actor_v9 as v9

OBS, ACT, ROWS = 5, 3, 64


def envs():
    return types.SimpleNamespace(
        single_action_space=gym.spaces.Box(-1.0, 1.0, (ACT,), np.float32),
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (OBS,), np.float32),
    )


def make_agent(module, posterior="none"):
    torch.manual_seed(0)
    agent = module.Agent(envs(), posterior, init_log_sigma2=-12.0)
    agent.normalize_matrices()
    return agent


def args_for(module, posterior="none", actor_code="none"):
    args = module.Args(norm_adv=False, clip_vloss=False, clip_coef_upper=0.2, posterior=posterior)
    if actor_code != "none":
        args.actor_code = actor_code
    args.batch_size = ROWS
    return args


def data():
    torch.manual_seed(1)
    obs = torch.randn(ROWS, OBS)
    native = torch.rand(ROWS, ACT).clamp(1e-3, 1 - 1e-3)
    return obs, native, torch.randn(ROWS), torch.randn(ROWS), torch.randn(ROWS), torch.zeros(ROWS)


def reference(agent, obs):
    with torch.no_grad():
        return (F.softplus(agent.actor(obs)) + 1.0).chunk(2, dim=-1)


def perturb_actor(agent, scale, seed=7):
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.add_(scale * torch.randn_like(parameter))


@pytest.mark.parametrize("posterior", ["none", "mdl"])
def test_without_the_prior_term_the_loss_is_v6(posterior):
    old, new = make_agent(v6, posterior), make_agent(v9, posterior)
    new.load_state_dict(old.state_dict())
    kl_weight = torch.tensor(0.3)
    old_alpha, old_beta = reference(new, data()[0])
    torch.manual_seed(2)
    want = v6.ppo_loss(old, *data(), args_for(v6, posterior), kl_weight)
    torch.manual_seed(2)
    got = v9.ppo_loss(new, *data(), args_for(v9, posterior), kl_weight, torch.tensor(5.0), old_alpha, old_beta)
    torch.testing.assert_close(got[0], want[0], rtol=0, atol=0)
    torch.testing.assert_close(got[1][:6], want[1], rtol=0, atol=0)
    assert float(got[1][6]) == 0.0


def test_prior_term_is_eta_times_the_reverse_kl():
    agent = make_agent(v9)
    obs = data()[0]
    old_alpha, old_beta = reference(agent, obs)
    perturb_actor(agent, 0.05)
    eta = torch.tensor(0.7)
    base = v9.ppo_loss(agent, *data(), args_for(v9), None, eta, old_alpha, old_beta)[0]
    loss, metrics = v9.ppo_loss(agent, *data(), args_for(v9, actor_code="variational"), None, eta, old_alpha, old_beta)
    alpha, beta = (F.softplus(agent.actor(obs)) + 1.0).chunk(2, dim=-1)
    want = torch.distributions.kl_divergence(Beta(alpha, beta), Beta(old_alpha, old_beta)).sum(-1).mean().detach()
    assert float(want) > 0
    torch.testing.assert_close(loss - base, eta * want, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(metrics[6], want, rtol=1e-5, atol=0)


def test_prior_term_is_zero_with_zero_gradient_at_the_previous_policy():
    agent = make_agent(v9)
    obs = data()[0]
    kl = v9.actor_kl(agent, obs, *reference(agent, obs))
    grads = torch.autograd.grad(kl, list(agent.actor.parameters()))
    assert float(kl.detach().abs()) < 1e-6
    assert max(float(g.abs().max()) for g in grads) < 1e-5


def test_beta_kl_matches_torch_both_ways():
    torch.manual_seed(3)
    a, b, c, d = (1.0 + 20.0 * torch.rand(256, dtype=torch.float64) for _ in range(4))
    torch.testing.assert_close(v9.beta_kl(a, b, c, d), torch.distributions.kl_divergence(Beta(a, b), Beta(c, d)),
                               rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(v9.beta_kl(c, d, a, b), torch.distributions.kl_divergence(Beta(c, d), Beta(a, b)),
                               rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("epsilon", [0.01, 0.1, 1.0])
def test_dual_temperature_meets_the_budget_and_minimises_the_dual(epsilon):
    torch.manual_seed(4)
    advantages = torch.randn(4096).pow(3) + 0.5  # skewed, heavy-tailed, not centred
    eta, kl, ess = v9.temperature_dual(advantages, epsilon, 64)
    weights = torch.softmax(advantages.double() / eta.double(), 0)
    torch.testing.assert_close((weights * (weights * advantages.numel()).log()).sum().float(),
                               torch.tensor(epsilon), rtol=1e-4, atol=0)
    torch.testing.assert_close(kl, torch.tensor(epsilon), rtol=1e-4, atol=0)
    assert 0 < float(ess) < 1

    def dual(value):
        a = advantages.double()
        return value * epsilon + value * (torch.logsumexp(a / value, 0) - math.log(a.numel()))
    centre = eta.double()
    assert dual(centre) <= dual(centre * 1.01) and dual(centre) <= dual(centre * 0.99)


def test_dual_temperature_is_in_advantage_units():
    torch.manual_seed(5)
    advantages = torch.randn(2048)
    eta, *_ = v9.temperature_dual(advantages, 0.05, 64)
    scaled, *_ = v9.temperature_dual(7.0 * advantages + 3.0, 0.05, 64)
    torch.testing.assert_close(scaled, 7.0 * eta, rtol=1e-4, atol=0)
    # Gaussian advantages: the tilt's KL is 1/2 var(A)/eta^2, so eta = std / sqrt(2 epsilon).
    torch.testing.assert_close(eta, advantages.std() / math.sqrt(0.1), rtol=0.05, atol=0)


def test_compiled_loss_sees_in_place_eta():
    torch.distributions.kl._KL_MEMOIZE.clear()
    torch._dynamo.reset()
    agent = make_agent(v9)
    args = args_for(v9, actor_code="variational")
    eta = torch.tensor(0.5)
    obs = data()[0]
    compiled = torch.compile(lambda *inputs: v9.ppo_loss(agent, *inputs[:6], args, None, *inputs[6:])[0],
                             fullgraph=True, dynamic=False)
    for step in range(3):
        old_alpha, old_beta = reference(agent, obs)
        perturb_actor(agent, 0.02, seed=step)
        eta.fill_(0.5 * (step + 1))
        got = compiled(*data(), eta, old_alpha, old_beta)
        want = v9.ppo_loss(agent, *data(), args, None, eta, old_alpha, old_beta)[0]
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_telemetry_leaves_state_alone_and_reports_the_update():
    agent = make_agent(v9)
    obs, native, logprobs, advantages = data()[:4]
    old_alpha, old_beta = reference(agent, obs)
    perturb_actor(agent, 0.05)
    state = torch.get_rng_state()
    with torch.no_grad():
        out = v9.actor_telemetry(agent, obs, native, logprobs, advantages, old_alpha, old_beta,
                                 torch.tensor(0.5), args_for(v9, actor_code="variational"))
    assert torch.equal(torch.get_rng_state(), state)
    assert all(p.grad is None for p in agent.parameters())
    assert all(torch.isfinite(v).all() for v in out.values())
    torch.testing.assert_close(out["actor/update_kl"], v9.actor_kl(agent, obs, old_alpha, old_beta).detach())
    prior = torch.autograd.grad(v9.actor_kl(agent, obs, old_alpha, old_beta), list(agent.actor.parameters()))
    norm = torch.linalg.vector_norm(torch.cat([g.flatten() for g in prior]))
    torch.testing.assert_close(out["grad/actor_prior_norm"], 0.5 * norm, rtol=1e-5, atol=0)


def test_validate_args():
    for kwargs in ({"actor_code": "predictive"}, {"actor_code": "variational", "actor_epsilon": 0.0},
                   {"actor_code": "variational", "actor_epsilon": 100.0},
                   {"actor_code": "variational", "dual_iterations": 0},
                   {"actor_code": "variational", "norm_adv": True}):
        with pytest.raises(ValueError):
            v9.validate_args(v9.Args(**{"norm_adv": False, **kwargs}))
    v9.validate_args(v9.Args(actor_code="variational", norm_adv=False))
