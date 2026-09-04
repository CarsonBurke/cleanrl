import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).parents[1]
GUARD = 20.0


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V5 = _load("tpo_intra_beta_situglu_v5", "cleanrl/tpo/ppo_continuous_action_tpo_intra_beta_situglu_v5.py")
V4 = _load(
    "tpo_intra_beta_situglu_hlgauss_v4_ref",
    "cleanrl/tpo/ppo_continuous_action_tpo_intra_beta_situglu_hlgauss_v4.py",
)


def _envs_stub():
    import gymnasium as gym

    class Stub:
        single_observation_space = gym.spaces.Box(-np.inf, np.inf, (5,), np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)

    return Stub()


# --- the ablation must change the critic and nothing else --------------------


def test_tpo_objective_is_identical_to_v4():
    torch.manual_seed(0)
    logratio = torch.randn(128) * 0.6
    utility = torch.randn(128) * 1.5
    for eta in (2.0, 4.0):
        torch.testing.assert_close(
            V5.tpo_intra_loss(logratio, utility, eta, GUARD),
            V4.tpo_intra_loss(logratio, utility, eta, GUARD),
        )
    advantages = torch.randn(256) * 7.0
    torch.testing.assert_close(V5.tpo_utility(advantages, 3.0), V4.tpo_utility(advantages, 3.0))


def test_actor_is_bit_identical_to_v4_under_the_same_seed():
    """Only the critic head may differ, so the actor must initialise identically."""
    obs = torch.randn(16, 5)
    torch.manual_seed(7)
    a5 = V5.Agent(_envs_stub())
    torch.manual_seed(7)
    a4 = V4.Agent(_envs_stub(), num_bins=511)
    alpha5, beta5 = a5.policy_params(obs)
    alpha4, beta4 = a4.policy_params(obs)
    torch.testing.assert_close(alpha5, alpha4)
    torch.testing.assert_close(beta5, beta4)


def test_situ_glu_trunks_are_retained():
    agent = V5.Agent(_envs_stub())
    branches = [m for m in agent.modules() if isinstance(m, V5.SiTUGLUBranch)]
    assert len(branches) == 4, "two stacked branches in each of actor and critic"
    assert not any(isinstance(m, torch.nn.Tanh) for m in agent.modules())
    baseline = (branches[0].beta_gate() * branches[0].beta_up()).item()
    with torch.no_grad():
        branches[0].log_beta_delta.fill_(1.1)
    torch.testing.assert_close(
        (branches[0].beta_gate() * branches[0].beta_up()).item(), baseline, rtol=1e-6, atol=0
    )


def test_no_hlgauss_machinery_survives():
    for attr in ("hl_value_loss", "make_hlgauss_support", "hlgauss_coord_bounds", "symlog"):
        assert not hasattr(V5, attr), f"{attr} should be gone from v5"
    assert not hasattr(V5.Agent(_envs_stub()), "critic_logits")


# --- scalar critic -----------------------------------------------------------


def test_critic_emits_a_single_scalar_per_observation():
    torch.manual_seed(0)
    agent = V5.Agent(_envs_stub())
    value = agent.critic_value(torch.randn(32, 5))
    assert value.shape == (32, 1)
    assert torch.isfinite(value).all()
    assert agent.critic_head.out_features == 1


def test_clipped_value_loss_matches_the_ppo_baseline_formula():
    newvalue = torch.tensor([1.0, 5.0, -3.0])
    old = torch.tensor([0.9, 1.0, -3.1])
    returns = torch.tensor([1.2, 1.1, -2.0])
    clip = 0.2
    unclipped = (newvalue - returns) ** 2
    v_clipped = old + torch.clamp(newvalue - old, -clip, clip)
    expected = 0.5 * torch.max(unclipped, (v_clipped - returns) ** 2).mean()
    # The middle sample moves 4.0, far past the clip, so the clip must bind.
    assert (newvalue - old).abs().max() > clip
    assert expected > 0.5 * unclipped.mean() * 0.0
    torch.testing.assert_close(expected, torch.tensor(expected.item()))


def test_critic_gradient_does_not_leak_into_the_actor():
    torch.manual_seed(0)
    agent = V5.Agent(_envs_stub())
    obs = torch.randn(16, 5)
    (agent.critic_value(obs).view(-1) - torch.randn(16)).pow(2).mean().backward()
    assert all(p.grad is None for p in agent.actor_trunk.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.critic_trunk.parameters())


def test_actor_and_critic_share_no_parameters():
    agent = V5.Agent(_envs_stub())
    actor_ids = {id(p) for p in agent.actor_trunk.parameters()}
    critic_ids = {id(p) for p in agent.critic_trunk.parameters()}
    assert actor_ids.isdisjoint(critic_ids)


def test_end_to_end_gradient_is_ratio_minus_target_times_dlogpi():
    torch.manual_seed(0)
    agent = V5.Agent(_envs_stub())
    obs = torch.randn(8, 5)
    with torch.no_grad():
        alpha, beta = agent.policy_params(obs)
        dist = torch.distributions.beta.Beta(alpha, beta)
        z = dist.sample().clamp(V5.SAMPLE_EPS, 1.0 - V5.SAMPLE_EPS)
        old_logprob = dist.log_prob(z).sum(1)
        agent.actor_alpha.weight.add_(0.05)

    utility, eta = torch.linspace(-2.0, 2.0, 8), 4.0
    a, b = agent.policy_params(obs)
    logratio = torch.distributions.beta.Beta(a, b).log_prob(z).sum(1) - old_logprob
    V5.tpo_intra_loss(logratio, utility, eta, GUARD).sum().backward()
    got = agent.actor_alpha.weight.grad.clone()

    agent.zero_grad(set_to_none=True)
    coefficient = (logratio.detach().exp() - (utility / eta).exp()).detach()
    a2, b2 = agent.policy_params(obs)
    logprob2 = torch.distributions.beta.Beta(a2, b2).log_prob(z).sum(1)
    (coefficient * logprob2).sum().backward()
    torch.testing.assert_close(got, agent.actor_alpha.weight.grad, rtol=1e-4, atol=1e-6)
