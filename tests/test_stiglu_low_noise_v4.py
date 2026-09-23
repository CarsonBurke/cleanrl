"""CUDA contracts for calibrated SiTU Gaussian laws and epoch trust acceptance."""

import copy
import math
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.distributions import Independent, Normal, kl_divergence

from cleanrl import ppo_continuous_action_stiglu_distribution_v2 as previous
from cleanrl import ppo_continuous_action_stiglu_low_noise_v4 as control
from cleanrl.shared.host_graph import make_host_mirror


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(autouse=True)
def deterministic_runtime():
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(71)
            yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def spaces():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        single_action_space=gym.spaces.Box(
            low=np.array([-3.0, 0.5, -0.2], dtype=np.float32),
            high=np.array([1.0, 5.5, 0.8], dtype=np.float32),
        ),
    )


def observations():
    return torch.tensor([[0.2, -0.5, 1.0], [-0.9, 0.4, -0.2], [0.8, 1.2, -0.7]], device="cuda")


def test_defaults_reproduce_v2_gaussian_law_and_seeded_initialization():
    torch.manual_seed(1)
    original = previous.Agent(spaces(), previous.Args(policy="gaussian")).cuda()
    torch.manual_seed(1)
    agent = control.Agent(spaces(), control.Args()).cuda()
    assert control.state_hash(agent.critic) == previous.state_hash(original.critic)
    assert control.state_hash(agent.actor) == previous.state_hash(original.actor)
    x = observations()
    with torch.no_grad():
        torch.testing.assert_close(agent.get_value(x), original.get_value(x), rtol=0, atol=0)
        mean, log_std = agent.policy_parameters(x)
        old_mean, old_log_std = original.policy_parameters(x)
        torch.testing.assert_close(mean, old_mean, rtol=0, atol=0)
        torch.testing.assert_close(log_std, old_log_std, rtol=0, atol=0)
        # Nonzero logits also match: equality cannot hide accidental mean scaling
        # behind the zero-initialized policy head.
        agent.actor[-1].weight.normal_(std=0.08)
        agent.actor[-1].bias.normal_(std=0.05)
        original.actor.load_state_dict(agent.actor.state_dict())
        native = torch.tensor([[0.1, -0.2, 0.8], [1.0, -0.3, 0.2], [-0.8, 0.2, 0.4]], device="cuda")
        mean, log_std = agent.policy_parameters(x)
        old_mean, old_log_std = original.policy_parameters(x)
        torch.testing.assert_close(agent.action_logprob(mean, log_std, native),
                                   original.action_logprob(old_mean, old_log_std, native), rtol=0, atol=0)


@pytest.mark.parametrize("whiten_mean", [False, True])
def test_low_noise_host_mirror_compiled_replay_and_physical_density(whiten_mean):
    initial_std = 1.0 / math.sqrt(6.0)
    agent = control.Agent(spaces(), control.Args(initial_std=initial_std, whiten_mean=whiten_mean)).cuda()
    x = observations()
    with torch.no_grad():
        _, initial_log_std = agent.policy_parameters(x)
        torch.testing.assert_close(initial_log_std.exp(), torch.full_like(initial_log_std, initial_std),
                                   rtol=2e-6, atol=2e-7)
        agent.actor[-1].weight.normal_(std=0.08)
        agent.actor[-1].bias.normal_(std=0.1)
        mirror = make_host_mirror(agent.actor, len(x))
        host_logits = mirror(x.cpu().numpy())
        sampler = control.HostGaussianSampler(agent, len(x))
        native_np, action_np = sampler(host_logits, np.random.default_rng(812))
        native = torch.from_numpy(native_np.copy()).cuda()
        old_mean = torch.from_numpy(sampler.first.copy()).cuda()
        old_log_std = torch.from_numpy(sampler.second.copy()).cuda()
        policy = torch.compile(agent.policy_parameters, fullgraph=True)
        mean, log_std = policy(x)
        torch.testing.assert_close(old_mean, mean, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(old_log_std, log_std, rtol=2e-5, atol=2e-6)
        expected_mean = agent.actor(x)[..., :agent.action_dim] * (initial_std if whiten_mean else 1.0)
        torch.testing.assert_close(mean, expected_mean, rtol=2e-5, atol=2e-6)
        distribution = Independent(Normal(old_mean, old_log_std.exp()), 1)
        expected_logprob = distribution.log_prob(native)
        torch.testing.assert_close(agent.action_logprob(old_mean, old_log_std, native), expected_logprob,
                                   rtol=2e-6, atol=2e-6)
        physical = agent.action_bias + agent.action_scale * native.tanh()
        np.testing.assert_allclose(action_np, physical.cpu().numpy(), rtol=2e-6, atol=2e-7)
        jacobian = (2 * (math.log(2) - torch.logaddexp(native, -native)) + agent.log_action_scale).sum(-1)
        torch.testing.assert_close(agent.physical_logprob(old_mean, old_log_std, native),
                                   expected_logprob - jacobian, rtol=2e-6, atol=2e-6)

        def statistics(obs, samples, means, log_stds):
            return control.rollout_statistics(agent, obs, samples, means, log_stds)

        replay = torch.compile(statistics, fullgraph=True)
        _, scored, drift = replay(x, native, old_mean, old_log_std)
        torch.testing.assert_close(scored, expected_logprob, rtol=2e-6, atol=2e-6)
        assert drift[0] < 1e-8
        assert drift[1] < 2e-5
        # Device replay must not silently replace the captured behavior density.
        agent.actor[-1].bias[:agent.action_dim].add_(0.4)
        _, changed_score, changed_drift = replay(x, native, old_mean, old_log_std)
        torch.testing.assert_close(changed_score, expected_logprob, rtol=2e-6, atol=2e-6)
        assert changed_drift[0] > 0.01
        assert changed_drift[1] > 0.01


def test_gaussian_joint_ratio_and_kl_include_every_action_coordinate():
    agent = control.Agent(spaces(), control.Args(initial_std=1 / math.sqrt(6), whiten_mean=True)).cuda().double()
    mean = torch.tensor([[0.2, -0.4, 0.1], [-0.1, 0.3, 0.2]], device="cuda", dtype=torch.float64)
    log_std = torch.tensor([[-1.0, -0.8, -1.2], [-0.9, -1.1, -0.7]], device="cuda", dtype=torch.float64)
    new_mean, new_log_std = mean + 0.13, log_std + 0.21
    native = torch.tensor([[0.2, -0.4, 25.0], [-21.0, 0.3, 0.1]], device="cuda", dtype=torch.float64)
    old = Independent(Normal(mean, log_std.exp()), 1)
    new = Independent(Normal(new_mean, new_log_std.exp()), 1)
    torch.testing.assert_close(agent.action_logprob(new_mean, new_log_std, native)
                               - agent.action_logprob(mean, log_std, native),
                               new.log_prob(native) - old.log_prob(native), rtol=1e-12, atol=1e-12)
    compiled_kl = torch.compile(agent.joint_kl, fullgraph=True)
    torch.testing.assert_close(compiled_kl(mean, log_std, new_mean, new_log_std),
                               kl_divergence(old, new), rtol=2e-10, atol=1e-10)
    mean_kl, scale_kl = control.gaussian_kl_parts(mean, log_std, new_mean, new_log_std)
    torch.testing.assert_close(mean_kl + scale_kl, kl_divergence(old, new), rtol=1e-12, atol=1e-12)
    jacobian = (2 * (math.log(2) - torch.logaddexp(native, -native)) + agent.log_action_scale).sum(-1)
    torch.testing.assert_close(agent.physical_logprob(mean, log_std, native), old.log_prob(native) - jacobian,
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("proposal", ["accepted", "backtracked", "nonfinite"])
@pytest.mark.parametrize("initialized_adam", [False, True])
def test_epoch_trust_restores_actor_optimizer_without_undoing_critic(proposal, initialized_adam):
    agent = control.Agent(spaces(), control.Args(initial_std=1 / math.sqrt(6), whiten_mean=True)).cuda()
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.003, eps=1e-5, fused=True)
    x = observations()

    def update(model, opt):
        opt.zero_grad(set_to_none=True)
        mean, _, value = model.get_policy_and_value(x)
        loss = (mean - 0.7).square().mean() + (value - 1.2).square().mean()
        loss.backward()
        opt.step()

    if initialized_adam:
        update(agent, optimizer)
    with torch.no_grad():
        old_mean, old_log_std = (value.clone() for value in agent.policy_parameters(x))
    before = control.snapshot_actor_epoch(agent.actor, optimizer)
    old_optimizer = copy.deepcopy(optimizer.state_dict())
    reference = copy.deepcopy(agent)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.003, eps=1e-5, fused=True)
    optimizer.param_groups[0]["lr"] = 0.5 if proposal != "accepted" else 1e-5
    update(agent, optimizer)
    proposed_optimizer = copy.deepcopy(optimizer.state_dict())
    critic_proposed = copy.deepcopy(agent.critic.state_dict())
    if proposal == "nonfinite":
        with torch.no_grad():
            agent.actor[-1].bias.fill_(float("nan"))

    @torch.no_grad()
    def constraint():
        mean, log_std = agent.policy_parameters(x)
        return agent.joint_kl(old_mean, old_log_std, mean, log_std).mean()

    limit = 0.02
    if proposal == "backtracked":
        assert constraint() > limit
    fraction = control.constrain_epoch(agent.actor, optimizer, before, constraint, limit)
    assert torch.isfinite(constraint()) and constraint() <= limit
    if proposal == "accepted":
        assert fraction == 1.0
    elif proposal == "backtracked":
        assert 0.0 < fraction < 1.0
    else:
        assert fraction == 0.0
        for parameter, original in zip(agent.actor.parameters(), before[0]):
            torch.testing.assert_close(parameter, original, rtol=0, atol=0)
    for name, value in agent.critic.state_dict().items():
        torch.testing.assert_close(value, critic_proposed[name], rtol=0, atol=0)

    # A real subsequent Adam step must follow the appropriate actor history.
    # This catches forgotten step counters, retained rejected moments, and
    # accidental creation of state when rejecting the very first optimizer step.
    reference.load_state_dict(agent.state_dict())
    reference_optimizer.load_state_dict(proposed_optimizer if fraction == 1.0 else old_optimizer)
    # The reference critic must always continue from the proposal history.
    proposed_ids = proposed_optimizer["param_groups"][0]["params"]
    parameter_ids = dict(zip(agent.parameters(), proposed_ids))
    for reference_parameter, parameter in zip(reference.critic.parameters(), agent.critic.parameters()):
        reference_optimizer.state[reference_parameter] = copy.deepcopy(proposed_optimizer["state"][parameter_ids[parameter]])
    optimizer.param_groups[0]["lr"] = reference_optimizer.param_groups[0]["lr"] = 0.003
    update(agent, optimizer)
    update(reference, reference_optimizer)
    with torch.no_grad():
        for actual, expected in zip(agent.get_policy_and_value(x), reference.get_policy_and_value(x)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
