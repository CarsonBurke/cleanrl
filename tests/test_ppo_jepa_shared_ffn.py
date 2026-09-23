"""CUDA regressions for a shared JEPA encoder with full detached PPO FFNs; use mlq."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action as baseline
from cleanrl import ppo_continuous_action_jepa_shared_ffn_v4 as model
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(autouse=True)
def cuda_runtime():
    assert torch.cuda.is_available(), "Run CUDA contracts through mlq"
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)


@pytest.fixture
def envs():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float32),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
    )


def make_agent(envs, mode="both"):
    return model.Agent(envs, mode, sigreg_num_proj=16, sigreg_proj_chunk=8).cuda()


def batch(agent):
    observations = torch.randn(32, 17, device="cuda")
    actions = torch.rand(32, 6, device="cuda") * 0.8 + 0.1
    with torch.no_grad():
        alpha, beta, value = agent.get_policy_and_value(observations)
        logprobs = agent.action_logprob(alpha, beta, actions)
    return (observations, actions, logprobs, torch.randn(32, device="cuda"),
            value.flatten() + torch.randn(32, device="cuda"), value.flatten())


@pytest.mark.parametrize("mode", ["actor", "critic", "both", "none"])
def test_ppo_trains_every_ffn_layer_without_changing_representation(envs, mode):
    agent = make_agent(envs, mode)
    policy, representation = agent.parameter_groups()
    policy_ids, representation_ids = {id(p) for p in policy}, {id(p) for p in representation}
    assert policy_ids.isdisjoint(representation_ids)
    assert policy_ids | representation_ids == {id(p) for p in agent.parameters()}
    assert len(policy_ids) == len(policy) and len(representation_ids) == len(representation)
    before_representation = [p.detach().clone() for p in representation]
    before_policy = [p.detach().clone() for p in policy]
    data = batch(agent)
    loss, _, _ = model.ppo_loss(agent, *data, model.Args(jepa_mode=mode), torch.randn_like(data[0]))
    loss.backward()
    for net in (agent.actor, agent.critic):
        for layer in (0, 2, 4):
            assert net[layer].weight.grad.norm() > 0
    assert all(p.grad is None or not bool(p.grad.any()) for p in representation)
    torch.nn.utils.clip_grad_norm_(policy, 0.5)
    torch.optim.Adam(policy, lr=3e-4, eps=1e-5, fused=True).step()
    for actual, previous in zip(representation, before_representation):
        torch.testing.assert_close(actual, previous, rtol=0, atol=0)
    for actual, previous in zip(policy, before_policy):
        assert not torch.equal(actual, previous)
    agent.zero_grad(set_to_none=True)
    agent.get_value(data[0]).sum().backward()
    for layer in (0, 2, 4):
        assert agent.critic[layer].weight.grad.norm() > 0
    if agent.encoder is not None:
        assert all(p.grad is None for p in agent.encoder.parameters())


@pytest.mark.parametrize("mode", ["actor", "critic", "both"])
def test_ssl_trains_both_temporal_encodings_without_changing_ppo_ffns(envs, mode):
    agent = make_agent(envs, mode)
    current = torch.randn(32, 17, device="cuda", requires_grad=True)
    following = torch.randn(32, 17, device="cuda", requires_grad=True)
    actions = torch.rand(32, 6, device="cuda")
    prediction, regularization, _ = agent.ssl_losses(agent.encoder(current), following, actions)
    (prediction + 0.09 * regularization).backward()
    assert current.grad.norm() > 0 and following.grad.norm() > 0
    for layer in (0, 2):
        assert agent.encoder[layer].weight.grad.norm() > 0
    policy, representation = agent.parameter_groups()
    assert all(p.grad is None for p in policy)
    before_policy = [p.detach().clone() for p in policy]
    before_encoder = [p.detach().clone() for p in agent.encoder.parameters()]
    torch.nn.utils.clip_grad_norm_(representation, 0.5)
    torch.optim.AdamW(representation, lr=5e-5, weight_decay=1e-3, fused=True).step()
    for actual, previous in zip(policy, before_policy):
        torch.testing.assert_close(actual, previous, rtol=0, atol=0)
    for actual, previous in zip(agent.encoder.parameters(), before_encoder):
        assert not torch.equal(actual, previous)


def test_both_uses_one_ssl_objective_not_one_per_consumer(envs):
    actor_only, both = make_agent(envs, "actor"), make_agent(envs, "both")
    both.encoder.load_state_dict(actor_only.encoder.state_dict())
    both.ssl.load_state_dict(actor_only.ssl.state_dict())
    data = batch(actor_only)
    following = torch.randn_like(data[0])
    losses, gradients = [], []
    for agent, mode in ((actor_only, "actor"), (both, "both")):
        torch.manual_seed(9)
        _, ssl_loss, _ = model.ppo_loss(agent, *data, model.Args(jepa_mode=mode), following)
        ssl_loss.backward()
        losses.append(ssl_loss.detach())
        gradients.append([p.grad.clone() for p in agent.encoder.parameters()])
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    for left, right in zip(*gradients):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["actor", "critic", "both", "none"])
def test_host_rollout_matches_cuda_policy_after_encoder_and_ffn_updates(envs, mode):
    agent = make_agent(envs, mode)
    mirror = make_host_mirror(agent.rollout_actor(), 32)
    observations = torch.randn(32, 17, device="cuda")
    host_observations = observations.cpu().numpy()
    for update in (False, True):
        if update:
            with torch.no_grad():
                agent.actor[0].weight.add_(0.01)
                if agent.encoder is not None:
                    agent.encoder[0].weight.add_(0.02)
            mirror.refresh()
        with torch.no_grad():
            alpha, beta, value = agent.get_policy_and_value(observations)
            host_logits = torch.from_numpy(mirror(host_observations).copy()).cuda()
            host_alpha, host_beta = (torch.nn.functional.softplus(host_logits) + 1).chunk(2, dim=-1)
            torch.testing.assert_close(host_alpha, alpha, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(host_beta, beta, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(agent.get_value(observations), value, rtol=0, atol=0)


def test_none_preserves_baseline_loss_and_ppo_update(envs):
    torch.manual_seed(1)
    reference = baseline.Agent(envs).cuda()
    torch.manual_seed(1)
    agent = make_agent(envs, "none")
    data = batch(agent)
    expected, expected_metrics = baseline.ppo_loss(reference, *data, baseline.Args())
    policy_loss, ssl_loss, metrics = model.ppo_loss(agent, *data, model.Args(jepa_mode="none"))
    torch.testing.assert_close(policy_loss, expected, rtol=0, atol=0)
    assert ssl_loss == 0
    torch.testing.assert_close(metrics, expected_metrics, rtol=0, atol=0)
    policy_loss.backward()
    expected.backward()
    for net in (reference, agent):
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5, fused=True).step()
    for actual, target in zip(agent.get_policy_and_value(data[0]), reference.get_policy_and_value(data[0])):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)


def test_compiled_shared_objective_preserves_ffn_gradients(envs):
    agent = make_agent(envs)
    data = batch(agent)
    following = torch.randn_like(data[0])
    args = model.Args()

    def objective(*inputs):
        return model.ppo_loss(agent, *inputs, args, following)

    reference_policy, _, _ = objective(*data)
    reference_policy.backward()
    policy, _ = agent.parameter_groups()
    expected_gradients = [p.grad.clone() for p in policy]
    agent.zero_grad(set_to_none=True)
    compiled = torch.compile(objective, mode="reduce-overhead", fullgraph=True)
    policy_loss, ssl_loss, _ = compiled(*data)
    (policy_loss + ssl_loss).backward()
    torch.testing.assert_close(policy_loss, reference_policy, rtol=2e-5, atol=2e-6)
    for parameter, expected in zip(policy, expected_gradients):
        torch.testing.assert_close(parameter.grad, expected, rtol=3e-4, atol=3e-6)
    for layer in (0, 2):
        assert agent.encoder[layer].weight.grad.norm() > 0

    def components(*inputs):
        return model.loss_components(agent, *inputs, args, following)[0]

    diagnostic = torch.compile(components, fullgraph=True, options={"triton.cudagraphs": False})
    before = [p.grad.clone() for p in agent.parameters()]
    measured = diagnostic(*data)
    balance = model.gradient_balance(agent, measured)
    for parameter, previous in zip(agent.parameters(), before):
        torch.testing.assert_close(parameter.grad, previous, rtol=0, atol=0)
    assert balance["balance/shared_prediction_grad_norm"] > 0
    assert balance["balance/shared_sigreg_grad_norm"] > 0
