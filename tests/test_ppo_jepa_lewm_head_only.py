"""CUDA contracts for fixed-capacity LeWM backbones and detached PPO heads; use mlq."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl import ppo_continuous_action as baseline
from cleanrl import ppo_continuous_action_jepa_lewm_head_only_v3 as lewm
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
    return lewm.Agent(envs, mode, sigreg_num_proj=16, sigreg_proj_chunk=8).cuda()


def batch(agent):
    observations = torch.randn(32, 17, device="cuda")
    actions = torch.rand(32, 6, device="cuda") * 0.8 + 0.1
    with torch.no_grad():
        alpha, beta, value = agent.get_policy_and_value(observations)
        logprobs = agent.action_logprob(alpha, beta, actions)
    return (observations, actions, logprobs, torch.randn(32, device="cuda"),
            value.flatten() + torch.randn(32, device="cuda"), value.flatten())


def nonzero(parameter):
    return parameter.grad is not None and bool(parameter.grad.norm() > 0)


@pytest.mark.parametrize("mode", ["actor", "critic", "both", "none"])
def test_backbone_budget_and_initial_policy_are_identical_to_ppo(envs, mode):
    torch.manual_seed(1)
    reference = baseline.Agent(envs).cuda()
    torch.manual_seed(1)
    agent = make_agent(envs, mode)
    observations = torch.randn(32, 17, device="cuda")
    for actual, expected in zip(agent.get_policy_and_value(observations), reference.get_policy_and_value(observations)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    counts = agent.parameter_counts()
    for name in ("actor", "critic"):
        reference_trunk = getattr(reference, name)
        assert counts[name + "_backbone"] == sum(p.numel() for p in reference_trunk[:4].parameters())
        assert counts[name + "_head"] == sum(p.numel() for p in reference_trunk[4].parameters())
    assert counts["actor_backbone"] == counts["critic_backbone"] == 5312
    assert counts["inference"] == sum(p.numel() for p in reference.parameters()) == 11469
    if mode == "none":
        assert counts["training_only"] == 0
    policy, representation = agent.parameter_groups()
    policy_ids, representation_ids = {id(p) for p in policy}, {id(p) for p in representation}
    assert policy_ids.isdisjoint(representation_ids)
    assert policy_ids | representation_ids == {id(p) for p in agent.parameters()}
    assert len(policy_ids) == len(policy) and len(representation_ids) == len(representation)


@pytest.mark.parametrize("mode", ["actor", "critic", "both", "none"])
def test_ppo_updates_only_heads_of_selected_backbones(envs, mode):
    agent = make_agent(envs, mode)
    data = batch(agent)
    components, _ = lewm.loss_components(agent, *data, lewm.Args(jepa_mode=mode), torch.randn_like(data[0]))
    (components[0] + components[1]).backward()
    for name in ("actor", "critic"):
        trunk = getattr(agent, name)
        selected = mode in (name, "both")
        for layer in (0, 2):
            if selected:
                assert trunk[layer].weight.grad is None or not nonzero(trunk[layer].weight)
            else:
                assert nonzero(trunk[layer].weight)
        assert nonzero(trunk[4].weight)
        branch = getattr(agent, name + "_ssl")
        if branch is not None:
            assert all(p.grad is None or not nonzero(p) for p in branch.parameters())
    # The standalone value API must enforce the same boundary, not bypass it.
    agent.zero_grad(set_to_none=True)
    agent.get_value(data[0]).sum().backward()
    if mode in ("critic", "both"):
        assert agent.critic[0].weight.grad is None
    else:
        assert nonzero(agent.critic[0].weight)
    assert nonzero(agent.critic[4].weight)


@pytest.mark.parametrize("mode", ["actor", "critic", "both"])
def test_ssl_updates_both_temporal_encodings_but_neither_ppo_head(envs, mode):
    agent = make_agent(envs, mode)
    current = torch.randn(32, 17, device="cuda", requires_grad=True)
    following = torch.randn(32, 17, device="cuda", requires_grad=True)
    native = torch.rand(32, 6, device="cuda")
    _, _, _, actor_latent, critic_latent = agent.get_policy_value_latents(current)
    actor_pred, actor_sigreg, critic_pred, critic_sigreg, _ = agent.ssl_losses(
        actor_latent, critic_latent, following, native,
    )
    (actor_pred + critic_pred + 0.09 * (actor_sigreg + critic_sigreg)).backward()
    assert current.grad.norm() > 0 and following.grad.norm() > 0
    for name in ("actor", "critic"):
        trunk = getattr(agent, name)
        for layer in (0, 2):
            if mode in (name, "both"):
                assert nonzero(trunk[layer].weight)
            else:
                assert trunk[layer].weight.grad is None
        assert trunk[4].weight.grad is None
    policy, representation = agent.parameter_groups()
    policy_before = [p.detach().clone() for p in policy]
    optimizer = torch.optim.AdamW(representation, lr=5e-5, weight_decay=1e-3, fused=True)
    torch.nn.utils.clip_grad_norm_(representation, 0.5)
    optimizer.step()
    for parameter, previous in zip(policy, policy_before):
        torch.testing.assert_close(parameter, previous, rtol=0, atol=0)


def test_sigreg_matches_lewm_statistic_and_temporal_batch_axes():
    regularizer = lewm.SIGReg(knots=17, num_proj=31, proj_chunk=7).cuda()
    embeddings = torch.randn(2, 64, 64, device="cuda", requires_grad=True)
    state = torch.cuda.get_rng_state()
    actual = regularizer(embeddings)
    torch.cuda.set_rng_state(state)
    directions = torch.randn(64, 31, device="cuda")
    directions = directions / directions.norm(dim=0)
    frequencies = (embeddings @ directions).unsqueeze(-1) * regularizer.t
    error = (frequencies.cos().mean(dim=1) - regularizer.phi).square() + frequencies.sin().mean(dim=1).square()
    expected = ((error @ regularizer.weights) * 64).mean()
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual, embeddings, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, embeddings)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-5, atol=3e-6)
    with torch.no_grad():
        healthy = torch.randn(2, 1024, 64, device="cuda")
        collapsed = torch.zeros_like(healthy)
        assert regularizer(healthy) < regularizer(collapsed) * 0.05


def test_none_preserves_baseline_loss_gradients_and_adam_update(envs):
    torch.manual_seed(1)
    reference = baseline.Agent(envs).cuda()
    torch.manual_seed(1)
    agent = make_agent(envs, "none")
    data = batch(agent)
    expected, expected_metrics = baseline.ppo_loss(reference, *data, baseline.Args())
    policy_loss, ssl_loss, metrics = lewm.ppo_loss(agent, *data, lewm.Args(jepa_mode="none"))
    torch.testing.assert_close(policy_loss, expected, rtol=0, atol=0)
    assert ssl_loss == 0
    torch.testing.assert_close(metrics, expected_metrics, rtol=0, atol=0)
    policy_loss.backward()
    expected.backward()
    for model in (reference, agent):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5, fused=True).step()
    for actual, target in zip(agent.get_policy_and_value(data[0]), reference.get_policy_and_value(data[0])):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["actor", "critic", "both", "none"])
def test_compiled_stochastic_objective_preserves_gradient_ownership(envs, mode):
    agent = make_agent(envs, mode)
    data = batch(agent)
    following = torch.randn_like(data[0])
    args = lewm.Args(jepa_mode=mode)

    def objective(*inputs):
        return lewm.ppo_loss(agent, *inputs, args, following)

    reference_policy, _, _ = objective(*data)
    reference_policy.backward()
    expected_head_gradients = [agent.actor[4].weight.grad.clone(), agent.critic[4].weight.grad.clone()]
    agent.zero_grad(set_to_none=True)
    compiled = torch.compile(objective, mode="reduce-overhead", fullgraph=True)
    policy_loss, ssl_loss, _ = compiled(*data)
    (policy_loss + ssl_loss).backward()
    # Random SIGReg directions may differ, but cannot change supervised head gradients.
    torch.testing.assert_close(policy_loss, reference_policy, rtol=2e-5, atol=2e-6)
    for name, expected in zip(("actor", "critic"), expected_head_gradients):
        torch.testing.assert_close(getattr(agent, name)[4].weight.grad, expected, rtol=3e-4, atol=3e-6)
    for name in ("actor", "critic"):
        for layer in (0, 2):
            assert nonzero(getattr(agent, name)[layer].weight)
    if mode != "none":
        def components(*inputs):
            return lewm.loss_components(agent, *inputs, args, following)[0]
        diagnostic = torch.compile(components, fullgraph=True, options={"triton.cudagraphs": False})
        before = [parameter.grad.clone() for parameter in agent.parameters()]
        measured = diagnostic(*data)
        balance = lewm.gradient_balance(agent, measured)
        for parameter, previous in zip(agent.parameters(), before):
            torch.testing.assert_close(parameter.grad, previous, rtol=0, atol=0)
        for name in ("actor", "critic"):
            if mode in (name, "both"):
                assert balance[f"balance/{name}_prediction_grad_norm"] > 0
                assert balance[f"balance/{name}_sigreg_grad_norm"] > 0
