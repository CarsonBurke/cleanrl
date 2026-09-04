"""CPU functional tests for the v8 head-split nextlat machinery.

Covers what unit parity cannot: param-group partition (critic head sees only
value grads, actor heads see PG + actor-KL), critic readout/teacher shapes,
and a full 4-way dual-backward + stash-sum + optimizer step executing with
finite grads on both share_backbone settings.
"""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

ROOT = Path(__file__).parents[1]


def _load_v8():
    spec = importlib.util.spec_from_file_location(
        "nextlat_v8_headsplit",
        ROOT / "cleanrl/iterthink/v24_d3bucket/ppo/ppo_continuous_action_iterthink_v24_beta_d3bucket_mtp_ppoadvnorm_batch_nextlat_v8_headsplit.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _DummyEnvs:
    single_observation_space = gym.spaces.Box(-10.0, 10.0, shape=(4,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))


def _tiny_args(module, share_backbone):
    return replace(
        module.Args(),
        hidden=8,
        k_blocks=1,
        n_experts=2,
        num_bins=7,
        critic_mtp_horizon=2,
        actor_dist="beta",
        share_backbone=share_backbone,
    )


def _param_ids(params):
    return {id(p) for p in params}


@pytest.mark.parametrize("share_backbone", [True, False])
def test_param_group_partition(share_backbone):
    torch.manual_seed(0)
    module = _load_v8()
    agent = module.Agent(_DummyEnvs(), _tiny_args(module, share_backbone))
    actor = agent.actor_parameters()
    critic = agent.critic_parameters()
    aux_actor = agent.nextlat_actor_parameters()
    aux_critic = agent.nextlat_critic_parameters()

    trunk = agent.trunk if share_backbone else None
    if share_backbone:
        trunk_ids = _param_ids(trunk.parameters())
        for group in (actor, critic, aux_actor, aux_critic):
            assert trunk_ids <= _param_ids(group), "shared trunk must be in every group"
    else:
        assert _param_ids(agent.actor_trunk.parameters()) <= _param_ids(actor)
        assert _param_ids(agent.actor_trunk.parameters()) <= _param_ids(aux_actor)
        assert _param_ids(agent.critic_trunk.parameters()) <= _param_ids(critic)
        assert _param_ids(agent.critic_trunk.parameters()) <= _param_ids(aux_critic)

    critic_head_ids = _param_ids(agent.critic_head.parameters())
    # Critic head sees only value-CE grads: excluded from the critic aux group.
    assert critic_head_ids <= _param_ids(critic)
    assert critic_head_ids.isdisjoint(_param_ids(aux_critic))
    # Actor heads see PG + actor-KL-distill grads: in both actor groups.
    head_ids = _param_ids(agent.actor_alpha_head.parameters()) | _param_ids(
        agent.actor_beta_head.parameters()
    )
    assert head_ids <= _param_ids(actor)
    assert head_ids <= _param_ids(aux_actor)
    # Dynamics + readout live only in their own aux group (+ trunk).
    dyn_ids = _param_ids(agent.nextlat_predictor.parameters())
    assert dyn_ids <= _param_ids(aux_actor)
    assert dyn_ids.isdisjoint(_param_ids(aux_critic))
    cdyn_ids = _param_ids(agent.nextlat_critic_dyn.parameters()) | _param_ids(
        agent.nextlat_critic_readout.parameters()
    )
    assert cdyn_ids <= _param_ids(aux_critic)
    assert cdyn_ids.isdisjoint(_param_ids(aux_actor))
    # Every parameter is covered by at least one group.
    all_ids = (
        _param_ids(actor) | _param_ids(critic) | _param_ids(aux_actor) | _param_ids(aux_critic)
    )
    assert _param_ids(agent.parameters()) <= all_ids


@pytest.mark.parametrize("share_backbone", [True, False])
def test_four_way_backward_runs_finite(share_backbone):
    torch.manual_seed(1)
    module = _load_v8()
    args = _tiny_args(module, share_backbone)
    agent = module.Agent(_DummyEnvs(), args)
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4)

    mb, act_dim, depth = 16, 2, 2
    obs = torch.randn(mb, 4)
    target_returns = torch.randn(mb)

    def build_losses():
        # Fresh graphs per pass, exactly as each minibatch builds them:
        # rollout sampling is no-grad, then the update pass replays z through
        # a grad-tracked forward (v21 z-replay).
        with torch.no_grad():
            _, z, _, _, _, _ = agent.get_action_and_value(obs)
        z = z.detach()
        action, _, _, _, _, feat = agent.get_action_and_value(obs, z)
        actions = action.detach()
        h_hat = agent.nextlat_predictor(torch.cat([feat, actions], dim=-1))
        hc_hat = agent.nextlat_critic_dyn(torch.cat([feat, actions], dim=-1))
        z_hat = agent.nextlat_critic_readout(hc_hat)
        with torch.no_grad():
            tgt_feat = agent.get_actor_feat(obs)
            t_dist, _, _ = agent._actor_dist(tgt_feat)
            tgt_logits = agent.get_value(obs)[:, 0]
        assert z_hat.shape == tgt_logits.shape == (mb, args.num_bins)
        a_loss = F.smooth_l1_loss(h_hat, tgt_feat).mean() + torch.distributions.kl_divergence(
            t_dist, agent._actor_dist(h_hat)[0]
        ).sum(-1).mean()
        c_loss = F.smooth_l1_loss(z_hat, tgt_logits).mean() + torch.distributions.kl_divergence(
            Categorical(logits=tgt_logits), Categorical(logits=z_hat)
        ).mean()
        v = F.mse_loss(agent.get_value(obs)[:, 0].float().mean(-1), target_returns)
        pg = -agent._actor_dist(feat)[0].log_prob(z.clamp(1e-6, 1 - 1e-6)).sum(-1).mean()
        return a_loss, c_loss, v, pg

    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()
    aux_actor_params = agent.nextlat_actor_parameters()
    aux_critic_params = agent.nextlat_critic_parameters()
    trunk_params = list(
        (agent.trunk if share_backbone else agent.actor_trunk).parameters()
    )

    # Two passes: the file zero-inits the critic head, so value-path trunk
    # grads are exactly 0 on the first pass; the second pass is steady state.
    for steady in (False, True):
        actor_loss, critic_loss, v_loss, pg_loss = build_losses()
        for loss in (actor_loss, critic_loss):
            assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        (0.5 * v_loss).backward(retain_graph=True)
        nn.utils.clip_grad_norm_(critic_params, 0.25)
        stashed = [(p, p.grad.detach().clone()) for p in critic_params if p.grad is not None]
        opt.zero_grad(set_to_none=True)
        actor_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(aux_actor_params, 0.25)
        for p in aux_actor_params:
            if p.grad is not None:
                stashed.append((p, p.grad.detach().clone()))
        opt.zero_grad(set_to_none=True)
        critic_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(aux_critic_params, 0.25)
        for p in aux_critic_params:
            if p.grad is not None:
                stashed.append((p, p.grad.detach().clone()))
        opt.zero_grad(set_to_none=True)
        pg_loss.backward()
        nn.utils.clip_grad_norm_(actor_params, 0.25)
        for p, g in stashed:
            p.grad = g if p.grad is None else p.grad + g
        for p in agent.parameters():
            assert p.grad is None or torch.isfinite(p.grad).all()
        if steady:
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in trunk_params
            )
        opt.step()
