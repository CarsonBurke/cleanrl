# PPO + Morphogenic Compute v22.
#
# METHOD. Channelized Property Flow. v20 showed that backward-only flow helps
# early but broad softmax credit spreads learning into noisy channels. v22 keeps
# v18's sparse forward behavior and gives gradients only through a small top-k
# set of plausible off-support routes/readout cells. Forward values stay sparse;
# backward credit is channelized rather than global.
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter

import cleanrl.morph.compute.ppo_continuous_action_morphcompute_v18 as v18mod
from cleanrl.morph.compute.ppo_continuous_action_morphcompute_v18 import (
    Agent as V18Agent,
    Args as V18Args,
    PropertyReadout,
    effective_support,
    entmax15,
    layer_init,
    make_env,
    mean_stat,
    signed_loss_with_safe_compute,
)

CHANNEL_FLOW_CREDIT = 0.01
CHANNEL_FLOW_TOPK = 4


def channelized_softmax(logits, topk, dim=-1, exclude_mask=None):
    if topk <= 0:
        return torch.zeros_like(logits)
    eligible_logits = logits if exclude_mask is None else logits.masked_fill(exclude_mask, -1e9)
    if topk >= logits.shape[dim]:
        return torch.softmax(eligible_logits, dim=dim)
    top_values, top_indices = torch.topk(eligible_logits, k=topk, dim=dim)
    mask = torch.zeros_like(logits, dtype=torch.bool).scatter(dim, top_indices, True)
    masked_logits = logits.masked_fill(~mask, -1e9)
    return torch.softmax(masked_logits, dim=dim)


def channel_entmax_route_with_floor(logits, dense_floor, alpha, dim=-1):
    if alpha != 1.5:
        raise ValueError("v22 only supports static entmax_alpha=1.5")
    sparse_route = entmax15(logits, dim=dim)
    soft_route = channelized_softmax(logits, CHANNEL_FLOW_TOPK, dim=dim, exclude_mask=sparse_route.detach() > 1e-6)
    route = sparse_route + CHANNEL_FLOW_CREDIT * (soft_route - soft_route.detach())
    if dense_floor <= 0.0:
        return route, sparse_route
    dense_route = torch.softmax(logits, dim=dim)
    route = (1.0 - dense_floor) * route + dense_floor * dense_route
    return route, sparse_route


v18mod.entmax_route_with_floor = channel_entmax_route_with_floor


@dataclass
class Args(V18Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    compute_coef: float = 0.05
    """keep v18's compute discipline; v22 changes credit routing, not cost structure"""
    prop_io_dense_floor: float = 0.0
    """property IO stays forward-sparse"""
    route_flow_credit_floor: float = 0.01
    """backward-only channelized route credit coefficient"""
    route_flow_topk: int = 4
    """top-k off-support route channels that receive backward credit"""
    bridge_coef: float = 0.03
    """zero-forward property-neighbor bridge coefficient"""
    bridge_topk: int = 4
    """top-k property-neighbor cells used for action/value bridge gradients"""
    actor_null_route_bias: float = 0.25
    """lower actor no-read bias so top-k credit is not spent on the null sink"""


class ChannelPropertyReadout(PropertyReadout):
    def __init__(self, cell_dim, property_dim, dense_floor, bridge_topk):
        super().__init__(cell_dim, property_dim, dense_floor)
        self.bridge_topk = bridge_topk

    def forward(self, query_prop, cell_prop, h, read_weights):
        logits = torch.einsum(
            "qp,bkp->bqk",
            self.query(query_prop),
            self.key(cell_prop)[None, :, :] + self.state_key(h),
        ) / np.sqrt(query_prop.shape[-1])
        logits = logits + self.relation_bias(query_prop, cell_prop)[None, :, :]
        logits = logits + torch.log(read_weights[:, None, :].clamp_min(1e-8))
        sparse_route = channel_entmax_route_with_floor(logits, 0.0, 1.5, dim=-1)[0]
        if self.dense_floor > 0.0:
            route = (1.0 - self.dense_floor) * sparse_route + self.dense_floor * torch.softmax(logits, dim=-1)
        else:
            route = sparse_route
        features = torch.bmm(route, self.value(h))
        bridge_route = channelized_softmax(logits, self.bridge_topk, dim=-1, exclude_mask=route.detach() > 1e-6)
        bridge_features = torch.bmm(bridge_route, self.value(h))
        entropy = -(route * torch.log(route.abs() + 1e-8)).sum(dim=-1).mean(dim=1) / np.log(max(cell_prop.shape[0], 2))
        support = effective_support(route, dim=-1).mean(dim=1)
        return self.out(features), self.out(bridge_features), entropy, support


class Agent(V18Agent):
    def __init__(self, envs, args=None):
        if args is None:
            args = Args()
        global CHANNEL_FLOW_CREDIT, CHANNEL_FLOW_TOPK
        CHANNEL_FLOW_CREDIT = args.route_flow_credit_floor
        CHANNEL_FLOW_TOPK = args.route_flow_topk
        super().__init__(envs, args)
        self.bridge_coef = args.bridge_coef
        self.actor_prop_read = ChannelPropertyReadout(args.cell_dim, args.property_dim, args.prop_io_dense_floor, args.bridge_topk)
        self.critic_prop_read = ChannelPropertyReadout(args.cell_dim, args.property_dim, args.prop_io_dense_floor, args.bridge_topk)
        self.actor_bridge_alpha = layer_init(nn.Linear(args.cell_dim, 1), std=0.01)
        self.actor_bridge_beta = layer_init(nn.Linear(args.cell_dim, 1), std=0.01)
        self.critic_bridge_value = layer_init(nn.Linear(args.cell_dim, 1), std=0.01)
        with torch.no_grad():
            self.actor.null_route_bias.fill_(args.actor_null_route_bias)

    def _bridge(self, x):
        return self.bridge_coef * (x - x.detach())

    def get_value(self, x):
        critic_features, critic_cells, critic_weights, _ = self.critic(x)
        prop_features, bridge_features, _, _ = self.critic_prop_read(
            self.value_prop, self.critic.cell_prop, critic_cells, critic_weights
        )
        base_value = self.critic_value(critic_features).squeeze(-1)
        prop_value = self.prop_value(prop_features).squeeze(-1).squeeze(1)
        bridge_value = self.critic_bridge_value(bridge_features).squeeze(-1).squeeze(1)
        return base_value + self._gate(self.critic_io_gate) * prop_value + self._bridge(bridge_value)

    def _actor_dist(self, actor_features, actor_cells, actor_weights):
        base_alpha_logits = self.actor_alpha(actor_features)
        base_beta_logits = self.actor_beta(actor_features)
        prop_features, bridge_features, read_entropy, read_support = self.actor_prop_read(
            self.action_prop, self.actor.cell_prop, actor_cells, actor_weights
        )
        gate = self._gate(self.actor_io_gate)
        alpha_logits = base_alpha_logits + gate * self.prop_alpha(prop_features).squeeze(-1)
        beta_logits = base_beta_logits + gate * self.prop_beta(prop_features).squeeze(-1)
        alpha_logits = alpha_logits + self._bridge(self.actor_bridge_alpha(bridge_features).squeeze(-1))
        beta_logits = beta_logits + self._bridge(self.actor_bridge_beta(bridge_features).squeeze(-1))
        alpha = 1.0 + F.softplus(alpha_logits)
        beta = 1.0 + F.softplus(beta_logits)
        dist = Beta(alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        return dist, to_action, read_entropy, read_support

    def get_action_and_value(self, x, z=None):
        actor_features, actor_cells, actor_weights, actor_stats = self.actor(x)
        critic_features, critic_cells, critic_weights, critic_stats = self.critic(x)
        dist, to_action, actor_read_entropy, actor_read_support = self._actor_dist(actor_features, actor_cells, actor_weights)
        critic_prop_features, critic_bridge_features, critic_read_entropy, critic_read_support = self.critic_prop_read(
            self.value_prop, self.critic.cell_prop, critic_cells, critic_weights
        )
        if z is None:
            z = dist.sample().clamp(1e-6, 1.0 - 1e-6)
        action = to_action(z)
        logprob = dist.log_prob(z).sum(1)
        entropy = dist.entropy().sum(1)
        value = (
            self.critic_value(critic_features).squeeze(-1)
            + self._gate(self.critic_io_gate) * self.prop_value(critic_prop_features).squeeze(-1).squeeze(1)
            + self._bridge(self.critic_bridge_value(critic_bridge_features).squeeze(-1).squeeze(1))
        )
        actor_stats = dict(actor_stats)
        critic_stats = dict(critic_stats)
        actor_stats["prop_read_entropy"] = actor_read_entropy
        actor_stats["prop_read_support"] = actor_read_support
        actor_stats["prop_io_gate"] = self._gate(self.actor_io_gate).expand_as(logprob)
        actor_stats["bridge_coef"] = x.new_full(logprob.shape, self.bridge_coef)
        actor_stats["compute"] = actor_stats["compute"] + self._gate(self.actor_io_gate) * actor_read_support / max(
            self.actor.K, 1
        )
        critic_stats["prop_read_entropy"] = critic_read_entropy
        critic_stats["prop_read_support"] = critic_read_support
        critic_stats["prop_io_gate"] = self._gate(self.critic_io_gate).expand_as(logprob)
        critic_stats["bridge_coef"] = x.new_full(logprob.shape, self.bridge_coef)
        critic_stats["compute"] = critic_stats["compute"] + self._gate(self.critic_io_gate) * critic_read_support / max(
            self.critic.K, 1
        )
        stats = {"actor": actor_stats, "critic": critic_stats}
        return action, z, logprob, entropy, value, actor_stats["compute"], critic_stats["compute"], stats


if __name__ == "__main__":
    args = tyro.cli(Args)
    CHANNEL_FLOW_CREDIT = args.route_flow_credit_floor
    CHANNEL_FLOW_TOPK = args.route_flow_topk
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    last_stats = None

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, z, logprob, _, value, _, _, last_stats = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            zs[step] = z
            logprobs[step] = logprob
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_zs = zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, _, newlogprob, entropy, newvalue, actor_compute, critic_compute, last_stats = agent.get_action_and_value(
                    b_obs[mb_inds], b_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                actor_multiplier = agent.compute_multiplier(actor_compute, args)
                critic_multiplier = agent.compute_multiplier(critic_compute, args)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss_per_sample = torch.max(pg_loss1, pg_loss2)
                pg_loss = signed_loss_with_safe_compute(pg_loss_per_sample, actor_multiplier, args.actor_compute_loss_floor)

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * (torch.max(v_loss_unclipped, v_loss_clipped) * critic_multiplier).mean()
                else:
                    v_loss = 0.5 * (((newvalue - b_returns[mb_inds]) ** 2) * critic_multiplier).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if last_stats is not None:
            for group in ("actor", "critic"):
                writer.add_scalar(f"morph/{group}_compute", mean_stat(last_stats, group, "compute"), global_step)
                writer.add_scalar(f"morph/{group}_active_cells", mean_stat(last_stats, group, "active_cells"), global_step)
                writer.add_scalar(f"morph/{group}_active_ticks", mean_stat(last_stats, group, "active_ticks"), global_step)
                writer.add_scalar(f"morph/{group}_expected_active_edges", mean_stat(last_stats, group, "expected_active_edges"), global_step)
                writer.add_scalar(f"morph/{group}_prop_read_entropy", mean_stat(last_stats, group, "prop_read_entropy"), global_step)
                writer.add_scalar(f"morph/{group}_prop_read_support", mean_stat(last_stats, group, "prop_read_support"), global_step)
                writer.add_scalar(f"morph/{group}_bridge_coef", mean_stat(last_stats, group, "bridge_coef"), global_step)
                writer.add_scalar(f"morph/{group}_prop_route_bias_std", mean_stat(last_stats, group, "prop_route_bias_std"), global_step)
        writer.add_scalar("morph/channel_flow_credit", CHANNEL_FLOW_CREDIT, global_step)
        writer.add_scalar("morph/channel_flow_topk", CHANNEL_FLOW_TOPK, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
