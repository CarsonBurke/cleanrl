# PPO + Morphogenic Compute v23.
#
# METHOD. Real Conductance Readout Tokens. v20/v22 exposed a useful lesson:
# backward-only routes can improve early learning, but invisible gradient paths
# eventually fight the real policy/value computation. v23 keeps v18's stable
# property-coordinated substrate and adds only real forward paths: a learned paid
# dense conductance lane inside substrate routing, plus actor/value readout tokens
# that route through cells and directly affect Beta logits/value.
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

from cleanrl.ppo_continuous_action_morphcompute_v18 import (
    Args as V18Args,
    PropertyMorphogenicSubstrate,
    PropertyReadout,
    ReLUSquared,
    effective_support,
    entmax_route_with_floor,
    layer_init,
    make_env,
    mean_stat,
    signed_loss_with_safe_compute,
)

SAMPLE_EPS = 1e-6


@dataclass
class Args(V18Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    conductance_gate_bias: float = -1.5
    """initial logit for real dense conductance lane mixed into substrate routes"""
    conductance_gate_max: float = 0.10
    """maximum real dense conductance lane strength"""
    readout_dense_floor: float = 0.05
    """real dense floor for paid readout token routing"""
    readout_gate_bias: float = -1.5
    """initial logit for paid readout token residual"""
    readout_gate_max: float = 0.75
    """maximum paid readout token residual scale"""
    readout_mix: float = 0.75
    """message mix into paid readout token update"""


class ConductanceSubstrate(PropertyMorphogenicSubstrate):
    @property
    def route_dense_floor(self):
        base = getattr(self, "_base_route_dense_floor", 0.0)
        if hasattr(self, "conductance_gate_logit"):
            return base + self.conductance_gate_max * torch.sigmoid(self.conductance_gate_logit)
        return base

    @route_dense_floor.setter
    def route_dense_floor(self, value):
        self._base_route_dense_floor = float(value)

    def __init__(self, obs_dim, args, num_readouts):
        super().__init__(obs_dim, args)
        self.num_readouts = num_readouts
        self.conductance_gate_max = args.conductance_gate_max
        self.conductance_gate_logit = nn.Parameter(torch.tensor(float(args.conductance_gate_bias)))
        self.readout_dense_floor = args.readout_dense_floor
        self.readout_mix = args.readout_mix
        self.readout_prop = nn.Parameter(torch.randn(num_readouts, args.property_dim) * 0.5)
        self.readout_seed = nn.Parameter(torch.randn(num_readouts, self.D) * 0.02)
        self.readout_prop_state = layer_init(nn.Linear(args.property_dim, self.D), std=0.5)
        self.readout_q = layer_init(nn.Linear(self.D, args.property_dim), std=0.2)
        self.readout_k = layer_init(nn.Linear(self.D, args.property_dim), std=0.2)
        self.readout_prop_q = layer_init(nn.Linear(args.property_dim, args.property_dim), std=0.5)
        self.readout_prop_k = layer_init(nn.Linear(args.property_dim, args.property_dim), std=0.5)
        self.readout_value = layer_init(nn.Linear(self.D, self.D), std=0.5)
        self.readout_scale = np.sqrt(args.property_dim)
        self.readout_relation = nn.Sequential(
            layer_init(nn.Linear(4 * args.property_dim, args.property_dim), std=0.5),
            ReLUSquared(),
            layer_init(nn.Linear(args.property_dim, 1), std=0.05),
        )
        self.readout_update = nn.Sequential(
            layer_init(nn.Linear(self.D, 2 * self.D)),
            ReLUSquared(),
            layer_init(nn.Linear(2 * self.D, self.D), std=0.5),
        )
        self.readout_out = nn.Sequential(layer_init(nn.Linear(self.D, self.D)), ReLUSquared())

    def _readout_relation_bias(self, query_prop):
        q = query_prop[:, None, :]
        c = self.cell_prop[None, :, :]
        pair = torch.cat(
            [
                q.expand(-1, self.K, -1),
                c.expand(query_prop.shape[0], -1, -1),
                q - c,
                q * c,
            ],
            dim=-1,
        )
        return self.readout_relation(pair).squeeze(-1)

    def _readout_tokens(self, h, read_weights):
        B = h.shape[0]
        token_prop_state = self.readout_prop_state(self.readout_prop)
        token_h = self.readout_seed[None, :, :] + token_prop_state[None, :, :]
        logits = torch.einsum(
            "brp,bkp->brk",
            self.readout_q(self.norm(token_h)) + self.readout_prop_q(self.readout_prop)[None, :, :],
            self.readout_k(self.norm(h)) + self.readout_prop_k(self.cell_prop)[None, :, :],
        ) / self.readout_scale
        logits = logits + self._readout_relation_bias(self.readout_prop)[None, :, :]
        logits = logits + torch.log(read_weights[:, None, :].clamp_min(1e-8))
        route, sparse_route = entmax_route_with_floor(logits, self.readout_dense_floor, self.entmax_alpha, dim=-1)
        msg = torch.bmm(route, self.readout_value(self.norm(h)))
        token_h = token_h + self.readout_update(self.norm(token_h + self.readout_mix * msg))
        entropy = -(route * torch.log(route + 1e-8)).sum(dim=-1).mean(dim=1) / np.log(max(self.K, 2))
        support = effective_support(route, dim=-1).mean(dim=1)
        readout_compute = support * self.num_readouts / max(self.K, 1)
        return self.readout_out(token_h), entropy, support, readout_compute

    def forward(self, x):
        pooled, h, read_weights, stats = super().forward(x)
        token_features, token_entropy, token_support, token_compute = self._readout_tokens(h, read_weights)
        stats = dict(stats)
        stats["compute"] = stats["compute"] + token_compute
        stats["conductance_gate"] = (self.conductance_gate_max * torch.sigmoid(self.conductance_gate_logit)).expand(x.shape[0])
        stats["readout_entropy"] = token_entropy
        stats["readout_support"] = token_support
        stats["readout_compute"] = token_compute
        return pooled, token_features, h, read_weights, stats


class Agent(nn.Module):
    def __init__(self, envs, args=None):
        super().__init__()
        if args is None:
            args = Args()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.actor = ConductanceSubstrate(obs_dim, args, action_dim)
        self.critic = ConductanceSubstrate(obs_dim, args, 1)
        self.actor_alpha = layer_init(nn.Linear(args.cell_dim, action_dim), std=0.01)
        self.actor_beta = layer_init(nn.Linear(args.cell_dim, action_dim), std=0.01)
        self.critic_value = layer_init(nn.Linear(args.cell_dim, 1), std=1.0)
        self.action_prop = nn.Parameter(torch.randn(action_dim, args.property_dim) * 0.5)
        self.value_prop = nn.Parameter(torch.randn(1, args.property_dim) * 0.5)
        self.actor_prop_read = PropertyReadout(args.cell_dim, args.property_dim, args.prop_io_dense_floor)
        self.critic_prop_read = PropertyReadout(args.cell_dim, args.property_dim, args.prop_io_dense_floor)
        self.prop_alpha = layer_init(nn.Linear(args.cell_dim, 1), std=0.01)
        self.prop_beta = layer_init(nn.Linear(args.cell_dim, 1), std=0.01)
        self.prop_value = layer_init(nn.Linear(args.cell_dim, 1), std=0.01)
        self.token_alpha = layer_init(nn.Linear(args.cell_dim, 1), std=0.01)
        self.token_beta = layer_init(nn.Linear(args.cell_dim, 1), std=0.01)
        self.token_value = layer_init(nn.Linear(args.cell_dim, 1), std=0.01)
        self.actor_io_gate = nn.Parameter(torch.tensor(args.prop_io_gate_bias))
        self.critic_io_gate = nn.Parameter(torch.tensor(args.prop_io_gate_bias))
        self.readout_gate_logit = nn.Parameter(torch.tensor(float(args.readout_gate_bias)))
        self.value_readout_gate_logit = nn.Parameter(torch.tensor(float(args.readout_gate_bias)))
        self.prop_io_max_scale = args.prop_io_max_scale
        self.readout_gate_max = args.readout_gate_max
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

    def _gate(self, gate):
        return self.prop_io_max_scale * torch.sigmoid(gate)

    def _readout_gate(self):
        return self.readout_gate_max * torch.sigmoid(self.readout_gate_logit)

    def _value_readout_gate(self):
        return self.readout_gate_max * torch.sigmoid(self.value_readout_gate_logit)

    def get_value(self, x):
        critic_features, critic_tokens, critic_cells, critic_weights, _ = self.critic(x)
        prop_features, _, _ = self.critic_prop_read(self.value_prop, self.critic.cell_prop, critic_cells, critic_weights)
        return (
            self.critic_value(critic_features).squeeze(-1)
            + self._gate(self.critic_io_gate) * self.prop_value(prop_features).squeeze(-1).squeeze(1)
            + self._value_readout_gate() * self.token_value(critic_tokens).squeeze(-1).squeeze(1)
        )

    def compute_multiplier(self, compute, args):
        return args.min_compute_multiplier + args.compute_coef * compute

    def _actor_dist(self, actor_features, actor_tokens, actor_cells, actor_weights):
        base_alpha_logits = self.actor_alpha(actor_features)
        base_beta_logits = self.actor_beta(actor_features)
        prop_features, prop_entropy, prop_support = self.actor_prop_read(
            self.action_prop, self.actor.cell_prop, actor_cells, actor_weights
        )
        gate = self._gate(self.actor_io_gate)
        readout_gate = self._readout_gate()
        alpha_logits = (
            base_alpha_logits
            + gate * self.prop_alpha(prop_features).squeeze(-1)
            + readout_gate * self.token_alpha(actor_tokens).squeeze(-1)
        )
        beta_logits = (
            base_beta_logits
            + gate * self.prop_beta(prop_features).squeeze(-1)
            + readout_gate * self.token_beta(actor_tokens).squeeze(-1)
        )
        alpha = 1.0 + F.softplus(alpha_logits)
        beta = 1.0 + F.softplus(beta_logits)
        dist = Beta(alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        return dist, to_action, prop_entropy, prop_support

    def get_action_and_value(self, x, z=None):
        actor_features, actor_tokens, actor_cells, actor_weights, actor_stats = self.actor(x)
        critic_features, critic_tokens, critic_cells, critic_weights, critic_stats = self.critic(x)
        dist, to_action, actor_prop_entropy, actor_prop_support = self._actor_dist(
            actor_features, actor_tokens, actor_cells, actor_weights
        )
        critic_prop_features, critic_prop_entropy, critic_prop_support = self.critic_prop_read(
            self.value_prop, self.critic.cell_prop, critic_cells, critic_weights
        )
        if z is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        logprob = dist.log_prob(z).sum(1)
        entropy = dist.entropy().sum(1)
        value = (
            self.critic_value(critic_features).squeeze(-1)
            + self._gate(self.critic_io_gate) * self.prop_value(critic_prop_features).squeeze(-1).squeeze(1)
            + self._value_readout_gate() * self.token_value(critic_tokens).squeeze(-1).squeeze(1)
        )
        actor_stats = dict(actor_stats)
        critic_stats = dict(critic_stats)
        actor_stats["prop_read_entropy"] = actor_prop_entropy
        actor_stats["prop_read_support"] = actor_prop_support
        actor_stats["prop_io_gate"] = self._gate(self.actor_io_gate).expand_as(logprob)
        actor_stats["compute"] = actor_stats["compute"] + self._gate(self.actor_io_gate) * actor_prop_support / max(
            self.actor.K, 1
        )
        actor_stats["readout_gate"] = self._readout_gate().expand_as(logprob)
        actor_stats["token_logit_norm"] = self.token_alpha(actor_tokens).squeeze(-1).detach().norm(dim=1)
        critic_stats["prop_read_entropy"] = critic_prop_entropy
        critic_stats["prop_read_support"] = critic_prop_support
        critic_stats["prop_io_gate"] = self._gate(self.critic_io_gate).expand_as(logprob)
        critic_stats["compute"] = critic_stats["compute"] + self._gate(self.critic_io_gate) * critic_prop_support / max(
            self.critic.K, 1
        )
        critic_stats["readout_gate"] = self._value_readout_gate().expand_as(logprob)
        critic_stats["token_value_norm"] = self.token_value(critic_tokens).squeeze(-1).squeeze(1).detach().abs()
        stats = {"actor": actor_stats, "critic": critic_stats}
        return action, z, logprob, entropy, value, actor_stats["compute"], critic_stats["compute"], stats


if __name__ == "__main__":
    args = tyro.cli(Args)
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
                writer.add_scalar(f"morph/{group}_conductance_gate", mean_stat(last_stats, group, "conductance_gate"), global_step)
                writer.add_scalar(f"morph/{group}_readout_entropy", mean_stat(last_stats, group, "readout_entropy"), global_step)
                writer.add_scalar(f"morph/{group}_readout_support", mean_stat(last_stats, group, "readout_support"), global_step)
                writer.add_scalar(f"morph/{group}_readout_gate", mean_stat(last_stats, group, "readout_gate"), global_step)
                writer.add_scalar(f"morph/{group}_prop_read_entropy", mean_stat(last_stats, group, "prop_read_entropy"), global_step)
                writer.add_scalar(f"morph/{group}_prop_read_support", mean_stat(last_stats, group, "prop_read_support"), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
