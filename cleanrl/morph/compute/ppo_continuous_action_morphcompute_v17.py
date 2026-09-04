# PPO + Morphogenic Compute v17.
#
# METHOD. Property-Hypernetwork Residual Morph Transformer. A standard Gaussian
# PPO actor/critic remains the stable always-on path. A separate morphogenic
# property substrate learns residual corrections to action mean, action logstd,
# and value. Observations, cells, actions, value query, and internal time are
# property-bearing objects; shared relation laws coordinate IO binding, recurrent
# cell routing, readout, lifecycle mass allocation, plastic trace writes, and
# compute diagnostics.
#
# DESIGN. v16 made morphology responsible for basic policy learning and collapsed
# into compute starvation or weak IO credit. v17 keeps baseline PPO trainability
# while making every morph component learnable: lifecycle is conserved mass over
# cells plus reservoir, plasticity is differentiable fast trace state, relation
# weights are property-conditioned basis mixtures, and compute cost is an optional
# additive regularizer after warmup, not a PPO loss multiplier.
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
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Residual morph substrate arguments
    max_cells: int = 32
    """maximum learned property cells"""
    cell_dim: int = 64
    """cell state width"""
    property_dim: int = 32
    """property coordinate width shared by obs/cells/actions/value/time"""
    max_ticks: int = 3
    """internal recurrent refinement ticks"""
    relation_bases: int = 8
    """number of shared message basis transforms mixed by property relations"""
    morph_residual_scale: float = 0.01
    """initial scale of morph residual path; nonzero so morphology receives task gradients"""
    morph_logstd_residual_scale: float = 0.001
    """initial scale of morph state-dependent logstd residual"""
    max_actor_residual_scale: float = 0.25
    """learned actor mean residual trust scale upper bound"""
    max_logstd_residual_scale: float = 0.1
    """learned actor logstd residual trust scale upper bound"""
    max_value_residual_scale: float = 1.0
    """learned critic value residual trust scale upper bound"""
    morph_internal_scale: float = 0.1
    """initial scale of internal morph recurrent refinement updates"""
    compute_coef: float = 0.0
    """optional additive compute regularizer strength; default 0 to first prove utility"""
    compute_warmup_frac: float = 0.25
    """fraction of training before full compute regularization is applied"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


class ResidualMorphSubstrate(nn.Module):
    """Property-coordinated morph substrate used only as a residual path."""

    def __init__(self, obs_dim, args):
        super().__init__()
        self.obs_dim = obs_dim
        self.K = args.max_cells
        self.D = args.cell_dim
        self.P = args.property_dim
        self.T = args.max_ticks
        self.B = args.relation_bases

        self.obs_prop = nn.Parameter(torch.randn(obs_dim, self.P) * 0.5)
        self.cell_prop = nn.Parameter(torch.randn(self.K, self.P) * 0.5)
        self.time_prop = nn.Parameter(torch.randn(self.T, self.P) * 0.2)
        self.cell_seed = nn.Parameter(torch.randn(self.K, self.D) * 0.02)
        self.reservoir_logit = nn.Parameter(torch.zeros(1))

        self.scalar_token = layer_init(nn.Linear(2, self.D), std=0.5)
        self.prop_state = layer_init(nn.Linear(self.P, self.D), std=0.5)
        self.context = nn.Sequential(
            layer_init(nn.Linear(obs_dim, self.D)),
            nn.Tanh(),
            layer_init(nn.Linear(self.D, self.D)),
            nn.Tanh(),
        )

        self.q = layer_init(nn.Linear(self.D, self.P), std=0.3)
        self.k = layer_init(nn.Linear(self.D, self.P), std=0.3)
        self.v = layer_init(nn.Linear(self.D, self.D), std=0.5)
        self.prop_q = layer_init(nn.Linear(self.P, self.P), std=0.5)
        self.prop_k = layer_init(nn.Linear(self.P, self.P), std=0.5)
        self.relation = nn.Sequential(
            layer_init(nn.Linear(4 * self.P, self.P), std=0.5),
            ReLUSquared(),
            layer_init(nn.Linear(self.P, 2 + self.B), std=0.05),
        )
        self.basis = nn.Parameter(torch.randn(self.B, self.D, self.D) * 0.02)

        gate_dim = 3 * self.D
        self.mass_head = layer_init(nn.Linear(gate_dim, 1), std=0.01)
        self.decay_head = layer_init(nn.Linear(gate_dim, 1), std=0.01, bias_const=1.0)
        self.write_head = layer_init(nn.Linear(gate_dim, 1), std=0.01)
        self.trace_write = nn.Sequential(
            layer_init(nn.Linear(gate_dim, self.D)),
            ReLUSquared(),
            layer_init(nn.Linear(self.D, self.D), std=0.2),
        )
        self.update = nn.Sequential(
            layer_init(nn.Linear(4 * self.D, 2 * self.D)),
            ReLUSquared(),
            layer_init(nn.Linear(2 * self.D, self.D), std=0.2),
        )
        self.norm = nn.LayerNorm(self.D)
        self.layer_scale = nn.Parameter(torch.full((self.T,), args.morph_internal_scale))

    def _pair_features(self, target_prop, source_prop):
        target = target_prop[:, None, :]
        source = source_prop[None, :, :]
        return torch.cat(
            [
                target.expand(-1, source_prop.shape[0], -1),
                source.expand(target_prop.shape[0], -1, -1),
                target - source,
                target * source,
            ],
            dim=-1,
        )

    def _relation_outputs(self, target_prop, source_prop):
        out = self.relation(self._pair_features(target_prop, source_prop))
        return out[..., 0], out[..., 1], out[..., 2:]

    def _mass(self, h, prop_state, context):
        B = h.shape[0]
        ctx = context[:, None, :].expand(-1, self.K, -1)
        logits = self.mass_head(torch.cat([self.norm(h), prop_state.expand(B, -1, -1), ctx], dim=-1)).squeeze(-1)
        reservoir = self.reservoir_logit.expand(B, 1)
        mass_all = torch.softmax(torch.cat([reservoir, logits], dim=1), dim=1)
        return mass_all[:, 1:], mass_all[:, 0]

    def _attend(self, target_h, target_prop, source_h, source_prop, source_mass=None):
        target = self.norm(target_h)
        source = self.norm(source_h)
        logits = torch.einsum(
            "bip,bjp->bij",
            self.q(target) + self.prop_q(target_prop)[None, :, :],
            self.k(source) + self.prop_k(source_prop)[None, :, :],
        ) / np.sqrt(self.P)
        relation_bias, relation_gate, basis_logits = self._relation_outputs(target_prop, source_prop)
        logits = logits + relation_bias[None, :, :]
        if source_mass is not None:
            logits = logits + torch.log(source_mass[:, None, :].clamp_min(1e-8))
        attn = torch.softmax(logits, dim=-1)
        basis_mix = torch.softmax(basis_logits, dim=-1)
        values = self.v(source)
        basis_values = torch.einsum("bjd,rde->brje", values, self.basis)
        pair_values = torch.einsum("ijr,brje->bije", basis_mix, basis_values)
        gate = torch.sigmoid(relation_gate)[None, :, :, None]
        msg = (attn[:, :, :, None] * gate * pair_values).sum(dim=2)
        entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1).mean(dim=1)
        support = (1.0 / attn.pow(2).sum(dim=-1).clamp_min(1e-6)).mean(dim=1)
        return msg, attn, entropy, support

    def forward(self, x):
        B = x.shape[0]
        context = self.context(x)
        cell_prop = self.cell_prop + self.time_prop[0][None, :]
        cell_prop_state = self.prop_state(cell_prop)[None, :, :]
        h = self.cell_seed[None, :, :] + cell_prop_state + context[:, None, :]

        obs_features = torch.stack([x, x.pow(2)], dim=-1)
        obs_tokens = self.scalar_token(obs_features) + self.prop_state(self.obs_prop)[None, :, :]
        input_msg, input_attn, input_entropy, input_support = self._attend(h, cell_prop, obs_tokens, self.obs_prop)
        h = h + input_msg

        trace = torch.zeros_like(h)
        route_entropy_sum = x.new_zeros(B)
        route_support_sum = x.new_zeros(B)
        write_sum = x.new_zeros(B)
        compute_sum = x.new_zeros(B)
        last_mass = None
        last_reservoir = None
        last_write = None
        last_decay = None
        final_prop = cell_prop

        for t in range(self.T):
            tick_prop = self.cell_prop + self.time_prop[t][None, :]
            final_prop = tick_prop
            tick_prop_state = self.prop_state(tick_prop)[None, :, :]
            mass, reservoir_mass = self._mass(h, tick_prop_state, context)
            source_mass = mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-8)
            msg, _, route_entropy, route_support = self._attend(h, tick_prop, h, tick_prop, source_mass)

            gate_input = torch.cat(
                [self.norm(h), tick_prop_state.expand(B, -1, -1), context[:, None, :].expand(-1, self.K, -1)],
                dim=-1,
            )
            decay = torch.sigmoid(self.decay_head(gate_input))
            write = torch.sigmoid(self.write_head(gate_input))
            trace = decay * trace + write * self.trace_write(torch.cat([self.norm(h), msg, context[:, None, :].expand(-1, self.K, -1)], dim=-1))
            update = self.update(torch.cat([self.norm(h), msg, trace, tick_prop_state.expand(B, -1, -1)], dim=-1))
            h = h + self.layer_scale[t] * (self.K * mass)[:, :, None] * update

            route_entropy_sum = route_entropy_sum + route_entropy
            route_support_sum = route_support_sum + route_support
            write_sum = write_sum + (mass * write.squeeze(-1)).sum(dim=1)
            compute_sum = compute_sum + (1.0 - reservoir_mass) * route_support / max(self.K, 1)
            last_mass, last_reservoir, last_write, last_decay = mass, reservoir_mass, write.squeeze(-1), decay.squeeze(-1)

        stats = {
            "compute": (input_support / max(self.obs_dim, 1) + compute_sum / max(self.T, 1) + write_sum / max(self.T, 1)) / 3.0,
            "active_cells": last_mass.sum(dim=1).pow(2) / last_mass.pow(2).sum(dim=1).clamp_min(1e-8),
            "active_ticks": x.new_full((B,), float(self.T)),
            "reservoir_mass": last_reservoir,
            "live_mass": last_mass.sum(dim=1),
            "cell_mass_entropy": -(
                torch.cat([last_reservoir[:, None], last_mass], dim=1)
                * torch.log(torch.cat([last_reservoir[:, None], last_mass], dim=1) + 1e-8)
            ).sum(dim=1)
            / np.log(self.K + 1),
            "route_entropy": route_entropy_sum / max(self.T, 1) / np.log(max(self.K, 2)),
            "route_support": route_support_sum / max(self.T, 1),
            "input_entropy": input_entropy / np.log(max(self.obs_dim, 2)),
            "input_support": input_support,
            "plasticity": last_write.mean(dim=1),
            "trace_decay": last_decay.mean(dim=1),
        }
        return h, last_mass, stats, final_prop

    def read(self, query_prop, h, mass, source_prop=None):
        if source_prop is None:
            source_prop = self.cell_prop
        B = h.shape[0]
        query_h = self.prop_state(query_prop)[None, :, :].expand(B, -1, -1)
        live_mass = mass.sum(dim=1).clamp(0.0, 1.0)
        source_mass = mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-8)
        features, attn, entropy, support = self._attend(query_h, query_prop, h, source_prop, source_mass)
        stats = {
            "read_entropy": entropy / np.log(max(self.K, 2)),
            "read_support": support,
            "read_compute": support / max(self.K, 1),
            "live_mass": live_mass,
        }
        return self.norm(features + query_h), stats


class Agent(nn.Module):
    def __init__(self, envs, args=None):
        super().__init__()
        if args is None:
            args = Args()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.actor_morph = ResidualMorphSubstrate(obs_dim, args)
        self.critic_morph = ResidualMorphSubstrate(obs_dim, args)
        self.action_prop = nn.Parameter(torch.randn(action_dim, args.property_dim) * 0.5)
        self.value_prop = nn.Parameter(torch.randn(1, args.property_dim) * 0.5)
        self.max_actor_residual_scale = args.max_actor_residual_scale
        self.max_logstd_residual_scale = args.max_logstd_residual_scale
        self.max_value_residual_scale = args.max_value_residual_scale
        self.actor_delta_mean = layer_init(nn.Linear(args.cell_dim, 1), std=0.01)
        self.actor_delta_logstd = layer_init(nn.Linear(args.cell_dim, 1), std=0.01)
        self.critic_delta_value = layer_init(nn.Linear(args.cell_dim, 1), std=0.01)
        self.actor_mean_res_log = nn.Parameter(torch.log(torch.tensor(args.morph_residual_scale)))
        self.actor_logstd_res_log = nn.Parameter(torch.log(torch.tensor(args.morph_logstd_residual_scale)))
        self.critic_res_log = nn.Parameter(torch.log(torch.tensor(args.morph_residual_scale)))

    def _bounded_scale(self, log_scale, max_value):
        return log_scale.exp().clamp(max=max_value)

    def _value_with_stats(self, x):
        base_value = self.critic(x).squeeze(-1)
        cells, mass, morph_stats, final_prop = self.critic_morph(x)
        value_feature, read_stats = self.critic_morph.read(self.value_prop, cells, mass, final_prop)
        raw_delta = self.critic_delta_value(value_feature).squeeze(-1).squeeze(1)
        delta = torch.tanh(raw_delta) * read_stats["live_mass"]
        critic_scale = self._bounded_scale(self.critic_res_log, self.max_value_residual_scale)
        value = base_value + critic_scale * delta
        stats = dict(morph_stats)
        stats["compute"] = stats["compute"] + read_stats["read_compute"]
        stats.update(read_stats)
        stats["residual_scale"] = critic_scale.expand_as(value)
        stats["residual_norm"] = delta.detach().abs()
        return value, stats

    def get_value(self, x):
        value, _ = self._value_with_stats(x)
        return value

    def get_action_and_value(self, x, action=None):
        base_mean = self.actor_mean(x)
        cells, mass, actor_stats, final_prop = self.actor_morph(x)
        action_features, read_stats = self.actor_morph.read(self.action_prop, cells, mass, final_prop)
        live_mass = read_stats["live_mass"][:, None]
        raw_delta_mean = self.actor_delta_mean(action_features).squeeze(-1)
        raw_delta_logstd = self.actor_delta_logstd(action_features).squeeze(-1)
        delta_mean = torch.tanh(raw_delta_mean) * live_mass
        delta_logstd = torch.tanh(raw_delta_logstd) * live_mass
        mean_scale = self._bounded_scale(self.actor_mean_res_log, self.max_actor_residual_scale)
        logstd_scale = self._bounded_scale(self.actor_logstd_res_log, self.max_logstd_residual_scale)
        action_mean = base_mean + mean_scale * delta_mean
        action_logstd = (self.actor_logstd.expand_as(action_mean) + logstd_scale * delta_logstd.clamp(-2.0, 2.0)).clamp(-5.0, 2.0)
        probs = Normal(action_mean, torch.exp(action_logstd))
        if action is None:
            action = probs.sample()
        value, critic_stats = self._value_with_stats(x)
        actor_stats = dict(actor_stats)
        actor_stats["compute"] = actor_stats["compute"] + read_stats["read_compute"]
        actor_stats.update(read_stats)
        actor_stats["residual_scale"] = mean_scale.expand(action.shape[0])
        actor_stats["logstd_residual_scale"] = logstd_scale.expand(action.shape[0])
        actor_stats["residual_norm"] = delta_mean.detach().norm(dim=1)
        stats = {"actor": actor_stats, "critic": critic_stats}
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            value,
            actor_stats["compute"],
            critic_stats["compute"],
            stats,
        )


def mean_stat(stats, group, name):
    return stats[group][name].detach().mean().item()


def compute_regularizer_coef(args, global_step):
    if args.compute_coef <= 0.0:
        return 0.0
    warmup_steps = max(int(args.total_timesteps * args.compute_warmup_frac), 1)
    return args.compute_coef * min(global_step / warmup_steps, 1.0)


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
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
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
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, _, _, last_stats = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
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
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        compute_coef_now = compute_regularizer_coef(args, global_step)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, actor_compute, critic_compute, last_stats = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
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

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                compute_loss = (actor_compute.mean() + critic_compute.mean()) * compute_coef_now
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + compute_loss

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
        writer.add_scalar("morph/compute_coef", compute_coef_now, global_step)
        if last_stats is not None:
            for group in ("actor", "critic"):
                writer.add_scalar(f"morph/{group}_compute", mean_stat(last_stats, group, "compute"), global_step)
                writer.add_scalar(f"morph/{group}_active_cells", mean_stat(last_stats, group, "active_cells"), global_step)
                writer.add_scalar(f"morph/{group}_reservoir_mass", mean_stat(last_stats, group, "reservoir_mass"), global_step)
                writer.add_scalar(f"morph/{group}_live_mass", mean_stat(last_stats, group, "live_mass"), global_step)
                writer.add_scalar(f"morph/{group}_cell_mass_entropy", mean_stat(last_stats, group, "cell_mass_entropy"), global_step)
                writer.add_scalar(f"morph/{group}_route_entropy", mean_stat(last_stats, group, "route_entropy"), global_step)
                writer.add_scalar(f"morph/{group}_route_support", mean_stat(last_stats, group, "route_support"), global_step)
                writer.add_scalar(f"morph/{group}_read_entropy", mean_stat(last_stats, group, "read_entropy"), global_step)
                writer.add_scalar(f"morph/{group}_read_support", mean_stat(last_stats, group, "read_support"), global_step)
                writer.add_scalar(f"morph/{group}_plasticity", mean_stat(last_stats, group, "plasticity"), global_step)
                writer.add_scalar(f"morph/{group}_trace_decay", mean_stat(last_stats, group, "trace_decay"), global_step)
                writer.add_scalar(f"morph/{group}_residual_scale", mean_stat(last_stats, group, "residual_scale"), global_step)
                writer.add_scalar(f"morph/{group}_residual_norm", mean_stat(last_stats, group, "residual_norm"), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
