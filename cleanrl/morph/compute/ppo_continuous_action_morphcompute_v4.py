# PPO + Morphogenic Compute v4.
#
# METHOD. Replace PPO's fixed actor/critic MLPs with separate differentiable
# computational substrates. Each substrate owns a maximum pool of latent cells.
# For each observation it predicts continuous laws over that substrate: growth
# and shrink pressure, coordinate-field routing, plasticity, persistence,
# abstraction temperature, and an adaptive compute budget. These laws softly
# determine how many cells are active, how strongly they exchange information,
# and how many recurrent update ticks are used. The graph is not edited
# discretely; the learned field induces the active computation.
#
# COMPUTE OBJECTIVE. PPO's policy and value losses are multiplied by a learned
# logical compute estimate:
#   policy loss = mean(stopgrad(mult_actor(s)) * clipped_ppo_loss(s))
#   value loss  = mean(mult_critic(s) * clipped_value_loss(s))
# where mult = 1 + compute_coef * normalized_compute. The multiplier remains
# differentiable so the model can reshape its substrate to optimize the
# loss-compute tradeoff, not just reward. Because the PPO actor surrogate is
# signed, the actor's morphogenesis gradient uses a zero-value multiplier term
# weighted by detached |clipped_ppo_loss|; otherwise good positive-advantage
# samples would perversely reward spending more compute.
#
# V2. Replace v1's unsquashed Normal actor with v168's Dreamer-style unimodal
# Beta actor: native z in [0, 1] is mapped linearly to the action bounds, with
# alpha,beta = 1 + softplus(head) so each dimension remains unimodal.
#
# V3. Replace all Tanh activations in the morphogenic substrate scaffolding with
# ReLU^2. The substrate's field/gating laws remain unchanged; this tests whether
# the higher-curvature, non-saturating positive activation improves MuJoCo credit
# flow versus bounded tanh features.
#
# V4. Replace the remaining GELU in the shared cell update law with ReLU^2, so
# every explicit substrate nonlinearity uses the same activation family. The
# raw observation embedding is simplified to a single linear projection; all
# nonlinearity lives inside the substrate update/readout and field gates. The
# readout is also kept as a plain Linear -> ReLU^2 projection, with no readout
# LayerNorm.
#
# HYPOTHESIS. MuJoCo policies benefit from state-dependent computational
# morphogenesis: easy states should use fewer cells/ticks, hard states should
# grow local computation and route through richer latent geometry. The compute
# multiplier should discourage unused structure while preserving the option to
# spend compute when it reduces actor or critic loss.
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

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


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

    # Morphogenic substrate arguments
    max_cells: int = 32
    """maximum latent cells available to each substrate"""
    cell_dim: int = 64
    """latent cell/state width"""
    field_dim: int = 8
    """coordinate dimension of the latent computational manifold"""
    max_ticks: int = 4
    """maximum recurrent compute ticks per forward pass"""
    compute_coef: float = 0.05
    """strength of differentiable compute multiplier on actor and critic losses"""
    min_compute_multiplier: float = 1.0
    """base multiplier added before compute pressure"""
    field_temp_min: float = 0.25
    """minimum routing/abstraction temperature"""
    field_temp_max: float = 4.0
    """maximum routing/abstraction temperature"""
    route_mix: float = 0.5
    """how strongly learned routing messages influence each cell update"""
    init_active_bias: float = -1.1
    """initial active-cell logit bias, about 25 percent active before input pressure"""
    init_tick_bias: float = -0.7
    """initial tick logit bias, favoring roughly one to two active ticks"""
    actor_compute_loss_floor: float = 0.01
    """small nonnegative floor for actor morphogenesis pressure when PPO loss is near zero"""

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
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
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


class MorphogenicSubstrate(nn.Module):
    """A fixed-capacity substrate whose active computation is generated by fields."""

    def __init__(self, obs_dim, args):
        super().__init__()
        self.K = args.max_cells
        self.D = args.cell_dim
        self.F = args.field_dim
        self.T = args.max_ticks
        self.temp_min = args.field_temp_min
        self.temp_max = args.field_temp_max
        self.route_mix = args.route_mix

        self.input = layer_init(nn.Linear(obs_dim, self.D))
        self.cell_seed = nn.Parameter(torch.randn(self.K, self.D) * 0.02)
        self.cell_pos = nn.Parameter(torch.randn(self.K, self.F) * 0.5)

        self.query = layer_init(nn.Linear(self.D, self.F), std=0.1)
        self.growth = layer_init(nn.Linear(self.D, self.K), std=0.01, bias_const=args.init_active_bias)
        self.shrink = layer_init(nn.Linear(self.D, self.K), std=0.01)
        self.plasticity = layer_init(nn.Linear(self.D, self.K), std=0.01)
        self.persistence = layer_init(nn.Linear(self.D, 1), std=0.01)
        self.temperature = layer_init(nn.Linear(self.D, 1), std=0.01)
        self.budget = layer_init(nn.Linear(self.D, 1), std=0.01)
        self.tick = layer_init(nn.Linear(self.D, self.T), std=0.01)

        tick_offsets = torch.linspace(0.0, -1.5, self.T)
        self.register_buffer("tick_offsets", tick_offsets)
        self.tick.bias.data.fill_(args.init_tick_bias)

        self.norm = nn.LayerNorm(self.D)
        self.update = nn.Sequential(
            layer_init(nn.Linear(self.D, 2 * self.D)),
            ReLUSquared(),
            layer_init(nn.Linear(2 * self.D, self.D), std=0.5),
        )
        self.readout = nn.Sequential(
            layer_init(nn.Linear(self.D, self.D)),
            ReLUSquared(),
        )

    def forward(self, x):
        B = x.shape[0]
        base = self.input(x)
        query = self.query(base)
        dist2_query = (query[:, None, :] - self.cell_pos[None, :, :]).pow(2).mean(dim=-1)

        growth_pressure = self.growth(base)
        shrink_pressure = F.softplus(self.shrink(base))
        active_logits = growth_pressure - shrink_pressure - 0.25 * dist2_query
        active = torch.sigmoid(active_logits)

        plasticity = torch.sigmoid(self.plasticity(base))
        persistence = torch.sigmoid(self.persistence(base))
        temp = self.temp_min + (self.temp_max - self.temp_min) * torch.sigmoid(self.temperature(base))
        budget = torch.sigmoid(self.budget(base))
        tick_gates = torch.sigmoid(self.tick(base) + self.tick_offsets)

        h = base[:, None, :] + self.cell_seed[None, :, :]
        coord_dist2 = (self.cell_pos[:, None, :] - self.cell_pos[None, :, :]).pow(2).mean(dim=-1)
        eye = torch.eye(self.K, device=x.device, dtype=x.dtype)
        route_logits = -coord_dist2[None, :, :] / temp[:, None, :]
        route_logits = route_logits + torch.log(active[:, None, :] + 1e-6)
        route_logits = route_logits.masked_fill(eye[None, :, :].bool(), -1e9)
        route = torch.softmax(route_logits, dim=-1)

        active_scale = active[:, :, None]
        plasticity_scale = plasticity[:, :, None]
        update_scale = budget[:, :, None] * active_scale * plasticity_scale
        for t in range(self.T):
            msg = torch.bmm(route, self.norm(h))
            mixed = h + self.route_mix * persistence[:, :, None] * msg
            delta = self.update(self.norm(mixed))
            h = h + tick_gates[:, t, None, None] * update_scale * delta

        read_weights = active / (active.sum(dim=1, keepdim=True) + 1e-6)
        pooled = torch.sum(read_weights[:, :, None] * h, dim=1)
        out = self.readout(pooled)

        active_cells = active.sum(dim=1)
        active_ticks = tick_gates.sum(dim=1)
        route_entropy = -(route * torch.log(route + 1e-8)).sum(dim=-1)
        route_entropy = (route_entropy * read_weights).sum(dim=1) / np.log(max(self.K - 1, 2))
        compute = (active_cells / self.K) * (active_ticks / self.T) * (0.5 + 0.5 * budget.squeeze(1))
        compute = compute.clamp(min=0.0)

        stats = {
            "compute": compute,
            "active_cells": active_cells,
            "active_ticks": active_ticks,
            "route_entropy": route_entropy,
            "growth_pressure": growth_pressure.mean(dim=1),
            "shrink_pressure": shrink_pressure.mean(dim=1),
            "plasticity": plasticity.mean(dim=1),
            "persistence": persistence.squeeze(1),
            "budget": budget.squeeze(1),
            "temperature": temp.squeeze(1),
        }
        return out, stats


class Agent(nn.Module):
    def __init__(self, envs, args=None):
        super().__init__()
        if args is None:
            args = Args()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        self.actor = MorphogenicSubstrate(obs_dim, args)
        self.critic = MorphogenicSubstrate(obs_dim, args)
        self.actor_alpha = layer_init(nn.Linear(args.cell_dim, action_dim), std=0.01)
        self.actor_beta = layer_init(nn.Linear(args.cell_dim, action_dim), std=0.01)
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))
        self.critic_value = layer_init(nn.Linear(args.cell_dim, 1), std=1.0)

    def get_value(self, x):
        critic_features, _ = self.critic(x)
        return self.critic_value(critic_features)

    def compute_multiplier(self, compute, args):
        return args.min_compute_multiplier + args.compute_coef * compute

    def _actor_dist(self, actor_features):
        alpha = 1.0 + F.softplus(self.actor_alpha(actor_features))
        beta = 1.0 + F.softplus(self.actor_beta(actor_features))
        dist = Beta(alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        return dist, to_action

    def get_action_and_value(self, x, z=None):
        actor_features, actor_stats = self.actor(x)
        critic_features, critic_stats = self.critic(x)
        dist, to_action = self._actor_dist(actor_features)
        if z is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        logprob = dist.log_prob(z).sum(1)
        entropy = dist.entropy().sum(1)

        stats = {
            "actor": actor_stats,
            "critic": critic_stats,
        }
        return (
            action,
            z,
            logprob,
            entropy,
            self.critic_value(critic_features),
            actor_stats["compute"],
            critic_stats["compute"],
            stats,
        )


def mean_stat(stats, group, name):
    return stats[group][name].detach().mean().item()


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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    last_stats = None

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, z, logprob, _, value, _, _, last_stats = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            zs[step] = z
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
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

        # bootstrap value if not done
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_zs = zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
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
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                actor_multiplier = agent.compute_multiplier(actor_compute, args)
                critic_multiplier = agent.compute_multiplier(critic_compute, args)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss_per_sample = torch.max(pg_loss1, pg_loss2)
                pg_task_loss = (pg_loss_per_sample * actor_multiplier.detach()).mean()
                actor_loss_magnitude = pg_loss_per_sample.detach().abs() + args.actor_compute_loss_floor
                actor_compute_loss = (actor_loss_magnitude * actor_multiplier).mean()
                pg_loss = pg_task_loss + actor_compute_loss - actor_compute_loss.detach()

                # Value loss
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
                    v_loss = 0.5 * (v_loss_max * critic_multiplier).mean()
                else:
                    v_loss_per_sample = (newvalue - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * (v_loss_per_sample * critic_multiplier).mean()

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

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if last_stats is not None:
            writer.add_scalar("morph/actor_compute", mean_stat(last_stats, "actor", "compute"), global_step)
            writer.add_scalar("morph/critic_compute", mean_stat(last_stats, "critic", "compute"), global_step)
            writer.add_scalar("morph/actor_active_cells", mean_stat(last_stats, "actor", "active_cells"), global_step)
            writer.add_scalar("morph/critic_active_cells", mean_stat(last_stats, "critic", "active_cells"), global_step)
            writer.add_scalar("morph/actor_active_cell_frac", mean_stat(last_stats, "actor", "active_cells") / args.max_cells, global_step)
            writer.add_scalar("morph/critic_active_cell_frac", mean_stat(last_stats, "critic", "active_cells") / args.max_cells, global_step)
            writer.add_scalar("morph/actor_active_ticks", mean_stat(last_stats, "actor", "active_ticks"), global_step)
            writer.add_scalar("morph/critic_active_ticks", mean_stat(last_stats, "critic", "active_ticks"), global_step)
            writer.add_scalar("morph/actor_active_tick_frac", mean_stat(last_stats, "actor", "active_ticks") / args.max_ticks, global_step)
            writer.add_scalar("morph/critic_active_tick_frac", mean_stat(last_stats, "critic", "active_ticks") / args.max_ticks, global_step)
            writer.add_scalar("morph/actor_route_entropy", mean_stat(last_stats, "actor", "route_entropy"), global_step)
            writer.add_scalar("morph/critic_route_entropy", mean_stat(last_stats, "critic", "route_entropy"), global_step)
            writer.add_scalar("morph/growth_pressure_mean", mean_stat(last_stats, "actor", "growth_pressure"), global_step)
            writer.add_scalar("morph/shrink_pressure_mean", mean_stat(last_stats, "actor", "shrink_pressure"), global_step)
            writer.add_scalar("morph/plasticity_mean", mean_stat(last_stats, "actor", "plasticity"), global_step)
            writer.add_scalar("morph/persistence_mean", mean_stat(last_stats, "actor", "persistence"), global_step)
            writer.add_scalar("morph/budget_mean", mean_stat(last_stats, "actor", "budget"), global_step)
            writer.add_scalar("morph/temperature_mean", mean_stat(last_stats, "actor", "temperature"), global_step)
            writer.add_scalar("morph/compute_multiplier_actor", actor_multiplier.detach().mean().item(), global_step)
            writer.add_scalar("morph/compute_multiplier_critic", critic_multiplier.detach().mean().item(), global_step)
            writer.add_scalar("morph/actor_compute_loss_magnitude", actor_loss_magnitude.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        class EvalAgent(Agent):
            def __init__(self, envs):
                super().__init__(envs, args)

            def get_action_and_value(self, x, z=None):
                action, _, logprob, entropy, value, _, _, _ = super().get_action_and_value(x, z)
                return action, logprob, entropy, value

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=EvalAgent,
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
