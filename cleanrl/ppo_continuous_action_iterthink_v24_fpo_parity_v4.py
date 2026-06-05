# PPO + FPO parity v4. Reference-style FPO in CleanRL.
#
# This intentionally drops the v24 distributional/rankgauss/shared-trunk stack.
# The goal is to test FPO in the regime assumed by ../fpo:
#   - raw unconstrained flow endpoint is stored and supervised;
#   - env receives tanh(raw_action);
#   - PPO ratio is exp(old_cfm_loss - new_cfm_loss);
#   - direct (obs_norm, x_t, t) flow MLP with LeCun init;
#   - separate scalar critic, reward scaling, simple normalized GAE.
#
# Hypothesis: previous v24 FPO variants failed because the CFM surrogate was
# grafted onto a bounded endpoint or onto machinery that assumes exact log-prob
# ratios. This version tests the FPO implementation details first.
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8000000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 1920
    anneal_lr: bool = False
    gamma: float = 0.995
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 16
    norm_adv: bool = True
    clip_coef: float = 0.05
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.0
    target_kl: Optional[float] = None
    unroll_length: int = 30

    reward_scale: float = 10.0
    normalize_observations: bool = True
    env_normalize_reward: bool = False
    env_normalize_observation: bool = False

    fpo_flow_steps: int = 10
    fpo_timestep_embed_dim: int = 8
    fpo_n_samples_per_action: int = 8
    fpo_average_losses_before_exp: bool = True
    fpo_discretize_t_for_training: bool = True
    fpo_policy_output_scale: float = 0.25
    fpo_source_std: float = 1.0
    fpo_feather_std: float = 0.0
    fpo_loss_clip: float = 3.0
    fpo_output_mode: str = "u_but_supervise_as_eps"

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma, normalize_obs, normalize_reward):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if normalize_obs:
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def lecun_linear(layer: nn.Linear):
    fan_in = layer.weight.shape[1]
    bound = np.sqrt(3.0 / fan_in)
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.zeros_(layer.bias)
    return layer


class RunningMeanStd(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var_sum", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("std_buf", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float32))

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, *self.mean.shape).float()
        batch_count = torch.tensor(float(x.shape[0]), device=x.device)
        new_count = self.count + batch_count
        diff_to_old_mean = x - self.mean
        new_mean = self.mean + diff_to_old_mean.sum(dim=0) / new_count.clamp_min(1.0)
        new_var_sum = (diff_to_old_mean * (x - new_mean)).sum(dim=0)
        new_std = torch.sqrt((new_var_sum / new_count.clamp_min(1.0)).clamp(1e-12, 1e12))

        self.mean.copy_(new_mean)
        self.var_sum.copy_(new_var_sum)
        self.std_buf.copy_(new_std)
        self.count.copy_(new_count)

    @property
    def std(self):
        return self.std_buf


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(lecun_linear(nn.Linear(dims[i], dims[i + 1])))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Agent(nn.Module):
    def __init__(self, envs, args: Args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        assert args.fpo_timestep_embed_dim % 2 == 0
        if args.fpo_output_mode not in ("u", "u_but_supervise_as_eps"):
            raise ValueError(f"unknown fpo_output_mode {args.fpo_output_mode}")

        self.obs_rms = RunningMeanStd((obs_dim,))
        self.normalize_observations = args.normalize_observations
        self.action_dim = act_dim
        self.flow_steps = args.fpo_flow_steps
        self.timestep_embed_dim = args.fpo_timestep_embed_dim
        self.n_samples_per_action = args.fpo_n_samples_per_action
        self.average_losses_before_exp = args.fpo_average_losses_before_exp
        self.discretize_t_for_training = args.fpo_discretize_t_for_training
        self.policy_output_scale = args.fpo_policy_output_scale
        self.source_std = args.fpo_source_std
        self.feather_std = args.fpo_feather_std
        self.loss_clip = args.fpo_loss_clip
        self.output_mode = args.fpo_output_mode

        self.policy = MLP((obs_dim + act_dim + args.fpo_timestep_embed_dim, 32, 32, 32, 32, act_dim))
        self.value = MLP((obs_dim, 256, 256, 256, 256, 256, 1))

    def norm_obs(self, obs):
        if not self.normalize_observations:
            return obs
        return (obs - self.obs_rms.mean) / self.obs_rms.std

    def schedule(self, device, dtype):
        return torch.linspace(1.0, 0.0, self.flow_steps + 1, device=device, dtype=dtype)

    def embed_timestep(self, t):
        assert t.shape[-1] == 1
        freqs = 2.0 ** torch.arange(self.timestep_embed_dim // 2, device=t.device, dtype=t.dtype)
        scaled_t = t * freqs
        return torch.cat([torch.cos(scaled_t), torch.sin(scaled_t)], dim=-1)

    def flow_forward(self, obs_norm, x_t, t):
        if obs_norm.dim() < x_t.dim():
            obs_norm = obs_norm.unsqueeze(-2).expand(*x_t.shape[:-1], obs_norm.shape[-1])
        return self.policy(torch.cat([obs_norm, x_t, self.embed_timestep(t)], dim=-1)) * self.policy_output_scale

    def cfm_loss(self, obs_norm, action, eps, t):
        x_t = t * eps + (1.0 - t) * action.unsqueeze(-2)
        velocity_pred = self.flow_forward(obs_norm, x_t, t)
        if self.output_mode == "u":
            velocity_gt = eps - action.unsqueeze(-2)
            return (velocity_pred - velocity_gt).square().mean(dim=-1)
        x0_pred = x_t - t * velocity_pred
        x1_pred = x0_pred + velocity_pred
        return (eps - x1_pred).square().mean(dim=-1)

    def sample_raw_action(self, obs_norm, deterministic=False):
        batch_shape = obs_norm.shape[:-1]
        device, dtype = obs_norm.device, obs_norm.dtype
        schedule = self.schedule(device, dtype)
        x_t = torch.randn(*batch_shape, self.action_dim, device=device, dtype=dtype) * self.source_std
        for i in range(self.flow_steps):
            t_now = schedule[i].expand(*batch_shape, 1)
            dt = schedule[i + 1] - schedule[i]
            x_t = x_t + dt * self.flow_forward(obs_norm, x_t, t_now)
        if (not deterministic) and self.feather_std > 0.0:
            x_t = x_t + torch.randn_like(x_t) * self.feather_std
        return x_t

    def sample_loss_terms(self, obs_norm, action):
        batch_shape = action.shape[:-1]
        device, dtype = action.device, action.dtype
        sample_shape = (*batch_shape, self.n_samples_per_action)
        eps = torch.randn(*sample_shape, self.action_dim, device=device, dtype=dtype) * self.source_std
        if self.discretize_t_for_training:
            schedule = self.schedule(device, dtype)[:-1]
            idx = torch.randint(self.flow_steps, (*sample_shape, 1), device=device)
            t = schedule[idx]
        else:
            t = torch.rand(*sample_shape, 1, device=device, dtype=dtype)
        initial_cfm_loss = self.cfm_loss(obs_norm, action, eps, t)
        return eps, t, initial_cfm_loss

    def get_value(self, obs):
        return self.value(self.norm_obs(obs))

    def get_action_and_value(self, obs):
        obs_norm = self.norm_obs(obs)
        action = self.sample_raw_action(obs_norm)
        eps, t, initial_cfm_loss = self.sample_loss_terms(obs_norm, action)
        entropy = torch.zeros(action.shape[0], device=action.device, dtype=action.dtype)
        return action, entropy, self.value(obs_norm), eps, t, initial_cfm_loss

    def evaluate_action_and_value(self, obs, action, eps, t, old_cfm_loss):
        obs_norm = self.norm_obs(obs)
        cfm_loss = self.cfm_loss(obs_norm, action, eps, t)
        if self.average_losses_before_exp:
            logratio = old_cfm_loss.mean(dim=-1, keepdim=True) - cfm_loss.mean(dim=-1, keepdim=True)
        else:
            logratio = (old_cfm_loss - cfm_loss).clamp(-self.loss_clip, self.loss_clip)
        ratio = logratio.exp()
        entropy = torch.zeros(action.shape[0], device=action.device, dtype=action.dtype)
        return ratio, logratio, entropy, self.value(obs_norm), cfm_loss


def prepare_minibatches(x, shuffle_indices, num_minibatches, unroll_length):
    suffix = x.shape[2:]
    x = x.swapaxes(0, 1).reshape((-1, unroll_length) + suffix)
    x = x[shuffle_indices]
    mb_subseqs = x.shape[0] // num_minibatches
    x = x.reshape((num_minibatches, mb_subseqs, unroll_length) + suffix)
    return x.swapaxes(1, 2)


@torch.no_grad()
def compute_gae_targets(agent, obs_mb, next_obs_mb, rewards_mb, term_dones_mb, truncations_mb, args):
    timesteps, batch_dim = rewards_mb.shape
    value_pred = agent.get_value(obs_mb.reshape((-1,) + obs_mb.shape[2:])).view(timesteps, batch_dim)
    bootstrap_value = agent.get_value(next_obs_mb[-1]).view(1, batch_dim)

    values_t_plus_1 = torch.cat([value_pred[1:], bootstrap_value], dim=0)
    discount = (1.0 - term_dones_mb) * args.gamma
    trunc_mask = 1.0 - truncations_mb
    deltas = (rewards_mb * args.reward_scale + discount * values_t_plus_1 - value_pred) * trunc_mask
    accum_scale = discount * args.gae_lambda * trunc_mask

    advantages_to_value = torch.zeros_like(rewards_mb)
    acc = torch.zeros(batch_dim, device=rewards_mb.device)
    for t_step in reversed(range(timesteps)):
        acc = deltas[t_step] + accum_scale[t_step] * acc
        advantages_to_value[t_step] = acc
    gae_values = advantages_to_value + value_pred

    gae_values_t_plus_1 = torch.cat([gae_values[1:], bootstrap_value], dim=0)
    advantages = (rewards_mb * args.reward_scale + discount * gae_values_t_plus_1 - value_pred) * trunc_mask
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return gae_values, advantages, value_pred


def total_grad_norm(parameters):
    params = list(parameters)
    total = torch.zeros((), device=params[0].device)
    used = False
    for p in params:
        if p.grad is not None:
            total = total + p.grad.detach().square().sum()
            used = True
    return total.sqrt() if used else total


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.fpo_timestep_embed_dim % 2 == 0
    assert args.num_steps % args.unroll_length == 0
    subseq_count = args.batch_size // args.unroll_length
    assert args.batch_size % args.unroll_length == 0
    assert subseq_count % args.num_minibatches == 0

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
    assert torch.cuda.is_available() and args.cuda, "FPO parity runs are CUDA-only"
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                args.gamma,
                args.env_normalize_observation,
                args.env_normalize_reward,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-8)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    term_dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    truncations = torch.zeros((args.num_steps, args.num_envs), device=device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    fpo_eps = torch.zeros(
        (args.num_steps, args.num_envs, args.fpo_n_samples_per_action) + envs.single_action_space.shape,
        device=device,
    )
    fpo_t = torch.zeros((args.num_steps, args.num_envs, args.fpo_n_samples_per_action, 1), device=device)
    fpo_old_cfm_loss = torch.zeros((args.num_steps, args.num_envs, args.fpo_n_samples_per_action), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_term_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            term_dones[step] = next_term_done

            with torch.no_grad():
                action, _, value, loss_eps, loss_t, old_cfm_loss = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                fpo_eps[step] = loss_eps
                fpo_t[step] = loss_t
                fpo_old_cfm_loss[step] = old_cfm_loss

            actions[step] = action
            env_action = torch.tanh(action)
            next_obs_np, reward, terminations, truncs, infos = envs.step(env_action.cpu().numpy())
            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device)
            truncations[step] = torch.tensor(truncs, dtype=torch.float32, device=device)
            next_term_done = torch.tensor(terminations, dtype=torch.float32, device=device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_obses[step] = next_obs

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            agent.obs_rms.update(obs)

        clipfracs = []
        ratio_means = []
        ratio_mins = []
        ratio_maxs = []
        cfm_losses = []
        adv_means = []
        adv_stds = []
        value_target_means = []
        value_error_stds = []
        for epoch in range(args.update_epochs):
            shuffle_indices = torch.randperm(subseq_count, device=device)
            mb_obs = prepare_minibatches(obs, shuffle_indices, args.num_minibatches, args.unroll_length)
            mb_next_obs = prepare_minibatches(next_obses, shuffle_indices, args.num_minibatches, args.unroll_length)
            mb_actions = prepare_minibatches(actions, shuffle_indices, args.num_minibatches, args.unroll_length)
            mb_rewards = prepare_minibatches(rewards.unsqueeze(-1), shuffle_indices, args.num_minibatches, args.unroll_length).squeeze(-1)
            mb_term_dones = prepare_minibatches(term_dones.unsqueeze(-1), shuffle_indices, args.num_minibatches, args.unroll_length).squeeze(-1)
            mb_truncations = prepare_minibatches(truncations.unsqueeze(-1), shuffle_indices, args.num_minibatches, args.unroll_length).squeeze(-1)
            mb_fpo_eps = prepare_minibatches(fpo_eps, shuffle_indices, args.num_minibatches, args.unroll_length)
            mb_fpo_t = prepare_minibatches(fpo_t, shuffle_indices, args.num_minibatches, args.unroll_length)
            mb_fpo_old_cfm_loss = prepare_minibatches(
                fpo_old_cfm_loss.unsqueeze(-1),
                shuffle_indices,
                args.num_minibatches,
                args.unroll_length,
            ).squeeze(-1)

            for mb_idx in range(args.num_minibatches):
                obs_mb = mb_obs[mb_idx]
                next_obs_mb = mb_next_obs[mb_idx]
                action_mb = mb_actions[mb_idx]
                rewards_mb = mb_rewards[mb_idx]
                term_dones_mb = mb_term_dones[mb_idx]
                truncations_mb = mb_truncations[mb_idx]
                gae_values, mb_advantages, _ = compute_gae_targets(
                    agent,
                    obs_mb,
                    next_obs_mb,
                    rewards_mb,
                    term_dones_mb,
                    truncations_mb,
                    args,
                )

                flat_obs = obs_mb.reshape((-1,) + envs.single_observation_space.shape)
                flat_actions = action_mb.reshape((-1,) + envs.single_action_space.shape)
                flat_eps = mb_fpo_eps[mb_idx].reshape((-1, args.fpo_n_samples_per_action) + envs.single_action_space.shape)
                flat_t = mb_fpo_t[mb_idx].reshape(-1, args.fpo_n_samples_per_action, 1)
                flat_old_cfm_loss = mb_fpo_old_cfm_loss[mb_idx].reshape(-1, args.fpo_n_samples_per_action)
                ratio, logratio, entropy, newvalue, cfm_loss = agent.evaluate_action_and_value(
                    flat_obs,
                    flat_actions,
                    flat_eps,
                    flat_t,
                    flat_old_cfm_loss,
                )
                ratio_means.append(ratio.mean().item())
                ratio_mins.append(ratio.min().item())
                ratio_maxs.append(ratio.max().item())
                cfm_losses.append(cfm_loss.mean().item())

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                flat_advantages = mb_advantages.reshape(-1)
                pg_loss1 = -flat_advantages.unsqueeze(-1) * ratio
                pg_loss2 = -flat_advantages.unsqueeze(-1) * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                flat_returns = gae_values.reshape(-1)
                flat_trunc_mask = (1.0 - truncations_mb).reshape(-1)
                v_error = (flat_returns - newvalue) * flat_trunc_mask
                v_loss = 0.5 * v_error.square().mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.max_grad_norm > 0.0:
                    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                else:
                    grad_norm = total_grad_norm(agent.parameters())
                optimizer.step()
                adv_means.append(mb_advantages.mean().item())
                adv_stds.append(mb_advantages.std().item())
                value_target_means.append(gae_values.mean().item())
                value_error_stds.append(v_error.std().item())

            if args.target_kl is not None:
                pseudo_kl = ((ratio - 1.0) - logratio).mean()
                if pseudo_kl > args.target_kl:
                    break

        with torch.no_grad():
            flat_obs_all = obs.reshape((-1,) + envs.single_observation_space.shape)
            values.copy_(agent.get_value(flat_obs_all).view(args.num_steps, args.num_envs))
            full_next_value = agent.get_value(next_obs).reshape(1, -1)
            full_advantages = torch.zeros_like(rewards)
            full_scaled_rewards = rewards * args.reward_scale
            lastgaelam = 0
            for t_step in reversed(range(args.num_steps)):
                if t_step == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_term_done
                    nextvalues = full_next_value
                else:
                    nextnonterminal = 1.0 - term_dones[t_step + 1]
                    nextvalues = values[t_step + 1]
                trunc_mask = 1.0 - truncations[t_step]
                delta = (full_scaled_rewards[t_step] + args.gamma * nextvalues * nextnonterminal - values[t_step]) * trunc_mask
                full_advantages[t_step] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * trunc_mask * lastgaelam
                )
            full_returns = full_advantages + values

        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_old_cfm = fpo_old_cfm_loss.reshape(-1, args.fpo_n_samples_per_action)
        y_pred, y_true = values.reshape(-1).cpu().numpy(), full_returns.reshape(-1).cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/grad_norm", float(grad_norm), global_step)
        writer.add_scalar("losses/fpo_cfm_loss", float(np.mean(cfm_losses)), global_step)
        writer.add_scalar("debug/fpo_ratio_mean", float(np.mean(ratio_means)), global_step)
        writer.add_scalar("debug/fpo_ratio_min", float(np.min(ratio_mins)), global_step)
        writer.add_scalar("debug/fpo_ratio_max", float(np.max(ratio_maxs)), global_step)
        writer.add_scalar("debug/fpo_old_cfm_loss", b_old_cfm.mean().item(), global_step)
        writer.add_scalar("debug/advantages_mean", float(np.mean(adv_means)), global_step)
        writer.add_scalar("debug/advantages_std", float(np.mean(adv_stds)), global_step)
        writer.add_scalar("debug/value_target_mean", float(np.mean(value_target_means)), global_step)
        writer.add_scalar("debug/value_error_std", float(np.mean(value_error_stds)), global_step)
        writer.add_scalar("debug/raw_action_std", b_actions.std().item(), global_step)
        writer.add_scalar("debug/raw_action_absmax", b_actions.abs().max().item(), global_step)
        writer.add_scalar("debug/tanh_action_saturation_frac", (torch.tanh(b_actions).abs() > 0.99).float().mean().item(), global_step)
        writer.add_scalar("debug/returns_mean_scaled", full_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std_scaled", full_returns.std().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
