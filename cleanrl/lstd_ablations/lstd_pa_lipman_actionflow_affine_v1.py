"""Beta-logit affine action flow with bounded-space Conditional Flow Matching.

The policy replaces the multivariate normal with an independent Beta source
followed by a state-conditioned affine coupling flow in logit action space.
The flow is exactly invertible, so PPO ratios use the true transformed density.

Flow matching is applied to the same coupling velocity blocks: sample source
z0 from the Beta policy, map to y0=logit(z0), map rollout actions to y1, then
train fixed-time affine coupling velocities along straight y-space paths to
match y1-y0 with normalized positive-advantage weighting.
"""
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


SAMPLE_EPS = 1e-6


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
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef_low: float = 0.2
    clip_coef_high: float = 0.28
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    clip_vloss: bool = True
    flow_steps: int = 4
    """number of affine coupling steps in logit action space"""
    flow_scale: float = 1.0
    """maximum tanh-bounded affine velocity scale"""
    cfm_coef: float = 0.03
    """coefficient for bounded-space CFM loss on the policy flow"""
    cfm_weight_clip: float = 5.0
    """maximum normalized positive-advantage CFM weight"""
    cfm_detach_trunk: bool = True
    """prevent the CFM loss from moving shared actor features"""
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


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


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def safe_logit(z):
    z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
    return torch.log(z) - torch.log1p(-z)


def log_sigmoid_jacobian(y):
    return F.logsigmoid(y) + F.logsigmoid(-y)


class AffineVelocityCouplingBlock(nn.Module):
    def __init__(self, action_dim, cond_dim, flip=False, flow_scale=1.0):
        super().__init__()
        self.action_dim = int(action_dim)
        self.split = self.action_dim // 2
        self.flip = bool(flip)
        self.flow_scale = float(flow_scale)
        left_dim = self.action_dim - self.split if self.flip else self.split
        right_dim = self.split if self.flip else self.action_dim - self.split
        hidden_dim = 64
        self.net = nn.Sequential(
            layer_init(nn.Linear(cond_dim + left_dim + 1, hidden_dim), std=0.5),
            nn.SiLU(),
            RMSNorm(hidden_dim),
            layer_init(nn.Linear(hidden_dim, hidden_dim), std=0.5),
            nn.SiLU(),
            RMSNorm(hidden_dim),
            layer_init(nn.Linear(hidden_dim, 2 * right_dim), std=0.0),
        )

    def _split(self, y):
        left, right = y[..., : self.split], y[..., self.split :]
        return (right, left) if self.flip else (left, right)

    def _merge(self, left, right):
        return torch.cat((right, left), dim=-1) if self.flip else torch.cat((left, right), dim=-1)

    def _params(self, left, cond, t):
        if t.ndim == 0:
            t = t.expand(left.shape[0], 1)
        elif t.ndim == 1:
            t = t.unsqueeze(-1)
        shift, raw_scale = self.net(torch.cat((cond, left, t.to(dtype=left.dtype, device=left.device)), dim=-1)).chunk(2, dim=-1)
        return shift, self.flow_scale * torch.tanh(raw_scale)

    def forward(self, y, cond, t, dt):
        left, right = self._split(y)
        shift, scale = self._params(left, cond, t)
        scale_dt = dt * scale
        gain = torch.exp(scale_dt)
        safe_scale = torch.where(scale.abs() > 1e-4, scale, torch.ones_like(scale))
        shift_gain = torch.where(
            scale.abs() > 1e-4,
            torch.expm1(scale_dt) / safe_scale,
            dt * (1.0 + 0.5 * scale_dt),
        )
        right = right * gain + shift_gain * shift
        return self._merge(left, right), scale_dt.sum(dim=-1)

    def inverse(self, y, cond, t, dt):
        left, right = self._split(y)
        shift, scale = self._params(left, cond, t)
        scale_dt = dt * scale
        gain = torch.exp(scale_dt)
        safe_scale = torch.where(scale.abs() > 1e-4, scale, torch.ones_like(scale))
        shift_gain = torch.where(
            scale.abs() > 1e-4,
            torch.expm1(scale_dt) / safe_scale,
            dt * (1.0 + 0.5 * scale_dt),
        )
        right = (right - shift_gain * shift) / gain
        return self._merge(left, right), -scale_dt.sum(dim=-1)

    def cfm_loss(self, y_t, cond, t, target_velocity, weights):
        left, right = self._split(y_t)
        _, target_right = self._split(target_velocity)
        shift, scale = self._params(left, cond, t)
        pred_right = scale * right + shift
        per_sample = (pred_right - target_right).square().mean(dim=-1)
        return (weights * per_sample).mean()


class BetaLogitActionFlow(nn.Module):
    def __init__(self, action_dim, cond_dim, flow_steps, flow_scale):
        super().__init__()
        self.flow_steps = int(flow_steps)
        self.blocks = nn.ModuleList(
            AffineVelocityCouplingBlock(action_dim, cond_dim, flip=bool(i % 2), flow_scale=flow_scale)
            for i in range(self.flow_steps)
        )

    def _time(self, index, batch, device, dtype):
        t = (index + 0.5) / max(self.flow_steps, 1)
        return torch.full((batch, 1), float(t), device=device, dtype=dtype)

    def forward(self, y0, cond):
        y = y0
        logdet = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)
        dt = 1.0 / max(self.flow_steps, 1)
        for i, block in enumerate(self.blocks):
            y, block_logdet = block(y, cond, self._time(i, y.shape[0], y.device, y.dtype), dt)
            logdet = logdet + block_logdet
        return y, logdet

    def inverse(self, y1, cond):
        y = y1
        logdet = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)
        dt = 1.0 / max(self.flow_steps, 1)
        for i in reversed(range(self.flow_steps)):
            block = self.blocks[i]
            y, block_logdet = block.inverse(y, cond, self._time(i, y.shape[0], y.device, y.dtype), dt)
            logdet = logdet + block_logdet
        return y, logdet

    def cfm_loss(self, y0, y1, cond, weights):
        target_velocity = y1 - y0
        losses = []
        for i, block in enumerate(self.blocks):
            t = self._time(i, y0.shape[0], y0.device, y0.dtype)
            y_t = (1.0 - t) * y0 + t * y1
            losses.append(block.cfm_loss(y_t, cond, t, target_velocity, weights))
        return torch.stack(losses).mean()


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)
        hidden_dim = 64
        self.act_dim = act_dim
        self.cfm_detach_trunk = bool(args.cfm_detach_trunk)

        self.actor_fc1 = layer_init(nn.Linear(obs_dim, hidden_dim))
        self.actor_norm1 = RMSNorm(hidden_dim)
        self.actor_fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.actor_norm2 = RMSNorm(hidden_dim)

        self.actor_out = layer_init(nn.Linear(hidden_dim, 2 * act_dim), std=0.01)
        self.action_flow = BetaLogitActionFlow(act_dim, hidden_dim, args.flow_steps, args.flow_scale)
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))
        self.register_buffer(
            "action_scale_logdet",
            torch.log(torch.tensor(envs.single_action_space.high - envs.single_action_space.low, dtype=torch.float32)).sum(),
        )

        self.critic_fc1 = layer_init(nn.Linear(obs_dim, hidden_dim))
        self.critic_norm1 = RMSNorm(hidden_dim)
        self.critic_fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.critic_norm2 = RMSNorm(hidden_dim)
        self.value_out = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def _actor_features(self, x):
        h = F.silu(self.actor_norm1(self.actor_fc1(x)))
        h = F.silu(self.actor_norm2(self.actor_fc2(h)))
        return h

    def _get_distribution(self, h):
        head_alpha, head_beta = self.actor_out(h).chunk(2, dim=-1)
        alpha = 1.0 + F.softplus(head_alpha)
        beta = 1.0 + F.softplus(head_beta)
        return Beta(alpha, beta)

    def _z_to_action(self, z):
        return self.action_low + (self.action_high - self.action_low) * z

    def _action_to_z(self, action):
        z = (action - self.action_low) / (self.action_high - self.action_low)
        return z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)

    def _flow_logprob(self, dist, z0, y0, y1, flow_logdet):
        return (
            dist.log_prob(z0).sum(dim=-1)
            + log_sigmoid_jacobian(y0).sum(dim=-1)
            - flow_logdet
            - log_sigmoid_jacobian(y1).sum(dim=-1)
            - self.action_scale_logdet
        )

    def cfm_loss(self, x, actions, advantages, weight_clip):
        h = self._actor_features(x)
        cfm_h = h.detach() if self.cfm_detach_trunk else h
        with torch.no_grad():
            source_dist = self._get_distribution(h.detach())
            z0 = source_dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            y0 = safe_logit(z0)
            y1 = safe_logit(self._action_to_z(actions.detach()))
            adv = advantages.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            weights = 1.0 + F.relu(adv)
            weights = weights.clamp(max=weight_clip)
            weights = weights / (weights.mean() + 1e-8)
        return self.action_flow.cfm_loss(y0, y1, cfm_h, weights)

    def _critic_features(self, x):
        h = F.silu(self.critic_norm1(self.critic_fc1(x)))
        h = F.silu(self.critic_norm2(self.critic_fc2(h)))
        return h

    def get_value(self, x):
        return self.value_out(self._critic_features(x))

    def get_action_and_value(self, x, action=None):
        h = self._actor_features(x)
        dist = self._get_distribution(h)
        if action is None:
            z0 = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            y0 = safe_logit(z0)
            y1, flow_logdet = self.action_flow(y0, h)
            z1 = torch.sigmoid(y1).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            action = self._z_to_action(z1)
        else:
            z1 = self._action_to_z(action)
            y1 = safe_logit(z1)
            y0, inv_logdet = self.action_flow.inverse(y1, h)
            flow_logdet = -inv_logdet
            z0 = torch.sigmoid(y0).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        logprob = self._flow_logprob(dist, z0, y0, y1, flow_logdet)
        entropy = -logprob
        return action, logprob, entropy, self.get_value(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   sync_tensorboard=True, config=vars(args), name=run_name,
                   monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box)

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

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
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
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio < (1 - args.clip_coef_low)) | (ratio > (1 + args.clip_coef_high))).float().mean().item()]
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef_low, 1 + args.clip_coef_high)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef_low, args.clip_coef_high)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                cfm_loss = agent.cfm_loss(
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                    b_advantages[mb_inds],
                    args.cfm_weight_clip,
                )
                loss = loss + args.cfm_coef * cfm_loss
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
        writer.add_scalar("losses/cfm_loss", cfm_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
