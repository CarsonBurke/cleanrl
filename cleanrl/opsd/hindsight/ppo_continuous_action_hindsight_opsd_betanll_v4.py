# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
#
# PPO + Hindsight-Conditioned OPSD Teacher, beta-NLL v4.
#
# Hypothesis:
# OPSD needs a real privileged target y*, not a scalar advantage heuristic. This
# variant keeps PPO as the grounded policy optimizer and adds an OPSD-style
# teacher that sees hindsight future context from the on-policy rollout:
# future observation deltas, discounted reward sums, and GAE. The teacher learns
# an advantage-improved action target y* with Gaussian beta-NLL (variance.detach()
# ** beta reweighting, as in ../beta-nll). For student learning, the teacher
# distribution is frozen once per rollout and distilled with KL(T || S). Unlike
# v2, negative-advantage samples do not become anti-action targets; they anchor
# near the behavior mean with lower teacher weight. v4 swaps the Gaussian PPO
# student for a v168-style state-dependent Beta policy, using the Beta action
# mean/std as the Gaussian approximation for OPSD teacher distillation. The
# student never sees the hindsight context.
import math
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


LOG_2PI = math.log(2.0 * math.pi)
SAMPLE_EPS = 1e-7
HINDSIGHT_OFFSETS = (1, 2, 4, 8, 16)
HINDSIGHT_REWARD_HORIZONS = (1, 4, 8, 16)


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
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
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

    # Hindsight OPSD teacher
    teacher_hidden_size: int = 64
    """hidden size for the hindsight-conditioned teacher network"""
    teacher_beta_nll_beta: float = 0.5
    """beta exponent for Gaussian beta-NLL: NLL terms are weighted by var.detach() ** beta"""
    teacher_loss_coef: float = 0.2
    """coefficient on the hindsight teacher beta-NLL loss"""
    teacher_var_min: float = 1e-4
    """minimum teacher variance"""
    teacher_var_max: float = 25.0
    """maximum teacher variance"""
    teacher_std_floor_ratio: float = 0.35
    """teacher KL std floor as a fraction of the current student std"""
    hindsight_weight_temp: float = 1.0
    """temperature for exponential weighting by hindsight advantage z-score"""
    hindsight_weight_min: float = 0.25
    """minimum normalized hindsight teacher weight"""
    hindsight_weight_max: float = 4.0
    """maximum normalized hindsight teacher weight"""
    hindsight_distill_beta: float = 0.05
    """final coefficient on KL(teacher || student)"""
    hindsight_distill_ramp_steps: int = 250000
    """global steps over which to ramp the hindsight distillation coefficient"""
    hindsight_kl_clip: float = 2.0
    """per-transition clip for the teacher-to-student KL before averaging"""
    ystar_adv_temp: float = 1.0
    """temperature for tanh-normalized hindsight advantage used in y*"""
    ystar_residual_scale: float = 1.0
    """scale on the behavior residual when forming positive-advantage y*"""
    teacher_negative_anchor_weight: float = 0.25
    """base teacher beta-NLL weight for non-positive advantage anchors"""
    teacher_positive_weight_bonus: float = 1.0
    """extra teacher beta-NLL weight for strongly positive normalized advantage"""
    teacher_weight_min: float = 0.1
    """minimum normalized teacher beta-NLL weight"""
    teacher_weight_max: float = 2.0
    """maximum normalized teacher beta-NLL weight"""
    torch_compile: bool = False
    """compile the policy and teacher MLP forwards with torch.compile when available"""
    torch_compile_mode: str = "reduce-overhead"
    """torch.compile mode for compiled MLP forwards"""

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


def maybe_compile(module, name, args):
    if not args.torch_compile:
        return module
    if not hasattr(torch, "compile"):
        print("torch.compile unavailable; running eagerly")
        return module
    print(f"compiling {name} with torch.compile(mode={args.torch_compile_mode!r})")
    return torch.compile(module, mode=args.torch_compile_mode)


def gaussian_beta_nll(mean, var, target, beta):
    ll = -0.5 * (((target - mean) ** 2 / var) + torch.log(var) + LOG_2PI)
    weight = var.detach().pow(beta)
    return -(ll * weight).sum(-1)


def gaussian_forward_kl(teacher_mean, teacher_std, student_mean, student_std):
    teacher_var = teacher_std.pow(2)
    student_var = student_std.pow(2)
    kl = torch.log(student_std / teacher_std)
    kl = kl + (teacher_var + (teacher_mean - student_mean).pow(2)) / (2.0 * student_var)
    return kl - 0.5


def normalize_batch(x):
    return (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True, unbiased=False) + 1e-8)


def clean_compiled_state_dict(module):
    return {key.replace("._orig_mod", ""): value for key, value in module.state_dict().items()}


def shifted_tensor(x, offset):
    result = torch.zeros_like(x)
    if offset < x.shape[0]:
        result[:-offset] = x[offset:]
    return result


def future_valid_mask(dones, offset):
    valid = torch.ones_like(dones)
    for j in range(1, offset + 1):
        shifted_done = torch.ones_like(dones)
        if j < dones.shape[0]:
            shifted_done[:-j] = dones[j:]
        valid = valid * (1.0 - shifted_done)
    return valid


def discounted_reward_sum(rewards, dones, horizon, gamma):
    total = torch.zeros_like(rewards)
    alive = torch.ones_like(rewards)
    for j in range(horizon):
        shifted_reward = shifted_tensor(rewards, j) if j > 0 else rewards
        total = total + alive * (gamma**j) * shifted_reward
        if j + 1 < horizon:
            shifted_done = torch.ones_like(dones)
            if j + 1 < dones.shape[0]:
                shifted_done[: -(j + 1)] = dones[j + 1 :]
            alive = alive * (1.0 - shifted_done)
    return total


def build_hindsight_context(obs, rewards, dones, advantages, gamma):
    continuous_features = []
    valid_features = []
    for offset in HINDSIGHT_OFFSETS:
        valid = future_valid_mask(dones, offset)
        future_obs = shifted_tensor(obs, offset)
        continuous_features.append((future_obs - obs) * valid.unsqueeze(-1))
        valid_features.append(valid.unsqueeze(-1))

    reward_sums = {}
    for horizon in HINDSIGHT_REWARD_HORIZONS:
        reward_sum = discounted_reward_sum(rewards, dones, horizon, gamma)
        reward_sums[horizon] = reward_sum
        continuous_features.append(reward_sum.unsqueeze(-1))

    continuous_features.append(advantages.unsqueeze(-1))

    continuous_context = torch.cat(continuous_features, dim=-1)
    flat_context = continuous_context.reshape((-1, continuous_context.shape[-1]))
    flat_context = normalize_batch(flat_context)
    context = torch.cat([flat_context.reshape_as(continuous_context), *valid_features], dim=-1)
    return context


def hindsight_advantage_z(hindsight_advantage):
    flat_adv = hindsight_advantage.reshape(-1)
    return (flat_adv - flat_adv.mean()) / (flat_adv.std(unbiased=False) + 1e-8)


def hindsight_weights(hindsight_advantage, args):
    adv_z = hindsight_advantage_z(hindsight_advantage)
    weights = torch.exp(adv_z / args.hindsight_weight_temp)
    weights = weights.clamp(args.hindsight_weight_min, args.hindsight_weight_max)
    weights = weights / (weights.mean() + 1e-8)
    return weights


def teacher_beta_nll_weights(hindsight_advantage, args):
    adv_z = hindsight_advantage_z(hindsight_advantage)
    positive_strength = torch.tanh(torch.relu(adv_z) / args.ystar_adv_temp)
    weights = args.teacher_negative_anchor_weight + args.teacher_positive_weight_bonus * positive_strength
    weights = weights.clamp(args.teacher_weight_min, args.teacher_weight_max)
    weights = weights / (weights.mean() + 1e-8)
    return weights


def build_ystar_actions(actions, behavior_mean, hindsight_advantage, args):
    adv_z = hindsight_advantage_z(hindsight_advantage).reshape(actions.shape[:-1])
    strength = torch.tanh(torch.relu(adv_z) / args.ystar_adv_temp).unsqueeze(-1)
    return behavior_mean + args.ystar_residual_scale * strength * (actions - behavior_mean)


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_beta = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 2 * action_dim), std=0.01),
        )
        self.register_buffer(
            "action_low",
            torch.tensor(envs.single_action_space.low, dtype=torch.float32),
        )
        self.register_buffer(
            "action_high",
            torch.tensor(envs.single_action_space.high, dtype=torch.float32),
        )

    def get_value(self, x):
        return self.critic(x).clone()

    def get_action_distribution(self, x):
        beta_head = self.actor_beta(x).clone()
        head_alpha, head_beta = beta_head.chunk(2, dim=-1)
        alpha = 1.0 + torch.nn.functional.softplus(head_alpha)
        beta = 1.0 + torch.nn.functional.softplus(head_beta)
        dist = Beta(alpha, beta)
        action_mean = self._z_to_action(dist.mean)
        action_std = dist.stddev * (self.action_high - self.action_low)
        return dist, action_mean, action_std.clamp_min(1e-6)

    def _z_to_action(self, z):
        return self.action_low + (self.action_high - self.action_low) * z

    def _action_to_z(self, action):
        z = (action - self.action_low) / (self.action_high - self.action_low)
        return z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)

    def get_action_and_value(self, x, action=None):
        probs, _, _ = self.get_action_distribution(x)
        if action is None:
            z = probs.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            action = self._z_to_action(z)
        else:
            z = self._action_to_z(action)
        return action, probs.log_prob(z).sum(1), probs.entropy().sum(1), self.critic(x).clone()


class HindsightTeacher(nn.Module):
    def __init__(self, obs_dim, context_dim, action_dim, hidden_size):
        super().__init__()
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim + context_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.mean = layer_init(nn.Linear(hidden_size, action_dim), std=0.01)
        self.logvar = layer_init(nn.Linear(hidden_size, action_dim), std=0.01)

    def forward(self, obs, hindsight_context, args):
        hidden = self.backbone(torch.cat([obs, hindsight_context], dim=-1)).clone()
        mean = self.mean(hidden).clone()
        logvar_min = math.log(args.teacher_var_min)
        logvar_max = math.log(args.teacher_var_max)
        logvar = self.logvar(hidden).clone().clamp(logvar_min, logvar_max)
        var = logvar.exp()
        return mean, var


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
    assert device.type == "cuda", "CUDA is required by this research variant"
    if args.torch_compile:
        torch.set_float32_matmul_precision("high")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    action_dim = int(np.prod(envs.single_action_space.shape))
    context_dim = obs_dim * len(HINDSIGHT_OFFSETS) + len(HINDSIGHT_REWARD_HORIZONS) + 1 + len(HINDSIGHT_OFFSETS)

    agent = Agent(envs).to(device)
    teacher = HindsightTeacher(obs_dim, context_dim, action_dim, args.teacher_hidden_size).to(device)
    agent.critic = maybe_compile(agent.critic, "agent.critic", args)
    agent.actor_beta = maybe_compile(agent.actor_beta, "agent.actor_beta", args)
    teacher.backbone = maybe_compile(teacher.backbone, "teacher.backbone", args)
    teacher.mean = maybe_compile(teacher.mean, "teacher.mean", args)
    teacher.logvar = maybe_compile(teacher.logvar, "teacher.logvar", args)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    teacher_optimizer = optim.Adam(teacher.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    action_means = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    action_stds = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
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

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            teacher_optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action_probs, action_mean, action_std = agent.get_action_distribution(next_obs)
                action = action_probs.sample()
                logprob = action_probs.log_prob(action).sum(1)
                value = agent.get_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            action_means[step] = action_mean
            action_stds[step] = action_std
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

            hindsight_advantage = advantages
            hindsight_context = build_hindsight_context(obs, rewards, dones, hindsight_advantage, args.gamma)
            b_hindsight_weights = hindsight_weights(hindsight_advantage, args)
            b_teacher_weights = teacher_beta_nll_weights(hindsight_advantage, args)

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_action_means = action_means.reshape((-1,) + envs.single_action_space.shape)
        b_action_stds = action_stds.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_hindsight_context = hindsight_context.reshape((args.batch_size, -1))
        b_ystar_actions = build_ystar_actions(actions, action_means, hindsight_advantage, args).reshape(
            (-1,) + envs.single_action_space.shape
        )

        fixed_teacher_means = []
        fixed_teacher_stds = []
        with torch.no_grad():
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                teacher_mean, teacher_var = teacher(b_obs[start:end], b_hindsight_context[start:end], args)
                teacher_std = teacher_var.sqrt()
                teacher_std = torch.maximum(teacher_std, args.teacher_std_floor_ratio * b_action_stds[start:end])
                fixed_teacher_means.append(teacher_mean)
                fixed_teacher_stds.append(teacher_std)
            b_fixed_teacher_mean = torch.cat(fixed_teacher_means, dim=0)
            b_fixed_teacher_std = torch.cat(fixed_teacher_stds, dim=0)

        # Optimizing the policy, value, and hindsight teacher networks
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        teacher_losses = []
        hindsight_kl_losses = []
        teacher_action_errors = []
        teacher_stds = []
        hindsight_weight_maxes = []
        active_hindsight_beta = args.hindsight_distill_beta * min(1.0, global_step / args.hindsight_distill_ramp_steps)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                _, student_mean, student_std = agent.get_action_distribution(b_obs[mb_inds])
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

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

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
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                hindsight_kl = gaussian_forward_kl(
                    b_fixed_teacher_mean[mb_inds],
                    b_fixed_teacher_std[mb_inds],
                    student_mean,
                    student_std,
                ).sum(-1)
                mb_hindsight_weights = b_hindsight_weights[mb_inds]
                hindsight_kl_loss = (mb_hindsight_weights * hindsight_kl.clamp(max=args.hindsight_kl_clip)).mean()

                entropy_loss = entropy.mean()
                policy_loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + v_loss * args.vf_coef
                    + active_hindsight_beta * hindsight_kl_loss
                )

                optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                mb_teacher_mean, mb_teacher_var = teacher(
                    b_obs[mb_inds],
                    b_hindsight_context[mb_inds],
                    args,
                )
                beta_nll = gaussian_beta_nll(
                    mb_teacher_mean,
                    mb_teacher_var,
                    b_ystar_actions[mb_inds],
                    args.teacher_beta_nll_beta,
                )
                teacher_loss = args.teacher_loss_coef * (b_teacher_weights[mb_inds] * beta_nll).mean()

                teacher_optimizer.zero_grad()
                teacher_loss.backward()
                nn.utils.clip_grad_norm_(teacher.parameters(), args.max_grad_norm)
                teacher_optimizer.step()

                teacher_losses.append(teacher_loss.item())
                hindsight_kl_losses.append(hindsight_kl_loss.item())
                teacher_action_errors.append((mb_teacher_mean.detach() - b_ystar_actions[mb_inds]).abs().mean().item())
                teacher_stds.append(mb_teacher_var.detach().sqrt().mean().item())
                hindsight_weight_maxes.append(mb_hindsight_weights.max().item())

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
        writer.add_scalar("losses/hindsight_teacher_beta_nll", np.mean(teacher_losses), global_step)
        writer.add_scalar("losses/hindsight_kl", np.mean(hindsight_kl_losses), global_step)
        writer.add_scalar("debug/hindsight_distill_beta", active_hindsight_beta, global_step)
        writer.add_scalar("debug/hindsight_teacher_abs_action_error", np.mean(teacher_action_errors), global_step)
        writer.add_scalar("debug/hindsight_teacher_std", np.mean(teacher_stds), global_step)
        writer.add_scalar("debug/hindsight_weight_max", np.mean(hindsight_weight_maxes), global_step)
        writer.add_scalar("debug/hindsight_advantage_mean", hindsight_advantage.mean().item(), global_step)
        writer.add_scalar("debug/hindsight_advantage_std", hindsight_advantage.std(unbiased=False).item(), global_step)
        with torch.no_grad():
            rollout_dist, _, rollout_action_std = agent.get_action_distribution(b_obs[: args.minibatch_size])
            writer.add_scalar(
                "debug/beta_concentration_sum",
                (rollout_dist.concentration1 + rollout_dist.concentration0).mean().item(),
                global_step,
            )
            writer.add_scalar("debug/beta_action_std", rollout_action_std.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(clean_compiled_state_dict(agent), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
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
