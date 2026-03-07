# Shared implementation for latent-imagination v2-v4.
# Latent Imagination PPO:
# 1. learn a stochastic short-horizon latent transition model p(z_{t+1} | z_t, a_t),
# 2. predict rewards and continuation in latent space,
# 3. optionally estimate several plausible near-future branches, and
# 4. keep the default training signal actor-safe by letting the model shape the
#    critic side only unless imagined-advantage mixing is explicitly enabled.
#
# Early versions underperformed because the actor was pushed directly by noisy
# model rollouts and the shared encoder was optimized for one-step prediction as
# much as for control. The v3 regime keeps the safer critic-side setup from v2,
# but reintroduces a small imagined-advantage term only late in training, once
# the model has had time to become directionally useful.
import copy
import os
import random
import time
from dataclasses import dataclass
from functools import partial

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

    # Latent imagination arguments
    latent_dim: int = 64
    """latent state width shared by the policy, critic, and world model"""
    model_hidden_dim: int = 128
    """hidden width of the transition, reward, and continuation models"""
    model_min_std: float = 0.05
    """minimum std for the stochastic latent transition"""
    model_max_std: float = 1.0
    """maximum std for the stochastic latent transition"""
    target_encoder_tau: float = 0.99
    """EMA factor for the target encoder used as a stable next-latent target"""
    imag_horizon: int = 5
    """number of imagined steps per branch"""
    imag_branches: int = 4
    """number of stochastic branches to average per starting point"""
    imag_adv_coef: float = 0.05
    """mixture weight for the imagined advantage in the PPO actor loss"""
    imag_start_fraction: float = 0.25
    """fraction of total training before imagined advantages are allowed to affect the actor"""
    imag_ramp_fraction: float = 0.25
    """fraction of total training used to ramp to the target imagined-advantage weight"""
    use_imag_conf_gate: bool = False
    """gate imagined actor corrections by model-confidence heuristics"""
    require_imag_sign_agreement: bool = True
    """only trust imagined actor corrections when real and imagined advantages agree in sign"""
    imag_std_gate_temperature: float = 1.0
    """temperature for converting branch disagreement into a confidence gate"""
    imag_adv_clip_multiplier: float = 2.0
    """clip imagined advantages to this multiple of batch real-advantage std before actor mixing"""
    model_coef: float = 0.25
    """overall weight on the latent-model loss"""
    transition_coef: float = 1.0
    """weight on the stochastic latent transition loss"""
    reward_coef: float = 1.0
    """weight on the one-step reward prediction loss"""
    value_consistency_coef: float = 0.1
    """weight on aligning predicted-next value with encoded-next value"""
    done_coef: float = 0.25
    """weight on the continuation / done prediction loss"""
    detach_model_encoder: bool = True
    """detach the shared encoder before world-model losses"""
    use_done_model: bool = False
    """learn a done / continuation head for imagined rollouts"""

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


def gaussian_nll(target, mean, std):
    log_std = torch.log(std + 1e-8)
    squared_error = ((target - mean) / (std + 1e-8)) ** 2
    return 0.5 * (squared_error + 2.0 * log_std)


def current_imagination_coef(args: Args, global_step: int) -> float:
    if args.imag_adv_coef <= 0:
        return 0.0
    if args.imag_ramp_fraction <= 0:
        return args.imag_adv_coef
    start_step = int(args.total_timesteps * args.imag_start_fraction)
    ramp_steps = max(1, int(args.total_timesteps * args.imag_ramp_fraction))
    progress = (global_step - start_step) / ramp_steps
    progress = min(1.0, max(0.0, progress))
    return args.imag_adv_coef * progress


def gated_imagination_advantages(
    args: Args,
    real_advantages: torch.Tensor,
    imag_advantages: torch.Tensor,
    imag_stds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    real_adv_std = real_advantages.std(unbiased=False).clamp_min(1e-6)
    clipped_imag_advantages = torch.clamp(
        imag_advantages,
        -args.imag_adv_clip_multiplier * real_adv_std,
        args.imag_adv_clip_multiplier * real_adv_std,
    )

    if args.use_imag_conf_gate:
        normalized_std = imag_stds / imag_stds.mean().clamp_min(1e-6)
        conf_gate = torch.exp(-normalized_std / max(args.imag_std_gate_temperature, 1e-6))
    else:
        conf_gate = torch.ones_like(imag_advantages)

    if args.require_imag_sign_agreement:
        sign_gate = (torch.sign(real_advantages) == torch.sign(imag_advantages)).float()
    else:
        sign_gate = torch.ones_like(imag_advantages)

    total_gate = conf_gate * sign_gate
    return clipped_imag_advantages, conf_gate, sign_gate, total_gate


def extract_model_next_obs(next_obs: np.ndarray, infos: dict) -> np.ndarray:
    model_next_obs = np.array(next_obs, copy=True)
    final_observations = infos.get("final_observation")
    if final_observations is None:
        return model_next_obs

    for env_idx, final_obs in enumerate(final_observations):
        if final_obs is not None:
            model_next_obs[env_idx] = final_obs
    return model_next_obs


class Agent(nn.Module):
    def __init__(
        self,
        envs,
        latent_dim: int = 64,
        model_hidden_dim: int = 128,
        model_min_std: float = 0.05,
        model_max_std: float = 1.0,
        use_done_model: bool = False,
    ):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))

        self.latent_dim = latent_dim
        self.model_min_std = model_min_std
        self.model_max_std = model_max_std
        self.use_done_model = use_done_model

        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, latent_dim)),
            nn.Tanh(),
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False

        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.transition_backbone = nn.Sequential(
            layer_init(nn.Linear(latent_dim + action_dim, model_hidden_dim)),
            nn.SiLU(),
            layer_init(nn.Linear(model_hidden_dim, model_hidden_dim)),
            nn.SiLU(),
        )
        self.transition_mean = layer_init(nn.Linear(model_hidden_dim, latent_dim), std=0.01)
        self.transition_logstd = layer_init(nn.Linear(model_hidden_dim, latent_dim), std=0.01)

        self.reward_model = nn.Sequential(
            layer_init(nn.Linear(2 * latent_dim + action_dim, model_hidden_dim)),
            nn.SiLU(),
            layer_init(nn.Linear(model_hidden_dim, 1), std=0.01),
        )
        if self.use_done_model:
            self.done_model = nn.Sequential(
                layer_init(nn.Linear(2 * latent_dim + action_dim, model_hidden_dim)),
                nn.SiLU(),
                layer_init(nn.Linear(model_hidden_dim, 1), std=0.01),
            )
        else:
            self.done_model = None

        self.register_buffer(
            "action_low",
            torch.tensor(envs.single_action_space.low, dtype=torch.float32),
        )
        self.register_buffer(
            "action_high",
            torch.tensor(envs.single_action_space.high, dtype=torch.float32),
        )

    def encode(self, obs):
        return self.encoder(obs)

    def encode_target(self, obs):
        with torch.no_grad():
            return self.target_encoder(obs)

    def clamp_action(self, action):
        return torch.max(torch.min(action, self.action_high), self.action_low)

    def get_dist_from_latent(self, latent):
        action_mean = self.actor_mean(latent)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std)

    def get_value_from_latent(self, latent):
        return self.critic(latent)

    def get_value(self, obs):
        return self.get_value_from_latent(self.encode(obs))

    def get_action_and_value(self, obs, action=None):
        latent = self.encode(obs)
        probs = self.get_dist_from_latent(latent)
        if action is None:
            action = probs.sample()
        value = self.get_value_from_latent(latent)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

    def transition_params(self, latent, env_action):
        model_input = torch.cat([latent, env_action], dim=-1)
        hidden = self.transition_backbone(model_input)
        residual_mean = self.transition_mean(hidden)
        mean = latent + residual_mean
        std_scale = torch.sigmoid(self.transition_logstd(hidden))
        std = self.model_min_std + (self.model_max_std - self.model_min_std) * std_scale
        return mean, std

    def predict_reward_done(self, latent, env_action, next_latent):
        model_input = torch.cat([latent, env_action, next_latent], dim=-1)
        reward = self.reward_model(model_input).squeeze(-1)
        done_logit = self.done_model(model_input).squeeze(-1) if self.done_model is not None else None
        return reward, done_logit

    def imagine_returns(self, obs, first_env_action, horizon: int, branches: int, gamma: float):
        latent = self.encode(obs)
        batch_size = latent.shape[0]

        latent = latent.unsqueeze(1).expand(-1, branches, -1).reshape(batch_size * branches, self.latent_dim)
        env_action = first_env_action.unsqueeze(1).expand(-1, branches, -1).reshape(batch_size * branches, -1)

        imagined_return = torch.zeros(batch_size * branches, device=latent.device)
        discount = torch.ones(batch_size * branches, device=latent.device)
        continuation = torch.ones(batch_size * branches, device=latent.device)

        for step in range(horizon):
            mean, std = self.transition_params(latent, env_action)
            next_latent = mean + torch.randn_like(std) * std
            reward_pred, done_logit = self.predict_reward_done(latent, env_action, next_latent)
            if done_logit is None:
                done_prob = torch.zeros_like(reward_pred)
            else:
                done_prob = torch.sigmoid(done_logit)

            imagined_return += discount * continuation * reward_pred
            continuation = continuation * (1.0 - done_prob)
            discount = discount * gamma
            latent = next_latent

            if step < horizon - 1:
                policy_action = self.get_dist_from_latent(latent).sample()
                env_action = self.clamp_action(policy_action)

        bootstrap = self.get_value_from_latent(latent).squeeze(-1)
        imagined_return += discount * continuation * bootstrap

        imagined_return = imagined_return.view(batch_size, branches)
        return imagined_return.mean(dim=1), imagined_return.std(dim=1, unbiased=False)

    @torch.no_grad()
    def update_target_encoder(self, tau: float):
        for target_param, online_param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target_param.data.mul_(tau).add_(online_param.data, alpha=1.0 - tau)


def main(args_class=Args):
    args = tyro.cli(args_class)
    assert args.imag_horizon > 0, "imag_horizon must be positive"
    assert args.imag_branches > 0, "imag_branches must be positive"
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

    agent = Agent(
        envs,
        latent_dim=args.latent_dim,
        model_hidden_dim=args.model_hidden_dim,
        model_min_std=args.model_min_std,
        model_max_std=args.model_max_std,
        use_done_model=args.use_done_model,
    ).to(device)
    optimizer = optim.Adam(
        [parameter for parameter in agent.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        eps=1e-5,
    )

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    env_actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    next_dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
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

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                value = value.flatten()
                executed_action = agent.clamp_action(action)
                values[step] = value
            actions[step] = action
            env_actions[step] = executed_action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs_np, reward, terminations, truncations, infos = envs.step(executed_action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            model_next_obs_np = extract_model_next_obs(next_obs_np, infos)

            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obses[step] = torch.tensor(model_next_obs_np, device=device)
            next_dones[step] = torch.tensor(next_done_np, device=device)

            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(next_done_np).to(device)

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
                    nextnonterminal = 1.0 - next_dones[t]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_env_actions = env_actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_rewards = rewards.reshape(-1)
        b_next_obs = next_obses.reshape((-1,) + envs.single_observation_space.shape)
        b_next_dones = next_dones.reshape(-1)

        with torch.no_grad():
            imag_coef = current_imagination_coef(args, global_step)
            if imag_coef > 0.0:
                b_imag_returns, b_imag_stds = agent.imagine_returns(
                    b_obs,
                    b_env_actions,
                    horizon=args.imag_horizon,
                    branches=args.imag_branches,
                    gamma=args.gamma,
                )
                b_imag_advantages = b_imag_returns - b_values
                b_imag_advantages_capped, b_conf_gate, b_sign_gate, b_total_gate = gated_imagination_advantages(
                    args,
                    b_advantages,
                    b_imag_advantages,
                    b_imag_stds,
                )
                if args.use_imag_conf_gate:
                    b_policy_advantages = b_advantages + imag_coef * b_total_gate * b_imag_advantages_capped
                else:
                    b_policy_advantages = (1.0 - imag_coef) * b_advantages + imag_coef * b_imag_advantages
            else:
                b_imag_returns = torch.zeros_like(b_values)
                b_imag_stds = torch.zeros_like(b_values)
                b_imag_advantages = torch.zeros_like(b_values)
                b_imag_advantages_capped = torch.zeros_like(b_values)
                b_conf_gate = torch.zeros_like(b_values)
                b_sign_gate = torch.zeros_like(b_values)
                b_total_gate = torch.zeros_like(b_values)
                b_policy_advantages = b_advantages

        # Optimizing the policy, value, and latent-model networks
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
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_policy_advantages[mb_inds]
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

                # Latent-model loss
                latent = agent.encode(b_obs[mb_inds])
                model_latent = latent.detach() if args.detach_model_encoder else latent
                target_next_latent = agent.encode_target(b_next_obs[mb_inds])
                pred_next_mean, pred_next_std = agent.transition_params(model_latent, b_env_actions[mb_inds])
                reward_pred, done_logit = agent.predict_reward_done(
                    model_latent,
                    b_env_actions[mb_inds],
                    target_next_latent,
                )

                transition_loss = gaussian_nll(target_next_latent, pred_next_mean, pred_next_std).mean()
                reward_loss = F.mse_loss(reward_pred, b_rewards[mb_inds])
                if done_logit is None:
                    done_loss = torch.zeros((), device=device)
                else:
                    done_loss = F.binary_cross_entropy_with_logits(done_logit, b_next_dones[mb_inds])
                predicted_next_value = agent.get_value_from_latent(pred_next_mean).view(-1)
                target_next_value = agent.get_value_from_latent(target_next_latent).view(-1).detach()
                value_consistency_loss = 0.5 * ((predicted_next_value - target_next_value) ** 2).mean()
                model_loss = (
                    args.transition_coef * transition_loss
                    + args.reward_coef * reward_loss
                    + args.value_consistency_coef * value_consistency_loss
                    + args.done_coef * done_loss
                )
                model_coef_now = args.model_coef

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + model_loss * model_coef_now

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                agent.update_target_encoder(args.target_encoder_tau)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        real_adv_np = b_advantages.cpu().numpy()
        imag_adv_np = b_imag_advantages.cpu().numpy()
        if np.std(real_adv_np) > 1e-8 and np.std(imag_adv_np) > 1e-8:
            imag_adv_corr = float(np.corrcoef(real_adv_np, imag_adv_np)[0, 1])
        else:
            imag_adv_corr = 0.0

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/model_loss", model_loss.item(), global_step)
        writer.add_scalar("losses/transition_loss", transition_loss.item(), global_step)
        writer.add_scalar("losses/reward_loss", reward_loss.item(), global_step)
        writer.add_scalar("losses/value_consistency_loss", value_consistency_loss.item(), global_step)
        writer.add_scalar("losses/done_loss", done_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("imagination/mix_coef", imag_coef, global_step)
        writer.add_scalar("imagination/imagined_return", b_imag_returns.mean().item(), global_step)
        writer.add_scalar("imagination/imagined_return_std", b_imag_stds.mean().item(), global_step)
        writer.add_scalar("imagination/imagined_advantage", b_imag_advantages.mean().item(), global_step)
        writer.add_scalar("imagination/imagined_advantage_capped", b_imag_advantages_capped.mean().item(), global_step)
        writer.add_scalar("imagination/advantage_correlation", imag_adv_corr, global_step)
        writer.add_scalar("imagination/conf_gate", b_conf_gate.mean().item(), global_step)
        writer.add_scalar("imagination/sign_gate", b_sign_gate.mean().item(), global_step)
        writer.add_scalar("imagination/total_gate", b_total_gate.mean().item(), global_step)
        writer.add_scalar("imagination/transition_std", pred_next_std.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=partial(
                Agent,
                latent_dim=args.latent_dim,
                model_hidden_dim=args.model_hidden_dim,
                model_min_std=args.model_min_std,
                model_max_std=args.model_max_std,
                use_done_model=args.use_done_model,
            ),
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


if __name__ == "__main__":
    main()
