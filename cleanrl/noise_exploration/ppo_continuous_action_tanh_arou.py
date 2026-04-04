# Tanh-AROU: Tanh-squashed Autoregressive OU Noise for PPO
#
# Major iteration motivation:
# - Prior correlated-exploration variants (e.g., OU-gSDE, MSOU-gSDE) keep PPO's
#   per-step log_prob as a marginal diagonal Gaussian, which means correlation
#   parameters (rho, mixing weights) receive ~no policy-gradient signal.
# - Standard CleanRL PPO uses ClipAction wrapper; large exploration can get
#   clipped by the env wrapper but still be scored under the unclipped Gaussian,
#   creating a harmful log_prob/environment mismatch (especially on Hopper/Walker).
#
# Key idea:
# Use an autoregressive Gaussian in *standardized noise space* z_t (OU/AR(1)),
# then apply a tanh squash to produce bounded actions. Compute exact log_prob of
# the squashed action via change-of-variables, so PPO gradients train rho.
#
# Dynamics:
#   z_t | (s_t, z_{t-1}) ~ N(rho(s_t) * z_{t-1}, (1-rho(s_t)^2) I)
#   y_t = mean(s_t) + std * z_t
#   a_t = tanh(y_t)  (bounded in (-1, 1))
#
# We treat (s_t, z_{t-1}) as the policy-conditioning variables and store z_{t-1}
# alongside transitions; value function also conditions on z_{t-1}.
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

    # Tanh-AROU specific arguments
    rho_min: float = 0.0
    """minimum OU correlation coefficient"""
    rho_max: float = 0.995
    """maximum OU correlation coefficient"""
    rho_init: float = 0.85
    """initial OU correlation coefficient"""
    rho_state_dependent: bool = True
    """learn a state-dependent rho(s) (else use a single scalar parameter)"""
    reset_z_on_done: bool = True
    """reset OU state for envs that ended an episode"""
    eps: float = 1e-6
    """numerical epsilon for stability (atanh + jacobian)"""

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


def atanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class Agent(nn.Module):
    def __init__(self, envs, rho_min: float, rho_max: float, rho_init: float, rho_state_dependent: bool, eps: float):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.rho_state_dependent = bool(rho_state_dependent)
        self.eps = float(eps)

        rho_frac = (rho_init - rho_min) / (rho_max - rho_min) if rho_max > rho_min else 0.5
        rho_frac = float(np.clip(rho_frac, 1e-6, 1.0 - 1e-6))
        rho_bias_init = math.log(rho_frac / (1.0 - rho_frac))
        self.rho_bias = nn.Parameter(torch.tensor([[rho_bias_init]], dtype=torch.float32))

        in_dim = obs_dim + action_dim  # condition on z_{t-1}

        self.critic = nn.Sequential(
            layer_init(nn.Linear(in_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_latent = nn.Sequential(
            layer_init(nn.Linear(in_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor_mean = layer_init(nn.Linear(64, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        if self.rho_state_dependent:
            self.actor_rho = layer_init(nn.Linear(64, 1), std=0.01)
        else:
            self.actor_rho = None

    def _augment(self, x: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, z_prev], dim=-1)

    def _rho(self, latent: torch.Tensor) -> torch.Tensor:
        if self.actor_rho is None:
            rho_logits = self.rho_bias.expand(latent.shape[0], 1)
        else:
            rho_logits = self.actor_rho(latent) + self.rho_bias
        rho = self.rho_min + (self.rho_max - self.rho_min) * torch.sigmoid(rho_logits)
        return torch.clamp(rho, min=self.rho_min, max=self.rho_max)  # (B,1)

    def get_value(self, x: torch.Tensor, z_prev: torch.Tensor):
        return self.critic(self._augment(x, z_prev=z_prev))

    def _dist_params(self, x: torch.Tensor, z_prev: torch.Tensor):
        latent = self.actor_latent(self._augment(x, z_prev=z_prev))
        mean = self.actor_mean(latent)
        std = torch.exp(self.actor_logstd.expand_as(mean)).clamp(min=self.eps)
        rho = self._rho(latent).expand(-1, self.action_dim)
        return mean, std, rho

    def sample(self, x: torch.Tensor, z_prev: torch.Tensor):
        mean, std, rho = self._dist_params(x, z_prev=z_prev)
        z = rho * z_prev + torch.sqrt(torch.clamp(1.0 - rho**2, min=self.eps)) * torch.randn_like(mean)
        y = mean + std * z
        a = torch.tanh(y)
        return a, z

    def logprob_entropy_value(self, x: torch.Tensor, z_prev: torch.Tensor, action: torch.Tensor):
        mean, std, rho = self._dist_params(x, z_prev=z_prev)

        a = torch.clamp(action, min=-1.0 + self.eps, max=1.0 - self.eps)
        y = atanh(a)

        loc_y = mean + std * (rho * z_prev)
        scale_y = std * torch.sqrt(torch.clamp(1.0 - rho**2, min=self.eps))

        base = Normal(loc_y, scale_y)
        logp_y = base.log_prob(y).sum(-1)
        log_det = torch.log(torch.clamp(1.0 - a**2, min=self.eps)).sum(-1)
        logp = logp_y - log_det

        # Approx entropy: entropy of base (pre-squash) Normal.
        entropy = base.entropy().sum(-1)
        value = self.get_value(x, z_prev=z_prev)
        return logp, entropy, value


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

    agent = Agent(
        envs,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        rho_init=args.rho_init,
        rho_state_dependent=args.rho_state_dependent,
        eps=args.eps,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    z_prev_buf = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    z_prev = torch.zeros((args.num_envs,) + envs.single_action_space.shape, device=device)

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
            z_prev_buf[step] = z_prev

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, z_new = agent.sample(next_obs, z_prev=z_prev)
                logprob, _, value = agent.logprob_entropy_value(next_obs, z_prev=z_prev, action=action)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            z_prev = z_new

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done_np).to(device)

            if args.reset_z_on_done:
                z_prev = z_prev * (1.0 - next_done).unsqueeze(-1)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, z_prev=z_prev).reshape(1, -1)
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
        b_z_prev = z_prev_buf.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
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

                newlogprob, entropy, newvalue = agent.logprob_entropy_value(
                    b_obs[mb_inds], z_prev=b_z_prev[mb_inds], action=b_actions[mb_inds]
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

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
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
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()

