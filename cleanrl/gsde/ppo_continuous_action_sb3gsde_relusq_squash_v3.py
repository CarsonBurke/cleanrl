# PPO + squashed SB3-style gSDE + ReLUSq body v3
#
# Hypothesis:
# Stable-Baselines3 generalized State Dependent Exploration is a stronger
# exploration baseline than state-independent diagonal Gaussian PPO for MuJoCo.
# This variant keeps CleanRL PPO intact but replaces the actor distribution with
# SB3's latent-dependent exploration matrix: rollout actions use
# a = mean(s) + latent(s) @ E, and PPO log-probs use the corresponding marginal
# diagonal Normal with Var[a_i | s] = sum_j latent_j(s)^2 std_ji^2.
#
# Novelty in this repo:
# - faithful SB3 gSDE mechanics in a single CleanRL file
# - per-step exploration-matrix resampling by default (sde_sample_freq=1)
# - ReLUSq actor/critic body
# - tanh-squashed gSDE actions with exact log-prob correction
# - LayerNorm -> tanh -> fixed scale before gSDE variance/noise computation
# - SB3 default feature-detach behavior
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
    gsde_log_std_init: float = -2.0
    """initial value for the gSDE log standard deviation matrix"""
    full_std: bool = True
    """use a latent_dim x action_dim std matrix instead of latent_dim x 1"""
    use_expln: bool = False
    """use SB3 expln() transform instead of exp() for gSDE std"""
    learn_sde_features: bool = False
    """allow std/noise gradients into actor latent features; SB3 on-policy default is False"""
    sde_sample_freq: int = 1
    """sample a new gSDE exploration matrix every n rollout steps; 1 = every step, -1 = rollout start only"""
    sde_latent_scale: float = 0.5
    """fixed scale applied after LayerNorm+tanh before gSDE variance/noise computation"""

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


class ReluSq(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


class StateDependentNoiseDistribution:
    def __init__(
        self,
        action_dim,
        latent_sde_dim,
        full_std=True,
        use_expln=False,
        learn_features=False,
        squash_output=True,
        epsilon=1e-6,
    ):
        self.action_dim = int(action_dim)
        self.latent_sde_dim = int(latent_sde_dim)
        self.full_std = bool(full_std)
        self.use_expln = bool(use_expln)
        self.learn_features = bool(learn_features)
        self.squash_output = bool(squash_output)
        self.epsilon = float(epsilon)
        self.exploration_mat = None
        self.exploration_matrices = None

    def get_std(self, log_std):
        if self.use_expln:
            below_threshold = torch.exp(log_std) * (log_std <= 0)
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            std = torch.exp(log_std)

        if self.full_std:
            return std
        return torch.ones(self.latent_sde_dim, self.action_dim, device=log_std.device, dtype=log_std.dtype) * std

    def sample_weights(self, log_std, batch_size=1):
        std = self.get_std(log_std)
        weights_dist = Normal(torch.zeros_like(std), std)
        self.exploration_mat = weights_dist.rsample()
        self.exploration_matrices = weights_dist.rsample((batch_size,))

    def _latent_for_sde(self, latent_sde):
        return latent_sde if self.learn_features else latent_sde.detach()

    def get_distribution(self, mean_actions, log_std, latent_sde):
        latent_sde = self._latent_for_sde(latent_sde)
        variance = torch.mm(latent_sde.pow(2), self.get_std(log_std).pow(2))
        return Normal(mean_actions, torch.sqrt(variance + self.epsilon))

    def get_noise(self, latent_sde):
        latent_sde = self._latent_for_sde(latent_sde)
        if self.exploration_matrices is None:
            raise RuntimeError("gSDE exploration matrices are uninitialized; call reset_noise() first")
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return torch.mm(latent_sde, self.exploration_mat)
        return torch.bmm(latent_sde.unsqueeze(1), self.exploration_matrices).squeeze(1)

    def sample(self, mean_actions, latent_sde):
        return mean_actions + self.get_noise(latent_sde)

    def squash(self, gaussian_actions):
        return torch.tanh(gaussian_actions) if self.squash_output else gaussian_actions

    def unsquash(self, actions):
        if not self.squash_output:
            return actions
        actions = actions.clamp(-1.0 + self.epsilon, 1.0 - self.epsilon)
        return 0.5 * (torch.log1p(actions) - torch.log1p(-actions))

    def log_prob(self, distribution, actions):
        gaussian_actions = self.unsquash(actions)
        log_prob = distribution.log_prob(gaussian_actions).sum(1)
        if self.squash_output:
            log_prob -= torch.log(1.0 - actions.pow(2) + self.epsilon).sum(1)
        return log_prob


class Agent(nn.Module):
    def __init__(
        self,
        envs,
        full_std=True,
        use_expln=False,
        learn_sde_features=False,
        gsde_log_std_init=-2.0,
        sde_latent_scale=0.5,
    ):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        latent_dim = 64

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_latent = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 64)),
            ReluSq(),
        )
        self.actor_mean = layer_init(nn.Linear(latent_dim, action_dim), std=0.01)
        self.sde_latent_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.sde_latent_scale = float(sde_latent_scale)
        self.action_dist = StateDependentNoiseDistribution(
            action_dim=action_dim,
            latent_sde_dim=latent_dim,
            full_std=full_std,
            use_expln=use_expln,
            learn_features=learn_sde_features,
            squash_output=True,
        )
        self.register_buffer("gsde_full_std_flag", torch.tensor(float(full_std)))
        self.register_buffer("gsde_use_expln_flag", torch.tensor(float(use_expln)))
        self.register_buffer("gsde_learn_features_flag", torch.tensor(float(learn_sde_features)))
        if full_std:
            self.log_std = nn.Parameter(torch.ones(latent_dim, action_dim) * gsde_log_std_init)
        else:
            self.log_std = nn.Parameter(torch.ones(latent_dim, 1) * gsde_log_std_init)
        self.reset_noise(batch_size=1)

    def reset_noise(self, batch_size=1):
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def _has_compatible_noise(self, batch_size, device):
        matrices = self.action_dist.exploration_matrices
        return matrices is not None and matrices.shape[0] == batch_size and matrices.device == device

    def load_state_dict(self, state_dict, strict=True, assign=False):
        saved_log_std = state_dict.get("log_std")
        if saved_log_std is not None and saved_log_std.shape != self.log_std.shape:
            self.log_std = nn.Parameter(torch.empty_like(saved_log_std))

        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.action_dist.full_std = bool(self.gsde_full_std_flag.item())
        self.action_dist.use_expln = bool(self.gsde_use_expln_flag.item())
        self.action_dist.learn_features = bool(self.gsde_learn_features_flag.item())
        self.reset_noise(batch_size=1)
        return result

    def get_value(self, x):
        return self.critic(x)

    def get_sde_latent(self, actor_features):
        return torch.tanh(self.sde_latent_norm(actor_features)) * self.sde_latent_scale

    def get_action_and_value(self, x, action=None):
        actor_features = self.actor_latent(x)
        action_mean = self.actor_mean(actor_features)
        latent_sde = self.get_sde_latent(actor_features)
        probs = self.action_dist.get_distribution(action_mean, self.log_std, latent_sde)
        if action is None:
            if not self._has_compatible_noise(action_mean.shape[0], action_mean.device):
                self.reset_noise(batch_size=action_mean.shape[0])
            gaussian_action = self.action_dist.sample(action_mean, latent_sde)
            action = self.action_dist.squash(gaussian_action).detach()
        logprob = self.action_dist.log_prob(probs, action)
        entropy = -logprob
        return action, logprob, entropy, self.critic(x)


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
        full_std=args.full_std,
        use_expln=args.use_expln,
        learn_sde_features=args.learn_sde_features,
        gsde_log_std_init=args.gsde_log_std_init,
        sde_latent_scale=args.sde_latent_scale,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
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
    agent.reset_noise(batch_size=args.num_envs)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        agent.reset_noise(batch_size=args.num_envs)
        for step in range(0, args.num_steps):
            if args.sde_sample_freq > 0 and step % args.sde_sample_freq == 0:
                agent.reset_noise(batch_size=args.num_envs)

            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
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
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

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
