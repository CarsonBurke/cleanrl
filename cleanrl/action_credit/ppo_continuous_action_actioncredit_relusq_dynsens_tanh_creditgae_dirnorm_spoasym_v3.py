# PPO + dynamics sensitivity credit-GAE + directional credit norm + asymmetric SPO + ReLU^2 v3.
#
# Keeps scalar PPO GAE/value learning as the trusted global signal. A learned
# one-step model predicts normalized-observation deltas and normalized rewards
# from bounded actions. The actor uses a tanh-squashed diagonal Gaussian with
# corrected log probabilities following the SAC/SB3 change-of-variables form.
# v3 keeps the SB3 gSDE/SAC numerical convention for the tanh Jacobian and
# changes credit normalization: the dynamics model supplies per-action
# direction within each sample, while scalar GAE supplies timestep magnitude.
# This avoids rare costate spikes globally shrinking almost all credit to zero.
# During PPO updates, the frozen model runs a backward costate scan through the
# rollout using the same gamma/lambda horizon as scalar GAE. Each timestep gets
# q_a(t), the local action sensitivity of reward plus value/future costate, and
# only that current q_a(t) is converted into a Gaussian score residual.
#
# Hypothesis: scalar returns identify temporal credit but not action-coordinate
# credit. A short-horizon dynamics model supplies local counterfactual geometry;
# PPO still controls the magnitude through scalar GAE; costate-GAE only
# redistributes pressure across action dimensions. This builds on the best
# tanh+SPO branch while testing whether GAE-style temporal credit improves the
# model sensitivity residual.
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
    """value-loss clipping coefficient; SPO replaces policy ratio clipping"""
    spo_eps_low: float = 0.40
    """SPO bound when ratio drift opposes the advantage direction"""
    spo_eps_high: float = 0.56
    """SPO bound when ratio drift agrees with the advantage direction"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    dynamics_learning_rate: float = 1e-3
    """learning rate for the one-step dynamics/reward model"""
    dynamics_update_epochs: int = 4
    """number of dynamics model passes over each rollout"""
    dynamics_num_minibatches: int = 8
    """number of minibatches for dynamics model training"""
    dynamics_reward_coef: float = 1.0
    """relative coefficient for reward prediction in dynamics model loss"""
    dynamics_grad_clip: float = 1.0
    """maximum norm for dynamics model gradients"""
    dynamics_credit_start: int = 4
    """first PPO iteration that uses dynamics-sensitivity action credit"""
    credit_resid_coef: float = 0.1
    """scale of the clipped dynamics-sensitivity credit-GAE actor residual"""
    credit_resid_clip: float = 1.5
    """per-action clip range after per-sample dynamics-credit normalization"""
    joint_kl_coef: float = 0.0
    """coefficient of an optional differentiable joint approximate-KL penalty"""
    joint_clip_coef: float = 0.0
    """coefficient of an optional squared joint clip-bound violation penalty"""
    target_kl: float = 0.03
    """the target KL divergence threshold"""

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


def inverse_tanh(x):
    eps = torch.finfo(x.dtype).eps
    x = x.clamp(-1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def tanh_log_prob_correction(gaussian_action, eps=1e-6):
    return torch.log(1.0 - torch.tanh(gaussian_action).square() + eps)


def finite_or_zero(x):
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class ReluSq(nn.Module):
    """f(x) = relu(x)^2."""

    def forward(self, x):
        return torch.relu(x).square()


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        self.critic_trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 64)),
            ReluSq(),
        )
        self.critic_value = layer_init(nn.Linear(64, 1), std=1.0)
        self.actor_trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 64)),
            ReluSq(),
        )
        self.actor_mean = layer_init(nn.Linear(64, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        critic_features = self.critic_trunk(x)
        return self.critic_value(critic_features)

    def get_action_and_value(self, x, action=None, gaussian_action=None):
        actor_features = self.actor_trunk(x)
        action_mean = self.actor_mean(actor_features)
        action_logstd = self.actor_logstd.clamp(-5.0, 2.0).expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if gaussian_action is None:
            if action is not None:
                gaussian_action = inverse_tanh(action)
                action = action.clamp(-1 + 1e-6, 1 - 1e-6)
            else:
                gaussian_action = probs.sample()
                action = torch.tanh(gaussian_action)
        elif action is None:
            action = torch.tanh(gaussian_action)
        else:
            action = action.clamp(-1 + 1e-6, 1 - 1e-6)
        logprob = probs.log_prob(gaussian_action) - tanh_log_prob_correction(gaussian_action)
        entropy_estimate = -logprob
        return action, gaussian_action, logprob, entropy_estimate, self.get_value(x), action_mean, action_std


class DynamicsModel(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + action_dim, 256)),
            ReluSq(),
            layer_init(nn.Linear(256, 256)),
            ReluSq(),
            layer_init(nn.Linear(256, obs_dim + 1), std=0.01),
        )
        self.obs_dim = obs_dim

    def forward(self, obs, action):
        pred = self.net(torch.cat([obs, action], dim=1))
        pred_delta = pred[:, : self.obs_dim]
        pred_reward = pred[:, self.obs_dim]
        return pred_delta, pred_reward


def snapshot_requires_grad(module):
    return [param.requires_grad for param in module.parameters()]


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def restore_requires_grad(module, flags):
    for param, requires_grad in zip(module.parameters(), flags):
        param.requires_grad_(requires_grad)


def compute_dynamics_credit_costate(
    agent,
    dynamics_model,
    obs,
    actions,
    gaussian_actions,
    action_means,
    action_stds,
    credit_continuations,
    credit_active,
    gamma,
    gae_lambda,
):
    credit_residuals = torch.zeros_like(actions)
    next_costate = torch.zeros_like(obs[0])
    agent_requires_grad = snapshot_requires_grad(agent)
    dynamics_requires_grad = snapshot_requires_grad(dynamics_model)
    set_requires_grad(agent, False)
    set_requires_grad(dynamics_model, False)
    try:
        for t in reversed(range(obs.shape[0])):
            mb_obs = obs[t].detach().clone().requires_grad_(True)
            mb_actions = actions[t].detach().clone().requires_grad_(True)
            pred_delta, pred_reward = dynamics_model(mb_obs, mb_actions)
            pred_next_obs = (mb_obs + pred_delta).clamp(-10.0, 10.0)
            mb_continuations = credit_continuations[t]
            current_value = agent.get_value(mb_obs).view(-1)
            next_value = agent.get_value(pred_next_obs).view(-1)
            future_costate_value = (next_costate.detach() * pred_next_obs).sum(1)
            objective = (
                pred_reward
                + gamma * mb_continuations * (next_value + gae_lambda * future_costate_value)
                - current_value
            )
            state_grad, action_grad = torch.autograd.grad(objective.sum(), (mb_obs, mb_actions))
            state_grad = finite_or_zero(state_grad)
            action_grad = finite_or_zero(action_grad)
            squash_grad = 1.0 - actions[t].square()
            score = (gaussian_actions[t] - action_means[t]) / (action_stds[t] + 1e-8)
            mb_credit = action_grad * squash_grad * action_stds[t] * score
            mb_credit = finite_or_zero(mb_credit)
            mb_credit = mb_credit - mb_credit.mean(1, keepdim=True)
            mb_credit = mb_credit * credit_active[t].unsqueeze(1)
            credit_residuals[t] = mb_credit.detach()
            next_costate = state_grad.detach() * credit_active[t].unsqueeze(1)
    finally:
        restore_requires_grad(agent, agent_requires_grad)
        restore_requires_grad(dynamics_model, dynamics_requires_grad)

    raw_credit_residual_rms = finite_or_zero(credit_residuals).square().mean().sqrt()
    return credit_residuals, raw_credit_residual_rms


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

    obs_dim = np.array(envs.single_observation_space.shape).prod()
    action_dim = np.prod(envs.single_action_space.shape)
    agent = Agent(envs).to(device)
    dynamics_model = DynamicsModel(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    dynamics_optimizer = optim.Adam(dynamics_model.parameters(), lr=args.dynamics_learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    model_next_observations = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    gaussian_actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    model_next_valids = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    action_means = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    action_stds = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)

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
                action, gaussian_action, logprob, _, value, action_mean, action_std = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            gaussian_actions[step] = gaussian_action
            logprobs[step] = logprob
            action_means[step] = action_mean
            action_stds[step] = action_std

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            model_next_obs = np.array(next_obs, copy=True)
            model_next_valid = np.ones(args.num_envs, dtype=np.float32)
            done_mask = np.logical_or(terminations, truncations)
            model_next_valid[done_mask] = 0.0
            if "final_observation" in infos:
                final_observations = infos["final_observation"]
                final_observation_mask = infos.get("_final_observation", np.ones(args.num_envs, dtype=bool))
                for env_idx, has_final_observation in enumerate(final_observation_mask):
                    if has_final_observation and final_observations[env_idx] is not None:
                        model_next_obs[env_idx] = final_observations[env_idx]
                        model_next_valid[env_idx] = 1.0
            next_done = done_mask
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            model_next_observations[step] = torch.tensor(model_next_obs, dtype=torch.float32, device=device)
            transition_dones[step] = next_done
            model_next_valids[step] = torch.tensor(model_next_valid, dtype=torch.float32, device=device)

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
            norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_model_next_observations = model_next_observations.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape((-1,) + envs.single_action_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_gaussian_actions = gaussian_actions.reshape((-1,) + envs.single_action_space.shape)
        b_action_means = action_means.reshape((-1,) + envs.single_action_space.shape)
        b_action_stds = action_stds.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_norm_advantages = norm_advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_rewards = rewards.reshape(-1)
        b_transition_dones = transition_dones.reshape(-1)
        b_model_next_valids = model_next_valids.reshape(-1)
        b_credit_continuations = 1.0 - b_transition_dones
        b_credit_active = b_model_next_valids

        # Train the local dynamics/reward model on real rollout transitions.
        dynamics_loss = torch.tensor(0.0, device=device)
        dynamics_delta_loss = torch.tensor(0.0, device=device)
        dynamics_reward_loss = torch.tensor(0.0, device=device)
        valid_model_inds = np.flatnonzero(b_model_next_valids.detach().cpu().numpy() > 0.5)
        if len(valid_model_inds) > 0:
            dynamics_minibatch_size = max(1, len(valid_model_inds) // args.dynamics_num_minibatches)
            for _ in range(args.dynamics_update_epochs):
                np.random.shuffle(valid_model_inds)
                for start in range(0, len(valid_model_inds), dynamics_minibatch_size):
                    end = start + dynamics_minibatch_size
                    mb_inds = valid_model_inds[start:end]
                    mb_obs = b_obs[mb_inds]
                    mb_actions = b_actions[mb_inds]
                    target_delta = b_model_next_observations[mb_inds] - mb_obs
                    target_reward = b_rewards[mb_inds]
                    pred_delta, pred_reward = dynamics_model(mb_obs, mb_actions)
                    dynamics_delta_loss = 0.5 * (pred_delta - target_delta).square().mean()
                    dynamics_reward_loss = 0.5 * (pred_reward - target_reward).square().mean()
                    dynamics_loss = dynamics_delta_loss + args.dynamics_reward_coef * dynamics_reward_loss
                    if not torch.isfinite(dynamics_loss):
                        continue

                    dynamics_optimizer.zero_grad()
                    dynamics_loss.backward()
                    dynamics_grad_norm = nn.utils.clip_grad_norm_(
                        dynamics_model.parameters(),
                        args.dynamics_grad_clip,
                    )
                    if torch.isfinite(dynamics_grad_norm):
                        dynamics_optimizer.step()

        if iteration >= args.dynamics_credit_start:
            credit_residuals, raw_credit_residual_rms = compute_dynamics_credit_costate(
                agent,
                dynamics_model,
                obs,
                actions,
                gaussian_actions,
                action_means,
                action_stds,
                b_credit_continuations.reshape(args.num_steps, args.num_envs),
                b_credit_active.reshape(args.num_steps, args.num_envs),
                args.gamma,
                args.gae_lambda,
            )
            if raw_credit_residual_rms.item() < 1e-6:
                b_credit_residuals = torch.zeros_like(b_actions)
            else:
                credit_residuals = credit_residuals - credit_residuals.mean(2, keepdim=True)
                sample_rms = credit_residuals.square().mean(2, keepdim=True).sqrt()
                credit_residuals = torch.where(
                    sample_rms > 1e-6,
                    credit_residuals / (sample_rms + 1e-8),
                    torch.zeros_like(credit_residuals),
                )
                max_abs = credit_residuals.abs().amax(2, keepdim=True)
                credit_residuals = credit_residuals / torch.clamp(max_abs / args.credit_resid_clip, min=1.0)
                credit_residuals = credit_residuals - credit_residuals.mean(2, keepdim=True)
                credit_residuals = finite_or_zero(credit_residuals)
                b_credit_residuals = credit_residuals.reshape((-1,) + envs.single_action_space.shape)
        else:
            b_credit_residuals = torch.zeros_like(b_actions)
            raw_credit_residual_rms = torch.tensor(0.0, device=device)
        dynamics_credit_confidence = (1.0 / (1.0 + dynamics_loss.detach())).clamp(0.0, 1.0)
        b_credit_residuals = b_credit_residuals * dynamics_credit_confidence

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        action_clipfracs = []
        approx_kls = []
        cleanrl_approx_kls = []
        scalar_spo_penalty = torch.zeros((), device=device)
        credit_spo_penalty = torch.zeros((), device=device)
        stop_update = False
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, newvalue, _, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                    b_gaussian_actions[mb_inds],
                )
                logratio = finite_or_zero(newlogprob - b_logprobs[mb_inds])
                bounded_logratio = logratio.clamp(-20.0, 20.0)
                ratio = bounded_logratio.exp()
                joint_logratio = bounded_logratio.sum(1).clamp(-20.0, 20.0)
                joint_ratio = joint_logratio.exp()
                joint_approx_kl = ((ratio - 1) - bounded_logratio).sum(1)

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-joint_logratio).mean()
                    approx_kl = joint_approx_kl.mean()
                    cleanrl_approx_kl = ((joint_ratio - 1) - joint_logratio).mean()
                    approx_kls.append(approx_kl.item())
                    cleanrl_approx_kls.append(cleanrl_approx_kl.item())
                    clipfracs += [((joint_ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                    action_clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.norm_adv:
                    mb_scalar_advantages = b_norm_advantages[mb_inds]
                else:
                    mb_scalar_advantages = b_advantages[mb_inds]

                mb_credit_residuals = b_credit_residuals[mb_inds] * mb_scalar_advantages.abs().unsqueeze(1)

                # Asymmetric SPO policy loss. The high bound is used when the
                # ratio drift agrees with the advantage sign, otherwise low.
                joint_ratio_diff = joint_ratio - 1.0
                scalar_eps = torch.where(
                    (mb_scalar_advantages * joint_ratio_diff) > 0,
                    torch.full_like(mb_scalar_advantages, args.spo_eps_high),
                    torch.full_like(mb_scalar_advantages, args.spo_eps_low),
                )
                scalar_spo_penalty = (
                    mb_scalar_advantages.abs() * joint_ratio_diff.square() / (2.0 * scalar_eps)
                )
                scalar_pg_loss = -(mb_scalar_advantages * joint_ratio - scalar_spo_penalty).mean()

                credit_advantages = mb_credit_residuals.detach()
                action_ratio_diff = ratio - 1.0
                credit_eps = torch.where(
                    (credit_advantages * action_ratio_diff) > 0,
                    torch.full_like(credit_advantages, args.spo_eps_high),
                    torch.full_like(credit_advantages, args.spo_eps_low),
                )
                credit_spo_penalty = credit_advantages.abs() * action_ratio_diff.square() / (2.0 * credit_eps)
                credit_pg_loss = -(
                    credit_advantages * ratio - credit_spo_penalty
                ).sum(1).mean() / action_dim
                pg_loss = scalar_pg_loss + args.credit_resid_coef * credit_pg_loss

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

                entropy_loss = entropy.sum(1).mean()
                joint_clip_violation = torch.relu((joint_ratio - 1.0).abs() - args.clip_coef)
                joint_clip_loss = joint_clip_violation.square().mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                if args.joint_kl_coef != 0.0:
                    loss = loss + joint_approx_kl.mean() * args.joint_kl_coef
                if args.joint_clip_coef != 0.0:
                    loss = loss + joint_clip_loss * args.joint_clip_coef
                if not torch.isfinite(loss):
                    stop_update = True
                    break

                optimizer.zero_grad()
                loss.backward()
                agent_grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                if torch.isfinite(agent_grad_norm):
                    optimizer.step()
                else:
                    stop_update = True
                    break

                if args.target_kl is not None and approx_kl > args.target_kl:
                    stop_update = True
                    break

            if stop_update:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/scalar_policy_loss", scalar_pg_loss.item(), global_step)
        writer.add_scalar("losses/credit_policy_loss", credit_pg_loss.item(), global_step)
        writer.add_scalar("losses/scalar_spo_penalty", scalar_spo_penalty.mean().item(), global_step)
        writer.add_scalar("losses/credit_spo_penalty", credit_spo_penalty.mean().item(), global_step)
        writer.add_scalar("losses/dynamics_loss", dynamics_loss.item(), global_step)
        writer.add_scalar("losses/dynamics_delta_loss", dynamics_delta_loss.item(), global_step)
        writer.add_scalar("losses/dynamics_reward_loss", dynamics_reward_loss.item(), global_step)
        writer.add_scalar("losses/dynamics_credit_confidence", dynamics_credit_confidence.item(), global_step)
        writer.add_scalar("losses/raw_credit_residual_rms", raw_credit_residual_rms.item(), global_step)
        writer.add_scalar("losses/credit_residual_rms", b_credit_residuals.square().mean().sqrt().item(), global_step)
        writer.add_scalar("losses/credit_residual_abs_mean", b_credit_residuals.abs().mean().item(), global_step)
        writer.add_scalar("losses/squashed_action_saturation_frac", (b_actions.abs() > 0.999).float().mean().item(), global_step)
        writer.add_scalar("losses/joint_clip_loss", joint_clip_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl_update_mean", np.mean(approx_kls), global_step)
        writer.add_scalar("losses/approx_kl_update_max", np.max(approx_kls), global_step)
        writer.add_scalar("losses/cleanrl_approx_kl", cleanrl_approx_kl.item(), global_step)
        writer.add_scalar("losses/cleanrl_approx_kl_update_mean", np.mean(cleanrl_approx_kls), global_step)
        writer.add_scalar("losses/cleanrl_approx_kl_update_max", np.max(cleanrl_approx_kls), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/action_clipfrac", np.mean(action_clipfracs), global_step)
        writer.add_scalar("losses/joint_ratio_mean", joint_ratio.mean().item(), global_step)
        writer.add_scalar("losses/joint_ratio_std", joint_ratio.std().item(), global_step)
        writer.add_scalar("losses/joint_ratio_min", joint_ratio.min().item(), global_step)
        writer.add_scalar("losses/joint_ratio_max", joint_ratio.max().item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
