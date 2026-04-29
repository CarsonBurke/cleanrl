# Introspective Advantage Critic PPO v3
#
# Ablation hypothesis:
# - Keep V(s) for TD(lambda)/GAE, but train an action-conditional self-model
#   that predicts both A(s, a) and its own log error variance.
# - Humans introspect by combining direct evidence with confidence-weighted
#   self-estimates. Here raw GAE is the evidence; A(s,a) is the self-estimate.
# - The policy blends toward predicted A(s,a) only when the critic reports low
#   uncertainty, otherwise it falls back toward raw GAE.
# - Regularize E_{a~pi(.|s)}[A(s, a)] ~= 0 so introspection stays action-relative.
#
# The goal is a more ambitious but safer advantage critic: denoise when it has a
# coherent internal model, defer to fresh trajectory evidence when it does not.
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
    """Toggle learning rate annealing for policy and advantage critic networks"""
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
    """Toggles whether or not to use a clipped loss for the bootstrap value function."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the bootstrap value loss"""
    adv_coef: float = 0.5
    """coefficient of the advantage critic loss"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    advantage_pretrain_epochs: int = 2
    """critic-only epochs on the current rollout before freezing policy advantages"""
    normalize_advantage_targets: bool = False
    """normalize GAE targets before fitting the advantage critic"""
    min_policy_advantage_mix: float = 0.05
    """minimum learned-advantage weight when the introspective critic is uncertain"""
    max_policy_advantage_mix: float = 0.85
    """maximum learned-advantage weight when the introspective critic is confident"""
    introspection_logvar_pivot: float = 0.0
    """log variance where learned and raw advantages receive midpoint weighting"""
    introspection_confidence_gain: float = 1.25
    """sharpness of confidence weighting from predicted log variance"""
    advantage_logvar_min: float = -6.0
    """minimum predicted log variance for heteroscedastic advantage loss"""
    advantage_logvar_max: float = 4.0
    """maximum predicted log variance for heteroscedastic advantage loss"""
    advantage_logvar_init: float = 1.0
    """initial log variance bias, making early policy updates defer toward raw GAE"""
    advantage_zero_mean_coef: float = 0.05
    """coefficient for E_pi[A(s,a)]^2 regularization"""
    advantage_zero_mean_samples: int = 4
    """number of policy action samples per state for zero-mean advantage regularization"""

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


class Agent(nn.Module):
    def __init__(self, envs, args=None):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)
        advantage_logvar_init = 1.0 if args is None else args.advantage_logvar_init
        self.value_baseline = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.advantage_backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim + action_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.advantage_mean = layer_init(nn.Linear(64, 1), std=0.01)
        self.advantage_logvar = layer_init(nn.Linear(64, 1), std=0.01, bias_const=advantage_logvar_init)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        return self.value_baseline(x)

    def get_action_dist(self, x):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std)

    def get_advantage(self, x, action):
        advantage_mean, _ = self.get_advantage_stats(x, action)
        return advantage_mean

    def get_advantage_stats(self, x, action):
        features = self.advantage_backbone(torch.cat((x, action), dim=1))
        return self.advantage_mean(features), self.advantage_logvar(features)

    def get_action_and_value(self, x, action=None):
        probs = self.get_action_dist(x)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_advantage(x, action)


def advantage_zero_mean_loss(agent, obs, sample_count):
    if sample_count <= 0:
        return torch.zeros((), device=obs.device)

    with torch.no_grad():
        probs = agent.get_action_dist(obs)
        sampled_actions = probs.sample((sample_count,))

    repeated_obs = obs.unsqueeze(0).expand(sample_count, *obs.shape).reshape(-1, obs.shape[-1])
    flat_actions = sampled_actions.reshape(-1, sampled_actions.shape[-1])
    sampled_advantages = agent.get_advantage(repeated_obs, flat_actions).view(sample_count, obs.shape[0])
    return sampled_advantages.mean(dim=0).pow(2).mean()


def clamp_advantage_logvar(logvar, args):
    return torch.clamp(logvar, args.advantage_logvar_min, args.advantage_logvar_max)


def advantage_introspection_loss(advantage_mean, advantage_logvar, target, args):
    advantage_logvar = clamp_advantage_logvar(advantage_logvar, args)
    residual = advantage_mean - target
    return 0.5 * (torch.exp(-advantage_logvar) * residual.pow(2) + advantage_logvar).mean()


def introspective_policy_advantages(advantage_mean, advantage_logvar, raw_advantages, args):
    advantage_logvar = clamp_advantage_logvar(advantage_logvar, args)
    confidence = torch.sigmoid(args.introspection_confidence_gain * (args.introspection_logvar_pivot - advantage_logvar))
    mix = args.min_policy_advantage_mix + (args.max_policy_advantage_mix - args.min_policy_advantage_mix) * confidence
    policy_advantages = mix * advantage_mean + (1.0 - mix) * raw_advantages
    return policy_advantages, mix, confidence


if __name__ == "__main__":
    args = tyro.cli(Args)
    if not 0.0 <= args.min_policy_advantage_mix <= args.max_policy_advantage_mix <= 1.0:
        raise ValueError("policy advantage mix bounds must satisfy 0 <= min <= max <= 1")
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
                action, logprob, _, _ = agent.get_action_and_value(next_obs)
                value = agent.get_value(next_obs)
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

        # Bootstrap GAE targets with V(s), then distill them into A(s, a).
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
            advantage_targets = advantages.clone()
            if args.normalize_advantage_targets:
                advantage_targets = (advantage_targets - advantage_targets.mean()) / (advantage_targets.std() + 1e-8)

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantage_targets = advantage_targets.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        # Fit A(s, a) to the current rollout before freezing policy advantages.
        b_inds = np.arange(args.batch_size)
        adv_loss = torch.zeros((), device=device)
        zero_mean_loss = torch.zeros((), device=device)
        v_loss = torch.zeros((), device=device)
        for pretrain_epoch in range(args.advantage_pretrain_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                newadv, newlogvar = agent.get_advantage_stats(b_obs[mb_inds], b_actions[mb_inds])
                newadv = newadv.view(-1)
                newlogvar = newlogvar.view(-1)
                adv_loss = advantage_introspection_loss(newadv, newlogvar, b_advantage_targets[mb_inds], args)
                zero_mean_loss = advantage_zero_mean_loss(agent, b_obs[mb_inds], args.advantage_zero_mean_samples)
                loss = adv_loss + args.advantage_zero_mean_coef * zero_mean_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        with torch.no_grad():
            b_predicted_advantages, b_predicted_logvars = agent.get_advantage_stats(b_obs, b_actions)
            b_predicted_advantages = b_predicted_advantages.view(-1)
            b_predicted_logvars = b_predicted_logvars.view(-1)
            b_advantages, b_policy_advantage_mix, b_advantage_confidence = introspective_policy_advantages(
                b_predicted_advantages,
                b_predicted_logvars,
                b_advantage_targets,
                args,
            )

        # Optimizing the policy and advantage critic
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newadv = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
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

                # Advantage critic loss
                newadv = newadv.view(-1)
                newlogvar = agent.get_advantage_stats(b_obs[mb_inds], b_actions[mb_inds])[1].view(-1)
                adv_loss = advantage_introspection_loss(newadv, newlogvar, b_advantage_targets[mb_inds], args)
                zero_mean_loss = advantage_zero_mean_loss(agent, b_obs[mb_inds], args.advantage_zero_mean_samples)

                # Bootstrap value loss
                newvalue = agent.get_value(b_obs[mb_inds]).view(-1)
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
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + v_loss * args.vf_coef
                    + adv_loss * args.adv_coef
                    + zero_mean_loss * args.advantage_zero_mean_coef
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        with torch.no_grad():
            final_advantages = agent.get_advantage(b_obs, b_actions).view(-1)
        y_pred, y_true = final_advantages.cpu().numpy(), b_advantage_targets.cpu().numpy()
        var_y = np.var(y_true)
        advantage_explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        value_explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/advantage_loss", adv_loss.item(), global_step)
        writer.add_scalar("losses/advantage_zero_mean_loss", zero_mean_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/value_explained_variance", value_explained_var, global_step)
        writer.add_scalar("losses/advantage_explained_variance", advantage_explained_var, global_step)
        writer.add_scalar("losses/policy_advantage_std", b_advantages.std().item(), global_step)
        writer.add_scalar("losses/policy_advantage_mix", b_policy_advantage_mix.mean().item(), global_step)
        writer.add_scalar("losses/advantage_confidence", b_advantage_confidence.mean().item(), global_step)
        writer.add_scalar("losses/target_advantage_std", b_advantage_targets.std().item(), global_step)
        writer.add_scalar("losses/predicted_advantage_std", b_predicted_advantages.std().item(), global_step)
        writer.add_scalar("losses/predicted_advantage_logvar", b_predicted_logvars.mean().item(), global_step)
        writer.add_scalar("losses/raw_return_mean", b_returns.mean().item(), global_step)
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
