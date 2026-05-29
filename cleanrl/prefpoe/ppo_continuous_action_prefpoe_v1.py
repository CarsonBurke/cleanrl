# PrefPoE (Preference-of-Experts) on vanilla PPO — paper-faithful v1
# (Kept as-is so v2 has a baseline to diff against.)
# v1 collapsed: with β=1.0 the pref MLE term was 100-1000× the PPO clip loss,
# σ_pref shrank 4× faster than σ_main, PoE then made fused≈pref, approx_kl
# exploded to >100, and explained_variance crashed. v2 rebalances β/α.
#
# Reference: arXiv:2511.08241 "Advantage-Guided Preference Fusion"
# Reference impl: prefpoe/vis.py (eval only; training inferred from paper).
#
# Key ideas:
#   1. Two diagonal-Gaussian heads share a Tanh MLP backbone with the critic:
#        π_main   = N(tanh(W_main φ),  exp(logstd_main))
#        π_pref   = N(tanh(W_pref φ),  exp(logstd_pref).clamp(1e-3, 2.0))
#      logstd parameters are global (per-action-dim, not state-conditioned).
#      Both initialized to -1.0 (σ ≈ 0.37) per the reference.
#
#   2. Product-of-Experts (Gaussian) fusion (diagonal):
#        1/σ_f² = 1/σ_m² + 1/σ_p²
#        μ_f    = σ_f² · (μ_m/σ_m² + μ_p/σ_p²)
#      Sampling, log-prob, and entropy ALL evaluated under π_f — the fused
#      distribution is the only behavioral policy; both heads see PPO gradients
#      through it.
#
#   3. Advantage-guided preference loss (the "AGPF" the paper names):
#        L_pref = -β · E_b[ A_norm(s,a) · log π_pref(a|s) ] - α · E_b[ H(π_pref) ]
#      A_norm is the per-batch advantage normalization (mean 0, std 1).
#      This pushes π_pref to assign high probability to high-advantage actions,
#      while H(π_pref) keeps it from collapsing. It's a separate gradient signal
#      from the PPO clip — the PPO loss flows through π_f; L_pref flows only
#      into the pref head.
#
# Hypothesis: faithfully matching vis.py architecture + adding the advantage-
# guided preference MLE term reproduces the paper's ~5600 HalfCheetah-v4 result.
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
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    # PrefPoE-specific
    pref_logstd_init: float = -1.0
    main_logstd_init: float = -1.0
    pref_std_min: float = 1e-3
    pref_std_max: float = 2.0
    fused_std_min: float = 1e-3
    fused_std_max: float = 2.0
    pref_loss_coef: float = 1.0  # β: advantage-weighted MLE on π_pref
    pref_ent_coef: float = 0.01  # α: entropy bonus on π_pref

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


class Agent(nn.Module):
    def __init__(self, envs, args: "Args"):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.args = args

        self.backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.critic_head = layer_init(nn.Linear(64, 1), std=1.0)
        self.actor_mean = layer_init(nn.Linear(64, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), args.main_logstd_init))
        self.preference_mean = layer_init(nn.Linear(64, action_dim), std=0.01)
        self.preference_logstd = nn.Parameter(torch.full((1, action_dim), args.pref_logstd_init))

    def get_value(self, x):
        return self.critic_head(self.backbone(x))

    def _dists(self, x):
        features = self.backbone(x)
        action_mean = torch.tanh(self.actor_mean(features))
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        pref_mean = torch.tanh(self.preference_mean(features))
        pref_std = torch.exp(self.preference_logstd.expand_as(pref_mean)).clamp(
            min=self.args.pref_std_min, max=self.args.pref_std_max
        )
        value = self.critic_head(features)

        action_prec = 1.0 / (action_std.square() + 1e-8)
        pref_prec = 1.0 / (pref_std.square() + 1e-8)
        combined_prec = action_prec + pref_prec
        combined_mean = (action_mean * action_prec + pref_mean * pref_prec) / combined_prec
        combined_std = (1.0 / torch.sqrt(combined_prec + 1e-8)).clamp(
            min=self.args.fused_std_min, max=self.args.fused_std_max
        )

        fused_dist = Normal(combined_mean, combined_std)
        pref_dist = Normal(pref_mean, pref_std)
        return fused_dist, pref_dist, value

    def get_action_and_value(self, x, action=None):
        fused_dist, pref_dist, value = self._dists(x)
        if action is None:
            action = fused_dist.sample()
        logprob = fused_dist.log_prob(action).sum(1)
        entropy = fused_dist.entropy().sum(1)
        pref_logprob = pref_dist.log_prob(action).sum(1)
        pref_entropy = pref_dist.entropy().sum(1)
        return action, logprob, entropy, value, pref_logprob, pref_entropy


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
                action, logprob, _, value, _, _ = agent.get_action_and_value(next_obs)
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

        # Per-batch advantage normalization for L_pref (paper).
        if args.norm_adv:
            b_advantages_pref = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        else:
            b_advantages_pref = b_advantages

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, pref_logprob, pref_entropy = agent.get_action_and_value(
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

                # PPO clip on the fused (behavioral) distribution.
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Advantage-guided MLE on the preference head.
                mb_adv_pref = b_advantages_pref[mb_inds]
                pref_pg_loss = -(mb_adv_pref * pref_logprob).mean()
                pref_ent_loss = pref_entropy.mean()

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
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + v_loss * args.vf_coef
                    + args.pref_loss_coef * pref_pg_loss
                    - args.pref_ent_coef * pref_ent_loss
                )

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
        writer.add_scalar("losses/pref_pg_loss", pref_pg_loss.item(), global_step)
        writer.add_scalar("losses/pref_entropy", pref_ent_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("policy/main_std_mean", torch.exp(agent.actor_logstd).mean().item(), global_step)
        writer.add_scalar("policy/pref_std_mean", torch.exp(agent.preference_logstd).mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
