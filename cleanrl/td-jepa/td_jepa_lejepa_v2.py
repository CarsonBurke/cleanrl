# TD-JEPA + LeJEPA v2 — online HalfCheetah
#
# v1 (official TD-JEPA, online) failed as a HalfCheetah controller: the actor
# maximized Tφ(sg(φ), â, z)ᵀ z for random skills, and reward entered only as
# z_r = lstsq(ψ(s'), r) on an L2-normalized task encoder. At 120k that residual
# was 123. HalfCheetah reward is x_vel - 0.1||a||²; a unit-sphere ψ cannot span it.
# Mixed-skill collection plus one-env logging produced 1.6k/4.0k lottery spikes.
#
# LeJEPA (Balestriero & LeCun, arXiv:2511.08544, ../lejepa): delete EMA, stop-grad,
# L2-norm, batch ortho. Collapse stop is SIGReg. Prediction target stays attached.
#
# This file:
#   e(s)              unconstrained encoder (no Norm, no EMA)
#   T(e, a) → e'      one-step predictor, BOTH sides live
#   L_jepa = ||T(e(s),a) - e(s')||² + λ SIGReg(e)     layout (1, B, D)
#   φ = [e, a, a⊙a, 1]
#   w_r = ridge(φ → r) every iteration, closed form, no bootstrap
#   Λ     vector GAE of φ  (the only bootstrapped object)
#   V     = w_r · Λ
#   π(e)  one Gaussian policy, PPO on GAE(V), 100% on-policy
#
# Hypothesis: an isotropic Euclidean chart plus a linearly readable reward feature
# makes successor-feature V a real HalfCheetah critic. Kill if EV(w_r·φ, r) stays
# ≪ 0.8 or EV(w_r·Λ, MC) loses to a detached scalar probe on e.
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

from cleanrl.shared.lejepa import SIGReg


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str | None = None
    capture_video: bool = False
    save_model: bool = False

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8_000_000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    jepa_coef: float = 1.0
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    emb_dim: int = 32
    hidden_dim: int = 256
    sigreg_weight: float = 0.09
    sigreg_num_proj: int = 256
    sigreg_knots: int = 17
    sigreg_ref_n: int = 128
    sf_ridge: float = 1e-3
    mc_window: int = 500

    compile: bool = False
    compile_mode: str = "reduce-overhead"

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name):
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
        return env

    return thunk


def bootstrap_observations(next_obs, truncations, infos):
    bootstrap_obs = np.array(next_obs, copy=True)
    truncations = np.asarray(truncations, dtype=bool)
    if not np.any(truncations):
        return bootstrap_obs
    final_observations = infos.get("final_observation")
    final_mask = infos.get("_final_observation")
    if final_observations is None:
        raise RuntimeError("truncated transition missing infos['final_observation']")
    for env_idx in np.flatnonzero(truncations):
        if final_mask is not None and not final_mask[env_idx]:
            raise RuntimeError(f"truncated environment {env_idx} has no final observation")
        final_observation = final_observations[env_idx]
        if final_observation is None:
            raise RuntimeError(f"truncated environment {env_idx} has no final observation")
        bootstrap_obs[env_idx] = final_observation
    return bootstrap_obs


def compute_gae(
    rewards,
    values,
    terminations,
    truncations,
    truncation_bootstrap_values,
    rollout_tail_value,
    gamma,
    gae_lambda,
):
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(rollout_tail_value)
    for t in reversed(range(rewards.shape[0])):
        ordinary_next_value = rollout_tail_value if t == rewards.shape[0] - 1 else values[t + 1]
        next_value = torch.where(truncations[t].bool(), truncation_bootstrap_values[t], ordinary_next_value)
        bootstrap_nonterminal = 1.0 - terminations[t]
        trace_nonterminal = 1.0 - torch.maximum(terminations[t], truncations[t])
        delta = rewards[t] + gamma * bootstrap_nonterminal * next_value - values[t]
        last_advantage = delta + gamma * gae_lambda * trace_nonterminal * last_advantage
        advantages[t] = last_advantage
    return advantages, advantages + values



def successor_features(
    features,
    terminations,
    truncations,
    truncation_bootstrap_features,
    rollout_tail_features,
    gamma,
):
    occupancy = torch.zeros_like(features)
    for t in reversed(range(features.shape[0])):
        if t == features.shape[0] - 1:
            ordinary_next = rollout_tail_features
        else:
            ordinary_next = occupancy[t + 1]
        next_occ = torch.where(
            truncations[t].bool().unsqueeze(-1),
            truncation_bootstrap_features[t],
            ordinary_next,
        )
        occupancy[t] = features[t] + gamma * (1.0 - terminations[t]).unsqueeze(-1) * next_occ
    return occupancy


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def phi_features(emb: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    ones = emb.new_ones(emb.shape[:-1] + (1,))
    return torch.cat([emb, action, action * action, ones], dim=-1)


def solve_reward_probe(phi: torch.Tensor, reward: torch.Tensor, ridge: float) -> torch.Tensor:
    phi64 = phi.double()
    reward64 = reward.double()
    if reward64.ndim == 1:
        reward64 = reward64.unsqueeze(-1)
    gram = phi64.T @ phi64
    rhs = phi64.T @ reward64
    scale = torch.diagonal(gram).mean().clamp_min(1e-12)
    gram = gram + ridge * scale * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram, rhs).squeeze(-1).to(phi.dtype)


def ev_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    var = target.var()
    return float(1.0 - (target - pred).var() / var.clamp_min(1e-12))


def effective_rank(x: torch.Tensor) -> float:
    x = x - x.mean(dim=0)
    s = torch.linalg.svdvals(x)
    p = s / s.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * (p + 1e-12).log()).sum()))


def truncated_mc_returns(rewards: torch.Tensor, terminations: torch.Tensor, gamma: float, horizon: int) -> torch.Tensor:
    t_steps, n_envs = rewards.shape
    returns = torch.zeros_like(rewards)
    for env in range(n_envs):
        for t in range(t_steps):
            acc = rewards.new_zeros(())
            discount = rewards.new_ones(())
            for k in range(horizon):
                idx = t + k
                if idx >= t_steps:
                    break
                acc = acc + discount * rewards[idx, env]
                if terminations[idx, env] > 0:
                    break
                discount = discount * gamma
            returns[t, env] = acc
    return returns


class Agent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, args: Args):
        super().__init__()
        self.emb_dim = args.emb_dim
        self.action_dim = action_dim
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, args.hidden_dim)),
            nn.GELU(),
            layer_init(nn.Linear(args.hidden_dim, args.emb_dim)),
        )
        self.predictor = nn.Sequential(
            layer_init(nn.Linear(args.emb_dim + action_dim, args.hidden_dim)),
            nn.GELU(),
            layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
            nn.GELU(),
            layer_init(nn.Linear(args.hidden_dim, args.emb_dim), std=0.01),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(args.emb_dim, args.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_dim, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.sigreg = SIGReg(knots=args.sigreg_knots, num_proj=args.sigreg_num_proj)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def predict_next(self, emb: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.cat([emb, action], dim=-1))

    def get_action(self, emb: torch.Tensor, action: torch.Tensor | None = None):
        action_mean = self.actor_mean(emb)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        probs = Normal(action_mean, action_logstd.exp())
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)


def jepa_loss(agent: Agent, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, args: Args):
    emb = agent.encode(obs)
    next_emb = agent.encode(next_obs)
    pred = agent.predict_next(emb, action)
    pred_loss = (pred - next_emb).pow(2).mean()
    # SIGReg wants (T, B, D). One-step batch is T=1. Statistic scales with B;
    # pin to ref_n so λ is batch-invariant.
    sig = agent.sigreg(emb.unsqueeze(0))
    sig = sig * (args.sigreg_ref_n / max(emb.shape[0], 1))
    return pred_loss + args.sigreg_weight * sig, pred_loss, sig, emb, next_emb


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
    if device.type != "cuda":
        raise RuntimeError("td_jepa_lejepa_v2 requires CUDA")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    sf_dim = args.emb_dim + 2 * action_dim + 1

    agent = Agent(obs_dim, action_dim, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    if args.compile:
        # Inductor only. Cudagraphs overwrite live JEPA intermediates across minibatches.
        update_jepa = torch.compile(jepa_loss, mode="default")
    else:
        update_jepa = jepa_loss

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    terminations_buffer = torch.zeros((args.num_steps, args.num_envs), device=device)
    truncations_buffer = torch.zeros((args.num_steps, args.num_envs), device=device)
    next_obs_buffer = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    truncation_bootstrap_values = torch.zeros((args.num_steps, args.num_envs), device=device)
    truncation_bootstrap_phi = torch.zeros((args.num_steps, args.num_envs, sf_dim), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    w_r = torch.zeros(sf_dim, device=device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            with torch.no_grad():
                emb = agent.encode(next_obs)
                action, logprob, _ = agent.get_action(emb)
                phi = phi_features(emb, action)
                values[step] = phi @ w_r
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device).view(-1)
            terminations_buffer[step] = torch.as_tensor(terminations, dtype=torch.float32, device=device)
            truncations_buffer[step] = torch.as_tensor(truncations, dtype=torch.float32, device=device)
            real_next = bootstrap_observations(next_obs_np, truncations, infos)
            next_obs_buffer[step] = torch.as_tensor(real_next, dtype=torch.float32, device=device)
            if np.any(truncations):
                final_obs = torch.as_tensor(real_next, dtype=torch.float32, device=device)
                with torch.no_grad():
                    final_emb = agent.encode(final_obs)
                    # No action at the timeout state; use a zero-action feature for the
                    # remaining discounted occupancy of the constant / embedding block.
                    zero_action = torch.zeros((args.num_envs, action_dim), device=device)
                    final_phi = phi_features(final_emb, zero_action)
                    final_values = final_phi @ w_r
                mask = torch.as_tensor(truncations, dtype=torch.bool, device=device)
                truncation_bootstrap_values[step] = torch.where(mask, final_values, torch.zeros_like(final_values))
                truncation_bootstrap_phi[step] = torch.where(mask.unsqueeze(-1), final_phi, torch.zeros_like(final_phi))
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        break

        with torch.no_grad():
            flat_obs = obs.reshape(-1, obs_dim)
            flat_act = actions.reshape(-1, action_dim)
            emb_buf = agent.encode(flat_obs).reshape(args.num_steps, args.num_envs, args.emb_dim)
            phi = phi_features(emb_buf, actions)
            tail_emb = agent.encode(next_obs)
            tail_action, _, _ = agent.get_action(tail_emb)
            tail_phi = phi_features(tail_emb, tail_action)
            sf = successor_features(
                phi,
                terminations_buffer,
                truncations_buffer,
                truncation_bootstrap_phi,
                tail_phi,
                args.gamma,
            )
            w_r = solve_reward_probe(phi.reshape(-1, sf_dim), rewards.reshape(-1), args.sf_ridge)
            values = (sf * w_r).sum(-1)
            tail_value = (tail_phi * w_r).sum(-1)
            truncation_bootstrap_values = (truncation_bootstrap_phi * w_r).sum(-1)
            advantages, returns = compute_gae(
                rewards,
                values,
                terminations_buffer,
                truncations_buffer,
                truncation_bootstrap_values,
                tail_value,
                args.gamma,
                args.gae_lambda,
            )
            mc = truncated_mc_returns(rewards, terminations_buffer, args.gamma, args.mc_window)
            reward_hat = phi.reshape(-1, sf_dim) @ w_r
            ev_reward = ev_score(reward_hat, rewards.reshape(-1))
            ev_sf = ev_score(values.reshape(-1), mc.reshape(-1))
            ev_e = ev_score(
                emb_buf.reshape(-1, args.emb_dim)
                @ solve_reward_probe(emb_buf.reshape(-1, args.emb_dim), mc.reshape(-1), args.sf_ridge),
                mc.reshape(-1),
            )
            emb_rank = effective_rank(emb_buf.reshape(-1, args.emb_dim))

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_next_obs = next_obs_buffer.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        old_approx_kl = torch.zeros((), device=device)
        approx_kl = torch.zeros((), device=device)
        last_metrics = {
            "pred_loss": torch.zeros((), device=device),
            "sigreg": torch.zeros((), device=device),
            "jepa": torch.zeros((), device=device),
            "pg": torch.zeros((), device=device),
            "entropy": torch.zeros((), device=device),
        }
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_emb = agent.encode(b_obs[mb_inds])
                _, newlogprob, entropy = agent.get_action(mb_emb.detach(), b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                jepa_total, pred_l, sig_l, _, _ = update_jepa(
                    agent, b_obs[mb_inds], b_actions[mb_inds], b_next_obs[mb_inds], args
                )
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.jepa_coef * jepa_total

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                last_metrics = {
                    "pred_loss": pred_l.detach(),
                    "sigreg": sig_l.detach(),
                    "jepa": jepa_total.detach(),
                    "pg": pg_loss.detach(),
                    "entropy": entropy_loss.detach(),
                }

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/policy_loss", last_metrics["pg"].item(), global_step)
        writer.add_scalar("losses/entropy", last_metrics["entropy"].item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/jepa", last_metrics["jepa"].item(), global_step)
        writer.add_scalar("losses/pred_loss", last_metrics["pred_loss"].item(), global_step)
        writer.add_scalar("ssl/sigreg", last_metrics["sigreg"].item(), global_step)
        writer.add_scalar("ssl/emb_effective_rank", emb_rank, global_step)
        writer.add_scalar("gate/ev_reward_probe", ev_reward, global_step)
        writer.add_scalar("gate/ev_sf_vs_mc", ev_sf, global_step)
        writer.add_scalar("gate/ev_e_vs_mc", ev_e, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
