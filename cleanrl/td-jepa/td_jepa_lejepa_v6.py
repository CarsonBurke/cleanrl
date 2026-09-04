# TD-JEPA + LeJEPA v6 — measured occupancy, no F_e head
#
# v5: SIGReg-on-e made EV(w_r·φ, r) = 0.87 and rank 31. Learned F_e TD
# diverged (0.77 → 2.4). Pathwise then only trusted a⊙a → stand-still −180.
#
# v6 deletes F_e. Occupancy of e is measured on the rollout:
#   Λ_e,t = e_t + γ Λ_e,t+1
#   W_map = ridge(e → Λ_e)                 closed form, d×d
#   w_r   = ridge(std([e, a, a⊙a, 1]) → r)
#   â = π(sg(e))
#   Λ̂_e(â) = e + γ T(sg(e), â) W_map      T is JEPA dynamics, not stepped
#   Λ̂ = std([(1-γ) Λ̂_e, â, â⊙â, 1])
#   L_π = - w_r · Λ̂
#
# Chart unchanged from v5: attached T + SIGReg(e). No projector, no F head,
# no DDPG target on occupancy. Encoder and T get JEPA only.
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
    num_minibatches: int = 32
    update_epochs: int = 8
    max_grad_norm: float = 0.5
    exploration_noise: float = 0.1

    emb_dim: int = 32
    hidden_dim: int = 256
    sigreg_weight: float = 0.09
    sigreg_num_proj: int = 256
    sigreg_knots: int = 17
    sigreg_ref_n: int = 128
    sf_ridge: float = 1e-3
    jepa_coef: float = 1.0

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


def successor_features(features, terminations, truncations, truncation_bootstrap, tail, gamma):
    occupancy = torch.zeros_like(features)
    for t in reversed(range(features.shape[0])):
        ordinary_next = tail if t == features.shape[0] - 1 else occupancy[t + 1]
        next_occ = torch.where(truncations[t].bool().unsqueeze(-1), truncation_bootstrap[t], ordinary_next)
        occupancy[t] = features[t] + gamma * (1.0 - terminations[t]).unsqueeze(-1) * next_occ
    return occupancy


def layer_init(layer, std=math.sqrt(2.0), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(in_dim, hidden, out_dim, out_std=math.sqrt(2.0)):
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, hidden)),
        nn.GELU(),
        layer_init(nn.Linear(hidden, hidden)),
        nn.GELU(),
        layer_init(nn.Linear(hidden, out_dim), std=out_std),
    )


def phi_immediate(emb, action):
    ones = action.new_ones(action.shape[:-1] + (1,))
    return torch.cat([emb, action, action * action, ones], dim=-1)


def compose_occupancy(emb_occ, action, gamma):
    return phi_immediate((1.0 - gamma) * emb_occ, action)


def column_standardize(x, mean=None, std=None):
    if mean is None:
        mean = x.mean(dim=0).clone()
        mean[-1] = 0.0
    if std is None:
        std = x.std(dim=0, unbiased=False).clamp_min(1e-6).clone()
        std[-1] = 1.0
    return (x - mean) / std, mean, std


def solve_ridge(inputs, targets, ridge):
    x = inputs.double()
    y = targets.double()
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    gram = x.T @ x
    rhs = x.T @ y
    scale = torch.diagonal(gram).mean().clamp_min(1e-12)
    gram = gram + ridge * scale * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram, rhs).to(inputs.dtype)


def ev_score(pred, target):
    var = target.var()
    return float(1.0 - (target - pred).var() / var.clamp_min(1e-12))


def effective_rank(x):
    x = x - x.mean(dim=0)
    s = torch.linalg.svdvals(x)
    p = s / s.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * (p + 1e-12).log()).sum()))


class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim, args):
        super().__init__()
        self.encoder = mlp(obs_dim, args.hidden_dim, args.emb_dim, out_std=1.0)
        self.predictor = mlp(args.emb_dim + action_dim, args.hidden_dim, args.emb_dim, out_std=0.01)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(args.emb_dim, args.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_dim, action_dim), std=0.01),
        )
        self.sigreg = SIGReg(knots=args.sigreg_knots, num_proj=args.sigreg_num_proj)

    def encode(self, obs):
        return self.encoder(obs)

    def predict_next(self, emb, action):
        return self.predictor(torch.cat([emb, action], dim=-1))

    def act(self, emb):
        return torch.tanh(self.actor(emb))


def jepa_loss(agent, obs, action, next_obs, args):
    emb = agent.encode(obs)
    next_emb = agent.encode(next_obs)
    pred = agent.predict_next(emb, action)
    pred_loss = (pred - next_emb).pow(2).mean()
    sig = agent.sigreg(emb.unsqueeze(0)) * (args.sigreg_ref_n / max(emb.shape[0], 1))
    return pred_loss + args.sigreg_weight * sig, pred_loss, sig


def predicted_occupancy(agent, emb, action, occupancy_map, gamma):
    next_emb = agent.predict_next(emb, action)
    return emb + gamma * next_emb @ occupancy_map


def actor_objective(agent, emb, occupancy_map, w_r, feat_mean, feat_std, gamma):
    action = agent.act(emb)
    occ = predicted_occupancy(agent, emb, action, occupancy_map, gamma)
    phi, _, _ = column_standardize(compose_occupancy(occ, action, gamma), feat_mean, feat_std)
    q = (phi * w_r).sum(-1)
    return -q.mean(), q, action


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
        raise RuntimeError("td_jepa_lejepa_v6 requires CUDA")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))

    agent = Agent(obs_dim, action_dim, args).to(device)
    jepa_opt = optim.Adam(
        list(agent.encoder.parameters()) + list(agent.predictor.parameters()),
        lr=args.learning_rate,
    )
    actor_opt = optim.Adam(agent.actor.parameters(), lr=args.learning_rate)
    jepa_fn = torch.compile(jepa_loss, mode="default") if args.compile else jepa_loss
    actor_fn = torch.compile(actor_objective, mode="default") if args.compile else actor_objective

    obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    next_obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    actions_buf = torch.zeros((args.num_steps, args.num_envs, action_dim), device=device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    terminations_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    truncations_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    trunc_boot_e = torch.zeros((args.num_steps, args.num_envs, args.emb_dim), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            jepa_opt.param_groups[0]["lr"] = frac * args.learning_rate
            actor_opt.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            with torch.no_grad():
                action = agent.act(agent.encode(next_obs))
                action = (action + args.exploration_noise * torch.randn_like(action)).clamp(-1.0, 1.0)
            actions_buf[step] = action
            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            rewards_buf[step] = torch.as_tensor(reward, dtype=torch.float32, device=device).view(-1)
            terminations_buf[step] = torch.as_tensor(terminations, dtype=torch.float32, device=device)
            truncations_buf[step] = torch.as_tensor(truncations, dtype=torch.float32, device=device)
            real_next = bootstrap_observations(next_obs_np, truncations, infos)
            next_obs_buf[step] = torch.as_tensor(real_next, dtype=torch.float32, device=device)
            if np.any(truncations):
                with torch.no_grad():
                    final_e = agent.encode(next_obs_buf[step])
                    boot = final_e / (1.0 - args.gamma)
                mask = torch.as_tensor(truncations, dtype=torch.bool, device=device)
                trunc_boot_e[step] = torch.where(mask.unsqueeze(-1), boot, torch.zeros_like(boot))
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        break

        with torch.no_grad():
            emb = agent.encode(obs_buf.reshape(-1, obs_dim)).reshape(args.num_steps, args.num_envs, args.emb_dim)
            tail_e = agent.encode(next_obs)
            tail_occ = tail_e / (1.0 - args.gamma)
            lam_e = successor_features(
                emb, terminations_buf, truncations_buf, trunc_boot_e, tail_occ, args.gamma
            )
            flat_e = emb.reshape(-1, args.emb_dim)
            flat_lam = lam_e.reshape(-1, args.emb_dim)
            occupancy_map = solve_ridge(flat_e, flat_lam, args.sf_ridge)
            phi_raw = phi_immediate(emb, actions_buf).reshape(-1, args.emb_dim + 2 * action_dim + 1)
            phi_std, feat_mean, feat_std = column_standardize(phi_raw)
            w_r = solve_ridge(phi_std, rewards_buf.reshape(-1), args.sf_ridge).squeeze(-1)
            ev_reward = ev_score(phi_std @ w_r, rewards_buf.reshape(-1))
            ev_map = ev_score(flat_e @ occupancy_map, flat_lam)
            emb_rank = effective_rank(flat_e)
            measured_q = (
                column_standardize(compose_occupancy(lam_e.reshape(-1, args.emb_dim), actions_buf.reshape(-1, action_dim), args.gamma), feat_mean, feat_std)[0]
                * w_r
            ).sum(-1)

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_next = next_obs_buf.reshape(-1, obs_dim)
        b_act = actions_buf.reshape(-1, action_dim)
        b_inds = np.arange(args.batch_size)
        last = {"pred": 0.0, "sig": 0.0, "jepa": 0.0, "q": 0.0}
        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start : start + args.minibatch_size]
                jepa_total, pred_l, sig_l = jepa_fn(agent, b_obs[mb], b_act[mb], b_next[mb], args)
                jepa_opt.zero_grad(set_to_none=True)
                (args.jepa_coef * jepa_total).backward()
                nn.utils.clip_grad_norm_(
                    list(agent.encoder.parameters()) + list(agent.predictor.parameters()),
                    args.max_grad_norm,
                )
                jepa_opt.step()

                with torch.no_grad():
                    mb_emb = agent.encode(b_obs[mb])
                actor_loss, q, _ = actor_fn(
                    agent,
                    mb_emb,
                    occupancy_map.detach(),
                    w_r.detach(),
                    feat_mean.detach(),
                    feat_std.detach(),
                    args.gamma,
                )
                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
                actor_opt.step()
                last = {
                    "pred": float(pred_l.detach()),
                    "sig": float(sig_l.detach()),
                    "jepa": float(jepa_total.detach()),
                    "q": float(q.detach().mean()),
                }

        writer.add_scalar("charts/learning_rate", actor_opt.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/jepa", last["jepa"], global_step)
        writer.add_scalar("losses/pred_loss", last["pred"], global_step)
        writer.add_scalar("ssl/sigreg", last["sig"], global_step)
        writer.add_scalar("ssl/emb_effective_rank", emb_rank, global_step)
        writer.add_scalar("gate/ev_reward_probe", ev_reward, global_step)
        writer.add_scalar("gate/ev_occupancy_map", ev_map, global_step)
        writer.add_scalar("losses/actor_q", last["q"], global_step)
        writer.add_scalar("losses/measured_q", float(measured_q.mean()), global_step)
        writer.add_scalar("charts/w_r_norm", float(w_r.norm()), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
