# TD-JEPA + LeJEPA v4 — projector-split chart, pathwise SF actor
#
# v2/v3 learning was bad for a representation reason, not just the actor.
# v2 @100k: EV(w_r·φ, r) = 0.38, EV(e, MC) = -0.19, return ≈ -330.
# v3 @48k:  EV(w_r·φ, r) = 0.39. Same unreadability. SIGReg was on the same e
# that w_r and F_e read. Isotropic N(0,I) strips the scale x_vel lives on
# (embopt FAMILY.md: ~25 EV points lost). a⊙a can only explain ctrl cost.
#
# v4: SIGReg on a projector p(e), never on e.
#   L_jepa = ||T(e(s),a) - e(s')||² + λ SIGReg(p(e))     both sides attached
#   φ = [e, a, a⊙a, 1]
#   w_r = ridge(φ → r)
#   F_e(e,a) → sg(e') + γ F_e(sg(e'), π(sg(e')))
#   Λ̂ = [F_e(e, â), â, â⊙â, 1]
#   L_π = - w_r · Λ̂
#
# Encoder+predictor+projector: JEPA only. Actor does not step F_e.
# Raw obs stored; current running stats at every use.
#
# Kill if EV(w_r·φ, r) stays ≪ 0.8 after 200k.
import math
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.lejepa import SIGReg
from cleanrl_utils.buffers import ReplayBuffer


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
    num_envs: int = 16
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 5_000
    gamma: float = 0.99
    tau: float = 0.005
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_frequency: int = 2
    learning_rate: float = 3e-4

    emb_dim: int = 32
    proj_dim: int = 16
    hidden_dim: int = 256
    sigreg_weight: float = 0.09
    sigreg_num_proj: int = 256
    sigreg_knots: int = 17
    sigreg_ref_n: int = 128
    sf_ridge: float = 1e-3
    updates_per_env_step: int = 1
    log_interval: int = 1_000

    compile: bool = False
    compile_mode: str = "reduce-overhead"


def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env.action_space.seed(seed)
        return env

    return thunk


def layer_init(layer: nn.Linear, std: float = math.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(in_dim: int, hidden: int, out_dim: int, out_std: float = math.sqrt(2.0)) -> nn.Sequential:
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, hidden)),
        nn.GELU(),
        layer_init(nn.Linear(hidden, hidden)),
        nn.GELU(),
        layer_init(nn.Linear(hidden, out_dim), std=out_std),
    )


class RunningMeanStd(nn.Module):
    def __init__(self, shape: int, epsilon: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(epsilon))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(x.shape[0]), device=x.device, dtype=self.count.dtype)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


def phi_immediate(action: torch.Tensor) -> torch.Tensor:
    ones = action.new_ones(action.shape[:-1] + (1,))
    return torch.cat([action, action * action, ones], dim=-1)


def compose_successor(emb_occ: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return torch.cat([emb_occ, phi_immediate(action)], dim=-1)


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


def soft_update_module(src: nn.Module, tgt: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


class Agent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, args: Args):
        super().__init__()
        self.action_dim = action_dim
        self.emb_dim = args.emb_dim
        self.encoder = mlp(obs_dim, args.hidden_dim, args.emb_dim, out_std=1.0)
        self.predictor = mlp(args.emb_dim + action_dim, args.hidden_dim, args.emb_dim, out_std=0.01)
        self.projector = mlp(args.emb_dim, args.hidden_dim, args.proj_dim, out_std=1.0)
        self.sf_head = mlp(args.emb_dim + action_dim, args.hidden_dim, args.emb_dim, out_std=0.01)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(args.emb_dim, args.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_dim, action_dim), std=0.01),
        )
        self.sigreg = SIGReg(knots=args.sigreg_knots, num_proj=args.sigreg_num_proj)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def project(self, emb: torch.Tensor) -> torch.Tensor:
        return self.projector(emb)

    def predict_next(self, emb: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.cat([emb, action], dim=-1))

    def embedding_occupancy(self, emb: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.sf_head(torch.cat([emb, action], dim=-1))

    def act(self, emb: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(emb))


def jepa_losses(agent: Agent, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, args: Args):
    emb = agent.encode(obs)
    next_emb = agent.encode(next_obs)
    pred = agent.predict_next(emb, action)
    pred_loss = (pred - next_emb).pow(2).mean()
    projected = agent.project(emb)
    sig = agent.sigreg(projected.unsqueeze(0)) * (args.sigreg_ref_n / max(projected.shape[0], 1))
    return pred_loss + args.sigreg_weight * sig, pred_loss, sig, emb, next_emb, projected


def sf_td_loss(
    agent: Agent,
    target_agent: Agent,
    emb: torch.Tensor,
    action: torch.Tensor,
    next_emb: torch.Tensor,
    dones: torch.Tensor,
    args: Args,
):
    pred = agent.embedding_occupancy(emb, action)
    with torch.no_grad():
        next_action = target_agent.act(next_emb)
        noise = (torch.randn_like(next_action) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
        next_action = (next_action + noise).clamp(-1.0, 1.0)
        target = next_emb + args.gamma * (1.0 - dones) * target_agent.embedding_occupancy(next_emb, next_action)
    return (pred - target).pow(2).mean(), pred, target


def actor_objective(agent: Agent, emb: torch.Tensor, w_r: torch.Tensor):
    action = agent.act(emb)
    occupancy = agent.embedding_occupancy(emb, action)
    q = (compose_successor(occupancy, action) * w_r).sum(-1)
    return -q.mean(), q, action


if __name__ == "__main__":
    args = tyro.cli(Args)
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
        raise RuntimeError("td_jepa_lejepa_v4 requires CUDA")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    sf_dim = args.emb_dim + 2 * action_dim + 1

    agent = Agent(obs_dim, action_dim, args).to(device)
    target_agent = Agent(obs_dim, action_dim, args).to(device)
    target_agent.load_state_dict(agent.state_dict())
    target_agent.requires_grad_(False)
    obs_rms = RunningMeanStd(obs_dim).to(device)

    jepa_opt = torch.optim.Adam(
        list(agent.encoder.parameters())
        + list(agent.predictor.parameters())
        + list(agent.projector.parameters()),
        lr=args.learning_rate,
    )
    sf_opt = torch.optim.Adam(agent.sf_head.parameters(), lr=args.learning_rate)
    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=args.learning_rate)

    jepa_fn = jepa_losses
    sf_fn = sf_td_loss
    actor_fn = actor_objective
    if args.compile:
        jepa_fn = torch.compile(jepa_losses, mode="default")
        sf_fn = torch.compile(sf_td_loss, mode="default")
        actor_fn = torch.compile(actor_objective, mode="default")

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    w_r = torch.zeros(sf_dim, device=device)
    global_step = 0
    update_step = 0
    metrics: dict[str, float] = {}

    while global_step < args.total_timesteps:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        obs_rms.update(obs_t)
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions_t = agent.act(agent.encode(obs_rms.normalize(obs_t)))
                actions_t = (actions_t + args.exploration_noise * torch.randn_like(actions_t)).clamp(-1.0, 1.0)
                actions = actions_t.cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs
        global_step += args.num_envs

        if global_step < args.learning_starts:
            continue

        for _ in range(args.num_envs * args.updates_per_env_step):
            update_step += 1
            batch = rb.sample(args.batch_size)
            obs_n = obs_rms.normalize(batch.observations)
            next_obs_n = obs_rms.normalize(batch.next_observations)
            jepa_total, pred_l, sig_l, emb, next_emb, projected = jepa_fn(
                agent, obs_n, batch.actions, next_obs_n, args
            )
            jepa_opt.zero_grad(set_to_none=True)
            jepa_total.backward()
            jepa_opt.step()

            with torch.no_grad():
                emb_sg = agent.encode(obs_n)
                next_emb_sg = agent.encode(next_obs_n)
                phi = compose_successor(emb_sg, batch.actions)
                w_r = solve_reward_probe(phi, batch.rewards.flatten(), args.sf_ridge)
                ev_reward = ev_score(phi @ w_r, batch.rewards.flatten())
                emb_rank = effective_rank(emb_sg)
                proj_rank = effective_rank(agent.project(emb_sg))

            sf_loss, _, _ = sf_fn(
                agent, target_agent, emb_sg, batch.actions, next_emb_sg, batch.dones, args
            )
            sf_opt.zero_grad(set_to_none=True)
            sf_loss.backward()
            sf_opt.step()

            actor_q = torch.zeros((), device=device)
            if update_step % args.policy_frequency == 0:
                actor_loss, actor_q, _ = actor_fn(agent, emb_sg, w_r.detach())
                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_opt.step()
                soft_update_module(agent.sf_head, target_agent.sf_head, args.tau)
                soft_update_module(agent.actor, target_agent.actor, args.tau)

            metrics = {
                "jepa": float(jepa_total.detach()),
                "pred_loss": float(pred_l.detach()),
                "sigreg": float(sig_l.detach()),
                "sf_td": float(sf_loss.detach()),
                "q": float(actor_q.detach().mean() if actor_q.ndim else actor_q.detach()),
                "ev_reward": ev_reward,
                "emb_rank": emb_rank,
                "proj_rank": proj_rank,
                "w_r_norm": float(w_r.norm()),
            }

        if global_step % args.log_interval < args.num_envs and metrics:
            sps = int(global_step / (time.time() - start_time))
            print("SPS:", sps)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("losses/jepa", metrics["jepa"], global_step)
            writer.add_scalar("losses/pred_loss", metrics["pred_loss"], global_step)
            writer.add_scalar("ssl/sigreg", metrics["sigreg"], global_step)
            writer.add_scalar("ssl/emb_effective_rank", metrics["emb_rank"], global_step)
            writer.add_scalar("ssl/proj_effective_rank", metrics["proj_rank"], global_step)
            writer.add_scalar("losses/sf_td", metrics["sf_td"], global_step)
            writer.add_scalar("losses/actor_q", metrics["q"], global_step)
            writer.add_scalar("gate/ev_reward_probe", metrics["ev_reward"], global_step)
            writer.add_scalar("charts/w_r_norm", metrics["w_r_norm"], global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
