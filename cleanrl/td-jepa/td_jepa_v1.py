# TD-JEPA v1 — online HalfCheetah port of Bagatella et al. 2025
# "TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning"
# https://arxiv.org/abs/2510.00739  (PDF: td_jepa_2510.00739v1.pdf)
#
# Official algorithm (facebookresearch/td_jepa, DMC proprio / cheetah):
#   φ(s)  state encoder, L2-normalized, dim 256, linear
#   ψ(s)  task encoder, L2-normalized, dim 50, 2-layer MLP
#   Tφ(φ, a, z)  twin predictor ≈ successor features of ψ under π_z
#   Tψ(ψ, a, z)  twin predictor, roles inverted
#   π(φ, z)      tanh-Gaussian, fixed std 0.2
#   L_TD = ||Tφ(φ(s),a,z) - ψ̄(s') - γ Tφ̄(φ̄(s'),a',z)||^2
#        + ||Tψ(ψ(s),a,z) - φ̄(s') - γ Tψ̄(ψ̄(s'),a',z)||^2
#   L_orth = -mean diag(ΦΦᵀ) + 0.5 mean_{i≠j} (ΦΦᵀ)²_{ij}   (same for Ψ)
#   L_π = -Tφ(sg(φ), â, z)ᵀ z
#   z ~ mix(hypersphere, ψ(s')) with p_goal=0.5
#
# Paper is offline / reward-free. This file keeps those losses untouched and
# only changes the data source: online HalfCheetah replay. Task vector z_r is
# inferred by least squares of r ≈ ψ(s')ᵀ z and used as the behavior skill.
# Hypothesis: multi-policy latent TD prediction yields a successor-feature
# family whose inferred skill is a strong HalfCheetah run policy.
import copy
import math
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.distributions.utils import _standard_normal
from torch.utils.tensorboard import SummaryWriter

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
    batch_size: int = 1024
    learning_starts: int = 5_000
    gamma: float = 0.98
    lr_phi: float = 1e-4
    lr_psi: float = 1e-4
    lr_predictor: float = 1e-4
    lr_actor: float = 1e-4
    weight_decay: float = 0.0
    encoder_target_tau: float = 0.001
    predictor_target_tau: float = 0.001
    phi_ortho_coef: float = 0.1
    psi_ortho_coef: float = 0.1
    train_goal_ratio: float = 0.5
    actor_std: float = 0.2
    stddev_clip: float = 0.3
    predictor_pessimism_penalty: float = 0.0
    actor_pessimism_penalty: float = 0.0

    phi_dim: int = 256
    psi_dim: int = 50
    encoder_hidden_dim: int = 256
    phi_hidden_layers: int = 0
    psi_hidden_layers: int = 2
    predictor_hidden_dim: int = 1024
    predictor_hidden_layers: int = 1
    predictor_embedding_layers: int = 2
    num_parallel: int = 2
    actor_hidden_dim: int = 1024
    actor_hidden_layers: int = 1
    actor_embedding_layers: int = 2
    norm_z: bool = True
    normalize_obs: bool = True

    z_infer_interval: int = 1_000
    z_infer_samples: int = 10_000
    collect_task_ratio: float = 0.5
    """fraction of parallel envs that follow the inferred reward skill z_r"""
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


def weight_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, DenseParallel):
        parallel_orthogonal_(module.weight.data, nn.init.calculate_gain("relu"))
        if module.bias is not None:
            module.bias.data.zero_()


def soft_update(src: nn.Module, tgt: nn.Module, tau: float) -> None:
    src_params = tuple(src.parameters())
    tgt_params = tuple(tgt.parameters())
    torch._foreach_mul_(tgt_params, 1.0 - tau)
    torch._foreach_add_(tgt_params, src_params, alpha=tau)


def parallel_orthogonal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    if tensor.ndimension() == 2:
        return nn.init.orthogonal_(tensor, gain=gain)
    if tensor.ndimension() < 3:
        raise ValueError("Only tensors with 2+ dimensions are supported")
    n_parallel = tensor.size(0)
    rows = tensor.size(1)
    cols = tensor.numel() // n_parallel // rows
    flattened = tensor.new(n_parallel, rows, cols).normal_(0, 1)
    qs = []
    for flat in torch.unbind(flattened, dim=0):
        if rows < cols:
            flat = flat.t()
        q, r = torch.linalg.qr(flat)
        q = q * r.diag().sign()
        if rows < cols:
            q = q.t()
        qs.append(q)
    qs = torch.stack(qs, dim=0)
    with torch.no_grad():
        tensor.view_as(qs).copy_(qs)
        tensor.mul_(gain)
    return tensor


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
        batch_count = torch.tensor(x.shape[0], device=x.device, dtype=self.count.dtype)
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


class Norm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)


class TruncatedNormal(torch.distributions.Normal):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, low: float = -1.0, high: float = 1.0, eps: float = 1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x: torch.Tensor) -> torch.Tensor:
        clamped = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        return x - x.detach() + clamped.detach()

    def sample(self, clip: float | None = None, sample_shape=torch.Size()) -> torch.Tensor:  # type: ignore[override]
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device) * self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        return self._clamp(self.loc + eps)


class DenseParallel(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_parallel: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_parallel = n_parallel
        self.weight = nn.Parameter(torch.empty(n_parallel, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(n_parallel, 1, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.baddbmm(self.bias, x, self.weight)


class ParallelLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, n_parallel: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_parallel, 1, normalized_shape))
        self.bias = nn.Parameter(torch.zeros(n_parallel, 1, normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, None, None, self.eps) * self.weight + self.bias


def linear(in_dim: int, out_dim: int, n_parallel: int = 1) -> nn.Module:
    if n_parallel > 1:
        return DenseParallel(in_dim, out_dim, n_parallel)
    return nn.Linear(in_dim, out_dim)


def layernorm(dim: int, n_parallel: int = 1) -> nn.Module:
    if n_parallel > 1:
        return ParallelLayerNorm(dim, n_parallel)
    return nn.LayerNorm(dim)


def simple_embedding(in_dim: int, hidden_dim: int, hidden_layers: int, n_parallel: int = 1) -> nn.Sequential:
    if hidden_layers < 2:
        raise ValueError("simple_embedding requires at least 2 layers")
    seq: list[nn.Module] = [linear(in_dim, hidden_dim, n_parallel), layernorm(hidden_dim, n_parallel), nn.Tanh()]
    for _ in range(hidden_layers - 2):
        seq += [linear(hidden_dim, hidden_dim, n_parallel), nn.ReLU()]
    seq += [linear(hidden_dim, hidden_dim // 2, n_parallel), nn.ReLU()]
    return nn.Sequential(*seq)


class BackwardMap(nn.Module):
    def __init__(self, obs_dim: int, out_dim: int, hidden_dim: int, hidden_layers: int):
        super().__init__()
        if hidden_layers == 0:
            layers: list[nn.Module] = [nn.Linear(obs_dim, out_dim)]
        else:
            layers = [nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
            for _ in range(hidden_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            layers += [nn.Linear(hidden_dim, out_dim)]
        layers += [Norm()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ForwardMap(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        z_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        embedding_layers: int,
        output_dim: int,
        num_parallel: int,
    ):
        super().__init__()
        self.num_parallel = num_parallel
        self.embed_z = simple_embedding(obs_dim + z_dim, hidden_dim, embedding_layers, num_parallel)
        self.embed_sa = simple_embedding(obs_dim + action_dim, hidden_dim, embedding_layers, num_parallel)
        seq: list[nn.Module] = []
        for _ in range(hidden_layers):
            seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
        seq += [linear(hidden_dim, output_dim, num_parallel)]
        self.Fs = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            action = action.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        z_dim: int,
        action_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        embedding_layers: int,
    ):
        super().__init__()
        self.embed_z = simple_embedding(obs_dim + z_dim, hidden_dim, embedding_layers)
        self.embed_s = simple_embedding(obs_dim, hidden_dim, embedding_layers)
        seq: list[nn.Module] = []
        for _ in range(hidden_layers):
            seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        seq += [nn.Linear(hidden_dim, action_dim)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, std: float) -> TruncatedNormal:
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))
        s_embedding = self.embed_s(obs)
        mu = torch.tanh(self.policy(torch.cat([s_embedding, z_embedding], dim=-1)))
        return TruncatedNormal(mu, torch.ones_like(mu) * std)


def project_z(z: torch.Tensor, norm_z: bool) -> torch.Tensor:
    if norm_z:
        return math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
    return z


def sample_z(size: int, z_dim: int, device: torch.device, norm_z: bool) -> torch.Tensor:
    return project_z(torch.randn(size, z_dim, device=device), norm_z)


def orth_loss(enc: torch.Tensor, off_diag: torch.Tensor, off_diag_sum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cov = enc @ enc.T
    diag = -cov.diag().mean()
    offdiag = 0.5 * (cov * off_diag).pow(2).sum() / off_diag_sum
    return offdiag + diag, diag, offdiag


def ensemble_stats(preds: torch.Tensor, pessimism_penalty: float) -> torch.Tensor:
    preds_mean = preds.mean(dim=0)
    if pessimism_penalty == 0.0:
        return preds_mean
    diffs = (preds.unsqueeze(0) - preds.unsqueeze(1)).abs()
    scale = preds.shape[0] ** 2 - preds.shape[0]
    unc = diffs.sum(dim=(0, 1)) / scale
    return preds_mean - pessimism_penalty * unc


def reward_inference(psi: torch.Tensor, reward: torch.Tensor, norm_z: bool) -> torch.Tensor:
    z = torch.linalg.lstsq(psi, reward).solution.T
    return project_z(z, norm_z)


class TDJEPA(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, args: Args):
        super().__init__()
        self.args = args
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.z_dim = args.psi_dim
        self.phi = BackwardMap(obs_dim, args.phi_dim, args.encoder_hidden_dim, args.phi_hidden_layers)
        self.psi = BackwardMap(obs_dim, args.psi_dim, args.encoder_hidden_dim, args.psi_hidden_layers)
        self.phi_predictor = ForwardMap(
            args.phi_dim,
            self.z_dim,
            action_dim,
            args.predictor_hidden_dim,
            args.predictor_hidden_layers,
            args.predictor_embedding_layers,
            args.psi_dim,
            args.num_parallel,
        )
        self.psi_predictor = ForwardMap(
            args.psi_dim,
            self.z_dim,
            action_dim,
            args.predictor_hidden_dim,
            args.predictor_hidden_layers,
            args.predictor_embedding_layers,
            args.phi_dim,
            args.num_parallel,
        )
        self.actor = Actor(
            args.phi_dim,
            self.z_dim,
            action_dim,
            args.actor_hidden_dim,
            args.actor_hidden_layers,
            args.actor_embedding_layers,
        )
        self.obs_rms = RunningMeanStd(obs_dim)
        self.apply(weight_init)
        self.target_phi = copy.deepcopy(self.phi)
        self.target_psi = copy.deepcopy(self.psi)
        self.target_phi_predictor = copy.deepcopy(self.phi_predictor)
        self.target_psi_predictor = copy.deepcopy(self.psi_predictor)
        for module in (self.target_phi, self.target_psi, self.target_phi_predictor, self.target_psi_predictor):
            module.requires_grad_(False)
        self.register_buffer("z_task", sample_z(1, self.z_dim, torch.device("cpu"), args.norm_z).squeeze(0))

    def maybe_normalize(self, obs: torch.Tensor, update: bool = False) -> torch.Tensor:
        if not self.args.normalize_obs:
            return obs
        if update:
            self.obs_rms.update(obs.detach())
        return self.obs_rms.normalize(obs)

    def encode_phi(self, obs: torch.Tensor) -> torch.Tensor:
        return self.phi(obs)

    def encode_psi(self, obs: torch.Tensor) -> torch.Tensor:
        return self.psi(obs)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = False) -> torch.Tensor:
        obs_n = self.maybe_normalize(obs, update=False)
        phi = self.encode_phi(obs_n)
        if z.dim() == 1:
            z = z.expand(obs.shape[0], -1)
        dist = self.actor(phi, z, self.args.actor_std)
        return dist.mean if mean else dist.sample()

    def sample_mixed_z(self, train_goal: torch.Tensor) -> torch.Tensor:
        batch = train_goal.shape[0]
        z = sample_z(batch, self.z_dim, train_goal.device, self.args.norm_z)
        perm = torch.randperm(batch, device=train_goal.device)
        goals = project_z(train_goal[perm], self.args.norm_z)
        mask = torch.rand(batch, 1, device=train_goal.device) < self.args.train_goal_ratio
        return torch.where(mask, goals, z)

    def sample_collect_z(self, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        z = sample_z(n, self.z_dim, device, self.args.norm_z)
        is_task = torch.rand(n, device=device) < self.args.collect_task_ratio
        z[is_task] = self.z_task.to(device)
        return z, is_task

    def refresh_finished_collect_z(
        self,
        collect_z: torch.Tensor,
        collect_is_task: torch.Tensor,
        finished: torch.Tensor,
    ) -> None:
        if not torch.any(finished):
            return
        n = int(finished.sum().item())
        new_z, new_is_task = self.sample_collect_z(n, collect_z.device)
        collect_z[finished] = new_z
        collect_is_task[finished] = new_is_task

    def update_targets(self) -> None:
        soft_update(self.phi, self.target_phi, self.args.encoder_target_tau)
        soft_update(self.psi, self.target_psi, self.args.encoder_target_tau)
        soft_update(self.phi_predictor, self.target_phi_predictor, self.args.predictor_target_tau)
        soft_update(self.psi_predictor, self.target_psi_predictor, self.args.predictor_target_tau)


def update_tdjepa(
    agent: TDJEPA,
    obs: torch.Tensor,
    action: torch.Tensor,
    next_obs: torch.Tensor,
    discount: torch.Tensor,
    z: torch.Tensor,
    off_diag: torch.Tensor,
    off_diag_sum: torch.Tensor,
    phi_encoder_opt: torch.optim.Optimizer,
    psi_encoder_opt: torch.optim.Optimizer,
    phi_predictor_opt: torch.optim.Optimizer,
    psi_predictor_opt: torch.optim.Optimizer,
) -> dict[str, torch.Tensor]:
    args = agent.args
    with torch.no_grad():
        next_phi = agent.target_phi(next_obs)
        next_psi = agent.target_psi(next_obs)
        next_action = agent.actor(next_phi, z, args.actor_std).sample(clip=args.stddev_clip)
        target_phi_pred = ensemble_stats(
            agent.target_phi_predictor(next_phi, z, next_action),
            args.predictor_pessimism_penalty,
        )
        target_psi_pred = ensemble_stats(
            agent.target_psi_predictor(next_psi, z, next_action),
            args.predictor_pessimism_penalty,
        )
        td_target_phi = next_psi + discount * target_phi_pred
        td_target_psi = next_phi + discount * target_psi_pred

    phi_enc = agent.phi(obs)
    psi_enc = agent.psi(obs)
    phi_preds = agent.phi_predictor(phi_enc, z, action)
    psi_preds = agent.psi_predictor(psi_enc, z, action)
    phi_td_loss = (phi_preds - td_target_phi).pow(2).sum(-1).mean()
    psi_td_loss = (psi_preds - td_target_psi).pow(2).sum(-1).mean()
    phi_orth, phi_orth_diag, phi_orth_off = orth_loss(phi_enc, off_diag, off_diag_sum)
    psi_orth, psi_orth_diag, psi_orth_off = orth_loss(psi_enc, off_diag, off_diag_sum)
    loss = phi_td_loss + psi_td_loss + args.phi_ortho_coef * phi_orth + args.psi_ortho_coef * psi_orth

    phi_predictor_opt.zero_grad(set_to_none=True)
    psi_predictor_opt.zero_grad(set_to_none=True)
    phi_encoder_opt.zero_grad(set_to_none=True)
    psi_encoder_opt.zero_grad(set_to_none=True)
    loss.backward()
    phi_predictor_opt.step()
    psi_predictor_opt.step()
    phi_encoder_opt.step()
    psi_encoder_opt.step()

    return {
        "tdjepa_loss": loss.detach(),
        "phi_tdjepa_loss": phi_td_loss.detach(),
        "psi_tdjepa_loss": psi_td_loss.detach(),
        "phi_orth_loss": phi_orth.detach(),
        "psi_orth_loss": psi_orth.detach(),
        "phi_orth_diag": phi_orth_diag.detach(),
        "psi_orth_diag": psi_orth_diag.detach(),
        "phi_orth_offdiag": phi_orth_off.detach(),
        "psi_orth_offdiag": psi_orth_off.detach(),
        "phi_norm": torch.linalg.vector_norm(phi_enc, dim=-1).mean().detach(),
        "psi_norm": torch.linalg.vector_norm(psi_enc, dim=-1).mean().detach(),
        "z_norm": torch.linalg.vector_norm(z, dim=-1).mean().detach(),
    }


def update_actor(
    agent: TDJEPA,
    obs: torch.Tensor,
    z: torch.Tensor,
    actor_opt: torch.optim.Optimizer,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        phi_enc = agent.phi(obs)
    dist = agent.actor(phi_enc, z, agent.args.actor_std)
    actor_action = dist.sample(clip=agent.args.stddev_clip)
    preds = agent.phi_predictor(phi_enc, z, actor_action)
    q = ensemble_stats((preds * z).sum(-1), agent.args.actor_pessimism_penalty)
    actor_loss = -q.mean()
    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_opt.step()
    return {"actor_loss": actor_loss.detach(), "q": q.mean().detach()}


def infer_task_z(agent: TDJEPA, rb: ReplayBuffer, device: torch.device) -> tuple[torch.Tensor, float]:
    available = rb.buffer_size if rb.full else rb.pos
    n = min(agent.args.z_infer_samples, available * rb.n_envs)
    batch = rb.sample(n)
    next_obs = agent.maybe_normalize(batch.next_observations, update=False)
    with torch.no_grad():
        psi = agent.psi(next_obs)
    z = reward_inference(psi, batch.rewards, agent.args.norm_z).squeeze(0)
    residual = (psi @ z.unsqueeze(-1) - batch.rewards).pow(2).mean().item()
    return z.to(device), residual


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
        raise RuntimeError("TD-JEPA v1 requires CUDA")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))

    agent = TDJEPA(obs_dim, action_dim, args).to(device)
    phi_encoder_opt = torch.optim.Adam(agent.phi.parameters(), lr=args.lr_phi, weight_decay=args.weight_decay)
    psi_encoder_opt = torch.optim.Adam(agent.psi.parameters(), lr=args.lr_psi, weight_decay=args.weight_decay)
    phi_predictor_opt = torch.optim.Adam(agent.phi_predictor.parameters(), lr=args.lr_predictor, weight_decay=args.weight_decay)
    psi_predictor_opt = torch.optim.Adam(agent.psi_predictor.parameters(), lr=args.lr_predictor, weight_decay=args.weight_decay)
    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=args.lr_actor, weight_decay=args.weight_decay)

    off_diag = 1.0 - torch.eye(args.batch_size, device=device)
    off_diag_sum = off_diag.sum()

    update_tdjepa_fn = update_tdjepa
    update_actor_fn = update_actor
    if args.compile:
        update_tdjepa_fn = torch.compile(update_tdjepa, mode=args.compile_mode)
        update_actor_fn = torch.compile(update_actor, mode=args.compile_mode)

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
    metrics: dict[str, torch.Tensor] = {}
    infer_mse = 0.0

    global_step = 0
    collect_z, collect_is_task = agent.sample_collect_z(args.num_envs, device)
    while global_step < args.total_timesteps:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if args.normalize_obs:
            agent.obs_rms.update(obs_t)
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = agent.act(obs_t, collect_z, mean=False).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs
        global_step += args.num_envs
        finished = torch.as_tensor(terminations | truncations, dtype=torch.bool, device=device)
        agent.refresh_finished_collect_z(collect_z, collect_is_task, finished)

        if global_step < args.learning_starts:
            continue

        if (global_step - args.learning_starts) % args.z_infer_interval < args.num_envs:
            z_task, infer_mse = infer_task_z(agent, rb, device)
            agent.z_task.copy_(z_task)
            collect_z[collect_is_task] = agent.z_task

        for _ in range(args.num_envs * args.updates_per_env_step):
            if args.compile:
                torch.compiler.cudagraph_mark_step_begin()
            batch = rb.sample(args.batch_size)
            obs_n = agent.maybe_normalize(batch.observations, update=False)
            next_obs_n = agent.maybe_normalize(batch.next_observations, update=False)
            with torch.no_grad():
                train_goal = agent.psi(next_obs_n)
                z = agent.sample_mixed_z(train_goal).clone()
            discount = args.gamma * (1.0 - batch.dones)
            metrics = update_tdjepa_fn(
                agent,
                obs_n,
                batch.actions,
                next_obs_n,
                discount,
                z,
                off_diag,
                off_diag_sum,
                phi_encoder_opt,
                psi_encoder_opt,
                phi_predictor_opt,
                psi_predictor_opt,
            )
            actor_metrics = update_actor_fn(agent, obs_n.detach(), z, actor_opt)
            metrics.update(actor_metrics)
            agent.update_targets()

        if global_step % args.log_interval < args.num_envs and metrics:
            sps = int(global_step / (time.time() - start_time))
            print("SPS:", sps)
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/z_task_norm", float(torch.linalg.vector_norm(agent.z_task)), global_step)
            writer.add_scalar("losses/reward_inference_mse", infer_mse, global_step)
            for key, value in metrics.items():
                writer.add_scalar(f"losses/{key}", float(value), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
