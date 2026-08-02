# ============================================================================
# DG v26 -- Otto et al. ICLR 2021 Differentiable Trust Region Layers on v25 Beta.
#
# Paper-faithful TRL (official boschresearch/trust-region-layers KLProjectionLayer):
#   • Split reverse-KL into mean part + cov part; separate bounds (default 0.03 / 0.001)
#   • Mean: closed-form Mahalanobis projection (Eq. 6 / mean_projection)
#   • Cov: diag precision dual η + KKT backward (DiagCovOnlyKLProjection)
#   • Surrogate on projected π̃, importance_ratio_clip=0
#   • Aux = get_trust_region_loss: reverse KL moments(π_θ || π̃.detach()) * trust_region_coeff
#
# Beta (paper only defines Gaussian layers): encode each Beta by its (μ, σ²) moments in
# native z-space, run the *exact* paper KL layer on those moments, decode back to (α,β).
# Both α,β are state-dependent → contextual_std=True (no set_std).
#
# Shell (your stack, not paper MLP): ThinkTrunk Beta, D3 critic 101/σ=0.75,
# actor_epochs=1, no advnorm, compile reduce-overhead.
# ============================================================================
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport, HLGaussSupport, symexp
from cleanrl.shared.trl_projection import (
    project_policy_beta_kl,
    trust_region_aux_loss_beta_moments,
    beta_kl_reverse,
    beta_to_mean_logstd,
    analytic_kl_diag,
)

EPS = 1e-6


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def _cudagraph_step_begin(enabled: bool):
    if enabled:
        torch.compiler.cudagraph_mark_step_begin()


def _clone_for_cg(t: torch.Tensor) -> torch.Tensor:
    return t.clone() if torch.is_tensor(t) else t


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
    compile: bool = True
    # default (not reduce-overhead): CUDA graphs fight the dual Newton/bisection autograd
    compile_mode: str = "default"

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8000000
    learning_rate: float = 3e-4
    actor_lr: float = None
    num_envs: int = 16
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    actor_epochs: int = 1
    critic_epochs: int = 10
    norm_adv: bool = False
    max_grad_norm: float = 0.5
    ent_coef: float = 0.0

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16
    critic_init_tau: float = 0.5

    # Match paper TRL run critic
    critic_d3bucket: bool = True
    critic_num_bins: int = 101
    critic_v_min: float = -9.90353755128617
    critic_v_max: float = 9.90353755128617
    critic_sigma_ratio: float = 0.75
    critic_symlog: bool = True
    critic_support_is_edges: bool = True
    critic_decode: str = "expected_scalar"
    reward_norm: bool = False

    # Official BaseProjectionLayer / KLProjectionLayer defaults
    trl_mean_bound: float = 0.03   # paper mean_bound
    trl_cov_bound: float = 0.001   # paper cov_bound
    trl_coeff: float = 1.0         # paper trust_region_coeff
    # Paper importance_ratio_clip=0 (unclipped IS on π̃) — hard-coded in loop

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def make_env(env_id, idx, capture_video, run_name, gamma, reward_norm=True):
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
        if reward_norm:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


def _branch_body(H):
    return nn.Sequential(
        layer_init(nn.Linear(H, H)),
        ReLUSquared(),
        layer_init(nn.Linear(H, H)),
    )


class ThinkBlock(nn.Module):
    def __init__(self, in_dim, H, n_experts):
        super().__init__()
        self.n_experts = n_experts
        self.in_proj = layer_init(nn.Linear(in_dim, H))
        self.resid_gate = nn.Parameter(torch.full((H,), 4.0))
        self.dense_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.dense = _branch_body(H)
        self.moe_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(H, n_experts))
        self.experts = nn.ModuleList([_branch_body(H) for _ in range(n_experts)])

    def forward(self, cat_feats, x0):
        x = self.in_proj(cat_feats)
        g = torch.sigmoid(self.resid_gate)
        x_in = g * x + (1.0 - g) * x0
        d_dense = self.dense(self.dense_norm(x_in))
        m_in = self.moe_norm(x_in)
        weights = torch.softmax(self.gate(m_in), dim=-1)
        all_out = torch.stack([e(m_in) for e in self.experts], dim=1)
        d_moe = (weights.unsqueeze(-1) * all_out).sum(dim=1)
        return x_in + d_dense + d_moe


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim, H, K, n_experts):
        super().__init__()
        self.entry = layer_init(nn.Linear(in_dim, H))
        self.blocks = nn.ModuleList([ThinkBlock(H * (k + 1), H, n_experts) for k in range(K)])
        cat_dim = H * (K + 1)
        self.out_norm = nn.RMSNorm(cat_dim, elementwise_affine=False)
        self.out_proj = layer_init(nn.Linear(cat_dim, H))

    def forward(self, x):
        x0 = self.entry(x)
        feats = [x0]
        for block in self.blocks:
            feats.append(block(torch.cat(feats, dim=-1), x0))
        return self.out_proj(self.out_norm(torch.cat(feats, dim=-1)))


class Agent(nn.Module):
    """v25 Beta actor + distributional critic (separate ThinkTrunks)."""

    def __init__(self, envs, num_bins, hidden=64, k_blocks=3, n_experts=16,
                 v_min=-10.0, v_max=10.0, critic_init_tau=0.5):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = hidden
        self.act_dim = act_dim
        self.critic_trunk = ThinkTrunk(obs_dim, H, k_blocks, n_experts)
        self.actor_trunk = ThinkTrunk(obs_dim, H, k_blocks, n_experts)
        self.alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
        self.critic_head = layer_init(nn.Linear(H, num_bins), std=0.1)
        with torch.no_grad():
            zc = torch.linspace(v_min, v_max, num_bins)
            self.critic_head.bias.copy_(-0.5 * (zc / critic_init_tau) ** 2)

    def get_value(self, x):
        return self.critic_head(self.critic_trunk(x))

    def get_alpha_beta(self, x):
        h = self.actor_trunk(x)
        alpha = 1.0 + F.softplus(self.alpha_head(h))
        beta = 1.0 + F.softplus(self.beta_head(h))
        return alpha, beta

    def get_action_and_value(self, x, z=None):
        alpha, beta = self.get_alpha_beta(x)
        dist = Beta(alpha, beta)
        if z is None:
            z = dist.sample().clamp(EPS, 1.0 - EPS)
        logp = dist.log_prob(z).sum(-1)
        entropy = dist.entropy().sum(-1)
        action = 2.0 * z - 1.0
        return z, action, logp, entropy, self.get_value(x), alpha.detach(), beta.detach()


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.critic_decode in ("expected_scalar", "scalar")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name, entity=args.wandb_entity,
            sync_tensorboard=True, config=vars(args), name=run_name, monitor_gym=True, save_code=True,
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
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma, args.reward_norm)
         for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)
    act_dim = int(np.prod(envs.single_action_space.shape))

    if args.critic_d3bucket:
        hlg = Dreamer3BucketHLGaussSupport(
            args.critic_num_bins, args.critic_v_min, args.critic_v_max, args.critic_sigma_ratio, device
        )
    else:
        hlg = HLGaussSupport(
            args.critic_num_bins, args.critic_v_min, args.critic_v_max, args.critic_sigma_ratio,
            device, use_symlog=args.critic_symlog, support_is_edges=args.critic_support_is_edges,
        )

    def value_logits_to_scalar(logits):
        if args.critic_d3bucket:
            if args.critic_decode == "scalar":
                probs = torch.softmax(logits, dim=-1)
                return symexp((probs * hlg.coord_support).sum(-1))
            return hlg.to_scalar(logits)
        return hlg.to_scalar(logits) if args.critic_decode == "scalar" else hlg.to_expected_scalar(logits)

    agent = Agent(
        envs, args.critic_num_bins, args.hidden, args.k_blocks, args.n_experts,
        args.critic_v_min, args.critic_v_max, args.critic_init_tau,
    ).to(device)

    actor_params = (
        list(agent.actor_trunk.parameters())
        + list(agent.alpha_head.parameters())
        + list(agent.beta_head.parameters())
    )
    critic_params = list(agent.critic_trunk.parameters()) + list(agent.critic_head.parameters())
    actor_base_lr = args.actor_lr if args.actor_lr is not None else args.learning_rate
    actor_opt = optim.Adam(actor_params, lr=actor_base_lr, eps=1e-5)
    critic_opt = optim.Adam(critic_params, lr=args.learning_rate, eps=1e-5)

    use_cg = bool(args.compile and args.compile_mode == "reduce-overhead")
    if args.compile:
        torch.set_float32_matmul_precision("high")
        agent = torch.compile(agent, mode=args.compile_mode, dynamic=False)
        print(
            f"[v26-trl-beta] torch.compile mode={args.compile_mode!r} cg={use_cg} "
            f"mean_bound={args.trl_mean_bound} cov_bound={args.trl_cov_bound} coeff={args.trl_coeff}"
        )

    def agent_get_value(x, need_grad: bool):
        _cudagraph_step_begin(use_cg)
        y = agent.get_value(x)
        return _clone_for_cg(y) if (use_cg and need_grad) else y

    def agent_get_alpha_beta(x, need_grad: bool):
        _cudagraph_step_begin(use_cg)
        a, b = agent.get_alpha_beta(x)
        if use_cg and need_grad:
            a, b = _clone_for_cg(a), _clone_for_cg(b)
        return a, b

    def agent_act(x):
        _cudagraph_step_begin(use_cg)
        return agent.get_action_and_value(x)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    zs = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    old_alphas = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
    old_betas = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            actor_opt.param_groups[0]["lr"] = frac * actor_base_lr
            critic_opt.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                z, action, logprob, _, value_logits, alpha, beta = agent_act(next_obs)
                values[step] = value_logits_to_scalar(value_logits).flatten()
            zs[step] = z
            logprobs[step] = logprob
            old_alphas[step] = alpha
            old_betas[step] = beta

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            next_value = value_logits_to_scalar(agent_get_value(next_obs, need_grad=False)).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
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
        b_zs = zs.reshape(-1, act_dim)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_old_a = old_alphas.reshape(-1, act_dim)
        b_old_b = old_betas.reshape(-1, act_dim)

        assert args.batch_size % args.minibatch_size == 0
        b_inds = np.arange(args.batch_size)

        for _ in range(args.critic_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start : start + args.minibatch_size]
                value_logits = agent_get_value(b_obs[mb], need_grad=True)
                v_loss = -(hlg.project(b_returns[mb]) * F.log_softmax(value_logits, dim=-1)).sum(-1).mean()
                critic_opt.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
                critic_opt.step()

        pg_losses, tr_losses, entropies = [], [], []
        kl_posts, mean_parts, cov_parts = [], [], []
        moment_kl_posts = []

        for _ in range(args.actor_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start : start + args.minibatch_size]
                alpha, beta = agent_get_alpha_beta(b_obs[mb], need_grad=True)
                oa, ob = b_old_a[mb], b_old_b[mb]
                z = b_zs[mb].clamp(EPS, 1.0 - EPS)

                # Paper KL layer on Beta moments → projected Beta(α̃,β̃)
                alpha_p, beta_p, mp, cp, mean, logstd, mean_p, logstd_p = project_policy_beta_kl(
                    alpha, beta, oa, ob,
                    args.trl_mean_bound, args.trl_cov_bound,
                )
                dist_p = Beta(alpha_p, beta_p)
                new_logp = dist_p.log_prob(z).sum(-1)
                entropy = dist_p.entropy().sum(-1)
                old_logp = Beta(oa, ob).log_prob(z).sum(-1)

                mb_adv = b_advantages[mb]
                if args.norm_adv:
                    # paper mujoco_config norm_advantages=true (shell default False)
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Unclipped IS on projected π̃ (paper importance_ratio_clip=0)
                ratio = (new_logp - old_logp).exp()
                pg_loss = -(ratio * mb_adv).mean()
                # Official get_trust_region_loss on moment Gaussians (projection detached inside)
                tr_loss = args.trl_coeff * trust_region_aux_loss_beta_moments(
                    mean, logstd, mean_p, logstd_p
                )
                loss = pg_loss + tr_loss - args.ent_coef * entropy.mean()

                actor_opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
                actor_opt.step()

                with torch.no_grad():
                    pg_losses.append(pg_loss.item())
                    tr_losses.append(tr_loss.item())
                    entropies.append(entropy.mean().item())
                    # True Beta KL after projection (diagnostic)
                    kl_post = beta_kl_reverse(alpha_p, beta_p, oa, ob).sum(-1)
                    kl_posts.append(kl_post.mean().item())
                    # Paper moment reverse-KL of projected vs old (mean+cov)
                    omean, ologstd = beta_to_mean_logstd(oa, ob)
                    moment_kl_posts.append(
                        analytic_kl_diag(mean_p, logstd_p, omean, ologstd).mean().item()
                    )
                    mean_parts.append(mp.mean().item() if torch.is_tensor(mp) else float(mp))
                    cov_parts.append(cp.mean().item() if torch.is_tensor(cp) else float(cp))

        y_pred = values.reshape(-1).cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", actor_opt.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", float(np.mean(pg_losses)), global_step)
        writer.add_scalar("losses/trust_region_loss", float(np.mean(tr_losses)), global_step)
        writer.add_scalar("losses/entropy", float(np.mean(entropies)), global_step)
        writer.add_scalar("losses/analytic_kl_proj", float(np.mean(kl_posts)), global_step)
        writer.add_scalar("losses/analytic_kl_proj_max", float(np.max(kl_posts)), global_step)
        writer.add_scalar("losses/moment_kl_proj", float(np.mean(moment_kl_posts)), global_step)
        writer.add_scalar("losses/trl_mean_part", float(np.mean(mean_parts)), global_step)
        writer.add_scalar("losses/trl_cov_part", float(np.mean(cov_parts)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        print(
            f"SPS: {sps}  beta_kl_proj={float(np.mean(kl_posts)):.4f} "
            f"moment_kl={float(np.mean(moment_kl_posts)):.4f} "
            f"mean_part={float(np.mean(mean_parts)):.4f} cov_part={float(np.mean(cov_parts)):.4f}"
        )
        writer.add_scalar("charts/SPS", sps, global_step)

    if args.save_model:
        path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), path)
        print("saved", path)
    envs.close()
    writer.close()
