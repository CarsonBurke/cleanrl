# ============================================================================
# DG v26 -- Otto et al. ICLR 2021 Differentiable Trust Region Layers (arXiv:2101.09207)
#
# Paper-faithful TRL *mechanism* (not their MLP/batch/epoch recipe):
#   • Diag Gaussian actor; project (μ,σ) onto per-state trust region around π_old
#   • KL projection (Table 1 best on HalfCheetah): mean Mahalanobis closed form;
#     cov precision mix with η from dual g(η) + KKT backward (official C++ logic)
#   • Surrogate on projected π̃, no ratio clip; aux trust_region_coeff * d(pred, π̃)
#   • Non-contextual σ: write projected log-std into Parameter after each step (set_std)
#
# Your shell: ThinkTrunk, D3 critic 101/σ=0.75, actor_epochs=1, no advnorm, 16×128, lr 3e-4,
# compile reduce-overhead (mark_step_begin + output clone).
# Implementation: cleanrl/shared/trl_projection.py
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
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport, HLGaussSupport, symexp


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def _cudagraph_step_begin(enabled: bool):
    if enabled:
        torch.compiler.cudagraph_mark_step_begin()


def _clone_for_cg(t: torch.Tensor) -> torch.Tensor:
    """Clone compiled outputs so the next CUDA-graph replay does not overwrite tensors
    still needed for backward (required for reduce-overhead multi-step training)."""
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
    # Full agent torch.compile; reduce-overhead = CUDA graphs. Requires mark_step_begin + output
    # clone each train step (see training loop). dynamic=False for fixed B.
    compile: bool = True
    compile_mode: str = "reduce-overhead"

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
    actor_epochs: int = 1  # YOUR stack (noadvnorm / 347) — not paper's 20
    critic_epochs: int = 10
    norm_adv: bool = False  # YOUR noadvnorm shell — not paper's True
    max_grad_norm: float = 0.5
    ent_coef: float = 0.0

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16
    critic_init_tau: float = 0.5

    critic_d3bucket: bool = True
    critic_num_bins: int = 101
    critic_v_min: float = -9.90353755128617
    critic_v_max: float = 9.90353755128617
    critic_sigma_ratio: float = 0.75
    critic_symlog: bool = True
    critic_support_is_edges: bool = True
    critic_decode: str = "expected_scalar"
    reward_norm: bool = False

    # --- paper TRL (BaseProjectionLayer defaults for KL) ---
    proj_type: str = "kl"  # only kl is fully paper-faithful dual here; kept for CLI clarity
    trl_mean_bound: float = 0.03   # official default mean_bound
    trl_cov_bound: float = 0.001   # official default cov_bound
    trl_coeff: float = 1.0         # paper trust_region_coeff
    trl_scale_prec: bool = True
    log_std_init: float = 0.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0

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
    def __init__(self, envs, num_bins, hidden=64, k_blocks=3, n_experts=16,
                 v_min=-10.0, v_max=10.0, critic_init_tau=0.5, log_std_init=0.0):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = hidden
        self.act_dim = act_dim
        self.critic_trunk = ThinkTrunk(obs_dim, H, k_blocks, n_experts)
        self.actor_trunk = ThinkTrunk(obs_dim, H, k_blocks, n_experts)
        self.actor_mean = layer_init(nn.Linear(H, act_dim), std=0.01)
        # Non-contextual std (paper mujoco_config contextual_std=false)
        self.actor_logstd = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        self.critic_head = layer_init(nn.Linear(H, num_bins), std=0.1)
        with torch.no_grad():
            zc = torch.linspace(v_min, v_max, num_bins)
            self.critic_head.bias.copy_(-0.5 * (zc / critic_init_tau) ** 2)

    def get_value(self, x):
        return self.critic_head(self.critic_trunk(x))

    def get_mean_logstd(self, x):
        h = self.actor_trunk(x)
        mean = self.actor_mean(h)
        # Broadcast log-std (clone after compiled forward when using CUDA graphs)
        logstd = self.actor_logstd.unsqueeze(0).expand(mean.shape[0], -1)
        return mean, logstd

    def get_action_and_value(self, x, action=None):
        mean, logstd = self.get_mean_logstd(x)
        std = logstd.exp()
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value_logits = self.get_value(x)
        return action, logp, entropy, value_logits, mean.detach(), logstd.detach()


from cleanrl.shared.trl_projection import (
    project_policy_kl,
    trust_region_aux_loss,
    analytic_kl_diag as analytic_kl,
)

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.proj_type == "kl", "v26 paper path implements KL projection only"
    assert args.critic_decode in ("expected_scalar", "scalar")
    # W2 paper mujoco bounds if user picks w2 without overriding (optional note only)

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
        args.critic_v_min, args.critic_v_max, args.critic_init_tau, args.log_std_init,
    ).to(device)

    # Optimizers bind eager Parameters before compile wraps the module.
    actor_params = list(agent.actor_trunk.parameters()) + list(agent.actor_mean.parameters()) + [agent.actor_logstd]
    critic_params = list(agent.critic_trunk.parameters()) + list(agent.critic_head.parameters())
    actor_base_lr = args.actor_lr if args.actor_lr is not None else args.learning_rate
    actor_opt = optim.Adam(actor_params, lr=actor_base_lr, eps=1e-5)
    critic_opt = optim.Adam(critic_params, lr=args.learning_rate, eps=1e-5)

    use_cg = bool(args.compile and args.compile_mode == "reduce-overhead")
    if args.compile:
        torch.set_float32_matmul_precision("high")
        # Full agent compile. reduce-overhead uses CUDA graphs: every train step must call
        # cudagraph_mark_step_begin() and clone outputs used for backward (below).
        agent = torch.compile(agent, mode=args.compile_mode, dynamic=False)
        print(f"[v26] torch.compile(agent, mode={args.compile_mode!r}, dynamic=False) proj={args.proj_type} cg={use_cg}")

    def agent_get_value(x, need_grad: bool):
        _cudagraph_step_begin(use_cg)
        y = agent.get_value(x)
        return _clone_for_cg(y) if (use_cg and need_grad) else y

    def agent_get_mean_logstd(x, need_grad: bool):
        _cudagraph_step_begin(use_cg)
        mean, logstd = agent.get_mean_logstd(x)
        if use_cg and need_grad:
            mean, logstd = _clone_for_cg(mean), _clone_for_cg(logstd)
        return mean, logstd

    def agent_act(x):
        _cudagraph_step_begin(use_cg)
        return agent.get_action_and_value(x)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    old_means = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)
    old_logstds = torch.zeros((args.num_steps, args.num_envs, act_dim), device=device)

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
                action, logprob, _, value_logits, mean, logstd = agent_act(next_obs)
                values[step] = value_logits_to_scalar(value_logits).flatten()
            actions[step] = action
            logprobs[step] = logprob
            old_means[step] = mean
            old_logstds[step] = logstd

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
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_old_means = old_means.reshape(-1, act_dim)
        b_old_logstds = old_logstds.reshape(-1, act_dim)

        # Fixed-shape views for CUDA graphs (always full minibatch except possibly last — use exact mb size)
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
        logstd_param = agent.actor_logstd
        last_logstd_p = None

        for _ in range(args.actor_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = b_inds[start : start + args.minibatch_size]
                mean, logstd = agent_get_mean_logstd(b_obs[mb], need_grad=True)
                logstd = logstd.clamp(args.log_std_min, args.log_std_max)
                omean, ologstd = b_old_means[mb], b_old_logstds[mb]

                mean_p, logstd_p, mp, cp = project_policy_kl(
                    mean, logstd, omean, ologstd,
                    args.trl_mean_bound, args.trl_cov_bound,
                    args.log_std_min, args.log_std_max,
                )
                last_logstd_p = logstd_p
                dist_p = Normal(mean_p, logstd_p.exp())
                new_logp = dist_p.log_prob(b_actions[mb]).sum(-1)
                entropy = dist_p.entropy().sum(-1)
                old_logp = Normal(omean, ologstd.exp()).log_prob(b_actions[mb]).sum(-1)

                mb_adv = b_advantages[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Unclipped IS surrogate (paper importance_ratio_clip=0); grads through projection
                ratio = (new_logp - old_logp).exp()
                pg_loss = -(ratio * mb_adv).mean()
                # Paper §4.4: aux is supervised only — projection detached (official get_trust_region_loss)
                tr_loss = args.trl_coeff * trust_region_aux_loss(
                    mean, logstd, mean_p, logstd_p, contextual_std=False
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
                    kl_posts.append(analytic_kl(mean_p, logstd_p, omean, ologstd).mean().item())
                    mean_parts.append(mp.mean().item() if torch.is_tensor(mp) else float(mp))
                    cov_parts.append(cp.mean().item() if torch.is_tensor(cp) else float(cp))

        # Official pg.py: hard-set non-contextual std once after all policy minibatches
        if last_logstd_p is not None:
            with torch.no_grad():
                logstd_param.copy_(last_logstd_p[0].clamp(args.log_std_min, args.log_std_max))

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
        writer.add_scalar("losses/trl_mean_part", float(np.mean(mean_parts)), global_step)
        writer.add_scalar("losses/trl_cov_part", float(np.mean(cov_parts)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        print(f"SPS: {sps}  kl_proj={float(np.mean(kl_posts)):.4f} max={float(np.max(kl_posts)):.4f} proj={args.proj_type}")
        writer.add_scalar("charts/SPS", sps, global_step)

    if args.save_model:
        path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), path)
        print("saved", path)
    envs.close()
    writer.close()
