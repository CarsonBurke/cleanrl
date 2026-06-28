# ============================================================================
# DELIGHTFUL POLICY GRADIENT (DG) -- clean Beta-policy implementation.
# Paper: arXiv:2603.14608v1 (Delightful Policy Gradient). Algorithm 1/2:
#
#     ell_t = clip(-log pi_theta(A_t | H_t), -C, C)     # surprisal (C=10)
#     chi_t = U_t * ell_t                               # "delight" = advantage * surprisal
#     w_t   = sigmoid(chi_t / eta)                       # gate, eta = 1, DETACHED
#     dtheta += w_t * U_t * grad_theta log pi(A_t)       # gated score-function update
#
# Intuition (paper): "amplify rare successes, suppress rare failures." A breakthrough
# (rare action, U>0) has high chi -> w->1; a blunder (rare action, U<0) has low chi -> w->0.
# This REQUIRES the surprisal ell >= 0 (rare action => HIGH ell), so that sign(chi)=sign(U)
# and the gate's asymmetry lands on the advantage sign as the paper draws it.
#
# WHY A BETA NEEDS A PEAK-REFERENCED SURPRISAL (the crux of the beta-version).
# The paper is written for Gaussian/discrete policies, where -log pi is dominated by the
# (>=0) Mahalanobis/− distance term and is mostly positive, so ell>=0 holds. For a Beta the
# log-NORMALIZER log B(alpha,beta) goes strongly NEGATIVE as the policy concentrates, dragging
# -log pi(a) negative (measured ~ -9.5 on a trained policy). The literal paper ell then makes
# chi = U*ell carry the OPPOSITE sign of U -- the gate INVERTS (suppresses breakthroughs,
# amplifies blunders). Fix: reference the surprisal to the policy's own PEAK,
#     ell = log pi(mode) - log pi(a)  >= 0,
# which cancels log B(alpha,beta) exactly and leaves the pure, >=0, concentration-RESPONSIVE
# "distance from the most-likely action" -- the exact Beta analog of the Gaussian Mahalanobis
# term the paper's -log pi reduces to. (`--dg-surprisal raw` keeps the literal paper -log pi
# clip for the inversion ablation.)
#
# FAITHFUL, CLEAN, ON RAW ADVANTAGES. No PPO ratio/clip, no advantage normalization
# (U = raw GAE), no rankgauss, no trust region, no entropy crutch (ent_coef=0). Standard
# 64x64 tanh MLP on SEPARATE actor/critic networks, standard max_grad_norm=0.5. The actor
# takes ONE faithful score pass per rollout (each sample used once, on-policy -- nothing to
# drift against, so no ratio needed); the critic refits for `critic_epochs` (pure regression,
# no off-policy bias) to keep the advantages informative.
#
# v2: HL-GAUSS DISTRIBUTIONAL CRITIC (replaces v1's scalar MSE head). The value head emits a
# categorical over a symlog support; targets are HL-Gauss-projected (Gaussian-smoothed two-hot)
# lambda-returns trained by cross-entropy, decoded as E[value]. Distributional regression is
# better-calibrated than MSE -> cleaner, lower-variance advantages, so the gate's chi = U*ell
# reads real breakthrough/blunder signal instead of value noise.
#
# WHY THE GATE SHOULD NOW PAY OFF. v1 showed plain score (no gate) on a clean scalar critic
# beats base PPO early but its KL grows unbounded (single-pass PG has no step-size control) and
# it destabilizes. The DG gate w = sigmoid(chi/eta) SATURATES on large |chi|, so it caps the
# per-sample gradient of extreme (large-advantage, surprising) samples -- an implicit,
# advantage-aware step limiter the ungated run lacks. With a calibrated HL-Gauss critic feeding
# clean advantages, the gate's stabilization is hypothesized to let DG overtake plain score as
# advantages grow late in training.
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

from cleanrl.shared.hl_gauss import HLGaussSupport

EPS = 1e-6  # clamp for Beta samples / mode to keep log_prob finite


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

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8000000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    actor_epochs: int = 1            # FAITHFUL DG: one on-policy score pass per rollout
    critic_epochs: int = 10          # critic is pure regression -> refit more for good advantages
    norm_adv: bool = False           # FAITHFUL: gate & score see the RAW advantage U = R - b
    ent_coef: float = 0.0            # paper DG uses no entropy bonus
    max_grad_norm: float = 0.5       # standard PPO clip (NOT the 0.25 dual-clip of the heavy base)
    target_kl: float = None

    # HL-Gauss distributional critic
    critic_num_bins: int = 101       # categorical support size
    critic_v_min: float = -10.0      # support min (symlog space): symexp(-10) ~ -2.2e4 raw
    critic_v_max: float = 10.0       # support max (symlog space)
    critic_sigma_ratio: float = 0.75 # HL-Gauss label sigma as a fraction of bin width (paper sweet spot)
    critic_symlog: bool = True       # symlog-scale targets (DreamerV3-style) -> robust to value range

    # DG-specific
    dg_use_gate: bool = True         # False => w=1 (plain REINFORCE score control / DG ablation)
    dg_surprisal: str = "peak_ref"   # "peak_ref" (ell=logp(mode)-logp(a)>=0, beta-correct) | "raw" (paper -logpi clip; INVERTS on beta)
    dg_eta: float = 1.0              # temperature eta in w = sigmoid(chi/eta)
    dg_clip: float = 10.0            # paper C: clip on the surprisal ell

    # to be filled in runtime
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
    """Beta-policy actor + scalar value critic, SEPARATE networks (no shared trunk).

    The action a in [-1, 1] is a fixed affine map of the native Beta sample z in (0, 1):
    a = 2z - 1. The Jacobian is the per-dim constant log 2, so it cancels in (i) the score
    gradient grad log pi(a) = grad log pi_z(z) and (ii) the peak-referenced surprisal
    log pi(mode) - log pi(a); we therefore work in native z-space throughout.
    """

    def __init__(self, envs, num_bins):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        # Distributional value head: emits `num_bins` logits over the HL-Gauss support.
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_bins), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        # alpha, beta = 1 + softplus(.) > 1  => strictly unimodal Beta with an interior mode.
        self.alpha_head = layer_init(nn.Linear(64, act_dim), std=0.01)
        self.beta_head = layer_init(nn.Linear(64, act_dim), std=0.01)

    def get_value(self, x):
        """Returns raw value LOGITS over the categorical support (caller decodes/projects)."""
        return self.critic(x)

    def _dist(self, x):
        h = self.actor(x)
        alpha = 1.0 + F.softplus(self.alpha_head(h))
        beta = 1.0 + F.softplus(self.beta_head(h))
        return Beta(alpha, beta)

    def get_action_and_value(self, x, z=None):
        """Returns (z_native, action, logp, ell_peak_ref, entropy, value_logits).

        ell_peak_ref = log pi(mode) - log pi(z) >= 0 is the >=0 surprisal; the caller clips it
        to [0, dg_clip] and applies the gate. value_logits are the categorical value logits
        (decoded via the HL-Gauss support by the caller).
        """
        dist = self._dist(x)
        if z is None:
            z = dist.sample().clamp(EPS, 1.0 - EPS)
        logp = dist.log_prob(z).sum(1)
        # Beta mode = (a-1)/(a+b-2); a,b > 1 so the denominator is > 0 (floored for safety).
        a, b = dist.concentration1, dist.concentration0
        mode = ((a - 1.0) / (a + b - 2.0).clamp_min(EPS)).clamp(EPS, 1.0 - EPS)
        logp_mode = dist.log_prob(mode).sum(1)
        ell = logp_mode - logp  # >= 0: the mode is the per-dim max-density point
        entropy = dist.entropy().sum(1)
        action = 2.0 * z - 1.0
        return z, action, logp, ell, entropy, self.critic(x)


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

    # HL-Gauss categorical support for the distributional value critic.
    hlg = HLGaussSupport(
        num_bins=args.critic_num_bins,
        v_min=args.critic_v_min,
        v_max=args.critic_v_max,
        sigma_ratio=args.critic_sigma_ratio,
        device=device,
        use_symlog=args.critic_symlog,
    )

    agent = Agent(envs, args.critic_num_bins).to(device)
    # Separate optimizers: the critic trains for many epochs (pure regression), the actor for
    # a single faithful DG pass -- decoupling them keeps each clean and avoids a shared trunk.
    actor_params = list(agent.actor.parameters()) + list(agent.alpha_head.parameters()) + list(agent.beta_head.parameters())
    critic_params = list(agent.critic.parameters())
    actor_opt = optim.Adam(actor_params, lr=args.learning_rate, eps=1e-5)
    critic_opt = optim.Adam(critic_params, lr=args.learning_rate, eps=1e-5)

    # Storage: store the NATIVE beta sample z (replayed to recompute logp at the same draw).
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
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
            actor_opt.param_groups[0]["lr"] = lrnow
            critic_opt.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                z, action, logprob, _, _, value_logits = agent.get_action_and_value(next_obs)
                values[step] = hlg.to_scalar(value_logits).flatten()
            zs[step] = z
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

        # GAE (computed once from rollout values) -- U = raw advantage.
        with torch.no_grad():
            next_value = hlg.to_scalar(agent.get_value(next_obs)).reshape(1, -1)
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
        b_zs = zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)

        # ---- Critic update: many epochs of pure MSE regression (no policy / off-policy bias) ----
        for epoch in range(args.critic_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]
                value_logits = agent.get_value(b_obs[mb_inds])
                # HL-Gauss distributional regression: CE(predicted, Gaussian-smoothed target).
                target_probs = hlg.project(b_returns[mb_inds])
                v_loss = -(target_probs * F.log_softmax(value_logits, dim=-1)).sum(-1).mean()
                critic_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
                critic_opt.step()

        # ---- Actor update: faithful DG gated score pass(es) on RAW advantages ----
        gate_means, surp_means, chi_stds, sign_agrees = [], [], [], []
        approx_kl = torch.zeros((), device=device)
        for epoch in range(args.actor_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]
                _, _, newlogprob, ell, entropy, _ = agent.get_action_and_value(b_obs[mb_inds], b_zs[mb_inds])

                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                if args.dg_surprisal == "raw":  # paper-literal: ell = clip(-log pi, -C, C); INVERTS on a sharp beta
                    surprisal = (-newlogprob).clamp(-args.dg_clip, args.dg_clip)
                else:  # peak_ref: ell = logp(mode) - logp(a) >= 0, clipped to [0, C]
                    surprisal = ell.clamp(0.0, args.dg_clip)
                chi = mb_adv * surprisal
                w = torch.sigmoid(chi / args.dg_eta).detach()
                if not args.dg_use_gate:
                    w = torch.ones_like(w)

                # Faithful DG: dtheta += w * U * grad log pi  <=>  minimize -(w * U * log pi).
                pg_loss = -(w * mb_adv * newlogprob).mean()
                actor_loss = pg_loss - args.ent_coef * entropy.mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
                actor_opt.step()

                with torch.no_grad():
                    gate_means.append(w.mean().item())
                    surp_means.append(surprisal.mean().item())
                    chi_stds.append(chi.std().item())
                    sign_agrees.append(((chi > 0) == (mb_adv > 0)).float().mean().item())
                    logratio = newlogprob - b_logprobs[mb_inds]
                    approx_kl = ((logratio.exp() - 1) - logratio).mean()
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", actor_opt.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/dg_gate_mean", float(np.mean(gate_means)), global_step)
        writer.add_scalar("charts/dg_surprisal_mean", float(np.mean(surp_means)), global_step)
        writer.add_scalar("charts/dg_chi_std", float(np.mean(chi_stds)), global_step)
        writer.add_scalar("charts/dg_sign_agree", float(np.mean(sign_agrees)), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
