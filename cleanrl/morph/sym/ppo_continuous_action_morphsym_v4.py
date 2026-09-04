# PPO + Morphogenic Symmetry v4.
#
# METHOD. v3 with a gated FFN. v3's per-unit template MLP was a plain
# Linear -> ReLU^2 -> Linear. v4 swaps it for a GLU-style gated variant
# ("ReGLU^2"):  out = ( ReLU^2(x @ W1) (elementwise*) (x @ W3) ) @ W2 .
# GLU-gated feedforwards (SwiGLU/GeGLU, PaLM/LLaMA) reliably beat plain FFNs;
# the multiplicative gate gives per-hidden-unit input-dependent modulation. The
# lineage's ReLU^2 activation is kept on the gate branch. The hidden width is
# scaled by 2/3 so the gated FFN (three matrices) has the SAME parameter count as
# v3's two-matrix FFN -- an isolated, fair test of gating alone. Everything else
# (learned positional field, static template weight-tying, multi-head positional
# routing, no-pooling output units, compute pricing) is unchanged from v3.
#
# HYPOTHESIS. Multiplicative gating inside the shared templates increases per-unit
# expressivity at matched params and improves over v3.
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

from cleanrl.morph.compute.ppo_continuous_action_morphcompute_v9 import (
    ReLUSquared,
    layer_init,
    make_env,
    mean_stat,
)
from cleanrl.morph.compute.ppo_continuous_action_morphcompute_v18 import entmax15, signed_loss_with_safe_compute
from cleanrl.morph.sym.ppo_continuous_action_morphsym_v3 import (
    Args as V3Args,
    Agent as V3Agent,
    SymGraph,
)


@dataclass
class Args(V3Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    glu_hidden_frac: float = 2.0 / 3.0
    """scale the gated-FFN hidden width by this so param count matches v3's plain FFN"""


class GatedTiedFFN(nn.Module):
    """Per-unit gated MLP whose weights are a static entmax mixture over M shared templates,
    keyed by the unit's positional features phi(p). ReGLU^2: (ReLU^2(x@W1) * (x@W3)) @ W2."""

    def __init__(self, D, H, M, phi_dim):
        super().__init__()
        self.D, self.H, self.M = D, H, M
        self.assign = layer_init(nn.Linear(phi_dim, M), std=0.5)
        self.W1 = nn.Parameter(torch.empty(M, D, H))  # gate branch (activated)
        self.b1 = nn.Parameter(torch.zeros(M, H))
        self.W3 = nn.Parameter(torch.empty(M, D, H))  # value branch (linear)
        self.b3 = nn.Parameter(torch.zeros(M, H))
        self.W2 = nn.Parameter(torch.empty(M, H, D))  # output projection
        self.b2 = nn.Parameter(torch.zeros(M, D))
        for m in range(M):
            nn.init.orthogonal_(self.W1[m], gain=np.sqrt(2.0))
            nn.init.orthogonal_(self.W3[m], gain=1.0)
            nn.init.orthogonal_(self.W2[m], gain=0.5)
        self.norm = nn.LayerNorm(D)
        self.act = ReLUSquared()

    def assignment(self, phi):
        return entmax15(self.assign(phi), dim=-1)  # (U, M)

    def forward(self, h, phi):
        A = self.assignment(phi)
        x = self.norm(h)
        gate = self.act(torch.einsum("bud,mdh->bumh", x, self.W1) + self.b1)
        value = torch.einsum("bud,mdh->bumh", x, self.W3) + self.b3
        hidden = gate * value
        out = torch.einsum("bumh,mhd->bumd", hidden, self.W2) + self.b2
        delta = torch.einsum("um,bumd->bud", A, out)
        return h + delta, A


class SymGraph4(SymGraph):
    """v3 SymGraph with the plain TiedFFN replaced by the gated ReGLU^2 FFN (matched params)."""

    def __init__(self, obs_dim, args, n_out):
        super().__init__(obs_dim, args, n_out)
        glu_H = max(1, int(round(args.hidden_dim * args.glu_hidden_frac)))
        self.field = GatedTiedFFN(self.D, glu_H, self.M, self.phi_dim)


class Agent(V3Agent):
    def __init__(self, envs, args=None):
        if args is None:
            args = Args()
        super().__init__(envs, args)
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.actor = SymGraph4(obs_dim, args, n_out=action_dim)
        self.critic = SymGraph4(obs_dim, args, n_out=1)


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
    last_stats = None

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, z, logprob, _, value, _, _, last_stats = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
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
        b_zs = zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, _, newlogprob, entropy, newvalue, actor_compute, critic_compute, last_stats = agent.get_action_and_value(
                    b_obs[mb_inds], b_zs[mb_inds]
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

                actor_multiplier = agent.compute_multiplier(actor_compute, args)
                critic_multiplier = agent.compute_multiplier(critic_compute, args)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss_per_sample = torch.max(pg_loss1, pg_loss2)
                pg_loss = signed_loss_with_safe_compute(pg_loss_per_sample, actor_multiplier, args.actor_compute_loss_floor)

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * (torch.max(v_loss_unclipped, v_loss_clipped) * critic_multiplier).mean()
                else:
                    v_loss = 0.5 * (((newvalue - b_returns[mb_inds]) ** 2) * critic_multiplier).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
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
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if last_stats is not None:
            for group in ("actor", "critic"):
                writer.add_scalar(f"morph/{group}_compute", mean_stat(last_stats, group, "compute"), global_step)
                writer.add_scalar(f"morph/{group}_eff_templates", mean_stat(last_stats, group, "eff_templates"), global_step)
                writer.add_scalar(f"morph/{group}_assign_support", mean_stat(last_stats, group, "assign_support"), global_step)
                writer.add_scalar(f"morph/{group}_attn_support", mean_stat(last_stats, group, "attn_support"), global_step)
                writer.add_scalar(f"morph/{group}_tick_frac", mean_stat(last_stats, group, "tick_frac"), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
