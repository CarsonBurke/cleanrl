# PPO + Morphogenic Symmetry v13  --  WIDTH AS A SAMPLED ACTION, HELD PER-EPISODE.
#
# v12 RESULT. Making per-template width a sampled Beta action DID unstick it (width_mean
# 0.88->0.83, eff_token_size 121->106 by 1.5M -- the first time width ever moved). BUT returns
# collapsed to ~829 @1.5M vs ~2540 for the deterministic baseline: resampling every token's
# width EVERY STEP makes the policy's own forward graph flicker each timestep (violent per-step
# architectural noise), which cripples RL. Motor actions may change per step; the ARCHITECTURE
# must not.
#
# v13 FIX. Sample width ONCE PER EPISODE and hold it fixed until the env resets (resample on
# done). The architecture is stable within an episode -> the policy learns against a consistent
# forward -> returns should recover, while width still explores across episodes and is credited
# by episode return (each step's advantage multiplies the same held-width log-prob, so the width
# policy gradient ~= episode-advantage-sum * d log p(width)). Everything else matches v12.
#
# MOTIVATION (unchanged). In v5..v11 per-template token width was a DETERMINISTIC gate that
# stayed pinned at the ceiling: the task wants full width and the compute tax is a tiny
# counter-gradient with no way to *try* narrower and discover it is fine. A pinned smooth
# parameter cannot explore; a sampled action can.
#
# IDEA. Make width an ACTION. Each of the M templates has a free Beta policy (mean + learned
# concentration = "log-var"), state-INDEPENDENT like PPO's action log-std (width cannot be an
# output of the forward it configures). Every env-step we SAMPLE a width fraction per template,
# feed it into the forward (per-unit channel mask over the FFN update), and treat it exactly
# like the motor action:
#   * its log-prob joins the PPO ratio (so advantage credits widths that produce good returns),
#   * its entropy joins the entropy bonus (exploration over widths),
#   * it is stored in the rollout buffer and replayed at update time.
# The compute cost of width is folded into the PER-STEP REWARD ( r' = r - width_reward_coef *
# active_channel_fraction ). This gives width a real DIRECTIONAL signal: narrowing is rewarded
# unless it hurts return, so the advantage discovers the emergent minimal token width -- the
# honest "compute-in-the-reward" analogue of the loss tax.
#
# The dense forward computes all D channels then masks (per-sample widths differ per env, so no
# ragged matmul). Real FLOP saving via per-template sized matmuls (v11) can be grafted on once
# width is confirmed to move. Attention-density / assignment compute stay on the v9 loss tax;
# only WIDTH moves to the reward+action mechanism (width excluded from the loss tax to avoid
# double counting). Critic uses the deterministic Beta MEAN width (value estimation is not an
# action). HYPOTHESIS: width now contracts to a task-justified value instead of staying pinned.
import os
import math
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter

from cleanrl.morph.compute.ppo_continuous_action_morphcompute_v9 import SAMPLE_EPS, effective_support, make_env, mean_stat
from cleanrl.morph.compute.ppo_continuous_action_morphcompute_v18 import signed_loss_with_safe_compute
from cleanrl.morph.sym.ppo_continuous_action_morphsym_v3 import participation_ratio
from cleanrl.morph.sym.ppo_continuous_action_morphsym_v9 import Args as V9Args, Agent as V9Agent, SymGraph9


@dataclass
class Args(V9Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    width_reward_coef: float = 0.15
    """per-step reward penalty coefficient on active-channel fraction (compute-in-the-reward)"""
    width_mean_init: float = 2.0
    """logit init for the Beta width mean; sigmoid(2)~0.88 -> starts wide, room to contract"""
    width_logconc_init: float = 1.5
    """init for log-concentration; higher -> tighter Beta (less width exploration)"""
    width_ent_coef: float = 0.01
    """dedicated entropy bonus on the width Beta policy (sustains width exploration; kept
    separate from the motor-action entropy since Beta differential entropy is unbounded)"""


class SymGraph12(SymGraph9):
    """v9 graph, but token width arrives as a per-sample per-template fraction `width_frac`
    (B, M) and is applied as a soft per-unit channel prefix-mask on the FFN update. The loss
    compute tax prices ONLY attention density + assignment (width is handled via reward+action).
    """

    def forward(self, x, width_frac):
        B = x.shape[0]
        phi = self.phi()
        h = self.input(x)[:, None, :] + self.pos_embed(phi)[None, :, :]
        tick_gates = torch.sigmoid(self.tick_bias)
        chan = torch.arange(self.D, device=x.device)  # (D,)

        attn_support_sum = x.new_zeros(B)
        eff_size_sum = x.new_zeros(B)
        last_A = None
        for t in range(self.T):
            mixed, support = self.mixer(h, phi)
            h = h + tick_gates[t] * (mixed - h)
            updated, A = self.field(h, phi)  # updated = h + delta, A: (U, M)
            delta = updated - h
            # per-unit width fraction (B, U) = mixture of the unit's assigned template widths
            wu = width_frac @ A.t()  # (B, U) in (0,1)
            active = (self.D * wu).unsqueeze(-1) - chan  # (B, U, D)
            mask = torch.clamp(active, 0.0, 1.0)  # soft channel prefix mask
            h = h + tick_gates[t] * (mask * delta)
            attn_support_sum = attn_support_sum + tick_gates[t] * support.mean(dim=(1, 2))
            eff_size_sum = eff_size_sum + mask.sum(dim=-1).mean(dim=1)  # avg active channels / unit
            last_A = A

        out_units = h[:, self.N :, :]

        eff_templates = participation_ratio(last_A.sum(dim=0))
        assign_support = effective_support(last_A, dim=-1).mean()
        eff_token_size = eff_size_sum / max(self.T, 1)  # (B,) avg active channels per unit
        tick_frac = tick_gates.mean()

        assign_cost = (assign_support / max(self.M, 1)) * tick_frac
        attn_cost = (attn_support_sum / max(self.U, 1)) / max(self.T, 1)
        wsum = self.assign_compute_weight + self.attn_compute_weight
        compute = (
            self.assign_compute_weight * assign_cost.expand(B) + self.attn_compute_weight * attn_cost
        ) / max(wsum, 1e-8)
        compute = compute.clamp(min=0.0)

        stats = {
            "compute": compute,
            "eff_templates": eff_templates.expand(B),
            "assign_support": assign_support.expand(B),
            "attn_support": (attn_support_sum / max(self.T, 1)).clamp(min=0.0),
            "eff_token_size": eff_token_size,
            "tick_frac": tick_frac.expand(B),
        }
        return out_units, stats


class Agent(V9Agent):
    """Adds a free per-template Beta width policy (mean + log-concentration), state-independent
    like the action log-std. Width is sampled, fed into the actor forward, and treated as part
    of the joint action (log-prob + entropy). The critic uses the deterministic mean width."""

    def __init__(self, envs, args=None):
        if args is None:
            args = Args()
        super().__init__(envs, args)
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.actor = SymGraph12(obs_dim, args, n_out=action_dim)
        self.critic = SymGraph12(obs_dim, args, n_out=1)
        self.M = args.num_templates
        self.width_mean_logit = nn.Parameter(torch.full((self.M,), float(args.width_mean_init)))
        self.width_logconc = nn.Parameter(torch.full((self.M,), float(args.width_logconc_init)))

    def width_dist(self):
        mean = torch.sigmoid(self.width_mean_logit)  # (M,) in (0,1)
        conc = torch.nn.functional.softplus(self.width_logconc) + 2.0  # (M,) >= 2
        alpha = mean * conc
        beta = (1.0 - mean) * conc
        return Beta(alpha, beta)

    def get_value(self, x):
        wdist = self.width_dist()
        wmean = wdist.mean.detach().unsqueeze(0).expand(x.shape[0], -1)  # (B, M) deterministic
        value_units, _ = self.critic(x, wmean)
        return self.value_head(value_units.squeeze(1)).squeeze(-1)

    def get_action_and_value(self, x, z=None, width=None):
        B = x.shape[0]
        wdist = self.width_dist()
        if width is None:
            width = wdist.sample((B,)).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)  # (B, M)
        width_logprob = wdist.log_prob(width).sum(-1)  # (B,)

        action_units, actor_stats = self.actor(x, width)  # actor uses SAMPLED width
        wmean = wdist.mean.detach().unsqueeze(0).expand(B, -1)  # detach: value loss must not train width center
        value_units, critic_stats = self.critic(x, wmean)  # critic uses MEAN width
        dist, to_action = self._dist(action_units)
        if z is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        logprob = dist.log_prob(z).sum(1) + width_logprob  # joint action log-prob
        entropy = dist.entropy().sum(1)  # motor-action entropy (width exploration handled separately)
        value = self.value_head(value_units.squeeze(1)).squeeze(-1)
        stats = {"actor": dict(actor_stats), "critic": dict(critic_stats)}
        return action, z, logprob, entropy, value, actor_stats["compute"], critic_stats["compute"], stats, width


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True,
                   config=vars(args), name=run_name, monitor_gym=True, save_code=True)
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

    M = args.num_templates
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    widths = torch.zeros((args.num_steps, args.num_envs, M)).to(device)
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
    # Width is held PER-EPISODE (sampled once, kept fixed until the env resets). This keeps the
    # architecture stable within an episode -- the policy can learn against a consistent forward --
    # while still exploring widths across episodes and being credited by episode return. (v12
    # resampled width every step; that per-step architectural flicker crippled learning.)
    with torch.no_grad():
        current_width = agent.width_dist().sample((args.num_envs,)).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, z, logprob, _, value, _, _, last_stats, width = agent.get_action_and_value(
                    next_obs, width=current_width
                )
                values[step] = value.flatten()
            zs[step] = z
            widths[step] = width
            logprobs[step] = logprob
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            # compute-in-the-reward: penalise active-channel fraction of the (held) width
            reward_t = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
            width_penalty = args.width_reward_coef * (last_stats["actor"]["eff_token_size"] / agent.actor.D)
            rewards[step] = reward_t - width_penalty
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            # resample width for envs that just finished -> fresh architecture for their next episode
            with torch.no_grad():
                if next_done.any():
                    fresh = agent.width_dist().sample((args.num_envs,)).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                    current_width = torch.where(next_done.bool().unsqueeze(1), fresh, current_width)
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
        b_widths = widths.reshape(-1, M)
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
                _, _, newlogprob, entropy, newvalue, actor_compute, critic_compute, last_stats, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_zs[mb_inds], b_widths[mb_inds]
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
                width_entropy = agent.width_dist().entropy().mean()  # sustain width exploration
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    - args.width_ent_coef * width_entropy
                    + v_loss * args.vf_coef
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
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        with torch.no_grad():
            wdist = agent.width_dist()
            writer.add_scalar("morph/width_mean", wdist.mean.mean().item(), global_step)
            writer.add_scalar("morph/width_std", wdist.stddev.mean().item(), global_step)
            writer.add_scalar("morph/width_min", wdist.mean.min().item(), global_step)
            writer.add_scalar("morph/width_entropy", wdist.entropy().mean().item(), global_step)
        if last_stats is not None:
            for group in ("actor", "critic"):
                writer.add_scalar(f"morph/{group}_compute", mean_stat(last_stats, group, "compute"), global_step)
                writer.add_scalar(f"morph/{group}_eff_templates", mean_stat(last_stats, group, "eff_templates"), global_step)
                writer.add_scalar(f"morph/{group}_assign_support", mean_stat(last_stats, group, "assign_support"), global_step)
                writer.add_scalar(f"morph/{group}_attn_support", mean_stat(last_stats, group, "attn_support"), global_step)
                writer.add_scalar(f"morph/{group}_eff_token_size", mean_stat(last_stats, group, "eff_token_size"), global_step)
                writer.add_scalar(f"morph/{group}_tick_frac", mean_stat(last_stats, group, "tick_frac"), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
