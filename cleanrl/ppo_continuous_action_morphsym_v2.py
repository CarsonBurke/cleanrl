# PPO + Morphogenic Symmetry v2.
#
# METHOD. v1 made spatial weight-tying learnable but two things blunted it: (1) the
# unit->template assignment depended only on a static coordinate, so tying was a
# fixed weight-init rather than genuine conditional computation; (2) it dropped the
# lineage's biggest score driver -- readout tokens that directly shape the policy
# and value. v2 fixes both while keeping the learned-symmetry thesis:
#   - CONTENT-DEPENDENT TYING. Assignment A = entmax(assign(coord) + gamma*assign_state(g))
#     where g is a per-sample global summary of the unit field. gamma starts small
#     so training begins at v1's static structural partition, then units may switch
#     templates by situation (a mixture-of-experts over shared weight-sets). The
#     structural symmetry interpretation survives via the persistent coord term.
#   - PAID READOUT TOKENS. A few learned tokens attend (entmax) over the unit field
#     and add gated, compute-priced residuals directly to the Beta alpha/beta logits
#     (actor) and the value (critic), as in morphcompute v23.
# Everything remains emergent and compute-bounded: distinct-template count, unit
# connectivity, and readout support are all charged to compute.
#
# HYPOTHESIS. Conditional weight-tying (learned symmetry that can also adapt per
# state) plus a direct readout->action path recovers and exceeds the lineage's
# performance while making the symmetry-discovery mechanism actually load-bearing.
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

from cleanrl.ppo_continuous_action_morphcompute_v9 import (
    ReLUSquared,
    SAMPLE_EPS,
    effective_support,
    layer_init,
    make_env,
    mean_stat,
)
from cleanrl.ppo_continuous_action_morphcompute_v18 import entmax15, signed_loss_with_safe_compute
from cleanrl.ppo_continuous_action_morphsym_v1 import Args as V1Args, UnitMixer, participation_ratio


@dataclass
class Args(V1Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    assign_state_scale: float = 0.25
    """initial gamma on the state-dependent term of the unit->template assignment"""
    num_readouts: int = 4
    """learned readout tokens that attend over the unit field and shape policy/value"""
    readout_gate_bias: float = -1.5
    """initial logit for the paid readout residual on Beta logits / value"""
    readout_gate_max: float = 0.75
    """maximum readout residual scale"""
    readout_compute_weight: float = 0.5
    """weight on readout-support compute cost"""


class ContentTiedField(nn.Module):
    """v1's TiedField with content-dependent assignment: units may switch templates by state."""

    def __init__(self, N, D, H, M, coord_dim, assign_state_scale):
        super().__init__()
        self.N, self.D, self.H, self.M = N, D, H, M
        self.assign = layer_init(nn.Linear(coord_dim, M), std=0.5)
        self.assign_state = layer_init(nn.Linear(D, M), std=0.5)
        self.assign_state_scale = assign_state_scale
        self.W1 = nn.Parameter(torch.empty(M, D, H))
        self.b1 = nn.Parameter(torch.zeros(M, H))
        self.W2 = nn.Parameter(torch.empty(M, H, D))
        self.b2 = nn.Parameter(torch.zeros(M, D))
        for m in range(M):
            nn.init.orthogonal_(self.W1[m], gain=np.sqrt(2.0))
            nn.init.orthogonal_(self.W2[m], gain=0.5)
        self.norm = nn.LayerNorm(D)

    def assignment(self, coord, summary):
        # coord term: (N, M) structural. state term: (B, M) global -> broadcast over units.
        logits = self.assign(coord)[None, :, :] + self.assign_state_scale * self.assign_state(summary)[:, None, :]
        return entmax15(logits, dim=-1)  # (B, N, M)

    def forward(self, h, coord):
        x = self.norm(h)
        summary = x.mean(dim=1)  # (B, D) global field summary
        A = self.assignment(coord, summary)  # (B, N, M)
        hidden = ReLUSquared()(torch.einsum("bnd,mdh->bnmh", x, self.W1) + self.b1)
        out = torch.einsum("bnmh,mhd->bnmd", hidden, self.W2) + self.b2
        delta = torch.einsum("bnm,bnmd->bnd", A, out)
        return h + delta, A


class ReadoutTokens(nn.Module):
    """Learned tokens that attend (entmax) over the unit field and emit content-dependent features
    that directly shape the policy/value, plus a compute charge for their routing support."""

    def __init__(self, num_tokens, D, attn_dim):
        super().__init__()
        self.num_tokens = num_tokens
        self.token = nn.Parameter(torch.randn(num_tokens, D) * 0.02)
        self.q = layer_init(nn.Linear(D, attn_dim), std=0.5)
        self.k = layer_init(nn.Linear(D, attn_dim), std=0.5)
        self.v = layer_init(nn.Linear(D, D), std=0.5)
        self.scale = np.sqrt(attn_dim)
        self.norm = nn.LayerNorm(D)
        self.out = nn.Sequential(layer_init(nn.Linear(D, D)), ReLUSquared())

    def forward(self, h):
        B, N = h.shape[0], h.shape[1]
        hn = self.norm(h)
        q = self.q(self.token)[None, :, :].expand(B, -1, -1)  # (B, R, A)
        logits = torch.einsum("bra,bna->brn", q, self.k(hn)) / self.scale  # (B, R, N)
        route = entmax15(logits, dim=-1)
        feats = torch.bmm(route, self.v(hn))  # (B, R, D)
        feats = feats + self.token[None, :, :]
        support = effective_support(route, dim=-1)  # (B, R)
        compute = support.mean(dim=1) * self.num_tokens / max(N, 1)  # (B,)
        return self.out(feats), support, compute


class SymField2(nn.Module):
    def __init__(self, obs_dim, args):
        super().__init__()
        self.N = args.num_units
        self.D = args.cell_dim
        self.T = args.sym_ticks
        self.M = args.num_templates
        self.template_compute_weight = args.template_compute_weight
        self.mixer_compute_weight = args.mixer_compute_weight
        self.readout_compute_weight = args.readout_compute_weight

        self.coord = nn.Parameter(torch.randn(self.N, args.coord_dim))
        self.input = layer_init(nn.Linear(obs_dim, self.D))
        self.coord_embed = layer_init(nn.Linear(args.coord_dim, self.D), std=0.5)
        self.mixer = UnitMixer(self.N, self.D, args.coord_dim, args.attn_dim, args.init_null_bias)
        self.field = ContentTiedField(
            self.N, self.D, args.hidden_dim, self.M, args.coord_dim, args.assign_state_scale
        )
        self.tick_bias = nn.Parameter(torch.full((self.T,), float(args.init_tick_bias)))
        self.read_query = layer_init(nn.Linear(args.coord_dim, 1), std=0.5)
        self.readout = nn.Sequential(layer_init(nn.Linear(self.D, self.D)), ReLUSquared())
        self.tokens = ReadoutTokens(args.num_readouts, self.D, args.attn_dim)

    def forward(self, x):
        B = x.shape[0]
        h = self.input(x)[:, None, :] + self.coord_embed(self.coord)[None, :, :]
        tick_gates = torch.sigmoid(self.tick_bias)

        mixer_support_sum = x.new_zeros(B)
        last_A = None
        for t in range(self.T):
            mixed, support = self.mixer(h, self.coord)
            updated, A = self.field(mixed, self.coord)
            h = h + tick_gates[t] * (updated - h)
            mixer_support_sum = mixer_support_sum + tick_gates[t] * support.mean(dim=1)
            last_A = A

        read_w = torch.softmax(self.read_query(self.coord).squeeze(-1), dim=0)  # (N,)
        pooled = torch.einsum("n,bnd->bd", read_w, h)
        out = self.readout(pooled)

        token_feats, token_support, token_compute = self.tokens(h)

        template_usage = last_A.sum(dim=1)  # (B, M)
        eff_templates = participation_ratio_batched(template_usage)  # (B,)
        tick_frac = tick_gates.mean()
        template_cost = (eff_templates / self.M) * tick_frac
        mixer_cost = (mixer_support_sum / max(self.N, 1)) / max(self.T, 1)
        wsum = self.template_compute_weight + self.mixer_compute_weight + self.readout_compute_weight
        compute = (
            self.template_compute_weight * template_cost
            + self.mixer_compute_weight * mixer_cost
            + self.readout_compute_weight * token_compute
        ) / wsum
        compute = compute.clamp(min=0.0)

        assign_support = effective_support(last_A, dim=-1).mean(dim=1)  # (B,) avg templates per unit
        stats = {
            "compute": compute,
            "eff_templates": eff_templates,
            "assign_support": assign_support,
            "mixer_support": (mixer_support_sum / max(self.T, 1)).clamp(min=0.0),
            "readout_support": token_support.mean(dim=1),
            "tick_frac": tick_frac.expand(B),
        }
        return out, token_feats, stats


def participation_ratio_batched(usage, eps=1e-6):
    """Per-row effective count: (sum)^2 / sum(sq) over last dim. usage (B, M) -> (B,)."""
    return usage.sum(dim=-1).pow(2) / usage.pow(2).sum(dim=-1).clamp_min(eps)


class Agent(nn.Module):
    def __init__(self, envs, args=None):
        super().__init__()
        if args is None:
            args = Args()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.actor = SymField2(obs_dim, args)
        self.critic = SymField2(obs_dim, args)
        self.actor_alpha = layer_init(nn.Linear(args.cell_dim, action_dim), std=0.01)
        self.actor_beta = layer_init(nn.Linear(args.cell_dim, action_dim), std=0.01)
        self.critic_value = layer_init(nn.Linear(args.cell_dim, 1), std=1.0)
        # readout token heads: pool R tokens then map to per-action / value residuals
        self.token_alpha = layer_init(nn.Linear(args.cell_dim, action_dim), std=0.01)
        self.token_beta = layer_init(nn.Linear(args.cell_dim, action_dim), std=0.01)
        self.token_value = layer_init(nn.Linear(args.cell_dim, 1), std=0.01)
        self.actor_readout_gate = nn.Parameter(torch.tensor(float(args.readout_gate_bias)))
        self.critic_readout_gate = nn.Parameter(torch.tensor(float(args.readout_gate_bias)))
        self.readout_gate_max = args.readout_gate_max
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

    def compute_multiplier(self, compute, args):
        return args.min_compute_multiplier + args.compute_coef * compute

    def _readout_gate(self, gate):
        return self.readout_gate_max * torch.sigmoid(gate)

    def get_value(self, x):
        critic_features, critic_tokens, _ = self.critic(x)
        token_pool = critic_tokens.mean(dim=1)  # (B, D)
        return (
            self.critic_value(critic_features).squeeze(-1)
            + self._readout_gate(self.critic_readout_gate) * self.token_value(token_pool).squeeze(-1)
        )

    def _dist(self, actor_features, actor_tokens):
        token_pool = actor_tokens.mean(dim=1)  # (B, D)
        gate = self._readout_gate(self.actor_readout_gate)
        alpha_logits = self.actor_alpha(actor_features) + gate * self.token_alpha(token_pool)
        beta_logits = self.actor_beta(actor_features) + gate * self.token_beta(token_pool)
        alpha = 1.0 + F.softplus(alpha_logits)
        beta = 1.0 + F.softplus(beta_logits)
        dist = Beta(alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        return dist, to_action

    def get_action_and_value(self, x, z=None):
        actor_features, actor_tokens, actor_stats = self.actor(x)
        critic_features, critic_tokens, critic_stats = self.critic(x)
        dist, to_action = self._dist(actor_features, actor_tokens)
        if z is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        logprob = dist.log_prob(z).sum(1)
        entropy = dist.entropy().sum(1)
        token_pool = critic_tokens.mean(dim=1)
        value = self.critic_value(critic_features).squeeze(-1) + self._readout_gate(
            self.critic_readout_gate
        ) * self.token_value(token_pool).squeeze(-1)
        actor_stats = dict(actor_stats)
        critic_stats = dict(critic_stats)
        actor_stats["readout_gate"] = self._readout_gate(self.actor_readout_gate).expand_as(logprob)
        critic_stats["readout_gate"] = self._readout_gate(self.critic_readout_gate).expand_as(logprob)
        stats = {"actor": actor_stats, "critic": critic_stats}
        return action, z, logprob, entropy, value, actor_stats["compute"], critic_stats["compute"], stats


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
                writer.add_scalar(f"morph/{group}_mixer_support", mean_stat(last_stats, group, "mixer_support"), global_step)
                writer.add_scalar(f"morph/{group}_readout_support", mean_stat(last_stats, group, "readout_support"), global_step)
                writer.add_scalar(f"morph/{group}_readout_gate", mean_stat(last_stats, group, "readout_gate"), global_step)
                writer.add_scalar(f"morph/{group}_tick_frac", mean_stat(last_stats, group, "tick_frac"), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
