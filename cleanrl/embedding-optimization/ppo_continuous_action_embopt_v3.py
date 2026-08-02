# Embedding Optimization v3 (embopt_v3) — family: cleanrl/embedding-optimization/ (see FAMILY.md)
#
# v1 -> v3 (v2's replay-argmax proposer was off-family — retrieval caps goals at achieved
# states and abandons the analytic gradient; retired without data): the proposer is restored
# to the family mechanism — gradient ascent on the frozen score surface
# W_hat(g) + alpha_reach * V(z, g), the same implicit-target regression the policy uses.
# v1's measured failure was NOT this mechanism but the added 0.5*||g||^2 prior: mode-seeking
# is wrong in high dim (the N(0,I_64) mass lives on the shell ||g|| ~ sqrt(dz) = 8, the mode
# at the origin decodes to the mean replay state = "stand still"). v3's fix is structural,
# not a penalty: proposals are reparameterized onto the typical shell,
# g = sqrt(dz) * unit(MLP(z)), so ascent moves along the shell where W_hat/V have data —
# no norm gradient fighting the score, no origin attractor, in-distribution by construction,
# and goals can still exceed anything in replay (open-ended, unlike v2's retrieval).
# Keeps v2's goal-reaching telemetry: rollout_goal_v / rollout_goal_dv, z_sqnorm.
#
# Method: the policy is deterministic and its ENTIRE objective is one differentiable scalar:
#     L_pi = -[ r_hat(z, pi(z,g)) + gamma_g * V(f(z, pi(z,g)), g) ]
# Gradients flow backward through the frozen critic V and frozen latent dynamics f into the
# action and into the policy parameters. No sampling, no likelihood ratios, no argmax, no
# contrastive terms. The world model and critic jointly define the target implicitly; the loss
# surface the policy descends IS the value landscape composed with dynamics (teacher forcing
# without materializing the teacher's action). The pan-goal-solver Phase-A objective
# minimize d(f(z,pi),g) is the lambda=1 special case: hindsight TD(1) gives V = gamma_g^k, so
# -log_gamma V ~ d and value ascent is distance descent.
#
# Components:
#   - Encoder E + dynamics f: LeJEPA-style, attached online target (no EMA, no stop-grad
#     asymmetry), SIGReg isotropy regularizer prevents collapse. Reward heads read detached z.
#   - V(z,g): hindsight-relabeled TD(lambda) along experienced paths (lambda returns, no GAE).
#     Sparse-at-goal reward => lambda=1 is pure regression V <- gamma_g^k, lambda<1 sharpens.
#   - Goal proposer P(z): gradient ascent on frozen W_hat(g) + alpha*V(z,g) (reward-rate of the
#     goal regime + reachability), with a small ||g||^2 prior matching the SIGReg'd marginal.
#   - Policy pi(z,g): tanh head, trained by L_pi on a mix of hindsight and proposed goals;
#     acts with additive Gaussian noise on proposed goals held for goal_refresh steps.
# Freezing is implemented as backward(inputs=<phase params>) + per-phase optimizers (no
# requires_grad flips; compile/cudagraph-safe). Family lesson made mandatory: goal-channel
# aliveness is measured every iteration (diagnostics/goal_sensitivity_action_mse).
#
# Hypothesis: hindsight BC failed for lack of an improvement operator; dV/da is an analytic
# improvement operator that exists as soon as V has any g-dependence, which hindsight TD
# guarantees by construction. One-step model traversal (SVG(1)-style) bounds compounding
# model error; long-horizon credit comes from V, not unrolling.
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


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """learning rate for all four optimizers (world model, critic, proposer, policy)"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per rollout segment"""
    anneal_lr: bool = True
    """toggle learning rate annealing for all optimizers"""
    gamma_goal: float = 0.98
    """goal discount defining the value metric V = gamma_goal^d"""
    td_lambda: float = 0.95
    """lambda for the hindsight lambda-return critic (1.0 = pure regression to gamma_goal^k)"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping (per phase)"""

    latent_dim: int = 64
    """embedding dimension"""
    hidden_dim: int = 256
    """MLP hidden width"""
    sigreg_coef: float = 0.09
    """weight of the SIGReg isotropy loss on encoder embeddings (canonical family weight)"""
    sigreg_proj: int = 256
    """number of random projections for SIGReg"""
    sigreg_knots: int = 17
    """quadrature knots for SIGReg"""

    replay_iters: int = 64
    """ring buffer capacity in rollout segments (transitions = replay_iters*num_steps*num_envs)"""
    minibatch_size: int = 2048
    """minibatch size for all update phases"""
    wm_updates: int = 16
    """world-model minibatch updates per iteration"""
    critic_updates: int = 16
    """critic minibatch updates per iteration"""
    proposer_updates: int = 8
    """proposer minibatch updates per iteration"""
    policy_updates: int = 16
    """policy minibatch updates per iteration"""
    critic_segments: int = 4
    """segments (fresh + sampled replay) used to build hindsight pairs each iteration"""
    hindsight_goals: int = 8
    """hindsight goals sampled per env per segment"""
    self_goal_coef: float = 0.1
    """weight of the V(g,g)=0.99 anchor loss"""
    neg_goal_coef: float = 0.25
    """weight of cross-trajectory negative pairs regressed to gamma_goal^num_steps"""
    rate_window: int = 16
    """forward window (steps) for the reward-rate target W_hat"""

    reward_coef: float = 1.0
    """weight of the r_hat term in the policy loss"""
    value_coef: float = 1.0
    """weight of the gamma_g*V term in the policy loss"""
    alpha_reach: float = 1.0
    """weight of the V(z,g) reachability term in the proposer score"""
    policy_goal_mix: float = 0.5
    """fraction of the policy batch trained on proposed (vs hindsight) goals"""
    expl_noise: float = 0.2
    """std of Gaussian exploration noise added to the deterministic action"""
    goal_refresh: int = 16
    """env steps a proposed goal is held during rollout before re-proposing"""
    warmup_steps: int = 10000
    """global env steps of uniform-random actions before the policy acts"""

    compile: bool = True
    """torch.compile the act/loss functions (CUDA graphs with reduce-overhead)"""
    compile_mode: str = "reduce-overhead"
    """torch.compile mode"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    # raw rewards on purpose: replayed reward targets are rescaled by one replay-wide
    # running scale at sample time, so all regression targets share a consistent scale
    # (NormalizeReward's per-collection-time scale would go stale in the ring buffer)
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(in_dim, hidden, out_dim, out_std=1.0):
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, hidden)),
        nn.SiLU(),
        layer_init(nn.Linear(hidden, hidden)),
        nn.SiLU(),
        layer_init(nn.Linear(hidden, out_dim), std=out_std),
    )


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regularizer (ECF test), as in the LeJEPA lineage."""

    def __init__(self, knots=17, num_proj=256):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, x):
        # x: (N, D) embedding batch
        A = torch.randn(x.size(-1), self.num_proj, device=x.device, dtype=x.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)
        x_t = (x @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()
        statistic = (err @ self.weights) * x.size(0)
        return statistic.mean()


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        do = int(np.array(envs.single_observation_space.shape).prod())
        da = int(np.prod(envs.single_action_space.shape))
        dz, dh = args.latent_dim, args.hidden_dim
        self.encoder = mlp(do, dh, dz)
        self.dyn = mlp(dz + da, dh, dz, out_std=0.01)  # residual: f(z,a) = z + dyn([z,a])
        self.reward_head = mlp(dz + da, dh, 1)
        self.rate_head = mlp(dz, dh, 1)
        self.critic = mlp(3 * dz, dh, 1)
        # out_std 1.0: only the direction matters (output is shell-projected), so start
        # with well-defined, state-dependent directions rather than near-zero vectors
        self.proposer = mlp(dz, dh, dz, out_std=1.0)
        self.policy = mlp(3 * dz, dh, da, out_std=0.01)
        self.dz = dz

    def encode(self, obs):
        return self.encoder(obs)

    def forward_dyn(self, z, a):
        return z + self.dyn(torch.cat([z, a], -1))

    def value(self, z, g):
        return torch.sigmoid(self.critic(torch.cat([z, g, g - z], -1))).squeeze(-1)

    def act(self, z, g):
        return torch.tanh(self.policy(torch.cat([z, g, g - z], -1)))

    def propose(self, z):
        # typical shell of the SIGReg'd N(0,I) marginal: ||g|| = sqrt(dz)
        raw = self.proposer(z)
        return (self.dz**0.5) * raw / raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def rhat(self, z, a):
        return self.reward_head(torch.cat([z, a], -1)).squeeze(-1)

    def what(self, z):
        return self.rate_head(z).squeeze(-1)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.rate_window <= args.num_steps, "rate_window must not exceed num_steps"
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
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    da = int(np.prod(envs.single_action_space.shape))
    do = int(np.array(envs.single_observation_space.shape).prod())
    dz = args.latent_dim
    T, E = args.num_steps, args.num_envs

    agent = Agent(envs, args).to(device)
    sigreg = SIGReg(knots=args.sigreg_knots, num_proj=args.sigreg_proj).to(device)

    wm_params = (
        list(agent.encoder.parameters())
        + list(agent.dyn.parameters())
        + list(agent.reward_head.parameters())
        + list(agent.rate_head.parameters())
    )
    critic_params = list(agent.critic.parameters())
    proposer_params = list(agent.proposer.parameters())
    policy_params = list(agent.policy.parameters())
    wm_opt = optim.Adam(wm_params, lr=args.learning_rate, eps=1e-5)
    critic_opt = optim.Adam(critic_params, lr=args.learning_rate, eps=1e-5)
    proposer_opt = optim.Adam(proposer_params, lr=args.learning_rate, eps=1e-5)
    policy_opt = optim.Adam(policy_params, lr=args.learning_rate, eps=1e-5)

    # ---- loss / act functions (compiled) -------------------------------------------------
    def rollout_forward(obs, cur_goal, refresh):
        z = agent.encode(obs)
        goal = torch.where(refresh, agent.propose(z), cur_goal)
        action = agent.act(z, goal)
        return z, goal, action

    def wm_loss_fn(o, a, o2, r, rate, pair_valid):
        z = agent.encode(o)
        z2 = agent.encode(o2)
        pred = agent.forward_dyn(z, a)
        denom = pair_valid.sum().clamp_min(1.0)
        pred_loss = (pair_valid * (pred - z2).square().mean(-1)).sum() / denom
        sig_loss = sigreg(z)
        zd = z.detach()
        r_loss = (agent.rhat(zd, a) - r).square().mean()
        w_loss = (agent.what(zd) - rate).square().mean()
        loss = pred_loss + args.sigreg_coef * sig_loss + r_loss + w_loss
        return loss, pred_loss.detach(), sig_loss.detach(), r_loss.detach(), w_loss.detach()

    def critic_loss_fn(z, g, target):
        v = agent.value(z, g)
        main = (v - target).square().mean()
        # 0.99 not 1.0: sigmoid can reach it, so the near-goal logit doesn't drift into
        # saturation and kill dV/dz' exactly where the policy needs gradient
        anchor = (agent.value(g, g) - 0.99).square().mean()
        # cross-trajectory negatives (batch is randomly sampled, so roll pairs each z with
        # an unrelated goal): anchors V's extrapolation off the achieved set toward "far",
        # making the proposer's reachability term and the policy's value ascent non-vacuous
        neg_target = args.gamma_goal**args.num_steps
        neg = (agent.value(z, torch.roll(g, 1, 0)) - neg_target).square().mean()
        return (
            main + args.self_goal_coef * anchor + args.neg_goal_coef * neg,
            main.detach(),
            v.mean().detach(),
        )

    def proposer_loss_fn(z):
        g = agent.propose(z)
        w = agent.what(g)
        v = agent.value(z, g)
        score = w + args.alpha_reach * v
        return -score.mean(), w.mean().detach(), v.mean().detach()

    def policy_loss_fn(z, g):
        a = agent.act(z, g)
        r = agent.rhat(z, a)
        v2 = agent.value(agent.forward_dyn(z, a), g)
        obj = args.reward_coef * r + args.value_coef * args.gamma_goal * v2
        return -obj.mean(), r.mean().detach(), v2.mean().detach()

    if args.compile:
        rollout_forward = torch.compile(rollout_forward, mode=args.compile_mode, dynamic=False)
        wm_loss_fn = torch.compile(wm_loss_fn, mode=args.compile_mode, dynamic=False)
        critic_loss_fn = torch.compile(critic_loss_fn, mode=args.compile_mode, dynamic=False)
        proposer_loss_fn = torch.compile(proposer_loss_fn, mode=args.compile_mode, dynamic=False)
        policy_loss_fn = torch.compile(policy_loss_fn, mode=args.compile_mode, dynamic=False)
        print(f"[embopt_v1] torch.compile(mode={args.compile_mode!r}, dynamic=False)")

    def mark_step():
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()

    # ---- segment replay ring buffer ------------------------------------------------------
    R = args.replay_iters
    buf_obs = torch.zeros((R, T + 1, E, do), device=device)
    buf_act = torch.zeros((R, T, E, da), device=device)
    buf_rew = torch.zeros((R, T, E), device=device)
    buf_done = torch.zeros((R, T + 1, E), device=device)  # done[t]=1 => obs[t] is a reset obs
    buf_rate = torch.zeros((R, T, E), device=device)
    buf_filled, buf_ptr = 0, 0

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(E).to(device)
    cur_goal = torch.zeros((E, dz), device=device)
    roll_goals = torch.zeros((T, E, dz), device=device)  # commanded goals of the fresh segment
    mb = args.minibatch_size

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            for opt in (wm_opt, critic_opt, proposer_opt, policy_opt):
                opt.param_groups[0]["lr"] = frac * args.learning_rate

        # ---- rollout --------------------------------------------------------------------
        s = buf_ptr
        for step in range(T):
            buf_obs[s, step] = next_obs
            buf_done[s, step] = next_done
            refresh = next_done.bool() | ((step % args.goal_refresh) == 0)
            with torch.no_grad():
                mark_step()
                _, goal, action = rollout_forward(next_obs, cur_goal, refresh.unsqueeze(-1))
                cur_goal = goal.clone()
                roll_goals[step] = cur_goal
                action = action.clone()
            if global_step < args.warmup_steps:
                action = torch.empty((E, da), device=device).uniform_(-1.0, 1.0)
            else:
                action = (action + args.expl_noise * torch.randn_like(action)).clamp(-1.0, 1.0)
            buf_act[s, step] = action
            global_step += E

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            buf_rew[s, step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(next_done_np.astype(np.float32)).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        buf_obs[s, T] = next_obs
        buf_done[s, T] = next_done

        # forward reward-rate target for W_hat (windowed mean within episode)
        with torch.no_grad():
            acc = torch.zeros((T, E), device=device)
            cnt = torch.zeros((T, E), device=device)
            alive = torch.ones((T, E), device=device)
            for k in range(args.rate_window):
                idx_hi = T - k
                # reward at t+k contributes to rate[t] if no reset strictly inside (t, t+k]
                if k > 0:
                    alive[: T - k] = alive[: T - k] * (1.0 - buf_done[s, k : T, :])
                acc[:idx_hi] += alive[:idx_hi] * buf_rew[s, k:, :][: idx_hi]
                cnt[:idx_hi] += alive[:idx_hi]
            buf_rate[s] = acc / cnt.clamp_min(1.0)

        buf_filled = min(buf_filled + 1, R)
        fresh_slot = buf_ptr
        buf_ptr = (buf_ptr + 1) % R
        F = buf_filled

        # ---- world model updates ---------------------------------------------------------
        # flat sampler over all stored transitions; pair validity masks reset boundaries
        pair_valid_all = (buf_done[:F, 1 : T + 1] == 0).float()  # (F,T,E)
        n_flat = F * T * E
        # one replay-wide reward scale so every regression target this iteration is
        # consistently scaled (rewards are stored raw; see make_env)
        r_scale = buf_rew[:F].std().clamp_min(1e-2)
        wm_stats = []
        for _ in range(args.wm_updates):
            flat = torch.randint(0, n_flat, (mb,), device=device)
            f_i = flat // (T * E)
            t_i = (flat // E) % T
            e_i = flat % E
            o = buf_obs[f_i, t_i, e_i]
            o2 = buf_obs[f_i, t_i + 1, e_i]
            a = buf_act[f_i, t_i, e_i]
            r = buf_rew[f_i, t_i, e_i] / r_scale
            rate = buf_rate[f_i, t_i, e_i] / r_scale
            pv = pair_valid_all[f_i, t_i, e_i]
            mark_step()
            loss, pl, sl, rl, wl = wm_loss_fn(o, a, o2, r, rate, pv)
            wm_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(wm_params, args.max_grad_norm)
            wm_opt.step()
            wm_stats.append((pl.item(), sl.item(), rl.item(), wl.item()))

        # ---- hindsight lambda-return targets (fresh + sampled replay segments) -----------
        S = min(args.critic_segments, F)
        seg = torch.randint(0, F, (S,), device=device)
        seg[0] = fresh_slot
        M = args.hindsight_goals
        with torch.no_grad():
            seg_obs = buf_obs[seg]  # (S,T+1,E,do)
            Z = agent.encode(seg_obs.reshape(-1, do)).reshape(S, T + 1, E, dz)
            epid = buf_done[seg].cumsum(dim=1)  # (S,T+1,E)
            j = torch.randint(1, T + 1, (S, E, M), device=device)  # goal arrival state index
            sidx = torch.arange(S, device=device)[:, None, None]
            eidx = torch.arange(E, device=device)[None, :, None]
            goals = Z[sidx, j, eidx]  # (S,E,M,dz)
            epid_j = epid.permute(0, 2, 1)[sidx, eidx, j]  # (S,E,M)
            # V(z_t, g) for all t
            z_all = Z[:, :, :, None, :].expand(S, T + 1, E, M, dz).reshape(-1, dz)
            g_all = goals[:, None].expand(S, T + 1, E, M, dz).reshape(-1, dz)
            Vs = agent.value(z_all, g_all).reshape(S, T + 1, E, M)
            targets = torch.zeros((S, T, E, M), device=device)
            Gacc = torch.zeros((S, E, M), device=device)
            ones = torch.ones((S, E, M), device=device)
            lam = args.td_lambda
            for t in range(T - 1, -1, -1):
                at_goal = j == (t + 1)
                Vnext = torch.where(at_goal, ones, Vs[:, t + 1])
                Gnext = torch.where(at_goal, ones, Gacc)
                Gacc = args.gamma_goal * ((1.0 - lam) * Vnext + lam * Gnext)
                targets[:, t] = Gacc
            t_range = torch.arange(T, device=device)[None, :, None, None]
            epid_t = epid[:, :T, :, None]  # (S,T,E,1)
            valid = (t_range < j[:, None]) & (epid_t == epid_j[:, None])  # (S,T,E,M)
            valid_idx = valid.reshape(-1).nonzero().squeeze(1)
            # statistically unreachable (S*E*M goals, resets ~1/1000 steps), but fail loudly
            assert valid_idx.numel() > 0, "no valid hindsight pairs this iteration"
            z_pairs = Z[:, :T]  # (S,T,E,dz)

        def sample_pairs(n):
            pick = valid_idx[torch.randint(0, valid_idx.numel(), (n,), device=device)]
            ps = pick // (T * E * M)
            pt = (pick // (E * M)) % T
            pe = (pick // M) % E
            pm = pick % M
            return z_pairs[ps, pt, pe], goals[ps, pe, pm], targets[ps, pt, pe, pm]

        # ---- critic updates --------------------------------------------------------------
        critic_stats = []
        for _ in range(args.critic_updates):
            zb, gb, tb = sample_pairs(mb)
            mark_step()
            loss, main, vmean = critic_loss_fn(zb, gb, tb)
            critic_opt.zero_grad(set_to_none=True)
            loss.backward(inputs=critic_params)
            nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
            critic_opt.step()
            critic_stats.append((main.item(), vmean.item()))

        # ---- proposer updates: ascend the frozen score surface along the shell -----------
        prop_stats = []
        for _ in range(args.proposer_updates):
            flat = torch.randint(0, n_flat, (mb,), device=device)
            with torch.no_grad():
                zb = agent.encode(buf_obs[flat // (T * E), (flat // E) % T, flat % E])
            mark_step()
            loss, wmean, vmean = proposer_loss_fn(zb)
            proposer_opt.zero_grad(set_to_none=True)
            loss.backward(inputs=proposer_params)
            nn.utils.clip_grad_norm_(proposer_params, args.max_grad_norm)
            proposer_opt.step()
            prop_stats.append((wmean.item(), vmean.item()))

        # ---- policy updates (frozen f, r_hat, V) -----------------------------------------
        n_prop = int(mb * args.policy_goal_mix)
        pol_stats = []
        for _ in range(args.policy_updates):
            zh, gh, _ = sample_pairs(mb - n_prop)
            flat = torch.randint(0, n_flat, (n_prop,), device=device)
            with torch.no_grad():
                zp = agent.encode(buf_obs[flat // (T * E), (flat // E) % T, flat % E])
                gp = agent.propose(zp)
            zb = torch.cat([zh, zp], 0)
            gb = torch.cat([gh, gp], 0)
            mark_step()
            loss, rmean, vmean = policy_loss_fn(zb, gb)
            policy_opt.zero_grad(set_to_none=True)
            loss.backward(inputs=policy_params)
            nn.utils.clip_grad_norm_(policy_params, args.max_grad_norm)
            policy_opt.step()
            pol_stats.append((loss.item(), rmean.item(), vmean.item()))

        # ---- diagnostics: is the goal channel alive? (family-mandated) -------------------
        with torch.no_grad():
            # goal-reaching telemetry on the fresh segment: is the policy climbing V toward
            # its commanded goal? (Z[0] is the fresh segment; seg[0] = fresh_slot)
            Vg = agent.value(Z[0, :T].reshape(-1, dz), roll_goals.reshape(-1, dz)).reshape(T, E)
            same_goal = (roll_goals[1:] == roll_goals[:-1]).all(-1) & (buf_done[fresh_slot, 1:T] == 0)
            samef = same_goal.float()
            rollout_goal_v = Vg.mean().item()
            rollout_goal_dv = (((Vg[1:] - Vg[:-1]) * samef).sum() / samef.sum().clamp_min(1.0)).item()
            z_sqnorm = Z[0, :T].square().sum(-1).mean().item()
            zb, gb, _ = sample_pairs(mb)
            a_matched = agent.act(zb, gb)
            a_shuffled = agent.act(zb, gb[torch.randperm(mb, device=device)])
            goal_sens = (a_matched - a_shuffled).square().mean().item()
            # commanded goal vs null goal (g = current z): the diagnostic that failed first
            # in every dead pan-goal-solver version — must lift off ~0
            a_null = agent.act(zb, zb)
            goal_removal = (a_matched - a_null).square().mean().item()
            # proposer collapse watch: output diversity across states, and goal
            # sensitivity measured on PROPOSED goals (hindsight goals are diverse by
            # construction, so the metrics above cannot see a constant proposer)
            gp_diag = agent.propose(zb)
            proposer_out_std = gp_diag.std(0).mean().item()
            a_prop = agent.act(zb, gp_diag)
            a_prop_sh = agent.act(zb, gp_diag[torch.randperm(mb, device=device)])
            proposed_goal_sens = (a_prop - a_prop_sh).square().mean().item()
        # measured (not assumed) balance of the two policy-loss terms at the action
        ad = agent.act(zb, gb).detach()
        a1 = ad.clone().requires_grad_(True)
        grad_r = torch.autograd.grad(agent.rhat(zb, a1).sum(), a1)[0]
        a2 = ad.clone().requires_grad_(True)
        grad_v = torch.autograd.grad(agent.value(agent.forward_dyn(zb, a2), gb).sum(), a2)[0]
        action_grad_rhat = (args.reward_coef * grad_r).norm(dim=-1).mean().item()
        action_grad_value = (args.value_coef * args.gamma_goal * grad_v).norm(dim=-1).mean().item()

        wm_m = np.mean(wm_stats, axis=0)
        cr_m = np.mean(critic_stats, axis=0)
        pr_m = np.mean(prop_stats, axis=0)
        po_m = np.mean(pol_stats, axis=0)
        writer.add_scalar("charts/learning_rate", wm_opt.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/wm_pred", wm_m[0], global_step)
        writer.add_scalar("losses/wm_sigreg", wm_m[1], global_step)
        writer.add_scalar("losses/reward_mse", wm_m[2], global_step)
        writer.add_scalar("losses/rate_mse", wm_m[3], global_step)
        writer.add_scalar("losses/critic", cr_m[0], global_step)
        writer.add_scalar("losses/policy", po_m[0], global_step)
        writer.add_scalar("diagnostics/critic_v_mean", cr_m[1], global_step)
        writer.add_scalar("diagnostics/proposer_what", pr_m[0], global_step)
        writer.add_scalar("diagnostics/proposer_v", pr_m[1], global_step)
        writer.add_scalar("diagnostics/z_sqnorm", z_sqnorm, global_step)
        writer.add_scalar("diagnostics/rollout_goal_v", rollout_goal_v, global_step)
        writer.add_scalar("diagnostics/rollout_goal_dv", rollout_goal_dv, global_step)
        writer.add_scalar("diagnostics/policy_rhat", po_m[1], global_step)
        writer.add_scalar("diagnostics/policy_next_v", po_m[2], global_step)
        writer.add_scalar("diagnostics/goal_sensitivity_action_mse", goal_sens, global_step)
        writer.add_scalar("diagnostics/goal_removal_action_mse", goal_removal, global_step)
        writer.add_scalar("diagnostics/proposer_out_std", proposer_out_std, global_step)
        writer.add_scalar("diagnostics/proposed_goal_sensitivity", proposed_goal_sens, global_step)
        writer.add_scalar("diagnostics/action_grad_rhat", action_grad_rhat, global_step)
        writer.add_scalar("diagnostics/action_grad_value", action_grad_value, global_step)
        writer.add_scalar("diagnostics/reward_scale", r_scale.item(), global_step)
        sps = int(global_step / (time.time() - start_time))
        print(f"iter={iteration} SPS: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)

    envs.close()
    writer.close()
