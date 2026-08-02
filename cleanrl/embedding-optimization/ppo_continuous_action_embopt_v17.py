# Embedding Optimization v17 (embopt_v17) — family: cleanrl/embedding-optimization/ (see FAMILY.md)
#
# v16 -> v17: REACHABILITY IS NOT A REWARD. v16 @1.1M spiked to -67 then flatlined
# ~-100; telemetry showed the proposer exploiting the additive h term: measured
# one-step progress scales with DISTANCE to the goal (Delta-mse ~ (g-z).dz), so the
# proposer maximized progress-magnitude by proposing anti-aligned far goals
# (rollout_goal_dist ~3.0 > random-pair baseline 2.0) and SOLD the rate term to do it
# (proposer_what went negative while proposer_prog climbed). A latent treadmill: big,
# honest, measured progress toward goals re-set every step; zero reward value
# (policy_rhat pinned ~-0.7). The gr16 window ablation was strictly worse (-200),
# settling per-step as the right refresh.
# Fix: the proposer score is GROUNDED PROMISED RATE ONLY — score = W_hat(g), with
# W_hat regressed at commanded goals toward delivered rates (kept from v16).
# Reachability enters exclusively through that grounding loop: an unreachable or
# useless promise is dragged to the mediocre rate actually delivered and devalued by
# facts. h(z,g) keeps its two proper jobs: local dynamics-aware credit inside the
# policy loss, and telemetry.
# Expected signatures if right: rollout_goal_dist falls toward <= 2, goal_promise
# turns positive (above-mean regimes) with goal_delivery chasing it.
# --- v16 header follows ---
# v15 -> v16: DELTA-ONLY VALUE. v15's TD(0) critic collapsed to the trivial fixed point
# d_hat = 0 everywhere within 400k (critic_self exactly 0, td/onestep parked at the
# Huber constant-violation value 0.5, action_grad_value exactly 0, goal channel dead,
# returns at the r_hat-greedy plateau). Post-mortem, two structural facts:
#   (1) The absolute level of a goal-distance critic is NOT identifiable from facts
#       about executed transitions. TD(0) is a pure difference constraint
#       (d(z) - d(z') = 1 per step); with the bootstrap supplying its own target and no
#       far labels anywhere, the constant solution d = 0 costs a flat Huber 0.5/sample
#       and needs zero discrimination — an attractor. v13 ratcheted UP, v15 collapsed
#       DOWN: same disease, opposite signs.
#   (2) The SIGReg'd latent is fast-mixing: per-step ||dz||^2 ~ 91 vs typical random-
#       pair distance^2 = 2*dz = 128 — one env step covers ~2/3 of typical separation,
#       so metric distance saturates after ~2 steps and NO labeling scheme can calibrate
#       a multi-step absolute distance head on this geometry (v13's stuck calib ~20+
#       steps was this). HalfCheetah obs are egocentric and quasi-periodic: there is no
#       slow progress coordinate to embed.
# The absolute level is also UNNECESSARY: the policy only ever consumes the one-step
# slope, which lives exactly in the 1-2 step range where the latent metric is still
# meaningful. v16 therefore: every consumer reads a measured or predicted CHANGE in
# per-dim embedding MSE toward the goal, at ONE-STEP horizon, and goals are re-proposed
# EVERY STEP: the proposer is a per-step steering signal — each step it re-answers
# "which regime direction maximizes value from here". Dense signal (every transition
# is a labeled goal outcome), horizon matched to the only range where the latent
# metric means anything. The traded-away 16-step goal commitment is deliberate:
# long-horizon knowledge lives in W_hat's regime map, not in holding one embedding
# target through a mixing latent (a frozen target is unholdable in a quasi-periodic
# system anyway — a per-step goal can lead the gait orbit like a carrot).
#   - CRITIC -> ONE-STEP PROGRESS PREDICTOR h(z,g): regressed on the measured one-step
#     outcome of pursuit, for every executed transition:
#         h(z_t, g_t) <- mse(z_t, g_t) - mse(z_{t+1}, g_t)    (g_t = commanded at t)
#     Pure outcome regression: no relabeling, no bootstrap, no self-referential
#     target. The g-dependence of the target is CAUSAL (the commanded goal drives the
#     action that produced z_{t+1}) — a multi-step endpoint's g-dependence would be
#     washed out by latent mixing (review finding on the window variant, which
#     degenerated to "farthest goal wins"). Random motion INCREASES expected MSE, so
#     unaligned goals earn honestly negative labels — collapse-proof in both
#     directions. Pairs straddling a reset are masked. Goal-conditioned phases sample
#     only the goal_recent_iters most recent segments (latent-frame drift; v15 notes).
#   - POLICY: goal term = immediate measured step through f plus predicted next-step
#     progress from the imagined state — two steps of differentiable goal credit.
#     Local-minima duty moves to the PROPOSER: with per-step re-proposal the goal
#     itself walks around metric barriers instead of the policy having to cross them.
#   - PROPOSER: score = W_hat(g) + prog_coef * h(z,g), UNCLAMPED (standard scalar
#     critic). The support clamp is replaced by grounding W_hat at its query points:
#     every commanded goal's promised rate is regressed toward the rate actually
#     delivered around its pursuit (wm phase), so fantasy promises are corrected by
#     facts and the proposer keeps a live gradient above the best observed rate.
# Facts only, per family direction: no hindsight relabeling, no counterfactuals; every
# trained target is a measured outcome of the goal actually pursued.
#
# Method (family core): deterministic policy, one differentiable scalar
#     L_pi = -[ reward_coef * r_hat(z, pi(z,g))
#               + value_coef * ( mse(z,g) - mse(f(z,pi(z,g)), g) + h(f(z,pi(z,g)), g) ) ]
# with gradients through the frozen latent dynamics f and frozen progress head
# (dh/dz' . df/da . dpi/dtheta — the family's analytic chain). No sampling, no
# likelihood ratios, no argmax, no contrastive terms.
# Components:
#   - Encoder E + dynamics f: LeJEPA-style (prediction MSE + SIGReg and NOTHING else
#     reach the WM; reward heads read detached z; all consumers detach).
#   - h(z,g) = critic([z,g,g-z]): predicted one-step MSE progress (linear head).
#   - Proposer P(z): shell-projected direction; ascends the frozen score above.
#   - Policy pi(z,g): tanh head on [z,g,g-z]; acts with additive Gaussian noise on
#     proposed goals held goal_refresh steps.
# Freezing = backward(inputs=<phase params>) + per-phase optimizers (compile-safe).
# Lineage details and per-version verdicts: FAMILY.md.
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
    """learning rate for the critic, proposer, and policy optimizers"""
    wm_lr: float = 1e-4
    """world-model learning rate (AdamW; le-wm reference uses 5e-5 offline)"""
    wm_weight_decay: float = 1e-3
    """world-model weight decay (le-wm reference value)"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per rollout segment"""
    anneal_lr: bool = True
    """toggle learning rate annealing for all optimizers"""
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
    sigreg_ref_n: int = 128
    """pinned N-factor for the SIGReg statistic (batch-invariant strength; le-wm batch size)"""

    replay_iters: int = 64
    """ring buffer capacity in rollout segments (transitions = replay_iters*num_steps*num_envs)"""
    minibatch_size: int = 2048
    """minibatch size for all update phases"""
    wm_updates: int = 16
    """world-model minibatch updates per iteration"""
    critic_updates: int = 16
    """progress-head minibatch updates per iteration"""
    proposer_updates: int = 8
    """proposer minibatch updates per iteration"""
    policy_updates: int = 16
    """policy minibatch updates per iteration"""
    goal_recent_iters: int = 8
    """goal-conditioned phases (progress head, policy) sample only this many most recent
    segments: buf_goal stores latent coordinates and the latent frame drifts under WM
    training, so older goal vectors point at stale semantic locations"""
    rate_window: int = 16
    """forward window (steps) for the reward-rate target W_hat"""

    reward_coef: float = 1.0
    """weight of the r_hat term in the policy loss"""
    value_coef: float = 1.0
    """weight of the one-step metric-progress term in the policy loss"""
    imag_probe_h: int = 8
    """horizon for the imag_err_h diagnostic (telemetry only; no training unroll)"""
    policy_goal_mix: float = 0.5
    """fraction of the policy batch trained on freshly proposed (vs stored commanded) goals"""
    expl_noise: float = 0.2
    """std of Gaussian exploration noise added to the deterministic action"""
    goal_refresh: int = 1
    """env steps a proposed goal is held during rollout (1 = re-propose every step:
    the goal is a per-step steering target; >1 restores held-goal windows)"""
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
    # raw obs AND raw rewards on purpose: both are re-normalized with CURRENT running
    # stats at sample time, so every replayed sample shares one consistent scale
    # (wrapper-side running normalization goes stale inside the ring buffer)
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


def mlp_ln(in_dim, hidden, out_dim, out_std=1.0):
    # norm-stabilized variant for the world model (le-wm has norm layers throughout)
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, hidden)),
        nn.LayerNorm(hidden),
        nn.SiLU(),
        layer_init(nn.Linear(hidden, hidden)),
        nn.LayerNorm(hidden),
        nn.SiLU(),
        layer_init(nn.Linear(hidden, out_dim), std=out_std),
    )


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regularizer (ECF test), as in the LeJEPA lineage."""

    def __init__(self, knots=17, num_proj=256, ref_n=128):
        super().__init__()
        self.num_proj = num_proj
        self.ref_n = ref_n
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
        # pinned ref_n, NOT x.size(0): the statistic scales linearly with N, so an
        # unpinned factor couples regularizer strength to minibatch size (0.09 was
        # balanced at batch 128; at 2048 it would be ~16x too strong)
        statistic = (err @ self.weights) * self.ref_n
        return statistic.mean()


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        do = int(np.array(envs.single_observation_space.shape).prod())
        da = int(np.prod(envs.single_action_space.shape))
        dz, dh = args.latent_dim, args.hidden_dim
        self.encoder = mlp_ln(do, dh, dz)
        self.dyn = mlp_ln(dz + da, dh, dz, out_std=0.01)  # residual: f(z,a) = z + dyn([z,a])
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

    def ldist(self, z, g):
        # fixed metric potential: per-dim embedding MSE to the goal — the same units as
        # wm_pred. The policy's goal term is literally "how much does my action reduce
        # embed MSE to g"; meaningful at the 1-2 step range, the only range any
        # consumer reads
        return (g - z).square().mean(-1)

    def prog(self, z, g):
        # predicted MSE-progress-to-go toward g by the end of the current hold window:
        # measured-outcome regression target, so a plain linear head — nothing here is
        # bootstrapped or self-referential
        return self.critic(torch.cat([z, g, g - z], -1)).squeeze(-1)

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
    assert args.num_steps % args.goal_refresh == 0, "num_steps must be a multiple of goal_refresh"
    assert args.num_steps > args.imag_probe_h, "num_steps too small for the imagination probe"
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
    sigreg = SIGReg(knots=args.sigreg_knots, num_proj=args.sigreg_proj, ref_n=args.sigreg_ref_n).to(device)

    # observation running stats (raw obs stored; normalized with CURRENT stats at use time)
    obs_mean = torch.zeros(do, device=device)
    obs_var = torch.ones(do, device=device)
    obs_count = torch.tensor(1e-4, device=device)

    def rms_update(x):
        bmean, bvar, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = bmean - obs_mean
        tot = obs_count + bc
        obs_mean.add_(delta * bc / tot)
        obs_var.copy_((obs_var * obs_count + bvar * bc + delta.square() * obs_count * bc / tot) / tot)
        obs_count.copy_(tot)

    def nobs(x):
        return ((x - obs_mean) / torch.sqrt(obs_var + 1e-8)).clamp(-10.0, 10.0)

    wm_params = (
        list(agent.encoder.parameters())
        + list(agent.dyn.parameters())
        + list(agent.reward_head.parameters())
        + list(agent.rate_head.parameters())
    )
    critic_params = list(agent.critic.parameters())
    proposer_params = list(agent.proposer.parameters())
    policy_params = list(agent.policy.parameters())
    wm_opt = optim.AdamW(wm_params, lr=args.wm_lr, weight_decay=args.wm_weight_decay, eps=1e-5)
    critic_opt = optim.Adam(critic_params, lr=args.learning_rate, eps=1e-5)
    proposer_opt = optim.Adam(proposer_params, lr=args.learning_rate, eps=1e-5)
    policy_opt = optim.Adam(policy_params, lr=args.learning_rate, eps=1e-5)
    opt_base_lrs = ((wm_opt, args.wm_lr), (critic_opt, args.learning_rate), (proposer_opt, args.learning_rate), (policy_opt, args.learning_rate))

    # ---- loss / act functions (compiled) -------------------------------------------------
    def rollout_forward(obs, cur_goal, refresh):
        z = agent.encode(obs)
        goal = torch.where(refresh, agent.propose(z), cur_goal)
        action = agent.act(z, goal)
        return z, goal, action

    def wm_loss_fn(o, a, o2, r, rate, pair_valid, gcmd, grate):
        z = agent.encode(o)
        z2 = agent.encode(o2)
        pred = agent.forward_dyn(z, a)
        denom = pair_valid.sum().clamp_min(1.0)
        pred_loss = (pair_valid * (pred - z2).square().mean(-1)).sum() / denom
        sig_loss = sigreg(z)
        zd = z.detach()
        r_loss = (agent.rhat(zd, a) - r).square().mean()
        w_loss = (agent.what(zd) - rate).square().mean()
        # ground W_hat at its OTHER query distribution (goal points): promised rate at
        # each commanded goal regressed to the rate actually delivered over its hold
        # window — the fact that replaces the proposer's support clamp
        wg_loss = (agent.what(gcmd) - grate).square().mean()
        loss = pred_loss + args.sigreg_coef * sig_loss + r_loss + w_loss + wg_loss
        return loss, pred_loss.detach(), sig_loss.detach(), r_loss.detach(), w_loss.detach(), wg_loss.detach()

    def critic_loss_fn(z, g, target, m):
        # measured-outcome regression: h(z,g) vs realized progress-to-go toward the
        # goal ACTUALLY pursued from z (MSE decrease from z to its window's end). No
        # bootstrap, no relabeling — nothing to collapse into. m masks pairs whose
        # remaining window straddles an episode cut.
        pred = agent.prog(z, g)
        denom = m.sum().clamp_min(1.0)
        loss = (m * nn.functional.smooth_l1_loss(pred, target, reduction="none")).sum() / denom
        return loss, ((pred * m).sum() / denom).detach()

    def proposer_loss_fn(z):
        g = agent.propose(z)
        # score = grounded promised rate ONLY (unclamped scalar critic; W_hat is
        # regressed at commanded goals toward delivered rates, see wm_loss_fn, so
        # fantasy or unreachable promises are corrected by facts). v16's additive
        # h term is gone: measured progress scales with goal distance, so paying for
        # predicted progress buys distance, not value (the treadmill exploit).
        # h is computed here for telemetry only.
        w = agent.what(g)
        p = agent.prog(z, g)
        return -w.mean(), w.mean().detach(), p.mean().detach()

    def policy_loss_fn(z, g):
        # one differentiable scalar, one model step. Goal term = immediate metric step
        # through f PLUS learned progress-to-go from the imagined next state — together
        # "how much closer to g will I be by window end". The h term is dynamics-aware
        # credit: it is what pulls the policy through the raw metric's local minima
        # (transient move-away states that historically end closer carry high h).
        a = agent.act(z, g)
        r = agent.rhat(z, a)
        z2 = agent.forward_dyn(z, a)
        q_prog = (agent.ldist(z, g) - agent.ldist(z2, g)) + agent.prog(z2, g)
        obj = args.reward_coef * r + args.value_coef * q_prog
        return -obj.mean(), r.mean().detach(), q_prog.mean().detach()

    if args.compile:
        rollout_forward = torch.compile(rollout_forward, mode=args.compile_mode, dynamic=False)
        wm_loss_fn = torch.compile(wm_loss_fn, mode=args.compile_mode, dynamic=False)
        critic_loss_fn = torch.compile(critic_loss_fn, mode=args.compile_mode, dynamic=False)
        proposer_loss_fn = torch.compile(proposer_loss_fn, mode=args.compile_mode, dynamic=False)
        policy_loss_fn = torch.compile(policy_loss_fn, mode=args.compile_mode, dynamic=False)
        print(f"[embopt_v17] torch.compile(mode={args.compile_mode!r}, dynamic=False)")

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
    buf_goal = torch.zeros((R, T, E, dz), device=device)  # commanded goal at each step
    buf_filled, buf_ptr = 0, 0

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    rms_update(next_obs)
    next_done = torch.zeros(E).to(device)
    cur_goal = torch.zeros((E, dz), device=device)
    roll_goals = torch.zeros((T, E, dz), device=device)  # commanded goals of the fresh segment
    w_idx = torch.arange(0, T, args.goal_refresh, device=device)  # goal-hold window starts
    mb = args.minibatch_size

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            for opt, base_lr in opt_base_lrs:
                opt.param_groups[0]["lr"] = frac * base_lr

        # ---- rollout --------------------------------------------------------------------
        s = buf_ptr
        for step in range(T):
            buf_obs[s, step] = next_obs
            buf_done[s, step] = next_done
            refresh = next_done.bool() | ((step % args.goal_refresh) == 0)
            with torch.no_grad():
                mark_step()
                _, goal, action = rollout_forward(nobs(next_obs), cur_goal, refresh.unsqueeze(-1))
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
            rms_update(next_obs)
            next_done = torch.Tensor(next_done_np.astype(np.float32)).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        buf_obs[s, T] = next_obs
        buf_done[s, T] = next_done
        buf_goal[s] = roll_goals

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
        # goal-conditioned phases sample only recent segments (stale latent-frame goals)
        K = min(args.goal_recent_iters, F)
        recent_slots = torch.arange(fresh_slot, fresh_slot - K, -1, device=device) % R
        n_recent = K * T * E

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
            o = nobs(buf_obs[f_i, t_i, e_i])
            o2 = nobs(buf_obs[f_i, t_i + 1, e_i])
            a = buf_act[f_i, t_i, e_i]
            r = buf_rew[f_i, t_i, e_i] / r_scale
            rate = buf_rate[f_i, t_i, e_i] / r_scale
            pv = pair_valid_all[f_i, t_i, e_i]
            # goal-grounding batch for W_hat: commanded goals (recent segments only —
            # older slots hold stale latent-frame coordinates) paired with the rate
            # actually delivered around their pursuit
            flat2 = torch.randint(0, n_recent, (mb,), device=device)
            g_f2 = recent_slots[flat2 // (T * E)]
            g_t2 = (flat2 // E) % T
            g_e2 = flat2 % E
            gcmd = buf_goal[g_f2, g_t2, g_e2]
            grate = buf_rate[g_f2, g_t2, g_e2] / r_scale
            mark_step()
            loss, pl, sl, rl, wl, wgl = wm_loss_fn(o, a, o2, r, rate, pv, gcmd, grate)
            wm_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(wm_params, args.max_grad_norm)
            wm_opt.step()
            wm_stats.append((pl.item(), sl.item(), rl.item(), wl.item(), wgl.item()))

        # ---- progress-head training pool: measured progress-to-go outcomes --------------
        # for EVERY step t of every hold window in the recent segments: (z_t, goal
        # commanded at t) -> realized decrease in embed MSE to that goal from t to the
        # window's end (wend = next refresh point; the goal is held on (t, wend]).
        # Encodings use the CURRENT encoder; pairs straddling a cut are masked.
        with torch.no_grad():
            Zs = agent.encode(nobs(buf_obs[recent_slots].reshape(-1, do))).reshape(K, T + 1, E, dz)
            wend_idx = ((torch.arange(T, device=device) // args.goal_refresh) + 1) * args.goal_refresh  # (T,)
            g_pool = buf_goal[recent_slots]  # (K,T,E,dz)
            epid_seg = buf_done[recent_slots].cumsum(dim=1)  # (K,T+1,E)
            m_pool = (epid_seg[:, :T] == epid_seg[:, wend_idx]).float()  # (K,T,E)
            tgt_pool = agent.ldist(Zs[:, :T], g_pool) - agent.ldist(Zs[:, wend_idx], g_pool)  # (K,T,E)
            z_f = Zs[:, :T].reshape(-1, dz)
            g_f = g_pool.reshape(-1, dz)
            tgt_f = tgt_pool.reshape(-1)
            m_f = m_pool.reshape(-1)
            n_pool = K * T * E

        # ---- progress-head updates -------------------------------------------------------
        critic_stats = []
        for _ in range(args.critic_updates):
            idx = torch.randint(0, n_pool, (mb,), device=device)
            mark_step()
            loss, pmean = critic_loss_fn(z_f[idx], g_f[idx], tgt_f[idx], m_f[idx])
            critic_opt.zero_grad(set_to_none=True)
            loss.backward(inputs=critic_params)
            nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
            critic_opt.step()
            critic_stats.append((loss.item(), pmean.item()))

        # ---- proposer updates: ascend the frozen score surface along the shell -----------
        # states from recent segments (h and W_hat's goal grounding both train there)
        prop_stats = []
        for _ in range(args.proposer_updates):
            flat = torch.randint(0, n_recent, (mb,), device=device)
            with torch.no_grad():
                zb = agent.encode(nobs(buf_obs[recent_slots[flat // (T * E)], (flat // E) % T, flat % E]))
            mark_step()
            loss, wmean, pmean = proposer_loss_fn(zb)
            proposer_opt.zero_grad(set_to_none=True)
            loss.backward(inputs=proposer_params)
            nn.utils.clip_grad_norm_(proposer_params, args.max_grad_norm)
            proposer_opt.step()
            prop_stats.append((wmean.item(), pmean.item()))

        # ---- policy updates (frozen f, r_hat) --------------------------------------------
        # goals = freshly proposed (n_prop) + stored commanded (rest); the stored half is
        # the progress head's training distribution, the proposed half leads it by one
        # proposer update (those goals become commanded data next iteration)
        n_prop = int(mb * args.policy_goal_mix)
        pol_stats = []
        for _ in range(args.policy_updates):
            flat = torch.randint(0, n_recent, (mb,), device=device)
            f_i = recent_slots[flat // (T * E)]
            t_i = (flat // E) % T
            e_i = flat % E
            with torch.no_grad():
                zb = agent.encode(nobs(buf_obs[f_i, t_i, e_i]))
                gp = agent.propose(zb[:n_prop])
            gb = torch.cat([gp, buf_goal[f_i, t_i, e_i][n_prop:]], 0)
            mark_step()
            loss, rmean, pmean = policy_loss_fn(zb, gb)
            policy_opt.zero_grad(set_to_none=True)
            loss.backward(inputs=policy_params)
            nn.utils.clip_grad_norm_(policy_params, args.max_grad_norm)
            policy_opt.step()
            pol_stats.append((loss.item(), rmean.item(), pmean.item()))

        # ---- diagnostics: is the goal channel alive? (family-mandated) -------------------
        with torch.no_grad():
            # fresh-segment encodings for all telemetry
            Zf = agent.encode(nobs(buf_obs[fresh_slot].reshape(-1, do))).reshape(T + 1, E, dz)
            epid_f = buf_done[fresh_slot].cumsum(dim=0)  # (T+1,E)
            # goal-approach telemetry on the fresh segment: per-step realized progress
            # toward the commanded goal (positive = approaching), and mean distance
            Dg = agent.ldist(Zf[:T], roll_goals)  # (T,E)
            rollout_goal_dist = Dg.mean().item()
            # realized per-step progress toward each step's OWN commanded goal (the
            # fresh slice of the training targets, mask-weighted)
            rollout_goal_prog = ((tgt_pool[0] * m_pool[0]).sum() / m_pool[0].sum().clamp_min(1.0)).item()
            z_sqnorm = Zf[:T].square().sum(-1).mean().item()
            # LeJEPA health: effective rank of the embedding covariance (participation
            # ratio, max = dz) — z_sqnorm alone cannot see dimensional collapse
            z_flat = Zf[:T].reshape(-1, dz)
            z_cent = z_flat - z_flat.mean(0)
            eig = torch.linalg.eigvalsh(z_cent.T @ z_cent / (z_flat.shape[0] - 1))
            z_eff_rank = (eig.sum().square() / eig.square().sum().clamp_min(1e-12)).item()
            z_top_share = (eig[-1] / eig.sum().clamp_min(1e-12)).item()
            # no-change baseline for wm_pred: per-dim Var(z' - z); if wm_pred is not well
            # below this, the dynamics has learned nothing beyond identity
            dstep = Zf[1:] - Zf[:-1]
            dmask = (buf_done[fresh_slot, 1 : T + 1] == 0).float().unsqueeze(-1)
            wm_delta_var = ((dstep.square() * dmask).sum() / (dmask.sum() * dz).clamp_min(1.0)).item()
            # imagined vs realized H-step unroll under EXECUTED actions (per-dim MSE;
            # compare wm_pred and wm_delta_var)
            H = args.imag_probe_h
            zc = Zf[: T - H]
            for i in range(H):
                zc = agent.forward_dyn(zc, buf_act[fresh_slot, i : i + T - H])
            imag_m = (epid_f[H:T] == epid_f[: T - H]).float()
            imag_err = (((zc - Zf[H:T]).square().mean(-1) * imag_m).sum() / imag_m.sum().clamp_min(1.0)).item()
            # "test them well": h calibration on the fresh segment (fresh pairs enter
            # the training pool this same iteration, so this measures fit on the newest
            # outcomes, not held-out generalization — promise/delivery below stays the
            # independent check)
            pred_fresh = agent.prog(Zs[0, :T].reshape(-1, dz), g_pool[0].reshape(-1, dz)).reshape(T, E)
            mf0 = m_pool[0]
            denf = mf0.sum().clamp_min(1.0)
            prog_calib_bias = (((pred_fresh - tgt_pool[0]) * mf0).sum() / denf).item()
            prog_calib_abs = (((pred_fresh - tgt_pool[0]).abs() * mf0).sum() / denf).item()
            # goal-channel aliveness on the progress-head/policy query distribution
            flat = torch.randint(0, n_recent, (mb,), device=device)
            f_i = recent_slots[flat // (T * E)]
            t_i = (flat // E) % T
            e_i = flat % E
            zb = agent.encode(nobs(buf_obs[f_i, t_i, e_i]))
            gb = buf_goal[f_i, t_i, e_i]
            a_matched = agent.act(zb, gb)
            a_shuffled = agent.act(zb, gb[torch.randperm(mb, device=device)])
            goal_sens = (a_matched - a_shuffled).square().mean().item()
            # commanded goal vs null goal (g = current z): the diagnostic that failed first
            # in every dead pan-goal-solver version — must lift off ~0
            a_null = agent.act(zb, zb)
            goal_removal = (a_matched - a_null).square().mean().item()
            # proposer collapse watch: output diversity across states, and goal
            # sensitivity measured on PROPOSED goals (stored commanded goals span proposer
            # history, so the metrics above cannot see a freshly collapsed proposer)
            gp_diag = agent.propose(zb)
            proposer_out_std = gp_diag.std(0).mean().item()
            w_real = agent.what(zb)
            what_real_mean = w_real.mean().item()
            what_real_max = w_real.max().item()
            a_prop = agent.act(zb, gp_diag)
            a_prop_sh = agent.act(zb, gp_diag[torch.randperm(mb, device=device)])
            proposed_goal_sens = (a_prop - a_prop_sh).square().mean().item()
            # promise vs delivery — the rate the proposer promised at each held goal vs
            # the rate the policy actually realized over the hold
            goal_promise = agent.what(roll_goals[w_idx]).mean().item()
            goal_delivery = (buf_rate[fresh_slot][w_idx] / r_scale).mean().item()
        # measured (not assumed) balance of the two policy-loss terms at the action
        ad = agent.act(zb, gb).detach()
        a1 = ad.clone().requires_grad_(True)
        grad_r = torch.autograd.grad(agent.rhat(zb, a1).sum(), a1)[0]
        a2 = ad.clone().requires_grad_(True)
        z2_diag = agent.forward_dyn(zb, a2)
        grad_v = torch.autograd.grad((agent.prog(z2_diag, gb) - agent.ldist(z2_diag, gb)).sum(), a2)[0]
        action_grad_rhat = (args.reward_coef * grad_r).norm(dim=-1).mean().item()
        action_grad_value = (args.value_coef * grad_v).norm(dim=-1).mean().item()

        wm_m = np.mean(wm_stats, axis=0)
        cr_m = np.mean(critic_stats, axis=0)
        pr_m = np.mean(prop_stats, axis=0)
        po_m = np.mean(pol_stats, axis=0)
        writer.add_scalar("charts/learning_rate", wm_opt.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/wm_pred", wm_m[0], global_step)
        writer.add_scalar("losses/wm_sigreg", wm_m[1], global_step)
        writer.add_scalar("losses/reward_mse", wm_m[2], global_step)
        writer.add_scalar("losses/rate_mse", wm_m[3], global_step)
        writer.add_scalar("losses/rate_goal_mse", wm_m[4], global_step)
        writer.add_scalar("losses/critic", cr_m[0], global_step)
        writer.add_scalar("losses/policy", po_m[0], global_step)
        writer.add_scalar("diagnostics/prog_pred_mean", cr_m[1], global_step)
        writer.add_scalar("diagnostics/proposer_what", pr_m[0], global_step)
        writer.add_scalar("diagnostics/proposer_prog", pr_m[1], global_step)
        writer.add_scalar("diagnostics/what_real_mean", what_real_mean, global_step)
        writer.add_scalar("diagnostics/what_real_max", what_real_max, global_step)
        writer.add_scalar("diagnostics/z_sqnorm", z_sqnorm, global_step)
        writer.add_scalar("diagnostics/z_eff_rank", z_eff_rank, global_step)
        writer.add_scalar("diagnostics/z_top_eig_share", z_top_share, global_step)
        writer.add_scalar("diagnostics/wm_delta_var", wm_delta_var, global_step)
        writer.add_scalar("diagnostics/imag_err_h", imag_err, global_step)
        writer.add_scalar("diagnostics/prog_calib_bias", prog_calib_bias, global_step)
        writer.add_scalar("diagnostics/prog_calib_abs", prog_calib_abs, global_step)
        writer.add_scalar("diagnostics/goal_promise", goal_promise, global_step)
        writer.add_scalar("diagnostics/goal_delivery", goal_delivery, global_step)
        writer.add_scalar("diagnostics/rollout_goal_dist", rollout_goal_dist, global_step)
        writer.add_scalar("diagnostics/rollout_goal_prog", rollout_goal_prog, global_step)
        writer.add_scalar("diagnostics/policy_rhat", po_m[1], global_step)
        writer.add_scalar("diagnostics/policy_prog", po_m[2], global_step)
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
