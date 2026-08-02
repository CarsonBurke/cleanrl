# Embedding Optimization v15 (embopt_v15) — family: cleanrl/embedding-optimization/ (see FAMILY.md)
#
# v13 -> v15 (v14 = exact-k label variant, cancelled): HINDSIGHT RELABELING REMOVED.
# Verdict on the relabeling branch: v13 finished 8M flat (~0 return at every checkpoint,
# @2M..@8M all within -20..+20) and v14 tracked it at 2M. The relabeling tower —
# ring-buffer segment recursions, sentinel arrival indices, evidence weights arbitrating
# three label sources — did not produce a critic the policy could climb, and it obscured
# the family's core loop. v15 rebuilds the critic on the EXECUTED TRANSITION only.
# Take an action, observe z_{t+1}; every critic label is a fact about that transition:
#     TD(0):    d(z_t, g_cmd) <- 1 + d(z_{t+1}, g_cmd)  (g_cmd = goal actually pursued)
#     one-step: d(z_t, z_{t+1}) -> 1                     (the observed step: "V(z, z+1)")
#     identity: d(z, z) -> 0                             (arrival semantics, no event)
# Episode cuts are masked, not asserted to a floor; the bootstrap is detached
# (semi-gradient) and CAPPED at num_steps — the 1 + d recursion is not a contraction
# and has no arrival event for never-approached goals, so the cap defines the head's
# dynamic range [0, num_steps] by label construction (unreachable goals saturate at the
# horizon instead of ratcheting upward without bound). Grounding logic: identity +
# one-step anchor the near field of the distance head ([z,g,g-z] with small g-z IS the
# anchors' input region, so near-arrival generalizes); TD(0) propagates those anchors
# backward along real paths toward the goals the policy actually pursued; goals never
# approached saturate at the cap — honest devaluation emerges instead of being asserted.
# Goal-conditioned phases (critic, policy) sample only the goal_recent_iters most recent
# segments: buf_goal stores latent COORDINATES and the latent frame drifts under WM
# training (SIGReg pins only the marginal), so old goal vectors point at stale semantic
# locations. The WM keeps the full buffer. Deleted with the relabeler:
# cross-trajectory negatives (asserted unmeasured "far" labels) and evidence weights
# (every remaining label IS evidence). The policy trains on proposed goals (fresh) +
# stored commanded goals — exactly the distribution the critic trains on.
# dist_calib_* becomes a PURE MEASUREMENT: d_hat(z_t, z_{t+k}) vs observed k for
# k in {2,4,8,16} — multi-step pairs are never trained on (k=1 is, so it is excluded);
# an independent generalization test at last.
#
# Method (family core, unchanged): deterministic policy, one differentiable scalar
#     L_pi = -[ reward_coef * r_hat(z, pi(z,g)) - value_coef * d_hat(f(z, pi(z,g)), g) ]
# with gradients through the frozen latent dynamics f and frozen distance critic.
# No sampling, no likelihood ratios, no argmax, no contrastive terms.
# Components:
#   - Encoder E + dynamics f: LeJEPA-style (prediction MSE + SIGReg and NOTHING else
#     reach the WM; reward heads read detached z; all consumers detach).
#   - d_hat(z,g) = softplus(critic([z,g,g-z]) + d_init): steps-to-go head, V = gamma^d;
#     trained by the three fact losses above (TD(0) in distance space, Huber).
#   - Proposer P(z): shell-projected direction; ascends support-clamped W_hat(g) minus
#     an aspiration-band penalty pinning d_hat(z,g) at the goal-hold horizon.
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
    gamma_goal: float = 0.98
    """goal discount defining the value metric V = gamma_goal^d"""
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
    """critic minibatch updates per iteration"""
    proposer_updates: int = 8
    """proposer minibatch updates per iteration"""
    policy_updates: int = 16
    """policy minibatch updates per iteration"""
    onestep_coef: float = 1.0
    """weight of the observed one-step fact d(z_t, z_{t+1}) -> 1"""
    self_coef: float = 1.0
    """weight of the identity fact d(z, z) -> 0 (the head's arrival grounding)"""
    goal_recent_iters: int = 8
    """goal-conditioned phases (critic, policy) sample only this many most recent
    segments: buf_goal stores latent coordinates and the latent frame drifts under WM
    training, so older goal vectors point at stale semantic locations"""
    rate_window: int = 16
    """forward window (steps) for the reward-rate target W_hat"""

    reward_coef: float = 1.0
    """weight of the r_hat term in the policy loss"""
    value_coef: float = 1.0
    """weight of the gamma_g*V term in the policy loss"""
    beta_band: float = 4.0
    """weight of the ((d_hat - goal_refresh)/goal_refresh)^2 band penalty in the proposer score"""
    d_init: float = 16.0
    """distance-head init bias: d_hat starts near the goal-hold horizon, not near 0"""
    imag_probe_h: int = 8
    """horizon for the imag_err_h diagnostic (telemetry only; no training unroll)"""
    policy_goal_mix: float = 0.5
    """fraction of the policy batch trained on freshly proposed (vs stored commanded) goals"""
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
        self.d_init = args.d_init
        self.gamma_goal = args.gamma_goal

    def encode(self, obs):
        return self.encoder(obs)

    def forward_dyn(self, z, a):
        return z + self.dyn(torch.cat([z, a], -1))

    def dist(self, z, g):
        # steps-to-go: nonnegative, gradient magnitude uniform at any range (unlike a
        # sigmoid value head, whose tail starves the policy's action-gradient)
        return nn.functional.softplus(self.critic(torch.cat([z, g, g - z], -1)) + self.d_init).squeeze(-1)

    def value(self, z, g):
        # gamma-space view of the same head (bootstrapping + telemetry continuity)
        return self.gamma_goal**self.dist(z, g)

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
    assert args.num_steps >= 16 and args.num_steps > args.imag_probe_h, "num_steps too small for the calibration/imagination probes"
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

    def critic_loss_fn(z, g, z2, pv):
        # every label is a fact about the executed transition (no relabeling):
        #   TD(0) in distance space toward the goal ACTUALLY pursued (semi-gradient,
        #   detached bootstrap), the observed one-step distance, and the identity
        #   anchor that gives the head its arrival semantics without an arrival event.
        # pv masks pairs straddling a reset — cuts are excluded, not asserted to a floor.
        # The bootstrap is capped at num_steps: 1 + d is not a contraction and commanded
        # goals have no arrival event, so the cap defines the head's dynamic range
        # [0, num_steps] by label construction (saturation instead of unbounded ratchet).
        d = agent.dist(z, g)
        boot = (1.0 + agent.dist(z2, g)).clamp(max=float(args.num_steps)).detach()
        denom = pv.sum().clamp_min(1.0)
        td = (pv * nn.functional.smooth_l1_loss(d, boot, reduction="none")).sum() / denom
        one = (pv * nn.functional.smooth_l1_loss(agent.dist(z, z2), torch.ones_like(d), reduction="none")).sum() / denom
        self_l = nn.functional.smooth_l1_loss(agent.dist(z, z), torch.zeros_like(d))
        loss = td + args.onestep_coef * one + args.self_coef * self_l
        return loss, td.detach(), one.detach(), self_l.detach(), (args.gamma_goal**d).mean().detach()

    def proposer_loss_fn(z, r_lo, r_hi):
        g = agent.propose(z)
        # support-clamped rate (fantasy capped at best OBSERVED regime; cap ratchets up
        # with performance) + aspiration band pinning goals at the goal-hold horizon.
        # The band's V is honest: a fantasy proposal gets commanded, is never approached,
        # and TD(0) bootstrap drift devalues it by real experience; the band then pushes
        # the proposer away from it.
        w = agent.what(g).clamp(r_lo, r_hi)
        d = agent.dist(z, g)
        score = (w - r_lo) - args.beta_band * ((d - args.goal_refresh) / args.goal_refresh).square()
        return -score.mean(), w.mean().detach(), (args.gamma_goal**d).mean().detach()

    def policy_loss_fn(z, g):
        # one differentiable scalar, one model step; distance replaces gamma^d value so
        # "one step closer" and "one step of normalized reward" share units and the
        # goal channel's action-gradient is range-independent
        a = agent.act(z, g)
        r = agent.rhat(z, a)
        d2 = agent.dist(agent.forward_dyn(z, a), g)
        obj = args.reward_coef * r - args.value_coef * d2
        return -obj.mean(), r.mean().detach(), (args.gamma_goal**d2).mean().detach()

    if args.compile:
        rollout_forward = torch.compile(rollout_forward, mode=args.compile_mode, dynamic=False)
        wm_loss_fn = torch.compile(wm_loss_fn, mode=args.compile_mode, dynamic=False)
        critic_loss_fn = torch.compile(critic_loss_fn, mode=args.compile_mode, dynamic=False)
        proposer_loss_fn = torch.compile(proposer_loss_fn, mode=args.compile_mode, dynamic=False)
        policy_loss_fn = torch.compile(policy_loss_fn, mode=args.compile_mode, dynamic=False)
        print(f"[embopt_v15] torch.compile(mode={args.compile_mode!r}, dynamic=False)")

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
            mark_step()
            loss, pl, sl, rl, wl = wm_loss_fn(o, a, o2, r, rate, pv)
            wm_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(wm_params, args.max_grad_norm)
            wm_opt.step()
            wm_stats.append((pl.item(), sl.item(), rl.item(), wl.item()))

        # ---- critic updates: TD(0) on executed transitions -------------------------------
        # recent-slot sampler; the goal input is the goal ACTUALLY pursued at that stored
        # step (buf_goal) — no relabeled columns, no recursion, no sentinels
        critic_stats = []
        for _ in range(args.critic_updates):
            flat = torch.randint(0, n_recent, (mb,), device=device)
            f_i = recent_slots[flat // (T * E)]
            t_i = (flat // E) % T
            e_i = flat % E
            with torch.no_grad():
                zb = agent.encode(nobs(buf_obs[f_i, t_i, e_i]))
                z2b = agent.encode(nobs(buf_obs[f_i, t_i + 1, e_i]))
            gb = buf_goal[f_i, t_i, e_i]
            pv = pair_valid_all[f_i, t_i, e_i]
            mark_step()
            loss, tdm, onem, selfm, vmean = critic_loss_fn(zb, gb, z2b, pv)
            critic_opt.zero_grad(set_to_none=True)
            loss.backward(inputs=critic_params)
            nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
            critic_opt.step()
            critic_stats.append((tdm.item(), onem.item(), selfm.item(), vmean.item()))

        # ---- proposer updates: ascend the frozen banded score surface along the shell ----
        prop_stats = []
        for _ in range(args.proposer_updates):
            flat = torch.randint(0, n_flat, (mb,), device=device)
            with torch.no_grad():
                zb = agent.encode(nobs(buf_obs[flat // (T * E), (flat // E) % T, flat % E]))
                mb_rates = buf_rate[flat // (T * E), (flat // E) % T, flat % E] / r_scale
                r_lo, r_hi = mb_rates.min(), mb_rates.max()
            mark_step()
            loss, wmean, vmean = proposer_loss_fn(zb, r_lo, r_hi)
            proposer_opt.zero_grad(set_to_none=True)
            loss.backward(inputs=proposer_params)
            nn.utils.clip_grad_norm_(proposer_params, args.max_grad_norm)
            proposer_opt.step()
            prop_stats.append((wmean.item(), vmean.item()))

        # ---- policy updates (frozen f, r_hat, V) -----------------------------------------
        # goals = freshly proposed (n_prop) + stored commanded (rest); the stored half is
        # the critic's training distribution, the proposed half leads it by one proposer
        # update (the critic sees those goals as commanded data next iteration)
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
            loss, rmean, vmean = policy_loss_fn(zb, gb)
            policy_opt.zero_grad(set_to_none=True)
            loss.backward(inputs=policy_params)
            nn.utils.clip_grad_norm_(policy_params, args.max_grad_norm)
            policy_opt.step()
            pol_stats.append((loss.item(), rmean.item(), vmean.item()))

        # ---- diagnostics: is the goal channel alive? (family-mandated) -------------------
        with torch.no_grad():
            # fresh-segment encodings for all telemetry
            Zf = agent.encode(nobs(buf_obs[fresh_slot].reshape(-1, do))).reshape(T + 1, E, dz)
            epid_f = buf_done[fresh_slot].cumsum(dim=0)  # (T+1,E)
            # goal-reaching telemetry on the fresh segment: is the policy climbing V
            # toward its commanded goal?
            Vg = agent.value(Zf[:T].reshape(-1, dz), roll_goals.reshape(-1, dz)).reshape(T, E)
            same_goal = (roll_goals[1:] == roll_goals[:-1]).all(-1) & (buf_done[fresh_slot, 1:T] == 0)
            samef = same_goal.float()
            rollout_goal_v = Vg.mean().item()
            rollout_goal_dv = (((Vg[1:] - Vg[:-1]) * samef).sum() / samef.sum().clamp_min(1.0)).item()
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
            # imagined vs realized H-step unroll under EXECUTED actions: the substrate of
            # the policy's exploitation gap (per-dim MSE; compare wm_pred and wm_delta_var)
            H = args.imag_probe_h
            zc = Zf[: T - H]
            for i in range(H):
                zc = agent.forward_dyn(zc, buf_act[fresh_slot, i : i + T - H])
            imag_m = (epid_f[H:T] == epid_f[: T - H]).float()
            imag_err = (((zc - Zf[H:T]).square().mean(-1) * imag_m).sum() / imag_m.sum().clamp_min(1.0)).item()
            # "test them well" (1): distance calibration is now a PURE MEASUREMENT —
            # observed transit pairs (z_t, z_{t+k}) are never trained on, so this probes
            # how the TD(0)+anchor critic generalizes along real paths, independently
            bias_acc, abs_acc, cnt_acc = 0.0, 0.0, 0.0
            for k in (2, 4, 8, 16):
                mk = (epid_f[k:T] == epid_f[: T - k]).float()
                dk = agent.dist(Zf[: T - k].reshape(-1, dz), Zf[k:T].reshape(-1, dz)).reshape(T - k, E)
                errk = dk - float(k)
                bias_acc += (errk * mk).sum().item()
                abs_acc += (errk.abs() * mk).sum().item()
                cnt_acc += mk.sum().item()
            dist_calib_bias = bias_acc / max(cnt_acc, 1.0)
            dist_calib_abs = abs_acc / max(cnt_acc, 1.0)
            # goal-channel aliveness on the critic/policy query distribution
            flat = torch.randint(0, n_flat, (mb,), device=device)
            zb = agent.encode(nobs(buf_obs[flat // (T * E), (flat // E) % T, flat % E]))
            gb = buf_goal[flat // (T * E), (flat // E) % T, flat % E]
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
            # "test them well" (2): promise vs delivery — the rate the proposer promised
            # at each held goal vs the rate the policy actually realized over the hold
            goal_promise = agent.what(roll_goals[w_idx]).mean().item()
            goal_delivery = (buf_rate[fresh_slot][w_idx] / r_scale).mean().item()
        # measured (not assumed) balance of the two policy-loss terms at the action
        ad = agent.act(zb, gb).detach()
        a1 = ad.clone().requires_grad_(True)
        grad_r = torch.autograd.grad(agent.rhat(zb, a1).sum(), a1)[0]
        a2 = ad.clone().requires_grad_(True)
        grad_v = torch.autograd.grad(agent.dist(agent.forward_dyn(zb, a2), gb).sum(), a2)[0]
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
        writer.add_scalar("losses/critic", cr_m[0], global_step)
        writer.add_scalar("losses/critic_onestep", cr_m[1], global_step)
        writer.add_scalar("losses/critic_self", cr_m[2], global_step)
        writer.add_scalar("losses/policy", po_m[0], global_step)
        writer.add_scalar("diagnostics/critic_v_mean", cr_m[3], global_step)
        writer.add_scalar("diagnostics/proposer_what", pr_m[0], global_step)
        writer.add_scalar("diagnostics/proposer_v", pr_m[1], global_step)
        writer.add_scalar("diagnostics/what_real_mean", what_real_mean, global_step)
        writer.add_scalar("diagnostics/what_real_max", what_real_max, global_step)
        writer.add_scalar("diagnostics/z_sqnorm", z_sqnorm, global_step)
        writer.add_scalar("diagnostics/z_eff_rank", z_eff_rank, global_step)
        writer.add_scalar("diagnostics/z_top_eig_share", z_top_share, global_step)
        writer.add_scalar("diagnostics/wm_delta_var", wm_delta_var, global_step)
        writer.add_scalar("diagnostics/imag_err_h", imag_err, global_step)
        writer.add_scalar("diagnostics/dist_calib_bias", dist_calib_bias, global_step)
        writer.add_scalar("diagnostics/dist_calib_abs", dist_calib_abs, global_step)
        writer.add_scalar("diagnostics/goal_promise", goal_promise, global_step)
        writer.add_scalar("diagnostics/goal_delivery", goal_delivery, global_step)
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
