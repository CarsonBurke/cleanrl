# Embedding Optimization v13 (embopt_v13) — family: cleanrl/embedding-optimization/ (see FAMILY.md)

# RAW BACKPORT (embopt_v13_raw): identical algorithm to v13 with the v19
# stationarity fixes only — no obs RMS (raw obs; encoder LayerNorm absorbs
# scale), no reward scaling (raw reward/rate regression targets), and split
# clip budgets (encoder+dyn vs reward-unit heads). v17/v18/v19 post-mortems:
# running scales are success-coupled and destroy learned behavior at the spike.
#
# v12 -> v13: EVIDENCE-WEIGHTED CRITIC. v12's new calibration probe caught a structural
# flaw within 400k: dist_calib_bias ~ +77 — the head saturated at "everything is ~128
# steps away" (critic_v_mean 0.077 => mean d_hat ~ 127), so the policy's strong new
# d-gradient (9.9 vs r_hat 2.2 — the starvation DID flip) was the gradient of a
# nearly-constant-in-truth function: amplified noise, oscillating returns (-113 best).
# Root cause is a units side-effect of the d-space move: commanded columns are a pure
# gamma-ratchet (no arrival can ever fire for a proposed embedding, so their targets can
# only decay toward floor regardless of partial progress) and negatives also push d ->
# 128; in gamma-space these floor-region labels carried tiny MSE gradients (benign
# pessimism), but Huber in d-space gives EVERY sample unit gradient — label volume wins
# (~75% of the batch says "far") and the head collapses onto the majority label.
# Fix: weight the critic loss by evidence strength. Hindsight pairs (exact ground-truth
# distances k = j - t) weight 1.0; commanded pairs (a "far unless proven otherwise"
# prior that structurally cannot register success) cmd_weight = 0.15; negatives coef
# 0.25 -> 0.05. The honest-devaluation loop survives at reduced gain; the exact labels
# reclaim the head. sample_pairs additionally returns the is-hindsight mask.
# --- v12 header follows ---
# v11 -> v12: DISTANCE-SPACE CRITIC. v11 verdict @2.2M: -162 = the v8 plateau; honest V
# alone does not restore locomotion. Component audit against the core idea (goal
# predictor finds best-value goals / value predictor gives accurate values of state
# differences / policy improves state value):
#   - V trains and tests well (no optimism gap, low loss);
#   - the POLICY's one-step improvement signal is starved: an honest V of a 16-step goal
#     genuinely changes little per action, V = sigmoid near its floor attenuates it
#     further, and r_hat drowns it 6:1 (action_grad 0.58 vs 0.09) — while one-step
#     greedy r_hat has no gait in horizon (policy_rhat -0.88, exactly v8);
#   - the proposer is ground into conservatism by the honest test it can't pass
#     (proposer_what ~ mean rate, not max; proposer_v 0.25 << 0.72 band target).
# Fix, staying strictly one-step (no unrolling): reparameterize value as DISTANCE.
#     d_hat(z,g) = softplus(critic([z,g,g-z]) + d_init)   (steps-to-go; V := gamma^d_hat)
#   - critic trains in step units: the SAME lambda-return recursion (gamma-space,
#     unchanged) converted once, d_target = log(G)/log(gamma) (arrival -> 0, floor ->
#     num_steps); Huber loss. Negatives regress to d = num_steps.
#   - policy: L_pi = -[reward_coef * r_hat - value_coef * d_hat(f(z,pi), g)]. Units are
#     commensurable ("one step closer" vs "one step of normalized reward") and
#     grad(d_hat) has UNIFORM magnitude at any range — no sigmoid-tail starvation.
#   - proposer band in d-space: score = (clamp(W_hat)-r_lo) - beta_band*((d_hat-H)/H)^2,
#     H = goal_refresh: "best-rate goal exactly 16 steps away", in native units.
# "Test them well" telemetry added: dist_calib_bias/abs (hindsight pairs have EXACT
# known step-distance k = j - t; log mean d_hat - k) and goal_promise/goal_delivery
# (W_hat(g) at proposal vs realized rate over the hold window).
# --- v11 header follows ---
# v9 -> v11 (v10 = cancelled SVG cost variant): UNROLLING REMOVED per user direction —
# SVG is not this family's mechanism, and v9's returns were high-variance and plateaued
# (+1154 @2.4M -> 523 +/- 118 @4.75M; 8-step gradient chains through a learned f with no
# trust region are noisy improvement steps). v9 changed TWO things; only one was the
# unroll. The other — COMMANDED-GOAL TD (V learns from what the policy actually pursues;
# unfulfilled promises devalued by real experience) — is kept: it is what collapsed the
# optimism gap (policy_next_v 0.70 -> 0.09) and it is exactly the family spec.
# v11 = the ORIGINAL one-step objective + the honest V:
#     L_pi = -[ r_hat(z, pi(z,g)) + gamma_g * V(f(z, pi(z,g)), g) ]
# This is also the missing attribution arm: v8 = one-step without honest V (-160);
# v9 = unroll + honest V (~1000, unstable); v11 = one-step + honest V. Family thesis:
# long-horizon credit belongs in V, not in unrolling — an honest V should let the
# one-step objective locomote at one-step variance.
# Proposer: seeker unroll removed; back to the v6 direct-output shell proposer with the
# aspiration band — but the band is now GROUNDED in a loop v6 never had: a fantasy
# proposal gets commanded, fails, is devalued by commanded-goal TD, and the band penalty
# then pushes the proposer away from it. Score:
#     (clamp(W_hat(g), r_lo, r_hi) - r_lo) - beta_band * (V(z,g) - gamma_g^16)^2
# Kept from v9: commanded-goal TD, pruned critic (lambda-main + negatives), imag_err_h
# telemetry (still the f-validity alarm at the policy's operating point, now H=8 probe).
# --- v9 header follows ---
# v8 -> v9: H-step imagined unroll — the anti-exploitation structure applied to the policy
# and the proposer at once. v8's 2M diagnosis: every infrastructure layer is verifiably
# healthy (wm_pred 0.008 vs 1.5 baseline, band holds, reward head accurate) yet
# policy_next_v climbs 0.52 -> 0.70 while realized V stays ~0.41 and rollout_goal_dv < 0:
# ONE-STEP lookahead against frozen heads is structurally cheap to exploit — a single f
# evaluation can't be forced to be self-consistent, and the patch-chase (v7 consistency,
# measurably null) can't outrun an optimizer that searches.
# (1) POLICY: L_pi = -[ sum_{i<H} gamma^i r_hat(z_i, a_i) + gamma^H V(z_H, g) ], H=8,
#     unrolled through frozen f. Compounding model error makes fantasy self-punishing,
#     and H steps of r_hat credit puts gait INSIDE the policy's own horizon (v8's
#     policy_rhat ~ -0.9 showed one-step greedy has no gait to find).
# (2) PROPOSER: replaced the shell-output network with a goal-SEEKER u(z') -> action,
#     unrolled goal_refresh=16 steps through frozen f; the goal IS the endpoint z_16.
#     Reachability is now structural — g is on f's manifold and exactly 16 model-steps
#     away BY CONSTRUCTION — which deletes the aspiration band, the V term in the score,
#     and the whole exploit-patch tower. Score: discounted imagined path reward +
#     gamma^16 * (support-clamped W_hat at the endpoint, rate->window-sum scaled).
#     v8 showed the fantasy migrates to the least-grounded ascended head (proposer_what
#     4.5 vs real support < 0 within 400k of unpinning W_hat); here W_hat is only ever
#     evaluated at f-images and its payoff stays support-clamped.
# (3) PRUNED by v7's own falsification: the V(f(z,a),g) consistency term (v7 == v6b at
#     matched steps) and its targets_next/action plumbing. Critic = lambda-return main
#     + cross-trajectory negatives + commanded-goal relabeling (4), nothing else.
# (4) COMMANDED-GOAL TD: V previously trained ONLY on hindsight pairs (goals = achieved
#     states) — the commanded (z, g_proposed) pairs the policy actually pursues at
#     rollout never received real targets, so V could promise 0.72 for a fantasy goal
#     forever and no data ever contradicted it (the substrate of policy_next_v 0.52->0.70
#     optimism). Fix is a second PAIR SOURCE for the SAME lambda-return loss: each stored
#     segment is also relabeled with its commanded goals (held goals at window starts,
#     stored in the ring buffer); the backward recursion runs with NO arrival event —
#     bootstrap from the segment-end V, gamma^T floor at episode cuts — so goals the
#     policy fails to approach are contracted toward the floor BY REAL EXPERIENCE.
#     Unfulfilled promises now get devalued; approached goals stay supported by
#     hindsight coverage. This is the graded off-manifold signal v7's FAMILY entry
#     recorded as the accepted gap.
# proposer_v telemetry is now a free calibration probe: V(z, g) at proposed goals should
# sit near gamma^16 ~ 0.72 if V is honest — deviation measures V's model-image error.
# New telemetry: imag_err_h (imagined H-step f-unroll under EXECUTED actions vs realized
# z_{t+H} on the fresh segment) — the direct measure of the exploitation gap's substrate.
# Watch item: u can still steer imagined states into f's own blind spots (self-consistent
# fantasy fixed points); compounding error + support-clamped payoff are the counterweights.
# --- v8 header follows ---
# v7 -> v8: auxiliary-loss prune (principle: every aux term is a patch hiding a design flaw).
# REMOVED: W_hat shell-negatives (falsified in v4, redundant since the v6 support clamp) and
# the V(g,g)=0.99 self-anchor (root cause of the v5 triviality exploit; unnecessary — the
# lambda-return recursion uses the literal constant 1 at arrival, so targets never depend on
# V(g,g)). KEPT with justification: V cross-trajectory negatives (not a regularizer — V's
# only off-manifold signal; the v6 band penalty can only reject fantasy because of them) and
# the v7 V(f(z,a),g) consistency term (trains V on its actual query distribution).
# The LeJEPA WM invariant is unchanged and now total: encoder+dynamics receive prediction
# MSE + SIGReg and NOTHING else; reward heads read detached z; all consumers detach.
# --- v7 header follows ---
# v6 -> v7: closes the identified signal gap where the policy consumes V. V trains only on
# encoder outputs, but the policy differentiates V exclusively at dynamics outputs
# f(z, pi(z,g)) — every policy gradient samples V out-of-distribution, in a direction pi
# itself controls (a slow-motion adversarial probe). v7 adds a consistency term to the
# critic loss: V(f(z_t, a_t), g) regressed toward the SAME lambda-return target as
# V(z_{t+1}, g) (arrival -> 1), weight vf_consistency_coef. Replay actions are policy+noise,
# so the training-query distribution tracks the policy's query distribution. f is traversed
# but not updated (backward inputs= critic params only).
# --- v6 header follows ---
# v5 -> v6: v5 fixed the world model (wm_pred 0.455 -> 0.012 and converging; SIGReg N-pinning
# + raw-obs consistent normalization + AdamW/LayerNorm) and produced a diverse, non-fantasy,
# state-dependent proposer — but the pendulum swung from fantasy to TRIVIALITY, measured at
# 1.7M: proposer_v = 0.96 (chosen goals ~2 steps away; gamma^d = 0.96 => d ~ 2). The
# multiplicative reachability gate over-rewards V, and it found the self-anchor exploit:
# V(g,g) is trained to 0.99, so g ~ z guarantees a near-perfect score. Held 16 steps, such
# goals go stale immediately (rollout_goal_dv = -0.05/step): the value term pulled the policy
# BACK toward where it just was while r_hat pushed forward — two balanced gradients in
# opposite directions = the flat return curve (policy_rhat pinned at -0.9).
# v6 pins the aspiration horizon instead of trading it off:
#     score = (clamp(W_hat(g), r_lo, r_hi) - r_lo) - beta_band * (V(z,g) - gamma_g^H)^2
# with H = goal_refresh (the goal-hold horizon; gamma_g^16 ~ 0.72). Triviality (V ~ 0.96)
# and fantasy (V ~ 0.08) are BOTH penalized; the rate term picks the best regime reachable
# at that horizon; and the policy's two loss terms align ("move toward a better regime 16
# steps out" and "collect reward now" point the same way for locomotion).
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
    critic_segments: int = 4
    """segments (fresh + sampled replay) used to build hindsight pairs each iteration"""
    hindsight_goals: int = 8
    """hindsight goals sampled per env per segment"""
    cmd_weight: float = 0.15
    """critic-loss weight of commanded-goal pairs (no-arrival ratchet: prior, not evidence)"""
    neg_goal_coef: float = 0.05
    """weight of cross-trajectory negative pairs regressed to distance num_steps"""
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


    # raw-reward backport: separate clip budgets so growing raw-reward residuals
    # cannot throttle the encoder/dyn gradient through a shared global norm
    wm_core_params = list(agent.encoder.parameters()) + list(agent.dyn.parameters())
    wm_head_params = list(agent.reward_head.parameters()) + list(agent.rate_head.parameters())
    wm_params = wm_core_params + wm_head_params
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

    def critic_loss_fn(z, g, target, hind):
        d = agent.dist(z, g)
        # lambda-return targets come from the gamma-space recursion; convert once to
        # step units (arrival G=1 -> 0, floor gamma^T -> T) and train where the policy
        # consumes the head: in distance space, Huber
        d_target = torch.log(target.clamp_min(1e-8)) / float(np.log(args.gamma_goal))
        # evidence weighting: hindsight labels are exact (k = j - t); commanded labels
        # are a no-arrival ratchet that can only say "far" — prior, not evidence. In
        # d-space Huber every sample pulls with unit gradient, so unweighted volume
        # (~75% far-labels) collapses the head onto the majority label (v12 lesson).
        w = hind + args.cmd_weight * (1.0 - hind)
        err = nn.functional.smooth_l1_loss(d, d_target, reduction="none")
        main = (err * w).sum() / w.sum().clamp_min(1.0)
        # cross-trajectory negatives: the head's ONLY off-manifold signal — what "far" means
        neg = nn.functional.smooth_l1_loss(
            agent.dist(z, torch.roll(g, 1, 0)),
            torch.full_like(d, float(args.num_steps)),
        )
        return main + args.neg_goal_coef * neg, main.detach(), (args.gamma_goal**d).mean().detach()

    def proposer_loss_fn(z, r_lo, r_hi):
        g = agent.propose(z)
        # support-clamped rate (fantasy capped at best OBSERVED regime; cap ratchets up
        # with performance) + aspiration band pinning goals at the goal-hold horizon.
        # The band's V is HONEST now (commanded-goal TD): a fantasy proposal gets
        # commanded, fails, is devalued by real experience, and the band then pushes the
        # proposer away from it — the grounding loop v6-v8 lacked.
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
        print(f"[embopt_v13_raw] torch.compile(mode={args.compile_mode!r}, dynamic=False)")

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
    next_done = torch.zeros(E).to(device)
    cur_goal = torch.zeros((E, dz), device=device)
    roll_goals = torch.zeros((T, E, dz), device=device)  # commanded goals of the fresh segment
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

        # ---- world model updates ---------------------------------------------------------
        # flat sampler over all stored transitions; pair validity masks reset boundaries
        pair_valid_all = (buf_done[:F, 1 : T + 1] == 0).float()  # (F,T,E)
        n_flat = F * T * E
        wm_stats = []
        for _ in range(args.wm_updates):
            flat = torch.randint(0, n_flat, (mb,), device=device)
            f_i = flat // (T * E)
            t_i = (flat // E) % T
            e_i = flat % E
            o = buf_obs[f_i, t_i, e_i]
            o2 = buf_obs[f_i, t_i + 1, e_i]
            a = buf_act[f_i, t_i, e_i]
            r = buf_rew[f_i, t_i, e_i]
            rate = buf_rate[f_i, t_i, e_i]
            pv = pair_valid_all[f_i, t_i, e_i]
            mark_step()
            loss, pl, sl, rl, wl = wm_loss_fn(o, a, o2, r, rate, pv)
            wm_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(wm_core_params, args.max_grad_norm)
            nn.utils.clip_grad_norm_(wm_head_params, args.max_grad_norm)
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
            # commanded-goal columns: the goals the policy ACTUALLY pursued during the
            # stored rollout (held goal at each refresh-window start), fed to the SAME
            # lambda-return loss with NO arrival event (j = T+1 sentinel never fires):
            # targets bootstrap from segment-end V and hit the floor at episode cuts, so
            # goals the policy fails to approach are devalued by real experience
            W = T // args.goal_refresh
            w_idx = torch.arange(0, T, args.goal_refresh, device=device)
            goals_cmd = buf_goal[seg][:, w_idx].permute(0, 2, 1, 3)  # (S,E,W,dz)
            j_cmd = torch.full((S, E, W), T + 1, device=device, dtype=j.dtype)
            epid_j_cmd = epid[:, w_idx].permute(0, 2, 1)  # (S,E,W): episode at window start
            goals = torch.cat([goals, goals_cmd], dim=2)
            j = torch.cat([j, j_cmd], dim=2)
            epid_j = torch.cat([epid_j, epid_j_cmd], dim=2)
            M = M + W  # all downstream shapes/decoding use the widened column count
            # V(z_t, g) for all t
            z_all = Z[:, :, :, None, :].expand(S, T + 1, E, M, dz).reshape(-1, dz)
            g_all = goals[:, None].expand(S, T + 1, E, M, dz).reshape(-1, dz)
            Vs = agent.value(z_all, g_all).reshape(S, T + 1, E, M)
            targets = torch.zeros((S, T, E, M), device=device)
            # bootstrap init from segment-end V: reachable only by no-arrival (commanded)
            # columns — hindsight targets in their valid region t < j are cut off from it
            # by the arrival reset at t = j-1
            Gacc = Vs[:, T].clone()
            ones = torch.ones((S, E, M), device=device)
            floor = torch.full_like(ones, args.gamma_goal**args.num_steps)
            dones_seg = buf_done[seg]  # (S,T+1,E)
            lam = args.td_lambda
            for t in range(T - 1, -1, -1):
                at_goal = j == (t + 1)
                # episode boundary at t+1: never bootstrap across a reset — an episode
                # that ends without arrival resolves to the unreached floor (affects only
                # commanded columns; valid hindsight pairs never straddle a cut)
                cut = (dones_seg[:, t + 1] > 0).unsqueeze(-1)
                Vnext = torch.where(cut, floor, torch.where(at_goal, ones, Vs[:, t + 1]))
                Gnext = torch.where(cut, floor, torch.where(at_goal, ones, Gacc))
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
            return (
                z_pairs[ps, pt, pe],
                goals[ps, pe, pm],
                targets[ps, pt, pe, pm],
                (pm < args.hindsight_goals).float(),
            )

        # ---- critic updates --------------------------------------------------------------
        critic_stats = []
        for _ in range(args.critic_updates):
            zb, gb, tb, hb = sample_pairs(mb)
            mark_step()
            loss, main, vmean = critic_loss_fn(zb, gb, tb, hb)
            critic_opt.zero_grad(set_to_none=True)
            loss.backward(inputs=critic_params)
            nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
            critic_opt.step()
            critic_stats.append((main.item(), vmean.item()))

        # ---- proposer updates: ascend the frozen banded score surface along the shell ----
        prop_stats = []
        for _ in range(args.proposer_updates):
            flat = torch.randint(0, n_flat, (mb,), device=device)
            with torch.no_grad():
                zb = agent.encode(buf_obs[flat // (T * E), (flat // E) % T, flat % E])
                mb_rates = buf_rate[flat // (T * E), (flat // E) % T, flat % E]
                r_lo, r_hi = mb_rates.min(), mb_rates.max()
            mark_step()
            loss, wmean, vmean = proposer_loss_fn(zb, r_lo, r_hi)
            proposer_opt.zero_grad(set_to_none=True)
            loss.backward(inputs=proposer_params)
            nn.utils.clip_grad_norm_(proposer_params, args.max_grad_norm)
            proposer_opt.step()
            prop_stats.append((wmean.item(), vmean.item()))

        # ---- policy updates (frozen f, r_hat, V) -----------------------------------------
        n_prop = int(mb * args.policy_goal_mix)
        pol_stats = []
        for _ in range(args.policy_updates):
            zh, gh, _, _ = sample_pairs(mb - n_prop)
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
            # LeJEPA health: effective rank of the embedding covariance (participation
            # ratio, max = dz) — z_sqnorm alone cannot see dimensional collapse
            z_flat = Z[0, :T].reshape(-1, dz)
            z_cent = z_flat - z_flat.mean(0)
            eig = torch.linalg.eigvalsh(z_cent.T @ z_cent / (z_flat.shape[0] - 1))
            z_eff_rank = (eig.sum().square() / eig.square().sum().clamp_min(1e-12)).item()
            z_top_share = (eig[-1] / eig.sum().clamp_min(1e-12)).item()
            # no-change baseline for wm_pred: per-dim Var(z' - z); if wm_pred is not well
            # below this, the dynamics has learned nothing beyond identity
            dstep = Z[0, 1:] - Z[0, :-1]
            dmask = (buf_done[fresh_slot, 1 : T + 1] == 0).float().unsqueeze(-1)
            wm_delta_var = ((dstep.square() * dmask).sum() / (dmask.sum() * dz).clamp_min(1.0)).item()
            # imagined vs realized H-step unroll under EXECUTED actions: the substrate of
            # the policy's exploitation gap (per-dim MSE; compare wm_pred and wm_delta_var)
            H = args.imag_probe_h
            zc = Z[0, : T - H]
            for i in range(H):
                zc = agent.forward_dyn(zc, buf_act[fresh_slot, i : i + T - H])
            imag_m = (epid[0, H:T] == epid[0, : T - H]).float()
            imag_err = (((zc - Z[0, H:T]).square().mean(-1) * imag_m).sum() / imag_m.sum().clamp_min(1.0)).item()
            zb, gb, _, _ = sample_pairs(mb)
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
            w_real = agent.what(zb)
            what_real_mean = w_real.mean().item()
            what_real_max = w_real.max().item()
            a_prop = agent.act(zb, gp_diag)
            a_prop_sh = agent.act(zb, gp_diag[torch.randperm(mb, device=device)])
            proposed_goal_sens = (a_prop - a_prop_sh).square().mean().item()
            # "test them well" (1): distance calibration — hindsight pairs have EXACT
            # known step-distance k = j - t, so the head can be scored directly
            pick_c = valid_idx[torch.randint(0, valid_idx.numel(), (4096,), device=device)]
            ps_c = pick_c // (T * E * M)
            pt_c = (pick_c // (E * M)) % T
            pe_c = (pick_c // M) % E
            pm_c = pick_c % M
            hmask = (pm_c < args.hindsight_goals).float()
            k_true = (j[ps_c, pe_c, pm_c] - pt_c).float()
            d_hat_c = agent.dist(z_pairs[ps_c, pt_c, pe_c], goals[ps_c, pe_c, pm_c])
            denom_c = hmask.sum().clamp_min(1.0)
            dist_calib_bias = (((d_hat_c - k_true) * hmask).sum() / denom_c).item()
            dist_calib_abs = (((d_hat_c - k_true).abs() * hmask).sum() / denom_c).item()
            # "test them well" (2): promise vs delivery — the rate the proposer promised
            # at each held goal vs the rate the policy actually realized over the hold
            goal_promise = agent.what(roll_goals[w_idx]).mean().item()
            goal_delivery = (buf_rate[fresh_slot][w_idx]).mean().item()
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
        writer.add_scalar("losses/policy", po_m[0], global_step)
        writer.add_scalar("diagnostics/critic_v_mean", cr_m[1], global_step)
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
        sps = int(global_step / (time.time() - start_time))
        print(f"iter={iteration} SPS: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)

    envs.close()
    writer.close()
