# Embedding Optimization v21 (embopt_v21) — family: cleanrl/embedding-optimization/ (see FAMILY.md)
#
# v20 -> v21: PERSISTENT EXPERIENCE-ANCHORED GOALS (a-priori redesign, user
# direction via the human-goal analogy: "most joy per unit time = become a
# doctor; the goal gets more specific as I learn; I think about my goal often
# and the best next step; no contrasting, no noise — the exact best step").
# Mapped to mechanism:
#   (1) GOAL = argmax of the rate map over REAL destinations. Candidates are the
#       top measured-rate states in recent replay; the winner by predicted rate
#       what(z) becomes the goal. No proposer network, no shell projection, no
#       delivery head: the goal evaluator IS the densely-grounded state-rate map,
#       queried on the real manifold where it trains. (Every prior version chose
#       goals the way no human ever has: abstract off-manifold points invented
#       fresh each step, credited by association — v19's spurious attribution.)
#   (2) PERSISTENT REFERENT, REFINING REPRESENTATION. The goal is stored as its
#       raw OBSERVATION and re-encoded every iteration: the referent stays fixed
#       while its latent representation sharpens with the world model ("doctor ->
#       anesthesiologist"). Also kills stale-latent-goal drift (v15 lesson) at
#       the root. Switching is HYSTERETIC: a challenger must beat the incumbent's
#       predicted rate by goal_switch_margin — you don't change careers for a 1%
#       raise, and the policy gets a stationary pursuit target.
#   (3) PER-STEP RE-EVALUATION OF PURSUIT, NOT OF THE GOAL. v16-v20 re-invented
#       the goal every step, so each progress label answered a DIFFERENT pursuit
#       question and q(z,a,g) learned none of them (rollout_goal_prog ~0.01).
#       With one persistent goal every transition labels the SAME pursuit
#       problem; label coherence is the fix, not label volume.
#   (4) Pursuit machinery unchanged from v20: deterministic policy, one scalar,
#       L_pi = -[reward_coef*rhat + rate_coef*qrate + value_coef*q(z,a,g)],
#       all three heads measured-outcome regressions, goal term ON (vc=1) —
#       goal-based policy learning is the point of the family.
# Decisive signatures: rollout_goal_prog should move off ~0 (real following at
# last), goal_promise vs goal_delivery converging, goal_switched rare after
# warm-up, and the v20_vc05 job (old shell-goal machinery, same chassis) is the
# direct A/B control for this goal redesign.
# --- v20 header follows ---
# v19 -> v20: LONG-HORIZON REWARD CREDIT BY REGRESSION; GOAL TERM OFF BY DEFAULT.
# v19's coefficient sweep was a clean causal experiment on the goal channel:
#   vc0 (goal off)  — completely STABLE, monotone -377 -> -171, zero swings;
#   vc1             — repeated spike-crash cycles (-123 -> -482, -45 -> -500);
#   vc4             — dormant, a -1068 excursion, +60 @2.2M, instant crash to -600.
# Instability is monotone in value_coef with stationary units — the goal channel
# itself is the destabilizer. Mechanism: SPURIOUS ATTRIBUTION. gwhat(g) <- delivered
# rate assumes the goal caused the rate, but measured rollout_goal_prog ~ 0.001-0.01:
# the policy achieves essentially ZERO real progress toward commanded goals. When the
# policy happens to run well, the goals commanded during success absorb the credit,
# the proposer chases them, the goals shift, and the goal-dominated action gradient
# (0.7-3.2 vs reward's 0.1-0.7) drags the behavior somewhere new — destroying the
# gait it took credit for. Success destabilizes the goal channel through the
# delivery map even with stationary units.
# vc0's stability exposes the complementary gap: r_hat(z,a) is IMMEDIATE reward, so
# pathwise ascent on it is myopic — it can lean forward but cannot learn momentum-
# building that pays off tens of steps later; hence the hard -175 plateau. The
# long-horizon credit was supposed to come from goals, which demonstrably don't steer.
# v20 supplies long-horizon credit in the family's own idiom — measured-outcome
# regression, no bootstrap, no TD, no counterfactuals:
#   qrate(z,a) <- buf_rate[t]  (the MEASURED mean reward over the next rate_window
#   steps of the executed trajectory — the same target what(z) regresses, now
#   action-conditioned so dq/da exists; the exact rhat pattern at horizon 16)
#     L_pi = -[ reward_coef * r_hat(z,a) + rate_coef * qrate(z,a)
#               + value_coef * q(z,a,g) ]        a = pi(z,g)
# with value_coef DEFAULT 0 (per the causal result above). The goal machinery
# (proposer, delivery map, qprog) keeps training for telemetry and future re-entry;
# r_hat stays in the objective because its immediate slope is the densest signal,
# while qrate tilts the ascent toward action patterns whose measured 16-step
# aftermath was high-rate.
# --- v19 header follows ---
# v18 -> v19: STATIONARY UNITS + REGRESSION-ONLY GOAL CREDIT. Fine-grained curves
# showed v17/v17_vc05/v18 all LEARNED to run and were then destroyed: v17 hit -41
# @96k (crashed by 144k), vc05 hit +327 @208k (crashed by 240k); every collapse
# landed on a ~-200 plateau that never recovered. Two structural causes, both fixed:
#   (1) NONSTATIONARY UNITS (axed: obs RMS + reward scale). Both running scales are
#       SUCCESS-COUPLED: the moment returns spike, buf_rew.std() grows (reward_scale
#       0.7 -> ~0.15), silently deflating every normalized-reward target — the
#       policy's reward gradient shrinks exactly when it finds something worth
#       keeping, while the goal term keeps fixed latent units and takes over. The
#       obs RMS shifts the encoder's input frame at the same moment (velocity stats
#       jump when running starts), invalidating stored latent goals and every
#       learned head. v19: RAW obs into the encoder (its LayerNorm absorbs scale),
#       RAW rewards/rates as regression targets. Success no longer rescales the
#       objective or moves the frame.
#   (2) MODEL-ERROR CROSS-TERM EXPLOIT (axed: dynamics in the policy loss). The
#       goal term read progress through f: ldist(z,g) - ldist(f(z,a),g). Progress
#       is LINEAR in prediction error while wm_pred is QUADRATIC: per-dim err 0.016
#       at dist^2 ~ 2 admits ~2*sqrt(0.016*2) ~ 0.36/dim of fake progress once
#       gradient ascent aligns f's error vector with (g - z'). Observed: policy_prog
#       claimed 0.46-0.60/step vs 0.03-0.06 measured (11-15x over-claim), a drag of
#       equal gradient magnitude to r_hat. v19: the goal term is a REGRESSION
#       Q-head, q(z,a,g) <- measured progress-to-go of the executed transition (the
#       v16 h-pool, now action-conditioned; the exact rhat pattern). The policy
#       ascends dq/da; q's errors at the policy's actions are corrected next
#       iteration because those actions (plus the expl_noise ball) are what gets
#       executed and labeled. No dynamics, no imagination, no DDPG machinery in the
#       policy loss — measured-outcome regression end to end.
# Also axed: v18's goal noise (v18 == v17 within noise, and with per-step
# re-proposal the delivery map already trains exactly where the proposer queries
# it — commanded goals ARE fresh proposer outputs).
# Method (v19 core): deterministic policy, one differentiable scalar
#     L_pi = -[ reward_coef * r_hat(z, pi(z,g)) + value_coef * q(z, pi(z,g), g) ]
# with r_hat and q both frozen regression surfaces of measured outcomes; gradients
# reach the policy only through dr/da and dq/da.
# --- v18 header follows ---
# v17 -> v18: ground the goal channel's DERIVATIVES, not just its values (review
# findings on v16/v17, submitted concurrently with v17 as a probe):
#   (1) GOAL EXPLORATION NOISE. v17 trains h and the delivery map only at
#       (z, propose(z)) — a 1-D manifold of the proposer's own outputs; every
#       gradient the proposer ascends w.r.t. g points in directions no data varies
#       (DDPG's critic-without-action-noise pathology, on the goal channel). v18
#       perturbs the commanded goal on the shell each step; the PERTURBED goal is
#       what the policy pursues and what gets stored, so every label remains a
#       measured outcome of the goal actually commanded. This also grounds the
#       policy's off-manifold h(f(z,a), g) query.
#   (2) SEPARATE DELIVERY HEAD. v17's wg_loss made ONE rate_head fit two colliding
#       regressions (state->rate and goal->delivered rate; under SIGReg both live at
#       ||x|| ~ 8), injecting state-label noise exactly where the proposer ascends.
#       v18 adds goal_rate_head: gwhat(g) <- rate delivered around g's pursuit, and
#       the proposer score is gwhat ONLY — a pure "goals like this empirically
#       delivered X" surface, hill-climbed under goal noise. what(z) stays a clean
#       state-rate map (telemetry + regime reference).
# --- v17 header follows ---
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
    """learning rate for the critic and policy optimizers"""
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
    rate_coef: float = 1.0
    """weight of the qrate (measured forward reward-rate) term in the policy loss"""
    value_coef: float = 1.0
    """weight of the goal-progress term in the policy loss (ON: goal-based policy
    learning is the family's point; v21 fixes the goal mechanism instead)"""
    goal_candidates: int = 256
    """how many top measured-rate states from recent replay are scored by the rate
    map when nominating a goal challenger each iteration"""
    goal_switch_margin: float = 0.1
    """raw-rate margin a challenger must beat the incumbent goal's predicted rate by
    before the goal switches (hysteresis: persistent referent, no goal fashion)"""
    imag_probe_h: int = 8
    """horizon for the imag_err_h diagnostic (telemetry only; no training unroll)"""
    expl_noise: float = 0.2
    """std of Gaussian exploration noise added to the deterministic action"""
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
    # RAW obs and RAW rewards everywhere — no running normalization of any kind.
    # v17/v18 collapse post-mortem: running scales are success-coupled — the moment
    # the policy starts earning reward, the units of its own objective shrink and
    # the encoder's input frame shifts, destroying exactly the behavior worth keeping
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
        # action-conditioned forward-rate head: measured mean reward over the next
        # rate_window steps as a function of the action taken NOW — the long-horizon
        # credit channel (rhat pattern at horizon rate_window; no bootstrap)
        self.qrate_head = mlp(dz + da, dh, 1)
        # action-conditioned progress head q(z,a,g) — the goal-channel twin of
        # reward_head: same input pattern, same regression-on-measured-outcomes role
        self.critic = mlp(3 * dz + da, dh, 1)
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

    def qprog(self, z, a, g):
        # predicted MSE-progress-to-go toward g over the current hold window, as a
        # function of the action taken NOW — a regression surface of measured
        # outcomes (exact rhat pattern). The policy ascends dq/da; no dynamics sit
        # between the action and the goal credit, so there is no model-error
        # cross-term to exploit
        return self.critic(torch.cat([z, g, g - z, a], -1)).squeeze(-1)

    def act(self, z, g):
        return torch.tanh(self.policy(torch.cat([z, g, g - z], -1)))

    def rhat(self, z, a):
        return self.reward_head(torch.cat([z, a], -1)).squeeze(-1)

    def what(self, z):
        return self.rate_head(z).squeeze(-1)

    def qrate(self, z, a):
        return self.qrate_head(torch.cat([z, a], -1)).squeeze(-1)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.rate_window <= args.num_steps, "rate_window must not exceed num_steps"
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

    # core (encoder+dyn) and reward-unit heads get SEPARATE clip budgets: raw-reward
    # residuals grow with returns, and a shared global norm would throttle the
    # encoder's LeJEPA gradient exactly when the policy starts succeeding — the same
    # success-coupling v19 exists to kill, relocated into the clip (review finding)
    wm_core_params = list(agent.encoder.parameters()) + list(agent.dyn.parameters())
    wm_head_params = (
        list(agent.reward_head.parameters())
        + list(agent.rate_head.parameters())
        + list(agent.qrate_head.parameters())
    )
    wm_params = wm_core_params + wm_head_params
    critic_params = list(agent.critic.parameters())
    policy_params = list(agent.policy.parameters())
    wm_opt = optim.AdamW(wm_params, lr=args.wm_lr, weight_decay=args.wm_weight_decay, eps=1e-5)
    critic_opt = optim.Adam(critic_params, lr=args.learning_rate, eps=1e-5)
    policy_opt = optim.Adam(policy_params, lr=args.learning_rate, eps=1e-5)
    opt_base_lrs = ((wm_opt, args.wm_lr), (critic_opt, args.learning_rate), (policy_opt, args.learning_rate))

    # ---- loss / act functions (compiled) -------------------------------------------------
    def rollout_forward(obs, goal):
        z = agent.encode(obs)
        return agent.act(z, goal)

    def wm_loss_fn(o, a, o2, r, rate, rv, pair_valid):
        z = agent.encode(o)
        z2 = agent.encode(o2)
        pred = agent.forward_dyn(z, a)
        denom = pair_valid.sum().clamp_min(1.0)
        pred_loss = (pair_valid * (pred - z2).square().mean(-1)).sum() / denom
        sig_loss = sigreg(z)
        zd = z.detach()
        r_loss = (agent.rhat(zd, a) - r).square().mean()
        w_loss = (agent.what(zd) - rate).square().mean()
        # long-horizon credit: measured forward rate of the EXECUTED action.
        # rv masks segment-tail steps whose forward window is truncated below
        # rate_window (short-horizon labels are noisier and this head feeds the
        # policy gradient directly, unlike the what telemetry surface)
        qr_denom = rv.sum().clamp_min(1.0)
        qr_loss = (rv * (agent.qrate(zd, a) - rate).square()).sum() / qr_denom
        loss = pred_loss + args.sigreg_coef * sig_loss + r_loss + w_loss + qr_loss
        return (
            loss,
            pred_loss.detach(),
            sig_loss.detach(),
            r_loss.detach(),
            w_loss.detach(),
            qr_loss.detach(),
        )

    def critic_loss_fn(z, a, g, target, m):
        # measured-outcome regression: q(z,a,g) vs realized progress toward the goal
        # ACTUALLY pursued from z under the action ACTUALLY taken (MSE decrease from
        # z to its window's end). No bootstrap, no relabeling, no dynamics — nothing
        # to collapse into or exploit. m masks pairs whose remaining window
        # straddles an episode cut.
        pred = agent.qprog(z, a, g)
        denom = m.sum().clamp_min(1.0)
        loss = (m * nn.functional.smooth_l1_loss(pred, target, reduction="none")).sum() / denom
        return loss, ((pred * m).sum() / denom).detach()

    def policy_loss_fn(z, g):
        # one differentiable scalar, no model step: every term is a frozen regression
        # surface of measured outcomes, so the ascent direction stays grounded on
        # the executed-action ball (expl_noise) that keeps labeling them. rhat gives
        # the immediate slope; qrate tilts toward action patterns whose measured
        # 16-step aftermath was high-rate (the long-horizon channel)
        a = agent.act(z, g)
        r = agent.rhat(z, a)
        qr = agent.qrate(z, a)
        qp = agent.qprog(z, a, g)
        obj = args.reward_coef * r + args.rate_coef * qr + args.value_coef * qp
        return -obj.mean(), r.mean().detach(), qr.mean().detach(), qp.mean().detach()

    if args.compile:
        rollout_forward = torch.compile(rollout_forward, mode=args.compile_mode, dynamic=False)
        wm_loss_fn = torch.compile(wm_loss_fn, mode=args.compile_mode, dynamic=False)
        critic_loss_fn = torch.compile(critic_loss_fn, mode=args.compile_mode, dynamic=False)
        policy_loss_fn = torch.compile(policy_loss_fn, mode=args.compile_mode, dynamic=False)
        print(f"[embopt_v21] torch.compile(mode={args.compile_mode!r}, dynamic=False)")

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
    # goal ANCHOR: a real observation. Its latent representation is re-derived with
    # the CURRENT encoder every iteration (fixed referent, refining representation);
    # the hysteresis rule below is the only thing that ever moves the referent.
    goal_obs = next_obs[0].clone()
    mb = args.minibatch_size

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            for opt, base_lr in opt_base_lrs:
                opt.param_groups[0]["lr"] = frac * base_lr

        # ---- rollout --------------------------------------------------------------------
        s = buf_ptr
        with torch.no_grad():
            # one persistent goal for the whole segment, in the current latent frame
            goal_roll = agent.encode(goal_obs.unsqueeze(0)).squeeze(0).expand(E, dz).contiguous()
        for step in range(T):
            buf_obs[s, step] = next_obs
            buf_done[s, step] = next_done
            with torch.no_grad():
                mark_step()
                action = rollout_forward(next_obs, goal_roll).clone()
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
        buf_goal[s] = goal_roll

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
            rv = (t_i <= T - args.rate_window).float()  # full-window rate labels only
            pv = pair_valid_all[f_i, t_i, e_i]
            mark_step()
            loss, pl, sl, rl, wl, qrl = wm_loss_fn(o, a, o2, r, rate, rv, pv)
            wm_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(wm_core_params, args.max_grad_norm)
            nn.utils.clip_grad_norm_(wm_head_params, args.max_grad_norm)
            wm_opt.step()
            wm_stats.append((pl.item(), sl.item(), rl.item(), wl.item(), qrl.item()))

        # ---- goal update: experience-anchored argmax of the rate map ---------------------
        # candidates = the top measured-rate states in recent replay (facts); the one
        # with the best PREDICTED rate challenges the incumbent goal, which yields
        # only if beaten by goal_switch_margin (hysteresis). The anchor is stored as
        # its raw observation and re-encoded per iteration: fixed referent, refining
        # representation as the world model improves.
        with torch.no_grad():
            o_rec = buf_obs[recent_slots][:, :T].reshape(-1, do)
            # mask truncated-window rates before topk (review): a MAX over
            # heteroscedastic estimates is biased toward the noisy short-window
            # segment-tail entries even though each is individually unbiased —
            # selection needs the mask the qrate regression already has
            valid_t = torch.arange(T, device=device) <= T - args.rate_window
            rates_rec = buf_rate[recent_slots].masked_fill(~valid_t.view(1, T, 1), float("-inf")).reshape(-1)
            cand = rates_rec.topk(min(args.goal_candidates, n_recent)).indices
            z_c = agent.encode(o_rec[cand])
            w_c = agent.what(z_c)
            j = w_c.argmax()
            goal_what = agent.what(agent.encode(goal_obs.unsqueeze(0))).squeeze(0)
            # fantasy guard (review): once the incumbent's segment ages out of the
            # ring, nothing constrains what() at the anchor except extrapolation —
            # an incumbent may not defend with a promise above the best rate
            # actually MEASURED in recent replay
            goal_what = torch.minimum(goal_what, rates_rec.max())
            goal_switched = 0.0
            if (w_c[j] - goal_what).item() > args.goal_switch_margin:
                goal_obs = o_rec[cand[j]].clone()
                goal_what = w_c[j]
                goal_switched = 1.0

        # ---- progress-head training pool: measured one-step pursuit outcomes ------------
        # for EVERY transition of the recent segments: (z_t, a_t, goal commanded at t)
        # -> realized decrease in embed MSE to that goal over the step. With one
        # persistent goal, every pair labels the SAME pursuit problem. Encodings use
        # the CURRENT encoder; pairs straddling a cut are masked.
        with torch.no_grad():
            Zs = agent.encode(buf_obs[recent_slots].reshape(-1, do)).reshape(K, T + 1, E, dz)
            wend_idx = torch.arange(1, T + 1, device=device)  # one-step pursuit horizon
            g_pool = buf_goal[recent_slots]  # (K,T,E,dz)
            epid_seg = buf_done[recent_slots].cumsum(dim=1)  # (K,T+1,E)
            m_pool = (epid_seg[:, :T] == epid_seg[:, wend_idx]).float()  # (K,T,E)
            tgt_pool = agent.ldist(Zs[:, :T], g_pool) - agent.ldist(Zs[:, wend_idx], g_pool)  # (K,T,E)
            z_f = Zs[:, :T].reshape(-1, dz)
            a_f = buf_act[recent_slots].reshape(-1, da)  # executed action of each pair
            g_f = g_pool.reshape(-1, dz)
            tgt_f = tgt_pool.reshape(-1)
            m_f = m_pool.reshape(-1)
            n_pool = K * T * E

        # ---- progress-head updates -------------------------------------------------------
        critic_stats = []
        for _ in range(args.critic_updates):
            idx = torch.randint(0, n_pool, (mb,), device=device)
            mark_step()
            loss, pmean = critic_loss_fn(z_f[idx], a_f[idx], g_f[idx], tgt_f[idx], m_f[idx])
            critic_opt.zero_grad(set_to_none=True)
            loss.backward(inputs=critic_params)
            nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
            critic_opt.step()
            critic_stats.append((loss.item(), pmean.item()))

        # ---- policy updates (frozen r_hat, qrate, q) -------------------------------------
        # goals = stored commanded goals: past encodings of the persistent anchor,
        # near-identical to the current one — the pursuit problem the pool labels
        pol_stats = []
        for _ in range(args.policy_updates):
            flat = torch.randint(0, n_recent, (mb,), device=device)
            f_i = recent_slots[flat // (T * E)]
            t_i = (flat // E) % T
            e_i = flat % E
            with torch.no_grad():
                zb = agent.encode(buf_obs[f_i, t_i, e_i])
            gb = buf_goal[f_i, t_i, e_i]
            mark_step()
            loss, rmean, qrmean, pmean = policy_loss_fn(zb, gb)
            policy_opt.zero_grad(set_to_none=True)
            loss.backward(inputs=policy_params)
            nn.utils.clip_grad_norm_(policy_params, args.max_grad_norm)
            policy_opt.step()
            pol_stats.append((loss.item(), rmean.item(), qrmean.item(), pmean.item()))

        # ---- diagnostics: is the goal channel alive? (family-mandated) -------------------
        with torch.no_grad():
            # fresh-segment encodings for all telemetry
            Zf = agent.encode(buf_obs[fresh_slot].reshape(-1, do)).reshape(T + 1, E, dz)
            epid_f = buf_done[fresh_slot].cumsum(dim=0)  # (T+1,E)
            # goal-approach telemetry on the fresh segment: per-step realized progress
            # toward the commanded goal (positive = approaching), and mean distance
            Dg = agent.ldist(Zf[:T], buf_goal[fresh_slot])  # (T,E)
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
            pred_fresh = agent.qprog(
                Zs[0, :T].reshape(-1, dz), buf_act[fresh_slot].reshape(-1, da), g_pool[0].reshape(-1, dz)
            ).reshape(T, E)
            mf0 = m_pool[0]
            denf = mf0.sum().clamp_min(1.0)
            prog_calib_bias = (((pred_fresh - tgt_pool[0]) * mf0).sum() / denf).item()
            prog_calib_abs = (((pred_fresh - tgt_pool[0]).abs() * mf0).sum() / denf).item()
            # goal-channel aliveness on the progress-head/policy query distribution
            flat = torch.randint(0, n_recent, (mb,), device=device)
            f_i = recent_slots[flat // (T * E)]
            t_i = (flat // E) % T
            e_i = flat % E
            zb = agent.encode(buf_obs[f_i, t_i, e_i])
            gb = buf_goal[f_i, t_i, e_i]
            a_matched = agent.act(zb, gb)
            # commanded goal vs null goal (g = current z): the diagnostic that failed first
            # in every dead pan-goal-solver version — must lift off ~0
            a_null = agent.act(zb, zb)
            goal_removal = (a_matched - a_null).square().mean().item()
            w_real = agent.what(zb)
            what_real_mean = w_real.mean().item()
            what_real_max = w_real.max().item()
            # promise vs delivery — the rate the goal's anchor promises (rate map at the
            # current goal encoding) vs the rate the policy actually realized this
            # segment. On the (rare) switch iterations promise reflects the NEW goal
            # while delivery reflects the old one — off by one iteration by design.
            goal_promise = goal_what.item()
            goal_delivery = buf_rate[fresh_slot].mean().item()
        # measured (not assumed) balance of the three policy-loss channels at the
        # action — UNSCALED grad norms (like-for-like; the coefficients are known
        # constants, and a zero value_coef must not silence the goal-channel curve)
        ad = agent.act(zb, gb).detach()
        a1 = ad.clone().requires_grad_(True)
        grad_r = torch.autograd.grad(agent.rhat(zb, a1).sum(), a1)[0]
        a2 = ad.clone().requires_grad_(True)
        grad_v = torch.autograd.grad(agent.qprog(zb, a2, gb).sum(), a2)[0]
        a3 = ad.clone().requires_grad_(True)
        grad_qr = torch.autograd.grad(agent.qrate(zb, a3).sum(), a3)[0]
        action_grad_rhat = grad_r.norm(dim=-1).mean().item()
        action_grad_value = grad_v.norm(dim=-1).mean().item()
        action_grad_qrate = grad_qr.norm(dim=-1).mean().item()

        wm_m = np.mean(wm_stats, axis=0)
        cr_m = np.mean(critic_stats, axis=0)
        po_m = np.mean(pol_stats, axis=0)
        writer.add_scalar("charts/learning_rate", wm_opt.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/wm_pred", wm_m[0], global_step)
        writer.add_scalar("losses/wm_sigreg", wm_m[1], global_step)
        writer.add_scalar("losses/reward_mse", wm_m[2], global_step)
        writer.add_scalar("losses/rate_mse", wm_m[3], global_step)
        writer.add_scalar("losses/qrate_mse", wm_m[4], global_step)
        writer.add_scalar("losses/critic", cr_m[0], global_step)
        writer.add_scalar("losses/policy", po_m[0], global_step)
        writer.add_scalar("diagnostics/prog_pred_mean", cr_m[1], global_step)
        writer.add_scalar("diagnostics/goal_switched", goal_switched, global_step)
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
        writer.add_scalar("diagnostics/policy_qrate", po_m[2], global_step)
        writer.add_scalar("diagnostics/policy_prog", po_m[3], global_step)
        writer.add_scalar("diagnostics/goal_removal_action_mse", goal_removal, global_step)
        writer.add_scalar("diagnostics/action_grad_rhat", action_grad_rhat, global_step)
        writer.add_scalar("diagnostics/action_grad_value", action_grad_value, global_step)
        writer.add_scalar("diagnostics/action_grad_qrate", action_grad_qrate, global_step)
        sps = int(global_step / (time.time() - start_time))
        print(f"iter={iteration} SPS: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)

    envs.close()
    writer.close()
