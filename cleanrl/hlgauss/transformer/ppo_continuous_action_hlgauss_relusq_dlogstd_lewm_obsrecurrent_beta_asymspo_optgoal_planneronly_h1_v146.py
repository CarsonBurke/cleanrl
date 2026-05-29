# v146: planner-owned optimal-goal H=1 MPC.
#
# v145 proved the intended structure works but kept PPO/SPO policy gradients
# active on planner-collected behavior. That mixes two controllers. v146 makes
# the gradient-solver MPC the behavior policy after warmup:
# - GoalSolver imagines g from an optimized WM witness action;
# - ActionSolver executes distance-only LeWM-style control to reach g;
# - PPO/SPO policy-gradient terms are masked once planner controls behavior;
# - the actor is trained only as an amortized warm-start/proposal for the planner.
#
# v145: optimal-goal H=1 MPC with LeWM-style GradientSolver.
#
# This is the intended two-stage controller:
# 1. imagine an optimal next obs-latent goal by optimizing an action through the
#    frozen WM return objective, then set g = WM(z_t, a_goal)[:obs];
# 2. execute a separate GradientSolver that minimizes summed final embedding
#    distance from WM(z_t, a_exec) to stopgrad(g).
#
# Unlike v143, the goal is not a free critic-favored latent near a detached
# anchor. It is always on the one-step WM action manifold. Unlike direct return
# MPC, the executed action is selected by final embedding distance to the
# imagined optimal goal.
#
# v143: obs/action-seeded H=1 goal-embedding solver.
#
# v142 still optimized an action witness (`a_seed`) inside GoalSolver. That
# mixed roles with the actual GradientSolver. v143 removes the optimized
# witness: GoalSolver optimizes only the free next obs-latent target `g`.
# `g` is parameterized as current_obs_latent + delta and initialized from the
# policy-predicted next obs latent, so the current state and action prior seed
# the imagined target while the ActionSolver remains solely responsible for
# finding actions that reach it.
#
# v142: fast H=1 goal-embedding solver.
#
# Same two-stage objective as v141, but reduces planner cost for iteration:
# one candidate, four GoalSolver Adam steps, and four ActionSolver Adam steps.
# This tests whether the goal-embedding mechanism still gives useful behavior
# when it is cheap enough to run at reasonable SPS.
#
# v141: H=1 goal-embedding solver with summed reachability defect.
#
# v140 confirmed the two-stage code path but let free g exploit the critic: the
# GoalSolver used mean latent defect over 512 obs-latent coordinates, so a large
# value increase dominated a visibly unreachable target. v141 keeps the exact
# H=1 GoalSolver -> distance-only ActionSolver structure, but scores
# reachability with summed squared latent defect. Logged action distance remains
# mean MSE for scale readability.
#
# v140: H=1 goal-embedding solver for unknown-goal best-policy control.
#
# This variant separates "imagine the optimal next embedding" from "find the
# action that reaches it." GoalSolver optimizes a free next obs-latent `g` for
# high predicted value, tied to a WM-predicted next latent by a reachability
# defect. ActionSolver then ignores return and minimizes only the LeWM-style
# distance from WM(z_t, a) to stopgrad(g). This tests whether an aspirational
# best-policy target embedding is a better control signal than directly taking
# the return gradient through actions.
#
# Key details:
# - H stays 1; the receding loop supplies the temporal behavior.
# - g contains only obs/core tokens, never reward/continuation outcome tokens.
# - GoalSolver optimizes only obs/core tokens; reward tokens are never free.
# - the executed action is chosen by distance to g, as in GradientSolver/LeWM.
# - planner behavior is not amortized into the actor by default.
#
# Carryovers from v139:
# - planner behavior is not amortized into the actor by default
# - rollout planner init honors inner_solver_init instead of always using a
#   sampled fused-policy action
# - the solver temporarily switches the agent to eval mode, matching reference
#   LeWM evaluation semantics even though current predictor dropout is zero
#
# v138: official-aligned LeWM gradient-solver control.
# after world-model warmup, each real environment action is selected by
# gradient descent through the frozen learned world model. The solver minimizes
# negative predicted return, picks the lowest-cost candidate sequence, and
# executes the first optimized action. The actor is then trained to amortize
# those planner actions; real returns still train the critic and real
# transitions still train the WM.
#
# This is intentionally not max-entropy, residual, SAC-like, or mixed-policy
# control. It is the direct baseline: "does the learned WM gradient solver
# produce useful behavior actions at all?"
#
# v137: multi-start tanh-parameterized inner action solver.
#
# This is the second one-factor ablation. It keeps v136's tanh-bounded solver
# and v135's H=1 default objective, but optimizes several candidate action
# sequences per rollout state and amortizes the lowest-final-cost candidate.
# Sample 0 is the policy initialization; remaining samples are Gaussian
# perturbations in normalized action coordinates.
#
# Defaults are intentionally small (`inner_solver_samples=4`) because compute
# cost scales roughly linearly with S while H remains unchanged.
#
# v136: tanh-parameterized inner action solver.
#
# This is the first one-factor ablation toward a stronger differentiable
# LEWM/MPC planner. It keeps v135's objective, H=1 default, S=1 default,
# PrefPoE losses, and amortization target unchanged. The only change is the
# inner solver variable:
#
#   v135 optimized raw bounded actions and clamped after each Adam step.
#   v136 optimizes unconstrained u with
#       a = action_mid + action_scale * tanh(u)
#
# Hypothesis: removing hard clamp/projection gives smoother action gradients
# near the MuJoCo action bounds and avoids Adam momentum repeatedly pushing
# against clipped coordinates. Runtime should be effectively identical.
#
# v135: replace Beta entropy regularizer with a LINEAR concentration penalty.
#
# Diagnosis driving v135:
#   Paper's non-collapse guarantee (§4 Theorem 2) is for Gaussian PoE. Its
#   entropy regularizer `−α·H(π_pref)` self-corrects because Gaussian entropy
#   gradient `∂H/∂σ = 1/σ` GROWS as σ shrinks (concentration rises). For Beta,
#   the analogous gradient `∂H/∂(α+β) ~ −1/(α+β)` VANISHES as concentration
#   grows — entropy reg is structurally too weak to bound runaway concentration.
#   We observed the predicted pathology across α_entropy ∈ {0.0, 0.5, 0.05}:
#     v132 (α=0.0):  pref → α+β≈500/dim Dirac → collapse @500k → −511 returns
#     v133 (α=0.5):  pref stuck uniform → no signal → +1144 @ 1.1M (v124 ≈ 2100)
#     v134 (α=0.05): same family — entropy reg either too strong or too weak
#                    at every operating point, no clean equilibrium.
#
# v135 fix: linear concentration penalty in place of entropy regularizer.
#   L_pref(v135) = −β₁·E[A_norm·log π_pref] + c_conc · E[(α_pref + β_pref − 2)]
#   The (α+β−2) term is 0 at the uniform Beta(1,1) and grows linearly. Its
#   gradient w.r.t. each pref output is CONSTANT in α+β — does not vanish at
#   high concentration, unlike H(Beta). This restores the self-correcting
#   dynamic that Gaussian-PoE gets from its entropy gradient scaling.
#
#   Defaults: α_entropy=0.0 (disabled — replaced by linear penalty),
#             concentration_penalty=0.01, cap=50 (defensive backstop kept).
#
#   Coefficient choice (c=0.01): with 6 dims, sum penalty when α+β=20/dim is
#     6·(20−2)·0.01 = 1.08, comparable to typical |β₁·A·log π_pref| ≈ 1–3.
#     At cap (α+β=100/dim), penalty = 6·98·0.01 = 5.88 → strong opposing force
#     well before clamp binds.
#
# Earlier history:
#   v132 (α_entropy=0.0):  pref ran to α+β≈500/dim → collapse @500k → −511 return
#   v133 (α_entropy=0.5):  pref stuck uniform → no signal → underperform @ 1M
#   v134 (α_entropy=0.05): launched briefly, killed (same structural issue).
#   v135: linear concentration penalty replaces entropy reg.
#
# Paper recipe (continuous, Gaussian):
#   π_θ      = N(μ_θ, Σ_θ)               main policy head
#   π_pref   = N(μ_pref, Σ_pref)         preference policy head (shared encoder)
#   π_fused ∝ π_θ · π_pref^{λ_pref}      Product-of-Experts (PoE)
#   L_pref   = −β_1·E[A_norm·log π_pref(a|s)] − α·H(π_pref)
#   L_cons   = KL(π_fused ‖ π_pref)      anchors fused toward pref
#   L_total  = L_PPO + w_pref·L_pref + w_cons·L_cons
#   sample, log_prob, entropy: all under π_fused
#
# Beta-PoE closed form (v132 carryover):
#   Beta(α_θ, β_θ) · Beta(α_pref, β_pref)^λ
#       ∝ Beta(α_θ + λ(α_pref−1),  β_θ + λ(β_pref−1))
#   With `α = 1 + softplus(.)` parameterization both α, β ≥ 1, so for any
#   λ ∈ [0, 1] the fused parameters are also ≥ 1 — fused stays a proper
#   unimodal Beta. The cap on pref does not change this: pref α,β are
#   still ≥ 1 since the cap only restricts the softplus output upward.
#
# v134 defaults:
#   lambda_pref=0.5, prefpoe_beta1=1.0, prefpoe_alpha_entropy=0.05,
#   w_pref=1.0, w_cons=0.1, prefpoe_pref_concentration_cap=50.
#
# Side conventions held from v124:
#   - inner solver still initialized from MAIN policy mean (the WM-Jacobian
#     signal is independent of the advantage-guided preference)
#   - amortize loss still regresses MAIN policy mean toward a_star_first
#     (separation of concerns — WM signal on main, advantage signal on pref)
#   - PPO ratio computed under fused (consistent with the paper's PPO
#     integration: "we sample from the fused distribution π_fused(a|s)")
#
# Empirical claim from the paper on HalfCheetah-v4: +321% over PPO baseline
# (1276 → 5375 at 2M steps). Ablation: PoE-alone (no advantage) +237%,
# Full PrefPoE +321%. NOT tested on Hopper-v4 or Walker2d-v4 in the paper.
#
# v124 carryover (kept verbatim):
#
# v124: LeWM-style first-order action solver replaces dream SPO.
#
# Per real rollout state we run an inner Adam loop on a free action variable
# through the (frozen) differentiable WM, minimizing
#     C(a; z_t) = -(r_hat(z_t, a) + gamma * V_hat(pred(z_t, a)))
# The actor probe is updated by MSE regression of its mean toward the
# post-inner-loop optimized action ("amortizer probe"), in addition to the
# existing real-rollout SPO anchor on env reward. WM training, HL-Gauss
# critic, outcome / continuation / successor-flow probes are unchanged
# from v123. Imagined SPO dream batch is removed.
#
# PPO + HL-Gauss with a LeWM-style action-conditioned latent world model.
#
# Key ideas:
# - a linear observation mixer maps the full raw observation vector to 8 recurrent latent tokens
# - an encoder transformer refines the mixed observation tokens directly
# - a separate predictor transformer rolls latent tokens forward from latent/action history
# - a standard relu-squared MLP PPO agent acts on detached mixed-observation latent tokens
# - Xavier/Glorot init on tokenizer and transformer layers
# - LeWM-style next-encoder-latent MSE: pred(z_t, a_t) targets encoder(o_{t+1})
# - 5-step teacher-forced WM training masked across episode boundaries
# - LeWM-style SIGReg regularizes the full encoded latent sequence toward an isotropic Gaussian
# - v59 predicts one state+outcome target embedding per transition:
#   target(obs_{t+1}, reward_t, continuation_t), with reward/continuation
#   entering the predicted embedding rather than detached dynamics heads
# - imagined actor uses asymmetric half-strength SPO on detached world-model latent rollouts
# - imagined critic uses an HL-Gauss value head for Dreamer-style lambda returns
# - v44: dream construction runs the WM in eval mode, termination uses soft continuation
#   for GAE without also masking by sampled terminal, and the unused dynamics value loss is disabled
# - v45: immediate reward readout was action-aware: r_hat = g(z_t, a_t, z_hat_{t+1})
# - v46: agent critic used a scalar ReLU-squared value head to avoid early HL-Gauss
#   support-edge bootstraps poisoning imagined PPO; v54 returns to HL-Gauss value targets
# - v47: imagined PPO uses one fixed dream buffer per rollout iteration, and the detached
#   latent agent input is RMS-normalized before the standard ReLU-squared actor/critic
# - v48: imagined lambda returns reset across non-learnable sampled-terminal steps; dreamed
#   diagnostics track advantage/action correlation and policy-neighborhood reward sensitivity
# - v51: dreamed rollouts are prompted with recent same-episode real latent/action context
#   before generated policy actions, matching Dreamer-style prompted generation
# - v52: predictor emits next latents through a LeWM-style projection instead of a
#   zero-initialized residual delta, removing the identity shortcut that made dreams
#   nearly action-indifferent
# - v53: expands the rollout bottleneck to 8 latent tokens and represents continuous
#   actions as one explicit predictor token per action dimension
# - v54: replaces SiLU/gated hidden activations with ReLU-squared throughout
#   the world model, predictor, action embedder, and readout projections
#   and uses standard Pre-LN residuals with parameter-golf-style residual mixing
# - v55: actor/critic use ReLU-squared MLP heads with RMSNorm after each hidden
#   activation, and real rollout PPO is restored as an anchor using stored
#   rollout latent tokens with real GAE advantages/returns
# - v56: widens actor/critic hidden layers to 256, removes the actor mean/logstd
#   clamps, and stores real rollout logprobs per action dimension like dreams
# - v57: aligns real rollout actor training with imagined actor training by using
#   the same asymmetric SPO objective; v67 restores asymmetric half-strength SPO
# - v58: separates actor and critic latent-input normalization so critic/value
#   gradients cannot move the actor feature scale, and logs real KL like SPO refs
# - v60: restores LeWM-style target-gradient flow for summary prediction while
#   detaching future target summaries when they are used as teacher-forced context
# - v61: restores LeWM-style multi-step predictor context and per-layer action
#   conditioning via AdaLN, and regularizes the full state+outcome summary
# - v62: replaces per-observation-dimension tokens with 8 mixed observation tokens:
#   obs_dim -> Linear(NUM_OBS_TOKENS * MODEL_DIM)
# - v63: constructs the full dreamed PPO batch in GPU chunks, stages it on CPU,
#   then streams shuffled dream minibatches back to CUDA for multi-epoch PPO
# - v64: removes learned summary/state query tokens; the recurrent world-model state
#   is now 8 mixed observation latent tokens plus 2 reward/continuation outcome tokens
# - v65: logs real and imagined PPO return targets separately for each agent minibatch
# - v66: replaces tanh-squashed Gaussian actions with D4-style Beta policies on
#   normalized action coordinates, linearly mapped to the environment action box
# - v67: restores asymmetric half-strength SPO for real and imagined Beta-policy updates
# - v68: fixes time-limit bootstrapping, boundary target validity, direct outcome-token
#   prediction gradients, observation-only SIGReg, and survival-weighted soft dream continuations
# - v69: keeps learned outcome target projections inside the JEPA embedding system:
#   full state+outcome summaries get SIGReg and reward/continue are decoded by
#   distance to split learned outcome-token codebooks, not a trained CE/BCE dynamics head
# - v82: returns to the v69 backbone but decodes reward/termination through detached
#   supervised probes from predicted outcome tokens, avoiding codebook-distance decode drift
# - v83: removes predicted-token outcome probes; reward/termination scalars are decoded
#   through inverse heads calibrated only on target outcome tokens, while predicted outcome
#   tokens remain trained by the JEPA embedding objective
# - v85: calibrates the same outcome decoder on both target tokens and detached predicted
#   tokens, closing the encoded-vs-predicted readout gap without reward/termination
#   gradients entering the JEPA predictor
# - v87: uses the detached reward probe as the reward interface for real and imagined
#   rollout targets from the start; CE to actual env rewards provides the anchor
# - v88: restores env-reward real PPO anchoring and adds behavior-action closed-loop
#   JEPA/probe training to reduce autoregressive dream exposure bias
# - v89: makes real/prompt/current summary outcome slots consistently mean previous
#   transition outcome; episode starts keep neutral previous-outcome padding
# - v95: replaces the attempted recurrent return token with a Dreamer4-style WM value
#   readout from generated summaries. Imagined lambda returns use predicted rewards plus
#   this WM value head, decoupling dream targets from the agent critic bootstrap.
# - v96: makes value a non-recurrent JEPA target token. The predictor emits value
#   tokens from latent/action context, they are matched to learned value target tokens,
#   and decoded for imagined lambda returns without feeding value labels back as context.
# - v97: makes value tokens recurrent. Predictor context contains core summaries plus
#   value tokens, and predicted value tokens are appended to dream history.
# - v98: simplifies recurrent value learning to two losses: target-token CE grounding
#   and value-token JEPA MSE. SIGReg is applied uniformly to valid current-step tokens,
#   not flattened past/history windows.
# - v99: keeps value prediction in the world model but removes value tokens from recurrent
#   dynamics context. The predictor rolls core state/outcome tokens only; value is read from
#   predicted core summaries for lambda returns and supervised against HL-Gauss value targets.
# - v101: adds Dreamer-style multi-token prediction supervision. Each predictor context
#   emits equal-weight +1..+4 future core embeddings; the shared outcome/value probes train
#   on every predicted future embed, while imagination still rolls one policy step at a time.
# - v102: prompts dreams from the latest rollout with a short real behavior-action prefix:
#   after the historical context, it appends actual rollout actions and encoded next summaries
#   before policy-controlled imagination begins.
# - v103: sets the real behavior-action prefix to the imagined horizon length by default,
#   grounding each dreamed segment after an equally long latest-rollout continuation.
# - v104: restores horizon parity: dynamics training horizon, real behavior prompt prefix,
#   and imagined PPO rollout horizon all default to 5 steps.
# - v105: replaces the predictor's flattened block-causal attention with Dreamer4-style
#   axial STSTS attention: space-only current-step token mixing and causal time-only
#   per-slot mixing, both using mask-free Flash-friendly SDPA.
# - v106: batches dream prompt cache prefill in one axial full-context pass, then uses
#   one-step cached decode only for newly imagined transitions; rollout/dream no-grad
#   paths use inference_mode to reduce Python/autograd overhead.
# - v108: increases dream generation chunk size to the full 32k rollout batch while
#   retaining CPU staging before imagined PPO minibatch updates.
# - v109: restores 16k dream staging after the 32k test, and reduces hot-path
#   synchronization: diagnostics are interval-gated, minibatch return logging is
#   deferred to iteration boundaries, rollout CPU->GPU copies are consolidated,
#   and dream actor calls avoid discarded critic forwards.
# - v110: restores v108's NumPy real-minibatch shuffle so PPO shuffling does not
#   consume the CUDA policy/dream RNG stream, and fully gates dream diagnostics.
# - v111: detaches recurrent predictor KV caches in closed-loop WM training and
#   avoids time-axis AdaLN parameter expansion copies.
# - v112: uses partial RoPE with 25% of each attention head rotated, leaving the
#   remaining channel capacity for non-positional content.
# - v113: restores 16-step imagined PPO rollout horizon while keeping the v112
#   5-step dynamics training horizon and 5-step real behavior prompt prefix.
# - v114: removes the auxiliary closed-loop WM training losses. Dynamics training
#   is teacher-forced/MTP JEPA plus detached outcome/value probe grounding only;
#   closed-loop use is limited to actual no-grad imagination rollouts.
# - v115: removes stale compatibility helpers and unused buffers, drops unused
#   dream value staging, avoids redundant MTP value-token reads, and gates
#   expensive scalar diagnostics.
# - v118: adds a detached TD2-CFM successor-summary flow probe with an EMA target
#   vector field. The LEJEPA backbone remains trained only by its embed/SIGReg
#   objectives; imagined WM values are derived from decoded successor rewards.
# - v119: replaces the full-summary successor flow with a reward-token TD2-CFM
#   successor model. Imagined WM values are the decoded expectation of sampled
#   successor reward tokens; stale WM value-token prediction losses are removed.
# - v120: implements the full TD2-CFM successor measure over all recurrent
#   state+outcome tokens, keeping v119's corrected gamma, sample count, shared
#   Bellman denominator, and no stale WM value-token fallback.
# - v121: makes the imagined value bridge explicit: the dream rollout computes
#   TDF tail value on the LEJEPA-predicted post-action summary, then logs the
#   one-step action-informed target r + gamma * continue * V(next).
# - v123: replaces full-summary successor TDF with a compact reward-sufficient
#   successor-feature flow. The flow still follows TD2-CFM, but value is read by
#   applying a learned reward functional to sampled successor features.
import copy
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.distributions.beta import Beta
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cleanrl.shared.hl_gauss import HLGaussSupport


MODEL_DIM = 64
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
FFN_MULT = 2
ROPE_FRACTION = 0.25
DYN_NUM_LAYERS = 2
PRED_AXES = ("space", "time", "space", "time", "space")
PRED_DROPOUT = 0.0
PRED_CONTEXT = 5
DEFAULT_PRED_CONTEXT = 5
NUM_OBS_TOKENS = 8
NUM_OUTCOME_TOKENS = 2
NUM_LATENT_TOKENS = NUM_OBS_TOKENS + NUM_OUTCOME_TOKENS
NUM_RECURRENT_TOKENS = NUM_LATENT_TOKENS
NUM_SIGREG_TOKENS = NUM_LATENT_TOKENS
NUM_SUCCESSOR_FEATURE_TOKENS = 2
MTP_PRED_LEN = 4
SCALAR_EMBED_DIM = 32
AGENT_INPUT_DIM = NUM_OBS_TOKENS * MODEL_DIM
AGENT_HIDDEN_DIM = 256
SIGREG_CHUNK_SIZE = 2048
SAMPLE_EPS = 1e-7


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
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    async_vector_env: bool = True
    """use subprocess vector envs to overlap CPU MuJoCo stepping"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    wm_update_epochs: int = 1
    """number of world-model epochs per rollout iteration after warmup starts"""
    agent_update_epochs: int = 4
    """number of PPO epochs per rollout iteration once agent training is enabled"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """PPO reference threshold used for ratio-drift diagnostics"""
    spo_eps_low: float = 0.40
    """SPO bound when ratio drift opposes the advantage direction; half-strength vs 0.20"""
    spo_eps_high: float = 0.56
    """SPO bound when ratio drift agrees with the advantage direction; half-strength vs 0.28"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    detach_world_model_from_agent: bool = True
    """if toggled, PPO and dreamed agent losses see detached world-model latent tokens"""

    # HL-Gauss specific
    num_bins: int = 51
    """number of bins for the categorical value head"""
    v_min: float = -5.0
    """minimum value of the support (in symlog space)"""
    v_max: float = 5.0
    """maximum value of the support (in symlog space)"""
    sigma_ratio: float = 0.5
    """sigma / bin_width ratio for HL-Gauss target smoothing"""

    # LeWM dynamics auxiliary
    dyn_horizon: int = 5
    """teacher-forced dynamics horizon"""
    pred_context: int = DEFAULT_PRED_CONTEXT
    """number of summary steps the predictor can attend over; v61 defaults to the full dynamics horizon"""
    dyn_latent_coef: float = 1.0
    """weight on next-dynamics-token prediction"""
    dyn_reward_coef: float = 0.25
    """weight on target reward-outcome inverse decoder calibration"""
    dyn_termination_coef: float = 0.25
    """weight on target continuation-outcome inverse decoder calibration"""
    successor_flow_coef: float = 0.25
    """weight on the detached TD2-CFM projected successor-feature flow"""
    successor_feature_reward_coef: float = 1.0
    """weight on reward-functional CE for projected successor features"""
    successor_feature_termination_coef: float = 1.0
    """weight on termination-functional BCE for projected successor features"""
    successor_flow_steps: int = 4
    """Euler midpoint steps used to sample full successor summaries for WM values"""
    successor_flow_train_steps: int = 4
    """Euler midpoint steps used for TD2 target-flow path samples during training"""
    successor_flow_target_decay: float = 0.999
    """EMA decay for the target successor-flow vector field"""
    successor_value_samples: int = 16
    """number of full-successor-flow samples averaged for imagined WM value estimates"""
    successor_value_chunk_size: int = 32768
    """maximum expanded successor-flow sample rows per value decode chunk"""
    reward_num_bins: int = 51
    """number of bins for the learned reward outcome token"""
    reward_v_min: float = -10.0
    """minimum reward support for the learned reward outcome token"""
    reward_v_max: float = 10.0
    """maximum reward support for the learned reward outcome token"""
    reward_sigma_ratio: float = 0.75
    """sigma / bin_width ratio for the auxiliary reward support"""
    imagine_horizon: int = 16
    """dream rollout horizon for Dreamer-style imagined GAE"""
    dream_prompt_len: int = DEFAULT_PRED_CONTEXT
    """real same-episode summary/action context length used to prompt dreamed rollouts"""
    dream_behavior_prefix_len: int = 5
    """number of latest-rollout behavior transitions appended to each dream prompt before policy rollout"""
    imagine_actor_coef: float = 1.0
    """weight on the imagined-rollout actor objective"""
    imagine_critic_coef: float = 0.5
    """weight on the imagined-rollout critic objective"""
    imagine_actor_ent_coef: float = 0.0
    """entropy bonus for the dreamed PPO actor update"""
    imagine_update_epochs: int = 4
    """number of PPO epochs over the fixed imagined rollout buffer"""
    imagine_start_step: int = 0
    """global step at which dreamed updates become active"""
    wm_warmup_steps: int = 100000
    """number of env steps to train only the world model before enabling agent updates"""
    sigreg_coef: float = 0.09
    """weight on the SIGReg latent anti-collapse regularizer"""
    sigreg_num_proj: int = 1024
    """number of random projections used by SIGReg"""
    sigreg_knots: int = 17
    """number of quadrature knots used by SIGReg"""
    sigreg_min_valid: int = 32
    """minimum valid samples required for a masked timestep to contribute to SIGReg"""
    dynamics_diagnostic_batch: int = 1024
    """number of real rollout starts used for detached dynamics diagnostics"""
    imagination_diagnostic_batch: int = 512
    """number of dreamed starts used for detached imagination control-signal diagnostics"""
    dream_build_batch_size: int = 16384
    """number of real rollout starts to dream at once before staging tensors on CPU"""
    action_sensitivity_samples: int = 8
    """number of random actions per state for reward action-sensitivity diagnostics"""
    diagnostics_interval: int = 10
    """run expensive dynamics/imagination diagnostics every N iterations; <=0 disables them"""

    # v124 inner-Adam action solver
    inner_solver_steps: int = 8
    """number of Adam steps taken on the free action variable each amortization minibatch"""
    inner_solver_lr: float = 0.1
    """Adam learning rate on the free action variable"""
    inner_solver_horizon: int = 1
    """planning horizon over which the inner solver minimizes cost; v140 uses H=1"""
    inner_solver_init: str = "policy_mean"
    """initialization for the inner solver action variable: 'policy_mean' or 'policy_sample'"""
    inner_solver_samples: int = 4
    """number of candidate action sequences optimized per state; sample 0 is the policy initialization"""
    inner_solver_sample_std: float = 0.25
    """stddev of normalized-action Gaussian perturbations for nonzero inner-solver samples"""
    planner_start_step: int = 100000
    """global env step at which real rollout behavior switches to solver-selected actions"""
    planner_execute: bool = True
    """if toggled, execute the inner solver's first optimized action during real rollout after planner_start_step"""
    planner_disable_pg: bool = True
    """if toggled with planner_execute, disable real actor SPO/PrefPoE policy-gradient terms after planner_start_step"""
    planner_amortize_behavior: bool = True
    """if toggled with planner_execute, train the actor toward logged planner behavior actions; default false avoids v138 actor-planner feedback"""
    planner_warm_start: bool = True
    """warm-start each MPC solve from the shifted tail of the previous optimized action sequence"""
    planner_receding_horizon: int = 1
    """number of optimized actions considered consumed before shifting the warm-start plan"""
    goal_solver_steps: int = 8
    """Adam steps for optimizing the imagined goal action through the frozen WM"""
    goal_solver_lr: float = 0.05
    """Adam learning rate for the imagined goal action"""
    goal_defect_coef: float = 1.0
    """unused in v145; kept for CLI compatibility with earlier goal variants"""
    goal_action_steps: int = 16
    """Adam steps for the action solver that minimizes WM-predicted obs-latent distance to the solved goal"""
    goal_action_lr: float = 0.1
    """Adam learning rate for the goal-reaching action solver"""
    amortize_coef: float = 1.0
    """weight on the actor amortization regression loss"""
    amortize_warmup_steps: int = 0
    """env steps after WM warmup before amortization gradients are applied"""
    real_spo_coef: float = 1.0
    """weight on the real-rollout SPO actor anchor (set to 0 to rely on amortization alone)"""
    # v133 PrefPoE (arXiv 2511.08241) — Beta-PoE adaptation, entropy-regularized.
    lambda_pref: float = 0.5
    """PoE weight on the preference head; fused Beta has
       α_fused = α_θ + λ_pref·(α_pref − 1), β_fused likewise. λ=0 → main only; λ=1 → full PoE."""
    prefpoe_beta1: float = 1.0
    """β_1 in the paper — strength of advantage guidance on the preference loss
       L_pref = −β_1·E[A_norm·log π_pref(a|s)] − α·H(π_pref)."""
    prefpoe_alpha_entropy: float = 0.0
    """α in the paper — entropy regularization on the preference head.
       DISABLED in v135 (default 0.0) because Beta entropy gradient vanishes
       with concentration (∂H/∂(α+β) ~ −1/(α+β)). Replaced by the linear
       prefpoe_concentration_penalty below, which has constant gradient.
       Leave at 0.0 unless explicitly experimenting."""
    prefpoe_concentration_penalty: float = 0.01
    """v135: linear penalty `c · E[(α_pref + β_pref − 2)]` added to L_pref.
       Replaces the broken Beta entropy regularizer. Penalty is 0 at uniform
       Beta(1,1) and grows linearly with concentration. Gradient is constant
       in (α+β) so does not vanish at high concentration (unlike H(Beta))."""
    prefpoe_pref_concentration_cap: float = 50.0
    """Hard cap on per-dim Beta concentration (α_pref ≤ cap, β_pref ≤ cap).
       Defensive backstop: even with entropy reg the unbounded `1+softplus`
       parameterization can drift to extreme values. Cap=50 ⇒ α+β ≤ 100/dim
       ⇒ Beta entropy ≥ −2.3/dim. Set to 0 (or negative) to disable."""
    prefpoe_w_pref: float = 1.0
    """w_pref in L_total = L_PPO + w_pref·L_pref + w_cons·L_cons."""
    prefpoe_w_cons: float = 0.1
    """w_cons — weight on the KL(π_fused ‖ π_pref) consistency anchor."""
    prefpoe_warmup_steps: int = 0
    """env steps before PrefPoE losses (L_pref, L_cons) and PoE fusion kick in.
       During warmup the actor behaves like v124 (sample/log_prob/entropy under main)."""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


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


def xavier_init_linear(layer, gain=1.0):
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def safe_mean(values):
    return float(np.mean(values)) if len(values) else 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class ReluSq(nn.Module):
    def forward(self, x):
        return torch.relu(x).square()


class ReluSqRMSHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, output_std=1.0):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            ReluSq(),
            RMSNorm(hidden_dim),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            ReluSq(),
            RMSNorm(hidden_dim),
            layer_init(nn.Linear(hidden_dim, output_dim), std=output_std),
        )

    def forward(self, x):
        return self.net(x)


def relu_sq(x):
    return torch.relu(x).square()


class SIGReg(nn.Module):
    """LeWM-style Sketched Isotropic Gaussian Regularizer."""

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

    def sample_projection(self, dim, device, dtype):
        A = torch.randn(dim, self.num_proj, device=device, dtype=dtype)
        A = A.div_(A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8))
        return A

    def forward(self, proj, A=None):
        # proj: (T, B, D)
        if A is None:
            A = self.sample_projection(proj.size(-1), proj.device, proj.dtype)
        t = self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        weights = self.weights.to(device=proj.device, dtype=proj.dtype)
        x_t = (proj @ A).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()


def build_rope_cache(num_obs_tokens, num_special_tokens, head_dim, device, base=10000.0, rope_fraction=ROPE_FRACTION):
    rotary_dim = int(head_dim * rope_fraction)
    rotary_dim = max(2, rotary_dim)
    rotary_dim -= rotary_dim % 2
    if rotary_dim <= 0 or rotary_dim > head_dim:
        raise ValueError(f"invalid rotary_dim={rotary_dim} for head_dim={head_dim}")
    total_tokens = num_obs_tokens + num_special_tokens
    theta = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim))
    positions = torch.arange(num_obs_tokens, device=device).float()
    freqs = torch.outer(positions, theta)

    cos = torch.ones(total_tokens, rotary_dim // 2, device=device)
    sin = torch.zeros(total_tokens, rotary_dim // 2, device=device)
    cos[num_special_tokens:] = torch.cos(freqs)
    sin[num_special_tokens:] = torch.sin(freqs)
    return cos, sin


def apply_rope(x, cos, sin):
    rotary_half_dim = cos.shape[-1]
    rotary_dim = rotary_half_dim * 2
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = x_rot[..., :rotary_half_dim], x_rot[..., rotary_half_dim:]
    x_rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    if x_pass.shape[-1] == 0:
        return x_rot
    return torch.cat([x_rot, x_pass], dim=-1)


def attention(q, k, v, dropout_p=0.0, attn_mask=None, is_causal=False, enable_gqa=False):
    if q.is_cuda and attn_mask is None:
        attn_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                    out = F.scaled_dot_product_attention(
                        q.to(attn_dtype),
                        k.to(attn_dtype),
                        v.to(attn_dtype),
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        enable_gqa=enable_gqa,
                    )
                return out.to(q.dtype)
            except RuntimeError:
                pass
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        enable_gqa=enable_gqa,
    )


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_q_heads, num_kv_heads, ffn_mult=2):
        super().__init__()
        assert dim % num_q_heads == 0
        assert num_q_heads % num_kv_heads == 0
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_q_heads
        self.kv_group_size = num_q_heads // num_kv_heads

        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.ffn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

        self.wq = nn.Linear(dim, num_q_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_q_heads * self.head_dim, dim, bias=False)

        ffn_dim = dim * ffn_mult
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = None

        for module in [self.wq, self.wk, self.wv, self.wo, self.w1, self.w2]:
            xavier_init_linear(module)

    def forward(self, x, rope_cos, rope_sin, *, x0, attn_mask=None):
        batch, seq_len, width = x.shape
        mix = self.resid_mix.to(dtype=x.dtype, device=x.device)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        h = self.attn_norm(x)
        q = self.wq(h).view(batch, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(self.q_norm(q), rope_cos, rope_sin)
        k = apply_rope(self.k_norm(k), rope_cos, rope_sin)

        attn_out = attention(q, k, v, attn_mask=attn_mask, enable_gqa=self.kv_group_size > 1)
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, width)
        x = x + self.attn_scale.to(dtype=x.dtype, device=x.device)[None, None, :] * self.wo(attn_out)

        h = self.ffn_norm(x)
        x = x + self.ffn_scale.to(dtype=x.dtype, device=x.device)[None, None, :] * self.w2(relu_sq(self.w1(h)))
        return x


class AxialAdaLNTransformerBlock(nn.Module):
    def __init__(self, dim, num_q_heads, num_kv_heads, axis, ffn_mult=2, dropout=0.0):
        super().__init__()
        if axis not in {"space", "time"}:
            raise ValueError(f"unknown predictor axis {axis}")
        assert dim % num_q_heads == 0
        assert num_q_heads % num_kv_heads == 0
        self.axis = axis
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_q_heads
        self.kv_group_size = num_q_heads // num_kv_heads
        self.dropout = dropout

        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.wq = nn.Linear(dim, num_q_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_q_heads * self.head_dim, dim, bias=False)

        ffn_dim = dim * ffn_mult
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.adaln = nn.Sequential(ReluSq(), nn.Linear(dim, 6 * dim))

        for module in [self.wq, self.wk, self.wv, self.wo, self.w1, self.w2]:
            xavier_init_linear(module)
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def _modulate(self, x, shift, scale):
        return x * (1.0 + scale) + shift

    def _to_axis(self, x):
        batch, time, space, width = x.shape
        if self.axis == "space":
            return x.reshape(batch * time, space, width), batch, time, space, width
        return x.permute(0, 2, 1, 3).contiguous().reshape(batch * space, time, width), batch, time, space, width

    def _from_axis(self, x, batch, time, space, width):
        if self.axis == "space":
            return x.reshape(batch, time, space, width)
        return x.reshape(batch, space, time, width).permute(0, 2, 1, 3).contiguous()

    def _adaln_params(self, step_action_features, batch, time, space):
        params = self.adaln(step_action_features.reshape(batch * time, -1)).reshape(batch, time, 6, -1)
        if self.axis == "space":
            params = params.reshape(batch * time, 1, 6, -1)
        return params.unbind(dim=2)

    def _modulate_axis(self, x_axis, shift, scale, batch, time, space, width):
        if self.axis == "space":
            return self._modulate(x_axis, shift, scale)
        x_4d = x_axis.reshape(batch, space, time, width)
        return self._modulate(
            x_4d,
            shift[:, None, :, :],
            scale[:, None, :, :],
        ).reshape(batch * space, time, width)

    def _apply_axis_gate(self, x_axis, residual, gate, batch, time, space, width):
        if self.axis == "space":
            return x_axis + gate * residual
        x_4d = x_axis.reshape(batch, space, time, width)
        residual_4d = residual.reshape(batch, space, time, width)
        return (x_4d + gate[:, None, :, :] * residual_4d).reshape(batch * space, time, width)

    def forward(self, x, step_action_features, rope_cos, rope_sin, return_cache=False, max_context=PRED_CONTEXT):
        x_axis, batch, time, space, width = self._to_axis(x)
        seq_len = x_axis.shape[1]
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = self._adaln_params(
            step_action_features,
            batch,
            time,
            space,
        )

        h = self._modulate_axis(
            self.attn_norm(x_axis),
            shift_attn,
            scale_attn,
            batch,
            time,
            space,
            width,
        )
        q = self.wq(h).view(x_axis.shape[0], seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(x_axis.shape[0], seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(x_axis.shape[0], seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        layer_cache = None
        if return_cache and self.axis == "time":
            layer_cache = (k[:, :, -max_context:].detach(), v[:, :, -max_context:].detach())
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        dropout_p = self.dropout if self.training else 0.0
        attn_out = attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=self.axis == "time",
            enable_gqa=self.kv_group_size > 1,
        )
        attn_out = self.wo(attn_out.transpose(1, 2).reshape(x_axis.shape[0], seq_len, width))
        if self.dropout > 0.0:
            attn_out = F.dropout(attn_out, p=self.dropout, training=self.training)
        x_axis = self._apply_axis_gate(x_axis, attn_out, gate_attn, batch, time, space, width)

        h = self._modulate_axis(
            self.ffn_norm(x_axis),
            shift_ffn,
            scale_ffn,
            batch,
            time,
            space,
            width,
        )
        ffn_out = self.w2(relu_sq(self.w1(h)))
        if self.dropout > 0.0:
            ffn_out = F.dropout(ffn_out, p=self.dropout, training=self.training)
        x_axis = self._apply_axis_gate(x_axis, ffn_out, gate_ffn, batch, time, space, width)
        out = self._from_axis(x_axis, batch, time, space, width)
        if return_cache:
            return out, layer_cache
        return out

    def forward_step(self, x, step_action_features, rope_cos, rope_sin, cache=None, max_context=PRED_CONTEXT):
        if self.axis == "space":
            return self.forward(x, step_action_features, rope_cos, rope_sin), cache

        x_axis, batch, time, space, width = self._to_axis(x)
        if time != 1:
            raise ValueError("forward_step expects exactly one current timestep")
        seq_len = 1
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = self._adaln_params(
            step_action_features,
            batch,
            time,
            space,
        )

        h = self._modulate_axis(
            self.attn_norm(x_axis),
            shift_attn,
            scale_attn,
            batch,
            time,
            space,
            width,
        )
        q = self.wq(h).view(x_axis.shape[0], seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(x_axis.shape[0], seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(x_axis.shape[0], seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if cache is None:
            raw_k, cached_v = k, v
        else:
            cached_k, cached_v = cache
            raw_k = torch.cat([cached_k, k], dim=2)
            cached_v = torch.cat([cached_v, v], dim=2)
            if raw_k.shape[2] > max_context:
                raw_k = raw_k[:, :, -max_context:]
                cached_v = cached_v[:, :, -max_context:]

        cache_len = raw_k.shape[2]
        q = apply_rope(q, rope_cos[cache_len - 1 : cache_len], rope_sin[cache_len - 1 : cache_len])
        k = apply_rope(raw_k, rope_cos[:cache_len], rope_sin[:cache_len])

        dropout_p = self.dropout if self.training else 0.0
        attn_out = attention(q, k, cached_v, dropout_p=dropout_p, enable_gqa=self.kv_group_size > 1)
        attn_out = self.wo(attn_out.transpose(1, 2).reshape(x_axis.shape[0], seq_len, width))
        if self.dropout > 0.0:
            attn_out = F.dropout(attn_out, p=self.dropout, training=self.training)
        x_axis = self._apply_axis_gate(x_axis, attn_out, gate_attn, batch, time, space, width)

        h = self._modulate_axis(
            self.ffn_norm(x_axis),
            shift_ffn,
            scale_ffn,
            batch,
            time,
            space,
            width,
        )
        ffn_out = self.w2(relu_sq(self.w1(h)))
        if self.dropout > 0.0:
            ffn_out = F.dropout(ffn_out, p=self.dropout, training=self.training)
        x_axis = self._apply_axis_gate(x_axis, ffn_out, gate_ffn, batch, time, space, width)
        return self._from_axis(x_axis, batch, time, space, width), (raw_k, cached_v)


class Agent(nn.Module):
    def __init__(
        self,
        envs,
        num_bins,
        reward_num_bins,
        detach_world_model_from_agent=True,
    ):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        self.act_dim = act_dim
        self.reward_num_bins = reward_num_bins
        self.detach_world_model_from_agent = detach_world_model_from_agent
        self.register_buffer(
            "action_low",
            torch.tensor(envs.single_action_space.low, dtype=torch.float32),
        )
        self.register_buffer(
            "action_high",
            torch.tensor(envs.single_action_space.high, dtype=torch.float32),
        )

        self.obs_input_norm = RMSNorm(obs_dim)
        self.obs_mix_proj = xavier_init_linear(nn.Linear(obs_dim, NUM_OBS_TOKENS * MODEL_DIM))
        self.obs_token_norm = RMSNorm(MODEL_DIM)

        self.dyn_embed_norm = RMSNorm(MODEL_DIM)
        self.dyn_layers = nn.ModuleList(
            [TransformerBlock(MODEL_DIM, NUM_Q_HEADS, NUM_KV_HEADS, FFN_MULT) for _ in range(DYN_NUM_LAYERS)]
        )
        self.dyn_final_norm = RMSNorm(MODEL_DIM)
        self.dyn_next_proj = xavier_init_linear(nn.Linear(MODEL_DIM, MODEL_DIM))

        self.pred_action_in_proj = xavier_init_linear(nn.Linear(1, SCALAR_EMBED_DIM))
        self.pred_action_out_proj = xavier_init_linear(nn.Linear(SCALAR_EMBED_DIM, MODEL_DIM))
        self.pred_action_dim_embed = nn.Parameter(torch.empty(act_dim, MODEL_DIM))
        nn.init.xavier_uniform_(self.pred_action_dim_embed)
        self.pred_action_cond_proj = xavier_init_linear(nn.Linear(act_dim, MODEL_DIM))
        self.pred_layers = nn.ModuleList(
            [
                AxialAdaLNTransformerBlock(
                    MODEL_DIM,
                    NUM_Q_HEADS,
                    NUM_KV_HEADS,
                    axis=axis,
                    ffn_mult=FFN_MULT,
                    dropout=PRED_DROPOUT,
                )
                for axis in PRED_AXES
            ]
        )
        self.pred_final_norm = RMSNorm(MODEL_DIM)
        self.pred_next_proj = xavier_init_linear(nn.Linear(MODEL_DIM, MODEL_DIM))
        self.pred_mtp_next_projs = nn.ModuleList(
            [xavier_init_linear(nn.Linear(MODEL_DIM, MODEL_DIM)) for _ in range(MTP_PRED_LEN - 1)]
        )

        head_dim = MODEL_DIM // NUM_Q_HEADS
        dyn_rope_cos, dyn_rope_sin = build_rope_cache(
            NUM_OBS_TOKENS, 0, head_dim, torch.device("cpu")
        )
        pred_tokens_per_step = act_dim + NUM_RECURRENT_TOKENS
        pred_space_rope_cos, pred_space_rope_sin = build_rope_cache(
            pred_tokens_per_step, 0, head_dim, torch.device("cpu")
        )
        pred_time_rope_cos, pred_time_rope_sin = build_rope_cache(
            PRED_CONTEXT, 0, head_dim, torch.device("cpu")
        )
        self.register_buffer("dyn_rope_cos", dyn_rope_cos)
        self.register_buffer("dyn_rope_sin", dyn_rope_sin)
        self.register_buffer("pred_space_rope_cos", pred_space_rope_cos)
        self.register_buffer("pred_space_rope_sin", pred_space_rope_sin)
        self.register_buffer("pred_time_rope_cos", pred_time_rope_cos)
        self.register_buffer("pred_time_rope_sin", pred_time_rope_sin)

        self.critic = ReluSqRMSHead(AGENT_INPUT_DIM, AGENT_HIDDEN_DIM, num_bins, output_std=1.0)
        self.actor_beta = ReluSqRMSHead(AGENT_INPUT_DIM, AGENT_HIDDEN_DIM, 2 * act_dim, output_std=0.01)
        # v132 PrefPoE: independent preference head, shared backbone via
        # `_actor_features_from_latents`. Same width as main; same `1+softplus`
        # parameterization so α_pref, β_pref ≥ 1 and the Beta-PoE closed form
        # is well-defined (see header).
        self.actor_pref_beta = ReluSqRMSHead(AGENT_INPUT_DIM, AGENT_HIDDEN_DIM, 2 * act_dim, output_std=0.01)
        self.actor_input_norm = RMSNorm(AGENT_INPUT_DIM)
        self.critic_input_norm = RMSNorm(AGENT_INPUT_DIM)
        successor_cond_dim = NUM_LATENT_TOKENS * MODEL_DIM
        successor_flow_dim = NUM_SUCCESSOR_FEATURE_TOKENS * MODEL_DIM
        self.successor_feature_input_norm = RMSNorm(successor_cond_dim)
        self.successor_feature_proj = ReluSqRMSHead(
            successor_cond_dim,
            AGENT_HIDDEN_DIM,
            successor_flow_dim,
            output_std=0.5,
        )
        self.successor_feature_token_norm = RMSNorm(MODEL_DIM)
        self.target_successor_feature_input_norm = copy.deepcopy(self.successor_feature_input_norm)
        self.target_successor_feature_proj = copy.deepcopy(self.successor_feature_proj)
        self.target_successor_feature_token_norm = copy.deepcopy(self.successor_feature_token_norm)
        for module in (
            self.target_successor_feature_input_norm,
            self.target_successor_feature_proj,
            self.target_successor_feature_token_norm,
        ):
            module.requires_grad_(False)
        self.successor_flow_cond_norm = RMSNorm(successor_cond_dim)
        self.successor_flow_cond_proj = ReluSqRMSHead(
            successor_cond_dim,
            AGENT_HIDDEN_DIM,
            successor_flow_dim,
            output_std=0.5,
        )
        self.successor_flow_state_norm = RMSNorm(successor_flow_dim)
        self.successor_flow_time_proj = nn.Sequential(
            xavier_init_linear(nn.Linear(1, MODEL_DIM)),
            ReluSq(),
            RMSNorm(MODEL_DIM),
            xavier_init_linear(nn.Linear(MODEL_DIM, successor_flow_dim)),
        )
        self.successor_flow_input_norm = RMSNorm(3 * successor_flow_dim)
        self.successor_flow_head = ReluSqRMSHead(
            3 * successor_flow_dim,
            AGENT_HIDDEN_DIM * 2,
            successor_flow_dim,
            output_std=0.1,
        )
        self.target_successor_flow_cond_norm = copy.deepcopy(self.successor_flow_cond_norm)
        self.target_successor_flow_cond_proj = copy.deepcopy(self.successor_flow_cond_proj)
        self.target_successor_flow_state_norm = copy.deepcopy(self.successor_flow_state_norm)
        self.target_successor_flow_time_proj = copy.deepcopy(self.successor_flow_time_proj)
        self.target_successor_flow_input_norm = copy.deepcopy(self.successor_flow_input_norm)
        self.target_successor_flow_head = copy.deepcopy(self.successor_flow_head)
        for module in self.target_successor_flow_modules():
            module.requires_grad_(False)
        self.successor_feature_reward_head = ReluSqRMSHead(
            successor_flow_dim,
            AGENT_HIDDEN_DIM,
            reward_num_bins,
            output_std=1.0,
        )
        self.successor_feature_termination_head = ReluSqRMSHead(
            successor_flow_dim,
            AGENT_HIDDEN_DIM,
            1,
            output_std=1.0,
        )
        nn.init.constant_(self.successor_feature_termination_head.net[-1].bias, -5.0)
        self.reward_outcome_input_norm = RMSNorm(reward_num_bins)
        self.reward_outcome_proj = xavier_init_linear(nn.Linear(reward_num_bins, MODEL_DIM))
        self.continuation_outcome_input_norm = RMSNorm(1)
        self.continuation_outcome_proj = xavier_init_linear(nn.Linear(1, MODEL_DIM))
        self.outcome_token_norm = RMSNorm(MODEL_DIM)
        self.register_buffer("reward_codebook_probs", torch.eye(reward_num_bins), persistent=False)
        self.reward_outcome_unproj = xavier_init_linear(nn.Linear(MODEL_DIM, reward_num_bins))
        self.continuation_outcome_unproj = xavier_init_linear(nn.Linear(MODEL_DIM, 1))
        nn.init.constant_(self.continuation_outcome_unproj.bias, -5.0)

    def _action_distribution(self, agent_input):
        # Main policy Beta (v124 semantics, kept for inner solver / amortize).
        beta_head = self.actor_beta(agent_input)
        head_alpha, head_beta = beta_head.chunk(2, dim=-1)
        alpha = 1.0 + F.softplus(head_alpha)
        beta = 1.0 + F.softplus(head_beta)
        return Beta(alpha, beta)

    def _pref_action_distribution(self, agent_input):
        # v133 PrefPoE: advantage-guided preference Beta head, with hard cap on
        # concentration (α_pref ≤ cap, β_pref ≤ cap) when cap > 1. Prevents
        # the v132 collapse where unbounded softplus drove α_pref+β_pref → ∞.
        beta_head = self.actor_pref_beta(agent_input)
        head_alpha, head_beta = beta_head.chunk(2, dim=-1)
        alpha = 1.0 + F.softplus(head_alpha)
        beta = 1.0 + F.softplus(head_beta)
        cap = getattr(self, "pref_concentration_cap", 0.0)
        if cap > 1.0:
            alpha = alpha.clamp(max=cap)
            beta = beta.clamp(max=cap)
        return Beta(alpha, beta)

    def _fuse_betas(self, main_dist, pref_dist, lambda_pref):
        # v132 PrefPoE: closed-form Beta-PoE.
        #   Beta(α_θ,β_θ) · Beta(α_pref,β_pref)^λ
        #     ∝ x^{α_θ−1+λ(α_pref−1)} · (1−x)^{β_θ−1+λ(β_pref−1)}
        #     = Beta(α_θ + λ(α_pref−1),  β_θ + λ(β_pref−1))
        # With both α, β ≥ 1 from the 1+softplus parameterization, fused
        # params are ≥ 1 ⇒ fused is a proper unimodal Beta.
        alpha_fused = main_dist.concentration1 + lambda_pref * (pref_dist.concentration1 - 1.0)
        beta_fused = main_dist.concentration0 + lambda_pref * (pref_dist.concentration0 - 1.0)
        return Beta(alpha_fused, beta_fused)

    def _action_distributions(self, agent_input, lambda_pref):
        """Return (main, pref, fused) Beta distributions for one shared agent_input.
        `lambda_pref == 0.0` short-circuits to (main, main, main) — equivalent
        to v124 behavior during the optional warmup phase."""
        main_dist = self._action_distribution(agent_input)
        if lambda_pref == 0.0:
            return main_dist, main_dist, main_dist
        pref_dist = self._pref_action_distribution(agent_input)
        fused_dist = self._fuse_betas(main_dist, pref_dist, lambda_pref)
        return main_dist, pref_dist, fused_dist

    def _z_to_action(self, action_z):
        return self.action_low + (self.action_high - self.action_low) * action_z

    def _action_to_z(self, action):
        action_z = (action - self.action_low) / (self.action_high - self.action_low)
        return action_z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)

    def _beta_action_logprob_entropy(self, dist, action=None, action_z=None, sum_logprob=True):
        if action_z is None:
            if action is not None:
                action_z = self._action_to_z(action)
            else:
                action_z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                action = self._z_to_action(action_z)
        else:
            action_z = action_z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            action = self._z_to_action(action_z)
        logprob_per_dim = dist.log_prob(action_z)
        entropy_per_dim = dist.entropy()
        if sum_logprob:
            logprob = logprob_per_dim.sum(1)
            entropy = entropy_per_dim.sum(1)
        else:
            logprob = logprob_per_dim
            entropy = entropy_per_dim
        return action, action_z, logprob, entropy

    def _action_mean_from_dist(self, dist):
        return self._z_to_action(dist.mean)

    def _action_std_from_dist(self, dist):
        return dist.stddev * (self.action_high - self.action_low)

    def _encode_dynamics_tokens(self, x):
        batch = x.shape[0]
        obs_flat = x.reshape(batch, -1)
        obs_tokens = self.obs_mix_proj(self.obs_input_norm(obs_flat))
        obs_tokens = self.obs_token_norm(obs_tokens.reshape(batch, NUM_OBS_TOKENS, MODEL_DIM))
        dyn_tokens = self.dyn_embed_norm(obs_tokens)

        dyn_x0 = dyn_tokens
        for layer in self.dyn_layers:
            dyn_tokens = layer(dyn_tokens, self.dyn_rope_cos, self.dyn_rope_sin, x0=dyn_x0)

        return self.dyn_final_norm(dyn_tokens)

    def _encode_obs_latent_tokens(self, x):
        obs_tokens = self._encode_dynamics_tokens(x)
        return self.dyn_next_proj(obs_tokens)

    def _reward_outcome_token(self, reward_probs):
        return self.outcome_token_norm(
            self.reward_outcome_proj(self.reward_outcome_input_norm(reward_probs))
        )

    def _continuation_outcome_token(self, continuations):
        continuation_input = continuations.unsqueeze(-1)
        return self.outcome_token_norm(
            self.continuation_outcome_proj(self.continuation_outcome_input_norm(continuation_input))
        )

    def _outcome_tokens_from_labels(self, reward_probs, continuations):
        reward_token = self._reward_outcome_token(reward_probs)
        continuation_token = self._continuation_outcome_token(continuations)
        return torch.stack([reward_token, continuation_token], dim=1)

    def _neutral_outcome_tokens(self, obs_tokens):
        reward_probs = self.reward_codebook_probs.to(
            device=obs_tokens.device,
            dtype=obs_tokens.dtype,
        )[self.reward_num_bins // 2].expand(obs_tokens.shape[0], -1)
        continuations = obs_tokens.new_ones((obs_tokens.shape[0],))
        return self._outcome_tokens_from_labels(reward_probs, continuations)

    def _latent_from_obs(self, obs_tokens, outcome_tokens):
        return torch.cat([obs_tokens, outcome_tokens], dim=1)

    def _encode_online_summary(self, x):
        obs_tokens = self._encode_obs_latent_tokens(x)
        return self._latent_from_obs(obs_tokens, self._neutral_outcome_tokens(obs_tokens))

    def encode_summary_with_outcomes(self, x, reward_probs, continuations):
        obs_tokens = self._encode_obs_latent_tokens(x)
        outcome_tokens = self._outcome_tokens_from_labels(reward_probs, continuations)
        return self._latent_from_obs(obs_tokens, outcome_tokens)

    def decode_outcomes(self, summary_tokens, detach_summary=True):
        outcome_tokens = summary_tokens[:, NUM_OBS_TOKENS:]
        if detach_summary:
            outcome_tokens = outcome_tokens.detach()
        reward_logits = self.reward_outcome_unproj(outcome_tokens[:, 0])
        termination_logits = self.continuation_outcome_unproj(outcome_tokens[:, 1]).squeeze(-1)
        return reward_logits, termination_logits

    def successor_features(self, summary_tokens, use_target=False, detach_summary=True):
        flat_summary = summary_tokens.reshape(summary_tokens.shape[0], -1)
        if detach_summary:
            flat_summary = flat_summary.detach()
        if use_target:
            input_norm = self.target_successor_feature_input_norm
            feature_proj = self.target_successor_feature_proj
            token_norm = self.target_successor_feature_token_norm
        else:
            input_norm = self.successor_feature_input_norm
            feature_proj = self.successor_feature_proj
            token_norm = self.successor_feature_token_norm
        features = feature_proj(input_norm(flat_summary))
        features = features.reshape(summary_tokens.shape[0], NUM_SUCCESSOR_FEATURE_TOKENS, MODEL_DIM)
        return token_norm(features)

    def decode_successor_feature_outcomes(self, feature_tokens, detach_features=True):
        if detach_features:
            feature_tokens = feature_tokens.detach()
        flat_features = feature_tokens.reshape(feature_tokens.shape[0], -1)
        reward_logits = self.successor_feature_reward_head(flat_features)
        termination_logits = self.successor_feature_termination_head(flat_features).squeeze(-1)
        return reward_logits, termination_logits

    def _actor_features_from_latents(self, latent_tokens):
        obs_tokens = latent_tokens[:, :NUM_OBS_TOKENS]
        return self.actor_input_norm(obs_tokens.reshape(obs_tokens.shape[0], -1))

    def _critic_features_from_latents(self, latent_tokens):
        obs_tokens = latent_tokens[:, :NUM_OBS_TOKENS]
        return self.critic_input_norm(obs_tokens.reshape(obs_tokens.shape[0], -1))

    def successor_flow_modules(self):
        return (
            self.successor_feature_input_norm,
            self.successor_feature_proj,
            self.successor_feature_token_norm,
            self.successor_flow_cond_norm,
            self.successor_flow_cond_proj,
            self.successor_flow_state_norm,
            self.successor_flow_time_proj,
            self.successor_flow_input_norm,
            self.successor_flow_head,
        )

    def target_successor_flow_modules(self):
        return (
            self.target_successor_feature_input_norm,
            self.target_successor_feature_proj,
            self.target_successor_feature_token_norm,
            self.target_successor_flow_cond_norm,
            self.target_successor_flow_cond_proj,
            self.target_successor_flow_state_norm,
            self.target_successor_flow_time_proj,
            self.target_successor_flow_input_norm,
            self.target_successor_flow_head,
        )

    @torch.no_grad()
    def update_successor_flow_target(self, decay):
        for target_module, online_module in zip(self.target_successor_flow_modules(), self.successor_flow_modules()):
            for target_param, online_param in zip(target_module.parameters(), online_module.parameters()):
                target_param.lerp_(online_param, 1.0 - decay)
            for target_buffer, online_buffer in zip(target_module.buffers(), online_module.buffers()):
                target_buffer.copy_(online_buffer)

    def _successor_flow_velocity(self, condition_tokens, successor_xt, flow_t, use_target=False):
        condition = condition_tokens.detach().reshape(condition_tokens.shape[0], -1)
        if successor_xt.dim() == 3:
            successor_xt = successor_xt.reshape(successor_xt.shape[0], -1)
        if flow_t.dim() == 0:
            flow_t = flow_t.expand(condition.shape[0])
        flow_t = flow_t.reshape(-1, 1).to(device=condition.device, dtype=condition.dtype)
        if use_target:
            cond_norm = self.target_successor_flow_cond_norm
            cond_proj = self.target_successor_flow_cond_proj
            state_norm = self.target_successor_flow_state_norm
            time_proj = self.target_successor_flow_time_proj
            input_norm = self.target_successor_flow_input_norm
            flow_head = self.target_successor_flow_head
        else:
            cond_norm = self.successor_flow_cond_norm
            cond_proj = self.successor_flow_cond_proj
            state_norm = self.successor_flow_state_norm
            time_proj = self.successor_flow_time_proj
            input_norm = self.successor_flow_input_norm
            flow_head = self.successor_flow_head
        cond = cond_proj(cond_norm(condition))
        xt = state_norm(successor_xt)
        t_embed = time_proj(flow_t)
        flow_input = input_norm(torch.cat([cond, xt, t_embed], dim=-1))
        velocity = flow_head(flow_input)
        return velocity.reshape(condition_tokens.shape[0], NUM_SUCCESSOR_FEATURE_TOKENS, MODEL_DIM)

    def integrate_successor_flow(
        self,
        condition_tokens,
        start_tokens=None,
        end_t=1.0,
        steps=4,
        use_target=False,
    ):
        if start_tokens is None:
            successor_tokens = torch.randn(
                condition_tokens.shape[0],
                NUM_SUCCESSOR_FEATURE_TOKENS,
                MODEL_DIM,
                device=condition_tokens.device,
                dtype=condition_tokens.dtype,
            )
        else:
            successor_tokens = start_tokens.reshape(
                condition_tokens.shape[0],
                NUM_SUCCESSOR_FEATURE_TOKENS,
                MODEL_DIM,
            )
        if torch.is_tensor(end_t):
            end_t_tensor = end_t.reshape(-1, 1, 1).to(
                device=condition_tokens.device,
                dtype=condition_tokens.dtype,
            )
        else:
            end_t_tensor = condition_tokens.new_full((condition_tokens.shape[0], 1, 1), float(end_t))
        dt = end_t_tensor / max(1, steps)
        for step_idx in range(max(1, steps)):
            mid_t = (step_idx + 0.5) * dt.squeeze(-1).squeeze(-1)
            velocity = self._successor_flow_velocity(
                condition_tokens,
                successor_tokens,
                mid_t,
                use_target=use_target,
            )
            successor_tokens = successor_tokens + dt * velocity
        return successor_tokens

    def successor_td2_flow_loss(
        self,
        source_summary_tokens,
        next_summary_tokens,
        absorbing_summary_tokens,
        terminal_mask,
        sample_weight,
        bootstrap_weight,
        gamma,
        target_flow_steps,
    ):
        flat_source = source_summary_tokens.reshape(-1, NUM_LATENT_TOKENS, MODEL_DIM).detach()
        flat_next_summary = next_summary_tokens.reshape(-1, NUM_LATENT_TOKENS, MODEL_DIM).detach()
        flat_next = self.successor_features(flat_next_summary, use_target=True, detach_summary=True).detach()
        flat_absorbing = torch.zeros_like(flat_next)
        flat_terminal = terminal_mask.reshape(-1).bool()
        flat_weight = sample_weight.reshape(-1)
        flat_bootstrap_weight = bootstrap_weight.reshape(-1)

        flow_t = torch.rand(flat_source.shape[0], device=flat_source.device, dtype=flat_source.dtype)
        source_noise = torch.randn_like(flat_next)
        path_t = flow_t.reshape(-1, 1, 1)
        direct_xt = source_noise.lerp(flat_next, path_t)
        direct_target_velocity = flat_next - source_noise
        direct_velocity = self._successor_flow_velocity(
            flat_source,
            direct_xt,
            flow_t,
            use_target=False,
        )
        direct_loss = F.mse_loss(
            direct_velocity,
            direct_target_velocity,
            reduction="none",
        ).mean(dim=(-1, -2))

        boot_noise = torch.randn_like(flat_next)
        with torch.no_grad():
            boot_xt = self.integrate_successor_flow(
                flat_next_summary,
                start_tokens=boot_noise,
                end_t=flow_t,
                steps=target_flow_steps,
                use_target=True,
            )
            boot_target_velocity = self._successor_flow_velocity(
                flat_next_summary,
                boot_xt,
                flow_t,
                use_target=True,
            )
            if bool(flat_terminal.any()):
                terminal_path_t = path_t[flat_terminal]
                terminal_noise = boot_noise[flat_terminal]
                terminal_absorbing = flat_absorbing[flat_terminal]
                boot_xt[flat_terminal] = terminal_noise.lerp(terminal_absorbing, terminal_path_t)
                boot_target_velocity[flat_terminal] = terminal_absorbing - terminal_noise
        boot_velocity = self._successor_flow_velocity(
            flat_source,
            boot_xt.detach(),
            flow_t,
            use_target=False,
            )
        boot_loss = F.mse_loss(
            boot_velocity,
            boot_target_velocity.detach(),
            reduction="none",
        ).mean(dim=(-1, -2))
        denom = flat_weight.sum().clamp_min(1.0)
        direct_loss = (direct_loss * flat_weight).sum() / denom
        boot_loss = (boot_loss * flat_bootstrap_weight).sum() / denom
        return (1.0 - gamma) * direct_loss + gamma * boot_loss

    def successor_value(self, summary_tokens, reward_support, gamma, samples=1, steps=4, chunk_size=0):
        batch = summary_tokens.shape[0]
        total = batch * samples
        if chunk_size <= 0 or total <= chunk_size:
            if samples > 1:
                condition_tokens = summary_tokens[:, None].expand(
                    batch,
                    samples,
                    NUM_LATENT_TOKENS,
                    MODEL_DIM,
                ).reshape(total, NUM_LATENT_TOKENS, MODEL_DIM)
            else:
                condition_tokens = summary_tokens
            successor_tokens = self.integrate_successor_flow(
                condition_tokens,
                steps=steps,
                use_target=False,
            )
            reward_logits, _ = self.decode_successor_feature_outcomes(successor_tokens, detach_features=True)
            successor_rewards = reward_support.to_scalar(reward_logits).reshape(batch, samples)
        else:
            reward_chunks = []
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                base_inds = torch.arange(start, end, device=summary_tokens.device) // samples
                condition_tokens = summary_tokens[base_inds]
                successor_tokens = self.integrate_successor_flow(
                    condition_tokens,
                    steps=steps,
                    use_target=False,
                )
                reward_logits, _ = self.decode_successor_feature_outcomes(successor_tokens, detach_features=True)
                reward_chunks.append(reward_support.to_scalar(reward_logits))
            successor_rewards = torch.cat(reward_chunks, dim=0).reshape(batch, samples)
        return successor_rewards.mean(dim=1) / (1.0 - gamma)

    def _value_from_agent_input(self, agent_input, hl_support):
        if hl_support is None:
            raise ValueError("hl_support is required for HL-Gauss value decoding")
        return hl_support.to_scalar(self.critic(agent_input))

    def _encode_critic_features(self, x):
        latent_tokens = self._encode_online_summary(x)
        if self.detach_world_model_from_agent:
            latent_tokens = latent_tokens.detach()
        return self._critic_features_from_latents(latent_tokens)

    def _assert_cached_predictor_no_dropout(self):
        if any(layer.dropout != 0.0 for layer in self.pred_layers):
            raise ValueError("cached predictor decode requires zero predictor dropout")

    def predict_next_latents_all_from_history(self, latent_history, action_history, return_mtp=False, return_cache=False):
        if return_cache:
            self._assert_cached_predictor_no_dropout()
        if latent_history.shape[2] != NUM_RECURRENT_TOKENS:
            raise ValueError(
                f"latent_history must have {NUM_RECURRENT_TOKENS} recurrent tokens, got {latent_history.shape[2]}"
            )
        batch, context_len, num_tokens, width = latent_history.shape
        if context_len > PRED_CONTEXT:
            raise ValueError(f"context_len={context_len} exceeds PRED_CONTEXT={PRED_CONTEXT}")

        action_tokens = self.pred_action_out_proj(relu_sq(self.pred_action_in_proj(action_history.unsqueeze(-1))))
        action_tokens = action_tokens + self.pred_action_dim_embed.view(1, 1, self.act_dim, width)
        tokens_per_step = self.act_dim + num_tokens
        pred_tokens = torch.cat([action_tokens, latent_history], dim=2)
        action_features = self.pred_action_cond_proj(action_history)
        space_rope_cos = self.pred_space_rope_cos[:tokens_per_step]
        space_rope_sin = self.pred_space_rope_sin[:tokens_per_step]
        time_rope_cos = self.pred_time_rope_cos[:context_len]
        time_rope_sin = self.pred_time_rope_sin[:context_len]
        predictor_cache = []
        for layer in self.pred_layers:
            if layer.axis == "space":
                if return_cache:
                    pred_tokens, layer_cache = layer(
                        pred_tokens,
                        action_features,
                        space_rope_cos,
                        space_rope_sin,
                        return_cache=True,
                        max_context=context_len,
                    )
                else:
                    pred_tokens = layer(pred_tokens, action_features, space_rope_cos, space_rope_sin)
            else:
                if return_cache:
                    pred_tokens, layer_cache = layer(
                        pred_tokens,
                        action_features,
                        time_rope_cos,
                        time_rope_sin,
                        return_cache=True,
                        max_context=context_len,
                    )
                else:
                    pred_tokens = layer(pred_tokens, action_features, time_rope_cos, time_rope_sin)
            if return_cache:
                predictor_cache.append(layer_cache)
        pred_tokens = self.pred_final_norm(pred_tokens)
        pred_latent_features = pred_tokens[:, :, self.act_dim :]
        pred_latent_features = pred_latent_features[:, :, :NUM_LATENT_TOKENS]
        pred_latents = self.pred_next_proj(pred_latent_features)
        if return_mtp:
            mtp_latents = [pred_latents]
            for mtp_proj in self.pred_mtp_next_projs:
                mtp_latents.append(mtp_proj(pred_latent_features))
            pred_mtp_latents = torch.stack(mtp_latents, dim=2)
            if return_cache:
                return pred_latents, pred_mtp_latents, predictor_cache
            return pred_latents, pred_mtp_latents
        if return_cache:
            return pred_latents, predictor_cache
        return pred_latents

    def dynamics_step_from_history_with_cache(self, summary_history, action_history):
        pred_next_latents, predictor_cache = self.predict_next_latents_all_from_history(
            summary_history,
            action_history,
            return_cache=True,
        )
        pred_next_summary = pred_next_latents[:, -1]
        pred_reward_logits, pred_termination_logits = self.decode_outcomes(pred_next_summary)
        return pred_next_summary, pred_reward_logits, pred_termination_logits, predictor_cache

    def predict_next_latents_cached(self, latent_tokens, action, predictor_cache=None, max_context=PRED_CONTEXT):
        self._assert_cached_predictor_no_dropout()
        if latent_tokens.shape[1] != NUM_RECURRENT_TOKENS:
            raise ValueError(
                f"latent_tokens must have {NUM_RECURRENT_TOKENS} recurrent tokens, got {latent_tokens.shape[1]}"
            )
        if max_context < 1 or max_context > PRED_CONTEXT:
            raise ValueError(f"max_context must be in [1, {PRED_CONTEXT}], got {max_context}")
        if predictor_cache is not None and len(predictor_cache) != len(self.pred_layers):
            raise ValueError("predictor_cache must have one entry per predictor layer")
        batch, num_tokens, width = latent_tokens.shape
        action_tokens = self.pred_action_out_proj(relu_sq(self.pred_action_in_proj(action.unsqueeze(-1))))
        action_tokens = action_tokens + self.pred_action_dim_embed.view(1, self.act_dim, width)
        tokens_per_step = self.act_dim + num_tokens
        pred_tokens = torch.cat([action_tokens, latent_tokens], dim=1).unsqueeze(1)
        action_features = self.pred_action_cond_proj(action).unsqueeze(1)
        space_rope_cos = self.pred_space_rope_cos[:tokens_per_step]
        space_rope_sin = self.pred_space_rope_sin[:tokens_per_step]
        time_rope_cos = self.pred_time_rope_cos[:max_context]
        time_rope_sin = self.pred_time_rope_sin[:max_context]
        if predictor_cache is None:
            predictor_cache = [None] * len(self.pred_layers)
        next_cache = []
        for layer_idx, layer in enumerate(self.pred_layers):
            if layer.axis == "space":
                pred_tokens, layer_cache = layer.forward_step(
                    pred_tokens,
                    action_features,
                    space_rope_cos,
                    space_rope_sin,
                    predictor_cache[layer_idx],
                    max_context=max_context,
                )
            else:
                pred_tokens, layer_cache = layer.forward_step(
                    pred_tokens,
                    action_features,
                    time_rope_cos,
                    time_rope_sin,
                    predictor_cache[layer_idx],
                    max_context=max_context,
                )
            next_cache.append(layer_cache)
        pred_tokens = self.pred_final_norm(pred_tokens)
        pred_latent_features = pred_tokens[:, 0, self.act_dim : self.act_dim + NUM_LATENT_TOKENS]
        pred_latents = self.pred_next_proj(pred_latent_features)
        return pred_latents, next_cache

    def dynamics_step_from_cache(self, latent_tokens, action, predictor_cache=None, max_context=PRED_CONTEXT):
        pred_next_latents, predictor_cache = self.predict_next_latents_cached(
            latent_tokens,
            action,
            predictor_cache,
            max_context=max_context,
        )
        pred_reward_logits, pred_termination_logits = self.decode_outcomes(pred_next_latents)
        return pred_next_latents, pred_reward_logits, pred_termination_logits, predictor_cache

    def predict_next_latents_from_history(self, latent_history, action_history):
        pred_latents = self.predict_next_latents_all_from_history(
            latent_history,
            action_history,
        )
        return pred_latents[:, -1]

    def dynamics_step_from_history(self, summary_history, action_history):
        if summary_history.shape[2] != NUM_RECURRENT_TOKENS:
            raise ValueError(
                f"summary_history must have {NUM_RECURRENT_TOKENS} core recurrent tokens, got {summary_history.shape[2]}"
            )
        latent_history = summary_history
        pred_next_latents = self.predict_next_latents_from_history(
            latent_history,
            action_history,
        )
        pred_next_summary = pred_next_latents
        pred_reward_logits, pred_termination_logits = self.decode_outcomes(pred_next_latents)
        return pred_next_summary, pred_reward_logits, pred_termination_logits

    def get_imagined_action_dist(self, summary_tokens):
        latent_tokens = summary_tokens
        if self.detach_world_model_from_agent:
            latent_tokens = latent_tokens.detach()
        agent_input = self._actor_features_from_latents(latent_tokens)
        return self._action_distribution(agent_input)

    def get_imagined_action_mean(self, summary_tokens):
        dist = self.get_imagined_action_dist(summary_tokens)
        return self._action_mean_from_dist(dist)

    def get_imagined_action_std(self, summary_tokens):
        dist = self.get_imagined_action_dist(summary_tokens)
        return self._action_std_from_dist(dist)

    def get_imagined_action_logprob_entropy(
        self,
        summary_tokens,
        action=None,
        action_z=None,
        sum_logprob=False,
    ):
        dist = self.get_imagined_action_dist(summary_tokens)
        return self._beta_action_logprob_entropy(
            dist,
            action=action,
            action_z=action_z,
            sum_logprob=sum_logprob,
        )

    def get_imagined_raw_action_mean(self, summary_tokens):
        latent_tokens = summary_tokens
        if self.detach_world_model_from_agent:
            latent_tokens = latent_tokens.detach()
        agent_input = self._actor_features_from_latents(latent_tokens)
        return self.actor_beta(agent_input)

    def get_imagined_value(self, summary_tokens, hl_support=None):
        latent_tokens = summary_tokens
        if self.detach_world_model_from_agent:
            latent_tokens = latent_tokens.detach()
        agent_input = self._critic_features_from_latents(latent_tokens)
        return self._value_from_agent_input(agent_input, hl_support)

    def get_imagined_value_logits(self, summary_tokens):
        latent_tokens = summary_tokens
        if self.detach_world_model_from_agent:
            latent_tokens = latent_tokens.detach()
        agent_input = self._critic_features_from_latents(latent_tokens)
        return self.critic(agent_input)

    def get_value(self, x, hl_support=None):
        agent_input = self._encode_critic_features(x)
        return self._value_from_agent_input(agent_input, hl_support)

    def get_value_logits_from_latents(self, latent_tokens):
        if self.detach_world_model_from_agent:
            latent_tokens = latent_tokens.detach()
        agent_input = self._critic_features_from_latents(latent_tokens)
        return self.critic(agent_input)

    def get_action_and_value_from_latents(
        self,
        latent_tokens,
        hl_support,
        action=None,
        action_z=None,
        sum_logprob=True,
        lambda_pref=0.0,
    ):
        # v132 PrefPoE: when `lambda_pref > 0`, sampling/logprob/entropy are
        # under the fused Beta. `lambda_pref = 0` is the v124 path (main only).
        if self.detach_world_model_from_agent:
            latent_tokens = latent_tokens.detach()
        actor_input = self._actor_features_from_latents(latent_tokens)
        critic_input = self._critic_features_from_latents(latent_tokens)
        _, _, sample_dist = self._action_distributions(actor_input, lambda_pref)
        action, action_z, logprob, entropy = self._beta_action_logprob_entropy(
            sample_dist,
            action=action,
            action_z=action_z,
            sum_logprob=sum_logprob,
        )
        value = self._value_from_agent_input(critic_input, hl_support)
        return action, action_z, logprob, entropy, value

    def get_action_logprob_entropy_from_latents(
        self,
        latent_tokens,
        action=None,
        action_z=None,
        sum_logprob=True,
        lambda_pref=0.0,
    ):
        # v132 PrefPoE: PPO ratio recompute uses fused when `lambda_pref > 0`.
        if self.detach_world_model_from_agent:
            latent_tokens = latent_tokens.detach()
        agent_input = self._actor_features_from_latents(latent_tokens)
        _, _, sample_dist = self._action_distributions(agent_input, lambda_pref)
        return self._beta_action_logprob_entropy(
            sample_dist,
            action=action,
            action_z=action_z,
            sum_logprob=sum_logprob,
        )

    def cost_from_action(
        self,
        summary_tokens,
        action,
        hl_support,
        reward_support,
        gamma,
        horizon=1,
    ):
        """Differentiable cost in `action` over `horizon` predictor steps.

        `summary_tokens` is the start summary (B, T, D) and should be detached.
        `action` has shape (B, horizon, A) for horizon>1 or (B, A) for horizon=1.
        Returns a tensor of shape (B,) with cost
            C = -sum_{k<H} gamma^k r_hat_k - gamma^H V_hat(z_hat_H)
        Gradient flows through the predictor / outcome unproj / HL-Gauss critic
        to `action`. Callers must freeze WM/probe/critic parameters during the
        inner loop so optimizer steps only update `action`.
        """
        if action.dim() == 2:
            action_seq = action.unsqueeze(1)
        else:
            action_seq = action
        if action_seq.shape[1] != horizon:
            raise ValueError(
                f"action_seq second dim {action_seq.shape[1]} != horizon {horizon}"
            )

        batch = summary_tokens.shape[0]
        summary_history = [summary_tokens]
        action_history = []
        discounted_reward_sum = summary_tokens.new_zeros(batch)
        continue_carry = summary_tokens.new_ones(batch)
        gamma_t = 1.0
        last_summary = summary_tokens
        for step_idx in range(horizon):
            action_step = action_seq[:, step_idx]
            action_history.append(action_step)
            context_len = min(PRED_CONTEXT, len(summary_history), len(action_history))
            pred_context = torch.stack(summary_history[-context_len:], dim=1)
            action_context = torch.stack(action_history[-context_len:], dim=1)
            pred_next_summary = self.predict_next_latents_from_history(
                pred_context,
                action_context,
            )
            # reward + continuation read off predicted outcome tokens (no detach)
            outcome_tokens = pred_next_summary[:, NUM_OBS_TOKENS:]
            reward_logits = self.reward_outcome_unproj(outcome_tokens[:, 0])
            termination_logits = self.continuation_outcome_unproj(outcome_tokens[:, 1]).squeeze(-1)
            reward_hat = reward_support.to_scalar(reward_logits)
            continue_hat = 1.0 - torch.sigmoid(termination_logits)
            discounted_reward_sum = discounted_reward_sum + gamma_t * continue_carry * reward_hat
            continue_carry = continue_carry * continue_hat
            gamma_t = gamma_t * gamma
            last_summary = pred_next_summary
            summary_history.append(pred_next_summary)

        # bootstrap value on the predicted final obs tokens via HL-Gauss critic
        obs_tokens = last_summary[:, :NUM_OBS_TOKENS]
        critic_input = self.critic_input_norm(obs_tokens.reshape(batch, -1))
        value_logits = self.critic(critic_input)
        value_hat = hl_support.to_scalar(value_logits)
        bootstrap = gamma_t * continue_carry * value_hat
        return -(discounted_reward_sum + bootstrap)


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.pred_context < 1 or args.pred_context > PRED_CONTEXT:
        raise ValueError(f"--pred-context must be in [1, {PRED_CONTEXT}]")
    if args.dyn_horizon < 1:
        raise ValueError("--dyn-horizon must be at least 1")
    if args.dyn_horizon < MTP_PRED_LEN:
        raise ValueError(f"--dyn-horizon must be at least {MTP_PRED_LEN} for MTP supervision")
    if args.dream_behavior_prefix_len != args.dyn_horizon:
        raise ValueError("--dream-behavior-prefix-len must equal --dyn-horizon for horizon parity")
    if args.dyn_horizon > args.pred_context:
        raise ValueError("--dyn-horizon must be <= --pred-context for teacher-forced contextual prediction")
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    # env setup
    vector_env_cls = gym.vector.AsyncVectorEnv if args.async_vector_env else gym.vector.SyncVectorEnv
    envs = vector_env_cls(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(
        envs,
        args.num_bins,
        args.reward_num_bins,
        detach_world_model_from_agent=args.detach_world_model_from_agent,
    ).to(device)
    # v133 PrefPoE: hard cap on pref Beta concentration (set <=0 to disable).
    agent.pref_concentration_cap = float(args.prefpoe_pref_concentration_cap)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    def module_parameters(*modules):
        params = []
        for module in modules:
            params.extend(list(module.parameters()))
        return params

    grad_clip_groups = [
        (
            "wm",
            [
                *module_parameters(agent.obs_input_norm, agent.obs_mix_proj, agent.obs_token_norm),
                *module_parameters(agent.dyn_embed_norm, agent.dyn_layers, agent.dyn_final_norm, agent.dyn_next_proj),
                *module_parameters(
                    agent.reward_outcome_input_norm,
                    agent.reward_outcome_proj,
                    agent.continuation_outcome_input_norm,
                    agent.continuation_outcome_proj,
                    agent.outcome_token_norm,
                ),
                agent.pred_action_in_proj.weight,
                agent.pred_action_in_proj.bias,
                agent.pred_action_out_proj.weight,
                agent.pred_action_out_proj.bias,
                agent.pred_action_dim_embed,
                agent.pred_action_cond_proj.weight,
                agent.pred_action_cond_proj.bias,
                *module_parameters(
                    agent.pred_layers,
                    agent.pred_final_norm,
                    agent.pred_next_proj,
                    agent.pred_mtp_next_projs,
                ),
            ],
        ),
        (
            "outcome_inverse",
            module_parameters(
                agent.reward_outcome_unproj,
                agent.continuation_outcome_unproj,
                agent.successor_feature_input_norm,
                agent.successor_feature_proj,
                agent.successor_feature_token_norm,
                agent.successor_feature_reward_head,
                agent.successor_feature_termination_head,
                agent.successor_flow_cond_norm,
                agent.successor_flow_cond_proj,
                agent.successor_flow_state_norm,
                agent.successor_flow_time_proj,
                agent.successor_flow_input_norm,
                agent.successor_flow_head,
            ),
        ),
        # v132 PrefPoE: pref head clipped together with main under the same
        # "actor" group so the max_grad_norm budget covers both heads.
        ("actor", module_parameters(agent.actor_input_norm, agent.actor_beta, agent.actor_pref_beta)),
        ("critic", module_parameters(agent.critic_input_norm, agent.critic)),
    ]

    def clip_grad_groups():
        for _, params in grad_clip_groups:
            params_with_grad = [param for param in params if param.grad is not None]
            if params_with_grad:
                nn.utils.clip_grad_norm_(params_with_grad, args.max_grad_norm)

    sigreg = SIGReg(knots=args.sigreg_knots, num_proj=args.sigreg_num_proj).to(device)
    hl_support = HLGaussSupport(args.num_bins, args.v_min, args.v_max, args.sigma_ratio, device, use_symlog=True)
    reward_support = HLGaussSupport(
        args.reward_num_bins,
        args.reward_v_min,
        args.reward_v_max,
        args.reward_sigma_ratio,
        device,
        use_symlog=False,
    )
    agent.reward_codebook_probs = reward_support.project(reward_support.support).detach()
    action_low = torch.tensor(envs.single_action_space.low, device=device)
    action_high = torch.tensor(envs.single_action_space.high, device=device)

    def masked_token_sigreg(token_latents, token_valids):
        if token_latents.dim() != 4 or token_valids.dim() != 2:
            raise ValueError(
                f"expected token_latents [B,H,T,D] and token_valids [B,H], got {tuple(token_latents.shape)} and {tuple(token_valids.shape)}"
            )
        valid_tokens = token_latents[token_valids].reshape(-1, token_latents.shape[-1])
        valid_count = valid_tokens.shape[0]
        if valid_count < args.sigreg_min_valid:
            return token_latents.sum() * 0.0
        A = sigreg.sample_projection(
            token_latents.shape[-1],
            token_latents.device,
            token_latents.dtype,
        )
        t = sigreg.t.to(device=valid_tokens.device, dtype=valid_tokens.dtype)
        phi = sigreg.phi.to(device=valid_tokens.device, dtype=valid_tokens.dtype)
        weights = sigreg.weights.to(device=valid_tokens.device, dtype=valid_tokens.dtype)
        cos_sum = valid_tokens.new_zeros(A.shape[1], t.numel())
        sin_sum = valid_tokens.new_zeros(A.shape[1], t.numel())

        def chunk_trig(chunk_tokens):
            x_t = (chunk_tokens @ A).unsqueeze(-1) * t
            return x_t.cos().sum(dim=0), x_t.sin().sum(dim=0)

        for token_chunk in valid_tokens.split(SIGREG_CHUNK_SIZE):
            chunk_cos, chunk_sin = checkpoint(
                chunk_trig,
                token_chunk,
                use_reentrant=False,
            )
            cos_sum = cos_sum + chunk_cos
            sin_sum = sin_sum + chunk_sin

        cos_mean = cos_sum / valid_count
        sin_mean = sin_sum / valid_count
        err = (cos_mean - phi).square() + sin_mean.square()
        statistic = (err @ weights) * valid_count
        return statistic.mean()

    def imagined_lambda_returns(rewards_hat, continues_hat, values_hat, learn_masks):
        returns = []
        gae = torch.zeros_like(values_hat[-1])
        for step in reversed(range(len(rewards_hat))):
            delta = rewards_hat[step] + args.gamma * continues_hat[step] * values_hat[step + 1] - values_hat[step]
            gae = delta + args.gamma * args.gae_lambda * continues_hat[step] * gae
            gae = torch.where(learn_masks[step], gae, torch.zeros_like(gae))
            returns.append(gae + values_hat[step])
        returns.reverse()
        return returns

    def pearson_corr(x, y):
        if x.numel() <= 1 or y.numel() <= 1:
            return x.sum() * 0.0
        x = x - x.mean()
        y = y - y.mean()
        denom = x.square().mean().sqrt() * y.square().mean().sqrt()
        return (x * y).mean() / denom.clamp_min(1e-8)

    def weighted_mean(values, weights):
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)

    def freeze_agent_grad():
        prev = []
        for param in agent.parameters():
            prev.append((param, param.requires_grad))
            param.requires_grad_(False)
        return prev

    def restore_agent_grad(prev):
        for param, was_req in prev:
            param.requires_grad_(was_req)

    def run_inner_action_solver(start_summary, init_actions):
        """Adam on an unconstrained pre-tanh action variable through the frozen WM.

        start_summary: (B, NUM_LATENT_TOKENS, MODEL_DIM), detached.
        init_actions: (B, H, A) policy initial actions inside the action box.
        Returns (a_star, initial_cost, final_cost), all detached.
        """
        prev_states = freeze_agent_grad()
        was_training = agent.training
        agent.eval()
        try:
            batch_size = init_actions.shape[0]
            num_samples = max(1, args.inner_solver_samples)
            action_mid = ((action_low + action_high) * 0.5).view(1, 1, -1)
            action_scale = ((action_high - action_low) * 0.5).clamp_min(1e-6).view(1, 1, -1)

            candidate_actions = init_actions.detach().unsqueeze(1).expand(
                -1,
                num_samples,
                -1,
                -1,
            ).clone()
            if num_samples > 1 and args.inner_solver_sample_std > 0.0:
                noise = torch.randn_like(candidate_actions[:, 1:]) * args.inner_solver_sample_std
                candidate_norm = (candidate_actions[:, 1:] - action_mid.view(1, 1, 1, -1)) / action_scale.view(
                    1,
                    1,
                    1,
                    -1,
                )
                candidate_actions[:, 1:] = action_mid.view(1, 1, 1, -1) + action_scale.view(
                    1,
                    1,
                    1,
                    -1,
                ) * (candidate_norm + noise).clamp(-1.0 + SAMPLE_EPS, 1.0 - SAMPLE_EPS)

            flat_init_actions = candidate_actions.reshape(
                batch_size * num_samples,
                args.inner_solver_horizon,
                -1,
            )
            flat_start_summary = start_summary[:, None].expand(
                -1,
                num_samples,
                -1,
                -1,
            ).reshape(batch_size * num_samples, NUM_LATENT_TOKENS, MODEL_DIM)

            init_norm = ((flat_init_actions - action_mid) / action_scale).clamp(
                -1.0 + SAMPLE_EPS,
                1.0 - SAMPLE_EPS,
            )
            action_u = torch.atanh(init_norm).requires_grad_(True)
            inner_opt = optim.Adam([action_u], lr=args.inner_solver_lr)

            def bounded_action():
                return action_mid + action_scale * torch.tanh(action_u)

            with torch.no_grad():
                initial_cost = agent.cost_from_action(
                    flat_start_summary,
                    bounded_action().detach(),
                    hl_support,
                    reward_support,
                    args.gamma,
                    horizon=args.inner_solver_horizon,
                ).reshape(batch_size, num_samples).detach()
            for _ in range(args.inner_solver_steps):
                action = bounded_action()
                cost = agent.cost_from_action(
                    flat_start_summary,
                    action,
                    hl_support,
                    reward_support,
                    args.gamma,
                    horizon=args.inner_solver_horizon,
                ).mean()
                inner_opt.zero_grad(set_to_none=True)
                cost.backward()
                inner_opt.step()
            with torch.no_grad():
                final_action = bounded_action()
                final_cost = agent.cost_from_action(
                    flat_start_summary,
                    final_action.detach(),
                    hl_support,
                    reward_support,
                    args.gamma,
                    horizon=args.inner_solver_horizon,
                ).reshape(batch_size, num_samples).detach()
                final_action = final_action.reshape(
                    batch_size,
                    num_samples,
                    args.inner_solver_horizon,
                    -1,
                )
                best_idx = final_cost.argmin(dim=1)
                batch_idx = torch.arange(batch_size, device=final_cost.device)
                best_action = final_action[batch_idx, best_idx]
                best_initial_cost = initial_cost[batch_idx, best_idx]
                best_final_cost = final_cost[batch_idx, best_idx]
        finally:
            agent.train(was_training)
            restore_agent_grad(prev_states)
        return best_action.detach(), best_initial_cost, best_final_cost

    def run_goal_embedding_solver(start_summary, init_actions):
        """Two-stage H=1 planner: imagine optimal reachable g, then solve action to reach g.

        GoalSolver optimizes an action through the frozen WM return objective and
        defines g as the resulting predicted next obs-latent. ActionSolver then
        minimizes summed final embedding distance from WM(z, a) to stopgrad(g);
        return is not used in the executed-action objective.
        """
        if args.inner_solver_horizon != 1:
            raise ValueError("v145 optimal-goal solver expects --inner-solver-horizon 1")
        prev_states = freeze_agent_grad()
        was_training = agent.training
        agent.eval()
        try:
            batch_size = init_actions.shape[0]
            num_samples = max(1, args.inner_solver_samples)
            action_mid = ((action_low + action_high) * 0.5).view(1, 1, -1)
            action_scale = ((action_high - action_low) * 0.5).clamp_min(1e-6).view(1, 1, -1)

            candidate_actions = init_actions.detach().unsqueeze(1).expand(
                -1,
                num_samples,
                -1,
                -1,
            ).clone()
            if num_samples > 1 and args.inner_solver_sample_std > 0.0:
                noise = torch.randn_like(candidate_actions[:, 1:]) * args.inner_solver_sample_std
                candidate_norm = (candidate_actions[:, 1:] - action_mid.view(1, 1, 1, -1)) / action_scale.view(
                    1,
                    1,
                    1,
                    -1,
                )
                candidate_actions[:, 1:] = action_mid.view(1, 1, 1, -1) + action_scale.view(
                    1,
                    1,
                    1,
                    -1,
                ) * (candidate_norm + noise).clamp(-1.0 + SAMPLE_EPS, 1.0 - SAMPLE_EPS)

            flat_start_summary = start_summary[:, None].expand(
                -1,
                num_samples,
                -1,
                -1,
            ).reshape(batch_size * num_samples, NUM_LATENT_TOKENS, MODEL_DIM)
            flat_init_actions = candidate_actions.reshape(batch_size * num_samples, 1, -1)

            def predict_one(summary, action_seq):
                return agent.predict_next_latents_from_history(
                    summary.unsqueeze(1),
                    action_seq,
                )

            goal_init_norm = ((flat_init_actions - action_mid) / action_scale).clamp(
                -1.0 + SAMPLE_EPS,
                1.0 - SAMPLE_EPS,
            )
            goal_action_u = torch.atanh(goal_init_norm).requires_grad_(True)
            goal_opt = optim.Adam([goal_action_u], lr=args.goal_solver_lr)

            def bounded_goal_proposal_action():
                return action_mid + action_scale * torch.tanh(goal_action_u)

            def goal_energy():
                goal_action = bounded_goal_proposal_action()
                energy = agent.cost_from_action(
                    flat_start_summary,
                    goal_action,
                    hl_support,
                    reward_support,
                    args.gamma,
                    horizon=1,
                )
                return energy

            with torch.no_grad():
                initial_goal_energy = goal_energy()
            for _ in range(args.goal_solver_steps):
                energy = goal_energy()
                goal_opt.zero_grad(set_to_none=True)
                energy.mean().backward()
                goal_opt.step()

            with torch.no_grad():
                final_goal_action = bounded_goal_proposal_action().detach()
                final_goal_energy = goal_energy()
                final_pred = predict_one(flat_start_summary, final_goal_action)
                final_goal_obs = final_pred[:, :NUM_OBS_TOKENS].detach()
                outcome_tokens = final_pred[:, NUM_OBS_TOKENS:]
                reward_logits = agent.reward_outcome_unproj(outcome_tokens[:, 0])
                termination_logits = agent.continuation_outcome_unproj(outcome_tokens[:, 1]).squeeze(-1)
                reward_hat = reward_support.to_scalar(reward_logits)
                continue_hat = 1.0 - torch.sigmoid(termination_logits)
                critic_input = agent.critic_input_norm(final_goal_obs.reshape(final_goal_obs.shape[0], -1))
                goal_value = hl_support.to_scalar(agent.critic(critic_input))
                goal_return = reward_hat + args.gamma * continue_hat * goal_value
                goal_defect = (final_goal_obs - final_goal_obs).square().mean(dim=(-1, -2))
                final_goal_action = final_goal_action.reshape(batch_size, num_samples, 1, -1)
                final_goal_energy = final_goal_energy.reshape(batch_size, num_samples)
                initial_goal_energy = initial_goal_energy.reshape(batch_size, num_samples)
                goal_return = goal_return.reshape(batch_size, num_samples)
                goal_value = goal_value.reshape(batch_size, num_samples)
                goal_defect = goal_defect.reshape(batch_size, num_samples)
                final_seed_action = flat_init_actions.reshape(batch_size, num_samples, 1, -1)
                final_goal_obs = final_goal_obs.reshape(batch_size, num_samples, NUM_OBS_TOKENS, MODEL_DIM)
                best_idx = final_goal_energy.argmin(dim=1)
                batch_idx = torch.arange(batch_size, device=final_goal_energy.device)
                best_goal_obs = final_goal_obs[batch_idx, best_idx].detach()
                best_seed_action = final_seed_action[batch_idx, best_idx].detach()
                best_goal_action = final_goal_action[batch_idx, best_idx].detach()
                best_initial_goal_energy = initial_goal_energy[batch_idx, best_idx]
                best_final_goal_energy = final_goal_energy[batch_idx, best_idx]
                best_goal_return = goal_return[batch_idx, best_idx]
                best_goal_value = goal_value[batch_idx, best_idx]
                best_goal_defect = goal_defect[batch_idx, best_idx]

            action_init_norm = ((best_seed_action - action_mid) / action_scale).clamp(
                -1.0 + SAMPLE_EPS,
                1.0 - SAMPLE_EPS,
            )
            action_u = torch.atanh(action_init_norm).requires_grad_(True)
            action_opt = optim.Adam([action_u], lr=args.goal_action_lr)

            def bounded_goal_action():
                return action_mid + action_scale * torch.tanh(action_u)

            def action_distance():
                pred_next = predict_one(start_summary, bounded_goal_action())
                pred_obs = pred_next[:, :NUM_OBS_TOKENS]
                return (pred_obs - best_goal_obs).square().sum(dim=(-1, -2))

            with torch.no_grad():
                initial_action_distance = action_distance().detach()
            last_action_grad_norm = None
            for _ in range(args.goal_action_steps):
                distance = action_distance()
                action_opt.zero_grad(set_to_none=True)
                distance.mean().backward()
                with torch.no_grad():
                    if action_u.grad is not None:
                        last_action_grad_norm = action_u.grad.detach().flatten(1).norm(dim=-1)
                action_opt.step()

            with torch.no_grad():
                final_action = bounded_goal_action().detach()
                final_action_distance = action_distance().detach()
                action_delta = final_action - best_seed_action
                goal_action_delta = best_goal_action - best_seed_action
                if last_action_grad_norm is None:
                    last_action_grad_norm = torch.zeros_like(final_action_distance)
            diagnostics = {
                "goal_initial_energy": best_initial_goal_energy.detach(),
                "goal_final_energy": best_final_goal_energy.detach(),
                "goal_return": best_goal_return.detach(),
                "goal_value": best_goal_value.detach(),
                "goal_defect": best_goal_defect.detach(),
                "action_abs_delta": action_delta.abs().mean(dim=(-1, -2)).detach(),
                "action_l2_delta": action_delta.flatten(1).norm(dim=-1).detach(),
                "goal_action_abs_delta": goal_action_delta.abs().mean(dim=(-1, -2)).detach(),
                "action_grad_norm": last_action_grad_norm.detach(),
            }
        finally:
            agent.train(was_training)
            restore_agent_grad(prev_states)
        return (
            final_action.detach(),
            initial_action_distance,
            final_action_distance,
            diagnostics,
        )

    def neutral_reward_probs(batch_size):
        return reward_support.project(torch.zeros(batch_size, device=device))

    @torch.inference_mode()
    def build_dream_prompt_context(
        flat_obs,
        flat_prev_reward_probs,
        flat_prev_continues,
        rollout_actions,
        rollout_next_obs,
        rollout_rewards,
        rollout_terminations,
        rollout_boundaries,
        rollout_valids,
    ):
        prompt_len = max(1, min(args.dream_prompt_len, args.pred_context))
        behavior_prefix_len = max(0, args.dream_behavior_prefix_len)
        flat_inds = torch.arange(args.batch_size, device=device)
        step_inds = flat_inds // args.num_envs
        env_inds = flat_inds % args.num_envs
        prompt_valids = step_inds >= (prompt_len - 1)

        prompt_summary_history = []
        for offset in range(prompt_len):
            hist_step = step_inds - (prompt_len - 1 - offset)
            safe_hist_step = hist_step.clamp(min=0)
            hist_flat_inds = safe_hist_step * args.num_envs + env_inds
            core_summary = agent.encode_summary_with_outcomes(
                flat_obs[hist_flat_inds],
                flat_prev_reward_probs[hist_flat_inds],
                flat_prev_continues[hist_flat_inds],
            )
            prompt_summary_history.append(core_summary.detach())

        prompt_action_history = []
        for offset in range(prompt_len - 1):
            action_step = step_inds - (prompt_len - 1 - offset)
            safe_action_step = action_step.clamp(min=0)
            prompt_action_history.append(rollout_actions[safe_action_step, env_inds].detach())

        for back_offset in range(prompt_len - 1):
            boundary_step = step_inds - 1 - back_offset
            safe_boundary_step = boundary_step.clamp(min=0)
            prompt_valids = prompt_valids & (boundary_step >= 0)
            prompt_valids = prompt_valids & (~rollout_boundaries[safe_boundary_step, env_inds].bool())

        for prefix_offset in range(behavior_prefix_len):
            prefix_step = step_inds + prefix_offset
            prefix_in_rollout = prefix_step < args.num_steps
            safe_prefix_step = prefix_step.clamp(max=args.num_steps - 1)
            prefix_action = rollout_actions[safe_prefix_step, env_inds]
            prefix_reward_probs = reward_support.project(rollout_rewards[safe_prefix_step, env_inds])
            prefix_continues = 1.0 - rollout_terminations[safe_prefix_step, env_inds]
            prefix_summary = agent.encode_summary_with_outcomes(
                rollout_next_obs[safe_prefix_step, env_inds],
                prefix_reward_probs,
                prefix_continues,
            )
            prompt_action_history.append(prefix_action.detach())
            prompt_summary_history.append(prefix_summary.detach())
            prefix_valid = (
                prefix_in_rollout
                & rollout_valids[safe_prefix_step, env_inds].bool()
                & (~rollout_boundaries[safe_prefix_step, env_inds].bool())
            )
            prompt_valids = prompt_valids & prefix_valid

        return prompt_summary_history, prompt_action_history, prompt_valids

    def build_dream_batch(prompt_summary_history, prompt_action_history, prompt_valids, run_diagnostics=False):
        states = []
        raw_actions = []
        action_zs = []
        old_logprobs = []
        values = []
        bridged_values = []
        learn_masks = []
        learn_weights = []
        rewards_hat = []
        continues_hat = []
        policy_reward_sensitivity_stds = []
        policy_reward_sensitivity_ranges = []
        policy_latent_sensitivity_stds = []
        policy_latent_sensitivity_ranges = []
        summary_history = [summary.detach() for summary in prompt_summary_history]
        action_history = [action.detach() for action in prompt_action_history]
        predictor_cache = None
        alive = prompt_valids.float()
        current_value = None
        diagnostic_n = min(args.imagination_diagnostic_batch, summary_history[-1].shape[0]) if run_diagnostics else 0
        sensitivity_k = max(2, args.action_sensitivity_samples)
        with torch.inference_mode():
            for _ in range(args.imagine_horizon):
                summary_state = summary_history[-1].detach()
                if current_value is None:
                    current_value = agent.successor_value(
                        summary_state,
                        reward_support,
                        args.gamma,
                        samples=args.successor_value_samples,
                        steps=args.successor_flow_steps,
                        chunk_size=args.successor_value_chunk_size,
                    )
                value = current_value
                states.append(summary_state)
                dream_action, dream_action_z, old_logprob, _ = agent.get_imagined_action_logprob_entropy(
                    summary_state,
                    sum_logprob=False,
                )
                action_history.append(dream_action.detach())
                if predictor_cache is None:
                    context_len = min(args.pred_context, len(summary_history), len(action_history))
                    pred_context = torch.stack(summary_history[-context_len:], dim=1)
                    action_context = torch.stack(action_history[-context_len:], dim=1)
                    (
                        pred_next_summary,
                        pred_reward_logits,
                        pred_termination_logits,
                        predictor_cache,
                    ) = agent.dynamics_step_from_history_with_cache(pred_context, action_context)
                else:
                    context_len = min(args.pred_context, len(summary_history), len(action_history))
                    (
                        pred_next_summary,
                        pred_reward_logits,
                        pred_termination_logits,
                        predictor_cache,
                    ) = agent.dynamics_step_from_cache(
                        summary_history[-1],
                        action_history[-1],
                        predictor_cache,
                        max_context=args.pred_context,
                    )
                if diagnostic_n > 0:
                    diag_alive = alive[:diagnostic_n].bool()
                    diag_cpu_rng_state = torch.random.get_rng_state()
                    diag_cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                    try:
                        diag_state = summary_state[:diagnostic_n]
                        diag_dist = agent.get_imagined_action_dist(diag_state)
                        diag_action_zs = diag_dist.sample((sensitivity_k,)).transpose(0, 1)
                        diag_action_zs = diag_action_zs.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                        diag_actions = agent._z_to_action(diag_action_zs)
                        diag_pred_context = torch.stack(
                            [summary[:diagnostic_n] for summary in summary_history[-context_len:]],
                            dim=1,
                        )
                        diag_pred_context = diag_pred_context.unsqueeze(1).expand(
                            -1, sensitivity_k, -1, -1, -1
                        ).reshape(
                            diagnostic_n * sensitivity_k,
                            context_len,
                            NUM_RECURRENT_TOKENS,
                            MODEL_DIM,
                        )
                        if context_len > 1:
                            previous_actions = torch.stack(
                                [action[:diagnostic_n] for action in action_history[-context_len:-1]],
                                dim=1,
                            )
                            previous_actions = previous_actions.unsqueeze(1).expand(
                                -1, sensitivity_k, -1, -1
                            )
                            diag_action_context = torch.cat(
                                [previous_actions, diag_actions.unsqueeze(2)],
                                dim=2,
                            )
                        else:
                            diag_action_context = diag_actions.unsqueeze(2)
                        diag_action_context = diag_action_context.reshape(
                            diagnostic_n * sensitivity_k,
                            context_len,
                            agent.act_dim,
                        )
                        diag_next_summary, diag_reward_logits, _ = agent.dynamics_step_from_history(
                            diag_pred_context,
                            diag_action_context,
                        )
                        diag_rewards = reward_support.to_scalar(diag_reward_logits).reshape(
                            diagnostic_n,
                            sensitivity_k,
                        )
                        diag_next_latents = diag_next_summary[:, :NUM_LATENT_TOKENS].reshape(
                            diagnostic_n,
                            sensitivity_k,
                            -1,
                        )
                        if bool(diag_alive.any()):
                            alive_diag_rewards = diag_rewards[diag_alive]
                            alive_diag_latents = diag_next_latents[diag_alive]
                            policy_reward_sensitivity_stds.append(
                                alive_diag_rewards.std(dim=1, unbiased=False).mean()
                            )
                            policy_reward_sensitivity_ranges.append(
                                (
                                    alive_diag_rewards.max(dim=1).values
                                    - alive_diag_rewards.min(dim=1).values
                                ).mean()
                            )
                            policy_latent_sensitivity_stds.append(
                                alive_diag_latents.std(dim=1, unbiased=False).norm(dim=-1).mean()
                            )
                            policy_latent_sensitivity_ranges.append(
                                (
                                    alive_diag_latents.max(dim=1).values
                                    - alive_diag_latents.min(dim=1).values
                                ).norm(dim=-1).mean()
                            )
                    finally:
                        torch.random.set_rng_state(diag_cpu_rng_state)
                        if diag_cuda_rng_state is not None:
                            torch.cuda.set_rng_state_all(diag_cuda_rng_state)
                raw_actions.append(dream_action)
                action_zs.append(dream_action_z)
                old_logprobs.append(old_logprob)
                values.append(value)
                learn_masks.append(alive > 1e-6)
                learn_weights.append(alive)
                pred_reward = reward_support.to_scalar(pred_reward_logits)
                termination_prob = torch.sigmoid(pred_termination_logits)
                pred_continue = 1.0 - termination_prob
                next_value = agent.successor_value(
                    pred_next_summary.detach(),
                    reward_support,
                    args.gamma,
                    samples=args.successor_value_samples,
                    steps=args.successor_flow_steps,
                    chunk_size=args.successor_value_chunk_size,
                )
                bridged_value = pred_reward + args.gamma * pred_continue * next_value
                rewards_hat.append(pred_reward)
                continues_hat.append(pred_continue)
                bridged_values.append(bridged_value)
                alive = alive * pred_continue
                summary_history.append(pred_next_summary.detach())
                current_value = next_value

            bootstrap_value = current_value
        returns = imagined_lambda_returns(rewards_hat, continues_hat, values + [bootstrap_value], learn_masks)
        states = torch.cat(states, dim=0)
        raw_actions = torch.cat(raw_actions, dim=0)
        action_zs = torch.cat(action_zs, dim=0)
        old_logprobs = torch.cat(old_logprobs, dim=0)
        values = torch.cat(values, dim=0)
        bridged_values = torch.cat(bridged_values, dim=0)
        learn_masks = torch.cat(learn_masks, dim=0)
        learn_weights = torch.cat(learn_weights, dim=0)
        returns = torch.cat(returns, dim=0)
        advantages = returns - values
        diagnostics = {}
        if not run_diagnostics:
            return (
                states,
                raw_actions,
                action_zs,
                old_logprobs,
                advantages,
                returns,
                learn_masks,
                learn_weights,
                diagnostics,
            )
        rewards_flat = torch.cat(rewards_hat, dim=0)
        continues_flat = torch.cat(continues_hat, dim=0)
        with torch.inference_mode():
            if bool(learn_masks.any()):
                diag_rewards = rewards_flat[learn_masks]
                diag_continues = continues_flat[learn_masks]
                diag_values = values[learn_masks]
                diag_bridged_values = bridged_values[learn_masks]
                diag_returns = returns[learn_masks]
                diag_advantages = advantages[learn_masks]
                diag_actions = raw_actions[learn_masks]
                diag_action_zs = action_zs[learn_masks]
            else:
                diag_rewards = rewards_flat
                diag_continues = continues_flat
                diag_values = values
                diag_bridged_values = bridged_values
                diag_returns = returns
                diag_advantages = advantages
                diag_actions = raw_actions
                diag_action_zs = action_zs
            action_dim_corrs = []
            action_z_dim_corrs = []
            for action_dim in range(agent.act_dim):
                action_dim_corrs.append(pearson_corr(diag_advantages, diag_actions[:, action_dim]).abs())
                action_z_dim_corrs.append(
                    pearson_corr(diag_advantages, diag_action_zs[:, action_dim]).abs()
                )
            action_dim_corrs = torch.stack(action_dim_corrs)
            action_z_dim_corrs = torch.stack(action_z_dim_corrs)
            action_norm = diag_actions.norm(dim=1)
            action_energy = diag_actions.square().sum(dim=1)
            diagnostics = {
                "reward_mean": diag_rewards.mean().item(),
                "reward_std": diag_rewards.std(unbiased=False).item(),
                "continue_mean": diag_continues.mean().item(),
                "learn_mask_frac": learn_masks.float().mean().item(),
                "learn_weight_mean": learn_weights.mean().item(),
                "prompt_valid_frac": prompt_valids.float().mean().item(),
                "behavior_prefix_len": float(max(0, args.dream_behavior_prefix_len)),
                "value_mean": diag_values.mean().item(),
                "value_max": diag_values.max().item(),
                "value_bridge_mean": diag_bridged_values.mean().item(),
                "value_bridge_delta_mean": (diag_bridged_values - diag_values).mean().item(),
                "value_bridge_delta_abs_mean": (diag_bridged_values - diag_values).abs().mean().item(),
                "bootstrap_value_mean": bootstrap_value.mean().item(),
                "return_mean": diag_returns.mean().item(),
                "return_std": diag_returns.std(unbiased=False).item(),
                "return_max": diag_returns.max().item(),
                "advantage_abs_mean": diag_advantages.abs().mean().item(),
                "advantage_std": diag_advantages.std(unbiased=False).item(),
                "advantage_action_norm_corr": pearson_corr(diag_advantages, action_norm).item(),
                "advantage_action_energy_corr": pearson_corr(diag_advantages, action_energy).item(),
                "advantage_action_dim_abs_corr_mean": action_dim_corrs.mean().item(),
                "advantage_action_dim_abs_corr_max": action_dim_corrs.max().item(),
                "advantage_action_z_dim_abs_corr_mean": action_z_dim_corrs.mean().item(),
                "advantage_action_z_dim_abs_corr_max": action_z_dim_corrs.max().item(),
            }
            if policy_reward_sensitivity_stds:
                diagnostics.update(
                    reward_policy_action_sensitivity_std=torch.stack(policy_reward_sensitivity_stds).mean().item(),
                    reward_policy_action_sensitivity_range=torch.stack(policy_reward_sensitivity_ranges).mean().item(),
                    latent_policy_action_sensitivity_std=torch.stack(policy_latent_sensitivity_stds).mean().item(),
                    latent_policy_action_sensitivity_range=torch.stack(policy_latent_sensitivity_ranges).mean().item(),
                )
        return (
            states,
            raw_actions,
            action_zs,
            old_logprobs,
            advantages,
            returns,
            learn_masks,
            learn_weights,
            diagnostics,
        )

    def build_dream_batch_eval(prompt_summary_history, prompt_action_history, prompt_valids, run_diagnostics=False):
        was_training = agent.training
        agent.eval()
        try:
            total_starts = prompt_valids.shape[0]
            chunk_size = args.dream_build_batch_size if args.dream_build_batch_size > 0 else total_starts
            tensor_chunks = [[] for _ in range(8)]
            diagnostic_sums = {}
            diagnostic_weight = 0

            for start in range(0, total_starts, chunk_size):
                end = min(start + chunk_size, total_starts)
                chunk_summaries = [summary[start:end] for summary in prompt_summary_history]
                chunk_actions = [action[start:end] for action in prompt_action_history]
                chunk_valids = prompt_valids[start:end]
                chunk_batch = build_dream_batch(
                    chunk_summaries,
                    chunk_actions,
                    chunk_valids,
                    run_diagnostics=run_diagnostics,
                )
                for idx, tensor in enumerate(chunk_batch[:8]):
                    tensor_chunks[idx].append(tensor.detach().cpu())
                chunk_weight = end - start
                for key, value in chunk_batch[8].items():
                    diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + value * chunk_weight
                diagnostic_weight += chunk_weight
                del chunk_batch

            tensors = tuple(torch.cat(chunks, dim=0) for chunks in tensor_chunks)
            diagnostics = {
                key: value / max(1, diagnostic_weight)
                for key, value in diagnostic_sums.items()
            }
            return tensors + (diagnostics,)
        finally:
            agent.train(was_training)

    @torch.inference_mode()
    def dynamics_diagnostics(
        flat_obs,
        flat_prev_reward_probs,
        flat_prev_continues,
        rollout_rewards,
        rollout_actions,
        rollout_terminations,
        rollout_boundaries,
        rollout_valids,
    ):
        num_starts = min(args.dynamics_diagnostic_batch, flat_obs.shape[0])
        if num_starts <= 0:
            return {}

        was_training = agent.training
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        agent.eval()
        try:
            sample_inds = torch.randperm(flat_obs.shape[0], device=device)[:num_starts]
            mb_step_inds = sample_inds // args.num_envs
            mb_env_inds = sample_inds % args.num_envs
            initial_core_summary = agent.encode_summary_with_outcomes(
                flat_obs[sample_inds],
                flat_prev_reward_probs[sample_inds],
                flat_prev_continues[sample_inds],
            )
            summary_history = [initial_core_summary.detach()]
            action_history = []
            alive = torch.ones(num_starts, device=device)
            pred_reward_sum = torch.zeros(num_starts, device=device)
            true_reward_sum = torch.zeros(num_starts, device=device)
            pred_discounted_return = torch.zeros(num_starts, device=device)
            true_discounted_return = torch.zeros(num_starts, device=device)
            valid_any = torch.zeros(num_starts, device=device, dtype=torch.bool)
            step_abs_errors = []
            step_biases = []
            term_briers = []
            horizon = min(args.dyn_horizon, args.imagine_horizon)

            for horizon_idx in range(horizon):
                future_step_inds = mb_step_inds + horizon_idx
                in_rollout = (future_step_inds < args.num_steps).float()
                safe_step_inds = future_step_inds.clamp(max=args.num_steps - 1)
                future_actions = rollout_actions[safe_step_inds, mb_env_inds]
                future_rewards = rollout_rewards[safe_step_inds, mb_env_inds]
                future_terminations = rollout_terminations[safe_step_inds, mb_env_inds]
                future_boundaries = rollout_boundaries[safe_step_inds, mb_env_inds]
                future_valids = rollout_valids[safe_step_inds, mb_env_inds]
                step_weight = alive * in_rollout * future_valids
                valid_mask = step_weight > 0.0

                action_history.append(future_actions)
                context_len = min(args.pred_context, len(summary_history), len(action_history))
                pred_context = torch.stack(summary_history[-context_len:], dim=1)
                action_context = torch.stack(action_history[-context_len:], dim=1)
                (
                    pred_next_summary,
                    pred_reward_logits,
                    pred_termination_logits,
                ) = agent.dynamics_step_from_history(pred_context, action_context)
                pred_reward = reward_support.to_scalar(pred_reward_logits)
                terminal_prob = torch.sigmoid(pred_termination_logits)

                if bool(valid_mask.any()):
                    reward_error = pred_reward - future_rewards
                    step_abs_errors.append(reward_error[valid_mask].abs().mean())
                    step_biases.append(reward_error[valid_mask].mean())
                    term_briers.append((terminal_prob[valid_mask] - future_terminations[valid_mask]).square().mean())

                discount = args.gamma ** horizon_idx
                pred_reward_sum = pred_reward_sum + pred_reward * step_weight
                true_reward_sum = true_reward_sum + future_rewards * step_weight
                pred_discounted_return = pred_discounted_return + pred_reward * step_weight * discount
                true_discounted_return = true_discounted_return + future_rewards * step_weight * discount
                valid_any |= valid_mask
                alive = alive * (1.0 - future_boundaries)
                summary_history.append(pred_next_summary.detach())

            metrics = {}
            if bool(valid_any.any()):
                metrics.update(
                    rollout_reward_step_mae=torch.stack(step_abs_errors).mean().item() if step_abs_errors else 0.0,
                    rollout_reward_step_bias=torch.stack(step_biases).mean().item() if step_biases else 0.0,
                    rollout_reward_sum_mae=(pred_reward_sum[valid_any] - true_reward_sum[valid_any]).abs().mean().item(),
                    rollout_reward_sum_bias=(pred_reward_sum[valid_any] - true_reward_sum[valid_any]).mean().item(),
                    rollout_discounted_return_mae=(
                        pred_discounted_return[valid_any] - true_discounted_return[valid_any]
                    ).abs().mean().item(),
                    rollout_discounted_return_bias=(
                        pred_discounted_return[valid_any] - true_discounted_return[valid_any]
                    ).mean().item(),
                    rollout_terminal_brier=torch.stack(term_briers).mean().item() if term_briers else 0.0,
                    rollout_valid_frac=valid_any.float().mean().item(),
                )

            sensitivity_n = min(256, num_starts)
            sensitivity_k = max(2, args.action_sensitivity_samples)
            sensitivity_summary = summary_history[0][:sensitivity_n]
            random_actions = action_low + torch.rand(
                sensitivity_n, sensitivity_k, agent.act_dim, device=device
            ) * (action_high - action_low)
            repeated_summary = sensitivity_summary.repeat_interleave(sensitivity_k, dim=0)
            flat_actions = random_actions.reshape(sensitivity_n * sensitivity_k, agent.act_dim)
            sensitivity_next_summary, sensitivity_reward_logits, _ = agent.dynamics_step_from_history(
                repeated_summary.unsqueeze(1),
                flat_actions.unsqueeze(1),
            )
            sensitivity_rewards = reward_support.to_scalar(sensitivity_reward_logits).reshape(sensitivity_n, sensitivity_k)
            sensitivity_latents = sensitivity_next_summary[:, :NUM_LATENT_TOKENS].reshape(sensitivity_n, sensitivity_k, -1)
            metrics.update(
                reward_action_sensitivity_std=sensitivity_rewards.std(dim=1, unbiased=False).mean().item(),
                reward_action_sensitivity_range=(
                    sensitivity_rewards.max(dim=1).values - sensitivity_rewards.min(dim=1).values
                ).mean().item(),
                latent_action_sensitivity_std=sensitivity_latents.std(dim=1, unbiased=False).norm(dim=-1).mean().item(),
                latent_action_sensitivity_range=(
                    sensitivity_latents.max(dim=1).values - sensitivity_latents.min(dim=1).values
                ).norm(dim=-1).mean().item(),
            )
            return metrics
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            agent.train(was_training)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    action_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    transition_actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    planner_executed = torch.zeros((args.num_steps, args.num_envs), device=device)
    agent_latents = torch.zeros((args.num_steps, args.num_envs, NUM_LATENT_TOKENS, MODEL_DIM)).to(device)
    prev_reward_probs = torch.zeros((args.num_steps, args.num_envs, args.reward_num_bins), device=device)
    prev_outcome_continues = torch.ones((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.ones((args.num_steps, args.num_envs)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    planner_next_init = None

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    imagined_steps = 0
    imagined_learnable_steps = 0
    real_minibatch_step = 0
    imagination_minibatch_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    current_prev_reward_probs = neutral_reward_probs(args.num_envs)
    neutral_env_reward_probs = current_prev_reward_probs
    current_prev_continues = torch.ones(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        run_diagnostics = args.diagnostics_interval > 0 and iteration % args.diagnostics_interval == 0
        rollout_planner_execute_flags = []
        rollout_goal_action_initial_distances = []
        rollout_goal_action_final_distances = []
        rollout_goal_initial_energies = []
        rollout_goal_final_energies = []
        rollout_goal_returns = []
        rollout_goal_values = []
        rollout_goal_defects = []
        rollout_goal_action_abs_deltas = []
        rollout_exec_action_abs_deltas = []
        rollout_exec_action_l2_deltas = []
        rollout_exec_action_grad_norms = []

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            prev_reward_probs[step] = current_prev_reward_probs
            prev_outcome_continues[step] = current_prev_continues

            with torch.inference_mode():
                rollout_latents = agent.encode_summary_with_outcomes(
                    next_obs,
                    current_prev_reward_probs,
                    current_prev_continues,
                )
                if args.detach_world_model_from_agent:
                    rollout_latents = rollout_latents.detach()
                # v132 PrefPoE: gate fusion behind warmup so the WM/critic
                # have a chance to reach v124-ish quality before the
                # advantage-guided preference starts steering exploration.
                prefpoe_active = global_step >= args.prefpoe_warmup_steps
                lambda_pref_now = args.lambda_pref if prefpoe_active else 0.0
                action, action_z, logprob, _, value = agent.get_action_and_value_from_latents(
                    rollout_latents,
                    hl_support,
                    sum_logprob=False,
                    lambda_pref=lambda_pref_now,
                )
                values[step] = value.flatten()
                agent_latents[step] = rollout_latents.detach()
            planner_active = (
                args.planner_execute
                and global_step >= args.planner_start_step
                and global_step >= args.wm_warmup_steps
            )
            if planner_active:
                planner_latents = rollout_latents.detach().clone()
                with torch.no_grad():
                    init_dist = agent._action_distribution(
                        agent._actor_features_from_latents(planner_latents)
                    )
                    if args.inner_solver_init == "policy_sample":
                        planner_init_z = init_dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                    else:
                        planner_init_z = init_dist.mean.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                    planner_init_action = agent._z_to_action(planner_init_z)
                policy_init_seq = planner_init_action.unsqueeze(1).expand(
                    -1, args.inner_solver_horizon, -1
                ).contiguous()
                planner_init_seq = policy_init_seq
                if (
                    args.planner_warm_start
                    and planner_next_init is not None
                    and planner_next_init.shape[1] > 0
                ):
                    warm_len = min(planner_next_init.shape[1], args.inner_solver_horizon)
                    planner_init_seq = policy_init_seq.clone()
                    planner_init_seq[:, :warm_len] = planner_next_init[:, :warm_len]
                (
                    planner_action_seq,
                    planner_initial_cost,
                    planner_final_cost,
                    planner_diagnostics,
                ) = run_goal_embedding_solver(
                    planner_latents,
                    planner_init_seq,
                )
                action = planner_action_seq[:, 0]
                if args.planner_warm_start:
                    consumed = min(
                        max(1, args.planner_receding_horizon),
                        args.inner_solver_horizon,
                    )
                    planner_next_init = planner_action_seq[:, consumed:].detach().clone()
                with torch.no_grad():
                    action_z = agent._action_to_z(action)
                    action, action_z, logprob, _ = agent.get_action_logprob_entropy_from_latents(
                        planner_latents,
                        action=action,
                        action_z=action_z,
                        sum_logprob=False,
                        lambda_pref=lambda_pref_now,
                    )
                    rollout_goal_action_initial_distances.append(planner_initial_cost.mean().item())
                    rollout_goal_action_final_distances.append(planner_final_cost.mean().item())
                    rollout_goal_initial_energies.append(
                        planner_diagnostics["goal_initial_energy"].mean().item()
                    )
                    rollout_goal_final_energies.append(
                        planner_diagnostics["goal_final_energy"].mean().item()
                    )
                    rollout_goal_returns.append(planner_diagnostics["goal_return"].mean().item())
                    rollout_goal_values.append(planner_diagnostics["goal_value"].mean().item())
                    rollout_goal_defects.append(planner_diagnostics["goal_defect"].mean().item())
                    rollout_goal_action_abs_deltas.append(
                        planner_diagnostics["goal_action_abs_delta"].mean().item()
                    )
                    rollout_exec_action_abs_deltas.append(
                        planner_diagnostics["action_abs_delta"].mean().item()
                    )
                    rollout_exec_action_l2_deltas.append(
                        planner_diagnostics["action_l2_delta"].mean().item()
                    )
                    rollout_exec_action_grad_norms.append(
                        planner_diagnostics["action_grad_norm"].mean().item()
                    )
            rollout_planner_execute_flags.append(float(planner_active))
            planner_executed[step] = float(planner_active)
            actions[step] = action
            action_zs[step] = action_z
            env_action = torch.clamp(action, action_low, action_high)
            transition_actions[step] = env_action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(env_action.detach().cpu().numpy())
            transition_termination = terminations
            transition_boundary = np.logical_or(terminations, truncations)
            transition_valid = (~transition_boundary).astype(np.float32)
            final_obs = infos.get("final_observation")
            final_obs_mask = infos.get("_final_observation")
            if final_obs is not None:
                transition_next_obs = np.array(next_obs, copy=True)
                if final_obs_mask is None:
                    final_obs_mask = [fo is not None for fo in final_obs]
                for env_idx, has_final in enumerate(final_obs_mask):
                    if has_final and final_obs[env_idx] is not None:
                        transition_next_obs[env_idx] = final_obs[env_idx]
                        transition_valid[env_idx] = 1.0
                    elif transition_boundary[env_idx]:
                        transition_valid[env_idx] = 0.0
                transition_next_obs_t = torch.as_tensor(transition_next_obs, device=device, dtype=torch.float32)
            else:
                transition_next_obs_t = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
            reward_tensor = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
            termination_tensor = torch.as_tensor(transition_termination, device=device, dtype=torch.float32)
            boundary_tensor_f = torch.as_tensor(transition_boundary, device=device, dtype=torch.float32)
            rewards[step] = reward_tensor
            transition_terminations[step] = termination_tensor
            transition_boundaries[step] = boundary_tensor_f
            transition_valids[step] = torch.as_tensor(transition_valid, device=device, dtype=torch.float32)
            next_obses[step] = transition_next_obs_t
            next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
            boundary_tensor = boundary_tensor_f.bool()
            current_prev_reward_probs = reward_support.project(reward_tensor)
            current_prev_continues = 1.0 - termination_tensor
            current_prev_reward_probs = torch.where(
                boundary_tensor[:, None],
                neutral_env_reward_probs,
                current_prev_reward_probs,
            )
            current_prev_continues = torch.where(
                boundary_tensor,
                torch.ones_like(current_prev_continues),
                current_prev_continues,
            )
            if (
                planner_next_init is not None
                and planner_next_init.shape[1] > 0
                and bool(boundary_tensor.any())
            ):
                planner_next_init[boundary_tensor] = 0.0

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # Bootstrap real PPO from environment rewards; probe rewards are logged
        # and used by imagined rollouts, but the real rollout remains the anchor.
        with torch.inference_mode():
            next_transition_values = agent.get_value(
                next_obses.reshape((-1,) + envs.single_observation_space.shape),
                hl_support,
            ).reshape(args.num_steps, args.num_envs)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                bootstrap_nonterminal = (1.0 - transition_terminations[t]) * transition_valids[t]
                lambda_nonterminal = 1.0 - transition_boundaries[t]
                delta = rewards[t] + args.gamma * next_transition_values[t] * bootstrap_nonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * lambda_nonterminal * lastgaelam
                )
            returns = advantages + values
            next_state_returns = torch.empty_like(returns)
            next_state_returns[:-1] = returns[1:]
            next_state_returns[-1] = next_transition_values[-1]
            boundary_next_returns = next_transition_values * (1.0 - transition_terminations) * transition_valids
            next_state_returns = torch.where(
                transition_boundaries.bool(),
                boundary_next_returns,
                next_state_returns,
            )

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape((-1,) + envs.single_action_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_action_zs = action_zs.reshape((-1,) + envs.single_action_space.shape)
        b_planner_executed = planner_executed.reshape(-1)
        b_agent_latents = agent_latents.reshape(-1, NUM_LATENT_TOKENS, MODEL_DIM)
        b_prev_reward_probs = prev_reward_probs.reshape(-1, args.reward_num_bins)
        b_prev_continues = prev_outcome_continues.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        world_model_only = global_step < args.wm_warmup_steps

        wm_b_inds = np.arange(args.batch_size)
        dyn_losses = []
        dyn_latent_losses = []
        dyn_reward_losses = []
        dyn_termination_losses = []
        successor_flow_losses = []
        successor_feature_reward_losses = []
        successor_feature_termination_losses = []
        pred_reward_decode_losses = []
        pred_termination_decode_losses = []
        mtp_latent_losses = []
        mtp_reward_losses = []
        mtp_termination_losses = []
        dyn_sigreg_losses = []
        lejepa_losses = []
        dyn_reward_mses = []
        dyn_termination_accs = []
        teacher_forced_latent_losses = []
        lejepa_obs_pred_mses = []
        lejepa_outcome_pred_mses = []

        for epoch in range(args.wm_update_epochs):
            np.random.shuffle(wm_b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = wm_b_inds[start:end]

                mb_size = len(mb_inds)
                mb_step_inds = torch.as_tensor(mb_inds // args.num_envs, device=device, dtype=torch.long)
                mb_env_inds = torch.as_tensor(mb_inds % args.num_envs, device=device, dtype=torch.long)
                horizon_offsets = torch.arange(args.dyn_horizon, device=device)
                future_step_inds = mb_step_inds[:, None] + horizon_offsets[None, :]
                in_rollout = (future_step_inds < args.num_steps).float()
                safe_step_inds = future_step_inds.clamp(max=args.num_steps - 1)
                env_inds = mb_env_inds[:, None].expand_as(safe_step_inds)

                future_actions = transition_actions[safe_step_inds, env_inds]
                future_rewards = rewards[safe_step_inds, env_inds]
                future_terminations = transition_terminations[safe_step_inds, env_inds]
                future_boundaries = transition_boundaries[safe_step_inds, env_inds]
                future_valids = transition_valids[safe_step_inds, env_inds]
                future_next_obs = next_obses[safe_step_inds, env_inds]

                initial_core_summary = agent.encode_summary_with_outcomes(
                    b_obs[mb_inds],
                    b_prev_reward_probs[mb_inds],
                    b_prev_continues[mb_inds],
                )
                reward_target_probs = reward_support.project(future_rewards.reshape(-1)).reshape(
                    mb_size * args.dyn_horizon,
                    -1,
                )
                future_continues = (1.0 - future_terminations).reshape(-1)
                target_core_summaries = agent.encode_summary_with_outcomes(
                    future_next_obs.reshape((-1,) + envs.single_observation_space.shape),
                    reward_target_probs,
                    future_continues,
                ).reshape(mb_size, args.dyn_horizon, NUM_LATENT_TOKENS, MODEL_DIM)
                zero_reward_probs = reward_support.project(
                    torch.zeros(mb_size * args.dyn_horizon, device=device)
                )
                absorbing_outcome_tokens = agent._outcome_tokens_from_labels(
                    zero_reward_probs,
                    torch.zeros(mb_size * args.dyn_horizon, device=device),
                ).reshape(mb_size, args.dyn_horizon, NUM_OUTCOME_TOKENS, MODEL_DIM)
                absorbing_core_summaries = torch.cat(
                    [
                        torch.zeros_like(target_core_summaries[:, :, :NUM_OBS_TOKENS]),
                        absorbing_outcome_tokens,
                    ],
                    dim=2,
                )
                teacher_core_history = torch.cat(
                    [initial_core_summary.unsqueeze(1), target_core_summaries[:, :-1].detach()],
                    dim=1,
                )
                teacher_window_latents = teacher_core_history
                target_future_summaries = target_core_summaries.reshape(
                    mb_size,
                    args.dyn_horizon,
                    NUM_SIGREG_TOKENS,
                    MODEL_DIM,
                )
                latent_history = teacher_window_latents
                target_next_latents = target_core_summaries

                (
                    pred_next_latents,
                    pred_mtp_latents,
                ) = agent.predict_next_latents_all_from_history(
                    latent_history,
                    future_actions,
                    return_mtp=True,
                )
                mtp_reward_logits, mtp_termination_logits = agent.decode_outcomes(
                    pred_mtp_latents.reshape(
                        mb_size * args.dyn_horizon * MTP_PRED_LEN,
                        NUM_LATENT_TOKENS,
                        MODEL_DIM,
                    ),
                )
                mtp_reward_logits = mtp_reward_logits.reshape(
                    mb_size,
                    args.dyn_horizon,
                    MTP_PRED_LEN,
                    -1,
                )
                mtp_termination_logits = mtp_termination_logits.reshape(
                    mb_size,
                    args.dyn_horizon,
                    MTP_PRED_LEN,
                )
                pred_reward_logits = mtp_reward_logits[:, :, 0]
                pred_termination_logits = mtp_termination_logits[:, :, 0]
                target_reward_logits, target_termination_logits = agent.decode_outcomes(
                    target_core_summaries.reshape(mb_size * args.dyn_horizon, NUM_LATENT_TOKENS, MODEL_DIM),
                )
                target_reward_logits = target_reward_logits.reshape(mb_size, args.dyn_horizon, -1)
                target_termination_logits = target_termination_logits.reshape(mb_size, args.dyn_horizon)
                reward_target_probs = reward_target_probs.reshape(
                    mb_size,
                    args.dyn_horizon,
                    -1,
                )

                prev_continues = torch.cat(
                    [
                        torch.ones(mb_size, 1, device=device),
                        1.0 - future_boundaries[:, :-1],
                    ],
                    dim=1,
                )
                step_weight = torch.cumprod(prev_continues, dim=1) * in_rollout
                latent_weight = step_weight * future_valids
                target_successor_features = agent.successor_features(
                    target_core_summaries.reshape(
                        mb_size * args.dyn_horizon,
                        NUM_LATENT_TOKENS,
                        MODEL_DIM,
                    ),
                    use_target=False,
                    detach_summary=True,
                ).reshape(
                    mb_size,
                    args.dyn_horizon,
                    NUM_SUCCESSOR_FEATURE_TOKENS,
                    MODEL_DIM,
                )
                successor_feature_reward_logits, successor_feature_termination_logits = (
                    agent.decode_successor_feature_outcomes(
                        target_successor_features.reshape(
                            mb_size * args.dyn_horizon,
                            NUM_SUCCESSOR_FEATURE_TOKENS,
                            MODEL_DIM,
                        ),
                        detach_features=False,
                    )
                )
                successor_feature_reward_logits = successor_feature_reward_logits.reshape(
                    mb_size,
                    args.dyn_horizon,
                    -1,
                )
                successor_feature_termination_logits = successor_feature_termination_logits.reshape(
                    mb_size,
                    args.dyn_horizon,
                )
                successor_feature_reward_loss = -(
                    reward_target_probs.detach()
                    * torch.log_softmax(successor_feature_reward_logits, dim=-1)
                ).sum(dim=-1)
                successor_feature_reward_loss = (
                    successor_feature_reward_loss * latent_weight
                ).sum() / latent_weight.sum().clamp_min(1.0)
                successor_feature_termination_loss = F.binary_cross_entropy_with_logits(
                    successor_feature_termination_logits,
                    future_terminations,
                    reduction="none",
                )
                successor_feature_termination_loss = (
                    successor_feature_termination_loss * latent_weight
                ).sum() / latent_weight.sum().clamp_min(1.0)
                successor_flow_loss = agent.successor_td2_flow_loss(
                    teacher_window_latents,
                    target_core_summaries,
                    absorbing_core_summaries,
                    future_terminations,
                    latent_weight,
                    latent_weight,
                    args.gamma,
                    args.successor_flow_train_steps,
                )

                mtp_latent_offset_losses = []
                mtp_reward_offset_losses = []
                mtp_termination_offset_losses = []
                for mtp_idx in range(MTP_PRED_LEN):
                    valid_horizon = args.dyn_horizon - mtp_idx
                    if valid_horizon <= 0:
                        continue
                    offset_weight = latent_weight[:, mtp_idx:]
                    offset_denom = offset_weight.sum().clamp_min(1.0)
                    offset_pred_latents = pred_mtp_latents[:, :valid_horizon, mtp_idx]
                    offset_target_latents = target_next_latents[:, mtp_idx:]
                    mtp_latent_loss = F.mse_loss(
                        offset_pred_latents,
                        offset_target_latents,
                        reduction="none",
                    ).mean(dim=(-1, -2))
                    mtp_latent_offset_losses.append(
                        (mtp_latent_loss * offset_weight).sum() / offset_denom
                    )
                    mtp_reward_loss = -(
                        reward_target_probs[:, mtp_idx:].detach()
                        * torch.log_softmax(mtp_reward_logits[:, :valid_horizon, mtp_idx], dim=-1)
                    ).sum(dim=-1)
                    mtp_reward_offset_losses.append(
                        (mtp_reward_loss * offset_weight).sum() / offset_denom
                    )
                    mtp_termination_loss = F.binary_cross_entropy_with_logits(
                        mtp_termination_logits[:, :valid_horizon, mtp_idx],
                        future_terminations[:, mtp_idx:],
                        reduction="none",
                    )
                    mtp_termination_offset_losses.append(
                        (mtp_termination_loss * offset_weight).sum() / offset_denom
                    )
                teacher_forced_latent_loss = torch.stack(mtp_latent_offset_losses).mean()
                dyn_pred_reward_loss = torch.stack(mtp_reward_offset_losses).mean()
                dyn_pred_termination_loss = torch.stack(mtp_termination_offset_losses).mean()
                target_reward_decode_loss = -(
                    reward_target_probs.detach() * torch.log_softmax(target_reward_logits, dim=-1)
                ).sum(dim=-1)
                dyn_target_reward_loss = (
                    target_reward_decode_loss * latent_weight
                ).sum() / latent_weight.sum().clamp_min(1.0)
                target_termination_decode_loss = F.binary_cross_entropy_with_logits(
                    target_termination_logits,
                    future_terminations,
                    reduction="none",
                )
                dyn_target_termination_loss = (
                    target_termination_decode_loss * latent_weight
                ).sum() / latent_weight.sum().clamp_min(1.0)

                teacher_forced_reward_loss = 0.5 * (dyn_target_reward_loss + dyn_pred_reward_loss)
                teacher_forced_termination_loss = 0.5 * (
                    dyn_target_termination_loss + dyn_pred_termination_loss
                )
                dyn_latent_loss = teacher_forced_latent_loss
                dyn_reward_loss = teacher_forced_reward_loss
                dyn_termination_loss = teacher_forced_termination_loss
                pred_reward_decode_losses.append(dyn_pred_reward_loss.item())
                pred_termination_decode_losses.append(dyn_pred_termination_loss.item())
                mtp_latent_losses.append(teacher_forced_latent_loss.item())
                mtp_reward_losses.append(dyn_pred_reward_loss.item())
                mtp_termination_losses.append(dyn_pred_termination_loss.item())
                successor_flow_losses.append(successor_flow_loss.item())
                successor_feature_reward_losses.append(successor_feature_reward_loss.item())
                successor_feature_termination_losses.append(successor_feature_termination_loss.item())
                if run_diagnostics:
                    with torch.no_grad():
                        reward_pred = reward_support.to_scalar(pred_reward_logits)
                        termination_pred = (torch.sigmoid(pred_termination_logits) >= 0.5).float()
                        for horizon_idx in range(args.dyn_horizon):
                            horizon_weight = latent_weight[:, horizon_idx]
                            denom = horizon_weight.sum().clamp_min(1.0)
                            reward_mse = (
                                (reward_pred[:, horizon_idx] - future_rewards[:, horizon_idx]).square()
                                * horizon_weight
                            ).sum() / denom
                            termination_acc = (
                                (termination_pred[:, horizon_idx] == future_terminations[:, horizon_idx]).float()
                                * horizon_weight
                            ).sum() / denom
                            dyn_reward_mses.append(reward_mse.item())
                            dyn_termination_accs.append(termination_acc.item())
                        obs_pred_mse = (
                            F.mse_loss(
                                pred_next_latents[:, :, :NUM_OBS_TOKENS],
                                target_next_latents[:, :, :NUM_OBS_TOKENS].detach(),
                                reduction="none",
                            ).mean(dim=(-1, -2))
                            * latent_weight
                        ).sum() / latent_weight.sum().clamp_min(1.0)
                        outcome_pred_mse = (
                            F.mse_loss(
                                pred_next_latents[:, :, NUM_OBS_TOKENS:],
                                target_next_latents[:, :, NUM_OBS_TOKENS:].detach(),
                                reduction="none",
                            ).mean(dim=(-1, -2))
                            * latent_weight
                        ).sum() / latent_weight.sum().clamp_min(1.0)
                        lejepa_obs_pred_mses.append(obs_pred_mse.item())
                        lejepa_outcome_pred_mses.append(outcome_pred_mse.item())
                dyn_sigreg_loss = masked_token_sigreg(target_future_summaries, latent_weight > 0.0)
                lejepa_loss = args.dyn_latent_coef * dyn_latent_loss + args.sigreg_coef * dyn_sigreg_loss
                outcome_probe_loss = (
                    args.dyn_reward_coef * dyn_reward_loss
                    + args.dyn_termination_coef * dyn_termination_loss
                    + args.successor_flow_coef * successor_flow_loss
                    + args.successor_feature_reward_coef * successor_feature_reward_loss
                    + args.successor_feature_termination_coef * successor_feature_termination_loss
                )
                wm_loss = lejepa_loss + outcome_probe_loss

                optimizer.zero_grad()
                wm_loss.backward()
                clip_grad_groups()
                optimizer.step()
                agent.update_successor_flow_target(args.successor_flow_target_decay)

                dyn_losses.append(wm_loss.item())
                dyn_latent_losses.append(dyn_latent_loss.item())
                dyn_reward_losses.append(dyn_reward_loss.item())
                dyn_termination_losses.append(dyn_termination_loss.item())
                dyn_sigreg_losses.append(dyn_sigreg_loss.item())
                lejepa_losses.append(lejepa_loss.item())
                teacher_forced_latent_losses.append(teacher_forced_latent_loss.item())

        reward_probe_error = None
        value_probe_error = None
        rollout_probe_rewards = None
        rollout_probe_values = None
        if run_diagnostics:
            with torch.inference_mode():
                reward_target_probs = reward_support.project(rewards.reshape(-1))
                target_continues = (1.0 - transition_terminations).reshape(-1)
                target_summaries = agent.encode_summary_with_outcomes(
                    next_obses.reshape((-1,) + envs.single_observation_space.shape),
                    reward_target_probs,
                    target_continues,
                )
                probe_reward_logits, _ = agent.decode_outcomes(target_summaries)
                rollout_probe_rewards = reward_support.to_scalar(probe_reward_logits).reshape(
                    args.num_steps,
                    args.num_envs,
                )
                rollout_probe_values = agent.successor_value(
                    target_summaries,
                    reward_support,
                    args.gamma,
                    samples=args.successor_value_samples,
                    steps=args.successor_flow_steps,
                    chunk_size=args.successor_value_chunk_size,
                ).reshape(
                    args.num_steps,
                    args.num_envs,
                )
                reward_probe_error = rollout_probe_rewards - rewards
                value_probe_error = rollout_probe_values - next_state_returns

        dyn_diagnostics = {}
        if run_diagnostics:
            dyn_diagnostics = dynamics_diagnostics(
                b_obs,
                b_prev_reward_probs,
                b_prev_continues,
                rewards,
                transition_actions,
                transition_terminations,
                transition_boundaries,
                transition_valids,
            )

        # v124: imagined SPO is gone. The actor is updated via the inner-Adam
        # action solver + amortization regression inside the real-rollout loop
        # below. We keep the dream-diagnostics scalars defined for back-compat
        # with the writer block at the bottom of the iteration, but they stay
        # empty.

        # Agent phase on a frozen world-model interface. Real rollouts provide
        # the PPO anchor and the start states for the inner-Adam action solver.
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        action_clipfracs = []
        real_approx_kls = []
        real_cleanrl_approx_kls = []
        real_logratio_abs_means = []
        real_logratio_max_abses = []
        real_action_logratio_abs_means = []
        real_actor_stds = []
        dream_clipfracs = []
        imagine_actor_losses = []
        imagine_actor_returns = []
        imagine_critic_losses = []
        imagine_old_approx_kls = []
        imagine_approx_kls = []
        imagine_cleanrl_approx_kls = []
        imagine_logratio_abs_means = []
        imagine_logratio_max_abses = []
        imagine_action_logratio_abs_means = []
        dream_action_clipfracs = []
        imagine_action_sat_fracs = []
        imagine_actor_mean_abs_means = []
        imagine_actor_mean_max_abses = []
        imagine_raw_actor_beta_head_abs_means = []
        imagine_raw_actor_beta_head_max_abses = []
        imagine_actor_stds = []
        real_spo_penalties = []
        imagine_spo_penalties = []
        inner_initial_costs = []
        inner_final_costs = []
        inner_cost_deltas = []
        amortize_losses = []
        amortize_action_errors = []
        # v133 PrefPoE diagnostics (entropy/KL/concentration trackers)
        prefpoe_pref_losses = []
        prefpoe_cons_losses = []
        prefpoe_main_entropies = []
        prefpoe_pref_entropies = []
        prefpoe_fused_entropies = []
        prefpoe_kl_fused_main = []
        prefpoe_pref_conc_sums = []
        prefpoe_pref_conc_max = []
        pg_loss = None
        v_loss = None
        entropy_loss = None
        old_approx_kl = None
        approx_kl = None
        dream_diagnostics = {}
        dream_diagnostic_values = {}
        imagine_explained_var = None
        imagine_explained_vars = []
        real_return_log_values = []
        imagination_return_log_values = []

        if not world_model_only:
            for epoch in range(args.agent_update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    real_return_log_values.append((real_minibatch_step, b_returns[mb_inds].mean().detach()))
                    real_minibatch_step += 1

                    mb_latents = b_agent_latents[mb_inds].detach()
                    planner_behavior_update = (
                        args.planner_execute
                        and args.planner_amortize_behavior
                        and global_step >= args.planner_start_step
                    )
                    planner_skip_amortize = (
                        args.planner_execute
                        and not args.planner_amortize_behavior
                        and global_step >= args.planner_start_step
                    )
                    mb_planner_executed = b_planner_executed[mb_inds] > 0.5
                    if planner_behavior_update and bool(mb_planner_executed.all()):
                        # In official-aligned solver mode, the behavior action
                        # has already been chosen by the WM gradient solver at
                        # collection time. Amortize the logged action instead of
                        # rerunning a now-stale inner solve after the WM update.
                        a_star_first = b_actions[mb_inds].detach()
                        inner_initial_cost = None
                        inner_final_cost = None
                    elif planner_skip_amortize and bool(mb_planner_executed.all()):
                        # v139: after MPC owns behavior, do not train the actor
                        # to chase planner actions. A dummy target keeps the
                        # tensor path simple; sample weights below zero the loss.
                        a_star_first = b_actions[mb_inds].detach()
                        inner_initial_cost = None
                        inner_final_cost = None
                    else:
                        # v124 path: run the inner-Adam action solver BEFORE
                        # building any outer-graph losses. freeze_agent_grad /
                        # restore_agent_grad toggle requires_grad on the
                        # agent's leaves; running the solver first guarantees
                        # every loss graph below is built under the fully
                        # restored requires_grad state.
                        with torch.no_grad():
                            init_dist = agent._action_distribution(
                                agent._actor_features_from_latents(mb_latents)
                            )
                            if args.inner_solver_init == "policy_sample":
                                init_action_z = init_dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                            else:
                                init_action_z = init_dist.mean.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                            init_action = agent._z_to_action(init_action_z)
                        init_action_seq = init_action.unsqueeze(1).expand(
                            -1, args.inner_solver_horizon, -1
                        ).contiguous()
                        a_star, inner_initial_cost, inner_final_cost = run_inner_action_solver(
                            mb_latents, init_action_seq
                        )
                        a_star_first = a_star[:, 0]
                        if planner_behavior_update and bool(mb_planner_executed.any()):
                            a_star_first = a_star_first.clone()
                            a_star_first[mb_planner_executed] = b_actions[mb_inds][
                                mb_planner_executed
                            ].detach()

                    # v132 PrefPoE: gate fusion behind warmup. lambda_pref_now=0
                    # short-circuits `_action_distributions` to (main, main, main),
                    # so PPO ratio / entropy reduce to v124 behavior during warmup.
                    prefpoe_active = global_step >= args.prefpoe_warmup_steps
                    lambda_pref_now = args.lambda_pref if prefpoe_active else 0.0
                    planner_pg_weight = torch.ones_like(b_planner_executed[mb_inds])
                    if args.planner_execute and args.planner_disable_pg:
                        planner_pg_weight = (~mb_planner_executed).float()
                    planner_pg_denom = planner_pg_weight.sum().clamp_min(1.0)
                    real_spo_coef_now = args.real_spo_coef
                    prefpoe_w_pref_now = args.prefpoe_w_pref
                    prefpoe_w_cons_now = args.prefpoe_w_cons
                    _, _, newlogprob, entropy = agent.get_action_logprob_entropy_from_latents(
                        b_agent_latents[mb_inds],
                        b_actions[mb_inds],
                        b_action_zs[mb_inds],
                        sum_logprob=False,
                        lambda_pref=lambda_pref_now,
                    )
                    action_logratio = newlogprob - b_logprobs[mb_inds]
                    action_ratio = action_logratio.exp()
                    logratio = action_logratio.sum(1)
                    ratio = logratio.exp()
                    action_approx_kl = ((action_ratio - 1) - action_logratio).sum(1)

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = action_approx_kl.mean()
                        cleanrl_approx_kl = ((ratio - 1) - logratio).mean()
                        real_approx_kls.append(approx_kl.item())
                        real_cleanrl_approx_kls.append(cleanrl_approx_kl.item())
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                        action_clipfracs += [
                            ((action_ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]
                        real_logratio_abs_means.append(logratio.abs().mean().item())
                        real_logratio_max_abses.append(logratio.abs().max().item())
                        real_action_logratio_abs_means.append(action_logratio.abs().mean().item())
                        real_actor_stds.append(
                            agent.get_imagined_action_std(b_agent_latents[mb_inds]).mean().item()
                        )

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std(unbiased=False) + 1e-8
                        )

                    ratio_diff = ratio - 1.0
                    spo_eps = torch.where(
                        (mb_advantages * ratio_diff) > 0,
                        torch.full_like(mb_advantages, args.spo_eps_high),
                        torch.full_like(mb_advantages, args.spo_eps_low),
                    )
                    spo_penalty = mb_advantages.abs() * ratio_diff.square() / (2.0 * spo_eps)
                    pg_terms = mb_advantages * ratio - spo_penalty
                    pg_loss = -((pg_terms * planner_pg_weight).sum() / planner_pg_denom)
                    real_spo_penalties.append(
                        ((spo_penalty * planner_pg_weight).sum() / planner_pg_denom).item()
                    )

                    value_logits = agent.get_value_logits_from_latents(b_agent_latents[mb_inds])
                    return_probs = hl_support.project(b_returns[mb_inds])
                    v_loss = -(
                        return_probs.detach() * torch.log_softmax(value_logits, dim=-1)
                    ).sum(dim=-1).mean()
                    entropy_loss = entropy.sum(1).mean()

                    # v124 amortization: regress MAIN actor mean toward optimized
                    # first-step action from the inner solver above. v132 keeps
                    # this on the main head only (WM-Jacobian signal is separate
                    # from the advantage-guided preference signal).
                    actor_features_mb = agent._actor_features_from_latents(mb_latents)
                    main_dist_mb, pref_dist_mb, fused_dist_mb = agent._action_distributions(
                        actor_features_mb, lambda_pref_now
                    )
                    actor_mean = agent._z_to_action(main_dist_mb.mean)
                    amortize_per_sample = F.mse_loss(
                        actor_mean,
                        a_star_first.detach(),
                        reduction="none",
                    ).mean(dim=-1)
                    amortize_weight = torch.ones_like(amortize_per_sample)
                    if planner_skip_amortize:
                        amortize_weight = (~mb_planner_executed).float()
                    amortize_denom = amortize_weight.sum().clamp_min(1.0)
                    amortize_loss = (amortize_per_sample * amortize_weight).sum() / amortize_denom

                    amortize_active = (
                        global_step >= args.wm_warmup_steps + args.amortize_warmup_steps
                    )
                    amortize_coef_now = args.amortize_coef if amortize_active else 0.0

                    # v135 PrefPoE losses (entropy reg → linear concentration penalty):
                    #   L_pref = −β_1·E[A_norm·log π_pref(a_z|s)]
                    #            − α·H(π_pref)                           (off by default)
                    #            + c_conc·E[(α_pref + β_pref − 2)]       (v135 new)
                    #   L_cons = KL(π_fused ‖ π_pref)
                    # The (α+β−2) penalty replaces the paper's H term because Beta
                    # entropy gradient vanishes at high concentration while linear
                    # penalty stays constant — needed to prevent v132-style runaway.
                    if (
                        prefpoe_active
                        and prefpoe_w_pref_now > 0.0
                        and bool((planner_pg_weight > 0.0).any())
                    ):
                        action_z_mb = b_action_zs[mb_inds].clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                        pref_logprob_per_dim = pref_dist_mb.log_prob(action_z_mb)
                        pref_logprob = pref_logprob_per_dim.sum(dim=-1)
                        pref_loss = -args.prefpoe_beta1 * (
                            mb_advantages.detach() * pref_logprob * planner_pg_weight
                        ).sum() / planner_pg_denom
                        if args.prefpoe_alpha_entropy > 0.0:
                            pref_entropy_term = (
                                pref_dist_mb.entropy().sum(dim=-1) * planner_pg_weight
                            ).sum() / planner_pg_denom
                            pref_loss = pref_loss - args.prefpoe_alpha_entropy * pref_entropy_term
                        if args.prefpoe_concentration_penalty > 0.0:
                            conc_excess = (
                                (pref_dist_mb.concentration1 - 1.0)
                                + (pref_dist_mb.concentration0 - 1.0)
                            ).sum(dim=-1)
                            conc_excess = (conc_excess * planner_pg_weight).sum() / planner_pg_denom
                            pref_loss = pref_loss + args.prefpoe_concentration_penalty * conc_excess
                    else:
                        pref_loss = torch.zeros((), device=mb_advantages.device)

                    if (
                        prefpoe_active
                        and prefpoe_w_cons_now > 0.0
                        and bool((planner_pg_weight > 0.0).any())
                    ):
                        cons_per_sample = torch.distributions.kl.kl_divergence(
                            fused_dist_mb, pref_dist_mb
                        ).sum(dim=-1)
                        cons_loss = (cons_per_sample * planner_pg_weight).sum() / planner_pg_denom
                    else:
                        cons_loss = torch.zeros((), device=mb_advantages.device)

                    loss = (
                        real_spo_coef_now * pg_loss
                        - args.ent_coef * entropy_loss
                        + args.vf_coef * v_loss
                        + amortize_coef_now * amortize_loss
                        + prefpoe_w_pref_now * pref_loss
                        + prefpoe_w_cons_now * cons_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    clip_grad_groups()
                    optimizer.step()

                    if inner_initial_cost is not None:
                        inner_initial_costs.append(inner_initial_cost.mean().item())
                        inner_final_costs.append(inner_final_cost.mean().item())
                        inner_cost_deltas.append((inner_final_cost - inner_initial_cost).mean().item())
                    amortize_losses.append(amortize_loss.item())
                    amortize_action_errors.append((actor_mean - a_star_first).abs().mean().item())
                    # v132 PrefPoE diagnostics (only meaningful once warmup elapsed
                    # and lambda_pref > 0; during warmup main_dist_mb == pref_dist_mb
                    # == fused_dist_mb so the entropies/KL collapse to identity).
                    prefpoe_pref_losses.append(pref_loss.item())
                    prefpoe_cons_losses.append(cons_loss.item())
                    with torch.no_grad():
                        prefpoe_main_entropies.append(main_dist_mb.entropy().sum(dim=-1).mean().item())
                        prefpoe_pref_entropies.append(pref_dist_mb.entropy().sum(dim=-1).mean().item())
                        prefpoe_fused_entropies.append(fused_dist_mb.entropy().sum(dim=-1).mean().item())
                        prefpoe_kl_fused_main.append(
                            torch.distributions.kl.kl_divergence(fused_dist_mb, main_dist_mb)
                            .sum(dim=-1).mean().item()
                        )
                        pref_conc_sum = (
                            pref_dist_mb.concentration1 + pref_dist_mb.concentration0
                        )
                        prefpoe_pref_conc_sums.append(pref_conc_sum.mean().item())
                        prefpoe_pref_conc_max.append(pref_conc_sum.max().item())

        # (v124: dream SPO block removed)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if real_return_log_values:
            real_steps, real_values = zip(*real_return_log_values)
            for log_step, log_value in zip(real_steps, torch.stack(real_values).cpu().tolist()):
                writer.add_scalar("returns/real_target", log_value, log_step)
        if imagination_return_log_values:
            imagination_steps, imagination_values = zip(*imagination_return_log_values)
            for log_step, log_value in zip(imagination_steps, torch.stack(imagination_values).cpu().tolist()):
                writer.add_scalar("returns/imagination_target", log_value, log_step)

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/world_model_only", float(world_model_only), global_step)
        if rollout_planner_execute_flags:
            writer.add_scalar("planner/execute_frac", safe_mean(rollout_planner_execute_flags), global_step)
        if rollout_goal_action_initial_distances:
            writer.add_scalar(
                "goal_solver/action_initial_distance",
                safe_mean(rollout_goal_action_initial_distances),
                global_step,
            )
            writer.add_scalar(
                "goal_solver/action_final_distance",
                safe_mean(rollout_goal_action_final_distances),
                global_step,
            )
            writer.add_scalar(
                "goal_solver/action_distance_delta",
                safe_mean(
                    [
                        final_distance - initial_distance
                        for initial_distance, final_distance in zip(
                            rollout_goal_action_initial_distances,
                            rollout_goal_action_final_distances,
                        )
                    ]
                ),
                global_step,
            )
        if rollout_goal_final_energies:
            writer.add_scalar("goal_solver/initial_energy", safe_mean(rollout_goal_initial_energies), global_step)
            writer.add_scalar("goal_solver/final_energy", safe_mean(rollout_goal_final_energies), global_step)
            writer.add_scalar("goal_solver/model_return", safe_mean(rollout_goal_returns), global_step)
            writer.add_scalar("goal_solver/value", safe_mean(rollout_goal_values), global_step)
            writer.add_scalar("goal_solver/defect", safe_mean(rollout_goal_defects), global_step)
        if rollout_exec_action_abs_deltas:
            writer.add_scalar(
                "goal_solver/goal_action_abs_delta",
                safe_mean(rollout_goal_action_abs_deltas),
                global_step,
            )
            writer.add_scalar(
                "goal_solver/exec_action_abs_delta",
                safe_mean(rollout_exec_action_abs_deltas),
                global_step,
            )
            writer.add_scalar(
                "goal_solver/exec_action_l2_delta",
                safe_mean(rollout_exec_action_l2_deltas),
                global_step,
            )
            writer.add_scalar(
                "goal_solver/exec_action_grad_norm",
                safe_mean(rollout_exec_action_grad_norms),
                global_step,
            )
        if v_loss is not None:
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("ratio/threshold_frac", np.mean(clipfracs), global_step)
            if real_approx_kls:
                writer.add_scalar("losses/approx_kl_update_mean", np.mean(real_approx_kls), global_step)
                writer.add_scalar("losses/approx_kl_update_max", np.max(real_approx_kls), global_step)
            if real_cleanrl_approx_kls:
                writer.add_scalar("losses/cleanrl_approx_kl", real_cleanrl_approx_kls[-1], global_step)
                writer.add_scalar(
                    "losses/cleanrl_approx_kl_update_mean", np.mean(real_cleanrl_approx_kls), global_step
                )
                writer.add_scalar(
                    "losses/cleanrl_approx_kl_update_max", np.max(real_cleanrl_approx_kls), global_step
                )
            if action_clipfracs:
                writer.add_scalar("ratio/action_threshold_frac", np.mean(action_clipfracs), global_step)
            if real_logratio_abs_means:
                writer.add_scalar("real_rollout/logratio_abs_mean", np.mean(real_logratio_abs_means), global_step)
                writer.add_scalar("real_rollout/logratio_max_abs", np.mean(real_logratio_max_abses), global_step)
                writer.add_scalar(
                    "real_rollout/action_logratio_abs_mean", np.mean(real_action_logratio_abs_means), global_step
                )
                writer.add_scalar("real_rollout/actor_std_mean", np.mean(real_actor_stds), global_step)
            if real_spo_penalties:
                writer.add_scalar("losses/spo_penalty", np.mean(real_spo_penalties), global_step)
        if inner_initial_costs:
            writer.add_scalar("inner_solver/initial_cost", np.mean(inner_initial_costs), global_step)
            writer.add_scalar("inner_solver/final_cost", np.mean(inner_final_costs), global_step)
            writer.add_scalar("inner_solver/cost_delta", np.mean(inner_cost_deltas), global_step)
        if amortize_losses:
            writer.add_scalar("losses/amortize_loss", np.mean(amortize_losses), global_step)
            writer.add_scalar("amortize/action_error_abs_mean", np.mean(amortize_action_errors), global_step)
        if prefpoe_pref_losses:
            writer.add_scalar("prefpoe/pref_loss", np.mean(prefpoe_pref_losses), global_step)
            writer.add_scalar("prefpoe/cons_loss", np.mean(prefpoe_cons_losses), global_step)
            writer.add_scalar("prefpoe/main_entropy", np.mean(prefpoe_main_entropies), global_step)
            writer.add_scalar("prefpoe/pref_entropy", np.mean(prefpoe_pref_entropies), global_step)
            writer.add_scalar("prefpoe/fused_entropy", np.mean(prefpoe_fused_entropies), global_step)
            writer.add_scalar("prefpoe/kl_fused_main", np.mean(prefpoe_kl_fused_main), global_step)
            writer.add_scalar("prefpoe/pref_conc_sum_mean", np.mean(prefpoe_pref_conc_sums), global_step)
            writer.add_scalar("prefpoe/pref_conc_sum_max", np.max(prefpoe_pref_conc_max), global_step)
            writer.add_scalar(
                "prefpoe/lambda_pref_now",
                args.lambda_pref if global_step >= args.prefpoe_warmup_steps else 0.0,
                global_step,
            )
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        dyn_loss_mean = safe_mean(dyn_losses)
        dyn_latent_loss_mean = safe_mean(dyn_latent_losses)
        dyn_reward_loss_mean = safe_mean(dyn_reward_losses)
        dyn_termination_loss_mean = safe_mean(dyn_termination_losses)
        successor_flow_loss_mean = safe_mean(successor_flow_losses)
        successor_feature_reward_loss_mean = safe_mean(successor_feature_reward_losses)
        successor_feature_termination_loss_mean = safe_mean(successor_feature_termination_losses)
        pred_reward_decode_loss_mean = safe_mean(pred_reward_decode_losses)
        pred_termination_decode_loss_mean = safe_mean(pred_termination_decode_losses)
        dyn_sigreg_loss_mean = safe_mean(dyn_sigreg_losses)
        lejepa_loss_mean = safe_mean(lejepa_losses)
        writer.add_scalar("losses/dyn_loss", dyn_loss_mean, global_step)
        writer.add_scalar("losses/lejepa_loss", lejepa_loss_mean, global_step)
        writer.add_scalar("losses/dyn_latent_loss", dyn_latent_loss_mean, global_step)
        writer.add_scalar("losses/outcome_probe_reward_loss", dyn_reward_loss_mean, global_step)
        writer.add_scalar("losses/outcome_probe_termination_loss", dyn_termination_loss_mean, global_step)
        writer.add_scalar("losses/dyn_sigreg_loss", dyn_sigreg_loss_mean, global_step)
        writer.add_scalar("lejepa/loss", lejepa_loss_mean, global_step)
        writer.add_scalar("lejepa/prediction_loss", dyn_latent_loss_mean, global_step)
        writer.add_scalar("lejepa/prediction_mse", safe_mean(teacher_forced_latent_losses), global_step)
        if lejepa_obs_pred_mses:
            writer.add_scalar("lejepa/obs_prediction_mse", safe_mean(lejepa_obs_pred_mses), global_step)
            writer.add_scalar("lejepa/outcome_prediction_mse", safe_mean(lejepa_outcome_pred_mses), global_step)
        writer.add_scalar("lejepa/sigreg_loss", dyn_sigreg_loss_mean, global_step)
        writer.add_scalar("dynamics/loss", dyn_loss_mean, global_step)
        writer.add_scalar("dynamics/reward_probe_loss", dyn_reward_loss_mean, global_step)
        writer.add_scalar("dynamics/pred_reward_loss", pred_reward_decode_loss_mean, global_step)
        writer.add_scalar("dynamics/mtp_latent_loss", safe_mean(mtp_latent_losses), global_step)
        writer.add_scalar("dynamics/mtp_reward_loss", safe_mean(mtp_reward_losses), global_step)
        writer.add_scalar("dynamics/mtp_termination_loss", safe_mean(mtp_termination_losses), global_step)
        writer.add_scalar("dynamics/successor_flow_loss", successor_flow_loss_mean, global_step)
        writer.add_scalar("dynamics/successor_feature_reward_loss", successor_feature_reward_loss_mean, global_step)
        writer.add_scalar(
            "dynamics/successor_feature_termination_loss",
            successor_feature_termination_loss_mean,
            global_step,
        )
        if dyn_reward_mses:
            writer.add_scalar("dynamics/reward_mse", safe_mean(dyn_reward_mses), global_step)
        if reward_probe_error is not None:
            writer.add_scalar("dynamics/real_probe_reward_bias", reward_probe_error.mean().item(), global_step)
            writer.add_scalar("dynamics/real_probe_reward_mae", reward_probe_error.abs().mean().item(), global_step)
            writer.add_scalar("dynamics/real_probe_reward_mean", rollout_probe_rewards.mean().item(), global_step)
            writer.add_scalar("dynamics/real_probe_value_bias", value_probe_error.mean().item(), global_step)
            writer.add_scalar("dynamics/real_probe_value_mae", value_probe_error.abs().mean().item(), global_step)
            writer.add_scalar("dynamics/real_probe_value_mean", rollout_probe_values.mean().item(), global_step)
        writer.add_scalar("dynamics/env_reward_mean", rewards.mean().item(), global_step)
        writer.add_scalar("dynamics/termination_probe_loss", dyn_termination_loss_mean, global_step)
        writer.add_scalar("dynamics/pred_termination_loss", pred_termination_decode_loss_mean, global_step)
        if dyn_termination_accs:
            writer.add_scalar("dynamics/termination_accuracy", safe_mean(dyn_termination_accs), global_step)
        for key, value in dyn_diagnostics.items():
            writer.add_scalar(f"dynamics/{key}", value, global_step)
        if imagine_critic_losses:
            writer.add_scalar("losses/imagine_critic_loss", np.mean(imagine_critic_losses), global_step)
        if imagine_actor_losses:
            writer.add_scalar("losses/imagine_actor_loss", np.mean(imagine_actor_losses), global_step)
        if imagine_spo_penalties:
            writer.add_scalar("losses/imagine_spo_penalty", np.mean(imagine_spo_penalties), global_step)
        if imagine_actor_returns:
            writer.add_scalar("imagination/returns", np.mean(imagine_actor_returns), global_step)
        if imagine_explained_var is not None:
            writer.add_scalar("losses/imagine_explained_variance", imagine_explained_var, global_step)
        writer.add_scalar("losses/real_rollout_explained_variance", explained_var, global_step)
        for key, diagnostic_values in dream_diagnostic_values.items():
            writer.add_scalar(f"imagination/{key}", safe_mean(diagnostic_values), global_step)
        if imagine_approx_kls:
            writer.add_scalar("losses/imagine_old_approx_kl", np.mean(imagine_old_approx_kls), global_step)
            writer.add_scalar("losses/imagine_approx_kl", np.mean(imagine_approx_kls), global_step)
            writer.add_scalar("ratio/imagine_threshold_frac", np.mean(dream_clipfracs), global_step)
        if imagine_cleanrl_approx_kls:
            writer.add_scalar("losses/imagine_cleanrl_approx_kl", np.mean(imagine_cleanrl_approx_kls), global_step)
            writer.add_scalar("ratio/imagine_action_threshold_frac", np.mean(dream_action_clipfracs), global_step)
        if imagine_logratio_abs_means:
            writer.add_scalar("imagination/logratio_abs_mean", np.mean(imagine_logratio_abs_means), global_step)
            writer.add_scalar("imagination/logratio_max_abs", np.mean(imagine_logratio_max_abses), global_step)
            writer.add_scalar(
                "imagination/action_logratio_abs_mean", np.mean(imagine_action_logratio_abs_means), global_step
            )
            writer.add_scalar("imagination/action_saturation_frac", np.mean(imagine_action_sat_fracs), global_step)
            writer.add_scalar("imagination/actor_mean_abs_mean", np.mean(imagine_actor_mean_abs_means), global_step)
            writer.add_scalar("imagination/actor_mean_max_abs", np.mean(imagine_actor_mean_max_abses), global_step)
            writer.add_scalar(
                "imagination/raw_actor_beta_head_abs_mean",
                np.mean(imagine_raw_actor_beta_head_abs_means),
                global_step,
            )
            writer.add_scalar(
                "imagination/raw_actor_beta_head_max_abs",
                np.mean(imagine_raw_actor_beta_head_max_abses),
                global_step,
            )
            writer.add_scalar("imagination/actor_std_mean", np.mean(imagine_actor_stds), global_step)
        writer.add_scalar("charts/imagined_steps", imagined_steps, global_step)
        writer.add_scalar("charts/imagined_learnable_steps", imagined_learnable_steps, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar(
            "charts/real_plus_imagined_SPS", int((global_step + imagined_steps) / (time.time() - start_time)), global_step
        )

    envs.close()
    writer.close()
