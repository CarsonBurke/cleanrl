# PPO + IterThink v24_beta + DELIGHTFUL POLICY GRADIENT (DG, arXiv:2603.14608v1).
#
# ============================ DG variant (v6) =================================
# v6 = faithful score-mode DG + DECOUPLED EPOCHS. The faithful regime (dg_mode=score,
# update_epochs=1, raw advantages, no PPO clip) FROZE: at 300k explained_variance~0.005,
# actor_grad_norm collapsed 0.45->0.04, approx_kl->0.001, episodic_return -255->-480. The
# gate was healthy (sign_agree=1.0, surprisal_mean~1, sat_frac~0); the failure was the
# CRITIC -- a single distributional-CE pass cannot fit the 511-bin head, so EV~0 makes the
# raw-GAE advantages ~noise, the gated score gradient cancels, and the actor stalls.
# FIX (keeps actor faithful): `--actor-epochs 1 --update-epochs 10` trains the actor for ONE
# on-policy score pass (no ratio/clip, raw advantage -- exactly the paper's update) while the
# critic refits for 10 epochs against the FIXED buffer target. Critic regression has no
# importance ratio and no off-policy bias, so multi-epoch critic fitting does NOT compromise
# the faithful single-pass actor; it just restores EV so the advantages carry signal.
# Critic-only epochs use agent.get_value (no actor forward) and a single clipped critic backward.
#
# ============================ DG variant (v5) =================================
# Built on the canonical iterthink_v24_beta base (ppo_continuous_action_iterthink_v24_dist.py).
# The actor update applies the paper's detached delight gate:
#
#   chi = U * ell  ("delight");  w = sigmoid(chi / eta), eta=1;  gate DETACHED.
#   gate_ppo mode: loss = stopgrad(w) * PPO_clipped_surrogate   (the trust-region on-policy port).
#
# v5 HEADLINE -- HYBRID surprisal (`--dg-surprisal hybrid`, DEFAULT): ell = 0.5*(ell_policy + ell_critic),
# the AVERAGE of the v3 policy action-tail and the v4 critic outcome-tail. Motivated by the v3-vs-v4
# A/B at matched steps: CRITIC surprisal warms up FASTER (@500k 1662 > policy 1480, beating base) but
# FADES as the critic calibrates (F_V(G)->Uniform => outcome signal -> quantile noise); POLICY surprisal
# scales BETTER late (@2M 5503 > critic 4670, matching base @4M 7872 vs 7869). The two channels decay at
# DIFFERENT times, so averaging keeps a live rarity signal throughout: outcome-rarity early, action-rarity
# late. Both terms are >= 0 (=> sign(chi)=sign(U) preserved, proposition-faithful) and E~1 when responsive,
# so the mean stays E~1 and eta=1 remains in the sigmoid's graded band. Up-weights samples rare in EITHER
# channel. `--dg-surprisal {hybrid,critic,cdf_tail,mode_ref,raw_clip,entref}` selects.
#
# v4 HEADLINE -- CRITIC (outcome) surprisal vs POLICY (action) surprisal.
# The "delight" needs a surprisal ell >= 0. There are TWO sources:
#   POLICY surprisal (v3 cdf_tail): how rare was the ACTION under pi -- ell = -log(2 min(F_a,1-F_a)).
#     Weakness ON-POLICY: a fresh draw's quantile F_a ~ Uniform, so ell is ~sampling noise; the
#     paper escapes this via off-policy replay (re-scoring old actions under a drifted policy).
#   CRITIC surprisal (v4 DEFAULT): how rare was the OUTCOME under the value model. Our critic is
#     DISTRIBUTIONAL, so the realized lambda-return G has a predicted CDF F_V; we use the same
#     tail form  ell_V = -log(2 min(F_V(G), 1-F_V(G))) >= 0. The realized return carries genuine
#     signal regardless of sampling, so critic surprisal is the better fit for the ON-POLICY
#     regime. >= 0 (sign(chi)=sign(U) preserved); E~1 when the critic is calibrated (same
#     responsive scale as policy cdf_tail); computed once from the rollout buffer (~free,
#     constant across epochs). Note chi = U * ell_V correlates with |U| (large |advantage| =>
#     return in the predicted tail), so the gate emphasizes confident large-advantage outcomes.
#   `--dg-surprisal {critic,cdf_tail,mode_ref,raw_clip,entref}` selects; rest of the gate is the
#   v3 machinery unchanged. The empirical A/B is critic vs cdf_tail on the strong base pipeline
#   (where policy cdf_tail already reached ~4486 @2M, the best DG result so far).
#
# The gate w in (0,1) reallocates gradient budget toward RARE SUCCESSES and away from RARE
# FAILURES (from maximizing chi*w + eta*H(w)). It is DETACHED so only the surrogate carries grad.
#
# v3 HEADLINE -- CDF-TAIL surprisal (`--dg-surprisal cdf_tail`, DEFAULT). The action surprisal
# is the true two-sided tail mass (p-value):
#   ell = MEAN_dims -log( 2 * min(F(a_i), 1 - F(a_i)) ),   F = Beta CDF (regularized inc. beta).
#   (MEAN not SUM over the 6 dims: per-dim E[ell]~1, so eta=1 stays in the gate's responsive band;
#    summing gives E[ell]~6 and saturates the gate -- the failure the experts flagged.)
# This is the expert-recommended measure. Lineage of the surprisal choice:
#   v1 raw_clip ell=clip(-logpi): for bounded Beta the density routinely EXCEEDS 1, so -logpi<0
#     ~65% of the time -> chi=U*ell flips sign -> gate INVERTS (sign_agree ~0.3). Broken.
#   v2 mode_ref ell=logpi(mode)-logpi(a)>=0: fixes the sign (sign_agree=1.0), but it measures
#     log-density DISTANCE FROM THE MODE, which for a fresh on-policy draw is a monotone function
#     of the action's own SAMPLING QUANTILE -- not its probabilistic rarity -- and it is
#     skew-biased (over-penalizes the long tail of an asymmetric Beta ~2x). On the strong base it
#     acted like quantile-correlated sample-dropout and underperformed.
#   v3 cdf_tail: ell >= 0 (=> sign(chi)=sign(U), proposition-faithful), ZERO at the median, and
#     -- unlike mode_ref -- CONCENTRATION-INVARIANT (a p99 action scores the same whether the Beta
#     is broad or razor-sharp, so eta=1 stays calibrated for the whole run) and MASS-SYMMETRIC
#     (equal-probability tails get equal surprisal). It is the literal "how rare was this action"
#     signal the paper's discrete propositions assume. min(F,1-F) is floored at 1e-6 (=> ell<=~13,
#     a principled cap on an already-nonnegative quantity, never a sign flip). The Beta CDF has no
#     torch builtin; a vectorized Lentz continued fraction (beta_cdf, validated <2e-3 vs MC) is
#     used, detached, and only in the update (compute_cdf), so rollout SPS is unaffected.
#   Also fixed a latent NaN: a_c can round to exactly 1.0 in fp32 (softplus underflow) making the
#   mode 0/0; the denominator is now floored. `mode_ref`/`raw_clip`/`entref` remain as ablations.
#   eta=1 fixed; U not standardized; no whiten/EMA/renorm.
#
# TWO REGIMES. The gate hovering near 0.5 is the paper's regime too (gentle reweighting), NOT a
# bug to engineer away. The paper's STABILITY comes from a trust region -- Retrace + target nets
# bound the off-policy update -- which is why eta=1 is robust over many gradient steps. Pure
# on-policy "score" mode discards that trust region, which is exactly why we saw KL explode at
# many epochs and the actor freeze at one. The faithful on-policy port keeps the trust region:
#   --dg-mode score     faithful Algorithm 2 (no ratio/clip) -- the user-requested literal test.
#   --dg-mode gate_ppo  the detached gate multiplies PPO's clipped surrogate; PPO's ratio-clip
#                       IS the on-policy implementation of the paper's KL trust region.
#
# HYPOTHESIS. Prior DG work here (only on the heavier d4hlgauss_symlog stack) dropped PPO's trust
# region and underperformed; the clean v24_beta base was never tested. If DG matches/beats base,
# the mechanism transfers to dense on-policy control; if score mode underperforms but gate_ppo
# matches base, it localizes the gain to the trust region rather than chi=U*ell. `--dg-enable
# False` reproduces the base byte-for-byte.
# =============================================================================
#
# SOFT MAX-ENTROPY add-on (--auto-entropy, gaussian only). This borrows SAC's
# tanh-squashed log-prob, target-entropy heuristic, and temperature dual, but keeps
# the PPO critic on the RAW reward return. Entropy enters the actor two ways:
#   (1) a current-state squashed-entropy actor bonus, -alpha * log pi_sq(a|s);
#   (2) a policy-only soft GAE whose one-step bootstrap adds alpha * H_sq(s_{t+1})
#       using the rollout/bootstrapped squashed log-prob sample.
# The distributional critic target is deliberately entropy-free so the fixed support
# remains calibrated. Off (default) => byte-identical to the v24 base.
#
# WHY v24. The v22/v23 state-dependent Gaussian std hit a 1/sigma^2 pathology
# (confident low-sigma states spike the mean gradient). dreamer4 avoids this two
# ways, and v24 ports BOTH faithfully behind one `--actor-dist` toggle, on the
# UNCHANGED v21 winner machinery (shared backbone, 2-way decoupled clip,
# rankgauss, clip-higher, tkl03) so the ONLY thing that varies is the action
# distribution — a clean A/B.
#
#   actor_dist="beta"  (DEFAULT, the "performs much better" path):
#       unimodal Beta, exactly dreamer4's continuous_dist_type='beta' (which
#       forces unimodal=True) and our beta_relusq:
#           alpha = 1 + softplus(head_a);  beta = 1 + softplus(head_b)   (>=1 => unimodal)
#       native support (0,1) is linearly rescaled to the env action range
#       [low, high]. Sampling clamps z to [eps, 1-eps]; log_prob/entropy are the
#       closed-form Beta values in native z-space (the constant rescale Jacobian
#       is dropped — it cancels in the PPO ratio and the entropy is a constant
#       offset). Bounded support => no squash saturation, no 1/sigma^2 blow-up,
#       no boundary mass leak, no bang-bang (unimodal).
#
#   actor_dist="gaussian"  (the matched control = state-dependent Gaussian scale):
#       dreamer4's Gaussian readout. This is NOT SAC's exact log-std head. It is a
#       state-dependent log-VARIANCE head (not a flat Parameter, not log-std),
#       SOFT-bounded by dreamer4's tanh-rescale (not a hard clamp, so the gradient
#       never dies at the bound):
#           lv  = rescale(tanh(raw_lv/(hi-lo)), (-1,1), (lo,hi))   # symmetric (lo,hi) => lv=0 at init
#           std = exp(0.5 * lv)
#       then the iterthink/SAC tanh-squash + stable Jacobian on the sample (mean
#       stays raw). SAC continuous-action instead uses a state-dependent log_std
#       head bounded to [-5, 2] and std = exp(log_std). Here logvar [-8, 8] implies
#       log_std [-4, 4], so the family matches but the scale parameterization and
#       bounds do not.
#
# PARITY NOTES (both dists): the rollout buffers the distribution-NATIVE sample
# (latent_zs) — pre-tanh z for gaussian, z in (0,1) for beta — and replays it on
# the update pass, so log_prob is recomputed at the same sample (identical to
# v21's z-replay). `actions` holds the env action (tanh(z) / rescaled z). The
# gaussian path is bit-identical to v21 except the flat logstd -> dreamer4 head.
# Bar to beat: v21 flat-Gaussian = 8774.
#
# --- inherited v21 notes ---
# PPO + IterThink v21 (SHARED BACKBONE + DECOUPLED GRAD CLIP). From v19.
#
# WHY v21. v19 used two independent ThinkTrunks (one actor, one critic). The
# classic MuJoCo-PPO result is that shared backbones LOSE, because the value
# loss gradient dominates the shared trunk and corrupts the policy's features.
# v21 tests whether we can have the representation-sharing benefit WITHOUT that
# cost, by decoupling the gradient magnitudes:
#   - share_backbone: one ThinkTrunk feeds both the actor head and the
#     (distributional) critic head; trunk is computed once per forward.
#   - separate_grad_clip: DUAL-BACKWARD clipping. The value gradient
#     (vf_coef * v_loss) and the policy gradient (pg_loss - ent) are each
#     backpropped and clipped to their OWN max-norm (critic_grad_clip /
#     actor_grad_clip), then summed on the shared trunk:
#         trunk.grad = clip_actor(d pg / d trunk) + clip_critic(d vl / d trunk)
#     so the distributional critic's large CE gradient can no longer swamp the
#     shared features. NOTE: the trunk's effective budget is the SUM of the two
#     clips, so each defaults to 0.25 (sum ~= v19's single 0.5 global clip).
# This is targeted: rankgauss already bounds the POLICY gradient (rank-only adv),
# so the dominant imbalance on a shared trunk is the critic -> clip it apart.
# Built on the v19 winner: adv_transform="rankgauss" + clip-higher (0.2/0.28).
# Both knobs are toggles, so this file also runs the {shared,separate} x
# {global,decoupled-clip} 2x2. The bar to beat: rankgauss_cliphigh ~= 8292 (towers).
#
# --- inherited v19 notes ---
# PPO + IterThink v19 (ADVANTAGE SHAPING — magnitude-preserving + attribution). From v17.
#
# WHY v19. A subagent review of v17 (CDF-rank distributional PG) found that in its
# STABLE regime the categorical critic is overconfident, so u=F_Z(G) is bimodal at
# 0/1, the probit saturates, and the advantage DEGENERATES to ≈sign(GAE) (corr 0.92);
# norm_adv then re-standardizes the ±3.3 spikes to ≈±1 binary. So v17 discards the
# advantage MAGNITUDE (the thing PPO needs) and is really a sign-of-TD-error update
# made trainable by KL control. v17's 5867@4M conflates THREE possible causes — the
# distribution, a bounded/outlier-robust advantage, and KL control — introduced at
# once. v19 disentangles them and adds the principled fix, via one `adv_transform`:
#
#   "v10"      : raw GAE (== v10 / dist_pg off). Baseline.
#   "cdf_probit": v17's CDF-rank u -> Phi^-1(u). Reference.
#   "tanh_std" : A~ = tanh( GAE_t / (kappa * sigma(s_t)) ).  THE FIX. Per-state
#                normalized by the critic's return std sigma(s) (v16's good idea),
#                but BOUNDED by tanh (fixes v16's blowup: tiny sigma -> saturate, not
#                explode) AND magnitude-preserving near 0 (fixes v17's sign-collapse:
#                linear in GAE for |GAE|<kappa*sigma). Note G_t-E[Z_t]=GAE_t exactly.
#   "tanh_gae" : A~ = tanh( zscore(GAE)_t / kappa ).  Robust-GAE CONTROL with NO
#                distribution — isolates "bounded/outlier-robust advantage" from the
#                distributional claim. If this matches v17, the distribution is
#                incidental and this is the cleaner lever.
#
# All paths keep the mean-value GAE and the distributional λ-return value target
# (v10) UNCHANGED; only the policy advantage is reshaped. sigma(s) is the std of the
# OLD rollout Z(s_t), floored at `sigma_floor_bins` bins. Pair with target_kl for the
# 2x2 attribution (v10/tanh_gae/cdf_probit x KL-cap). Control: v17 / v10.
import os
import random
import time
from dataclasses import dataclass
from math import log
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import HLGaussSupport

SAMPLE_EPS = 1e-6  # clamp Beta samples off the open-interval boundary (avoid log(0))


def rescale(t, old_range, new_range):
    # Linear map between ranges, matching dreamer4's discrete_continuous_embed_readout.rescale.
    old_min, old_max = old_range
    new_min, new_max = new_range
    return (t - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


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

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8000000
    learning_rate: float = 3e-4
    num_envs: int = 16
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10           # CRITIC epochs (distributional regression; also the loop bound)
    actor_epochs: int = -1            # ACTOR epochs; -1 => follow update_epochs (coupled). Set 1 for a
    #   faithful single on-policy actor pass while the critic still refits update_epochs times
    #   (decoupled v6 fix: a 1-epoch critic gives explained_variance~0 and starves the advantages).
    norm_adv: bool = True
    clip_coef: float = 0.2           # PPO clip (lower bound 1-clip_coef; also upper if no high)
    clip_coef_high: float = 0.28     # "clip-higher" (DAPO): looser UPPER bound 1+clip_coef_high
    ent_coef: float = 0.0
    # --- Delightful Policy Gradient (DG, arXiv:2603.14608v1) ---
    dg_enable: bool = True            # master switch; False => byte-identical to the v24_beta base
    dg_use_gate: bool = True          # False (with dg_enable) => w=1: plain REINFORCE/score control
    #   in the SAME regime, isolating the gate (the paper's DG-vs-REINFORCE comparison)
    dg_mode: str = "score"            # "score" (faithful Alg.2: no ratio/clip) | "gate_ppo" (gate x PPO surrogate)
    dg_surprisal: str = "hybrid"      # "hybrid" (0.5*(policy+critic) tail, v5) | "critic" (outcome tail, v4) | "cdf_tail" (action tail) | "mode_ref" | "raw_clip" | "entref"
    dg_eta: float = 1.0               # temperature eta in w = sigmoid(chi/eta)
    dg_clip: float = 10.0             # symmetric clip on the surprisal ell/ell~
    # SCALE the surprisal (not chi) by a RUNNING/EMA std so eta=1 is meaningful while
    # PRESERVING the chi=0 anchor and keeping chi=U*ell~ -> 0 as U -> 0 (relax to PG near
    # optimum). This is the principled scale fix; normalize ell, never chi.
    dg_surprisal_norm: bool = False   # (ablation only; not in the paper) EMA-std surprisal scale
    # DISCREDITED: batch-whitening chi mean-centers (destroys the absolute breakthrough/
    # blunder anchor -> rank-within-batch, batch-coupled) AND keeps chi/sigma ~ O(1) as
    # U->0 (amplifies late-training gate noise). Kept only for ablation; prefer dg_surprisal_norm.
    dg_whiten_chi: bool = False       # (discouraged) per-minibatch whiten of chi
    dg_renorm: bool = False           # divide loss by mean(w) to preserve effective step size (off = faithful)
    # SOFT-ADVANTAGE max-ent (gaussian only; --auto-entropy). Entropy enters the POLICY
    # ADVANTAGE only, NEVER the critic's target. The critic regresses to the RAW reward
    # return (fits the fixed support [v_min,v_max]); a SEPARATE soft-advantage GAE adds the
    # bootstrap bonus b_t = α·H_sq(s_{t+1}), estimated as the single-sample SQUASHED entropy
    # -logπ_sq(a|s), matching SAC's next_state_log_pi units. Rationale: forcing the bounded
    # categorical critic to learn the soft value both wastes capacity and overflows the
    # support (the softboot failure: edge_mass→0.9, expl_var→0). alpha is
    # auto-tuned by SAC's exact dual (target_entropy = -|A|; alpha_loss = -α(logπ + tgt)).
    # The explicit α·H actor bonus supplies the CURRENT-step entropy gradient (the soft
    # advantage's current-state entropy is action-independent => cancels in the PG term);
    # the soft advantage supplies FUTURE-entropy credit. Works WITH rankgauss: the soft
    # value reorders advantages and rankgauss preserves order/sign (magnitude is incidental).
    auto_entropy: bool = False
    target_entropy: Optional[float] = None  # SAC heuristic default = -|A| (act_dim)
    alpha_lr: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    target_kl: float = 0.03          # KL early-stop leash (the iterthink-line winner; shared trunk runs hot without it)

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # Distributional critic support. Tight + well-resolved (vs v9's ±20/255).
    num_bins: int = 511
    v_min: float = -10.0
    v_max: float = 10.0
    critic_init_tau: float = 0.5   # init Z ≈ N(0, tau^2), sharp at 0
    value_symlog: bool = False

    # Advantage shaping (v19). Selects how the policy advantage is formed from the
    # GAE advantage and/or the value distribution Z(s). See header.
    #   "v10" | "cdf_probit" | "tanh_std" | "tanh_gae" | "clip_z"
    #   "rankgauss" | "rankgauss_signed" | "rankgauss_temp" | "rankgauss_signmag"
    adv_transform: str = "rankgauss"

    # Scope ablations: where advantage shaping / normalization are computed.
    #   adv_transform_scope: "batch" (default) ranks/shapes over the whole rollout
    #     once per iteration; "minibatch" recomputes the shaping within each
    #     minibatch (the idiomatic scope of norm_adv). Only matters for shaping
    #     transforms (rankgauss, ...); "v10" is identity so scope is moot.
    #   norm_adv_scope: "minibatch" (default, idiomatic PPO) standardizes each
    #     minibatch; "batch" standardizes once over the whole rollout.
    adv_transform_scope: str = "batch"
    norm_adv_scope: str = "minibatch"

    # v24: action distribution. "beta" (unimodal, dreamer4 default) | "gaussian"
    # (dreamer4 state-dependent log-VARIANCE head, not SAC log_std; soft tanh-rescale bound).
    actor_dist: str = "beta"
    logvar_min: float = -8.0         # gaussian: soft log-var bound (symmetric => std=1 at init)
    logvar_max: float = 8.0          # std in [exp(-4), exp(4)] = [0.018, 54.6]

    tanh_kappa: float = 2.0          # tanh saturation scale (in per-state std units)
    sigma_floor_bins: float = 2.0    # floor sigma(s) at this many bins (avoid /~0)
    clip_z_c: float = 2.0            # Winsorize bound for "clip_z" (in std units)
    rank_tanh_kappa: float = 1.5     # tanh temperature for "rankgauss_temp" (<inf=harder)
    pos_neg_alpha: float = 0.5       # PMPO-style: weight +adv by 2a, -adv by 2(1-a); 0.5=off
    # CDF/probit knobs (used by "cdf_probit" and "rankgauss").
    cdf_probit_clamp: float = 0.999

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

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
    if hasattr(layer, "bias") and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


def _branch_body(H):
    # Plain pre-act MLP. Standard sqrt(2) init on both Linears.
    return nn.Sequential(
        layer_init(nn.Linear(H, H)),
        ReLUSquared(),
        layer_init(nn.Linear(H, H)),
    )


class ThinkBlock(nn.Module):
    """Bounded convex residual mix of x and x0, then parallel dense + soft MoE."""

    def __init__(self, in_dim, H, n_experts):
        super().__init__()
        self.n_experts = n_experts

        self.in_proj = layer_init(nn.Linear(in_dim, H))
        # Bounded convex residual gate: g = sigmoid(resid_gate) per channel.
        # init +4 → g ≈ 0.982 → x_in ≈ x at start.
        self.resid_gate = nn.Parameter(torch.full((H,), 4.0))

        # Dense branch. RMSNorm without learnable affine (no free per-channel γ).
        self.dense_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.dense = _branch_body(H)

        # Soft MoE branch (softmax over all experts, no top-K).
        self.moe_norm = nn.RMSNorm(H, elementwise_affine=False)
        self.gate = layer_init(nn.Linear(H, n_experts))
        self.experts = nn.ModuleList([_branch_body(H) for _ in range(n_experts)])

    def forward(self, cat_feats, x0):
        x = self.in_proj(cat_feats)                                   # (B, H)
        g = torch.sigmoid(self.resid_gate)                            # (H,)
        x_in = g * x + (1.0 - g) * x0                                 # (B, H), convex

        d_dense = self.dense(self.dense_norm(x_in))                   # (B, H)

        m_in = self.moe_norm(x_in)
        weights = torch.softmax(self.gate(m_in), dim=-1)              # (B, E)
        all_out = torch.stack([e(m_in) for e in self.experts], dim=1) # (B, E, H)
        d_moe = (weights.unsqueeze(-1) * all_out).sum(dim=1)          # (B, H)

        return x_in + d_dense + d_moe


class ThinkTrunk(nn.Module):
    def __init__(self, in_dim, H, K, n_experts):
        super().__init__()
        self.entry = layer_init(nn.Linear(in_dim, H))
        self.blocks = nn.ModuleList()
        for k in range(K):
            block_in_dim = H * (k + 1)
            self.blocks.append(ThinkBlock(block_in_dim, H, n_experts))
        cat_dim = H * (K + 1)
        self.out_norm = nn.RMSNorm(cat_dim, elementwise_affine=False)
        self.out_proj = layer_init(nn.Linear(cat_dim, H))

    def forward(self, x):
        x0 = self.entry(x)
        feats = [x0]
        for block in self.blocks:
            feats.append(block(torch.cat(feats, dim=-1), x0))
        return self.out_proj(self.out_norm(torch.cat(feats, dim=-1)))


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        H = args.hidden
        self.share_backbone = args.share_backbone
        if self.share_backbone:
            # One trunk feeds both heads (computed once per forward).
            self.trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        else:
            self.critic_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
            self.actor_trunk = ThinkTrunk(obs_dim, H, args.k_blocks, args.n_experts)
        # Categorical critic with a PEAKED init: small weight + Gaussian-logit
        # bias so the initial value distribution is sharp at 0 (not uniform),
        # preventing the distributional-bootstrap blowup that sank v9.
        self.critic_head = layer_init(nn.Linear(H, args.num_bins), std=0.1)
        with torch.no_grad():
            z = torch.linspace(args.v_min, args.v_max, args.num_bins)
            self.critic_head.bias.copy_(-0.5 * (z / args.critic_init_tau) ** 2)
        # v24: action distribution. Both parameterizations are dreamer4-faithful;
        # the Gaussian path is tanh-squashed like SAC but uses log-variance, not log_std.
        self.actor_dist = args.actor_dist
        if self.actor_dist == "gaussian":
            # dreamer4 Gaussian: mean head + state-dependent log-VARIANCE head.
            self.actor_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.actor_logvar_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.logvar_min, self.logvar_max = args.logvar_min, args.logvar_max
        elif self.actor_dist == "beta":
            # dreamer4 unimodal Beta: two concentration heads, alpha,beta = 1 + softplus.
            self.actor_alpha_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.actor_beta_head = layer_init(nn.Linear(H, act_dim), std=0.01)
            self.register_buffer(
                "action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32)
            )
            self.register_buffer(
                "action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32)
            )
        else:
            raise ValueError(f"unknown actor_dist {self.actor_dist}")

    def _actor_dist(self, actor_feat):
        # Build the action distribution and the native-space transforms.
        # Returns (dist, to_action, log_det_fn) where:
        #   to_action(z): map a NATIVE sample z to the env action.
        #   log_det_fn(z): per-sample log|d action / d z| correction to SUBTRACT
        #                  from dist.log_prob(z) (0 where the map is volume-constant).
        if self.actor_dist == "gaussian":
            mean = self.actor_head(actor_feat)
            raw_lv = self.actor_logvar_head(actor_feat)
            # dreamer4 soft tanh-rescale bound on log-variance (smooth, no dead grad).
            lv = rescale(
                (raw_lv / (self.logvar_max - self.logvar_min)).tanh(),
                (-1.0, 1.0),
                (self.logvar_min, self.logvar_max),
            )
            std = (0.5 * lv).exp()
            dist = Normal(mean, std)
            to_action = torch.tanh
            log_det_fn = lambda z: 2.0 * (log(2.0) - z - F.softplus(-2.0 * z))
            return dist, to_action, log_det_fn
        # beta
        alpha = 1.0 + F.softplus(self.actor_alpha_head(actor_feat))
        beta = 1.0 + F.softplus(self.actor_beta_head(actor_feat))
        dist = Beta(alpha, beta)
        to_action = lambda z: self.action_low + (self.action_high - self.action_low) * z
        log_det_fn = lambda z: 0.0  # constant linear rescale: drops out of the PPO ratio
        return dist, to_action, log_det_fn

    def _trunks(self, x):
        # Return (actor_feat, critic_feat). When sharing, the SAME trunk output
        # is fed to both heads, computed only once.
        if self.share_backbone:
            feat = self.trunk(x)
            return feat, feat
        return self.actor_trunk(x), self.critic_trunk(x)

    def get_value(self, x):
        # Returns value LOGITS (B, num_bins); caller converts via support.
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat)

    def get_action_and_value(self, x, z=None, compute_cdf=False):
        # z is the distribution-NATIVE sample (pre-tanh for gaussian; in (0,1) for
        # beta). When replaying from the buffer it is passed back in; log_prob is
        # recomputed at the same native sample (v21's z-replay, generalized).
        # compute_cdf=True (update only) also returns the CDF-tail surprisal; the
        # incomplete-beta CF is skipped during rollout to keep SPS up.
        actor_feat, critic_feat = self._trunks(x)
        dist, to_action, log_det_fn = self._actor_dist(actor_feat)
        if z is None:
            z = dist.sample()
            if self.actor_dist == "beta":
                z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        log_prob = (dist.log_prob(z) - log_det_fn(z)).sum(1)
        value_logits = self.critic_head(critic_feat)
        if self.actor_dist == "gaussian":
            # Reparameterized SQUASHED-entropy estimate H_sq = E_ε[-logπ_sq(tanh(μ+σε))].
            # Base-Normal H = dist.entropy() is monotone↑ in σ, so an entropy bonus rails σ
            # to the ceiling -> tanh saturates -> squashed H collapses, while the α-dual
            # (which targets squashed H) cranks α up: a runaway. The squashed H is BOUNDED
            # with an interior max in σ, so maximizing it settles σ at a finite optimum and
            # is consistent with the α target. Fresh rsample => gradient flows to μ,σ
            # (independent of the replayed z used for the PPO ratio).
            zr = dist.rsample()
            entropy = (dist.log_prob(zr) - log_det_fn(zr)).sum(1).neg()
        else:
            entropy = dist.entropy().sum(1)
        # Peak (mode) log-density per sample, for the DG mode-referenced surprisal
        # ell = logp(mode) - logp(a) >= 0 -- the bounded-Beta analog of the Gaussian's
        # Mahalanobis tail distance. The constant action-rescale Jacobian cancels in the
        # difference, so the raw native-space dist.log_prob is used (no log_det term).
        cdf_ell = None
        if self.actor_dist == "beta":
            a_c, b_c = dist.concentration1, dist.concentration0  # alpha, beta (both > 1)
            # NaN-safe mode: a_c can round to EXACTLY 1.0 in fp32 (softplus underflow) so
            # (a_c-1)/(a_c+b_c-2) -> 0/0; floor the denominator (clamp does NOT repair NaN).
            mode = ((a_c - 1.0) / (a_c + b_c - 2.0).clamp_min(SAMPLE_EPS)).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            logp_peak = dist.log_prob(mode).sum(1)
            if compute_cdf:
                cdf_ell = dg_cdf_tail_surprisal(z, dist)
        else:  # gaussian: mode at the mean (mode_ref is intended for beta)
            logp_peak = dist.log_prob(dist.mean).sum(1)
        return action, z, log_prob, entropy, value_logits, logp_peak, cdf_ell

    def actor_parameters(self):
        # Params receiving the POLICY gradient (incl. the shared trunk). The two
        # distribution heads are clipped together as one actor group (2-way
        # decoupled clip; no separate std budget — gaussian's variance head and
        # both beta concentration heads sit in the same group).
        trunk = self.trunk if self.share_backbone else self.actor_trunk
        if self.actor_dist == "gaussian":
            heads = list(self.actor_head.parameters()) + list(self.actor_logvar_head.parameters())
        else:
            heads = list(self.actor_alpha_head.parameters()) + list(self.actor_beta_head.parameters())
        return list(trunk.parameters()) + heads

    def critic_parameters(self):
        # Params receiving the VALUE gradient (incl. the shared trunk).
        trunk = self.trunk if self.share_backbone else self.critic_trunk
        return list(trunk.parameters()) + list(self.critic_head.parameters())


def categorical_project(probs, atoms, support, v_min, v_max, bin_width):
    """C51 projection: distribute `probs` (B, n) sitting at positions `atoms`
    (B, n) onto the fixed `support` (n,) via linear interpolation between
    neighbouring bins. Mass-preserving (atoms are pre-clamped to [v_min, v_max]).
    """
    n = support.shape[0]
    tz = atoms.clamp(v_min, v_max)
    b = (tz - v_min) / bin_width                       # (B, n) fractional bin pos
    lo = b.floor()
    hi = b.ceil()
    lo_idx = lo.clamp(0, n - 1).long()
    hi_idx = hi.clamp(0, n - 1).long()
    m = torch.zeros_like(probs)
    m.scatter_add_(1, lo_idx, probs * (hi - b))        # mass to lower bin
    m.scatter_add_(1, hi_idx, probs * (b - lo))        # mass to upper bin
    # When b is integer, lo == hi and both weights are 0; (1 - (hi - lo)) == 1
    # routes the full mass to that bin. Otherwise hi - lo == 1 → adds 0.
    m.scatter_add_(1, lo_idx, probs * (1.0 - (hi - lo)))
    return m


def distributional_lambda_returns(
    rewards, dones, next_done, value_probs, bootstrap_probs, support, v_min, v_max, bin_width, gamma, gae_lambda
):
    """Backward recursion for the distributional λ-return G^λ (probs per step).

        G^λ_t =_D r_t + γ·nonterm·[ (1-λ)·Z(s_{t+1}) + λ·G^λ_{t+1} ]

    Mean-matches the scalar GAE λ-return. Shapes: rewards/dones (T, B);
    value_probs (T, B, n); bootstrap_probs (B, n) = Z(s_T). Returns (T, B, n).
    Entropy/soft-value terms are NOT injected here — the critic regresses to the raw
    reward return; max-ent enters the policy advantage separately (see --auto-entropy).
    """
    T = rewards.shape[0]
    target = torch.zeros_like(value_probs)
    g_next = bootstrap_probs                            # G^λ_{T} ≡ bootstrap
    for t in reversed(range(T)):
        if t == T - 1:
            nonterminal = 1.0 - next_done               # (B,)
            z_next = bootstrap_probs                    # Z(s_T)
        else:
            nonterminal = 1.0 - dones[t + 1]
            z_next = value_probs[t + 1]                 # Z(s_{t+1})
        mix = (1.0 - gae_lambda) * z_next + gae_lambda * g_next          # (B, n)
        gn = (gamma * nonterminal).unsqueeze(-1)        # (B, 1)
        atoms = rewards[t].unsqueeze(-1) + gn * support  # (B, n) transformed atoms
        g_next = categorical_project(mix, atoms, support, v_min, v_max, bin_width)
        target[t] = g_next
    return target


def shape_advantage(gae, sigma, u, args, device):
    """Map raw GAE -> policy advantage per args.adv_transform. Works on a full
    batch or a single minibatch (sigma/u must be sliced to match gae)."""
    if args.adv_transform == "v10":
        return gae
    elif args.adv_transform == "tanh_std":
        # Per-state-normalized, bounded, magnitude-preserving near 0 (THE FIX).
        return torch.tanh(gae / (args.tanh_kappa * sigma))
    elif args.adv_transform == "tanh_gae":
        gz = (gae - gae.mean()) / (gae.std() + 1e-8)
        return torch.tanh(gz / args.tanh_kappa)
    elif args.adv_transform == "cdf_probit":
        centered = (2.0 * u - 1.0)
        c = args.cdf_probit_clamp
        return ((2.0 ** 0.5) * torch.erfinv(centered.clamp(-c, c).cpu())).to(device)
    elif args.adv_transform == "clip_z":
        # Winsorized z-score: ONE standardize + hard tail clip. tanh's hard cousin;
        # preserves bulk magnitude exactly (linear in [-c,c]) and is ~self-normalizing
        # (clipping a unit-var signal barely changes its std), so norm_adv ~ no-op.
        gz = (gae - gae.mean()) / (gae.std() + 1e-8)
        return gz.clamp(-args.clip_z_c, args.clip_z_c)
    elif args.adv_transform == "rankgauss":
        # Rank-Gaussian: the principled single op. Replaces each advantage by the
        # Gaussian quantile of its empirical rank -> self-normalizing (bounded,
        # zero-mean, ~unit-scale, fully outlier-immune; no kappa, no double z-score).
        # This is cdf_probit with the EMPIRICAL batch rank instead of the critic's
        # per-state CDF (distribution-free; the distribution was shown incidental).
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)  # 0..n-1
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        centered = (2.0 * uq - 1.0).clamp(-c, c)
        return ((2.0 ** 0.5) * torch.erfinv(centered.cpu())).to(device)
    elif args.adv_transform == "rankgauss_signed":
        # Sign-anchored rank-Gaussian: rank positives and negatives SEPARATELY so the
        # advantage's zero-crossing is preserved exactly. Plain rankgauss anchors the
        # sign flip at the batch MEDIAN, not 0, so on skewed GAE it can assign the
        # wrong sign to near-median samples — and PPO needs the sign right. Each sign
        # group is mapped to its half of the Gaussian by its within-group rank.
        c = args.cdf_probit_clamp
        out = torch.zeros_like(gae)
        for side in (gae > 0, gae < 0):
            if side.any():
                g = gae[side]
                r = g.argsort().argsort().to(torch.float32)
                half = (r + 0.5) / float(g.numel())                  # (0,1) in group
                uq = torch.where(g > 0, 0.5 + 0.5 * half, 0.5 * half)  # correct side
                ctr = (2.0 * uq - 1.0).clamp(-c, c)
                out[side] = ((2.0 ** 0.5) * torch.erfinv(ctr.cpu())).to(device)
        return out
    elif args.adv_transform == "rankgauss_temp":
        # Rank-Gaussian + tanh temperature: compress the Gaussian quantiles' extremes
        # harder (rank_tanh_kappa < inf). Probes the "more robust still" axis the kappa
        # sweep motivated (tanh_gae kappa=1 > kappa=2). Smaller kappa => harder.
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        centered = (2.0 * uq - 1.0).clamp(-c, c)
        z = ((2.0 ** 0.5) * torch.erfinv(centered.cpu())).to(device)
        return torch.tanh(z / args.rank_tanh_kappa)
    elif args.adv_transform == "rankgauss_signmag":
        # Sign-correct WITHOUT count distortion: take plain rankgauss's GLOBAL-rank
        # magnitude, then force the sign to match the raw advantage. Fixes the flaw in
        # rankgauss_signed (per-group half-Gaussian over-amplifies the minority sign by
        # COUNT); here magnitude still reflects global rank extremity and only the ~9%
        # near-zero "flips" get re-signed. Nonlinear (not a shift) => survives norm_adv.
        n = gae.numel()
        ranks = gae.argsort().argsort().to(torch.float32)
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        mag = ((2.0 ** 0.5) * torch.erfinv((2.0 * uq - 1.0).clamp(-c, c).cpu())).to(device).abs()
        return torch.sign(gae) * mag
    else:
        raise ValueError(f"unknown adv_transform {args.adv_transform}")


def _betacf(x, a, b, iters=50):
    """Continued fraction for the regularized incomplete beta (Lentz). Vectorized,
    detached use only (the DG gate is stop-grad). Numerical Recipes betacf."""
    fpmin = 1e-30
    qab = a + b; qap = a + 1.0; qam = a - 1.0
    c = torch.ones_like(x)
    d = 1.0 - qab * x / qap
    d = torch.where(d.abs() < fpmin, torch.full_like(d, fpmin), d)
    d = 1.0 / d
    h = d.clone()
    for m in range(1, iters):
        m_ = float(m); m2 = 2.0 * m_
        aa = m_ * (b - m_) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = torch.where(d.abs() < fpmin, torch.full_like(d, fpmin), d); d = 1.0 / d
        c = 1.0 + aa / c; c = torch.where(c.abs() < fpmin, torch.full_like(c, fpmin), c)
        h = h * d * c
        aa = -(a + m_) * (qab + m_) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = torch.where(d.abs() < fpmin, torch.full_like(d, fpmin), d); d = 1.0 / d
        c = 1.0 + aa / c; c = torch.where(c.abs() < fpmin, torch.full_like(c, fpmin), c)
        h = h * d * c
    return h


def beta_cdf(x, a, b):
    """Regularized incomplete beta I_x(a,b) = Beta CDF. Validated to <2e-3 vs Monte Carlo."""
    x = x.clamp(1e-6, 1.0 - 1e-6)
    lbeta = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    bt = (a * x.log() + b * (1.0 - x).log() - lbeta).exp()
    use_lower = x < (a + 1.0) / (a + b + 2.0)
    lower = bt * _betacf(x, a, b) / a
    upper = 1.0 - bt * _betacf(1.0 - x, b, a) / b
    return torch.where(use_lower, lower, upper).clamp(0.0, 1.0)


def dg_cdf_tail_surprisal(z, dist):
    """Two-sided CDF-tail surprisal ell = MEAN_dims -log(2*min(F, 1-F)).
    The EXPERTS' recommended measure: true probabilistic rarity (mass in the tail),
    >= 0 (zero at the median), CONCENTRATION-INVARIANT (a p99 action scores the same
    whether the Beta is broad or razor-sharp), and MASS-SYMMETRIC (unlike mode_ref it
    does not over-penalize the long tail of a skewed Beta). Detached / stop-grad.

    MEAN over the 6 action dims (not sum): per-dim E[ell]~1, so the mean keeps E[ell]~1 and
    eta=1 stays in the sigmoid's responsive band. Summing instead gives E[ell]~6, which
    saturates the gate (gate_mean -> 0.2, chi_std -> 3-6) and destroys the graded reweighting
    -- the exact failure the experts warned about. The mean of concentration-invariant,
    nonnegative, median-zero per-dim terms preserves all three properties."""
    F = beta_cdf(z, dist.concentration1, dist.concentration0)
    tail = torch.minimum(F, 1.0 - F).clamp_min(1e-6)  # bounds per-dim ell <= -log(2e-6) ~ 13
    return (-(2.0 * tail).log()).mean(1)


def dg_critic_tail_surprisal(target_probs, value_probs, support):
    """CRITIC (outcome) surprisal: CDF-tail of the realized lambda-return under the rollout
    critic's PREDICTED return distribution.  ell_V = -log(2*min(F_V(G), 1-F_V(G))) >= 0.

    The critic analog of the policy cdf_tail: instead of "how rare was this ACTION under the
    policy", it asks "how rare was this OUTCOME under the value model" -- how far in the tail of
    the critic's predicted return distribution the actual return G landed. Unlike policy
    surprisal (whose value for a fresh on-policy draw is just the action's own sampling quantile
    ~ Uniform noise), the realized return carries genuine signal, so this is the surprisal that
    fits the ON-POLICY regime. >= 0 (sign(chi)=sign(U) preserved), and E~1 when the critic is
    calibrated (F_V(G) ~ Uniform) -- same responsive scale as policy cdf_tail. Computed once from
    buffer tensors (rollout-time critic), so it is constant across PPO epochs and ~free."""
    G = (support[None, :] * target_probs).sum(-1)                          # realized lambda-return (soft mean)
    F_V = (value_probs * (support[None, :] <= G[:, None]).float()).sum(-1)  # predicted CDF at G
    tail = torch.minimum(F_V, 1.0 - F_V).clamp_min(1e-6)                    # bounds ell_V <= -log(2e-6) ~ 13
    return -(2.0 * tail).log()


def dg_raw_surprisal(logprob, entropy, args, logp_peak=None, cdf_ell=None, critic_ell=None):
    """Raw (un-scaled, un-clipped) surprisal ell.

    `critic` (DEFAULT, v4): OUTCOME surprisal ell_V = -log(2*min(F_V(G),1-F_V(G))), the tail of
    the realized return under the critic's predicted return distribution. Carries on-policy
    signal (the action-quantile noise that afflicts policy surprisal does not apply to outcomes).
    `cdf_tail` (v3): POLICY surprisal ell = -log(2*min(F(a),1-F(a))) -- true action tail mass.
    `mode_ref` (v2): ell = logp(mode) - logp(a) >= 0; tracks the action's SAMPLING QUANTILE,
    skew-biased on asymmetric Beta.
    `raw_clip`: paper-literal ell = -logp -- INVERTS the gate for Beta (density > 1 => -logp<0).
    `entref`: ell~ = -logp - H -- mean-zero per state but also breaks ell>=0."""
    if args.dg_surprisal == "hybrid":
        # average of policy action-tail and critic outcome-tail. Both >= 0 (sign preserved)
        # and both E~1 when responsive, so the mean stays E~1 -- eta=1 in the sigmoid band.
        # Captures rarity in EITHER channel: critic warms up faster early (outcome signal is
        # informative before the critic calibrates), policy scales better late (action-tail
        # stays informative after F_V(G)->Uniform); the average aims to keep both.
        return 0.5 * (cdf_ell + critic_ell)
    if args.dg_surprisal == "critic":
        return critic_ell
    if args.dg_surprisal == "cdf_tail":
        return cdf_ell
    if args.dg_surprisal == "mode_ref":
        return logp_peak - logprob  # logp(mode) - logp(a) >= 0
    if args.dg_surprisal == "entref":
        return -logprob - entropy
    return -logprob  # "raw_clip" -- paper-literal, zero at density=1


def delight_gate(advantages, surprisal_raw, args, ell_scale=1.0):
    """Delightful Policy Gradient gate w = sigmoid(chi / eta), chi = U * ell.

    Scales the SURPRISAL (not chi) by a running/EMA std `ell_scale` so eta=1 is meaningful
    while keeping the chi=0 anchor and chi -> 0 as U -> 0 (relax to PG near the optimum).
    Returns (gate, surprisal, chi) grad-attached; the CALLER detaches the gate.
    """
    surprisal = surprisal_raw
    if args.dg_surprisal_norm:
        surprisal = surprisal / (ell_scale + 1e-8)
    surprisal = surprisal.clamp(-args.dg_clip, args.dg_clip)
    chi = advantages * surprisal
    if args.dg_whiten_chi:  # discouraged: breaks the anchor & amplifies late-training noise
        chi = (chi - chi.mean()) / (chi.std(unbiased=False) + 1e-8)
    gate = torch.sigmoid(chi / args.dg_eta)
    return gate, surprisal, chi


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.adv_transform_scope in ("batch", "minibatch")
    assert args.norm_adv_scope in ("batch", "minibatch")
    # batch-scope norm reuses the batch-shaped advantage, so it needs batch-scope shaping.
    assert not (args.adv_transform_scope == "minibatch" and args.norm_adv_scope == "batch"), \
        "norm_adv_scope=batch requires adv_transform_scope=batch"
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
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()

    # Soft max-entropy temperature (gaussian only). log_alpha is learned so the
    # policy-only future-entropy bonus AND the actor entropy bonus self-tune to
    # hold the SQUASHED entropy at target_entropy via SAC's temperature dual.
    auto_alpha = args.auto_entropy and args.actor_dist == "gaussian"
    if auto_alpha:
        act_dim = int(np.prod(envs.single_action_space.shape))
        # SAC heuristic: target the squashed-policy entropy at -|A| (sac_continuous_action.py
        # target_entropy = -prod(action_space.shape)). The squashed entropy lives in the
        # tanh-Gaussian space (action ∈ [-1,1] => bounded, can be negative), so this is the
        # parity-correct target for the -logπ_squashed estimate (NOT base-Normal H).
        target_entropy = args.target_entropy if args.target_entropy is not None else -float(act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)

    hl_support = HLGaussSupport(
        args.num_bins,
        args.v_min,
        args.v_max,
        0.5,  # sigma_ratio unused (categorical Bellman target, no Gaussian projection)
        device,
        use_symlog=args.value_symlog,
    )
    support = hl_support.support                       # (num_bins,) linear support
    bin_width = hl_support.bin_width

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    value_probs = torch.zeros((args.num_steps, args.num_envs, args.num_bins)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    dg_ell_ema_std = None  # running std of the DG surprisal (only for the dg_surprisal_norm ablation)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, ent, value_logits, _, _ = agent.get_action_and_value(next_obs)
                p = torch.softmax(value_logits, dim=-1)
                value_probs[step] = p
                values[step] = (p * support).sum(dim=-1)
            actions[step] = action
            latent_zs[step] = z
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
            # SOFT-ADVANTAGE max-ent: entropy enters the POLICY ADVANTAGE only, NEVER the
            # critic's regression target. The bonus b_t = α·H_sq(s_{t+1}) is estimated with
            # a single squashed log-prob sample, in the same units as SAC's
            # next_state_log_pi. Making the bounded categorical critic *learn* it would
            # (a) waste its predictive capacity and (b) inflate the target off its fixed support
            # [v_min,v_max] (the softboot failure: edge_mass→0.9, expl_var→0). Instead the
            # critic regresses to the RAW reward return (control-proven to fit, edge_mass≈0)
            # and the entropy is added to a SEPARATE soft advantage used only for the PG.
            if auto_alpha:
                # Sample a' ~ π(·|s_T) for the bootstrap entropy (SAC's single-sample).
                _, _, boot_logprob, _, boot_logits, _, _ = agent.get_action_and_value(next_obs)
                bootstrap_probs = torch.softmax(boot_logits, dim=-1)            # (B, n) = Z(s_T)
                alpha_r = log_alpha.exp().detach()
                # b_t = α·H(s_{t+1}); H(s_{t+1}) = -logπ(a_{t+1}|s_{t+1}) from the rollout,
                # and H(s_T) = -logπ(a'|s_T) for the final bootstrap step.
                next_value_bonus = torch.zeros_like(rewards)
                next_value_bonus[:-1] = alpha_r * (-logprobs[1:])
                next_value_bonus[-1] = alpha_r * (-boot_logprob)
            else:
                bootstrap_probs = torch.softmax(agent.get_value(next_obs), dim=-1)   # (B, n) = Z(s_T)
                next_value_bonus = None
            next_value = (bootstrap_probs * support).sum(dim=-1).reshape(1, -1)
            # REWARD GAE: critic-consistent advantage + return (entropy-free => fits support).
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
            # SOFT-ADVANTAGE GAE (POLICY ONLY): same recursion with the entropy-augmented
            # next value V(s')+α·H(s'). Unbiased (state-dependent baseline); its large
            # magnitude is harmless — rankgauss at the optim step rank-normalizes it while
            # PRESERVING the entropy-induced reordering (the actual exploration signal).
            if auto_alpha:
                policy_adv = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * (nextvalues + next_value_bonus[t]) * nextnonterminal - values[t]
                    policy_adv[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            else:
                policy_adv = advantages
            # Critic target: RAW reward λ-return (entropy-free => no support overflow).
            target_probs = distributional_lambda_returns(
                rewards, dones, next_done, value_probs, bootstrap_probs,
                support, args.v_min, args.v_max, bin_width, args.gamma, args.gae_lambda,
            )
            # Per-state return std sigma(s_t) from the OLD rollout Z(s_t). Note
            # E[Z(s_t)] == values[t], so G_t - E[Z_t] == GAE advantage exactly.
            sigma = (value_probs * (support - values.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0).sqrt()  # (T,B)
            sigma = sigma.clamp_min(args.sigma_floor_bins * bin_width)
            # CDF-rank u (only used by the cdf_probit path; also a calibration probe).
            cdf_frac = ((returns.unsqueeze(-1) - support) / bin_width + 0.5).clamp(0.0, 1.0)  # (T,B,n)
            u = (value_probs * cdf_frac).sum(dim=-1)        # (T,B) CDF position
            u_edge_frac = ((u < 0.05) | (u > 0.95)).float().mean().item()  # calib probe (uniform≈0.1)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = policy_adv.reshape(-1)            # policy GAE (soft when auto_alpha; else raw)
        b_sigma = sigma.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_probs = target_probs.reshape(-1, args.num_bins)
        b_u = u.reshape(-1)
        # CRITIC (outcome) surprisal: tail of the realized return under the rollout critic's
        # predicted return distribution. Detached, constant across epochs (buffer quantity).
        if args.dg_enable and args.dg_surprisal in ("critic", "hybrid"):
            b_value_probs = value_probs.reshape(-1, args.num_bins)
            b_critic_ell = dg_critic_tail_surprisal(b_target_probs, b_value_probs, support)
        else:
            b_critic_ell = None
        # Policy advantage: shape the GAE per `adv_transform`. With
        # adv_transform_scope="minibatch" the shaping is deferred into the update
        # loop (recomputed per minibatch); the batch shaping below is then used
        # only for the diagnostics. norm_adv_scope="batch" standardizes the shaped
        # advantage once here instead of per minibatch.
        gae = b_advantages
        b_policy_adv = shape_advantage(gae, b_sigma, b_u, args, device)
        if args.norm_adv and args.norm_adv_scope == "batch":
            b_policy_adv_normed = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        # Diagnostics: how different is the shaped advantage from raw GAE?
        az = (gae - gae.mean()) / (gae.std() + 1e-8)
        pz = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        adv_corr = (az * pz).mean().item()
        adv_sign_agree = (torch.sign(az) == torch.sign(pz)).float().mean().item()

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        actor_epochs = args.actor_epochs if args.actor_epochs >= 0 else args.update_epochs
        for epoch in range(args.update_epochs):
            do_actor = epoch < actor_epochs
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if not do_actor:
                    # CRITIC-ONLY epoch (decoupled v6): refit the distributional critic against
                    # the FIXED buffer target (pure regression, zero off-policy bias -> no ratio/
                    # clip needed) while the faithful single-pass actor stays untouched. This is
                    # the fix for explained_variance~0 from a 1-epoch critic starving the GAE.
                    value_logits = agent.get_value(b_obs[mb_inds])
                    value_log_probs = torch.log_softmax(value_logits, dim=-1)
                    v_loss = -(b_target_probs[mb_inds] * value_log_probs).sum(dim=-1).mean()
                    optimizer.zero_grad(set_to_none=True)
                    (args.vf_coef * v_loss).backward()
                    if args.separate_grad_clip:
                        critic_gn = nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
                    else:
                        critic_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    continue

                _, _, newlogprob, entropy, value_logits, logp_peak, cdf_ell = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds], compute_cdf=(args.dg_enable and args.dg_surprisal in ("cdf_tail", "hybrid"))
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.adv_transform_scope == "minibatch":
                    mb_raw_adv = shape_advantage(b_advantages[mb_inds], b_sigma[mb_inds], b_u[mb_inds], args, device)
                else:
                    mb_raw_adv = b_policy_adv[mb_inds]
                mb_advantages = mb_raw_adv
                if args.norm_adv:
                    if args.norm_adv_scope == "batch":
                        mb_advantages = b_policy_adv_normed[mb_inds]
                    else:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # PMPO-style asymmetric pos/neg weighting: scale positive-advantage
                # samples by 2*alpha and negatives by 2*(1-alpha) (alpha=0.5 => identity).
                # alpha>0.5 emphasizes reinforcing good actions over suppressing bad ones.
                # Split on the SHAPED advantage's sign (pre-norm = the true advantage sign).
                if args.pos_neg_alpha != 0.5:
                    mb_advantages = mb_advantages * torch.where(
                        mb_raw_adv >= 0,
                        2.0 * args.pos_neg_alpha,
                        2.0 * (1.0 - args.pos_neg_alpha),
                    )

                # Asymmetric "clip-higher" when clip_coef_high is set: looser upper bound
                # gives positive-advantage actions more room to grow (counters collapse).
                clip_hi = args.clip_coef if args.clip_coef_high is None else args.clip_coef_high
                if args.dg_enable:
                    # Delightful Policy Gradient: gate each sample by w = sigmoid(chi/eta),
                    # chi = U * surprisal. Gate is DETACHED -> only the score / surrogate
                    # carries gradient (the gate reweights, it does not add its own term).
                    dg_surp_raw = dg_raw_surprisal(
                        newlogprob, entropy, args, logp_peak, cdf_ell,
                        critic_ell=(b_critic_ell[mb_inds] if b_critic_ell is not None else None),
                    )
                    if args.dg_surprisal_norm:
                        # Track the surprisal scale with a slow EMA (not a per-batch std):
                        # gives a stationary unit so eta=1 is meaningful without the batch
                        # coupling / late-noise amplification of whitening chi directly.
                        with torch.no_grad():
                            bstd = dg_surp_raw.std(unbiased=False).item()
                        dg_ell_ema_std = bstd if dg_ell_ema_std is None else (
                            args.dg_surprisal_norm_decay * dg_ell_ema_std
                            + (1.0 - args.dg_surprisal_norm_decay) * bstd
                        )
                    dg_scale = dg_ell_ema_std if (
                        args.dg_surprisal_norm and dg_ell_ema_std) else 1.0
                    dg_gate, dg_surprisal, dg_chi = delight_gate(
                        mb_advantages, dg_surp_raw, args, ell_scale=dg_scale
                    )
                    w = dg_gate.detach()
                    if not args.dg_use_gate:
                        # No-gate control: w=1 => plain REINFORCE/score (the paper's REINFORCE
                        # baseline) in the SAME regime, to isolate the gate's contribution.
                        w = torch.ones_like(w)
                    dg_sat_frac = ((w < 0.02) | (w > 0.98)).float().mean()
                    # sign(chi)==sign(U) agreement: ~1.0 => gate tracks breakthrough/blunder
                    # (paper's intent); ~0.5 => surprisal sign is scrambling the gate.
                    dg_sign_agree = ((dg_chi > 0) == (mb_advantages > 0)).float().mean()
                    if args.dg_mode == "score":
                        # Faithful Algorithm 2: gated score-function, no ratio, no clip.
                        per_sample = -(w * mb_advantages * newlogprob)
                    else:  # "gate_ppo": keep the PPO trust region, gate the surrogate.
                        s = torch.max(
                            -mb_advantages * ratio,
                            -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi),
                        )
                        per_sample = w * s
                    pg_loss = per_sample.mean()
                    if args.dg_renorm:
                        pg_loss = pg_loss / (w.mean() + 1e-8)
                else:
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Distributional value loss: cross-entropy to the (fixed)
                # distributional λ-return target. No value clipping.
                value_log_probs = torch.log_softmax(value_logits, dim=-1)
                v_loss = -(b_target_probs[mb_inds] * value_log_probs).sum(dim=-1).mean()

                entropy_loss = entropy.mean()

                if auto_alpha:
                    # SAC's temperature dual (sac_continuous_action.py), on the
                    # SQUASHED log-prob: alpha_loss = (-α·(logπ + target_entropy)).mean().
                    # With target_entropy=-|A|, drives E[logπ_squashed] -> |A|,
                    # equivalently E[-logπ_squashed] -> -|A|.
                    # The SAME α weights the explicit CURRENT-step actor entropy bonus below
                    # (the soft return's current-state entropy is action-independent => zero
                    # in the PG term, so the bonus supplies the actual entropy gradient).
                    ent_coef_eff = log_alpha.exp().detach()
                    alpha_loss = (-log_alpha.exp() * (newlogprob.detach() + target_entropy)).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                else:
                    ent_coef_eff = args.ent_coef

                if args.separate_grad_clip:
                    # DUAL-BACKWARD decoupled clipping. Backprop value and policy
                    # gradients separately, clip each to its own max-norm, then sum
                    # on the (possibly shared) trunk so the critic's CE gradient
                    # cannot swamp the policy's contribution to shared features.
                    optimizer.zero_grad(set_to_none=True)
                    (args.vf_coef * v_loss).backward(retain_graph=True)
                    critic_gn = nn.utils.clip_grad_norm_(critic_params, args.critic_grad_clip)
                    # Stash clipped value grads (must survive the zero before the
                    # policy backward; the shared trunk is in this set).
                    value_grads = [(p, p.grad.detach().clone()) for p in critic_params if p.grad is not None]
                    optimizer.zero_grad(set_to_none=True)
                    (pg_loss - ent_coef_eff * entropy_loss).backward()
                    actor_gn = nn.utils.clip_grad_norm_(actor_params, args.actor_grad_clip)
                    # trunk.grad currently = clip_actor(d pg / d trunk); add the
                    # stashed clip_critic(d vl / d trunk). critic_head gets value grad only.
                    for p, g in value_grads:
                        p.grad = g if p.grad is None else p.grad + g
                    optimizer.step()
                else:
                    loss = pg_loss - ent_coef_eff * entropy_loss + v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    critic_gn = actor_gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            # Early-stop only in the COUPLED regime; when decoupled (actor_epochs < update_epochs)
            # we must run all critic epochs, so never break on the actor's KL.
            if (args.target_kl is not None and actor_epochs >= args.update_epochs
                    and approx_kl > args.target_kl):
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Support-adequacy instrumentation: edge-bin mass should stay ≈ 0.
        edge_mass = (b_target_probs[:, 0] + b_target_probs[:, -1]).mean().item()

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        if args.dg_enable:
            # DG diagnostics (last minibatch). Healthy: gate_mean ~0.5 (anchor intact),
            # chi_std O(1) (gate active), sat_frac small (not pinned at 0/1) -- the expert's
            # saturation monitor. ell_scale is the EMA surprisal std used for normalization.
            writer.add_scalar("charts/dg_gate_mean", dg_gate.mean().item(), global_step)
            writer.add_scalar("charts/dg_surprisal_mean", dg_surprisal.mean().item(), global_step)
            writer.add_scalar("charts/dg_chi_std", dg_chi.std().item(), global_step)
            writer.add_scalar("charts/dg_gate_sat_frac", dg_sat_frac.item(), global_step)
            writer.add_scalar("charts/dg_sign_agree", dg_sign_agree.item(), global_step)
        if auto_alpha:
            writer.add_scalar("losses/alpha", log_alpha.exp().item(), global_step)
            writer.add_scalar("losses/target_entropy", target_entropy, global_step)
            # squashed entropy H(s) = -logπ; per-step bonus; and the entropy-domination
            # probe: ratio of soft-advantage std to reward-advantage std (>>1 => entropy
            # swamps the reward signal in the ranking; should fall toward ~1 as alpha anneals).
            writer.add_scalar("debug/squashed_entropy", (-logprobs).mean().item(), global_step)
            writer.add_scalar("debug/soft_bootstrap_bonus", next_value_bonus.mean().item(), global_step)
            writer.add_scalar("debug/soft_adv_std_ratio", (policy_adv.std() / (advantages.std() + 1e-8)).item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/returns_absmax", b_returns.abs().max().item(), global_step)
        writer.add_scalar("debug/target_edge_mass", edge_mass, global_step)
        # corr≈1 -> shaped advantage ~ raw GAE (reshaping adds little); lower -> the
        # transform genuinely changes the gradient direction/magnitude.
        writer.add_scalar("debug/distpg_corr_with_gae", adv_corr, global_step)
        writer.add_scalar("debug/distpg_sign_agree", adv_sign_agree, global_step)
        writer.add_scalar("debug/u_edge_frac", u_edge_frac, global_step)
        writer.add_scalar("debug/sigma_mean", b_sigma.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
