# LeJEPA successor-innovation PPO v4.
#
# No GAE, Q-learning, or PopArt. The critic predicts the state-only infinite-horizon
# geometric successor embedding Psi(s) = E[sum_k gamma^k phi_{t+k} | s]. The policy uses
# its one-step vector Bellman innovation:
#
#   E_t = phi_t + gamma * Psi(s_{t+1}) - Psi_cf(s_t)
#   A_t = w_reward dot E_t
#
# Psi_cf is CLA's action-marginalized, model-computed state baseline; it is independent of
# the behavior action and therefore remains a valid policy-gradient baseline. LeJEPA is
# load-bearing inside phi and Psi, while the observed next state—not a latent rollout—is
# used in the innovation. This changes only CLA's sampled lambda trace and preserves its
# proven shared trunk, successor critic, reward readout, PPO trust region, and dynamics gate.
import os
import random
import time
from dataclasses import dataclass
from math import log

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

from cleanrl.shared.lejepa import (
    ActionEncoder,
    ARPredictor,
    CompiledModule,
    MLP,
    SIGReg,
    StateEncoder,
)

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
    num_minibatches: int = 32
    update_epochs: int = 10
    # NOTE: the reference run this variant is built against (exp-name
    # "iterthink_v24_beta_d3bucket_mtp_ppoadvnorm_batch_v1", 8278 @8M) was NOT a separate
    # file -- it was mbpercnorm_v2 launched with --norm-adv --norm-adv-scope batch
    # --no-ret-percnorm. Those three are baked in as defaults here so the baseline is
    # reproduced by default rather than by remembering flags.
    norm_adv: bool = True            # ppoadvnorm: plain PPO advantage standardization...
    # --- Percentile advantage normalization (disabled: superseded by norm_adv) ---
    ret_percnorm: bool = False       # scale policy advantage by S = max(floor, P95-P5) of returns
    ret_perc_scope: str = "minibatch"  # "minibatch" = fresh per-mb P95-P5 of mb returns (NO EMA, like the
    #                                  old per-mb advnorm); "batch" = fresh WHOLE-ROLLOUT P95-P5, one S per
    #                                  update, NO EMA (the batch-vs-mb ablation); "ema" = v1's global EMA.
    ret_perc_rate: float = 0.01      # EMA rate on the percentiles (only used when scope=="ema")
    ret_perc_lo: float = 0.05        # P5
    ret_perc_hi: float = 0.95        # P95
    ret_perc_floor: float = 1.0      # scale floor S=max(floor, .) (DreamerV3 limit=1.0)
    clip_coef: float = 0.2           # PPO clip (lower bound 1-clip_coef; also upper if no high)
    clip_coef_high: float = 0.28     # "clip-higher" (DAPO): looser UPPER bound 1+clip_coef_high
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5       # used only when separate_grad_clip=False (global clip)
    target_kl: float = 0.03          # KL early-stop leash (the iterthink-line winner; shared trunk runs hot without it)

    # v21: shared backbone + decoupled (dual-backward) gradient clipping.
    share_backbone: bool = True      # one ThinkTrunk for both actor and critic heads
    separate_grad_clip: bool = True  # clip policy- and value-gradients to their own norms
    actor_grad_clip: float = 0.25    # max-norm for the policy gradient (incl. its trunk part)
    critic_grad_clip: float = 0.25   # max-norm for the value gradient (incl. its trunk part)

    # The HL-Gauss bucket critic is GONE. The critic head now emits successor features
    # (see header). MTP semantics and the horizon masks are retained verbatim -- only the
    # per-horizon target changed from a 511-bin scalar-return distribution to a
    # K-dimensional latent occupancy vector.
    critic_mtp_horizon: int = 6

    # --- LeJEPA successor-feature value pathway -------------------------------------
    emb_dim: int = 32                # d. At hist_len=1 the obs manifold is 17-dim, so e is rank
    #                                  <= 17 regardless and effective rank ~17 is NOT a failure
    #                                  signal. At hist_len>1 that ceiling no longer binds (the
    #                                  encoder sees H*17 inputs), so a RISE in ssl/emb_effective_rank
    #                                  is the direct evidence lever 2 did something.
    ssl_hidden: int = 256            # encoder / projector MLP width
    pred_depth: int = 2              # causal transformer depth
    pred_heads: int = 4
    pred_dim_head: int = 32
    pred_mlp_dim: int = 256
    # -- v3 LEVER 1: autoregressive multi-horizon prediction -------------------------
    pred_horizon: int = 1            # K. Number of AUTOREGRESSIVE rollout rounds. K=1 is v2
    #                                  (single-step, teacher-forced). K>1 feeds the predictor's
    #                                  OWN output back as its input, so error COMPOUNDS and the
    #                                  encoder is pressured to make e ROLLABLE rather than
    #                                  merely one-step-predictable. The reference (../le-wm)
    #                                  rolls out only at EVAL (jepa.py:61), never in training --
    #                                  there is no implementation to copy for this.
    pred_ctx: int = 3                # context frames before the first predicted frame.
    #                                  seq_len is DERIVED as pred_ctx + pred_horizon, so K=1
    #                                  gives seq_len=4 = LeWM's history_size(3)+num_preds(1).
    # -- v3 LEVER 2: history-dependent embeddings ------------------------------------
    sigreg_weight: float = 0.09      # lambda. NOTE: the Epps-Pulley statistic scales with batch
    #                                  size, so the reference value does not transfer verbatim.
    sigreg_num_proj: int = 1024
    sigreg_proj_chunk: int = 256     # exact (statistic is a mean over directions); bounds memory
    ssl_lr: float = 5e-5             # LeWM reference optimizer
    ssl_weight_decay: float = 1e-3
    ssl_batch: int = 1024            # sequences per SSL minibatch (SIGReg is ~1.1GB here, ~10GB at 8192)
    ssl_steps_target: int = 64       # SSL optimizer steps per iteration, HELD CONSTANT across K.
    #   Replaces v2's fixed ssl_epochs=8, which would have silently confounded the headline
    #   experiment. Steps per iteration are ssl_epochs * (n_seq // ssl_batch), and
    #   n_seq = (num_steps // seq_len) * num_envs SHRINKS as the horizon grows: at K=1,
    #   seq_len=4 -> 8192 seqs -> 64 steps; at K=4, seq_len=7 -> 4672 seqs -> only 32. The
    #   K=4 arm would have received HALF the encoder gradient (and half the total SIGReg
    #   pull, since SIGReg applies per step), so any regression would be unattributable
    #   between "the AR objective is bad" and "the encoder was trained half as much".
    #   The epoch count is DERIVED from this instead. Note the fix deliberately adjusts
    #   epochs and not ssl_batch: the Epps-Pulley statistic scales with the batch size, so
    #   rescaling ssl_batch would hold steps constant while quietly changing the effective
    #   sigreg_weight -- trading one confound for a subtler one.
    ssl_grad_clip: float = 1.0       # own clip (reference lewm.yaml gradient_clip_val), fully
    #                                  separate from PPO's -- see ssl/grad_norm for the pre-clip value
    sf_alpha: float = 1.0            # 1.0 = pure latent prediction, ZERO scalar regression in the
    #                                  network. <1 mixes in MSE(w_r.psi, w_r.Lambda) -- the fallback
    #                                  if the pure form underfits, not the starting point.
    # -- COUNTERFACTUAL LATENT ADVANTAGE ---------------------------------------------
    cla_m: int = 4                   # M counterfactual actions sampled from pi per state.
    #   The learned Psi(s) is replaced by a policy-marginalized model expansion when the
    #   action-conditioned dynamics model is reliable. M=0 disables the expansion.
    cla_beta: float = 1.0            # blend: psi <- (1-beta)*psi + beta*psi_cf. 0.0 is the
    #                                  parent; 1.0 is the full model-marginalized baseline.
    cla_dyn_r2_gate: float = 0.85    # lo end of the RAMP on held-out per-dim R^2. Below this
    #   the counterfactual is off entirely. Same discipline as w_r_solved: a cold dynamics net
    #   would otherwise inject garbage into the baseline for the first iterations, which is
    #   precisely when early damage compounds hardest in RL.
    cla_dyn_r2_full: float = 0.95    # hi end: full cla_beta. A RAMP rather than a threshold
    #   because R^2 will sit near the bar and toggle; each toggle would swap the critic's
    #   target between two Bellman operators at full amplitude while the Lambda standardizer
    #   EMA lags ~100 iterations behind, producing an oscillation that reads as RL noise.
    dyn_hidden: int = 256
    dyn_lr: float = 3e-4
    dyn_steps_target: int = 64       # optimizer steps per iteration, matching ssl_steps_target
    dyn_batch: int = 1024
    sf_ridge: float = 1e-3           # ridge coefficient for the closed-form w_r solve
    sf_target_ema: float = 0.01      # EMA rate for per-coordinate successor-target scaling
    mc_window: float = 500           # truncated-MC horizon for the UNRIGGED EV gate (gamma^500=0.0066)

    # Keep observation normalization, but disable return/reward normalization for
    # this ablation. NormalizeReward tracks discounted returns internally and
    # divides rewards by their running std, so it must remain off here.
    normalize_reward: bool = False
    clip_reward: bool = False

    # Optional transforms of the successor-innovation advantage.
    #   "v10" | "cdf_probit" | "tanh_std" | "tanh_gae" | "clip_z"
    #   "rankgauss" | "rankgauss_signed" | "rankgauss_temp" | "rankgauss_signmag"
    adv_transform: str = "v10"       # d3percnorm: identity -- NO rankgauss. DreamerV3 percentile
    #                                  norm (below) is the sole advantage scaler ("no advantage norm").

    # Scope ablations: where advantage shaping / normalization are computed.
    #   adv_transform_scope: "batch" (default) ranks/shapes over the whole rollout
    #     once per iteration; "minibatch" recomputes the shaping within each
    #     minibatch (the idiomatic scope of norm_adv). Only matters for shaping
    #     transforms (rankgauss, ...); "v10" is identity so scope is moot.
    #   norm_adv_scope: "minibatch" (default, idiomatic PPO) standardizes each
    #     minibatch; "batch" standardizes once over the whole rollout.
    adv_transform_scope: str = "batch"
    norm_adv_scope: str = "batch"    # ...standardized once over the whole rollout ("batch")

    # v24: action distribution. "beta" (unimodal, dreamer4 default) | "gaussian"
    # (dreamer4 state-dependent log-VARIANCE head, not SAC log_std; soft tanh-rescale bound).
    actor_dist: str = "beta"
    logvar_min: float = -8.0         # gaussian: soft log-var bound (symmetric => std=1 at init)
    logvar_max: float = 8.0          # std in [exp(-4), exp(4)] = [0.018, 54.6]

    tanh_kappa: float = 2.0          # tanh saturation scale (in per-state std units)
    clip_z_c: float = 2.0            # Winsorize bound for "clip_z" (in std units)
    rank_tanh_kappa: float = 1.5     # tanh temperature for "rankgauss_temp" (<inf=harder)
    pos_neg_alpha: float = 0.5       # PMPO-style: weight +adv by 2a, -adv by 2(1-a); 0.5=off
    # CDF/probit knobs (used by "cdf_probit" and "rankgauss").
    cdf_probit_clamp: float = 0.999

    hidden: int = 64
    k_blocks: int = 3
    n_experts: int = 16

    # --- torch.compile ---------------------------------------------------------------
    # reduce-overhead (CUDA graph trees) is NOT usable on the actor/critic path: v_loss and
    # pg_loss share the trunk forward, and the update runs backward twice with
    # retain_graph=True, cloning and re-adding grads in between. Cudagraph outputs live in
    # the graph pool, clip_grad_norm_ mutates them in place, and the second backward replays
    # over live refs -> either "accessing tensor output of CUDAGraphs that has been
    # overwritten" or a re-record per minibatch (x320/iter), which is SLOWER than eager.
    # So the trunk gets inductor fusion without graphs; the SSL nets (separate optimizer,
    # single plain backward) do get reduce-overhead.
    compile: bool = False            # validate the port eager first -- compile changes numerics
    compile_mode: str = "reduce-overhead"  # applies to the SSL nets only; see above for why the
    #                                        actor/critic trunk cannot take cudagraphs at all
    compile_ssl_cudagraphs: bool = False   # DEFAULT OFF, and not out of caution: LeJepaSSL.forward
    #   chains encoder -> action_encoder -> predictor, and each cudagraph-wrapped call issues its
    #   own cudagraph_mark_step_begin(), which invalidates the still-pending backward of the
    #   modules called earlier in the SAME forward. It raises "accessing tensor output of
    #   CUDAGraphs that has been overwritten" on the FIRST ssl_loss.backward() -- reproduced --
    #   and suppress_errors does NOT catch it (that only covers dynamo compile-time errors).
    #   Cloning outputs does not help either: the invalidated tensors are the saved intermediates.
    #   The SSL path is ~3% of wall clock, so inductor fusion without graphs costs ~nothing.

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    @property
    def seq_len(self) -> int:
        """Derived, not a flag: pred_ctx + pred_horizon.

        A property rather than a field so it cannot drift out of sync with the levers --
        an independent seq_len would silently truncate the AR rollout the moment
        pred_horizon was raised without it. Properties carry no annotation, so this is
        not a dataclass field and tyro does not surface it as a CLI argument.
        """
        return self.pred_ctx + self.pred_horizon


def make_env(env_id, idx, capture_video, run_name, gamma, normalize_reward, clip_reward):
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
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        if clip_reward:
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


class ObsDynamics(nn.Module):
    """One-step action-conditioned observation dynamics: s' = s + D(s, a).

    OBSERVATION space, not latent, and that is the whole point of this variant. The
    measured ceiling on any latent-reading critic is gate/ev_latent_cap ~0.33 @1M / 0.66
    late, against gate/ev_trunk_probe ~0.61 / 0.88 for raw obs. The SIGReg latent is a
    LOSSY bottleneck for value -- whitening toward N(0,I) is active pressure against the
    scale structure the reward is made of -- so a counterfactual next-STATE valued by the
    trunk is worth ~25 EV points more than a counterfactual next-LATENT.

    Predicts the DELTA. Under NormalizeObservation s is O(1) while the per-step change is
    far smaller, so regressing s' directly would spend most of its capacity re-emitting its
    own input and report a flattering R^2 for doing nothing.

    Output layer is ZERO-INIT, so at initialisation s_hat' == s exactly. That makes the
    untrained model an identity rather than a noise source: the counterfactual baseline
    degrades to phi(s,a) + gamma*psi(s), which is a stale-by-one-step value estimate rather
    than garbage. Combined with the r2 gate this means a cold model cannot poison the run.

    Also predicts CONTINUATION. psi_cf adds gamma*psi(s_hat') for every sampled action, but
    the learned psi it replaces was fit against targets masked by boot=(1-term)*valid, so it
    already discounts states likely to terminate. Without a termination estimate the computed
    baseline would bootstrap straight through death and be systematically INFLATED at exactly
    the pre-failure states that matter most -- and the "unbiased baseline" claim would fail
    there. HalfCheetah never terminates, so on the headline benchmark this head sees only
    negatives; the logit bias is initialised at +8 (p_cont = 0.99966) so that costs ~3e-4 of
    the bootstrap rather than the 0.5x a zero-init logit would.
    """

    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.delta_head = nn.Linear(hidden, obs_dim)
        self.cont_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.cont_head.weight)
        nn.init.constant_(self.cont_head.bias, 8.0)

    def forward(self, s, a):
        """-> (delta, cont_logit). cont_logit is the CONTINUATION logit, not termination."""
        h = self.body(torch.cat([s, a], dim=-1))
        return self.delta_head(h), self.cont_head(h)[..., 0]


class LeJepaSSL(nn.Module):
    """Encoder + causal action-conditioned predictor + SIGReg.

    Trained by EXACTLY two terms: a next-embedding prediction MSE with BOTH branches
    attached (no stop-gradient on the target -- SIGReg is what prevents collapse, and a
    stop-grad or EMA teacher would re-introduce the second unanchored timescale this
    design exists to remove), and SIGReg itself.

    Its only job is to define what the embedding e means. It never sees a reward, and it
    never receives gradient from the value path -- psi and w_r both read sg(e).
    """

    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        self.horizon = args.pred_horizon
        # LEVER 2: the encoder reads H stacked frames. e_t = f(s_{t-H+1..t}).
        self.encoder = StateEncoder(obs_dim, args.emb_dim, args.ssl_hidden)
        self.action_encoder = ActionEncoder(act_dim, args.emb_dim)
        self.predictor = ARPredictor(
            num_frames=args.seq_len,
            depth=args.pred_depth,
            heads=args.pred_heads,
            mlp_dim=args.pred_mlp_dim,
            input_dim=args.emb_dim,
            hidden_dim=args.emb_dim,
            dim_head=args.pred_dim_head,
            dropout=0.0,
            emb_dropout=0.0,
        )
        self.pred_proj = MLP(args.emb_dim, args.ssl_hidden, args.emb_dim)
        self.sigreg = SIGReg(
            num_proj=args.sigreg_num_proj, proj_chunk=args.sigreg_proj_chunk
        )

    def forward(self, obs_seq, act_seq, cum_cont, sigreg_weight):
        """obs_seq (N,L,H*obs); act_seq (N,L,act); cum_cont (N,L) = cumprod of 1-boundary.

        AUTOREGRESSIVE multi-horizon rollout. Round k consumes round k-1's OUTPUT as its
        input, so prediction error compounds exactly as it would at inference -- which is
        the whole point: a teacher-forced one-step loss is satisfiable by a near-linear
        whitened map of s (one-step MuJoCo prediction is close to trivial), and that is
        why v2's encoder failed to earn its place against the no-encoder ablation.

        THE SHIFT IS WHAT MAKES IT MULTI-STEP, and it is the easy thing to get wrong. A
        causal transformer's output at position l is always "predict l+1 given context
        through l". Feeding predictions back WITHOUT shifting the action conditioning just
        re-predicts l+1 from a noisier context -- still one step, no compounding, and the
        loss curve looks entirely plausible. To reach l+2 the context must ADVANCE: round
        k runs on inputs that stand for frames k..L-1 and actions a[k..L-1], so its output
        at position l predicts emb[l+k+1]. Each round drops one position.

        Density is preserved: EVERY position contributes at EVERY horizon, so raising K
        costs K predictor passes but loses only K-1 supervised pairs per sequence.
        """
        emb = self.encoder(obs_seq)                              # (N,L,d)
        act_emb = self.action_encoder(act_seq)                   # (N,L,d)
        L = emb.shape[1]
        err_terms, mask_terms = [], []
        x = emb                                                  # round 0 input: all TRUE
        for k in range(self.horizon):
            lk = L - k                                           # x is (N, lk, d)
            # pos_offset=k: this round's tokens stand for frames k..L-1, so they must carry
            # those positional embeddings rather than restarting the window at 0.
            out = self.pred_proj(self.predictor(x, act_emb[:, k : k + lk], pos_offset=k))
            n_valid = lk - 1                                     # out[l] ~ emb[l+k+1]
            if n_valid <= 0:
                break
            # Target is ATTACHED on both branches -- no stop-grad, no EMA teacher. SIGReg
            # is what prevents collapse; a stop-grad here would re-add the second
            # unanchored timescale this design exists to remove.
            err_terms.append((out[:, :n_valid] - emb[:, k + 1 : k + 1 + n_valid]).pow(2).mean(-1))
            # Valid iff the whole window obs[0 .. k+1+l] is one episode. cum_cont[j] is the
            # product of cont[0..j], and cont[i]==1 means obs[i]->obs[i+1] is intra-episode,
            # so cum_cont[j]==1 <=> obs[0..j+1] are contiguous. Target frame k+1+l therefore
            # needs cum_cont[k+l], NOT cum_cont[k+1+l]: the latter additionally demands the
            # transition OUT of the target frame be intra-episode, which no prediction
            # depends on, and it silently drops one valid pair per horizon whenever a chunk's
            # exit transition happens to be a reset.
            mask_terms.append(cum_cont[:, k : k + n_valid])
            x = out[:, : lk - 1]                                 # feed predictions forward
        err = torch.cat(err_terms, dim=1)
        mask = torch.cat(mask_terms, dim=1)
        # Masked mean, NOT .mean() over a zeroed tensor (which would divide by the full count).
        # Pooled uniformly across horizons: later horizons carry larger error and so dominate
        # the gradient, which is what "make e rollable" MEANS. Per-horizon values are logged
        # separately (ssl/pred_loss_h*) so compounding is visible rather than inferred.
        pred_loss = (err * mask).sum() / mask.sum().clamp_min(1.0)
        # Stacked as a TENSOR, not floats: this runs 64x per iteration, and a float() per
        # horizon here would add K host syncs to every SSL step and break the compiled graph.
        per_horizon = torch.stack(
            [(e * m).sum() / m.sum().clamp_min(1.0) for e, m in zip(err_terms, mask_terms)]
        ).detach()
        # THE TRANSPOSE IS LOAD-BEARING. SIGReg reduces the empirical characteristic
        # function over dim -3 and scales by size(-2); both resolve to the BATCH only in
        # (T, B, D) layout. Passing (N, L, d) would average the CF over just L samples,
        # silently disabling collapse protection while still logging a plausible number.
        sigreg_loss = self.sigreg(emb.transpose(0, 1))           # (L, N, d)
        return pred_loss + sigreg_weight * sigreg_loss, pred_loss, sigreg_loss, per_horizon


def phi_features(emb, obs, action):
    """phi = [e, s, a, a*a, 1].

    The raw (already NormalizeObservation-scaled) state block is v2's whole change. See
    the header: without it a linear w_r cannot recover the velocity term from a whitened
    nonlinear latent, and v1 measured the resulting residual at lag-1 autocorrelation
    0.474 -- structured error, the expensive kind, injected into every value estimate.

    The action blocks are not decoration. HalfCheetah's reward is x_vel - 0.1*||a||^2, and
    the control cost is not a function of state AT ALL -- so a probe on e alone carries an
    irreducible residual that scales with policy action magnitude, i.e. the very
    non-stationarity this design removes leaks straight back in. `a*a` lets a LINEAR probe
    capture MuJoCo ctrl cost exactly.

    The trailing constant is the regression intercept, carried as a REAL feature rather
    than a separate bias so that V = w_r . psi stays an exact identity (a bias term would
    have no discounted-sum counterpart in psi). It earns its keep twice over: the
    observation is normalized, so e has a running-mean offset that shifts as the policy's
    state distribution moves, and the constant's own discounted sum is 1/(1-gamma)
    truncated at episode end -- that coordinate quietly encodes expected remaining
    episode length.
    """
    ones = emb[..., :1].new_ones(emb.shape[:-1] + (1,))
    return torch.cat([emb, obs, action, action * action, ones], dim=-1)


def counterfactual_psi(agent, dyn, psi_raw, flat_obs, emb_flat, gamma, m_samples, w_r, chunk=32768):
    """psi_cf(s) = (1/M) sum_m [ phi(s, a^m) + gamma * psi(s + D(s, a^m)) ],  a^m ~ pi(.|s).

    A COMPUTED state baseline. Psi(s) predicts expected successor occupancy; this evaluates
    its policy expectation directly by pushing M counterfactual actions through the dynamics
    model and valuing the results. It is a one-step
    model-based expansion of the value, which is strictly better-informed than psi(s) alone
    wherever the critic is wrong -- and it is not computable from data, because it needs
    next-states for actions the agent never took.

    UNBIASED AS A BASELINE. The average is over pi at s, so psi_cf carries NO dependence on
    the action actually taken. An action-dependent baseline would bias the policy gradient;
    this one cannot. The actions are sampled through the SAME native-z -> to_action path the
    rollout used, so phi(s,a^m) and D(s,a^m) see actions in the identical representation to
    the executed one (the Beta actor's z lives in (0,1) and is rescaled, not tanh-squashed).

    Model is used at k=1 ONLY, which is where it was measured reliable: AR training to K=4
    changed the representation (effective rank 18->29) and still cost 5% return, so there is
    no evidence supporting deeper rollout here.

    THE COST OF FINITE M, MEASURED RATHER THAN ASSUMED. Replacing a learned baseline with an
    M-sample estimate injects zero-mean noise into it. That noise is UNBIASED -- the a^m are
    drawn independently of the executed a_t -- but it is not free: the current state's
    baseline carries coefficient exactly -1, so the noise passes into the innovation
    undamped.
    The second return value is the per-state Monte-Carlo standard error of w_r.psi_cf, in
    value units, so `cla/cf_se_frac` can be read against the advantage spread instead of
    guessed at. If that fraction is large, M is the knob, not cla_beta.
    """
    outs, ses = [], []
    for start in range(0, flat_obs.shape[0], chunk):
        s_c = flat_obs[start : start + chunk]
        e_c = emb_flat[start : start + chunk]
        actor_feat, _ = agent._trunks(s_c)
        dist, to_action, _ = agent._actor_dist(actor_feat)
        acc = None
        # SHIFTED-DATA variance (centred on the first draw), not sum-of-squares. w_r.term is
        # O(10) while its spread across a^m can be O(1e-3); the naive E[v^2]-E[v]^2 cancels
        # to fp32 noise and reports a spurious ~5e-3 SE for a distribution with zero spread.
        v0 = None
        v_sum = torch.zeros(s_c.shape[0], device=s_c.device)
        v_sq = torch.zeros(s_c.shape[0], device=s_c.device)
        for _ in range(m_samples):
            z = dist.sample()
            if agent.actor_dist == "beta":
                z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            a = to_action(z)
            d_s, cont_logit = dyn(s_c, a)
            s_next = s_c + d_s
            # psi_raw is REQUIRED here, not cosmetic: the critic head regresses STANDARDIZED
            # Lambda, while phi_features is in raw units. Summing them unconverted would add
            # two different scales and silently corrupt the baseline.
            psi_n = psi_raw(agent.get_value(s_next)[:, 0])
            p_cont = torch.sigmoid(cont_logit).unsqueeze(-1)
            term = phi_features(e_c, s_c, a) + gamma * p_cont * psi_n
            acc = term if acc is None else acc + term
            v_m = term @ w_r
            v0 = v_m if v0 is None else v0
            d_m = v_m - v0
            v_sum += d_m
            v_sq += d_m * d_m
        outs.append(acc / m_samples)
        # Population variance of the M scalar draws -> standard error of their mean.
        # clamp_min before sqrt: at M=1 this is exactly 0 and the subtraction can still land
        # a few ulps negative.
        # Bessel-corrected: the M draws estimate the spread of a distribution, so the
        # population form would understate the standard error by ~13% at M=4.
        var_m = (v_sq - v_sum * v_sum / m_samples).clamp_min(0.0) / max(m_samples - 1, 1)
        ses.append((var_m / m_samples).sqrt())
    return torch.cat(outs, dim=0), torch.cat(ses, dim=0)


def solve_reward_probe(phi, reward, ridge):
    """Closed-form ridge solve of w_r from phi -> immediate reward.

    Solved, not gradient-fit: it removes a learning rate, removes the cold-start phase
    where a random V would drive the actor, and makes "thin, fast, stationary readout"
    literal -- w_r is recomputed optimally every iteration. Done in float64; the normal
    equations square the condition number and phi's blocks have very different scales.
    """
    phi64 = phi.double()
    gram = phi64.T @ phi64
    rhs = phi64.T @ reward.double()
    scale = torch.diagonal(gram).mean().clamp_min(1e-12)
    gram = gram + ridge * scale * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve(gram, rhs).to(phi.dtype)


def ev_score(pred, target):
    var = target.var()
    return float(1.0 - (target - pred).var() / var.clamp_min(1e-12))


def successor_innovation(
    phi,
    psi_current,
    psi_next,
    terminations,
    transition_valids,
    gamma,
):
    """One-step Bellman innovation of an infinite-horizon successor embedding."""
    bootstrap = ((1.0 - terminations) * transition_valids).unsqueeze(-1)
    target = phi + gamma * bootstrap * psi_next
    return target, target - psi_current


def chunk_sequences(x, seq_len):
    """(T, B, D) -> (n*B, L, D) with time contiguous within a chunk and env fixed.

    The buffer is T-MAJOR, so the usual flatten gives index t*B + e. Chunking the
    FLATTENED tensor would produce sequences that walk across envs at a fixed timestep
    rather than across time -- and because adjacent envs look similar under
    NormalizeObservation, that yields a perfectly plausible prediction loss that no curve
    would ever catch. Chunk before flattening.

    Round-trip: x[n*L + l, b] == out[n*B + b, l].
    """
    t, b = x.shape[0], x.shape[1]
    tail = x.shape[2:]
    n = t // seq_len
    return (
        x[: n * seq_len]
        .view(n, seq_len, b, *tail)
        .permute(0, 2, 1, *range(3, 3 + len(tail)))
        .reshape(n * b, seq_len, *tail)
    )


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
        # SUCCESSOR-FEATURE critic head. Same shape and MTP semantics as the HL-Gauss head
        # it replaces -- only the per-horizon target space changed, from 511 scalar-return
        # bucket logits to K = emb_dim + obs_dim + 2*act_dim + 1 occupancy dimensions.
        # Horizon h predicts the one-step bootstrapped successor target at t+h.
        # Zero-init makes the first target equal phi rather than bootstrapping from noise.
        self.sf_dim = args.emb_dim + obs_dim + 2 * act_dim + 1  # [e, s, a, a*a, 1]
        self.critic_mtp_horizon = args.critic_mtp_horizon
        self.critic_head = layer_init(
            nn.Linear(H, args.critic_mtp_horizon * self.sf_dim, bias=False), std=0.1
        )
        with torch.no_grad():
            self.critic_head.weight.zero_()
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
        # Returns successor features (B, mtp, sf_dim); horizon 0 is psi(s_t), whose
        # scalar readout w_r . psi_0 is V(s_t).
        _, critic_feat = self._trunks(x)
        return self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.sf_dim)

    def get_action_and_value(self, x, z=None):
        # z is the distribution-NATIVE sample (pre-tanh for gaussian; in (0,1) for
        # beta). When replaying from the buffer it is passed back in; log_prob is
        # recomputed at the same native sample (v21's z-replay, generalized).
        actor_feat, critic_feat = self._trunks(x)
        dist, to_action, log_det_fn = self._actor_dist(actor_feat)
        if z is None:
            z = dist.sample()
            if self.actor_dist == "beta":
                z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = to_action(z)
        log_prob = (dist.log_prob(z) - log_det_fn(z)).sum(1)
        value_sf = self.critic_head(critic_feat).view(-1, self.critic_mtp_horizon, self.sf_dim)
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
        return action, z, log_prob, entropy, value_sf

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


def shape_advantage(advantage, args, device):
    """Optionally reshape a raw successor-innovation advantage.

    The base's "tanh_std" and "cdf_probit" transforms are GONE, not stubbed. Both read
    the critic's per-state value DISTRIBUTION (sigma(s) in bin units, and u = Z(s)'s CDF
    at the return), and this variant has no distributional critic to read -- the head
    predicts successor features. Feeding them a constant sigma=1 / u=0 placeholder would
    make cdf_probit return the same constant for every sample (a zero policy gradient
    after norm_adv) while logging a perfectly healthy-looking curve, so the branches are
    deleted and the arg is rejected at startup instead.
    """
    if args.adv_transform == "v10":
        return advantage
    elif args.adv_transform == "tanh_gae":
        gz = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        return torch.tanh(gz / args.tanh_kappa)
    elif args.adv_transform == "clip_z":
        # Winsorized z-score: ONE standardize + hard tail clip. tanh's hard cousin;
        # preserves bulk magnitude exactly (linear in [-c,c]) and is ~self-normalizing
        # (clipping a unit-var signal barely changes its std), so norm_adv ~ no-op.
        gz = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        return gz.clamp(-args.clip_z_c, args.clip_z_c)
    elif args.adv_transform == "rankgauss":
        # Rank-Gaussian: the principled single op. Replaces each advantage by the
        # Gaussian quantile of its empirical rank -> self-normalizing (bounded,
        # zero-mean, ~unit-scale, fully outlier-immune; no kappa, no double z-score).
        # This is cdf_probit with the EMPIRICAL batch rank instead of the critic's
        # per-state CDF (distribution-free; the distribution was shown incidental).
        n = advantage.numel()
        ranks = advantage.argsort().argsort().to(torch.float32)  # 0..n-1
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        centered = (2.0 * uq - 1.0).clamp(-c, c)
        return ((2.0 ** 0.5) * torch.erfinv(centered.cpu())).to(device)
    elif args.adv_transform == "rankgauss_signed":
        # Sign-anchored rank-Gaussian: rank positives and negatives SEPARATELY so the
        # advantage's zero-crossing is preserved exactly. Plain rankgauss anchors the
        # sign flip at the batch MEDIAN, not 0, so on a skewed advantage it can assign the
        # wrong sign to near-median samples — and PPO needs the sign right. Each sign
        # group is mapped to its half of the Gaussian by its within-group rank.
        c = args.cdf_probit_clamp
        out = torch.zeros_like(advantage)
        for side in (advantage > 0, advantage < 0):
            if side.any():
                g = advantage[side]
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
        n = advantage.numel()
        ranks = advantage.argsort().argsort().to(torch.float32)
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
        n = advantage.numel()
        ranks = advantage.argsort().argsort().to(torch.float32)
        uq = (ranks + 0.5) / n
        c = args.cdf_probit_clamp
        mag = ((2.0 ** 0.5) * torch.erfinv((2.0 * uq - 1.0).clamp(-c, c).cpu())).to(device).abs()
        return torch.sign(advantage) * mag
    else:
        raise ValueError(f"unknown adv_transform {args.adv_transform}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.adv_transform_scope in ("batch", "minibatch")
    assert args.norm_adv_scope in ("batch", "minibatch", "batch_retstd", "minibatch_retstd")
    # batch-scope norm reuses the batch-shaped advantage, so it needs batch-scope shaping.
    assert not (args.adv_transform_scope == "minibatch" and args.norm_adv_scope in ("batch", "batch_retstd")), \
        "norm_adv_scope=batch/batch_retstd requires adv_transform_scope=batch"
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

    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this ablation")
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                args.gamma,
                args.normalize_reward,
                args.clip_reward,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # pred_ctx=0 does NOT crash -- it corrupts. The AR loop's `break` leaves per_horizon with
    # K-1 entries while the logging accumulator has K, and at K=2 that broadcasts a length-1
    # tensor across both slots, so every horizon reports the same number and the compounding
    # ratio reads 1.0 (i.e. "the rollout isn't compounding") for a purely cosmetic reason.
    if args.pred_ctx < 1:
        raise ValueError(f"pred_ctx must be >= 1, got {args.pred_ctx}")

    # Fail loud at startup rather than degrading silently mid-run.
    if args.adv_transform in ("tanh_std", "cdf_probit"):
        raise ValueError(
            f"adv_transform={args.adv_transform!r} reads the critic's per-state value "
            "DISTRIBUTION, which this variant does not have (the head predicts successor "
            "features, not bucket logits). Use v10 / tanh_gae / clip_z / rankgauss*."
        )
    n_seq_per_iter = (args.num_steps // args.seq_len) * args.num_envs
    if n_seq_per_iter < args.ssl_batch:
        raise ValueError(
            f"ssl_batch={args.ssl_batch} exceeds the {n_seq_per_iter} sequences a rollout "
            "yields ((num_steps // seq_len) * num_envs). The SSL loop drops the last ragged "
            "minibatch (SIGReg's statistic scales with batch size), so it would take ZERO "
            "steps: the encoder would stay at random init and every sf/ and gate/ metric "
            "would be computed on a frozen random projection."
        )

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # Param groups for decoupled clipping. With a shared backbone, the trunk
    # appears in BOTH lists (it receives policy and value gradients separately).
    actor_params = agent.actor_parameters()
    critic_params = agent.critic_parameters()

    # --- LeJEPA successor-feature value pathway -------------------------------------
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim_sf = int(np.prod(envs.single_action_space.shape))
    sf_dim = agent.sf_dim

    # SSL nets live in their OWN top-level module, deliberately not a submodule of Agent:
    # otherwise their parameters would enter agent.parameters() (the PPO optimizer),
    # actor_parameters()/critic_parameters(), and the 0.25 clip budget -- silently changing
    # the very thing being held fixed.
    ssl = LeJepaSSL(obs_dim, act_dim_sf, args).to(device)
    dyn = ObsDynamics(obs_dim, act_dim_sf, args.dyn_hidden).to(device)
    dyn_optimizer = optim.Adam(dyn.parameters(), lr=args.dyn_lr)
    # Gate state: no counterfactual baseline until the model has been MEASURED good this
    # iteration. Starts closed, so iteration 1 runs the parent's baseline exactly.
    dyn_r2_last = 0.0
    dyn_mse_last = 0.0
    dyn_cont_acc = 0.0
    ssl_optimizer = optim.AdamW(
        ssl.parameters(), lr=args.ssl_lr, weight_decay=args.ssl_weight_decay
    )

    if args.compile:
        # retain_graph in the dual-backward is incompatible with donated buffers.
        import torch._functorch.config
        import torch._dynamo

        torch._functorch.config.donated_buffer = False
        torch._dynamo.config.suppress_errors = True   # never let compile break the run
        torch._dynamo.config.recompile_limit = 64     # trunk alone sees 4 (shape, grad) combos
        if agent.share_backbone:
            agent.trunk = CompiledModule(agent.trunk, cudagraphs=False)
        else:
            agent.actor_trunk = CompiledModule(agent.actor_trunk, cudagraphs=False)
            agent.critic_trunk = CompiledModule(agent.critic_trunk, cudagraphs=False)
        for name in ("encoder", "action_encoder", "predictor"):
            setattr(ssl, name, CompiledModule(
                getattr(ssl, name), mode=args.compile_mode, cudagraphs=args.compile_ssl_cudagraphs
            ))

    # Per-dimension EMA standardization of the SF target. The blocks of phi are wildly
    # heteroscedastic: for a whitened e block std(sum gamma^k e) ~ tau ~ 20 (the effective
    # autocorrelation horizon, NOT 1/(1-gamma)=100), while the a*a block is strictly
    # positive with a large mean and small variance, and the constant block is ~1/(1-gamma)
    # exactly. A single global rescale leaves the MSE dominated by the wrong block.
    sf_mean = torch.zeros(sf_dim, device=device)
    sf_std = torch.ones(sf_dim, device=device)
    sf_stat_count = 0
    # w_r starts at zero => V == 0 for the first rollout, which is the correct "no
    # information" baseline rather than random noise driving the actor.
    w_r = torch.zeros(sf_dim, device=device)
    # This makes the vector innovation zero on iteration one. Use the scalar one-step TD
    # innovation for that single update; from iteration two the embedding projection is live.
    w_r_solved = False

    # Fixed INDEX set (not a fixed obs batch) for the frame-drift probe: the states are
    # re-drawn from each rollout, so the probe never goes stale w.r.t. the observation
    # normalizer. Stride 31 is COPRIME with num_envs; the buffer is T-major (i = t*B + e),
    # so a stride sharing a factor with num_envs (e.g. 32 at num_envs=16) would alias onto
    # a single environment and measure drift on one trajectory instead of the state marginal.
    drift_probe_idx = torch.arange(0, args.num_steps * args.num_envs, 31, device=device)[:1024]

    def psi_raw(sf_standardized):
        """Un-standardize the head output back into phi-accumulation units."""
        return sf_standardized * sf_std + sf_mean

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    transition_terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_boundaries = torch.zeros((args.num_steps, args.num_envs)).to(device)
    transition_valids = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    # DreamerV3 percentile advantage-norm running stats (EMA of the P5/P95 return percentiles).
    ema_ret_lo, ema_ret_hi, ema_perc_inited, ret_perc_scale = 0.0, 1.0, False, 1.0

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            # The SSL optimizer is separate, so the base's anneal (which writes
            # param_groups[0] of `optimizer` only) does not reach it. Anneal it too: it
            # freezes the encoder frame late in training, which is exactly when the policy
            # is refining and a drifting frame would do the most damage to the bootstrap.
            ssl_optimizer.param_groups[0]["lr"] = frac * args.ssl_lr

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, ent, value_sf = agent.get_action_and_value(next_obs)
                values[step] = psi_raw(value_sf[:, 0]) @ w_r
            actions[step] = action
            latent_zs[step] = z
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            transition_boundary = np.logical_or(terminations, truncations)
            transition_valid = (~transition_boundary).astype(np.float32)
            transition_next_obs = np.array(next_obs_np, copy=True)
            final_obs = infos.get("final_observation")
            final_obs_mask = infos.get("_final_observation")
            if final_obs is not None:
                if final_obs_mask is None:
                    final_obs_mask = [fo is not None for fo in final_obs]
                for env_idx, has_final in enumerate(final_obs_mask):
                    if has_final and final_obs[env_idx] is not None:
                        transition_next_obs[env_idx] = final_obs[env_idx]
                        transition_valid[env_idx] = 1.0
                    elif transition_boundary[env_idx]:
                        transition_valid[env_idx] = 0.0

            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
            transition_terminations[step] = torch.as_tensor(
                terminations, device=device, dtype=torch.float32
            )
            transition_boundaries[step] = torch.as_tensor(
                transition_boundary, device=device, dtype=torch.float32
            )
            transition_valids[step] = torch.as_tensor(transition_valid, device=device, dtype=torch.float32)
            next_obses[step] = torch.as_tensor(transition_next_obs, device=device, dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            next_done = torch.as_tensor(transition_boundary, device=device, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            flat_obs_shape = (-1,) + envs.single_observation_space.shape
            psi_next = psi_raw(agent.get_value(next_obses.reshape(flat_obs_shape))[:, 0]).reshape(
                args.num_steps, args.num_envs, sf_dim
            )
            # Kept as trunk features, not just psi: the EV gate below needs the SAME
            # features to fit its detached scalar probe, and this variant already does two
            # full-batch trunk forwards per iteration where the base did one.
            critic_feat_buf = agent._trunks(obs.reshape(flat_obs_shape))[1]
            psi_cur = psi_raw(
                agent.critic_head(critic_feat_buf).view(-1, args.critic_mtp_horizon, sf_dim)[:, 0]
            ).reshape(args.num_steps, args.num_envs, sf_dim)

            # ---- COUNTERFACTUAL BASELINE ----------------------------------------------
            # Placed before the innovation so both ends use the same baseline family.
            # Replacing only Psi(s_t) would create a systematic mismatch with Psi(s_{t+1}).
            # hist_len is GONE in this variant, not defaulted off: H in {1,3,16} was measured
            # monotonically harmful (reward_probe_r2 0.963/0.965/0.946, resid_ac_lag10
            # 0.134/0.206/0.373), and carrying it would force a history stack for next_obses
            # too. Dead levers are removed, not left as flags.
            emb_buf = ssl.encoder(obs.reshape(flat_obs_shape)).reshape(
                args.num_steps, args.num_envs, args.emb_dim
            )
            # RAMP, not a step. A hard threshold on a quantity that will sit near it makes
            # cla_active toggle between iterations, and each toggle swaps sf_target between
            # two different Bellman operators at FULL amplitude -- while the Lambda
            # standardizer EMA (rate max(0.01, 1/count)) needs ~100 iterations to catch up.
            # The symptom is a multi-iteration oscillation in returns/EV that reads as
            # ordinary RL noise. Linear credit between lo and hi removes the discontinuity
            # and lets a mediocre model contribute proportionally instead of not at all.
            b_eff = args.cla_beta * min(
                max(
                    (dyn_r2_last - args.cla_dyn_r2_gate)
                    / max(args.cla_dyn_r2_full - args.cla_dyn_r2_gate, 1e-8),
                    0.0,
                ),
                1.0,
            )
            cla_active = args.cla_m > 0 and b_eff > 0.0
            if cla_active:
                emb_flat = emb_buf.reshape(-1, args.emb_dim)
                psi_cf_cur, cf_se_cur = counterfactual_psi(
                    agent, dyn, psi_raw, obs.reshape(flat_obs_shape), emb_flat,
                    args.gamma, args.cla_m, w_r,
                )
                psi_cf_cur = psi_cf_cur.reshape(args.num_steps, args.num_envs, sf_dim)

                # Reuse the shifted state-baseline sample, which also halves the cost. In the parent
                # psi_next[t] == psi_cur[t+1] BITWISE off-boundary, because next_obses[t] and
                # obs[t+1] are the same array (see the rollout: obs[step] = next_obs,
                # next_obses[step] = transition_next_obs, which differ only where
                # final_observation is substituted).
                # Calling counterfactual_psi a second time on next_obses would draw
                # INDEPENDENT actions, so psi_cf_next[t] - psi_cf_cur[t+1] would be zero-mean
                # noise between adjacent state baselines. Reusing the shifted tensor keeps
                # the state-only baseline coherent; only boundaries and the rollout tail
                # require recomputation.
                psi_cf_next = torch.empty_like(psi_cf_cur)
                psi_cf_next[:-1] = psi_cf_cur[1:]
                need = transition_boundaries > 0.5
                need[-1] = True
                need_idx = need.reshape(-1).nonzero(as_tuple=True)[0]
                so = next_obses.reshape(flat_obs_shape)[need_idx]
                cf_b, _ = counterfactual_psi(
                    agent, dyn, psi_raw, so, ssl.encoder(so), args.gamma, args.cla_m, w_r,
                )
                psi_cf_next.view(-1, sf_dim)[need_idx] = cf_b

                # Reported in VALUE units, not as a mean over raw Lambda coordinates whose
                # scales differ by two orders of magnitude (the constant coordinate is
                # 1/(1-gamma) = 100; the whitened e block is ~tau = 20). w_r . (psi_cf - psi)
                # is directly comparable to the advantage spread it has to move.
                cla_gap = float(((psi_cf_cur - psi_cur) @ w_r).abs().mean()) * b_eff
                cf_se = float(cf_se_cur.mean()) * b_eff
                psi_cur = (1.0 - b_eff) * psi_cur + b_eff * psi_cf_cur
                psi_next = (1.0 - b_eff) * psi_next + b_eff * psi_cf_next
                # values[] was recorded DURING the rollout from the unblended psi. Leaving it
                # would make the scalar residual r_t + gamma*V_cf(s_{t+1}) - V_learned(s_t):
                # two different value functions on the two ends, so it stops telescoping and
                # carries a systematic (V_learned - V_cf) offset at every step. That corrupts
                # `returns`, and with it losses/explained_variance and debug/returns_* --
                # exactly the silently-plausible-curve failure. Rebuilt from the SAME w_r that
                # produced values[] (the previous iteration's solve, see adv_vector's note),
                # so at beta=0 this reproduces values[] to floating-point noise. copy_ rather
                # than rebind, to keep the preallocated rollout buffer's identity.
                values_learned = values.clone()
                values.copy_(psi_cur @ w_r)
            else:
                cla_gap = 0.0
                cf_se = 0.0
                values_learned = values

            next_transition_values = psi_next @ w_r
            # ---- SUCCESSOR-FEATURE basis --------------------------------------------
            # Built ONCE per iteration and reused by the SSL chunker below, so the encoder
            # sees byte-identical inputs in both places (a second, independently built stack
            # is exactly how a lagged-frame off-by-one hides).
            # emb_buf / obs_hist were built above, before the counterfactual baseline.
            # phi keeps the RAW s_t block, NOT obs_hist -- see hist_len's note: putting lagged
            # obs in phi would let a linear w_r capture acceleration without the encoder, which
            # would confound the lever with a much cheaper change.
            phi = phi_features(emb_buf, obs, actions)         # (T, B, sf_dim)

            # ---- truncated-MC scaffolding (moved AHEAD of Lambda) --------------------
            # `avail[t]` counts reward terms actually present in mc_ret[t]: it resets at a
            # reset boundary and at the rollout tail, so masking on avail >= mc_window
            # removes BOTH the boundary bias and the tail bias in one condition. The EV
            # gate below still consumes mc_ret/mc_mask; the tau estimator needs them here.
            #
            mc_ret = torch.zeros_like(rewards)
            mc_avail = torch.zeros_like(rewards)
            mc_run = torch.zeros_like(rewards[0])
            avail_run = torch.zeros_like(rewards[0])
            for t in reversed(range(args.num_steps)):
                cont = 1.0 - transition_boundaries[t]
                mc_run = rewards[t] + args.gamma * cont * mc_run
                avail_run = 1.0 + cont * avail_run
                mc_ret[t] = mc_run
                mc_avail[t] = avail_run
            mc_mask = (mc_avail >= args.mc_window).reshape(-1)
            n_mc = int(mc_mask.sum().item())

            # ---- Infinite-horizon successor innovation ------------------------------
            # Psi is the expected full geometric feature occupancy. A single Bellman
            # innovation therefore carries long-horizon information without sampling a
            # trace of future behavior actions into the current score term.
            sf_target, sf_residual = successor_innovation(
                phi,
                psi_cur,
                psi_next,
                transition_terminations,
                transition_valids,
                args.gamma,
            )
            adv_vector = sf_residual @ w_r

            # A direct reward TD innovation is used only on iteration one, while w_r is
            # necessarily zero. Thereafter the policy is driven entirely by the embedding
            # innovation. No recursive trace exists in either path.
            scalar_td = (
                rewards
                + args.gamma
                * next_transition_values
                * (1.0 - transition_terminations)
                * transition_valids
                - values
            )
            advantages = adv_vector if w_r_solved else scalar_td
            _a, _b = adv_vector.reshape(-1), scalar_td.reshape(-1)
            adv_vec_corr = float(
                ((_a - _a.mean()) * (_b - _b.mean())).mean()
                / (_a.std().clamp_min(1e-12) * _b.std().clamp_min(1e-12))
            )
            returns = scalar_td + values
            policy_adv = advantages
            # Batch-level percentile advantage normalization (scopes "ema" and "batch"). Both compute the
            # whole-rollout P5/P95 once and scale policy_adv by one S; "ema" smooths the percentiles with a
            # global EMA across iterations (v1), "batch" uses the FRESH per-rollout spread (no EMA -- the
            # batch-vs-mb ablation). scope=="minibatch" SKIPS this and scales fresh per-mb in the update loop,
            # leaving policy_adv RAW here. Divide-only. NOTE `returns` is NOT the critic target in this
            # file (the critic regresses sf_target); it is the scalar TD target, kept for
            # the percentile scale and the EV diagnostics.
            if args.ret_percnorm and args.ret_perc_scope in ("ema", "batch"):
                flat_ret = returns.reshape(-1)
                qs = torch.tensor([args.ret_perc_lo, args.ret_perc_hi], device=device)
                lo, hi = torch.quantile(flat_ret, qs).tolist()
                if args.ret_perc_scope == "ema":
                    if not ema_perc_inited:
                        ema_ret_lo, ema_ret_hi, ema_perc_inited = lo, hi, True
                    else:
                        r = args.ret_perc_rate
                        ema_ret_lo += r * (lo - ema_ret_lo)
                        ema_ret_hi += r * (hi - ema_ret_hi)
                    spread = ema_ret_hi - ema_ret_lo
                else:  # "batch": fresh whole-rollout percentile spread, no EMA
                    spread = hi - lo
                ret_perc_scale = max(args.ret_perc_floor, spread)
                policy_adv = policy_adv / ret_perc_scale
            # Closed-form w_r on IMMEDIATE reward -- no bootstrapping, no multi-step. This
            # is the only scalar regression anywhere in the design. Solved after the innovation so
            # values[] and next_transition_values[] both use the same (previous) w_r and the
            # advantage stays self-consistent.
            flat_phi = phi.reshape(-1, sf_dim)
            flat_rew = rewards.reshape(-1)
            w_r = solve_reward_probe(flat_phi, flat_rew, args.sf_ridge)
            w_r_solved = True
            reward_resid = flat_rew - flat_phi @ w_r
            reward_r2 = 1.0 - (reward_resid.var() / flat_rew.var().clamp_min(1e-12)).item()
            # R^2 alone is the WRONG gate: r_t uses the frame-skip AVERAGED velocity while the
            # observation carries instantaneous qvel, so a structural residual of ~1-3% is
            # expected and harmless IF it is white. What costs EV is a gait-phase-locked,
            # speed-correlated residual, which shows up as autocorrelation, not as R^2.
            resid_tb = reward_resid.reshape(args.num_steps, args.num_envs)
            resid_c = resid_tb - resid_tb.mean()
            resid_var = resid_c.var().clamp_min(1e-12)
            reward_resid_ac = [
                ((resid_c[:-k] * resid_c[k:]).mean() / resid_var).item() for k in (1, 5, 10, 20)
            ]

            # Per-dimension EMA standardization of the target (see setup for why).
            # The 1/count warmup is NOT cosmetic. e's own scale moves fast for the first
            # few dozen iterations while SIGReg whitens it, but 1/sf_target_ema = 100
            # iterations is 3.3M steps of lag -- the standardization would spend a third of
            # the run tracking a distribution the encoder left behind. rate = 1/count is an
            # exact running mean early (and an exact init at count == 1, replacing a
            # separate first-iteration branch) and decays into the plain EMA once the
            # encoder settles.
            sf_stat_count += 1
            sf_rate = max(args.sf_target_ema, 1.0 / sf_stat_count)
            flat_tgt = sf_target.reshape(-1, sf_dim)
            tgt_mean, tgt_std = flat_tgt.mean(0), flat_tgt.std(0).clamp_min(1e-6)
            sf_mean = sf_mean + sf_rate * (tgt_mean - sf_mean)
            sf_std = sf_std + sf_rate * (tgt_std - sf_std)

            # MTP: horizon h regresses Lambda_{t+h}. Masks are the base's, verbatim -- a
            # future target is valid only when no reset boundary lies between source and
            # target state and it stays inside the rollout.
            mtp = args.critic_mtp_horizon
            sf_mtp = sf_target.new_zeros((args.num_steps, args.num_envs, mtp, sf_dim))
            return_mtp_mask = torch.zeros(
                (args.num_steps, args.num_envs, mtp), dtype=torch.bool, device=device
            )
            for h in range(mtp):
                valid_len = args.num_steps - h
                if valid_len <= 0:
                    break
                valid_h = torch.ones((valid_len, args.num_envs), dtype=torch.bool, device=device)
                for k in range(h):
                    valid_h &= transition_boundaries[k : k + valid_len] == 0
                sf_mtp[:valid_len, :, h] = sf_target[h : h + valid_len]
                return_mtp_mask[:valid_len, :, h] = valid_h
            sf_mtp = (sf_mtp - sf_mean) / sf_std              # head regresses standardized units

            # ---- three-way EV gate: the falsifier, computed on an UNRIGGED target ----
            # The usual EV target `b_returns = advantages + values` CONTAINS the critic's
            # own bootstrapped values, and at dt=0.05 the errors are strongly
            # state-autocorrelated -- a critic scores well against a target built from
            # itself. Every predictor below is scored against the same truncated
            # Monte-Carlo discounted return instead.
            #
            # mc_ret / mc_avail / mc_mask are computed EARLIER now (the per-dimension
            # tau estimator needs them before Lambda is formed). Only the probe-residual
            # recursion stays here: it needs reward_resid, i.e. the w_r solved this
            # iteration, which does not exist until after Lambda.
            mc_resid = torch.zeros_like(rewards)          # discounted PROBE residual
            resid_run = torch.zeros_like(rewards[0])
            resid_tb2 = reward_resid.reshape(args.num_steps, args.num_envs)
            for t in reversed(range(args.num_steps)):
                cont = 1.0 - transition_boundaries[t]
                resid_run = resid_tb2[t] + args.gamma * cont * resid_run
                mc_resid[t] = resid_run
            if n_mc >= 256:
                flat_mc = mc_ret.reshape(-1)[mc_mask]
                feat_mc = critic_feat_buf[mc_mask]
                ones_mc = feat_mc.new_ones(feat_mc.shape[0], 1)
                # (1) reference: what a scalar critic reading THIS trunk could achieve.
                #     Detached closed-form probe, so no scalar-return gradient ever enters
                #     the network -- it replaces the in-run HL-Gauss comparator that the
                #     head swap removed.
                trunk_feat_mc = torch.cat([feat_mc, ones_mc], dim=-1)
                ev_trunk_probe = ev_score(
                    trunk_feat_mc @ solve_reward_probe(trunk_feat_mc, flat_mc, args.sf_ridge), flat_mc
                )
                # (3) how much reward-relevant signal is LINEARLY present in the
                #     INSTANTANEOUS embedding. Not a ceiling on psi: psi reads the trunk,
                #     not e_t, and accumulates phi over time, so (2) > (3) is expected and
                #     is in fact the successor-feature construction earning its keep.
                #     Treat (3) as a floor-diagnostic on the encoder, not an upper bound.
                e_mc = torch.cat([emb_buf.reshape(-1, args.emb_dim)[mc_mask], ones_mc], dim=-1)
                ev_latent_cap = ev_score(
                    e_mc @ solve_reward_probe(e_mc, flat_mc, args.sf_ridge), flat_mc
                )
                # THE decisive reward-probe metric, and the reason R^2 and the AC lags are
                # both kept only as secondary reads. Since V = w_r.Lambda and
                # Lambda = sum gamma^k phi, we have w_r.Lambda = G_t - sum gamma^k eps_{t+k}:
                # the value error IS the discounted sum of probe residuals. R^2 alone is
                # actively misleading here -- measured offline, adding s' to phi raises R^2
                # from 0.61 to 0.94 while making THIS number WORSE (0.77 -> 0.84), because
                # the discounted sum amplifies a small correlated residual far more than a
                # large white one. Ratio to the value's own spread; lower is better.
                value_err_frac = float(
                    mc_resid.reshape(-1)[mc_mask].std() / flat_mc.std().clamp_min(1e-12)
                )
                # (2) the treatment: the ONLINE value the actor actually consumed.
                ev_sf = ev_score(values.reshape(-1)[mc_mask], flat_mc)
                # THE DIRECT H1 TEST, and a much sharper one than cla/baseline_gap: both
                # numbers predict the SAME truncated-MC target, one from the learned critic
                # and one after the model-based expansion. ev_sf > ev_sf_learned is evidence
                # the counterfactual model knows something the critic does not; ev_sf <
                # ev_sf_learned means the model is injecting bias and cla_beta is too high.
                # At beta=0 (or a closed gate) the two are the same tensor and must agree.
                ev_sf_learned = ev_score(values_learned.reshape(-1)[mc_mask], flat_mc)
                # (1) and (3) are IN-SAMPLE ridge fits, i.e. optimistic upper bounds; (2) is
                # an honest out-of-sample prediction. So (3) vs (1) is the apples-to-apples
                # comparison -- (3) << (1) means the LATENT is the bottleneck and no amount
                # of psi quality can rescue this. (2) << (3) with (3) ~ (1) instead points
                # at the SF machinery (recursion, standardization), not the encoder.
            else:
                ev_trunk_probe = ev_latent_cap = ev_sf = value_err_frac = float("nan")
                ev_sf_learned = float("nan")

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = policy_adv.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_sf_target = sf_mtp.reshape(-1, args.critic_mtp_horizon, sf_dim)
        b_target_mask = return_mtp_mask.reshape(-1, args.critic_mtp_horizon)
        # Shape the successor innovation per `adv_transform`. With
        # adv_transform_scope="minibatch" the shaping is deferred into the update
        # loop (recomputed per minibatch); the batch shaping below is then used
        # only for the diagnostics. norm_adv_scope="batch" standardizes the shaped
        # advantage once here instead of per minibatch.
        advantage = b_advantages
        b_policy_adv = shape_advantage(advantage, args, device)
        if args.norm_adv and args.norm_adv_scope == "batch":
            b_policy_adv_normed = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        elif args.norm_adv and args.norm_adv_scope == "batch_retstd":
            # retstd: divide-only by the return std (v18 batch_retstd port). Preserves sign/center,
            # scales by the same return-spread family as ret_percnorm but using std not P95-P5.
            b_policy_adv_normed = b_policy_adv / b_returns.std().clamp(min=args.ret_perc_floor)
        # Diagnostics: how different is the shaped advantage from the raw innovation?
        az = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        pz = (b_policy_adv - b_policy_adv.mean()) / (b_policy_adv.std() + 1e-8)
        adv_corr = (az * pz).mean().item()
        adv_sign_agree = (torch.sign(az) == torch.sign(pz)).float().mean().item()

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, value_sf = agent.get_action_and_value(
                    b_obs[mb_inds], b_latent_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.adv_transform_scope == "minibatch":
                    mb_raw_adv = shape_advantage(b_advantages[mb_inds], args, device)
                else:
                    mb_raw_adv = b_policy_adv[mb_inds]
                mb_advantages = mb_raw_adv
                if args.norm_adv:
                    if args.norm_adv_scope in ("batch", "batch_retstd"):
                        mb_advantages = b_policy_adv_normed[mb_inds]
                    elif args.norm_adv_scope == "minibatch_retstd":
                        # per-mb analog of batch_retstd: divide-only by THIS minibatch's return std
                        # (preserves sign/center; local & reactive, like the per-mb percentile norm below).
                        mb_advantages = mb_advantages / b_returns[mb_inds].std().clamp(min=args.ret_perc_floor)
                    else:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                # v2 per-minibatch percentile norm: scale by S = max(floor, P95-P5) of THIS minibatch's
                # returns, recomputed fresh each minibatch (no EMA). Same statistic/divide-only as v1's
                # global EMA, but local & reactive -- the per-mb analog of the old advnorm.
                if args.ret_percnorm and args.ret_perc_scope == "minibatch":
                    mb_ret = b_returns[mb_inds]
                    qs = torch.tensor([args.ret_perc_lo, args.ret_perc_hi], device=mb_ret.device)
                    lo, hi = torch.quantile(mb_ret, qs)
                    mb_perc_scale = torch.clamp(hi - lo, min=args.ret_perc_floor)
                    mb_advantages = mb_advantages / mb_perc_scale
                    ret_perc_scale = mb_perc_scale.item()

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
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + clip_hi)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # SUCCESSOR-FEATURE value loss: per-horizon masked MSE against the
                # standardized successor target, summed over valid horizons per row.
                # No scalar-return regression anywhere at the default sf_alpha=1.
                sf_tgt = b_sf_target[mb_inds]
                sf_err = value_sf - sf_tgt                                    # (B, mtp, sf_dim)
                value_mask = b_target_mask[mb_inds].to(sf_err.dtype)          # (B, mtp)
                v_loss = (sf_err.pow(2).mean(-1) * value_mask).sum(dim=-1).mean()
                if args.sf_alpha < 1.0:
                    # Reward-direction term. Its target still comes from the latent
                    # successor innovation, so the bootstrap stays in latent space and the
                    # thesis is intact; sf_alpha only decides how much of psi's capacity is
                    # spent on the one direction that is actually read out.
                    scalar_err = (sf_err @ (w_r * sf_std)).pow(2)             # (B, mtp)
                    v_loss = args.sf_alpha * v_loss + (1.0 - args.sf_alpha) * (
                        scalar_err * value_mask
                    ).sum(dim=-1).mean()
                # Per-horizon psi MSE (last minibatch of the last epoch is what gets logged).
                sf_per_h_mse = (sf_err.detach().pow(2).mean(-1) * value_mask).sum(0) / value_mask.sum(
                    0
                ).clamp_min(1)

                entropy_loss = entropy.mean()

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

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ---- LeJEPA SSL step -----------------------------------------------------------
        # ONCE PER ITERATION, OUTSIDE the 320-minibatch PPO loop: inside it costs +100-200%
        # wall clock, outside it costs +10-20%. Placed AFTER the PPO update so the encoder
        # frame is frozen across the entire target-construction + critic-fitting phase. The
        # residual drift is one ITERATION's worth (ssl_steps_target = 64
        # steps at defaults, not one), which is exactly what ssl/frame_drift_* measures.
        with torch.no_grad():
            # obs_hist, not obs: the SAME stack the phi/emb_buf path used above.
            seq_obs = chunk_sequences(obs, args.seq_len)
            seq_act = chunk_sequences(actions, args.seq_len)
            # Masking ONLY the crossing transition (1 - boundaries[l]) is not enough: chunks
            # are cut on fixed multiples of seq_len in rollout time, not on episode
            # boundaries, so a reset at intra-chunk position 0 still leaves later positions
            # predicting normally while causal attention reads position 0 -- a state from the
            # PREVIOUS episode. The CUMPROD requires the whole context to be same-episode,
            # which is the actual precondition for the prediction to be well-posed. Prefix
            # positions before the first boundary are kept, so a straddling chunk is
            # degraded, not discarded.
            # Passed to the SSL forward UNSLICED now: with an AR rollout each horizon needs a
            # different window of it, so the slicing belongs where the horizons are known.
            seq_cont = chunk_sequences(1.0 - transition_boundaries, args.seq_len)
            seq_cum_cont = seq_cont.cumprod(dim=1)
        n_seq = seq_obs.shape[0]
        ssl_pred_l = ssl_sig_l = ssl_gn_sum = 0.0
        ssl_steps = 0
        # Accumulated as a TENSOR on device and synced once at the end of the iteration,
        # rather than .item()'d inside the 64-step inner loop.
        ssl_per_h = torch.zeros(args.pred_horizon, device=device)
        # e BEFORE this iteration's SSL step, on states from THIS rollout. Paired with the
        # post-step embedding of the SAME inputs below, this isolates frame movement from
        # distribution shift without needing to keep a copy of the old encoder.
        probe_emb_before = emb_buf.reshape(-1, args.emb_dim)[drift_probe_idx].clone()
        # Derived so the SSL gradient budget is invariant to pred_horizon -- see
        # ssl_steps_target. At K=1 this reproduces v2 exactly: 8192//1024 = 8 steps/epoch,
        # 64/8 = 8 epochs.
        ssl_steps_per_epoch = max(1, n_seq // args.ssl_batch)
        ssl_n_epochs = max(1, round(args.ssl_steps_target / ssl_steps_per_epoch))
        for _ in range(ssl_n_epochs):
            perm = torch.randperm(n_seq, device=device)
            # DROP-LAST, not for tidiness: the SIGReg statistic scales with the batch size
            # (it multiplies by proj.size(-2)), so a ragged final minibatch would silently
            # reweight the regularizer -- and under dynamic=False it also forces a recompile.
            for s in range(0, n_seq - args.ssl_batch + 1, args.ssl_batch):
                idx = perm[s : s + args.ssl_batch]
                ssl_loss, pred_l, sig_l, per_h = ssl(
                    seq_obs[idx], seq_act[idx], seq_cum_cont[idx], args.sigreg_weight
                )
                ssl_optimizer.zero_grad(set_to_none=True)
                ssl_loss.backward()
                ssl_gn = nn.utils.clip_grad_norm_(ssl.parameters(), args.ssl_grad_clip)
                ssl_optimizer.step()
                ssl_pred_l += pred_l.item()
                ssl_sig_l += sig_l.item()
                ssl_per_h += per_h
                ssl_gn_sum += float(ssl_gn)
                ssl_steps += 1
        # ---- one-step dynamics fit ------------------------------------------------------
        # Same schedule as the SSL step: once per iteration, OUTSIDE the PPO minibatch loop,
        # and on the rollout that has just been consumed. Trained on the DELTA and scored by
        # R^2 against the delta's own variance, NOT against s' -- an R^2 computed against s'
        # would read ~0.99 for the trivial identity predictor and the gate would pass on a
        # model that had learned nothing. The continuation head shares the body and is
        # trained by BCE on the SAME batch (see dyn_keep for why the mask changed).
        with torch.no_grad():
            dyn_s = obs.reshape(flat_obs_shape)
            dyn_a = actions.reshape(-1, act_dim_sf)
            dyn_next = next_obses.reshape(flat_obs_shape)
            dyn_delta = dyn_next - dyn_s
            dyn_term = transition_terminations.reshape(-1)
            # Keep on `valids`, NOT on `not boundary`. next_obses carries the TRUE successor
            # at a boundary (final_observation is substituted in the rollout, and valid is set
            # to 1 there); it is only the reset observation when no final_observation was
            # available, which is exactly what valid=0 marks. Masking boundaries out would
            # throw away real physics AND leave the continuation head with no positives at
            # all, so it could never learn that anything terminates.
            dyn_keep = transition_valids.reshape(-1) > 0.5
            dyn_s, dyn_a = dyn_s[dyn_keep], dyn_a[dyn_keep]
            dyn_delta, dyn_term = dyn_delta[dyn_keep], dyn_term[dyn_keep]
        # HELD-OUT split. The gate exists to stop a bad model from entering the baseline, so
        # it has to read a generalisation number: an in-sample R^2 on 32k points fit by a
        # 256-wide MLP would open the gate on memorisation. Per-dimension R^2 averaged, not
        # a pooled ratio -- pooled, one loud dimension (root x-velocity) can carry the score
        # while the joint angles the model actually needs for phi are unlearned.
        n_all = dyn_s.shape[0]
        n_val = n_all // 10
        if n_all - n_val >= args.dyn_batch:
            split = torch.randperm(n_all, device=device)
            va, tr = split[:n_val], split[n_val:]
            vs, vaa, vd, vt = dyn_s[va], dyn_a[va], dyn_delta[va], dyn_term[va]
            dyn_s, dyn_a = dyn_s[tr], dyn_a[tr]
            dyn_delta, dyn_term = dyn_delta[tr], dyn_term[tr]
            n_dyn = dyn_s.shape[0]
            steps_per_epoch = max(1, n_dyn // args.dyn_batch)
            for _ in range(max(1, round(args.dyn_steps_target / steps_per_epoch))):
                perm = torch.randperm(n_dyn, device=device)
                for st_ in range(0, n_dyn - args.dyn_batch + 1, args.dyn_batch):
                    idx = perm[st_ : st_ + args.dyn_batch]
                    pred, cont_logit = dyn(dyn_s[idx], dyn_a[idx])
                    dyn_loss = F.mse_loss(pred, dyn_delta[idx]) + F.binary_cross_entropy_with_logits(
                        cont_logit, 1.0 - dyn_term[idx]
                    )
                    dyn_optimizer.zero_grad(set_to_none=True)
                    dyn_loss.backward()
                    nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
                    dyn_optimizer.step()
            with torch.no_grad():
                vpred, vlogit = dyn(vs, vaa)
                resid = vpred - vd
                # Mean-SQUARED residual, not resid.var(0): a constant per-dimension offset is
                # a real prediction error and var() would score it as unbiased, opening the
                # gate on a biased model. Clamped per dimension before averaging -- R^2 is
                # unbounded below, so one dimension whose delta variance hits the floor could
                # otherwise contribute ~-1e12 and pin the mean shut for the rest of the run.
                dyn_r2_last = float(
                    (1.0 - resid.pow(2).mean(0) / vd.var(0).clamp_min(1e-8))
                    .clamp(-1.0, 1.0)
                    .mean()
                )
                dyn_mse_last = float(resid.pow(2).mean())
                dyn_cont_acc = float(
                    ((vlogit > 0).float() == (1.0 - vt)).float().mean()
                )
        ssl_pred_l /= max(ssl_steps, 1)
        ssl_sig_l /= max(ssl_steps, 1)
        ssl_gn_sum /= max(ssl_steps, 1)

        # ---- encoder health + frame drift ----------------------------------------------
        # SIGReg pins the DISTRIBUTION of e, not its coordinate frame: N(0,I) is
        # rotation-invariant and the prediction loss cannot pin the frame either (the
        # predictor co-rotates). But psi_old and w_r are functions of COORDINATES, so a
        # drifting frame makes the bootstrap target stale. This is a non-stationarity the
        # design INTRODUCES, so it is measured rather than pre-emptively patched.
        #   frame_drift_raw ~ 1 and frame_drift_rot ~ 1  -> stable, bootstrap is sound
        #   frame_drift_raw << 1 but frame_drift_rot ~ 1 -> pure rotation, the failure mode
        # frame_drift_rot is in [0,1] by von Neumann's trace inequality; frame_drift_raw is
        # in [-1,1] (it goes negative under anti-correlation). Both are 1 iff no drift.
        #
        # Both embeddings are of the SAME inputs drawn from THIS rollout, taken before and
        # after this iteration's SSL step. A probe frozen at iteration 1 would be wrong in
        # two ways: those are post-NormalizeObservation vectors carrying iteration-1
        # running stats, so by mid-training they decode to raw states the agent never
        # visits -- and OOD inputs drift MORE, so the metric would cry wolf about a frame
        # instability the actual bootstrap never sees.
        with torch.no_grad():
            # obs_hist, matching probe_emb_before (which came from emb_buf, itself built from
            # obs_hist). Re-encoding raw obs here would silently compare embeddings of two
            # DIFFERENT inputs under hist_len>1 and report frame drift that is really a
            # difference in input width.
            probe_emb_after = ssl.encoder(obs.reshape(flat_obs_shape)[drift_probe_idx]).float()
            a = probe_emb_before - probe_emb_before.mean(0)
            b_ = probe_emb_after - probe_emb_after.mean(0)
            na, nb = a.pow(2).sum(), b_.pow(2).sum()
            denom = (0.5 * (na + nb)).clamp_min(1e-12)
            drift_raw = float(1.0 - (b_ - a).pow(2).sum() / (2.0 * denom))
            drift_rot = float(torch.linalg.svdvals(b_.T.double() @ a.double()).sum() / denom.double())
            # e's marginal: SIGReg should drive these to (0, I). Read from the POST-step
            # embedding, so this panel and ssl/sigreg_epps_pulley describe the same encoder
            # -- emb_buf is the PRE-step encoder and would lag by a full 64 SSL steps,
            # making the very first ssl/emb_std point describe the random init.
            # Effective rank is expected near 17 (the observation dimension), NOT 32: a
            # single-state encoder cannot exceed the manifold's rank, and that is not a
            # failure -- random 1-D projections of a rank-17 pushforward still look
            # near-Gaussian by mixing, so the statistic converges anyway.
            emb_mean_abs = float(probe_emb_after.mean(0).abs().mean())
            emb_std = float(probe_emb_after.std(0).mean())
            cov_e = (b_.T @ b_) / b_.shape[0]
            eig = torch.linalg.eigvalsh(cov_e.double()).clamp_min(0)
            p_eig = eig / eig.sum().clamp_min(1e-12)
            eff_rank = float(torch.exp(-(p_eig * (p_eig + 1e-12).log()).sum()))

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/ret_perc_scale", ret_perc_scale, global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/actor_grad_norm", float(actor_gn), global_step)
        writer.add_scalar("losses/critic_grad_norm", float(critic_gn), global_step)
        writer.add_scalar("debug/returns_mean", b_returns.mean().item(), global_step)
        writer.add_scalar("debug/returns_std", b_returns.std().item(), global_step)
        writer.add_scalar("debug/returns_absmax", b_returns.abs().max().item(), global_step)
        # corr≈1 -> shaping adds little; lower -> the
        # transform genuinely changes the gradient direction/magnitude.
        writer.add_scalar("debug/shaped_adv_corr", adv_corr, global_step)
        writer.add_scalar("debug/distpg_sign_agree", adv_sign_agree, global_step)

        # ---- successor-feature diagnostics -------------------------------------
        # R^2 is the weak gate; a structural ~1-3% residual is expected (frame-skip
        # averaged reward velocity vs instantaneous qvel). The residual's
        # AUTOCORRELATION is what actually costs EV: white residual at R^2=0.98
        # costs ~0.2% EV, a gait-phase-locked one at the same R^2 costs ~2%.
        writer.add_scalar("sf/value_err_frac", value_err_frac, global_step)
        writer.add_scalar("sf/reward_probe_r2", reward_r2, global_step)
        for lag, ac in zip((1, 5, 10, 20), reward_resid_ac):
            writer.add_scalar(f"sf/reward_resid_ac_lag{lag}", ac, global_step)
        for h in range(args.critic_mtp_horizon):
            writer.add_scalar(f"sf/psi_mse_h{h}", sf_per_h_mse[h].item(), global_step)
        writer.add_scalar("sf/successor_std_emb", sf_std[: args.emb_dim].mean().item(), global_step)
        # The raw-state block's scale is pinned by NormalizeObservation, not SIGReg; watch
        # it for the slow normalizer drift that replaces v1's (dead) frame-drift risk.
        writer.add_scalar("sf/successor_std_obs",
                          sf_std[args.emb_dim : args.emb_dim + obs_dim].mean().item(), global_step)
        writer.add_scalar("sf/successor_std_act",
                          sf_std[args.emb_dim + obs_dim : -1].mean().item(), global_step)
        writer.add_scalar("sf/successor_absmean_emb", sf_mean[: args.emb_dim].abs().mean().item(), global_step)
        writer.add_scalar("sf/w_r_norm", w_r.norm().item(), global_step)

        writer.add_scalar("innovation/vector_scalar_td_corr", adv_vec_corr, global_step)
        writer.add_scalar("innovation/vector_std", adv_vector.std().item(), global_step)
        writer.add_scalar("innovation/scalar_td_std", scalar_td.std().item(), global_step)

        # ---- the falsifier: EV against truncated-MC returns, three predictors ----------
        writer.add_scalar("gate/ev_sf_online", ev_sf, global_step)          # (2) treatment
        writer.add_scalar("gate/ev_trunk_probe", ev_trunk_probe, global_step)  # (1) reference
        writer.add_scalar("gate/ev_latent_cap", ev_latent_cap, global_step)    # (3) ceiling
        writer.add_scalar("gate/mc_frac", n_mc / (args.num_steps * args.num_envs), global_step)

        # ---- SSL path -------------------------------------------------------------------
        writer.add_scalar("cla/ev_sf_learned", ev_sf_learned, global_step)
        # Monte-Carlo standard error of the computed baseline, as a FRACTION of the advantage
        # spread it is differenced against. This is the price of finite M, and it is the
        # number that decides whether a null result means "the model knows nothing" or "the
        # signal was buried under sampling noise". Scaled by cla_beta because that is how
        # much of psi_cf actually reaches the advantage.
        writer.add_scalar("cla/cf_se", cf_se, global_step)
        writer.add_scalar(
            "cla/cf_se_frac", cf_se / max(float(b_advantages.std()), 1e-12), global_step
        )
        writer.add_scalar("cla/dyn_r2", dyn_r2_last, global_step)
        writer.add_scalar("cla/dyn_mse", dyn_mse_last, global_step)
        writer.add_scalar("cla/dyn_cont_acc", dyn_cont_acc, global_step)
        writer.add_scalar("cla/beta_eff", b_eff if args.cla_m > 0 else 0.0, global_step)
        # H1's falsifier. If the computed baseline never differs from the learned one, the
        # model adds nothing the critic did not already know and this is a null experiment.
        writer.add_scalar("cla/baseline_gap", cla_gap, global_step)
        writer.add_scalar("ssl/pred_loss", ssl_pred_l, global_step)
        # THE diagnostic for lever 1. Horizon-1 error alone cannot tell a rollable encoder
        # from a one-step-memorizing one; the RATIO h{K}/h1 is the compounding rate, and it
        # is the number that says whether the AR pressure did anything. If it sits near 1.0
        # the rollout is not actually compounding and the shift is wrong.
        per_h_mean = (ssl_per_h / max(ssl_steps, 1)).tolist()
        for h, val in enumerate(per_h_mean, start=1):
            writer.add_scalar(f"ssl/pred_loss_h{h}", val, global_step)
        if len(per_h_mean) > 1:
            writer.add_scalar(
                "ssl/pred_compound_ratio", per_h_mean[-1] / max(per_h_mean[0], 1e-12), global_step
            )
        writer.add_scalar("ssl/sigreg_epps_pulley", ssl_sig_l, global_step)
        writer.add_scalar("ssl/steps_per_iter", ssl_steps, global_step)
        writer.add_scalar("ssl/grad_norm", ssl_gn_sum, global_step)
        writer.add_scalar("ssl/emb_mean_abs", emb_mean_abs, global_step)
        writer.add_scalar("ssl/emb_std", emb_std, global_step)
        writer.add_scalar("ssl/emb_effective_rank", eff_rank, global_step)
        writer.add_scalar("ssl/frame_drift_raw", drift_raw, global_step)
        writer.add_scalar("ssl/frame_drift_rot", drift_rot, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
