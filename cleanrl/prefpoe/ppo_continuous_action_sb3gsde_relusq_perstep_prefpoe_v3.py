# PPO + SB3 gSDE per-step + ReLU² MLP + PrefPoE v3 (relusq ablation)
#
# Ablation of `ppo_continuous_action_sb3gsde_tanh_perstep_prefpoe_v3.py`:
# only change is the non-action hidden activations — Tanh → ReLU²
# (a.k.a. ReluSq, f(x) = relu(x)²). Applied to:
#   - critic (2 hidden layers)
#   - actor_latent (shared encoder feeding actor_mean, log_std, pref_mean)
# Action distributions, gSDE machinery, PoE fusion, and PrefPoE losses
# are identical to the Tanh v3. Hypothesis: ReLU² gives a sharper non-
# linearity → richer features. Tanh saturates at ±1 and squashes high-
# magnitude features that gSDE's `latent_sde² @ exp(log_std)²` relies on
# for state-conditioned exploration scale.
#
# Original v3 header follows.
# ---
# PPO + SB3 gSDE per-step + Tanh MLP + PrefPoE v3
#
# v3 motivation: v2 stuck at −600 returns for 8M steps. Three independent
# analyses converged on the same root cause and fix.
#
# v2 collapse diagnosis:
#   pref_log_std stable, but main_entropy GREW 5.6 → 13.1 (σ_main runaway upward).
#   KL(fused, main) grew 0.25 → 79 (fused decoupled from main entirely).
#   By end of run, τ_m/τ_f = 0.125/2.125 = 5.9% — main contributed only 6% of
#   the fused precision; 94% came from pref. Main was gradient-starved.
#
# Root cause (PoE precision-saturation gradient starvation):
#   σ_fused² = 1/(1/σ_main² + λ/σ_pref²) saturates at σ_pref/√λ as σ_main → ∞.
#   Every gradient on log σ_main from a fused-conditioned loss factors as
#     (τ_m/τ_f) · g(...)
#   so as σ_main grows, the prefactor → 0 — main's σ becomes a flat direction.
#   Adam noise + entropy-bonus pressure walk σ_main upward without restoring
#   force; the upward walk further reduces τ_m/τ_f, deepening the starvation.
#   μ_main is attenuated by the same factor (94% of action mean came from pref).
#   The paper's Figure 2(b) shows main entropy DECAYING monotonically: clear
#   evidence main IS trained directly in the paper, not via fused.
#
# v3 fix (paper-faithful re-reading):
#   1) PPO ratio computed under π_MAIN, not π_fused.
#      Sampling stays under fused (paper §3.4 explicit). But the PPO surrogate
#      restores its full-strength `(z²−1)` gradient on log σ_main — no τ_m/τ_f
#      prefactor — so σ_main is properly trained.
#      This is the off-policy IS-correct formulation; relies on KL(fused,main)
#      staying small (enforced by cons_loss).
#   2) tanh on pref_mean before PoE fusion.
#      Reference implementation in the paper's OpenReview supplementary
#      (vis.py:41-115) applies tanh to BOTH actor_mean and pref_mean before
#      fusion. We keep main_mean un-tanh'd (preserving gSDE's interface) but
#      add tanh to pref_mean to bound its contribution.
#   3) Clamp σ_fused at [1e-3, 2.0] (reference impl defensive cap).
#      Prevents fused inheriting unbounded growth from any single component.
#   4) Hyperparameters moved into the paper's disclosed ranges (Table A.1):
#      β₁         = 0.2   (was 1.0; paper range [0.1, 0.4])
#      α_entropy  = 0.2   (was 0.3; paper range [0.1, 0.4])
#      λ_pref     = 0.5   (paper range [0.2, 0.8])
#      w_pref     = 0.05  (was 1.0; paper range [0.005, 0.2])
#      w_cons     = 0.1   (paper range [0.02, 0.2])
#      pref_log_std init = −1.0 (was −0.5; matches reference impl)
#   5) target_kl removed (None). With ratio under main and stable σ_main,
#      KL is meaningful again; the 0.02 cap in v2 was triggering early-stops.
#
# Earlier history (chronological):
#   v1: state-dep pref_log_std → σ_pref runaway DOWN under fused-sampling
#       (advantage-gradient on log σ stays positive when samples ≠ from pref)
#   v2: state-indep pref_log_std fixed σ_pref but exposed the OTHER half of
#       the precision-saturation degeneracy — σ_main runaway UP
#   v3: ratio-under-main breaks the degeneracy; reference-aligned safety details
#
# Hypothesis:
#   PrefPoE (arXiv 2511.08241) is paper-correct in Gaussian-policy space.
#   Our gSDE baseline already uses a Gaussian marginal `N(μ(s), σ²(s))`, so
#   we can apply PrefPoE's machinery WITHOUT the Beta-adaptation gymnastics
#   that destabilized earlier attempts (cleanrl/hlgauss/.../prefpoe_v132–v135).
#
# Architecture (paper eq 1 + eq 5 + eq 6, applied to gSDE marginal):
#   π_main   = N(μ_main(s), σ_main²(s))      gSDE distribution
#              μ_main(s)    = actor_mean(actor_latent(s))
#              σ_main²(s)_i = Σ_j latent_j(s)² · exp(log_std)_ji²       (gSDE)
#   π_pref   = N(μ_pref(s), σ_pref²(s))      paper eq (1), reference-aligned:
#              μ_pref(s)    = tanh(pref_mean(actor_latent(s)))   (v3: tanh)
#              σ_pref       = exp(pref_log_std)·1               (state-INDEP v2)
#   π_fused ∝ π_main · π_pref^λ_pref         paper eq (5), diagonal closed form:
#              1/σ²_fused_i = 1/σ²_main_i + λ_pref/σ²_pref_i
#              μ_fused_i    = σ²_fused_i · (μ_main_i/σ²_main_i + λ·μ_pref_i/σ²_pref_i)
#              σ_fused      clamped to [1e-3, 2.0]              (v3: ref impl)
#   Behavior: sample from fused (paper §3.4).
#   PPO ratio: computed under π_MAIN (v3 fix). See header notes above.
#
# Losses (paper eq 2 + eq 6):
#   L_PPO   = standard clipped surrogate, with sampling under fused
#   L_pref  = −β₁·E[A_norm · log π_pref(a|s)] − α·H(π_pref)
#   L_cons  = KL(π_fused ‖ π_pref)            (Gaussian closed form)
#   L_total = L_PPO + w_pref·L_pref + w_cons·L_cons
#
# Notes:
#   - We share actor_latent (paper's "shared encoder f_enc").
#   - We do NOT use gSDE noise sampling for the fused action — we sample
#     fused via direct reparam `a = μ_fused + σ_fused · ε`. gSDE's per-step
#     state-consistent noise is preserved only on the MAIN distribution
#     parameters; the actual behavior is fused. (gSDE's exploration matrix
#     structure can't be transferred to fused without solving a non-trivial
#     marginal-to-matrix inverse problem; the paper's PoE sampling is the
#     cleaner choice.)
#   - During the optional warmup (`prefpoe_warmup_steps`), λ_pref=0 so the
#     run is bit-equivalent to the gSDE baseline. After warmup, λ_pref takes
#     its configured value and PrefPoE kicks in.
#   - Gaussian entropy gradient `∂H/∂σ = 1/σ` grows as σ shrinks, providing
#     the self-correction that the Beta variant lacked. No concentration
#     cap needed.
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
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.tensorboard import SummaryWriter


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

    # PPO / env
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    """v3: with ratio under main (not fused), KL is meaningful again. v2's 0.02
       cap was triggering early-stops because fused-fused ratios were inflated
       by σ_main runaway, not by genuine policy lurches. Removed by default."""

    # gSDE (unchanged from baseline)
    gsde_log_std_init: float = -2.0
    full_std: bool = True
    use_expln: bool = False
    learn_sde_features: bool = False
    sde_sample_freq: int = 1

    # PrefPoE (arXiv 2511.08241). Defaults sit in the middle of the paper's
    # Table A.1 disclosed ranges, validated against the OpenReview reference impl.
    lambda_pref: float = 0.5
    """λ_pref in paper eq (5). Paper range [0.2, 0.8]."""
    prefpoe_beta1: float = 0.2
    """β₁ in paper eq (2). Paper range [0.1, 0.4]. v2 used 1.0 (5× too high)."""
    prefpoe_alpha_entropy: float = 0.2
    """α in paper eq (2). Paper range [0.1, 0.4]."""
    prefpoe_w_pref: float = 0.05
    """w_pref in paper eq (6). Paper range [0.005, 0.2]. v2 used 1.0 (20× too high)."""
    prefpoe_w_cons: float = 0.1
    """w_cons in paper eq (6). Paper range [0.02, 0.2]."""
    prefpoe_warmup_steps: int = 0
    """Env steps before PrefPoE engages. Until then λ_pref=0 (= gSDE baseline)."""
    pref_log_std_min: float = -1.5
    """Floor on pref log_std (σ_pref ≥ exp(-1.5) ≈ 0.22). Matches the paper's
       HC pref-entropy range from Fig 2(a) (≈ [2.5, 10] for 6-D Gaussian
       ⇔ σ_pref ≈ [0.49, 1.7], so 0.22 sits just below the paper's empirical floor)."""
    pref_log_std_max: float = 0.5
    """Ceiling on pref log_std (σ_pref ≤ exp(0.5) ≈ 1.65). Matches paper's
       upper end. v1 had pref_log_std_max=2 (σ ≤ 7.4); the head outputs hit
       5+ unclamped, indicating runaway."""
    pref_log_std_init: float = -1.0
    """Initial pref log_std (σ_pref ≈ 0.37). Matches reference impl (vis.py:60).
       v2 used -0.5 (σ_pref ≈ 0.61) which was too wide for HC's actuation range."""

    # filled at runtime
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
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ReluSq(nn.Module):
    """f(x) = relu(x)² — quadratic for x≥0, zero for x<0."""

    def forward(self, x):
        return torch.relu(x).square()


class StateDependentNoiseDistribution:
    """SB3 gSDE mechanics (verbatim from baseline)."""

    def __init__(self, action_dim, latent_sde_dim, full_std=True, use_expln=False, learn_features=False, epsilon=1e-6):
        self.action_dim = int(action_dim)
        self.latent_sde_dim = int(latent_sde_dim)
        self.full_std = bool(full_std)
        self.use_expln = bool(use_expln)
        self.learn_features = bool(learn_features)
        self.epsilon = float(epsilon)
        self.exploration_mat = None
        self.exploration_matrices = None

    def get_std(self, log_std):
        if self.use_expln:
            below_threshold = torch.exp(log_std) * (log_std <= 0)
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            std = torch.exp(log_std)

        if self.full_std:
            return std
        return torch.ones(self.latent_sde_dim, self.action_dim, device=log_std.device, dtype=log_std.dtype) * std

    def sample_weights(self, log_std, batch_size=1):
        std = self.get_std(log_std)
        weights_dist = Normal(torch.zeros_like(std), std)
        self.exploration_mat = weights_dist.rsample()
        self.exploration_matrices = weights_dist.rsample((batch_size,))

    def _latent_for_sde(self, latent_sde):
        return latent_sde if self.learn_features else latent_sde.detach()

    def get_distribution(self, mean_actions, log_std, latent_sde):
        latent_sde = self._latent_for_sde(latent_sde)
        variance = torch.mm(latent_sde.pow(2), self.get_std(log_std).pow(2))
        return Normal(mean_actions, torch.sqrt(variance + self.epsilon))

    def get_noise(self, latent_sde):
        latent_sde = self._latent_for_sde(latent_sde)
        if self.exploration_matrices is None:
            raise RuntimeError("gSDE exploration matrices are uninitialized; call reset_noise() first")
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return torch.mm(latent_sde, self.exploration_mat)
        return torch.bmm(latent_sde.unsqueeze(1), self.exploration_matrices).squeeze(1)

    def sample(self, mean_actions, latent_sde):
        return mean_actions + self.get_noise(latent_sde)


def fuse_diagonal_gaussians(main_dist, pref_dist, lambda_pref, std_min=1e-3, std_max=2.0):
    """Closed-form diagonal Gaussian Product-of-Experts (paper eq 5).

    1/σ²_fused = 1/σ²_main + λ/σ²_pref
    μ_fused    = σ²_fused · (μ_main/σ²_main + λ·μ_pref/σ²_pref)

    σ_fused is clamped to [std_min, std_max] (reference impl: vis.py:108-114).

    Returns a Normal whose loc/scale carry gradient to main AND pref params.
    """
    main_var = main_dist.scale.square()
    pref_var = pref_dist.scale.square()
    inv_main = 1.0 / (main_var + 1e-8)
    inv_pref = 1.0 / (pref_var + 1e-8)
    fused_inv_var = inv_main + lambda_pref * inv_pref
    fused_var = 1.0 / (fused_inv_var + 1e-8)
    fused_mean = fused_var * (main_dist.loc * inv_main + lambda_pref * pref_dist.loc * inv_pref)
    fused_std = fused_var.sqrt().clamp(min=std_min, max=std_max)
    return Normal(fused_mean, fused_std)


class Agent(nn.Module):
    def __init__(
        self,
        envs,
        full_std=True,
        use_expln=False,
        learn_sde_features=False,
        gsde_log_std_init=-2.0,
        pref_log_std_min=-3.0,
        pref_log_std_max=2.0,
        pref_log_std_init=-0.5,
    ):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        latent_dim = 64

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_latent = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 64)),
            ReluSq(),
        )
        # Main (gSDE)
        self.actor_mean = layer_init(nn.Linear(latent_dim, action_dim), std=0.01)
        self.action_dist = StateDependentNoiseDistribution(
            action_dim=action_dim,
            latent_sde_dim=latent_dim,
            full_std=full_std,
            use_expln=use_expln,
            learn_features=learn_sde_features,
        )
        self.register_buffer("gsde_full_std_flag", torch.tensor(float(full_std)))
        self.register_buffer("gsde_use_expln_flag", torch.tensor(float(use_expln)))
        self.register_buffer("gsde_learn_features_flag", torch.tensor(float(learn_sde_features)))
        if full_std:
            self.log_std = nn.Parameter(torch.ones(latent_dim, action_dim) * gsde_log_std_init)
        else:
            self.log_std = nn.Parameter(torch.ones(latent_dim, 1) * gsde_log_std_init)
        self.reset_noise(batch_size=1)

        # Preference head (paper eq 1): state-dependent Gaussian on the shared latent.
        # `pref_mean` is initialized small (std=0.01) like the main mean. `pref_log_std`
        # outputs log σ_pref(s); we initialize its bias to gsde_log_std_init so pref
        # starts at a comparable scale to main.
        self.pref_mean = layer_init(nn.Linear(latent_dim, action_dim), std=0.01)
        # v2: state-INDEPENDENT pref log_std. Single global parameter, like
        # vanilla PPO's `log_std` or gSDE's `log_std`. Avoids the v1 runaway
        # where per-state σ_pref(s) heads got pushed toward 0 by the
        # advantage-weighted log-prob gradient under fused sampling.
        self.pref_log_std = nn.Parameter(torch.full((action_dim,), float(pref_log_std_init)))
        self.pref_log_std_min = float(pref_log_std_min)
        self.pref_log_std_max = float(pref_log_std_max)

    def reset_noise(self, batch_size=1):
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def _has_compatible_noise(self, batch_size, device):
        matrices = self.action_dist.exploration_matrices
        return matrices is not None and matrices.shape[0] == batch_size and matrices.device == device

    def load_state_dict(self, state_dict, strict=True, assign=False):
        saved_log_std = state_dict.get("log_std")
        if saved_log_std is not None and saved_log_std.shape != self.log_std.shape:
            self.log_std = nn.Parameter(torch.empty_like(saved_log_std))

        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.action_dist.full_std = bool(self.gsde_full_std_flag.item())
        self.action_dist.use_expln = bool(self.gsde_use_expln_flag.item())
        self.action_dist.learn_features = bool(self.gsde_learn_features_flag.item())
        self.reset_noise(batch_size=1)
        return result

    def get_value(self, x):
        return self.critic(x)

    def _pref_distribution(self, latent_sde):
        """Paper eq (1) realized on the shared latent.

        v2: pref_log_std is state-INDEPENDENT (single global parameter).
        v3: pref_mean is tanh-squashed to bound its contribution to the fused
            mean (reference impl vis.py:103). pref_log_std is also clamped at
            the FORWARD pass to give a hard upper bound on σ_pref (the parameter
            itself remains unconstrained for gradient flow).
        """
        pref_mu = torch.tanh(self.pref_mean(latent_sde))
        pref_log_std = self.pref_log_std.clamp(self.pref_log_std_min, self.pref_log_std_max)
        pref_std = pref_log_std.exp().expand_as(pref_mu)
        return Normal(pref_mu, pref_std)

    def get_dists(self, x):
        """Compute (main, pref) distributions on a batch of observations.

        Returns latent_sde too because it is needed for gSDE sampling and for
        the PoE/loss math.
        """
        latent_sde = self.actor_latent(x)
        action_mean = self.actor_mean(latent_sde)
        main_dist = self.action_dist.get_distribution(action_mean, self.log_std, latent_sde)
        pref_dist = self._pref_distribution(latent_sde)
        return main_dist, pref_dist, latent_sde

    def get_action_and_value(self, x, action=None, lambda_pref=0.0):
        """Sample/score actions.

        Sampling: when lambda_pref > 0, behavior = π_fused (paper §3.4).
        Scoring (for PPO ratio): always under π_MAIN (v3 fix).

        v3 rationale: the PPO surrogate uses ratio = π_main_new(a)/π_main_old(a)
        so that the full-strength gradient `∂logπ_main/∂θ_main` reaches the main
        params without the τ_m/τ_f attenuation factor that gradient-starved
        v2's σ_main. The off-policy correction is implicit: a IS drawn from fused
        ≠ π_main, but PPO's clipping bounds the IS-weight variance. fused and
        main stay close by mathematical construction (fused = main · pref^λ
        with σ_fused ≤ σ_main); L_cons = KL(π_fused ‖ π_pref) is a separate
        regularizer that prevents pref from drifting away from fused — it does
        NOT directly anchor fused to main.

        Returns: action, logprob_under_MAIN, entropy_under_MAIN, value,
                 main_dist, pref_dist, fused_dist (None if λ==0).
        """
        main_dist, pref_dist, latent_sde = self.get_dists(x)
        if lambda_pref > 0.0:
            fused_dist = fuse_diagonal_gaussians(main_dist, pref_dist, lambda_pref)
        else:
            fused_dist = None

        if action is None:
            if lambda_pref > 0.0:
                # Sample from fused (paper §3.4 — fused defines the behavior).
                # Detached: action selection is a sample, not a differentiable op.
                action = fused_dist.rsample().detach()
            else:
                # gSDE baseline path: structured per-step noise via exploration mat.
                if not self._has_compatible_noise(main_dist.loc.shape[0], main_dist.loc.device):
                    self.reset_noise(batch_size=main_dist.loc.shape[0])
                action = self.action_dist.sample(main_dist.loc, latent_sde).detach()

        # PPO ratio is computed under π_MAIN — the policy whose params we update.
        # This is the v3 fix: restores full-strength `(z²−1)` gradient on log σ_main.
        logprob = main_dist.log_prob(action).sum(1)
        entropy = main_dist.entropy().sum(1)
        return action, logprob, entropy, self.critic(x), main_dist, pref_dist, fused_dist


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

    agent = Agent(
        envs,
        full_std=args.full_std,
        use_expln=args.use_expln,
        learn_sde_features=args.learn_sde_features,
        gsde_log_std_init=args.gsde_log_std_init,
        pref_log_std_min=args.pref_log_std_min,
        pref_log_std_max=args.pref_log_std_max,
        pref_log_std_init=args.pref_log_std_init,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    agent.reset_noise(batch_size=args.num_envs)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # PrefPoE warmup gate: λ=0 makes this iteration bit-equivalent to gSDE baseline.
        prefpoe_active = global_step >= args.prefpoe_warmup_steps
        lambda_pref_now = args.lambda_pref if prefpoe_active else 0.0

        agent.reset_noise(batch_size=args.num_envs)
        for step in range(0, args.num_steps):
            if args.sde_sample_freq > 0 and step % args.sde_sample_freq == 0:
                agent.reset_noise(batch_size=args.num_envs)

            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, _, _, _ = agent.get_action_and_value(
                    next_obs, lambda_pref=lambda_pref_now
                )
                values[step] = value.flatten()
            actions[step] = action
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
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        pref_losses_log = []
        cons_losses_log = []
        main_ent_log = []
        pref_ent_log = []
        fused_ent_log = []
        kl_fused_pref_log = []
        kl_fused_main_log = []
        pref_log_std_mean_log = []
        pref_log_std_min_log = []
        pref_log_std_max_log = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                action, newlogprob, entropy, newvalue, main_dist_mb, pref_dist_mb, fused_dist_mb = (
                    agent.get_action_and_value(
                        b_obs[mb_inds],
                        b_actions[mb_inds],
                        lambda_pref=lambda_pref_now,
                    )
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

                # Clipped PPO surrogate (sampling under fused — paper-faithful).
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Critic loss (clipped).
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                # PrefPoE losses (paper eq 2 + eq 6).
                if prefpoe_active and args.prefpoe_w_pref > 0.0:
                    # a is the action taken under fused; eval its log-prob under pref.
                    pref_logprob = pref_dist_mb.log_prob(b_actions[mb_inds]).sum(dim=-1)
                    pref_loss = -args.prefpoe_beta1 * (
                        mb_advantages.detach() * pref_logprob
                    ).mean()
                    if args.prefpoe_alpha_entropy > 0.0:
                        pref_entropy_term = pref_dist_mb.entropy().sum(dim=-1).mean()
                        pref_loss = pref_loss - args.prefpoe_alpha_entropy * pref_entropy_term
                else:
                    pref_loss = torch.zeros((), device=device)

                if prefpoe_active and args.prefpoe_w_cons > 0.0 and fused_dist_mb is not None:
                    # KL(π_fused ‖ π_pref) — paper eq (6) consistency anchor.
                    cons_loss = kl_divergence(fused_dist_mb, pref_dist_mb).sum(dim=-1).mean()
                else:
                    cons_loss = torch.zeros((), device=device)

                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + v_loss * args.vf_coef
                    + args.prefpoe_w_pref * pref_loss
                    + args.prefpoe_w_cons * cons_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                pref_losses_log.append(pref_loss.item())
                cons_losses_log.append(cons_loss.item())
                with torch.no_grad():
                    main_ent_log.append(main_dist_mb.entropy().sum(dim=-1).mean().item())
                    pref_ent_log.append(pref_dist_mb.entropy().sum(dim=-1).mean().item())
                    if fused_dist_mb is not None:
                        fused_ent_log.append(fused_dist_mb.entropy().sum(dim=-1).mean().item())
                        kl_fused_pref_log.append(
                            kl_divergence(fused_dist_mb, pref_dist_mb).sum(dim=-1).mean().item()
                        )
                        kl_fused_main_log.append(
                            kl_divergence(fused_dist_mb, main_dist_mb).sum(dim=-1).mean().item()
                        )
                    # v2: pref_log_std is a global parameter, not a head.
                    pref_log_std_param = agent.pref_log_std.detach()
                    pref_log_std_mean_log.append(pref_log_std_param.mean().item())
                    pref_log_std_min_log.append(pref_log_std_param.min().item())
                    pref_log_std_max_log.append(pref_log_std_param.max().item())

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
        writer.add_scalar("prefpoe/lambda_pref_now", lambda_pref_now, global_step)
        if pref_losses_log:
            writer.add_scalar("prefpoe/pref_loss", np.mean(pref_losses_log), global_step)
            writer.add_scalar("prefpoe/cons_loss", np.mean(cons_losses_log), global_step)
            writer.add_scalar("prefpoe/main_entropy", np.mean(main_ent_log), global_step)
            writer.add_scalar("prefpoe/pref_entropy", np.mean(pref_ent_log), global_step)
            if fused_ent_log:
                writer.add_scalar("prefpoe/fused_entropy", np.mean(fused_ent_log), global_step)
                writer.add_scalar("prefpoe/kl_fused_pref", np.mean(kl_fused_pref_log), global_step)
                writer.add_scalar("prefpoe/kl_fused_main", np.mean(kl_fused_main_log), global_step)
            writer.add_scalar("prefpoe/pref_log_std_mean", np.mean(pref_log_std_mean_log), global_step)
            writer.add_scalar("prefpoe/pref_log_std_min", np.min(pref_log_std_min_log), global_step)
            writer.add_scalar("prefpoe/pref_log_std_max", np.max(pref_log_std_max_log), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
