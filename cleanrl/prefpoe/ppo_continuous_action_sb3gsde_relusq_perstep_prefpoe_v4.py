# PPO + SB3 gSDE per-step + ReLU²-family MLP + PrefPoE v4 (relusq ablation, fixed)
#
# Background. v3 of this ablation just swapped Tanh -> ReluSq (f(x)=relu(x)²)
# in BOTH the critic and the shared actor_latent and collapsed to ~-400 returns
# by 1M steps on HalfCheetah-v4, while the Tanh counterpart reaches +3670 by 3M.
#
# Diagnosis (why naive ReluSq breaks this specific architecture):
#   The shared actor_latent feeds THREE consumers simultaneously:
#       (i)   the main actor mean   μ_main(s) = actor_mean(actor_latent(s))
#       (ii)  the gSDE variance     σ²_main(s)_i = Σ_j latent_j(s)² · exp(log_std)_ji²
#       (iii) the pref head         μ_pref(s) = pref_mean(actor_latent(s))
#   gSDE's variance formula already squares the latent. ReluSq squares the
#   pre-activation. Composing the two means σ²_main grows like latent⁴ in the
#   live half of the units while the other ~50% of units are dead (ReLU has
#   ~50% dead at init with zero bias; squaring keeps them dead). Effective
#   fan-in is halved, the surviving units carry larger magnitudes, and
#   σ_main(s) becomes wildly heavy-tailed and unstable. Combined with the
#   `layer_init(std=sqrt(2))` Kaiming-for-ReLU gain (mis-calibrated for ReluSq)
#   the forward variance through actor_latent is far outside the regime gSDE
#   was tuned for, and the actor mean / pref head inherit the same pathology.
#
# Fix (chosen after evaluating five options):
#   (1) Replace ReluSq with LeakyReluSq, f(x)=(0.5x + 0.5·relu(x))²
#       = x² for x≥0, 0.25·x² for x<0 (from ppo_continuous_action_lrelusq_v1.py
#       line 119). This keeps the quadratic non-linearity that motivated the
#       ablation but passes negative pre-activations through at quarter
#       magnitude, eliminating the dead-neuron collapse.
#   (2) Bound the actor_latent output magnitude with a final tanh before it
#       feeds gSDE. This protects the σ²_main = Σ latent² · exp(log_std)²
#       formula from heavy-tailed latents without changing gSDE's interface or
#       parameter shapes. The hidden layers stay LeakyReluSq (rich features),
#       only the LAST activation in actor_latent becomes tanh.
#   (3) Keep the critic on the ReluSq family (LeakyReluSq) end-to-end. The
#       critic doesn't feed gSDE, so the heavy-tailed forward path is harmless
#       there — and the quadratic feature is exactly the kind of thing value
#       learning benefits from on MuJoCo.
#   (4) Keep `layer_init(std=sqrt(2))` for now. It's mis-calibrated in theory
#       but the final tanh in actor_latent normalizes the scale that flows into
#       gSDE in practice, so the remaining variance miscalibration only
#       affects the hidden layers which are robust to it.
# This is option (a)+(e) from the requested menu. Rationale for combining:
# (a) alone (LeakyReluSq, no tanh-bound) still feeds an unbounded squared
# quantity into gSDE's already-squared formula — improvement but not principled.
# (e) alone (ReluSq + tanh bound) leaves the dead-neuron collapse in the
# hidden layers untreated. Together they handle both failure modes cleanly.
#
# Audit-derived changes from Tanh v4 (verbatim parity for a fair ablation):
#   #1 PPO ratio under π_fused (not π_main) — reverted v3's split path.
#   #4 Per-BATCH advantage normalization (not per-minibatch).
#   #5 Removed tanh on pref_mean (symmetric with un-tanh'd μ_main).
#   #6 λ_pref = 1.0 (was 0.5).
#   #7 Removed hard clamp on pref_log_std parameter; wide bounds only.
#   #8 α_entropy = 0.05 (was 0.2) — compensates for sum-over-action-dim entropy.
#   (pref_log_std_init = -1.0 unchanged.)
#
# Architecture summary (paper eq 1 + eq 5 + eq 6, applied to gSDE marginal):
#   π_main   = N(μ_main(s), σ_main²(s))
#       actor_latent(s): LeakyReluSq → LeakyReluSq → Tanh   (Tanh on output)
#       μ_main(s)        = actor_mean(actor_latent(s))
#       σ²_main(s)_i     = Σ_j latent_j(s)² · exp(log_std)_ji²        (gSDE)
#   π_pref   = N(μ_pref(s), σ_pref²(s))
#       μ_pref(s)        = pref_mean(actor_latent(s))                 (no tanh)
#       σ_pref           = exp(pref_log_std)·1                        (state-INDEP)
#   π_fused ∝ π_main · π_pref^λ_pref, λ_pref=1.0, σ_fused clamped [1e-3, 2.0]
#   Behavior: sample from fused. Score: log_prob & entropy under fused.
#   Critic: LeakyReluSq end-to-end (no gSDE interaction → no tanh needed).
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

    # gSDE (unchanged from baseline)
    gsde_log_std_init: float = -2.0
    full_std: bool = True
    use_expln: bool = False
    learn_sde_features: bool = False
    sde_sample_freq: int = 1

    # PrefPoE (arXiv 2511.08241). v4 defaults align with the OpenReview reference
    # implementation (verbatim parity with Tanh v4).
    lambda_pref: float = 1.0
    prefpoe_beta1: float = 0.2
    prefpoe_alpha_entropy: float = 0.05
    prefpoe_w_pref: float = 0.05
    prefpoe_w_cons: float = 0.1
    prefpoe_warmup_steps: int = 0
    pref_log_std_init: float = -1.0
    pref_log_std_min: float = -5.0
    pref_log_std_max: float = 2.0

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


class LeakyReluSq(nn.Module):
    """f(x) = (0.5*x + 0.5*relu(x))^2 — equals x^2 for x>=0, 0.25*x^2 for x<0.

    Same quadratic feature as ReluSq on the positive half, but x<0 passes
    through at quarter magnitude (still positive output, slope retained
    through the square). This eliminates the dead-neuron collapse that
    breaks gSDE's σ_main(s) = Σ latent² · exp(log_std)² formula when
    naive ReluSq is used in the shared actor_latent.
    """

    def forward(self, x):
        return (0.5 * x + 0.5 * torch.relu(x)).square()


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
    """Closed-form diagonal Gaussian Product-of-Experts (paper eq 5)."""
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
        pref_log_std_min=-5.0,
        pref_log_std_max=2.0,
        pref_log_std_init=-1.0,
    ):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        latent_dim = 64

        # Critic: LeakyReluSq end-to-end. No gSDE interaction here; the
        # quadratic non-linearity is free to add representational power to V(s).
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            LeakyReluSq(),
            layer_init(nn.Linear(64, 64)),
            LeakyReluSq(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Actor latent: LeakyReluSq hidden layers (rich quadratic features,
        # no dead-neuron collapse), then a Tanh on the OUTPUT to bound the
        # magnitude that feeds gSDE's σ²_main = Σ latent² · exp(log_std)²
        # formula. Without the output tanh, an unbounded squared quantity
        # gets squared AGAIN by gSDE → heavy-tailed σ_main; this is the
        # specific failure mode that collapsed the v3 ablation.
        self.actor_latent = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            LeakyReluSq(),
            layer_init(nn.Linear(64, 64)),
            LeakyReluSq(),
            layer_init(nn.Linear(64, latent_dim)),
            nn.Tanh(),
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

        # Preference head (paper eq 1): state-dependent mean on the shared latent,
        # state-INDEPENDENT log_std. No tanh on pref_mean (audit #5, symmetric
        # with un-tanh'd μ_main). pref_log_std parameter has wide bounds only.
        self.pref_mean = layer_init(nn.Linear(latent_dim, action_dim), std=0.01)
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
        """Paper eq (1) on the shared latent. v4: μ_pref un-tanh'd, pref_log_std
        is a state-INDEP global parameter with wide bounds only."""
        pref_mu = self.pref_mean(latent_sde)
        pref_log_std = self.pref_log_std.clamp(self.pref_log_std_min, self.pref_log_std_max)
        pref_std = pref_log_std.exp().expand_as(pref_mu)
        return Normal(pref_mu, pref_std)

    def get_dists(self, x):
        latent_sde = self.actor_latent(x)
        action_mean = self.actor_mean(latent_sde)
        main_dist = self.action_dist.get_distribution(action_mean, self.log_std, latent_sde)
        pref_dist = self._pref_distribution(latent_sde)
        return main_dist, pref_dist, latent_sde

    def get_action_and_value(self, x, action=None, lambda_pref=0.0):
        """Sample/score actions under π_fused (paper-faithful, reference-aligned).

        v4: sampling, log_prob, and entropy ALL go through π_fused when
        lambda_pref > 0. Safety: σ_fused clamped to [1e-3, 2.0]; λ_pref=1.0.

        Returns: action, logprob_under_FUSED, entropy_under_FUSED, value,
                 main_dist, pref_dist, fused_dist (None if λ==0).
        """
        main_dist, pref_dist, latent_sde = self.get_dists(x)
        if lambda_pref > 0.0:
            fused_dist = fuse_diagonal_gaussians(main_dist, pref_dist, lambda_pref)
            sample_dist = fused_dist
        else:
            fused_dist = None
            sample_dist = main_dist

        if action is None:
            if lambda_pref > 0.0:
                action = sample_dist.rsample().detach()
            else:
                if not self._has_compatible_noise(main_dist.loc.shape[0], main_dist.loc.device):
                    self.reset_noise(batch_size=main_dist.loc.shape[0])
                action = self.action_dist.sample(main_dist.loc, latent_sde).detach()

        logprob = sample_dist.log_prob(action).sum(1)
        entropy = sample_dist.entropy().sum(1)
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

        # v4: per-BATCH advantage normalization (audit #4).
        if args.norm_adv:
            b_advantages_norm = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        else:
            b_advantages_norm = b_advantages

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

                mb_advantages = b_advantages_norm[mb_inds]

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

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

                if prefpoe_active and args.prefpoe_w_pref > 0.0:
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
