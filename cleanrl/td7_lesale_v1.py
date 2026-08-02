# TD7-LeSALE v1 — stable end-to-end latent dynamics for TD7.
#
# TD7's SALE encoder predicts the next normalized state embedding from (state embedding, action),
# but a detached target plus per-sample AvgL1Norm does not control covariance/rank and the predictor
# can mostly copy adjacent MuJoCo states while ignoring action. LeSALE keeps TD7's actor, critic, LAP,
# target clipping, policy checkpointing, and 250-step frozen-encoder boundary, while adding:
#   1. an identity-initialized residual transition zsa = zs + delta(zs, action), with action-dependent
#      AdaLN-style shift/scale/gating for a stronger, staged state-action interaction;
#   2. a separate UNIFORM replay batch for representation learning, avoiding TD-error-biased latent
#      geometry from LAP samples;
#   3. optional frozen-orthogonal subspace SIGReg on current/next embeddings, with TD7 AvgL1 scale
#      corrected by sqrt(2/pi), to prevent collapse without forcing full 256-D Gaussian geometry;
#   4. an optional attached next-embedding target, matching LeWM's end-to-end JEPA gradient flow once
#      subspace anti-collapse regularization is enabled.
#   5. optional JEDI conditional EDM denoising as an additive representation objective; the stock
#      predictor remains TD7's deterministic, action-differentiable critic/exploration interface.
#   6. optional direct JEDI endpoint control, using a fixed Gaussian prior and frozen denoiser
#      snapshots so the critic sees the final three-step Euler latent without control-time variance.
#
# Arms: residual control (--no-use-subsig), subspace SIGReg (defaults), and fully end-to-end
# (--attached-target). MuJoCo observations are Markov, so no temporal transformer is used.
import copy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.nn.utils.parametrizations import orthogonal
from torch.utils.tensorboard import SummaryWriter

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, torch.backends.cudnn.deterministic=False"""
    cuda: bool = True
    """if toggled, CUDA is enabled"""
    track: bool = False
    """track with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """Weights and Biases project name"""
    wandb_entity: str | None = None
    """Weights and Biases entity"""
    capture_video: bool = False
    """capture evaluation videos"""
    save_model: bool = False
    """save an inference artifact at the end"""

    env_id: str = "HalfCheetah-v4"
    """environment id"""
    total_timesteps: int = 8_000_000
    """total environment steps"""
    num_envs: int = 1
    """TD7 requires one environment for 1:1 train/env steps and episode checkpoints"""
    learning_starts: int = 25_000
    """uniform-random collection steps before training"""
    use_checkpoints: bool = True
    """train at episode boundaries and evaluate the robust checkpoint"""
    eval_freq: int = 5_000
    """evaluation interval"""
    eval_eps: int = 10
    """evaluation episodes"""
    buffer_size: int = 1_000_000
    """replay capacity"""
    gpu_replay: bool = False
    """store replay tensors on CUDA to avoid per-update CPU round trips and pageable copies"""
    batch_size: int = 256
    """replay batch size"""
    gamma: float = 0.99
    """discount"""
    target_update_rate: int = 250
    """hard target and frozen-encoder update interval"""
    exploration_noise: float = 0.1
    """fixed behavior action noise used when neither SAC nor SDNoise is enabled"""
    sd_noise: bool = False
    """use learned state-dependent additive Gaussian exploration without SAC value targets"""
    sd_target_sigma: float = 0.1
    """target geometric-mean per-dimension standard deviation for SDNoise"""
    sd_alpha: float = 0.2
    """fixed SDNoise entropy coefficient when automatic tuning is disabled"""
    sd_alpha_lr: float = 1e-3
    """SDNoise automatic-temperature learning rate"""
    sd_alpha_autotune: bool = True
    """tune SDNoise temperature toward action_dim * log(sd_target_sigma)"""
    sd_log_std_min: float = -5.0
    """minimum additive Gaussian log standard deviation"""
    sd_log_std_max: float = 0.0
    """maximum additive Gaussian log standard deviation"""
    target_policy_noise: float = 0.2
    """target policy smoothing noise"""
    noise_clip: float = 0.5
    """target policy noise clipping"""
    policy_freq: int = 2
    """delayed policy update interval"""
    lap_alpha: float = 0.4
    """LAP prioritization exponent"""
    min_priority: float = 1.0
    """minimum LAP priority and Huber threshold"""
    max_eps_when_checkpointing: int = 20
    """episodes per robust checkpoint assessment"""
    steps_before_checkpointing: int = 750_000
    """training steps before full checkpoint assessment"""
    reset_weight: float = 0.9
    """best-min-return discount when full assessment begins"""
    zs_dim: int = 256
    """SALE latent dimension"""
    hidden_dim: int = 256
    """network hidden dimension"""
    encoder_lr: float = 3e-4
    """encoder learning rate"""
    critic_lr: float = 3e-4
    """critic learning rate"""
    actor_lr: float = 3e-4
    """actor learning rate"""
    pc_actor: bool = False
    """replace SDNoise actor backpropagation with prospective predictive coding"""
    pc_actor_batch_size: int = 0
    """PC-only actor minibatch size; zero uses the full replay batch"""
    pc_actor_inference_steps: int = 10
    """reverse Gauss-Seidel sweeps per delayed actor update"""
    pc_actor_inference_scale: float = 1.0
    """block-solve step scale for PC actor state inference"""
    pc_actor_nudge: float = 0.05
    """raw-output terminal nudge used for inverse-nudge hidden-layer learning"""
    pc_actor_normalize_terminal_force: bool = True
    """normalize the raw clipped-Q-plus-entropy force before PC inference"""
    pc_actor_force_rms_min: float = 1e-3
    """minimum divisor used by actor terminal-force RMS normalization"""
    pc_actor_curvature_damping: float = 0.05
    """positive damping added to every shared hidden-state curvature block"""
    pc_actor_adam_beta1: float = 0.9
    """first-moment coefficient of the no-decay local actor optimizer"""
    pc_actor_adam_beta2: float = 0.999
    """second-moment coefficient of the no-decay local actor optimizer"""
    pc_actor_adam_epsilon: float = 1e-8
    """denominator epsilon of the no-decay local actor optimizer"""
    sac_policy: bool = False
    """use a squashed-Gaussian SAC policy and entropy-regularized control losses"""
    sac_alpha: float = 0.2
    """fixed SAC entropy coefficient when automatic tuning is disabled"""
    sac_alpha_lr: float = 1e-3
    """automatic-temperature learning rate, matching baseline SAC's q_lr"""
    sac_autotune: bool = True
    """automatically tune SAC temperature toward minus the action dimension"""
    sac_log_std_min: float = -5.0
    """minimum Gaussian policy log standard deviation, matching baseline SAC"""
    sac_log_std_max: float = 2.0
    """maximum Gaussian policy log standard deviation, matching baseline SAC"""
    sac_compensate_policy_delay: bool = True
    """perform policy_freq actor/temperature updates when the delayed SAC block fires"""
    hl_gauss_critic: bool = False
    """replace scalar twin-Q regression with symlog HL-Gauss categorical critics"""
    hl_gauss_num_bins: int = 511
    """number of categorical bins per Q head"""
    hl_gauss_v_min: float = -9.90353755128617
    """minimum symlog support edge, corresponding to raw -20000"""
    hl_gauss_v_max: float = 9.90353755128617
    """maximum symlog support edge, corresponding to raw 20000"""
    hl_gauss_sigma_ratio: float = 2.0
    """HL-Gauss label standard deviation in bin-width units"""
    torch_compile: bool = False
    """compile the static encoder, critic, and actor loss regions"""
    compile_mode: str = "default"
    """torch.compile mode; default is the validated no-CUDA-graph training path"""
    fused_adam: bool = False
    """use the CUDA fused Adam implementation for all three optimizers"""
    tf32: bool = False
    """allow TF32 matrix multiplications while retaining FP32 parameters and losses"""

    use_subsig: bool = True
    """regularize frozen orthogonal latent subspaces toward Gaussian marginals"""
    use_full_obs_sigreg: bool = False
    """replace subspace sketches with LeWM's full-ambient observation SIGReg"""
    attached_target: bool = False
    """allow prediction-loss gradients through the next-state embedding, as in LeWM"""
    residual_predictor: bool = True
    """use the identity-initialized action-modulated predictor instead of stock SALE"""
    prediction_from_lap: bool = False
    """train the prediction term on the critic's LAP batch while keeping SIGReg uniform"""
    subsig_coef: float = 1e-3
    """subspace SIGReg coefficient, calibrated to ~0.1 of initial prediction gradient"""
    subsig_subspaces: int = 8
    """number of independently frozen row-orthonormal latent subspaces"""
    subsig_dim: int = 32
    """dimension of each frozen latent subspace"""
    subsig_num_proj: int = 32
    """one-dimensional Gaussian sketches per subspace (256 total by default)"""
    subsig_knots: int = 17
    """Epps-Pulley integration knots"""
    sigreg_batch_size: int = 0
    """examples used by each SIGReg estimator; zero uses the full prediction batch"""
    control_sigreg_batch_size: int = -1
    """control observation SIGReg examples; -1 inherits sigreg_batch_size, zero uses full batch"""
    lewm_projected_aux: bool = False
    """train a separate attached-target LeWM projector space without changing TD7 control latents"""
    lewm_private_dynamics: bool = False
    """predict projected LeWM transitions with private action-conditioned dynamics instead of stock zsa"""
    lewm_hidden_dim: int = 512
    """hidden width of the LeWM projector and prediction-projector MLPs"""
    lewm_coef: float = 1.0
    """weight of the complete projected LeWM objective"""
    lewm_warmup_steps: int = 10_000
    """linear warmup for projected LeWM gradients into the shared encoder"""
    lewm_sigreg_coef: float = 0.09
    """full-dimensional SIGReg weight used inside the projected LeWM objective"""
    lewm_sigreg_num_proj: int = 1024
    """fresh full-dimensional Gaussian projections used by LeWM SIGReg"""
    lewm_rollout_aux: bool = False
    """use a split-trunk recurrent LeWM/JEDI model instead of the one-step projector auxiliary"""
    lewm_rollout_horizon: int = 3
    """contiguous recurrent latent-prediction horizon"""
    lewm_rollout_dim: int = 128
    """dimension of the private recurrent world-model coordinate"""
    lewm_aux_trunk_cap: float = 0.35
    """maximum combined auxiliary/shared-trunk gradient norm relative to Stock SALE"""
    encoder_max_grad_norm: float = 1.0
    """encoder gradient clipping, matching LeWM's stabilization"""
    latent_log_freq: int = 500
    """training-step interval for latent geometry diagnostics"""
    reward_token_aux: bool = False
    """predict reward through a separate scalar Gaussianized token"""
    reward_token_coef: float = 1.0
    """weight of the decoded symlog-reward objective"""
    reward_token_sigreg_coef: float = 0.001
    """one-dimensional SIGReg weight for the reward token"""
    reward_token_warmup_steps: int = 10_000
    """linear warmup for reward-token gradients"""
    reward_token_shared_scale: float = 0.1
    """reward-token gradient scale at the SALE latent branch point"""
    reward_control_cost_coef: float = 0.1
    """known HalfCheetah action-cost coefficient removed from the learned reward target"""
    reward_sigreg_tokenizer_only: bool = False
    """route reward-token SIGReg into the tokenizer without modifying SALE latents"""
    policy_mean_aux: bool = False
    """predict the frozen deterministic target actor's action mean from the SALE state latent"""
    policy_mean_coef: float = 1.0
    """weight of deterministic policy-mean prediction"""
    lejepa_outcome_tokens: bool = False
    """jointly predict reward and next-policy LeJEPA tokens with independent SIGReg"""
    outcome_from_transition: bool = False
    """predict outcome tokens from the shared zsa transition instead of parallel (zs, action) context"""
    outcome_token_coef: float = 1.0
    """full weight applied independently to reward- and policy-token objectives"""
    outcome_sigreg_coef: float = 1.0
    """full weight of each outcome token's independent Epps-Pulley SIGReg statistic"""
    outcome_policy_sigreg_num_proj: int = 32
    """fresh full-space projections for the next-policy token's SIGReg"""
    outcome_sigreg_batch_normalized: bool = True
    """divide outcome EP statistics by batch size; disable for exact LeWM scaling"""
    outcome_policy_source: str = "target"
    """policy token source: target snapshots or the current deployed behavior policy"""
    outcome_policy_include_log_std: bool = False
    """append SAC log-std to the deterministic policy-center target token"""
    semantic_outcome_tokens: bool = False
    """use dedicated semantic reward/policy tokens with direct distributional targets"""
    semantic_outcome_token_dim: int = 64
    """width of each dedicated reward and policy semantic token"""
    semantic_reward_num_bins: int = 51
    """number of direct or encoded HL-Gauss reward bins"""
    semantic_reward_raw_min: float = -40.0
    """minimum raw reward represented by the semantic reward token"""
    semantic_reward_raw_max: float = 40.0
    """maximum raw reward represented by the semantic reward token"""
    semantic_reward_sigma_ratio: float = 0.75
    """HL-Gauss label sigma as a fraction of the configured reward-coordinate bin width"""
    semantic_reward_prior_floor: float = 1e-20
    """probability floor for the zero-reward categorical initialization"""
    latent_outcome_tokens: bool = False
    """predict attached reward and policy target embeddings rather than their decoded values"""
    isometric_outcome_tokens: bool = False
    """predict attached outcome targets from learned non-collapsing isometric encoders"""
    policy_beta_nll: bool = False
    """replace the isometric policy-token JEPA loss with direct policy-moment Beta NLL"""
    policy_beta_nll_eps: float = 1e-5
    """keep normalized policy-moment targets inside the open Beta support"""
    policy_beta_nll_coef: float = -1.0
    """policy Beta-NLL weight; negative reuses outcome_token_coef for v7 compatibility"""
    policy_beta_max_precision: float = 0.0
    """maximum Beta precision beyond the unimodal +1 offsets; zero keeps v7 unbounded"""
    reward_hlgauss_ce: bool = False
    """replace isometric reward JEPA with direct HL-Gauss categorical prediction"""
    reward_hlgauss_symlog: bool = True
    """space direct reward HL-Gauss bins uniformly in symlog rather than raw coordinates"""
    dreamer_loss_normalization: bool = False
    """RMS-normalize each main dynamics loss using Dreamer4's lagged squared-loss EMA"""
    loss_normalization_beta: float = 0.95
    """EMA beta for Dreamer4 loss normalization"""
    loss_normalization_eps: float = 1e-6
    """minimum RMS divisor for Dreamer4 loss normalization"""
    adaptive_outcome_grad_equalization: bool = False
    """equalize reward/policy upstream gradient budgets against the representation objective"""
    outcome_grad_equalization_interval: int = 500
    """encoder-update interval for exact shared-gradient recalibration"""
    outcome_semantic_coef: float = 0.5
    """target-token reconstruction weight that anchors latent outcome semantics"""

    jedi_aux: bool = False
    """add conditional EDM denoising while retaining the stock SALE prediction objective"""
    jedi_coef: float = 1.0
    """weight of the JEDI latent denoising objective"""
    jedi_coef_warmup_steps: int = 10_000
    """linear warmup for JEDI's shared-representation weight"""
    jedi_lr: float = 1e-3
    """denoiser learning rate; the default keeps encoder LR at the paper's 0.3 ratio"""
    jedi_weight_decay: float = 1e-2
    """AdamW weight decay for the denoiser only"""
    jedi_warmup_steps: int = 1_000
    """linear denoiser learning-rate warmup"""
    jedi_max_grad_norm: float = 1.0
    """denoiser gradient clipping threshold"""
    jedi_blocks: int = 2
    """number of vector AdaLN residual blocks in the denoiser"""
    jedi_time_dim: int = 64
    """sinusoidal diffusion-time embedding dimension"""
    jedi_p_mean: float = -0.4
    """mean of log sigma inherited from DIAMOND"""
    jedi_p_std: float = 1.2
    """standard deviation of log sigma inherited from DIAMOND"""
    jedi_sigma_data: float = 1.0
    """EDM data scale used by JEDI"""
    jedi_clamp: float = 3.0
    """soft latent bound C(z)=s*tanh(z/s)"""
    jedi_endpoint_control: bool = False
    """replace critic-consumed stock zsa with the final deterministic JEDI endpoint"""
    jedi_canonical_control_latents: bool = False
    """use JEDI's bounded latent coordinate consistently throughout actor/critic control"""
    jedi_exact_actor_gradients: bool = False
    """differentiate the actor through the frozen three-step endpoint instead of a SALE STE"""
    jedi_endpoint_mix_steps: int = 10_000
    """snapshot-aligned updates over which stock zsa hands off to the direct endpoint"""
    jedi_endpoint_diag_priors: int = 8
    """fixed alternative priors used only to diagnose conditional sampler variance"""

    controllability_exploration: bool = False
    """replace some Gaussian behavior noise with persistent Q-shielded latent-direction guidance"""
    ctrl_basis_batch: int = 512
    """replay states used to estimate the local controllability covariance"""
    ctrl_modes: int = 16
    """top controllability eigenvectors retained for persistent exploration directions"""
    ctrl_perturb_std: float = 0.2
    """per-action perturbation scale used to estimate the controllability basis and Q frontier"""
    ctrl_hold_steps: int = 50
    """behavior steps for which one latent direction persists"""
    ctrl_start_training_steps: int = 50_000
    """training updates before guided exploration can begin (about 75k environment steps)"""
    ctrl_ramp_steps: int = 75_000
    """updates over which guided-step probability ramps to its maximum"""
    ctrl_max_probability: float = 0.5
    """maximum fraction of behavior steps using latent-direction guidance"""
    ctrl_q_slack: float = 0.25
    """allowed pessimistic-Q drop in local Q-frontier units"""
    ctrl_q_floor: float = 0.05
    """minimum local Q-frontier scale"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def avg_l1_norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def symlog(x):
    return x.sign() * torch.log1p(x.abs())


def symexp(x):
    return x.sign() * torch.expm1(x.abs())


JEDI_LATENT_SCALE = float(np.sqrt(2.0 / np.pi))


def jedi_control_latent(x, clamp=3.0):
    """Map raw AvgL1 SALE latents into JEDI's canonical bounded control coordinate."""
    return clamp * torch.tanh(JEDI_LATENT_SCALE * x / clamp) / JEDI_LATENT_SCALE


def lap_huber(td_loss, min_priority=1.0):
    return torch.where(td_loss < min_priority, 0.5 * td_loss.square(), min_priority * td_loss).sum(1).mean()


def sdnoise_alpha_loss(log_alpha, entropy_proxy, target_entropy):
    """Temperature loss whose descent raises alpha when Gaussian scale is below target."""
    return (
        -log_alpha.exp() * (target_entropy - entropy_proxy.detach())
    ).mean()


class LAPBuffer:
    def __init__(
        self, state_dim, action_dim, device, max_size, batch_size, max_action, gpu_storage=False
    ):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.device = device
        self.batch_size = batch_size
        self.gpu_storage = gpu_storage
        self.episode_boundary_cpu = np.zeros((self.max_size, 1), dtype=np.bool_)
        if gpu_storage:
            self.state = torch.zeros((self.max_size, state_dim), device=device)
            self.action = torch.zeros((self.max_size, action_dim), device=device)
            self.next_state = torch.zeros((self.max_size, state_dim), device=device)
            self.reward = torch.zeros((self.max_size, 1), device=device)
            self.not_done = torch.zeros((self.max_size, 1), device=device)
            self.successor_policy_valid = torch.zeros((self.max_size, 1), device=device)
            self.episode_boundary = torch.zeros(
                (self.max_size, 1), dtype=torch.bool, device=device
            )
        else:
            self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
            self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
            self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
            self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
            self.not_done = np.zeros((self.max_size, 1), dtype=np.float32)
            self.successor_policy_valid = np.zeros((self.max_size, 1), dtype=np.float32)
            self.episode_boundary = np.zeros((self.max_size, 1), dtype=np.bool_)
        self.priority = torch.zeros(self.max_size, device=device)
        self.max_priority = torch.ones((), device=device)
        self.normalize_actions = float(max_action)

    def add(
        self,
        state,
        action,
        next_state,
        reward,
        done,
        episode_boundary=False,
        successor_policy_valid=None,
    ):
        if successor_policy_valid is None:
            successor_policy_valid = 1.0 - float(done)
        if self.gpu_storage:
            self.state[self.ptr].copy_(torch.as_tensor(state, dtype=torch.float32, device=self.device))
            self.action[self.ptr].copy_(
                torch.as_tensor(action, dtype=torch.float32, device=self.device) / self.normalize_actions
            )
            self.next_state[self.ptr].copy_(
                torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
            )
            self.reward[self.ptr] = float(reward)
            self.not_done[self.ptr] = 1.0 - float(done)
            self.successor_policy_valid[self.ptr] = float(successor_policy_valid)
            self.episode_boundary[self.ptr] = bool(episode_boundary)
        else:
            self.state[self.ptr] = state
            self.action[self.ptr] = action / self.normalize_actions
            self.next_state[self.ptr] = next_state
            self.reward[self.ptr] = reward
            self.not_done[self.ptr] = 1.0 - done
            self.successor_policy_valid[self.ptr] = successor_policy_valid
            self.episode_boundary[self.ptr] = episode_boundary
        self.priority[self.ptr] = self.max_priority
        self.episode_boundary_cpu[self.ptr] = episode_boundary
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        csum = torch.cumsum(self.priority[: self.size], dim=0)
        values = torch.rand(self.batch_size, device=self.device) * csum[-1]
        indices = torch.searchsorted(csum, values)
        if self.gpu_storage:
            self.ind = indices
            return (
                self.state[indices],
                self.action[indices],
                self.next_state[indices],
                self.reward[indices],
                self.not_done[indices],
            )
        self.ind = indices.cpu().numpy()
        return (
            torch.as_tensor(self.state[self.ind], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.action[self.ind], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.next_state[self.ind], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.reward[self.ind], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.not_done[self.ind], dtype=torch.float32, device=self.device),
        )

    def update_priority(self, priority):
        self.priority[self.ind] = priority.reshape(-1).detach()
        self.max_priority.copy_(torch.maximum(self.max_priority, priority.max()))

    def reset_max_priority(self):
        self.max_priority.copy_(self.priority[: self.size].max())


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)

    def forward(self, state, zs):
        action = avg_l1_norm(self.l0(state))
        action = torch.cat([action, zs], dim=1)
        action = F.relu(self.l1(action))
        action = F.relu(self.l2(action))
        return torch.tanh(self.l3(action))


class SDNoiseActor(Actor):
    """TD7 actor with learned additive Gaussian behavior noise.

    The deterministic mean and target-policy path are identical to ``Actor``. The extra
    log-standard-deviation head is initialized in a forked RNG stream so enabling SDNoise does not
    perturb the StockSIG critic/encoder initialization or the actor mean's initial parameters.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        zs_dim=256,
        hdim=256,
        log_std_min=-5.0,
        log_std_max=0.0,
        head_seed=1,
    ):
        super().__init__(state_dim, action_dim, zs_dim, hdim)
        with torch.random.fork_rng():
            torch.manual_seed(head_seed)
            self.log_std_head = nn.Linear(hdim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def _features(self, state, zs):
        features = avg_l1_norm(self.l0(state))
        features = torch.cat([features, zs], dim=1)
        features = F.relu(self.l1(features))
        return F.relu(self.l2(features))

    def forward(self, state, zs):
        return torch.tanh(self.l3(self._features(state, zs)))

    def policy_stats(self, state, zs):
        features = self._features(state, zs)
        mean = torch.tanh(self.l3(features))
        log_std = torch.tanh(self.log_std_head(features))
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1.0)
        return mean, log_std

    def sample_additive(self, state, zs, epsilon):
        mean, log_std = self.policy_stats(state, zs)
        action = (mean + log_std.exp() * epsilon).clamp(-1, 1)
        entropy_proxy = log_std.sum(dim=1, keepdim=True)
        return action, entropy_proxy


def avg_l1_norm_vjp(preactivation, cotangent, eps=1e-8):
    """Exact VJP of ``avg_l1_norm`` for a local output cotangent."""
    raw_scale = preactivation.abs().mean(dim=-1, keepdim=True)
    scale = raw_scale.clamp_min(eps)
    direct = cotangent / scale
    scale_term = (
        preactivation.sign()
        * (cotangent * preactivation).sum(dim=-1, keepdim=True)
        / (preactivation.shape[-1] * scale.square())
    )
    return direct - (raw_scale > eps).to(preactivation.dtype) * scale_term


def td7_pc_actor_free_phase(actor, state, zs):
    """Exact SDNoiseActor forward expressed in its three Gaussian PC states."""
    l0_preactivation = F.linear(state, actor.l0.weight, actor.l0.bias)
    z0 = avg_l1_norm(l0_preactivation)
    z1 = F.linear(
        torch.cat([z0, zs], dim=1), actor.l1.weight, actor.l1.bias
    )
    z2 = F.linear(F.relu(z1), actor.l2.weight, actor.l2.bias)
    features = F.relu(z2)
    raw_output = torch.cat(
        [
            F.linear(features, actor.l3.weight, actor.l3.bias),
            F.linear(features, actor.log_std_head.weight, actor.log_std_head.bias),
        ],
        dim=1,
    )
    return l0_preactivation, (z0, z1, z2), raw_output


def td7_pc_actor_policy_from_raw(actor, raw_output, epsilon):
    """Apply the existing bounded mean/log-std parameterization to a raw output leaf."""
    mean_raw, log_std_raw = raw_output.chunk(2, dim=1)
    mean = torch.tanh(mean_raw)
    bounded_log_std = torch.tanh(log_std_raw)
    log_std = actor.log_std_min + 0.5 * (
        actor.log_std_max - actor.log_std_min
    ) * (bounded_log_std + 1.0)
    action = (mean + log_std.exp() * epsilon).clamp(-1, 1)
    return action, log_std


def td7_pc_actor_batch(state, fixed_zs, policy_noise, batch_size):
    """Select the leading PC-only minibatch without changing full-batch RNG draws."""
    if batch_size < 0:
        raise ValueError("pc_actor_batch_size must be nonnegative")
    if batch_size == 0:
        return state, fixed_zs, policy_noise
    size = min(batch_size, state.shape[0])
    return state[:size], fixed_zs[:size], policy_noise[:size]


def td7_pc_actor_curvature_factors(actor, free_states, damping):
    """Shared free-state Gauss-Newton blocks for the exact TD7 actor graph."""
    if damping < 0:
        raise ValueError("pc_actor_curvature_damping must be nonnegative")
    z0, z1, z2 = free_states
    hidden = z0.shape[1]
    eye = torch.eye(hidden, device=z0.device, dtype=z0.dtype)
    local = (1.0 + damping) * eye

    z0_weight = actor.l1.weight[:, :hidden]
    block0 = local + z0_weight.T @ z0_weight

    derivative1 = (z1 > 0).to(z1.dtype)
    derivative1_gram = derivative1.T @ derivative1 / derivative1.shape[0]
    block1 = local + (actor.l2.weight.T @ actor.l2.weight) * derivative1_gram

    output_weight = torch.cat([actor.l3.weight, actor.log_std_head.weight], dim=0)
    derivative2 = (z2 > 0).to(z2.dtype)
    derivative2_gram = derivative2.T @ derivative2 / derivative2.shape[0]
    block2 = local + (output_weight.T @ output_weight) * derivative2_gram
    factors, _ = torch.linalg.cholesky_ex(
        torch.stack([block0, block1, block2]), check_errors=False
    )
    return tuple(factors.unbind(0))


def make_td7_pc_actor_settle_core(args):
    num_steps = args.pc_actor_inference_steps
    scale = args.pc_actor_inference_scale

    def core(
        state,
        zs,
        initial_states,
        weights,
        biases,
        factors,
        target,
    ):
        z0, z1, z2 = initial_states
        l0_weight, l1_weight, l2_weight, mean_weight, std_weight = weights
        l0_bias, l1_bias, l2_bias, mean_bias, std_bias = biases
        hidden = z0.shape[1]
        output_weight = torch.cat([mean_weight, std_weight], dim=0)
        output_bias = torch.cat([mean_bias, std_bias], dim=0)
        for _ in range(num_steps):
            mean2 = F.linear(F.relu(z1), l2_weight, l2_bias)
            output_error = target - F.linear(F.relu(z2), output_weight, output_bias)
            gradient2 = z2 - mean2 - (z2 > 0).to(z2.dtype) * F.linear(
                output_error, output_weight.T
            )
            z2 = z2 - scale * torch.cholesky_solve(gradient2.T, factors[2]).T

            mean1 = F.linear(torch.cat([z0, zs], dim=1), l1_weight, l1_bias)
            downstream2 = z2 - F.linear(F.relu(z1), l2_weight, l2_bias)
            gradient1 = z1 - mean1 - (z1 > 0).to(z1.dtype) * F.linear(
                downstream2, l2_weight.T
            )
            z1 = z1 - scale * torch.cholesky_solve(gradient1.T, factors[1]).T

            mean0 = avg_l1_norm(F.linear(state, l0_weight, l0_bias))
            downstream1 = z1 - F.linear(
                torch.cat([z0, zs], dim=1), l1_weight, l1_bias
            )
            gradient0 = z0 - mean0 - F.linear(
                downstream1, l1_weight[:, :hidden].T
            )
            z0 = z0 - scale * torch.cholesky_solve(gradient0.T, factors[0]).T
        return z0, z1, z2

    compile_kwargs = {"dynamic": False, "fullgraph": True}
    if args.compile_mode != "default":
        compile_kwargs["mode"] = args.compile_mode
    return torch.compile(core, **compile_kwargs)


class AugmentedLocalAdam:
    """No-decay Adam ascent over local augmented [weight | bias] matrices."""

    def __init__(self, layers, beta1, beta2, epsilon):
        self.layers = list(layers)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.first = [
            torch.zeros(
                layer.weight.shape[0],
                layer.weight.shape[1] + 1,
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
            for layer in self.layers
        ]
        self.second = [torch.zeros_like(moment) for moment in self.first]
        self.step_count = 0

    @torch.no_grad()
    def step(self, directions, learning_rate):
        if len(directions) != len(self.layers):
            raise ValueError("direction and local optimizer structures differ")
        self.step_count += 1
        correction1 = 1.0 - self.beta1**self.step_count
        correction2 = 1.0 - self.beta2**self.step_count
        update_rms = []
        for layer, direction, first, second in zip(
            self.layers, directions, self.first, self.second
        ):
            first.mul_(self.beta1).add_(direction, alpha=1.0 - self.beta1)
            second.mul_(self.beta2).addcmul_(
                direction, direction, value=1.0 - self.beta2
            )
            normalized = (first / correction1) / (
                (second / correction2).sqrt() + self.epsilon
            )
            layer.weight.add_(normalized[:, :-1], alpha=learning_rate)
            layer.bias.add_(normalized[:, -1], alpha=learning_rate)
            update_rms.append((learning_rate * normalized).square().mean().sqrt())
        return update_rms


class TD7PCActorTrainer:
    """PC hidden learning with exact free-endpoint SDNoise output-head learning."""

    def __init__(self, actor, args):
        self.actor = actor
        self.args = args
        self.layers = [
            actor.l0,
            actor.l1,
            actor.l2,
            actor.l3,
            actor.log_std_head,
        ]
        self.optimizer = AugmentedLocalAdam(
            self.layers,
            args.pc_actor_adam_beta1,
            args.pc_actor_adam_beta2,
            args.pc_actor_adam_epsilon,
        )
        self.compiled_core = (
            make_td7_pc_actor_settle_core(args) if args.torch_compile else None
        )

    def _weights(self):
        return tuple(layer.weight for layer in self.layers)

    def _biases(self):
        return tuple(layer.bias for layer in self.layers)

    @torch.no_grad()
    def settle_and_directions(self, state, zs, terminal_force):
        preactivation0, free_states, free_output = td7_pc_actor_free_phase(
            self.actor, state, zs
        )
        target = free_output + self.args.pc_actor_nudge * terminal_force
        factors = td7_pc_actor_curvature_factors(
            self.actor,
            free_states,
            self.args.pc_actor_curvature_damping,
        )
        if self.compiled_core is None:
            states = make_td7_pc_actor_settle_core_eager(
                self.actor,
                state,
                zs,
                free_states,
                factors,
                target,
                self.args,
            )
        else:
            states = self.compiled_core(
                state,
                zs,
                free_states,
                self._weights(),
                self._biases(),
                factors,
                target,
            )
        z0, z1, z2 = states
        free_output_features = F.relu(free_states[2])
        residual0 = z0 - avg_l1_norm(
            F.linear(state, self.actor.l0.weight, self.actor.l0.bias)
        )
        features1 = torch.cat([z0, zs], dim=1)
        residual1 = z1 - F.linear(
            features1, self.actor.l1.weight, self.actor.l1.bias
        )
        features2 = F.relu(z1)
        residual2 = z2 - F.linear(
            features2, self.actor.l2.weight, self.actor.l2.bias
        )
        output_features = F.relu(z2)
        mean_error, std_error = (
            target
            - torch.cat(
                [
                    F.linear(output_features, self.actor.l3.weight, self.actor.l3.bias),
                    F.linear(
                        output_features,
                        self.actor.log_std_head.weight,
                        self.actor.log_std_head.bias,
                    ),
                ],
                dim=1,
            )
        ).chunk(2, dim=1)
        inverse_nudge = 1.0 / self.args.pc_actor_nudge
        l0_error = avg_l1_norm_vjp(preactivation0, residual0)
        mean_force, std_force = terminal_force.chunk(2, dim=1)
        directions = [
            augmented_outer_mean(l0_error, state) * inverse_nudge,
            augmented_outer_mean(residual1, features1) * inverse_nudge,
            augmented_outer_mean(residual2, features2) * inverse_nudge,
            augmented_outer_mean(mean_force, free_output_features),
            augmented_outer_mean(std_force, free_output_features),
        ]
        free_stack = torch.cat(free_states, dim=1)
        settled_stack = torch.cat(states, dim=1)
        settled_energy = 0.5 * (
            residual0.square().sum(dim=1)
            + residual1.square().sum(dim=1)
            + residual2.square().sum(dim=1)
            + mean_error.square().sum(dim=1)
            + std_error.square().sum(dim=1)
        ).mean()
        free_energy = 0.5 * (
            self.args.pc_actor_nudge * terminal_force
        ).square().sum(dim=1).mean()
        diagnostics = {
            "response_rms": (
                (settled_stack - free_stack).square().mean().sqrt()
                / self.args.pc_actor_nudge
            ),
            "residual_rms": torch.cat(
                [residual0, residual1, residual2], dim=1
            ).square().mean().sqrt(),
            "terminal_force_rms": terminal_force.square().mean().sqrt(),
            "settled_energy": settled_energy,
            "free_energy": free_energy,
            "energy_ratio": settled_energy / free_energy.clamp_min(1e-12),
            "hidden_direction_norm": torch.stack(
                [direction.square().sum() for direction in directions[:3]]
            ).sum().sqrt(),
            "exact_head_direction_norm": torch.stack(
                [direction.square().sum() for direction in directions[3:]]
            ).sum().sqrt(),
            "direction_norm": torch.stack(
                [direction.square().sum() for direction in directions]
            ).sum().sqrt(),
        }
        return directions, diagnostics

    @torch.no_grad()
    def step(self, state, zs, terminal_force, learning_rate):
        directions, diagnostics = self.settle_and_directions(
            state, zs, terminal_force
        )
        update_rms = self.optimizer.step(directions, learning_rate)
        diagnostics["update_rms"] = torch.stack(update_rms).mean()
        return diagnostics


def augmented_outer_mean(error, features):
    augmented = torch.cat(
        [features, torch.ones_like(features[:, :1])], dim=1
    )
    return torch.einsum("bo,bi->oi", error, augmented) / error.shape[0]


@torch.no_grad()
def make_td7_pc_actor_settle_core_eager(
    actor, state, zs, initial_states, factors, target, args
):
    z0, z1, z2 = (value.clone() for value in initial_states)
    hidden = z0.shape[1]
    output_weight = torch.cat([actor.l3.weight, actor.log_std_head.weight], dim=0)
    output_bias = torch.cat([actor.l3.bias, actor.log_std_head.bias], dim=0)
    for _ in range(args.pc_actor_inference_steps):
        mean2 = F.linear(F.relu(z1), actor.l2.weight, actor.l2.bias)
        output_error = target - F.linear(F.relu(z2), output_weight, output_bias)
        gradient2 = z2 - mean2 - (z2 > 0).to(z2.dtype) * F.linear(
            output_error, output_weight.T
        )
        z2 -= args.pc_actor_inference_scale * torch.cholesky_solve(
            gradient2.T, factors[2]
        ).T

        mean1 = F.linear(
            torch.cat([z0, zs], dim=1), actor.l1.weight, actor.l1.bias
        )
        downstream2 = z2 - F.linear(F.relu(z1), actor.l2.weight, actor.l2.bias)
        gradient1 = z1 - mean1 - (z1 > 0).to(z1.dtype) * F.linear(
            downstream2, actor.l2.weight.T
        )
        z1 -= args.pc_actor_inference_scale * torch.cholesky_solve(
            gradient1.T, factors[1]
        ).T

        mean0 = avg_l1_norm(F.linear(state, actor.l0.weight, actor.l0.bias))
        downstream1 = z1 - F.linear(
            torch.cat([z0, zs], dim=1), actor.l1.weight, actor.l1.bias
        )
        gradient0 = z0 - mean0 - F.linear(
            downstream1, actor.l1.weight[:, :hidden].T
        )
        z0 -= args.pc_actor_inference_scale * torch.cholesky_solve(
            gradient0.T, factors[0]
        ).T
    return z0, z1, z2


class SACActor(nn.Module):
    """TD7-conditioned squashed Gaussian actor matching CleanRL's baseline SAC."""

    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.mean_head = nn.Linear(hdim, action_dim)
        self.log_std_head = nn.Linear(hdim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def policy_stats(self, state, zs):
        features = avg_l1_norm(self.l0(state))
        features = torch.cat([features, zs], dim=1)
        features = F.relu(self.l1(features))
        features = F.relu(self.l2(features))
        mean = self.mean_head(features)
        log_std = torch.tanh(self.log_std_head(features))
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1.0)
        return mean, log_std

    def deterministic(self, state, zs):
        mean, _ = self.policy_stats(state, zs)
        return torch.tanh(mean)

    def forward(self, state, zs):
        # Preserve TD7 call-site semantics: forward is the noiseless bounded policy center.
        return self.deterministic(state, zs)

    def sample(self, state, zs, epsilon):
        mean, log_std = self.policy_stats(state, zs)
        pre_tanh = mean + log_std.exp() * epsilon
        action = torch.tanh(pre_tanh)
        # Normal log density plus the tanh change-of-variables correction. TD7 stores
        # normalized actions, so no environment action-scale term belongs here.
        log_prob = -0.5 * (
            epsilon.square() + 2.0 * log_std + np.log(2.0 * np.pi)
        )
        log_prob = log_prob - torch.log(1.0 - action.square() + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True), torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.q01 = nn.Linear(state_dim + action_dim, hdim)
        self.q1 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)
        self.q02 = nn.Linear(state_dim + action_dim, hdim)
        self.q4 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, 1)

    def forward(self, state, action, zsa, zs):
        state_action = torch.cat([state, action], dim=1)
        embeddings = torch.cat([zsa, zs], dim=1)
        q1 = avg_l1_norm(self.q01(state_action))
        q1 = torch.cat([q1, embeddings], dim=1)
        q1 = F.elu(self.q1(q1))
        q1 = F.elu(self.q2(q1))
        q1 = self.q3(q1)
        q2 = avg_l1_norm(self.q02(state_action))
        q2 = torch.cat([q2, embeddings], dim=1)
        q2 = F.elu(self.q4(q2))
        q2 = F.elu(self.q5(q2))
        q2 = self.q6(q2)
        return torch.cat([q1, q2], dim=1)


class HLGaussCritic(nn.Module):
    """Twin categorical Q critic trained from scalar symlog HL-Gauss targets."""

    def __init__(
        self,
        state_dim,
        action_dim,
        zs_dim=256,
        hdim=256,
        num_bins=511,
        v_min=-9.90353755128617,
        v_max=9.90353755128617,
        sigma_ratio=2.0,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.q01 = nn.Linear(state_dim + action_dim, hdim)
        self.q1 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, num_bins)
        self.q02 = nn.Linear(state_dim + action_dim, hdim)
        self.q4 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, num_bins)

        edges = torch.linspace(v_min, v_max, num_bins + 1)
        support = 0.5 * (edges[:-1] + edges[1:])
        self.register_buffer("edges", edges)
        self.register_buffer("support", support)
        self.register_buffer("scalar_support", symexp(support))
        self.register_buffer(
            "sigma", torch.tensor(sigma_ratio * (v_max - v_min) / num_bins)
        )

        # A categorical uniform prior over this wide support has enormous variance and lets
        # tiny edge probabilities dominate action gradients. Start from the exact finite
        # HL-Gauss projection of zero, as in v213, before constructing the target copy.
        with torch.no_grad():
            zero_log_probs = self.project(
                torch.zeros(1, 1, device=self.edges.device)
            ).clamp_min(1e-20).log()[0]
            for head in (self.q3, self.q6):
                head.weight.zero_()
                head.bias.copy_(zero_log_probs)

    def logits(self, state, action, zsa, zs):
        state_action = torch.cat([state, action], dim=1)
        embeddings = torch.cat([zsa, zs], dim=1)
        q1 = avg_l1_norm(self.q01(state_action))
        q1 = torch.cat([q1, embeddings], dim=1)
        q1 = F.elu(self.q1(q1))
        q1 = F.elu(self.q2(q1))
        q1 = self.q3(q1)
        q2 = avg_l1_norm(self.q02(state_action))
        q2 = torch.cat([q2, embeddings], dim=1)
        q2 = F.elu(self.q4(q2))
        q2 = F.elu(self.q5(q2))
        q2 = self.q6(q2)
        return torch.stack([q1, q2], dim=1)

    def decode_logits(self, logits):
        return (torch.softmax(logits, dim=-1) * self.scalar_support).sum(dim=-1)

    def forward(self, state, action, zsa, zs):
        return self.decode_logits(self.logits(state, action, zsa, zs))

    def project(self, targets):
        target_coord = symlog(targets.squeeze(-1)).clamp(
            self.edges[0], self.edges[-1]
        )
        cdf = torch.erf(
            (self.edges.unsqueeze(0) - target_coord.unsqueeze(-1))
            / (self.sigma * np.sqrt(2.0))
        )
        normalizer = cdf[..., -1:] - cdf[..., :1]
        probabilities = cdf[..., 1:] - cdf[..., :-1]
        return probabilities / normalizer.clamp_min(1e-10)

    def edge_mass(self, logits):
        probabilities = torch.softmax(logits, dim=-1)
        return (probabilities[..., 0] + probabilities[..., -1]).mean()


class UniformLAPBuffer(LAPBuffer):
    """TD7 LAP replay plus an independent uniform sampler for auxiliary representation learning."""

    def sample_uniform(self, batch_size=None, rng=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        if rng is None:
            ind = np.random.randint(0, self.size, size=batch_size)
        else:
            ind = rng.integers(0, self.size, size=batch_size)
        if self.gpu_storage:
            ind = torch.as_tensor(ind, dtype=torch.long, device=self.device)
            return self.state[ind], self.action[ind], self.next_state[ind]
        return (
            torch.as_tensor(self.state[ind], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.action[ind], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.next_state[ind], dtype=torch.float32, device=self.device),
        )

    def sample_uniform_with_reward(self, batch_size=None, rng=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        if rng is None:
            ind = np.random.randint(0, self.size, size=batch_size)
        else:
            ind = rng.integers(0, self.size, size=batch_size)
        if self.gpu_storage:
            ind = torch.as_tensor(ind, dtype=torch.long, device=self.device)
            return (
                self.state[ind],
                self.action[ind],
                self.next_state[ind],
                self.reward[ind],
                self.successor_policy_valid[ind],
            )
        return (
            torch.as_tensor(self.state[ind], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.action[ind], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.next_state[ind], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.reward[ind], dtype=torch.float32, device=self.device),
            torch.as_tensor(
                self.successor_policy_valid[ind], dtype=torch.float32, device=self.device
            ),
        )

    def sample_sequences(self, horizon, batch_size=None, rng=None):
        """Uniform contiguous transitions that never cross a reset or the replay write pointer."""
        batch_size = self.batch_size if batch_size is None else batch_size
        rng = np.random.default_rng() if rng is None else rng
        if self.size <= horizon:
            raise RuntimeError("not enough replay entries for sequence sampling")
        base = self.ptr if self.size == self.max_size else 0
        accepted = []
        offsets = np.arange(horizon, dtype=np.int64)
        while sum(chunk.size for chunk in accepted) < batch_size:
            needed = batch_size - sum(chunk.size for chunk in accepted)
            logical = rng.integers(0, self.size - horizon, size=max(2 * needed, 32))
            physical = (base + logical[:, None] + offsets[None, :]) % self.max_size
            if horizon > 1:
                valid = ~self.episode_boundary_cpu[physical[:, :-1], 0].any(axis=1)
                logical = logical[valid]
            if logical.size:
                accepted.append(logical[:needed])
        logical = np.concatenate(accepted)[:batch_size]
        physical = (base + logical[:, None] + offsets[None, :]) % self.max_size
        if self.gpu_storage:
            indices = torch.as_tensor(physical, dtype=torch.long, device=self.device)
            states = torch.cat(
                [self.state[indices[:, :1]], self.next_state[indices]], dim=1
            )
            return states, self.action[indices], self.reward[indices]
        states = np.concatenate(
            [self.state[physical[:, :1]], self.next_state[physical]], axis=1
        )
        return (
            torch.as_tensor(states, dtype=torch.float32, device=self.device),
            torch.as_tensor(self.action[physical], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.reward[physical], dtype=torch.float32, device=self.device),
        )


class ResidualEncoder(nn.Module):
    """SALE state encoder with an identity-initialized, action-modulated residual transition."""

    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)

        self.transition_in = nn.Linear(zs_dim, hdim)
        self.transition_norm = nn.LayerNorm(hdim, elementwise_affine=False)
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, hdim),
            nn.SiLU(),
            nn.Linear(hdim, hdim),
            nn.SiLU(),
        )
        self.action_modulation = nn.Linear(hdim, 3 * hdim)
        self.transition_hidden = nn.Linear(hdim, hdim)
        self.transition_out = nn.Linear(hdim, zs_dim, bias=False)

        # At initialization gate=0, hence zsa==zs. As in AdaLN-zero, the gate learns first; the
        # state/action transformation comes online progressively instead of injecting random drift.
        nn.init.zeros_(self.action_modulation.weight)
        nn.init.zeros_(self.action_modulation.bias)

    def zs(self, state):
        zs = F.elu(self.zs1(state))
        zs = F.elu(self.zs2(zs))
        return avg_l1_norm(self.zs3(zs))

    def zsa(self, zs, action):
        h = self.transition_norm(self.transition_in(zs))
        shift, scale, gate = self.action_modulation(self.action_embed(action)).chunk(3, dim=-1)
        h = h * (1.0 + scale) + shift
        h = F.elu(self.transition_hidden(F.elu(h)))
        delta = self.transition_out(torch.tanh(gate) * h)
        return zs + delta


class StockEncoder(nn.Module):
    """Original TD7 SALE encoder, retained for an isolated SubSIG/uniform-replay arm."""

    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

    def state_features(self, state):
        return F.elu(self.zs2(F.elu(self.zs1(state))))

    def zs_from_features(self, features):
        return avg_l1_norm(self.zs3(features))

    def zs(self, state):
        return self.zs_from_features(self.state_features(state))

    def zsa(self, zs, action):
        zsa = F.elu(self.zsa1(torch.cat([zs, action], dim=1)))
        zsa = F.elu(self.zsa2(zsa))
        return self.zsa3(zsa)


class SubspaceSIGReg(nn.Module):
    """Epps-Pulley Gaussian matching in fixed row-orthonormal latent subspaces."""

    def __init__(self, dim, num_subspaces, subspace_dim, num_proj, knots, device, seed):
        super().__init__()
        if subspace_dim > dim:
            raise ValueError(f"subsig_dim ({subspace_dim}) must be <= zs_dim ({dim})")
        self.num_proj = num_proj
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

        bases = []
        for _ in range(num_subspaces):
            matrix = torch.randn(dim, subspace_dim, device=device, generator=self.generator)
            q, _ = torch.linalg.qr(matrix, mode="reduced")
            bases.append(q.T)
        self.register_buffer("bases", torch.stack(bases))  # (K, ds, D)

        t = torch.linspace(0, 3, knots, dtype=torch.float32, device=device)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32, device=device)
        weights[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)
        self.register_buffer(
            "ambient_directions",
            torch.empty(num_subspaces, dim, num_proj, dtype=torch.float32, device=device),
        )

    @torch.no_grad()
    def resample_directions(self):
        """Refresh stochastic projections outside compiled loss regions."""
        directions = torch.randn(
            self.bases.size(0),
            self.bases.size(1),
            self.num_proj,
            device=self.bases.device,
            generator=self.generator,
        )
        directions.div_(directions.norm(dim=1, keepdim=True).clamp_min(1e-8))
        self.ambient_directions.copy_(torch.einsum("ksd,ksm->kdm", self.bases, directions))

    def forward(self, z):
        """z: (T, B, D); Gaussian matching is independent at each temporal position."""
        projected = torch.einsum("tbd,kdm->tkbm", z, self.ambient_directions)
        x_t = projected.unsqueeze(-1) * self.t
        err = (x_t.cos().mean(dim=2) - self.phi).square() + x_t.sin().mean(dim=2).square()
        statistic = (err @ self.weights) * z.size(1)
        return statistic.mean()


class FullSIGReg(nn.Module):
    """LeWM's full-ambient Epps-Pulley regularizer with RNG outside compiled graphs."""

    def __init__(self, dim, num_proj, knots, device, seed):
        super().__init__()
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        t = torch.linspace(0, 3, knots, dtype=torch.float32, device=device)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32, device=device)
        weights[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)
        self.register_buffer(
            "directions", torch.empty(dim, num_proj, dtype=torch.float32, device=device)
        )

    @torch.no_grad()
    def resample_directions(self):
        self.directions.normal_(generator=self.generator)
        self.directions.div_(self.directions.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def forward(self, z, sample_weight=None):
        """z: (T, B, D); match a standard Gaussian independently at each time index."""
        projected = z @ self.directions
        x_t = projected.unsqueeze(-1) * self.t
        effective_batch = z.new_tensor(float(z.size(1)))
        if sample_weight is None:
            cos_mean = x_t.cos().mean(dim=1)
            sin_mean = x_t.sin().mean(dim=1)
        else:
            weight = sample_weight.reshape(1, z.size(1), 1, 1)
            effective_batch = sample_weight.sum().clamp_min(1.0)
            cos_mean = (x_t.cos() * weight).sum(dim=1) / effective_batch
            sin_mean = (x_t.sin() * weight).sum(dim=1) / effective_batch
        err = (cos_mean - self.phi).square() + sin_mean.square()
        statistic = (err @ self.weights) * effective_batch
        return statistic.mean()


class LeWMProjectionMLP(nn.Module):
    """LeWM projector MLP; stateless BN keeps diagnostics from mutating control state."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, track_running_stats=False),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class LeWMRolloutProjector(nn.Module):
    """Sample-local projector suitable for both replay training and batch-one rollout."""

    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class LeWMResidualDynamics(nn.Module):
    """Action-conditioned residual dynamics in the private recurrent WM coordinate."""

    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim + action_dim),
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, latent, action):
        return latent + self.net(torch.cat([latent, action], dim=-1))


class LeWMResidualProjection(nn.Module):
    """Identity-initialized output projection that preserves recurrent copy dynamics."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, latent):
        return latent + self.net(latent)


class ScalarSIGReg(nn.Module):
    """Epps-Pulley standard-normal matching for one scalar token."""

    def __init__(self, knots, device):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32, device=device)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32, device=device)
        weights[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, token):
        x_t = token.unsqueeze(-1) * self.t
        error = (x_t.cos().mean(dim=0) - self.phi).square()
        error = error + x_t.sin().mean(dim=0).square()
        return ((error @ self.weights) * token.size(0)).mean()


class RewardTokenHead(nn.Module):
    """Action-conditioned scalar reward token with a nonlinear symlog decoder."""

    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.tokenizer = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.decoder_hidden_weight = nn.Parameter(torch.empty(64, 1))
        self.decoder_hidden_bias = nn.Parameter(torch.linspace(-3.0, 3.0, 64))
        self.decoder_output_weight = nn.Parameter(torch.empty(1, 64))
        self.decoder_output_bias = nn.Parameter(torch.zeros(1))
        # softplus(-4.6) ~= 0.01: start monotone without an exponentially huge symexp output.
        nn.init.constant_(self.decoder_hidden_weight, -4.6)
        nn.init.constant_(self.decoder_output_weight, -4.6)

    def tokenize(self, zsa, shared_scale):
        context = zsa.detach() + shared_scale * (zsa - zsa.detach())
        return self.tokenizer(context)

    def decode(self, token):
        hidden = F.softplus(
            F.linear(
                token,
                F.softplus(self.decoder_hidden_weight),
                self.decoder_hidden_bias,
            )
        )
        decoded = F.linear(
            hidden,
            F.softplus(self.decoder_output_weight),
            self.decoder_output_bias,
        )
        return decoded

    def forward(self, zsa, shared_scale):
        token = self.tokenize(zsa, shared_scale)
        return token, self.decode(token)


class PolicyMeanHead(nn.Module):
    """Predict TD7's deterministic target-policy mean from the observation latent."""

    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, zs):
        return torch.tanh(self.net(zs))


class MonotoneRewardTokenizer(nn.Module):
    """Strictly monotone scalar tokenizer, so its Gaussian coordinate cannot discard reward order."""

    def __init__(self, num_knots=17):
        super().__init__()
        self.register_buffer("knots", torch.linspace(-3.0, 3.0, num_knots))
        self.log_width = nn.Parameter(torch.zeros(num_knots))
        self.log_weight = nn.Parameter(torch.full((num_knots,), -4.6))
        self.log_base_scale = nn.Parameter(torch.tensor(0.54132485))  # softplus(x) = 1
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, symlog_reward):
        width = F.softplus(self.log_width).clamp_min(1e-3)
        features = torch.tanh(
            (symlog_reward - self.knots.unsqueeze(0)) / width.unsqueeze(0)
        )
        nonlinear = features @ F.softplus(self.log_weight).unsqueeze(1)
        return self.bias + F.softplus(self.log_base_scale) * symlog_reward + nonlinear


class ResidualPolicyTokenizer(nn.Module):
    """Near-identity policy token whose learned coordinate is stabilized by its own SIGReg."""

    def __init__(self, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, policy_mean):
        return policy_mean + self.net(policy_mean)


class LeJEPAOutcomeTokens(nn.Module):
    """Attached LeJEPA targets for reward and the deterministic policy at the next state."""

    def __init__(
        self,
        latent_dim,
        action_dim,
        policy_token_dim,
        hidden_dim,
        knots,
        from_transition=False,
    ):
        super().__init__()
        self.from_transition = from_transition
        context_dim = latent_dim if from_transition else latent_dim + action_dim
        self.reward_tokenizer = MonotoneRewardTokenizer(knots)
        self.policy_tokenizer = ResidualPolicyTokenizer(policy_token_dim, hidden_dim)
        self.reward_predictor = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.policy_predictor = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, policy_token_dim),
        )

    def forward(
        self,
        state_token,
        action,
        reward,
        next_policy_mean,
        transition_token=None,
    ):
        if self.from_transition:
            if transition_token is None:
                raise ValueError("transition_token is required for shared-transition outcomes")
            context = transition_token
        else:
            context = torch.cat([state_token, action], dim=-1)
        reward_target = self.reward_tokenizer(symlog(reward))
        policy_target = self.policy_tokenizer(next_policy_mean)
        reward_prediction = self.reward_predictor(context)
        policy_prediction = self.policy_predictor(context)
        return reward_prediction, reward_target, policy_prediction, policy_target


class SemanticOutcomeTokens(nn.Module):
    """Dedicated 64-D-style readout tokens for reward and the successor behavior policy.

    Reward is a proper categorical semantic target, not a Gaussianized scalar embedding:
    uniformly spaced coordinates cover symlog(raw_min) through symlog(raw_max), labels use
    HL-Gauss smoothing, and scalar diagnostics decode E[symexp(bin)]. Policy directly predicts
    the bounded action mean plus a [-1, 1]-normalized log-standard-deviation vector.
    """

    distributional = True

    def __init__(
        self,
        latent_dim,
        action_dim,
        token_dim,
        reward_num_bins,
        reward_raw_min,
        reward_raw_max,
        reward_sigma_ratio,
        reward_prior_floor,
    ):
        super().__init__()
        self.reward_raw_min = reward_raw_min
        self.reward_raw_max = reward_raw_max
        self.reward_num_bins = reward_num_bins
        reward_coord_min = np.sign(reward_raw_min) * np.log1p(abs(reward_raw_min))
        reward_coord_max = np.sign(reward_raw_max) * np.log1p(abs(reward_raw_max))
        reward_support = torch.linspace(reward_coord_min, reward_coord_max, reward_num_bins)
        self.register_buffer("reward_support", reward_support)
        self.register_buffer("reward_scalar_support", symexp(reward_support))
        self.reward_bin_width = (reward_coord_max - reward_coord_min) / (reward_num_bins - 1)
        self.reward_sigma = reward_sigma_ratio * self.reward_bin_width

        self.reward_query = nn.Parameter(torch.empty(token_dim))
        self.policy_query = nn.Parameter(torch.empty(token_dim))
        nn.init.normal_(self.reward_query, std=token_dim**-0.5)
        nn.init.normal_(self.policy_query, std=token_dim**-0.5)
        self.reward_tokenizer = nn.Sequential(
            nn.RMSNorm(latent_dim),
            nn.Linear(latent_dim, token_dim),
            nn.SiLU(),
        )
        self.policy_tokenizer = nn.Sequential(
            nn.RMSNorm(latent_dim),
            nn.Linear(latent_dim, token_dim),
            nn.SiLU(),
        )
        self.reward_action_proj = nn.Linear(action_dim, token_dim, bias=False)
        self.reward_predictor = nn.Sequential(
            nn.RMSNorm(token_dim),
            nn.Linear(token_dim, reward_num_bins),
        )
        self.policy_predictor = nn.Sequential(
            nn.RMSNorm(token_dim),
            nn.Linear(token_dim, 2 * action_dim),
        )
        nn.init.xavier_uniform_(self.reward_action_proj.weight)
        nn.init.xavier_uniform_(self.policy_predictor[-1].weight)
        nn.init.zeros_(self.policy_predictor[-1].bias)

        # Match v215's calibrated zero-reward prior. Initially the reward head is independent of
        # its token, then receives a useful token gradient as soon as its decoder leaves zero.
        with torch.no_grad():
            zero_prior = self.project_reward(torch.zeros(1))
            self.reward_predictor[-1].weight.zero_()
            self.reward_predictor[-1].bias.copy_(
                zero_prior[0].clamp_min(reward_prior_floor).log()
            )

    def project_reward(self, reward):
        reward_coord = symlog(reward.reshape(-1)).clamp(
            self.reward_support[0], self.reward_support[-1]
        )
        support = self.reward_support.unsqueeze(0)
        half_width = 0.5 * self.reward_bin_width
        upper = (support + half_width - reward_coord.unsqueeze(-1)) / self.reward_sigma
        lower = (support - half_width - reward_coord.unsqueeze(-1)) / self.reward_sigma
        probs = 0.5 * (
            torch.erf(upper / np.sqrt(2.0)) - torch.erf(lower / np.sqrt(2.0))
        )
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-10)

    def reward_to_scalar(self, logits):
        return (torch.softmax(logits, dim=-1) * self.reward_scalar_support).sum(dim=-1)

    def gradient_parameter_groups(self):
        return (
            [self.reward_query, *self.reward_tokenizer.parameters()],
            [*self.reward_action_proj.parameters(), *self.reward_predictor.parameters()],
            [self.policy_query, *self.policy_tokenizer.parameters()],
            list(self.policy_predictor.parameters()),
        )

    def forward(
        self,
        state_token,
        action,
        reward,
        next_policy_moments,
        transition_token=None,
    ):
        del state_token
        if transition_token is None:
            raise ValueError("semantic outcome tokens require a transition token")
        reward_token = self.reward_query + self.reward_tokenizer(transition_token)
        policy_token = self.policy_query + self.policy_tokenizer(transition_token)
        reward_logits = self.reward_predictor[1](
            self.reward_predictor[0](reward_token) + self.reward_action_proj(action)
        )
        policy_moments = self.policy_predictor(policy_token)
        return (
            reward_logits,
            self.project_reward(reward),
            policy_moments,
            next_policy_moments,
            reward_token,
            policy_token,
        )


class LatentOutcomeTokens(nn.Module):
    """Attached 64-D reward/policy targets predicted only in latent space.

    Fixed semantic observations first enter learned target encoders: an HL-Gauss reward
    distribution and the successor Gaussian policy moments. Separate world-side predictors map
    the private transition into the same token spaces. Decoders train only on target tokens, so
    reward CE and policy reconstruction anchor meaning without directly supervising the world
    transition; attached latent MSE is the sole predictor-to-outcome objective.
    """

    latent_targets = True

    def __init__(
        self,
        latent_dim,
        action_dim,
        token_dim,
        reward_num_bins,
        reward_raw_min,
        reward_raw_max,
        reward_sigma_ratio,
        reward_prior_floor,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.reward_raw_min = reward_raw_min
        self.reward_raw_max = reward_raw_max
        reward_coord_min = np.sign(reward_raw_min) * np.log1p(abs(reward_raw_min))
        reward_coord_max = np.sign(reward_raw_max) * np.log1p(abs(reward_raw_max))
        reward_support = torch.linspace(reward_coord_min, reward_coord_max, reward_num_bins)
        self.register_buffer("reward_support", reward_support)
        self.register_buffer("reward_scalar_support", symexp(reward_support))
        self.reward_bin_width = (reward_coord_max - reward_coord_min) / (reward_num_bins - 1)
        self.reward_sigma = reward_sigma_ratio * self.reward_bin_width

        self.reward_tokenizer = nn.Sequential(
            nn.LayerNorm(reward_num_bins),
            nn.Linear(reward_num_bins, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.policy_tokenizer = nn.Sequential(
            nn.LayerNorm(2 * action_dim),
            nn.Linear(2 * action_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.reward_predictor = nn.Sequential(
            nn.RMSNorm(latent_dim),
            nn.Linear(latent_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.policy_predictor = nn.Sequential(
            nn.RMSNorm(latent_dim),
            nn.Linear(latent_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.reward_action_proj = nn.Linear(action_dim, token_dim, bias=False)
        self.reward_decoder = nn.Sequential(
            nn.RMSNorm(token_dim),
            nn.Linear(token_dim, reward_num_bins),
        )
        self.policy_decoder = nn.Sequential(
            nn.RMSNorm(token_dim),
            nn.Linear(token_dim, 2 * action_dim),
        )
        nn.init.xavier_uniform_(self.reward_action_proj.weight)
        nn.init.xavier_uniform_(self.policy_decoder[-1].weight)
        nn.init.zeros_(self.policy_decoder[-1].bias)

        # Target-token reward semantics start as the calibrated zero-reward distribution. The
        # target encoder becomes semantically anchored once this decoder takes its first update.
        with torch.no_grad():
            zero_prior = self.project_reward(torch.zeros(1))
            self.reward_decoder[-1].weight.zero_()
            self.reward_decoder[-1].bias.copy_(
                zero_prior[0].clamp_min(reward_prior_floor).log()
            )

    def project_reward(self, reward):
        reward_coord = symlog(reward.reshape(-1)).clamp(
            self.reward_support[0], self.reward_support[-1]
        )
        support = self.reward_support.unsqueeze(0)
        half_width = 0.5 * self.reward_bin_width
        upper = (support + half_width - reward_coord.unsqueeze(-1)) / self.reward_sigma
        lower = (support - half_width - reward_coord.unsqueeze(-1)) / self.reward_sigma
        probs = 0.5 * (
            torch.erf(upper / np.sqrt(2.0)) - torch.erf(lower / np.sqrt(2.0))
        )
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-10)

    def reward_to_scalar(self, logits):
        return (torch.softmax(logits, dim=-1) * self.reward_scalar_support).sum(dim=-1)

    def gradient_parameter_groups(self):
        return (
            [*self.reward_tokenizer.parameters(), *self.reward_decoder.parameters()],
            [*self.reward_predictor.parameters(), *self.reward_action_proj.parameters()],
            [*self.policy_tokenizer.parameters(), *self.policy_decoder.parameters()],
            list(self.policy_predictor.parameters()),
        )

    def forward(
        self,
        state_token,
        action,
        reward,
        next_policy_moments,
        transition_token=None,
    ):
        del state_token
        if transition_token is None:
            raise ValueError("latent outcome tokens require a transition token")
        reward_distribution = self.project_reward(reward)
        reward_target = self.reward_tokenizer(reward_distribution)
        policy_target = self.policy_tokenizer(next_policy_moments)
        reward_prediction = self.reward_predictor(transition_token)
        reward_prediction = reward_prediction + self.reward_action_proj(action)
        policy_prediction = self.policy_predictor(transition_token)
        reward_reconstruction_logits = self.reward_decoder(reward_target)
        policy_reconstruction = self.policy_decoder(policy_target)
        return (
            reward_prediction,
            reward_target,
            policy_prediction,
            policy_target,
            reward_reconstruction_logits,
            reward_distribution,
            policy_reconstruction,
            next_policy_moments,
        )


class IsometricOutcomeTokens(nn.Module):
    """Gradient-shaped target encoders that cannot discard outcome information.

    By default, reward probabilities and successor policy moments enter learned semi-orthogonal
    linear maps. Their columns remain orthonormal, so the 64-D targets preserve input distances
    exactly even as attached latent-MSE gradients rotate their coordinates. Optional direct
    Beta-NLL policy and HL-Gauss reward objectives remove their corresponding target maps.
    """

    isometric_targets = True

    def __init__(
        self,
        latent_dim,
        action_dim,
        token_dim,
        reward_num_bins,
        reward_raw_min,
        reward_raw_max,
        reward_sigma_ratio,
        policy_beta_nll=False,
        policy_beta_nll_eps=1e-5,
        policy_beta_max_precision=0.0,
        reward_hlgauss_ce=False,
        reward_hlgauss_symlog=True,
    ):
        super().__init__()
        if token_dim < 2 * action_dim or (
            not reward_hlgauss_ce and token_dim < reward_num_bins
        ):
            raise ValueError(
                "isometric target width must cover every embedded semantic input dimension"
            )
        initial_beta_precision = 2.0 * np.log(2.0)
        if (
            not np.isfinite(policy_beta_max_precision)
            or policy_beta_max_precision < 0.0
            or 0.0 < policy_beta_max_precision <= initial_beta_precision
        ):
            raise ValueError(
                "policy_beta_max_precision must be zero or finite and exceed 2*log(2)"
            )
        self.action_dim = action_dim
        self.policy_beta_nll = policy_beta_nll
        self.policy_beta_nll_eps = policy_beta_nll_eps
        self.policy_beta_max_precision = policy_beta_max_precision
        self.reward_hlgauss_ce = reward_hlgauss_ce
        self.reward_hlgauss_symlog = reward_hlgauss_symlog
        self.register_buffer("reward_trunk_scale", torch.ones(()))
        self.register_buffer("policy_trunk_scale", torch.ones(()))
        self.reward_raw_min = reward_raw_min
        self.reward_raw_max = reward_raw_max
        if reward_hlgauss_symlog:
            reward_coord_min = np.sign(reward_raw_min) * np.log1p(abs(reward_raw_min))
            reward_coord_max = np.sign(reward_raw_max) * np.log1p(abs(reward_raw_max))
        else:
            reward_coord_min = reward_raw_min
            reward_coord_max = reward_raw_max
        reward_support = torch.linspace(reward_coord_min, reward_coord_max, reward_num_bins)
        self.register_buffer("reward_support", reward_support)
        reward_scalar_support = (
            symexp(reward_support) if reward_hlgauss_symlog else reward_support.clone()
        )
        self.register_buffer("reward_scalar_support", reward_scalar_support)
        self.reward_bin_width = (reward_coord_max - reward_coord_min) / (reward_num_bins - 1)
        self.reward_sigma = reward_sigma_ratio * self.reward_bin_width

        # Direct reward prediction deletes this legacy tokenizer below. Keep its input width at
        # v8/v11's 51 bins so changing only the direct readout resolution cannot perturb the
        # subsequent policy-head RNG stream.
        legacy_reward_num_bins = 51 if reward_hlgauss_ce else reward_num_bins
        self.reward_tokenizer = orthogonal(
            nn.Linear(legacy_reward_num_bins, token_dim, bias=False),
            orthogonal_map="cayley",
            use_trivialization=False,
        )
        if policy_beta_nll:
            # Consume the legacy policy-token RNG stream so the reward branch remains exactly
            # initialized as in v6. The replacement Beta head is initialized later in a fork.
            legacy_policy_tokenizer = orthogonal(
                nn.Linear(2 * action_dim, token_dim, bias=False),
                orthogonal_map="cayley",
                use_trivialization=False,
            )
        else:
            self.policy_tokenizer = orthogonal(
                nn.Linear(2 * action_dim, token_dim, bias=False),
                orthogonal_map="cayley",
                use_trivialization=False,
            )
        self.reward_predictor = nn.Sequential(
            nn.RMSNorm(latent_dim),
            nn.Linear(latent_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        legacy_policy_predictor = nn.Sequential(
            nn.RMSNorm(latent_dim),
            nn.Linear(latent_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        if not policy_beta_nll:
            self.policy_predictor = legacy_policy_predictor
        self.reward_action_proj = nn.Linear(action_dim, token_dim, bias=False)
        nn.init.xavier_uniform_(self.reward_action_proj.weight)
        if policy_beta_nll:
            with torch.random.fork_rng():
                policy_readout = nn.Linear(
                    token_dim, 4 * action_dim, bias=False
                )
                nn.init.normal_(policy_readout.weight, std=1e-3)
                self.policy_predictor = nn.Sequential(
                    nn.RMSNorm(latent_dim),
                    nn.Linear(latent_dim, token_dim),
                    nn.SiLU(),
                    policy_readout,
                )
            del legacy_policy_tokenizer, legacy_policy_predictor
        if reward_hlgauss_ce:
            # Build the direct semantic branch only after consuming v8's complete initialization
            # stream, so the bounded policy-Beta branch remains an exact architectural control.
            with torch.random.fork_rng():
                direct_reward_predictor = nn.Sequential(
                    nn.RMSNorm(latent_dim),
                    nn.Linear(latent_dim, token_dim),
                    nn.SiLU(),
                )
                direct_reward_action_proj = nn.Linear(
                    action_dim, token_dim, bias=False
                )
                # The output layer is zeroed below, so isolate its variable-size initialization
                # from the meaningful action projection. Consume the legacy 51-bin stream outside
                # the nested fork to preserve v9/v11 initialization exactly at every resolution.
                with torch.random.fork_rng():
                    direct_reward_readout = nn.Sequential(
                        nn.RMSNorm(token_dim),
                        nn.Linear(token_dim, reward_num_bins),
                    )
                legacy_reward_readout = nn.Linear(token_dim, 51)
                nn.init.xavier_uniform_(direct_reward_action_proj.weight)
                del legacy_reward_readout
                with torch.no_grad():
                    zero_prior = self.project_reward(torch.zeros(1))
                    direct_reward_readout[-1].weight.zero_()
                    direct_reward_readout[-1].bias.copy_(
                        zero_prior[0].clamp_min(1e-20).log()
                    )
            self.reward_predictor = direct_reward_predictor
            self.reward_action_proj = direct_reward_action_proj
            self.reward_readout = direct_reward_readout
            del self.reward_tokenizer

    def project_reward(self, reward):
        reward_coord = reward.reshape(-1)
        if self.reward_hlgauss_symlog:
            reward_coord = symlog(reward_coord)
        reward_coord = reward_coord.clamp(
            self.reward_support[0], self.reward_support[-1]
        )
        support = self.reward_support.unsqueeze(0)
        half_width = 0.5 * self.reward_bin_width
        upper = (support + half_width - reward_coord.unsqueeze(-1)) / self.reward_sigma
        lower = (support - half_width - reward_coord.unsqueeze(-1)) / self.reward_sigma
        probs = 0.5 * (
            torch.erf(upper / np.sqrt(2.0)) - torch.erf(lower / np.sqrt(2.0))
        )
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-10)

    def decode_reward_token(self, token):
        if self.reward_hlgauss_ce:
            raise RuntimeError("direct HL-Gauss reward predictions are logits, not tokens")
        return token @ self.reward_tokenizer.weight

    def reward_to_scalar(self, logits):
        return (torch.softmax(logits, dim=-1) * self.reward_scalar_support).sum(dim=-1)

    @torch.no_grad()
    def set_trunk_scales(self, scales):
        self.reward_trunk_scale.copy_(scales[0])
        self.policy_trunk_scale.copy_(scales[1])

    def decode_policy_token(self, token):
        if self.policy_beta_nll:
            raise RuntimeError("policy-moment Beta predictions are parameters, not tokens")
        return token @ self.policy_tokenizer.weight

    def policy_moment_beta(self, raw_parameters):
        alpha_raw, beta_raw = raw_parameters.chunk(2, dim=-1)
        if self.policy_beta_max_precision > 0.0:
            # Separate location from confidence and bound the latter. Deterministic targets make
            # unconstrained Beta MLE drive alpha + beta to infinity. The offset preserves v7's
            # initial alpha = beta = 1 + softplus(0) while capping concentration thereafter.
            mean = torch.sigmoid(alpha_raw)
            initial_precision = 2.0 * np.log(2.0)
            precision_fraction = initial_precision / self.policy_beta_max_precision
            precision_offset = np.log(
                precision_fraction / (1.0 - precision_fraction)
            )
            precision = self.policy_beta_max_precision * torch.sigmoid(
                beta_raw + precision_offset
            )
            return 1.0 + mean * precision, 1.0 + (1.0 - mean) * precision
        return F.softplus(alpha_raw) + 1.0, F.softplus(beta_raw) + 1.0

    def policy_moment_beta_nll(self, raw_parameters, policy_moment_target):
        target = (0.5 * (policy_moment_target + 1.0)).clamp(
            self.policy_beta_nll_eps, 1.0 - self.policy_beta_nll_eps
        )
        alpha, beta = self.policy_moment_beta(raw_parameters)
        log_probability = (
            (alpha - 1.0) * target.log()
            + (beta - 1.0) * torch.log1p(-target)
            + torch.lgamma(alpha + beta)
            - torch.lgamma(alpha)
            - torch.lgamma(beta)
        )
        return -log_probability, target, alpha, beta

    def gradient_parameter_groups(self):
        policy_target_parameters = (
            []
            if self.policy_beta_nll
            else list(self.policy_tokenizer.parameters())
        )
        reward_target_parameters = (
            []
            if self.reward_hlgauss_ce
            else list(self.reward_tokenizer.parameters())
        )
        reward_predictor_parameters = [
            *self.reward_predictor.parameters(),
            *self.reward_action_proj.parameters(),
        ]
        if self.reward_hlgauss_ce:
            reward_predictor_parameters.extend(self.reward_readout.parameters())
        return (
            reward_target_parameters,
            reward_predictor_parameters,
            policy_target_parameters,
            list(self.policy_predictor.parameters()),
        )

    def forward(
        self,
        state_token,
        action,
        reward,
        next_policy_moments,
        transition_token=None,
    ):
        del state_token
        if transition_token is None:
            raise ValueError("isometric outcome tokens require a transition token")
        reward_distribution = self.project_reward(reward)
        reward_target = reward_distribution
        if not self.reward_hlgauss_ce:
            reward_target = self.reward_tokenizer(reward_distribution)
        policy_target = next_policy_moments
        if not self.policy_beta_nll:
            policy_target = self.policy_tokenizer(next_policy_moments)
        reward_transition = transition_token.detach() + self.reward_trunk_scale * (
            transition_token - transition_token.detach()
        )
        policy_transition = transition_token.detach() + self.policy_trunk_scale * (
            transition_token - transition_token.detach()
        )
        reward_prediction = self.reward_predictor(reward_transition)
        reward_prediction = reward_prediction + self.reward_action_proj(action)
        if self.reward_hlgauss_ce:
            reward_prediction = self.reward_readout(reward_prediction)
        policy_prediction = self.policy_predictor(policy_transition)
        return (
            reward_prediction,
            reward_target,
            policy_prediction,
            policy_target,
            reward_distribution,
            next_policy_moments,
        )

    @torch.no_grad()
    def inverse_diagnostics(
        self,
        action,
        reward,
        next_policy_moments,
        transition_token,
        policy_valid,
        policy_logstd_half_range,
    ):
        (
            reward_prediction,
            _,
            policy_prediction,
            _,
            reward_distribution,
            policy_semantic_target,
        ) = self(None, action, reward, next_policy_moments, transition_token)
        if self.reward_hlgauss_ce:
            recovered_reward_probs = torch.softmax(reward_prediction, dim=-1)
            reward_diagnostic = -(
                reward_distribution * F.log_softmax(reward_prediction, dim=-1)
            ).sum(dim=-1).mean()
        else:
            recovered_reward = self.decode_reward_token(reward_prediction)
            reward_diagnostic = F.mse_loss(recovered_reward, reward_distribution)
            recovered_reward_probs = recovered_reward.clamp_min(0.0)
            recovered_reward_probs = recovered_reward_probs / (
                recovered_reward_probs.sum(dim=-1, keepdim=True).clamp_min(1e-10)
            )
        if self.policy_beta_nll:
            alpha, beta = self.policy_moment_beta(policy_prediction)
            recovered_policy = 2.0 * alpha / (alpha + beta) - 1.0
        else:
            recovered_policy = self.decode_policy_token(policy_prediction)
        policy_sample_mse = F.mse_loss(
            recovered_policy, policy_semantic_target, reduction="none"
        ).mean(dim=-1, keepdim=True)
        policy_inverse_mse = (
            policy_sample_mse * policy_valid
        ).sum() / policy_valid.sum().clamp_min(1.0)
        decoded_reward = (
            recovered_reward_probs * self.reward_scalar_support
        ).sum(dim=-1)
        reward_inverse_decode_mae = (
            decoded_reward - reward.reshape(-1)
        ).abs().mean()
        policy_error = (recovered_policy - policy_semantic_target).abs()
        policy_mean_mae = (
            policy_error[:, : self.action_dim].mean(dim=-1, keepdim=True)
            * policy_valid
        ).sum() / policy_valid.sum().clamp_min(1.0)
        policy_logstd_mae = (
            policy_error[:, self.action_dim :].mean(dim=-1, keepdim=True)
            * policy_valid
        ).sum() / policy_valid.sum().clamp_min(1.0)
        diagnostics = {
            "policy_inverse_mse": policy_inverse_mse,
            "policy_mean_mae": policy_mean_mae,
            "policy_logstd_mae": policy_logstd_mae * policy_logstd_half_range,
            "reward_pred_edge_mass": (
                recovered_reward_probs[:, 0] + recovered_reward_probs[:, -1]
            ).mean(),
        }
        diagnostics[
            "reward_hlgauss_ce" if self.reward_hlgauss_ce else "reward_inverse_mse"
        ] = reward_diagnostic
        diagnostics[
            "reward_decode_mae"
            if self.reward_hlgauss_ce
            else "reward_inverse_decode_mae"
        ] = reward_inverse_decode_mae
        if self.policy_beta_nll:
            concentration = alpha + beta
            beta_std = 2.0 * torch.sqrt(
                alpha * beta
                / (concentration.square() * (concentration + 1.0))
            )
            diagnostics.update(
                {
                    "policy_beta_std": (
                        beta_std.mean(dim=-1, keepdim=True) * policy_valid
                    ).sum()
                    / policy_valid.sum().clamp_min(1.0),
                    "policy_beta_concentration": (
                        concentration.mean(dim=-1, keepdim=True) * policy_valid
                    ).sum()
                    / policy_valid.sum().clamp_min(1.0),
                }
            )
        return diagnostics


class JEDIResidualBlock(nn.Module):
    """Vector AdaLN residual block for noise- and transition-conditioned denoising."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, x, condition):
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        residual = self.norm(x) * (1.0 + scale) + shift
        return x + torch.tanh(gate) * self.mlp(residual)


class JEDIVectorDenoiser(nn.Module):
    """Small EDM F-network conditioned on TD7's deterministic (zs, zsa) pair."""

    def __init__(self, latent_dim, hidden_dim, time_dim, num_blocks):
        super().__init__()
        if time_dim % 2:
            raise ValueError("jedi_time_dim must be even")
        half_dim = time_dim // 2
        frequencies = torch.exp(torch.linspace(0.0, np.log(1_000.0), half_dim))
        self.register_buffer("time_frequencies", frequencies)
        self.input_projection = nn.Linear(latent_dim, hidden_dim)
        self.context_projection = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(JEDIResidualBlock(hidden_dim) for _ in range(num_blocks))
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, latent_dim)
        # Start from the EDM skip estimate. Stock SALE and SubSIG therefore own the representation
        # initially, while the denoiser learns its output map before shaping the shared context.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, noised_input, c_noise, context_zs, context_zsa):
        phase = 2.0 * np.pi * c_noise * self.time_frequencies
        time_embedding = torch.cat([phase.sin(), phase.cos()], dim=-1)
        condition = self.context_projection(torch.cat([context_zs, context_zsa], dim=-1))
        condition = condition + self.time_projection(time_embedding)
        hidden = self.input_projection(noised_input)
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output_projection(F.silu(self.output_norm(hidden)))


class JEDIEndpointSampler(nn.Module):
    """JEDI/DIAMOND three-step Euler endpoint with a fixed, batch-invariant Gaussian prior."""

    def __init__(
        self,
        denoiser,
        fixed_prior,
        sigma_data=1.0,
        clamp=3.0,
        canonical_context=False,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.sigma_data = sigma_data
        self.clamp = clamp
        self.latent_scale = JEDI_LATENT_SCALE
        self.canonical_context = canonical_context
        self.register_buffer(
            "sigmas",
            torch.tensor([5.0, 0.28308171033859253, 0.002, 0.0], dtype=torch.float32),
        )
        self.register_buffer("fixed_prior", fixed_prior.detach().clone())

    def _soft_bound(self, value):
        return self.clamp * torch.tanh(value / self.clamp)

    def _bound_denoised(self, value):
        if self.canonical_context:
            # The training target is already C(alpha*z). Applying C again would move even a
            # perfect x0 prediction. A support clamp is identity on valid canonical targets.
            return value.clamp(-self.clamp, self.clamp)
        return self._soft_bound(value)

    def _contexts(self, zs, stock_zsa):
        if self.canonical_context:
            return self.latent_scale * zs, self.latent_scale * stock_zsa
        return (
            self._soft_bound(self.latent_scale * zs),
            self._soft_bound(self.latent_scale * stock_zsa),
        )

    def endpoint_with_prior(self, zs, stock_zsa, prior):
        context_zs, context_zsa = self._contexts(zs, stock_zsa)
        x = prior.expand(zs.size(0), -1)
        for index in range(3):
            sigma = self.sigmas[index]
            next_sigma = self.sigmas[index + 1]
            denominator = torch.sqrt(sigma.square() + self.sigma_data**2)
            c_in = denominator.reciprocal()
            c_skip = self.sigma_data**2 / denominator.square()
            c_out = sigma * self.sigma_data / denominator
            c_noise = (sigma.log() / 4.0).expand(zs.size(0), 1)
            prediction_f = self.denoiser(c_in * x, c_noise, context_zs, context_zsa)
            denoised = self._bound_denoised(c_skip * x + c_out * prediction_f)
            derivative = (x - denoised) / sigma
            x = x + (next_sigma - sigma) * derivative
        # The EDM branch is trained after sqrt(2/pi) scaling. Undo that fixed scale so the critic's
        # endpoint remains comparable to stock SALE's raw AvgL1-normalized next-latent estimate.
        return x / self.latent_scale

    def forward(self, zs, stock_zsa):
        return self.endpoint_with_prior(zs, stock_zsa, self.fixed_prior)

    @torch.no_grad()
    def trajectory_diagnostics(self, zs, stock_zsa):
        context_zs, context_zsa = self._contexts(zs, stock_zsa)
        x = self.fixed_prior.expand(zs.size(0), -1)
        step_rms = []
        clamp_fraction = []
        for index in range(3):
            sigma = self.sigmas[index]
            next_sigma = self.sigmas[index + 1]
            denominator = torch.sqrt(sigma.square() + self.sigma_data**2)
            c_in = denominator.reciprocal()
            c_skip = self.sigma_data**2 / denominator.square()
            c_out = sigma * self.sigma_data / denominator
            c_noise = (sigma.log() / 4.0).expand(zs.size(0), 1)
            prediction_f = self.denoiser(c_in * x, c_noise, context_zs, context_zsa)
            raw_denoised = c_skip * x + c_out * prediction_f
            denoised = self._bound_denoised(raw_denoised)
            derivative = (x - denoised) / sigma
            next_x = x + (next_sigma - sigma) * derivative
            step_rms.append((next_x - x).square().mean().sqrt())
            clamp_fraction.append((raw_denoised.abs() > self.clamp).float().mean())
            x = next_x
        return torch.stack(step_rms), torch.stack(clamp_fraction)


def freeze(module):
    module.requires_grad_(False)
    return module


def equalized_outcome_trunk_scales(
    representation_grad_norm,
    reward_raw_grad_norm,
    policy_raw_grad_norm,
):
    """Split one representation-gradient budget equally across two outcome branches."""

    half_budget = 0.5 * representation_grad_norm
    return torch.stack(
        (
            half_budget / reward_raw_grad_norm.clamp_min(1e-12),
            half_budget / policy_raw_grad_norm.clamp_min(1e-12),
        )
    )


class DreamerLossNormalizer(nn.Module):
    """Dreamer4's lagged RMS loss normalization, reproduced exactly."""

    def __init__(self, beta=0.95, eps=1e-6):
        super().__init__()
        self.register_buffer("exp_avg_sq", torch.ones(()))
        self.register_buffer("last_rms", torch.ones(()), persistent=False)
        self.beta = beta
        self.eps = eps

    def forward(self, loss, update_ema=None):
        if loss.numel() != 1:
            raise ValueError("DreamerLossNormalizer expects one scalar loss")
        if update_ema is None:
            update_ema = self.training
        rms = self.exp_avg_sq.sqrt()
        self.last_rms.copy_(rms.detach())
        if update_ema:
            self.exp_avg_sq.lerp_(loss.detach().square(), 1.0 - self.beta)
        return loss / rms.clamp_min(self.eps)


class EncoderLossCore(nn.Module):
    """One static owner for SALE prediction, anti-collapse, and optional JEDI gradients."""

    def __init__(
        self,
        encoder,
        subsig,
        jedi_denoiser,
        args,
        lewm_projector=None,
        lewm_dynamics=None,
        lewm_pred_projector=None,
        full_sigreg=None,
        reward_token_head=None,
        reward_token_sigreg=None,
        policy_mean_head=None,
        outcome_tokens=None,
        reward_outcome_sigreg=None,
        policy_outcome_sigreg=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.subsig = subsig
        self.jedi_denoiser = jedi_denoiser
        self.lewm_projector = lewm_projector
        self.lewm_dynamics = lewm_dynamics
        self.lewm_pred_projector = lewm_pred_projector
        self.full_sigreg = full_sigreg
        self.reward_token_head = reward_token_head
        self.reward_token_sigreg = reward_token_sigreg
        self.policy_mean_head = policy_mean_head
        self.outcome_tokens = outcome_tokens
        self.reward_outcome_sigreg = reward_outcome_sigreg
        self.policy_outcome_sigreg = policy_outcome_sigreg
        self.prediction_from_lap = args.prediction_from_lap
        self.attached_target = args.attached_target
        self.subsig_coef = args.subsig_coef

        self.jedi_coef = args.jedi_coef
        self.jedi_p_mean = args.jedi_p_mean
        self.jedi_p_std = args.jedi_p_std
        self.jedi_sigma_data = args.jedi_sigma_data
        self.jedi_clamp = args.jedi_clamp
        self.jedi_latent_scale = 1.0 if lewm_projector is not None else JEDI_LATENT_SCALE
        self.canonical_control_latents = args.jedi_canonical_control_latents
        self.lewm_sigreg_coef = args.lewm_sigreg_coef
        self.reward_token_sigreg_coef = args.reward_token_sigreg_coef
        self.reward_token_shared_scale = args.reward_token_shared_scale
        self.reward_control_cost_coef = args.reward_control_cost_coef
        self.reward_sigreg_tokenizer_only = args.reward_sigreg_tokenizer_only
        self.policy_mean_coef = args.policy_mean_coef
        self.outcome_token_coef = args.outcome_token_coef
        self.policy_beta_nll_coef = (
            args.outcome_token_coef
            if args.policy_beta_nll_coef < 0.0
            else args.policy_beta_nll_coef
        )
        self.outcome_sigreg_coef = args.outcome_sigreg_coef
        self.outcome_semantic_coef = args.outcome_semantic_coef
        self.outcome_sigreg_batch_normalized = args.outcome_sigreg_batch_normalized
        self.outcome_from_transition = args.outcome_from_transition
        self.sigreg_batch_size = args.sigreg_batch_size
        self.control_sigreg_batch_size = args.control_sigreg_batch_size
        self.dreamer_loss_normalization = args.dreamer_loss_normalization
        self.prediction_loss_normalizer = None
        self.lewm_prediction_loss_normalizer = None
        self.reward_outcome_loss_normalizer = None
        self.policy_outcome_loss_normalizer = None
        if self.dreamer_loss_normalization:
            normalizer_kwargs = {
                "beta": args.loss_normalization_beta,
                "eps": args.loss_normalization_eps,
            }
            self.prediction_loss_normalizer = DreamerLossNormalizer(
                **normalizer_kwargs
            )
            self.lewm_prediction_loss_normalizer = DreamerLossNormalizer(
                **normalizer_kwargs
            )
            self.reward_outcome_loss_normalizer = DreamerLossNormalizer(
                **normalizer_kwargs
            )
            self.policy_outcome_loss_normalizer = DreamerLossNormalizer(
                **normalizer_kwargs
            )
        policy_logstd_min = args.sac_log_std_min if args.sac_policy else args.sd_log_std_min
        policy_logstd_max = args.sac_log_std_max if args.sac_policy else args.sd_log_std_max
        self.policy_logstd_half_range = 0.5 * (policy_logstd_max - policy_logstd_min)

    def _soft_bound(self, value):
        return self.jedi_clamp * torch.tanh(value / self.jedi_clamp)

    def _control_zs(self, raw_zs):
        if self.canonical_control_latents:
            return jedi_control_latent(raw_zs, self.jedi_clamp)
        return raw_zs

    def _stock_prediction(self, raw_zs, action):
        prediction = self.encoder.zsa(self._control_zs(raw_zs), action)
        if self.canonical_control_latents:
            prediction = jedi_control_latent(prediction, self.jedi_clamp)
        return prediction

    def _world_latents(self, raw_zs, action, stock_prediction, raw_next_zs):
        if self.lewm_projector is None:
            return raw_zs, stock_prediction, raw_next_zs
        batch_size = raw_zs.size(0)
        projected_pair = self.lewm_projector(torch.cat([raw_zs, raw_next_zs], dim=0))
        projected_zs, projected_next_zs = projected_pair.split(batch_size, dim=0)
        prediction_input = stock_prediction
        if self.lewm_dynamics is not None:
            prediction_input = self.lewm_dynamics(projected_zs, action)
        return (
            projected_zs,
            self.lewm_pred_projector(prediction_input),
            projected_next_zs,
        )

    def _jedi_components(self, prediction_zs, prediction, prediction_next_zs, sigma_normal, epsilon):
        if self.canonical_control_latents:
            context_zs = self.jedi_latent_scale * self._control_zs(prediction_zs)
            context_zsa = self.jedi_latent_scale * prediction
        else:
            context_zs = self._soft_bound(self.jedi_latent_scale * prediction_zs)
            context_zsa = self._soft_bound(self.jedi_latent_scale * prediction)
        clean = self._soft_bound(self.jedi_latent_scale * prediction_next_zs).detach()

        log_sigma = self.jedi_p_mean + self.jedi_p_std * sigma_normal
        sigma = log_sigma.exp()
        sigma_data = self.jedi_sigma_data
        denominator = torch.sqrt(sigma.square() + sigma_data**2)
        c_in = denominator.reciprocal()
        c_skip = sigma_data**2 / denominator.square()
        c_out = sigma * sigma_data / denominator
        c_noise = log_sigma / 4.0

        noisy = clean + sigma * epsilon
        prediction_f = self.jedi_denoiser(c_in * noisy, c_noise, context_zs, context_zsa)
        # Algebraically identical to (clean - c_skip * noisy) / c_out, but stable as sigma -> 0.
        target_f = (sigma * clean / sigma_data - sigma_data * epsilon) / denominator
        return prediction_f, target_f, noisy, clean, c_skip, c_out, context_zs, context_zsa, sigma

    @torch.no_grad()
    def jedi_diagnostics(
        self,
        prediction_zs,
        prediction_action,
        prediction,
        prediction_next_zs,
        sigma_normal,
        epsilon,
    ):
        world_zs, world_prediction, world_next_zs = self._world_latents(
            prediction_zs, prediction_action, prediction, prediction_next_zs
        )
        (
            prediction_f,
            target_f,
            noisy,
            clean,
            c_skip,
            c_out,
            context_zs,
            context_zsa,
            sigma,
        ) = self._jedi_components(
            world_zs, world_prediction, world_next_zs, sigma_normal, epsilon
        )
        # The EDM F-target reconstructs the already-bounded clean latent exactly. Reapplying the
        # bound here would make even a perfect F predictor report nonzero reconstruction error.
        denoised = c_skip * noisy + c_out * prediction_f
        skipped = c_skip * noisy
        sample_denoised_mse = (denoised - clean).square().mean(dim=1)
        sample_skip_mse = (skipped - clean).square().mean(dim=1)
        high_noise = (sigma[:, 0] > 1.0).to(sample_denoised_mse.dtype)
        high_count = high_noise.sum().clamp_min(1.0)

        shuffled_action = prediction_action.roll(1, dims=0)
        if self.lewm_dynamics is not None:
            shuffled_prediction = self.lewm_pred_projector(
                self.lewm_dynamics(world_zs, shuffled_action)
            )
        else:
            shuffled_prediction = self._stock_prediction(
                prediction_zs, shuffled_action
            )
            if self.lewm_pred_projector is not None:
                shuffled_prediction = self.lewm_pred_projector(shuffled_prediction)
        if self.canonical_control_latents:
            shuffled_context_zsa = self.jedi_latent_scale * shuffled_prediction
        else:
            shuffled_context_zsa = self._soft_bound(
                self.jedi_latent_scale * shuffled_prediction
            )
        shuffled_f = self.jedi_denoiser(
            noisy / torch.sqrt(sigma.square() + self.jedi_sigma_data**2),
            sigma.log() / 4.0,
            context_zs,
            shuffled_context_zsa,
        )
        shuffled_loss = F.mse_loss(shuffled_f, target_f)
        jedi_loss = F.mse_loss(prediction_f, target_f)
        return {
            "loss": jedi_loss,
            "x0_mse": sample_denoised_mse.mean(),
            "skip_x0_mse": sample_skip_mse.mean(),
            "high_x0_mse": (sample_denoised_mse * high_noise).sum() / high_count,
            "high_skip_x0_mse": (sample_skip_mse * high_noise).sum() / high_count,
            "shuffled_context_loss_ratio": shuffled_loss / jedi_loss.clamp_min(1e-12),
            "context_action_spread": (prediction_f - shuffled_f).square().mean().sqrt(),
            "sigma_mean": sigma.mean(),
            "sigma_min": sigma.min(),
            "sigma_max": sigma.max(),
            "prediction_f_rms": prediction_f.square().mean().sqrt(),
            "target_f_rms": target_f.square().mean().sqrt(),
        }

    def forward(
        self,
        state,
        action,
        next_state,
        enc_state,
        enc_action,
        enc_next_state,
        enc_reward=None,
        policy_mean_target=None,
        policy_valid_target=None,
        jedi_sigma_normal=None,
        jedi_epsilon=None,
        jedi_weight=None,
        lewm_weight=None,
        reward_token_weight=None,
    ):
        if self.prediction_from_lap:
            prediction_zs = self.encoder.zs(state)
            prediction_next_zs = self.encoder.zs(next_state)
            prediction = self._stock_prediction(prediction_zs, action)
        else:
            prediction_zs = self.encoder.zs(enc_state)
            prediction_next_zs = self.encoder.zs(enc_next_state)
            prediction = self._stock_prediction(prediction_zs, enc_action)
        prediction_target = prediction_next_zs
        if self.canonical_control_latents:
            prediction_target = jedi_control_latent(prediction_target, self.jedi_clamp)
        if not self.attached_target:
            prediction_target = prediction_target.detach()
        prediction_loss = F.mse_loss(prediction, prediction_target)

        lewm_prediction_loss = prediction_loss.new_zeros(())
        full_sigreg_loss = prediction_loss.new_zeros(())
        reward_token_loss = prediction_loss.new_zeros(())
        reward_token_sigreg_loss = prediction_loss.new_zeros(())
        reward_token_mae = prediction_loss.new_zeros(())
        reward_token_std = prediction_loss.new_zeros(())
        policy_mean_loss = prediction_loss.new_zeros(())
        policy_mean_mae = prediction_loss.new_zeros(())
        policy_mean_std = prediction_loss.new_zeros(())
        outcome_reward_prediction_loss = prediction_loss.new_zeros(())
        outcome_reward_sigreg_loss = prediction_loss.new_zeros(())
        outcome_policy_prediction_loss = prediction_loss.new_zeros(())
        outcome_policy_sigreg_loss = prediction_loss.new_zeros(())
        outcome_reward_mae = prediction_loss.new_zeros(())
        outcome_policy_mae = prediction_loss.new_zeros(())
        outcome_reward_target_std = prediction_loss.new_zeros(())
        outcome_policy_target_std = prediction_loss.new_zeros(())
        outcome_reward_clipped_fraction = prediction_loss.new_zeros(())
        outcome_reward_pred_edge_mass = prediction_loss.new_zeros(())
        outcome_reward_semantic_std = prediction_loss.new_zeros(())
        outcome_policy_semantic_std = prediction_loss.new_zeros(())
        outcome_reward_semantic_loss = prediction_loss.new_zeros(())
        outcome_policy_semantic_loss = prediction_loss.new_zeros(())
        outcome_reward_reconstruction_mae = prediction_loss.new_zeros(())
        outcome_policy_mean_mae = prediction_loss.new_zeros(())
        outcome_policy_logstd_mae = prediction_loss.new_zeros(())
        world_zs, world_prediction, world_next_zs = self._world_latents(
            prediction_zs, enc_action, prediction, prediction_next_zs
        )
        if self.lewm_projector is not None:
            # LeWM intentionally keeps the future projector target attached. Full-dimensional
            # SIGReg and the separate projector space prevent this gradient from collapsing or
            # directly destabilizing TD7's control coordinate.
            lewm_prediction_loss = F.mse_loss(world_prediction, world_next_zs)
            sigreg_count = (
                world_zs.size(0)
                if self.sigreg_batch_size <= 0
                else self.sigreg_batch_size
            )
            full_sigreg_loss = self.full_sigreg(
                torch.stack(
                    [world_zs[:sigreg_count], world_next_zs[:sigreg_count]]
                )
            )
        if self.reward_token_head is not None:
            reward_token, predicted_symlog_reward = self.reward_token_head(
                prediction, self.reward_token_shared_scale
            )
            control_cost = self.reward_control_cost_coef * enc_action.square().sum(
                dim=1, keepdim=True
            )
            forward_reward = enc_reward + control_cost
            reward_token_loss = F.smooth_l1_loss(
                predicted_symlog_reward, symlog(forward_reward)
            )
            sigreg_token = reward_token
            if self.reward_sigreg_tokenizer_only:
                sigreg_token = self.reward_token_head.tokenize(prediction, 0.0)
            reward_token_sigreg_loss = self.reward_token_sigreg(sigreg_token)
            predicted_total_reward = symexp(predicted_symlog_reward) - control_cost
            reward_token_mae = (predicted_total_reward - enc_reward).abs().mean()
            reward_token_std = reward_token.std(unbiased=False)
        if self.policy_mean_head is not None:
            predicted_policy_mean = self.policy_mean_head(prediction_zs)
            policy_mean_loss = F.mse_loss(predicted_policy_mean, policy_mean_target)
            policy_mean_mae = (predicted_policy_mean - policy_mean_target).abs().mean()
            policy_mean_std = predicted_policy_mean.std(unbiased=False)
        if self.outcome_tokens is not None:
            outcome_outputs = self.outcome_tokens(
                prediction_zs,
                enc_action,
                enc_reward,
                policy_mean_target,
                world_prediction if self.outcome_from_transition else None,
            )
            if getattr(self.outcome_tokens, "isometric_targets", False):
                (
                    predicted_reward_token,
                    reward_target_token,
                    predicted_policy_token,
                    policy_target_token,
                    reward_distribution,
                    policy_semantic_target,
                ) = outcome_outputs
                if self.outcome_tokens.reward_hlgauss_ce:
                    outcome_reward_prediction_loss = -(
                        reward_target_token
                        * F.log_softmax(predicted_reward_token, dim=-1)
                    ).sum(dim=-1).mean()
                else:
                    outcome_reward_prediction_loss = F.mse_loss(
                        predicted_reward_token, reward_target_token
                    )
                if self.outcome_tokens.policy_beta_nll:
                    (
                        policy_element_nll,
                        _policy_beta_target,
                        policy_alpha,
                        policy_beta,
                    ) = self.outcome_tokens.policy_moment_beta_nll(
                        predicted_policy_token, policy_target_token
                    )
                    policy_latent_sample_loss = policy_element_nll.mean(
                        dim=-1, keepdim=True
                    )
                    predicted_policy_mean = (
                        2.0 * policy_alpha / (policy_alpha + policy_beta) - 1.0
                    )
                    outcome_policy_mae = (
                        (predicted_policy_mean - policy_target_token)
                        .abs()
                        .mean(dim=-1, keepdim=True)
                        * policy_valid_target
                    ).sum() / policy_valid_target.sum().clamp_min(1.0)
                    policy_target_count = (
                        policy_valid_target.sum() * policy_target_token.size(-1)
                    ).clamp_min(1.0)
                    policy_target_mean = (
                        policy_target_token * policy_valid_target
                    ).sum() / policy_target_count
                    outcome_policy_target_std = torch.sqrt(
                        (
                            (policy_target_token - policy_target_mean).square()
                            * policy_valid_target
                        ).sum()
                        / policy_target_count
                    )
                    predicted_policy_mean_average = (
                        predicted_policy_mean * policy_valid_target
                    ).sum() / policy_target_count
                    outcome_policy_semantic_std = torch.sqrt(
                        (
                            (
                                predicted_policy_mean
                                - predicted_policy_mean_average
                            ).square()
                            * policy_valid_target
                        ).sum()
                        / policy_target_count
                    )
                else:
                    policy_latent_sample_loss = F.mse_loss(
                        predicted_policy_token,
                        policy_target_token,
                        reduction="none",
                    ).mean(dim=-1, keepdim=True)
                outcome_policy_prediction_loss = (
                    policy_latent_sample_loss * policy_valid_target
                ).sum() / policy_valid_target.sum().clamp_min(1.0)
                if self.outcome_tokens.reward_hlgauss_ce:
                    decoded_reward = self.outcome_tokens.reward_to_scalar(
                        predicted_reward_token
                    )
                    outcome_reward_mae = (
                        decoded_reward - enc_reward.reshape(-1)
                    ).abs().mean()
                else:
                    outcome_reward_mae = (
                        predicted_reward_token - reward_target_token
                    ).abs().mean()
                if not self.outcome_tokens.policy_beta_nll:
                    outcome_policy_mae = (
                        (predicted_policy_token - policy_target_token)
                        .abs()
                        .mean(dim=-1, keepdim=True)
                        * policy_valid_target
                    ).sum() / policy_valid_target.sum().clamp_min(1.0)
                outcome_reward_target_std = (
                    enc_reward.std(unbiased=False)
                    if self.outcome_tokens.reward_hlgauss_ce
                    else reward_target_token.std(unbiased=False)
                )
                if not self.outcome_tokens.policy_beta_nll:
                    policy_target_count = (
                        policy_valid_target.sum() * policy_target_token.size(-1)
                    ).clamp_min(1.0)
                    policy_target_mean = (
                        policy_target_token * policy_valid_target
                    ).sum() / policy_target_count
                    outcome_policy_target_std = torch.sqrt(
                        (
                            (policy_target_token - policy_target_mean).square()
                            * policy_valid_target
                        ).sum()
                        / policy_target_count
                    )
                outcome_reward_semantic_std = predicted_reward_token.std(unbiased=False)
                if not self.outcome_tokens.policy_beta_nll:
                    outcome_policy_semantic_std = predicted_policy_token.std(
                        unbiased=False
                    )
                outcome_reward_clipped_fraction = (
                    (enc_reward < self.outcome_tokens.reward_raw_min)
                    | (enc_reward > self.outcome_tokens.reward_raw_max)
                ).to(enc_reward.dtype).mean()
                if self.outcome_tokens.reward_hlgauss_ce:
                    predicted_reward_probs = torch.softmax(
                        predicted_reward_token, dim=-1
                    )
                    outcome_reward_pred_edge_mass = (
                        predicted_reward_probs[:, 0]
                        + predicted_reward_probs[:, -1]
                    ).mean()
            elif getattr(self.outcome_tokens, "latent_targets", False):
                (
                    predicted_reward_token,
                    reward_target_token,
                    predicted_policy_token,
                    policy_target_token,
                    reward_reconstruction_logits,
                    reward_distribution,
                    policy_reconstruction,
                    policy_semantic_target,
                ) = outcome_outputs
                # These are the only world-to-outcome objectives. Targets stay attached so their
                # encoders co-adapt, while semantic reconstruction plus FullSIG prevents collapse.
                outcome_reward_prediction_loss = F.mse_loss(
                    predicted_reward_token, reward_target_token
                )
                policy_latent_sample_loss = F.mse_loss(
                    predicted_policy_token, policy_target_token, reduction="none"
                ).mean(dim=-1, keepdim=True)
                outcome_policy_prediction_loss = (
                    policy_latent_sample_loss * policy_valid_target
                ).sum() / policy_valid_target.sum().clamp_min(1.0)
                sigreg_count = (
                    reward_target_token.size(0)
                    if self.sigreg_batch_size <= 0
                    else self.sigreg_batch_size
                )
                outcome_reward_sigreg_loss = self.reward_outcome_sigreg(
                    reward_target_token[:sigreg_count].unsqueeze(0)
                )
                outcome_policy_sigreg_loss = self.policy_outcome_sigreg(
                    policy_target_token[:sigreg_count].unsqueeze(0),
                    policy_valid_target[:sigreg_count],
                )
                if self.outcome_sigreg_batch_normalized:
                    outcome_reward_sigreg_loss = outcome_reward_sigreg_loss / sigreg_count
                    valid_sigreg_count = policy_valid_target[:sigreg_count].sum().clamp_min(1.0)
                    outcome_policy_sigreg_loss = (
                        outcome_policy_sigreg_loss / valid_sigreg_count
                    )
                outcome_reward_semantic_loss = -(
                    reward_distribution
                    * F.log_softmax(reward_reconstruction_logits, dim=-1)
                ).sum(dim=-1).mean()
                policy_semantic_sample_loss = F.mse_loss(
                    policy_reconstruction, policy_semantic_target, reduction="none"
                ).mean(dim=-1, keepdim=True)
                outcome_policy_semantic_loss = (
                    policy_semantic_sample_loss * policy_valid_target
                ).sum() / policy_valid_target.sum().clamp_min(1.0)
                outcome_reward_mae = (
                    predicted_reward_token - reward_target_token
                ).abs().mean()
                outcome_policy_mae = (
                    (predicted_policy_token - policy_target_token)
                    .abs()
                    .mean(dim=-1, keepdim=True)
                    * policy_valid_target
                ).sum() / policy_valid_target.sum().clamp_min(1.0)
                outcome_reward_target_std = reward_target_token.std(unbiased=False)
                policy_target_count = (
                    policy_valid_target.sum() * policy_target_token.size(-1)
                ).clamp_min(1.0)
                policy_target_mean = (
                    policy_target_token * policy_valid_target
                ).sum() / policy_target_count
                outcome_policy_target_std = torch.sqrt(
                    (
                        (policy_target_token - policy_target_mean).square()
                        * policy_valid_target
                    ).sum()
                    / policy_target_count
                )
                outcome_reward_semantic_std = predicted_reward_token.std(unbiased=False)
                outcome_policy_semantic_std = predicted_policy_token.std(unbiased=False)
                decoded_reward = self.outcome_tokens.reward_to_scalar(
                    reward_reconstruction_logits
                )
                outcome_reward_reconstruction_mae = (
                    decoded_reward - enc_reward.reshape(-1)
                ).abs().mean()
                action_dim = self.outcome_tokens.action_dim
                policy_reconstruction_error = (
                    policy_reconstruction - policy_semantic_target
                ).abs()
                outcome_policy_mean_mae = (
                    policy_reconstruction_error[:, :action_dim].mean(dim=-1, keepdim=True)
                    * policy_valid_target
                ).sum() / policy_valid_target.sum().clamp_min(1.0)
                normalized_logstd_mae = (
                    policy_reconstruction_error[:, action_dim:].mean(dim=-1, keepdim=True)
                    * policy_valid_target
                ).sum() / policy_valid_target.sum().clamp_min(1.0)
                outcome_policy_logstd_mae = (
                    normalized_logstd_mae * self.policy_logstd_half_range
                )
                outcome_reward_clipped_fraction = (
                    (enc_reward < self.outcome_tokens.reward_raw_min)
                    | (enc_reward > self.outcome_tokens.reward_raw_max)
                ).to(enc_reward.dtype).mean()
                reward_reconstruction_probs = torch.softmax(
                    reward_reconstruction_logits, dim=-1
                )
                outcome_reward_pred_edge_mass = (
                    reward_reconstruction_probs[:, 0]
                    + reward_reconstruction_probs[:, -1]
                ).mean()
            elif getattr(self.outcome_tokens, "distributional", False):
                (
                    predicted_reward_token,
                    reward_target_token,
                    predicted_policy_token,
                    policy_target_token,
                    reward_semantic_token,
                    policy_semantic_token,
                ) = outcome_outputs
                outcome_reward_prediction_loss = -(
                    reward_target_token
                    * F.log_softmax(predicted_reward_token, dim=-1)
                ).sum(dim=-1).mean()
                outcome_policy_prediction_loss = F.mse_loss(
                    predicted_policy_token, policy_target_token, reduction="none"
                ).mean(dim=-1, keepdim=True)
                outcome_policy_prediction_loss = (
                    outcome_policy_prediction_loss * policy_valid_target
                ).sum() / policy_valid_target.sum().clamp_min(1.0)
                decoded_reward = self.outcome_tokens.reward_to_scalar(
                    predicted_reward_token
                )
                outcome_reward_mae = (
                    decoded_reward - enc_reward.reshape(-1)
                ).abs().mean()
                outcome_policy_mae = (
                    (predicted_policy_token - policy_target_token)
                    .abs()
                    .mean(dim=-1, keepdim=True)
                    * policy_valid_target
                ).sum() / policy_valid_target.sum().clamp_min(1.0)
                outcome_reward_target_std = enc_reward.std(unbiased=False)
                policy_target_count = (
                    policy_valid_target.sum() * policy_target_token.size(-1)
                ).clamp_min(1.0)
                policy_target_mean = (
                    policy_target_token * policy_valid_target
                ).sum() / policy_target_count
                outcome_policy_target_std = torch.sqrt(
                    (
                        (policy_target_token - policy_target_mean).square()
                        * policy_valid_target
                    ).sum()
                    / policy_target_count
                )
                outcome_reward_clipped_fraction = (
                    (enc_reward < self.outcome_tokens.reward_raw_min)
                    | (enc_reward > self.outcome_tokens.reward_raw_max)
                ).to(enc_reward.dtype).mean()
                predicted_reward_probs = torch.softmax(
                    predicted_reward_token, dim=-1
                )
                outcome_reward_pred_edge_mass = (
                    predicted_reward_probs[:, 0] + predicted_reward_probs[:, -1]
                ).mean()
                outcome_reward_semantic_std = reward_semantic_token.std(unbiased=False)
                outcome_policy_semantic_std = policy_semantic_token.std(unbiased=False)
            else:
                (
                    predicted_reward_token,
                    reward_target_token,
                    predicted_policy_token,
                    policy_target_token,
                ) = outcome_outputs
                # LeJEPA target embeddings remain attached. Their own SIGReg objectives replace an
                # EMA/stop-gradient teacher and prevent the joint predictor/target collapse.
                outcome_reward_prediction_loss = F.mse_loss(
                    predicted_reward_token, reward_target_token
                )
                outcome_policy_prediction_loss = F.mse_loss(
                    predicted_policy_token, policy_target_token
                )
                sigreg_count = (
                    reward_target_token.size(0)
                    if self.sigreg_batch_size <= 0
                    else self.sigreg_batch_size
                )
                outcome_reward_sigreg_loss = self.reward_outcome_sigreg(
                    reward_target_token[:sigreg_count]
                )
                outcome_policy_sigreg_loss = self.policy_outcome_sigreg(
                    policy_target_token[:sigreg_count].unsqueeze(0)
                )
                if self.outcome_sigreg_batch_normalized:
                    outcome_reward_sigreg_loss = (
                        outcome_reward_sigreg_loss / sigreg_count
                    )
                    outcome_policy_sigreg_loss = (
                        outcome_policy_sigreg_loss / sigreg_count
                    )
                outcome_reward_mae = (
                    predicted_reward_token - reward_target_token
                ).abs().mean()
                outcome_policy_mae = (
                    predicted_policy_token - policy_target_token
                ).abs().mean()
                outcome_reward_target_std = reward_target_token.std(unbiased=False)
                outcome_policy_target_std = policy_target_token.std(unbiased=False)

        if self.prediction_from_lap:
            sigreg_zs = self.encoder.zs(enc_state)
            sigreg_next_zs = self.encoder.zs(enc_next_state)
        else:
            sigreg_zs = prediction_zs
            sigreg_next_zs = prediction_next_zs

        sigreg_loss = prediction_loss.new_zeros(())
        if self.subsig is not None:
            requested_sigreg_count = self.control_sigreg_batch_size
            if requested_sigreg_count < 0:
                requested_sigreg_count = self.sigreg_batch_size
            sigreg_count = (
                sigreg_zs.size(0)
                if requested_sigreg_count <= 0
                else requested_sigreg_count
            )
            sigreg_input = torch.stack(
                [sigreg_zs[:sigreg_count], sigreg_next_zs[:sigreg_count]]
            ) * np.sqrt(2.0 / np.pi)
            sigreg_loss = self.subsig(sigreg_input)

        jedi_loss = prediction_loss.new_zeros(())
        if self.jedi_denoiser is not None:
            prediction_f, target_f, *_ = self._jedi_components(
                world_zs,
                world_prediction,
                world_next_zs,
                jedi_sigma_normal,
                jedi_epsilon,
            )
            jedi_loss = F.mse_loss(prediction_f, target_f)
        normalized_prediction_loss = prediction_loss
        if self.prediction_loss_normalizer is not None:
            normalized_prediction_loss = self.prediction_loss_normalizer(
                prediction_loss
            )
        encoder_loss = normalized_prediction_loss + self.subsig_coef * sigreg_loss
        if self.jedi_denoiser is not None:
            encoder_loss = encoder_loss + jedi_weight * jedi_loss
        if self.lewm_projector is not None:
            normalized_lewm_prediction_loss = lewm_prediction_loss
            if self.lewm_prediction_loss_normalizer is not None:
                normalized_lewm_prediction_loss = (
                    self.lewm_prediction_loss_normalizer(lewm_prediction_loss)
                )
            lewm_loss = (
                normalized_lewm_prediction_loss
                + self.lewm_sigreg_coef * full_sigreg_loss
            )
            encoder_loss = encoder_loss + lewm_weight * lewm_loss
        if self.reward_token_head is not None:
            reward_objective = (
                reward_token_loss
                + self.reward_token_sigreg_coef * reward_token_sigreg_loss
            )
            encoder_loss = encoder_loss + reward_token_weight * reward_objective
        if self.policy_mean_head is not None:
            encoder_loss = encoder_loss + self.policy_mean_coef * policy_mean_loss
        if self.outcome_tokens is not None:
            if getattr(self.outcome_tokens, "isometric_targets", False):
                reward_outcome_loss = outcome_reward_prediction_loss
                policy_outcome_loss = outcome_policy_prediction_loss
            elif getattr(self.outcome_tokens, "latent_targets", False):
                reward_outcome_loss = (
                    outcome_reward_prediction_loss
                    + self.outcome_sigreg_coef * outcome_reward_sigreg_loss
                    + self.outcome_semantic_coef * outcome_reward_semantic_loss
                )
                policy_outcome_loss = (
                    outcome_policy_prediction_loss
                    + self.outcome_sigreg_coef * outcome_policy_sigreg_loss
                    + self.outcome_semantic_coef * outcome_policy_semantic_loss
                )
            elif getattr(self.outcome_tokens, "distributional", False):
                reward_outcome_loss = outcome_reward_prediction_loss
                policy_outcome_loss = outcome_policy_prediction_loss
            else:
                reward_outcome_loss = (
                    outcome_reward_prediction_loss
                    + self.outcome_sigreg_coef * outcome_reward_sigreg_loss
                )
                policy_outcome_loss = (
                    outcome_policy_prediction_loss
                    + self.outcome_sigreg_coef * outcome_policy_sigreg_loss
                )
            policy_outcome_coef = (
                self.policy_beta_nll_coef
                if getattr(self.outcome_tokens, "policy_beta_nll", False)
                else self.outcome_token_coef
            )
            normalized_reward_outcome_loss = reward_outcome_loss
            normalized_policy_outcome_loss = policy_outcome_loss
            if self.reward_outcome_loss_normalizer is not None:
                normalized_reward_outcome_loss = (
                    self.reward_outcome_loss_normalizer(reward_outcome_loss)
                )
                normalized_policy_outcome_loss = (
                    self.policy_outcome_loss_normalizer(policy_outcome_loss)
                )
            encoder_loss = (
                encoder_loss
                + self.outcome_token_coef * normalized_reward_outcome_loss
                + policy_outcome_coef * normalized_policy_outcome_loss
            )
        return (
            encoder_loss,
            prediction_loss,
            sigreg_loss,
            jedi_loss,
            lewm_prediction_loss,
            full_sigreg_loss,
            reward_token_loss,
            reward_token_sigreg_loss,
            reward_token_mae,
            reward_token_std,
            policy_mean_loss,
            policy_mean_mae,
            policy_mean_std,
            outcome_reward_prediction_loss,
            outcome_reward_sigreg_loss,
            outcome_policy_prediction_loss,
            outcome_policy_sigreg_loss,
            outcome_reward_mae,
            outcome_policy_mae,
            outcome_reward_target_std,
            outcome_policy_target_std,
            outcome_reward_clipped_fraction,
            outcome_reward_pred_edge_mass,
            outcome_reward_semantic_std,
            outcome_policy_semantic_std,
            outcome_reward_semantic_loss,
            outcome_policy_semantic_loss,
            outcome_reward_reconstruction_mae,
            outcome_policy_mean_mae,
            outcome_policy_logstd_mae,
        )


class LeWMRolloutLossCore(nn.Module):
    """Three-step attached LeWM dynamics plus JEDI in a gradient-firewalled branch."""

    def __init__(self, encoder, projector, dynamics, pred_projector, full_sigreg, denoiser, args):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.dynamics = dynamics
        self.pred_projector = pred_projector
        self.full_sigreg = full_sigreg
        self.denoiser = denoiser
        self.horizon = args.lewm_rollout_horizon
        self.jedi_p_mean = args.jedi_p_mean
        self.jedi_p_std = args.jedi_p_std
        self.jedi_sigma_data = args.jedi_sigma_data
        self.jedi_clamp = args.jedi_clamp
        self.sigreg_coef = args.lewm_sigreg_coef

    def _bound(self, value):
        return self.jedi_clamp * torch.tanh(value / self.jedi_clamp)

    def _embed_sequence(self, states, trunk_scale):
        batch, length, state_dim = states.shape
        features = self.encoder.state_features(states.reshape(batch * length, state_dim))
        wm_features = features.detach() + trunk_scale * (features - features.detach())
        return self.projector(wm_features).reshape(batch, length, -1)

    def _rollout(self, real_latents, actions):
        prediction = real_latents[:, 0]
        predictions = []
        for index in range(self.horizon):
            prediction = self.pred_projector(
                self.dynamics(prediction, actions[:, index])
            )
            predictions.append(prediction)
        return torch.stack(predictions, dim=1)

    def _jedi_loss(self, context_zs, context_prediction, clean, sigma_normal, epsilon):
        context_zs = self._bound(context_zs)
        context_prediction = self._bound(context_prediction)
        clean = self._bound(clean).detach()
        log_sigma = self.jedi_p_mean + self.jedi_p_std * sigma_normal
        sigma = log_sigma.exp()
        denominator = torch.sqrt(sigma.square() + self.jedi_sigma_data**2)
        c_in = denominator.reciprocal()
        c_skip = self.jedi_sigma_data**2 / denominator.square()
        c_out = sigma * self.jedi_sigma_data / denominator
        noisy = clean + sigma * epsilon
        prediction_f = self.denoiser(
            c_in * noisy,
            log_sigma / 4.0,
            context_zs,
            context_prediction,
        )
        target_f = (
            sigma * clean / self.jedi_sigma_data - self.jedi_sigma_data * epsilon
        ) / denominator
        denoised = c_skip * noisy + c_out * prediction_f
        return F.mse_loss(prediction_f, target_f), F.mse_loss(denoised, clean)

    def forward(self, states, actions, sigma_normal, epsilon, trunk_scale):
        real_latents = self._embed_sequence(states, trunk_scale)
        predictions = self._rollout(real_latents, actions)
        horizon_mse = (predictions - real_latents[:, 1:]).square().mean(dim=(0, 2))
        weights = horizon_mse.new_tensor([0.5**index for index in range(self.horizon)])
        rollout_loss = (horizon_mse * weights).sum() / weights.sum()
        full_sigreg_loss = self.full_sigreg(real_latents.transpose(0, 1))
        jedi_loss, jedi_x0_mse = self._jedi_loss(
            real_latents[:, 0], predictions[:, 0], real_latents[:, 1], sigma_normal, epsilon
        )
        return rollout_loss, full_sigreg_loss, jedi_loss, jedi_x0_mse, horizon_mse

    @torch.no_grad()
    def diagnostics(self, states, actions, sigma_normal, epsilon, trunk_scale):
        real_latents = self._embed_sequence(states, trunk_scale)
        predictions = self._rollout(real_latents, actions)
        horizon_mse = (predictions - real_latents[:, 1:]).square().mean(dim=(0, 2))
        copy_mse = (real_latents[:, :1] - real_latents[:, 1:]).square().mean(dim=(0, 2))
        shuffled_actions = actions.roll(1, dims=0)
        shuffled_predictions = self._rollout(real_latents, shuffled_actions)
        shuffled_mse = (shuffled_predictions - real_latents[:, 1:]).square().mean()
        correct_mse = (predictions - real_latents[:, 1:]).square().mean()
        jedi_loss, jedi_x0_mse = self._jedi_loss(
            real_latents[:, 0], predictions[:, 0], real_latents[:, 1], sigma_normal, epsilon
        )
        return {
            "rollout_mse": horizon_mse,
            "copy_mse": copy_mse,
            "shuffled_action_ratio": shuffled_mse / correct_mse.clamp_min(1e-12),
            "jedi_loss": jedi_loss,
            "jedi_x0_mse": jedi_x0_mse,
            "latent_rms": real_latents.square().mean().sqrt(),
            "rollout_rms": predictions.square().mean().sqrt(),
        }


class CriticLossCore(nn.Module):
    """Static TD7 target and twin-critic loss with tensor-valued mutable inputs."""

    def __init__(
        self,
        critic,
        critic_target,
        actor_target,
        fixed_encoder,
        fixed_encoder_target,
        args,
        endpoint_sampler=None,
        endpoint_sampler_target=None,
        control_mix=None,
        control_mix_target=None,
    ):
        super().__init__()
        self.critic = critic
        self.critic_target = critic_target
        self.actor_target = actor_target
        self.fixed_encoder = fixed_encoder
        self.fixed_encoder_target = fixed_encoder_target
        self.endpoint_sampler = endpoint_sampler
        self.endpoint_sampler_target = endpoint_sampler_target
        if endpoint_sampler is not None:
            self.register_buffer("control_mix", control_mix)
            self.register_buffer("control_mix_target", control_mix_target)
        self.gamma = args.gamma
        self.min_priority = args.min_priority
        self.lap_alpha = args.lap_alpha
        self.canonical_control_latents = args.jedi_canonical_control_latents
        self.jedi_clamp = args.jedi_clamp
        self.sac_policy = args.sac_policy
        self.hl_gauss_critic = args.hl_gauss_critic

    def _control_zs(self, raw_zs):
        if self.canonical_control_latents:
            return jedi_control_latent(raw_zs, self.jedi_clamp)
        return raw_zs

    def _stock_prediction(self, encoder, control_zs, action):
        prediction = encoder.zsa(control_zs, action)
        if self.canonical_control_latents:
            prediction = jedi_control_latent(prediction, self.jedi_clamp)
        return prediction

    def forward(
        self, state, action, next_state, reward, not_done, noise, alpha, min_target, max_target
    ):
        with torch.no_grad():
            fixed_target_raw_zs = self.fixed_encoder_target.zs(next_state)
            fixed_target_control_zs = self._control_zs(fixed_target_raw_zs)
            next_log_pi = None
            if self.sac_policy:
                next_action, next_log_pi, _ = self.actor_target.sample(
                    next_state, fixed_target_control_zs, noise
                )
            else:
                next_action = (
                    self.actor_target(next_state, fixed_target_control_zs) + noise
                ).clamp(-1, 1)
            fixed_target_stock_zsa = self._stock_prediction(
                self.fixed_encoder_target, fixed_target_control_zs, next_action
            )
            fixed_target_zsa = fixed_target_stock_zsa
            if self.endpoint_sampler_target is not None:
                fixed_target_direct_zsa = self.endpoint_sampler_target(
                    fixed_target_control_zs, fixed_target_stock_zsa
                )
                fixed_target_zsa = torch.lerp(
                    fixed_target_stock_zsa,
                    fixed_target_direct_zsa,
                    self.control_mix_target,
                )
            q_target = self.critic_target(
                next_state, next_action, fixed_target_zsa, fixed_target_control_zs
            ).min(1, keepdim=True)[0]
            if self.sac_policy:
                q_target = q_target - alpha * next_log_pi
            q_target = reward + not_done * self.gamma * q_target.clamp(min_target, max_target)
            fixed_raw_zs = self.fixed_encoder.zs(state)
            fixed_control_zs = self._control_zs(fixed_raw_zs)
            fixed_stock_zsa = self._stock_prediction(
                self.fixed_encoder, fixed_control_zs, action
            )
            fixed_zsa = fixed_stock_zsa
            if self.endpoint_sampler is not None:
                fixed_direct_zsa = self.endpoint_sampler(fixed_control_zs, fixed_stock_zsa)
                fixed_zsa = torch.lerp(fixed_stock_zsa, fixed_direct_zsa, self.control_mix)

        critic_edge_mass = state.new_zeros(())
        if self.hl_gauss_critic:
            q_logits = self.critic.logits(
                state, action, fixed_zsa, fixed_control_zs
            )
            q = self.critic.decode_logits(q_logits)
            target_probabilities = self.critic.project(q_target)
            critic_loss = -(
                target_probabilities.unsqueeze(1)
                * torch.log_softmax(q_logits, dim=-1)
            ).sum(dim=-1).sum(dim=1).mean()
            critic_edge_mass = self.critic.edge_mass(q_logits)
        else:
            q = self.critic(state, action, fixed_zsa, fixed_control_zs)
        td_loss = (q - q_target).abs()
        if not self.hl_gauss_critic:
            critic_loss = lap_huber(td_loss, self.min_priority)
        priority = td_loss.max(1)[0].clamp(min=self.min_priority).pow(self.lap_alpha).detach()
        return (
            critic_loss,
            priority,
            q_target.amin(),
            q_target.amax(),
            q.mean(),
            q.amin(),
            q.amax(),
            fixed_control_zs,
            critic_edge_mass,
        )


class ActorLossCore(nn.Module):
    """Static deterministic-policy loss; critic parameters are frozen by the caller."""

    def __init__(
        self,
        actor,
        critic,
        fixed_encoder,
        args,
        endpoint_sampler=None,
        control_mix=None,
    ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.fixed_encoder = fixed_encoder
        self.endpoint_sampler = endpoint_sampler
        self.canonical_control_latents = args.jedi_canonical_control_latents
        self.exact_endpoint_gradients = args.jedi_exact_actor_gradients
        self.jedi_clamp = args.jedi_clamp
        self.sac_policy = args.sac_policy
        self.sd_noise = args.sd_noise
        if endpoint_sampler is not None:
            self.register_buffer("control_mix", control_mix)

    def forward(self, state, fixed_control_zs, policy_noise, alpha):
        log_pi = state.new_zeros((state.size(0), 1))
        if self.sac_policy:
            actor_action, log_pi, _ = self.actor.sample(
                state, fixed_control_zs, policy_noise
            )
        elif self.sd_noise:
            actor_action, log_pi = self.actor.sample_additive(
                state, fixed_control_zs, policy_noise
            )
        else:
            actor_action = self.actor(state, fixed_control_zs)
        stock_zsa = self.fixed_encoder.zsa(fixed_control_zs, actor_action)
        if self.canonical_control_latents:
            stock_zsa = jedi_control_latent(stock_zsa, self.jedi_clamp)
        actor_fixed_zsa = stock_zsa
        if self.endpoint_sampler is not None:
            if self.exact_endpoint_gradients:
                direct_zsa = self.endpoint_sampler(fixed_control_zs, stock_zsa)
                actor_fixed_zsa = torch.lerp(stock_zsa, direct_zsa, self.control_mix)
            else:
                # Legacy v5 interface: exact endpoint forward, SALE action Jacobian backward.
                with torch.no_grad():
                    direct_zsa = self.endpoint_sampler(fixed_control_zs, stock_zsa)
                    forward_zsa = torch.lerp(stock_zsa, direct_zsa, self.control_mix)
                actor_fixed_zsa = forward_zsa + stock_zsa - stock_zsa.detach()
        q_values = self.critic(
            state, actor_action, actor_fixed_zsa, fixed_control_zs
        )
        if self.sac_policy:
            actor_loss = (alpha * log_pi - q_values.min(1, keepdim=True)[0]).mean()
        elif self.sd_noise:
            actor_loss = -q_values.mean() - alpha * log_pi.mean()
        else:
            actor_loss = -q_values.mean()
        return actor_loss, log_pi.mean()


class LeSALEAgent:
    def __init__(self, state_dim, action_dim, max_action, args: Args, device, writer: SummaryWriter):
        self.args = args
        self.device = device
        self.writer = writer
        self.canonical_control_latents = args.jedi_canonical_control_latents

        if args.sac_policy:
            self.actor = SACActor(
                state_dim,
                action_dim,
                args.zs_dim,
                args.hidden_dim,
                args.sac_log_std_min,
                args.sac_log_std_max,
            ).to(device)
        elif args.sd_noise:
            self.actor = SDNoiseActor(
                state_dim,
                action_dim,
                args.zs_dim,
                args.hidden_dim,
                args.sd_log_std_min,
                args.sd_log_std_max,
                args.seed + 7101,
            ).to(device)
        else:
            self.actor = Actor(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.pc_actor_trainer = None
        if args.pc_actor:
            for parameter in self.actor.parameters():
                parameter.requires_grad_(False)
            self.actor_optimizer = None
            self.pc_actor_trainer = TD7PCActorTrainer(self.actor, args)
        else:
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=args.actor_lr, fused=args.fused_adam
            )
        self.actor_target = freeze(copy.deepcopy(self.actor))
        self.target_entropy = -float(action_dim)
        self.log_alpha = None
        self.alpha_optimizer = None
        if args.sac_policy and args.sac_autotune:
            self.log_alpha = torch.zeros((), requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=args.sac_alpha_lr, fused=args.fused_adam
            )
            self.alpha_tensor = torch.ones((), device=device)
        elif args.sd_noise and args.sd_alpha_autotune:
            self.target_entropy = action_dim * float(np.log(args.sd_target_sigma))
            self.log_alpha = torch.zeros((), requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=args.sd_alpha_lr, fused=args.fused_adam
            )
            self.alpha_tensor = torch.ones((), device=device)
        elif args.sd_noise:
            self.alpha_tensor = torch.tensor(args.sd_alpha, device=device)
        else:
            self.alpha_tensor = torch.tensor(args.sac_alpha, device=device)

        critic_cls = HLGaussCritic if args.hl_gauss_critic else Critic
        if args.hl_gauss_critic:
            self.critic = critic_cls(
                state_dim,
                action_dim,
                args.zs_dim,
                args.hidden_dim,
                args.hl_gauss_num_bins,
                args.hl_gauss_v_min,
                args.hl_gauss_v_max,
                args.hl_gauss_sigma_ratio,
            ).to(device)
        else:
            self.critic = critic_cls(
                state_dim, action_dim, args.zs_dim, args.hidden_dim
            ).to(device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr, fused=args.fused_adam
        )
        self.critic_target = freeze(copy.deepcopy(self.critic))

        encoder_cls = ResidualEncoder if args.residual_predictor else StockEncoder
        self.encoder = encoder_cls(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=args.encoder_lr, fused=args.fused_adam
        )
        self.fixed_encoder = freeze(copy.deepcopy(self.encoder))
        self.fixed_encoder_target = freeze(copy.deepcopy(self.encoder))

        self.checkpoint_actor = freeze(copy.deepcopy(self.actor))
        self.checkpoint_encoder = freeze(copy.deepcopy(self.encoder))

        self.replay_buffer = UniformLAPBuffer(
            state_dim,
            action_dim,
            device,
            args.buffer_size,
            args.batch_size,
            max_action,
            gpu_storage=args.gpu_replay,
        )

        self.subsig = None
        if args.use_full_obs_sigreg:
            self.subsig = FullSIGReg(
                args.zs_dim,
                args.lewm_sigreg_num_proj,
                args.subsig_knots,
                device,
                args.seed + 1701,
            )
        elif args.use_subsig:
            self.subsig = SubspaceSIGReg(
                args.zs_dim,
                args.subsig_subspaces,
                args.subsig_dim,
                args.subsig_num_proj,
                args.subsig_knots,
                device,
                args.seed + 1701,
            )

        self.lewm_projector = None
        self.lewm_dynamics = None
        self.lewm_pred_projector = None
        self.full_sigreg = None
        self.lewm_weight_tensor = None
        self.lewm_trunk_scale_tensor = None
        self.rollout_projector = None
        self.rollout_dynamics = None
        self.rollout_pred_projector = None
        self.rollout_full_sigreg = None
        self.wm_optimizer = None
        self.reward_token_head = None
        self.reward_token_sigreg = None
        self.reward_token_optimizer = None
        self.reward_token_weight_tensor = None
        self.policy_mean_head = None
        self.policy_mean_optimizer = None
        self.outcome_tokens = None
        self.reward_outcome_sigreg = None
        self.policy_outcome_sigreg = None
        self.outcome_optimizer = None
        if args.lewm_projected_aux:
            fork_devices = []
            if device.type == "cuda":
                fork_devices = [
                    device.index if device.index is not None else torch.cuda.current_device()
                ]
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(args.seed + 5201)
                self.lewm_projector = LeWMProjectionMLP(
                    args.zs_dim, args.lewm_hidden_dim
                ).to(device)
                if args.lewm_private_dynamics:
                    self.lewm_dynamics = LeWMResidualDynamics(
                        args.zs_dim,
                        action_dim,
                        args.lewm_hidden_dim,
                    ).to(device)
                self.lewm_pred_projector = LeWMProjectionMLP(
                    args.zs_dim, args.lewm_hidden_dim
                ).to(device)
            self.full_sigreg = FullSIGReg(
                args.zs_dim,
                args.lewm_sigreg_num_proj,
                args.subsig_knots,
                device,
                args.seed + 5202,
            )
            self.lewm_weight_tensor = torch.zeros((), device=device)
            self.encoder_optimizer = torch.optim.AdamW(
                [
                    {"params": self.encoder.parameters(), "weight_decay": 0.0},
                    {"params": self.lewm_projector.parameters(), "weight_decay": 1e-3},
                    *(
                        [{"params": self.lewm_dynamics.parameters(), "weight_decay": 1e-3}]
                        if self.lewm_dynamics is not None
                        else []
                    ),
                    {"params": self.lewm_pred_projector.parameters(), "weight_decay": 1e-3},
                ],
                lr=args.encoder_lr,
                fused=args.fused_adam,
            )
        if args.lewm_rollout_aux:
            fork_devices = []
            if device.type == "cuda":
                fork_devices = [
                    device.index if device.index is not None else torch.cuda.current_device()
                ]
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(args.seed + 5301)
                self.rollout_projector = LeWMRolloutProjector(
                    args.hidden_dim, args.lewm_rollout_dim, args.lewm_hidden_dim
                ).to(device)
                self.rollout_dynamics = LeWMResidualDynamics(
                    args.lewm_rollout_dim, action_dim, args.lewm_hidden_dim
                ).to(device)
                self.rollout_pred_projector = LeWMResidualProjection(
                    args.lewm_rollout_dim, args.lewm_hidden_dim
                ).to(device)
            self.rollout_full_sigreg = FullSIGReg(
                args.lewm_rollout_dim,
                args.lewm_sigreg_num_proj,
                args.subsig_knots,
                device,
                args.seed + 5302,
            )
            self.lewm_weight_tensor = torch.zeros((), device=device)
            self.lewm_trunk_scale_tensor = torch.ones((), device=device)
            self.lewm_stock_grad_ema = torch.zeros((), device=device)
            self.lewm_aux_grad_ema = torch.zeros((), device=device)
            self.wm_optimizer = torch.optim.AdamW(
                [
                    *self.rollout_projector.parameters(),
                    *self.rollout_dynamics.parameters(),
                    *self.rollout_pred_projector.parameters(),
                ],
                lr=args.encoder_lr,
                weight_decay=1e-3,
                fused=args.fused_adam,
            )
        if args.reward_token_aux:
            fork_devices = []
            if device.type == "cuda":
                fork_devices = [
                    device.index if device.index is not None else torch.cuda.current_device()
                ]
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(args.seed + 6101)
                self.reward_token_head = RewardTokenHead(
                    args.zs_dim, action_dim, args.hidden_dim
                ).to(device)
            self.reward_token_sigreg = ScalarSIGReg(args.subsig_knots, device)
            self.reward_token_optimizer = torch.optim.AdamW(
                self.reward_token_head.parameters(),
                lr=args.encoder_lr,
                weight_decay=1e-3,
                fused=args.fused_adam,
            )
            self.reward_token_weight_tensor = torch.zeros((), device=device)
        if args.policy_mean_aux:
            fork_devices = []
            if device.type == "cuda":
                fork_devices = [
                    device.index if device.index is not None else torch.cuda.current_device()
                ]
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(args.seed + 6201)
                self.policy_mean_head = PolicyMeanHead(
                    args.zs_dim, action_dim, args.hidden_dim
                ).to(device)
            self.policy_mean_optimizer = torch.optim.AdamW(
                self.policy_mean_head.parameters(),
                lr=args.encoder_lr,
                weight_decay=1e-3,
                fused=args.fused_adam,
            )
        if args.lejepa_outcome_tokens:
            policy_token_dim = (
                2 * action_dim
                if args.sac_policy and args.outcome_policy_include_log_std
                else action_dim
            )
            fork_devices = []
            if device.type == "cuda":
                fork_devices = [
                    device.index if device.index is not None else torch.cuda.current_device()
                ]
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(args.seed + 6301)
                self.outcome_tokens = LeJEPAOutcomeTokens(
                    args.zs_dim,
                    action_dim,
                    policy_token_dim,
                    args.hidden_dim,
                    args.subsig_knots,
                    args.outcome_from_transition,
                ).to(device)
            self.reward_outcome_sigreg = ScalarSIGReg(args.subsig_knots, device)
            self.policy_outcome_sigreg = FullSIGReg(
                policy_token_dim,
                args.outcome_policy_sigreg_num_proj,
                args.subsig_knots,
                device,
                args.seed + 6302,
            )
            self.outcome_optimizer = torch.optim.AdamW(
                self.outcome_tokens.parameters(),
                lr=args.encoder_lr,
                weight_decay=1e-3,
                fused=args.fused_adam,
            )
        elif args.semantic_outcome_tokens:
            fork_devices = []
            if device.type == "cuda":
                fork_devices = [
                    device.index if device.index is not None else torch.cuda.current_device()
                ]
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(args.seed + 6401)
                self.outcome_tokens = SemanticOutcomeTokens(
                    args.zs_dim,
                    action_dim,
                    args.semantic_outcome_token_dim,
                    args.semantic_reward_num_bins,
                    args.semantic_reward_raw_min,
                    args.semantic_reward_raw_max,
                    args.semantic_reward_sigma_ratio,
                    args.semantic_reward_prior_floor,
                ).to(device)
            self.outcome_optimizer = torch.optim.AdamW(
                self.outcome_tokens.parameters(),
                lr=args.encoder_lr,
                weight_decay=1e-3,
                fused=args.fused_adam,
            )
        elif args.latent_outcome_tokens:
            fork_devices = []
            if device.type == "cuda":
                fork_devices = [
                    device.index if device.index is not None else torch.cuda.current_device()
                ]
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(args.seed + 6501)
                self.outcome_tokens = LatentOutcomeTokens(
                    args.zs_dim,
                    action_dim,
                    args.semantic_outcome_token_dim,
                    args.semantic_reward_num_bins,
                    args.semantic_reward_raw_min,
                    args.semantic_reward_raw_max,
                    args.semantic_reward_sigma_ratio,
                    args.semantic_reward_prior_floor,
                ).to(device)
            self.reward_outcome_sigreg = FullSIGReg(
                args.semantic_outcome_token_dim,
                args.outcome_policy_sigreg_num_proj,
                args.subsig_knots,
                device,
                args.seed + 6502,
            )
            self.policy_outcome_sigreg = FullSIGReg(
                args.semantic_outcome_token_dim,
                args.outcome_policy_sigreg_num_proj,
                args.subsig_knots,
                device,
                args.seed + 6503,
            )
            self.outcome_optimizer = torch.optim.AdamW(
                self.outcome_tokens.parameters(),
                lr=args.encoder_lr,
                weight_decay=1e-3,
                fused=args.fused_adam,
            )
        elif args.isometric_outcome_tokens:
            fork_devices = []
            if device.type == "cuda":
                fork_devices = [
                    device.index if device.index is not None else torch.cuda.current_device()
                ]
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(args.seed + 6601)
                self.outcome_tokens = IsometricOutcomeTokens(
                    args.zs_dim,
                    action_dim,
                    args.semantic_outcome_token_dim,
                    args.semantic_reward_num_bins,
                    args.semantic_reward_raw_min,
                    args.semantic_reward_raw_max,
                    args.semantic_reward_sigma_ratio,
                    args.policy_beta_nll,
                    args.policy_beta_nll_eps,
                    args.policy_beta_max_precision,
                    args.reward_hlgauss_ce,
                    args.reward_hlgauss_symlog,
                ).to(device)
            target_encoder_parameters = []
            if not args.reward_hlgauss_ce:
                target_encoder_parameters.extend(
                    self.outcome_tokens.reward_tokenizer.parameters()
                )
            if not args.policy_beta_nll:
                target_encoder_parameters.extend(
                    self.outcome_tokens.policy_tokenizer.parameters()
                )
            predictor_parameters = [
                *self.outcome_tokens.reward_predictor.parameters(),
                *self.outcome_tokens.reward_action_proj.parameters(),
                *self.outcome_tokens.policy_predictor.parameters(),
            ]
            if args.reward_hlgauss_ce:
                predictor_parameters.extend(
                    self.outcome_tokens.reward_readout.parameters()
                )
            optimizer_groups = [
                {
                    "params": predictor_parameters,
                    "weight_decay": 1e-3,
                }
            ]
            if target_encoder_parameters:
                optimizer_groups.insert(
                    0,
                    {
                        "params": target_encoder_parameters,
                        "weight_decay": 0.0,
                    },
                )
            self.outcome_optimizer = torch.optim.AdamW(
                optimizer_groups,
                lr=args.encoder_lr,
                fused=args.fused_adam,
            )
        self.encoder_trainable_parameters = list(self.encoder.parameters())
        if self.lewm_projector is not None:
            self.encoder_trainable_parameters.extend(self.lewm_projector.parameters())
            if self.lewm_dynamics is not None:
                self.encoder_trainable_parameters.extend(self.lewm_dynamics.parameters())
            self.encoder_trainable_parameters.extend(self.lewm_pred_projector.parameters())

        self.jedi_denoiser = None
        self.jedi_optimizer = None
        self.jedi_generator = None
        self.jedi_weight_tensor = None
        if args.jedi_aux:
            # Denoiser construction must not advance the baseline CUDA/CPU streams: doing so would
            # change behavior noise and target smoothing before the auxiliary loss has any effect.
            fork_devices = []
            if device.type == "cuda":
                fork_devices = [
                    device.index if device.index is not None else torch.cuda.current_device()
                ]
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(args.seed + 4301)
                self.jedi_denoiser = JEDIVectorDenoiser(
                    args.lewm_rollout_dim if args.lewm_rollout_aux else args.zs_dim,
                    args.hidden_dim,
                    args.jedi_time_dim,
                    args.jedi_blocks,
                ).to(device)
            self.jedi_optimizer = torch.optim.AdamW(
                self.jedi_denoiser.parameters(),
                lr=args.jedi_lr,
                weight_decay=args.jedi_weight_decay,
                fused=args.fused_adam,
            )
            self.jedi_generator = torch.Generator(device=device)
            self.jedi_generator.manual_seed(args.seed + 4302)
            self.jedi_weight_tensor = torch.zeros((), device=device)

        self.fixed_jedi_denoiser = None
        self.fixed_jedi_denoiser_target = None
        self.fixed_jedi_sampler = None
        self.fixed_jedi_sampler_target = None
        self.jedi_endpoint_prior = None
        self.jedi_endpoint_diag_priors = None
        self.jedi_control_mix_tensor = None
        self.fixed_jedi_control_mix = None
        self.fixed_jedi_control_mix_target = None
        if args.jedi_endpoint_control:
            self.fixed_jedi_denoiser = freeze(copy.deepcopy(self.jedi_denoiser))
            self.fixed_jedi_denoiser_target = freeze(copy.deepcopy(self.jedi_denoiser))
            endpoint_generator = torch.Generator(device=device)
            endpoint_generator.manual_seed(args.seed + 4303)
            self.jedi_endpoint_prior = torch.randn(
                (1, args.zs_dim), device=device, generator=endpoint_generator
            )
            self.jedi_endpoint_diag_priors = torch.randn(
                (args.jedi_endpoint_diag_priors, args.zs_dim),
                device=device,
                generator=endpoint_generator,
            )
            self.fixed_jedi_sampler = freeze(
                JEDIEndpointSampler(
                    self.fixed_jedi_denoiser,
                    self.jedi_endpoint_prior,
                    args.jedi_sigma_data,
                    args.jedi_clamp,
                    args.jedi_canonical_control_latents,
                ).to(device)
            )
            self.fixed_jedi_sampler_target = freeze(
                JEDIEndpointSampler(
                    self.fixed_jedi_denoiser_target,
                    self.jedi_endpoint_prior,
                    args.jedi_sigma_data,
                    args.jedi_clamp,
                    args.jedi_canonical_control_latents,
                ).to(device)
            )
            self.jedi_control_mix_tensor = torch.zeros((), device=device)
            self.fixed_jedi_control_mix = torch.zeros((), device=device)
            self.fixed_jedi_control_mix_target = torch.zeros((), device=device)

        self.encoder_loss_core = EncoderLossCore(
            self.encoder,
            self.subsig,
            None if args.lewm_rollout_aux else self.jedi_denoiser,
            args,
            self.lewm_projector,
            self.lewm_dynamics,
            self.lewm_pred_projector,
            self.full_sigreg,
            self.reward_token_head,
            self.reward_token_sigreg,
            self.policy_mean_head,
            self.outcome_tokens,
            self.reward_outcome_sigreg,
            self.policy_outcome_sigreg,
        ).to(device)
        self.encoder_loss_eager = self.encoder_loss_core
        self.rollout_loss_core = None
        self.rollout_loss_eager = None
        if args.lewm_rollout_aux:
            self.rollout_loss_core = LeWMRolloutLossCore(
                self.encoder,
                self.rollout_projector,
                self.rollout_dynamics,
                self.rollout_pred_projector,
                self.rollout_full_sigreg,
                self.jedi_denoiser,
                args,
            )
            self.rollout_loss_eager = self.rollout_loss_core
        self.critic_loss_core = CriticLossCore(
            self.critic,
            self.critic_target,
            self.actor if args.sac_policy else self.actor_target,
            self.fixed_encoder,
            self.fixed_encoder_target,
            args,
            self.fixed_jedi_sampler,
            self.fixed_jedi_sampler_target,
            self.fixed_jedi_control_mix,
            self.fixed_jedi_control_mix_target,
        )
        self.actor_loss_core = ActorLossCore(
            self.actor,
            self.critic,
            self.fixed_encoder,
            args,
            self.fixed_jedi_sampler,
            self.fixed_jedi_control_mix,
        )
        if args.torch_compile:
            compile_kwargs = {"dynamic": False, "fullgraph": True}
            if args.compile_mode != "default":
                compile_kwargs["mode"] = args.compile_mode
            self.encoder_loss_core = torch.compile(self.encoder_loss_core, **compile_kwargs)
            if self.rollout_loss_core is not None:
                self.rollout_loss_core = torch.compile(
                    self.rollout_loss_core, **compile_kwargs
                )
            self.critic_loss_core = torch.compile(self.critic_loss_core, **compile_kwargs)
            if not args.pc_actor:
                self.actor_loss_core = torch.compile(self.actor_loss_core, **compile_kwargs)

        self.max_action = max_action
        self.training_steps = 0
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8
        self.max_target = 0.0
        self.min_target = 0.0
        self.max_target_tensor = torch.zeros((), device=device)
        self.min_target_tensor = torch.zeros((), device=device)
        self.observed_max_target = torch.full((), -1e8, device=device)
        self.observed_min_target = torch.full((), 1e8, device=device)

        self.action_dim = action_dim
        self.sd_actor_generator = torch.Generator(device=device)
        self.sd_actor_generator.manual_seed(args.seed + 7102)
        self.ctrl_basis_generator = torch.Generator(device=device)
        self.ctrl_basis_generator.manual_seed(args.seed + 2903)
        self.ctrl_block_generator = torch.Generator(device=device)
        self.ctrl_block_generator.manual_seed(args.seed + 2904)
        self.ctrl_np_rng = np.random.default_rng(args.seed + 2903)
        self.model_np_rng = np.random.default_rng(args.seed + 5303)
        self.ctrl_basis = None
        self.ctrl_direction = None
        self.ctrl_hold_remaining = 0
        self.ctrl_block_guided = False
        self.ctrl_q_frontier = args.ctrl_q_floor
        self.ctrl_basis_explained = 0.0
        self.ctrl_attempts = 0
        self.ctrl_accepts = 0
        self.ctrl_scale_sum = 0.0
        self.ctrl_pending = None
        self.ctrl_guided_count = 0
        self.ctrl_guided_realized_sum = 0.0
        self.ctrl_noise_count = 0
        self.ctrl_noise_realized_sum = 0.0
        self.ctrl_corr_n = 0
        self.ctrl_corr_x = 0.0
        self.ctrl_corr_y = 0.0
        self.ctrl_corr_x2 = 0.0
        self.ctrl_corr_y2 = 0.0
        self.ctrl_corr_xy = 0.0

    def _control_zs(self, raw_zs):
        if self.canonical_control_latents:
            return jedi_control_latent(raw_zs, self.args.jedi_clamp)
        return raw_zs

    def _stock_prediction(self, encoder, control_zs, action):
        prediction = encoder.zsa(control_zs, action)
        if self.canonical_control_latents:
            prediction = jedi_control_latent(prediction, self.args.jedi_clamp)
        return prediction

    def _fixed_control_zsa(self, control_zs, action):
        stock_zsa = self._stock_prediction(self.fixed_encoder, control_zs, action)
        if self.fixed_jedi_sampler is None:
            return stock_zsa
        direct_zsa = self.fixed_jedi_sampler(control_zs, stock_zsa)
        return torch.lerp(stock_zsa, direct_zsa, self.fixed_jedi_control_mix)

    def _pc_actor_terminal_force(self, state, control_zs, policy_noise):
        """Differentiate only the frozen control boundary into a raw actor-output leaf."""
        with torch.no_grad():
            _, _, free_output = td7_pc_actor_free_phase(
                self.actor, state, control_zs
            )
        with torch.enable_grad():
            raw_output = free_output.detach().requires_grad_(True)
            actor_action, log_std = td7_pc_actor_policy_from_raw(
                self.actor, raw_output, policy_noise
            )
            actor_fixed_zsa = self._fixed_control_zsa(control_zs, actor_action)
            q_values = self.critic(
                state, actor_action, actor_fixed_zsa, control_zs
            )
            per_example_objective = (
                q_values.mean(dim=1) + self.alpha_tensor.detach() * log_std.sum(dim=1)
            )
            terminal_force = torch.autograd.grad(
                per_example_objective.sum(), raw_output
            )[0]
        raw_force_rms = terminal_force.detach().square().mean().sqrt()
        if self.args.pc_actor_normalize_terminal_force:
            terminal_force = terminal_force / raw_force_rms.clamp_min(
                self.args.pc_actor_force_rms_min
            )
        return (
            terminal_force.detach(),
            -per_example_objective.detach().mean(),
            log_std.detach().sum(dim=1).mean(),
            raw_force_rms,
        )

    def _pc_actor_update(self, state, fixed_zs, policy_noise):
        """Apply one PC actor update on its configured deterministic replay subset."""
        pc_state, pc_fixed_zs, pc_policy_noise = td7_pc_actor_batch(
            state,
            fixed_zs,
            policy_noise,
            self.args.pc_actor_batch_size,
        )
        (
            terminal_force,
            actor_loss,
            actor_log_pi,
            raw_terminal_force_rms,
        ) = self._pc_actor_terminal_force(
            pc_state, pc_fixed_zs, pc_policy_noise
        )
        diagnostics = self.pc_actor_trainer.step(
            pc_state,
            pc_fixed_zs,
            terminal_force,
            self.args.actor_lr,
        )
        diagnostics["raw_terminal_force_rms"] = raw_terminal_force_rms
        diagnostics["batch_size"] = terminal_force.new_tensor(pc_state.shape[0])
        return actor_loss, actor_log_pi, diagnostics

    def _min_q(self, state, action, control_zs):
        zsa = self._fixed_control_zsa(control_zs, action)
        return self.critic(state, action, zsa, control_zs).min(dim=1)[0]

    def _refresh_controllability_basis(self):
        if not self.args.controllability_exploration:
            return
        if self.training_steps < self.args.ctrl_start_training_steps:
            return
        states, _, _ = self.replay_buffer.sample_uniform(
            self.args.ctrl_basis_batch, rng=self.ctrl_np_rng
        )
        with torch.no_grad():
            raw_zs = self.fixed_encoder.zs(states)
            zs = self._control_zs(raw_zs)
            base_action = self.actor(states, zs)
            perturbation = torch.randn(
                base_action.shape,
                device=self.device,
                generator=self.ctrl_basis_generator,
            ) * self.args.ctrl_perturb_std
            plus_action = (base_action + perturbation).clamp(-1, 1)
            minus_action = (base_action - perturbation).clamp(-1, 1)
            plus_pred = self._fixed_control_zsa(zs, plus_action)
            minus_pred = self._fixed_control_zsa(zs, minus_action)
            action_distance = (plus_action - minus_action).norm(dim=1, keepdim=True).clamp_min(1e-6)
            effects = (plus_pred - minus_pred) / action_distance
            effects = effects - effects.mean(dim=0, keepdim=True)
            covariance = effects.T @ effects / max(effects.size(0) - 1, 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance.float())
            modes = min(self.args.ctrl_modes, eigenvectors.size(1))
            candidate_values = eigenvalues[-modes:]
            valid = candidate_values >= eigenvalues[-1].clamp_min(1e-12) * 1e-3
            if not bool(valid.any()) or float(eigenvalues[-1]) <= 1e-12:
                self.ctrl_basis = None
                self.ctrl_direction = None
                self.ctrl_hold_remaining = 0
                self.ctrl_block_guided = False
                return
            self.ctrl_basis = eigenvectors[:, -modes:][:, valid].T.contiguous()
            self.ctrl_basis_explained = float(
                candidate_values[valid].sum() / eigenvalues.clamp_min(0).sum().clamp_min(1e-12)
            )

            base_q = self._min_q(states, base_action, zs)
            plus_q = self._min_q(states, plus_action, zs)
            frontier = float((plus_q - base_q).abs().mean())
            self.ctrl_q_frontier = max(
                self.args.ctrl_q_floor,
                0.95 * self.ctrl_q_frontier + 0.05 * frontier,
            )
        self.ctrl_direction = None
        self.ctrl_hold_remaining = 0
        self.ctrl_block_guided = False

    def _ensure_controllability_direction(self):
        if self.ctrl_basis is None:
            return
        if self.ctrl_direction is None or self.ctrl_hold_remaining <= 0:
            mode = int(
                torch.randint(
                    self.ctrl_basis.size(0),
                    (),
                    device=self.device,
                    generator=self.ctrl_block_generator,
                )
            )
            sign = -1.0 if float(
                torch.rand((), device=self.device, generator=self.ctrl_block_generator)
            ) < 0.5 else 1.0
            self.ctrl_direction = sign * self.ctrl_basis[mode]
            self.ctrl_block_guided = float(
                torch.rand((), device=self.device, generator=self.ctrl_block_generator)
            ) < self._guided_probability()
            self.ctrl_hold_remaining = self.args.ctrl_hold_steps
        self.ctrl_hold_remaining -= 1

    def _guided_action(self, state, zs, base_action):
        self.ctrl_attempts += 1
        with torch.enable_grad():
            differentiable_action = base_action.detach().requires_grad_(True)
            prediction = self._fixed_control_zsa(zs.detach(), differentiable_action)
            progress = (prediction * self.ctrl_direction).sum()
            action_gradient = torch.autograd.grad(progress, differentiable_action)[0]
        gradient_rms = action_gradient.square().mean().sqrt()
        if not torch.isfinite(gradient_rms) or float(gradient_rms) < 1e-8:
            return base_action, False

        normalized_gradient = action_gradient / gradient_rms
        scales = (1.0, 0.5, 0.25)
        with torch.no_grad():
            full_candidate = (
                base_action + self.args.ctrl_perturb_std * normalized_gradient
            ).clamp(-1, 1)
            candidates = torch.cat(
                [base_action + scale * (full_candidate - base_action) for scale in scales],
                dim=0,
            )
            state_rep = state.expand(len(scales), -1)
            zs_rep = zs.expand(len(scales), -1)
            base_q = self._min_q(state, base_action, zs)[0]
            candidate_q = self._min_q(state_rep, candidates, zs_rep)

            eye = torch.eye(self.action_dim, device=self.device)
            plus_probes = (base_action + self.args.ctrl_perturb_std * eye).clamp(-1, 1)
            minus_probes = (base_action - self.args.ctrl_perturb_std * eye).clamp(-1, 1)
            probes = torch.cat([plus_probes, minus_probes], dim=0)
            probe_state = state.expand(2 * self.action_dim, -1)
            probe_zs = zs.expand(2 * self.action_dim, -1)
            probe_q = self._min_q(probe_state, probes, probe_zs)
            local_q_scale = (probe_q - base_q).abs().median().clamp_min(self.args.ctrl_q_floor)
            threshold = base_q - self.args.ctrl_q_slack * local_q_scale
            safe = candidate_q >= threshold
            for index, scale in enumerate(scales):
                if bool(safe[index]):
                    self.ctrl_accepts += 1
                    self.ctrl_scale_sum += scale
                    return candidates[index : index + 1], True
        return base_action, False

    def _guided_probability(self):
        progress = (self.training_steps - self.args.ctrl_start_training_steps) / max(
            self.args.ctrl_ramp_steps, 1
        )
        return self.args.ctrl_max_probability * float(np.clip(progress, 0.0, 1.0))

    def record_exploration_outcome(self, next_state):
        if self.ctrl_pending is None:
            return
        zs, direction, predicted_progress, guided = self.ctrl_pending
        self.ctrl_pending = None
        with torch.no_grad():
            next_state = torch.as_tensor(
                np.asarray(next_state).reshape(1, -1), dtype=torch.float32, device=self.device
            )
            next_zs = self._control_zs(self.fixed_encoder.zs(next_state))
            realized_progress = float(((next_zs - zs) * direction).sum())
        if guided:
            self.ctrl_guided_count += 1
            self.ctrl_guided_realized_sum += realized_progress
            self.ctrl_corr_n += 1
            self.ctrl_corr_x += predicted_progress
            self.ctrl_corr_y += realized_progress
            self.ctrl_corr_x2 += predicted_progress * predicted_progress
            self.ctrl_corr_y2 += realized_progress * realized_progress
            self.ctrl_corr_xy += predicted_progress * realized_progress
        else:
            self.ctrl_noise_count += 1
            self.ctrl_noise_realized_sum += realized_progress

    def reset_exploration_direction(self):
        self.ctrl_direction = None
        self.ctrl_hold_remaining = 0
        self.ctrl_block_guided = False
        self.ctrl_pending = None

    def _log_controllability(self):
        if not self.args.controllability_exploration:
            return
        attempts = max(self.ctrl_attempts, 1)
        accepts = max(self.ctrl_accepts, 1)
        guided_count = max(self.ctrl_guided_count, 1)
        noise_count = max(self.ctrl_noise_count, 1)
        numerator = self.ctrl_corr_n * self.ctrl_corr_xy - self.ctrl_corr_x * self.ctrl_corr_y
        denominator = np.sqrt(
            max(self.ctrl_corr_n * self.ctrl_corr_x2 - self.ctrl_corr_x**2, 0.0)
            * max(self.ctrl_corr_n * self.ctrl_corr_y2 - self.ctrl_corr_y**2, 0.0)
        )
        correlation = numerator / denominator if denominator > 1e-12 else 0.0
        self.writer.add_scalar("exploration/ctrl_probability", self._guided_probability(), self.training_steps)
        self.writer.add_scalar("exploration/ctrl_accept_frac", self.ctrl_accepts / attempts, self.training_steps)
        self.writer.add_scalar("exploration/ctrl_mean_scale", self.ctrl_scale_sum / accepts, self.training_steps)
        self.writer.add_scalar("exploration/ctrl_q_frontier", self.ctrl_q_frontier, self.training_steps)
        self.writer.add_scalar("exploration/ctrl_basis_explained", self.ctrl_basis_explained, self.training_steps)
        self.writer.add_scalar("exploration/ctrl_progress_correlation", correlation, self.training_steps)
        self.writer.add_scalar(
            "exploration/ctrl_guided_realized_progress",
            self.ctrl_guided_realized_sum / guided_count,
            self.training_steps,
        )
        self.writer.add_scalar(
            "exploration/ctrl_noise_realized_progress",
            self.ctrl_noise_realized_sum / noise_count,
            self.training_steps,
        )

    def select_action(self, state, use_checkpoint=False, use_exploration=True):
        with torch.no_grad():
            state = torch.as_tensor(state.reshape(1, -1), dtype=torch.float32, device=self.device)
            if self.args.sac_policy:
                if use_checkpoint:
                    zs = self._control_zs(self.checkpoint_encoder.zs(state))
                    action = self.checkpoint_actor.deterministic(state, zs)
                else:
                    zs = self._control_zs(self.fixed_encoder.zs(state))
                    if use_exploration:
                        action, _, _ = self.actor.sample(
                            state, zs, torch.randn((1, self.action_dim), device=self.device)
                        )
                    else:
                        action = self.actor.deterministic(state, zs)
                return action.cpu().numpy().flatten() * self.max_action
            if self.args.sd_noise:
                if use_checkpoint:
                    zs = self._control_zs(self.checkpoint_encoder.zs(state))
                    actor = self.checkpoint_actor
                else:
                    zs = self._control_zs(self.fixed_encoder.zs(state))
                    actor = self.actor
                if use_exploration:
                    action, _ = actor.sample_additive(
                        state, zs, torch.randn((1, self.action_dim), device=self.device)
                    )
                else:
                    action = actor(state, zs)
                return action.clamp(-1, 1).cpu().numpy().flatten() * self.max_action
            if use_checkpoint:
                zs = self._control_zs(self.checkpoint_encoder.zs(state))
                action = self.checkpoint_actor(state, zs)
            else:
                zs = self._control_zs(self.fixed_encoder.zs(state))
                action = self.actor(state, zs)
            guided = False
            ordinary_noise = None
            if use_exploration:
                # Always consume the baseline noise draw, even on guided steps, so replay/LAP and
                # target-smoothing RNG streams remain common-random-number aligned with StockSIG.
                ordinary_noise = torch.randn_like(action) * self.args.exploration_noise
            if (
                use_exploration
                and not use_checkpoint
                and self.args.controllability_exploration
                and self.ctrl_basis is not None
            ):
                self._ensure_controllability_direction()
                if self.ctrl_block_guided:
                    action, guided = self._guided_action(state, zs, action)
            if use_exploration:
                if not guided:
                    action = action + ordinary_noise
                action = action.clamp(-1, 1)
                if self.args.controllability_exploration and self.ctrl_direction is not None:
                    predicted_next = self._fixed_control_zsa(zs, action)
                    predicted_progress = float(((predicted_next - zs) * self.ctrl_direction).sum())
                    self.ctrl_pending = (
                        zs.detach(),
                        self.ctrl_direction.detach().clone(),
                        predicted_progress,
                        guided,
                    )
            return action.clamp(-1, 1).cpu().numpy().flatten() * self.max_action

    def _log_latent_geometry(self, zs, next_zs, pred_zs, action):
        with torch.no_grad():
            flat = torch.cat([zs, next_zs], dim=0)
            centered = flat - flat.mean(dim=0, keepdim=True)
            cov = centered.T @ centered / max(flat.size(0) - 1, 1)
            eig = torch.linalg.eigvalsh(cov.double()).clamp_min(1e-12)
            prob = eig / eig.sum()
            effective_rank = torch.exp(-(prob * prob.log()).sum())

            shuffled = action[torch.randperm(action.size(0), device=action.device)]
            control_zs = self._control_zs(zs)
            control_next_zs = self._control_zs(next_zs)
            shuffled_pred = self._stock_prediction(self.encoder, control_zs, shuffled)
            action_spread = (pred_zs - shuffled_pred).square().mean().sqrt()
            transition_rms = (control_next_zs - control_zs).square().mean().sqrt()
            delta_rms = (pred_zs - control_zs).square().mean().sqrt()
            copy_mse = (control_zs - control_next_zs).square().mean()
            prediction_mse = (pred_zs - control_next_zs).square().mean()
            coordinate_std = centered.std(dim=0)

            self.writer.add_scalar("latent/effective_rank", effective_rank.item(), self.training_steps)
            self.writer.add_scalar("latent/max_eigen_frac", prob[-1].item(), self.training_steps)
            self.writer.add_scalar("latent/action_spread", action_spread.item(), self.training_steps)
            self.writer.add_scalar(
                "latent/action_spread_over_transition",
                (action_spread / transition_rms.clamp_min(1e-8)).item(),
                self.training_steps,
            )
            self.writer.add_scalar("latent/transition_rms", transition_rms.item(), self.training_steps)
            self.writer.add_scalar("latent/delta_rms", delta_rms.item(), self.training_steps)
            self.writer.add_scalar(
                "latent/prediction_over_copy_mse",
                (prediction_mse / copy_mse.clamp_min(1e-8)).item(),
                self.training_steps,
            )
            self.writer.add_scalar("latent/coordinate_std_min", coordinate_std.min().item(), self.training_steps)
            self.writer.add_scalar("latent/coordinate_std_mean", coordinate_std.mean().item(), self.training_steps)
            self.writer.add_scalar("latent/coordinate_std_max", coordinate_std.max().item(), self.training_steps)

    def _log_jedi(self, diagnostics):
        if diagnostics is None:
            return
        x0_mse = diagnostics["x0_mse"]
        skip_x0_mse = diagnostics["skip_x0_mse"]
        high_x0_mse = diagnostics["high_x0_mse"]
        high_skip_x0_mse = diagnostics["high_skip_x0_mse"]
        values = {
            **diagnostics,
            "skip_improvement": 1.0 - x0_mse / skip_x0_mse.clamp_min(1e-12),
            "high_skip_improvement": 1.0
            - high_x0_mse / high_skip_x0_mse.clamp_min(1e-12),
        }
        for name, value in values.items():
            self.writer.add_scalar(f"jedi/{name}", value.item(), self.training_steps)

    def _log_reward_token(self, zs, prediction, action, reward):
        if self.reward_token_head is None:
            return
        with torch.no_grad():
            token, predicted_forward_symlog = self.reward_token_head(
                prediction, self.args.reward_token_shared_scale
            )
            control_cost = self.args.reward_control_cost_coef * action.square().sum(
                dim=1, keepdim=True
            )
            target_forward_symlog = symlog(reward + control_cost)
            shuffled_action = action.roll(1, dims=0)
            shuffled_prediction = self.encoder.zsa(zs, shuffled_action)
            _, shuffled_forward_symlog = self.reward_token_head(
                shuffled_prediction, self.args.reward_token_shared_scale
            )
            correct_mse = F.mse_loss(predicted_forward_symlog, target_forward_symlog)
            shuffled_mse = F.mse_loss(shuffled_forward_symlog, target_forward_symlog)
            predicted_total = symexp(predicted_forward_symlog) - control_cost
            centered_prediction = predicted_total - predicted_total.mean()
            centered_reward = reward - reward.mean()
            correlation = (centered_prediction * centered_reward).mean()
            correlation = correlation / (
                centered_prediction.square().mean().sqrt()
                * centered_reward.square().mean().sqrt()
            ).clamp_min(1e-8)
            centered_token = token - token.mean()
            token_std = centered_token.square().mean().sqrt().clamp_min(1e-8)
            token_skew = (centered_token / token_std).pow(3).mean()
            self.writer.add_scalar(
                "reward_token/shuffled_action_loss_ratio",
                (shuffled_mse / correct_mse.clamp_min(1e-12)).item(),
                self.training_steps,
            )
            self.writer.add_scalar(
                "reward_token/raw_total_correlation",
                correlation.item(),
                self.training_steps,
            )
            self.writer.add_scalar(
                "reward_token/token_mean", token.mean().item(), self.training_steps
            )
            self.writer.add_scalar(
                "reward_token/token_skew", token_skew.item(), self.training_steps
            )

    def _log_lewm_geometry(self, zs, action, next_zs, prediction):
        if self.lewm_projector is None:
            return
        with torch.no_grad():
            batch_size = zs.size(0)
            projected_pair = self.lewm_projector(torch.cat([zs, next_zs], dim=0))
            projected_zs, projected_next = projected_pair.split(batch_size, dim=0)
            prediction_input = prediction
            if self.lewm_dynamics is not None:
                prediction_input = self.lewm_dynamics(projected_zs, action)
            projected_prediction = self.lewm_pred_projector(prediction_input)
            flat = torch.cat([projected_zs, projected_next], dim=0)
            centered = flat - flat.mean(dim=0, keepdim=True)
            covariance = centered.T @ centered / max(flat.size(0) - 1, 1)
            eigenvalues = torch.linalg.eigvalsh(covariance.double()).clamp_min(1e-12)
            probabilities = eigenvalues / eigenvalues.sum()
            effective_rank = torch.exp(-(probabilities * probabilities.log()).sum())
            coordinate_std = flat.std(dim=0, unbiased=False)
            radius = flat.square().sum(dim=1).sqrt()
            self.writer.add_scalar(
                "lewm/effective_rank", effective_rank.item(), self.training_steps
            )
            self.writer.add_scalar(
                "lewm/max_eigen_frac", probabilities[-1].item(), self.training_steps
            )
            self.writer.add_scalar(
                "lewm/prediction_mse",
                F.mse_loss(projected_prediction, projected_next).item(),
                self.training_steps,
            )
            self.writer.add_scalar(
                "lewm/coordinate_std_mean",
                coordinate_std.mean().item(),
                self.training_steps,
            )
            self.writer.add_scalar(
                "lewm/radius_cv",
                (radius.std(unbiased=False) / radius.mean().clamp_min(1e-12)).item(),
                self.training_steps,
            )
            if self.lewm_dynamics is not None:
                shuffled_prediction = self.lewm_pred_projector(
                    self.lewm_dynamics(projected_zs, action.roll(1, dims=0))
                )
                correct_mse = F.mse_loss(projected_prediction, projected_next)
                shuffled_mse = F.mse_loss(shuffled_prediction, projected_next)
                self.writer.add_scalar(
                    "lewm/shuffled_action_loss_ratio",
                    (shuffled_mse / correct_mse.clamp_min(1e-12)).item(),
                    self.training_steps,
                )
                self.writer.add_scalar(
                    "lewm/action_spread",
                    (projected_prediction - shuffled_prediction)
                    .square()
                    .mean()
                    .sqrt()
                    .item(),
                    self.training_steps,
                )

    def _log_jedi_endpoint(self, state, action, next_state):
        if self.fixed_jedi_sampler is None:
            return
        diagnostic_batch = min(state.size(0), 32)
        diag_state = state[:diagnostic_batch]
        diag_action = action[:diagnostic_batch]
        diag_next_state = next_state[:diagnostic_batch]
        with torch.no_grad():
            raw_zs = self.fixed_encoder.zs(diag_state)
            zs = self._control_zs(raw_zs)
            stock_zsa = self._stock_prediction(self.fixed_encoder, zs, diag_action)
            direct_zsa = self.fixed_jedi_sampler(zs, stock_zsa)
            repeated_zsa = self.fixed_jedi_sampler(zs, stock_zsa)
            raw_next_zs = self.fixed_encoder.zs(diag_next_state)
            next_zs = self._control_zs(raw_next_zs)
            step_rms, clamp_fraction = self.fixed_jedi_sampler.trajectory_diagnostics(
                zs, stock_zsa
            )

            num_priors = self.jedi_endpoint_diag_priors.size(0)
            repeated_zs = zs.unsqueeze(0).expand(num_priors, -1, -1).reshape(
                num_priors * diagnostic_batch, -1
            )
            repeated_stock = stock_zsa.unsqueeze(0).expand(num_priors, -1, -1).reshape(
                num_priors * diagnostic_batch, -1
            )
            prior_bank = self.jedi_endpoint_diag_priors.unsqueeze(1).expand(
                -1, diagnostic_batch, -1
            ).reshape(num_priors * diagnostic_batch, -1)
            prior_endpoints = self.fixed_jedi_sampler.endpoint_with_prior(
                repeated_zs, repeated_stock, prior_bank
            ).reshape(num_priors, diagnostic_batch, -1)
            prior_variance = prior_endpoints.var(dim=0, unbiased=False).mean()

            stock_q = self.critic(diag_state, diag_action, stock_zsa, zs)
            direct_q = self.critic(diag_state, diag_action, direct_zsa, zs)
            endpoint_values = {
                "control_mix": self.fixed_jedi_control_mix,
                "target_control_mix": self.fixed_jedi_control_mix_target,
                "fixed_prior_repeat_error": (direct_zsa - repeated_zsa).abs().max(),
                "prior_output_variance": prior_variance,
                "direct_vs_stock_mse": F.mse_loss(direct_zsa, stock_zsa),
                "direct_vs_stock_cosine": F.cosine_similarity(
                    direct_zsa, stock_zsa, dim=1
                ).mean(),
                "direct_vs_next_mse": F.mse_loss(direct_zsa, next_zs),
                "stock_vs_next_mse": F.mse_loss(stock_zsa, next_zs),
                "direct_vs_raw_next_mse": F.mse_loss(direct_zsa, raw_next_zs),
                "direct_to_next_rms_ratio": direct_zsa.square().mean().sqrt()
                / next_zs.square().mean().sqrt().clamp_min(1e-8),
                "stock_to_next_rms_ratio": stock_zsa.square().mean().sqrt()
                / next_zs.square().mean().sqrt().clamp_min(1e-8),
                "critic_q_direct_vs_stock": (direct_q - stock_q).abs().mean(),
            }
            for index in range(3):
                endpoint_values[f"step_{index + 1}_update_rms"] = step_rms[index]
                endpoint_values[f"step_{index + 1}_clamp_fraction"] = clamp_fraction[index]

        jacobian_batch = min(diagnostic_batch, 16)
        with torch.enable_grad():
            jacobian_action = diag_action[:jacobian_batch].detach().requires_grad_(True)
            jacobian_zs = self._control_zs(
                self.fixed_encoder.zs(diag_state[:jacobian_batch])
            ).detach()
            jacobian_stock = self._stock_prediction(
                self.fixed_encoder, jacobian_zs, jacobian_action
            )
            jacobian_direct = self.fixed_jedi_sampler(jacobian_zs, jacobian_stock)
            direction = self.jedi_endpoint_diag_priors[0].expand(jacobian_batch, -1)
            stock_projection = (jacobian_stock * direction).sum()
            direct_projection = (jacobian_direct * direction).sum()
            stock_gradient = torch.autograd.grad(
                stock_projection, jacobian_action, retain_graph=True
            )[0]
            direct_gradient = torch.autograd.grad(direct_projection, jacobian_action)[0]
            endpoint_values["stock_action_jacobian_rms"] = stock_gradient.square().mean().sqrt()
            endpoint_values["exact_action_jacobian_rms"] = direct_gradient.square().mean().sqrt()

        for name, value in endpoint_values.items():
            self.writer.add_scalar(
                f"jedi_endpoint/{name}", value.detach().item(), self.training_steps
            )

    def train(self):
        self.training_steps += 1
        if self.jedi_control_mix_tensor is not None:
            control_progress = min(
                self.training_steps / max(self.args.jedi_endpoint_mix_steps, 1),
                1.0,
            )
            self.jedi_control_mix_tensor.fill_(control_progress)
        if self.args.torch_compile and self.training_steps == 1:
            print("compiling static encoder, critic, and actor training graphs on their first uses")

        state, action, next_state, reward, not_done = self.replay_buffer.sample()
        enc_reward = None
        enc_not_done = None
        if self.reward_token_head is not None or self.outcome_tokens is not None:
            (
                enc_state,
                enc_action,
                enc_next_state,
                enc_reward,
                enc_not_done,
            ) = self.replay_buffer.sample_uniform_with_reward()
        else:
            enc_state, enc_action, enc_next_state = self.replay_buffer.sample_uniform()
        policy_mean_target = None
        if self.policy_mean_head is not None:
            with torch.no_grad():
                target_policy_zs = self.fixed_encoder_target.zs(enc_state)
                policy_mean_target = self.actor_target(enc_state, target_policy_zs)
        elif self.outcome_tokens is not None:
            with torch.no_grad():
                if self.args.outcome_policy_source == "behavior":
                    target_policy_zs = self.fixed_encoder.zs(enc_next_state)
                    policy_actor = self.actor
                else:
                    target_policy_zs = self.fixed_encoder_target.zs(enc_next_state)
                    policy_actor = self.actor_target
                if (
                    self.args.semantic_outcome_tokens
                    or self.args.latent_outcome_tokens
                    or self.args.isometric_outcome_tokens
                ):
                    policy_raw_mean, policy_log_std = policy_actor.policy_stats(
                        enc_next_state, target_policy_zs
                    )
                    policy_center = (
                        torch.tanh(policy_raw_mean)
                        if self.args.sac_policy
                        else policy_raw_mean
                    )
                    log_std_min = (
                        self.args.sac_log_std_min
                        if self.args.sac_policy
                        else self.args.sd_log_std_min
                    )
                    log_std_max = (
                        self.args.sac_log_std_max
                        if self.args.sac_policy
                        else self.args.sd_log_std_max
                    )
                    normalized_log_std = (
                        2.0
                        * (policy_log_std - log_std_min)
                        / (log_std_max - log_std_min)
                        - 1.0
                    )
                    policy_mean_target = torch.cat(
                        [policy_center, normalized_log_std], dim=-1
                    )
                elif self.args.outcome_policy_include_log_std:
                    policy_raw_mean, policy_log_std = policy_actor.policy_stats(
                        enc_next_state, target_policy_zs
                    )
                    policy_mean_target = torch.cat(
                        [torch.tanh(policy_raw_mean), policy_log_std], dim=-1
                    )
                else:
                    policy_mean_target = policy_actor(
                        enc_next_state, target_policy_zs
                    )
        sequence_states = None
        sequence_actions = None
        diagnostic_sequence_states = None
        diagnostic_sequence_actions = None
        if self.rollout_loss_core is not None:
            sequence_states, sequence_actions, _ = self.replay_buffer.sample_sequences(
                self.args.lewm_rollout_horizon,
                rng=self.model_np_rng,
            )
            if self.training_steps % self.args.latent_log_freq == 0:
                (
                    diagnostic_sequence_states,
                    diagnostic_sequence_actions,
                    _,
                ) = self.replay_buffer.sample_sequences(
                    self.args.lewm_rollout_horizon,
                    rng=self.model_np_rng,
                )
        if self.subsig is not None:
            self.subsig.resample_directions()
        if self.full_sigreg is not None:
            self.full_sigreg.resample_directions()
        if isinstance(self.reward_outcome_sigreg, FullSIGReg):
            self.reward_outcome_sigreg.resample_directions()
        if self.policy_outcome_sigreg is not None:
            self.policy_outcome_sigreg.resample_directions()
        if self.rollout_full_sigreg is not None:
            self.rollout_full_sigreg.resample_directions()

        jedi_sigma_normal = None
        jedi_epsilon = None
        if self.jedi_denoiser is not None:
            jedi_sigma_normal = torch.randn(
                (enc_state.size(0), 1),
                device=self.device,
                generator=self.jedi_generator,
            )
            jedi_epsilon = torch.randn(
                (
                    enc_state.size(0),
                    self.args.lewm_rollout_dim
                    if self.args.lewm_rollout_aux
                    else self.args.zs_dim,
                ),
                device=self.device,
                generator=self.jedi_generator,
            )
            coef_warmup = min(
                self.training_steps / max(self.args.jedi_coef_warmup_steps, 1),
                1.0,
            )
            self.jedi_weight_tensor.fill_(self.args.jedi_coef * coef_warmup)
        if self.lewm_weight_tensor is not None:
            lewm_warmup = min(
                self.training_steps / max(self.args.lewm_warmup_steps, 1),
                1.0,
            )
            self.lewm_weight_tensor.fill_(self.args.lewm_coef * lewm_warmup)
        if self.reward_token_weight_tensor is not None:
            reward_warmup = min(
                self.training_steps / max(self.args.reward_token_warmup_steps, 1),
                1.0,
            )
            self.reward_token_weight_tensor.fill_(
                self.args.reward_token_coef * reward_warmup
            )

        # Gradient-alignment diagnostics need multiple eager autograd traversals. All other
        # steady-state updates use the compiled owner.
        rollout_firewall_step = self.rollout_loss_core is not None and (
            self.training_steps == 1
            or self.training_steps % self.args.target_update_rate == 0
        )
        outcome_equalization_step = (
            self.args.adaptive_outcome_grad_equalization
            and self.training_steps % self.args.outcome_grad_equalization_interval == 0
        )
        alignment_step = rollout_firewall_step or self.training_steps == 1 or (
            (
                self.jedi_denoiser is not None
                or self.lewm_projector is not None
                or self.rollout_loss_core is not None
                or self.reward_token_head is not None
                or self.outcome_tokens is not None
            )
            and self.training_steps in {1_000, 10_000}
        ) or outcome_equalization_step
        encoder_core = (
            self.encoder_loss_eager
            if self.args.torch_compile and alignment_step
            else self.encoder_loss_core
        )
        (
            encoder_loss,
            prediction_loss,
            sigreg_loss,
            jedi_loss,
            lewm_prediction_loss,
            full_sigreg_loss,
            reward_token_loss,
            reward_token_sigreg_loss,
            reward_token_mae,
            reward_token_std,
            policy_mean_loss,
            policy_mean_mae,
            policy_mean_std,
            outcome_reward_prediction_loss,
            outcome_reward_sigreg_loss,
            outcome_policy_prediction_loss,
            outcome_policy_sigreg_loss,
            outcome_reward_mae,
            outcome_policy_mae,
            outcome_reward_target_std,
            outcome_policy_target_std,
            outcome_reward_clipped_fraction,
            outcome_reward_pred_edge_mass,
            outcome_reward_semantic_std,
            outcome_policy_semantic_std,
            outcome_reward_semantic_loss,
            outcome_policy_semantic_loss,
            outcome_reward_reconstruction_mae,
            outcome_policy_mean_mae,
            outcome_policy_logstd_mae,
        ) = encoder_core(
            state,
            action,
            next_state,
            enc_state,
            enc_action,
            enc_next_state,
            enc_reward,
            policy_mean_target,
            enc_not_done,
            jedi_sigma_normal,
            jedi_epsilon,
            self.jedi_weight_tensor,
            self.lewm_weight_tensor,
            self.reward_token_weight_tensor,
        )
        unit_loss_scale = prediction_loss.new_ones(())
        prediction_loss_scale = unit_loss_scale
        lewm_prediction_loss_scale = unit_loss_scale
        reward_outcome_loss_scale = unit_loss_scale
        policy_outcome_loss_scale = unit_loss_scale
        if self.encoder_loss_eager.dreamer_loss_normalization:
            prediction_loss_scale = (
                self.encoder_loss_eager.prediction_loss_normalizer.last_rms
                .clamp_min(self.args.loss_normalization_eps)
                .reciprocal()
            )
            lewm_prediction_loss_scale = (
                self.encoder_loss_eager.lewm_prediction_loss_normalizer.last_rms
                .clamp_min(self.args.loss_normalization_eps)
                .reciprocal()
            )
            reward_outcome_loss_scale = (
                self.encoder_loss_eager.reward_outcome_loss_normalizer.last_rms
                .clamp_min(self.args.loss_normalization_eps)
                .reciprocal()
            )
            policy_outcome_loss_scale = (
                self.encoder_loss_eager.policy_outcome_loss_normalizer.last_rms
                .clamp_min(self.args.loss_normalization_eps)
                .reciprocal()
            )

        next_outcome_trunk_scales = None
        if outcome_equalization_step:
            shared_parameters = list(self.encoder.parameters())
            representation_objective = (
                prediction_loss + self.args.subsig_coef * sigreg_loss
            )
            if self.lewm_projector is not None:
                representation_objective = representation_objective + (
                    self.lewm_weight_tensor
                    * (
                        lewm_prediction_loss
                        + self.args.lewm_sigreg_coef * full_sigreg_loss
                    )
                )

            def shared_grad_norm(loss):
                gradients = torch.autograd.grad(
                    loss,
                    shared_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                squared_norm = loss.new_zeros(())
                for gradient in gradients:
                    if gradient is not None:
                        squared_norm = squared_norm + gradient.square().sum()
                return squared_norm.sqrt()

            representation_grad_norm = shared_grad_norm(representation_objective)
            reward_scaled_grad_norm = shared_grad_norm(
                outcome_reward_prediction_loss
            )
            policy_scaled_grad_norm = shared_grad_norm(
                outcome_policy_prediction_loss
            )
            reward_raw_grad_norm = reward_scaled_grad_norm / (
                self.outcome_tokens.reward_trunk_scale.abs().clamp_min(1e-12)
            )
            policy_raw_grad_norm = policy_scaled_grad_norm / (
                self.outcome_tokens.policy_trunk_scale.abs().clamp_min(1e-12)
            )
            next_outcome_trunk_scales = equalized_outcome_trunk_scales(
                representation_grad_norm,
                reward_raw_grad_norm,
                policy_raw_grad_norm,
            ).detach()
            self.writer.add_scalar(
                "outcome_equalization/representation_grad_norm",
                representation_grad_norm.item(),
                self.training_steps,
            )
            self.writer.add_scalar(
                "outcome_equalization/reward_raw_grad_norm",
                reward_raw_grad_norm.item(),
                self.training_steps,
            )
            self.writer.add_scalar(
                "outcome_equalization/policy_raw_grad_norm",
                policy_raw_grad_norm.item(),
                self.training_steps,
            )
            self.writer.add_scalar(
                "outcome_equalization/reward_trunk_scale",
                next_outcome_trunk_scales[0].item(),
                self.training_steps,
            )
            self.writer.add_scalar(
                "outcome_equalization/policy_trunk_scale",
                next_outcome_trunk_scales[1].item(),
                self.training_steps,
            )

        rollout_horizon_mse = None
        rollout_jedi_x0_mse = None
        next_trunk_scale = None
        if self.rollout_loss_core is not None:
            firewall_step = rollout_firewall_step
            rollout_core = self.rollout_loss_eager if firewall_step else self.rollout_loss_core
            (
                lewm_prediction_loss,
                full_sigreg_loss,
                jedi_loss,
                rollout_jedi_x0_mse,
                rollout_horizon_mse,
            ) = rollout_core(
                sequence_states,
                sequence_actions,
                jedi_sigma_normal,
                jedi_epsilon,
                self.lewm_trunk_scale_tensor,
            )
            rollout_objective = (
                lewm_prediction_loss + self.args.lewm_sigreg_coef * full_sigreg_loss
            )
            weighted_rollout_objective = (
                self.lewm_weight_tensor * rollout_objective
                + self.jedi_weight_tensor * jedi_loss
            )
            encoder_loss = encoder_loss + weighted_rollout_objective

            if firewall_step:
                # Calibrate against final coefficients, not warmup coefficients. The resulting
                # scale is conservative throughout warmup and cannot sawtooth above the cap.
                full_aux_objective = (
                    self.args.lewm_coef * rollout_objective
                    + self.args.jedi_coef * jedi_loss
                )
                shared_params = [
                    *self.encoder.zs1.parameters(),
                    *self.encoder.zs2.parameters(),
                ]
                stock_grads = torch.autograd.grad(
                    prediction_loss, shared_params, retain_graph=True, allow_unused=True
                )
                aux_grads = torch.autograd.grad(
                    full_aux_objective,
                    shared_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                stock_vector = torch.cat(
                    [
                        torch.zeros_like(param).flatten() if grad is None else grad.flatten()
                        for param, grad in zip(shared_params, stock_grads)
                    ]
                )
                aux_vector = torch.cat(
                    [
                        torch.zeros_like(param).flatten() if grad is None else grad.flatten()
                        for param, grad in zip(shared_params, aux_grads)
                    ]
                )
                stock_norm = stock_vector.norm()
                aux_norm = aux_vector.norm()
                current_scale = self.lewm_trunk_scale_tensor.clamp_min(1e-6)
                raw_aux_norm = aux_norm / current_scale
                if self.training_steps == 1:
                    self.lewm_stock_grad_ema.copy_(stock_norm.detach())
                    self.lewm_aux_grad_ema.copy_(raw_aux_norm.detach())
                else:
                    self.lewm_stock_grad_ema.lerp_(stock_norm.detach(), 0.1)
                    self.lewm_aux_grad_ema.lerp_(raw_aux_norm.detach(), 0.1)
                next_trunk_scale = torch.minimum(
                    torch.minimum(
                        current_scale.new_ones(()),
                        self.args.lewm_aux_trunk_cap
                        * self.lewm_stock_grad_ema
                        / self.lewm_aux_grad_ema.clamp_min(1e-12),
                    ),
                    self.args.lewm_aux_trunk_cap
                    * stock_norm
                    / raw_aux_norm.clamp_min(1e-12),
                )
                self.writer.add_scalar(
                    "lewm_rollout/aux_to_stock_trunk_ratio",
                    (
                        next_trunk_scale
                        * raw_aux_norm
                        / stock_norm.clamp_min(1e-12)
                    ).item(),
                    self.training_steps,
                )
                self.writer.add_scalar(
                    "lewm_rollout/update_cosine_to_stock",
                    F.cosine_similarity(
                        stock_vector
                        + aux_vector * (next_trunk_scale / current_scale),
                        stock_vector,
                        dim=0,
                    ).item(),
                    self.training_steps,
                )

        if self.subsig is not None and self.training_steps == 1:
            shared_params = [
                *self.encoder.zs1.parameters(),
                *self.encoder.zs2.parameters(),
                *self.encoder.zs3.parameters(),
            ]
            prediction_grads = torch.autograd.grad(
                prediction_loss, shared_params, retain_graph=True, allow_unused=True
            )
            sigreg_grads = torch.autograd.grad(
                sigreg_loss, shared_params, retain_graph=True, allow_unused=True
            )
            zero = prediction_loss.new_zeros(())
            prediction_grad_norm = torch.sqrt(
                sum((grad.square().sum() for grad in prediction_grads if grad is not None), zero)
            )
            sigreg_grad_norm = torch.sqrt(
                sum((grad.square().sum() for grad in sigreg_grads if grad is not None), zero)
            )
            weighted_ratio = (
                self.args.subsig_coef
                * sigreg_grad_norm
                / (prediction_loss_scale * prediction_grad_norm).clamp_min(1e-12)
            )
            prediction_vector = torch.cat(
                [grad.flatten() for grad in prediction_grads if grad is not None]
            )
            sigreg_vector = torch.cat([grad.flatten() for grad in sigreg_grads if grad is not None])
            gradient_cosine = F.cosine_similarity(prediction_vector, sigreg_vector, dim=0)
            self.writer.add_scalar("losses/initial_shared_prediction_grad_norm", prediction_grad_norm.item(), 1)
            self.writer.add_scalar("losses/initial_shared_subsig_grad_norm", sigreg_grad_norm.item(), 1)
            self.writer.add_scalar(
                "losses/initial_shared_subsig_grad_ratio", weighted_ratio.item(), self.training_steps
            )
            self.writer.add_scalar("losses/initial_shared_subsig_grad_cosine", gradient_cosine.item(), 1)

        if self.reward_token_head is not None and self.training_steps in {1, 1_000, 10_000}:
            shared_params = list(self.encoder.parameters())
            reward_objective = (
                reward_token_loss
                + self.args.reward_token_sigreg_coef * reward_token_sigreg_loss
            )
            prediction_grads = torch.autograd.grad(
                prediction_loss, shared_params, retain_graph=True, allow_unused=True
            )
            reward_grads = torch.autograd.grad(
                reward_objective, shared_params, retain_graph=True, allow_unused=True
            )
            prediction_vector = torch.cat(
                [
                    torch.zeros_like(param).flatten() if grad is None else grad.flatten()
                    for param, grad in zip(shared_params, prediction_grads)
                ]
            )
            reward_vector = torch.cat(
                [
                    torch.zeros_like(param).flatten() if grad is None else grad.flatten()
                    for param, grad in zip(shared_params, reward_grads)
                ]
            )
            self.writer.add_scalar(
                "reward_token/final_shared_grad_ratio",
                (
                    reward_vector.norm()
                    / (prediction_loss_scale * prediction_vector.norm()).clamp_min(1e-12)
                ).item(),
                self.training_steps,
            )
            self.writer.add_scalar(
                "reward_token/shared_prediction_grad_cosine",
                F.cosine_similarity(reward_vector, prediction_vector, dim=0).item(),
                self.training_steps,
            )

        if self.outcome_tokens is not None and self.training_steps in {1, 1_000, 10_000}:
            shared_params = list(self.encoder.parameters())
            prediction_grads = torch.autograd.grad(
                prediction_loss, shared_params, retain_graph=True, allow_unused=True
            )
            prediction_vector = torch.cat(
                [
                    torch.zeros_like(param).flatten() if grad is None else grad.flatten()
                    for param, grad in zip(shared_params, prediction_grads)
                ]
            )
            for name, token_loss in (
                ("reward", outcome_reward_prediction_loss),
                ("policy", outcome_policy_prediction_loss),
            ):
                token_coef = (
                    self.encoder_loss_eager.policy_beta_nll_coef
                    if name == "policy"
                    and getattr(self.outcome_tokens, "policy_beta_nll", False)
                    else self.args.outcome_token_coef
                )
                if self.encoder_loss_eager.dreamer_loss_normalization:
                    token_coef = token_coef * (
                        reward_outcome_loss_scale
                        if name == "reward"
                        else policy_outcome_loss_scale
                    )
                token_grads = torch.autograd.grad(
                    token_loss, shared_params, retain_graph=True, allow_unused=True
                )
                token_vector = torch.cat(
                    [
                        torch.zeros_like(param).flatten() if grad is None else grad.flatten()
                        for param, grad in zip(shared_params, token_grads)
                    ]
                )
                self.writer.add_scalar(
                    f"outcome_tokens/{name}_shared_grad_ratio",
                    (
                        token_coef
                        * token_vector.norm()
                        / (
                            prediction_loss_scale * prediction_vector.norm()
                        ).clamp_min(1e-12)
                    ).item(),
                    self.training_steps,
                )
                self.writer.add_scalar(
                    f"outcome_tokens/{name}_prediction_grad_cosine",
                    F.cosine_similarity(
                        token_vector, prediction_vector, dim=0
                    ).item(),
                    self.training_steps,
                )

        if (
            self.jedi_denoiser is not None
            and self.rollout_loss_core is None
            and self.training_steps in {1_000, 10_000}
        ):
            shared_params = [
                *self.encoder.zs1.parameters(),
                *self.encoder.zs2.parameters(),
                *self.encoder.zs3.parameters(),
            ]
            prediction_grads = torch.autograd.grad(
                prediction_loss, shared_params, retain_graph=True, allow_unused=True
            )
            jedi_grads = torch.autograd.grad(
                jedi_loss, shared_params, retain_graph=True, allow_unused=True
            )
            prediction_vector = torch.cat(
                [
                    torch.zeros_like(param).flatten() if grad is None else grad.flatten()
                    for param, grad in zip(shared_params, prediction_grads)
                ]
            )
            jedi_vector = torch.cat(
                [
                    torch.zeros_like(param).flatten() if grad is None else grad.flatten()
                    for param, grad in zip(shared_params, jedi_grads)
                ]
            )
            prediction_grad_norm = prediction_vector.norm()
            jedi_grad_norm = jedi_vector.norm()
            weighted_jedi_ratio = (
                self.jedi_weight_tensor
                * jedi_grad_norm
                / (prediction_loss_scale * prediction_grad_norm).clamp_min(1e-12)
            )
            gradient_cosine = F.cosine_similarity(prediction_vector, jedi_vector, dim=0)
            self.writer.add_scalar(
                "jedi/shared_grad_norm", jedi_grad_norm.item(), self.training_steps
            )
            self.writer.add_scalar(
                "jedi/weighted_shared_grad_ratio",
                weighted_jedi_ratio.item(),
                self.training_steps,
            )
            self.writer.add_scalar(
                "jedi/shared_prediction_grad_cosine",
                gradient_cosine.item(),
                self.training_steps,
            )

        if self.lewm_projector is not None and self.training_steps in {1_000, 10_000}:
            shared_params = [
                *self.encoder.zs1.parameters(),
                *self.encoder.zs2.parameters(),
                *self.encoder.zs3.parameters(),
            ]
            lewm_objective = (
                lewm_prediction_loss_scale * lewm_prediction_loss
                + self.args.lewm_sigreg_coef * full_sigreg_loss
            )
            prediction_grads = torch.autograd.grad(
                prediction_loss, shared_params, retain_graph=True, allow_unused=True
            )
            lewm_grads = torch.autograd.grad(
                lewm_objective, shared_params, retain_graph=True, allow_unused=True
            )
            prediction_vector = torch.cat(
                [
                    torch.zeros_like(param).flatten() if grad is None else grad.flatten()
                    for param, grad in zip(shared_params, prediction_grads)
                ]
            )
            lewm_vector = torch.cat(
                [
                    torch.zeros_like(param).flatten() if grad is None else grad.flatten()
                    for param, grad in zip(shared_params, lewm_grads)
                ]
            )
            prediction_grad_norm = prediction_vector.norm()
            lewm_grad_norm = lewm_vector.norm()
            weighted_lewm_ratio = (
                self.lewm_weight_tensor
                * lewm_grad_norm
                / (prediction_loss_scale * prediction_grad_norm).clamp_min(1e-12)
            )
            gradient_cosine = F.cosine_similarity(
                prediction_vector, lewm_vector, dim=0
            )
            self.writer.add_scalar(
                "lewm/weighted_shared_grad_ratio",
                weighted_lewm_ratio.item(),
                self.training_steps,
            )
            self.writer.add_scalar(
                "lewm/shared_prediction_grad_cosine",
                gradient_cosine.item(),
                self.training_steps,
            )

        latent_diagnostics = None
        jedi_diagnostics = None
        rollout_diagnostics = None
        isometric_outcome_diagnostics = None
        if self.training_steps % self.args.latent_log_freq == 0:
            with torch.no_grad():
                diagnostic_zs = self.encoder.zs(enc_state)
                diagnostic_next_zs = self.encoder.zs(enc_next_state)
                diagnostic_prediction = self._stock_prediction(
                    self.encoder, self._control_zs(diagnostic_zs), enc_action
                )
                latent_diagnostics = (
                    diagnostic_zs,
                    diagnostic_next_zs,
                    diagnostic_prediction,
                )
                if getattr(self.outcome_tokens, "isometric_targets", False):
                    _, diagnostic_world_prediction, _ = (
                        self.encoder_loss_eager._world_latents(
                            diagnostic_zs,
                            enc_action,
                            diagnostic_prediction,
                            diagnostic_next_zs,
                        )
                    )
                    isometric_outcome_diagnostics = (
                        self.outcome_tokens.inverse_diagnostics(
                            enc_action,
                            enc_reward,
                            policy_mean_target,
                            diagnostic_world_prediction,
                            enc_not_done,
                            self.encoder_loss_eager.policy_logstd_half_range,
                        )
                    )
                if self.jedi_denoiser is not None and self.rollout_loss_core is None:
                    jedi_zs = diagnostic_zs
                    jedi_next_zs = diagnostic_next_zs
                    jedi_prediction = diagnostic_prediction
                    jedi_action = enc_action
                    if self.args.prediction_from_lap:
                        jedi_zs = self.encoder.zs(state)
                        jedi_next_zs = self.encoder.zs(next_state)
                        jedi_prediction = self._stock_prediction(
                            self.encoder, self._control_zs(jedi_zs), action
                        )
                        jedi_action = action
                    jedi_diagnostics = self.encoder_loss_eager.jedi_diagnostics(
                        jedi_zs,
                        jedi_action,
                        jedi_prediction,
                        jedi_next_zs,
                        jedi_sigma_normal,
                        jedi_epsilon,
                    )
                if self.rollout_loss_eager is not None:
                    rollout_diagnostics = self.rollout_loss_eager.diagnostics(
                        diagnostic_sequence_states,
                        diagnostic_sequence_actions,
                        jedi_sigma_normal,
                        jedi_epsilon,
                        self.lewm_trunk_scale_tensor,
                    )

        self.encoder_optimizer.zero_grad(set_to_none=True)
        if self.wm_optimizer is not None:
            self.wm_optimizer.zero_grad(set_to_none=True)
        if self.reward_token_optimizer is not None:
            self.reward_token_optimizer.zero_grad(set_to_none=True)
        if self.policy_mean_optimizer is not None:
            self.policy_mean_optimizer.zero_grad(set_to_none=True)
        if self.outcome_optimizer is not None:
            self.outcome_optimizer.zero_grad(set_to_none=True)
        if self.jedi_optimizer is not None:
            self.jedi_optimizer.zero_grad(set_to_none=True)
        encoder_loss.backward()
        encoder_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.encoder_trainable_parameters, self.args.encoder_max_grad_norm
        )
        wm_grad_norm = None
        if self.wm_optimizer is not None:
            wm_parameters = [
                *self.rollout_projector.parameters(),
                *self.rollout_dynamics.parameters(),
                *self.rollout_pred_projector.parameters(),
            ]
            wm_grad_norm = torch.nn.utils.clip_grad_norm_(
                wm_parameters, self.args.encoder_max_grad_norm
            )
        reward_token_grad_norm = None
        if self.reward_token_optimizer is not None:
            if self.args.reward_sigreg_tokenizer_only:
                tokenizer_norm = torch.nn.utils.clip_grad_norm_(
                    self.reward_token_head.tokenizer.parameters(),
                    self.args.encoder_max_grad_norm,
                )
                decoder_parameters = [
                    self.reward_token_head.decoder_hidden_weight,
                    self.reward_token_head.decoder_hidden_bias,
                    self.reward_token_head.decoder_output_weight,
                    self.reward_token_head.decoder_output_bias,
                ]
                decoder_norm = torch.nn.utils.clip_grad_norm_(
                    decoder_parameters, self.args.encoder_max_grad_norm
                )
                reward_token_grad_norm = torch.maximum(tokenizer_norm, decoder_norm)
            else:
                reward_token_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.reward_token_head.parameters(), self.args.encoder_max_grad_norm
                )
        policy_mean_grad_norm = None
        if self.policy_mean_optimizer is not None:
            policy_mean_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy_mean_head.parameters(), self.args.encoder_max_grad_norm
            )
        outcome_grad_norms = None
        if self.outcome_optimizer is not None:
            if hasattr(self.outcome_tokens, "gradient_parameter_groups"):
                outcome_parameter_groups = (
                    self.outcome_tokens.gradient_parameter_groups()
                )
            else:
                outcome_parameter_groups = (
                    self.outcome_tokens.reward_tokenizer.parameters(),
                    self.outcome_tokens.reward_predictor.parameters(),
                    self.outcome_tokens.policy_tokenizer.parameters(),
                    self.outcome_tokens.policy_predictor.parameters(),
                )
            outcome_grad_norms = tuple(
                torch.nn.utils.clip_grad_norm_(
                    parameters, self.args.encoder_max_grad_norm
                )
                for parameters in outcome_parameter_groups
            )
        jedi_grad_norm = None
        if self.jedi_optimizer is not None:
            jedi_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.jedi_denoiser.parameters(), self.args.jedi_max_grad_norm
            )
            warmup = min(
                self.training_steps / max(self.args.jedi_warmup_steps, 1),
                1.0,
            )
            for group in self.jedi_optimizer.param_groups:
                group["lr"] = self.args.jedi_lr * warmup
        self.encoder_optimizer.step()
        if self.wm_optimizer is not None:
            self.wm_optimizer.step()
        if self.reward_token_optimizer is not None:
            self.reward_token_optimizer.step()
        if self.policy_mean_optimizer is not None:
            self.policy_mean_optimizer.step()
        if self.outcome_optimizer is not None:
            self.outcome_optimizer.step()
        if self.jedi_optimizer is not None:
            self.jedi_optimizer.step()
        if next_trunk_scale is not None:
            self.lewm_trunk_scale_tensor.copy_(next_trunk_scale.detach())
        if next_outcome_trunk_scales is not None:
            self.outcome_tokens.set_trunk_scales(next_outcome_trunk_scales)

        if self.args.sac_policy:
            noise = torch.randn_like(action)
        else:
            noise = (torch.randn_like(action) * self.args.target_policy_noise).clamp(
                -self.args.noise_clip, self.args.noise_clip
            )
        (
            critic_loss,
            priority,
            q_target_min,
            q_target_max,
            q_mean,
            q_current_min,
            q_current_max,
            fixed_zs,
            critic_edge_mass,
        ) = self.critic_loss_core(
            state,
            action,
            next_state,
            reward,
            not_done,
            noise,
            self.alpha_tensor,
            self.min_target_tensor,
            self.max_target_tensor,
        )
        self.observed_max_target.copy_(torch.maximum(self.observed_max_target, q_target_max))
        self.observed_min_target.copy_(torch.minimum(self.observed_min_target, q_target_min))
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.replay_buffer.update_priority(priority)

        if self.training_steps % self.args.policy_freq == 0:
            self.critic.requires_grad_(False)
            actor_updates = (
                self.args.policy_freq
                if self.args.sac_policy and self.args.sac_compensate_policy_delay
                else 1
            )
            actor_log_pi = state.new_zeros(())
            alpha_loss = state.new_zeros(())
            actor_grad_norm = state.new_zeros(())
            pc_actor_diagnostics = None
            for _ in range(actor_updates):
                if self.args.sac_policy:
                    policy_noise = torch.randn_like(action)
                elif self.args.sd_noise:
                    policy_noise = torch.randn(
                        action.shape,
                        dtype=action.dtype,
                        device=action.device,
                        generator=self.sd_actor_generator,
                    )
                else:
                    policy_noise = torch.zeros_like(action)
                if self.args.pc_actor:
                    actor_loss, actor_log_pi, pc_actor_diagnostics = (
                        self._pc_actor_update(state, fixed_zs, policy_noise)
                    )
                    actor_grad_norm = pc_actor_diagnostics["direction_norm"]
                else:
                    actor_loss, actor_log_pi = self.actor_loss_core(
                        state, fixed_zs, policy_noise, self.alpha_tensor
                    )
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_grad_norm = torch.sqrt(
                        sum(
                            (
                                parameter.grad.detach().square().sum()
                                for parameter in self.actor.parameters()
                                if parameter.grad is not None
                            ),
                            state.new_zeros(()),
                        )
                    )
                    self.actor_optimizer.step()
                if self.alpha_optimizer is not None:
                    with torch.no_grad():
                        if self.args.sac_policy:
                            _, temperature_stat, _ = self.actor.sample(
                                state, fixed_zs, torch.randn_like(action)
                            )
                        else:
                            _, temperature_log_std = self.actor.policy_stats(
                                state, fixed_zs
                            )
                            temperature_stat = temperature_log_std.sum(
                                dim=1, keepdim=True
                            )
                    if self.args.sac_policy:
                        alpha_loss = (
                            -self.log_alpha.exp()
                            * (temperature_stat + self.target_entropy)
                        ).mean()
                    else:
                        alpha_loss = sdnoise_alpha_loss(
                            self.log_alpha,
                            temperature_stat,
                            self.target_entropy,
                        )
                    self.alpha_optimizer.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    self.alpha_tensor.copy_(self.log_alpha.exp().detach())
            self.critic.requires_grad_(True)
            if self.training_steps % self.args.latent_log_freq == 0:
                self.writer.add_scalar("losses/actor_loss", actor_loss.item(), self.training_steps)
                if self.args.sac_policy:
                    self.writer.add_scalar(
                        "sac/log_pi", actor_log_pi.item(), self.training_steps
                    )
                    self.writer.add_scalar(
                        "sac/alpha", self.alpha_tensor.item(), self.training_steps
                    )
                    if self.alpha_optimizer is not None:
                        self.writer.add_scalar(
                            "sac/alpha_loss", alpha_loss.item(), self.training_steps
                        )
                    if self.args.hl_gauss_critic:
                        self.writer.add_scalar(
                            "hl_gauss/actor_grad_norm",
                            float(actor_grad_norm),
                            self.training_steps,
                        )
                elif self.args.sd_noise:
                    with torch.no_grad():
                        mean, log_std = self.actor.policy_stats(state, fixed_zs)
                        sigma = log_std.exp()
                        sqrt_two = np.sqrt(2.0)
                        clip_probability = 0.5 * torch.erfc(
                            (1.0 - mean) / (sigma * sqrt_two)
                        ) + 0.5 * torch.erfc((1.0 + mean) / (sigma * sqrt_two))
                    self.writer.add_scalar(
                        "sdnoise/entropy_proxy", actor_log_pi.item(), self.training_steps
                    )
                    self.writer.add_scalar(
                        "sdnoise/alpha", self.alpha_tensor.item(), self.training_steps
                    )
                    self.writer.add_scalar(
                        "sdnoise/sigma_mean", sigma.mean().item(), self.training_steps
                    )
                    self.writer.add_scalar(
                        "sdnoise/sigma_min", sigma.min().item(), self.training_steps
                    )
                    self.writer.add_scalar(
                        "sdnoise/sigma_max", sigma.max().item(), self.training_steps
                    )
                    self.writer.add_scalar(
                        "sdnoise/action_clip_probability",
                        clip_probability.mean().item(),
                        self.training_steps,
                    )
                    if self.alpha_optimizer is not None:
                        self.writer.add_scalar(
                            "sdnoise/alpha_loss", alpha_loss.item(), self.training_steps
                        )
                    if pc_actor_diagnostics is not None:
                        for name, value in pc_actor_diagnostics.items():
                            self.writer.add_scalar(
                                f"pc_actor/{name}", value.item(), self.training_steps
                            )

        if self.training_steps % self.args.target_update_rate == 0:
            with torch.no_grad():
                old_zs = self.fixed_encoder.zs(enc_state)
                new_zs = self.encoder.zs(enc_state)
                snapshot_jump = F.mse_loss(old_zs, new_zs)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())
            if self.fixed_jedi_denoiser is not None:
                self.fixed_jedi_denoiser_target.load_state_dict(
                    self.fixed_jedi_denoiser.state_dict()
                )
                self.fixed_jedi_denoiser.load_state_dict(self.jedi_denoiser.state_dict())
                self.fixed_jedi_control_mix_target.copy_(self.fixed_jedi_control_mix)
                self.fixed_jedi_control_mix.copy_(self.jedi_control_mix_tensor)
            self.replay_buffer.reset_max_priority()
            self.max_target_tensor.copy_(self.observed_max_target)
            self.min_target_tensor.copy_(self.observed_min_target)
            self.max_target = float(self.max_target_tensor)
            self.min_target = float(self.min_target_tensor)
            self.writer.add_scalar("latent/snapshot_jump", snapshot_jump.item(), self.training_steps)
            self._refresh_controllability_basis()

        if self.training_steps % self.args.latent_log_freq == 0:
            self.writer.add_scalar("losses/encoder_loss", encoder_loss.item(), self.training_steps)
            self.writer.add_scalar("losses/prediction_loss", prediction_loss.item(), self.training_steps)
            if self.encoder_loss_eager.dreamer_loss_normalization:
                self.writer.add_scalar(
                    "losses/normalized_prediction_loss",
                    (prediction_loss_scale * prediction_loss).item(),
                    self.training_steps,
                )
                self.writer.add_scalar(
                    "losses/prediction_loss_rms",
                    self.encoder_loss_eager.prediction_loss_normalizer.last_rms.item(),
                    self.training_steps,
                )
            self.writer.add_scalar("losses/subsig_loss", sigreg_loss.item(), self.training_steps)
            if self.jedi_denoiser is not None:
                self.writer.add_scalar("losses/jedi_loss", jedi_loss.item(), self.training_steps)
                self.writer.add_scalar(
                    "losses/weighted_jedi_loss",
                    self.jedi_weight_tensor.item() * jedi_loss.item(),
                    self.training_steps,
                )
                self.writer.add_scalar(
                    "jedi/coefficient", self.jedi_weight_tensor.item(), self.training_steps
                )
                self.writer.add_scalar(
                    "jedi/grad_norm", float(jedi_grad_norm), self.training_steps
                )
                self.writer.add_scalar(
                    "jedi/learning_rate",
                    self.jedi_optimizer.param_groups[0]["lr"],
                    self.training_steps,
                )
            if self.lewm_projector is not None:
                self.writer.add_scalar(
                    "losses/lewm_prediction_loss",
                    lewm_prediction_loss.item(),
                    self.training_steps,
                )
                self.writer.add_scalar(
                    "losses/lewm_full_sigreg_loss",
                    full_sigreg_loss.item(),
                    self.training_steps,
                )
                self.writer.add_scalar(
                    "losses/weighted_lewm_loss",
                    self.lewm_weight_tensor.item()
                    * (
                        (
                            lewm_prediction_loss_scale * lewm_prediction_loss
                        ).item()
                        + self.args.lewm_sigreg_coef * full_sigreg_loss.item()
                    ),
                    self.training_steps,
                )
                if self.encoder_loss_eager.dreamer_loss_normalization:
                    self.writer.add_scalar(
                        "losses/normalized_lewm_prediction_loss",
                        (lewm_prediction_loss_scale * lewm_prediction_loss).item(),
                        self.training_steps,
                    )
                    self.writer.add_scalar(
                        "losses/lewm_prediction_loss_rms",
                        self.encoder_loss_eager.lewm_prediction_loss_normalizer.last_rms.item(),
                        self.training_steps,
                    )
                self.writer.add_scalar(
                    "lewm/coefficient",
                    self.lewm_weight_tensor.item(),
                    self.training_steps,
                )
            if self.rollout_loss_core is not None:
                self.writer.add_scalar(
                    "lewm_rollout/trunk_scale",
                    self.lewm_trunk_scale_tensor.item(),
                    self.training_steps,
                )
                self.writer.add_scalar(
                    "lewm_rollout/wm_grad_norm",
                    float(wm_grad_norm),
                    self.training_steps,
                )
                self.writer.add_scalar(
                    "lewm_rollout/full_sigreg_loss",
                    full_sigreg_loss.item(),
                    self.training_steps,
                )
                self.writer.add_scalar(
                    "lewm_rollout/jedi_x0_mse",
                    rollout_jedi_x0_mse.item(),
                    self.training_steps,
                )
                for index, value in enumerate(rollout_horizon_mse):
                    self.writer.add_scalar(
                        f"lewm_rollout/train_mse_h{index + 1}",
                        value.item(),
                        self.training_steps,
                    )
                if rollout_diagnostics is not None:
                    for index, value in enumerate(rollout_diagnostics["rollout_mse"]):
                        copy = rollout_diagnostics["copy_mse"][index]
                        self.writer.add_scalar(
                            f"lewm_rollout/mse_over_copy_h{index + 1}",
                            (value / copy.clamp_min(1e-12)).item(),
                            self.training_steps,
                        )
                    for name in (
                        "shuffled_action_ratio",
                        "latent_rms",
                        "rollout_rms",
                    ):
                        self.writer.add_scalar(
                            f"lewm_rollout/{name}",
                            rollout_diagnostics[name].item(),
                            self.training_steps,
                        )
            if self.reward_token_head is not None:
                self.writer.add_scalar(
                    "reward_token/symlog_forward_loss",
                    reward_token_loss.item(),
                    self.training_steps,
                )
            if self.policy_mean_head is not None:
                self.writer.add_scalar(
                    "policy_mean/loss", policy_mean_loss.item(), self.training_steps
                )
                self.writer.add_scalar(
                    "policy_mean/mae", policy_mean_mae.item(), self.training_steps
                )
                self.writer.add_scalar(
                    "policy_mean/predicted_std",
                    policy_mean_std.item(),
                    self.training_steps,
                )
                self.writer.add_scalar(
                    "policy_mean/grad_norm",
                    float(policy_mean_grad_norm),
                    self.training_steps,
                )
                self.writer.add_scalar(
                    "reward_token/sigreg_loss",
                    reward_token_sigreg_loss.item(),
                    self.training_steps,
                )
                self.writer.add_scalar(
                    "reward_token/raw_total_mae",
                    reward_token_mae.item(),
                    self.training_steps,
                )
                self.writer.add_scalar(
                    "reward_token/token_std",
                    reward_token_std.item(),
                    self.training_steps,
                )
                self.writer.add_scalar(
                    "reward_token/grad_norm",
                    float(reward_token_grad_norm),
                    self.training_steps,
                )
            if self.outcome_tokens is not None:
                outcome_values = {
                    "reward_prediction_loss": outcome_reward_prediction_loss,
                    "reward_sigreg_loss": outcome_reward_sigreg_loss,
                    "policy_prediction_loss": outcome_policy_prediction_loss,
                    "policy_sigreg_loss": outcome_policy_sigreg_loss,
                    "reward_token_mae": outcome_reward_mae,
                    "policy_token_mae": outcome_policy_mae,
                    "reward_target_std": outcome_reward_target_std,
                    "policy_target_std": outcome_policy_target_std,
                    "reward_clipped_fraction": outcome_reward_clipped_fraction,
                    "reward_pred_edge_mass": outcome_reward_pred_edge_mass,
                    "reward_semantic_std": outcome_reward_semantic_std,
                    "policy_semantic_std": outcome_policy_semantic_std,
                    "reward_semantic_loss": outcome_reward_semantic_loss,
                    "policy_semantic_loss": outcome_policy_semantic_loss,
                    "reward_reconstruction_mae": outcome_reward_reconstruction_mae,
                    "policy_mean_mae": outcome_policy_mean_mae,
                    "policy_logstd_mae": outcome_policy_logstd_mae,
                }
                if getattr(self.outcome_tokens, "isometric_targets", False):
                    if getattr(self.outcome_tokens, "reward_hlgauss_ce", False):
                        outcome_values["reward_hlgauss_ce"] = outcome_values.pop(
                            "reward_prediction_loss"
                        )
                        outcome_values["reward_decode_mae"] = outcome_values.pop(
                            "reward_token_mae"
                        )
                        outcome_values["weighted_reward_hlgauss_ce"] = (
                            self.args.outcome_token_coef
                            * reward_outcome_loss_scale
                            * outcome_values["reward_hlgauss_ce"]
                        )
                        outcome_values.pop("reward_sigreg_loss")
                        outcome_values.pop("reward_semantic_loss")
                        outcome_values.pop("reward_reconstruction_mae")
                    else:
                        outcome_values["reward_inverse_mse"] = outcome_values.pop(
                            "reward_semantic_loss"
                        )
                        outcome_values["reward_inverse_decode_mae"] = outcome_values.pop(
                            "reward_reconstruction_mae"
                        )
                    outcome_values["policy_inverse_mse"] = outcome_values.pop(
                        "policy_semantic_loss"
                    )
                    outcome_values.update(isometric_outcome_diagnostics)
                if getattr(self.outcome_tokens, "policy_beta_nll", False):
                    outcome_values["policy_beta_nll"] = outcome_values.pop(
                        "policy_prediction_loss"
                    )
                    outcome_values["policy_moment_mae"] = outcome_values.pop(
                        "policy_token_mae"
                    )
                    outcome_values["policy_moment_target_std"] = outcome_values.pop(
                        "policy_target_std"
                    )
                    outcome_values["policy_beta_mean_std"] = outcome_values.pop(
                        "policy_semantic_std"
                    )
                    outcome_values.pop("policy_sigreg_loss")
                    outcome_values["weighted_policy_beta_nll"] = (
                        self.encoder_loss_eager.policy_beta_nll_coef
                        * policy_outcome_loss_scale
                        * outcome_values["policy_beta_nll"]
                    )
                if self.encoder_loss_eager.dreamer_loss_normalization:
                    outcome_values["reward_loss_rms"] = (
                        self.encoder_loss_eager.reward_outcome_loss_normalizer.last_rms
                    )
                    outcome_values["policy_loss_rms"] = (
                        self.encoder_loss_eager.policy_outcome_loss_normalizer.last_rms
                    )
                for name, value in outcome_values.items():
                    self.writer.add_scalar(
                        f"outcome_tokens/{name}", value.item(), self.training_steps
                    )
                if not getattr(self.outcome_tokens, "reward_hlgauss_ce", False):
                    self.writer.add_scalar(
                        "outcome_tokens/weighted_reward_sigreg_loss",
                        self.args.outcome_sigreg_coef
                        * outcome_reward_sigreg_loss.item(),
                        self.training_steps,
                    )
                if not getattr(
                    self.outcome_tokens, "policy_beta_nll", False
                ):
                    self.writer.add_scalar(
                        "outcome_tokens/weighted_policy_sigreg_loss",
                        self.args.outcome_sigreg_coef
                        * outcome_policy_sigreg_loss.item(),
                        self.training_steps,
                    )
                outcome_gradient_names = [
                    "reward_tokenizer_grad_norm",
                    "reward_predictor_grad_norm",
                    "policy_tokenizer_grad_norm",
                    "policy_predictor_grad_norm",
                ]
                if getattr(self.outcome_tokens, "policy_beta_nll", False):
                    outcome_gradient_names[2] = None
                if getattr(self.outcome_tokens, "reward_hlgauss_ce", False):
                    outcome_gradient_names[0] = None
                for name, value in zip(outcome_gradient_names, outcome_grad_norms):
                    if name is None:
                        continue
                    self.writer.add_scalar(
                        f"outcome_tokens/{name}", float(value), self.training_steps
                    )
            self.writer.add_scalar(
                "losses/weighted_subsig_loss",
                self.args.subsig_coef * sigreg_loss.item(),
                self.training_steps,
            )
            self.writer.add_scalar("losses/encoder_grad_norm", float(encoder_grad_norm), self.training_steps)
            self.writer.add_scalar("losses/critic_loss", critic_loss.item(), self.training_steps)
            self.writer.add_scalar("losses/q_values", q_mean.item(), self.training_steps)
            if self.args.hl_gauss_critic:
                self.writer.add_scalar(
                    "hl_gauss/edge_mass", critic_edge_mass.item(), self.training_steps
                )
                self.writer.add_scalar(
                    "hl_gauss/target_min", q_target_min.item(), self.training_steps
                )
                self.writer.add_scalar(
                    "hl_gauss/target_max", q_target_max.item(), self.training_steps
                )
                self.writer.add_scalar(
                    "hl_gauss/current_q_min", q_current_min.item(), self.training_steps
                )
                self.writer.add_scalar(
                    "hl_gauss/current_q_max", q_current_max.item(), self.training_steps
                )
            self.writer.add_scalar("charts/max_target", self.max_target, self.training_steps)
            self.writer.add_scalar("charts/min_target", self.min_target, self.training_steps)
            diagnostic_zs, diagnostic_next_zs, diagnostic_prediction = latent_diagnostics
            self._log_latent_geometry(diagnostic_zs, diagnostic_next_zs, diagnostic_prediction, enc_action)
            self._log_lewm_geometry(
                diagnostic_zs,
                enc_action,
                diagnostic_next_zs,
                diagnostic_prediction,
            )
            self._log_reward_token(
                diagnostic_zs,
                diagnostic_prediction,
                enc_action,
                enc_reward,
            )
            self._log_jedi(jedi_diagnostics)
            self._log_jedi_endpoint(state, action, next_state)
            self._log_controllability()

    def maybe_train_and_checkpoint(self, ep_timesteps, ep_return):
        self.eps_since_update += 1
        self.timesteps_since_update += ep_timesteps
        self.min_return = min(self.min_return, ep_return)
        if self.min_return < self.best_min_return:
            self.train_and_reset()
        elif self.eps_since_update == self.max_eps_before_update:
            self.best_min_return = self.min_return
            self.checkpoint_actor.load_state_dict(self.actor.state_dict())
            self.checkpoint_encoder.load_state_dict(self.fixed_encoder.state_dict())
            self.train_and_reset()

    def train_and_reset(self):
        for _ in range(self.timesteps_since_update):
            if self.training_steps == self.args.steps_before_checkpointing:
                self.best_min_return *= self.args.reset_weight
                self.max_eps_before_update = self.args.max_eps_when_checkpointing
            self.train()
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8


def evaluate(agent, eval_env, eval_eps, use_checkpoint):
    returns = np.zeros(eval_eps)
    for ep in range(eval_eps):
        state, _ = eval_env.reset()
        done = False
        while not done:
            action = agent.select_action(np.asarray(state), use_checkpoint=use_checkpoint, use_exploration=False)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            returns[ep] += reward
            done = terminated or truncated
    return returns.mean()


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "TD7 requires num_envs=1 (1:1 train/env-step ratio and episodic checkpointing)"
    if args.attached_target and not (args.use_subsig or args.use_full_obs_sigreg):
        raise ValueError(
            "attached_target requires observation SIGReg to prevent end-to-end JEPA collapse"
        )
    if args.jedi_aux and args.attached_target:
        raise ValueError("jedi_aux requires a detached future prediction target")
    if args.jedi_aux and args.jedi_sigma_data <= 0:
        raise ValueError("jedi_sigma_data must be positive")
    if args.jedi_aux and args.jedi_clamp <= 0:
        raise ValueError("jedi_clamp must be positive")
    if args.jedi_aux and args.jedi_blocks < 1:
        raise ValueError("jedi_blocks must be at least one")
    if args.jedi_endpoint_control and not args.jedi_aux:
        raise ValueError("jedi_endpoint_control requires jedi_aux")
    if args.jedi_endpoint_control and args.jedi_endpoint_mix_steps < 1:
        raise ValueError("jedi_endpoint_mix_steps must be positive")
    if args.jedi_endpoint_control and args.jedi_endpoint_diag_priors < 1:
        raise ValueError("jedi_endpoint_diag_priors must be positive")
    if args.jedi_canonical_control_latents and not args.jedi_endpoint_control:
        raise ValueError("jedi_canonical_control_latents requires jedi_endpoint_control")
    if args.jedi_exact_actor_gradients and not args.jedi_endpoint_control:
        raise ValueError("jedi_exact_actor_gradients requires jedi_endpoint_control")
    if args.lewm_projected_aux and args.attached_target:
        raise ValueError("lewm_projected_aux attaches only its isolated projector target")
    if args.lewm_private_dynamics and not args.lewm_projected_aux:
        raise ValueError("lewm_private_dynamics requires lewm_projected_aux")
    if args.lewm_projected_aux and args.jedi_endpoint_control:
        raise ValueError("lewm_projected_aux is incompatible with direct endpoint control")
    if args.lewm_projected_aux and args.lewm_sigreg_num_proj < 1:
        raise ValueError("lewm_sigreg_num_proj must be positive")
    if args.lewm_rollout_aux and args.lewm_projected_aux:
        raise ValueError("choose either lewm_rollout_aux or lewm_projected_aux")
    if args.lewm_rollout_aux and not args.jedi_aux:
        raise ValueError("lewm_rollout_aux requires jedi_aux")
    if args.lewm_rollout_aux and args.residual_predictor:
        raise ValueError("lewm_rollout_aux currently requires the StockEncoder trunk")
    if args.lewm_rollout_aux and args.jedi_endpoint_control:
        raise ValueError("lewm_rollout_aux is incompatible with direct endpoint control")
    if args.lewm_rollout_aux and args.lewm_rollout_horizon < 2:
        raise ValueError("lewm_rollout_horizon must be at least two")
    if args.lewm_rollout_aux and not 0 < args.lewm_aux_trunk_cap <= 1:
        raise ValueError("lewm_aux_trunk_cap must be in (0, 1]")
    if args.reward_token_aux and not 0 < args.reward_token_shared_scale <= 1:
        raise ValueError("reward_token_shared_scale must be in (0, 1]")
    if args.reward_token_aux and args.reward_control_cost_coef < 0:
        raise ValueError("reward_control_cost_coef must be nonnegative")
    if args.policy_mean_aux and args.policy_mean_coef <= 0:
        raise ValueError("policy_mean_coef must be positive")
    outcome_mode_count = sum(
        (
            args.lejepa_outcome_tokens,
            args.semantic_outcome_tokens,
            args.latent_outcome_tokens,
            args.isometric_outcome_tokens,
        )
    )
    if outcome_mode_count > 1:
        raise ValueError("choose one reward/policy outcome-token implementation")
    if args.policy_beta_nll and not args.isometric_outcome_tokens:
        raise ValueError("policy Beta NLL requires isometric reward outcome tokens")
    if args.reward_hlgauss_ce and not args.isometric_outcome_tokens:
        raise ValueError("direct reward HL-Gauss requires isometric outcome mode")
    if not args.reward_hlgauss_symlog and not args.reward_hlgauss_ce:
        raise ValueError("raw reward HL-Gauss support requires direct reward HL-Gauss")
    if args.dreamer_loss_normalization and not (
        0.0 <= args.loss_normalization_beta < 1.0
    ):
        raise ValueError("loss_normalization_beta must lie in [0, 1)")
    if args.dreamer_loss_normalization and (
        not np.isfinite(args.loss_normalization_eps)
        or args.loss_normalization_eps <= 0.0
    ):
        raise ValueError("loss_normalization_eps must be positive")
    if args.adaptive_outcome_grad_equalization and not (
        args.isometric_outcome_tokens
        and args.reward_hlgauss_ce
        and args.policy_beta_nll
    ):
        raise ValueError(
            "outcome gradient equalization requires direct HLG reward and policy Beta NLL"
        )
    if args.adaptive_outcome_grad_equalization and args.dreamer_loss_normalization:
        raise ValueError("choose gradient equalization or scalar loss normalization")
    if args.adaptive_outcome_grad_equalization and not (
        args.outcome_token_coef == 1.0 and args.policy_beta_nll_coef == 1.0
    ):
        raise ValueError("outcome gradient equalization requires unit outcome coefficients")
    if (
        args.adaptive_outcome_grad_equalization
        and args.outcome_grad_equalization_interval < 1
    ):
        raise ValueError("outcome_grad_equalization_interval must be positive")
    if args.policy_beta_nll and not (
        0.0 < args.policy_beta_nll_eps < 0.5
    ):
        raise ValueError("policy_beta_nll_eps must lie in (0, 0.5)")
    if args.policy_beta_nll and not (
        args.policy_beta_nll_coef == -1.0
        or np.isfinite(args.policy_beta_nll_coef)
        and args.policy_beta_nll_coef > 0.0
    ):
        raise ValueError("policy_beta_nll_coef must be positive or -1 to reuse outcome_token_coef")
    if not np.isfinite(args.policy_beta_max_precision) or args.policy_beta_max_precision < 0.0:
        raise ValueError("policy_beta_max_precision must be nonnegative")
    if 0.0 < args.policy_beta_max_precision <= 2.0 * np.log(2.0):
        raise ValueError(
            "policy_beta_max_precision must exceed the initial precision 2*log(2)"
        )
    if outcome_mode_count and (
        args.reward_token_aux or args.policy_mean_aux
    ):
        raise ValueError("outcome tokens replace the legacy reward/policy auxiliaries")
    if outcome_mode_count and args.outcome_token_coef <= 0:
        raise ValueError("outcome_token_coef must be positive")
    if (args.lejepa_outcome_tokens or args.latent_outcome_tokens) and args.outcome_sigreg_coef <= 0:
        raise ValueError("outcome_sigreg_coef must be positive")
    if (args.lejepa_outcome_tokens or args.latent_outcome_tokens) and args.outcome_policy_sigreg_num_proj < 1:
        raise ValueError("outcome_policy_sigreg_num_proj must be positive")
    if args.outcome_from_transition and not (
        args.lejepa_outcome_tokens
        or args.semantic_outcome_tokens
        or args.latent_outcome_tokens
        or args.isometric_outcome_tokens
    ):
        raise ValueError("outcome_from_transition requires outcome tokens")
    if (
        args.semantic_outcome_tokens
        or args.latent_outcome_tokens
        or args.isometric_outcome_tokens
    ) and not args.outcome_from_transition:
        raise ValueError("semantic and latent outcome tokens require transition readout")
    if (
        args.semantic_outcome_tokens
        or args.latent_outcome_tokens
        or args.isometric_outcome_tokens
    ) and not (
        args.sd_noise or args.sac_policy
    ):
        raise ValueError("semantic policy targets require a Gaussian behavior policy")
    if (
        args.semantic_outcome_tokens
        or args.latent_outcome_tokens
        or args.isometric_outcome_tokens
    ) and args.semantic_outcome_token_dim < 1:
        raise ValueError("semantic_outcome_token_dim must be positive")
    if (
        args.semantic_outcome_tokens
        or args.latent_outcome_tokens
        or args.isometric_outcome_tokens
    ) and (
        args.semantic_reward_num_bins < 3 or args.semantic_reward_num_bins % 2 != 1
    ):
        raise ValueError("semantic_reward_num_bins must be odd and at least three")
    if (
        args.semantic_outcome_tokens
        or args.latent_outcome_tokens
        or args.isometric_outcome_tokens
    ) and not (
        np.isfinite(args.semantic_reward_raw_min)
        and np.isfinite(args.semantic_reward_raw_max)
        and args.semantic_reward_raw_min < args.semantic_reward_raw_max
        and np.isclose(-args.semantic_reward_raw_min, args.semantic_reward_raw_max)
    ):
        raise ValueError("semantic reward support must be finite and symmetric")
    if (
        args.semantic_outcome_tokens
        or args.latent_outcome_tokens
        or args.isometric_outcome_tokens
    ) and (
        not np.isfinite(args.semantic_reward_sigma_ratio)
        or args.semantic_reward_sigma_ratio <= 0
    ):
        raise ValueError("semantic_reward_sigma_ratio must be positive")
    if (
        args.semantic_outcome_tokens
        or args.latent_outcome_tokens
    ) and not (
        0 < args.semantic_reward_prior_floor < 1
    ):
        raise ValueError("semantic_reward_prior_floor must be in (0, 1)")
    if args.latent_outcome_tokens and args.outcome_semantic_coef <= 0:
        raise ValueError("outcome_semantic_coef must be positive")
    if (
        args.isometric_outcome_tokens
        and not args.reward_hlgauss_ce
        and args.semantic_outcome_token_dim < args.semantic_reward_num_bins
    ):
        raise ValueError("isometric outcome width must cover the reward target dimensions")
    if args.outcome_from_transition and args.prediction_from_lap:
        raise ValueError(
            "outcome_from_transition requires uniform prediction replay so contexts and targets align"
        )
    if args.use_full_obs_sigreg and args.use_subsig:
        raise ValueError("choose either full observation SIGReg or subspace SIGReg")
    if args.sigreg_batch_size < 0 or args.sigreg_batch_size > args.batch_size:
        raise ValueError("sigreg_batch_size must be zero or at most batch_size")
    if (
        args.control_sigreg_batch_size < -1
        or args.control_sigreg_batch_size > args.batch_size
    ):
        raise ValueError(
            "control_sigreg_batch_size must be -1, zero, or at most batch_size"
        )
    if args.outcome_policy_source not in {"target", "behavior"}:
        raise ValueError("outcome_policy_source must be target or behavior")
    if args.sac_policy and args.sd_noise:
        raise ValueError("sac_policy and sd_noise are mutually exclusive Gaussian policies")
    if args.pc_actor and not args.sd_noise:
        raise ValueError("pc_actor currently requires the SDNoise actor")
    if args.pc_actor and args.jedi_endpoint_control:
        raise ValueError("pc_actor currently supports the stock SALE action path only")
    if args.pc_actor_batch_size < 0:
        raise ValueError("pc_actor_batch_size must be nonnegative")
    if args.pc_actor_inference_steps < 1:
        raise ValueError("pc_actor_inference_steps must be positive")
    if args.pc_actor_inference_scale <= 0:
        raise ValueError("pc_actor_inference_scale must be positive")
    if args.pc_actor_nudge <= 0:
        raise ValueError("pc_actor_nudge must be positive")
    if args.pc_actor_force_rms_min <= 0:
        raise ValueError("pc_actor_force_rms_min must be positive")
    if args.pc_actor_curvature_damping < 0:
        raise ValueError("pc_actor_curvature_damping must be nonnegative")
    if not 0 <= args.pc_actor_adam_beta1 < 1:
        raise ValueError("pc_actor_adam_beta1 must be in [0, 1)")
    if not 0 <= args.pc_actor_adam_beta2 < 1:
        raise ValueError("pc_actor_adam_beta2 must be in [0, 1)")
    if args.pc_actor_adam_epsilon <= 0:
        raise ValueError("pc_actor_adam_epsilon must be positive")
    if args.sac_policy and args.use_checkpoints:
        raise ValueError("SAC behavior requires per-step training; disable TD7 checkpoint gating")
    if args.sac_policy and args.controllability_exploration:
        raise ValueError("SAC policy sampling replaces external controllability exploration")
    if args.sd_noise and args.controllability_exploration:
        raise ValueError("SDNoise sampling replaces external controllability exploration")
    if args.sac_alpha <= 0:
        raise ValueError("sac_alpha must be positive")
    if args.sac_alpha_lr <= 0:
        raise ValueError("sac_alpha_lr must be positive")
    if args.sac_log_std_min >= args.sac_log_std_max:
        raise ValueError("sac_log_std_min must be below sac_log_std_max")
    if args.sd_alpha <= 0:
        raise ValueError("sd_alpha must be positive")
    if args.sd_alpha_lr <= 0:
        raise ValueError("sd_alpha_lr must be positive")
    if args.sd_log_std_min >= args.sd_log_std_max:
        raise ValueError("sd_log_std_min must be below sd_log_std_max")
    if not np.exp(args.sd_log_std_min) <= args.sd_target_sigma <= np.exp(
        args.sd_log_std_max
    ):
        raise ValueError("sd_target_sigma must lie within the configured SDNoise scale range")
    if args.hl_gauss_critic and args.hl_gauss_num_bins < 3:
        raise ValueError("hl_gauss_num_bins must be at least three")
    if args.hl_gauss_critic and args.hl_gauss_num_bins % 2 != 1:
        raise ValueError("hl_gauss_num_bins must be odd to retain an exact zero bin")
    if args.hl_gauss_critic and not (
        np.isfinite(args.hl_gauss_v_min)
        and np.isfinite(args.hl_gauss_v_max)
        and args.hl_gauss_v_min < args.hl_gauss_v_max
    ):
        raise ValueError("HL-Gauss critic support must be finite and strictly increasing")
    if args.hl_gauss_critic and not np.isclose(
        -args.hl_gauss_v_min, args.hl_gauss_v_max
    ):
        raise ValueError("HL-Gauss critic support must be symmetric around zero")
    if args.hl_gauss_critic and (
        not np.isfinite(args.hl_gauss_sigma_ratio)
        or args.hl_gauss_sigma_ratio <= 0
    ):
        raise ValueError("hl_gauss_sigma_ratio must be positive")
    if args.outcome_policy_include_log_std and not (
        args.sac_policy and args.lejepa_outcome_tokens
    ):
        raise ValueError(
            "policy log-std tokens require SAC policy and LeJEPA outcome tokens"
        )
    if args.compile_mode not in {"default", "max-autotune-no-cudagraphs"}:
        raise ValueError(
            "compile_mode must be default or max-autotune-no-cudagraphs; CUDA-graph modes are "
            "unsafe for TD7's interleaved encoder/critic/actor training regions"
        )

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
        "|param|value|\n|-|-|\n%s" % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if device.type != "cuda":
        raise RuntimeError("TD7-LeSALE research runs require CUDA")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    eval_env = gym.make(args.env_id)
    eval_env.action_space.seed(args.seed + 100)

    state_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    max_action = float(envs.single_action_space.high[0])
    agent = LeSALEAgent(state_dim, action_dim, max_action, args, device, writer)

    start_time = time.time()
    allow_train = False
    obs, _ = envs.reset(seed=args.seed)
    eval_seeded = False

    for global_step in range(args.total_timesteps + 1):
        if global_step % args.eval_freq == 0:
            if not eval_seeded:
                eval_env.reset(seed=args.seed + 100)
                eval_seeded = True
            eval_return = evaluate(agent, eval_env, args.eval_eps, use_checkpoint=args.use_checkpoints)
            writer.add_scalar("eval/episodic_return", eval_return, global_step)
            print(f"global_step={global_step}, eval_return={eval_return:.3f}")

        if args.sac_policy and global_step >= args.learning_starts:
            allow_train = True
        if allow_train:
            actions = agent.select_action(np.asarray(obs[0]))[None]
        else:
            actions = np.asarray([envs.single_action_space.sample() for _ in range(envs.num_envs)])

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        real_next_obs = next_obs[0]
        if truncations[0] or terminations[0]:
            real_next_obs = infos["final_observation"][0]
        done = float(terminations[0] and not truncations[0])
        agent.record_exploration_outcome(real_next_obs)
        agent.replay_buffer.add(
            obs[0],
            actions[0],
            real_next_obs,
            rewards[0],
            done,
            episode_boundary=bool(terminations[0] or truncations[0]),
            successor_policy_valid=not bool(terminations[0]),
        )
        obs = next_obs

        if (
            allow_train
            and not args.use_checkpoints
            and (not args.sac_policy or global_step > args.learning_starts)
        ):
            agent.train()

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is None:
                    continue
                ep_return = float(info["episode"]["r"])
                ep_length = int(info["episode"]["l"])
                print(f"global_step={global_step}, episodic_return={ep_return:.3f}")
                writer.add_scalar("charts/episodic_return", ep_return, global_step)
                writer.add_scalar("charts/episodic_length", ep_length, global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if allow_train and args.use_checkpoints:
                    agent.maybe_train_and_checkpoint(ep_length, ep_return)
                if global_step >= args.learning_starts:
                    allow_train = True
                agent.reset_exploration_direction()
                break

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        artifact = {
            "args": vars(args),
            "actor": agent.actor.state_dict(),
            "encoder": agent.encoder.state_dict(),
            "critic": agent.critic.state_dict(),
            "checkpoint_actor": agent.checkpoint_actor.state_dict(),
            "checkpoint_encoder": agent.checkpoint_encoder.state_dict(),
        }
        if agent.jedi_denoiser is not None:
            artifact["jedi_denoiser"] = agent.jedi_denoiser.state_dict()
        if agent.lewm_projector is not None:
            artifact["lewm_projector"] = agent.lewm_projector.state_dict()
            if agent.lewm_dynamics is not None:
                artifact["lewm_dynamics"] = agent.lewm_dynamics.state_dict()
            artifact["lewm_pred_projector"] = agent.lewm_pred_projector.state_dict()
        if agent.rollout_projector is not None:
            artifact["rollout_projector"] = agent.rollout_projector.state_dict()
            artifact["rollout_dynamics"] = agent.rollout_dynamics.state_dict()
            artifact["rollout_pred_projector"] = (
                agent.rollout_pred_projector.state_dict()
            )
            artifact["lewm_trunk_scale"] = agent.lewm_trunk_scale_tensor
        if agent.reward_token_head is not None:
            artifact["reward_token_head"] = agent.reward_token_head.state_dict()
        if agent.policy_mean_head is not None:
            artifact["policy_mean_head"] = agent.policy_mean_head.state_dict()
        if agent.outcome_tokens is not None:
            artifact["outcome_tokens"] = agent.outcome_tokens.state_dict()
        if agent.log_alpha is not None:
            artifact["log_alpha"] = agent.log_alpha.detach().cpu()
        if agent.fixed_jedi_denoiser is not None:
            artifact["fixed_jedi_denoiser"] = agent.fixed_jedi_denoiser.state_dict()
            artifact["fixed_jedi_denoiser_target"] = (
                agent.fixed_jedi_denoiser_target.state_dict()
            )
            artifact["jedi_endpoint_prior"] = agent.jedi_endpoint_prior
            artifact["fixed_jedi_control_mix"] = agent.fixed_jedi_control_mix
            artifact["fixed_jedi_control_mix_target"] = agent.fixed_jedi_control_mix_target
        torch.save(artifact, model_path)
        print(f"model saved to {model_path}")

    envs.close()
    eval_env.close()
    writer.close()
