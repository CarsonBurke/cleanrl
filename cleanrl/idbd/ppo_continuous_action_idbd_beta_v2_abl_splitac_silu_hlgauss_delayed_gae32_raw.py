# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
"""
**32-step truncated GAE** + fully raw advantages (incr_raw scale stack).

- num_steps=32, gae_lambda=0.95 (not 1-step incr, not 2048)
- no NormalizeReward (clip ±10 only)
- no d3 retnorm, no mean/std advnorm
- HL-Gauss value_symlog=True, support [−10,10] in symlog

splitac IDBD + SiLU + α_init=0.05, θ=0.05, epochs=1, mb=num_envs.
"""
import math
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
from torch.distributions.beta import Beta
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import HLGaussSupport

SAMPLE_EPS = 1e-6  # clamp Beta samples off open-interval boundary (avoid log(0))


class IDBD(Optimizer):
    """Parameter-wise Incremental Delta-Bar-Delta (Sutton 1992).

    Linear LMS (paper) uses features x; here each parameter is treated as a
    unit-feature weight and the backprop gradient g = ∂L/∂w plays the role of
    -δx. Updates (elementwise):

        β  ← β - θ · g · h
        α  ← exp(β)   (clamped)
        w  ← w - α · g
        h  ← [1 - α]₊ · h - α · g

    h tracks recent weight changes Δw = -α g. Matching gradient signs grow α.

    Diagnostics (see `pop_diagnostics`):
      - α spread / cap fraction: is meta-learning differentiating step-sizes?
      - meta_dot = E[-g·h]: >0 means successive grads agree → growing α (healthy)
      - h_abs: trace alive? near-0 means no memory / no meta signal
      - effective_step = E[α|g|]: actual weight-update magnitude
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        meta_lr: float = 0.05,
        max_alpha: float = 0.1,
        eps: float = 1e-8,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr (initial alpha): {lr}")
        if meta_lr < 0.0:
            raise ValueError(f"Invalid meta_lr: {meta_lr}")
        if max_alpha <= 0.0:
            raise ValueError(f"Invalid max_alpha: {max_alpha}")
        defaults = dict(lr=lr, meta_lr=meta_lr, max_alpha=max_alpha, eps=eps)
        super().__init__(params, defaults)
        self._reset_step_accum()

    def _reset_step_accum(self):
        self._acc = {
            "n": 0,
            "meta_dot": 0.0,  # sum of (-g*h); positive => grow α
            "meta_abs": 0.0,  # sum of |g*h|
            "h_abs": 0.0,
            "eff_step": 0.0,  # sum of α|g|
            "sign_agree": 0.0,  # sum of 1[sign(g)==sign(-h)] i.e. g opposes h? wait
            # sign agreement for *growing* α: sign(-g)==sign(h) i.e. g and h opposite
            # since Δw=-αg, h~Δw; same direction of weight updates means g same sign as previous g
            # meta_signal = -g*h; agree when -g*h > 0
            "grow_frac": 0.0,  # fraction of params with -g*h > 0 this step (sum of counts)
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            meta_lr = group["meta_lr"]
            max_alpha = group["max_alpha"]
            init_beta = math.log(group["lr"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("IDBD does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["beta"] = torch.full_like(p, init_beta)
                    state["h"] = torch.zeros_like(p)

                beta = state["beta"]
                h = state["h"]

                # Meta-learning signal *before* updating h (uses current memory)
                # meta_signal_i = -g_i h_i; E[meta_signal]>0 ⇒ α growing
                meta_signal = -(grad * h)
                n = grad.numel()
                self._acc["n"] += n
                self._acc["meta_dot"] += meta_signal.sum().item()
                self._acc["meta_abs"] += meta_signal.abs().sum().item()
                self._acc["h_abs"] += h.abs().sum().item()
                self._acc["grow_frac"] += (meta_signal > 0).sum().item()

                # Meta-update log step-sizes: β ← β - θ g h = β + θ · meta_signal
                beta.addcmul_(grad, h, value=-meta_lr)
                beta.clamp_(max=math.log(max_alpha))
                alpha = beta.exp()

                # Weight update: w ← w - α g  (true Δw = -α g)
                step = alpha * grad
                self._acc["eff_step"] += step.abs().sum().item()
                p.add_(step, alpha=-1.0)

                # Trace of recent updates: h ← [1-α]₊ h + Δw
                decay = (1.0 - alpha).clamp_(min=0.0)
                h.mul_(decay).sub_(step)

        return loss

    @torch.no_grad()
    def pop_diagnostics(self):
        """Snapshot α state + flush per-step meta accumulators (call once per PPO iter)."""
        alphas = []
        h_vals = []
        init_alphas = []
        max_alphas = []
        for group in self.param_groups:
            max_alpha = group["max_alpha"]
            init_a = group["lr"]
            for p in group["params"]:
                state = self.state[p]
                if "beta" not in state:
                    continue
                a = state["beta"].exp().clamp(max=max_alpha).flatten()
                alphas.append(a)
                h_vals.append(state["h"].flatten())
                init_alphas.append(torch.full_like(a, init_a))
                max_alphas.append(torch.full_like(a, max_alpha))

        out = {
            "alpha_mean": 0.0,
            "alpha_median": 0.0,
            "alpha_std": 0.0,
            "alpha_p10": 0.0,
            "alpha_p90": 0.0,
            "alpha_min": 0.0,
            "alpha_max": 0.0,
            "alpha_log_mean": 0.0,
            "frac_at_max": 0.0,
            "frac_above_init": 0.0,
            "frac_below_init": 0.0,
            "alpha_vs_init_ratio": 1.0,
            "h_abs_mean": 0.0,
            "meta_dot_mean": 0.0,
            "meta_abs_mean": 0.0,
            "grow_frac": 0.0,
            "eff_step_mean": 0.0,
        }
        if alphas:
            a = torch.cat(alphas)
            a0 = torch.cat(init_alphas)
            amax = torch.cat(max_alphas)
            hcat = torch.cat(h_vals)
            out["alpha_mean"] = a.mean().item()
            out["alpha_median"] = a.median().item()
            out["alpha_std"] = a.std(unbiased=False).item()
            out["alpha_p10"] = a.quantile(0.1).item()
            out["alpha_p90"] = a.quantile(0.9).item()
            out["alpha_min"] = a.min().item()
            out["alpha_max"] = a.max().item()
            out["alpha_log_mean"] = a.clamp_min(1e-12).log().mean().exp().item()  # geom mean
            out["frac_at_max"] = (a >= amax * 0.99).float().mean().item()
            out["frac_above_init"] = (a > a0 * 1.05).float().mean().item()
            out["frac_below_init"] = (a < a0 * 0.95).float().mean().item()
            out["alpha_vs_init_ratio"] = (a.mean() / (a0.mean() + 1e-12)).item()
            out["h_abs_mean"] = hcat.abs().mean().item()

        n = max(self._acc["n"], 1)
        out["meta_dot_mean"] = self._acc["meta_dot"] / n
        out["meta_abs_mean"] = self._acc["meta_abs"] / n
        out["grow_frac"] = self._acc["grow_frac"] / n
        out["eff_step_mean"] = self._acc["eff_step"] / n
        # h_abs from accum is average over steps; prefer current state above
        if out["h_abs_mean"] == 0.0 and self._acc["n"] > 0:
            out["h_abs_mean"] = self._acc["h_abs"] / n

        self._reset_step_accum()
        return out


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
    """the entity (team of wandb's project)"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 0.05
    """initial per-parameter α = exp(β); Sutton 1992 Exp.1–2 use α_i = 0.05"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """LR annealing (off: IDBD adapts α; if on, re-inits β toward annealed α)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 0
    """number of mini-batches; 0 => minibatch_size = idbd_batch_size (incremental)"""
    update_epochs: int = 1
    """1 epoch for all paper-gap ablations (baseline v2 keeps 10)"""
    norm_adv: bool = False
    """no per-batch mean/std advantage normalization"""
    # DreamerV3-style return percentile scale (retnorm, impl=perc)
    ret_percnorm: bool = False
    """off: raw GAE advantages (no d3 retnorm)"""
    ret_perc_rate: float = 0.01
    """EMA rate on return percentiles (DreamerV3 default)"""
    ret_perc_lo: float = 0.05
    """lower percentile (P5)"""
    ret_perc_hi: float = 0.95
    """upper percentile (P95)"""
    ret_perc_floor: float = 1.0
    """minimum scale S (DreamerV3 retnorm limit)"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """value clipping for HL-Gauss (off by default; CE on projected returns)"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # IDBD-specific
    idbd_meta_lr: float = 0.05
    """meta step-size θ (linear demo 0.05; paper long-run fig uses 0.001)"""
    idbd_max_alpha: float = 1.0
    """hard cap on α; paper optimal relevant α ~0.13, fixed-α sweep up to 0.25"""
    idbd_batch_size: int = 0
    """optimizer batch size; 0 => num_envs (batch size 1 × env count). Set 1 for pure sample-wise."""
    # HL-Gauss critic (reward-norm returns — narrow support, no ±20k raw range)
    num_bins: int = 101
    """categorical value bins"""
    v_min: float = -10.0
    """HL-Gauss support min in **symlog** coords (raw returns; ~symlog(-2e4))"""
    v_max: float = 10.0
    """HL-Gauss support max in **symlog** coords (~symlog(+2e4))"""
    sigma_ratio: float = 0.75
    """HL-Gauss sigma as fraction of bin width"""
    value_symlog: bool = True
    """symlog HL-Gauss for raw (non-reward-normalized) returns"""
    compile: bool = True
    """torch.compile the agent (mode=reduce-overhead) for many tiny fixed-shape opt steps"""
    compile_mode: str = "reduce-overhead"
    """torch.compile mode: default|reduce-overhead|max-autotune"""


    target_tau: float = 0.005
    """Polyak τ for critic bootstrap target (0 = online V)"""
    log_every: int = 2048
    """log IDBD/loss every N env-steps"""
    idbd_meta_lr_stream: float = 0.005
    """θ for correlated delayed stream (0 = use idbd_meta_lr)"""

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
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # no NormalizeReward — raw rewards (per-step clip only)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class BetaActor(nn.Module):
    """Trunk + concentration heads as one module (friendly to reduce-overhead CUDA graphs)."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.SiLU(),
            layer_init(nn.Linear(64, 64)),
            nn.SiLU(),
        )
        self.alpha_head = layer_init(nn.Linear(64, act_dim), std=0.01)
        self.beta_head = layer_init(nn.Linear(64, act_dim), std=0.01)

    def forward(self, x):
        # dreamer4 unimodal Beta: concentration = 1 + softplus >= 1
        feat = self.trunk(x)
        alpha = 1.0 + F.softplus(self.alpha_head(feat))
        beta = 1.0 + F.softplus(self.beta_head(feat))
        return alpha, beta


class Agent(nn.Module):
    """Unimodal Beta actor + HL-Gauss categorical critic (SiLU activations)."""

    def __init__(self, envs, num_bins: int):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        self.act_dim = act_dim
        self.num_bins = num_bins
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.SiLU(),
            layer_init(nn.Linear(64, 64)),
            nn.SiLU(),
            layer_init(nn.Linear(64, num_bins), std=1.0),
        )
        self.actor = BetaActor(obs_dim, act_dim)
        self.register_buffer(
            "action_low",
            torch.as_tensor(envs.single_action_space.low, dtype=torch.float32),
        )
        self.register_buffer(
            "action_high",
            torch.as_tensor(envs.single_action_space.high, dtype=torch.float32),
        )

    def get_value_logits(self, x):
        return self.critic(x)

    def get_value(self, x, hl_support: HLGaussSupport):
        return hl_support.to_scalar(self.critic(x))

    def _actor_dist(self, x):
        alpha, beta = self.actor(x)
        return Beta(alpha, beta)

    def _z_to_action(self, z):
        return self.action_low + (self.action_high - self.action_low) * z

    def _action_to_z(self, action):
        scale = self.action_high - self.action_low
        return ((action - self.action_low) / scale).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)

    def get_action_and_value(self, x, hl_support: HLGaussSupport, action=None, z=None):
        """
        z is the distribution-native sample in (0,1). Rollout stores z and
        replays it for log_prob (v215). Value is E[bin] from HL-Gauss critic.
        """
        dist = self._actor_dist(x)
        if z is None:
            if action is None:
                z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
                action = self._z_to_action(z)
            else:
                z = self._action_to_z(action)
        else:
            z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            if action is None:
                action = self._z_to_action(z)
        log_prob = dist.log_prob(z).sum(1)
        entropy = dist.entropy().sum(1)
        value = hl_support.to_scalar(self.critic(x))
        return action, z, log_prob, entropy, value, dist



if __name__ == "__main__":
    args = tyro.cli(Args)
    H = args.num_steps
    # Recommended lower θ for correlated stream; 0 keeps Args default
    if args.idbd_meta_lr_stream > 0:
        args.idbd_meta_lr = args.idbd_meta_lr_stream
    args.minibatch_size = int(args.num_envs)
    args.batch_size = int(args.num_envs)
    args.num_iterations = args.total_timesteps // args.num_envs
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
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
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

    hl_support = HLGaussSupport(
        args.num_bins,
        args.v_min,
        args.v_max,
        args.sigma_ratio,
        device,
        use_symlog=args.value_symlog,
    )
    agent = Agent(envs, args.num_bins).to(device)
    actor_params = list(agent.actor.parameters())
    critic_params = list(agent.critic.parameters())
    actor_optimizer = IDBD(
        actor_params,
        lr=args.learning_rate,
        meta_lr=args.idbd_meta_lr,
        max_alpha=args.idbd_max_alpha,
    )
    critic_optimizer = IDBD(
        critic_params,
        lr=args.learning_rate,
        meta_lr=args.idbd_meta_lr,
        max_alpha=args.idbd_max_alpha,
    )
    optimizers = [actor_optimizer, critic_optimizer]

    import copy

    online_critic = agent.critic
    critic_target = copy.deepcopy(online_critic).to(device)
    for p in critic_target.parameters():
        p.requires_grad_(False)

    if args.compile:
        agent = torch.compile(agent, mode=args.compile_mode)
        print(
            f"torch.compile + delayed mature GAE H={H} λ={args.gae_lambda} "
            f"meta_lr={args.idbd_meta_lr} τ={args.target_tau}"
        )

    obs_dim = envs.single_observation_space.shape
    act_dim = envs.single_action_space.shape
    buf_obs = torch.zeros((H, args.num_envs) + obs_dim, device=device)
    buf_zs = torch.zeros((H, args.num_envs) + act_dim, device=device)
    buf_logprobs = torch.zeros((H, args.num_envs), device=device)
    buf_rewards = torch.zeros((H, args.num_envs), device=device)
    buf_dones = torch.zeros((H, args.num_envs), device=device)
    buf_values = torch.zeros((H, args.num_envs), device=device)
    filled = 0

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
    next_done = torch.zeros(args.num_envs, device=device)
    ema_ret_lo, ema_ret_hi = 0.0, 1.0
    ema_perc_inited = False
    ret_perc_scale = 1.0

    v_loss = pg_loss = entropy_loss = old_approx_kl = approx_kl = torch.zeros((), device=device)
    clipfracs = [0.0]
    beta_conc_means = [1.0]
    beta_ent_means = [0.0]
    explained_var = 0.0

    print(
        f"splitac_silu_hlgauss_delayed_gae32_raw: DELAYED mature GAE H={H} λ={args.gae_lambda} "
        f"every-step optim on oldest only (n={args.num_envs}), "
        f"meta_lr={args.idbd_meta_lr}, α0={args.learning_rate}, τ={args.target_tau}, RAW (no rewnorm/retnorm/advnorm), HL symlog"
    )

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = lrnow

        # done flag of current observation (CleanRL: stored before step)
        done_t = next_done
        obs_t = next_obs

        with torch.no_grad():
            action, z, logprob, _, value, dist = agent.get_action_and_value(obs_t, hl_support)
            value = value.view(-1)

        next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        next_done_np = np.logical_or(terminations, truncations)
        reward_t = torch.as_tensor(reward, device=device, dtype=torch.float32).view(-1)
        next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
        next_done = torch.as_tensor(next_done_np, device=device, dtype=torch.float32)
        global_step += args.num_envs

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # append to chronological buffer (oldest at 0)
        if filled < H:
            idx = filled
            filled += 1
        else:
            buf_obs[:-1].copy_(buf_obs[1:].clone())
            buf_zs[:-1].copy_(buf_zs[1:].clone())
            buf_logprobs[:-1].copy_(buf_logprobs[1:].clone())
            buf_rewards[:-1].copy_(buf_rewards[1:].clone())
            buf_dones[:-1].copy_(buf_dones[1:].clone())
            buf_values[:-1].copy_(buf_values[1:].clone())
            idx = H - 1

        buf_obs[idx] = obs_t
        buf_zs[idx] = z
        buf_logprobs[idx] = logprob
        buf_rewards[idx] = reward_t
        buf_dones[idx] = done_t
        buf_values[idx] = value

        # need full horizon before mature sample exists
        if filled < H:
            continue

        # GAE over window; oldest index 0 has full H-step multi-step path + bootstrap
        with torch.no_grad():
            if args.target_tau > 0:
                next_value = hl_support.to_scalar(critic_target(next_obs)).view(-1)
            else:
                next_value = agent.get_value(next_obs, hl_support).view(-1)
            advantages = torch.zeros_like(buf_rewards)
            lastgaelam = 0.0
            for t in reversed(range(H)):
                if t == H - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - buf_dones[t + 1]
                    nextvalues = buf_values[t + 1]
                delta = buf_rewards[t] + args.gamma * nextvalues * nextnonterminal - buf_values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + buf_values

            # mature = oldest only
            mb_adv = advantages[0].clone()
            mb_ret = returns[0].clone()
            mb_val = buf_values[0].clone()
            mb_obs = buf_obs[0]
            mb_zs = buf_zs[0]
            mb_logp = buf_logprobs[0]


        if args.ret_percnorm:
            # EMA scale from mature returns only (small n=num_envs; floor=1 stabilizes)
            qs = torch.tensor([args.ret_perc_lo, args.ret_perc_hi], device=device)
            lo, hi = torch.quantile(mb_ret.detach(), qs).tolist()
            if not ema_perc_inited:
                ema_ret_lo, ema_ret_hi, ema_perc_inited = lo, hi, True
            else:
                r = args.ret_perc_rate
                ema_ret_lo += r * (lo - ema_ret_lo)
                ema_ret_hi += r * (hi - ema_ret_hi)
            ret_perc_scale = max(args.ret_perc_floor, ema_ret_hi - ema_ret_lo)
            mb_adv = mb_adv / ret_perc_scale


        # single optim step on mature batch (num_envs), time-ordered, no shuffle
        _, _, newlogprob, entropy, newvalue, dist = agent.get_action_and_value(
            mb_obs, hl_support, z=mb_zs
        )
        logratio = newlogprob - mb_logp
        ratio = logratio.exp()
        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs = [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
            beta_conc_means = [
                (0.5 * (dist.concentration1 + dist.concentration0)).mean().item()
            ]
            beta_ent_means = [entropy.mean().item()]

        if args.norm_adv and mb_adv.numel() > 1:
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

        # single-use: A2C score is natural; keep PPO clip for continuity (ratio≈1 first use)
        pg_loss1 = -mb_adv * ratio
        pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        newvalue_logits = agent.get_value_logits(mb_obs)
        target_probs = hl_support.project(mb_ret)
        log_probs = torch.log_softmax(newvalue_logits, dim=-1)
        v_loss = -(target_probs * log_probs).sum(dim=-1).mean()
        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        for opt in optimizers:
            opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(actor_params + critic_params, args.max_grad_norm)
        for opt in optimizers:
            opt.step()

        if args.target_tau > 0:
            with torch.no_grad():
                for p, tp in zip(online_critic.parameters(), critic_target.parameters()):
                    tp.mul_(1.0 - args.target_tau).add_(p, alpha=args.target_tau)

        if global_step % args.log_every < args.num_envs or iteration == args.num_iterations:
            y_pred = mb_val.detach().cpu().numpy()
            y_true = mb_ret.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            idbd_a = actor_optimizer.pop_diagnostics()
            idbd_c = critic_optimizer.pop_diagnostics()
            idbd = idbd_a
            writer.add_scalar("charts/learning_rate", actor_optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("idbd/alpha_mean", idbd["alpha_mean"], global_step)
            writer.add_scalar("idbd/alpha_median", idbd["alpha_median"], global_step)
            writer.add_scalar("idbd/alpha_std", idbd["alpha_std"], global_step)
            writer.add_scalar("idbd/alpha_p10", idbd["alpha_p10"], global_step)
            writer.add_scalar("idbd/alpha_p90", idbd["alpha_p90"], global_step)
            writer.add_scalar("idbd/alpha_min", idbd["alpha_min"], global_step)
            writer.add_scalar("idbd/alpha_max", idbd["alpha_max"], global_step)
            writer.add_scalar("idbd/alpha_geom_mean", idbd["alpha_log_mean"], global_step)
            writer.add_scalar("idbd/alpha_vs_init_ratio", idbd["alpha_vs_init_ratio"], global_step)
            writer.add_scalar("idbd/frac_at_max", idbd["frac_at_max"], global_step)
            writer.add_scalar("idbd/frac_above_init", idbd["frac_above_init"], global_step)
            writer.add_scalar("idbd/frac_below_init", idbd["frac_below_init"], global_step)
            writer.add_scalar("idbd/meta_dot_mean", idbd["meta_dot_mean"], global_step)
            writer.add_scalar("idbd/meta_abs_mean", idbd["meta_abs_mean"], global_step)
            writer.add_scalar("idbd/grow_frac", idbd["grow_frac"], global_step)
            writer.add_scalar("idbd/h_abs_mean", idbd["h_abs_mean"], global_step)
            writer.add_scalar("idbd/eff_step_mean", idbd["eff_step_mean"], global_step)
            writer.add_scalar("idbd/critic_alpha_mean", idbd_c["alpha_mean"], global_step)
            writer.add_scalar("idbd/critic_meta_dot_mean", idbd_c["meta_dot_mean"], global_step)
            writer.add_scalar("policy/beta_mean_concentration", float(np.mean(beta_conc_means)), global_step)
            writer.add_scalar("policy/beta_entropy", float(np.mean(beta_ent_means)), global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)

            writer.add_scalar("charts/ret_perc_scale", ret_perc_scale, global_step)
            writer.add_scalar("charts/ret_perc_lo", ema_ret_lo, global_step)
            writer.add_scalar("charts/ret_perc_hi", ema_ret_hi, global_step)

            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)
            print(
                f"SPS: {sps}  "
                f"α={idbd['alpha_mean']:.2e}(p10={idbd['alpha_p10']:.2e},p90={idbd['alpha_p90']:.2e})  "
                f"grow={idbd['grow_frac']:.2f} meta={idbd['meta_dot_mean']:.2e} "
                f"cap={idbd['frac_at_max']:.2f} βconc={float(np.mean(beta_conc_means)):.2f}"
            )

    envs.close()
    writer.close()
