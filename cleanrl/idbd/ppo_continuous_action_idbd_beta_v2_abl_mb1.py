# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
"""
PPO + IDBD + unimodal Beta action policy (v2).

v1 used α_init = 3e-4 (CleanRL Adam default). Sutton (1992) Exp. 1–2 sets
β so that **α_i = 0.05 for all i** at init; relevant α later ~0.13. v2 matches
that paper init and raises the α cap so the paper optimum is not hard-clamped.

Unimodal Beta (v215 / dreamer4): concentrations = 1 + softplus; z ~ Beta; action
affine of z. Optional whole-agent torch.compile(reduce-overhead).

Paper: cleanrl/idbd/sutton_1992_idbd.pdf
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
    num_steps: int = 2048
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
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
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
    idbd_batch_size: int = 1
    """optimizer batch size = 1 (paper-style single-example updates)"""
    compile: bool = True
    """torch.compile the agent (mode=reduce-overhead) for many tiny fixed-shape opt steps"""
    compile_mode: str = "reduce-overhead"
    """torch.compile mode: default|reduce-overhead|max-autotune"""

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
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
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
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
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
    """MLP critic + unimodal Beta actor (v215 / dreamer4 style)."""

    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        self.act_dim = act_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
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

    def get_value(self, x):
        return self.critic(x)

    def _actor_dist(self, x):
        alpha, beta = self.actor(x)
        return Beta(alpha, beta)

    def _z_to_action(self, z):
        return self.action_low + (self.action_high - self.action_low) * z

    def _action_to_z(self, action):
        scale = self.action_high - self.action_low
        return ((action - self.action_low) / scale).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)

    def get_action_and_value(self, x, action=None, z=None):
        """
        z is the distribution-native sample in (0,1). Rollout stores z and
        replays it for log_prob (v215). If only action is given, invert the
        affine map (constant Jacobian → drops out of the PPO ratio).
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
        # log_det of affine is constant log(high-low); cancels in PPO ratio
        log_prob = dist.log_prob(z).sum(1)
        entropy = dist.entropy().sum(1)
        return action, z, log_prob, entropy, self.critic(x), dist


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    # Incremental IDBD: default minibatch = num_envs ("batch size 1 × env count")
    if args.idbd_batch_size <= 0:
        args.idbd_batch_size = args.num_envs
    args.minibatch_size = int(args.idbd_batch_size)
    if args.num_minibatches <= 0:
        args.num_minibatches = max(1, args.batch_size // args.minibatch_size)
    else:
        # honor explicit num_minibatches (overrides idbd_batch_size)
        args.minibatch_size = max(1, args.batch_size // args.num_minibatches)
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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    if args.compile:
        # Whole agent; Dynamo graph-breaks on Beta sample/log_prob (fine, fullgraph=False).
        # No multi-pass warmup — first real step compiles (tiny MLP, seconds not minutes).
        agent = torch.compile(agent, mode=args.compile_mode)
        print(f"torch.compile(agent, mode={args.compile_mode!r})")
    optimizer = IDBD(
        agent.parameters(),
        lr=args.learning_rate,
        meta_lr=args.idbd_meta_lr,
        max_alpha=args.idbd_max_alpha,
    )

    # ALGO Logic: Storage setup (z = Beta-native sample in (0,1), for exact replay)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    print(
        f"IDBD Beta v2 abl_mb1: minibatch_size={args.minibatch_size} "
        f"(~{args.batch_size // args.minibatch_size} steps/epoch × {args.update_epochs} epochs), "
        f"meta_lr={args.idbd_meta_lr}, init_alpha={args.learning_rate} (paper α=0.05)"
    )

    for iteration in range(1, args.num_iterations + 1):
        # Optional global scale on initial α (does not rewrite adapted β unless anneal)
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            for group in optimizer.param_groups:
                group["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic (store native Beta z for PPO replay)
            with torch.no_grad():
                action, z, logprob, _, value, dist = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            zs[step] = z
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
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

        # bootstrap value if not done
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

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_zs = zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network (many small IDBD steps)
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        beta_conc_means = []
        beta_ent_means = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, newvalue, dist = agent.get_action_and_value(
                    b_obs[mb_inds], z=b_zs[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                    # Beta concentration diagnostics (mean over dims/batch)
                    beta_conc_means.append(
                        (0.5 * (dist.concentration1 + dist.concentration0)).mean().item()
                    )
                    beta_ent_means.append(entropy.mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    # with mb size 1, std is undefined — fall back to batch-level or skip
                    if mb_advantages.numel() > 1:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    else:
                        # single sample: center with full-batch stats (still scale advantages)
                        mb_advantages = (mb_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
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
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        idbd = optimizer.pop_diagnostics()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # IDBD health: α distribution
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
        # IDBD health: meta-learning signal (this update phase)
        writer.add_scalar("idbd/meta_dot_mean", idbd["meta_dot_mean"], global_step)
        writer.add_scalar("idbd/meta_abs_mean", idbd["meta_abs_mean"], global_step)
        writer.add_scalar("idbd/grow_frac", idbd["grow_frac"], global_step)
        writer.add_scalar("idbd/h_abs_mean", idbd["h_abs_mean"], global_step)
        writer.add_scalar("idbd/eff_step_mean", idbd["eff_step_mean"], global_step)
        writer.add_scalar("policy/beta_mean_concentration", float(np.mean(beta_conc_means)), global_step)
        writer.add_scalar("policy/beta_entropy", float(np.mean(beta_ent_means)), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print(
            f"SPS: {int(global_step / (time.time() - start_time))}  "
            f"α={idbd['alpha_mean']:.2e}(p10={idbd['alpha_p10']:.2e},p90={idbd['alpha_p90']:.2e})  "
            f"grow={idbd['grow_frac']:.2f} meta={idbd['meta_dot_mean']:.2e} "
            f"cap={idbd['frac_at_max']:.2f} βconc={float(np.mean(beta_conc_means)):.2f}"
        )
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
