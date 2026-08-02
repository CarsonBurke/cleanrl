# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
"""
**SGD control for the IDBD streaming family** (sepclip base, IDBD → plain SGD).

Diagnostics on all 8M IDBD runs show α frozen at init (0.05) with meta_dot
~-2e-7: the meta-learner never acted, so sepclip ≡ SGD(0.05) + per-head clip.
This file makes that explicit to settle attribution:
  - optimizer: torch.optim.SGD(lr=0.05), no momentum — identical update to
    inert IDBD; everything else byte-identical to sepclip.
  - --anneal-lr now actually anneals (IDBD's flag only re-seeded β at init):
    hypothesis — the late plateau is a constant-step noise floor, and the
    per-head clip (which pins update magnitude) is the implicit advnorm.
  - logs pre-clip grad norms + clip fraction to test the clip-as-normalizer
    story directly.
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
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import HLGaussSupport

SAMPLE_EPS = 1e-6  # clamp Beta samples off open-interval boundary (avoid log(0))


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
    """SGD learning rate (matches the inert-IDBD α init of the sepclip run)"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 32
    """truncated rollout: 32-step GAE windows (not 1-step incr, not 2048 batch)"""
    anneal_lr: bool = False
    """linear LR anneal to 0 over training (actually works here, unlike IDBD's flag)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 0
    """number of mini-batches; 0 => minibatch_size = idbd_batch_size (incremental)"""
    update_epochs: int = 1
    """1 epoch for all paper-gap ablations (baseline v2 keeps 10)"""
    norm_adv: bool = False
    """no advantage normalization (raw GAE)"""
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

    # HL-Gauss critic (reward-norm returns — narrow support, no ±20k raw range)
    num_bins: int = 101
    """categorical value bins"""
    v_min: float = -20.0
    """support min on *normalized* return scale"""
    v_max: float = 20.0
    """support max on *normalized* return scale"""
    sigma_ratio: float = 0.75
    """HL-Gauss sigma as fraction of bin width (Farebrother default-ish)"""
    compile: bool = True
    """torch.compile the agent (mode=reduce-overhead) for many tiny fixed-shape opt steps"""
    compile_mode: str = "reduce-overhead"
    """torch.compile mode: default|reduce-overhead|max-autotune"""


    target_tau: float = 0.005
    """Polyak τ for critic bootstrap target (0 = online V)"""
    log_every: int = 2048
    """log grads/loss every N env-steps"""

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
        use_symlog=False,
    )
    agent = Agent(envs, args.num_bins).to(device)
    actor_params = list(agent.actor.parameters())
    critic_params = list(agent.critic.parameters())
    actor_optimizer = torch.optim.SGD(actor_params, lr=args.learning_rate)
    critic_optimizer = torch.optim.SGD(critic_params, lr=args.learning_rate)
    optimizers = [actor_optimizer, critic_optimizer]

    import copy

    online_critic = agent.critic
    critic_target = copy.deepcopy(online_critic).to(device)
    for p in critic_target.parameters():
        p.requires_grad_(False)

    if args.compile:
        agent = torch.compile(agent, mode=args.compile_mode)
        print(
            f"torch.compile + delayed GAE H={H} + separate A/C grad clip "
            f"SGD lr={args.learning_rate} anneal={args.anneal_lr} τ={args.target_tau}"
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

    # accumulated between log points: pre-clip grad norms + how often clip binds
    acc_actor_norm = acc_critic_norm = 0.0
    acc_actor_clipped = acc_critic_clipped = 0
    acc_grad_n = 0

    print(
        f"sgd_stream_v1: DELAYED GAE H={H} λ={args.gae_lambda} "
        f"every-step optim oldest (n={args.num_envs}), separate actor/critic grad clip "
        f"(max_norm={args.max_grad_norm} each), "
        f"SGD lr={args.learning_rate}, anneal={args.anneal_lr}, "
        f"τ={args.target_tau}, rewnorm, noadvnorm"
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
        # Independent per-head clips (not joint ||g_a||^2+||g_c||^2)
        if args.max_grad_norm > 0.0:
            actor_norm = nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
            critic_norm = nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
            acc_actor_norm += actor_norm.item()
            acc_critic_norm += critic_norm.item()
            acc_actor_clipped += int(actor_norm.item() > args.max_grad_norm)
            acc_critic_clipped += int(critic_norm.item() > args.max_grad_norm)
            acc_grad_n += 1
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
            gn = max(acc_grad_n, 1)
            actor_norm_mean = acc_actor_norm / gn
            critic_norm_mean = acc_critic_norm / gn
            actor_clip_frac = acc_actor_clipped / gn
            critic_clip_frac = acc_critic_clipped / gn
            acc_actor_norm = acc_critic_norm = 0.0
            acc_actor_clipped = acc_critic_clipped = 0
            acc_grad_n = 0
            writer.add_scalar("charts/learning_rate", actor_optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("grads/actor_preclip_norm", actor_norm_mean, global_step)
            writer.add_scalar("grads/critic_preclip_norm", critic_norm_mean, global_step)
            writer.add_scalar("grads/actor_clip_frac", actor_clip_frac, global_step)
            writer.add_scalar("grads/critic_clip_frac", critic_clip_frac, global_step)
            writer.add_scalar("policy/beta_mean_concentration", float(np.mean(beta_conc_means)), global_step)
            writer.add_scalar("policy/beta_entropy", float(np.mean(beta_ent_means)), global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)

            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)
            print(
                f"SPS: {sps}  lr={actor_optimizer.param_groups[0]['lr']:.2e}  "
                f"|g_a|={actor_norm_mean:.3f}(clip {actor_clip_frac:.2f})  "
                f"|g_c|={critic_norm_mean:.3f}(clip {critic_clip_frac:.2f})  "
                f"βconc={float(np.mean(beta_conc_means)):.2f}"
            )

    envs.close()
    writer.close()
