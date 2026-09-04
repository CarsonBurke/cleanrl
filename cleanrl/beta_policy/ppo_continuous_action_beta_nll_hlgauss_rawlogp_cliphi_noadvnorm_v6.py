# PPO + raw-logprob DG gate clip-hi no-advnorm HL-Gauss critic v6.
#
# Variant from CleanRL's continuous-action PPO:
# - Replace the Gaussian policy with a v168-style unimodal Beta policy on native
#   z in (0, 1), then map z linearly to the environment action bounds.
# - Replay stored z during PPO updates so the behavior ratio is computed in the
#   same density space used for sampling.
# - Gate raw PPO advantages with the paper's continuous-action DG gate:
#     ell = clamp(-old_logprob, -beta_nll_clip, beta_nll_clip)
#     gate = sigmoid(gate_coef * A * ell)
#   This intentionally does not z-score advantages or rescale sigmoid output.
# - Combine the strongest early ablations from v1: no advantage normalization and
#   no beta-NLL weight min/max bounds.
# - Use IterThink-style asymmetric policy clipping:
#     max(-A*r, -A*clamp(r, 1-clip_coef, 1+clip_coef_high)).
# - Replace scalar MSE critic regression with a Dreamer3-bucket HL-Gauss critic.
#   The scalar bucket range is symmetric and derived from the normalized clipped
#   reward bound, reward_clip / (1 - gamma), then represented in symlog space.
#
# Hypothesis: bounded Beta support removes tanh/clip mismatch at action limits.
# Using the paper's raw clipped log-density surprisal should make the gate a
# closer DG comparison than peak-referenced Beta NLL.
# HL-Gauss value labels should improve critic gradients by preserving target
# locality over the actual normalized return range.
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.hl_gauss import Dreamer3BucketHLGaussSupport, symlog


SAMPLE_EPS = 1e-6


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
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
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
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_coef_high: float = 0.28
    """the looser upper surrogate clipping coefficient"""
    clip_vloss: bool = True
    """kept for CLI compatibility; HL-Gauss critic always uses categorical value loss"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    beta_nll_clip: float = 10.0
    """clip for raw log-density surprisal and legacy peak-referenced beta NLL helpers"""
    beta_nll_gate_coef: float = 1.0
    """scale for the sign-aware DG advantage gate"""
    num_bins: int = 511
    """number of HL-Gauss critic buckets; must be odd for an exact zero bucket"""
    value_sigma_to_bin_ratio: float = 0.75
    """HL-Gauss label smoothing sigma as a fraction of symlog bucket width"""
    reward_clip: float = 10.0
    """absolute clip applied after reward normalization by the environment wrapper"""
    norm_reward: bool = True
    """whether to normalize rewards and clip the normalized rewards"""
    return_abs_bound: float = 0.0
    """override absolute normalized return support; <=0 uses reward_clip / (1 - gamma)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma, reward_clip=10.0, norm_reward=True):
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
        if norm_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -reward_clip, reward_clip))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def logprob_to_gate(advantages, logprob, clip, gate_coef):
    clipped_surprisal = (-logprob).clamp(-clip, clip)
    advantages = advantages.detach()
    gate = torch.sigmoid(gate_coef * advantages * clipped_surprisal)
    return clipped_surprisal, gate.detach()


def beta_nll_to_gate(advantages, beta_nll, clip, gate_coef):
    beta_nll = beta_nll.clamp(0.0, clip)
    return beta_nll, torch.sigmoid(gate_coef * advantages.detach() * beta_nll).detach()


def normalized_return_abs_bound(args):
    if args.return_abs_bound > 0:
        return float(args.return_abs_bound)
    if not args.norm_reward:
        raise ValueError("set return_abs_bound explicitly when reward normalization is disabled")
    if args.gamma >= 1.0:
        raise ValueError("gamma must be < 1.0 when deriving the HL-Gauss return range")
    return float(args.reward_clip / (1.0 - args.gamma))


def hlgauss_coord_bounds(args):
    return_abs_bound = normalized_return_abs_bound(args)
    coord_max = float(symlog(torch.tensor(return_abs_bound, dtype=torch.float32)).item())
    return -coord_max, coord_max


def make_hlgauss_support(args, device):
    coord_min, coord_max = hlgauss_coord_bounds(args)
    return Dreamer3BucketHLGaussSupport(
        args.num_bins,
        coord_min,
        coord_max,
        args.value_sigma_to_bin_ratio,
        device,
    )


def hl_value_loss(value_logits, returns, hl_support):
    value_log_probs = torch.log_softmax(value_logits, dim=-1)
    target_probs = hl_support.project(returns)
    return -(target_probs.detach() * value_log_probs).sum(dim=-1).mean()


def clipped_policy_loss(mb_advantages, ratio, clip_coef, clip_coef_high):
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef_high)
    return torch.max(pg_loss1, pg_loss2).mean()


class Agent(nn.Module):
    def __init__(self, envs, args=None):
        super().__init__()
        if args is None:
            args = Args()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.value_num_bins = args.num_bins
        self.value_sigma_to_bin_ratio = args.value_sigma_to_bin_ratio
        self.value_return_abs_bound = normalized_return_abs_bound(args)
        self._value_support = None
        self._value_support_device = None
        self.critic_backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.critic_head = layer_init(nn.Linear(64, args.num_bins), std=0.1)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor_alpha = layer_init(nn.Linear(64, action_dim), std=0.01)
        self.actor_beta = layer_init(nn.Linear(64, action_dim), std=0.01)
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))

    def _support_args(self):
        args = Args(
            num_bins=self.value_num_bins,
            value_sigma_to_bin_ratio=self.value_sigma_to_bin_ratio,
            return_abs_bound=self.value_return_abs_bound,
        )
        return args

    def value_support(self, device):
        if self._value_support is None or self._value_support_device != device:
            self._value_support = make_hlgauss_support(self._support_args(), device)
            self._value_support_device = device
        return self._value_support

    def get_value_logits(self, x):
        return self.critic_head(self.critic_backbone(x))

    def get_value(self, x):
        logits = self.get_value_logits(x)
        return self.value_support(x.device).to_scalar(logits).unsqueeze(-1)

    def _dist(self, x):
        h = self.actor(x)
        alpha = 1.0 + F.softplus(self.actor_alpha(h))
        beta = 1.0 + F.softplus(self.actor_beta(h))
        return Beta(alpha, beta)

    def _z_to_action(self, z):
        return self.action_low + (self.action_high - self.action_low) * z

    def _action_to_z(self, action):
        z = (action - self.action_low) / (self.action_high - self.action_low)
        return z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)

    def beta_peak_nll(self, dist, z):
        alpha = dist.concentration1
        beta = dist.concentration0
        mode = ((alpha - 1.0) / (alpha + beta - 2.0).clamp_min(SAMPLE_EPS)).clamp(
            SAMPLE_EPS, 1.0 - SAMPLE_EPS
        )
        return (dist.log_prob(mode) - dist.log_prob(z)).sum(1)

    def beta_nll_gate(self, advantages, beta_nll, clip, gate_coef):
        return beta_nll_to_gate(advantages, beta_nll, clip, gate_coef)

    def logprob_gate(self, advantages, logprob, clip, gate_coef):
        return logprob_to_gate(advantages, logprob, clip, gate_coef)

    def beta_nll_weights(self, dist, z, clip, weight_min, weight_max):
        beta_nll = self.beta_peak_nll(dist, z).clamp(0.0, clip)
        beta_nll_mean = beta_nll.mean()
        weights = torch.where(
            beta_nll_mean > 1e-8,
            beta_nll / beta_nll_mean,
            torch.ones_like(beta_nll),
        )
        return beta_nll, weights.clamp(weight_min, weight_max).detach()

    def get_beta_action_and_value(self, x, z=None):
        dist = self._dist(x)
        if z is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        else:
            z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = self._z_to_action(z)
        logprob = dist.log_prob(z).sum(1)
        beta_nll = self.beta_peak_nll(dist, z)
        value_logits = self.get_value_logits(x)
        value = self.value_support(x.device).to_scalar(value_logits).unsqueeze(-1)
        return action, z, logprob, dist.entropy().sum(1), value, beta_nll, value_logits

    def get_action_and_value(self, x, action=None):
        z = None if action is None else self._action_to_z(action)
        action, _, logprob, entropy, value, _, _ = self.get_beta_action_and_value(x, z)
        return action, logprob, entropy, value


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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("ppo_continuous_action_beta_nll_hlgauss_rawlogp_cliphi_noadvnorm_v6 requires CUDA")
    device = torch.device("cuda")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                args.gamma,
                args.reward_clip,
                args.norm_reward,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    hl_support = agent.value_support(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    action_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    beta_nlls = torch.zeros((args.num_steps, args.num_envs)).to(device)
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

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, action_z, logprob, _, value, beta_nll, _ = agent.get_beta_action_and_value(next_obs)
                values[step] = value.flatten()
            action_zs[step] = action_z
            beta_nlls[step] = beta_nll
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
        b_action_zs = action_zs.reshape((-1,) + envs.single_action_space.shape)
        b_beta_nlls = beta_nlls.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        beta_nll_means = []
        raw_surprisal_means = []
        raw_surprisal_stds = []
        raw_surprisal_gate_arg_means = []
        raw_surprisal_gate_arg_stds = []
        raw_surprisal_gate_p05s = []
        raw_surprisal_gate_p50s = []
        raw_surprisal_gate_p95s = []
        beta_nll_gate_means = []
        beta_nll_gate_stds = []
        beta_concentration_means = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, newvalue, _, newvalue_logits = agent.get_beta_action_and_value(
                    b_obs[mb_inds],
                    b_action_zs[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                dist = agent._dist(b_obs[mb_inds])
                mb_advantages = b_advantages[mb_inds]
                raw_surprisal, beta_nll_gate = logprob_to_gate(
                    mb_advantages,
                    b_logprobs[mb_inds],
                    args.beta_nll_clip,
                    args.beta_nll_gate_coef,
                )
                beta_nll = b_beta_nlls[mb_inds].clamp(0.0, args.beta_nll_clip)

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        (
                            (ratio < (1.0 - args.clip_coef))
                            | (ratio > (1.0 + args.clip_coef_high))
                        )
                        .float()
                        .mean()
                        .item()
                    ]
                    beta_nll_means.append(beta_nll.mean().item())
                    gate_arg = args.beta_nll_gate_coef * mb_advantages * raw_surprisal
                    raw_surprisal_means.append(raw_surprisal.mean().item())
                    raw_surprisal_stds.append(raw_surprisal.std(unbiased=False).item())
                    raw_surprisal_gate_arg_means.append(gate_arg.mean().item())
                    raw_surprisal_gate_arg_stds.append(gate_arg.std(unbiased=False).item())
                    gate_quantiles = torch.quantile(beta_nll_gate, torch.tensor([0.05, 0.5, 0.95], device=device))
                    raw_surprisal_gate_p05s.append(gate_quantiles[0].item())
                    raw_surprisal_gate_p50s.append(gate_quantiles[1].item())
                    raw_surprisal_gate_p95s.append(gate_quantiles[2].item())
                    beta_nll_gate_means.append(beta_nll_gate.mean().item())
                    beta_nll_gate_stds.append(beta_nll_gate.std(unbiased=False).item())
                    beta_concentration_means.append(
                        (dist.concentration1 + dist.concentration0).mean().item()
                    )

                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                mb_advantages = mb_advantages * beta_nll_gate.detach()

                # Policy loss
                pg_loss = clipped_policy_loss(
                    mb_advantages,
                    ratio,
                    args.clip_coef,
                    args.clip_coef_high,
                )

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = hl_value_loss(newvalue_logits, b_returns[mb_inds], hl_support)

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

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/beta_nll_mean", np.mean(beta_nll_means), global_step)
        writer.add_scalar("losses/raw_surprisal_mean", np.mean(raw_surprisal_means), global_step)
        writer.add_scalar("losses/raw_surprisal_std", np.mean(raw_surprisal_stds), global_step)
        writer.add_scalar("losses/raw_surprisal_gate_arg_mean", np.mean(raw_surprisal_gate_arg_means), global_step)
        writer.add_scalar("losses/raw_surprisal_gate_arg_std", np.mean(raw_surprisal_gate_arg_stds), global_step)
        writer.add_scalar("losses/raw_surprisal_gate_p05", np.mean(raw_surprisal_gate_p05s), global_step)
        writer.add_scalar("losses/raw_surprisal_gate_p50", np.mean(raw_surprisal_gate_p50s), global_step)
        writer.add_scalar("losses/raw_surprisal_gate_p95", np.mean(raw_surprisal_gate_p95s), global_step)
        writer.add_scalar("losses/beta_nll_gate_p05", np.mean(raw_surprisal_gate_p05s), global_step)
        writer.add_scalar("losses/beta_nll_gate_p50", np.mean(raw_surprisal_gate_p50s), global_step)
        writer.add_scalar("losses/beta_nll_gate_p95", np.mean(raw_surprisal_gate_p95s), global_step)
        writer.add_scalar("losses/beta_nll_gate_mean", np.mean(beta_nll_gate_means), global_step)
        writer.add_scalar("losses/beta_nll_gate_std", np.mean(beta_nll_gate_stds), global_step)
        writer.add_scalar("losses/beta_concentration_mean", np.mean(beta_concentration_means), global_step)
        writer.add_scalar("losses/value_return_abs_bound", agent.value_return_abs_bound, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        def eval_make_env(env_id, idx, capture_video, run_name, gamma):
            return make_env(
                env_id,
                idx,
                capture_video,
                run_name,
                gamma,
                args.reward_clip,
                args.norm_reward,
            )

        episodic_returns = evaluate(
            model_path,
            eval_make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=lambda eval_envs: Agent(eval_envs, args),
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
