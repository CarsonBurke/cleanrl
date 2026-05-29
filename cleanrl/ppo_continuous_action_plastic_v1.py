# PPO + Plastic v1: ReDo-style selective plasticity on top of the v19 winner.
#
# v19 (v15 architecture, FIRE disabled) was the new HC best at final20=4715,
# peak=5044. The FIRE ablations (v18 1-shock vs v19 0-shock) showed each
# blanket NS+rescale shock cost ~300-400 points off the eventual peak by
# forcing the productive weights to recover from scratch. The architectural
# win (tanh-squashed Gaussian + ReLU^2 trunk + RMSNorm pre-activation) does
# all the work; blanket shocks are a tax.
#
# Hypothesis: targeted plasticity beats blanket shocks. Most neurons are
# productive and should be left alone; only the consistently underutilised
# ones should be reinitialised. This is the ReDo procedure (Sokar et al,
# ICML 2023, "The Dormant Neuron Phenomenon in Deep RL").
#
# Per ReLU^2 layer, after each rollout we measure mean |h_i| over the
# rollout batch. Normalise: s_i = mean|h_i| / (1/H sum_j mean|h_j|). If
# s_i <= tau, neuron i is dormant. Reset:
#   - incoming Linear row -> orthogonal re-init at original gain
#   - incoming bias -> 0
#   - any RMSNorm gain on that channel -> 1
#   - next Linear column -> 0 (so the freshly reset neuron does not
#     immediately disrupt the next layer)
#   - Adam first/second moments for affected weight slices -> 0
# Productive neurons are untouched, including the policy-critical std=0.01
# actor head (which was the entire reason for fire_skip in earlier variants).
#
# Architecture unchanged from v19. No FIRE code path remains.
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
from math import log
from torch.distributions.normal import Normal
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

    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 8000000
    learning_rate: float = 3e-4
    num_envs: int = 16
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

    redo_every_iters: int = 5
    """run dormant-neuron detection + reset every N PPO iterations"""
    redo_tau: float = 0.025
    """neuron i is dormant if mean|h_i| / mean_j(mean|h_j|) <= tau"""

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
    layer._init_gain = float(std)
    return layer


class ReLUSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).pow(2)


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            ReLUSquared(),
            layer_init(nn.Linear(64, 64)),
            ReLUSquared(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            ReLUSquared(),
            layer_init(nn.Linear(64, 64)),
            nn.RMSNorm(64),
            ReLUSquared(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, z=None):
        """z is the pre-tanh latent. None => sample fresh; else re-evaluate at stored z."""
        mean = self.actor_mean(x)
        std = self.actor_logstd.expand_as(mean).exp()
        probs = Normal(mean, std)
        if z is None:
            z = probs.sample()
        action = torch.tanh(z)
        log_det = 2.0 * (log(2.0) - z - nn.functional.softplus(-2.0 * z))
        log_prob = (probs.log_prob(z) - log_det).sum(1)
        return action, z, log_prob, probs.entropy().sum(1), self.critic(x)


def _layer_triples(seq: nn.Sequential):
    """For each ReLUSquared in `seq`, yield (relusq_idx, incoming_linear,
    optional rmsnorm_chunk, next_linear). The rmsnorm chunk is any RMSNorm
    sitting between the incoming Linear and the ReLUSquared, which scales
    the same per-channel feature and should also be reset when the
    upstream neuron is dormant. next_linear is the next nn.Linear after
    the ReLUSquared (may have other ReLUSquared/RMSNorm in between for
    the v15 layout — we just take the next Linear)."""
    modules = list(seq)
    triples = []
    for i, m in enumerate(modules):
        if not isinstance(m, ReLUSquared):
            continue
        in_linear = None
        rms = None
        for j in range(i - 1, -1, -1):
            mj = modules[j]
            if isinstance(mj, nn.RMSNorm) and rms is None:
                rms = mj
            elif isinstance(mj, nn.Linear):
                in_linear = mj
                break
        next_linear = None
        for j in range(i + 1, len(modules)):
            if isinstance(modules[j], nn.Linear):
                next_linear = modules[j]
                break
        if in_linear is None:
            continue
        triples.append((i, in_linear, rms, next_linear))
    return triples


def _zero_adam_state(optimizer: optim.Optimizer, param: torch.Tensor, *, rows=None, cols=None):
    state = optimizer.state.get(param, None)
    if not state:
        return
    for k in ("exp_avg", "exp_avg_sq"):
        if k not in state:
            continue
        t = state[k]
        if rows is not None:
            t[rows] = 0.0
        if cols is not None:
            t[:, cols] = 0.0


@torch.no_grad()
def measure_dormant(agent: Agent, obs_batch: torch.Tensor, tau: float):
    """Returns dict mapping (net_name, relusq_idx) -> bool mask of dormant
    neurons (True == dormant). Score per ReDo: s_i = mean|h_i| / mean_j mean|h_j|."""
    activations: dict[tuple, torch.Tensor] = {}
    hooks = []
    for net_name, net in (("actor_mean", agent.actor_mean), ("critic", agent.critic)):
        for i, m in enumerate(net):
            if isinstance(m, ReLUSquared):
                def make_hook(key):
                    def hook(_mod, _inp, out):
                        activations[key] = out.detach().abs().mean(dim=0)
                    return hook
                hooks.append(m.register_forward_hook(make_hook((net_name, i))))
    agent.actor_mean(obs_batch)
    agent.critic(obs_batch)
    for h in hooks:
        h.remove()
    dormant = {}
    for key, mean_abs in activations.items():
        denom = mean_abs.mean().clamp_min(1e-8)
        score = mean_abs / denom
        dormant[key] = score <= tau
    return dormant


@torch.no_grad()
def reset_dormant_neurons(agent: Agent, dormant: dict, optimizer: optim.Optimizer):
    """In-place ReDo reset. Returns total #neurons reset across all layers."""
    total = 0
    for net_name, net in (("actor_mean", agent.actor_mean), ("critic", agent.critic)):
        for relusq_idx, in_linear, rms, next_linear in _layer_triples(net):
            mask = dormant.get((net_name, relusq_idx))
            if mask is None or not mask.any():
                continue
            idx = mask.nonzero(as_tuple=True)[0]
            total += idx.numel()

            new_rows = torch.empty(idx.numel(), in_linear.weight.size(1), device=in_linear.weight.device, dtype=in_linear.weight.dtype)
            nn.init.orthogonal_(new_rows, gain=in_linear._init_gain)
            in_linear.weight.data[idx] = new_rows
            in_linear.bias.data[idx] = 0.0
            _zero_adam_state(optimizer, in_linear.weight, rows=idx)
            _zero_adam_state(optimizer, in_linear.bias, rows=idx)

            if rms is not None:
                rms.weight.data[idx] = 1.0
                _zero_adam_state(optimizer, rms.weight, rows=idx)

            if next_linear is not None:
                next_linear.weight.data[:, idx] = 0.0
                _zero_adam_state(optimizer, next_linear.weight, cols=idx)
    return total


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

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    latent_zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

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
                action, z, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
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
        b_latent_zs = latent_zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_latent_zs[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

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
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        if args.redo_every_iters > 0 and iteration % args.redo_every_iters == 0:
            dormant = measure_dormant(agent, b_obs, tau=args.redo_tau)
            n_reset = reset_dormant_neurons(agent, dormant, optimizer)
            writer.add_scalar("plastic/n_reset", n_reset, global_step)
            for (net_name, idx), mask in dormant.items():
                writer.add_scalar(f"plastic/dormant/{net_name}_{idx}", int(mask.sum().item()), global_step)
            if n_reset > 0:
                print(f"[Plastic] global_step={global_step} reset {n_reset} dormant neurons")

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
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
