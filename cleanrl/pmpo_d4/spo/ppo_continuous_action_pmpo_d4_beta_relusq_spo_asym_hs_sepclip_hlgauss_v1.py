# PMPO-D4 + Beta + ReluSq + SPO asym (half-strength) + sepclip + HL-Gauss critic.
#
# Sibling of pmpo_d4_beta_relusq_spo_asym_hs_sepclip_v1; only the value head
# changes. Actor + SPO objective + half-strength epsilons + separate actor /
# critic clip-grad-norm pattern are unchanged.
#
#   critic: nn.Sequential MLP → Linear(64, num_bins)  # logits over fixed bins
#   bootstrap: E[support · softmax(logits)]   (HLGaussSupport.to_scalar)
#   value loss: cross-entropy on Gaussian-projected returns (no vclip; PPO's
#               ratio-style vclip doesn't fit a categorical head)
#
# HL-Gauss support utility (cleanrl/shared/hl_gauss.py):
#   project: scalar return → Gaussian-smoothed categorical over fixed bins
#   to_scalar: logits → scalar via expectation under softmax
#   use_symlog=True maps real-valued returns into a compressed range so a
#   small fixed [v_min, v_max] window covers wide HC reward distributions.
#
# Half-strength SPO bounds carry over: ε_low=0.40, ε_high=0.56 (2x baseline,
# half-penalty). Matches the unmodified `pmpo_d4_beta_spo_asym_halfstrength_v1`
# 4720 baseline plus sepclip plus HL-Gauss critic.
#
# Separate-clip pattern (orbit-wars-style) carries over too: after a single
#     loss.backward()
# we apply two independent clips on the disjoint actor and critic subnets:
#     clip_grad_norm_(agent.actor.parameters(),  0.5)   # only sees pg + ent grad
#     clip_grad_norm_(agent.critic.parameters(), 0.5)   # only sees vf grad
# No shared trunk → no retain_graph / clone-grad dance needed. The HL-Gauss
# CE loss flows only through self.critic; the SPO policy loss flows only
# through self.actor.
#
# Hypothesis. HL-Gauss critic gradients are categorical-CE rather than MSE,
# with a fundamentally different scale / variance profile. Under a single
# global clip, this scale mismatch is what likely starved the policy
# gradient. Separate clips should let HL-Gauss train at its native scale
# without dragging the actor's step size around — and the combination
# (HL-Gauss + sepclip) should be cleanly additive rather than fighting.

import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).resolve().parents[1]))
from cleanrl.shared.hl_gauss import HLGaussSupport

SAMPLE_EPS = 1e-7


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
    norm_adv: bool = True
    """Toggles advantage normalization (PPO-style minibatch standardization)."""
    num_bins: int = 51
    """number of bins for the HL-Gauss categorical critic"""
    v_min: float = -5.0
    """min of the value support (in symlog space when use_symlog=True)"""
    v_max: float = 5.0
    """max of the value support (in symlog space when use_symlog=True)"""
    sigma_ratio: float = 0.5
    """projection sigma in bin-widths for HL-Gauss target smoothing"""
    use_symlog: bool = True
    """whether to symlog-compress returns before HL-Gauss projection"""
    spo_eps_low: float = 0.40
    """SPO penalty bound — half-strength (2x baseline 0.20), drift opposes advantage"""
    spo_eps_high: float = 0.56
    """SPO penalty bound — half-strength (2x baseline 0.28), drift agrees with advantage"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping (applied independently to actor and critic)"""

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


class ReluSq(nn.Module):
    """f(x) = relu(x)^2 — x^2 for x>=0, 0 for x<0."""
    def forward(self, x):
        return torch.relu(x).square()


class Agent(nn.Module):
    def __init__(self, envs, num_bins):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.action_dim = action_dim
        self.num_bins = num_bins

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, num_bins), std=1.0),
        )
        # Actor outputs 2*action_dim: [pre-softplus α heads, pre-softplus β heads].
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 2 * action_dim), std=0.01),
        )
        self.register_buffer(
            "action_low",
            torch.tensor(envs.single_action_space.low, dtype=torch.float32),
        )
        self.register_buffer(
            "action_high",
            torch.tensor(envs.single_action_space.high, dtype=torch.float32),
        )

    def get_value_logits(self, x):
        return self.critic(x)

    def get_value(self, x, hl_support):
        return hl_support.to_scalar(self.critic(x))

    def get_action_distribution(self, x):
        head = self.actor(x)
        head_alpha, head_beta = head.chunk(2, dim=-1)
        alpha = 1.0 + nn.functional.softplus(head_alpha)
        beta = 1.0 + nn.functional.softplus(head_beta)
        return Beta(alpha, beta)

    def _z_to_action(self, z):
        return self.action_low + (self.action_high - self.action_low) * z

    def _action_to_z(self, action):
        return ((action - self.action_low) / (self.action_high - self.action_low)).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)

    def get_action_and_value(self, x, action=None):
        dist = self.get_action_distribution(x)
        if action is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            action = self._z_to_action(z)
        else:
            z = self._action_to_z(action)
        log_prob = dist.log_prob(z).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, self.critic(x), dist  # critic now returns logits (B, num_bins)


def evaluate_policy(model_path, make_env, env_id, eval_episodes, run_name, model, device, gamma, num_bins):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, True, run_name, gamma)])
    agent = model(envs, num_bins).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).to(device)
            actions, _, _, _, _ = agent.get_action_and_value(obs_tensor)
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    envs.close()
    return episodic_returns


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

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args.num_bins).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    hl_support = HLGaussSupport(
        args.num_bins, args.v_min, args.v_max, args.sigma_ratio, device, use_symlog=args.use_symlog,
    )

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
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
                action, logprob, _, value_logits, _ = agent.get_action_and_value(next_obs)
                values[step] = hl_support.to_scalar(value_logits)
            actions[step] = action
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
            next_value = agent.get_value(next_obs, hl_support).reshape(1, -1)
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
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        old_approx_kl = torch.zeros((), device=device)
        approx_kl = torch.zeros((), device=device)
        spo_penalty_mean = torch.zeros((), device=device)
        actor_grad_norm = torch.zeros((), device=device)
        critic_grad_norm = torch.zeros((), device=device)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue_logits, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # KL(old||new) approximations http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # SPO with asymmetric ε. Per-sample bound is ε_high when
                # drift direction agrees with advantage sign (more
                # permissive in the "right" direction), ε_low otherwise.
                ratio_diff = ratio - 1.0
                with_adv = (mb_advantages * ratio_diff) > 0  # bool [N]
                eps = torch.where(
                    with_adv,
                    torch.full_like(mb_advantages, args.spo_eps_high),
                    torch.full_like(mb_advantages, args.spo_eps_low),
                )
                pg_surrogate = mb_advantages * ratio
                spo_penalty = mb_advantages.abs() * ratio_diff.pow(2) / (2.0 * eps)
                pg_loss = -(pg_surrogate - spo_penalty).mean()
                spo_penalty_mean = spo_penalty.detach().mean()

                # Value loss: HL-Gauss cross-entropy on projected returns.
                # No vclip — categorical doesn't have a clean ratio analogue.
                target_probs = hl_support.project(b_returns[mb_inds])
                log_probs_v = torch.log_softmax(newvalue_logits, dim=-1)
                v_loss = -(target_probs * log_probs_v).sum(dim=-1).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                # Disjoint actor / critic subnets — single combined backward
                # leaves actor params with only ∂(pg + ent)/∂θ_actor and
                # critic params with only ∂(vf_coef·v_loss)/∂θ_critic, so
                # we can clip each subnet independently.
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    agent.actor.parameters(), args.max_grad_norm
                )
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    agent.critic.parameters(), args.max_grad_norm
                )
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Beta diagnostics
        with torch.no_grad():
            dist_diag = agent.get_action_distribution(b_obs)
            mean_alpha = dist_diag.concentration1.mean().item()
            mean_beta = dist_diag.concentration0.mean().item()
            sum_conc = (dist_diag.concentration1 + dist_diag.concentration0).mean().item()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("diag/spo_penalty", spo_penalty_mean.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("diag/beta_mean_alpha", mean_alpha, global_step)
        writer.add_scalar("diag/beta_mean_beta", mean_beta, global_step)
        writer.add_scalar("diag/beta_concentration_sum", sum_conc, global_step)
        writer.add_scalar("diag/actor_grad_norm", actor_grad_norm.item(), global_step)
        writer.add_scalar("diag/critic_grad_norm", critic_grad_norm.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        episodic_returns = evaluate_policy(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            model=Agent,
            device=device,
            gamma=args.gamma,
            num_bins=args.num_bins,
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
