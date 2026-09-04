# Delightful PPO with plain-EMA standard-rollout actor updates v6.
#
# DG multiplies each on-policy score term by sigmoid(A * ell / eta), where ell
# is clipped continuous-action surprisal. Standard PPO rollout collection uses
# 16 environments for 128 steps; the actor takes exactly one full-batch step on
# that fresh rollout, while only the critic uses minibatches.
# Immediate rewards use only the paper's decay-0.999 EMA RMS. Advantages and
# delight are not normalized, whitened, or otherwise rescaled. State-dependent
# log sigma uses SAC's smooth bounds and its natural zero-head initialization,
# which maps to log sigma=-1.5. Actions use SAC's tanh transform and the exact
# bounded-action density for both the policy score and surprisal.
#
# Hypothesis: the paper's raw DG gate and reward EMA can learn without explicit
# advantage-scale calibration when paired with PPO's established rollout.
# Reference: https://arxiv.org/abs/2603.14608v1
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro
from torch import nn, optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
INITIAL_LOG_STD = -1.5
RAW_LOG_STD_INIT = float(
    np.arctanh(
        2.0 * (INITIAL_LOG_STD - LOG_STD_MIN) / (LOG_STD_MAX - LOG_STD_MIN) - 1.0
    )
)
DEFAULT_LAYER_STD = np.sqrt(2.0)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """CUDA is required for training"""
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
    """total timesteps of the experiment"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    target_actor_batch_size: int = 2048
    """required standard PPO rollout batch size"""
    anneal_lr: bool = False
    """toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for generalized advantage estimation"""
    num_minibatches: int = 32
    """the number of critic mini-batches; the actor always uses the full batch"""
    update_epochs: int = 10
    """the number of critic epochs; the actor always uses one epoch"""
    clip_coef: float = 0.2
    """the PPO surrogate clipping coefficient"""
    clip_vloss: bool = True
    """whether to use a clipped value-function loss"""
    vf_coef: float = 0.5
    """the value-loss coefficient"""
    max_grad_norm: float = 0.5
    """the maximum gradient norm"""
    reward_ema_decay: float = 0.999
    """decay of the immediate-reward second-moment EMA"""
    reward_norm_eps: float = 1e-8
    """numerical stability constant for EMA reward scaling"""
    delight_temperature: float = 1.0
    """temperature eta in the delightful gate"""
    surprisal_clip: float = 10.0
    """absolute clip bound for continuous-action surprisal"""

    # to be filled in at runtime
    batch_size: int = 0
    """the batch size (computed at runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed at runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed at runtime)"""


class EMARewardNormalizer:
    """Scale immediate rewards by their bias-corrected EMA root mean square."""

    def __init__(self, decay=0.999, epsilon=1e-8):
        if not 0.0 <= decay < 1.0:
            raise ValueError("reward EMA decay must be in [0, 1)")
        self.decay = decay
        self.epsilon = epsilon
        self.squared_reward_ema = 0.0
        self.num_updates = 0

    def normalize(self, rewards):
        batch_second_moment = float(np.mean(np.square(rewards, dtype=np.float64)))
        self.squared_reward_ema = (
            self.decay * self.squared_reward_ema
            + (1.0 - self.decay) * batch_second_moment
        )
        self.num_updates += 1
        correction = 1.0 - self.decay**self.num_updates
        reward_rms = np.sqrt(self.squared_reward_ema / correction)
        return rewards / (reward_rms + self.epsilon), reward_rms


def make_env(env_id, idx, capture_video, run_name, gamma):
    del gamma  # Kept in the signature for compatibility with CleanRL's evaluator.

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        return env

    return thunk


def layer_init(layer, std=DEFAULT_LAYER_STD, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def delightful_gate(advantages, logprobs, temperature=1.0, clip_bound=10.0):
    """Return the detached Algorithm 2 gate and its diagnostics."""
    if temperature <= 0.0:
        raise ValueError("delight temperature must be positive")
    if clip_bound <= 0.0:
        raise ValueError("surprisal clip must be positive")
    surprisal = (-logprobs.detach()).clamp(-clip_bound, clip_bound)
    delight = advantages.detach() * surprisal
    gate = torch.sigmoid(delight / temperature)
    return gate, surprisal, delight


def delightful_ppo_loss(advantages, ratios, gate, clip_coef):
    """Apply DG weights to the standard clipped PPO surrogate."""
    pg_loss1 = -advantages * ratios
    pg_loss2 = -advantages * torch.clamp(ratios, 1.0 - clip_coef, 1.0 + clip_coef)
    return (gate * torch.maximum(pg_loss1, pg_loss2)).mean()


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor_mean = layer_init(nn.Linear(64, action_dim), std=0.01)
        self.actor_logstd = layer_init(
            nn.Linear(64, action_dim), std=0.01, bias_const=RAW_LOG_STD_INIT
        )
        self.register_buffer(
            "action_scale",
            torch.as_tensor(
                (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor(
                (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        if not torch.all(torch.isfinite(self.action_scale)) or not torch.all(
            self.action_scale > 0.0
        ):
            raise ValueError(
                "tanh Gaussian actions require finite, non-degenerate action bounds"
            )

    def get_value(self, x):
        return self.critic(x)

    def get_action_distribution(self, x):
        actor_features = self.actor_trunk(x)
        action_mean = self.actor_mean(actor_features)
        action_logstd = torch.tanh(self.actor_logstd(actor_features))
        action_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            action_logstd + 1.0
        )
        return Normal(action_mean, action_logstd.exp()), action_logstd

    def _action_and_logprob_from_raw(self, probs, raw_action):
        squashed_action = torch.tanh(raw_action)
        action = squashed_action * self.action_scale + self.action_bias
        log_tanh_jacobian = 2.0 * (
            np.log(2.0) - raw_action - torch.nn.functional.softplus(-2.0 * raw_action)
        )
        logprob = (
            probs.log_prob(raw_action)
            - log_tanh_jacobian
            - torch.log(self.action_scale)
        ).sum(1)
        return action, logprob

    def sample_action_and_value(self, x):
        probs, _ = self.get_action_distribution(x)
        raw_action = probs.rsample()
        action, logprob = self._action_and_logprob_from_raw(probs, raw_action)
        return action, raw_action, logprob, -logprob.detach(), self.critic(x)

    def get_action_and_value(self, x, action=None, raw_action=None):
        if action is None and raw_action is None:
            action, _, logprob, entropy, value = self.sample_action_and_value(x)
            return action, logprob, entropy, value
        if raw_action is None:
            raise ValueError(
                "raw_action is required when evaluating a stored tanh action"
            )
        probs, _ = self.get_action_distribution(x)
        transformed_action, logprob = self._action_and_logprob_from_raw(
            probs, raw_action
        )
        if action is None:
            action = transformed_action
        return (
            action,
            logprob,
            -logprob.detach(),
            self.critic(x),
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    if args.batch_size != args.target_actor_batch_size:
        raise ValueError(
            "DG requires a fresh actor batch of "
            f"{args.target_actor_batch_size} transitions, got "
            f"num_envs * num_steps = {args.batch_size}"
        )
    if args.batch_size % args.num_minibatches != 0:
        raise ValueError("critic minibatches must divide the rollout batch exactly")
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
    hyperparameter_rows = "\n".join(
        f"|{key}|{value}|" for key, value in vars(args).items()
    )
    writer.add_text("hyperparameters", f"|param|value|\n|-|-|\n{hyperparameter_rows}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not args.cuda:
        raise ValueError("this experiment requires CUDA; --no-cuda is unsupported")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but is not available")
    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, i, args.capture_video, run_name, args.gamma)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action spaces are supported"
    )

    agent = Agent(envs).to(device)
    actor_parameters = list(agent.actor_trunk.parameters())
    actor_parameters += list(agent.actor_mean.parameters())
    actor_parameters += list(agent.actor_logstd.parameters())
    actor_optimizer = optim.Adam(actor_parameters, lr=args.learning_rate, eps=1e-5)
    critic_optimizer = optim.Adam(
        agent.critic.parameters(), lr=args.learning_rate, eps=1e-5
    )
    reward_normalizer = EMARewardNormalizer(
        args.reward_ema_decay,
        args.reward_norm_eps,
    )

    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    raw_actions = torch.zeros_like(actions)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    reward_rms = 1.0

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            actor_optimizer.param_groups[0]["lr"] = frac * args.learning_rate
            critic_optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, raw_action, logprob, _, value = agent.sample_action_and_value(
                    next_obs
                )
                values[step] = value.flatten()
            actions[step] = action
            raw_actions[step] = raw_action
            logprobs[step] = logprob

            next_obs_np, raw_reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            normalized_reward, reward_rms = reward_normalizer.normalize(raw_reward)
            rewards[step] = torch.as_tensor(
                normalized_reward, dtype=torch.float32, device=device
            )
            next_done_np = np.logical_or(terminations, truncations)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(
                next_done_np, dtype=torch.float32, device=device
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_raw_actions = raw_actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # One actor epoch and one full-batch step. The rollout log-density is
        # fresh under the current actor, so the gate never consumes stale data.
        _, newlogprob, _, _ = agent.get_action_and_value(
            b_obs, b_actions, raw_action=b_raw_actions
        )
        logratio = newlogprob - b_logprobs
        ratio = logratio.exp()
        gate, surprisal, delight = delightful_gate(
            b_advantages,
            b_logprobs,
            temperature=args.delight_temperature,
            clip_bound=args.surprisal_clip,
        )
        pg_loss = delightful_ppo_loss(b_advantages, ratio, gate, args.clip_coef)
        actor_loss = pg_loss

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(
            actor_parameters, args.max_grad_norm
        )
        actor_optimizer.step()

        # Measure the realized full-batch policy change after the sole actor step.
        with torch.no_grad():
            _, post_logprob, _, _ = agent.get_action_and_value(
                b_obs, b_actions, raw_action=b_raw_actions
            )
            _, post_action_logstd = agent.get_action_distribution(b_obs)
            post_logratio = post_logprob - b_logprobs
            post_ratio = post_logratio.exp()
            old_approx_kl = (-post_logratio).mean()
            approx_kl = ((post_ratio - 1.0) - post_logratio).mean()
            clipfrac = ((post_ratio - 1.0).abs() > args.clip_coef).float().mean().item()

        # The critic retains PPO's minibatched, multi-epoch regression. It has a
        # separate optimizer and disjoint parameters, so this cannot update the actor.
        b_inds = np.arange(args.batch_size)
        critic_losses = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]
                newvalue = agent.get_value(b_obs[mb_inds]).view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = (
                        0.5 * torch.maximum(v_loss_unclipped, v_loss_clipped).mean()
                    )
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                critic_optimizer.zero_grad()
                (args.vf_coef * v_loss).backward()
                nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                critic_optimizer.step()
                critic_losses.append(v_loss.item())

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y

        writer.add_scalar(
            "charts/learning_rate", actor_optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("charts/reward_rms", reward_rms, global_step)
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        writer.add_scalar("losses/value_loss", np.mean(critic_losses), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", (-b_logprobs).mean().item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", clipfrac, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar(
            "diagnostics/actor_grad_norm", actor_grad_norm.item(), global_step
        )
        writer.add_scalar(
            "diagnostics/advantage_mean", b_advantages.mean().item(), global_step
        )
        writer.add_scalar(
            "diagnostics/advantage_std", b_advantages.std().item(), global_step
        )
        writer.add_scalar(
            "diagnostics/advantage_rms",
            b_advantages.square().mean().sqrt().item(),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/normalized_reward_rms",
            rewards.square().mean().sqrt().item(),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/delight_gate_mean", gate.mean().item(), global_step
        )
        writer.add_scalar(
            "diagnostics/delight_gate_saturation",
            ((gate < 0.01) | (gate > 0.99)).float().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/surprisal_mean", surprisal.mean().item(), global_step
        )
        writer.add_scalar(
            "diagnostics/surprisal_std", surprisal.std().item(), global_step
        )
        writer.add_scalar(
            "diagnostics/surprisal_rms",
            surprisal.square().mean().sqrt().item(),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/negative_surprisal_fraction",
            (surprisal < 0.0).float().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/surprisal_clip_fraction",
            (surprisal.abs() == args.surprisal_clip).float().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/delight_mean", delight.mean().item(), global_step
        )
        writer.add_scalar("diagnostics/delight_std", delight.std().item(), global_step)
        writer.add_scalar(
            "diagnostics/delight_rms",
            delight.square().mean().sqrt().item(),
            global_step,
        )
        normalized_actions = (b_actions - agent.action_bias) / agent.action_scale
        writer.add_scalar(
            "diagnostics/action_saturation_fraction",
            (normalized_actions.abs() > 0.95).float().mean().item(),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/action_logstd_mean",
            post_action_logstd.mean().item(),
            global_step,
        )
        print("SPS:", int(global_step / (time.time() - start_time)))

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
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "PPO",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    envs.close()
    writer.close()
