# PPO + EPO (Evolutionary Policy Optimization) v2
#
# Faithful single-file port of EPO (Su et al., 2025, arXiv:2503.19037) on top of
# the Beta + ReLUSq PPO actor from hc_beta_relusq_v1. EPO trains *one* shared
# network that realizes a whole population of behaviors by conditioning on a
# per-genome learnable latent, and shares experience across the population so a
# diverse exploratory frontier accelerates a single exploitation policy.
#
# v2 closes three fidelity gaps an alignment audit found against the reference
# `--epo` path:
#   * `filter_leader` block-subsetting: each off-policy share contributes only
#     ONE block's relabeled transitions (custom_utils.py:filter_leader), not the
#     whole rollout, so the single on-policy copy isn't drowned by off-policy
#     data (matches off_policy_ratio semantics, a2c_common.py:965-966).
#   * `coef_cond` per-genome exploration scale: each genome gets a learnable
#     concentration bias (the Beta analog of the reference per-genome learnable
#     sigma `nn.Parameter(zeros(K, act_dim))`, network_builder.py:293-295) — an
#     explicit, state-independent per-genome exploration-magnitude knob.
#   * latents init as `randn` (std 1, like `extra_params`), default genome_dim 32.
# The per-genome entropy spectrum is the OPTIONAL `--ir-type entropy` add-on
# (off in default `--epo`, which falls to a single global entropy coef,
# a2c_continuous.py:150); it stays available via `ent_coef_max` but defaults OFF
# so genome diversity comes from the latents + coef_cond, as in core EPO.
#
# What `--epo` enables in the reference (rl_games) = `mixed_expl_learn_param` + `lf`:
#   1. Latent-conditioned shared actor-critic. The population is split into K
#      genomes; each genome owns a *learnable* latent z_k (nn.Parameter, the
#      reference's `extra_params`). The net sees concat(obs, z_genome). Distinct
#      latents -> distinct policies from one set of weights.
#      (rl_games network_builder.py: extra_params + forward concat)
#   2. Population blocks: K genomes mapped to contiguous blocks of the parallel
#      envs (block_size = num_envs // K).
#   3. Exploration spectrum: a per-genome entropy-bonus coefficient that ranges
#      linearly from `ent_coef_max` (explorer, genome 0) down to 0 (exploiter,
#      genome K-1). (rl_games a2c_continuous.py: per-block entropy_coef)
#   4. Leader-follower experience sharing (`augment_batch_for_mixed_expl`): each
#      block's transitions are relabeled under *other* genomes, the follower's
#      critic recomputes value and a 1-step TD return/advantage, while the
#      behavior policy's old log-prob is reused. PPO's ratio clipping turns this
#      into a safe off-policy update, cross-pollinating exploratory experience
#      into every genome (and thus the exploiter).
#
# Hypothesis:
# A learnable-latent population with a built-in exploration->exploitation
# entropy spectrum and leader-follower sharing explores MuJoCo more effectively
# than a single PPO policy, while the shared exploiter genome (entropy coef 0)
# converges to a strong deployable policy. The reported `charts/episodic_return`
# tracks the exploiter genome's envs = the deployed EPO policy.
#
# Adaptations for the CleanRL MuJoCo setting:
# - Beta actor + ReLUSq hidden activations kept from the base file.
# - Genome latent concatenated to obs at the network input (cleaner than the
#   reference's in-obs id-replacement; semantically identical).
# - Own-genome ("leader") data uses full GAE; shared ("follower") data uses the
#   reference's 1-step TD return/advantage under the follower critic.
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
import torch.nn.functional as F
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter


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
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy (baseline / fallback; EPO uses the per-genome spectrum)"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # EPO specific arguments
    num_genomes: int = 4
    """size of the latent-conditioned population (K). num_envs must be divisible by it"""
    genome_dim: int = 32
    """dimension of each learnable genome latent z_k (reference param_size=32)"""
    ent_coef_max: float = 0.0
    """OPTIONAL per-genome entropy-bonus spectrum (the `--ir-type entropy` add-on); linspace(ent_coef_max, 0) over genomes. 0 = off (core --epo default: diversity from latents + coef_cond)"""
    share_repeats: int = 1
    """leader-follower off-policy repeats: how many one-block genome relabelings to add to the PPO batch each update (0 disables sharing)"""

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
    def forward(self, x):
        return torch.relu(x).pow(2)


class Agent(nn.Module):
    """Shared actor-critic conditioned on a per-genome learnable latent (EPO)."""

    def __init__(self, envs, num_genomes, genome_dim):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.num_genomes = num_genomes

        # The population: one learnable latent per genome (rl_games `extra_params`,
        # initialized randn std 1 to match the reference network_builder.py:212).
        self.genomes = nn.Parameter(torch.randn(num_genomes, genome_dim))
        # coef_cond analog: a learnable per-genome concentration bias. Subtracted
        # from both Beta head logits, it lowers concentration -> raises action
        # variance, giving each genome an explicit exploration-magnitude knob
        # (Beta analog of the reference per-genome sigma, network_builder.py:293).
        self.genome_conc_bias = nn.Parameter(torch.zeros(num_genomes, action_dim))

        in_dim = obs_dim + genome_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(in_dim, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 64)),
            ReluSq(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(in_dim, 64)),
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

    def _resolve_idx(self, x, genome_idx):
        # genome_idx: None -> exploiter genome (last, lowest exploration); else LongTensor (batch,)
        if genome_idx is None:
            genome_idx = x.new_full((x.shape[0],), self.num_genomes - 1, dtype=torch.long)
        return genome_idx

    def _condition(self, x, genome_idx):
        return torch.cat([x, self.genomes[genome_idx]], dim=-1)

    def get_value(self, x, genome_idx=None):
        genome_idx = self._resolve_idx(x, genome_idx)
        return self.critic(self._condition(x, genome_idx))

    def _beta(self, x, genome_idx):
        head_alpha, head_beta = self.actor(self._condition(x, genome_idx)).chunk(2, dim=-1)
        # per-genome exploration scale: lower concentration -> higher variance
        bias = self.genome_conc_bias[genome_idx]
        alpha = 1.0 + F.softplus(head_alpha - bias)
        beta = 1.0 + F.softplus(head_beta - bias)
        return Beta(alpha, beta)

    def get_action_distribution(self, x, genome_idx=None):
        return self._beta(x, self._resolve_idx(x, genome_idx))

    def _z_to_action(self, z):
        return self.action_low + (self.action_high - self.action_low) * z

    def _action_to_z(self, action):
        z = (action - self.action_low) / (self.action_high - self.action_low)
        return z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)

    def get_action_and_value(self, x, genome_idx=None, action=None):
        genome_idx = self._resolve_idx(x, genome_idx)
        probs = self._beta(x, genome_idx)
        if action is None:
            z = probs.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
            action = self._z_to_action(z)
        else:
            z = self._action_to_z(action)
        return action, probs.log_prob(z).sum(1), probs.entropy().sum(1), self.critic(self._condition(x, genome_idx))


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    assert args.num_envs % args.num_genomes == 0, "num_envs must be divisible by num_genomes"
    block_size = args.num_envs // args.num_genomes
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

    agent = Agent(envs, args.num_genomes, args.genome_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # EPO population layout: contiguous blocks of envs share a genome.
    #   genome 0 = strongest explorer (top of entropy spectrum)
    #   genome K-1 = exploiter (entropy coef 0) = deployed policy
    env_genome = (torch.arange(args.num_envs, device=device) // block_size).long()  # (num_envs,)
    ent_spectrum = torch.linspace(args.ent_coef_max, 0.0, args.num_genomes, device=device)  # (K,)
    exploiter_genome = args.num_genomes - 1

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

            # ALGO LOGIC: action logic — each env acts under its block's behavior genome.
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, env_genome)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                # Per-genome diagnostics; headline metric = exploiter genome (deployed policy).
                for env_idx, info in enumerate(infos["final_info"]):
                    if info and "episode" in info:
                        g = int(env_genome[env_idx].item())
                        ep_r = info["episode"]["r"]
                        writer.add_scalar(f"charts/episodic_return_genome_{g}", ep_r, global_step)
                        if g == exploiter_genome:
                            print(f"global_step={global_step}, episodic_return={ep_r}")
                            writer.add_scalar("charts/episodic_return", ep_r, global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        elif g == 0:
                            print(f"global_step={global_step}, genome0_return={ep_r}")

        # bootstrap value if not done — GAE per env under each env's behavior genome (own-genome / "leader" data)
        with torch.no_grad():
            next_value = agent.get_value(next_obs, env_genome).reshape(1, -1)
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

        # flatten the batch (own-genome data)
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_genome = env_genome.repeat(args.num_steps)  # (T*N,) genome id per sample

        # ---- EPO leader-follower experience sharing (filter_leader semantics) -----
        # Per repeat, take ONE source block's transitions, relabel them under a
        # different (follower) genome, recompute value + a 1-step TD return/advantage
        # with the follower critic, and keep the behavior old log-prob. PPO clipping
        # makes this a safe off-policy update. Subsetting to a single block (vs the
        # whole rollout) keeps the on-policy:off-policy ratio balanced, matching the
        # reference `lf` filter_leader / off_policy_ratio behavior.
        with torch.no_grad():
            for rep in range(args.share_repeats):
                src = random.randint(0, args.num_genomes - 1)  # source block to share
                tgt = (src + random.randint(1, args.num_genomes - 1)) % args.num_genomes  # follower genome
                lo, hi = src * block_size, (src + 1) * block_size
                n_blk = hi - lo

                o_blk = obs[:, lo:hi]  # (T, bs, obs_dim)
                fg = torch.full((args.num_steps * n_blk,), tgt, dtype=torch.long, device=device)

                # follower critic values over the block's rollout + bootstrap state
                vf = agent.get_value(o_blk.reshape(-1, o_blk.shape[-1]), fg).reshape(args.num_steps, n_blk)
                vf_last = agent.get_value(
                    next_obs[lo:hi], torch.full((n_blk,), tgt, dtype=torch.long, device=device)
                ).reshape(-1)

                # 1-step TD return/advantage under the follower critic
                vf_next = torch.empty_like(vf)
                vf_next[:-1] = vf[1:]
                vf_next[-1] = vf_last
                nnt = torch.empty_like(vf)
                nnt[:-1] = 1.0 - dones[1:, lo:hi]
                nnt[-1] = 1.0 - next_done[lo:hi]
                returns_f = rewards[:, lo:hi] + args.gamma * vf_next * nnt
                adv_f = returns_f - vf

                b_obs = torch.cat([b_obs, o_blk.reshape((-1,) + envs.single_observation_space.shape)], 0)
                b_actions = torch.cat([b_actions, actions[:, lo:hi].reshape((-1,) + envs.single_action_space.shape)], 0)
                b_logprobs = torch.cat([b_logprobs, logprobs[:, lo:hi].reshape(-1)], 0)  # behavior old log-prob, reused
                b_advantages = torch.cat([b_advantages, adv_f.reshape(-1)], 0)
                b_returns = torch.cat([b_returns, returns_f.reshape(-1)], 0)
                b_values = torch.cat([b_values, vf.reshape(-1)], 0)
                b_genome = torch.cat([b_genome, fg], 0)

        # Optimizing the policy and value network
        total_size = b_obs.shape[0]
        b_inds = np.arange(total_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, total_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_genome = b_genome[mb_inds]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], mb_genome, b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

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

                # Per-genome entropy bonus: explorer genomes get a strong bonus,
                # the exploiter genome gets none (EPO exploration spectrum).
                mb_ent_coef = ent_spectrum[mb_genome] + args.ent_coef
                entropy_loss = (mb_ent_coef * entropy).mean()
                loss = pg_loss - entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # explained variance over the own-genome (leader) GAE slice only, so the
        # diagnostic stays a clean GAE fit and isn't blended with follower 1-step TD targets
        y_pred, y_true = b_values[: args.batch_size].cpu().numpy(), b_returns[: args.batch_size].cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
