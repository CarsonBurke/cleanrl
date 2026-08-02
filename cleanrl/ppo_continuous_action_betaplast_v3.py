# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
#
# BetaPlast v3: learned "genes" — low-rank FiLM perturbations replace per-neuron gains.
#
# v3 changes over v2 (SiLU trunk + v168-style Beta action policy kept as-is):
# - The 128 independent per-neuron gains are replaced by a gene vector z in R^k
#   (k=8), sampled per env per iteration from the same signed-Beta construction,
#   with ONE learned concentration per gene dim (mutation-scale self-adaptation in
#   gene space; 16 rollouts against 8 dials instead of 128 -> far better SNR).
# - A learned FiLM decoder maps genes to per-layer modulation of the actor trunk:
#     h <- h * (1 + A_l z) + B_l z
#   Loading columns are unit-normalized in the forward pass: the decoder learns only
#   DIRECTIONS of variation, all scale lives in the bounded gene concentrations —
#   so the rigidity attractor cannot shrink the decoder to zero and scale cannot
#   blow up. Modulations get a +-0.9 safety clamp (gamma can never flip sign).
# - Antithetic pairing is a full vector mirror (z, -z on env pairs): exact
#   first-order cancellation.
# - Genes co-modulate many neurons through each learned column: perturbations are
#   coordinated functional variations (an ES-style learned low-rank covariance),
#   not independent scalar jiggles. The additive B path can move neurons whose
#   output sits at zero — multiplicative gains never could.
#
# Core mechanism otherwise unchanged, with NO meta objective:
# 1. Coherent parameter-space exploration: (u, r) frozen per iteration, replayed by
#    env id during updates -> PPO ratios exactly consistent.
# 2. Pathwise learning: z = delta_g * r * (1 - u^(1/c)) is differentiable in the
#    gene concentration c with frozen noise; the FiLM loadings A_l, B_l are trained
#    by the same clipped surrogate (directions align with advantage-correlated
#    variation). E[z] = 0 pins the mean policy at the unperturbed net.
# 3. Plasticity gates the step size: each neuron's POST-Adam update (input weight
#    row + bias) is scaled by its total FiLM sensitivity sum_j (A_ij^2 + B_ij^2) *
#    Var(z_j), relative to the layer mean, clamped. Plasticity is now an emergent
#    property of where the learned gene manifold points.
#
# Hypothesis: a learned low-dimensional variation manifold concentrates both the
# exploration budget and the plasticity signal on the few directions that matter,
# fixing v1/v2's slow per-neuron credit assignment (c-spread of only ~2.6-3.5
# after 2.4M steps).
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
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # BetaPlast specific arguments
    gene_dim: int = 8
    """dimension of the gene vector z"""
    gene_delta: float = 0.5
    """half-width of each gene dim's signed-Beta support (-delta, +delta)"""
    plast_conc_init: float = 3.0
    """initial per-gene Beta concentration c (higher = more rigid)"""
    plast_conc_max: float = 30.0
    """concentration cap: enforces a plasticity floor (min gene variance)"""
    plast_gate_min: float = 0.25
    """lower clamp on the relative per-neuron post-Adam step gate"""
    plast_gate_max: float = 4.0
    """upper clamp on the relative per-neuron post-Adam step gate"""

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


class Agent(nn.Module):
    def __init__(self, envs, args=None):
        super().__init__()
        if args is None:  # eval helpers instantiate Agent(envs)
            args = Args()
        self.gene_dim = args.gene_dim
        self.gene_delta = args.gene_delta
        self.conc_max = args.plast_conc_max
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.SiLU(),
            layer_init(nn.Linear(64, 64)),
            nn.SiLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_fc1 = layer_init(nn.Linear(obs_dim, 64))
        self.actor_fc2 = layer_init(nn.Linear(64, 64))
        self.actor_alpha = layer_init(nn.Linear(64, act_dim), std=0.01)
        self.actor_beta = layer_init(nn.Linear(64, act_dim), std=0.01)
        self.register_buffer("action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32))
        # per-gene plasticity: concentration c = 1 + (conc_max - 1) * sigmoid(rho)
        p_init = (args.plast_conc_init - 1.0) / (args.plast_conc_max - 1.0)
        rho_init = float(np.log(p_init / (1.0 - p_init)))
        self.gene_rho = nn.Parameter(torch.full((args.gene_dim,), rho_init))
        # FiLM decoder loadings (column-normalized in forward): genes -> per-layer
        # multiplicative (A) and additive (B) modulation of the actor trunk
        # sign-matrix init (+-1/sqrt(64)): unit-norm columns AND equal row norms,
        # so the step gates start exactly at 1.0 for every neuron
        def loading_init():
            signs = torch.randint(0, 2, (64, args.gene_dim), dtype=torch.float32) * 2.0 - 1.0
            return nn.Parameter(signs / 8.0)

        self.A1 = loading_init()
        self.B1 = loading_init()
        self.A2 = loading_init()
        self.B2 = loading_init()
        # frozen per-iteration gene noise, set by resample_gene_noise
        self.log_u = self.r = None

    def concentrations(self):
        return 1.0 + (self.conc_max - 1.0) * torch.sigmoid(self.gene_rho)

    def resample_gene_noise(self, num_envs, device):
        # one gene draw per env pair per iteration; antithetic full-vector mirror
        assert num_envs % 2 == 0, "antithetic gene noise requires an even number of envs"
        half = num_envs // 2
        self.log_u = torch.rand(half, self.gene_dim, device=device).clamp(1e-6, 1.0).log()
        self.r = torch.randint(0, 2, (half, self.gene_dim), device=device, dtype=torch.float32) * 2.0 - 1.0

    def genes(self):
        # signed Beta(1, c) gene: z = delta_g * r * (1 - u^(1/c)), differentiable in c
        c = self.concentrations()
        z_half = self.gene_delta * self.r * (1.0 - torch.exp(self.log_u / c))
        return torch.stack([z_half, -z_half], dim=1).reshape(-1, self.gene_dim)  # (num_envs, k)

    @staticmethod
    def _colnorm(m):
        return m / m.norm(dim=0, keepdim=True).clamp_min(1e-6)

    def film(self):
        # genes -> per-env FiLM params; unit-norm columns = pure directions,
        # +-0.9 clamp keeps gamma sign-safe in degenerate tails
        z = self.genes()
        g1 = 1.0 + (z @ self._colnorm(self.A1).T).clamp(-0.9, 0.9)
        b1 = (z @ self._colnorm(self.B1).T).clamp(-0.9, 0.9)
        g2 = 1.0 + (z @ self._colnorm(self.A2).T).clamp(-0.9, 0.9)
        b2 = (z @ self._colnorm(self.B2).T).clamp(-0.9, 0.9)
        return g1, b1, g2, b2

    def _dist(self, x, env_ids=None):
        h = F.silu(self.actor_fc1(x))
        if env_ids is not None:
            g1, b1, g2, b2 = self.film()
            h = h * g1[env_ids] + b1[env_ids]
            h = F.silu(self.actor_fc2(h))
            h = h * g2[env_ids] + b2[env_ids]
        else:
            h = F.silu(self.actor_fc2(h))
        alpha = 1.0 + F.softplus(self.actor_alpha(h))
        beta = 1.0 + F.softplus(self.actor_beta(h))
        return Beta(alpha, beta)

    def _z_to_action(self, z):
        return self.action_low + (self.action_high - self.action_low) * z

    def _action_to_z(self, action):
        z = (action - self.action_low) / (self.action_high - self.action_low)
        return z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)

    def get_value(self, x):
        return self.critic(x)

    def get_beta_action_and_value(self, x, z=None, env_ids=None):
        dist = self._dist(x, env_ids)
        if z is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        else:
            z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = self._z_to_action(z)
        logprob = dist.log_prob(z).sum(1)
        return action, z, logprob, dist.entropy().sum(1), self.critic(x)

    def get_action_and_value(self, x, action=None, env_ids=None):
        z = None if action is None else self._action_to_z(action)
        action, _, logprob, entropy, value = self.get_beta_action_and_value(x, z, env_ids)
        return action, logprob, entropy, value


def plasticity_gates(agent, args):
    # relative per-neuron step gate from total FiLM sensitivity:
    # v_i = sum_j (A_ij^2 + B_ij^2) * Var(z_j), Var(z_j) = delta_g^2 * 2/((c_j+1)(c_j+2))
    with torch.no_grad():
        c = agent.concentrations()
        zvar = (agent.gene_delta**2) * 2.0 / ((c + 1.0) * (c + 2.0))
        gates = []
        for A, B in ((agent.A1, agent.B1), (agent.A2, agent.B2)):
            An, Bn = Agent._colnorm(A), Agent._colnorm(B)
            v = ((An.square() + Bn.square()) * zvar).sum(dim=1)
            gates.append((v / v.mean()).clamp(args.plast_gate_min, args.plast_gate_max))
        return gates


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

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    gated_layers = [agent.actor_fc1, agent.actor_fc2]

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rollout_env_ids = torch.arange(args.num_envs, device=device)
    b_env_ids = torch.arange(args.batch_size, device=device) % args.num_envs

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

        # fresh gene noise, frozen for this rollout and its update epochs
        agent.resample_gene_noise(args.num_envs, device)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, z, logprob, _, value = agent.get_beta_action_and_value(next_obs, env_ids=rollout_env_ids)
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

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, newvalue = agent.get_beta_action_and_value(
                    b_obs[mb_inds], z=b_zs[mb_inds], env_ids=b_env_ids[mb_inds]
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

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)

                # scale each gated neuron's realized Adam step by its plasticity gate:
                # w <- w_prev + gate * (w_adam - w_prev). Adam's m/v state stays untouched.
                gates = plasticity_gates(agent, args)
                prev = [(layer.weight.detach().clone(), layer.bias.detach().clone()) for layer in gated_layers]
                optimizer.step()
                with torch.no_grad():
                    for layer, gate, (w_prev, b_prev) in zip(gated_layers, gates, prev):
                        layer.weight.copy_(w_prev + gate.unsqueeze(1) * (layer.weight - w_prev))
                        layer.bias.copy_(b_prev + gate * (layer.bias - b_prev))

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
        with torch.no_grad():
            c = agent.concentrations()
            gate1, gate2 = plasticity_gates(agent, args)
            g1, b1, g2, b2 = agent.film()
            writer.add_scalar("plast/gene_c_mean", c.mean().item(), global_step)
            writer.add_scalar("plast/gene_c_min", c.min().item(), global_step)
            writer.add_scalar("plast/gene_c_max", c.max().item(), global_step)
            writer.add_scalar("plast/gamma1_absdev_mean", (g1 - 1.0).abs().mean().item(), global_step)
            writer.add_scalar("plast/gamma1_absdev_max", (g1 - 1.0).abs().max().item(), global_step)
            writer.add_scalar("plast/beta1_abs_mean", b1.abs().mean().item(), global_step)
            writer.add_scalar("plast/gamma2_absdev_mean", (g2 - 1.0).abs().mean().item(), global_step)
            writer.add_scalar("plast/beta2_abs_mean", b2.abs().mean().item(), global_step)
            writer.add_scalar("plast/gate1_min", gate1.min().item(), global_step)
            writer.add_scalar("plast/gate1_max", gate1.max().item(), global_step)
            writer.add_scalar("plast/gate2_min", gate2.min().item(), global_step)
            writer.add_scalar("plast/gate2_max", gate2.max().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
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
