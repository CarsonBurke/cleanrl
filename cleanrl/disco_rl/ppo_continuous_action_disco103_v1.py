# Disco103 v1: DeepMind's discovered RL update rule (Oh et al., Nature 2025) on MuJoCo.
# The frozen Disco103 meta-network (cleanrl/shared/disco103.py, parity-tested against
# the JAX original) writes the policy, y and z targets; the agent only distils them,
# with a Retrace categorical Q, an action-conditional LSTM model, an EMA target network
# and trajectory replay, all as in disco_rl's agent.
# Continuous port: Disco103 is equivariant over a discrete action set, so each state's
# set is K sampling-importance-resampled candidates from the behaviour Beta (the taken
# action first) with logits log pi - log mu. As K grows this categorical tends to pi, so
# the distilled KL becomes a sampled projection: grad = -sum_k (p_hat_k - p_k) grad log pi(a_k).
# Hypothesis: a rule meta-learned on Atari transfers through that representation.
import copy
import os
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared import disco103
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import gather_metrics
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

SAMPLE_EPS = 1e-6
NATIVE_TASKS = frozenset(("HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"))


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
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """Disco103's Adam learning rate"""
    max_abs_update: float = 1.0
    """element-wise clip on the Adam direction before the learning rate (disco_rl optimizer)"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 32
    """environment steps per env between learner phases"""
    gamma: float = 0.99
    """value discount (Disco103 used 0.997 on Atari)"""
    num_candidates: int = 16
    """K: actions per state presented to the meta-network (taken action + K-1 behaviour draws)"""
    segment_length: int = 32
    """T: replayed trajectory length; segments carry one extra successor row (disco_rl: 29)"""
    batch_segments: int = 64
    """B: trajectories per learner step"""
    replay_ratio: int = 8
    """expected learner visits per stored transition"""
    replay_capacity: int = 16384
    """stored transitions (rounded to whole rollouts per env); ~64 learner steps of staleness"""
    num_bins: int = 601
    """categorical value bins over signed-hyperbolic space"""
    max_abs_value: float = 30.0
    """half-range of the bins in h-space; |Q| up to ~960 normalized-reward units"""
    torso: tuple[int, ...] = (256, 256)
    """shared torso widths (disco_rl Atari used 512, 512)"""
    model_lstm_size: int = 128
    """action-conditional model LSTM width"""
    model_head_hidden: int = 128
    """hidden width of the model's z, Q and auxiliary-policy heads"""
    log_interval: int = 16
    """iterations between metric logs (one host sync each)"""

    # Execution controls
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads per run"""
    compile: bool = True
    """compile the learner step"""
    compile_mode: str = "reduce-overhead"
    """PyTorch compilation mode for the fixed-shape learner step"""
    non_blocking_transfers: bool = False
    """opt into event-protected asynchronous pinned transfers"""
    staggered_starts: bool = True
    """stagger parallel environments; warmup counts toward total_timesteps"""

    # to be filled in runtime
    updates_per_iteration: int = 0
    """learner steps per rollout (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def haiku_linear(in_features, out_features, std=None):
    """nn.Linear with hk.Linear's TruncatedNormal(1/sqrt(fan_in)), or (std), weight init."""
    layer = nn.Linear(in_features, out_features)
    std = std if std is not None else in_features ** -0.5
    nn.init.trunc_normal_(layer.weight, std=std, a=-2.0 * std, b=2.0 * std)
    nn.init.zeros_(layer.bias)
    return layer


def haiku_mlp(*sizes):
    layers = []
    for index, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        if index:
            layers.append(nn.ReLU())
        layers.append(haiku_linear(fan_in, fan_out))
    return nn.Sequential(*layers)


def concentrations(logits):
    return (1.0 + F.softplus(logits)).chunk(2, dim=-1)


def beta_log_prob(alpha, beta, native):
    """Native-space Beta log density summed over action dimensions."""
    return ((alpha - 1.0) * native.log() + (beta - 1.0) * torch.log1p(-native)
            + torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)).sum(-1)


def beta_entropy(alpha, beta):
    total = alpha + beta
    return (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(total)
            - (alpha - 1.0) * torch.digamma(alpha) - (beta - 1.0) * torch.digamma(beta)
            + (total - 2.0) * torch.digamma(total)).sum(-1)


def beta_kl(alpha, beta, other_alpha, other_beta):
    """KL(Beta(alpha, beta) || Beta(other)) summed over action dimensions."""
    total = alpha + beta
    return (torch.lgamma(other_alpha) + torch.lgamma(other_beta) - torch.lgamma(other_alpha + other_beta)
            - torch.lgamma(alpha) - torch.lgamma(beta) + torch.lgamma(total)
            + (alpha - other_alpha) * torch.digamma(alpha) + (beta - other_beta) * torch.digamma(beta)
            + (other_alpha - alpha + other_beta - beta) * torch.digamma(total)).sum(-1)


def sample_beta(alpha, beta):
    """Beta draws as Gamma ratios, clamped like the host sampler."""
    x, y = torch._standard_gamma(alpha), torch._standard_gamma(beta)
    return (x / (x + y)).clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)


class Agent(nn.Module):
    """disco_rl's MLP agent: torso -> (policy, y) heads plus an action-conditional LSTM model.

    The model transitions from a root embedding of the torso on an action and
    predicts z (meta-learned), Q (categorical) and the next state's policy.
    """

    def __init__(self, obs_dim, act_dim, args):
        super().__init__()
        widths = (obs_dim, *args.torso)
        layers = []
        for index, (fan_in, fan_out) in enumerate(zip(widths[:-1], widths[1:])):
            if index:
                layers.append(nn.ReLU())
            layers.append(haiku_linear(fan_in, fan_out))
        # hk.nets.MLP does not activate its final layer; the heads read it linearly.
        self.torso = nn.Sequential(*layers)
        width, lstm = args.torso[-1], args.model_lstm_size
        self.policy_head = haiku_linear(width, 2 * act_dim, std=1e-2)
        self.y_head = haiku_linear(width, disco103.PREDICTION_SIZE, std=1e-2)
        self.root = haiku_linear(width, lstm)
        self.transition = haiku_linear(act_dim + lstm, 4 * lstm)
        self.z_head = haiku_mlp(lstm, args.model_head_hidden, disco103.PREDICTION_SIZE)
        self.q_head = haiku_mlp(lstm, args.model_head_hidden, args.num_bins)
        self.aux_head = haiku_mlp(lstm, args.model_head_hidden, 2 * act_dim)
        self.act_dim = act_dim

    def actor(self):
        """Unregistered view of the acting path for the host mirror."""
        return nn.Sequential(*self.torso, self.policy_head)

    def state(self, obs):
        embedding = self.torso(obs)
        return embedding, self.policy_head(embedding), self.y_head(embedding)

    def model(self, embedding, native_actions):
        """One LSTM transition per candidate: [..., H], [..., K, A] -> z, Q logits, hidden."""
        cell = self.root(embedding)
        input_weight, hidden_weight = self.transition.weight.split(
            (self.act_dim, self.transition.out_features // 4), 1)
        root_gates = F.linear(torch.tanh(cell), hidden_weight, self.transition.bias)
        gates = F.linear(2.0 * native_actions - 1.0, input_weight) + root_gates.unsqueeze(-2)
        hidden, _ = disco103.lstm_gates(gates, cell.unsqueeze(-2))
        return self.z_head(hidden), self.q_head(hidden), hidden


class SegmentReplay:
    """Per-env FIFO trajectory ring on the GPU; samples uniform length-T segments."""

    def __init__(self, capacity, num_envs, fields, device):
        self.capacity, self.num_envs, self.head = capacity, num_envs, 0
        self.storage = {name: torch.zeros((capacity, num_envs, *shape), device=device)
                        for name, shape in fields.items()}
        self.rows = torch.zeros((), dtype=torch.long, device=device)
        self.inserted = 0

    def insert(self, rollout):
        steps = next(iter(rollout.values())).shape[0]
        for name, value in rollout.items():
            self.storage[name][self.head:self.head + steps].copy_(value)
        self.head = (self.head + steps) % self.capacity
        self.inserted += steps
        self.rows.fill_(self.inserted)

    def sample(self, batch, length):
        """Graph-safe: reads the GPU row count, never the host one."""
        newest = self.rows - length
        oldest = (self.rows - self.capacity).clamp_min(0)
        draw = torch.rand(batch, device=self.rows.device)
        start = oldest + (draw * (newest - oldest + 1).to(draw.dtype)).long().clamp_max(newest - oldest)
        time_index = (start[None] + torch.arange(length, device=start.device)[:, None]) % self.capacity
        env_index = torch.randint(0, self.num_envs, (batch,), device=start.device)
        return {name: value[time_index, env_index[None]] for name, value in self.storage.items()}


class DiscoAdam(torch.optim.Optimizer):
    """disco_rl optimizer: Adam direction, element-wise clip to max_abs_update, then -lr."""

    def __init__(self, params, lr, max_abs_update, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, dict(lr=lr, max_abs_update=max_abs_update, betas=betas, eps=eps))
        self.count = 0

    @torch.no_grad()
    def step(self):
        self.count += 1
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            grads = [p.grad for p in params]
            for p in params:
                if not self.state[p]:
                    self.state[p]["mu"], self.state[p]["nu"] = torch.zeros_like(p), torch.zeros_like(p)
            mu = [self.state[p]["mu"] for p in params]
            nu = [self.state[p]["nu"] for p in params]
            beta1, beta2 = group["betas"]
            torch._foreach_lerp_(mu, grads, 1.0 - beta1)
            torch._foreach_mul_(nu, beta2)
            torch._foreach_addcmul_(nu, grads, grads, value=1.0 - beta2)
            denominator = torch._foreach_sqrt(nu)
            torch._foreach_div_(denominator, (1.0 - beta2 ** self.count) ** 0.5)
            torch._foreach_add_(denominator, group["eps"])
            update = torch._foreach_div(mu, denominator)
            torch._foreach_mul_(update, 1.0 / (1.0 - beta1 ** self.count))
            torch._foreach_clamp_min_(update, -group["max_abs_update"])
            torch._foreach_clamp_max_(update, group["max_abs_update"])
            torch._foreach_add_(params, update, alpha=-group["lr"])


METRICS = ("loss/total", "loss/policy_kl", "loss/y", "loss/z", "loss/aux_policy", "loss/value",
           "policy/entropy", "policy/concentration", "policy/candidate_ess", "policy/target_ess",
           "policy/target_taken_prob", "policy/rho", "value/v", "value/q_target",
           "value/adv_abs", "value/normalized_td_abs", "value/target_q_spread")


class DiscoLearner(nn.Module):
    """One Disco103 learner step on a replayed batch (disco_rl Agent.learner_step)."""

    def __init__(self, agent, replay, args):
        super().__init__()
        self.agent = agent
        self.target = copy.deepcopy(agent).requires_grad_(False)
        self.meta = disco103.Disco103()
        self.value = disco103.TwoHotValue(args.num_bins, args.max_abs_value)
        self.replay = replay
        self.args = args
        self.register_buffer("lifetime_state", self.meta.initial_state())
        self.register_buffer("adv_ema", disco103.ema_init())
        self.register_buffer("td_ema", disco103.ema_init())

    def candidates(self, alpha, beta, count):
        return sample_beta(alpha.unsqueeze(-2).expand(*alpha.shape[:-1], count, alpha.shape[-1]),
                           beta.unsqueeze(-2).expand(*beta.shape[:-1], count, beta.shape[-1]))

    def proposal(self, taken, mu_alpha, mu_beta, alpha, beta):
        """Action set (taken action first), log mu, and the set's proposal log-density.

        Every candidate is a behaviour draw, so the proposal is mu itself.
        """
        del alpha, beta
        candidates = torch.cat((taken, self.candidates(mu_alpha, mu_beta, self.args.num_candidates - 1)), -2)
        log_mu = beta_log_prob(mu_alpha.unsqueeze(-2), mu_beta.unsqueeze(-2), candidates)
        return candidates, log_mu, log_mu

    def loss(self):
        args = self.args
        steps = args.segment_length
        # T+1 rows: row t+1 is step t's successor unless the episode ended at t.
        batch = self.replay.sample(args.batch_segments, steps + 1)
        obs, final_obs = batch["observations"], batch["transition_observations"][:-1]
        rewards, terminations = batch["rewards"][:-1], batch["terminations"][:-1]
        resets = torch.maximum(terminations, batch["truncations"][:-1])
        mu_alpha, mu_beta = concentrations(batch["behaviour_logits"])

        embedding, policy, y = self.agent.state(obs)
        alpha, beta = concentrations(policy)
        candidates, log_mu, log_proposal = self.proposal(
            batch["native_actions"].unsqueeze(-2), mu_alpha, mu_beta, alpha.detach(), beta.detach())
        log_pi = beta_log_prob(alpha.unsqueeze(-2), beta.unsqueeze(-2), candidates)
        logits = log_pi - log_proposal
        z, q, hidden = self.agent.model(embedding, candidates)
        aux_alpha, aux_beta = concentrations(self.agent.aux_head(hidden[:-1, :, 0]))

        with torch.no_grad():
            # Final observations of finished episodes, with online-policy candidates (uniform logits).
            final_embedding, final_policy, final_y = self.agent.state(final_obs)
            final_alpha, final_beta = concentrations(final_policy)
            final_candidates = self.candidates(final_alpha, final_beta, args.num_candidates)
            final_z, final_q, _ = self.agent.model(final_embedding, final_candidates)
            target_embedding, target_policy, target_y = self.target.state(obs)
            target_alpha, target_beta = concentrations(target_policy)
            target_logits = beta_log_prob(target_alpha.unsqueeze(-2), target_beta.unsqueeze(-2),
                                          candidates) - log_proposal
            target_z, target_q, _ = self.target.model(target_embedding, candidates)
            final_target_embedding, _, final_target_y = self.target.state(final_obs)
            final_target_z, final_target_q, _ = self.target.model(final_target_embedding, final_candidates)

            def successor(current, final):
                ended = resets.bool().view(*resets.shape, *[1] * (final.dim() - 2))
                return torch.where(ended, final, current[1:])

            # value_utils.get_value_outs with Q networks: values weight Q by the online policy.
            weights = torch.softmax(logits.detach(), -1)
            q_scalar, target_q_scalar = self.value.scalar(q.detach()), self.value.scalar(target_q)
            values = (weights * q_scalar).sum(-1)
            target_values = (weights * target_q_scalar).sum(-1)
            next_values = successor(values, self.value.scalar(final_q).mean(-1))
            next_target_values = successor(target_values, self.value.scalar(final_target_q).mean(-1))
            next_alpha, next_beta = successor(alpha.detach(), final_alpha), successor(beta.detach(), final_beta)
            rho = torch.exp(log_pi[:-1, :, 0].detach() - log_mu[:-1, :, 0])
            trace = disco103.TD_LAMBDA * rho.clamp_max(1.0)
            discounts = args.gamma * (1.0 - terminations)
            continues = 1.0 - resets
            # Retrace: G_t = r_t + g_t [V'(s_t+1) + c_t+1 (G_t+1 - Q'(s_t+1, a_t+1))], cut at resets.
            target = rewards[-1] + discounts[-1] * next_target_values[-1]
            q_targets = [target]
            for t in range(steps - 2, -1, -1):
                correction = continues[t] * trace[t + 1] * (target - target_q_scalar[t + 1, :, 0])
                target = rewards[t] + discounts[t] * (next_target_values[t] + correction)
                q_targets.append(target)
            q_target = torch.stack(q_targets[::-1])
            advantages = q_target - target_values[:-1]
            qv_advantages = target_q_scalar[:-1] - target_values[:-1].unsqueeze(-1)
            q_td = q_target - q_scalar[:-1, :, 0]
            adv_ema = disco103.ema_update(self.adv_ema, advantages)
            td_ema = disco103.ema_update(self.td_ema, q_td)
            pi_hat, y_hat, z_hat, lifetime_state = self.meta(disco103.MetaInputs(
                logits=logits[:-1].detach(),
                next_logits=successor(logits.detach(), torch.zeros_like(final_q[..., 0])),
                behaviour_logits=(log_mu - log_proposal)[:-1], target_logits=target_logits[:-1],
                rewards=rewards, terminations=terminations, resets=resets,
                values=values[:-1], next_values=next_values, advantages=advantages,
                normalized_advantages=disco103.ema_normalize(adv_ema, advantages),
                y=y[:-1].detach(), next_y=successor(y.detach(), final_y),
                target_y=target_y[:-1], next_target_y=successor(target_y, final_target_y),
                z=z[:-1].detach(), next_z=successor(z.detach(), final_z),
                target_z=target_z[:-1], next_target_z=successor(target_z, final_target_z),
                target_q=target_q_scalar[:-1], qv_advantages=qv_advantages,
                normalized_qv_advantages=disco103.ema_normalize(adv_ema, qv_advantages),
            ), self.lifetime_state)

        policy_loss = disco103.categorical_kl(pi_hat, logits[:-1])
        y_loss = disco103.categorical_kl(y_hat, y[:-1])
        z_loss = disco103.categorical_kl(z_hat, z[:-1, :, 0])
        aux_loss = beta_kl(next_alpha, next_beta, aux_alpha, aux_beta) * (1.0 - terminations)
        value_loss = self.value.loss(q[:-1, :, 0], q_target)
        losses = torch.stack((policy_loss.mean(), y_loss.mean(), z_loss.mean(), aux_loss.mean(),
                              value_loss.mean()))
        costs = losses.new_tensor((disco103.PI_COST, disco103.Y_COST, disco103.Z_COST,
                                   disco103.AUX_POLICY_COST, disco103.VALUE_COST))
        total = (costs * losses).sum()

        with torch.no_grad():
            def ess(categorical_logits):
                probabilities = torch.softmax(categorical_logits, -1)
                return (1.0 / probabilities.square().sum(-1)).mean() / args.num_candidates

            metrics = torch.stack((
                total, *losses, beta_entropy(alpha, beta).mean(), (alpha + beta).mean(),
                ess(logits[:-1]), ess(pi_hat), torch.softmax(pi_hat, -1)[..., 0].mean(), rho.mean(),
                values.mean(), q_target.mean(), advantages.abs().mean(),
                disco103.ema_normalize(td_ema, q_td, subtract_mean=False).abs().mean(),
                target_q_scalar.std(-1).mean(),
            ))
        return total, metrics, torch.stack((adv_ema, td_ema)), lifetime_state


def validate_args(args):
    if not args.cuda:
        raise ValueError("Disco103 requires CUDA")
    if args.num_candidates < 2 or args.segment_length < 2:
        raise ValueError("need at least two candidates and two steps per segment")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    samples = args.replay_ratio * args.num_steps * args.num_envs
    if samples % (args.batch_segments * args.segment_length):
        raise ValueError("replay_ratio * num_steps * num_envs must be a multiple of B * T")
    args.updates_per_iteration = samples // (args.batch_segments * args.segment_length)
    args.replay_capacity = max(args.num_steps, args.replay_capacity // args.num_envs
                               // args.num_steps * args.num_steps)
    if args.replay_capacity <= args.segment_length:
        raise ValueError("replay must hold a full segment per env")
    return args


def make_training_env(args, run_name):
    backend = args.env_backend
    if backend == "auto":
        backend = "native" if args.env_id in NATIVE_TASKS and gym.__version__ == "0.29.1" else "sync"
    return make_mujoco_vector_env(
        args.env_id, args.num_envs, backend=backend,
        num_threads=min(args.env_threads, args.num_envs),
        capture_video=args.capture_video, run_name=run_name,
    )


def main():
    args = validate_args(tyro.cli(Args))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    # Full FP32: candidate Q differences are smaller than TF32 rounding of the readout.
    configure_runtime(cudnn_deterministic=args.torch_deterministic,
                      matmul_precision="highest", allow_tf32=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    horizon = episode_horizon(args.env_id) if args.staggered_starts and args.num_envs > 1 else 0
    args.num_iterations = (args.total_timesteps - horizon * args.num_envs) // (args.num_steps * args.num_envs)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   sync_tensorboard=True, config=vars(args), name=run_name,
                   monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"runs/{run_name}")
    resources = ExitStack()
    resources.callback(writer.close)
    try:
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                        "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()))
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        space = envs.single_action_space
        obs_shape = envs.single_observation_space.shape
        act_dim = int(np.prod(space.shape))
        agent = Agent(int(np.prod(obs_shape)), act_dim, args).to(device)
        replay = SegmentReplay(args.replay_capacity, args.num_envs, {
            "observations": obs_shape, "transition_observations": obs_shape,
            "native_actions": (act_dim,), "behaviour_logits": (2 * act_dim,),
            "rewards": (), "terminations": (), "truncations": (),
        }, device)
        learner = DiscoLearner(agent, replay, args).to(device)
        optimizer = DiscoAdam(agent.parameters(), args.learning_rate, args.max_abs_update)
        online_parameters = list(agent.parameters())
        target_parameters = list(learner.target.parameters())
        learner_loss = learner.loss
        if args.compile:
            learner_loss = torch.compile(learner_loss, mode=args.compile_mode, fullgraph=True)

        host_actor = make_host_mirror(agent.actor(), args.num_envs)
        sampler = np.random.default_rng(args.seed)
        sample_actions = make_beta_sampler(args.num_envs, act_dim, np.asarray(space.low).reshape(-1),
                                           np.asarray(space.high).reshape(-1))

        def act(observations):
            logits = host_actor(observations)
            native, action = sample_actions(logits, sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return logits, native, action.reshape((args.num_envs,) + space.shape)

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   store_transition_observations=True,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={"observations": obs_shape, "native_actions": (act_dim,),
                                           "behaviour_logits": (2 * act_dim,)})
        resources.callback(transfer.close)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        metric_sums = torch.zeros(len(METRICS) + 1, device=device)
        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)
        if horizon:
            phases = compute_phase_offsets(args.num_envs, horizon, args.seed)
            writer.add_text("initial_phase_offsets", ",".join(map(str, phases)))
            warm = run_phase_warmup(envs, obs_norm=obs_norm, rew_norm=rew_norm,
                                    act_fn=lambda observations: act(observations)[2], horizon=horizon,
                                    phase_offsets=phases, seed=args.seed)
            next_obs_np, global_step, suppress = warm.next_obs, warm.transitions, warm.suppress_mask
        else:
            raw_obs, _ = envs.reset(seed=args.seed)
            next_obs_np, global_step = obs_norm.normalize(raw_obs), 0
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start_time, global_step)
        interval_start, interval_step, updates = time.perf_counter(), global_step, 0

        for iteration in range(1, args.num_iterations + 1):
            host_actor.refresh()
            for step in range(args.num_steps):
                with timer.span("rollout", use_cuda=False):
                    obs_step = next_obs_np
                    logits, native, host_action = act(obs_step)
                with timer.span("env", use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(host_action)
                with timer.span("normalize_transfer", use_cuda=False):
                    reward = rew_norm.normalize(raw_reward, terms)
                    next_obs_np, transition_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    transfer.push(step, reward, terms, truncs, transition_observations=transition_obs,
                                  observations=obs_step, native_actions=native, behaviour_logits=logits)
                global_step += args.num_envs
                for index, info in enumerate(infos.get("final_info", ())):
                    if info and "episode" in info:
                        if suppress[index]:
                            suppress[index] = False
                            continue
                        episode_return = float(info["episode"]["r"])
                        print(f"global_step={global_step}, episodic_return={episode_return}")
                        writer.add_scalar("charts/episodic_return", episode_return, global_step)
                        writer.add_scalar("charts/episodic_length", float(info["episode"]["l"]), global_step)

            with timer.span("update"):
                batch = transfer.upload()
                replay.insert({"observations": batch.fields["observations"],
                               "transition_observations": batch.transition_observations,
                               "native_actions": batch.fields["native_actions"],
                               "behaviour_logits": batch.fields["behaviour_logits"],
                               "rewards": batch.rewards, "terminations": batch.terminations,
                               "truncations": batch.truncations})
                ready = (replay.inserted > args.segment_length and replay.inserted * args.num_envs
                         >= args.batch_segments * (args.segment_length + 1))
                for _ in range(args.updates_per_iteration if ready else 0):
                    if args.compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    loss, metrics, emas, lifetime_state = learner_loss()
                    learner.adv_ema.copy_(emas[0])
                    learner.td_ema.copy_(emas[1])
                    learner.lifetime_state.copy_(lifetime_state)
                    metric_sums[:-1] += metrics
                    agent.zero_grad(set_to_none=True)
                    loss.backward()
                    metric_sums[-1] += torch.linalg.vector_norm(
                        torch.stack(torch._foreach_norm([p.grad for p in online_parameters])))
                    # disco_rl moves the target toward the pre-update parameters.
                    torch._foreach_lerp_(target_parameters, online_parameters,
                                         1.0 - disco103.TARGET_PARAMS_COEFF)
                    optimizer.step()
                    updates += 1

            if iteration % args.log_interval == 0 or iteration == args.num_iterations:
                if updates:
                    logged = gather_metrics({name: metric_sums[i] / updates
                                             for i, name in enumerate((*METRICS, "grad/norm"))})
                    if any(not np.isfinite(value) for value in logged.values()):
                        raise FloatingPointError(f"nonfinite Disco103 learner metrics: {logged}")
                    for name, value in logged.items():
                        writer.add_scalar(name, value, global_step)
                    metric_sums.zero_()
                    updates = 0
                now = time.perf_counter()
                writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
                writer.add_scalar("charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step)
                writer.add_scalar("charts/learner_updates", optimizer.count, global_step)
                for phase, timing in timer.summary().items():
                    writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
                timer.reset()
                print(f"SPS: {int(global_step / (now - start_time))}")
                interval_start, interval_step = now, global_step

        torch.save({"model": agent.state_dict(), "args": vars(args), "global_step": global_step},
                   f"runs/{run_name}/{args.exp_name}.pt")
        if args.save_model:
            torch.save(agent.state_dict(), f"runs/{run_name}/{args.exp_name}.cleanrl_model")
    finally:
        resources.close()


if __name__ == "__main__":
    main()
