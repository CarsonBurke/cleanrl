# PPO + packed-static Sparse-K coordinate control v4.
#
# This is a causal control, not a topology learner. It preserves v1's fixed
# random graph and PPO loop while using the flat gather/index_add kernel needed
# by future variable fan-in. The paired arms have identical initial functions:
# `effective` stores w_eff~N(0,2/k) and sums it directly; `raw` stores
# w_raw=sqrt(k)*w_eff and divides its contribution by sqrt(k). SGD would map the
# coordinates predictably, but Adam plus global clipping makes the raw arm's
# function-space optimization materially different. Capacity, depth, probes,
# compute pricing, and rewiring are disabled.
import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional

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

    # Sparse-K
    width: int = 256
    """hidden width N (units per layer)"""
    k: int = 64
    """incoming connections per unit (dense-64 fan-in match)"""
    num_hidden_layers: int = 2
    """number of sparse hidden layers"""
    pool: str = "prev"
    """prev | prior — previous-layer only vs all-prior prefix pool"""
    weight_coordinate: str = "effective"
    """effective | raw — paired Adam-coordinate ablation"""
    rewire: str = "none"
    """none | set | thresh | learned | meta"""
    zeta: float = 0.3
    """SET: fraction of lowest-|w| edges to rewire on episode end"""
    utility_threshold: float = 0.02
    """thresh: rewire edges with EMA utility below this (absolute)"""
    utility_ema: float = 0.9
    """EMA rate for edge utility (|pre| * |w|)"""
    learned_thresh_init: float = 0.1
    """learned: initial global threshold X = softplus(logit)"""
    learned_thresh_tau: float = 0.1
    """learned: soft gate temperature; gate = sigmoid((u - X) / tau); wider ⇒ less saturation"""
    meta_q: float = 0.1
    """meta: fraction of mature edges (lowest u) to rewire on episode end"""
    meta_age_min: int = 100
    """meta: optim-steps before an edge is eligible for rewire"""
    meta_h_decay: float = 0.9
    """meta: EMA decay for grad memory h"""
    meta_u_decay: float = 0.9
    """meta: EMA decay for usefulness u"""
    compile: bool = False
    compile_mode: str = "default"

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
    return layer


def _inv_softplus(y: float) -> float:
    """x such that softplus(x) = y for y > 0."""
    y = max(float(y), 1e-6)
    return math.log(math.expm1(y))


class SparseKLinear(nn.Module):
    """Linear layer with fixed fan-in K hard connections per output unit."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int,
        zeta: float = 0.3,
        rewire_mode: str = "none",
        utility_threshold: float = 0.02,
        utility_ema: float = 0.9,
        learned_thresh_tau: float = 0.05,
        meta_q: float = 0.1,
        meta_age_min: int = 100,
        meta_h_decay: float = 0.9,
        meta_u_decay: float = 0.9,
        weight_coordinate: str = "effective",
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.k = int(min(k, in_features))
        self.zeta = float(zeta)
        self.rewire_mode = rewire_mode
        self.utility_threshold = float(utility_threshold)
        self.utility_ema = float(utility_ema)
        self.learned_thresh_tau = float(learned_thresh_tau)
        self.meta_q = float(meta_q)
        self.meta_age_min = int(meta_age_min)
        self.meta_h_decay = float(meta_h_decay)
        self.meta_u_decay = float(meta_u_decay)
        if weight_coordinate not in ("effective", "raw"):
            raise ValueError(f"weight_coordinate must be effective|raw, got {weight_coordinate}")
        self.weight_coordinate = weight_coordinate
        self.fully_dense = self.k >= self.in_features
        # Optional shared Parameter set by Agent for rewire_mode == "learned"
        self.threshold_logit: Optional[nn.Parameter] = None

        self.weight = nn.Parameter(torch.empty(self.out_features, self.k))
        self.bias = nn.Parameter(torch.zeros(self.out_features))
        self.register_buffer("indices", torch.zeros(self.out_features, self.k, dtype=torch.long))
        self.register_buffer(
            "destinations",
            torch.arange(self.out_features).repeat_interleave(self.k),
            persistent=False,
        )
        # thresh: start high so cold edges are not mass-rewired before EMA has signal.
        # learned: start near X so soft gate is not saturated (needs ∇X early).
        if rewire_mode == "thresh":
            init_u = max(utility_threshold * 10.0, 1.0)
        elif rewire_mode == "learned":
            init_u = float(utility_threshold)
        else:
            init_u = 0.0
        self.register_buffer("utility", torch.full((self.out_features, self.k), init_u))
        # Meta (grad-agreement) state — only used when rewire_mode == "meta"
        self.register_buffer("meta_h", torch.zeros(self.out_features, self.k))
        self.register_buffer("meta_u", torch.zeros(self.out_features, self.k))
        self.register_buffer("age", torch.zeros(self.out_features, self.k))
        self.last_rewired = 0
        self._track_utility = rewire_mode in ("thresh", "learned", "set") and not self.fully_dense

        self.reset_parameters()
        self._init_indices()

    def reset_parameters(self):
        # Draw the same effective coefficients in both arms. The raw arm stores
        # an exactly mapped parameterization without consuming additional RNG.
        std = math.sqrt(2.0 / max(self.k, 1))
        nn.init.normal_(self.weight, mean=0.0, std=std)
        if self.weight_coordinate == "raw":
            with torch.no_grad():
                self.weight.mul_(math.sqrt(self.k))
        nn.init.zeros_(self.bias)

    def effective_weight(self) -> torch.Tensor:
        if self.weight_coordinate == "raw":
            return self.weight / math.sqrt(self.k)
        return self.weight

    def _init_indices(self):
        with torch.no_grad():
            if self.fully_dense:
                self.indices.copy_(
                    torch.arange(self.in_features, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(self.out_features, -1)
                )
            else:
                for i in range(self.out_features):
                    self.indices[i] = torch.randperm(self.in_features)[: self.k]

    def resolve_threshold(self) -> torch.Tensor:
        """Scalar threshold tensor (may require grad if learned)."""
        if self.rewire_mode == "learned":
            if self.threshold_logit is None:
                raise RuntimeError("learned rewire requires threshold_logit on layers")
            return F.softplus(self.threshold_logit)
        return self.weight.new_tensor(self.utility_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Deliberately use the packed flat-edge kernel in both arms. The only
        # difference is the stored Adam coordinate.
        sources = self.indices.reshape(-1)
        gathered = x[:, sources]
        contrib = gathered * self.weight.reshape(-1)
        if self.weight_coordinate == "raw":
            contrib = contrib / math.sqrt(self.k)
        if self.rewire_mode == "learned" and not self.fully_dense:
            # Soft keep-gate so global X = softplus(logit) receives task gradients.
            # High utility relative to X → gate→1; below X → gate→0.
            # Centered gate (0.5 at u=X) keeps early gradients alive.
            X = self.resolve_threshold()
            tau = max(self.learned_thresh_tau, 1e-4)
            gate = torch.sigmoid((self.utility.detach() - X) / tau)
            contrib = contrib.view(x.shape[0], self.out_features, self.k) * gate
            contrib = contrib.reshape(x.shape[0], -1)
        y = x.new_zeros((x.shape[0], self.out_features))
        y.index_add_(1, self.destinations, contrib)
        y = y + self.bias
        if self._track_utility and self.training:
            with torch.no_grad():
                pre = gathered.detach().abs().mean(dim=0).view(self.out_features, self.k)
                u = pre * self.effective_weight().detach().abs()
                self.utility.mul_(self.utility_ema).add_(u, alpha=1.0 - self.utility_ema)
        return y

    @torch.no_grad()
    def update_meta_from_grad(self) -> None:
        """IDBD-flavored usefulness: h←EMA(-g), u←EMA((-g)·h). Call after backward."""
        if self.rewire_mode != "meta" or self.fully_dense:
            return
        g = self.weight.grad
        if g is None:
            return
        lh = self.meta_h_decay
        lu = self.meta_u_decay
        self.meta_h.mul_(lh).add_(-g, alpha=1.0 - lh)
        s = (-g) * self.meta_h
        self.meta_u.mul_(lu).add_(s, alpha=1.0 - lu)
        self.age.add_(1.0)

    def rewire(self, optimizer: Optional[optim.Optimizer] = None) -> int:
        if self.rewire_mode == "none" or self.fully_dense:
            self.last_rewired = 0
            return 0
        if self.rewire_mode == "set":
            n = self._rewire_set(optimizer)
        elif self.rewire_mode in ("thresh", "learned"):
            n = self._rewire_thresh(optimizer)
        elif self.rewire_mode == "meta":
            n = self._rewire_meta(optimizer)
        else:
            raise ValueError(f"unknown rewire_mode={self.rewire_mode}")
        self.last_rewired = n
        return n

    def _rewire_set(self, optimizer: Optional[optim.Optimizer]) -> int:
        flat_abs = self.weight.data.abs().reshape(-1)
        n = flat_abs.numel()
        n_drop = max(1, int(self.zeta * n))
        _, flat_idx = torch.topk(flat_abs, n_drop, largest=False)
        rows = torch.div(flat_idx, self.k, rounding_mode="floor")
        cols = flat_idx % self.k
        return self._replace_slots(rows, cols, optimizer)

    def _rewire_thresh(self, optimizer: Optional[optim.Optimizer]) -> int:
        with torch.no_grad():
            X = float(self.resolve_threshold().detach())
        mask = self.utility < X
        if not bool(mask.any()):
            return 0
        rows, cols = mask.nonzero(as_tuple=True)
        return self._replace_slots(rows, cols, optimizer)

    def _rewire_meta(self, optimizer: Optional[optim.Optimizer]) -> int:
        """Among mature edges (age ≥ T), rewire bottom meta_q by usefulness u."""
        mature = self.age >= float(self.meta_age_min)
        n_mature = int(mature.sum().item())
        if n_mature <= 0:
            return 0
        n_drop = max(1, int(self.meta_q * n_mature))
        n_drop = min(n_drop, n_mature)
        # Only rank mature edges (avoids picking +inf placeholders)
        mature_idx = mature.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        mature_u = self.meta_u.reshape(-1)[mature_idx]
        _, local = torch.topk(mature_u, n_drop, largest=False)
        flat_idx = mature_idx[local]
        rows = torch.div(flat_idx, self.k, rounding_mode="floor")
        cols = flat_idx % self.k
        return self._replace_slots(rows, cols, optimizer)

    def _replace_slots(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor,
        optimizer: Optional[optim.Optimizer],
    ) -> int:
        n = int(rows.numel())
        if n == 0:
            return 0
        device = self.weight.device
        std = math.sqrt(2.0 / max(self.k, 1))
        if self.weight_coordinate == "raw":
            std *= math.sqrt(self.k)
        # Vectorized random candidates; resolve collisions per-row with a few retries
        for attempt in range(8):
            cand = torch.randint(0, self.in_features, (n,), device=device)
            # reject candidates already present in the same row (except the dying slot)
            for i in range(n):
                r = int(rows[i])
                c = int(cols[i])
                row_idx = self.indices[r]
                # allow cand if not in row, or equals the slot being replaced
                if (row_idx == cand[i]).any() and int(row_idx[c]) != int(cand[i]):
                    # resample this one
                    for _ in range(16):
                        alt = torch.randint(0, self.in_features, (1,), device=device).item()
                        if not (row_idx == alt).any() or int(row_idx[c]) == alt:
                            cand[i] = alt
                            break
            break

        self.indices[rows, cols] = cand
        self.weight.data[rows, cols] = torch.randn(n, device=device) * std
        self.utility[rows, cols] = 0.0
        self.meta_h[rows, cols] = 0.0
        self.meta_u[rows, cols] = 0.0
        self.age[rows, cols] = 0.0

        if optimizer is not None:
            state = optimizer.state.get(self.weight)
            if state is not None:
                for key, buf in state.items():
                    if torch.is_tensor(buf) and buf.shape == self.weight.shape:
                        buf[rows, cols] = 0
        return n


class SparseTrunk(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        width: int,
        k: int,
        num_layers: int,
        any_prior: bool,
        rewire_mode: str,
        zeta: float,
        utility_threshold: float,
        utility_ema: float,
        learned_thresh_tau: float = 0.05,
        meta_q: float = 0.1,
        meta_age_min: int = 100,
        meta_h_decay: float = 0.9,
        meta_u_decay: float = 0.9,
        weight_coordinate: str = "effective",
    ):
        super().__init__()
        self.any_prior = any_prior
        self.width = width
        self.layers = nn.ModuleList()
        layer_kw = dict(
            zeta=zeta,
            rewire_mode=rewire_mode,
            utility_threshold=utility_threshold,
            utility_ema=utility_ema,
            learned_thresh_tau=learned_thresh_tau,
            meta_q=meta_q,
            meta_age_min=meta_age_min,
            meta_h_decay=meta_h_decay,
            meta_u_decay=meta_u_decay,
            weight_coordinate=weight_coordinate,
        )
        # layer 0: obs -> width
        self.layers.append(SparseKLinear(obs_dim, width, k, **layer_kw))
        for i in range(1, num_layers):
            in_dim = obs_dim + i * width if any_prior else width
            self.layers.append(SparseKLinear(in_dim, width, k, **layer_kw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts: List[torch.Tensor] = [x]
        h = x
        for i, layer in enumerate(self.layers):
            if i == 0:
                inp = x
            elif self.any_prior:
                inp = torch.cat(acts, dim=-1)
            else:
                inp = h
            h = F.silu(layer(inp))
            acts.append(h)
        return h

    def rewire_all(self, optimizer: Optional[optim.Optimizer] = None) -> int:
        total = 0
        for layer in self.layers:
            total += layer.rewire(optimizer)
        return total

    def update_meta_from_grad(self) -> None:
        for layer in self.layers:
            layer.update_meta_from_grad()

    def sparse_layers(self) -> List[SparseKLinear]:
        return list(self.layers)


class Agent(nn.Module):
    def __init__(self, envs, args: Args):
        super().__init__()
        self.args = args
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        any_prior = args.pool == "prior"
        # For learned mode, init layer cold-utility relative to init X
        util_thresh = (
            args.learned_thresh_init if args.rewire == "learned" else args.utility_threshold
        )
        common = dict(
            width=args.width,
            k=args.k,
            num_layers=args.num_hidden_layers,
            any_prior=any_prior,
            rewire_mode=args.rewire,
            zeta=args.zeta,
            utility_threshold=util_thresh,
            utility_ema=args.utility_ema,
            learned_thresh_tau=args.learned_thresh_tau,
            meta_q=args.meta_q,
            meta_age_min=args.meta_age_min,
            meta_h_decay=args.meta_h_decay,
            meta_u_decay=args.meta_u_decay,
            weight_coordinate=args.weight_coordinate,
        )
        self.actor_trunk = SparseTrunk(obs_dim, **common)
        self.critic_trunk = SparseTrunk(obs_dim, **common)
        # One global threshold for all sparse layers (actor + critic)
        if args.rewire == "learned":
            self.threshold_logit = nn.Parameter(
                torch.tensor(_inv_softplus(args.learned_thresh_init), dtype=torch.float32)
            )
            for layer in self.actor_trunk.sparse_layers() + self.critic_trunk.sparse_layers():
                layer.threshold_logit = self.threshold_logit
        else:
            self.register_parameter("threshold_logit", None)

        self.actor_alpha = layer_init(nn.Linear(args.width, act_dim), std=0.01)
        self.actor_beta = layer_init(nn.Linear(args.width, act_dim), std=0.01)
        self.critic_out = layer_init(nn.Linear(args.width, 1), std=1.0)
        self.register_buffer(
            "action_low", torch.tensor(envs.single_action_space.low, dtype=torch.float32)
        )
        self.register_buffer(
            "action_high", torch.tensor(envs.single_action_space.high, dtype=torch.float32)
        )
        self._rewire_frozen = False
        self.last_rewired_total = 0

    def freeze_rewire(self, frozen: bool):
        self._rewire_frozen = frozen

    def current_threshold(self) -> Optional[torch.Tensor]:
        if self.args.rewire == "learned" and self.threshold_logit is not None:
            return F.softplus(self.threshold_logit)
        if self.args.rewire == "thresh":
            return torch.tensor(self.args.utility_threshold)
        return None

    def rewire_on_episode_end(self, optimizer: Optional[optim.Optimizer] = None) -> int:
        if self._rewire_frozen or self.args.rewire == "none":
            return 0
        n = self.actor_trunk.rewire_all(optimizer) + self.critic_trunk.rewire_all(optimizer)
        self.last_rewired_total = n
        return n

    def update_meta_from_grad(self) -> None:
        if self.args.rewire != "meta":
            return
        self.actor_trunk.update_meta_from_grad()
        self.critic_trunk.update_meta_from_grad()

    def _dist(self, x: torch.Tensor) -> Beta:
        h = self.actor_trunk(x)
        alpha = 1.0 + F.softplus(self.actor_alpha(h))
        beta = 1.0 + F.softplus(self.actor_beta(h))
        return Beta(alpha, beta)

    def _z_to_action(self, z: torch.Tensor) -> torch.Tensor:
        return self.action_low + (self.action_high - self.action_low) * z

    def _action_to_z(self, action: torch.Tensor) -> torch.Tensor:
        z = (action - self.action_low) / (self.action_high - self.action_low)
        return z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic_out(self.critic_trunk(x))

    def get_beta_action_and_value(self, x: torch.Tensor, z: Optional[torch.Tensor] = None):
        dist = self._dist(x)
        if z is None:
            z = dist.sample().clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        else:
            z = z.clamp(SAMPLE_EPS, 1.0 - SAMPLE_EPS)
        action = self._z_to_action(z)
        logprob = dist.log_prob(z).sum(1)
        value = self.get_value(x)
        return action, z, logprob, dist.entropy().sum(1), value

    def get_action_and_value(self, x, action=None):
        z = None if action is None else self._action_to_z(action)
        action, _, logprob, entropy, value = self.get_beta_action_and_value(x, z)
        return action, logprob, entropy, value


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.pool in ("prev", "prior"), f"pool must be prev|prior, got {args.pool}"
    if args.rewire != "none":
        raise ValueError("packed coordinate control requires --rewire none")
    if args.weight_coordinate not in ("effective", "raw"):
        raise ValueError("weight_coordinate must be effective|raw")
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("packed SPK experiments require CUDA")

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
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.compile:
        agent = torch.compile(agent, mode=args.compile_mode)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    zs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    episode_rewire_events = 0
    episode_rewire_edges = 0

    # unwrap compile for rewire helpers
    def raw_agent():
        return agent._orig_mod if hasattr(agent, "_orig_mod") else agent

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        raw_agent().freeze_rewire(False)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, z, logprob, _, value = agent.get_beta_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            zs[step] = z
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done_np.astype(np.float32)).to(device)

            # Episode-end rewiring (shared topology; trigger if any env finished)
            if next_done_np.any() and args.rewire != "none":
                n = raw_agent().rewire_on_episode_end(optimizer)
                episode_rewire_events += 1
                episode_rewire_edges += n

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
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_zs = zs.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Freeze connectivity during PPO update
        raw_agent().freeze_rewire(True)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        named_sparse_layers = [
            (f"actor/layer_{index}", layer)
            for index, layer in enumerate(raw_agent().actor_trunk.sparse_layers())
        ] + [
            (f"critic/layer_{index}", layer)
            for index, layer in enumerate(raw_agent().critic_trunk.sparse_layers())
        ]
        preclip_grad_norm_sum = torch.zeros((), device=device)
        clip_coefficient_sum = torch.zeros((), device=device)
        clipped_steps = torch.zeros((), device=device)
        sampled_layer_grad_norms = {}
        sampled_layer_effective_grad_norms = {}
        sampled_layer_updates = {}
        optimizer_steps = 0
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, newvalue = agent.get_beta_action_and_value(
                    b_obs[mb_inds], b_zs[mb_inds]
                )
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
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                sample_diagnostics = optimizer_steps == 0
                if sample_diagnostics:
                    effective_before = {
                        name: layer.effective_weight().detach().clone()
                        for name, layer in named_sparse_layers
                    }
                    for name, layer in named_sparse_layers:
                        stored_gradient = layer.weight.grad.detach()
                        sampled_layer_grad_norms[name] = stored_gradient.norm()
                        if layer.weight_coordinate == "raw":
                            effective_gradient = stored_gradient * math.sqrt(layer.k)
                        else:
                            effective_gradient = stored_gradient
                        sampled_layer_effective_grad_norms[name] = effective_gradient.norm()
                preclip_grad_norm = nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm
                )
                # Meta usefulness uses post-backward grads (before Adam step)
                if args.rewire == "meta":
                    raw_agent().update_meta_from_grad()
                optimizer.step()
                with torch.no_grad():
                    if sample_diagnostics:
                        sampled_layer_updates = {
                            name: (
                                layer.effective_weight() - effective_before[name]
                            ).abs().mean()
                            for name, layer in named_sparse_layers
                        }
                    clip_coefficient = torch.clamp(
                        args.max_grad_norm / (preclip_grad_norm + 1e-6), max=1.0
                    )
                    clip_coefficient_sum += clip_coefficient
                    clipped_steps += (preclip_grad_norm > args.max_grad_norm).to(
                        clipped_steps.dtype
                    )
                    preclip_grad_norm_sum += preclip_grad_norm
                    optimizer_steps += 1

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        raw_agent().freeze_rewire(False)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # sparse diagnostics
        with torch.no_grad():
            util_means = []
            abs_w = []
            meta_u_means = []
            mature_fracs = []
            for trunk in (raw_agent().actor_trunk, raw_agent().critic_trunk):
                for layer in trunk.sparse_layers():
                    abs_w.append(layer.effective_weight().detach().abs().mean().item())
                    if layer._track_utility:
                        util_means.append(layer.utility.mean().item())
                    if layer.rewire_mode == "meta" and not layer.fully_dense:
                        meta_u_means.append(layer.meta_u.mean().item())
                        mature_fracs.append(
                            (layer.age >= float(layer.meta_age_min)).float().mean().item()
                        )
            writer.add_scalar("sparse/mean_abs_w", float(np.mean(abs_w)) if abs_w else 0.0, global_step)
            if util_means:
                writer.add_scalar("sparse/mean_utility", float(np.mean(util_means)), global_step)
            if meta_u_means:
                writer.add_scalar("sparse/mean_meta_u", float(np.mean(meta_u_means)), global_step)
                writer.add_scalar("sparse/mature_frac", float(np.mean(mature_fracs)), global_step)
            writer.add_scalar("sparse/rewire_events_iter", episode_rewire_events, global_step)
            writer.add_scalar("sparse/rewire_edges_iter", episode_rewire_edges, global_step)
            thr = raw_agent().current_threshold()
            if thr is not None:
                writer.add_scalar("sparse/utility_threshold", float(thr.detach()), global_step)
        episode_rewire_events = 0
        episode_rewire_edges = 0

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar(
            "diagnostics/preclip_grad_norm",
            float(preclip_grad_norm_sum / optimizer_steps),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/clip_coefficient",
            float(clip_coefficient_sum / optimizer_steps),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/clipped_minibatch_fraction",
            float(clipped_steps / optimizer_steps),
            global_step,
        )
        for name, _ in named_sparse_layers:
            writer.add_scalar(
                f"diagnostics/stored_grad_norm_sample/{name}",
                float(sampled_layer_grad_norms[name]),
                global_step,
            )
            writer.add_scalar(
                f"diagnostics/effective_grad_norm_sample/{name}",
                float(sampled_layer_effective_grad_norms[name]),
                global_step,
            )
            writer.add_scalar(
                f"diagnostics/effective_update_mean_abs_sample/{name}",
                float(sampled_layer_updates[name]),
                global_step,
            )
        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        writer.add_scalar("charts/SPS", sps, global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(raw_agent().state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
