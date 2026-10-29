# TD7 PAIRED REPLAY COVERAGE v26 — joint critic and action-SALE history.
# Hypothesis: persistent state/direction coverage improves held-out contrast
# transfer while actor/alpha remain fresh main-only LAP exactly as in v25.
# Historical raw point transitions and atomic antithetic first-transition pairs
# are uniformly sampled, relabeled with one-step current frozen-behavior
# continuation and evolving target twin-min/encoder/range bounds. Fresh labels
# remain lambda(.95)/32; each channel mixes fresh/history equally when enabled.
# Lane0 and post-boundary branch rows never enter history. SALE also receives
# history; representation coverage, one-step versus lambda labels and expanded
# bounds are explicit confounds. No actor replay or trust-region claim.
# Permanent rings and behavior snapshots precede graph capture; no full-ring
# refresh. All physical interactions retain v25's counted budget and clocks.
import copy
import ctypes
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
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.graph_step import capture_steps
from cleanrl.shared.mujoco_env import NativeMujocoVectorEnv, make_mujoco_vector_env
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.timing import PhaseTimer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    compile: bool = True
    """compile the losses with Inductor before capturing the train-step graphs"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_steps: int = 1024
    """steps per environment per rollout"""
    env_threads: int = 2
    """maximum physics threads"""
    updates_per_step: float = 1.0
    """fresh-rollout minibatch updates per environment step"""
    warmup_rollouts: int = 1
    """recorded rollouts of latent-free exploration (mu=0, sigma=warmup_sigma) before training starts"""
    warmup_sigma: float = 0.6
    """std of the Gaussian warmup policy (the TD7 uniform warmup has no Gaussian latent; 0.6 ~ its std)"""
    trace_lambda: float = 0.95
    """mixing coefficient for frozen behavior-policy traces; 0 gives frozen one-step labels"""
    trace_horizon: int = 32
    """independent temporal trace segment length; interaction rollout remains 1024 steps"""
    hermite_order: int = 2
    """paired odd/even supervision requires both Gaussian Hermite orders"""
    eval_interval: int = 2
    """evaluate the policy mean every N rollouts"""
    eval_eps: int = 10
    """number of evaluation episodes (run as one batched rollout)"""
    batch_size: int = 256
    """minibatch size sampled from the latest rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_update_rate: int = 250
    """hard-update the target networks and fixed encoders every N training steps"""
    target_sigma: float = 0.1
    """per-dimension std defining the entropy lower bound; not a noise ceiling"""
    target_smoothing_sigma: float = 0.2
    """legacy target query std; unused by frozen behavior-policy labels"""
    alpha_lr: float = 1e-3
    """learning rate of the alpha (log_alpha) optimizer"""
    policy_freq: int = 2
    """the frequency of training the policy"""
    lap_alpha: float = 0.4
    """within-rollout LAP prioritization exponent"""
    min_priority: float = 1.0
    """minimum sampling priority; critic objective is squared error"""
    zs_dim: int = 256
    """dimensionality of the SALE embeddings"""
    hidden_dim: int = 256
    """hidden layer width of all networks"""
    encoder_lr: float = 3e-4
    """the learning rate of the encoder optimizer"""
    critic_lr: float = 3e-4
    """the learning rate of the critic optimizer"""
    max_grad_norm: float = 0.5
    """encoder/critic/actor clipping required by shared standards; added versus TD7 base"""
    actor_lr: float = 3e-4
    """the learning rate of the actor optimizer"""


# Log-std bounds of the state-dependent Gaussian: sigma in [~6.7e-3, 1] for actions in [-1, 1].
LOG_STD_MIN = -5
LOG_STD_MAX = 0


def lambda_returns(rewards, not_done, boundaries, next_values, gamma, trace_lambda, trace_horizon=32):
    """Time-major lane-local traces with independent segments and autoreset cuts.

    Terminations mask all bootstrap. Truncations bootstrap their final observation
    but stop recursion. Each segment's final real row uses its own next value;
    padding only batches a short final segment and cannot affect real targets.
    """
    assert rewards.shape == not_done.shape == boundaries.shape == next_values.shape
    assert rewards.shape[0] > 0 and trace_horizon > 0
    length = rewards.shape[0]
    segments = (length + trace_horizon - 1) // trace_horizon
    padding = segments * trace_horizon - length
    cuts = boundaries.clone()
    cuts[-1] = 1  # preserve the last real bootstrap when final segment is padded

    def segmented(x, fill=0):
        if padding:
            x = torch.cat([x, x.new_full((padding,) + x.shape[1:], fill)])
        return x.reshape(segments, trace_horizon, *x.shape[1:]).transpose(0, 1)

    r, mask, cut, values = segmented(rewards), segmented(not_done), segmented(cuts, 1), segmented(next_values)
    continuation = values[-1]
    targets = []
    for t in range(trace_horizon - 1, -1, -1):
        trace = torch.where(cut[t].bool(), values[t], continuation)
        continuation = r[t] + gamma * mask[t] * ((1 - trace_lambda) * values[t] + trace_lambda * trace)
        targets.append(continuation)
    targets = torch.stack(targets[::-1]).transpose(0, 1)
    return targets.reshape(segments * trace_horizon, *rewards.shape[1:])[:length]


def avg_l1_norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def avg_l1_norm_jvp(x, t, eps=1e-8):
    """avg_l1_norm and its forward-mode derivative along tangent t."""
    scale = x.abs().mean(-1, keepdim=True).clamp(min=eps)
    y = x / scale
    return y, (t - y * (x.sign() * t).mean(-1, keepdim=True)) / scale


def elu_jvp(x, t):
    return F.elu(x), torch.where(x > 0, t, t * x.clamp(max=0).exp())


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.sale = zs_dim > 0  # zs_dim=0: state-only actor, ``zs`` is ignored
        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)
        self.l3_logstd = nn.Linear(hdim, action_dim)

    def forward(self, state, zs):
        a = avg_l1_norm(self.l0(state))
        if self.sale:
            a = torch.cat([a, zs], 1)
        a = F.relu(self.l1(a))
        a = F.relu(self.l2(a))
        log_std = torch.tanh(self.l3_logstd(a))
        return torch.tanh(self.l3(a)), LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)


class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.zs_dim = zs_dim
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

    def zs(self, state):
        zs = F.elu(self.zs1(state))
        zs = F.elu(self.zs2(zs))
        return avg_l1_norm(self.zs3(zs))

    def zsa(self, zs, action):
        zsa = F.elu(self.zsa1(torch.cat([zs, action], 1)))
        zsa = F.elu(self.zsa2(zsa))
        return self.zsa3(zsa)

    def zsa_jvp(self, zs, action, t_action):
        """zsa(zs, action) and its derivative along an action tangent (zs held fixed)."""
        h, t = elu_jvp(self.zsa1(torch.cat([zs, action], 1)), t_action @ self.zsa1.weight[:, self.zs_dim :].T)
        h, t = elu_jvp(self.zsa2(h), t @ self.zsa2.weight.T)
        return self.zsa3(h), t @ self.zsa3.weight.T


class EnsembleLinear(nn.Module):
    """``members`` independent nn.Linear layers applied to (members, batch, in) as one baddbmm."""

    def __init__(self, members, in_features, out_features):
        super().__init__()
        bound = in_features**-0.5  # nn.Linear's default init bound, for weight and bias
        self.weight = nn.Parameter(torch.empty(members, in_features, out_features).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(members, 1, out_features).uniform_(-bound, bound))

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)

    def jvp(self, x, t):
        """Value and tangent in one bmm: the tangent rows get the weight but not the bias."""
        out = torch.bmm(torch.cat([x, t], 1), self.weight)
        return out[:, : x.shape[1]] + self.bias, out[:, x.shape[1] :]


class PreActionLatentTwinCritic(nn.Module):
    """Fused twin critic conditioned on (policy mean, policy std, pre-action latent)."""

    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256):
        super().__init__()
        self.hdim, self.state_dim = hdim, state_dim
        latent_dim = 3 * action_dim  # mean | std | pre-action latent
        self.q0 = nn.Linear(state_dim + latent_dim, 2 * hdim)  # both members side by side
        self.q1 = EnsembleLinear(2, 2 * zs_dim + hdim, hdim)
        self.q2 = EnsembleLinear(2, hdim, hdim)
        self.q3 = EnsembleLinear(2, hdim, 1)
        bound = (state_dim + latent_dim) ** -0.5  # per-member nn.Linear init
        nn.init.uniform_(self.q0.weight, -bound, bound)
        nn.init.uniform_(self.q0.bias, -bound, bound)

    def forward(self, state, latent, pre_action, zsa, zs, t_latent=None, t_pre_action=None, t_zsa=None):
        batch = state.shape[0]
        latent_input = torch.cat([latent, pre_action], 1)
        pre = self.q0(torch.cat([state, latent_input], 1)).view(batch, 2, self.hdim)
        embeddings = torch.cat([zsa, zs], 1).expand(2, batch, -1)
        if t_latent is None:
            h = avg_l1_norm(pre).transpose(0, 1)
            h = F.elu(self.q1(torch.cat([h, embeddings], 2)))
            h = F.elu(self.q2(h))
            return self.q3(h).squeeze(-1).transpose(0, 1)
        t_latent_input = torch.cat([t_latent, t_pre_action], 1)
        t_pre = (t_latent_input @ self.q0.weight[:, self.state_dim :].T).view(batch, 2, self.hdim)
        h, t = avg_l1_norm_jvp(pre, t_pre)
        t_embeddings = torch.cat([t_zsa, torch.zeros_like(zs)], 1).expand(2, batch, -1)
        h = torch.cat([h.transpose(0, 1), embeddings], 2)
        t = torch.cat([t.transpose(0, 1), t_embeddings], 2)
        h, t = elu_jvp(*self.q1.jvp(h, t))
        h, t = elu_jvp(*self.q2.jvp(h, t))
        q, t = self.q3.jvp(h, t)
        return q.squeeze(-1).transpose(0, 1), t.squeeze(-1).transpose(0, 1)


class HostPolicy:
    """NumPy mirror of the actor (and fixed_encoder.zs when it feeds the actor) for acting on the host.

    ``refresh`` pulls the device weights in one flat D2H copy; call it whenever the device policy
    changes (after every training burst).
    """

    def __init__(self, encoder: Encoder, actor: Actor):
        self.sale = actor.sale
        encoder_sources = [
            (encoder.zs1.weight,), (encoder.zs1.bias,), (encoder.zs2.weight,), (encoder.zs2.bias,),
            (encoder.zs3.weight,), (encoder.zs3.bias,),
        ]
        self._sources = (encoder_sources if self.sale else []) + [
            (actor.l0.weight,), (actor.l0.bias,),
            (actor.l1.weight,), (actor.l1.bias,), (actor.l2.weight,), (actor.l2.bias,),
            (actor.l3.weight, actor.l3_logstd.weight), (actor.l3.bias, actor.l3_logstd.bias),
        ]
        shapes = [(sum(t.shape[0] for t in group),) + tuple(group[0].shape[1:]) for group in self._sources]
        self._flat = np.zeros(sum(int(np.prod(s)) for s in shapes), dtype=np.float32)
        self._views, offset = [], 0
        for shape in shapes:
            size = int(np.prod(shape))
            view = self._flat[offset : offset + size].reshape(shape)
            self._views.append(view.T if view.ndim == 2 else view)  # weights as (in, out)
            offset += size
        self.action_dim = actor.l3.out_features

    def refresh(self):
        flat = torch.cat([t.detach().reshape(-1) for group in self._sources for t in group])
        self._flat[:] = flat.cpu().numpy()

    def __call__(self, state):
        w0, b0, wl1, bl1, wl2, bl2, wh, bh = self._views[-8:]
        a = state @ w0 + b0
        a /= np.maximum(np.abs(a).mean(-1, keepdims=True), 1e-8)
        if self.sale:
            w1, b1, w2, b2, w3, b3 = self._views[:6]
            zs = state @ w1 + b1
            zs = np.where(zs > 0, zs, np.expm1(np.minimum(zs, 0)))
            zs = zs @ w2 + b2
            zs = np.where(zs > 0, zs, np.expm1(np.minimum(zs, 0)))
            zs = zs @ w3 + b3
            zs /= np.maximum(np.abs(zs).mean(-1, keepdims=True), 1e-8)
            a = np.concatenate([a, zs], 1)
        a = np.maximum(a @ wl1 + bl1, 0)
        a = np.maximum(a @ wl2 + bl2, 0)
        out = a @ wh + bh
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (np.tanh(out[:, self.action_dim :]) + 1)
        return np.tanh(out[:, : self.action_dim]), log_std


def gaussian_kl(mu_old, sigma_old, mu, log_std):
    """Per-state KL(old || new) of diagonal Gaussians, split into its mean and scale terms."""
    mean_term = (0.5 * ((mu_old - mu) / log_std.exp()).square()).sum(-1)
    ratio = sigma_old.log() - log_std
    scale_term = (0.5 * (2 * ratio).exp() - ratio - 0.5).sum(-1)
    return mean_term, scale_term


def segment_anchor_steps(num_steps, trace_horizon, anchors_per_lane=8):
    """Fixed segment starts chosen independently of action noise and returns."""
    if trace_horizon < 1 or anchors_per_lane < 1:
        raise ValueError("horizon and anchor count must be positive")
    if num_steps % trace_horizon or num_steps // trace_horizon < anchors_per_lane:
        raise ValueError("paired runs need at least eight complete trace segments")
    return (np.arange(anchors_per_lane) * (num_steps // trace_horizon) // anchors_per_lane) * trace_horizon


def rollout_physics_budget(num_steps, num_envs, trace_horizon, anchors_per_lane=8):
    """All native branch lanes step, including masked rows after boundaries."""
    return num_steps * num_envs + anchors_per_lane * num_envs * trace_horizon


def paired_hermite_mse(pred_plus, pred_minus, target_plus, target_minus):
    """Average two branch squared losses; sum independent twin members."""
    return 0.5 * ((pred_plus - target_plus).square() + (pred_minus - target_minus).square()).sum(-1).mean()


def defined_stat_ratio(numerator, denominator):
    """Undefined zero-norm/variance statistics are NaN, not orthogonality."""
    valid = denominator > 0
    safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
    return torch.where(valid, numerator / safe_denominator, torch.full_like(numerator, float("nan")))


def branch_validity(boundary):
    """Valid before a lane's first boundary, including its boundary transition."""
    prior = torch.cat([torch.ones_like(boundary[:1]), 1 - boundary[:-1]], dim=0)
    return prior.cumprod(dim=0)


def masked_uniform_weight(valid_fit):
    """Fixed-slot uniform sampling estimates the mean over eligible observations."""
    count = valid_fit.sum()
    return valid_fit * valid_fit.numel() / count.clamp_min(1), count


def masked_target_extrema(target, fitting):
    """Empty fitting batches return neutral extrema for running-range updates."""
    fitting = fitting.reshape(-1, 1) > 0
    maximum = torch.where(fitting, target, -torch.inf).max()
    minimum = torch.where(fitting, target, torch.inf).min()
    return maximum, minimum


def observation_mixture(main, branch, main_count, branch_count):
    weight = branch_count / (main_count + branch_count)
    return (1 - weight) * main + weight * branch


def prediction_label_errors(old, new, initial_label, final_label):
    """Cross both prediction clocks with both label clocks."""
    return torch.stack([(prediction - label).square().mean()
                        for prediction in (old, new)
                        for label in (initial_label, final_label)])


def copy_episode_metadata(source, destination, source_index, destination_index, observation):
    """Copy shadow episode age/statistics/RNG without aliasing any live state.

    Native stepping reads _episode_lengths and overwrites TimeLimit from it.
    Historical return is copied for bookkeeping, never reported as a branch score.
    ``observation`` is authoritative rollout state: individual warmup resets can
    update physical state and rollout observations without refreshing vector caches.
    """
    for field in ("_episode_lengths", "_episode_returns", "_episode_starts"):
        getattr(destination, field)[destination_index] = getattr(source, field)[source_index]
    destination._limits[destination_index]._elapsed_steps = source._limits[source_index]._elapsed_steps
    src, dst = source._bases[source_index], destination._bases[destination_index]
    dst.np_random.bit_generator.state = copy.deepcopy(src.np_random.bit_generator.state)
    src_stats, dst_stats = source._statistics[source_index], destination._statistics[destination_index]
    for name in ("return_queue", "length_queue", "episode_count"):
        if hasattr(src_stats, name):
            setattr(dst_stats, name, copy.deepcopy(getattr(src_stats, name)))
    np.copyto(destination.observations[destination_index], observation)


class PairedBranches:
    """Full-state native shadow lanes; live state, RNG and statistics stay untouched."""

    def __init__(self, source, args):
        if not isinstance(source, NativeMujocoVectorEnv) or source._has_postprocessors:
            raise ValueError("paired snapshots require the raw native MuJoCo backend")
        self.source = source
        self.args = args
        self.starts = segment_anchor_steps(args.num_steps, args.trace_horizon)
        self.count = len(self.starts) * args.num_envs
        self.envs = make_mujoco_vector_env(args.env_id, self.count, backend="native", num_threads=args.env_threads)
        if not isinstance(self.envs, NativeMujocoVectorEnv) or self.envs._has_postprocessors:
            self.envs.close()
            raise ValueError("shadow environments must also be raw native MuJoCo")
        self.envs.reset(seed=args.seed + 10000)
        # dlsym resolves linked MuJoCo's public copyData API through the native
        # library. Full mjData includes time, controls and solver warm-start state.
        self._copy_data = self.envs._native.mj_copyData
        self._copy_data.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self._copy_data.restype = ctypes.c_void_p
        self.captured = np.zeros(self.count, dtype=bool)
        self.start_rows = (self.starts[:, None] * args.num_envs + np.arange(args.num_envs)).reshape(-1)
        self.lanes = np.tile(np.arange(args.num_envs), len(self.starts))
        self.start_times = np.repeat(self.starts, args.num_envs)

    def begin(self):
        self.captured.fill(False)

    def capture(self, step, obs):
        matches = np.flatnonzero(self.starts == step)
        if not len(matches):
            return
        block = int(matches[0])
        for lane in range(self.args.num_envs):
            index = block * self.args.num_envs + lane
            src, dst = self.source._bases[lane], self.envs._bases[index]
            if (src.model.nq, src.model.nv, src.model.na, src.model.nu) != (dst.model.nq, dst.model.nv, dst.model.na, dst.model.nu):
                raise ValueError("snapshot source/shadow model dimensions disagree")
            copied = self._copy_data(dst.data._address, dst.model._address, src.data._address)
            if not copied:
                raise RuntimeError("mj_copyData failed")
            copy_episode_metadata(self.source, self.envs, lane, index, obs[lane])
            self.captured[index] = True

    def collect(self, agent, policy, warmup, max_action):
        """Same future raw innovations, with each branch's own behavior inputs."""
        if not self.captured.all():
            raise RuntimeError("every scheduled source state must be captured before branching")
        args, f = self.args, agent._fields
        main = agent.staging_np
        obs = self.envs.observations.copy().astype(np.float32)
        expected = main[self.start_rows, f["state"]]
        if not np.array_equal(obs, expected):
            raise RuntimeError("first branch observations differ from source snapshots")
        active = np.ones(self.count, dtype=bool)
        source_active = np.ones(self.count, dtype=bool)
        coupling = []
        useful = 0
        for offset in range(args.trace_horizon):
            if offset == 0:
                mu = main[self.start_rows, f["mu"]].copy()
                sigma = main[self.start_rows, f["sigma"]].copy()
            elif warmup:
                mu = np.zeros((self.count, agent.action_dim), dtype=np.float32)
                sigma = np.full_like(mu, args.warmup_sigma)
            else:
                mu, log_std = policy(obs)
                sigma = np.exp(log_std)
            source_rows = (self.start_times + offset) * args.num_envs + self.lanes
            eps = main[source_rows, f["eps"]].copy()
            if offset == 0:
                eps *= -1
            raw = mu + sigma * eps
            action = np.clip(raw, -1, 1)
            common = active & source_active
            if common.any():
                state_rms = float(np.sqrt(np.square(obs[common] - main[source_rows, f["state"]][common]).mean()))
                action_rms = float(np.sqrt(np.square(action[common] - main[source_rows, f["action"]][common]).mean()))
            else:
                state_rms = action_rms = float("nan")
            coupling.append((int(common.sum()), state_rms, action_rms))
            useful += int(active.sum())
            next_obs, reward, term, trunc, infos = self.envs.step(action * max_action)
            next_obs = next_obs.astype(np.float32)
            boundaries = term | trunc
            transition_obs = transition_observations(next_obs, boundaries, infos)
            rows = agent.branch_staging_np[offset * self.count:(offset + 1) * self.count]
            values = dict(state=obs, action=action, mu=mu, sigma=sigma, eps=eps, pre_action=raw,
                          next_state=transition_obs, reward=np.where(active, reward, 0)[:, None],
                          not_done=np.where(active, ~term, False)[:, None],
                          boundary=np.where(active, boundaries, True)[:, None])
            for name, value in values.items():
                rows[:, f[name]] = value
            active &= ~boundaries
            source_active &= ~main[source_rows, f["boundary"]][:, 0].astype(bool)
            obs = next_obs
        return self.count * args.trace_horizon, useful, coupling

    def close(self):
        self.envs.close()


def replay_mixture(fresh, historical, enabled):
    """Recover fresh exactly when empty; otherwise equal coverage weights."""
    return (fresh + enabled * historical) / (1 + enabled)


def one_step_targets(reward, not_done, next_value, gamma):
    # Truncation keeps not_done=1 and uses the final observation; true terminals
    # have not_done=0. Boundary flags are needed for traces, not one-step labels.
    return reward + gamma * not_done * next_value


class RawHistoryRing:
    """Owned raw copies; pair rows are one atomic entry, never independent slots.

    Append runs outside optimizer capture. Sampling uses a live GPU count and
    fixed shapes, including a zero initialized sentinel while empty.
    """
    def __init__(self, capacity, entry_shape, device):
        self.capacity = capacity
        self.rows = torch.zeros((capacity, *entry_shape), device=device)
        self.stamps = torch.zeros(capacity, device=device)
        self.count = torch.zeros((), dtype=torch.long, device=device)
        self.position = 0
        self.size = 0

    @torch.no_grad()
    def append(self, rows, step):
        n = rows.shape[0]
        if not n:
            return
        # Keep the most recent suffix if a caller supplies more than capacity.
        rows = rows[-self.capacity:]
        n = rows.shape[0]
        indices = (torch.arange(n, device=self.rows.device) + self.position) % self.capacity
        self.rows.index_copy_(0, indices, rows)
        self.stamps.index_fill_(0, indices, step)
        self.position = (self.position + n) % self.capacity
        self.size = min(self.capacity, self.size + n)
        self.count.fill_(self.size)

    def sample(self, batch_size):
        index = (torch.rand(batch_size, device=self.rows.device) * self.count.clamp_min(1)).long()
        return self.rows.index_select(0, index), self.stamps.index_select(0, index)


class TD7Agent:
    def __init__(self, state_dim, action_dim, args: Args, device, writer: SummaryWriter):
        assert 0 <= args.trace_lambda <= 1 and args.trace_horizon > 0
        assert args.hermite_order == 2, "paired supervision requires odd and even terms"
        assert args.num_envs >= 2, "lane0 is held out from critic fitting"
        self.anchor_steps = segment_anchor_steps(args.num_steps, args.trace_horizon)
        self.args = args
        self.device = device
        self.writer = writer
        self.state_dim, self.action_dim = state_dim, action_dim
        adam = dict(fused=True)

        actor_zs_dim = args.zs_dim
        self.actor = Actor(state_dim, action_dim, actor_zs_dim, args.hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, actor_zs_dim, args.hidden_dim).to(device)
        self.critic = PreActionLatentTwinCritic(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.critic_target = PreActionLatentTwinCritic(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.fixed_encoder = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        self.fixed_encoder_target = Encoder(state_dim, action_dim, args.zs_dim, args.hidden_dim).to(device)
        # Deep copies preserve the original seed1 construction/RNG sequence.
        self.behavior_actor = copy.deepcopy(self.actor)
        self.behavior_encoder = copy.deepcopy(self.fixed_encoder)
        self.prefit_critic = copy.deepcopy(self.critic)
        targets = [
            (self.critic_target, self.critic),
            (self.fixed_encoder, self.encoder),
            (self.fixed_encoder_target, self.encoder),
        ]
        targets.extend([(self.actor_target, self.actor), (self.behavior_actor, self.actor),
                        (self.behavior_encoder, self.fixed_encoder), (self.prefit_critic, self.critic)])
        for target, source in targets:
            target.load_state_dict(source.state_dict())
            target.requires_grad_(False)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, **adam)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr, **adam)
        # (destination, source) pairs of every hard update, in TD7's order (target before fixed).
        self._hard_updates = [
            (list(self.critic_target.parameters()), list(self.critic.parameters())),
            (list(self.fixed_encoder_target.parameters()), list(self.fixed_encoder.parameters())),
            (list(self.fixed_encoder.parameters()), list(self.encoder.parameters())),
        ]
        optimizers = [self.encoder_optimizer, self.critic_optimizer]

        self.target_entropy = action_dim * float(np.log(args.target_sigma))
        self._hard_updates.insert(0, (list(self.actor_target.parameters()), list(self.actor.parameters())))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, **adam)
        self.log_alpha = torch.zeros((), device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr, **adam)
        optimizers += [self.actor_optimizer, self.alpha_optimizer]

        # Fresh-rollout rows retain both the clipped environment action and its pre-action latent.
        s, a = state_dim, action_dim
        self._fields = {}
        offset = 0
        for name, width in (("state", s), ("action", a), ("mu", a), ("sigma", a), ("eps", a), ("pre_action", a), ("next_state", s), ("reward", 1), ("not_done", 1), ("boundary", 1), ("target", 1)):
            self._fields[name] = slice(offset, offset + width)
            offset += width
        self.row_dim = offset
        self.rollout_size = args.num_envs * args.num_steps
        self.pair_count = len(self.anchor_steps) * args.num_envs
        self.branch_size = args.trace_horizon * self.pair_count
        self.pair_batch_size = args.batch_size
        anchors = torch.as_tensor(self.anchor_steps, device=device)
        lanes = torch.arange(args.num_envs, device=device)
        self.anchor_indices = (anchors[:, None] * args.num_envs + lanes).flatten()
        self.pair_fit_indices = torch.arange(self.pair_count, device=device).view(-1, args.num_envs)[:, 1:].flatten()
        self.fitting_mask = torch.arange(self.rollout_size, device=device) % args.num_envs != 0
        self.fitting_count = args.num_steps * (args.num_envs - 1)
        self.branch_valid = torch.ones(self.branch_size, device=device)
        initial_fit = (torch.arange(self.branch_size, device=device) % args.num_envs != 0).float()
        initial_weight, initial_count = masked_uniform_weight(initial_fit)
        self.branch_importance = initial_weight.clone()
        self.branch_fit_count = initial_count.clone()
        self.priority = torch.ones(self.rollout_size, device=device)
        self.last = torch.full((1,), self.rollout_size - 1, dtype=torch.long, device=device)
        self.updates_per_rollout = round(self.rollout_size * args.updates_per_step)
        self.staging = torch.zeros(self.rollout_size, self.row_dim, pin_memory=True)
        self.staging_np = self.staging.numpy()
        self.fresh = torch.zeros(self.rollout_size, self.row_dim, device=device)
        self.branch_staging = torch.zeros(self.branch_size, self.row_dim, pin_memory=True)
        self.branch_staging_np = self.branch_staging.numpy()
        self.branches = torch.zeros(self.branch_size, self.row_dim, device=device)
        self.continuation_mu = torch.zeros(self.rollout_size, action_dim, device=device)
        self.continuation_sigma = torch.zeros_like(self.continuation_mu)
        self.branch_continuation_mu = torch.zeros(self.branch_size, action_dim, device=device)
        self.branch_continuation_sigma = torch.zeros_like(self.branch_continuation_mu)
        self.initial_targets = torch.zeros(self.rollout_size, 1, device=device)
        self.initial_branch_targets = torch.zeros(self.branch_size, 1, device=device)
        self.reward_targets = torch.zeros_like(self.initial_targets)
        self.branch_reward_targets = torch.zeros_like(self.initial_branch_targets)
        self.point_history = RawHistoryRing(262144, (self.row_dim,), device)
        self.pair_history = RawHistoryRing(2048, (2, self.row_dim), device)
        self.behavior_warmup = torch.zeros((), dtype=torch.bool, device=device)
        self.history_step = torch.zeros((), device=device)
        self.refresh_count = 0
        self.prefit_pair_center = torch.zeros(self.pair_count, 2, device=device)
        self.prefit_pair_odd = torch.zeros_like(self.prefit_pair_center)
        self.prefit_pair_even = torch.zeros_like(self.prefit_pair_center)

        f32 = dict(device=device, dtype=torch.float32)
        self.max_priority = torch.ones((), **f32)
        self.running_max = torch.full((), -1e8, **f32)
        self.running_min = torch.full((), 1e8, **f32)
        self.max_target = torch.zeros((), **f32)
        self.min_target = torch.zeros((), **f32)
        self.metric_names = [
            "encoder_loss", "critic_loss", "q_values", "twin_disagreement", "target_std",
            "actor_loss", "sigma_mean", "sigma_min", "sigma_max", "preaction_point_gap", "importance_mean",
            "fresh_direct", "history_direct", "fresh_hermite", "history_hermite",
            "fresh_sale", "history_sale", "history_point_age", "history_pair_age",
        ]
        self.metrics = torch.zeros(len(self.metric_names), **f32)

        maybe_compile = (
            (lambda fn: torch.compile(fn, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}))
            if args.compile
            else (lambda fn: fn)
        )
        self._behavior_continuation = maybe_compile(self._behavior_continuation_fn)
        self._lambda_targets = maybe_compile(self._lambda_targets_fn)
        self._relabel_history = maybe_compile(self._relabel_history_fn)
        self._pair_predictions = maybe_compile(self._pair_predictions_fn)
        self._pair_loss = maybe_compile(self._pair_loss_fn)
        self._encoder_loss = maybe_compile(self._encoder_loss_fn)
        self._critic_loss = maybe_compile(self._critic_loss_fn)
        self._actor_loss = maybe_compile(self._actor_loss_fn)
        self._entropy = maybe_compile(self._entropy_fn)
        # TD7's delayed actor cadence, using only current-rollout minibatches.
        steps = [self._make_step(update_actor=False), self._make_step(update_actor=True)]
        self.graphs = capture_steps(
            steps,
            optimizers=optimizers,
            state_tensors=[self.priority, self.max_priority, self.running_max, self.running_min, self.metrics],
        )

        self.policy = HostPolicy(self.fixed_encoder, self.actor)
        self.policy.refresh()
        self.training_steps = 0
        self.current_global_step = 0

    # ---- fresh rollout -------------------------------------------------------------------------------
    def stage(self, step, **values):
        """Write one vector-env step: every value is (num_envs, width)."""
        rows = self.staging_np[step * self.args.num_envs : (step + 1) * self.args.num_envs]
        for name, value in values.items():
            rows[:, self._fields[name]] = value

    def flush(self, warmup=False, global_step=0):
        """Cache behavior coordinates; only critic/paired encoder labels refresh."""
        self.current_global_step = global_step
        self.history_step.fill_(global_step)
        self.behavior_warmup.fill_(warmup)
        with torch.no_grad():
            torch._foreach_copy_(list(self.behavior_actor.parameters()), list(self.actor.parameters()))
            torch._foreach_copy_(list(self.behavior_encoder.parameters()), list(self.fixed_encoder.parameters()))
            torch._foreach_copy_(list(self.prefit_critic.parameters()), list(self.critic.parameters()))
        self.fresh.copy_(self.staging)
        self.branches.copy_(self.branch_staging)
        self.priority.fill_(1.0)
        self.max_priority.fill_(1.0)
        with torch.no_grad():
            boundary = self.branches[:, self._fields["boundary"]].view(self.args.trace_horizon, self.pair_count)
            self.branch_valid.copy_(branch_validity(boundary).flatten())
            eligible = self.branch_valid * (torch.arange(self.branch_size, device=self.device) % self.args.num_envs != 0)
            weight, count = masked_uniform_weight(eligible)
            self.branch_importance.copy_(weight)
            self.branch_fit_count.copy_(count)
            # Compute both outputs together: actor is still the collection policy.
            mu, sigma = self._behavior_continuation(self.fresh, warmup)
            self.continuation_mu.copy_(mu)
            self.continuation_sigma.copy_(sigma)
            mu, sigma = self._behavior_continuation(self.branches, warmup)
            self.branch_continuation_mu.copy_(mu)
            self.branch_continuation_sigma.copy_(sigma)
            self.refresh_targets()
            self.initial_targets.copy_(self.fresh[:, self._fields["target"]])
            self.initial_branch_targets.copy_(self.branches[:, self._fields["target"]])
        self.refresh_count = 0
        self.pair_diagnostics(global_step, "prefit")

    def _behavior_continuation_fn(self, rollout, warmup):
        f = self._fields
        if warmup:
            mu = torch.zeros_like(rollout[:, f["mu"]])
            sigma = torch.full_like(mu, self.args.warmup_sigma)
        else:
            state = rollout[:, f["next_state"]]
            mu, log_std = self.actor(state, self.fixed_encoder.zs(state))
            sigma = log_std.exp()
        return mu, sigma

    @torch.no_grad()
    def append_history(self, global_step):
        """End-of-iteration ownership transfer, after all fresh fitting (warmup too)."""
        eligible = (self.branch_valid > 0) & (torch.arange(self.branch_size, device=self.device) % self.args.num_envs != 0)
        self.point_history.append(torch.cat([self.fresh[self.fitting_mask], self.branches[eligible]]), global_step)
        plus = self.fresh.index_select(0, self.anchor_indices.index_select(0, self.pair_fit_indices))
        minus = self.branches[:self.pair_count].index_select(0, self.pair_fit_indices)
        self.pair_history.append(torch.stack([plus, minus], dim=1), global_step)
        self.writer.add_scalar("history/point_occupancy", self.point_history.size, global_step)
        self.writer.add_scalar("history/pair_occupancy", self.pair_history.size, global_step)

    def _relabel_history_fn(self, rows):
        """Overwrite sampled copies only; never trust a stored historical label."""
        f = self._fields
        with torch.no_grad():
            state = rows[:, f["next_state"]]
            mu, log_std = self.behavior_actor(state, self.behavior_encoder.zs(state))
            mu = torch.where(self.behavior_warmup, torch.zeros_like(mu), mu)
            sigma = torch.where(self.behavior_warmup, torch.full_like(mu, self.args.warmup_sigma), log_std.exp())
            zs = self.fixed_encoder_target.zs(state)
            q = self.critic_target(state, torch.cat([mu, sigma], 1), mu,
                                   self.fixed_encoder_target.zsa(zs, mu), zs)
            value = q.min(1, keepdim=True)[0].clamp(self.min_target, self.max_target)
            rows[:, f["target"]] = one_step_targets(rows[:, f["reward"]], rows[:, f["not_done"]], value, self.args.gamma)
        return rows

    def _lambda_targets_fn(self, rollout, mu, sigma, time_steps, lanes):
        f, args = self._fields, self.args
        next_state = rollout[:, f["next_state"]]
        zs = self.fixed_encoder_target.zs(next_state)
        zsa = self.fixed_encoder_target.zsa(zs, mu)
        q = self.critic_target(next_state, torch.cat([mu, sigma], 1), mu, zsa, zs)
        value = q.min(1, keepdim=True)[0].clamp(self.min_target, self.max_target)
        shape = (time_steps, lanes, 1)
        reward, mask, boundary = [rollout[:, f[name]].view(shape) for name in ("reward", "not_done", "boundary")]
        target = lambda_returns(reward, mask, boundary, value.view(shape), args.gamma, args.trace_lambda, args.trace_horizon)
        reward_only = lambda_returns(reward, mask, boundary, torch.zeros_like(value).view(shape), args.gamma, args.trace_lambda, args.trace_horizon)
        return target.reshape(-1, 1), reward_only.reshape(-1, 1)

    @torch.no_grad()
    def refresh_targets(self):
        target, reward = self._lambda_targets(self.fresh, self.continuation_mu, self.continuation_sigma,
                                             self.args.num_steps, self.args.num_envs)
        self.fresh[:, self._fields["target"]].copy_(target)
        self.reward_targets.copy_(reward)
        target, reward = self._lambda_targets(self.branches, self.branch_continuation_mu, self.branch_continuation_sigma,
                                             self.args.trace_horizon, self.pair_count)
        self.branches[:, self._fields["target"]].copy_(target)
        self.branch_reward_targets.copy_(reward)

    # ---- losses -------------------------------------------------------------------------------
    def _encoder_loss_fn(self, batch, importance=None):
        f = self._fields
        with torch.no_grad():
            next_zs = self.encoder.zs(batch[:, f["next_state"]])
        pred_zs = self.encoder.zsa(self.encoder.zs(batch[:, f["state"]]), batch[:, f["action"]])
        per_row = (pred_zs - next_zs).square().mean(1)
        return per_row.mean() if importance is None else (importance * per_row).mean()

    def _direct_values(self, batch):
        f = self._fields
        state = batch[:, f["state"]]
        action, mu, sigma, raw = [batch[:, f[name]] for name in ("action", "mu", "sigma", "pre_action")]
        with torch.no_grad():
            zs = self.fixed_encoder.zs(state)
            zs2 = torch.cat([zs, zs])
            zsa = self.fixed_encoder.zsa(zs2, torch.cat([action, action]))
        q = self.critic(torch.cat([state, state]),
                        torch.cat([torch.cat([mu, sigma], 1), torch.cat([action, torch.zeros_like(sigma)], 1)]),
                        torch.cat([raw, action]), zsa, zs2)
        return q[:len(state)], q[len(state):], zs

    def _critic_loss_fn(self, batch, importance):
        pre, point, zs = self._direct_values(batch)
        target = batch[:, self._fields["target"]]
        residual_point = point - target
        loss = (importance[:, None] * residual_point.square()).sum(1).mean()
        with torch.no_grad():
            diagnostics = torch.stack([point.mean(), (point[:, 0] - point[:, 1]).abs().mean(), target.std(unbiased=False),
                                       target.max(), target.min(), (pre - point).mean(), importance.mean()])
        return loss, residual_point.abs().detach(), diagnostics, zs

    def _pair_predictions_fn(self, plus):
        return self._pair_predictions_with_networks(plus, self.fixed_encoder, self.critic)

    def _pair_predictions_with_networks(self, plus, encoder, critic):
        """One center, opposite raw mean tangents, equal scale tangents."""
        f = self._fields
        state, mu, sigma, eps = [plus[:, f[name]] for name in ("state", "mu", "sigma", "eps")]
        d = sigma * eps
        e = 0.5 * sigma * (eps.square() - 1)
        with torch.no_grad():
            zs = encoder.zs(state)
            zs2 = torch.cat([zs, zs])
            mu2 = torch.cat([mu, mu])
            zsa, t_zsa = encoder.zsa_jvp(zs2, mu2, torch.cat([d, -d]))
        latent = torch.cat([mu, sigma], 1)
        q, tangent = critic(torch.cat([state, state]), torch.cat([latent, latent]), mu2, zsa, zs2,
                                 torch.cat([torch.cat([d, e], 1), torch.cat([-d, e], 1)]),
                                 torch.cat([d, -d]), t_zsa)
        n = len(state)
        prediction_plus, prediction_minus = q[:n] + tangent[:n], q[n:] + tangent[n:]
        return prediction_plus, prediction_minus, q[:n]

    def _pair_loss_fn(self, plus, minus):
        pred_plus, pred_minus, center = self._pair_predictions_fn(plus)
        f = self._fields
        target_plus, target_minus = plus[:, f["target"]], minus[:, f["target"]]
        hermite = paired_hermite_mse(pred_plus, pred_minus, target_plus, target_minus)
        with torch.no_grad():
            extrema = torch.stack([torch.maximum(target_minus.max(), target_plus.max()), torch.minimum(target_minus.min(), target_plus.min())])
        return hermite, extrema

    def _actor_loss_fn(self, state, fixed_zs, alpha):
        mu, log_std = self.actor(state, fixed_zs)
        latent = torch.cat([mu, log_std.exp()], 1)
        q = self.critic(state, latent, mu, self.fixed_encoder.zsa(fixed_zs, mu), fixed_zs)
        return -q.mean() - alpha * log_std.sum(-1).mean(), log_std

    def _entropy_fn(self, state, fixed_zs):
        with torch.no_grad():
            return self.actor(state, fixed_zs)[1].sum(-1)

    # ---- train step ---------------------------------------------------------------------------
    def _make_step(self, update_actor):
        args, device = self.args, self.device

        def step():
            csum = torch.cumsum(self.priority, 0)
            ind = torch.searchsorted(csum, torch.rand(args.batch_size, device=device) * csum.gather(0, self.last))
            ind = torch.minimum(ind, self.last)  # the scan is only monotone up to rounding
            batch = self.fresh.index_select(0, ind)
            probability = (self.priority.index_select(0, ind) / csum.gather(0, self.last)).detach()
            fitting = self.fitting_mask.index_select(0, ind)
            importance = (fitting / (self.fitting_count * probability)).detach()
            pair_ind = self.pair_fit_indices.index_select(0, torch.randint(self.pair_fit_indices.numel(), (self.pair_batch_size,), device=device))
            plus = self.fresh.index_select(0, self.anchor_indices.index_select(0, pair_ind))
            minus = self.branches[:self.pair_count].index_select(0, pair_ind)
            branch_ind = torch.randint(self.branch_size, (args.batch_size,), device=device)
            branch_batch = self.branches.index_select(0, branch_ind)
            branch_importance = self.branch_importance.index_select(0, branch_ind)

            historical, point_stamp = self.point_history.sample(args.batch_size)
            historical = self._relabel_history(historical)
            historical_pair, pair_stamp = self.pair_history.sample(self.pair_batch_size)
            historical_pair = self._relabel_history(historical_pair.flatten(0, 1)).view(self.pair_batch_size, 2, self.row_dim)
            point_enabled = (self.point_history.count > 0).float()
            pair_enabled = (self.pair_history.count > 0).float()

            self.encoder_optimizer.zero_grad(set_to_none=False)
            fresh_sale = observation_mixture(
                self._encoder_loss(batch), self._encoder_loss(branch_batch, branch_importance),
                self.rollout_size, self.branch_fit_count)
            history_sale = self._encoder_loss(historical)
            encoder_loss = replay_mixture(fresh_sale, history_sale, point_enabled)
            encoder_loss.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), args.max_grad_norm)
            self.encoder_optimizer.step()

            self.critic_optimizer.zero_grad(set_to_none=False)
            main_direct, td, diagnostics, fixed_zs = self._critic_loss(batch, importance)
            pair_hermite, extrema = self._pair_loss(plus, minus)
            branch_direct, _, _, _ = self._critic_loss(branch_batch, branch_importance)
            direct = observation_mixture(main_direct, branch_direct, self.fitting_count, self.branch_fit_count)
            history_direct, _, history_diagnostics, _ = self._critic_loss(historical, torch.ones_like(importance))
            history_hermite, history_extrema = self._pair_loss(historical_pair[:, 0], historical_pair[:, 1])
            critic_loss = (replay_mixture(direct, history_direct, point_enabled) +
                           replay_mixture(pair_hermite, history_hermite, pair_enabled)) / 2
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), args.max_grad_norm)
            self.critic_optimizer.step()

            with torch.no_grad():
                branch_max, branch_min = masked_target_extrema(branch_batch[:, self._fields["target"]], branch_importance)
                torch.maximum(self.running_max, torch.maximum(torch.maximum(diagnostics[3], extrema[0]), branch_max), out=self.running_max)
                torch.minimum(self.running_min, torch.minimum(torch.minimum(diagnostics[4], extrema[1]), branch_min), out=self.running_min)
                history_max, history_min = masked_target_extrema(historical[:, self._fields["target"]], point_enabled.expand(args.batch_size))
                pair_max, pair_min = masked_target_extrema(historical_pair[:, :, self._fields["target"]].flatten(0, 1), pair_enabled.expand(2 * self.pair_batch_size))
                torch.maximum(self.running_max, torch.maximum(history_max, pair_max), out=self.running_max)
                torch.minimum(self.running_min, torch.minimum(history_min, pair_min), out=self.running_min)
                self.metrics[11:].copy_(torch.stack([
                    direct.detach(), history_direct.detach(), pair_hermite.detach(), history_hermite.detach(),
                    fresh_sale.detach(), history_sale.detach(),
                    point_enabled * (self.history_step - point_stamp).mean(),
                    pair_enabled * (self.history_step - pair_stamp).mean()]))
                priority = td.max(1)[0].clamp(min=args.min_priority).pow(args.lap_alpha)
                self.priority[ind] = priority
                torch.maximum(self.max_priority, priority.max(), out=self.max_priority)
                self.metrics[0].copy_(encoder_loss.detach())
                self.metrics[1].copy_(critic_loss.detach())
                self.metrics[2:5].copy_(diagnostics[:3])
                self.metrics[9:11].copy_(diagnostics[5:7])

            if update_actor:
                state = batch[:, self._fields["state"]]
                self.actor_optimizer.zero_grad(set_to_none=False)
                actor_loss, log_std = self._actor_loss(state, fixed_zs, self.log_alpha.detach().exp())
                actor_loss.backward(inputs=list(self.actor.parameters()))
                nn.utils.clip_grad_norm_(self.actor.parameters(), args.max_grad_norm)
                self.actor_optimizer.step()
                entropy = self._entropy(state, fixed_zs)
                self.alpha_optimizer.zero_grad(set_to_none=False)
                alpha_loss = -(self.log_alpha.exp() * (self.target_entropy - entropy)).mean()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                with torch.no_grad():
                    sigma = log_std.detach().exp()
                    self.metrics[5:9].copy_(torch.stack([actor_loss.detach(), sigma.mean(), sigma.min(), sigma.max()]))

        return step

    def hard_update(self):
        with torch.no_grad():
            for destination, source in self._hard_updates:
                torch._foreach_copy_(destination, source)
            self.max_priority.copy_(self.priority.max())
            self.max_target.copy_(self.running_max)
            self.min_target.copy_(self.running_min)

    def train(self):
        self.training_steps += 1
        update_actor = self.training_steps % self.args.policy_freq == 0
        self.graphs[int(update_actor)].replay()
        if self.training_steps % self.args.target_update_rate == 0:
            self.hard_update()
            self.refresh_targets()
            self.refresh_count += 1
        if self.training_steps % 500 == 0:
            values = dict(zip(self.metric_names, self.metrics.tolist()))
            for name in ("encoder_loss", "critic_loss", "q_values", "twin_disagreement", "target_std", "preaction_point_gap", "importance_mean"):
                self.writer.add_scalar(f"losses/{name}", values[name], self.current_global_step)
            for name in self.metric_names[11:]:
                self.writer.add_scalar(f"history/{name}", values[name], self.current_global_step)
            self.writer.add_scalar("charts/max_target", self.max_target.item(), self.current_global_step)
            self.writer.add_scalar("charts/min_target", self.min_target.item(), self.current_global_step)
            if self.training_steps % 1000 == 0:
                for name in ("sigma_mean", "sigma_min", "sigma_max"):
                    self.writer.add_scalar(f"charts/{name}", values[name], self.current_global_step)
                self.writer.add_scalar("losses/actor_loss", values["actor_loss"], self.current_global_step)
                self.writer.add_scalar("losses/alpha", self.log_alpha.exp().item(), self.current_global_step)

    @torch.no_grad()
    def pair_diagnostics(self, global_step, phase):
        """Held-out genuine contrasts; bootstrap labels remain approximations.

        Frozen-label errors do not change their labels during fitting. Pair
        covariance across anchors is descriptive, not a conditional noise test;
        eight held-out anchors also have temporal/episode dependence.
        """
        f = self._fields
        all_plus = self.fresh.index_select(0, self.anchor_indices)
        all_minus = self.branches[:self.pair_count]
        pred_plus, pred_minus, center = self._pair_predictions(all_plus)
        odd = 0.5 * (pred_plus - pred_minus)
        even = 0.5 * (pred_plus + pred_minus)
        direct_pre, direct_point, _ = self._direct_values(torch.cat([all_plus, all_minus]))
        pre_contrast = direct_pre[:self.pair_count] - direct_pre[self.pair_count:]
        point_contrast = direct_point[:self.pair_count] - direct_point[self.pair_count:]
        plus_reward = self.reward_targets.index_select(0, self.anchor_indices)
        minus_reward = self.branch_reward_targets[:self.pair_count]
        reward_contrast = plus_reward - minus_reward
        frozen_plus = self.initial_targets.index_select(0, self.anchor_indices)
        frozen_minus = self.initial_branch_targets[:self.pair_count]
        if phase == "prefit":
            self.prefit_pair_center.copy_(center)
            self.prefit_pair_odd.copy_(odd)
            self.prefit_pair_even.copy_(even)
        metrics = {}
        if phase == "postfit":
            # Old critic with the postfit fixed encoder: old-old predictions
            # already live in prefit buffers, and current-current are above.
            crossed_plus, crossed_minus, _ = self._pair_predictions_with_networks(all_plus, self.fixed_encoder, self.prefit_critic)
            crossed_odd = 0.5 * (crossed_plus - crossed_minus)
            crossed_even = 0.5 * (crossed_plus + crossed_minus)
        for split, mask in (("fit", torch.arange(self.pair_count, device=self.device) % self.args.num_envs != 0),
                            ("holdout", torch.arange(self.pair_count, device=self.device) % self.args.num_envs == 0)):
            if phase == "postfit":
                old_odd = self.prefit_pair_odd[mask].mean(1)
                new_odd = odd[mask].mean(1)
                metrics.update({
                    f"{split}/center_displacement_rms": (center[mask].mean(1) - self.prefit_pair_center[mask].mean(1)).square().mean().sqrt(),
                    f"{split}/odd_displacement_rms": (new_odd - old_odd).square().mean().sqrt(),
                    f"{split}/odd_prefit_postfit_cosine": defined_stat_ratio((old_odd * new_odd).sum(), old_odd.norm() * new_odd.norm()),
                })
            if phase == "postfit":
                latest_plus, latest_minus = all_plus[:, f["target"]], all_minus[:, f["target"]]
                for channel, old, new, initial, final in (
                    ("odd", self.prefit_pair_odd, odd, 0.5 * (frozen_plus - frozen_minus), 0.5 * (latest_plus - latest_minus)),
                    ("even", self.prefit_pair_even, even, 0.5 * (frozen_plus + frozen_minus), 0.5 * (latest_plus + latest_minus)),
                ):
                    errors = prediction_label_errors(old[mask], new[mask], initial[mask], final[mask])
                    for name, value in zip(("old_initial", "old_final", "new_initial", "new_final"), errors):
                        metrics[f"{split}/clock_matrix/{channel}_{name}_mse"] = value
                    crossed = crossed_odd if channel == "odd" else crossed_even
                    for clock, label in (("initial", initial), ("final", final)):
                        metrics[f"{split}/crossed_encoder/{channel}_oldcritic_newencoder_{clock}_mse"] = (crossed[mask] - label[mask]).square().mean()
                    metrics[f"{split}/crossed_encoder/{channel}_encoder_only_displacement_rms"] = (crossed[mask] - old[mask]).square().mean().sqrt()
                    metrics[f"{split}/crossed_encoder/{channel}_critic_after_encoder_displacement_rms"] = (new[mask] - crossed[mask]).square().mean().sqrt()
                    metrics[f"{split}/{channel}_label_movement_rms"] = (final[mask] - initial[mask]).square().mean().sqrt()
            for label_name, gp, gm in (("frozen", frozen_plus, frozen_minus),
                                       ("latest", all_plus[:, f["target"]], all_minus[:, f["target"]])):
                label_odd, label_even = 0.5 * (gp - gm), 0.5 * (gp + gm)
                predicted_odd = odd[mask].mean(1)
                actual_odd = label_odd[mask, 0]
                centered_pred = predicted_odd - predicted_odd.mean()
                centered_actual = actual_odd - actual_odd.mean()
                covariance = (centered_pred * centered_actual).mean()
                prefix = f"{split}/{label_name}"
                metrics.update({
                    f"{prefix}/odd_mse": (odd[mask] - label_odd[mask]).square().mean(),
                    f"{prefix}/odd_bias": (odd[mask] - label_odd[mask]).mean(),
                    f"{prefix}/odd_covariance": covariance,
                    f"{prefix}/odd_correlation": defined_stat_ratio(covariance, (centered_pred.square().mean() * centered_actual.square().mean()).sqrt()),
                    f"{prefix}/odd_rms_ratio": defined_stat_ratio(predicted_odd.square().mean().sqrt(), actual_odd.square().mean().sqrt()),
                    f"{prefix}/even_mse": (even[mask] - label_even[mask]).square().mean(),
                    f"{prefix}/even_bias": (even[mask] - label_even[mask]).mean(),
                    f"{prefix}/preaction_contrast_mse": (pre_contrast[mask] - (gp - gm)[mask]).square().mean(),
                    f"{prefix}/point_contrast_mse": (point_contrast[mask] - (gp - gm)[mask]).square().mean(),
                    f"{prefix}/label_contrast_std": (gp - gm)[mask].std(unbiased=False),
                    f"{prefix}/reward_contrast_std": reward_contrast[mask].std(unbiased=False),
                    f"{prefix}/bootstrap_contrast_std": ((gp - gm) - reward_contrast)[mask].std(unbiased=False),
                    f"{prefix}/bootstrap_contrast_mean": ((gp - gm) - reward_contrast)[mask].mean(),
                })
        values = torch.stack(list(metrics.values())).tolist()
        for name, value in zip(metrics, values):
            self.writer.add_scalar(f"paired/{phase}/{name}", value, global_step)
        self.writer.add_scalar("paired/anchor_count", self.pair_count, global_step)
        self.writer.add_scalar("paired/fit_anchor_count", self.pair_fit_indices.numel(), global_step)
        self.writer.add_scalar("paired/unique_initial_directions", self.pair_count, global_step)
        self.writer.add_scalar("paired/draws_per_rollout", self.updates_per_rollout * self.pair_batch_size, global_step)
        self.writer.add_scalar("paired/fit_reuse_per_rollout", self.updates_per_rollout * self.pair_batch_size / self.pair_fit_indices.numel(), global_step)
        self.writer.add_scalar("paired/direct_channel_branch_weight", (self.branch_fit_count / (self.fitting_count + self.branch_fit_count)).item(), global_step)
        self.writer.add_scalar("charts/branch_fit_observations", self.branch_fit_count.item(), global_step)
        self.branch_diagnostics(global_step, phase)

    @torch.no_grad()
    def branch_diagnostics(self, global_step, phase):
        """Heldout direct errors on valid continuation rows, same label clocks."""
        f = self._fields
        metrics = {}
        for offset in sorted(set((0, min(8, self.args.trace_horizon - 1),
                                  min(16, self.args.trace_horizon - 1), self.args.trace_horizon - 1))):
            start = offset * self.pair_count
            rows = self.branches[start:start + self.pair_count]
            mask = (self.branch_valid[start:start + self.pair_count] > 0) & (torch.arange(self.pair_count, device=self.device) % self.args.num_envs == 0)
            pre, point, _ = self._direct_values(rows)
            count = mask.sum()
            metrics[f"offset_{offset:02d}/valid_holdout_count"] = count
            for clock, target in (("initial", self.initial_branch_targets[start:start + self.pair_count]),
                                   ("latest", rows[:, f["target"]])):
                for channel, prediction in (("preaction", pre), ("point", point)):
                    error = (prediction - target).square().mean(1)
                    metrics[f"offset_{offset:02d}/{clock}/{channel}_mse"] = defined_stat_ratio((error * mask).sum(), count)
            label_shift = (rows[:, f["target"]] - self.initial_branch_targets[start:start + self.pair_count]).square()[:, 0]
            metrics[f"offset_{offset:02d}/label_movement_rms"] = defined_stat_ratio((label_shift * mask).sum(), count).sqrt()
        values = torch.stack(list(metrics.values())).tolist()
        for name, value in zip(metrics, values):
            self.writer.add_scalar(f"branch_direct/{phase}/{name}", value, global_step)

    # ---- one rollout of learning -------------------------------------------------------------
    @torch.no_grad()
    def actor_query_diagnostics(self, global_step):
        """Same held-out fresh states, current representation, two actor clocks."""
        state = self.fresh[::self.args.num_envs, self._fields["state"]]
        zs = self.fixed_encoder.zs(state)
        old_mu, old_log_std = self.behavior_actor(state, self.behavior_encoder.zs(state))
        old_mu = torch.where(self.behavior_warmup, torch.zeros_like(old_mu), old_mu)
        old_sigma = torch.where(self.behavior_warmup, torch.full_like(old_mu, self.args.warmup_sigma), old_log_std.exp())
        mu, log_std = self.actor(state, zs)
        q_old = self.critic(state, torch.cat([old_mu, old_sigma], 1), old_mu,
                            self.fixed_encoder.zsa(zs, old_mu), zs)
        q_new = self.critic(state, torch.cat([mu, log_std.exp()], 1), mu,
                            self.fixed_encoder.zsa(zs, mu), zs)
        # Analytic Gaussian tail probability estimates expected executed clipping.
        root2 = np.sqrt(2)
        saturation = lambda mean, sigma: (0.5 * (1 + torch.erf((-1 - mean) / (sigma * root2))) +
                                         0.5 * (1 - torch.erf((1 - mean) / (sigma * root2)))).mean()
        stats = torch.stack([(q_new - q_old).mean(), q_old.mean(), q_new.mean(),
                             saturation(old_mu, old_sigma), saturation(mu, log_std.exp())]).tolist()
        for name, value in zip(("predicted_center_gain", "behavior_center", "current_center",
                                "behavior_saturation", "current_saturation"), stats):
            self.writer.add_scalar(f"actor_queries/{name}", value, global_step)

    def learn(self, global_step):
        self.current_global_step = global_step
        # Compare against the actual Gaussian that generated this staged rollout.
        state = self.fresh[:, self._fields["state"]]
        mu_old = self.fresh[:, self._fields["mu"]]
        sigma_old = self.fresh[:, self._fields["sigma"]]
        for _ in range(self.updates_per_rollout):
            self.train()
        with torch.no_grad():
            mu, log_std = self.actor(state, self.fixed_encoder.zs(state))
            mean_term, scale_term = gaussian_kl(mu_old, sigma_old, mu, log_std)
            stats = torch.stack([(mean_term + scale_term).mean(), mean_term.mean(), scale_term.mean()]).tolist()
        for tag, stat in zip(("actor/kl", "actor/kl_mean_term", "actor/kl_scale_term"), stats):
            self.writer.add_scalar(tag, stat, global_step)
        self.writer.add_scalar("actor/kl_state_count", self.rollout_size, global_step)
        self.writer.add_scalar("charts/optimizer_steps", self.training_steps, global_step)
        self.writer.add_scalar("critic_targets/mean", self.fresh[:, self._fields["target"]].mean().item(), global_step)
        self.writer.add_scalar("critic_targets/std", self.fresh[:, self._fields["target"]].std().item(), global_step)
        self.actor_query_diagnostics(global_step)
        self.pair_diagnostics(global_step, "postfit")
        self.writer.add_scalar("critic_refresh/count", self.refresh_count, global_step)
        self.policy.refresh()


def transition_observations(next_obs, boundaries, infos):
    """next_obs with autoreset rows replaced by the finished episode's final observation."""
    if not boundaries.any():
        return next_obs
    transition_obs = next_obs.copy()
    for index in np.flatnonzero(boundaries):
        transition_obs[index] = infos["final_observation"][index]
    return transition_obs


def evaluate(policy: HostPolicy, eval_envs, max_action):
    """Deterministic policy returns (the latent mean), one episode per env, in lockstep."""
    states = np.stack([env.reset()[0] for env in eval_envs]).astype(np.float32)
    returns = np.zeros(len(eval_envs))
    active = np.ones(len(eval_envs), dtype=bool)
    while active.any():
        actions, _ = policy(states)
        for i in np.flatnonzero(active):
            state, reward, terminated, truncated, _ = eval_envs[i].step(actions[i] * max_action)
            states[i] = state
            returns[i] += reward
            active[i] = not (terminated or truncated)
    return returns.mean()


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    configure_runtime()
    assert torch.cuda.is_available(), "CUDA is required"
    device = torch.device("cuda")

    envs = make_mujoco_vector_env(args.env_id, args.num_envs, num_threads=args.env_threads)
    eval_envs = [gym.make(args.env_id) for _ in range(args.eval_eps)]
    for i, eval_env in enumerate(eval_envs):
        eval_env.reset(seed=args.seed + 100 + i)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    state_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    max_action = float(envs.single_action_space.high[0])
    shape = (args.num_envs, action_dim)

    agent = TD7Agent(state_dim, action_dim, args, device, writer)
    branches = PairedBranches(envs, args)
    timer = PhaseTimer()
    warmup_mu = np.zeros(shape, dtype=np.float32)
    warmup_sigma = np.full(shape, args.warmup_sigma, dtype=np.float32)

    start_time = time.time()
    # One unrecorded horizon of warmup-policy draws spreads the episode ages across the envs.
    horizon = episode_horizon(args.env_id)
    phase_offsets = compute_phase_offsets(args.num_envs, horizon, args.seed)
    writer.add_text("initial_phase_offsets", ",".join(map(str, phase_offsets)))
    warm = run_phase_warmup(
        envs, obs_norm=None, horizon=horizon, phase_offsets=phase_offsets, seed=args.seed,
        act_fn=lambda _: np.clip(args.warmup_sigma * rng.standard_normal(shape, dtype=np.float32), -1, 1) * max_action,
    )
    obs = warm.next_obs
    global_step = warm.transitions
    suppress = warm.suppress_mask.copy()
    rollout_cost = rollout_physics_budget(args.num_steps, args.num_envs, args.trace_horizon)
    num_iterations = (args.total_timesteps - warm.transitions) // rollout_cost
    ordinary_steps = warm.transitions
    branch_steps = 0
    branch_useful_steps = 0
    writer.add_text("paired_anchor_steps", ",".join(map(str, agent.anchor_steps)))
    for iteration in range(1, num_iterations + 1):
        training = iteration > args.warmup_rollouts
        branches.begin()
        for step in range(args.num_steps):
            with timer.span("rollout", use_cuda=False):
                if training:
                    mu, log_std = agent.policy(obs)
                    sigma = np.exp(log_std)
                else:
                    mu, sigma = warmup_mu, warmup_sigma
                eps = rng.standard_normal(shape, dtype=np.float32)
                pre_action = mu + sigma * eps
                action = np.clip(pre_action, -1, 1)
            with timer.span("env", use_cuda=False):
                branches.capture(step, obs)
                next_obs, reward, terminated, truncated, infos = envs.step(action * max_action)
            with timer.span("rollout", use_cuda=False):
                next_obs = next_obs.astype(np.float32)
                agent.stage(
                    step, state=obs, action=action, mu=mu, sigma=sigma, eps=eps, pre_action=pre_action,
                    next_state=transition_observations(next_obs, terminated | truncated, infos),
                    reward=reward[:, None], not_done=1.0 - terminated[:, None],
                    boundary=(terminated | truncated)[:, None],
                )
                obs = next_obs
            global_step += args.num_envs
            ordinary_steps += args.num_envs
            for index, info in enumerate(infos.get("final_info", ())):
                if info and "episode" in info:
                    if suppress[index]:
                        suppress[index] = False
                        continue
                    episode_return = float(info["episode"]["r"])
                    print(f"global_step={global_step}, episodic_return={episode_return:.3f}")
                    writer.add_scalar("charts/episodic_return", episode_return, global_step)
                    writer.add_scalar("charts/episodic_length", float(info["episode"]["l"]), global_step)

        with timer.span("branch", use_cuda=False):
            physics, useful, coupling = branches.collect(agent, agent.policy, not training, max_action)
        branch_steps += physics
        branch_useful_steps += useful
        global_step += physics
        if global_step > args.total_timesteps:
            raise RuntimeError("counted training physics exceeded the declared budget")
        for offset, (count, state_rms, action_rms) in enumerate(coupling):
            writer.add_scalar(f"branch_coupling/offset_{offset:02d}/alive_pairs", count, global_step)
            writer.add_scalar(f"branch_coupling/offset_{offset:02d}/state_rms", state_rms, global_step)
            writer.add_scalar(f"branch_coupling/offset_{offset:02d}/action_rms", action_rms, global_step)
        writer.add_scalar("charts/ordinary_transitions", ordinary_steps, global_step)
        writer.add_scalar("charts/branch_transitions", branch_steps, global_step)
        writer.add_scalar("charts/branch_useful_transitions", branch_useful_steps, global_step)
        with timer.span("update"):
            agent.flush(warmup=not training, global_step=global_step)
            if training:
                agent.learn(global_step)
            agent.append_history(global_step)
        if iteration % args.eval_interval == 0 or iteration == num_iterations:
            with timer.span("eval", use_cuda=False):
                eval_return = evaluate(agent.policy, eval_envs, max_action)
            writer.add_scalar("eval/episodic_return", eval_return, global_step)
            print(f"global_step={global_step}, eval_return={eval_return:.3f}")
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        for name, stats in timer.summary().items():
            writer.add_scalar(f"time/{name}_s", stats["total_s"], global_step)
        timer.reset()

    torch.save(dict(args=vars(args), actor=agent.actor.state_dict(), critic=agent.critic.state_dict(),
                    fixed_encoder=agent.fixed_encoder.state_dict(), global_step=global_step,
                    optimizer_steps=agent.training_steps, ordinary_transitions=ordinary_steps,
                    branch_transitions=branch_steps, branch_useful_transitions=branch_useful_steps,
                    declared_total_timesteps=args.total_timesteps), f"runs/{run_name}/checkpoint.pt")
    branches.close()
    envs.close()
    for eval_env in eval_envs:
        eval_env.close()
    writer.close()
