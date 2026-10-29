# Sepclip constraint transport v15: intrinsic actor-matrix ANVIL comparison.
# Project gradients before moments/polar; transport tangent rails between unit
# matrices; project adapted directions before retraction. Ambient mode preserves
# v12 updates. Hypothesis: nonlinear ambient normal forces distort policy motion.
import os
import random
import copy
import mujoco
import time
from contextlib import ExitStack
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import Beta
from torch.utils.tensorboard import SummaryWriter

from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import (
    TruncationBootstrapCache, device_minibatches, explained_variance,
    gather_metrics, get_gae_fn,
)
from cleanrl.shared.host_actor import SiTUGLUBranch, init_situglu_branch
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import make_beta_sampler, sample_beta_actions
from cleanrl.shared.staggered_envs import (
    compute_phase_offsets, episode_horizon, run_phase_warmup,
)
from cleanrl.shared.timing import PhaseTimer
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

# ANVIL2 maps and rail constants from modded-nanogpt record #360.
POLAR_EXPRESS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)
ANVIL_MAPS = (
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
)


def spectral_cascade(g, method, precision):
    """Small-matrix cascade; FP32 is an explicit numerical ablation on CUDA."""
    x = g.to(torch.bfloat16 if precision == "bf16" else torch.float32)
    tall = x.shape[-2] > x.shape[-1]
    if tall:
        x = x.mT
    if method == "anvil2":
        # Match upstream rounding boundaries: normalize from the FP32 trace
        # of the unnormalized BF16 Gram and reuse that Gram for the first map.
        x = x.contiguous()
        gram = x @ x.mT
        denominator = gram.diagonal(dim1=-2, dim2=-1).float().sum(-1)[..., None, None].sqrt() * 1.05 + 1e-6
        x = (x.float() / denominator).to(x.dtype)
        gram = (gram.float() / denominator.square()).to(x.dtype)
        for k, (a, b, c) in enumerate(ANVIL_MAPS):
            if k:
                gram = x @ x.mT
            # addbmm/addmm accumulates in FP32 and rounds the combined
            # polynomial once, like upstream's ba_plus_cAA Triton kernel.
            if x.ndim == 3:
                polynomial = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
                x = torch.baddbmm(x, polynomial, x, beta=a)
            else:
                polynomial = torch.addmm(gram, gram, gram, beta=b, alpha=c)
                x = torch.addmm(x, polynomial, x, beta=a)
    else:
        x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
        for a, b, c in POLAR_EXPRESS:
            gram = x @ x.mT
            x = a * x + (b * gram + c * (gram @ gram)) @ x
    return (x.mT if tall else x).float()


def matrix_update(p, velocity, second, grad, scalars, beta2, scale, method, precision, normuon_momentum, slow_clock):
    lr, momentum, fast_beta, fast_weight, slow_beta = scalars.unbind()
    if method == "anvil2":
        velocity[0].lerp_(grad, 1 - fast_beta)
        velocity[1].lerp_(grad, .02 if slow_clock == "epoch" else 1 - slow_beta)
        blend = fast_weight * velocity[0] + (1 - fast_weight) * velocity[1]
    else:
        velocity[0].lerp_(grad, 1 - normuon_momentum)
        blend = velocity[0]
        momentum = normuon_momentum
    # Out-of-place: .float() is NOT a copy when PPO gradients are already FP32.
    v = spectral_cascade(grad.lerp(blend, momentum), method, precision)
    red_dim = -1 if grad.shape[-2] >= grad.shape[-1] else -2
    second.lerp_(v.square().mean(red_dim, keepdim=True), 1 - beta2)
    adapted = v * second.clamp_min(1e-10).rsqrt()
    v = adapted * (v.norm(dim=(-2, -1), keepdim=True)
                   / adapted.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-10))
    if method == "anvil2" and precision == "bf16":
        # Upstream rounds the equalized direction to BF16 before applying it
        # to its FP32 master weights; preserve that numerical boundary.
        v = v.bfloat16().float()
    p.sub_(lr * scale * v)


def tangent_projection(point, vector, dim):
    """Orthogonal projection onto each unit-vector tangent, without zero guards.

    Division by the actual squared norm also removes FP32 normalization drift.
    A zero point violates the constraint and deliberately produces nonfinite data.
    """
    return vector - point * ((point * vector).sum(dim, keepdim=True)
                             / point.square().sum(dim, keepdim=True))


def sphere_parallel_transport(old, new, vector, dim):
    """Shortest-geodesic transport between unit, non-antipodal endpoints.

    No denominator floor: antipodal endpoints have no unique shortest geodesic.
    Unit tangent retraction has positive old/new inner product and excludes this
    singularity; the caller records violations rather than disguising them.
    """
    return vector - ((vector * new).sum(dim, keepdim=True)
                     / (1 + (old * new).sum(dim, keepdim=True))) * (old + new)


def intrinsic_matrix_update(p, velocity, second, grad, scalars, beta2, scale,
                            method, precision, normuon_momentum, slow_clock,
                            base_point, diagnostics, constraint_dim):
    """Graph-safe intrinsic update; diagnostics[0] is a sticky pathology flag.

    Remaining entries: minimum transport denominator, maximum unit-norm error,
    maximum final-rail radial magnitude, maximum pre-retraction direction radial
    magnitude. Shape and normalization axis are static compilation arguments.
    """
    axis = constraint_dim - 2  # Original matrix dim 0/1 -> stacked dim -2/-1.
    old = p.clone()
    initial_base_error = (base_point.square().sum(axis) - 1).abs().amax()
    # The trainer normalizes all matrices after Adam's independent auxiliary
    # update. Reconcile its possible FP32 renormalization with cached endpoints.
    initial_denominator = 1 + (base_point * old).sum(axis, keepdim=True)
    for rail in range(2):
        velocity[rail].copy_(sphere_parallel_transport(base_point, old, velocity[rail], axis))
    projected_grad = tangent_projection(old, grad, axis)
    lr, momentum, fast_beta, fast_weight, slow_beta = scalars.unbind()
    if method == "anvil2":
        velocity[0].lerp_(projected_grad, 1 - fast_beta)
        velocity[1].lerp_(projected_grad, .02 if slow_clock == "epoch" else 1 - slow_beta)
        blend = fast_weight * velocity[0] + (1 - fast_weight) * velocity[1]
    else:
        velocity[0].lerp_(projected_grad, 1 - normuon_momentum)
        blend = velocity[0]
        momentum = normuon_momentum
    v = spectral_cascade(projected_grad.lerp(blend, momentum), method, precision)
    red_dim = -1 if grad.shape[-2] >= grad.shape[-1] else -2
    second.lerp_(v.square().mean(red_dim, keepdim=True), 1 - beta2)
    adapted = v * second.clamp_min(1e-10).rsqrt()
    v = adapted * (v.norm(dim=(-2, -1), keepdim=True)
                   / adapted.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-10))
    if method == "anvil2" and precision == "bf16":
        v = v.bfloat16().float()
    # The cascade and lane adaptation need not preserve tangent directions.
    # Project AFTER their last BF16 rounding, without restoring the lost norm.
    direction = tangent_projection(old, v, axis)
    candidate = old - lr * scale * direction
    candidate_norm = torch.linalg.vector_norm(candidate, dim=axis, keepdim=True)
    endpoint = candidate / candidate_norm
    moved = (lr * scale * direction).square().sum(axis, keepdim=True) > 0
    endpoint = torch.where(moved, endpoint, old)
    denominator = 1 + (old * endpoint).sum(axis, keepdim=True)
    for rail in range(2):
        velocity[rail].copy_(sphere_parallel_transport(old, endpoint, velocity[rail], axis))
    p.copy_(endpoint)
    base_point.copy_(endpoint)
    unit_error = torch.stack(((old.square().sum(axis) - 1).abs().amax(),
                             (endpoint.square().sum(axis) - 1).abs().amax(),
                             initial_base_error,
                             (candidate_norm < 1 - 1e-4).any().float()))
    minimum_denominator = torch.minimum(initial_denominator.amin(), denominator.amin())
    invalid = (~torch.isfinite(p).all() | ~torch.isfinite(velocity).all()
               | ~torch.isfinite(second).all() | (minimum_denominator <= 0)
               | (unit_error.amax() > 1e-4))
    diagnostics[0].copy_(torch.maximum(diagnostics[0], invalid.float()))
    diagnostics[1].copy_(minimum_denominator)
    diagnostics[2].copy_(unit_error.amax())
    diagnostics[3].copy_(torch.stack([(rail * endpoint).sum(axis).abs().amax()
                                    for rail in velocity]).amax())
    diagnostics[4].copy_((direction * old).sum(axis).abs().amax())


class MatrixOptimizer(torch.optim.Optimizer):
    """Permanent stacked state, checkpoint-safe references and graph-safe scalars.

    FP32 parameters already supply master-weight precision. No language-model
    decay or storage-layout-specific c_proj multiplier is imposed on nGPT.
    """
    def __init__(self, params, lr=.005, method="anvil2", momentum=.95,
                 beta2=.9, precision="bf16", total_steps=4880, slow_clock="rollout", compile=True,
                 geometry="ambient", constraint_dims=None):
        if method not in {"normuon", "anvil2"} or precision not in {"bf16", "fp32"}:
            raise ValueError("invalid matrix optimizer or precision")
        if slow_clock not in {"rollout", "epoch"}:
            raise ValueError("slow_clock must be rollout or epoch")
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        params = list(params)
        if geometry not in {"ambient", "intrinsic"}:
            raise ValueError("geometry must be ambient or intrinsic")
        if constraint_dims is not None:
            constraint_dims = list(constraint_dims)
            if len(constraint_dims) != len(params) or any(dim not in (0, 1) for dim in constraint_dims):
                raise ValueError("constraint_dims must specify matrix axis 0 or 1 for each parameter")
        if geometry == "intrinsic" and constraint_dims is None:
            raise ValueError("intrinsic matrices require explicit normalization axes")
        super().__init__(params, dict(lr=lr, method=method, momentum=momentum,
                                     beta2=beta2, precision=precision,
                                     total_steps=total_steps, steps=0, slow_clock=slow_clock,
                                     rollout_pending=True, rollout_decay=.98**10, rollouts=0,
                                     geometry=geometry, constraint_dims=constraint_dims))
        self.update = torch.compile(matrix_update, fullgraph=True, dynamic=False) if compile else matrix_update
        self.intrinsic_update = (torch.compile(intrinsic_matrix_update, fullgraph=True, dynamic=False)
                                 if compile else intrinsic_matrix_update)
        self._rebuild_stacks()

    def _rebuild_stacks(self):
        self.stacks = []
        self.host_scalars = []
        for group in self.param_groups:
            group.setdefault("geometry", "ambient")
            group.setdefault("constraint_dims", None)
            if not group["params"]:
                raise ValueError("matrix optimizer groups must be nonempty")
            first = group["params"][0]
            group["lr"] = torch.as_tensor(group["lr"], device=first.device, dtype=torch.float32).clone()
            group["scalars"] = torch.zeros(5, device=first.device)
            self.host_scalars.append(torch.empty(4, dtype=torch.float32, pin_memory=True))
            by_shape = {}
            dims = group["constraint_dims"] or [None] * len(group["params"])
            if len(dims) != len(group["params"]) or (group["geometry"] == "intrinsic"
                    and any(dim not in (0, 1) for dim in dims)):
                raise ValueError("incompatible normalization-axis metadata")
            for p, dim in zip(group["params"], dims):
                if p.ndim != 2 or p.dtype != torch.float32 or p.device.type != "cuda":
                    raise ValueError("matrix optimizer requires FP32 CUDA matrices")
                by_shape.setdefault((p.shape, dim), []).append(p)
            for (shape, dim), ps in by_shape.items():
                state = self.state[ps[0]]
                lanes = (len(ps), shape[0], 1) if shape[0] >= shape[1] else (len(ps), 1, shape[1])
                if not state:
                    state["velocity"] = torch.zeros(2, len(ps), *shape, device=first.device)
                    state["second"] = torch.zeros(lanes, device=first.device)
                    state["state_version"] = 2
                    state["constraint_dim"] = dim
                    if group["geometry"] == "intrinsic":
                        state["base_point"] = torch.stack(ps).detach().clone()
                        state["diagnostics"] = torch.zeros(5, device=first.device)
                if state["velocity"].shape != (2, len(ps), *shape) or state["second"].shape != lanes:
                    raise ValueError("incompatible stacked optimizer checkpoint")
                if group["geometry"] == "intrinsic":
                    if (state.get("state_version") != 2 or state.get("constraint_dim") != dim
                            or "base_point" not in state or state["base_point"].shape != (len(ps), *shape)
                            or "diagnostics" not in state or state["diagnostics"].shape != (5,)):
                        raise ValueError("incompatible intrinsic optimizer checkpoint")
                self.stacks.append((group, ps, state, max(1., shape[0] / shape[1]) ** .5))

    def load_state_dict(self, state_dict):
        if len(state_dict["param_groups"]) != len(self.param_groups):
            raise ValueError("incompatible optimizer group count")
        for current, saved in zip(self.param_groups, state_dict["param_groups"]):
            if saved.get("geometry", "ambient") != current["geometry"]:
                raise ValueError("cannot reinterpret ambient momentum as intrinsic tangent momentum")
            if saved.get("constraint_dims") != current["constraint_dims"]:
                raise ValueError("normalization-axis metadata must match on checkpoint continuation")
        result = super().load_state_dict(state_dict)
        self._rebuild_stacks()
        return result

    @torch.no_grad()
    def set_lr(self, lr):
        for group in self.param_groups:
            group["lr"].fill_(lr)

    def begin_rollout(self, updates):
        if updates <= 0:
            raise ValueError("rollout must contain optimizer updates")
        for group in self.param_groups:
            group["rollout_pending"] = True
            group["rollout_decay"] = .98 ** updates
            group["rollouts"] += 1

    @staticmethod
    def rail_schedule(step, total):
        # Preserve reference phase fractions (1194 LM steps), not LM token counts.
        t = step * 1194 / total
        lookahead = .85 + (.93 - .85) * t / 240 if t < 240 else .93
        if t > 1144:
            lookahead = .93 - (.93 - .85) * min((t - 1144) / 50, 1.)
        return lookahead, (.85 if t >= 514 else lookahead), (.4385 if t >= 514 else 1.)

    @torch.no_grad()
    def step(self):
        for group, staging in zip(self.param_groups, self.host_scalars):
            if group["method"] == "anvil2":
                momentum, fast_beta, weight = self.rail_schedule(group["steps"], group["total_steps"])
            else:
                momentum, fast_beta, weight = group["momentum"], group["momentum"], 1.
            group["scalars"][0].copy_(group["lr"])
            slow_beta = (.98 if group["slow_clock"] == "epoch" else
                         group["rollout_decay"] if group["rollout_pending"] else 1.)
            group["rollout_pending"] = False
            staging[0], staging[1], staging[2], staging[3] = momentum, fast_beta, weight, slow_beta
            # Blocking upload prevents rewriting pinned storage before DMA completes.
            group["scalars"][1:].copy_(staging)
            group["steps"] += 1
        for group, ps, state, scale in self.stacks:
            if any(p.grad is None for p in ps):
                raise RuntimeError("all selected matrices must receive gradients")
            stacked, grads = torch.stack(ps), torch.stack([p.grad for p in ps])
            arguments = (stacked, state["velocity"], state["second"], grads,
                         group["scalars"], group["beta2"], scale, group["method"],
                         group["precision"], group["momentum"], group["slow_clock"])
            if group["geometry"] == "intrinsic":
                self.intrinsic_update(*arguments, state["base_point"], state["diagnostics"], state["constraint_dim"])
            else:
                self.update(*arguments)
            torch._foreach_copy_(ps, list(stacked.unbind(0)))


def make_optimizers(agent, args):
    if args.actor_optimizer == "adam":
        return None, optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
    matrices = [layer.weight for name, trunk in (("actor", agent.actor), ("critic", agent.critic))
                if args.normuon_network in (name, "both")
                for stage in (trunk.first, trunk.second)
                for layer in (stage[0].gate, stage[0].up, stage[0].down)]
    dims = [dim for name, trunk in (("actor", agent.actor), ("critic", agent.critic))
            if args.normuon_network in (name, "both")
            for stage in (trunk.first, trunk.second) for dim in (1, 1, 0)]
    matrix_ids = {id(p) for p in matrices}
    auxiliary = [p for p in agent.parameters() if id(p) not in matrix_ids]
    return (MatrixOptimizer(matrices, lr=args.normuon_lr, method=args.matrix_method,
                            momentum=args.normuon_momentum, beta2=args.normuon_beta2,
                            precision=args.matrix_precision,
                            total_steps=max(1, args.num_iterations * args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)),
                            slow_clock=args.slow_clock, compile=args.compile,
                            geometry=args.matrix_geometry,
                            constraint_dims=dims if args.matrix_geometry == "intrinsic" else None),
            optim.Adam(auxiliary, lr=args.learning_rate, eps=1e-5, fused=True))


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
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 8000000
    """total timesteps of the experiments"""
    learning_rate: float = 0.0024
    """the learning rate of the optimizer"""
    actor_optimizer: str = "anvil2"
    """anvil2 control or exact all-parameter Adam control"""
    normuon_network: str = "actor"
    """which hidden matrices use NorMuon: actor, critic, or both"""
    matrix_geometry: str = "intrinsic"
    """intrinsic tangent ANVIL or exact v12 ambient actor-matrix control"""
    normuon_lr: float = 0.0025
    """hidden-matrix LR; polar steps use sqrt(max(1, out/in)) scaling"""
    normuon_momentum: float = 0.95
    normuon_beta2: float = 0.9
    matrix_method: str = "anvil2"
    """anvil2 twin-rail cascade or normuon control"""
    slow_clock: str = "rollout"
    """slow rail data clock: rollout or original epoch control"""
    matrix_precision: str = "bf16"
    """cascade arithmetic; fp32 is an explicit numerical ablation"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 1024
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 1
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """lower policy ratio clipping coefficient, also the value clipping radius"""
    clip_coef_upper: float = 0.2
    """upper policy ratio clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # Execution controls, independent of PPO's batch and optimizer settings.
    env_backend: str = "auto"
    """native for supported v4 MuJoCo; sync for other continuous environments"""
    env_threads: int = 2
    """maximum physics threads per run"""
    compile: bool = True
    """compile deterministic policy statistics, PPO loss and GAE"""
    compile_mode: str = "reduce-overhead"
    """PyTorch compilation mode for fixed-shape paths"""
    non_blocking_transfers: bool = False
    """opt into event-protected asynchronous pinned transfers"""
    staggered_starts: bool = True
    """stagger parallel environments; warmup counts toward total_timesteps"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


# Public evaluation and historical GAE compatibility helpers; not the training path.
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


def bootstrap_observations(next_obs, truncations, infos):
    """Replace autoreset observations with final observations at time limits."""
    bootstrap_obs = np.array(next_obs, copy=True)
    truncations = np.asarray(truncations, dtype=bool)
    if not np.any(truncations):
        return bootstrap_obs

    final_observations = infos.get("final_observation")
    final_mask = infos.get("_final_observation")
    if final_observations is None:
        raise RuntimeError("truncated transition missing infos['final_observation']")

    for env_idx in np.flatnonzero(truncations):
        if final_mask is not None and not final_mask[env_idx]:
            raise RuntimeError(f"truncated environment {env_idx} has no final observation")
        final_observation = final_observations[env_idx]
        if final_observation is None:
            raise RuntimeError(f"truncated environment {env_idx} has no final observation")
        bootstrap_obs[env_idx] = final_observation
    return bootstrap_obs


def compute_gae(
    rewards,
    values,
    terminations,
    truncations,
    truncation_bootstrap_values,
    rollout_tail_value,
    gamma,
    gae_lambda,
):
    """Compute GAE with distinct bootstrap and reset-boundary semantics."""
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(rollout_tail_value)
    for t in reversed(range(rewards.shape[0])):
        ordinary_next_value = rollout_tail_value if t == rewards.shape[0] - 1 else values[t + 1]
        next_value = torch.where(
            truncations[t].bool(),
            truncation_bootstrap_values[t],
            ordinary_next_value,
        )
        bootstrap_nonterminal = 1.0 - terminations[t]
        trace_nonterminal = 1.0 - torch.maximum(terminations[t], truncations[t])
        delta = rewards[t] + gamma * bootstrap_nonterminal * next_value - values[t]
        last_advantage = delta + gamma * gae_lambda * trace_nonterminal * last_advantage
        advantages[t] = last_advantage
    return advantages, advantages + values


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualMLP(nn.Module):
    """Unit residual directions with variance-restored branches and readouts."""

    def __init__(self, observation_dim, output_dim, output_std):
        super().__init__()

        def stage(in_dim):
            return nn.Sequential(init_situglu_branch(SiTUGLUBranch(in_dim, 64)))

        self.first = stage(observation_dim)
        self.second = stage(64)
        self.head = nn.Sequential(layer_init(nn.Linear(64, output_dim), std=output_std))
        self.readout_gain = nn.Parameter(torch.ones(output_dim))

    def forward(self, x):
        h = F.normalize(self.first(x), p=2, dim=-1)
        branch = F.normalize(self.second(8.0 * h), p=2, dim=-1)
        return self.readout_gain * self.head(8.0 * F.normalize(h + branch, p=2, dim=-1))


class ResidualHostMirror:
    """Compose fused FP32 stage mirrors without changing shared implementations.

    Stage outputs are permanent buffers; copy the first before calling the
    second, and add into our own permanent buffer before evaluating the head.
    """

    def __init__(self, actor, num_envs):
        self.first = make_host_mirror(actor.first, num_envs)
        self.second = make_host_mirror(actor.second, num_envs)
        self.head = make_host_mirror(actor.head, num_envs)
        self.hidden = np.empty((num_envs, 64), dtype=np.float32)
        self.branch = np.empty_like(self.hidden)
        self.squared = np.empty_like(self.hidden)
        self.norm = np.empty((num_envs, 1), dtype=np.float32)
        self.actor = actor
        self.readout_gain = np.empty(actor.readout_gain.numel(), dtype=np.float32)
        self.refresh()

    def refresh(self):
        self.first.refresh()
        self.second.refresh()
        self.head.refresh()
        np.copyto(self.readout_gain, self.actor.readout_gain.detach().cpu().numpy())

    def _normalize(self, values):
        np.multiply(values, values, out=self.squared)
        np.sum(self.squared, axis=1, keepdims=True, out=self.norm)
        np.sqrt(self.norm, out=self.norm)
        # Match F.normalize's zero-vector definition, including its epsilon.
        np.maximum(self.norm, np.float32(1e-12), out=self.norm)
        np.divide(values, self.norm, out=values)

    def __call__(self, observations):
        np.copyto(self.hidden, self.first(observations))
        self._normalize(self.hidden)
        np.multiply(self.hidden, np.float32(8.0), out=self.branch)
        np.copyto(self.branch, self.second(self.branch))
        self._normalize(self.branch)
        np.add(self.hidden, self.branch, out=self.hidden)
        self._normalize(self.hidden)
        np.multiply(self.hidden, np.float32(8.0), out=self.hidden)
        output = self.head(self.hidden)
        np.multiply(output, self.readout_gain, out=output)
        return output


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        space = envs.single_action_space
        if not isinstance(space, gym.spaces.Box):
            raise TypeError("Beta PPO requires a Box action space")
        low, high = np.asarray(space.low), np.asarray(space.high)
        if not (np.isfinite(low).all() and np.isfinite(high).all() and np.all(high > low)):
            raise ValueError("Beta PPO requires finite, strictly ordered action bounds")
        self.action_shape = tuple(space.shape)
        self.action_dim = int(np.prod(space.shape))
        observation_dim = int(np.prod(envs.single_observation_space.shape))
        self.register_buffer("action_low", torch.as_tensor(low.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(high.reshape(-1).copy(), dtype=torch.float32))
        self.register_buffer("action_scale", self.action_high - self.action_low)
        if not torch.isfinite(self.action_scale).all() or not (self.action_scale > 0).all():
            raise ValueError("action bounds must have a finite positive FP32 range")
        self.register_buffer("log_action_scale", self.action_scale.log())
        self.critic = ResidualMLP(observation_dim, 1, output_std=1.0)
        self.actor = ResidualMLP(observation_dim, 2 * self.action_dim, output_std=0.01)

    @torch.no_grad()
    def normalize_matrices(self):
        """Match nGPT's matrix axes without changing optimizer moments or biases."""
        for trunk in (self.actor, self.critic):
            for stage in (trunk.first, trunk.second):
                branch = stage[0]
                for weight, dim in ((branch.gate.weight, 1),
                                    (branch.up.weight, 1),
                                    (branch.down.weight, 0)):
                    weight.div_(torch.linalg.vector_norm(weight, dim=dim, keepdim=True))
            weight = trunk.head[0].weight
            weight.div_(torch.linalg.vector_norm(weight, dim=1, keepdim=True))

    def get_value(self, x):
        return self.critic(x)

    def get_policy_and_value(self, x):
        alpha, beta = (F.softplus(self.actor(x)) + 1.0).chunk(2, dim=-1)
        return alpha, beta, self.critic(x)

    def action_logprob(self, alpha, beta, native_action):
        distribution = Beta(alpha, beta, validate_args=False)
        return (distribution.log_prob(native_action) - self.log_action_scale).sum(-1)

    def get_action_and_value(self, x, action=None):
        """Public API uses physical actions; training retains native samples."""
        alpha, beta, value = self.get_policy_and_value(x)
        if action is None:
            native, physical = sample_beta_actions(alpha, beta, self.action_low, self.action_high)
            action = physical.reshape((x.shape[0],) + self.action_shape)
        else:
            native = ((action.reshape(x.shape[0], -1) - self.action_low) / self.action_scale).clamp(
                SAMPLE_EPS, 1.0 - SAMPLE_EPS
            )
        distribution = Beta(alpha, beta, validate_args=False)
        logprob = (distribution.log_prob(native) - self.log_action_scale).sum(-1)
        entropy = (distribution.entropy() + self.log_action_scale).sum(-1)
        return action, logprob, entropy, value


def ppo_loss(agent, observations, native_actions, old_logprobs, advantages, returns, old_values, args):
    """Pure clipped PPO loss on native Beta samples; no inverse action scaling."""
    alpha, beta, newvalue = agent.get_policy_and_value(observations)
    distribution = Beta(alpha, beta, validate_args=False)
    newlogprob = (distribution.log_prob(native_actions) - agent.log_action_scale).sum(-1)
    entropy = (distribution.entropy() + agent.log_action_scale).sum(-1)
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio < 1 - args.clip_coef) | (ratio > 1 + args.clip_coef_upper)).float().mean()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef_upper)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = old_values + torch.clamp(newvalue - old_values, -args.clip_coef, args.clip_coef)
        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - returns) ** 2).mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()
    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    metrics = torch.stack((pg_loss.detach(), v_loss.detach(), entropy_loss.detach(),
                           old_approx_kl, approx_kl, clipfrac))
    return loss, metrics


def exact_policy_kl(agent, observations, old_alpha, old_beta):
    alpha, beta, _ = agent.get_policy_and_value(observations)
    old_sum, new_sum = old_alpha + old_beta, alpha + beta
    # Explicit Beta KL avoids distribution registry dispatch inside fullgraph.
    return (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(new_sum)
            - torch.lgamma(old_alpha) - torch.lgamma(old_beta) + torch.lgamma(old_sum)
            + (old_alpha - alpha) * torch.digamma(old_alpha)
            + (old_beta - beta) * torch.digamma(old_beta)
            + (new_sum - old_sum) * torch.digamma(old_sum)).sum(-1).mean()


def validate_args(args):
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) <= 0:
        raise ValueError("environment, rollout, minibatch and epoch counts must be positive")
    if args.env_backend not in {"auto", "native", "threaded", "sync"} or args.env_threads <= 0:
        raise ValueError("invalid environment backend or thread count")
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size == 0:
        raise ValueError("num_minibatches cannot exceed batch_size")
    if args.norm_adv and (args.minibatch_size < 2 or args.batch_size % args.minibatch_size == 1):
        raise ValueError("advantage normalization requires at least two samples per minibatch")
    if args.normuon_network not in {"actor", "critic", "both"}:
        raise ValueError("normuon_network must be actor, critic, or both")
    if not (np.isfinite(args.learning_rate) and args.learning_rate > 0
            and np.isfinite(args.normuon_lr) and args.normuon_lr > 0):
        raise ValueError("optimizer learning rates must be finite and positive")
    if not (0 <= args.normuon_momentum < 1 and 0 <= args.normuon_beta2 < 1):
        raise ValueError("NorMuon momentum and beta2 must lie in [0, 1)")
    if args.matrix_method not in {"normuon", "anvil2"} or args.matrix_precision not in {"bf16", "fp32"}:
        raise ValueError("invalid matrix method or precision")
    if args.slow_clock == "rollout" and (args.num_minibatches != 1 or args.target_kl is not None):
        raise ValueError("rollout clock requires one full minibatch and a fixed epoch count")
    if args.slow_clock not in {"rollout", "epoch"}:
        raise ValueError("invalid slow clock")
    if args.actor_optimizer not in {"adam", "anvil2"}:
        raise ValueError("actor_optimizer must be adam or anvil2")
    if args.matrix_geometry not in {"ambient", "intrinsic"}:
        raise ValueError("matrix_geometry must be ambient or intrinsic")
    if args.matrix_geometry == "intrinsic" and args.normuon_network != "actor":
        raise ValueError("v15 intrinsic comparison keeps critic updates on Adam")
    if not args.cuda:
        raise ValueError("the shared PPO trainer requires CUDA")
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


# MuJoCo 2.3.3 predates mj_getState/mj_setState. These are the complete
# integration inputs in that version's mjdata.h, including externally supplied
# controls/forces and solver warmstarts. Derived kinematics are reconstructed.
INTEGRATION_FIELDS = ('qpos', 'qvel', 'act', 'qacc_warmstart', 'ctrl',
                      'qfrc_applied', 'xfrc_applied', 'mocap_pos', 'mocap_quat',
                      'userdata', 'plugin_state', 'eq_active')


def integration_backend():
    return 'mjSTATE_INTEGRATION' if hasattr(mujoco, 'mj_getState') else 'mjdata_complete_integration_fields'


def integration_state(base):
    if hasattr(mujoco, 'mj_getState'):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(base.model, spec), dtype=np.float64)
        mujoco.mj_getState(base.model, base.data, state, spec)
        return state
    return np.concatenate((np.array([base.data.time]),
                           *(np.asarray(getattr(base.data, name)).reshape(-1).copy()
                             for name in INTEGRATION_FIELDS if hasattr(base.data, name))))


def set_integration_state(base, state):
    if hasattr(mujoco, 'mj_setState'):
        mujoco.mj_setState(base.model, base.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        return
    base.data.time = float(state[0])
    offset = 1
    for name in INTEGRATION_FIELDS:
        if hasattr(base.data, name):
            field = np.asarray(getattr(base.data, name))
            size = field.size
            field[...] = state[offset:offset+size].reshape(field.shape)
            offset += size
    if offset != len(state):
        raise ValueError(f'Integration state size mismatch: expected {offset}, got {len(state)}')


class TransitionCapture:
    """Read-only checkpoint buffer; no draws or normalization-state mutations."""

    def __init__(self, envs, num_steps):
        bases = tuple(env.unwrapped for env in envs.envs)
        sizes = [integration_state(base).size for base in bases]
        if len(set(sizes)) != 1:
            raise ValueError("physics capture requires homogeneous integration states")
        n, obs = envs.num_envs, envs.single_observation_space.shape
        self.arrays = {
            "obs_means": np.empty((num_steps, n, *obs), dtype=np.float64),
            "obs_variances": np.empty((num_steps, n, *obs), dtype=np.float64),
            "episode_age": np.empty((num_steps, n), dtype=np.int32),
            "raw_obs": np.empty((num_steps, n, *obs), dtype=np.float64),
            "integration_states": np.empty((num_steps, n, sizes[0]), dtype=np.float64),
            "physical_actions": np.empty((num_steps, n, *envs.single_action_space.shape), dtype=np.float32),
            "raw_next_obs": np.empty((num_steps, n, *obs), dtype=np.float64),
            "normalized_next_obs": np.empty((num_steps, n, *obs), dtype=np.float32),
            "normalized_transition_obs": np.empty((num_steps, n, *obs), dtype=np.float32),
            "raw_rewards": np.empty((num_steps, n), dtype=np.float64),
            "reward_std": np.empty((num_steps, n), dtype=np.float64),
        }

    def before(self, step, envs, actions, obs_norm):
        np.copyto(self.arrays["obs_means"][step], obs_norm.means)
        np.copyto(self.arrays["obs_variances"][step], obs_norm.variances)
        if hasattr(envs, "_episode_lengths"):
            np.copyto(self.arrays["episode_age"][step], envs._episode_lengths)
        else:
            raise ValueError("transition microscopy requires native episode-age counters")
        for index, env in enumerate(envs.envs):
            base = env.unwrapped
            np.copyto(self.arrays["raw_obs"][step, index], base._get_obs())
            np.copyto(self.arrays["integration_states"][step, index], integration_state(base))
        np.copyto(self.arrays["physical_actions"][step], actions)

    def after(self, step, raw_obs, raw_rewards, next_obs, transition_obs, rew_norm):
        for key, value in (("raw_next_obs", raw_obs), ("raw_rewards", raw_rewards),
                           ("normalized_next_obs", next_obs), ("normalized_transition_obs", transition_obs)):
            np.copyto(self.arrays[key][step], value)
        np.sqrt(rew_norm.variances + rew_norm.epsilon, out=self.arrays["reward_std"][step])


def main():
    args = validate_args(tyro.cli(Args))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_runtime(cudnn_deterministic=args.torch_deterministic,
                      matmul_precision="highest", allow_tf32=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    horizon = episode_horizon(args.env_id) if args.staggered_starts and args.num_envs > 1 else 0
    args.num_iterations = (args.total_timesteps - horizon * args.num_envs) // args.batch_size
    if args.num_iterations <= 0:
        raise ValueError("total_timesteps must cover phase warmup and a full rollout")
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
        writer.add_text("policy", "Beta: alpha,beta=1+softplus(head); FP32; native-action storage; host actor mirror")
        writer.add_text("architecture", "residual SiTU-GLU; unit-L2 matrices and residual directions; sqrt(width) branch/readout scale; learned output gains; no learned mixing")
        envs = make_training_env(args, run_name)
        resources.callback(envs.close)
        agent = Agent(envs).to(device)
        normalize_matrices = agent.normalize_matrices
        if args.compile:
            normalize_matrices = torch.compile(normalize_matrices, fullgraph=True,
                                               options={"triton.cudagraphs": False})
        normalize_matrices()
        matrix_optimizer, optimizer = make_optimizers(agent, args)
        torch.save({"model": agent.state_dict(), "args": vars(args),
                    "global_step": 0, "capture": "initialization_before_warmup"},
                   f"runs/{run_name}/initial_checkpoint.pt")
        actor_parameters, critic_parameters = tuple(agent.actor.parameters()), tuple(agent.critic.parameters())
        value_model = agent.get_value

        def rollout_statistics(observations, native):
            """Old log-probabilities and values for a whole uploaded rollout in one forward."""
            alpha, beta, value = agent.get_policy_and_value(observations)
            return value.flatten(), agent.action_logprob(alpha, beta, native), alpha, beta

        def loss_model(observations, native, old_logprobs, advantages, returns, old_values):
            return ppo_loss(agent, observations, native, old_logprobs, advantages, returns, old_values, args)

        policy_kl = lambda obs, alpha, beta: exact_policy_kl(agent, obs, alpha, beta)
        if args.compile:
            policy_kl = torch.compile(policy_kl, fullgraph=True, options={"triton.cudagraphs": False})
            rollout_statistics = graph_compile(rollout_statistics)
            # Batched final-observation counts vary; avoid fixed-shape graph recaptures.
            value_model = torch.compile(value_model, fullgraph=True, dynamic=True,
                                        options={"triton.cudagraphs": False})
            loss_model = torch.compile(loss_model, mode=args.compile_mode, fullgraph=True, dynamic=False)
        gae_fn = get_gae_fn(compiled=args.compile, mode=args.compile_mode)
        obs_shape = envs.single_observation_space.shape
        # The rollout never touches the GPU: act from an FP32 host mirror of the
        # actor, stage everything in pinned memory, upload once per rollout.
        host_actor = ResidualHostMirror(agent.actor, args.num_envs)
        action_low, action_high = (buffer.cpu().numpy() for buffer in (agent.action_low, agent.action_high))
        sampler = np.random.default_rng(args.seed)
        sample_actions = make_beta_sampler(args.num_envs, agent.action_dim, action_low, action_high)

        def act(observations):
            native, action = sample_actions(host_actor(observations), sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("policy produced nonfinite actions")
            return native, action.reshape((args.num_envs,) + agent.action_shape)

        transfer = RolloutTransfer(args.num_steps, args.num_envs, obs_shape, device,
                                   non_blocking=args.non_blocking_transfers,
                                   fields={"observations": obs_shape, "native_actions": (agent.action_dim,)})
        resources.callback(transfer.close)
        bootstraps = TruncationBootstrapCache(args.num_steps, args.num_envs, obs_shape)
        obs_norm = VectorObsNorm(args.num_envs, obs_shape)
        rew_norm = VectorRewardNorm(args.num_envs, args.gamma)
        # Shuffling must not consume the policy sampler's CUDA random stream.
        shuffle_generator = torch.Generator(device=device).manual_seed(args.seed)
        max_updates = args.update_epochs * ((args.batch_size + args.minibatch_size - 1) // args.minibatch_size)
        update_metrics = torch.empty((max_updates, 6), device=device)
        grad_norms = torch.empty((max_updates, 2), device=device)
        epoch_kls = torch.empty(args.update_epochs, device=device)
        timer = PhaseTimer()
        start_time = time.perf_counter()
        suppress = np.zeros(args.num_envs, dtype=bool)

        def warmup_action(observations):
            return act(observations)[1]

        if horizon:
            phases = compute_phase_offsets(args.num_envs, horizon, args.seed)
            writer.add_text("initial_phase_offsets", ",".join(map(str, phases)))
            warm = run_phase_warmup(envs, obs_norm=obs_norm, rew_norm=rew_norm,
                                    act_fn=warmup_action, horizon=horizon,
                                    phase_offsets=phases, seed=args.seed)
            next_obs_np, global_step, suppress = warm.next_obs, warm.transitions, warm.suppress_mask
        else:
            raw_obs, _ = envs.reset(seed=args.seed)
            next_obs_np, global_step = obs_norm.normalize(raw_obs), 0
        writer.add_scalar("timing/warmup_s", time.perf_counter() - start_time, global_step)
        interval_start, interval_step = time.perf_counter(), global_step

        captured_steps = set()
        for iteration in range(1, args.num_iterations + 1):
            if args.anneal_lr:
                fraction = 1.0 - (iteration - 1.0) / args.num_iterations
                optimizer.param_groups[0]["lr"] = fraction * args.learning_rate
                if matrix_optimizer is not None:
                    matrix_optimizer.set_lr(fraction * args.normuon_lr)
            capture_targets = (4000000, args.num_iterations * args.batch_size + horizon * args.num_envs)
            capturing = any(global_step < target <= global_step + args.batch_size for target in capture_targets)
            dense = TransitionCapture(envs, args.num_steps) if capturing else None
            capture_rng = {"sampler": copy.deepcopy(sampler.bit_generator.state),
                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
                "shuffle": shuffle_generator.get_state()} if capturing else None
            update_trace = []
            bootstraps.reset()
            host_actor.refresh()
            for step in range(args.num_steps):
                with timer.span("rollout", use_cuda=False):
                    obs_step = next_obs_np
                    native, host_action = act(obs_step)
                    if dense is not None:
                        dense.before(step, envs, host_action, obs_norm)
                with timer.span("env", use_cuda=False):
                    raw_obs, raw_reward, terms, truncs, infos = envs.step(host_action)
                with timer.span("normalize_transfer", use_cuda=False):
                    reward = rew_norm.normalize(raw_reward, terms)
                    next_obs_np, transition_obs = obs_norm.normalize_step(raw_obs, terms, truncs, infos)
                    if dense is not None:
                        dense.after(step, raw_obs, raw_reward, next_obs_np, transition_obs, rew_norm)
                    bootstraps.push_normalized(step, truncs, transition_obs)
                    transfer.push(step, reward, terms, truncs, observations=obs_step, native_actions=native)
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

            with timer.span("gae"), torch.no_grad():
                batch = transfer.upload()
                b_obs = batch.fields["observations"].flatten(0, 1)
                b_native = batch.fields["native_actions"].flatten(0, 1)
                b_values, b_logprobs, old_alpha, old_beta = rollout_statistics(b_obs, b_native)
                values = b_values.view(args.num_steps, args.num_envs)
                next_obs = transfer.observation(next_obs_np)
                tail_value = value_model(next_obs).flatten()
                truncation_values = bootstraps.resolve(value_model, device)
                advantages, returns = gae_fn(
                    batch.rewards, values, batch.terminations, batch.truncations,
                    truncation_values, tail_value, args.gamma, args.gae_lambda,
                )
                b_advantages = advantages.flatten().clone()
                b_returns = returns.flatten().clone()
            for capture_step in capture_targets:
                if global_step >= capture_step and capture_step not in captured_steps:
                    torch.save({
                        "model": agent.state_dict(), "args": vars(args),
                        "global_step": global_step, "optimizer": optimizer.state_dict(),
                        "matrix_optimizer": matrix_optimizer.state_dict() if matrix_optimizer is not None else None,
                        "batch": {"obs": b_obs, "native": b_native, "logprob": b_logprobs,
                                  "advantages": b_advantages, "returns": b_returns, "values": b_values,
                                  "alpha": old_alpha, "beta": old_beta},
                        "transitions": {"rewards": batch.rewards, "terminations": batch.terminations,
                            "truncations": batch.truncations, "truncation_values": truncation_values,
                            "tail_value": tail_value, **dense.arrays},
                        "rollout_start_rng": capture_rng,
                        "capture_protocol": {"timing": "pre_update", "state_spec": integration_backend(),
                            "integration_fields": list(INTEGRATION_FIELDS), "integration_time_included": True,
                            "reward_scale": "post-update std actually applied; raw reward separately stored",
                            "normalization": "recorded per-env rows; no pooled substitution"},
                        "obs_norm": {name: getattr(obs_norm, name) for name in ("means", "variances", "counts")},
                        "rew_norm": {name: getattr(rew_norm, name) for name in ("returns", "means", "variances", "counts")},
                    }, f"runs/{run_name}/diagnostic_snapshot_{global_step}.pt")
                    captured_steps.add(capture_step)
            updates = 0
            if matrix_optimizer is not None:
                matrix_optimizer.begin_rollout(max_updates)
            with timer.span("update"):
                for epoch in range(args.update_epochs):
                    for indices in device_minibatches(args.batch_size, args.minibatch_size, device, shuffle_generator):
                        if args.compile:
                            torch.compiler.cudagraph_mark_step_begin()
                        loss, metrics = loss_model(
                            b_obs[indices], b_native[indices], b_logprobs[indices],
                            b_advantages[indices], b_returns[indices], b_values[indices],
                        )
                        agent.zero_grad(set_to_none=True)
                        loss.backward()
                        grad_norms[updates, 0] = nn.utils.clip_grad_norm_(actor_parameters, args.max_grad_norm, foreach=True)
                        grad_norms[updates, 1] = nn.utils.clip_grad_norm_(critic_parameters, args.max_grad_norm, foreach=True)
                        if dense is not None:
                            pre_weights = {name: p.detach().clone() for name, p in agent.named_parameters()}
                            gradients = {name: p.grad.detach().clone() for name, p in agent.named_parameters()}
                        if matrix_optimizer is not None:
                            matrix_optimizer.step()
                        optimizer.step()
                        if dense is not None:
                            ambient = {name: p.detach() - pre_weights[name] for name, p in agent.named_parameters()}
                        normalize_matrices()
                        if dense is not None:
                            update_trace.append({"epoch": epoch, "update": updates,
                                "gradient_after_clip": gradients, "ambient_delta": ambient,
                                "matrix_geometry": args.matrix_geometry,
                                "matrix_delta_stage": ("intrinsic endpoint before global normalization"
                                                       if args.matrix_geometry == "intrinsic"
                                                       else "ambient proposal before global normalization"),
                                "retracted_delta": {name: p.detach() - pre_weights[name]
                                                    for name, p in agent.named_parameters()}})
                        update_metrics[updates].copy_(metrics)
                        updates += 1
                    with torch.no_grad():
                        epoch_kls[epoch] = policy_kl(b_obs, old_alpha, old_beta)
                    # Preserve last-minibatch KL, checked after a complete epoch.
                    # This optional control-flow synchronization is intentional.
                    if args.target_kl is not None and update_metrics[updates - 1, 4] > args.target_kl:
                        break

            if dense is not None:
                torch.save({"global_step": global_step, "trace": update_trace,
                            "model_after_update": agent.state_dict()},
                           f"runs/{run_name}/diagnostic_updates_{global_step}.pt")
            with torch.no_grad():
                current_alpha, current_beta, _ = agent.get_policy_and_value(b_obs)
                old_concentration = old_alpha + old_beta
                concentration = current_alpha + current_beta
                policy_metrics = {
                    "policy/concentration": concentration.mean(),
                    "policy/mean_step_rms": ((current_alpha / concentration
                        - old_alpha / old_concentration).square().mean()).sqrt(),
                    "policy/log_concentration_step": (concentration.log()
                        - old_concentration.log()).mean(),
                }
                for bank, (group, ps, state, _) in enumerate(matrix_optimizer.stacks if matrix_optimizer is not None else ()):
                    if group["geometry"] == "intrinsic":
                        for index, label in enumerate(("pathology", "minimum_transport_denominator",
                                                      "maximum_unit_norm_error", "maximum_rail_radial",
                                                      "maximum_direction_radial")):
                            policy_metrics[f"intrinsic/bank{bank}_{label}"] = state["diagnostics"][index]
                        continue
                    gradient = torch.stack([p.grad for p in ps])
                    _, lookahead, fast_beta, fast_weight, slow_beta = group["scalars"].unbind()
                    blend = state["velocity"][0]
                    if group["method"] == "anvil2":
                        blend = fast_weight * blend + (1 - fast_weight) * state["velocity"][1]
                    nesterov = gradient.lerp(blend, lookahead)
                    norm = nesterov.norm(dim=(-2, -1))
                    cosine = (gradient * blend).sum(dim=(-2, -1)) / (
                        gradient.norm(dim=(-2, -1)) * blend.norm(dim=(-2, -1))).clamp_min(1e-20)
                    policy_metrics[f"matrix_last/bank{bank}_grad_momentum_cosine"] = cosine.mean()
                    policy_metrics[f"matrix_last/bank{bank}_nesterov_norm"] = norm.mean()
                    policy_metrics[f"matrix_last/bank{bank}_epsilon_fraction"] = (1e-6 / (norm + 1e-6)).mean()
            last = update_metrics[updates - 1]
            logged = gather_metrics({
                **policy_metrics,
                "losses/post_first_epoch_exact_kl": epoch_kls[0],
                "losses/post_final_epoch_exact_kl": epoch_kls[epoch],
                "losses/mean_preupdate_kl": update_metrics[:updates, 4].mean(),
                "losses/policy_loss": last[0], "losses/value_loss": last[1],
                "losses/entropy": last[2], "losses/old_approx_kl": last[3],
                "losses/approx_kl": last[4], "losses/clipfrac": update_metrics[:updates, 5].mean(),
                "losses/explained_variance": explained_variance(b_values, b_returns),
                "grad/actor_preclip_norm": grad_norms[:updates, 0].mean(),
                "grad/critic_preclip_norm": grad_norms[:updates, 1].mean(),
            })
            if any(value != 0 for name, value in logged.items()
                   if name.startswith("intrinsic/") and name.endswith("_pathology")):
                details = {name: value for name, value in logged.items() if name.startswith("intrinsic/")}
                raise FloatingPointError(f"invalid intrinsic constraint/transport state: {details}")
            if any(not np.isfinite(value) for name, value in logged.items()
                   if name != "losses/explained_variance"):
                raise FloatingPointError("nonfinite PPO learner metrics")
            for name, value in logged.items():
                writer.add_scalar(name, value, global_step)
            now = time.perf_counter()
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/normuon_learning_rate", args.normuon_lr * (fraction if args.anneal_lr else 1.0), global_step)
            writer.add_scalar("charts/SPS", int(global_step / (now - start_time)), global_step)
            writer.add_scalar("charts/interval_SPS", (global_step - interval_step) / (now - interval_start), global_step)
            for phase, timing in timer.summary().items():
                writer.add_scalar(f"timing/{phase}_s", timing["total_s"], global_step)
            timer.reset()
            print(f"SPS: {int(global_step / (time.perf_counter() - start_time))}")
            interval_start, interval_step = time.perf_counter(), global_step

        torch.save({"model": agent.state_dict(), "args": vars(args), "global_step": global_step,
                    "obs_norm": {name: getattr(obs_norm, name) for name in ("means", "variances", "counts")},
                    "rew_norm": {name: getattr(rew_norm, name) for name in ("returns", "means", "variances", "counts")}},
                   f"runs/{run_name}/{args.exp_name}.pt")
        transfer.close()
        envs.close()
        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
            from cleanrl_utils.evals.ppo_eval import evaluate
            episodic_returns = evaluate(
                model_path, make_env, args.env_id, eval_episodes=10,
                run_name=f"{run_name}-eval", Model=Agent, device=device, gamma=args.gamma,
            )
            for index, episodic_return in enumerate(episodic_returns):
                writer.add_scalar("eval/episodic_return", episodic_return, index)
            if args.upload_model:
                from cleanrl_utils.huggingface import push_to_hub
                repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
                repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
                push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")
    finally:
        resources.close()


if __name__ == "__main__":
    main()
