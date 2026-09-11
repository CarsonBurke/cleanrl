"""Replay one real on-policy PPO rollout from a preRMS SiTU checkpoint on CUDA.

Run through mlq. This is an optimizer-reset diagnostic, NOT a continuation of
historical training: bundles lack Adam/reward-normalizer/environment/RNG state.
The saved observation moments are restored and updated by the shared normalizer;
reward moments are freshly calibrated during one stochastic phase-warmup horizon.
No synthetic observations, actions, advantages, or parameter perturbations enter
updates. The historical sphere run is inventoried, never loaded as preRMS.
"""

import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.distributions import Beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanrl.ppo_continuous_action_32xlr_1mb_noadvnorm_normres_v2 import Agent, ppo_loss
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.ppo_loop import TruncationBootstrapCache, device_minibatches, get_gae_fn
from cleanrl.shared.rollout_transfer import RolloutTransfer
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import sample_beta_actions_host
from cleanrl.shared.staggered_envs import compute_phase_offsets, episode_horizon, run_phase_warmup
from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

ROOT = Path(__file__).resolve().parents[1]
SPHERE_RUN = ROOT / "runs/HalfCheetah-v4__ppo_32xlr_1mb_noadvnorm_stiglu_sphere_50M__1__1788662844"
LR = 0.0096
EPOCHS = 10
ADAM_EPS = 1e-5
MAX_GRAD_NORM = 0.5


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(value):
    """CUDA reductions; host synchronization is deferred until JSON conversion."""
    value = value.detach().double().flatten()
    quantiles = torch.quantile(value, value.new_tensor([0.5, 0.95, 0.99]))
    return {"mean": value.mean(), "p50": quantiles[0], "p95": quantiles[1],
            "p99": quantiles[2], "min": value.min(), "max": value.max()}


def json_ready(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def beta_forward_kl(a, b, c, d):
    """Closed-form KL(Beta(a,b) || Beta(c,d)), summed over action axes."""
    log_b_old = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    log_b_new = torch.lgamma(c) + torch.lgamma(d) - torch.lgamma(c + d)
    return (log_b_new - log_b_old + (a - c) * torch.digamma(a)
            + (b - d) * torch.digamma(b)
            + (c + d - a - b) * torch.digamma(a + b)).sum(-1)


def policy_change_arrays(old_a, old_b, new_a, new_b, native, action_scale):
    """Distribution changes, independent of rewards/advantages and their signs."""
    a, b, c, d = (item.double() for item in (old_a, old_b, new_a, new_b))
    old_k, new_k = a + b, c + d
    old_mean, new_mean = a / old_k, c / new_k
    # These coordinate interventions are valid positive-parameter Betas even if
    # a resulting shape parameter is below the model head's >1 constraint.
    mean_kl = beta_forward_kl(a, b, new_mean * old_k, (1 - new_mean) * old_k)
    concentration_kl = beta_forward_kl(a, b, old_mean * new_k, (1 - old_mean) * new_k)
    old_lp = Beta(a, b, validate_args=False).log_prob(native.double()).sum(-1)
    new_lp = Beta(c, d, validate_args=False).log_prob(native.double()).sum(-1)
    logratio = new_lp - old_lp
    return {
        "analytic_forward_kl": beta_forward_kl(a, b, c, d),
        "sampled_k1": -logratio,
        "sampled_k3": torch.expm1(logratio) - logratio,
        "mean_only_coordinate_kl": mean_kl,
        "concentration_only_coordinate_kl": concentration_kl,
        "native_mean_shift_l2": (new_mean - old_mean).norm(dim=-1),
        "physical_mean_shift_l2": ((new_mean - old_mean) * action_scale).norm(dim=-1),
        "log_concentration_shift_l2": (new_k.log() - old_k.log()).norm(dim=-1),
        "native_mean_absolute_shift": (new_mean - old_mean).abs(),
        "concentration_relative_shift": (new_k - old_k) / old_k,
        "old_concentration": old_k,
        "new_concentration": new_k,
    }


def residual_rms(trunk, observations):
    """Actual unnormalized residual-stream RMS before each preRMS block."""
    h = trunk.in_proj(observations)
    block_inputs = []
    for block, gate, norm in zip(trunk.blocks, trunk.block_gates, trunk.block_norms):
        block_inputs.append(h.square().mean(-1).sqrt())
        h = h + gate.sigmoid() * block(norm(h) * trunk.branch_input_scale)
    return torch.stack(block_inputs, dim=-1), h.square().mean(-1).sqrt()


def parameter_groups(agent):
    groups = {}
    for network in ("actor", "critic"):
        for label, prefix in (("stem", "0.in_proj."), ("branches", "0.blocks."),
                              ("gates", "0.block_gates."), ("head", "1.")):
            groups[f"{network}.{label}"] = {
                name: parameter for name, parameter in agent.named_parameters()
                if name.startswith(f"{network}.{prefix}")
            }
    covered = [name for group in groups.values() for name in group]
    if len(covered) != len(set(covered)) or set(covered) != dict(agent.named_parameters()).keys():
        raise ValueError("Parameter grouping must partition every actor/critic parameter")
    return groups


def norm_sum(tensors):
    return torch.stack([tensor.detach().double().square().sum() for tensor in tensors]).sum().sqrt()


def gradients(groups):
    group_norms = {name: norm_sum([p.grad for p in group.values()]) for name, group in groups.items()}
    networks = {network: torch.stack([norm.square() for name, norm in group_norms.items()
                                     if name.startswith(network + ".")]).sum().sqrt()
                for network in ("actor", "critic")}
    return {"groups": group_norms, "networks": networks,
            "joint": torch.stack([norm.square() for norm in networks.values()]).sum().sqrt()}


def motion(groups, previous, optimizer):
    result = {}
    for name, group in groups.items():
        parameter_norm = norm_sum([previous[key] for key in group])
        delta = norm_sum([parameter - previous[key] for key, parameter in group.items()])
        epsilon_shares = []
        for parameter in group.values():
            state = optimizer.state[parameter]
            bias_correction = 1 - optimizer.param_groups[0]["betas"][1] ** state["step"]
            denominator = (state["exp_avg_sq"] / bias_correction).sqrt()
            epsilon_shares.append((ADAM_EPS / (denominator + ADAM_EPS)).flatten())
        result[name] = {
            "parameter_l2_before": parameter_norm, "update_l2": delta,
            "relative_l2": torch.where(parameter_norm > 0, delta / parameter_norm,
                                        torch.full_like(delta, float("nan"))),
            "adam_epsilon_denominator_fraction": summary(torch.cat(epsilon_shares)),
        }
    return result


def effective_down_motion(agent, previous):
    """Resolve each finite effective-matrix row step relative to its old row."""
    result = {}
    for network in ("actor", "critic"):
        blocks = []
        trunk = getattr(agent, network)[0]
        for index, (block, gate) in enumerate(zip(trunk.blocks, trunk.block_gates)):
            prefix = f"{network}.0."
            old_gate = previous[f"{prefix}block_gates.{index}"].double().sigmoid()
            old_down = previous[f"{prefix}blocks.{index}.down.weight"].double()
            old = old_gate[:, None] * old_down
            new = gate.double().sigmoid()[:, None] * block.down.weight.double()
            delta = new - old
            old_energy = old.square().sum(-1)
            update_energy = delta.square().sum(-1)
            radial_coefficient = torch.where(old_energy > 0, (delta * old).sum(-1) / old_energy, 0.0)
            radial = radial_coefficient[:, None] * old
            radial_energy = radial.square().sum(-1)
            tangent_energy = (delta - radial).square().sum(-1)
            radial_fraction = torch.where(update_energy > 0, radial_energy / update_energy, 0.0)
            tangent_fraction = torch.where(update_energy > 0, tangent_energy / update_energy, 0.0)
            relative_step = torch.where(old_energy > 0, (update_energy / old_energy).sqrt(), 0.0)
            blocks.append({
                "block": index,
                "old_row_l2": summary(old_energy.sqrt()),
                "effective_relative_row_motion": summary(relative_step),
                "radial_signed_relative_row_motion": summary(radial_coefficient),
                "radial_squared_fraction": summary(radial_fraction),
                "tangent_squared_fraction": summary(tangent_fraction),
                "rows": {"radial_squared_fraction": radial_fraction,
                         "tangent_squared_fraction": tangent_fraction,
                         "effective_relative_motion": relative_step},
                "zero_old_rows": (old_energy == 0).sum(),
                "zero_update_rows": (update_energy == 0).sum(),
            })
        result[network] = blocks
    return result


def checkpoint_path(args):
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        return args.checkpoint
    paths = sorted(args.run.glob("*.cleanrl_model"))
    if len(paths) != 1:
        raise ValueError(f"--run needs exactly one .cleanrl_model; found {len(paths)} in {args.run}")
    return paths[0]


def load_bundle(path):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or not {"args", "model", "obs_norm"} <= bundle.keys():
        raise ValueError("Need exact model/args/obs_norm bundle; bare sphere state dicts cannot reconstruct normalization")
    saved = bundle["args"]
    if (saved.get("placement"), saved.get("norm_kind"), saved.get("activation")) != ("pre", "rms", "stiglu"):
        raise ValueError("Only exact normres_v2 preRMS SiTU bundles are supported; sphere is not interchangeable")
    if saved.get("env_id") != "HalfCheetah-v4":
        raise ValueError("This comparison is scoped to HalfCheetah-v4")
    required = {"num_envs", "num_steps", "gamma", "gae_lambda", "clip_coef", "clip_vloss", "ent_coef", "vf_coef"}
    if not required <= saved.keys():
        raise ValueError(f"Missing PPO configuration: {sorted(required - saved.keys())}")
    if saved["num_envs"] <= 0 or saved["num_steps"] <= 0:
        raise ValueError("Saved rollout dimensions must be positive")
    return bundle


def collect_rollout(agent, envs, config, norm_state, seed, resources):
    n, steps = config.num_envs, config.num_steps
    obs_shape = envs.single_observation_space.shape
    obs_norm = VectorObsNorm(n, obs_shape, epsilon=norm_state["epsilon"], clip=norm_state["clip"])
    for key in ("means", "variances", "counts"):
        values = norm_state[key].numpy()
        target = getattr(obs_norm, key)
        if target.shape != values.shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid saved observation normalizer {key}")
        if key != "means" and (np.any(values < 0) or (key == "counts" and np.any(values == 0))):
            raise ValueError(f"Invalid saved observation normalizer {key}")
        target[...] = values
    reward_norm = VectorRewardNorm(n, config.gamma)
    mirror = make_host_mirror(agent.actor, n)
    low, high = agent.action_low.cpu().numpy(), agent.action_high.cpu().numpy()
    rng = np.random.default_rng(seed)

    def act(observations):
        native, physical = sample_beta_actions_host(mirror(observations), low, high, rng)
        if not np.isfinite(physical).all():
            raise FloatingPointError("Nonfinite host Beta actions")
        return native, physical.reshape((n,) + agent.action_shape)

    horizon = episode_horizon(config.env_id)
    phases = compute_phase_offsets(n, horizon, seed)
    warm = run_phase_warmup(envs, obs_norm=obs_norm, rew_norm=reward_norm,
                            act_fn=lambda obs: act(obs)[1], horizon=horizon,
                            phase_offsets=phases, seed=seed)
    next_obs = warm.next_obs
    transfer = RolloutTransfer(steps, n, obs_shape, torch.device("cuda"),
                               fields={"observations": obs_shape, "native_actions": (agent.action_dim,)})
    resources.callback(transfer.close)
    bootstraps = TruncationBootstrapCache(steps, n, obs_shape)
    raw_reward_sum = np.zeros(n, dtype=np.float64)
    terminations = truncations = 0
    for step in range(steps):
        observations = next_obs
        native, physical = act(observations)
        raw, reward, terms, truncs, infos = envs.step(physical)
        raw_reward_sum += reward
        normalized_reward = reward_norm.normalize(reward, terms)
        next_obs, transition_obs = obs_norm.normalize_step(raw, terms, truncs, infos)
        bootstraps.push_normalized(step, truncs, transition_obs)
        transfer.push(step, normalized_reward, terms, truncs, observations=observations, native_actions=native)
        terminations += int(terms.sum())
        truncations += int(truncs.sum())
    batch = transfer.upload()
    provenance = {
        "environment_seed": seed, "action_rng_seed": seed, "env_backend": "native",
        "num_envs": n, "steps_per_env": steps, "rows": n * steps,
        "host_actor": type(mirror).__name__, "phase_offsets": phases.tolist(),
        "warmup_steps_per_env": horizon, "warmup_transitions": warm.transitions,
        "terminations": terminations, "truncations": truncations,
        "truncation_bootstrap_rows": len(bootstraps),
        "raw_reward_sum_per_env_not_episode_returns": raw_reward_sum.tolist(),
        "observation_normalization": "restored saved per-env moments; shared online updates during warmup and rollout",
        "observation_counts_before": norm_state["counts"].tolist(),
        "observation_counts_after": obs_norm.counts.tolist(),
        "reward_normalization": "RESET (not saved); shared online normalization, calibrated by stochastic phase warmup",
        "reward_moments_after": {key: getattr(reward_norm, key).tolist()
                                 for key in ("means", "variances", "counts", "returns")},
    }
    return batch, transfer.observation(next_obs), bootstraps, provenance


def replay(agent, batch, tail_observations, bootstraps, config, seed):
    # No eager learner fallback. Disable CUDA graphs because diagnostic snapshots
    # remain live across forwards and parameter-restoration interventions.
    compile_options = {"fullgraph": True, "options": {"triton.cudagraphs": False}}

    def statistics(observations, native):
        a, b, value = agent.get_policy_and_value(observations)
        return a, b, value.flatten(), agent.action_logprob(a, b, native)

    def loss(observations, native, old_logprob, advantages, returns, old_values):
        return ppo_loss(agent, observations, native, old_logprob, advantages, returns, old_values, config)

    def streams(observations):
        return residual_rms(agent.actor[0], observations), residual_rms(agent.critic[0], observations)

    compiled_stats = torch.compile(statistics, **compile_options)
    compiled_loss = torch.compile(loss, **compile_options)
    compiled_streams = torch.compile(streams, **compile_options)
    compiled_changes = torch.compile(policy_change_arrays, **compile_options)
    compiled_value = torch.compile(agent.get_value, dynamic=True, **compile_options)
    gae = get_gae_fn(compiled=True, mode="default")
    observations = batch.fields["observations"].flatten(0, 1)
    native = batch.fields["native_actions"].flatten(0, 1)
    with torch.no_grad():
        old_a, old_b, old_values, old_logprob = compiled_stats(observations, native)
        truncation_values = bootstraps.resolve(compiled_value, "cuda")
        tail_value = compiled_value(tail_observations).flatten()
        advantages, returns = gae(batch.rewards, old_values.view(config.num_steps, config.num_envs),
                                  batch.terminations, batch.truncations, truncation_values,
                                  tail_value, config.gamma, config.gae_lambda)
        advantages, returns = advantages.flatten().clone(), returns.flatten().clone()

    groups = parameter_groups(agent)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=ADAM_EPS, fused=True)
    shuffle = torch.Generator(device="cuda").manual_seed(seed)
    result = {"data": {"observations": summary(observations), "native_actions": summary(native),
                       "normalized_rewards": summary(batch.rewards), "raw_gae": summary(advantages),
                       "returns": summary(returns), "old_values": summary(old_values)},
              "parameter_groups": {name: {"names": list(group), "numel": sum(p.numel() for p in group.values())}
                                   for name, group in groups.items()},
              "updates": []}

    def changes(a, b, c, d):
        arrays = compiled_changes(a, b, c, d, native, agent.action_scale)
        return {name: summary(array) for name, array in arrays.items()}

    def rms():
        measurements = compiled_streams(observations)
        return {network: {"block_inputs": [summary(inputs[:, index]) for index in range(inputs.shape[1])],
                          "before_final_norm": summary(final)}
                for network, (inputs, final) in zip(("actor", "critic"), measurements)}

    for epoch in range(EPOCHS):
        with torch.no_grad():
            before_a, before_b, _, _ = compiled_stats(observations, native)
            before_metrics = changes(old_a, old_b, before_a, before_b)
            before_rms = rms()
            previous = {name: parameter.detach().clone() for name, parameter in agent.named_parameters()}
        indices = device_minibatches(config.batch_size, config.batch_size, "cuda", shuffle)[0]
        optimizer.zero_grad(set_to_none=True)
        objective, loss_metrics = compiled_loss(observations[indices], native[indices], old_logprob[indices],
                                                advantages[indices], returns[indices], old_values[indices])
        objective.backward()
        raw_gradients = gradients(groups)
        total_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
        clipped_gradients = gradients(groups)
        optimizer.step()
        with torch.no_grad():
            after_a, after_b, _, _ = compiled_stats(observations, native)
            update = {
                "optimizer_update": epoch + 1,
                "rollout_reference_before": before_metrics,
                "rollout_reference_after": changes(old_a, old_b, after_a, after_b),
                "immediate_step_reference_after": changes(before_a, before_b, after_a, after_b),
                "loss_before_step": dict(zip(("policy_loss", "value_loss", "entropy", "sampled_k1",
                                               "sampled_k3", "clip_fraction"), loss_metrics.unbind())),
                "objective_before_step": objective.detach(),
                "gradients_before_joint_clip": raw_gradients,
                "gradients_after_joint_clip": clipped_gradients,
                "joint_clip_multiplier": (MAX_GRAD_NORM / (total_norm + 1e-6)).clamp(max=1),
                "parameter_motion": motion(groups, previous, optimizer),
                "residual_rms_before": before_rms, "residual_rms_after": rms(),
            }
            update["effective_down_matrix_motion"] = effective_down_motion(agent, previous)
            if epoch == 0:
                full_step = {name: parameter.detach().clone() for name, parameter in agent.named_parameters()}
                interventions = {}
                restore_sets = {name: list(group) for name, group in groups.items() if name.startswith("actor.")}
                restore_sets["actor.trunk"] = [name for name in previous if name.startswith("actor.0.")]
                restore_sets["actor.down"] = [name for name in previous
                                               if name.startswith("actor.0.blocks.") and name.endswith(".down.weight")]
                parameters = dict(agent.named_parameters())
                for label, names in restore_sets.items():
                    try:
                        for name in names:
                            parameters[name].copy_(previous[name])
                        a, b, _, _ = compiled_stats(observations, native)
                        interventions[label] = changes(old_a, old_b, a, b)
                    finally:
                        for name in names:
                            parameters[name].copy_(full_step[name])
                update["counterfactual_restore_preupdate_actor_group"] = interventions
            result["updates"].append(update)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="Run directory containing exactly one .cleanrl_model")
    source.add_argument("--checkpoint", type=Path, help="Exact normres_v2 preRMS SiTU bundle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sphere-run", type=Path, default=SPHERE_RUN,
                        help="Inventory a sphere reference without assuming checkpoint compatibility")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env-threads", type=int, default=2)
    args = parser.parse_args()
    if args.env_threads < 1 or not 0 <= args.seed < 2 ** 32:
        parser.error("env-threads must be positive and seed must be in [0, 2**32)")
    path = checkpoint_path(args)
    bundle = load_bundle(path)
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; submit this diagnostic through mlq")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = SimpleNamespace(**bundle["args"])
    config.norm_adv = False
    config.batch_size = config.num_envs * config.num_steps
    sphere_paths = sorted(args.sphere_run.glob("*.cleanrl_model"))
    source_paths = [Path(__file__), ROOT / "cleanrl/ppo_continuous_action_32xlr_1mb_noadvnorm_normres_v2.py"]
    source_paths.extend(ROOT / "cleanrl/shared" / name for name in
                        ("norm_residual.py", "host_actor.py", "host_graph.py", "host_kernel.c",
                         "mujoco_env.py", "mujoco_batch.c", "ppo_loop.py", "gae.py", "runtime.py",
                         "rollout_transfer.py", "sampling.py", "vector_norm.py", "staggered_envs.py"))
    report = {
        "schema_version": 1,
        "checkpoint": {"path": str(path.resolve()), "sha256": sha256(path), "saved_args": bundle["args"]},
        "runtime": {"device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                    "learner": "CUDA float32 torch.compile fullgraph; no eager fallback",
                    "metric_math": "CUDA float64 analytic KL and reductions", "tf32": False,
                    "cuda_graphs": "disabled for compiled model/metric functions; shared compiled GAE defaults",
                    "seed": args.seed, "env_threads": args.env_threads},
        "source_sha256": {str(item.relative_to(ROOT)): sha256(item) for item in source_paths},
        "replay_contract": {"learning_rate": LR, "epochs": EPOCHS, "minibatches": 1,
                            "advantage_normalization": False, "optimizer": "RESET Adam fused, default betas=(0.9,0.999)",
                            "adam_epsilon": ADAM_EPS, "joint_actor_critic_grad_clip": MAX_GRAD_NORM,
                            "lr_annealing": False, "target_kl_stop": False,
                            "other_loss_and_gae_settings": "saved checkpoint args via exact trainer ppo_loss",
                            "dataset": "one full saved-size on-policy rollout, held fixed across ten updates"},
        "sphere_reference": {"run": str(args.sphere_run.resolve()),
                             "checkpoint_files": [str(item.resolve()) for item in sphere_paths],
                             "status": "unsupported_checkpoint" if sphere_paths else "no_checkpoint_available",
                             "reason": "No exact compatible sphere bundle adapter; no sphere model is fabricated or substituted"},
        "metric_definitions": {
            "analytic_forward_kl": "sum_action KL(Beta(reference)||Beta(current)) per observed state; closed form in float64; unclamped",
            "rollout_reference": "reference is checkpoint policy on fixed normalized rollout observations",
            "immediate_step_reference": "reference is policy immediately before this optimizer step",
            "sampled_k1": "log p_reference(a)-log p_current(a) on original native rollout actions; can be negative",
            "sampled_k3": "exp(log p_current(a)-log p_reference(a))-1-log p_current(a)+log p_reference(a)",
            "immediate_step_sampling_caveat": "after step one, native actions are NOT sampled from immediate-step reference; immediate sampled KL is off-policy, analytic KL remains exact",
            "coordinate_kl": "mean-only holds reference concentration, concentration-only holds reference mean; valid Beta counterfactuals, NOT additive decomposition",
            "relative_l2": "||parameter_after-parameter_before||_2 / ||parameter_before||_2 per disjoint group",
            "restoration": "after FIRST full-batch optimizer step, restore one actor group to pre-step tensors, evaluate, then restore full post-step state; not additive attribution",
            "head_and_trunk_only": "restoring actor.trunk measures head-only motion; restoring actor.head measures trunk-only motion",
            "gate_and_down_restoration": "actor.gates restores only external residual sigmoid gates; actor.down restores only branch down matrices, leaving internal SiTU gate/up matrices and all other updated tensors unchanged",
            "effective_down_matrix_motion": "M=diag(sigmoid(external_gate))W_down per block; project finite row delta onto old M row, report radial/tangent squared fractions and signed radial relative step; exact rowwise geometric split, not functional KL attribution; zero denominators report zero with explicit row counts",
            "residual_rms": "sqrt(mean_channels(h**2)) over actual pre-normalization block inputs and final residual output, both networks",
            "adam_epsilon_denominator_fraction": "eps/(sqrt(bias-corrected exp_avg_sq)+eps), over coordinates per group",
            "historical_metric_timing": "trainer logs final epoch PRE-step sampled KL; report explicitly measures both sides of all ten steps",
        },
        "limitations": [
            "Adam state, reward normalization, environment state and RNG state are not in the checkpoint; this is not a historical optimizer replay or resumed training.",
            "Constant requested LR .0096 may differ from the checkpoint's late annealed LR; no claim to reproduce historical KL spikes numerically.",
            "Newly calibrated reward normalization changes raw GAE scale and joint actor/critic clipping compared with mature training; inspect reported GAE and gradient norms.",
            "Host mirror drives genuine stochastic on-policy rollout; CUDA FP32 policy statistics can differ by floating-point roundoff from host sampling logits.",
            "Exact KL describes the continuous Beta, not the tiny boundary mass introduced by native sample clipping to [1e-6,1-1e-6].",
            "One seeded fixed-rollout diagnostic measures local update sensitivity, not expected training return or causal explanation of historical return gaps.",
            "Current shared source hashes describe this diagnostic runtime, not necessarily historical training source.",
        ],
    }
    with ExitStack() as resources:
        envs = make_mujoco_vector_env(config.env_id, config.num_envs, backend="native", num_threads=args.env_threads)
        resources.callback(envs.close)
        # Construct and initialize directly on CUDA; CPU use is limited to state
        # deserialization, native environment, normalizers, and host rollout mirror.
        with torch.device("cuda"):
            agent = Agent(envs, placement="pre", norm_kind="rms", activation="stiglu")
        low, high = (np.asarray(item).reshape(-1) for item in
                     (envs.single_action_space.low, envs.single_action_space.high))
        if not np.array_equal(low, bundle["model"]["action_low"].numpy()) or not np.array_equal(high, bundle["model"]["action_high"].numpy()):
            raise ValueError("Checkpoint and native environment action bounds differ")
        agent.load_state_dict(bundle["model"], strict=True)
        report["parameter_count"] = sum(parameter.numel() for parameter in agent.parameters())
        batch, tail, bootstraps, provenance = collect_rollout(agent, envs, config, bundle["obs_norm"], args.seed, resources)
        report["rollout"] = provenance
        report["replay"] = replay(agent, batch, tail, bootstraps, config, args.seed)
    payload = json.dumps(json_ready(report), indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(f"Optimizer-reset CUDA PPO diagnostic: {EPOCHS} full-batch updates; JSON {args.output}")


if __name__ == "__main__":
    main()
