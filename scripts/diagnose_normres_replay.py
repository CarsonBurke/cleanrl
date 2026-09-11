"""Replay one trusted pre-collapse PPO update; run through mlq, not as a benchmark."""

import argparse
import copy
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Beta

# Also support direct invocation from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanrl.shared.host_graph import HostGraphActor, make_host_mirror
from cleanrl.shared.rollout_graph import graph_compile
from cleanrl.shared.runtime import configure_runtime

TRAINER = "cleanrl.ppo_continuous_action_32xlr_1mb_noadvnorm_normres_twohot_indclip_v4"
BATCH_KEYS = (
    "observations",
    "native_actions",
    "old_logprobs",
    "advantages",
    "targets",
    "returns",
    "old_values",
)


def make_agent(trainer, args, snapshot, device):
    observations = snapshot["batch"]["observations"]
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=tuple(observations.shape[1:]), dtype=np.float32),
        single_action_space=gym.spaces.Box(
            snapshot["model"]["action_low"].numpy().copy(),
            snapshot["model"]["action_high"].numpy().copy(),
            dtype=np.float32,
        ),
    )
    # Initialization also stays on CUDA; no environment is created or stepped.
    with torch.device(device):
        agent = trainer.Agent(
            spaces,
            placement=args.placement,
            norm_kind=args.norm_kind,
            activation=args.activation,
            value_loss=args.value_loss,
            value_num_bins=args.value_num_bins,
            value_max_abs=args.value_max_abs,
            value_spacing=args.value_spacing,
        )
    agent.load_state_dict(copy.deepcopy(snapshot["model"]))
    return agent


def beta_kl(alpha0, beta0, alpha1, beta1):
    """Exact KL(reference || candidate), FP64, summed over action dimensions."""
    a, b, c, d = (x.double() for x in (alpha0, beta0, alpha1, beta1))
    total = a + b
    return (
        torch.lgamma(c)
        + torch.lgamma(d)
        - torch.lgamma(c + d)
        - torch.lgamma(a)
        - torch.lgamma(b)
        + torch.lgamma(total)
        + (a - c) * torch.digamma(a)
        + (b - d) * torch.digamma(b)
        + (c + d - total) * torch.digamma(total)
    ).sum(-1)


def error_stats(candidate, reference):
    error = candidate.double() - reference.double()
    return torch.stack((error.abs().max(), error.square().mean().sqrt()))


def ratio_stats(logprobs, reference):
    ratio = (logprobs.double() - reference.double()).exp()
    return torch.stack((ratio.min(), ratio.mean(), ratio.max(), (ratio - 1).abs().max()))


def norm(tensors):
    return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(x) for x in tensors]))


@torch.no_grad()
def initial_diagnostics(agent, args, batch, snapshot):
    def statistics(observations, native):
        alpha, beta, value = agent.get_policy_and_value(observations)
        return alpha, beta, agent.action_logprob(alpha, beta, native), value.flatten()

    # Match the captured batched rollout-statistics compiler configuration.
    statistics = graph_compile(statistics)
    outputs = statistics(batch["observations"], batch["native_actions"])
    alpha0, beta0, cuda_logprobs, values0 = (x.detach().clone() for x in outputs)
    del outputs
    mirror = make_host_mirror(agent.actor, args.num_envs)
    if not isinstance(mirror, HostGraphActor):
        raise RuntimeError("This diagnostic requires the real fused HostGraphActor")
    obs = snapshot["batch"]["observations"].numpy()
    concentrations = np.empty((len(obs), 2 * agent.action_dim), dtype=np.float32)
    for start in range(0, len(obs), args.num_envs):
        # The mirror reuses its output buffer. Own each result before calling again.
        logits = mirror(np.ascontiguousarray(obs[start : start + args.num_envs])).copy()
        concentration = np.logaddexp(0.0, logits, dtype=np.float32)
        concentration += 1.0
        concentrations[start : start + args.num_envs] = concentration
    host_alpha, host_beta = torch.from_numpy(concentrations).to(alpha0.device).chunk(2, dim=-1)
    host_logprobs = agent.action_logprob(host_alpha, host_beta, batch["native_actions"])
    old_logprobs = batch["old_logprobs"]
    comparisons = {
        "host alpha - CUDA": error_stats(host_alpha, alpha0),
        "host beta - CUDA": error_stats(host_beta, beta0),
        "host logp - stored": error_stats(host_logprobs, old_logprobs),
        "CUDA logp - stored": error_stats(cuda_logprobs, old_logprobs),
        "host logp - CUDA": error_stats(host_logprobs, cuda_logprobs),
        "CUDA value - stored": error_stats(values0, batch["old_values"]),
    }
    # A single transfer for these diagnostics, not one synchronization per scalar.
    errors = torch.stack(tuple(comparisons.values())).cpu().numpy()
    ratios = (
        torch.stack((ratio_stats(host_logprobs, old_logprobs), ratio_stats(cuda_logprobs, old_logprobs))).cpu().numpy()
    )
    host_kl = beta_kl(host_alpha, host_beta, alpha0, beta0)
    kl_summary = torch.stack((host_kl.mean(), host_kl.max())).cpu().numpy()
    print("\nInitial policy on captured normalized observations (no resampling):")
    print(f"{'comparison':27s} {'max_abs':>13s} {'RMS':>13s}")
    for name, row in zip(comparisons, errors):
        print(f"{name:27s} {row[0]:13.6g} {row[1]:13.6g}")
    print(f"{'first-forward exp(logp-stored)':32s} {'min':>12s} {'mean':>12s} {'max':>12s} {'max|r-1|':>12s}")
    for name, row in zip(("HostGraph", "CUDA compiled"), ratios):
        print(f"{name:32s}" + "".join(f" {x:12.6g}" for x in row))
    print(f"FP64 analytic KL(host || CUDA): mean={kl_summary[0]:.9g} max={kl_summary[1]:.9g}")
    return alpha0, beta0, host_alpha, host_beta


def compare_final(agent, optimizer, snapshot):
    """Report actual discrepancies, without declaring approximate equality."""
    names, rows = [], []
    for name, current in agent.state_dict().items():
        expected = snapshot["post_model"][name].to(current.device)
        names.append(name)
        rows.append(error_stats(current, expected))
    errors = torch.stack(rows).cpu().numpy()
    worst = np.argsort(errors[:, 0])[::-1][:5]
    print("\nCaptured post_model vs independent/currentLR (no tolerance relaxation):")
    print(f"all_state_max_abs={errors[:, 0].max():.9g}; exactly_equal={bool(np.all(errors[:, 0] == 0))}")
    for index in worst:
        print(f"  {names[index]:55s} max={errors[index, 0]:.9g} RMS={errors[index, 1]:.9g}")
    # Adam comparison detects mismatched steps/moments even if parameter drift is tiny.
    expected_state = snapshot["post_optimizer"]["state"]
    actual_state = optimizer.state_dict()["state"]
    moment_names, moment_rows = [], []
    if actual_state.keys() != expected_state.keys():
        raise ValueError("Final Adam state parameter IDs differ from the capture")
    for parameter_id, state in actual_state.items():
        if state.keys() != expected_state[parameter_id].keys():
            raise ValueError("Final Adam state fields differ from the capture")
        for name, value in state.items():
            expected = expected_state[parameter_id][name]
            if isinstance(value, torch.Tensor):
                moment_names.append(f"{parameter_id}/{name}")
                moment_rows.append(error_stats(value.to(agent.action_low.device), expected.to(agent.action_low.device)))
            elif value != expected:
                print(f"Adam {parameter_id}/{name}: replay={value!r} capture={expected!r}")
    if moment_rows:
        moment_errors = torch.stack(moment_rows).cpu().numpy()
        index = int(np.argmax(moment_errors[:, 0]))
        print(f"Adam tensor max_abs={moment_errors[index, 0]:.9g} at {moment_names[index]}")


def replay_branch(trainer, args, snapshot, device, references, name, joint, lr_scale):
    agent = make_agent(trainer, args, snapshot, device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
    # load_state_dict copies/casts into this optimizer; deepcopy also owns all CPU state.
    optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
    for group in optimizer.param_groups:
        if group["eps"] != 1e-5 or not group["fused"]:
            raise ValueError("Capture is not the required fused Adam eps=1e-5")
        group["lr"] *= lr_scale
    batch = {key: snapshot["batch"][key].to(device, copy=True) for key in BATCH_KEYS}
    fixed_before = {key: value.clone() for key, value in batch.items()}
    indices_sequence = [indices.to(device, copy=True) for indices in snapshot["minibatch_indices"]]
    actor = tuple(agent.actor.parameters())
    critic = tuple(agent.critic.parameters())
    actor_head = tuple(agent.actor[-1].parameters())
    critic_head = tuple(agent.critic[-1].parameters())
    groups = (actor, critic, actor_head, critic_head)
    # One copy per parameter; heads are subsets of the first two groups.
    before = {id(parameter): torch.empty_like(parameter) for parameter in agent.parameters()}

    def loss_model(observations, native, old_logprobs, advantages, targets):
        return trainer.ppo_loss(agent, observations, native, old_logprobs, advantages, targets, args)

    loss_model = torch.compile(loss_model, mode="reduce-overhead", fullgraph=True, dynamic=False)
    alpha0, beta0, host_alpha0, host_beta0 = references

    def policy_diagnostics(observations, native, old_logprobs, returns):
        alpha, beta, values = agent.get_policy_and_value(observations)
        # Evaluate all estimators in FP64, using precisely the saved native samples.
        logprob = (
            Beta(alpha.double(), beta.double(), validate_args=False).log_prob(native.double())
            - agent.log_action_scale.double()
        ).sum(-1)
        logratio = logprob - old_logprobs.double()
        cuda_kl = beta_kl(alpha0, beta0, alpha, beta)
        host_kl = beta_kl(host_alpha0, host_beta0, alpha, beta)
        error = values.flatten().double() - returns.double()
        return torch.stack(
            (
                (logratio.expm1() - logratio).mean(),
                -logratio.mean(),
                cuda_kl.mean(),
                cuda_kl.max(),
                host_kl.mean(),
                error.mean(),
                error.square().mean().sqrt(),
            )
        )

    # These extra forwards do not join the loss's CUDA-graph tree or own its outputs.
    policy_diagnostics = graph_compile(policy_diagnostics)
    rows = torch.empty((len(indices_sequence), 15), dtype=torch.float64, device=device)
    loss_rows = torch.empty((len(indices_sequence), 6), device=device)
    with torch.no_grad():
        initial = policy_diagnostics(
            batch["observations"], batch["native_actions"], batch["old_logprobs"], batch["returns"]
        ).clone()
    for step, indices in enumerate(indices_sequence):
        with torch.no_grad():
            for parameter in agent.parameters():
                before[id(parameter)].copy_(parameter)
        torch.compiler.cudagraph_mark_step_begin()
        loss, metrics = loss_model(
            batch["observations"][indices],
            batch["native_actions"][indices],
            batch["old_logprobs"][indices],
            batch["advantages"][indices],
            batch["targets"][indices],
        )
        optimizer.zero_grad(set_to_none=True)
        # AOTAutograd compiles the backward of the real, fullgraph PPO loss.
        loss.backward()
        with torch.no_grad():
            head_norms = torch.stack((norm(p.grad for p in actor_head), norm(p.grad for p in critic_head)))
        if joint:
            actor_norm, critic_norm = norm(p.grad for p in actor), norm(p.grad for p in critic)
            torch.nn.utils.clip_grad_norm_(actor + critic, args.max_grad_norm, foreach=True)
        else:
            actor_norm, critic_norm = trainer.clip_gradients(actor, critic, args.max_grad_norm)
        optimizer.step()
        with torch.no_grad():
            rows[step, :4].copy_(torch.stack((actor_norm, critic_norm, head_norms[0], head_norms[1])))
            rows[step, 4:8].copy_(
                torch.stack(tuple(norm(parameter - before[id(parameter)] for parameter in group) for group in groups))
            )
            # Consume graph-owned metrics now, before the next mark/forward.
            loss_rows[step].copy_(metrics)
            rows[step, 8:].copy_(
                policy_diagnostics(
                    batch["observations"], batch["native_actions"], batch["old_logprobs"], batch["returns"]
                )
            )
        del loss, metrics
    unchanged = torch.stack([torch.eq(batch[key], fixed_before[key]).all() for key in BATCH_KEYS]).cpu().tolist()
    print(f"\n{name}: lr={[group['lr'] for group in optimizer.param_groups]} steps={len(indices_sequence)}")
    print("Weighted objective gradients BEFORE clipping; d* are AFTER Adam parameter-step L2 norms.")
    print("KLs: fixed whole batch, FP64; sampled estimators use captured FP32 old_logprobs.")
    print("analytic KL direction is original || updated; KLhost uses the original host concentrations.")
    initial_cpu = initial.cpu().numpy()
    print(
        "initial "
        + " ".join(
            f"{key}={value:.7g}"
            for key, value in zip(
                ("sampleKL", "neglogKL", "KLcuda", "KLcudaMax", "KLhost", "bias", "RMSE"), initial_cpu
            )
        )
    )
    headers = (
        "step",
        "gActor",
        "gCritic",
        "gAHead",
        "gCHead",
        "dActor",
        "dCritic",
        "dAHead",
        "dCHead",
        "sampleKL",
        "neglogKL",
        "KLcuda",
        "KLcudaMax",
        "KLhost",
        "bias",
        "RMSE",
    )
    print(f"{headers[0]:>4s}" + "".join(f" {header:>11s}" for header in headers[1:]))
    for step, row in enumerate(rows.cpu().numpy(), 1):
        print(f"{step:4d}" + "".join(f" {value:11.5g}" for value in row))
    objective = loss_rows.cpu().numpy()
    print("PPO pre-step minibatch metrics (original FP32 objective):")
    print(
        f"{'step':>4s} {'policy':>11s} {'value':>11s} {'entropy':>11s} {'neglogKL':>11s} {'sampleKL':>11s} {'clipfrac':>11s}"
    )
    for step, row in enumerate(objective, 1):
        print(f"{step:4d}" + "".join(f" {value:11.5g}" for value in row))
    print("fixed batch unchanged: " + " ".join(f"{key}={ok}" for key, ok in zip(BATCH_KEYS, unchanged)))
    if not all(unchanged):
        raise RuntimeError(f"{name} mutated captured batch inputs")
    if not joint and lr_scale == 1.0:
        compare_final(agent, optimizer, snapshot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "snapshot", type=Path, help="Trusted own update_probe_*.pt (pickle; do not load untrusted files)"
    )
    path = parser.parse_args().snapshot
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    if snapshot["format_version"] != 1 or snapshot["trainer_module"] != TRAINER:
        raise ValueError("Unsupported snapshot format or trainer")
    trainer = importlib.import_module(snapshot["trainer_module"])
    args = trainer.Args(**snapshot["args"])
    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    if not args.compile or args.compile_mode != "reduce-overhead":
        raise ValueError("Capture must use compiled reduce-overhead updates")
    if len(snapshot["batch"]["observations"]) != args.batch_size or args.batch_size % args.num_envs:
        raise ValueError("Captured observations do not match Args batch layout")
    if not snapshot["minibatch_indices"]:
        raise ValueError("Snapshot has no optimizer steps")
    configure_runtime(cudnn_deterministic=args.torch_deterministic, matmul_precision="highest", allow_tf32=False)
    device = torch.device("cuda")
    print(f"snapshot={path} global_step={snapshot['global_step']} trainer={snapshot['trainer_module']}")
    print(
        f"objective={args.value_loss} vf_coef={args.vf_coef} ent_coef={args.ent_coef} norm_adv={args.norm_adv} "
        f"clip_coef={args.clip_coef} max_grad_norm={args.max_grad_norm} batch={args.batch_size} "
        f"epochs={args.update_epochs} num_minibatches={args.num_minibatches}"
    )
    print(
        f"architecture={args.placement}/{args.norm_kind}/{args.activation} reward_norm={args.reward_norm} "
        f"support={args.value_num_bins}/{args.value_spacing}/+/-{args.value_max_abs} "
        f"schedule={args.total_timesteps} seed={args.seed}"
    )
    print("Capture input immutability: " + " ".join(f"{key}={ok}" for key, ok in snapshot["immutable_inputs"].items()))
    print("One-update diagnostic only: no environment interactions, new samples, or policy benchmark claims.")
    # Every construction starts with the same RNG state and restores it afterwards.
    # The reconstructed networks are deterministic; all sampled data comes from the snapshot.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        reference_agent = make_agent(trainer, args, snapshot, device)
        reference_batch = {key: snapshot["batch"][key].to(device, copy=True) for key in BATCH_KEYS}
        references = initial_diagnostics(reference_agent, args, reference_batch, snapshot)
    del reference_agent, reference_batch
    for name, joint, lr_scale in (
        ("independent/currentLR", False, 1.0),
        ("joint/currentLR", True, 1.0),
        ("independent/quarterLR", False, 0.25),
    ):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            replay_branch(trainer, args, snapshot, device, references, name, joint, lr_scale)


if __name__ == "__main__":
    main()
