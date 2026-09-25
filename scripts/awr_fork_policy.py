"""Continue a captured training state under any trainer: the forked-continuation bench.

    CLEANRL_PROBE_MODULE=<trainer module> CLEANRL_FORK_SNAPSHOT=<policy_snapshot_*.pt> \\
    CLEANRL_FORK_STEPS=8000000 python scripts/awr_fork_policy.py <the trainer's usual arguments>

The snapshot (scripts/awr_capture_policy.py, format 3) supplies the actor and critic weights, every
parameter's Adam state and the observation/reward normalizer statistics. They are loaded into the
trainer just before its phase warmup, so the warmup and every rollout after it use the captured policy.
The learning-rate schedule continues exactly where the snapshot's run was: --total-timesteps and
--learning-rate are rewritten so that the fork's first iteration uses the snapshot iteration's rate and
the linear anneal keeps the original slope. The run stops after CLEANRL_FORK_STEPS environment steps.

One update from a snapshot cannot see what compounds across updates (the data the policy collects next,
the critic co-adapting, Adam's state, entropy drift). Forks of several trainers from one snapshot, all
with the same seed, compare those dynamics from a matched state. Run it through mlq.
"""

import importlib
import inspect
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Arguments a fork must share with the snapshot's run for the continuation to be the same problem.
MATCHED_ARGS = ("env_id", "num_envs", "num_steps", "gamma", "gae_lambda", "num_minibatches")


class ForkComplete(Exception):
    """Raised through the trainer once the fork has run its steps."""


def rewrite_argv(argv, snapshot):
    """Replace --total-timesteps/--learning-rate so the anneal continues from the snapshot's iteration."""
    args = snapshot["args"]
    remaining = args["num_iterations"] - snapshot["iteration"] + 1
    horizon_steps = args["total_timesteps"] - args["num_iterations"] * args["batch_size"]
    kept = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token in ("--total-timesteps", "--learning-rate", "--anneal-lr", "--no-anneal-lr"):
            skip = token in ("--total-timesteps", "--learning-rate")
            continue
        if token.startswith(("--total-timesteps=", "--learning-rate=")):
            continue
        kept.append(token)
    if args["anneal_lr"]:
        # The trainer's schedule: lr(i) = (1 - (i - 1)/N) lr0, i = 1..N.
        learning_rate = (1.0 - (snapshot["iteration"] - 1.0) / args["num_iterations"]) * args["learning_rate"]
        schedule = ["--anneal-lr"]
    else:
        learning_rate, schedule = args["learning_rate"], ["--no-anneal-lr"]
    # num_iterations = (total - horizon * num_envs) // batch_size must come out to `remaining`.
    total = remaining * args["batch_size"] + horizon_steps
    return kept + ["--total-timesteps", str(total), "--learning-rate", repr(learning_rate), *schedule]


def load_state(local, snapshot, device):
    agent, optimizer = local["agent"], local["optimizer"]
    agent.load_state_dict({key: value.to(device) for key, value in snapshot["model"].items()}, strict=True)
    saved = snapshot["optimizer_state"]
    parameters = dict(agent.named_parameters())
    if set(saved) != set(parameters):
        raise ValueError(f"optimizer state covers {sorted(set(saved) ^ set(parameters))[:4]}... differently")
    for name, param in parameters.items():
        optimizer.state[param] = {key: value.to(device=param.device) for key, value in saved[name].items()}
    # The native kernels hold these arrays' addresses: copy in place, never rebind. The running
    # discounted returns belong to the snapshot's episodes, which a fork does not continue.
    for name, value in snapshot["obs_norm"].items():
        getattr(local["obs_norm"], name)[...] = value
    for name, value in snapshot["rew_norm"].items():
        if name != "returns":
            getattr(local["rew_norm"], name)[...] = value
    local["host_actor"].refresh()


def main():
    trainer_module = os.environ["CLEANRL_PROBE_MODULE"]
    snapshot_path = Path(os.environ["CLEANRL_FORK_SNAPSHOT"])
    fork_steps = int(float(os.environ["CLEANRL_FORK_STEPS"]))
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    if snapshot.get("format_version", 0) < 3:
        raise ValueError(f"{snapshot_path}: no full optimizer state; recapture with scripts/awr_capture_policy.py")
    sys.argv = [sys.argv[0], *rewrite_argv(sys.argv[1:], snapshot)]
    trainer = importlib.import_module(trainer_module)
    original_warmup, original_minibatches = trainer.run_phase_warmup, trainer.device_minibatches
    loaded = False

    def main_locals():
        frame = inspect.currentframe().f_back.f_back
        try:
            if frame is None or frame.f_code is not trainer.main.__code__:
                raise RuntimeError("the hook was not called directly by trainer.main")
            return frame.f_locals
        finally:
            del frame

    def forked_warmup(*args, **kwargs):
        nonlocal loaded
        local = main_locals()
        for name in MATCHED_ARGS:
            if getattr(local["args"], name) != snapshot["args"][name]:
                raise ValueError(f"fork {name}={getattr(local['args'], name)} != snapshot's {snapshot['args'][name]}")
        load_state(local, snapshot, next(local["agent"].parameters()).device)
        local["writer"].add_text("fork", f"snapshot {snapshot_path} (step {snapshot['global_step']}, "
                                         f"{snapshot['trainer_module']}) under {trainer_module}")
        print(f"FORK from {snapshot_path} step={snapshot['global_step']} iteration={snapshot['iteration']} "
              f"lr={local['args'].learning_rate:.3g}", flush=True)
        loaded = True
        return original_warmup(*args, **kwargs)

    def bounded_minibatches(*args, **kwargs):
        local = main_locals()
        if not loaded:
            raise RuntimeError("the trainer updated before its phase warmup: the fork was never loaded")
        if local["global_step"] >= fork_steps and local["updates"] == 0:
            raise ForkComplete
        yield from original_minibatches(*args, **kwargs)

    trainer.run_phase_warmup, trainer.device_minibatches = forked_warmup, bounded_minibatches
    try:
        trainer.main()
    except ForkComplete:
        print(f"FORK complete after {fork_steps} steps", flush=True)
    finally:
        trainer.run_phase_warmup, trainer.device_minibatches = original_warmup, original_minibatches
    if not loaded:
        raise RuntimeError("the trainer never ran its phase warmup: nothing was forked")


if __name__ == "__main__":
    main()
