"""Snapshot an unmodified trainer's rollout policy at chosen steps for scripts/awr_update_bench.py.

    python scripts/awr_capture_policy.py <the unchanged trainer arguments>

CLEANRL_PROBE_MODULE picks the trainer (default: the AWR-family PPO control) and
CLEANRL_CAPTURE_STEPS the comma-separated global steps (default 4M,12M,24M,40M).
Each snapshot is taken as an update begins, so the weights are the policy that
collected that batch, and the actor's Adam moments are those the update starts from; every parameter's Adam state
(critic included) is kept too, for scripts/awr_fork_policy.py. The run stops after the last snapshot. Run it through mlq.
"""

import importlib
import inspect
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_TRAINER_MODULE = "cleanrl.awr.ppo_continuous_action_ppo_sepclip_control_v1"
DEFAULT_CAPTURE_STEPS = "4000000,12000000,24000000,40000000"


class CaptureComplete(Exception):
    """Raised through the trainer once every requested snapshot is on disk."""


def cpu_owned(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: cpu_owned(item) for key, item in value.items()}
    return value


def main():
    trainer_module = os.environ.get("CLEANRL_PROBE_MODULE", DEFAULT_TRAINER_MODULE)
    capture_steps = [int(float(step)) for step in
                     os.environ.get("CLEANRL_CAPTURE_STEPS", DEFAULT_CAPTURE_STEPS).split(",")]
    trainer = importlib.import_module(trainer_module)
    original_minibatches = trainer.device_minibatches
    captured = 0

    def capture_minibatches(*args, **kwargs):
        nonlocal captured
        frame = inspect.currentframe()
        main_frame = frame.f_back if frame is not None else None
        del frame
        try:
            if main_frame is None or main_frame.f_code is not trainer.main.__code__:
                raise RuntimeError("device_minibatches was not resumed directly by trainer.main")
            local = main_frame.f_locals
            global_step = int(local["global_step"])
            if captured < len(capture_steps) and global_step >= capture_steps[captured] and local["updates"] == 0:
                obs_norm, rew_norm, agent = local["obs_norm"], local["rew_norm"], local["agent"]
                optimizer_state = local["optimizer"].state
                # The actor's real Adam moments as this update begins: the bench's step geometry.
                actor_adam = {name: cpu_owned(dict(optimizer_state[param]))
                              for name, param in agent.actor.named_parameters() if param in optimizer_state}
                # Every parameter's Adam state by qualified name, critic included: what a fork resumes from.
                optimizer_state = {name: cpu_owned(dict(optimizer_state[param]))
                                   for name, param in agent.named_parameters() if param in optimizer_state}
                snapshot = {
                    "format_version": 3,
                    "actor_adam": actor_adam,
                    "optimizer_state": optimizer_state,
                    "iteration": int(local["iteration"]),
                    "adam_hyperparameters": {key: value for key, value in local["optimizer"].param_groups[0].items()
                                             if key != "params"},
                    "trainer_module": trainer_module,
                    "args": dict(vars(local["args"])),
                    "global_step": global_step,
                    "learning_rate": float(local["optimizer"].param_groups[0]["lr"]),
                    "model": cpu_owned(agent.state_dict()),
                    "obs_norm": {name: getattr(obs_norm, name).copy() for name in ("means", "variances", "counts")},
                    "rew_norm": {name: getattr(rew_norm, name).copy()
                                 for name in ("returns", "means", "variances", "counts")},
                }
                path = Path("runs") / local["run_name"] / f"policy_snapshot_{global_step}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(snapshot, path)
                print(f"CAPTURE step={global_step} requested={capture_steps[captured]} path={path}", flush=True)
                captured += 1
                if captured == len(capture_steps):
                    raise CaptureComplete
        finally:
            del main_frame
        yield from original_minibatches(*args, **kwargs)

    trainer.device_minibatches = capture_minibatches
    try:
        trainer.main()
    except CaptureComplete:
        pass
    finally:
        trainer.device_minibatches = original_minibatches
    if captured != len(capture_steps):
        raise RuntimeError(f"trainer finished after {captured}/{len(capture_steps)} snapshots")


if __name__ == "__main__":
    main()
