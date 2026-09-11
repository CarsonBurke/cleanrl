"""Capture one unmodified trainer update; all CLI arguments belong to the trainer."""

import importlib
import inspect
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TRAINER_MODULE = "cleanrl.ppo_continuous_action_32xlr_1mb_noadvnorm_normres_twohot_indclip_v4"
BATCH_LOCALS = {
    "observations": "b_obs",
    "native_actions": "b_native",
    "old_logprobs": "b_logprobs",
    "advantages": "b_advantages",
    "targets": "b_targets",
    "returns": "b_returns",
    "old_values": "b_values",
}


def cpu_owned(value):
    """Detach tensors into owned CPU storage, including tensors already on CPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: cpu_owned(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_owned(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_owned(item) for item in value)
    return value


def bitwise_unchanged(before: torch.Tensor, current: torch.Tensor):
    if before.shape != current.shape or before.dtype != current.dtype:
        return False
    after = current.detach().to(device="cpu", copy=True)
    # Float equality would miss signed-zero changes and reject unchanged NaNs.
    return torch.equal(before.contiguous().view(torch.uint8), after.contiguous().view(torch.uint8))


def main():
    capture_step = int(os.environ.get("CLEANRL_CAPTURE_STEP", "7061120"))
    if capture_step <= 0:
        raise ValueError("CLEANRL_CAPTURE_STEP must be positive")
    trainer = importlib.import_module(TRAINER_MODULE)
    original_minibatches = trainer.device_minibatches
    snapshot = None
    fixed_inputs = None
    captured_path = None

    def capture_minibatches(*args, **kwargs):
        nonlocal snapshot, fixed_inputs, captured_path
        # This is a generator: its caller on first resume is the trainer's for loop,
        # not the wrapper's main() or the original minibatch factory.
        frame = inspect.currentframe()
        if frame is None:
            raise RuntimeError("Update capture requires Python frame inspection")
        main_frame = frame.f_back
        del frame
        main_locals = None
        try:
            if main_frame is None or main_frame.f_code is not trainer.main.__code__:
                raise RuntimeError("device_minibatches was not resumed directly by trainer.main")
            main_locals = main_frame.f_locals
            if captured_path is not None or main_locals["global_step"] != capture_step:
                yield from original_minibatches(*args, **kwargs)
                return
            if main_locals["updates"] == 0:
                if snapshot is not None:
                    raise RuntimeError("Requested capture update started more than once")
                fixed_inputs = {key: main_locals[name] for key, name in BATCH_LOCALS.items()}
                snapshot = {
                    "format_version": 1,
                    "trainer_module": TRAINER_MODULE,
                    "args": cpu_owned(vars(main_locals["args"])),
                    "global_step": int(main_locals["global_step"]),
                    "model": cpu_owned(main_locals["agent"].state_dict()),
                    "optimizer": cpu_owned(main_locals["optimizer"].state_dict()),
                    "batch": cpu_owned(fixed_inputs),
                    "minibatch_indices": [],
                }
            if snapshot is None or fixed_inputs is None:
                raise RuntimeError("Requested update was not captured before its first optimizer step")
            for indices in original_minibatches(*args, **kwargs):
                snapshot["minibatch_indices"].append(cpu_owned(indices))
                yield indices
            # Refresh after main has executed each yielded loss/backward/Adam step.
            main_locals = main_frame.f_locals
            if main_locals["updates"] == main_locals["max_updates"]:
                if len(snapshot["minibatch_indices"]) != main_locals["max_updates"]:
                    raise RuntimeError("Captured minibatch count does not match the complete update")
                snapshot["post_model"] = cpu_owned(main_locals["agent"].state_dict())
                snapshot["post_optimizer"] = cpu_owned(main_locals["optimizer"].state_dict())
                snapshot["immutable_inputs"] = {
                    key: bitwise_unchanged(snapshot["batch"][key], tensor) for key, tensor in fixed_inputs.items()
                }
                path = Path("runs") / main_locals["run_name"] / f"update_probe_{capture_step}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(snapshot, path)
                captured_path = path
                changed = [key for key, unchanged in snapshot["immutable_inputs"].items() if not unchanged]
                print(
                    f"CAPTURE step={capture_step} updates={main_locals['updates']} "
                    f"immutable={len(BATCH_LOCALS) - len(changed)}/{len(BATCH_LOCALS)} path={path}",
                    flush=True,
                )
                snapshot = None
                fixed_inputs = None
                if changed:
                    raise RuntimeError(f"Fixed update inputs changed; evidence saved to {path}: {', '.join(changed)}")
        finally:
            # Never retain a caller frame across generator completion or failure.
            del main_locals
            del main_frame

    setattr(trainer, "device_minibatches", capture_minibatches)
    try:
        trainer.main()
    finally:
        setattr(trainer, "device_minibatches", original_minibatches)
        snapshot = None
        fixed_inputs = None
    if captured_path is None:
        raise RuntimeError(f"Trainer finished without capturing a complete update at step {capture_step}")


if __name__ == "__main__":
    main()
