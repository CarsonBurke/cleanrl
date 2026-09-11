"""Capture full, unmodified trainer updates; all CLI arguments belong to the trainer."""

import importlib
import inspect
import os
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_TRAINER_MODULE = "cleanrl.ppo_continuous_action_32xlr_1mb_noadvnorm_normres_twohot_indclip_v4"
CAPTURE_STEPS = (2_000_000, 4_500_000, 5_500_000, 8_000_000, 16_000_000, 24_000_000, 32_000_000, 40_000_000, 48_000_000)
BATCH_LOCALS = {
    "observations": "b_obs",
    "native_actions": "b_native",
    "old_logprobs": "b_logprobs",
    "advantages": "b_advantages",
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
    trainer_module = os.environ.get("CLEANRL_PROBE_MODULE", DEFAULT_TRAINER_MODULE)
    trainer = importlib.import_module(trainer_module)
    original_minibatches = trainer.device_minibatches
    snapshot: dict[str, Any] = {}
    fixed_inputs: dict[str, torch.Tensor] = {}
    captured_count = 0

    def capture_minibatches(*args, **kwargs):
        nonlocal captured_count
        # On first resume the caller is the trainer's for loop, not the factory.
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
            global_step = int(main_locals["global_step"])
            if snapshot and global_step != snapshot["global_step"]:
                raise RuntimeError("Captured update ended before all optimizer steps completed")
            if not snapshot:
                if (
                    captured_count == len(CAPTURE_STEPS)
                    or global_step < CAPTURE_STEPS[captured_count]
                    or main_locals["updates"] != 0
                ):
                    yield from original_minibatches(*args, **kwargs)
                    return
                fixed_inputs.update({key: main_locals[name] for key, name in BATCH_LOCALS.items()})
                fixed_inputs["targets"] = main_locals.get("b_targets", main_locals["b_returns"])
                fixed_inputs["rewards"] = main_locals["batch"].rewards.flatten()
                snapshot.update(
                    {
                        "format_version": 1,
                        "trainer_module": trainer_module,
                        "args": cpu_owned(vars(main_locals["args"])),
                        "global_step": global_step,
                        "model": cpu_owned(main_locals["agent"].state_dict()),
                        "optimizer": cpu_owned(main_locals["optimizer"].state_dict()),
                        "batch": cpu_owned(fixed_inputs),
                        "minibatch_indices": [],
                    }
                )
            for indices in original_minibatches(*args, **kwargs):
                snapshot["minibatch_indices"].append(cpu_owned(indices))
                yield indices
            # Refresh after main has executed every yielded loss/backward/Adam step.
            main_locals = main_frame.f_locals
            if main_locals["updates"] == main_locals["max_updates"]:
                if len(snapshot["minibatch_indices"]) != main_locals["max_updates"]:
                    raise RuntimeError("Captured minibatch count does not match the complete update")
                snapshot["post_model"] = cpu_owned(main_locals["agent"].state_dict())
                snapshot["post_optimizer"] = cpu_owned(main_locals["optimizer"].state_dict())
                snapshot["immutable_inputs"] = {
                    key: bitwise_unchanged(snapshot["batch"][key], tensor) for key, tensor in fixed_inputs.items()
                }
                path = Path("runs") / main_locals["run_name"] / f"critic_probe_{global_step}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(snapshot, path)
                changed = [key for key, unchanged in snapshot["immutable_inputs"].items() if not unchanged]
                print(
                    f"CAPTURE step={global_step} requested={CAPTURE_STEPS[captured_count]} "
                    f"updates={main_locals['updates']} "
                    f"immutable={len(fixed_inputs) - len(changed)}/{len(fixed_inputs)} path={path}",
                    flush=True,
                )
                captured_count += 1
                snapshot.clear()
                fixed_inputs.clear()
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
        snapshot.clear()
        fixed_inputs.clear()
    if captured_count != len(CAPTURE_STEPS):
        raise RuntimeError(
            f"Trainer finished after {captured_count}/{len(CAPTURE_STEPS)} complete captures; "
            f"missing requested steps {CAPTURE_STEPS[captured_count:]}"
        )


if __name__ == "__main__":
    main()
