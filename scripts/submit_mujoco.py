"""Submit CUDA MuJoCo trainers with explicit machine-wide resource settings.

This entrypoint opts into the standard N16 / 8M / seed-1 experiment geometry;
it does not change the trainer's algorithm defaults. Novel scripts stay at one
concurrent job until measured. Extra trainer options follow ``--``.
"""

import argparse
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = Path("cleanrl/ppo_continuous_action.py")


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, default=BASE_SCRIPT)
    parser.add_argument("--name", help="queue and TensorBoard experiment name; defaults to script stem")
    parser.add_argument("--env-id", default="HalfCheetah-v4")
    parser.add_argument("--num-envs", type=positive, default=16)
    parser.add_argument("--total-timesteps", type=positive, default=8_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env-threads", type=positive, default=2)
    parser.add_argument("--max-parallel-runs", type=positive,
                        help="default 6 for base N16/2-thread PPO; 1 for uncharacterized configurations")
    parser.add_argument("--env-spin", type=int,
                        help="pause iterations before parking; defaults to 5000 at the standard operating point, else 0")
    parser.add_argument("--time-limit", default="2h")
    parser.add_argument("--after-success", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true", help="print the exact submission without queueing")
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    script = (ROOT / args.script).resolve()
    if not script.is_file():
        parser.error(f"trainer does not exist: {script}")
    extra = args.trainer_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    # The queue budget must describe the actual trainer invocation, not an
    # earlier flag silently overridden by a later duplicate.
    reserved = {"--num-envs", "--env-threads", "--seed", "--env-id", "--total-timesteps", "--exp-name"}
    for token in extra:
        if token.split("=", 1)[0].replace("_", "-") in reserved:
            parser.error(f"set {token.split('=', 1)[0]} before -- so resource settings remain consistent")
    standard = (script == (ROOT / BASE_SCRIPT).resolve() and args.num_envs == 16
                and args.env_threads == 2 and args.env_id in {"HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"}
                and not extra)
    limit = args.max_parallel_runs if args.max_parallel_runs is not None else (6 if standard else 1)
    spin = args.env_spin if args.env_spin is not None else (5000 if standard and limit <= 6 else 0)
    if spin < 0:
        parser.error("--env-spin must be nonnegative")
    name = args.name or script.stem
    command = ["mlq", "submit", "--name", name, "--max-parallel-runs", str(limit),
               "--time-limit", args.time_limit, "--cwd", str(ROOT),
               "--env", "OMP_NUM_THREADS=1", "--env", "MKL_NUM_THREADS=1",
               "--env", f"CLEANRL_ENV_SPIN={spin}"]
    for job in args.after_success:
        command.extend(["--after-success", job])
    command.extend(["--", str(ROOT / ".venv/bin/python"), "-u", str(script),
                    "--env-id", args.env_id, "--num-envs", str(args.num_envs),
                    "--exp-name", name, "--total-timesteps", str(args.total_timesteps),
                    "--seed", str(args.seed), "--env-threads", str(args.env_threads)])
    # Maintained trainers compile by default. Leave explicit compile-mode and
    # algorithm options to the trainer instead of injecting duplicate flags.
    command.extend(extra)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
