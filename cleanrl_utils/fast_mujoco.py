"""Opt existing single-file trainers into shared MuJoCo execution backends.

Run through mlq, for example::

    .venv/bin/python -m cleanrl_utils.fast_mujoco --threads 4 \
        cleanrl/vmpo/ppo_continuous_action_iterthink_v24_beta_vmpo_v30_dreamer_bucket_moment_hlgauss_reward_norm.py \
        --env-id HalfCheetah-v4 --seed 1 --total-timesteps 8000000

The trainer keeps its model, rollout, normalization, RNG and optimizer logic.
Only its Gymnasium SyncVectorEnv constructor is replaced, within this process.
Unsupported wrappers fail explicitly on the native path; use --backend threaded
to retain arbitrary Python wrappers, or --backend sync for the reference.
"""

import argparse
from contextlib import contextmanager
from pathlib import Path
import runpy
import sys

import gymnasium as gym
import torch

from cleanrl.shared.mujoco_env import NativeMujocoVectorEnv, ThreadedMujocoVectorEnv
from cleanrl.shared.runtime import configure_compile_cache


@contextmanager
def vector_backend(backend, threads):
    """Scope constructor substitution so importing this module changes nothing."""
    if backend not in {"sync", "threaded", "native"}:
        raise ValueError(f"unknown backend: {backend}")
    if threads < 1:
        raise ValueError("threads must be positive")
    if backend == "sync":
        yield
        return
    original = gym.vector.SyncVectorEnv

    def constructor(env_fns, observation_space=None, action_space=None, copy=True):
        if observation_space is not None or action_space is not None:
            raise ValueError("custom vector spaces require --backend sync")
        cls = NativeMujocoVectorEnv if backend == "native" else ThreadedMujocoVectorEnv
        return cls(env_fns, num_threads=threads, copy=copy)

    gym.vector.SyncVectorEnv = constructor
    try:
        yield
    finally:
        gym.vector.SyncVectorEnv = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sync", "threaded", "native"), default="native")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("script", type=Path)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.script.is_file():
        parser.error(f"training script not found: {args.script}")
    # Tiny CPU tensor operations should not compete with physics workers.
    # The original trainer continues to choose every precision/determinism flag.
    torch.set_num_threads(1)
    configure_compile_cache()
    old_argv = sys.argv
    old_path = sys.path.copy()
    sys.argv = [str(args.script), *args.script_args]
    sys.path.insert(0, str(args.script.resolve().parent))
    try:
        with vector_backend(args.backend, args.threads):
            runpy.run_path(str(args.script), run_name="__main__")
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path


if __name__ == "__main__":
    main()
