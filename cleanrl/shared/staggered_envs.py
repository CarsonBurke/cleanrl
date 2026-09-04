"""Phase-staggered (offset-start) parallel environments.

STANDARD: all new continuous-control versions that roll out with ``num_envs``
parallel environments MUST establish staggered episode phases through this
module instead of hand-rolled warmup loops. Do not retrofit frozen files.

Problem
-------
``SyncVectorEnv`` steps every sub-environment in lockstep. When all envs reset
together, equal episode horizons keep their ages synchronous forever: every
rollout then covers one narrow slice of episode time, timeouts land on the
same step, and truncation bootstraps correlate across the batch. The rollout
behaves like ``num_steps`` correlated samples instead of
``num_steps * num_envs`` covering all episode ages -- the reason small batches
generalize poorly.

Solution (V-MPO v9 design, FAMILY.md)
------------------------------------
Spend one episode horizon on unrecorded stochastic warmup before the first
recorded rollout. Deterministically space environment ages at
``horizon / num_envs`` and permute the env-to-phase assignment with a
seed-local RNG. Scheduled single-env resets establish those phases. After
warmup, envs run continuously (autoreset only) so every rollout spans all
episode ages uniformly. Warmup transitions count against the total budget::

    from cleanrl.shared.staggered_envs import (
        compute_phase_offsets,
        episode_horizon,
        run_phase_warmup,
    )

    horizon = episode_horizon(args.env_id)          # e.g. 1000, from gym spec
    phase_offsets = compute_phase_offsets(args.num_envs, horizon, args.seed)
    writer.add_text("initial_phase_offsets", ",".join(map(str, phase_offsets)))
    warm = run_phase_warmup(
        envs, obs_norm=obs_norm, act_fn=sample_warmup_action,
        horizon=horizon, phase_offsets=phase_offsets,
        seed=args.seed, rew_norm=rew_norm,  # None when rewards stay raw
    )
    next_obs_np = warm.next_obs
    global_step = warm.transitions                    # num_envs * horizon
    suppress_next_episode_log = warm.suppress_mask
    args.num_iterations = (args.total_timesteps - warm.transitions) // batch

Contracts (do not "improve" without renaming):
- ``act_fn`` maps normalized float32 ``(N, *obs)`` to action ``(N, *act)``
  arrays. Warmup actions must be stochastic draws from the initial behavior
  policy, never greedy means: greedy warmup collapses the phase spread the
  warmup exists to create.
- Raw rewards flow through ``rew_norm`` (burns in return stats) but the
  normalized values are discarded; ``RecordEpisodeStatistics`` still sees raw
  rewards so reported returns stay in environment units.
- Resets land at step ``horizon - offset`` per env, including offset 0
  (reset on the final warmup step); subsequent horizons are unchanged.
  ``suppress_mask`` is true exactly for nonzero-offset envs: their
  phase-aligned episode started mid-warmup at the scheduled reset, so its
  first completion still contains warmup steps and must be skipped once.
  Offset-0 episodes start at the recording boundary and are logged.
- ``compute_phase_offsets`` uses an isolated RNG: global ``np.random`` state
  is untouched, so seeding downstream sampling is unaffected.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass
class PhaseWarmupResult:
    next_obs: np.ndarray  # normalized float32 (N, *obs): first recorded obs
    transitions: int  # num_envs * horizon, charged against the total budget
    suppress_mask: np.ndarray  # bool (N,): skip one completion log where True
    phase_offsets: np.ndarray  # int (N,): established episode ages


def episode_horizon(env_id) -> int:
    """Finite ``max_episode_steps`` for ``env_id``; staggering needs a horizon."""
    spec = gym.spec(env_id)
    if spec.max_episode_steps is None:
        raise ValueError("phase staggering requires a finite episode horizon")
    return int(spec.max_episode_steps)


def compute_phase_offsets(num_envs, horizon, seed) -> np.ndarray:
    """Evenly spaced ages ``arange(N) * horizon // N``, permuted seed-locally."""
    offsets = np.arange(num_envs, dtype=np.int64) * horizon // num_envs
    np.random.default_rng(seed).shuffle(offsets)
    return offsets


def _default_single_reset(envs, index):
    obs, _ = envs.envs[index].reset()
    return obs


def run_phase_warmup(
    envs,
    *,
    obs_norm,
    act_fn,
    horizon,
    phase_offsets,
    seed,
    rew_norm=None,
    single_reset=None,
) -> PhaseWarmupResult:
    """Run one unrecorded horizon; return the recording-ready state."""
    num_envs = len(phase_offsets)
    reset_at = {}
    for i, offset in enumerate(phase_offsets):
        reset_at.setdefault(int(horizon - int(offset)), []).append(i)
    if single_reset is None:
        single_reset = lambda index: _default_single_reset(envs, index)  # noqa: E731

    raw_next_obs, _ = envs.reset(seed=seed)
    next_obs = obs_norm.normalize(raw_next_obs)

    for warmup_step in range(1, int(horizon) + 1):
        raw_next_obs, raw_rew, terms, truncs, infos = envs.step(act_fn(next_obs))
        if rew_norm is not None:
            rew_norm.normalize(raw_rew, terms)
        next_obs, _ = obs_norm.normalize_step(raw_next_obs, terms, truncs, infos)
        for i in reset_at.get(warmup_step, ()):
            reset_obs = np.asarray(single_reset(i))
            next_obs[i] = obs_norm.normalize(reset_obs[None, ...], rows=slice(i, i + 1))[0]

    return PhaseWarmupResult(
        next_obs=next_obs,
        transitions=num_envs * int(horizon),
        suppress_mask=np.asarray(phase_offsets) != 0,
        phase_offsets=np.asarray(phase_offsets, dtype=np.int64),
    )
