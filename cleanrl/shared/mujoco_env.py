"""Batched Gymnasium-v4 MuJoCo stepping with the original MuJoCo engine.

``make_mujoco_vector_env(..., backend="native", num_threads=4)`` replaces a
SyncVectorEnv without changing reset RNG, rewards, observations, action clipping,
or same-step autoreset. Native physics runs in one C call, spread across a
persistent worker pool whose idle threads park rather than spin (see
``mujoco_batch.c`` for why, and ``CLEANRL_ENV_SPIN`` to override); NumPy
batches the surrounding Gymnasium bookkeeping. There is no GPU physics
conversion or floating-point approximation.

The specialized path deliberately supports only the unmodified Gymnasium 0.29
HalfCheetah/Hopper/Walker2d v4 tasks. Rendering uses SyncVectorEnv. ``sync`` is
the reference; ``threaded`` retains the entire original Python wrapper stack.
Native builds once in the user's cache using the installed MuJoCo headers and
shared library. A C compiler with POSIX threads is required; failure is
explicit.
Canonical legacy normalization/clipping is batched over the original wrapper
states; arbitrary transforms retain their original per-environment callbacks.
"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import time
import warnings

import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import concatenate

from cleanrl.shared.legacy_normalization import CanonicalLegacyNormalization
from cleanrl.shared.vector_norm import make_raw_continuous_env, ufunc_clip


def _native_library():
    import mujoco

    package = Path(mujoco.__file__).resolve().parent
    libraries = sorted(package.glob("libmujoco.so.*"))
    if len(libraries) != 1:
        raise RuntimeError("native MuJoCo backend requires one installed libmujoco.so")
    source = Path(__file__).with_name("mujoco_batch.c")
    library = libraries[0]
    fingerprint = hashlib.sha256(
        source.read_bytes() + str(library).encode() + mujoco.__version__.encode()
    ).hexdigest()[:20]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    directory = cache / "cleanrl" / "mujoco-native" / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "mujoco_batch.so"
    if not output.exists():
        # Concurrent processes compile separate files and atomically publish.
        fd, temporary = tempfile.mkstemp(suffix=".so", dir=directory)
        os.close(fd)
        try:
            subprocess.run(
                ["cc", "-O3", "-std=c11", "-fPIC", "-shared", "-pthread",
                 "-I", str(package / "include"), str(source), str(library),
                 f"-Wl,-rpath,{package}", "-o", temporary],
                check=True, capture_output=True, text=True,
            )
            os.replace(temporary, output)
        except (OSError, subprocess.CalledProcessError) as error:
            detail = getattr(error, "stderr", str(error))
            raise RuntimeError(f"Unable to build native MuJoCo backend: {detail}") from error
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    lib = ctypes.CDLL(str(output))
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    pointers = ctypes.POINTER(ctypes.c_void_p)
    lib.cleanrl_pool_create.argtypes = [
        ctypes.c_int, pointers, pointers, doubles, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, doubles, doubles, doubles,
    ]
    lib.cleanrl_pool_create.restype = ctypes.c_void_p
    # One pointer argument per step: ctypes converts nothing else, and the
    # batch descriptor (models, data, buffers, frame_skip) is bound once.
    lib.cleanrl_pool_step.argtypes = [ctypes.c_void_p]
    lib.cleanrl_pool_step.restype = None
    lib.cleanrl_pool_destroy.argtypes = [ctypes.c_void_p]
    lib.cleanrl_pool_destroy.restype = None
    return lib


def _worker_spin_budget():
    """Pause iterations an idle physics worker burns before parking.

    Default 0: park immediately. Measured on a 12-core 9900X with three
    concurrent 16-env HalfCheetah rollouts, parking gave 161.7k aggregate SPS
    on 5.1 cores against 164.8k on 11.9 cores for an unbounded spin -- the same
    throughput within noise for 2.3x less CPU, which is what lets several runs
    share the machine (see scripts/benchmark_rollout_scale.py). A lone run on
    an idle box gains roughly 10% wall from spinning, so set CLEANRL_ENV_SPIN
    (in pause iterations) when a single run owns the machine.
    """
    value = os.environ.get("CLEANRL_ENV_SPIN")
    if value is None:
        return 0
    budget = int(value)
    if budget < 0:
        raise ValueError("CLEANRL_ENV_SPIN must be non-negative")
    return budget


class _ResetObserver(gym.Wrapper):
    """Keep batched episode state correct even for envs.envs[i].reset()."""

    callback = None

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if self.callback is not None:
            self.callback()
        return result


class ThreadedMujocoVectorEnv(gym.vector.SyncVectorEnv):
    """Reference wrapper semantics, with independent env steps on worker threads."""

    def __init__(self, env_fns, *, num_threads, copy=True):
        super().__init__(env_fns, copy=copy)
        self._executor = ThreadPoolExecutor(max_workers=num_threads)

    @staticmethod
    def _step_one(pair):
        env, action = pair
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            final_obs, final_info = obs, info
            obs, info = env.reset()
            info["final_observation"] = final_obs
            info["final_info"] = final_info
        return obs, reward, term, trunc, info

    def step_wait(self):
        infos = {}
        dense = isinstance(self.observations, np.ndarray)
        observations = None if dense else []
        results = self._executor.map(self._step_one, zip(self.envs, self._actions))
        for i, (obs, reward, term, trunc, info) in enumerate(results):
            if dense:
                self.observations[i] = obs
            else:
                observations.append(obs)
            self._rewards[i], self._terminateds[i], self._truncateds[i] = reward, term, trunc
            self._add_info(infos, info, i)
        if not dense:
            self.observations = concatenate(self.single_observation_space, observations, self.observations)
        return (deepcopy(self.observations) if self.copy else self.observations,
                self._rewards.copy(), self._terminateds.copy(),
                self._truncateds.copy(), infos)

    def close_extras(self, **kwargs):
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)
        super().close_extras(**kwargs)


class NativeMujocoVectorEnv(gym.vector.SyncVectorEnv):
    """Exact v4 task equations with batched native physics and bookkeeping.

    ``batch_legacy_normalization=False`` retains per-environment normalization
    calls for comparison. Otherwise canonical pure-clip pipelines batch their
    statistics math automatically; ``uses_batched_normalization`` reports which
    path is active, including fallback after runtime callback changes.
    """

    def __init__(self, env_fns, *, env_id=None, num_threads=1, copy=True,
                 batch_legacy_normalization=True):
        import mujoco
        from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
        from gymnasium.envs.mujoco.hopper_v4 import HopperEnv
        from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv

        if gym.__version__ != "0.29.1":
            raise ValueError("native task equations require Gymnasium 0.29.1; use backend='sync'")
        super().__init__([lambda fn=fn: _ResetObserver(fn()) for fn in env_fns], copy=copy)
        expected = {"HalfCheetah-v4": HalfCheetahEnv, "Hopper-v4": HopperEnv,
                    "Walker2d-v4": Walker2dEnv}
        self._bases = tuple(env.unwrapped for env in self.envs)
        if env_id is None:
            env_id = self._bases[0].spec.id
        if env_id not in expected or any(type(env) is not expected[env_id] for env in self._bases):
            self.close()
            raise ValueError("native backend supports standard HalfCheetah/Hopper/Walker2d-v4 only")
        first = self._bases[0]
        if (not isinstance(self.single_observation_space, gym.spaces.Box)
                or self.single_action_space != first.action_space):
            self.close()
            raise ValueError("native backend requires Box observations and the original action space")
        configuration = ("frame_skip", "dt", "_forward_reward_weight", "_ctrl_cost_weight",
                         "_exclude_current_positions_from_observation", "_healthy_reward",
                         "_terminate_when_unhealthy", "_healthy_z_range", "_healthy_angle_range",
                         "_healthy_state_range")
        self._postprocessors = []
        transformations = (gym.wrappers.NormalizeObservation, gym.wrappers.TransformObservation,
                           gym.wrappers.NormalizeReward, gym.wrappers.TransformReward)
        for outer, base in zip(self.envs, self._bases):
            wrapper = outer.env
            stack = []
            processors = []
            while isinstance(wrapper, gym.Wrapper):
                if any(key in wrapper.__dict__ for key in ("step", "action", "observation", "reward")):
                    self.close()
                    raise ValueError("native backend does not support modified wrapper methods")
                stack.append(type(wrapper))
                if type(wrapper) in transformations:
                    if getattr(wrapper, "is_vector_env", False):
                        self.close()
                        raise ValueError("native backend requires per-environment normalizers")
                    processors.append(wrapper)
                wrapper = wrapper.env
            prefix = 0
            while prefix < len(stack) and stack[prefix] in transformations:
                prefix += 1
            self._postprocessors.append(tuple(reversed(processors)))
            stack = stack[prefix:]
            required = [gym.wrappers.ClipAction, gym.wrappers.RecordEpisodeStatistics,
                        gym.wrappers.FlattenObservation, gym.wrappers.TimeLimit]
            suffixes = ([], [gym.wrappers.OrderEnforcing],
                        [gym.wrappers.PassiveEnvChecker],
                        [gym.wrappers.OrderEnforcing, gym.wrappers.PassiveEnvChecker])
            if stack[:4] != required or stack[4:] not in suffixes:
                self.close()
                raise ValueError("native backend requires the standard raw continuous-env wrapper stack")
            if (base.render_mode is not None or any(
                    getattr(base, key, None) != getattr(first, key, None) for key in configuration)
                    or (base.model.nq, base.model.nv, base.model.nu) !=
                    (first.model.nq, first.model.nv, first.model.nu)
                    or any(key in base.__dict__ for key in (
                        "step", "do_simulation", "_get_obs", "control_cost",
                        "_step_mujoco_simulation", "state_vector"))):
                self.close()
                raise ValueError("native backend requires homogeneous, unmodified v4 task configurations")
        # Python callbacks acquire the GIL and may not be safe inside native
        # parallel regions; fail explicitly instead of silently changing them.
        for name in ("mjcb_control", "mjcb_passive", "mjcb_sensor", "mjcb_contactfilter", "mjcb_time",
                     "mjcb_act_dyn", "mjcb_act_gain", "mjcb_act_bias"):
            getter = getattr(mujoco, "get_" + name, None)
            if getter is not None and getter() is not None:
                self.close()
                raise ValueError(f"native backend does not support installed {name} callbacks")
        self._native = _native_library()
        self.num_threads = int(num_threads)
        if self.num_threads < 1:
            self.close()
            raise ValueError("num_threads must be positive")
        self._env_id = env_id
        base = self._bases[0]
        self._models = (ctypes.c_void_p * self.num_envs)(*(env.model._address for env in self._bases))
        self._data = (ctypes.c_void_p * self.num_envs)(*(env.data._address for env in self._bases))
        self._positions = np.empty((self.num_envs, base.model.nq), dtype=np.float64)
        self._velocities = np.empty((self.num_envs, base.model.nv), dtype=np.float64)
        self._raw_observations = np.empty(
            (self.num_envs, base.model.nq + base.model.nv -
             int(base._exclude_current_positions_from_observation)), dtype=np.float64,
        )
        self._before = np.empty(self.num_envs, dtype=np.float64)
        # Clipped actions are needed twice per step: as float64 controls for
        # MuJoCo and in the policy's own dtype for the task's control cost.
        # Both buffers are reused so no step allocates.
        self._controls = np.empty((self.num_envs, base.model.nu), dtype=np.float64)
        self._clipped = {}
        self._cost_dtypes = {}
        self._pool = None
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self._episode_starts = np.zeros(self.num_envs, dtype=np.float32)
        self._statistics = []
        self._limits = []
        for i, outer in enumerate(self.envs):
            env = outer.env
            statistics = limit = None
            while isinstance(env, gym.Wrapper):
                if isinstance(env, gym.wrappers.RecordEpisodeStatistics):
                    statistics = env
                if isinstance(env, gym.wrappers.TimeLimit):
                    limit = env
                env = env.env
            if statistics is None or limit is None:
                self.close()
                raise ValueError("native backend requires episode statistics and a finite TimeLimit")
            self._statistics.append(statistics)
            self._limits.append(limit)
            outer.callback = lambda i=i: self._reset_statistics(i)
        self._horizons = np.asarray([limit._max_episode_steps for limit in self._limits])
        self._ready = False
        self._has_postprocessors = any(self._postprocessors)
        self._batched_normalization = (
            CanonicalLegacyNormalization.from_wrappers(self._postprocessors)
            if batch_legacy_normalization else None
        )
        self.uses_batched_normalization = self._batched_normalization is not None
        # The pool borrows these buffers for its lifetime, so they must never
        # be reallocated; every step writes controls in place and reads the
        # results back out of the same storage.
        self._pool = self._native.cleanrl_pool_create(
            self.num_envs, self._models, self._data, self._controls,
            base.frame_skip, self.num_threads, _worker_spin_budget(),
            self._before, self._positions, self._velocities,
        )
        if not self._pool:
            self.close()
            raise RuntimeError("Unable to start the native MuJoCo worker pool")

    def _reset_statistics(self, index):
        stats = self._statistics[index]
        self._episode_returns[index] = 0
        self._episode_lengths[index] = 0
        self._episode_starts[index] = stats.episode_start_times[0]
        # Preserve per-environment statistics inspection as views of the batch.
        stats.episode_returns = self._episode_returns[index:index + 1]
        stats.episode_lengths = self._episode_lengths[index:index + 1]
        stats.episode_start_times = self._episode_starts[index:index + 1]

    def reset_wait(self, seed=None, options=None):
        result = super().reset_wait(seed=seed, options=options)
        self._ready = True
        return result

    def reset_at(self, index, **kwargs):
        return self.envs[index].reset(**kwargs)

    def close_extras(self, **kwargs):
        # Workers must be joined before the sub-environments release their
        # mjData. Tolerates the constructor's early failure paths, which call
        # close() before the pool exists, and repeated close() calls.
        pool = getattr(self, "_pool", None)
        if pool:
            self._pool = None
            self._native.cleanrl_pool_destroy(pool)
        super().close_extras(**kwargs)

    def step_async(self, actions):
        actions = np.asarray(actions)
        if actions.shape != self.action_space.shape:
            raise ValueError(f"expected action shape {self.action_space.shape}, got {actions.shape}")
        # Clip into a buffer reused per input dtype: the control cost must keep
        # the policy's dtype (Gym's ClipAction does), so the dtype is part of
        # the observable contract and cannot be normalized away here.
        clipped = self._clipped.get(actions.dtype)
        if clipped is None:
            clipped = np.empty(self.action_space.shape, dtype=actions.dtype)
            self._clipped[actions.dtype] = clipped
        self._actions = ufunc_clip(actions, self.single_action_space.low,
                                   self.single_action_space.high, out=clipped)

    def _postprocess(self, index, observation, reward, terminated):
        """Run the original legacy normalizer/transform instances in stack order."""
        for wrapper in self._postprocessors[index]:
            if type(wrapper) is gym.wrappers.NormalizeObservation:
                observation = wrapper.normalize(np.array([observation]))[0]
            elif type(wrapper) is gym.wrappers.TransformObservation:
                observation = wrapper.observation(observation)
            elif type(wrapper) is gym.wrappers.NormalizeReward:
                batch_reward = np.array([reward])
                wrapper.returns = wrapper.returns * wrapper.gamma * (1 - terminated) + batch_reward
                reward = wrapper.normalize(batch_reward)[0]
            else:
                reward = wrapper.reward(reward)
        return observation, reward

    def step_wait(self):
        if not self._ready:
            raise gym.error.ResetNeeded("Call reset before stepping the environment")
        base = self._bases[0]
        actions = self._actions
        np.copyto(self._controls, actions)
        self._native.cleanrl_pool_step(self._pool)
        x = self._positions[:, 0]
        velocity = (x - self._before) / base.dt
        forward = base._forward_reward_weight * velocity
        # Keep the input action dtype through square/sum, as Gym's ClipAction
        # and task control_cost do (float32 policy actions must stay float32).
        costs = np.sum(np.square(actions), axis=1)
        # NumPy 1.x scalar multiplication promotes task np.float32 scalars to
        # float64; explicit promotion matches that behavior in batched form.
        # The result depends only on the two dtypes, so probe it once per dtype
        # instead of allocating a scalar array every step.
        cost_dtype = self._cost_dtypes.get(costs.dtype)
        if cost_dtype is None:
            cost_dtype = np.asarray(costs[0] * base._ctrl_cost_weight).dtype
            self._cost_dtypes[costs.dtype] = cost_dtype
        costs = costs.astype(cost_dtype, copy=False) * base._ctrl_cost_weight
        if self._env_id == "HalfCheetah-v4":
            rewards = forward - costs
            terms = np.zeros(self.num_envs, dtype=bool)
        else:
            z, angle = self._positions[:, 1], self._positions[:, 2]
            healthy = ((base._healthy_z_range[0] < z) & (z < base._healthy_z_range[1]) &
                       (base._healthy_angle_range[0] < angle) & (angle < base._healthy_angle_range[1]))
            if self._env_id == "Hopper-v4":
                low, high = base._healthy_state_range
                healthy &= np.all((low < self._positions[:, 2:]) & (self._positions[:, 2:] < high), axis=1)
                healthy &= np.all((low < self._velocities) & (self._velocities < high), axis=1)
            rewards = (forward + (healthy | base._terminate_when_unhealthy) * base._healthy_reward) - costs
            terms = ~healthy if base._terminate_when_unhealthy else np.zeros(self.num_envs, dtype=bool)
        offset = int(base._exclude_current_positions_from_observation)
        width = base.model.nq - offset
        self._raw_observations[:, :width] = self._positions[:, offset:]
        if self._env_id == "HalfCheetah-v4":
            self._raw_observations[:, width:] = self._velocities
        else:
            np.clip(self._velocities, -10, 10, out=self._raw_observations[:, width:])
        if not self._has_postprocessors:
            self.observations[...] = self._raw_observations
        # RecordEpisodeStatistics uses float32 += a scalar reward: preserve
        # its rounding order instead of accumulating rewards in float64.
        self._episode_returns += rewards.astype(np.float32)
        self._episode_lengths += 1
        for limit, elapsed in zip(self._limits, self._episode_lengths.tolist()):
            limit._elapsed_steps = elapsed
        truncs = self._episode_lengths >= self._horizons
        boundaries = terms | truncs
        active = ~boundaries
        infos = {}
        columns = {"x_position": x, "x_velocity": velocity}
        if self._env_id == "HalfCheetah-v4":
            columns.update(reward_run=forward, reward_ctrl=-costs)
        if active.any():
            for key, values in columns.items():
                infos[key] = np.where(active, values, 0.0)
                infos["_" + key] = active.copy()
        processed_observations = self._raw_observations
        if self._batched_normalization is not None:
            normalized = self._batched_normalization.normalize(
                self._raw_observations, rewards, terms
            )
            if normalized is None:
                self._batched_normalization = None
                self.uses_batched_normalization = False
            else:
                processed_observations, rewards = normalized
                self.observations[...] = processed_observations
        per_row_processing = self._has_postprocessors and self._batched_normalization is None
        processed_rows = range(self.num_envs) if per_row_processing else np.flatnonzero(boundaries)
        for i in processed_rows:
            # Legacy transforms can return a different dtype than the declared
            # space: preserve it in final_observation just as SyncVectorEnv does.
            observation = processed_observations[i].copy()
            if boundaries[i]:
                final_info = {key: values[i] for key, values in columns.items()}
                final_info["episode"] = {
                    "r": self._episode_returns[i:i + 1].copy(),
                    "l": self._episode_lengths[i:i + 1].copy(),
                    "t": np.round(time.perf_counter() - self._episode_starts[i:i + 1], 6),
                }
                # RecordEpisodeStatistics completes and clears its counters
                # before outer transforms run. A transform inspecting its
                # wrapper state must see the just-completed episode in queues.
                stats = self._statistics[i]
                # Gym indexes its one-element arrays with a scalar boolean,
                # then extends the queue: each entry is a shape-(1,) snapshot.
                stats.return_queue.append(self._episode_returns[i:i + 1].copy())
                stats.length_queue.append(self._episode_lengths[i:i + 1].copy())
                stats.episode_count += np.int64(1)
                self._episode_returns[i] = 0
                self._episode_lengths[i] = 0
                self._episode_starts[i] = time.perf_counter()
            if per_row_processing and self._postprocessors[i]:
                observation, rewards[i] = self._postprocess(i, observation, rewards[i], terms[i])
            self.observations[i] = observation
            if not boundaries[i]:
                continue
            reset_observation, reset_info = self.envs[i].reset()
            self.observations[i] = reset_observation
            reset_info["final_observation"] = observation
            reset_info["final_info"] = final_info
            self._add_info(infos, reset_info, i)
        return (self.observations.copy() if self.copy else self.observations,
                rewards, terms, truncs, infos)


def make_mujoco_vector_env(env_id, num_envs, *, backend="native", num_threads=None,
                           capture_video=False, run_name="", copy=True,
                           batch_legacy_normalization=True, **env_kwargs):
    """Construct shared vector envs; kwargs are forwarded unchanged to gym.make.

    Native defaults to one physics thread to avoid guessing GPU-job CPU budgets.
    Benchmark explicit thread counts for the actual env count and machine.
    Rendering always uses the original wrapper path, with an explicit warning.
    """
    if backend not in {"sync", "threaded", "native"}:
        raise ValueError(f"unknown MuJoCo vector backend: {backend!r}")
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    threads = 1 if num_threads is None else int(num_threads)
    if threads < 1:
        raise ValueError("num_threads must be positive")
    if capture_video or env_kwargs.get("render_mode") is not None:
        if backend != "sync":
            warnings.warn("Rendering uses the sync MuJoCo backend", stacklevel=2)
        backend = "sync"

    def thunk(index):
        if not env_kwargs:
            return make_raw_continuous_env(env_id, index, capture_video, run_name)

        def create():
            kwargs = dict(env_kwargs)
            if capture_video and index == 0:
                kwargs["render_mode"] = "rgb_array"
            env = gym.make(env_id, **kwargs)
            if capture_video and index == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            env = gym.wrappers.FlattenObservation(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return gym.wrappers.ClipAction(env)
        return create

    env_fns = [thunk(i) for i in range(num_envs)]
    if backend == "sync":
        return gym.vector.SyncVectorEnv(env_fns, copy=copy)
    if backend == "threaded":
        return ThreadedMujocoVectorEnv(env_fns, num_threads=threads, copy=copy)
    return NativeMujocoVectorEnv(
        env_fns, env_id=env_id, num_threads=threads, copy=copy,
        batch_legacy_normalization=batch_legacy_normalization,
    )
