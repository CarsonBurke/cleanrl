"""Native full-horizon lifetimes for independent recurrent arithmetic programs."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence

import numpy as np
import torch

from cleanrl.collective_control.control import ObservationAdapter
from cleanrl.collective_control.lifetime import calibrate_adapter

from .genome import Genome
from .policy import ProgramController

HORIZON = 1000
OBSERVATION_CONTRACT = {
    "physics": "HalfCheetah-v4 native raw observation, components 0:17",
    "reward": "component 17: previous RAW native reward; zero at every physical reset",
    "boundary": "component 18: one only for the first action of each physical episode",
    "encoding": "asinh((input-mean)/scale), exactly once; reward/boundary mean=0, scale=1",
    "previous_action": "own signed nominal action, held fixed across internal ticks; zero at physical reset",
    "hidden": "no gain, task label, or executed post-gain action is a policy input",
}
TASK_CONTRACT = {
    "env_id": "HalfCheetah-v4",
    "horizon": HORIZON,
    "nominal": "fresh 1000-step query only, gain=1, germline initial state",
    "positive": "1000-step support then independent-reset 1000-step query; shared positive hidden gain",
    "gain_distribution": "independent Uniform[0.5,1] per positive lifetime",
    "action": "clamp nominal output to [-1,1], then multiply by hidden gain",
    "fitness": "unmodified native query return; any active nonfinite/undefined computation makes lifetime ineligible",
    "state": "germline between lives; retained support-to-query except reset/recent interventions",
    "donor": "support gain=1.5-query_gain, query gain unchanged",
    "same_task": "same gain; deterministic independently namespaced support reset seed",
    "recent": "germline reset then replay last 8 actual support inputs and previous nominal actions, no physics",
    "no_reward": "zero support reward inputs only, query feedback unchanged",
    "diagnostic_caveat": "interventions alone do not establish learning; phase, feedback and recent-transition confounds remain",
}
_INTERVENTIONS = {"intact", "reset", "donor", "recent", "no_reward", "same_task"}
_PROFILES = {"nominal", "positive"}


def _integer(value: Any, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its permitted range")
    return value


@dataclass
class ProgramSuite:
    support_seeds: list[int]
    query_seeds: list[int]
    gains: np.ndarray

    def __post_init__(self) -> None:
        self.support_seeds = list(self.support_seeds)
        self.query_seeds = list(self.query_seeds)
        raw = np.asarray(self.gains)
        if raw.dtype.kind not in "fiu":
            raise ValueError("gains must be real numbers")
        self.gains = raw.astype(np.float64, copy=True)
        self.validate()
        self.support_seeds = [int(seed) for seed in self.support_seeds]
        self.query_seeds = [int(seed) for seed in self.query_seeds]

    def validate(self, profile: str | None = None) -> None:
        if profile is not None and profile not in _PROFILES:
            raise ValueError(f"unknown profile: {profile}")
        count = len(self.support_seeds)
        gains = np.asarray(self.gains)
        if count == 0 or len(self.query_seeds) != count or gains.shape != (count,):
            raise ValueError("suite needs equal positive support/query/gain counts")
        if gains.dtype.kind not in "fiu" or not np.all(np.isfinite(gains)):
            raise ValueError("gains must be real and finite")
        if np.any(gains < 0.5) or np.any(gains > 1):
            raise ValueError("gains must be positive and in [0.5,1]")
        if profile == "nominal" and not np.all(gains == 1):
            raise ValueError("nominal gains must equal one")
        for seed in (*self.support_seeds, *self.query_seeds):
            _integer(seed, "reset seed", maximum=int(np.iinfo(np.uint32).max))

    def to_json(self) -> dict[str, Any]:
        self.validate()
        return {"support_seeds": self.support_seeds.copy(), "query_seeds": self.query_seeds.copy(),
                "gains": self.gains.tolist()}

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ProgramSuite":
        return cls(value["support_seeds"], value["query_seeds"], value["gains"])


def make_suite(seed: int, namespace: int, generation: int, count: int,
               profile: str = "nominal") -> ProgramSuite:
    """Independent gain/reset streams preserve every prefix when count changes."""
    if profile not in _PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    entropy = [_integer(seed, "seed"), _integer(namespace, "namespace"),
               _integer(generation, "generation")]
    count = _integer(count, "count", minimum=1)
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(entropy).spawn(3)]
    support = streams[0].integers(0, 2**32, count, dtype=np.uint32).tolist()
    query = streams[1].integers(0, 2**32, count, dtype=np.uint32).tolist()
    gains = streams[2].uniform(0.5, 1.0, count) if profile == "positive" else np.ones(count)
    return ProgramSuite(support, query, gains)


def _same_task_seeds(seeds: Sequence[int]) -> list[int]:
    result = []
    for seed in seeds:
        derived = int(np.random.SeedSequence([seed, 0x5A4E7A5C]).generate_state(1, dtype=np.uint32)[0])
        # Keep the intervention a genuinely different physical reset even in a hash collision.
        result.append(derived if derived != seed else (derived + 1) % 2**32)
    return result


@dataclass
class ProgramResult:
    support_returns: np.ndarray
    query_returns: np.ndarray
    support_valid: np.ndarray
    query_valid: np.ndarray
    first_actions: np.ndarray
    query_blocks: np.ndarray
    action_square_sum: np.ndarray
    metadata: dict[str, Any]

    @property
    def scores(self) -> np.ndarray:
        return np.where(self.support_valid & self.query_valid, self.query_returns, -np.inf)


class ProgramEvaluator:
    """CUDA policies with an LRU of at most four native population-major batches.

    Validity is read from the device once per physical phase. Actuator zeros from
    dead programs keep native batched physics well-defined, never restore fitness.
    Nominal results have zero support returns and vacuously true support validity.
    Work counters exclude graph capture and distinguish physical steps from replay.
    """

    def __init__(self, adapter: ObservationAdapter, max_nodes: int, ticks: int,
                 device: torch.device, num_threads: int = 8):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("program policies require CUDA; CPU fallback is disabled")
        self.max_nodes = _integer(max_nodes, "max_nodes", minimum=1)
        self.ticks = _integer(ticks, "ticks", minimum=1)
        self.num_threads = _integer(num_threads, "num_threads", minimum=1)
        mean, scale = np.asarray(adapter.mean), np.asarray(adapter.scale)
        if (mean.shape != (19,) or scale.shape != (19,) or mean.dtype.kind not in "fiu"
                or scale.dtype.kind not in "fiu" or not np.all(np.isfinite(mean))
                or not np.all(np.isfinite(scale)) or np.any(scale <= 0)):
            raise ValueError("adapter requires 19 finite means and positive finite scales")
        if not np.array_equal(mean[-2:], [0, 0]) or not np.array_equal(scale[-2:], [1, 1]):
            raise ValueError("reward and boundary adapter entries must have mean zero and scale one")
        self.adapter = ObservationAdapter(mean.astype(np.float32, copy=True), scale.astype(np.float32, copy=True))
        if (not np.all(np.isfinite(self.adapter.mean)) or not np.all(np.isfinite(self.adapter.scale))
                or np.any(self.adapter.scale <= 0)):
            raise ValueError("adapter must be representable in float32")
        self._slots: OrderedDict[tuple[int, int], tuple[Any, ProgramController]] = OrderedDict()
        self.evaluation_transitions = 0
        self.logical_node_updates = 0
        self.capacity_node_updates = 0
        self.replay_transitions = 0
        self.replay_node_updates = 0

    def __enter__(self) -> "ProgramEvaluator":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        while self._slots:
            _, (env, _) = self._slots.popitem(last=False)
            env.close()

    @torch.inference_mode()
    def evaluate(self, genomes: Sequence[Genome], suite: ProgramSuite,
                 profile: str = "nominal", intervention: str = "intact") -> ProgramResult:
        from cleanrl.shared.mujoco_env import make_mujoco_vector_env

        if intervention not in _INTERVENTIONS:
            raise ValueError(f"unknown intervention: {intervention}")
        suite.validate(profile)
        if profile == "nominal" and intervention != "intact":
            raise ValueError("support interventions require the positive profile")
        if not genomes:
            raise ValueError("evaluation needs at least one genome")
        for genome in genomes:
            if not isinstance(genome, Genome):
                raise ValueError("each candidate must be one Genome")
            genome.validate(19, 6, self.max_nodes)
        population, lives = len(genomes), len(suite.query_seeds)
        lanes = population * lives
        logical_per_step = sum(len(genome.nodes) for genome in genomes) * lives * self.ticks
        capacity_per_step = lanes * self.max_nodes * self.ticks
        key = population, lives
        if key not in self._slots:
            if len(self._slots) == 4:
                _, (evicted_env, _) = self._slots.popitem(last=False)
                evicted_env.close()
            env = make_mujoco_vector_env("HalfCheetah-v4", lanes, backend="native",
                                         num_threads=self.num_threads, copy=False)
            try:
                if (env.single_observation_space.shape != (17,) or env.single_action_space.shape != (6,)
                        or not np.all(env.single_action_space.low == -1)
                        or not np.all(env.single_action_space.high == 1)):
                    raise ValueError("native HalfCheetah observation/action contract changed")
                controller = ProgramController(list(genomes), self.adapter, self.max_nodes, self.ticks, self.device)
            except BaseException:
                env.close()
                raise
            self._slots[key] = env, controller
        else:
            env, controller = self._slots[key]
            controller.reload(list(genomes))
            self._slots.move_to_end(key)
        indices = np.repeat(np.arange(population), lives).tolist()
        controller.reset(lanes, indices)
        gains = np.tile(suite.gains, population).astype(np.float32)[:, None]
        support_gains = 1.5 - gains if intervention == "donor" else gains
        support_seeds = _same_task_seeds(suite.support_seeds) if intervention == "same_task" else suite.support_seeds
        inputs = np.empty((lanes, 19), dtype=np.float32)
        executed = np.empty((lanes, 6), dtype=np.float32)
        action_squares = np.empty((lanes, 6), dtype=np.float64)
        action_square_sum = np.zeros(lanes, dtype=np.float64)
        support_returns = np.zeros(lanes, dtype=np.float64)
        query_returns = np.zeros(lanes, dtype=np.float64)
        support_valid = np.ones(lanes, dtype=bool)
        query_blocks = np.zeros((lanes, 10), dtype=np.float64)
        first_actions = np.empty((lanes, 6), dtype=np.float32)
        recent_inputs = np.empty((8, lanes, 19), dtype=np.float32) if intervention == "recent" else None
        recent_previous = torch.empty((8, lanes, 6), device=self.device) if intervention == "recent" else None

        for query in ((True,) if profile == "nominal" else (False, True)):
            if query and profile == "positive":
                if intervention in {"reset", "recent"}:
                    lifetime_valid = controller.valid.clone()
                    controller.reset(lanes, indices)
                    controller.valid.logical_and_(lifetime_valid)
                if intervention == "recent":
                    for index in range(8):
                        controller.previous_action.copy_(recent_previous[index])
                        controller._action_buffer(recent_inputs[index])
                        self.replay_transitions += lanes
                        self.replay_node_updates += logical_per_step
            seeds = suite.query_seeds if query else support_seeds
            observation, _ = env.reset(seed=np.tile(seeds, population).tolist())
            controller.previous_action.zero_()
            inputs[:, 17] = 0.0
            inputs[:, 18] = 1.0
            episode_gains = gains if query else support_gains
            returns = query_returns if query else support_returns
            for step in range(HORIZON):
                if np.shape(observation) != (lanes, 17) or not np.all(np.isfinite(observation)):
                    raise RuntimeError("native HalfCheetah produced invalid observations")
                inputs[:, :17] = observation
                if not query and intervention == "recent" and step >= HORIZON - 8:
                    index = step - (HORIZON - 8)
                    recent_inputs[index] = inputs
                    recent_previous[index].copy_(controller.previous_action)
                action = controller._action_buffer(inputs)
                if query:
                    if step == 0:
                        first_actions[:] = action
                    np.square(action, out=action_squares, dtype=np.float64)
                    action_square_sum += action_squares.sum(axis=1)
                np.multiply(action, episode_gains, out=executed)
                observation, reward, terminated, truncated, _ = env.step(executed)
                self.evaluation_transitions += lanes
                self.logical_node_updates += logical_per_step
                self.capacity_node_updates += capacity_per_step
                if (np.shape(reward) != (lanes,) or not np.all(np.isfinite(reward))
                        or np.shape(terminated) != (lanes,) or np.shape(truncated) != (lanes,)):
                    raise RuntimeError("native HalfCheetah produced invalid rewards or boundary flags")
                if np.any(terminated) or not np.all(np.asarray(truncated) == (step == HORIZON - 1)):
                    raise RuntimeError("HalfCheetah must never terminate and must truncate exactly at step 1000")
                returns += reward
                if query:
                    query_blocks[:, step // 100] += reward
                inputs[:, 17] = 0.0 if not query and intervention == "no_reward" else reward
                inputs[:, 18] = 0.0
            phase_valid = controller.valid.cpu().numpy().copy()
            if query:
                query_valid = phase_valid
            else:
                support_valid = phase_valid
        metadata = {
            "profile": profile,
            "intervention": intervention,
            "actual_support_seeds": list(support_seeds) if profile == "positive" else [],
            "actual_support_gains": support_gains[:lives, 0].tolist() if profile == "positive" else [],
            "query_seeds": list(suite.query_seeds),
            "query_gains": gains[:lives, 0].tolist(),
        }
        shape = population, lives
        return ProgramResult(support_returns.reshape(shape), query_returns.reshape(shape),
                             support_valid.reshape(shape), query_valid.reshape(shape),
                             first_actions.reshape(population, lives, 6),
                             query_blocks.reshape(population, lives, 10),
                             action_square_sum.reshape(shape), metadata)
