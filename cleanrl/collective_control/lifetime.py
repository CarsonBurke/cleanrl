"""Paired support/query lifetimes for one coherent recurrent organism.

Interventions are diagnostics, not proofs of learning: phase, recent-transition,
and feedback confounds remain possible. Query returns alone are fitness; support
physics and optional recent-input replay are separately accounted computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from numbers import Integral
from typing import Any, Sequence

import numpy as np
import torch

from cleanrl.shared.rollout_graph import RolloutStepGraph

from .control import CollectiveController, ObservationAdapter, calibrate_observations
from .genome import Genome

HORIZON = 1000
OBSERVATION_CONTRACT = {
    "physics": "HalfCheetah-v4 native raw observation, 17 components",
    "reward": "component 17: asinh(previous raw reward); zero at each physical reset",
    "boundary": "component 18: 1 only on the first action of each physical episode",
    "encoding": "0.5 + 0.5*tanh((input-mean)/scale); reward/boundary mean=0, scale=1",
    "previous_action": "own nominal normalized proposal; reset to 0.5 at every physical reset",
    "hidden": "actuator gain and executed post-gain action are never policy inputs",
}
TASK_CONTRACT = {
    "env_id": "HalfCheetah-v4",
    "horizon": HORIZON,
    "lifetime": "one full support then one independent-reset full query; one shared hidden scalar gain",
    "gain_distribution": "independent random sign times Uniform[0.5,1] per lifetime",
    "action": "hidden gain multiplies nominal action in native [-1,1] action coordinates",
    "fitness": "query native raw return only; support fully counted as physics computation",
    "state": "germline reset at lifetime start, retained across support/query in intact",
    "diagnostic_caveat": "interventions do not establish learning; phase, recent-transition, and feedback confounds",
}
_INTERVENTIONS = {"intact", "reset", "donor", "recent", "no_reward"}


def _integer(value: Any, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its permitted range")
    return value


@dataclass
class LifetimeSuite:
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

    def validate(self) -> None:
        count = len(self.support_seeds)
        if count == 0 or len(self.query_seeds) != count or np.shape(self.gains) != (count,):
            raise ValueError("suite needs equal positive support/query/gain counts")
        gains = np.asarray(self.gains)
        if gains.dtype.kind not in "fiu" or not np.all(np.isfinite(gains)):
            raise ValueError("gains must be real and finite")
        if np.any(gains == 0) or np.any(np.abs(gains) > 1):
            raise ValueError("gains must be nonzero with absolute value at most one")
        for seed in (*self.support_seeds, *self.query_seeds):
            _integer(seed, "reset seed", maximum=int(np.iinfo(np.uint32).max))

    def to_json(self) -> dict[str, Any]:
        self.validate()
        return {"support_seeds": self.support_seeds.copy(), "query_seeds": self.query_seeds.copy(),
                "gains": self.gains.tolist()}

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "LifetimeSuite":
        return cls(value["support_seeds"], value["query_seeds"], value["gains"])


def make_suite(seed: int, namespace: int, generation: int, count: int) -> LifetimeSuite:
    """Use independent streams, not sign balancing or cross-lifetime conditioning."""
    entropy = [_integer(seed, "seed"), _integer(namespace, "namespace"),
               _integer(generation, "generation")]
    count = _integer(count, "count", minimum=1)
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(entropy).spawn(4)]
    support = streams[0].integers(0, 2**32, count, dtype=np.uint32).tolist()
    query = streams[1].integers(0, 2**32, count, dtype=np.uint32).tolist()
    signs = 2 * streams[2].integers(0, 2, count) - 1
    gains = signs * streams[3].uniform(0.5, 1.0, count)
    return LifetimeSuite(support, query, gains)


def calibrate_adapter(seed: int, num_threads: int) -> ObservationAdapter:
    seed = _integer(seed, "seed")
    num_threads = _integer(num_threads, "num_threads", minimum=1)
    seeds = np.random.SeedSequence([seed, 0xADA9719]).generate_state(4, dtype=np.uint32).tolist()
    physics = calibrate_observations("HalfCheetah-v4", seeds, 128, num_threads)
    if physics.mean.shape != (17,) or physics.scale.shape != (17,):
        raise ValueError("HalfCheetah calibration must have 17 observation dimensions")
    return ObservationAdapter(np.concatenate((physics.mean, np.zeros(2, dtype=np.float32))),
                              np.concatenate((physics.scale, np.ones(2, dtype=np.float32))))


@dataclass
class LifetimeResult:
    support_returns: np.ndarray
    query_returns: np.ndarray
    first_actions: np.ndarray
    query_blocks: np.ndarray

    @property
    def scores(self) -> np.ndarray:
        return self.query_returns


@torch.inference_mode()
def _capture_preserving_state(controller: CollectiveController) -> None:
    """Capture can execute recurrence; it must be invisible to experimental state."""
    if controller._graph is not None:
        return
    state = controller.state.clone()
    previous = controller.previous_action.clone()
    try:
        controller._graph = RolloutStepGraph(controller._policy, 1, controller.state.shape[0],
                                            (19,), controller.device)
    finally:
        controller.state.copy_(state)
        controller.previous_action.copy_(previous)


class LifetimeEvaluator:
    """CUDA-only controller and native physics cache keyed by (population, lives).

    `first_actions` are nominal query actions, before the hidden gain. `recent`
    replays actual support inputs but does not replay physics: a positive effect
    can reflect recent-transition reconstruction, not an acquired learning rule.
    `reset` retains phase confounds and `no_reward` changes feedback as well as
    support trajectories. `donor` changes only the support actuator gain.
    """

    def __init__(self, adapter: ObservationAdapter, max_nodes: int, device: torch.device,
                 num_threads: int = 8):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("lifetime policies require CUDA; CPU fallback is disabled")
        self.max_nodes = _integer(max_nodes, "max_nodes", minimum=1)
        self.num_threads = _integer(num_threads, "num_threads", minimum=1)
        mean, scale = np.asarray(adapter.mean), np.asarray(adapter.scale)
        if (mean.shape != (19,) or scale.shape != (19,) or mean.dtype.kind not in "fiu"
                or scale.dtype.kind not in "fiu" or not np.all(np.isfinite(mean))
                or not np.all(np.isfinite(scale)) or np.any(scale <= 0)):
            raise ValueError("adapter requires 19 finite means and positive finite scales")
        if not np.array_equal(mean[-2:], [0, 0]) or not np.array_equal(scale[-2:], [1, 1]):
            raise ValueError("reward and boundary adapter entries must have mean zero and scale one")
        self.adapter = ObservationAdapter(mean.astype(np.float32, copy=True), scale.astype(np.float32, copy=True))
        if not np.all(np.isfinite(self.adapter.mean)) or not np.all(np.isfinite(self.adapter.scale)) or np.any(self.adapter.scale <= 0):
            raise ValueError("adapter must be representable in float32")
        self._slots: OrderedDict[tuple[int, int], tuple[Any, CollectiveController]] = OrderedDict()
        self.evaluation_transitions = 0
        self.replay_transitions = 0
        # Node work excludes capture warmup; physics and replay are distinct.
        self.node_updates = 0
        self.replay_node_updates = 0

    def __enter__(self) -> "LifetimeEvaluator":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        for env, _ in self._slots.values():
            env.close()
        self._slots.clear()

    def _validate_genomes(self, genomes: Sequence[Genome]) -> None:
        if len(genomes) == 0:
            raise ValueError("evaluation needs at least one genome")
        for genome in genomes:
            if not isinstance(genome, Genome):
                raise ValueError("each candidate must be one Genome, not an ensemble")
            genome.validate(6, self.max_nodes, False)
            _integer(genome.next_id, "next node identity")
            for output in genome.outputs:
                _integer(output, "output identity")
            if genome.authority is not None:
                raise ValueError("a single organism has no authority weighting")
            for node in genome.nodes:
                _integer(node.node_id, "node identity")
                if len(node.sources) != 2:
                    raise ValueError("nodes need exactly two sources")
                for source in node.sources:
                    kind = _integer(source.kind, "source kind", maximum=4)
                    index = _integer(source.index, "source index")
                    if (kind == 0 and index >= 19) or (kind == 2 and index >= 6):
                        raise ValueError("source index exceeds observation/action interface")

    @torch.inference_mode()
    def evaluate(self, genomes: Sequence[Genome], suite: LifetimeSuite,
                 intervention: str = "intact") -> LifetimeResult:
        from cleanrl.shared.mujoco_env import make_mujoco_vector_env

        if intervention not in _INTERVENTIONS:
            raise ValueError(f"unknown intervention: {intervention}")
        suite.validate()
        self._validate_genomes(genomes)
        population, lives = len(genomes), len(suite.support_seeds)
        lanes = population * lives
        updates_per_tick = sum(len(genome.nodes) for genome in genomes) * lives
        key = population, lives
        teams = [[genome] for genome in genomes]
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
                controller = CollectiveController(teams, self.adapter, env.single_action_space.low,
                                                  env.single_action_space.high, self.max_nodes,
                                                  "uniform", self.device)
            except BaseException:
                env.close()
                raise
            self._slots[key] = env, controller
        else:
            env, controller = self._slots[key]
            controller.reload(teams)
            self._slots.move_to_end(key)
        indices = np.repeat(np.arange(population), lives).tolist()
        controller.reset(lanes, indices)
        _capture_preserving_state(controller)
        gains = np.tile(suite.gains, population).astype(np.float32)[:, None]
        support_gains = -gains if intervention == "donor" else gains
        inputs = np.empty((lanes, 19), dtype=np.float32)
        executed = np.empty((lanes, 6), dtype=np.float32)
        support_returns = np.zeros(lanes, dtype=np.float64)
        query_returns = np.zeros(lanes, dtype=np.float64)
        query_blocks = np.zeros((lanes, 10), dtype=np.float64)
        first_actions = np.empty((lanes, 6), dtype=np.float32)
        recent_inputs = np.empty((8, lanes, 19), dtype=np.float32) if intervention == "recent" else None
        recent_previous = torch.empty((8, lanes, 1, 6), device=self.device) if intervention == "recent" else None

        for query in (False, True):
            if query:
                if intervention in {"reset", "recent"}:
                    controller.reset(lanes, indices)
                if intervention == "recent":
                    for index in range(8):
                        controller.previous_action.copy_(recent_previous[index])
                        controller._action_buffer(recent_inputs[index])
                        self.replay_transitions += lanes
                        self.replay_node_updates += updates_per_tick
            # Reset physical state independently of retained/restored recurrence.
            seeds = suite.query_seeds if query else suite.support_seeds
            observation, _ = env.reset(seed=np.tile(seeds, population).tolist())
            controller.previous_action.fill_(0.5)
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
                if query and step == 0:
                    first_actions[:] = action
                np.multiply(action, episode_gains, out=executed)
                observation, reward, terminated, truncated, _ = env.step(executed)
                self.evaluation_transitions += lanes
                self.node_updates += updates_per_tick
                if (np.shape(reward) != (lanes,) or not np.all(np.isfinite(reward))
                        or np.shape(terminated) != (lanes,) or np.shape(truncated) != (lanes,)):
                    raise RuntimeError("native HalfCheetah produced invalid rewards or boundary flags")
                if np.any(terminated) or not np.all(np.asarray(truncated) == (step == HORIZON - 1)):
                    raise RuntimeError("HalfCheetah must never terminate and must truncate exactly at step 1000")
                returns += reward
                if query:
                    query_blocks[:, step // 100] += reward
                if not query and intervention == "no_reward":
                    inputs[:, 17] = 0.0
                else:
                    np.arcsinh(reward, out=inputs[:, 17])
                inputs[:, 18] = 0.0
        return LifetimeResult(support_returns.reshape(population, lives), query_returns.reshape(population, lives),
                              first_actions.reshape(population, lives, 6), query_blocks.reshape(population, lives, 10))
