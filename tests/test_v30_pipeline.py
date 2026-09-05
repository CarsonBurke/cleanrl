"""Fixed-work CUDA equivalence replay, NOT a shortened training experiment.

Run through mlq only. Execute the actual frozen and proxy source blocks with
identical initial state and replayed sampling RNG: full 1,000-step phase warmup,
then two 39-step rollouts at N=16. The second rollout includes a time-limit
boundary and a nonzero critic, which zero-head synthetic learner tests miss.
No score, convergence or speed claim follows from this fixture.
"""

import ast
import copy
import importlib
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from cleanrl.shared.runtime import configure_runtime
from cleanrl_utils.reference_loss import load_reference_loss


def _one(items):
    items = list(items)
    assert len(items) == 1, "source fixture anchor changed; review extraction"
    return items[0]


def _assigns(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets
    )


def _iteration(nodes):
    return _one(node for node in nodes if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == "iteration")


def _blocks(reference, proxy):
    """Select existing statements, without rewriting their expressions."""
    ref_tree = ast.parse(Path(reference.__file__).read_text())
    ref_main = _one(node for node in ref_tree.body if isinstance(node, ast.If)
                    and "__name__" in ast.unparse(node.test)).body
    ref_loop = _iteration(ref_main)
    start = _one(i for i, node in enumerate(ref_main) if _assigns(node, "observation_means"))
    ref_gae = _one(i for i, node in enumerate(ref_loop.body) if isinstance(node, ast.With)
                   and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                           and n.func.id == "gae_model" for n in ast.walk(node)))
    ref_log = _one(i for i, node in enumerate(ref_loop.body) if isinstance(node, ast.If)
                   and isinstance(node.test, ast.Name) and node.test.id == "should_log")
    proxy_tree = ast.parse(Path(proxy.__file__).read_text())
    proxy_main = _one(node for node in proxy_tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "main")
    proxy_body = _one(node for node in proxy_main.body if isinstance(node, ast.Try)).body
    proxy_start = _one(i for i, node in enumerate(proxy_body) if isinstance(node, ast.FunctionDef)
                       and node.name == "rollout_model")
    proxy_loop = _iteration(proxy_body)
    proxy_update = _one(i for i, node in enumerate(proxy_loop.body) if isinstance(node, ast.With)
                        and any(isinstance(item.context_expr, ast.Call)
                                and isinstance(item.context_expr.func, ast.Attribute)
                                and item.context_expr.func.attr == "span"
                                and item.context_expr.args
                                and isinstance(item.context_expr.args[0], ast.Constant)
                                and item.context_expr.args[0].value == "update"
                                for item in node.items))
    return {
        "reference_setup": ref_main[start:ref_main.index(ref_loop)],
        "reference_rollout": ref_loop.body[:ref_gae],
        "reference_gae": [ref_loop.body[ref_gae]],
        "reference_update": ref_loop.body[ref_gae + 1:ref_log],
        "proxy_setup": proxy_body[proxy_start:proxy_body.index(proxy_loop)],
        "proxy_gae": proxy_loop.body[1:proxy_update],
        "proxy_update": proxy_loop.body[proxy_update:proxy_update + 2],
    }


def _execute(statements, namespace, filename):
    exec(compile(ast.Module(body=statements, type_ignores=[]), filename, "exec"), namespace)


class _Writer:
    def __init__(self):
        self.episodes = []

    def add_text(self, *args, **kwargs):
        pass

    def add_scalar(self, tag, value, step):
        if tag.startswith("charts/episodic_"):
            self.episodes.append((tag, float(value), step))


class _ActionRecorder:
    def __init__(self, env):
        self.env = env
        self.actions = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, actions):
        self.actions.append(np.array(actions, copy=True))
        return self.env.step(actions)


def _close(expected, actual, *, exact=False, atol=1e-6, rtol=1e-6):
    assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0 if exact else atol,
                               rtol=0 if exact else rtol, check_dtype=True)


def _normalization(reference, candidate):
    for prefix, normalizer in (("observation", candidate["obs_norm"]),
                                ("reward_return", candidate["rew_norm"])):
        for field in ("means", "variances", "counts"):
            np.testing.assert_array_equal(reference[f"{prefix}_{field}"], getattr(normalizer, field))
    np.testing.assert_array_equal(reference["discounted_returns"], candidate["rew_norm"].returns)
    np.testing.assert_array_equal(reference["suppress_next_episode_log"], candidate["suppress"])
    assert reference["global_step"] == candidate["collector"].total_steps
    assert reference["writer"].episodes == candidate["writer"].episodes
    np.testing.assert_array_equal(reference["envs"].actions, candidate["envs"].actions)


def test_v30_source_fixture_anchors():
    """CPU-only: fail clearly if either production loop is reorganized."""
    proxy = importlib.import_module("scripts.train_mujoco_throughput")
    blocks = _blocks(proxy.reference, proxy)
    for name, statements in blocks.items():
        assert statements, name
        compile(ast.Module(body=statements, type_ignores=[]), name, "exec")
    assert isinstance(blocks["proxy_update"][-1], ast.AugAssign)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="queued CUDA integration required")
def test_v30_full_phase_rollout_and_two_updates_match():
    proxy = importlib.import_module("scripts.train_mujoco_throughput")
    reference = proxy.reference
    blocks = _blocks(reference, proxy)
    args = proxy.Args(num_envs=16, seed=1)
    proxy.validate(args)
    assert args.initial_phase_warmup_steps == 1000 and args.num_steps == 39
    assert args.total_timesteps == 8_000_000 and args.target_update_period > 2
    assert not getattr(args, "fused_projection", False) and not getattr(args, "fused_temperature", False)
    configure_runtime(cudnn_deterministic=args.torch_deterministic)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    envs = [
        _ActionRecorder(gym.vector.SyncVectorEnv([
            reference.make_env(args.env_id, i, False, "v30_equivalence") for i in range(args.num_envs)
        ])),
        _ActionRecorder(proxy.make_mujoco_vector_env(args.env_id, args.num_envs,
                                                    backend="native", num_threads=4)),
    ]
    candidate = None
    try:
        model = reference.Agent(envs[0], args).to(device)
        models = [model, copy.deepcopy(model)]
        spaces = []
        limit = float(np.log1p(args.value_support_limit))
        for module, agent, env in zip((reference, proxy), models, envs):
            target = copy.deepcopy(agent).requires_grad_(False)
            duals = torch.nn.Parameter(torch.tensor(
                [args.initial_alpha_mean, args.initial_alpha_concentration], device=device))
            support = proxy.Dreamer3BucketHLGaussSupport(
                args.num_value_bins, -limit, limit, args.value_sigma_to_bin_ratio, device)
            optimizer = torch.optim.Adam([*agent.parameters(), duals], lr=args.learning_rate,
                                          betas=(0.9, 0.999), eps=1e-8, fused=True)
            namespace = dict(vars(module), args=args, agent=agent, target=target,
                             target_agent=target, duals=duals, support=support, hl_support=support,
                             optimizer=optimizer, envs=env, device=device, writer=_Writer(),
                             autocast_dtype=torch.bfloat16,
                             observation_shape=env.single_observation_space.shape,
                             action_shape=env.single_action_space.shape,
                             warmup_transitions=args.num_envs * args.initial_phase_warmup_steps,
                             return_percentile_levels=torch.tensor(
                                 [args.return_percentile_low, args.return_percentile_high], device=device))
            raw_loss, _ = load_reference_loss(reference, namespace)
            namespace["raw_loss"] = raw_loss
            spaces.append(namespace)
        baseline, candidate = spaces
        for name in ("rollout_model", "gae_model", "update_loss_model"):
            fn, _ = load_reference_loss(reference, baseline, name=name)
            baseline[name] = torch.compile(fn, mode=args.compile_mode, fullgraph=True, dynamic=False)

        before_warmup = torch.cuda.get_rng_state()
        print("v30 parity: reference full phase warmup", flush=True)
        _execute(blocks["reference_setup"], baseline, reference.__file__)
        after_warmup = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(before_warmup)
        print("v30 parity: candidate full phase warmup", flush=True)
        _execute(blocks["proxy_setup"], candidate, proxy.__file__)
        assert torch.equal(after_warmup, torch.cuda.get_rng_state()), "warmup sampling RNG changed"
        _normalization(baseline, candidate)
        np.testing.assert_array_equal(baseline["phase_offsets"], candidate["offsets"])
        _close(baseline["next_observation"], candidate["collector"].next_observation, exact=True)
        truncations = 0
        for iteration in (1, 2):
            print(f"v30 parity: rollout {iteration}", flush=True)
            baseline["iteration"] = candidate["iteration"] = iteration
            before_rollout = torch.cuda.get_rng_state()
            _execute(blocks["reference_rollout"], baseline, reference.__file__)
            after_rollout = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(before_rollout)
            batch = candidate["collector"].collect()
            candidate["batch"] = batch
            assert torch.equal(after_rollout, torch.cuda.get_rng_state()), "rollout sampling RNG changed"
            _normalization(baseline, candidate)
            pairs = {
                "observations": batch.observations,
                "native_actions": batch.policy["native_action"],
                "old_alphas": batch.policy["alpha"], "old_betas": batch.policy["beta"],
                "rewards": batch.transitions.rewards,
                "terminations_buffer": batch.transitions.terminations,
                "boundaries_buffer": batch.transitions.terminations.bool() | batch.transitions.truncations.bool(),
                "next_observations": batch.transitions.transition_observations,
                "next_observation": batch.next_observation,
            }
            for name, actual in pairs.items():
                _close(baseline[name], actual, exact=True)
            _close(baseline["values"], batch.policy["value"])
            truncations += int(batch.transitions.truncations.sum())
            if iteration == 2:
                assert torch.count_nonzero(batch.policy["value"]) > 0, "zero head hides critic differences"

            _execute(blocks["reference_gae"], baseline, reference.__file__)
            _execute(blocks["proxy_gae"], candidate, proxy.__file__)
            _close(baseline["value_targets"], candidate["targets"])
            _close(baseline["advantages"], candidate["advantages"])
            _close(baseline["return_percentile_scale"], candidate["scale"])
            _execute(blocks["reference_update"], baseline, reference.__file__)
            # Release the reference autograd result before the next learner; retained
            # comparison values are detached, never live parameter snapshots.
            expected_loss = baseline.pop("total_loss").detach().clone()
            expected_metrics = baseline.pop("metrics").detach().clone()
            print(f"v30 parity: candidate update {iteration}", flush=True)
            _execute(blocks["proxy_update"], candidate, proxy.__file__)
            _close(expected_loss, candidate["loss"])
            _close(expected_metrics, candidate["metrics"])
            assert torch.equal(after_rollout, torch.cuda.get_rng_state()), "update consumed RNG"
            assert baseline["target_age_batches"] == candidate["age"] == iteration
            baseline_params = [*models[0].parameters(), baseline["duals"]]
            candidate_params = [*models[1].parameters(), candidate["duals"]]
            for expected, actual in zip(baseline_params, candidate_params):
                _close(expected.detach(), actual.detach(), atol=1e-7)
                expected_state = baseline["optimizer"].state[expected]
                actual_state = candidate["optimizer"].state[actual]
                assert expected_state.keys() == actual_state.keys()
                for key in expected_state:
                    _close(expected_state[key], actual_state[key], atol=1e-7)
            for expected, actual in zip(baseline["target_agent"].parameters(), candidate["target"].parameters()):
                _close(expected, actual, exact=True)
        assert truncations > 0, "fixture must exercise final-observation truncation bootstrapping"
        assert candidate["collector"].total_steps == 16 * (1000 + 2 * 39)
    finally:
        if candidate is not None and "collector" in candidate:
            candidate["collector"].close()
        else:
            envs[1].close()
        envs[0].close()
