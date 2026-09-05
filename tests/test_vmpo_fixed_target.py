"""CPU-only source contracts for v54's sole algorithm change: fixed target hold.

Execute the actual scheduling statements with scalar stand-ins; importing the
trainer or constructing a model, environment, optimizer, or CUDA context is not
needed. Numerical learner equivalence is protected by AST identity with v53.
"""

import ast
from contextlib import nullcontext
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_DIRECTORY = Path(__file__).resolve().parents[1] / "cleanrl" / "vmpo"
_PREFIX = "ppo_continuous_action_iterthink_v24_beta_vmpo_"
_REFERENCE = _DIRECTORY / f"{_PREFIX}v53_no_percentile_scaling.py"
_CANDIDATE = _DIRECTORY / f"{_PREFIX}v54_fixed_target_hold.py"


def _one(items):
    items = list(items)
    assert len(items) == 1, "source contract changed; review extraction"
    return items[0]


def _tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def _definition(tree, name):
    return _one(node for node in ast.walk(tree)
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)


def _loop(tree):
    return _one(node for node in ast.walk(tree)
                if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and node.target.id == "iteration")


def _assigns(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets)


def _schedule(tree):
    """Require the complete age/promotion block outside the logging branch."""
    body = _loop(tree).body
    start = _one(index for index, node in enumerate(body)
                 if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
                 and node.target.id == "target_age_batches")
    stop = _one(index for index, node in enumerate(body)
                if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == "target_promoted")
    assert start < stop
    return body[start:stop + 1]


def _compile(statements):
    return compile(ast.Module(body=statements, type_ignores=[]), str(_CANDIDATE), "exec")


@pytest.mark.parametrize("name", [
    "layer_init", "ReLUSquared", "branch_body", "FusedExperts", "ThinkBlock",
    "ThinkTrunk", "Agent", "beta_log_prob", "beta_kl", "decoupled_beta_kl",
    "rollout_model", "update_loss_model",
])
def test_v54_model_and_loss_are_ast_identical_to_v53(name):
    assert ast.dump(_definition(_tree(_CANDIDATE), name)) == ast.dump(
        _definition(_tree(_REFERENCE), name))


def test_v54_optimizer_configuration_is_identical_and_not_reinitialized_in_loop():
    reference, candidate = _tree(_REFERENCE), _tree(_CANDIDATE)
    ref_optimizer = _one(node for node in ast.walk(reference) if _assigns(node, "optimizer"))
    new_optimizer = _one(node for node in ast.walk(candidate) if _assigns(node, "optimizer"))
    assert ast.dump(new_optimizer) == ast.dump(ref_optimizer)
    assert not any(_assigns(node, "optimizer") for node in ast.walk(_loop(candidate)))
    schedule = ast.Module(body=_schedule(candidate), type_ignores=[])
    assert not any(isinstance(node, ast.Name) and node.id == "optimizer" for node in ast.walk(schedule))
    copies = [node for node in ast.walk(_loop(candidate))
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and isinstance(node.func.value, ast.Name)
              and node.func.value.id == "target_agent" and node.func.attr == "load_state_dict"]
    assert len(copies) == 1 and copies[0] in list(ast.walk(schedule))
    step_calls = [node for node in ast.walk(_loop(candidate))
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Name)
                  and node.func.value.id == "optimizer" and node.func.attr == "step"]
    assert len(step_calls) == 1


class _ScalarModel:
    def __init__(self):
        self.value = 0
        self.copies = []

    def state_dict(self):
        return {"learner_update": self.value}

    def load_state_dict(self, state):
        self.value = state["learner_update"]
        self.copies.append(self.value)


@pytest.mark.parametrize("log_interval", [1, 7, 10, 333])
@pytest.mark.parametrize("mean_kl", [0.0, 1e9, float("nan")])
def test_v54_promotes_after_exactly_100_updates_independent_of_kl_and_logging(log_interval, mean_kl):
    tree = _tree(_CANDIDATE)
    schedule = _compile(_schedule(tree))
    log_assignment = _one(node for node in _loop(tree).body if _assigns(node, "should_log"))
    assert _loop(tree).body.index(log_assignment) < _loop(tree).body.index(_schedule(tree)[0])
    log_decision = _compile([log_assignment])
    source, target = _ScalarModel(), _ScalarModel()
    # Scalar optimizer stand-in: promotion must preserve these existing moments.
    optimizer = SimpleNamespace(state={"step": 0, "exp_avg": [0.2], "exp_avg_sq": [0.3]})
    namespace = dict(
        args=SimpleNamespace(target_update_period=100, log_interval=log_interval,
                             epsilon_alpha_mean=0.007071067811865476),
        agent=source, target_agent=target, optimizer=optimizer,
        torch=SimpleNamespace(no_grad=nullcontext),
        target_age_batches=0, target_promotions=0, mean_kl_value=mean_kl,
    )
    for update in range(1, 351):
        source.value = update
        optimizer.state["step"] = update
        before_state = copy.deepcopy(optimizer.state)
        namespace.update(iteration=update, should_log=update % log_interval == 0 or update == 1)
        # Match source order: the upcoming promotion is known before the update,
        # so its logging decision can request the pre-update dual snapshot.
        exec(log_decision, namespace)
        exec(schedule, namespace)
        promoted = update % 100 == 0
        assert namespace["target_promoted"] is promoted
        assert namespace["promote_for_mean_kl"] is False
        assert namespace["target_age_batches"] == update % 100
        assert namespace["target_promotions"] == update // 100
        assert target.value == update // 100 * 100
        assert optimizer.state == before_state
        assert namespace["optimizer"] is optimizer
        # Promotion events remain visible even when they are not regular logs.
        assert namespace["should_log"] == (promoted or update % log_interval == 0 or update == 1)
    assert target.copies == [100, 200, 300]


def test_v54_target_period_default_and_validation_are_logging_independent():
    tree = _tree(_CANDIDATE)
    defaults = {node.target.id: ast.literal_eval(node.value)
                for node in _definition(tree, "Args").body
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id in {"target_update_period", "log_interval"}}
    assert defaults["target_update_period"] == 100
    # v53's divisibility guard must not survive: interval 7 is valid for hold 100.
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and any(isinstance(child, ast.Raise) for child in ast.walk(node)):
            names = {child.attr for child in ast.walk(node.test) if isinstance(child, ast.Attribute)}
            assert not {"target_update_period", "log_interval"} <= names


def test_v53_warmup_and_gae_match_the_validated_v30_migration_reference():
    v30 = _tree(_DIRECTORY / f"{_PREFIX}v30_dreamer_bucket_moment_hlgauss_reward_norm.py")
    v53 = _tree(_REFERENCE)
    assert ast.dump(_definition(v53, "gae_model")) == ast.dump(_definition(v30, "gae_model"))
    def warmup(tree):
        return _one(node for node in ast.walk(tree)
                    if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                    and node.target.id == "warmup_step")

    assert ast.dump(warmup(v53)) == ast.dump(warmup(v30))


def test_shared_normalization_matches_actual_v53_functions_at_mixed_boundaries():
    """Synthetic NumPy inputs only: no simulator, preprocessing job, or model."""
    from cleanrl.shared.vector_norm import VectorObsNorm, VectorRewardNorm

    reference = _tree(_REFERENCE)
    namespace = {"np": np}
    exec(_compile([_definition(reference, name) for name in (
        "normalize_observations", "normalize_rewards", "normalize_vector_step",
    )]), namespace)
    rng = np.random.default_rng(1)
    obs_norm, reward_norm = VectorObsNorm(4, (3,)), VectorRewardNorm(4, 0.99)
    obs_state = [obs_norm.means.copy(), obs_norm.variances.copy(), obs_norm.counts.copy()]
    reward_state = [reward_norm.returns.copy(), reward_norm.means.copy(),
                    reward_norm.variances.copy(), reward_norm.counts.copy()]
    initial = rng.normal(size=(4, 3))
    expected = namespace["normalize_observations"](initial, *obs_state)
    np.testing.assert_array_equal(obs_norm.normalize(initial), expected.astype(np.float32))
    for step in range(80):
        observations = rng.normal(size=(4, 3)) * 3
        rewards = rng.normal(size=4) * 7
        terms = np.array([step % 3 == 0, False, step % 7 == 0, False])
        truncs = np.array([False, step % 5 == 0, step % 11 == 0, False])
        final_observations = np.empty(4, dtype=object)
        final_observations[:] = None
        for row in np.flatnonzero(terms | truncs):
            final_observations[row] = rng.normal(size=3) * 17
        infos = {"final_observation": final_observations, "_final_observation": terms | truncs}
        expected_reward = namespace["normalize_rewards"](rewards, terms, *reward_state, 0.99)
        expected_obs = namespace["normalize_vector_step"](observations, terms, truncs, infos, *obs_state)
        np.testing.assert_array_equal(reward_norm.normalize(rewards, terms), expected_reward.astype(np.float32))
        actual_obs = obs_norm.normalize_step(observations, terms, truncs, infos)
        for actual, expected in zip(actual_obs, expected_obs):
            np.testing.assert_array_equal(actual, expected.astype(np.float32))
        for actual, expected in zip((obs_norm.means, obs_norm.variances, obs_norm.counts), obs_state):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip((reward_norm.returns, reward_norm.means,
                                     reward_norm.variances, reward_norm.counts), reward_state):
            np.testing.assert_array_equal(actual, expected)
