"""CPU-only validation of trusted source extraction and exact loop substitution."""

import ast
import hashlib
from pathlib import Path
import sys
import textwrap
from types import ModuleType, SimpleNamespace

import pytest

from cleanrl_utils.reference_loss import _V30_TEMPERATURE_LOOP, load_reference_loss


REFERENCE = Path(__file__).parents[1] / "cleanrl/vmpo/ppo_continuous_action_iterthink_v24_beta_vmpo_v30_dreamer_bucket_moment_hlgauss_reward_norm.py"


def make_module(tmp_path, source):
    path = tmp_path / "trusted_reference.py"
    path.write_text(source)
    module = ModuleType("trusted_reference")
    module.__file__ = str(path)
    return module


@pytest.fixture
def original_temperature_loop():
    tree = ast.parse(REFERENCE.read_text())
    function, = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == "update_loss_model"]
    loop, = [node for node in ast.walk(function) if isinstance(node, ast.For)]
    # This independently pins the checked-in reference's actual syntax tree.
    assert hashlib.sha256(ast.dump(loop, include_attributes=False).encode()).hexdigest() == _V30_TEMPERATURE_LOOP
    return ast.unparse(loop)


@pytest.fixture
def fake_solver(monkeypatch):
    calls = []
    result = object()

    def solve(*args):
        calls.append(args)
        return result

    module = ModuleType("cleanrl.shared.vmpo_temperature")
    module.solve_log_temperature = solve
    # Exercise import wiring without importing Triton, constructing CUDA
    # tensors, or replacing the real solver outside this test's lifetime.
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls, result


def temperature_source(loop):
    return (
        "raise AssertionError('module execution is forbidden')\n"
        "if __name__ == '__main__':\n"
        "    def update_loss_model(inputs):\n"
        "        centered_advantages, selected, log_selected_count, log_eta_low, log_eta_high = inputs\n"
        + textwrap.indent(loop, "        ") + "\n"
        "        return log_eta_high, {'after': after_loop()}\n"
    )


def test_extracts_main_guard_function_with_namespace_override_without_running_module(tmp_path):
    source = (
        "raise AssertionError('module executed')\n"
        "if __name__ == '__main__':\n"
        "    raise AssertionError('training started')\n"
        "    def update_loss_model(value):\n"
        "        return value + offset, {'offset': offset}\n"
    )
    module = make_module(tmp_path, source)
    module.offset = 3
    namespace = {"offset": 5}
    before = Path(module.__file__).read_bytes()
    function, digest = load_reference_loss(module, namespace)
    assert function(7) == (12, {"offset": 5})
    assert digest == hashlib.sha256(source.encode()).hexdigest()
    assert load_reference_loss(module, namespace)[1] == digest
    assert namespace == {"offset": 5}
    assert module.offset == 3 and not hasattr(module, "update_loss_model")
    assert Path(module.__file__).read_bytes() == before


def test_exact_temperature_loop_only_is_substituted(tmp_path, original_temperature_loop, fake_solver):
    source = temperature_source(original_temperature_loop)
    module = make_module(tmp_path, source)
    calls, solver_result = fake_solver
    after_calls = []

    def after_loop():
        after_calls.append(True)
        return "untouched code"

    namespace = {"args": SimpleNamespace(epsilon_eta=0.01), "after_loop": after_loop}
    function, digest = load_reference_loss(module, namespace, fused_temperature=True)
    inputs = tuple(object() for _ in range(5))
    actual, metrics = function(inputs)
    assert actual is solver_result
    assert calls == [(*inputs, 0.01)]
    assert metrics == {"after": "untouched code"} and after_calls == [True]
    assert digest == hashlib.sha256(source.encode()).hexdigest()
    assert Path(module.__file__).read_text() == source
    assert "_shared_solve_log_temperature" not in namespace


@pytest.mark.parametrize("old,new", [("range(32)", "range(31)"),
                                     ("0.5 *", "0.25 *"),
                                     ("mid_kl >", "mid_kl >=")])
def test_changed_temperature_algorithm_fails_closed(tmp_path, original_temperature_loop, fake_solver, old, new):
    assert old in original_temperature_loop
    source = temperature_source(original_temperature_loop.replace(old, new))
    module = make_module(tmp_path, source)
    with pytest.raises(ValueError, match="exact frozen v30 recurrence"):
        load_reference_loss(module, {}, fused_temperature=True)
    # The unfused extractor still supports trusted alternative algorithms.
    assert callable(load_reference_loss(module, {})[0])
    assert fake_solver[0] == []
    assert Path(module.__file__).read_text() == source


def test_duplicate_matching_loops_fail_closed(tmp_path, original_temperature_loop, fake_solver):
    module = make_module(tmp_path, temperature_source(original_temperature_loop + "\n" + original_temperature_loop))
    with pytest.raises(ValueError, match="exact frozen v30 recurrence"):
        load_reference_loss(module, {}, fused_temperature=True)


@pytest.mark.parametrize("source", ["pass\n", "def update_loss_model(): pass\ndef update_loss_model(): pass\n"])
def test_missing_or_duplicate_named_functions_are_rejected(tmp_path, source):
    module = make_module(tmp_path, source)
    with pytest.raises(ValueError, match="exactly one"):
        load_reference_loss(module, {})


def test_cosmetic_source_changes_keep_substitution_but_change_auditable_hash(tmp_path, original_temperature_loop, fake_solver):
    source = temperature_source(original_temperature_loop)
    module = make_module(tmp_path, source)
    first_hash = load_reference_loss(module, {}, fused_temperature=True)[1]
    cosmetic_source = "# Same algorithm, different source revision.\n\n" + source
    Path(module.__file__).write_text(cosmetic_source)
    second_hash = load_reference_loss(module, {}, fused_temperature=True)[1]
    assert first_hash != second_hash
    assert second_hash == hashlib.sha256(cosmetic_source.encode()).hexdigest()
    assert Path(module.__file__).read_text() == cosmetic_source
