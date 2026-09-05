"""Load a frozen single-file loss for execution-only benchmark adapters.

Some historical trainers define their loss inside the main guard. Extracting
that named function lets a throughput benchmark reuse the actual loss without
starting the historical training loop or maintaining another algorithm copy.
This executes trusted local repository code, just as importing the trainer does.
"""

import ast
import hashlib
from pathlib import Path


# SHA256 of ast.dump(include_attributes=False) for the frozen v30 temperature
# loop. Whitespace/comments are irrelevant; any arithmetic/name change fails
# closed instead of silently substituting a different algorithm's recurrence.
_V30_TEMPERATURE_LOOP = "d5fbe86772797be720b57c045e3125c532054c13a9864b91d98109d803827732"


def load_reference_loss(module, namespace, name="update_loss_model", *, fused_temperature=False):
    path = Path(module.__file__)
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name!r} function in {path}")
    globals_ = dict(vars(module), **namespace)
    if fused_temperature:
        from cleanrl.shared.vmpo_temperature import solve_log_temperature

        loops = [node for node in ast.walk(matches[0]) if isinstance(node, ast.For)]
        eligible = [node for node in loops if hashlib.sha256(
            ast.dump(node, include_attributes=False).encode()).hexdigest() == _V30_TEMPERATURE_LOOP]
        if len(eligible) != 1:
            raise ValueError("fused temperature requires the exact frozen v30 recurrence")

        class ReplaceTemperature(ast.NodeTransformer):
            def visit_For(self, node):
                if node is eligible[0]:
                    replacement = ast.parse(
                        "log_eta_high = _shared_solve_log_temperature(centered_advantages, "
                        "selected, log_selected_count, log_eta_low, log_eta_high, args.epsilon_eta)"
                    ).body[0]
                    return ast.copy_location(replacement, node)
                return self.generic_visit(node)

        matches[0] = ast.fix_missing_locations(ReplaceTemperature().visit(matches[0]))
        globals_["_shared_solve_log_temperature"] = solve_log_temperature
    exec(compile(ast.Module(body=matches, type_ignores=[]), str(path), "exec"), globals_)
    return globals_[name], hashlib.sha256(source.encode()).hexdigest()
