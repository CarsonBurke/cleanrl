"""Exercise the public dry-run CLI without contacting the machine queue."""

from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def command(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts/submit_mujoco.py"),
                           "--dry-run", *args], capture_output=True, text=True)


def test_uncharacterized_work_does_not_inherit_standard_concurrency():
    for options in [("--num-envs", "64"), ("--env-threads", "4"),
                    ("--", "--update-epochs", "20")]:
        result = command(*options)
        assert result.returncode == 0, result.stderr
        submission = shlex.split(result.stdout)
        assert submission[submission.index("--max-parallel-runs") + 1] == "1"
        assert "CLEANRL_ENV_SPIN=0" in submission


def test_explicit_resource_budget_overrides_automatic_policy():
    result = command("--num-envs", "64", "--max-parallel-runs", "2", "--env-spin", "100")
    assert result.returncode == 0, result.stderr
    submission = shlex.split(result.stdout)
    assert submission[submission.index("--max-parallel-runs") + 1] == "2"
    assert "CLEANRL_ENV_SPIN=100" in submission
    trainer = submission[submission.index("--") + 1:]
    assert trainer[trainer.index("--num-envs") + 1] == "64"


def test_trainer_arguments_cannot_silently_invalidate_queue_budget():
    for flag in ["--num-envs=64", "--env_threads=8", "--exp-name=hidden"]:
        result = command("--", flag)
        assert result.returncode != 0
        assert not result.stdout, "invalid submissions must never reach mlq"
