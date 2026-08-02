# TD7-LeSALE LeJEPA SAC v11 — v10 representation with baseline SAC control semantics.
#
# Retains v10's attached three-token reference-batch SIGReg world model. Replaces deterministic
# TD7 control with a tanh-Gaussian policy, entropy Bellman target, min-Q SAC actor objective,
# automatic temperature tuning, sampled behavior, and baseline compensated policy updates.
import runpy
import sys
from pathlib import Path


def _has_option(flag):
    return any(argument == flag or argument.startswith(flag + "=") for argument in sys.argv)


def _enable_unless_overridden(flag):
    negative = "--no-" + flag.removeprefix("--")
    if not _has_option(flag) and not _has_option(negative):
        sys.argv.append(flag)


def _disable_unless_overridden(flag):
    negative = "--no-" + flag.removeprefix("--")
    if not _has_option(flag) and not _has_option(negative):
        sys.argv.append(negative)


for flag in ("--sac-policy", "--sac-autotune", "--sac-compensate-policy-delay"):
    _enable_unless_overridden(flag)
_disable_unless_overridden("--use-checkpoints")
defaults = {
    "--learning-starts": "5000",
    "--sac-alpha": "0.2",
    "--sac-alpha-lr": "0.001",
    "--sac-log-std-min": "-5.0",
    "--sac-log-std-max": "2.0",
    "--exp-name": "td7_lesale_lejepa_sac_v11",
}
for flag, value in defaults.items():
    if not _has_option(flag):
        sys.argv.extend([flag, value])

runpy.run_path(
    str(Path(__file__).with_name("td7_lesale_lejepa_refbatch128_v10.py")),
    run_name="__main__",
)
