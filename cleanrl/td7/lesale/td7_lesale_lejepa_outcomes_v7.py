# TD7-LeSALE LeJEPA outcomes v7 — independent observation, reward, and policy tokens.
#
# Stock SALE predicts the next observation token. Two full-strength attached-target LeJEPA
# branches predict symlog reward and the deterministic next-policy token from the same z(s,a).
# Each target token owns its own SIGReg; no EMA, stop-gradient target, warmup, or gradient surgery.
import runpy
import sys
from pathlib import Path


def _has_option(flag):
    return any(argument == flag or argument.startswith(flag + "=") for argument in sys.argv)


def _enable_unless_overridden(flag):
    negative = "--no-" + flag.removeprefix("--")
    if not _has_option(flag) and not _has_option(negative):
        sys.argv.append(flag)


for flag in (
    "--torch-compile",
    "--gpu-replay",
    "--use-subsig",
    "--lejepa-outcome-tokens",
):
    _enable_unless_overridden(flag)
if not _has_option("--residual-predictor") and not _has_option("--no-residual-predictor"):
    sys.argv.append("--no-residual-predictor")
if not _has_option("--fused-adam") and not _has_option("--no-fused-adam"):
    sys.argv.append("--no-fused-adam")
if not _has_option("--tf32") and not _has_option("--no-tf32"):
    sys.argv.append("--no-tf32")
defaults = {
    "--subsig-coef": "0.0002",
    "--outcome-token-coef": "1.0",
    "--outcome-sigreg-coef": "1.0",
    "--outcome-policy-sigreg-num-proj": "32",
    "--exp-name": "td7_lesale_lejepa_outcomes_v7",
}
for flag, value in defaults.items():
    if not _has_option(flag):
        sys.argv.extend([flag, value])

runpy.run_path(str(Path(__file__).with_name("td7_lesale_v1.py")), run_name="__main__")
