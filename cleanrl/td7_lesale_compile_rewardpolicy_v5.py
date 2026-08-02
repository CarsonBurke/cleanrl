# TD7-LeSALE compile reward-policy v5 — full-strength outcome auxiliaries on compile-only v3.
#
# Preserves the recorded Stock SALE control path and SubSIG. The action-conditioned zsa predicts a
# scalar Gaussian reward token with unit-weight SIGReg and no shared-gradient attenuation; zs also
# predicts the frozen deterministic target actor's mean action. Neither auxiliary enters TD targets.
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
    "--reward-token-aux",
    "--policy-mean-aux",
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
    "--reward-token-coef": "1.0",
    "--reward-token-sigreg-coef": "1.0",
    "--reward-token-warmup-steps": "1",
    "--reward-token-shared-scale": "1.0",
    "--policy-mean-coef": "1.0",
    "--exp-name": "td7_lesale_compile_rewardpolicy_v5",
}
for flag, value in defaults.items():
    if not _has_option(flag):
        sys.argv.extend([flag, value])

runpy.run_path(str(Path(__file__).with_name("td7_lesale_v1.py")), run_name="__main__")
