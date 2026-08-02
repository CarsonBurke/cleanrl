# TD7-LeSALE compile reward-policy routed v6 — full outcome losses with safe ownership.
#
# Reward reconstruction and deterministic policy-mean prediction update SALE at full strength.
# Scalar reward SIGReg remains weight 1 but updates only the reward tokenizer, preventing a marginal
# shape objective from overwriting the winning observation representation. Actor/critic are unchanged.
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
    "--reward-sigreg-tokenizer-only",
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
    "--exp-name": "td7_lesale_compile_rewardpolicy_routed_v6",
}
for flag, value in defaults.items():
    if not _has_option(flag):
        sys.argv.extend([flag, value])

runpy.run_path(str(Path(__file__).with_name("td7_lesale_v1.py")), run_name="__main__")
