# TD7-LeSALE compile reward-token v4 — compile-only v3 plus auxiliary reward prediction.
#
# Preserves the recorded compile-only control path: Stock SALE, detached next-observation target,
# uniform SubSIG at coefficient 2e-4, compiled static losses, and GPU replay. A private scalar token
# predicted from zsa is weakly Gaussianized with scalar SIGReg and monotonically decoded to symlog
# forward reward; HalfCheetah's known 0.1||a||^2 control cost is handled analytically.
import runpy
import sys
from pathlib import Path


def _has_option(flag):
    return any(argument == flag or argument.startswith(flag + "=") for argument in sys.argv)


def _enable_unless_overridden(flag):
    negative = "--no-" + flag.removeprefix("--")
    if not _has_option(flag) and not _has_option(negative):
        sys.argv.append(flag)


_enable_unless_overridden("--torch-compile")
_enable_unless_overridden("--gpu-replay")
_enable_unless_overridden("--use-subsig")
_enable_unless_overridden("--reward-token-aux")
if not _has_option("--residual-predictor") and not _has_option("--no-residual-predictor"):
    sys.argv.append("--no-residual-predictor")
if not _has_option("--fused-adam") and not _has_option("--no-fused-adam"):
    sys.argv.append("--no-fused-adam")
if not _has_option("--tf32") and not _has_option("--no-tf32"):
    sys.argv.append("--no-tf32")
if not _has_option("--subsig-coef"):
    sys.argv.extend(["--subsig-coef", "0.0002"])
if not _has_option("--exp-name"):
    sys.argv.extend(["--exp-name", "td7_lesale_compile_rewardtoken_v4"])

runpy.run_path(str(Path(__file__).with_name("td7_lesale_v1.py")), run_name="__main__")
