# TD7-LeSALE LeWM-JEDI v7 — stable control latents plus a separate predictive world-model space.
#
# Actor and critic retain StockSIG's deterministic raw SALE interface. A separate MLP+BN projector
# space receives attached future-target prediction, full-dimensional fresh SIGReg, and JEDI
# denoising. This imports LeWM's end-to-end geometry without exposing TD7 to sampled endpoints.
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
_enable_unless_overridden("--jedi-aux")
_enable_unless_overridden("--lewm-projected-aux")
_enable_unless_overridden("--use-subsig")
_enable_unless_overridden("--gpu-replay")
if not _has_option("--residual-predictor") and not _has_option("--no-residual-predictor"):
    sys.argv.append("--no-residual-predictor")
if not _has_option("--subsig-coef"):
    sys.argv.extend(["--subsig-coef", "0.0002"])
if not _has_option("--exp-name"):
    sys.argv.extend(["--exp-name", "td7_lesale_lewm_jedi_v7"])

runpy.run_path(str(Path(__file__).with_name("td7_lesale_v1.py")), run_name="__main__")
