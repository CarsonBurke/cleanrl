# TD7 StockSIG SDNoise PCActor v1 -- actor-only prospective predictive coding.
#
# Preserves the recorded StockSIG SDNoise replay, encoder, twin critic, targets,
# checkpoints, and temperature updates. Only actor parameter backpropagation is
# replaced by ten reverse-GS PC sweeps for the hidden stack and exact
# free-endpoint local directions for both output heads, using no-decay Adam ascent.
import runpy
import sys
from pathlib import Path


def _has_option(flag):
    return any(argument == flag or argument.startswith(flag + "=") for argument in sys.argv)


def _enable_unless_overridden(flag):
    negative = "--no-" + flag.removeprefix("--")
    if not _has_option(flag) and not _has_option(negative):
        sys.argv.append(flag)


def _set_unless_overridden(flag, value):
    if not _has_option(flag):
        sys.argv.extend([flag, value])


_enable_unless_overridden("--torch-compile")
_enable_unless_overridden("--sd-noise")
_enable_unless_overridden("--use-subsig")
_enable_unless_overridden("--pc-actor")
if not _has_option("--residual-predictor") and not _has_option("--no-residual-predictor"):
    sys.argv.append("--no-residual-predictor")
_set_unless_overridden("--subsig-coef", "0.0002")
if not _has_option("--exp-name"):
    sys.argv.extend(["--exp-name", "td7_stocksig_sdnoise_pcactor_v1"])
runpy.run_path(str(Path(__file__).with_name("td7_lesale_v1.py")), run_name="__main__")
