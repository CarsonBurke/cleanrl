# TD7 StockSIG SDNoise PCActor b32 v2 -- smaller stochastic PC actor updates.
#
# Preserves v1's critic, encoder, replay batch, full-shape policy-noise draw,
# temperature update, and exact output-head directions. Only terminal-force and
# local PC actor learning use the first 32 rows of the randomized replay batch.
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
_set_unless_overridden("--pc-actor-batch-size", "32")
_set_unless_overridden("--subsig-coef", "0.0002")
if not _has_option("--exp-name"):
    sys.argv.extend(["--exp-name", "td7_stocksig_sdnoise_pcactor_b32_v2"])
runpy.run_path(str(Path(__file__).with_name("td7_lesale_v1.py")), run_name="__main__")
