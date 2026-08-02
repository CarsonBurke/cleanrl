# TD7 StockSIG SDNoise LeJEPA isometric64 v6 — attached non-collapsing outcome targets.
#
# Keeps v3's isolated attached observation dynamics and FullSIG. HL-Gauss reward distributions and
# next-policy moments enter learned isometric 64-D target encoders shaped by attached latent MSE.
# The world predicts only those tokens: no outcome decoder, direct semantic loss, or outcome SIGReg.
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


def _set_unless_overridden(flag, value):
    if not _has_option(flag):
        sys.argv.extend([flag, value])


_enable_unless_overridden("--isometric-outcome-tokens")
for flag in (
    "--lejepa-outcome-tokens",
    "--semantic-outcome-tokens",
    "--latent-outcome-tokens",
):
    _disable_unless_overridden(flag)

defaults = {
    "--semantic-outcome-token-dim": "64",
    "--semantic-reward-num-bins": "51",
    "--semantic-reward-raw-min": "-40",
    "--semantic-reward-raw-max": "40",
    "--semantic-reward-sigma-ratio": "0.75",
    "--outcome-token-coef": "0.5",
    "--exp-name": "td7_stocksig_sdnoise_lejepa_isometric64_v6",
}
for flag, value in defaults.items():
    _set_unless_overridden(flag, value)

runpy.run_path(
    str(Path(__file__).with_name("td7_stocksig_sdnoise_lejepa_dualproj_v3.py")),
    run_name="__main__",
)
