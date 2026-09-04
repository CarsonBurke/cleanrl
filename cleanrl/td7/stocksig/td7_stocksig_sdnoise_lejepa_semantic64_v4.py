# TD7 StockSIG SDNoise LeJEPA semantic64 v4 — semantic reward and policy readout tokens.
#
# Keeps v3's isolated attached LeWM projector dynamics and FullSIG geometry. Replaces scalar/
# action-width Gaussianized outcome embeddings with dedicated 64-D semantic tokens: reward uses a
# 51-bin HL-Gauss target over symlog([-40, 40]), while policy predicts the next SDNoise mean and
# normalized log-standard-deviation directly. Outcome tokens need no separate SIGReg.
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


_enable_unless_overridden("--semantic-outcome-tokens")
_disable_unless_overridden("--lejepa-outcome-tokens")

defaults = {
    "--semantic-outcome-token-dim": "64",
    "--semantic-reward-num-bins": "51",
    "--semantic-reward-raw-min": "-40",
    "--semantic-reward-raw-max": "40",
    "--semantic-reward-sigma-ratio": "0.75",
    # v215 weights both reward and policy-forecast semantics at 0.5 beside latent dynamics.
    "--outcome-token-coef": "0.5",
    "--exp-name": "td7_stocksig_sdnoise_lejepa_semantic64_v4",
}
for flag, value in defaults.items():
    _set_unless_overridden(flag, value)

runpy.run_path(
    str(Path(__file__).with_name("td7_stocksig_sdnoise_lejepa_dualproj_v3.py")),
    run_name="__main__",
)
