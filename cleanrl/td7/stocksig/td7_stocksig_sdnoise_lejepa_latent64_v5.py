# TD7 StockSIG SDNoise LeJEPA latent64 v5 — attached latent outcome prediction.
#
# Keeps v3's isolated attached LeWM dynamics and observation FullSIG. Reward HL-Gauss and next
# SDNoise moments are encoded into separate attached 64-D target tokens with their own LeWM-style
# FullSIG. The private world transition predicts only those latents; target-only decoders anchor
# reward/policy semantics without direct world-to-scalar regression.
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


_enable_unless_overridden("--latent-outcome-tokens")
_disable_unless_overridden("--lejepa-outcome-tokens")
_disable_unless_overridden("--semantic-outcome-tokens")

defaults = {
    "--semantic-outcome-token-dim": "64",
    "--semantic-reward-num-bins": "51",
    "--semantic-reward-raw-min": "-40",
    "--semantic-reward-raw-max": "40",
    "--semantic-reward-sigma-ratio": "0.75",
    "--outcome-token-coef": "1.0",
    "--outcome-semantic-coef": "0.5",
    "--exp-name": "td7_stocksig_sdnoise_lejepa_latent64_v5",
}
for flag, value in defaults.items():
    _set_unless_overridden(flag, value)

runpy.run_path(
    str(Path(__file__).with_name("td7_stocksig_sdnoise_lejepa_dualproj_v3.py")),
    run_name="__main__",
)
