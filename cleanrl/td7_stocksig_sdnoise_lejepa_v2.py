# TD7 StockSIG SDNoise LeJEPA v2 — attached observation, reward, and policy targets.
#
# Extends StockSIG SDNoise v1 with LeWM-calibrated FullSIG: 128-sample statistics, 1024 fresh
# projections, and raw coefficient 0.09. The shared zsa transition predicts the attached next-state,
# reward, and policy tokens; each target family is regularized independently against N(0, I).
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


for flag in (
    "--use-full-obs-sigreg",
    "--attached-target",
    "--lejepa-outcome-tokens",
    "--outcome-from-transition",
):
    _enable_unless_overridden(flag)
for flag in (
    "--use-subsig",
    "--outcome-sigreg-batch-normalized",
):
    _disable_unless_overridden(flag)

defaults = {
    "--sigreg-batch-size": "128",
    "--subsig-coef": "0.09",
    "--lewm-sigreg-num-proj": "1024",
    "--outcome-token-coef": "1.0",
    "--outcome-sigreg-coef": "0.09",
    "--outcome-policy-sigreg-num-proj": "1024",
    "--outcome-policy-source": "behavior",
    "--exp-name": "td7_stocksig_sdnoise_lejepa_v2",
}
for flag, value in defaults.items():
    _set_unless_overridden(flag, value)

runpy.run_path(
    str(Path(__file__).with_name("td7_stocksig_sdnoise_v1.py")),
    run_name="__main__",
)
