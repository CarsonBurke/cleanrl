# TD7 StockSIG SDNoise LeJEPA dual-projector v3 — Gaussian JEPA, normalized control.
#
# Retains StockSIG's AvgL1 control latents, detached stock prediction target, and coefficient-2e-4
# SubSIG. A separate unnormalized LeWM projector coordinate has an attached future target and raw
# FullSIG; its private predicted transition feeds attached reward and behavior-policy token heads.
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


_enable_unless_overridden("--lewm-projected-aux")
_enable_unless_overridden("--lewm-private-dynamics")
_enable_unless_overridden("--use-subsig")
_disable_unless_overridden("--use-full-obs-sigreg")
_disable_unless_overridden("--attached-target")

defaults = {
    "--subsig-coef": "0.0002",
    "--control-sigreg-batch-size": "0",
    "--lewm-hidden-dim": "2048",
    "--lewm-coef": "1.0",
    "--lewm-warmup-steps": "1",
    "--lewm-sigreg-coef": "0.09",
    "--lewm-sigreg-num-proj": "1024",
    "--exp-name": "td7_stocksig_sdnoise_lejepa_dualproj_v3",
}
for flag, value in defaults.items():
    _set_unless_overridden(flag, value)

runpy.run_path(
    str(Path(__file__).with_name("td7_stocksig_sdnoise_lejepa_v2.py")),
    run_name="__main__",
)
