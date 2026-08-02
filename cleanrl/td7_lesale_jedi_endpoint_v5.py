# TD7-LeSALE JEDI endpoint v5 — final diffusion latent as the critic's zsa feature.
#
# Uses JEDI/DIAMOND's three-step Euler schedule with one fixed private Gaussian prior and no churn.
# A snapshot-aligned 10k handoff moves critic inputs from stock SALE to the endpoint; actor forwards
# the endpoint while using stock SALE's straight-through action Jacobian. TF32/fused Adam stay off.
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
_enable_unless_overridden("--jedi-endpoint-control")
_enable_unless_overridden("--use-subsig")
_enable_unless_overridden("--gpu-replay")
if not _has_option("--residual-predictor") and not _has_option("--no-residual-predictor"):
    sys.argv.append("--no-residual-predictor")
if not _has_option("--subsig-coef"):
    sys.argv.extend(["--subsig-coef", "0.0002"])
if not _has_option("--exp-name"):
    sys.argv.extend(["--exp-name", "td7_lesale_jedi_endpoint_v5"])

runpy.run_path(str(Path(__file__).with_name("td7_lesale_v1.py")), run_name="__main__")
