# TD7-LeSALE JEDI auxiliary v4 — StockSIG plus conditional latent EDM denoising.
#
# Retains stock SALE MSE, its detached future target, SubSIG, and TD7's deterministic zsa control
# path. JEDI is an additive end-to-end context objective only. Compilation defaults on; TF32 and
# fused Adam stay off so the first comparison isolates the algorithm rather than numeric kernels.
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
_enable_unless_overridden("--use-subsig")
_enable_unless_overridden("--gpu-replay")
if not _has_option("--residual-predictor") and not _has_option("--no-residual-predictor"):
    sys.argv.append("--no-residual-predictor")
if not _has_option("--subsig-coef"):
    sys.argv.extend(["--subsig-coef", "0.0002"])
if not _has_option("--exp-name"):
    sys.argv.extend(["--exp-name", "td7_lesale_jedi_aux_v4"])

runpy.run_path(str(Path(__file__).with_name("td7_lesale_v1.py")), run_name="__main__")
