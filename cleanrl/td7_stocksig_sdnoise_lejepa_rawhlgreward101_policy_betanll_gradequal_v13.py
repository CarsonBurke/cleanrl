# TD7 StockSIG SDNoise LeJEPA raw-coordinate 101-bin reward ablation v13.
#
# Ablates only v12's symlog coordinate transform: 101 uniformly spaced raw-reward bins cover
# [-40, 40] with HL-Gauss sigma 0.5 bin widths. Gradient equalization and policy Beta stay fixed.
import runpy
import sys
from pathlib import Path


def _has_option(flag):
    return any(argument == flag or argument.startswith(flag + "=") for argument in sys.argv)


def _disable_unless_overridden(flag):
    negative = "--no-" + flag.removeprefix("--")
    if not _has_option(flag) and not _has_option(negative):
        sys.argv.append(negative)


def _set_unless_overridden(flag, value):
    if not _has_option(flag):
        sys.argv.extend([flag, value])


_disable_unless_overridden("--reward-hlgauss-symlog")
_set_unless_overridden(
    "--exp-name",
    "td7_stocksig_sdnoise_lejepa_rawhlgreward101_policy_betanll_gradequal_v13",
)

runpy.run_path(
    str(
        Path(__file__).with_name(
            "td7_stocksig_sdnoise_lejepa_hlgreward101_policy_betanll_gradequal_v12.py"
        )
    ),
    run_name="__main__",
)
