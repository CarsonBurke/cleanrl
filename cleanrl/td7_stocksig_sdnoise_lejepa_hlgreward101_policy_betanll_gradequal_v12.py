# TD7 StockSIG SDNoise LeJEPA 101-bin direct reward with gradient equalization v12.
#
# Keeps v11 unchanged except for a finer Stop-Regressing/v215-style immediate-reward target:
# 101 HL-Gauss bins uniformly span symlog(raw reward) for raw [-40, 40], with sigma 0.5 bins.
import runpy
import sys
from pathlib import Path


def _has_option(flag):
    return any(argument == flag or argument.startswith(flag + "=") for argument in sys.argv)


def _set_unless_overridden(flag, value):
    if not _has_option(flag):
        sys.argv.extend([flag, value])


defaults = {
    "--semantic-reward-num-bins": "101",
    "--semantic-reward-sigma-ratio": "0.5",
    "--semantic-reward-raw-min": "-40.0",
    "--semantic-reward-raw-max": "40.0",
    "--exp-name": "td7_stocksig_sdnoise_lejepa_hlgreward101_policy_betanll_gradequal_v12",
}
for flag, value in defaults.items():
    _set_unless_overridden(flag, value)

runpy.run_path(
    str(
        Path(__file__).with_name(
            "td7_stocksig_sdnoise_lejepa_hlgreward_policy_betanll_gradequal_v11.py"
        )
    ),
    run_name="__main__",
)
