# TD7 StockSIG SDNoise LeJEPA bounded policy-Beta-NLL v8.
#
# Keeps v7's reward target and direct next-policy-moment likelihood, but prevents deterministic
# targets from driving Beta confidence to infinity. Policy NLL has its own small coefficient so
# its shared-encoder gradient is comparable to the useful auxiliary scale rather than dominant.
import runpy
import sys
from pathlib import Path


def _has_option(flag):
    return any(argument == flag or argument.startswith(flag + "=") for argument in sys.argv)


def _set_unless_overridden(flag, value):
    if not _has_option(flag):
        sys.argv.extend([flag, value])


defaults = {
    "--policy-beta-nll-coef": "0.05",
    "--policy-beta-max-precision": "30",
    "--exp-name": "td7_stocksig_sdnoise_lejepa_rewardiso_policy_betanll_bounded_v8",
}
for flag, value in defaults.items():
    _set_unless_overridden(flag, value)

runpy.run_path(
    str(Path(__file__).with_name("td7_stocksig_sdnoise_lejepa_rewardiso_policy_betanll_v7.py")),
    run_name="__main__",
)
