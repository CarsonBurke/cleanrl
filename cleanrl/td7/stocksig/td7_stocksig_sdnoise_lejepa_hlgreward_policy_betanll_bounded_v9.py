# TD7 StockSIG SDNoise LeJEPA direct-HLG-reward bounded-policy-Beta v9.
#
# Keeps v8's bounded, low-weight policy-moment Beta NLL. Reward is no longer a JEPA target:
# the private world transition directly predicts a 51-bin HL-Gauss distribution over symlog reward.
import runpy
import sys
from pathlib import Path


def _has_option(flag):
    return any(argument == flag or argument.startswith(flag + "=") for argument in sys.argv)


def _enable_unless_overridden(flag):
    negative = "--no-" + flag.removeprefix("--")
    if not _has_option(flag) and not _has_option(negative):
        sys.argv.append(flag)


def _set_unless_overridden(flag, value):
    if not _has_option(flag):
        sys.argv.extend([flag, value])


_enable_unless_overridden("--reward-hlgauss-ce")
defaults = {
    "--outcome-token-coef": "0.05",
    "--exp-name": "td7_stocksig_sdnoise_lejepa_hlgreward_policy_betanll_bounded_v9",
}
for flag, value in defaults.items():
    _set_unless_overridden(flag, value)

runpy.run_path(
    str(Path(__file__).with_name("td7_stocksig_sdnoise_lejepa_rewardiso_policy_betanll_bounded_v8.py")),
    run_name="__main__",
)
