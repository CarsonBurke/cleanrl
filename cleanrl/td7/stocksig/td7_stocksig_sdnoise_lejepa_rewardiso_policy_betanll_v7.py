# TD7 StockSIG SDNoise LeJEPA reward-isometric policy-Beta-NLL v7.
#
# Keeps v6's attached isometric HL-Gauss reward target and the same next-state SDNoise policy-moment
# targets. It replaces only the policy target tokenizer and latent MSE with a Dreamer4-style
# unimodal Beta NLL over those normalized moments, using a 64-D hidden prediction token.
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


_enable_unless_overridden("--policy-beta-nll")

defaults = {
    "--policy-beta-nll-eps": "1e-5",
    "--exp-name": "td7_stocksig_sdnoise_lejepa_rewardiso_policy_betanll_v7",
}
for flag, value in defaults.items():
    _set_unless_overridden(flag, value)

runpy.run_path(
    str(Path(__file__).with_name("td7_stocksig_sdnoise_lejepa_isometric64_v6.py")),
    run_name="__main__",
)
