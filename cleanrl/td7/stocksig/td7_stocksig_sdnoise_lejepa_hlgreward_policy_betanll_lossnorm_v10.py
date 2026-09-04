# TD7 StockSIG SDNoise LeJEPA direct outcomes with Dreamer4 loss normalization v10.
#
# Keeps v9's direct HL-Gauss reward and bounded policy-Beta objectives. Stock observation, private
# LeWM observation, reward, and policy prediction losses each use Dreamer4's lagged RMS coercion;
# outcome coefficients are one, so no hand-tuned relative reward/policy loss scale remains.
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


_enable_unless_overridden("--dreamer-loss-normalization")
defaults = {
    "--loss-normalization-beta": "0.95",
    "--loss-normalization-eps": "1e-6",
    "--outcome-token-coef": "1.0",
    "--policy-beta-nll-coef": "1.0",
    "--exp-name": "td7_stocksig_sdnoise_lejepa_hlgreward_policy_betanll_lossnorm_v10",
}
for flag, value in defaults.items():
    _set_unless_overridden(flag, value)

runpy.run_path(
    str(Path(__file__).with_name("td7_stocksig_sdnoise_lejepa_hlgreward_policy_betanll_bounded_v9.py")),
    run_name="__main__",
)
