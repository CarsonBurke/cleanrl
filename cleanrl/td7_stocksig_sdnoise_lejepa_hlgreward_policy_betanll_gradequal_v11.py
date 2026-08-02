# TD7 StockSIG SDNoise LeJEPA direct outcomes with shared-gradient equalization v11.
#
# Restores v9's raw representation objective. Reward CE and bounded policy Beta NLL train their
# heads at unit strength, while independently routed world-model gradients are recalibrated every
# 500 updates so the two outcome branches equally split one representation-gradient budget.
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


_enable_unless_overridden("--adaptive-outcome-grad-equalization")
_disable_unless_overridden("--dreamer-loss-normalization")
defaults = {
    "--outcome-grad-equalization-interval": "500",
    "--outcome-token-coef": "1.0",
    "--policy-beta-nll-coef": "1.0",
    "--exp-name": "td7_stocksig_sdnoise_lejepa_hlgreward_policy_betanll_gradequal_v11",
}
for flag, value in defaults.items():
    _set_unless_overridden(flag, value)

runpy.run_path(
    str(Path(__file__).with_name("td7_stocksig_sdnoise_lejepa_hlgreward_policy_betanll_bounded_v9.py")),
    run_name="__main__",
)
