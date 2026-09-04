# TD7-LeSALE LeJEPA SAC HL-Gauss v12 — categorical twin Q and full policy moments.
#
# Extends v11 with twin 511-bin symlog HL-Gauss Q heads trained from the scalar soft
# Bellman target. Actor/targets consume expected raw Q; LAP priorities use decoded TD error.
# The policy LeJEPA token expands from tanh(mean) to [tanh(mean), log_std].
import runpy
import sys
from pathlib import Path


def _has_option(flag):
    return any(argument == flag or argument.startswith(flag + "=") for argument in sys.argv)


def _enable_unless_overridden(flag):
    negative = "--no-" + flag.removeprefix("--")
    if not _has_option(flag) and not _has_option(negative):
        sys.argv.append(flag)


for flag in ("--hl-gauss-critic", "--outcome-policy-include-log-std"):
    _enable_unless_overridden(flag)
defaults = {
    "--hl-gauss-num-bins": "511",
    "--hl-gauss-v-min": "-9.90353755128617",
    "--hl-gauss-v-max": "9.90353755128617",
    "--hl-gauss-sigma-ratio": "2.0",
    "--exp-name": "td7_lesale_lejepa_sac_hlgauss_v12",
}
for flag, value in defaults.items():
    if not _has_option(flag):
        sys.argv.extend([flag, value])

runpy.run_path(
    str(Path(__file__).with_name("td7_lesale_lejepa_sac_v11.py")),
    run_name="__main__",
)
