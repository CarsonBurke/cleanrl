# TD7-LeSALE LeJEPA reference-SIGReg v9 — attached observation/reward/policy targets.
#
# Three independent LeWM-strength SIGRegs regularize observation, symlog-reward, and current
# behavior-policy target embeddings. Stock observation targets are attached; outcome predictors
# remain sibling heads over (z_s, action). No EMA, warmup, conflict projection, or gradient scaling.
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


for flag in (
    "--torch-compile",
    "--gpu-replay",
    "--use-full-obs-sigreg",
    "--attached-target",
    "--lejepa-outcome-tokens",
):
    _enable_unless_overridden(flag)
for flag in (
    "--use-subsig",
    "--outcome-sigreg-batch-normalized",
):
    _disable_unless_overridden(flag)
if not _has_option("--residual-predictor") and not _has_option("--no-residual-predictor"):
    sys.argv.append("--no-residual-predictor")
if not _has_option("--fused-adam") and not _has_option("--no-fused-adam"):
    sys.argv.append("--no-fused-adam")
if not _has_option("--tf32") and not _has_option("--no-tf32"):
    sys.argv.append("--no-tf32")
defaults = {
    "--subsig-coef": "0.09",
    "--lewm-sigreg-num-proj": "1024",
    "--outcome-token-coef": "1.0",
    "--outcome-sigreg-coef": "0.09",
    "--outcome-policy-sigreg-num-proj": "1024",
    "--outcome-policy-source": "behavior",
    "--exp-name": "td7_lesale_lejepa_refsig_behavior_v9",
}
for flag, value in defaults.items():
    if not _has_option(flag):
        sys.argv.extend([flag, value])

runpy.run_path(str(Path(__file__).with_name("td7_lesale_v1.py")), run_name="__main__")
