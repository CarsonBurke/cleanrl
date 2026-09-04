# TD7 StockSIG SDNoise v1 — compiled StockSIG with state-dependent Gaussian exploration.
#
# Base: the recorded td7_lesale_compileonly_v3 arm (stock SALE predictor, detached next-latent
# target, uniform representation replay, SubSIG coefficient 2e-4, default-mode torch.compile).
# Fused Adam, TF32, and GPU replay remain off, matching that run's recorded hyperparameters.
# The actor adds a learned diagonal log-std head. Behavior and actor updates use clipped additive
# Gaussian actions mu(s) + sigma(s)*epsilon, while critic targets retain TD7's deterministic target
# mean and fixed target-policy smoothing. Entropy affects only the actor, never Bellman targets.
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


_enable_unless_overridden("--torch-compile")
_enable_unless_overridden("--sd-noise")
_enable_unless_overridden("--use-subsig")
if not _has_option("--residual-predictor") and not _has_option("--no-residual-predictor"):
    sys.argv.append("--no-residual-predictor")
_set_unless_overridden("--subsig-coef", "0.0002")
if not _has_option("--exp-name"):
    sys.argv.extend(["--exp-name", "td7_stocksig_sdnoise_v1"])
runpy.run_path(str(Path(__file__).with_name("td7_lesale_v1.py")), run_name="__main__")
