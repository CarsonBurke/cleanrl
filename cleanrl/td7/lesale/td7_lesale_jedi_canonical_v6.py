# TD7-LeSALE JEDI canonical v6 — one bounded latent coordinate for prediction and control.
#
# SALE learns B(z_t), a -> B(z_{t+1}); actor, critic, and the three-step JEDI sampler all consume
# that same coordinate. Its sampler uses an identity-on-support clamp, avoiding a second tanh of an
# already bounded target, and the actor differentiates through the actual frozen Euler sampler.
# Raw SALE embeddings remain available only to SubSIG and representation-geometry diagnostics.
import runpy
import sys
from pathlib import Path


def _has_option(flag):
    return any(argument == flag or argument.startswith(flag + "=") for argument in sys.argv)


def _enable_unless_overridden(flag):
    negative = "--no-" + flag.removeprefix("--")
    if not _has_option(flag) and not _has_option(negative):
        sys.argv.append(flag)


_enable_unless_overridden("--torch-compile")
_enable_unless_overridden("--jedi-aux")
_enable_unless_overridden("--jedi-endpoint-control")
_enable_unless_overridden("--jedi-canonical-control-latents")
_enable_unless_overridden("--jedi-exact-actor-gradients")
_enable_unless_overridden("--use-subsig")
_enable_unless_overridden("--gpu-replay")
if not _has_option("--residual-predictor") and not _has_option("--no-residual-predictor"):
    sys.argv.append("--no-residual-predictor")
if not _has_option("--subsig-coef"):
    sys.argv.extend(["--subsig-coef", "0.0002"])
if not _has_option("--exp-name"):
    sys.argv.extend(["--exp-name", "td7_lesale_jedi_canonical_v6"])

runpy.run_path(str(Path(__file__).with_name("td7_lesale_v1.py")), run_name="__main__")
