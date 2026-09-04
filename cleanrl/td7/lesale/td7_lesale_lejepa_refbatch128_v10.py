# TD7-LeSALE LeJEPA reference-batch v10 — exact local SIGReg sample calibration.
#
# Keeps v9's three raw LeWM SIGRegs at lambda=0.09, but evaluates each empirical
# characteristic function on 128 replay examples, matching LeWM's local loader batch.
import runpy
import sys
from pathlib import Path


def _has_option(flag):
    return any(argument == flag or argument.startswith(flag + "=") for argument in sys.argv)


if not _has_option("--sigreg-batch-size"):
    sys.argv.extend(["--sigreg-batch-size", "128"])
if not _has_option("--exp-name"):
    sys.argv.extend(["--exp-name", "td7_lesale_lejepa_refbatch128_v10"])

runpy.run_path(
    str(Path(__file__).with_name("td7_lesale_lejepa_refsig_behavior_v9.py")),
    run_name="__main__",
)
