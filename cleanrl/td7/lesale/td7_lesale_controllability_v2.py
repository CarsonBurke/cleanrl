# TD7-LeSALE controllability v2 — versioned launcher for the v2 ablation surface.
#
# The self-contained implementation remains in td7_lesale_v1.py so the continuing StockSIG v1 run
# and the v2 arms share byte-identical TD7/StockSIG machinery. v2 adds two independently gated arms:
# LAP prediction + uniform SubSIG, and persistent controllability-subspace behavior exploration.
import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("td7_lesale_v1.py")), run_name="__main__")
