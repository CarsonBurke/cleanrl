# TD7-LeSALE compiled v3 — static full-graph training base for future LeJEPA variants.
#
# Uses default-mode torch.compile over separate encoder, critic, and actor loss owners, fused Adam,
# and TF32. Random draws and mutable TD7 bookkeeping stay outside compiled regions. In particular,
# reduce-overhead is intentionally excluded because interleaved loss owners are unsafe with shared
# CUDA-graph pools. The implementation remains in td7_lesale_v1 so eager and compiled execution use
# exactly the same algorithm and can be ablated without code drift.
import runpy
import sys
from pathlib import Path


def _enable_unless_overridden(flag):
    negative = "--no-" + flag.removeprefix("--")
    if flag not in sys.argv and negative not in sys.argv:
        sys.argv.append(flag)


_enable_unless_overridden("--torch-compile")
_enable_unless_overridden("--fused-adam")
_enable_unless_overridden("--tf32")
_enable_unless_overridden("--gpu-replay")
if "--exp-name" not in sys.argv:
    sys.argv.extend(["--exp-name", "td7_lesale_compile_v3"])
runpy.run_path(str(Path(__file__).with_name("td7_lesale_v1.py")), run_name="__main__")
