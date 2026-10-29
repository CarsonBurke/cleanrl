"""List scalar tags of selected runs (red-team 08)."""
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

for pat in sys.argv[1:]:
    for d in sorted(Path("runs").glob(f"HalfCheetah-v4__{pat}__1__*")):
        acc = EventAccumulator(str(d), size_guidance={"scalars": 1})
        acc.Reload()
        print(d.name, "\n  ", " ".join(sorted(acc.Tags()["scalars"])))
