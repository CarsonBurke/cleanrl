"""Red-team 08: print selected hyperparameters of runs from the tfevents text summary."""
import re, sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

KEYS = sys.argv[1].split(",")
for name in sys.argv[2:]:
    for d in sorted(Path("runs").glob(f"HalfCheetah-v4__{name}__1__*"))[:1]:
        acc = EventAccumulator(str(d), size_guidance={"tensors": 0}); acc.Reload()
        txt = ""
        for t in acc.Tags().get("tensors", []):
            if "hyperparameters" in t:
                txt = acc.Tensors(t)[0].tensor_proto.string_val[0].decode()
        hp = dict(re.findall(r"\|([^|\n]+)\|([^|\n]*)\|", txt))
        print(name, {k: hp.get(k) for k in KEYS if k in hp})
