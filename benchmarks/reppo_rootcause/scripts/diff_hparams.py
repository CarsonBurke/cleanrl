"""Diff the logged `hyperparameters` text summary of two runs (substring match on run dir names)."""
import re
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[3] / "runs"


def hparams(pattern: str) -> dict[str, str]:
    (run,) = [d for d in ROOT.iterdir() if re.fullmatch(rf"HalfCheetah-v4__{re.escape(pattern)}__1__\d+", d.name)][-1:]
    ea = EventAccumulator(str(run), size_guidance={"tensors": 1, "scalars": 1})
    ea.Reload()
    tag = next(t for t in ea.Tags()["tensors"] if "hyperparameters" in t)
    text = ea.Tensors(tag)[0].tensor_proto.string_val[0].decode()
    return dict(re.findall(r"\|([^|\n]+)\|([^|\n]*)\|", text))


a, b = hparams(sys.argv[1]), hparams(sys.argv[2])
for k in sorted(set(a) | set(b)):
    if a.get(k) != b.get(k) and k.strip() not in ("exp_name",):
        print(f"  {k}: {a.get(k)!r} vs {b.get(k)!r}")
print("(end diff)")
