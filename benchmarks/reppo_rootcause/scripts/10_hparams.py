"""Dump the logged `hyperparameters` text of every REPPO-related HalfCheetah run, then query it.

    .venv/bin/python benchmarks/reppo_rootcause/scripts/10_hparams.py dump
    .venv/bin/python benchmarks/reppo_rootcause/scripts/10_hparams.py show REGEX key[,key...]

`dump` reads only the head of each tfevents file (the text summary is written before the first rollout) and
writes data/10_hparams.json. `show` prints the chosen keys for matching runs ("-" when the version has no such
flag). Read-only on runs/.
"""
import json
import re
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "benchmarks/reppo_rootcause/data/10_hparams.json"
NAMES = re.compile(r"reppo|qprop|ipg|latent_critic")


def read(run):
    for events in sorted(run.glob("events.out.tfevents.*")):
        for index, event in enumerate(EventFileLoader(str(events)).Load()):
            for value in event.summary.value:
                if "hyperparameters" in value.tag:
                    text = value.tensor.string_val[0].decode()
                    return {k.strip(): v.strip() for k, v in re.findall(r"\|([^|\n]+)\|([^|\n]*)\|", text)}
            if index > 50:
                break
    return None


def main():
    if sys.argv[1] == "dump":
        table = {}
        for run in sorted((ROOT / "runs").glob("HalfCheetah-v4__*")):
            if NAMES.search(run.name):
                table[run.name] = read(run)
        OUT.write_text(json.dumps(table, indent=0, sort_keys=True))
        print(f"{len(table)} runs, {sum(v is None for v in table.values())} without hyperparameters -> {OUT}")
        return
    table = json.loads(OUT.read_text())
    regex, keys = re.compile(sys.argv[2]), sys.argv[3].split(",")
    for name, params in table.items():
        if regex.search(name) and params:
            short = re.sub(r"HalfCheetah-v4__(.*)__1__\d*(\d{5})$", r"\1 [\2]", name)
            print(f"{short[:66]:66s} " + " ".join(f"{key}={params.get(key, '-')}" for key in keys))


if __name__ == "__main__":
    main()
