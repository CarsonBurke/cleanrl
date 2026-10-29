"""Print matched-step return rows from the root-cause TSVs for run-name regexes.

    .venv/bin/python benchmarks/reppo_rootcause/scripts/10_rows.py [--tsv reppo|all|baseline] REGEX [REGEX ...]

Columns: last_step(M), @0.5M, @1M, @2M, @3M, @4M, @6M, @8M, @12M, @16M, last20. Reads only the TSVs written by
dump_runs.py (same +-50k window as scripts/score_runs.py --at); no tfevents are opened.
"""
import argparse
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {"reppo": "reppo_runs.tsv", "all": "all_hc_runs.tsv", "baseline": "baseline_runs.tsv"}
STEPS = ("@500k", "@1000k", "@2000k", "@3000k", "@4000k", "@6000k", "@8000k", "@12000k", "@16000k", "last20")


def short(name):
    match = re.match(r"HalfCheetah-v4__(.*)__1__(\d+)$", name)
    return f"{match.group(1)} [{match.group(2)[-5:]}]" if match else name


def cell(value):
    try:
        return f"{float(value):7.0f}"
    except (TypeError, ValueError):
        return "      -"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", default="reppo", choices=FILES)
    parser.add_argument("patterns", nargs="+")
    cli = parser.parse_args()
    with open(ROOT / FILES[cli.tsv]) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for pattern in cli.patterns:
        regex = re.compile(pattern)
        print(f"## {pattern}")
        for row in rows:
            if not regex.search(row["run"]) or row.get("last_step") in (None, "EMPTY", ""):
                continue
            try:
                last = f"{float(row['last_step']) / 1e6:5.2f}M"
            except ValueError:
                continue
            print(f"{short(row['run'])[:74]:74s} {last} " + " ".join(cell(row.get(step)) for step in STEPS))


if __name__ == "__main__":
    main()
