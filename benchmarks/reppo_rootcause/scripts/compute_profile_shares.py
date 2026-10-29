"""Categorize a v36+ `update_profile.txt` (torch.profiler table, top 40 rows by CUDA time) into kernel classes.

The table is truncated at 40 rows, so the classes are reported against the file's own "Self CUDA time total";
the remainder is the unlisted tail. Wrapper rows (compiled regions, aten::*) carry no self CUDA time.

Usage: .venv/bin/python benchmarks/reppo_rootcause/scripts/compute_profile_shares.py PROFILE.txt [...]
"""
from __future__ import annotations

import re
import sys

UNITS = {"us": 1e-3, "ms": 1.0, "s": 1e3}
CLASSES = (("matmul", ("gemm", "cutlass", "splitKreduce", "aten::mm", "addmm")),
           ("optimizer/clip", ("multi_tensor", "vectorized_elementwise")),
           ("triton norm/act/loss", ("triton_",)),
           ("memory", ("Memset", "memcpy", "Memcpy")))


def milliseconds(cell):
    match = re.fullmatch(r"([\d.]+)(us|ms|s)", cell)
    return float(match.group(1)) * UNITS[match.group(2)] if match else None


def classify(name):
    # Triton kernels that fuse the bias add into a matmul epilogue are elementwise work, not GEMMs.
    if name.startswith("triton_"):
        return "triton norm/act/loss"
    for label, needles in CLASSES:
        if any(needle in name for needle in needles):
            return label
    return "other"


def main():
    for path in sys.argv[1:]:
        lines = open(path).read().splitlines()
        header = re.split(r"\s{2,}", lines[1].strip())
        self_cuda, calls = header.index("Self CUDA"), header.index("# of Calls")
        total = milliseconds(next(line for line in lines if line.startswith("Self CUDA time total")).split()[-1])
        sums, counts = {}, {}
        for line in lines[3:]:
            cells = re.split(r"\s{2,}", line.strip())
            if len(cells) != len(header):
                continue
            time = milliseconds(cells[self_cuda])
            if not time:
                continue
            label = classify(cells[0])
            sums[label] = sums.get(label, 0.0) + time
            counts[label] = counts.get(label, 0) + int(cells[calls])
        listed = sum(sums.values())
        print(f"{path}\n  self CUDA total {total:.1f} ms; top-40 rows cover {listed:.1f} ms ({listed / total:.0%})")
        for label, time in sorted(sums.items(), key=lambda item: -item[1]):
            print(f"  {label:22s} {time:7.1f} ms  {time / total:5.1%}  {counts[label]:6d} launches "
                  f"({1e3 * time / counts[label]:.1f} us each)")
        print(f"  {'unlisted tail':22s} {total - listed:7.1f} ms  {(total - listed) / total:5.1%}")


if __name__ == "__main__":
    main()
