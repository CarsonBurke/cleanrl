"""D (controls). Negative and positive controls for the cross-lane head-to-head of 07_D3_combined.py.

    CUDA_VISIBLE_DEVICES= .venv/bin/python benchmarks/reppo_rootcause/scripts/07_D4_controls.py

negative: the critic field with its rows permuted across the chunk (same marginal, no state alignment) -> gain ~ 0
positive: a second, independent unbiased estimator (sf_gae1) scored against the references -> bias-free, gain > 0
leak    : the critic field scored against a reference built from the SAME lanes only (what the cross-lane rule avoids)
"""
import importlib.util
import io
from pathlib import Path

import numpy as np


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common, d1, d2, d3 = (_load(n, f) for n, f in (("common07", "07_common.py"), ("d1", "07_D_headtohead.py"),
                                              ("d2", "07_D2_chunk_reference.py"), ("d3", "07_D3_combined.py")))
for name in ("v23_4M", "v29h8_8M"):
    d = common.load(name)
    est = d2.fields(d)
    rng = np.random.default_rng(0)
    truth = d1.truth_fields(d, 100)[0]["both"].mean(-1)
    lane, step = d.row, np.arange(len(d.q)) // 16
    block = lane * 16 + step * 16 // 1024
    lanes = np.repeat(np.arange(16), 16)
    counts = np.bincount(block, minlength=256).astype(float)
    feats = d1.feature_maps(d)["linear(108)"]
    variants = {"critic (as is)": est["critic"]}
    for i in range(3):
        variants[f"critic rows permuted #{i}"] = est["critic"][rng.permutation(len(lane))]
    variants["sf_gae1 (independent unbiased estimator)"] = est["sf_gae1"]
    c_oracle = d1.contributions(truth, feats[d.index], np.arange(len(d.index)), len(d.index))
    print(f"{name} | linear(108), ref sf_gae0.99")
    for label, field in variants.items():
        c_chunk = {k: d1.contributions(est[k], feats, block, 256) for k in d3.ESTIMATORS}
        c_chunk["critic"] = d1.contributions(field, feats, block, 256)
        res = {}
        d3.analyse(c_chunk, counts, lanes, c_oracle, np.ones(len(d.index)), lane[d.index], "sf_gae0.99", io.StringIO(), "x", res)
        r = res["x|sf_gae0.99|critic"]
        print(f"   {label:42s} gain {r['gain']:+.4f} [{r['gain_ci'][0]:+.4f}, {r['gain_ci'][1]:+.4f}] P(>0) {r['p_gain_pos']:.3f}")
