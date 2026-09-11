"""Low-boundary return refinement; identical learners, broader matched searches.

V5 selected the lowest LR in every family. Retain all v5 candidates, extend
learning rates downward, and include zero auxiliary weight. This is adaptive
development refinement, not an independent new algorithm or a fresh holdout.

Run as a module through mlq. Coverage filtering can join nonadjacent timestamps:
these are retained-bar log returns, not guaranteed five-minute returns or P&L.
"""

import hashlib
import json
import math
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl.plasticity import panel_distributional_eval_v1 as frozen
from cleanrl.plasticity import panel_frontier_eval_v3 as frontier
from cleanrl.plasticity import panel_hd
from cleanrl.plasticity import panel_predictive_memory_eval_v4 as memory_eval
from cleanrl.plasticity.panel_predictive_memory_v4 import CausalState
from cleanrl.plasticity.panel_return_representation_v5 import Config, GROUPS, Learner
from cleanrl.plasticity.panel_stream import _ewma, build_panel
from cleanrl.shared import runtime

# All v5 candidates retained; expand BOTH baseline and auxiliary search budgets.
LRS = (3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


def configurations():
    return tuple(Config(group, lr, beta2, 0.)
                 for group in GROUPS[:3] for lr in LRS for beta2 in (.5, .9, .95, .99, .999, .9999)) + tuple(
        Config(group, lr, .999, weight)
        for group in GROUPS[3:] for lr in LRS for weight in (0., .01, .03, .1, 1., 10.))


REFERENCE_NAMES = ("zero_return", "ridge_signed_1")
METRICS = tuple(f"{unit}_{name}" for unit in ("normalized", "raw") for name in frozen.METRICS)
PRIMARY = "memory_aux"
HORIZONS = (1, 4, 16, 32)
SIGN_NAMESPACE = "panel_return_refinement_v6_signed_label_rademacher_seed1"
SOURCE_FILES = tuple(dict.fromkeys((*memory_eval.SOURCE_FILES,
    "cleanrl/plasticity/panel_return_representation_v5.py",
    "benchmarks/plasticity/panel_frontier_v3_evidence.json",
    "cleanrl/plasticity/panel_return_representation_eval_v5.py",
    "cleanrl/plasticity/panel_return_refinement_eval_v6.py")))


@dataclass
class Args(panel_hd.Args):
    output_dir: str = ""
    stage: Literal["development", "confirmation"] = "development"
    reference_result: str = ""
    historical_result: str = ""
    rank_offset: int = 0
    target: str = "ret"
    width: int = 128
    bins: int = 33
    seed: int = 1
    lrs: tuple[float, ...] = LRS
    view: Literal["real", "rademacher"] = "real"
    log_every: int = 4096


def validate_args(a):
    if not a.output_dir.strip():
        raise ValueError("--output-dir required; no resume")
    if (a.target, a.n_stocks, a.lags, a.width, a.bins, a.seed, tuple(a.lrs)) != ("ret", 200, 32, 128, 33, 1, LRS):
        raise ValueError("fixed dimensions, signed-return target, seed1 and LR grid required")
    if (a.min_coverage, a.vol_span, a.first_session_before) != (.9, 200, "2018-01-01"):
        raise ValueError("historical coverage, span200 and session preprocessing required")
    if a.log_every < 16:
        raise ValueError("log-every must fit graph16")
    if a.rank_offset != (0 if a.stage == "development" else 400):
        raise ValueError("development rank0; confirmation rank400")
    if bool(a.reference_result) != (a.stage == "confirmation" or a.view == "rademacher"):
        raise ValueError("confirmation/null require --reference-result; real development cannot inherit locks")
    if bool(a.historical_result) != (a.stage == "confirmation"):
        raise ValueError("confirmation requires --historical-result only")
    if a.stage == "confirmation" and a.view != "real":
        raise ValueError("null is development-only")


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    return {name: frozen.file_sha256(root / name) for name in SOURCE_FILES}


def array_identity(array):
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


def return_series(a, close):
    """Exactly Bank's retained-close arithmetic; denominator through forecast t.

    Arrays retain the separate market column. Labels are NOT clipped or centered.
    Missing/abs>.2 returns enter EWMA as zero, exactly as panel_hd.series.
    """
    keep = np.isfinite(close[:, 1:]).mean(1) >= a.min_coverage
    retained = close[keep]
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.diff(np.log(retained), axis=0, prepend=np.nan)
        raw[np.abs(raw) > .2] = np.nan
        scale = np.sqrt(_ewma(np.square(raw), a.vol_span))
        denominator = np.maximum(scale, 1e-6)
        previous = np.concatenate((np.full((1, retained.shape[1]), np.nan), denominator[:-1]))
        observed_signed = raw / previous
        normalized = np.concatenate((raw[1:, 1:] / denominator[:-1, 1:], np.zeros((1, retained.shape[1] - 1))))
    observed_valid = np.isfinite(observed_signed[:, 1:])
    valid = observed_valid & np.roll(observed_valid, -1, axis=0)
    valid &= np.isfinite(denominator[:, 1:]) & np.isfinite(normalized)
    valid[:a.lags + 1] = False
    valid[-1] = False
    raw_target = np.concatenate((raw[1:, 1:], np.zeros((1, retained.shape[1] - 1))))
    return {"raw_return": raw, "scale": scale, "denominator": denominator[:, 1:],
            "normalized_target": np.where(valid, normalized, 0.),
            "raw_target": np.where(valid, raw_target, 0.), "valid": valid,
            "observed_valid": observed_valid,
            "observed_signed": np.where(np.isfinite(observed_signed), observed_signed, 0.), "keep": keep}


def randomized_labels(normalized, raw):
    """Independent per (forecast,stock) signs; no future-label permutation/read."""
    seed = int.from_bytes(hashlib.sha256(SIGN_NAMESPACE.encode()).digest()[:8], "little")
    generator = np.random.Generator(np.random.PCG64(seed))
    signs = generator.integers(0, 2, size=normalized.shape, dtype=np.int8) * 2 - 1
    return normalized * signs, raw * signs, array_identity(signs)


def data_identity(a, bank, close, symbols, ts, series):
    identity = frozen.data_identity(a, bank)
    # Bank.mu is irrelevant to signed returns and is not used by any predictor.
    identity.pop("mu")
    identity.pop("kappa")
    kept_ts = ts[series["keep"]]
    names = [str(s) for s in symbols]
    if len(names) != bank.N or len(set(names)) != bank.N or "SPY" in names:
        raise ValueError("expected unique stock symbols with SPY only in separate column0")
    boundaries = {"stream_start": bank.L + 1, "selection_start": int(.4 * bank.T), "cut": bank.cut,
                  "development_stop": bank.T - 1, "confirmation_start": 90145, "last_forecast": bank.T - 2}
    identity.update(configs=[asdict(c) for c in configurations()], symbols=names, shared_market_symbol="SPY",
        calendar_boundaries={name: {"bar": i, "timestamp": str(kept_ts[i])} for name, i in boundaries.items() if i < bank.T},
        retained_timestamp_sha256=hashlib.sha256(np.ascontiguousarray(kept_ts).tobytes()).hexdigest(),
        return_arrays={name: array_identity(series[name]) for name in ("raw_return", "scale")})
    return identity


class RidgeReference:
    """Pooled affine ridge1 on 12 trailing signed-return sums plus intercept.

    Horizon1/4/16/32 sums use un-clipped causal unit-RMS own, SPY and
    cross-sectional signed returns through t. Missing observations contribute
    zero; cross-sectional means use current-valid stocks only. Not an oracle.
    """

    def __init__(self, observed_signed, observed_valid, device):
        self.signed = torch.as_tensor(observed_signed, dtype=torch.float64, device=device)
        valid = torch.as_tensor(observed_valid, dtype=torch.bool, device=device)
        self.cross = torch.where(valid, self.signed[:, 1:], 0.).sum(1) / valid.sum(1).clamp_min(1)
        self.lags = torch.arange(32, device=device)
        self.horizons = torch.tensor(HORIZONS, device=device) - 1
        self.gram = torch.zeros((13, 13), dtype=torch.float64, device=device)
        self.rhs = torch.zeros(13, dtype=torch.float64, device=device)
        self.coefficient = torch.zeros(13, dtype=torch.float64, device=device)

    def state_tensors(self):
        return self.gram, self.rhs, self.coefficient

    def features(self, t):
        indices = t - self.lags
        cumulative = self.signed.index_select(0, indices).cumsum(0).index_select(0, self.horizons)
        cross = self.cross.index_select(0, indices).cumsum(0).index_select(0, self.horizons)
        own = cumulative[:, 1:].T
        return torch.cat((own, cumulative[:, 0].expand(own.shape[0], 4),
                          cross.expand(own.shape[0], 4), torch.ones_like(own[:, :1])), dim=1)

    @torch.no_grad()
    def step(self, t, y, mask, fitting):
        x = self.features(t)
        prediction = (x @ self.coefficient)[None]
        fit_mask = mask & fitting
        masked = torch.where(fit_mask[:, None], x, 0.)
        self.gram.add_(masked.T @ masked)
        self.rhs.add_(masked.T @ torch.where(fit_mask, y.double(), 0.))
        return prediction

    @torch.no_grad()
    def fit(self):
        self.coefficient.copy_(torch.linalg.solve(self.gram + torch.eye(13, dtype=torch.float64, device=self.gram.device), self.rhs))
        if not bool(torch.isfinite(self.coefficient).all()):
            raise FloatingPointError("ridge1 coefficient solve is nonfinite")


def metric_row(prediction, normalized, raw, denominator, mask):
    """FP64 SSE of normalized forecasts and actual raw-log-return forecasts."""
    p = torch.where(mask[None], prediction.double(), 0.)
    y = torch.where(mask, normalized.double(), 0.)
    raw_p = p * torch.where(mask, denominator.double(), 0.)[None]
    raw_y = torch.where(mask, raw.double(), 0.)
    n = prediction.shape[0]
    count = mask.sum().double().expand(n)
    return torch.stack(tuple(value for forecast, target in ((p, y), (raw_p, raw_y)) for value in
        ((forecast - target[None]).square().sum(1), forecast.square().sum(1),
         (forecast * target[None]).sum(1), target.square().sum().expand(n), count)), dim=1)


class Runner(frozen.Runner):
    """Inherit fullgraph 1/8/16 capture, complete reset and bitwise replay checks."""

    def __init__(self, bank, model, series, state, ring_size, score_start, score_stop):
        self.bank, self.model, self.state = bank, model, state
        self.start, self.stop = bank.L + 1, bank.T - 1
        self.score_start, self.score_stop, self.ring_size = score_start, score_stop, ring_size
        self.index = torch.tensor(self.start, dtype=torch.int64, device=bank.dev)
        self.target = torch.tensor(series["normalized_target"], dtype=torch.float64, device=bank.dev)
        self.raw_target = torch.tensor(series["raw_target"], dtype=torch.float64, device=bank.dev)
        self.denominator = torch.tensor(series["denominator"], dtype=torch.float64, device=bank.dev)
        self.valid = torch.tensor(series["valid"], dtype=torch.bool, device=bank.dev)
        self.vol_target = torch.roll(bank.zt[:, 1:], -1, 0).square().clamp_max(25.)
        self.ridge = RidgeReference(series["observed_signed"], series["observed_valid"], bank.dev)
        self.output_names = (*model.output_names, *REFERENCE_NAMES)
        self.raw = torch.zeros((self.stop - self.start, len(self.output_names), len(METRICS)), dtype=torch.float64, device=bank.dev)
        self.prediction_ring = torch.zeros((ring_size, len(self.output_names), bank.N), dtype=torch.float64, device=bank.dev)
        self.mutable = (*model.state_tensors(), *state.state_tensors(), *self.ridge.state_tensors(), self.index, self.raw, self.prediction_ring)
        self.compile_seconds = self.capture_seconds = 0.

    @torch.no_grad()
    def step(self):
        index = self.index.reshape(1)
        self.state.observe(self.index)
        y = self.target.index_select(0, index).squeeze(0)
        mask = self.valid.index_select(0, index).squeeze(0)
        vol = self.vol_target.index_select(0, index).squeeze(0)
        learned = self.model.step(self.state.frames(self.index), y.float(), vol, mask)
        ridge = self.ridge.step(self.index, y, mask, self.index < self.bank.cut)
        prediction = torch.cat((learned.double(), torch.zeros_like(y[None]), ridge), dim=0)
        visible = (self.index >= self.score_start) & (self.index < self.score_stop)
        row = metric_row(prediction, y, self.raw_target.index_select(0, index).squeeze(0),
                         self.denominator.index_select(0, index).squeeze(0), mask & visible)
        self.raw.index_copy_(0, (self.index - self.start).reshape(1), row[None])
        self.prediction_ring.index_copy_(0, ((self.index - self.start) % self.ring_size).reshape(1),
                                         torch.where(visible, prediction, 0.)[None])
        self.index.add_(1)


def reduce_metrics(raw):
    return [{"normalized": n, "raw": r} for n, r in zip(frozen.reduce_metrics(raw[:, :, :5]), frozen.reduce_metrics(raw[:, :, 5:]))]


def select_locks(raw, configs, start, selection_start, cut, consumed, health=None):
    if consumed != cut or len(raw) != cut - start or not start <= selection_start < cut:
        raise ValueError("locks require exactly prefix rows, no suffix")
    if raw.shape[1:] != (len(configs) + len(REFERENCE_NAMES), len(METRICS)):
        raise ValueError("candidate/reference metric columns disagree")
    # No raw-unit performance, volatility metric, or suffix outcome selects a lock.
    scores = frozen.reduce_metrics(raw[selection_start - start:, :, :5])
    alive = np.isfinite(raw[:, :, :5]).all(axis=(0, 2))
    if health is not None:
        alive[:len(configs)] &= np.asarray(health, dtype=bool)
    locks = {}
    for group in GROUPS:
        columns = [i for i, c in enumerate(configs) if c.family == group]
        candidates = [{"column": i, **asdict(configs[i]), **scores[i],
                       "eligible": bool(alive[i] and scores[i]["count"] and math.isfinite(scores[i]["mse"]))} for i in columns]
        eligible = [c for c in candidates if c["eligible"]]
        winner = min(eligible, key=lambda c: c["mse"]) if eligible else None
        locks[group] = {"column": winner["column"] if winner else None,
            "config": asdict(configs[winner["column"]]) if winner else None,
            "status": "locked" if winner else "failed_nonfinite_or_empty", "candidates": candidates,
            "selection_start_inclusive": selection_start, "selection_stop_exclusive": cut, "locked_before_bar": cut}
    return locks


def persist_prefix_locks(root, raw, result):
    identity = result["data_identity"]
    if (result["stage"], result["view"]) != ("development", "real") or len(raw) != identity["cut"] - identity["stream_start"] or result["consumed_until_exclusive"] != identity["cut"]:
        raise ValueError("prefix locks cannot use suffix rows or null/confirmation data")
    proof = {"version": 6, "source_sha256": result["source_sha256"], "data_identity": identity,
             "raw_prefix_sha256": frontier.raw_prefix_hash(raw), "health": result["prefix_health"],
             "locks": result["locks"], "locked_before_bar": identity["cut"]}
    path = root / "prefix_locks.json"
    with path.open("x") as stream:
        json.dump(frozen.finite_json(proof), stream, indent=2, allow_nan=False)
    result["prefix_lock_artifact"] = {"path": path.name, "sha256": frozen.file_sha256(path)}


def load_reference(path, hashes, configs):
    path = Path(path)
    source = json.loads(path.read_text())
    if (source.get("version"), source.get("stage"), source.get("view"), source.get("status")) != (6, "development", "real", "completed"):
        raise ValueError("reference must be completed real development")
    if source.get("source_sha256") != hashes or source.get("configs") != [asdict(c) for c in configs]:
        raise ValueError("reference source hashes/configs differ")
    identity = source["data_identity"]
    artifact = source["prefix_lock_artifact"]
    proof_path = path.parent / artifact["path"]
    if frozen.file_sha256(proof_path) != artifact["sha256"]:
        raise ValueError("prefix proof digest mismatch")
    proof = json.loads(proof_path.read_text())
    raw_path = path.parent / source["artifacts"]["raw_metrics"]
    if frozen.file_sha256(raw_path) != source["artifact_sha256"]["raw_metrics"]:
        raise ValueError("reference raw metric digest mismatch")
    count = identity["cut"] - identity["stream_start"]
    with np.load(raw_path, allow_pickle=False) as stored:
        if not np.array_equal(stored["bar_index"][:count], np.arange(identity["stream_start"], identity["cut"])):
            raise ValueError("reference prefix bar indices differ")
        raw = np.stack([stored[key][:count] for key in METRICS], axis=-1)
    if (proof.get("version") != 6 or proof.get("source_sha256") != hashes or proof.get("data_identity") != identity
            or proof.get("locked_before_bar") != identity["cut"] or proof.get("raw_prefix_sha256") != frontier.raw_prefix_hash(raw)
            or proof.get("locks") != source["locks"] or proof.get("health") != source["prefix_health"]):
        raise ValueError("reference prefix provenance mismatch")
    locks = select_locks(raw, configs, identity["stream_start"], identity["selection_start"], identity["cut"], identity["cut"], proof["health"])
    if frozen.finite_json(locks) != source["locks"]:
        raise ValueError("locks disagree with prefix-only normalized MSE selection")
    return locks, {"path": str(path.resolve()), "sha256": frozen.file_sha256(path), "prefix_sha256": artifact["sha256"],
                   "development_identity": identity, "development_consumed_until_exclusive": source["consumed_until_exclusive"]}


def selected_columns(locks, learned_count):
    selected = {name: row["column"] for name, row in locks.items() if row["column"] is not None}
    selected.update({name: learned_count + i for i, name in enumerate(REFERENCE_NAMES)})
    return selected


def summary(raw, start, low, high, locks, learned_count):
    selected = selected_columns(locks, learned_count)
    rows = reduce_metrics(raw[low - start:high - start])
    result = {"start_inclusive": low, "stop_exclusive": high, "all_candidates": rows,
              "selected": {g: rows[i] for g, i in selected.items()}, "paired_blocks": []}
    for block in range(low, high, 4096):
        end = min(high, block + 4096)
        metrics = reduce_metrics(raw[block - start:end - start])
        comparisons = {}
        for group in GROUPS:
            if group not in selected:
                continue
            comparisons[group] = {unit: {other: {
                "primary_sse": metrics[selected[group]][unit]["residual_sse"],
                "comparator_sse": metrics[i][unit]["residual_sse"],
                "count": metrics[i][unit]["count"],
                "mse_difference": metrics[selected[group]][unit]["mse"] - metrics[i][unit]["mse"]}
                for other, i in selected.items() if other != group} for unit in ("normalized", "raw")}
        result["paired_blocks"].append({"start_inclusive": block, "stop_exclusive": end, "comparisons": comparisons})
    return result


def persist_progress(root, result, journal, predictions):
    journal.flush()
    if predictions is not None:
        predictions.flush()
    frozen.save_json(root / "results.json", result)


def interrupt_run(signum, frame):
    raise KeyboardInterrupt(f"received signal {signum}")


def stream_identity(bank, runner):
    identity = frontier.stream_identity(bank, runner.target, runner.valid)
    identity.update({name: array_identity(tensor.detach().cpu().numpy()) for name, tensor in {
        "raw_target": runner.raw_target, "denominator": runner.denominator, "vol_target": runner.vol_target,
        "ridge_signed": runner.ridge.signed, "ridge_cross": runner.ridge.cross,
        "observed_valid": runner.state.observed_valid}.items()})
    return identity


def main():
    a = tyro.cli(Args)
    validate_args(a)
    root = Path(a.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("output must be empty; no resume")
    result = {"version": 6, "stage": a.stage, "view": a.view, "status": "initializing", "args": asdict(a),
        "source_sha256": source_hashes(), "locks": {}, "primary": PRIMARY,
        "protocol": {"target": "r[t+1]/max(trailing_RMS[t],1e-6), uncentered/unclipped; raw forecast=prediction*denominator[t]",
        "auxiliary": "min(bank.zt[t+1,1:]^2,25), uncentered; volatility is a teaching signal, never primary evidence",
        "information": "Frozen clipped Bank features; old t-1..t-32, latest t..t-31, memory EWMAs through t",
        "ridge": "Fixed pooled ridge1 including intercept; 12 own/SPY/cross signed cumulative horizons1/4/16/32; fit first60%, zero prefix forecast, frozen suffix",
        "selection": "Independent per-group LR/beta2/aux-weight lock on40-60% normalized MSE only; raw MSE never selects",
        "null": f"Independent signed-label Rademacher signs: {SIGN_NAMESPACE}; paired raw/normalized signs, auxiliary unchanged, real locks inherited",
        "scope": "All historical markets already research-consumed; no globally pristine holdout, profitability or exact five-minute claim",
        "confirmation": "rank400 trained from start with development locks; historical authentication; scores only[90145,T-1)",
        "runtime": "seed1 CUDA FP32 learners/FP64 metrics; TF32 off; fullgraph and CUDA graphs1/8/16; no eager fallback or resume"}}
    writer = SummaryWriter(str(root))
    runner = predictions = journal = None
    started = time.perf_counter()
    previous_terminate = signal.signal(signal.SIGTERM, interrupt_run)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
        if Path(a.cache).exists():
            with np.load(a.cache, allow_pickle=False) as cached:
                if int(cached["rank_offset"]) != a.rank_offset or int(cached["n_stocks"]) != a.n_stocks:
                    raise ValueError("refusing to overwrite another cohort cache")
        bank = panel_hd.Bank(a)
        close, volume, symbols, ts = build_panel(a)
        series = return_series(a, close)
        identity = data_identity(a, bank, close, symbols, ts, series)
        result["data_identity"] = identity
        if not np.array_equal(series["valid"], bank.valid.cpu().numpy()):
            raise ValueError("signed-return mask differs from Bank current/next mask")
        np.savez(root / "return_arrays.npz", raw_return=series["raw_return"], scale=series["scale"],
                 denominator=series["denominator"], retained_timestamps=ts[series["keep"]])
        result["artifacts"] = {"return_arrays": "return_arrays.npz", "raw_metrics": "raw_metrics.npz"}
        configs = configurations()
        result["configs"] = [asdict(c) for c in configs]
        if a.reference_result:
            result["locks"], result["lock_source"] = load_reference(a.reference_result, result["source_sha256"], configs)
            previous = result["lock_source"]["development_identity"]
            if a.stage == "development" and previous != identity:
                raise ValueError("null must match real data identity")
            if a.stage == "confirmation":
                overlap = sorted(set(previous["symbols"]) & set(identity["symbols"]))
                if overlap or bank.T - 1 <= 90145:
                    raise ValueError("confirmation requires disjoint stocks and suffix beyond90145")
                result["cohort_overlap_with_development"] = overlap
                historical_path = Path(a.historical_result)
                historical = json.loads(historical_path.read_text())
                result["historical_authentication"] = memory_eval.authenticate_confirmation(identity, historical)
                result["historical_source"] = {"path": str(historical_path.resolve()), "sha256": frozen.file_sha256(historical_path)}
            with (root / "inherited_locks.json").open("x") as stream:
                json.dump(frozen.finite_json({"locks": result["locks"], "source": result["lock_source"],
                    "target_identity": identity, "historical_authentication": result.get("historical_authentication")}), stream, indent=2, allow_nan=False)
            result["inherited_lock_sha256"] = frozen.file_sha256(root / "inherited_locks.json")
        if a.view == "rademacher":
            series["normalized_target"], series["raw_target"], result["rademacher_signs"] = randomized_labels(series["normalized_target"], series["raw_target"])
        state = CausalState(bank, torch.tensor(series["observed_valid"], device=bank.dev))
        # Its unused volatility-reference coordinates must not contain prefix mu.
        state.own.zero_()
        state.cross.zero_()
        for t in range(bank.L + 1):
            state.observe(torch.tensor(t, device=bank.dev))
        del close, volume
        model = Learner(bank.F, a.width, configs, bank.dev, bins=a.bins, seed=a.seed, num_samples=bank.N)
        score_start, score_stop = (bank.L + 1 if a.stage == "development" else 90145), bank.T - 1
        runner = Runner(bank, model, series, state, a.log_every, score_start, score_stop)
        observed_masks = series["valid"][runner.start:runner.stop]
        return_updates = np.concatenate(([0], np.cumsum(observed_masks.any(1), dtype=np.int64)))
        auxiliary_updates = []
        for group in model.groups:
            if group.auxiliary:
                paired = observed_masks[:, group.permutation.cpu().numpy()] if group.permuted else observed_masks
                auxiliary_updates.append(np.concatenate(([0], np.cumsum((observed_masks & paired).any(1), dtype=np.int64))))
            else:
                auxiliary_updates.append(None)
        result.update(output_names=list(runner.output_names), metric_names=list(METRICS), costs=model.costs,
            parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters),
            mutable_bytes=sum(t.numel() * t.element_size() for t in runner.mutable),
            consumed_until_exclusive=runner.start, score_start_inclusive=score_start, score_stop_exclusive=score_stop,
            prediction_valid_until_exclusive=None)
        result["stream_sha256_before"] = stream_identity(bank, runner)
        journal = np.lib.format.open_memmap(root / "raw_metric_journal.npy", mode="w+", dtype=np.float64,
            shape=(score_stop - score_start, len(runner.output_names), len(METRICS)))
        result["artifacts"]["raw_metric_journal"] = "raw_metric_journal.npy"
        result["checkpoint_raw_bounds"] = [score_start, score_start]
        persist_progress(root, result, journal, None)
        graphs = runner.capture()
        result.update(status="running", compile_seconds=runner.compile_seconds, capture_seconds=runner.capture_seconds,
                      full_state_capture_verified=True, replay_seconds=0.)
        host_raw = np.zeros(tuple(runner.raw.shape), dtype=np.float64)
        prediction_start = bank.cut if a.stage == "development" else score_start
        prediction_columns = None

        def open_selected_predictions():
            columns = sorted(set(selected_columns(result["locks"], len(configs)).values()))
            result.update(prediction_columns=columns, prediction_output_names=[runner.output_names[i] for i in columns],
                prediction_start_inclusive=prediction_start, prediction_valid_until_exclusive=prediction_start,
                prediction_units="unclipped normalized signed mean; multiply return_arrays denominator[bar] for raw log-return forecast")
            result["artifacts"]["selected_predictions"] = "selected_predictions.npy"
            storage = np.lib.format.open_memmap(root / "selected_predictions.npy", mode="w+", dtype=np.float64,
                shape=(score_stop - prediction_start, len(columns), bank.N))
            return storage, torch.tensor(columns, device=bank.dev)

        if result["locks"]:
            predictions, prediction_columns = open_selected_predictions()
        boundaries = sorted(set(range(runner.start + a.log_every, runner.stop, a.log_every)) | {bank.cut, score_start, score_stop})
        consumed = runner.start
        health = model.candidate_finite().cpu().numpy()
        for end in boundaries:
            if end <= consumed:
                continue
            before = consumed
            tick = time.perf_counter()
            blocks, remain = divmod(end - consumed, 16)
            for _ in range(blocks):
                graphs[16].replay()
            if remain >= 8:
                graphs[8].replay()
                remain -= 8
            for _ in range(remain):
                graphs[1].replay()
            torch.cuda.synchronize()
            result["replay_seconds"] += time.perf_counter() - tick
            consumed = int(runner.index.cpu())
            if consumed != end or int(state.observed_steps.cpu()) != consumed:
                raise RuntimeError("causal state/graph index clock disagreement")
            updates = consumed - runner.start
            for group, auxiliary_count in zip(model.groups, auxiliary_updates):
                if not np.all(group.adam_steps.cpu().numpy() == return_updates[updates]):
                    raise RuntimeError("signed-return optimizer clocks disagree")
                if auxiliary_count is not None:
                    expected = auxiliary_count[updates] * np.asarray([c.aux_weight > 0 for c in group.configs])
                    if not np.array_equal(group.auxiliary_steps.cpu().numpy(), expected):
                        raise RuntimeError("auxiliary optimizer clocks disagree")
            result["consumed_until_exclusive"] = consumed
            low, high = max(before, score_start), min(end, score_stop)
            if low < high:
                host_raw[low - runner.start:high - runner.start] = runner.raw[low - runner.start:high - runner.start].cpu().numpy()
                journal[low - score_start:high - score_start] = host_raw[low - runner.start:high - runner.start]
                export_low = max(low, prediction_start)
                if predictions is not None and export_low < high:
                    assert prediction_columns is not None
                    indices = (torch.arange(export_low, high, device=bank.dev) - runner.start) % a.log_every
                    selected = runner.prediction_ring.index_select(0, indices).index_select(1, prediction_columns)
                    predictions[export_low - prediction_start:high - prediction_start] = selected.cpu().numpy()
                    result["prediction_valid_until_exclusive"] = high
            health = model.candidate_finite().cpu().numpy()
            result["candidate_health"] = health.tolist()
            if end == bank.cut:
                if a.stage == "development" and a.view == "real":
                    prefix = host_raw[:bank.cut - runner.start]
                    result["prefix_health"] = health.tolist()
                    result["locks"] = select_locks(prefix, configs, runner.start, identity["selection_start"], bank.cut, end, health)
                    persist_prefix_locks(root, prefix, result)
                    predictions, prediction_columns = open_selected_predictions()
                runner.ridge.fit()
                result["ridge_reference"] = {"penalty": 1., "intercept_penalized": True,
                    "fit_start_inclusive": runner.start, "fit_stop_exclusive": bank.cut,
                    "horizons": HORIZONS, "coefficient": runner.ridge.coefficient.cpu().tolist(),
                    "prefix_predictions": "zero", "suffix_coefficients": "frozen"}
            if low < high:
                interval = reduce_metrics(host_raw[low - runner.start:high - runner.start])
                for name, row in zip(runner.output_names, interval):
                    for unit in ("normalized", "raw"):
                        writer.add_scalar(f"{a.view}/{name}/{unit}_mse", row[unit]["mse"], high)
                writer.flush()
            result["checkpoint_raw_bounds"] = [score_start, max(score_start, min(consumed, score_stop))]
            persist_progress(root, result, journal, predictions)
            print(json.dumps({"status": "running", "next_bar": consumed, "score_stop": score_stop}), flush=True)
        regions = {"confirmation": (score_start, score_stop)} if a.stage == "confirmation" else {
            "fit": (runner.start, identity["selection_start"]), "selection": (identity["selection_start"], bank.cut),
            "prefix": (runner.start, bank.cut), "suffix": (bank.cut, score_stop), "full": (runner.start, score_stop)}
        result["phase_metrics"] = {name: summary(host_raw, runner.start, low, high, result["locks"], len(configs)) for name, (low, high) in regions.items()}
        result["locked_candidate_status"] = {g: "failed_no_prefix_candidate" if row["column"] is None else
            ("finite" if health[row["column"]] and np.isfinite(host_raw[:, row["column"]]).all() else "nonfinite")
            for g, row in result["locks"].items()}
        result["stream_sha256_after"] = stream_identity(bank, runner)
        result["source_sha256_after"] = source_hashes()
        result["cache_sha256_after"] = frozen.file_sha256(a.cache)
        if (result["stream_sha256_before"] != result["stream_sha256_after"] or result["source_sha256_after"] != result["source_sha256"]
                or result["cache_sha256_after"] != identity["cache_sha256"]):
            raise RuntimeError("source/data/stream bytes changed")
        result.update(status="completed", source_and_stream_unchanged=True)
    except KeyboardInterrupt as error:
        result.update(status="interrupted", failure={"type": type(error).__name__, "message": str(error)})
        raise
    except Exception as error:
        result.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        try:
            if runner is not None:
                consumed = int(runner.index.cpu())
                result["consumed_until_exclusive"] = consumed
                low, high = runner.score_start, max(runner.score_start, min(consumed, runner.score_stop))
                raw = runner.raw[low - runner.start:high - runner.start].cpu().numpy()
                np.savez_compressed(root / "raw_metrics.npz", bar_index=np.arange(low, high),
                    output_names=np.asarray(runner.output_names),
                    normalized_residual_sse=raw[:, :, 0], normalized_prediction_energy=raw[:, :, 1],
                    normalized_cross_sum=raw[:, :, 2], normalized_target_energy=raw[:, :, 3], normalized_count=raw[:, :, 4],
                    raw_residual_sse=raw[:, :, 5], raw_prediction_energy=raw[:, :, 6],
                    raw_cross_sum=raw[:, :, 7], raw_target_energy=raw[:, :, 8], raw_count=raw[:, :, 9])
                result["raw_metric_bounds"] = [low, high]
                if journal is not None:
                    journal[:high - low] = raw
                    result["checkpoint_raw_bounds"] = [low, high]
                    persist_progress(root, result, journal, predictions)
                # Ring pages not exported before interruption are not advertised.
                result["artifact_sha256"] = {key: frozen.file_sha256(root / filename) for key, filename in result.get("artifacts", {}).items()
                    if (root / filename).is_file()}
        finally:
            frozen.save_json(root / "results.json", result)
            writer.close()
            signal.signal(signal.SIGTERM, previous_terminate)


if __name__ == "__main__":
    main()
