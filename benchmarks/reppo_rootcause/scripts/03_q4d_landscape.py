"""Q4 (return landscape of v55-dual): re-derive noise cost, the one-iteration update slice (slope at 0, gain of the
actual step) and the critic-ascent slice from the saved per-episode returns, with episode-bootstrap CIs."""
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
RNG = np.random.default_rng(0)
BASE = REPO / "runs/landscape"


def ci(values):  # paired per-episode values -> mean and bootstrap 95% CI
    values = np.asarray(values)
    draws = values[RNG.integers(0, len(values), (4000, len(values)))].mean(1)
    return f"{values.mean():+8.1f} [{np.percentile(draws, 2.5):+8.1f},{np.percentile(draws, 97.5):+8.1f}]"


def slope_at_zero(coef, returns, window):
    """Per-episode OLS slope of return on c over |c| <= window (quadratic fit), then mean and CI over episodes."""
    coef = np.asarray(coef)
    keep = np.abs(coef) <= window
    X = np.stack([np.ones(keep.sum()), coef[keep], coef[keep] ** 2], 1)
    beta = np.linalg.lstsq(X, np.asarray(returns)[keep], rcond=None)[0]  # (3, episodes)
    return beta[1], beta[2]


for step in (1015424, 4013696, 7913088):
    print(f"\n=== snapshot {step}")
    spread = json.load(open(BASE / f"HalfCheetah-v4__reppo_landscape_ref_v55dual__1__1790711607__snapshot_{step}/spread_all.json"))
    coef, ret = np.array(spread["coefficients"]), np.array(spread["returns"])
    i0, i1 = int(np.argmin(np.abs(coef))), int(np.argmin(np.abs(coef - 1)))
    cost = ret[i0] - ret[i1]
    print(f"noise cost J(0)-J(1): {ci(cost)}  = {100 * cost.mean() / ret[i0].mean():.1f}% of J(0)={ret[i0].mean():.0f} (published 5.0/12.3/11.0%)")
    rel = cost / ret[i0].mean()
    draws = np.array([cost[i].mean() / ret[i0][i].mean() for i in RNG.integers(0, 32, (4000, 32))])
    print(f"   relative cost 95% CI [{100 * np.percentile(draws, 2.5):.1f}%, {100 * np.percentile(draws, 97.5):.1f}%]; noise amplitudes {np.round(coef, 2).tolist()}")
    for kind in ("update", "ascent"):
        path = BASE / f"HalfCheetah-v4__reppo_landscape_ref_v55dual_pairs__1__1790712519__snapshot_{step}/{kind}_all.json"
        d = json.load(open(path))
        coef = np.array(d["coefficients"])
        i0, i1 = int(np.argmin(np.abs(coef))), int(np.argmin(np.abs(coef - 1)))
        print(f" {kind} slice: cos(direction, actual update) = {d['update_cosine'][0]:.4f}; |update| {d['update_norm']:.4f}; raw direction norm {d['direction_raw_norms'][0]:.3f}")
        for scale in ("0.0", "1.0"):
            ret = np.array(d["per_scale"][scale]["returns"])
            gain = ret[i1] - ret[i0]
            best = int(np.argmax(ret.mean(1)))
            line = f"   noise x{scale}: J(c=1)-J(c=0) {ci(gain)}; best c on grid {coef[best]:+.2f} (gain {ret[best].mean() - ret[i0].mean():+.0f})"
            for window in (0.5, 1.0):
                s, c2 = slope_at_zero(coef, ret, window)
                line += f"; slope@0 |c|<={window}: {ci(s)}"
            print(line + f"  [published fit slope {d['fit'][scale]['slope']:+.1f}+-{d['fit'][scale]['slope_se']:.1f}]")
        crit = np.array(d["critic"])
        print(f"   critic surrogate along slice: predicted gain at c=1 {crit[i1] - crit[i0]:+.4f} scaled; corr(critic curve, true curve) det {d['critic_true_corr']['0.0']:+.2f} stoch {d['critic_true_corr']['1.0']:+.2f} over {len(coef)} grid points (c in [{coef.min():.1f},{coef.max():.1f}])")
        # curve correlation restricted to the local region |c| <= 1 (what an update can use)
        for scale in ("0.0",):
            true = np.array(d["per_scale"][scale]["returns"]).mean(1)
            keep = np.abs(coef) <= 1.0
            print(f"   corr(critic, true det curve) restricted to |c|<=1 ({keep.sum()} points): {np.corrcoef(crit[keep], true[keep])[0, 1]:+.2f}")
