"""Tabulate 14_gate1_results_*.json: gain [CI], cosine and P(better than the critic alone) per stage, space, reference."""
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data"
ROWS = ("critic", "sf_gae0.95", "unit_sum", "unit_sum_mb", "unit_sum_gae0.8", "opt_insample", "opt_half", "unit_sum_half", "crit_norm_pre",
        "crit_norm_act", "crit_clip_pre", "crit_clip_act", "crit_drop1", "unit_sum_clippre", "critic_perm", "sf_perm", "unit_sum_critperm")
EXTRA = ("rho_noise", "rho_error", "noise_share_critic", "noise_share_sf", "cos_realized_critic_sf", "expectation_cos_critic_sf",
         "unit_sum_cos_if_independent", "unit_sum_cos_measured")
naive = "naive" in sys.argv[1:]
for name in ("v23_4M", "v29h8_8M", "v23_12M", "v29h8_12M", "v23_30M"):
    path = OUT / f"14_gate1_results_{name}.json"
    if not path.exists():
        continue
    res = json.loads(path.read_text())
    for space in ("linear(108)", "lastlayer(3078)", "actor_params(full)"):
        for ref in ("sf_gae0.99", "sf_gae0.95"):
            tag = f"{name}|{space}" + ("|naive-norm" if naive else "") + f"|{ref}"
            print(f"\n## {tag}")
            for row in ROWS:
                r = res[f"{tag}|{row}"]
                g = r["gain"]
                cos = r.get("cos")
                cos_text = f"cos {cos['point']:+.2f} [{cos['ci'][0]:+.2f}, {cos['ci'][1]:+.2f}]" if cos else "cos   (8-lane halves)   "
                print(f"  {row:18s} gain {g['point']:+.3f} [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}]  {cos_text}  P(better) {g['p_better']:.2f}")
            print("  " + "; ".join(f"{k} {res[f'{tag}|{k}']['point']:+.2f} [{res[f'{tag}|{k}']['ci'][0]:+.2f}, {res[f'{tag}|{k}']['ci'][1]:+.2f}]" for k in EXTRA))
