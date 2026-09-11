"""Post-hoc development diagnosis, not a new learner or held-out result."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path("runs/panel_return_hierarchy_v7_development")
OUT = Path("runs/panel_return_hierarchy_v7_leverage_diagnostic")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    OUT.mkdir(exist_ok=False)
    result = json.loads((ROOT / "results.json").read_text())
    if result["status"] != "completed":
        raise RuntimeError("development evidence incomplete")
    with np.load(ROOT / "return_arrays.npz") as arrays:
        raw = torch.as_tensor(arrays["raw_return"], dtype=torch.float64, device="cuda")
        scale = torch.as_tensor(arrays["scale"], dtype=torch.float64, device="cuda").clamp_min(1e-6)
        denominator = torch.as_tensor(arrays["denominator"], dtype=torch.float64, device="cuda")
    signed = raw / torch.cat((torch.full_like(scale[:1], torch.nan), scale[:-1]))
    observed = torch.isfinite(signed[:, 1:])
    valid = observed & torch.roll(observed, -1, 0)
    valid &= torch.isfinite(denominator)
    valid[:result["data_identity"]["stream_start"]] = False
    valid[-1] = False
    signed = torch.where(torch.isfinite(signed), signed, 0.)
    cross = torch.where(observed, signed[:, 1:], 0.).sum(1) / observed.sum(1).clamp_min(1)
    # Algebraically equivalent rolling sums; used only for diagnostic bins.
    cumulative = signed.cumsum(0)
    cross_cumulative = cross.cumsum(0)
    magnitude = torch.ones_like(signed[:, 1:])
    for horizon in (1, 4, 16, 32):
        window = cumulative - torch.cat((torch.zeros_like(cumulative[:horizon]), cumulative[:-horizon]))
        cross_window = cross_cumulative - torch.cat((torch.zeros_like(cross_cumulative[:horizon]), cross_cumulative[:-horizon]))
        magnitude = torch.maximum(magnitude, window[:, 1:].abs())
        magnitude = torch.maximum(magnitude, window[:, :1].abs())
        magnitude = torch.maximum(magnitude, cross_window[:, None].abs())
    start = result["prediction_start_inclusive"]
    stop = result["prediction_valid_until_exclusive"]
    mask = valid[start:stop]
    target = torch.where(mask, raw[start + 1:stop + 1, 1:], 0.)
    target_energy = target.square().sum().item()
    expected = result["phase_metrics"]["suffix"]["selected"]["zero_return"]["raw"]
    if mask.sum().item() != expected["count"] or not np.isclose(target_energy, expected["target_energy"], rtol=1e-10, atol=1e-10):
        raise RuntimeError("reconstructed targets differ from recorded metrics")
    summary = {"status": "completed", "role": "posthoc_development_diagnostic", "source_result": str(ROOT / "results.json"),
               "max_absolute_feature": magnitude[valid].max().item(), "suffix_count": expected["count"],
               "raw_target_energy": target_energy, "families": {}}
    predictions = np.load(ROOT / "selected_predictions.npy", mmap_mode="r")
    for family in ("pooled_ridge", "independent_ridge", "hierarchical_ridge"):
        column = result["prediction_columns"].index(result["locks"][family]["column"])
        prediction = torch.as_tensor(predictions[:, column, :], dtype=torch.float64, device="cuda") * denominator[start:stop]
        prediction = torch.where(mask, prediction, 0.)
        energy = prediction.square()
        excess = energy - 2 * prediction * target
        recorded = result["phase_metrics"]["suffix"]["selected"][family]["raw"]
        if not np.isclose(energy.sum().item(), recorded["prediction_energy"], rtol=1e-9, atol=1e-10):
            raise RuntimeError("prediction units or columns disagree with recorded metrics")
        summary["families"][family] = {"prediction_energy": energy.sum().item(), "excess_sse": excess.sum().item(), "leverage_bins": {}}
        for threshold in (10, 100, 1000):
            selected = mask & (magnitude[start:stop] > threshold)
            summary["families"][family]["leverage_bins"][str(threshold)] = {
                "count": selected.sum().item(), "prediction_energy": energy[selected].sum().item(),
                "excess_sse": excess[selected].sum().item(), "target_energy": target[selected].square().sum().item()}
    with np.load(ROOT / "posterior_state.npz") as state:
        gram = torch.as_tensor(state["stock_gram"], dtype=torch.float64, device="cuda")
        summary["largest_stock_gram_diagonal"] = gram.diagonal(dim1=-2, dim2=-1).max().item()
    with SummaryWriter(str(OUT)) as writer:
        writer.add_scalar("diagnostic/max_absolute_feature", summary["max_absolute_feature"], stop)
        for family, row in summary["families"].items():
            writer.add_scalar(f"diagnostic/{family}/raw_excess_sse", row["excess_sse"], stop)
    (OUT / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
