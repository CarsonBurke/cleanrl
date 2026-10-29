"""Actual-shape r/ell credit from saved conditional physical PG; CUDA/mlq only.

No actor, environment, fit, or candidate selection. See
benchmarks/sepclip_weight_microscopy/analytical_chart_identification.md.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cleanrl.shared.runtime import configure_runtime


FIELD_SCOPE = {
    "finite_raw": "Finite discounted raw return, zero tail, continuation policy fixed",
    "finite_normalized": "Finite discounted frozen-normalized/clipped return, zero tail, continuation policy fixed",
    "bootstrap_normalized": "Frozen-normalized/clipped return plus estimated old-critic terminal tail, continuation policy fixed",
}


def serial(value):
    """Strict JSON: withheld/nonfinite entries become null, never clamped."""
    if isinstance(value, torch.Tensor):
        return serial(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def actual_geometry(alpha, beta):
    """Smooth interior geometry evaluated at unchanged recorded shape values."""
    if alpha.dtype != torch.float64 or beta.dtype != torch.float64:
        raise ValueError("Promote actual FP32 shape values unchanged to FP64 first")
    if alpha.shape != beta.shape or not bool(torch.isfinite(alpha).all() & torch.isfinite(beta).all()):
        raise ValueError("Invalid shape arrays")
    if not bool(((alpha > 1) & (beta > 1)).all()):
        raise ValueError("Actual-shape chart requires alpha>1 and beta>1; no boundary clamp")
    x, y = alpha - 1, beta - 1
    tau = x + y
    q = x / tau
    radial = x * y / tau
    jacobian = torch.stack((torch.stack((radial, x), -1),
                            torch.stack((-radial, y), -1)), -2)
    common = torch.polygamma(1, alpha + beta)
    fisher = torch.stack((torch.stack((torch.polygamma(1, alpha) - common, -common), -1),
                          torch.stack((-common, torch.polygamma(1, beta) - common), -1)), -2)
    chart_fisher = jacobian.mT @ fisher @ jacobian
    return dict(x=x, y=y, tau=tau, q=q, r=x.log()-y.log(), ell=tau.log(),
                jacobian=jacobian, shape_fisher=fisher, chart_fisher=chart_fisher)


def transform_credit(logit_pg, sigmoid, geometry):
    """Hybrid smooth-logit chain: recorded-logit sigmoid, actual-shape L/F."""
    if logit_pg.shape[-1] != 2 or sigmoid.shape[-1] != 2:
        raise ValueError("Last axis must be paired alpha/beta logits")
    if not bool(torch.isfinite(logit_pg).all() & torch.isfinite(sigmoid).all()) or not bool((sigmoid > 0).all()):
        raise ValueError("Nonfinite credit or zero smooth-logit derivative; withhold instead of clamp")
    shape_pg = logit_pg / sigmoid
    chart_pg = (geometry["jacobian"].mT @ shape_pg.unsqueeze(-1)).squeeze(-1)
    shape_natural = torch.linalg.solve(geometry["shape_fisher"], shape_pg.unsqueeze(-1)).squeeze(-1)
    chart_natural = torch.linalg.solve(geometry["chart_fisher"], chart_pg.unsqueeze(-1)).squeeze(-1)
    shape_dual = (shape_pg * shape_natural).sum(-1)
    chart_dual = (chart_pg * chart_natural).sum(-1)
    mapped = (geometry["jacobian"] @ chart_natural.unsqueeze(-1)).squeeze(-1)
    scale = torch.diag_embed(sigmoid)
    logit_fisher = scale @ geometry["shape_fisher"] @ scale
    logit_dual = (logit_pg * torch.linalg.solve(logit_fisher, logit_pg.unsqueeze(-1)).squeeze(-1)).sum(-1)
    return dict(shape_pg=shape_pg, chart_pg=chart_pg,
        unit_fisher_coordinate_slopes=chart_pg/geometry["chart_fisher"].diagonal(dim1=-2, dim2=-1).sqrt(),
        chart_dual_energy=chart_dual, shape_dual_energy=shape_dual, logit_dual_energy=logit_dual,
        duality_absolute_error=(chart_dual-shape_dual).abs(),
        duality_relative_error=torch.where(shape_dual > 0, (chart_dual-shape_dual).abs()/shape_dual, (chart_dual-shape_dual).abs()),
        logit_duality_absolute_error=(logit_dual-shape_dual).abs(),
        natural_shape_direction_absolute_error=(mapped-shape_natural).norm(dim=-1),
        natural_shape_direction_relative_error=torch.where(shape_natural.norm(dim=-1) > 0,
            (mapped-shape_natural).norm(dim=-1)/shape_natural.norm(dim=-1), (mapped-shape_natural).norm(dim=-1)))


def paired(values):
    return torch.stack(values.chunk(2, -1), -1)


def analyze_row(row, device="cuda"):
    if row["oracle"].get("first_action_only") is not True:
        raise ValueError("Require explicit first_action_only=True metadata")
    initial = row["initial_policy"]
    logits32 = torch.as_tensor(initial["logits"], dtype=torch.float32, device=device)
    alpha32 = torch.as_tensor(initial["alpha"], dtype=torch.float32, device=device)
    beta32 = torch.as_tensor(initial["beta"], dtype=torch.float32, device=device)
    if alpha32.ndim != 2 or beta32.ndim != 2 or logits32.ndim != 2 or logits32.shape != (alpha32.shape[0], 2*alpha32.shape[1]) or alpha32.shape != beta32.shape or alpha32.shape[1] == 0:
        raise ValueError("Initial policy must have Sx2D logits and SxD shapes")
    if not bool(torch.isfinite(logits32).all()):
        raise ValueError("Nonfinite recorded initial policy logits")
    expected_shapes = paired(1+F.softplus(logits32))
    recorded_shapes = torch.stack((alpha32, beta32), -1)
    upper_ulp = torch.nextafter(expected_shapes, torch.full_like(expected_shapes, float("inf"))) - expected_shapes
    lower_ulp = expected_shapes-torch.nextafter(expected_shapes, torch.full_like(expected_shapes, -float("inf")))
    ulp = torch.where(torch.isfinite(upper_ulp), torch.maximum(upper_ulp, lower_ulp), lower_ulp)
    shape_error = (recorded_shapes-expected_shapes).abs()
    if not bool(torch.isfinite(recorded_shapes).all()) or not bool((shape_error <= 2*ulp).all()):
        raise ValueError("Recorded shapes disagree with FP32 1+softplus(logits) beyond two ULP")
    selected = row.get("selected_states")
    if not isinstance(selected, list) or len(selected) != alpha32.shape[0] or not selected or not all(isinstance(state, dict) for state in selected):
        raise ValueError("Selected state count must match initial policy state count")
    if not all(all(type(state.get(key)) is int and state[key] >= 0 for key in ("t", "row", "episode_age")) for state in selected):
        raise ValueError("Require selected state t,row,episode_age integer identity metadata")
    horizons = row["oracle"].get("horizons")
    if not isinstance(horizons, list) or not horizons or not all(isinstance(horizon, dict) for horizon in horizons):
        raise ValueError("Require nonempty physical PG horizon objects")
    alpha, beta = alpha32.double(), beta32.double()
    sigmoid = paired(logits32.double().sigmoid())
    latent = paired(F.softplus(logits32)).double()
    actual_excess = torch.stack((alpha-1, beta-1), -1)
    positive_excess = actual_excess > 0
    relative_excess_difference = torch.full_like(actual_excess, float("nan"))
    relative_excess_difference[positive_excess] = (latent-actual_excess)[positive_excess]/actual_excess[positive_excess]
    interior = (alpha > 1) & (beta > 1) & torch.isfinite(alpha) & torch.isfinite(beta)
    latent_interior = (latent > 0).all(-1) & torch.isfinite(latent).all(-1)
    actual_chart = torch.full((*alpha.shape, 2), float("nan"), dtype=torch.float64, device=device)
    latent_chart = torch.full_like(actual_chart, float("nan"))
    selected_excess = actual_excess[interior]
    selected_latent = latent[latent_interior]
    actual_chart[interior] = torch.stack((selected_excess[..., 0].log()-selected_excess[..., 1].log(), selected_excess.sum(-1).log()), -1)
    latent_chart[latent_interior] = torch.stack((selected_latent[..., 0].log()-selected_latent[..., 1].log(), selected_latent.sum(-1).log()), -1)
    usable = interior & torch.isfinite(paired(logits32)).all(-1) & torch.isfinite(sigmoid).all(-1) & (sigmoid > 0).all(-1)
    interior_logit_mask = usable.clone()
    geometry = actual_geometry(alpha[usable], beta[usable]) if bool(usable.any()) else None
    raw_geometry = geometry
    fisher_withheld = torch.zeros_like(usable)
    logit_fisher_withheld = torch.zeros_like(usable)
    if geometry is not None:
        # Diagnose loss of SPD in the declared FP64 arithmetic, rather than
        # silently regularizing or inventing precision in an extreme shape.
        _, shape_info = torch.linalg.cholesky_ex(geometry["shape_fisher"])
        _, chart_info = torch.linalg.cholesky_ex(geometry["chart_fisher"])
        logit_scale = torch.diag_embed(sigmoid[usable])
        logit_fisher = logit_scale @ geometry["shape_fisher"] @ logit_scale
        geometry["smooth_logit_fisher"] = logit_fisher
        _, logit_info = torch.linalg.cholesky_ex(logit_fisher)
        finite_logit = torch.isfinite(logit_fisher).all((-1, -2))
        finite = torch.isfinite(geometry["shape_fisher"]).all((-1, -2)) & torch.isfinite(geometry["chart_fisher"]).all((-1, -2)) & finite_logit
        logit_fisher_withheld[usable] = (logit_info != 0) | ~finite_logit
        bad = (shape_info != 0) | (chart_info != 0) | (logit_info != 0) | ~finite
        fisher_withheld[usable] = bad
        usable = usable & ~fisher_withheld
        geometry = actual_geometry(alpha[usable], beta[usable]) if bool(usable.any()) else None
    def scatter(values, mask=None):
        if mask is None:
            mask = usable
        result = torch.full((*usable.shape, *values.shape[1:]), float("nan"), dtype=torch.float64, device=device)
        result[mask] = values
        return result
    static = {key: scatter(value, interior_logit_mask) for key, value in raw_geometry.items()} if raw_geometry is not None else {}
    output = dict(snapshot=row.get("snapshot"), snapshot_sha256=row.get("snapshot_sha256"),
        global_step=row.get("global_step"), actual_fp32_alpha=alpha, actual_fp32_beta=beta,
        shape_logit_replay_max_abs_error=float(shape_error.max()),
        shape_logit_replay_tolerance="Two FP32 ULP of separately reevaluated 1+softplus(recorded logits), permitting fused/eager arithmetic differences; recorded shapes are never replaced",
        selected_state_coordinates=[{key: state.get(key) for key in ("t", "row", "episode_age")} for state in selected],
        input_normalization_scope=row["oracle"].get("normalization"),
        input_first_action_sampling=row.get("first_action_sampling"),
        coordinate_order=["mode_logit_r", "log_excess_concentration_ell"],
        paired_logit_order=["alpha", "beta"], actual_shape_interior=interior,
        usable_chart_mask=usable, withheld_state_action_coordinates=(~usable).nonzero(),
        numerical_fisher_withheld_coordinates=fisher_withheld.nonzero(),
        numerical_smooth_logit_fisher_withheld_coordinates=logit_fisher_withheld.nonzero(),
        shape_one_coordinates=((alpha == 1) | (beta == 1)).nonzero(),
        shape_below_one_coordinates=((alpha < 1) | (beta < 1)).nonzero(),
        nonfinite_shape_coordinates=(~(torch.isfinite(alpha) & torch.isfinite(beta))).nonzero(),
        nonpositive_shape_coordinates=((alpha <= 0) | (beta <= 0)).nonzero(),
        latent_pre_add_fp32_softplus_excess=latent, actual_rounded_excess=actual_excess,
        latent_minus_actual_excess=latent-actual_excess,
        latent_minus_actual_relative_excess_on_positive_actual=relative_excess_difference,
        actual_shape_chart_r_ell=actual_chart, latent_pre_add_chart_r_ell=latent_chart,
        latent_minus_actual_chart_r_ell=latent_chart-actual_chart,
        geometry=static, horizons=[])
    for horizon in horizons:
        fields = {}
        for label, saved in horizon["policy_logit_pg"].items():
            if label not in FIELD_SCOPE:
                raise ValueError(f"Unknown objective label {label}; do not silently relabel")
            mean = paired(torch.as_tensor(saved["mean"], dtype=torch.float64, device=device))
            replicas = paired(torch.as_tensor(saved["replicate_values"], dtype=torch.float64, device=device))
            if mean.shape != (*alpha.shape, 2) or replicas.ndim != 4 or replicas.shape[1:] != mean.shape or len(replicas) < 2:
                raise ValueError("Require >=2 complete Sx2D PG replicas")
            if not bool(torch.isfinite(mean).all() & torch.isfinite(replicas).all()):
                raise ValueError("Nonfinite saved physical PG")
            replica_mean = replicas.mean(0)
            mean_error = float((mean-replica_mean).abs().max())
            if not torch.allclose(mean, replica_mean, rtol=1e-10, atol=1e-10):
                raise ValueError("Saved mean does not agree with its PG replicas")
            info = dict(objective_scope=FIELD_SCOPE[label], original_logit_pg_mean=mean,
                original_logit_pg_replicas=replicas, replica_count=len(replicas),
                saved_mean_replica_max_abs_error=mean_error)
            if geometry is not None:
                transformed = transform_credit(mean[usable], sigmoid[usable], geometry)
                converted = [transform_credit(replica[usable], sigmoid[usable], geometry) for replica in replicas]
                finite_credit = torch.ones(int(usable.sum()), dtype=torch.bool, device=device)
                for evaluation in (transformed, *converted):
                    for value in evaluation.values():
                        finite_credit &= torch.isfinite(value).flatten(1).all(-1) if value.ndim > 1 else torch.isfinite(value)
                def withhold(values):
                    return torch.where(finite_credit.reshape((-1,)+(1,)*(values.ndim-1)), values, float("nan"))
                field_mask = torch.zeros_like(usable)
                field_mask[usable] = finite_credit
                info["mean_field"] = {key: scatter(withhold(value)) for key, value in transformed.items()}
                chart_replicas = torch.stack([scatter(withhold(value["chart_pg"])) for value in converted])
                slope_replicas = torch.stack([scatter(withhold(value["unit_fisher_coordinate_slopes"])) for value in converted])
                info.update(chart_pg_replicas=chart_replicas, unit_fisher_coordinate_slope_replicas=slope_replicas,
                    chart_pg_replica_sd=chart_replicas.std(0, unbiased=True),
                    chart_pg_replica_mc_se=chart_replicas.std(0, unbiased=True)/len(replicas)**.5,
                    mean_field_dual_energy_on_usable_coordinates=float(transformed["chart_dual_energy"][finite_credit].sum()/alpha.shape[0]) if bool(finite_credit.any()) else None,
                    finite_transformed_credit_mask=field_mask,
                    nonfinite_transformed_credit_withheld_coordinates=(usable & ~field_mask).nonzero(),
                    usable_state_action_count=int(finite_credit.sum()), total_state_action_count=usable.numel(),
                    aggregate_scope="Sum action-dimension dual energies divided by ALL selected state count; withheld coordinates are absent, no completion assumed",
                    dual_energy_warning="Point-estimated noisy field energy; not true-PG signal, actor reachability, fit efficiency or candidate selection")
                if not bool(finite_credit.any()):
                    info["withheld_reason"] = "All transformed credit coordinates overflowed or became nonfinite in declared arithmetic"
            else:
                info["withheld_reason"] = "No interior actual-shape coordinates with finite nonzero smooth-logit derivative and numerically SPD shape/chart/smooth-logit Fisher"
            fields[label] = info
        output["horizons"].append(dict(horizon=horizon["horizon"], discount=horizon.get("discount"),
            tail_weight=horizon.get("tail_weight"), fields=fields))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; queue this diagnostic through mlq with maxParallelRuns1")
    source = Path(args.input)
    raw = source.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list) or not data["results"] or not all(isinstance(row, dict) for row in data["results"]):
        raise ValueError("Input must contain a nonempty list of dense PG result objects")
    result = dict(protocol=vars(args), input_sha256=hashlib.sha256(raw).hexdigest(),
        input_path=str(source.resolve()), input_source_sha256=data.get("source_sha256"),
        input_imported_source_sha256=data.get("imported_source_sha256"),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependency_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/"cleanrl/shared/runtime.py", ROOT/"benchmarks/sepclip_weight_microscopy/analytical_chart_identification.md")},
        interpretation="Hybrid smooth-logit covector conversion: FP64 sigmoid of recorded FP32 logits, exact smooth chart/Fisher at unchanged actual FP32 shapes",
        limitations=["Conditional initial-action physical PG only; continuation policy fixed, selected states and frozen normalizers",
            "Replica SD/SE describes conditional action/continuation Monte Carlo noise, not seed/state-population uncertainty",
            "Coordinate slopes are isolated unit-Fisher slopes, not an actor-realizable joint intervention",
            "Actual-shape boundary chart values are withheld without precision clamps",
            "Candidate uses pre-addition latent softplus excess; reported discrepancy prevents pretending this is bit-exact candidate arithmetic",
            "Sampler clipping/quantile-truncation score approximations are inherited from the input diagnostic"],
        results=[analyze_row(row) for row in data["results"]])
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(serial(result), indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
