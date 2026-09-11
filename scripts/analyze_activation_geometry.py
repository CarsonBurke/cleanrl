"""Fixed-data CUDA activation geometry measurements; run only through mlq.

No training or return evaluation. With --checkpoint, collect 4096 stochastic
on-policy observations using its frozen per-environment observation statistics.
All trunks see that identical batch and an independent standard Gaussian batch.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanrl.shared.host_actor import (
    LReluSqPair,
    SiTUGLUBranch,
    justnorm,
    make_lrelu_sphere_trunk,
    make_situ_sphere_trunk,
    situ_glu,
)
from cleanrl.shared.host_graph import make_host_mirror
from cleanrl.shared.mujoco_env import make_mujoco_vector_env
from cleanrl.shared.norm_residual import NormResidualTrunk, make_norm_residual_trunk
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import sample_beta_actions_host


SEED = 1
ROWS = 4096
NUM_ENVS = 16
ENV_ID = "HalfCheetah-v4"
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value):
    return float(value.item())


def ratio(numerator, denominator):
    # A zero-energy input has no measurable sensitivity; report 0, not NaN.
    return scalar(torch.where(denominator > 0, numerator / denominator, 0.0))


def summarize(x):
    """Empirical moments, not assumptions about utility of a channel mean."""
    x = x.double()
    energy = x.square().sum()
    centered = x - x.mean(dim=0, keepdim=True)
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered / (x.shape[0] - 1)).clamp_min(0)
    total = eigenvalues.sum()
    if scalar(total) > 0:
        probabilities = eigenvalues / total
        effective_rank = scalar(torch.exp(-(probabilities * probabilities.clamp_min(1e-300).log()).sum()))
        participation_rank = ratio(total.square(), eigenvalues.square().sum())
    else:
        effective_rank = participation_rank = 0.0
    return {
        "rms": scalar(x.square().mean().sqrt()),
        "mean_row_l2": scalar(x.norm(dim=-1).mean()),
        "zero_rows": int((x.norm(dim=-1) == 0).sum().item()),
        "channel_mean_energy_fraction": ratio(x.mean(dim=-1).square().sum() * x.shape[-1], energy),
        "dataset_mean_vector_energy_fraction": ratio(x.mean(dim=0).square().sum() * x.shape[0], energy),
        "centered_covariance_effective_rank": effective_rank,
        "centered_covariance_participation_rank": participation_rank,
        "positive_fraction": scalar((x > 0).double().mean()),
        "negative_fraction": scalar((x < 0).double().mean()),
        "zero_fraction": scalar((x == 0).double().mean()),
    }


def cosine(a, b):
    a, b = a.double(), b.double()
    valid = (a.norm(dim=-1) > 0) & (b.norm(dim=-1) > 0)
    return {
        "mean": scalar(F.cosine_similarity(a[valid], b[valid], dim=-1).mean()) if bool(valid.any()) else 0.0,
        "valid_rows": int(valid.sum().item()),
    }


def directional_probe(block, x, y, epsilon, isotropic_direction):
    """Symmetric finite differences, with equal per-row relative input steps.

    Radial/tangent splitting is exact for the measured finite output vector.
    Only in the differential limit is the radial component precisely what the
    normalization map removes; its tangent is additionally divided by ||y||.
    The directly measured normalized finite difference is reported separately.
    """
    result = {}
    for name, direction in (("positive_radial_scale", x), ("isotropic", isotropic_direction)):
        dx = epsilon * direction
        plus, minus = block(x + dx), block(x - dx)
        dy = ((plus - minus) * 0.5).double()
        unit_y = justnorm(y.double())
        radial = (dy * unit_y).sum(dim=-1, keepdim=True) * unit_y
        tangent = dy - radial
        dy_energy = dy.square().sum()
        dx_energy = dx.double().square().sum()
        y_energy = y.double().square().sum()
        normalized_dy = (justnorm(plus.double()) - justnorm(minus.double())) * 0.5
        tangent_derivative = tangent / y.double().norm(dim=-1, keepdim=True).clamp_min(1e-12)
        forward_change = plus.double() - y.double()
        result[name] = {
            "epsilon": epsilon,
            "input_relative_l2": math.sqrt(ratio(dx_energy, x.double().square().sum())),
            "raw_output_relative_l2": math.sqrt(ratio(dy_energy, y_energy)),
            "raw_relative_sensitivity": math.sqrt(ratio(dy_energy, y_energy)) / epsilon,
            "raw_absolute_directional_gain": math.sqrt(ratio(dy_energy, dx_energy)),
            "radial_energy_fraction_removed_by_normalization_differential": ratio(radial.square().sum(), dy_energy),
            "tangent_energy_fraction": ratio(tangent.square().sum(), dy_energy),
            "radial_tangent_energy_closure_error": ratio((radial.square().sum() + tangent.square().sum() - dy_energy).abs(), dy_energy),
            "normalized_finite_sensitivity": scalar(normalized_dy.square().sum(dim=-1).mean().sqrt()) / epsilon,
            "normalized_linearized_sensitivity": scalar(tangent_derivative.square().sum(dim=-1).mean().sqrt()) / epsilon,
            "normalization_linearization_relative_error": math.sqrt(ratio((normalized_dy - tangent_derivative).square().sum(), normalized_dy.square().sum())),
            "positive_step_raw_relative_l2": math.sqrt(ratio(forward_change.square().sum(), y_energy)),
            "positive_step_normalized_rms_l2": scalar((justnorm(plus.double()) - unit_y).square().sum(dim=-1).mean().sqrt()),
            "zero_output_perturbation_rows": int((dy.square().sum(dim=-1) == 0).sum().item()),
        }
    return result


def bias_intervention(block, x):
    """Functional interventions leave the measured branch parameters untouched."""
    scale = 2.0
    original = {"lin1_l2": scalar(block.lin1.bias.norm()), "lin2_l2": scalar(block.lin2.bias.norm())}
    scenarios = {
        "actual_biases": (block.lin1.bias, block.lin2.bias),
        "zero_bias_control": (torch.zeros_like(block.lin1.bias), torch.zeros_like(block.lin2.bias)),
        "synthetic_nonzero_biases_not_trained": (
            0.5 * torch.sin(torch.arange(block.dim, device=x.device, dtype=x.dtype) + 1),
            0.5 * torch.cos(torch.arange(block.dim, device=x.device, dtype=x.dtype) + 1),
        ),
    }
    result = {"matrix_scale": scale, "original_bias_norms": original, "scenarios": {}}
    for name, (b1, b2) in scenarios.items():
        base = F.linear(block.act(F.linear(x, block.lin1.weight, b1)), block.lin2.weight, b2)
        scaled = F.linear(block.act(F.linear(x, scale * block.lin1.weight, b1)), scale * block.lin2.weight, b2)
        expected = scale ** 3 * base
        result["scenarios"][name] = {
            "lin1_bias_l2": scalar(b1.norm()), "lin2_bias_l2": scalar(b2.norm()),
            "relative_error_vs_degree3_scaling": math.sqrt(ratio((scaled.double() - expected.double()).square().sum(), expected.double().square().sum())),
            "raw_relative_output_change": math.sqrt(ratio((scaled.double() - base.double()).square().sum(), base.double().square().sum())),
            "normalized_output_rms_l2_change": scalar((justnorm(scaled.double()) - justnorm(base.double())).square().sum(dim=-1).mean().sqrt()),
            "normalized_output_max_l2_change": scalar((justnorm(scaled.double()) - justnorm(base.double())).norm(dim=-1).max()),
            "output_cosine": cosine(base, scaled),
        }
    return result


def measure_trunk(trunk, observations, epsilon):
    captured, handles = {}, []
    for index, block in enumerate(trunk.blocks):
        entry = captured[index] = {}
        def capture_branch(module, inputs, output, entry=entry):
            entry.update(input=inputs[0], output=output)
        handles.append(block.register_forward_hook(capture_branch))
        if isinstance(trunk, NormResidualTrunk):
            def capture_stream(module, inputs, entry=entry):
                entry["stream"] = inputs[0]
            handles.append(trunk.block_norms[index].register_forward_pre_hook(capture_stream))
    try:
        output = trunk(observations)
    finally:
        for handle in handles:
            handle.remove()
    blocks = []
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    for index, block in enumerate(trunk.blocks):
        entry = captured[index]
        x, y = entry["input"], entry["output"]
        stream = entry.get("stream", x)
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        noise = justnorm(noise) * x.norm(dim=-1, keepdim=True)
        if isinstance(block, SiTUGLUBranch):
            gate, up = block.gate(x), block.up(x)
            preactivations = {"gate": summarize(gate), "up": summarize(up)}
            activation = situ_glu(gate, up)
        else:
            preactivation = block.lin1(x)
            preactivations = {"lin1": summarize(preactivation)}
            activation = block.act(preactivation)
        item = {
            "index": index, "input": summarize(x), "output": summarize(y),
            "stream": summarize(stream), "activation": summarize(activation),
            "preactivations": preactivations, "branch_stream_cosine": cosine(y, stream),
            "branch_input_cosine": cosine(y, x),
            "directional_probes": {str(e): directional_probe(block, x, y, e, noise) for e in (epsilon, epsilon / 2)},
        }
        if isinstance(block, LReluSqPair):
            item["matrix_scale_bias_intervention"] = bias_intervention(block, x)
        blocks.append(item)
    return {"output": summarize(output), "blocks": blocks}


def collect_checkpoint_observations(path):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    config, state, norm = bundle["args"], bundle["model"], bundle["obs_norm"]
    if config["env_id"] != ENV_ID or config["placement"] != "pre":
        raise ValueError("Expected a saved v2 pre-norm HalfCheetah-v4 bundle")
    means, variances = (norm[key].numpy() for key in ("means", "variances"))
    obs_dim = state["actor.0.in_proj.weight"].shape[1]
    if means.shape != (NUM_ENVS, obs_dim) or variances.shape != means.shape:
        raise ValueError("Checkpoint must contain 16 per-environment observation normalizers")
    if not np.isfinite(means).all() or not np.isfinite(variances).all() or np.any(variances < 0):
        raise ValueError("Invalid saved observation moments")
    width = state["actor.0.in_proj.weight"].shape[0]
    n_blocks = sum(key.startswith("actor.0.block_gates.") for key in state)
    actor = nn.Sequential(
        make_norm_residual_trunk(obs_dim, width, n_blocks, placement="pre",
                                 norm_kind=config["norm_kind"], activation=config["activation"]),
        nn.Linear(width, state["actor.1.weight"].shape[0]),
    ).to("cuda").eval()
    actor.load_state_dict({key.removeprefix("actor."): value for key, value in state.items() if key.startswith("actor.")})
    mirror = make_host_mirror(actor, NUM_ENVS)
    sampler = np.random.default_rng(SEED)
    envs = make_mujoco_vector_env(ENV_ID, NUM_ENVS, backend="native", num_threads=2)
    try:
        low, high = (np.asarray(value, dtype=np.float32).reshape(-1) for value in
                     (envs.single_action_space.low, envs.single_action_space.high))
        if not np.array_equal(low, state["action_low"].numpy()) or not np.array_equal(high, state["action_high"].numpy()):
            raise ValueError("Checkpoint and environment action bounds differ")
        raw, _ = envs.reset(seed=SEED)
        observations = np.empty((ROWS, obs_dim), dtype=np.float32)
        terminations = truncations = 0
        for start in range(0, ROWS, NUM_ENVS):
            normalized = (np.asarray(raw, dtype=np.float64) - means) / np.sqrt(variances + norm["epsilon"])
            if norm["clip"] is not None:
                normalized = np.clip(normalized, -norm["clip"], norm["clip"])
            normalized = normalized.astype(np.float32)
            observations[start:start + NUM_ENVS] = normalized
            _, action = sample_beta_actions_host(mirror(normalized), low, high, sampler)
            if not np.isfinite(action).all():
                raise FloatingPointError("Checkpoint policy produced nonfinite actions")
            raw, _, terms, truncs, _ = envs.step(action.reshape((NUM_ENVS,) + envs.single_action_space.shape))
            terminations += int(terms.sum())
            truncations += int(truncs.sum())
    finally:
        envs.close()
    provenance = {
        "checkpoint": str(path.resolve()), "checkpoint_sha256": sha256(path),
        "saved_args": config, "policy": "saved actor, stochastic Beta actions via host mirror",
        "host_mirror": type(mirror).__name__, "env_id": ENV_ID, "env_backend": "native",
        "num_envs": NUM_ENVS, "env_threads": 2, "environment_seed": SEED,
        "action_rng_seed": SEED, "rows": ROWS, "steps_per_env": ROWS // NUM_ENVS,
        "normalization": "frozen saved per-env means and variances; no moment updates or row pooling",
        "normalizer_shape": list(means.shape), "normalizer_epsilon": norm["epsilon"],
        "normalizer_clip": norm["clip"], "normalizer_counts": norm["counts"].tolist(),
        "terminations": terminations, "truncations": truncations,
        "sampling": "synchronous reset then contiguous observations immediately before each sampled action; no burn-in",
    }
    return actor[0], torch.from_numpy(observations).to("cuda"), provenance


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Exact path to a saved v2 pre-norm bundle")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Relative symmetric finite-difference step; also measure epsilon/2")
    args = parser.parse_args()
    if not math.isfinite(args.epsilon) or not 0 < args.epsilon < 1:
        parser.error("epsilon must be finite and between 0 and 1")
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; run this research script through mlq")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    datasets, models, trained_provenance = {}, {}, None
    obs_dim, width, n_blocks = 17, 64, 3
    if args.checkpoint is not None:
        trained, observations, trained_provenance = collect_checkpoint_observations(args.checkpoint)
        obs_dim, width, n_blocks = trained.in_dim, trained.width, trained.n_blocks
        datasets["checkpoint_on_policy"] = observations
        models["trained_checkpoint_actor"] = trained
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    datasets["gaussian_design_point"] = torch.randn((ROWS, obs_dim), device="cuda", generator=generator)
    for name, factory in (("sphere_stiglu", make_situ_sphere_trunk), ("sphere_lrelusq", make_lrelu_sphere_trunk)):
        torch.manual_seed(SEED)
        models["init_" + name] = factory(obs_dim, width, n_blocks).to("cuda").eval()
    for activation in ("stiglu", "lrelusq"):
        for kind in ("layer", "rms"):
            torch.manual_seed(SEED)
            models[f"init_pre_{kind}_{activation}"] = make_norm_residual_trunk(
                obs_dim, width, n_blocks, placement="pre", norm_kind=kind, activation=activation,
            ).to("cuda").eval()
        models[f"init_pre_rms_{activation}"].load_state_dict(models[f"init_pre_layer_{activation}"].state_dict())
    common_projection = models["init_sphere_stiglu"].in_proj.state_dict()
    for name, trunk in models.items():
        if name.startswith("init_"):
            trunk.in_proj.load_state_dict(common_projection)
    report = {
        "schema_version": 1, "seed": SEED,
        "runtime": {"device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                    "dtype": "float32 model, float64 metric reductions/covariance", "matmul_precision": "highest", "tf32": False},
        "source_sha256": {str(path): sha256(ROOT / path) for path in
                          (Path("cleanrl/shared/host_actor.py"), Path("cleanrl/shared/norm_residual.py"),
                           Path("cleanrl/shared/host_graph.py"), Path("scripts/analyze_activation_geometry.py"))},
        "checkpoint_data_provenance": trained_provenance,
        "gaussian_provenance": {"distribution": "iid N(0,1)", "seed": SEED, "rows": ROWS, "obs_dim": obs_dim},
        "comparison_contract": "All models receive exactly the same tensors per dataset. Init trunks share input projection; preLayer/preRMS pairs share every parameter within activation. Cross-activation branch weights are not matched.",
        "metric_definitions": {
            "channel_mean_energy_fraction": "sum_rows width*mean_channels(x)^2 / sum(x^2); empirical energy along the all-ones channel direction, not a claim of useless DC",
            "dataset_mean_vector_energy_fraction": "rows*||mean_rows(x)||^2 / sum(x^2); distinct from per-row channel means",
            "effective_rank": "exp(entropy of normalized eigenvalues of sample-centered covariance)); zero covariance returns zero",
            "perturbation": "One seeded isotropic direction per row/block shared across models with equal widths; ||dx_i||=epsilon*||x_i||. Radial uses x*(1+/-epsilon), positive scales. Repeat epsilon/2 for finite-difference stability.",
            "radial_removal": "dy=(B(x+dx)-B(x-dx))/2. radial=<dy,yhat>yhat; tangent=dy-radial. Exact energy split of finite dy; D(normalize)_y dy=tangent/||y||, not an exact nonlinear finite-step removal fraction.",
            "zero_denominators": "Ratios with zero denominator report zero; inspect zero-row and zero-perturbation counts. Sphere projection uses repository epsilon=1e-12.",
        },
        "bias_counterexample": {
            "status": "analytic counterexample plus functional interventions; synthetic biases are not trained weights",
            "identity": "For degree-2 phi and c>0, W2 phi(W1 x) scales by c^3 when both matrices scale by c. With biases fixed: c W2 phi(c W1 x+b1)+b2 need not equal c^3[W2 phi(W1 x+b1)+b2].",
            "explicit_example": "W1=W2=I, x=(1,2), b1=(1,0), b2=0, c=2: original=(4,4), matrix-scaled=(18,32), not collinear. Thus normalization does not remove this matrix-only rescaling.",
            "explicit_original": [4, 4], "explicit_matrix_scaled": [18, 32],
            "scope": "LeakyReLU-squared pairs have trainable biases; current SiTU gate/up/down are bias-free. Positive input homogeneity also fails with nonzero pair biases; SiTU is not homogeneous even without biases.",
        },
        "limitations": [
            "Init-only geometry and a trained pre-norm actor are not a trained sphere comparison; no sphere checkpoint is assumed.",
            "Geometry correlations cannot establish the cause of return gaps or predict the SiTU training swap result.",
            "Finite perturbations and synthetic matrix/bias changes are local interventions on frozen branches, not learning dynamics or training interventions.",
            "The fixed on-policy dataset belongs only to the checkpoint actor; it is off-policy for every init trunk and covers 256 post-reset steps per environment, not a stationary full-episode distribution.",
            "Gaussian observations isolate an isotropic input design point, not necessarily Gaussian hidden activations.",
            "Current shared source hashes describe this probe, not the historical runtime used to train the checkpoint.",
            "Model modules execute on CUDA with no gradients; the explicitly requested native host policy mirror executes rollout inference on CPU. No optimizer, training, Jacobian materialization, or compilation benchmark is run.",
        ],
        "datasets": {},
    }
    for dataset_name, observations in datasets.items():
        report["datasets"][dataset_name] = {"observations": summarize(observations), "models": {}}
        for model_name, trunk in models.items():
            report["datasets"][dataset_name]["models"][model_name] = {
                "state": "trained" if model_name.startswith("trained_") else "init_only",
                **measure_trunk(trunk, observations, args.epsilon),
            }
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(f"Activation geometry: {len(datasets)} fixed datasets x {len(models)} trunks; {ROWS} rows each; JSON {args.output}")
    print("Init correlations and frozen-branch interventions only; no training-return causal claim.")


if __name__ == "__main__":
    main()
