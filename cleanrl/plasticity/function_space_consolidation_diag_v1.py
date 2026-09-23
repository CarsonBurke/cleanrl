"""Cheap falsification test for function-space consolidation (one-step counterfactual).

Question. At a block boundary the fast tier has moved phi -> z, displacement
d = z - phi. Function-space consolidation would keep only the part of d that
changes the network's outputs on recent inputs (the row space of the output
Jacobian J at phi) and discard the null-space remainder. Before building that
into the compiled protocol, measure on plain fast-tier trajectories whether the
projected move phi + J^+ J d has lower clean held-out risk than the full move z.
If it does not, the idea is dead for a few seconds of compute.

Per boundary (block of 32 single-sample steps, iid_online regime, locked v8
betas, fast tiers at their locked lr and at the hotter lr the two-tier reset
selected) we record clean held-out excess risk at phi, at z, at phi + d_par and
at phi + d_perp for Jacobians over the last 1, 4 and 16 blocks (rank 32, 128,
512 in 5377 parameters), plus displacement energies and output changes. The
trajectory itself is never altered (K = 1), so this is a one-step
counterfactual, which is the cheap and conservative version of the question.
"""

import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from cleanrl.plasticity import network_bayes_stream_v2 as reference
from cleanrl.plasticity import optimizer_proxy_model_v8 as fast
from cleanrl.plasticity.covariance_sketch_eval_v3 import save_json
from cleanrl.plasticity.optimizer_proxy_eval_v9 import generator
from cleanrl.shared import runtime


@dataclass
class Args:
    seed: int = 1
    hidden: int = 64
    input_dim: int = 17
    samples: int = 65536
    block: int = 32
    held_out: int = 8192
    output: str = ""


# (method, lr, beta1, beta2, weight_decay, head_lr_scale): v8 locks and the v10 two-tier selections.
CANDIDATES = {
    "adamw_locked": ("adamw", 3e-4, .995, .9, 0.0, 1.0),
    "adamw_hot": ("adamw", 5e-3, .995, .9, 0.0, 1.0),
    "polar_locked": ("polar", 1e-4, .995, .95, 0.0, 8.0),
    "polar_hot": ("polar", 2e-3, .995, .95, 0.0, 8.0),
}
HISTORY_BLOCKS = (1, 4, 16)


def flatten(layers):
    return torch.cat([layer.reshape(layer.shape[0], -1) for layer in layers], dim=1)


def unflatten(vector, like):
    out, offset = [], 0
    for layer in like:
        size = layer[0].numel()
        out.append(vector[:, offset:offset + size].reshape(layer.shape))
        offset += size
    return out


def jacobian(weights, x):
    """Output Jacobian [K, B, P] at weights [K, O, I+1] on inputs x [B, D], via batched VJPs."""
    k, b = weights[0].shape[0], x.shape[0]
    tiled = [w.repeat_interleave(b, dim=0) for w in weights]          # [K*B, O, I+1]
    h1, h2, _ = reference.forward(tiled, x)                           # [K*B, B, H]
    score = torch.eye(b, device=x.device, dtype=x.dtype).repeat(k, 1)  # row kb selects sample b
    d3 = score.unsqueeze(-1)
    d2 = d3 * tiled[2][..., :-1] * (1 - h2.square())
    d1 = (d2 @ tiled[1][..., :-1]) * (1 - h1.square())
    grads = [torch.cat((d1.transpose(-1, -2) @ x, d1.sum(dim=1).unsqueeze(-1)), dim=-1),
             torch.cat((d2.transpose(-1, -2) @ h1, d2.sum(dim=1).unsqueeze(-1)), dim=-1),
             torch.cat((d3.transpose(-1, -2) @ h2, d3.sum(dim=1).unsqueeze(-1)), dim=-1)]
    return flatten(grads).reshape(k, b, -1)


def project(jac, d):
    """Row-space projection of d [K, P] onto rows of jac [K, B, P] in float64 with relative damping."""
    j = jac.double()
    gram = j @ j.transpose(-1, -2)
    damping = 1e-8 * gram.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).unsqueeze(-1)
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    coefficients = torch.linalg.solve(gram + damping * eye, (j @ d.double().unsqueeze(-1)))
    return (j.transpose(-1, -2) @ coefficients).squeeze(-1).to(d.dtype)


def clean_risk(weights, x, y):
    return (reference.forward(weights, x)[2] - y).square().mean(-1)


def main():
    args = tyro.cli(Args)
    runtime.configure_runtime()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no fallback")
    root = Path(args.output or f"runs/FunctionSpaceDiag__v1__{args.seed}__{time.time_ns()}")
    root.mkdir(parents=True)
    initial = reference.init_weights(args, generator(args.seed, 1), "cuda")
    teacher = reference.draw_teacher(args, generator(args.seed, 2), "cuda")
    train_gen = generator(args.seed, 3)
    xs = torch.randn(args.samples, args.input_dim, generator=train_gen, device="cuda")
    ys = reference.teach(teacher, xs) + torch.randn(args.samples, generator=train_gen, device="cuda")
    held_gen = generator(args.seed, 7)  # fresh namespace; never the protocol's validation (4) or test (5)
    hx = torch.randn(args.held_out, args.input_dim, generator=held_gen, device="cuda")
    hy = reference.teach(teacher, hx)
    names = list(CANDIDATES)
    k = len(names)
    hyper = {key: torch.tensor([CANDIDATES[n][i] for n in names], device="cuda")[:, None, None]
             for i, key in ((1, "lr"), (2, "beta1"), (3, "beta2"), (4, "weight_decay"), (5, "head_lr_scale"))}
    methods = [CANDIDATES[n][0] for n in names]
    weights = [w.unsqueeze(0).repeat(k, 1, 1) for w in initial]
    m = [torch.zeros_like(w) for w in weights]
    v = [torch.zeros_like(w) for w in weights]
    step = torch.zeros((), dtype=torch.int64, device="cuda")
    zeros = torch.zeros(k, 1, device="cuda")
    compiled = torch.compile(fast.transition, fullgraph=True, options={"triton.cudagraphs": False})
    groups = {method: torch.tensor([i for i, mm in enumerate(methods) if mm == method], device="cuda") for method in set(methods)}
    rows = []
    phi = [w.clone() for w in weights]
    started = time.perf_counter()
    for t in range(args.samples):
        x, y = xs[t:t + 1], ys[t:t + 1]
        previous = [w.clone() for w in weights]
        grads, corrections = fast.gradients(weights, previous, x, y, zeros, zeros, zeros, "regression", "adamw")
        merged = [torch.empty_like(w) for w in weights]
        for method, index in groups.items():
            sub = lambda layers: [layer[index] for layer in layers]  # noqa: E731
            sub_hyper = {key: value[index] for key, value in hyper.items()}
            nw, nm, nv, _ = compiled(sub(weights), sub(previous), sub(m), sub(v), step, sub(grads), sub(corrections),
                                     sub_hyper["lr"], sub_hyper["beta1"], sub_hyper["beta2"], sub_hyper["weight_decay"],
                                     method, 1, sub_hyper["head_lr_scale"])
            for layer, dest in zip(nw, merged):
                dest[index] = layer
            for src, dst in zip(nm + nv, m + v):
                dst[index] = src
        weights, step = merged, step + 1
        if (t + 1) % args.block:
            continue
        d = flatten(weights) - flatten(phi)
        risk_phi, risk_z = clean_risk(phi, hx, hy), clean_risk(weights, hx, hy)
        out_block = (reference.forward(weights, xs[t + 1 - args.block:t + 1])[2]
                     - reference.forward(phi, xs[t + 1 - args.block:t + 1])[2]).square().mean(-1)
        out_held = (reference.forward(weights, hx)[2] - reference.forward(phi, hx)[2]).square().mean(-1)
        row = {"step": t + 1, "risk_phi": risk_phi.tolist(), "risk_z": risk_z.tolist(), "energy": d.square().sum(-1).tolist(),
               "output_change_block": out_block.tolist(), "output_change_held": out_held.tolist()}
        for history in HISTORY_BLOCKS:
            n = history * args.block
            if t + 1 < n:
                continue
            jac = jacobian(phi, xs[t + 1 - n:t + 1])
            d_par = project(jac, d)
            if t + 1 == args.block and history == 1:
                # Contract: the projected move reproduces the full move's first-order output change on the block inputs.
                torch.testing.assert_close(jac @ d_par.unsqueeze(-1), jac @ d.unsqueeze(-1), rtol=1e-3, atol=1e-6)
            par = unflatten(flatten(phi) + d_par, phi)
            perp = unflatten(flatten(weights) - d_par, phi)
            row[f"risk_par_{history}"] = clean_risk(par, hx, hy).tolist()
            row[f"risk_perp_{history}"] = clean_risk(perp, hx, hy).tolist()
            row[f"energy_par_{history}"] = d_par.square().sum(-1).tolist()
        rows.append(row)
        phi = [w.clone() for w in weights]
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    summary = {}
    half = len(rows) // 2
    for i, name in enumerate(names):
        entry = {}
        for label, subset in (("all", rows), ("second_half", rows[half:])):
            base_gain = [r["risk_z"][i] - r["risk_phi"][i] for r in subset]
            stats = {"blocks": len(subset), "mean_risk_change_full": sum(base_gain) / len(base_gain),
                     "mean_energy": sum(r["energy"][i] for r in subset) / len(subset),
                     "risk_phi_mean": sum(r["risk_phi"][i] for r in subset) / len(subset)}
            for history in HISTORY_BLOCKS:
                key = f"risk_par_{history}"
                have = [r for r in subset if key in r]
                par_gain = [r[key][i] - r["risk_phi"][i] for r in have]
                perp_gain = [r[f"risk_perp_{history}"][i] - r["risk_phi"][i] for r in have]
                full = [r["risk_z"][i] - r["risk_phi"][i] for r in have]
                stats[f"history_{history}"] = {
                    "mean_risk_change_projected": sum(par_gain) / len(par_gain),
                    "mean_risk_change_nullspace_only": sum(perp_gain) / len(perp_gain),
                    "projected_beats_full_fraction": sum(p < f for p, f in zip(par_gain, full)) / len(full),
                    "energy_fraction_in_rowspace": sum(r[f"energy_par_{history}"][i] for r in have) / sum(r["energy"][i] for r in have),
                }
            entry[label] = stats
        summary[name] = entry
    result = {"args": asdict(args), "candidates": {n: dict(zip(("method", "lr", "beta1", "beta2", "weight_decay", "head_lr_scale"), c))
                                                    for n, c in CANDIDATES.items()},
              "seconds": seconds, "summary": summary, "rows": rows}
    save_json(root / "results.json", result)
    print(json.dumps({"seconds": round(seconds, 1), "root": str(root)}))
    for name, entry in summary.items():
        s = entry["second_half"]
        line = {h: {"proj": round(s[f"history_{h}"]["mean_risk_change_projected"], 6),
                    "null": round(s[f"history_{h}"]["mean_risk_change_nullspace_only"], 6),
                    "win": round(s[f"history_{h}"]["projected_beats_full_fraction"], 3),
                    "efrac": round(s[f"history_{h}"]["energy_fraction_in_rowspace"], 3)} for h in HISTORY_BLOCKS}
        print(json.dumps({"candidate": name, "full": round(s["mean_risk_change_full"], 6), "risk": round(s["risk_phi_mean"], 5), **{str(h): v for h, v in line.items()}}))


if __name__ == "__main__":
    with torch.no_grad():
        main()
