"""Closed-loop function-space consolidation on plain fast-tier trajectories (diagnostic, not the protocol).

v1 measured a one-step counterfactual: at each block boundary the displacement
projected onto the row space of the output Jacobian over recent inputs lowered
clean held-out risk more than the full displacement, and the null-space part
raised it (locked AdamW and polar, iid_online). v2 closes the loop: candidates
with history h > 0 deploy phi' = phi + P_J (z - phi) at every boundary and the
fast tier restarts from phi' (moments kept), candidates with h = 0 are the
plain fast tier. Same four fast tiers as v1, each open and closed at history
1, 4 and 16 blocks: 16 trajectories, one batched step loop. Clean held-out
risk of the deployed weights is recorded at every boundary; the summary reports
the mean over 16 protocol-like checkpoints and over the second half.

Not a selection experiment: learning rates are fixed at the v8 lock and the
v10 two-tier selection, so a win here motivates the swept protocol family and
a loss kills the idea without one.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from cleanrl.plasticity import network_bayes_stream_v2 as reference
from cleanrl.plasticity import optimizer_proxy_model_v8 as fast
from cleanrl.plasticity.covariance_sketch_eval_v3 import save_json
from cleanrl.plasticity.function_space_consolidation_diag_v1 import CANDIDATES, clean_risk, flatten, jacobian, project, unflatten
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


HISTORIES = (0, 1, 4, 16)


def main():
    args = tyro.cli(Args)
    runtime.configure_runtime()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no fallback")
    root = Path(args.output or f"runs/FunctionSpaceDiag__v2__{args.seed}__{time.time_ns()}")
    root.mkdir(parents=True)
    initial = reference.init_weights(args, generator(args.seed, 1), "cuda")
    teacher = reference.draw_teacher(args, generator(args.seed, 2), "cuda")
    train_gen = generator(args.seed, 3)
    xs = torch.randn(args.samples, args.input_dim, generator=train_gen, device="cuda")
    ys = reference.teach(teacher, xs) + torch.randn(args.samples, generator=train_gen, device="cuda")
    held_gen = generator(args.seed, 7)
    hx = torch.randn(args.held_out, args.input_dim, generator=held_gen, device="cuda")
    hy = reference.teach(teacher, hx)
    names = [f"{base}/h{h}" for base in CANDIDATES for h in HISTORIES]
    bases = [n.split("/")[0] for n in names]
    histories = torch.tensor([int(n.split("/h")[1]) for n in names], device="cuda")
    k = len(names)
    hyper = {key: torch.tensor([CANDIDATES[b][i] for b in bases], device="cuda")[:, None, None]
             for i, key in ((1, "lr"), (2, "beta1"), (3, "beta2"), (4, "weight_decay"), (5, "head_lr_scale"))}
    methods = [CANDIDATES[b][0] for b in bases]
    groups = {method: torch.tensor([i for i, mm in enumerate(methods) if mm == method], device="cuda") for method in set(methods)}
    closed = {h: torch.tensor([i for i, n in enumerate(names) if n.endswith(f"/h{h}")], device="cuda") for h in HISTORIES if h}
    weights = [w.unsqueeze(0).repeat(k, 1, 1) for w in initial]
    m = [torch.zeros_like(w) for w in weights]
    v = [torch.zeros_like(w) for w in weights]
    step = torch.zeros((), dtype=torch.int64, device="cuda")
    zeros = torch.zeros(k, 1, device="cuda")
    compiled = torch.compile(fast.transition, fullgraph=True, options={"triton.cudagraphs": False})
    phi = [w.clone() for w in weights]
    risks, energies = [], []
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
        z_flat, phi_flat = flatten(weights), flatten(phi)
        deployed = z_flat.clone()
        for h, index in closed.items():
            n = min(h * args.block, t + 1)
            jac = jacobian([layer[index] for layer in phi], xs[t + 1 - n:t + 1])
            deployed[index] = phi_flat[index] + project(jac, z_flat[index] - phi_flat[index])
        weights = unflatten(deployed, phi)
        phi = [w.clone() for w in weights]
        risks.append(clean_risk(weights, hx, hy))
        energies.append((deployed - phi_flat).square().sum(-1))
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    risk = torch.stack(risks)          # [blocks, K]
    energy = torch.stack(energies)
    blocks = risk.shape[0]
    checkpoints = risk[torch.arange(1, 17, device="cuda") * (blocks // 16) - 1]
    summary = {}
    for i, name in enumerate(names):
        base, h = name.split("/h")
        summary[name] = {"fast_tier": base, "history_blocks": int(h),
                         "sustained_risk_16_checkpoints": float(checkpoints[:, i].mean()),
                         "second_half_mean_risk": float(risk[blocks // 2:, i].mean()),
                         "endpoint_risk": float(risk[-1, i]),
                         "path_length_energy": float(energy[:, i].sum()),
                         "finite": bool(torch.isfinite(risk[:, i]).all())}
    result = {"args": asdict(args), "candidates": names, "seconds": seconds, "summary": summary,
              "risk_curve": risk.cpu().tolist(), "energy_curve": energy.cpu().tolist()}
    save_json(root / "results.json", result)
    print(json.dumps({"seconds": round(seconds, 1), "root": str(root)}))
    for base in CANDIDATES:
        open_risk = summary[f"{base}/h0"]["sustained_risk_16_checkpoints"]
        line = {"fast_tier": base, "open": round(open_risk, 5)}
        for h in HISTORIES[1:]:
            s = summary[f"{base}/h{h}"]
            line[f"h{h}"] = f"{s['sustained_risk_16_checkpoints']:.5f} ({100 * (s['sustained_risk_16_checkpoints'] / open_risk - 1):+.1f}%, energy x{s['path_length_energy'] / summary[f'{base}/h0']['path_length_energy']:.2f})"
        print(json.dumps(line))


if __name__ == "__main__":
    with torch.no_grad():
        main()
