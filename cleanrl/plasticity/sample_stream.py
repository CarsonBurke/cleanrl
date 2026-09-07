"""Dense, sample-scalar plasticity on a true single-pass stream.

All arms and their LR candidates consume identical data in one compiled update,
replayed in CUDA-graph chunks. This is NOT a per-perceptron optimizer: each sample
has one weight shared across the network. ``agree`` is a heuristic, not a precision
estimator: a nonlinear network's gradient noise is neither isotropic nor independent
of its residual. ``oracle`` knows the injected noise scale and is a reference, NOT a
mathematical bound on finite-horizon Adam performance. Homoscedasticity makes this
reference equal to Adam, but does not require every other heuristic to be null.

Normalization and clipping do not guarantee matched realized mean update size;
reported weight moments diagnose this, and every arm tunes its own LR. Selection
uses sustained clean-target validation MSE, never test MSE. Only after LR selection
are saved checkpoints scored on the untouched test split. Segment endpoints and
sample-weighted checkpoint means accompany exact prequential clean-stream metrics.
Default seed 1 is exploratory evidence, not a cross-seed significance claim.

For Student-t df > 2, --noise is the conditional SD at log-scale zero. For df <= 2,
variance does not exist: --noise is a Student-t scale and no variance/precision
interpretation is warranted. The df=2 case is deliberately nonzero.

    mlq submit --name sample-dense --max-parallel-runs 1 --time-limit 2h \
      --cwd /absolute/repository/path -- .venv/bin/python \
      cleanrl/plasticity/sample_stream.py --samples 65536 --seed 1 --switch-at 0.5
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import torch
from torch.utils.tensorboard import SummaryWriter

METHODS = ("adam", "oracle", "agree", "agree_shuffle", "huber",
           "hetvar", "hetvar_r2", "hetvar_t", "hetvar_ta", "hetvar_shuffle")
D_IN, H, D_OUT = 17, 64, 1


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seeds", type=int, default=1, help="paired task draws; default is exploratory")
    p.add_argument("--samples", type=int, default=65536)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr-grid", type=float, nargs="+",
                   default=[1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    p.add_argument("--hetero", type=float, default=2.0, help="log-noise-scale half-range; 0 is constant")
    p.add_argument("--noise", type=float, default=1.0, help="median conditional SD, or t scale if df <= 2")
    p.add_argument("--tail-df", type=float, default=0.0, help="Student-t df (0 = Gaussian)")
    p.add_argument("--switch-at", type=float, default=0.0, help="fraction; 0 = stationary")
    p.add_argument("--ema", type=float, default=0.999, help="per-sample EMA for c normalization")
    p.add_argument("--cap", type=float, default=20.0)
    p.add_argument("--huber-k", type=float, default=1.345)
    p.add_argument("--readout-lr", type=float, default=0.002,
                   help="per-sample normalized-LMS rate; batch * rate must be < 2")
    p.add_argument("--readout-decay", type=float, default=0.0)
    p.add_argument("--nu", type=float, default=5.0)
    p.add_argument("--var-floor", type=float, default=0.02)
    p.add_argument("--validation", type=int, default=4096)
    p.add_argument("--test", type=int, default=4096)
    p.add_argument("--eval-every", type=int, default=4096, help="samples between clean-target checkpoints")
    p.add_argument("--graph-steps", type=int, default=32, help="updates per host replay call")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--output-dir", default="runs")
    a = p.parse_args()
    a.methods = [m.strip() for m in a.methods.split(",")]
    if not a.methods or len(set(a.methods)) != len(a.methods) or any(m not in METHODS for m in a.methods):
        p.error(f"--methods must contain distinct names from {METHODS}")
    for key in ("seeds", "samples", "batch", "validation", "test", "eval_every", "graph_steps"):
        if getattr(a, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if a.seed < 0 or a.seed + a.seeds + 1000 >= 2 ** 63:
        p.error("seed range must fit a nonnegative signed 64-bit integer")
    if a.samples % a.batch or a.eval_every % a.batch:
        p.error("--samples and --eval-every must be divisible by --batch")
    if a.graph_steps > a.samples // a.batch:
        p.error("--graph-steps cannot exceed the number of updates")
    for key in ("hetero", "noise", "tail_df", "switch_at", "ema", "cap", "huber_k",
                "readout_lr", "readout_decay", "nu", "var_floor"):
        if not math.isfinite(getattr(a, key)):
            p.error(f"--{key.replace('_', '-')} must be finite")
    if a.hetero < 0 or a.noise <= 0 or a.tail_df < 0:
        p.error("hetero and tail-df must be nonnegative; noise must be positive")
    if not 0 <= a.switch_at < 1 or not 0 < a.ema < 1:
        p.error("switch-at must be in [0,1); ema must be in (0,1)")
    if a.switch_at and int(a.samples // a.batch * a.switch_at) == 0:
        p.error("switch must leave at least one update in each segment")
    if a.cap < 1 or a.huber_k <= 0 or a.nu <= 0 or not 0 < a.var_floor <= 1:
        p.error("cap must be >= 1; huber-k and nu positive; var-floor in (0,1]")
    if a.readout_lr <= 0 or a.readout_lr * a.batch >= 2 or a.readout_decay < 0:
        p.error("require 0 < batch * readout-lr < 2 and nonnegative readout-decay")
    if a.readout_lr * a.readout_decay >= 1:
        p.error("readout-lr * readout-decay must be < 1")
    if (len(a.lr_grid) < 3 or any(not math.isfinite(lr) or lr <= 0 for lr in a.lr_grid)
            or any(x >= y for x, y in zip(a.lr_grid, a.lr_grid[1:]))):
        p.error("--lr-grid requires at least three strictly increasing, finite positive values")
    return a


def init_mlp(S, G, gen, device):
    def layer(o, i, std):
        # Same orthogonal initialization as the standard dense PPO trunk.
        w = torch.randn(S, max(o, i), min(o, i), generator=gen, device=device)
        q, _ = torch.linalg.qr(w)
        w = (q if o > i else q.transpose(-1, -2)) * std
        return w.unsqueeze(1).expand(S, G, o, i).clone(), torch.zeros(S, G, o, device=device)
    W1, b1 = layer(H, D_IN, math.sqrt(2))
    W2, b2 = layer(H, H, math.sqrt(2))
    W3, b3 = layer(D_OUT, H, 1.0)
    return [W1, b1, W2, b2, W3, b3]


def teacher(S, gen, device):
    Ws = [torch.randn(S, H, D_IN, generator=gen, device=device) / math.sqrt(D_IN),
          torch.randn(S, H, H, generator=gen, device=device) / math.sqrt(H) * 1.5,
          torch.randn(S, D_OUT, H, generator=gen, device=device) / math.sqrt(H) * 3.0]
    v = torch.randn(S, D_IN, generator=gen, device=device)
    return Ws, v / v.norm(dim=-1, keepdim=True)


def teach(Ws, x):  # x (S,B,D_IN) -> (S,B,1)
    h = torch.tanh(torch.einsum("shi,sbi->sbh", Ws[0], x))
    h = torch.tanh(torch.einsum("shi,sbi->sbh", Ws[1], h))
    return torch.einsum("soh,sbh->sbo", Ws[2], h)


def forward(P, x):  # x (S,B,D_IN) shared over G
    W1, b1, W2, b2, W3, b3 = P
    h1 = torch.tanh(torch.einsum("sgoi,sbi->sgbo", W1, x) + b1.unsqueeze(2))
    h2 = torch.tanh(torch.einsum("sgoi,sgbi->sgbo", W2, h1) + b2.unsqueeze(2))
    out = torch.einsum("sgoi,sgbi->sgbo", W3, h2) + b3.unsqueeze(2)
    return h1, h2, out


def make_state(method, P0, segments, batch):
    S, G = P0[0].shape[:2]
    device = P0[0].device
    q = {"P": [p.clone() for p in P0], "M": [torch.zeros_like(p) for p in P0],
         "V": [torch.zeros_like(p) for p in P0],
         "clean_sum": torch.zeros(segments, S, G, device=device)}
    if method == "adam":
        return q  # No gate, readout, normalization, or weight-moment state.
    q.update({key: torch.ones(S, G, device=device) for key in ("c_ema",)})
    q.update({key: torch.zeros(S, G, device=device) for key in ("lvl_sum", "lvl_sq")})
    if method in ("agree_shuffle", "hetvar_shuffle") and batch == 1:
        q["c_prev"] = torch.ones(S, G, batch, device=device)
    if method == "huber":
        q["r_scale"] = torch.ones(S, G, device=device)
    if method.startswith("hetvar"):
        q.update({key: torch.zeros(S, G, H + 1, device=device) for key in ("u_mu", "u_lv", "u_tw")})
        q.update({key: torch.zeros(S, G, device=device) for key in ("v_pred", "v_twin")})
        q["var_ema"] = torch.ones(S, G, device=device)
        if batch == 1:
            q["prev_c2"] = torch.ones(S, G, batch, device=device)
        if method == "hetvar_ta":
            q["z2_ema"] = torch.ones(S, G, device=device)
            q["z4_ema"] = 3.0 * torch.ones(S, G, device=device)
    return q


def update(states, counter, xs, ys, clean, sigma, permutations, lr, a, switch):
    """One all-arm update. Tensor indexing/counters keep replay free of host sync."""
    S, G, B = a.seeds, len(a.lr_grid), a.batch
    beta = a.ema ** B
    x = xs.index_select(0, counter).squeeze(0)
    y = ys.index_select(0, counter).squeeze(0)
    target = clean.index_select(0, counter).squeeze(0)
    scale = sigma.index_select(0, counter).squeeze(0)
    bc1 = 1 - 0.9 ** (counter.to(torch.float32) + 1)
    bc2 = 1 - 0.999 ** (counter.to(torch.float32) + 1)
    for method, q in zip(a.methods, states):
        P, M, V = q["P"], q["M"], q["V"]
        h1, h2, out = forward(P, x)
        error = (out - target.unsqueeze(1)).square().mean((-1, -2))
        if a.switch_at:
            after = (counter >= switch).to(error.dtype)
            q["clean_sum"][0].add_(error * (1 - after))
            q["clean_sum"][1].add_(error * after)
        else:
            q["clean_sum"][0].add_(error)
        r = out - y.unsqueeze(1)
        d3 = r
        d2 = torch.einsum("sgbo,sgoi->sgbi", d3, P[4]) * (1 - h2 * h2)
        d1 = torch.einsum("sgbo,sgoi->sgbi", d2, P[2]) * (1 - h1 * h1)
        xg = x.unsqueeze(1).expand(S, G, B, D_IN)
        deltas, inputs = (d1, d2, d3), (xg, h1, h2)
        if method == "oracle":
            c_raw = scale.square().reciprocal().unsqueeze(1).expand(S, G, B)
        elif method in ("agree", "agree_shuffle"):
            dot = torch.zeros_like(r[..., 0])
            gsq = torch.zeros_like(dot)
            msq = torch.zeros_like(dot[..., 0])
            for li, (d, inp) in enumerate(zip(deltas, inputs)):
                Mw, Mb = M[2 * li], M[2 * li + 1]
                dot = dot + torch.einsum("sgbo,sgoi,sgbi->sgb", d, Mw, inp)
                dot = dot + torch.einsum("sgbo,sgo->sgb", d, Mb)
                gsq = gsq + d.square().sum(-1) * (inp.square().sum(-1) + 1.0)
                msq = msq + Mw.square().sum((-1, -2)) + Mb.square().sum(-1)
            cos2 = (dot.square() / (gsq * msq.unsqueeze(-1)).clamp_min(1e-30)).clamp(0, 1 - 1e-6)
            c_raw = torch.where(msq.unsqueeze(-1) > 0, cos2 / (1 - cos2), torch.ones_like(cos2))
        elif method == "huber":
            ar = r.squeeze(-1).abs()
            # Retain the historical mean-absolute-residual scale (not a MAD estimate).
            q["r_scale"].mul_(beta).add_(ar.mean(-1), alpha=(1 - beta) * 1.4826)
            c_raw = (a.huber_k * q["r_scale"].unsqueeze(-1) / ar.clamp_min(1e-12)).clamp_max(1)
        elif method.startswith("hetvar"):
            phi = torch.cat((h2, torch.ones_like(r)), -1)
            rr = r.squeeze(-1)
            u_mu, u_lv, u_tw = q["u_mu"], q["u_lv"], q["u_tw"]
            p_mu = torch.einsum("sgbk,sgk->sgb", phi, u_mu)
            dev_lv = torch.einsum("sgbk,sgk->sgb", phi[..., :-1], u_lv[..., :-1])
            dev_tw = torch.einsum("sgbk,sgk->sgb", phi[..., :-1], u_tw[..., :-1])
            q["v_pred"].mul_(beta).add_(dev_lv.square().mean(-1), alpha=1 - beta)
            q["v_twin"].mul_(beta).add_(dev_tw.square().mean(-1), alpha=1 - beta)
            # Heuristic twin shrinkage; its evolving representation is not a JS theorem.
            shrink = (1 - q["v_twin"] / q["v_pred"].clamp_min(1e-12)).clamp(0, 1)
            p_lv = (u_lv[..., -1:] + shrink.unsqueeze(-1) * dev_lv).clamp(-12, 12)
            centred = rr if method == "hetvar_r2" else rr - p_mu
            var_hat = p_lv.exp()
            q["var_ema"].mul_(beta).add_(var_hat.mean(-1), alpha=1 - beta)
            c_raw = var_hat.clamp_min(a.var_floor * q["var_ema"].unsqueeze(-1)).reciprocal()
            if method in ("hetvar_t", "hetvar_ta"):
                z2 = centred.square() / var_hat
                nu = a.nu
                if method == "hetvar_ta":
                    kappa = q["z4_ema"] / q["z2_ema"].square().clamp_min(1e-12)
                    nu_hat = (4 * kappa - 6) / (kappa - 3).clamp_min(1e-3)
                    nu = torch.where(kappa > 3.05, nu_hat, torch.full_like(kappa, 1e6)).unsqueeze(-1)
                    z2c = z2.clamp_max(1e4)
                    q["z2_ema"].mul_(beta).add_(z2c.mean(-1), alpha=1 - beta)
                    q["z4_ema"].mul_(beta).add_(z2c.square().mean(-1), alpha=1 - beta)
                c_raw = c_raw * (nu + 1) / (nu + z2)
            phin = phi / phi.square().sum(-1, keepdim=True)
            c2 = centred.square()
            raw_lv = torch.einsum("sgbk,sgk->sgb", phi, u_lv).clamp(-12, 12)
            raw_tw = torch.einsum("sgbk,sgk->sgb", phi, u_tw).clamp(-12, 12)
            e_lv = (1 - c2 / raw_lv.exp()).clamp(-20, 1)
            mis = torch.roll(c2, 1, dims=-1) if B > 1 else q["prev_c2"]
            e_tw = (1 - mis / raw_tw.exp()).clamp(-20, 1)
            u_mu.sub_(a.readout_lr * torch.einsum("sgb,sgbk->sgk", p_mu - rr, phin))
            u_lv.sub_(a.readout_lr * torch.einsum("sgb,sgbk->sgk", e_lv, phin))
            u_tw.sub_(a.readout_lr * torch.einsum("sgb,sgbk->sgk", e_tw, phin))
            if B == 1:
                q["prev_c2"].copy_(c2)
            if a.readout_decay:
                u_lv[..., :-1].mul_((1 - a.readout_lr * a.readout_decay) ** B)
        if method != "adam":
            q["c_ema"].mul_(beta).add_(c_raw.mean(-1), alpha=1 - beta)
            c = (c_raw / q["c_ema"].unsqueeze(-1).clamp_min(1e-12)).clamp_max(a.cap)
            if method in ("agree_shuffle", "hetvar_shuffle"):
                if B == 1:
                    previous = q["c_prev"].clone()
                    q["c_prev"].copy_(c)
                    c = previous
                else:
                    order = permutations.index_select(0, counter).squeeze(0)
                    c = c.index_select(-1, order)
            q["lvl_sum"].add_(c.mean(-1))
            q["lvl_sq"].add_(c.square().mean(-1))
        for li, (d, inp) in enumerate(zip(deltas, inputs)):
            cd = d if method == "adam" else c.unsqueeze(-1) * d
            grads = (torch.einsum("sgbo,sgbi->sgoi", cd, inp) / B, cd.mean(2))
            for j, g in enumerate(grads):
                idx = 2 * li + j
                m, vv = M[idx], V[idx]
                m.mul_(0.9).add_(g, alpha=0.1)
                vv.mul_(0.999).addcmul_(g, g, value=0.001)
                step = (m / bc1) / ((vv / bc2).sqrt() + 1e-8)
                P[idx].sub_((lr if j == 0 else lr.squeeze(-1)) * step)
    counter.add_(1)


def tensors(states):
    return [tensor for q in states for value in q.values()
            for tensor in (value if isinstance(value, list) else [value])]


def save_json(path, value):
    """Strict JSON, including failed LR candidates as null rather than NaN."""
    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {k: finite(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(v) for v in item]
        return item
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(finite(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@torch.no_grad()
def run(a, writer, evidence, path):
    if not torch.cuda.is_available():
        raise RuntimeError("This compiled CUDA proxy requires a CUDA GPU; no CPU fallback")
    dev = torch.device("cuda")
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(a.seed)
    S, G, B = a.seeds, len(a.lr_grid), a.batch
    steps = a.samples // B
    switch = int(steps * a.switch_at) if a.switch_at else steps
    bounds = [0, switch, steps] if a.switch_at else [0, steps]
    checkpoints = sorted(set(range(a.eval_every // B, steps + 1, a.eval_every // B)) | set(bounds[1:]))
    # Explicit independent RNG namespaces: changing a held-out split size cannot
    # change the teacher, initial network, stream, noise, or shuffle control.
    def generator(offset):
        return torch.Generator(device=dev).manual_seed(a.seed + offset)
    Ws, v = teacher(S, generator(0), dev)
    Ws2, _ = teacher(S, generator(1), dev)
    P0 = init_mlp(S, G, generator(2), dev)
    xs = torch.randn(steps, S, B, D_IN, generator=generator(3), device=dev)
    logsig = a.hetero * torch.tanh(torch.einsum("si,tsbi->tsb", v, xs))
    sigma = a.noise * logsig.exp()
    noise_gen = generator(4)
    eps = torch.randn(steps, S, B, generator=noise_gen, device=dev)
    if a.tail_df:
        # Gamma's generator argument controls the chi-square draw on the SAME GPU.
        chi = 2 * torch._standard_gamma(torch.full_like(eps, a.tail_df / 2), generator=noise_gen)
        eps = eps / (chi / a.tail_df).sqrt()
        if a.tail_df > 2:
            eps.mul_(math.sqrt((a.tail_df - 2) / a.tail_df))
    clean = torch.empty(steps, S, B, 1, device=dev)
    for segment, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        weights = Ws if segment == 0 else Ws2
        for start in range(lo, hi, 1024):
            end = min(start + 1024, hi)
            x = xs[start:end].transpose(0, 1).reshape(S, -1, D_IN)
            clean[start:end].copy_(teach(weights, x).reshape(S, end - start, B, 1).transpose(0, 1))
    ys = clean + (sigma * eps).unsqueeze(-1)
    if not (torch.isfinite(ys).all() & torch.isfinite(sigma.square().reciprocal()).all()).item():
        raise ValueError("Generated noise/scale overflows float32; choose representable noise parameters")
    permutations = (torch.rand(steps, B, generator=generator(5), device=dev).argsort(-1)
                    if B > 1 and any(m.endswith("shuffle") for m in a.methods)
                    else torch.empty(0, dtype=torch.long, device=dev))
    x_val = torch.randn(S, a.validation, D_IN, generator=generator(6), device=dev)
    val_targets = [teach(Ws, x_val)] + ([teach(Ws2, x_val)] if a.switch_at else [])
    states = [make_state(method, P0, len(bounds) - 1, B) for method in a.methods]
    counter = torch.zeros(1, dtype=torch.long, device=dev)
    lr = torch.tensor(a.lr_grid, device=dev).view(1, G, 1, 1)
    mutable = tensors(states)
    initial = [t.clone() for t in mutable]

    def reset():
        counter.zero_()
        for dst, src in zip(mutable, initial):
            dst.copy_(src)

    compiled = torch.compile(update, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    args = (states, counter, xs, ys, clean, sigma, permutations, lr, a, switch)
    setup_start = time.perf_counter()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            reset()
            compiled(*args)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    reset()
    single = torch.cuda.CUDAGraph()
    with torch.cuda.graph(single):
        compiled(*args)
    reset()
    chunk = torch.cuda.CUDAGraph()
    with torch.cuda.graph(chunk):
        for _ in range(a.graph_steps):
            compiled(*args)
    reset()
    torch.cuda.synchronize()
    evidence["setup_seconds"] = time.perf_counter() - setup_start
    evidence["device"] = torch.cuda.get_device_name(dev)
    evidence["cuda_version"] = torch.version.cuda
    evidence["dtype"] = "float32; highest matmul precision (TF32 disabled)"
    evidence["actual_switch_sample"] = switch * B if a.switch_at else None
    evidence["checkpoint_samples"] = [t * B for t in checkpoints]
    evidence["noise_scale_semantics"] = "Student-t scale; variance undefined" if 0 < a.tail_df <= 2 else "conditional SD"
    save_json(path, evidence)
    snapshots, val_curve, durations = [], [], []
    elapsed_steps, update_seconds = 0, 0.0
    begin, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for stop in checkpoints:
        interval = stop - elapsed_steps
        begin.record()
        chunks, remainder = divmod(interval, a.graph_steps)
        for _ in range(chunks):
            chunk.replay()
        for _ in range(remainder):
            single.replay()
        end_event.record()
        segment = 0 if stop <= switch else 1
        values = torch.stack([(forward(q["P"], x_val)[2] - val_targets[segment].unsqueeze(1))
                              .square().mean((-1, -2)) for q in states])
        # Copies at evaluation boundaries only; never copy or sync each update.
        snapshots.append([[p.clone() for p in q["P"]] for q in states])
        val_curve.append(values.cpu())
        update_seconds += begin.elapsed_time(end_event) / 1000
        durations.append(interval)
        elapsed_steps = stop
        for mi, method in enumerate(a.methods):
            for gi, rate in enumerate(a.lr_grid):
                writer.add_scalar(f"validation/{method}/lr_{rate:g}", values[mi, :, gi].mean().item(), stop * B)
        evidence["validation_curve"] = [v.tolist() for v in val_curve]
        evidence["completed_samples"] = stop * B
        evidence["update_seconds"] = update_seconds
        save_json(path, evidence)
        writer.flush()
    val_curve = torch.stack(val_curve)  # checkpoint, method, task draw, LR
    weights = torch.tensor(durations, dtype=torch.float64)
    score = (val_curve.double() * weights[:, None, None, None]).sum(0) / steps
    score_mean = score.mean(1)
    eligible = torch.isfinite(val_curve).all(0).all(1) & torch.isfinite(score_mean)
    chosen = torch.where(eligible, score_mean, torch.inf).argmin(-1)
    if not eligible.any(-1).all():
        failed = [m for m, ok in zip(a.methods, eligible.any(-1).tolist()) if not ok]
        raise RuntimeError(f"No finite validation LR for {failed}; see saved validation trajectory")
    # Lock and persist choices BEFORE even creating the untouched test split.
    evidence["selected_lr_indices"] = chosen.tolist()
    evidence["validation_sustained_grid"] = score.tolist()
    evidence["invalid_lr_candidates"] = (~eligible).tolist()
    evidence["status"] = "lr_locked"
    save_json(path, evidence)
    x_test = torch.randn(S, a.test, D_IN, generator=generator(7), device=dev)
    test_targets = [teach(Ws, x_test)] + ([teach(Ws2, x_test)] if a.switch_at else [])
    test_curve = []
    for ci, stop in enumerate(checkpoints):
        segment = 0 if stop <= switch else 1
        row = []
        for mi, method in enumerate(a.methods):
            gi = int(chosen[mi])
            params = [p[:, gi:gi + 1] for p in snapshots[ci][mi]]
            mse = (forward(params, x_test)[2].squeeze(1) - test_targets[segment]).square().mean((-1, -2))
            row.append(mse.cpu())
            writer.add_scalar(f"test/{method}/clean_mse", mse.mean().item(), stop * B)
        test_curve.append(torch.stack(row))
    test_curve = torch.stack(test_curve)
    sustained = (test_curve.double() * weights[:, None, None]).sum(0) / steps
    evidence["test_curve"] = test_curve.tolist()
    evidence["zero_predictor_test_mse_by_segment"] = [y.square().mean((-1, -2)).cpu().tolist() for y in test_targets]
    evidence["results"] = {}
    for mi, (method, q) in enumerate(zip(a.methods, states)):
        gi = int(chosen[mi])
        segment_metrics = []
        for si, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
            indices = [ci for ci, stop in enumerate(checkpoints) if lo < stop <= hi]
            avg = (test_curve[indices, mi].double() * weights[indices, None]).sum(0) / (hi - lo)
            segment_metrics.append({"start_sample": lo * B, "end_sample": hi * B,
                                    "test_checkpoint_mean_per_draw": avg.tolist(),
                                    "test_endpoint_per_draw": test_curve[indices[-1], mi].tolist(),
                                    "clean_prequential_mse_per_draw": (q["clean_sum"][si, :, gi] / (hi - lo)).cpu().tolist()})
        mean = sustained[mi].mean().item()
        sem = sustained[mi].std().item() / math.sqrt(S) if S > 1 else None
        if method == "adam":
            level_mean, level_sd = [1.0] * S, [0.0] * S
        else:
            lm = q["lvl_sum"][:, gi] / steps
            level_mean = lm.cpu().tolist()
            level_sd = (q["lvl_sq"][:, gi] / steps - lm.square()).clamp_min(0).sqrt().cpu().tolist()
        result = {"lr": a.lr_grid[gi], "lr_edge": gi in (0, G - 1),
                  "validation_sustained_mse": score_mean[mi, gi].item(),
                  "test_sustained_mse": mean, "test_sustained_per_draw": sustained[mi].tolist(),
                  "test_sustained_sem": sem, "weight_mean_per_draw": level_mean,
                  "weight_sd_per_draw": level_sd, "segments": segment_metrics}
        evidence["results"][method] = result
        writer.add_scalar(f"summary/{method}/test_sustained_mse", mean, a.samples)
        writer.add_scalar(f"summary/{method}/lr_edge", int(result["lr_edge"]), a.samples)
        print(f"{method:16s} lr={a.lr_grid[gi]:.2g} validation={score_mean[mi, gi]:.6f} "
              f"test_sustained={mean:.6f} sem={sem if sem is not None else 'n/a (one task draw)'}"
              + (" EDGE: expand grid before a tuned comparison" if result["lr_edge"] else ""), flush=True)
    if "adam" in a.methods:
        ai = a.methods.index("adam")
        for mi, method in enumerate(a.methods):
            diff = sustained[mi] - sustained[ai]
            evidence["results"][method]["paired_test_difference_vs_adam"] = {
                "per_draw": diff.tolist(), "mean": diff.mean().item(),
                "sem": diff.std().item() / math.sqrt(S) if S > 1 else None}
    evidence["stream_samples_per_second"] = a.samples / update_seconds
    evidence["candidate_samples_per_second"] = a.samples * len(a.methods) * G * S / update_seconds
    evidence["status"] = "completed"
    save_json(path, evidence)
    writer.add_scalar("throughput/stream_samples_per_second", evidence["stream_samples_per_second"], a.samples)
    print(f"Measured graph update time {update_seconds:.3f}s; "
          f"{evidence['stream_samples_per_second']:.1f} stream samples/s (all arms/grids); "
          f"setup {evidence['setup_seconds']:.3f}s. Evidence: {path}", flush=True)


def main():
    a = parse()
    directory = Path(a.output_dir) / f"sample_stream_{time.time_ns()}_seed{a.seed}"
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "results.json"
    evidence = {"schema_version": 2, "status": "starting", "config": vars(a),
                "command": sys.argv, "torch_version": torch.__version__,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "evidence_scope": "seed-1 exploratory" if a.seeds == 1 and a.seed == 1 else
                                  ("single-task exploratory" if a.seeds == 1 else "paired independent task draws"),
                "selection_metric": "sample-weighted right-endpoint clean validation checkpoint mean",
                "test_policy": "LR choices persisted before test generation; test never selects LR",
                "oracle_policy": "injected inverse-noise-scale reference, not an upper bound",
                "sem_policy": "across task draws only, conditional on shared validation LR selection; no significance test"}
    save_json(path, evidence)
    writer = SummaryWriter(str(directory))
    writer.add_text("config", json.dumps(evidence, indent=2))
    start = time.perf_counter()
    try:
        run(a, writer, evidence, path)
    except BaseException:
        evidence["status"] = "failed"
        evidence["error"] = traceback.format_exc()
        raise
    finally:
        evidence["wall_seconds"] = time.perf_counter() - start
        save_json(path, evidence)
        writer.close()


if __name__ == "__main__":
    main()
