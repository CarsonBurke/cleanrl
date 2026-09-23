"""Streaming MLP on the high-dimensional vol panel: shared GatePolar (polar / resistance-gated decay) vs Adam.

Same stream as `panel_hd_mlp.py`: fresh 257-256-256-1 tanh net, one optimizer step per bar on that
bar's cross-section (~190 stocks), predict-before-update, scored on the last 40% of time while
learning. Relative MSE vs the constant fit-window mean (1.0 = nothing captured); also eight equal
test-window blocks for paired comparison. `--perm` runs the time-permuted-target control (nothing
learnable). One (method, config) per job; results.json under runs/.

  adam        : torch.optim.Adam, the recorded reference (0.7501 at lr 3e-4, seed 1)
  gate_polar  : cleanrl.shared.gate_polar.GatePolar on mlp_groups(net); weight_decay 0 is plain polar

    .venv/bin/python cleanrl/plasticity/panel_hd_mlp_gate_v1.py --method gate_polar --lr 3e-4 --weight-decay 0.3 --gate-beta 0.9999
"""
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

sys.path.insert(0, "/home/marvin/Documents/repositories/cleanrl")
from cleanrl.plasticity.panel_hd import Args as HDArgs, Bank  # noqa: E402
from cleanrl.plasticity.panel_hd_mlp import mlp  # noqa: E402
from cleanrl.shared.gate_polar import GatePolar, mlp_groups  # noqa: E402

BLOCKS = 8


@dataclass
class Args(HDArgs):
    width: int = 256
    seed: int = 1
    method: str = "gate_polar"
    lr: float = 3e-4
    weight_decay: float = 0.0
    gate_beta: float = 0.9999
    head_lr_scale: float = 1.0
    polar_scale: float = 0.2
    perm: bool = False
    compile: bool = True
    output: str = ""


def make_optimizer(a, net):
    if a.method == "adam":
        return torch.optim.Adam(net.parameters(), lr=a.lr)
    if a.method == "gate_polar":
        return GatePolar(mlp_groups(net, head_lr_scale=a.head_lr_scale), lr=a.lr, betas=(0.9, 0.999), eps=1e-8,
                         weight_decay=a.weight_decay, gate_beta=a.gate_beta, polar_scale=a.polar_scale,
                         compile=a.compile)
    raise ValueError(f"unknown method {a.method}")


def stream(a, bank, yy, vv):
    """One pass: returns (relative MSE on the test window, per-block relative MSE, seconds)."""
    T, L, F, cut, dev = bank.T, bank.L, bank.F, bank.cut, bank.dev
    torch.manual_seed(a.seed)
    net = mlp(F, a.width, dev)
    opt = make_optimizer(a, net)
    se = torch.zeros(BLOCKS, device=dev)
    ss = torch.zeros(BLOCKS, device=dev)
    started = time.perf_counter()
    for t in range(L + 1, T - 1):
        mk = vv[t]
        if not bool(mk.any()):
            continue
        x = bank.feats(t)[mk]
        yb = yy[t][mk]
        res = net(x).squeeze(-1) - yb
        if t >= cut:
            block = min((t - cut) * BLOCKS // (T - cut), BLOCKS - 1)
            se[block] += res.detach().square().sum()
            ss[block] += yb.square().sum()
        loss = res.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    total = (se.sum() / ss.sum()).item()
    if not (torch.isfinite(se).all() and torch.isfinite(ss).all()) or total != total:
        raise FloatingPointError("nonfinite stream score")
    return total, (se / ss).tolist(), seconds


def main():
    a = tyro.cli(Args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    bank = Bank(a)
    name, yy, vv = bank.streams()[1 if a.perm else 0]
    tag = f"{a.method}_lr{a.lr:g}_wd{a.weight_decay:g}_gb{a.gate_beta:g}_hs{a.head_lr_scale:g}{'_perm' if a.perm else ''}"
    root = Path(a.output or f"runs/PanelHDMLPGate__v1__{tag}__{a.seed}__{time.time_ns()}")
    root.mkdir(parents=True)
    print(f"panel {bank.T} bars x {bank.N} stocks, F={bank.F}, width {a.width}, stream {name}, {tag}")
    rel, blocks, seconds = stream(a, bank, yy, vv)
    result = {"args": asdict(a), "stream": name, "relative_mse": rel, "blocks": blocks, "seconds": seconds,
              "parameters": sum(p.numel() for p in mlp(bank.F, a.width, bank.dev).parameters())}
    (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"  stream {name} {tag}: {rel:.5f} ({seconds:.0f}s)")
    print("    blocks " + " ".join(f"{b:.4f}" for b in blocks))


if __name__ == "__main__":
    main()
