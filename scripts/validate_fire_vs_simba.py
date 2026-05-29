"""Validate cleanrl PPO FIRE implementation against the SAC Simba reference.

Two things to verify:

(1) Numerical equivalence of the projection operation: our `apply_fire`
    (PyTorch) must compute the same post-FIRE weight as Simba SAC's
    `orthogonal_project_layer` (JAX) on the same input matrix. We port
    Simba's JAX recipe to PyTorch line-for-line and compare element-wise.

(2) Actual magnitude ratios on a real cleanrl PPO Agent: instantiate
    Agent(envs) and report Frobenius/spectral norms before/after FIRE
    for every Linear, so claims like "actor head explodes ~30x" can be
    checked numerically rather than back-of-envelope.

Also runs the same projection on a Simba-style head spec (d_in=128,
d_out=action_dim, init gain=1.0) to confirm the asymmetry vs PPO is
real and matches my claim direction.

Usage:
    .venv/bin/python scripts/validate_fire_vs_simba.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

# Import OUR apply_fire and newton_schulz from v3 (canonical FIRE)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cleanrl.ppo_continuous_action_fire_v3 import (  # noqa: E402
    Agent,
    apply_fire as ours_apply_fire,
    newton_schulz as ours_newton_schulz,
)


# ---------------------------------------------------------------------------
# Faithful PyTorch port of Simba's `orthogonalize_via_newton_schulz` and
# `orthogonal_project_layer` from rl/sac/scale_rl/agents/simba/projection.py.
#
# JAX original (cubic = quintic with third coefficient 0):
#   ns_coeffs = (1.5, -0.5, 0.0); ns_steps = 10
#   def newton_schulz_iterator(x, c):
#       a = x @ x.T
#       b = c[1]*a + c[2]*a@a
#       return c[0]*x + b @ x
#   if x.shape[0] > x.shape[1]: x = x.T; transposed = True   # make WIDE
#   x = x / (||x|| + eps)
#   for _ in range(ns_steps): x = newton_schulz_iterator(x, c)
#   if transposed: x = x.T
#
# scaler_type='muon':
#   d_in, d_out = x.shape  # JAX kernel shape is (in, out)
#   scale = sqrt(d_out / d_in)
#   return scale * NS(x)
# ---------------------------------------------------------------------------


def simba_newton_schulz(x: torch.Tensor, ns_steps: int = 10, eps: float = 1e-8) -> torch.Tensor:
    """Port of Simba's NS. Uses the 'transpose to WIDE' convention
    (shape[0] > shape[1] => transpose), then iterates with a = x @ x.T."""
    assert x.ndim == 2
    c = (1.5, -0.5, 0.0)
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    x = x / (torch.linalg.norm(x) + eps)
    for _ in range(ns_steps):
        a = x @ x.T
        b = c[1] * a + c[2] * (a @ a)
        x = c[0] * x + b @ x
    if transposed:
        x = x.T
    return x


def simba_project_kernel_muon(
    kernel: torch.Tensor, ns_steps: int = 10
) -> torch.Tensor:
    """Faithful port of Simba's `orthogonal_project_layer` with
    scaler_type='muon'. NOTE: Simba's `kernel` is (d_in, d_out) per
    flax convention. We follow that convention here, so callers must
    transpose a PyTorch weight `(d_out, d_in)` -> `(d_in, d_out)` to
    feed this function the same shape Simba would see."""
    assert kernel.ndim == 2
    d_in, d_out = kernel.shape[0], kernel.shape[1]
    scale = (d_out / d_in) ** 0.5
    return scale * simba_newton_schulz(kernel, ns_steps=ns_steps)


# ---------------------------------------------------------------------------
# (1) Numerical equivalence on synthetic matrices spanning the layer
# shapes that appear in PPO + Simba.
# ---------------------------------------------------------------------------

def assert_close(name: str, ours: torch.Tensor, simba: torch.Tensor, atol: float = 1e-5):
    diff = (ours - simba).abs().max().item()
    rel = diff / (simba.abs().max().item() + 1e-12)
    status = "OK" if diff < atol else "FAIL"
    print(f"  [{status}] {name:40s}  max|diff|={diff:.3e}  rel={rel:.3e}")
    return diff < atol


def compare_projection(shape_torch: tuple[int, int], seed: int = 0) -> bool:
    """shape_torch is the PyTorch nn.Linear weight shape (d_out, d_in).
    The Simba kernel for the same Linear is the transpose: (d_in, d_out)."""
    g = torch.Generator().manual_seed(seed)
    W_torch = torch.randn(shape_torch, generator=g)
    # Ours operates on PyTorch (d_out, d_in) directly:
    d_out, d_in = shape_torch
    ours_ns = ours_newton_schulz(W_torch.clone(), num_iters=10)
    ours_scale = (d_out / d_in) ** 0.5
    ours_out = ours_ns * ours_scale
    # Simba operates on (d_in, d_out) — feed the transpose:
    W_simba_kernel = W_torch.clone().T  # (d_in, d_out)
    simba_out_kernel = simba_project_kernel_muon(W_simba_kernel, ns_steps=10)
    # Transpose back to (d_out, d_in) for comparison:
    simba_out = simba_out_kernel.T
    return assert_close(f"shape (d_out={d_out}, d_in={d_in})", ours_out, simba_out)


def numerical_equivalence_check():
    print("=" * 78)
    print("(1) Numerical equivalence: ours vs Simba on shared matrices")
    print("=" * 78)
    shapes = [
        # PPO trunk/head shapes (HalfCheetah obs_dim=17, action_dim=6, hidden=64)
        (64, 17),   # trunk input
        (64, 64),   # trunk hidden
        (1, 64),    # critic head
        (6, 64),    # actor head (HalfCheetah)
        # Simba shapes (hidden=128, action_dim=6)
        (128, 17),  # trunk input
        (128, 128), # hidden
        (1, 128),   # critic head
        (6, 128),   # actor head
        # Pathological (very wide / very tall)
        (4, 256),
        (256, 4),
        (1024, 32),
        (32, 1024),
    ]
    all_ok = True
    for sh in shapes:
        all_ok &= compare_projection(sh)
    print()
    print(f"Overall: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ---------------------------------------------------------------------------
# (2) Actual magnitudes on a real cleanrl PPO Agent
# ---------------------------------------------------------------------------

def spec_norms(W: torch.Tensor) -> tuple[float, float]:
    """Return (Frobenius, spectral) norms."""
    fro = float(torch.linalg.norm(W))
    spec = float(torch.linalg.svdvals(W)[0])
    return fro, spec


def per_layer_report(label: str, agent: Agent) -> dict[str, tuple[float, float]]:
    print(f"\n  --- {label} ---")
    print(f"  {'layer':30s}  {'shape':>12s}  {'||W||_F':>10s}  {'||W||_2':>10s}")
    out: dict[str, tuple[float, float]] = {}
    for name, m in agent.named_modules():
        if isinstance(m, nn.Linear):
            fro, spec = spec_norms(m.weight.data)
            shape = f"({m.weight.shape[0]},{m.weight.shape[1]})"
            print(f"  {name:30s}  {shape:>12s}  {fro:10.4f}  {spec:10.4f}")
            out[name] = (fro, spec)
    return out


def real_agent_check():
    print()
    print("=" * 78)
    print("(2) Real PPO Agent: init magnitudes vs post-FIRE magnitudes")
    print("=" * 78)
    torch.manual_seed(0)
    envs = gym.vector.SyncVectorEnv([lambda: gym.wrappers.ClipAction(gym.wrappers.FlattenObservation(gym.make("HalfCheetah-v4")))])
    agent = Agent(envs)
    pre = per_layer_report("INIT (before FIRE)", agent)
    ours_apply_fire(agent, num_iters=10)
    post = per_layer_report("POST-FIRE (NS + Muon)", agent)

    print()
    print("  Frobenius-norm ratio (post-FIRE / init):")
    print(f"  {'layer':30s}  {'pre F':>10s}  {'post F':>10s}  {'ratio':>10s}")
    for name in pre:
        pf, _ = pre[name]
        qf, _ = post[name]
        ratio = qf / pf
        marker = "  *** EXPLODES" if ratio > 5 else ("  (shrinks)" if ratio < 0.5 else "")
        print(f"  {name:30s}  {pf:10.4f}  {qf:10.4f}  {ratio:10.4f}{marker}")


# ---------------------------------------------------------------------------
# (3) Same analytical exercise for Simba-style head (uniform gain=1.0)
# ---------------------------------------------------------------------------

def simba_head_check():
    print()
    print("=" * 78)
    print("(3) Simba-style head spec (kernel_init_scale=1.0)")
    print("=" * 78)
    # Build a Linear matching Simba's actor head: in=128 hidden, out=6 action
    # Simba init uses orthogonal at scale=1.0 (PyTorch equivalent: gain=1.0).
    torch.manual_seed(0)
    head = nn.Linear(128, 6)  # PyTorch weight shape (6, 128)
    torch.nn.init.orthogonal_(head.weight, gain=1.0)
    pre_f, pre_s = spec_norms(head.weight.data)
    print(f"  Simba actor head (gain=1.0): ||W||_F={pre_f:.4f}  ||W||_2={pre_s:.4f}")

    # Apply OUR apply_fire on a mini-module
    class M(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.h = h
    m = M(head)
    ours_apply_fire(m, num_iters=10)
    post_f, post_s = spec_norms(head.weight.data)
    print(f"  Post-FIRE:                   ||W||_F={post_f:.4f}  ||W||_2={post_s:.4f}")
    print(f"  Frobenius ratio: {post_f/pre_f:.4f}  (Simba's head SHRINKS by ~{pre_f/post_f:.2f}x)")

    # And the PPO actor head equivalent (gain=0.01, in=64, out=6)
    torch.manual_seed(0)
    head_ppo = nn.Linear(64, 6)
    torch.nn.init.orthogonal_(head_ppo.weight, gain=0.01)
    pre_f, pre_s = spec_norms(head_ppo.weight.data)
    print()
    print(f"  PPO actor head (gain=0.01):  ||W||_F={pre_f:.4f}  ||W||_2={pre_s:.4f}")
    m = M(head_ppo)
    ours_apply_fire(m, num_iters=10)
    post_f, post_s = spec_norms(head_ppo.weight.data)
    print(f"  Post-FIRE:                   ||W||_F={post_f:.4f}  ||W||_2={post_s:.4f}")
    print(f"  Frobenius ratio: {post_f/pre_f:.4f}  (PPO's head EXPLODES by ~{post_f/pre_f:.2f}x)")


if __name__ == "__main__":
    eq_ok = numerical_equivalence_check()
    real_agent_check()
    simba_head_check()
    print()
    if eq_ok:
        print(">>> Numerical equivalence vs Simba: PASS")
    else:
        print(">>> Numerical equivalence vs Simba: FAIL")
