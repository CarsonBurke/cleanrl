# Spec: Hyperspherical DenseNet + soft-MoE trunk (hyperMoE)

Status: BACKLOG (run after the shared-backbone hyperspherical test concludes).
Owner context: SimBaV2-on-PPO line, base = `ppo_continuous_action_iterthink_v24_beta_simbav2_v1.py`.

## Why

Holding the v24_beta PPO machinery fixed (Beta actor, HL-Gauss symlog critic, rankgauss,
clip-higher 0.2/0.28, decoupled dual-clip 0.25, target_kl 0.03, 16x2048, 8M), the ONLY
difference between v24_beta (~9785) and the SimBaV2 hyperspherical port is the TRUNK:

- v24_beta trunk = **DenseNet** (dense cross-depth concat) + **soft-MoE** (16 experts,
  softmax over all, no top-k) + convex residual gate + RMSNorm, **shared backbone**.
- SimBaV2 trunk = hyperspherical residual MLP (l2-norm everywhere, unit-norm weights via
  per-step projection, learnable Scalers, LERP residual), separate backbones.

Empirically the SimBaV2 trunk caps ~6-7k on HalfCheetah. Key realization: **the hypersphere
is a conditioning/normalization scheme, ORTHOGONAL to expressiveness.** It fixes gradient
conditioning / plasticity (which on-policy PPO DOES suffer) but supplies no inductive bias.
v24 wins on FUNCTION CLASS: MoE conditional computation + dense feature reuse + shared
coupling. Conditioning != inductive bias. So the two should STACK, not compete.

Ablations that motivate this (all on the shared base, only the trunk varies):
- actor capacity is NOT the lever: actor 256/2 (v6) <= actor 128/1 (v5) at every checkpoint.
- the hyperspherical *value head* helped (v5 > v4: 5932 vs 5092 @4M), so on-sphere readout matters.
- critic is lr-sensitive where v24's is not; both-3e-4 best, EV recovers from a transient dip.

## Hypothesis

Applying hyperspherical normalization to v24's DenseNet+MoE trunk (keeping the dense
reach-back + soft-MoE + shared backbone) gives v24's expressive mechanism SimBaV2's
conditioning, and EXCEEDS v24 (>9785). If it merely matches v24, the hypersphere adds
nothing on-policy at this scale; if it beats v24, conditioning was a real headroom and the
path is to then scale capacity/depth (SimBaV2's monotonic-scaling property).

## Design (port v24's ThinkTrunk -> hyperspherical)

Reuse the ported primitives already in the SimBaV2 file: `l2normalize`, `HyperDense`
(bias-free, orthogonal init, per-step unit-sphere weight projection), `Scaler`
(reparam gain), `HyperMLP` (inverted bottleneck), and `project_hyperdense_weights`.

Convert `ThinkBlock` / `ThinkTrunk`:
- entry / in_proj / out_proj Linears  -> `HyperDense` + `Scaler`, output `l2normalize`d.
- `_branch_body` (dense branch + each MoE expert) -> `HyperMLP` (HyperDense->Scaler->ReLU+eps
  ->HyperDense->l2normalize), inverted-bottleneck expansion 2-4x.
- MoE gate Linear -> `HyperDense` (logits need not be on the sphere; softmax over experts).
- combine: weighted sum of unit-norm expert outputs (+ dense branch + convex-residual x_in),
  then `l2normalize` to return the residual stream to the sphere (LERP-style:
  `z <- l2normalize(x_in + alpha*(branch_sum))` with a learnable per-channel `Scaler` alpha).
- RMSNorm -> `l2normalize` everywhere.
- DenseNet reach-back: concat prior block OUTPUTS (already unit-norm) and `l2normalize` the
  concat before the next block's HyperDense in_proj (as the HyperEmbedder does on input).
- KEEP: dense reach-back concat, soft-MoE (16 experts, softmax all, no top-k), convex
  residual gate, **shared backbone** + decoupled dual-clip.

Open design choices to sweep:
- where alpha/LERP sits (per block vs per branch); expert count (8/16); expansion (2/4);
  narrow (H=64, v24's winning geometry) vs wide (H=256/512, SimBaV2's claim that wide works).
- whether the MoE expert outputs are individually l2-normalized before the softmax mix
  (mix-then-normalize vs normalize-then-mix) — affects how the convex combo lands on the sphere.

## Eval

HalfCheetah-v4, 16 envs, 8M, seed 1 (single-seed screening). Bar = v24_beta 9785.
Track explained_variance + approx_kl + the per-block alpha/Scaler norms to confirm the
hyperspherical conditioning is active (stable EV at hot lr is the tell).

## First variant to run

Narrow first (H=64, matches v24's winning geometry), shared backbone, 16 experts, expansion
2, hyperspherical value head. Then a wide variant (H=256) to test the scaling claim.
