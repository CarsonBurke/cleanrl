# Constraint-aware rollout-clock ANVIL v15

Source: `cleanrl/ppo_continuous_action_sepclip_constraint_transport_v15.py`.
New implementation copied from frozen transition-microscopy v12, with the
original actor, PPO loss, Adam readout/gains and Adam critic. No architecture,
advantage, sampling, LR schedule, gradient-clipping budget, or rollout-clock
change. `--matrix-geometry ambient` uses the original v12 matrix update;
`--matrix-geometry intrinsic` selects the new actor-matrix mechanism.

## Hypothesis and intervention

Unit rows for gate/up and unit columns for down define products of spheres.
An ambient normal force can become a transverse force after a nonlinear polar
map; final weight normalization does not remove that conversion. The controlled
intervention removes normal forces before nonlinear optimizer operations and
carries the rails in the actual tangent spaces.

For each matrix normalization vector w:

    P_w(g) = g - w*(w dot g)/(w dot w).
    PT_(w->y)(v) = v - [(v dot y)/(1+w dot y)]*(w+y).

The second formula is shortest-geodesic sphere parallel transport for unit,
non-antipodal endpoints and tangent v. Each actor matrix carries explicit
normalization-axis metadata; shape does not determine its constraint.

Every intrinsic update performs:

1. Reconcile cached rail endpoints with the current caller-normalized weights.
2. Project the fresh gradient before fast/slow EMA updates and the polar map.
3. Preserve the reference rail schedule, rollout slow-clock, BF16 cascade,
   longer-axis second-moment adaptation, and existing polar-norm convention.
4. Project the adapted direction after the final BF16 rounding. Do not restore
   its lost norm: that would reinflate normal force as tangent amplitude.
5. Apply the same LR/aspect multiplier, retract each changed constraint vector
   to its sphere, and parallel-transport both rails to the new endpoint.

The slow rail remains held between fresh-rollout gradient ingestions, but is
transported after every matrix retraction. Diagonal adaptation statistics keep
the reference lane convention; this is not a complete intrinsic covariance
optimizer or a global natural-gradient budget. Actor global gradient clipping
still precedes the intrinsic matrix projection, matching the control.

## State, compilation, and failure contract

`MatrixOptimizer(..., geometry='ambient', constraint_dims=None)` preserves the
legacy public constructor by default. Intrinsic mode requires one explicit 0/1
axis per flat parameter. The trainer defaults to intrinsic, actor-only matrices.
Stack tuples retain their four-element interface. Version-2 intrinsic bank state
adds `constraint_dim`, owned `base_point`, and persistent device `diagnostics`.
Loading a different geometry or normalization-axis layout is rejected; ordinary
loading rebuilds all cached bank references. Legacy ambient state can continue
through the default ambient constructor, without interpreting its rails as
intrinsic tangent state.

Updates compile with static shapes and contain no CPU model, `.item()`, or host
branch on a tensor. A valid unit tangent retraction has w dot y > 0, so transport
denominator 1+w dot y > 1: its own update cannot reach an antipodal singularity.
Zero norm, nonfinite data, invalid unit state, or invalid transport denominator
sets a sticky device pathology flag. The existing host metric boundary raises
`FloatingPointError`; NaNs are not replaced and the transport denominator has no
epsilon floor. Existing reference spectral/adaptation stabilizers remain intact.

The trainer still globally normalizes the independent Adam head and critic after
each update. Intrinsic matrices are already retracted; any additional FP32
renormalization is reconciled by endpoint transport at their next update.
Zero projected gradient with zero rails does not force a matrix movement.

Diagnostics report the sticky flag, minimum transport denominator, unit-norm
error and rail/direction radial residuals. Existing ambient momentum telemetry
is retained only in ambient mode; comparing a last pre-update gradient with
transported endpoint rails would mix tangent bases. Dense trace metadata marks
intrinsic matrix deltas as already internally retracted. Joint Fisher and real
chords remain measurements, never a policy-update normalization budget.
Historical audit scripts constructing v4/v7 optimizers cannot reconstruct v15
intrinsic counterfactuals unchanged. They must select this version explicitly;
otherwise transported rails would be silently treated as ambient state. Actual
trace retracted deltas and physical closed-loop policy replay retain their usual
interpretation, subject to the explicit delta-stage metadata.

## Attribution and verification

Intrinsic mode can change both direction and retained amplitude. A benchmark
gain would establish this combined mechanism's value, not prove that transport
alone caused it. Compare identical original-actor, LR, seed, horizon, clock and
sampling controls; inspect actual physical credit and joint motion separately.

Independent CUDA contracts in `tests/test_sepclip_constraint_transport_v15.py`
cover row/column transport isometry and reversibility, the normal-force witness,
matrix/Adam ownership, compiled checkpoint continuation, rail phase/clock,
geometry-load rejection and exact same-source ambient v12 update parity.
Only syntax compilation was performed by the implementation owner; CUDA
contracts and full runs must execute through the parent's mlq submissions.
