# Restart geometry review

Read-only independent review after supervisor restart. No model execution, submissions, or frozen-source edits were performed by this reviewer. Source inspection is distinct from CUDA numerical contract evidence.

## Provenance reconciliation

Current diagnostic/trainer/test hashes below agree with `diagnostic_readiness.json` where that manifest lists them. Plotting-script discrepancy is recorded separately in `restart_reconciliation.json`; the historical readiness manifest was preserved.

| Source | SHA256 |
| --- | --- |
| `scripts/audit_sepclip_actual_oracle_step.py` | `80e4dbc117808da90e9b5b87febc856a2c26d708ac3a4e9c2a56542ffa64c8e5` |
| `scripts/audit_sepclip_oracle_geometry.py` | `03bca505a2e00e1b7ca2d93b8982beb4343dc8b24bcd7287100b593611fbc678` |
| `scripts/audit_sepclip_dense_pg.py` | `1f4053ebe4854063ce19e36b05be37dc3e036d94e368b9b424595c5b98b4c058` |
| `scripts/audit_sepclip_closed_loop_update.py` | `59dec8199edcadb6d02264ad3a8f9545d7a4d5785ad98a25660b1ed05619c208` |
| `tests/test_sepclip_actual_oracle_step.py` | `c8d905cb782dabbe57186d4cd6e3760a16bbbe0f9c6198227af00a4f67079f17` |
| `tests/test_sepclip_oracle_geometry.py` | `41ce2a52a9e59864a8f29fbc32d292810503552dce07a0ab0aeffcc5ae47f7ff` |
| `tests/test_sepclip_closed_loop_update.py` | `c3482ab7c4a7fd33be10c629438467aa7b7f7a9308b5ce548a9649b68082c72e` |
| `cleanrl/ppo_continuous_action_sepclip_transition_microscopy_v12.py` | `b8063ba4c3806d7812d6422c1b4c531d11def5ed58e1a85ccd10cd4591d66867` |
| `cleanrl/ppo_continuous_action_sepclip_rollout_clock_anvil_v7.py` | `07f5f9636103dd5fdc9896de3eb781078df8bb1cb9b3ad87bcb79907fc619c8d` |

## Source-derived findings

The repaired retained-range bound uses the same left singular basis for the whitened score field F^-1/2 g and whitened actor tangent motion F^1/2 J delta. Its numerator and denominator use those projected coordinates, and discarded motion energy is reported separately. The regression contract includes a witness for the previous invalid >100 efficiency ratio. No additional blocking arithmetic defect was identified by inspection.

The reported default rank is an FP64 SVD of an FP32 derivative. FP64 tangent projection removes constraint-normal rounding but cannot recover derivative accuracy. The sensitivity table at relative thresholds 1e-7, 1e-6, 1e-5 and 1e-4 must accompany rank and capacity claims. Exact finite KL uses actual FP32 shape arithmetic with FP64 KL algebra; diagnostic Fisher reevaluates shapes from logits in FP64 and remains explicitly a local continuous diagnostic.

Two-replica signed cross-energy is a useful unresolved-signal discriminator, not a reliable confidence certificate by itself. Negative historical replica cosines (-0.029/-0.041 at H64/H256) prevent selecting training from the fitted historical field. Pair entries share replicas and cannot supply extra independent CI samples. A stable scalar actual-update pairing does not establish stability of every coordinate of the full score field.

## Interpretation gates

1. Verify dense physical restoration, observed-action replay, row-wise normalization, saved state mapping, initial actor logits, action-score replay and GAE/truncation boundary contracts.
2. Resolve physical PG signal with signed cross-replica energy, scalar actual-gain uncertainty/sign agreement, and independent-noise fitted-direction gain. Preserve negative estimates. If uncertainty remains unresolved, increase targeted conditional Monte Carlo information rather than interpret fitted efficiency.
3. Require conclusions to survive FP32-sensitive singular cutoffs. Report required parameter norm, retained and discarded motion energy, finite retraction discrepancy and actual finite KL. Tiny numerical modes and large unrealizable inverse-Jacobian fits do not establish architecture capacity.
4. Independently verify the real nonlinear compiled actor ambient-chord JVP against finite differences before interpreting all-time directional PG. Both first and ten-epoch chords must include hidden-layer perturbations; a bias-only or linear-actor witness is insufficient. Finite-difference parameter rounding and step-size stability must be visible.
5. Keep conditional first-action PG separate from all-time closed-loop PG. Use the latter with matched frozen-normalized/raw rewards and terminal-critic objective to assess the selected GAE transport comparison. Sampler clamp atoms and truncated uniform mass limit exact-score assertions.
6. Select a causal intervention only when an independently supported physical mechanism, trustworthy numerical direction and realizable finite step agree. Local finite-state interpolation, fixed-state conditional MC and literature benchmark comparisons do not establish global optimality or held-out gait generalization.

## Queue state at restart review

All 15 outstanding owned jobs 11757 through 11812 were queued with zero attempts in the direct mlq audit. Ready gates were blocked by unrelated protected job 11802 while unrelated job 11800 ran. No duplicates were submitted. Dense, actual-step, geometry, chart and closed-loop result artifacts were absent at review time.

## Pending independent validation

A new nonlinear chord-JVP contract is being prepared by the physics reviewer. This document will be supplemented by independent source review and queued CUDA execution evidence; no numerical pass is asserted here.

## Nonlinear chord contract source review

Reviewed the new, unexecuted `tests/test_sepclip_closed_loop_nonlinear_jvp.py` at SHA256 `bb19548063cb6fa059a79d9909cdf5f68c8a75179c9abb849a58990b23115ec7`. Its independent compiled primal finite differences use actual saved Adam/Anvil first and ten-epoch ambient chords, require movement in all six hidden matrices and readout, and inspect eight symmetric step sizes. Requiring two adjacent sizes with relative JVP error below 3%, cosine above .999, effective rounded-chord error below 2%, and output-rounding telemetry below 3% is a meaningful numerical contract. Requested 4M and 8M checkpoint coverage to match both closed-loop targets; the reviewed initial draft covered 8M only. Output-rounding telemetry is heuristic rather than a rigorous error bound. A vector-JVP pass does not independently certify a score-weighted physical derivative sign when terms nearly cancel. No model was executed during this review.

Reviewed revised contract at SHA256 `2a12a26a26b2e549c61f18cb2bb8f0be1d337a2db4ae9cd543b18e0491a19ab6`: 4M/8M parameterization, snapshot/trace provenance and explicitly heuristic roundoff label are present. Source review approves this revision for queued numerical execution; no execution pass is asserted.

## Independent analytical readout/precision review

Reviewed `analytical_readout_and_precision.md` at SHA256 `d9a370f364fea2607ef6d40d8047de5f1a533ec47cd8fd3c505dc528d41ace7e`. Local free-linear-readout parametrization at nonzero gain, the zero-weight/nonzero-bias exception, rho*g_rho=w dot g_w+b*g_b, differential radial compensation, ideal Beta immediate-cost derivatives and v13 excess-precision mean coupling all have correct signs and factors. For kappa20 and physical mean .8, q=17/18 and dmean/dell=.08, dlogkappa/dell=.9. Verified all four listed source/artifact hashes and historical gain ranges/qualitative layer pairing signs directly from scalar JSON.

Requested a wording correction: head, gain and bias share the same actor gradient clip scalar, which by itself preserves the raw-gradient identity. Coordinatewise Adam denominators and history can break functional compensation; cross-family aggregate actor norm can change the shared scalar. The stated compensation is differential; exact finite radial head rescaling by1+r is represented by gain rho*(1+r) and bias b/(1+r), away from1+r=0. No model execution or new measurement was performed.
