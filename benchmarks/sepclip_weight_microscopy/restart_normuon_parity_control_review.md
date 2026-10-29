# Restart independent NorMuon parity/control review

Read-only source/AST review plus direct mlq job-state inspection. No model/test execution, submission, cancellation or source edits. No benchmark/parity pass is asserted.

## Verdict and job identity

No actionable optimizer-port or model-semantic defect found. Existing jobs11810 (parity) and11811 (fresh8M control) remain queued with zero attempts;11811 depends on successful11810. At review time the ready parity job was blocked by higher-priority unrelated11837. Both declared maxParallelRuns1. No jobs were duplicated or touched.

The reference is the historical v4 `normuon_fast` configuration, not earlier original-row/other-axis NorMuon settings. Its wrapper sets matrix method normuon, lr.005, momentum.85, beta2.95 at `scripts/run_sepclip_anvil2_v4.sh:8`, and shared seed1/8M/16env/2threads, auxiliary lr.0024, no advantage normalization/no clipped value loss, upper PPO clip.2 at lines22-27. Queued11811 includes the exact special optimizer flags. Its omitted shared flags are correct v12 defaults: lr.0024 at line262, norm_advFalse at292, upper clip.2 at296, clip_vlossFalse at298, max_grad_norm.5 at304. BF16 and actor-only hidden selection are explicit queue flags.

## Exact model/optimizer semantics

Static AST equality holds for v4 versus v12 `ResidualMLP`, `Agent`, `layer_init`, `spectral_cascade` and `ppo_loss`. Thus the architecture, initial tensor-generation code, Beta head/native likelihood/value loss, nonlinear actor forward function and spectral arithmetic are unchanged by the port. This is source evidence, not executed tensor parity.

V12 `actor_optimizer='anvil2'` is matrix-versus-all-Adam plumbing, not proof of running the Anvil algorithm. Actual matrix identity is `matrix_method='normuon'`; `make_optimizers` at212-226 selects six actor hidden gate/up/down matrices and puts all head/gain/bias and critic parameters into auxiliary fused Adam(eps1e-5). This matches v4 lines197-209.

The v12 NorMuon branch at91-110 uses velocity0 EMA with beta.85, then `grad.lerp(velocity0,.85)`, the same spectral cascade, long-axis lane second moment beta.95, lane inverse-root adaptation and Frobenius renormalization. Its final BF16 rounding conditional is only for Anvil, so NorMuon keeps the adapted direction FP32 after the BF16 cascade, exactly as v4 lines90-111. No second slow rail or transport is active in this branch.

`begin_rollout` at171-177 changes metadata. `step` at190-208 computes clock-dependent slow_beta, but the NorMuon matrix-update branch ignores it; momentum/fastweight are fixed by the selected method. Metadata differences are therefore semantically inert for the actual NorMuon matrix motion. The fresh loop at887-915 clips actor and critic separately at.5, steps matrix then auxiliary Adam, and normalizes matrices afterward. Unit-row gate/up/head and unit-column down projection is at497-507. Bias/gain and optimizer moments are not altered by that projection.

## What the existing parity gate checks

`tests/test_sepclip_microscopy_normuon.py:20-45` builds frozen v4 and v12 rollout/epoch variants with identical exact-best settings and seed initialization. It compiles the same PPO learner and intentionally uses different rollout-clock metadata.

Lines49-68 compare all current model-state entries exactly, matrix stack scales/velocity/second exactly, dormant velocity1 zero, and auxiliary Adam moment fields exactly. Lines70-102 use two distinct synthetic batches, each ten PPO updates, compare clipped parameter gradients before each optimizer step, then assert exact weights/state again after step and normalization. Lines103-108 ensure twenty updates and nonzero warmed active matrix history/lane state. This covers more than a single zero-history update and specifically falsifies accidental clock dependence in the NorMuon branch.

A pass would establish compiled local learner/update parity at these synthetic batches/settings and warm histories. It would validate instrumentation port semantics without pretending synthetic data is benchmark evidence.

## What it does not certify

The gate has no physical rollout, host Beta sampler or observation/reward-normalization trajectory, GAE/truncation replay, staggered warmup or diagnostic capture/RNG comparison. Its128-sample synthetic learner and eager normalization are not the full benchmark batch/compiled-normalization execution path. It does not validate all possible optimizer histories, save/load/resume equivalence, environmental determinism, full8M scores or physical/gait/PG findings for NorMuon. Current Agent/PPO source identity and existing observational contracts reduce those concerns but cannot replace fresh run/artifact validation.

After parity,11811 remains necessary to supply actual normalized NorMuon checkpoints/traces and benchmark behavior. Anvil/Adam microscope numbers must not be relabeled as NorMuon measurements before those artifacts exist.

## Causal comparison scope

The intended fresh comparison remains valid as an architecture-matched benchmark of optimizer CONFIGURATIONS: same frozen v12 network, action law, PPO objective/shared controls and execution protocol, with explicit method-specific matrix lr/momentum/beta2/clock choices. NorMuon best-reference flags differ from Anvil's chosen flags/schedules and from all-parameter Adam. Consequently a score difference cannot isolate algorithm identity, slow-rail clock, spectral flattening or any single metric by itself. Mechanistic claims still require matched physical credit, actual joint functional steps and closed-loop evidence. No such causal result is available from this static review or queued gate.

## Reviewed hashes

- `tests/test_sepclip_microscopy_normuon.py`: `4a65a58613b9b9dfaa6164bf144e3e4657d25f354d4dde8d95ca23e53627c64a`
- `cleanrl/ppo_continuous_action_sepclip_anvil2_v4.py`: `1bf4cb185907e87409d466d130081e99951b99b8ef5ce8ed01e361703efb99de`
- `cleanrl/ppo_continuous_action_sepclip_transition_microscopy_v12.py`: `b8063ba4c3806d7812d6422c1b4c531d11def5ed58e1a85ccd10cd4591d66867`
- `scripts/run_sepclip_anvil2_v4.sh`: `64ecb6aee265e75f078364a604405c749cf07a73b062966ee8ae2e6cdabda05c`
