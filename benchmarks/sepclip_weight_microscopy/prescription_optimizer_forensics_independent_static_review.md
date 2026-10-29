# Prescription optimizer forensics: independent static review

Preparation approved at the exact hashes below. This is a source and mathematical review, not numerical execution or acceptance of a captured optimizer reconstruction. No models, tests, GPU work or queue submissions were performed by this reviewer.

| File | SHA256 |
| --- | --- |
| scripts/audit_sepclip_prescription_optimizer_forensics.py | fe744392f3355222848f5d1e1b3193e3fb580527119d07410ab15c3518145e32 |
| tests/test_sepclip_prescription_optimizer_forensics.py | 2e95eff6a4a643cba5d2c850124ee9eb7178ae16fcdc022c2bf8d77b1a90dda3 |
| scripts/sepclip_prescription_replay.py, frozen dependency | 7edfc432775a0098a60d63cb3c8642c65a599c3e8ae8fa2cc2f3354885d1dcf9 |
| cleanrl/ppo_continuous_action_sepclip_additive_prerms_anvil_v14.py, actual reference | 27a3e7a84e2c5d0ab1679f84cc4ddb8dad5eb1b6e2bc5f66850ad93ccf1012b0 |

AST-only expansion finds 24 CUDA cases. Earlier evolving source/test hashes are superseded preparation versions, not executed failures. Frozen loader, trainer and existing queued diagnostics were not changed.

## Replay and source contracts

The audit explicitly rejects non-v14 modules. It uses the actual frozen factory and loader, then executes its actual matrix optimizer, fused Adam and compiled matrix normalization. It does not run a v15 intrinsic optimizer through v14 or v7 semantics.

Factory bank members are recovered by parameter object identity and actual factory ordering. Saved serialized parameter-ID lists must equal fresh factory lists before loading. The output additionally binds each serialized ID to its parameter name, records first-member stack-state ownership, and requires exact nonduplicated coverage of all actor and critic parameters. Same-shaped matrix or Adam routes cannot silently evade the ordering guard. This establishes the known factory convention; it is not a general proof against arbitrary adversarial edits to both ID lists and their associated state values. Actual continuation parity remains required.

The snapshot is taken after current-iteration learning-rate annealing and the rollout, before begin_rollout: trainer lines 891–895, 950–965 and 969. The audit derives iteration after subtracting the actual staggered warmup transitions, with no warmup for a single environment. It checks the current annealed Adam and FP32 matrix LR, prior Adam steps, matrix steps/rollouts, prior pending flag, and zero unused NorMuon slow rail. Full-batch ten epochs, compile enabled and absent target-KL stopping are required. begin_rollout(10) is called once on the private replay before all ten saved updates.

Recorded gradients are the source gradients after clipping, not independently reconstructed PPO gradients. The actual source step orders matrix update before fused Adam, captures the ambient delta, then normalizes and captures the retracted delta (trainer lines 982–994). The audit preserves that order. Every parameter's ambient and retracted delta must be bitwise equal to the recorded array at every epoch. Every final state tensor must be bitwise equal to the independently saved endpoint. One-ULP discrepancies fail these captured-record gates; the internal-stage tolerance cannot relax them.

The agent and complete optimizer state trees are deep-copied before loading, preventing same-device loader aliases from mutating captured moments. End-of-audit sentinels verify that the caller model and both original optimizer state trees remain unchanged. Alias-mutation coverage exercises matrix rails/second moments, Adam moments and the private agent.

## Actual method stages and signs

Frozen matrix_update lines 90–116 and MatrixOptimizer.step lines 191–216 define the reviewed order. Anvil uses the actual updated fast/slow rails and scheduled mixture, then gradient.lerp(blend, momentum). NorMuon uses only the actual fast rail and its configured momentum. The frozen helper supplies the actual spectral cascade, including BF16 internal arithmetic and source polynomial/Gram operations; no alternative polar map is substituted.

Lane reduction is over columns for tall/square matrices, producing retained-row moments, and over rows for wide matrices, producing retained-column moments. The actual second-moment EMA, 1e-10 floor, inverse-square-root adaptation, restoration of the cascade's Frobenius norm, aspect scale and negative learning-rate master subtraction are preserved. Final BF16 direction rounding occurs only for Anvil with BF16 precision, as in the source. The separately compiled stage graph includes the FP32 master subtraction, rather than comparing an ideal proposal directly with a stored, rounded subtraction.

Internal stage values are reconstructed from the actual post-step rails, pre-step second moment, recorded clipped gradient and current source schedule scalars. They are not independently captured historical intermediate tensors. The reconstructed lane second moment must match the actual replay state within rtol 2e-6 and zero absolute allowance. The reconstructed master delta additionally requires the declared elementwise engineering allowance and norm-relative error at most 1e-3; null recorded motion requires exact zero. These are explicit numerical acceptance conditions, not analytic error bounds or bitwise stage certification. BF16 discontinuities or separate-graph fusion differences may cause a valid capture to be withheld, which is preferable to accepting unverified stage telemetry.

The reported -gradient dot delta is a positive predicted decrease in the saved clipped loss-gradient direction. It is not a physical-return increase. Before/after Adam moment summaries pair the current recorded gradient with the respective moment and report actual source betas, epsilon and LR. They do not claim to reconstruct an independently saved bias-corrected Adam preconditioner or an alternative update.

The geometry metadata uses the actual loader's axes: additive v14 actor body/input projection and residual mixing/gains are free Euclidean parameters; only actually constrained head/critic matrices receive unit-axis labels. Global normalization corrections are measured without projecting free body matrices onto a fictitious unit-row tangent space.

## Meaningful coverage and withholding

The 24 unexecuted cases include three actual source methods with nonzero warm rails and Adam moments, 12 captured-input corruptions, aliasing, tiny/null and one-ULP delta rejection, one-ULP endpoint rejection, same-shaped serialized-ID permutation, rebuilding a stale scalar cache, single-environment warmup accounting, and three actual compiled kernel versus independently compiled stage comparisons. NorMuon/BF16, Anvil/BF16 and Anvil/FP32 final-rounding distinctions are checked.

The synthetic continuation fixture performs two real source warm rollouts, then explicitly prefixes its counters to a later valid schedule. This meaningfully exercises nonzero warm-state continuation and the later two-rail schedule; it does not establish that its moments came from 2,500 historical training updates. Actual checkpoint provenance and actual ten-epoch bitwise parity must be assessed separately after MLQ execution. A passing synthetic gate is necessary evidence for the implementation, not a substitute for each real artifact's gates.

No blocking source or mathematical defect was found at the reviewed final hashes. Accept real forensic telemetry only after its queued numerical contracts and every actual-capture replay/stage gate pass, with unchanged source/checkpoint/trace/training manifest/execution pins. Withhold affected telemetry on any parity failure. Even complete parity establishes optimizer continuation on the recorded clipped gradients, not independently reliable physical PG, score efficiency, causal benchmark attribution, or v15 warm intrinsic behavior.
