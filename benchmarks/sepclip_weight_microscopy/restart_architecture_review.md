# Restart architecture and optimizer review

Status: independent static review after supervisor restart. No GPU execution, submission, trainer modification, or completed causal claim by this reviewer.

The recorded `diagnostic_readiness.json` hashes match the current diagnostic, trainer and contract sources except `scripts/plot_sepclip_microscopy.py`: recorded `e748d0d869a8f7b464e3127cdb2dc04c283d1a4abe211b5b5ef5db3fc92cf431`, observed `b89d6e3f022a0c8b10ff8c230ab75edec5efce13c67e3c2db492800e3f77a33e`. Reconcile figure provenance before execution; this mismatch does not establish a numerical defect.

Jobs 11810 and 11811 were queued at inspection, with 11811 correctly dependent on successful 11810. Neither may be treated as completed or duplicated. The NorMuon contract compares the frozen v4 reference and two v12 clock configurations through twenty compiled updates on nonzero warm history. It checks exact gradients, actor/critic parameters, velocity/second moments, auxiliary Adam moments and inactive second-rail state. This meaningfully gates optimizer instrumentation parity, but cannot replace the fresh 8M training control or establish benchmark improvement.

The anchored chart contracts cover the exact alpha/beta control, initial policy and hidden/input Jacobian, changed head/gain gradient flow, host mirror parity, inverse-softplus tiny-excess arithmetic, shape-coordinate derivatives and saved-policy evaluation semantics. The numerical audit compares actual FP32 shapes/actions with FP64 likelihood, entropy, score and KL references and exercises the real host sampler. Its domain contract deliberately expects nonfinite outcomes at extreme positive precision. Passing that contract reports known domain failures; it does not certify that a training trajectory will remain reliable. Inspect numerical errors and actual visited concentration before selecting training.

The chart is an intervention in conditioning and the fixed-anchor feature family. It preserves the initial policy and hidden derivative mathematically while changing head/gain gradients; subsequent hidden derivatives depend on the evolving feature anchor. A return gain cannot be attributed uniquely to precision conditioning without the matched alpha/beta control and functional evidence.

Historical committee measurements reject a blanket infinitesimal/sign-blind SiTU diagnosis on visited states. Paired sample-to-mean/mode improvements recover roughly 1k, predominantly through changed forward speed, while leaving most of the stated 20k gap. These historical measurements do not select a new architecture or establish a globally optimal gait.

Before causal selection, require independently resolved scalar oracle pairing, replicate/horizon agreement, rank-cutoff sensitivity, identical retained-range score/motion comparisons, achieved rather than requested Fisher budgets, finite KL/Taylor validity, clipping-factor telemetry, and actual saved-weight first/ten-epoch closed-loop effects. Large-step disagreement with local linearization invalidates an optimizer-efficiency interpretation. Local finite-state interpolation and critic-tailed finite continuations remain explicitly limited oracles.

No new blocking source defect or evidence-selected architecture candidate was found in this round. Pending numerical and physical jobs must supply the missing measurements.
