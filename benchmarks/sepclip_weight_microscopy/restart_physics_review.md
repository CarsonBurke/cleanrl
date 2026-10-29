# Restart physics committee review

Read-only source review after supervisor restart; no model execution or queue submissions performed by this reviewer. Existing sources remain unchanged.

## Reviewed sources

- `scripts/audit_sepclip_dense_pg.py`: `1f4053ebe4854063ce19e36b05be37dc3e036d94e368b9b424595c5b98b4c058`
- `scripts/audit_sepclip_physics.py`: `1d53db1d68f6fb188fb7207513b606a652f70682029e278031cd3adb25605114`
- `scripts/audit_sepclip_gait.py`: `cc5f8b35167c3beffcc71561474dcf81c79f3ad68b7b82ab35816f04f4d7e68e`
- `scripts/audit_sepclip_closed_loop_update.py`: `59dec8199edcadb6d02264ad3a8f9545d7a4d5785ad98a25660b1ed05619c208`
- `tests/test_sepclip_closed_loop_update.py`: `c3482ab7c4a7fd33be10c629438467aa7b7f7a9308b5ce548a9649b68082c72e`
- `tests/test_sepclip_dense_pg.py`: `cf54ad6684999b49b1683028892201a36f7126b48648d9eb8e27984368421086`
- `tests/test_sepclip_gait.py`: `540c7b8bef4ee0db0ec52a911f39cec10198a0aff2625365a3cdd01dde5deb71`
- `tests/test_sepclip_closed_loop_nonlinear_jvp.py`: `2a12a26a26b2e549c61f18cb2bb8f0be1d337a2db4ae9cd543b18e0491a19ab6`

## Findings and evidence gates

- Dense conditional PG uses independent first actions with shared continuation uniforms and a leave-one-out baseline. Conditional on continuation noise, this baseline is independent of each first action and avoids self-baseline shrinkage. Independent replicas combine action and continuation noise.
- Dense physical restoration, saved observed-action transition replay, normalization replay, truncation bootstrap, and analytic Beta score reconstruction are materially different reliability checks. Inspect each numerical result; analytic score PG relative error is recorded without an enforced threshold.
- Dense observed-action Q minus the old estimated V has one original action per state. Replicas reduce continuation noise, not action-population uncertainty, and estimated V is not independently known V_pi. Finite and old-critic-tail objectives must stay separately labeled.
- Full closed-loop directional PG sums all time-dependent score terms with gamma-to-time weighting. Fixed-terminal GAE and augmented MC share the old critic terminal function. Lambda=1 agreement establishes algebra, not critic accuracy or infinite-horizon truth.
- Old, first actual update, and exact full-update endpoint policies use seven paired hybrid arms. Their finite interaction is delta_all minus delta_t0_only minus delta_after_t0; this is not an additive critic-bias or gradient decomposition.
- Ambient actual endpoint chord JVP is not the derivative of a sphere-retracted path. Finite gain minus predicted derivative mixes curvature, feedback, and noise. Directional primal mismatch is telemetry, requiring numerical inspection.
- Historical strongest checkpoint physical PG replica cosines were -0.029 at H64 and -0.041 at H256; chart-field cosines were +0.098 and -0.006. These do not justify interpreting fitted efficiency or selecting a causal intervention. Require signed cross-replica signal energy and held-out direction pairing, consistent horizons through H512, and tail sensitivity.
- Gait discovery selects actions solely from discovery replicas; independent validation replicas assess winners. Native action perturbation .025 is physical action perturbation .05. Contact groups are binary rear/front contact patterns rather than temporal stride phase. Contacts and forces sampled once per environment step are neither all simulator substeps nor contact impulses.
- Frozen normalizers and selected early/middle episode ages limit generalization to online occupancy. Four noise replicas quantify conditional Monte Carlo error, not state-population or training-seed uncertainty. Continuous Beta scores omit clamping atom corrections; probability alone does not bound gradient bias.
- Strongest existing frozen full-episode physical evaluation: sample9030.22, mean10030.00, mode10207.08. Mode still leaves9792.92 points to20000; removing stochasticity does not explain the full gap. Direct torque cost accounts for only a small fraction of sample-versus-mean loss.

## New nonlinear JVP contract (unexecuted)

`tests/test_sepclip_closed_loop_nonlinear_jvp.py` independently evaluates actual saved Adam and Anvil4M/8M first/full-update ambient chords through a compiled primal path, compared with the audit compiled JVP. It includes all actor banks and six hidden gate/up/down matrices, head and gain motion. Eight centered finite-difference sizes cover1/32 through4 times the chord. It reports effective FP32 rounded parameter motion, derivative relative error/cosine, and an heuristic output-rounding diagnostic (not a mathematical error bound). Two adjacent useful sizes must pass3% relative error, cosine>.999, rounded motion error<2%, and output roundoff floor<3%. These are preregistered numerical contract tolerances, not established empirical results. All sweep measurements remain visible so curvature/quantization failures can be diagnosed.

Existing closed-loop contracts covered exact bias-only JVP and zero direction but lacked nonlinear saved-chord finite differences. The new contract closes that gap only after a queued CUDA pass. It does not validate continuous-score sampling derivatives or certify statistical PG signal.

Recommended queue command (root submits after independent review):

```bash
mlq submit --name sepclip_microscopy_nonlinear_jvp_contract --max-parallel-runs 1 --cwd "$PWD" --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env CLEANRL_ENV_SPIN=5000 -- .venv/bin/python -m pytest -q -s tests/test_sepclip_closed_loop_nonlinear_jvp.py
```

Independent geometry reviewer approved the AD-independent ambient chord contract and requested4M coverage as well as8M; implemented before execution. Vector JVP agreement alone cannot certify a near-cancellation score-weighted PG sign.

Each printed numerical row includes exact checkpoint/trace SHA256 and the test, closed-loop, dense, physics and frozen trainer source SHA256. Checkpoint and trace hashes are checked again after evaluation.

## Strong saved physical reference addition

New `scripts/audit_sepclip_strong_reference.py` SHA256 `9c2f313d5bfcfb23da6e34785acab3b2ec290a38c8a59b8aeedf20e04d15e7df` and `tests/test_sepclip_strong_reference.py` SHA256 `5a61b5e675192b014d17615a949dba59bf6ac78f8fdfd2cb24d9971f5e5baa5d` are syntax checked and independently reviewed by architecture committee, not executed by this reviewer. They strictly load the inventoried50M pre-RMS control checkpoint through its correct Agent architecture; pin checkpoint SHA256; validate16 exact normalizer rows, epsilon/clip, state keys and physical bounds; and evaluate16 paired seed1 episodes per sample/mean/mode with host-vs-compiledCUDA parity checked on every visited observation. Full joint/contact traces are optional.

Pre-execution independent review corrected a semantic defect: rounded FP32 Beta alpha or beta equal1 is valid, and both1 is uniform. Actual rounded-law mode uses excess_a/(excess_a+excess_b), with explicit midpoint uniform tie; addition before subtraction would falsely erase a nextafter1 excess. Dedicated boundary/uniform/nextafter contracts now cover this. Intended logit-excess mode discrepancy and rounded shape counts remain telemetry. Historical sepclip mode and this actual-rounded-law mode convention must be distinguished when discrepancies matter.

This50M reference has unmatched budget and architecture and historical execution source unpinned. Physical replay does not select causal training, explain optimizer/PG mechanisms, or establish an optimum. No ML execution or submissions were performed by this reviewer.
