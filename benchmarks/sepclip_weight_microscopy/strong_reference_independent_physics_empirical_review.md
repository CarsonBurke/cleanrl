# Independent empirical review: saved 50M Beta reference

Accept11827 as a complete physical reference measurement, with the limitations below. Job11827 succeeded, attempt9469, exit0, maxParallelRuns1; prerequisite11826 succeeded with five CUDA contracts passed. This review read source/manifests/JSON and saved NPZ traces and performed CPU NumPy/SciPy summaries only. No checkpoint model was deserialized, no GPU/model/test/job was executed or submitted.

strong_reference_physics.json SHA256 is c82012104073e6c039d71001b71b225348ff223f6078cdbb28ee91bf74a7d3b4. The checkpoint byte hash matches1f09a49740f656482f4e01a71f1070cfe13ebd1e44a52210eca8db73f0089532. All thirteen reported source hashes match current files and both the contract/evaluation execution manifests, including audit9c2f313d…, prerms_adamh_v22 trainer514cfd7c…, frozen physics1d53db1d…, gaitcc5f8b35… and host/environment/normalizer dependencies. Exact trace and execution-manifest hashes and independently computed numbers are recorded in strong_reference_independent_physics_numbers.json.

## Completion and parity

All three saved traces contain1000 steps x16 episodes: actions/concentration/mean/mode(1000,16,6), velocity/cost(1000,16), qpos/qvel(1000,16,9), geometry contact/force(1000,16,8). Every numeric array is finite, dt=.05, normalization_rows exactly0..15. Source execution requires no termination or early truncation and all16 TimeLimit truncations exactly at the1000th step. All three modes reset with seed1, so initial states and normalization rows are paired. This is full-episode reset evaluation, not a matched saved-occupancy first-action fork.

Saved normalizer metadata is complete FP64 means/variances(16,17), counts(16), epsilon1e-8 and clip10. Per-row array hashes are recorded; the executed strict-loader contract validates those arrays and architecture Args. Normalizers remain frozen throughout evaluation. Historical execution source_pinned=false remains explicit: present source strictly loads the checkpoint, but historical trainer byte identity is not independently certified.

Host/CUDA forwards are checked at every visited observation, and permanent output buffers are copied. Worst visited absolute logit/alpha/beta discrepancies across all modes are each at most6.104e-5. The executed criterion is2e-5+2e-6*max(1,global output scale) for the current batched output, not a per-element relative guarantee. All visits passed that criterion. This is policy execution parity, not training parity or a derivative oracle.

## Physical outcome

| Frozen control law | Mean episode return | Episode SE | Mean velocity | Mean control cost | Mean Beta concentration |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sample |14925.93|69.53|15.3296|.403672|30.7986|
| Mean |15769.63|27.80|16.1536|.383979|32.0985|
| Actual rounded-shape mode |16173.90|48.76|16.6038|.429950|31.8274|

Return ranges are14025.97–15199.18,15445.63–15873.63 and15527.52–16320.46, respectively. Each return reconstructs exactly from the saved sum of velocity minus stored control cost. Independent pre-action root-x displacement over dt agrees with the reported velocity to3.58e-8 for the999 transitions whose next pre-action qpos is saved. The final transition position is not independently present in the NPZ, but its return is included.

Minor telemetry arithmetic limitation: the reference audit multiplies the FP32 action-square/sum by cost weight in FP32, while the native environment promotes the FP32 square/sum before the scalar multiplication. Recomputed native costs differ from stored costs by at most3.58e-8; recovered velocity differs by the same order. Raw environment returns are unaffected. This is disclosed numerical telemetry roundoff, not a return or protocol blocker. No source was edited.

On the16 paired reset/normalization rows, mean-minus-sample is+843.70 (SE55.29), and mode-minus-sample is+1247.97 (SE42.31). Every row's difference is positive: minimum+577.25 and+987.67. Approximate pointwise t15 intervals are[725.85,961.56] and[1157.78,1338.16]. These are descriptive paired-row summaries: there is one stochastic sampled episode per row, the frozen rows/initial states differ, and this does not estimate conditional action-noise variance separately or independent training-seed uncertainty. They are not multiplicity-adjusted superiority certificates.

Concentration is measured at visited observations, so its mode-to-mode difference includes changed occupancy. Maximum observed concentrations are57.41/55.12/55.44. Rounded shape-one events are real valid Beta boundaries: sample alpha_equal_one243/beta1622, mean169/287, mode142/243, out of96000 state/joint observations per mode. Both shapes equal one never occurred. Actual mode uses excess_a/(excess_a+excess_b), including boundary0/1 and a stated uniform midpoint tie if needed. Intended logit-excess versus actual rounded native mode maximum differences are6.32e-6/7.34e-7/1.46e-6; none of the intended modes was nonfinite. An interior r/ell chart would nevertheless need to withhold these actual shape-one coordinates rather than clamp them.

## Action and gait observations

Mean summed squared action changes per control step are5.685/5.356/6.399. Sample action magnitude exceeds.95 in30.38% of joint/step observations; mean and mode fractions are6.00% and57.55%. The fast mode controller also has the largest average control cost and action-change metric; faster behavior is not universally lower actuator effort. Concentration around31 is compatible with this achieved controller, neither necessary nor sufficient in isolation.

Only named rear/front feet appear in sampled ground-contact telemetry. Across16000 observations per mode, rear/front counts are139/59(sample),95/34(mean),89/39(mode); duty fractions are .8688%/.3688%, .5938%/.2125%, .5563%/.2438%. Other six named geometries have zero observed contact. Sample mean normal forces averaged over all observations are10.66(rear)/3.61(front); mean6.87/2.02 and mode7.85/2.42 as approximately recorded. These are sparse pre-action observations of the last solver substep of the prior control step. They are not integrated contact impulses or high-resolution physical stance fractions; do not infer flight duration, contact efficiency or a causal gait mechanism from the small duty values. Step metric SEs ignore temporal correlation and cannot supply16000 independent physical observations.

## Comparison limits

This saved prerms_adamh_v22 controller establishes that the Beta family and this environment/physical evaluator can exceed a universal10k policy ceiling: sampled return is14.93k, deterministic mode16.17k. It does not establish that the sepclip actor reaches those scores under8M training, or that a particular normalization/gradient/optimizer mechanism causes the difference. Reference training used50M steps and a different actor/optimizer configuration; historical source execution is unpinned, and frozen-normalizer inverse-CDF evaluation differs from historical live sampling/online normalization. This is not a matched TD7 rerun, a20k benchmark result, or an optimum/value/PG oracle. Retain sampled versus mean versus actual-shape mode as distinct control laws and compare each only within its declared protocol. No benchmark causal or efficiency/chart-selection conclusion follows from this reference alone.
