# Independent empirical gait and action-intervention review

Job11886 succeeded, attempt9478, maxParallelRuns1; recovered split gate11885 passed21 contracts. Original failed/skipped jobs remain separate. Artifact gait_dense_recovered_v2.json SHA256 is c0d3d1d9296949be50e5027530ca859460009411be8e2528ea8c30f3744c7d4a. Executed gait sourcecc5f8b35167c3beffcc71561474dcf81c79f3ad68b7b82ab35816f04f4d7e68e matches current/frozen source. Detailed independent JSON/NPZ arithmetic, trace/snapshot/input/manifest hashes are saved in gait_dense_recovered_v2_independent_numbers.json. No model/GPU/test/job execution or submission occurred in this review.

## Acceptance and population

All four Adam/Anvil4M/8M snapshots match saved byte hashes; all numeric artifact leaves and all twelve saved traces are finite. Each trace has1000x8 episodes, frozen normalization rows0..7, dt=.05. Returns reconstructed from saved rewards agree within2.01e-11 summation-order roundoff (a strict bitwise axis-reduction comparison was unnecessarily stronger than the source's separate episode reductions). Recomputed native FP32 action-square/sum followed by promoted control cost agrees with qpos displacement velocity within4.44e-16 for999 observable transitions. The final next position is not saved. Command torque is action times actuator gear, commanded power uses that torque times same pre-action joint velocity; control penalty .1*sum(action²) is a different normalized-action reward cost, not mechanical energy.

Host/CUDA actor parity errors on the recorded probe are5.72e-6 at4M and9.54e-6 at8M, under the frozen1e-4 gate. Unlike the strong-reference driver, this does not check every visited observation. The existing frozen gait trace code also does not explicitly assert terminal flags during the1000-step reset episode; standard HalfCheetah semantics and complete finite traces are the relevant evidence, not a newly measured flag inventory.

Effective local action-oracle population is32 dense training states per capture, despite protocol.states=12 default: matched dense input replaces the reset-trace discovery population. Raw observations, native actions, per-state means/variances, t,row,age and stored advantages match dense_pg_controls.json exactly in order. Candidate continuations restore captured integration fields and use those state's own frozen normalizers. The gait artifact does not itself report a numerical restoration/saved-transition replay error; upstream dense matched-state replay was zero on the identical snapshots/states, and the gait restore primitive is source-reviewed. Do not falsely report a new gait-specific zero replay measurement. Dense source hash is attached through execution/protocol rather than a dedicated input digest field; this review pins its current4928741a… bytes independently.

## Descriptive reset gait

| Capture | Sample return / mean velocity | Mean-policy return / velocity | Mode-policy return / velocity |
| --- | --- | --- | --- |
| Adam4M |6740 /7.052|7625 /7.912|7949 /8.313|
| Adam8M |8575 /8.901|9376 /9.683|9530 /9.902|
| Anvil4M |6412 /6.714|7786 /8.064|8256 /8.602|
| Anvil8M |8792 /9.120|9709 /10.017|9946 /10.322|

Anvil8M sample/mean/mode control costs average .32776/.30830/.37632. The50M strong reference's corresponding speeds are15.330/16.154/16.604 and costs .40367/.38398/.42995. Faster reference behavior is compatible with higher actuator cost; neither low cost nor more deterministic control alone explains the gap. Counts of reset states differ8 versus16, training budgets/actors differ, and frozen normalizer rows belong to different policies. These are physical descriptors, not matched benchmark or causal architecture comparisons.

Observed v12 sample rear/front contact duties are7.50%/4.65%(Adam4M),7.16%/3.54%(Adam8M),7.45%/7.09%(Anvil4M),6.64%/3.55%(Anvil8M). Sample rear median onset periods are.55/.55/.55/.60 seconds; front .80/.95/.60/.95 seconds. Mean observed root displacements between contact onsets span5.28–7.17 rear and6.12–13.00 front. These intervals are between SAMPLED final-solver-substep contact events and can skip physical contacts; they are not certified strides.

That limitation is exposed by the reference: deterministic mean/mode full velocities16+ coexist with very sparse observed complete cycles whose aggregate velocity is only about4.2–4.65. Those complete cycles primarily cover early, slower transient windows; absent steady sampled contacts do not mean absent contact impulses. All reference median observed stance durations are.05 seconds, exactly one control-frame resolution. Contact/actuator forces are read before the current action at the prior step's final solver substep. Sparse sampled-contact duty, foot phase and commanded joint power are useful descriptors, but cannot establish stance time, flight, impulse efficiency or a causal gait class without finer physical evidence.

## Conditional first-action interventions

The old actor controls all continuation steps. Candidate native-action perturbation±.025 means physical action±.05; the normalized six-joint directed candidate has native norm at most.025*sqrt(6)=.06124 and physical norm at most.12247, after bounds. Mean/mode candidates can move farther and are not equal-budget perturbations. Discovery uses two independent continuation streams; four separate held-out streams evaluate frozen choices. Every state/candidate shares continuation uniforms within a replica, and validation never selects the candidate. Actions are17 candidates including horizon-specific directed proposals; H64/H256 targets remain separate finite discounted RAW returns.

Below, independent conditional intervals use four validation-replica STATE-AVERAGE paired gains and t3=3.182; source selected_gain_summary SE describes between-state variation and must not replace this MCSE. A state-positive fraction also is not a direction-confidence interval.

| Capture | H | Discovery-selected gain, pointwise95% | Directed gain, pointwise95% |
| --- | ---: | --- | --- |
| Adam4M |64|-.930 [-5.788,3.929]|-3.251 [-5.695,-.807]|
| Adam4M |256|-5.562 [-19.489,8.364]|-.312 [-11.485,10.861]|
| Adam8M |64|-.176 [-6.979,6.626]|1.980 [-1.690,5.650]|
| Adam8M |256|4.422 [-3.287,12.130]|.178 [-16.240,16.596]|
| Anvil4M |64|-2.610 [-5.516,.297]|-.065 [-8.322,8.193]|
| Anvil4M |256|-6.020 [-20.580,8.540]|.175 [-4.217,4.567]|
| Anvil8M |64|1.476 [-2.051,5.004]|2.970 [-4.151,10.090]|
| Anvil8M |256|-7.619 [-27.979,12.740]|-4.705 [-20.299,10.890]|

No positive discovery-selected or directed aggregate clears even a pointwise95% sign gate. Adam4M/H64 directed has a pointwise negative interval, but the multiple captures/horizons/candidate families were not registered for a familywise sign claim. Anvil8M/H64 directed improves26/32 state means, yet its aggregate interval crosses zero. Choosing the best fixed candidate after inspecting these held-out fields consumes that holdout and adds multiplicity. Local finite action effects cannot be multiplied by episode length, called closed-loop policy improvements, or converted into infinite-return PG/optimality.

## Dense contact/PG attachment

All saved raw-PG phase means and counts recompute exactly from the matched dense fields and restored contact masks. Phase codes0/1/2/3 denote neither/front/rear/both CURRENT contact, not temporal stride phase. Counts are25/2/5/0(Adam4M),25/3/4/0(Adam8M),28/1/3/0(Anvil4M),30/1/1/0(Anvil8M). Empty both-foot groups remain null. Single-state front/rear Anvil8M means are not population mechanisms or replicated contact causal evidence. Attached dense fields inherit unresolved two-replica reliability and draw/score law gaps. This is conditional descriptive grouping; contact was not manipulated in that calculation.

Disposition: accept complete source-matched physical descriptors and the finite held-out action-intervention measurements, including failures/noisy gains. No positive aggregate intervention or microscopic gait/gradient/optimizer cause is established by this artifact.
