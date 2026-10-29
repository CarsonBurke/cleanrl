# Independent v14 physical replay acceptance and feedback interpretation

Reviewed the three completed saved-policy probes using only source reads, mlq state reads and model-free NumPy/SciPy arithmetic over existing JSON. No models, GPU work, tests or new jobs were run. Exact full source, training-execution, checkpoint, trace, producer-execution and JUnit hashes plus all recomputed signed contrasts are in `v14_closed_loop_independent_physics_numbers.json`. This note introduces no field-reliability or benchmark-efficiency result.

## Provenance and numerical acceptance

| Capture | Job / attempt | Artifact SHA256 |
|---|---|---|
| Adam4M | 11900 / 9485 | 772f959c14ec220438c5883fa20203e40fc7a4d28468535c21f378e001b76042 |
| Adam8M | 11902 / 9487 | 5cf5c1b8025d8b68a66570ca6a411fd817100d9dcc465b185eda0801e3dbd75e |
| Anvil4M | 11904 / 9489 | 9243b264c8dbdbc600773cbc14ab0e89ace8faed6210588d2294e32ffdb4c941 |

All three jobs succeeded at exit0 with maxParallelRuns1. Existing recovered replay/driver gate11897 has49 JUnit cases and chord gate11898 has16; both contain zero failures/errors/skips. I inspected their saved XML, not reran them. Producer source hash9bb20ffde6c2e3aa3c73572f38edd960658e49fe9f3278f6875f79b0a4d8e493 and loader7edfc432775a0098a60d63cb3c8642c65a599c3e8ae8fa2cc2f3354885d1dcf9 match each producer execution and current bytes. Nested original training execution source/environment hashes also reconcile to current bytes. Failed historical gates/attempts and unrelated queued jobs were preserved.

The strict loader dispatch is the actual pinned v14 additive pre-RMS module/Args, including Adam and Anvil through their recorded args. Recorded old alpha, beta, values and logprob errors are exactly0, as are frozen-row observation normalization and native/physical historical action errors. Snapshot/trace/manifest/training execution byte hashes match the CLI pins and artifact. Floor-shape counts are1132,3123,495 in Adam4M/Adam8M/Anvil4M captures: rounded shape=1 is valid and retained. No old logits were independently saved. Actual current compiled logits are not an independent historical latent-logit certification.

Each trace contains ten actual updates; endpoint is exact saved full model_after_update. First bank is old FP32 weights plus the stored first retracted delta, not an independently saved first model. Cumulative endpoint replay errors are1.4552e-11,0,1.1642e-10 respectively; constrained-axis unit error is at most1.1921e-7. The actor replay uses every actual bank parameter and actual future source forward, with no v7 reconstruction, tangent projection or optimizer-state replay.

Each artifact uses32 selected own training states, seven arms, four independent sampled noise replicas, and one deterministic mean/mode continuation. All numeric JSON leaves are finite. Every sampled/deterministic replica has integration restore and saved physical first-transition error0; the producer restores again after that check. Maximum initial ages400,395,400 plus H512 remain below the1000-step reset boundary; producer rejects every observed termination/truncation. This is within-no-reset conditional replay, not a reset episode benchmark.

Independent clip(raw_reward/per-state reward_std,−10,10) then FP32→FP64 reward reconstruction is bitwise exact. Raw and normalized finite discounted/undiscounted sums reconstruct to at most3.411e-13. Every paired gain and four-way interaction array reconstructs exactly, and aggregate paired t3 intervals match. Native FP32 action-square/sum with runtime scalar-cost dtype is checked against env info in the producer at1e-10; independent accumulated raw reward=velocity−control cost discrepancies are at most3.553e-12. Action-space geometry/bounds and inverse-CDF finite checks are fail closed; artifacts do not save every action, so this is producer-contract acceptance rather than a fresh all-action replay.

Tail dtype detail: gamma**H multiplied by the FP32 terminal array rounds that product to FP32 before adding to FP64 finite returns; raw tail then scales by each frozen reward_std. Reconstructing this actual law is bitwise exact. A wholly FP64 product differs by at most3.945e-5 raw at H64, and roughly2.8e-7 at H512. This discrepancy is numerical rounding, not critic error. Old fixed critic tails are separate augmented objectives, not true infinite returns.

## Paired finite effects

Intervals below are pointwise95% t3 continuation-noise intervals over four state-average replica contrasts, conditional on these32 fixed states. They preserve all seven arms inside a replicate. They are not intervals over32 independent training seeds. No family adjustment was prespecified; nested horizons and raw/normalized/tail metrics are dependent analyses.

| H512 capture | First full | Endpoint full | Endpoint−first | Endpoint t0 only | Endpoint after t0 | Endpoint interaction |
|---|---|---|---|---|---|---|
| Adam4M | -0.925 [-2.248, +0.399] | +0.035 [-3.975, +4.046] | +0.960 [-3.615, +5.535] | +1.382 [-1.001, +3.765] | -9.879 [-21.199, +1.442] | +8.532 [-0.402, +17.466] |
| Adam8M | +2.615 [-2.837, +8.068] | +3.460 [+0.384, +6.535] | +0.845 [-4.549, +6.238] | +4.322 [-1.929, +10.572] | -4.203 [-12.595, +4.190] | +3.341 [-12.671, +19.352] |
| Anvil4M | +1.123 [-1.325, +3.570] | -3.208 [-5.935, -0.481] | -4.331 [-7.543, -1.119] | -0.790 [-4.990, +3.410] | -1.454 [-7.336, +4.428] | -0.964 [-10.287, +8.358] |

Adam8M endpoint gains in all four aggregate replicas (+4.678,+2.421,+5.447,+1.293), although only15/32 state-mean gains are positive. Its H512 normalized endpoint is+.015640 [.003614,.027665], raw fixed-old-tail endpoint+3.425 [.268,6.582]. The preserved Anvil4M endpoint loses in all four aggregate replicas (−1.034,−5.188,−3.036,−3.575), with10/32 state-mean gains positive; normalized endpoint−.026057 [−.047489,−.004624], raw fixed-old-tail−3.211 [−5.945,−.477]. These supportive objectives are correlated views of the same continuations. H256 Anvil endpoint−first is−4.472 [−7.186,−1.758], while endpoint full−3.152 [−6.446,.143] includes0. Adam8M H64 andH256 full endpoint intervals also include0. Familywise reliability is not established by these pointwise signs.

Adam4M is a concrete feedback counterexample: H64 endpoint after-t0-only−2.417 [−3.608,−1.226], interaction+1.942 [.184,3.700], while full endpoint−.168 [−4.091,3.755] is unresolved. H256 later-only−9.537 [−18.813,−.261], interaction+8.152 [.757,15.548], full+.199 [−3.032,3.431]. The H512 uncertainty grows and neither component excludes0. These are exploratory conditional observations; repeating nested horizons does not provide independent confirmation.

The exact finite identity is: full gain = t0-only gain + after-t0-only gain + (Q_new/new − Q_new/old − Q_old/new + Q_old/old). The interaction contains both the changed state caused by the first action and the changed future actor responding to that state under the fixed exogenous quantiles. It is not merely a per-step reward derivative or a critic term. Later-only deliberately retains the old first action and thus follows a different subsequent state path from full. Nonzero interaction forbids interpreting first-action credit plus independent later credit as the full update effect. Mean/CI of the interaction must come from the same paired four-way arrays, not independently summed arm SEs.

At H512 the Adam8M endpoint t0/later/interaction intervals all include0 despite the full gain; Anvil4M also has unresolved component intervals despite a full loss. Therefore these results identify only the finite joint effect under the stated continuation law, not a unique microscopic cause of either sign. Adam4M suggests potentially consequential feedback coupling but cannot prove that mechanism explains its entire training curve.

| H512 deterministic capture | First mean / endpoint mean | First mode / endpoint mode |
|---|---|---|
| Adam4M | +0.785 / +3.091 | +4.028 / +7.082 |
| Adam8M | -5.237 / -8.618 | +8.600 / +10.706 |
| Anvil4M | +3.462 / +3.476 | +1.588 / -3.898 |

These deterministic effects have no continuation-noise interval and no seed/state-population certainty. Mode is the actual rounded-shape law using separate excess subtraction; when both shapes equal1 it selects the midpoint from nonunique uniform modes. Adam8M mean loss versus mode gain, and Anvil4M mean gain versus mode loss, show why a single deterministic controller cannot stand in for the sampled training objective.

## Physical units and remaining identification limits

H512 sampled endpoint velocity/control-cost contributions are Adam4M−.07664/−.11194, Adam8M+3.41596/−.04389, Anvil4M−3.22281/−.01481. Reward gain equals velocity contribution minus control cost contribution. These are discounted sums of m/s and environment control penalties, not final running velocity, mechanical work or episode score. The mean signs mostly follow velocity rather than control cost, but this algebra alone does not locate a joint, contact or weight-layer cause.

Own reward_std ranges are Adam4M201.233–235.388, Adam8M206.625–241.948, Anvil4M119.420–126.285. The three independently selected occupancy/state/normalizer populations differ. Raw rewards share physical units; that does not make the populations matched. No across-checkpoint matched-state covector, across-optimizer efficiency, chart selection, dense-field reliability or optimizer superiority follows from comparing these contrasts.

No reward clipping is realized at any arm/mode/horizon. Native clip atoms are realized: sampled H512 old/endpoint total counts over4×32 trajectories are Adam4M2/5, Adam8M3/2, Anvil4M1/2. Those are dimension events accumulated over512 steps, not episode counts. Boundary shapes are legal and frequent. Shared quantiles lie in[1e-7,1−1e-7], and native clipping/FP32 shape/draw rounding define this diagnostic sampling law; native host-sampler RNG equivalence is not certified. Finite interventions under this law remain measurable, whereas exact native-law score PG is neither estimated nor certified by this driver.

Disposition: accept the actual-source saved-state finite interventions and their arithmetic/provenance gates; retain Adam8M positive and Anvil4M negative H512 sampled endpoint observations as exploratory pointwise conditional effects. Retain the seven-arm feedback identity and deterministic control reversals as evidence against substituting first-action or mean/mode credit for full sampled feedback. Do not promote this evidence to a global optimizer cause, true-value/GAE bias mechanism, gait optimum, infinite-return conclusion or benchmark score claim. Failed original attempts remain preserved.
