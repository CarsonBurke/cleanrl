# Independent empirical review: actual v12 chords with matched dense credit

Job 11887 succeeded, attempt9479, maxParallelRuns=1. This review used source reading, JSON/scalar arithmetic and streaming hashes only; no models, GPU, tests, or queue mutation. Detailed independently recomputed contrasts are in actual_step_dense_recovered_v2_independent_numbers.json.

## Provenance and capture scope

Artifact SHA256 `cb5d01beb923b46dc810633cf6a0c1a84cc4a165bd51eb03acaad04abc4d98ef`; execution manifest execution_sources/sepclip_microscopy_actual_step_dense_rank_recovered_v2_1791340654568107497.json SHA256 `a8360c1f9b581f31acc4f8123a59df54d107819d33776d09b799d434e4b371c0`. Producer SHA256 `0d23d4b61059a255b7c566fd8088ebea629f357905a3fd7b35928cbb5b848953` matches artifact, manifest and current bytes. All seven recorded imports match current bytes. Oracle dense_pg_controls.json SHA256 `4928741aade1d67d125f765186c397ef7a3560e3d94999c2ffd8e0a10e1531e5` matches all rows and disk.

One broad execution-inventory file has subsequently changed: NEW scripts/audit_sepclip_prescription_layer_geometry.py (recorded 4c9ddf963e1c36ea45deaa020f9932cb4f032dcd0994a97b90b6ce796c8b16fa; reviewed current 8c0f22fc08f9eb9b8535127f36a1ae67b4450c763b7014012b738d0a156e1046). It is absent from the consumer import inventory and not executed by this command. This unrelated preparation drift does not alter actual-step producer/replay provenance. Other manifest source/environment hashes matched at review time.

All four snapshots and four traces independently match their recorded byte pins (streamed without deserialization):
- /home/marvin/Documents/repositories/cleanrl/runs/HalfCheetah-v4__sepclip_microscopy_v12_adam_control__1__1791260590/diagnostic_snapshot_4013696.pt: `8b17da17c1a4d6ed31e62076f76debfc01763e6f0a8b95f401674e55415d6197`.
- /home/marvin/Documents/repositories/cleanrl/runs/HalfCheetah-v4__sepclip_microscopy_v12_adam_control__1__1791260590/diagnostic_updates_4013696.pt: `762e1629a4d2b547fa3d5442c081c2ff99ea62c05c3f97a46477304b14f9fa6b`.
- /home/marvin/Documents/repositories/cleanrl/runs/HalfCheetah-v4__sepclip_microscopy_v12_adam_control__1__1791260590/diagnostic_snapshot_7995008.pt: `5d7979136fb15329c9bd6d0e6340cde6206d8b0956c21f273f223bbb851b3f30`.
- /home/marvin/Documents/repositories/cleanrl/runs/HalfCheetah-v4__sepclip_microscopy_v12_adam_control__1__1791260590/diagnostic_updates_7995008.pt: `7cdb678b6a24b260f543cca4771c234e6338bdf2de8d6fd4c428d7a18cd38926`.
- /home/marvin/Documents/repositories/cleanrl/runs/HalfCheetah-v4__sepclip_microscopy_v12_anvil_control__1__1791260697/diagnostic_snapshot_4013696.pt: `18d6879248fb472d9d84c4254b5052918f2432bf731c398f0f87ad8138ae56cb`.
- /home/marvin/Documents/repositories/cleanrl/runs/HalfCheetah-v4__sepclip_microscopy_v12_anvil_control__1__1791260697/diagnostic_updates_4013696.pt: `25709629234dc4f8bd43f9c6597247d25557be071702d8bfdb66895c2c2ffbde`.
- /home/marvin/Documents/repositories/cleanrl/runs/HalfCheetah-v4__sepclip_microscopy_v12_anvil_control__1__1791260697/diagnostic_snapshot_7995008.pt: `06819047d869a87baa62ea4d4f4e4a18044f0c52c71fa3beea2b9a66cab7c587`.
- /home/marvin/Documents/repositories/cleanrl/runs/HalfCheetah-v4__sepclip_microscopy_v12_anvil_control__1__1791260697/diagnostic_updates_7995008.pt: `1bb42d0cc7faa006422f055f5f43264a0782356d8703b6b340618b5a455efe1b`.

Each capture has 32 matched selected states, 16 first-action samples and two independent full replicas, H256/H512, gamma .99, seed1. These are the NEW matched dense fields, not historical11781/physics_complete.json. Consumer source validates selected state mapping against its CUDA-loaded snapshots and saved initial logits (actor max error1.90735e-6–7.62939e-6); check exact values in artifact.

Trace scope is exactly 10 optimizer updates across 10 epochs (one full minibatch per epoch), starting at snapshots4013696/7995008. First bank is old FP32 weights plus the first stored actual retracted FP32 delta; there is no independently saved first model. Endpoint is exact model_after_update after all10 epochs, not cumulative delta sums. Replay cumulative errors are2.32831e-10,8.52651e-14,2.32831e-10,0 in Adam4M/8M,Anvil4M/8M order; these do not prove first-bank bit recovery. Saved unit-axis residuals remain FP32-sized. Source uses the compatible legacy v7 actor factory, strict complete state loading, and applies old-FP64 unit-row/column tangent projection only in the explicitly tangent comparison.

Queue dependency11776 numerical/source contracts succeeded;11885 gait+actual scalar contracts report21passed. These suite passes are not blanket independent certification of every measured full actor Jacobian or true physical gradient law. Failed/skipped prior attempts remain distinct.

## Fixed actual motion scalar descriptive gate

For each of4 captures ×2 horizons ×2 timing banks ×3 motions, each objective yields48 contrasts. The three motions are total old-base actor-Jacobian chord response, old-base tangent-only chord response, and actual finite logit difference. Pairing is sum over12 logit coordinates and mean over32 states. The same direction is used for each objective; no direction is fitted. R2 mean, SE=abs(replica0−replica1)/2 and t95 halfwidth=12.7062047364×SE were independently recomputed; all summary, interval and mean-field pairing differences are at most8.88178e-16. Paired tail subtracts SAME-replica bootstrap-minus-finite values before computing its SE.

Gate: approximate conditional t95 interval excludes zero AND halfwidth <= .5*abs(mean). This preregistered descriptive gate is not familywise inference, an exact R2 normality certificate, a between-seed/state CI or a sampler-bias/true-PG certificate. The six successes are strongly dependent variants of one endpoint contrast.

| Objective | Contrasts | Exclude zero only | Pass full descriptive gate |
|---|---:|---:|---:|
| finite_raw | 48 | 3 | 0 |
| finite_normalized | 48 | 3 | 3 |
| bootstrap_normalized | 48 | 3 | 3 |
| paired_estimated_tail_increment | 48 | 3 | 0 |

Thus6/144 objective contrasts pass;0/48 paired-tail contrasts pass. All6 are NEGATIVE Anvil8M/H512 ten-epoch endpoint pairings. No first-update contrast passes.

| Objective / motion, Anvil8M H512 endpoint | Mean | Conditional t95 interval | Halfwidth / abs(mean) |
|---|---:|---:|---:|
| finite_normalized / actual_endpoint_jvp | -1.1179345e-05 | [-1.557547141e-05, -6.78321858e-06] | 0.39323649 |
| finite_normalized / tangent_only_jvp | -1.116295559e-05 | [-1.54329374e-05, -6.892973784e-06] | 0.38251355 |
| finite_normalized / finite_actual_logit_change | -1.118227565e-05 | [-1.558059781e-05, -6.783953484e-06] | 0.39332979 |
| bootstrap_normalized / actual_endpoint_jvp | -1.112345369e-05 | [-1.364843215e-05, -8.598475222e-06] | 0.22699591 |
| bootstrap_normalized / tangent_only_jvp | -1.110704554e-05 | [-1.350686516e-05, -8.707225911e-06] | 0.21606282 |
| bootstrap_normalized / finite_actual_logit_change | -1.112652381e-05 | [-1.36536124e-05, -8.59943521e-06] | 0.22712292 |

Raw H512 endpoint pairings also exclude zero in all3 motions but fail the halfwidth gate (total: mean−.00272780219, CI[−.00432538620,−.00113021817], relativehalfwidth≈.585668). The corresponding normalized credit is about−1.118e-5. Normalization is a per-selected-state frozen scaling/clipping objective, not a universal constant rescale. Although sampled reward clips are zero, no raw-law interpretation follows from normalized reliability.

The paired estimated-tail increment along that total endpoint is +5.58913e-8, CI[−1.81526e-6,+1.92704e-6], unresolved; it is not the true critic bias. At Anvil4M/H512 first update, all3 paired-tail motions exclude zero but fail the full precision gate (total+5.21734e-6, CI[+1.03366e-6,+9.40101e-6], relativehalfwidth≈.801879). Bootstrap/finite agreement at the negative endpoint is a robustness observation under this old-tail estimate, not proof tail bias is absent.

## Actual finite motion, costs and precision

| Capture / motion | JVP Fisher Q | FP64 actual-shape KL / Q | Unweighted logit linearization relative error | Normal-vs-total JVP relative norm | Joint Q / diagonal Q |
|---|---:|---:|---:|---:|---:|
| Adam4M / first | 0.00528180834 | 1.00783066 | 0.0198933144 | 0.00017756136 | 3.84885422 |
| Adam4M / 10epochendpoint | 0.016227343 | 1.00487456 | 0.0625338692 | 0.00610069946 | 1.1840053 |
| Adam8M / first | 2.91180286e-06 | 0.999721022 | 0.00105072659 | 0.000459457202 | 4.47493124 |
| Adam8M / 10epochendpoint | 8.28721325e-05 | 1.00201279 | 0.00142866956 | 8.15574214e-05 | 3.98949155 |
| Anvil4M / first | 0.00218332069 | 1.00571998 | 0.0106677222 | 0.000355439575 | 3.66224578 |
| Anvil4M / 10epochendpoint | 0.0120306523 | 0.989440005 | 0.0400653001 | 0.00768382686 | 0.932072492 |
| Anvil8M / first | 2.23221083e-07 | 0.998081605 | 0.00395490921 | 0.00165095842 | 3.54226176 |
| Anvil8M / 10epochendpoint | 1.75113459e-05 | 0.999481843 | 0.000690549033 | 0.0002503877 | 3.58706532 |

These costs are horizon-independent saved policy motions, so the eight rows must not be counted twice across H256/H512. KL/Fisher ratios .989440–1.007831 and unweighted logit discrepancies .00069055–.0625339 are substantially better than noise-fitted proposals. Adam4M endpoint is the least linear (6.253%); Anvil8M endpoint is .0691% and its scalar finite-logit pairing has the same descriptive negative sign. Neither finite-logit pairing nor close KL agreement is a finite return or an all-time PG: score field concerns only the first action with the continuation policy fixed. Small global unweighted logit error cannot bound a near-cancelling scalar directional error without checking that pairing.

Old-base total chord JVP includes normal components of the actual retracted chord; tangent-only removes them. Normal contributions span .00008156–.00768383 of total response norm. Layer costs include all9 actor parameter groups (six body matrices, head weight/bias, gain). Gram sums/diagonals/cross sums independently agree within1.73472e-18. First-step joint/diagonal ratios3.54226–4.47493 show constructive cross-layer motion. Endpoints range .93207–3.98949; Anvil4M endpoint has net negative cross term. Do not allocate total cost by isolated-layer costs alone.

Fisher precision is FP64 softplus/sigmoid/trigamma reevaluated from recorded FP32 logits; it is not exactly the Fisher of the recorded rounded FP32 shapes. Jacobian starts FP32 and is cast to FP64 before tangent projection/SVD; an FP64 SVD does not recover lost derivative accuracy. Finite exact KL forms actual FP32 softplus+1 shapes then upgrades KL algebra. Local native mean/total-concentration metrics are not v13 actual-excess r/ell.

## Rank, retained actual motion and fitted-direction limits

| Capture | Configured rank | Effective spectral rank | Condition number | Ranks at cutoff1e-6 / 1e-5 / 1e-4 |
|---|---:|---:|---:|---:|
| Adam4M | 384 | 33.858618 | 107388.372 | 384 / 383 / 373 |
| Adam8M | 384 | 34.517475 | 4258793.11 | 382 / 372 / 353 |
| Anvil4M | 384 | 48.368301 | 152600.818 | 384 / 383 / 378 |
| Anvil8M | 384 | 31.629801 | 11775699.2 | 377 / 368 / 354 |

Configured384/384 rank uses relativecutoff2.94476e-12, not a justified FP32 derivative noise floor. At1e-4, the FIRST tangent motion retains at least .9999999998668 of whitened motion energy despite rank353–378. This measures actual first response lying in dominant sampled-state modes; it does not establish useful alignment with the true PG. Threshold-sensitivity fields explicitly concern FIRST tangent motion only. Endpoint retained-motion telemetry exists only at the configured cutoff and is effectively1 (tiny above1 by2e-13 is arithmetic noise); there is no endpoint1e-4 sensitivity claim in this artifact.

Cross-replica fitted-direction evaluation is on the SAME selected states with different action/continuation noise. It is not held-out-state generalization. Sixteen fitted proposals have finite KL / intended local budget3852.1–73,896,305 and finite whitened retraction error30.28–4588.71. Fit/evaluate reciprocal directions reuse the same two fields and are not independent confirmations. These fitted candidates are unusable finite-budget proposals here; algebraic reachable-energy/efficiency fields remain exploratory.

## Reliability disposition

The dense full-field issue persists: Adam8M signed dual cross-energy is negative at both horizons; other positives alone do not supply a confidence bound. Independent physics review reports Anvil8M state(t311,row10) contributes117.84% of signed H512 dual cross-energy, leaving the other31 states net negative. Full-field positive signal flags in the output are weaker than reliability acceptance. The six descriptive negative scalar contrasts may coexist with this unresolved field; they do not solve it. Do not remove the dominant state after observing this diagnostic.

The oracle draws truncated inverse-CDF uniforms [1e-7,1−1e-7], uses rounded/clipped native actions and smooth continuous Beta scores. Zero observed first-action clamps does not certify the exact native sampler/RNG/atom-score law or unbiasedness; future-action law is likewise diagnostic. All statements retain conditional first-action, fixed-selected-state, R2 and old-critic-tail limitations.

Accept provenance/replay and actual-motion descriptive measurements with these scopes. Withhold optimizer efficiency, near-optimum, chart preference, whole-policy update benefit and a causal explanation of benchmark differences. Physical closed-loop causal arms and prospective benchmark outcomes must be adjudicated separately; chartv13 remains deferred. Historical11781 was H64/H256 on historical unmatched physics fields and cannot be merged as an extra replica of this experiment.
