# Independent empirical acceptance: captured v14 full actor chords

Jobs11899/11901/11903 succeeded at maxParallelRuns1, after the prepared chord gate11898. This review used JSON/scalar arithmetic, source inspection and streaming byte hashes only; no models, CUDA, tests or submissions. Numerical evidence is for additive v14 Adam4M/8M and Anvil4M captures only. Upcoming NorMuon and layerV2 results are not assumed. Detailed pins and every requested step are in v14_captured_chord_independent_numbers.json.

## Provenance and replay

- adam4m: artifact SHA256 `cc4e58aa1a3203e658c4171219d1b40f5bf0dff220560bb6029e961036a5a434`; execution manifest SHA256 `0659c105c03c44cda797373c0b64a75b5552aab762fb4a9a400a5d0ee74bfdb9`. All recorded source/environment inventory hashes and four command pins (snapshot, trace, frozen source manifest, training execution manifest) independently match current bytes.
- adam8m: artifact SHA256 `388b641f3a031f37a4b8bac15caa07b48a867a5c4c91e3209b5c3e1070e3fe76`; execution manifest SHA256 `73023ed80cfd093286f0612d99075ce7646976e21240095a5952946272120c4f`. All recorded source/environment inventory hashes and four command pins (snapshot, trace, frozen source manifest, training execution manifest) independently match current bytes.
- anvil4m: artifact SHA256 `5661a617e7a9ca361bffe3b62035794577f61c7bad411a34cf49c925c81e232f`; execution manifest SHA256 `846c8040777eba32ab6a7aa6b6cf3bd17a7ca4538702f8f1f630611d2af0e662`. All recorded source/environment inventory hashes and four command pins (snapshot, trace, frozen source manifest, training execution manifest) independently match current bytes.

All recorded old alpha/beta/log-prob/value replay max errors are0. Frozen observation-normalization and native/physical action errors are0. Ten-epoch traces have10updates; cumulative endpoint reconstruction residuals1.45519e-11,0,1.16415e-10 in Adam4M/8M,Anvil4M order. First bank remains old+stored retractedFP32delta, not an independently saved first checkpoint; full endpoint is exact model_after_update. First direction uses storedFP32delta; endpoint chord is FP32 subtraction of exact saved endpoint and old.

Each chord covers ALL18584 actor parameters and ALL16384 saved observations. Separately compiled primal and AD-graph primal agree bitwise at the old bank in all three outputs for all six chords. Source uses strict fullgraph,dynamicFalse actual actor functional_call and actualsoftplus+1 shape decode, with neighboring FP32 ambient additions and no constraint projection/retraction. This is an actual captured-chord audit, not a sampled toy model or an oldv12 unit-body substitution.

## All requested step sizes

Error columns are relativeL2 versus smooth AD. Floor is max across latent/alpha/beta of the output-storage ULP heuristic relative to that channel JVP norm. Cosine is minimum across channels. Every row is measured, finite and representable; no best-h or monotonic-convergence selection is made.

| Capture / chord | h | Parameter-direction relative error | Latent / alpha / beta FD relative error | Minimum cosine | Max storage-floor ratio |
|---|---:|---:|---:|---:|---:|
| adam4m first | 1 | 7.451728e-07 | 6.889754e-05 / 0.0001335666 / 0.002001234 | 0.9999980152 | 1.899052e-05 |
| adam4m first | 0.5 | 7.158449e-05 | 4.882165e-05 / 6.333057e-05 / 0.0005128195 | 0.9999998703 | 3.798605e-05 |
| adam4m first | 0.25 | 0.0001225469 | 8.177924e-05 / 9.573643e-05 / 0.000149909 | 0.9999999888 | 7.599585e-05 |
| adam4m first | 0.125 | 0.0002377649 | 0.0001599719 / 0.0001928513 / 0.0001506076 | 0.9999999814 | 0.0001520083 |
| adam4m first | 0.0625 | 0.0004697172 | 0.0003244247 / 0.0003938469 / 0.0002961883 | 0.9999999236 | 0.0003040264 |
| adam4m endpoint | 1 | 2.28571e-07 | 0.0007807076 / 0.0009239923 / 0.0009480707 | 0.9999995843 | 1.140259e-05 |
| adam4m endpoint | 0.5 | 8.909479e-06 | 0.0001972099 / 0.0002346113 / 0.0002399248 | 0.9999999734 | 2.281556e-05 |
| adam4m endpoint | 0.25 | 1.592653e-05 | 7.473293e-05 / 8.030371e-05 / 8.441278e-05 | 0.9999999966 | 4.564341e-05 |
| adam4m endpoint | 0.125 | 3.071999e-05 | 0.0001117481 / 0.000107995 / 0.000119607 | 0.9999999929 | 9.130508e-05 |
| adam4m endpoint | 0.0625 | 5.862196e-05 | 0.000222619 / 0.0002186519 / 0.0002331228 | 0.9999999728 | 0.0001826196 |
| adam8m first | 1 | 0 | 0.001460206 / 0.001546853 / 0.00143155 | 0.9999988037 | 0.001199653 |
| adam8m first | 0.5 | 0.005852905 | 0.003186529 / 0.00347424 / 0.003024016 | 0.9999939724 | 0.00239932 |
| adam8m first | 0.25 | 0.01001608 | 0.005949879 / 0.006282674 / 0.005828851 | 0.9999802836 | 0.004798639 |
| adam8m first | 0.125 | 0.01905372 | 0.01204319 / 0.01269839 / 0.01185121 | 0.9999195783 | 0.009597241 |
| adam8m first | 0.0625 | 0.03800187 | 0.02375999 / 0.02518943 / 0.02323833 | 0.9996844252 | 0.01919432 |
| adam8m endpoint | 1 | 2.425394e-07 | 0.0001373095 / 0.0001488992 / 0.000145531 | 0.999999989 | 0.000106093 |
| adam8m endpoint | 0.5 | 0.0005183615 | 0.0002842104 / 0.0002812187 / 0.0003012706 | 0.9999999551 | 0.0002121939 |
| adam8m endpoint | 0.25 | 0.0009302361 | 0.000567119 / 0.0005695895 / 0.0005916372 | 0.9999998253 | 0.000424425 |
| adam8m endpoint | 0.125 | 0.001792577 | 0.001170445 / 0.001161891 / 0.001236088 | 0.9999992367 | 0.000848838 |
| adam8m endpoint | 0.0625 | 0.003526066 | 0.002292167 / 0.002280188 / 0.002413489 | 0.9999971094 | 0.001697672 |
| anvil4m first | 1 | 3.875844e-07 | 3.670911e-05 / 0.0001873328 / 0.0001221754 | 0.9999999829 | 1.963217e-05 |
| anvil4m first | 0.5 | 4.614276e-05 | 4.71347e-05 / 5.903871e-05 / 6.332761e-05 | 0.999999998 | 3.927678e-05 |
| anvil4m first | 0.25 | 8.136016e-05 | 9.033411e-05 / 7.319863e-05 / 0.0001079211 | 0.9999999942 | 7.855694e-05 |
| anvil4m first | 0.125 | 0.0001527382 | 0.0001811562 / 0.0001440596 / 0.0002165364 | 0.9999999767 | 0.0001571239 |
| anvil4m first | 0.0625 | 0.0003017651 | 0.0003644066 / 0.0002953217 / 0.0004317111 | 0.9999999088 | 0.000314265 |
| anvil4m endpoint | 1 | 1.578784e-07 | 0.0008155863 / 0.0009656666 / 0.001129184 | 0.9999994613 | 4.7045e-06 |
| anvil4m endpoint | 0.5 | 5.282426e-06 | 0.000203626 / 0.0002427635 / 0.0003030086 | 0.9999999601 | 9.413101e-06 |
| anvil4m endpoint | 0.25 | 9.250111e-06 | 5.718545e-05 / 6.480744e-05 / 8.249719e-05 | 0.999999997 | 1.883388e-05 |
| anvil4m endpoint | 0.125 | 1.725322e-05 | 5.154578e-05 / 4.643126e-05 / 5.688652e-05 | 0.9999999984 | 3.767446e-05 |
| anvil4m endpoint | 0.0625 | 3.498352e-05 | 9.858599e-05 / 8.86712e-05 / 0.0001044211 | 0.9999999946 | 7.535839e-05 |

## Empirical acceptance and limitations

All six chords have adjacent h1/.5 comparisons supporting the direction numerically: channel error at those two scales at most.347424%, parameter-direction error at most.585290%, adjacent-response difference at most.379041% of JVP norm, cosine above.99999397, output-storage spacing heuristic at most.239932%. These are descriptive numerical observations, not a newly preregistered automated universal accuracy gate. Accept the measured old-base FULL-chord smooth derivatives at this aggregate precision on the saved batch; do not infer sub-error layer or per-coordinate accuracy.

Adam4M first beta exhibits clear curvature at h1: .200123% error drops to .051282% at.5 and .014991% at.25, before rounding dominates at finer scales. Adam4M endpoint and Anvil4M endpoint show initial roughlyquadratic error reduction followed by a turn-up. Anvil4M first latent error instead rises as h shrinks, illustrating why decreasing h is not automatically better. Full endpoint nearby central derivatives are centered at OLD weights; they do not certify that a forward one-sided full-update Taylor prediction is equally accurate.

Adam8M first is much smaller: parameter chord norm.000374494 versus Adam4M .0294315. At h1 latent/alpha/beta errors .1460%/.1547%/.1432%; at h.5 they are .3187%/.3474%/.3024%; at h1/16 .02324–.02519 relative (2.324–2.519%). At h1/16 parameter-direction error3.80019% and3.98080% of originally nonzero parameter elements vanish, while storage heuristic is1.714–1.919% of channelJVP. This scale cannot be used to claim convergence or tighter derivative precision. Adam8M endpoint also worsens as h decreases, though at h1/16 errors stay below.241349% and parameter-direction error.352607%. Retain these failed-tightening facts rather than selecting a favorable minimum.

Alpha and beta are separately audited; pooled latent alignment does not validate either shape channel. Old shape1 counts are Adam4M alpha413,beta774 (55state/action positions overlap), Adam8M alpha172,beta2951, Anvil4M alpha0,beta495. Loader floor_shape_count uses OR over channels, hence1132 versus413+774=1187 is consistent. There are98304 state/action entries per channel. Floor subset local responses/relative errors are not supplied: aggregate small shape errors can coexist with nonzero smooth derivatives and zero rounded changes on floor plateaus. This is not an actual-excess-chart interior certificate or proof of hidden-layer starvation.

Storage floors use neighboring towardzero FP32 spacing summed/(2h). They exclude accumulated compiled-primal arithmetic error and binade-specific subtleties, so they are heuristics, not hard roundoff bounds. Parameter displacement error and center drift are independent telemetry; representability alone does not require each coordinate to survive or the rounded direction to equal the requested chord. Larger h can hide curvature; smaller h can amplify rounding.

The independent centered primal catches common AD omissions along THESE six complete directions. It is not a certificate for every isolated layer, arbitrary direction, future-state occupancy or Fisher-weighted near-null mode; coordinate errors/cancellations can be masked by aggregate norms. Separate layer/full-JVP sums and shapeFisher error gates are still required. The prepared layerV1 failed uniform-special-function accuracy; neither chord success nor independentgraphprimal equality repairs that numerical metric defect. LayerV2 CUDA results remain separate.

No physical PG, finite return, optimizer efficacy, benchmark-causal explanation, near-optimum or chart-selection conclusion follows. Stored latent logits were not independently recorded; the audit uses the source-pinned compiled actor and avoids inventing them by inverse shape-floor reconstruction. v13 remains unselected/deferred.
