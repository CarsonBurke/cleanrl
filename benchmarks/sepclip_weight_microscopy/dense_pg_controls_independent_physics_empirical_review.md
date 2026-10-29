# Independent empirical review: dense matched physical PG

Job11765 succeeded, attempt9462, exit0, maxParallelRuns1. This review performed JSON-only NumPy/SciPy FP64 algebra and streaming source/artifact hashes; no torch import, model, GPU, test or job execution. Detailed per-state and objective numbers are saved in dense_pg_controls_independent_physics_numbers.json.

Artifact dense_pg_controls.json SHA256: 4928741aade1d67d125f765186c397ef7a3560e3d94999c2ffd8e0a10e1531e5. Executed manifest: execution_sources/sepclip_microscopy_dense_pg_controls_1791339981091961481.json. Its hash is recorded in the numerical receipt. Artifact source_sha256 is a singular producer digest despite the key name: e7c6fd46e185e158bc7b00b11b9f86f94d3dbd885d6bc6bfea80146f57f2ddac, matching executed and current dense source. Required physics, v12/v7 trainer, MuJoCo/vector normalization/runtime/host actor/loop source hashes also match execution. All four snapshot byte hashes match disk. No snapshot tensor was deserialized for this review.

## Artifact acceptance

Inventory matches four Adam/Anvil captures at4013696/7995008 steps,32 unique selected states,16 independent first-action draws and two independent full action/continuation replicas, H1/16/64/256/512, gamma.99, seed1. Every numeric input leaf is finite. Selected normalizer replay is exact; restored-observation and saved first-transition raw reward/next observation replay errors are zero in both replicas for every capture. Historical GAE and applied-reward replay errors are zero. Truncation bootstrap replay errors are7.153e-7 for both Adam captures and4.768e-7 for both Anvil captures, below2e-5. Historical rollout truncations are16 at4M and17 at8M for each optimizer; these are replayed historical boundaries, not continuation resets. Maximum selected episode ages400/395 with H512 stay below1000.

Analytic score-PG versus stored-action autograd relative errors are4.744e-7,3.864e-7,6.911e-7,1.223e-6 in Adam4M/8M,Anvil4M/8M order. They support numerical consistency of the stored smooth score chain, not an unbiased physical score-law identity. No artifact provenance/replay blocker was found. Cross-checking saved state coordinates directly against deserialized snapshot tensors remains the consumer's matched-state gate; this review checked byte identity, uniqueness and artifact layout rather than running that model-side consumer.

## Two-replica agreement

The table reports finite-normalized fields. Euclidean cosine uses the saved logit field directly. Dual cosine and energies independently reconstruct the existing smooth Fisher: FP64 softplus/sigmoid from saved FP32 logits, 2x2 Beta trigamma blocks, symmetric inverse square roots, equal state weights. This is not actual FP32 rounded-shape Fisher certification. C=dot(w0,w1)/32. N=.5*norm(w0-w1)^2/32. Estimated noise of the mean field is N/2. With exactly two replicas, noise-corrected mean energy equals C algebraically; these are not two independent signal estimates.

| Capture | H | Euclidean cosine | Dual cosine | Signed C | N | (N/2)/C | Positive state C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adam4M |256| .0931 | .0476 | .00124441 | .0248947 |10.003|18/32|
| Adam4M |512| .0839 | .0353 | .00100277 | .0274428 |13.683|17/32|
| Adam8M |256| .0069 | -.0353 | -.000534810 | .0161742 | -15.121 |15/32|
| Adam8M |512| .0093 | -.0376 | -.000636704 | .0182090 | -14.299 |16/32|
| Anvil4M |256| .0561 | .0560 | .00107986 | .0186712 |8.645|21/32|
| Anvil4M |512| .0446 | .0651 | .00153005 | .0225244 |7.361|18/32|
| Anvil8M |256| .5967 | .2327 | .00823206 | .0283045 |1.719|18/32|
| Anvil8M |512| .5918 | .2309 | .00869971 | .0298246 |1.714|14/32|

Negative C and ratios remain signed; they are unresolved signal estimates, not negative true squared signal or proof of zero population PG. Positive C also lacks a field-energy confidence interval atR2. The Adam4M and Anvil4M mean-field noise estimates exceed signed signal by13.7x and7.36x atH512. Anvil8M's lower ratio still assigns about63% of observed mean-field energy to estimated noise, with substantial state concentration.

At Anvil8M/H512, selected state8=(t311,row10) contributes .328043 to the unaveraged dual cross sum, or117.84% of the total; the other31 states together have negative cross-energy. This occurs in Euclidean geometry too: the same state contributes105.92% of that signed total. AtH256 the respective shares are114.69%/105.04%. H512 per-state dual median cosines are .0283, .00655, .0204 and -.0221 for Adam4M/8M,Anvil4M/8M. Stronger Anvil8M aggregate cosine therefore does not establish broad matched-state alignment. Removing a state after noticing this would alter the target population and consume the diagnostic; retain it and disclose concentration.

Smooth Fisher smallest eigenvalues/max block condition numbers are6.888e-8/4.472e4,6.349e-11/2.763e7,5.586e-9/5.381e5 and1.712e-11/9.505e7 in the same capture order. All CPU-computed blocks are SPD. Conditioning and precision-qualified metric differences remain separate from statistical signal; SPD alone does not certify numerical accuracy or actor reachability.

Raw finite and normalized-plus-tail fields retain the same broad reliability pattern. Their full Euclidean/dual/per-state statistics are in the numerical receipt. No ratio is being interpreted as optimizer efficiency or an optimum bound.

## Tail and sampling sensitivity

AtH256, the paired bootstrap-minus-finite field changes individual replica Euclidean norms by5.09%–20.41%; in smooth-dual norm,6.03%–17.85%. AtH512 those ranges fall to .456%–1.820% and .495%–1.587%. H512 paired-tail observed-Q means are .01853/.01865, .02115/.02046, .01828/.01832 and .02085/.02081 normalized-return units. The observed bootstrap Q minus finite Q agrees with the recorded critic-tail identity within FP64 roundoff. H512 tail weight is .005823976768663671. Small field-norm tail increments do not bound a near-cancellation fixed-direction sign or certify the old critic tail as true value.

All sampled first-action clamp counts and observed-action clamp counts are zero in both replicas for all captures; all normalized reward-clip counts are zero atH256/H512. There are3072 independent first-action dimensions per replica. Theoretical first-state clipping mass means/maxima are:

| Capture | Mean mass/dimension | Maximum mass/dimension | Expected independent first clamp dimensions/replica |
| --- | ---: | ---: | ---: |
| Adam4M |4.005e-6|1.967e-5|.01230|
| Adam8M |5.457e-6|2.526e-5|.01677|
| Anvil4M |3.600e-6|2.238e-5|.01106|
| Anvil8M |6.397e-6|2.854e-5|.01965|

Zero observed clamp counts are unsurprising at those expected counts. They do not eliminate omitted atom/quantile/FP32 quantization score corrections. Uniforms remain truncated to[1e-7,1-1e-7] and scores remain continuous Beta scores on rounded/clipped native values. Theoretical probability mass is not a derivative-bias bound. Dense future-action clamp telemetry is not supplied by this artifact, and no exact native-host sampler equivalence was certified.

## Disposition

Accept this artifact as a complete, provenance-matched conditional physical measurement with successful replay gates. Historical poor full-field replicate agreement is not resolved generally; Anvil8M has stronger aggregate agreement concentrated in one selected state, while Adam8M remains signed-negative in dual cross-energy. These empirical results justify withholding fitted physical efficiency, capacity/near-optimum and chart-selection conclusions.

Fixed actual-direction reliability is a different question and remains pending11887: evaluate signed replica pairings and paired raw/normalized/tail comparisons with their R2 uncertainty, objective/path identity and multiplicity limits. A resolved scalar can coexist with unresolved full-field energy. The present review does not supply those scalar pairings, actual first/ten-epoch closed-loop gains, a physical PG bias certificate, or a causal explanation for benchmark scores. User-authorized exploratory prescriptions remain separately adjudicated by their prospective score and replay protocol.
