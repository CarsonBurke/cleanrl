# Independent empirical chart and historical rank review

This is JSON/scalar and source inspection only. No models, tests, GPU work or queue mutation were performed by this reviewer. Jobs 11775 and 11781 succeeded with declared maxParallelRuns=1. These measurements establish numerical limitations; they do not establish reliable physical PG, causal score explanations, near-optimality, or chart preference. V13 remains unselected/deferred.

## Provenance and numerical gates

- chart_numerics.json: SHA256 `e029d7598d93caf2e5534b0c108ff426dac601de0712c24f3f5638e17079d4b0`.
- oracle_geometry_historical.json: SHA256 `0fa869b8c4027d1287fb093b2a08ceae37a2d2136c7134cbfc327ed3c36787e4`.
- physics_complete.json: SHA256 `4e8cff45b006f0860a637af6aa56585a273178c08f36e0c2fb2abba7f779f44f`.
- Execution manifest sepclip_microscopy_chart_numerics_1791340126128131125.json: SHA256 `a529bafb13dbdd7d817acede152ef643076c57473b1fd57cdf30358593711a9c`; exit 0; all recorded source/environment hashes match current bytes.
- Execution manifest sepclip_microscopy_oracle_geometry_historical_rank_1791340140374419326.json: SHA256 `9a1d3c1cb77161fea826920e05ddc2dd006351dc0477f0035bcd016f6aaa157d`; exit 0; all recorded source/environment hashes match current bytes.
- Execution manifest sepclip_microscopy_chart_numerics_contracts_1791334852719693525.json: SHA256 `f2fc6f5cf3b4e0e92afd1079abb5793a0f23c82b9583ef35f2c9a558f192d3cd`; exit 0; all recorded source/environment hashes match current bytes.
- Execution manifest sepclip_microscopy_oracle_rank_contracts_1791334863668302402.json: SHA256 `4b637fcbaa0f7254938e09a1bf01a4d64beb37be6817c4f4e4b3efccc58a7a3a`; exit 0; all recorded source/environment hashes match current bytes.

Artifact script hashes match execution: chart `b4d6e5f5573dbfd1b8749d2e88f774ad3c0b31170714addc173632d98cba60a6`; rank `03bca505a2e00e1b7ca2d93b8982beb4343dc8b24bcd7287100b593611fbc678`. Candidate decoder source `35e9d4d704c4c5bb2a9a6706b84507d6d6f0e867fdb10bf43c5323f4d0cdd68e` matches frozen v13. Queue logs report gate 11774: 4 passed; gate 11776: 22 passed across geometry and actual-step suites. These are distinct from failed 11770 (26 passed, 2 failed); the NEW chart V2 repair has no execution claim here. Earlier failed geometry gate is also preserved.

All six snapshot byte hashes independently match every recorded row; recorded oracle hash matches current physics_complete.json:
- /home/marvin/Documents/repositories/cleanrl/runs/HalfCheetah-v4__sepclip_mechanisms_rollout_clock__1__1791241877/diagnostic_snapshot_4013696.pt: `253afd0111b500475a4777a941e36483ef4c648fa92de0e5016d453a4c78927f`.
- /home/marvin/Documents/repositories/cleanrl/runs/HalfCheetah-v4__sepclip_mechanisms_rollout_clock__1__1791241877/diagnostic_snapshot_7995008.pt: `254481ce20051e0a7d8e84717b8fed41c80ab14f8592f1202a30692c45f2cd1d`.
- /home/marvin/Documents/repositories/cleanrl/runs/HalfCheetah-v4__sepclip_microscopy_v12_adam_control__1__1791260590/diagnostic_snapshot_4013696.pt: `8b17da17c1a4d6ed31e62076f76debfc01763e6f0a8b95f401674e55415d6197`.
- /home/marvin/Documents/repositories/cleanrl/runs/HalfCheetah-v4__sepclip_microscopy_v12_adam_control__1__1791260590/diagnostic_snapshot_7995008.pt: `5d7979136fb15329c9bd6d0e6340cde6206d8b0956c21f273f223bbb851b3f30`.
- /home/marvin/Documents/repositories/cleanrl/runs/HalfCheetah-v4__sepclip_anvil2_v4_anvil_half__1__1791178146/diagnostic_snapshot_7995008.pt: `fcbc726f7c121d6f5ff72c613a3d0321a6d2fab996ca31c1f3cd805bb38c7a12`.
- /home/marvin/Documents/repositories/cleanrl/runs/HalfCheetah-v4__sepclip_mechanisms_rollout_clock_full__1__1791242016/diagnostic_snapshot_7995008.pt: `3a6a696f7142349591c26a49f784366924da227dcb5ee494a6201333d19980fc`.

## Actual compiled chart stress measurements

35 shape-grid cases × 7 perturbations = 245 KL comparisons, with 108 negative FP32 KL values; all FP64 same-rounded-shape KL values are positive (minimum 1.03243e-12). Finite FP32 densities/scores throughout this grid are therefore insufficient to certify numerical accuracy. Each concentration has five requested modes and 35 KL comparisons.

| Requested total concentration | Max log-density abs error | Max entropy abs error | Max joint chart score relative norm error | Negative FP32 KL / 35 | Largest KL abs error |
|---|---:|---:|---:|---:|---:|
| 3 | 2.7103245e-07 | 3.3177115e-07 | 3.8212674e-07 | 12 | 1.4731504e-07 |
| 10 | 2.0665636e-06 | 9.1074505e-07 | 9.4482775e-07 | 8 | 3.7393431e-06 |
| 30 | 7.7230928e-06 | 1.520319e-05 | 4.5407575e-06 | 19 | 1.8315271e-05 |
| 100 | 5.5577287e-05 | 3.998293e-05 | 2.6069271e-06 | 9 | 8.1587032e-05 |
| 1000 | 0.00057014855 | 0.00028818257 | 6.7233384e-05 | 15 | 0.00082120835 |
| 10000 | 0.01721447 | 0.0078992362 | 0.00012434599 | 28 | 0.02484471 |
| 1000000 | 1.3180359 | 1.8180352 | 0.0027337481 | 17 | 1.9311958 |

At concentration 1e6, mode .99, precision displacement .01 gives FP32 KL −1.21135759 versus FP64 same-shape KL 2.50875484e-5. The error is catastrophic even though all participating values are finite. FP64 also has finite cancellation error; these comparisons are not arbitrary-precision certifications.

Pooling across the five modes and five action probes, channel-separated relative L2 errors reveal masking by the larger mode score:
- concentration 30: mode 1.08651423e-07; precision 3.45149412e-06.
- concentration 10000: mode 7.77323143e-06; precision 0.0034487095.
- concentration 1e+06: mode 4.65524978e-05; precision 0.340860552.

Reference-law caveat: audit_sepclip_chart_numerics.py:65–73 uses FP32 exp(log-excess) and sigmoid chart factors, then casts them to FP64. It does NOT subtract 1 from the final rounded alpha/beta to obtain actual excess, despite its docstring. The density reference itself uses identical actual rounded FP32 shapes. Thus the reported chart-score discrepancy includes smooth decode/ideal-chain differences as well as FP32 density-gradient arithmetic; it is not an isolated exact actual-excess covector measurement. The separately prepared chart-credit audit addresses that convention explicitly.

Actual v13 compiled decoder stress has 87 rows: 72 finite forwards/backwards and 15 nonfinite rows (ell 90/100), serialized as null without clamps. Thirty rows have alpha=beta=1; 42 have at least one shape=1. At ell −20 all three ordinary mode probes already round to [1,1], while smooth AD reports ell gradient ≈2.06115e-9. At ell −120/−105 shape gradients are zero. At ell 0, extreme mode logits can still put one shape exactly at 1; therefore a finite decoder cannot certify an interior actual-excess chart. The sampled ell88 probes are finite and ell90 fail; this sampled boundary is not an exhaustive safe-domain certificate.

Native sampler uses 98,304 draws per grid row. Host/CUDA shapes match exactly and physical rescale max error is zero. Mean deviations span −2.50609 to +1.88807 theoretical SE; variance relative errors −.0126649 to +.0140245 and asymptotic SE deviations −2.00741 to +2.18788. Clipped count per row is 0–1; theoretical clip probability at most 3.41099e-6. This supports these sampled marginal moment/shape checks only; it does not prove exact atom-score unbiasedness, finite-RNG law or H512 physical credit reliability.

## Historical tangent interpolation and finite retraction

Inventory: 6 snapshots × H64/H256 × 3 objective fields × 2 replicate indices = 72 rows, all 16 saved states and 192 functional coordinates. Every row has state_distribution_matched=false; actor reproduction error 7.62939e-6–1.23978e-5. Input is historical physics_complete.json, NOT the pending dense matched physical H512 oracle. Objectives are raw finite, normalized finite and normalized bootstrap PG.

| Snapshot | Rank at cutoff 1e-6 / 1e-5 / 1e-4 | Condition number | Fitted parameter norm range | Finite exact KL range | Retraction relative linearization error |
|---|---:|---:|---:|---:|---:|
| sepclip_mechanisms_rollout_clock 4M | 192 / 192 / 190 | 21914.164 | 2.514684–4.450085 | 2.040339–7.583101 | 26.1599–41.8252 |
| sepclip_mechanisms_rollout_clock 8M | 192 / 190 / 182 | 636250.99 | 34.50573–133.3833 | 49.86773–236.8126 | 196.3808–1352.929 |
| sepclip_microscopy_v12_adam_control 4M | 192 / 192 / 190 | 45032.448 | 2.030447–4.846139 | 0.3578898–3.544642 | 13.35907–36.92338 |
| sepclip_microscopy_v12_adam_control 8M | 190 / 187 / 183 | 4617072.2 | 50.54482–701.2583 | 24.49589–769.7956 | 74.31295–1993.578 |
| sepclip_anvil2_v4_anvil_half 8M | 192 / 188 / 177 | 653130.66 | 53.78025–183.9445 | 39.54727–109.6072 | 120.159–274.124 |
| sepclip_mechanisms_rollout_clock_full 8M | 191 / 190 / 180 | 1120517.4 | 29.4288–207.8533 | 27.39452–104.4594 | 87.68095–794.4511 |

Configured relative cutoff is 2.94475555e-12 (FP64 arithmetic heuristic); all fits retain rank192 and nearly unit projected energy/cosine. All remain rank192 at 1e-7. This does not recover FP32 derivative accuracy. Jacobians originate in the FP32 actor, then undergo FP64 tangent projection/SVD. Effective spectral ranks are only 14.9287–25.4910. At cutoff1e-4 the historical noisy-field retained dual-energy fractions span .660651–.999095 across snapshots/objectives/replicates. These are algebraic projections of unresolved fields, not validated retained-range efficiency. No actual-motion retained-range fields are present in this artifact.

Every configured fit targets Fisher quadratic .001 and achieves it arithmetically. Every measured finite exact KL exceeds .001: .3578898–769.7956, or 357.9–769,796 times the target. Finite retracted Fisher quadratic is .177423–3974.5824; linearization errors are 13.3591–1993.5781. These fitted proposals cannot be called local realizable budget steps. They are neither actual optimizer first/ten-epoch chords nor physical causal interventions. Smaller retained-cutoff fits have no finite retraction evaluation in this output.

Layer Gram cross terms mostly cancel large isolated movements: diagonal quadratic .795819–79469.9399 against joint≈.001, with corresponding large negative cross quadratic. This directly invalidates diagonal-only layer cost allocation for these interpolation directions. It is not the sign pattern or modal allocation of actual Adam/Anvil updates.

Precision distinction: local Fisher uses FP64 softplus/Fisher recalculated from exact FP32 logits (a nearby smooth policy); finite KL uses actual FP32 softplus+1 rounded shapes and only upgrades KL algebra to FP64. Tangent matrices are projected against normalized FP64 weights. Native chart coordinates here are mean alpha/(alpha+beta) and total concentration alpha+beta, not candidate mode/excess r/ell; coordinate Fisher cross-correlation reaches approximately −.602527 to +.596979. Cross terms are required.

## Acceptance and withheld conclusions

Accept these artifacts as provenance-reconciled numerical stress and historical interpolation measurements, with their explicit precision and sampling domains. They show that scalar finiteness, FP64 full arithmetic rank and an achieved infinitesimal target are inadequate evidence of usable finite policy motion. They do not establish insufficient architecture capacity: the saved-state Jacobian spans all 192 coordinates at the permissive cutoff. They do not establish global expressivity, generalization, a trustworthy numerical rank below the FP32 noise floor, optimizer efficiency, near-optimality, chart preference or a cause of sub10k versus20k performance.

Before any such interpretation: require reliable matched dense PG replicates/objective scope; actual ambient first/end chord JVP/finite-difference acceptance; signed fixed-direction replica pairings separate from fitted projections; finite-step KL/Fisher linearity at the intended budget and retained cutoff; explicit actual-rounded-shape interior/Fisher SPD and cross-metric contracts; and separate causal physical closed-loop evidence. Do not select v13 on this stress result.
