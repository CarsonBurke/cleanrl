# Dense physical PG reliability: conditional estimator and decision gates

Read-only committee review. No models, tests, jobs, or source changes. Pending job 11765 remains the four-v12-snapshot, 32-state, 16-first-action-draw, two-independent-replica experiment with H=1/16/64/256/512. Historical H64/H256 field cosines -0.029/-0.041 remain unresolved; user-authorized prescription trials are exploratory rather than justified by these fields.

Reviewed source SHA256:

- scripts/audit_sepclip_dense_pg.py: e7c6fd46e185e158bc7b00b11b9f86f94d3dbd885d6bc6bfea80146f57f2ddac
- scripts/audit_sepclip_actual_oracle_step.py: 0d23d4b61059a255b7c566fd8088ebea629f357905a3fd7b35928cbb5b848953
- scripts/audit_sepclip_oracle_geometry.py: 03bca505a2e00e1b7ca2d93b8982beb4343dc8b24bcd7287100b593611fbc678
- scripts/audit_sepclip_physics.py: 1d53db1d68f6fb188fb7207513b606a652f70682029e278031cd3adb25605114

The separate prescription closed-loop preparation has received geometry's final independent static approval at driver 9bb20ffde6c2e3aa3c73572f38edd960658e49fe9f3278f6875f79b0a4d8e493 and test b2bbd1bdacf3d4a6bf858a3a0f2923c5393ab157b377562d23268fbb896d8052. This is not numerical execution or validation of physical PG.

## Exact target and baseline

Dense lines 266–284 draw 16 independent first-action quantiles per selected state, append the one saved observed action as a separate branch, and use the continuous Beta logit score evaluated on the rounded/clipped native first action. Later steps 295–329 share fresh per-state inverse-CDF uniforms across all 17 branches. Across replica indices both first actions and continuation uniforms change; the two fields combine both noise sources. Neither 16 branches nor 32 states are independent training seeds. The saved observed-action branch is excluded from the 16-action LOO baseline and field estimate.

For one fixed state, condition on all future shared exogenous uniforms U. Let s_i be the first-action score and Q_i=Q(A_i,U). Lines 115–120 compute:

g = mean_i s_i * (Q_i - mean_{j!=i} Q_j)
  = sum_i (s_i - mean(s)) * (Q_i - mean(Q)) / (M-1).

This is the sample covariance, not an extra critic-subtracted estimator. With independent first actions, the LOO baseline is independent of its own action conditional on U; shared continuation noise is therefore a valid variance-control construction under an exact zero-mean score law. Nonlinear branch-specific future policies do not invalidate that independence: given U, each trajectory is a deterministic function of its own first action.

More generally E[g|U] = E[s Q|U] - E[s] E[Q|U]. Exact continuous Beta scores have E[s]=0. Here the actual draw/score laws differ, so centering is not automatically innocuous or an exact PG repair. A constant objective yields zero sample covariance by construction; that does not prove the score law or gradient is correct.

The field estimates a conditional FIRST-action covector with the old actor fixed thereafter, at 32 selected saved training states and per-state frozen normalizers. Raw finite, normalized finite, and normalized plus old estimated critic tail are separate objectives. Dense normalized rewards retain FP64 after clipping; this differs from the trainer/closed-loop per-reward FP32 law. The reported per-layer observed-action Q-minus-old-V comparison (213–235) supplies one observed first action per state; it does not become conditional mean PG by averaging continuations. None of these quantities is the all-time policy derivative or the learner's full stored-occupancy clipped surrogate.

## Draw/score law gaps that more samples cannot repair

First uniforms are uniform on [1e-7,1-1e-7], not [0,1]. The inverse-CDF result is clipped to [1e-6,1-1e-6] and rounded to FP32. Score evaluation uses continuous Beta density/digamma at this physical native value, with a smooth sigmoid(logit) chain on actual recorded FP32 shapes. Thus:

1. Quantile truncation creates parameter-dependent native-action support boundaries; its derivative is not certified by merely centering the ordinary Beta score.
2. Clipping creates atoms. Correct score terms for atom probabilities differ from continuous density scores at the clamp. FP32 rounding creates additional quantization bins. The oracle is a diagnostic smooth-score pairing, not a certified derivative of the exact finite-precision action law.
3. The excluded quantile probability 2e-7 and theoretical clipping tail masses are probability diagnostics, not gradient-bias bounds. A score-weighted error can be large relative to a near-zero directional signal even when event probability is small. Finite tail or raw reward magnitude and parameter derivatives matter.
4. Recorded analytic score-versus-autograd PG replay tests the smooth stored-action chain; it does not validate action-distribution score identity, clipping atoms, continuation law, or physical derivative.

For scale only, six joints over H512 give a union bound 6*512*2e-7=0.0006144 for excluded-quantile events under a common underlying untruncated-uniform coupling. The normalized zero-tail return has absolute bound 10*sum_t .99^t <1000. This would bound return-expectation differences for that isolated quantile-law coupling by at most 2*1000*0.0006144=1.2288, before rounding/clipping differences. It provides NO score-gradient or sign bound and does not bound raw return. A small scalar PG must not borrow this bound as a reliability certificate.

## Interpret 11765 before selecting more work

First require the finite-source, normalization/GAE replay, first-transition and analytic-score replay gates already recorded. Inspect clipping atoms/masses and per-state return ranges. Preserve every objective and horizon separately; H512 zero-tail convergence is an attenuation diagnostic, not true terminal critic accuracy.

For each actual first/end motion fixed independently of the new oracle noise, use actual_step fixed_motion_objective_pairings (43–105): per-replica state-average scalar pairing, Student-t interval, and paired estimated-tail increment. First declare whether the motion is its ambient chord JVP, finite logit difference, tangent intervention, or a model-fitted direction. A finite logit difference pairing is not a finite-return gain. Read raw/normalized/tail comparisons on identical states, motion and replica order. Existing JSON has no replica IDs, so paired tail alignment assumes producer ordering and does not detect arbitrary reordering.

With R=2, the 95% conditional t critical value is 12.706. For scalar replica values x1,x2, SE=abs(x1-x2)/2 and interval half-width=6.353*abs(x1-x2). A same-sign pair alone is weak evidence. A predeclared useful screening gate is: approximate interval excludes zero AND half-width <=0.5*abs(mean), with the approximation and fixed-state scope explicit. At R=2 the latter requires abs(mean) >=12.706*abs(x1-x2), a deliberately demanding rule. A passing interval is not protection against draw-law bias, multiple comparisons, or optional stopping. Do not select the best of many horizons/layers using unadjusted intervals.

Inspect Fisher-whitened signed cross-energy and half squared replica difference in actual_step 200–237. The cross-energy estimates the squared mean of THIS estimator under identical independent replica laws; with the draw/score gap it is not automatically true physical-PG energy. Negative cross-energy and negative noise-corrected energy must remain signed and mean unresolved signal. At R=2 there is only one pair. For R>2, pair entries share replicas; never use the number of pairs as an independent sample count.

Fixed scalar pairings may be resolved while the 384-coordinate selected-state field remains noisy. That supports only those fixed local scalar pairings; it does not justify model-reachable efficiency or chart preference. Conversely, a stable nonzero full-field cosine is not required to interpret a demonstrably stable fixed scalar. It also does not certify simulator feedback effects.

Rank is a separate numerical question. oracle_geometry 107–139 compares declared SVD relative cutoffs, including 1e-7 through 1e-4; arithmetic rank is not FP32 derivative accuracy. Require retained-range matching for field and actual motion, report discarded motion and finite reconstruction errors. High reachable fraction computed from a noisy field is not signal recovery. Cross-replica fitted-direction evaluation (actual_step 276–309) reduces fit-noise self-assessment but still uses the same physical states; both ordered fit/test pairs are not independent validation samples. Its denominator remains a noisy held-field norm, so efficiency interpretation stays withheld while signal energy is unresolved.

## Evidence-gated follow-up, only if needed after results

- If fixed scalar intervals are adequate but full-field energy is unresolved: retain the scalar conclusions, withhold fitted efficiencies/chart selection, and prioritize already-queued actual closed-loop first/end finite outcomes. Do not increase compute simply to make a cosine attractive.
- If scalar intervals remain broad: predeclare a small set of actual fixed directions and matched objectives, and obtain more independent FULL replicas at those states before fitting directions. R=4 or R=8 reduces the extreme t penalty; these are proposed designs, not authorized/submitted jobs or guaranteed sufficient budgets. Estimate required R using observed scalar variance and a meaningful target half-width, with uncertainty from the current two-replica variance explicit. More replicas estimate uncertainty; a larger M can reduce conditional action variance but is not itself a substitute for independent replicas.
- Existing mixed replicas cannot identify whether first-action or continuation noise dominates. If that choice changes the next allocation, use a limited crossed/nested design: fixed first-action panels with independent continuation streams, and independent first-action panels sharing a fixed continuation stream, followed by independent complete replicates for the final interval. Shared panels require clustered analysis; M branches do not supply M independent continuation replicates. Choose M versus continuation replication from this variance evidence, rather than automatically moving 16 to64 or merely lengthening H.
- For variance control, an independently trained/frozen per-state or per-state-U baseline is valid under the matched zero-mean score law; current LOO already provides that kind of conditional centering. A critic conditioned on the evaluated action is not an action-independent baseline. Reusing oracle targets to fit and evaluate a baseline needs cross-fitting. Antithetic first actions make other-action LOO baselines dependent on the evaluated action and can bias the ordinary construction; do not insert antithetics or unrandomized quasi-Monte-Carlo and keep the current IID formulas.
- If atoms or truncated-law sensitivity can matter at the observed scalar signal scale, more samples only sharpen a biased estimate. A separately specified law-consistent estimator or explicitly bounded smooth-law approximation would be required. Its numerical score identities and objective-matched physical checks would themselves need independent validation; it would not become an automatic truth oracle. Keep failed attempts and previous results.

No efficiency, chart candidate, critic-bias explanation, or causal training result follows solely from the pending dense run. Exploratory prescription score outcomes may proceed under the user's explicit choice while this oracle remains unresolved.
