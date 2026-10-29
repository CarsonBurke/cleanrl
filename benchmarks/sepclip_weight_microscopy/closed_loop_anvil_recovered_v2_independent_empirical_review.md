# Independent empirical review: actual Anvil update feedback

Job11892 succeeded, attempt9480, exit0, maxParallelRuns1. Prerequisite11891 passed all23 V2 contracts, including independent real nonlinear first/endpoint chords at both Anvil checkpoints. The original failed11795 and skipped11812/11825 are not replaced as evidence. Artifact closed_loop_anvil_recovered_v2.json SHA256 is0f8b45d7875bff727a61399be63404a455a63171f105440c045415979c5661c4. Independent JSON-only CPU calculations and exact hashes are saved in closed_loop_anvil_recovered_v2_independent_numbers.json. No model/GPU/test/job execution occurred in this review.

## Numerical acceptance

Artifact producer2b6dd11213f0aa28a1d6472124c8863e8ca96ba190229e6a1eace4ea1c815e99 and helper dense/physics/v12 hashes match executed/current sources. Both Anvil4M/8M snapshot and update-trace byte hashes match disk. All numeric artifact leaves are finite; every selected state dictionary matches the earlier dense measurement exactly. There are32 states, seven arms, four independent sampled continuation replicas, one deterministic mean/mode fork, H64/256/512, gamma.99 and lambda.95. Full integration restore error is0. Gait's separate first-action candidate population and this actor-bank intervention are not the same experiment.

Each trace has10 actual updates across10 epochs. Exact endpoint model is retained; first policy is old FP32 weights plus the first saved retracted delta, not independently saved first weights. Cumulative endpoint reconstruction errors are2.328e-10 at4M and0 at8M. Unit-axis norm errors are at most1.788e-7. No tensor-shape inference or new architecture is being evaluated: these remain the exact original v12 Anvil actor/update sources.

All paired gains and fourway interactions recompute exactly from saved per-arm replica/state values. The matched endpoint-minus-first comparison was independently paired within replica. Recomputed old-trace normalized rewards match clip-then-FP32 law exactly. Independent all-time finite score-return PG recomputation differs by at most5.69e-14 atH512; paired GAE-minus-augmented-MC recomputation differs by at most3.56e-14. Lambda1 telescoping errors are at most1.66e-12 raw and6.00e-15 normalized atH512. These are arithmetic checks, not true critic accuracy.

Directional-primal versus actual old forward maximum differences are1.907e-5/3.004e-5 at4M/8M, consistent with distinct compiled batched execution paths; this is recorded FP32 precision telemetry rather than bitwise equality. Independent nonlinear chord contracts passed at the executed source. A vector JVP pass does not certify a cancellation-dominated score-weighted physical sign.

## Paired finite effects

Sampled H512 discounted RAW gains relative to old_all, averaged over the32 fixed states:

| Capture | First full policy | Ten-epoch endpoint full policy | Endpoint minus first |
| --- | --- | --- | --- |
| Anvil4M | -1.125 [-17.182,14.932] |2.238 [-5.360,9.836]|3.363 [-7.808,14.535]|
| Anvil8M | -1.424 [-9.744,6.896] |2.784 [-3.894,9.462]|4.208 [-2.531,10.947]|

Intervals are pointwise approximate Student-t3 conditional continuation intervals, with shared quantiles across arms and fixed per-state normalizers. All sampled full-policy and endpoint-versus-first sign comparisons remain unresolved. Corresponding H64/H256 full-policy intervals also include zero. These are local finite update effects, not whole-episode benchmark returns or confidence intervals across training seeds/states.

The Anvil8M/H512 endpoint hybrid decomposition is informative:

| Intervention | Raw gain, pointwise95% |
| --- | --- |
| Endpoint at t0 only, old actor later | -8.631 [-10.327,-6.935] |
| Old actor at t0, endpoint later |4.574 [-1.054,10.201]|
| Full endpoint feedback interaction |6.842 [2.264,11.419]|
| Full endpoint |2.784 [-3.894,9.462]|

Each interaction is formed within the same replica as full-first-later+old; all covariance from common noise is preserved. Its four replica means are5.848,9.799,8.424,3.294. A negative t0-only effect can coexist with an uncertain positive full-policy gain through later changes and interaction. It would be incorrect to label the weights globally harmful from the t0 sign, or globally helpful from the full mean. H256 endpoint t0-only is also pointwise negative(-7.368 [-10.098,-4.638]), but its nested horizon/noise is not an independent replication. The unregistered family of modes/horizons/arms/objectives means these pointwise results remain exploratory, not familywise causal sign claims.

Normalized H512 endpoint t0 gain is-.03437 [-.04021,-.02854], interaction+.02734 [.00798,.04670], full endpoint+.01079 [-.01462,.03619]. Separately adding the fixed old-critic terminal estimate produces raw t0 gain-8.677, interaction+6.844 and full+2.766, retaining the qualitative pattern. This tail agreement does not certify infinite-horizon credit. Anvil4M hybrid gains/interactions all have pointwise intervals crossing zero.

Deterministic H512 controls show substantial mode dependence on these SAME selected initial states:

| Capture | Mean first / endpoint full gain | Intended-logit-mode first / endpoint full gain |
| --- | --- | --- |
| Anvil4M |-.964 /-2.842|.936 /-2.244|
| Anvil8M |-.193 /1.158|-4.151 /-21.833|

These are exact simulator outcomes under the stated deterministic control convention, with no continuation-noise CI; zero MCSE is not zero state/seed uncertainty. The old v12 mode helper uses intended softplus-excess logits, unlike the strong reference's actual rounded-shape mode. Do not silently compare them as identical finite-precision policies. This deterministic mode effect is separate from sampled training returns.

## Full old-trajectory score credit and GAE

H512 all-time raw directional PG for first/endpoint chords is-.570 [-6.236,5.097]/-.565 [-8.155,7.025] at4M and+.0281 [-.0368,.0931]/+.1485 [-.0891,.3861] at8M. All are unresolved. These sum gamma^t score times reward-to-go at EVERY visited time; they are not dense t0 covectors multiplied byH. The fixed old-critic baseline and separately estimated terminal-tail objectives must retain their own units and pairing.

At8M/H512, raw GAE(lambda.95)-minus-tail-augmented-MC directional differences are-.0200 [-.0625,.0224] for the first chord and-.1128 [-.2310,.00543] for endpoint. The pointwise intervals still include zero. This compares two same-trajectory finite augmented estimators with fixed old critic; neither the small lambda1 arithmetic residual nor the lambda.95 difference measures independent true value error or explains benchmark credit failure. Baseline-subtracted/unsubtracted PG remain distinct variance diagnostics under the smooth score law, whose exact zero-score-mean identity is not certified for clipped/quantized sampling.

Old sampled trajectories at4M have no realized native clamp atoms in four replicas, expected counts about.346–.349 per replica. At8M observed counts are0,1,0,0, expected.578–.584, maximum per-dimension clipping probability3.158e-5. The one realized atom makes the omitted atom-score caveat empirical, although event probability alone cannot bound directional bias. Uniform truncation and FP32 quantization remain. Old normalized reward clipping counts are zero; raw returns and per-reward FP32 normalized returns have different units/scales. No exact native host-RNG equivalence or physical score-law bias bound is newly supplied.

Disposition: accept the matched actual-bank finite feedback measurement and successful derivative/arithmetic contracts. Preserve unresolved sampled full-policy effects, exploratory negative t0/positive-interaction evidence, deterministic mode-specific outcomes, and unresolved all-time score-credit signs separately. These results demonstrate why first-action credit and full closed-loop update effects require different interpretation. They do not establish optimizer efficiency, infinite-return optimum, global actor failure, a critic-bias mechanism, or the cause of future benchmark scores.
