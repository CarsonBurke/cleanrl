# The abstract optimum and our gaps (HalfCheetah-v4, 8M, seed 1)

Synthesis of four independent reviews (theory, repo history across 28 families, literature, diagnostics
of the v47 crawl). The measured facts come from the repo; the reasoning is first-principles and the
literature is used only as a check.

## What an ideal agent does

1. **Uses every transition to its full information value.** Q for nearby policies shares almost all of
   its structure, so data from past policies stays informative. REPPO discards each rollout after 16
   passes. TD7 (off-policy replay, full Q, no clip) measures 17.1k@1M in this repo, against 6.4k for
   the best on-policy run. This is the largest measured gap: about 4x at 1M.
2. **Holds a belief-shaped policy.** The action distribution should be the agent's belief about where
   good actions are: wide where the argmax of Q is uncertain and narrow only where it is known. It is a
   real distribution (possibly multimodal) that concentrates as evidence accumulates, not a point
   sharpened by a fixed rule. A fixed entropy target (Gaussian REPPO) caps precision late (det. 12.2k
   plateau). Curvature-driven sharpening (Beta without entropy) collapses early, because the critic's
   curvature at its own argmax is negative by construction, and the bias grows like 1/sigma since the
   critic only sees data within sigma of the mode. PPO-Beta anneals without bias but slowly (14.4k at
   50M) and also concentrates to a point; it is not the target either.
3. **Moves by the reach of its knowledge, not by its own noise.** A per-state KL trust region makes the
   mode step about 0.1 sigma. Once sigma shrinks, the step shrinks with it even when dQ/da is steep
   (v47: action std 0.19 -> 0.058, dqda norm 4x the reference, return slope 174/M after 4M). The
   right step bound is the region where the critic's value estimate is trustworthy.
4. **Represents action contrasts at their own scale.** The action-dependent part of Q is ~1/40 of the
   HL-Gauss label width, and held-out ranking correlation is 0.01-0.14. TF32 alone halved learning. The
   actor climbs a signal near the critic's noise floor.

## Gaps, ranked by measured or expected impact on 8M returns

1. Data reuse (gap 1): 4x at 1M.
2. The coupled triad of spread, step size and critic information (gaps 2 and 3). Every REPPO variant
   either pays a fixed-noise tax or crawls.
3. Action-contrast signal-to-noise (gap 4).

## Directions and minimal ablations (v36 FP32 base; one change each)

- **A. Step geometry decoupled from sigma (gap 3).** On v47-independent, replace the mode's share of
  the per-state KL with a fixed action-space bound on mode movement, and keep a KL bound only on
  concentration. Prediction: the crawl disappears despite the collapse. Falsifier: it still crawls,
  which means information decay (the critic side) dominates, not step size.
- **B. Thompson population: the population is the distribution (gaps 2 and 3).** K heads, each
  climbing its own bootstrapped critic (independent data masks and targets). The behavior policy is
  the mixture, whose spread is the critics' disagreement about the argmax: epistemic uncertainty that
  shrinks exactly as the critics agree, with no entropy term. The mode trust region is measured in
  units of the mixture's spread (the belief), not each head's own noise. This builds on v45/v46,
  where a shared-critic population prevented the flip but converged because diversity came only from
  initialization. Falsifier: head disagreement collapses as fast as the v47 sigma did, or no gain over
  the shared-critic population.
- **C. Off-policy full-Q critic on replay (gap 1).** Keep the pathwise full-Q actor and train the critic
  with bootstrap targets from the current policy on replayed transitions (valid off-policy), instead
  of on-policy lambda-returns on stale data, which is why REPPO replay failed in v32-v34. This is the
  direction toward TD7's measured sample efficiency.
- **D. Contrast-scale critic (gap 4).** A centered advantage head with its own loss scale. v26
  (dueling) exists; check its record before re-running.

Order: A first (one flag, decisive about the mechanism), then B (the principled fix for gaps 2 and 3),
then C (largest ceiling, largest change).

## Red-team corrections (adopted)

- **Gap 1 is confounded with update rate.** At 1M, TD7 has taken ~16x more sample-gradients and ~8000x
  more policy iterations than REPPO (61 iterations). First test: iteration rate alone (rollout 1024 ->
  128/256 steps per env, same passes per sample): jobs reppo_iterrate_v36_ns128 / ns256. If this closes
  much of the 1M gap, replay (C) is unnecessary.
- **"Belief-shaped" was missing a premise.** TD7 reaches 17k@1M with a point policy plus fixed 0.1 noise.
  The critic can only identify dQ/da where behavior spreads actions, so a spread that shrinks with
  evidence starves the critic. Separate two things: a constant local perturbation floor (serves the
  critic's slope; not a schedule) and epistemic spread over the argmax (B).
- **Step geometry.** The KL gate binds in every run (kl at 0.030, violation 0.33-0.36). A sigma-free step
  extrapolates beyond the data. Test A (mode moves at most delta in m, |d log nu| bounded) against a
  fixed nu floor (std >= 0.1): if only the floor helps, information decay is the cause, not step size.
- **B done right.** Randomized prior functions plus per-head targets for real critic diversity. Every
  head updates on all envs' states (the pathwise loss needs no on-policy states; v46's K>4 failure was
  per-block states), and acting is partitioned one head per env per episode (temporally consistent
  Thompson sampling). Keep a constant within-head floor.
- **C, if needed, is Retrace** on REPPO's lambda-returns with stored Beta log-probs, not 1-step
  targets (v32/v34 already did that).
- **Flip bistability can decide single-seed verdicts.** Prefer the population base (0 of 8 heads
  flipped) or forks for mechanism ablations.

## Direction B rejected (ensembles); pivot to distributional

Ensembles/populations of critics are out. Spread should come from the return distribution Z(s,a)
that the categorical critic already models, not from disagreement between K scalar Qs. The actor
should stop climbing scalar E[Z]: any objective linear in pi has a point-mass optimum, which is why
every variant either needs an entropy patch or collapses.

## Update after the iteration-rate / v48 screen (superseded for B)

Update rate is refuted as the TD7 lever (+20% at 1M for 8x iterations). New measured law: critic ranking
correlation grows with behavior spread (0 at std 0.1, 0.1 at 0.3, 0.29 at 0.45). The score is the
behavior's own return during training, so the optimal behavior minimizes regret: explore where
information is worth its cost. For that, posterior (Thompson) sampling is the principled answer, so B
goes next. Heads use fixed within-head noise (no learned spread; KL steps then measure pure mode
movement, with no sigma coupling). Spread beyond that is the heads' disagreement, from critics made
diverse by randomized priors and per-head bootstraps.
