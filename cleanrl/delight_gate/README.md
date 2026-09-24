# Delightful Policy Gradient on a Beta policy

Osband et al. (2026) DG: score weight w = sigmoid(U * l / eta), l = -log pi(a|s), detached.
Setup for every run: residual SiTU-GLU nGPT v7 base, V(s) critic, reward norm, one
full-batch on-policy gradient step per rollout (no epochs, minibatches, ratios or replay).
Control: `dgate_pg_n64_v1` = same file with gate = 1. HalfCheetah-v4, 8M steps, seed 1.

| version | run | final | entropy@8M |
|-|-|-|-|
| control | dgate_pg_n64_v1 | 6695 | -4.1 |
| v1 | dg_tail_eta{0.33,1} (gate on GAE U, eta = eta0 * RMS(U)) | -280 (uniform policy) | max |
| v3 | dg_gae_density_abs_eta1 (literal Algorithm 2) | 5698 | -1.5 |
| v3 | dg_tdc_tail_eta1 (gate on delta - D(s)) | 4757 | +0.16 |
| v3 | eg_tdc (hard gate 1{delta - D(s) > 0}) | 4078 | -5.3 |
| v4 | mg_gae_density_abs_eta1 (literal Alg. 2, mean-only) | 6470 | -2.8 |
| v4 | mg_td_tail_eta1 (td gate, mean-only) | 5633 | -2.5 |
| v4 | mg_gae_tail_eta1 (GAE gate, mean-only) | 4662 | -1.9 |

## Mechanisms (each confirmed by the next version)

1. v1: noisy U in the gate. With U = A + eps, E[sigmoid(U l/eta) U grad log pi] contains
   Var(eps) grad H / (4 eta): entropy ascent. RMS temperature made it ~11x stronger.
   v3 fix: gate on the one-step delta, a function of (s, a) in deterministic MuJoCo.
2. v3: even noise-free, a unimodal policy with a locally linear advantage A = g x gets
   location update = 0.5x PG (same direction, no between-state reweighting), plus a
   positive log-std term ~ g^2 that PG does not have: an adaptive entropy bonus that Adam
   amplifies. Every soft-gated v3 arm held entropy 3-5 nats above the control.
3. v4: gate only the Beta location score (m = alpha/kappa, kappa detached); kappa keeps PG's
   weight. Recovers +770..+900 over the matching v3 arm; the literal arm ties the control.
   Stronger (more discriminating) gates score lower, and GAE gating is worst: for a skewed
   Beta E[l d log p/dm] = dH/dm != 0, so U noise pulls the mean toward the range centre.

## v5: paper reward scale, gate centering and utilization

Ours divides rewards by the running std of the discounted return; the paper by an EMA
(0.999) of the per-step reward std. Measured ratio: ~100-145x, so our literal Algorithm 2
had |chi| ~ 0.15, gate utilization 2E|w - 1/2| = 0.05 (effectively PG), and its density
surprisal was negative on 91% of samples (inverted gate). v5 evaluates only the gate in
paper units (score/critic unchanged; Adam makes them scale-free).

| run | final | utilization | saturated | inverted | entropy@8M |
|-|-|-|-|-|-|
| alg2, our scale, mean-only | 6470 | 0.05 | 0.00 | 0.91 | -2.8 |
| td+tail, paper scale, mean-only | 5852 | 0.47 | 0.27 | 0 | -2.4 |
| td+tail, paper scale, whole | 4936 | 0.51 | 0.32 | 0 | -0.1 |
| alg2, paper scale, mean-only | 4175 | 0.75 | 0.62 | 0.83 | -1.4 |
| alg2, paper scale, whole | 3778 | 0.85 | 0.78 | 0.04 | +1.8 |

The reward-norm difference is real and large, and it explains why the literal arm tied
PG: its gate was off. Turning the gate on hurts, monotonically in utilization; the
correctly centred (tail, never inverted) gate hurts least. Whole-score gating raises
entropy until the density surprisal turns positive (inverted 0.83 -> 0.04).

## Conclusion

On HalfCheetah under single-step on-policy PG, correctly implemented DG does not beat the
matched PG control: its only first-order effect on a unimodal continuous policy is variance
inflation, and once that is removed the location reweighting is neutral-to-harmful. DG's
claimed gains come from tail discovery on exploration-limited tasks, which this benchmark
does not exercise.
