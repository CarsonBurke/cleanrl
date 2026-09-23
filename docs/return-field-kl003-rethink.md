# KL 0.03 result and a rethink of critic supervision

The intervention proposal below was rejected by the user and will not be
implemented. It remains here as a record of the abandoned direction. The
subsequent [vector costate design](vector-costate-design.md) rethinks the model's
prediction target and reward-aligned training pressure instead.

## Completed experiment

MLQ **7511** completed fresh HalfCheetah-v4 training with seed 1, 50 CG
iterations, and `--trust-kl 0.03`. No model was resumed. Parallel limit 1,
60-minute time limit; 8,044,160 environment transitions collected.

| Collected steps, approximately | KL 0.01, learned CG50 | KL 0.03, learned CG50 |
|---|---:|---:|
| 1M | 158.52 | 569.96 |
| 2M | 640.99 | 1104.81 |
| 4M | 1334.30 | 1613.50 |
| 6M | 1980.52 | 1680.47 |
| Final 8.04M | **2337.50** | **1916.55** |

Scores are means of the last 100 completed training episodes. Mean realized KL
was 0.0299791, maximum 0.0299999. The larger budget was actually used; it did
not remain merely a ceiling. Mean CG relative residual was 0.04101 versus
0.02964 at KL 0.01. No line-search expansion-cap hits occurred.

The early improvement did not persist. Final return fell **18.0%** relative to
the KL 0.01 run. This single seed does not establish a universal optimum KL,
but it gives no basis for promoting the larger-step configuration or scaling
it to 50M. The trainer source is unchanged; KL is an existing argument.

Full [run result](../runs/HalfCheetah-v4__return_field_natural_v9_kl003_cg50_learned_8M__1__1789541262788219790/result.json).

## What needs rethinking

The current 132-coordinate field predicts conditional reward-score moments.
Those population quantities are aligned with return maximization, but each
observed label multiplies an action score by a noisy reward sum. Additional
heads do not create additional causal observations. Tensor PopArt conditions
regression; it does not establish that the fitted sum has accurate direction.

Another potential limitation is policy context: local alpha/beta parameters do
not specify how the policy acts at future states. Retained critic weights can
therefore carry stale future-policy information. Neither this nor target noise
has been isolated as the cause of the KL experiment's result.

Merely forcing the fitted field's aggregate parameter gradient to match the
sampled gradient would reproduce that noisy sampled estimator. Similarly,
compatible linear normal equations recover a sampled natural gradient; that
identity alone does not supply a stronger learned critic. These are not the
next proposed changes.

## Proposed next model-training design: intervention-supervised credit

This is a design proposal, **not an implemented or evaluated trainer**. It
retains the direct vector critic but changes the information used to train it.

At an on-policy source state and episode age, select an actuator i and a
temporal band b independently of the sampled action and outcomes:

1. Sample the other actuator values once. Draw a_i and a_i' independently from
   the current Beta distribution for actuator i.
2. Clone the complete simulator and normalization state. Execute the two
   actions, differing only at actuator i.
3. Continue both real simulator branches under the same frozen policy with
   shared future sampling randomness, through the selected band's upper end
   or the episode boundary.
4. Observe each branch's actual reward sum in that band. Train the selected
   pair of whitened alpha/beta coordinates with

       y[b,i] = 0.5 * (z_i(a_i) - z_i(a_i')) * (R_b - R_b').

The target has the same desired conditional expectation as the present field:

    E[y[b,i] | source] = E[z_i(a_i) * R_b | source].

To see why, expand the product. The two matching terms have identical means.
Each cross term vanishes because the initial own-actuator draw is independent
of the other branch's reward and its centered score has mean zero. Conditioning
on the common other-actuator values and future random stream preserves this
argument. Initial antithetic draws would invalidate that independence and are
not interchangeable with independent draws.

The label now measures a controlled change in actual future reward. Common
state-dependent offsets cancel. Shared continuation randomness may also cancel
some unrelated trajectory variation. This is an expectation identity, **not a
claim that the paired estimator always has lower variance**: long chaotic
continuations may separate enough to erase its benefit.

There is no scalar value bootstrap, GAE, generated latent reward decoder, or
imagined tail. Scalar task rewards remain observed measurements; the critic
still predicts independent temporal and actuator/statistic coordinates. The
actor combines these coordinates only when forming its return gradient, then
uses the existing coupled Fisher and exact joint KL constraint.

## Details that determine whether this is practical

- Simulator cloning is privileged access. This is a MuJoCo-specific training
  protocol change, not an ordinary observation-only PPO comparison. It must be
  reported explicitly, and its physical state, wrapper age, normalization
  history, and random-stream restoration need numerical contracts.
- All source and branch transitions count toward the same 8M budget. No free
  counterfactual interactions or resumed models. Complete source-policy episodes
  supply training-return reporting; partial branch rewards are not comparable
  episode scores.
- Sample one band rather than always paying for the whole remaining episode.
  Stopping at that band's upper endpoint is exact for that head and needs no
  tail estimate. Band/actuator sampling probabilities must be explicit; use
  inverse selection weights where needed to preserve intended fitting weights.
- Unobserved heads are masked, not assigned zero targets. Coordinate PopArt
  updates use observed per-head counts. Known post-episode support remains zero.
- Reuse branch prefixes for additional observational band-score targets only
  where the entire corresponding band was actually observed. This provides
  dense short-horizon supervision without inventing missing long-horizon labels.
- Keep source-state weighting for the actor separate from intervention branch
  occupancy. Branching preferentially revisits some ages; blindly using all
  branch states for the actor would change the objective's state weighting.
- Shared random streams must preserve each branch's Beta marginal. Using common
  quantiles is one possible coupling; matching gamma-sampler seeds alone does
  not ensure useful coordinate-wise coupling when rejection counts differ.
- Freeze the actor throughout source collection and all paired continuations.
  Positive predicted actor gain still does not guarantee improved real return.

The hypothesis is now specific: controlled action comparisons will provide
more learnable directional credit per counted interaction than observational
long-return score products. It remains untested. Fresh end-to-end return, with
all branch costs included, is the deciding outcome; no held-out variance gate
would establish success.
