# OPSD family charter — on-policy self-distillation for continuous control

Status date: 2026-08-25. Env: **HalfCheetah-v4 only**, seed 1.
Supersedes `OPSD_NORM_ABLATIONS.md` as the family's definition; that file's measured
ablation results remain valid and are summarized below.

---

## 1. What this family IS

One network. Two contexts. The actor teaches itself.

```
rollout once  ->  record (obs, action taken, policy that took it, advantage it got)
              ->  replay each step: teacher = pi(. | s, privileged),  student = pi(. | s, absent)
              ->  minimize per-action-dim divergence(teacher || student).  Done.
```

- The **teacher is the student**, evaluated with a privileged conditioning input that the
  acting policy did not have: the credit `delta_t` that the recorded action actually earned.
- The improvement operator is a **conditional shift**, not a weighted gradient. The
  advantage is an *input* to the network, never a multiplier on a gradient.
- **No PPO.** No importance ratio, no clipped surrogate, no advantage-weighted policy
  gradient anywhere in a backward. `pg_loss` / `approx_kl` / `clipfrac` are telemetry only.
- The update is **supervised regression onto detached targets**. It has the shape of a
  pretraining pass, not of a policy-optimization step. §3 makes that a hard requirement.

## 2. HARD EXCLUSIONS — not in scope, do not re-derive these

These have been ruled out by direction, not by experiment. Every one of them is a way of
smuggling *search* or *ensembling* into a method whose whole claim is that a single actor
bootstraps itself from one trajectory.

| excluded | why it is out |
| --- | --- |
| **Contrastive / multi-sample / multi-candidate** teachers | Sampling K actions per state and ranking them is search. The teacher must be built from the ONE action that was taken. |
| **Ensembles, twin critics, population methods** | Same reason: information from parallel copies, not from self-bootstrapping. |
| **Q-learning / action-conditioned critics** | Out by direction. The value object is **V(s)**. No Q(s,a), no twin critics, no target networks, no DPG/max backup. |
| **Simulator-privileged probing** (save/restore physics, evaluate counterfactual actions) | Not an RL assumption, and it is search wearing a distillation costume. |
| **Outcome conditioning** (feeding the episode's final return / success as context) | Ruled out by direction; also degenerate (provably a fixed point in earlier arms). |
| **Finetuning / hyperparameter grids** | The family advances by mechanism, not by coefficient search. |
| **Other envs** | HalfCheetah-v4 only. Hopper/Walker are out of scope. |

### Recorded drift — 2026-08-25

I violated the first and third rows: I built `probedistill_deepvalue_v1/v2` and
`probedistill_amplify_v3`, which probe **K=8 counterfactual actions per state** through raw
MuJoCo with a `qpos/qvel/qacc_warmstart/time` save-restore, and (v3) amplified that to 64
candidates via a learned Q head. That lineage scores well on this disk (its ancestor reaches
10411 vs PPO's 8278) precisely *because* it is search. **Jobs 3669 / 3677 / 3681 cancelled.**
The files stay on disk as a labelled dead end; they are not part of this family.

The measurement that made the violation obvious is throughput: those runs sustained
**448 SPS against PPO's 3001** — 6.7x slower. A method that is supposed to be a supervised
pass cannot be 6.7x more expensive than the thing it replaces.

## 3. THROUGHPUT IS A DIAGNOSTIC, NOT A GOAL

Throughput is **downstream** of a correct solution, never a target to optimize. Do not chase
SPS, do not tune for it, and do not trade mechanism for it.

The reason it matters is diagnostic: this family's update is regression onto fixed, detached,
once-computed targets. There is no ratio to re-evaluate, no trust region to line-search, no
candidate set to score. A correct member of this family is therefore *structurally* cheap --
roughly a supervised pass over the batch -- and should come out FASTER than PPO without
anyone optimizing for it. So when an arm is slow, the slowness is **evidence that it is
computing something the family has excluded**, and that is what makes it worth reading.
Slow != wrong on its own; slow means "go look for the search you accidentally added".

That is exactly how the 2026-08-25 drift was caught: 448 SPS against PPO's 3001 was the
symptom that surfaced the counterfactual probing, not a performance complaint.

Measured throughput across the disk, read as a diagnostic:

| variant | SPS | score @8M | what the SPS tells us |
| --- | --- | --- | --- |
| PPO baseline (reference) | 3001 | 8278 | the thing we replace |
| `advcond_v6` (family champion) | 2417 | 8718 | slower than PPO -- the two-context forward is being paid twice |
| `advcond_v11` (v6 + deep value readout) | 3928 | 8842 | faster, but the readout bought nothing (§4) |
| probe ancestor | 1013 | 10411 | excluded (§2); the 3x slowdown IS the search |
| best on disk (probe + all-layer tape) | 1568 | 11015 | excluded (§2) |
| probe arms, cancelled | 448 | — | excluded (§2); 6.7x slower than PPO |

The one durable engineering fact worth keeping: the acting forward is **launch-bound, not
compute-bound** (eager `act` costs 1149 us at batch 16 and 1277 us at batch 256 -- flat in
batch size). `--async-envs` and `--compile-act` default to `True` on `advcond_v6/v10/v11`.
This is context for reading SPS numbers, not a work item.

## 4. Measured family results (still valid)

### The champion, and where its margin comes from

`cond_scale=ema_rms` (scale-only conditioning, no mean subtraction), 8M, seed 1:

| | @500k | @1M | @2M | @4M | @6M | @8M |
| --- | --- | --- | --- | --- | --- | --- |
| PPO baseline (same chassis) | **1576** | 3079 | 5012 | 6716 | — | 8278 |
| **`ema_rms`, kappa=1** | 994 | **3826** | **6267** | 7680 | **8598** | **8819** |
| `ema_rms`, kappa=2 | 1524 | 3993 | 6084 | **7848** | 8486 | 8249 |

+6.5% over PPO at 8M from one variable: how the advantage is scaled before being used as a
conditioning input. `score_runs.py` last-20 reads 8718 +-498.

### Why advantage normalization is the load-bearing detail

PPO normalizes advantages to control **the scale of a gradient**. This family never
multiplies a gradient by an advantage — `delta_t` is an **input** to a learned conditional.
Batch mean/sd is the wrong tool for an input:

1. **Mean subtraction destroys the natural zero.** `delta_t = 0` means "exactly as V(s)
   expected". After subtracting the batch mean, the least-bad action in an all-bad batch is
   relabelled *positive*: the conditional is taught a false sign.
2. **Batch sd makes the units non-stationary**, so the margin `adv_boost` means a different
   real gain every iteration.

Counter-constraint: `delta` cannot be fed raw either, because `AdvEmbed`'s Fourier
frequencies are fixed at 0.5..8 and only resolve an O(1) input, while raw `delta` reaches
RMS ~14.7 against a +-3 clamp. Scale-only tracking is the resolution.

Entropy preservation orders `raw` > `ema_rms` > `batch`, exactly as the false-sign argument
predicts: teaching a wrong sign has to be paid for by sharpening the conditional.

### The family's real disease: the operator runs out of things to say

From the champion's own telemetry:

| metric | 0k | 500k | 1M | 1.5M | 1.75M |
| --- | --- | --- | --- | --- | --- |
| `advantage_std` | 3.94 | 6.25 | 14.56 | 14.44 | **14.72** |
| `cond_gap` | 0.030 | 0.022 | 0.016 | 0.011 | **0.010** |
| `distill_kl` | 0.162 | 0.096 | 0.066 | 0.046 | **0.039** |
| teacher - student entropy | -0.146 | -0.093 | -0.068 | -0.049 | **-0.046** |
| student entropy | -0.72 | -1.90 | -3.70 | -5.23 | **-5.71** |
| explained variance | 0.38 | 0.95 | 0.97 | 0.96 | 0.94 |

`delta`'s spread **grew 3.2x** — the channel is not starving. The policy **sharpens** (5 nats
of entropy gone) and the two contexts converge onto each other: `cond_gap` 3x down,
`distill_kl` 4x down. The improvement signal *is* that gap. The critic is not implicated
(EV 0.94-0.97 throughout).

**This is the thing to attack.** It is also why the measured edge decays monotonically
against PPO (+24% @1M -> +6.4% @8M, measured against v6's own parent chassis): a teacher
built from one action and one scalar carries exactly the information content of a policy
gradient, so it can only reach the same asymptote sooner. Raising the asymptote requires the
privileged context to say something the student structurally cannot infer — **without**
resorting to anything in §2.

### Falsified predictions, kept so they are not retried

- `clone_coef` has a **non-monotone optimum at 1.0**: 0.5 -> 4636 @2M, 1.0 -> 5365, 2.0 -> 4340.
- Removing the clone term's sharpening while keeping its calibration (v9) is **worse**
  (4136 @2M) despite behaving exactly as designed. On HalfCheetah the sharpening is a
  feature, not the pathology.
- Linear kappa annealing over 2M did **not** capture both ends (4->1: 4415, 4->2: 5864,
  both below constant kappa=2).
- Deep value readout does **not** transfer: +1355 on one chassis, **+124 (inside noise)**
  on `advcond_v11`. Both deep-head runs share a signature — higher EV early, *lower* EV at
  8M (0.914 vs 0.973). The family's critic is already at EV 0.97; there is no value-fit
  headroom to buy.

## 5. Noise floor — read every comparison against this

`v3_mb128_e4` and `v5_a4c4_mb128` are **the same algorithm on the same seed**, differing only
in floating-point association, and read **2908 vs 2581 @1M**. So ~300 points at 1M is pure
numerical noise and only seed 1 exists. **Differences below ~400 are not findings.**
`advcond_v11`'s +124 over v6 is inside this floor.

## 6. WHERE DOES THE OPTIMIZATION PRESSURE COME FROM?

This is the load-bearing question for the whole family, and getting it wrong is what capped
every arm so far at PPO+6%.

### 6.1 Why the LLM version works and the naive port cannot

In OPSD-for-LLMs the teacher conditions on the ground-truth solution `y*` and **rationalizes**
it. That works because the teacher is already a capable reasoner: the privileged context does
not teach it anything, it *unlocks* competence the weights already have. Distillation then
compresses "capable model with a hint" into "capable model without one".

A 64-unit MLP on HalfCheetah has no such latent competence and no in-context reasoning.
Showing it `delta_t = +0.3` unlocks nothing. The conditional `pi(a|s, delta)` has to LEARN
the credit -> action mapping from the same data it is trying to improve on. That is the
circularity, and the family's own telemetry shows it closing: `cond_gap` 0.030 -> 0.010 and
`distill_kl` 0.162 -> 0.039 while the policy sheds 5 nats. **The teacher stops being able to
say anything the student does not already say.**

### 6.2 The precise diagnosis: the credit channel is mostly noise at fixed s

`cond_gap` collapsing to 0.010 does not mean the channel starved -- `advantage_std` GREW 3.2x
over the same window. It means the network learned to **ignore** `delta`, and it was right to:
at a fixed state, the mutual information between `a_t` and `delta_t` is small next to the
critic-error and myopia noise in `delta_t`. A conditional trained on a near-noise input
correctly converges to ignoring it. Everything the family has tried on this axis -- batch vs
`ema_rms` vs `raw` scaling, kappa ladders, kappa annealing, KL-targeted dosing, future-window
pooling in the `hopsd` line -- is a different encoding of the SAME low-SNR scalar, which is
why they all land in 8200-8800.

### 6.3 THE RESOLUTION: pressure is Bayes, not a loss term

The objection that kills the naive reading of this family is: conditional MLE has no term
that pushes return up, so every policy is a fixed point of "predict what you did", so
letting the network learn how to use credit *removes the optimization pressure* that a
hand-coded policy loss supplies.

That objection is wrong, and the reason is worth stating exactly, because it dictates every
design choice downstream. The teacher is queried at a **better outcome than the one that
happened**, so

    T(a|s)  =  p(a | s, outcome better)  ∝  p(outcome better | s, a) · S(a|s)

The teacher is the **posterior** over actions given a good outcome; the student is the
**prior**. Distilling `S <- T` is posterior sharpening on the event "good outcome", and
`T = S` **if and only if** `p(outcome better | s, a)` does not depend on `a`. So the fixed
points are exactly the states where the outcome is action-independent — local optimality —
and *not* every policy. Pressure exists, it is nonzero wherever the policy is improvable,
and nothing about it is hand-coded: the weighting is whatever likelihood the network fit.

This is what separates the family from outcome conditioning (§2). Outcome conditioning feeds
the episode's terminal return, which is action-independent for any single step, so
`p(better|s,a)` is flat, so it is a fixed point *everywhere* — provably degenerate, as the
earlier arms measured. The one-step realized consequence is the opposite: maximally
action-attributable.

**The corollary is the whole engineering programme:**

> Pressure and channel informativeness are THE SAME QUANTITY. If the conditioning channel
> carries no information about the action, `p(better|s,a)` is flat in `a`, `T = S`, and the
> method is inert — regardless of how large the dose is.

So the way to get pressure is never to add a term. It is to keep the conditioning channel
action-attributable. Every "the operator ran out of things to say" observation in §4 is this
one fact seen from a different angle.

### 6.4 Why the champion lost its pressure, measured

`advcond_v6` fed one scalar, `delta_t = r_t + gamma·V(s_{t+1}) - V(s_t)`. Its own telemetry:

| step | 0k | 500k | 1M | 1.5M | 1.75M |
| --- | --- | --- | --- | --- | --- |
| `advantage_std` | 3.94 | 6.25 | 14.56 | 14.44 | **14.72** |
| `cond_gap` | 0.030 | 0.022 | 0.016 | 0.011 | **0.010** |
| `distill_kl` | 0.162 | 0.096 | 0.066 | 0.046 | **0.039** |

The channel got 3.7x **louder** while the network used it 3x **less**. It did not starve; the
network learned to ignore it, and was right to. Measured causes:

- A compressed credit scalar is un-learnable. Synthetic Beta policy, 6 dims, credit with 10%
  action-attributable variance: **R²(action | credit scalar) = 0.0000**, against
  **R²(action | surprisal) = 0.9506**. A channel that predicts nothing about the action is a
  channel MLE drops for free — which is what `cond_gap = 0.010` is showing.
- `delta` is a **sum of an exactly observed term and a difference of two large learned
  values**. One scalar makes the zero-noise part share a channel with the critic-error part,
  and the critic-error part wins precisely as the policy sharpens (entropy -0.5 -> -9.6) and
  the reward scale grows. v6 had already moved from GAE(0.95) to the 1-step residual for this
  reason, and had already made the channel expensive to ignore with fixed Fourier features.
  Neither helps: **this is an SNR problem, not an encoding one.**
- The observed part *is* learnable. 60k real HalfCheetah-v4 steps, gain in R² over state
  alone: `r_t` -> **+0.1001** on `||a||²` and **+0.0839** on `a_1`; `s_{t+1}` -> **+0.4661**
  on `a_1`. Reward variance is 94.4% forward-velocity and 6.1% control cost, and the control
  cost is a *deterministic* function of the action (corr with `-||a||²` = 1.000000).

### 6.5 Direction

1. **Stop compressing the credit signal.** `ppo_continuous_action_opsd_conseq_v1.py` feeds
   the three quantities `delta` is built from as separate scaled channels — observed reward
   surprise, the critic's state-change, and long-horizon GAE credit. Channels 1+2 span v6's
   scalar, so **v6 is the special case with the readout pinned at (1,1) and the horizon
   pinned at lambda=0**; here the network learns both, per state. `--cond-mode delta`
   reproduces v6 bit for bit (verified: 15 TensorBoard scalars, zero differing), so the
   control is free and the comparison is exact. Falsification instrument:
   `debug/chan_use_*` ablates each channel out of the query. If all three decay like v6's
   single one did, the decomposition is not the cure.
2. **Judge it on the DECAY, not the level.** The lineage signature is +24% @1M shrinking to
   +6.4% @8M. A pressure fix must *flatten* that curve. An arm that lifts the early curve and
   still decays has not addressed §6.3.
3. **Honest scope.** Decomposition is an estimator and allocation fix. The improvement
   direction is still built from the action-credit correlation, which for an
   exponential-family policy is the policy gradient's own information — so it buys a
   better-conditioned estimate, a learned rather than pinned readout, and equal per-state
   allocation. At a fixed 8M budget that is exactly the lever that moves score, but it is not
   a claim about the asymptote. **The only route to a different information source that stays
   in charter is a learned one-step transition Jacobian** giving
   `grad_a [r_hat + gamma·V(f(s,a))]` — V(s) stays the value object, the model is supervised
   regression on observed transitions, no counterfactual stepping, no candidates. That remains
   the ceiling play; it is strictly more machinery than (1), so (1) goes first.
4. **Evidence that the information argument is right.** The only arm on this disk that ever
   escaped the PG band (11015 vs the 8278-8842 band) is the probe lineage — and it escaped by
   evaluating counterfactual actions through the simulator, i.e. by finite-differencing the
   very Jacobian route 3 would learn. It is excluded (§2), but it is the natural experiment
   confirming that escaping the band requires escaping the score-function information source.

### 6.6 Route 3 is BUILT and VALIDATED — 2026-08-25

`cleanrl/opsd/core/jacteach/ppo_continuous_action_opsd_jacteach_v1.py`. §6.5 item 3 called this the ceiling play
and deferred it as "strictly more machinery". The machinery turned out to be one 2-layer MLP,
and the two load-bearing questions were both settled by measurement *before* any GPU slot was
spent.

**Q1 — is a learned one-step Jacobian accurate enough to carry a direction?** Yes, and it is
cheap. 120k real HalfCheetah-v4 transitions, learned J against the simulator's own central
differences on 300 held-out states:

| n_train | held-out Δs R² | cos(Jᵀe_vx) | cos(full J) | J rel.err |
|---|---|---|---|---|
| 5,000 | 0.841 | **0.955** | 0.969 | 0.260 |
| 20,000 | 0.906 | 0.968 | 0.975 | 0.236 |
| 100,000 | 0.944 | **0.975** | 0.975 | 0.234 |

`obs[8] = qvel[0]` is forward velocity, which is 94.4% of this env's reward variance, so
`cos(Jᵀe_vx)` *is* the accuracy of the improvement direction. Two structural facts fell out:

- **Direction is excellent, magnitude is biased and stays biased.** Rel. error sticks at 0.23
  while cosine improves. So a design may use the direction and must *not* use the norm — take
  the dose from the KL budget, never from the gradient magnitude.
- **No warmup problem.** 0.955 at 5k samples; one rollout is 32,768. The accuracy gate is a
  guardrail against iteration 1, not a schedule.
- Actions are torques, so J is dominated by the qvel block: median |∂s/∂a| is 0.0052 on qpos
  rows against 0.1359 on qvel rows, a 26x split. Regression targets must be standardized per
  dimension or the fit sees velocities only.

**Q2 — does the pathwise direction actually say something the credit channel does not?** Yes.
`debug/jac_align`, the cosine in the policy's own metric between the analytic direction and
the parent's own teacher displacement, measures **−0.06 .. +0.02, i.e. ~0, on every ladder
run**. Orthogonal. This was the pre-registered kill condition: ~1 would have meant the
pathwise route merely rediscovers the score function, and the arm would have been abandoned
rather than tuned.

**The confound that orthogonality creates, and the fix.** Orthogonal directions ADD in
quadrature, so mixing inflates the displacement — measured 2.3x the parent's `distill_kl`.
Since §4 already establishes that dose alone moves score (κ 1/2/3/4 → 5365/6077/5624/3128
@2M), an uncorrected arm confounds "better direction" with "bigger step". Two corrections
were needed, and the second was found only by measuring:

1. **Specify the angle, do not let it emerge.** The first lever added `jac_step * g_hat` and
   renormalized, so the achieved angle depended on `‖u_cred‖` — which grows with the
   teacher's aggressiveness and drifts through training. The same `jac_step = 0.10` rotated
   the teacher 57° at batch 1024 and 21° at batch 32768. A lever whose meaning moves with
   batch size cannot support a matched-step ladder. Replaced by an exact in-plane rotation:
   verified in float64 over 20k rows, achieved cosine equals the argument to 10 decimals,
   `‖u_new‖ = ‖u_cred‖` to 7.4e-16, and `jac_cos = 1` is an algebraic identity (8.9e-16)
   rather than a skipped branch — a strictly stronger control than the branch-skip.
2. **Match the DELIVERED budget, not the displacement norm.** The loss charges a per-dim
   *clipped* KL. The parent's direction saturates that clip hardest, so it delivers the least
   divergence per unit displacement, and *any* rotation de-saturates the clip and delivers
   more. The cosine therefore cannot double as a dose control. Resolved with one bisected
   scalar per iteration against the parent's own delivered KL on that batch
   (`debug/jac_dose_scale`).

Ladder, **champion config** (16 envs × 2048, mb128, a4/c4, 8 iterations, seed 1):

| jac_cos | jac_rot | jac_align | dose_scale | distill_kl | vs v6 | cond_gap |
|---|---|---|---|---|---|---|
| parent | — | — | — | 0.11192 | 1.00x | 0.02851 |
| **0.95** | 0.9500 | +0.001 | 0.9857 | 0.11897 | **1.06x** | 0.03031 |
| 0.85 | 0.8500 | +0.046 | 0.9543 | 0.13924 | 1.24x | 0.03227 |
| 0.50 | 0.5000 | +0.006 | 0.8864 | 21.52324 | 192x | 0.03544 |

**The 0.50 row is a finding, not a dosing failure.** Snapshot-time dose is matched by
construction, so 192x means the *student cannot follow* a teacher rotated 60°: the clip binds
on many dims, those dims go locally flat, the student receives no gradient on them, and the
divergence never comes down across the update epochs. There is a followability limit on how
far the teacher can be turned at a fixed budget, and it sits well inside 60°. Launched arms
are 0.95 (clean, +6%) and 0.85 (aggressive, +24% disclosed).

**Charter compliance, explicitly.** V(s) remains the value object — no Q(s,a), no twin critics,
no target networks, no max backup (§2). The model is supervised regression on the observed
transitions of the single trajectory: no counterfactual stepping, no candidate sets, no
save/restore, and it is never rolled forward, so compounding error does not arise (§2's
simulator-privileged-probing exclusion is respected — this is the *learned*, in-charter analog
of what the 11015 probe did through the simulator). The head is deliberately NOT on the shared
trunk, because routing model gradients through the actor's representation would confound "the
analytic direction helps" with "next-state prediction is a good auxiliary task", and §4 already
records this family being burned by that class of confound.

**Safety property.** Dose is gated on the model's measured held-out one-step R² (`debug/jac_gate`),
held-out by construction since the head trains only on earlier rollouts. Gate closed ⇒ zero
displacement ⇒ the file *is* the parent. `--jac-step 0` is bit-exact against
`ppo_continuous_action_opsd_advcond_v6.py` (18 scalars, zero differing), so the champion's 8718
@8M is this arm's control for free, and the arm cannot lose through model incompetence — only
through a measured, licensed step. The head's init draws are taken inside `fork_rng` precisely
so that control holds.

**Judged on §6.5 item 2:** the decay, not the level. Parent is +24% @1M → +6.4% @8M over PPO.
Read @4M–@8M; a win that appears only @1M is a dose artifact.

**Provenance correction — 2026-08-25.** The first launch of both `jacteach_v1` and
`conseq_v1` used the FILE defaults (`num_minibatches=32`, `actor_epochs=1`) instead of the
champion's `--num-minibatches 128 --actor-epochs 4`: 32 gradient steps per iteration against
the champion's 512. §4 already records gradient-step count as the dominant lever on this
chassis, and both arms duly trailed v6 at matched *early* steps (368k: v6 +775, jacteach −185,
conseq −23) — a configuration artifact, with the mechanisms left untested. Cancelled and
relaunched with the champion's exact flags (jobs 3698 `cos 0.95`, 3699 `cos 0.85`, 3700
`conseq`). **Rule going forward: the champion's flags are part of the champion. A new arm's
launch line must be diffed against the incumbent RUN's recorded `hyperparameters` text, not
against the file defaults.**

### 6.7 Route 3 RESULT — two completed 8M runs, and a wall — 2026-08-25

`jacteach_v1` at `--jac-cos 0.95` is **the best arm this family has produced**. Both arms ran
the full 8M on HalfCheetah-v4, seed 1, champion config.

| variant | @1M | @2M | @4M | @6M | @8M | last-20 | ±CI95 |
|---|---|---|---|---|---|---|---|
| PPO baseline | 3079 | 5012 | 6716 | 8278 | — | — | — |
| `hopsd_v19_twowindow` (ref) | 4416 | 6249 | — | — | — | — | — |
| v6 (incumbent) | 3826 | 6267 | 7713 | 8579 | 8812 | 8718 | ±498 |
| v11 (prev. best) | 3163 | 6330 | 7854 | 8756 | 8882 | 8842 | ±152 |
| jacteach cos 0.85 | 3700 | 6800 | 8184 | 8420 | 8862 | 8854 | ±114 |
| **jacteach cos 0.95** | **4370** | **6701** | **8106** | **8835** | 8859 | **8915** | ±131 |

**What is established.** The mechanism worked exactly as specified and the confounds are
closed:
- `jac_rot` held at 0.9500/0.8500 for the entire run — the lever is exact and scale-free.
- `jac_gate` pinned 1.0, `jac_r2` ~0.95 — prediction 2 confirmed; the head is competent.
- `jac_align` 0.02–0.13 all run — prediction 1 confirmed; the pathwise direction stays
  orthogonal to the credit direction. It really is a different information source.
- `jac_dose_scale` 0.97–0.99 — prediction 4 confirmed; the rotation was nearly free.
- **Dose is ruled out as the explanation.** The κ arms are pure dose increases with no
  rotation: delivered `distill_kl` 0.0116 → 0.0207 → 0.0395 scores 8718 → 8129 → 7845, i.e.
  monotonically *worse*. cos 0.95 sits at 0.0163 on that curve, where dose-only predicts
  ~8400, and scores 8915. cos 0.85 sits at 0.0580, past κ=1.5's 0.0395, and scores 8854.
  Both rotated arms beat the dose-only trend by 550–1000.
- A free internal control: cos 0.95 and cos 0.85 reached the *same* delivered dose at 7.4M
  (0.0168 both) and differed by 437 @7M. Rotation matters at matched dose, and less is more.

**PREDICTION 3 IS FALSIFIED, and this is the important part.** The claim was that a direction
carrying information the policy gradient lacks would *flatten* the decay. It does not. Margin
over PPO: v6 was +24% @1M → +6.4% @8M; jacteach cos 0.95 is **+42% @1M → +34% @2M → +21% @4M
→ +7.0% @8M**. Better everywhere, same shape. Every good arm in this family now lands in
**8812–8882 at 8M** — v6, v11, and both rotated arms, a spread of 70 against CIs of ±114 to
±498. Four materially different actor-side mechanisms, one number.

**Therefore: the 8M asymptote on this chassis is not set by the improvement direction.** §6.3
diagnosed the credit channel and §6.6 fixed the information source; both raised the curve and
neither moved the wall. The actor-side lever is spent. Note what the wall is *not*: it is not
the environment, because other lineages on this same disk reach 10411 (`..._tpomd_v5_dyntrust`,
itself already a pure-distillation actor with no PG) and 11015 (`tpomd_alllayer_residual_td_v25`),
i.e. +17% and +24% over this family's best. Both differ from this chassis in the CRITIC/value
pathway, not in how the actor is taught — and v11's deep value readout, the one value-side
change this family tried, is the only arm that ever moved the 8M number at all (+124, inside
noise).

**Direction, revised.** Stop attacking the actor's teacher on this chassis; it is saturated at
~8.9k. The score-maximizing move is to carry the (now validated, orthogonal, dose-controlled)
analytic direction onto the substrate that reaches 10411–11015. That is a substrate change,
not a mechanism change, so the mechanism evidence above transfers as-is.

## 7. FRONTIER AUDIT - my "best in-charter" claim was WRONGLY SCOPED - 2026-08-25

I claimed jacteach cos0.95 (8915) was the best in-charter model on this disk. That is FALSE.
An independent audit scored all 1388 HalfCheetah-v4 runs, filtered to last-20 > 8915 with
>=7.5M steps, mapped each to its source file, and grepped every §2 exclusion.

Correct scoping:
- Best **OPSD-family** result: 8915 (`jacteach_cos0p95_8M`). This claim stands.
- Best **in-charter** result on the disk: **10362 +-92** (`entppo_v4_samplemask_a03_mask`,
  `cleanrl/iterthink/v24_d3bucket/other/ppo_continuous_action_iterthink_v24_beta_d3bucket_mtp_entppo_v4_samplemask.py`),
  zero §2-exclusion hits. Runner-up 10351 (`v24_beta_v162critic_mtp_v1`).
- The 11015 and 10371 runs ARE excluded as §2 predicted: all three carry the
  `qpos/qvel/qacc_warmstart` save/restore counterfactual-probing block (11 hits each).
  `dg_ppoaux_c00ctrl_v10` (9224) rides a 262k replay ring + frozen target critic.

### 7.1 The structure of the gap - the two mechanisms own OPPOSITE phases

| variant | @1M | @2M | @4M | @6M | @8M | last-20 |
|---|---|---|---|---|---|---|
| `jacteach_cos0p95` (target direction) | **4370** | **6701** | 8106 | 8835 | 8859 | 8915 +-131 |
| `entppo_v4_samplemask` (step control) | 3041 | 5465 | **8890** | **9873** | **10312** | 10362 +-92 |

jacteach is +44% @1M and +23% @2M, then loses from ~3M onward. The crossover is real and
measurable. These are orthogonal mechanisms: one builds the TARGET (analytic rotation of the
improvement direction), one controls the STEP (per-sample KL-drift mask + entropy alpha=0.3).
The family spent 8 arms polishing targets on a chassis whose step control saturates at ~8.8k.

**Redirection.** The remaining lever is not another target refinement on the advcond chassis.
It is the analytic-direction teacher ON the entppo chassis, keeping the per-sample mask as the
trust region -> `ppo_continuous_action_entppo_jacteach_v1.py`. The credit direction there is
free (`grad_a[A * log pi(a|s)]`, the PG direction) so advcond's conditioning apparatus is NOT
ported. Cancelled `jacteach_v2_rterm` (3755) to fund this: a +27%-of-direction fix on a
chassis capped at 8.8k is worth less than the chassis gap itself.

### 7.2 Arms measured and NOT launched (cull on evidence, before spending a slot)

- `jacteach_v2_stategate` (per-state accuracy gate replacing the batch gate). Built, bit-exact
  control passes in BOTH small and champion configs, and the heteroscedasticity motivating it
  is real (err_row p99/median = 11x). But the gate is measured **near-inert**: at steady state
  ~94% of states fully open, mean gate 0.976, i.e. only ~2.4% of nominal dose withheld;
  `jac_rot` 0.951-0.956 vs v1's exact 0.950. Alive but too small to pay for 8M steps.
  Useful by-product: the per-row statistic is an exact decomposition of the global one
  (mean(err_row) + jac_r2 == 1.0000000 in float32 at every iteration).
- `jacteach_v2_rterm` (adds the omitted `grad_a r_hat` control-cost term). Verified NOT a
  no-op: `jac_rterm_r2` 0.959 held-out, and the omitted term carries **27% of the direction's
  magnitude** (`jac_rterm_frac` 0.270), dose unchanged. Cancelled only for chassis reasons,
  not because it failed. **Re-test this on the entppo chassis if the port works.**

### 7.3 Corrections to THIS document, forced by review - 2026-08-25

1. §6.7 said the 8812-8882 band was "four materially different **actor-side** mechanisms, one
   number". Wrong: v6, cos0.85 and cos0.95 differ only on the ACTOR side, but v11 differs from
   v6 only on the VALUE side. The band is therefore "every lever tried so far, actor *or*
   value, lands in the same band" - a strictly WORSE prior than the original phrasing implied.
2. v11's +124 is **inside** the ~400 noise floor this document itself records in §5, and §4
   already lists the deep readout under falsified predictions (+1355 on one chassis, +124 here).
   So the value-side prior was never established. Consequence: a null result from a
   value-side combination cannot distinguish "the mechanisms do not compose" from "the
   value-side gain was never real". That is an underpowered experiment.
3. §4 reads 8819 and §6.7 reads 8812 for the same v6 arm. Immaterial against a ~400 floor,
   but the inconsistency is mine.

**Second cull, on that basis.** `ppo_continuous_action_opsd_jacteach_v3_deepvalue.py` is built
and fully verified (117/117 byte-identical value-side port, bit-exact vs v11 at `--jac-cos 1.0`
across 17 tags, deep head confirmed to REMOVE 32,576 params so "more capacity" is unavailable
as an explanation, exact all-zero init fixed point, actor mechanism unperturbed). It is NOT
launched: its parent effect is inside the noise floor, so at one queue lease it cannot return a
decidable answer, while the chassis port targets a measured +1447 gap.

Standing rule this session established: build, instrument, and measure the mechanism at 262k
BEFORE spending 8M steps, then cull on the measurement. Three arms built, all three verified
alive, two culled on measured smallness rather than on taste (`v2_stategate` 2.4% of dose,
`v3_deepvalue` parent inside noise). Only the arm attacking the measured gap gets the slot.

## 8. THE CHASSIS PORT, AND A CLEAN NEGATIVE RESULT - 2026-08-25

`cleanrl/opsd/core/jacteach/ppo_continuous_action_entppo_jacteach_v1.py` = entppo_v4_samplemask chassis (10362)
+ the jacteach analytic direction, one flag apart. Built and fully verified:
`--improve surrogate` is bit-exact to the chassis parent (36 tags, 720 points, zero
differing); rotation algebra exact in float64 (max |dcos| 1.2e-15, norm ratio 6.7e-16);
`jac_rot` exactly 0.9500; `jac_align` -0.028..+0.026, i.e. the analytic direction is
orthogonal to the credit direction on THIS chassis too, so it carries genuinely new
information; dose match to <=0.13%.

A by-product worth keeping: the credit direction's PARAMETERIZATION is load-bearing.
Differentiating w.r.t. the action argument instead of the distribution's mean yields the
ANTI-teacher - over 12 fresh agents one real chassis surrogate step moves the Beta mean with
cosine **0.9988** to the mean-parameterized direction and **-0.977** to the action-argument
reading, and the two are not exact negations for a Beta (per-row cos = 0.58).

### 8.1 FALSIFIED: pure forward-KL distillation cannot hold entropy on this chassis

Two failures, both measured at 262k, champion config (16x2048, mb32, 10 epochs):

1. **The chassis's trust region goes INERT.** In distill mode the student's realized drift is
   ~20x smaller than the surrogate's (`approx_kl` 5.8e-4 vs 9.9e-3), so at the chassis's
   native `tr_sample_eps=0.1` the per-sample mask rejects **exactly nothing**. Recalibrating
   to `eps=0.002` restores biting (4.1-5.2% masked) - the knob works.
2. **Entropy leaks upward and alpha is NOT the cause.** I predicted the chassis's alpha=0.3
   entropy bonus was the second entropy-raising force and that alpha=0 would fix it. FALSE:

   | arm | d(entropy) over 262k | tr_mask_frac | approx_kl | ret20 |
   |---|---|---|---|---|
   | surrogate (control) | **-0.038** | 0.006 | 0.0099 | **193** |
   | distill, alpha=0.3 | +0.075 | 0.041 | 0.00058 | -133 |
   | distill, alpha=0.1 | +0.074 | 0.048 | 0.00060 | -140 |
   | distill, alpha=0.0 | +0.075 | 0.052 | 0.00067 | -141 |

   d(entropy) is invariant to alpha across a 0.3 -> 0 sweep. The leak is intrinsic to the
   OBJECTIVE. Raising the dose instead makes it acute: `teach_step>=1.0` reaches a UNIFORM
   Beta (alpha=beta=1) within 4 iterations, after which ratio_max=1.000 and the policy is
   frozen.

**Diagnosis.** Forward KL is mass-covering, so a mean-shift teacher at FIXED concentration
gives the student no reason to ever sharpen; limited trunk capacity across per-state teachers
is then answered by broadening. The donor did not have this failure because its teacher was an
**MLE fit of the advantage-tilted distribution - mean AND concentration** - and tilting toward
the upper tail SHARPENS (this document's own §6.4 note: "tail conditioning shrinks spread").
By specifying a mean-shift-only teacher I removed the family's entire sharpening channel.
That was my error, not the chassis's and not the agent's.

### 8.2 The corrected composition

The surrogate is the entropy-LOWERING force that this chassis needs and that the family's
distillation objective does not supply. So the analytic direction should enter as an
ADDITIVE, KL-dosed auxiliary on top of the unmodified surrogate, not as a replacement for it:
the mask keeps biting at its native eps (it is measured on the surrogate's own drift scale),
entropy keeps its downward pressure, and at zero dose the file is bit-exact to the 10362
chassis. -> `--improve both`.

### 8.3 PRE-REGISTERED success criterion, written BEFORE the arm runs

The entppo lineage's four completed 8M arms span **9939 / 10076 / 10269 / 10362** - a 423-point
spread across four DIFFERENT mechanisms, which independently reproduces this document's ~400
noise floor (§5). Consequence: an 8M endpoint alone CANNOT decide this experiment with one
seed. jacteach's own effect on its chassis was +197 (+2.3%); the same relative transfer onto
10362 would be ~+240, i.e. inside the band and undecidable.

So the endpoint is NOT the test. jacteach's actual measured signature is EARLY, and it is huge:

| | @1M | @2M | @4M |
|---|---|---|---|
| jacteach vs ITS chassis (advcond_v6) | 4370 vs 3826 = **+14%** | 6701 vs 6267 = **+7%** | 8106 vs 7713 |
| jacteach vs entppo chassis | 4370 vs 3041 = **+44%** | 6701 vs 5465 = **+23%** | 8106 vs 8890 |

**Pre-registered PASS:** `@1M >= 3400` and `@2M >= 6000` against the chassis's 3041 / 5465,
with the entropy and mask gates (§8.2) holding all run. That is a >=12% / >=10% early move,
comfortably outside the 423 floor, and it is decidable by ~2M instead of 8M.

**Pre-registered FAIL:** early curve inside the chassis's own band (@1M < 3200, @2M < 5700)
=> the analytic direction does not transfer off the advcond chassis, and the +44% gap was a
property of the advcond chassis's weak early phase rather than of the mechanism.

**Pre-registered KILL (abort the run):** `losses/entropy` drift turns positive, or
`debug/tr_mask_frac` goes to 0. Either means the composition broke in the way §8.1 measured.

### 8.4 `--improve both` - the additive auxiliary. LAUNCHED (job 3842)

`--improve both`: chassis surrogate, entropy term, per-sample mask at native eps=0.1, epoch
logic and value loss all untouched; the rotated dose-matched teacher enters as ONE additive
masked quadratic pull of the student's Beta mean. Verified:

- **Zero-dose bit-exactness**: `--jac-aux-coef 0` vs `--improve surrogate`, 36 shared tags /
  720 points / **0 differing**. Not a skipped branch - at coef 0 the transition head is still
  built, trained, R2-gated, the rotation and bisection still run; only `0.0 * aux_loss` enters.
- **Entropy gate PASSES at the default** and is honestly a DOSE gate, not an unconditional one:
  d(entropy) = -0.032 / -0.039 / -0.032 / **-0.039** at 1.0 / 1.4 / 2.6 / **5.05%** declared
  dose against the surrogate's -0.038 (no trend - noise), then -0.019 at 14.9% and **+0.023**
  at 47.8%. An over-dosed `both` fails exactly the gate `distill` failed.
- **Mask stays alive at every dose**: `tr_mask_frac` 0.0053-0.0071 vs surrogate 0.0062 - never
  zero (so the auxiliary is visible to the trust region, unlike `distill`) and never inflated
  (so it is not driving the drift). `approx_kl` 0.0091-0.0105 vs 0.0101.
- **Mechanism alive**: `jac_rot` exactly 0.95 from iteration 3 (0.96577 at it2 is the gate ramp,
  `1 - 0.68463*0.05`, correct to 5dp); `jac_align` -0.044..+0.010 (orthogonal - new information);
  `jac_r2` -> 0.916; `jac_dose_scale` >= 0.99914; no non-finite scalar after iteration 1.

Chosen dose `jac_aux_coef=0.011` (5.05% declared), the top of the verified-open entropy band
with ~3x margin to the measured knee, selected from the gates and NOT from the 262k return.

**The honest weakness, and it is why §8.5 exists.** `debug/actor_clip_sat` is **1.000 in every
iteration**: the actor grad clip is always saturated, so the auxiliary's DELIVERED influence is
`||grad_aux||/||grad_pg||` = **0.0047** - about 10x below its declared dose. The entropy gate
caps the delivered effect at ~1.5%. A 0.47% nudge is very unlikely to reproduce a +44% early
move, so this arm is a safety-first probe, not the mechanism at strength.

### 8.5 `--improve rotate` - the design the two failures point at

Both failures share one cause: they changed the SIZE of the update.
`distill` replaced the objective with a bounded mass-covering pull (drift 20x too small, mask
inert, entropy up); `both` adds displacement (entropy fine only while the addition is tiny).
The donor did NEITHER - its rotation was **norm-preserving**, so only the direction moved.

The previous agent measured the fact that makes this portable: one real chassis surrogate step
moves the Beta mean with cosine **0.9988** (min 0.9856 over 12 agents) to
`u_cred = grad_mean[A*log pi(a|s)]`. The chassis's realized update direction in mean-space IS
`u_cred`. So the donor's mechanism applies to the chassis's OWN update: take the chassis's
per-sample mean-space gradient `g_ch`, form `g_rot = rotate_to_cos(g_ch, u_an, 0.95)` with
`||g_rot|| == ||g_ch||`, and add a LINEAR corrector delivering exactly `g_rot - g_ch`.

Why this should pass the gates the others failed: the update's magnitude is unchanged by
construction, so it injects no extra drift (mask and `approx_kl` should be nearly identical to
the surrogate's), and it survives the saturated actor clip precisely BECAUSE the norm is
preserved. It delivers the full ~18-degree rotation instead of 0.47%.
Falsifiable and pre-registered: realized `cos(displacement, g_ch)` must come out at ~0.95 -
if it reads 0.90 (=2c-1) the corrector is double-counted, and if ~1.0 the sign is inverted.

### 8.6 FALSIFIED, and the reason is the most useful thing this session found

`ppo_continuous_action_entppo_jacteach_v2_rotate.py`, `--improve rotate`. The rotation was
delivered **perfectly** and the design still failed at its own default cosine:

- `rot_cos_check` == the requested cosine exactly at every rung; `rot_norm_check` == **1.00000**
  everywhere; `rot_delta_ratio` == sqrt(2(1-cos)) to 5 decimals. Identity control bit-exact
  (36 shared tags, 720 and 784 points, zero differing). The sign was found EMPIRICALLY as the
  task required, and the wrong sign is a genuine trap: it yields `2*g_ch - g_rot`, whose cosine
  (2-c)/sqrt(5-4c) = 0.9585 at c=0.95 *looks* almost right and is separated from the correct
  branch only by the norm (1.0954 vs 1.00000).
- The unclipped actor gradient was even SHORTENED (`rot_pgrad_ratio` 0.624 at cos 0.95) and the
  realized displacement never grew (`rot_norm_ratio` <= 1 at every rung).
- Yet at cos 0.95: `tr_mask_frac` **20x** the surrogate's, `approx_kl` **17x**, entropy slope
  **36x**, `ratio_max` 69.6 vs 5.55, and the 0.15 EMERGENCY breaker fired in 3 of 8 iterations
  (6 of 8 at cos 0.85, which forfeited half its epoch budget).

**Two corrections to my own reasoning, both measured.**

1. The 0.9988 cosine I built the design on is a **gradient-space** quantity, not a realized
   displacement. The literal per-row instrument I specified has NO SIGNAL: `rot_adam_cos` reads
   0.021 for the UN-ROTATED control, because one optimizer step moves each sample's mean mostly
   through parameters SHARED with every other sample. The instrument was falsified, not the
   mechanism - correctly reported instead of tuned around.
2. The well-posed matched-arm reading, `cos(dm_rotated, dm_surrogate)`, shows the turn is
   **AMPLIFIED 2.4x-4.1x**: 8.1 -> 32.9 deg, **18.2 -> 62.3 deg**, 31.8 -> 74.9 deg.

**THE RESULT.** *Per-sample norm preservation in action space is NOT dose preservation for a
shared-parameter policy.* Rotating every sample's target independently makes the per-sample
targets mutually inconsistent; the network cannot satisfy them, so the parameter-space update
is dominated by their DISAGREEMENT, which is pure added drift. The donor never met this because
its student regressed onto the teacher through a per-dim CLIPPED KL (tau), which absorbs exactly
that disagreement; a linear corrector has nothing to absorb it. This also explains §8.1
retrospectively: every failure in this section changed the realized dose while appearing to
hold something else fixed.

### 8.7 The safe band the diagnosis predicts - LAUNCHED (jobs 3856, 3857)

If the failure is amplified disagreement rather than the mechanism, the gates must be recoverable
at much smaller angles while keeping strength far above §8.4's 0.0047. Measured (262k, champion
config, `_x sur_` = multiple of the surrogate):

| jac_cos | d(entropy) | x | tr_mask_frac | x | approx_kl | x | delivered strength | realized turn | breaker |
|---|---|---|---|---|---|---|---|---|---|
| 1.0 (=chassis) | -0.0381 | 1.0 | 0.00622 | 1.0 | 0.00990 | 1.0 | 0 | 0 deg | 0 |
| 0.9995 | -0.0668 | 1.8 | 0.00730 | 1.2 | 0.01121 | 1.1 | 0.0270 | 7.4 deg | 0 |
| **0.999** | -0.0857 | 2.3 | **0.00722** | **1.2** | **0.01079** | **1.1** | **0.0382** | 11.0 deg | **0** |
| 0.995 | -0.1705 | 4.5 | 0.01263 | 2.0 | 0.01446 | 1.5 | 0.0855 | 25.4 deg | 0 |

The prediction holds: the mask and KL gates are intact (1.1-1.2x) at cos 0.999 with **8.1x** the
delivered mechanism strength of the additive arm now running. Entropy slope is the only elevated
gate (2.3x) and is small in absolute terms. Launched cos 0.999 as primary and 0.995 to bracket
the upper edge; the same pre-registered criterion as §8.3 applies (@1M >= 3400, @2M >= 6000
against the chassis's 3041 / 5465, gates holding all run).

### 8.8 RESULT - the additive arm at 8M: the mechanism buys RATE, not CEILING (twice measured)

`entjac_aux_c011_8M` completed. HalfCheetah-v4, seed 1, identical config to the chassis parent:

| | @500k | @1M | @2M | @3M | @4M | @6M | @8M | last-20 |
|---|---|---|---|---|---|---|---|---|
| chassis `entppo_v4_samplemask` | 1238 | 3041 | 5465 | 7281 | **8890** | **9873** | 10312 | **10362 +-92** |
| `entjac_aux_c011` (this arm) | 1250 | **3223** | **6089** | **8087** | 8792 | 9775 | 10106 | 10296 +-147 |
| delta | +1% | **+6.0%** | **+11.4%** | **+11.1%** | -1.1% | -1.0% | -2.0% | -0.6% |

Every gate held for the full 8M - `losses/entropy` -6.881 vs the chassis's -6.877 (i.e.
INDISTINGUISHABLE, and note §8.1's `distill` failed exactly here), `tr_mask_frac` 0.0039 vs
0.0030, `approx_kl` 0.0078 vs 0.0055, EV 0.962 vs 0.964. The mechanism stayed alive and correct
for 244 iterations: `jac_rot` 0.9500, `jac_align` -0.028 (orthogonal to the credit direction
from start to finish, so it never collapsed into rediscovering the policy gradient), `jac_r2`
0.985, delivered `aux_pg_grad_ratio` 0.0045.

**Verdict against the §8.3 pre-registration: PARTIAL PASS, and the endpoint is a TIE.**
@2M 6089 clears the 6000 bar; @1M 3223 misses the 3400 bar. The 8M endpoint (10296 vs 10362) is
inside both the +-CI95 overlap and the lineage's own 423-point noise band, so it is a tie, NOT a
win and NOT a loss. Reported as a tie rather than as "-66".

**The finding that generalizes.** The analytic-direction teacher is now measured on TWO
unrelated chassis and it has the SAME signature on both: a large early/mid-training
acceleration that decays to nothing by 8M.

| chassis | @1M | @2M | @8M |
|---|---|---|---|
| advcond_v6 (8718 -> 8915) | +14% | +7% | +2.3% |
| entppo_v4_samplemask (10362 -> 10296) | +6% | **+11.4%** | -0.6% |

This is a rate improvement, not a ceiling improvement - and it independently confirms the
prediction this document made in §6.5 item 1 ("it buys RATE, not CEILING") on a chassis that
had never been tested when that was written. The crossover on the strong chassis is ~3.5M.

**What that implies about the ceiling.** Two chassis with completely different step-control
machinery converge to their own asymptote regardless of how good the improvement DIRECTION is.
So on this benchmark the 8M ceiling is not set by the quality of the update direction. The
remaining open question - whether more delivered mechanism strength converts rate into ceiling -
is exactly what the §8.7 rotation arms test at 8.1x and 18x this arm's delivered strength.

### 8.9 DOSE-RESPONSE IS REAL - delivered mechanism strength maps monotonically to early rate

Matched steps, HalfCheetah-v4 seed 1, all four arms on the SAME chassis and config. "Delivered"
is the measured ratio of the mechanism's actor-gradient contribution to the chassis's own:

| arm | delivered | @500k | @1M | @1.5M | vs chassis @1.5M |
|---|---|---|---|---|---|
| chassis (`--improve surrogate`) | 0 | 1238 | 3041 | 4270 | - |
| `aux_c011` (additive, §8.4) | 0.0045 | 1250 | 3223 | 4781 | **+12%** |
| `rot_c0p999` (rotation, §8.7) | 0.045 | 1480 | 3498 | 4977 | **+17%** |
| `rot_c0p995` (rotation, §8.7) | 0.100 | **1757** | **4353** | **6091** | **+43%** |

Two things this establishes that no single arm could:

1. **The mechanism is not a placebo and the §8.4 tie was a DOSE artifact, not a mechanism
   failure.** Early rate rises monotonically with delivered strength across a 22x range
   (0.0045 -> 0.100), and the §8.6 diagnosis is what made the strong doses reachable at all:
   the failure at cos 0.95 was amplified per-sample disagreement, so backing the ANGLE off
   while keeping the norm-preserving form recovers the gates and still delivers 22x the
   additive arm's strength.
2. **jacteach's original signature reproduces on a chassis 1600 points stronger.**
   `rot_c0p995` is **+43% @1M** (4353 vs 3041), against the +44% @1M the mechanism showed
   versus this chassis on its own donor chassis. The early effect is a property of the
   MECHANISM, not of the weak advcond early phase - which was the live alternative hypothesis
   in §8.3's FAIL clause, and it is now ruled out.

Pre-registered §8.3 criterion (@1M >= 3400, @2M >= 6000): `rot_c0p995` clears the @1M bar by
28% and `rot_c0p999` clears it too. Gates at 1.7M: breaker trips 0/51 (c0.999) and 1/51
(c0.995); `jac_align` -0.008 / -0.028 (still orthogonal, mechanism not collapsing into the
policy gradient); `jac_r2` 0.975 / 0.977; EV 0.847 / 0.875; `rot_delta_ratio` exactly
sqrt(2(1-cos)) = 0.0447 / 0.1000.

OPEN, and it is the only question left: whether this rate advantage survives to 8M or decays
at ~3.5M as the additive arm's did. Entropy is the watch item - c0.995 is at -5.37 by 1.7M,
sharpening faster than the chassis, and premature entropy collapse is the mechanism by which a
rate advantage would convert into a LOWER ceiling.

## 9. THE CAUSAL RESULT - direction quality trades against exploration - 2026-08-26

A 22x dose sweep on ONE chassis, all four arms identical except delivered mechanism strength,
traces a monotone rate/ceiling frontier:

| arm | delivered | @1M | @2M | @3M | @4M | @6M | 8M |
|---|---|---|---|---|---|---|---|
| chassis | 0 | 3041 | 5465 | 7281 | **8890** | **9873** | **10362 +-92** |
| `aux_c011` | 0.0045 | 3223 | 6089 | **8087** | 8792 | 9775 | 10296 +-147 |
| `rot_c0p999` | 0.045 | 3498 | 6289 | 8053 | 9085 | ~9529 @6.9M | (below) |
| `rot_c0p995` | 0.100 | **4353** | **7360** | 8468 | 8855 | ~9219 @7.1M | (below) |

Early rate rises monotonically with dose (+43% @1M at the top rung); the asymptote FALLS
monotonically with dose. And the cause is measured, not inferred - **entropy at matched steps is
monotone in dose at EVERY step**, while explained variance stays healthy (0.93-0.98) in all four
arms, so it is an exploration loss and not a critic failure:

| entropy | @1M | @2M | @3M | @4M | @6M |
|---|---|---|---|---|---|
| chassis (0) | -2.15 | -4.10 | -5.13 | -6.26 | **-6.81** |
| aux (0.0045) | -2.48 | -4.57 | -5.74 | -6.53 | **-6.75** |
| rot (0.045) | -2.74 | -5.08 | -6.60 | -7.83 | -8.47 |
| rot (0.100) | -3.43 | -6.25 | -7.74 | -8.89 | **-9.94** |

**THE MECHANISM.** A better improvement direction makes every step more self-consistent, so the
policy sharpens faster; on HalfCheetah at 8M the asymptote is set by how long exploration
survives, so a better direction is spent buying rate and is CHARGED to the ceiling. Net ~0.

**This retroactively explains this whole family.** Eight arms of target-quality work on the
advcond chassis all landed in 8812-8882 (§6.7), and the two best in-charter chassis on the disk
land at 10362/10351 by two unrelated mechanisms. The family kept improving the DIRECTION, which
is a rate lever, and then measured it against a CEILING benchmark. §6.5 item 1 predicted exactly
this ("it buys RATE, not CEILING") and it is now confirmed at 4 doses on 2 chassis.

**Reframing that is honestly available:** at a 2M budget this mechanism is a large win
(7360 vs 5465, **+35%**) and at 1M a larger one (+43%). It is a sample-efficiency mechanism.
The 8M benchmark is the wrong instrument for it.

### 9.1 The fix the diagnosis dictates - LAUNCHED (jobs 3865, 3866)

`ppo_continuous_action_entppo_jacteach_v3_anneal.py`, `--jac-anneal-frac`: fold the rotation
linearly to the IDENTITY over the first `frac` of training. Spend the dose while entropy is
still high enough to afford it, then BE the chassis for the second half, where the chassis's own
entropy schedule sets the asymptote. The handback point is not a guess - the return crossover is
measured at ~4M of 8M, hence `frac=0.5`.

Verified before launch:
- `--jac-anneal-frac 0` is **BIT-EXACT** to v2 across 55 tags.
- Anneal active: `debug/jac_rot` ramps 0.9974 -> 0.9975 -> 0.9988 -> **1.0** and holds;
  `rot_delta_ratio` 0.0724 -> 0.0707 -> 0.05 -> **0.0**. The dose provably reaches zero.
- Entropy delta **-0.025** vs the surrogate's -0.038 and the un-annealed 0.995's **-0.171**:
  the arm sharpens LESS than the chassis, i.e. the exploration charge is not merely reduced,
  it is removed at this horizon.

Arms: `(cos 0.995, frac 0.5)` primary, and `(cos 0.99, frac 0.5)` - a dose the un-annealed
gates FORBADE (mask 2.7x, entropy slope 6.1x), admissible now precisely because it is only ever
applied early. Pre-registered PASS: 8M last-20 above 10362 + 423, with @2M retaining most of the
+35%. Pre-registered FAIL: 8M inside the 10362 +- 423 band => the rate/ceiling trade cannot be
broken by scheduling, and the ceiling is set by something neither direction nor its schedule
touches.

## 10. FINAL RESULT - six arms, five doses, one chassis, HalfCheetah-v4 seed 1 - 2026-08-26

| arm | delivered dose | @1M | @2M | @4M | @6M | **8M last-20** | H@8M | EV@8M |
|---|---|---|---|---|---|---|---|---|
| chassis `entppo_v4_samplemask` | 0 | 3041 | 5465 | **8890** | **9873** | **10362 +-92** | -6.87 | 0.967 |
| `aux_c011` | 0.0045 | 3223 | 6089 | 8792 | 9775 | **10296 +-147** | -6.88 | 0.954 |
| `rot_c0p999` | 0.045 | 3498 | 6289 | **9085** | 9685 | 9664 +-561 | -8.49 | 0.988 |
| `rot_c0p995` | 0.100 | 4353 | 7360 | 8855 | 9285 | 9457 +-215 | -11.31 | 0.920 |
| `anneal_c0p995` | 0.100 -> 0 | 4133 | 7070 | 7948 | 9385 | 9808 +-161 | -7.57 | 0.953 |
| `anneal_c0p99` | 0.141 -> 0 | **4542** | **7703** | 7765 | 9374 | 9851 +-98 | -7.08 | 0.959 |

### 10.1 What was achieved

- **Sample efficiency: a large, dose-ordered win.** Best @1M **4542 vs 3041 = +49%**; best @2M
  **7703 vs 5465 = +41%**. Monotone in delivered dose across a 31x range, reproduced on two
  unrelated chassis (the donor showed +44% @1M against this same chassis's curve).
- **8M ceiling: a TIE, not a win.** 10296 +-147 vs 10362 +-92 - overlapping CIs, and inside the
  lineage's own 423-point spread. Reported as a tie.

### 10.2 The causal claim, and the experiment that tests it

Claim: a better improvement direction is spent buying RATE and is CHARGED to the CEILING through
premature entropy collapse.

Evidence 1 (correlational, 4 doses): entropy at matched steps is monotone in dose at EVERY step,
while EV stays 0.92-0.99 in every arm - an exploration loss, not a critic failure.

Evidence 2 (**causal, dose held fixed**): the annealed arms apply the SAME early dose as their
un-annealed twins and differ only in whether the dose is handed back.

| early dose | schedule | H@8M | 8M last-20 |
|---|---|---|---|
| 0.100 | held | **-11.31** | **9457** |
| 0.100 | annealed to 0 by 4M | **-7.57** | **9808** |

Recovering entropy recovers ceiling, at matched early dose. That is the mechanism confirmed by
intervention rather than by correlation. The recovery is **partial**: -7.57 never returns to the
chassis's -6.87 and 9808 never returns to 10362, so **the exploration charge is real but only
partly refundable** - what is spent early is not fully recoverable later.

### 10.3 §9.1 pre-registration: FAIL, as written

PASS required 8M > 10362 + 423. Measured 9808 / 9851. So: **the rate/ceiling trade cannot be
broken by scheduling the dose.** The 8M ceiling on this benchmark is set by something that
neither the direction's quality nor its schedule touches.

### 10.4 What this settles for the family

Eight arms of target-quality work on advcond all landed in 8812-8882; the two best in-charter
chassis on the disk land at 10362/10351 by unrelated mechanisms; and a 31x dose sweep of the
best direction mechanism this family has moves the 8M number by 0. The family has been
optimizing a RATE lever against a CEILING benchmark. §6.5 item 1 called this and it is now
established by intervention.

**The one honest reframing:** as a sample-efficiency method this is a strong result (+41% @2M,
+49% @1M, monotone in dose, gates intact, mechanism verified alive for 244 iterations with
`jac_align` orthogonal throughout). As an 8M-score method it is spent.

**Where the ceiling actually lives, if anyone continues:** not in the actor's direction. The
only lever in this whole audit that moved an 8M number outside a noise band was a critic/value
change on a DIFFERENT chassis (+1355), and the two best in-charter arms differ from each other
in CRITIC GEOMETRY, not in actor mechanism. That is the next place to look, and this document
now contains the measurement showing why it is not the actor.

## 11. SYSTEM BOTTLENECK ANALYSIS of the corrected incumbent (10362) - 2026-08-26

Ranked by MEASURED effect size on this chassis or its direct family, not by novelty.

| # | candidate bottleneck | status | evidence |
|---|---|---|---|
| 1 | **Value readout capacity** | **OPEN, largest known effect** | `edmvalue_e2e` +1355 over its ancestor at IDENTICAL core config on this chassis family, effect GROWS with horizon (+5% @2M, +21% @4M, +21% @6M, +16% @8M), actor unchanged |
| 2 | Actor improvement direction | **CLOSED by intervention** | 31x dose sweep, +49% @1M decaying to 0 by 4M at every dose; annealing recovers ceiling only partly (§10.2) |
| 3 | Trust-region enforcement | closed within family | mask > abort (10362 vs 9756@7M); alpha interior optimum at 0.3 (3942/4808/4467 @2M); beta +2% at +-1% |
| 4 | Critic GEOMETRY (vs capacity) | closed as a lever | `v24_beta_v162critic_mtp_v1` = 10351 vs 10362 - a tie by a different critic geometry, so geometry alone is not the binder |
| 5 | Late exploration schedule | open but thin | H@8M -6.87 here, yet `edmvalue` scored 9810 at H -10.69 on its chassis, so entropy is not universally binding; a rising-alpha schedule is coefficient work, which §2 excludes |
| 6 | Data / throughput | not a mechanism | §3: throughput is a diagnostic, not a goal |

**Decision.** Pursue #1 as a single clean lever. It is the only candidate with a large,
config-clean, same-family measurement, and it is the only one whose effect signature GROWS with
horizon - the exact complement of the mechanism §10 exhausted. #2-#4 are closed by measurement,
#5 is excluded as coefficient search, #6 is not a mechanism.

**Instrument warning carried forward.** Explained variance is disqualified as the bottleneck
instrument here: EV@8M 0.970 (ancestor, 8455) vs 0.952 (deep-value, 9810). The lower-EV arm wins
by 16%. Any future claim of the form "the critic is fine, EV is high" is unsupported on this
benchmark, INCLUDING the version of that claim I made earlier in §9 about my own arms.

**Not queued, and why.** Combining the deep readout with §8.4's auxiliary (which is free sample
efficiency at 0.0045 dose) is deliberately NOT queued yet: it would confound a first clean test
of lever #1 with a mechanism already measured to be neutral at 8M. If #1 wins, that combination
becomes the warranted follow-up.

## 12. VALUE-READOUT PORT: LARGE CLEAN REGRESSION, and the reason is INTERACTION - 2026-08-27

Job 3881, `ppo_continuous_action_entppo_deepvalue_v1.py --deep-value`, cancelled at 4.0M/8M on
overwhelming evidence. Bottleneck #1 from §11 is CLOSED as a lever on this chassis.

| checkpoint | incumbent | deep-value | delta |
|---|---|---|---|
| @1M | 3041 | 1884 | -38% |
| @2M | 5465 | 3746 | -31% |
| @3M | 7281 | 4726 | -35% |
| @4M | 8890 | 5395 | **-39%** |

-3495 at 4M is 8.7x the ~400 noise floor, and the deficit WIDENS with horizon. P1 failed in
substance at all four checkpoints; P2 failed outright (no advantage exists to be
time-extensive). Cancelled rather than paying 30 more min to trip P1's formal
two-consecutive-checkpoint clause, which would have changed nothing and blocked foreign job 3882.

### 12.1 The mechanism of failure - P3 called it, and named the right suspect

| | approx_kl | tr_mask_frac | value_loss | EV |
|---|---|---|---|---|
| chassis | 0.0389 | 0.0749 | 8.1814 | 0.928 |
| deep-value | 0.0514 (+32%) | 0.1051 (+40%) | 8.1851 (equal) | 0.894 |

**`value_loss` is IDENTICAL.** The deep readout is neither a better nor a worse critic - it does
not move the axis it was built to move. The whole regression is ACTOR-SIDE COLLATERAL: under
`share_backbone=True` the deep head's value gradient floods the shared trunk, policy drift rises
32%, and entppo's sample-mask therefore DISCARDS 40% MORE OF EVERY BATCH. The actor trains on
materially less data. The 262k port-time telemetry saw this coming (gates diverged 2x, disjoint
over 3 seeds); I read it as a curiosity to note rather than a stop signal.

### 12.2 What this actually teaches - a recombination hazard, not a value-readout verdict

The donor measured +1355 from this readout WITH `share_backbone=True`. It still failed here. The
difference is that the donor's chassis had **no trust-region mask to poison**. Two mechanisms,
each measured-good alone, compose destructively through a channel neither one owns: the shared
trunk's clip group. This is the general hazard of crossbar-mining and it is the strongest reason
to stop doing it.

Note the honest scope: this does NOT show the value readout is worthless. It shows the readout
cannot be delivered through a shared trunk that an entropy/trust-region-gated actor depends on.
A `--no-share-backbone` arm would isolate it - and is deliberately NOT queued, because at that
point the experiment is two levers and a chassis change to chase a mechanism whose own
`value_loss` says it changed nothing.

### 12.3 RECORDED DRIFT - 2026-08-27 (second entry; see also §2)

Prompted by direct challenge. Drift from §2 hard exclusions: **zero** - V(s) only, no sim probing,
no Q(s,a), no twin critics/targets/replay/ensembles, no outcome conditioning, HalfCheetah only.
That discipline held and is why the 11015/10371/9224 runs stay disqualified as incumbents.

Drift from §1, what this family IS: **total.** §1 requires no PPO, no importance ratio, no clipped
surrogate, no advantage-weighted PG in a backward, with the actor teaching itself via privileged
`delta_t` conditioning and supervised regression onto detached targets. Job 3881 ran the PPO
sample-mask chassis (clipped surrogate, ratio, advantage-weighted PG, `approx_kl` load-bearing in
the mask) and contained NO self-distillation whatsoever. Path: §7 (frontier wrongly scoped) -> §8
(OPSD demoted from algorithm to auxiliary term on PPO) -> §10 (auxiliary exhausted: rate, no
ceiling, 31x dose) -> §11 (bottleneck points somewhere OPSD does not reach) -> §12 (pure critic
architecture). This document's title no longer describes §11-§12.

The more important drift is against `AGENTS.md`, which is the real mandate and asks for
"significant architectural and algorithmic innovation". §12 was a **recombination of two
already-measured in-repo mechanisms**, chosen because recombination has a better measured hit
rate than invention. It was well-controlled (bit-exact control, params -32,576, pre-registered
P1-P4) and it still taught mostly about plumbing. Optimizing for the scoreboard selected for
crossbar-mining, and §12.2 is the bill.

### 12.4 AUTOCULL CALIBRATED AND ARMED - 2026-08-27

`--enforce health` (nan/stall/dead/plateau) would NOT have saved the §12 run: audit reports
`never` for it, with `behind` fired but unenforced from 4M. Catching runs like it requires
`--enforce all`. Validated that before arming it, using `--audit` (the winner-killer test):

| run | 8M result | audit verdict |
|---|---|---|
| `entppo_v4_samplemask_a03_mask` (incumbent) | 10362 | never - survived 28 ckpts |
| `entjac_aux_c011_8M` | 10296 | never |
| `entjac_anneal_c0p995_a0p5_8M` | 9808 | never |
| `entjac_anneal_c0p99_a0p5_8M` | 9851 | never |
| `entjac_rot_c0p995_8M` | 9457 | never |
| `entjac_deepvalue_v1_8M` (§12 failure) | -39% @4M | **behind from 4M** |

5 known-good runs spared, 1 known-bad caught. The `anneal_c0p99` case is the load-bearing one:
it COLLAPSED mid-training (7723 @2M -> 6327 @3M -> 7106 @4M) and still finished 9851, and the
judge still says `never` - so the thresholds tolerate collapse-and-recover, which is the exact
failure mode that would make an auto-culler a winner-killer on exploration methods.

Armed as a supervised process: `autocull.py --ref entppo_v4_samplemask_a03_mask --enforce all
--yes --watch 600`. The `--ref` is PINNED to the incumbent rather than left on `ref=auto`, because
`--ref-scan` eviction means launching several fresh arms can push the historical champion out of
the auto candidate set and silently soften the bar.

## 13. STATE-CORRELATED NOISE - the pivot back to invention - 2026-08-27

`cleanrl/beta_policy/ppo_continuous_action_entppo_corrnoise_v1.py`. Chosen because §9/§10 established BY
INTERVENTION that the 8M ceiling here is exploration-governed, and because `AGENTS.md` names
state-correlated noise as a target that this campaign had never touched. Not a crossbar
recombination - this mechanism does not exist anywhere in the repo.

  z_t = rho*z_{t-1} + sqrt(1-rho^2)*xi_t,   a_t = mu(s_t) + L(s_t) z_t,   L = diag(sigma) + U V^T

Conditioning on the RECORDED z_{t-1} - the agent's own internal state, not privileged env
information - makes a_t | s_t, z_{t-1} exactly Gaussian, so PPO's ratio stays exact with no
surrogate approximation and entropy stays closed-form, leaving the sample-mask machinery intact.
Rationale: white per-step noise largely self-cancels in HalfCheetah, so entropy buys jitter rather
than displacement; correlated noise spends the SAME nats on sustained coordinated limb pushes.

### 13.1 The headwind I predicted DOES NOT EXIST - measured, and I was wrong

I argued that conditional entropy = const + log|det L| + 0.5*sum log(1-rho_i^2) is decreasing in
|rho| and would pull rho to 0. That is true of the entropy TERM, but `ent_coef = 0` on this
chassis, so entropy is not in the loss. The channel is the alpha*KL proximal term, and there the
-H(new) half is CANCELLED to ~1% by the cross-entropy half of the same term:

| rho | \|KL-ent half\| | \|KL-cross half\| | \|NET KL\| | \|PG\| | ratio |
|---|---|---|---|---|---|
| 0.4 | 0.3488 | 0.3429 | 0.0068 | 0.0198 | 2.9x |
| 0.7 | 0.6083 | 0.6034 | 0.0066 | 0.0337 | 5.1x |
| 0.8 | 0.6952 | 0.6895 | 0.0089 | 0.0312 | 3.5x |

At zero drift the net KL gradient on raw_rho is identically 0 to fp32. KL is a PROXIMAL ANCHOR
that resists CHANGING rho; it has no preference for rho=0. Reading only the -H half over-states
the force ~100x. **Consequence, and it improves the experiment:** there is no restoring force to
blame, so if learned rho stays near 0 that is a REAL NEGATIVE about correlated exploration rather
than an artifact to argue away.

### 13.2 Verification that actually earns trust

- Bit-exact control 0/720 differing points in THREE configurations, `charts/episodic_return`
  among the compared tags, so trajectories are identical, not just summary stats.
- Density correctness: covariance vs analytic (1-rho^2)LL^T max elementwise err 1.37e-3 = 1.31 MC
  sigma; E[-log p] matches closed-form entropy to 0.46 MC sigma; log_prob vs an INDEPENDENT
  `torch.distributions.MultivariateNormal` max diff 5.07e-6; rollout vs replay log_prob differ by
  EXACTLY 0, so the ratio replays the density that generated the sample.
- Realized action-space lag-1 autocorrelation equals rho to 3-4 digits (0.7992 at rho 0.8);
  white-noise control 0.0014; rho forced to 0 gives 0.0019, so the low-rank part alone adds no
  temporal structure.
- Reset correctness: 0/592 boundary steps carry the stale latent, 23408/23408 non-boundary steps
  carry it; post-reset latent passes KS vs N(0,1) (p 0.36-0.96). Fresh-draw, not zeros, chosen for
  stationarity (latent variance 0.982 at boundaries vs 1.005 elsewhere).

### 13.3 Two known risks, carried forward deliberately

1. **Mask/clip inflation - the §12 killer.** At rho 0.8: `tr_mask_frac` 0.0342 vs control 0.0062,
   pre-clip `actor_grad_norm` 5.9 vs 1.0 against `actor_grad_clip` 0.25. Distinct from §12 in
   KIND - there the inflation was collateral from a flooded shared trunk, here it is the intended
   mechanism (a genuinely noisier action process) - but the failure surface is the same one. 96.6%
   of the batch survives, epochs 10/10, no breaker trip, `ratio_max` LOWER than control.
2. **EV is worse early** (0.300 at rho 0.8 vs 0.851 control at 262k). Per §11/§12 EV is NOT the
   gate on this benchmark, so this is logged, not treated as fatal.

### 13.4 The Beta -> Gaussian CONFOUND, and why two arms

Enabling corr-noise forces `actor_dist` Gaussian, while the chassis default is Beta. So
corr-vs-chassis conflates correlation with a distribution-family switch. No matched Gaussian-actor
8M run exists on this chassis (the 171 "gauss" run dirs are all HL-GAUSS CRITIC, unrelated). The
control therefore has to occupy a slot:

- **job 3900** `--corr-rho-fixed 0.8` - full dose, the measured-clean setting (0.9 acceptable,
  0.95 is the numerical ceiling where the step is ~70x renormalized and clip-dominated).
- **job 3901** `--no-corr-noise --actor-dist gaussian` - matched control isolating the family switch.

Both `maxParallelRuns=2`, `--max-attempts 1`, `--time-limit 6h`, seed 1, champion config. Learned
rho is deliberately NOT one of the two arms: it reached only ~1.6e-2 at 262k (3% of horizon), so
it would most likely return an uninformative null. Full-dose-first answers the SCIENCE question
(does temporal correlation help at all); learned rho is the follow-up only if it does.

### 13.5 RESULT - both arms killed, and the CONTROL is the finding - 2026-08-27

Autocull (§12.4) cancelled both arms. Both verdicts were correct and together they saved ~7 h of
GPU, which is exactly what the supervisor is for.

| job | arm | verdict | evidence |
|---|---|---|---|
| 3900 | `--corr-rho-fixed 0.8` | **DEAD @896k** | return -74 (origin -227), gain 153 vs the reference's 1102 = 0.11x. Chassis is ~3041 by 1M. |
| 3901 | `--no-corr-noise --actor-dist gaussian` | **BEHIND @4M** | @3M 3607/6052 = 0.61x, @4M 4308/7954 = 0.55x |

**The control is the headline. The Beta -> Gaussian family switch alone costs ~45%.** The chassis's
Beta actor is load-bearing, and nobody had ever measured that on this line. 171 run dirs matching
"gauss" were all HL-GAUSS CRITIC; the actor family had never been ablated.

**Therefore corrnoise_v1 never got a fair test.** Enabling `--corr-noise` forces `actor_dist`
Gaussian, so the mechanism started in a 45% hole before a single nat of correlation was applied.
P4 was written as a confound guard; it turns out to be fatal. The v1 file is retired, NOT because
correlated exploration is refuted, but because it was delivered on a crippled carrier.

**rho = 0.8 is separately destructive.** 3901 was learning fine early (0.49x ref at 896k, slope
+1335%/1M) while 3900 was flat at NEGATIVE return, so ~0.8 correlation hurts far beyond the family
switch. The mechanism is legible: conditional entropy falls by -0.5*sum log(1-rho^2) = 3.07 nats
at rho 0.8, so the policy becomes much more DETERMINISTIC per step while wandering slowly in a
persistent random direction. That is not exploration, it is executing a randomly-biased policy for
long stretches - it wrecks credit assignment, and it showed up in advance as tr_mask_frac 5.5x and
pre-clip actor_grad_norm 5.9 vs a 0.25 clip (§13.3 risk 1, which fired).

**Prediction verdicts.** P1 FAIL (both arms culled far below every threshold). P2 N/A - rho was
pinned, not learned. P3 untested. P4 **FAIL and decisively so**: the corr arm did not beat the
matched Gaussian control, it lost to it catastrophically, and the file's own criterion says
abandon in favour of asking what the family switch is worth. Answered: the switch is worth -45%.

### 13.6 What this dictates for v2 - correlation must ride ON Beta

The fix is not a smaller rho on a Gaussian actor; it is to stop switching the family. Route that
keeps the chassis's Beta marginal EXACTLY while still injecting temporal correlation - a Gaussian
copula in the Beta's own quantile space:

  z_t = rho*z_{t-1} + sqrt(1-rho^2)*xi_t,   u_t = Phi(z_t),   a_t = F_Beta^{-1}(u_t | s_t)

Conditioned on the recorded z_{t-1}, z_t is Gaussian, so u_t has a known conditional density and
a_t's conditional density follows exactly by change of variables (the Phi and F^{-1} Jacobians are
both available in closed form). Properties that make this the right design:
  * the MARGINAL action distribution is EXACTLY the chassis's Beta, so there is no family-switch
    penalty to pay and no 45% hole to climb out of;
  * at rho = 0 it reduces EXACTLY to the chassis, so the bit-exact control gate still applies;
  * rho becomes the ONLY lever, which is what a clean test of correlated exploration requires;
  * per-step conditional entropy still contracts with |rho|, so rho must be SMALL - 0.8 is
    already known destructive. Test a low fixed dose and a learned rho, not a high pin.

## 14. BETACORR - coherent exploration that is ADDITIVE, not traded - 2026-08-27

`cleanrl/beta_policy/ppo_continuous_action_entppo_betacorr_v1.py`. Jobs 3903 (g=0.25) / 3904 (g=0.5), both
rho=0.9, maxParallelRuns=2, --max-attempts 1, --time-limit 8h, seed 1, champion config.

  z_t = rho*z_{t-1} + sqrt(1-rho^2)*xi_t,   a_t ~ Beta( params(s_t) shifted by g*z_t )

**The design pivot: condition on z_t, not z_{t-1}.** v1 stored the PREVIOUS latent, which made the
conditional covariance (1-rho^2)LL^T and contracted per-step conditional entropy by 3.07 nats at
rho 0.8 - the policy went nearly deterministic per step while drifting in a persistent random
direction, which is a randomly-biased policy rather than exploration. Storing and conditioning on
the latent ACTUALLY USED makes pi(a|s,z_t) a FULL-entropy Beta. Exploration becomes strictly
ADDITIVE - undiminished per-step Beta stochasticity PLUS a coherent slow drift of the mode -
instead of v1's trade of per-step entropy for temporal coherence. And because the conditional stays
Beta, there is no family switch and none of §13.5's -45% penalty.

### 14.1 The claim that coherence is FREE - measured, and it holds

Every rho sits INSIDE the control's own fragility band, on the exact axis that killed §12 and §13:

| arm | tr_mask_frac vs ctl | pre-clip actor_grad_norm vs ctl | ratio_max | epochs | breaker | NaN |
|---|---|---|---|---|---|---|
| rho 0 / 0.5 / 0.9 / 0.95 | 0.84-1.03x | 0.82-1.06x | LOWER than ctl in all 6 arms | 10/10 | 0 | none |

`losses/entropy` at iteration 1 is flat in rho to +-0.02 nats (-0.937 / -0.930 / -0.951 / -0.953).
Largest safe rho = 0.95, the largest tested. Contrast v1, where rho 0.8 gave tr_mask_frac 5.5x and
actor_grad_norm 5.9 against a 0.25 clip. Cost: 22.96 vs 23.35 ms per minibatch, +6 MiB VRAM.

### 14.2 Verification

- Bit-exact control 0/720 differing in BOTH `--no-beta-corr` and `--corr-gain-fixed 0`,
  `charts/episodic_return` included. Required a dedicated `torch.Generator` for xi so
  `Beta.sample()`'s RNG stream is untouched, and excluding the frozen `raw_gain` from the actor
  clip group.
- Conditional is EXACTLY the shifted Beta: KS p 0.235-0.978 on 1e6 samples, while the UNSHIFTED
  Beta is rejected at p=0. `log_prob` vs `scipy.stats.beta` max abs diff 1.088e-6. Rollout-vs-replay
  `log_prob` difference exactly 0.0.

### 14.3 Three honest negatives, recorded before the result

1. **The copula route was abandoned** (and rightly): torch 2.12 exposes no betainc/betaincinv and
   `Beta.cdf/.icdf` raise NotImplementedError, but the decisive argument is structural - the
   copula's exact conditional KL has no closed form, so the chassis's load-bearing exact all-action
   KL would have become a hybrid. Not worth an exact marginal.
2. **rho gets NO gradient, structurally** - pi(a|s,z_t) does not contain rho. Making it learnable
   requires declaring z_t part of the action, which puts log N(z_t; rho z_prev, 1-rho^2) back into
   the log-probs and reinstates v1's contraction (-4.98 nats at rho 0.9). Correctly rejected. rho
   is a structural constant; g is the dose. The three-way decomposition now points at raw_gain:
   KL halves cancel to ~10% (v1: ~1%) and PG out-weighs the net trust-region force 5.1x (8.7x at
   the true kl_coef).
3. **g DOES cost conditional entropy** - not rho, but g: 0.52 nats below control at iteration 1 at
   g=1.0, recovering to 0.04 by 262k. A Beta whose mode is displaced inside a fixed [0,1] support
   is necessarily narrower than a centred one. 6x smaller than v1's contraction, bounded by the
   dose, and partly repaid in the marginal (cond-minus-marginal gap 0.42 nats). The z_t-marginal is
   a Beta MIXTURE, not a Beta (best-fit Beta rejected, marginal variance +17.6% at g=1) - the
   CONDITIONAL is exactly Beta and the conditional is what PPO consumes. No overclaim.

### 14.4 Why g=0.25/0.5 and not learned g or g=1.0

At 262k: control +225, learned g +243 (its g reached only 0.019, so that arm IS the control),
pinned g=1.0 arms +32 / -65 / -85 / -59 at rho 0 / 0.5 / 0.9 / 0.95. g=1.0 displaces the mode by
up to 0.87 of the action range - a very heavy dose - and is behind at 3% of the horizon. Learned g
collapsing toward 0 is NOT decisive: g's benefit is better exploration paying off later, exactly
the long-horizon return a myopic early gradient on g cannot see. So the test is a FIXED dose
bracket at the coherence setting measured free, with g the only variable - the same dose-response
design that produced §8.9, this campaign's strongest evidence.

### 14.5 RESULT - the mechanism WORKS EARLY and the dose does not retire itself - 2026-08-27

Both 8M arms completed (jobs 3903, 3904). Matched checkpoints, HalfCheetah-v4 seed 1:

| arm | @1M | @2M | @3M | @4M | @6M | @8M | final +-CI95 |
|---|---|---|---|---|---|---|---|
| chassis | 3041 | 5465 | 7281 | 8890 | 9873 | 10312 | 10362 +-92 |
| g=0.25 | 3008 | 5451 | 7046 | 8642 | 9780 | 10302 | 10309 +-103 |
| g=0.5 | 3093 | **5972** | **7726** | 8748 | 9299 | 9373 | 8961 **+-1391** |

g=0.25 is a clean TIE (overlapping CI at every checkpoint). g=0.5 is **+507 @2M (+9.3%) and
+445 @3M (+6.1%)**, both above the ~400 noise floor, then CROSSES OVER and finishes -9.1% down.
Dose-monotone early (g=0.5 > g=0.25 at 2M and 3M), so the mechanism is genuinely delivered and
genuinely helps - while the policy is coarse.

**Why it crosses over, measured.** The dose never retires, so late in training it is an
irreducible noise floor that blocks convergence:

| | approx_kl @4M | @6M | @8M | tr_mask_frac @8M | return tail >=7M (mean/std/min/max) |
|---|---|---|---|---|---|
| chassis | 0.0393 | 0.0242 | **0.0061** | 0.0039 | 10223 / 561 / -270 / 10696 |
| g=0.25 | 0.0348 | 0.0436 | 0.0171 | 0.0038 | 10173 / 502 / -57 / 10709 |
| g=0.5 | 0.0813 | 0.0673 | **0.0665** | 0.0168 | 9803 / **1568** / -274 / **10755** |

The chassis's policy drift decays to 0.0061 as it settles; g=0.5 stalls at 0.0665, 11x higher, and
never converges. It REACHES the same peak (max 10755 vs 10696) but oscillates around it with 2.8x
the tail variance. It is not that the arm cannot find the gait - it is kicked off it forever.
EV is HIGHER for g=0.5 (0.9749 vs 0.9636), a third independent confirmation that EV is not the gate.

**Prediction verdicts.** P1 FAIL at both doses on the >=10762 bar (10309 and 8961). P2 partially:
the pinned arms delivered the mechanism (mode_shift_absmean 0.0228 at g=0.5 vs 0.0118 at g=0.25),
but the LEARNED-g arm was never run at 8M because 262k showed g -> 0.019. P4-analogue: the early
gain is real, dose-monotone, and reproducible across two doses.

**This is the THIRD independent mechanism on this chassis with the same signature** - buys early
RATE, does not raise the 8M CEILING (§10 actor-direction across a 31x dose sweep; §14.5 here). The
pattern is now strong enough to be a property of the chassis, not of any one mechanism.

## 15. BETACORR v2 - retire the dose - 2026-08-27

`cleanrl/beta_policy/ppo_continuous_action_entppo_betacorr_v2.py`. Jobs 3910 (g0=0.5) / 3911 (g0=1.0), both
`--corr-gain-anneal-frac 0.5` (linear retirement to exactly 0 by 4M, then held at 0), rho 0.9,
maxParallelRuns 2, --max-attempts 1, --time-limit 8h, seed 1.

Direct consequence of §14.5: the early gain and the late cost are the SAME fixed dose, and the
crossover is measured between 3M and 4M, so retirement by 4M is placed on the crossover. This is
the design §9.1/§10.2 already validated once on a different mechanism - there, holding dose 0.100
gave 9457 while annealing the same early dose to 0 by 4M gave 9808, i.e. annealing recovered
ceiling. Difference in this arm's favour: g=0.25 is ALREADY ceiling-neutral here (10309 vs 10362),
so unlike §10.2 there is no ceiling deficit to claw back, only a late noise floor to remove.

g0=1.0 is included because a fixed g=1.0 was rejected only for its LATE cost (262k returns
+32/-65/-85/-59); with the dose retiring, the late cost is gone and the arm tests whether early
gain keeps scaling with dose. Two doses, one variable, same dose-response design as §8.9.

**v2 verification.** py_compile clean. With annealing off, v2 is BITWISE v1: 57 shared tags, 1140
points, 0 DIFFERING at --num-envs 4 --num-steps 256 --num-minibatches 4 --seed 1 with
--corr-gain-fixed 0.5. The schedule is confirmed live: `debug/corr_gain_scale` traces
1.0/0.9/.../0.1/0.0/0.0 and `debug/corr_gain_absmean` traces 0.5/0.45/.../0.05/0.0/0.0, so the
dose really does retire to exactly 0 and stay there. Implementation is 4 anchors: an Args knob, a
neutral `corr_gain_scale = 1.0`, one multiply inside `corr_gain()`, and one schedule update at the
top of the iteration loop.

### 15.1 First v2 pair lost to an EXTERNAL cancel; g0=1.0 culled on evidence - 2026-08-27

Jobs 3910 / 3911 were both cancelled at 1.728M. **Autocull did not do it** - its last verdict on
both was `keep` (3910 at 0.89x ref, slope +526%/1M, inside the <3M grace window), and only one
autocull process existed. The mlq record shows `cancelled by request` for both with an IDENTICAL
`finished_at` (1787890015111), i.e. a single bulk external cancel, not two independent verdicts.
Every FOREIGN job in that window (3905-3909, the vit-n16 chain) succeeded. Recorded as an
environmental hazard, not a result: non-protected arms on this box can be terminated mid-flight.

**g0=1.0 annealed: DOMINATED, culled without resubmission.** At matched steps it lost to both the
chassis and g0=0.5 everywhere it was observed:

| | @912k vs ref | @1M | @1.728M |
|---|---|---|---|
| g0=0.5 ann | 0.89x | 3097 (chassis 3041) | 5119 |
| g0=1.0 ann | 0.46x | 1895 (0.62x chassis) | 3192 |

Annealing removes a dose's LATE cost; it cannot repair an early dose heavy enough to stop the
policy learning in the first place. Consistent with §13.5, where rho 0.8 on the Gaussian carrier
was flat at negative return. The heavy-dose direction is now closed from two independent files.

### 15.2 Resubmitted pair - dose held, COHERENCE LENGTH varied

Jobs 3912 (rho 0.9) and 3913 (rho 0.95), both `--corr-gain-fixed 0.5 --corr-gain-anneal-frac 0.5`,
8M, seed 1, champion config, maxParallelRuns 2, --max-attempts 1, --time-limit 8h. 3912 is an exact
re-run of the lost 3910 config; 3913 varies ONLY rho.

Why rho is the justified second lever now that dose is bracketed: gate G measured rho 0.95 fully
inside the control's own fragility band (tr_mask_frac 0.84-1.03x, pre-clip actor_grad_norm
0.82-1.06x, ratio_max BELOW control, 10/10 epochs, zero breaker trips) with `losses/entropy` flat
in rho to +-0.02 nats. So additional coherence is free on every axis the chassis is fragile on,
and rho 0.9 -> 0.95 doubles the latent's correlation time (~10 -> ~20 steps) at an unchanged dose.
The question it answers: is the useful quantity the perturbation MAGNITUDE (g, now known to trade
early rate against late convergence) or the COHERENCE LENGTH (rho, measured free)? Nothing in the
campaign has tested the latter at 8M.

## 16. LEDGER - where this campaign actually stands - 2026-08-27

### 16.1 Scoreboard. Net improvement over the incumbent: ZERO

| run | 8M return | vs incumbent |
|---|---|---|
| `entppo_v4_samplemask_a03_mask` (in-charter INCUMBENT) | **10362 +-92** | - |
| `betacorr_g0p25_r09` | 10309 +-103 | tie (overlapping CI) |
| `entjac_aux_c011` | 10296 +-147 | tie |
| best pure-OPSD ever (`jacteach_cos0p95`) | 8915 | -1447 |
| `betacorr_g0p5_r09` | 8961 +-1391 | -1401, late instability |
| `entjac_deepvalue_v1` (culled @4M) | -39% @4M | dead |
| `corrnoise_rho08` (culled @896k) | negative return | dead |

Nothing has beaten 10362. Two arms tied it. That is the bottom line and it should not be dressed up.

### 16.2 The structural finding, which IS the campaign's real output

THREE independent mechanisms, on the same chassis, with the same signature: they buy early
sample-efficiency and do NOT raise the 8M ceiling.

| mechanism | early effect | 8M ceiling |
|---|---|---|
| actor direction (§10, 31x dose sweep) | +49% @1M, 1.49x steps-to-8000 | unchanged (10296 vs 10362) |
| value readout capacity (§12) | none - `value_loss` IDENTICAL | -39% (collateral, not capacity) |
| coherent exploration (§14) | +9.3% @2M, +6.1% @3M, dose-monotone | unchanged at g=0.25, -9% at g=0.5 |

**Inference: the 8M ceiling on this chassis is not set by exploration quality or by critic
capacity.** It is most consistent with late-phase CONVERGENCE being the binding constraint - the
chassis's own `approx_kl` decays to 0.0061 by 8M as it settles onto a gait, and every mechanism
that keeps perturbing it past ~4M loses ceiling while REACHING the same peak (g=0.5 max 10755 vs
chassis 10696). That is a different target from anything tried in §8-§15 and is where the next
real effort belongs.

### 16.3 Findings worth keeping regardless of score

1. **Beta vs Gaussian actor is worth ~+45%** on this chassis (@3M 6052 vs 3607, @4M 7954 vs 4308).
   Never previously measured on this line; the 171 "gauss" run dirs were all HL-Gauss CRITIC.
   The single largest effect this campaign found, and it was found by a CONTROL arm.
2. **EV is disqualified as the bottleneck instrument**, now three times over (§11 0.970/8455 vs
   0.952/9810; §12 identical `value_loss` with -39% return; §14.5 g=0.5 had the HIGHEST EV 0.9749
   and lost). Never again argue "the critic is fine, EV is high" on this benchmark.
3. **Coherence can be made free.** Conditioning on z_t rather than z_{t-1} removes the (1-rho^2)
   conditional-entropy contraction entirely: all rho in {0, 0.5, 0.9, 0.95} sit inside the
   control's own fragility band with entropy flat to +-0.02 nats.
4. **Coherence LENGTH beats DOSE.** rho 0.95 > rho 0.9 at matched steps (@3M 8027 vs 7567, @4M
   9092 vs 8725) at identical dose, and rho 0.95 beat the chassis @3M by +746 (+10.2%). Partial
   evidence only - both runs were externally cancelled near 5M.
5. **Annealing removes a dose's late cost but cannot rescue too heavy an early dose** (g0=1.0
   annealed was 0.62x chassis @1M and lost to g0=0.5 everywhere observed).

### 16.4 HAZARD, recorded separately from model evidence

Three PAIRS of my 8M arms were terminated mid-flight by an EXTERNAL `mlq cancel` (3910/3911 at
1.728M; 3912/3913 at 4.9-5.0M). Autocull is exonerated for 3910-3913: its logged verdict was
`keep` immediately before each disappearance (5 consecutive `keep`s for 3912/3913). Only
non-protected arms were affected; every foreign job in those windows succeeded. This is an
environment property, NOT model evidence, and no conclusion in §14-§15 rests on it. Consequence
for planning: an 8M slot on this box is not reliably available, so a candidate that needs a full
8M horizon to show its effect is a poor bet relative to one that shows a matched-checkpoint effect
by 3-4M.

## 17. SF-OCCUPANCY - back inside OPSD, with a VECTOR channel - 2026-08-27

`cleanrl/opsd/core/sfocc/ppo_continuous_action_opsd_sfocc_v1.py`. Parent: `opsd_jacteach_v1` (8915, best
pure-OPSD on this disk). BUILD ONLY, no runs, no mlq jobs.

### 17.1 Drift correction (second one this campaign, and the one that mattered)
The first attempt at this idea was specced onto the `entppo` PPO chassis as a
"policy-neutral SF instrument". That was drift on two axes at once, and it was cancelled
before it produced anything:
- It reintroduced PPO (clipped surrogate, importance ratio, advantage-weighted PG), which
  §1 forbids outright.
- It cost ~1900 SPS on a chassis whose value is ThinkTrunk+MoE+HL-Gauss+MTP at 10 epochs
  x 32 minibatches - the opposite of what a supervised-regression method should cost.
Correction: OPSD proper, on the OPSD chassis, no policy loss in any backward, and lean
enough to run ~10k SPS. Deleting jacteach's per-sample action-Jacobian is where the speed
comes from; SF replaces that operator with two forward-only heads.

### 17.2 What is actually swapped
jacteach and every prior OPSD arm conditioned the teacher on a SCALAR (Fourier features of
the realized advantage) and built the improvement direction from a learned one-step
Jacobian. This file changes exactly those two things:

| | prior OPSD | sfocc |
|---|---|---|
| privileged channel | scalar advantage, Fourier-embedded | 32-dim occupancy surprise |
| improvement query | inflated advantage / rotated Jacobian direction | shift `delta` along `w` |
| critic | HL-Gauss MTP scalar `V(s)` | vector `psi(s,a)`, value `= w . psi` |
| advantages / GAE | yes | NONE - deleted, nothing for an advantage to be |

Everything else - one network, two contexts, `clone_loss` = `-log p(a_taken | s, c)`,
`distill_loss` = exact per-dim Beta KL onto a detached teacher - is unchanged. So this is a
like-for-like swap of the improvement operator on a measured OPSD chassis.

### 17.3 Why a vector channel, stated as the measured prediction it is
§6.4 measured `R^2(action | credit scalar) = 0.0000` against
`R^2(action | s_{t+1}) = +0.4661`, while `cond_gap` decayed 0.030 -> 0.010 as
`advantage_std` grew 3.7x. The channel was louder and emptier at the same time. The
occupancy surprise is the long-horizon version of the quantity that scored 0.4661,
delivered as 32 dims instead of the one that scored 0.0000. `sf/chan_action_r2` is the
direct vector re-measurement of that 0.0000 and is the file's decisive number.

Geometric motivation, honestly bounded: `psi` is a discounted occupancy, value is LINEAR
in it, so `grad_psi J = w` exactly and constantly - it does not shrink at a local optimum
and does not get buried by sampling noise as the policy sharpens, which is the specific way
every prior arm died. The bound on that claim: the ACTION-space projection
`(d psi/d a)^T w = grad_a (w . psi)` is algebraically identical to Q-ascent, so the
geometry buys no new DIRECTION. What it can buy is an estimator that does not blind itself
and a teacher whose spread encodes how many ways there are to be good - i.e. precision
allocation, which is where the remaining headroom was localized in §16.

### 17.4 Engineered against two named hazards
1. **Action leakage.** Parent lines ~329-330: if `a_taken` appears on both sides,
   `clone_loss` collapses to an identity map and runs to -inf while teaching nothing. The
   naive channel `psi_target - psi` contains `phi(s, a_taken)` explicitly and hands over
   the action. Fix: build the channel from the FUTURE ONLY -
   `c = [gamma * nonterm * boot] - [psi(s, a_bar) - phi(s, a_bar)]` at the student's mean
   action - so `a_t` enters only through the real transition `s_{t+1}`. That is the
   difference between privileged information and leakage. Gate: `sf/cond_action_r2`.
2. **phi collapse.** Trained on reward alone, phi converges to reward and psi becomes a
   scalar critic in a vector coat. Decorrelation penalty + `sf/phi_eff_rank`. Recorded
   limitation: rank is NECESSARY, NOT SUFFICIENT - uncorrelated noise has full rank. The
   `dynphi` sibling arm exists precisely to cover that hole.

### 17.5 Ablation ladder (siblings copied from this file)
- `sfocc_v1` - core. `--cond-mode {occ, occ_w}` makes vector-vs-scalar a one-flag ablation
  with everything else byte-identical, which is better controlled than comparing against
  the legacy advantage channel. `--distill-coef 0` is the null control.
- `sfocc_aspire_v1` - `delta` calibrated by achievement rate instead of fixed, testing
  whether a relative reference resists the absorption that took `cond_gap` to 0.010.
- `sfocc_dynphi_v1` - next-feature prediction pressure on phi, covering the
  full-rank-but-uninformative hole in 17.4.2.

### 17.6 RESULT - the vector channel carries information; the scalar one does not
BUILD + VERIFY ONLY. Nothing launched, nothing queued, `mlq` untouched (0 leases, 0 jobs).
All numbers below are real, measured on this disk, HalfCheetah-v4 seed 1, CUDA, at
`--num-envs 16 --num-steps 256 --num-minibatches 32 --actor-epochs 4 --critic-epochs 4
--total-timesteps 65536` = 16 iterations / 512 actor steps. THE GPU WAS SHARED with a
foreign job at ~98% utilization, so every SPS figure is a lower bound.

**P1 PASSED, and it is the result.** `sf/chan_action_r2` = `R^2(a_taken | channel)`:

| arm | R2(a \| channel) | R2(a \| state) | ratio |
|---|---|---|---|
| `occ` (32-dim vector) | **0.2240** | 0.0477 | **4.7x** |
| `occ_w` (its own scalar projection) | **0.0003** | 0.0933 | 0.003x |

The scalar projection of the *same quantity*, in the same file, one flag apart, carries
LESS than the state alone - reproducing §6.4's `R^2(action | credit scalar) = 0.0000` in a
completely different encoding of a completely different quantity. The vector version
carries 4.7x what the state carries. At iteration 1, with matched fresh policies
(entropy -0.48 in both), the ordering already holds unconfounded: 0.0655 vs 0.0032 against
0.0036 state-only.

Confirmed independently INSIDE the loss the network minimizes, which is worth more than the
R^2 because nothing about it is a probe: `-clone_nll` is the mean log-density in the
privileged context, `-entropy` the same for the unconditioned marginal, so the difference is
what the channel buys.
- `occ`:   4.0348 - 3.3800 = **+0.655 nats**
- `occ_w`: 1.2801 - 1.2834 = **-0.003 nats**
§6.4's own sentence was that a channel predicting nothing about the action is one MLE drops
for free. `occ_w` drops it for free. `occ` does not.

Third independent argument it is not a sharpening artifact: in the `--distill-coef 0` null
control the policy sharpens much further (entropy -5.42) and the *state-only* control
inflates to 0.2926 while the channel stays flat at 0.0654 - the opposite direction to an
artifact.

**P2 PASSED.** `cond_action_r2` 0.2840 against a 0.9 threshold; `clone_nll` -4.03 at student
entropy -3.38 is 0.65 nats of gain, not a Dirac. The future-only construction held: phi(s_t,
a_t) cancels out of the channel and a_t survives only through s_{t+1}.

**P4 PASSED on level, and the agent's own reading of it was WRONG - recorded because it is
an easy number to quote in the flattering direction.** `occ` cond_gap 0.0081 / distill_kl
0.0122 are 8x and 122x the parent's no-op signature (0.001 / 0.0001), so the teacher is not
the student. BUT `occ_w` reads HIGHER on both (0.0175 / 0.0442) while carrying ~750x less
action information, because AdvEmbed's frequencies run to 8 so a constant scalar query moves
16 Fourier channels hard, whereas the vector query moves 32 channels by delta*w_unit_i
(`query_support` 0.18). So **cond_gap and distill_kl measure the trunk's response amplitude
to the query, NOT the information in the channel.** They are no-op detectors and NOT
scalar-vs-vector discriminators. That job belongs entirely to `chan_action_r2` and the
clone_nll-vs-entropy gap.

**P5 PASSED.** Batch held at 32768 so only env count varies; marginal is iterations 1->2:

| envs | cumulative @65536 | marginal | parent |
|---|---|---|---|
| 16 | 9451 | 11678 | 4300 |
| 32 | 10222 | 13664 | -- |
| 64 | 12327 | 17035 | 6013 @128 envs |

2.2x the parent at 16 envs and past its 128-env figure with 16; ~10k reached at 16 envs.
Peak VRAM 0.218 GiB, flat. This came from deleting the per-sample autograd Jacobian and the
6x511-bin critic softmax - per §3 a diagnostic that no search was smuggled in, not an
achievement.

**Also measured, also not flattering:** `teacher_conc_ratio` 0.9963 -> 1.0021, i.e. the query
SHIFTS the policy and barely SHARPENS it, confirming quantitatively on this mechanism the
family's standing observation that mass-covering distillation against a fixed-concentration
teacher has no sharpening channel. And `query_support` 0.18: the query zeroes every channel
component orthogonal to w. Both are the `aspire` sibling's brief.

### 17.7 P3's one unmet gate, and the two-version fix
`sf/psi_r2` MISSED its `>0.5` bar and is recorded as UNMET, not passed. The diagnosis and
repair are worth keeping because the first repair was wrong in an instructive way.

| | v1 | v2 | v3 |
|---|---|---|---|
| `sf/psi_r2` | -1.9084 | -0.6009 | **+0.2124** |
| `sf/psi_bias_frac` | 0.6415 | 0.4751 | **0.0048** |
| `sf/delta` | 0.6726 | 1.3970 | 0.6510 |
| `sf/chan_action_r2` | 0.2402 | 0.2260 | 0.2078 |
| `losses/explained_variance` | 0.2574 | 0.3603 | 0.3453 |

- **v1 diagnosis.** 60% of psi's residual was pure per-dim LEVEL error, and the cause is
  arithmetic: phi's LayerNorm centers ACROSS dims, so each dim keeps a nonzero mean, and the
  lambda-return multiplies it by 1/(1-gamma*lam) ~ 17.5, while psi_net starts at std=0.01
  outputting ~0. The head had to travel ~17 per dim on 512 Adam steps before R^2 could reach 0.
- **v2** made psi regress in standardized target space (level from a buffer, head learns only
  variation). psi_r2 -1.91 -> -0.60 and the init now starts at exactly 0.0 as designed. But
  it introduced a REAL ORDERING BUG: the stats update sat between BLOCK B and BLOCK C, so the
  channel differenced a realized footprint built under old stats against a predicted footprint
  under new stats. Cost, measured: `delta` inflated 2x (0.67 -> 1.40) and `chan_action_r2`
  slipped. v1's claim that "the level cancels" in the channel was already only an 83%
  cancellation - `c = gamma*nonterm*boot - (psi - phi)` weights the level by `gamma*nonterm`
  on one side and by 1 on the other - so the channel was never immune to psi's level.
- **v3** moves the update strictly AFTER the channel (one parameterization per iteration) and
  uses PURE BATCH STATISTICS instead of the 0.99 EMA. The EMA was the second half of the
  problem: psi's target is a BOOTSTRAP, so it drifts as psi learns, and a ~100-iteration lag
  becomes exactly the level error the standardization exists to remove (`psi_bias_frac` climbed
  straight back to 0.4751 in v2). The target is rebuilt every iteration over 4096x32 samples,
  standard error std/64 - nothing to smooth. This REMOVES a knob rather than adding one.
- **v3 verdict, honestly.** The level pathology is solved and proven solved
  (`psi_bias_frac` 0.6415 -> 0.0048) and `delta` is back to v1's uninflated value, confirming
  the ordering diagnosis. But `psi_r2 = 0.2124` STILL misses `>0.5`, and its trace is FLAT at
  0.21-0.29 across 15 iterations rather than climbing - so this is no longer an init or lag
  problem, it is a genuine variation-fit limit. psi explains ~23% of its target's per-dim
  variation. The reward-relevant projection does better (`explained_variance` 0.345, up from
  0.257), which is the projection the mechanism actually uses. **P3 remains 2 of 3.** Whether
  a longer run or a better basis closes it is open; `dynphi` tests the basis.

### 17.8 Status
Three core versions built and measured, `v3` is the core. Two sibling arms building against
v3's measured baseline. Nothing launched, nothing queued, no `mlq` job submitted at any
point. The pre-registered kill criterion did NOT fire: the vector channel carries real
action-attributable information where its scalar predecessor carried none, which is the one
thing this campaign needed to know before spending an 8M slot on the family.

### 17.9 ASPIRE - achievement-calibrated query. A2 FAILS, and the failure generalizes
`cleanrl/opsd/core/sfocc/ppo_continuous_action_opsd_sfocc_aspire_v1.py`, from v3, gated `--aspire`.
`--no-aspire` reproduces v3 bit-exactly (33 algorithmic tags / 624 points / **0 differing**;
the only 16 differing points are `charts/SPS`). Gradient isolation re-verified both ways
(0 of 227 agent params from `sf_total.backward()`, agent grad norm exactly 0.0; 0 of 14 sf
params from the actor backward). `log_delta` is a plain float, never in an autograd graph.

Mechanism: `log_delta += aspire_lr * (achieved - aspire_target)` with
`achieved = mean(w_unit . c > delta)`, so the request sits at the `aspire_target` quantile of
the student's OWN realized-gain distribution and is re-derived every iteration - a relative
reference by construction. Controller correctness proven from the logged series alone:
the recursion holds at all 15 steps to 8.3e-08, `exp(log_delta)` matches `sf/delta` to
1.6e-08 (no off-by-one), and on a frozen batch 400 steps converge to delta 0.7066 /
achieved 0.2495 against the empirical 0.75-quantile 0.7046 - **the analytic fixed point to
0.3%**, independently reproduced by a reviewer on a different batch to 0.17%.

**A2 FAILS, and this is the arm's real contribution.** `teacher_conc_ratio` END 0.9933
against v3's 0.9937 - 0.0004 apart, no movement. But the strong version comes free from the
same runs: v3's OWN delta swings **2.21x** within 16 iterations (0.5692 -> 1.2596 -> 0.6510)
and across both arms request magnitude spans **2.40x** (0.5240..1.2596), while
`teacher_conc_ratio` never leaves 0.9933-1.0115. **Over every magnitude these runs visited,
the concentration response to request magnitude is indistinguishable from zero.** Therefore
no recalibration of `delta` in that range can deliver sharpening, and the whole class of
delta-tuning follow-ups is dead. The lever is how the CONDITIONAL responds to the request,
not how large the request is. Boundary honestly stated: the +-10x the band permits is
UNTESTED, and the direct experiment (one extra teacher forward at 2*delta) is not in the file.

**THE FIXED-TARGET DEFECT IS NOW QUANTIFIED, and this was supposed to be untestable at this
length.** `sf/aspire_achieved` is logged in BOTH modes, so under `--no-aspire` it measures
how often v3's FIXED one-sigma request is actually met: it collapses **0.2283 -> 0.0215**
(min 0.0144), i.e. the request drifts from the **77th percentile to the 97.85th percentile**
of realized gains as the policy sharpens. For a fixed-shape distribution `P(X > std(X))` is
scale-invariant, so this is the realized-gain distribution's right tail CONTRACTING relative
to its spread - the teacher is being queried further and further off its own data over
training. That is the off-distribution-query-fidelity problem, measured directly in 16
iterations instead of needing the parent's 1.75M steps, and it is the defect `aspire` was
built for even though `aspire` did not get to demonstrate the repair.

**A1 SPLITS.** Clamp half passes perfectly (`aspire_clamp_frac` 0.0000 at all 16 iterations).
Convergence UNMET: `achieved` holds within 0.01 of 0.25 for iterations 5-9 then slides to
~0.09. Cause is arithmetic, not noise - the error signal is negative throughout so each step
is at most 0.0125 in log space, and the controller used only **41% of the ~20% authority
available in 16 iterations**. `aspire_lr=0.05` is one e-fold per 80 iterations down; an 8M
run is 244 iterations at file defaults. It is under-powered against a 16-iteration proxy BY
DESIGN. **The gain was deliberately NOT retuned**: tuning a time constant on a 16-iteration
proxy is the coefficient search §2 forbids.

**A3 NOT MET.** `chan_action_r2` END 0.1753 vs v3's 0.2078. The channel/state RATIO is higher
in aspire at 12 of 16 iterations, but that is the DENOMINATOR - its `chan_action_r2` beats v3
at only 2 of 16 points while its state-only control is lower at 12 of 16. Recorded as unproven
in either direction, one seed, and no channel-damage mechanism identified (the channel
construction is untouched). **A4 REFUSED as a claim**: cond_gap 0.008859 vs 0.008921 is equal
to four decimals; 65,536 steps is 3.7% of where the parent's decay appeared.

**Cost: free.** No SPS separation between v3 / `--no-aspire` / `--aspire` in a back-to-back
triple or an interleaved 2x3 sweep (ranges overlap completely; the foreign GPU load moved
during the session so all figures are lower bounds). VRAM identical to four decimals. A
priori cost is one already-computed dot product, a comparison, a mean and an exp, ONCE per
iteration.

**Two real bugs caught before any GPU slot was spent**, both long-run-only:
1. *Integrator windup* - the first draft re-clamped `delta` at read time but integrated the
   UNCLAMPED state, so a state pushed outside the band would latch for thousands of
   iterations. Fixed by writing the clamp back into the state; verified a poisoned
   `log(spread)+50` is caught and the next iteration resumes ordinary control.
2. *NaN latch* - one NaN in the channel makes `spread` NaN, every band comparison False (NaN
   compares False both ways), `min(max(NaN,lo),hi) = NaN`, `exp(NaN) = NaN`, and the state
   stays NaN for the entire run **while `aspire_clamp_frac` still reads 0.0000**. v3 cannot
   have this failure because it rebuilds delta from scratch each iteration; aspire has the
   state as its only source from iteration 2 on. Fixed to fall back to v3's self-healing
   expression and count as a clamp; recovery measured.
Both fixes are behaviour-neutral on the reported runs (0 differing), as expected since
`clamp_frac` is 0 throughout - they matter only for the long run this arm targets.

### 17.10 DYNPHI - next-feature pressure on phi. D2 PASSES and it is the payoff
`cleanrl/opsd/core/sfocc/ppo_continuous_action_opsd_sfocc_dynphi_v1.py`, from v3, gated `--dynphi`.
`--no-dynphi` reproduces v3 **bit-exactly** (33 tags / 624 points / 0 differing, including all
64 `charts/episodic_return` points; `charts/SPS` excluded as wall clock). Every arm is
bit-deterministic across repeats (4 v3 / 6 no-dynphi / 4 dynphi runs, pairwise 0 differing).

Mechanism: `f(phi(s,a), a) -> phi(s', a_bar(s'))`, MSE, target under `no_grad` and `phi_pred`
NOT detached so the gradient shapes phi. `a_bar(s')` is reused from `sf_feature_targets`,
which already computed it for psi's bootstrap, so the mechanism costs zero extra forwards on
that side. NOT a target network: the target is `phi_net` at CURRENT parameters recomputed
every minibatch, so the lag is exactly zero - caching it per iteration WOULD be a stale
target and a violation. Gradient isolation re-verified with `f` in the SF group, and the dyn
term ALONE gives `phi_net` grad norm 1.35595, proving the pressure reaches phi rather than
stopping at `f`.

| gate | | v3 | dynphi |
|---|---|---|---|
| **D2 PASS** | `sf/chan_action_r2` | 0.2078 | **0.2940** (+41%) |
| | `sf/state_action_r2` (control) | 0.1435 | 0.1383 |
| | channel/state ratio | 1.45x | **2.13x** (+47%) |
| | `sf/channel_nats` (in-loss) | 0.6766 | **1.0161** (+50%) |
| **D3 pass, noisy** | `sf/psi_r2` | 0.2124 | 0.3364 (+58%) |
| | `losses/explained_variance` | 0.3453 | 0.3791 |
| **D4 PASS** | `sf/w_r2` | 0.9599 | 0.9602 |
| **D1 NOT MET** | `sf/phi_dyn_r2` | -- | 0.7130 (from -0.7641) |
| | `sf/phi_eff_rank` | 20.6792 | 16.5464 |
| | `sf/phi_var_mean` | 0.3370 | **0.1256** |

**D2 is the payoff and it is not a sharpening artifact**: the state-only control barely moved
(0.1383 vs 0.1435) while the channel rose 41%, dynphi is above v3 at all 14 iterations from
the third onward from an IDENTICAL iteration-1 value, and the in-loss `channel_nats` agrees
independently at +50%. A dynamics-relevant basis makes the occupancy surprise materially more
action-attributable, which is exactly the claim.

**D1 IS NOT MET AS PRE-REGISTERED and the arm says so first.** D1 is a conjunction; clauses 1
and 2 pass (`phi_dyn_r2` 0.7130; `eff_rank` 16.55 is 4.1x the >4 bar) and clause 3 FAILS
(`phi_var_mean` 0.1256 against the control's 0.3370). Verdict: **REAL AND PARTIALLY
DEGENERATE.**
- Real: R^2 cannot be bought by concentration because it normalizes by the target's own
  variance; `phi_dyn_mse` fell 80x while `phi_var_mean` fell 6.7x, so the variance-normalized
  error fell ~12x.
- Degenerate: the arm DERIVED that `phi_var_mean == 1 - ||E[phi]||^2/phi_dim` exactly, because
  non-affine LayerNorm fixes `||phi_i||^2 = phi_dim`. So the instrument is bounded in [0,1]
  and measures ANGULAR concentration, not scale: dynphi's rows end within **21 deg** of one
  common direction versus the control's **35 deg**, consuming 85% of the initial spread versus
  60%. `phi_offdiag_absmean`'s controllable excess above the 1/31 floor nearly doubles
  (0.0060 -> 0.0115). All three collapse instruments are strictly worse than the control at
  EVERY post-init iteration, so this is not endpoint noise.
- **Unresolved:** the relative decline over the last six iterations is a roughly constant
  ~5%/iteration (3.1, 2.7, 8.1, 4.9, 5.1, 4.9%) with no downward trend - precisely the "slow
  geometric collapse" alternative that 16 points cannot exclude. Escalation NAMED and
  DELIBERATELY NOT IMPLEMENTED: a VICReg-style variance floor is a new coefficient and belongs
  in its own arm under the no-grids rule.

**Cost, and the arm refused a false pass.** End-to-end SPS could NOT resolve it: foreign GPU
contention swung ~3.6x during the session (468-2154 SPS for identical work), dynphi's fastest
repeat BEAT every v3 repeat, and `--no-dynphi` (bit-identical to v3, strictly less work)
straddled v3 in both directions. An earlier back-to-back pair under 97% contention read +1.0%
and would have been a contention-masked FALSE PASS; that draft was corrected. Isolated
in-process A/B on the only changed hot path, arms alternating, `cuda.synchronize` around each
block, median of 11 repeats x 400 steps: **+29.4%** (1169.94 -> 1513.48 us/step), ABOVE the
~15% bar and reported as a finding. End-to-end that is ~+2% (128 SF steps/iteration =>
+44 ms/iteration on a 32-45 s run). Both numbers are true and neither is quotable alone: the
mechanism makes its own path ~30% dearer and that path is small. Cause is kernel-launch bound,
not FLOP bound, at 1024x64. VRAM +1.5% (0.0890 vs 0.0877 GiB).

**Latent bug found and handled by measurement, not by adding code.** The dyn term is the only
consumer of `b_next_obs` that needs `(s, s')` to be a real transition on EVERY row - psi's
target tolerates invalid rows because `nonterm` zeroes the bootstrap, but a regression target
cannot - and the correct mask is `transition_valids`, NOT `nonterm`. Measured on
HalfCheetah-v4 with this exact rollout code: 10,400 rows, 8 boundaries, 0 terminations,
`final_observation` supplied on all 8, and `transition_valids == 0` on ZERO rows. So a mask
would be provably dead weight in every minibatch on the only env the charter admits. Following
BLOCK B's precedent the invariant is ASSERTED once per iteration with a message naming the
exact fix - one reduction instead of one per minibatch, and a loud failure instead of silently
regressing wrong pairs if the env or vector-env API changes.

**Limitation disclosed rather than glossed:** `phi_dyn_r2` is measured on a phi the dyn term
has itself been shaping, so it does not separate "the pressure created predictability" from
"phi was already predictable and f found it". The clean control is a probe-only mode (f trained
with `phi_pred` detached), which is a second mechanism and out of scope. `--dyn-coef 0` does
NOT substitute - `0*dyn_mse` backprops zeros, Adam takes a zero step, and f would report from
its random init - and that trap is documented in `Args`.

A red-team audit found 2 HIGH + 5 MEDIUM + 8 LOW issues, **almost all overclaiming in the
header, all fixed**, including relabelling D1 from PASS to NOT MET and correcting a leakage
reassurance that had used the wrong instrument and inverted its own conclusion.

### 17.11 LEDGER for section 17
Five files built, all compiling, all verified by measurement. **Nothing launched. No `mlq` job
submitted at any point.** Parents untouched (`jacteach_v1` bce53f1f, `sfocc_v3` 3fa42dcd).

| file | role | status |
|---|---|---|
| `opsd_sfocc_v1` | OPSD-SF core, vector channel + half-space query | P1/P2/P4/P5 pass, P3 2 of 3 |
| `opsd_sfocc_v2` | psi in standardized target space | superseded; introduced a real ordering bug |
| `opsd_sfocc_v3` | **the core.** update after the channel, batch stats | level pathology solved; psi_r2 still unmet |
| `opsd_sfocc_aspire_v1` | achievement-calibrated delta | A2 FAILS generally; quantified the fixed-target defect |
| `opsd_sfocc_dynphi_v1` | next-feature pressure on phi | **D2 PASS +41%**; D1 partially degenerate |

What section 17 established, in order of how much it is worth:
1. **The vector channel carries action-attributable information; its own scalar projection
   carries none** (0.2240 vs 0.0003 against a 0.0477 state control, +0.655 vs -0.003 nats).
   §6.4's `R^2(action | credit scalar) = 0.0000` was a property of SCALAR ENCODING, not of
   hindsight conditioning. This is the one thing the campaign needed before spending an 8M slot.
2. **A dynamics-relevant basis improves the channel a further 41%** (ratio 1.45x -> 2.13x),
   confirming that phi's quality - not the query, not the operator - is the live lever.
3. **The delta-recalibration class is dead for sharpening.** Zero concentration response across
   a 2.40x span of request magnitudes. Do not build another arm that tunes the request size.
4. **The fixed request drifts off-distribution as the policy sharpens** (77th -> 97.85th
   percentile of realized gains), which is a real defect measurable in 16 iterations.
5. **The teacher still only SHIFTS, never SHARPENS** (`teacher_conc_ratio` within 0.007 of 1.0
   in every arm at every magnitude). This is the largest unexplained gap and the natural next
   target: the lever is how the CONDITIONAL responds to a request, not the request.
6. `psi_r2` never cleared 0.5. Best is dynphi's 0.3364. The level pathology is fixed
   (`psi_bias_frac` 0.0048); what remains is variation fit, and it is unresolved.

Best candidate if a slot is ever spent: **dynphi**, with a variance floor arm built first to
settle whether phi's angular contraction settles or collapses - that question is unanswerable
at 65,536 steps and it gates the whole basis story.

### 17.12 DYNPHI-VARFLOOR - long-run decision arm queued, 2026-08-28
`cleanrl/opsd/core/sfocc/ppo_continuous_action_opsd_sfocc_dynphi_varfloor_v1.py`, from
`opsd_sfocc_dynphi_v1`. BUILD + REVIEW + QUEUE ONLY at this point; no return result is
claimed. The parent and incumbent remain untouched:

| source | MD5 | SHA256 |
|---|---|---|
| `opsd_sfocc_v3` | `3fa42dcde0a8b0f4c8e815c015720eb1` | `08d80ba3421ecd58482ef204502bece334a243bbc92d49ac2473308149a87f12` |
| `opsd_sfocc_dynphi_v1` | `010316cad4f814e44606a7178f6e7c89` | `3198960f4125d1d800678167924bb18e18b6afa27dd5a11ddc96deeebb0ec8ef` |
| `opsd_sfocc_dynphi_varfloor_v1` | `8bcac327e1dd4f577612c1db7fdbb4eb` | `86a33ef3ef7f2fe06e0a6689c476224325ad76aa1d72cdb40f2733521a2fd754` |

**Single change.** On each SF minibatch, add the VICReg per-coordinate hinge
`mean(relu(0.5 - sqrt(var(phi_j) + 1e-4)))` with coefficient 1.0. No grid. The
per-dimension 0.5 standard-deviation floor directly implies `phi_var_mean >= 0.25` when
all coordinates satisfy it. If LayerNorm row mean-square stays near 1, as the new
`phi_row_ms_{mean,min,p10}` tags test rather than assume, this corresponds to retaining
about 30 degrees of mean angular spread: permissive to v3's measured 35 degrees and
incompatible with dynphi's 21-degree proxy endpoint. The hinge is inactive above the
floor and does not weaken `dyn_coef`, so it tests whether dynphi's channel gain survives
without buying predictability through contraction.

Interpretation is deliberately counterfactual-safe. The constrained arm alone cannot
show where untreated dynphi would settle. `phi_floor_active_frac` only says whether the
floor is active on its own trajectory; the matched no-floor arm supplies the untreated
trajectory. `phi_var_floor` is the pre-update full-batch hinge, while
`losses/sf_var_floor` is what the optimizer paid over minibatches. `sf_grad_norm` and
`sf_grad_clip_frac` expose competition for the 0.5 SF clip budget, but cannot by
themselves attribute clipping to the floor. The exact-collapse limitation is also
recorded: the epsilon keeps the denominator finite, but an exactly constant coordinate
has zero variance gradient and cannot be resurrected by this term alone.

**Verification.** `py_compile` passes and five focused CPU tests pass: hinge form and
direction, exact-collapse zero-gradient behavior, phi-only gradient isolation, parent
loss equality under `--no-var-floor`, and exact enabled-loss decomposition. No reduced
GPU run was used and the 65,536-step proxy is not performance evidence. An independent
red-team review found no loss/gradient/optimizer blocker and caused the LayerNorm-energy
and counterfactual claims above to be narrowed before queueing.

**Exact long-run lineage.** There was no prior SF-OCC model checkpoint or long run on
disk; section 17.1-17.11 were proxy measurements only. To prevent the dirty shared checkout
from changing queued code, the jobs execute read-only, hash-addressed source snapshots
under `artifacts/source-snapshots/<SHA256>/`. All three are HalfCheetah-v4, seed 1,
8,000,000 steps, 16 envs x 2048 steps, 32 minibatches, actor epochs 1, critic epochs 4,
`cond_mode=occ`, CUDA, deterministic, async envs, and `compile_act`. No W&B, extra seed,
priority, retry, time limit, or autocull policy.

| job | role | mechanism |
|---|---|---|
| `3958` | v3 incumbent | source `08d80...`; core only |
| `3959` | matched untreated dynphi | source `86a33...`; `--dynphi --no-var-floor` |
| `3960` | variance-floor arm | source `86a33...`; `--dynphi --var-floor --phi-std-floor 0.5 --var-floor-coef 1.0` |

Each declares `maxParallelRuns=1`. Jobs 3955-3957 were cancelled before start and
replaced by these snapshot-backed jobs; they consumed no GPU work. At submission the
shared queue had healthy foreign work and an effective limit of 1, which was preserved.
Compare only at matched 500k, 1M, 2M, 4M and 8M checkpoints. The decision set is return,
`phi_var_mean`, row energy, floor activity/loss, `phi_dyn_r2`, `psi_r2` against its
original 0.5 bar, `w_r2 > 0.9`, channel/state action-R2 ratio, channel nats,
`cond_action_r2` leakage, explained variance, and the SF gradient clip telemetry.

### 17.13 ACTTEACH - one always-active SF action teacher, 2026-08-28
The three 8M section-17 runs answered the performance question decisively:

| arm | @500k | @1M | @2M | @4M | @8M |
|---|---:|---:|---:|---:|---:|
| v3 | -169 | -345 | -435 | -470 | -367 |
| dynphi, no floor | -279 | -418 | -451 | -556 | -575 |
| dynphi + variance floor | -341 | -350 | -470 | -559 | -572 |
| JacTeach cos 0.95 reference | 1464 | 4370 | 6701 | 8106 | 8859 |

This is not delayed return learning. The auxiliary models learned: near 8M `psi_r2` was
0.875/0.916/0.895, `w_r2` 0.971/0.891/0.907, and both dynphi arms reached
`phi_dyn_r2 > 0.9`. The variance floor changed `phi_var_mean` from about 0.087 to 0.307
while return stayed -574/-572. Representation collapse is real but is not the return
blocker.

The missing operation was action optimization. v3 trained
`p(a_taken | s, realized_occupancy_residual)` and then assumed a synthetic `delta*w` query
would invert to an improving action. No loss trained that implication. In
`cleanrl/opsd/core/sfocc/ppo_continuous_action_opsd_sfocc_actteach_v1.py`, the only new algorithm version,
the updated SF critic instead constructs the detached teacher directly:

`g = d[w.psi(s, action(z))]/dz` at the student Beta mean, followed by the exact local
fixed-concentration Beta-Fisher direction and a per-state exact forward-KL bisection to
0.10 nats. Teacher and snapshot student share concentration. The critic runs first, the
teacher is frozen once, and the actor still minimizes only hindsight clone NLL plus
unclipped detached teacher KL.

**No gate.** Critic fidelity never changes control flow. A fidelity gate would remove the
only optimization pressure early and create a hidden curriculum. The fixed KL trust region
bounds the teacher step; exact-zero gradients are the only zero-dose case, handled without
a numerical threshold. `teacher_kl_shortfall_frac` exposes action-bound infeasibility.

Independent review found no gradient-path or update-order blocker and caused four fixes
before queueing:
1. full-batch pre/post teacher-student KL, gap, progress, and alignment now measure whether
   the actor actually followed the frozen teacher; the old `cond_gap` meaning was removed;
2. inherited per-dimension KL clipping was removed so the teacher loss is always active;
3. `clamp_min(1e-12)` normalization became an exact-zero mask;
4. `teacher_psi_r2`, `teacher_w_r2`, and `teacher_w_norm` are measured after the critic
   update that actually supplies the action derivative.

Focused checks pass: `py_compile`; exact 0.10 KL geometry; alpha/beta >= 1; positive
predicted gain; exact-zero behavior; no actor or SF parameter grads from teacher
construction; detached teacher/snapshot tensors; and a constructed half-step reporting
0.5 follow progress, cosine 1.0, and reduced post-update KL/gap. AST inspection finds only
two backwards (SF supervised loss and actor supervised loss) plus one
`autograd.grad`, whose sole requested input is detached latent action `z`.

| source | MD5 | SHA256 |
|---|---|---|
| `opsd_sfocc_v3` | `3fa42dcde0a8b0f4c8e815c015720eb1` | `08d80ba3421ecd58482ef204502bece334a243bbc92d49ac2473308149a87f12` |
| `opsd_sfocc_actteach_v1` | `c58c943a2e54ff754c92d23555b60d4b` | `5108336d152969d5cc7c8d18df4d126c85c16fea80c1dea25d805080a956ff1b` |

Job `4044`, the sole compute iteration, executes the read-only hash-addressed source
snapshot. HalfCheetah-v4, seed 1, 8M steps, 16 envs x 2048, 32 minibatches, actor epoch 1,
critic epochs 4, `cond_mode=occ`, CUDA, deterministic, async envs, and compiled acting
forward. `maxParallelRuns=1`, max attempts 1, two-hour time limit, priority 0, no W&B,
autocull, retry, extra seed, sibling, or coefficient ladder. At submission the shared queue
had one active lease and effective limit 1, so this job waits without displacing foreign
work.

Decision checkpoints are 500k, 1M, and 2M. Pressure requires snapshot KL near 0.10 with low
shortfall and positive predicted gain. Actor use requires positive follow progress/alignment
and post-update KL/gap below pre-update values. The critic snapshot requires
`teacher_psi_r2` and `teacher_w_r2 > 0.5`. The run is rejected if clearly behind v3 at
1M-2M; success requires a positive sustained return curve, not merely a less-negative
endpoint.

### 17.14 ACTTEACH FULLBETA V2 - restore the teacher's actual dose, 2026-08-28

Job `4044` proved that action-space SF improvement is useful but not stable under the v1
projection. Return peaked near 900 around 1M, held near 600 through 4M, then fell to 9.
The post-update critic did not collapse: `teacher_psi_r2` remained about 0.8 and
`teacher_w_r2` about 0.99. The nominal 0.10-KL teacher did collapse:

| step | delivered KL | shortfall frac | boundary frac | follow alignment | return |
|---:|---:|---:|---:|---:|---:|
| 0.79M | 0.0638 | 0.549 | 0.803 | 0.412 | 845 |
| 1.05M | 0.0387 | 0.789 | 0.886 | 0.169 | 923 |
| 4.00M | 0.0131 | 0.945 | 0.960 | 0.135 | 589 |
| 8.00M | 0.0031 | 0.994 | 0.983 | 0.013 | 9 |

V1 preserved concentration and required `alpha,beta >= 1`, restricting its mean to
`[1/k, 1-1/k]`. It also capped the KL search scale at 4. Scale reached 3.99 while the
action-gradient remained finite. KL reduction became persistently negative at 0.655M and
shortfall exceeded 50% at 0.786M, both before the return peak. Thus the teacher silently
removed its own improvement pressure despite having no explicit gate. The shared
privileged-clone trunk may explain why the student stopped following the weakened teacher,
but the run lacks per-loss gradient cosine telemetry, so that attribution remains
unproven. The state-only policy nevertheless became diffuse: entropy moved from about
-1.77 near the peak to -0.31 and `state_action_r2` from about 0.30 to 0.026.

`cleanrl/opsd/core/sfocc/ppo_continuous_action_opsd_sfocc_actteach_fullbeta_v2.py` makes one correction.
It differentiates `w.psi` through both detached Beta parameters, solves the exact 2x2 Beta
Fisher system per action dimension, normalizes the full per-state natural direction, and
moves both parameters. Concentration can therefore rise after one parameter reaches its
floor instead of stopping the mean. The KL interval begins at scale 1 and doubles until it
brackets 0.10; failure after the defensive expansion budget raises rather than weakening
the teacher. A 24-step bisection then supplies the detached target. Exact-zero gradients
remain the only zero-dose case.

New telemetry makes the correction decidable:

- `teacher_kl_abs_error` and `_max`: exact-dose projection error;
- `teacher_bracket_expansions` and `_max`: dynamic search cost;
- `teacher_concentration_ratio` and `teacher_log_concentration_change`: how the full-family
  projection escaped the fixed-concentration boundary;
- `teacher_zero_grad_frac`: the only legitimate zero-pressure population;
- `teacher_boundary_frac`: now the fraction of teacher alpha/beta parameters at their
  floor, not fixed-concentration mean infeasibility.

The actor remains pure OPSD: hindsight clone NLL plus unclipped forward KL from one frozen,
detached SF teacher. No value, advantage, ratio, policy-gradient loss, fidelity gate,
warm-up, or teacher schedule was added. The one permitted corrective run keeps the v1
HalfCheetah-v4 seed-1 8M configuration and has no sibling, coefficient ladder, or extra
seed. Success requires exact 0.10 teacher pressure, positive actor following, critic R2
above 0.5, and a positive return curve that does not repeat v1's post-1M decay.

Independent review confirmed the Fisher blocks, inverse signs, per-state normalization,
forward-KL order, monotone clamped ray, dynamic bracket, update order, detached gradient
paths, and unchanged pure-supervision actor loss. It found two fail-fast blockers and two
telemetry defects before queueing. The final source:

1. defines exact zero from raw gradient components and rescales every nonzero per-state
   gradient before the Fisher solve, so float32 quadratic-form underflow cannot disable a
   teacher;
2. rejects non-finite gradients, Fisher blocks, directions, every bracket/bisection KL,
   final teacher, dose, and predicted gain instead of letting NaN comparisons look
   bracketed;
3. requires the final nonzero-gradient dose to be within 1% of 0.10;
4. excludes exact-zero states from `teacher_nonpositive_gain_frac` and uses emitted tag
   names in the header.

Focused verification passed: `py_compile`; LSP diagnostics; 8,192-state stress geometry
spanning Beta parameters from approximately 1.001 to 101; mean delivered KL 0.0999997,
maximum error 0.0001124, zero shortfall, positive predicted gain, alpha/beta floors
preserved, and no actor/SF parameter gradients; exact-zero gradients return the exact
snapshot with zero KL; nonzero gradients scaled to about 1e-30 still deliver 0.10 KL; NaN
gradients fail immediately; and 1,024-row full/chunked construction agrees within
7.2e-7 with aggregate metric differences below 1e-10. The independent reviewer re-read
the fixes and marked the candidate safe to queue.

| source | MD5 | SHA256 |
|---|---|---|
| `opsd_sfocc_actteach_fullbeta_v2` | `2e414164c096490221264be7fd29835b` | `bb55184e575d9b90125605a000cf98ce849ed6de7c8feb6d11333deb863b1c51` |

Job `4070` is the sole corrective compute run. It executes the hash-addressed immutable
snapshot for HalfCheetah-v4, seed 1, 8M steps, 16 envs x 2048, CUDA, deterministic async
envs, and compiled acting forward. `maxParallelRuns=1`, attempts 1, two-hour time limit,
priority 0, no W&B, autocull, retry, sibling, ladder, or extra seed. It entered the shared
queue as `backfill_window_open: job 4064` without displacing the active foreign lease.

### 17.15 ACTTEACH FIXEDCONC V3 - remove the concentration ratchet, 2026-08-29

Job `4070` materially outperformed v1 but did not complete. It reached 5.36M environment
steps before `sf_action_teacher` raised `RuntimeError: full-Beta teacher missed target KL`;
there is no 8M endpoint. Its 20-episode return peaked at 2939.6 at 2.992M, fell to 929.0
at 3.360M, and recovered only partially to 1891.3 at termination. The learning curve was
therefore positive but unstable, not a successful final fix.

| step | 20-ep return | student entropy | teacher concentration ratio | post-update KL reduction |
|---:|---:|---:|---:|---:|
| 0.50M | 773.7 | -3.54 | 1.024 | 0.526 |
| 1.00M | 720.9 | -6.67 | 1.072 | 0.380 |
| 1.50M | 1914.0 | -11.02 | 1.110 | 0.335 |
| 2.00M | 2467.0 | -14.68 | 1.133 | 0.244 |
| 2.50M | 2672.0 | -16.30 | 1.109 | 0.104 |
| 3.00M | 2893.0 | -17.46 | 1.112 | -0.024 |
| 3.50M | 2213.0 | -17.56 | 1.104 | -0.009 |
| 4.00M | 1184.0 | -15.77 | 1.061 | 0.155 |
| 5.00M | 1607.0 | -17.22 | 1.096 | 0.047 |
| 5.36M | 1891.3 | -17.63 | 1.109 | -0.082 |

V2 delivered its nominal pressure: snapshot KL remained 0.10 with zero shortfall and
positive predicted gain. The post-update critic also remained usable: `teacher_w_r2`
stayed around 0.98-0.996 and `teacher_psi_r2` was usually 0.74-0.88 after its early fit,
with a temporary 0.61 reading at 2M. The remaining failure was actor geometry:

- the teacher/snapshot concentration ratio exceeded 1 on every update, with median 1.089;
- student entropy crossed -5 at 0.72M, -10 at 1.38M, -15 at 2.13M, and -17 at 2.59M;
- during the second half, median one-update teacher-KL reduction was 0.001 and 48.8% of
  updates increased KL;
- maximum per-state dose error first exceeded `1e-4` at 1.47M, `5e-4` at 2.59M, and
  `8e-4` at 3.51M. Entropy and maximum error had correlation -0.856. The error eventually
  crossed the explicit 1% limit and produced the observed fail-fast termination.

The full-Beta natural gradient was mathematically valid, but it optimized a larger family
than the critic objective justified. `w.psi(s, action(mean(alpha,beta)))` is invariant to
concentration. Its two-parameter natural direction nevertheless acquired a radial component
from the Beta Fisher geometry, and the exact-KL projection emitted a concentration increase
on every rollout. Forward-KL distillation then trained the actor toward that sharper target.
This is a direct concentration-ratchet mechanism. The coincidence between sharpening,
degraded following, and return oscillation is strong evidence, but one seed does not prove
that the ratchet alone caused every return reversal.

`cleanrl/opsd/core/sfocc/ppo_continuous_action_opsd_sfocc_actteach_fixedconc_v3.py` removes the radial
degree of freedom instead of gating it. Policy and teacher now share
`k = 2 * (1 + log(2)) = 3.386...`, the original zero-logit policy's initial concentration:

```text
alpha = k * mu
beta  = k * (1 - mu)
```

The actor has one raw mean-logit head and no concentration head. Paired stable constructions
`alpha = k*exp(logsigmoid(u))` and `beta = k*exp(logsigmoid(-u))` keep both parameters and
their gradients representable far beyond the range where a float32 sigmoid output rounds
to 0 or 1. Either parameter may fall below 1, so the teacher can approach an action boundary
without v1's early infeasible interval. The same raw finite logit supplies the snapshot,
critic gradient, Fisher geometry, and teacher displacement; reverse reconstruction from
rounded Beta means is forbidden.

The teacher uses the exact fixed-concentration Fisher metric

```text
I_u = k^2 * (trigamma(alpha) + trigamma(beta)) * (d mu / d u)^2.
```

The per-state natural direction is normalized to unit Fisher norm. Dynamic bracketing and
40-step float64 bisection target total forward
`KL(teacher || actual float32 snapshot)=0.10`; the dose check is repeated after casting the
detached target to the representation consumed by the actor. Exact-zero gradients preserve
the exact original snapshot, including its rounding. Nonfinite raw logits, gradients,
metrics, directions, KL values, targets, or gains fail explicitly. Zero-dose rows are
excluded from relative follow reduction and their absolute drift is logged separately as
`teacher_zero_dose_drift`.

The actor contract is unchanged: one hindsight clone NLL plus one detached teacher forward
KL, with critic first, frozen teacher second, actor last. There is no PPO term, value or
advantage tensor, fidelity gate, warm-up, entropy bonus, dose schedule, or
threshold-controlled teacher.

Fixed concentration removes v2's radial sharpening pressure, not all possible entropy loss.
As a fixed-k Beta mean approaches 0 or 1, variance and differential entropy still fall.
Whether the resulting boundary behavior is useful task exploitation or another exploration
failure remains an empirical risk; there is no v3 run from which to decide it.

Focused verification passed without an ML run:

- `py_compile` and LSP diagnostics report no errors;
- 8,192 interior/boundary-directed states delivered mean KL `0.099999997`, maximum error
  `2.8e-9`, zero shortfall, positive predicted gain, mean target concentration ratio 1.0,
  and no actor parameter gradients;
- a second 8,192-state stress at raw mean logits `[-80,-40,-20,20,40,80]` delivered mean
  KL `0.099999971`, maximum error `2.9e-8`, zero shortfall, fixed concentration, and
  bit-identical full-batch versus 127-row chunks. Predicted action gain is numerically zero
  once float32 actions themselves saturate, an explicit residual finite-precision limit;
- a nonzero gradient scaled to `1e-30` still delivered KL 0.10; exact-zero gradients
  returned the exact snapshot; NaN gradients and infinite raw logits raised immediately;
- zero-dose follow reduction remains 0 while `teacher_zero_dose_drift` separately reported
  induced actor drift;
- the scalar Fisher matched projection of the exact 2x2 Beta Fisher to relative error
  `4.7e-16`; 4,096 random rays spanning raw logits `[-80,80]` had positive ascent dot
  products, finite KL, and no KL decrease over scales 0 through 8;
- one actual distillation update reduced teacher/student KL from 0.1000 to 0.0651 while
  preserving concentration to float32 rounding (`4.8e-7` maximum error).

Independent review first found two projection blockers and one telemetry defect: the initial
epsilon-truncated mean could still exhaust outward KL at its finite boundary; reconstructing
a surrogate teacher logit could disagree with the actor's saturated backward path; and
zero-dose rows could dominate relative follow reduction. The raw-logit parameterization,
paired stable Beta construction, shared coordinate, finite-logit check, exact zero snapshot,
and zero-dose telemetry above are the corresponding fixes.

The re-review found no remaining high- or medium-severity blocker. Its two low findings were
also fixed: active-dose KL reduction now divides by the active count rather than the whole
batch, follow KL/drift uses nonnegative float64 evaluation, and teacher construction rejects
non-positive or nonfinite snapshot parameters after paired-logsigmoid evaluation. A
mixed-dose check with exactly 512 active and 512 zero-dose rows matched manual active-only
reduction (`0.324151`) and zero-dose drift (`0.002645`) exactly. A deliberately
unrepresentable raw logit of 110 now fails explicitly at snapshot validation, and the same
positivity/finiteness check runs over the post-update rollout batch before another rollout
can consume an invalid policy. This is a finite-precision boundary, not a claim that every
finite real logit maps to a representable positive float32 Beta parameter.

The final reviewer re-read the post-update guard and reported no unresolved correctness
defect or high/medium blocker; an unseen state with a much more extreme next-rollout logit
remains a theoretical finite-domain limit, not an observed path.


| source | MD5 | SHA256 |
|---|---|---|
| `opsd_sfocc_actteach_fixedconc_v3` | `37f9bfd1abbf6031789b513a8c7cb56f` | `d5387706e3aec7c76ec25a2911971f55c93e9ee1a559695a7fbeb12a3a29bf17` |

The user subsequently authorized one v3 benchmark. Job `4106`,
`opsd-sfocc-actteach-fixedconc-v3`, runs the hash-addressed snapshot above on
HalfCheetah-v4 for 8M steps, seed 1, 16 envs, CUDA, deterministic async envs, and compiled
acting forward. It has `maxParallelRuns=1`, priority 0, and one attempt; no W&B, sibling,
coefficient ladder, or extra seed. It entered the queue as
`backfill_window_open: job 4105`.

### 17.16 THREE STRUCTURAL ABLATIONS - build only, 2026-08-29

The fixed-concentration run finished at 5915 last-20 without v1/v2's return collapse, but
its endpoint exposed three independent limits: student entropy reached `-1.45e8`, 28.7% of
full-dose teacher endpoints had nonpositive gain under their own critic, and the learned
feature covariance retained only about 6.4 effective dimensions. Three independent files
isolate those mechanisms; none is cumulative:

1. `cleanrl/opsd/core/sfocc/ppo_continuous_action_opsd_sfocc_actteach_censoredgauss_v1.py` replaces only
   the actor/teacher distribution geometry. Its bounded mean and fixed latent-Gaussian
   scale match v3's zero-logit action variance; the environment sees a censored action,
   while clone NLL fits the exact uncensored latent. The teacher uses the corresponding
   fixed-scale Fisher metric and treats 0.10 KL as a feasible-box upper bound.
2. `cleanrl/opsd/core/sfocc/ppo_continuous_action_opsd_sfocc_actteach_monotone_v1.py` keeps v3's policy and
   exact full-dose natural target. Rows whose finite 0.10-KL endpoint does not improve the
   frozen critic halve only their own scale until gain is strictly positive or the exact
   snapshot is the only representable target. Full-dose candidate telemetry remains
   comparable; accepted-dose telemetry is additive.
3. `cleanrl/opsd/core/sfocc/ppo_continuous_action_opsd_sfocc_actteach_coherentbasis_v1.py` keeps v3's actor
   and teacher. It fits `phi+w`, freezes that basis, rebuilds the feature lambda-return,
   then fits only `psi`. The two stages reuse the same `critic_epochs` permutations, so the
   staged update consumes the parent's RNG draws rather than shifting actor order or future
   action samples.

All three files pass `py_compile` and Pyright diagnostics. Focused mathematical checks
verify Gaussian interior KL `0.100000003`, exact zero-gradient snapshots and boundary
shortfall; monotone full-dose KL `0.099999993`, one-step curved backtracking to KL
`0.0249802` with positive gain, and exact zero-gradient snapshots; and disjoint/exhaustive
staged optimizer ownership, gradient isolation, post-basis target ordering, and permutation
reuse. Independent review found no remaining high-, medium-, or low-severity defect after
the telemetry, RNG, validation, and allocation findings were fixed. These are build-only
ablations: no source snapshot, MLQ job, training result, coefficient ladder, or extra seed
exists yet.

### 17.17 TRUE PRIVILEGED-POSTERIOR OPSD - isolate rationalization, 2026-08-29

The three section-17.16 structural ablations all underperformed by user report. They changed
distribution geometry, target acceptance, and SF basis coherence, but retained the external
action-conditioned SF critic that constructs the teacher. That common dependency was the
wrong boundary: it is not canonical OPSD, and the three negative arms did not justify another
SF correction. The next comparison returns to one shared policy with privileged posterior
information and isolates rationalization gradients from the zero-context student.

`cleanrl/opsd/core/teachers/ppo_continuous_action_opsd_residualteacher_v1.py` preserves the v6 student's exact
zero-context trunk and Beta heads. A separate, zero-initialized residual adapter consumes
detached base features plus a present Fourier embedding of the realized one-step TD residual.
One fresh-data clone pass updates only that adapter. Its optimistic `residual + 1` query is
then frozen, and the base actor receives only clipped forward teacher-to-student KL; value
regression retains the proven shared v6 trunk. Numeric zero is still a present context, while
absence bypasses the adapter.

`cleanrl/opsd/core/teachers/ppo_continuous_action_opsd_residualteacher_jac_v1.py` changes only the frozen
teacher's direction. A one-step TransitionHead, trained only on observed transitions, rotates
the residual-OPSD displacement toward `d[gamma V(s')]/dz` while retaining cosine `0.95` in
the policy's Fisher/std coordinate. The gate uses R2 measured on the next rollout before any
fit to it. Exact-plane rotation preserves direction semantics; bidirectional scalar
bisection matches the unrotated teacher's delivered clipped-KL budget. The transition
normalization statistics refresh after held-out evaluation but before the current fit, so
the fitted head and the statistics used to decode it remain a consistent pair.

Both arms log post-rationalization channel nats, realized/query teacher KL, adapter residual
and gradient norms, final teacher-to-student KL reduction, entropy, conditioning gaps, and
clip rates. The hybrid additionally logs held-out R2/gate, pathwise alignment, achieved
rotation, clamp fraction, dose scale, relative dose error, and transition loss.

Both sources pass `py_compile` and Pyright diagnostics. Focused checks verify bit-exact v6
base parameters and post-construction RNG state, exact teacher/student identity at adapter
initialization, explicit present-zero context, exhaustive/disjoint parameter ownership, and
adapter-only clone versus base-only distillation gradients. Synthetic Fisher-plane checks
achieve cosine `0.95` within `2.98e-7` and match both over- and under-delivered clipped-KL
budgets to numerical zero. Independent review found and prompted the transition-statistics
ordering correction above, then confirmed that correction and found no other material
defect.

| arm | source SHA-256 | MLQ job |
|---|---|---|
| `opsd_residualteacher_v1` | `18bb20b91521f76a98c74aa5399d657b2b5edc7df0af2d95d1ac21b19a6017ff` | `4229` |
| `opsd_residualteacher_jac_v1` | `9aff8391adccc417ff5b770ba7dffd91a8e49b25052f0bf3a787ca1d00bffbff` | `4230` |

Jobs `4229` and `4230` execute the hash-addressed read-only snapshots on HalfCheetah-v4,
seed 1, for 8M steps with 16 envs x 2048, 128 minibatches, four actor and critic epochs,
EMA-RMS conditioning, margin 1, CUDA, deterministic async envs, and compiled acting. Each
declares `maxParallelRuns=3`, priority 0, and one attempt; the time limit is deliberately
omitted, with no autocull or W&B. Both entered queued behind higher-priority job `4228`.
Results require a joint postmortem before any further iteration.

### 17.18 RESIDUAL TEACHER POSTMORTEM - direction, not isolation, 2026-08-29

The user reported the result clearly: the Jacobian arm was substantially better, while the
residual-teacher arm underperformed. Both jobs were cancelled once the comparison was
decisive, at 4.192M and 5.168M steps respectively, rather than spending the remaining
compute. Matched-step returns show the crossover:

| arm | @500k | @1M | @2M | @3M | @4M |
|---|---:|---:|---:|---:|---:|
| residual teacher | 562.4 | 992.0 | 1213.7 | 2487.9 | 2478.1 |
| residual teacher + Jacobian | 547.8 | 757.5 | 3572.0 | 4682.9 | 5235.5 |

The pure arm's early lead rules out a simple startup failure. It learned a privileged channel,
but that channel did not produce sustained policy improvement. At 4M it had the larger
adapter residual RMS (`0.7801` versus `0.5331`), actor gradient norm (`2.43` versus `2.09`),
and distillation loss (`0.0334` versus `0.0211`), yet less than half the return. More teacher
movement or a stronger optimization signal was therefore not the missing mechanism.

The Jacobian intervention was both active and clean at 4M: held-out transition R2 was
`0.9856`, the gate was fully open, the achieved OPSD cosine was exactly `0.95`, dose scale
was `0.9863`, relative dose error was `1.05e-5`, and no action dimension hit the boundary
clamp. Its pathwise direction had cosine only `0.03495` with the residual-OPSD displacement.
The arm therefore supplied nearly orthogonal information rather than rediscovering the
privileged action-credit direction or winning through extra KL dose.

Both arms exposed a common student-tracking weakness. Fixed-teacher KL reduction was
negative at 4M, but much worse without the Jacobian (`-0.02531` versus `-0.009653`): the
joint value update through the shared trunk moved the final policy farther from its teacher
in aggregate. That does not erase the useful component of the actor gradient, but it means
canonical gradient isolation alone is insufficient on this chassis. The pure-arm hypothesis
is falsified. The supported mechanism is the held-out, one-step pathwise direction; the next
research question is how to preserve that direction against shared-trunk critic drift without
adding Q-learning, search, or extra teacher dose. No implementation or follow-up benchmark is
started pending postmortem review.

### 17.19 CRITIC-SPLIT JACOBIAN v2 - protect the supported direction, 2026-08-30

`cleanrl/opsd/core/teachers/ppo_continuous_action_opsd_residualteacher_jac_criticsplit_v2.py` tests the specific
bottleneck exposed in §17.18. The actor retains the v1 zero-context trunk and Beta heads, while
the value function receives a private ThinkTrunk initialized as an exact copy of the actor
trunk. Actor, critic, and residual adapter have disjoint exhaustive parameter sets and separate
Adam states. Distillation therefore cannot update the critic, and HL-Gauss regression cannot
update or globally clip the actor. The adapter clone pass and four actor/critic passes are
unchanged. All rollout values, bootstrap values, and the Jacobian's `V(s')` use the private
critic. There is still no PPO, Q function, search, extra teacher pass, or extra teacher dose.

**Hypothesis.** The v1 Jacobian direction was useful but partially erased by value gradients
through the shared trunk. Removing that write path should make fixed-teacher KL reduction
positive while retaining the parent arm's held-out R2, exact `0.95` rotation, and matched
clipped-KL dose. If tracking becomes positive without a matched-step return improvement over
job `4230`, shared-trunk interference was visible but not the score-limiting bottleneck; critic
separation should then be rejected rather than tuned.

The source passes `py_compile` and Pyright diagnostics. Focused checks establish exact parent
actor/critic outputs, sampled actions, construction RNG state, and actor/head parameters at
initialization; exact copied-trunk equality; exhaustive disjoint ownership; and adapter-only,
actor-only, and critic-only gradients for their three losses. Independent review found one
non-default schedule drift: the TransitionHead would have trained during critic-only epochs
when `critic_epochs > actor_epochs`. Gating that update on `do_actor` restores the parent's
schedule; re-review confirmed the correction and found no material remaining issue.

The benchmark runs the read-only snapshot
`f21432aa62e472a65116c25201039876f82b87e7a2803e1ef90603cd96eb80ea` as MLQ job `4258`:
HalfCheetah-v4, 8M steps, seed 1, 16 envs x 2048, 128 minibatches, four actor and critic
epochs, `jac_cos=0.95`, EMA-RMS conditioning, margin 1, async envs, and compiled acting.
It declares `maxParallelRuns=3`, priority 0, and one attempt. The time limit is deliberately
omitted; there is no autocull or W&B. The job entered `running` immediately. Results require
a mechanism-and-score postmortem before another version is started.

### 17.20 CRITIC-SPLIT RESULT - close the residual-teacher line, 2026-08-30

The user reported that job `4258` underperformed and was not an ablation worth further
compute. It had already been cancelled by request when cancellation was attempted here, at
3.92M recorded steps. Matched returns were `368.0 / 962.0 / 3608.9 / 5039.7` at
500k / 1M / 2M / 3M. The split arm modestly exceeded job `4230` late, but both remain far
below the measured 10,362 ENT-PPO chassis (`7281` at 3M). Plumbing around the same teacher
is therefore the wrong scale of intervention.

The intervention also falsified its own mechanism. At 3M the actor and critic trunks had
diverged by relative distance `0.7756`, yet fixed-teacher KL reduction remained negative
(`-0.003689`). The Jacobian path was healthy: held-out R2 `0.9796`, gate `1`, rotation
`0.95`, and dose relative error `6.53e-6`. Critic writes were removed and the student still
moved away from the frozen teacher in aggregate. Shared-trunk critic interference was visible
in v1 but was not the algorithmic bottleneck.

This closes the residual-posterior teacher line. Its observational one-step posterior is not
a reliable long-horizon policy-improvement operator, and per-state distillation is itself
poorly realized across minibatches. No critic split, loss-weight change, tracking correction,
or other local teacher ablation is warranted. The next intervention stays with the OPSD
grounding problem but removes the teacher: learn the actor's gradient rule directly from
critic-predicted post-update score.

### 17.21 CRITIC-META OPSD v1 - ground the actor gradient in predicted score, 2026-08-30

**Correction to the research direction.** Standard OPSD has a grounding problem because the
latent must jointly discover desire and competence. GRPO supplies objective outcome grounding
but still prescribes a score-function update. Neither is the optimum sought here. The
intervention changes how the actor is given gradients: a learned scalar loss chooses actor
gradients according to critic-predicted post-update score. There is no posterior teacher,
outcome conditioning, candidate ranking, or contrastive actor objective. Advantage-weighted
likelihood is optional evidence, not the definition of improvement.

**Hypothesis.** Direct critic score supplies an immediately grounded actor direction, while a
meta-learned loss can improve its geometry and exploration behavior by optimizing the return
predicted after the actual optimizer step. This can escape both OPSD's latent semantics and
GRPO's fixed gradient form. The hypothesis fails if the learned direction cannot outperform
the identical direct-Q optimizer step under an independent critic, even when both critics fit
fresh Bellman targets.

`ppo_continuous_action_criticmeta_opsd_v1.py` implements the test:

- A reparameterized tanh-Gaussian action remains inside the actor graph. The inner frozen
  Bellman critic supplies both Q and live dQ/da; critic parameters receive no actor/meta
  gradients.
- The learned loss emits Q, transformed-entropy, mean-coordinate, log-scale-coordinate, and
  optional TD-likelihood terms. TD is absent from all shared features and can enter only the
  explicit likelihood branch. Zero output initialization gives Q weight 1 and every optional
  path weight 0, making the initial loss exactly unscaled `-Q`.
- The outer objective differentiates through global-norm clipping and the exact next
  functional Adam step. A second critic has an independent initialization, optimizer, target
  critic, and replay draws; its frozen snapshot alone scores the post-update actor. This
  prevents the critic that supplied dQ/da from validating its own action extrapolation.
- A direct-Q Adam shadow uses identical actor states, optimizer state, clipping, meta states,
  and action noise. Gradient cosine, parameter-update cosine, and independent predicted gain
  expose whether learned rotation actually beats direct score ascent.
- The actor update commits only when independent predicted gain is positive and exact
  transformed-Gaussian KL is at most 0.03. KL-safe negative proposals still train the loss
  generator. The committed actor is checked against the scored shadow to 1e-6.
- Separate diagnostics report both critics' fresh normalized Bellman RMSE, logged/actor/shadow
  disagreement, each actor-loss term's gradient norm, coefficient mean/absolute mean/RMS,
  accepted updates, independent predicted gain, direct-Q gain, and next-fitted-critic
  confirmation. Confirmation is emitted only when the previous actor proposal was accepted.

**Pre-launch evidence.** Pyright reports 0 errors. The focused CUDA mathematical check proved
exact initial `-Q`, exact initial direct/learned gradients, TD isolation from every shared
output, identical direct/learned Adam shadows at initialization, finite nonzero outer
meta-gradients (`L1=0.213381`), independent critic parameters, and committed/shadow maximum
error `1.49e-8`. Independent adversarial review returned GO with no high or medium correctness
finding. Its accepted-only confirmation finding was fixed; its remaining low risk is that
repeated KL-unsafe proposals do not meta-train, so `meta/meta_update_applied` must be watched.

**Pre-registered decision rules.**

1. Mechanism validity requires finite nonzero `critic/action_gradient_rms`, both fresh
   normalized Bellman RMSEs materially below 1, bounded critic disagreement, and positive
   next-fitted-critic confirmation on accepted updates. NaN, stalled learning, or repeated
   disagreement growth invalidates the run rather than counting as a score result.
2. Learned-loss value requires `meta/learned_minus_direct_predicted_gain` to be nonnegative
   over sustained windows once the loss departs from initialization. If it remains negative,
   critic grounding works only as direct Q ascent and the meta-loss claim fails.
3. Score comparisons use the measured ENT-PPO chassis: 1238 / 3041 / 5465 / 7281 at
   500k / 1M / 2M / 3M and 10,362 at 8M. Two matched checkpoints more than 5% below the chassis
   are a failure, but no performance cull occurs before 3M because exploration-heavy methods
   can be late. An 8M last-20 return above 10,785 clears the established 423-point lineage
   spread; a result inside that band is a tie.

The read-only benchmark snapshot is
`eac60e4222f4f24469e8765f6968f8ea11c235cdd7f5a796300b88c8d27eecd2`, submitted as MLQ job
`4305`: HalfCheetah-v4, 8M steps, seed 1, 16 environments, 16 rollout steps, four updates per
critic per iteration, compiled rollout/Bellman functions, and eager second-order actor/meta
path. It declares `maxParallelRuns=1`, priority 0, and one attempt because the doubled critics
and higher-order gradient have not been coexistence-characterized. The time limit is
deliberately omitted; there is no autocull or W&B. It is queued behind the machine's protected
work.

### 17.22 CRITIC-META v1 RESULT AND v2 - remove gradient admission gates, 2026-08-30

The user reported that job `4305` plateaued. It was cancelled at 1.92M steps rather than spend
more compute on a structurally stalled actor. Last-20 return was `273.1 +-10.2`, with matched
returns `267.1 / 269.7` at 500k / 1M. This is not a weak version of the intended mechanism; the
actor was mostly prevented from receiving it.

The critics were not the immediate failure. At 1.9M, primary and independent fresh normalized
Bellman RMSE were `0.381 / 0.368`, action-gradient RMS was `2.776`, and independent predicted
gain was positive `0.0536`. But proposed KL was `0.0402` against the `0.03` gate, so both
`actor_update_accepted` and `meta_update_applied` were zero. The same zero-update state appears
at the sampled 100k, 500k, 1M, 1.5M, and 1.9M checkpoints. A positive critic score could not
train the actor or loss generator because an unrelated binary admission rule overrode it.
Predicted-success gating had the same structural defect: noisy critic sign was used to discard
the gradient instead of train the gradient generator. The user's diagnosis is the operative
one: gating on KL and success was the main issue.

**v2 decision.** Critic score is a continuous training signal, not permission to learn.
`ppo_continuous_action_criticmeta_opsd_v2.py` removes `max_actor_kl` and both admission
conditions. Every finite learned actor gradient is clipped, shadow-scored, and committed;
every finite outer gradient updates the learned loss. Predicted gain, critic disagreement,
next-critic confirmation, and exact transformed-Gaussian KL remain diagnostics only. Bad
proposals are necessary meta-training data: the outer loss must change the future gradient,
not retroactively erase the current one.

The correction preserves the grounding contract: exact initial `-Q`, live inner dQ/da with
frozen critic parameters, independent outer critic, exact differentiable Adam, direct-Q
shadow comparator, and committed-shadow parity. The focused CUDA check constructed a proposal
that both old gates would reject (`KL=0.032008`, independent gain `-3.31e-5`) and proved that
v2 still updated both systems: actor/shadow error `3.73e-9` and learned-loss parameter L1
change `0.39137`. Pyright reports 0 errors. Independent review returned GO and verified one
unconditional actor step and one unconditional meta step per learned iteration, with no
remaining hidden gate.

The read-only v2 snapshot is
`10f6cbacda3f4fa6588c0ae456c4076d5bc5148326967afd726851ccad05ae62`, submitted as MLQ job
`4309`: HalfCheetah-v4, 8M steps, seed 1, 16 environments, 16 rollout steps, four updates per
critic per iteration, compiled rollout/Bellman functions, and eager second-order actor/meta
path. It declares `maxParallelRuns=1`, priority 0, one attempt, no time limit, no autocull, and
no W&B. It is queued behind job `4308`.

**v2 gate.** `meta/actor_update_applied` and `meta/meta_update_applied` must remain exactly 1
after critic warmup; any other value is an implementation failure. Score and mechanism
criteria otherwise remain §17.21's direct-Q, next-critic, Bellman-fit, and chassis comparisons.

### 17.23 CRITIC-META v2 RESULT AND CRITIC-METRIC v3 - eliminate gradient bypass, 2026-08-30

The user reported that ungated job `4309` clearly collapsed. It was cancelled by request at
3.60M steps. Returns were `-290 / -538 / -129 / -603 / -603` at
100k / 250k / 500k / 750k / 1M, and last-20 return was `-602.3 +-0.5`. Removing the v1 gates
was necessary, but v2 exposed that the learned loss was not structurally critic-grounded.

The terminal behavior identifies the failure exactly. By 750k, action saturation was `1.0`,
mean/min/max log standard deviation were all `-2.3026 = log(0.1)`, and raw reward was exactly
`-0.6` per step: six boundary actions paid their control cost while producing no forward
motion. Full-step KL rose `0.074 -> 0.116 -> 0.965 -> 1,794 -> 74,711` from
100k / 250k / 500k / 750k / 1M.

The gradient-path telemetry supplies the cause:

| actor-gradient norm | ~100k | ~250k | ~500k | ~750k | ~1M |
|---|---:|---:|---:|---:|---:|
| critic score | 3.00 | 0.67 | 4.49 | **0** | **0** |
| free coordinates | 1.78 | 2.22 | 4.15 | 73 | 675 |
| optional likelihood | 0.94 | 0.30 | 1.41 | 33 | 32 |
| transformed entropy | **7.02** | **13.69** | **11.03** | **1,006** | **9,747** |

The learned entropy coefficient became `-0.790 / -0.824` at 750k / 1M, explicitly minimizing
transformed entropy toward the tanh boundary. Once saturated, the critic path vanished through
the tanh Jacobian, while entropy, direct-coordinate writes, and Adam momentum remained live.
The meta-learner exploited its frozen one-step scorer with gradients that bypassed dQ/da.
Independent critic initialization did not remove shared replay/model extrapolation error.

**v3 hypothesis.** Learned actor geometry can remain expressive without allowing a
critic-independent gradient. `ppo_continuous_action_criticmetric_opsd_v3.py` replaces the
entire scalar loss generator with a per-state positive-definite action-space metric:

`actor direction = action Jacobian transpose * metric * dQ/da`.

The metric is `(I + R)(I + R)^T + epsilon I`, trace-normalized to action dimension. Its output
starts at zero, so the metric starts at identity and the actor gradient is exactly direct
`-Q`. Its derivative at identity is live. Every possible learned direction has positive
first-order dot product with dQ/da, and if dQ/da vanishes every current actor gradient
vanishes. There is no entropy, likelihood, mean-coordinate, or log-scale branch; distribution
scale changes only through the critic-scored reparameterized action.

Every finite actor and metric update still applies. Adam directions below policy KL `0.01`
remain unchanged; larger directions are continuously rescaled to the cap rather than
rejected. The active-cap scale is solved to absolute tolerance and carries the implicit
meta-gradient of `KL(scale, metric) = 0.01`; Adam moments advance once before parameters are
overwritten with the scored capped shadow. The direct-Q comparator uses the same root solve
without building the unnecessary implicit-gradient graph. Fallback root solves are counted.

The outer objective is validation-critic predicted gain minus the absolute disagreement with
training-critic predicted gain. Subtracting pre-update score cancels absolute critic
calibration, while disagreement penalizes shared extrapolation continuously rather than
creating another success gate. Critics train from replay size 2,048; actor and metric updates
start at 25,600 so the scorer is fitted before defining actor geometry. This is a data
precondition only: after it, all finite updates apply. The threshold is evidence-based—the v1
static-policy critics improved from normalized Bellman RMSE `~0.997` at 2k to `~0.563` at 25k.

**Pre-launch proof.** Pyright reports 0 errors. Focused CUDA checks measured identity-metric
error `1.19e-7`, initial direct-gradient error `5.82e-11`, exactly zero actor gradient when
dQ/da was zero, positive perturbed-metric minimum eigenvalue `0.344`, positive minimum ascent
dot, full/applied KL `28.824 / 0.010000002`, and exact shadow commit. A deliberately negative
conservative-gain proposal still produced finite nonzero metric gradients. The implicit cap
gradient matched central finite difference with relative error `0.00185`; an inactive cap
kept scale exactly 1. Independent adversarial review returned GO after verifying the cap
solve, implicit derivative, optimizer state/order, confirmation state, conservative objective,
direct comparator, warmup, and absence of hidden gates or gradient bypasses.

**Pre-registered mechanism rules.**

1. After 25,600 steps, actor and metric update-applied metrics must remain exactly 1.
2. Applied KL must never exceed `0.010001`; cap fallback rate, step scale, and full-step KL are
   diagnostics, never admission criteria.
3. Minimum metric eigenvalue and ascent dot must remain positive within numerical tolerance;
   trace must remain 6. Action alignment or trace failure invalidates the implementation.
4. Sustained action saturation above `0.95` together with raw reward near `-0.6` is the
   already-proven terminal collapse and warrants immediate cancellation.
5. Learned value requires conservative gain above its identically normalized direct-Q shadow
   and positive next-fitted-critic confirmation over sustained windows. Score rules remain the
   §17.21 chassis checkpoints and 10,785 endpoint pass.

The read-only v3 snapshot is
`6ab57dcc605a44521e415d6d578b266c8e8a03ff1dd70e0b3382f336b7aaef34`, submitted as MLQ job
`4315`: HalfCheetah-v4, 8M steps, seed 1, 16 environments, 16 rollout steps, four updates per
critic per iteration, compiled rollout/Bellman functions, and eager second-order metric path.
It declares `maxParallelRuns=1`, priority 0, one attempt, no time limit, no autocull, and no
W&B. It is queued behind job `4312`.
