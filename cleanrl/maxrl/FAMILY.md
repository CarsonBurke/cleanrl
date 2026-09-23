# MaxRL family

Maximum-likelihood RL for dense-reward continuous control, built on Beta-PPO.
Paper: `maxrl_2602.02710v3.pdf` (Tajwar et al., "Maximum Likelihood Reinforcement
Learning", arXiv 2602.02710v3). Earlier attempts are in `_archive/`, kept for provenance
only. They were not consulted when writing this family.

## The paper in five lines

- Setup: a hidden sequence z (e.g. a chain of thought) is sampled, then a deterministic
  decoder produces the answer, which is checked by a binary verifier. The success
  probability is p(x) = P(f(z) = y* | x).
- RL ascends p(x). Maximum likelihood ascends log p(x) = −Σ_k fail@k / k, and
  grad log p = Σ_k (1/k) grad pass@k. RL is the first-order term of this expansion.
- grad log p = E[score | success], the score averaged over the success-conditioned
  distribution of z. With N rollouts per prompt, averaging scores over the K successes
  is unbiased for the expansion truncated at T = N.
- In practice: per-prompt advantage (r − r̂)/r̂, where GRPO would use (r − r̂)/std.
  Weight on a prompt with pass rate p: w(p) = (1 − (1−p)^T)/p, versus 1 for RL.
- Non-binary rewards (App. M.4): log E[r] for r ≥ 0, one maze experiment. The paper
  lists continuous rewards, PPO-style off-policy training and multi-turn RL as future work.

## Translation to dense-reward MuJoCo

| paper | here |
|-|-|
| prompt x | state s_t |
| hidden sequence z | future action noise from s_t |
| deterministic decoder f | MuJoCo dynamics, deterministic given actions |
| success likelihood 1[f(z) = y*] | exp(β G), G the return from s_t |
| per-prompt mean r̂ over N rollouts | critic ψ(s): an amortized normalizer (the MC mean is unavailable) |
| truncation T | β (cumulant order) |

The five points that determine the design:

1. **Within a prompt, MaxRL points the same way as REINFORCE.** (r − μ)/μ is REINFORCE
   rescaled by 1/μ. So `A / V(s)` with a linear reward is only a per-state learning rate,
   and its sign breaks when V < 0. The success likelihood must be non-linear in the return.
2. **The exponential is forced.** Rewards have an arbitrary offset, so the objective must
   satisfy log E[g(G + c)] = const + log E[g(G)]. That is Cauchy's exponential equation,
   so g = exp(β·). The objective (1/β) log E[e^{βG}] is the cumulant generating function:
   mean + β var / 2 + ... Its first-order term is RL, which mirrors the paper's expansion
   with β playing the role of T.
3. **The latent-generation model fits MuJoCo exactly.** Given the actions, the dynamics
   are deterministic, so all return variance comes from the policy. Optimism is therefore
   over controllable outcomes, not luck-seeking. As β → ∞ the objective tends to the
   maximum return, which a deterministic optimal policy attains.
4. **Critic = per-prompt normalizer.** The LINEX loss (e^{bu} − bu − 1)/b², with
   u = target − ψ, is minimized exactly at ψ = (1/b) log E[e^{b·target}], so single
   samples suffice. Soft GAE over ψ gives the advantage A; the policy uses (e^{bA} − 1)/b.
   At the fixed point E[e^{bA} | s] = 1 for every state, which is MaxRL's per-prompt
   normalization. AWR and V-MPO instead take a batch-level softmax, which weights states
   by how good their samples look (the GRPO-like coupling across prompts).
5. **The multi-epoch objective is an M-step.** A weighted maximum-likelihood fit to the
   success-conditioned action distribution is bounded, and it sets the policy's spread to
   the spread of successful actions instead of shrinking it steadily.

Known approximations:
- With γ < 1 and GAE λ = 0.95, ψ is a mixture of the recursive soft value (λ = 0) and the
  non-recursive CGF (λ = 1). `maxrl/w_mean` measures the residual miscalibration.
- A critic error e(s) multiplies state s's weight mass by e^{−b e(s)}. That changes the
  weight across states but not the direction within a state. Errors at later states
  enter the exponent in an action-dependent way, which is the same class of bias as
  ordinary GAE with λ < 1.

## Versions

| version | idea | status |
|-|-|-|
| `ppo_continuous_action_maxrl_cgf_v1.py` | LINEX soft-value critic, advantage (e^{bA} − 1)/b in PPO's clipped surrogate; β = κ/std(A) per batch; ablations `critic_objective=mse`, `advantage_transform=linear`; actor and critic clipped separately | done |
| `ppo_continuous_action_maxrl_cgf_mstep_v2.py` | the clipped surrogate replaced by the weighted-ML M-step, e^{bA}/mean · log π, with decoupled mean/concentration KL budgets (MPO-style Lagrangians) | done, regression |
| `ppo_continuous_action_maxrl_cgf_mstep_cv_v3.py` | v2 with the "+1" self-fit term replaced by its exact expectation, (w−1) log π − KL(old‖π) | done, regression |
| `ppo_continuous_action_maxrl_difficulty_v4.py` | the paper's cross-state weight w_T(p(s)) on the dense GAE advantage; p(s) = Φ((V − τ)/σ), τ = a batch return quantile | done, regression |
| `ppo_continuous_action_maxrl_critic_group_v5.py` | the layer-cake continuous MaxRL, J_T = Σ_{k≤T} (1/k) E[max_k G], estimated over N actions per state scored by a learned A(s,a) head; PPO remains the k = 1 term, and only the higher-order excess is added | done; best MaxRL variant: N=32 T=8 = 8760; max-weighting beats a magnitude-matched linear term |

## Results (HalfCheetah-v4, 8M steps, seed 1, 16 envs)

Reference: `ppo_continuous_action` = 7468 (last 20 episodes), 4082 at 2M, 6067 at 4M.

### v1

| run | final (last 20) | @1M | @2M | @4M | entropy @8M | ESS frac |
|-|-|-|-|-|-|-|
| PPO reference | 7468 | 2250 | 4082 | 6067 | −4.49 | – |
| **linear_k2** (soft critic, PPO weighting) | **7958** | 3063 | 5097 | 6788 | −4.83 | 0.39 |
| k05 (exp weights) | 6317 | **3749** | **5329** | 6432 | −4.28 | 0.64 |
| k1 | 6328 | 2956 | 4667 | 5577 | −4.53 | 0.19 |
| k2 | 5661 | 1124 | 3371 | 4754 | −4.63 | 0.28 |
| k4 | 1284 | 516 | 537 | 1307 | −3.15 | 0.016 |
| mse_k2 (exp over a mean critic) | 6341 | 2143 | 3827 | 5627 | −5.01 | 0.0003 |

(Rows for jobs that crashed at 32k steps on the fixed shadowing bug are noise in `score_runs`.)

Readings:
- **The soft critic alone is the win** (+490 final; ahead at every checkpoint). With λ < 1,
  soft GAE equals V-GAE plus potential shaping by the risk premium c(s) = ψ − V ≈ β Var/2,
  roughly γ(1−λ) Σ (γλ)^l c(s_{t+l+1}) − c(s_t). Actions that lead to states with a wide
  spread of future returns get credit. Its entropy falls *faster* than PPO's, so it is
  directed optimism rather than extra noise.
- **Exponential weights in the clipped surrogate speed up early learning and then
  plateau.** k05 is the fastest of all runs to 2M, then flat at about 6.3k. The clip
  saturates the tail samples after about one step, while the bulk, at about −1 after
  centering, keeps pushing density down and spreads the policy (k05 has the highest
  entropy). That is a mismatch between the surrogate and the objective, which motivates v2.
- **The per-state normalization matters.** With the mean critic, E[e^{bA} | s] ≈ e^{b²Var/2}:
  log w_mean was 3.5 and ESS collapsed to 0.03%. It was worse than the soft critic at every
  checkpoint.
- Large κ collapses: at k4, ESS is 1.6%.

### Batch 2 and 3: the control overturns the v1 reading

| run | final | avg all (AUC) | @1M | @2M | @4M |
|-|-|-|-|-|-|
| **linear κ=0.01** (≈ PPO + separate actor/critic clipping, the control) | **8467** | **6634** | 3836 | 5940 | 7546 |
| linear κ=0.25 | 8352 | 6481 | 3807 | 5635 | 7327 |
| linear κ=0.5 | 7944 | 6183 | 3425 | 5138 | 6995 |
| linear κ=1 | 8315 | 6588 | 3588 | 5906 | 7637 |
| linear κ=2 | 7958 | 6096 | 3063 | 5097 | 6788 |
| linear κ=4 | 7488 | 5668 | 3009 | 4774 | 6130 |
| linear κ=8 | 6140 | 4464 | 2629 | 4066 | 4896 |
| v3 control-variate M-step κ=0.5 / 1 / 2 | 7701 / 6255 / 5428 | 5669 / 5362 / 3624 | 2317 / 2871 / 1118 | 4685 / 4870 / 2716 | 6384 / 6227 / 4313 |
| v2 M-step κ=0.5 / 1 / 2 | 5799 / 6412 / 4858 | 3864 / 4696 / 3535 | 1288 / 1812 / 1658 | 2432 / 3791 / 2880 | 4378 / 5398 / 4045 |
| PPO reference | 7468 | 5282 | 2250 | 4082 | 6067 |

- **Linear κ dose-response.** With the linear transform, κ affects only the soft critic's target. AUC falls roughly monotonically with κ: 6634 at 0.01, about 6500 for κ ≤ 1, about 6100 at κ = 2, then 5668 and 4464. κ = 1 is a single-seed bump inside the noise. So the risk-seeking critic target costs performance in proportion to κ; it never helps.
- **v2 against v3.** The control variate is worth about +1000 to +1800 AUC at matched κ. v3 κ=0.5 is the best M-step run, but it still trails the control by about 970 AUC, and it is slowest early (2317 at 1M against 3836). The weighted-ML fit learns more slowly than the clipped surrogate, because its multipliers never bound, so its step size was set by the learning rate alone.

- **The whole "soft critic" gain was separate gradient clipping.** Clipping actor and
  critic separately gives about +1000 on this seed and is unrelated to MaxRL. With a
  single global clip, critic gradient spikes shrink the actor's step. The soft critic
  adds nothing on top of it, and higher κ is monotonically worse. The shaping argument
  above was therefore wrong. From here on, the control is linear κ=0.01.
- **Exponential tilting of any form loses to the control:** clipped surrogate (v1), M-step
  (v2), and control-variate M-step (v3). The control variate clearly helped the M-step,
  but its Lagrange multipliers collapsed to about 1e-8, so the KL budgets never bound; the
  learning rate and the gradient clip set the step.
- **Diagnosis: shift invariance removes MaxRL's mechanism.** MaxRL's gains come from the
  cross-prompt weight w(p) = 1/p, which amplifies hard prompts, and that needs an
  *absolute* success scale. The exponential likelihood is shift-invariant by construction,
  so it has no notion of difficulty. For small β its gradient is β·Cov(G, score) per
  state, the same weighting across states as RL. All that remains is tilting toward the
  upper tail within a state, which does not help here. The paper's effect needs a real
  success event. That is v4.


### v4: the paper's cross-state weight on the dense advantage

| run | final | @1M | @2M | @4M | p mean | weight ESS |
|-|-|-|-|-|-|-|
| T=1 (control; reproduces linear κ=0.01 within 10) | 8457 | 3796 | 5896 | 7511 | 0.15 | 1 |
| T=16, q=0.9 | 8204 | 3294 | 5682 | 7446 | 0.14 | 0.77 |
| T=8, q=0.9 | 6491 | 3530 | 5086 | 5976 | 0.11 | 0.91 |
| T=4, q=0.9 | 6286 | 3501 | 4900 | 5536 | 0.14 | 0.97 |
| T=8, q=0.5 | 5975 | 2289 | 4474 | 5465 | 0.46 | 0.65 |
| T=64, q=0.9 | 5475 | 2967 | 4276 | 5110 | 0.10 | 0.61 |

Every T > 1 is slower at 1M than the control, and none beats it at 8M. The final scores
are not monotone in T; 6k-versus-8k plateaus look like the policy locking into different
gaits, which one seed cannot separate. The early-phase numbers are the cleaner signal, and
they say the weighting slows learning.

### v5: layer-cake MaxRL over critic-scored action groups

| run | final | avg all (AUC) | @1M | @2M | @4M | entropy @6M |
|-|-|-|-|-|-|-|
| N=16, T=16 | **8645** | 6985 | 3968 | 6281 | 7971 | −6.52 |
| N=16, T=8 | 8483 | **7081** | **4353** | **6851** | **8005** | −5.83 |
| control (v1 linear κ=0.01) | 8467 | 6634 | 3836 | 5940 | 7546 | −5.05 |
| N=16, T=1 (v5 code, no excess term) | 8351 | 6529 | 3460 | 5888 | 7420 | −5.05 |
| N=4, T=4 | 8294 | 6673 | 3944 | 6059 | 7661 | −5.95 |
| N=16, T=4 | 8199 | 6579 | 3814 | 5910 | 7565 | −5.80 |
| N=16, T=2 | 6066 | 5258 | 3685 | 5139 | 5878 | −6.64 |

- This is the first MaxRL variant ahead of the control. T=8 is ahead at every checkpoint, with +450 AUC. T=16 has the best final score. Every T > 1 is ahead of T=1 at 1M. T=2 looks like a gait lock-in: its entropy is the lowest and its score plateaus at 6k.
- **The predicted mechanism did not appear.** Theory says the policy spread should grow while the reward is still sloped. Instead entropy falls *faster* than the control's, and the Beta concentration is flat (about 22 against 21.6). The gain looks like critic-guided sharpening toward the best sampled actions.
- The A(s,a) head explains only about 2% of the variance of single-sample GAE. Part of that is the noise ceiling of GAE targets, but the group scores are rough.
**Ablation and scale (jobs 9133-9136):**

| run | final | avg all (AUC) | @1M | @2M | @4M | entropy @6M |
|-|-|-|-|-|-|-|
| N=32, T=8 | **8760** | 6958 | 3875 | 6274 | 7892 | −6.30 |
| N=32, T=32 | 8539 | 6782 | 3821 | 6074 | 7747 | −6.40 |
| linear, magnitude-matched to N=16 T=16 | 8094 | 6879 | 4481 | 6816 | 7870 | **−7.29** |
| linear, magnitude-matched to N=16 T=8 | 7301 | 5478 | 3669 | 4910 | 5940 | −5.82 |

- **Max-weighting beats a linear critic-group term of the same magnitude.** The final scores are 8645 against 8094 and 8483 against 7301, and the areas under the curve are 6985 against 6879 and 7081 against 5478. The linear term learns fast early and then plateaus. Max-weighting keeps improving.
- **The paper's "resists sharpening" claim holds relative to linear, not relative to PPO.** Any critic-group term sharpens the policy. The linear one sharpens most (entropy −7.3). Max-weighting at the same magnitude keeps entropy higher (−6.5). So the prediction that spread grows along improving directions shows up as *less collapse* than the linear term, not as more entropy than PPO.
- **Scale.** N=32 T=8 has the best final score of the family (8760, +290 over the control). Its early phase is slower than N=16 T=8. T=8 beats T=N at both group sizes. The paper's full T=N looks too aggressive here, and the useful order is set by T, not by N.
- **The A(s,a) head is still weak**, explaining 0.02-0.08 of the variance of single-sample GAE. It is the obvious bottleneck.
- **Caveat.** One seed, with bimodal gait plateaus (n16_t2 and linear-t8 both stuck near 6k). The ablation margins are large, but each one rests on a single run.

## Next

1. **Better group decoder (v6).** The groups are only as good as A(s,a).
   - Regress A(s,a) on a lower-variance target than raw GAE; for example, an n-step TD target with this head's own bootstrap, which gives an expected-SARSA-style Q.
   - Consider training on the group actions as well, e.g. a distributional A head or a short world-model rollout per candidate.
2. **Normalize the order term by its weight mass (≈ H_T − 1).** Then T changes only the shape, and step size stays constant. At present T is confounded with step size, and the flat T=4 result may reflect that.
3. **Keep T=8 and N=32** as the working point for v6. Retire T=N and the linear form.
4. **Check separate gradient clipping on Hopper and Walker2d** before promoting it to the shared baseline.

## Reconsideration: v1-v4 tested my translation, not MaxRL

The conclusion below says v1-v4 failed. It does not show that MaxRL fails.

- **No groups.** MaxRL's estimator averages scores over the K successes among N samples of the *same* prompt. Every version here had N = 1 sample per state, and with N = 1 the MaxRL estimator *is* REINFORCE. A critic mean is not a substitute, because the 1/K weighting is a statistic of the group itself.
- **"The exponential is forced" was wrong.** It assumed the objective has the form log E[g(G)]. The paper's actual structure is the pass@k expansion. Its faithful continuous version is layer-cake: integrate the binary objective over thresholds τ of the event {G > τ}. That gives J_T = Σ_{k≤T} (1/k) E[max of k draws of G]. It is shift-equivariant, so it needs no success scale and no exponential.
- **v4's success event was a batch-level artifact.** It was not a per-prompt pass rate.
- **Weak evidence.** One seed in a bimodal environment (6k versus 8k gaits), and in v2/v3 the trust region never bound.

**v5.** For each state, sample N actions, score them with an A(s,a) head regressed on GAE, and sort them. Rank i gets c_i = Σ_{j<i} Δ_j ω_T(N−j), where ω_T(K) = (1/N) Σ_{m<T} C(N−K,m)/C(N−1,m). Here ω_N = 1/K is the paper's weight and ω_1 = 1/N is REINFORCE.

Verified by exact enumeration (scratchpad scripts, subagent):
- The layer-cake identity and the binary estimator are exact. The paper's centered r/K − 1/N is unbiased for J_{N−1}, not J_N.
- The full estimator needs a q_(1)/N term. It vanishes for the *excess* (ω_T(N) = 1/N for every T), so v5's excess-only form is complete.
- Centering within the group costs an O(1/N) bias, accepted at N = 16.
- The mechanism: under a linear local reward, dJ/dlog σ > 0 for T > 1 (it is 0 for T = 1). Near a concave optimum, σ shrinks slightly faster than under RL. Away from it, σ grows. MaxRL therefore acts as *adaptive exploration*: it widens the policy while the reward is still sloped and sharpens it at the peak.
- The gradient scale grows about like H_T.

Remaining approximations: the learned A(s,a) scores, not rollouts, form the group, so the estimate is only as good as that head. The groups are one-step, and the continuation is on-policy.

## Conclusion (v1-v4)

Both of MaxRL's ingredients, *as I translated them*, were tested in dense-reward MuJoCo. Neither translation helped.

1. **Within-state likelihood tilting** (exponential success likelihood, v1-v3): it loses to
   the control in the clipped surrogate, the M-step and the control-variate M-step.
2. **Cross-state difficulty weighting** w_T(p) with an absolute success event (v4): it
   slows learning, monotonically in how aggressive the weight is.

**Why.** MaxRL targets sparse binary verification. There, RL's per-prompt signal
∇p vanishes as p → 0, and 1/p restores it. With dense rewards, GAE already gives every
state a well-scaled, per-state-centered signal. Reweighting states by difficulty then only
distorts the on-policy state weighting and adds variance. There is no vanishing signal to
recover. The exponential form's shift invariance removes difficulty altogether, which is
correct for dense rewards, and what remains is tail-seeking. On deterministic MuJoCo that
is not a better search direction than the mean.

**By-product (not MaxRL):** clipping the actor's and critic's gradients separately gave
8467 and 8457 against PPO's 7468 (+13%), ahead at every checkpoint (1M: 3.8k vs 2.3k).
With a single global norm of 0.5, the critic's gradient share shrinks the actor's step.
One seed and one environment so far; it should be checked on Hopper and Walker2d, and as a
change to the shared baseline, before anything is built on it.

Where MaxRL could still apply here: tasks with a genuine binary event and a vanishing
signal, e.g. Hopper and Walker2d survival early in training. That test was not run; the
standing instruction is HalfCheetah only.
