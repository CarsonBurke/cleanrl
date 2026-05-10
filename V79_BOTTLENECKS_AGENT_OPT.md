# V79 Agent / Optimization Bottleneck Report

Scope: `cleanrl/hlgauss/transformer/ppo_continuous_action_hlgauss_relusq_dlogstd_lewm_obsrecurrent_beta_asymspo_actionsa_axialrope_perceiver4readout_obsid_streamdream_v79.py` and the active HalfCheetah run `lewm_actionsa_axialrope_perceiver4readout_obsid_bootstrapfix_v79_8m`.

Run snapshot: the run was active while this report was written. It moved from about 3.8M to 4.5M real env steps during analysis; `scripts/score_runs.py v79 --runs-dir runs --last 20` fluctuated from the mid-400s to high-500s. At the last read, 4.528M steps, last-20 return was about 464 and last-50 about 360. This is still a large recovery over v76-v78 local runs, but the agent is far below a normal HalfCheetah PPO learning curve and remains high variance.

## Evidence Snapshot

| Signal | Recent value | Read |
|---|---:|---|
| `charts/episodic_return` | latest last-20 mean 464; observed 445-574 while active | Improved vs v76-v78 but unstable and low for 4M+ HalfCheetah steps. |
| `losses/explained_variance` | last-20 mean 0.06 | Real critic is barely explaining real GAE returns. |
| `losses/imagine_explained_variance` | last-20 mean 0.91 | Dream critic fits model returns, not necessarily useful returns. |
| `returns/real_gae_minibatch` | recent about 0.07 | Real targets are positive but shrinking. |
| `returns/dream_lambda_minibatch` | recent about -0.08 | Dream targets push a different objective. |
| `imagination/model_episode_return_mean` | last-20 mean -0.21 | Dream rollouts are still net negative. |
| `imagination/learn_weight_mean` | last-20 mean 0.18 | 16-step dreams are heavily survival-discounted. |
| `losses/approx_kl` / `cleanrl_approx_kl` | about 0.039 / 0.040 | Real updates still move the joint policy materially. |
| `losses/clipfrac` / `action_clipfrac` | about 0.30 / 0.056 | Joint ratio is often outside PPO clip even when per-action ratios are not. |
| `losses/imagine_approx_kl` / `imagine_clipfrac` | about 0.014 / 0.18 | Dream updates are smaller by KL but still often joint-ratio clipped diagnostically. |
| `real_rollout/actor_std_mean` | about 0.24 | Policy is already fairly narrow in env action units. |
| `imagination/actor_std_mean` | about 0.18 | Dream states see an even narrower policy. |
| `imagination/raw_actor_beta_head_abs_mean/max` | about 10.9 / 42.3 | Beta heads are high-magnitude and likely over-concentrated. |
| `imagination/actor_mean_abs_mean/max` | about 0.49 / 0.94 | Means are biased toward large action magnitudes; near-bound means exist. |
| `dynamics/reward_action_sensitivity_std` | about 0.011 | Reward remains nearly action-insensitive. |
| `dynamics/latent_action_sensitivity_std` | about 3.55 | Actions move latents, but not reward. |
| `charts/imagined_steps` | 70.3M vs 4.53M real | About 15.5 imagined samples per real sample have been generated. |

## Bottlenecks

### 1. Actor/Critic Readout Is Shared At The Most Sensitive Interface

The agent uses one Perceiver readout module with 8 learned queries, then splits the first 4 tokens for the actor and the last 4 for the critic. Actor and critic have separate input RMSNorms and heads, but the readout block and query tensor are shared and clipped as one `agent_readout` group. With recent real critic explained variance near 0.06 and imagined critic explained variance near 0.91, the shared readout is being asked to serve two value distributions with very different grounding quality.

Risk: critic gradients can shape the common readout into dream-return features while the actor depends on the same readout for real action probabilities. The split queries do not guarantee split representation because attention weights, key/value projections, and output layers are shared.

Actionable experiments:
- v80-agent-a: separate actor and critic Perceiver readouts, keeping the same 4-token width and heads. Success criterion: real explained variance rises above 0.25 without lowering last-50 return.
- v80-agent-b: keep shared readout but add readout diagnostics: actor/critic query attention entropy, actor-vs-critic feature cosine, and separate readout grad norms from actor loss vs value loss.
- v80-agent-c: freeze the readout for imagined updates and train it only on real PPO for 500k-1M steps. If returns improve, dream critic gradients are contaminating the shared interface.

### 2. Real Value Learning Is The Clearest Agent-Side Failure

The real critic is weak despite v79's outcome-aware bootstrap fix. Value loss sits around 1.2 and recent real explained variance is only about 0.06, while the dream critic explains model returns very well. This says the critic architecture can fit a target, but either the real target interface is too noisy, the HL-Gauss symlog support is poorly matched to real returns, or imagined critic updates dominate the value head toward model-return geometry.

Actionable experiments:
- Add a scalar value head in parallel with the HL-Gauss head for diagnostics and optionally real-value loss. Use it only for real rollout value prediction at first; keep HL-Gauss for dream returns.
- Run a critic-isolation ablation: real actor loss on, real value loss on, imagined actor loss off, imagined critic loss off. This separates "weak critic because representation is bad" from "weak critic because dreams are overwriting it."
- Increase real critic update ratio without changing actor updates: one extra value-only pass over real latents per rollout. Success criterion: real EV improves and real GAE minibatch variance falls without KL inflation.

### 3. Real And Imagined Updates Are Balanced By Gradient Steps, Not By Trustworthy Signal

Each rollout uses 4 real agent epochs and 4 imagined epochs. With 16 envs and 2048 steps, the real batch is about 32.8k samples. The dream batch is about 32.5k valid starts times horizon 16, or about 520k dream states per rollout. Minibatch size is scaled by horizon, so real and dream have a similar number of optimizer steps, but dream steps carry roughly 16x more samples and currently optimize near-zero or negative model returns.

Evidence: `returns/real_gae_minibatch` is positive around 0.07, while `returns/dream_lambda_minibatch` is about -0.08 and `imagination/model_episode_return_mean` is about -0.21. The dream critic learns these targets well, which can make the combined agent confident in the wrong auxiliary objective.

Actionable experiments:
- v80-update-a: halve `imagine_actor_coef` to 0.25 or 0.5 and keep `imagine_critic_coef` fixed. Success criterion: real return improves while dream EV may remain high.
- v80-update-b: train imagined critic only for 500k steps, no imagined actor. If performance improves, imagined actor gradients are the harmful component.
- v80-update-c: gate imagined actor updates by dream target quality: require positive `model_episode_return_mean` or reward-action sensitivity above a threshold before applying imagined actor loss.

### 4. SPO Is Not PPO, And Current KL Semantics Hide The Trust Region

The real and imagined policy losses use asymmetric half-strength SPO: `-(A * ratio - |A| * (ratio-1)^2 / (2 eps))`. This is a smooth ratio penalty, not PPO's clipped surrogate. The code logs PPO-style `clipfrac`, but clipping is diagnostic only. The effective eps values, 0.40 and 0.56, are much wider than standard PPO's 0.20 and asymmetric with respect to whether policy drift agrees with advantage sign.

The KL metrics also mix semantics. `approx_kl` is the sum of per-action `((r_i - 1) - log r_i)`, while `cleanrl_approx_kl` is computed from the joint ratio. Both are useful approximations, but neither is true Beta-distribution KL because old alpha/beta parameters are not stored. With joint clipfrac around 0.30 and historical summed logratio maxima above 3, the real policy can move much more than standard PPO would allow.

Actionable experiments:
- Add a pure PPO clipped-surrogate ablation with the same Beta policy and v79 bootstrap fix. If PPO wins, SPO is too permissive for this architecture.
- Store old Beta alpha/beta or mean/concentration and log true `KL(old_beta || new_beta)` per dimension and summed. Use that for early stopping or adaptive SPO eps.
- Try per-dimension SPO with a summed objective instead of one joint ratio. This matches the current action-level diagnostics and may avoid joint-ratio explosions in 6D action space.

### 5. Grouped Grad Clipping Helps, But The Shared Optimizer Still Couples Objectives

v79 clips four groups separately: world model, agent readout, actor, critic. This is better than one global norm, but it does not isolate actor and critic through the shared readout, and all parameters still share one Adam optimizer schedule and moment history. During real updates, the loss combines actor and value terms before one backward pass. During dream updates, the same readout receives imagined actor and imagined critic gradients from model targets.

Actionable experiments:
- Split optimizers: `wm_optimizer`, `actor_optimizer`, `critic_optimizer`, and `readout_optimizer` or separate actor/critic readout optimizers. Keep the same LR initially.
- Log unclipped and clipped grad norms per group, plus update/weight norm ratios. Without these, grouped clipping is only an assumption, not evidence.
- Alternate actor and critic backward passes for real updates to measure conflicting readout gradients before summing them.

### 6. Beta Policy Is Becoming High-Concentration And Low-Entropy

The Beta policy constrains alpha and beta to be at least 1, so it cannot represent U-shaped boundary-seeking distributions; it represents uniform-to-unimodal policies. Current diagnostics show large raw Beta heads, low action std, negative differential entropy, and mean actions near bounds even though sampled action saturation is low. This is a "peaked near a strong mean" regime, not broad exploration.

Risk: the policy can become confident before the critic is accurate. In a weak-value regime, SPO then amplifies noisy advantage signs through a narrow distribution. The Beta parameterization may also make boundary control awkward: it can place a mode near a boundary, but cannot explicitly prefer both extremes or maintain high edge probability.

Actionable experiments:
- Reparameterize as mean plus concentration: `mean=sigmoid(m)`, `concentration=softplus(c)+2`, with concentration cap or penalty. Log concentration directly.
- Add a low coefficient entropy or concentration regularizer until real EV exceeds a threshold.
- Compare against tanh-normal on v79 with no other changes. If tanh-normal improves early control, the Beta policy is a bottleneck rather than the world model.

### 7. Action-Token Policy-Stat Conditioning May Leak Policy Identity Into Dynamics

Predictor action tokens contain `[sampled action_z, policy mean_z, policy std_z]`. This is richer than action conditioning and likely helped v79 recover latent action sensitivity. But it also means the model's transition token depends on the behavior policy's statistics, not only on the executed action. After policy updates, the same state/action can produce different predictor inputs because mean/std changed.

Risk: the world model can learn "what this policy tends to do" rather than environment dynamics. That can reduce counterfactual validity exactly where imagined actor updates need it.

Actionable experiments:
- Policy-stat dropout: randomly replace mean/std channels with detached constants during WM training and dreams.
- Action-only ablation for dream rollouts, while retaining policy-stat conditioning for teacher-forced training diagnostics.
- Store and compare counterfactual predictions for fixed state/action under old vs current policy stats. The reward delta should be near zero if the model is environment-dynamic rather than policy-identity-dynamic.

### 8. Dream Buffer Economics Are Expensive For The Current Signal Quality

v79 stages the fixed dream buffer on CPU and streams minibatches back to CUDA. For HalfCheetah, each rollout builds roughly 520k dream states. The state tensor alone is about `520k * 10 tokens * 64 dims * 4 bytes`, or about 1.3 GB, before actions, logprobs, values, returns, masks, and Python list overhead. The latest run has generated about 70M imagined samples, but those samples have negative model episode return and low survival weights.

This is a poor compute trade until dream rewards become useful. The total-SPS metric counts imagined samples and looks high, but wall-clock agent improvement is still real-env-return limited.

Actionable experiments:
- Reduce `imagine_horizon` to 8 until `model_episode_return_mean` turns positive. This halves memory and should reduce low-weight tail updates.
- Stream-generate dream minibatches per epoch instead of materializing the full buffer, or keep the fixed buffer but pin CPU memory and use non-blocking transfers.
- Prioritize valid/high-weight dream steps instead of training uniformly over all horizon positions. Current `learn_mask_frac` is 1.0 but `learn_weight_mean` is only 0.18, so many samples are nearly dead.

## Priority Next Experiments

1. **v80_ppo_beta_sepvalue**: keep v79 architecture, replace SPO with PPO clipped surrogate, add scalar real-value auxiliary head, and log true Beta KL by storing old alpha/beta. This directly tests whether optimization semantics, not architecture, are blocking real PPO learning.
2. **v81_sepreadout_realanchor**: split actor/critic readouts and disable imagined actor updates for the first 1M steps after WM warmup. This tests whether dream actor gradients and shared readout coupling are hurting the real anchor.
3. **v82_concentration_reg**: mean/concentration Beta head with concentration cap/penalty and entropy until real EV > 0.25. This tests premature policy narrowing.
4. **v83_dream_gated_h8**: horizon 8, imagined actor enabled only when dream episode return is positive and reward-action sensitivity is above the recent v79 baseline. This tests whether dream updates become useful when low-quality model targets are filtered.

## Bottom Line

v79 fixed an important bootstrap bug and is clearly better than v76-v78, but the agent-side evidence points to weak real value learning, permissive non-PPO policy movement, shared readout gradient coupling, and overconfident Beta policies. Dream updates are currently expensive and internally learnable, but their targets are not aligned with real control. The next work should first restore a stricter real PPO/value anchor, log true Beta KL and grad norms, then reintroduce imagined actor pressure only when dream rewards are demonstrably control-relevant.
