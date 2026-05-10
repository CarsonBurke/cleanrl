# V79 World-Model Bottleneck Report

Scope: `ppo_continuous_action_hlgauss_relusq_dlogstd_lewm_obsrecurrent_beta_asymspo_actionsa_axialrope_perceiver4readout_obsid_streamdream_v79.py` and the active HalfCheetah run `lewm_actionsa_axialrope_perceiver4readout_obsid_bootstrapfix_v79_8m`.

Run snapshot: the active v79 run is at about 3.4M / 8M env steps. `score_runs.py v79 --runs-dir runs --last 20` reports last-20 episodic return around 359-464 depending on the read instant, with recent per-rollout means mostly 280-464 and frequent failed episodes below 0 mixed with 500-650 episodes. This is a clear improvement over v78/v77/v76 in the local runs, but still far below a competitive PPO-quality HalfCheetah trajectory.

## Evidence

Latest v79 metrics observed near global step 3.40M:

| Signal | Latest / Recent | Interpretation |
|---|---:|---|
| `charts/episodic_return` | last-20 about 464, min historical -385, max 655 | v79 escapes the v78 collapse but remains unstable and low-score. |
| `lejepa/prediction_mse` | 0.88 | Overall JEPA loss has plateaued. |
| `lejepa/obs_prediction_mse` | 1.10 | Observation-token prediction remains the dominant latent error. |
| `lejepa/outcome_prediction_mse` | 0.0068 | Outcome tokens are easy to match in teacher-forced space. |
| `outcome_probe/reward_mse` | 0.029 | One-step decoded reward probe looks numerically decent. |
| `outcome_probe/reward_ce` | 2.97 | Reward distribution remains broad/uncertain despite low scalar MSE. |
| `dynamics/rollout_reward_step_mae` | 0.17 | Free-run one-step reward error is meaningful for HalfCheetah rewards. |
| `dynamics/rollout_reward_step_bias` | -0.13 to -0.14 | Free-run reward prediction is systematically pessimistic. |
| `dynamics/rollout_reward_sum_bias` | about -0.66 to -0.72 | Reward bias accumulates across the diagnostic horizon. |
| `dynamics/reward_action_sensitivity_std` | 0.011 | Reward is almost action-insensitive under random action perturbations. |
| `dynamics/latent_action_sensitivity_std` | 3.0+ | Latents move strongly with action, but reward decoding mostly ignores that movement. |
| `imagination/reward_mean` | about 0.01 recent | Dream rewards are near zero. |
| `imagination/model_episode_return_mean` | about -0.20 recent | Free-run dreamed episodes are not useful positive-control targets. |
| `imagination/continue_mean` | about 0.65 | Dreams decay quickly; 16-step rollouts carry low survival weight. |
| `losses/explained_variance` | about 0.0-0.1 recent | Real critic remains weak. |
| `losses/imagine_explained_variance` | about 0.91 | Dream critic fits model targets well, but those targets are not grounded enough. |

Local run comparison at latest available points:

| Run | Last-20 return | Key dynamics signal |
|---|---:|---|
| v76 FA | 55.8 | low SIGReg quality, low latent action sensitivity |
| v77 | 49.3 | better reward free-run error but weak reward/action sensitivity |
| v78 | 67.1 | full-summary SIGReg restored latent sensitivity but performance collapsed |
| v79 | 464.4 snapshot | bootstrap fix helps control, but free-run reward/continue drift remains |

## Bottlenecks

### 1. Outcome-Token Feedback Is Not a Sufficient Control Signal

v79 uses outcome tokens as part of the target/future summaries, but current world-model training still starts from neutral current summaries and skips current-slot SIGReg. Real acting uses previous outcome tokens when available, while WM training bootstraps from neutral online summaries. This is defensible for causality, but it creates a distribution split: the agent conditions on previous reward/continue context, while the predictor is trained from neutral current context at rollout starts.

The evidence is mixed: outcome-token MSE is excellent, but scalar reward CE remains broad and free-run rewards drift negative. The predictor can copy/match the outcome token geometry under teacher forcing, but that does not imply it learns a control-relevant reward manifold under its own generated summaries.

Actionable experiment:
- v80: train a second WM path from `encode_summary_with_optional_outcome(obs_t, r_{t-1}, c_{t-1}, has_outcome)` into `(obs_{t+1}, r_t, c_t)` while keeping the neutral path for first-step causality. Weight it 50/50 with the existing neutral-start loss. This directly matches the agent/dream input distribution without leaking current reward labels.

### 2. Free-Run Reward/Continue Drift Is the Main Imagination Failure

Teacher-forced outcome prediction looks good, but free-run diagnostics expose the bottleneck. Reward step bias is persistently negative, reward sum bias compounds to roughly -0.7 over the diagnostic horizon, dreamed episode return is near or below zero, and `continue_mean` is only about 0.65. Since imagined actor/critic updates optimize these dream targets, the auxiliary policy pressure is either weak or anti-correlated with true long-horizon HalfCheetah progress.

The dream critic's high explained variance is therefore not reassuring; it says the critic fits the model's returns, not that the model returns are useful.

Actionable experiment:
- v80/v81: separate free-run calibration from JEPA fitting. Add a small direct detached diagnostic head for reward/continue used only for an auxiliary calibration loss on free-run unrolls, while keeping outcome-token decoding as the primary JEPA geometry. Optimize 3-5 step free-run scalar reward sum and continuation Brier with low coefficients, then report whether reward sum bias and `imagination/model_episode_return_mean` improve.

### 3. Axial Predictor Moves Latents But Does Not Route Action Effects Into Reward

The axial predictor now produces large latent action sensitivity (`latent_action_sensitivity_std` around 3.0), but reward sensitivity remains tiny (`reward_action_sensitivity_std` around 0.011). This is the sharpest bottleneck: actions affect the predicted latent state, but the learned reward outcome geometry does not decode those changes into meaningful reward variation.

This suggests the action tokens are either affecting observation-token subspaces that are weakly coupled to the outcome codebooks, or the reward outcome token is too easy to satisfy from state/context priors and ignores action-conditioned residuals.

Actionable experiments:
- Add a reward-residual pathway conditioned on `(z_t, a_t, z_hat_{t+1})` as a calibrated auxiliary, not a replacement for outcome tokens. Penalize disagreement between residual reward and outcome-token reward during teacher forcing and free-run diagnostics.
- Add an action-contrastive JEPA term: for the same summary, compare predicted next summaries under real action vs shuffled/random action and require the reward outcome token distance to reflect observed reward deltas. This directly attacks reward action-insensitivity.

### 4. SIGReg / Current-Context Tradeoff Is Still Unresolved

v78's full-summary SIGReg produced healthy latent action sensitivity but poor returns. v79 skips current-slot SIGReg because current summaries carry neutral outcome placeholders, and that fixed much of the control collapse. However, v79 still has high obs prediction MSE and a plateaued JEPA loss. The model may now preserve outcome geometry while leaving current observation geometry under-regularized or mismatched against future target geometry.

The tradeoff is not simply "more SIGReg is better." Full-summary SIGReg can regularize fake neutral/current outcome structure; skipping it can leave the online current distribution weakly aligned with target future summaries.

Actionable experiment:
- Split SIGReg by token type and time role. Apply observation-token SIGReg on current and future summaries; apply outcome-token SIGReg only on real labeled future summaries. Log separate obs-current, obs-future, outcome-future SIGReg terms. This should preserve the v79 bootstrap fix while reducing online/target geometry mismatch.

### 5. Target Geometry Is Too Easy for Outcomes and Too Hard for State

Outcome MSE collapses to near zero while observation-token MSE stays around 1.1. This imbalance means the model may learn a low-dimensional, label-codebook-like outcome geometry without learning enough controllable state geometry to support extrapolation. The free-run model then emits plausible outcome tokens locally but drifts under self-generated obs tokens.

Actionable experiments:
- Normalize loss contributions by token group so obs-token prediction is not diluted by easy outcome tokens.
- Add horizon-split diagnostics for obs MSE, outcome MSE, reward bias, and continuation Brier at h=1..5. The current aggregate hides whether error is immediate, compounding, or boundary-specific.
- Add cosine/variance diagnostics for online current summaries vs future target summaries to detect geometry mismatch directly.

### 6. Diagnostics Need More Control-Causality Resolution

Current diagnostics are directionally useful, but they do not isolate where control information is lost. The key missing split is action effect on obs tokens versus reward outcome token versus decoded reward.

Add diagnostics:
- `dynamics/obs_action_sensitivity_std` and `dynamics/outcome_action_sensitivity_std` separately.
- `dynamics/reward_token_action_sensitivity_std` before codebook decoding.
- Horizon-indexed free-run reward MAE/bias and terminal Brier.
- Real-vs-dream return correlation for matched rollout starts.
- Counterfactual action rank: for K sampled actions at a real state, does predicted reward rank correlate with true next reward from short environment probes? Use sparingly as an evaluation script, not in training.

## Next Experiments

1. v80: matched previous-outcome context training.
   Train predictor starts from both neutral current summaries and previous-outcome current summaries. Success criterion: free-run reward sum bias magnitude below 0.3 and last-20 return above v79 without worse instability.

2. v81: split SIGReg by role.
   Current obs tokens get SIGReg; future obs+outcome targets get role-specific SIGReg; neutral current outcome tokens remain excluded. Success criterion: lower obs prediction MSE and no v78-style return collapse.

3. v82: action-contrastive outcome reward.
   Add shuffled/random action contrast on predicted reward outcome tokens. Success criterion: `dynamics/reward_action_sensitivity_std` rises materially, while reward free-run MAE does not increase.

4. v83: low-weight free-run calibration.
   Add 3-5 step scalar reward-sum and continuation calibration losses on detached free-run unrolls. Success criterion: dreamed model episode return becomes positive and real explained variance improves.

## Bottom Line

v79 fixed an important bootstrap/context bug and is materially better than v76-v78 locally, but the world model is still not supplying a reliable control objective. The failure is not teacher-forced outcome prediction; it is the conversion of action-conditioned latent movement into calibrated free-run reward/continue predictions. The next work should make the predictor train on the same previous-outcome context used by the agent, split SIGReg by token role, and add explicit action-contrast/free-run diagnostics so reward causality cannot hide behind low outcome-token MSE.
