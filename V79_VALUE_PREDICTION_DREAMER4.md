# Value Prediction for LEJEPA, Informed by Dreamer4

Scope:
- Local LEJEPA implementation: `cleanrl/hlgauss/transformer/ppo_continuous_action_hlgauss_relusq_dlogstd_lewm_obsrecurrent_beta_asymspo_actionsa_axialrope_perceiver4readout_obsid_streamdream_v79.py`
- Dreamer4 reference: `../dreamer4/dreamer4/dreamer4.py`, `../dreamer4/dreamer4/trainers.py`, `../dreamer4/train_cartpole_with_dynamics_rl.py`

## Executive Recommendation

Do not add a second, incompatible "world-model value head" to LEJEPA as the default. Dreamer4 does not train value as part of the world-model representation loss. It attaches a value head to the world model's agent-token representation, then trains that head with actor-critic targets from real or dreamed experience. In v79, the analogous mechanism already exists: the HL-Gauss critic reads LEJEPA summary tokens through the agent readout and is trained on real GAE returns plus dreamed lambda returns.

The idiomatic implementation direction is therefore:

1. Keep reward and continuation as outcome-token world-model targets.
2. Keep value as an actor-critic prediction over summary tokens, trained by HL-Gauss return targets.
3. Let dreams use predicted reward/continuation outcome tokens to build lambda returns, and use the critic only for per-state values and bootstrap values.
4. Only test "value as an outcome token" as a controlled ablation, excluded from actor/critic readout at first, because it is nonstationary, policy-dependent, and can inject critic errors into the JEPA target geometry.

## How Dreamer4 Predicts and Uses Value

Dreamer4's `DynamicsWorldModel` owns the dynamics backbone, reward prediction, policy head, and value head, but it separates their roles.

### Value Head Attachment

Dreamer4 builds one main token stream:

`[flow token, latent space tokens, proprio token, state pred token, registers, action token, reward token, agent token]`

The main transformer produces updated tokens. Optionally, separate actor and critic transformer branches further process the same token sequence. The policy head reads `actor_agent_tokens`; the value head reads `critic_agent_tokens`. Without those optional branches, both fall back to the main `agent_tokens`.

The value head is an MLP from an agent-token embedding to the same symlog/two-hot bin space used by Dreamer4's reward encoder. Scalar values are decoded by `reward_encoder.bins_to_scalar_value(value_bins)`.

Key point: value attaches to the agent representation, not to the latent flow prediction head and not to reward MTP heads.

### World-Model Training Does Not Train Value

Dreamer4's world-model `forward()` loss includes latent flow/shortcut losses, reward prediction, terminal prediction, optional action MTP behavior cloning, optional state prediction, and optional SSL losses such as latent AR/SIGReg. It does not include value loss.

Reward prediction is multi-token prediction: each shifted agent token predicts several future reward labels via an ensemble of reward heads. Action prediction is also MTP when action loss is enabled. However, generation consumes the one-step head autoregressively: generated reward uses reward head index 0, and generated policy actions use action head index 0. MTP is a training regularizer for richer local predictive structure; dream rollouts still advance one step at a time.

### Dream Generation Stores Values for Actor-Critic Learning

`generate(..., return_for_policy_optimization=True)` requests generated actions, rewards, logprobs, values, and terminals. During generation:

- the model is put in eval mode and restored afterward;
- reward is decoded from the current clean agent embedding;
- terminal probability is decoded from the generated latent state;
- action is sampled from the policy head;
- value is decoded from the value head on the same clean agent embedding;
- the generated `Experience` carries rewards, actions, old logprobs, old values, masks/lens, and optional stored agent embeddings.

The trainer often calls `generate(horizon + 1)` so the final generated state can serve as a bootstrap node for value targets.

### Actor-Critic Learning Builds Return Targets

`learn_from_experience()` computes GAE/lambda returns from rewards, old values, masks, and terminal/lens information. Then:

- policy loss is PPO/SPO/PMPO over action logprob ratios and advantages;
- value loss is cross-entropy between value logits and two-hot encoded lambda returns;
- by default `only_learn_policy_value_heads=True`, so stored or replayed agent embeddings are detached and the dynamics backbone is not updated by policy/value losses;
- trainers use separate optimizers for `policy_head_parameters()` and `value_head_parameters()`.

Dreamer4 therefore treats value as an actor-critic learner over world-model states, not as a world-model target. This is the important design lesson for LEJEPA.

## Current LEJEPA v79 Value and Outcome Structure

v79 summaries have:

- `NUM_OBS_TOKENS = 8`
- `NUM_OUTCOME_TOKENS = 2`
- total summary tokens: 8 observation tokens plus reward and continuation outcome tokens

The outcome tokens are learned target embeddings:

- reward labels are projected to a reward distribution over `reward_num_bins`, then through `reward_outcome_proj`;
- continuation labels are projected through `continuation_outcome_proj`;
- decoding is distance-to-codebook over learned outcome token codebooks, not a separate trained CE/BCE dynamics head.

The predictor maps summary/action history to the next full summary. The LEJEPA objective is MSE from predicted next summary to target future summary, plus SIGReg. Reward and termination CE/BCE are detached probes over predicted outcome tokens, not optimized heads.

v79 also fixed an important value alignment issue: rollout values and real GAE bootstraps now use summaries with matching previous/arrival outcome context instead of bootstrapping from neutral-outcome next observations. Acting uses `encode_summary_with_optional_outcome()` with the previous reward/continuation when available. World-model training still starts from neutral current summaries and predicts labeled future summaries.

The existing critic is already the LEJEPA equivalent of Dreamer4's value head:

- actor and critic read summary tokens through the shared Perceiver readout;
- the readout is split into actor and critic tokens;
- `critic` outputs HL-Gauss logits;
- real PPO trains it on real GAE returns;
- dreamed PPO trains it on lambda returns constructed from predicted dream rewards/continuations plus a critic bootstrap.

## Should Value Be an Outcome Token?

Default answer: no.

Reward and continuation are environment transition outcomes. They are part of the world dynamics target: after taking action `a_t` from state `s_t`, the model should predict the arrival summary for `s_{t+1}` and the transition outcome `(r_t, c_t)`.

Value is different:

- it is policy-dependent;
- it changes as the actor and critic improve;
- it is bootstrapped from the same critic being trained;
- it is not an environment observable;
- it is a long-horizon estimate, not an immediate transition label.

Putting value into the same outcome-token target geometry makes the world-model target nonstationary and circular. It risks teaching the predictor to preserve critic artifacts instead of environment structure. It also creates leakage risk if actor/critic read a target-encoded value token during real training.

A value token is only defensible as an ablation or auxiliary diagnostic. If added, it should initially be excluded from actor and critic readout, excluded or separately handled in SIGReg, and trained from detached return targets.

## Idiomatic LEJEPA Implementation Plan

### Preferred Path: No New Value Outcome Token

Keep the current conceptual split and make it explicit:

- `encode_target_summary()` contains observation, reward, and continuation targets.
- `decode_outcomes()` decodes only reward and termination/continuation.
- `critic` remains the single value prediction mechanism.
- real and imagined value losses continue to use HL-Gauss targets.
- actor/critic gradients do not update the predictor/encoder unless explicitly ablated.

Target construction:

- Real value target: current v79 GAE is the right shape. Values at rollout time come from outcome-aware current summaries. Bootstrap values come from target next summaries that include the just-observed transition reward and continuation. Return targets are projected by `HLGaussSupport.project()`.
- Dream value target: generate predicted summaries autoregressively; decode reward and continuation from predicted outcome tokens; compute lambda returns with critic values plus a final bootstrap; project returns to HL-Gauss; train critic logits on those targets.
- No world-model value target is needed.

Gradient flow:

- LEJEPA loss updates encoder, outcome projections, predictor, and SIGReg-relevant parameters.
- Real/dream actor-critic losses update actor/critic/readout parameters through detached summary tokens when `detach_world_model_from_agent=True`.
- Dream construction runs under `torch.no_grad()` and eval mode, as v79 already does.
- Return targets are detached before value CE, as v79 already does with `return_probs.detach()` and `dream_return_probs.detach()`.

This matches Dreamer4's separation while staying native to the current v79 file.

### Optional Ablation: Value as a Third Outcome Token

If we want to test whether a value target helps dream calibration, add it as a versioned ablation, not as the default architecture.

Possible construction:

- Add `NUM_VALUE_OUTCOME_TOKENS = 1`, making summaries `[obs tokens, reward token, continuation token, value token]`.
- Encode the value target with an HL-Gauss or symlog two-hot projection and a learned `value_outcome_proj`, analogous to reward.
- For transition target `t`, attach value for the arrival state `s_{t+1}`, not the departure state. That means the predicted next summary contains `(obs_{t+1}, r_t, c_t, V_target(s_{t+1}))`.
- Build real value-token targets from detached lambda returns shifted to the arrival state where valid. For the final bootstrap state, use detached critic bootstrap if needed.
- During dreams, compute value-token targets only after the full dreamed rollout is available, using detached dreamed lambda returns.

Critical restrictions for the first ablation:

- Do not feed the value token into actor or critic readout. Let policy/value heads read only obs plus reward/continuation tokens, or mask the value token out in `_agent_readout_features_from_latents()`.
- Do not use the value token for the critic bootstrap. Use the critic head, as Dreamer4 does.
- Do not let critic loss backprop into value-token target construction.
- Keep value-token loss coefficient small and separately logged.
- Prefer excluding value-token dimensions from SIGReg initially, or apply token-specific SIGReg with a separate statistic.

If performance improves only when the critic reads the value token, that is likely target leakage or self-confirming bootstrap, not a reliable world model.

## SIGReg and Collapse Concerns

Reward and continuation outcome tokens already have low-dimensional label structure. Value is even more compressible and nonstationary. Adding a value token to full-summary SIGReg can distort the representation in two ways:

- The value token may collapse to a smooth scalar manifold that satisfies return targets but does not preserve controllable state information.
- SIGReg may fight the natural low-dimensional geometry of reward/continue/value labels, causing the predictor to spend capacity isotropizing labels rather than modeling dynamics.

Recommended SIGReg policy:

- Keep current neutral current-slot exclusion.
- Keep observation-token SIGReg separate from outcome-token SIGReg.
- If a value token is ablated, log and regularize it separately; do not silently include it in the same flattened full-summary SIGReg.
- Track token variance, norm, pairwise cosine, and codebook entropy for reward, continuation, and optional value tokens.

The collapse signal to watch is not just low prediction MSE. v79 already shows outcome-token MSE can be excellent while free-run rewards remain poorly calibrated. Any value-token ablation must be judged by free-run calibration and real returns, not teacher-forced value-token MSE.

## Dream Use

For the default design, dreams should use value exactly as Dreamer4 does:

1. Start from same-episode prompt summaries and action features.
2. Roll the predictor one step at a time.
3. Decode reward and continuation from predicted outcome tokens.
4. Query the critic on each dreamed summary for values.
5. Query the critic on the final dreamed summary for bootstrap.
6. Compute lambda returns using predicted rewards and soft continuations.
7. Train imagined actor and critic on the fixed dreamed batch.

Do not train the predictor to output a value that replaces the critic bootstrap. That would make the model both the reward source and the terminal value source, reducing the independent correction provided by the actor-critic learner.

For a value-token ablation, use the value token only for diagnostics first:

- compare decoded value-token scalar to critic value;
- compare decoded value-token scalar to realized real/dream lambda targets;
- measure whether value-token error correlates with dream reward bias;
- do not use it in the actor or critic objective until it is demonstrably calibrated.

## Logging Plan

Keep current logs and add splits that answer whether value prediction is helping control rather than just fitting targets.

Default critic logs:

- `value/real_loss_ce`
- `value/real_target_mean`
- `value/real_pred_mean`
- `value/real_target_std`
- `value/real_pred_std`
- `value/real_mse`
- `value/real_explained_variance`
- `value/dream_loss_ce`
- `value/dream_target_mean`
- `value/dream_pred_mean`
- `value/dream_mse`
- `value/dream_explained_variance`
- `value/bootstrap_mean`
- `value/bootstrap_abs_mean`

Dream calibration logs:

- `imagination/reward_sum_bias_by_horizon`
- `imagination/return_bias_by_horizon`
- `imagination/critic_vs_model_return_corr`
- `imagination/value_delta_mean = V(s_t) - lambda_return_t`
- `imagination/value_delta_abs_mean`

If value-token ablation is implemented:

- `lejepa/value_outcome_prediction_mse`
- `value_token/scalar_mae`
- `value_token/scalar_bias`
- `value_token/codebook_entropy`
- `value_token/token_norm`
- `value_token/token_variance`
- `value_token/cosine_to_reward_token`
- `value_token/cosine_to_continuation_token`
- `sigreg/value_token_loss`
- `grad/value_token_to_encoder_norm`
- `grad/critic_to_encoder_norm` should remain zero in the default detached setting.

## Ablation Plan

Run ablations in versioned files so the trail remains clear.

1. `v80_value_logging`
   - No architecture change.
   - Add the logging above and verify whether the existing critic is underfitting real returns, dream returns, or both.
   - Success: clearer diagnosis; no score regression expected.

2. `v81_value_readout_split`
   - Keep no value token.
   - Ensure actor/critic readout can be logged with obs-only versus obs+outcome variants.
   - Test whether reward/continue outcome tokens help or harm critic calibration.

3. `v82_value_outcome_diagnostic`
   - Add a value outcome token but exclude it from actor/critic readout and dream bootstraps.
   - Train it from detached arrival-state lambda targets with low coefficient.
   - Success: decoded value token predicts returns better than chance without worsening reward/continue calibration or SIGReg statistics.

4. `v83_value_outcome_control`
   - Only if v82 is calibrated.
   - Test using the value token as an auxiliary feature for the critic, not as a replacement for critic output.
   - Guardrail: if real explained variance improves while dream reward calibration worsens, reject it.

5. `v84_separate_dreamer_value_branch`
   - Only if the shared readout is clearly insufficient.
   - Add a Dreamer4-style critic branch/readout over summary tokens, not a dynamics value head.
   - This is a separate mechanism, but still compatible because it attaches to the same summary representation and trains only from actor-critic targets.

## Bottom Line

Dreamer4's value prediction is actor-critic value learning over world-model states, not an additional world-model prediction target. The v79 LEJEPA architecture already has the right analogue: an HL-Gauss critic over summary tokens used for real and dreamed lambda returns.

The next idiomatic step is to improve value logging and calibration around the existing critic. A value outcome token is possible, but it should be treated as a risky auxiliary ablation with strict gradient, readout, SIGReg, and dream-use constraints. It should not replace the critic and should not be allowed to create a self-confirming value shortcut through the world-model target geometry.
