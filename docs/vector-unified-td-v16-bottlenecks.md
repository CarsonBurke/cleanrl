# V16 bottleneck audit

Read-only audit of v16 source and its existing completed training record. No new training, checkpoint rollout, held-out gate, or algorithm change. Independent review checked the algebra and interpretation.

## 1. Measured critic conditioning failure

Each logged critic gradient norm is the mean of the 18 pre-clipping norms in that rollout. The clip threshold is 0.5.

| Transitions | Median logged preclip norm | Mean vector TD residual RMS | Rollouts with minimum kernel std at floor |
|---|---:|---:|---:|
| 0–2M | 103.3 | 4.993 | 0/60 |
| 2–4M | 569.0 | 4.213 | 40/61 |
| 4–6M | 1,054.8 | 3.830 | 36/61 |
| 6–8M | 3,370.3 | 4.151 | 24/63 |

The final 20 rollouts averaged a preclip norm of **3,422.5**, TD residual RMS **4.091**, and minimum kernel reference standard deviation **1.028e-6**. Kernel variance is floored at 1e-12, giving a standard-deviation floor of 1e-6. Standardization divides each centered kernel by this scale, and gradients flow through the learned powers and their normalization.

This is direct evidence of stiff fitting and at least some almost-degenerate or tiny-scale kernel/state pairs. It does not identify the responsible layer, prove that all kernels collapsed, or measure their Gram spectrum. The standard deviation is under fixed Beta(2,2), so its small value cannot itself be blamed on the collection policy's concentration. Global clipping followed by Adam does not imply that effective parameter updates shrink by the simple preclip/clip ratio. Raising the clip threshold would not resolve the diagnosis.

Relevant source: `kernel_statistics` and `UnifiedCritic.predict`, plus the full-batch clipping path.

## 2. The predictive supervision is much narrower than the tensor suggests

The model emits 7 × 48 coefficients, but each sampled state/action supervises **seven accumulated reward-component means**. Those coefficients are coordinates of an action-value function, not 336 independently grounded predictions about the future. No latent transition or richer future consequence distribution is supervised.

The actor's scalar return function has exactly the form:

`sum_c Q_c(s,a) = sum_f [sum_c C_cf(s)] * feature_f(s,a)`

Because the components share the same trunk and action basis, summing the rows gives a scalar coefficient head of the same function class. Vector targets can still improve representation learning, but the actor has not gained a richer return-function class simply because the coefficient tensor is large. Summing before expectation is algebraically identical to summing afterward and is **not a bug**. The structural limitation is what gets supervised, not where the final utility sum occurs.

At a single state/action, seven values underdetermine the coefficients; this is not a proof of population non-identifiability. Shared neural structure and adequate action coverage can supply additional constraints. Their adequacy was not measured here.

## 3. Analytic integration has constrained the critic

Action dependence passes through 12 log-score coordinates and 35 prescribed-support power-product kernels. Their shapes depend on state, but their supports and positive exponent range remain constrained. There is no unrestricted joint state-action trunk. Exact Beta integration is exact for this restricted fitted family, not for the environmental action-value function.

This is a verified architectural restriction and a plausible approximation bottleneck, not an isolated explanation of the low score. In particular, the log-score span plus the intercept can recover the correct local population gradient at an unrestricted conditional least-squares optimum. Poor finite-data fitting, shared-network approximation, and inaccurate finite-update curvature remain the relevant concerns; merely enlarging the basis is not a proved fix.

## 4. Measured actor conditioning failure

CG relative residual exceeded 0.1 on **58/245** updates. The phase breakdown was 0/60, 0/61, 17/61, and **41/63** for the same successive 2M windows. In the final window its mean was **0.1744**. This worsened around the late plateau, but does not establish causality.

The solver uses 50 unpreconditioned iterations on a damped actor Fisher. Its logged residual is the recursively maintained CG residual, not an independently recomputed final linear-system residual. Candidate KL checks keep measured updates feasible, but do not make inaccurate directions optimal. The nonlinear ray search is a local one-dimensional procedure, not a global constrained optimizer.

This cannot explain the entire deficit: v16 was already weak in the first 2M, when mean solver residual was approximately 0.00205 and none exceeded 0.1.

## 5. Model error passes directly into the actor

The actor uses only the fitted critic's expected improvement. Observed rewards affect critic fitting, but no sampled reward/TD-residual correction enters the actor estimator after that fit. Every proposal was accepted; positive modeled gain therefore provides no independent check that environmental credit is correct. This is standard approximate actor-critic exposure to critic error, not a proof that a residual correction would solve it.

A small relative TD error against large bootstrapped targets does not establish accurate action-value predictions or policy gradients. Neither the local derivative tests nor exact feature integration establish that the fitted continuation is correct. The record lacks per-parameter-group gradient norms, feature Gram spectra, component-wise TD statistics, and score-projected residuals on the training batch, so the contribution of each source cannot presently be quantified.

## Priority

The strongest measured problem is numerical/statistical conditioning of the learned feature parameterization. The strongest structural problem is that the purported rich future latent remains a restricted, seven-component action-value regression. Actor conditioning is an additional measured limitation, especially late in training. Unifying the model removed an artificial architectural distinction, but did not address these deeper issues. None of these findings requires reinstating separate short/long or outcome/effect models, minibatching, replay, or repeated target sweeps.

Evidence: `cleanrl/ppo_continuous_action_vector_unified_td_v16.py` and `runs/HalfCheetah-v4__vector_unified_td_v16_8M__1__1789585442529891760/result.json`.
