# Plasticity measurements

The family question remains the opening of the original `FAMILY.md`: **each
perceptron decides, from its state on the current sample, how that sample should
move its incoming weights.** Better features, prediction heads, or scalar learning
rates do not establish that premise.

The joint-posterior line tests a broader mechanism: current-state Jacobians act
through learned cross-neuron covariance. Its gains must survive direction-erased
controls; they are not evidence for autonomous scalar neuron gates. Keep this
distinction explicit rather than rebranding every forecasting gain as plasticity.

The current covariance-free optimizer frontier is **matrix-direction geometry**,
not prediction transport. Transport's dense-stream gains failed matched PPO
transfer and the redesigned clipped-policy proxy. A Muon-inspired five-step
polar update, with independently tuned AdamW head/bias updates, reduces fresh
confirmation excess risk **16.8% online / 7.1% drifting reuse / 9.0% clipped bandit**
against tuned AdamW. RMS-only controls do not reproduce the cross-regime gain.
Training costs **2.3–2.7x AdamW**; the >=20% confirmation target remains unmet.
These are seed-1 proxy results, not a new PPO result or evidence for autonomous
state-dependent neuron gates. See `per_unit.md` for protocol, controls and limits.

The prediction audit separates better information from better inference.
Latest-frame categorical volatility forecasts improve substantially over the
old scalar baseline, but long memory adds only **0.47%** over the same latest
head. Direct raw-return gains are only **0.065%** on the rank400 confirmation
and **0.125%** on an additional cohort; auxiliary volatility teaching selects
weight zero. Hierarchical Gaussian return inference fails catastrophically on
that additional cohort despite numerical posterior agreement. Within-segment sparse
inference improves some streams, but fresh recurrence is **25.4% worse** than
v3 and fails promotion. These results do not establish optimal prediction or a
brain-like learning mechanism. See `sparse_signal.md` and `finance_panel.md`.

| Sheet | Recorded configurations and measurements |
|---|---|
| [measurement.md](measurement.md) | LR bracketing, task/init seeding, controls, Adam gate placement, batching, execution and verification. |
| [ppo_runs.md](ppo_runs.md) | HalfCheetah trajectories, LR controls, sphere sdplast, pcbatch and PPO rollout statistics. |
| [per_unit.md](per_unit.md) | Unit/connection gates, hidden-layer density, allocation, novelty, precision, pc, bins, covariance and meta-descent. |
| [per_sample.md](per_sample.md) | Heteroscedastic cells, predictability oracles/estimators, corrected sample streams and SPY. |
| [sparse_signal.md](sparse_signal.md) | Sparse-stream selectivity, clean recovery, noise absorption, fan-in and support changes. |
| [finance_panel.md](finance_panel.md) | Volatility versus actual returns: categorical losses, causal information timing, memory, shared representations, and hierarchical inference; raw-zero and signed-null controls. |
| [prospective_consolidation.md](prospective_consolidation.md) | Energy-based two-tier optimizer: innovation-whitened per-parameter consolidation gains over AdamW/polar fast tiers, with uniform-gain and scalar-gain controls; literature synthesis in `../reference/lit_*.md`. |
| [cross_neuron.md](cross_neuron.md) | Low-rank cross-neuron EKF: rank curve against the full-covariance reference on the dense stream, and on the vol panel. |

| Harness in `cleanrl/plasticity/` | Measurement |
|---|---|
| `sample_stream.py` | Validation-selected sustained, segment, endpoint and predict-before-update clean errors under heteroscedastic/heavy-tailed noise and switches. |
| `noisy_stream_diagnostic.py` | Sparse signal reconstruction, distractor leakage, signed cross term, clean risk and gate selectivity. |
| `ppo_signal_legibility.py` | PPO dL/dz state-conditioned SNR, ANOVA, cross-window correlations and tanh-slope variation. |
| `unit_bayes_stream_v1.py` | Neuron-block covariance versus Adam, shared, scalar and shuffled operators on online clean risk. |
| `network_bayes_stream_v2.py` | Full cross-neuron covariance versus block/Adam/direction-erased controls, memory and runtime. |
| `unit_metadescent_stream_v1.py` | Next-sample hypergradient-trained local Adam multipliers versus identity/bias/shared/shuffle clean risk. |
| `predictive_transport_stream_v1.py`, `predictive_transport_eval_v1.py` | Matrix-free prediction-change transport in Adam's first moment; six-arm, independently LR/beta-tuned complete dense streams. Ungated transport wins over Adam; current per-row stability gate is nearly inert. |
| `predictive_transport_stream_v2.py`, `predictive_transport_eval_v2.py` | All-FP32 transport with no previous Jacobian; tangent, implicit and robust ablations, matched 224-pair LR/beta refinement, and selected-only fullstream timing. |
| `predictive_transport_stream_v3.py`, `predictive_transport_eval_v3.py` | Fused FP32 Triton preserves v2 selected accuracy; initial 16-warp kernel lowers tangent single-candidate latency but regresses other transport arms. |
| `optimizer_proxy_eval_v8.py`, `optimizer_proxy_model_v8.py` | Matched LR/beta1/beta2/decay/head-rate searches across online regression, drifting batch reuse and signed clipped-policy optimization; polar directions versus transport and RMS controls. |
| `optimizer_proxy_eval_v9.py`, `optimizer_proxy_model_v9.py` | Per-family grids and auxiliary state; tier families (`tier_adamw`, `tier_polar`, `tier_vel_adamw`) against `look_adamw`/`scalar_adamw` controls and the v8 AdamW/polar grids. |
| `optimizer_proxy_eval_v10.py`, `optimizer_proxy_model_v10.py` | v9 protocol plus `look_polar` (uniform two-tier reset on the polar fast tier, k0 down to .1); lean per-family plan `optimizer_proxy_v10_plan.json`. |
| `function_space_consolidation_diag_v1.py`, `_v2.py` | Cheap falsification of function-space (Jacobian row-space) consolidation on plain fast-tier trajectories: one-step counterfactual and closed loop. Dead in closed loop. |
| `structured_generalization_diag_v1.py`, `rethink/RethinkD.md` | Reapproach from the original goal: proxy excess risk is bias/tracking, not variance (derived from job 5895 curves); premise test with a structured teacher and off-marginal held-out sets fired: same in-distribution risk, 34% lower distractor risk with an energy prior; polar beats AdamW off-marginal by 35%. |
| `optimizer_proxy_eval_v11.py`, `optimizer_proxy_model_v11.py`, `structured_generalization_diag_v2.py` | structured_ood regime (sparse-input teacher, off-marginal held-out scoring) and SNR-shrunk matrix families. Shrinkage null as pre-registered (gamma uniform); AdamW 47% worse than polar off-marginal under identical in-distribution selection; look_polar (slow deployed tier, k0 .05, hot fast tier) -25% / -30% vs polar in-dist / off-marginal, front-loaded, one regime, not promotable while the v10 bandit regression stands. See `rethink/RethinkD.md`. |
| `optimizer_proxy_confirm_v1.py` | Prelocked configurations on fresh seed-1 RNG namespaces; full horizons, no retuning, paired held-out risk and single-candidate training cost. |
| `predictive_segment_eval_v5.py`, `predictive_hazard_eval_v6.py`, `predictive_structure_eval_v7.py` | Full sparse stationary/change/null/recurrent/two-support streams; segment and hazard uncertainty, singleton structural inference, frozen noisy-prefix AdamW selection and fresh namespace confirmation. |
| `panel_predictive_memory_eval_v4.py` | Old/latest/memory information ablations for volatility, scalar/categorical/QLIKE heads, authenticated forward interval and randomized targets. |
| `panel_return_representation_eval_v5.py`, `panel_return_refinement_eval_v6.py` | Raw-zero signed-return evaluation; matched LR/beta2 searches and shared volatility representation versus permuted and zero-weight controls. |
| `panel_return_hierarchy_eval_v7.py` | Pooled, independent and hierarchical Gaussian return posteriors; online positive Schur updates, independent batch audits and prefix-locked cohort/null evaluation. |
| `ppo_continuous_action_sdplast_relsnr_v4.py` | Relative-SNR geometric-mean-one level invariance; no PPO benchmark recorded. |
| `ppo_continuous_action_sphere_sdplast_v5.py` | Sphere PPO return and separate pre-Adam data/post-Adam level effects. |
| `ppo_continuous_action_sphere_sdplast_v7.py` | Sphere PPO with exponent-one SNR and hidden gate/up coverage. |
| `ppo_continuous_action_sphere_sdplast_v8.py` | Batch and per-unit residual-energy gating against an EMA reference. |
| `ppo_continuous_action_sphere_sdplast_v9.py` | Asymmetric suppression/inflation gate envelope. |
| `stock_stream.py` | SPY prequential real/permuted error and noise absorption across LRs. |
| `predictive_conditional_stock_eval_v4.py` | Causal calibration/stacking/context controls versus frozen static forecasts and matched-context Adam; no substantial context gain. |
| `predictive_available_stock_eval_v5.py` | Same targets with original/latest available input windows, four prefix-locked Adam banks, and the causal raw-zero normalization control. See the SPY audit in `per_sample.md`. |
| `hidden_stream.py` | Hidden-layer sparse-input recovery, density, LR, batch and allocation controls. |
| `novelty_stream.py` | Shared/per-region teacher acquisition/retention and novelty/precision gates. |
| `precision_audit.py` | Task-resampled precision comparisons, LR truncation, null calibration and unit-oracle headroom. |
| `dense_boundary.py` | Density-dependent allocation error and dense teacher-switch recovery. |
| `pc_stream.py` | Field-conditional gates across batch sizes with independently selected LRs and shuffle controls. |
| `ppo_continuous_action_pcbatch_v1.py` | HalfCheetah per-(unit,sample) minibatch gating versus off/scalar/shuffle. |
| `hetvar_stream.py` | Heteroscedastic weighting, predictability oracles, residual estimators and parameter-input bins. |
| `panel_stream.py`, `panel_hd.py`, `panel_hd_mlp.py` | Cross-sectional vol panel ceilings (ridge, offline MLP) and streaming Adam across LR/width. |
| `panel_hd_gate.py` | Per-sample RLS precision weighting on the panel versus its within-bar shuffle. |
| `panel_hd_pp.py` | Per-parameter James-Stein evidence gate on Adam steps versus none/shuffle, blocked test scores, permuted stream. |
| `lowrank_bayes_stream_v1.py` | Rank-r precision-form EKF versus Adam / full covariance / diagonal on the v2 dense stream. |
| `covariance_sketch_stream_v3.py`, `covariance_sketch_eval_v3.py` | Prior-whitened covariance downdate sketch versus full, block, diagonal and direction-erased controls; faster but less accurate than full covariance on both complete dense streams. |
| `iterated_bayes_stream_v4.py`, `iterated_bayes_eval_v4.py` | Fixed-prior nonlinear observation refinement with one posterior conditioning; two/four refinements lose to full EKF on the complete heteroscedastic-switch stream. |
| `panel_hd_ekf.py`, `panel_hd_floor.py` | Low-rank EKF on the vol panel (batched per-bar update); in-sample whole-period fit. |
| `panel_hd_coh.py`, `panel_hd_evt.py` | One-matmul per-(sample,parameter) coherence/mass/clipping arms; event-driven sequential-test descent. |
| `optimizer_proxy_eval_v12.py`, `optimizer_proxy_model_v12.py`, `structured_generalization_diag_v3.py` | Oracle ceiling for decay on the irrelevant columns (-26% polar / -36% AdamW in-dist, -52% / -79% off-marginal) and the readable tag: not the gradient's own direction (agreement 0.50 on distractors) but resistance to a pull. Resistance-gated decay (gate = Phi of the slow gradient mean's z-score against the weight's sign): no regression in the three dense regimes; structured_ood gate_polar -7.9% / -18.8% vs polar and gate_adamw -12% / -53% vs AdamW at pole .9999, monotone in the pole, below the 20% bar. See `rethink/RethinkD.md` sections 13-14. |
| `cleanrl/shared/gate_polar.py`, `panel_hd_mlp_gate_v1.py`, `cleanrl/ppo_continuous_action_gatepolar_v1.py` | Torch `GatePolar` (polar trunks, Adam heads, resistance-gated decoupled decay) reproducing proxy v12 in float64; transfer tests on the vol-panel MLP stream and base Beta PPO, rules pre-registered in `rethink/RethinkD.md` section 15. |
