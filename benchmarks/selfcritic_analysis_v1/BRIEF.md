# Brief: deep analysis and prescriptions for the TD7 latent-Q line (HalfCheetah-v4)

You lead an analysis team. Spawn a bunch of subagents (parallel, each with a distinct angle) and synthesize their findings. This is ANALYSIS AND PRESCRIPTION ONLY: do not submit training runs, do not run GPU work, do not edit existing code. Read code, read TensorBoard event files (CPU), do math, write the report. Repo: /home/marvin/Documents/repositories/cleanrl (read CLAUDE.md and cleanrl/td7/FAMILY.md first). Use `.venv/bin/python`. TensorBoard runs live in `runs/HalfCheetah-v4__<exp_name>__1__<timestamp>/`; helpers in `scripts/_runs.py`, `scripts/score_runs.py <substr...> --at 100k,300k,1M --metrics <tags>`. Eval tag: `eval/episodic_return` (TD7 checkpoint policy, deterministic); `charts/episodic_return` is the stochastic training return.

## Goal
Maximize HalfCheetah-v4 return (seed 1, ~2M steps) with a genuinely novel TD7-based agent. The user's vision: an ambitious "conscious", introspective, self-critical system whose goal is highest return; the actor should surprise the critic, the critic should make strong accurate extrapolations; favours "TD in belief coordinates" and analytical reasoning over reflexive standard tricks; skeptical of multi-pass hacks. Recent ideas are "not impressive"; we need a diagnosis of WHY and prescriptions that could clearly beat the reference.

## Reference (best)
`cleanrl/td7/latentq/td7_latentq_preaction_fresh_optimism_v27.py` (= latentq_preaction_v5 + a fresh-data optimism probe). Critic Q~(s, mu, sigma, u, zsa, zs) over the actor's pre-action belief N(mu, sigma^2) with Hermite/Stein tangent labels from realized noise, plus behavior and point queries; target (next_mu, 0.2, next_mu) twin-min; actor ascends Q~(s, mu, sigma, mu) with zsa(mu). SALE, LAP, checkpointing, CUDA-graph steps. Eval: 12.0k@100k, 15.8k@300k, 17.2k@400k, 18.5k@800k, 18.6k@1.2M, ~19.2k@1.7M. Fresh optimism (critic minus realized discounted return on unseen episodes) is consistently -50..-75.

## What failed (files in cleanrl/td7/...; see FAMILY.md for more history)
- `selfcritic/td7_selfmodel_v1.py`: b(s,mu,sigma,u,age) regresses the critic's own error vs realized returns on the last 20k transitions; actor ascends Q~ - relu(b). Eval 11.6k@300k, 15.0k@1.2M; live training policy collapsed (training return 1.1k@1.3M). fresh/sm_corr ~0 (b never localized error); window mean error swung -300..+350; q_values fell 880->286. Our hypothesis: ~20 effective samples (episode-shared return tails), so b only learns a global offset, and its action gradient is uninformative/exploitable noise.
- `selfcritic/td7_selfcritic_deliberate_v1.py`: self-model + deliberation (planner terminal = Q~ mean - relu(b)). Eval 12.2k@300k; b swings dominate plan scores (plan_gain spikes to 93).
- `selfcritic/td7_deliberate_v1.py` (job 14745, may still be running): learned one-step model plans 3 steps over 64 belief draws, best becomes belief mean, executed = pick + sigma*eps. Eval 15.3k@300k (v27 15.8k), 7x slower (105 SPS); keeps the mean <1% of the time; plan_gain ~4-7 vs Q~1000; fresh optimism turned positive (+40..+226).
- `heatbelief/` (exact Gaussian-bump belief critic, dictionary atoms): lowest TD loss, worst return. `consequenceq/` (model-grounded Q = r + gamma V(s')): ~4x more accurate fresh values, slower improvement; belief_residual_v4 optimistic, collapsed tail.
- Cross-family lessons we believe (verify or refute!): critic accuracy does not predict return; error SIGN matters (optimistic critics lose, v27 is pessimistic); every variant lacking zsa(zs, a) in the critic underperformed; parameter-conditioned critics are unidentifiable; extra hidden belief dims get zero true gradient.

## Questions for the team (assign angles to subagents)
1. Root-cause each failure from the actual metrics (tfevents), not from our story. Where is our story wrong or incomplete?
2. What actually makes v27 work, and what limits it at ~19k? (gradient flow through zsa(mu), the pessimism mechanism, sigma dynamics, checkpointing effects, where its learning plateaus). Compare against TD7 base `cleanrl/td7_continuous_action.py` / other td7 runs present in runs/.
3. Is "self-critical introspection" salvageable in a form that is identifiable and not exploitable? (e.g. per-state error with proper effective sample size, error models on TD residual statistics instead of MC returns, using introspection for data collection/exploration rather than the actor gradient, conservative bounds, ensembles/epistemic signals that the actor cannot game.) Do the math on identifiability and on what gradient the actor actually receives.
4. Is deliberation salvageable cheaply (it must not cost 7x SPS) — or is it winner's-curse selection on noise? Quantify.
5. Survey the most relevant recent literature/ideas (2023-2026) for off-policy continuous control beyond TD7 (e.g. BRO, SimBa, CrossQ, TD-MPC2, BBF-style scaling, distributional/HL-Gauss critics, MR.Q) and judge which mechanisms would compose with the latent-belief critic. Be critical: we want novel, not a port.

## Deliverable
Write `benchmarks/selfcritic_analysis_v1/REPORT.md` (concise, dense, no LaTeX; say what was checked vs inferred):
- Diagnosis per failed variant with evidence (metric, step, value).
- What limits v27.
- 3-5 ranked prescriptions, each: mechanism, why it should beat v27 (expected effect and the gradient/covariance reasoning), the cheapest decisive experiment, failure signature to watch, and implementation sketch against v27's code (which functions change).
- Things to stop doing.
Keep the final reply to a short summary pointing at the report.
