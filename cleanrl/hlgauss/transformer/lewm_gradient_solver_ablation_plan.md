# LEWM Gradient Solver Ablation Plan

Goal: make the LEWM world model useful as a differentiable action optimizer for MuJoCo PPO without hiding performance or stability effects behind bundled changes.

## Baseline

- `v135`: one candidate action sequence per rollout state.
- Horizon default `H=1`.
- Inner solver optimizes raw bounded actions with Adam, then clamps to the action box.
- Actor receives an MSE amortization target from the first optimized action.
- PPO/SPO and PrefPoE remain the real-environment anchor.

## Ablation Ladder

1. `v136`: tanh-bounded inner-solver variables only.
   - Keep `H=1`, `S=1`, solver steps, learning rate, objective, and losses unchanged.
   - Replace raw action + clamp with unconstrained `u` and `action = action_mid + action_scale * tanh(u)`.
   - Hypothesis: smoother action gradients near bounds and fewer Adam/clamp artifacts.
   - Cost impact: essentially unchanged.

2. `v137`: multi-sample action sequences.
   - Add `inner_solver_samples`, default small (`4` or `8`).
   - Keep `H=1`.
   - Initialize sample 0 from policy mean and remaining samples from policy/noisy starts.
   - Select best final-cost sample per state for amortization.
   - Hypothesis: avoids local single-start failures without increasing model rollout length.
   - Cost impact: roughly linear in `S`; use small defaults first.

3. `v139`: short multi-step planning horizon.
   - Increase horizon to `H=3` with same `S` as prior best.
   - Optimize full sequences, still amortize only first action.
   - Hypothesis: captures contact/setup effects in Hopper and Walker that one-step value gradients miss.
   - Cost impact: roughly linear in `H`, with higher autograd memory from unrolled dynamics.

4. `officialsolver_v138`: official-aligned LeWM gradient-solver control.
   - After WM warmup, run the gradient solver during real rollout and execute the first optimized action.
   - Train the actor to amortize logged planner behavior actions.
   - Disable actor PPO/SPO and PrefPoE policy-gradient terms once planner behavior starts.
   - Keep critic/value and WM training on real transitions.
   - Hypothesis: tests whether the learned WM gradient solver can control MuJoCo directly before adding entropy, priors, residuals, or SAC-like machinery.
   - Cost impact: high; solver runs on every real env step after warmup.
   - Result: collapsed on HalfCheetah seed 1. Last-10 return was about `-739`, while `v136` and `v137` finished around `1048` and `983`.
   - Diagnosis: this was not actually comparable to the official LeWM reference. It used a learned online return objective, `-sum gamma^k r_hat - gamma^H V_hat`, with `H=1`, `S=4`, and 8 Adam steps. The reference LeWM gradient solver minimizes final goal-embedding distance with `H=5`, action blocks, warm starts, `S=100`, and 30 AdamW steps.
   - Failure signature: planner execution fraction reached `1.0` after warmup, PPO/SPO policy-gradient terms were disabled for planner-collected samples, and the actor became a behavior clone of the planner. Returns flattened near zero through about 512k steps, then collapsed around 640k.
   - Planner cost did not show useful improvement: rollout cost deltas were tiny, roughly `-0.0005` to `-0.0068`, and predicted planner cost rose from about `-1.26` at 131k to `+0.49` at 655k and `+2.16` by 983k. The planner was not finding a strong high-return action; it was selecting the least-bad action under a weak one-step objective.
   - The collapse coincided with behavior-policy mismatch and poor value fit: `real_rollout/logratio_abs_mean` jumped to about `2.07` at 655k, action threshold fraction to about `0.50`, and real-rollout explained variance stayed around `0.3-0.45` instead of `~0.97` for v136/v137.
   - Root cause: replacing PPO action selection with a myopic, online return-gradient planner removed the policy-learning anchor before the WM/critic provided a reliable control objective. This is different from demonstrating that the official LeWM gradient solver is weak.

5. `officialsolver_warmstart_bestpolicy_v139`: implementation-corrected best-policy MPC.
   - Keep the unknown-goal objective: optimize predicted real return plus terminal value, not goal-embedding distance.
   - Fix rollout/update init mismatch: rollout planner now honors `inner_solver_init` instead of always initializing from a sampled fused-policy action.
   - Disable planner-behavior actor amortization by default. In v138, after planner takeover, PPO/SPO was zeroed and the actor mostly mean-cloned stale planner actions; this matched the delayed KL/ratio explosion.
   - Add shifted-plan warm starts and raise default solver strength to `H=3`, `S=8`, 16 Adam steps. This is still far cheaper than the reference LeWM `H=5`, `S=100`, 30 AdamW steps, but makes the MPC path non-myopic and gives warm start a real tail.
   - Switch the agent to eval mode inside the inner solver while parameters are frozen, matching reference eval semantics.
   - Hypothesis: if v138 collapse was mainly an actor-planner feedback/initialization bug, v139 should avoid the post-500k KL explosion and sustain or improve the near-zero early planner behavior.
   - Cost impact: high; roughly an order of magnitude above v138's already-slow planner path.

6. `v140`: H=1 goal-embedding solver for unknown-goal control.
   - Split planning into two stages:
     - GoalSolver optimizes a free next obs-latent target `g` for high predicted value.
     - ActionSolver minimizes only `MSE(WM(z_t, a_t), stopgrad(g))`, matching the LeWM goal-distance control shape.
   - Keep `H=1` so the receding controller supplies temporal behavior.
   - Result: failed immediately. The free target exploited the critic/value head: early model return reached about `45` while real returns stayed poor. Mean-MSE reachability was too weak over the full latent.
   - Diagnosis: this proved the two-stage path runs, but not that the target is useful. A weak reachability penalty makes `g` a fantasy value target.

7. `v141`: summed reachability defect.
   - Keep the exact v140 structure, but make GoalSolver reachability use summed squared latent error instead of mean MSE.
   - ActionSolver still logs mean MSE for readability and still optimizes only distance to `g`.
   - Result: fixed the largest fantasy-target failure; goal value dropped from `~45` to near `0` after first planner windows and defect fell below `0.001`.
   - Cost impact: too slow at the original `S=4`, 8 GoalSolver steps, and 8 ActionSolver steps; planner SPS fell into the low hundreds.

8. `v142`: cheap H=1 goal-embedding solver.
   - Reduce planner cost to `S=1`, 4 GoalSolver steps, and 4 ActionSolver steps.
   - Result so far on HalfCheetah seed 1: last-10 return `-4.8` at 192k, improving sharply after planner activation; SPS about `367` by 196k.
   - Diagnostic: ActionSolver barely changes the action because the optimized goal remains close to the initial WM-predicted next obs. Current distance deltas are only about `-1e-6` to `-6e-6`. This means v142 is closer to selecting/perturbing a near-policy one-step target than a strong LeWM-style action solve.

9. `v143`: obs/action-seeded goal target, no optimized `a_seed`.
   - Remove the optimized GoalSolver witness action. Parameterize `g = current_obs_latent + delta`.
   - Initialize `delta` from the policy/action-predicted next obs latent: `WM(z_t, a_init)[:obs] - current_obs`.
   - Penalize distance from `g` to that action-predicted anchor; the separate ActionSolver remains responsible for finding an action to reach `g`.
   - Hypothesis: avoids mixing goal imagination with a second hidden action optimizer while still seeding the goal from both current state and policy/action structure.
   - Result on HalfCheetah seed 1 before stopping: last-10 return `0.5` at 208k. v142 was `-6.1` at 256k. Both were stopped because diagnostics showed they were not meaningfully exploiting differentiability.

## v140-v143 Investigation

- The ActionSolver autograd path itself is not detached: `action_u` receives gradients through `WM(z_t, a)` and the frozen WM parameters do not block gradients to the action variable.
- The practical action solve is nearly inert. In v142/v143, `goal_solver/action_distance_delta` stayed around `-1e-6` to `-6e-6`, so the final executed action is almost the seed/policy action.
- The strongest scaling bug is that ActionSolver minimizes mean MSE over `8 * 64` obs-latent coordinates, while LeWM's reference criterion uses summed final embedding distance. This divides the action-gradient signal by about `512`.
- v143 does not use a WM gradient path inside GoalSolver. It creates `anchor_obs = WM(z_t, a_init)` under `no_grad`, detaches it, then optimizes `goal_value - defect_to_anchor`. The WM only appears later in ActionSolver.
- v142 does use WM gradients in GoalSolver, but it optimizes a hidden seed action there; the later ActionSolver is initialized from that already-optimized action and becomes mostly redundant.
- The current planner takeover disables the learning anchor: after `planner/execute_frac = 1.0`, `planner_disable_pg=True` zeros real PPO/SPO policy-gradient terms and `planner_amortize_behavior=False` zeros planner cloning. That leaves the actor almost unchanged while a weak planner owns behavior.
- The reference LeWM GradientSolver is much stronger: external reachable goal embedding, summed final embedding distance, `H=5`, action blocks, `S=100`, and `30` AdamW steps. Our cheap variants used an invented critic target, `H=1`, `S=1`, four goal steps, four action steps, and mean action-distance MSE.

Next implementation target: a true H=1 direct action-gradient baseline should remove free-goal indirection, use a summed/scaled objective, keep PPO/SPO actor learning active unless the planner is clearly strong, and log gradient/action update norms. For a goal-embedding variant, the target must be generated in the action-tangent reachable space or selected from strong multi-start action rollouts, not optimized as a critic-only latent near a detached anchor.

10. Later: uncertainty and prior shaping.
   - Add a TD-MPC2-style uncertainty penalty only after the basic planner proves useful.
   - Add actor-prior regularization on action sequences if multi-sample optimization produces off-policy action targets.

11. `v144`: direct H=1 action-gradient planner. Rejected before evaluation.
   - Remove free-goal indirection from rollout planning and call the direct frozen-WM action solver.
   - Defaults: `H=1`, `S=4`, 16 Adam steps, objective scale `10`.
   - Keep real PPO/SPO actor learning active after planner takeover (`planner_disable_pg=False`) so planner behavior does not freeze the actor.
   - Log `direct_solver/{initial_cost,final_cost,cost_delta,action_abs_delta,action_l2_delta,last_grad_norm}`. This variant must show nontrivial action deltas and cost deltas before it is worth scaling to Hopper/Walker.
   - Status: abandoned because it is direct return MPC, not the intended "imagine optimal goal, then GradientSolver to get there" structure.

12. `v145`: optimal-goal H=1 MPC with LeWM-style GradientSolver.
   - GoalSolver optimizes a witness action through the frozen WM return objective and sets `g = WM(z_t, a_goal)[:obs]`.
   - ActionSolver starts from the policy/seed action and minimizes only summed final obs-embedding distance to `stopgrad(g)`.
   - This keeps `g` on the one-step WM action manifold and prevents free critic-fantasy latents.
   - Defaults: `H=1`, `S=4`, 8 GoalSolver Adam steps, 16 ActionSolver Adam steps, `planner_disable_pg=False`.
   - Diagnostics: summed action-distance initial/final/delta plus goal-action movement, executed-action movement, and action-gradient norm.
   - If H=1 shows meaningful `exec_action_*` and distance deltas, next scale-up is a longer witness/solver horizon (`H=3-5`) rather than returning to direct-return action MPC.
   - Early HalfCheetah seed 1 result: last-10 return `-10.6` at 128k and `-2.2` at 144k. First planner window (`131072`) showed real goal reaching: summed action distance `0.000162 -> 0.000006`, executed action moved `0.1465` mean absolute / `0.4373` L2 from seed, and goal witness action moved `0.1371` mean absolute from seed. Cost is high but acceptable for diagnosis: planner SPS about `343`.

13. `v146`: planner-owned optimal-goal H=1 MPC.
   - Same controller as v145, but removes the conceptual conflict with PPO policy gradients.
   - After planner takeover, the gradient-solver MPC is the behavior policy. PPO/SPO/PrefPoE policy-gradient terms are masked (`planner_disable_pg=True`).
   - The actor is retained only as an amortized warm-start/proposal network (`planner_amortize_behavior=True`), trained toward logged planner actions.
   - Critic/value and WM training still use real transitions; the actor is not competing with the planner for control.
   - Hypothesis: if v145's early performance was from the solver rather than PPO interference, v146 should keep the useful early behavior while making the architecture cleaner and closer to LeWM-style MPC.
   - Status: rejected before planner diagnostics. It still distilled planner actions into the actor, which makes the actor part of the algorithm rather than just legacy scaffolding.

14. `v147`: pure optimal-goal gradient-solver MPC.
   - No PPO/SPO actor objective, no PrefPoE actor objective, no planner-action distillation.
   - Data bootstrap uses random actions until `planner_start_step`.
   - Planner init defaults to zero plus multi-start action noise, not actor policy output.
   - After WM warmup, the two-stage gradient-solver MPC owns behavior:
     - optimize a WM witness action to imagine high-return reachable `g`;
     - execute a separate distance-only GradientSolver to reach `stopgrad(g)`.
   - WM, reward/continuation, and value/critic still train from real transitions. The actor network remains in the file only because the codebase couples actor and critic in one `Agent`.
   - This is the clean baseline for "use the WM differentiably"; actor distillation or learned proposal networks should be explicit later ablations for speed, not core behavior.
   - Status: stopped after finding residual actor amortization in the pre-planner/non-planner training path. Early HalfCheetah seed 1 signal was useful (`-2.4` last-10 at 128k after planner activation), but the implementation still allowed old v124 actor distillation before planner takeover, so it was not a clean pure-MPC result.

15. `v148`: corrected pure optimal-goal gradient-solver MPC.
   - Removes actor involvement from the executed path:
     - rollout values are computed by the critic directly, with no actor sample/logprob call;
     - bootstrap actions are uniform random, then planner actions;
     - planner init allows only `zero` or `random`, not actor mean/sample;
     - actor heads are frozen and excluded from the optimizer.
   - Removes the old PPO/SPO/PrefPoE/amortization training path from execution. After WM warmup, the only agent update is critic fitting to real rollout returns; control remains online differentiable MPC through the WM.
   - This is the actual answer to the distillation concern: no hidden behavior cloning, no PPO policy competing with the planner, and no learned actor cache. If we later want a proposal network for speed, it should be an explicit ablation against this baseline.
   - Early HalfCheetah seed 1 status: random-bootstrap phase at 96k was `-294.1` last-10, as expected. After planner activation it improved to roughly `-73` last-10 at 112k. SPS improved to about `4530` before planner activation because the actor-loss/inner-solver training path is gone.
   - Status: superseded by `v149` before completion because the actor-free training path exposed a logger bug: PPO metric scalars were still written unconditionally when only `v_loss` existed.

16. `v149`: pure-MPC bugfix restart.
   - Same algorithm as `v148`.
   - Fixes actor-free logging so missing PPO/entropy/KL scalars are skipped.
   - Extends random bootstrap to `global_step < max(planner_start_step, wm_warmup_steps)` so non-default warmup settings cannot fall through to midpoint actions before planner activation.
   - Removes the unreachable legacy PPO/amortization update block from the source; the live post-warmup agent update is critic-only.
   - Launched HalfCheetah seed 1 as `lewm_optgoal_purempc_h1_v149`.
   - Early HalfCheetah seed 1 status: random-bootstrap phase at 96k was `-294.1` last-10; first planner window improved to `-65.2` last-10 at 112k. Run is still active.
   - Final HalfCheetah seed 1 status: finished around `-216.4` last-20 at 976k after initially reaching near-zero returns. Diagnostics suggest an objective mismatch rather than a planner-gradient detach:
     - ActionSolver continued to reduce its goal distance (`0.00529 -> 0.000267` at 983k) and moved actions nontrivially (`~0.20` mean abs), so the distance solver was active.
     - GoalSolver's optimized model return drifted from slightly positive to negative (`0.085` at 622k to `-0.640` at 983k), mostly through the value term.
     - Reward action sensitivity collapsed (`std 0.0029` at 327k to `0.000165` at 983k), leaving GoalSolver with little immediate reward gradient.
     - Because v149 defines `g` as obs tokens only, it may discard reward/continuation outcome information from the high-return witness action before the distance solver executes.

17. `v150`: full-summary goal-distance ablation.
   - Same pure-MPC algorithm as v149, but `g` is the full predicted next latent summary, not only obs tokens.
   - ActionSolver minimizes summed distance to all tokens in `stopgrad(g)`, preserving reward/continuation outcome tokens from the optimized witness action.
   - Adds diagnostics `goal_solver/exec_model_return` and `goal_solver/exec_minus_goal_return` to measure whether ActionSolver preserves or destroys GoalSolver's model-return objective.
   - Launched HalfCheetah seed 1 as `lewm_optgoal_fullg_h1_v150`.
   - Early HalfCheetah seed 1 status: first planner windows are slightly better than v149 (`-62.3` last-10 at 112k vs v149 `-65.2`; `-0.8` last-10 at 128k). TensorBoard goal diagnostics have not flushed yet at the time of this note, so the return-preservation diagnosis is pending.
   - Stopped at 224k after confirming the same near-zero plateau. Diagnostics showed `goal_solver/model_return` and `goal_solver/exec_model_return` matching closely, so the two-stage GoalSolver -> ActionSolver handoff was not the main issue. The likely math bug is that MPC optimized reward decoded from generated outcome tokens, letting the optimized latent carry a favorable reward token without producing real forward progress.

18. `v151`: grounded action-aware transition reward.
   - Same pure optimal-goal H=1 MPC structure as v150.
   - Replaces planner reward in `cost_from_action` with an action-aware transition reward head `r(z_t, a_t, z_{t+1})`, trained directly on real rewards.
   - Keeps continuation from predicted outcome tokens for now, but removes the obvious reward-token exploit path from the differentiable return objective.
   - Fixes the scalar-gradient math bug found in v149/v150: predicted reward/termination losses now backprop through predicted latents instead of detaching at `decode_outcomes`, and the transition-reward loss also backprops into predicted next latents. Without this, MPC differentiates through scalar paths that the action-conditioned predictor was not trained to make reward-sensitive.
   - Launch command: `lewm_optgoal_actionreward_h1_v151` on HalfCheetah seed 1.
   - Early status: first planner window at 112k stayed near random-bootstrap returns instead of v150's near-zero jump. This likely confounds the detach fix with a newly initialized transition reward head that is not useful at planner takeover.

19. `v152`: isolated reward-gradient dynamics fix.
   - Copy v150 full-summary pure-MPC controller unchanged except for the concrete math bug found by review.
   - Change predicted MTP outcome decoding in WM training to `detach_summary=False`, so reward/termination prediction losses train the action-conditioned predictor latents, not just the inverse outcome heads.
   - This is the clean ablation to test the suspected collapse cause: v150's planner optimized reward/termination through predicted tokens that those scalar losses never trained to be action-sensitive.
   - Early HalfCheetah seed 1 status: recovered after planner activation but still plateaued near stationary behavior: `-75.8` mean at 112k, `-12.5` at 128k, `-2.3` at 144k. This is worse than v150 at the same steps (`-70.4`, `-0.25`, `0.56`), and v152's first model return was lower (`0.273` vs v150 `0.605` at the first goal diagnostics). The detach bug is real, but fixing it alone does not solve locomotion.

20. `v153`: direct reward-only MPC.
   - Drops the H=1 full-summary ActionSolver path. For H=1 it is redundant: GoalSolver's optimized action already reaches its own predicted goal, and the distance-only solver can only add optimizer error.
   - Executes the optimized action sequence from the frozen-WM action GradientSolver directly.
   - Defaults to `H=3`, 16 Adam steps, 4 samples.
   - Removes critic bootstrap from planner cost for this ablation; early planner return is reward-only so an untrained/undertrained value function cannot dominate action choice.
   - Keeps v152's `detach_summary=False` fix for predicted reward/termination training.
   - Final HalfCheetah seed 1 status: collapsed badly, ending around `-607.8` last-20 at 976k. Subagent review found no ignored-action or H>1 plumbing bug. Optimized actions were executed, warm-start shape was correct, and cost sign/termination handling were sane. Diagnostics showed the likely failure: reward surface was almost action-flat (`reward_action_sensitivity_range` around `0.002` late; planner grad norm around `3e-5`) while Adam still moved actions heavily (`exec_action_abs_delta` around `0.33`), so it optimized tiny reward-head artifacts into harmful real actions.

21. `v154`: direct reward MPC with one-step scalar supervision.
   - Same direct H=3 reward-only MPC mechanics as v153.
   - Keeps latent MTP for dynamics, but scalar reward/termination CE only trains `mtp_idx == 0`.
   - Rationale from subagent reward audit: MTP scalar reward/termination losses for offsets `>0` supervise future rewards from a shared timestep representation without intervening future actions, plausibly forcing the predictor trunk toward action-averaged future reward and flattening the immediate action-conditioned reward surface.
   - Launch target: `lewm_optgoal_direct_reward_onestep_h3_v154` on HalfCheetah seed 1, with `--diagnostics-interval 1` for early reward action-sensitivity comparison.
   - Stopped after reference review. Early diagnostics did not support continuing: one-step scalar supervision raised pre-planner reward action sensitivity, but reward bias worsened substantially and first planner window was worse than v153 (`-86.9` mean at 112k vs v153 `-71.9`).

22. Reference audit conclusion.
   - LeWM GradientSolver is not reward maximization. It solves actions to reach a fixed, real future goal embedding drawn from dataset/eval state. Cost is final predicted embedding MSE to a detached encoded goal. Our v150-v152 instead invented a goal by optimizing model return, and v153-v154 dropped goal reaching entirely for learned reward maximization.
   - TD-MPC2 is not reward-only gradient ascent. Its planner uses action-conditioned reward plus conservative terminal Q/uncertainty and a learned policy prior/search distribution. Our direct reward variants removed the terminal value/uncertainty guard and used a weaker reward readout from predicted outcome tokens.
   - High-confidence implementation mismatches still worth fixing before another planner result:
     - v150's predicted reward/termination CE detached predicted latents; v152+ fixed this.
     - Future-offset scalar MTP reward heads are not action-conditioned on intervening future actions; v154 tested one-step-only scalar CE but did not help enough.
     - WM/planner reward target is Gym normalized/clipped reward, not raw HalfCheetah reward; TD-MPC2 avoids online reward normalization and uses symlog two-hot scaling.
     - Goal diagnostics include a bogus zero `goal_defect`, so prior goal-reachability logs overstate correctness.
   - Next aligned implementation should pick one reference, not a hybrid guess:
     - LeWM-aligned: planner solves to fixed reachable future obs embeddings from real replay/high-return trajectories, with LeWM-scale samples/steps and multi-action execution.
     - TD-MPC2-aligned: add action-conditioned reward, target-Q ensemble/conservative terminal value, policy prior/CEM-style search, and remove online reward normalization from the model target.

23. `v155`: LEJEPA-correct reward-imagined goal.
   - User correction: this is a LEJEPA model, so embedding prediction should train the predictor, while reward/termination CE are probes and should not shape the predicted embedding dynamics.
   - Starts from v150 full-summary reward-imagined goal MPC, not the v153/v154 direct reward path.
   - Keeps reward/termination probes detached from predicted latents during training (`decode_outcomes` default detach), reverting the mistaken v152 interpretation.
   - Fixes the actual LEJEPA stop-grad issue: latent MSE now detaches target embeddings while keeping predicted embeddings attached.
   - GoalSolver selects the best imagined reachable latent by reward-probe score; ActionSolver then minimizes full-summary distance to that latent.
   - Reward probe uses symlog HL-Gauss targets with default support `[-3, 3]` in symlog space.
   - Fixes the bogus `goal_defect` diagnostic to report final goal-reaching distance instead of zero.
   - Launch target: `lewm_optgoal_lejepa_rewardgoal_h1_v155` on HalfCheetah seed 1.
   - Diagnosis at 256k: the graph is connected, but the optimized surface is nearly flat. GoalSolver changed actions by roughly `0.04-0.05` mean abs per dimension and ActionSolver reduced the latent goal defect by about 97%, yet GoalSolver's predicted reward improvement was only about `1e-4` and fell over time. The problem is not that ActionSolver fails to reach the imagined latent; it is that H=1 immediate reward does not expose a useful "best imagined reward state" objective.

24. `v156`: multi-step LEJEPA reward-imagined goal.
   - Direct response to the v155 learning bottleneck. v155 selected `g` from `WM(z_t, a_t)` and optimized immediate reward only, so it was not really differentiating toward a high-reward future state.
   - Keeps the LEJEPA split from v155: latent prediction is attached to the predictor, target embeddings are stop-grad, and reward/termination remain detached probes.
   - GoalSolver now optimizes an action sequence through the frozen WM/probe over `H=3` and scores `-sum gamma^h r_hat`. It then sets `g` to the final predicted latent summary at horizon H.
   - ActionSolver also optimizes an H-step action sequence, minimizing full-summary final-latent distance to `stopgrad(g)`. Rollout executes the first action receding-horizon style and warm-starts from the shifted tail.
   - Adds `goal_solver/goal_action_grad_norm` to distinguish reward-objective gradient strength from the existing ActionSolver latent-distance gradient norm.
   - Subagent review found no planner math or tensor-shape blockers. Launch target: `lewm_optgoal_lejepa_rewardgoal_h3_v156` on HalfCheetah seed 1.
   - Early HalfCheetah seed 1 status: first planner window improved sharply (`-61.5` mean group at 112k, then about `30.4` mean group at 128k). `score_runs.py --last 32` at 128k reports `-9.8 +/- 16.7` because the window includes both groups. This is much stronger than v155's near-zero plateau, but cumulative SPS after planner activation is only about `133`.

25. `v157`: faster multi-step LEJEPA reward-imagined goal.
   - Same objective as v156, but targets the measured planner bottleneck.
   - Adds `planner_action_block=3`: solve once for the H-step plan and execute the first 3 actions before replanning, LeWM-style. This should cut solve frequency by about 3x relative to v156's every-step replanning.
   - Initializes ActionSolver from the GoalSolver witness sequence that generated `g`, then uses `goal_action_steps=2` for refinement. Since that sequence already reaches `g` by construction, this removes most redundant goal-reaching backward passes without changing the distance-to-`g` objective.
   - Defers planner diagnostic `.item()` syncs until rollout summary and logs `planner/solve_frac`.
   - Boundary handling conservatively clears the buffered plan on any vector-env reset.
   - Sub-review result: do not launch as the main next result. It should be faster, but it changes semantics too much:
     - initializing ActionSolver from the same `best_goal_action` that generated `g` makes the distance solve nearly tautological and collapses toward direct reward-action MPC;
     - default `planner_action_block=3` with `H=3` executes the whole plan open-loop before replanning, weakening MPC.
   - Keep only as a possible explicit open-loop/direct-action speed ablation.

26. `v158`: safer faster multi-step LEJEPA reward-imagined goal.
   - Starts again from v156, not v157.
   - Preserves v156 controller semantics: replan every env step, initialize ActionSolver from the independent seed action sequence, and execute only the first action.
   - Performance changes are limited to lower-risk reductions:
     - `goal_action_steps=8` instead of 16, keeping a real distance-to-`g` refinement but halving the second-stage backward budget;
     - defer planner diagnostic `.item()` host syncs until rollout-summary logging.
   - This should give a modest speedup without invalidating the v156 result. Launch target: `lewm_optgoal_lejepa_rewardgoal_h3_fast_safe_v158` after sub-review.
   - Result: failed tradeoff. It improved cumulative SPS at first planner diagnostics (`183` vs v156's `133`) but underperformed badly:
     - 128k last-16: v158 `14.5 +/- 2.9` vs v156 `29.8 +/- 5.6`;
     - 144k last-16: v158 decayed to `4.0 +/- 2.0`, with one negative episode in the group.
   - Diagnostics point to the ActionSolver budget as the immediate regression: at 131k, final goal distance worsened from v156 `0.000279` to v158 `0.000859` (~3.1x), while scalar `exec_model_return` stayed deceptively similar. The real env appears sensitive to latent-goal mismatch that the learned reward probe does not penalize enough.
   - Do not continue this branch as the main line.

27. `v159`: diagnostic-sync-only isolation.
   - Starts from v156 and keeps `goal_action_steps=16`.
   - Only changes planner diagnostic aggregation to avoid per-step `.item()` syncs.
   - Purpose: run the exact v156 controller beyond the 128k positive window to determine whether later decay is intrinsic to the H=3 reward-goal planner, rather than caused by the v158 speed cut.
   - Launch target: `lewm_optgoal_lejepa_rewardgoal_h3_diagfast_v159`.
   - Early status: at 112k, last-16 is `-49.4 +/- 17.9`. This is still the first planner window, not enough to prove/disprove later decay.

28. `v160`: terminal-return imagined-goal MPC.
   - TD-MPC2-style correction to v159/v156's short reward-only GoalSolver.
   - GoalSolver cost becomes `-(sum_h gamma^h r_hat_h + c_v gamma^H V_hat(z_H))`.
   - Uses the existing HL-Gauss critic as terminal return prediction, with `planner_terminal_value_coef=1.0` and `planner_terminal_value_clip=10.0`.
   - Keeps the two-stage LEJEPA goal structure and full `goal_action_steps=16`; no direct-reward execution and no open-loop action blocks.
   - This is not full TD-MPC2: no Q ensemble, no uncertainty penalty, no actor prior/CEM. It tests whether adding return prediction fixes the myopic H=3 reward-goal objective before adding heavier machinery.
   - Launch target: `lewm_optgoal_lejepa_returngoal_h3_v160` after sub-review.
   - Killed at 176k after flatlining. Group means were much worse than v156:
     - v156: 112k `-49.4`, 128k `29.8`.
     - v160: 112k `-147.9`, 128k `-125.2`, 144k `-19.9`, 160k `12.8`, 176k `3.7`.
   - Independent review found no evidence of a detach/sign bug in the planner:
     - GoalSolver and ActionSolver both optimize differentiable action variables through `predict_next_latents_from_history` while WM/probe/critic parameters are frozen.
     - `cost_from_action` accumulates normalized reward, multiplies continuation forward, adds `gamma^H * continue * V(z_H)`, and returns negative return for minimization.
   - Diagnostics mostly rule out the two-stage handoff as the immediate flatline cause:
     - `exec_minus_goal_return` stayed tiny (`-0.00186` at 131k, `-0.000289` at 164k).
     - `action_final_distance` stayed small (`0.000603` at 131k, `0.000264` at 164k).
     - So ActionSolver reached/preserved the model-selected goal; the selected goal/objective was weak.
   - Diagnostics also rule out terminal-value clamp saturation:
     - `goal_solver/value` was only `0.588` at 131k and `0.290` at 164k, far from the `±10` clamp.
     - The planner signal weakened: `model_return` fell `0.845 -> 0.519`, `goal_action_grad_norm` fell `1.41e-4 -> 3.07e-5`, and action deltas shrank.
   - Supported root cause: v160 used the wrong kind of terminal return signal. It queried a single state-value critic on imagined terminal latents, trained from GAE over normalized/clipped rollout rewards, rather than a TD-MPC2-style action-conditioned conservative Q ensemble. The critic was poorly fit early (`explained_variance=-13.37` at 131k) and became a weak/unstable landscape for goal selection.
   - Do not continue with full-coef terminal `V` as the main line. A small coefficient sweep could confirm sensitivity, but the more principled next algorithmic step is a real terminal Q/conservative value objective or a cleaner reward-target calibration diagnostic, not another blind H/S increase.

29. `v161`: lookahead-return imagined-goal MPC.
   - Replaces v160's terminal `V_pi(z_H)` with online differentiable continuation return.
   - GoalSolver optimizes an action sequence of length `H+K` through the frozen WM/probe:
     - `J = sum_{t=0}^{H+K-1} gamma^t c_t r_hat_t`;
     - `c_{t+1} = c_t * p_continue_hat_t`.
   - At `H`, the continuation prediction context is reset to `[z_H]`, so the `K` lookahead estimates an online finite-horizon `V*_K(z_H)` rather than value of the full witness trajectory context.
   - The imagined goal is `g = stopgrad(z_H)`, not a candidate sampled terminal state and not a critic-favored free latent.
   - ActionSolver remains LeWM-style: optimize an independent `H`-step action sequence to minimize summed full-summary distance to `g`, execute only the first action, and replan every env step.
   - Defaults: `H=3`, `K=2`, `S=4`, `goal_solver_steps=8`, `goal_action_steps=16`, `planner_terminal_value_coef=0`.
   - Added diagnostics to distinguish objective quality from handoff quality:
     - `goal_solver/return_prefix_h`, `goal_solver/return_continuation_k`, `goal_solver/return_total_initial`, `goal_solver/return_total_final`;
     - `goal_solver/exec_total_return_with_goal_suffix`, `goal_solver/exec_minus_goal_total_return`;
     - `goal_solver/continue_product_h`, `goal_solver/continue_product_hk`;
     - `goal_solver/goal_action_prefix_abs_delta`, `goal_solver/goal_action_suffix_abs_delta`.
   - Pass/fail criteria:
     - GoalSolver should improve total and continuation return materially over the seed.
     - ActionSolver should keep final distance low.
     - `exec_total_return_with_goal_suffix` should stay close to `return_total_final`; otherwise full-summary goal distance is losing value-relevant details.
     - Real returns should beat v156's early positive window, not just recover to the near-zero plateau.

## Benchmark Discipline

- Change one mechanism per version.
- Keep default command shape at 16 envs and versioned experiment names.
- Run HalfCheetah first for early signal, then Hopper and Walker2d after the variant shows plausible learning.
- Watch `inner_solver/cost_delta`, `losses/amortize_loss`, policy KL, action saturation, and final benchmark score.
- Stop or deprioritize variants that add large solver cost without improving early return trend.
