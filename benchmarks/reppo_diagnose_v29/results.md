# REPPO latent-roll diagnostic results

Window means; absent windows are not extrapolated. Queue outcomes are separate.

| Run | Recorded M steps | 6–8M | 10–12M | 14–16M |
|---|---:|---:|---:|---:|
| reppo_diagnose_v29_h8_noreward_16M | 16.088 | 9,888 | 11,232 | 11,544 |
| reppo_diagnose_v29_h8_reward_16M | 16.088 | 10,441 | 11,376 | 11,517 |
| reppo_diagnose_v29_parent_lam08_16M | 16.088 | 8,757 | 10,165 | 10,944 |
| reppo_latentroll_v27_h8_50M | 15.854 | 10,441 | 11,376 | 11,514 |
| reppo_refresh_v25_lam0p8_50M | 49.987 | 8,757 | 10,165 | 10,944 |

2 combined oracle reports available; raw statistics are in results.json.

Oracle conclusions must distinguish first-action changes from full-policy changes,
frozen pre/post critics, soft versus raw returns, and confidence intervals.

## Simulator policy comparisons

Direct 1000-step continuing-state soft returns; fixed pre-update temperature, no critic tail.
Intervals resample environment trajectories; they do not include training-seed variation.

| Arm / checkpoint | Update | Critic predicted first-action gain | Simulator gain | Trajectory 95% interval | Actual minus predicted 95% interval | Full-policy raw discounted gain |
|---|---|---:|---:|---:|---:|---:|
| reppo_diagnose_v29_h8_reward_16M / 12.009M | last_actor_update | 0.00023 | 0.16029 | [0.01456, 0.29996] | [0.01452, 0.29851] | 3.9661 |
| reppo_diagnose_v29_h8_reward_16M / 12.009M | update | 0.00516 | 0.01704 | [-0.14060, 0.15058] | [-0.14508, 0.14539] | 10.5554 |
| reppo_diagnose_v29_h8_reward_16M / 8.011M | last_actor_update | 0.00122 | -0.01370 | [-0.11510, 0.08611] | [-0.11659, 0.08512] | 1.1824 |
| reppo_diagnose_v29_h8_reward_16M / 8.011M | update | -0.00049 | -0.05184 | [-0.14379, 0.04765] | [-0.14282, 0.04781] | -5.0800 |

## Finite-difference evidence

300-step soft labels retain critic bootstrap tails, including at earlier time limits.

The legacy pooled metric is mean directional gain divided by RMS oracle gradient magnitude,
not ordinary pooled cosine. Heavy-tailed gradient norms can lower it even with aligned directions.

| Arm / checkpoint | Subset | Gradient center | Normalized directional gain (h=.2) | SE | Within-state advantage correlation |
|---|---|---|---:|---:|---:|
| reppo_diagnose_v29_h8_reward_16M / 12.009M | all | star | 0.1251 | 0.1831 | 0.1541 |
| reppo_diagnose_v29_h8_reward_16M / 12.009M | all | a0 | 0.1448 | 0.0479 | 0.1541 |
| reppo_diagnose_v29_h8_reward_16M / 12.009M | cruising | star | 0.1428 | 0.1831 | 0.1483 |
| reppo_diagnose_v29_h8_reward_16M / 12.009M | cruising | a0 | 0.1461 | 0.0538 | 0.1483 |
| reppo_diagnose_v29_h8_reward_16M / 8.011M | all | star | 0.1365 | 0.0573 | 0.1996 |
| reppo_diagnose_v29_h8_reward_16M / 8.011M | all | a0 | 0.1719 | 0.0511 | 0.1996 |
| reppo_diagnose_v29_h8_reward_16M / 8.011M | cruising | star | 0.1484 | 0.0611 | 0.2832 |
| reppo_diagnose_v29_h8_reward_16M / 8.011M | cruising | a0 | 0.1621 | 0.0562 | 0.2832 |
