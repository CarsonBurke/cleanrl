# OPSD/JacTeach campaign - reproducibility record

Generated 2026-08-26. HalfCheetah-v4, seed 1 for every arm. All runs completed to 8M
(7,984,000 logged steps). Core config identical across every arm below:
`--num-envs 16 --num-steps 2048 --num-minibatches 32 --update-epochs 10 --learning-rate 3e-4 --total-timesteps 8000000 --seed 1`.


## Artifacts, exact

| arm | delivered dose | source file | run dir | mlq job |
|---|---|---|---|---|
| `incumbent_chassis` | 0.0 | `cleanrl/iterthink/v24_d3bucket/other/ppo_continuous_action_iterthink_v24_beta_d3bucket_mtp_entppo_v4_samplemask.py` | `runs/HalfCheetah-v4__entppo_v4_samplemask_a03_mask__1__1787102306` | (pre-existing, foreign to this session) |
| `aux_c011` | 0.0045 | `cleanrl/opsd/core/jacteach/ppo_continuous_action_entppo_jacteach_v1.py` | `runs/HalfCheetah-v4__entjac_aux_c011_8M__1__1787793298` | 3842 |
| `rot_c0p999` | 0.045 | `cleanrl/opsd/core/jacteach/ppo_continuous_action_entppo_jacteach_v2_rotate.py` | `runs/HalfCheetah-v4__entjac_rot_c0p999_8M__1__1787799296` | 3856 |
| `rot_c0p995` | 0.1 | `cleanrl/opsd/core/jacteach/ppo_continuous_action_entppo_jacteach_v2_rotate.py` | `runs/HalfCheetah-v4__entjac_rot_c0p995_8M__1__1787799296` | 3857 |
| `anneal_c0p995` | 0.1 | `cleanrl/opsd/core/jacteach/ppo_continuous_action_entppo_jacteach_v3_anneal.py` | `runs/HalfCheetah-v4__entjac_anneal_c0p995_a0p5_8M__1__1787804862` | 3865 |
| `anneal_c0p99` | 0.141 | `cleanrl/opsd/core/jacteach/ppo_continuous_action_entppo_jacteach_v3_anneal.py` | `runs/HalfCheetah-v4__entjac_anneal_c0p99_a0p5_8M__1__1787804862` | 3866 |

## Arm-specific flags (beyond the shared core config)

- `incumbent_chassis`: `--tr-mode mask --tr-sample-eps 0.1 --ent-alpha 0.3 --kl-beta 0.3` (file defaults)
- `aux_c011`: `--improve both --jac-cos 0.95 --jac-aux-coef 0.011`
- `rot_c0p999`: `--improve rotate --jac-cos 0.999`
- `rot_c0p995`: `--improve rotate --jac-cos 0.995`
- `anneal_c0p995`: `--improve rotate --jac-cos 0.995 --jac-anneal-frac 0.5`
- `anneal_c0p99`: `--improve rotate --jac-cos 0.99 --jac-anneal-frac 0.5`

## Return at matched checkpoints (last-20 mean +- CI95)

| arm | @1M | @2M | @3M | @4M | @6M | @8M |
|---|---|---|---|---|---|---|
| `incumbent_chassis` | 3062±52 | 5484±37 | 7235±65 | 8881±112 | 9857±107 | 10362±92 |
| `aux_c011` | 3154±128 | 5987±213 | 8144±100 | 8879±112 | 9838±217 | 10296±147 |
| `rot_c0p999` | 3497±57 | 6237±63 | 8030±263 | 9123±142 | 9789±122 | 9664±561 |
| `rot_c0p995` | 4310±118 | 7214±328 | 8547±129 | 9003±240 | 9275±222 | 9457±215 |
| `anneal_c0p995` | 4164±55 | 7050±78 | 8253±723 | 8173±955 | 9461±124 | 9808±161 |
| `anneal_c0p99` | 4511±54 | 7723±137 | 6327±1148 | 7106±1250 | 9368±162 | 9851±98 |

## Sample efficiency: steps to first reach last-20 >= threshold

| arm | 5000 | 7000 | 8000 | 9000 | 10000 |
|---|---|---|---|---|---|
| `incumbent_chassis` | 1.82M | 2.85M | 3.30M | 4.08M | 6.24M |
| `aux_c011` | 1.57M | 2.34M | 2.90M | 4.06M | 5.87M |
| `rot_c0p999` | 1.52M | 2.24M | 2.88M | 3.73M | 6.54M |
| `rot_c0p995` | 1.17M | 1.82M | 2.37M | 3.31M | never |
| `anneal_c0p995` | 1.23M | 1.94M | 2.51M | 3.50M | never |
| `anneal_c0p99` | 1.10M | 1.71M | 2.22M | 4.61M | never |

## Corrected incumbent scope

- **In-charter incumbent (HalfCheetah-v4, seed 1): 10362 +-92**, `entppo_v4_samplemask_a03_mask`, file `cleanrl/iterthink/v24_d3bucket/other/ppo_continuous_action_iterthink_v24_beta_d3bucket_mtp_entppo_v4_samplemask.py`. Zero OPSD_FAMILY.md §2 exclusion hits.

- **Best OPSD-family result: 8915** (`jacteach_cos0p95_8M`) - a family-scoped claim only.

- **EXCLUDED from incumbency** (OPSD_FAMILY.md §2, simulator-privileged probing: `qpos`/`qvel`/`qacc_warmstart` save-restore + counterfactual stepping, 11 hits each): `tpomd_alllayer_residual_td_v25_hc8m` 11015, `...tpoprobe_v1_k8_eta6_coef1_hc8m` 10371. Also excluded: `dg_ppoaux_c00ctrl_v10` 9224 (262k replay ring + frozen target critic).

- **Verdict of this session's 6 arms vs the incumbent: TIE at 8M** (best `aux_c011` 10296+-147 vs 10362+-92, overlapping CI95 and inside the lineage's own 423-point spread), with a **real monotone sample-efficiency gain** (steps to 8000: 3.30M -> 2.22M, 1.49x; steps to 10000: 6.24M -> 5.87M for `aux_c011`).


## Methodological correction: EV cannot see this bottleneck

| run | EV@8M | value_loss@8M | H@8M | return |
|---|---|---|---|---|
| `ppoadvnorm_batch_v1` (ancestor) | 0.970 | 8.73 | -8.80 | 8455 |
| `edmvalue_e2e_8m_v1` (deep V readout) | 0.952 | 7.60 | -10.69 | **9810** |
| `entppo_v4_samplemask` (incumbent) | 0.967 | 7.87 | -6.87 | **10362** |

The arm with the LOWER explained variance scores 16% higher. So explained variance is
not a valid instrument for 'is the critic the bottleneck', and my earlier inference
'EV healthy 0.92-0.99 => not a critic failure' does not support what I used it for.
`edmvalue_e2e_8m_v1` ran the IDENTICAL core config to the incumbent, so its +1355 over
its own ancestor is a config-clean, same-chassis-family, value-side result whose effect
GROWS with horizon (+5% @2M, +21% @4M, +21% @6M, +16% @8M) - the opposite signature to
this session's actor-direction mechanism, which decayed to 0 by 4M at every dose.

