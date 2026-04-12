# HL-Gauss Transformer PPO Variants

This folder contains the self-attention PPO forks explored for the HL-Gauss continuous-control line.

Files:
- `ppo_continuous_action_hlgauss_silu_dlogstd_tokenaffine_sa_transformer_base_v1.py`
- `ppo_continuous_action_hlgauss_silu_dlogstd_tokenaffine_sandwich_smallinit_v2.py`
- `ppo_continuous_action_hlgauss_silu_dlogstd_tokenaffine_sandwich_smallinit_headnorm_v3.py`
- `ppo_continuous_action_hlgauss_silu_dlogstd_tokenaffine_sandwich_smallinit_clean_v4.py`
- `ppo_continuous_action_hlgauss_silu_dlogstd_tokenmlp_sandwich_smallinit_v5.py`
- `ppo_continuous_action_hlgauss_silu_dlogstd_tokenmlp_prenormonly_v6.py`
- `ppo_continuous_action_hlgauss_silu_dlogstd_tokenmlp_periln_smallinit_v7.py`
- `ppo_continuous_action_hlgauss_silu_dlogstd_tokenmlp_periln_xaviertrunk_v8.py`
- `ppo_continuous_action_hlgauss_silu_dlogstd_tokenmlp_periln_xaviereverything_v9.py`
- `ppo_continuous_action_hlgauss_silu_dlogstd_tokenmlp_periln_xaviertrunk_cleanembed_v10.py`
- `ppo_continuous_action_hlgauss_silu_dlogstd_tokenmlp_periln_xaviertrunk_cleanembed_dg_v10_1.py`
- `ppo_continuous_action_hlgauss_silu_dlogstd_tokenmlp_periln_xaviertrunk_cleanembed_unitscale_unscaledqkv_v11.py`
- `ppo_continuous_action_hlgauss_silu_dlogstd_tokenmlp_periln_xaviertrunk_cleanembed_vnormonly_v12.py`
- `ppo_continuous_action_hlgauss_silu_dlogstd_tokenmlp_periln_xaviertrunk_cleanembed_unitscalesdpa_v13.py`

## HalfCheetah-v4 Results

Metric:
- `charts/episodic_return`, last 20 episodes, scored with `./.venv/bin/python scripts/score_runs.py ... --env HalfCheetah-v4 --last 20`

Note:
- Early runs used older experiment names such as `hlgauss_silu_dlogstd_sa_transformer_v10`. The filenames here were renamed later for clarity.

| Variant | File | Mean | ±CI95 | Avg all | Steps | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `v10` | `tokenmlp_periln_xaviertrunk_cleanembed_v10` | 4843.9 | 66.0 | 3065.2 | 6,480,000 | Best run so far. Clean embed path + Peri-LN + Xavier trunk. |
| `v6` | `tokenmlp_prenormonly_v6` | 3962.8 | 158.5 | 2227.8 | 7,984,000 | Pre-norm only. Much healthier PPO update stats than sandwich/Peri-LN variants. |
| `v8` | `tokenmlp_periln_xaviertrunk_v8` | 3796.0 | 88.5 | 2356.5 | 6,160,000 | Peri-LN with Xavier trunk. Strong, but below `v10`. |
| `v10.1` | `tokenmlp_periln_xaviertrunk_cleanembed_dg_v10_1` | 2998.4 | 42.8 | 1617.3 | 3,136,000 | Corrected DG run. Improved over early DG attempt, still below `v10`. |
| `v5` | `tokenmlp_sandwich_smallinit_v5` | 2077.4 | 370.0 | 1161.5 | 2,352,000 | Token MLP embedder was a major upgrade over affine tokenization. |
| `v12` | `tokenmlp_periln_xaviertrunk_cleanembed_vnormonly_v12` | 1536.2 | 93.7 | 619.9 | 1,792,000 | Isolated unscaled `v`-norm ablation. Underperformed `v10`. |
| `v4` | `tokenaffine_sandwich_smallinit_clean_v4` | 1113.4 | 22.9 | 393.2 | 1,648,000 | Reverted the failed CLS head norm change. |
| `v11` | `tokenmlp_periln_xaviertrunk_cleanembed_unitscale_unscaledqkv_v11` | 1042.6 | 24.8 | 329.0 | 1,920,000 | Combined unit SDPA + scaled QK norm + unscaled `v` norm. Underperformed. |
| `v13` | `tokenmlp_periln_xaviertrunk_cleanembed_unitscalesdpa_v13` | 530.5 | 45.2 | 29.4 | 1,168,000 | Isolated unit-SDPA ablation. Underperformed. |
| `v7` | `tokenmlp_periln_smallinit_v7` | 438.5 | 63.6 | -30.4 | 640,000 | Peri-LN structure with small-init trunk. Worse than `v6`. |
| `v1` | `tokenaffine_sa_transformer_base_v1` | 101.9 | 41.2 | -158.4 | 624,000 | Initial affine-token transformer baseline. |
| `v3` | `tokenaffine_sandwich_smallinit_headnorm_v3` | -105.4 | 22.4 | -225.0 | 400,000 | Extra CLS head RMSNorm made PPO hotter, not cooler. |
| `v9` | `tokenmlp_periln_xaviereverything_v9` | -141.4 | 32.1 | -386.4 | 368,000 | Xavier on PPO heads was a bad transfer from LM init practice. |
| `v2` | `tokenaffine_sandwich_smallinit_v2` | -327.2 | 31.7 | -340.9 | 96,000 | Smoke-only check. Not a meaningful benchmark. |

## Other Envs

Only early sanity runs exist so far:

| Env | Variant | Mean | Steps |
| --- | --- | ---: | ---: |
| `Hopper-v4` | `v1` | 123.7 | 110,528 |
| `Walker2d-v4` | `v1` | 19.2 | 110,032 |

These are not enough to compare variants. The meaningful benchmark trail so far is HalfCheetah-focused.
