# IDBD family (PPO + Incremental Delta-Bar-Delta)

## Reference

- **Paper**: Sutton (1992), *Adapting Bias by Gradient Descent: An Incremental Version of Delta-Bar-Delta*  
  Local: [`sutton_1992_idbd.pdf`](sutton_1992_idbd.pdf) · text extract: [`sutton_1992_idbd.txt`](sutton_1992_idbd.txt)
- **Linear LMS demo** (α init = exp(β) = 0.05, θ = 0.05): [`idbd_linear_demo.py`](idbd_linear_demo.py)

### Paper facts we use

- Per-input step sizes: `α_i = exp(β_i)` (eq. 4)
- Updates (eqs. 5–6 / Fig. 2): β ← β + θ δ x h; α ← e^β; w ← w + α δ x; h ← [1−α x²]₊ h + α δ x
- **Init (Exp. 1–2)**: “The βᵢ … were set initially such that **αᵢ = 0.05**, for all i”
- Long-run relevant α ≈ **0.13**; irrelevant α → very small; fixed-α sweep 0.05–0.25
- Meta-rate θ is the free parameter (paper long run uses θ = 0.001; demo uses 0.05)

## Versions

| File | Notes |
|------|--------|
| `ppo_continuous_action_idbd_v1.py` | Gaussian actor, IDBD, α_init=3e-4 (PPO Adam default — **wrong vs paper**) |
| `ppo_continuous_action_idbd_beta_v1.py` | Unimodal Beta actor (v215-style), same wrong α_init |
| `ppo_continuous_action_idbd_beta_v2.py` | Beta + **α_init=0.05** (paper), raised α cap |
| `ppo_continuous_action_idbd_beta_v2_abl_*.py` | Paper-gap ablations (see [`ABLATIONS.md`](ABLATIONS.md)) |

Submit from the repo root through the machine-wide ML queue:

```bash
mlq submit --name idbd_beta_v2 --max-parallel-runs 3 --cwd "$PWD" -- \
  .venv/bin/python -u cleanrl/idbd/ppo_continuous_action_idbd_beta_v2.py \
  --env-id HalfCheetah-v4 --num-envs 16 --exp-name idbd_beta_v2 \
  --total-timesteps 8000000 --seed 1 --compile --compile-mode reduce-overhead
```
