# TD7 variants

Off-policy TD7 continuous-control line (base `../td7_continuous_action.py`
stays flat). MuJoCo benchmark, seed 1.

Subfamilies:

- `lesale/` — LEWM/SALE latent-action world-model line (compile, JEDI,
  LE-JEPA, LEWM-rollout threads).
- `stocksig/` — stock-signal + state-dependent noise + LE-JEPA reward/policy
  heads (dual-proj incl. WPPG variants, HL-Gauss reward betas, isometric/
  latent64/semantic64, PC-actor).
- `ctx/` — context-conditioned actor/critic (v1-v3).
- `noise/` — exploration-noise sweeps (betanoise, sdnoise, softbeta, softgauss).
- `misc/` — hop/hist/hvm memory, search, stepmem, tap, rank-greedy,
  delightful-PG port.
