# Critic uncertainty

Can a critic that knows how wrong it is improve PPO's advantages?

Base: `awr/ppo_continuous_action_ppo_sepclip_control_v1.py` (reference run `sepclip_control_8M`, HalfCheetah 8M ≈ 8.4k).

The critic is a bootstrapped 4-member ensemble, with V as the ensemble mean. A separate
variance net regresses two quantities per state:

- `total(s)`: E[Â² | s], where Â is the advantage from a GAE pass at fixed λ0
- `epistemic(s)`: the distilled variance across ensemble members

| `--uncertainty-mode` | Change |
|-|-|
| `none` | Ensemble-mean critic at fixed λ. Control for the two modes below. |
| `lambda` | Per-state GAE λ with log-odds(λ) = β log(ρ/(1−ρ)), where ρ = epistemic/total. The batch mean of λ is held at λ0. |
| `weight` | Mean-1 policy-loss weights per state ∝ total(s)^−p (inverse-variance). |

Why per-state λ and not advantage weighting: a baseline that depends only on the state never
biases the policy gradient, so the critic's error does harm only where GAE bootstraps through V.
Inverse-variance mixing of the bootstrap V (error e²) against the continued return (noise σ²)
gives λ = e²/(e²+σ²). Action-dependent confidence weights would also bias the policy toward
familiar actions, which works against exploration.

Per-step λ is supported by the shared GAE (`ppo_loop.compute_gae`, Triton kernel in `shared/gae.py`).

## Results (HalfCheetah-v4, 8M, seed 1; final = mean of last 20 episodes)

| Run | default init | `--match-base-init` |
|-|-|-|
| `none` (ensemble control) | 6162 | **9017** |
| `lambda` β=1 | 5550 | 4491 |
| `lambda` β=0.5 | 5577 | — |
| `weight` p=1 | 5303 | 8152 |
| `none --no-bootstrap` | 6439 | — |

The reference `sepclip_control_8M` scores 8454.

- The ~2.3k drop of the first control came from **actor initialization**, not from the
  ensemble. Building the extra critics first shifted the actor's RNG draw. With the base
  draw replayed, the same control scores 9017. Init/seed variance on this base is therefore
  over 2k at 8M, and single-run gaps smaller than that are weak evidence.
- Both uses of uncertainty lost to their own control under both inits: per-state λ by
  −600 / −4500, and state weights by −860 / −865. Their signs agree across the two inits.
- The uncertainty signal is weak. Predicted vs realized log Â² correlates at 0.15–0.34, so
  advantage noise varies only modestly between states. Per-state λ ranged over roughly
  0.89–0.99 (10th–90th percentile).
