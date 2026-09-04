# TPO — Trust-region Planning Optimization

`ppo_continuous_action_tpo_*`: intra-trajectory Beta critics with KL control,
dynamics-eta, factored/no-advnorm/selfnorm variants.

- `md/` — TPO-MD (mirror-descent + one-step probe dynamics): prednext,
  all-layer, horizon-tape, local-PC-opt, spread-temp and think-trunk threads.
