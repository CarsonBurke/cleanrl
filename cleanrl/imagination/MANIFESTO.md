# Latent Imagination Manifesto

This folder is for PPO variants that use learned short-horizon future models without turning into Q-learning.

## Thesis

PPO does not need a stronger actor target on day one. It needs a better evaluator.

The imagination strategy here is:

- learn a short-horizon stochastic latent model
- keep PPO's clipped actor grounded in real GAE early
- let the model improve critic-side structure first
- only allow imagined advantages into the actor late, and only in small amounts

## Non-Negotiables

- No Q head
- No early actor steering from immature model rollouts
- No hidden representation takeover by the world model
- Short horizons only
- Benchmark on real MuJoCo returns, not proxy metrics

## Lessons So Far

### v1

Directly mixing imagined advantages into PPO too early creates bias. Shared encoder training also lets one-step prediction interfere with control features.

### v2

Detaching the model from the actor encoder and moving the model to critic-side auxiliary training is much safer.

### v3

The run begins to look useful only after the model has had time to mature. That suggests a delayed-coupling regime:

- critic-side model training from the start
- actor-side imagined advantage only after substantial training progress
- slow ramp to a small mixture coefficient

## Working Rule

Imagination is a late-phase signal amplifier, not an early-phase substitute for PPO.
