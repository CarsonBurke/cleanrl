# INTACT v9: reward-weighted action likelihood through the inverse law

## Hypothesis

INTACT may provide a useful policy parameterization without requiring either
PPO or accurate derivatives of predicted returns. v9 trains a deterministic
intent prescriber through the conditional action law using positively weighted
likelihood of actions actually executed. Weights come from real-reward GAE
with an empirical KL constraint against uniform transition selection.

This is advantage-weighted maximum-likelihood policy fitting, not a novel
optimizer. The research hypothesis concerns the inverse-law parameterization
and its interaction with predictive representation learning. There are no v9
performance results established by this design document.

## Exact actor objective

Let `x` denote observation and previous physical action, `z = encoder(x)`, and
`g = prescriber_theta(z, previous_action)`. The actor distribution is:

```
pi_theta(a | x) = inverse_law(a | z, g, previous_action)
L_actor = -sum_i weight_i * log pi_theta(action_i | x_i)
```

During differentiation of this loss, encoder and inverse-law parameters are
frozen. Gradients pass through the law's intent input into the prescriber. They
are the exact chain-rule derivative of the current deployed action
distribution's likelihood. They are not a surrogate for the environment's
action derivative. An inaccurate dynamics or reward model therefore cannot
inject an imagined-return gradient into this actor objective.

The ideal unrestricted conditional fit recovers the action marginal of the
weighted factual data. With noisy advantages this marginal depends on expected
sample weights conditional on state and action; it is not generally obtained
by exponentiating an exact expected advantage. Finite data, constrained policy
expressivity, and approximate optimization weaken the idealized statement.

## Why this fits an action quotient

Two intent vectors that produce the same conditional action distribution have
the same actor loss. The prescriber need not reproduce a particular latent
endpoint or choose among action-equivalent endpoints using a Euclidean latent
distance. This removes v8's explicit endpoint-density fit and Gaussian sampling
around its learned goal means.

The law still receives factual local and future-goal inverse supervision, and
the world representation still receives its predictive objectives. Those
objectives are intended to make intent-conditioned action control a useful
inductive bias. They do not prove that a selected intent causes a corresponding
future state. v9 is an action policy with an inverse-law interface, not a
validated goal-reaching planner.

## Comparisons and limitations

- **Weighted versus uniform v9:** isolates real-return selection within the
  same architecture and auxiliary learning. With a fixed law and encoder,
  uniform fitting should approximately reproduce behavior. Substantial drift
  warrants checking fitting error and changes caused by auxiliary updates.
- **v9 versus v6:** with matching world objectives, compares action likelihood
  fitting against PPO and its model control variate. v6's completed seed-1
  HalfCheetah reference has final last-20 return 5,668.2; compare at matched
  timesteps rather than against this endpoint prematurely.
- **v9 versus v8:** compares complete algorithms. v8 also fits a stochastic
  goal mixture and reward-weights the action law itself; v9's actor only updates
  the prescriber, while factual inverse supervision updates the law. Their
  difference cannot be attributed solely to endpoint versus quotient fitting.
- **Eventual direct-actor control:** train a direct Beta policy with identical
  weights, rollout settings, critic targets, and comparable capacity. If it
  matches v9, the weighting method may work while INTACT has no demonstrated
additional benefit.

## Initial execution record

On 2026-09-19, the coordinating agent submitted CUDA behavioral contracts as
`mlq` job **8391**, with declared maximum parallel runs **1** and a 20-minute
limit. The H8 HalfCheetah training run is job **8392**, dependent on successful
completion of 8391, with declared maximum parallel runs **1** and a two-hour
limit. Its requested benchmark is 8M steps, seed 1, 16 native environments,
and two environment threads. Both were reported queued when this record was
written; these submissions are not test results or training evidence.

The inverse law can restrict attainable action distributions or have poorly
conditioned intent sensitivity. A deterministic intent and conditional
factorized Beta also lack v8's mixture-induced action multimodality. Leaving
factual intent support can expose those limitations, although the action
likelihood remains exact for the distribution actually executed.

World updates change encoder and law parameters independently of the actor
objective. Freezing them for actor differentiation does not freeze deployment
across an update. Inspect encoder/law drift separately from prescriber changes.
The global empirical KL constraint limits sample-weight concentration; it does
not bound deployed policy KL or guarantee monotonic return improvement.

## Evidence required

Use full seed-1 HalfCheetah benchmarks and matched 1M, 2M, and 8M comparisons;
report final last-20 returns separately. Monitor empirical selection KL, ESS,
weighted action NLL, conditional entropy, action-law response to prescribed
versus altered intents, gradient norms, and behavior changes across updates.

Improved likelihood without improved real return is not success. Poor fitting
with weak intent sensitivity suggests a restrictive inverse-law interface.
Good fitting but poor returns leaves weight quality, exploration, finite-data
generalization, and auxiliary behavior drift as competing explanations. A
weighted advantage over uniform supports reward selection; only matched
direct-actor evidence can establish a benefit from the INTACT structure.
