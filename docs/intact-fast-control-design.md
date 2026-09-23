# INTACT: direct control and a measured learner budget

The September 19 constraint is at least 40,000 environment transitions/second
as a target, with direct action selection and no candidate search. Return and
end-to-end throughput are joint acceptance criteria. Extra planning computation
is not an acceptable substitute for learning a useful control mapping.

## Evidence and immediate experiment

The v9 job 8392 was cancelled by request at approximately 6.6M transitions.
The coordinating agent did not issue that cancellation. Its final recorded
last-20 episode return is 3148.3; matched 1M/2M/6M returns are
701.6/1690.6/3127.1. This is partial, seed-1 evidence, not an 8M result.

A live interval measured 27,188.6 SPS with per-rollout update 0.415 s,
action inference 0.03429 s, environment 0.1313 s, and GAE 0.004728 s.
These are one interval, not a distribution or an aggregate concurrency study.
At 16,384 transitions/rollout, 40k SPS allows 0.4096 s total. Holding other
costs fixed requires update time near 0.22 s, approximately a 47% reduction.
Inference fusion by itself cannot close this gap. Final interval metrics were
28,071.6 SPS and 0.3884 s update; normal interval variation matters.

v10 is an execution comparator preserving v9's learning objectives, optimizer
ownership, clipping, epochs, minibatches, and projection count. It moves
indexing inside the compiled loss, packs metrics, and compiles tensor-only
gradient clipping without manual CUDA graph capture. Numerical trajectory
identity is not promised across fused floating-point reductions.

`scripts/benchmark_intact_update_v10.py` measures matched production-size
learner workloads in separate processes and writes JSON plus TensorBoard
metrics. Synthetic update timing cannot establish environment SPS, sample
efficiency, aggregate throughput, or improved returns. Full 8M seed-1 native
HalfCheetah runs remain necessary. Do not treat smaller regularizer projection
counts or fewer epochs as execution-only changes.

The shared manual capture implementation is explicitly experimental and unsafe
with compiled peer inference. No production integration is justified by an
isolated microbenchmark pass.

Validation jobs submitted September 19: v9 learner benchmark **8430**, v10 CUDA
equivalence contracts **8431**, and v10 learner benchmark **8433** (requires
8431 success). All use maximum parallel runs **1**, ordinary priority **0**,
and a **20-minute** attempt limit. These identifiers record submissions, not
successful tests or performance measurements. No automatic retries or culling
are configured for these bounded validation workloads.

Moving gathers inside a CUDA-graphed loss may copy more input data per call
because the complete rollout is now an argument. The benchmark deliberately
passes that batch explicitly, matching production. A net speedup is not assumed.

## Architectural direction

The next scientific comparison should train an action-conditioned critic on
real `(state, action, reward, next_state, terminated)` transitions. Its target
is `reward + gamma * (1 - terminated) * target_Q(next_state, target_action)`.
Time limits bootstrap from the factual final observation, never an autoreset
observation. A small actor learns an amortized action mapping using this
critic. No candidate rollout, intent refinement, or test-time optimization is
needed. Critic derivatives remain an approximation and need empirical checks;
removing imagined-model derivatives does not make critic gradients exact.

[TD3](https://proceedings.mlr.press/v80/fujimoto18a.html) provides an established
reference for twin critics, target networks, and delayed actor updates.
[TD7/SALE](https://arxiv.org/abs/2306.02451) is a relevant representation-learning
comparison. Neither reference establishes that adding our JEPA objectives will
improve this repository's scores or throughput.

Keep the deployment graph small. Compare a direct action head with a
prescriber-through-inverse-law head under the same factual critic, data,
training budget, and exploration law. The latter retains INTACT's control
interface but can constrain actions and drift when auxiliary world learning
updates its law. A direct head removes that specific coupling; it should be
called a JEPA actor-critic unless evidence establishes an INTACT mechanism.

Do not optimize an arbitrary latent goal density merely because the encoder
provides latent coordinates. Many latent goals can yield the same action;
action-space value is the relevant comparison. Predictive accuracy alone does
not demonstrate control utility. Retain an auxiliary loss only when its
benchmark benefit justifies its measured compute cost.

Replay, target-network timescales, changing observation normalization, and
actor-gradient conditioning must be designed together before implementation.
Stored normalized states become inconsistent as running statistics change;
store raw observations or specify a fixed representation contract. The current
v9 rollout TD(lambda) value target is not an off-policy action critic and
renaming its weighted likelihood objective would not implement this design.

For a retained stochastic Beta law, the actor objective should evaluate an
actual differentiable action sample, not only the distribution mean. A mean-only
objective does not train concentration according to sampled-action returns.
One sample is a Monte Carlo gradient estimator during learning, not candidate
search at deployment. Twin critics can consume normalized observations and
previous actions directly, plus detached JEPA features, so their only input is
not a moving latent coordinate system. Give critic, prescriber, and auxiliary
world parameters explicit gradient ownership. An entropy term changes the
control objective and must be declared and compared rather than silently added.

## Decision criteria

1. Validate equivalent losses, gradients, clipping ownership, and repeated
   compiled updates before measuring v10 speed.
2. Measure warmed production-size learner time separately from compilation.
   Preserve full-training SPS and return curves as the deployment evidence.
3. Compare matched-step returns and elapsed time to those returns; reject a
   speed improvement that merely removes effective learning.
4. Characterize aggregate SPS/core before raising concurrency above one.
5. Profile the regularizer and optimizer tail before any schedule reduction.
   Treat a changed regularizer or update ratio as a named algorithm experiment.
