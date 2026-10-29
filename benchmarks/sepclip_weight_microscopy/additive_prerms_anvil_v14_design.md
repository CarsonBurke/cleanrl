# Additive pre-RMS actor, v14

The candidate is an architecture package: signed input projection, pre-RMS
residual branches, learned channel amplitudes, and unconstrained actor body
matrices. It is not a normalization-placement-only ablation. The original
v12 actor remains available through `--actor-architecture original`; that path
retains construction order, initialization RNG, normalization, matrix ownership,
host mirror, PPO loss, rollout-clock optimizer equations and critic.

For normalized observation x, the additive actor computes:

```
h0 = P x + p
h1 = h0 + sigmoid(g1) * SiTU1(RMS(h0))
h2 = h1 + sigmoid(g2) * SiTU2(RMS(h1))
logits = readout_gain * (W_head h2 + b_head)
alpha, beta = chunk(1 + softplus(logits))
```

Width is 64, both SiTU branches have bottleneck width 43, RMS epsilon is 1e-5
and RMS has no learned affine parameters. There is no branch-output norm or
final stream norm. Consequently the projection provides a signed linear path
through both blocks, while nonlinear branches can change stream amplitude.
Pre-RMS limits branch-input scale without deleting the identity stream's
magnitude. Sigmoid channel amplitudes use the established pre-RMS initialization
g1 = g2 = -1.5 (initial amplitude approximately .1824); they remain learnable.

Initialization uses the shared pre-RMS initializer: orthogonal projection gain
sqrt(64 / observation_dim), zero projection bias, SiTU gate/up initialization
targeting preactivation second moment 2, and down matrices targeting raw branch
second moment .5. Those are isotropic-input distributional targets, not measured
guarantees for the MuJoCo observation distribution. The fixed output multiplier
is 1: projected components have mean squared scale approximately 1, matching
the original actor's `8 * unit_direction` component mean square 1. The candidate
retains fluctuations in stream scale that the original actor removes.

`normalize_matrices()` leaves all additive body weights unconstrained, including
the projection, so its reference initialization scales survive. The head keeps
v12 unit-row weight retraction, readout gains and bias. The critic is precisely
the original residual SiTU critic with its original matrix retractions.

For actor-only Anvil, all seven body matrices belong exclusively to the existing
MatrixOptimizer: P and each branch's gate, up and down. Projection bias, two
channel amplitude vectors, and head weight/bias/readout gain belong to fused
Adam, together with the unchanged critic. No actor body matrix is silently
assigned to Adam. `--actor-optimizer adam` assigns every parameter to the original
fused Adam control. Existing method, precision, clipping and PPO flags remain
available without modifying their defaults.

For HalfCheetah (17 observations, 6 actions), original actor parameters are
13,262 and additive actor parameters are 18,584, an increase of 5,322 (40.13%).
The new projection contributes 1,152 parameters (1,088 weights and 64 biases).
Changing the first branch input from 17 to 64 adds 4,042 parameters; the two
amplitude vectors add 128. The Beta head stays at 792 parameters, and the critic
stays at 12,536. Therefore score changes cannot be attributed solely to norm
placement or solely to optimizer geometry; the learned function family and
parameter budget also change.

The additive host mirror lowers the complete trunk and linear head through
`host_graph.make_host_mirror`, which natively supports this pre-RMS SiTU graph,
then multiplies the existing readout gain in place before Beta sampling.
There is no unfused fallback for this trunk. Original manual mirror composition
is retained for the original control. All rollout, normalization, timing and
transition/update microscopy remain v12; snapshots contain all new parameters
and `args.actor_architecture`. Reload with that architecture, not an assumed
original Agent. The optional evaluation factory also preserves this choice.

The discriminating comparisons are additive Adam versus additive Anvil, and
each against the verified original control. Improved returns for both additive
optimizers would support the architecture package; improvement only for Anvil
would suggest an interaction with its matrix updates. Neither result alone
identifies which package component caused the change or establishes a 20k
attainable optimum. No new training or performance claim is made by this source
change. The parent queues CUDA parity, host/compile, ownership and checkpoint
contracts before fresh 8M HalfCheetah seed-1 training.
