# Successor-latent critic v3

Status: CUDA contracts passed; the fixed-policy experiment completed without
passing its promotion gate. Overall gradient validation is inconclusive because
reference signal/noise is low; reward decoding independently misses its threshold.
There is no measured policy-performance improvement.

This follows the named v2 negative result but changes the central learned object.
It is not another family of scalar-return control coefficients. No other method
source was inspected. The v2 infrastructure and its named gate artifact were the
only existing research files read.

## Learned object

Let x_t = [normalized s_t, 2*a_t-1, (normalized s_(t+1)-normalized s_t)/sqrt(2), 1].
The final indicator distinguishes real transitions from the absorbing zero outcome.
A learned 64-dimensional encoder maps these complete transition coordinates into
latent space. It is calibrated with full-transition reconstruction plus covariance
regularization, independently of reward, then frozen before successor fitting.
This prevents the Bellman geometry from drifting during critic learning. Neither
reconstruction nor regularization proves information preservation; heldout
reconstruction error is explicitly part of the gate. The encoder and reconstruction
decoder map the absorbing zero transition to zero by construction.

The critic G(s,a,noise) generates the **joint distribution** of encoded x_(t+T), where
P(T=k)=(1-gamma)*gamma^k. It does not predict an expected feature vector. Dependence
among future-state, action, and displacement coordinates is part of its target.

Its distributional Bellman equation is:

    with probability 1-gamma: emit the current real transition latent;
    otherwise: sample the successor critic at the next state and a policy action.

The implementation uses up to 32 real transitions before bootstrapping. Sampling
one geometric horizon implicitly gives the correct mixture weights. A reset or
rollout boundary shortens the real segment. A true terminal event bootstraps to an
absorbing zero latent; a time limit uses the physical final observation and a new
policy action. The current terminal transition itself remains a real target.

The generator is fitted with a proper joint energy score:

    0.5*||prediction1-target|| + 0.5*||prediction2-target||
      - 0.5*||prediction1-prediction2||

Predictions use independent noise. The last term prevents fitting a conditional
point forecast from being the population optimum. Squaring these distances would
destroy that argument. The target network is copied once per fitting rollout and
frozen throughout that rollout's updates. No reward or return enters this loss.

## Vector advantages and actor credit

A separate nonlinear decoder R(x) learns **immediate** reward from real transition
latents. Its gradients never train the successor critic. R(0)=0 is architectural.

For each action coordinate i, hold its observed action a_i fixed and draw all
other coordinates independently from the frozen policy. Compare its predicted
future latent to a second prediction with only a_i independently resampled. Both
predictions use the same other actions and generator noise:

    delta_Z_i = G(s, [a_i, b_-i], noise) - G(s, [b_i, b_-i], noise)
    A_i = (R(G_current_i) - R(G_reference_i)) / (1-gamma)
    policy gradient = sum_i score_i(a_i) * A_i

All coordinates and their counterfactuals are batched as [states, actions, features].
The latent advantage has shape [N,D,F]; the actor-facing advantage has shape [N,D].
The decoder evaluates each future outcome before any averaging. Distinct action
interventions therefore receive distinct credit; there is no shared scalar GAE
inside this estimator. The ultimate scalar reward objective still selects among
predicted outcomes, but it does not supervise or compress the critic's Bellman
representation. Final utility evaluation per action coordinate is unavoidable if
these advantages are to optimize the environment's reward.

If the conditional mean utility of the model is correct, the expected coordinate
score gradient is correct: its reference term is independent of observed a_i, and
the current term integrates over independent policy actions b_-i. The model can be
wrong, so this is an identity of the target estimator, not an unbiasedness claim for
the trained network. Finite counterfactual sampling can add variance; marginalizing
other actions is not an automatic empirical variance improvement.

A secondary diagnostic also computes pathwise policy credit: pull the decoder's
latent cotangent through the generative critic into actions, then through implicit
Beta sample derivatives into policy logits. It does not determine gate promotion.
Its detached-reference subtraction has no effect on that derivative and is not
claimed as a pathwise variance-reduction mechanism.

Shared noise defines a coupling inside the learned generator, not an identified
physical coupling of two alternative trajectories.

For the true successor distribution, E[R(G)]/(1-gamma) equals discounted action
value. R(E[G]) generally does not; applying nonlinear reward to a mean successor
would introduce a different objective. Here the decoder is applied per sample.

## What is implemented and what is not

The new file implements the predictive critic, immediate reward decoder, explicit
latent vector advantages, coordinate-score and pathwise gradient estimators, and a fixed-policy
experiment. It does **not** yet implement an actor-training algorithm. Actor
optimization is deliberately deferred until the derivative mechanism is tested.
The scalar value network copied from v2 is solely a frozen reference/control for
Monte Carlo score-gradient measurement, not an input to the new estimator.

This is a model-based estimator and can be biased. Distribution prediction error
does not bound action-derivative error. A direct bootstrapped costate critic was
considered but not implemented: discounting does not guarantee contraction after
multiplication by dynamics Jacobians. This distributional formulation avoids that
specific instability, but it still requires the empirical gradient audit.

## Fixed-policy protocol

- Actor: the checkpoint named in the previous agent's gate artifact; never updated.
- Seed: 1; HalfCheetah-v4; 16 environments; 2 environment threads.
- Calibration: 32 rollouts; reward normalization, reference value calibration, and reward-independent transition representation fitting.
- Fitting: 128 rollouts; normalizers and reference value frozen; train generator
  and immediate reward decoder.
- Holdout: 32 fresh rollouts; all parameters, target models, and normalizers frozen.
- Each rollout has 32,768 transitions; including phase warmup, 6,307,456 transitions.
- Model-gradient expectation uses four fresh action/noise samples per state.
- Thirty-two fixed actor-parameter projections are used only for heldout evaluation,
  never to train the generator. They are held fixed across holdout rollouts so the
  mean vectors can be compared.
- Reference gradients use lambda=1 Monte Carlo returns with a frozen value bootstrap
  at time limits and rollout tails. This reference is not an exact infinite-horizon
  oracle and its limitations remain in the report.

Predeclared screening criterion: reference mean-gradient signal/noise at least 3;
paired-rollout bootstrap upper95 relative model-gradient error below 0.5; lower95
gradient cosine above 0.5; immediate-reward normalized MSE below 0.05; transition-representation normalized reconstruction MSE below 0.1. An uncertain
reference is **inconclusive**, not evidence of either an improvement or a failure.
Confidence is conditional on this checkpoint and these projections, not RL seeds.
Passing permits a subsequent actor experiment; it does not establish that model
exploitation is absent or that unmeasured gradient directions are correct.

Diagnostics also report joint energy scores, dispersion, and aggregate latent outcome-difference
second-moment rank (not per-state action-Jacobian rank). A separate observational prediction check uses starting states
with at least 512 real transitions before reset and geometric horizons below 512.
Its omitted tail mass, gamma^512, is explicitly logged. The two-sample-mean comparison
is a diagnostic, not an independently trained baseline.

## Jobs and validation

- MLQ 7423: ten CUDA numerical contracts, parallel limit 1, time limit 20 minutes.
- MLQ 7424: fixed-policy mechanism experiment, depends on 7423 succeeding,
  parallel limit 1, time limit 90 minutes.
- Normal priority; no automatic retries or performance culling. The experiment is
  fixed-policy evidence collection, not a learning curve to cull by episodic return.
- TensorBoard metrics and gate.json are written under the standard runs directory.
- Python syntax compilation and all ten CUDA contracts passed. Both queued jobs completed successfully; process success is separate from mechanism-gate success.
- Independent review checked the distributional Bellman construction, energy score,
  Beta differentiation, target indexing, and parameter freezing. Review identified
  physical terminal observations and heldout predictive diagnostics as necessary;
  both are implemented.

Numerical contracts cover geometric discount mass; termination/truncation indexing;
complete transition features and final observations; learned-representation vector supervision, absorbing origin, and reward-gradient isolation; distribution versus mean energy scoring;
coordinate-counterfactual effects and expected Beta score gradients; Beta reparameterization against analytic derivatives; compiled generator gradients
and separation from reward updates; actor JVPs against score autograd; and gate
rejection of biased or uncertain evidence.

## Completed fixed-policy evidence

MLQ 7423 passed all ten numerical contracts. MLQ 7424 completed all 6,307,456
transitions, including 32 heldout rollouts. The actor remained unchanged and all
models were frozen throughout holdout. No actor-training benchmark was launched.

| Measurement | Result | Predeclared requirement |
|---|---:|---:|
| Representation normalized reconstruction MSE | 0.03658 | <0.1 |
| Immediate reward normalized MSE | 0.17716 | <0.05 |
| Reference gradient signal/noise | 1.15995 | >=3 |
| Relative projected mean-gradient error | 2.46168 | upper95 <0.5 |
| Upper95 relative gradient error | 2.50864 | <0.5 |
| Lower95 gradient cosine | -0.41826 | >0.5 |
| Model/reference mean-gradient norm ratio | 2.08030 | diagnostic |

The gate records `passed=false`, `status=inconclusive`. Low reference SNR prevents
reliable gradient validation; it does not excuse the independently inadequate
reward decoder. The large observed gradient discrepancy is a concern, but these
data do not establish the true gradient angle precisely. Aggregate latent outcome
difference rank is about 16.7; this is not a per-state action-Jacobian rank or
proof of correct credit assignment.

Interpretation: rich reconstructive latents alone did not yield validated useful
policy credit. Two problems need separating before another actor experiment:
reward decoding must be accurate without feeding scalar reward gradients into the
predictive representation, and the gradient reference must have sufficient signal.
Improved broad distribution fit or much lower estimator variance cannot substitute
for either requirement. These results do not isolate whether reward information
was lost in encoding or simply underfit by the decoder, and they do not establish
that the successor model learned the relevant conditional action effects.

Evidence: [gate.json](../runs/HalfCheetah-v4__successor_latent_v3_fixed_policy_gate__1__1789532757572439873/gate.json).
