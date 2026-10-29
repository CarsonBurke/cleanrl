# Conditional periodic-gait response model

This appendix derives a local response model. It does not establish that the saved policy occupies a particular basin, follows an exact periodic orbit, or has measured Floquet multipliers. No training, simulation, TD7 rerun, or new numerical measurement was performed.

## Exact discrete closed-loop derivative, when differentiable

Let x contain the physical integration state needed to determine the next environment transition. Let a be the torque command, and let f(x, a) denote one full environment step, including its internal physics substeps. Use the recorded frozen observation-normalization row and observation map o(x).

For a deterministic mean or mode policy, write a = a_theta(x), and evaluate derivatives along its trajectory:

    B_t = partial f / partial a
    A_t = partial f / partial x + B_t * partial a_theta / partial x
    H_t = partial a_theta / partial theta
    delta_x_(t+1) = A_t * delta_x_t + B_t * H_t * delta_theta

These are chain-rule identities on a differentiable transition branch. They include feedback: a changed state changes later torques. A policy Jacobian evaluated only on the original initial state cannot describe that propagation.

MuJoCo contact changes can alter the active dynamical branch. Differentiating only an unchanged-contact vector field can omit the movement of contact times. A continuous-time hybrid approximation needs the appropriate event-time correction, often expressed as a saltation map. A discrete finite-difference derivative of the complete transition instead includes the simulator's actual transition behavior where that derivative exists. At nonsmooth boundaries, report directional or one-sided responses; a single linear Jacobian need not exist.

The control command is held across physics substeps. Consequently the continuous-time autonomous-flow picture and the environment-step feedback map are distinct models. The equations above use the latter.

## Mean and mode have different physical Jacobians

For bounds [-1, 1], let c = alpha + beta, d = alpha - beta. The deterministic physical actions are:

    a_mean = d / c
    a_mode = d / (c - 2), provided alpha > 1 and beta > 1

Their differentials are:

    delta_a_mean = (delta_alpha - delta_beta) / c
                   - d * (delta_alpha + delta_beta) / c^2

    delta_a_mode = (delta_alpha - delta_beta) / (c - 2)
                   - d * (delta_alpha + delta_beta) / (c - 2)^2

The mode formula is undefined for an exactly uniform Beta. FP32 shapes can equal one even when their mathematical parameterization is strictly greater than one; any implementation fallback must therefore be included in the measured derivative law.

In the original head, delta_alpha = sigmoid(z_alpha) * delta_z_alpha, and similarly for beta. The actor supplies delta_z = J_actor * delta_theta. Its normalized branch derivative is:

    derivative normalize(b) = (I - y*y_transpose) / norm(b), y = b / norm(b)

Away from observation-clipping boundaries, frozen observation normalization contributes its diagonal inverse-standard-deviation map. Clipped coordinates contribute zero local derivative. These maps, SiTU gate derivatives, the readout rows, and learned gains all enter H_t and the state-feedback derivative.

A parameter step with a small Frobenius norm can therefore change action sensitivity substantially on particular gait phases. Conversely, a large matrix change can be mostly invisible to the actuator.

## Adjoint sensitivity separates force from credit

For a finite objective J = sum_t gamma^t * r(x_t, a_t) + gamma^N * Phi(x_N), define the derivatives of the composed per-step reward:

    L_x,t = gamma^t * [r_x,t + H_x,t_transpose * r_a,t]
    L_theta,t = gamma^t * H_t_transpose * r_a,t
    H_x,t = partial a_theta / partial x

Backward adjoints satisfy:

    lambda_N = gamma^N * gradient Phi(x_N)
    lambda_t = L_x,t + A_t_transpose * lambda_(t+1)

The first-order return change is:

    delta_J = lambda_0_transpose * delta_x_0
              + sum_t delta_theta_transpose * H_t_transpose
                  * [gamma^t * r_a,t + B_t_transpose * lambda_(t+1)]

This is an exact deterministic derivative on the assumed smooth trajectory. The bracket is the physical value of changing the current action, including its subsequent closed-loop consequences.

HalfCheetah's step velocity reward depends on the displacement produced by f. Its immediate action derivative therefore includes that displacement response, as well as the direct quadratic torque penalty. Replacing the bracket with only a torque-cost gradient would discard the main performance mechanism.

The policy-gradient score is a different object. It estimates expected return sensitivity through stochastic action probabilities. Mean/mode adjoints diagnose those deterministic laws; they do not validate the sampled Beta policy gradient. For a sampled-feedback linearization, write a_theta(x, epsilon) conditional on exogenous noise, including its actual sampling law. Expected sampled return still needs a stochastic derivative or the matched full-trajectory score/paired-return diagnostics.

## Periodic response and phase, conditionally

Suppose the deterministic trajectory forms a relative periodic gait over N environment steps. Horizontal position may translate by a distance D while the internal state repeats. Remove that translation coordinate before interpreting internal stability: the policy does not observe global horizontal position, so translation symmetry can create a neutral direction unrelated to a gait defect.

On the reduced internal state, define:

    M = A_(N-1) * ... * A_0
    K = sum_t [A_(N-1) * ... * A_(t+1)] * B_t * H_t

An empty product is the identity. One-period variations obey:

    delta_x_N = M * delta_x_0 + K * delta_theta

For a fixed-period reduced orbit with no remaining neutral directions, the perturbed repeating state solves:

    (I - M) * delta_x_0 = K * delta_theta

Its solution describes a shifted local orbit, not the transient from an arbitrary initial condition. Near-singularity of I - M can amplify a policy perturbation. Non-normal products can also create substantial transient amplification even when all asymptotic multipliers lie inside the unit circle.

The eigenvalues of M are local discrete-orbit multipliers. A continuous-time autonomous periodic orbit additionally has a neutral phase direction when event corrections are included correctly. A sampled control map does not automatically have an exact unit phase multiplier: control-clock locking and the definition of the return section matter. A Poincare section can remove continuous phase freedom when that model applies. Neither a unit multiplier nor instability should be inferred from an unqualified full-state spectrum.

If the gait period T and translation D can vary smoothly, mean speed is v = D/T:

    delta_v = delta_D / T - D * delta_T / T^2

A faster gait could arise from greater displacement per cycle, a shorter cycle, or both. Whether a particular normalized weight change increases either quantity depends on phase-resolved forcing B_t * H_t * delta_theta and its adjoint pairing. It does not follow from increased torque magnitude, a matrix singular value, or a lower policy-loss value alone.

## Connection to ambient polar optimization

For a unit row or column W, an ambient optimizer direction U becomes the actual retracted chord:

    delta_W = normalize_axis(W - eta*U) - W

Its first-order tangent approximation removes the radial component of U. Polarization, tangent projection, lane adaptation, and transport through the actor Jacobian generally do not commute. The resulting delta_theta first enters H_t, then B_t, and is propagated by products of A_t. The physically useful metric is therefore phase- and trajectory-dependent, beyond either the ambient matrix spectrum or the local Beta Fisher.

The independent 2-by-2 radial counterexample demonstrates that ambient polarization can create tangential motion from a gradient with zero constrained gradient. It does not establish that this is the main observed limitation. The testable question is whether the actual current optimizer rotates weights into phase responses supported by reliable physical credit, or into locally cheap responses that fail after closed-loop propagation.

## Conditional explanations and unavailable measurements

The reported approximately 10.58 m/s deterministic mode policy identifies a measured behavior under its evaluation protocol. It does not locate a stable periodic orbit or prove a basin boundary. A substantially faster policy might use a different stride, contact sequence, period, or feedback response; those possibilities remain hypotheses.

Three mechanically distinct failures are possible:

- Useful action sensitivity exists, but ambient spectral remapping and retraction direct the update away from its adjoint-supported phases.
- The local update has useful first-action sensitivity, but later state feedback damps or reverses its physical benefit.
- A faster response requires a substantial contact/period change, so the current local linear model cannot predict the transition reliably.

None is selected by the derivation. We currently lack phase-resolved complete-transition derivatives, validated reduced periodic trajectories, monodromy matrices, event corrections, orbit adjoints, and matched measurements of displacement/period changes for optimizer chords.

Existing matched physical-state, actual-chord, and full closed-loop return diagnostics can discriminate useful directions without first estimating Floquet multipliers. Their results must retain normalization, objective/tail, sampler, finite-retraction, contact-nonsmoothness, and conditional-noise qualifications. A discrepancy between a local prediction and a finite return change combines nonlinear response, feedback, and Monte Carlo error; it is not automatically evidence of critic bias or a particular optimizer defect.
