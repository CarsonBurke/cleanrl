# Independent Beta geometry appendix

This is an analytical audit of `root_analytical_transport.md`, not a new measurement or an explanation established by benchmark evidence. No models, environments or ML jobs were executed. The relevant distinction is between output-chart conditioning, the reachable distribution family, intrinsic KL cost and the network's restricted functional tangent space.

## Exact conditional credit and the six-action qualification

At a fixed state, let U have independent Beta coordinates, with physical actions A=2U−1. Freeze the continuation policy and define Q(A) as the expected continuation return after the entire action vector. Provided the required log-weighted expectations are finite and differentiation under the integral is valid,

- d E[Q]/d alpha_i = Cov(Q(A), log U_i).
- d E[Q]/d beta_i = Cov(Q(A), log(1−U_i)).

These are vector-action identities. Equivalently define Q_i(u)=E[Q(2U−1) | U_i=u], integrating the other five coordinates against their CURRENT distributions. The same covariances use Q_i(U_i). Q_i generally changes when another coordinate's law changes. Treating six independently fitted scalar first-action oracles as a globally separable Q is therefore unjustified. Coupled actuators and subsequent contact feedback can make the conditional slopes depend strongly on the other actions.

The physical action derivative of a sufficiently smooth learned Q satisfies dQ_i/du=2 E[dQ/dA_i | U_i=u], under the corresponding interchange assumptions. The Beta covariance still integrates the whole current distribution; it is not generally the derivative at its mean. A state-only V critic provides no direct substitute for this action derivative.

For an exact, unclipped score at the original policy, a state-dependent baseline cancels in expectation. The covariance interpretation does not transfer verbatim to finite-data GAE, biased or truncated credit, or the sign-dependent PPO clipping mask. With clipping, both the sampling weights and which actions contribute change. A first-action conditional intervention also freezes subsequent policy changes; a full policy derivative needs the contributions at all visited times.

## Fisher metric, logits and head gains

For one coordinate write kappa=alpha+beta and t(x)=trigamma(x). In shape coordinates the exact Fisher matrix is

    F_shape = [[t(alpha)−t(kappa), −t(kappa)],
               [−t(kappa), t(beta)−t(kappa)]].

This follows by taking the covariance of the two log-action scores, or differentiating the Beta log normalizer. The trigamma recurrence and asymptotic expansion used below are given by [NIST DLMF §5.15](https://dlmf.nist.gov/5.15); the matrix and subsequent geometric conclusions are derived here.

For alpha=1+softplus(z_alpha), beta=1+softplus(z_beta), let D=diag(sigmoid(z_alpha),sigmoid(z_beta)). Then

    g_z = D*g_shape; F_z = D*F_shape*D.

At every finite logit D is invertible in exact arithmetic. A tiny beta-logit derivative scales BOTH its score gradient and its Fisher cost. It does not by itself prove that useful policy motion is intrinsically expensive. Finite precision, a damped inverse, bounded optimizer steps or elementwise Adam can prevent the formal inverse from cancelling this scaling.

For z_i=h_i*(w_i·feature+b_i), the trunk Jacobian has a factor h_i, while the gain derivative is w_i·feature+b_i. Negative gains reverse effective head directions. Changing a gain changes the actual logits, distribution and Fisher; it is not merely a coordinate change unless weights/biases are transformed inversely to preserve the entire policy. Even such a compensating transformation can change an approximate optimizer. Unit-row constraints limit weight rescaling, while a zero gain temporarily eliminates that head's trunk derivative. A gain can restore Euclidean derivative scale without certifying a better KL-efficient physical direction.

For a fixed state and independent action coordinates, output Fisher is block diagonal. The shared network pullback E[J_z,theta^T F_z J_z,theta] generally couples every head and layer and can be singular. Separate per-head inverses are not the inverse of this shared-network metric.

## The asymmetric endpoint limit

Set alpha=a tending to infinity, beta=b fixed and positive, with b near one for the shape-floor case. To leading order,

    F_shape = [[b/a² + O(a⁻³), −1/a + O(a⁻²)],
               [−1/a + O(a⁻²), t(b)−1/a + O(a⁻²)]].

Use coordinates x=log a and b. Their Fisher approaches the finite matrix

    F_(x,b) → [[b, −1], [−1, t(b)]].

Its determinant b*t(b)−1 is positive. At b=1 it equals pi²/6−1, approximately0.645; there is no singularity in this limit's properly scaled output chart. The original shape metric nevertheless has a small eigenvalue of order a⁻², and beta softplus close to its floor introduces additional logit conditioning. These are distinct coordinate effects.

The physical mean m=2a/(a+b)−1 has derivatives

    dm/da = 2b/(a+b)²; dm/db = −2a/(a+b)².

In (x,b) coordinates its leading gradient is (2/a)*(b,−1), which is (2/a) times the first column of the limiting Fisher. Consequently

    F_(x,b)^−1 grad m = (2/a, 0) + lower-order terms,
    (grad m)^T F^−1 grad m = 4b/a² + lower-order terms.

In shape coordinates the same natural direction has leading d alpha=2, d beta=0. For locally linear physical Q with a finite slope q'=dQ/dA near the endpoint, multiply that direction by q': d alpha≈2q', and

    g_Q^T F^−1 g_Q ≈ 4b*(q')²/a² ≈ (q')²*Var(A).

The equalities are leading asymptotics for fixed b and a smooth conditional Q, not exact finite-a identities. At an exactly attained beta=1 constraint, an unconstrained natural direction can request a forbidden negative beta change at lower order. The alpha-only direction remains feasible and has the same leading mean/KL efficiency; this does not license ignoring the one-sided constraint in a finite update. The softplus chart remains interior mathematically, although floating-point rounding can attain the floor. In particular, the vanishing beta-logit derivative does not obstruct endpoint-mean ascent through alpha. A bounded Euclidean alpha-logit step produces mean motion of order a⁻²; a locally KL-normalized shape step can change alpha by order a and produces mean motion of order a⁻¹. This explains why coordinate conditioning can matter while intrinsic geometry still imposes a limit.

For local KL budget epsilon, with KL≈0.5*delta^T F delta, the maximum first-order physical-mean displacement is approximately

    sqrt(2*epsilon)*sqrt((grad m)^T F^−1 grad m)
      ≈ 2*sqrt(2*epsilon*b)/a.

The mean endpoint gap is d=1−m≈2b/a, giving relative displacement approximately sqrt(2*epsilon/b). Thus the absolute displacement shrinks with the gap, while a fixed small KL can still close a fixed FRACTION of that gap. This concerns the sampled mean law; it is not a deterministic mode-policy limit or a gait score ceiling.

There is also an exact inequality independent of the endpoint asymptotic. Since grad m=Cov(A,score), projecting centered A onto the score span gives

    (grad m)^T F^−1 grad m ≤ Var(A).

The endpoint limit approaches equality because a*(1−U) tends to a Gamma(b,1) variable and the leading alpha score is proportional to its centered value. This argument uses the full invertible output Fisher; restricting to the network's reachable tangent directions can only reduce the optimal mean response at the same local policy cost.

## Shape floors impose a variance ceiling, not a variance floor

For native mean mu and total concentration kappa, alpha=kappa*mu and beta=kappa*(1−mu). Shapes at least one require

    kappa ≥ max(1/mu, 1/(1−mu)).

In physical mean coordinates m=2mu−1 this is kappa≥2/(1−|m|), while physical variance is (1−m²)/(kappa+1). Therefore

    Var(A) ≤ (1−m²)*(1−|m|)/(3−|m|).

At finite softplus logits the boundary inequality is strict. There is no positive variance floor: concentration can increase without bound at any interior mean. The restriction is that a near-bound mean cannot coexist with broad sampled noise; near the upper bound the maximum variance is approximately (1−m)². This is a genuine reachable-distribution restriction that a bijective chart change cannot remove.

It does not imply that optimal locomotion requires distributions outside this envelope. A deterministic high-torque gait might benefit from small noise. Nor does it establish that the KL-efficient direction of improved gait consists of moving the sampled mean further toward the bound: phase changes, other actuators and state feedback can dominate. Compare measured mean/mode feedback, supported physical covectors and concentration directions before selecting this mechanism.

## Smoothness, noise and the scope of natural-gradient invariance

At fixed state, if Q is sufficiently smooth with a controlled third-order remainder over the distribution's support, then

    E[Q(A)] ≈ Q(m) + 0.5*trace(H_Q(m)*Cov(A)).

Independent heads make the instantaneous covariance diagonal, so mixed Hessian entries do not appear in this SECOND-order expectation term. They still affect how its slopes change when several means move, and do not remove actuator coupling. Near an endpoint, Gamma-limit standardized skewness remains finite rather than becoming Gaussian; absolute noise width tending to zero can justify a smooth Taylor expansion, but a Gaussian-noise argument cannot.

A small variance alone does not justify this expansion across contact transitions, narrow unstable gait basins or slopes/curvature that grow with concentration. The asymptotic q' result requires a fixed or uniformly well-behaved conditional Q; a concentration-dependent continuation field or changing closed-loop policy violates that simple interpretation. The direct quadratic torque penalty is smooth, but the velocity contribution is a dynamical return, not an immediate quadratic function of torque.

An exact natural gradient is invariant under a smooth locally invertible coordinate transformation with a nonsingular Fisher: g_c=T^T*g_z, F_c=T^T*F_z*T, and T*F_c^−1*g_c=F_z^−1*g_z. This is first-order invariance of a distribution tangent, not invariance of finite additive parameter steps or a damped/diagonal optimizer. With a singular network Fisher, the invariant statement concerns the uniquely optimized reachable functional tangent or its quotient; an arbitrary Euclidean pseudoinverse need not transform like a tensor.

A nonlinear output chart implemented by new LINEAR heads can change the reachable state-dependent family. It need not be a global reparameterization of the old linear heads, even when initialization matches perfectly. Unit-row constraints, head-gain parametrization and normalization can further change which tangent fields are reachable. Those are architecture changes and must be assessed separately from output natural-gradient invariance.

## Consequences for the current investigation

The covariance and conditioning statements in the root review are correct when read as fixed-continuation partial derivatives and local chart statements. Their strongest necessary qualifications are vector-action marginalization, clipping/credit departures, shared-network Fisher coupling and the distinction between finite output coordinates and the limiting shape boundary.

No analytical argument here identifies the benchmark bottleneck. The useful measurement separation is: reliable physical covector; reachable joint policy direction; true joint Fisher cost; actual optimizer displacement; and finite closed-loop return. Endpoint shape conditioning, distribution-family noise restrictions and gain-induced transport can then be distinguished rather than combined into a claim that Beta alone prevents a strong gait.
