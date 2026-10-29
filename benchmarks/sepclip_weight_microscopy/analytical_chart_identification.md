# Identification of the anchored Beta chart

Analytical source review only. No models, tests, queue submissions or trainer changes. Reviewed v13 SHA256: `35e9d4d704c4c5bb2a9a6706b84507d6d6f0e867fdb10bf43c5323f4d0cdd68e`. The proposal remains unselected. This analysis concerns output coordinates, readout families and causal identification; radial/polar transport is analyzed separately in `root_analytical_transport.md`.

## The actual coordinates

For one independent action dimension, take the actual stored FP32 shape values a and b, promoted unchanged to FP64 for diagnostic algebra. On the interior a>1 and b>1, define:

    x = a-1; y = b-1; tau = x+y
    q = x/tau; r = log(x/y); ell = log(tau)
    a = 1+tau*q; b = 1+tau*(1-q)
    v = q*(1-q)

The native Beta mode is q; physical mode is 2*q-1. These are NOT the native mean mu=a/(a+b) and total concentration kappa=a+b. Their relationships are:

    kappa = 2+tau
    mu = (1+tau*q)/(2+tau)
    physical_mean = tau*(2*q-1)/(2+tau)
    physical_variance = (1-physical_mean^2)/(kappa+1)

The exact smooth interior shape Jacobian L = d(a,b)/d(r,ell) is:

    L = [[tau*v, x], [-tau*v, y]]
    inverse L = [[1/x, -1/y], [1/tau, 1/tau]]

Its determinant is tau^2*v, positive in the interior. At a rounded shape equal to one the actual-shape mode chart is on a boundary; when both shapes equal one it is undefined and the Beta is uniform with nonunique mode. Such a policy is a valid Beta distribution. It must be reported, not assigned an invented interior mode or silently replaced with intended latent shapes.

FP32 qualification matters: v13 forms the anchor chart from `softplus(z)` BEFORE adding one, not from subtracting one from the rounded shape. Let latent x_tilde=softplus(z_a), y_tilde=softplus(z_b). Those latent excesses can be positive when actual FP32 a-1 or b-1 is zero, and generally differ by addition/subtraction rounding. The formulas above are exact diagnostic geometry of the recorded interior shapes; they are not a claim of bit-exact equivalence to every FP32 decoder intermediate. The numerical contract must quantify actual-shape versus latent-chart discrepancies. Differentiating a floating-point map literally as a real function is also different from the smooth operation derivatives used by autograd.

## Covectors and Fisher

Let g_shape=(g_a,g_b) be a physical-PG covector with respect to shapes, for the specified conditional or full-trajectory objective. Then:

    g_r = tau*v*(g_a-g_b)
    g_ell = x*g_a+y*g_b
    g_chart = transpose(L)*g_shape

For an existing logit covector g_z, and original logits z with shape derivatives s_a=sigmoid(z_a), s_b=sigmoid(z_b), the original smooth-autograd chain gives g_shape=(g_z_a/s_a,g_z_b/s_b). This division becomes numerically ill-conditioned near very negative logits; retain precision and separately report the actual-shape boundary. If reconstructing exact ideal logits from recorded excess shapes instead, use z_star=inverse_softplus(x,y) and s_star=(1-exp(-x),1-exp(-y)); s_star need not equal the recorded FP32 sigmoid(z). Do not mix these two interpretations invisibly.

At the recorded shape values, the Beta shape Fisher is:

    A = trigamma(a)-trigamma(a+b)
    B = trigamma(b)-trigamma(a+b)
    C = -trigamma(a+b)
    F_shape = [[A,C],[C,B]]
    F_chart = transpose(L)*F_shape*L

In explicit entries:

    F_rr = (tau*v)^2*(A+B-2*C)
    F_r_ell = tau*v*[x*(A-C)+y*(C-B)]
    F_ell_ell = x^2*A+2*x*y*C+y^2*B

Keep F_r_ell and signs. Independent coordinate magnitudes are not a joint natural direction. An isolated chart-coordinate unit-Fisher slope is g_i/sqrt(F_ii); the joint dual squared is transpose(g_chart)*inverse(F_chart)*g_chart. Across saved states, aggregate the SAME objective/state weighting before comparing actor-realizable directions and include the actor Jacobian.

The ideal inverse-chart-to-logit Jacobian is T=diag(1/s_star)*L. For any invertible exact coordinate change, g_chart=transpose(T)*g_z and F_chart=transpose(T)*F_z*T, so T*inverse(F_chart)*g_chart=inverse(F_z)*g_z. Reparameterization alone cannot improve an exact natural-gradient first-order direction. Adam, clipping, moment history, finite retraction, numerical arithmetic and changed readout families break that invariance and must be identified separately.

## Transform the oracle's mean/concentration field

The existing oracle uses native mean mu and TOTAL concentration kappa. The derivative from chart coordinates is:

    dmu/dr = tau*v/kappa
    dmu/dell = tau*(2*q-1)/kappa^2
    dkappa/dr = 0
    dkappa/dell = tau
    D = [[tau*v/kappa, tau*(2*q-1)/kappa^2], [0,tau]]

Therefore g_chart=transpose(D)*g_mean_kappa and F_chart=transpose(D)*F_mean_kappa*D. This is the transform needed before selecting v13 from the existing oracle. At fixed r, changing excess precision ALSO changes mean except at q=.5. At fixed ell, changing r changes mean and variance at constant total concentration. Neither coordinate is a globally pure noise knob or pure torque knob.

Counterexample: a=3,b=2 gives tau=3,q=2/3,kappa=5,mu=.6. Then dmu/dr=2/15, dmu/dell=.04, dkappa/dell=3. If the native mean covector is one and total-concentration covector zero, g_ell=.04 is nonzero. Calling the original concentration covector zero does not establish that the candidate precision coordinate is useless. Conversely choosing g_mu=1,g_kappa=-.04/3 yields g_ell=0 despite nonzero mean and concentration covectors. Large coordinate magnitudes alone cannot select the chart.

## Anchoring preserves one derivative, not every gradient

Write the normalized hidden feature as f(s), including the factor eight. Let:

    a0(f) = diag(gain0)*(W0*f+b0)       fixed readout on CURRENT features
    t(f) = diag(gain)*(W*f+b)          trainable readout
    d(f) = t(f)-a0(f)
    c(f) = C(a0(f))+d(f)
    z(f) = inverse_C(c(f))

C maps original logits to mode-logit/excess-log-concentration; the first half of d shifts r and the second shifts ell. The anchor buffers freeze only the initial readout, not hidden features. At initialization t=a0. In exact smooth algebra inverse_C(C(a0))=a0, giving the SAME initial policy and hidden-feature derivative. Specifically, d(t-a0)/df=0 initially, so dz/df=d a0/df.

For a TRAINABLE head/gain/bias perturbation, the anchor is fixed. Initially dz=T*d t rather than dz=d t. Thus the original hidden PG is initially preserved for identical output covectors, but head and gain PG are transformed. Unit-row constraints apply to the new trainable head, and its tangent gradient is projected AFTER that transformed covector. Identical parameter count does not give identical gradient scales, Fisher blocks, optimizer moments, clipping budget or initial tangent function space over multiple states: T(s) is state dependent. Even when the raw hidden gradient initially agrees, the shared actor gradient-norm clip can rescale it differently because the transformed head/gain gradients change the aggregate actor norm. Thus identical raw hidden PG does not prove identical first hidden optimizer motion.

After the head changes, dt/df-d a0/df need not vanish. Hidden gradients then include both the changing anchored C(a0(f)) derivative and changing delta-readout derivative through the nonlinear inverse chart. Same initial hidden Jacobian is therefore an initialization identity, not a lasting architectural equivalence. FP32 inverse approximation and arithmetic add a separately measurable identity error.

## Readout family changes: a concrete witness

With a fixed feature line f(u), consider a constant symmetric anchor, constant ell and a learned r=k*u. The candidate can express constant total concentration with nonconstant mean:

    a(u)=1+tau*sigmoid(k*u)
    b(u)=1+tau*(1-sigmoid(k*u))
    a(u)+b(u)=2+tau

An original affine-logit readout on the same feature line produces a+b=2+softplus(A*u+B)+softplus(C*u+D). Its second derivative is positive whenever either slope is nonzero. It cannot keep concentration exactly constant while varying mean over an interval. This establishes a changed functional readout family on its ambient feature domain, not merely a renamed optimizer coordinate. The actual hidden features lie on a norm-eight sphere; a nonconstant straight feature line is not itself a valid visited-state trajectory on that sphere. This witness therefore does not prove an empirical approximation gap on the visited feature manifold. That stronger assertion needs a matched feature-state fit or a separate valid-manifold witness.

Do not claim a strict superset of all original functions. To reproduce an arbitrary original affine target z_target(f) under a fixed anchor generally requires delta(f)=C(z_target(f))-C(a0(f)), which need not be affine in f. Equal trainable count and initial policy do not prove set inclusion. The candidate adds useful nonlinear patterns in some directions while constraining others; later-family expansion refers to these additional patterns, not a proved global containment.

## Which contrasts identify what

| Contrast | Identifiable result | Unresolved alternative |
| --- | --- | --- |
| Original head versus anchored chart, matched optimizer/seed/protocol | Net chart-family intervention effect | Conditioning, changed head tangent space, later expressivity and finite arithmetic coexist |
| Adam and Anvil each within original and chart families | Optimizer-by-chart interaction | An interaction alone does not identify a particular Fisher or radial mechanism |
| Original family with a pullback output preconditioner and original forward unchanged | Coordinate-dependent optimizer conditioning at fixed function family | Approximate/state-aggregated pullback, moments and finite steps still need validation |
| Fixed original hidden trunk with fitted original versus chart readouts on independent physical states | Readout approximation difference for that fixed trunk | Does not certify closed-loop learnability or fresh benchmark improvement |
| Exact full natural-gradient chart versus original at the same function and Fisher cost | Ideal first-order coordinate invariance | Numerical approximations and changed parameter tangent range can invalidate premise |
| Same initialization then measured head/hidden JVP, moments, clipping and finite KL | Which optimization link first diverges | First divergence alone is not causal benchmark attribution |

These are identification designs, not new submissions or selected training plans. A hidden-trunk freeze is a diagnostic function-family comparison, not authorized benchmark finetuning.

## Falsifiable evidence gates

1. Transform a resolved physical covector/Fisher into r,ell using the actual recorded interior shapes; retain latent-chart and rounding discrepancies. Native mu/kappa evidence cannot be relabeled as candidate mode/precision evidence.
2. Show poor original optimizer response to a reliable realizable direction at joint Fisher cost, not just larger chart gradients. Require cross-noise agreement, numerical-cutoff stability and finite-step fidelity.
3. Validate identical initial actual shapes/likelihood, host/CUDA actor, first hidden JVP and intentionally changed head/gain JVP; quantify rounding boundaries, sampler atoms and inverse-chart range failures.
4. Identify whether benefit occurs within the original forward family under a principled pullback preconditioner, or requires candidate-only responses such as fixed-concentration mean variation. Without that contrast, call a benchmark gain a chart-family result rather than proof of conditioning alone.
5. In a fresh selected causal experiment, report actual first/ten-epoch policy displacements, head versus hidden gradients, clipping masks, moment dynamics and closed-loop gait effects alongside benchmark score. Physical uncertainty and literature protocol differences remain explicit.

No present derivation establishes that v13 will improve score, that representation is the dominant limitation, or that 20k is attainable with this intervention.
