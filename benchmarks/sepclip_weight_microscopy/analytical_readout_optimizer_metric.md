# Effective affine geometry of a unit head, gain and bias

Pure analytic derivation plus existing gain-summary arithmetic; no models, tests, jobs or source changes. This supplements [analytical_readout_and_precision.md](analytical_readout_and_precision.md), which establishes fixed-feature local readout surjectivity at nonzero gain. The SGD calculation below is a hypothetical reference geometry, NOT the actual Adam trainer or evidence of optimizer inferiority, starvation or a20k bottleneck.

## Joint first-order SGD map

Fix hidden features h and consider one output logit z=rho*(w dot h+b), with norm(w)=1. For a differentiable scalar loss, let t be its upstream logit LOSS cotangent, u=E[t*h], v=E[t], and P=I-w*w^T. For equal learning rate eta, intrinsic sphere SGD plus ordinary bias/gain SGD gives:

    delta_w = -eta*rho*P*u
    delta_b = -eta*rho*v
    delta_rho = -eta*(w dot u+b*v)

For ordinary ambient SGD followed by row normalization, the same delta_w holds at first order. Define effective affine parameters a=rho*w and c=rho*b. Applying the product rule and retaining first order gives:

    delta_a = rho*delta_w+w*delta_rho
    delta_c = rho*delta_b+b*delta_rho
    [delta_a;delta_c] = -eta*M*[u;v]
    M = rho^2*diag(P,1) + [w;b]*[w;b]^T

In block form:

    M = [[rho^2*P+w*w^T, b*w],
         [b*w^T, rho^2+b^2]]

This M is the induced effective-affine gradient MOBILITY/preconditioner. Where nonsingular, the corresponding effective gradient metric is M inverse; calling M itself a metric without this convention would reverse the gradient interpretation. The loss's effective-affine covector is exactly[u;v] at fixed features. If all head gradients receive one positive actor clipping scalar, the same derivation merely replaces eta by eta times that scalar at this step.

## Eigenstructure and conditioning

Every effective weight direction perpendicular to w has eigenvalue rho^2. The remaining two-dimensional space is spanned by the radial weight direction w and bias coordinate. Its mobility matrix is:

    B = [[1,b],[b,rho^2+b^2]]
    determinant(B) = rho^2
    T = 1+rho^2+b^2
    lambda_plus/minus = (T +/- sqrt(T^2-4*rho^2))/2

For rho!=0, M is positive definite, B has positive eigenvalues, and their product is rho^2. The trace/discriminant expressions follow directly from the2-by2 matrix; no bias or optimizer statistics are needed to derive them. Large bias can strongly couple radial weight and bias and separate these eigenvalues. No actual trained bias value is inferred here.

At rho=1,b=0, M is the identity on the entire effective affine space. The joint constrained-weight/gain/bias SGD update therefore recovers free-affine SGD to first order at that initialization, even though each parameter chart route separately looks different. The same equality holds at rho=-1,b=0. At b=0 with other nonzero gains, radial mobility is1 while tangent and bias mobility are rho^2; later geometry is generally anisotropic. Finite retraction, finite changes in gain/bias, changing hidden features and coordinate-dependent learning rates create additional departures from this first-order reference.

At rho=0, tangent and ordinary bias routes vanish and M=[w;b]*[w;b]^T has rank at most1. This is the known singular chart case, not a conclusion about the measured positive gains.

## What existing gain ranges permit us to quantify

The historical representation JSON reports twelve actor-output gains at each selected checkpoint. Squaring only their recorded extrema gives the tangent-mobility scales in the hypothetical equal-rate SGD formula:

| v7 readout checkpoint | Recorded gain range | Derived rho^2 range |
| --- | --- | --- |
| half-rate4M | [2.02130,2.95008] | [4.08565,8.70297] |
| half-rate8M | [2.42589,3.67725] | [5.88495,13.52220] |
| full-rate4M | [1.88012,2.47479] | [3.53484,6.12458] |
| full-rate8M | [2.14272,3.04605] | [4.59126,9.27839] |

These ranges establish nonsingular gain coordinates in those summaries. They do not give the actual radial/bias eigenvalues, because b is not supplied by these summary rows. They do not give Adam's actual effective step, policy Fisher cost, moments, or physical credit quality. A scalar output gain also rescales hidden-network cotangents, but the upstream Beta loss cotangent changes with the policy; these numbers cannot be promoted into measured global gradient amplification.

## Joint head stationarity removes a head normal-force ambiguity

For rho!=0, all THREE raw head routes being stationary imply:

    rho*P*u=0
    rho*v=0
    w dot u+b*v=0

The bias equation gives v=0. The weight equation gives P*u=0, so u is parallel to w. The gain equation then gives w dot u=0, hence u=0. Thus joint stationarity is equivalent to zero effective-affine loss covector[u;v], and the ambient head-weight covector rho*u also vanishes. A nonzero purely radial head-weight force cannot remain while its bias and gain routes are simultaneously stationary in this nonsingular unregularized readout chart.

This is stronger than observing only a zero tangent head-weight covector. In that case u may still be radial, and the gain/bias equations can carry its effective credit. It also distinguishes the head from hidden gate/up/down rows that lack corresponding independent row-gain routes and contain nonhomogeneous SiTU/channel interactions. The head result cannot be copied to those body constraints.

The statement concerns raw mathematical gradients of the same scalar loss, not zero current optimizer displacement. Nonzero stored Adam moments can move a parameter after its current gradient vanishes; numerical floors or very small gradients can make a step appear zero without exact stationarity. Direct parameter regularization would add terms outside the fixed-feature affine loss identity. No such extra term is assumed in this derivation.

## Actual Adam breaks the simple equal-rate mobility

The trainer uses independent elementwise Adam histories/denominators for w,b,rho, then retracts the head row. Their realized first-order effective increments are still:

    delta_a = rho*P*delta_w_ambient+w*delta_rho
    delta_c = rho*delta_b+b*delta_rho

But these increments need not equal -eta*M*[u;v]. Moment lag can make them depend on earlier credit. Even a frozen diagonal ambient weight preconditioner does not generally commute with P; preconditioning before retraction can convert radial covector components into tangent movement. Treating that extrinsic operation as intrinsic equal-rate SGD would hide the ordering distinction. Bias/gain preconditioners differ too, and actor-wide clipping can change their history scales across steps.

Local free-affine capacity therefore does not guarantee that independent Adam routes execute a particular coordinated free-affine update. Conversely, a departure from M is not evidence of inferior physical optimization: it may provide useful adaptation, or harmful transport, depending on resolved credit, joint functional cost and finite feedback. The existing source/summary information supplies none of those causal conclusions for this readout identity.

Artifact used for the gain ranges: `representation_historical.json`, SHA256 `df18acf6ec60be52f3ab8a34488eae7d3330a4cc9c1014950ae4ca42546c75e2`. All equations concern fixed features and first-order ideal arithmetic; actual joint actor updates and benchmark scores remain separate evidence.
