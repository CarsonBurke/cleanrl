# A valid feature-sphere witness for the anchored chart

Pure analytic verification, with no models, tests, jobs or source changes. The claim concerns two readout families on the SAME fixed norm-eight feature circle in exact real arithmetic. It is not a visited-state capacity proof, a strict-superset claim, an actual trained-policy result or a causal selection of v13.

## Original affine shape logits on a great circle

Take orthonormal feature directions e1,e2 and h(t)=8*(e1*cos(t)+e2*sin(t)). Every point has norm8. The original affine shape-logit head, including learned row gains and biases, restricts to:

    za(t) = C+A*cos(t)+B*sin(t)
    zb(t) = F+D*cos(t)+E*sin(t)

All six coefficients are finite real constants. Unit head-row constraints can restrict their attainable values, but cannot invalidate an impossibility proved for this larger unrestricted affine family. A zero gain simply gives constant logits; biases only change C,F.

Suppose total concentration is constant on any open interval of this circle. Since a=1+softplus(za) and b=1+softplus(zb), this means softplus(za)+softplus(zb)=tau0, a positive constant for finite logits. Exponentiating on that real interval avoids any complex-log branch issue:

    exp(za)+exp(zb)+exp(za+zb) = exp(tau0)-1

Set wa=(A-i*B)/2, wb=(D-i*E)/2 and z=exp(i*t). On the unit circle, za=C+wa*z+conjugate(wa)/z, with the analogous expression for zb. The left side continues to the holomorphic function on the connected punctured complex plane:

    H(z) = exp(C)*exp(wa*z+conjugate(wa)/z)
           + exp(F)*exp(wb*z+conjugate(wb)/z)
           + exp(C+F)*exp((wa+wb)*z+conjugate(wa+wb)/z)

Conjugated coefficients are FIXED constants; there is no conjugation of the complex variable. The identity theorem applies because the constant equality has accumulation points within this domain, even if originally known only on an open circle arc. Thus H would be constant on the entire punctured plane.

## A dominant exponential rules out every nonconstant affine case

Group any repeated exponents in the finite set W={wa,wb,wa+wb}. A repeated exponent also has the same conjugated reciprocal coefficient, so its terms combine with a strictly positive real coefficient, a sum of exp(C),exp(F),exp(C+F). No duplicate-term cancellation is possible.

If either wa or wb is nonzero, W contains a nonzero element w0. A ray direction u can be chosen with Re(w0*u)>0; for example u=conjugate(w0)/abs(w0). Perturb u slightly while preserving positive maximum real projection, avoiding the finitely many tie directions Re((w-w')*u)=0 between distinct exponents. There is then a UNIQUE maximizing exponent w_star with Re(w_star*u)=m>0 and a strictly positive gap to every other exponent.

Along z=R*u as R tends to positive infinity, factor out the dominating exponential:

    H(R*u) / exp(R*w_star*u)
      = c_star*exp(conjugate(w_star)/(R*u)) + terms tending to0
      -> c_star > 0

The reciprocal exponential tends to1, whereas the other terms decay relative to the dominant one by their strict real-projection gaps. Consequently abs(H(R*u)) grows like c_star*exp(R*m), contradicting constancy. Complex phase oscillation cannot cancel a unique dominant term; its coefficient has a nonzero limiting modulus.

Therefore wa=wb=0, equivalently A=B=D=E=0. BOTH original logits must be constant on the circle whenever their softplus sum is constant on an open arc. Their shapes and mean are then constant too. This result covers affine heads with arbitrary gains/biases and hence also the actual unit-row original head family.

## The anchored chart has a contrasting constrained-head construction

For a chosen action pair, choose e1,e2 perpendicular to the two frozen anchor-head rows. At width64 such an orthogonal two-plane exists. At the original zero-bias, unit-gain initialization, both anchor logits are then exactly0 throughout this feature circle. More generally any symmetric constant anchor suffices, but the zero anchor is a concrete feasible one. The frozen anchor still differentiates through current features; constancy here is a restriction to this particular circle, not a stop-gradient assumption.

For the trainable head, take the mode-residual slot's unit row along e1, gain1/8, and zero bias. Its trainable output is cos(t), choosing the explicit witness k=1. Take the precision-residual slot's unit row perpendicular to the circle, with zero bias; its output is0 there. These satisfy the trained unit-row constraint. The readout is not required to retain mutual row orthogonality after training. At this zero anchor, the chart becomes:

    r(t) = cos(t)
    ell(t) = log(2*log(2))
    tau = 2*log(2)
    a(t) = 1+tau*sigmoid(r(t))
    b(t) = 1+tau*(1-sigmoid(r(t)))

Total concentration2+tau is constant, while native mean [1+tau*sigmoid(cos(t))]/(2+tau) varies on the circle. Both excess shapes are at least tau*sigmoid(-1)=.3728319759167927, with log excess at least-.9866274275399418, strictly above-10. Both shapes are therefore strictly greater than1.

This bound also establishes source exactness in ideal real arithmetic for the actual v13 decoder. `inverse_softplus_from_log` uses a truncated small-excess series below log excess-10, so claiming its exact inverse identity for arbitrary finite k would be too strong. The explicit k=1 witness remains entirely in its large branch x+log(-expm1(-x)), the exact inverse softplus for positive x. No truncated-series approximation enters this construction. FP32 decoder/shape rounding remains a separate numerical contract and can break literal constant-concentration equality at rounding scale.

Inspected decoder source `cleanrl/ppo_continuous_action_sepclip_anchored_precision_chart_v13.py` SHA256: `35e9d4d704c4c5bb2a9a6706b84507d6d6f0e867fdb10bf43c5323f4d0cdd68e`. The generic affine impossibility proof above is unaffected by the decoder implementation qualification.

This provides a genuine norm-eight feature-sphere function that the chart can represent and the original affine softplus head cannot represent on the SAME feature circle. It strengthens the earlier ambient-line witness without assuming that a nonconstant straight line is a valid normalized feature path.

## Precisely limited conclusion

The construction changes only the readout comparison while holding the feature-circle domain fixed. It does not prove that the trained MuJoCo trunk visits or can generate that circle, that its actual initial random anchor shares this constancy on the visited trajectory, or that the mean/concentration pattern is useful for gait. Choosing another trunk can change the features and defeats a claim about whole-network impossibility. Approximate fit on finitely many points is also different from exact equality on a continuum.

The witness establishes an additional functional pattern, not set inclusion: representing arbitrary original affine-logit functions with the anchored chart generally requires non-affine residual chart corrections. It proves neither a strict global superset nor a benchmark advantage. Statistical oracle reliability, state-dependent actor reachability, numerical conditioning, finite updates and independent physical causal controls remain separate evidence gates.
