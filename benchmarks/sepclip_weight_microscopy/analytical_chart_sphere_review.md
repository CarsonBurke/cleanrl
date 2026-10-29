# Independent norm-eight chart witness review

Pure analytical/source review. No models, tests, jobs or original-source changes. Reviewed after the new saved-field chart-credit diagnostic received final physics source review.

## Verdict

The current explicit k=1 witness is mathematically valid and constraint feasible. It strengthens the former ambient straight-line witness to a norm-eight feature circle. No blocking proof defect remains. The conclusion is a changed readout-family pattern on a fixed feature domain, not a visited-state/whole-network capacity proof, strict function-class inclusion or benchmark advantage.

## Constant-concentration impossibility for the original head

On h(t)=8*(e1*cos(t)+e2*sin(t)), each ordinary gain-weight-bias head logit is affine in cos(t),sin(t). Exponentiating constant softplus sum gives exp(za)+exp(zb)+exp(za+zb)=exp(tau0)-1. The exponential Laurent expressions are holomorphic on the connected punctured plane; an open real-circle arc has accumulation points within that domain. The identity theorem therefore legitimately extends the constant equality to all nonzero complex z. Conjugated coefficients are constants, not conjugation of z, so holomorphicity is preserved. No complex logarithm branch was introduced.

Duplicate exponents wa,wb,wa+wb have identical reciprocal coefficients and strictly positive real combined coefficients. They cannot cancel. If any exponent is nonzero, choose a unit ray with positive projection onto it; a small perturbation preserves a positive maximum and avoids the finite tie directions for distinct exponents. Thus a unique nonzero maximizing exponent with a positive real-growth gap exists. Dividing by its exponential makes every other term tend to zero and its reciprocal factor tend to one. The limiting coefficient is positive/nonzero, so the modulus grows exponentially despite phase rotation. A constant function cannot do that. All nonconstant cosine/sine logit coefficients must therefore vanish.

This argument works for the larger unrestricted affine head family. Restrictions from unit-row heads cannot defeat an impossibility already proved for that superset. Constant total concentration forces both original logits/shapes/mean constant on the circle in exact arithmetic.

## Candidate construction and actual source

Two anchor rows span at most two dimensions in width64, leaving at least62 orthogonal dimensions. An orthonormal two-plane orthogonal to both rows is feasible. With the actual initial zero biases/unit gains, both frozen anchor logits are zero along that circle. This is restriction to a domain, not a stop-gradient claim.

A trained unit mode row along e1 with gain1/8 and bias0 gives cos(t). A unit precision row orthogonal to the circle with bias0 gives zero, with any finite gain (one suffices). The implementation requires independent unit row norms, not pairwise orthogonality after training. The construction therefore obeys the actual head constraints. The anchored decoder has r=cos(t), tau=2*log(2), fixed ell, and nonconstant mean at fixed total concentration.

The current k=1 refinement matters for literal source identification: all excess logs stay above-10, so the actual decoder takes x+log(-expm1(-x)), exactly inverse softplus in ideal real arithmetic. Its truncated small-excess polynomial branch is never used. An arbitrary finite-k exact-source claim would have omitted that approximation, but the explicit witness closes the issue. FP32 rounding can still perturb exact constant concentration; the note correctly keeps that separate.

## Scope and falsification

A norm-eight circle is a valid pointwise hidden-feature domain, but no evidence establishes that the actual trunk generates or visits this circle. Changing the trunk can implement different features and defeat a whole-network impossibility inference. Approximate finite-sample fit is weaker than exact continuum equality. The candidate is not proved to contain every original affine-logit function, since the anchored chart correction needed for an arbitrary target can be non-affine in features.

Independent physical-credit reliability, trustworthy actor reachability on actual states, numerical range/conditioning, finite optimizer effects, closed-loop gait and fresh matched benchmark controls remain necessary for causal selection. This proof identifies representational non-equivalence at the fixed readout, not a measured cause of sub10k performance.

## Reviewed hashes

- `benchmarks/sepclip_weight_microscopy/analytical_chart_sphere_witness.md`: `9d45b98bdf92925f9a2236fb2c8669b74ed8379355c63102dc199617d5820a2d`
- `cleanrl/ppo_continuous_action_sepclip_anchored_precision_chart_v13.py`: `35e9d4d704c4c5bb2a9a6706b84507d6d6f0e867fdb10bf43c5323f4d0cdd68e`
