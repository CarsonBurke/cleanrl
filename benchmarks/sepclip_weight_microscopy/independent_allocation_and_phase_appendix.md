# Independent allocation, torque and phase appendix

Read-only analytical continuation of `layer_historical_fp64_measurement_review.md`, `analytical_gate_credit.md` and the root transport note. The only new work is inspecting existing JSON/source and deriving identities. No ML or environments were executed. This appendix does not identify a benchmark bottleneck.

## What the accepted allocation establishes

The accepted layer audit concerns one next-rollout full-batch optimizer proposal at each saved4M/8M policy, evaluated on a deterministic512-state stored-training subset. Its mean/concentration split is a linear decomposition of output displacement in NATIVE mean mu=alpha/(alpha+beta) and TOTAL concentration kappa=alpha+beta. It is not a split in v13 mode/excess-precision coordinates, a complete training-trajectory audit or a physically validated intervention.

For local shape changes,

    dmu=((1−mu)*dalpha−mu*dbeta)/kappa; dkappa=dalpha+dbeta.
    mean part: (dalpha,dbeta)=(kappa*dmu,−kappa*dmu).
    concentration part: (dalpha,dbeta)=(mu*dkappa,(1−mu)*dkappa).

`scripts/audit_sepclip_layer_transport.py:127` maps these parts back through the softplus derivatives before computing the Fisher Gram and stored-action advantage pairing. This linear algebra is correct in the finite interior, subject to floating-point conditioning. The resulting output fields need not individually be realizable by the same constrained shared network, or by feasible finite steps at the shape boundary.

The reviewed six snapshots report mean-only quadratic cost99.55–100.80% of joint cost and concentration-only cost0.351–0.581%. These are NOT exclusive budget shares: mean and concentration are not Fisher-orthogonal. Their sum requires the signed cross term, which explains values above100%. For example the exact one-coordinate Fisher cross entry is

    F_mu,kappa = kappa*[mu*trigamma(alpha)−(1−mu)*trigamma(beta)].

It vanishes for symmetric shapes but generally does not vanish. The evidence rejects dominant concentration-only displacement cost in those proposals. It does not show that concentration has no physical credit, that every useful mean direction was followed, or that concentration was negligible throughout training.

The signed pairings in that artifact use stored actions and GAE on the512-state subset. Negative concentration pairings for the rollout-clock variants, and positive ones for v4half, concern that surrogate covector. They are not signs of true contact-noise effects. Rollout full8M has negative mean AND concentration subset pairing, so its total sign cannot be assigned to concentration alone.

## Small metric cost can still affect expected gait

At fixed mean, concentration changes all centered moments of the action law. With a fixed, sufficiently smooth vector-action conditional Q and controlled Taylor remainder,

    E[Q(A)] ≈ Q(m)+0.5*sum_i Q_ii(m)*V_i,
    V_i=(1−m_i²)/(kappa_i+1),
    d E[Q]/d kappa_i ≈ −0.5*Q_ii(m)*V_i/(kappa_i+1).

This is a physical-mean-preserving concentration derivative. Negative local curvature favors reduced variance; positive curvature can favor increased variance. An optimizer cost fraction does not measure that curvature or the derivative's sign. For contact-sensitive continuation returns, the exact covariance with the concentration score remains valid even when this Taylor expansion does not.

A useful bound prevents overstating the effect. For a fixed state and output tangent v, let S_v be its score. Then

    |Cov(Q,S_v)| ≤ sqrt(Var(Q)*E[S_v²]).

Thus small Fisher cost bounds first-order credit if conditional return variance is bounded. It does not make credit zero. A0.5% isolated cost has roughly7% of the score standard deviation of a unit-cost direction, before credit alignment and cross terms. This is neither a claim of a7% return contribution nor evidence of a huge hidden effect. Rare contact failures with large return consequences can give concentration useful covariance, but that possibility requires measured, independent conditional credit.

The all-time policy effect adds changed state occupancy and feedback. State-dependent variance near vulnerable stride phases can differ from the pooled average; temporal action correlation is yet another object. These heads sample conditionally independent actions, so neither the cost fraction nor a concentration adjustment establishes correlated exploration. No phase-conditioned concentration credit is present in the accepted allocation table.

## Near-bound mode is different from near-bound mean

The historical representation JSON evaluates256 stored visited observations, giving1536 action-coordinate entries per snapshot. For v7full8M it reports fractions of absolute action above.95: mode42.1875%, mean0%, one sampled draw14.2578%. At.98 they are mode34.4401%, mean0%, sampled4.94792%. V7half8M mode is48.0469% above.95 and40.3646% above.98; its mean is0% for both thresholds. These observations belong to the representation subset and its private sampling draw, not the512-state layer subset or a phase-conditioned physical evaluation.

For kappa>2 and interior shapes, the exact physical mode p and mean m satisfy

    p=(alpha−beta)/(kappa−2)=kappa*m/(kappa−2).

Consequently a mode close to a bound can coexist with a materially interior mean. At kappa=10, p=.975 gives m=.78, for example. Shapes at least one imply |m|≤1−2/kappa, while the mode can approach a bound as one excess shape approaches zero. This is a property of the particular trained distribution, not proof of a fixed torque ceiling. Larger learned concentration/gain or another reachable policy can change the mean; whether that change improves gait is unmeasured by threshold proportions.

Pooled absolute thresholds discard actuator sign, phase, duration and whether torque assists or opposes velocity. Strong locomotion need not maximize time at actuator bounds. The measured mode policy is a concrete high-torque counterfactual, but replacing sampling by mode also changes its state trajectory and feedback. Threshold fractions on the original stored observations do not describe its new occupancy.

Existing `physics_complete.json` gives v7full8M paired gains relative to sampling of999.781689 for mean and1176.860353 for mode: their mean difference is177.078664 return. This corresponds to the frozen evaluation protocol in that artifact, not a training improvement or a new experiment. It argues against simply replacing the learned policy by its mode supplying the remaining roughly10k gap. It does not exclude a different phase-dependent high-torque policy from being substantially better.

## Connection to the gate-credit chain

A mean-dominated output displacement says little about which gate channels carry useful physical credit. The gate covector contains downstream cotangent, full signed k'(g), capped-up value and the row-tangent input; feature normalization applies another projection. The accepted visited cap summaries do not reconstruct their joint, credit-weighted product. Concentration can alter downstream policy-score covariance without consuming much output Fisher cost, while activation or normalization can remove a useful mean direction despite a large mean allocation.

The falsifiable separation is therefore: phase-dependent physical mean and concentration credit; stable reachable joint action fields; their actual cross-inclusive Fisher costs; and finite closed-loop validation. Current pooled allocation, mode thresholds and cap statistics establish none of those causal links by themselves.
