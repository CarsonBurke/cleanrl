# Independent review of oracle-reliability algebra

Read-only analytical review of `analytical_oracle_reliability.md`. No model/test execution, new samples, jobs or causal claims. The stated identities are correct under the document's fixed selected states, fixed whitening, and independent identically distributed replica fields with a common finite covariance. Gaussian field noise is not required.

## Signed pair energy and its exact variance

Let x_r=mu+epsilon_r, R>=2, E[epsilon_r]=0 and Cov(epsilon_r)=C. Write P=R*(R-1)/2. Directly expanding the empirical mean and unbiased sample covariance gives:

    U = sum_(i<j)[x_i dot x_j]/P
      = norm(mean(x))^2 - trace(sample_covariance)/R

Thus E[U]=norm(mu)^2 and retaining its sign is essential. A finite-sample negative realization is not proof that the population signal is exactly zero; a positive realization is not proof of resolution.

The centered estimator splits into orthogonal first- and second-order terms:

    U-E[U] = (2/R)*sum_i[mu dot epsilon_i]
             + (1/P)*sum_(i<j)[epsilon_i dot epsilon_j]

The first term has variance4*mu^T*C*mu/R. Every distinct degenerate pair term is uncorrelated with the other pair terms, including pairs sharing one index: conditioning on the unshared mean-zero noise makes their covariance zero. Each pair term has second moment trace(C*C), so the second term has variance trace(C*C)/P. Their mutual covariance is zero. Therefore:

    Var(U) = 4*mu^T*C*mu/R + 2*trace(C*C)/(R*(R-1))

For R=2 this is2*mu^T*C*mu+trace(C*C), as stated. Finite second moments of each independent finite-dimensional replica suffice for this variance identity: the bilinear pair expression does not require a Gaussian assumption or a fourth moment of one replica. Individual displayed norm/covariance terms could have poorly behaved higher moments while their U-statistic cancellation still has finite variance.

Actual pair energies are not independent observations. For q_ij=x_i dot x_j, overlapping q_ij and q_ik have covariance mu^T*C*mu; disjoint pairs are independent under the stated assumptions. At mu=0 overlapping pair energies are uncorrelated but still need not be independent. Treating R*(R-1)/2 pairs as that many independent replications is therefore invalid.

## Fixed and selected directional credit

For a direction v fixed conditional on prior training/snapshot evidence and independent of evaluation replicas, y_r=x_r dot v has mean mu dot v and replica variance v^T*C*v. Its average has variance v^T*C*v/R. This is narrower than whole-field mean-squared noise trace(C)/R. It correctly allows resolved directional credit alongside unresolved field cosine/energy.

A saved update is random over training histories, but it can be conditioned on when independent oracle draws evaluate it. That conditional inference does not estimate variation across training seeds, checkpoint selection, or new initial-state populations. If whitening, selected states or objective are fitted using evaluation replicas too, their dependence must also be included; the simple fixed-v claim then needs additional conditioning or independent data.

For v fitted from one replica and an independent test replica x_test, E[x_test dot v | fit]=mu dot v and Var(x_test dot v | fit)=v^T*C*v. For the mean of R_test independent held-out replicas, divide this variance by R_test, not by the count of all fit/test arrangements. Reciprocal two-replica fits/tests share the fit/test data and cannot be treated as independent replications. Searching several directions and choosing the winner on held-out results uses that holdout for selection; its original conditional interpretation no longer supplies an unbiased selected-winner confirmation.

## Geometric implications

For fixed Fisher whitening, the oracle field and actual motion must use compatible inverse-root/root transforms and the same state weighting. Their pairing then has the intended policy-credit units. A retained-range Cauchy ratio is valid only when BOTH field and motion are projected into that same retained range. Discarded motion and field energies remain separate.

A mathematically valid Cauchy ratio involving a noisy realized field is not automatically efficiency relative to the unknown mean physical field. Its random numerator and field-norm denominator can remain unresolved. The signed U signal estimator is also not a drop-in positive denominator: clipping negative U or taking its square root conceals uncertainty. Fixed-direction pairings and their independent conditional uncertainty must precede such efficiency claims.

Small singular values amplify fitted parameter coefficients and uncertainty even if output interpolation appears accurate on selected states. Threshold stability, achieved joint Fisher cost and actual finite retracted response therefore remain independent gates. Stable credit for the historical first/endpoint chord does not certify a new inverse-Jacobian-fitted direction, its unvisited-state behavior or closed-loop gain.

## Precise qualifications suggested

No correction to the displayed energy/variance/directional formulas is needed. Two wording qualifications would make their interpretation sharper:

1. The exact variance formula is a population identity; substituting a covariance estimated from two or four replicas does not make an exact confidence interval. With non-Gaussian directional noise, small-R Student intervals are approximate, not distribution-free guarantees. Whole-field covariance estimates at tiny R cannot resolve all omitted eigendirections.
2. Numerical correctness of the retained-range Cauchy construction certifies an algebraic bound for the observed field. Reliability relative to the unknown mean field remains a separate statistical question; do not label a valid observed-field ratio as resolved physical optimizer efficiency.

The existing document's held-out-selection, finite-objective, sampling-law and physical-oracle caveats are substantively correct. None of this algebra identifies a benchmark bottleneck or authorizes causal training before the stated reliability gates.
