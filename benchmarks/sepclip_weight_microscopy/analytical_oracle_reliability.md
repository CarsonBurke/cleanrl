# Reliability of physical credit is direction dependent

Analytical extension of existing replica gates. No new samples, tests, score estimates or causal claims. Assume the R independently redrawn conditional first-action/continuation fields share the same selected states, sampling law, units, horizon, and fixed Fisher whitening. Write x_r=mu+epsilon_r, E[epsilon_r]=0, covariance C. This idealization does not repair the sampler-law approximation, terminal uncertainty or selected-state bias.

## Signed energy is unbiased, not automatically resolved

For independent replicas, E[x_i dot x_j]=norm(mu)^2. The average over distinct pairs equals:

    U = norm(mean(x))^2 - trace(sample_covariance)/R

where sample_covariance uses denominator R-1. U is a signed unbiased squared-signal estimator. Negative U is compatible with nonzero signal when noise is large; clipping U to zero introduces bias and conceals resolution failure. Positive U is also compatible with large uncertainty.

The exact variance under independent identical replicas with finite covariance is:

    Var(U) = 4*mu^T*C*mu/R + 2*trace(C*C)/(R*(R-1))

This follows by writing each pair as norm(mu)^2 + mu dot(epsilon_i+epsilon_j) + epsilon_i dot epsilon_j. Distinct pairs sharing an index share the linear-noise term. The degenerate pair-noise terms are uncorrelated but do not turn the entire pair list into independent observations. For R=2, Var(U)=2*mu^T*C*mu+trace(C*C). A single positive pair cannot estimate either variance term reliably. The population variance identity does not make a tiny-R plug-in variance or approximate Student interval exact; near-normal scalar replica behavior remains a separate assumption.

The mean field itself has mean-squared noise trace(C)/R. Weak-direction noise can therefore dominate a high-dimensional cosine or fitted inverse-Jacobian norm even while a particular functional direction has stable credit. Whitening matters: changing a noisy tiny Fisher eigenvalue can change these energies substantially, so numerical geometry and statistical reliability are coupled.

## A fixed motion asks a narrower question

For a motion v chosen independently of these evaluation replicas, y_r=x_r dot v has:

    E[y_r]=mu dot v
    Var(mean(y))=v^T*C*v/R

This is the uncertainty that the existing actual-gain scalar intervals estimate conditionally on the selected states and fixed actual chord. It need not scale with trace(C). A stable positive y does not establish reliable coordinates orthogonal to v or a trustworthy fitted optimum, and a noisy field cosine does not by itself invalidate every actual-update directional measurement.

The actual saved update is selected using original training data, but independent redrawn oracle actions/noise can treat that chord as fixed conditional on the saved evidence. Reusing those same evaluation replicas to choose v changes the problem. If v is a fitted direction from x_i, the large self-pairing x_i dot v is selected to look good. The held-out x_j dot v is the relevant independent-noise check. Its expectation conditional on the fit is mu dot v, subject to the same physical-law and objective qualifications.

With only two replicas, fitting each and scoring the other supplies two coupled fit/test contrasts. Their two entries are not independent experiments or training seeds. They are useful direction-level diagnostics, not a confidence guarantee for a selected architecture. Searching many cutoffs, layers, horizons and charts on the same two replicas introduces additional selection; require a declared criterion and fresh targeted conditional information if a promising conclusion is near its noise margin.

## Why inverse geometry can make apparent efficiency unstable

A fitted actor response uses the retained singular coordinates of a whitened functional Jacobian. Dividing field components by small singular values can produce a large parameter change from modest score-field noise. The local fit may interpolate the fixed selected states while requiring a finite retraction that destroys the intended response elsewhere or along the actual trajectory.

Consequently independently useful field components, threshold-stable realizability, bounded fitted parameter displacement, actual joint Fisher cost, and finite-response fidelity are separate gates. Report retained-field and retained-motion Cauchy ratios only with numerator and denominator in the same range, keeping discarded energy explicit. A stable scalar pairing for the historical chord does not certify a newly fitted small-mode intervention. A correct Cauchy bound for the observed noisy field is an algebraic bound on that field, not resolved efficiency relative to its unknown physical mean.

## Incoming-result rule

First assess the fixed actual first/endpoint scalar pairings with replica signs/intervals and finite closed-loop effects. Next assess whether a proposed new direction survives held-out noise and numerical cutoffs. Use whole-field cross-energy/cosine to disclose unresolved field structure, without interpreting either sign as a universal certificate. If the candidate relies on unresolved coordinates, acquire targeted conditional information rather than rank a fitted efficiency or select fresh causal training prematurely.
