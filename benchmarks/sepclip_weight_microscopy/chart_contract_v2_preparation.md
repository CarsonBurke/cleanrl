# Versioned chart contract preparation after 11770

Failed attempt 11770 is preserved: 26 passed, two failed. Original `tests/test_sepclip_anchored_precision_chart.py` SHA256 d5e8e3f8a95a3fb2e18c5ea4f855a7430d0b81141f1a4c066276fcb469ca5e95 and frozen v13 trainer SHA256 35e9d4d704c4c5bb2a9a6706b84507d6d6f0e867fdb10bf43c5323f4d0cdd68e are unchanged.

NEW `tests/test_sepclip_anchored_precision_chart_v2.py` prepared SHA256 5eda9a4ba29369f7c3e51e73e7dea2b035c23bcbcb502ff901bf168217fb9fc9; nine AST-counted test cases. No models, tests or jobs executed/submitted during preparation. Independent architecture review is requested. No candidate is selected. Future execution remains mlq-only with maxParallelRuns1 and immutable reviewed source pins.

The compilation failure is Inductor CantSplit involving dynamic batch symbol s33 in full input jacrev. The v2 tests request dynamic=False for all compiled paths, retaining fullgraph CUDA and the same full input Jacobian comparison. This is a prepared workaround, not proof the compilation issue is fixed. An additional independently compiled centered primal comparison contracts the full input Jacobian against a direction at three neighboring step sizes for original and chart actors; no JVP implementation supplies its finite-difference reference.

The second failure was FP64 alpha error4.32471836120385e-11, relative1.7393441750334208e-12 at expected shape about24.86. The frozen inverse helper's large formula is mathematically exact: for excess y>0, z=log(expm1(y))=y+log(1-exp(-y)). The approximation comes from the OUTER default F.softplus threshold20: when z>20 it returns z rather than log(1+exp(z)). Thus:

    resulting shape = 1+y - deficit
    deficit = -log(1-exp(-y))
    exp(-y) <= deficit <= exp(-y)/(1-exp(-y))
    d(shape)/d(log y) = y/(1-exp(-y))

The gradient surplus over ideal y is y*exp(-y)/(1-exp(-y)). The threshold occurs at y>20+log(1+exp(-20)), not a shape/concentration cap. Shape deficit and derivative surplus are explicitly included in new expectations, rather than hidden in looser tolerances. The standalone20.01-excess fixture requires the deficit to exceed the explicit ordinary FP64 budget by over100 times.

For the small inverse branch, y<exp(-10), the frozen formula is P=log(y)+y/2+y^2/24. Use sinh(y/2)/(y/2)=product_n(1+t_n), t_n=y^2/(4*pi^2*n^2). Because t-t^2/2 <= log(1+t) <= t, sum t_n=y^2/24 and sum t_n^2/2=y^4/2880:

    0 <= P-log(expm1(y)) <= y^4/2880

Differentiating the product with respect to log(y), the derivative excess is sum 2*t_n^2/(1+t_n), so it lies between zero and y^4/720. Softplus's derivative is at most1 and sigmoid is1/4-Lipschitz. Hence the inverse-induced shape error is at most y^4/2880 and the log-y shape-chain error is bounded by y^4/720 + [y/(1-exp(-y))]*y^4/(4*2880). These are exact-real approximation bounds; at the small-branch threshold the value bound is below1.5e-21.

The FP64 engineering numerical budget separately uses two stable reference/computation paths, each budgeted for128 primitive/rounding errors, unit roundoff u=2^-53 and gamma128=128*u/(1-128*u). Absolute allowance is2*gamma128*(1+scale), using full precision for the chart composition and the corresponding scalar scale for isolated inverse checks. This is an explicit conservative error model for modest-conditioned fixtures, not a proven libdevice error guarantee or arbitrary broad rtol covering threshold bias. Actual compiled primitive accuracy and reassociation remain subject to queued validation. Shape/chain approximation terms are kept separate from this budget.

Independent references use positive-expm1 inverse below excess1, stable log1p identity above1, exact-softplus logaddexp anchor shapes and multiplicative precision instead of the decoder's log-excess implementation. The chart fixture asserts mode-logit magnitudes below20, isolating outer threshold behavior from approximation in intermediate softplus terms. All original initialization/control/head-flow, host mirror, hidden PG, tiny inverse and saved evaluation contracts are retained. All prepared cases remain unexecuted and the original failure evidence remains intact.

Prepared coverage amendment: final ready-for-review test hash e6693ef5416d0b894fda7a344106e1cddbac46997a4ceda2815a8bc342997949, retaining initial5eda hash above as history; count remains nine. Wide-chart derivatives now independently check both alpha and beta channel signs/factors, using a fresh compiled forward for beta instead of retaining a donated backward graph. For beta, r derivative is -q*K_beta and ell derivative is K_beta, with the same separately derived approximation and roundoff budgets. No numerical execution is claimed.

Final independent source review approved e6693ef5416d0b894fda7a344106e1cddbac46997a4ceda2815a8bc342997949, recorded in `chart_contract_v2_architecture_review.md`. Reviewer verified approximation signs/factors/bounds, fresh beta backward, full Jacobian contraction and unchanged original test/trainer hashes. The gamma128 allowance remains explicitly conditional engineering error budgeting; dynamic=False and the added numerical contracts remain unexecuted. No source/model change or chart selection follows from this preparation approval.
