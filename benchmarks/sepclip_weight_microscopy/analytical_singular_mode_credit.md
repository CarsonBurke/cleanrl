# Singular modes of credit, history and realized motion

Analytical inspection only. No model/test/GPU execution, new job, or existing trainer/script/report modification. This note uses the parent's `ppo_epoch_credit_appendix.md`, `root_analytical_transport.md`, the current layer-audit source and stored `layer_historical.json` scalar entries. It does not establish physical PG or select a spectral intervention.

## What the audit decomposes

Let G be the CURRENT full-rollout clipped negative PPO-loss gradient for one bank. Let L=(1-mu)*G+mu*[w*fast+(1-w)*slow] be the lookahead supplied to the spectral map. The audit computes L=U*diag(sigma)*V^T, not an SVD of G. For lookahead mode Ei=ui*vi^T it records:

    singular_value_i = sigma_i
    clipped_gradient_component_i = inner(Ei,G)
    actual_proposal_component_i = di = inner(Ei,D)

D is old weight minus matrix-only updated weight, so the optimizer subtracts D. Its mode counterfactual is NOT simply the finite chord -di*Ei. It normalizes that single proposal on the bank's constrained rows/columns:

    Delta_i = retract(W-di*Ei)-W
    ui_logits = J_actor_bank * Delta_i

The reported subset gain is average_selected[Ahat*score_old dot ui_logits]. It uses512 deterministic selected stored states/actions and advantages. The modal Fisher entry is .5*average_selected[ui_logits^T*F_logit*ui_logits]. No independent physical oracle enters these quantities.

For the AMBIENT isolated proposal -di*Ei, full-batch clipped-loss descent pairing equals di*inner(Ei,G). That product is computable from the JSON. It differs from raw-gradient pairing by the actor clipping scale, and differs from the actual retracted-mode pairing by normal/tangent conversion and finite retraction. The JSON does not supply the full-batch retracted-mode pairing, so it cannot be recovered by relabeling the ambient product. Current full-batch credit, subset credit, lookahead history and physical PG are four distinct objects.

The historical snapshots are saved before the next rollout update. Their reconstructed first proposal therefore uses the fresh policy's rollout credit, with ratio1/all-active in ideal arithmetic. Its moments already contain PRIOR update history. General later-epoch clipping-mask mechanisms apply to that history and later trajectory; the selected first-proposal scalar table does not itself demonstrate a current inactive-mask event.

## Quantitative witness: v7 half-rate rollout clock,4M, first gate

Exact row: `runs/HalfCheetah-v4__sepclip_mechanisms_rollout_clock__1__1791241877/diagnostic_snapshot_4013696.pt`, bank `first.0.gate.weight` (43-by17, row constraint). Indices are zero-based. Entries below are stored scalars; the ambient pairing column is only their elementary product.

| Mode | Lookahead sigma | Current clipped G component | Proposal d | Ambient full-batch clipped-loss descent pairing | Retracted-mode subset PG gain | Isolated modal Fisher cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | .002276607 | .008531870 | .001818003 | +1.55110e-5 | +9.74275e-6 | 3.72224e-5 |
| 1 | .001507857 | .004480226 | .001821348 | +8.16005e-6 | +9.28876e-6 | 1.30966e-5 |
| 2 | .001220126 | .002871770 | .001966201 | +5.64648e-6 | +9.51462e-6 | 6.16856e-6 |
| 14 | .000252960 | .000463965 | .002086857 | +9.68229e-7 | +8.31052e-7 | 1.01492e-6 |
| 15 | .000182774 | -.0000271103 | .002055263 | -5.57188e-8 | -1.23307e-6 | 4.51192e-7 |
| 16 | .000160808 | .000171818 | .002052942 | +3.52731e-7 | -4.43631e-6 | 4.00580e-7 |

Mode15 points against the CURRENT clipped covector in this ambient isolated comparison even though its lookahead singular value and realized proposal coefficient are positive. History and subsequent transforms can produce this; no present scalar isolates the fast rail, slow rail, lane adaptation or retraction as its cause. Mode16 has positive ambient current-credit pairing but negative retracted SUBSET pairing. Different state averages and retraction mean there is no contradiction, and neither sign is physical PG.

Lookahead sigma0/sigma16 is14.1573, while d16/d0 is1.12923. Weak lookahead singular modes receive comparable recorded diagonal proposal coefficients here. That measures realized spectral/lane allocation, not weak-mode signal strength, useful gait significance, or an exact-polar guarantee.

For this bank the full proposal has positive full-batch clipped pairing5.11142e-5 before tangent projection and4.77684e-5 after it. Raw-gradient actual-retracted descent pairing is4.77663e-5, while subset actual-retracted gain is6.55006e-5. Reported radial norm fractions are .22582(current G), .28666(lookahead) and .26765(proposal); tangent gradient/proposal cosine is .45776. These norm/cosine observations do not quantify lost physical credit. The six isolated subset modal gains sum2.37078e-5, and their diagonal Fisher entries sum5.83543e-5. Neither sum reconstructs the bank gain6.55006e-5 or its cost .000133106: unlisted modes, missing offdiagonal/residual proposal terms, modal cross-Fisher terms and nonlinear separate retractions remain.

## Lane adaptation defeats a diagonal-only spectral story

For a rectangular43-by17 bank, thin U has17 columns and V is17-by17. A complete realized proposal can be written:

    D = U*C*V^T + R
    C = U^T*D*V
    U^T*R = 0

The audit records only selected diagonal Cii. It omits modes3–13, all Cij for i!=j, and R outside the lookahead's left column space. The cascade alone is spectral in ideal exact arithmetic. However row lane scaling gives D proportional to A_row*U*f(sigma)*V^T: U^T*A_row*U need not be diagonal, and A_row*U need not stay in span(U). Column lane scaling has analogous right-space mixing. BF16 rounding can add further departures. Therefore the recorded diagonal coefficients cannot reconstruct the actual bank proposal or allocate all full-batch credit to singular modes.

Even complete ambient decomposition would need tangent/retraction handling. Row projection is linear at fixed W, but subtracting each mode and retracting it separately does not sum to the simultaneous finite retraction. Functional Fisher cost also contains offdiagonal mode terms .5*E[ui_logits^T*F*uj_logits], and cross-bank/head terms. A sum of isolated positive costs cannot substitute for the joint cost. Selected negative mode gains are evidence about these isolated counterfactuals only, not an additive decomposition of actual finite training damage.

## Why current credit and historical singular directions separate

The parent's exact later-epoch covector is average[Ahat*rho*M(Ahat,rho)*S_theta]. Changes in score/Jacobian, ratio weight and sign-conditioned clipping mask can each rotate it. Positive A remains active below the lower bound; negative A remains active above the upper bound. Clipfrac is not zero-PG fraction. Removing one active score component can change direction without greatly changing aggregate norm.

For one rail with decay beta and lookahead weight mu:

    v_new = beta*v_old + (1-beta)*G_current
    L = (1-mu*beta)*G_current + mu*beta*v_old

A component can reverse or become inactive in G_current while remaining in L through v_old. The two-rail Anvil map compounds this: fast history follows each epoch; rollout slow history samples only each rollout's first gradient with decay .98^K, then freezes through the remaining epochs. Actor-head Adam movement simultaneously changes later ratios and body Jacobians. Shared actor-wide clipping rescales current head/body contributions before their separate moments, and its changing factor changes historical relative weighting.

The current gradient may be exact for the stored surrogate yet different from a physical all-time derivative because stored state weighting and continuation credit differ. A moment may improve physical direction despite lower current-surrogate pairing, or preserve a surrogate/noise mode after it ceases to help physically. These possibilities require resolved held-out physical directional measurements; lookahead sigma, current component or diagonal proposal magnitude alone cannot choose among them.

## Provenance and interpretation limits

The historical layer execution manifest reports successful exit0, maxParallelRuns1, and audit hash `c560aebdb89e47a18580bd84103571525c7b19744a68056c20c21b930ee67dc6`. Current audit hash differs: `23fb1162c8c9c26d14320e4b744b9667feef29d5a143bf38d7eee6fccdb87b0e`. The current source explains field semantics; it is not claimed to be a byte-identical replay of historical execution. The manifest examined does not pin the imported frozen trainer source. This note preserves the historical scalar record and makes no new numerical execution claim.

| Inspected artifact/current source | SHA256 |
| --- | --- |
| `layer_historical.json` | `ba19fa734771e4eb15378403876f0599caf5b081bc95cb368a9218014e387d80` |
| current frozen v7 trainer | `07f5f9636103dd5fdc9896de3eb781078df8bb1cb9b3ad87bcb79907fc619c8d` |
| `ppo_epoch_credit_appendix.md` | `d78dbf8ce168196d897dfee985617f27e879ae4694af3c64d7c41619cb3cd2d1` |

No scalar table here resolves independent physical oracle replicas, finite-horizon tails, sampled-law clamp atoms, matched state occupancy or full ten-epoch feedback. It supports a more precise decomposition of measured optimizer behavior, not a spectral pruning recommendation, optimizer verdict or sub10k-versus20k causal explanation.
