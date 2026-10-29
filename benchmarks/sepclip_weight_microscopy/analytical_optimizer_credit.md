# PPO credit and optimizer transport

Status: exact source algebra and existing scalar-JSON inspection, not new model execution or physical-gradient evidence. Sources are frozen `ppo_continuous_action_sepclip_rollout_clock_anvil_v7.py`, `scripts/audit_sepclip_layer_transport.py`, `layer_historical.json`, and `root_analytical_transport.md`. No training, test, GPU execution, new submission, or trainer modification was performed.

## The active mask is conditioned on advantage sign

For sample i, ratio ri=exp(log pi_theta(ai|si)-log pi_old(ai|si)), fixed advantage Ai, and score Si=gradient log pi_theta(ai|si), PPO's ascent covector away from clipping boundaries is:

    g = average_i[Ai * ri * mi * Si]
    mi = 1 if Ai>0 and ri<1+epsilon_upper
    mi = 1 if Ai<0 and ri>1-epsilon_lower
    Ai=0 contributes zero

The minimized actor-loss gradient is G=-g. At the exact clipping boundaries PyTorch's clamp/max subgradient convention applies; the expression above is the piecewise derivative off those boundaries. Entropy coefficient is zero in this audited protocol, and actor/critic trunks are independent, so critic loss adds no actor covector directly.

Concrete witness: take four equally weighted samples with (A,r)=(+1,.5),(-1,1.5),(+1,1.5),(-1,.5), and bounds[.8,1.2]. Logged clipfrac is100%, but only the last two have zero PG contribution; the first two remain active. Conversely, A=0 yields zero PG even inside the clipping interval. Clipfrac therefore does not measure zero-PG fraction, conditional credit magnitude, or useful physical signal.

Later epochs change ri, Si and the sign-conditioned mask while keeping the stored Ai fixed. A two-covector witness illustrates rotation: two positive-advantage samples with score vectors e1 and e2 have equal active weights initially, so their sum points along e1+e2. If the first ratio rises above1.2 while the second remains active, the sum points along e2. This is an algebraic mask mechanism; these abstract score vectors are not a claim about measured MuJoCo dynamics. Even without a mask transition, differing ratio weights or feature/score Jacobians rotate the sum. Body-bank gradients are the resulting signed, masked feature-cotangent outer-product sums, not unweighted gait covariances.

## Actor clipping couples hidden banks and Adam heads

The trainer applies one actor-wide norm factor to ALL actor parameters before either optimizer:

    c = min(1, .5/(norm(G_actor)+1e-6))
    G_hidden_clipped = c * G_hidden
    G_head_clipped = c * G_head

Head weights, head bias and readout_gain use Adam; actor gate/up/down banks use Anvil. The critic receives its own independent clipping factor. Thus a larger head gradient can attenuate the hidden gradient entering Anvil, and a larger hidden gradient can attenuate the head gradient entering Adam. The factor is scalar at one step and does not rotate that instantaneous actor covector. It changes history weights when c differs across optimizer steps. A witness is a current gradient c*e2 with old moment e1: changing c changes the angle of their blended direction even though the current gradient's own angle is unchanged.

Neither optimizer's final displacement generally shrinks in direct proportion to c. With zero history and negligible additive epsilons, Adam's first m/sqrt(v) and Anvil's normalized cascade largely remove common positive scale. With nonzero history, varying c, denominator floors, BF16 rounding and actual epsilons, scale cancellation is incomplete. It is incorrect either to diagnose clipping starvation from small norms alone or to assert clipping never matters because the optimizer normalizes scale. The head has parameterwise second moments and bias correction, while Anvil has spectral transformation followed by lane second moments; the shared clipping factor does not make their geometries equal.

## Fast and slow rails preserve different credit histories

For clipped matrix gradient Gj at update j, Anvil stores:

    fast_j = beta_fast_j*fast_previous + (1-beta_fast_j)*Gj
    blend_j = w_j*fast_j + (1-w_j)*slow_j
    lookahead_j = (1-momentum_j)*Gj + momentum_j*blend_j

The source schedule varies momentum/fast decay; blend weight is1 before mapped schedule time514 and .4385 afterwards. Slow-clock handling is especially specific. For a rollout containing K updates, the rollout-clock rail performs:

    slow_first = .98^K*slow_previous + (1-.98^K)*G_first
    slow_later = slow_first throughout the remaining rollout updates

The epoch-clock rail instead produces, after K updates:

    slow_end = .98^K*slow_previous + .02*sum_j[.98^(K-j)*Gj]

These endpoints agree for constant Gj, but differ when PPO ratio masks, scores or actor clipping change across epochs. The rollout clock is a first-epoch credit anchor; it is not an average of all ten later-epoch clipped gradients. With K=10, its first-gradient coefficient is about .1829. A concrete witness with old rail0, G_first=e1 and G_remaining=e2 leaves the rollout slow rail entirely along e1, whereas the epoch rail contains both axes. This is a history-selection distinction, not proof that either history is closer to the unknown physical PG.

NorMuon uses one rail with its declared momentum and no Anvil twin-rail schedule. The fresh parity/control jobs are required before transferring historical optimizer conclusions to that branch.

## Cascade, lane adaptation and retraction are separate maps

Anvil converts the lookahead to BF16 when requested, transposes tall matrices, constructs its normalized Gram, and applies six polynomial maps. In exact arithmetic each map changes singular values as sigma -> a*sigma+b*sigma^3+c*sigma^5. It is an approximate spectral map, not a guaranteed exact polar factor. Gram arithmetic, its1e-6 denominator and BF16 rounding are part of the implementation.

For the resulting V, the lane second moment follows:

    lane_second = beta2*old_second + (1-beta2)*mean_axis(V^2)
    H = V/sqrt(max(lane_second,1e-10))
    D = H * norm(V)/max(norm(H),1e-10)

For an m-by-n bank with m>=n, averaging over columns gives a separate row scale; for m<n, averaging over rows gives a separate column scale. Gate/up first banks43-by17 have row adaptation; second43-by64 and down64-by43 adapt columns and rows respectively. These lane transforms need not preserve the cascade's singular vectors. Restoring Frobenius norm retains magnitude before final BF16 rounding, not its policy Fisher cost. Shape scaling additionally multiplies by max(1,m/n)^.5 and learning rate.

The parameter then undergoes its row/column unit retraction. For a unit row w and proposal -eta*D, first-order surviving motion is -eta*[D-w*(w dot D)]. Finite retraction is exactly unit(w-eta*D)-w and includes higher-order chord effects. The head's Adam direction is retracted on unit rows too; head biases and gain are not. Projecting a direction after nonlinear preconditioning does not equal preconditioning a tangent gradient.

Concrete normal-to-tangent witness from the independent analytical report: W=[[1,0],[.6,.8]], ambient G=W, so the raw tangent gradient is zero for every row. The ideal polar factor Q=[[3,-1],[1,3]]/sqrt(10) has nonzero row-tangent projections [0,-1]/sqrt(10) and [-.8,.6]/sqrt(10). Retraction rotates rows even though the constrained objective .5*norm(W)^2 is constant. This proves a possible extrinsic conversion, not that it dominates the measured Anvil cascade. Adam can exhibit the same class of effect: for unit w=(1,1)/sqrt(2), radial gradient G=w and an elementwise positive preconditioner diag(1,2), its tangent projection is(-.5,.5)/sqrt(2), rather than zero. A constraint normal force is not automatically useful credit merely because a later map turns it tangent.

Past tangent moments also become partly normal after the parameter base moves. Projecting stored rails at the present W is a local counterfactual; without previous base points it is not identified as exact historical parallel transport.

## What existing layer scalars actually show

The historical layer audit computes ONE next-rollout full-batch proposal on a private loaded optimizer, then evaluates a deterministic512-state subset of stored training states/actions/GAE. Its pairing is average[A*score_old dot J*Delta] on that subset. It is not held-out physical PG, a full ten-epoch effect, or an environment-return derivative. At the full-batch old base, -G dot Delta is stored-surrogate credit; subset pairings need not agree with it.

The joint functional motion is u=sum_l[J_l*Delta_l]. Gram entry(l,m)=.5*average[u_l^T*F_logit*u_m], so the full sum is joint quadratic policy cost. Cross terms in the recorded rows are positive and supply roughly70–78% of that cost. Summing bank-only positive costs misses most of the joint motion here.

| Historical proposal | Joint quadratic cost | Cross cost | Finite logit/JVP relative error | Subset first-order mean / concentration pairing |
| --- | ---: | ---: | ---: | --- |
| rollout-clock half,4M | .00363957 | .00271899 | .01054 | +.00032016 / -.00000389 |
| rollout-clock full,4M | .01198588 | .00863630 | .02344 | +.00044349 / -.00001691 |
| rollout-clock full,8M | .000000711788 | .000000526219 | .00239 | -.00000330649 / -.0000000256548 |
| historical epoch-clock half,4M | .00554068 | .00431689 | .01160 | +.0000472734 / +.0000106324 |

For full-rate4M second-up, the actual retracted step has positive full-batch raw-loss descent pairing .000112961 while its subset stored-PG pairing is -.0000190617. This is a concrete observed warning against equating a subset sign with optimizer descent failure, and certainly against calling that sign physical PG. The full-rate8M aggregate subset pairing is negative despite positive listed full-batch hidden-bank descent pairings; subset occupation and head contribution both matter.

Projecting gradients/rails before the cascade sometimes increases raw descent pairing but also changes functional cost, and can worsen subset pairing. For full-rate4M first-gate, stored-subset pairing changes from+.0000904951 to+.0000840155 while its isolated quadratic cost increases from .000994060 to .001493346. These uncalibrated counterfactuals do not establish a better direction or an intervention gain. Cross-bank costs must also be included when comparing a joint counterfactual.

The recorded small finite logit/JVP errors support local function linearization for these ONE proposals on the selected states. They do not prove a ten-epoch finite update or closed-loop return approximation. Historical tiny-step exact-KL entries include negative values (for example epoch-clock half8M:-8.90941e-7); those are arithmetic artifacts, not negative true divergence. Use repaired FP64 same-shape execution only after its queued gate completes. Existing positive Fisher quadratics and function-difference errors are distinct measurements, not permission to trust these old tiny exact-KL numbers.

## Attribution remains gated

A useful investigation separately resolves sign-conditioned stored credit, actor clipping factors, moment history, cascade/lane transformation, post-retraction tangent/chord motion, joint functional Fisher cost, finite-policy change and physical feedback. Existing scalar observations locate distinctions worth measuring; they do not identify the physical ascent covector. Until independent physical pairing, horizon/tail reliability and closed-loop effects resolve, none establishes that head Adam, hidden Anvil, the rollout slow rail or architecture explains the sub10k-versus20k gap. No causal training candidate is selected by this algebra.
