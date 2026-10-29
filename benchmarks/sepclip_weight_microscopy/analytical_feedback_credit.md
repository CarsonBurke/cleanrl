# Closed-loop physical feedback and credit derivation

Source-derived analytical investigation, not a new measurement. No model/GPU execution, queue submission, or frozen-source change. This extends root_analytical_transport.md and the recorded physics, representation, critic and architecture artifacts. Equations below are ideal real-arithmetic local identities; recorded FP32, sampler clipping, online normalization, contacts and finite tails require the stated qualifications.

## State and observation are different objects

Let X_t be the complete MuJoCo integration state, including root position, joint positions/velocities, warmstarts and external controls. Let o_t = O(X_t) be the 17-dimensional policy observation. Root forward position is omitted from O; it remains in dynamics and displacement reward. A policy-observation Jacobian cannot reconstruct every integration-state direction. Restoring o alone is consequently insufficient for a counterfactual contact trajectory.

With frozen row means m and variances v, x_t = clip((o_t-m)/sqrt(v+epsilon), -10,10). Away from clipping boundaries,

D_x,O = diag(1[abs((o-m)/sqrt(v+epsilon)) < 10]/sqrt(v+epsilon)).

Thus D_x,X = D_x,O * D_O,X. Observation clipping removes individual coordinate sensitivity, not a radial feature direction. Actor feature normalization removes a different direction later. Physical speed perturbation is a selected qvel perturbation in X and its propagated O; it is not uniform radial rescaling of all 17 normalized coordinates.

Online normalization instead makes the row normalizer part of the closed-loop state. For a singleton Welford observation update with prior count n, delta=o-m, new count n'=n+1:

m' = m + delta/n'; v' = n*v/n' + n*delta^2/(n')^2.

The emitted observation uses the updated statistics. Therefore the feedback Jacobian includes sensitivity of m',v' to o, and sensitivity to their accumulated history. Counts do not vary with a local action perturbation until episode/row scheduling changes; means and variances do. Frozen diagnostics intentionally remove that adaptive feedback, and no inference about its absence in training follows from frozen evaluation. Reward normalization has another history variable R_t = gamma*R_(t-1)*(1-terminated_t)+r_t plus running return statistics; its scale and reward clip affect GAE units. Matching raw and normalized objectives requires the actual recorded applied scale and clip, not just multiplying by a final reward standard deviation.

## Actor and Beta control maps

For sepclip, write b1(x)=down1*SiTU(gate1*x,up1*x), h=normalize(b1), b2=normalize(branch2(8h)), f=8*normalize(h+b2), z=gain*(W_head*f+bias). Every normalize(y) has derivative (I-n*n^T)/norm(y), n=y/norm(y), outside its epsilon regime. The full J_z,x contains all three projectors and SiTU gate/up derivatives. Removing a branch radial response does not remove every physically useful state response.

For each joint, alpha=1+softplus(z_a), beta=1+softplus(z_b), k=alpha+beta. Native mean mu=alpha/k, physical mean a_mean=2*mu-1. Its exact derivatives are

d(a_mean)/dz_a = 2*beta*sigmoid(z_a)/k^2; d(a_mean)/dz_b = -2*alpha*sigmoid(z_b)/k^2.

Let A=alpha-1 and B=beta-1. The interior native mode q=A/(A+B), physical mode a_mode=2*q-1, has derivatives

d(a_mode)/dz_a = 2*B*sigmoid(z_a)/(A+B)^2; d(a_mode)/dz_b = -2*A*sigmoid(z_b)/(A+B)^2.

Mean and mode therefore have different feedback Jacobians even with identical actor logits. Mode preserves the relative excess ratio; mean includes the shape-floor pseudocounts. Mean/mode deployment is a different controller, not a variance-only intervention. Rounded FP32 shapes equal 1 create boundary modes; both 1 is uniform with nonunique mode. The new strong-reference diagnostic explicitly chooses midpoint for the uniform tie and computes (alpha-1)/[(alpha-1)+(beta-1)], avoiding addition/subtraction cancellation. Historical sepclip intended-logit mode can differ; its discrepancy must remain visible.

For sampled control, expose exogenous uniform epsilon_t and define U_t = inverse_CDF_Beta(epsilon_t;alpha,beta), a_t=2*U_t-1. Where pdf>0 and the inverse is smooth,

dU/dalpha = -(partial CDF/partial alpha)/pdf(U); dU/dbeta = -(partial CDF/partial beta)/pdf(U).

The conditional feedback J_a,x therefore depends on epsilon_t, not just the mean Jacobian. The experimental inverse-CDF uniform range excludes 2e-7 mass and native samples clamp at 1e-6/1-1e-6. Clamped regions have zero pathwise derivative but shape-dependent atom probability, so continuous Beta scores alone omit atom terms. Shared uniforms are a variance-reduction coupling of finite physical outcomes, not a guarantee that pathwise derivatives exist or that gamma rejection sampling has the same coupling.

## Dynamics, contacts, and physical reward

For a fixed controller/noise realization, X_(t+1)=F(X_t,a_theta(X_t,epsilon_t)). Define J_t=D_a,X and K_t=D_a,theta for fixed X. On a differentiable contact/solver branch,

M_t = F_X + F_a*J_t; B_t = F_a*K_t.

For parameter direction dtheta with fixed initial state, sensitivity L_0=0 and L_(t+1)=M_t*L_t+B_t*dtheta. Expanded, an injection at time j reaches time t through M_(t-1)*...*M_(j+1)*B_j. This is why a first-action intervention is not H times a full controller change: the full policy injects at every visited state, with state-dependent directions, and earlier changes also alter later injection locations.

MuJoCo contacts are piecewise responses involving active geometry, solver contacts, friction and impact timing. A local Jacobian on one active contact branch does not include a finite perturbation that changes the contact sequence. In continuous-time hybrid idealizations that change event time, the variational flow includes a saltation correction; the discrete simulator instead needs its actual branch-dependent map or paired finite interventions. Observed final-substep contact force is not an impulse integral. Tiny action perturbations may nevertheless create large finite stride changes near a contact transition. This is a reason to retain contact/gait trajectories, not proof that any particular PG or optimizer is wrong.

HalfCheetah raw reward is r(X,a) = forward_weight*(rootx_next-rootx)/dt - control_weight*sum_j a_j^2, with dt=.05 and default weights1/.1. Holding X fixed, its local action derivative is

r_a = (forward_weight/dt)*D_a rootx_next - 2*control_weight*a.

The second term is the immediate torque-cost derivative. The first depends on the actual physical dynamics; it is not simply a commanded torque or joint-power term. Expected direct cost for one joint is control_weight*(physical_mean^2+physical_variance). A reduction in noise can change both this tax and the entire subsequent contact/velocity flow. Existing strongest sepclip sample-to-mean gains around 1000 points and direct cost around 373 points cannot supply its remaining 9793 points to 20000.

## Physical adjoints and score credit are two representations of one specified objective

Fix J_H(theta)=E[sum_(t=0)^(H-1) gamma^t*r_t + gamma^H*C(X_H)] with fixed initial states and a terminal function C held independent of theta. For a differentiable deterministic or conditional-noise trajectory define p_H=C_X and

p_t = r_X + J_t^T*r_a + gamma*M_t^T*p_(t+1).

Then its pathwise parameter derivative is

sum_t gamma^t*K_t^T*(r_a + gamma*F_a^T*p_(t+1)).

This is the true physical adjoint for that conditional differentiable trajectory. It includes future dynamics and actor state feedback. A critic gradient is not automatically this adjoint, and a value network fitted to normalized self-bootstrapped targets does not certify it. Contact non-smoothness and integrability can invalidate exchanging pathwise derivative and expectation.

For the unclamped stochastic Beta law, the same specified expected-return derivative has likelihood-ratio representation

grad J_H = E[sum_t gamma^t*S_t*(G_t^C-b_t)],

where S_t=grad_theta log pi_theta(a_t|x_t) with the state held fixed inside this conditional score, G_t^C=sum_(j=t)^(H-1) gamma^(j-t)*r_j + gamma^(H-t)*C(X_H), and b_t can be any state/history-only baseline independent of the current action. Dependence of future occupancy on theta is already accounted for through earlier action scores and their return-to-go. Adding actor input-state chain derivatives again inside S_t double-counts that occupancy effect. The all-time sum cannot be replaced by a scaled first-action score covariance.

The baseline cancels in expectation because E[S_t|history before a_t]=0; it can still change estimator variance. A current-state value error alone is consequently not a bias certificate for the fresh unclipped expected score PG. Future value error in a bootstrapped credit estimator, finite-sample estimation, later PPO masks, and altered sampled-law atoms are separate issues.

## GAE is a credit estimator, not a physical dynamics adjoint

On one no-reset realized finite trajectory, define fixed old values V_t, V_H=C(X_H), and delta_t=r_t+gamma*V_(t+1)-V_t. GAE(lambda)_t=sum_(j=t)^(H-1) (gamma*lambda)^(j-t)*delta_j. GAE(1)_t=G_t^C-V_t pathwise. This telescoping identity requires the same terminal C and correct pre-reset truncation semantics; it does not establish that C is a true infinite-horizon value.

Define realized continuation error e_j=V_j-G_j^C. With the same terminal tail e_H=0, the exact paired identity is

GAE(lambda)_t - (G_t^C-V_t) = gamma*(1-lambda)*sum_(j=t+1)^(H-1) (gamma*lambda)^(j-t-1)*e_j.

The score-weighted directional transport difference is sum_t gamma^t*(S_t dot dtheta) times this expression. Future e_j depends on earlier actions through state and continuation reward, so it generally does not cancel like a state-only current baseline. Here G_j^C is a noisy realized return, not independently known V_pi. This identifies the difference between two estimators on the same finite objective; it cannot label the entire difference as critic bias, or distinguish conditional covariance from MC noise without replicas.

Discount is part of the objective and the propagation weights. Lambda changes the credit estimator; it does not replace the physical feedback operator M_t, nor say that a contact effect lasts only1/(1-gamma*lambda). At gamma=.99, gamma^256≈.0763 and gamma^512≈.00582 remain tail weights, not rigorous omitted-bias bounds.

## Clipping makes credit and later update paths interact

For saved state/action, fixed stored advantage A, ratio rho=pi_theta/pi_old, and score S_theta, PPO ascent away from clip boundaries is A*rho*S_theta*active(A,rho). Positive A is inactive only above upper ratio; negative A is inactive only below lower ratio. Logged two-sided clip fraction is not the inactive-gradient fraction.

A state-only shift of A may change sign and magnitude, which can change the mask and no longer gives the fresh score baseline cancellation. Later weights and optimizer moments depend on all prior masked minibatch gradients; new ratios then change later masks. The first update and ten-epoch endpoint therefore test different points along a credit-dependent optimizer path. A first-positive/full-negative physical result implicates that later trajectory, but does not by itself identify clipping, wrong credit, spectral transport, momentum or finite-step curvature as the cause. Retained trace masks, actual gradients, real chords, independent physical pairings and finite controllers resolve successive links.

## What the saved stronger PPO reference can falsify

The recorded saved reference has complete normalizer and Beta weights, approximately 14522.7 last20 training-return scalars, and 50M budget. Its physical evaluation is pending. Unlike sepclip it has an affine input projection h0=P*x+b, three pre-RMS residual updates h_(l+1)=h_l+sigmoid(g_l)*B_l(RMS(h_l)*s), final RMS and unconstrained head. Signed information can enter through P*x and persist in the identity residual stream. This provides a signed anchor within the stream; final RMS still removes its final feature radial direction and therefore does not automatically preserve input radius.

For d-dimensional RMS N(h)=h/q, q=sqrt(mean(h^2)+epsilon), its exact Jacobian is I/q - h*h^T/(d*q^3). Applied to h, D_N*h = epsilon*h/q^3. At epsilon 0 a purely homogeneous final stream would also be input-radially null. However pre-RMS residual stream is not generally homogeneous: ignoring biases/epsilon, for input scaling c, h1(c*x)=c*P*x + gated_B(N(P*x)), mixing degree1 and degree0 terms. The final direction can change as c changes. Affine biases further break homogeneity. This is a precise route to preserving signed/radial response, not proof that the trained reference uses it for speed.

If frozen sample/mean/mode physical evaluations exceed sepclip substantially, they falsify a universal ~10k attainable ceiling of the shared 1+softplus Beta family under this MuJoCo task. They do NOT falsify a sepclip-specific trained tangent restriction, because architecture, optimizer, training duration and trajectory history differ. Strong mean/mode speed with similar concentration could reject a claim that higher concentration alone is necessary; high concentration would be compatible with its achieved gait, establishing neither necessity nor sufficiency in isolation, nor an optimizer explanation.

Existing scalar 14522.7 training logs cannot establish the pending frozen reference velocity, controller gain, contact gait or radial Jacobian. Saved stronger-reference action/contact traces can show achieved stride/duty/velocity and stochastic disruption, but traces alone cannot demonstrate that a signed anchor or input-radial response caused them. A large return advantage with similar contact categories also does not mean similar local feedback Jacobians. Direct JVPs on physically meaningful state directions, supported counterfactual perturbations and matched fresh causal architecture contrasts would be needed for attribution; this note requests no new jobs merely to fill wait time.

Likewise, a weaker frozen evaluation would not prove the historical high score false: online normalization, historical sampler, source provenance and reset occupancy differ. The current source strictly loads the saved artifact and its live parity checks establish the present replay law; they cannot retroactively pin the historical executed source.

## Evidence decisions before any causal selection

1. Establish actual replay law, normalizer/bounds/provenance and sampled-score limitations.
2. Resolve independently held-out physical directional signal and horizon/tail stability; historical strongest PG replicas with cosines -.029/-.041 do not do this.
3. Compare stored GAE/clip credit to that supported finite physical objective, retaining all-time versus first-action distinctions.
4. Validate nonlinear real-chord JVP and joint functional cost before optimizer efficiency claims.
5. Interpret paired first/full-update physical gains and hybrid feedback interactions with conditional noise intervals and contact changes.
6. Use strong-reference physical behavior as an achieved-controller comparator, with unmatched 50M budget explicit; causal training remains separately evidence-gated.

No closed-loop equation or stronger scalar benchmark here establishes a 20k optimum, selects a chart or signed-anchor trainer, or bypasses unresolved physical-gradient reliability.

Independent architecture mathematical redteam verified the fixed-terminal adjoint, all-time conditional score/occupancy distinction, paired GAE future-error identity, RMS Jacobian and signed-anchor/radial caveat. It corrected concentration attribution: observing high concentration together with a strong jointly different controller establishes neither concentration necessity nor sufficiency in isolation.
