# PPO active sets, joint Fisher motion, and physical credit

Analytical continuation of `ppo_epoch_credit_appendix.md`. The numerical rows
below are accepted layer-transport measurements at 4,013,696 steps, not new
experiments or physical-return evidence. Scores assume the ideal continuous
Beta density; the host sampler's clipping atoms remain a separately measured
approximation and invalidate an exact score identity if consequential.

## 1. A finite likelihood change is not a torque displacement

For native action u_j in (0,1), old shapes alpha_j,beta_j, and finite shape
changes da_j,db_j, the exact joint log ratio is:

    ell = sum_j[da_j*log(u_j) + db_j*log(1-u_j)
          - logB(alpha_j+da_j,beta_j+db_j) + logB(alpha_j,beta_j)].

The fixed action-rescaling Jacobian cancels. Shapes are formed by
alpha=1+softplus(z_alpha), beta=1+softplus(z_beta); finite da/db must use the
actual new shapes, not sigmoid(z)*dz in this exact formula.

For infinitesimal logits, ell is approximately xi=S_z dot dz. In shape
coordinates, t_alpha=log(u)-digamma(alpha)+digamma(alpha+beta), with the
analogous t_beta. S_z=(sigmoid(z_alpha)*t_alpha, sigmoid(z_beta)*t_beta).
For each action block, the observed log-density Hessian in logits is:

    H_logp = -F_z + diag(sigmoid(z)*(1-sigmoid(z))*t_shape).
    F_z = D*[diag(trigamma(alpha),trigamma(beta))
                     - trigamma(alpha+beta)*ones(2,2)]*D,
    D = diag(sigmoid(z_alpha),sigmoid(z_beta)).

Thus ell=xi+.5*dz^T*H_logp*dz+O(||dz||^3) for a fixed logit displacement.
An actor parameter displacement additionally changes dz through its nonlinear
Jacobian, normalization, and matrix retraction. A small average logit
linearization error does not certify accurate masks for samples close to a
clipping boundary. Tiny-alpha-minus-one FP32 arithmetic and endpoint clipping
also require the actual policy shapes and sampling law, not idealized formulas.

## 2. The active set is sign selected, joint across all actions, and epoch dependent

With ratio bounds .8 and 1.2, log boundaries are:

    ell_lower = log(.8) = -.223144; ell_upper = log(1.2) = .182322.
    inactive = (Ahat>0 and ell>ell_upper)
            or (Ahat<0 and ell<ell_lower).

An infinitesimal predictor substitutes xi for ell, but classification depends
on the boundary margin and the higher-order residual above. The ratio is joint
over all action dimensions: a likelihood change in one joint can remove the
entire sampled advantage contribution, including gradients for other joints.
A clipped positive sample with a high ratio contributes no policy-loss gradient;
the corresponding negative sample remains active. Ratio clip fraction therefore
neither equals inactive fraction nor measures removed useful credit.

At each epoch k, for ascent gradients:

    g_unclipped,k = mean[rho_k*Ahat*S_theta,k].
    g_clipped,k = g_unclipped,k
                   - mean[rho_k*Ahat*S_theta,k*indicator(inactive_k)].

This removed vector is the required credit-weighted measurement. A count or KL
cannot replace it. Ahat correlates with sampled action scores; treating advantage
signs as independent coin flips would change the relevant distribution.
As theta changes, the scores, actor Jacobian, ratios, and removed sample vectors
all change. The finite-batch clipped objective is continuous but piecewise
smooth: crossing a boundary can abruptly remove a sample's gradient, without a
fresh trajectory or new advantage. Historical momentum can retain that vector.
These are possible rotations of the optimization field, not evidence that any
particular late epoch or rail is harmful.

## 3. Joint Fisher cost specifies a second moment, not the active set

At fixed s, let dz(s)=sum_l dz_l(s) include all trunk, head, bias and gain
changes. For the exact old Beta action law, E[xi|s]=0 and
E[xi^2|s]=dz(s)^T*F_z(s)*dz(s). Averaging over the declared state distribution:

    Q = .5*E_s[dz^T*F_z*dz] = .5*E_(s,a)[xi^2].
    Q = sum_(l,m) G_lm, where G_lm=.5*E_s[dz_l^T*F_z*dz_m].

The accepted measurements are:

| 4M rollout-clock snapshot | Joint Q | Exact finite old-to-new KL | sqrt(2Q) | Cross-term fraction | Physical mean-motion RMS |
|---|---:|---:|---:|---:|---:|
| Half rate | .00363957 | .00362900 | .08532 | 74.71% | .006597 |
| Full rate | .01198588 | .01217121 | .15483 | 72.05% | .014005 |

Here sqrt(2Q) is the old-density RMS *linearized* log ratio, not the empirical
stored-action RMS or the finite-ratio standard deviation. The positive clipping
threshold is 2.14 and 1.18 times those scales; the negative threshold's magnitude
is 2.62 and 1.44 times them. These ratios give scale only. No Gaussian tail or
inactive-fraction estimate follows: Beta log scores can have endpoint tails,
state-dependent variance, and sign-dependent advantage correlation. Q fixes
only an aggregated second moment, not where credit falls relative to thresholds.

Cross terms reinforce. Summing isolated layer costs would understate joint Q
by factors about 3.95 and 3.58, respectively. Exact finite KL agrees with Q
within about .3% and 1.55%; reported average logit linearization errors are
1.05% and 2.34%. This validates these local functional measurements. It does
not certify active-set prediction, optimizer direction, or physical usefulness.
These rows describe a measured next update, not a ten-epoch cumulative KL.

## 4. Probability credit becomes physical torque through mean and precision

For physical actions on [-1,1], define kappa=alpha+beta and
m=(alpha-beta)/kappa. Local motion satisfies:

    dm = 2*(beta*da-alpha*db)/kappa^2.
    dlog(kappa) = (da+db)/kappa.
    physical_variance = (1-m^2)/(kappa+1).
    dlog(physical_variance) = -2*m*dm/(1-m^2) - dkappa/(kappa+1).

Increasing likelihood at a credited observed action is not simply moving mean
torque toward it. It can alter precision or a mixture of mean and precision.
Precision changes can alter sampled joint torques and contact timing despite
zero mean-torque change. Conversely a small torque mean displacement can cause
large gait changes through state-dependent contact and feedback. The measured
mean-motion RMS above establishes neither direction of locomotion benefit nor
its magnitude. It averages states/actions and contains no gait-phase credit.

In a differentiable control problem, score and pathwise action derivatives can
agree in expectation under suitable regularity. A single observed Beta score
does not equal a torque derivative, and no contact-smoothness assumption has
been established here. Physical conditional-return forks and actual closed-loop
returns are the independent connection to gait, with their stated horizon,
noise, occupancy, normalization, and tail restrictions.

## 5. Why neither KL nor loss improvement proves gait improvement

Fisher cost is quadratic: dz and -dz have identical Q but opposite pairing with
a fixed physical credit covector. Loss improvement establishes ascent for the
stored clipped surrogate, not that its credit matches finite physical returns.
Even correct critic credit at stored states does not remove the learner versus
propagated physical-occupancy distinction derived in the companion appendix.

For a specified physical field, joint cost and signed pairing together permit
direction-efficiency comparisons. Only when the field is independently resolved,
the objectives match, and the finite step remains valid can poor pairing support
an optimizer-transport explanation. Negative finite closed-loop return with
positive all-time directional prediction instead exposes finite-step curvature
or occupancy response; active-mask rotation alone does not identify that cause.

Evidence source: `investigation_report.json` ->
`restart_completed_layer_measurement` (job 11757, limit 1).
Formula anchors: `audit_sepclip_layer_transport.py` (`fisher_gram`,
`distribution_motion`) and sepclip v7 `ppo_loss`. No new jobs or source changes.
