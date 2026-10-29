# PPO epoch credit transport: exact identities and identification limits

Analytical appendix; no new experiment or measured causal conclusion. Applies to
the raw-advantage, full-batch sepclip v7/v12 setting unless stated otherwise.
Notation uses ascent gradients; implementation differentiates the negative loss.

## 1. What changes within a rollout

The stored rollout fixes observations s_i, actions a_i, old log probabilities,
advantages Ahat_i, and value targets. Updating the critic during the ten epochs
does not recompute those stored advantages. Let theta_k be the actor before
update k, rho_ki = pi_theta_k(a_i|s_i)/pi_old(a_i|s_i), and
S_ki = grad_theta log pi_theta_k(a_i|s_i). S includes the changing network
Jacobian and Beta score, not merely a fixed output score vector.

For lower/upper ratio bounds l and u, away from nonsmooth boundary ties:

    M_ki(A) = 0 if (A > 0 and rho_ki > u) or (A < 0 and rho_ki < l);
              1 otherwise.
    g_k = mean_i[rho_ki * M_ki(Ahat_i) * Ahat_i * S_ki].

For A=0 its contribution is zero. At exact boundary ties the autodiff convention
can choose a subgradient. A positive-advantage action below l remains active;
a negative-advantage action above u remains active. Consequently the logged
ratio `clipfrac` is not the inactive-gradient fraction.

At k=0, rho=1 and M=1, so g_0 = mean[Ahat*S_old]. For later epochs the exact
empirical change can be split as:

    g_k - g_0 = mean[Ahat * ((S_k-S_old)
                       + (rho_k-1)*S_k + rho_k*(M_k-1)*S_k)].

These are, respectively, changed score/Jacobian, importance weighting, and
removed clipping contributions. This is an algebraic decomposition, not three
independent interventions. Clipping can remove useful components or suppress
estimation errors; the inactive fraction alone cannot decide which happened.
Defaults have entropy coefficient zero and separate actor/critic parameters.
With entropy enabled, add its gradient explicitly before attributing actor motion.

## 2. State baselines: importance sampling still cancels, clipping generally does not

For any fixed state-only baseline b(s), with matching sampling density, full
support, differentiation under the integral, and an exact score:

    E_a~pi_old[rho_theta * S_theta * b(s)]
      = b(s) * integral pi_theta(a|s) * S_theta(a|s) da = 0.

Thus ordinary *unclipped* action importance sampling preserves exact baseline
cancellation at every theta. A ratio different from one does not invalidate this
identity. It holds for any fixed state distribution, including the old rollout
occupancy; it does not make that occupancy the current-policy occupancy.

Clipping replaces the integrand by rho*M(A)*S*b. The selected action region
depends on the ratio and advantage sign, so its integral need not vanish. Worse,
changing b can change the sign of A and hence the selected region itself.
At the fresh policy rho=1 all samples are active, restoring the unclipped identity.

Let a reference advantage A and estimated advantage Ahat=A+epsilon use the
same state/action distribution and theta. Their clipped gradient difference is:

    g(Ahat)-g(A) = E[rho*S*(M(Ahat)*epsilon + (M(Ahat)-M(A))*A)].

This separates amplitude error under the estimated mask from mask changes.
Even state-only epsilon can contribute later in PPO. A raw RMS advantage error
or mask disagreement rate does not measure this vector: score direction, ratio,
error sign, and cancellation across samples all matter.

On a finite fixed rollout, mean[rho*S*b] is generally nonzero even unclipped.
If b is fixed independently of the sampled action, its population mean is zero
but its empirical contribution is baseline noise, not evidence of critic bias.
A critic fitted to the same sampled trajectories, or a direction selected after
examining them, introduces dependence; the simple fixed-baseline unbiasedness
argument then needs conditioning, independent evaluation, or cross-fitting.
Batch advantage centering/scaling also uses random sample statistics and can
change signs/masks; it is not an exact harmless baseline transformation.

## 3. What critic approximation contributes to stored GAE

Let e_t=V_est(s_t)-V_pi_old(s_t). On a no-boundary segment with H remaining
residuals and the same terminal bootstrap, the pathwise GAE difference is:

    A_GAE(V_est)-A_GAE(V_pi_old)
      = -e_t
        + gamma*(1-lambda)*sum_{j=1}^{H-1}(gamma*lambda)^(j-1)*e_(t+j)
        + gamma*(gamma*lambda)^(H-1)*e_(t+H).

Termination masks and truncation final-observation bootstraps require their
actual boundary terms. The formula compares two GAE estimates on the same
realized path; it does not identify V_pi_old from self-bootstrapped targets.
In fresh unclipped expectation, -e_t cancels as a state baseline. Future errors
can depend on the initial action through the visited states and survive score
weighting. At lambda=1 only the terminal error remains besides -e_t; it is not
automatically zero. Infinite MC cancellation requires a vanishing tail or a
correct terminal value, not the label "lambda=1" alone.

Even exact old-policy advantages define an old-occupancy surrogate at later
theta. Action importance sampling does not replace old continuation actions
or transport the state distribution. Distinguish this surrogate mismatch from
critic error, clipping, and finite-trajectory reward noise.

## 4. Uniform learner occupancy is not propagated physical occupancy

Let d_B be the uniform empirical distribution over stored rollout observations.
Ignoring clipping and advantage error, the learner differentiates the surrogate:

    L_B(theta)=E_s~d_B E_a~pi_old[rho_theta(s,a)*A_pi_old(s,a)].
    grad L_B(theta_old)=E_s~d_B E_a~pi_old[S_old*A_pi_old].

There is no outer gamma^t on the learner's sampled state rows. Dense first-action
forks, uniformly averaged over the same states, estimate this conditional score
field for finite-H Q with the continuation policy fixed. Selected early-age
states instead define their own empirical d_S; their average is not automatically
the full-rollout d_B average.

The closed-loop physical objective starting from a fixed initial distribution nu
is J_H(theta;nu)=E[sum_{t=0}^{H-1} gamma^t*r_t]. Its derivative is:

    grad J_H(theta_old;nu)
      = sum_t gamma^t E_s~d_t(nu,pi_old) E_a~pi_old[S_old*Q_(H-t)(s,a)].

Here d_t is the distribution after t physical steps. Both the propagated state
weights and the remaining horizon change with t. For the infinite objective
and a valid vanishing tail, define
d_gamma(nu,pi)=(1-gamma)*sum_t gamma^t*d_t(nu,pi). Then:

    grad J_infinite(theta_old;nu)
      = (1-gamma)^-1 E_s~d_gamma(nu,pi_old) E_a~pi_old[S_old*A_pi_old].

Taking nu=d_B does not make d_gamma=d_B: states reached from those initial
states receive additional weight. Equality up to the positive scalar follows
under a stationary-distribution assumption with exact infinite old-policy Q,
or another demonstrated occupancy match. A finite collected rollout, episode
age selection, resets, frozen normalizers, and finite Q horizons do not establish
that assumption. Finite H also changes credit even if occupancy were stationary.

Consequently a positive learner-surrogate pairing and negative all-time
closed-loop physical derivative can coexist with an exact critic and exact
Monte Carlo expectations. That contrast is valuable evidence about actual update
effectiveness for the selected physical objective, but not direct identification
of critic bias. First-action matched-state comparisons and same-trajectory
GAE-versus-MC transport address narrower matched-occupancy questions.

A two-state counterexample makes the distinction concrete. Suppose the exact
conditional score-gradient in a shared parameter direction is +1 at A and -.75
at B, d_B gives each state half the weight, and both states transition to B
independently of the action. The uniform surrogate derivative is +.125. Starting
from that same mixture, the infinite physical derivative at gamma=.99 is
.125-.75*(.99/.01)=-74.125. The critic can be exact: only state weighting differs.
This example establishes possibility, not the degree of mismatch in these runs.

Existing reports explicitly guard all-time versus first-action scope and fixed
selected states. Those guards should not be read as a proof of occupancy equality.

## 5. Momentum follows a changing field, not a repeated fixed gradient

For one EMA rail v_k=beta*v_(k-1)+(1-beta)*g_k and the reference lookahead
q_k=(1-mu)*g_k+mu*v_k:

    q_k = (1-mu*beta)*g_k + mu*beta*v_(k-1).

Historical mass survives even after the current mask or Jacobian removes or
reverses a component. With beta=.95 the EMA's nominal mean age is 19 updates;
the scale is update count, not independent-rollout count. Correlated epochs do
not make those gradients statistically identical: their objectives' local
derivatives change deterministically with theta and clipping.

ANVIL blends fast and slow rails before its lookahead. In v7 the slow rail is
updated only on the first update of a fresh rollout, with decay .98^K for K
planned updates; it is held during later epochs. It therefore carries fresh
rollout gradients deliberately, not the current epoch's active score field.
Fast rails still follow within-rollout gradients. Norms, second moments,
spectral transforms, retraction, and simultaneous Adam readout movement then
change the resulting motion. A negative raw gradient/momentum cosine alone
does not prove a harmful policy step or identify which rail caused it.

## 6. Minimum conditions for calling an epoch harmful

| Observation | Defensible interpretation | Missing condition for a stronger claim |
|---|---|---|
| Stored PPO loss improves; independent finite-return directional pairing is negative | Surrogate and physical objectives disagree along this measured motion | Match raw/normalized units, state occupancy, all-time vs first-action objective, terminal tail, and sampling law; resolve paired uncertainty before assigning a mechanism |
| Stored gradient pairs positively but optimizer motion negatively with the same independent physical field | Local optimizer transport can lose useful direction | Comparable joint Fisher cost, actual combined actor motion, and finite closed-loop validation |
| First update improves paired closed-loop return; endpoint worsens | The remaining update sequence harms this finite conditional return | It does not identify clipping, critic error, momentum, or Jacobian change separately |
| Endpoint finite gain is negative despite positive directional prediction | Linear prediction fails at this step size | Ambient endpoint chord versus constrained tangent and nonlinear occupancy effects remain mixed |
| Mask replacement changes a score-weighted derivative | Estimated credit affects the later clipped target | Independent reference advantage accuracy, finite tail stability, and a resolved vector/directional difference |

An old-state conditional first-action PG is not the full closed-loop policy
derivative. Neither is a finite frozen-critic-tail derivative the true infinite
return derivative. Independent continuation replicas reduce continuation noise;
when the first action is held fixed they do not remove its sampling noise.
Conditional confidence intervals do not cover selected-state or training-seed
uncertainty. These restrictions apply before naming an epoch, critic, or
optimizer as the benchmark bottleneck.

Source anchors: `ppo_continuous_action_sepclip_rollout_clock_anvil_v7.py`
(`ppo_loss`, `matrix_update`, `begin_rollout`), `audit_sepclip_dense_pg.py`
(`clipping_sensitivity`), and `audit_sepclip_closed_loop_update.py`
(`finite_directional_pg`, `matched_gae_transport`). No source edits or ML jobs
were needed for this appendix.
