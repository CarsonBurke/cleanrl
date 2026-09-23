# Vector costate critic: prediction and optimization pressure

Status: mathematical design, independently reviewed. Implemented in
`cleanrl/ppo_continuous_action_latent_costate_v10.py`; numerical contracts and
training execution are recorded in [latent-costate-v10.md](latent-costate-v10.md).
This replaces the rejected intervention/contrastive proposal. No simulator
branches, scalar value prediction, GAE, or generated latent reward decoding.

## Prediction

Predict the marginal remaining reward associated with changing each state
feature, and learn how actions change those features. A large vector of return
correlations is not sufficient: the vector needs a recursive prediction target
that carries reward consequences through the dynamics.

Let s denote physical state, a the physical action, and t episode age. Define:

    f(s,a)             = physical next state
    rho(s,a)           = complete immediate reward, including transition effects
    a = g_theta(s,eps) = reparameterized stochastic Beta policy
    lambda(s,t)        = gradient of expected remaining reward with respect to s

There is no network predicting the scalar value whose gradient defines lambda.
That scalar defines the mathematical objective, not an architectural bottleneck.
Exact costates are integrable; arbitrary unrelated vector heads would not be a
more expressive correct solution.

For the actual observed transition and action:

    q_a = rho_a + f_a^T lambda(s_next,t+1)
    q_s = rho_s + f_s^T lambda(s_next,t+1)
    lambda_target = q_s + g_s^T q_a

Subscripts denote Jacobians/gradients. Taking the expectation over the action
noise gives the differential Bellman equation. Set lambda(s,T)=0 at the fixed
episode boundary. These equations use gamma=1, matching total episode reward.

The policy-state term g_s^T q_a is necessary: changing a state also changes
the policy's action distribution. Omitting it would not evaluate the current
closed-loop policy.

When computing the target, lambda_next is a **stopped cotangent**. Differentiate
`rho + stopgrad(lambda_next) dot f`, not a composition that differentiates
lambda_next itself. The latter would add an erroneous Hessian term. Detach the
completed target before critic regression.

## Latent features without a scalar readout

A physically anchored implementation can use critic features

    h_chi(s) = concatenate(s, learned_features_chi(s))

and predict a latent covector c_psi(h,t). Its physical costate is the pullback

    lambda(s,t) = J_h(s)^T c_psi(h(s),t).

The identity coordinates prevent a collapsed encoder from erasing all physical
directions. Training and comparing costates in fixed physical coordinates
avoids treating changes of latent coordinates as changes of physical value.
The output is a vector; there is no scalar value head or scalar return decoder.
Overcomplete latent coordinates are not independently identifiable, and extra
dimensions alone are not a claim of greater useful information.

Learn these critic features through the vector Bellman loss and the effect of
its errors on actor credit. Keep the actor's parameters separate: changing the
critic encoder must not silently change the behavior policy outside its KL
constraint.

Model one-step physical transitions using real transitions. Query the next
costate on the **observed next state**, not an averaged or imagined future latent.
There is no learned decoding of an expected latent into expected reward.

## How reward enters

For HalfCheetah, a candidate dynamics output contains next physical state and
forward displacement d. The immediate reward has progress and actuator-cost
components. With simulator time interval dt and control coefficient k:

    rho_a = d_a / dt - 2 k a
    rho_s = d_s / dt

The environment's exact reward and action conventions must be verified before
implementation. Calling this reward contraction analytic does not eliminate
model error: the progress derivative still depends on the displacement
Jacobian. Physical reward components anchor the derivative target; a learned
long-horizon scalar value does not mediate their credit.

## Actor credit

For the action sample actually used in the transition:

    credit_eta = (partial a / partial eta)^T q_a
    gradient_theta = (partial eta / partial theta)^T credit_eta

Here eta contains all alpha/beta coordinates. A valid implicit Beta derivative
or equivalent reparameterization is required for the observed sample. Merely
attaching gradients to a host-sampled action is invalid. Native-to-physical
action scaling and the actor's softplus derivatives enter exactly once.

At one sampled action, an actuator's alpha and beta credits share q_a but have
different sampling derivatives. Across samples these can produce distinct mean
and concentration updates. Their relationship is imposed by the policy family,
not an unwanted scalar value bottleneck.

The actor receives stopped q_a values. It updates its own parameters through
the policy derivative and a joint trust constraint; it cannot alter the critic
or dynamics model to manufacture favorable predictions.

## Pressure on prediction errors

Let e_next be the differential Bellman residual at a next state. Its direct
effect on the preceding transition's policy-coordinate credit is

    E_eta = (partial a / partial eta)^T f_a^T e_next.

Use both a full vector Bellman residual and this actor-relevant projection.
The projection explicitly penalizes errors that corrupt mean/concentration
credit. It cannot replace the full residual: its per-sample rank is at most
the action dimension, leaving other state directions unobserved.

Freeze the projection matrices during that critic update. Separately fit the
transition model to actual observations; do not let it reduce the critic loss
by changing its Jacobian to hide errors, or increase predicted reward by
inventing dynamics. The feature representation must remain physically anchored
while learning which prediction directions matter to the return gradient.

## What has to be true for this to work

- One-step prediction accuracy does not establish Jacobian accuracy. The central
  hypothesis is that physically structured transition learning supplies useful
  action and state derivatives in the encountered distribution. This remains
  an empirical risk, not a solved detail.
- Contacts can make derivatives poorly behaved. Finite-horizon derivative
  recursion can amplify errors: gamma=1 provides no general contraction, and
  even discounting would not automatically control the closed-loop Jacobian.
- Physical state sufficiency and observation normalization need explicit
  handling. A frozen affine observation transform has a well-defined derivative;
  changing normalizer statistics must not silently become physical dynamics.
- Learned latent coordinates introduce gauge and off-manifold ambiguities.
  Physical anchoring and pullback make the target interpretable but do not
  establish identification of every latent component.
- A KL constraint controls policy movement, not exploitation of model errors.
- Correct derivatives and costates yield the expected-return policy gradient
  under differentiability assumptions. Neither a learned approximation nor
  gradient ascent guarantees a globally optimal policy.

This design supplies a different prediction problem and recursive optimization
pressure. The proposed success criterion remains fresh end-to-end training
return. An accurate-looking dynamics fit or a separate variance gate would not
establish success.
