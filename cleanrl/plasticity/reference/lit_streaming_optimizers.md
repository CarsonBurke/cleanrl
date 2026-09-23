# Optimizers for streaming / online / noisy-gradient settings, and why Adam generalizes worse

Literature survey, 2026-09-11. Emphasis 2021-2026. Math is plain ASCII. Notation: g = stochastic
gradient, m = first-moment EMA, v = second-moment EMA, theta/w = parameters, alpha/eta = step size,
b1/b2 = EMA decay. "State/param" = extra floats stored per scalar parameter beyond the weight.
Adam's state is 2 floats/param (m, v); cost = 1 gradient per step + O(d) elementwise ops.

Verification level per paper: [F] = full text read (pdftotext or arXiv HTML render), [A] = abstract
or a secondary summary only. Claims from [A] papers are reproduced with less confidence.

---------------------------------------------------------------------------------------------------
## Thread 1. Bayesian-filtering / Kalman views of optimization
---------------------------------------------------------------------------------------------------

### 1.1 Aitchison 2020, "Bayesian filtering unifies adaptive and non-adaptive neural network optimization methods" (NeurIPS) [F]
arXiv 1807.07540.

**Model (exact).** Factorised posterior over each weight w_i separately. Objective per datapoint is
quadratic with shared Hessian H and datapoint-dependent mode: L_a(w) = -1/2 w^T H w + xi_a^T w,
E[xi_a]=0, Cov[xi_a]=H (Fisher identity). The *latent* is w*_i(t), the optimum of coordinate i
given the current values mu_{-i}(t) of all other coordinates:

    w*_i(t) = -(1/H_ii) H_{-i,i}^T mu_{-i}(t)                                         (10)

Observation = backpropagated gradient at the current estimate mu_i:

    g_i(t) | w*_i(t)  ~  N( H_ii (w*_i(t) - mu_i(t)),  H_ii )                          (12)

Dynamics: because the other parameters are being optimized, w*_i(t) moves. Aitchison models the
motion of mu_{-i} as a discretised Ornstein-Uhlenbeck process and therefore

    w*_i(t+1) | w*_i(t)  ~  N( (1 - eta^2/(2 sigma^2)) w*_i(t),  eta^2 )               (14)

with eta^2 = eta_mu^2 H_{-i,i}^T H_{-i,i}; sigma^2 is the stationary variance (prior variance over
the weight; matches He-init scale ~1/fan_in ~ 1e-3).

**Kalman filter (exact, per parameter).** Approximate H_ii ~ g^2(t):

    sigma2_prior(t) = (1 - eta^2/(2 sigma^2))^2 sigma2_post(t-1) + eta^2               (17b)
    1/sigma2_post(t) = 1/sigma2_prior(t) + g^2(t)                                       (18a)
    mu_prior(t)      = (1 - lambda) mu_post(t-1)          # lambda replaces eta^2/(2 sigma^2), = decoupled weight decay
    mu_post(t)       = mu_prior(t) + sigma2_post(t) * <g(t)>   # <g> = bias-corrected EMA of g (momentum)

So the *learning rate of parameter i is its posterior variance* sigma2_post,i(t).

**Fixed point / where Adam's square root comes from.** Plugging 18a into 17b and dropping small
terms gives a quadratic whose solution is

    1/sigma2_post ~= (1/(2 sigma^2)) * ( 1 + sqrt( 1 + 4 (sigma^2 / (eta / sqrt(<g^2>)))^2 ) )      (21)

Limits:
  * low information, eta/sqrt<g^2> >> sigma^2 :  sigma2_post -> sigma^2  (constant lr = SGD; set
    sigma^2 = eta_SGD / minibatch_size).
  * high information, eta/sqrt<g^2> << sigma^2 :  sigma2_post -> eta / sqrt(<g^2>)  (Adam with
    eta = eta_Adam, and weight decay is on mu not on the normalized gradient => AdamW).
  * If the optimum were *static* (no OU dynamics, eta=0) the precision would accumulate
    1/sigma2 = 1/sigma^2 + sum_t g^2, giving a *mean-square* normalizer (AdaGrad-like), not RMS.
    The RMS (sqrt) only appears because per-step information gain g^2 is balanced by per-step
    process noise eta^2. This is the paper's central claim: Adam's sqrt(v) is the steady state
    of a Kalman filter whose latent target does a random walk with variance eta^2 per step.

**Algorithms.** AdaBayes = eqs above with Adam-style EMAs for <g>, <g^2>. AdaBayes-FP: set
sigma2_post directly to the fixed point (21). AdaBayes-FP -> AdamW exactly as sigma^2 -> inf.
State/param: m, v, sigma2_post (3 floats; FP variant 2). Cost ~ Adam.

**Numbers (Table 1, CIFAR test error %, ResNet-34 / DenseNet-121).**
  CIFAR-10:  SGD 5.17/5.58, Adam 7.11/6.69, AdamW 5.08/5.19, AdaBayes 4.84/4.56, AdaBayes-FP 5.23/4.91
  CIFAR-100: SGD 22.71/21.29, Adam 27.59/26.64, AdamW 24.85/23.48, AdaBayes 22.92/22.09, FP 23.12/22.60
Test *loss* on CIFAR-100 is still clearly better for SGD (0.83 vs 0.97). Converges faster than Adam
early, generalisation ~ SGD.

**Known failures / caveats.** (i) H_ii ~ g^2 is admitted to be crude (same GM approximation
Khan et al. criticise, see 1.3). (ii) The OU model is a modelling choice, not derived. (iii) Only
CIFAR-scale evidence; never tested on transformers or RL. (iv) Two hyperparameters (eta, sigma^2)
instead of one. (v) In a streaming RL setting the "other parameters move" noise eta^2 is not
constant (policy changes), which is exactly the unmodelled part.

### 1.2 Khan & Rue 2023, "The Bayesian Learning Rule" (JMLR 24:1-46) [F]
arXiv 2107.04562.

**Rule.** For a candidate posterior q_lambda in an exponential family with natural parameter lambda,

    lambda_{t+1} = lambda_t - rho_t * ntilde_lambda [ E_q[ loss(theta) ] - H(q) ]

where ntilde is the natural gradient (Fisher-preconditioned) and H is entropy. Momentum variant
(App. F): + gamma_t (lambda_t - lambda_{t-1}) in natural-parameter space.

**Unified view (Table 1 of the paper), posterior family -> algorithm:**
  * Gaussian, fixed covariance + delta method (E_q[f] ~ f(mean))         -> gradient descent / SGD
  * Gaussian, full covariance + delta method                              -> Newton's method
      (online form: S_{t+1} = (1-rho) S_t + rho * Hessian;  m <- m - rho S^{-1} grad)
  * Gaussian, diagonal cov + delta + stochastic approx + Hessian~g^2 (GM approx)
      + square-root scaling + separate slow lr for scale vector           -> RMSprop / Adam
  * Mixture of Gaussians + responsibility approximation                   -> Dropout
  * Bernoulli + delta method                                              -> straight-through estimator
  * Gaussian diagonal, Gauss-Newton Hessian, NO sqrt                      -> OGN (online Gauss-Newton)
  * OGN without the delta method (expectation via weight perturbation)   -> VOGN
  * Exponential family, rho=1                                             -> conjugate Bayes / Kalman filter
  * Gaussian + delta method                                               -> Laplace's method
  * Mean-field exp. family, local rho=1                                   -> SVI / VMP

**The "correct" update (BLR-Newton for diagonal Gaussian, eq. 44/46):**

    s_{t+1}   = (1 - rho) s_t + rho * E_q[ diag Hessian ]      (Gauss-Newton: (1/M) sum_i (grad_j l_i)^2 per-example)
    theta_{t+1} = theta_t - alpha * ( E_q[ grad ] + delta*theta_t ) / ( s_{t+1} + delta )
    theta ~ N(theta_t, 1/(N s_t)) for the expectation (VOGN); delta = prior precision (weight decay)

Differences from Adam: (a) no square root; (b) Hessian estimate is the mean of per-example squared
gradients, not the square of the minibatch mean; (c) gradient evaluated at perturbed weights;
(d) weight decay inside the preconditioned term. The paper states explicitly that Adam's sqrt is
there "to take care of the factor 1/M^2 in the square of the mini-batch gradient" and that using
the squared minibatch gradient as a Hessian "results in poor performance as shown by Khan et al.
(2018, Thm 1)". Cost of OGN/VOGN ~ Adam except per-example gradient squares need custom autograd.
Claim: VOGN matches Adam accuracy with better calibration (Osawa et al. 2019, ImageNet ResNet-18).

### 1.3 Khan et al. 2018, "Fast and scalable Bayesian deep learning by weight-perturbation in Adam" (ICML) — VOGN / Vprop / Vadam [F]
arXiv 1806.04854.

    theta_t ~ N(mu_t, sigma2_t),  sigma2_t = 1 / ( N (s_t + lambda~) )
    VOGN:  s_{t+1}  = (1-beta) s_t + beta * hhat(theta_t),  hhat_j = (1/M) sum_{i in M} (grad_j f_i(theta_t))^2   (GGN, per-example)
           mu_{t+1} = mu_t - alpha * ( ghat(theta_t) + lambda~ mu_t ) / ( s_{t+1} + lambda~ )
    Vprop: same but s uses ghat o ghat (minibatch square) and the denominator is sqrt(s_{t+1}) + lambda~   (= RMSprop with perturbation)
    Vadam: Adam-form with momentum reweighted by (s_t + lambda~)/(s_{t+1} + lambda~)  (eq. 18)

**Theorem 1 (why g^2 is a bad Hessian).** With minibatch size M out of N, for the minibatch
gradient ghat_j:  E[ghat_j^2] = w h_j + (1-w) g_j^2,  w = (N-M)/(M(N-1)). So for M=1 the squared
gradient is an unbiased GGN estimate; for M=N it is purely the squared full gradient and contains
no curvature. Consequence for streaming (M=1): the squared single-sample gradient IS a valid
curvature estimate, so a *no-sqrt* preconditioner (Newton-like) is defensible exactly in the
batch-size-1 regime — the opposite of the large-batch regime where the sqrt is a patch.
Limitation: VOGN needs per-example gradients; requires small lr for M=1; not tested on RL.

### 1.4 Zhang, Sun, Duvenaud, Grosse 2018, "Noisy natural gradient as variational inference" (ICML) — Noisy Adam [F, ar5iv]
arXiv 1712.02390. Algorithm 1:

    gamma_in = lambda/(N eta);  gamma = gamma_in + gamma_ex
    w ~ N( mu, (lambda/N) diag(f + gamma_in)^{-1} )
    v = grad_w log p(y|x,w) - gamma_in * w
    m = b1 m + (1-b1) v
    f = b2 f + (1-b2) (grad_w log p)^2          # diagonal Fisher
    mu = mu + alpha * ( m/(1-b1^k) ) / ( f + gamma )     # NO square root

Same message as VOGN (no sqrt = natural gradient with a diagonal Fisher posterior). Damping gamma
plays the role of Adam's eps but is derived from the prior. Claims: matches Adam accuracy, better
uncertainty; the K-FAC variant (Noisy K-FAC) is the main contribution. State/param: m, f (2) plus
the sampled noise. No RL/streaming evidence.

### 1.5 Shen et al. 2024, "Variational learning is effective for large deep networks" (ICML) — IVON [F]
arXiv 2402.17641. Algorithm 1 (Improved Variational Online Newton):

    sigma = 1 / sqrt( lambda (h + delta) )          # lambda = effective sample size (~N), delta = weight decay / prior precision
    theta ~ N(m, sigma^2)                            # one sample per step
    ghat = grad at theta
    hhat = ghat * (theta - m) / sigma^2              # reparameterization-trick Hessian estimate (no per-example grads)
    g = b1 g + (1-b1) ghat
    h = b2 h + (1-b2) hhat + 0.5 (1-b2)^2 (h - hhat)^2 / (h + delta)     # keeps h > 0
    m = m - alpha * ( g/(1-b1^t) + delta m ) / ( h + delta )             # NO square root

State/param: g, h (2, same as Adam) + the noise sample. Cost: ~Adam (GPT-2 125M: 18.5 h vs 15.0 h
AdamW, i.e. +23%; ResNet-20 same time). Claims: GPT-2 773M ppl 12.6 vs AdamW 13.0; 355M 14.1 vs
14.5; ResNet-50 ImageNet top-1 77.46 vs AdamW 75.16, ECE 0.022 vs 0.066. Failures: does not work
with BatchNorm; h_0 in [0.01,1] critical; b2 must be ~1-1e-5 (0.999 unstable); needs gradient
clipping (xi=1e-3) on transformers; lambda too small -> unstable. Note the Hessian estimator
hhat = g * eps/sigma is unbiased but very high-variance with one sample; b2 ~ 1 - 1e-5 is what
makes it usable — i.e. in a fast-drifting stream the curvature memory is ~1e5 steps long.

### 1.6 Vuckovic 2018, "Kalman gradient descent" [F, arXiv HTML]
arXiv 1810.12273. Latent = true gradient g_t (random walk), position x_t; observation = stochastic
gradient. Joint state [x; g] with transition [[I, -alpha I],[0, I]], process noise sigma_Q,
measurement noise sigma_R. Update:

    v_{t+1} = (I - Ktilde_t) v_t + Ktilde_t * ghat_t      # Ktilde_t = adaptive Kalman gain, 0 < K < I
    x_{t+1} = x_t - alpha v_{t+1}

i.e. momentum with a *learned, time-varying* decay. Full covariance is 2d x 2d -> O(d^2.8);
block-diagonalised in practice ("distributed KGD"). Claims: bounded error variance, convergence;
better than SGD in high-noise toy/NN experiments. Failures: cost; block approximation loses the
cross-parameter structure; hyperparameters sigma_Q, sigma_R replace the momentum constant and
are not easier to tune; no large-scale evidence.

### 1.7 Davtyan et al. 2022, "KOALA: a Kalman optimization algorithm with loss adaptivity" (AAAI) [F]
arXiv 2107.03331. Latent = weights x_k (random walk with Q); *observation = the scalar minibatch
loss*, treated as a noisy measurement of a target value L_target (which is decreased slowly).
Linearising the loss gives a scalar Kalman gain (covariance approximated as scaled identity P):

    x_k = xhat_k - [ P_hat (L(xhat) - L_target) / ( P_hat ||grad L||^2 + R ) ] * grad L
    P_k = R P_hat / ( P_hat ||grad L||^2 + R )

This is a Polyak-step-like normalized gradient with a learned scalar gain; KOALA-M adds momentum
as an augmented state. R (measurement noise) is estimated online from loss variance. State: one
momentum buffer + scalars. Claims: CIFAR-10 ResNet-18 5.46% err vs (their) SGD 7.53, Adam 6.46;
CIFAR-100 ResNet-50 22.34 vs AdamW 27.23, AdaBelief 23.07; ImageNet-32 slightly behind SGD.
Failures: scalar gain, no per-parameter adaptation; "skip step when gradient norm below threshold"
unjustified; convergence only for the vanilla variant.

### 1.8 Ollivier 2018, "Online natural gradient as a Kalman filter" (EJS) [A]
arXiv 1703.00209. Result: the extended Kalman filter for estimating a fixed parameter of a
probabilistic model from a stream is *exactly* online natural gradient descent with lr 1/t and
Fisher matrix = posterior precision; with a fading-memory (forgetting) filter the lr becomes
constant. For recurrent models, joint filtering over states and parameters = natural gradient on
top of RTRL. Provides the general theorem that AdaBayes and BLR instantiate diagonally. No
experiments of note.

**Thread-1 takeaways.** Three independent derivations (Aitchison, Khan-Rue/Khan 2018, Zhang 2018)
agree: the principled per-parameter step is  alpha_i = 1/(curvature_i + prior_precision) with
curvature estimated from per-example squared gradients (or a sample-based estimator) — no square
root; the square root in Adam is a patch for using the squared *minibatch mean* gradient and, in
Aitchison's model, the steady state of a moving target. Nobody in this thread evaluated on RL or
non-stationary streams; all use b2 ~ 1 - 1e-5 (IVON) or fixed sigma^2 (AdaBayes), i.e. they
assume the curvature/noise is stationary over 1e4-1e5 steps.

---------------------------------------------------------------------------------------------------
## Thread 2. Continual / streaming RL optimizers
---------------------------------------------------------------------------------------------------

### 2.1 Elsayed, Vasan, Mahmood 2024, "Streaming deep RL finally works" — ObGD [F]
arXiv 2410.14606. Setting: batch size 1, no replay, no target network, eligibility traces.

**Derivation.** Effective step size (Kearney 2023) xi = (delta - delta+)/delta where delta+ is the
error after the update on the *same* sample; xi > 1 = overshoot. For semi-gradient TD(lambda)
with update alpha delta z under local linearity:

    xi = alpha z^T ( gamma grad v(x') - grad v(x) )  <=  alpha ||z||_1     (assuming each entry of |gamma grad v(x') - grad v(x)| <= 1)
       <= kappa alpha ||z||_1  <=  kappa alpha delta_bar ||z||_1,   delta_bar = max(|delta|, 1),  kappa > 1      (3)

**Algorithm 3, ObGD (exact):**

    z <- gamma lambda z + grad_w v(x)          # eligibility trace (of the semi-gradient); for supervised, z = grad f
    delta_bar = max(|delta|, 1)
    M = alpha kappa delta_bar ||z||_1
    alpha_eff = min( alpha / M , alpha )  = alpha / max(1, M)
    w <- w + alpha_eff * delta * z

State/param: the trace z only (1 float). Cost: one forward/backward + an L1 norm; no extra
forward pass (vs. backtracking, Algorithm 2, which needs repeated forwards). Defaults: alpha = 1,
kappa = 2 (critic), 3 (actor), lambda = 0.8, gamma = 0.99.

**Appendix B, Adaptive ObGD (RMSprop-style):**

    v <- beta v + (1-beta) (delta z)^2
    M = alpha kappa delta_bar || z / (sqrt(v) + eps) ||_1
    alpha_eff = min(alpha/M, alpha);   w <- w + alpha_eff delta z / (sqrt(v)+eps)

Other stream-x components: LayerNorm everywhere, SparseInit (90% of input weights zero), running
observation normalisation, reward scaling by std of the discounted return, entropy bonus.
**Claims:** stream AC(lambda) beats PPO1/SAC1 (batch-1 versions) on all MuJoCo tasks, beats PPO
and SAC on DMC Dog stand/walk ("best known model-free"), stream Q(lambda) beats DQN on 9 of the
Atari games tested; averaged over 30 runs x 20M steps (MuJoCo/DMC), 10 x 200M (Atari). Adam had
to have its lr lowered far below 3e-4 to avoid divergence in streaming. Ablation: removing ObGD
or LayerNorm makes learning unstable.
**Failures / criticisms:** (i) The bound assumes |gamma grad v(x') - grad v(x)|_i <= 1 with no
check. (ii) Sharifnassab et al. 2026 (2.3): the ObGD test is "a safety test on a moving TD error",
not a control of the prediction change; when consecutive gradients align it permits large
unintended changes. (iii) "Squeezing more from the stream" 2026 finds ObGD conflicts with SGD when
another loss (SPR) shares parameters; Stream Q(lambda) underperforms SGD-based variants there.
(iv) No noise model at all: the same step is taken for a noisy delta as for a reliable one, only
overshoot is bounded.

### 2.2 Javed, Sharifnassab, Sutton 2024, "SwiftTD: a fast and robust algorithm for temporal difference learning" (RLC, outstanding paper) [F]
Linear TD with binary features. Three ideas: (1) per-feature step-size optimisation (IDBD-style
meta-gradient on the *lambda-return* error, using True Online TD(lambda) so the target is not
delayed), (2) an overshoot bound applied when *adding to the eligibility vector*, (3) step-size
decay whenever the bound fires. Algorithm 1 (defaults eps=0.99, eta=0.1, eta_min=e^-15,
alpha_init=1e-7):

    v = sum_i w[i] phi[i];  delta' = r + gamma v - v_old
    for active-trace i:
        delta_w[i] = delta' z[i] - z_delta[i] v_delta;   w[i] += delta_w[i]
        beta[i] += e^{beta[i]} theta (delta' - v_delta) p[i];  beta[i] = clip(beta[i], ln eta_min, ln eta)   # IDBD meta step
        h_old[i]=h[i]; h[i]=h_temp[i]; h_temp[i] = h[i] + delta' zbar[i] - z_delta[i] v_delta
        z_delta[i]=0; (z,p,zbar)[i] *= gamma lambda
    tau = sum_{phi[i]!=0} e^{beta[i]} phi[i]^2        # correction ratio of this step
    T   = sum_{phi[i]!=0} z[i] phi[i]
    for phi[i] != 0:
        z_delta[i] = min(1, eta/tau) e^{beta[i]} phi[i]        # OVERSHOOT BOUND: total effective step <= eta
        z[i] += z_delta[i] (1 - T);  p[i] += phi[i] h[i];  zbar[i] += z_delta[i](1 - T - phi[i] zbar[i])
        h_temp[i] -= h_old[i] phi[i] (z[i]-z_delta[i]) - h[i] z_delta[i] phi[i]
        if tau > eta:  beta[i] += phi[i]^2 ln(eps);  (h_temp,h,zbar)[i] = 0     # STEP-SIZE DECAY

Correction ratio tau = sum_i alpha_i phi_i^2 = fraction of the error removed by one update (=1
means prediction jumps to target). eta = 0.1 bounds it. State/param: beta, z, p, zbar, h, h_old,
h_temp, z_delta (8 floats per feature). Claims: lower lifetime error than True Online TD(lambda)
on every game of the Atari Prediction Benchmark; robust across ~5 orders of magnitude of
alpha_init and theta; works with a one-layer conv net. Differences from IDBD/Autostep: exact
lambda-return meta-gradient; overshoot bound placed in the trace update; decay rather than
Autostep's hard normalisation. Limitations: linear or last-layer only; prediction not control
(Swift-Sarsa 2025 extends to control); per-feature not per-weight for deep nets.

### 2.3 Sharifnassab, Elsayed, Mahmood, Sutton 2026, "Intentional updates for streaming RL" [F, arXiv HTML 2604.19033]
Principle: specify the intended change of the *function output* on the sampled input, then solve
for the step size (first-order). Value: want V_{w+}(s_t) - V_w(s_t) ~= eta delta_t.

    Intentional TD:  alpha_t = eta / sqrt( sigma_bar_t * <rho_t z_t, z_t> )
        z = eligibility trace, rho_t = RMSProp-style per-parameter normaliser 1/(sqrt(nu_t)+eps),
        sigma_bar_t = discounted running aggregate of normalised gradient magnitudes (scalar)
    Intentional PG:  want log pi_{t+1}(a|s) - log pi_t(a|s) ~= eta A_t / A_bar_t ;
        alpha_t = eta / ( A_bar_t ||g_t||_2^2 ),  A_bar_t = EMA of |advantage|

State/param: z, nu (2) + scalars. Cost: "one Intentional AC update > 100x fewer FLOPs than one
SAC update". Claims: MuJoCo 5M steps, 30 runs, Ant 5513 +- 54 vs StreamAC 4898 +- 84, stable on
all tasks with one shared setting (eta_critic 0.5, eta_actor 0.05); DMC same; MinAtar competitive
with batch; Intentional TD(lambda) less sensitive to gradient scale than StreamTD. Measured
actual/intended change ratio 1st-99th pct 0.94-1.03. Limitations (stated): action-dependent step
size can reweight actions and change the expected policy-gradient direction; small gradient ->
large step -> unintended change at *other* states (conservative safeguards "did not help");
eta still per-algorithm. Diagnosis of ObGD: its overshoot test is on the moving TD error and can
miss large changes when gradients align.

### 2.4 "Revisiting Adam for streaming RL" 2026 [F, arXiv HTML 2605.06764]
Diagnosis of plain Adam at batch size 1: (a) unbounded MSE gradients turn large TD errors into
huge steps; (b) default b1=0.9 bad; (c) *epsilon acts as an SNR filter*: large eps suppresses
updates on rarely-active / noisy features. Adaptive Q(lambda), Algorithm 1:

    g = grad q(s,a,w);  z = gamma lambda z + g;  v = gamma lambda v + (1 - gamma lambda) g^2
    rho = z / (sqrt(v) + eps);  delta_hat = clip(delta, -1, 1);  w += eta delta_hat rho
    traces reset on exploration/episode end; b = 0.999; eps = 0.1 (vs 1e-8)

Claims: beats StreamQ (ObGD) on MinAtar by mean and IQM; Atari-55 ~3x human vs ~2x for C51 and
StreamQ; P(improvement over StreamQ) = 0.65. Limitations: heuristic SNR argument; Atari/MinAtar
only. Important corroboration: eps=0.1 is a *huge* noise floor — this is the same "shorten updates
in high-relative-variance directions" mechanism as Balles-Hennig's gamma (5.1), implemented
crudely.

### 2.5 Sutton 1992, IDBD (AAAI) [F]; Mahmood & Sutton 2012, Autostep (IJCNN) [F]; Degris et al. 2024, "Step-size optimization for continual learning" [F]
IDBD (linear, LMS): delta = y* - y; for each i:

    beta_i += theta * delta * x_i * h_i            # meta step on log step size
    alpha_i = exp(beta_i)
    w_i    += alpha_i delta x_i
    h_i     = h_i [1 - alpha_i x_i^2]_+ + alpha_i delta x_i      # trace of recent weight changes = d w_i / d beta_i

Meaning: step size grows when the current update delta x_i is positively correlated with the
trace of past updates h_i (consistent direction), shrinks when they anti-correlate (oscillation).
State/param: beta, h (2). Autostep (tuning-free, mu = 1e-2, tau = 1e4):

    v_i = max( |delta x_i h_i|, v_i + (1/tau) alpha_i x_i^2 ( |delta x_i h_i| - v_i ) )     # running max normaliser, makes meta-update unitless
    alpha_i *= exp( mu delta x_i h_i / v_i )
    M = max( sum_i alpha_i x_i^2 , 1 );  alpha_i /= M                                     # effective-step-size (overshoot) normalisation
    w_i += alpha_i delta x_i;  h_i = h_i (1 - alpha_i x_i^2) + alpha_i delta x_i

Claim: one mu works across all tested problems, beats IDBD/ALAP/RLS at a single meta setting.
Degris et al. 2024 (arXiv 2401.17401): RMSProp/Adam do step-size *normalisation*, IDBD does
step-size *optimisation*; on a 2-D regression with one constant and one sign-flipping target
weight, RMSProp/Adam trajectories move *away* from the optimal step-size vector (alpha1=0,
alpha2~0.33) while IDBD approaches it and matches oracle SGD; IDBD's meta-step sensitivity shifts
5 orders of magnitude with target scale (motivating Autostep). "Optimizing step-sizes in deep
neural-networks in a practical way for continual learning is still an open research question."
Noise: IDBD's meta-gradient delta x_i h_i has E = correlation of successive true updates but its
variance scales with the noise of delta; with i.i.d. noise the signal is O(||true grad||^2) while
the noise is O(sigma^2 d)-ish, forcing tiny theta. Autostep's running-max normaliser and SwiftTD's
bound+decay are the two known fixes; neither is a noise model.

### 2.6 Dohare et al. 2024, "Loss of plasticity in deep continual learning" (Nature 632) — Continual Backprop [F, PMC]
Utility of hidden unit i in layer l: u_i = eta u_i + (1-eta) |h_i| sum_k |w_ik| (eta = 0.99;
the arXiv version 2306.13812 also mean-corrects the activation and divides by the incoming-weight
L1 — check Methods before implementing). Every step, among units older than maturity threshold m,
reinitialise a fraction rho (1e-5 to 1e-4, ~1 unit per 200-500 updates) with lowest utility:
resample incoming weights from the init distribution, set outgoing weights to zero, reset utility
and age. Claims: on Continual ImageNet / Permuted MNIST / slowly-changing regression, plain
backprop, Adam, dropout, BatchNorm all lose plasticity; Adam "quickly lost almost all diversity"
(dead units, rank collapse); L2 and shrink-and-perturb keep it; continual backprop keeps it
indefinitely. RL: PPO on Ant with friction changed every 2M steps collapses; PPO + continual
backprop + L2 keeps learning; on *stationary* Ant, PPO collapses after 20M steps while CBP keeps
improving. Limitations: replacement rate must be tuned (robust range), L2/S&P sensitive; a
non-gradient random component is claimed *necessary*. Related: Kumar, Marklund, Van Roy 2023
"L2 Init" [A] — regulariser lambda ||theta - theta_0||^2 instead of ||theta||^2; consistently
mitigates plasticity loss with one hyperparameter.

### 2.7 Lan & Mahmood 2023, "Elephant neural networks: born to be a continual learner" [F]; Lan, Vasan, Mahmood 2025 [A]
Elephant(x) = 1 / (1 + |x/a|^d): bell-shaped, a = width, d = slope; both the function *and its
gradient* are sparse (Lemma 4.3). Argument: the change of output at x from an update on x_t is
-alpha dL/df <grad_w f(x), grad_w f(x_t)> (eq. 1); sparse gradients make this inner product local,
so learning on x_t does not disturb far-away x. Claims: single-pass Split-MNIST without replay;
streaming regression without forgetting. 2025 follow-up: RL with the replay buffer cut by 99%,
Atari-10, continuous control. This is a representation fix, not an optimizer, but it directly
targets the "unpredictable target" problem by making interference local. No Alberta paper was
found that makes an optimizer *ignore unpredictable targets* per se; the closest are SwiftTD's
step-size decay (features that keep triggering the overshoot bound get their step size decayed
toward 0) and FADE (2.8).

### 2.8 "Learning to forget: continual learning with adaptive weight decay" (FADE) 2026 [F, arXiv HTML 2604.27063]
Per-parameter decay rate lambda_i meta-learned with an IDBD-style trace:

    gamma_i += theta_lambda delta x_i g_i;   lambda_i = exp(gamma_i)
    g_i = g_i [1 - lambda_i - alpha x_i^2]_+ - lambda_i w_i           # g_i ~ d w_i / d gamma_i
    w_i = (1 - lambda_i) w_i + alpha delta x_i

Head-only (linear derivation). Claims: nonlinear teacher-student tracking, FADE+SGD MSE 0.0073
vs AdamW 0.0138; FADE+IDBD best on linear tracking. Limitation: extending to hidden layers
"plateaus below head-only FADE".

### 2.9 Other 2026 streaming work [A]
"Towards batch-to-streaming deep RL for continuous control" (2603.08588): S2AC / SDAC streaming
versions of SAC/TD3 using ObGD; L2 norm of critic weights much lower under ObGD than Adam
("different solution geometries"). "Squeezing more from the stream" (2602.09396): SPR auxiliary
loss in streaming; needs gradient orthogonalisation against momentum history; ObGD and SGD
conflict on shared parameters. "Streaming RL under partial observability with RTRL" (2605.24709).

---------------------------------------------------------------------------------------------------
## Thread 3. Non-diagonal / geometry optimizers and their noise behaviour
---------------------------------------------------------------------------------------------------

### 3.1 Muon (Jordan 2024 blog) [F]; Liu et al. 2025 "Muon is scalable for LLM training" [F]
For each 2-D weight W (m x n):

    M = mu M + G                       (Nesterov: use G + mu M)
    X = M / ||M||_F ;  5 x  X <- a X + b (X X^T) X + c (X X^T)^2 X,  (a,b,c) = (3.4445, -4.7750, 2.0315)
    O ~= U V^T (polar factor);   W -= lr * O           (Moonlight: W -= lr * 0.2 sqrt(max(m,n)) O + lr wd W)

State/param: 1 (momentum). Cost overhead fraction ~ T m / B (T=5 NS steps, m=width, B=batch
tokens): 0.7% for nanoGPT, <1% at 405B scale — but proportionally larger at small batch. Claims:
NanoGPT speedrun 1.35x; CIFAR-10 94% in 2.6 vs 3.3 A100-s; Moonlight (Liu 2025): ~2x FLOP
efficiency vs AdamW (52% of FLOPs to match) at 400M-1.5B; 3B/16B MoE on 5.7T tokens. Why it helps
(Jordan): SGD/Adam updates for matrices are near-low-rank; orthogonalisation "increases the scale
of rare directions". Noise: the blog says nothing; Liu 2025 reports attention-logit blowups
needing RMSNorm-gamma decay, and no advantage when fine-tuning a checkpoint pretrained with AdamW.

### 3.2 Bernstein & Newhouse 2024 "Old optimizer, new norm" [A]; "Modular duality" (ICML 2025) [A]
Switching off EMAs: Adam = sign descent = steepest descent under the max-of-max (elementwise
l_inf) norm; Shampoo without accumulation = steepest descent under the spectral norm (= Muon);
Prodigy/Lion map similarly. Recipe: assign each tensor an operator norm by its role (Linear:
spectral / RMS->RMS; Embed: l_inf->RMS), dualise the gradient under that norm. Explicitly frames
EMA as the noise-handling layer *on top of* the geometry — i.e. the theory is deterministic and
says nothing about how the dual map interacts with noise.

### 3.3 Why orthogonalisation hurts or helps under noise
* "Denoise first, orthogonalize later: understanding momentum in Muon via spectral filtering" 2026
  [F, 2606.03899]. Polar factor is scale-invariant: applied to G = G_sig + Xi it "erases the
  amplitude gap between signal and noise directions". Theorem 1: with momentum window
  T = 1/(1-beta), signal singular values sigma_k(M_t) >= c lambda_k - sqrt(eta)/(2T-1)^{1/4}
  while the noise floor is <= sqrt(eta)/(2T-1)^{1/4}; Theorem 2: pre-polar (Muon's order:
  momentum then orthogonalise) recovers alignment >= 1 - C'/sqrt(2T-1); post-polar
  (orthogonalise each gradient then average) and polar-only are bounded away from 1; Theorem 3:
  in the low-SNR rank-1 spiked model, polar-only alignment <= 1/sqrt(m) -> 0 with width. Verdict:
  Muon is only noise-robust because momentum filters *before* the polar step; the polar step
  itself is maximally noise-amplifying (it whitens noise directions up to unit singular value).
* NAMO, "Adam improves Muon" 2026 [F, 2602.17080]: "orthogonalization is an unbounded operation
  that may amplify the impact of noise in the momentum matrix". Fix: scalar Adam-style scale,
  alpha_t = ||Mhat_t|| / (sqrt(vhat_t) + eps), Theta -= eta alpha_t O_t (NAMO-D: per-column D_t).
  GPT-2 124M @50k: AdamW 3.0643, Muon 3.0435, NAMO 3.0351, NAMO-D 3.0246; wider stable lr range.
* ROOT 2025 [A, 2511.20626]: two Muon fragilities — NS coefficient precision depends on matrix
  dimension; outliers in the momentum are amplified by the polynomial NS iteration. Fix:
  dimension-specific NS coefficients + proximal (soft-threshold) outlier removal before
  orthogonalising. Claims better than Muon and Adam "in noisy and non-convex scenarios".
* "Muon is not that special: random or inverted spectra work just as well" 2026 [A, 2605.11181]:
  Kaon replaces the singular values with random values (keeping U, V) and matches Muon. What
  matters is the singular *vectors* and step-size optimality (alignment + descent potential),
  not spectral flattening — which suggests the noise-amplification of flattening is tolerated,
  not beneficial.
* PolarGrad 2025 [F, 2505.21799]: X_{k+1} = X_k - gamma tr(H_k) U_k where U_k H_k = polar(G_k)
  (momentum variant on M_k). tr(H) = nuclear norm restores "null-gradient consistency" (update
  -> 0 as G -> 0; Muon's does not, so Muon keeps taking unit-size steps on pure noise). Rates:
  linear under strong convexity with factor 1/(r_k kappa_H); stochastic case degrades by
  (1+delta)^2/(1-eps)^4 and plateaus at a noise floor requiring lr decay. Claims better than Adam
  and Muon on GPT-2 small / Qwen2.5 pretraining. Directly relevant: on a noisy stream Muon's
  unit-norm step is *pure noise amplification* when the true gradient is small; PolarGrad's
  nuclear-norm scaling is the minimal fix.
* Scion (Pethick et al. 2025, ICML) [F, 2502.07529]: x += gamma lmo(d), d = (1-a) d + a g;
  norm-ball choice gives normalised SGD (l2), signSGD (l_inf), Muon (spectral). Memory: weights +
  gradient only. Constant momentum: converges to a noise ball of radius ~ sigma (Thm 5.4);
  vanishing momentum: O(1/n^{1/4}) (Thm 5.5). 3B: 2.882 vs Muon 2.909 vs Adam 3.024; lr invariant
  to width.
* SOAP (Vyas et al. 2024) [F, 2409.11321]: Adam run in the eigenbasis (Q_L, Q_R) of Shampoo's
  L = sum G G^T, R = sum G^T G, refreshed every f steps. State per m x n matrix: 2m^2 + 2n^2 + 3mn.
  360M/2M-token batch: >= 40% fewer iterations, >= 35% less wall clock than AdamW; at 256K batch
  only >= 25% fewer iterations — the gain *shrinks with smaller batch*, i.e. with more noise.
* Dion (Ahn et al. 2025) [F, 2504.05295]: rank-r power iteration on the momentum with error
  feedback M <- B - (1-mu) P R^T (residual kept in the momentum buffer, no extra state).
  State: m n + n r. "Larger models are more robust to sparsification", scales better with batch
  size; full-rank Dion beats Muon at 3B. Low rank + error feedback is itself a noise filter.

**Thread-3 takeaways.** Orthogonalisation/whitening is beneficial *only after* temporal filtering
(momentum window T) and only if the update magnitude is tied back to the signal (nuclear norm,
NAMO scalar, or an overshoot bound); otherwise the polar step turns a noise matrix into a
full-rank unit-spectral-norm step. Every large-batch gain reported (SOAP, Muon, Scion) shrinks
as batch size drops. No paper in this thread evaluates at batch size 1 or in RL.

---------------------------------------------------------------------------------------------------
## Thread 4. Noise / variance-aware Adam variants
---------------------------------------------------------------------------------------------------

Format: rule | state/param | cost | strongest claim | failures.

**Schedule-Free (Defazio et al. 2024, NeurIPS)** [F, 2405.15682]
    y_t = (1-beta) z_t + beta x_t;   z_{t+1} = z_t - gamma grad f(y_t);   x_{t+1} = (1-c_{t+1}) x_t + c_{t+1} z_{t+1},  c_{t+1} = 1/(t+1)
    AdamW form: z -= gamma_t g/(sqrt v + eps) - gamma_t lambda y;  c_{t+1} = gamma_t^2 / sum gamma_i^2
State: z, x (+v) = 3. Cost: 1 gradient. Claim: E[F(x_T) - F*] <= DG/sqrt(T) for *any* beta;
AlgoPerf 2024 self-tuning winner; beats cosine on CIFAR/IWSLT/nanoGPT. Failures: BatchNorm
mismatch between y and x; still needs warmup; DeepSpeech worse. For streams: x is a uniform
(1/t) average of all iterates — provably the wrong thing under drift; would need c bounded below,
at which point it is Polyak averaging with a window.

**AdEMAMix (Pagliardini et al. 2024)** [F, 2409.03137]
    m1 = b1 m1 + (1-b1) g;  m2 = b3 m2 + (1-b3) g  (b3 ~ 0.9999, no bias correction);  v as Adam
    theta -= eta ( m1hat + alpha m2 ) / ( sqrt(vhat) + eps ) + wd;  alpha ~ 5-10, alpha and b3 warmed up
State 3. Claim: 1.3B on 101B tokens ~= AdamW on 197B (+95% token efficiency). Failures: early
divergence without schedulers; no gain on small data; *explicitly harms adaptation to sudden
distribution shift* (retaining 1e4-step-old gradients).

**MARS (Yuan et al. 2024, ICML 2025)** [F, 2411.10438]
    c_t = g(x_t, xi_t) + gamma b1/(1-b1) ( g(x_t, xi_t) - g(x_{t-1}, xi_t) );  c_t <- clip_norm(c_t, 1)
    m = b1 m + (1-b1) c_t;  then AdamW / Lion / Shampoo preconditioning
State 2-3. Cost: a second gradient at x_{t-1} on the *same* minibatch (MARS-approx reuses
g(x_{t-1}, xi_{t-1})). Claim: GPT-2 large 2.53 vs AdamW 2.56 at 50B tokens, reaches AdamW's final
loss in 27B tokens. Failures: gamma=0.025 is small (the variance-reduction term is nearly off);
the extra gradient doubles cost; in streaming with batch 1 the "same minibatch" reevaluation is
one extra forward/backward per sample.

**Cautious optimizers (Liang et al. 2024, ICLR 2026)** [A, 2411.16085]
    u <- u * 1[u * g > 0] * numel / sum(mask)    (elementwise, any momentum optimizer)
Claim: 1.47x Llama pretraining, 1.28x MAE; preserves Adam's Hamiltonian. Under i.i.d. noise the
mask fires randomly for low-SNR coordinates (P(sign agree) -> 1/2) so it halves and rescales
their steps — an accidental SNR gate.

**Grams (Cao et al. 2024)** [F, 2412.17107]
    u = mhat/(sqrt(vhat)+eps);  uhat = sign(g) o |u|;  w -= eta uhat
Claim: Llama-60M 1000 steps ppl 38.60 vs C-Adam 43.21 vs Adam 49.83; "strictly better descent
than Cautious" (Thm 4.3). Failure: direction is the *raw current gradient's sign*, i.e. maximal
noise sensitivity; the evidence is 1000-step runs.

**ADOPT (Taniguchi et al. 2024, NeurIPS)** [A, 2411.02853]
    m = b1 m + (1-b1) g / max( sqrt(v_{t-1}), eps );  v = b2 v + (1-b2) g^2;  theta -= alpha m
Claim: optimal O(1/sqrt T) for any b2 without bounded-noise assumption (decorrelates g_t from its
own normaliser). Same structural move as LaProp (normalise-then-average).

**AdaBelief (Zhuang et al. 2020, NeurIPS)** [A, 2010.07468]
    s = b2 s + (1-b2) (g - m)^2 + eps;  theta -= alpha mhat / ( sqrt(shat) + eps )
"Belief": m is the prediction of the next gradient; large innovation (g - m) -> distrust -> small
step; in flat directions with consistent small gradients (g ~ m) it takes *large* steps where
Adam takes small ones. Claims: ImageNet ~ SGD accuracy, better GAN stability. Criticism: the
extra "+ eps inside s" is doing real work; the innovation variance s ~ sigma^2 (not sigma^2+mu^2)
so the effective step is m/sigma, unbounded as sigma -> 0 — hence the eps hack. This is the
closest existing thing to an innovation-based Kalman gain.

**LaProp (Ziyin et al. 2020)** [F, 2002.04839]
    n = nu n + (1-nu) g^2;  m = mu m + (1-mu) g / ( sqrt(n/c_n) + eps );  theta -= lambda m / c_m
Bound |m/c_m| <= 1/sqrt(1-nu) vs Adam's 1/(1 - mu/sqrt(nu)) which needs mu < sqrt(nu). Key point:
in Adam a gradient spike inflates v and *divides the whole accumulated momentum* by it
("momentum can immediately vanish"); LaProp normalises each gradient before averaging so a spike
only contributes one bounded term. Interpolates to sign-momentum at nu -> 0 where Adam diverges.
Claims: better on noisy tasks, transformers, RL. Failure: nu still trial-and-error.

**Sophia (Liu et al. 2023)** [A, 2305.14342]: m EMA; h = diag Hessian every k steps (Hutchinson or
Gauss-Newton-Bartlett); theta -= eta clip( m / max(gamma h, eps), 1 ). Claim 2x vs Adam on
GPT-2. Failure: Kaddour et al. 2023 "No train no gain" (NeurIPS) [A]: with matched compute and a
fully decayed lr baseline, Sophia's and Lion's gains vanish on BERT/T5; both highly sensitive to
tuning.

**Lion (Chen et al. 2023)** [A, 2302.06675]: c = b1 m + (1-b1) g; theta -= eta (sign(c) + lambda
theta); m = b2 m + (1-b2) g. State 1. Claim: ViT ImageNet +2%, up to 5x compute saving on JFT.
Failures (stated): gains grow with batch size, worse at small batch; needs 3-10x smaller lr.

**Prodigy (Mishchenko & Defazio 2023) / D-Adaptation** [F, 2306.06101]
    d_{k+1} = max( d_k,  sum_i b2^{(k-i)/2} d_i^2 <g_i, x_0 - x_i>  /  || sum_i b2^{(k-i)/2} d_i^2 g_i ||_1 )
State: d plus two EMAs (r, s). Claim: ~hand-tuned Adam on VGG/ResNet/ViT/RoBERTa/GPT; improves
D-Adaptation by sqrt(log(D/d0)). Failure for streams: estimate is monotone non-decreasing and
built on distance from *initialisation* x_0 — under drift the distance grows without bound and
the lr can only ratchet up.

**Lookahead (Zhang et al. 2019)** [A]: fast weights theta run k steps; slow phi += a (theta_k - phi);
theta <- phi. On the noisy quadratic model the variance fixed point is strictly below the inner
optimizer's at the same lr. **LAWA (Kaddour 2022)** [A, 2209.14981]: average of the last k
end-of-epoch checkpoints; dozens of epochs saved on ImageNet ResNet-50 and RoBERTa. Both are
weight-space low-pass filters; under drift the window k must shrink with drift rate.

**Optimistic / extragradient / Nesterov as gradient prediction** [A, Daskalakis et al. 2018
1711.00141]: OMD x_{t+1} = x_t - 2 eta g_t + eta g_{t-1} = x_t - eta g_t - eta (g_t - g_{t-1}),
i.e. take the step plus a correction that assumes g_{t+1} ~= g_t. Optimistic Adam for GANs.
Nesterov evaluates the gradient at the extrapolated point; extragradient evaluates at a predicted
point and applies at the base. Under i.i.d. gradient noise the predictor 2g_t - g_{t-1} has 5x
the noise variance of g_t, so raw-gradient prediction is *anti*-robust; prediction must be done on
a filtered signal (momentum) — which is exactly Nesterov-momentum, and is why AdEMAMix / Muon use
it rather than OMD. Weight-space forecasters (WNN 2023, NiNo 2025, XGrad, Leap+Verify 2026) are
learned extrapolators, not optimizers, and only validated on stationary training.

**SNR / trust-region readings of Adam (2026)** [F]
* MoLS (2605.05794): Adam's effective signal step is D ~= 1/sqrt(1 + 1/S), S = mu^2/sigma^2; for
  S << 1 it collapses to sqrt(S) (noise damping); embeddings/heads in LLMs sit at S < 0.1. Fix:
  module-level lr scale alpha_m = sqrt(S_base/S_m) estimated once in warmup. 1.06 ppl on LLaMA-1.3B.
* "A trust-region framework for moment estimation" (2608.04026): per-weight constraint
  E|Delta|^p <= mu^p; alpha(t) = delta_p(t)/(kappa_p(t) lambda_2(t)); Adam = the p=2 case with eps
  as the floor; proposes p=4 (kurtosis-aware) normalisation; schedules emerge as trust-region
  radius programs. GPT-2-124M only, preliminary.
* "Why Adam can beat SGD: second-moment normalisation yields sharper tails" (2603.03099) [A]:
  high-probability bound for Adam scales as delta^{-1/2} vs delta^{-1} for SGD under bounded
  variance — normalisation is a tail-control device.
* "Adapt or forget: provable tradeoffs between Adam and SGD in nonstationary optimization"
  (2605.04269) [F]: tracking a drifting minimiser (drift Delta, noise sigma, strong convexity mu).
  SGD floor ~ Delta^2/(mu^2 alpha^2) + d sigma^2 alpha/mu. Adam's floor carries an extra term
  G^4 q+^4 (1 - b2)/(q-^2 mu^2) (q+ = 1/eps) from preconditioner perturbations; noise-dominated ->
  Adam wins (first-moment averaging + preconditioning), drift-dominated -> SGD wins (stale m and
  perturbed v "compound the cost of nonstationarity"). Larger eps damps the second-moment
  variability and stabilises Adam under drift at the price of slower adaptation. This is the
  first theory paper to name the b2 / eps memory as the culprit in streams — consistent with
  "Revisiting Adam for streaming RL" finding eps=0.1.

---------------------------------------------------------------------------------------------------
## Thread 5. Theory of the Adam/SGD generalisation gap and noise geometry
---------------------------------------------------------------------------------------------------

### 5.1 Balles & Hennig 2018, "Dissecting Adam: the sign, magnitude and variance of stochastic gradients" (ICML) [F]
Decomposition: with m, v EMAs, Adam's step per coordinate is
    alpha * m_i / sqrt(v_i)  =  alpha * sign(m_i) * gamma_i,   gamma_i = 1 / sqrt(1 + etahat_i^2),   etahat_i^2 = (v_i - m_i^2)/m_i^2  (relative variance)
So Adam = sign descent x a variance-adaptation factor in [0,1]. Lemma 1 (optimal factors that
minimise E||gamma phat - p||^2 for an unbiased estimate phat with variance sigma^2):
    for the stochastic gradient direction:   gamma_i = p_i^2 / (p_i^2 + sigma_i^2) = 1 / (1 + eta_i^2)
    for the sign direction:                  gamma_i = 2 rho_i - 1 = erf( 1/(sqrt(2) eta_i) ),  rho_i = P[sign(g_i) = sign(grad_i)]
Adam's implicit 1/sqrt(1+eta^2) is a good approximation to erf(1/(sqrt2 eta)) — i.e. Adam is
*near-optimal variance adaptation for sign descent*, and 1/(1+eta^2) is the optimal factor for a
non-sign step. Practical estimate with momentum (bias of the EMA variance, eq. 21-25):
    rho(b,t) = (1-b)(1+b^{t+1}) / ((1+b)(1-b^{t+1}))
    s = (v - m^2) / (1 - rho(b,t));   gamma = m^2 / ( m^2 + rho(b,t) s )
    M-SVAG:  theta -= alpha * gamma o m           (state m, v; cost = Adam)
    M-SSD:   theta -= alpha * sign(m)              (sign momentum, no variance adaptation)
Findings (Fashion-MNIST, CIFAR-10/100 ResNet, War&Peace char-RNN, 10 seeds): (1) methods cluster
by sign vs non-sign; the sign is "by far the dominant" component of Adam; (2) sign is
problem-dependent: better train loss on P1/P3, worse on the language task; (3) variance
adaptation is never worse and often better than its base (M-SVAG > M-SGD on CIFAR-100 and
language); (4) the generalisation harm of Adam on CIFAR-100 is reproduced by M-SSD (sign only)
and *absent* in M-SVAG => "generalization-harming effects of Adam are caused by the sign aspect
rather than the element-wise adaptive step sizes". Why it did not get adopted: gains were modest
and mostly on training loss; a year later transformers made sign-like methods the winner
(Kunstner 2023/2024 explain why: sign descent handles heavy-tailed class imbalance), and M-SVAG
throws away exactly the sign; the (v - m^2) estimate is biased/noisy early and the authors report
that bias corrections "destabilise"; no large-scale follow-up. Nobody re-tested it in RL, where
the sign is *not* known to be the useful component.

### 5.2 Zhou et al. 2020, "Towards theoretically understanding why SGD generalizes better than Adam" (NeurIPS) [F]
Model both as Levy-driven SDEs with symmetric alpha-stable (SalphaS) noise, tail index alpha
(alpha=2 Gaussian). Theorem 1 (escape from basin Omega with escaping set W):
    (1-rho)/(1+u+rho) <= E exp( -u m(W) Theta(eps^-1) Gamma ) <= (1+rho)/(1+u-rho),   Theta(eps^-1) = (2/alpha) eps^alpha
so E[Gamma] = O( 1 / (m(W) Theta(eps^-1)) ) = O( eps^{-alpha} / m(W) ), with eps ~ lr^{(alpha-1)/alpha}
and m(W) a Radon measure of the escaping set. For a quadratic basin, W_SGD = {y : y^T Sigma H Sigma
y >= h} while W_Adam = {y : y^T Sigma Q^{-1} H Q^{-1} Sigma y >= h} with Q the diagonal
preconditioner: Adam's coordinate-wise rescaling "diminishes the anisotropic structure in gradient
noise" and shrinks m(W) => longer escape time => Adam stays in sharper basins. Second mechanism:
Adam's EMA of gradients smooths the noise and *lightens its tails* (larger alpha) and needs a
smaller lr (1e-3 vs 1e-2), both increasing eps^{-alpha}. Empirically SGD's noise has smaller tail
index at some iterations. Message for noisy streams: the anisotropy of the noise is what makes SGD
find flat minima; any per-coordinate whitening destroys it; EMA-smoothing helps stability but
removes the rare large jumps that escape sharp basins.

### 5.3 Xie et al. 2022, "Adaptive inertia: disentangling the effects of adaptive learning rate and momentum" (ICML oral) — Adai [A, 2006.15815]
    v = b2 v + (1-b2) g^2;  b1_t = clip( 1 - b0 * vhat / mean(vhat), 0, 1-eps );  m = b1_t m + (1-b1_t) g;  theta -= eta m
Parameter-wise *momentum* (inertia) instead of parameter-wise lr: coordinates with large gradient
variance get *more* inertia (longer averaging), small-variance ones less. Theory: adaptive lr
escapes saddles fast but cannot select flat minima like SGD; momentum helps saddle escape and
"almost does not affect flat-minima selection". Claim: beats SGD and Adam variants on nets where
flat minima matter. This is the one design that keeps SGD's anisotropic noise (5.2) while still
using v.

### 5.4 Zhang et al. 2020, "Why are adaptive methods good for attention models" (NeurIPS) [A, 1912.03194]
Gradient noise on BERT/transformers is heavy-tailed; SGD can fail to converge, clipped SGD
converges; tight bounds for adaptive methods under heavy tails; ACClip = adaptive coordinate-wise
clipping. Empirically helps BERT.

### 5.5 Kunstner et al. 2023, "Noise is not the main factor behind the gap between SGD and Adam on transformers, but sign descent might be" (ICLR) [F]
Vary batch size up to full batch on PTB/WikiText-2/SQuAD (transformers) and MNIST/CIFAR (CNNs).
Findings: the gap *does not shrink* in full batch (Figure 2, "similar or larger"); Adam improves
with batch size, SGD does not exploit lower noise; in full batch, sign descent with momentum
closes most of the gap, normalised GD does not; at small batch sign descent is the worst.
Explicitly refutes the heavy-tail hypothesis (5.4) as the *main* factor. Open: design sign-like
methods robust to small batches; why transformers specifically (vocabulary size, normalisation).
Limitations: constant lr, short horizons, training loss focus.

### 5.6 Kunstner et al. 2024, "Heavy-tailed class imbalance and why Adam outperforms gradient descent on language models" (NeurIPS) [F]
Mechanism: with Zipfian token frequencies pi_k, the gradient on class-k parameters scales with
pi_k; GD converges at rate ~ 1/(pi_k t) on rare classes; sign descent / Adam normalise the scale
and get e^{-ct} regardless of frequency. Reproduced in full batch (so not noise), across
transformers, ResNets, ViTs, linear softmax models; up-weighting rare classes fixes SGD. Direct
implication: Adam's advantage is an *invariance to per-parameter gradient scale*, and the setting
that needs it is many rarely-active parameters. In MuJoCo RL with dense observations this
structure is largely absent, which is consistent with the folklore that Adam's edge over SGD is
smaller there.

### 5.7 Zhao et al. 2024, "Deconstructing what makes a good optimizer for language models" [F, 2407.07972]
Adam, Lion, Adafactor, Signum (sign momentum with b1=b2) perform comparably in optimum and in
hyperparameter robustness; SGD is clearly worse. Adalayer ablation: adaptivity is only *needed*
on the last layer and LayerNorm parameters; other layers can use fixed init-based lr ratios.
Practical conclusion: choose by memory/implementation, tune lr and momentum.

### 5.8 Other
"Why Adam can beat SGD: second-moment normalisation yields sharper tails" 2026 [A] (see 4);
"Adapt or forget" 2026 [F] (see 4) is the only theory of Adam vs SGD *under drift*.

**Thread-5 synthesis.** Two mechanisms are established with evidence: (i) Adam's sign-like
invariance to per-coordinate gradient scale wins on problems with many rarely-active parameters
(class imbalance), independent of noise; (ii) Adam's coordinate-wise whitening isotropises the
noise and, with EMA tail-lightening, lengthens escape from sharp basins; the sign, not the
variance adaptation, is what harms generalisation. Nothing in this thread says Adam is better
*because* it handles noise; Balles-Hennig say the variance-adaptation part is fine and should be
kept on top of SGD.

---------------------------------------------------------------------------------------------------
## Thread 6. Hypergradient / meta-descent per-parameter learning rates online
---------------------------------------------------------------------------------------------------

**Baydin et al. 2018, "Online learning rate adaptation with hypergradient descent" (ICLR)** [F, ar5iv]
    alpha_t = alpha_{t-1} - beta * g_t . d u_{t-1}/d alpha  = alpha_{t-1} + beta <g_t, g_{t-1}>   (SGD-HD, u = -alpha g)
    Adam-HD: d u/d alpha = -mhat/(sqrt(vhat)+eps);  multiplicative variant divides by ||g_t|| ||d u/d alpha||
State: one extra gradient copy (scalar alpha in the paper; per-parameter is a trivial extension =
IDBD with theta -> beta and h -> g_{t-1}). Claims: removes sensitivity to alpha_0. Limitations
(stated): beta must be tuned; assumes the optimal alpha changes slowly; no stochastic guarantee.
Known behaviour: alpha grows large early then collapses and fluctuates near zero. Chu, Gao, Ye,
Udell 2025 (ICML) [A, 2502.11229] give the first convergence analysis, explain the instability,
add safeguards, show local superlinear convergence and L-BFGS-level performance — on
*deterministic convex* problems; noise not covered.

**MetaOptimize (Sharifnassab & Sutton 2024, ICML 2025)** [F, 2402.02342]
Meta-objective = discounted future loss F_t^gamma = (1-gamma) sum_{tau>t} gamma^{tau-t-1} f_tau.
L-approximation: H_{t+1} = gamma (I - alpha_t Hess f_t) H_t - Y_t alpha_t grad f_t;
beta_{t+1} = beta_t - eta H_t^T grad f_t (beta = log step size; Hessian-free variants drop the
Hessian term). Wraps Adam. Claims: matches well-tuned cosine schedules on CIFAR-10, ImageNet,
TinyStories. Limitations: backward-view approximation degrades with larger eta (like eligibility
traces); blockwise versions gave no gain on ImageNet; designed for slowly changing environments,
rapid shifts unexplored; Hessian terms too noisy to use.

**IDBD lineage** — see 2.5 (IDBD, Autostep), 2.2 (SwiftTD), 2.8 (FADE), TIDBD (Kearney et al.
2018, TD step sizes via stochastic meta-descent) [A]. Robustness to noise: none of these has a
noise model. The meta-gradient delta x h is a product of two noisy quantities; Autostep's
running-max normalisation and SwiftTD's bound + decay are empirical stabilisers. IDBD's
sensitivity to the target scale spans 5 orders of magnitude (Degris 2024). The one principled
noise-aware step-size learner in the literature is the Kalman posterior variance (Thread 1),
which nobody has coupled to a meta-gradient.

---------------------------------------------------------------------------------------------------
## Synthesis
---------------------------------------------------------------------------------------------------

### (a) The ten most relevant rules

| Name | State / param | Update rule (ASCII) | What it does with gradient NOISE | Strongest claim | Known failure |
|---|---|---|---|---|---|
| AdaBayes (Aitchison 2020) | m, v, s2 (3) | s2p = (1-e^2/2S^2)^2 s2 + e^2; 1/s2 = 1/s2p + g^2; mu = (1-l) mu + s2 <g>. FP: 1/s2 = (1/2S^2)(1+sqrt(1+4 S^4 <g^2>/e^2)) | lr_i = posterior variance: -> S^2 (SGD) when g^2 small, -> e/sqrt<g^2> (Adam) when g^2 large; noise g^2 shrinks lr, process noise e^2 keeps it from collapsing | CIFAR-10 ResNet-34 4.84% vs SGD 5.17 / AdamW 5.08; generalises like SGD, starts like Adam | H_ii ~ g^2 crude; assumes constant target-drift e; never tested on RL/transformers |
| IVON / VOGN (Shen 2024, Khan 2018) | g, h (2) + sample | th ~ N(m, 1/(L(h+d))); hh = g (th-m)/s^2; h = b2 h + (1-b2) hh + corr; m -= a (g/(1-b1^t) + d m)/(h+d) | No sqrt: step = grad / curvature; curvature from per-example g^2 (M=1 makes g^2 unbiased, Thm 1); weight noise regularises | GPT-2 773M 12.6 vs 13.0 ppl; ResNet-50 77.46 vs 75.16 | b2 ~ 1-1e-5 required (1e5-step memory); no BatchNorm; h0, lambda sensitive |
| ObGD (Elsayed 2024) | z (1) | z = gl z + grad; M = a k max(|d|,1) ||z||_1; a_eff = a/max(1,M); w += a_eff d z | Nothing explicit: bounds the *effective step* (fraction of error removed) <= 1/k; same step for noisy and reliable d | Streaming AC beats PPO/SAC on DMC Dog; beats PPO1/SAC1 everywhere; alpha=1 works | Bound is on a moving TD target and can miss large changes (Sharifnassab 2026); conflicts with SGD on shared params; beaten by Adaptive Q(l) with eps=0.1 |
| SwiftTD (Javed 2024) | beta, z, p, zbar, h x3, zd (8) | beta_i += e^beta theta (d' - v_d) p_i; tau = sum e^beta phi^2; zd = min(1, eta/tau) e^beta phi; decay beta_i += phi^2 ln(eps) when tau > eta | Meta-gradient raises lr of features whose updates correlate over time, lowers oscillating ones; overshoot bound + decay push chronically-overshooting (unpredictable) features toward lr 0 | Beats True Online TD(l) on every Atari prediction game; robust over 5 orders of magnitude of init lr and meta lr | Linear / last layer only; prediction not control; no noise model |
| M-SVAG (Balles & Hennig 2018) | m, v (2) | s = (v-m^2)/(1-rho); gamma = m^2/(m^2 + rho s); th -= a gamma o m; rho(b,t) = (1-b)(1+b^{t+1})/((1+b)(1-b^{t+1})) | gamma_i = 1/(1+sigma^2/mu^2) = optimal MSE shrinkage of a noisy direction; shortens low-SNR coordinates, keeps SGD's direction/anisotropy | >= M-SGD on all 4 tasks; no generalisation harm (harm isolated to the sign) | Modest gains; loses the sign that wins on transformers; s estimate unstable early |
| AdaBelief (2020) | m, s (2) | s = b2 s + (1-b2)(g-m)^2 + eps; th -= a mhat/(sqrt(shat)+eps) | Innovation-based: step ~ m/sigma_innov; large surprise -> small step; consistent small gradients -> large step | ImageNet ~ SGD; GAN stability | Unbounded as sigma -> 0 (needs eps inside s); innovation variance ignores drift vs noise |
| LaProp (2020) / ADOPT (2024) | m, n (2) | n = nu n + (1-nu) g^2; m = mu m + (1-mu) g/(sqrt(n)+eps) [ADOPT: /sqrt(n_{t-1})]; th -= a m | Normalise each gradient *before* averaging: a spike cannot erase the accumulated momentum; bounded |m| <= 1/sqrt(1-nu) | ADOPT: optimal rate for any b2 w/o bounded noise; LaProp better on noisy/RL tasks | nu still by trial; ADOPT evidence mostly convergence theory |
| Muon + NAMO / PolarGrad (2024-26) | M (1) [+ v scalar] | M = mu M + G; O = NS5(M/||M||_F); W -= lr c O [NAMO: c = ||M||/(sqrt v + eps); PolarGrad: c = ||G||_*] | Polar step whitens noise to unit spectral norm; only safe after momentum filtering (signal gap sqrt(eta)/(2T-1)^{1/4}); scalar/nuclear scale restores update -> 0 as signal -> 0 | ~2x FLOP efficiency vs AdamW (Moonlight); NAMO 3.035 vs Muon 3.044 vs AdamW 3.064 | Gains shrink with batch size (SOAP, Lion, Scion); outlier amplification (ROOT); nothing at batch 1 or RL |
| AdEMAMix (2024) | m1, m2, v (3) | th -= eta (m1hat + alpha m2)/(sqrt vhat + eps), b3 ~ 0.9999, alpha ~ 5-10 | Long window averages noise over 1e4 steps; short window reacts | 1.3B @101B tokens = AdamW @197B | *Explicitly* harms adaptation to distribution shift; early divergence without schedulers |
| Adaptive Q(l) ("Revisiting Adam for streaming RL" 2026) | z, v (2) | z = gl z + g; v = gl v + (1-gl) g^2; w += eta clip(d,-1,1) z/(sqrt v + eps), eps = 0.1 | Huge eps = hard SNR floor: rarely-active/noisy coordinates get ~0 step; trace-decay-matched v | Atari-55 ~3x human vs 2x StreamQ; beats ObGD on MinAtar | Heuristic; eps=0.1 is scale-dependent; Atari only |

Honourable mentions: Adai (per-parameter *momentum* instead of lr keeps noise anisotropy for flat
minima); Schedule-Free (optimal for stationary convex, wrong average for drift); Cautious/Grams
(accidental SNR gates via sign agreement); Continual Backprop / L2-Init / Elephant (non-optimizer
plasticity fixes that any streaming optimizer must be paired with).

### (b) Ideas not yet combined in one optimizer that could plausibly beat Adam on a noisy stream

1. **Kalman posterior-variance step size under an output-space overshoot bound.** AdaBayes gives
   a per-parameter lr that interpolates SGD <-> Adam by the local information g^2 vs process noise
   e^2, but has no safety against a single bad sample; ObGD/SwiftTD/Intentional-TD bound the
   *fraction of error removed* on the current sample but have no noise model (same step for a
   reliable and an unreliable delta). Composition: lr_i from the AdaBayes fixed point, then
   alpha_eff = min(1, eta / sum_i lr_i z_i^2) (SwiftTD's correction ratio, computed on the actual
   trace) as the overshoot clamp. Neither camp cites the other. Cost = Adam + one dot product.

2. **Replace Adam's eps with the Balles-Hennig shrinkage inside a trace-based streaming rule.**
   "Revisiting Adam for streaming RL" shows eps = 0.1 (a crude, scale-dependent SNR floor) beats
   ObGD; M-SVAG's gamma_i = m_i^2/(m_i^2 + rho s_i) is the scale-free, MSE-optimal version of the
   same floor and was shown not to harm generalisation. Nobody has put gamma on the eligibility
   trace (z in place of m, v of delta*z) in RL. It also directly addresses Degris et al.'s
   complaint: gamma is derived from the objective (expected squared error of the step), not a
   normalisation heuristic.

3. **Innovation-driven process noise (adaptive Kalman Q) to auto-tune b2/eps per parameter for
   drift vs noise.** "Adapt or forget" proves the b2/eps memory is what makes Adam lose to SGD
   under drift; AdEMAMix documents the same failure empirically; AdaBelief already computes the
   innovation (g - m)^2 but uses it as a *denominator*, not to estimate how fast the target moves.
   Standard adaptive Kalman filtering estimates the process-noise variance from the innovation
   sequence (innovation variance = measurement noise + propagated process noise); in AdaBayes'
   model that sets e^2 per parameter and therefore the SGD<->Adam interpolation and the effective
   forgetting rate. No deep-learning optimizer estimates its own forgetting rate from innovations.

4. **Denoise-then-orthogonalise the eligibility trace, with nuclear-norm / NAMO scaling and an
   overshoot bound, for actor/critic matrices.** The Muon line shows orthogonalisation is only
   noise-safe after temporal filtering with window T = 1/(1-beta) and only if the step magnitude
   is tied to the signal (PolarGrad's ||G||_*, NAMO's ||M||/sqrt(v)); an eligibility trace
   z = gl z + g *is* a temporal filter with T = 1/(1-gl). Orthogonalising z (rank-r Dion-style
   with error feedback to keep it cheap at batch 1) and clamping with ObGD's ||.||_1 bound has not
   been tried; all Muon-family evidence is at batch >= 2^16 tokens.

5. **Adaptive inertia + normalise-before-average.** Zhou 2020 says per-coordinate lr whitening
   destroys the noise anisotropy that finds flat basins; Adai keeps SGD's step and adapts the
   momentum per coordinate instead; LaProp/ADOPT show that normalising each gradient before
   averaging makes momentum spike-proof. A rule m_i = b1_i m_i + (1-b1_i) clip(g_i / sqrt(n_i),
   c), theta -= eta m, with b1_i from the relative variance (long memory where noisy, short where
   clean) keeps direction = SGD, is spike-proof, and has per-coordinate *time constants* rather
   than per-coordinate *scales* — which is exactly the axis "Adapt or forget" identifies as the
   noise/drift trade-off. Not combined anywhere.

Cross-cutting warning from the evidence: every "beats Adam" claim above that was tested at
matched compute with a decayed-lr baseline (Kaddour 2023) or at small batch (Lion, SOAP, Scion,
Kunstner 2023 on sign descent) shrank or vanished; the only rules with batch-size-1 evidence are
ObGD, SwiftTD, Intentional updates, Adaptive Q(lambda), IDBD/Autostep/FADE, and none of those has
a noise model. The Bayesian-filter rules have the noise model and no streaming evidence.

---------------------------------------------------------------------------------------------------
## Papers touched (62): 42 read in full text / full HTML render [F], 20 at abstract or secondary-summary level [A]
---------------------------------------------------------------------------------------------------
[F] Aitchison 2020; Khan & Rue 2023; Khan et al. 2018 (VOGN/Vadam); Zhang et al. 2018 (Noisy
Adam); Shen et al. 2024 (IVON); Vuckovic 2018 (KGD); Davtyan et al. 2022 (KOALA); Elsayed et al.
2024 (ObGD); Javed et al. 2024 (SwiftTD); Sutton 1992 (IDBD); Mahmood & Sutton 2012 (Autostep);
Degris et al. 2024; Dohare et al. 2024 (Nature, PMC); Lan & Mahmood 2023 (Elephant); Sharifnassab
et al. 2026 (Intentional updates); Revisiting Adam for streaming RL 2026; FADE 2026; Squeezing
the stream 2026; Jordan 2024 (Muon blog); Liu et al. 2025 (Muon scalable); PolarGrad 2025; Scion
2025; SOAP 2024; Dion 2025; NAMO 2026; Denoise-first 2026; Schedule-Free 2024; AdEMAMix 2024;
MARS 2024; LaProp 2020; Grams 2024; Prodigy 2023; Zhou et al. 2020; Kunstner et al. 2023;
Kunstner et al. 2024; Zhao et al. 2024; Balles & Hennig 2018; Adapt or Forget 2026; MoLS 2026;
Trust-region moment framework 2026; Baydin et al. 2018; MetaOptimize 2024/25.
[A] Ollivier 2018; Kumar et al. 2023 (L2 Init); Batch-to-streaming 2026; Bernstein & Newhouse
2024 (Old optimizer new norm); Modular duality 2025; Kaon 2026; ROOT 2025; Cautious 2024; ADOPT
2024; AdaBelief 2020; Sophia 2023; Lion 2023; Lookahead 2019; LAWA 2022; Kaddour et al. 2023 (No
train no gain); Daskalakis et al. 2018 (Optimistic Adam); Xie et al. 2022 (Adai); Zhang et al.
2020 (heavy tails); Sharper tails 2026; Chu et al. 2025 (provable HDM).
