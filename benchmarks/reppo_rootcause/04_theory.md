# 04 — Theory: what an on-policy action-value critic can know, and what a pathwise actor gets from it

Role: first-principles analysis (statistics, identifiability, gradient dynamics). Derivations are mine; the
measured facts are taken from `cleanrl/reppo/FAMILY.md`, `OPTIMUM.md` and the task brief and are treated as
approximately right. Code read: `cleanrl/reppo/ppo_continuous_action_reppo_noisefork_v60.py` (critic, actor,
target refresh), headers of v26 / v51 / v63.

Status tags used throughout: **[P]** proven (algebra, exact under the stated assumptions), **[N]** checked
numerically in a toy here, **[H]** heuristic / order-of-magnitude, **[U]** unsure, flagged.

Toy scripts (CPU, numpy/torch, <= 3 threads), outputs saved next to them:

| script | output | what it checks |
|---|---|---|
| `scripts/04_toy_equiv.py` | `data/04_toy_equiv.txt` | SF gradient == pathwise-through-regression-critic; SNR law; per-state slope formulas |
| `scripts/04_toy_leak.py` | `data/04_toy_leak.txt` | collinearity leak in an exact LQ problem, MLP critic in (s,a) vs orthogonal (s,eps) coordinates |
| `scripts/04_toy_hlgauss.py` | `data/04_toy_hlgauss.txt` | HL-Gauss aliasing, CE geometry, CE vs MSE on a 1%-variance contrast |
| `scripts/04_toy_chaos.py` | `data/04_toy_chaos.txt` | exact Q^pi on a chaotic map: roughness, first- vs zeroth-order variance, slope horizon |

Notation. a = mu(s) + sigma eps, eps ~ N(0, I_d), d = 6. V = E_eps Q. g(s) = d/dm E_eps Q(s, m + sigma eps) at
m = mu(s) (the sigma-smoothed slope). sigma_A = within-state sd of Q (0.22). nu = sd of everything in a
regression residual y - V_hat(s) that is not the true action effect. N = 16384. All numbers in scaled-reward
units unless a return is quoted.

---

## 0. Summary of conclusions

1. **Identification [P].** Under on-policy sampling the population squared loss separates exactly into a
   level term and Hermite-weighted action terms: E(Q_hat - y)^2 = E_s (m_hat - V)^2 + sigma^2 E_s |g_hat - g|^2
   + (higher orders) + noise. So the smoothed slope IS identified, with weight sigma_A^2 (1% of target variance,
   ~5% of the residual). Everything that goes wrong is finite-sample or optimisation, not population bias.
2. **Information bound [P, N].** The leak-free estimate of that slope is the regression of the residual on eps,
   and that is the score-function (SF) gradient: with compatible features the pathwise gradient through the
   regression critic equals the SF gradient sample by sample (cos 0.998 at N = 16384 in the toy). A fully
   on-policy pathwise method has **no within-state information PPO lacks**. Any gain must come from (a) a
   better baseline/bootstrapping (at most 1/(1-(gamma lambda)^2), 8.7x in samples vs GAE 0.95 and only if V is
   accurate at the scale of an action's effect), (b) cross-iteration reuse of a slowly-moving slope, (c) valid
   structural assumptions in (s, a). REPPO as built harvests none of the three.
3. **The oracle numbers are what an un-pooled, single-sample slope gives [P, N for the formula; H for the
   match].** A critic that interpolates one residual per state with a minimum-norm slope has fresh-action
   contrast corr = sigma_A / (d sqrt(nu^2/(d-2) + sigma_A^2/d)) = 0.117 at nu = 0.6 (oracle: 0.11) and inflated
   contrast sd 0.31 (oracle 0.21 = truth-sized although 99% noise). Inverting the pooled formula, the critic's
   slope at a state is worth **K_eff ~ 0.2-1.2 on-policy samples**, where the batch could support K ~ 50-100.
4. **The leak is real, has a derivable sign, and is a transient whose time constant scales as
   Var_s(mu)/sigma^2 [P for the linear case, N for an MLP].** For an under-gained feedback policy the association
   slope is opposite to the causal slope (Simpson's paradox: effort marks bad states). In the exact LQ toy the
   (s,a) MLP gives contrast corr -0.24, true-on-critic slope -0.31, self-consistency -0.14 with truth-sized
   contrast sd after 16 passes; at sd(mu)/sigma = 6.4 it is still -0.34 / -0.29 after 64 passes of fresh data and
   only reaches +0.10 after 16 batches. It reproduces the measured law "critic ranking improves with behaviour
   spread" and predicts the late wrong-sign regime. The orthogonal parameterisation is immune in the toy.
5. **HL-Gauss is not a resolution floor [N].** Decode(project(y)) aliasing is 3e-5; the CE is exactly quadratic
   in the predicted mean with curvature 1/s^2. It is a ~2-3x rate handicap on a 1%-variance component (toy), not
   the reason the slope is wrong.
6. **v61/v62 did not remove target noise [P].** The Q(lambda) correction subtracts the critic's own contrast,
   which has corr 0.11 with the truth and the same sd: Var(A - A_hat) = 1.7 sigma_A^2. It is exact only at the
   fixed point. Its neutrality says nothing about whether target noise matters.
7. **Step length [H, but the numbers close].** From the measured noise cost and slope-at-0, a quadratic model
   predicts the true optimum along the update at 7.1x the taken step (measured 5.4-7.8x). The step is limited by
   the sigma-denominated KL gate, not by curvature. Separately, the step realises only ~6% of the first-order
   improvement available at its own KL radius: direction quality is the larger loss (~15x) and length the
   smaller (~3.5x).
8. **Optimum.** The statistically right on-policy estimator is a hybrid: SF/GAE actor gradient (leak-free by
   construction), control-variated by a critic whose slope is estimated orthogonally, pooled across states to
   K >= 50 per degree of freedom, Kalman-averaged across ~10-30 iterations, with a longer horizon for the slope
   than for the level, and a step set by evidence rather than by sigma. Expected gain over PPO: 1.5-3x in
   sample efficiency, not the 4x@1M of TD7; that gap needs replay, whose real benefit is breaking the
   a = mu(s) collinearity, not sample count.

---

## 1. Identifiability

### 1.1 What the data identifies [P]

Fix s and expand any critic along the behaviour noise in the Hermite basis of eps:

    Q_hat(s, mu(s) + sigma eps) = m_hat(s) + sigma g_hat(s).eps + (sigma^2/2) sum_ij C_hat_ij(s) He2_ij(eps) + ...

and the same for the truth (m = V, g, C). Hermite polynomials are orthogonal under N(0, I), so with
y = Q + xi, E[xi | s, a] = 0:

    E (Q_hat - y)^2 = E_s (m_hat - V)^2 + sigma^2 E_s |g_hat - g|^2 + (sigma^4/2) E_s ||C_hat - C||_F^2 + ... + E xi^2

Consequences:

- The sigma-smoothed slope g(s) and curvature C(s) are identified; nothing finer than scale sigma is, and nothing
  outside ~2 sigma of mu(s) is.
- Level error and slope error do not interact in function space. A regressor that could set m_hat(s) and
  g_hat(s) independently would have an unbiased slope whatever its level error. So "state error leaks into
  the action slope" is never a property of the estimand; it needs (i) parameter coupling between m_hat and g_hat
  or (ii) finite samples.
- The slope's weight in the loss is sigma^2 |g|^2 = sigma_A^2 = 0.048 against a level term of 0.81 still
  unexplained (level error sd 0.90) and a target-noise term. Whenever a parameter direction lowers the level term
  at the price of slope error, the optimiser takes it up to a 17:1 exchange rate.

### 1.2 The leak: collinearity plus finite optimisation time [P linear, N nonlinear]

With (s, a) as inputs, a parameter that changes the response to `a` also changes the level on the data
manifold, because a = mu(s) + sigma eps is mostly a function of s. Linear case, features [phi(s), a]:
by Frisch-Waugh the OLS action coefficient is

    b = Cov(a_perp, y) / Var(a_perp),   a_perp = a - Proj_phi(a) = sigma eps + mu_perp(s)
    b = g + Cov(mu_perp, V_perp) / (sigma^2 + Var(mu_perp))            (scalar form)

where mu_perp, V_perp are the parts of mu(s), V(s) the state features have not explained. The second term is the
leak: an association across states between "what the policy does here" and "how good here is", divided by a
denominator that is only sigma^2 when the state part is perfectly partialled out. It is zero iff mu_perp = 0
(the state model fully explains the policy mean) or V_perp is uncorrelated with it.

A flexible net can represent mu(s) exactly, so at the global optimum the leak vanishes. The issue is the
path. Take two features x and a = k x + sigma eps, target y = v x + g sigma eps, gradient flow from zero.
The feature covariance has eigenvalues lambda_1 ~ 1 + k^2 (direction (1, k): "x and a together") and
lambda_2 ~ sigma^2 / (1 + k^2) (the only direction that distinguishes a from x). Gradient flow fits lambda_1
first, giving

    w_a(early) = k v / (1 + k^2)      (association)        w_a(t >> 1/lambda_2) = g      (causal)

`04_toy_leak.py` part A [N]: v = 2.3, g = 0.44. k = 1, sigma = 0.5: w_a = 0.95, 0.67, 0.44 at t = 1, 10, 100.
k = 1, sigma = 0.2: 0.99, 1.03, 0.54, 0.44 at t = 1, 10, 100, 1000. k = 2, sigma = 0.2: 0.91, 0.88, 0.66, 0.44.
The time to replace association by causation is 1/lambda_2 = (1 + k^2)/sigma^2 = (Var_s(mu) + sigma^2)/sigma^2
in units of the level's fitting time. That ratio is ~5 early in REPPO (between-state sd of the tanh action
~0.5-0.7, within-state ~0.3) and 25-100 late (sigma down, means saturating). **[H]** for the real numbers.

Also note the size: the association slope is O(v k/(1+k^2)) ~ 1 per unit action, against a true |g| ~ 0.44
spread over 6 dimensions. Even a few-percent residue of it is comparable to the signal.

### 1.3 The sign: Simpson's paradox for feedback controllers [P in LQ, N]

The leak is not sign-neutral noise. In a scalar regulator x' = x + a, a = -k x + sigma eps, V = -p x^2,
the true slope at the mean is g = -2 p gamma (1 - k) x: for an under-gained policy (k below the optimum) the
truth says "push harder", g is parallel to mu(s). The association says the opposite: |a| is large exactly
where |x| is large, i.e. where V is low. A critic that represents -p x^2 partly as -c a^2 (the two are
collinear on the manifold) has dQ_hat/da = -2 c a, antiparallel to g. Its own one-step backup
r + gamma V_hat(x + a) is right, because V_hat(x') is evaluated on the manifold where the level is fit. So:

- critic slope vs truth: wrong sign;
- critic slope vs its own backup: negative correlation;
- critic contrast sd: of the right order, because the association is strong.

Over-gained policies (k above optimum) flip the relation: the leak is then aligned with the truth and looks
like a good critic. Policies that are still improving by doing more of what they do are in the first class.

Exact check, `04_toy_leak.py` part B [N]: 17-dim linear system, 6 actions, quadratic+linear reward,
gain = kappa x LQR gain, closed-form Q^pi (discounted Lyapunov). MLP 2x256 SiLU on (s, a); REPPO's fitting
budget (16 passes of 2048 on each fresh 16384 on-policy batch, warm-started, policy frozen = best case);
between-state sd scaled to 2.3. "Orthogonal" is V(s) + g(s).eps with separate nets (v63's linear part) on the
same data. Held-out states, 8 fresh actions each.

| regime | batches | param. | level corr | contrast sd true / hat | contrast corr | true-on-critic slope | self-consistency |
|---|---|---|---|---|---|---|---|
| kappa 0.5, sigma 0.5 (sd(mu)/sigma 2.0), no noise | 1 | Q(s,a) | 0.933 | 0.29 / 0.31 | **-0.24** | **-0.31** | **-0.14** |
| | 1 | V+g.eps | 0.951 | 0.29 / 0.22 | +0.82 | +1.06 | +0.76 |
| | 4 | Q(s,a) | 0.998 | 0.30 / 0.24 | +0.74 | +0.91 | +0.71 |
| | 16 | Q(s,a) | 1.000 | 0.30 / 0.29 | +0.99 | +1.00 | +0.95 |
| | 16 | V+g.eps | 0.999 | 0.30 / 0.27 | +0.92 | +1.00 | +0.89 |
| same + noise 0.6 + rough state term 0.9 | 1 | Q(s,a) | 0.892 | 0.29 / 0.33 | -0.28 | -0.31 | -0.11 |
| | 4 | Q(s,a) | 0.971 | 0.30 / 0.24 | +0.54 | +0.65 | +0.50 |
| | 4 | V+g.eps | 0.967 | 0.30 / 0.29 | +0.90 | +0.92 | +0.81 |
| | 16 | Q(s,a) | 0.986 | 0.30 / 0.28 | +0.92 | +0.97 | +0.80 |
| | 16 | V+g.eps | 0.989 | 0.30 / 0.28 | +0.90 | +0.96 | +0.79 |
| kappa 1.5 (over-gained), same noise | 1 | Q(s,a) | 0.871 | 0.36 / 0.17 | +0.69 | +1.46 | +0.69 |
| kappa 0.5, **sigma 0.15** (sd(mu)/sigma 6.4), same noise | 1 | Q(s,a) | 0.897 | 0.082 / 0.141 | **-0.53** | -0.32 | **-0.40** |
| | 4 | Q(s,a) | 0.974 | 0.084 / 0.134 | **-0.34** | -0.22 | **-0.29** |
| | 4 | V+g.eps | 0.968 | 0.085 / 0.102 | +0.83 | +0.68 | +0.71 |
| | 8 | Q(s,a) | 0.975 | 0.083 / 0.110 | -0.19 | -0.15 | -0.15 |
| | 16 | Q(s,a) | 0.988 | 0.084 / 0.084 | +0.10 | +0.12 | +0.11 |
| | 16 | V+g.eps | 0.988 | 0.086 / 0.097 | +0.80 | +0.71 | +0.71 |
| sigma 0.15 + level random-walk 0.3/iter | 4 | Q(s,a) | 0.970 | 0.084 / 0.135 | -0.35 | -0.23 | -0.33 |
| | 8 | Q(s,a) | 0.963 | 0.083 / 0.111 | -0.22 | -0.17 | -0.23 |
| | 16 | Q(s,a) | 0.978 | 0.084 / 0.090 | +0.03 | +0.05 | +0.15 |
| | 16 | V+g.eps | 0.979 | 0.086 / 0.097 | +0.79 | +0.70 | +0.68 |

(Full table, including the level-drift 1.0 case, in `data/04_toy_leak.txt`.)

Reading:

- **Can the mechanism produce a negative self-consistency correlation and a small or wrong-sign real slope with
  a truth-sized contrast sd? Yes [N]**, with no target noise at all (row 1), and at a level corr (0.93) equal to
  the oracle's. The toy's self-consistency at sd(mu)/sigma = 6.4 after 4 batches is -0.29 (oracle: -0.33).
- It is a **transient**. With a stationary policy and sd(mu)/sigma = 2 the (s,a) critic recovers in 4-16 batches
  and then is *better* than the orthogonal one in the noiseless case (0.99 vs 0.92), because Q is globally simple
  in (s, a) here. At sd(mu)/sigma = 6.4 it is still at +0.10 after 16 batches (260k samples of a
  frozen policy) while the orthogonal form sits at +0.80 from batch 4 on.
- Its strength rises as sigma falls relative to the spread of mu. That is the repo's measured law (critic ranking
  corr 0 at std 0.1, 0.1 at 0.3, 0.29 at 0.45) and it predicts the late regime (true slope <= 0, critic
  predicts gain).
- **What the toy does not reproduce [U]:** at sd(mu)/sigma ~ 2 (REPPO at 1M) the toy's leak is gone within ~10
  batches, yet the oracle at 1M shows corr 0.11. Either REPPO's slope field moves or is complex enough that the
  transient never ends, or the 1M number is mostly the variance mechanism of section 2.4. I think the second is
  the larger share at 1M and the leak the larger share late; the placebo experiment (E2) separates them.
- An action-independent drifting level (random walk in function space, 0.3 or 1.0 per iteration) slows the
  recovery only mildly. A drift whose association with mu is *systematic* (the real case: V rises where the
  policy has just changed) would re-inject the leak each iteration; I did not model that. **[U]**

### 1.4 The estimator that removes the leak, and why v63 could still fail

Removing the leak means partialling the state out of both sides (Robinson's partially linear model / double ML):
regress y - V_hat(s) on a - mu(s) = sigma eps. Because eps is independent of s by construction,
E[eps (y - V_hat) | s] = sigma g(s) for ANY V_hat: level error costs variance, never bias **[P]**. That is v63's
design, and it is also exactly the score-function estimator (section 2.1). Hence:

**The leak-free critic slope carries precisely the information PPO already uses.** v63 = PPO's gradient, routed
through an extra regression stage, then through REPPO's actor machinery (per-state KL gate, entropy dual) that
was shaped around a different signal. By the data-processing inequality it cannot beat PPO per sample; it can
only match it if the intermediate slope net pools as well as the policy net does. So v63 was right about the
bias and offered no source of gain. That it regressed against the leaky base is the informative part:

1. *It discarded absolute action coordinates.* Q(s, a) = r(s, a) + gamma V^pi(f(s, a)): r and f do not depend
   on the policy, so in absolute coordinates the action dependence is a slowly-moving object that data from
   earlier iterations and from neighbouring states with different mu still constrain. A(s, z) in behaviour-noise
   coordinates is re-centred every iteration and is purely local (slope + diagonal curvature). In the noiseless
   toy the (s,a) critic ends at 0.99 vs 0.92 for the same reason.
2. *It discarded the association itself, which may be a useful heuristic early* **[H, U]**. Climbing the
   association slope means "everywhere, act more like you act in high-value states". That is not a policy
   gradient, but for a gait where good behaviour is nearly state-independent in kind (large, well-phased torques)
   it correlates with improvement, in the way return-conditioned imitation does. It would help early and become
   the bias floor late. This is the only reading under which the leaky critic beats an unbiased one of equal
   variance. Test: E2.
3. *Its variance was never controlled.* With one sample per state the orthogonal slope has the single-sample SNR
   of section 2.3. A 256-wide A-net trained for 16 passes on residuals that are > 90% noise memorises them
   (v51 measured exactly this for a contrast head: 0.25 on training rows, 0.08 fresh). "Exact targets" could not
   rescue it for the reason in 2.5. A's own gradient accuracy was not measured; until it is, v63 refutes nothing
   about the leak.

**Is cross-state generalisation of the slope the source of any pathwise advantage?** On-policy: yes, it is the
only one besides bootstrapping and cross-iteration memory, because the within-state information is the SF
statistic. And generalisation in (s, a) is exactly where the leak lives: "identification by smoothness across
states with different mu" is valid only where Q varies more slowly in s than mu does, and is confounding
otherwise. The two cannot be separated by architecture alone; they are separated by data in which a varies at
fixed s by more than sigma, i.e. replay over many policies.

---

## 2. Information bound: score function vs pathwise through a regression critic

### 2.1 Equivalence in the linear-in-noise limit [P, N]

Policy mean mu_theta(s) with Jacobian J(s). SF/GAE gradient with baseline b:

    G_sf = (1/N) sum_i J_i^T eps_i (y_i - b_i) / sigma,       E G_sf = E_s J^T g(s)          (Stein's lemma)

Regression critic Q_hat = V_hat(s) + (a - mu(s))^T W psi(s) (compatible when mu = W_pi psi). Normal equations
for W, with V_hat partialled out:

    sum_i (psi_i psi_i^T (x) sigma^2 eps_i eps_i^T) vec(W) = sum_i sigma eps_i psi_i^T (y_i - V_hat_i)

As eps eps^T averages to I: W Sigma_psi = (1/N) sum_i eps_i psi_i^T (y_i - V_hat_i)/sigma = G_sf, and the
pathwise gradient through the critic is (1/N) sum_i (dQ_hat/da)_i psi_i^T = W Sigma_psi. So

    pathwise-through-compatible-critic  ==  G_sf  (same sufficient statistic);   W itself  ==  natural gradient.

This is the compatible-function-approximation identity, rederived. Per state with K samples the OLS slope is
(sum eps eps^T)^-1 sum eps (y - V_hat)/sigma -> the SF average; with K < d it is the minimum-norm
interpolant eps (y - V_hat)/(sigma |eps|^2), the SF sample rescaled.

`04_toy_equiv.py` [N] (d = 6, 24 features, level sd 2.3, sigma_A 0.22): cos(pathwise, SF) = 0.965, 0.991, 0.998,
0.999 at N = 1024, 4096, 16384, 65536; with target noise 0.66 both have the same cos with the true gradient
(0.762/0.756, 0.928/0.936, 0.980/0.983, 0.995/0.995), matching |G|^2/(|G|^2 + tr Sigma/N) (0.780, 0.928, 0.981,
0.995).

Drawing many reparameterised actions per state in the actor changes nothing: E_eps grad_a Q_hat(s, mu + sigma
eps) = E_eps[Q_hat eps]/sigma (Stein). It integrates the critic exactly; it adds no information about Q.

### 2.2 So when can on-policy pathwise beat PPO at all?

Only by using something the SF estimator on one batch does not:

| source | what it needs from the critic | bound on the gain | harvested by REPPO v60? |
|---|---|---|---|
| (a) bootstrapped credit: replace the noisy continuation by V(s') | V accurate *at the scale of an action's effect on s'* (level error difference between the two next states << sigma_A) | variance factor 1/(1 - (gamma lambda)^2): 8.7x vs GAE 0.95, 2.7x vs lambda 0.8; zero if V is not that accurate | No. lambda = 0.8 assumes it; the oracle says V's state gradient is not accurate enough (one-step route no better) |
| (b) reuse across iterations: Q's action slope moves slowly (r, f fixed; curvature term ~1-4%/iter, section 5) | slope stored in policy-independent coordinates and not rewritten by each fit | up to W ~ 10-30 iterations of data for the slope | No. v49 measured that one iteration rewrites the contrasts by as much as their own sd |
| (c) smoothness across states (pool the slope field) | a slope field with few degrees of freedom, estimated orthogonally | from K_eff ~ 1 to K ~ 50-100 per dof: corr 0.11 -> 0.6-0.8 if the field is that simple | No. Per-sample memorisation (2.4). PPO does this pooling in the policy net |
| (d) many action samples per state | nothing | none (2.1) | n/a |
| (e) larger valid step because Q models a neighbourhood | curvature (He2 coefficients), identified with weight sigma^4 ||C||^2, far below the slope | none on-policy; the identified region is ~2 sigma for both | No, and PPO has the same radius |
| (f) the dimension factor (a true gradient gives d numbers per state, a return gives one projection) | a slope learned from more than projections: a dynamics model composed with grad V | (d + 1) = 7 in per-state variance, but only if grad V along the action-reachable directions is known from elsewhere | No. On-policy, the spread of s' along those directions is itself made by sigma eps, so regressing V on it is the SF estimator again |

(f) deserves one more sentence because it is the usual argument for pathwise methods. A model with R^2 0.996
gives df/da. The slope is then r_a + gamma B(s)^T grad V(s'). But grad V along B is learned from how values
differ between states that differ along B, and on-policy those differences are produced by past action noise.
It is the same eps, one step shifted. The model contributes nothing that the Markov structure in (a) did not.
That is the first-principles reason the measured "model R^2 0.996, advantage ranking no better" is expected
rather than surprising **[P for the information statement, H for its completeness]**.

### 2.3 SNR formulas [P, N]

Per sample: g_hat = eps (y - b)/sigma, E g_hat = g, Cov = [(nu^2 + sigma_A^2) I + sigma^2 g g^T]/sigma^2, where
sigma_A^2 = sigma^2 |g|^2 in the linear regime. The SNR does not depend on sigma separately, only through
sigma_A = sigma |g| and nu. Pooling K samples at (effectively) one state and predicting the contrast of a fresh
action:

    corr^2(K) = sigma_A^2 / (sigma_A^2 + ((d + 1) sigma_A^2 + d nu^2) / K)
    K_half    = (d + 1) + d nu^2 / sigma_A^2            (samples for corr^2 = 1/2)

Checked by Monte Carlo to 3 decimals (`data/04_toy_equiv.txt`, e.g. nu 0.6: K = 1, 6, 24, 96 -> 0.140, 0.326,
0.562, 0.808 against 0.138, 0.323, 0.563, 0.806).

Minimum-norm single-sample interpolation (what a net that fits each residual with the smallest action slope
does):

    corr = sigma_A / (d sqrt(nu^2/(d-2) + sigma_A^2/d)),     sd(contrast_hat) = sqrt(nu^2/(d-2) + sigma_A^2/d)

MC: nu 0.3, 0.6, 0.9 -> 0.202, 0.107, 0.089 (formula 0.210, 0.117, 0.080); sd 0.17, 0.32, 0.46.

The residual noise for a lambda-return. y_t - V_hat(s_t) = sum_k (gamma lambda)^k delta_{t+k}, delta = A_true +
(gamma e_{t+1} - e_t), e = level error. Hence

    nu^2(lambda) = sigma_A^2 [1/(1 - (gamma lambda)^2) - 1]  +  Var( -e_t + gamma (1 - lambda) sum_k (gamma lambda)^k e_{t+k+1} )

First term (future sampled actions; effective horizon 1/(1 - gamma lambda)): 1.68 sigma_A^2 at lambda 0.8,
7.66 sigma_A^2 at 0.95, which is the measured "~90% of a lambda-return advantage's variance" (7.66/8.66 = 88%).
Second term (critic level error): vanishes if e is constant over the 1/(1 - gamma lambda) = 4.8-step window and
tends to Var(e) = 0.81 = 17 sigma_A^2 if e decorrelates within it. Its autocorrelation time is unmeasured **[U]**.
The repo's v51 number (contrast head reaches corr 0.25 with its own training targets, "about the per-sample
ceiling ~0.3") implies nu^2 ~ 10-15 sigma_A^2, i.e. nu ~ 0.7-0.85: the level-error term is not small.

| regime | nu^2/sigma_A^2 | single-sample target corr | K_half | cells N/K_half |
|---|---|---|---|---|
| exact V, one step (ideal bootstrapping) | 0 | 1.00 (still one projection) | 7 | 2340 |
| lambda 0.8, perfect level | 1.68 | 0.61 | 17 | 960 |
| GAE 0.95, perfect level | 7.66 | 0.34 | 53 | 310 |
| what v51 implies for REPPO | ~12-15 | 0.25-0.28 | 80-97 | 170-205 |

"Cells" = how many independent slope vectors the batch can pin to corr^2 = 1/2. So one rollout supports a slope
field with a few hundred degrees of freedom (x d), at best ~2300 with a perfect V. Global direction quality for
any estimator using these statistics: cos^2 = |G|^2 / (|G|^2 + tr Sigma / N).

### 2.4 Reading the oracle through these formulas [H]

Inverting corr(K) at the measured fresh-action corr 0.11: K_eff = ((d+1) + d nu^2/sigma_A^2) c^2/(1 - c^2) =
0.21 (nu^2 = 1.68 sigma_A^2), 0.66 (7.66), 1.2 (15). **The critic's slope at a state is worth about one on-policy
sample or less.** The minimum-norm formula gives 0.117 at nu = 0.6 with contrast sd 0.31; the oracle has 0.11
and 0.21 (a mildly shrunk version). And the repo's own measurements say the same thing three ways: training-row
corr 0.25 vs fresh 0.08 (v51), contrasts rewritten by their own sd every iteration (v49), contrast sd equal
to the truth's while 99% of its variance is unrelated to it (oracle).

So the picture I can defend is: **the critic memorises one noisy projection per state, does not pool the slope
across states, does not carry it across iterations; the actor then pools across states exactly as PPO's does.**
End to end that is PPO's information passed through a lossy and, late, biased channel. It explains why PPO
baselines sit in the same range or above without any appeal to bins or architecture. Caveat: the match to 0.11
uses a nu I inferred, not measured; E1 measures it.

### 2.5 Why "exact targets" (v61/v62) were not exact [P]

The corrected target G_t = r_t + gamma [V'_{t+1} + lambda (G_{t+1} - Q'(s_{t+1}, a_{t+1}))] removes the future
action's advantage by subtracting the critic's estimate of it. With corr rho = 0.11 between critic and true
contrasts and equal sds, each future step contributes Var(A - A_hat) = sigma_A^2 + sigma_Ahat^2 - 2 rho
sigma_A sigma_Ahat = 0.048 + 0.044 - 0.010 = 0.082 = 1.7 sigma_A^2: **more** noise than before, not less. The
correction is exact only at its fixed point; the optimal control-variate coefficient would be rho
sigma_A/sigma_Ahat ~ 0.11 and would remove 1.2% of that variance. v61/v62 therefore test "add ~70% to one
component of nu^2"; against a level-error-dominated nu^2 ~ 12-15 sigma_A^2 that is a < 10% change, consistent
with the null result. The conclusion drawn in FAMILY.md ("target variance is not REPPO's bottleneck") does not
follow; the experiment could not have shown otherwise.

---

## 3. Scale and loss geometry

### 3.1 HL-Gauss cross-entropy [P, N]

Labels are a Gaussian of sd s_l = 1.5 integrated over bins of width 2 (ratio 0.75). For a Gaussian-shaped
prediction with mean m and sd s_p, CE = log s_p + (s_l^2 + (m - y)^2)/(2 s_p^2), so dCE/dm = (m - y)/s_p^2:
linear in the error with no dead zone at sub-bin scale. `04_toy_hlgauss.py`: aliasing of
decode(project(y)) is 3.0e-5 max over y in [99, 101]; dCE/dm matches (m - y)/s^2 to 3 significant digits from
0.01 to 6 units for s_p = 1.5 and 2.5. The mean of a 151-bin softmax is a continuous function of the logits; bin
width is not a resolution limit for the mean. "Contrasts are 0.1 of a bin" is true and irrelevant.

What the CE does change:

- The effective MSE weight is 1/s_p^2 with s_p >= 1.5 (the probe found 1.5 plus 0.9 excess, s_p ~ 1.75): a
  constant, irrelevant under Adam and norm clipping.
- For a free softmax, large errors saturate (|p - label| is bounded): CE down-weights outliers. Mildly good for
  a contrast buried in heavy-tailed level noise.
- Moving the mean by delta requires a logit perturbation delta (c_i - m)/s_p^2 across ~6 active bins, a direction
  that depends on the level m. The head has to learn the action effect as a level-dependent logit pattern
  instead of one scalar. This is a conditioning cost, and it shows in the toy [N]: same trunk and data, inputs
  (s, eps) so the leak is absent, level sd 2.3 around 100, contrast sd 0.22, noise 0.6:

  | loss | 1 batch | 3 batches | 6 batches |
  |---|---|---|---|
  | HL-Gauss CE | corr 0.26, sd ratio 0.75 | 0.65, 0.80 | 0.86, 0.91 |
  | MSE (standardised) | corr 0.67, sd ratio 0.67 | 0.88, 0.90 | 0.87, 1.01 |

  CE reaches the same place ~3x later. In a tracking problem a 3x slower slope is a real handicap, but it is a
  rate, not a floor, and both end identically. Consistent with scalar-vs-distributional being roughly neutral
  in the repo.

### 3.2 Why a 1%-variance component is learned late [P linear, H nets]

Three multiplicative penalties on the slope modes, all independent of the loss:

1. Gradient share. The per-sample gradient is residual x d(output)/d(theta); the slope signal is
   sigma_A^2/(sigma_A^2 + nu^2 + e^2) ~ 3-7% of the residual's variance. The rest is noise for the slope
   parameters.
2. Kernel eigenvalue. The function g(s).(a - mu(s)) is a product of a state feature and the *difference of two
   nearly collinear inputs*. Its eigenvalue under the data distribution is ~ sigma^2/(Var(mu) + sigma^2) of a
   plain state mode (part A above). Gradient descent learns modes in order of eigenvalue; early stopping (16
   passes, then the target moves) is a low-pass filter that removes exactly these.
3. Input scale. tanh actions enter with within-state sd ~0.3 early and ~0.1 late next to 17 unit-variance
   observations (RMSNorm after the first linear layer does not change the ratio).

Jointly these say: the fit converges on the level in a few passes and on the slope in tens to hundreds, while
in between the action coefficient holds the association (1.2).

### 3.3 Reparameterisations, and why the attempts so far did not work

The requirement set is: (i) orthogonality (slope regressed on eps, or residual-on-residual), (ii) its own scale
(target standardised by an estimate of sigma_A, own head, own optimiser), (iii) variance control (pooling,
cross-iteration averaging, capacity), (iv) policy-independent coordinates for memory. Each attempt satisfied
one or two:

| attempt | (i) orthogonal | (ii) own scale | (iii) variance | (iv) memory | my reading |
|---|---|---|---|---|---|
| v26 dueling Q = V + A(s,a) - E A | no: A regresses G - V(s) on (s, a); its target is > 90% state residual and its input contains the proxy for s. Centring removes the mean of A over actions, not the leaked slope | yes | no | yes | wrong in detail: it gave the leak its own network |
| v51 contrast head on (G - E Q')/s_y | no (inputs (s, a)) | yes, standardised | no (train 0.25 / fresh 0.08 = memorisation) | yes | right about scale: the only one with a gain (8343 vs 6757 at 3M, same code path), consistent with the ~3x rate effect in 3.1 |
| v63 V(s) + A(s, z) | yes | partly (MSE on raw residuals, own optimiser) | no | no (local, re-centred each iteration) | right about bias, wrong about where the remaining information is (1.4) |
| v61/62 exact targets | n/a | n/a | intended, but increases noise (2.5) | n/a | not a test of target noise |
| v50 anchor to previous contrasts | no | n/a | yes | yes | averaging a biased, compressed estimate compounds the compression, as observed |
| v59 scalar head | no | no | no | yes | removes a handicap that is not the bottleneck |

None was wrong in principle. None combined (i) with (iii). Given 2.1, a variant that has (i) and nothing else
is PPO with extra steps, and a variant without (i) keeps the association.

Other reparameterisations asked about:

- *Advantage standardisation / residual Q = V + sigma_A A_hat*: fixes (ii), a rate gain of the order seen in 3.1.
- *Learning dQ/da directly (gradient matching)*: there is no gradient target on-policy other than the SF sample
  eps (y - V)/sigma, so "regress g_hat(s) on eps (y - V)/sigma" is the orthogonal estimator again, with noise
  ((d + 1) sigma_A^2 + d nu^2)/sigma^2 per sample. It is the cleanest way to write it and makes (iii) explicit
  (it is a d-vector regression with known noise level, so shrinkage can be set from K_half).
- *Difference loss between two actions at the same state*: needs two actions per state. The simulator cannot be
  reset, but the same information is available from v60's fork harness offline; on-line it would require
  antithetic pairs across parallel envs that share a state, which staggered envs do not.

---

## 4. Horizon and chaos

### 4.1 What Q^pi looks like in a, and what its pathwise gradient would give [P, N]

Two different objects must be kept apart.

*Deterministic rollout return R(a_t; fixed future noise).* d s_{t+k}/d a_t grows like exp(Lambda k). The
derivative of the discounted return is sum_k gamma^k r'(s_{t+k}) d s_{t+k}/d a_t; its variance grows like
(gamma^2 E|f'|^2)^k. With Lambda above -ln gamma = 0.01/step the sum is dominated by the far horizon, and well
before that its variance exceeds the SF estimator's. This is the differentiable-simulator failure, and it does
not need contacts. Toy (`04_toy_chaos.py`, circle map, Lyapunov +0.30/step, gamma 0.9) [N], per-sample variance
in units of E g^2, exact V bootstrap at step n, both estimators unbiased:

| n | 1 | 2 | 3 | 5 | 8 | 12 | 20 | 40 |
|---|---|---|---|---|---|---|---|---|
| zeroth order (SF) | 2.6 | 3.7 | 4.6 | 6.0 | 7.3 | 8.1 | 8.6 | 8.7 |
| first order (through the simulator) | 1.35 | 4.1 | 8.6 | 33 | 232 | 3.0e3 | 5.3e5 | 2.6e11 |

In the contracting control (Lyapunov -0.14) first order stays within 0.3-24. First order wins only for n <= 1-2
Lyapunov-limited steps; SF saturates because the mean effect of an action is forgotten.

*Q^pi(s, a), the expectation over future policy noise.* This is smooth, but not at scale sigma for free:
V = r + gamma (K_sigma * V) o f is a sigma-convolution composed with an expanding map, so V is smooth at scale
~sigma/|f'| and Q^pi(s, .) inherits that. A coupling argument bounds its variation: two first actions delta apart
produce path distributions with total-variation distance <~ c |delta|/sigma (if the action-induced shift lies in
directions later noise can cancel), so |Q(s, a + delta) - Q(s, a)| <~ S |delta|/sigma with S the sd of the
return from a state. **Q^pi can vary by O(S) across one sigma and has structure at sigma divided by the
one-step expansion** [P for the bound given controllability, H otherwise]. Toy: pointwise slope rms 25.4 vs
sigma-smoothed 16.5 (ratio 1.5), corr between them 0.73, and only 68% of the within-state variance of Q is
linear in eps.

For HalfCheetah: S ~ 0.9 (the probe's action-invariant excess spread of Z) against sigma_A = 0.22, so Q varies
by a quarter of its available range across one sigma; and the oracle's slope ratio falls from 0.37 at h = 0.1
to 0.21 at h = 0.3 (both below sigma = 0.5), which is what a critic that is smoother or rougher than the truth
at sub-sigma scale would show. I cannot say from these two numbers which. **[U]**

### 4.2 Is a smoothed critic the right object? Yes, at scale sigma, and that is the SF estimand [P]

The actor's objective E_eps Q_hat(s, mu + sigma eps) has gradient E_eps grad Q_hat = E[Q_hat eps]/sigma. With
Q_hat = Q^pi it is the SF gradient exactly. The pathwise estimator differs only in variance: Var_eps(grad_a Q)
against Var(Q eps/sigma). If Q has structure of amplitude rho at length l < sigma, pathwise sample variance
~ rho^2/l^2 and SF ~ rho^2/sigma^2: pathwise is worse by (sigma/l)^2. A neural critic is smooth by spectral
bias, which is why pathwise-through-critic is usable at all; but then its sub-sigma structure is not data (one
action per state identifies none of it). Whatever fine structure dQ_hat/da has is association or memorised
noise. The correct object is the sigma-smoothed slope g(s), and a critic head that outputs g(s) directly
(linear in eps, plus He2 if wanted) is the minimal representation of everything identifiable.

### 4.3 Best target and horizon for the slope differ from those for the level [N in the toy, H for MuJoCo]

The slope target's signal is the *mean* effect of a_t on later rewards, which accumulates over about a mixing
time and then stops; its noise accumulates as sigma_A^2 sum (gamma lambda)^(2k). Truncating at n and
bootstrapping with V_hat replaces the remaining mean effect by V_hat's response to a state perturbation that has
grown by exp(Lambda n). Therefore:

- at n = 1, V_hat must be accurate at the scale |B sigma| of a one-step action effect;
- at larger n it only needs to be accurate at the scale exp(Lambda n) |B sigma|;
- the best n for the slope is about ln(l_V / |B sigma|)/Lambda, where l_V is the scale at which V_hat is
  accurate, and n = 1 is nearly useless when l_V is a few times |B sigma|.

Toy, V_hat = V blurred at length l (level error grows with l), exact bias, MC variance, chaotic map:

| V_hat blur l / sigma | slope bias^2 / E g^2 at n = 1, 2, 3, 5, 8, 12 | best n (K = 256 pooled) | MSE at n = 1 -> best |
|---|---|---|---|
| 1 | 0.062, 0.011, 0.002, 0, 0, 0 | 3 | 0.072 -> 0.021 |
| 4 | 0.55, 0.29, 0.077, 0.010, 0, 0 | 8 | 0.56 -> 0.039 |
| 16 | 0.95, 0.75, 0.49, 0.063, 0.001, 0 | 8 | 0.96 -> 0.064 |

ln(4)/0.30 = 4.6 and ln(16)/0.30 = 9.2 steps: the rule holds. With a critic blurred at 4 sigma the one-step slope
target has lost 55% of the signal and is a 14x worse estimate than the best horizon; the *level* target of the
same critic is fine at short n (see the "lvl err" rows in `data/04_toy_chaos.txt`). So lambda_slope > lambda_level
in general, and the gap widens the coarser V_hat is.

For REPPO: lambda = 0.8 gives an effective horizon of 4.8 steps. If branches decorrelate in ~100 steps, Lambda
~ 0.02-0.05/step [H], and the action effect needs tens of steps to be amplified to a scale the on-policy V_hat
resolves. Then most of the slope in REPPO's target is not real reward differences but the bootstrap critic's
own response V'(s_{t+1}) to the action, i.e. the critic's slope is trained largely on its own one-step backup,
which is the leak-prone quantity. (It does not even reach that fixed point: self-consistency is -0.33.) This is
also why the one-step model route and the R^2 0.996 world model do not help: both ask V_hat to be accurate at
scale |B sigma|. And it is how TD7 can use one-step targets: a million replayed transitions resolve V at that
scale.

The repo's lambda tests do not settle this: v53 changed a contrast head's lambda on a leaky, memorising
estimator; v61/62 at lambda 0.95 added noise (2.5). The slope-horizon claim is untested. E3 tests it offline.

A discount for the slope: the slope signal has decayed by the mixing time regardless of gamma, so a slope
estimator can use a shorter effective discount than the level (less noise, no bias) *provided* the truncated
tail is covered by a V_hat that is accurate at the amplified scale. Same trade, same rule.

---

## 5. Nonstationarity and rate

Quantities per iteration (61 per 1M steps, 128 critic and 32 actor steps each):

- Mean shift. Per-state KL 0.03 means |delta mu|/sigma = sqrt(0.06) = 0.245 in total norm, 0.10 per dimension.
- Curvature, from the measured noise cost. By the performance-difference lemma the per-step reward lost to
  noise is -(sigma^2/2) tr H. J(0) - J(1) = 5% at 1M, i.e. ~225 return at ~4.5k. Per sigma^2 of displacement in
  one direction (isotropy assumed): 225/6 = 37.5 return.
- Slope change from curvature: sigma^2 |H_jj| x 0.10 against sigma g_j ~ 0.09: **1-4% per iteration** [H]. The
  slope field is stationary over ~25-100 iterations early, apart from the unmeasured part due to V^pi changing
  [U]. The critic's contrasts, by contrast, change by 100% of their sd per iteration (v49). So the critic is
  not tracking a moving slope; it is redrawing noise around a nearly fixed one. **Estimation noise per
  iteration dominates tracking lag by one to two orders of magnitude early.** Tracking lag only matters for the
  level (Q rises by ~0.2 per iteration on average, small against the 0.9 error).
- First-order gain available at the taken radius: E[sigma |g|] x 0.245 = 0.22 x 0.245 = 0.054 per step in scaled
  units = 0.54 raw reward per step = **+540 return per iteration** if the mean shift were parallel to g
  everywhere (upper bound: treats all within-state sd as linear, 60-70% is more likely). Measured slope at 0:
  +32 +- 4 return per iteration at 1M. **Realised direction efficiency ~ 0.06** (0.06-0.10 with the linear
  share). It is of the same size as the critic's contrast corr (0.11-0.15): the actor's cross-state averaging
  is not buying more than the critic had per state, which is what a coherent (leak) error or a non-pooling
  actor would give, and not what independent noise would give. [H]
- Optimal length along the taken direction: gain 32 c, curvature cost 37.5 x (0.245 c)^2 = 2.25 c^2, maximum at
  **c = 7.1**. Measured true peak 7.8 (deterministic) / 5.4 (stochastic). The "5x too timid" result is
  reproduced from two other measurements with no free parameter, which says: early, the landscape along the step
  is linear out to the noise-curvature scale, and the step is set by the trust region alone. The KL gate is
  denominated in sigma; the information that a step is safe is not. At the peak the gain would be ~114 per
  iteration instead of 32, i.e. ~3.5x.

Ranking of the three terms early (1M): direction quality (factor ~10-15 on the table) > step length (~3.5x) >>
tracking lag (~1). They compound only partly: a 7x step in a direction with efficiency 0.06 is at its curvature
limit already.

Late (7.9M): sigma is smaller and means are saturated, so sd(mu)/sigma is large (leak time constant up by
5-20x, 1.2), the true slope along the step is <= 0 (the policy is near a stationary point of the true smoothed
objective at its own noise level), and what remains in the critic is association plus noise: cos(critic
gradient, step) 0.008 and a predicted gain where there is none. At a stationary point the signal needed is second
order (how to shed the 11% noise cost without losing the information the noise buys), which a first-order
critic slope cannot provide whatever its accuracy. The late plateau is therefore over-determined: no true
first-order signal, a biased estimate of it, and a step that still moves 0.245 sigma per iteration.

---

## 6. Ranked root causes

### RC1. The on-policy pathwise design has PPO's information and extracts less of it (variance; no pooling, no memory)

(a) Mechanism: 2.1-2.4. Within-state action information is the SF statistic. The critic stores about one noisy
projection per state (K_eff <~ 1), is rewritten each iteration, and the actor then does the pooling PPO does.

(b) Explains: fresh-action corr 0.11 with truth-sized sd; train 0.25 vs fresh 0.08 (v51); contrast drift = contrast
sd (v49); neutrality of dueling / distributional / scalar heads; PPO at parity; replay being the only thing
with a 4x gap (it changes the information, 2.2). Does not explain: the *negative* self-consistency (variance
gives 0), the late wrong sign, v63 doing worse than base.

(c) Decisive measurement E1 (offline, existing oracle + one checkpoint): (i) corr of single-sample lambda-return
residuals G - V_hat with the true advantage at the taken action, for lambda 0.8 and 0.95; (ii) fit a pooled
orthogonal slope g_hat(s) = W psi(s) by ridge regression of (G - V_hat) eps/sigma on P random features of s,
P in {64, 256, 1024}, from ONE rollout, and score it on the oracle.
  - RC1 right and the field is poolable: (i) 0.25-0.35; (ii) oracle contrast corr 0.4-0.7 at P ~ 256 (formula:
    K = N/P = 64, nu = 0.6 -> 0.74 if the truth lies in the span), far above the critic's 0.11, from the same data.
  - RC1 right but the field is not poolable (information-limited for every on-policy method): (ii) stays <= 0.2
    for all P. Then nothing on-policy helps and the answer is replay.
  - RC1 wrong: (i) > 0.6 (targets are clean and the critic still fails): the problem is fitting/bias, go to RC2.

### RC2. Collinearity leak: the action slope holds the cross-state association (bias, grows as sigma/sd(mu) falls)

(a) Mechanism: 1.2-1.3. The direction that separates a from s has eigenvalue ~ sigma^2/(Var(mu) + sigma^2); until
it is resolved the action coefficient is Cov(mu_perp, V_perp)/(sigma^2 + Var(mu_perp)), with a sign opposite to
the truth for under-gained feedback.

(b) Explains: negative self-consistency (-0.33); truth-sized contrast sd; corr rising with behaviour spread
(0, 0.1, 0.29 at std 0.1, 0.3, 0.45); late "critic predicts gain, truth <= 0"; raw critic gradient nearly
orthogonal to the useful step and destructive along itself (a coherent, non-averaging error); the direction
efficiency not exceeding the per-state corr. Does not explain: v63's regression (unless the association is
useful early, 1.4 item 2); the 1M corr by itself (toy transient is over by then at sd(mu)/sigma ~ 2).

(c) Decisive experiment E2, "placebo actions" (offline critic fits on a saved rollout; no environment): train
the critic from the same initialisation on the same states and targets, once with the real actions and once
with a = tanh(mu(s) + sigma eps') for fresh eps' (targets untouched, so the true slope w.r.t. the placebo action
is exactly zero). On held-out states compare the two action-gradient fields.
  - Leak dominant: placebo within-state contrast sd >= 0.6 of the real critic's (>= 0.13), and
    cos(grad_a Q_placebo, grad_a Q_real) >= 0.5 (the slope is a function of s, not of the action taken).
  - Variance dominant (RC1): placebo sd similar, cos ~ 0 +- 0.05 (each memorises its own eps).
  - Critic sound: placebo sd <= 0.2 of real.
  Repeat at an 8M checkpoint: prediction cos >= 0.8 late if RC2 explains the late regime. A second, even cheaper
  signature: cos between the critic's action gradient and the pooled-across-states regression slope of V_hat on
  the action; RC2 predicts >= 0.3, otherwise ~0.

### RC3. The slope's target horizon is too short for an on-policy V (bootstrap self-reference)

(a) Mechanism: 4.3. With lambda 0.8 (4.8 steps) the slope in the target is mostly V'(s_{t+1})'s response to the
action, which requires V accurate at scale |B sigma|; in a chaotic system the on-policy V is accurate only at
a scale the perturbation reaches after ln(l_V/|B sigma|)/Lambda steps.

(b) Explains: one-step model route no better; R^2 0.996 world model useless for ranking; TD7 working with
one-step targets (replay resolves V finely); the oracle's "V's state gradient is inaccurate too". Does not
explain: why lambda 0.95 did not help in v61 (confounded, 2.5) or v53 (leaky estimator). Uncertain because
Lambda and l_V are not measured for HalfCheetah.

(c) Decisive measurement E3 (oracle branching data, no training): the fraction of the true within-state contrast
variance realised by the first n steps of reward (truncated true contrast vs full), n = 1, 5, 20, 50, 100; and
the corr with truth of the hybrid "true n-step rewards + critic V' at step n".
  - RC3 right: < 30% realised at n = 5, > 80% by n = 50; hybrid corr rises from ~0.15 (n = 1) to > 0.5 by
    n ~ 20-50 before single-rollout noise takes over.
  - RC3 wrong: > 70% realised by n = 5 (then a short horizon is fine and the fault is entirely V's fine-scale
    error, i.e. RC1/RC2).

### RC4. Step geometry: the trust region is denominated in sigma, not in evidence

(a) Mechanism: section 5. Early the landscape along the step is linear to 7x the step; the gate holds it at 1x.
Late it lets a step of 0.245 sigma through with no true signal behind it.

(b) Explains: 5x too timid early with the peak location predicted from the noise cost (7.1 vs 5.4-7.8); late
damage. Does not explain: the low direction efficiency; why v57/v58 regressed (I have not audited them; a line
search on a direction with efficiency 0.06 and a critic that cannot track a 7x faster level would not be
expected to deliver the nominal 3.5x). Lower rank because PPO shares the radius and performs alike.

(c) Experiment E5: from the 1M checkpoint, fork with the mean's KL budget x c^2 for c = 2, 4 for ~20 iterations
with the critic's pass count scaled likewise. Prediction if RC4 matters: return slope per iteration x 1.8 and
x 3 (sub-linear from curvature); if the critic cannot keep its level, slope unchanged and bellman_rms up > 1.5x.

### RC5. Loss scale and geometry (HL-Gauss bins, shared head)

(a) A rate handicap of ~3x on a low-variance component (3.1), not a resolution floor. (b) Explains v51's gain;
does not explain anything about sign or generalisation; refuted as the bottleneck by aliasing 3e-5 and by
v59. (c) Already measured; no further test needed. A cheap confirmation: the placebo/real fits of E2 with an MSE
contrast head should reach the same end state ~3x sooner.

### RC6. Roughness of the true Q in a at sub-sigma scale (chaos)

(a) 4.1: structure at sigma/expansion, so the pointwise gradient is a noisier and slightly different object from
the smoothed slope. (b) Could contribute to the h-dependence of the oracle slope. Does not limit a critic that
targets the smoothed slope. (c) E6: from the oracle, the share of true within-state variance linear in eps;
toy value 0.68. If < 0.3 in HalfCheetah, sigma is too large for any first-order method and curvature terms must
be modelled; if > 0.6 this cause is negligible.

My ordering by expected share of the gap to a good on-policy method: RC1 > RC2 > RC3 > RC4 > RC5 > RC6 early;
RC2 > RC4 > RC1 late. My confidence in the top two being the top two is moderate (~65%); in their relative
order at 1M, low. E1 and E2 are both offline and together decide it.

---

## 7. The statistically optimal fully-on-policy estimator for this regime

Derived from the constraints above, not from a survey.

1. **Actor gradient = score function with a critic control variate, not pathwise through Q.** The unbiased,
   leak-free statistic is sum J^T eps (y - b)/sigma. A critic enters as b(s) (level) and, if its slope is good,
   as an action-dependent control variate whose mean is added back analytically:
   grad = E[J^T eps (y - V_hat - sigma g_hat.eps)/sigma] + E[J^T g_hat]. With g_hat exact the first term has only
   the nu-noise; with g_hat useless it is PPO; with g_hat wrong it stays unbiased. This is the form in which a
   bad critic costs variance and never direction. (It is Q-Prop's structure; the repo's qprop variants on other
   bases are not evidence about it here because their critics were (s, a) regressions with the leak.)
2. **Slope critic = orthogonal vector regression.** Regress eps (y - V_hat)/sigma on a state encoder,
   g_hat(s) = W psi(s), with shrinkage set from the known noise level ((d + 1) sigma_A^2 + d nu^2)/sigma^2 and a
   capacity of a few hundred effective degrees of freedom per rollout (N/K_half = 170-960). No (s, a) joint input
   for the slope. V_hat is cross-fitted (the slope uses a V that did not see that row's target) so the
   memorised residual cannot be read back.
3. **Cross-iteration filter on the slope.** The slope moves 1-4% per iteration from curvature. A Kalman / EMA
   update of W with gain ~1/W_eff, W_eff ~ 10-30, expressed in absolute pre-tanh coordinates (the slope at mu_k
   is transported by the diagonal curvature estimate), multiplies the effective sample count by W_eff. This is
   the legitimate on-policy version of replay: it reuses estimates, not transitions, so no importance weights and
   no stale returns.
4. **Separate horizons.** Level target lambda_level ~ 0.8-0.9 (low noise). Slope target lambda_slope ~ 0.95-0.98,
   chosen as ln(l_V/|B sigma|)/Lambda steps (E3 gives the number); the future-action noise this admits is what
   item 1's control variate and item 3's averaging are for.
5. **Step from evidence.** Mean step per state = argmax of the estimated gain minus the measured curvature cost,
   c* = (estimated slope along the step) / (sigma^2 |H|) scaled by the estimate's reliability (corr^2 from item
   2's known noise). Early this is ~7x the current step; late it goes to zero where the slope's standard error
   exceeds it. The exploration scale sigma is then set by what the estimator needs (SNR is independent of
   sigma in the linear regime, 2.3, so sigma should be as small as keeps Q linear over +-sigma and the leak
   irrelevant, and no smaller than the scale at which V is resolved) rather than by an entropy target.
6. **If one departure from on-policy is allowed:** keep the last W rollouts only for the slope regression and
   the dynamics-through-V route, with one-step (or Retrace) residuals. Replay's decisive property is that it
   gives Var(a | s) >> sigma^2, which removes the collinearity outright; that, not the sample count, is why TD7
   can learn dQ/da.

Expected gain over PPO, with the uncertainty stated:

- From item 1+2 alone: none by construction (it is PPO's statistic). It only removes REPPO's deficit.
- From bootstrapped credit (item 4 with a better V): bounded by 8.7x in samples against GAE 0.95; realistically
  1.2-2x, because V is the limiting object.
- From cross-iteration averaging (item 3): up to W_eff ~ 10-30 in slope samples early, when the direction
  efficiency is ~0.06 and the field is nearly stationary; realistically a 2-4x better direction (efficiency
  0.15-0.25), less late.
- From step length (item 5): up to 3.5x early along a good direction, shrinking to nothing by mid-training.
- Net: I would expect 1.5-3x in steps-to-a-given-return in the first 2-4M and a smaller late noise-cost plateau,
  i.e. a few thousand return at 8M at most, **not** TD7's regime. If E1 shows the slope field is not poolable
  (corr <= 0.2 for every P), the expected gain of any fully on-policy scheme over PPO is ~zero and the only
  remaining lever is replay.

## 8. Caveats on this report

- All HalfCheetah-specific numbers are the repo's; I did not re-measure any. nu (the per-sample residual noise),
  the autocorrelation of the level error, Lambda and l_V are inferred or unknown. The match of 0.117 to the
  oracle's 0.11 depends on nu ~ 0.6 and should be read as "a single-sample, un-pooled slope is sufficient to
  produce the oracle's numbers", not as proof that it does.
- The leak toy is linear dynamics with a linear policy and a simple slope field; it shows the mechanism and its
  scaling, not its magnitude in MuJoCo. Its "rough state term" and "level drift" are action-independent
  additions to the targets, a modelling shortcut.
- The chaos toy is one-dimensional, where the SF dimension penalty (d + 1) is 2 rather than 7, and where
  "controllability by later noise" holds trivially.
- The direction-efficiency and peak-location arithmetic assumes an isotropic action Hessian, treats the whole
  within-state sd as linear, and uses a discounted advantage for an undiscounted return. Each is good to a
  factor ~2; the peak-location agreement (7.1 vs 5.4-7.8) may be partly luck.
- I did not audit v57/v58 (step-length variants) or the qprop/IPG files outside `cleanrl/reppo/`.
