# Learning toward your own prediction: literature survey for an "optimizer-side JEPA"

Date: 2026-09-11. Scope: systems that optimize toward THEIR OWN prediction of the future / of the target
instead of the raw noisy realized target, why that gives denser and more generalizable signal, and what the
per-parameter optimizer-side analogue would be. Every entry below was verified by fetching the abstract or
full text (arXiv / ar5iv / venue page) unless marked "[recalled]" (title+venue verified, internal detail from memory).

Notation used throughout (ASCII): sg() = stop-gradient; EMA(tau): theta_bar <- tau*theta_bar + (1-tau)*theta;
g_t = stochastic gradient at step t; eta = step size.

---------------------------------------------------------------------------------------------------
## Thread 1. Self-predictive representation learning (EMA target / stop-gradient / JEPA family)
---------------------------------------------------------------------------------------------------

### 1.1 BYOL — Grill et al., NeurIPS 2020. https://arxiv.org/abs/2006.07733
Mechanism. Online net f_theta -> projector g_theta -> predictor q_theta predicts the target net's projection of
another view; target net xi = EMA of theta. No negatives.
  L = || q_theta(z_theta(v)) / ||.|| - sg( z_xi(v') ) / ||.|| ||^2 ;  xi <- tau*xi + (1-tau)*theta
Strongest claim. 74.3% ImageNet top-1 linear probe with ResNet-50 (79.6% with wider ResNet).
Collapse / failure. Table 5(a) tau ablation (300 ep): tau=1 (frozen random target) 18.8%; tau=0.9 68.4%;
tau=0.99 72.5%; tau=0.999 69.8%; tau=0 (target = sg(online), i.e. "predict yourself now") 0.3% = collapse.
Why no collapse (their argument): with optimal predictor q*(z) = E[z'_xi | z_theta] the online update follows
  grad_theta E[ sum_i Var(z'_xi,i | z_theta) ],
i.e. the online net cannot lower conditional variance by discarding information, so constant solutions are
unstable equilibria; the slow target keeps the predictor near-optimal. Mean Teacher (no predictor) collapses
without the supervised loss.
Optimizer analogue. tau=0 collapse is the key warning: a target that equals the current iterate carries no
information; the target must lag (be a slow average of past iterates) and the predictor must be fast.

### 1.2 SimSiam — Chen & He, CVPR 2021. https://arxiv.org/abs/2011.10566
Mechanism. Same two-branch cosine loss but the target is sg(z) of the *same* encoder (no EMA); a predictor h on
the other branch.  L = -1/2 [ cos(h(z1), sg(z2)) + cos(h(z2), sg(z1)) ].
Hypothesis: stop-gradient makes it an EM-like alternating optimization over (encoder, per-image latent eta).
Strongest claim. Competitive ImageNet linear probe without negatives, momentum encoder, or large batch.
Failure. Removing stop-gradient collapses immediately (loss -> -1, output std -> 0); removing the predictor also fails.
Optimizer analogue. The "predictor + stop-grad" pair is the minimum: something must be optimized fast on top of
the frozen target; without the fast part, the target and the learner become the same variable and the loss is
trivially minimized.

### 1.3 Understanding Self-Predictive Learning for RL — Tang et al., ICML 2023. https://arxiv.org/abs/2212.03319
(PMLR: https://proceedings.mlr.press/v202/tang23d/tang23d.pdf)
Mechanism / math. Linear representation Phi in R^{|S| x k}, predictor P in R^{k x k}, objective
  min_{Phi,P} E_{x~d, y~P^pi(.|x)} || P^T Phi^T x - sg(Phi^T y) ||^2
with semi-gradient on Phi and a predictor that is optimal at every instant ("fast predictor"):
  P_t = argmin_P E || P^T Phi_t^T x - Phi_t^T y ||^2  (closed form: P_t = (Phi^T D Phi)^{-1} Phi^T D P^pi Phi)
  dPhi/dt = - grad_Phi ||.||^2 evaluated with the target held constant.
Theorem (non-collapse). Under (i) optimal predictor and (ii) semi-gradient, the k x k covariance Phi_t^T Phi_t is
CONSTANT in time: the representation rotates but never loses rank/norm. For symmetric P^pi the dynamics ascend
the trace objective tr(Phi^T P^pi Phi) (PCA on the transition matrix); bidirectional variant does SVD.
Failure. Ablations: full gradient (no sg) -> columns collapse to each other; predictor corrupted / slow -> all
representation vectors converge to the same vector. The "fast target / slow predictor" phrase in the task
statement is inverted relative to the paper: the condition is FAST PREDICTOR (near-optimal at all times),
SLOW REPRESENTATION (semi-gradient, and in practice EMA target), i.e. two-timescale.
Strongest claim. Analytical: covariance preserved exactly; empirical: DMLab-30 gains on some tasks with
bidirectional variant.
Optimizer analogue. Two-timescale structure: the "predictor" (map from current state to predicted future state)
must be re-fit faster than the thing it predicts changes; the thing it predicts must not receive gradient from
the prediction loss. Rank preservation is the invariant to monitor.

### 1.4 Bridging state and history representations: understanding self-predictive RL — Ni et al., ICLR 2024.
https://arxiv.org/abs/2401.08898
Mechanism. ZP (next-latent prediction) loss with three targets: online (phi), detached (sg(phi)), EMA (tau in (0,1)).
Prop. 3: l2 ZP with stop-grad has stationary points satisfying the self-predictive condition; online target lacks
this guarantee. Thm 3 (linear): with stop-grad target phi^T phi retains its initial value along continuous-time
dynamics (rank preserved); online target has no such protection.
Strongest claim. Empirically "online target -> lower returns and low-rank representations"; "detached or EMA
target mitigates collapse". Recommendation: l2 + EMA target first.
Optimizer analogue. Same invariant as 1.3; EMA target is the robust default.

### 1.5 SPR — Schwarzer et al., ICLR 2021. https://arxiv.org/abs/2007.05929
Mechanism. Latent transition model h(z_t, a_t) predicts z_{t+1..t+K}; targets from an EMA encoder; normalized l2 /
cosine loss; augmentation on both branches. Strongest claim. Atari-100k median HNS 0.415 (+55% over prior SOTA),
superhuman on 7/26. Failure. Depends on EMA target (tau=0.99 default); without augmentation gains shrink.

### 1.6 BYOL-Explore — Guo et al., NeurIPS 2022. https://arxiv.org/abs/2206.08332
Mechanism. One latent prediction loss (BYOL-style, EMA target) trains representation + world model and its
per-step loss is the intrinsic reward. Strongest claim. Solves all 10 hardest DM-HARD-8 tasks with a single
objective (their headline). Failure. Prediction error toward an EMA target is non-stationary; needs reward
normalization / prioritization. Optimizer analogue: the surprise of the optimizer's own forecast (||g_t -
predicted||) is a usable signal for adaptivity (trust region / step size), not only a target.

### 1.7 I-JEPA — Assran et al., CVPR 2023. https://arxiv.org/abs/2301.08243
Mechanism. Context encoder + predictor predict EMA target-encoder representations of masked target blocks;
loss in representation space. Argument: predicting in latent space discards pixel detail that is irrelevant to
semantics, so the target is "denser in meaning per bit". Strongest claim. ViT-H/14 trained on ImageNet in <72h
on 16 A100s, strong linear probe / low-shot without hand-crafted augmentation; more compute-efficient than MAE.
Failure. EMA target + masking design is load-bearing; small target blocks lose semantics.

### 1.8 V-JEPA — Bardes et al., 2024 (arXiv 2404.08471); V-JEPA 2 — Assran et al., 2025 (arXiv 2506.09985).
Mechanism. Masked spatio-temporal feature prediction toward an EMA target encoder, no pixel decoder.
Strongest claims. V-JEPA ViT-H/16: K400 81.9, SSv2 72.2, IN1K 77.9 frozen; V-JEPA 2: 1B params, >1M hours of
video, zero-shot robot planning. Failure. Same stop-grad/EMA dependence.

### 1.9 LeJEPA — Balestriero & LeCun, 2025. https://arxiv.org/abs/2511.08544
Mechanism. Proves the isotropic Gaussian is the embedding distribution minimizing downstream prediction risk
and adds SIGReg (sketched isotropic Gaussian regularization, linear time). Result: JEPA prediction loss +
SIGReg with NO stop-gradient, NO teacher-student, NO EMA, ~50 lines. Strongest claim. 79% ImageNet-1K linear
probe, ViT-H/14, across 60+ architectures. Optimizer analogue. Collapse can be prevented by an explicit
distributional regularizer on the predicted quantity instead of asymmetry; e.g. constrain the predicted update
distribution's covariance.

### 1.10 Learning by reconstruction produces uninformative features — Balestriero & LeCun, 2024.
https://arxiv.org/abs/2402.11337
Claim. Pixel-reconstruction targets spend capacity on the high-variance pixel subspace: TinyImageNet projected
to the top subspace holding 90% of pixel variance gives 45% supervised accuracy; the BOTTOM subspace with 20% of
variance gives 55%. Perceptual features are learned last. Motivation for latent (predicted) targets.
Optimizer analogue. The raw stochastic gradient's variance is dominated by directions that do not matter for the
objective; a latent/forecast target filters them.

### 1.11 TD-MPC2 — Hansen, Su, Wang, ICLR 2024 (verified via proceedings page + TD-MPC PMLR).
Mechanism. Latent consistency loss: || d(z_t, a_t) - sg(h_bar(s_{t+1})) ||^2 with h_bar an EMA encoder; SimNorm
(group-wise simplex normalization) to prevent latent blowup. Strongest claim. 104 tasks with one hyperparameter
set; 317M-param agent for 80 tasks. Failure. Latent norm blowup without SimNorm (the "explode" collapse mode
rather than the "constant" collapse mode).

### 1.12 DreamerV3 — Hafner et al., 2023. https://arxiv.org/abs/2301.04104
Mechanism. Dynamics loss KL(sg(post) || prior) weight 0.5 trains the predictor toward the target; representation
loss KL(post || sg(prior)) weight 0.1 regularizes the target toward the predictor; free bits (clip at 1 nat)
stop the KL from collapsing to zero. Optimizer analogue. Asymmetric two-way coupling: predictor pulled hard
toward observed state, state pulled weakly toward the prediction — a concrete template for a predicted-target
that is "regularized toward but not replaced by" the forecast.

---------------------------------------------------------------------------------------------------
## Thread 2. Bootstrapping in RL: learning from your own prediction
---------------------------------------------------------------------------------------------------

### 2.1 Why target networks stabilise TD methods — Fellows, Smith, Whiteson, ICML 2023. https://arxiv.org/abs/2302.12537
Mechanism. Semi-gradient TD's expected update has Jacobian J_TD = E[(gamma*grad v(s') - grad v(s)) grad v(s)^T]
(linear: gamma*Phi' - Phi); when the gamma-term dominates, J_TD has eigenvalues with positive real part ->
divergence (deadly triad). A target network makes it "partially fitted policy evaluation": k steps on a frozen
target then copy:
  w_{kl+i+1} = w_{kl+i} + alpha * delta(w_{kl+i}, w_bar_l);  w_bar_{l+1} = w_{k(l+1)}.
Stability condition C(alpha,k) < 1 with C = |1-alpha*lambda_H|^{k-1} ||J_TD|| + (1+|1-alpha*lambda_H|^{k-1})||J_FPE||:
for any admissible alpha a finite k exists that stabilizes even when TD(k=1) diverges. Convergence to an error
ball of radius alpha*sigma_k/(1-c).
Strongest claim. Baird's counterexample converges with k >= 500 (alpha 0.01, gamma 0.99) where k=1 diverges.
Failure. k too small -> divergence; k too large -> slow (fitted). Paper does not analyze Polyak/soft targets.
Optimizer analogue. A frozen/slow "target copy" of the iterate turns an ill-conditioned (non-symmetric) update
operator into a contraction. The optimizer-side condition is: the forecast must be held fixed long enough that
the inner problem is nearly solved before the forecast moves.

### 2.2 Bias-variance error bounds for TD updates — Kearns & Singh, COLT 2000. [recalled bound form]
https://www.learningtheory.org/colt2000/papers/KearnsSingh.pdf
Mechanism. Phased TD(k) with n trajectories per phase: error recursion of the form
  Delta_{t+1} <= gamma^k * Delta_t (bias from bootstrapping on own estimate) + ((1-gamma^k)/(1-gamma)) * sqrt(3 log(k/delta)/n) (variance of realized rewards)
and an analogous bound for TD(lambda). Larger k: faster contraction but larger asymptotic error; small k: slower
but lower variance. Yields decreasing-k schedules beating any fixed k.
Optimizer analogue. Interpolate between own forecast (low variance, biased) and realized gradient (unbiased,
noisy) with a schedule that trusts the forecast more as it becomes accurate.

### 2.3 Averaging n-step returns reduces variance — Daley, White, Machado, ICML 2024. https://arxiv.org/abs/2402.03903
Theorem. Any compound return (weighted average of n-step returns) with the same contraction modulus as a given
n-step return has strictly lower variance. Two-bootstrap returns improve DQN/PPO sample efficiency.
Optimizer analogue. Averaging several forecast horizons (1-step, k-step) at fixed "contraction" beats any single
horizon in variance.

### 2.4 Streaming deep RL finally works — Elsayed, Vasan, Mahmood, 2024. https://arxiv.org/abs/2410.14606
Mechanism. stream-TD/Q/AC(lambda): eligibility traces (lambda-return = geometric mix of own bootstrap and realized
return), ObGD (overshooting-bounded step size), sparse-init, no replay, no batch, NO target network.
Strongest claim. Best model-free performance on DM-Control Dog; stable streaming on MuJoCo/Atari where batch
methods fail ("stream barrier").
Relevance. Evidence that on noisy streaming data the interpolation own-prediction<->realized (lambda) plus a
bounded-overshoot step (a trust check against the forecast) is what makes it stable, while a lagged target
copy was dropped. Companion: AVG (Vasan et al., NeurIPS 2024, https://arxiv.org/abs/2411.15370) — incremental
policy gradient with normalization/scaling, first real-robot incremental deep RL.

---------------------------------------------------------------------------------------------------
## Thread 3. Self-distillation, noisy labels, and learning from own predictions in supervised learning
---------------------------------------------------------------------------------------------------

### 3.1 A closer look at memorization — Arpit et al., ICML 2017. https://arxiv.org/abs/1706.05394
Claim. DNNs fit simple / shared patterns first and memorize noise later; gradient dynamics differ on noise vs
real data; tuned regularization (dropout) hurts noise-fitting without hurting real-data generalization; the
dataset itself, not capacity alone, sets the degree of memorization. This is the empirical basis for every
"early prediction is cleaner than the label" method below.

### 3.2 Early-learning regularization (ELR) — Liu, Niles-Weed, Razavian, Fernandez-Granda, NeurIPS 2020.
https://arxiv.org/abs/2007.00151
Mechanism. Temporal-ensemble target t^(k) = beta*t^(k-1) + (1-beta)*p^(k) (beta ~ 0.7-0.99), and
  L_ELR = L_CE + (lambda/n) sum_i log(1 - <p_i, t_i>).
Lemma 2: gradient is (1/n) sum grad N (p_i - y_i + lambda*g_i); the regularizer's gradient pushes p_i toward
t_i, which (a) keeps a gradient alive on correctly-labeled examples after CE has vanished on them and (b) cancels
the CE gradient on mislabeled examples, preventing memorization. Theorem 1 (linear, high-dim): three phases —
early learning where the gradient aligns with the true separator, then vanishing clean-gradient with growing
mislabeled coefficients, then memorization.
Strongest claim. CIFAR-10 40% sym 91.4% (ELR+), CIFAR-100 40% sym 68.4%, Clothing1M 74.8% (SOTA at the time).
Failure. If beta is too small the target tracks the current (already-memorizing) model; if lambda is too large
the model locks onto early errors (confirmation bias).
Optimizer analogue. The EMA of the model's OWN past outputs is a better target than the label during the
memorization phase because memorization is late and slow: an EMA of past updates/iterates is likewise a better
target than the current noisy gradient in the phase where the raw gradient is dominated by sample noise.

### 3.3 Bootstrapping loss — Reed et al., ICLR-W 2015. https://arxiv.org/abs/1412.6596
Target = beta*y + (1-beta)*q (soft) or beta*y + (1-beta)*argmax q (hard), q = current prediction.
Robust to MNIST label corruption and detection with noisy boxes. Failure: with q from the CURRENT model (no lag)
it is confirmation-biased; ELR's EMA fixes exactly this.

### 3.4 Co-teaching — Han et al., NeurIPS 2018. https://arxiv.org/abs/1804.06872
Two nets select small-loss samples for each other (memorization effect => small loss = clean). Optimizer
analogue: a second copy that decides which stochastic updates to trust.

### 3.5 DivideMix — Li, Socher, Hoi, ICLR 2020. https://arxiv.org/abs/2002.07394
Per-sample-loss GMM splits clean/noisy; two nets co-divide; MixMatch with own predictions as pseudo-labels.
CIFAR-10 at 90% noise and Clothing1M SOTA at the time.

### 3.6 Self-adaptive training — Huang, Zhang, Zhang, NeurIPS 2020 / TPAMI 2022.
https://proceedings.neurips.cc/paper/2020/file/e0ab531ec312161511493b002f9be2ee-Paper.pdf
Refurbished label y^r <- alpha*y^r + (1-alpha)*p_hat with sample weights; improves noisy-label, adversarial and
linear-eval results. Same EMA-of-own-prediction mechanism as ELR, applied to the label itself.

### 3.7 Temporal ensembling — Laine & Aila, ICLR 2017. https://arxiv.org/abs/1610.02242
Z <- alpha*Z + (1-alpha)*z, bias-corrected, consistency loss ||z - Z/(1-alpha^t)||^2. SVHN 500 labels: 18.44 ->
7.05% error; CIFAR-10 4000 labels 18.63 -> 16.55%; tolerant to incorrect labels. Failure: targets update once per
epoch -> stale on large data.

### 3.8 Mean Teacher — Tarvainen & Valpola, NeurIPS 2017. https://arxiv.org/abs/1703.01780
Average WEIGHTS instead of predictions: theta' <- alpha*theta' + (1-alpha)*theta; consistency to teacher output.
SVHN 250 labels 4.35% error (beats TE with 1000). Failure: without a supervised term it collapses (BYOL App.).
This is the first explicit move from output-EMA to weight-EMA as the "predicted target" — the direct precursor
of the optimizer-side idea.

### 3.9 Born-again networks — Furlanello et al., ICML 2018. https://arxiv.org/abs/1805.04770
Same-architecture student on teacher's soft outputs, repeated generations; DenseNet CIFAR-100 15.5% error.
CWTM/DKPP ablations show both the argmax-confidence weighting and the non-argmax "dark knowledge" carry signal.

### 3.10 Self-distillation amplifies regularization in Hilbert space — Mobahi, Farajtabar, Bartlett, NeurIPS 2020.
https://arxiv.org/abs/2002.05715
Exact claim. Regularized regression f* = argmin R(f) s.t. (1/K) sum (f(x_k)-y_k)^2 <= eps, R an l2 operator
norm; representer solution f*(x) = g_x^T (cI + G)^{-1} y with G the Gram matrix (eigenvalues d_k). Round t
retrains on the previous round's predictions y_t = f_{t-1}(X). In the eigenbasis the coefficients are
  B_t[k,k] = prod_{i=0}^{t} d_k / (c_i + d_k),
a per-round soft-threshold that shrinks small-eigenvalue directions FASTER than large ones (Thm 5: the ratio
B[k,k]/B[j,j] for d_j < d_k grows exponentially in t). So self-distillation = progressively limiting the number
of basis functions (amplified regularization / sparsification), NOT new information. Prop. 4: guaranteed
non-collapse rounds t_bar = (||y_0||/sqrt(K eps) - 1)/kappa, kappa = d_max/d_min; beyond that the solution
falls below the loss tolerance and collapses to zero. Thm 6: sparsification strongest near interpolation (eps -> 0).
Strongest claim. Inverted-U of test accuracy in rounds on CIFAR-10/100 (ResNet/VGG), matching the theory.
Optimizer analogue. Repeatedly fitting your own prediction is a low-pass filter in the eigenbasis of the
problem's kernel/Hessian; a few rounds denoise, too many under-fit. An optimizer that steps toward its own
forecast will suppress small-curvature directions first; that is the regularization AND the collapse.

### 3.11 A statistical perspective on distillation — Menon, Rawat, Reddi, Kim, Kumar, ICML 2021. https://arxiv.org/abs/2005.10419
Exact claim. Let p*(x) = Bayes class-probability. Bayes-distilled risk R_*(f,S) = (1/N) sum_n p*(x_n)^T l(f(x_n)).
Lemma 1: Var[R_*(f,S)] <= Var[R_onehot(f,S)] (strictly, unless p* is one-hot): soft labels are a lower-variance
estimator of the population risk. Prop. 3 (bias-variance): for a teacher p^t,
  E[(R^t(f,S) - R(f))^2] <= O( Var_teacher / N ) + O( E_x ||p^t(x) - p*(x)||_2^2 ),
so distillation helps iff the teacher is a better estimate of p* than the one-hot label, in MSE. Consequences:
calibration matters more than accuracy; temperature and label smoothing are bias-for-variance trades.
Empirical: teacher MSE to p* predicts student quality better than teacher accuracy.
Optimizer analogue. The "dense target" is denser only because it is closer to the conditional expectation of the
noisy target. A forecast of the update is worth using only if E||forecast - E[g]||^2 < Var[g] per coordinate:
this is the acceptance test.

### 3.12 Distilling the knowledge in a NN — Hinton, Vinyals, Dean, 2015. https://arxiv.org/abs/1503.02531
Soft targets (temperature T) carry more information per case and "much less variance in the gradient between
training cases", allowing fewer data / higher LR. MNIST: student generalizes to a digit class it never saw.

### 3.13 Revisiting KD via label smoothing — Yuan et al., CVPR 2020. https://arxiv.org/abs/1909.11723
KD is learned label-smoothing regularization: reversed KD (student -> teacher) and poorly-trained teachers still
help; Tf-KD (self-teacher) up to +0.65% ImageNet. Interpretation: much of "dark knowledge" is regularization.

### 3.14 Progressive self-KD — Kim et al., ICCV 2021. https://arxiv.org/abs/2006.12000
Target (1-alpha_t) y + alpha_t p_{t-1} with p from the previous epoch and alpha_t growing; gains in accuracy,
noisy-label robustness (gradient rescaling = hard-example mining) and calibration. Explicit "step toward your
lagged self" rule.

### 3.15 Distillation ~ early stopping — Dong, Hou, Yang, 2019 (arXiv 1910.01255) [abstract via secondary sources].
Overparameterized nets fit high-eigenvalue (informative) directions first (anisotropic information retrieval);
shifting supervision to own outputs prevents late fitting of noise; works at 60-80% symmetric noise.

### 3.16 Understanding self-distillation in the presence of label noise — Das & Sanghavi, ICML 2023.
https://arxiv.org/abs/2301.13304
Student loss mixes teacher prediction and label with xi; for regularized linear regression under high label
noise the optimal xi > 1 (extrapolate PAST the teacher, away from the label); holds empirically with CE at
30-50% corruption. Optimizer analogue: over-relaxation toward the forecast can be optimal under heavy noise.

---------------------------------------------------------------------------------------------------
## Thread 4. Optimizer-level analogues
---------------------------------------------------------------------------------------------------

### 4.1 Polyak-Juditsky averaging — SIAM J. Control Optim. 30(4):838-855, 1992.
Averaged iterate x_bar_t = (1/t) sum x_s of SGD with slowly decaying steps is asymptotically normal with the
optimal (Fisher / minimax) covariance; the raw iterate is not. This is THE streaming-data result: on an i.i.d.
stream, the average of your own past states is a strictly better estimator than the current state.

### 4.2 SWA — Izmailov et al., UAI 2018. https://arxiv.org/abs/1803.05407
Average snapshots along a constant/cyclic-LR trajectory; flatter minima; better test accuracy on CIFAR/ImageNet.

### 4.3 Early weight averaging / LAWA — Sanyal et al., COLM 2024. https://arxiv.org/abs/2306.03241
Averaging the latest checkpoints with wide spacing at high LR speeds LLM pretraining; beats EMA/SWA baselines.

### 4.4 EMA of weights: dynamics and benefits — Morales-Brotons, Vogels, Hendrikx, TMLR 2024. https://arxiv.org/abs/2411.18704
Mechanism. theta_ema <- alpha*theta_ema + (1-alpha)*theta, effective window ~1/(1-alpha). EMA behaves like
running at a lower LR without decaying the LR (noise kept for exploration, averaged out for evaluation).
Strongest claims. CIFAR-100 WRN-28-10 81.07 -> 82.72; Tiny-ImageNet +1.94pp; CIFAR-100N 40% noise: CE 55.5 ->
EMA 65.15 (competitive with specialized noisy-label methods); ECE 11.75 -> 9.46; EMA models good early ->
explains their use as teachers (Mean Teacher, BYOL, MoCo).
Failure. Gains vanish at low LR; lag at high alpha (see "EMA without the lag", arXiv 2508.00180).

### 4.5 Lookahead — Zhang, Lucas, Hinton, Ba, NeurIPS 2019. https://arxiv.org/abs/1907.08610
Mechanism. Fast weights theta run k inner steps of any optimizer from slow weights phi; then
  phi <- phi + alpha*(theta_k - phi),  theta <- phi.
The slow weights step toward where the fast optimizer went: literally "step toward the optimizer's own
short-horizon forecast of its state".
Noisy-quadratic analysis (Prop. 2). SGD variance fixed point V_SGD* = eta^2 A^2 Sigma / (I - (I - eta A)^2);
  V_LA* = alpha^2 (I-(I-eta A)^{2k}) / [ alpha^2 (I-(I-eta A)^{2k}) + 2 alpha (1-alpha)(I-(I-eta A)^k) ] * V_SGD*,
strictly smaller than V_SGD* for alpha in (0,1). Deterministic quadratic: helps in the under-damped regime.
Strongest claim. PTB LSTM perplexity 57.72 (Lookahead-Adam) vs 59.33 (Adam); CIFAR/ImageNet small gains;
robust to inner hyperparameters. Failure. In the over-damped regime it slows convergence; alpha -> 1 reduces to
the inner optimizer, alpha -> 0 freezes.

### 4.6 Schedule-Free — Defazio et al., 2024. https://arxiv.org/abs/2405.15682
  y_t = (1-beta) z_t + beta x_t;  z_{t+1} = z_t - eta grad f(y_t);  x_{t+1} = (1-c_{t+1}) x_t + c_{t+1} z_{t+1}, c_t = 1/t.
Gradient is evaluated at an interpolation between the running average x (the "prediction of where we end up")
and the raw iterate z; beta=0 is Polyak-Ruppert, beta=1 is primal averaging. Won AlgoPerf 2024 self-tuning track.
Optimizer analogue. Already an optimizer-side self-prediction: the gradient is queried at the forecast point.

### 4.7 Nesterov / optimistic / extra-gradient. Sutskever et al., ICML 2013 (NAG as momentum with gradient at the
look-ahead point theta + mu*v). Optimistic gradient (Popov 1980): x_{k+1} = x_k - eta(2F(x_k) - F(x_{k-1})),
i.e. linear extrapolation of the gradient sequence; extragradient (Korpelevich 1976) evaluates at a predicted
next point. EMA-Nesterov (arXiv 2605.25395, 2026): the one-step difference used as the look-ahead direction is
too noisy in deep learning; replace it with an EMA of parameter updates (low-frequency trend); competitive on
LM pretraining with Adam/SOAP/Muon.

### 4.8 Anderson acceleration for DL — Pasini et al., 2021. https://arxiv.org/abs/2110.14813; Damped Anderson
mixing for deep RL — Sun et al., NeurIPS 2021, https://arxiv.org/abs/2110.08896.
Fits a linear model of the fixed-point residual sequence and extrapolates; unstable under mini-batch noise
(oscillations scale inversely with batch), fixed by adaptive moving averaging with a relative-std trigger;
in RL needs damping + a non-expansive (MellowMax) operator.

### 4.9 Learned / analytic weight forecasting.
- Introspection — Sinha et al., 2017. https://arxiv.org/abs/1704.04959: a net trained on MNIST weight
  histories predicts future weights for CIFAR/ImageNet; faster convergence.
- Weight Nowcaster Networks — Jang et al., ICML 2023. https://proceedings.mlr.press/v202/jang23b.html:
  periodically forecast near-future weights and skip steps; wall-clock savings across tasks.
- NiNo — Knyazev et al., ICLR 2025. https://arxiv.org/abs/2409.04434: graph-based nowcaster over neuron
  connectivity; up to 50% fewer Adam steps in vision/language.
- Weight prediction boosts AdamW — Guan, 2023. https://arxiv.org/abs/2302.00195: run AdamW's own rule s steps
  ahead, take gradients at the predicted weights.
- Gradient Flow Matching — Shou, Ding, Gao, 2025. https://arxiv.org/abs/2505.20221: optimizer-aware vector
  field forecasting weight trajectories to convergence.
- Leap+Verify — 2026. https://arxiv.org/abs/2602.19580: speculative K-step weight prediction with a held-out-loss
  verify/rollback. KEY NEGATIVE RESULT: Adam-moment extrapolation "fails catastrophically" (predicted loss
  100-10,000x actual) at GPT-2 124M and Qwen 1.5B; finite-difference linear/quadratic predictors accepted 24-37%
  of K=5 leaps only in stable/transition regimes detected by an activation-cosine Lyapunov proxy.
Failure mode common to all: extrapolation beyond the horizon where the trajectory is linear; must be gated.

### 4.10 Kalman-filter optimizers.
- Ollivier, Online natural gradient as a Kalman filter, EJS 2018. https://arxiv.org/abs/1703.00209: extended KF
  on a static parameter = online natural gradient; a dynamics model on the parameter is exactly the missing piece.
- KOALA — Davtyan et al., AAAI 2022. https://arxiv.org/abs/2107.03331: loss as a noisy observation of a reference
  optimum, parameter dynamics model can encode momentum/Adam; KOALA++ (2025, arXiv 2506.04432) with
  structured gradient covariance.
Optimizer analogue. The Kalman predict step IS a self-forecast of the parameter; the update step trusts it in
proportion to the forecast's covariance vs observation noise.

### 4.11 Grokfast — Lee et al., 2024. https://arxiv.org/abs/2405.20233
  mu <- alpha*mu + (1-alpha)*g;  g_hat = g + lambda*mu.
Treat per-parameter gradient history as a signal; amplify the slow (generalizing) component; >50x faster
grokking; synergy with weight decay. Failure: lambda too large explodes; requires weight decay.

### 4.12 Noisy quadratic model — Zhang et al., NeurIPS 2019. https://arxiv.org/abs/1907.04164
Analysis tool for bias/variance of SGD/momentum/preconditioning/averaging; averaging extends the critical batch
size. Use this model to analyze any proposed forecast-target rule before running it.

### 4.13 Streaming / continual evidence for weight EMA.
Soutif-Cormerais et al., CoLLAs 2023, https://arxiv.org/abs/2306.16817: EMA of weights at test time drastically
raises performance and stability (reduces the stability gap) in online continual learning. CoMA/CoFiMA
(Marouf et al., ECCV 2024, https://arxiv.org/abs/2312.08977): continual (Fisher-weighted) model averaging.

---------------------------------------------------------------------------------------------------
## Thread 5. Prospective / predictive neurons (brief)
---------------------------------------------------------------------------------------------------

### 5.1 Urbanczik & Senn, Neuron 81(3):521-528, 2014. Learning by the dendritic prediction of somatic spiking.
Plasticity minimizes the mismatch between somatic firing and the dendritic prediction of it:
  dW ∝ (S_soma - phi(V_dend)) * dendritic eligibility; one rule covers supervised/unsupervised/RL depending on
what drives the soma.

### 5.2 Brea, Gaal, Urbanczik, Senn, PLoS CB 2016. Prospective coding by spiking neurons. Neurons learn to fire
ahead of their delayed input: output encodes the predicted future of the input.

### 5.3 Haider et al., NeurIPS 2021. Latent Equilibrium. https://arxiv.org/abs/2110.14549. Neuron and synapse
dynamics derived from a prospective energy on generalized position and momentum; phase-advanced output
u_breve = u + tau du/dt makes inference quasi-instantaneous independent of depth.

### 5.4 Senn et al., eLife 2024. A neuronal least-action principle for real-time learning in cortical circuits.
https://elifesciences.org/articles/89674
Prospective (discounted-future) voltage u_tilde(t) = (1/tau) int_t^inf u(t') exp(-(t'-t)/tau) dt', so
u = u_tilde - tau du_tilde/dt. Lagrangian
  L = 1/2 sum_i ( u_i - sum_j W_ij r_bar_j )^2 + beta/2 sum_o (u*_o - u_o)^2,
action A = int L[u_tilde, du_tilde/dt] dt; Euler-Lagrange gives the voltage dynamics; plasticity
  dW/dt ∝ e_bar * r_bar^T,  e_bar_i = u_i - sum_j W_ij r_bar_j
with rates r = rho(u) + tau d rho/dt (prospective). The error is between the ACTUAL somatic state and the
PROSPECTIVE (predicted) dendritic state; because outputs are phase-advanced, look-ahead and low-pass cancel and
error propagates with no layer delay ("prospective configuration" made real-time).
Optimizer analogue. Define the optimizer's prospective state theta_tilde = theta + tau * dtheta/dt (a look-ahead
of the trajectory) and minimize the mismatch between the realized update and the prospective one.

---------------------------------------------------------------------------------------------------
## Thread 6. Dense vs sparse targets
---------------------------------------------------------------------------------------------------
Covered by 3.11 (Menon: variance reduction iff teacher closer to p* than the label), 3.12 (Hinton: more bits per
case, lower gradient variance), 3.13 (Yuan: largely regularization), 1.10 (Balestriero-LeCun: pixel targets are
dominated by uninformative variance), 1.7 (I-JEPA: latent targets drop irrelevant detail), 2.3 (Daley:
averaged bootstraps strictly lower variance at equal contraction). The consistent formal content is:
"dense" = "closer to the conditional expectation of the noisy target"; the benefit is variance reduction and
the cost is the bias ||target - E[noisy target]||^2, and it is a win only when the former exceeds the latter.

---------------------------------------------------------------------------------------------------
## SYNTHESIS
---------------------------------------------------------------------------------------------------

(a) Recurring mechanisms by which "learn toward your own prediction" works and does not collapse

1. Conditional-expectation denoising. A prediction is useful exactly when it is closer to E[target | input] than
   the realized sample is: Menon et al. 2021 (Lemma 1 / Prop. 3: variance drops, bias ||p^t - p*||^2 is the
   price); Hinton 2015; Daley et al. 2024 for bootstrapped returns; Polyak-Juditsky 1992 for iterates.
2. Two-timescale asymmetry. The predictor must be (near-)optimal fast; the predicted quantity must not receive
   gradient from the prediction loss and should move slowly (semi-gradient + EMA). Tang et al. 2023: with fast
   predictor + stop-gradient, Phi^T Phi is exactly conserved (no collapse); Ni et al. 2024 same for EMA targets;
   BYOL tau=0 gives 0.3% (collapse), tau=1 gives 18.8% (no learning), tau~0.99 optimal.
3. Lag as stabilization of an ill-conditioned operator. A frozen or slow target copy turns a non-contractive
   update into a contraction for finite k (Fellows et al. 2023, C(alpha,k) < 1; Baird converges at k >= 500).
4. Spectral low-pass / early-learning. Fitting your own prediction attenuates small-eigenvalue directions first
   (Mobahi et al. 2020: coefficients prod d_k/(c+d_k)); networks learn signal directions before noise (Arpit
   2017), so an EMA of past predictions is cleaner than the label (ELR: 91% on CIFAR-10 at 40% noise). Same
   mechanism causes collapse after ~t_bar rounds.
5. Variance-reduced fixed point by stepping toward a short-horizon forecast of yourself. Lookahead: phi <- phi +
   alpha(theta_k - phi) has a strictly smaller noisy-quadratic variance fixed point for alpha in (0,1); Schedule-
   Free queries the gradient at the interpolation between the average and the iterate.
6. Gating / verification instead of asymmetry. LeJEPA removes stop-grad and EMA with a distributional
   regularizer; Leap+Verify accepts forecasts only when a held-out loss confirms them and shows Adam-moment
   extrapolation diverges by 100-10,000x without a gate; ObGD in stream-x bounds overshoot per sample.
7. Prospective state. Neurons minimize the mismatch to their own look-ahead u + tau du/dt (Senn 2024, Haider
   2021); errors are computed against the predicted, not the delayed, state.

(b) Optimizer-side versions (per-parameter), with collapse risk

1. Forecast-blended gradient (Menon/TD-lambda analogue). Keep m_t = EMA of past updates as the forecast of the
   next update; apply u_t = (1-lam) g_t + lam m_t, with lam chosen per parameter by the empirical test
   E||m - g||^2 < Var[g] (e.g. from Adam's second moment minus the forecast's squared error). Collapse: lam -> 1
   is pure momentum extrapolation and runs away (Leap+Verify); break it with the variance test and a cap on lam.
2. Two-timescale predictor over parameter state (Tang/Ni analogue). Slow state theta_bar (EMA), fast linear
   predictor P (per-parameter scalar or low-rank) fit to map (theta_bar, m) -> next theta_bar; step theta toward
   sg(P(theta_bar)) plus the raw gradient. Collapse risk: predictor too slow / theta_bar too fast -> all
   coordinates drift together (rank loss). Monitor the covariance of the update ensemble across coordinates
   (the optimizer's Phi^T Phi).
3. Partially-fitted target iterate (Fellows analogue). Freeze theta_bar for k steps, run k inner steps on
   L(theta) + (rho/2)||theta - theta_bar - hat_delta||^2 where hat_delta is the forecast displacement; then copy.
   Collapse: k too small diverges, k too large freezes; choose k so the inner problem is nearly solved.
4. Spectral self-distillation of updates (Mobahi analogue). Grokfast-style g_hat = g + lambda*EMA(g) is already
   an eigen-reweighting; the optimizer-side JEPA version is to re-fit the update to its own smoothed history a
   bounded number of "rounds" per phase. Collapse: over-filtering underfits (only high-curvature directions
   survive); bound the rounds or reset the filter on loss-plateau detection.
5. Lookahead-toward-forecast (Zhang/Defazio analogue). Replace Lookahead's realized fast-weight endpoint by a
   forecast: phi <- phi + alpha (theta_hat_k - phi) with theta_hat_k = phi + k * EMA(displacement), verified
   against the realized theta_k when available. Collapse: alpha -> 1 with a bad forecast; keep alpha < 1 and
   fall back to the realized endpoint when the forecast error exceeds the realized step's variance.
6. Kalman parameter filter (Ollivier/KOALA analogue). State = (theta, velocity); predict step = the forecast;
   update step weights the raw gradient by K = P_pred / (P_pred + R) with R the gradient noise (Adam v).
   Collapse: process-noise underestimation makes K -> 0 (ignores data). Keep a floor on process noise.
7. Prospective iterate (Senn analogue). Compute the gradient at theta_tilde = theta + tau * m (NAG/optimistic
   form) but with tau adapted to the EMA-of-updates rather than the last step (EMA-Nesterov). Collapse: none of
   the constant type; failure is oscillation when tau exceeds the trajectory's linear horizon.

(c) Evidence on noisy streaming data
- Strong: Polyak-Juditsky (asymptotically optimal averaging on an i.i.d. stream); Morales-Brotons 2024 (weight
  EMA +9.7pp under 40% label noise); ELR / temporal ensembling / Mean Teacher (own-prediction targets resist
  label noise); Soutif-Cormerais 2023 (weight EMA fixes the stability gap in online continual learning);
  stream-x (lambda-return + bounded-overshoot step makes streaming deep RL stable with no target net).
- Weak / negative for extrapolation: Anderson acceleration destabilizes under mini-batch noise (Pasini 2021);
  Adam-moment weight extrapolation fails catastrophically without verification (Leap+Verify 2026); Nesterov's
  one-step look-ahead direction is too noisy in deep learning (EMA-Nesterov 2026).
- Untested: a learned per-parameter predictor of the update stream (WNN/NiNo are periodic batch forecasters on
  supervised training, not streaming RL); Kalman optimizers have not been evaluated on nonstationary RL.
Net: the evidence supports "step toward a lagged average of yourself" (mechanisms 1-5) on noisy streams and
argues against "step toward a linear extrapolation of yourself" unless gated (mechanism 6).
