# Energy-based learning frameworks and what they imply at the optimizer level

Literature survey, September 2026. Scope: energy-based learning (EBMs, JEPA, equilibrium propagation, predictive coding, forward-forward, least-action/Lagrangian learning, and thermodynamic/Langevin views of SGD vs Adam), 2005-2026 with emphasis on 2021-2026. For each paper: citation, mechanism, key quantitative claim, and a one-line implication for a per-parameter streaming optimizer that would replace Adam.

Notation is plain text: E = energy, z = latent activities, W/theta = weights, eta/alpha = step sizes, g = gradient, H = Hessian, I(z) = Fisher information, T = temperature. "Verified" means the abstract/body was fetched and read in this session; two items flagged as partially verified were confirmed only via search snippets.

---

## Thread 1. EBMs and the "inference-then-learning" pattern

### 1.1 Krotov & Hopfield 2016, "Dense Associative Memory for Pattern Recognition", NeurIPS 29
https://proceedings.neurips.cc/paper/2016/hash/eaae339c4d89fc102edd9dbdb6a28915-Abstract.html
Mechanism: replaces the quadratic Hopfield energy with E = -sum_mu F(xi_mu . sigma) for a rapidly growing F (polynomial of degree n, or exponential). Retrieval is descent on E; the same dynamics is dual to a one-hidden-layer feedforward net with activation F'. Storage capacity grows as N^(n-1) instead of ~0.14N. Interpolates between "feature matching" (small n) and "prototype" (large n) regimes.
Quantitative claim: capacity scales polynomially/exponentially in N; MNIST classification experiments show higher-n models are more robust to adversarial/noisy inputs (prototype regime).
Optimizer implication: an energy whose minima are "prototypes" gives retrieval that is robust to input perturbation; for an optimizer this is the ancestor of "relax to a fixed point, then update with a local rule".

### 1.2 Ramsauer et al. 2020/2021, "Hopfield Networks is All You Need", ICLR 2021
https://arxiv.org/abs/2008.02217 (semanticscholar / scispace entries verified)
Mechanism: continuous modern Hopfield energy E(xi) = -lse(beta, X^T xi) + 1/2 xi^T xi + const. The concave-convex-procedure update xi_new = X softmax(beta X^T xi) is exactly transformer attention; one update retrieves with exponentially small error; three kinds of fixed points (global average, metastable subset averages, single-pattern).
Quantitative claim: exponential storage in dimension d; retrieval in one step; used as a drop-in Hopfield layer.
Optimizer implication: "one energy-descent step is exact retrieval" means fixed-point relaxation need not be iterative if the energy is designed for it; also shows that beta (inverse temperature) selects between averaging and memorization, an analogue of the noise-temperature knob in Thread 6.

### 1.3 LeCun 2022, "A Path Towards Autonomous Machine Intelligence" (v0.9.2, OpenReview) and Dawid & LeCun 2024, "Introduction to latent variable energy-based models", J. Stat. Mech. 2024
https://openreview.net/pdf?id=BZ5a1r-kVsf ; https://arxiv.org/abs/2306.02572
Mechanism: an EBM assigns scalar F(x,y); inference is argmin_y F(x,y). Latent-variable EBMs infer z by energy minimization F(x,y) = min_z E(x,y,z). Training is either contrastive (push up energy of negatives) or regularized/architectural: shape the energy so that low-energy volume is limited (e.g., limit the information content of z, VICReg-style variance/covariance terms). JEPA: encode x and y, predict s_y from s_x (plus latent z); energy = prediction error in embedding space; a regularizer on the embedding prevents collapse.
Quantitative claim: position paper; no benchmarks.
Optimizer implication: separates "state inference by energy descent" from "parameter learning", and argues that collapse is a property of the energy's volume of low-energy states, which must be controlled by a regularizer rather than by heuristics.

### 1.4 Assran et al. 2023, "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (I-JEPA), CVPR 2023
https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html
Mechanism: context encoder + predictor predict representations of masked target blocks produced by an EMA target encoder; loss in embedding space; no pixel reconstruction, no hand-crafted augmentation. Collapse prevented by the asymmetric EMA teacher + stop-gradient (heuristics).
Quantitative claim: ViT-H/14 on ImageNet-1k linear probe competitive with augmentation-based methods, with fewer epochs than MAE-style methods.
Optimizer implication: the "EMA teacher + stop-gradient" pair is an optimizer-level trick (a slow-timescale copy of the parameters supplies targets); LeJEPA's later result that it can be removed matters for streaming settings where a lagging copy is a liability.

### 1.5 Bardes et al. 2024, "V-JEPA" (Revisiting feature prediction for learning visual representations from video); Assran et al. 2025, "V-JEPA 2", arXiv 2506.09985
https://arxiv.org/abs/2506.09985
Mechanism: same predictive-in-embedding-space objective on video with high masking ratios; V-JEPA 2 adds an action-conditioned predictor for planning by energy minimization over action sequences.
Quantitative claim: V-JEPA 2 (1B+ params, 1M+ hours video) reports SOTA motion understanding and zero-shot robot planning by minimizing predictor energy in latent space.
Optimizer implication: planning-by-energy-minimization is the "inference" phase of a latent EBM applied to actions: the same relax-then-act pattern at decision time.

### 1.6 Balestriero & LeCun 2025, "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics", arXiv 2511.08544 (v3, Nov 2025)
https://arxiv.org/abs/2511.08544
Mechanism (read carefully):
- Theory: for downstream linear probes, an anisotropic embedding covariance inflates both bias (Lemma 1) and variance (Lemma 2) of least-squares estimators; for nonlinear probes (kNN, kernels) Theorem 1 shows the isotropic Gaussian is the unique minimizer of integrated squared bias among distributions with a fixed scalar covariance constraint. Hence isotropic Gaussian is the "optimal" embedding law when the downstream task is unknown.
- SIGReg (Sketched Isotropic Gaussian Regularization): instead of matching a d-dimensional distribution, project embeddings onto |A| random unit directions a (256-1024 per step) and, for each, apply a univariate goodness-of-fit test between projected embeddings and N(0,1). The test is Epps-Pulley: EP = N * integral |phi_hat(t) - phi(t)|^2 w(t) dt with phi_hat the empirical characteristic function (1/n) sum exp(i t x_j) and phi = exp(-t^2/2), w a Gaussian weight. SIGReg = (1/|A|) sum_a T({a^T f_theta(x_n)}). Linear cost in N and d; Theorem 4: bounded gradients regardless of input distribution; all_reduce-friendly.
- Loss: L = lambda/V sum_v SIGReg(z_{.,v}) + (1-lambda)/B sum_n ||mu_n - z_{n,v'}||^2 where mu_n is the mean global-view embedding; lambda ~ 0.05 is the single hyperparameter.
- Why heuristics vanish: collapse is dimensional (covariance rank drops) or total; a full-distribution isotropy constraint forbids both by construction, so stop-gradient, EMA teacher, predictor asymmetry, register tokens, and schedulers are not needed.
Quantitative claim: ViT-H/14 79% frozen linear probe on ImageNet-1k; stable across 60+ architectures / 10+ datasets; stable at 1.8B (ViT-g); Spearman 85-99% between training loss and downstream accuracy (label-free model selection); ~50 lines of code; in-domain pretraining on Galaxy10 beats DINOv2/v3 transfer.
Optimizer implication: a cheap, sketch-based (random-projection, characteristic-function) isotropy penalty on a stream of vectors is a drop-in regularizer; for an optimizer it suggests regularizing the second-moment geometry of the *update or gradient stream* toward isotropy (whitening) rather than per-coordinate rescaling, and replacing slow-copy heuristics (EMA targets) with a distributional constraint.

---

## Thread 2. Equilibrium propagation, dual propagation, "energy minimization in activity space gives the gradient"

### 2.1 Scellier & Bengio 2017, "Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation", Frontiers Comp. Neurosci. 11:24
https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00024/full
Mechanism: total energy F(theta, beta, s) = E(theta, s) + beta C(s, y). Free phase: relax s to s0 = argmin E. Nudged phase: relax under F with small beta to s_beta. Parameter update
  Delta theta ∝ (1/beta) [ dE/dtheta(s_beta) - dE/dtheta(s0) ]
and the theorem: lim_{beta->0} of this equals -dJ/dtheta, the gradient of the cost at the free fixed point (equivalent to recurrent backprop / implicit differentiation). With Hopfield energy E = -1/2 sum W_ij rho(s_i) rho(s_j) + ..., the update is a contrastive Hebbian difference rho_i rho_j |_beta - rho_i rho_j |_0.
Quantitative claim: MNIST, 1-3 hidden layers, 0% train error and 2-3% test error.
Optimizer implication: gradient = finite difference of a *local* quantity between two equilibria; the effective learning rate is scaled by 1/beta and the estimator is biased O(beta) but the bias is in the direction of a proximal/trust-region solution (see Millidge 2022, Innocenti 2023).

### 2.2 Laborieux & Zenke 2022, "Holomorphic Equilibrium Propagation Computes Exact Gradients Through Finite Size Oscillations", NeurIPS 35
https://arxiv.org/abs/2209.00530
Mechanism: extend the energy to a holomorphic function of complex activities; drive the teaching signal around a circle of finite radius in the complex plane; by Cauchy's integral formula the exact gradient is the first Fourier coefficient of the oscillating dE/dtheta, so finite-amplitude nudging gives exact gradients with no separate phases (continuous-time oscillation).
Quantitative claim: first EP result on ImageNet 32x32 matching backprop; gradient estimate is robust to finite nudging and to noise; deeper networks benefit from *larger* finite teaching amplitudes.
Optimizer implication: averaging a periodically modulated local signal (lock-in detection) cancels the finite-step bias of a two-point finite-difference estimator; a streaming optimizer can obtain unbiased curvature-aware directions by demodulating a sinusoidally perturbed error stream instead of differencing two snapshots.

### 2.3 Hoier, Staudt & Zach 2023, "Dual Propagation: Accelerating Contrastive Hebbian Learning with Dyadic Neurons", ICML 2023 (PMLR 202)
https://arxiv.org/abs/2302.01228
Mechanism: each neuron carries two states (s+, s-); their mean encodes activity and their difference encodes error (the "error/activity duality"). The lifted energy is a min-max (saddle) problem solvable layerwise in closed form in a single inference pass, so no two-phase relaxation is needed.
Quantitative claim: matches backprop in accuracy and runtime on CIFAR and ImageNet32x32, versus >100x slowdown for two-phase CHL/EP.
Optimizer implication: the "positive" and "negative" states can be co-located in one pass; for an optimizer, keep two running estimates of each parameter's state (nudged and free) whose difference is the update; this is structurally a symmetric finite difference around the current point.

### 2.4 Scellier, Ernoult, Kendall & Kumar 2023, "Energy-based learning algorithms for analog computing: a comparative study", NeurIPS 2023
https://arxiv.org/abs/2312.15103
Mechanism: compares CL, EP (positive, negative, centered), coupled learning on deep convolutional Hopfield networks. Centered EP uses two nudges of opposite sign (+beta, -beta) and differences them: a symmetric finite-difference gradient estimator with O(beta^2) bias instead of O(beta).
Quantitative claim: negative perturbations beat positive; centered EP best and the gap widens with task difficulty; new SOTA for DCHNs on MNIST/F-MNIST/SVHN/CIFAR-10/CIFAR-100 with 13.5x faster simulation (16-bit).
Optimizer implication: when the update is a finite difference of local statistics, use a symmetric (centered) difference; bias, not variance, is what limits finite-nudging estimators.

### 2.5 Millidge, Song, Salvatori, Lukasiewicz & Bogacz 2022, "Backpropagation at the Infinitesimal Inference Limit of Energy-Based Models: Unifying PC, EP and CHL", arXiv 2206.02629
https://arxiv.org/abs/2206.02629
Mechanism: at a free-phase equilibrium of any EBM, an infinitesimal nudge produces activity changes proportional to backprop errors ("infinitesimal inference limit"); PC, EP and CHL differ only in energy function and nudging scheme and all reduce to BP there, with Hebbian-form weight updates. Away from the limit (finite nudging) the approximation degrades and the update becomes something else (the trust-region/proximal object of Thread 3).
Quantitative claim: theoretical; experiments confirm gradient alignment falls off as nudge grows.
Optimizer implication: "natural" EBM learning is backprop plus a finite-step correction; the correction is what the rest of this survey characterizes.

### 2.6 Peters & Talatchian 2025, "Harnessing uncertainty when learning through Equilibrium Propagation in neural networks", arXiv 2503.22810
https://arxiv.org/abs/2503.22810
Mechanism: trains EP networks with finite parameter/activity uncertainty (hardware noise) and samples the network according to the CLT; noise acts as a regularizer.
Quantitative claim: a critical uncertainty threshold, independent of dataset, beyond which learning fails; convergence and MNIST-variant accuracy *improve* for finite noise, with optimum near the critical level.
Optimizer implication: relaxation-based learning tolerates and benefits from a tuned amount of injected noise; a streaming optimizer can treat gradient noise as a controlled temperature, not something to average away.

### 2.7 Kubo, Delanois & Bazhenov 2025, "Toward Lifelong Learning in Equilibrium Propagation: Sleep-like and Awake Rehearsal for Enhanced Stability", arXiv 2508.14081
https://arxiv.org/abs/2508.14081
Mechanism: EP-trained multilayer RNNs with a sleep-like replay consolidation (unsupervised free-phase replay after each task) plus awake rehearsal.
Quantitative claim: EP+SRC matches or exceeds BPTT-trained networks on Fashion-MNIST, KMNIST, CIFAR-10 and ImageNet in sequential-task forgetting; combining sleep and awake replay is strongest.
Optimizer implication: an energy model supplies its own generative replay from the free phase; the optimizer can consolidate by descending the energy on self-generated states (no stored data) between tasks.

### 2.8 Pourcel, Basu, Ernoult & Gilra 2025/2026, "Lagrangian-based Equilibrium Propagation: generalisation to arbitrary boundary conditions & equivalence with Hamiltonian Echo Learning", arXiv 2506.06248; and Lopez-Pastor & Marquardt 2023, "Self-Learning Machines Based on Hamiltonian Echo Backpropagation", PRX 13, 031020
https://arxiv.org/abs/2506.06248 ; https://arxiv.org/abs/2103.04992
Mechanism: EP over trajectories instead of fixed points (Generalized Lagrangian EP); boundary conditions at the trajectory ends pick the algorithm; HEB/RHEL (time-reverse the Hamiltonian dynamics, inject a small error, the echo carries the gradient) is a special case. Forward-only, local, two passes.
Quantitative claim: theoretical equivalence; HEB shows autonomous parameter update in time-reversible physical systems.
Optimizer implication: the gradient over a time window can be computed by running the *same* dynamics backwards from the end state with a small error kick; an online rule can approximate this by a reversed replay of the recent state buffer instead of storing activations.

---

## Thread 3. Predictive coding as an optimizer

### 3.1 Millidge, Tschantz & Buckley 2022, "Predictive Coding Approximates Backprop Along Arbitrary Computation Graphs", Neural Computation 34(6):1329-1368
https://arxiv.org/abs/2006.04182
Mechanism: PC energy F = sum_l 1/2 ||eps_l||^2 with eps_l = z_l - f_l(W_l z_{l-1}). Inference: dz_l = -eta dF/dz_l = -eta (eps_l - f'(...) W_{l+1}^T eps_{l+1}). At the fixed point (with the "fixed-prediction" assumption) eps_l equals the BP error delta_l, so dF/dW_l = eps_{l+1} f'(.) z_l^T is the BP gradient. Extends to CNNs, RNNs, LSTMs.
Quantitative claim: PC converges to exact BP gradients asymptotically and rapidly in practice; matches BP accuracy on the tested architectures.
Optimizer implication: the weight update is always rank-1 per sample: (local error) x (local input)^T; what the "optimizer" controls is how the local error is computed (relaxed vs. propagated).

### 3.2 Millidge, Salvatori, Song, Bogacz & Lukasiewicz 2022, "Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation?", IJCAI-22 pp. 5538-5545
https://www.ijcai.org/proceedings/2022/0774.pdf
Mechanism: survey; frames PC as variational inference on a hierarchical Gaussian model; catalogs "natural PC" advantages (locality, parallelism, no phases if incremental, better in online/continual settings).
Quantitative claim: survey.
Optimizer implication: the claims of online/continual robustness are attributed to the relaxation phase making all weight updates *mutually consistent* for the current sample.

### 3.3 Song, Millidge, Salvatori, Lukasiewicz, Xu & Bogacz 2024, "Inferring neural activity before plasticity as a foundation for learning beyond backpropagation", Nature Neuroscience 27:348-358
https://www.nature.com/articles/s41593-023-01514-1 ; PMC7615830
Mechanism: "prospective configuration": with input and target clamped, relax hidden activities z <- z - gamma dE/dz to convergence z*; only then update Delta w = -alpha dE/dw |_{z*}, which for PC is Delta W_l = alpha eps_{l+1} f(z_l)^T. Key concept: target alignment = cosine between (target - output) and (output after update - output). BP updates layers independently and their side effects interfere; PC first finds the activity configuration that all updates will jointly produce, so updates do not fight each other.
Quantitative claims: target alignment stays ~0.95+ with depth for PC vs. falling to ~0.6 at 25 layers for BP; lower test error with fewer samples per class on FashionMNIST; online learning (batch size 1) markedly better than BP; Q-learning on Acrobot/MountainCar/CartPole better rewards; alternating two 5-class tasks: less forgetting and faster relearning; concept-drift (label shuffle) "particularly large" advantage; 15-layer net converges faster.
Optimizer implication: the single most important qualitative claim: BP's per-parameter updates are individually correct but *jointly* inconsistent (interference); a relaxation step that makes updates consistent before applying them is what buys online/continual robustness, and batch averaging in BP/Adam is partly a substitute for this consistency.

### 3.4 Innocenti, Singh & Buckley 2023, "Understanding Predictive Coding as an Adaptive Trust-Region Method", arXiv 2305.18188 (ICML 2023 Workshop on Localized Learning; also titled "...as a Second-Order Trust-Region Method" on OpenReview)
https://arxiv.org/abs/2305.18188
Exact claims (read from the PDF):
- Energy F = sum_l 1/2 ||Pi_l^{1/2}(z_l - f_l(W_l z_{l-1}))||^2 (Eq. 1); inference Delta z = -eta dF/dz (Eq. 2); learning Delta W = -alpha dF/dW at z* (Eq. 3).
- Second-order Taylor expansion of F around the feedforward activities z_t (Eq. 5): F(z) = L(z_t) + g_L(z_t)^T Delta z + 1/2 Delta z^T I(z_t) Delta z + O(Delta z^3), where g_L is the loss gradient and I(z_t) is the Fisher information of the target under p(y|z). This is a trust-region subproblem (Eq. 4: argmin f(z) s.t. Delta z in {Delta z^T A Delta z <= r}) with a linear model of the loss and an *adaptive, second-order* geometry A = I(z_t). Its solution is z* ~ z_t - I(z_t)^{-1} g_L(z_t) (Eq. 6): a damped Gauss-Newton/natural-gradient step *in activity space*.
- Weight gradient at equilibrium (Eq. 7): dF/dW|_{z*} ~ (dz_t/dW) I(z_t)^{-1} g_L(z_t) + g_L(W). So the PC weight update interpolates between BP's loss gradient g_L(W) and the trust-region inference solution mapped back to weight space through the Jacobian dz_t/dW. In directions of high Fisher information (high latent variance) the update is biased toward the TR solution; in directions of low Fisher information it looks like GD.
- Theorem A.3: for any non-degenerate 1-hidden-unit linear MLP y = w2 w1 x, GD on the equilibrated energy escapes the origin (sign-flip) saddle faster than GD on the MSE loss, because both Hessian eigenvalues at the saddle are smaller for the energy: lambda(H_L) = +/- xy vs lambda(H_F*) = 1/2(-y^2 +/- sqrt(y^4 + 4 x^2 y^2)); i.e., less attracted along the stable direction and more repelled along the unstable one.
- Theorem A.4: the minima of the equilibrated energy are flatter than the loss minima (lambda_max(H_F*) = lambda_max(H_L)/(1+w2^2)), so PC converges slower near a minimum but is more robust to i.i.d. Gaussian weight perturbations there (Figure 6: perturbed-MSE lower for PC than BP; noise N(0, 0.5)). The paper explicitly says this "could be important in more biological, online settings".
- Figure 2: cosine similarity to the optimal weight direction over the first 5 batches is highest for a trust-region Newton method -(H + 2I)^{-1} grad, then PC, then SGD.
- Figure 3: on deep chains with saddle-inducing (tanh-type) activations PC trains "significantly faster" than BP; no such advantage with ReLU (which breaks the sign-flip symmetry).
Optimizer implication: PC = a Gauss-Newton-like step computed in *activity* space (dimension = width, cheap) and pushed into weight space through the local Jacobian; a streaming optimizer can get second-order behaviour by preconditioning the *error signal per layer* with an inverse Fisher of that layer's output (a small matrix), rather than per-parameter second moments as in Adam.

### 3.5 Innocenti, Achour, Singh & Buckley 2024, "Only Strict Saddles in the Energy Landscape of Predictive Coding Networks?", NeurIPS 2024
https://arxiv.org/abs/2408.11979
Mechanism: Theorem 1: for deep linear nets the equilibrated energy is a rescaled MSE, F* = (1/2N) sum_i (y_i - W_{L:1} x_i)^T S^{-1} (y_i - W_{L:1} x_i), with S = I + sum_{l=2}^{L} W_{L:l} W_{L:l}^T. Theorem 2: the origin, a non-strict (zero-Hessian) saddle of the MSE for depth > 1, is a strict saddle of F* independent of depth (the last diagonal Hessian block is non-zero). Conjecture, supported empirically, that all saddles of F* are strict.
Quantitative claim: 5-layer MLPs on MNIST/F-MNIST and a CNN on CIFAR-10 initialized at scale 5e-3 with lr 1e-3: PC escapes the origin saddle "substantially faster" than BP.
Optimizer implication: the relaxation divides the residual by (I + sum of products of downstream weights): an automatic, weight-dependent rescaling of the error that removes vanishing-gradient degeneracy; a streaming rule could apply an analogous per-layer whitening of the error by the downstream Jacobian Gram matrix.

### 3.6 Salvatori, Song, Yordanov, Millidge, Xu, Sha, Emde, Bogacz & Lukasiewicz 2022/2024, "A Stable, Fast, and Fully Automatic Learning Algorithm for Predictive Coding Networks" (incremental PC, iPC), arXiv 2212.00720 (ICLR 2024)
https://arxiv.org/abs/2212.00720
Mechanism: instead of T inference steps then one weight step, update z and W simultaneously every step (derived from incremental EM); no external phase control; convergence guarantees.
Quantitative claim: iPC beats original PC on a large set of image-classification benchmarks and on conditional and masked language models in test accuracy, efficiency and hyperparameter robustness, and matches BP on image classification.
Optimizer implication: the relaxation does not need to finish before weights move; interleaving one activity step with one weight step per sample is stable and *more* robust to hyperparameters, i.e., a streaming optimizer can keep a slowly-relaxing auxiliary state per layer that is nudged one step per sample.

### 3.7 Innocenti, Achour & Buckley 2025, "muPC: Scaling Predictive Coding to 100+ Layer Networks", NeurIPS 2025
https://arxiv.org/abs/2505.13124
Mechanism: shows the standard PC parameterization is unscalable: the condition number of the activity Hessian grows with depth/width and training time, and feedforward initialization vanishes/explodes with depth. A Depth-muP parameterization fixes both and gives zero-shot transfer of both weight and activity learning rates across widths and depths.
Quantitative claim: residual PCNs up to 128 layers train reliably on MNIST/Fashion-MNIST with little tuning; learning-rate transfer across width and depth.
Optimizer implication: the inner relaxation is itself an ill-conditioned optimization; any optimizer built on it needs a parameterization (or preconditioner) that keeps the activity-space condition number O(1).

### 3.8 Alonso, Millidge, Krichmar & Neftci 2022, "A Theoretical Framework for Inference Learning", NeurIPS 2022
https://proceedings.neurips.cc/paper_files/paper/2022/hash/f242c4cba2467637256722cb679642bd-Abstract-Conference.html
Mechanism: inference learning (the PC training algorithm) closely approximates *implicit* SGD, the proximal update theta_{t+1} = theta_t - lr * grad L(theta_{t+1}), for small nudging and activity step sizes; explicit SGD (backprop) is the forward-Euler counterpart.
Quantitative claim: IL is stable across a wider range of learning rates, less sensitive to hyperparameters, converges faster at minibatch size 1, and matches BP with Adam at larger batches ("extensive simulations").
Optimizer implication: the relaxation phase effectively evaluates the gradient at the *post-update* point; a streaming optimizer can emulate this with a proximal/implicit step (e.g., one or two fixed-point iterations of theta' = theta - lr grad L(theta')), which is known to be stable for large lr and heavy-tailed noise.

### 3.9 Alonso, Krichmar & Neftci 2024, "Understanding and Improving Optimization in Predictive Coding Networks", AAAI 2024 38(10):10812-10820
https://arxiv.org/abs/2305.13562
Mechanism: (1) a PC circuit variant with less computation; (2) a memory-free optimizer for IL that avoids Adam; (3) theory on when IL is sensitive to second- and higher-order information, governed by gamma = ratio of activity to weight step size.
Quantitative claim: IL can reduce loss faster than BP per step and reaches good minima without Adam's memory.
Optimizer implication: the activity/weight step-size ratio is the knob controlling how much curvature the update uses; this is the EBM analogue of Adam's beta2/eps.

### 3.10 Mali, Salvatori & Ororbia 2024, "Tight Stability, Convergence, and Robustness Bounds for Predictive Coding Networks", arXiv 2410.04708
https://arxiv.org/abs/2410.04708
Mechanism: dynamical-systems analysis; PC inference is Lyapunov stable, so small random perturbations are contracted; PC approximates quasi-Newton updates (Hessian-structure analysis), closer to quasi-Newton than target propagation.
Quantitative claim: 9 theorems; PC converges in fewer iterations than BP under stated assumptions; no headline numbers in the abstract.
Optimizer implication: an update computed at a Lyapunov-stable fixed point of a contraction is automatically perturbation-robust; noise injected before the relaxation is attenuated by the contraction factor.

### 3.11 Ishikawa, Yokota & Karakida 2025, "Local Loss Optimization in the Infinite Width: Stable Parameterization of Predictive Coding Networks and Target Propagation", ICLR 2025
https://proceedings.iclr.cc/paper_files/paper/2025/file/7ab63a5314680e2f083cb288abeaeb8e-Paper-Conference.pdf
Mechanism: maximal-update parameterization for PC. Theorem 4.2 (linear nets, MSE): the equilibrium error at layer l is e*_l = (gamma_L/gamma_l) W_{L:l+1}^T (I + C_gamma(W))^{-1} (W_{L:1} x - y) with C_gamma = sum_i (gamma_L/gamma_{i-1}) W_{L:i} W_{L:i}^T; the output error is (I + C_gamma)^{-1} (f - y). So PC's gradient interpolates between first-order GD and a damped Gauss-Newton-target update depending on the inference step-size scaling; in the infinite-width limit with gamma_L = Theta(1) it reduces to first-order GD (Corollary 4.3), i.e., the second-order flavour is a finite-width / large-last-layer-step effect.
Quantitative claim: muP enables learning-rate transfer across widths for PC; cosine-similarity plots show PC gradient -> BP gradient as width grows.
Optimizer implication: the second-order preconditioning that PC provides is (I + J J^T)^{-1} applied to the output residual, with J the downstream Jacobian; this is Levenberg-Marquardt damping with unit damping, and the scaling of the last-layer inference step is what keeps it alive at scale.

---

## Thread 4. Forward-Forward and local goodness

### 4.1 Hinton 2022, "The Forward-Forward Algorithm: Some Preliminary Investigations", arXiv 2212.13345
https://www.cs.toronto.edu/~hinton/FFA13.pdf
Mechanism: each layer has a local objective: p(positive) = sigma(sum_j y_j^2 - theta) (Eq. 1), goodness = sum of squared ReLU activities before layer-norm; positive pass raises goodness on real data, negative pass lowers it on negative data; layer normalization passes only orientation, forcing the next layer to learn new features. The weight update is the local derivative of the logistic goodness loss: rank-1 in (input, activity) with a scalar sign/gain. Hinton notes that with layer-norm the update does not change the normalized output for that input, so simultaneous online updates in many layers do not interfere. Motivations: analog hardware (two forward passes remove A/D converters) and "mortal computation".
Quantitative claim: MNIST 1.36% test error (4 hidden layers x 2000 ReLU, 60 epochs) vs ~1.4% for backprop in the same permutation-invariant setting (1.1% with regularizers); 0.64% with a local receptive-field/top-down variant; CIFAR-10 with 2-3 hidden layers of 11x11 local receptive fields: FF slightly worse than BP, gap does not grow with depth, but BP reduces training error much faster.
Optimizer implication: a per-layer scalar objective turns the update into (scalar gain) x (rank-1 Hebbian term); the interesting streaming property is that layer-norm makes the update *orthogonal to the current output*, which is a built-in form of interference avoidance.

### 4.2 Terres-Escudero, Del Ser & Garcia Bringas 2024, "A Contrastive Symmetric Forward-Forward Algorithm (SFFA) for Continual Learning Tasks", CoLLAs 2024
https://arxiv.org/abs/2409.07387
Mechanism: split each layer into positive and negative neurons; fitness = activity of positive neurons / total activity, which symmetrizes the gradient between positive and negative examples and induces sparse, specialized units.
Quantitative claim: non-continual: MNIST 97.87 (SFFA) vs 98.25 (BP), Fashion-MNIST 89.39 vs 89.47 (FFA 85.75). Continual, Split-MNIST class-IL with replay: SFFA 90.03 vs BP 87.75; with GEM 69.73 vs 62.85; domain-IL replay 95.17 vs 94.16; but task-IL with EWC/SI SFFA is 3-4 points worse.
Optimizer implication: symmetric positive/negative gradients plus sparsity reduce forgetting under replay-style streams; a per-parameter rule could gate updates by a sparse "ownership" signal so that new data mostly touches under-used parameters.

(Note: "Signal Propagation" (Kohan, Rietman & Siegelmann 2023) was not fetched in this session and is therefore not counted; it is a single-forward-pass local method related to FF.)

---

## Thread 5. Least action / Lagrangian / Hamiltonian / thermodynamic views of learning over time

### 5.1 Betti & Gori 2016, "The principle of least cognitive action", Theoretical Computer Science 633:83-99
https://www.sciencedirect.com/science/article/pii/S0304397515005526
Mechanism: weights w(t) are Lagrangian coordinates. Kinetic energy K = 1/2 sum_i m_i (T w_i)^2 with T a differential operator (velocity, or higher-order); potential V(t, w) = loss/constraint penalty; dissipation via the factor psi(t) = exp(theta t) multiplying the Lagrangian. With T = D (first derivative) and gamma = -1 the Euler-Lagrange equations are D^2 w_i + theta D w_i + dV/dw_i = 0 (Eq. 10): damped oscillators = heavy-ball / momentum GD, with theta the friction. With T = alpha_0 + alpha_1 D + alpha_2 D^2 the EL equations are fourth order (Eq. 11) and Routh-Hurwitz stability forces a sign flip of gamma ("gamma sign flip"). Learning is dissipative: potential energy is converted to kinetic energy of the weights and dissipated; "boundary" and "bartering" energies account for the balance. Theorem 3.1: stationarity under fixed endpoint conditions; the paper shows that on supervised pairs the EL equations collapse to classic gradient descent in the strong-dissipation limit.
Quantitative claim: theoretical.
Optimizer implication: momentum SGD *is* the least-action solution with kinetic penalty on weight velocity and exponential dissipation; the action penalizes *change* of weights over time, not their value, and the dissipation rate is the time-weighting that makes recent data matter more.

### 5.2 Betti & Gori 2019, "Least Action Principles and Well-Posed Learning Problems", arXiv 1907.02517
https://arxiv.org/abs/1907.02517
Mechanism: proves existence of an actual minimum (not just a stationary point) of the cognitive action; the fourth-order EL equations; causality is enforced by initial (Cauchy) conditions instead of terminal conditions, made possible by dissipation, so the dynamics can be integrated forward in time on a stream.
Quantitative claim: theoretical.
Optimizer implication: a well-posed online learning law must be causal; dissipation is what lets a two-point boundary problem be replaced by an initial-value problem.

### 5.3 Betti, Faggi, Gori, Tiezzi, Marullo, Meloni & Melacci 2022, "Continual Learning through Hamilton Equations", CoLLAs 2022 (PMLR 199)
https://proceedings.mlr.press/v199/betti22a/betti22a.pdf
Mechanism: frames online learning as optimal control: state y(t) = neuron outputs with dynamics y' = f(y, alpha, z(t)), control alpha(t) = weights, Lagrangian = loss, cost functional = sequential risk integral_0^T l dt. Hamilton-Jacobi-Bellman gives the value function; the method of characteristics gives Hamilton equations y' = D_p H, p' = -D_y H with costate p a Lagrange multiplier that is "a sort of delta error propagated backward in time", boundary condition p(T) = 0. Because p(T) = 0 is a terminal condition, they enforce it approximately in a causal way by filtering the input stream so that p -> 0 (Section 3.1). Case study: unsupervised optical flow from a video stream.
Quantitative claim: proof of concept on optical flow (Horn-Schunck functional), no benchmark numbers.
Optimizer implication: the "correct" online gradient over a stream is a costate that runs backward from the horizon; a causal optimizer must approximate it, and the approximation error is proportional to how much the current constraint has to be forced (|p|); large |p| is a signal to slow down or filter the input.

### 5.4 Haider, Ellenberger, Kriener, Jordan, Senn & Petrovici 2021, "Latent Equilibrium: A unified learning theory for arbitrarily fast computation with arbitrarily slow neurons", NeurIPS 2021
https://arxiv.org/abs/2110.14549
Mechanism: prospective coordinate u_breve = u + tau du/dt (position plus momentum). Energy E(u_breve) = 1/2 ||u_breve - W phi(u_breve) - b||^2 + beta L(u_breve) (Eq. 2). Stationarity dE/du_breve = 0 gives leaky-integrator dynamics tau u' = -u + W phi(u_breve) + b + e (Eq. 3) with top-down errors e = phi'(u_breve) W^T [u_breve - W phi(u_breve) - b], recursively e_l = phi' W_{l+1}^T e_{l+1}, a variant of BP. Because the dynamics stay on a constant-energy manifold, plasticity can be *continuously* active: W' = eta_W [u_breve - W r - b] r^T (Eq. 5), output layer = delta rule. Approximates BP as beta -> 0.
Quantitative claims: MNIST FC 784-300-100-10: 1.98 +/- 0.11% test error vs 1.93 +/- 0.14% for SGD-trained ANN; HIGGS 27.6 vs 27.8%; LeNet-5: MNIST 1.1 vs 1.08%, CIFAR-10 38.0 +/- 1.3% vs 39.4 +/- 5.6%; with FA 2.6% MNIST. Without prospective coding, tau = 10 ms and 100 ms presentations stay below 90% MNIST accuracy. Robustness section: multiplicative Gaussian heterogeneity on time constants and additive temporal noise on outputs degrade gracefully, and adapting time constants recovers performance.
Optimizer implication: using an extrapolated (position + tau x velocity) state as the coordinate on which the energy is evaluated makes the error signal consistent under continuous change; the streaming analogue is to compute the error on a look-ahead estimate of the parameters (Nesterov-style extrapolation) so that continuously applied updates remain consistent.

### 5.5 Senn, Dold, Kungl, Ellenberger, Jordan, Bengio, Sacramento & Petrovici 2024/2025, "A neuronal least-action principle for real-time learning in cortical circuits", eLife 13:RP89674
https://elifesciences.org/reviewed-preprints/89674
Mechanism: action A = integral of a Lagrangian = sum of somato-dendritic mismatch energies + beta x behavioural cost, but minimized w.r.t. *prospective* (look-ahead, discounted-future) voltages u_tilde, which are the canonical coordinates. Euler-Lagrange equations give voltage dynamics with apical prospective errors that propagate instantaneously through layers; plasticity is "postsynaptic error x low-pass filtered presynaptic rate", Delta W ∝ e_bar_i r_bar_j. Theorem 1 (rt-DeEP): this local rule performs gradient descent on the Lagrangian and on the output cost at every moment for any nudging beta >= 0; Theorem 2 (rt-DeEL): lateral interneuron circuits learn to extract the dendritic error. "Moving equilibrium hypothesis".
Quantitative claim: MNIST with 5 ms stimulus presentation reaches performance comparable to error backpropagation.
Optimizer implication: minimizing an integrated mismatch over time with prospective coordinates yields a rule that is *exactly* gradient descent at every instant on a moving target; the streaming version is "filter the presynaptic input, evaluate the error prospectively, never wait for settling".

### 5.6 Donatella, Duffield, Aifer, Melanson, Crooks & Coles 2024, "Thermodynamic Natural Gradient Descent", arXiv 2405.13817 (npj Unconventional Computing 2026)
https://arxiv.org/html/2405.13817v1
Mechanism: NGD requires solving (F + lambda I) x = g with F the Fisher/GGN. An analog Ornstein-Uhlenbeck system dx = -(F x - g) dt + noise has a Boltzmann stationary distribution whose *mean* is (F)^{-1} g, so equilibrating the analog system solves the linear system without sequential CG iterations. Analog time t interpolates: t = 0 gives GD, t -> infinity gives NGD (converges by ~50 tau). A delay between gradient computation and update acts like momentum.
Quantitative claim: MNIST: reaches target accuracy 2-3 orders of magnitude faster than Adam in (modelled) wall-clock; better generalization at smaller batch; DistilBERT/SQuAD: bare TNGD below Adam, TNGD-Adam hybrid best; runtime O(b d N + t) vs O(N^3) or O(c b N) for CG.
Optimizer implication: a noisy relaxation whose mean is the natural-gradient direction is a legitimate way to precondition; a streaming optimizer can keep a per-layer auxiliary vector relaxed by a few OU steps toward (F + lambda I)^{-1} g each iteration and use the partially relaxed vector (an interpolation between GD and NGD controlled by the number of relaxation steps).

---

## Thread 6. Energy/physics views of the optimizer itself: SGD as Langevin dynamics, temperature, flat minima, Adam vs SGD

### 6.1 Mandt, Hoffman & Blei 2017, "Stochastic Gradient Descent as Approximate Bayesian Inference", JMLR 18(134):1-35
https://arxiv.org/abs/1704.04289
Mechanism: constant-step SGD near a minimum is an Ornstein-Uhlenbeck process with stationary Gaussian distribution; step size/batch size set the temperature; one can choose step size (or a preconditioner) to minimize KL to the posterior; extends to momentum and to a Polyak-averaged sampler; quantifies bias of SGLD-type methods.
Quantitative claim: closed-form optimal constant step size eta* = 2 S / (N tr(B B^T)) style expressions; empirical KL comparisons.
Optimizer implication: an optimizer is a sampler at temperature ~ eta/B; preconditioning changes the *shape* of the stationary distribution, so per-coordinate Adam scaling changes which posterior the optimizer samples.

### 6.2 Chaudhari & Soatto 2018, "Stochastic gradient descent performs variational inference, converges to limit cycles for deep networks", ICLR 2018
https://arxiv.org/abs/1710.11029
Mechanism: SGD minimizes an average potential Phi over the weight posterior plus an entropy term (variational inference), but Phi != loss f unless the gradient-noise covariance is isotropic / proportional to the Hessian. Stationary density rho ∝ exp(-Phi / T) with T = eta / (2b). In deep nets the mini-batch gradient covariance has rank as low as ~1% of dimension, so noise is highly anisotropic, the dynamics are non-conservative, and the most likely trajectories are closed limit cycles rather than Brownian motion around critical points.
Quantitative claim: gradient covariance rank ~1% of dimension; empirical trajectories consistent with limit cycles.
Optimizer implication: what SGD actually minimizes is loss + T x (an entropy-like term) with T = eta/2b, on a potential shaped by the noise covariance; an optimizer that reshapes noise (Adam) changes the implicit objective, not just the speed.

### 6.3 Chaudhari, Choromanska, Soatto, LeCun, Baldassi, Borgs, Chayes, Sagun & Zecchina 2017, "Entropy-SGD: Biasing Gradient Descent Into Wide Valleys", ICLR 2017 (J. Stat. Mech. 2019)
https://arxiv.org/abs/1611.01838
Mechanism: replaces f(x) with local entropy F(x, gamma) = log integral exp(-f(x') - gamma/2 ||x - x'||^2) dx'; its gradient is gamma (x - <x'>) where <x'> is the mean of the local Gibbs measure, estimated by an inner SGLD loop; gamma is annealed ("scoping"). Provably smoother landscape and better uniform-stability generalization bound.
Quantitative claim: competitive or better generalization on CIFAR-10 CNNs and PTB RNNs at similar wall-clock.
Optimizer implication: the update direction is (current params - mean of a local Langevin cloud): a rank-free, noise-averaged proximal direction that favours wide valleys; a streaming variant keeps one slowly-relaxing Langevin copy of the parameters and pulls toward it.

### 6.4 Smith & Le 2018, "A Bayesian Perspective on Generalization and Stochastic Gradient Descent", ICLR 2018
https://arxiv.org/abs/1710.06451
Mechanism: Bayesian evidence penalizes sharp minima (Occam factor) and is reparameterization invariant; SGD noise scale g ~ eta N / B drives parameters to high-evidence minima; predicts an optimal batch size at fixed learning rate and the linear scaling rule B ∝ eta.
Quantitative claim: optimum batch size at fixed lr on small CNNs; B_opt ∝ eta and ∝ N.
Optimizer implication: eta/B is the single temperature parameter; changing either without the other changes what the optimizer converges to, not just how fast.

### 6.5 Simsekli, Sagun & Gurbuzbalaban 2019, "A Tail-Index Analysis of Stochastic Gradient Noise in Deep Neural Networks", ICML 2019
https://arxiv.org/abs/1901.06053
Mechanism: measured gradient noise is heavy-tailed (alpha-stable, tail index < 2), so SGD is better modelled by a Levy-driven SDE whose jumps escape narrow basins; escape time depends on basin width, not depth (metastability), and generalization relates to the tail index (later: Hausdorff dimension, NeurIPS 2020).
Quantitative claim: estimated tail indices well below 2 across architectures; lower index correlates with better generalization.
Optimizer implication: heavy-tailed noise is a feature for basin selection; an optimizer that clips or normalizes it (Adam's second moment) makes escape from sharp basins harder.

### 6.6 Zhou, Feng, Ma, Xiong, Hoi & E 2020, "Towards Theoretically Understanding Why SGD Generalizes Better Than ADAM in Deep Learning", NeurIPS 2020
https://proceedings.neurips.cc/paper/2020/hash/f3f27a324736617f20abbf2ffd806f6d-Abstract.html
Mechanism: models SGD and Adam as Levy-driven SDEs; escape time from a basin scales positively with the basin's Radon measure and negatively with noise heaviness. Adam's per-coordinate rescaling by sqrt(v) removes the anisotropy of the noise, which enlarges the effective Radon measure of basins, and its exponential gradient averaging lightens the noise tails; both increase Adam's escape time, so Adam stays in sharp basins that SGD leaves.
Quantitative claim: escape-time bounds; SGD escapes sharp minima faster; experiments consistent.
Optimizer implication: two specific Adam mechanisms hurt basin selection: (a) coordinate-wise variance normalization (isotropizes noise), (b) first-moment EMA (lightens tails). A replacement should preserve noise anisotropy and tails while still controlling scale.

### 6.7 Yang, Tang & Tu 2023, "Stochastic Gradient Descent Introduces an Effective Landscape-Dependent Regularization Favoring Flat Solutions", Physical Review Letters 130:237101
https://arxiv.org/abs/2206.01246
Mechanism: for a loss with a degenerate valley, solve the Fokker-Planck equation with anisotropic noise (variance Delta in the sharp direction, Delta/kappa in the flat one, kappa > 1). Steady state P_ss ∝ exp(-L_eff / Delta) with L_eff = L + L_SGD, L_SGD(x) = Delta (1 - kappa) ln F(x) (Eq. 7), F(x) = flatness = 1/sqrt(lambda(x)); in the realistic model L_SGD ≈ Delta(1-kappa) ln F + (kappa - 1) L (Eq. 11). Strength grows with learning rate and batch-to-batch variance (Delta_S); stronger noise shortens convergence to flat solutions up to a critical noise level (a second-order phase transition in the flatness order parameter) beyond which training diverges.
Quantitative claim: exact effective-loss formula; (sigma, eta) phase diagram with a critical point.
Optimizer implication: the flatness regularizer is -T (kappa-1) ln(flatness), so it exists only because noise is anisotropic and landscape-dependent (larger in sharp directions); an optimizer that equalizes noise across coordinates (Adam) sets kappa -> 1 and removes the regularizer; keep the anisotropy.

### 6.8 Kunstner, Chen, Lavington & Schmidt 2023, "Noise Is Not the Main Factor Behind the Gap Between SGD and Adam on Transformers, but Sign Descent Might Be", arXiv 2304.13960 (ICLR 2023)
https://arxiv.org/abs/2304.13960
Mechanism: varies batch size up to full batch on transformers; the Adam-SGD gap persists with no noise, so heavy-tailed noise robustness is not the explanation; in the large-batch limit Adam behaves like sign descent with momentum.
Quantitative claim: gap persists at full batch; sign descent with momentum closes most of it.
Optimizer implication: on transformers Adam's benefit is the *sign/normalization* geometry (per-coordinate scale invariance against heterogeneous curvature), which is exactly the mechanism that Thread 6.6-6.7 say hurts basin selection; a replacement must decouple "scale normalization for conditioning" from "noise isotropization".

### 6.9 Khan & Rue 2023, "The Bayesian Learning Rule", JMLR 24(281):1-46
https://arxiv.org/abs/2107.04562
Mechanism: natural-gradient descent on the variational free energy E_q[loss] - H(q) over an exponential-family q. With a Gaussian q and diagonal covariance, using squared gradients as the precision estimate, the rule reduces to RMSprop/Adam-like updates; the entropy term penalizes concentrated posteriors and thereby favours flat minima and provides uncertainty.
Quantitative claim: unifying derivation (ridge, Newton, Kalman, SGD, RMSprop, Adam-like, dropout).
Optimizer implication: Adam is (approximately) free-energy minimization with a *diagonal* Gaussian posterior; its "temperature" is implicit in the precision estimate; a richer q (block-diagonal, or one that tracks noise anisotropy) recovers the second-order/trust-region and flatness effects the EBM literature gets from relaxation.

### 6.10 Zhang et al. 2026, "Beyond a Single Explanation of the Adam-SGD Gap", arXiv 2606.14259
https://arxiv.org/abs/2606.14259
Mechanism: controlled study across vision, language, genomics, graphs; no single factor (heavy-tailed vocabulary, heterogeneity, noise) explains the gap; there is a crossover batch size below which SGD wins and above which Adam wins, captured by a theoretical gap model.
Quantitative claim: Adam advantage persists under uniform vocabulary yet nearly disappears under a heavy-tailed one; reverses toward SGD in softmax-attention models; crossover batch size observed in all settings.
Optimizer implication: at small batch (the streaming regime) SGD-like geometry is already competitive with or better than Adam; the streaming optimizer design space should start from SGD-with-temperature, not from Adam.

---

## Synthesis: what the energy-based lineage does differently from Adam at the level of "how parameter changes are computed from an error stream"

Adam computes, per parameter, m/(sqrt(v)+eps): a first-moment EMA divided by a second-moment EMA of the *raw* gradient stream. Every step is a function of that parameter's own history only. The energy-based lineage differs in eight concrete ways:

1. Relax state before touching weights (consistency across the network). PC/EP/prospective configuration first solve an inner problem over activities with the target clamped, then update all weights from the relaxed state. Song et al. 2024 (Nature Neurosci.) show this keeps "target alignment" ~0.95 at 25 layers where BP falls to ~0.6, and that it is why PC beats BP at batch size 1, under concept drift, and in alternating-task continual learning. Streaming rule: keep a per-layer auxiliary "relaxed error" state; take one relaxation step per sample (iPC, Salvatori 2024 shows one interleaved step is enough and more hyperparameter-robust) and update weights from it, so the updates of different layers are computed against one mutually consistent target.

2. The relaxed error is a damped Gauss-Newton / trust-region step computed in activity space, not parameter space. Innocenti et al. 2023: the PC inference solves argmin of a linearized loss under an adaptive quadratic constraint with A = Fisher information of the output, so z* ≈ z_t - I(z_t)^{-1} g_L, and the weight gradient interpolates between BP's gradient and that TR solution pushed through dz/dW (Eq. 7). Innocenti 2024 and Ishikawa 2025 give the closed form: the output residual is multiplied by (I + sum_l W_{L:l} W_{L:l}^T)^{-1}. Streaming rule: precondition the *per-layer error vector* (dimension = layer width) by (I + J J^T)^{-1} or a running estimate of it, instead of preconditioning per-parameter by 1/sqrt(v); rank-1 outer product with the input then gives the weight update.

3. The weight update is always local error x local activity (rank-1 Hebbian), so all "cleverness" lives in the error, not in per-parameter statistics. Millidge 2022, Haider 2021 (W' = eta [u_breve - W r - b] r^T), Senn 2024 (Delta W ∝ e_bar_i r_bar_j with low-pass filtered presynaptic rate). Streaming rule: filter the *input* (presynaptic) side with a low-pass, compute the error on a prospective (look-ahead) state, and form the outer product; do not keep a second-moment buffer per weight.

4. Implicit/proximal rather than explicit step. Alonso et al. 2022 (NeurIPS): inference learning approximates implicit SGD, theta_{t+1} = theta_t - lr grad L(theta_{t+1}), which explains its stability across learning rates and its advantage at minibatch size 1. Streaming rule: one or two fixed-point iterations of the implicit update (evaluate the gradient at the extrapolated post-update point) per sample.

5. Finite-difference gradients from two equilibria, centered, or demodulated. EP's update is (1/beta)[dE/dtheta(s_beta) - dE/dtheta(s_0)]; Scellier 2023 shows centered (+/-beta) differencing is best, Laborieux & Zenke 2022 that a sinusoidally driven teaching signal makes the gradient the first Fourier coefficient (exact for finite amplitude, noise-robust, ImageNet32 parity). Streaming rule: maintain a nudged and a free copy of the error state (dual propagation makes them one pass) and use their difference; or modulate the nudge and demodulate the update stream.

6. Regularize toward isotropy of the representation, not of the noise. LeJEPA's SIGReg (Epps-Pulley characteristic-function test on random 1-D projections, 256-1024 directions, linear cost, bounded gradients, single lambda ≈ 0.05) removes stop-gradient, EMA teachers and schedulers. Streaming rule: apply a sketched isotropy penalty to per-layer activations/errors (cheap, O(N d)), which keeps the (I + J J^T) preconditioner of item 2 well conditioned and replaces slow-copy heuristics.

7. Temperature/entropy: SGD minimizes loss + T x entropy-like term with T = eta/2b (Chaudhari & Soatto 2018, Mandt 2017, Smith & Le 2018), and the flatness regularizer L_SGD = -T(kappa-1) ln(flatness) exists *only* because gradient noise is anisotropic and landscape-dependent (Yang, Tang & Tu 2023, PRL). Zhou 2020: Adam's per-coordinate normalization isotropizes noise (kappa -> 1) and its EMA lightens tails, so it escapes sharp basins slower. Streaming rule: normalize gradient *scale* per layer (or per tensor) rather than per coordinate, so the anisotropy and heavy tails of the noise are preserved while conditioning is controlled; add explicit temperature via eta/B instead of via second-moment division.

8. Action penalizes change of state over time, with dissipation. Betti & Gori 2016: kinetic term on weight velocity plus e^{theta t} weighting gives D^2 w + theta D w + dV/dw = 0 (momentum GD as least action); higher-order kinetic terms give fourth-order laws; Betti 2022: the online gradient is a costate that should vanish at the horizon, and its magnitude measures how hard the current sample is being forced. Latent Equilibrium / neuronal least action: evaluate everything on prospective coordinates u + tau du/dt so continuously applied updates stay consistent. Streaming rule: treat the momentum buffer as a physical velocity with an explicit friction theta, evaluate the error at the extrapolated parameters (Nesterov-style), and use the size of the required "constraint force" (residual after relaxation) as the signal to reduce the step or increase friction.

Net effect: Adam's per-parameter division by sqrt(v) is replaced by (a) per-layer error preconditioning derived from relaxation, (b) an implicit/proximal step, (c) preserved anisotropic noise at a controlled temperature, (d) an explicit kinetic/dissipative time structure; the per-weight state becomes velocity only, and all second-order information lives in width-sized, per-layer buffers.
