# How the brain trades energy against learning, and what it implies for a streaming optimizer

Literature survey, September 2026. Scope: 2001-2026, emphasis 2021-2026. Every entry was
verified to exist (publisher/arXiv/PubMed page or abstract fetched, or DOI-bearing search hit).
Entries marked [PARTIAL] exist but the mechanistic details below came from abstracts / secondary
summaries rather than the full text, so specific numbers there should be re-checked before quoting.

Notation: "Streaming implication" = what the finding suggests for a per-parameter online
optimizer meant to replace Adam.

---

## Thread 1. Metabolic cost of synaptic plasticity and energy-efficient plasticity rules

### 1.1 Li & van Rossum (2020). Energy efficient synaptic plasticity. eLife 9:e50804. DOI 10.7554/eLife.50804
Mechanism: metabolic cost of plasticity modelled as sum over updates of |dw|^alpha (alpha=1, results
hold for 0 <= alpha <~ 2). Because sequential learning makes weights random-walk (each update partly
undone by the next pattern), total path length far exceeds the straight-line distance to the
solution. Synaptic caching: w = s + l, a cheap, decaying transient component s accumulates updates
and is consolidated into the expensive persistent component l only when |s| crosses a threshold.
Key numbers: perceptron storing 1,900 patterns on 1,000 synapses uses ~900x the minimum energy;
multilayer backprop ~20x the minimum; caching gives up to ~10-fold savings, and energy diverges as
load approaches capacity. Consolidation threshold trades maintenance cost of transient store vs
consolidation cost vs forgetting.
Streaming implication: keep a cheap fast accumulator per parameter and commit to the "real" weight
only when accumulated change exceeds a threshold; count |dw| path length, not |w|, as the cost.

### 1.2 Pache & van Rossum (2023). Energetically efficient learning in neuronal networks. Curr Opin Neurobiol 83:102779. DOI 10.1016/j.conb.2023.102779
Review. Argues computational plasticity models almost never include the metabolic cost of learning
and that "orders of magnitude of energy can be saved by tweaking standard learning rules"; catalogs
caching, sparse/competitive updating, and error-gating as the main levers.
Streaming implication: energy-aware learning rules are a design space, not a single trick; the
three levers are (a) fewer commits, (b) fewer parameters touched per step, (c) fewer steps.

### 1.3 van Rossum & Pache (2024). Competitive plasticity to reduce the energetic costs of learning. PLoS Comput Biol 20(10):e1012553. DOI 10.1371/journal.pcbi.1012553 (arXiv 2304.02594)
Mechanism: two energy measures, M0 = number of weight writes (alpha -> 0) and M1 = sum |dw|
(alpha = 1, "protein synthesis"). Three algorithms: (i) competitive selection: only the k synapses
with largest |gradient| are updated each trial, reselected each trial; (ii) subnet coordination:
plasticity restricted to neurons whose in- and out-connections are both plastic; (iii) synaptic
caching. Key numbers: for naive backprop, M0 and M1 both grow ~ sqrt(N_hidden); extrapolated to
macaque V1 sizes naive backprop needs ~10^5x more writes and ~700x more M1 than necessary. Subnet
coordination makes energy independent of network size (optimal plastic subnet ~60-100 units);
competitive selection keeps max accuracy essentially unchanged while fixed masks degrade it;
generalization at a fixed training error is only "minimally" worse. Competitive selection's target
set churns (Simpson index ~0.6 vs 0.01 for a fixed mask).
Streaming implication: top-k-by-|gradient| sparse updating with a churning mask loses almost nothing
and removes most write cost; a fixed sparse mask does hurt.

### 1.4 Pache & van Rossum (2023). Lazy learning: a biologically-inspired plasticity rule for fast and energy efficient synaptic plasticity. arXiv:2303.16067
Mechanism: error-gated plasticity: update only on misclassified samples (zero update when the
prediction is already right). Key numbers: 7.6x faster training than matched backprop; 99.2% test
accuracy on Extended MNIST with a single-layer MLP; no hyperparameter tuning.
Streaming implication: gate the update on whether the error exceeds a threshold; do not spend
updates polishing samples already inside the noise floor.

### 1.5 Girard, Jiang & van Rossum (2023). Estimating the energy requirements for long term memory formation. arXiv:2301.09565
Mechanism: combines fly starvation data (LTM shortens starved lifespan) with biophysical estimates
to bound the energy of consolidation. Key number: ~10 mJ per bit stored, orders of magnitude above
hardware; reason unknown.
Streaming implication: consolidation is the expensive step; the fast/transient store should absorb
most traffic.

### 1.6 Karbowski (2019). Metabolic constraints on synaptic learning and memory. J Neurophysiol 122(4):1473-1490. DOI 10.1152/jn.00092.2019 (arXiv 1910.07414)
Mechanism: estimates spine-level ATP cost of plasticity from protein phosphorylation and turnover,
and derives a "metabolic rate of learning" for cascade (multi-state) synapses. Key numbers: energy
of plasticity is 4.0-11.2% of the energy of fast excitatory transmission; longer memories cost
proportionally more; new learning is a small fraction of the baseline cost of maintaining old
memories.
Streaming implication: maintenance dominates; the cost of "remembering" is paid continuously, so
persistent parameters should be few and rarely rewritten.

### 1.7 Mery & Kawecki (2005). A cost of long-term memory in Drosophila. Science 308(5725):1148. DOI 10.1126/science.1111331
Mechanism: empirical. Inducing protein-synthesis-dependent LTM (spaced training) reduces survival
under starvation/desiccation by ~20%; anesthesia-resistant memory (no protein synthesis) has no
detectable cost.
Streaming implication: the biological system literally pays for consolidation, not for transient
learning; treat the two tiers as different resources.

### 1.8 Placais & Preat (2013). To favor survival under food shortage, the brain disables costly memory. Science 339(6118):440-442. DOI 10.1126/science.1226018
Mechanism: starved flies actively block aversive LTM formation; forcing LTM back on shortens
starved lifespan by ~1/3. Consolidation is gated by an energy-state signal.
Streaming implication: consolidation should be a gated decision that can be switched off under a
budget, not an unconditional side effect of every gradient.

### 1.9 Placais, de Tredern et al. (2017). Upregulated energy metabolism in the Drosophila mushroom body is the trigger for long-term memory. Nat Commun 8:15510. DOI 10.1038/ncomms15510
Mechanism: flies double sucrose intake early in LTM formation; a specific dopaminergic pair
raises mushroom-body energy flux via DAMB receptors; this energy switch is both necessary and
sufficient for LTM.
Streaming implication: a neuromodulatory "worth consolidating" signal precedes and licenses the
expensive write.

### 1.10 Malkin, O'Donnell & Houghton (2026). Energy budgets govern synaptic precision and its regulation during plasticity. arXiv:2602.15787
Mechanism: synapses minimise postsynaptic variance subject to a fixed mean and an energy budget;
on plasticity they reallocate energy proportionally to the change. Key numbers: precision scales
as sigma^-2 proportional to E^5; five datasets cluster near the minimal-energy boundary; dominant
cost resembles calcium pumping.
Streaming implication: precision (low noise) of a parameter is itself a metered resource; spend it
where the weight is large/important, accept noise elsewhere.

---

## Thread 2. Energy budgets of neural computation and how they constrain learning

### 2.1 Attwell & Laughlin (2001). An energy budget for signaling in the grey matter of the brain. J Cereb Blood Flow Metab 21(10):1133-1145. DOI 10.1097/00004647-200110000-00001
Key numbers (rodent grey matter): action potentials 47%, postsynaptic glutamate effects 34%,
resting potential 13%, glutamate recycling 3%; +1 spike/neuron/s raises O2 use by 145 mL/100 g/h;
predicts distributed codes with <= 15% of neurons active.
Streaming implication: activity (not storage) dominates the budget; sparse activation is the first-
order lever.

### 2.2 Laughlin, de Ruyter van Steveninck & Anderson (1998). The metabolic cost of neural information. Nat Neurosci 1:36-41. DOI 10.1038/236
Key numbers: ~10^4 ATP per bit at a chemical synapse, 10^6-10^7 ATP per bit for graded/spike coding
in blowfly retina; in noise-limited systems many low-capacity pathways are cheaper per bit than one
high-capacity one.
Streaming implication: cheap, low-precision channels in parallel beat one high-precision channel
when noise, not capacity, limits you.

### 2.3 [PARTIAL] Harris, Jolivet & Attwell (2012). Synaptic energy use and supply. Neuron 75(5):762-777. DOI 10.1016/j.neuron.2012.08.019
Review of pre- and postsynaptic ATP use, ATP supply to spines/terminals, and how plasticity and
brain state change synaptic energy use. Existence verified (PubMed 22958818); full-text numbers not
fetched (paywall), so do not quote a specific pre/post split from this survey.
Streaming implication: synaptic transmission is the dominant signalling cost, so the number of
active synapses, not the number of stored weights, is what a metabolic prior penalises.

### 2.4 Levy & Calvert (2021). Communication consumes 35 times more energy than computation in the human cortex, but both costs are needed to predict synapse number. PNAS 118(18):e2008173118. DOI 10.1073/pnas.2008173118
Key numbers: cortical grey matter ~3 W of ~17-20 W; communication (axonal/synaptic transmission)
35x computation (dendritic integration); neurons are ~10^8 above the Landauer bound per bit;
bits/J maximised when (synapses per neuron x success rate) ~ 2,000, matching anatomy.
Streaming implication: moving information (reading gradients across parameters, all-reduce, memory
traffic) is the expensive part; per-parameter local rules that avoid communication are the
biologically favoured regime.

### 2.5 Balasubramanian (2021). Brain power. PNAS 118(32):e2107022118. DOI 10.1073/pnas.2107022118
Commentary on 2.4; frames the 10^8 gap as the price of speed under communication cost and points
to efficient-coding theory as the design principle.
Streaming implication: same as 2.4.

### 2.6 Niven & Laughlin (2008). Energy limitation as a selective pressure on the evolution of sensory systems. J Exp Biol 211:1792-1804. DOI 10.1242/jeb.017574
Review across vertebrates/invertebrates showing morphology, sampling density and coding precision
of sensory systems are set by energy cost; reduced expenditure explains many features.
Streaming implication: precision is adaptive: resolution should be allocated where marginal
information per joule is highest.

### 2.7 Sengupta, Stemmler & Friston (2013). Information and efficiency in the nervous system - a synthesis. PLoS Comput Biol 9(7):e1003157. DOI 10.1371/journal.pcbi.1003157
Mechanism: "complexity minimisation lemma": minimising variational free energy (accuracy minus
complexity) also minimises thermodynamic free energy, so Bayes-optimal coding and metabolic
efficiency coincide. Cites sensory preprocessing compressing ~36 Gb/s to ~20 Mb/s (~1,500x).
Streaming implication: a KL/complexity penalty on the posterior over parameters is the same thing
as an energy penalty; the Bayesian-learning-rule family is the principled form of "cheap updates".

### 2.8 Padamsey, Katsanevaki, Dupuy & Rochefort (2022). Neocortex saves energy by reducing coding precision during food scarcity. Neuron 110(2):280-296. DOI 10.1016/j.neuron.2021.10.024
Key numbers: food restriction lowers AMPA conductance, cutting synaptic ATP use by 29%; firing
rates preserved; orientation tuning broadens by 32% and fine discrimination degrades.
Streaming implication: under budget pressure the brain trades precision, not activity; an optimizer
can likewise lower per-parameter precision/step-resolution before it lowers update frequency.

---

## Thread 3. Synapses as Bayesian / uncertainty-tracking learners

### 3.1 Aitchison, Jegminat, Menendez, Pfister, Pouget & Latham (2021). Synaptic plasticity as Bayesian inference. Nat Neurosci 24:565-571. DOI 10.1038/s41593-021-00809-5 (arXiv 1410.1029)
Mechanism: each synapse tracks a mean and variance over its (log-)weight. Learning rate for the
mean is proportional to posterior variance; variance shrinks with presynaptic activity (information
arrives only when the presynaptic cell fires) and grows by drift otherwise. Predictions: (1) more
uncertain synapses change more under an LTP protocol; (2) uncertainty (and PSP variability) is
higher for low-presynaptic-rate synapses; (3) PSP variability communicates uncertainty. Network
results: faster convergence than the delta rule; the rule is Adam/natural-gradient-like in that the
effective step is the gradient scaled by a per-parameter second-order quantity.
Streaming implication: per-parameter learning rate = posterior variance; variance decreases with
how much evidence that parameter has received (its input activity), and rises with a drift term so
rarely-used parameters stay plastic.

### 3.2 Kappel, Habenschuss, Legenstein & Maass (2015). Network plasticity as Bayesian inference. PLoS Comput Biol 11(11):e1004485. DOI 10.1371/journal.pcbi.1004485
Mechanism: synaptic sampling: d theta = b[grad log p(x|theta) + grad log p(theta)] dt +
sqrt(2bT) dW, i.e. Langevin sampling from the posterior with a sparsity prior and temperature T.
Key claims: bimodal priors on an RBM keep test log-likelihood stable under prolonged training
(no overfitting) where ML learning overfits; reproduces power-law survival of new spines; after
lesions the network recovers ~75% of performance within simulated 1-2 h by continuous rewiring.
Streaming implication: add temperature-scaled noise and a sparsity prior to each parameter's
update; the resulting sampler is self-regularising and self-repairing.

### 3.3 Fusi, Drew & Abbott (2005). Cascade models of synaptically stored memories. Neuron 45(4):599-611. DOI 10.1016/j.neuron.2005.02.001
Mechanism: each synapse has a cascade of states with geometrically decreasing plasticity
(metaplastic transitions). Result: power-law forgetting instead of exponential, combining high
initial storage with long retention.
Streaming implication: keep a per-parameter "depth" state; deep (consolidated) parameters get
exponentially smaller learning rates and are promoted only by repeated consistent updates.

### 3.4 Benna & Fusi (2016). Computational principles of synaptic memory consolidation. Nat Neurosci 19:1697-1706. DOI 10.1038/nn.4401
Mechanism: bidirectional cascade: a chain of coupled variables with exponentially growing time
constants, giving a ~t^-1/2 forgetting kernel. Key number: memory capacity scales almost linearly
with number of synapses vs sqrt(N) for simple bounded synapses.
Streaming implication: a chain of EMAs of the weight itself (not of gradients) with exponentially
spaced time constants, each pulling its neighbour, is the consolidation module; slow chain
elements act as an implicit weight-change penalty toward a long-horizon average.

### 3.5 Jegminat, Surace & Pfister (2022). Learning as filtering: implications for spike-based plasticity. PLoS Comput Biol 18(2):e1009721. DOI 10.1371/journal.pcbi.1009721
Mechanism: Bayesian filtering of a drifting weight from spike observations ("Synaptic Filter").
Mean dynamics reproduce STDP; variance dynamics predict spike-timing-dependent changes of EPSP
variability; explains two plasticity experiments that optimisation-based rules cannot.
Streaming implication: treat the target weight as a hidden state with drift; the Kalman gain
(posterior variance / (posterior variance + observation noise)) is the learning rate.

### 3.6 Malkin, O'Donnell, Houghton & Aitchison (2024). Signatures of Bayesian inference emerge from energy efficient synapses. eLife 13:RP92595 (arXiv 2309.03194)
Mechanism: stochastic synapses with four power-law reliability costs (calcium efflux sigma^-1/2,
vesicle membrane sigma^-2/3, actin sigma^-4/3, trafficking sigma^-2) trained on classification.
Result: energy-optimal synaptic variance correlates with Bayesian posterior variance (imperfectly);
predictions confirmed on data: low-variability synapses have higher input rates (Ko et al. 2013,
slope -0.71) and lower learning rates (Sjostrom et al. 2003, slope 1.47). The reliability cost
bounds the ELBO, so minimising it tightens the variational bound.
Streaming implication: you can get Bayesian-like per-parameter learning rates without explicit
inference by charging for precision: parameters allowed to be noisy should also be the ones with
larger steps.

### 3.7 Aitchison (2020). Bayesian filtering unifies adaptive and non-adaptive neural network optimization methods. NeurIPS 33 (arXiv 1807.07540)
Mechanism: treat optimisation as Bayesian filtering with backpropagated gradients as observations
and other parameters' motion as drift. Result: AdaBayes interpolates SGD-like and Adam-like
behaviour, recovers AdamW, generalises like SGD.
Streaming implication: this is the existence proof that "plasticity rate proportional to posterior
uncertainty" reduces to a concrete Adam-family optimizer; the synaptic literature adds what Adam
lacks: variance shrinking with input activity and growing with drift, plus consolidation.

---

## Thread 4. Free-energy / predictive coding as energy minimisation; prospective configuration

### 4.1 Friston (2010). The free-energy principle: a unified brain theory? Nat Rev Neurosci 11:127-138. DOI 10.1038/nrn2787
Mechanism: perception and action minimise variational free energy (an upper bound on surprise);
prediction error is the common currency.
Streaming implication: the objective is surprise (prediction error), with a complexity term that
penalises changes to beliefs, i.e. an implicit weight-change penalty.

### 4.2 Whittington & Bogacz (2017). An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity. Neural Comput 29(5):1229-1262. DOI 10.1162/NECO_a_00949
Mechanism: supervised predictive-coding network whose relaxation to equilibrium yields weight
updates that approximate backprop using only local pre x error products.
Streaming implication: infer activities first (relaxation), then update weights locally.

### 4.3 Millidge, Tschantz & Buckley (2022). Predictive coding approximates backprop along arbitrary computation graphs. Neural Comput 34(6):1329-1368 (arXiv 2006.04182)
Mechanism: shows PC converges to exact backprop gradients on arbitrary graphs (CNNs, RNNs, LSTMs).
Streaming implication: the "infer-then-update" family is not restricted to MLPs.

### 4.4 Millidge, Seth & Buckley (2022). Predictive coding: a theoretical and experimental review. arXiv:2107.12979
Review covering PC as variational inference, its energy function, and its relation to backprop and
biological data.
Streaming implication: reference for the energy function E = sum of precision-weighted squared
prediction errors.

### 4.5 Song, Millidge, Salvatori, Lukasiewicz, Xu & Bogacz (2024). Inferring neural activity before plasticity as a foundation for learning beyond backpropagation. Nat Neurosci 27:348-358. DOI 10.1038/s41593-023-01514-1 (bioRxiv 10.1101/2022.05.17.492325)
Mechanism (read carefully): the network is an energy-based model with E = sum over layers of
squared prediction errors. Learning has two phases. (1) Clamp input and output to (stimulus, target)
and let hidden activities relax by gradient descent on E until convergence: this infers the
"prospective configuration", the activity pattern the network should have after learning.
(2) Update every weight with a local Hebbian rule on the relaxed activities (delta w proportional to
pre x post-error). What relaxation buys: the weight changes of different layers are computed
against a consistent future state, so they do not fight each other. "Target alignment" (cosine
between the direction the output actually moves and the direction to the target) stays ~0.95 for
prospective configuration in deep random networks vs falling to ~0.4 for backprop at 10 layers; on
the bear/salmon toy task ~0.8 vs ~0.3 on the first step.
Claimed advantages, with numbers from the paper's figures: interference: after training task 2 in
alternating 5-class FashionMNIST, task-1 error ~25% vs ~45% for BP. Online learning (batch 1):
~3.5% vs ~6% test error, and much weaker dependence on batch size. Concept drift (periodic label
permutation): ~15% vs ~35%. Few-shot (10 examples/class): ~30% vs ~50%. Depth: 15-layer
FashionMNIST ~6% vs ~8%, gap grows with depth. CIFAR-10 CNN: ~45% vs ~48%. RL: CartPole reaches
~500 vs ~300 reward within 1,000 episodes; Acrobot and MountainCar similar. Reproduces three
behavioural/neural phenomena BP cannot: generalisation of motor adaptation to an untrained context,
extinction-induced fear enhancement (~70% freezing), and the mPFC BOLD inversion in reversal
learning with anticorrelated options. Noise robustness per se is not a headline claim; the online
and drift results are the closest.
Streaming implication: before committing a weight change, first settle what the internal state
"should" be given the new evidence, then move weights toward that settled state; in a per-parameter
optimizer this is the difference between chasing the raw gradient and chasing a filtered/predicted
target that all parameters agree on.

### 4.6 Ali, Ahmad, de Groot, van Gerven & Kietzmann (2022). Predictive coding is a consequence of energy efficiency in recurrent neural networks. Patterns 3(12):100639. DOI 10.1016/j.patter.2022.100639
Mechanism: RNN trained on ordered MNIST/CIFAR sequences with an L1 penalty on unit preactivation
(proxy for spiking + synaptic transmission cost). Prediction units (sustained activity, evidence
integration) and error units (near-zero median preactivation, fast lateral inhibition) self-organise;
activity approaches a theoretical lower energy bound as sequences become predictable; lesioning
prediction units abolishes temporal integration.
Streaming implication: an activity/energy penalty alone induces predictive coding; predicting the
next input is the cheapest code, so "learn to predict your own future" is what a metabolic prior
selects for.

---

## Thread 5. Sleep, consolidation gating, and synaptic downscaling as energy saving

### 5.1 Tononi & Cirelli (2014). Sleep and the price of plasticity. Neuron 81(1):12-34. DOI 10.1016/j.neuron.2013.12.025
Mechanism (SHY): wake learning produces net potentiation, which raises energy and supply needs,
lowers SNR and saturates learning; sleep renormalises total synaptic strength by down-selection,
protecting strong/consistent synapses and pruning weak/inconsistent ones.
Streaming implication: periodic multiplicative shrink of all parameters, protecting those that were
reinforced consistently, restores headroom and SNR.

### 5.2 de Vivo, Bellesi, Marshall, Bushong, Ellisman, Tononi & Cirelli (2017). Ultrastructural evidence for synaptic scaling across the wake/sleep cycle. Science 355(6324):507-510. DOI 10.1126/science.aah5982
Key numbers: 6,920 synapses reconstructed by serial EM in mouse cortex; axon-spine interface
shrinks ~18% after sleep, proportionally to size (scaling), and mainly in the ~80% of weaker,
plastic synapses; the largest ~20% are spared.
Streaming implication: downscaling is size-proportional and spares the largest/most stable
parameters; i.e. decay toward the consolidated state, not toward zero.

### 5.3 Tononi & Cirelli (2020). Sleep and synaptic down-selection. Eur J Neurosci 51(1):413-421. DOI 10.1111/ejn.14335
Update of SHY emphasising down-selection (competitive, activity-dependent) over uniform scaling,
consistent with 5.2.
Streaming implication: shrink parameters whose recent updates were inconsistent more than those
whose updates were consistent.

### 5.4 Gonzalez, Sokolov, Krishnan, Delanois & Bazhenov (2020). Can sleep protect memories from catastrophic forgetting? eLife 9:e51005. DOI 10.7554/eLife.51005
Mechanism: biophysical thalamocortical model; new learning degrades old memories, and a simulated
sleep phase (spontaneous replay under slow oscillations with STDP) reverses the damage and
strengthens all memories, orthogonalising their representations.
Streaming implication: an offline, label-free replay phase with a local rule can undo interference
without stored data.

### 5.5 Tadros, Krishnan, Ramyaa & Bazhenov (2022). Sleep-like unsupervised replay reduces catastrophic forgetting in artificial neural networks. Nat Commun 13:7742. DOI 10.1038/s41467-022-34938-7
Mechanism: after each task, switch ReLU to Heaviside, scale weights by max layer activation,
drive with Poisson noise matched to mean pixel statistics, and apply a Hebbian rule (potentiate
sequentially active pairs, depress post-active/pre-silent pairs). Key numbers (class-incremental,
old-task accuracy before -> after sleep): MNIST 19.5% -> 48.5%, Fashion-MNIST 19.7% -> 41.7%,
CIFAR-10 19.0% -> 44.6%.
Streaming implication: consolidation can be a separate phase with its own cheap, label-free rule;
per-parameter it looks like "reinforce what spontaneously co-activates, depress what does not".

### 5.6 Hoel (2021). The overfitted brain: dreams evolved to assist generalization. Patterns 2(5):100244. DOI 10.1016/j.patter.2021.100244
Hypothesis: daily learning overfits; dreams inject corrupted/noisy inputs (like dropout/noise
injection) to improve generalisation. No quantitative model, but ties the sleep thread to the
generalisation thread.
Streaming implication: noise injected during consolidation is a regulariser, not a nuisance.

---

## Thread 6. Energy / metabolic constraints that improve generalisation, abstraction or robustness (2017-2026)

### 6.1 Whittington, Dorrell, Ganguli & Behrens (2023). Disentanglement with biological constraints: a theory of functional cell types. ICLR 2023 (Outstanding Paper honourable mention); arXiv:2210.01768 [PARTIAL on loss details]
Mechanism: nonnegativity of activity and weights plus energy penalties on activity and weights
provably force linear networks to become single-factor selective (disentangled); empirically holds
for nonlinear nets and VAEs; explains grid vs object-vector cell types and when the brain instead
entangles (when task factors are entangled). Exact penalty form (L1 vs L2) not verified here.
Streaming implication: an energy penalty on activity is an abstraction prior; the optimizer should
not undo it (Adam's per-coordinate normalisation partly cancels weight-norm penalties; a metabolic
optimizer should preserve them).

### 6.2 Stroud, Wojcik, Jensen, Kusunoki, Kadohisa, Buckley, Duncan, Stokes & Lengyel (2025). Effects of noise and metabolic cost on cortical task representations. eLife 13:RP94961. DOI 10.7554/eLife.94961
Mechanism: RNNs on a context-dependent task swept over noise sigma and L2 firing-rate cost lambda.
Task performance stays > 0.95 across most of the grid, but representational geometry moves from
"maximal" (all stimuli represented) to "minimal" (irrelevant stimuli suppressed via activity-silent
subthreshold dynamics, which cost nothing). High-noise/high-cost networks match monkey lateral PFC,
where decodability of irrelevant features falls with learning while the XOR variable strengthens.
Streaming implication: energy cost plus noise selects low-dimensional, task-minimal representations
that are robust to irrelevant variation; this is "noise robustness by metabolic cost".

### 6.3 Ali et al. (2022) (see 4.6): energy efficiency alone produces predictive coding, i.e. a
representation that is invariant to the predictable part of the input.

### 6.4 Pache & van Rossum (2023), Lazy learning (see 1.4): error-gated updating gave state-of-the-art
EMNIST accuracy with 7.6x fewer updates; the energy constraint did not cost generalisation.

### 6.5 van Rossum & Pache (2024) (see 1.3): competitive top-k plasticity keeps accuracy while cutting
writes; generalisation loss "minimal". A fixed sparse mask, by contrast, does hurt.

### 6.6 Inoue, Rohrbein & Knoblauch (2026). Guiding sparse neural networks with neurobiological principles to elicit biologically plausible representations. arXiv:2603.03234
Mechanism: learning rule that yields sparsity, lognormal weights and Dale's law without explicit
enforcement. Claims: enhanced adversarial robustness and better few-shot generalisation; numbers
"preliminary" and not in the abstract.
Streaming implication: weak evidence, but consistent with 6.1/6.2.

### 6.7 Inoue, Rohrbein & Knoblauch (2026). Constrained Hebbian learning supports efficient representational allocation under structural constraints. arXiv:2607.16027
Mechanism: excitatory competitive Hebbian rule under synaptic-maintenance cost and limited
connectivity. Result: lower task-information cost than sparse BP/DDTP at matched task information;
retains less input information while preserving performance; slight accuracy loss on some datasets.
Framed as resource allocation, not accuracy maximisation.
Streaming implication: metabolic constraints buy compression (drop input information that is not
task-relevant), which is the mechanism behind the abstraction claims.

### 6.8 Zenke, Poole & Ganguli (2017). Continual learning through synaptic intelligence. ICML 70:3987-3995
Mechanism: each parameter accumulates its path-integral contribution to loss reduction (omega =
sum of -grad x dw) and a quadratic penalty on subsequent change is weighted by that importance.
Reduces forgetting on split/permuted MNIST comparable to EWC.
Streaming implication: the weight-change penalty should be per-parameter and proportional to how
much loss that parameter's past changes actually bought, which is a cheap online statistic.

### 6.9 Padamsey et al. (2022) (see 2.8) is the counterexample: when the budget is cut after
learning, precision, not activity, is sacrificed and discrimination degrades 32%. Energy
constraints help representations when applied during learning, and hurt when imposed on a fixed
code.

---

## Thread 7. Dendritic / compartment-level prediction: neurons that optimise toward their own prediction

### 7.1 Urbanczik & Senn (2014). Learning by the dendritic prediction of somatic spiking. Neuron 81(3):521-528. DOI 10.1016/j.neuron.2013.11.030
Mechanism: two-compartment neuron; dendritic synapses learn to make the dendritic potential V_d
predict somatic firing: delta w proportional to [phi(V_soma) - phi(V_d*)] x (filtered presynaptic
input), where V_d* is the dendritic prediction of the somatic voltage (attenuated). Same rule does
supervised learning when the soma is driven by a teacher, unsupervised when driven by other
inputs, and RL when somatic fluctuations serve as exploration; equivalent to a widely used point-
neuron RL rule.
Streaming implication: the learning signal is (actual - own prediction), computed locally; the
"teacher" only ever enters as a nudge on the soma.

### 7.2 Brea, Gaal, Urbanczik & Senn (2016). Prospective coding by spiking neurons. PLoS Comput Biol 12(6):e1005003. DOI 10.1371/journal.pcbi.1005003
Mechanism: with an STDP window slightly wider than the PSP, the 7.1 rule bootstraps so that the
neuron's current rate represents its expected, discounted future rate; equivalent to TD(lambda)
with eligibility traces. Key numbers: ~20 ms plasticity window yields effective discount time
constants of seconds; learns ramps ~600 ms before predicted events, associations across 1-2 s
gaps, ~50 ms advancement of a time-varying signal, and accelerated replay of sequences.
Streaming implication: bootstrapping toward one's own discounted future prediction (TD-style) is
implementable with a single local trace; the target is the neuron's own future, not the raw input.

### 7.3 Haider, Ellenberger, Kriener, Jordan, Senn & Petrovici (2021). Latent Equilibrium: a unified learning theory for arbitrarily fast computation with arbitrarily slow neurons. NeurIPS 34 (arXiv 2110.14549)
Mechanism: neurons output a prospective (phase-advanced) rate r = rho(u) + tau d/dt rho(u); neuron
and synapse dynamics are derived from a prospective energy function of generalised position and
momentum. Result: a network of slow leaky neurons computes and learns as if instantaneous;
biologically plausible approximation of backprop with continuously active local plasticity,
competitive on standard benchmarks.
Streaming implication: a look-ahead term (state + tau x its derivative) removes lag between the
parameter and its target, so updates are computed against where the system is going, not where it
was.

### 7.4 Senn, Dold, Kungl, Ellenberger, Jordan, Bengio, Sacramento & Petrovici (2024). A neuronal least-action principle for real-time learning in cortical circuits. eLife 12:RP89674. DOI 10.7554/eLife.89674
Mechanism: Lagrangian = sum of somato-dendritic mismatch energies (squared difference between
somatic voltage and weighted basal input) plus an output cost, minimised with respect to future
voltages. Prospective rates r = rho(u) + tau rho'(u) du/dt and prospective apical errors give
"instantaneous voltage-to-voltage transfer" across layers. Theorem rt-DeEP: local plasticity
dW proportional to error x low-pass presynaptic rate performs gradient descent on the Lagrangian
at every instant, without waiting for relaxation; beta -> 0 gives gradient descent on behavioural
cost. Numbers: 784-500-10 MNIST network with presentation times of 5-200 ms (tau = 10 ms) matches
backprop; without look-ahead rates learning fails at fast presentations unless plasticity is
disabled during transients; recurrent net reproduces held-out channels of 56-electrode intracranial
EEG.
Streaming implication: with a prospective target you can update continuously during transients;
without it you must gate plasticity to quasi-stationary periods. This is the direct theoretical
support for "optimize toward own prediction of the future".

### 7.5 Ellenberger, Haider, Jordan, Max, Jaras, Kriener, Benitez & Petrovici (2026). Backpropagation through space, time and the brain. Nat Commun 17:66 (arXiv 2403.16933)
Mechanism: Generalised Latent Equilibrium: neuron-local mismatch energy, prospective coding for
the forward pass (a spatio-temporal convolution), temporally inverted signals for the backward
pass approximating adjoints; online, fully local approximation of BPTT.
Streaming implication: the prospective/retrospective pair (look-ahead forward, look-back backward)
is the local substitute for storing the whole trajectory.

### 7.6 Brandt, Petrovici, Senn, Wilmes & Benitez (2024). Prospective and retrospective coding in cortical neurons. arXiv:2405.14810
Mechanism: shows biophysically how cortical neurons advance output relative to input: sodium
inactivation makes spike timing depend on voltage and its derivative; adaptation processes at
several timescales produce advanced responses to slow modulations.
Streaming implication: the derivative term in 7.3/7.4 is not a modelling trick; real neurons
implement it.

### 7.7 [PARTIAL] Luczak, McNaughton & Kubo (2022). Neurons learn by predicting future activity. Nat Mach Intell 4:62-72. DOI 10.1038/s42256-021-00430-y
Mechanism (from abstract and press coverage; full text not fetched): a neuron predicts its own
later activity from its earlier activity; weight changes are driven by the difference between the
actual later activity and this prediction; the rule is derived from a metabolic principle (minimise
own synaptic activity while maximising impact on local blood supply by recruiting other neurons,
the "lazy neuron principle"); the resulting rule approximates backprop-like learning and is
supported by in vivo recordings in which early activity predicts later activity.
Streaming implication: the most explicit statement in the literature that "optimise toward own
prediction of the future" is the energy-minimising rule for a single unit.

### 7.8 Sacramento, Costa, Bengio & Senn (2018). Dendritic cortical microcircuits approximate the backpropagation algorithm. NeurIPS 31:8721-8732
Mechanism: apical dendrites hold the error as a mismatch between lateral interneuron predictions
and top-down feedback; basal synapses learn by the 7.1 rule; analytically approximates backprop.
Streaming implication: error = (what arrived) - (what my local model predicted would arrive).

### 7.9 Mikulasch, Rudelt, Wibral & Priesemann (2023). Where is the error? Hierarchical predictive coding through dendritic error computation. Trends Neurosci 46(1):45-59. DOI 10.1016/j.tins.2022.09.007
Review arguing prediction errors live in dendritic voltages, not separate error neurons; unifies
7.1/7.8 with hierarchical PC.
Streaming implication: errors are per-unit local quantities; no separate error channel is needed.

---

## Synthesis: recurring principles and their optimizer mappings

Eight principles recur across the seven threads. For each: strongest citation, then a one-sentence
mapping to a per-parameter streaming rule intended to replace Adam.

1. Charge for weight change, not weight size. Energy of plasticity is the path length sum|dw|
   (Li & van Rossum 2020; Karbowski 2019), and the fly pays ~20% of starved lifespan for a
   protein-synthesis memory (Mery & Kawecki 2005). Mapping: add a per-parameter L1 cost on the
   committed step and an importance-weighted quadratic cost on drift from the consolidated value
   (Zenke et al. 2017), rather than decoupled decay toward zero.

2. Do not chase noise: update only when error exceeds the noise floor. Lazy learning updates only
   on misclassified samples and is 7.6x faster with no accuracy loss (Pache & van Rossum 2023);
   competitive plasticity updates only top-k |gradient| with a churning mask (van Rossum & Pache
   2024). Mapping: gate each parameter's step on |g| exceeding c times its running gradient std
   (a per-parameter SNR test), and skip the write otherwise.

3. Two-tier store: accumulate cheaply, consolidate rarely, and only persistent changes. Synaptic
   caching (transient s, persistent l, commit when |s| > theta) gives up to 10x savings; biology
   gates consolidation by an energy-state signal (Placais & Preat 2013; Placais et al. 2017) and
   uses bidirectional cascades with t^-1/2 forgetting for near-linear capacity (Benna & Fusi 2016).
   Mapping: keep a fast per-parameter accumulator that decays, and move the slow weight only when
   the accumulator crosses a threshold, with a chain of slower EMAs of the weight itself pulling
   toward the consolidated value.

4. Plasticity rate proportional to posterior uncertainty; uncertainty shrinks with evidence and
   grows with drift. Aitchison et al. 2021; Jegminat et al. 2022 (Kalman gain); Malkin et al.
   2024 show the same signature emerges from precision costs alone. Mapping: per-parameter step
   = sigma2_i x g_i, with sigma2_i decreasing by an amount proportional to that parameter's input
   activity (fan-in energy, not gradient magnitude) each step and increasing by a fixed drift
   variance, i.e. a Kalman gain instead of Adam's 1/sqrt(v).

5. Infer the target state before changing weights. Prospective configuration relaxes hidden
   activity to the post-learning configuration first, keeping target alignment ~0.95 vs ~0.4 for
   backprop at depth 10, halving task-1 error under interference (25% vs 45%) and online error
   (3.5% vs 6%) (Song et al. 2024). Mapping: for each parameter, maintain a filtered estimate of
   where it should be (a predicted target from its own recent trajectory and error history), and
   step toward that estimate rather than along the instantaneous gradient.

6. Optimise toward your own look-ahead prediction, and you may update continuously. Prospective
   rates r = rho(u) + tau d/dt rho(u) let local plasticity do exact real-time gradient descent
   during 5 ms transients where non-prospective networks fail (Senn et al. 2024; Haider et al.
   2021; Brea et al. 2016 gives the TD(lambda) form). Mapping: the update target is the
   parameter's own extrapolated future value (current filtered estimate plus tau times its
   velocity) and the error is (realised - predicted); this removes the lag that forces Adam to be
   smoothed with large beta.

7. Energy penalties on activity are abstraction priors; keep them, do not cancel them.
   Nonnegativity plus activity/weight energy provably disentangles (Whittington et al. 2023);
   noise plus firing cost yields minimal, task-relevant, low-dimensional codes matching monkey PFC
   (Stroud et al. 2025); an L1 preactivation cost alone produces predictive coding (Ali et al.
   2022). Mapping: apply activity and weight-norm penalties before any per-parameter
   normalisation, and never rescale them by the second moment.

8. Periodic proportional downscaling that spares consolidated parameters. Sleep shrinks synapses
   ~18%, proportionally to size and mostly in the weakest 80% (de Vivo et al. 2017); label-free
   Hebbian replay recovers old-task accuracy from ~19% to 42-48% (Tadros et al. 2022). Mapping: on
   a slow clock, multiply fast-tier parameters toward their slow-tier value, with the shrink
   fraction inversely related to update consistency (sign agreement of recent steps).

Cross-cutting caution: communication is 35x computation (Levy & Calvert 2021), so any rule needing
non-local statistics (global norms, all-reduce of second moments) is fighting the biology; every
quantity above is per-parameter and local.

---

## Verification notes

Verified by fetching full text or abstract page: 1.1, 1.3, 1.4, 1.5, 1.10, 2.4, 2.7, 3.1, 3.2, 3.6,
4.5, 4.6, 5.5, 6.2, 6.6, 6.7, 7.2, 7.4, 7.5, 7.6, 7.3 (arXiv abstract), 6.1 (arXiv abstract).
Verified by DOI-bearing search result (PubMed/publisher/arXiv listing): 1.2, 1.6, 1.7, 1.8, 1.9,
2.1, 2.2, 2.3, 2.5, 2.6, 2.8, 3.3, 3.4, 3.5, 3.7, 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.6, 6.8,
7.1, 7.7, 7.8, 7.9.
Partial: 2.3 (Harris 2012, paywalled; no numbers quoted), 6.1 (Whittington 2023; penalty form not
confirmed), 7.7 (Luczak 2022; mechanism from abstract and press coverage only).
Not found: no 2020+ paper that quantitatively models synaptic downscaling specifically as an
energy-saving optimisation with numbers; the sleep thread's quantitative work (Bazhenov group) is
about interference, with energy only as motivation.

Total unique papers: 51 (48 fully verified for existence and headline claim, 3 partial).
