# sparse-nn — Morphogenic Perceptrons (utility-driven sparse MLPs)

Guiding notes for this family. Soft direction only: implement versions against these constraints, not against a frozen spec.

---

## PRESCRIPTION (current doctrine)

**Doctrine:** Prefer learning over hand-modelled “useful synapses.” Update weights often, rewire often, **meta-learn wiring** so return selects structure. No hand utility \(u_e\), no soft capacity gates, no IDBD optimizer.

### Mechanism (v_next — implement this, not more utility essays)

**Inner net (fixed compute shape)**  
- Per unit: exactly **K hard incoming edges** (indices + weights).  
- Separate actor/critic; **SiLU**; **unimodal Beta** policy.  
- Default size: `N=256`, `K=64`, `L=2`, pool=`prev` first (prior only after L grows or as a one-factor add-on).  
- Forward: gather–weighted sum only. **No** soft masks on edges.

**Weight updates — often**  
- PPO with **`update_epochs=1`** (near on-policy w.r.t. a moving graph).  
- Standard envs/steps/minibatches unless something breaks (16 envs, 2048 steps).  
- Adam on \(w\) and heads as usual.

**Rewires — often, stochastic, learned rate/keep**  
- After **every optim step** (default): each edge drops with  
  \(p_e = \sigma(\phi_e)\)  
  where \(\phi_e\) is a **learnable logit per edge** (meta-params; same shape as weights, not shared with \(w\)).  
- On drop: **random** new source in the allowed set; \(w_e \leftarrow\) small init; clear Adam state for that slot; reset any edge-local meta state.  
- Cap churn if needed: max fraction \(q_{\max}\) (e.g. 0.05) of edges per step via sampling without replacement among proposed drops — **cap only**, not a hand utility rank.  
- Init \(\phi_e\) so mean \(p\) is low (e.g. ~1–2%) so the run starts near-static and **learning** can raise rewiring where it helps.

**Meta-learning wiring (this is the “learn better wiring”)**  
- \(\phi\) is **not** trained by hand saliency. Train \(\phi\) with a **return-based** bandit/REINFORCE signal each PPO iteration:  
  - baseline \(b \leftarrow\) EMA of mean episodic return (or mean return over the last rollout);  
  - advantage \(A = R - b\);  
  - for each edge that was considered this iter,  
    \(\mathcal{L}_\phi \mathrel{+}= -\log \pi(\text{drop or keep}\mid\phi_e)\, A\)  
    (or only on edges that actually sampled a drop decision);  
  - step \(\phi\) with its own small Adam (or same opt, lower LR).  
- Optional later (one factor): **learned regrow** — small logits over sources for freed slots; same REINFORCE on return. Random regrow is the v_next default.

**Explicitly out of scope for v_next**  
- Hand \(u_e\) (mag, \(|pre||w|\), grad-agreement “meta”, saliency essays).  
- Absolute/learned threshold on magnitude with soft gates.  
- IDBD step-size optimizer as the mainline.  
- Dense controls unless a comparison is needed later.

### Success / kill criteria (use `scripts/score_runs.py`)

- **Success:** last-100 mean return **clearly above** `spk_static_prior` / `spk_static_prev` (~5.8k class) at 8M, with **non-trivial** rewire rate that is **not** constant random thrash (log mean \(p\), fraction dropped/step, return).  
- **Kill / simplify:** return ≤ static and \(\phi\) collapses to “never rewire” or “always thrash” → reduce \(q_{\max}\), raise init keep, or ES on a lower-dim \(\phi\) (per-layer / global) before inventing new utilities.

### Queue naming

`spk_metalearn_wire_v1` — epochs=1, rewire every optim step, per-edge \(\phi\), REINFORCE on return, random regrow, pool=prev (or prior if matching static baseline).

---

## Premise

Morphcompute already explores **state-dependent soft circuits over a few rich vector cells** (entmax routing, lookback, paid edges, plasticity). That does not scale to scalar units: soft scores over a large pool are O(NP) / O(N²) and reintroduce the cost the sparsity story is supposed to remove.

The bet for this family:

> **Hard, fixed-K synapses on many cheap units + aggressive, stochastic utility-driven rewiring of connections** can give DenseNet-style any-prior connectivity and brain-like pruning/clustering at O(NK) cost — a path to gigantic MLPs without dense O(N²) matmuls. Topology is a first-class learner, not a rare maintenance chore.

**Unit of rewiring = connections (synapses / edges), not perceptrons (neurons / units).**  
The neuron set is fixed (or only slowly grown as capacity). What changes every optim step is *which* inputs a unit reads: drop a low-utility edge, attach a new source. SET/RigL/synapses are the same abstraction — mask or index over weights — not neuron birth/death (NEAT-style). Optional later: dead units with no useful edges; that is a side effect, not the rewire primitive.

Intuition (brain-shaped, not biology cargo-cult):

- **Perceptrons** are fixed compute sites with capacity K incoming slots.
- **Connections** are the synapses that fill those slots: each may target **any previous layer** (prefix pool), not only the layer above.
- Utility and rewiring attach to **edges** (and optionally edge age/protection), not to whole units.
- Optional **dendritic structure**: group a unit’s K synapses into D compartments so clusters form without a second global graph search.
- Ideal outcome: **huge width/depth at fixed FLOP**, better feature circuits than a dense 64–256 MLP on MuJoCo continuous control.

### Training regime (intentional, not accidental)

- Rewiring is **utility-dependent** and runs **on episode end** (not a global optim-step schedule, not RigL ΔT).
- **Connections** evolve; units and K stay fixed.
- PPO keeps **standard batching** (see variant plan for why episode-end rewire does not force 1-epoch / no-minibatch).
- Freeze connectivity **during** the PPO update phase so multi-epoch passes do not thrash G mid-optimization.

## Soft direction

### Do

1. **Fixed-K hard indices** into a prefix pool  
   `pool = concat(all earlier activations)`; each unit gathers K sources, weighted sum, nonlinearity. Any-prior-layer access is free in topology; cost stays O(NK). Indices \(I\) are **data/state-independent within a forward** but **may change every optim step**.

2. **Utility drop + random regrow of edges (SET default)**  
   On **episode end**: drop low-utility / low-\|w\| **edges**, **regrow randomly** into the allowed pool. Neuron count and K stay put. Threshold rule (`utility < X`) is a planned alternative to fraction drop. New edge weights small/zero; **reset Adam moments** on rewired slots only.

3. **Standard PPO outer loop**  
   Keep usual epochs/minibatches; freeze G during the update. Log stickiness, rewire counts, returns.

4. **Optional dendrites as blocked K**  
   D dendrites × (K/D) synapses with restricted source groups; optional dendritic nonlinearity then soma. Dendrite-level reassignment can be slower than synapse rewire if needed.

5. **Log real capacity, not illusory width**  
   Effective rank, rewire rate, stickiness (how long edges survive), utility gap, obs→action path length, dead-edge fraction. Separate actor/critic trunks (CircuitNet lesson).

6. **Compare fairly**  
   Matched-FLOP dense MLP, DenseNet-style concat dense, static-sparse (no rewire), rare-rewire ablation. HalfCheetah: seed=1, 16 envs, 8M steps, CUDA.

### Do not

- Soft entmax / attention over the full unit pool (hard edges only).
- Wiring *schedules* (ΔT, cosine α) as the main mechanism — utility + episode boundary instead.
- Dense control runs; compare sparse variants to each other.
- Multiple confounders in one ablation; minor coef sweeps before the qualitative rule is shown to matter.
- Materialize full dense \(N\times P\) when pool is large.

### Cost mental model

| Design | Scaling |
|--------|---------|
| Soft routing / scores over pool | O(NP) — kill |
| Fixed-K gather, dynamic \(I\) | O(NK) — target |
| Dense layer | O(N²) |
| Episode-end rewire | cheap; shapes fixed, only index buffer changes |

`I` changes on episode ends (not every GPU kernel launch). Still treat \(I\) as a tensor input, not a once-per-run compile constant.

### Map from morphcompute (keep / kill)

| Morphcompute idea | At connection / unit scale |
|-------------------|---------------------------|
| Any previous layer / lookback | Keep as prefix-pool **edge** indices |
| Sparse edges | Keep as fixed-K hard edges per unit |
| Prune by utility | Keep as discrete **edge** rewiring (**often**); not unit deletion |
| Soft entmax every step | Kill |
| State-dependent soft graph every step | Kill; hard \(I\) updates on **episode end** via utility |
| Paid compute multiplier | Optional; fixed-K already caps cost |
| Plasticity fields | Utility + rewire |
| Null / no-read | Dead synapses (w≈0) that get rewired |
| Cell coords / properties | Cluster ids + dendrite affinity for rewire proposals |

## Variant plan

**Primary rule:** sparsity is **per perceptron** — each unit has exactly **K incoming connections**. Rewiring reassigns **edges only**, not units. No soft routing, no wiring *schedules* (no ΔT / cosine drop fraction): rewiring is **utility-driven** and runs **on episode end**.

### Shared scaffold

| Knob | Choice |
|------|--------|
| Env | HalfCheetah-v4, seed=1, 16 envs, 8M, CUDA |
| Policy | **Unimodal Beta** (v215 / betaplast style): `α,β = 1 + softplus(heads)`, native z∈(0,1) → linear action bounds |
| Activation | **SiLU** (not tanh) |
| Trunk | Separate actor / critic; L=2 hidden layers width **N**, fan-in **K** per unit |
| Density | **K matched to dense fan-in**, not “extra-sparse.” CleanRL dense is 64-wide fully connected → **K=64**. Use **N>K** (default **N=256**) so prev-layer-only is actually incomplete; each unit is *as connected as* a dense-64 unit, with more units. First layer: K=min(64, obs_dim) or full obs if obs_dim≤64 (HalfCheetah obs is small — first layer can be dense-to-obs without drama). |
| Drop / grow (SET) | Drop lowest-\|w\| fraction ζ of **edges**; random regrow into allowed pool; new w small/0; reset Adam on those slots |
| When to rewire | **On env episode end** (per-env), not on optim step and not on a global schedule |
| PPO batching | **Keep defaults** (epochs=10, num_minibatches=32, num_steps=2048). See reasoning below. |
| No dense control | Sparse variants only; compare to each other |

### Minibatches / off-policy (reasoned, not ablated)

Rewire-every-**optim-step** would make multi-epoch PPO topology-off-policy → then “1 epoch / no minibatches” might matter.

**Per-episode rewiring** is different:

1. During a rollout, G only jumps when an env `done`s — sparse in time vs every SGD step.
2. Freeze **G during the PPO update** (epochs × minibatches). Then the ratio is wrong only for *θ* (normal PPO), not for mid-update topology thrash.
3. Rollout logprobs still mix a few G’s (episodes ending mid-rollout). That is mild non-stationarity, similar to a non-stationary policy — not a reason to collapse the batch. PPO already tolerates multi-epoch reuse of the same rollout under changing θ.
4. Collapsing minibatches mainly reduces gradient noise / changes effective LR; it does not restore “collected under G_now” for old transitions. Wrong tool for topology mismatch.

**Conclusion:** keep standard PPO batching; rewire on episode end; freeze connectivity while optimizing. Revisit batching only if diagnostics show topology churn mid-rollout is destroying ratios (e.g. huge clip fractions correlated with rewire rate).

### Queue variants (one factor at a time)

Factor A = **pool**: prev-layer only vs **all-prior** (prefix concat).  
Factor B = **rewire rule**: none vs SET vs threshold utility.

| # | exp_name | Pool | Rewire | What it isolates |
|---|----------|------|--------|------------------|
| **1** | `spk_static_prev_v1` | previous layer | none | Fixed-K sparse MLP baseline |
| **2** | `spk_static_prior_v1` | all prior layers | none | Same K, long-range sources without evolution |
| **3** | `spk_set_prev_v1` | previous layer | SET (mag drop + random regrow) on episode end | Does SET help under prev-only? |
| **4** | `spk_set_prior_v1` | all prior layers | SET on episode end | SET + any-prior (likely main interest) |
| **5** | `spk_thresh_prior_v1` | all prior layers | **Reconnect slot if utility < X** (no fixed ζ schedule) | Utility threshold vs SET’s rank/fraction drop |
| **6+** | — | — | — | Decide after 1–5 have signal; next steps might be ζ/X sensitivity only if broken, dendrites, sampled RigL grow, N scale-up — **one change each** |

**Variant 5 detail:** each edge tracks utility (e.g. EMA of \|w\| or \|w·pre\|). On episode end, every slot with `utility < X` is randomly rewired (and utility/Adam reset). No “drop 30% of edges” quota — connectivity is stable where useful, fluid where not. Pick X from a simple scale (e.g. relative to running median of utilities, or absolute on \|w\|) and keep it fixed for v1; do not sweep X until the rule is shown to matter vs SET.

**SET detail (3–4):** paper-like ζ≈0.3 of edges closest to zero, random regrow, on episode end only. No cosine ΔT.

### Build order

1. Shared `SparseKLayer` + Beta/SiLU PPO shell (flags: `any_prior`, `rewire={none,set,thresh}`).  
2. One script or thin wrappers for 1–5; submit all five to `mlq` with `--max-parallel-runs 3`.  
3. Read returns + stickiness; only then add #6+.

### Queue CLI (after build)

```bash
mlq submit --name spk_static_prev_v1 --max-parallel-runs 3 --cwd "$PWD" -- \
  .venv/bin/python -u cleanrl/sparse-nn/ppo_continuous_action_spk_v1.py \
  --env-id HalfCheetah-v4 --num-envs 16 --exp-name spk_static_prev_v1 \
  --total-timesteps 8000000 --seed 1 --rewire none --pool prev
```

(Exact flags TBD at implement time.)

## Related code

| Resource | Path | Notes |
|----------|------|--------|
| **synapses** (AlliedToasters) | [`../../../synapses`](../../../synapses) from this folder; clone lives at **`../synapses`** relative to the cleanrl repo root (`https://github.com/AlliedToasters/synapses.git`) | PyTorch **SET** reference: truly sparse weight matrices (not dense×mask), SET rewiring, and **optimizer buffer recycle** when connections reset. Starting point for implementation patterns — not a drop-in for any-prior / per-step RL. |

## References

| Paper | Local | Notes |
|-------|--------|--------|
| Mocanu et al., *Scalable training…* (Nat. Commun. 2018 / arXiv:1707.04780) | [`mocanu_2018_set.pdf`](mocanu_2018_set.pdf) · [`mocanu_2018_set.txt`](mocanu_2018_set.txt) | **SET** — magnitude drop + **random regrow**. Our primary algorithmic ancestor. |
| Evci et al., *Rigging the Lottery…* (ICML 2020 / arXiv:1911.11134) | [`evci_2020_rigl.pdf`](evci_2020_rigl.pdf) · [`evci_2020_rigl.txt`](evci_2020_rigl.txt) | **RigL** — same drop idea, **different regrow** (gradient). See clarified notes below. |
| Atashgahi et al., *QuickSelection…* (MLJ / arXiv:2012.00560) | [`atashgahi_2021_quickselection.pdf`](atashgahi_2021_quickselection.pdf) | SET DAE + neuron strength for feature selection. |

### What RigL actually does (clarified)

RigL is **not** “smarter pruning only.” Both SET and RigL keep a **fixed number of nonzero weights** and periodically **change which** connections exist. They differ only in **how new edges are chosen**:

```text
every ΔT optimizer steps (until Tend):
  1. DROP  — among currently ACTIVE weights, remove the fraction with
             smallest |θ|  (same idea as SET / magnitude pruning)
  2. GROW  — among currently INACTIVE possible weights, activate the
             same count with largest |∂L/∂θ|  evaluated on this step’s batch
  3. Init new weights to 0; continue sparse SGD on the new mask
```

Important details:

| Piece | Behavior |
|-------|----------|
| **Drop** | Magnitude on **existing** edges only. Cheap. Same family as SET. |
| **Grow** | Needs **gradients w.r.t. missing weights** — i.e. a dense (or large) grad tensor for candidates, not just active params. That is the costly bit. |
| **When** | Only every ΔT steps (paper default 100), not every step; update fraction α annealed (cosine); often stop rewiring before training ends. |
| **Vs SET** | SET grow = **uniform random** among missing edges (no dense grad). RigL grow = **top-\|grad\|** among missing edges. |
| **Vs SNFS** | SNFS grows by momentum and may track dense stats **every** step; RigL’s dense work is **infrequent**, so train FLOPs stay ~sparse if ΔT is large. |
| **Why they claim it helps** | Static masks get stuck in bad basins; growing high-grad edges steers topology toward loss reduction faster than random. |

For **our** high-frequency, small-batch, 1-epoch PPO setting:

- RigL grow every optim step ⇒ either pay dense-grad cost constantly, or approximate with sampled candidates.
- Random (SET) grow stays **O(dropped edges)** and injects topology exploration — **preferred default**.
- RigL remains a useful **ablation** / optional hybrid (e.g. 90% random + 10% grad-sampled grow), not the first line.

### SET (random regrow) — short

- Init ER sparse bipartite layers; each epoch: drop ζ nearest-zero weights; add equal **random** edges.
- Paper defaults: ζ≈0.3; ε≈11–20. Sparse from design; often matches dense with ≪ parameters.
- **synapses** implements this with real sparse mats + optimizer state hygiene on recycle.
- For us: **drop by utility/magnitude, regrow random, often** — same spirit, faster outer loop.

### Map onto sparse-nn premise

| Our idea | SET | RigL | Our default |
|----------|-----|------|-------------|
| Fixed K / fixed edge budget | yes | yes | yes |
| Any-prior / cross-layer | no | no | **yes (extension)** |
| Drop | magnitude | magnitude | magnitude / utility |
| **Regrow** | **random** | top-\|grad\| | **random** |
| How often | each epoch | every ΔT≈100 | **episode end** |
| Outer loop | multi-epoch SGD | multi-epoch SGD | **standard PPO** (freeze G in update) |
| Dendrites / clusters | hubs only | no | optional later |

Any-prior pools, dendrites, and high-frequency RL rewiring are **beyond** both papers; SET’s random regrow is the direct import for grow.

Related in-repo lineages (do not treat as ground truth; many are regressions):

- `ppo_continuous_action_morphcompute_*.py` — soft cell substrates, paid edges, lookback
- `specs/circuit_net_spec.md` + `circuitnet_*` — sparse expert circuits; post-mortem on illusory width / inert routing
- `ppo_continuous_action_densenet_v1.py` — dense any-prior concat (topology inspiration, not sparsity)

## Versions

| File | Notes |
|------|--------|
| [`ppo_continuous_action_spk_v1.py`](ppo_continuous_action_spk_v1.py) | Sparse-K trunks, SiLU, Beta. `--rewire none\|set\|thresh\|learned\|meta`. Meta: IDBD-style grad-agreement usefulness, age-protected bottom-q% rewire (no soft gate, not IDBD optimizer). |

| exp_name | pool | rewire |
|----------|------|--------|
| `spk_static_prev_v1` | prev | none |
| `spk_static_prior_v1` | prior | none |
| `spk_set_prev_v1` | prev | set |
| `spk_set_prior_v1` | prior | set |
| `spk_thresh_prior_v1` | prior | thresh (X=0.02, null) |
| `spk_thresh02_prior_v1` | prior | thresh X=0.2 |
| `spk_learned_thresh_prior_v1` | prior | learned global X (failed / soft-gate) |
| `spk_meta_rewire_prior_v1` | prior | **meta** grad-agreement, q=0.1, age≥100 |
