# PMPO-D4

Plastic Mirror-descent Policy Optimization, 4th iteration: Beta policies with
ReLU^2 backbones and SPO-asymmetric trust regions. Benchmark: HalfCheetah-v4.

Subfamilies:

- `spo/` — SPO-asymmetric variants (incl. half-strength, cubic, sep-clip,
  HL-Gauss ports and the Tanh-backbone sibling).
- `beta/` — ReLU^2 Beta core (adv-norm, clip-DAPO, actor-hidden sweeps).
- `geometry/` — output-geometry sweep (actioncell, tanh-normal/mean, nosquash,
  ln-mean, HL-Gauss actions, cat/gauss bins).

Sibling `../pmpo/` keeps the original PMPO v1-v3 + LSTD references.
