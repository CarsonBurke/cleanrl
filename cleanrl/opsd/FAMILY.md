# OPSD — On-Policy Self-Distillation

Teacher = student evaluated with privileged hindsight conditioning; the loss
is a per-action-dim divergence(teacher || student) with gradients through the
student only. Papers/charter: `OPSD_FAMILY.md` (supersedes
`OPSD_NORM_ABLATIONS.md`), paper `rl/2601.18734v3.pdf`, repro `OPSD_REPRO.md`.
Benchmark: HalfCheetah-v4, seed 1.

Subfamilies:

- `hopsd/` — Hindsight OPSD (v1-v44) ported onto the iterthink base.
- `hindsight/` — PPO kept as optimizer + hindsight beta-NLL teacher.
- `core/{advcond,sfocc,jacteach,covtilt,teachers,other}/` — pure-distillation
  variants, no PPO (advantage-conditioned, SF-occupancy, joint-action-teacher
  incl. entppo ports, covariance-tilt, residual teachers).
- `rl/` — `rl_opsd_*` natural-grad / logprob / target-fit redesigns.

`critic/`, `metacritic/` hold critic-side meters and probe-distillation;
only policy-distillation algorithms live here.
