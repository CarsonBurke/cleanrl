# IterThink

Iterative-thinking PPO variants: a shared ThinkTrunk backbone with Beta
policies, Dreamer3-bucket HL-Gauss / distributional critics, and latent
world-model / planning auxiliaries. Benchmark: HalfCheetah-v4, 8M steps.

Subfamilies:

- `v24_d4hlgauss/{rawret,invobs,dg,dsrg,other}/` — Dreamer4 HL-Gauss symlog
  critic line (`rawret` = raw-return heads incl. TPO-MD transfer probes,
  `invobs` = inverse-obs slots, `dg` = delight-gated, `dsrg` = decoupled
  reward-gain variants).
- `v24_d3bucket/{ppo,pmpo,other}/` — multi-token-prediction critic line.
- `dg/`, `critic_variants/`, `policy_dist/` — v24 delight-gate, v162/critic10/
  C51/rawsymlog critics, and policy-distribution sweeps (logit-flow, autoreg,
  mixture, full-cov, sacbeta).
- `lewm/`, `tpo/` — latent world-model / belief-trunk thread and TPO planning
  thread (both also appear under `_v24beta_` names; grouped by mechanism).
- `affinerms/`, `wm_k6/`, `memory_tokens/` — affine-RMS exact-KL line,
  k6 world-model/tokenizer line, tokenformer/loopbank/rolemem memory sweeps.
- `critic_sweeps/`, `outcomes/`, `early/`, `misc/` — soft-critic/soft-GAE/TD-flow
  critics, outcome/reward-model heads, v1-v23 early ablations, one-offs.

Related: `vmpo/` (holds the `fpo` flow-policy subdivisions of this line),
`tpo/` (sibling TPO family), `dg_beta/` (standalone Delightful PG).
