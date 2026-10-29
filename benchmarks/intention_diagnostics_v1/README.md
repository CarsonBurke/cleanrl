# intention_diagnostics_v1 (INCOMPLETE: stopped when the direction was dropped)

Offline checks on whether a "k-step intention" design is worth building. In that design an actor picks g = zs(s_{t+k}), a head pi(s, g) decodes actions, and a critic Q(s, g) values g. Script: `diag.py`. Data: `runs/HalfCheetah-v4__td7_latentq_delight_reference_v10_latent_all__1__1791861435/checkpoint.pt`, written by `cleanrl/td7/delight/td7_latentq_delight_reference_v10.py` at step 737k, eval about 10.5k. The checkpoint holds the replay: 720,896 rows of raw obs and actions. We used the last 20 rollouts (327,680 rows) and held out the last 2 rollouts. We restored MuJoCo from obs with qpos[0] = 0 (root x does not affect dynamics or reward). Check: max next-obs error 8e-4, median 0, max reward error 7e-6.

Jobs: 14784 (stages C and A, finished, `results/stages_C_A.json`) and 14785 (stage R, cancelled partway; there is no JSON for it, and the numbers below are from its log).

## C: how much the first action varies among near-identical (s, g)

Space: s with each dimension z-scored, plus g = zs(s_{t+k}) with each dimension z-scored, each block scaled to unit norm. Neighbours from the same env within 10 steps are excluded. Metric: action variance inside a neighbourhood divided by the total action variance, K=8 neighbours, 5000 queries.

| k | neighbour radius / random-pair distance | spread (CI95) | spread in the tightest 10% |
|---|---|---|---|
| 0 (s only) | 0.17 | 0.389 (0.382-0.395) | 0.308 |
| 1 | 0.24 | 0.145 (0.142-0.149) | 0.092 |
| 4 | 0.29 | 0.319 (0.313-0.324) | 0.217 |
| 8 | 0.31 | 0.402 (0.395-0.408) | 0.256 |

The spread comes from the policy mean differing between neighbours, not from noise: the policy-mean spread ratio is about the same as the action spread ratio. The actor is bang-bang: 83.5% of mean-action dimensions have |mu| > 0.99. At k >= 4, knowing g adds almost nothing about a_t beyond what s already gives.

Head error on held-out data (error / action variance). k=1: MSE head 0.014, mixture head mode 0.023, mixture best-of-5 0.009. k=4: MSE head 0.176, mixture mode 0.222. Verdict: **for k >= 4 the head is badly under-determined. A mixture head picking its top mode did not beat the MSE head**, so this is not clean multimodality.

## A: how much of the critic's zsa gradient the action can actually move

Setup: 20k replay states. J = d zsa / da at the actor mean. We measure the share of ||dQ/dzsa||^2 that lies in range(J).

- Share in range(J): **0.032 (0.032-0.033)**, median 0.017. A random direction gives 0.023.
- Yet the zsa path carries 83% of |dQ/da| (cosine 0.89 with the full action gradient).
- zsa predicts the next zs with held-out R² 0.990.

Verdict: **intentions must be on-manifold by construction**. About 97% of the critic's latent gradient points where execution cannot follow. Not measured: the same share against the MuJoCo finite-difference Jacobian (it was part of R).

## R: does ascending the intention critic deliver what it claims (partial)

Setup: 4000 held-out restored states. rho = realized / claimed change in value. Realized = return over the executed steps plus γ^k V from the checkpoint's critic. The fitted critics are accurate on held-out data: R² 0.999 for k=1, 4 and 8.

Action-space ascent, k=1, tail0 (checkpoint critic, per eta):
- eta 0.05: rho 1.25 (1.10-1.40)
- eta 0.1: rho 1.20 (1.05-1.35)
- eta 0.2: rho 1.07 (0.87-1.27)

A critic fit from scratch on actions gives rho 0.76 / 0.65 / 0.36 at the same etas. The tail16 variant was noisy or negative; I could not interpret it before the stop.

Intention ascent, k=1, MSE head, eta = 0.1 to 1.0 times the median displacement D_k:
- Claimed gain: 52-166 reward units.
- rho: **-0.12 to 0.00**, with CIs about ±0.01.
- Execution error / step: 1.1-1.6.
- Movement along the intended direction: about 0.

The mixture head at eta 0.1 gave the same result. Not measured: k=4 and k=8, and the A check against MuJoCo.

## Decision (based on partial data)

Rule: kill if rho_g < action rho (or < 0.5), or if execution error exceeds the step at a useful eta. For k=1, rho_g is about 0, against about 1.2 for action-space ascent, and execution error is above the step. **Kill.** Also flagged: on-manifold intentions are required (A), and the head is under-determined for k >= 4 (C).
