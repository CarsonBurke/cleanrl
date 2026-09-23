# Latent costate v11: three separate ablations

The reference is the completed fresh v10 run with projection weight zero (job 7527), final last-100 training return 5831.05. This study changes one setting per arm; it does not combine discounting, full-batch fitting, and fewer Bellman target refreshes.

| Arm | Gamma | Fitting batch | Critic target refreshes / rollout | Model Adam steps / rollout | Critic Adam steps / rollout |
| --- | ---: | ---: | ---: | ---: | ---: |
| v10 reference | 1 | 4096 | 8 | 80 | 64 |
| Gamma only | 0.99 | 4096 | 8 | 80 | 64 |
| Full batch only | 1 | 32768 | 8 | 10 | 8 |
| No repeated sweeps only | 1 | 4096 | 1 | 80 | 64 |

Each arm collects fresh data from random initialization: HalfCheetah-v4, seed 1, nominal 8M transitions, 16 environments, 2048 steps per rollout, two environment threads. All use the same latent costate architecture, fixed normalization, dynamics fitting loss, no projection loss, no replay, no GAE, and exact measured mean KL budget 0.03. No checkpoint is loaded. v10 is unchanged.

## Hypotheses and interpretation

- Gamma 0.99 attenuates long-range derivative propagation and model-Jacobian error. The future-state cotangent alone is discounted; immediate progress and action cost derivatives are not. The actor retains uniform rollout-state weighting. This is a discounted-costate improvement surrogate, not an exact episode-start discounted-return gradient. Evaluation remains undiscounted episodic training return.
- Full-batch fitting removes within-pass optimizer ordering and minibatch-gradient variation. Ten model passes and eight critic passes match the reference's data exposure, **not** its number of Adam updates. The model and critic have eight times fewer optimizer updates; changing their learning rates to compensate would be another intervention. Actor fitting already uses a full rollout and is unchanged.
- Removing repeated sweeps holds one detached Bellman target fixed for all eight critic fitting passes. This isolates repeated target recomputation from fitting effort. Bootstrapping remains; this is not a zero-future-costate arm. After fitting, actor credit is evaluated from the final critic exactly as in the reference, without another critic fit.

Discounting does not guarantee contraction of the derivative Bellman recursion. Full-batch fitting does not fix model-Jacobian errors. A single seed can identify promising configurations, not establish robust superiority or prove the latent representation caused the result.

## Actor-update accounting

The reference artifact contains 245 iterations and 245 positive accepted scales. Each iteration collects 32768 transitions and commits at most one actor update. There were 2940 line-search proposals (12 per update), not 2940 committed updates. Fifty conjugate-gradient iterations solve the natural-gradient direction; they do not train the actor fifty times. The measured mean KL per accepted update was 0.0299465.

The nominal 8M budget rounds up to 245 complete rollouts (8028160 transitions), plus 16000 stochastic warmup transitions: 8044160 total. The final policy update is followed by no new rollout, so the final training-return window principally measures policies before that last update. Consistent aggregate improvement is compatible with 245 substantial policy changes, but is not a guarantee of improvement at each step.

v11 logs cumulative attempted and accepted actor updates, model/critic optimizer steps, and critic training-target refreshes in TensorBoard and JSON.

## Verification and jobs

CUDA contracts: job 7540, **23 passed**, parallel limit 1, time limit 20 minutes. Initial job 7539 exposed gamma rounding through an FP32 continuation mask in the FP64 reference test (22 passed, 1 failed); multiplication now retains the costate precision before applying gamma. Tests include discount placement against full autograd (gamma 0, 0.99, 1; terminal and continuing), observed Beta transport, latent pullback gradients, compiled critic/actor paths, batch coverage, and independent target-refresh/fitting schedules.

Independent source review found no blocking defect. Training jobs: **7541 gamma 0.99**, **7542 full batch**, **7543 no repeated sweeps**; all depend on successful contract job 7540. All training jobs use parallel limit 1 and time limit 60 minutes. Underperforming runs are assessed after sufficient training; a short noisy regression alone is not a cancellation criterion.

## Results

All values below are last-100 stochastic training episode returns, seed 1. The two weak arms were cancelled under the repository's instruction to stop clearly underperforming experiments; their stopped values are not 8M results.

| Arm | Job | Matched 2.01M return | Last return | Last collected steps | Status | Mean actual KL |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| v10 reference | 7527 | 4073.23 | 5831.05 | 8,044,160 | Completed | 0.029947 |
| Gamma 0.99 only | 7541 | 3286.71 | 5534.41 | 8,044,160 | Completed | 0.029917 |
| Full batch only | 7542 | -63.98 | 3.84 | 2,866,816 | Cancelled | 0.029983 |
| No repeated sweeps only | 7543 | 805.16 | 946.80 | 4,013,696 | Cancelled | 0.015516 |

Gamma 0.99 finishes 5.1% below the reference. Its final next-costate RMS is lower (6.97 versus 9.29), but smaller derivatives do not establish better credit, and this run does not support promoting the discount change.

Full batch peaks at 178.46 around 1.46M, then returns near zero. The actual actor KL remains near 0.03, so the failure is not explained by actor updates being uniformly smaller. The model/critic have eight times fewer Adam steps per rollout. This rejects the equal-data-pass full-batch schedule; it does not isolate an inherent disadvantage of full-batch gradients or determine whether dynamics fitting, critic fitting, or both are responsible.

No repeated sweeps peaks at 1060.87 around 2.83M, falls to 85.02 at 3.00M, then recovers to 946.80 at 4.01M (reference 5620.76). It retains the reference's optimizer steps and dataset passes. Repeated target refreshes therefore matter in the current implementation, but this comparison does not establish that eight is optimal. Mean actual actor KL is 0.01552 rather than 0.02995: the changed targets also induce different actor directions and accepted step sizes. An identical ceiling does not guarantee an identical realized update.

None of these ablations is promoted. Keep the existing v10 gamma 1/minibatch/eight-refresh reference. These experiments do not isolate latent-feature utility or establish model-Jacobian accuracy.

## Evidence

- [Learning curves and costate magnitudes](latent-costate-ablation-v11-results.png)
- [Vector graphic](latent-costate-ablation-v11-results.svg)
- [Machine-readable measurements](latent-costate-ablation-v11-results.json)
- Reproducible aggregation: `scripts/latent_costate_ablation_v11_report.py`; MLQ job 7544 succeeded, parallel limit 1, time limit 5 minutes.
- Each cancelled run retains its original `progress.json` and a separate `ablation_outcome.json`; no completed result or 8M score is fabricated.
- Accepted actor counts: reference 245/245, gamma 245/245, full-batch 87/87, no-sweeps 122/122. Completed 8M runs also record 19600 model and15680 critic optimizer steps. The gamma arm has 1960 critic target refreshes; no-sweeps has 122 across 122 rollouts.

Frozen v11 source SHA256: `3e37768176de50d8b06efc7e5dc800d76bcb854ac590c1f9549dcc30de1b8434`.
