# DG-Beta — Delightful Policy Gradients

Clean Beta-policy implementation of Delightful Policy Gradients
(arXiv:2603.14608v1): surprisal-gated score-function updates
(`w = sigmoid(adv * surprisal / eta)`, detached). Linear v1-v30 sweep plus
KL-trust/cap/rescale, fullbatch clip-ablations, think-trunk and successor-feature
critic ports. Benchmark: HalfCheetah-v4, seed 1.
