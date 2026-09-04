# Beta Policy

Beta-distribution policy line: beta-NLL objectives (incl. HL-Gauss hybrids and
clip-higher/no-advnorm/unbounded ablations), beta-plasticity (`betaplast`,
incl. RIPO), and entropy-PPO beta-correction / correlated-noise ports
(`entppo_betacorr`, `entppo_corrnoise`). `../gsde/` keeps the
state-dependent-exploration Beta/ReLU^2 variants; the split is NLL/plasticity
vs exploration noise.
