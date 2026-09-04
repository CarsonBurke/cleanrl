# Native PG

Score-first native policy gradients with sigma/trust-region ablations: keeps
the raw native-reward update from the DG experiments, factors state-dependent
vs global sigma against an optional exact-KL backtrack of one fresh Adam
proposal. No data, gradient, or gate reuse.
