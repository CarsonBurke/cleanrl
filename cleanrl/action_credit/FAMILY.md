# Action Credit

Per-action credit baselines with ReLU^2 backbones: a per-action-dimension
value head on the shared actor trunk gives `A_i = return - V_i(s)` for
per-coordinate PPO ratios, keeping the scalar critic for GAE. Covers
dynsens/tanh/credit-GAE/score-projection/zerosum-prediction ablations.
