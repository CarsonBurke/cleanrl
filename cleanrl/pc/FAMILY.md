# Predictive Coding (PC)

Streaming predictive-coding actor-critics with Fisher output geometry and
exact per-env/per-parameter TD(lambda) traces: no rollout buffer, no GAE, no
PPO objective — ten reverse block Gauss-Seidel sweeps propagate output errors
through Gaussian hidden states under a streaming AdamW optimizer. Includes
Fisher/clamped/bounded/momentum/eligibility variants and the conventional
backprop TD(lambda) control.
