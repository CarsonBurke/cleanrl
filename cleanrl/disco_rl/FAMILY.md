# DiscoRL variants

Ports of DeepMind's discovered update rule, Disco103 (Oh et al., "Discovering
State-of-the-art Reinforcement Learning Algorithms", Nature 2025;
github.com/google-deepmind/disco_rl, Apache 2.0) to MuJoCo continuous control.
HalfCheetah-v4, seed 1, 8M steps.

The meta-network itself is frozen and shared: `cleanrl/shared/disco103.py` holds
the PyTorch port and the released `disco103_weights.npz`.
`tests/test_disco103.py` pins it to the JAX original through a fixture written by
`scripts/disco103_reference.py`, which runs disco_rl in a separate JAX
environment. Variants change the agent around that network, never the network.

Continuous port (v1). Disco103 is permutation-equivariant over a discrete action
set. Each state's set is K candidates: the taken action first, then K-1 draws
from the stored behaviour Beta. Their logits are log pi - log mu (sampling-
importance resampling), so behaviour logits are zero and the categorical tends to
pi as K grows. Distilling pi_hat then becomes a sampled projection,
grad = -sum_k (p_hat_k - p_k) grad log pi(a_k). The rest follows disco_rl's
agent:

- Q: a categorical two-hot over signed-hyperbolic space, from an action-
  conditional LSTM model rooted at the torso, with Retrace targets from an EMA
  target network (coefficient 0.9).
- y and z: meta-learned prediction heads, also distilled.
- Next-state policy prediction (aux_pi): predicts the next state's Beta.
- Optimizer and data: Adam with element-wise update clipping, over trajectory
  replay.

Gymnasium autoreset replaces dm_env LAST steps. Replayed segments carry T+1
rows, and row t+1 is step t's successor. At episode ends, the successor is the
stored final observation, with candidates drawn from the online policy.

- `ppo_continuous_action_disco103_v1.py`: faithful port, K=16, T=32, B=64,
  replay ratio 8, 16k-transition replay. All candidates are draws from mu, so
  Disco103's flattening of pi_hat (an entropy pull toward uniform on Atari)
  pulls toward the stale behaviour policy.
- `ppo_continuous_action_disco103_mixture_v2.py`: the taken action plus K-1
  draws from the current pi, with balance-heuristic weights over
  q = (mu + (K-1) pi)/K. The flat target is about the current pi, and the
  behaviour channel carries the off-policyness.
- `disco103_td7_v3.py`: Disco103 as the actor update on TD7 latent-Q v8 (replay
  actor, SALE, ns64). The critic, LAP replay, cadence and warmup match the base:
  one batch-256 replay update per env step and one executed action per env step.
  The policy factorizes into one categorical per action dim over evenly spaced
  bins, so the action set is enumerated exactly and no candidates are sampled.
  Each dim is one meta-batch row.
  - Per-bin Q is the critic's pre-action query at the policy's latent, with that
    dim moved to the bin. This approximates the marginal Q_d(s, b) that a
    factorized policy gradient needs.
  - The advantage adds a joint-Retrace return residual, weighted by the other
    dims' clipped ratios.
  - Actor segments are 16 x 16 steps from the newest 64k transitions.
  - Compared at 2M steps against `td7_latentq_onpolicy_v8_replay_sale_ns64`.
