# Critic Architecture & Value Learning

Experiments modifying the critic's architecture, inputs, or training objective to improve value estimation and provide richer gradient signal to the actor.

## Variants

| File | Approach |
|-|-|
| `_logits_critic` | Distribution-conditioned critic: V(s, mu, sigma) sees actor's distribution parameters, providing directional gradient signal |
| `_logits_critic_base` | Base version of logits-aware critic |
| `_critic_latent` | Critic receives actor's latent representation (gradients flow back through actor trunk) |
| `_critic_latent_detach` | Same as above but with detached gradients (no backprop through actor) |
| `_adv_heads` | Advantage decomposition: auxiliary heads predict mean-direction and exploration-magnitude components |
| `_dg` | Delightful Policy Gradient: gates policy gradient terms with sigmoid(advantage * surprisal) to amplify rare breakthroughs |
