from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v16_core


# v16: Close all remaining gaps vs dreamer4 reference:
# 1. RSSM with discrete categorical stochastic state (16x16)
# 2. Decoder + symlog-MSE reconstruction loss
# 3. KL balancing with free nats (dyn=1.0, rep=0.1)
# 4. Slow critic (EMA) + percentile return normalization
# 5. State-dependent std (sigmoid-squashed [0.1, 1.0])
# 6. REINFORCE + entropy actor loss (replaces PMPO)
# 7. Separate actor/critic optimizers
# 8. Longer imagination horizon (15)
# 9. Continue head (learned discount)
# 10. No obs/reward normalization wrappers - symlog handles scale
# 11. RMSNorm + SiLU throughout
@dataclass
class Args(latent_imagination_v16_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v16_core.main(Args)
