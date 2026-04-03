from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v17_6_core


# v17.6: Full dreamer4 hyperparams on dreamerv3 RSSM:
# - LR 3e-4, no entropy, no return norm
# - Imagination horizon 8 (d4), gamma 0.99 (horizon=100)
# - PMPO with discount-weighted masked means
# - KL to prior 0.3, value coef 0.5
# - actor_minstd 0.05 (d4)
@dataclass
class Args(latent_imagination_v17_6_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v17_6_core.main(Args)
