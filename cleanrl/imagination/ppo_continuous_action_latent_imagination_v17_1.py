from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v17_1_core


# v17.1: Bug fixes for dreamerv3/dreamer4 alignment:
# 1. Discount-weighted PMPO actor loss (was binary mask)
# 2. Discount weight includes starting state continuation
# 3. Discount-weighted critic loss
# 4. KL direction fixed to KL(old||new) matching dreamer4
# 5. Value clipping replaced with dreamerv3 slow-critic regularization
@dataclass
class Args(latent_imagination_v17_1_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v17_1_core.main(Args)
