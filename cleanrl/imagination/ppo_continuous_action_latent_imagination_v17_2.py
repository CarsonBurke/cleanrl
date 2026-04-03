from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v17_2_core


# v17.2: Full dreamerv3/dreamer4 alignment:
# 1. RSSM sized to dreamerv3 defaults (512 deter, 32x32 stoch, 256 hidden)
# 2. Learning rates 1e-4 (was 3e-4)
# 3. PMPO with boolean mask + masked_mean (dreamer4 style)
# 4. Raw advantages (no return normalization for PMPO)
# 5. Value clipping (dreamer4) replaces slow-critic regularization
# 6. No discount weighting on critic loss
# 7. MLP units 512 (was 256)
# 8. Max grad norm 1000 (was 100)
@dataclass
class Args(latent_imagination_v17_2_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v17_2_core.main(Args)
