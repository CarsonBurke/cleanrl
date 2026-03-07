from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v17_core


# v17: Further alignment with dreamer4 reference:
# 1. Clipped value loss (clip=0.4) instead of plain twohot CE
# 2. EMA mean/std return normalization (normalizes both returns and values)
# 3. Learning rate 3e-4 (matches dreamer4)
# 4. Remove broken KL constraint (was always 0), store old log probs properly
# 5. All v16 features retained: RSSM, PMPO, continue head, symlog, etc.
@dataclass
class Args(latent_imagination_v17_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v17_core.main(Args)
