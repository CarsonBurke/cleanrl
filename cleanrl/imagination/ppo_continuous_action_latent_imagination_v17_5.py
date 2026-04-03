from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v17_5_core


# v17.5: Dreamer4-aligned PMPO on dreamerv3 base:
# 1. PMPO with discount-weighted masked means (dreamer4: alive_weights[mask] * logprobs[mask])
# 2. No entropy in actor loss (dreamer4 uses KL to prior instead)
# 3. Discount-weighted KL to rollout-time policy (dreamer4 prior regularization)
# 4. All other settings from v17.3 (percentile return norm, LR 1e-4, dreamerv3 critic)
@dataclass
class Args(latent_imagination_v17_5_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v17_5_core.main(Args)
