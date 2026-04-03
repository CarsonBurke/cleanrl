from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v18_core


# v18: align the RL update path with dreamer4 where architecture permits:
# 1. GAE with rollout-time critic values instead of learned-continuation lambda returns
# 2. Value clipping anchored on rollout-time old values
# 3. Reverse PMPO KL implemented as KL(old || new)
# 4. Gaussian actor uses mean/log-variance, matching the reference parameterization
# 5. Keep comments honest: this remains RSSM-based, not the dreamer4 architecture
@dataclass
class Args(latent_imagination_v18_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v18_core.main(Args)
