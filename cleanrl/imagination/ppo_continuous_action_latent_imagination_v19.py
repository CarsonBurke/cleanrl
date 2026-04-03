from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v19_core


# v19: replace the RSSM with a causal transformer world model while keeping
# the v18 actor-critic update path:
# 1. latent state comes from transformer history instead of GRU + stochastic state
# 2. imagined rollout appends autoregressive transformer tokens
# 3. actor/critic still train with GAE, PMPO, reverse KL, and clipped values
@dataclass
class Args(latent_imagination_v19_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v19_core.main(Args)
