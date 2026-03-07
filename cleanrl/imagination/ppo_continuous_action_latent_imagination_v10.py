from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v10_core


# v10: no PPO. Fixes the PMPO actor-gradient bug in v9 and adds replay-action
# cloning so the actor has a grounded behavior-learning path alongside
# imagination updates.
@dataclass
class Args(latent_imagination_v10_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v10_core.main(Args)
