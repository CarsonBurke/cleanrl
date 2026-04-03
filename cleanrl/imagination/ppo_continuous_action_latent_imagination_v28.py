from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v28_core


# v28: Fix PMPO actor loss explosion — EMA target actor, return normalization,
# log_prob clamping for stability.
@dataclass
class Args(latent_imagination_v28_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v28_core.main(Args)
