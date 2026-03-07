from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v12_core


# v12: adds reward-relevant behavior weighting and replayed rollout-policy KL so
# the behavior phase learns a stronger prior from better online segments.
@dataclass
class Args(latent_imagination_v12_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v12_core.main(Args)
