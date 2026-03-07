from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_core


# v3: safer critic-side latent imagination with delayed, small imagined actor
# mixing late in training once the model has had time to mature.
@dataclass
class Args(latent_imagination_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_core.main(Args)
