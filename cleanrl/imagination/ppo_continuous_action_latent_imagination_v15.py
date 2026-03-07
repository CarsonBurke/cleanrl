from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v15_core


# v15: closer Dreamer4 phase alignment with behavior MTP before imagination and
# uniform imagination contexts afterward.
@dataclass
class Args(latent_imagination_v15_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v15_core.main(Args)
