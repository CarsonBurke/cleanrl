from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v14_core


# v14: adds Dreamer4-style relevant-sequence mixture sampling for behavior and
# imagination updates.
@dataclass
class Args(latent_imagination_v14_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v14_core.main(Args)
