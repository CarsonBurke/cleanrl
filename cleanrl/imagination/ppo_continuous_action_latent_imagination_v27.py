from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v27_core


# v27: Bug fixes over v26 — fixed critic clipped value loss, proper grad clipping,
# vectorized seed construction, more seed diversity.
@dataclass
class Args(latent_imagination_v27_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v27_core.main(Args)
