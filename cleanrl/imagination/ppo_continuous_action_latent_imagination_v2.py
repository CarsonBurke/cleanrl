from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_core


# v2 fork: keep the safer critic-side world-model setup, but remove the v3
# delayed imagined-advantage schedule so the actor stays on pure real GAE.
@dataclass
class Args(latent_imagination_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    imag_adv_coef: float = 0.0


if __name__ == "__main__":
    latent_imagination_core.main(Args)
