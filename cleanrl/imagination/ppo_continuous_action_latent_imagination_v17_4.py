from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v17_4_core


# v17.4: Pure dreamerv3 reference alignment:
# Actor loss: -(weight * (logpi * advantages + ent_coeff * entropy)).mean()
# No PMPO, no KL constraint — exact dreamerv3 REINFORCE actor
# All other settings match dreamerv3 ref (LR 1e-4, entropy 3e-4, etc.)
@dataclass
class Args(latent_imagination_v17_4_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v17_4_core.main(Args)
