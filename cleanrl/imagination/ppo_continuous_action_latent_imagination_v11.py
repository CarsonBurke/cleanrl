from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v11_core


# v11: aligns the no-PPO loop more closely with the local dreamer4 reference:
# multi-token action/reward training, behavior phase before imagination, and
# rollout-time prior regularization.
@dataclass
class Args(latent_imagination_v11_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v11_core.main(Args)
