from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v9_core


# v9: no PPO. Real data only trains the world model and supplies replayed
# contexts. The same actor/value heads are improved in frozen-model imagination
# using PMPO, lambda-returns, and an optional persistent prior snapshot.
@dataclass
class Args(latent_imagination_v9_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v9_core.main(Args)
