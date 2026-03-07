from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v13_core


# v13: switches imagination learning to a Dreamer-style two-stage path:
# generate imagined experience under an old policy/value snapshot, then optimize
# the current policy/value on that fixed imagined batch.
@dataclass
class Args(latent_imagination_v13_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v13_core.main(Args)
