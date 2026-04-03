from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v20_core


# v20: tokenized transformer context for Dreamer4-style control alignment:
# 1. explicit observation / prev-action / prev-reward / agent token chunks
# 2. actor and critic read only from the agent token
# 3. imagination feeds predicted next-step tokens back into context
@dataclass
class Args(latent_imagination_v20_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v20_core.main(Args)
