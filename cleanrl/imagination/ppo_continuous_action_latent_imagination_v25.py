from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v25_core


# v25: keep the Dreamer4-inspired token world model from v24, but optimize the
# implementation around the hot path. This version removes Python-side sequence
# bookkeeping loops from training, builds imagination seed states in one
# transformer pass, and tightens cached attention masking for rollout.
# 1. explicit observation / prev-action / prev-reward / agent token chunks
# 2. actor and critic read only from the agent token
# 3. imagination feeds predicted next-step tokens back into context
@dataclass
class Args(latent_imagination_v25_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v25_core.main(Args)
