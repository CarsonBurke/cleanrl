from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v24_core


# v24: port the main remaining Dreamer4-style transformer ideas that fit this
# codebase: softclamped grouped attention, SwiGLU FFNs, value-residual mixing,
# reward embeddings into the agent token, register tokens, and loss
# normalization, while keeping v23's batched observe and cached online path.
# 1. explicit observation / prev-action / prev-reward / agent token chunks
# 2. actor and critic read only from the agent token
# 3. imagination feeds predicted next-step tokens back into context
@dataclass
class Args(latent_imagination_v24_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v24_core.main(Args)
