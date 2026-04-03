from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v26_core


# v26: Dreamer v4-aligned training loop. 1 env default (imagination-heavy),
# unified train_step with WM backward then frozen-WM imagination + AC.
# Encode+observe once, reuse for both WM loss and imagination seeding.
@dataclass
class Args(latent_imagination_v26_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v26_core.main(Args)
