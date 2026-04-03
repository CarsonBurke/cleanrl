"""Imagination v8.4 — Full v8 with imagination disabled.

Same as v8 (encoder, world model, context RNN, all machinery present) but with
imagination_loss_coef=0 so imagination rollouts, BC loss, prior, and phase
scheduling never activate. Tests whether imagination actually helps or if
the world model auxiliary loss alone explains v8's performance.
"""
from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v8_core


@dataclass
class Args(latent_imagination_v8_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    imagination_loss_coef: float = 0.0
    imagination_bc_coef: float = 0.0
    imagination_bc_after_start_coef: float = 0.0


if __name__ == "__main__":
    latent_imagination_v8_core.main(Args)
