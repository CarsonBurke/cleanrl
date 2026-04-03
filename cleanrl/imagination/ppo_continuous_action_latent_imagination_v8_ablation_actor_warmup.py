from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v8_core
from cleanrl.imagination.ppo_continuous_action_latent_imagination_v8_ablation_no_worldmodel import Args as CleanArgs


@dataclass
class Args(CleanArgs):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    behavior_actor_warmup_fraction: float = 0.0625
    behavior_actor_ramp_fraction: float = 0.0625


if __name__ == "__main__":
    latent_imagination_v8_core.main(Args)
