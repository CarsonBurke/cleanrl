from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v8_core
from cleanrl.imagination.ppo_continuous_action_latent_imagination_v8_ablation_actor_warmup import Args as WarmupArgs


@dataclass
class Args(WarmupArgs):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    extra_critic_update_epochs: int = 1
    extra_critic_start_fraction: float = 0.0625


if __name__ == "__main__":
    latent_imagination_v8_core.main(Args)
