from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v8_core
from cleanrl.imagination.ppo_continuous_action_latent_imagination_v8_ablation_actor_warmup import Args as WarmupArgs


@dataclass
class Args(WarmupArgs):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 6e-4


class Agent(latent_imagination_v8_core.Agent):
    def behavior_optimizer_groups(self, args: Args, default_lr: float):
        actor_lr = args.actor_learning_rate if args.actor_learning_rate > 0 else default_lr
        critic_lr = args.critic_learning_rate if args.critic_learning_rate > 0 else default_lr
        return [
            {
                "params": list(self.encoder.parameters()) + list(self.critic.parameters()),
                "lr": critic_lr,
            },
            {
                "params": list(self.actor_mean.parameters()) + [self.actor_logstd],
                "lr": actor_lr,
            },
        ]


if __name__ == "__main__":
    latent_imagination_v8_core.main(Args, agent_class=Agent)
