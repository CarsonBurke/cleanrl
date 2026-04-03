import copy
from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v8_core
from cleanrl.imagination.ppo_continuous_action_latent_imagination_v8_ablation_no_worldmodel import Args as CleanArgs


@dataclass
class Args(CleanArgs):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


class Agent(latent_imagination_v8_core.Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.critic_encoder = copy.deepcopy(self.encoder)

    def get_value(self, obs):
        return self.critic(self.critic_encoder(obs))

    def get_action_and_value(self, obs, action=None):
        actor_latent = self.encode(obs)
        probs = self.get_dist_from_latent(actor_latent)
        if action is None:
            action = probs.sample()
        value = self.critic(self.critic_encoder(obs))
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

    def behavior_parameters(self):
        params = []
        params.extend(self.encoder.parameters())
        params.extend(self.actor_mean.parameters())
        params.append(self.actor_logstd)
        params.extend(self.critic_encoder.parameters())
        params.extend(self.critic.parameters())
        return params


if __name__ == "__main__":
    latent_imagination_v8_core.main(Args, agent_class=Agent)
