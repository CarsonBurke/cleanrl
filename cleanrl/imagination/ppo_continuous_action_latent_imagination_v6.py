from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v6_core


# v6: world-model-first training structure. Keep PPO as the real-data trust
# region, but split behavior learning from world-model learning so later
# imagination-heavy variants do not rely on one shared backward pass.
@dataclass
class Args(latent_imagination_v6_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    imag_adv_coef: float = 0.08
    imag_branches: int = 1
    use_multi_horizon_actor_stats: bool = True
    use_imag_conf_gate: bool = True
    require_imag_sign_agreement: bool = True
    sign_only_imag_actor: bool = True
    imag_std_gate_temperature: float = 1.0
    imag_adv_clip_multiplier: float = 1.5
    use_multi_horizon_model_loss: bool = True
    multi_horizon_steps: tuple[int, ...] = (1, 2, 4, 8)
    multi_horizon_coef: float = 1.0
    model_coef: float = 0.25
    model_coef_after_imag: float = 0.1
    world_model_update_epochs: int = 1
    world_model_max_grad_norm: float = 1.0


if __name__ == "__main__":
    latent_imagination_v6_core.main(Args)
