from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v7_core


# v7: PPO learns only from real GAE. A separate Dreamer-style imagination phase
# freezes the world model, rolls out one latent trajectory per real context,
# trains an imagination value head with lambda-returns, and applies a PMPO
# sign-based actor loss plus a policy-prior KL.
@dataclass
class Args(latent_imagination_v7_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    imag_adv_coef: float = 0.0
    use_imag_conf_gate: bool = False
    require_imag_sign_agreement: bool = False
    sign_only_imag_actor: bool = False
    use_multi_horizon_model_loss: bool = True
    multi_horizon_steps: tuple[int, ...] = (1, 2, 4, 8)
    multi_horizon_coef: float = 1.0
    model_coef: float = 0.25
    model_coef_after_imag: float = 0.1
    world_model_update_epochs: int = 1
    world_model_max_grad_norm: float = 1.0
    imagination_start_fraction: float = 0.25
    imagination_ramp_fraction: float = 0.25
    imagination_loss_coef: float = 1.0
    imagination_horizon: int = 8
    imagination_update_epochs: int = 1
    imagination_num_contexts: int = 1024
    imagination_lambda: float = 0.95
    imagination_value_coef: float = 0.5
    imagination_prior_coef: float = 0.3
    imagination_alpha: float = 0.5


if __name__ == "__main__":
    latent_imagination_v7_core.main(Args)
