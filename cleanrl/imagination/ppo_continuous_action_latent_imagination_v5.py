from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v5_core


# v5: direct multi-horizon world-model supervision plus sign-only imagined
# actor correction. Imagination stays small, gated, and late, while the model
# shifts to a slower auxiliary role once actor-side imagination turns on.
@dataclass
class Args(latent_imagination_v5_core.Args):
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


if __name__ == "__main__":
    latent_imagination_v5_core.main(Args)
