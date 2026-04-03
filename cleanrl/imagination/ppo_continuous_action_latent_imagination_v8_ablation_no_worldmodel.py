# Ablation: v8 with NO world model training at all.
# - model_coef=0.0, model_coef_after_imag=0.0 (world model loss zeroed)
# - imagination_loss_coef=0.0 (no imagination, which needs the world model)
# - value_consistency_coef=0.0 (part of world model loss)
# - target_encoder_tau=1.0 (target encoder unused without world model)
#
# What remains: ONLY the shared encoder→latent→{actor,critic} architecture
# trained by standard PPO loss. This isolates whether the architectural
# change (shared latent bottleneck) alone explains v8's performance.
from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v8_core


@dataclass
class Args(latent_imagination_v8_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    imag_adv_coef: float = 0.0
    use_imag_conf_gate: bool = False
    require_imag_sign_agreement: bool = False
    sign_only_imag_actor: bool = False
    use_multi_horizon_model_loss: bool = True
    multi_horizon_steps: tuple[int, ...] = (1, 2, 4, 8)
    multi_horizon_coef: float = 1.0
    world_model_update_epochs: int = 0
    world_model_max_grad_norm: float = 1.0
    imagination_start_fraction: float = 0.25
    imagination_ramp_fraction: float = 0.25
    imagination_horizon: int = 8
    imagination_update_epochs: int = 1
    imagination_num_contexts: int = 0
    imagination_lambda: float = 0.95
    imagination_value_coef: float = 0.5
    imagination_prior_coef: float = 0.3
    imagination_alpha: float = 0.5
    imagination_bc_coef: float = 0.0
    imagination_bc_after_start_coef: float = 0.0
    context_hidden_dim: int = 64
    imagination_num_bins: int = 255
    imagination_bin_range: float = 3.0
    # ABLATION: everything disabled — pure latent-PPO architecture
    model_coef: float = 0.0
    model_coef_after_imag: float = 0.0
    value_consistency_coef: float = 0.0
    imagination_loss_coef: float = 0.0
    target_encoder_tau: float = 1.0


if __name__ == "__main__":
    latent_imagination_v8_core.main(Args)
