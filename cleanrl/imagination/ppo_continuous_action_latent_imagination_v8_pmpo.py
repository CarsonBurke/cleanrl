from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v8_pmpo_core


# v8_pmpo: Fork of v8 with PMPO replacing PPO's clipped surrogate for the
# behavior policy. Imagination phase already uses PMPO-style loss unchanged.
@dataclass
class Args(latent_imagination_v8_pmpo_core.Args):
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
    imagination_bc_coef: float = 1.0
    imagination_bc_after_start_coef: float = 0.05
    context_hidden_dim: int = 64
    imagination_num_bins: int = 255
    imagination_bin_range: float = 3.0
    # PMPO behavior defaults
    norm_adv: bool = False
    pmpo_alpha: float = 0.5
    pmpo_kl_coef: float = 0.3
    pmpo_reverse_kl: bool = True


if __name__ == "__main__":
    latent_imagination_v8_pmpo_core.main(Args)
