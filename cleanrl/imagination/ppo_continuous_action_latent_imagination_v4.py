from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_core


# v4: confidence-gated imagined actor correction. The actor only sees the
# imagined signal when branch disagreement is low and the imagined and real
# advantages point in the same direction.
@dataclass
class Args(latent_imagination_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    imag_adv_coef: float = 0.08
    use_imag_conf_gate: bool = True
    require_imag_sign_agreement: bool = True
    imag_std_gate_temperature: float = 1.0
    imag_adv_clip_multiplier: float = 1.5


if __name__ == "__main__":
    latent_imagination_core.main(Args)
