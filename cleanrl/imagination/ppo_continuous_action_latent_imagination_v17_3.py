from dataclasses import dataclass
import os

from cleanrl.imagination import latent_imagination_v17_3_core


# v17.3: Exact dreamerv3 reference alignment:
# 1. PercentileReturnNorm (rate=0.01, limit=1.0) — matches ref exactly
# 2. Advantages = (returns - values) / scale — not double-normalize
# 3. LR 1e-4 for all (was 3e-4)
# 4. Entropy 3e-4 (was 0.01)
# 5. Max grad norm 1000 (was 100)
# 6. Weight = cumprod(con) — no /cont_target_val
# 7. Critic: discount-weighted (ret_loss + slow_reg * slow_loss) — matches ref
# 8. Entropy is discount-weighted in actor loss
# 9. PMPO uses boolean mask + masked_mean (dreamer4 style)
@dataclass
class Args(latent_imagination_v17_3_core.Args):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]


if __name__ == "__main__":
    latent_imagination_v17_3_core.main(Args)
