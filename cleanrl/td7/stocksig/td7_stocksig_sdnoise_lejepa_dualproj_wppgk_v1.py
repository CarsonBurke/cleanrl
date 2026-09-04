# TD7 dualproj v3 + WPPG K-sample action transport (arXiv 2603.02576) v1.
#
# Baseline: td7_stocksig_sdnoise_lejepa_dualproj_v3 (17446 @ 1.44M on HalfCheetah-v4).
# Only the actor objective changes; encoder, world model, critic and replay untouched.
#
# WHY THIS AND NOT THE TRUST REGION. The proximal term was tested first
# (wppgtr_v1, coef=0.3) and it LOSES: 4866 vs 9327 @100k, 7993 vs 11916 @200k.
# That is the predicted failure -- TD7 carries no importance-sampling ratio, so
# there is no stale surrogate for a trust region to protect, and penalising
# movement against a lagged snapshot brakes sustained directional improvement,
# which in a DPG-family method IS learning. Measured w2 was only 1.2e-3
# (||dmu|| ~ 0.035 over a 100-update lag), i.e. ~0.04% of the loss value yet ~2%
# of the actor gradient -- enough to cost a third of the score.
#
# What survives is the SAMPLING half of WPPG's actor rule. SDNoiseActor is
# stochastic: a = clamp(mu + sigma*eps, -1, 1), so the K=1 actor gradient is a
# ONE-SAMPLE estimate of E_eps[grad_a Q(s, a(s,eps))]. WPPG transports K actions
# per state instead of one. Averaging K independent draws cuts the eps-induced
# gradient variance by ~1/K with the SAME expectation -- no bias, no new
# hyperparameter tension, no brake on the update direction.
#
# Second-order benefit from the clamp: gradient is identically zero in
# coordinates where mu + sigma*eps saturates past +-1. At K=1 a saturated draw
# wastes that coordinate's update entirely; at K=4 the unsaturated draws still
# carry signal, so the effective gradient is denser as well as lower-variance.
#
# Hypothesis: variance reduction in the actor gradient improves sample
# efficiency, and unlike the proximal term it cannot slow the policy down,
# because it changes only the estimator's noise and not its mean.
#
# Cost: K critic evaluations on the actor step, which runs every policy_freq
# steps. K=4 is the variance/compute knee -- 1/K is already 0.25 there.
import runpy
import sys
from pathlib import Path


def _has_option(flag):
    return any(argument == flag or argument.startswith(flag + "=") for argument in sys.argv)


def _set_unless_overridden(flag, value):
    if not _has_option(flag):
        sys.argv.extend([flag, value])


defaults = {
    "--wppg-action-samples": "4",
    "--exp-name": "td7_dualproj_wppgk_v1",
}
for flag, value in defaults.items():
    _set_unless_overridden(flag, value)

runpy.run_path(
    str(Path(__file__).with_name("td7_stocksig_sdnoise_lejepa_dualproj_v3.py")),
    run_name="__main__",
)
