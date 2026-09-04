# TD7 dualproj v3 + WPPG Wasserstein-proximal actor (arXiv 2603.02576) v1.
#
# Baseline: td7_stocksig_sdnoise_lejepa_dualproj_v3 (17465 @ 1.44M on HalfCheetah-v4).
# Only the actor objective changes; encoder, world model, critic and replay are untouched.
#
# WHAT IS ACTUALLY BORROWED, AND WHAT IS NOT. WPPG's headline actor rule --
# transport K sampled actions along eta*grad_a Q by regressing onto a target
# displacement -- is, at one inner step, ALGEBRAICALLY THE SAME UPDATE TD7
# already performs. WPPG's Delta = a' - a0 is identically zero at the point of
# evaluation, so its gradient is -2*eta*(da/dtheta)^T grad_a Q, against TD7's
# -(da/dtheta)^T grad_a Q; clip and Adam are invariant to the uniform 2*eta, and
# the entropy noise xi has std sqrt(2*tau*eta) = 4.5e-3. Porting it verbatim
# would be a no-op. Likewise the paper's density-free machinery (plug-in mixture
# entropy, implicit pushforward policy) exists to serve policies with no
# tractable log-density; SDNoiseActor has one, so it buys nothing here.
#
# What does NOT collapse is the PROXIMAL TERM. Against the current policy it is
# zero with zero gradient, which is precisely why the paper's single-step scheme
# degenerates to a deterministic policy gradient. Measured against a LAGGED
# policy it is nonzero and bounds how far the action distribution may move
# PER STATE -- a constraint Adam's per-parameter normalization cannot express,
# for the same reason PPO's ratio clip is not redundant with Adam.
#
# SDNoiseActor is a diagonal Gaussian in (mean, log_std) before its clamp, so
# the 2-Wasserstein distance is closed form:
#     W2^2 = ||mu - mu_old||^2 + ||sigma - sigma_old||^2
# This is EXACT rather than the paper's sampled approximation, needs no nested
# autograd.grad, and stays inside torch.compile(fullgraph=True). Critically it
# is computed PRE-clamp: the actor's .clamp(-1,1) kills the gradient in
# saturated coordinates, which is exactly where a trust region must still act.
#
# Hypothesis: bounding per-state policy movement stabilizes the actor against a
# critic whose action-gradient field shifts as the encoder and world model
# co-adapt, without the pessimism cost of shrinking the learning rate globally.
#
# wppg_tr_tau = 0.01 => the reference policy lags ~100 actor updates, so the
# penalty measures drift over that horizon rather than single-step jitter.
#
# SCALE. TD7's actor loss uses a raw -Q (O(1e3) on HalfCheetah at convergence)
# while W2^2 over a 100-update lag is O(1e-2..1). The penalty is therefore
# multiplied by detached mean|Q|, making wppg_tr_coef dimensionless -- the
# fraction of current Q value one unit of W2^2 costs. Without this a coefficient
# of 1.0 would be a rounding error and the run would silently test nothing.
# wppg/w2_penalty is logged so an inert trust region is distinguishable from a
# harmful one. This arm is coef=3.0, the strong end of the bracket; v1 is coef=0.3.
import runpy
import sys
from pathlib import Path


def _has_option(flag):
    return any(argument == flag or argument.startswith(flag + "=") for argument in sys.argv)


def _enable_unless_overridden(flag):
    negative = "--no-" + flag.removeprefix("--")
    if not _has_option(flag) and not _has_option(negative):
        sys.argv.append(flag)


def _set_unless_overridden(flag, value):
    if not _has_option(flag):
        sys.argv.extend([flag, value])


_enable_unless_overridden("--wppg-trust-region")

defaults = {
    "--wppg-tr-coef": "3.0",
    "--wppg-tr-tau": "0.01",
    "--exp-name": "td7_dualproj_wppgtr_v2",
}
for flag, value in defaults.items():
    _set_unless_overridden(flag, value)

runpy.run_path(
    str(Path(__file__).with_name("td7_stocksig_sdnoise_lejepa_dualproj_v3.py")),
    run_name="__main__",
)
