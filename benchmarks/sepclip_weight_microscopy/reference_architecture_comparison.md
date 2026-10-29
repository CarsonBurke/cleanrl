# Saved stronger-reference architecture comparison

Status: current-source static comparison; no new physical evaluation, model forward, or causal conclusion. The saved `prerms_adamh_v22_control_50M` policy scores 14522.67 over the last twenty training-return scalars at 49987008 steps. It has complete saved actor/critic weights, Args and sixteen observation-normalization rows. This 50M historical training budget differs from the sepclip 8M investigation, and its execution source is not pinned. Current source compatibility and strict loading cannot retroactively establish identical historical source content.

The saved reference uses `optimizer_kind=adam`, so the Hyperball/AdamH alternative in the same file is inactive. Every parameter is unconstrained Adam with rate .0096 and eps1e-5. No per-row/column retraction is applied, and `grad_clip=False`. The sepclip v7 uses rate .0024 for Adam auxiliaries, its selected matrix optimizer for actor branch banks, and repeated unit-row gate/up/head plus unit-column down normalization in both actor and critic. Its independent actor/critic .5 clipping is a separate conditioning difference. These differences cannot be assigned causal responsibility from cross-method scores.

The reference actor/critic each start with a biased learned 17-to64 projection, then three 64-to64 SiTU branches with43 gated channels. Its pre-RMS residual equations are:

    h0 = E*x + b
    hj+1 = hj + sigmoid(rj) * Bj(RMS(hj))
    logits = Whead * [RMS(h3)/8] + bhead

RMS is non-affine with epsilon1e-5; each residual gate is a learned64-vector initialized to-1.5. The identity stream therefore carries a signed linear input anchor through every block before final RMS. The initial input projection gain is sqrt(64/17), SiTU gate/up initialization gain is sqrt(2), and branch down initialization targets raw variance .5. The actor head's orthogonal .01 gain remains intact; it is not immediately retracted to unit rows. The critic head begins at gain1. There is no separate learned readout-gain vector.

Sepclip v7 instead consumes the17-coordinate observation directly in its first bias-free17-to64 SiTU branch, normalizes that branch output, adds one normalized second64-to64 branch, then normalizes the sum:

    h = unit(B1(x))
    u = unit(B2(8*h))
    logits = gain * [Whead * (8*unit(h+u)) + bhead]

There is no learned linear observation projection or observation identity skip around B1. Gate/up rows and down columns are normalized after initialization and each update. Head row retraction removes the actor's initial .01 row-norm factor; gain starts at1. Parameter count, depth, bias locations, branch/output normalization and head scales all differ from the saved reference.

Both use the same raw SiTU scalar product. Writing q(g)=4*tanh(g/4)*sigmoid(g), d(u)=25*tanh(u/25), a branch is D*[q(Gx)*d(Ux)]. Its input derivative is:

    DB = D*[diag(q'(Gx)*d(Ux))*G + diag(q(Gx)*d'(Ux))*U]
    q'(g) = sech(g/4)^2*sigmoid(g) + 4*tanh(g/4)*sigmoid(g)*(1-sigmoid(g))
    d'(u) = sech(u/25)^2

For the reference block, local transport is I + diag(sigmoid(r))*DB*J_RMS. This supplies an identity term even where a raw SiTU branch has a weak derivative. J_RMS(h)=I/s - h*hT/(64*s^3), with s=sqrt(mean(h^2)+epsilon). Learned bias/gates and upstream/downstream matrices still determine actual physical-direction transport; the identity term alone is not a proof of better reachability.

For sepclip, unit-normalization away from its epsilon floor has Jacobian (I-v*vT)/norm(y), so first-branch feature-radial sensitivity is projected out before the residual addition. Its full chain contains the first branch Jacobian, first unit projector, second branch derivative and unit projector, and final residual-sum unit projector. Near-zero SiTU's quadratic leading term does not imply the whole normalized actor has a vanishing Jacobian: angular sensitivity can increase inversely with branch/input norm. Historical visited-state measurements already reject a blanket near-zero/sign-blind diagnosis. Compare signed local physical-coordinate JVPs and oracle-directed reachability on actual states before selecting an anchor intervention.

Both actor distributions use alpha,beta=1+softplus(shape logits), native Beta likelihood, [-1,1] physical rescaling, and sampled training control. The reference differs additionally in rollout2048 versus v7's1024, critic clipping enabled versus disabled, and gradient clipping disabled versus enabled. These are disclosed controls, not a basic data-reuse explanation or a mechanism attribution.

The forthcoming physical reference diagnostic can establish this saved policy's speed, costs, torque and contact/gait under frozen exact normalization rows and labeled sample/mean/mode controls. It cannot provide a matched optimizer contrast, global optimum oracle, or proof that any listed architectural distinction explains the return gap.

Current-source hashes are recorded in the companion inventory and appended below; historical execution hashes remain unavailable.

| Current source | SHA256 |
| --- | --- |
| `cleanrl/ppo_continuous_action_prerms_adamh_v22.py` | `514cfd7c4b65ab891fa421ddd784c27f346502cf34e0eca0206b0d0c422478a3` |
| `cleanrl/ppo_continuous_action_sepclip_rollout_clock_anvil_v7.py` | `07f5f9636103dd5fdc9896de3eb781078df8bb1cb9b3ad87bcb79907fc619c8d` |
| `cleanrl/shared/norm_residual.py` | `ad3d7e9da1283cbcea4143d27cc4f4d8d3523fd17f954bfb821cb71a90394fee` |
| `cleanrl/shared/host_actor.py` | `679925e692d862719cf278c5e0b9d4bdd2f04dd5d8c6f0cac5d9c7d58e4114cf` |

## Independent review of new physical reference diagnostic

Reviewed source and focused contracts without execution. Strict checkpoint hash, Args architecture selection, exact sixteen-row normalizer metadata, strict actor/critic keys, physical bounds, and host-versus-compiled-CUDA parity at every visited observation form meaningful gates. Permanent action/logit buffers are copied and reset seeds are paired across sample/mean/mode. Episode boundary assumptions and descriptive step-metric uncertainty are explicit.

Two source issues were identified before execution and corrected by the physics owner in the new diagnostic only: rounded alpha or beta equal1 is a valid Beta law and must not abort sample/mean evaluation; and computing modal denominator as alpha+beta-2 loses a one-ULP nearuniform excess. The diagnostic now computes (alpha-1)+(beta-1), preserves one-sided boundary modes, and uses an explicit midpoint tie for uniform alpha=beta=1. Focused tests include nextafter(1,2) one-sided and symmetric cases. Intended-logit versus actual rounded-law mode discrepancy is separately recorded. Existing shared physics helpers and frozen trainers were not changed.

No additional blocking static defect found after those repairs. Approved for queued CUDA contract execution, not represented as a passed contract or completed physical measurement. Historical execution source remains unpinned, 50M budget remains unmatched, and this evaluation cannot identify optimizer/PG/architecture causality.

- `scripts/audit_sepclip_strong_reference.py`: `9c2f313d5bfcfb23da6e34785acab3b2ec290a38c8a59b8aeedf20e04d15e7df`
- `tests/test_sepclip_strong_reference.py`: `5a61b5e675192b014d17615a949dba59bf6ac78f8fdfd2cb24d9971f5e5baa5d`
