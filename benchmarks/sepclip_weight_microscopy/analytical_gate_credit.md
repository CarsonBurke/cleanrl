# SiTU gate credit: activation, constraint and normalization

Analytical source review and existing summary-scalar inspection only. The sole new numerical operation is ordinary scalar bisection of an analytic derivative; no model/test/GPU execution, training, submission or frozen-source modification. None of the following establishes gradient starvation or a causal benchmark bottleneck.

## The full gate is nonmonotone

Define the actual gate multiplier and capped up function:

    k(g) = 4*tanh(g/4)*sigmoid(g)
    t(u) = 25*tanh(u/25)
    feature = k(g)*t(u)

The FULL gate derivative is:

    k'(g) = sech(g/4)^2*sigmoid(g)
            + 4*tanh(g/4)*sigmoid(g)*(1-sigmoid(g))
    t'(u) = sech(u/25)^2

The reported gate-cap derivative sech(g/4)^2 is only one factor in the first term; it omits sigmoid scaling and the signed second term. A healthy cap derivative does not prove that the full gate derivative is large or positive.

Set z=exp(g/2)>0. Exact simplification gives:

    k'(g) = [4*z^2 / ((z+1)^2*(1+z^2)^2)] * (z^3+z^2+z-1)

The polynomial is strictly increasing for z>0 because its derivative3*z^2+2*z+1 is positive. It has exactly one positive root. Independently evaluating the direct derivative by scalar bisection on[-2,-.5] gives:

    g_min = -1.2187557268720126
    k(g_min) = -.2697689954724918
    k'(g_min) approximately -5.55e-17

k decreases from its limit0-negative as g moves from negative infinity toward this minimum, then increases through k(0)=0 toward its positive-infinity limit4. Thus k'(g)<0 below the minimum and k'(g)>0 above it. The derivative zero is cancellation at a nonzero activation, not sigmoid saturation or vanishing capped gate.

| g | k(g) | Full k'(g) | Gate-cap derivative |
| --- | ---: | ---: | ---: |
| -8 | -.00129315 | -.00126902 | .0706508 |
| -2 | -.220343 | -.100330 | .786448 |
| -1.21875573 | -.269769 | approximately0 | .912622 |
| -1 | -.263475 | +.0601932 | .940015 |
| 0 | 0 | .5 | 1 |
| 1 | .716200 | .879822 | .940015 |
| 8 | 3.85482 | .0719198 | .0706508 |

## A gate-row gradient has several independent factors

For one branch B(x)=D*[k(Gx)*t(Ux)], let w_i be a unit gate row, g_i=w_i dot x, u_i=U_i dot x, and lambda_B the upstream ASCENT cotangent at the raw branch output. Define q_i=D_column_i dot lambda_B. The ambient gate-row covector is q_i*k'(g_i)*t(u_i)*x. Its row-sphere tangent projection is:

    g_gate_i_tangent = q_i*k'(g_i)*t(u_i)*[x-w_i*(w_i dot x)]

Away from numerical normalization epsilon regimes this is the exact local constrained covector for that fixed sample. Its squared norm is [q_i*k'(g_i)*t(u_i)]^2*[norm(x)^2-g_i^2]. A batch sums these vectors before taking its norm, so positive/negative credit and state directions can cancel.

Different mechanisms can make it small or zero:

- Full activation derivative cancellation k'(g_i)=0 near the gate minimum.
- Up value t(u_i)=0 at u_i=0, independent of the gate derivative.
- Geometric row constraint x parallel w_i, making the tangent input factor zero.
- Downstream cotangent q_i=0, including policy-score cancellation, down-column alignment, and feature-normalization projections.
- Opposing sample contributions, even when every individual factor is nonzero.

None follows merely from a small averaged matrix norm. Anvil or Adam history/preconditioning may subsequently convert ambient normal forces into tangent motion, but that does not make the raw constrained covector nonzero or establish useful physical credit.

The gate-zero derivative does not automatically silence its whole channel. An up row has covector q_i*k(g_i)*t'(u_i)*x, and a down column has covector lambda_B*[k(g_i)*t(u_i)], each with its own sphere projection when constrained. At g_min, k is nonzero and these other routes may remain active. At g=0, k=0 while k'=.5, so the gate route can remain active if u!=0. At u=0, gate/down routes vanish locally but the up route can remain active if k(g)!=0. At both g=u=0 all first derivatives of that channel's product vanish; other channels and actor parameters need not. These are local algebraic rescue routes, not measured compensation in the trained policy.

## Normalization and the second-branch bound

The second branch receives x2=8*h where h=unit(B1(x)). For unit gate/up rows, Cauchy-Schwarz gives |g_i|<=8 and |u_i|<=8 in ideal arithmetic. Consequently t'(u_i)>=sech(.32)^2=.90420547 and |t(u_i)|<=7.73767303. Its up cap cannot enter arbitrarily deep saturation under this input/row constraint. The full gate still includes its interior derivative zero and negative side; the bound does not guarantee healthy gate credit.

The first branch consumes normalized observations directly, so its gate/up bounds are norm(x), not8. Observation coordinate clipping is yet another constraint and does not imply that the first branch has a fixed8 input norm. FP32 normalization/retraction introduces rounding in both bounds; the statements above describe the mathematical constraint rather than an exact floating-point inequality.

For raw output b!=0, unit(b) has Jacobian [I-h*h^T]/norm(b). Therefore lambda_B after this normalization is the corresponding tangent-projected downstream cotangent divided by norm(b). A raw derivative parallel to b is removed even if k' and t' are nonzero. This feature-radial removal differs from the parameter-row projection x-w*(w dot x), and from the activation stationary point. Small normalization denominators can also amplify surviving angular derivatives rather than uniformly suppress them.

The second branch cannot see a change that only rescales the first raw feature while leaving h fixed. This is a specific normalization-direction nullspace, not global velocity blindness. For the first actor branch, an input-radial perturbation x changes h according to [I-h*h^T]*J_B1*x/norm(B1). It vanishes for an exactly homogeneous raw branch but not generally for SiTU at finite gates. A physical velocity perturbation changes selected qvel/observation coordinates; it is not simultaneous radial scaling of all normalized positions and velocities. Its useful policy and closed-loop effects require that actual directional chain.

## What the stored visited summaries measure

The historical representation audit stores marginal gate/up preactivation summaries, sigmoid summaries, gate/up CAP derivative summaries, odd/even product energies, feature/input Jacobian spectra, selected radial JVPs and normalization projection diagnostics. It does not store the full k'(g) distribution or joint per-channel credit products q*k'(g)*t(u)*tangent_input.

Each gate summary below pools11008 entries (256 visited observations times43 channels). The narrower Jacobian/radial diagnostics use32 observations.

| 8M policy / branch | Gate RMS | Up RMS | Mean gate-cap derivative | Mean up-cap derivative | Odd/even product energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| v7 half / first | 1.13398 | 1.04490 | .928880 | .998258 | .466828 |
| v7 half / second | 1.17504 | 1.27797 | .925023 | .997400 | .484309 |
| v7 full / first | 1.53889 | 1.63180 | .906259 | .995861 | .872508 |
| v7 full / second | 1.38788 | 1.45117 | .899780 | .996651 | .551896 |

These values reject a blanket claim of deeply saturated caps or infinitesimal even-only features. They do NOT resolve the gate-product derivative. For full8M first gates, p05=-2.07193 and p25=-.817902 bracket g_min; for its second gates, p05=-2.16618 and p25=-1.08377 also bracket it. Quantiles show that the stationary threshold lies within an occupied negative-gate range, but do not establish concentration near the root, an exact negative-derivative fraction, weighted credit loss or starvation. Full distributions could differ substantially while matching these summaries.

The half/full8M final-unit input-radial JVP norms have means .371191/.525349; first-normalization removed feature-Jacobian energy fractions have means .271424/.239986. Radial response is therefore not universally zero on those selected inputs. These aggregate measurements do not identify physical-velocity response or attribute any loss to the stationary gate versus normalization/constraint factors.

Separate cap and sigmoid means cannot reconstruct the expectation of their nonlinear signed derivative products. Without raw paired g/u values and downstream cotangents, neither the full k' distribution nor its useful-credit weighting is recoverable from this JSON. Approximate inversion of pooled quantiles or multiplying summary means would manufacture unmeasured evidence.

## Precise hypotheses, still unselected

A specific activation hypothesis would require many visited, physically useful channel credits to lie near g_min and therefore have small k' while up/down routes and other channels fail to recover the desired response. A constraint hypothesis instead concerns tangent_input or down-column projections. A normalization hypothesis concerns raw feature-radial alignment. These are different explanations with different witnesses; an activation offset changes more than one factor and cannot be selected from cap summaries alone.

Current evidence supports inspecting the full chain, not alleging starvation, choosing a gate offset, or explaining sub10k-versus20k by this scalar nonmonotonicity. Physical direction reliability, finite updates and independent causal controls remain necessary.

Source/artifact hashes at inspection:

- `cleanrl/shared/host_actor.py`: `679925e692d862719cf278c5e0b9d4bdd2f04dd5d8c6f0cac5d9c7d58e4114cf`
- `scripts/audit_sepclip_representation.py`: `377c39e256b25b00c705714910315cf18d23c0db8e2e61abc4d3ddda24467fb7`
- `representation_historical.json`: `df18acf6ec60be52f3ab8a34488eae7d3330a4cc9c1014950ae4ca42546c75e2`
