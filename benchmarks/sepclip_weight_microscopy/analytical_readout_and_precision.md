# Readout coordinates, precision, and physical credit

This is source-derived algebra plus explicitly identified historical scalar measurements. No model execution or new causal result is claimed. It supplements the chart, optimizer-credit, and physical-feedback committee notes.

## A unit head row with a learned gain is locally a free linear readout

For one output, the frozen sepclip head is z = rho*(w·h+b), with ||w||=1 and h=8*q. Define effective linear weight v=rho*w and effective bias c=rho*b. For every nonzero v, choose rho=||v||, w=v/||v||, and b=c/rho. This represents an arbitrary ordinary linear readout row, including its bias. Thus unit head rows alone do not establish a fixed readout-capacity ceiling against the pre-RMS reference's unconstrained linear head.

The differential is delta_v = w*delta_rho + rho*delta_w, delta_c = b*delta_rho + rho*delta_b. With delta_w tangent to w and rho nonzero, every effective (delta_v,delta_c) is attainable:

    delta_rho = w·delta_v
    delta_w = (delta_v-w*(w·delta_v))/rho
    delta_b = (delta_c-b*delta_rho)/rho

The parametrization has a singularity at rho=0. In particular, a zero effective weight with a nonzero constant bias is not represented by finite rho,b; it is a limit of this chart. This qualification does not support a generic capacity bottleneck at the measured positive gains.

For an ascent logit covector t(s), gradients are:

    g_w = rho*E[t*h]
    g_b = rho*E[t]
    g_rho = E[t*(w·h+b)]
    rho*g_rho = w·g_w + b*g_b

The last identity shows that radial head credit is related to the gain and bias covectors, rather than automatically lost. An ambient radial head displacement delta_w=r*w can be reproduced to first order by delta_rho=r*rho and delta_b=-r*b. Exact finite compensation for w→(1+r)*w is rho→(1+r)*rho and b→b/(1+r), where1+r is nonzero. The actual independent Adam updates need not perform that coordinated compensation: their moments and coordinatewise denominators differ. Head weights, gain, and bias share one actor clipping scalar; multiplying their gradients by that same scalar preserves the gradient identity. A changed chart can change the total actor norm and therefore this shared factor. This is an optimization-conditioning question, not a generic missing linear function class.

The source's initial head projection cancels its .01 scale while gain starts at one. Restoring gain=.01 changes the initial policy and this chart's conditioning. Upstream/head-weight gradients scale with rho for a fixed logit covector, while g_rho does not. The Beta score covector also changes, so a universal hundredfold gradient claim would be wrong.

Historical representation_historical.json reports v7-half gains at4M in[2.0213,2.9501] and at8M in[2.4259,3.6773]. These are nonsingular readout coordinates. They do not certify good optimizer conditioning or future gain growth. The stronger reference's feature stream differs, so the fixed-feature equivalence does not equate the two complete architectures.

## Gate and up-row constraints are different from the head chart

For a bias-free SiTU channel, f(g,u)=4*tanh(g/4)*sigmoid(g)*25*tanh(u/25). Scaling a gate row changes both sigmoid and tanh across states; a single output gain generally cannot undo that change. Uniform down scaling is a gauge of a normalized branch, but individual column scaling generally changes relative channel contributions. Uniform up scaling is only approximately linear when |u/25| is small. Therefore a head gain's ability to encode arbitrary readout magnitudes does not restore every radial degree of freedom removed inside the actor.

The body distinction is about a state-dependent feature map and its Jacobian, not simply the presence of normalized matrices. A normal force at a constrained stationary point still need not be useful tangent credit. root_analytical_transport.md's exact polar counterexample shows how extrinsic preconditioning can convert such a force to tangent motion. Neither that example nor radial norm fractions establish a primary performance cause.

## Exact immediate torque-cost gradients are small and do not include gait

Let physical mean m=(alpha-beta)/kappa and kappa=alpha+beta. For the native Beta action mapped to[-1,1]:

    variance(a) = (1-m*m)/(kappa+1)
    E[a*a] = m*m + (1-m*m)/(kappa+1)
    E[control_cost] = .1*sum_j E[a_j*a_j]
    d E[cost]/d m_j = .2*m_j*kappa_j/(kappa_j+1)
    d E[cost]/d log(kappa_j) = -.1*kappa_j*(1-m_j*m_j)/(kappa_j+1)^2

These are exact fixed-state identities for the ideal, unclipped Beta law; sampler uniform truncation and action clipping are excluded. They exclude changed future states, contact timing, occupancy, and velocity. If all six concentrations are at least16, the direct sampled-versus-mean tax is at most .6/17 per step, about35.3 points over1000 steps if states and means were held fixed. This is a conditional bound; historical concentrations do not all satisfy that lower bound, and a changed deterministic controller changes its trajectory. The measured v7-half sample→mean decomposition attributed about707 points to velocity and18.4 to control cost, supporting that broader feedback distinction without using this conditional bound as a measured return theorem.

## Candidate excess precision also moves the mean

For v13, tau=kappa-2, q=sigmoid(r), ell=log(tau), and m=tau*(2*q-1)/kappa. Its differentials are:

    d m/d r = 2*(1-2/kappa)*q*(1-q)
    d m/d ell = 2*m/kappa
    d log(kappa)/d ell = 1-2/kappa

Therefore for covectors in physical(mean,log-total-concentration) coordinates:

    g_r = [2*(1-2/kappa)*q*(1-q)]*g_m
    g_ell = (2*m/kappa)*g_m + (1-2/kappa)*g_logkappa

For illustration, kappa=20,m=.8 gives q=17/18, d m/d ell=.08 and d log(kappa)/d ell=.9. A positive excess-precision covector can therefore come from increasing the mean torque, even if the pure precision covector is zero or negative. It is not evidence of a noise-removal channel by itself. The full Fisher must transform by the same Jacobian, with its cross term retained. These are hand-algebra values, not measured policy statistics.

The chart committee separately distinguishes exact-shape interior coordinates from the candidate's pre-addition softplus-excess arithmetic, which can differ at FP32 shape-one boundaries. No chart is selected here.

## What the existing layer numbers permit

At v7-half4M, layer_historical.json's stored-advantage subset pairing is dominated by body components; its head bias and gain components are slightly negative while the head weight component is positive. The jointly deployed direction has positive total subset pairing and reinforcing Fisher cross terms. At8M the head-weight subset pairing is slightly negative while the six body components are positive. These signs concern the stored surrogate on the selected subset. They do not establish physical credit quality, global minibatch descent, a starving head, or a causal return loss.

The useful investigation separates: physical credit covector; masked stored PPO covector; representation pullback; constrained gradient; optimizer-history transformation; actual retracted chord; joint policy motion; and closed-loop physical feedback. None of the coordinate identities replaces the independent physical-PG and finite-policy checks.

Source/artifact hashes for this derivation:

```json
{
  "cleanrl/ppo_continuous_action_sepclip_rollout_clock_anvil_v7.py": "07f5f9636103dd5fdc9896de3eb781078df8bb1cb9b3ad87bcb79907fc619c8d",
  "cleanrl/ppo_continuous_action_prerms_adamh_v22.py": "514cfd7c4b65ab891fa421ddd784c27f346502cf34e0eca0206b0d0c422478a3",
  "benchmarks/sepclip_weight_microscopy/representation_historical.json": "df18acf6ec60be52f3ab8a34488eae7d3330a4cc9c1014950ae4ca42546c75e2",
  "benchmarks/sepclip_weight_microscopy/layer_historical.json": "ba19fa734771e4eb15378403876f0599caf5b081bc95cb368a9218014e387d80"
}
```
