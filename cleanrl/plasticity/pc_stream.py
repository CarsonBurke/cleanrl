"""Predictive coding proper as per-perceptron, per-sample, state-conditional plasticity.

# WHAT THIS FILE TESTS

The former lead in this family (`novelty_stream.py --method precision`) is an
ANISOTROPIC-PRECISION rule: each perceptron accumulates the second moment of the
input directions it has already been modified along and damps its step through
`(I + lambda C)^-1`. It is a *discriminative* statement -- shrink where you are
committed -- and it is a full `D x D` object per unit. Its headline died: it is a
learning-rate artifact, and nothing in this family may cite it as support.

Predictive coding says something stronger and more local: each unit carries a
GENERATIVE MODEL of its own input, and learns only from the PRECISION-WEIGHTED
PREDICTION ERROR -- the part of the current sample its own model did not
anticipate, scaled by its confidence in that state.

    x_hat_i = U_i U_i^T x_i        (unit i's reconstruction of its receptive field)
    eps_i   = x_i - x_hat_i        (what unit i did not anticipate)
    pi_i    = e_i / (e_i + s_i)    (Wiener confidence: is this real novelty or my
                                    usual noise? e_i = |eps_i|^2/D, s_i the unit's
                                    running mean of e_i)
    dW_i    = -lr * delta_i * (pi_i * eps_i)^T

`U_i` is learned online by Oja's subspace rule driven by the unit's OWN prediction
error, weighted by how much the unit actually moved on that sample. The model is a
compressed, error-driven description of the receptive field the unit has committed
to, not an accumulated second moment.

Two ways this is not the old lead:
  1. The operator is rank-`R` and LEARNED by minimising its own prediction error,
     not the exact ridge inverse of an accumulated Gram matrix. (Note: the ridge
     reconstruction `x_hat = C(C + aI)^-1 x` would make `eps = a(C+aI)^-1 x`,
     i.e. *exactly* the old lead. The entire content of "predictive coding
     proper" is that the model is low-rank and error-driven instead.)
  2. It REMOVES the anticipated component rather than shrinking it, so the novel
     component keeps full step size. The old lead shrinks everything, unevenly.

Everything is per-sample, so the rule is BATCH AGNOSTIC: `--batch 1` streaming is
the intended operating point and is exercised by the default sweep. There are no
EMAs; the two running references are cumulative means, i.e. sufficient statistics
of the stream with no forgetting horizon to tune.

TRAPS THIS FILE AVOIDS, all of which have already cost runs in this family:
  - the plasticity NEVER touches the outgoing weight `w2`. `w2` starts at zero, so
    a level read off its gradient pins both at zero forever and the run scores
    exactly 1.0000 with nothing learned.
  - there is NO mean-one budget conservation. Forcing levels to mean one makes a
    rule exactly blind to a global shift.
  - the LR grid BRACKETS every arm's optimum. A mechanism that damps the step is
    running at a lower effective learning rate, so a grid whose floor sits above
    the control's optimum pays the mechanism for descending toward an optimum the
    control was never allowed to reach. Measured here: on the default task Adam's
    optimum is 1e-4, below the 3e-4 floor of the grid this family had been using,
    and at the truncated grid every arm below "wins" while at the correct grid
    every arm loses. The report warns when a winner sits at a grid edge.
  - seeds resample the ENTIRE task -- teachers, supports, stream and init. Seeding
    only the init understates the across-seed spread by ~4x and manufactures
    significance.

# ARMS

  adam            plain Adam, swept over the same LR grid
  pc              the rule above
  pc_scalar       identical, but `eps` collapsed to its scalar magnitude along
                  `x` -- realized plasticity matched EXACTLY, direction destroyed
  pc_frozen       identical, but `U` frozen at init (which is ~0), so the rule
                  degenerates to the confidence rescale alone
  pc_random       identical, but `U` frozen at a random ORTHONORMAL rank-R
                  subspace: removes exactly as many input dimensions as `pc`, but
                  the wrong ones. Isolates "the right subspace" from "any
                  subspace of the same size".
  pc_shared       identical to `pc`, but every unit in the layer shares ONE model
  pc_shuffle      identical to `pc`, and the models are learned identically, but
                  at APPLICATION time the per-unit operators are permuted across
                  units. Mean and cross-unit dispersion of the operators are
                  preserved exactly; only the state->unit correspondence is
                  destroyed. If the arm ties this control, state-dependence
                  contributed zero whatever the headline says.
  precision       the old lead, replicated verbatim from `novelty_stream.py`
  precision_diag  its diagonal shadow (provably cannot represent a direction)

`precision`/`precision_diag` are re-implemented here rather than cited so the
head-to-head runs on identical tasks, identical inits and identical seeds.

# TASK

`regions` well-separated input supports presented in sequence, each with its OWN
target function. A single shared teacher creates no conflict between contexts and
cannot test anything -- it is kept only as the `--no-per-region-teacher`
INERTNESS control. `--shuffle` removes the sequential shift entirely.

Two support geometries, because the first one has a measured defect:

  `--region-mode centre`    x ~ N(c_r, I), |c_r| = shift. The original harness.
      Measured here: median |cos| between different units' learned model
      directions is 0.989, because a shift-12 centre dominates every unit's
      input. Per-unit content is structurally absent.
  `--region-mode subspace`  x ~ scale * A_r z + N(0, I), `A_r` a random
      orthonormal D x k basis, zero mean. Regions are near-orthogonal subspaces
      rather than distant points.

  `--field-fraction f`  each perceptron's incoming row is masked to a random
      fraction `f` of the inputs. THIS is the property that makes per-unit
      content possible at all: in a fully-connected layer every unit's per-sample
      gradient is `delta_h * x`, i.e. all units' gradients are PARALLEL, so there
      is no per-unit direction to find on the current sample. Masking the rows
      gives units genuinely different input geometry, and the measured autonomy
      goes 0.989 -> 0.182. `f = 1.0` is the original fully-connected task.

Three numbers, each normalised by the zero-predictor on that region:
  acquisition -- error on the region trained on last
  retention   -- error on every earlier region
  overall     -- mean over all regions

Plus `plast`, the realized mean plasticity (the linear form `<x, A_i x>/<x, x>`
of whatever operator the arm applies, so no gain can be attributed to a global
step-size change), and `auton`, the median |cos| between different units' learned
model directions.

Seeds are PAIRED: seed `s` gets a bit-identical task and bit-identical initial
weights in every arm, so every arm-vs-arm comparison is a paired test, and the
report states the seed count each observed effect actually needs.

# RESULTS

Headline, 128 paired seeds, batch 1 streaming, defaults except the LR grid:

  .venv/bin/python cleanrl/plasticity/pc_stream.py --seeds 128 \
      --lr-grid 1e-5 3e-5 1e-4 3e-4 1e-3 \
      --method adam,pc,pc_scalar,pc_frozen,pc_random,pc_shared,pc_shuffle

  arm          lr     acquire  retain  overall  plast  auton   vs adam (paired)
  adam         1e-4   0.4962   0.6780  0.6553   1.000  -
  pc           3e-5   0.5887   0.6424  0.6357   0.549  0.134  -0.0196 +-0.0067  t -2.94
  pc_scalar    3e-5   0.5909   0.6602  0.6516   0.379  0.134  -0.0037 +-0.0063  t -0.59
  pc_frozen    1e-4   0.4962   0.6780  0.6553   1.000  0.071  +0.0000 +-0.0000  t +1.10
  pc_random    1e-4   0.5016   0.6781  0.6560   0.804  0.072  +0.0007 +-0.0011  t +0.62
  pc_shared    1e-4   0.4957   0.6784  0.6556   0.995  1.000  +0.0003 +-0.0002  t +1.40
  pc_shuffle   1e-4   0.4967   0.6774  0.6549   0.891  0.135  -0.0005 +-0.0001  t -3.12

Every arm's best LR is interior to the grid. `pc` beats adam AND all four
controls: vs `pc_scalar` +0.0159 +-0.0026 (t +6.07), vs `pc_frozen` +0.0196
+-0.0067 (t +2.94), vs `pc_random` +0.0203 +-0.0066 (t +3.09), vs `pc_shared`
+0.0199 +-0.0067 (t +2.98), vs `pc_shuffle` +0.0192 +-0.0066 (t +2.89). The
whole gain is retention (-0.0356) bought with acquisition (+0.0925).

`pc_scalar` carries LOWER realized plasticity than `pc` (0.379 vs 0.549) and is
still worse, so the gain is not a step-size effect in either direction.

What each control settles:
  - `pc_shuffle` ties adam (-0.0005) while carrying operators with the same mean
    and the same cross-unit dispersion as `pc`. Only their attachment to the
    RIGHT units pays; the distribution of operators is worth nothing.
  - `pc_scalar` ties adam, so the DIRECTION of the prediction error is the whole
    mechanism and its magnitude is worth nothing.
  - `pc_random` ties adam, so removing eight input dimensions is worth nothing
    unless they are the eight the unit has actually committed to.
  - `pc_shared` ties adam (+0.0003) with a genuine single layer-wide model
    (autonomy exactly 1.000, |u| 0.116). Its model grows far more slowly than the
    per-unit ones because averaging the Oja update over units with different
    receptive fields cancels the field-specific components -- a single model
    cannot serve heterogeneous receptive fields. This is the control that
    beat the per-unit version for the old anisotropic-precision rule; here it
    does not.

Where it does NOT work, all measured, none of it hidden:
  - `--field-fraction 1.0` (fully connected): pc -0.0096 vs adam but `pc_shuffle`
    -0.0097, and pc vs pc_shuffle is -0.0000 (t -0.00). Autonomy 0.712. The
    state->unit correspondence contributes EXACTLY ZERO, because every unit's
    per-sample gradient is delta_h * x and therefore parallel. Any apparent gain
    there is the level.
  - `--batch 8`: pc -0.0139 (t -2.55) at 32 seeds -- real but weaker, and the
    model rate needs retuning. The rule is strongest exactly where the family's
    requirement points, at batch 1.
  - `--pc-precision`: pc +0.0045 (t +0.58), i.e. null. The confidence factor is
    a disguised learning-rate change.
  - `--no-per-region-teacher` (no conflict): pc -0.0016 (t -0.86) and ties its
    own shuffle (+0.0011, t +0.58). INERT, as it should be, not harmful.
  - `--shuffle` (no sequential shift): pc -0.0097 (t -6.44), beating pc_shuffle
    by +0.0091 (t +6.31). So the gain does NOT require the sequential shift; it
    is a per-unit conditioning gain on a mixture, not a forgetting-specific one.

Against the anisotropic-precision form, same stream, same inits, same 64 seeds,
batch 1: adam 0.6443, pc 0.6250 (-0.0193, t -2.34), precision 0.6918 (+0.0475,
t +5.25), precision_diag 0.6898 (+0.0455, t +5.25). Predictive coding proper is
stronger by 0.0668 overall, and the anisotropic-precision form is decisively
WORSE than plain Adam. Its realized plasticity collapses to 0.038 at batch 1: the
accumulated Gram matrix keeps shrinking the step until almost nothing is left.

Two protocol facts that this file exists to enforce, both of which reversed
earlier conclusions in this family:
  - the LR grid must BRACKET every arm's optimum. On the fully-connected task at
    the grid this family had been using (floor 3e-4), pc/precision/everything
    "beat" adam; adam's true optimum is 1e-4, and at a bracketing grid every one
    of those arms LOSES. The report warns when a winner sits at a grid edge.
  - seeds must resample the whole task. Under init-only seeding the same batch-8
    configuration gave pc -0.0179 at t -6.37; resampling the task leaves the
    effect at -0.0139 but drops it to t -2.55, because the across-seed spread is
    ~6x larger than init-only seeding suggests.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
import tyro

METHODS = ("adam", "pc", "pc_scalar", "pc_frozen", "pc_random", "pc_shared",
           "pc_shuffle", "precision", "precision_diag")
PC_METHODS = ("pc", "pc_scalar", "pc_frozen", "pc_random", "pc_shared", "pc_shuffle")
TEACHER_WIDTH = 8


@dataclass
class Args:
    input_dim: int = 64
    hidden: int = 64
    regions: int = 8
    """number of sequentially presented input supports"""
    per_region_teacher: bool = True
    """Each region gets its OWN target function. With one shared teacher every
    region wants the same mapping, so no unit ever faces conflicting demands and
    a state-conditional rule has nothing to buy. `--no-per-region-teacher` is the
    INERTNESS control, not a test."""
    region_mode: str = "centre"
    """`centre` (distant Gaussian clusters, the original harness) or `subspace`
    (zero-mean near-orthogonal subspaces)"""
    region_rank: int = 4
    """dimension of each region's subspace under `--region-mode subspace`"""
    shift: float = 12.0
    """distance of each cluster centre from the origin, or the RMS radius of the
    region subspace under `--region-mode subspace`"""
    field_fraction: float = 0.25
    """Fraction of inputs each perceptron is wired to. 1.0 is the original
    fully-connected task, on which every unit's per-sample gradient is parallel
    to the same input vector and per-unit content cannot exist."""
    samples_per_region: int = 2000
    noise_std: float = 0.5
    batch: int = 1
    """Batch agnosticism is a hard requirement, so the default is streaming."""
    method: str = "pc_family"
    """`pc_family` (default) is adam plus the PC arm and its four controls, and
    is what the acceptance sweep runs. `all` adds the two `precision` arms, which
    carry a dense D x D geometry per perceptron and cost ~20x the rest of the
    sweep combined at batch 1 -- run those explicitly with `--method
    adam,pc,precision,precision_diag` rather than paying for them every time."""
    lr_grid: tuple[float, ...] = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    """Must BRACKET every arm's optimum. Damping lowers the EFFECTIVE learning
    rate, so a grid whose floor sits above the control's optimum pays the
    mechanism for an LR correction the control was never allowed to make."""
    shuffle: bool = False
    """draw every sample from a random region: the no-shift reference"""

    rank: int = 8
    """rank of each perceptron's generative model of its own input"""
    pc_lr: float = 3e-3
    """Oja rate for that model, applied to the input-energy-normalised update.
    Swept at batch 1: 1e-3 / 3e-3 / 1e-2 give pc-vs-adam -0.0078 / -0.0135 /
    +0.0045, i.e. at 1e-2 the model over-commits (realized plasticity 0.335) and
    the arm goes null. The rate is per UPDATE, so it must be retuned when the
    batch size changes."""
    pc_init: float = 0.02
    """scale of `U` at init, in units of 1/sqrt(D). Small enough that plasticity
    starts at ~1 and the run begins bit-close to Adam."""
    pc_precision: bool = False
    """The Wiener confidence factor. MEASURED TO BE HARMFUL and off by default:
    it contributes a near-constant ~0.49 level (its state-dependence is weak) and
    a level is fungible with the learning rate, so it just moves the arm down its
    own LR curve while costing acquisition. With it on, pc ties adam; with it off,
    pc wins. `--pc-precision` reproduces the falsification. It also fails the
    family's requirement that an uncertainty be a fraction in [0,1] mapping pure
    noise to 0: under pure noise this one sits at 0.5, not 0."""
    pc_commit_cap: float = 4.0
    """ceiling on the commitment weight of one sample's model update"""
    apply: str = "both"
    """where the operator acts: `grad` (per-sample, pre-Adam -- the literal PC
    rule), `step` (post-Adam, the insertion point the old lead uses), or `both`"""

    precision_lambda: float = 1.0
    """the old lead's weight on a perceptron's accumulated input geometry"""
    precision_decay: float = 0.0

    eval_samples: int = 1024
    seeds: int = 32
    """Paired seeds, each resampling the WHOLE task. The report states the
    minimum detectable effect and the seed count each observed effect needs."""
    seed: int = 1
    device: str = "cuda"


def make_teachers(args, device, gen):
    """One target function per (seed, region). Seeds resample the task."""
    shape = (args.seeds, args.regions, args.input_dim, TEACHER_WIDTH)
    first = torch.randn(shape, device=device, generator=gen) / np.sqrt(args.input_dim)
    second = torch.randn(shape[:2] + (TEACHER_WIDTH,), device=device,
                         generator=gen) * 0.7
    if not args.per_region_teacher:
        first = first[:, :1].expand_as(first).contiguous()
        second = second[:, :1].expand_as(second).contiguous()
    return first, second


def teacher_forward(x, first, second):
    """x (K,B,D), first (K,D,M), second (K,M) -> (K,B)."""
    return torch.einsum("kbm,km->kb", torch.tanh(torch.einsum("kbd,kdm->kbm", x,
                                                              first)), second)


def make_supports(args, device, gen):
    """Where each region lives in input space, per seed."""
    if args.region_mode == "centre":
        centre = torch.randn((args.seeds, args.regions, args.input_dim),
                             device=device, generator=gen)
        return centre / centre.norm(dim=-1, keepdim=True) * args.shift
    if args.region_mode != "subspace":
        raise ValueError("region_mode must be centre or subspace")
    basis = torch.randn((args.seeds, args.regions, args.input_dim, args.region_rank),
                        device=device, generator=gen)
    flat, _ = torch.linalg.qr(basis.reshape(-1, args.input_dim, args.region_rank))
    return flat.reshape(basis.shape).contiguous()


def draw_inputs(support, noise, args):
    """support (K,D) or (K,D,k), noise (K,B,D) -> (K,B,D)."""
    if args.region_mode == "centre":
        return support.unsqueeze(1) + noise
    latent = noise[..., :args.region_rank]
    scale = args.shift / np.sqrt(args.region_rank)
    return scale * torch.einsum("kbj,kdj->kbd", latent, support) + noise


def evaluate(w1, w2, first, second, support, args, device, gen, order):
    """Per-region error, normalised by that config's own zero-predictor."""
    n_cfg, hidden = w1.shape[0], w1.shape[1]
    scores = []
    for region in range(args.regions):
        noise = torch.randn((args.seeds, args.eval_samples, args.input_dim),
                            device=device, generator=gen)[order]
        x = draw_inputs(support[:, region], noise, args)
        clean = teacher_forward(x, first[:, region], second[:, region])
        prediction = (torch.tanh(torch.einsum("khd,kbd->kbh", w1, x))
                      * w2.reshape(n_cfg, 1, hidden)).sum(-1)
        scores.append((prediction - clean).square().mean(1)
                      / clean.square().mean(1).clamp_min(1e-12))
    return torch.stack(scores)                                    # (R, K)


def make_generative_model(n_cfg, hidden, dim, rank, scale, device, gen,
                          orthonormal=False):
    """Each perceptron's model of its own receptive field."""
    u = torch.randn((n_cfg, hidden, dim, rank), device=device, generator=gen)
    if orthonormal:
        q, _ = torch.linalg.qr(u.reshape(-1, dim, rank))
        return q.reshape(n_cfg, hidden, dim, rank).contiguous()
    return u.mul_(scale / np.sqrt(dim))


def model_autonomy(u):
    """Do different perceptrons commit to DIFFERENT directions?

    1.0 = every unit models the same direction, so there is no per-unit content
    whatever the arm's headline says; 0 = mutually orthogonal. On the
    fully-connected task this measures 0.989, which is the structural reason a
    per-perceptron rule cannot differ from a per-layer one there.
    """
    n_cfg, hidden, dim, _ = u.shape
    if hidden < 2:
        return np.ones(n_cfg)
    top = torch.linalg.svd(u.reshape(-1, dim, u.shape[-1]),
                           full_matrices=False)[0][..., 0]
    top = (top / top.norm(dim=-1, keepdim=True).clamp_min(1e-12)
           ).reshape(n_cfg, hidden, dim)
    cosine = torch.einsum("khd,kgd->khg", top, top).abs()
    off = ~torch.eye(hidden, dtype=torch.bool, device=u.device)
    return cosine[:, off].median(dim=-1).values.cpu().numpy()


def run_group(args, method, lrs, device, teachers, supports):
    """One method, every (lr, seed) as a batch dimension.

    Seed `s` has the same task and the same initial weights in every arm, so all
    arm-vs-arm comparisons are PAIRED, while the task itself is resampled across
    seeds so the spread is the real one.
    """
    dim, hidden, rank = args.input_dim, args.hidden, args.rank
    configs = [{"lr": v, "seed": s} for v in lrs for s in range(args.seeds)]
    n_cfg, batch = len(configs), args.batch
    order = torch.tensor([c["seed"] for c in configs], device=device)
    rows_index = torch.arange(n_cfg, device=device)

    first = teachers[0][order]                       # (K, R, D, M)
    second = teachers[1][order]                      # (K, R, M)
    support = supports[order]                        # (K, R, D[, k])

    init_gen = torch.Generator(device=device).manual_seed(args.seed + 991)
    base = torch.randn((args.seeds, hidden, dim), device=device,
                       generator=init_gen) / np.sqrt(dim)
    if args.field_fraction < 1.0:
        keep = (torch.rand((args.seeds, hidden, dim), device=device,
                           generator=init_gen) < args.field_fraction).float()
        base, field = base * keep, keep[order]
    else:
        field = None
    w1 = base[order].contiguous()
    w2 = torch.zeros((n_cfg, 1, hidden), device=device)
    m1, v1 = torch.zeros_like(w1), torch.zeros_like(w1)
    m2, v2 = torch.zeros_like(w2), torch.zeros_like(w2)
    lr = torch.tensor([c["lr"] for c in configs], device=device).view(n_cfg, 1, 1)

    data_gen = torch.Generator(device=device).manual_seed(args.seed)

    is_pc = method in PC_METHODS
    learns_model = method in ("pc", "pc_scalar", "pc_shared", "pc_shuffle")
    scalarised = method == "pc_scalar"
    shared_model = method == "pc_shared"
    permuted = method == "pc_shuffle"
    if is_pc:
        model_gen = torch.Generator(device=device).manual_seed(args.seed + 4242)
        u = make_generative_model(args.seeds, hidden, dim, rank, args.pc_init,
                                  device, model_gen,
                                  orthonormal=(method == "pc_random"))[order]
        if shared_model:
            # ONE model for the whole layer. It must also START identical across
            # units: averaging Oja updates over units whose models begin at
            # different random directions cancels to ~0 (64 near-orthogonal
            # directions), which silently pins the arm at exactly Adam and makes
            # it a vacuous control rather than a per-layer rule.
            u = u[:, :1].expand(-1, hidden, -1, -1)
        u = u.contiguous()
        if field is not None and not shared_model:
            u = u * field.unsqueeze(-1)
        # cumulative sufficient statistics, NOT EMAs: no forgetting horizon
        floor_sum = torch.zeros((n_cfg, hidden), device=device)
        commit_sum = torch.zeros((n_cfg, hidden), device=device)
        perm_gen = torch.Generator(device=device).manual_seed(args.seed + 77)

    is_precision = method in ("precision", "precision_diag")
    if is_precision:
        geometry = torch.eye(dim, device=device).expand(n_cfg, hidden, dim, dim).clone()

    grad_ratio_sum = torch.zeros((n_cfg,), device=device)
    step_ratio_sum = torch.zeros((n_cfg,), device=device)
    pi_sum = torch.zeros((n_cfg,), device=device)
    unexplained_sum = torch.zeros((n_cfg,), device=device)

    on_grad = args.apply in ("grad", "both")
    on_step = args.apply in ("step", "both")
    steps_per_region = max(args.samples_per_region // batch, 1)
    updates = 0
    for region in range(args.regions):
        for _ in range(steps_per_region):
            if args.shuffle:
                choice = torch.randint(0, args.regions, (args.seeds,),
                                       device=device, generator=data_gen)[order]
            else:
                choice = torch.full((n_cfg,), region, device=device,
                                    dtype=torch.long)
            noise = torch.randn((args.seeds, batch, dim), device=device,
                                generator=data_gen)[order]
            x = draw_inputs(support[rows_index, choice], noise, args)   # (K,B,D)
            clean = teacher_forward(x, first[rows_index, choice],
                                    second[rows_index, choice])
            y = clean + args.noise_std * torch.randn(
                (args.seeds, batch), device=device, generator=data_gen)[order]

            pre = torch.einsum("khd,kbd->kbh", w1, x)             # (K, B, H)
            act = torch.tanh(pre)
            rows = w2.reshape(n_cfg, 1, hidden)
            residual = (act * rows).sum(-1) - y                   # (K, B)
            delta = residual.unsqueeze(2) * rows * (1.0 - act.square())   # (K,B,H)

            g2 = torch.einsum("kb,kbh->kh", residual, act).view(n_cfg, 1, hidden) / batch
            g1 = torch.einsum("kbh,kbd->khd", delta, x) / batch
            updates += 1

            if is_pc:
                dh = delta.permute(0, 2, 1)                       # (K,H,B)
                # what this perceptron is actually wired to
                xu = (x.unsqueeze(1) if field is None
                      else field.unsqueeze(2) * x.unsqueeze(1))   # (K,H,B,D)
                xu_energy = (xu * xu).sum(-1).clamp_min(1e-30)    # (K,H,B)
                if permuted:
                    # destroy ONLY the state->unit correspondence: the operators
                    # are learned exactly as in `pc`, then dealt to the wrong
                    # units, so their mean and cross-unit dispersion are identical
                    order_h = torch.argsort(torch.rand((n_cfg, hidden), device=device,
                                                       generator=perm_gen), dim=-1)
                    applied = torch.gather(
                        u, 1, order_h.view(n_cfg, hidden, 1, 1).expand(-1, -1, dim, rank))
                    applied_floor = torch.gather(floor_sum, 1, order_h)
                else:
                    applied, applied_floor = u, floor_sum

                latent = torch.einsum("khdr,khbd->khbr", applied, xu)
                eps = xu - torch.einsum("khdr,khbr->khbd", applied, latent)
                overlap = (eps * xu).sum(-1).clamp_min(0.0)       # (K,H,B)
                energy = eps.square().sum(-1) / dim
                if args.pc_precision:
                    reference = applied_floor / max(updates - 1, 1)
                    pi = energy / (energy + reference.unsqueeze(-1)).clamp_min(1e-30)
                else:
                    pi = torch.ones_like(energy)
                ratio = pi * overlap / xu_energy                  # (K,H,B)
                pi_sum += pi.mean((1, 2))
                unexplained_sum += (overlap / xu_energy).mean((1, 2))
                if on_grad:
                    g1 = torch.einsum("khb,khbd->khd", dh * (ratio if scalarised
                                                             else pi),
                                      xu if scalarised else eps) / batch
                    grad_ratio_sum += ratio.mean((1, 2))
                else:
                    grad_ratio_sum += 1.0

            beta1, beta2 = 0.9, 0.999
            m1.mul_(beta1).add_(g1, alpha=1 - beta1)
            v1.mul_(beta2).addcmul_(g1, g1, value=1 - beta2)
            m2.mul_(beta1).add_(g2, alpha=1 - beta1)
            v2.mul_(beta2).addcmul_(g2, g2, value=1 - beta2)
            bias1, bias2 = 1 - beta1 ** updates, 1 - beta2 ** updates
            step1 = (m1 / bias1) / ((v1 / bias2).sqrt() + 1e-8)
            step2 = (m2 / bias1) / ((v2 / bias2).sqrt() + 1e-8)

            if is_pc and on_step:
                latent_s = torch.einsum("khdr,khd->khr", applied, step1)
                kept = step1 - torch.einsum("khdr,khr->khd", applied, latent_s)
                confidence = pi.mean(-1)                          # (K,H)
                s_energy = step1.square().sum(-1).clamp_min(1e-30)
                ratio_s = confidence * (kept * step1).sum(-1).clamp_min(0.0) / s_energy
                step1 = (ratio_s.unsqueeze(-1) * step1 if scalarised
                         else confidence.unsqueeze(-1) * kept)
                step_ratio_sum += ratio_s.mean(-1)
            elif is_pc:
                step_ratio_sum += 1.0

            if is_precision:
                if args.precision_decay:
                    identity = torch.eye(dim, device=device)
                    geometry.mul_(1 - args.precision_decay).add_(
                        args.precision_decay * identity)
                projected = torch.einsum("khde,khe->khd", geometry, step1)
                energy_s = (step1 * step1).sum(-1)
                own = torch.where(energy_s > 0,
                                  (projected * step1).sum(-1) / energy_s.clamp_min(1e-30),
                                  torch.ones_like(energy_s))
                step_ratio_sum += own.mean(-1)
                grad_ratio_sum += 1.0
                step1 = projected
                commitment = delta.abs().mean(1).unsqueeze(-1)    # (K,H,1)
                direction = x.mean(1)                             # (K,D)
                mapped = torch.einsum("khde,ke->khd", geometry, direction)
                weight = args.precision_lambda * commitment
                denominator = (1.0 + (weight.squeeze(-1)
                                      * torch.einsum("khd,kd->kh", mapped, direction))
                               ).clamp_min(1e-12)
                geometry -= ((weight.unsqueeze(-1) * mapped.unsqueeze(-1)
                              * mapped.unsqueeze(-2))
                             / denominator.view(n_cfg, hidden, 1, 1))
                if method == "precision_diag":
                    geometry = torch.diag_embed(
                        torch.diagonal(geometry, dim1=-2, dim2=-1))
            elif not is_pc:
                grad_ratio_sum += 1.0
                step_ratio_sum += 1.0

            # The plasticity acts on the INCOMING row only. Gating `w2` as well
            # deadlocks the unit: `w2` starts at zero, so a gate read off its
            # gradient pins both at zero forever.
            w1 -= lr * step1
            w2 -= lr * step2
            if field is not None:
                w1.mul_(field)

            if is_pc:
                floor_sum += energy.mean(-1).detach()
            if learns_model:
                # Oja's subspace rule on the unit's OWN prediction error, weighted
                # by how far the unit actually moved on this sample, normalised by
                # input energy so the rate is scale-free. The commitment reference
                # is a CUMULATIVE mean: a sufficient statistic of the stream, not
                # an EMA with a horizon to tune.
                magnitude = dh.abs()
                commit_sum += magnitude.mean(-1).detach()
                commit = (magnitude * (max(updates, 1)
                                       / commit_sum.clamp_min(1e-20)).unsqueeze(-1)
                          ).clamp_max(args.pc_commit_cap)
                if permuted:
                    # the model still learns from ITS OWN state
                    latent = torch.einsum("khdr,khbd->khbr", u, xu)
                    eps = xu - torch.einsum("khdr,khbr->khbd", u, latent)
                du = torch.einsum("khb,khbd,khbr->khdr", commit / xu_energy,
                                  eps, latent)
                if shared_model:
                    du = du.mean(1, keepdim=True).expand_as(du)
                # `du` sums over the batch, so dividing by `batch` here left the
                # PER-STEP increment batch-invariant while the step COUNT falls as
                # samples/batch -- total model movement scaled as 1/B, and the gate
                # was inert at large batch (multiplier 0.982, dispersion 0.020 at
                # B=256 vs 0.728/0.424 at B=1). Arms compared across batch sizes
                # were comparing a firing gate against a dead one. `pc_lr` is now
                # a per-SAMPLE rate; identical at B=1, batch-invariant above.
                u += args.pc_lr * du
                if field is not None and not shared_model:
                    u.mul_(field.unsqueeze(-1))
                norm = u.norm(dim=-2, keepdim=True).clamp_min(1e-12)
                u.mul_(norm.clamp_max(2.0) / norm)

    per_region = evaluate(w1, w2, first, second, support, args, device, data_gen,
                          order)
    grad_ratio = (grad_ratio_sum / updates).cpu().numpy()
    step_ratio = (step_ratio_sum / updates).cpu().numpy()
    return configs, {
        "acquisition": per_region[-1].cpu().numpy(),
        "retention": per_region[:-1].mean(0).cpu().numpy(),
        "overall": per_region.mean(0).cpu().numpy(),
        "grad_ratio": grad_ratio,
        "step_ratio": step_ratio,
        "plasticity": grad_ratio * step_ratio,
        "pi": (pi_sum / updates).cpu().numpy() if is_pc else np.ones(n_cfg),
        "unexplained": ((unexplained_sum / updates).cpu().numpy() if is_pc
                        else np.ones(n_cfg)),
        "u_norm": (u.norm(dim=-2).mean((1, 2)).cpu().numpy() if is_pc
                   else np.zeros(n_cfg)),
        "autonomy": model_autonomy(u) if is_pc else np.zeros(n_cfg),
    }


KEYS = ("acquisition", "retention", "overall", "grad_ratio", "step_ratio",
        "plasticity", "pi", "unexplained", "u_norm", "autonomy")


def best_config(args, configs, out):
    """Best LR by mean overall, plus that LR's per-seed vectors."""
    best, row = None, None
    for value in args.lr_grid:
        picks = [i for i, c in enumerate(configs) if c["lr"] == value]
        stats = {k: float(np.mean(out[k][picks])) for k in KEYS}
        if best is None or stats["overall"] < best:
            best = stats["overall"]
            row = (value, stats, out["overall"][picks], out["acquisition"][picks],
                   out["retention"][picks])
    return row


def paired(diff):
    """Paired mean, standard error, t and sd. Seeds are matched across arms."""
    n = len(diff)
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n > 1 else 0.0
    return mean, sem, (mean / sem if sem > 0 else 0.0), sd


def main():
    args = tyro.cli(Args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    methods = {"all": METHODS, "pc_family": ("adam",) + PC_METHODS}.get(
        args.method) or tuple(args.method.split(","))
    for name in methods:
        if name not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
    if args.apply not in ("grad", "step", "both"):
        raise ValueError("apply must be grad, step or both")

    gen = torch.Generator(device=device).manual_seed(args.seed)
    teachers = make_teachers(args, device, gen)
    supports = make_supports(args, device, gen)

    print(f"{args.regions} regions x {args.samples_per_region} samples, "
          f"mode={args.region_mode}, shift {args.shift}, field {args.field_fraction}, "
          f"batch {args.batch}, rank {args.rank}, apply={args.apply}, "
          f"precision={args.pc_precision}, per_region_teacher="
          f"{args.per_region_teacher}, shuffle={args.shuffle}, "
          f"{len(methods)} arms x {len(args.lr_grid)} lrs x {args.seeds} seeds")
    start = time.perf_counter()
    results = {name: run_group(args, name, args.lr_grid, device, teachers, supports)
               for name in methods}
    print(f"done in {time.perf_counter() - start:.1f}s\n")

    rows = {name: best_config(args, *results[name]) for name in methods}
    print(f"{'method':>15} {'lr':>7} | {'acquire':>8} {'retain':>8} {'overall':>8} "
          f"| {'plast':>6} {'grad':>6} {'step':>6} {'pi':>6} {'unexp':>6} "
          f"{'|u|':>6} {'auton':>6}")
    for name in methods:
        value, stats, *_ = rows[name]
        print(f"{name:>15} {value:>7} | {stats['acquisition']:>8.4f} "
              f"{stats['retention']:>8.4f} {stats['overall']:>8.4f} | "
              f"{stats['plasticity']:>6.3f} {stats['grad_ratio']:>6.3f} "
              f"{stats['step_ratio']:>6.3f} {stats['pi']:>6.3f} "
              f"{stats['unexplained']:>6.3f} {stats['u_norm']:>6.3f} "
              f"{stats['autonomy']:>6.3f}")

    edge = [n for n in methods if rows[n][0] in (min(args.lr_grid), max(args.lr_grid))]
    if edge:
        print(f"\nWARNING: best LR sits at a grid EDGE for {', '.join(edge)}. A "
              f"damping mechanism lowers the EFFECTIVE learning rate, so if the "
              f"control cannot reach that rate the comparison is a learning-rate "
              f"artifact. Widen --lr-grid.")

    if args.seeds < 2:
        return
    for anchor in ("adam", "pc"):
        if anchor not in methods:
            continue
        others = [n for n in methods if n != anchor
                  and (anchor == "adam" or n.startswith("pc"))]
        if not others:
            continue
        base_overall, base_acq, base_ret = rows[anchor][2:5]
        print(f"\npaired vs {anchor} over {args.seeds} seeds "
              f"(negative = better than {anchor})")
        print(f"{'method':>15} | {'d overall':>10} {'sem':>8} {'t':>7} "
              f"{'seeds@80%':>10} | {'d acquire':>10} {'d retain':>10}")
        for name in others:
            _, _, overall, acquire, retain = rows[name]
            mean, sem, t, sd = paired(overall - base_overall)
            needed = (int(np.ceil((2.8 * sd / abs(mean)) ** 2))
                      if abs(mean) > 0 and sd > 0 else 0)
            print(f"{name:>15} | {mean:>+10.4f} {sem:>8.4f} {t:>+7.2f} "
                  f"{needed:>10d} | {float(np.mean(acquire - base_acq)):>+10.4f} "
                  f"{float(np.mean(retain - base_ret)):>+10.4f}")

    if "adam" not in methods:
        return
    reference = rows["adam"][2]
    spread, level = float(np.std(reference, ddof=1)), float(np.mean(reference))
    mde = 2.8 * spread / np.sqrt(args.seeds)
    print(f"\nresolution: adam overall {level:.4f}, across-seed sd {spread:.4f}. "
          f"An UNPAIRED test at {args.seeds} seeds detects {mde:.4f} "
          f"({100 * mde / level:.1f}% of the level) at 80% power. Pairing shrinks "
          f"this to the `seeds@80%` columns above, which is the honest seed "
          f"requirement for each arm's OBSERVED effect.")


if __name__ == "__main__":
    main()
