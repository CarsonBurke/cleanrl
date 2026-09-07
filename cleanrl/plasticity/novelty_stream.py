"""Per-perceptron, per-sample, state-conditional plasticity under distribution shift.

# WHY THIS TASK EXISTS

Every earlier member of this family (v1-v9, mirror, alloc, kalman, spike) computes a
multiplier from GRADIENT STATISTICS ACCUMULATED OVER HISTORY. That answers an
optimizer question -- "which coordinates have a consistent descent direction?" -- and
it is not the premise. The premise is that each perceptron decides from ITS OWN STATE
ON THE CURRENT SAMPLE. A level refreshed every 50 steps from a t-statistic is a
function of history, not of state, and it is per-connection, not per-perceptron.

The reason that drift was invisible is that the old diagnostic
(`noisy_stream_diagnostic.py`, `hidden_stream.py`) is an iid Gaussian stream whose only
structure is "some input columns are junk". Such a stream can reward exactly ONE
capability -- feature selection -- and contains NO STATE to condition on. So it selects
for whichever mechanism is the best feature-selection statistic, and state-dependence
is unmeasurable by construction. It cannot falsify the premise either way.

This file measures the other thing that is called plasticity: the capacity to keep
learning, and to not destroy what is already learned. That is inherently per-sample and
per-unit, and it only exists under a NON-STATIONARY input distribution.

# TASK

A fixed teacher over the whole input space. The input distribution moves: `regions`
Gaussian clusters presented in sequence, one after another. The target function never
changes, so anything forgotten was destroyed by interference, not by a moved target.

Three numbers, each normalised by the zero-predictor:
  acquisition -- error on the region being trained on now (can we still learn?)
  retention   -- error on every earlier region (did learning here break there?)
  overall     -- error on the full mixture

# MECHANISM

`novelty`: each perceptron reads its own preactivation `z_i` against a slow EWMA of its
own preactivation mean and variance. If this sample sits where the unit has already
been operating, the unit is competent here and is protected. If it sits in the tail,
the unit has not fit this region and stays plastic. The levels are renormalised to mean
one ACROSS UNITS on every sample, so the rule reallocates one sample's step budget
rather than changing its size -- there is no global learning-rate component, which is
the confound that faked four previous wins in this family.

`familiar` is the same rule with the direction inverted. It exists because a
reallocation rule that helps in either direction is measuring something else.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
import tyro

METHODS = ("adam", "novelty", "familiar", "uniform", "learned", "learned_shared",
           "statebin", "statebin_pooled", "precision", "precision_diag",
           "precision_shared")


@dataclass
class Args:
    input_dim: int = 64
    hidden: int = 64
    regions: int = 8
    """number of sequentially presented input clusters"""
    per_region_teacher: bool = True
    """Each region gets its OWN target function. With one global smooth teacher
    every region wants the same mapping, so no unit ever faces conflicting
    demands, interference is unattributable (measured: per-unit alignment with
    past regions was a coin flip, |r| <= 0.014 for every state feature), and
    state-conditional plasticity is worth exactly nothing. Conflict between
    contexts that share weights is the ONLY thing a state-conditional rule can
    buy that a per-parameter learning rate cannot."""
    shift: float = 3.0
    """distance of each cluster centre from the origin"""
    samples_per_region: int = 4000
    noise_std: float = 0.5
    batch: int = 8
    method: str = "all"
    lr_grid: tuple[float, ...] = (3e-4, 1e-3, 3e-3)
    novelty_rate: float = 0.01
    """EWMA rate for each unit's own preactivation statistics"""
    novelty_cap: float = 8.0
    """ceiling on a unit's level"""
    level_min: float = 0.0
    """floor on a unit's level"""
    shuffle: bool = False
    """draw every sample from a random region: the no-shift reference"""
    precision_lambda: float = 1.0
    """weight on a perceptron's accumulated input geometry. Its plasticity for
    the current sample is `x^T (I + lambda C)^-1 x / x^T x`: free along input
    directions it has not yet committed to, damped along the ones it has."""
    precision_decay: float = 0.0
    """leak on that geometry, i.e. how fast a perceptron forgives old commitments"""
    state_bins: int = 4
    """`statebin`: how many state cells each perceptron keeps evidence in"""
    refresh_every: int = 50
    """`statebin`: steps between recalibrations of the twin"""
    meta_lr: float = 3e-2
    """step size for each perceptron's own plasticity readout (`learned`)"""
    logit_cap: float = 2.0
    """bound on the readout's logit, so plasticity lies in [e^-c, e^+c]"""
    audit: bool = False
    """Instead of testing a mechanism, measure whether ANY per-unit state feature
    carries information about the ideal per-unit plasticity. This task has a
    ground truth: a unit's step is good exactly insofar as it does not raise the
    loss on regions already learned. So compute, per unit per step, the alignment
    between the step it is about to take and the descent direction of the PAST
    regions' loss, then ask which state features predict that alignment. If none
    do, the premise is not testable here and no mechanism should be built."""
    audit_every: int = 25
    eval_samples: int = 2048
    seeds: int = 4
    seed: int = 1
    device: str = "cuda"


def teacher_forward(x, teacher):
    w1, w2 = teacher
    return torch.tanh(x @ w1) @ w2


def make_teachers(args, device, gen):
    """One target function per region, or one shared function for every region."""
    def draw():
        return (torch.randn((args.input_dim, 8), device=device, generator=gen)
                / np.sqrt(args.input_dim),
                torch.randn((8,), device=device, generator=gen) * 0.7)
    if args.per_region_teacher:
        return [draw() for _ in range(args.regions)]
    shared = draw()
    return [shared] * args.regions


def evaluate(w1, w2, teachers, centres, args, device, gen):
    """Per-region error, normalised by the zero-predictor on that region."""
    scores = []
    for teacher, centre in zip(teachers, centres):
        x = centre + torch.randn((args.eval_samples, args.input_dim),
                                 device=device, generator=gen)
        clean = teacher_forward(x, teacher)
        prediction = (torch.tanh(torch.einsum("khd,bd->kbh", w1, x))
                      * w2.reshape(-1, 1, args.hidden)).sum(-1)
        scores.append(((prediction - clean).square().mean(1)
                       / clean.square().mean().clamp_min(1e-12)))
    return torch.stack(scores)                                    # (R, K)


def audit_step(w1, w2, m1, v1, *, beta1_pow, pre, act, rows, residual, x,
               z_mean, z_var, centres, teachers, args, device, gen):
    """Per-unit alignment with the PAST regions' descent direction, plus the
    per-unit state features that a plasticity rule would be allowed to read."""
    n_cfg, hidden, dim = w1.shape
    # ground truth: does this unit's pending step agree with what the already
    # learned regions want? Negative alignment means this step causes forgetting.
    past = torch.cat([c + torch.randn((256, dim), device=device, generator=gen)
                      for c in centres])
    clean_past = torch.cat([teacher_forward(chunk, past_teacher) for chunk, past_teacher
                            in zip(past.split(256), teachers)])
    act_past = torch.tanh(torch.einsum("khd,bd->kbh", w1, past))
    residual_past = (act_past * rows).sum(-1) - clean_past
    back_past = residual_past.unsqueeze(2) * rows * (1.0 - act_past.square())
    grad_past = torch.einsum("kbh,bd->khd", back_past, past) / past.shape[0]

    step = (m1 / (1 - 0.9 ** beta1_pow)) / ((v1 / (1 - 0.999 ** beta1_pow)).sqrt() + 1e-8)
    alignment = -(step * grad_past).sum(-1) / (
        step.norm(dim=-1).clamp_min(1e-12) * grad_past.norm(dim=-1).clamp_min(1e-12))

    deviation = ((pre.detach() - z_mean).abs()
                 / z_var.sqrt().clamp_min(1e-6)).mean(1)
    features = {
        "novelty |z-mu|/sd": deviation,
        "|preactivation|": pre.detach().abs().mean(1),
        "saturation 1-tanh^2": (1.0 - act.square()).mean(1),
        "|activation|": act.abs().mean(1),
        "|outgoing weight|": rows.abs().squeeze(1).expand(n_cfg, hidden),
        "incoming row norm": w1.norm(dim=-1),
        "|unit error share|": (residual.unsqueeze(2) * rows * act).abs().mean(1),
    }
    return {"alignment": alignment, "features": {k: v for k, v in features.items()}}


def report_audit(rows):
    """Pooled correlation of each state feature with the ideal-plasticity signal."""
    align = torch.cat([r["alignment"].reshape(-1) for r in rows]).double()
    print(f"\naudit: {len(rows)} snapshots, {align.numel()} unit-observations")
    print(f"  alignment with past-region descent: mean {align.mean():+.4f} "
          f"sd {align.std():.4f}, fraction harmful {(align < 0).double().mean():.3f}")
    print(f"\n  {'per-unit state feature':>24} | pearson r | spearman r")
    for name in rows[0]["features"]:
        value = torch.cat([r["features"][name].reshape(-1) for r in rows]).double()
        keep = torch.isfinite(value) & torch.isfinite(align)
        v, a = value[keep], align[keep]
        pearson = ((v - v.mean()) * (a - a.mean())).mean() / (
            v.std().clamp_min(1e-12) * a.std().clamp_min(1e-12))
        rank_v = v.argsort().argsort().double()
        rank_a = a.argsort().argsort().double()
        spearman = ((rank_v - rank_v.mean()) * (rank_a - rank_a.mean())).mean() / (
            rank_v.std().clamp_min(1e-12) * rank_a.std().clamp_min(1e-12))
        print(f"  {name:>24} | {pearson:+9.4f} | {spearman:+10.4f}")


def fdp_level(t_obs, t_null):
    """One minus the false-discovery proportion, calibrated against the twin.

    This is the mirror statistic, kept verbatim because it is the one thing in
    this family that measured real capability. What changes is what it is
    INDEXED BY: mirror accumulated evidence per connection over TIME, which can
    only ever say "this weight is useful in general". Here the same evidence is
    accumulated per (perceptron, state cell), so it says "this weight is useful
    WHEN MY STATE LOOKS LIKE THIS" -- the one claim no EMA or RMS accumulator can
    represent, because those are functions of time and this is a function of the
    input.
    """
    width = t_obs.shape[-1]
    null_sorted = t_null.sort(dim=-1).values
    obs_sorted = t_obs.sort(dim=-1).values
    false_ge = width - torch.searchsorted(null_sorted, t_obs.contiguous(), right=False)
    total_ge = (width - torch.searchsorted(obs_sorted, t_obs.contiguous(),
                                           right=False)).clamp_min(1)
    return (1.0 - false_ge.float() / total_ge.float()).clamp_(0.0, 1.0)


def report_conflict(ev_sum, ev_twin, args):
    """Does a perceptron's accumulated direction in one state cell OPPOSE its
    direction in another? That is the only quantity in this whole family that a
    per-weight accumulator provably cannot represent: it is a property of PAIRS
    of states, so no function of time can carry it. The twin supplies the null --
    the same arithmetic on sign-randomised evidence, where any apparent conflict
    is chance.
    """
    def cosines(evidence):
        unit = evidence / evidence.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        gram = torch.einsum("khid,khjd->khij", unit, unit)
        cells = gram.shape[-1]
        upper = torch.triu(torch.ones((cells, cells), device=gram.device),
                           diagonal=1).bool()
        return gram[..., upper].reshape(-1)

    observed, null = cosines(ev_sum), cosines(ev_twin)
    keep_o = torch.isfinite(observed) & (observed != 0)
    keep_n = torch.isfinite(null) & (null != 0)
    observed, null = observed[keep_o].double(), null[keep_n].double()
    print(f"\ncross-state conflict: {observed.numel()} (perceptron, state-pair) "
          f"observations")
    for label, value in (("observed", observed), ("twin null", null)):
        quantiles = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64,
                                 device=value.device)
        q = torch.quantile(value, quantiles).tolist()
        print(f"  {label:>10} cosine: mean {value.mean():+.4f}  "
              f"q05 {q[0]:+.3f} q25 {q[1]:+.3f} med {q[2]:+.3f} q75 {q[3]:+.3f} "
              f"q95 {q[4]:+.3f}  frac<0 {(value < 0).double().mean():.3f}")
    threshold = torch.quantile(null, torch.tensor([0.05], dtype=torch.float64,
                                                  device=null.device)).item()
    rate = (observed < threshold).double().mean().item()
    print(f"  conflict beyond the twin's 5th percentile ({threshold:+.3f}): "
          f"{rate:.3f} of observations  (chance = 0.050, so "
          f"{rate / 0.05:.1f}x enrichment)")


def state_features(pre, act):
    """What a perceptron is allowed to read: its own state on THIS sample.

    Deliberately raw. Earlier versions hand-picked a statistic (a t-test, an
    FDP level, a novelty z-score) and each one was a guess about which signal
    matters; the audit then scored those guesses against another guess about what
    ideal plasticity is. Both guesses are removed here: the unit gets raw state
    and the objective decides what to do with it.
    """
    return torch.stack([pre, pre.abs(), act, 1.0 - act.square(),
                        torch.ones_like(pre)], dim=-1)          # (K, B, H, F)


def run(args, configs, device):
    n_cfg = len(configs)
    dim, hidden = args.input_dim, args.hidden
    gen = torch.Generator(device=device).manual_seed(args.seed)
    teachers = make_teachers(args, device, gen)
    centres = [torch.randn((dim,), device=device, generator=gen) for _ in range(args.regions)]
    centres = [c / c.norm() * args.shift for c in centres]

    lr = torch.tensor([c["lr"] for c in configs], device=device).view(n_cfg, 1, 1)
    kind = [c["method"] for c in configs]
    is_novel = torch.tensor([k == "novelty" for k in kind], device=device).view(n_cfg, 1, 1)
    is_familiar = torch.tensor([k == "familiar" for k in kind], device=device).view(n_cfg, 1, 1)
    is_uniform = torch.tensor([k == "uniform" for k in kind], device=device).view(n_cfg, 1, 1)
    any_state = bool((is_novel | is_familiar | is_uniform).any().item())

    w1 = torch.randn((n_cfg, hidden, dim), device=device, generator=gen) / np.sqrt(dim)
    w2 = torch.zeros((n_cfg, 1, hidden), device=device)
    m1, v1 = torch.zeros_like(w1), torch.zeros_like(w1)
    m2, v2 = torch.zeros_like(w2), torch.zeros_like(w2)
    # each unit's own state statistics: the only thing the rule is allowed to read
    z_mean = torch.zeros((n_cfg, 1, hidden), device=device)
    z_var = torch.ones((n_cfg, 1, hidden), device=device)
    # each perceptron's own readout from its own state -> its own plasticity.
    # Zero init means plasticity is exactly one, so the run starts bit-identical
    # to Adam and any departure is something the objective actually wanted.
    n_feat = 5
    readout = torch.zeros((n_cfg, hidden, n_feat), device=device)
    is_learned = torch.tensor([k == "learned" for k in kind], device=device).view(n_cfg, 1, 1)
    is_shared = torch.tensor([k == "learned_shared" for k in kind],
                             device=device).view(n_cfg, 1, 1)
    any_learned = bool((is_learned | is_shared).any().item())
    # mirror evidence, indexed by (perceptron, state cell) instead of by time
    bins = args.state_bins
    ev_sum = torch.zeros((n_cfg, hidden, bins, dim), device=device)
    ev_sq = torch.zeros((n_cfg, hidden, bins, dim), device=device)
    ev_twin = torch.zeros((n_cfg, hidden, bins, dim), device=device)
    cell_level = torch.ones((n_cfg, hidden, bins), device=device)
    is_bin = torch.tensor([k == "statebin" for k in kind], device=device).view(n_cfg, 1, 1)
    is_bin_pooled = torch.tensor([k == "statebin_pooled" for k in kind],
                                 device=device).view(n_cfg, 1, 1)
    any_bin = bool((is_bin | is_bin_pooled).any().item())
    sign_gen = torch.Generator(device=device).manual_seed(args.seed + 7)
    # PREDICTIVE-CODING FORM. A scalar cannot say "move for these inputs and not
    # for those" -- that is a statement about DIRECTIONS in input space, so the
    # object has to be anisotropic and per perceptron. Each unit carries the
    # inverse geometry of the inputs it has already been modified along, and its
    # step is precision-weighted: free where its input is novel to it, damped
    # where it is already committed. `precision_diag` keeps ONLY the diagonal,
    # which is exactly an RMS/EMA accumulator and provably cannot represent a
    # direction -- so the pair isolates the state-conditional content itself.
    inverse_geometry = torch.eye(dim, device=device).expand(n_cfg, hidden, dim, dim)
    inverse_geometry = inverse_geometry.clone()
    is_precision = torch.tensor([k == "precision" for k in kind],
                                device=device).view(n_cfg, 1, 1)
    is_precision_diag = torch.tensor([k == "precision_diag" for k in kind],
                                     device=device).view(n_cfg, 1, 1)
    # ONE geometry for the whole layer. The probe found that units mostly commit
    # to the SAME direction here (median |cos| 0.77 between their most-damped
    # directions), because a shift-12 region centre dominates every unit's input.
    # If this arm matches the per-unit arm, then the directional content is doing
    # the work and the PER-PERCEPTRON part of the premise is inert on this task.
    is_precision_shared = torch.tensor([k == "precision_shared" for k in kind],
                                       device=device).view(n_cfg, 1, 1)
    is_precision = is_precision | is_precision_shared
    any_precision = bool((is_precision | is_precision_diag).any().item())

    batch, updates = args.batch, 0
    steps_per_region = max(args.samples_per_region // batch, 1)
    # dispersion ACROSS UNITS within one sample: the mean is one by construction,
    # so only the spread says whether a per-perceptron decision was made at all.
    level_spread = torch.zeros((n_cfg,), device=device)
    level_mean = torch.zeros((n_cfg,), device=device)
    audit_rows = []
    plasticity_report = torch.zeros((n_cfg,), device=device)
    report_count = 0
    spread_count = 0
    for region in range(args.regions):
        for _ in range(steps_per_region):
            if args.shuffle:
                pick = int(torch.randint(0, args.regions, (1,), generator=gen,
                                         device=device).item())
                centre = centres[pick]
            else:
                centre = centres[region]
            x = centre + torch.randn((batch, dim), device=device, generator=gen)
            clean = teacher_forward(x, teachers[pick if args.shuffle else region])
            y = clean + args.noise_std * torch.randn((batch,), device=device, generator=gen)
            pre = torch.einsum("khd,bd->kbh", w1, x)              # (K, B, H)
            act = torch.tanh(pre)
            rows = w2.reshape(n_cfg, 1, hidden)
            residual = (act * rows).sum(-1) - y                   # (K, B)

            # The level scales the REALIZED step, never the gradient: Adam divides
            # out any persistent per-row rescale of its input.
            #
            # The level is deliberately UNCONSTRAINED -- no budget conservation.
            # Forcing the levels to mean one is an invented mechanic, not part of
            # the premise, and it makes the rule exactly blind to a global shift
            # (when every unit's state is novel, the normalisation cancels it).
            # The learning-rate confound is handled by the `uniform` control arm
            # below instead of by constraining the mechanism.
            level = torch.ones((n_cfg, 1, hidden), device=device)
            if any_state:
                # a unit's competence at THIS sample, read from its own state only
                deviation = (pre.detach() - z_mean).abs() / z_var.sqrt().clamp_min(1e-6)
                novel = deviation.clamp(0.0, args.novelty_cap).mean(1, keepdim=True)
                raw = torch.where(is_novel | is_uniform, novel,
                                  1.0 / novel.clamp_min(1e-3))
                # `uniform` gets the identical GLOBAL magnitude with the
                # per-perceptron differentiation removed. Any gain it reproduces
                # was a learning-rate effect, not state-dependent plasticity.
                raw = torch.where(is_uniform, raw.mean(-1, keepdim=True), raw)
                level = torch.where(is_novel | is_familiar | is_uniform,
                                    raw.clamp(args.level_min, args.novelty_cap), level)
                z_mean.mul_(1 - args.novelty_rate).add_(
                    args.novelty_rate * pre.detach().mean(1, keepdim=True))
                z_var.mul_(1 - args.novelty_rate).add_(
                    args.novelty_rate * pre.detach().var(1, unbiased=False, keepdim=True))

            g2 = torch.einsum("kb,kbh->kh", residual, act).view(n_cfg, 1, hidden) / batch
            back = residual.unsqueeze(2) * rows * (1.0 - act.square())
            g1 = torch.einsum("kbh,bd->khd", back, x) / batch
            updates += 1
            level_spread += level.std(-1).mean(-1)
            level_mean += level.mean((1, 2))
            spread_count += 1
            if args.audit and region > 0 and updates % args.audit_every == 0:
                audit_rows.append(audit_step(w1, w2, m1, v1, beta1_pow=updates,
                                             pre=pre, act=act, rows=rows,
                                             residual=residual, x=x, z_mean=z_mean,
                                             z_var=z_var, centres=centres[:region],
                                             teachers=teachers[:region], args=args,
                                             device=device, gen=gen))
            if any_bin:
                # which state cell is this perceptron in, on this sample
                standard = (pre.detach() - z_mean) / z_var.sqrt().clamp_min(1e-6)
                cell = ((standard + 2.0) * (bins / 4.0)).floor().clamp_(0, bins - 1).long()
                per_sample = (residual.unsqueeze(2) * rows * (1.0 - act.square())
                              ).unsqueeze(-1) * x.view(1, batch, 1, dim)  # (K,B,H,D)
                flat = cell.permute(0, 2, 1).reshape(n_cfg, hidden, batch, 1)
                contribution = per_sample.permute(0, 2, 1, 3)             # (K,H,B,D)
                ev_sum.scatter_add_(2, flat.expand(-1, -1, -1, dim), contribution)
                ev_sq.scatter_add_(2, flat.expand(-1, -1, -1, dim), contribution.square())
                flips = torch.randint(0, 2, (1, 1, batch, 1), device=device,
                                      generator=sign_gen, dtype=torch.float32
                                      ).mul_(2.0).sub_(1.0)
                ev_twin.scatter_add_(2, flat.expand(-1, -1, -1, dim),
                                     contribution * flips)
                if updates % args.refresh_every == 0:
                    scale = ev_sq.clamp_min(1e-30)
                    t_obs = (ev_sum.square() / scale).sum(-1).sqrt()
                    t_null = (ev_twin.square() / scale).sum(-1).sqrt()
                    cell_level = fdp_level(t_obs.reshape(n_cfg, -1),
                                           t_null.reshape(n_cfg, -1)
                                           ).view(n_cfg, hidden, bins)
                current = cell_level.gather(2, cell.permute(0, 2, 1)).mean(-1)
                pooled = cell_level.mean(-1, keepdim=True).expand(-1, -1, bins)
                pooled = pooled.gather(2, cell.permute(0, 2, 1)).mean(-1)
                level = torch.where(is_bin, current.unsqueeze(1),
                                    torch.where(is_bin_pooled, pooled.unsqueeze(1), level))
            if any_learned:
                feats = state_features(pre.detach(), act.detach()).mean(1)  # (K,H,F)
                logit = (readout * feats).sum(-1, keepdim=True).transpose(1, 2)
                logit = torch.where(is_shared, logit.mean(-1, keepdim=True), logit)
                level = torch.where(is_learned | is_shared,
                                    logit.clamp(-args.logit_cap, args.logit_cap).exp(),
                                    level)
            beta1, beta2 = 0.9, 0.999
            m1.mul_(beta1).add_(g1, alpha=1 - beta1)
            v1.mul_(beta2).addcmul_(g1, g1, value=1 - beta2)
            m2.mul_(beta1).add_(g2, alpha=1 - beta1)
            v2.mul_(beta2).addcmul_(g2, g2, value=1 - beta2)
            bias1, bias2 = 1 - beta1 ** updates, 1 - beta2 ** updates
            step1 = (m1 / bias1) / ((v1 / bias2).sqrt() + 1e-8)
            step2 = (m2 / bias1) / ((v2 / bias2).sqrt() + 1e-8)
            if any_precision:
                if args.precision_decay:
                    identity = torch.eye(dim, device=device)
                    inverse_geometry.mul_(1 - args.precision_decay).add_(
                        args.precision_decay * identity)
                projected = torch.einsum("khde,khe->khd", inverse_geometry, step1)
                # a perceptron's plasticity on THIS sample: how much of its own
                # input direction is still unexplained by its own history
                energy = (step1 * step1).sum(-1)
                own = torch.where(energy > 0,
                                  (projected * step1).sum(-1) / energy.clamp_min(1e-30),
                                  torch.ones_like(energy))
                # Report the realized plasticity; do NOT feed it back as a gate.
                # The projection acts in the INPUT space of this unit's incoming
                # row, so it must not touch the outgoing weight: gating that too
                # deadlocks the unit (w2 starts at zero, so its incoming gradient
                # starts at zero, so a gate read off that gradient pins both at
                # zero forever -- which is exactly what the first run did).
                plasticity_report += own.mean(-1)
                report_count += 1
                step1 = torch.where((is_precision | is_precision_diag).view(n_cfg, 1, 1),
                                    projected, step1)
                # Sherman-Morrison: record this commitment, weighted by how far
                # the unit actually moved along it
                commitment = (residual.unsqueeze(2) * rows * (1.0 - act.square())
                              ).abs().mean(1).unsqueeze(-1)          # (K, H, 1)
                direction = x.mean(0)                                # (D,)
                mapped = torch.einsum("khde,e->khd", inverse_geometry, direction)
                weight = args.precision_lambda * commitment
                denominator = (1.0 + (weight.squeeze(-1)
                                      * (mapped * direction).sum(-1))).clamp_min(1e-12)
                update = (weight.unsqueeze(-1) * mapped.unsqueeze(-1)
                          * mapped.unsqueeze(-2)) / denominator.view(n_cfg, hidden, 1, 1)
                inverse_geometry -= torch.where(
                    is_precision.view(n_cfg, 1, 1, 1) | is_precision_diag.view(
                        n_cfg, 1, 1, 1), update, torch.zeros_like(update))
                if bool(is_precision_shared.any().item()):
                    layer_wide = inverse_geometry.mean(1, keepdim=True).expand_as(
                        inverse_geometry)
                    inverse_geometry = torch.where(
                        is_precision_shared.view(n_cfg, 1, 1, 1), layer_wide,
                        inverse_geometry)
                if bool(is_precision_diag.any().item()):
                    diagonal_only = torch.diag_embed(
                        torch.diagonal(inverse_geometry, dim1=-2, dim2=-1))
                    inverse_geometry = torch.where(
                        is_precision_diag.view(n_cfg, 1, 1, 1), diagonal_only,
                        inverse_geometry)
            w1 -= lr * step1 * level.view(n_cfg, hidden, 1)
            w2 -= lr * step2 * level
            if any_learned:
                # Objective: the loss on the samples that arrive NEXT. Nothing
                # about retention, forgetting or noise is asserted -- the future
                # of the actual stream is the only yardstick, and whether
                # protecting a unit helps is something it has to discover.
                x_next = centre + torch.randn((batch, dim), device=device, generator=gen)
                target = teacher_forward(x_next, teachers[region])
                act_next = torch.tanh(torch.einsum("khd,bd->kbh", w1, x_next))
                rows_next = w2.reshape(n_cfg, 1, hidden)
                residual_next = (act_next * rows_next).sum(-1) - target
                g2_next = torch.einsum("kb,kbh->kh", residual_next, act_next
                                       ).view(n_cfg, 1, hidden) / batch
                back_next = residual_next.unsqueeze(2) * rows_next * (1 - act_next.square())
                g1_next = torch.einsum("kbh,bd->khd", back_next, x_next) / batch
                # dL_next/dlevel_h = -lr * <grad_next_h, step_h>, exact first order
                d_level = -(lr.view(n_cfg, 1, 1) * (
                    (g1_next * step1).sum(-1).unsqueeze(1) + g2_next * step2))
                d_logit = d_level * level                      # level = exp(logit)
                grad_readout = d_logit.transpose(1, 2) * feats
                if bool(is_shared.any().item()):
                    shared_grad = grad_readout.mean(1, keepdim=True).expand_as(grad_readout)
                    grad_readout = torch.where(is_shared.view(n_cfg, 1, 1),
                                               shared_grad, grad_readout)
                readout -= args.meta_lr * torch.where(
                    (is_learned | is_shared).view(n_cfg, 1, 1), grad_readout,
                    torch.zeros_like(grad_readout))

    if any_bin and args.audit:
        report_conflict(ev_sum, ev_twin, args)
    if audit_rows:
        report_audit(audit_rows)
    per_region = evaluate(w1, w2, teachers, centres, args, device, gen)  # (R, K)
    return {"acquisition": per_region[-1].cpu().numpy(),
            "retention": per_region[:-1].mean(0).cpu().numpy(),
            "overall": per_region.mean(0).cpu().numpy(),
            "level_spread": (level_spread / max(spread_count, 1)).cpu().numpy(),
            "level_mean": (level_mean / max(spread_count, 1)).cpu().numpy(),
            "plasticity": (plasticity_report / max(report_count, 1)).cpu().numpy()}


def main():
    args = tyro.cli(Args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    methods = METHODS if args.method == "all" else tuple(args.method.split(","))
    for name in methods:
        if name not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
    configs = [{"method": m, "lr": v, "seed": s}
               for m in methods for v in args.lr_grid for s in range(args.seeds)]
    print(f"{args.regions} regions x {args.samples_per_region} samples, shift "
          f"{args.shift}, batch {args.batch}, shuffle={args.shuffle}, "
          f"{len(configs)} configs")
    start = time.perf_counter()
    out = run(args, configs, device)
    print(f"{len(configs)} configs in one pass, {time.perf_counter() - start:.1f}s\n")

    print(f"{'method':>10} {'lr':>7} | {'acquire':>8} {'retain':>8} {'overall':>8} | lvl mean lvl sd  plast")
    for name in methods:
        best, row = None, None
        for value in args.lr_grid:
            picks = [i for i, c in enumerate(configs)
                     if c["method"] == name and c["lr"] == value]
            stats = {k: float(np.mean(out[k][picks])) for k in
                     ("acquisition", "retention", "overall", "level_spread",
                      "level_mean", "plasticity")}
            if best is None or stats["overall"] < best:
                best, row = stats["overall"], (value, stats)
        value, stats = row
        print(f"{name:>10} {value:>7} | {stats['acquisition']:>8.4f} "
              f"{stats['retention']:>8.4f} {stats['overall']:>8.4f} | "
              f"{stats['level_mean']:>8.3f} {stats['level_spread']:.3f} "
              f"{stats['plasticity']:>6.3f}")


if __name__ == "__main__":
    main()
