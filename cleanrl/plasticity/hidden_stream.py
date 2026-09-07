"""Mirror plasticity inside a hidden layer: does it work for a NETWORK?

Everything measured so far -- the noisy-stream diagnostic and the market stream
-- used a single linear unit, where "each perceptron decides" is a statement
about one perceptron. This is the untested claim that matters for transfer to
PPO and to language models: a hidden unit's incoming weights receive a gradient
that is a PRODUCT of the input and a backpropagated error whose sign flips as
the layer above reorganises, and the per-connection SNR is far worse than the
per-output-unit SNR of a linear probe.

TASK. A dense teacher: `x ~ N(0, I)` in `input_dim` dimensions of which only the
first `useful` actually matter, target `y = teacher(x[:useful]) + noise` with the
noise variance dominating. A two-layer student must both fit the teacher and
refuse the useless input columns. Batch size 1, single pass, no replay. Test MSE
is measured against the NOISE-FREE teacher output on fresh samples, so absorbing
noise is punished rather than rewarded.

WHY BOTH GRANULARITIES ARE RUN. Per-connection evidence at realistic
per-connection SNR is often too weak to ever open (measured: never opens at SNR
0.05 over 400 steps, which is correct refusal). Pooling the energy
`sum_j t_ji^2` over the units that share an input multiplies the evidence by the
layer width and turns that refusal into a decision, at the cost of denying a
single unit the right to disagree with its neighbours about an input. This
script measures which trade wins.

Every method is reported at ITS OWN best learning rate over a shared grid, and
all configs advance in ONE vectorized pass, so the whole sweep costs one run.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import tyro

METHODS = ("sgd", "adam", "mirror_conn", "mirror_input", "mirror_alloc",
           "softhinge_conn", "softhinge_input", "softhinge_alloc",
           "softhinge_credit", "softhinge_free", "softhinge_uniform",
           "softhinge_ownnull", "softhinge_ownfree", "softhinge_canon", "softhinge_prior", "softhinge_odds", "softhinge_floor", "kalman",
           "spike", "cellshrink", "cellshrink_marginal", "cellshrink_shuffled",
           "oracle")


@dataclass
class Args:
    input_dim: int = 1024
    useful: int = 4
    """input columns that actually drive the teacher"""
    hidden: int = 64
    samples: int = 60000
    """SAMPLE budget, held fixed across batch sizes so `steps = samples // batch`.
    Batching exists to average gradient noise; if a per-connection plasticity
    level does that job, batch 1 should match or beat a large batch at equal
    sample cost. That is the claim this argument exists to test."""
    batch: int = 1
    eval_steps: int = 4096
    noise_std: float = 2.0
    """target noise; the teacher's own output has unit-ish scale"""
    method: str = "all"
    lr_grid: tuple[float, ...] = (3e-4, 1e-3, 3e-3)
    alloc_cap: float = 8.0
    """ceiling on the reallocated level. `mirror_alloc` rescales the levels to
    mean one, so shut connections FUND larger steps on confident ones instead of
    the layer merely taking a smaller step. The capped `1 - FDP` level can never
    exceed 1, so it cannot raise the step on the coordinates that actually carry
    signal -- and those coordinates' own noise, not the junk, is what sets the
    usable learning rate (the ORACLE's optimum is at the same LR as Adam's)."""
    prior_var: float = 1e-3
    """`kalman` only: prior variance on each weight, i.e. its INITIAL plasticity.
    There is no learning rate: the step size IS the posterior variance."""
    process_var: float = 0.0
    """`kalman` only: process noise added to every posterior variance each step.
    This is the principled form of forgetting -- it does not discard evidence,
    it re-admits uncertainty, so a connection whose world changed regains
    plasticity while one that was merely quiet does not."""
    lr_warmup: int = 0
    """linear LR warmup over this many steps, for EVERY method. `mirror_alloc`'s
    level is an evidence-driven schedule (small while evidence is thin, large
    once it accrues), and a constant-LR oracle cannot express that. This is the
    control that decides whether beating the oracle is a scheduling effect."""
    track_levels: bool = False
    hinge_sharpness: float = 24.0
    """softplus sharpness for the `softhinge_*` levels."""
    level_floor: float = 0.01
    """floor inside the global stability factor: caps it at 1/level_floor."""
    prior_steps: float = 20000.0
    """`softhinge_prior`: samples over which the fully-plastic prior decays."""
    teacher_switch_at: float = 0.0
    """fraction of the stream after which the teacher is redrawn (dense regime change)."""
    het_ratio: float = 1.0
    """noise multiplier in the loud regime; the regime is a function of the input
    itself (sign of the leading useful coordinate), so it is visible in every
    unit's own preactivation. At 1.0 the stream is homoscedastic and any
    state-conditional rule MUST be inert -- that is the control, not a failure."""
    state_cells: int = 8
    """`cellshrink`: state cells each perceptron accumulates evidence in"""
    refresh_every: int = 50
    score_after: int = 0
    seed: int = 1
    cuda: bool = True


def teacher_forward(x, args, weights):
    """A fixed nonlinear teacher reading only the first `useful` columns."""
    first, second = weights
    return torch.tanh(x[..., :args.useful] @ first).mul(second).sum(-1)


def cell_weight(total, square, count, minimum=4.0):
    """The fraction of a state cell's teaching signal that is real, in [0, 1].

    Sufficient statistics only -- `n`, `sum delta`, `sum delta^2` -- accumulated
    over the STREAM, one sample at a time, indexed by the perceptron's own state.
    Nothing here is an EMA: there is no decay horizon and no forgetting, so the
    estimate strengthens with evidence and the rule is batch-agnostic. Batch size
    changes only how fast evidence arrives, never what is computed, so batch 1 is
    the native case rather than a degraded one.

    The naive explained fraction is biased, because E[mean^2] = mu^2 + sigma^2/n.
    Subtracting that is exact:

        explained = max(mean^2 - s2/n, 0)
        weight    = explained / (explained + s2/n)

    which is the shrinkage factor for a mean with estimated variance. Pure noise
    gives exactly 0, noiseless signal exactly 1. There is no scale to tune, and
    the weight cannot exceed one, so it reweights DATA rather than rescaling the
    step -- a per-sample weight that varies with state survives Adam, whereas a
    persistent per-row scale is divided straight back out.
    """
    safe = count.clamp_min(1.0)
    mean = total / safe
    variance = (square / safe - mean.square()).clamp_min(0.0) * safe / (safe - 1).clamp_min(1.0)
    mean_variance = (variance / safe).clamp_min(1e-30)
    explained = (mean.square() - mean_variance).clamp_min(0.0)
    weight = explained / (explained + mean_variance)
    return torch.where(count >= minimum, weight, torch.ones_like(weight))


def softhinge_level(t_obs_sq, t_null_sq, sharpness):
    """Saturating, differentiable certainty against a family-wise null.

        certainty = softplus(k * (1 - z^2/t^2)) / k

    `z^2` is the LARGEST squared statistic the sign-randomized twin produces, so
    it is the family-wise null over however many coordinates are being tested --
    no threshold constant, no scale. The level saturates at 1 once evidence
    accrues (which is what a plain ratio gate fails to do: it keeps taxing the
    signal), and a null coordinate sits at softplus(-k)/k, far below the 1/sqrt(D)
    magnitude at which absorbed noise starts to matter.
    """
    z_sq = t_null_sq.flatten(1).max(dim=1).values.view(-1, *([1] * (t_obs_sq.dim() - 1)))
    return F.softplus(sharpness * (1.0 - z_sq / t_obs_sq.clamp_min(1e-30))) / sharpness


def fdp_level(t_obs, t_null):
    """One minus the false-discovery proportion, per config row."""
    width = t_obs.shape[-1]
    null_sorted = t_null.sort(dim=-1).values
    obs_sorted = t_obs.sort(dim=-1).values
    false_ge = width - torch.searchsorted(null_sorted, t_obs.contiguous(), right=False)
    total_ge = (width - torch.searchsorted(obs_sorted, t_obs.contiguous(),
                                           right=False)).clamp_min(1)
    return (1.0 - false_ge.float() / total_ge.float()).clamp_(0.0, 1.0)


def run_all(args, configs, device):
    n_cfg = len(configs)
    dim, hidden = args.input_dim, args.hidden
    gen = torch.Generator(device=device).manual_seed(args.seed)
    # list, not tuple: a regime change redraws the teacher in place
    teacher = [torch.randn((args.useful, 4), device=device, generator=gen),
               torch.randn((4,), device=device, generator=gen) * 0.7]
    regime_step = 0  # set once `steps` is known, below

    lr = torch.tensor([c["lr"] for c in configs], device=device).view(n_cfg, 1, 1)
    is_adam = torch.tensor([c["method"] != "sgd" for c in configs],
                           device=device).view(n_cfg, 1, 1)
    gate_kind = [c["method"] for c in configs]

    scale1 = 1.0 / np.sqrt(dim)
    w1 = torch.randn((n_cfg, hidden, dim), device=device, generator=gen) * scale1
    w2 = torch.zeros((n_cfg, 1, hidden), device=device)
    m1, v1 = torch.zeros_like(w1), torch.zeros_like(w1)
    m2, v2 = torch.zeros_like(w2), torch.zeros_like(w2)
    a1, q1, r1 = (torch.zeros_like(w1) for _ in range(3))
    level1 = torch.ones_like(w1)
    is_credit = torch.tensor([c["method"] == "softhinge_credit" for c in configs],
                             device=device).view(n_cfg, 1, 1)
    is_canon = torch.tensor([c["method"] == "softhinge_canon" for c in configs],
                            device=device).view(n_cfg, 1, 1)
    oracle_mask = torch.zeros((1, 1, dim), device=device)
    oracle_mask[..., :args.useful] = 1.0

    # KALMAN: per-connection posterior variance IS the plasticity. No lr.
    # SPIKE: posterior variance TIMES the probability the connection is relevant.
    # The Kalman denominator makes reallocation automatic -- shutting a junk
    # connection lowers `sum(P phi^2)`, which enlarges the step on every
    # surviving one. No cap, no learning rate, no budget bookkeeping.
    post1 = torch.full_like(w1, args.prior_var)
    post2 = torch.full_like(w2, args.prior_var)
    obs_var = torch.ones((n_cfg,), device=device)
    is_kalman = torch.tensor([c["method"] in ("kalman", "spike") for c in configs],
                             device=device).view(n_cfg, 1, 1)
    any_kalman = bool(is_kalman.any().item())
    # state-cell sufficient statistics, per (config, unit, cell)
    n_cells = args.state_cells
    # PER CONNECTION, per state cell. A per-perceptron scalar multiplies a unit's
    # whole row, so it cannot express which INPUTS are junk -- measured: all three
    # unit-level arms landed at 0.71-0.75 against 0.200 for per-connection
    # mirror_alloc, indistinguishable from each other, because the task's
    # difficulty is per-connection and a row scalar is structurally blind to it.
    # This is mirror's granularity (mirror being exactly the marginal case) with
    # state conditioning added and no EMA anywhere.
    cell_n = torch.zeros((n_cfg, hidden, n_cells, 1), device=device)
    cell_sum = torch.zeros((n_cfg, hidden, n_cells, dim), device=device)
    cell_sq = torch.zeros((n_cfg, hidden, n_cells, dim), device=device)
    state_mean = torch.zeros((n_cfg, hidden), device=device)
    state_scale = torch.ones((n_cfg, hidden), device=device)
    state_count = torch.zeros((n_cfg, hidden), device=device)
    is_cell = torch.tensor([c["method"] == "cellshrink" for c in configs],
                           device=device).view(n_cfg, 1, 1)
    is_cell_marginal = torch.tensor([c["method"] == "cellshrink_marginal" for c in configs],
                                    device=device).view(n_cfg, 1, 1)
    is_cell_shuffled = torch.tensor([c["method"] == "cellshrink_shuffled" for c in configs],
                                    device=device).view(n_cfg, 1, 1)
    any_cell = bool((is_cell | is_cell_marginal | is_cell_shuffled).any().item())
    cell_gen = torch.Generator(device=device).manual_seed(args.seed + 11)
    weight_sum = torch.zeros((n_cfg,), device=device)
    weight_spread = torch.zeros((n_cfg,), device=device)
    weight_count = 0
    squared = torch.zeros((n_cfg,), device=device)
    trivial = torch.zeros((n_cfg,), device=device)
    scored = 0
    sign_gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    trajectory = []

    batch = args.batch
    steps = max(args.samples // batch, 1)
    if args.teacher_switch_at > 0:
        regime_step = int(steps * args.teacher_switch_at)
    for step in range(steps):
        if regime_step and step == regime_step:
            # DENSE regime change: same inputs, new target function. Nothing to
            # reject, everything to relearn -- the RL-shaped case, and the cell no
            # harness here has covered (all were sparse-junk and stationary).
            teacher[0] = torch.randn_like(teacher[0])
            teacher[1] = torch.randn_like(teacher[1]) * 0.7
        x = torch.randn((batch, dim), device=device, generator=gen)
        clean = teacher_forward(x, args, teacher)                  # (B,)
        # regime is a FUNCTION OF THE INPUT, so it is legible in unit state: the
        # same sample is loud for every unit that sees it. This is the only task
        # property under which "decide from your own state how much to move" has
        # anything to decide.
        loud = (x[:, 0] > 0).float()
        scale = args.noise_std * (1.0 + loud * (args.het_ratio - 1.0))
        y = clean + scale * torch.randn((batch,), device=device, generator=gen)
        act = torch.tanh(torch.einsum("khd,bd->kbh", w1, x))       # (K, B, H)
        rows = w2.reshape(n_cfg, 1, hidden)
        prediction = (act * rows).sum(-1)                          # (K, B)
        residual = prediction - y
        if step >= args.score_after:
            squared += (prediction.detach() - clean).square().mean(1)
            trivial += clean.square().mean()
            scored += 1

        # manual backward of 0.5 * mean_b residual^2
        g2 = torch.einsum("kb,kbh->kh", residual, act).view(n_cfg, 1, hidden) / batch
        back = residual.unsqueeze(2) * rows * (1.0 - act.square())  # (K, B, H)
        g1 = torch.einsum("kbh,bd->khd", back, x) / batch

        if any_cell:
            # the perceptron's own state, standardised by its own running scale
            unit_state = torch.einsum("khd,bd->kbh", w1, x).detach().mean(1)
            centred = (unit_state - state_mean) / state_scale.clamp_min(1e-6)
            cell = ((centred + 2.0) * (n_cells / 4.0)).floor().clamp_(0, n_cells - 1).long()
            marginal = torch.zeros_like(cell)
            chosen = torch.where(is_cell_marginal.view(n_cfg, 1), marginal, cell)
            if bool(is_cell_shuffled.any().item()):
                scrambled = torch.randint(0, n_cells, cell.shape, device=device,
                                          generator=cell_gen)
                chosen = torch.where(is_cell_shuffled.view(n_cfg, 1), scrambled, chosen)
            # this sample's per-connection gradient for every unit
            signal = (residual.unsqueeze(2) * rows * (1.0 - act.square())
                      ).detach().mean(1).unsqueeze(-1) * x.mean(0).view(1, 1, dim)
            index = chosen.view(n_cfg, hidden, 1, 1)
            gathered = index.expand(-1, -1, -1, dim)
            weight = cell_weight(cell_sum.gather(2, gathered).squeeze(2),
                                 cell_sq.gather(2, gathered).squeeze(2),
                                 cell_n.gather(2, index).squeeze(2).expand(-1, -1, dim))
            cell_n.scatter_add_(2, index, torch.ones_like(index, dtype=cell_n.dtype))
            cell_sum.scatter_add_(2, gathered, signal.unsqueeze(2))
            cell_sq.scatter_add_(2, gathered, signal.square().unsqueeze(2))
            state_count += 1.0
            rate = (1.0 / state_count.clamp_min(1.0))
            state_mean += rate * (unit_state - state_mean)
            state_scale += rate * ((unit_state - state_mean).abs() - state_scale)
            # reweight this sample's contribution, per connection
            gate_all = (is_cell | is_cell_marginal | is_cell_shuffled).view(n_cfg, 1, 1)
            g1 = torch.where(gate_all, g1 * weight, g1)
            weight_sum += weight.mean((-1, -2))
            weight_spread += weight.std(-1).mean(-1)
            weight_count += 1

        # evidence from the RAW gradient of the hidden layer.
        #
        # `softhinge_credit` tests a specific leak. A hidden unit's gradient is
        # delta_h * x with delta_h = residual * w2_h * (1 - a^2), and w2_h changes
        # SIGN and SCALE while evidence accumulates, so sum_t g mixes credit from
        # eras when the unit meant something different downstream. The statistic
        # then dilutes itself for reasons that have nothing to do with whether the
        # input coordinate is predictive. Dividing by the unit's current outgoing
        # magnitude before accumulating makes the evidence invariant to that,
        # leaving only the input-side question the statistic is meant to answer.
        # Headroom being probed: the same statistic reaches 0.04x the
        # zero-predictor on a linear stream but only 0.18x through one hidden
        # layer, a 4.5x gap that is 25x larger than the entire per-unit ceiling.
        credit_scale = w2.abs().view(n_cfg, hidden, 1).clamp_min(1e-6)
        evidence_g = torch.where(is_credit | is_canon, g1 / credit_scale, g1)
        a1 += evidence_g
        q1 += evidence_g * evidence_g
        flips = torch.randint(0, 2, (1, 1, dim), device=device, generator=sign_gen,
                              dtype=torch.float32).mul_(2.0).sub_(1.0)
        r1.addcmul_(evidence_g, flips)
        if step % args.refresh_every == 0:
            scale = q1.sqrt().clamp_min(1e-30)
            t_obs, t_null = a1.abs() / scale, r1.abs() / scale
            hinge_conn = softhinge_level(t_obs.square(), t_null.square(),
                                         args.hinge_sharpness)
            row_obs, row_null = t_obs.square().sum(1), t_null.square().sum(1)
            hinge_row = softhinge_level(row_obs, row_null,
                                        args.hinge_sharpness).unsqueeze(1).expand_as(w1)
            # UNCONSTRAINED: a perceptron sets its own effective learning rate from
            # its own evidence, with no budget and no mean-one renormalisation. The
            # scale is `t^2/z^2`, how many times the family-wise null its evidence
            # is, so a unit with strong evidence takes a LARGE step and one with
            # none takes ~0 -- and crucially, when NO unit has evidence every level
            # falls together, which a conserved budget cannot express (normalising
            # to mean one makes the rule exactly blind to a global shift, and forces
            # a zero-sum trade where protecting one unit MUST amplify another).
            # The global scale it leaves free is precisely what the LR sweep absorbs,
            # so this is the arm that decides whether conservation was load-bearing
            # or was scaffolding I invented.
            row_z = row_null.flatten(1).max(dim=1).values.view(-1, 1)
            # AUTONOMOUS: each perceptron thresholds against ITS OWN twin, not the
            # layer's worst. The family-wise max is a population normalisation --
            # it couples every unit's certainty to the noisiest coordinate anywhere
            # in the layer, so one unit getting noisier lowers everyone's level.
            # Here the only quantities a unit reads are its own accumulated
            # evidence and its own sign-randomised twin.
            own_z = t_null.square().max(dim=2, keepdim=True).values
            own_obs = t_obs.square()
            own_cert = F.softplus(args.hinge_sharpness
                                  * (1.0 - own_z / own_obs.clamp_min(1e-30))
                                  ) / args.hinge_sharpness
            free_row = (softhinge_level(row_obs, row_null, args.hinge_sharpness)
                        * (row_obs / row_z.clamp_min(1e-30))
                        ).clamp_(0.0, args.alloc_cap).unsqueeze(1).expand_as(w1)
            conn = fdp_level(t_obs.reshape(n_cfg, -1),
                             t_null.reshape(n_cfg, -1)).view_as(w1)
            pooled = fdp_level(t_obs.square().sum(1), t_null.square().sum(1))
            pooled = pooled.unsqueeze(1).expand_as(w1)
            for index, kind in enumerate(gate_kind):
                if kind == "mirror_conn":
                    level1[index] = conn[index]
                elif kind == "mirror_input":
                    level1[index] = pooled[index]
                elif kind == "mirror_alloc":
                    row = pooled[index]
                    level1[index] = (row * (row.numel() / row.sum().clamp_min(1e-12))
                                     ).clamp_(0.0, args.alloc_cap)
                elif kind == "softhinge_conn":
                    level1[index] = hinge_conn[index]
                elif kind == "softhinge_input":
                    level1[index] = hinge_row[index]
                elif kind == "softhinge_floor":
                    # level = c_i / (mean(c) + c_floor)
                    #
                    # The global factor is a STABILITY term, not a budget. Adam's
                    # usable LR is set by the noise arriving through ALL coordinates;
                    # once most are closed the layer admits far less noise and a
                    # larger step is stable for the survivors, which is where the
                    # measured 10-39x amplification comes from and why a rule capped
                    # at 1 gives up 2x. But 1/mean(c) has no floor, so when NOTHING
                    # is admitted it manufactures amplification out of nothing and
                    # forces units to absorb noise -- measured 3.5x more than plain
                    # Adam on an unlearnable stream. The floor bounds the global
                    # factor at 1/c_floor, so a layer with no evidence anywhere
                    # stays shut instead of being inflated by its own emptiness.
                    cell = own_cert[index]
                    level1[index] = (cell / (cell.mean() + args.level_floor)
                                     ).clamp_(0.0, args.alloc_cap)
                elif kind == "softhinge_odds":
                    # AUTONOMOUS and UNBOUNDED ABOVE. A perceptron sets its own
                    # effective learning rate relative to the baseline with no
                    # population term of any kind:
                    #     level = c / (1 - c + 1/cap)
                    # c -> 0 gives level -> 0 (a unit with no evidence does not
                    # move, no matter what its neighbours report -- measured: the
                    # mean-normalised form absorbs 3.5x MORE noise than plain Adam
                    # on a stream with nothing to learn, because it cannot express
                    # "nobody should move"), and c -> 1 gives level -> cap, so a
                    # unit that is certain can take steps far ABOVE baseline, which
                    # is the dynamic range the bounded certainty could never have.
                    # Unlike t^2/z^2 this does not grow with sample count, so it is
                    # a plasticity decision rather than a learning-rate ramp.
                    cell = own_cert[index]
                    level1[index] = cell / (1.0 - cell + 1.0 / args.alloc_cap)
                elif kind == "softhinge_prior":
                    # NO competition anywhere: absolute per-unit certainty with an
                    # honest prior. With no evidence a unit is fully plastic and
                    # decays toward its own verdict as evidence arrives, instead of
                    # starting frozen and being rescued by its neighbours' failure.
                    # prior = 1/(1 + n/n0) -> the rule is a genuine posterior-like
                    # interpolation, and a unit that says "nothing here" STAYS at
                    # ~0 no matter what the rest of the layer reports.
                    weight_prior = 1.0 / (1.0 + step / max(args.prior_steps, 1))
                    cell = own_cert[index]
                    level1[index] = cell + (1.0 - cell) * weight_prior
                elif kind == "softhinge_canon":
                    # THE CANONICAL FORM: lr_eff(i) = L * c_i / mean(c), where c_i
                    # is a perceptron's own saturating certainty against its OWN
                    # twin, on evidence made invariant to its own outgoing weight
                    # scale. No population threshold, no budget cap semantics, no
                    # magnitude reward, and the global effective LR is left exactly
                    # equal to the baseline's.
                    cell = own_cert[index]
                    level1[index] = (cell * (cell.numel() / cell.sum().clamp_min(1e-12))
                                     ).clamp_(0.0, args.alloc_cap)
                elif kind == "softhinge_ownnull":
                    # autonomous threshold, global factor still 1/mean
                    cell = own_cert[index]
                    level1[index] = (cell * (cell.numel() / cell.sum().clamp_min(1e-12))
                                     ).clamp_(0.0, args.alloc_cap)
                elif kind == "softhinge_ownfree":
                    # autonomous threshold AND no population term at all: the
                    # effective learning rate is baseline x the unit's own
                    # certainty, nothing else. The global scale is the swept LR.
                    level1[index] = own_cert[index]
                elif kind == "softhinge_free":
                    level1[index] = free_row[index]
                elif kind == "softhinge_uniform":
                    # matched MEAN, zero dispersion: isolates how much of the
                    # result is per-connection differentiation and how much is a
                    # learning-rate change in disguise.
                    level1[index] = free_row[index].mean().expand_as(w1[index])
                elif kind == "softhinge_credit":
                    row = hinge_row[index]
                    level1[index] = (row * (row.numel() / row.sum().clamp_min(1e-12))
                                     ).clamp_(0.0, args.alloc_cap)
                elif kind == "softhinge_alloc":
                    row = hinge_row[index]
                    level1[index] = (row * (row.numel() / row.sum().clamp_min(1e-12))
                                     ).clamp_(0.0, args.alloc_cap)
                elif kind == "oracle":
                    level1[index] = oracle_mask.expand_as(w1[index:index + 1])[0]
                elif kind == "spike":
                    level1[index] = pooled[index]

        warm = min(1.0, (step + 1) / args.lr_warmup) if args.lr_warmup else 1.0
        beta1, beta2 = 0.9, 0.999
        m1.mul_(beta1).add_(g1, alpha=1 - beta1)
        v1.mul_(beta2).addcmul_(g1, g1, value=1 - beta2)
        m2.mul_(beta1).add_(g2, alpha=1 - beta1)
        v2.mul_(beta2).addcmul_(g2, g2, value=1 - beta2)
        bias1, bias2 = 1 - beta1 ** (step + 1), 1 - beta2 ** (step + 1)
        step1 = (m1 / bias1) / ((v1 / bias2).sqrt() + 1e-8)
        step2 = (m2 / bias1) / ((v2 / bias2).sqrt() + 1e-8)
        gradient_step1 = (warm * lr) * torch.where(is_adam, step1, g1) * level1
        gradient_step2 = (warm * lr) * torch.where(is_adam, step2, g2)
        if any_kalman:
            if batch != 1:
                raise ValueError("kalman requires batch 1")
            # features of the local linearization: g = residual * phi
            phi2 = act.reshape(n_cfg, 1, hidden)
            phi1 = (rows * (1.0 - act.square())).reshape(n_cfg, hidden, 1) * x.view(1, 1, dim)
            obs_var.mul_(0.999).add_(0.001 * residual.squeeze(1).detach().square())
            # `level1` is 1 for `kalman` and the inclusion probability for `spike`
            eff1 = post1 * level1
            denom = ((eff1 * phi1.square()).sum((1, 2))
                     + (post2 * phi2.square()).sum((1, 2)) + obs_var)
            gain = denom.view(n_cfg, 1, 1).clamp_min(1e-30)
            kal1 = eff1 * phi1 * residual.view(n_cfg, 1, 1) / gain
            kal2 = post2 * phi2 * residual.view(n_cfg, 1, 1) / gain
            # information gained about a weight the spike says is zero is zero,
            # so an irrelevant connection never becomes falsely confident.
            post1 -= (eff1 * phi1).square() / gain
            post2 -= (post2 * phi2).square() / gain
            if args.process_var:
                post1 += args.process_var
                post2 += args.process_var
            post1.clamp_(min=0.0)
            post2.clamp_(min=0.0)
            gradient_step1 = torch.where(is_kalman, kal1, gradient_step1)
            gradient_step2 = torch.where(is_kalman, kal2, gradient_step2)
        w1 -= gradient_step1
        w2 -= gradient_step2
        if args.track_levels and step % max(steps // 8, 1) == 0:
            trajectory.append((step, level1[:, :, :args.useful].mean((1, 2)).clone(),
                               level1[:, :, args.useful:].mean((1, 2)).clone()))

    with torch.no_grad():
        squared_eval = torch.zeros((n_cfg,), device=device)
        variance = torch.zeros((n_cfg,), device=device)
        for _ in range(args.eval_steps):
            x = torch.randn((dim,), device=device, generator=gen)
            clean = teacher_forward(x, args, teacher)
            act = torch.tanh(w1 @ x)
            prediction = (w2.squeeze(1) * act).sum(-1)
            squared_eval += (prediction - clean).square()
            variance += clean.square()
    useful_w = w1[:, :, :args.useful].square().mean((1, 2)).sqrt()
    junk_w = w1[:, :, args.useful:].square().mean((1, 2)).sqrt()
    return {"test_mse": (squared_eval / args.eval_steps).cpu().numpy(),
            "trivial": (variance / args.eval_steps).cpu().numpy(),
            "prequential": (squared / max(scored, 1)).cpu().numpy(),
            "useful_w": useful_w.cpu().numpy(), "junk_w": junk_w.cpu().numpy(),
            "level_useful": level1[:, :, :args.useful].mean((1, 2)).cpu().numpy(),
            "level_junk": level1[:, :, args.useful:].mean((1, 2)).cpu().numpy(),
            "post_useful": post1[:, :, :args.useful].mean((1, 2)).cpu().numpy(),
            "post_junk": post1[:, :, args.useful:].mean((1, 2)).cpu().numpy(),
            "cell_weight": (weight_sum / max(weight_count, 1)).cpu().numpy(),
            "cell_spread": (weight_spread / max(weight_count, 1)).cpu().numpy(),
            "cell_useful": (weight_sum * 0 + 0).cpu().numpy(),
            "trajectory": [(st, u.cpu().numpy(), j.cpu().numpy())
                           for st, u, j in trajectory]}


def main():
    args = tyro.cli(Args)
    methods = METHODS if args.method == "all" else tuple(args.method.split(","))
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"method must be `all` or from {METHODS}")
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda" if args.cuda else "cpu")
    configs = [{"method": m, "lr": lr} for m in methods for lr in args.lr_grid]
    started = time.perf_counter()
    out = run_all(args, configs, device)
    elapsed = time.perf_counter() - started
    ratio = out["test_mse"] / out["trivial"]
    print(f"{args.input_dim} inputs ({args.useful} useful) -> {args.hidden} hidden, "
          f"noise_std={args.noise_std}, {args.samples} samples, "
          f"batch {args.batch} ({max(args.samples // args.batch, 1)} steps)")
    print(f"{len(configs)} configs in one pass, {elapsed:.1f}s\n")
    print(f"{'method':>12s} {'lr':>7s} | {'test/zero':>9s} | {'|w| useful':>10s} "
          f"{'|w| junk':>9s} {'ratio':>6s} | {'lvl use':>7s} {'lvl junk':>8s}")
    for method in methods:
        rows = [(ratio[i], configs[i]["lr"], i) for i, c in enumerate(configs)
                if c["method"] == method]
        best, lr, index = min(rows, key=lambda row: row[0])
        grid = " ".join(f"{c_lr:g}:{c_ratio:.4f}" for c_ratio, c_lr, _index in rows)
        print(f"{method:>12s} {lr:>7g} | {best:9.5f} | {out['useful_w'][index]:10.4f} "
              f"{out['junk_w'][index]:9.5f} "
              f"{out['useful_w'][index] / max(out['junk_w'][index], 1e-12):6.1f} | "
              f"{out['level_useful'][index]:7.4f} {out['level_junk'][index]:8.4f}"
              f"\n{'':13s}grid {grid}")


if __name__ == "__main__":
    main()
