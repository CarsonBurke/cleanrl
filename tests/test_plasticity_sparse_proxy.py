"""CUDA contract tests, not short-horizon evidence of optimizer quality; run via mlq."""

from itertools import product

import pytest
import torch

from cleanrl.plasticity import noisy_stream_diagnostic as sparse
from cleanrl.shared import runtime

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@torch.no_grad()
def test_compiled_evidence_ratio_preserves_exact_one_observation_rank_ties():
    # Values from the first divergent refresh in mlq5292: ordinary Triton
    # reciprocal division made one exact ratio 0.99999994 and another 1,
    # changing null/observed tail counts from 3/5 to 1/3.
    total = torch.tensor([[-1.1246975660324097, -2.802335023880005]], device="cuda")
    square = torch.tensor([[1.2649446725845337, 7.853081703186035]], device="cuda")
    compiled = torch.compile(sparse.evidence_ratios, fullgraph=True,
                             options={"triton.cudagraphs": False})
    observed, null = compiled(total, -total, square)
    torch.testing.assert_close(observed, torch.ones_like(total), rtol=0, atol=0)
    torch.testing.assert_close(null, observed, rtol=0, atol=0)


def test_halfway_move_has_equal_observation_counts_and_frozen_final_support():
    args = sparse.Args(input_dim=6, signal_inputs=2, steps=12, switch_at=.5, switch_to=3)
    stream = sparse.Stream(args, torch.device("cuda"), regime=False)
    support = stream.draw(args.steps, 1)[-1]
    expected_old = torch.tensor([1., 1., 0., 0., 0., 0.], device="cuda")
    expected_new = torch.tensor([0., 0., 0., 1., 1., 0.], device="cuda")
    torch.testing.assert_close(support[:6], expected_old.expand(6, -1), rtol=0, atol=0)
    torch.testing.assert_close(support[6:], expected_new.expand(6, -1), rtol=0, atol=0)
    frozen = stream.draw(2, 0, frozen=True)[-1]
    torch.testing.assert_close(frozen, expected_new.expand(2, -1), rtol=0, atol=0)


@pytest.mark.parametrize("switch_to", [1, 3])
def test_multi_input_support_moves_returns_and_partitions_gate_evidence(switch_to):
    args = sparse.Args(input_dim=6, signal_inputs=2, seeds=2, steps=12,
                       switch_at=0.25, switch_back=0.75, switch_to=switch_to)
    stream = sparse.Stream(args, torch.device("cuda"), regime=False)
    times = torch.tensor([3, 4, 9, 10, 12], device="cuda")
    observed = stream.support(times)
    expected_indices = [[0, 1], [switch_to, switch_to + 1],
                        [switch_to, switch_to + 1], [0, 1], [0, 1]]
    # One-based training: three old observations, six moved, three returned.
    schedule = stream.support(torch.arange(1, args.steps + 1, device="cuda"))
    assert schedule[:, 0].sum().item() == 6
    assert schedule[:, switch_to + 1].sum().item() == 6
    assert schedule.sum(-1).eq(args.signal_inputs).all().item()
    frozen = stream.draw(2, 0, frozen=True)[-1]
    torch.testing.assert_close(frozen, schedule[-1].expand(2, -1), rtol=0, atol=0)
    raw = torch.tensor([[1., 2., 4., 8., 16., 32.],
                        [3., 5., 7., 11., 13., 17.]], device="cuda")
    level = raw / 7
    active = torch.tensor([[1., 0., 1., 1., 0., 1.],
                           [1., 1., 0., 1., 1., 0.]], device="cuda")
    noisy = torch.tensor([0., 1.], device="cuda")
    for support, indices in zip(observed, expected_indices):
        expected = torch.zeros(6, device="cuda")
        expected[indices] = 1
        torch.testing.assert_close(support, expected, rtol=0, atol=0)
        # Score actual gate evidence, not just a mask helper: current support
        # must never leak into distractor numerators or denominators on a return.
        gate = sparse.Gate("energy", raw.shape, args, torch.device("cuda"))
        gate.record(raw, level, active, noisy, support)
        distractors = [index for index in range(6) if index not in indices]
        signal_active = active[:, indices]
        distractor_active = active[:, distractors]
        expected_acc = torch.stack([
            (raw[:, indices] * signal_active).sum(), signal_active.sum(),
            (raw[:, distractors] * distractor_active).sum(), distractor_active.sum(),
            (level[:, indices] * signal_active).sum(),
            (level[:, distractors] * distractor_active).sum(),
            (level[0, indices] * signal_active[0]).sum(), signal_active[0].sum(),
            (level[1, indices] * signal_active[1]).sum(), signal_active[1].sum(),
        ])
        torch.testing.assert_close(gate.acc[:10], expected_acc)


@pytest.mark.parametrize("indices", [[0, 1], [1, 2]])
def test_exact_clean_risk_matches_exhaustive_bernoulli_outcomes(indices):
    # The final coordinate is an exposed regime flag, hence p=.5 rather than
    # feature_prob. Signed, nonzero-mean errors make the cross term indispensable.
    p = torch.tensor([.2, .2, .2, .2, .5], device="cuda", dtype=torch.float64)
    x = torch.tensor(list(product((0., 1.), repeat=5)), device="cuda", dtype=torch.float64)
    mass = torch.where(x.bool(), p, 1 - p).prod(-1)
    support = torch.zeros(5, device="cuda", dtype=torch.float64)
    support[indices] = 1
    weight = torch.tensor([[.25, .6, -.3, .7, -.4],
                           [1.2, -.2, .5, -.8, .9]], device="cuda", dtype=torch.float64)
    clean = x @ support
    predictions = x @ weight.T
    signal_error = x[:, indices] @ (weight[:, indices] - 1).T
    distractors = [index for index in range(5) if index not in indices]
    leakage = x[:, distractors] @ weight[:, distractors].T
    risk = sparse.clean_risk(weight, support, p)
    expected = {
        "exact_mse": (mass[:, None] * (predictions - clean[:, None]).square()).sum(0),
        "signal_reconstruction_mse": (mass[:, None] * signal_error.square()).sum(0),
        "distractor_leakage_mse": (mass[:, None] * leakage.square()).sum(0),
        "signal_distractor_cross": (mass[:, None] * 2 * signal_error * leakage).sum(0),
        "exact_trivial": (mass * clean.square()).sum().expand(2),
        "exact_mean_predictor_mse": (mass * (clean - (mass * clean).sum()).square()).sum().expand(2),
    }
    for name, value in expected.items():
        torch.testing.assert_close(risk[name], value, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(risk["exact_mse"], risk["signal_reconstruction_mse"]
                               + risk["distractor_leakage_mse"] + risk["signal_distractor_cross"],
                               rtol=1e-13, atol=1e-14)
    assert (risk["signal_distractor_cross"].abs() > 1e-3).all().item()


@torch.no_grad()
def reference_trajectory(args, method, samples):
    """Independent uncaptured recurrence for consumer-visible loss and predictions."""
    xs, ys, _, flags, supports = samples
    weight = torch.zeros((args.seeds, args.input_dim), device="cuda")
    first, second = torch.zeros_like(weight), torch.zeros_like(weight)
    gate = sparse.Gate(method, weight.shape, args, torch.device("cuda")) \
        if method in {"energy", "snr", "statewiener"} else None
    control = torch.Generator(device="cuda").manual_seed(args.seed + 3_000_017)
    signs = torch.cat([torch.where(torch.rand((min(args.chunk_steps, args.steps - start),
                                             args.seeds, 1), device="cuda", generator=control) < .5,
                                  -1., 1.)
                       for start in range(0, args.steps, args.chunk_steps)])
    post_w = torch.zeros_like(weight)
    post_var = torch.full_like(weight, args.bayes_prior)
    logit = torch.full_like(weight, args.bayes_logit0)
    resid_var = torch.ones((args.seeds, 1), device="cuda")
    noise_sum = torch.zeros_like(resid_var)
    error = torch.zeros(args.seeds, device="cuda")
    evidence_sum = torch.zeros_like(weight)
    evidence_sq = torch.zeros_like(weight)
    twin_sum = torch.zeros_like(weight)
    level_buf = torch.zeros_like(weight)
    for step, (x, y, noisy, support, sign) in enumerate(zip(xs, ys, flags, supports, signs), 1):
        if method == "oracle":
            # Independent out-of-place masking preserves overlapping coordinates,
            # but forgets both moments when an input leaves and later returns.
            weight = weight * support
            first = first * support
            second = second * support
        prediction = (weight * x).sum(-1)
        error = error + (prediction - (x * support).sum(-1)).square()
        delta = (y - prediction).unsqueeze(-1)
        grad = -delta * x
        if method == "sgd":
            weight = weight - args.lr * grad
            continue
        if method == "bayes":
            contribution = post_w * x
            err_out = delta + logit.sigmoid() * contribution
            err_in = err_out - contribution
            logit = (logit + (err_out.square() - err_in.square()) / (2 * resid_var)).clamp(
                -args.bayes_logit_cap, args.bayes_logit_cap)
            gain = post_var * x / ((post_var * x.square()).sum(-1, keepdim=True) + resid_var)
            post_w = post_w + gain * delta
            post_var = (post_var - gain * x * post_var).clamp_min(1e-8) + args.bayes_q * args.bayes_prior
            weight = logit.sigmoid() * post_w
            noise_sum = noise_sum + delta.square()
            resid_var = (noise_sum / step).clamp_min(1e-12)
            continue
        first = .9 * first + .1 * grad
        second = .999 * second + .001 * grad.square()
        update = args.lr * (first / (1 - .9 ** step)) / ((second / (1 - .999 ** step)).sqrt() + 1e-5)
        if method == "oracle":
            update = update * support
        elif method in {"mirror", "softveto", "smoothgate", "softhinge", "softhinge_amp"}:
            t_sq = evidence_sum.square() / evidence_sq.clamp_min(1e-30)
            twin_sq = twin_sum.square() / evidence_sq.clamp_min(1e-30)
            if method == "mirror":
                if (step - 1) % args.gate_every == 0:
                    observed = t_sq.sqrt()
                    false_ge = (twin_sq.sqrt()[:, None, :] >= observed[:, :, None]).sum(-1)
                    total_ge = (observed[:, None, :] >= observed[:, :, None]).sum(-1).clamp_min(1)
                    level_buf = (1 - false_ge.float() / total_ge).clamp(0, 1)
            else:
                z_sq = torch.quantile(twin_sq, args.adaptive_q, dim=1, keepdim=True).clamp_min(1) \
                    if args.adaptive_z else args.veto_z ** 2
                if method in {"softveto", "smoothgate"}:
                    power = args.soft_exponent if method == "softveto" else args.gate_power
                    level_buf = (t_sq / (t_sq + z_sq)).pow(power)
                else:
                    raw = 1 - z_sq / t_sq.clamp_min(1e-30)
                    level_buf = torch.nn.functional.softplus(args.hinge_sharpness * raw) / args.hinge_sharpness
                    if method == "softhinge_amp":
                        level_buf = level_buf / (level_buf.mean(-1, keepdim=True) + args.level_floor)
            update = update * level_buf
            evidence = grad / resid_var.clamp_min(1e-8) \
                if method == "mirror" and args.gls_evidence else grad
            keep = 1 - args.evidence_decay * (x != 0).float()
            evidence_sum = keep * evidence_sum + evidence
            evidence_sq = keep.square() * evidence_sq + evidence.square()
            twin_sum = keep * twin_sum + evidence * sign
            if method == "mirror" and args.gls_evidence:
                resid_var = .99 * resid_var + .01 * delta.square()
        elif gate is not None:
            active = (x != 0).float()
            state = torch.stack((torch.ones_like(x), noisy[:, None].expand_as(x)), -1)
            raw = gate.statistic(grad, active, state, sign)
            level = gate.level(raw, active)
            gate.record(raw, level, active, noisy, support)
            update = update * level
        if method == "adamw":
            weight = weight * (1 - args.lr * args.weight_decay)
        weight = weight - update
    return weight, error / args.steps, gate


@pytest.mark.parametrize("method", sparse.METHODS)
@torch.no_grad()
def test_compiled_run_preserves_common_data_exact_tail_and_initial_state(method, monkeypatch):
    runtime.configure_runtime(matmul_precision="highest", allow_tf32=False)
    args = sparse.Args(task="regime", input_dim=6, signal_inputs=2, regime_index=5,
                       feature_prob=.6, quiet_std=.35, noisy_std=.8, seeds=2, seed=1,
                       steps=11, chunk_steps=4, gate_every=2, switch_at=.3,
                       switch_back=.75, switch_to=1, lr=.09, stat_beta=.6,
                       gate_lr=.08, bayes_logit0=-.5, adaptive_z=True,
                       gls_evidence=True, evidence_decay=.1, eval_steps=13, eval_batch_size=5,
                       plot="capture-only", plot_window=13)
    device = torch.device("cuda")
    original_draw = sparse.Stream.draw
    observed = {False: [], True: []}

    def record_draw(self, count, start, *, frozen=False):
        # Observe REAL production samples without replacing sampling or updates.
        result = original_draw(self, count, start, frozen=frozen)
        observed[frozen].append(tuple(value.clone() for value in result))
        return result

    # Match CLI arm isolation: otherwise independent method closures exhaust
    # Dynamo's per-code guard limit rather than exercising their CUDA graphs.
    torch.compiler.reset()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(sparse.Stream, "draw", record_draw)
            out = sparse.run(args, method, device)
    finally:
        torch.compiler.reset()

    common = sparse.Stream(args, device, regime=True)
    expected_blocks = [common.draw(min(args.chunk_steps, args.steps - start + 1), start)
                       for start in range(1, args.steps + 1, args.chunk_steps)]
    assert len(observed[False]) == len(expected_blocks)
    for actual, expected in zip(observed[False], expected_blocks):
        for value, common_value in zip(actual, expected):
            torch.testing.assert_close(value, common_value, rtol=0, atol=0)
    # Twin sign draws must not perturb features, noise, flags, or held-out data.
    heldout = sparse.Stream(args, device, regime=True, seed_offset=10_000_019)
    expected_test = [heldout.draw(min(args.eval_batch_size, args.eval_steps - start), 0, frozen=True)
                     for start in range(0, args.eval_steps, args.eval_batch_size)]
    assert len(observed[True]) == len(expected_test)
    for actual, expected in zip(observed[True], expected_test):
        for value, common_value in zip(actual, expected):
            torch.testing.assert_close(value, common_value, rtol=0, atol=0)
    assert out["optimizer_steps"].item() == args.steps
    assert out["finite_weights"].all().item()
    torch.testing.assert_close(out["exact_mse"], out["signal_reconstruction_mse"]
                               + out["distractor_leakage_mse"] + out["signal_distractor_cross"],
                               rtol=1e-12, atol=1e-12)

    samples = tuple(torch.cat(values) for values in zip(*expected_blocks))
    test_samples = tuple(torch.cat(values) for values in zip(*expected_test))
    weight, prequential, gate = reference_trajectory(args, method, samples)
    prediction = (test_samples[0] * weight).sum(-1)
    expected_mse = (prediction.double() - test_samples[2]).square().mean(0)
    # Predictions and prequential loss expose wrong warmup restoration,
    # stale moments, attribute rebinding, and padded tail optimizer steps.
    torch.testing.assert_close(out["trace"][0], prediction[:, 0], rtol=3e-4, atol=3e-6)
    torch.testing.assert_close(out["test_mse"], expected_mse, rtol=3e-4, atol=3e-6)
    torch.testing.assert_close(out["prequential_clean_mse"], prequential, rtol=3e-4, atol=3e-6)
    p = torch.full((args.input_dim,), args.feature_prob, device=device)
    p[args.regime_index] = .5
    expected_risk = sparse.clean_risk(weight, samples[-1][-1], p)
    for name, expected in expected_risk.items():
        torch.testing.assert_close(out[name], expected, rtol=3e-4, atol=3e-6)
    if gate is not None:
        torch.testing.assert_close(out["acc"], gate.acc, rtol=3e-4, atol=3e-6)
    if method == "oracle":
        assert out["stale"].eq(0).all().item()
        assert out["distract"].eq(0).all().item()
        torch.testing.assert_close(out["exact_ablated_mse"], out["exact_mse"], rtol=0, atol=0)
