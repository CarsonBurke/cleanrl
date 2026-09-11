"""Independent mathematical contracts; CUDA execution must be queued with mlq."""

import pytest
import torch

from cleanrl.plasticity.panel_return_hierarchy_v7 import Config, GROUPS, Learner, configurations

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; queue via mlq")


def small_configs():
    return (Config(GROUPS[0], 2.), Config(GROUPS[0], 11.),
            Config(GROUPS[1], 2.), Config(GROUPS[1], 7.),
            Config(GROUPS[2], 2., 3.), Config(GROUPS[2], 2., 19.),
            Config(GROUPS[2], 7., 3.), Config(GROUPS[2], 7., 19.))


def joint_mean(history, stocks, features, local, global_):
    """Build the full joint precision in [global, stock0, ...] coordinates.

    This oracle does not use a Schur complement or the online update equations.
    """
    eye = torch.eye(features, dtype=torch.float64, device="cuda")
    precision = torch.zeros(((stocks + 1) * features,) * 2, dtype=torch.float64, device="cuda")
    rhs = torch.zeros((stocks + 1) * features, dtype=torch.float64, device="cuda")
    precision[:features, :features] = (global_ + stocks * local) * eye
    for stock in range(stocks):
        part = slice((stock + 1) * features, (stock + 2) * features)
        precision[part, part] = local * eye
        precision[:features, part] = -local * eye
        precision[part, :features] = -local * eye
        for x, y, mask in history:
            if bool(mask[stock]):
                precision[part, part] += torch.outer(x[stock], x[stock])
                rhs[part] += x[stock] * y[stock]
    return torch.linalg.solve(precision, rhs).reshape(stocks + 1, features)[1:]


def batch_mean(history, stocks, features, config):
    if config.family == GROUPS[2]:
        return joint_mean(history, stocks, features, config.lambda_local, config.lambda_global)
    eye = torch.eye(features, dtype=torch.float64, device="cuda")
    grams = config.lambda_local * eye.expand(stocks, -1, -1).clone()
    rhs = torch.zeros((stocks, features), dtype=torch.float64, device="cuda")
    for x, y, mask in history:
        for stock in range(stocks):
            if bool(mask[stock]):
                grams[stock] += torch.outer(x[stock], x[stock])
                rhs[stock] += x[stock] * y[stock]
    if config.family == GROUPS[0]:
        gram = grams.sum(0) - (stocks - 1) * config.lambda_local * eye
        return torch.linalg.solve(gram, rhs.sum(0)).expand(stocks, -1)
    return torch.linalg.solve(grams, rhs.unsqueeze(-1)).squeeze(-1)


def test_search_has_64_genuinely_distinct_priors_per_family():
    configs = configurations()
    for group in GROUPS:
        values = [c for c in configs if c.family == group]
        assert len(values) == len(set(values)) == 64
    hierarchical = [c for c in configs if c.family == GROUPS[2]]
    assert {(c.lambda_local, c.lambda_global) for c in hierarchical} == {
        (10. ** i, 10. ** j) for i in range(8) for j in range(8)}
    for group in GROUPS[:2]:
        priors = {c.lambda_local for c in configs if c.family == group}
        assert {10. ** i for i in range(8)} <= priors
        assert min(priors) == .01 and max(priors) == 1e10


@cuda
@torch.no_grad()
def test_preupdate_predictions_match_independent_full_joint_and_batch_ridge():
    generator = torch.Generator(device="cuda").manual_seed(19)
    stocks, features = 3, 4
    configs = small_configs()
    learner = Learner(stocks, features, configs)
    history = []
    for bar in range(10):
        x = torch.randn((stocks, features), generator=generator, device="cuda", dtype=torch.float64)
        y = torch.randn(stocks, generator=generator, device="cuda", dtype=torch.float64)
        mask = torch.tensor([bar % 3 != 0, bar % 4 != 0, bar < 5], device="cuda")
        expected = torch.stack([(batch_mean(history, stocks, features, c) * x).sum(-1) for c in configs])
        actual = learner.step(x, torch.where(mask, y, float("nan")), mask).clone()
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
        history.append((x, y, mask))
    assert bool(learner.candidate_finite().all())


@cuda
@torch.no_grad()
def test_missing_label_cannot_teach_other_stocks_and_current_labels_cannot_change_forecast():
    configs = small_configs()
    left, right = Learner(3, 2, configs), Learner(3, 2, configs)
    x = torch.tensor([[1., 2.], [-1., 3.], [2., -.5]], dtype=torch.float64, device="cuda")
    mask = torch.tensor([True, False, True], device="cuda")
    y = torch.tensor([.2, float("nan"), -.7], dtype=torch.float64, device="cuda")
    changed = y.clone()
    changed[1] = 1e100
    first = left.step(x, y, mask).clone()
    second = right.step(x, changed, mask).clone()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(left.predict(x)[0], right.predict(x)[0], rtol=0, atol=0)
    # A fully missing bar leaves every future posterior prediction unchanged.
    before = left.predict(x)[0].clone()
    left.step(x, torch.full_like(y, float("nan")), torch.zeros_like(mask))
    torch.testing.assert_close(left.predict(x)[0], before, rtol=0, atol=0)
    # Observed labels teach other stocks only AFTER the returned forecast.
    before = right.predict(x)[0].clone()
    forecast = right.step(x, torch.tensor([8., 4., -3.], device="cuda"), torch.ones_like(mask)).clone()
    torch.testing.assert_close(forecast, before, rtol=0, atol=0)
    assert not torch.equal(right.predict(x)[0], before)


@cuda
@torch.no_grad()
def test_unobserved_stock_borrows_global_information_but_independent_control_does_not():
    configs = (Config(GROUPS[0], 2.), Config(GROUPS[1], 2.), Config(GROUPS[2], 2., 3.))
    model = Learner(2, 1, configs)
    x = torch.ones((2, 1), device="cuda", dtype=torch.float64)
    model.step(x, torch.tensor([4., float("nan")], device="cuda"), torch.tensor([True, False], device="cuda"))
    prediction = model.predict(x)[0]
    assert prediction[1, 1].item() == 0.
    torch.testing.assert_close(prediction[2, 1], torch.tensor(4. / 5.5, device="cuda", dtype=torch.float64))
    assert prediction[2, 0] > prediction[2, 1] > 0


@cuda
@torch.no_grad()
def test_solver_and_finiteness_failures_stay_failed_after_finite_predictions_return():
    configs = (Config(GROUPS[0], 2.), Config(GROUPS[1], 2.), Config(GROUPS[2], 2., 3.))
    learner = Learner(2, 2, configs)
    x = torch.ones((2, 2), dtype=torch.float64, device="cuda")
    learner.gram.copy_(-2. * torch.eye(2, device="cuda", dtype=torch.float64))
    learner.predict(x)
    assert learner.solver_failures[0].item() == 1
    assert not bool(learner.candidate_finite()[0])
    learner.gram.zero_()
    assert bool(torch.isfinite(learner.predict(x)[0]).all())
    assert not bool(learner.candidate_finite()[0])
    learner.step(x, torch.tensor([float("nan"), 1.], device="cuda"), torch.ones(2, dtype=torch.bool, device="cuda"))
    assert not bool(learner.candidate_finite().any())


@cuda
@torch.no_grad()
def test_natural_statistic_audit_detects_recursive_drift_without_repairing_it():
    configs = small_configs()
    model = Learner(3, 2, configs)
    x = torch.tensor([[1., 2.], [-1., 3.], [2., -.5]], dtype=torch.float64, device="cuda")
    y = torch.tensor([.2, .4, -.7], dtype=torch.float64, device="cuda")
    mask = torch.tensor([True, False, True], device="cuda")
    history = []
    for step in range(5):
        label = y + step / 10.
        model.step(x, label, mask)
        history.append((x, label, mask))
    for config in configs:
        audit = model.audit(config)
        assert audit["status"] == "agreement"
        batch = torch.tensor(audit["batch_coefficients"], device="cuda", dtype=torch.float64)
        if config.family == GROUPS[2]:
            batch = batch[1:]
        if config.family == GROUPS[0]:
            batch = batch.expand(3, -1)
        torch.testing.assert_close(batch, batch_mean(history, 3, 2, config), rtol=1e-11, atol=1e-12)
    before = model.audit(configs[4])["batch_coefficients"]
    model.m[0, 0, 0].add_(.1)
    corrupted = model.m.clone()
    audit = model.audit(configs[4])
    assert audit["status"] == "disagreement"
    assert audit["batch_coefficients"] == before
    assert audit["normal_equation_residual_max_abs"] > .1
    torch.testing.assert_close(model.m, corrupted, rtol=0, atol=0)
