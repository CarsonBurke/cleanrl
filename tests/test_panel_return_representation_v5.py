"""CUDA numerical contracts; run only through the main agent's mlq queue."""

import pytest
import torch

from cleanrl.plasticity.panel_return_representation_v5 import Config, GROUPS, LRS, Learner, configurations

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA contracts; queue with mlq")


def inputs(samples=7, dim=5):
    generator = torch.Generator(device="cuda").manual_seed(23)
    old = torch.randn((samples, dim), device="cuda", generator=generator)
    latest = torch.randn((samples, dim), device="cuda", generator=generator)
    memory = torch.cat((latest, torch.randn((samples, 24), device="cuda", generator=generator)), dim=1)
    signed = torch.tensor([-.8, .3, 1.7, -.1, -2.1, .9, .4], device="cuda")[:samples]
    return (old, latest, memory), signed, signed.square().clamp_max(25)


def independent_labels(target, bins=33):
    """Uniform barycentric labels, independent of the production projector."""
    coordinate = target * ((bins - 1) / 25.)
    lower = coordinate.floor().long().clamp(0, bins - 1)
    upper = (lower + 1).clamp_max(bins - 1)
    fraction = coordinate - lower
    labels = torch.nn.functional.one_hot(lower, bins) * (1. - fraction[:, None])
    return labels + torch.nn.functional.one_hot(upper, bins) * fraction[:, None]


def autograd_forward(params, x):
    w1, b1, w2, b2, wr, br = params[:6]
    h1 = torch.tanh(x @ w1.T + b1)
    h2 = torch.tanh(h1 @ w2.T + b2)
    return (h2 @ wr.T + br).squeeze(-1), h2


def restore(model, snapshot):
    with torch.no_grad():
        for actual, saved in zip(model.state_tensors(), snapshot):
            actual.copy_(saved)


def test_configuration_grid_keeps_return_and_auxiliary_searches_separate():
    configs = configurations()
    assert len(configs) == 90
    for family in GROUPS[:3]:
        assert {(c.lr, c.beta2, c.aux_weight) for c in configs if c.family == family} == {
            (lr, beta2, 0.) for lr in LRS for beta2 in (.9, .99, .999)}
    for family in GROUPS[3:]:
        assert {(c.lr, c.beta2, c.aux_weight) for c in configs if c.family == family} == {
            (lr, .999, weight) for lr in LRS for weight in (.1, 1., 10.)}


@cuda
@pytest.mark.parametrize("beta2", [.9, .99, .999])
def test_whole_manual_shared_update_matches_autograd_adam_with_missing_labels(beta2):
    frames, signed, vol = inputs()
    configs = tuple(Config(family, .001, beta2, .7 if family in GROUPS[3:] else 0.) for family in GROUPS)
    model = Learner(5, 128, configs, "cuda", num_samples=7)
    references = [[p[0].detach().clone().requires_grad_() for p in bank.parameters] for bank in model.groups]
    optimizers = [torch.optim.Adam(params, lr=.001, betas=(.9, beta2), eps=1e-8, foreach=False)
                  for params in references]
    masks = torch.tensor([[1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 1],
                          [1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0]],
                         dtype=torch.bool, device="cuda")
    saw_distinct_normalizers = False
    for step, mask in enumerate(masks):
        y = torch.where(mask, signed + .07 * step, float("nan"))
        v = torch.where(mask, vol + .11 * step, float("nan"))
        expected_forecasts = []
        for bank, params, optimizer, frame in zip(model.groups, references, optimizers, model.frame_indices):
            mean, h2 = autograd_forward(params, frames[frame])
            expected_forecasts.append(mean.detach())
            # Index valid samples rather than reproducing manual zero masking.
            loss = (mean[mask] - y[mask]).square().mean()
            if bank.auxiliary:
                source = model.permutation if bank.permuted else torch.arange(7, device="cuda")
                paired_mask = mask & mask[source]
                saw_distinct_normalizers |= bool(paired_mask.sum() != mask.sum())
                if bool(paired_mask.any()):
                    labels = independent_labels(v[source][paired_mask])
                    logits = h2[paired_mask] @ params[6].T + params[7]
                    loss = loss - .7 * (labels * logits.log_softmax(-1)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        actual = model.step(frames, y, v, mask)
        torch.testing.assert_close(actual, torch.stack(expected_forecasts), atol=4e-6, rtol=4e-6)
        for bank, params, optimizer in zip(model.groups, references, optimizers):
            for actual_parameter, expected_parameter, first, second in zip(
                    bank.parameters, params, bank.first_moments, bank.second_moments):
                torch.testing.assert_close(actual_parameter[0], expected_parameter, atol=4e-6, rtol=4e-6)
                state = optimizer.state[expected_parameter]
                torch.testing.assert_close(first[0], state["exp_avg"], atol=3e-7, rtol=5e-5)
                torch.testing.assert_close(second[0], state["exp_avg_sq"], atol=3e-8, rtol=5e-5)
    assert saw_distinct_normalizers
    assert model.candidate_finite().all()


@cuda
@torch.no_grad()
def test_zero_auxiliary_weight_reproduces_baseline_observables_without_padded_state():
    frames, signed, vol = inputs()
    configs = tuple(Config(family, .003, .99) for family in GROUPS[2:])
    model = Learner(5, 128, configs, "cuda", num_samples=7)
    for step in range(6):
        mask = torch.arange(7, device="cuda") != step
        forecast = model.step(frames, torch.where(mask, signed, float("nan")),
                              torch.where(mask, vol, float("nan")), mask)
        torch.testing.assert_close(forecast[1:], forecast[:1].expand(2, -1), rtol=0, atol=0)
        for other in model.groups[1:]:
            for baseline, auxiliary in zip(model.groups[0].parameters, other.parameters[:6]):
                torch.testing.assert_close(baseline, auxiliary, rtol=0, atol=0)
    assert len(model.groups[0].parameters) == 6
    assert len(model.groups[1].parameters) == 8


@cuda
@torch.no_grad()
def test_batched_candidate_hyperparameters_match_independent_learners():
    frames, signed, vol = inputs()
    configs = (Config("memory_aux", .001, .9, .1), Config("memory_aux", .003, .999, 10.),
               Config("latest_adam", .001, .99), Config("latest_adam", .003, .9),
               Config("memory_aux_permuted", .003, .99, 1.))
    joint = Learner(5, 128, configs, "cuda", num_samples=7)
    independent = [Learner(5, 128, (config,), "cuda", num_samples=7) for config in configs]
    for step in range(4):
        mask = torch.arange(7, device="cuda") != step
        expected = torch.cat([model.step(frames, signed, vol, mask).clone() for model in independent])
        actual = joint.step(frames, signed, vol, mask)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
        for bank, indices in zip(joint.groups, joint.indices):
            for row, column in enumerate(indices.tolist()):
                for batched, separate in zip(bank.parameters, independent[column].groups[0].parameters):
                    torch.testing.assert_close(batched[row], separate[0], atol=3e-6, rtol=3e-6)


@cuda
@torch.no_grad()
def test_hidden_initialization_and_return_output_match_across_information_sets():
    frames, signed, vol = inputs()
    configs = tuple(Config(family, .001, aux_weight=1. if family in GROUPS[3:] else 0.) for family in GROUPS)
    model = Learner(5, 128, configs, "cuda", num_samples=7)
    baseline = model.groups[0].parameters
    for bank in model.groups[1:]:
        torch.testing.assert_close(bank.parameters[0][:, :, :5], baseline[0], rtol=0, atol=0)
        for actual, expected in zip(bank.parameters[1:4], baseline[1:4]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if bank.parameters[0].shape[-1] > 5:
            assert (bank.parameters[0][:, :, 5:] == 0).all()
    torch.testing.assert_close(model.groups[-1].parameters[6], model.groups[-2].parameters[6], rtol=0, atol=0)
    torch.testing.assert_close(model.groups[-1].parameters[7], model.groups[-2].parameters[7], rtol=0, atol=0)
    forecast = model.step(frames, signed, vol, torch.ones(7, device="cuda", dtype=torch.bool))
    assert (forecast == 0).all()
    assert model.costs[3]["active_parameters"] - model.costs[2]["active_parameters"] == 33 * 129
    for cost, bank in zip(model.costs, model.groups):
        assert cost["allocated_parameters"] == sum(p.numel() for p in bank.parameters)


@cuda
@torch.no_grad()
def test_fixed_cross_sectional_permutation_and_missing_source_hold_auxiliary_optimizer():
    frames, signed, vol = inputs()
    model = Learner(5, 128, (Config("memory_aux_permuted", .001, aux_weight=1.),), "cuda", num_samples=7)
    bank = model.groups[0]
    expected = torch.randperm(7, generator=torch.Generator(device="cpu").manual_seed(1), device="cpu").cuda()
    assert torch.equal(model.permutation, expected)
    full_mask = torch.ones(7, dtype=torch.bool, device="cuda")
    model.step(frames, signed, vol, full_mask)
    frozen_aux = [t.clone() for t in (*bank.parameters[6:], *bank.first_moments[6:],
                                    *bank.second_moments[6:], bank.auxiliary_steps)]
    mask = torch.zeros_like(full_mask)
    moved = (model.permutation != torch.arange(7, device="cuda")).nonzero()[0, 0]
    mask[moved] = True
    before = [t.clone() for t in model.state_tensors()]
    first = model.step(frames, signed, vol, mask).clone()
    updated = [t.clone() for t in model.state_tensors()]
    for actual, saved in zip((*bank.parameters[6:], *bank.first_moments[6:],
                              *bank.second_moments[6:], bank.auxiliary_steps), frozen_aux):
        assert torch.equal(actual, saved)
    restore(model, before)
    # A missing SOURCE excludes its auxiliary value, even at a valid destination.
    poisoned_vol = torch.where(mask, vol, float("nan"))
    second = model.step(frames, signed, poisoned_vol, mask)
    assert torch.equal(first, second)
    for actual, saved in zip(model.state_tensors(), updated):
        assert torch.equal(actual, saved)
    assert torch.equal(model.permutation, expected)


@cuda
@torch.no_grad()
def test_empty_labels_hold_all_optimizer_state_and_failure_is_sticky():
    frames, signed, vol = inputs()
    model = Learner(5, 128, (Config("memory_aux", .001, aux_weight=1.),), "cuda", num_samples=7)
    bank = model.groups[0]
    model.step(frames, signed, vol, torch.ones(7, device="cuda", dtype=torch.bool))
    owned_optimizer = (*bank.parameters, *bank.first_moments, *bank.second_moments,
                       bank.adam_steps, bank.auxiliary_steps)
    saved = [t.clone() for t in owned_optimizer]
    model.step(frames, torch.full_like(signed, float("nan")), torch.full_like(vol, float("nan")),
               torch.zeros(7, device="cuda", dtype=torch.bool))
    for actual, expected in zip(owned_optimizer, saved):
        assert torch.equal(actual, expected)
    bank.parameters[5].fill_(float("nan"))
    forecast = model.step(frames, signed, vol, torch.zeros(7, device="cuda", dtype=torch.bool))
    assert torch.isnan(forecast).all()  # Never replace a failed return with zero or volatility.
    assert not model.candidate_finite().any()
    for actual, expected in zip(owned_optimizer, saved):
        actual.copy_(expected)
    model.step(frames, signed, vol, torch.ones(7, device="cuda", dtype=torch.bool))
    assert torch.isfinite(model.prediction).all()
    assert not model.candidate_finite().any()


@cuda
@torch.no_grad()
def test_current_labels_and_mask_do_not_change_current_forecast_after_whole_state_restore():
    frames, signed, vol = inputs()
    configs = tuple(Config(family, .001, aux_weight=1. if family in GROUPS[3:] else 0.) for family in GROUPS)
    model = Learner(5, 128, configs, "cuda", num_samples=7)
    mask = torch.ones(7, device="cuda", dtype=torch.bool)
    for _ in range(3):
        model.step(frames, signed, vol, mask)
    before = [t.clone() for t in model.state_tensors()]
    first = model.step(frames, signed, vol, mask).clone()
    trained = [p.clone() for p in model.parameters]
    restore(model, before)
    alternate_mask = mask.clone()
    alternate_mask[::2] = False
    second = model.step(frames, -signed * 3., 25. - vol, alternate_mask)
    assert torch.equal(first, second)
    assert any(not torch.equal(actual, previous) for actual, previous in zip(model.parameters, trained))


@cuda
@torch.no_grad()
def test_fullgraph_cuda_capture_replays_with_complete_state_restoration():
    frames, signed, vol = inputs()
    configs = tuple(Config(family, .001, aux_weight=1. if family in GROUPS[3:] else 0.) for family in GROUPS)
    model = Learner(5, 128, configs, "cuda", num_samples=7)
    mask = torch.tensor([True, False, True, True, False, True, True], device="cuda")
    model.step(frames, signed, vol, mask)
    before = [t.clone() for t in model.state_tensors()]
    compiled = torch.compile(model.step, fullgraph=True, mode="default")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                compiled(frames, signed, vol, mask)
        stream.synchronize()
        restore(model, before)
        with torch.cuda.graph(graph, stream=stream):
            compiled(frames, signed, vol, mask)
    finally:
        torch.cuda.synchronize()
        restore(model, before)
    for actual, saved in zip(model.state_tensors(), before):
        assert torch.equal(actual, saved)
    graph.replay()
    torch.cuda.synchronize()
    after = [t.clone() for t in model.state_tensors()]
    first_forecast = model.prediction.clone()
    restore(model, before)
    graph.replay()
    torch.cuda.synchronize()
    for actual, saved in zip(model.state_tensors(), after):
        assert torch.equal(actual, saved)
    restore(model, before)
    signed.neg_()
    vol.copy_(25. - vol)
    mask.logical_not_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(model.prediction, first_forecast)


@cuda
@torch.no_grad()
def test_auxiliary_twohot_preserves_uncentered_volatility_expectation():
    model = Learner(5, 128, (Config("memory_aux", .001, aux_weight=1.),), "cuda", num_samples=7)
    targets = torch.tensor([0., .03, .37, 1., 7.3, 24.9, 25.], device="cuda")
    labels = model.projector.project(targets)
    torch.testing.assert_close(labels.sum(-1), torch.ones_like(targets))
    torch.testing.assert_close((labels * model.support).sum(-1), targets, atol=2e-6, rtol=2e-6)
