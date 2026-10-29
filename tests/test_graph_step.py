import pytest
import torch

from cleanrl.shared.graph_step import capture_steps

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def _setup():
    torch.manual_seed(0)
    a = torch.nn.Linear(4, 4).cuda()
    b = torch.nn.Linear(4, 1).cuda()
    opt_a = torch.optim.Adam(a.parameters(), lr=1e-2, fused=True)
    opt_b = torch.optim.Adam(b.parameters(), lr=1e-2, fused=True)
    counter = torch.zeros((), device="cuda")
    data = torch.randn(8, 4, device="cuda")
    return a, b, opt_a, opt_b, counter, data


def _make_steps(a, b, opt_a, opt_b, counter, data, noise=None):
    def step_a():
        opt_a.zero_grad(set_to_none=False)
        a(data).square().mean().backward()
        opt_a.step()
        counter.add_(1)

    def step_b():
        opt_b.zero_grad(set_to_none=False)
        x = data if noise is None else data + noise()
        b(a(x).detach()).square().mean().backward()
        opt_b.step()
        counter.add_(10)

    return step_a, step_b


def test_warmup_is_invisible_and_replay_matches_eager():
    a, b, opt_a, opt_b, counter, data = _setup()
    before = [p.detach().clone() for p in [*a.parameters(), *b.parameters()]]
    graphs = capture_steps(list(_make_steps(a, b, opt_a, opt_b, counter, data)), optimizers=[opt_a, opt_b],
                           state_tensors=[counter])
    torch.cuda.synchronize()
    for p, saved in zip([*a.parameters(), *b.parameters()], before):
        assert torch.equal(p, saved)
    assert counter.item() == 0
    assert all(float(s["step"]) == 0 for o in (opt_a, opt_b) for s in o.state.values())

    # Reference: identical modules updated eagerly by the same schedule.
    ra, rb, ropt_a, ropt_b, rcounter, rdata = _setup()
    ref_a, ref_b = _make_steps(ra, rb, ropt_a, ropt_b, rcounter, rdata)
    for _ in range(3):
        graphs[0].replay()
        graphs[1].replay()
        ref_a()
        ref_b()
    torch.cuda.synchronize()
    assert counter.item() == rcounter.item() == 33
    for p, r in zip([*a.parameters(), *b.parameters()], [*ra.parameters(), *rb.parameters()]):
        torch.testing.assert_close(p, r, rtol=1e-5, atol=1e-6)


def test_replays_draw_fresh_randomness():
    a, b, opt_a, opt_b, counter, data = _setup()
    draws = torch.zeros(2, 8, 4, device="cuda")
    slot = torch.zeros((), dtype=torch.long, device="cuda")

    def noise():
        x = torch.randn(8, 4, device="cuda")
        draws.index_copy_(0, slot.view(1), x[None])
        return x

    step_a, step_b = _make_steps(a, b, opt_a, opt_b, counter, data, noise)
    (graph,) = capture_steps([step_b], optimizers=[opt_b], state_tensors=[counter, draws])
    graph.replay()
    slot.fill_(1)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(draws[0], draws[1])


def test_rejects_unfused_optimizer():
    layer = torch.nn.Linear(2, 2).cuda()
    with pytest.raises(ValueError, match="fused=True"):
        capture_steps([lambda: None], optimizers=[torch.optim.Adam(layer.parameters())])
