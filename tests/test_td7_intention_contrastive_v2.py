"""Replay-future sampling, label and gradient contracts of td7_intention_contrastive_v2 (CUDA, no env)."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "cleanrl/td7/intention/td7_intention_contrastive_v2.py"
SPEC = importlib.util.spec_from_file_location("td7_intention_contrastive_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

STATE_DIM, ACTION_DIM, MAX_STEPS, BUFFER = 4, 2, 200, 1000


class NullWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, name, value, step):
        self.scalars.setdefault(name, []).append(value)


def make_agent(**overrides):
    args = MODULE.Args(buffer_size=BUFFER, compile=False, **overrides)
    return MODULE.TD7Agent(STATE_DIM, ACTION_DIM, MAX_STEPS, args, torch.device("cuda"), NullWriter())


def fill(agent, episodes, global_step=0):
    """Fake episodes: state = [episode, t, 0, 0], next_state = [episode, t + 1, ...], reward = 1000 e + t."""
    rng = np.random.default_rng(0)
    for episode, (length, terminated) in enumerate(episodes):
        for t in range(length):
            action = rng.uniform(-1, 1, ACTION_DIM).astype(np.float32)
            agent.stage(
                state=np.array([episode, t, 0, 0], np.float32), action=action, mu=action,
                sigma=np.full(ACTION_DIM, 0.1, np.float32), eps=np.zeros(ACTION_DIM, np.float32),
                next_state=np.array([episode, t + 1, 0, 0], np.float32), reward=1000.0 * episode + t,
                not_done=0.0 if (terminated and t == length - 1) else 1.0,
            )
        agent.flush(global_step, terminated)


# Lengths sum to 3110 rows in a 1000-slot ring: two wraps, an episode straddling slot 999 -> 0, the oldest
# surviving episode partly overwritten; terminated and truncated episodes alternate.
EPISODES = [(l, i % 2 == 0) for i, l in enumerate([180, 150, 200, 37, 120, 200, 64, 190, 3, 200, 140, 77,
                                                   200, 1, 160, 200, 95, 133, 200, 160, 100, 100])]


@pytest.fixture(scope="module")
def agent():
    agent = make_agent()
    fill(agent, EPISODES)
    return agent


def test_ring_holds_the_newest_rows(agent):
    assert agent.written == sum(l for l, _ in EPISODES) and agent.size == BUFFER
    abs_index = agent.row_abs.cpu().numpy()
    assert sorted(abs_index) == list(range(agent.written - BUFFER, agent.written))


def test_sampled_futures_follow_their_episode(agent):
    n, gamma, f = 400_000, agent.args.gamma, agent._fields
    csum = torch.cumsum(agent.priority, 0)
    rng_state = torch.cuda.get_rng_state()
    ind, batch, future_state, future_reward, absorbed, first_step, stats = agent._sample_futures(csum, n)
    torch.cuda.set_rng_state(rng_state)
    torch.rand(n, device="cuda")  # the row draw
    u = 1 - torch.rand(n, device="cuda")
    raw = torch.floor(torch.log(u) / math.log(gamma)).long().cpu().numpy()  # k - 1 before clipping

    batch, future_state = batch.cpu().numpy(), future_state.cpu().numpy()
    future_reward, absorbed, first_step = future_reward.cpu().numpy(), absorbed.cpu().numpy(), first_step.cpu().numpy()
    lengths = np.array([l for l, _ in EPISODES])
    terminated = np.array([term for _, term in EPISODES])
    episode, t = batch[:, f["state"]][:, 0].astype(int), batch[:, f["state"]][:, 1].astype(int)
    remaining = batch[:, f["remaining"]][:, 0].astype(int)
    np.testing.assert_array_equal(remaining, lengths[episode] - 1 - t)
    np.testing.assert_array_equal(batch[:, f["ep_terminated"]][:, 0] > 0.5, terminated[episode])

    # Offsets ~ Geometric(1 - gamma) before clipping: mean k = 1/(1-gamma), P(k = 1) = 1-gamma.
    assert abs((raw + 1).mean() - 1 / (1 - gamma)) < 1.0
    assert abs((raw == 0).mean() - (1 - gamma)) < 1e-3
    np.testing.assert_array_equal(first_step, raw == 0)

    # Same episode, never past its end, offset = min(k-1, remaining); reward is r_{t+k-1}.
    effective = np.minimum(raw, remaining)
    np.testing.assert_array_equal(future_state[:, 0].astype(int), episode)
    np.testing.assert_array_equal(future_state[:, 1].astype(int), t + 1 + effective)
    assert (future_state[:, 1] <= lengths[episode]).all()
    np.testing.assert_array_equal(future_reward, 1000.0 * episode + t + effective)
    # Only rows still in the ring are read: every future step of a surviving row is itself present.
    present = {(int(s[0]), int(s[1])) for s in agent.storage[:, f["next_state"]].cpu().numpy()}
    assert all((e, s) in present for e, s in set(zip(episode.tolist(), future_state[:, 1].astype(int).tolist())))

    # Terminal handling: past the end of a terminated episode -> ⊥; truncated -> clipped, not absorbed.
    past_end = raw > remaining
    np.testing.assert_array_equal(absorbed, past_end & terminated[episode])
    assert absorbed.any() and (past_end & ~terminated[episode]).any()
    stats = stats.cpu().numpy()
    assert stats[2] == 0.0  # no stale futures
    np.testing.assert_allclose(stats[0], effective.mean() + 1, rtol=1e-5)


def test_labels_route_absorbed_rows_to_the_absorbing_column():
    agent = make_agent(td_mix=0.5, batch_size=4)
    logits = torch.randn(2, 4, 5, device="cuda")
    target_logits = torch.randn(2, 4, 5, device="cuda")
    absorbed = torch.tensor([False, True, False, True], device="cuda")
    first_step = torch.tensor([True, False, False, False], device="cuda")
    not_done = torch.tensor([[1.0], [1.0], [0.0], [0.0]], device="cuda")
    loss, ce, acc = agent._contrastive_ce(logits, absorbed, first_step, target_logits, not_done)
    own = torch.eye(4, 5, device="cuda").bool() & ~first_step[:, None]  # k > 1: own column leaves w
    w = target_logits.masked_fill(own, float("-inf")).softmax(-1).mean(0)
    assert (w[:, :4].diagonal()[1:] == 0).all() and w[0, 0] > 0
    e = torch.eye(5, device="cuda")
    mc = torch.stack([e[0], e[4], e[2], e[4]])
    td = torch.stack([e[0], w[1], e[4], e[4]])  # k = 1 -> own column; else not_done w + done ⊥
    labels = 0.5 * mc + 0.5 * td
    torch.testing.assert_close(ce, -(labels * logits.log_softmax(-1)).sum(-1).mean())
    expected_acc = (logits.argmax(-1) == torch.tensor([0, 4, 2, 4], device="cuda")).float().mean()
    torch.testing.assert_close(acc, expected_acc)


@pytest.mark.parametrize("td_mix", [0.0, 0.5])
def test_captured_steps_run_finite(td_mix):
    agent = make_agent(td_mix=td_mix)
    fill(agent, EPISODES)
    for _ in range(600):
        agent.train()
    metrics = dict(zip(agent.metric_names, agent.metrics.tolist()))
    assert all(math.isfinite(v) for v in metrics.values())
    assert metrics["stale_futures"] == 0.0 and 1.0 <= metrics["future_offset"] <= 200
    fill(agent, [(150, True), (200, False)], global_step=10_000)  # fresh/* through the sampled bank
    assert all(math.isfinite(v[-1]) for k, v in agent.writer.scalars.items() if k.startswith("fresh/"))


def test_action_reaches_the_judge_only_through_zsa(agent):
    f = agent._fields
    batch = agent.storage[:64]
    state, action = batch[:, f["state"]], batch[:, f["action"]].clone().requires_grad_()
    future = batch[:, f["next_state"]]
    zs = agent.fixed_encoder.zs(state)
    g = agent.fixed_encoder.zsa(zs, action)
    g_leaf = g.detach().requires_grad_()
    psi_f = agent.psi(agent.fixed_encoder.zs(future), future)
    q = agent._readout(state, zs, g_leaf, psi_f, batch[:, f["reward"]].squeeze(1))[0].sum()
    (dq_dg,) = torch.autograd.grad(q, g_leaf)
    (via_zsa,) = torch.autograd.grad(g, action, dq_dg, retain_graph=True)
    q_full = agent._readout(state, zs, g, psi_f, batch[:, f["reward"]].squeeze(1))[0].sum()
    (dq_da,) = torch.autograd.grad(q_full, action)
    torch.testing.assert_close(dq_da, via_zsa)
    # The judge loss leaves the actor and the online encoder untouched.
    for p in [*agent.actor.parameters(), *agent.encoder.parameters()]:
        p.grad = None
    csum = torch.cumsum(agent.priority, 0)
    _, b, fs, fr, ab, fst, _ = agent._sample_futures(csum, agent.args.batch_size)
    loss, _, _ = agent._critic_loss(b, fs, fr, ab, fst)
    loss.backward()
    assert all(p.grad is None for p in [*agent.actor.parameters(), *agent.encoder.parameters()])
