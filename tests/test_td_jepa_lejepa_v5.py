import importlib.util
from pathlib import Path

import torch

SCRIPT = Path(__file__).parents[1] / "cleanrl" / "td-jepa" / "td_jepa_lejepa_v5.py"
SPEC = importlib.util.spec_from_file_location("td_jepa_lejepa_v5", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def tiny_args(**overrides):
    values = dict(
        emb_dim=6,
        hidden_dim=16,
        sigreg_num_proj=16,
        sigreg_ref_n=8,
        gamma=0.9,
        policy_noise=0.0,
        noise_clip=0.0,
        sf_ridge=1e-8,
    )
    values.update(overrides)
    return MODULE.Args(**values)


def test_sigreg_is_on_embedding_not_a_projector():
    source = SCRIPT.read_text()
    assert "sigreg(emb.unsqueeze(0))" in source
    assert "class Agent" in source
    assert "self.projector" not in source
    args = tiny_args()
    agent = MODULE.Agent(5, 2, args)
    total, pred, sig, emb, _ = MODULE.jepa_losses(
        agent, torch.randn(10, 5), torch.randn(10, 2), torch.randn(10, 5), args
    )
    total.backward()
    assert emb.shape[-1] == args.emb_dim
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.encoder.parameters())


def test_occupancy_is_average_not_sum():
    occ = torch.ones(3, 4)
    action = torch.zeros(3, 2)
    phi = MODULE.compose_occupancy(occ, action, gamma=0.9)
    torch.testing.assert_close(phi[:, :4], 0.1 * occ)


def test_column_std_freezes_intercept_and_is_reusable():
    x = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [2.0, 4.0, 1.0],
            [4.0, 8.0, 1.0],
        ]
    )
    y, mean, std = MODULE.column_standardize(x)
    assert mean[-1].item() == 0.0
    assert std[-1].item() == 1.0
    torch.testing.assert_close(y[:, -1], torch.ones(3))
    y2, _, _ = MODULE.column_standardize(x, mean, std)
    torch.testing.assert_close(y, y2)


def test_actor_q_uses_scaled_occupancy_and_shared_standardizer():
    torch.manual_seed(0)
    args = tiny_args()
    agent = MODULE.Agent(5, 2, args)
    emb = torch.randn(8, args.emb_dim)
    action = torch.tanh(torch.randn(8, 2))
    phi_raw = MODULE.phi_immediate(emb, action)
    phi_std, mean, std = MODULE.column_standardize(phi_raw)
    w = MODULE.solve_reward_probe(phi_std, torch.randn(8), 1e-6)
    loss, q, act = MODULE.actor_objective(agent, emb, w, mean, std, args.gamma)
    loss.backward()
    assert q.ndim == 1
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.actor.parameters())
    assert all(p.grad is None for p in agent.encoder.parameters())
    # Replay-fitted std applied to (1-γ)F, not a fresh F-scale.
    occ = agent.embedding_occupancy(emb, act)
    composed = MODULE.compose_occupancy(occ, act, args.gamma)
    stdized, _, _ = MODULE.column_standardize(composed, mean, std)
    torch.testing.assert_close(q, (stdized * w).sum(-1), atol=1e-5, rtol=1e-5)
