import importlib.util
from pathlib import Path

import torch

SCRIPT = Path(__file__).parents[1] / "cleanrl" / "td-jepa" / "td_jepa_lejepa_v3.py"
SPEC = importlib.util.spec_from_file_location("td_jepa_lejepa_v3", SCRIPT)
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


def test_compose_successor_keeps_action_quadratic():
    occ = torch.randn(4, 6)
    action = torch.randn(4, 2)
    phi = MODULE.compose_successor(occ, action)
    assert phi.shape == (4, 6 + 2 + 2 + 1)
    torch.testing.assert_close(phi[:, :6], occ)
    torch.testing.assert_close(phi[:, 6:8], action)
    torch.testing.assert_close(phi[:, 8:10], action * action)


def test_reward_probe_reads_action_cost_from_composed_features():
    torch.manual_seed(0)
    occ = torch.randn(128, 4)
    action = torch.randn(128, 2)
    phi = MODULE.compose_successor(occ, action)
    reward = 0.4 * occ[:, 0] - 0.1 * (action * action).sum(-1) + 0.25
    w = MODULE.solve_reward_probe(phi, reward, 1e-8)
    assert MODULE.ev_score(phi @ w, reward) > 0.99


def test_sf_td_target_is_next_embedding_plus_discounted_occupancy():
    torch.manual_seed(1)
    args = tiny_args()
    agent = MODULE.Agent(5, 2, args)
    target = MODULE.Agent(5, 2, args)
    target.load_state_dict(agent.state_dict())
    emb = torch.randn(8, 6)
    action = torch.tanh(torch.randn(8, 2))
    next_emb = torch.randn(8, 6)
    dones = torch.zeros(8, 1)
    dones[:2] = 1.0
    loss, pred, tgt = MODULE.sf_td_loss(agent, target, emb, action, next_emb, dones, args)
    assert pred.shape == (8, 6)
    assert torch.isfinite(loss)
    with torch.no_grad():
        next_a = target.act(next_emb)
        expected = next_emb + args.gamma * (1.0 - dones) * target.embedding_occupancy(next_emb, next_a)
    torch.testing.assert_close(tgt, expected)
    torch.testing.assert_close(tgt[:2], next_emb[:2])


def test_actor_path_uses_vector_successor_not_scalar_gae():
    source = SCRIPT.read_text()
    assert "compute_gae" not in source
    assert "pg_loss" not in source
    assert "L_π = - w_r · Λ̂" in source
    assert "actor_objective" in source


def test_actor_objective_is_pathwise_through_action_and_not_encoder():
    torch.manual_seed(2)
    args = tiny_args()
    agent = MODULE.Agent(5, 2, args)
    obs = torch.randn(8, 5)
    emb = agent.encode(obs).detach().requires_grad_(True)
    w_r = torch.randn(args.emb_dim + 2 * 2 + 1)
    loss, q, action = MODULE.actor_objective(agent, emb, w_r)
    loss.backward()
    assert action.requires_grad
    assert q.ndim == 1
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.actor.parameters())
    assert all(p.grad is None for p in agent.encoder.parameters())
    # Immediate a⊙a block is in the objective, so action grads are nonzero.
    assert emb.grad is not None
