import importlib.util
from pathlib import Path

import torch

SCRIPT = Path(__file__).parents[1] / "cleanrl" / "td-jepa" / "td_jepa_lejepa_v6.py"
SPEC = importlib.util.spec_from_file_location("td_jepa_lejepa_v6", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def tiny_args(**overrides):
    values = dict(emb_dim=6, hidden_dim=16, sigreg_num_proj=16, sigreg_ref_n=8, gamma=0.9, sf_ridge=1e-8)
    values.update(overrides)
    return MODULE.Args(**values)


def test_no_learned_occupancy_head():
    source = SCRIPT.read_text()
    assert "sf_head" not in source
    assert "embedding_occupancy" not in source
    assert "predicted_occupancy" in source
    agent = MODULE.Agent(5, 2, tiny_args())
    assert not hasattr(agent, "sf_head")


def test_measured_occupancy_is_discounted_sum_of_embeddings():
    e = torch.zeros(3, 1, 2)
    e[0, 0] = torch.tensor([1.0, 0.0])
    e[1, 0] = torch.tensor([0.0, 1.0])
    e[2, 0] = torch.tensor([2.0, 2.0])
    term = torch.zeros(3, 1)
    trunc = torch.zeros(3, 1)
    boot = torch.zeros(3, 1, 2)
    tail = torch.tensor([[4.0, 0.0]])
    occ = MODULE.successor_features(e, term, trunc, boot, tail, gamma=0.5)
    torch.testing.assert_close(occ[2, 0], e[2, 0] + 0.5 * tail[0])
    torch.testing.assert_close(occ[0, 0], e[0, 0] + 0.5 * occ[1, 0])


def test_occupancy_map_recovers_linear_successor():
    torch.manual_seed(0)
    e = torch.randn(200, 5)
    # Λ = e (I + 0.5 A) roughly linear
    a = torch.randn(5, 5) * 0.1
    lam = e + e @ a
    w = MODULE.solve_ridge(e, lam, 1e-6)
    pred = e @ w
    assert MODULE.ev_score(pred, lam) > 0.99


def test_actor_paths_through_predictor_not_encoder():
    torch.manual_seed(1)
    args = tiny_args()
    agent = MODULE.Agent(5, 2, args)
    emb = torch.randn(8, args.emb_dim)
    w_map = torch.eye(args.emb_dim)
    feat = torch.randn(8, args.emb_dim + 5)
    feat[:, -1] = 1.0
    phi, mean, std = MODULE.column_standardize(feat)
    w_r = MODULE.solve_ridge(phi, torch.randn(8), 1e-6).squeeze(-1)
    loss, q, action = MODULE.actor_objective(agent, emb, w_map, w_r, mean, std, args.gamma)
    loss.backward()
    assert q.ndim == 1
    assert action.requires_grad or any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.actor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.actor.parameters())
    assert all(p.grad is None for p in agent.encoder.parameters())
    # Predictor may receive grad through T(e,â) but is not stepped in the actor opt.
    assert any(p.grad is not None for p in agent.predictor.parameters())
