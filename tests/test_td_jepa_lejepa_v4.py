import importlib.util
from pathlib import Path

import torch

SCRIPT = Path(__file__).parents[1] / "cleanrl" / "td-jepa" / "td_jepa_lejepa_v4.py"
SPEC = importlib.util.spec_from_file_location("td_jepa_lejepa_v4", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def tiny_args(**overrides):
    values = dict(
        emb_dim=8,
        proj_dim=4,
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


def test_sigreg_is_on_projector_not_embedding():
    source = SCRIPT.read_text()
    assert "agent.project(emb)" in source
    assert "sigreg(emb.unsqueeze(0))" not in source
    torch.manual_seed(0)
    args = tiny_args()
    agent = MODULE.Agent(5, 2, args)
    obs = torch.randn(12, 5)
    action = torch.randn(12, 2)
    next_obs = torch.randn(12, 5)
    total, pred, sig, emb, next_emb, projected = MODULE.jepa_losses(agent, obs, action, next_obs, args)
    assert projected.shape == (12, args.proj_dim)
    assert emb.shape == (12, args.emb_dim)
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.projector.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.encoder.parameters())


def test_reward_probe_can_use_unwhitened_embedding_scale():
    torch.manual_seed(1)
    e = torch.randn(200, 6) * 4.0 + 2.0
    a = torch.randn(200, 2)
    phi = MODULE.compose_successor(e, a)
    reward = 1.7 * e[:, 0] - 0.1 * (a * a).sum(-1)
    w = MODULE.solve_reward_probe(phi, reward, 1e-8)
    assert MODULE.ev_score(phi @ w, reward) > 0.99


def test_actor_is_pathwise_vector_successor():
    source = SCRIPT.read_text()
    assert "compute_gae" not in source
    assert "pg_loss" not in source
    torch.manual_seed(2)
    args = tiny_args()
    agent = MODULE.Agent(5, 2, args)
    emb = torch.randn(8, args.emb_dim)
    w_r = torch.randn(args.emb_dim + 5)
    loss, q, action = MODULE.actor_objective(agent, emb, w_r)
    loss.backward()
    assert q.ndim == 1
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.actor.parameters())
    assert all(p.grad is None for p in agent.encoder.parameters())
    assert all(p.grad is None for p in agent.projector.parameters())


def test_obs_rms_is_current_stats_not_wrapper():
    rms = MODULE.RunningMeanStd(3)
    x = torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    rms.update(x)
    y = rms.normalize(x)
    assert torch.isfinite(y).all()
    assert y.mean().abs() < 1e-4
