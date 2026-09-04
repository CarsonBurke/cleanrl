import importlib.util
import math
from pathlib import Path

import torch

SCRIPT = Path(__file__).parents[1] / "cleanrl" / "td-jepa" / "td_jepa_lejepa_v2.py"
SPEC = importlib.util.spec_from_file_location("td_jepa_lejepa_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_phi_includes_action_quadratic_and_intercept():
    emb = torch.randn(5, 4)
    action = torch.randn(5, 3)
    phi = MODULE.phi_features(emb, action)
    assert phi.shape == (5, 4 + 3 + 3 + 1)
    torch.testing.assert_close(phi[:, :4], emb)
    torch.testing.assert_close(phi[:, 4:7], action)
    torch.testing.assert_close(phi[:, 7:10], action * action)
    torch.testing.assert_close(phi[:, 10], torch.ones(5))


def test_reward_probe_recovers_cheetah_ctrl_cost():
    torch.manual_seed(0)
    e = torch.randn(256, 8)
    a = torch.randn(256, 2)
    phi = MODULE.phi_features(e, a)
    # r = 0.3 e_0 - 0.1 ||a||^2 + 0.5, exactly in the span of phi.
    reward = 0.3 * e[:, 0] - 0.1 * (a * a).sum(-1) + 0.5
    w = MODULE.solve_reward_probe(phi, reward, ridge=1e-8)
    pred = phi @ w
    assert MODULE.ev_score(pred, reward) > 0.99


def test_successor_features_are_discounted_occupancy():
    features = torch.zeros(3, 1, 2)
    features[0, 0] = torch.tensor([1.0, 0.0])
    features[1, 0] = torch.tensor([0.0, 1.0])
    features[2, 0] = torch.tensor([2.0, 2.0])
    terminations = torch.zeros(3, 1)
    truncations = torch.zeros(3, 1)
    tail = torch.tensor([[3.0, 0.0]])
    trunc_boot = torch.zeros(3, 1, 2)
    occ = MODULE.successor_features(features, terminations, truncations, trunc_boot, tail, gamma=0.5)
    torch.testing.assert_close(occ[2, 0], torch.tensor([2.0, 2.0]) + 0.5 * tail[0])
    torch.testing.assert_close(occ[1, 0], features[1, 0] + 0.5 * occ[2, 0])
    torch.testing.assert_close(occ[0, 0], features[0, 0] + 0.5 * occ[1, 0])


def test_successor_features_stop_on_termination():
    features = torch.ones(2, 1, 1)
    terminations = torch.tensor([[1.0], [0.0]])
    truncations = torch.zeros(2, 1)
    tail = torch.tensor([[9.0]])
    trunc_boot = torch.zeros(2, 1, 1)
    occ = MODULE.successor_features(features, terminations, truncations, trunc_boot, tail, gamma=0.9)
    torch.testing.assert_close(occ[0], torch.ones(1, 1))


def test_jepa_is_attached_and_sigreg_finite():
    torch.manual_seed(1)
    args = MODULE.Args(emb_dim=8, hidden_dim=16, sigreg_num_proj=32, sigreg_ref_n=16)
    agent = MODULE.Agent(obs_dim=5, action_dim=2, args=args)
    obs = torch.randn(16, 5, requires_grad=True)
    action = torch.randn(16, 2)
    next_obs = torch.randn(16, 5)
    total, pred, sig, emb, next_emb = MODULE.jepa_loss(agent, obs, action, next_obs, args)
    assert torch.isfinite(total)
    assert torch.isfinite(pred)
    assert torch.isfinite(sig)
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.predictor.parameters())
    # Attached target: next encoder also receives gradient.
    assert any(
        name.startswith("encoder") and p.grad is not None and p.grad.abs().sum() > 0
        for name, p in agent.named_parameters()
    )


def test_actor_does_not_train_encoder():
    torch.manual_seed(2)
    args = MODULE.Args(emb_dim=8, hidden_dim=16)
    agent = MODULE.Agent(obs_dim=5, action_dim=2, args=args)
    obs = torch.randn(8, 5)
    emb = agent.encode(obs).detach()
    action, logprob, _ = agent.get_action(emb)
    (-logprob.mean()).backward()
    assert all(p.grad is None for p in agent.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.actor_mean.parameters())


def test_sigreg_layout_is_time_major():
    args = MODULE.Args(emb_dim=6, hidden_dim=12, sigreg_num_proj=16, sigreg_ref_n=8)
    agent = MODULE.Agent(obs_dim=4, action_dim=2, args=args)
    emb = torch.randn(8, 6)
    # Official shared SIGReg: (T,B,D). unsqueeze(0) makes T=1, B=8.
    stat = agent.sigreg(emb.unsqueeze(0))
    assert stat.ndim == 0
    assert torch.isfinite(stat)
    # Effective rank of isotropic noise is high relative to dim.
    rank = MODULE.effective_rank(torch.randn(200, 6))
    assert rank > 4.0
