import importlib.util
import math
from pathlib import Path

import torch
import torch.nn.functional as F

SCRIPT = Path(__file__).parents[1] / "cleanrl" / "td-jepa" / "td_jepa_v1.py"
SPEC = importlib.util.spec_from_file_location("td_jepa_v1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def tiny_args(**overrides):
    values = dict(
        phi_dim=8,
        psi_dim=4,
        encoder_hidden_dim=16,
        phi_hidden_layers=0,
        psi_hidden_layers=1,
        predictor_hidden_dim=16,
        predictor_hidden_layers=1,
        predictor_embedding_layers=2,
        num_parallel=2,
        actor_hidden_dim=16,
        actor_hidden_layers=1,
        actor_embedding_layers=2,
        normalize_obs=False,
        batch_size=32,
        gamma=0.98,
        phi_ortho_coef=0.1,
        psi_ortho_coef=0.1,
        predictor_pessimism_penalty=0.0,
        actor_pessimism_penalty=0.0,
        actor_std=0.2,
        stddev_clip=0.3,
        train_goal_ratio=0.5,
        norm_z=True,
    )
    values.update(overrides)
    return MODULE.Args(**values)


def test_norm_scales_to_sqrt_dim():
    x = torch.randn(7, 9)
    y = MODULE.Norm()(x)
    torch.testing.assert_close(y.norm(dim=-1), torch.full((7,), math.sqrt(9.0)))


def test_backward_map_is_l2_normalized():
    enc = MODULE.BackwardMap(obs_dim=5, out_dim=6, hidden_dim=8, hidden_layers=0)
    out = enc(torch.randn(11, 5))
    torch.testing.assert_close(out.norm(dim=-1), torch.full((11,), math.sqrt(6.0)), atol=1e-5, rtol=1e-5)


def test_forward_map_twin_shape():
    pred = MODULE.ForwardMap(8, 4, 3, hidden_dim=16, hidden_layers=1, embedding_layers=2, output_dim=4, num_parallel=2)
    out = pred(torch.randn(5, 8), torch.randn(5, 4), torch.randn(5, 3))
    assert out.shape == (2, 5, 4)


def test_reward_inference_recovers_linear_task():
    torch.manual_seed(0)
    psi = F.normalize(torch.randn(200, 6), dim=-1) * math.sqrt(6.0)
    z_true = MODULE.project_z(torch.randn(1, 6), True)
    reward = psi @ z_true.T
    z_hat = MODULE.reward_inference(psi, reward, True)
    assert F.cosine_similarity(z_hat, z_true).item() > 0.99


def test_orth_loss_penalizes_off_diagonal_and_rewards_unit_diag():
    off = 1.0 - torch.eye(4)
    loss_i, diag_i, off_i = MODULE.orth_loss(torch.eye(4), off, off.sum())
    assert off_i.item() == 0.0
    assert diag_i.item() == -1.0

    loss_c, _, off_c = MODULE.orth_loss(torch.ones(4, 3), off, off.sum())
    assert off_c.item() > 0.0
    assert loss_c.item() > loss_i.item()


def test_ensemble_stats_is_twin_mean_without_pessimism():
    preds = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    torch.testing.assert_close(MODULE.ensemble_stats(preds, 0.0), torch.tensor([[2.0, 3.0]]))


def test_td_target_stops_bootstrap_on_terminal_and_update_is_finite():
    torch.manual_seed(1)
    args = tiny_args()
    agent = MODULE.TDJEPA(obs_dim=5, action_dim=2, args=args)
    obs = torch.randn(32, 5)
    next_obs = torch.randn(32, 5)
    action = torch.tanh(torch.randn(32, 2))
    dones = torch.zeros(32, 1)
    dones[:4] = 1.0
    z = MODULE.sample_z(32, args.psi_dim, obs.device, True)
    off = 1.0 - torch.eye(32)

    with torch.no_grad():
        next_phi = agent.target_phi(next_obs)
        next_psi = agent.target_psi(next_obs)
        next_action = agent.actor(next_phi, z, args.actor_std).sample(clip=args.stddev_clip)
        boot_phi = MODULE.ensemble_stats(agent.target_phi_predictor(next_phi, z, next_action), 0.0)
        expected = next_psi + args.gamma * (1.0 - dones) * boot_phi
        assert torch.all(expected[:4] == next_psi[:4])
        assert not torch.allclose(expected[4:], next_psi[4:])

    metrics = MODULE.update_tdjepa(
        agent,
        obs,
        action,
        next_obs,
        args.gamma * (1.0 - dones),
        z,
        off,
        off.sum(),
        torch.optim.Adam(agent.phi.parameters(), lr=1e-3),
        torch.optim.Adam(agent.psi.parameters(), lr=1e-3),
        torch.optim.Adam(agent.phi_predictor.parameters(), lr=1e-3),
        torch.optim.Adam(agent.psi_predictor.parameters(), lr=1e-3),
    )
    assert torch.isfinite(metrics["tdjepa_loss"])
    assert torch.isfinite(metrics["phi_tdjepa_loss"])
    assert torch.isfinite(metrics["psi_tdjepa_loss"])


def test_actor_loss_routes_through_successor_inner_product():
    torch.manual_seed(2)
    args = tiny_args()
    agent = MODULE.TDJEPA(obs_dim=5, action_dim=2, args=args)
    obs = torch.randn(16, 5)
    z = MODULE.sample_z(16, args.psi_dim, obs.device, True)
    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=1e-3)
    metrics = MODULE.update_actor(agent, obs, z, actor_opt)
    assert torch.isfinite(metrics["actor_loss"])
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.actor.parameters())
    assert all(p.grad is None for p in agent.phi.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.phi_predictor.parameters())

def test_mixed_z_is_projected_and_permuted_away_from_same_index():
    torch.manual_seed(3)
    args = tiny_args(train_goal_ratio=1.0)
    agent = MODULE.TDJEPA(obs_dim=5, action_dim=2, args=args)
    goals = agent.psi(torch.randn(20, 5)).detach()
    projected = MODULE.project_z(goals, True)
    z = agent.sample_mixed_z(goals)
    torch.testing.assert_close(z.norm(dim=-1), torch.full((20,), math.sqrt(args.psi_dim)), atol=1e-5, rtol=1e-5)
    # Official TD-JEPA permutes goals so z_i is not ψ(s'_i).
    same_index = (z - projected).abs().sum(dim=-1) < 1e-5
    assert not same_index.all()
    # Multiset of z equals the projected goals (permutation, not a new set).
    z_sorted, _ = z.sort(dim=0)
    g_sorted, _ = projected.sort(dim=0)
    torch.testing.assert_close(z_sorted, g_sorted, atol=1e-5, rtol=1e-5)


def test_collect_z_refreshes_only_finished_envs_and_tracks_task_skill():
    torch.manual_seed(4)
    args = tiny_args(collect_task_ratio=0.0)
    agent = MODULE.TDJEPA(obs_dim=5, action_dim=2, args=args)
    collect_z, collect_is_task = agent.sample_collect_z(4, torch.device("cpu"))
    assert not collect_is_task.any()
    before = collect_z.clone()
    finished = torch.tensor([True, False, True, False])
    agent.refresh_finished_collect_z(collect_z, collect_is_task, finished)
    assert not torch.allclose(collect_z[0], before[0])
    torch.testing.assert_close(collect_z[1], before[1])
    assert not torch.allclose(collect_z[2], before[2])
    torch.testing.assert_close(collect_z[3], before[3])

    args_task = tiny_args(collect_task_ratio=1.0)
    agent_task = MODULE.TDJEPA(obs_dim=5, action_dim=2, args=args_task)
    z, is_task = agent_task.sample_collect_z(5, torch.device("cpu"))
    assert is_task.all()
    torch.testing.assert_close(z, agent_task.z_task.expand_as(z))
