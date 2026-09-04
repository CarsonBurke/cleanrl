import importlib.util
from pathlib import Path

import torch

SCRIPT = Path(__file__).parents[1] / "cleanrl" / "td7" / "misc" / "td7_delightful_pg_v1.py"
SPEC = importlib.util.spec_from_file_location("td7_delightful_pg_v1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_delightful_gate_uses_signed_clipped_surprisal_and_detaches():
    advantages = torch.tensor([[2.0], [-2.0]], requires_grad=True)
    logprobs = torch.tensor([[-3.0], [-3.0]], requires_grad=True)
    gate, surprisal, delight = MODULE.delightful_gate(advantages, logprobs)
    torch.testing.assert_close(surprisal, torch.tensor([[3.0], [3.0]]))
    torch.testing.assert_close(delight, torch.tensor([[6.0], [-6.0]]))
    torch.testing.assert_close(gate, torch.sigmoid(torch.tensor([[6.0], [-6.0]])))
    assert not gate.requires_grad

    _, clipped, _ = MODULE.delightful_gate(
        torch.ones(3, 1),
        torch.tensor([[-100.0], [100.0], [-1.0]]),
    )
    torch.testing.assert_close(clipped, torch.tensor([[10.0], [-10.0], [1.0]]))


def test_fixed_action_score_is_stochastic_bounded_and_actor_differentiable():
    actor = MODULE.Actor(state_dim=5, action_dim=3, zs_dim=7, hdim=11)
    state = torch.randn(13, 5)
    zs = torch.randn(13, 7)

    with torch.no_grad():
        action, raw_action, sampled_logprob = actor.sample_score(state, zs)
    assert action.shape == raw_action.shape == (13, 3)
    assert sampled_logprob.shape == (13, 1)
    assert not raw_action.requires_grad
    assert torch.all(action.abs() <= 1.0)

    score_logprob = actor.log_prob_from_raw(state, zs, raw_action)
    (-score_logprob.mean()).backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in actor.parameters())


def test_reparameterized_sample_remains_only_for_entropy_and_soft_target():
    actor = MODULE.Actor(state_dim=4, action_dim=2, zs_dim=6, hdim=10)
    state = torch.randn(8, 4)
    zs = torch.randn(8, 6)
    action, logprob = actor.sample(state, zs)
    assert action.requires_grad
    assert logprob.requires_grad
    assert action.shape == (8, 2)
    assert logprob.shape == (8, 1)
    assert torch.all(action.abs() <= 1.0)


def test_td7_dg_actor_has_no_pathwise_q_objective_or_importance_ratio():
    source = SCRIPT.read_text()
    assert "actor_q = self.critic(" in source
    assert "baseline_q = self.critic(" in source
    assert "advantages = actor_q - baseline_q" in source
    assert "score_logpi = self.actor.log_prob_from_raw(" in source
    assert "dg_loss = -(gate.detach() * advantages.detach() * score_logpi).mean()" in source
    assert "actor_loss = dg_loss + self.alpha * entropy_logpi.mean()" in source
    assert "actor_loss = (self.alpha * actor_logpi - actor_Q" not in source
    assert "importance_ratio" not in source
    assert "behavior_logprob" not in source
    assert "raw_action = normal.sample()" in source
    assert "raw_action = normal.rsample()" in source


def test_td7_infrastructure_and_paper_batch_are_retained():
    args = MODULE.Args()
    assert args.batch_size == 256
    assert args.dg_eta == 1.0
    assert args.dg_surprisal_clip == 10.0
    assert args.policy_freq == 2
    assert args.use_checkpoints
    assert args.alpha_autotune
    assert args.soft_bellman
