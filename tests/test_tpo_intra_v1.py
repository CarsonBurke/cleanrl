import importlib.util
import sys
from pathlib import Path

import torch
from torch.distributions.normal import Normal


ROOT = Path(__file__).parents[1]


def _load():
    path = ROOT / "cleanrl/tpo/ppo_continuous_action_tpo_intra_v1.py"
    spec = importlib.util.spec_from_file_location("ppo_continuous_action_tpo_intra_v1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def test_utility_whitens_and_clips():
    advantages = torch.tensor([-8.0, -1.0, 0.0, 1.0, 8.0])
    utility = MODULE.tpo_utility(advantages, utility_clip=1.5)
    assert torch.allclose(utility, utility.clamp(-1.5, 1.5))
    unclipped = MODULE.tpo_utility(advantages, utility_clip=0.0)
    torch.testing.assert_close(unclipped.mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(unclipped.std(), torch.tensor(1.0), atol=1e-6, rtol=0)


def test_variance_free_batch_is_actor_neutral():
    """No advantage variance -> zero utility -> zero gradient at the behaviour policy."""
    advantages = torch.full((16,), 3.25)
    utility = MODULE.tpo_utility(advantages, utility_clip=3.0)
    torch.testing.assert_close(utility, torch.zeros(16))

    logratio = torch.zeros(16, requires_grad=True)
    MODULE.tpo_intra_loss(logratio, utility, eta=4.0, logratio_guard=20.0).sum().backward()
    torch.testing.assert_close(logratio.grad, torch.zeros(16))


def test_loss_is_zero_and_minimal_at_the_target_ratio():
    utility = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    eta = 4.0
    at_target = utility / eta
    loss_at_target = MODULE.tpo_intra_loss(at_target, utility, eta=eta, logratio_guard=20.0)
    torch.testing.assert_close(loss_at_target, torch.zeros(5), atol=1e-6, rtol=0)

    for offset in (-1.0, -0.3, 0.3, 1.0):
        elsewhere = MODULE.tpo_intra_loss(at_target + offset, utility, eta=eta, logratio_guard=20.0)
        assert torch.all(elsewhere > 0.0)


def test_gradient_is_ratio_minus_target_ratio():
    utility = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    eta = 2.0
    logratio = torch.tensor([-0.7, -0.2, 0.0, 0.2, 0.7], requires_grad=True)
    MODULE.tpo_intra_loss(logratio, utility, eta=eta, logratio_guard=20.0).sum().backward()
    expected = logratio.detach().exp() - (utility / eta).exp()
    torch.testing.assert_close(logratio.grad, expected)


def test_gradient_vanishes_only_at_the_target_and_is_convex():
    utility = torch.tensor([1.0])
    eta = 2.0
    grid = torch.linspace(-2.0, 2.0, 401).requires_grad_(True)
    loss = MODULE.tpo_intra_loss(grid, utility.expand(401), eta=eta, logratio_guard=20.0)
    (grad,) = torch.autograd.grad(loss.sum(), grid, create_graph=True)
    # Unique sign change of the gradient at logratio = utility / eta.
    sign_changes = (grad[:-1].sign() != grad[1:].sign()).sum()
    assert sign_changes.item() == 1
    crossing = grid.detach()[(grad.detach().abs()).argmin()]
    torch.testing.assert_close(crossing, utility[0] / eta, atol=1e-2, rtol=0)
    # Convex in logratio: second derivative is the (positive) ratio.
    (curvature,) = torch.autograd.grad(grad.sum(), grid)
    assert torch.all(curvature > 0.0)


def test_first_step_matches_the_policy_gradient_up_to_eta():
    """At ratio 1 with small utility, -dL/dlogratio ~= utility / eta."""
    utility = torch.tensor([-0.02, -0.01, 0.01, 0.02])
    eta = 4.0
    logratio = torch.zeros(4, requires_grad=True)
    MODULE.tpo_intra_loss(logratio, utility, eta=eta, logratio_guard=20.0).sum().backward()
    torch.testing.assert_close(-logratio.grad, utility / eta, atol=1e-4, rtol=0)


def test_suppression_gradient_is_bounded_but_amplification_is_not():
    """grad = ratio - target: bounded by the target as ratio -> 0, linear as ratio grows."""
    utility = torch.tensor([0.0])
    eta = 1.0
    collapsing = torch.tensor([-30.0], requires_grad=True)
    MODULE.tpo_intra_loss(collapsing, utility, eta=eta, logratio_guard=20.0).backward()
    assert collapsing.grad.abs().item() <= 1.0 + 1e-5

    exploding = torch.tensor([3.0], requires_grad=True)
    MODULE.tpo_intra_loss(exploding, utility, eta=eta, logratio_guard=20.0).backward()
    torch.testing.assert_close(exploding.grad, torch.tensor([torch.e**3 - 1.0]))


def test_logratio_guard_prevents_overflow():
    utility = torch.tensor([0.0, 0.0])
    huge = torch.tensor([200.0, -200.0], requires_grad=True)
    loss = MODULE.tpo_intra_loss(huge, utility, eta=1.0, logratio_guard=20.0)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert torch.isfinite(huge.grad).all()


def test_gradient_flows_through_a_gaussian_policy_as_ratio_minus_target():
    mean = torch.tensor([[0.3, -0.2]], requires_grad=True)
    logstd = torch.tensor([[0.0, 0.0]], requires_grad=True)
    action = torch.tensor([[0.9, -0.4]])
    old_logprob = Normal(torch.tensor([[0.0, 0.0]]), torch.tensor([[1.0, 1.0]])).log_prob(action).sum(1)

    dist = Normal(mean, logstd.exp())
    logprob = dist.log_prob(action).sum(1)
    logratio = logprob - old_logprob
    utility = torch.tensor([1.5])
    eta = 3.0
    MODULE.tpo_intra_loss(logratio, utility, eta=eta, logratio_guard=20.0).sum().backward()

    scale = (logratio.detach().exp() - (utility / eta).exp()).item()
    # Reference: scale * d logpi / d theta.
    ref_mean = torch.tensor([[0.3, -0.2]], requires_grad=True)
    ref_logstd = torch.tensor([[0.0, 0.0]], requires_grad=True)
    ref_logprob = Normal(ref_mean, ref_logstd.exp()).log_prob(action).sum(1)
    ref_logprob.backward()
    torch.testing.assert_close(mean.grad, scale * ref_mean.grad)
    torch.testing.assert_close(logstd.grad, scale * ref_logstd.grad)


def test_batch_scope_target_is_shared_across_minibatches():
    """Batch-scope utilities make the target identical no matter the shuffle."""
    torch.manual_seed(0)
    advantages = torch.randn(256)
    utilities = MODULE.tpo_utility(advantages, utility_clip=3.0)
    perm = torch.randperm(256)
    torch.testing.assert_close(MODULE.tpo_utility(advantages, 3.0)[perm], utilities[perm])
    # Per-minibatch whitening does not agree with the global target.
    chunk = perm[:64]
    local = MODULE.tpo_utility(advantages[chunk], utility_clip=3.0)
    assert not torch.allclose(local, utilities[chunk], atol=1e-3)
