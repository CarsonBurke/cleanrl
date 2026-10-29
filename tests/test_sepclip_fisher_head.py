"""Meaningful CUDA contracts; run this suite through mlq, never bare launch."""
import copy
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from cleanrl.ppo_continuous_action_sepclip_fisher_head_v8 import (
    Args, FisherHead, beta_logit_fisher, factorize_effective_head,
    head_covariances, make_optimizers, natural_head_direction,
)


def make_actor(features=5, actions=3):
    actor = nn.Module()
    actor.head = nn.Sequential(nn.Linear(features, 2 * actions, device='cuda'))
    actor.readout_gain = nn.Parameter(torch.ones(2 * actions, device='cuda'))
    with torch.no_grad():
        actor.head[0].weight.copy_(F.normalize(actor.head[0].weight, dim=1))
    return actor


def test_exact_beta_fisher_matches_kl_hessian():
    logits = torch.tensor([-.8, .5, 1.2, -.3], device='cuda', dtype=torch.float64)
    old_a, old_b = (1 + F.softplus(logits.detach())).chunk(2)
    old = torch.distributions.Beta(old_a, old_b)

    def exact_kl(new):
        a, b = (1 + F.softplus(new)).chunk(2)
        return torch.distributions.kl_divergence(old, torch.distributions.Beta(a, b)).sum()

    hessian = torch.autograd.functional.hessian(exact_kl, logits)
    fisher = beta_logit_fisher(logits)
    expected = torch.zeros_like(hessian)
    for action in range(2):
        indices = torch.tensor([action, action + 2], device='cuda')
        expected[indices[:, None], indices[None, :]] = fisher[action]
    torch.testing.assert_close(expected, hessian, atol=1e-10, rtol=1e-9)


@torch.no_grad()
def test_factorization_preserves_logits_and_unit_rows():
    torch.manual_seed(1)
    actor = make_actor()
    p = torch.randn_like(actor.head[0].weight)
    c = torch.randn_like(actor.head[0].bias)
    features = torch.randn(32, p.shape[1], device='cuda')
    expected = F.linear(features, p, c)
    factorize_effective_head(actor.head[0].weight, actor.head[0].bias,
                            actor.readout_gain, p, c)
    actual = actor.readout_gain * actor.head(features)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actor.head[0].weight.norm(dim=1), torch.ones_like(actor.readout_gain))


def test_zero_effective_row_remains_finite_and_recovers_effective_gradient():
    actor = make_actor()
    p = torch.randn_like(actor.head[0].weight)
    c = torch.randn_like(actor.head[0].bias)
    p[0].zero_()
    with torch.no_grad():
        factorize_effective_head(actor.head[0].weight, actor.head[0].bias,
                                actor.readout_gain, p, c)
        actor.head[0].weight.copy_(F.normalize(actor.head[0].weight, dim=1, eps=1e-12))
    features = torch.randn(32, p.shape[1], device='cuda')
    actual = actor.readout_gain * actor.head(features)
    torch.testing.assert_close(actual, F.linear(features, p, c), atol=2e-6, rtol=2e-6)
    assert torch.isfinite(actual).all()
    actual[:, 0].sum().backward()
    effective_gradient = actor.head[0].weight.grad[0] / actor.readout_gain[0]
    torch.testing.assert_close(effective_gradient, features.sum(0), atol=2e-5, rtol=2e-5)


@torch.no_grad()
def test_kronecker_solve_and_undamped_quadratic_radius():
    torch.manual_seed(2)
    features = torch.randn(96, 5, device='cuda')
    logits = torch.randn(96, 6, device='cuda')
    damping, radius = .01, torch.tensor(.001, device='cuda')
    a, b, ai, bi = head_covariances(features, logits, damping)
    g = torch.randn(3, 2, 6, device='cuda')
    ad = a + damping * a.diagonal().mean() * torch.eye(6, device='cuda')
    bd = b + damping * b.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None] * torch.eye(2, device='cuda')
    independent = torch.stack([torch.linalg.solve(torch.kron(bd[i].contiguous(), ad.contiguous()),
                                                g[i].flatten()).view(2, 6) for i in range(3)])
    quadratic = sum(.5 * independent[i].flatten() @ torch.kron(b[i].contiguous(), a.contiguous())
                    @ independent[i].flatten() for i in range(3))
    expected = independent * (radius / quadratic).sqrt()
    delta, predicted = natural_head_direction(g, a, b, ai, bi, radius)
    torch.testing.assert_close(delta, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(predicted, radius, atol=1e-8, rtol=1e-5)
    # The metric is undamped: damping must not sneak into the radius definition.
    direct_quadratic = sum(.5 * delta[i].flatten() @ torch.kron(b[i].contiguous(), a.contiguous())
                           @ delta[i].flatten() for i in range(3))
    torch.testing.assert_close(direct_quadratic, radius, atol=1e-8, rtol=1e-5)


@torch.no_grad()
def test_compiled_state_continuation_gradient_ownership_and_zero_step():
    torch.manual_seed(3)
    actor = make_actor()
    optimizer = FisherHead(actor)
    features = torch.randn(128, 5, device='cuda')
    optimizer.refresh(features, actor.readout_gain * actor.head(features))
    restored_actor = copy.deepcopy(actor)
    restored = FisherHead(restored_actor)
    restored.load_state_dict(optimizer.state_dict())
    for radius in (.001, .0003):
        optimizer.radius.fill_(radius)
        restored.radius.fill_(radius)
        weight_grad = torch.randn_like(actor.head[0].weight)
        bias_grad = torch.randn_like(actor.head[0].bias)
        for model in (actor, restored_actor):
            model.head[0].weight.grad = weight_grad.clone()
            model.head[0].bias.grad = bias_grad.clone()
        optimizer.step()
        restored.step()
        for original, resumed in zip(actor.parameters(), restored_actor.parameters()):
            torch.testing.assert_close(original, resumed, atol=0, rtol=0)
        torch.testing.assert_close(actor.head[0].weight.grad, weight_grad, atol=0, rtol=0)
        torch.testing.assert_close(actor.head[0].bias.grad, bias_grad, atol=0, rtol=0)
        torch.testing.assert_close(optimizer.predicted_kl, optimizer.radius, atol=1e-8, rtol=1e-5)
    before = copy.deepcopy(actor.state_dict())
    actor.head[0].weight.grad.zero_()
    actor.head[0].bias.grad.zero_()
    optimizer.step()
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert optimizer.predicted_kl == 0


def test_optimizer_membership_keeps_natural_head_exclusive():
    from cleanrl.ppo_continuous_action_sepclip_fisher_head_v8 import ResidualMLP
    agent = nn.Module()
    agent.actor = ResidualMLP(17, 12, .01).cuda()
    agent.critic = ResidualMLP(17, 1, 1.).cuda()
    for mode in ('anvil2', 'adam'):
        args = Args(actor_matrix_optimizer=mode)
        args.batch_size = args.minibatch_size = 16384
        args.num_iterations = 488
        matrix, adam = make_optimizers(agent, args)
        selected = {id(p) for group in adam.param_groups for p in group['params']}
        head = {id(agent.actor.head[0].weight), id(agent.actor.head[0].bias), id(agent.actor.readout_gain)}
        assert not selected & head
        critic = {id(p) for p in agent.critic.parameters()}
        assert critic <= selected
        actor_hidden = {id(layer.weight) for stage in (agent.actor.first, agent.actor.second)
                        for layer in (stage[0].gate, stage[0].up, stage[0].down)}
        if mode == 'adam':
            assert matrix is None
            assert actor_hidden <= selected
        else:
            assert not actor_hidden & selected
            assert {id(p) for group in matrix.param_groups for p in group['params']} == actor_hidden


@torch.no_grad()
def test_local_policy_geometry_matches_surrogate_derivative_and_action_affine_invariance(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from audit_sepclip_mechanisms import local_policy_geometry

    torch.manual_seed(14)
    logits = torch.randn(256, 6, device='cuda', dtype=torch.float64)
    a, b = (1+F.softplus(logits)).chunk(2, -1)
    old = torch.distributions.Beta(a, b)
    native = old.sample()
    advantages = torch.randn(256, device='cuda', dtype=torch.float64)
    lambda1 = torch.randn_like(advantages)
    log_scale = torch.tensor([2., 3., 5.], device='cuda', dtype=torch.float64).log()
    batch = dict(native=native, logprob=(old.log_prob(native)-log_scale).sum(-1),
                 advantages=advantages, advantages_mc=lambda1)
    direction = torch.randn_like(logits)
    epsilon = 1e-5
    after = logits+epsilon*direction
    result = local_policy_geometry(logits, after, batch, log_scale)
    next_a, next_b = (1+F.softplus(after)).chunk(2, -1)
    new = torch.distributions.Beta(next_a, next_b)
    ratio = ((new.log_prob(native)-log_scale).sum(-1)-batch['logprob']).exp()
    expected = (advantages*(ratio-1)).mean()
    expected_lambda1 = (lambda1*(ratio-1)).mean()
    assert abs(result['first_order_gae_gain']-float(expected)) < 1e-9
    assert abs(result['first_order_lambda1_gain']-float(expected_lambda1)) < 1e-9
    assert abs(result['first_order_clipped_gae_gain']-float(expected)) < 1e-9
    native_batch = dict(batch, logprob=old.log_prob(native).sum(-1))
    native_result = local_policy_geometry(logits, after, native_batch, torch.zeros_like(log_scale))
    for name in result:
        assert abs(result[name]-native_result[name]) < 1e-12
    # Exercise both flat clipping branches and both active sign branches.
    before_ratio = torch.tensor([.7, .9, 1.1, 1.3], device='cuda', dtype=logits.dtype).repeat(64)
    signed = torch.where(torch.arange(256, device='cuda')%8<4, 1., -1.).to(logits.dtype)
    clipped_batch = dict(batch, advantages=signed, logprob=batch['logprob']-before_ratio.log())
    clipped_result = local_policy_geometry(logits, after, clipped_batch, log_scale)
    after_ratio = ((new.log_prob(native)-log_scale).sum(-1)-clipped_batch['logprob']).exp()
    objective = lambda r: torch.minimum(signed*r, signed*r.clamp(.8,1.2)).mean()
    clipped_difference = objective(after_ratio)-objective(before_ratio)
    assert abs(clipped_result['first_order_clipped_gae_gain']-float(clipped_difference)) < 1e-9
