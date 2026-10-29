"""Signed-density DG and a recoverable projected entropy dual; CUDA models via mlq."""
import ast
from dataclasses import asdict
import inspect
import math
import os
import textwrap
from types import SimpleNamespace

import pytest
import torch

from cleanrl.td7.delight import td7_latentq_delight_reference_v10 as previous
from cleanrl.td7.delight import td7_latentq_delight_projected_v11 as trainer


def syntax(value):
    return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(value))))


def policy(seed=4):
    generator = torch.Generator().manual_seed(seed)
    mu = torch.randn(12, 3, generator=generator, dtype=torch.float64).mul(.3).requires_grad_()
    log_std = torch.randn(12, 3, generator=generator, dtype=torch.float64).mul(.2).sub(2).requires_grad_()
    half = torch.randn(2, 12, 3, generator=generator, dtype=torch.float64)
    eps = torch.cat([half, -half])
    action = (mu + log_std.exp() * eps).detach()
    advantage = (eps.sum(-1) - .4 * eps.square().sum(-1)).requires_grad_()
    return mu, log_std, eps, action, advantage


def test_critic_and_queries_remain_frozen():
    for name in ('Actor', 'Encoder', 'PreActionLatentTwinCritic', 'gaussian_log_probability', 'gaussian_kl'):
        assert syntax(getattr(trainer, name)) == syntax(getattr(previous, name))
    for name in ('_critic_loss_fn', '_encoder_loss_fn', '_advantage', '_entropy_fn', 'stage', 'flush', 'hard_update'):
        assert syntax(getattr(trainer.TD7Agent, name)) == syntax(getattr(previous.TD7Agent, name))


@pytest.mark.parametrize('argument,value', [('surprisal', 'relative'), ('actor_baseline', 'modal'), ('target_smoothing_sigma', 0.), ('gate_scope', 'mean')])
def test_checkpoint_rejects_algorithm_changes(argument, value):
    args = trainer.Args()
    checkpoint = {'format': 'td7_latentq_delight_projected_v11', 'version': 1, 'args': asdict(args)}
    trainer.validate_checkpoint_args(checkpoint, args)
    alternate = trainer.Args(**{**asdict(args), argument: value})
    if getattr(args, argument) == value:
        value = 'density' if argument == 'surprisal' else value
        alternate = trainer.Args(**{**asdict(args), argument: value})
    with pytest.raises(ValueError, match=argument):
        trainer.validate_checkpoint_args(checkpoint, alternate)


@pytest.mark.parametrize('format,version', [('td7_latentq_delight_reference_v10', 1), ('td7_latentq_delight_projected_v11', 2)])
def test_wrong_checkpoint_is_rejected(format, version):
    args = trainer.Args()
    with pytest.raises(ValueError):
        trainer.validate_checkpoint_args({'format': format, 'version': version, 'args': asdict(args)}, args)


def networks():
    if os.environ.get('CLEANRL_RUN_GPU_TESTS') != '1':
        pytest.skip('actual models require queued CUDA: CLEANRL_RUN_GPU_TESTS=1')
    trainer.configure_runtime()
    torch.manual_seed(23)
    device = torch.device('cuda')
    sizes = dict(zs_dim=8, hidden_dim=16)
    actor = trainer.Actor(5, 3, 8, 16).to(device=device, dtype=torch.float64)
    critic = trainer.PreActionLatentTwinCritic(5, 3, 8, 16).to(device=device, dtype=torch.float64)
    encoder = trainer.Encoder(5, 3, 8, 16).to(device=device, dtype=torch.float64)
    states = torch.randn(24, 5, device=device, dtype=torch.float64)
    zs = encoder.zs(states).detach()
    half = torch.randn(2, 24, 3, device=device, dtype=torch.float64)
    return actor, critic, encoder, states, zs, torch.cat([half, -half]), sizes


def stub(module, args, actor, critic, encoder, state, **extra):
    agent = SimpleNamespace(args=args, actor=actor, critic=critic, fixed_encoder=encoder,
                            state_dim=state.shape[-1], **extra)
    for name in ('_gate', '_advantage', '_actor_loss_fn', 'gradient_probe'):
        setattr(agent, name, getattr(module.TD7Agent, name).__get__(agent))
    return agent


def flat(loss, actor, **kwargs):
    return torch.cat([x.reshape(-1) for x in torch.autograd.grad(loss, list(actor.parameters()),
        allow_unused=True, materialize_grads=True, **kwargs)])


@pytest.mark.parametrize('surprisal', ['relative', 'density'])
def test_gate_formula_keeps_signed_density_and_double_prefactor(surprisal):
    mu, log_std, eps, _, advantage = policy()
    weight, rarity, _ = trainer.delight_weight(advantage, eps, log_std=log_std, surprisal=surprisal, cap=2.)
    expected = .5 * eps.square().sum(-1)
    if surprisal == 'density':
        expected = (expected + log_std.sum(-1) + .5 * mu.shape[-1] * math.log(2 * math.pi)).clamp(-2., 2.)
        assert (expected < 0).any() and (expected == -2).any()
    else:
        expected = expected.clamp(max=2.)
    assert not weight.requires_grad and not rarity.requires_grad
    torch.testing.assert_close(rarity, expected)
    torch.testing.assert_close(weight, 2 * torch.sigmoid(advantage * expected))
    zero, _, _ = trainer.delight_weight(torch.zeros_like(advantage), eps, log_std=log_std, surprisal=surprisal)
    torch.testing.assert_close(zero, torch.ones_like(zero))


@pytest.mark.parametrize('surprisal', ['relative', 'density'])
def test_only_frozen_gated_score_is_differentiated(surprisal):
    mu, log_std, eps, action, advantage = policy()
    loss, diagnostics = trainer.delight_score_loss(action, mu, log_std, advantage, surprisal=surprisal)
    with torch.no_grad():
        weight = trainer.delight_weight(advantage, eps, log_std=log_std, surprisal=surprisal)[0]
    expected = -(weight * advantage.detach() * trainer.gaussian_log_probability(action, mu, log_std)).mean()
    actual_grad = torch.autograd.grad(loss, (mu, log_std), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (mu, log_std), retain_graph=True)
    for actual, wanted in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, wanted)
    loss.backward()
    assert advantage.grad is None and not diagnostics.requires_grad


def test_relative_actor_score_remains_v10_and_pg_is_gate_independent():
    mu, log_std, _, action, advantage = policy()
    before = previous.delight_score_loss(action, mu, log_std, advantage)[0]
    after = trainer.delight_score_loss(action, mu, log_std, advantage, surprisal='relative')[0]
    density_pg = trainer.delight_score_loss(action, mu, log_std, advantage, objective='pg', surprisal='density')[0]
    relative_pg = trainer.delight_score_loss(action, mu, log_std, advantage, objective='pg', surprisal='relative')[0]
    for one, two in [(before, after), (density_pg, relative_pg)]:
        for actual, wanted in zip(torch.autograd.grad(one, (mu, log_std), retain_graph=True),
                                  torch.autograd.grad(two, (mu, log_std), retain_graph=True)):
            torch.testing.assert_close(actual, wanted)


@pytest.mark.parametrize('entropy,gap_sign', [(-8., -1), (-4., 1)])
def test_direct_dual_has_nonzero_correct_gradient_at_zero(entropy, gap_sign):
    alpha = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    observed = torch.tensor(entropy, dtype=torch.float64, requires_grad=True)
    loss = trainer.projected_alpha_loss(alpha, observed, -6.)
    derivative, entropy_derivative = torch.autograd.grad(loss, (alpha, observed), allow_unused=True)
    assert derivative * gap_sign > 0
    torch.testing.assert_close(derivative, torch.tensor(entropy + 6., dtype=torch.float64))
    assert entropy_derivative is None
    with torch.no_grad():
        alpha.sub_(.01 * derivative).clamp_(min=0.)
    assert (alpha > 0) == (entropy < -6.)


def test_projected_dual_recovers_from_boundary_without_an_exponential_multiplier():
    alpha = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([alpha], lr=.01)
    # At the boundary, high entropy projects outward updates back to zero.
    for _ in range(20):
        optimizer.zero_grad()
        trainer.projected_alpha_loss(alpha, torch.tensor(-4.), -6.).backward()
        optimizer.step()
        with torch.no_grad():
            alpha.clamp_(min=0.)
    assert alpha.item() == 0
    # Retained Adam moments can delay a reversal, but cannot extinguish the dual gradient.
    for _ in range(30):
        optimizer.zero_grad()
        trainer.projected_alpha_loss(alpha, torch.tensor(-8.), -6.).backward()
        torch.testing.assert_close(alpha.grad, torch.tensor(-2., dtype=torch.float64))
        optimizer.step()
        with torch.no_grad():
            alpha.clamp_(min=0.)
    assert alpha.item() > .1
    assert trainer.Args().initial_alpha == 0.
    assert not trainer.effective_entropy_alpha(alpha, True).requires_grad
    assert trainer.effective_entropy_alpha(alpha, False).item() == 0.


@pytest.mark.parametrize('surprisal', ['relative', 'density'])
@pytest.mark.parametrize('objective', ['pg', 'delight'])
@pytest.mark.parametrize('alpha', [0., .08])
def test_cuda_actor_gradient_matches_explicit_frozen_score(surprisal, objective, alpha):
    actor, critic, encoder, state, zs, eps, sizes = networks()
    args = trainer.Args(actor_baseline='latent', surprisal=surprisal, actor_objective=objective, **sizes)
    agent = stub(trainer, args, actor, critic, encoder, state)
    actual = agent._actor_loss_fn(state, zs, torch.tensor(alpha, device=state.device, dtype=state.dtype), eps)[0]
    actual_gradient = flat(actual, actor)
    mu, log_std = actor(state, zs)
    with torch.no_grad():
        u, advantage = agent._advantage(state, zs, mu, log_std.exp(), eps, 'latent')
        rarity = .5 * eps.square().sum(-1)
        rarity = rarity.clamp(max=args.surprisal_cap) if surprisal == 'relative' else (
            rarity + log_std.sum(-1) + .5 * mu.shape[-1] * math.log(2 * math.pi)).clamp(-args.surprisal_cap, args.surprisal_cap)
        weight = 2 * torch.sigmoid(advantage * rarity / args.delight_temperature) if objective == 'delight' else torch.ones_like(advantage)
    expected = -(weight * advantage * trainer.gaussian_log_probability(u, mu, log_std)).mean() - alpha * log_std.sum(-1).mean()
    torch.testing.assert_close(actual_gradient, flat(expected, actor), atol=1e-10, rtol=1e-8)
    assert all(parameter.grad is None for parameter in [*critic.parameters(), *encoder.parameters()])


@pytest.mark.parametrize('surprisal', ['relative', 'density'])
def test_cuda_probe_keeps_parameters_gradients_and_rng_unchanged(surprisal):
    actor, critic, encoder, state, zs, eps, sizes = networks()
    alpha = torch.tensor(.08, device=state.device, dtype=state.dtype, requires_grad=True)
    agent = stub(trainer, trainer.Args(actor_baseline='latent', surprisal=surprisal, **sizes),
                 actor, critic, encoder, state, probe_state=state, probe_zs=zs, probe_eps=eps, entropy_alpha=alpha)
    all_parameters = [*actor.parameters(), *critic.parameters(), *encoder.parameters(), alpha]
    before = [p.detach().clone() for p in all_parameters]
    host_rng, device_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    probe = agent.gradient_probe()
    assert probe.isfinite().all()
    assert all(torch.equal(saved, current) for saved, current in zip(before, all_parameters))
    assert all(p.grad is None for p in all_parameters)
    assert torch.equal(host_rng, torch.get_rng_state())
    assert torch.equal(device_rng, torch.cuda.get_rng_state())


def test_density_gate_requires_scale_and_rejects_unknown_surprisal():
    _, log_std, eps, _, advantage = policy()
    with pytest.raises(ValueError, match="log_std"):
        trainer.delight_weight(advantage, eps, surprisal="density")
    with pytest.raises(ValueError, match="surprisal"):
        trainer.delight_weight(advantage, eps, log_std=log_std, surprisal="unknown")
