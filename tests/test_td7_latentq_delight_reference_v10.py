"""Point-Q delight with a greedy-action or latent-V reference; whole score gated."""
import ast
from dataclasses import asdict
import inspect
import os
import textwrap
from types import SimpleNamespace

import pytest
import torch

from cleanrl.td7.delight import td7_latentq_delight_cubature_raw_v9 as reference
from cleanrl.td7.delight import td7_latentq_delight_hermite_integrated_v8 as checkpointed
from cleanrl.td7.delight import td7_latentq_delight_reference_v10 as trainer

DIAGNOSTIC = {name: index for index, name in enumerate(trainer.DELIGHT_DIAGNOSTICS)}


def syntax(value):
    return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(value))))


def policy(samples=4, batch=64, action_dim=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    mu = torch.randn(batch, action_dim, generator=generator, dtype=torch.float64).mul(0.5).requires_grad_()
    log_std = torch.randn(batch, action_dim, generator=generator, dtype=torch.float64).mul(0.3).sub(1.5).requires_grad_()
    eps = torch.randn(samples // 2, batch, action_dim, generator=generator, dtype=torch.float64)
    eps = torch.cat([eps, -eps])
    return mu, log_std, eps, (mu + log_std.exp() * eps).detach()


def antithetic(samples, like, action_dim=3):
    eps = torch.randn(samples // 2, like.shape[0], action_dim, device=like.device, dtype=like.dtype)
    return torch.cat([eps, -eps])


def networks(seed, state_dim=5, action_dim=3, zs_dim=8, hidden=16, batch=32, dtype=torch.float32):
    """Model forwards are CUDA-only and run under the queue, never on the host CPU."""
    if os.environ.get("CLEANRL_RUN_GPU_TESTS") != "1":
        pytest.skip("model cases run on CUDA through mlq: set CLEANRL_RUN_GPU_TESTS=1")
    trainer.configure_runtime()
    device = torch.device("cuda")
    torch.manual_seed(seed)
    actor = trainer.Actor(state_dim, action_dim, zs_dim, hidden).to(device=device, dtype=dtype)
    critic = trainer.PreActionLatentTwinCritic(state_dim, action_dim, zs_dim, hidden).to(device=device, dtype=dtype)
    encoder = trainer.Encoder(state_dim, action_dim, zs_dim, hidden).to(device=device, dtype=dtype)
    state = torch.randn(batch, state_dim, device=device, dtype=dtype)
    return actor, critic, encoder, state, dict(zs_dim=zs_dim, hidden_dim=hidden)


def stub(args, actor, critic, encoder, state, **extra):
    agent = SimpleNamespace(args=args, state_dim=state.shape[1], actor=actor, critic=critic,
                            fixed_encoder=encoder, **extra)
    for name in ("_gate", "_advantage", "_actor_loss_fn", "gradient_probe"):
        setattr(agent, name, getattr(trainer.TD7Agent, name).__get__(agent))
    return agent


def flat_gradient(loss, actor, **kwargs):
    return torch.cat([g.reshape(-1) for g in torch.autograd.grad(
        loss, list(actor.parameters()), allow_unused=True, materialize_grads=True, **kwargs)])


def gradients(loss, mu, log_std):
    return torch.autograd.grad(loss, (mu, log_std))


def test_critic_and_data_path_are_the_reference_runs():
    for name in ("Actor", "Encoder", "EnsembleLinear", "PreActionLatentTwinCritic", "HostPolicy",
                 "gaussian_log_probability", "gaussian_kl", "lap_huber"):
        assert syntax(getattr(reference, name)) == syntax(getattr(trainer, name))
    for name in ("_encoder_loss_fn", "_critic_loss_fn", "_entropy_fn", "hard_update", "stage", "flush"):
        assert syntax(getattr(reference.TD7Agent, name)) == syntax(getattr(trainer.TD7Agent, name))
    for name in ("train", "checkpoint_state", "restore_checkpoint_state"):
        assert syntax(getattr(checkpointed.TD7Agent, name)) == syntax(getattr(trainer.TD7Agent, name))


def test_defaults_differ_from_the_raw_cubature_run_only_by_reference_and_scope():
    before, after = asdict(reference.Args()), asdict(trainer.Args())
    assert before.pop("surprisal") == "relative"
    assert (before.pop("gate_scope"), after.pop("gate_scope")) == ("mean", "all")
    assert after.pop("actor_baseline") == "modal"
    assert after.pop("entropy_floor") is True
    for name in ("resume", "checkpoint_path", "checkpoint_interval_seconds", "exp_name"):
        after.pop(name)
    before.pop("exp_name")
    assert before == after


def test_odd_advantage_leaves_the_gated_location_gradient_equal_to_pg():
    mu, log_std, eps, pre_action = policy()
    advantage = (eps * torch.linspace(-2, 3, eps.shape[-1], dtype=eps.dtype)).sum(-1)
    gated, _ = trainer.delight_score_loss(pre_action, mu, log_std, advantage, objective="delight")
    ungated, _ = trainer.delight_score_loss(pre_action, mu, log_std, advantage, objective="pg")
    (gated_mu, gated_scale), (pg_mu, pg_scale) = gradients(gated, mu, log_std), gradients(ungated, mu, log_std)
    torch.testing.assert_close(gated_mu, pg_mu)
    # Delight on an odd advantage acts through the scale score alone, and upward.
    assert (pg_scale - gated_scale).sum() > 0.1 * advantage.abs().mean()


def test_reference_offset_changes_delight_but_not_paired_location_pg():
    mu, log_std, eps, pre_action = policy(seed=1)
    advantage = (eps * 1.5).sum(-1) + 0.3 * eps.square().sum(-1)
    offset = torch.linspace(-1, 1, eps.shape[1], dtype=eps.dtype)
    for objective, invariant in (("pg", True), ("delight", False)):
        plain, _ = trainer.delight_score_loss(pre_action, mu, log_std, advantage, objective=objective)
        shifted, _ = trainer.delight_score_loss(pre_action, mu, log_std, advantage + offset, objective=objective)
        equal = torch.allclose(gradients(plain, mu, log_std)[0], gradients(shifted, mu, log_std)[0])
        assert equal is invariant


def test_only_the_score_is_differentiated():
    mu, log_std, eps, pre_action = policy(seed=2)
    advantage = eps.sum(-1).requires_grad_()
    loss, diagnostics = trainer.delight_score_loss(pre_action, mu, log_std, advantage)
    loss.backward()
    assert advantage.grad is None and not diagnostics.requires_grad
    assert mu.grad.abs().sum() > 0 and log_std.grad.abs().sum() > 0


def test_mean_scope_keeps_pg_scale_gradient_and_matches_the_reference_trainer():
    mu, log_std, eps, pre_action = policy(seed=3)
    advantage = eps.sum(-1) - 0.2 * eps.square().sum(-1) + 0.4
    kwargs = dict(objective="delight", temperature=0.7, cap=4.0)
    for scale in ("raw", "rms"):
        ours, diagnostics = trainer.delight_score_loss(
            pre_action, mu, log_std, advantage, scale=scale, scope="mean", **kwargs)
        theirs, expected = reference.delight_score_loss(
            pre_action, mu, log_std, advantage, surprisal="relative", scale=scale, scope="mean", **kwargs)
        for ours_grad, theirs_grad in zip(gradients(ours, mu, log_std), gradients(theirs, mu, log_std)):
            torch.testing.assert_close(ours_grad, theirs_grad)
        torch.testing.assert_close(diagnostics[:5], expected[:5])
    ungated, _ = trainer.delight_score_loss(pre_action, mu, log_std, advantage, objective="pg")
    mean_scope, _ = trainer.delight_score_loss(pre_action, mu, log_std, advantage, scope="mean")
    torch.testing.assert_close(gradients(mean_scope, mu, log_std)[1], gradients(ungated, mu, log_std)[1])


def test_coefficient_diagnostics_are_the_output_space_gradients():
    mu, log_std, eps, pre_action = policy(seed=4)
    advantage = eps.sum(-1) - 0.5 * eps.square().sum(-1) + 1.0
    count = advantage.shape[1] * mu.shape[1]
    loss, gated = trainer.delight_score_loss(pre_action, mu, log_std, advantage)
    gated_mu, gated_scale = gradients(loss, mu, log_std)
    loss, ungated = trainer.delight_score_loss(pre_action, mu, log_std, advantage, objective="pg")
    pg_mu, pg_scale = gradients(loss, mu, log_std)
    torch.testing.assert_close(gated[DIAGNOSTIC["scale_coefficient_full_dg"]], -gated_scale.sum() * advantage.shape[1] / count)
    torch.testing.assert_close(gated[DIAGNOSTIC["scale_coefficient_pg"]], -pg_scale.sum() * advantage.shape[1] / count)
    torch.testing.assert_close(
        gated[DIAGNOSTIC["scale_coefficient_full_inner"]] + gated[DIAGNOSTIC["scale_coefficient_full_outer"]],
        gated[DIAGNOSTIC["scale_coefficient_full_dg"]])
    torch.testing.assert_close(gated[DIAGNOSTIC["mean_coefficient_ratio"]], gated_mu.norm() / pg_mu.norm())
    torch.testing.assert_close(
        gated[DIAGNOSTIC["mean_coefficient_cosine"]], (gated_mu * pg_mu).sum() / (gated_mu.norm() * pg_mu.norm()))
    torch.testing.assert_close(ungated[DIAGNOSTIC["mean_coefficient_ratio"]], torch.ones((), dtype=eps.dtype))
    torch.testing.assert_close(ungated[DIAGNOSTIC["gate_mean"]], torch.ones((), dtype=eps.dtype))


def test_sharp_greedy_optimum_can_inflate_scale_and_the_split_shows_why():
    """U = -c*eps^2 against the greedy action: PG always shrinks; delight flips sign once the gate closes the tail."""
    nodes = torch.linspace(-6, 6, 4001, dtype=torch.float64)
    weights = torch.exp(-0.5 * nodes.square())
    weights = weights / weights.sum()
    mu, log_std = torch.zeros(1, 1, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64)

    def scale_coefficients(curvature):
        advantage = -curvature * nodes.square()
        weight = 2 * torch.sigmoid(advantage * (0.5 * nodes.square()).clamp(max=10.0))
        _, diagnostics = trainer.delight_score_loss(nodes[:, None, None], mu, log_std, advantage[:, None])
        torch.testing.assert_close(diagnostics[DIAGNOSTIC["gate_mean"]], weight.mean())
        return (weights * weight * advantage * (nodes.square() - 1)).sum(), (weights * advantage * (nodes.square() - 1)).sum()

    weak, weak_pg = scale_coefficients(0.1)
    sharp, sharp_pg = scale_coefficients(2.0)
    assert weak_pg < weak < 0 and sharp_pg < 0 < sharp


@pytest.mark.parametrize("baseline,calls", [("modal", [5 * 32]), ("latent", [4 * 32, 32])])
def test_actor_loss_spends_five_critic_rows_per_state_on_the_chosen_reference_only(baseline, calls):
    actor, critic, encoder, state, sizes = networks(5)
    seen = []
    critic.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].shape[0]))
    agent = stub(trainer.Args(actor_baseline=baseline, **sizes), actor, critic, encoder, state)
    loss, log_std, diagnostics = agent._actor_loss_fn(
        state, encoder.zs(state).detach(), torch.tensor(0.05, device=state.device), antithetic(4, state))
    assert seen == calls
    assert diagnostics.shape == (len(trainer.DELIGHT_DIAGNOSTICS) + 4,) and diagnostics.isfinite().all()
    loss.backward()
    assert all(parameter.grad is None for parameter in [*critic.parameters(), *encoder.parameters()])
    assert sum(parameter.grad.abs().sum() for parameter in actor.parameters()) > 0


def test_reference_arms_differ_by_a_per_state_constant():
    """Same draws: the two advantages differ per state only, which paired location PG cannot see."""
    actor, critic, encoder, state, sizes = networks(6)
    fixed_zs, eps = encoder.zs(state).detach(), antithetic(4, state)
    with torch.no_grad():
        mu, log_std = actor(state, fixed_zs)
        agent = stub(trainer.Args(**sizes), actor, critic, encoder, state)
        modal, latent = (agent._advantage(state, fixed_zs, mu, log_std.exp(), eps, baseline)[1]
                         for baseline in ("modal", "latent"))
    gap = latent - modal
    torch.testing.assert_close(gap, gap[:1].expand_as(gap), atol=1e-5, rtol=0)
    assert gap.abs().max() > 1e-3
    location = {}
    for objective in ("pg", "delight"):
        for baseline in ("modal", "latent"):
            agent = stub(trainer.Args(actor_baseline=baseline, actor_objective=objective, **sizes),
                         actor, critic, encoder, state)
            actor.zero_grad()
            agent._actor_loss_fn(state, fixed_zs, torch.zeros((), device=state.device), eps)[0].backward()
            location[objective, baseline] = actor.l3.weight.grad.clone()
    torch.testing.assert_close(location["pg", "modal"], location["pg", "latent"], atol=1e-5, rtol=1e-4)
    assert not torch.allclose(location["delight", "modal"], location["delight", "latent"], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("baseline", ["modal", "latent"])
@pytest.mark.parametrize("objective,scope", [("pg", "all"), ("delight", "mean"), ("delight", "all")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradient_probe_compares_objectives_on_the_stored_inputs(objective, scope, baseline, dtype):
    actor, critic, encoder, state, sizes = networks(10, dtype=dtype)
    args = trainer.Args(actor_objective=objective, gate_scope=scope, actor_baseline=baseline, **sizes)
    fixed_zs, eps = encoder.zs(state).detach(), antithetic(args.actor_samples, state)
    agent = stub(args, actor, critic, encoder, state, probe_state=state, probe_zs=fixed_zs, probe_eps=eps,
                 log_alpha=torch.tensor(-2.0, device=state.device, requires_grad=True))
    before = [parameter.detach().clone() for parameter in actor.parameters()]
    generator_states = torch.get_rng_state(), torch.cuda.get_rng_state(state.device)
    probe = dict(zip(trainer.GRADIENT_PROBE, agent.gradient_probe().tolist(), strict=True))
    assert torch.equal(generator_states[0], torch.get_rng_state())
    assert torch.equal(generator_states[1], torch.cuda.get_rng_state(state.device))
    assert all(parameter.grad is None for parameter in [*actor.parameters(), *critic.parameters(), agent.log_alpha])
    assert all(torch.equal(a, b) for a, b in zip(before, actor.parameters()))

    def trained(alpha, **overrides):
        other = stub(trainer.Args(**{**asdict(args), **overrides}), actor, critic, encoder, state)
        return flat_gradient(other._actor_loss_fn(state, fixed_zs, torch.tensor(alpha, device=state.device), eps)[0], actor)

    pg = trained(0.0, actor_objective="pg")
    gated = trained(0.0, actor_objective="delight", gate_scope="all")
    located = trained(0.0, actor_objective="delight", gate_scope="mean")
    reward = trained(0.0)
    assert probe["pg_norm"] == pytest.approx(pg.norm().item(), rel=1e-5)
    assert probe["delight_all_cosine"] == pytest.approx(torch.cosine_similarity(pg, gated, 0).item(), abs=1e-5)
    assert probe["delight_mean_difference_ratio"] == pytest.approx(((located - pg).norm() / pg.norm()).item(), rel=1e-4)
    assert probe["delight_all_difference_ratio"] == pytest.approx(((gated - pg).norm() / pg.norm()).item(), rel=1e-4)
    # Reward and entropy parts add up to the gradient of the loss the optimizer sees.
    entropy = trained(agent.log_alpha.exp().item()) - reward
    assert probe["entropy_norm"] == pytest.approx(entropy.norm().item(), rel=1e-3)
    assert probe["entropy_reward_ratio"] == pytest.approx((entropy.norm() / reward.norm()).item(), rel=1e-3)
    assert probe["entropy_reward_cosine"] == pytest.approx(torch.cosine_similarity(entropy, reward, 0).item(), abs=1e-4)

    # The other reference is the other arm's gated gradient on the same draws.
    swapped = trained(0.0, actor_objective="delight", gate_scope="all",
                      actor_baseline="latent" if baseline == "modal" else "modal")
    assert probe["other_reference_full_cosine"] == pytest.approx(torch.cosine_similarity(swapped, gated, 0).item(), abs=1e-5)
    with torch.no_grad():
        mu, log_std = actor(state, fixed_zs)
        modal, latent = (agent._advantage(state, fixed_zs, mu, log_std.exp(), eps, name)[1] for name in ("modal", "latent"))
    assert probe["reference_gap_mean"] == pytest.approx((latent - modal).mean().item(), abs=1e-6)
    assert probe["reference_sign_disagreement"] == pytest.approx(((modal > 0) != (latent > 0)).float().mean().item())

    # Inner and outer scale parts are the gated full gradient minus its location part.
    mu, log_std = actor(state, fixed_zs)
    advantage = modal if baseline == "modal" else latent
    weight = trainer.delight_weight(advantage, eps)[0]
    coefficient = -(weight * advantage)[..., None] * (eps.square() - 1) / advantage.numel()
    parts = [flat_gradient(log_std, actor, grad_outputs=(coefficient * mask).sum(0), retain_graph=True)
             for mask in (eps.square() < 1, eps.square() >= 1)]
    assert probe["scale_inner_norm"] == pytest.approx(parts[0].norm().item(), rel=1e-4)
    assert probe["scale_outer_norm"] == pytest.approx(parts[1].norm().item(), rel=1e-4)
    assert probe["scale_inner_outer_cosine"] == pytest.approx(torch.cosine_similarity(*parts, 0).item(), abs=1e-4)
    location = flat_gradient(trainer.delight_score_loss(
        mu + log_std.exp().detach() * eps, mu, log_std.detach(), advantage)[0], actor)
    combined = parts[0] + parts[1] + location
    if dtype == torch.float64:
        # Algebra check: TF32 cannot conceal a wrong score decomposition.
        torch.testing.assert_close(combined, gated, atol=1e-10, rtol=1e-8)
    else:
        # Production TF32 rounds each VJP cotangent independently. Bound the
        # numerical error by the parts, rather than a potentially canceled sum.
        parts_norm = parts[0].norm() + parts[1].norm() + location.norm()
        assert (combined - gated).norm() <= 2e-3 * parts_norm


def test_probe_inputs_are_captured_and_checkpointed():
    assert {"probe_state", "probe_zs", "probe_eps"} <= set(trainer.CHECKPOINT_BUFFERS)
    source = inspect.getsource(trainer.TD7Agent.__init__)
    assert "self.probe_state, self.probe_zs, self.probe_eps]" in source
    assert not hasattr(trainer, "gaussian_cubature") and not hasattr(trainer, "integrated_hermite_score_loss")


@pytest.mark.parametrize("scope", ["mean", "all"])
def test_scale_diagnostic_matches_the_scope_actually_trained(scope):
    mu, log_std, eps, pre_action = policy(seed=19)
    advantage = eps.sum(-1) - 0.2 * eps.square().sum(-1)
    loss, diagnostics = trainer.delight_score_loss(pre_action, mu, log_std, advantage, scope=scope)
    _, scale_gradient = gradients(loss, mu, log_std)
    expected = -scale_gradient.sum() / log_std.shape[-1]
    torch.testing.assert_close(diagnostics[DIAGNOSTIC["scale_coefficient_trained"]], expected)


def test_matching_pg_moments_and_exact_centering_do_not_determine_dg():
    """A higher-order Gaussian advantage invisible to native H still has DG signal."""
    import numpy as np
    nodes, weights = np.polynomial.hermite_e.hermegauss(128)
    eps = torch.from_numpy(nodes)[:, None, None]
    probability = torch.from_numpy(weights / np.sqrt(2 * np.pi))
    advantage = eps[:, 0, 0].pow(3) - 3 * eps[:, 0, 0]
    advantage += 0.4 * (eps[:, 0, 0].pow(4) - 6 * eps[:, 0, 0].square() + 3)
    score = torch.stack([eps[:, 0, 0], eps[:, 0, 0].square() - 1], -1)
    torch.testing.assert_close((probability * advantage).sum(), torch.zeros((), dtype=torch.float64), atol=1e-13, rtol=0)
    pg = (probability[:, None] * advantage[:, None] * score).sum(0)
    torch.testing.assert_close(pg, torch.zeros(2, dtype=torch.float64), atol=1e-13, rtol=0)
    gate = trainer.delight_weight(advantage[:, None], eps)[0][:, 0]
    dg = (probability[:, None] * gate[:, None] * advantage[:, None] * score).sum(0)
    assert dg[0] > 0.4 and dg[1] > 2.0  # doubled gate; native second-order H is zero
