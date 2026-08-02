from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from cleanrl.td7_lesale_v1 import (
    Args,
    Critic,
    LeSALEAgent,
    SDNoiseActor,
    StockEncoder,
    TD7PCActorTrainer,
    avg_l1_norm,
    avg_l1_norm_vjp,
    make_td7_pc_actor_settle_core_eager,
    td7_pc_actor_curvature_factors,
    td7_pc_actor_free_phase,
    td7_pc_actor_policy_from_raw,
)


def pc_args(**overrides):
    values = dict(
        hidden_dim=6,
        zs_dim=5,
        pc_actor=True,
        pc_actor_inference_steps=10,
        pc_actor_inference_scale=1.0,
        pc_actor_nudge=0.05,
        pc_actor_curvature_damping=0.05,
        pc_actor_adam_beta1=0.9,
        pc_actor_adam_beta2=0.999,
        pc_actor_adam_epsilon=1e-8,
        torch_compile=False,
        compile_mode="default",
    )
    values.update(overrides)
    return Args(**values)


def augmented_grad(weight_grad, bias_grad):
    return torch.cat([weight_grad, bias_grad[:, None]], dim=1)


def pc_energy(actor, state, zs, states, target):
    z0, z1, z2 = states
    mean0 = avg_l1_norm(F.linear(state, actor.l0.weight, actor.l0.bias))
    mean1 = F.linear(
        torch.cat([z0, zs], dim=1), actor.l1.weight, actor.l1.bias
    )
    mean2 = F.linear(F.relu(z1), actor.l2.weight, actor.l2.bias)
    output_features = F.relu(z2)
    output = torch.cat(
        [
            F.linear(output_features, actor.l3.weight, actor.l3.bias),
            F.linear(
                output_features,
                actor.log_std_head.weight,
                actor.log_std_head.bias,
            ),
        ],
        dim=1,
    )
    return 0.5 * (
        (z0 - mean0).square().sum(dim=1)
        + (z1 - mean1).square().sum(dim=1)
        + (z2 - mean2).square().sum(dim=1)
        + (target - output).square().sum(dim=1)
    ).mean()


def test_pc_free_phase_exactly_matches_sdnoise_actor_forward():
    torch.manual_seed(0)
    actor = SDNoiseActor(4, 3, zs_dim=5, hdim=6, head_seed=11)
    state = torch.randn(7, 4)
    zs = torch.randn(7, 5)
    epsilon = torch.randn(7, 3)
    _, _, raw_output = td7_pc_actor_free_phase(actor, state, zs)
    action, log_std = td7_pc_actor_policy_from_raw(actor, raw_output, epsilon)
    expected_mean, expected_log_std = actor.policy_stats(state, zs)
    expected_action = (expected_mean + expected_log_std.exp() * epsilon).clamp(-1, 1)
    torch.testing.assert_close(action, expected_action, rtol=0, atol=0)
    torch.testing.assert_close(log_std, expected_log_std, rtol=0, atol=0)


def test_avg_l1_norm_vjp_matches_autograd():
    torch.manual_seed(1)
    value = torch.randn(8, 6, dtype=torch.float64, requires_grad=True)
    cotangent = torch.randn_like(value)
    expected = torch.autograd.grad((avg_l1_norm(value) * cotangent).sum(), value)[0]
    actual = avg_l1_norm_vjp(value.detach(), cotangent)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_reverse_gs_gradients_are_exact_pc_energy_gradients():
    torch.manual_seed(2)
    actor = SDNoiseActor(4, 3, zs_dim=5, hdim=6, head_seed=12).double()
    state = torch.randn(5, 4, dtype=torch.float64)
    zs = torch.randn(5, 5, dtype=torch.float64)
    _, free_states, raw_output = td7_pc_actor_free_phase(actor, state, zs)
    states = [
        (free + 0.1 * torch.randn_like(free)).requires_grad_(True)
        for free in free_states
    ]
    target = raw_output.detach() + 0.1 * torch.randn_like(raw_output)
    energy = pc_energy(actor, state, zs, states, target)
    expected = torch.autograd.grad(energy, states)

    z0, z1, z2 = states
    hidden = z0.shape[1]
    mean0 = avg_l1_norm(F.linear(state, actor.l0.weight, actor.l0.bias))
    mean1 = F.linear(torch.cat([z0, zs], dim=1), actor.l1.weight, actor.l1.bias)
    mean2 = F.linear(F.relu(z1), actor.l2.weight, actor.l2.bias)
    output_weight = torch.cat([actor.l3.weight, actor.log_std_head.weight], dim=0)
    output_bias = torch.cat([actor.l3.bias, actor.log_std_head.bias], dim=0)
    output_error = target - F.linear(F.relu(z2), output_weight, output_bias)
    gradient2 = z2 - mean2 - (z2 > 0).to(z2.dtype) * F.linear(
        output_error, output_weight.T
    )
    downstream2 = z2 - F.linear(F.relu(z1), actor.l2.weight, actor.l2.bias)
    gradient1 = z1 - mean1 - (z1 > 0).to(z1.dtype) * F.linear(
        downstream2, actor.l2.weight.T
    )
    downstream1 = z1 - F.linear(
        torch.cat([z0, zs], dim=1), actor.l1.weight, actor.l1.bias
    )
    gradient0 = z0 - mean0 - F.linear(
        downstream1, actor.l1.weight[:, :hidden].T
    )
    for actual, reference in zip((gradient0, gradient1, gradient2), expected):
        # pc_energy is a batch mean, while state inference uses per-example gradients.
        torch.testing.assert_close(actual / state.shape[0], reference, rtol=1e-10, atol=1e-10)


def test_hidden_directions_are_negative_energy_gradients_and_heads_are_exact():
    torch.manual_seed(3)
    args = pc_args(pc_actor_nudge=0.02)
    actor = SDNoiseActor(4, 3, zs_dim=5, hdim=6, head_seed=13).double()
    trainer = TD7PCActorTrainer(actor, args)
    state = torch.randn(7, 4, dtype=torch.float64)
    zs = torch.randn(7, 5, dtype=torch.float64)
    terminal_force = 0.2 * torch.randn(7, 6, dtype=torch.float64)
    directions, _ = trainer.settle_and_directions(state, zs, terminal_force)

    with torch.no_grad():
        _, free_states, free_output = td7_pc_actor_free_phase(actor, state, zs)
        target = free_output + args.pc_actor_nudge * terminal_force
        factors = td7_pc_actor_curvature_factors(
            actor, free_states, args.pc_actor_curvature_damping
        )
        states = make_td7_pc_actor_settle_core_eager(
            actor, state, zs, free_states, factors, target, args
        )
    energy = pc_energy(actor, state, zs, [value.detach() for value in states], target)
    parameters = []
    for layer in trainer.layers:
        parameters.extend([layer.weight, layer.bias])
    gradients = torch.autograd.grad(energy, parameters)
    expected_hidden = [
        -augmented_grad(gradients[index], gradients[index + 1]) / args.pc_actor_nudge
        for index in range(0, 6, 2)
    ]
    for actual, reference in zip(directions[:3], expected_hidden):
        torch.testing.assert_close(actual, reference, rtol=2e-9, atol=2e-9)
    free_features = F.relu(free_states[2])
    mean_force, std_force = terminal_force.chunk(2, dim=1)
    expected_heads = [
        torch.cat(
            [
                force.T @ free_features / state.shape[0],
                force.mean(dim=0, keepdim=True).T,
            ],
            dim=1,
        )
        for force in (mean_force, std_force)
    ]
    for actual, reference in zip(directions[3:], expected_heads):
        torch.testing.assert_close(actual, reference, rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize("dominant_head", ["mean", "std"])
def test_production_shape_exact_heads_do_not_cross_contaminate(dominant_head):
    torch.manual_seed(31 if dominant_head == "mean" else 32)
    action_dim = 6
    actor = SDNoiseActor(
        17, action_dim, zs_dim=256, hdim=256, head_seed=131
    )
    args = pc_args(hidden_dim=256, zs_dim=256)
    trainer = TD7PCActorTrainer(actor, args)
    state = torch.randn(32, 17)
    zs = torch.randn(32, 256)
    dominant = torch.randn(32, action_dim)
    subordinate = 1e-4 * torch.randn(32, action_dim)
    terminal_force = (
        torch.cat([dominant, subordinate], dim=1)
        if dominant_head == "mean"
        else torch.cat([subordinate, dominant], dim=1)
    )

    directions, diagnostics = trainer.settle_and_directions(
        state, zs, terminal_force
    )
    _, _, raw_output = td7_pc_actor_free_phase(actor, state, zs)
    exact_objective = (raw_output * terminal_force).sum() / state.shape[0]
    exact_gradients = torch.autograd.grad(
        exact_objective, tuple(actor.parameters())
    )
    exact_directions = [
        augmented_grad(exact_gradients[index], exact_gradients[index + 1])
        for index in range(0, len(exact_gradients), 2)
    ]

    for actual, exact in zip(directions[3:], exact_directions[3:]):
        torch.testing.assert_close(actual, exact, rtol=2e-6, atol=2e-7)
        assert F.cosine_similarity(actual.flatten(), exact.flatten(), dim=0) > 0.999999
    for actual, exact in zip(directions[:3], exact_directions[:3]):
        assert torch.isfinite(actual).all()
        assert actual.norm() > 0
        assert F.cosine_similarity(actual.flatten(), exact.flatten(), dim=0) > 0
    assert torch.isfinite(diagnostics["hidden_direction_norm"])
    assert torch.isfinite(diagnostics["exact_head_direction_norm"])


def test_inverse_nudge_compensation_has_linear_small_nudge_limit():
    torch.manual_seed(4)
    actor = SDNoiseActor(4, 3, zs_dim=5, hdim=6, head_seed=14)
    state = 0.2 * torch.randn(8, 4)
    zs = 0.2 * torch.randn(8, 5)
    force = 0.1 * torch.randn(8, 6)
    medium = TD7PCActorTrainer(
        actor, pc_args(pc_actor_nudge=2e-3, pc_actor_inference_steps=20)
    ).settle_and_directions(state, zs, force)[0]
    small = TD7PCActorTrainer(
        actor, pc_args(pc_actor_nudge=1e-3, pc_actor_inference_steps=20)
    ).settle_and_directions(state, zs, force)[0]
    for medium_direction, small_direction in zip(medium, small):
        cosine = F.cosine_similarity(
            medium_direction.flatten(), small_direction.flatten(), dim=0
        )
        ratio = medium_direction.norm() / small_direction.norm().clamp_min(1e-12)
        assert cosine > 0.999
        assert 0.99 < ratio < 1.01


def test_boundary_leaf_force_recovers_existing_actor_parameter_gradient():
    torch.manual_seed(5)
    actor = SDNoiseActor(4, 3, zs_dim=5, hdim=6, head_seed=15).double()
    encoder = StockEncoder(4, 3, zs_dim=5, hdim=6).double()
    critic = Critic(4, 3, zs_dim=5, hdim=6).double()
    for module in (encoder, critic):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    state = torch.randn(7, 4, dtype=torch.float64)
    zs = encoder.zs(state).detach()
    epsilon = torch.randn(7, 3, dtype=torch.float64)
    alpha = torch.tensor(0.17, dtype=torch.float64)

    mean, log_std = actor.policy_stats(state, zs)
    action = (mean + log_std.exp() * epsilon).clamp(-1, 1)
    zsa = encoder.zsa(zs, action)
    objective = critic(state, action, zsa, zs).mean(dim=1) + alpha * log_std.sum(dim=1)
    parameters = tuple(actor.parameters())
    expected = torch.autograd.grad(objective.sum(), parameters, retain_graph=True)

    _, _, raw_output = td7_pc_actor_free_phase(actor, state, zs)
    raw_leaf = raw_output.detach().requires_grad_(True)
    leaf_action, leaf_log_std = td7_pc_actor_policy_from_raw(actor, raw_leaf, epsilon)
    leaf_zsa = encoder.zsa(zs, leaf_action)
    leaf_objective = (
        critic(state, leaf_action, leaf_zsa, zs).mean(dim=1)
        + alpha * leaf_log_std.sum(dim=1)
    )
    terminal_force = torch.autograd.grad(leaf_objective.sum(), raw_leaf)[0]
    actual = torch.autograd.grad(raw_output, parameters, grad_outputs=terminal_force)
    for recovered, reference in zip(actual, expected):
        torch.testing.assert_close(recovered, reference, rtol=2e-9, atol=2e-9)


def test_pc_agent_preserves_actor_state_dict_and_never_builds_actor_grads():
    torch.manual_seed(6)
    args = pc_args(
        sd_noise=True,
        use_subsig=False,
        residual_predictor=False,
        buffer_size=64,
        batch_size=4,
        fused_adam=False,
        gpu_replay=False,
    )
    writer = SimpleNamespace(add_scalar=lambda *args, **kwargs: None)
    agent = LeSALEAgent(4, 3, 1.0, args, torch.device("cpu"), writer)
    assert agent.actor_optimizer is None
    assert agent.pc_actor_trainer is not None
    assert all(not parameter.requires_grad for parameter in agent.actor.parameters())
    assert all(parameter.grad is None for parameter in agent.actor.parameters())
    assert agent.actor.state_dict().keys() == agent.actor_target.state_dict().keys()
    assert agent.actor.state_dict().keys() == agent.checkpoint_actor.state_dict().keys()

    state = torch.randn(4, 4)
    with torch.no_grad():
        zs = agent.fixed_encoder.zs(state)
    agent.critic.requires_grad_(False)
    force, _, _, _ = agent._pc_actor_terminal_force(
        state, zs, torch.randn(4, 3)
    )
    diagnostics = agent.pc_actor_trainer.step(state, zs, force, args.actor_lr)
    agent.critic.requires_grad_(True)
    assert torch.isfinite(diagnostics["update_rms"])
    assert all(parameter.grad is None for parameter in agent.actor.parameters())


def test_pc_terminal_force_normalization_preserves_direction_and_reports_raw_rms():
    torch.manual_seed(7)
    args = pc_args(
        sd_noise=True,
        use_subsig=False,
        residual_predictor=False,
        buffer_size=64,
        batch_size=4,
        fused_adam=False,
        gpu_replay=False,
    )
    writer = SimpleNamespace(add_scalar=lambda *args, **kwargs: None)
    agent = LeSALEAgent(4, 3, 1.0, args, torch.device("cpu"), writer)
    agent.critic.requires_grad_(False)
    state = torch.randn(4, 4)
    with torch.no_grad():
        zs = agent.fixed_encoder.zs(state)
    noise = torch.randn(4, 3)

    agent.args.pc_actor_normalize_terminal_force = False
    raw_force, _, _, raw_rms = agent._pc_actor_terminal_force(state, zs, noise)
    agent.args.pc_actor_normalize_terminal_force = True
    normalized_force, _, _, normalized_raw_rms = agent._pc_actor_terminal_force(
        state, zs, noise
    )

    torch.testing.assert_close(normalized_raw_rms, raw_rms)
    torch.testing.assert_close(
        normalized_force,
        raw_force / raw_rms.clamp_min(args.pc_actor_force_rms_min),
    )
    assert torch.isfinite(raw_rms)
    assert raw_rms > 0
    assert normalized_force.square().mean().sqrt() == pytest.approx(1.0)


def test_pc_opt_in_preserves_baseline_initialization_rng_and_state_dict_schema():
    common = dict(
        sd_noise=True,
        use_subsig=False,
        residual_predictor=False,
        buffer_size=64,
        batch_size=4,
        fused_adam=False,
        gpu_replay=False,
        torch_compile=False,
    )
    writer = SimpleNamespace(add_scalar=lambda *args, **kwargs: None)
    torch.manual_seed(71)
    baseline = LeSALEAgent(
        4, 3, 1.0, Args(pc_actor=False, **common), torch.device("cpu"), writer
    )
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(71)
    pc = LeSALEAgent(
        4, 3, 1.0, Args(pc_actor=True, **common), torch.device("cpu"), writer
    )
    pc_rng = torch.get_rng_state().clone()

    assert Args().pc_actor is False
    torch.testing.assert_close(pc_rng, baseline_rng, rtol=0, atol=0)
    for baseline_module, pc_module in (
        (baseline.actor, pc.actor),
        (baseline.critic, pc.critic),
        (baseline.encoder, pc.encoder),
    ):
        assert baseline_module.state_dict().keys() == pc_module.state_dict().keys()
        for name, baseline_tensor in baseline_module.state_dict().items():
            torch.testing.assert_close(
                pc_module.state_dict()[name], baseline_tensor, rtol=0, atol=0
            )


def test_local_actor_optimizer_has_no_implicit_decay():
    torch.manual_seed(8)
    actor = SDNoiseActor(2, 1, zs_dim=2, hdim=3, head_seed=17)
    trainer = TD7PCActorTrainer(actor, pc_args(hidden_dim=3, zs_dim=2))
    with torch.no_grad():
        actor.l0.weight.fill_(2.0)
        actor.l0.bias.fill_(3.0)
    zero_directions = [
        torch.zeros(layer.weight.shape[0], layer.weight.shape[1] + 1)
        for layer in trainer.layers
    ]
    trainer.optimizer.step(zero_directions, learning_rate=0.01)
    torch.testing.assert_close(actor.l0.weight, torch.full_like(actor.l0.weight, 2.0))
    torch.testing.assert_close(actor.l0.bias, torch.full_like(actor.l0.bias, 3.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for compile parity")
def test_cuda_compiled_and_eager_pc_actor_directions_match():
    torch.manual_seed(9)
    eager_args = pc_args(hidden_dim=8, zs_dim=5, pc_actor_inference_steps=2)
    compiled_args = pc_args(
        hidden_dim=8,
        zs_dim=5,
        pc_actor_inference_steps=2,
        torch_compile=True,
    )
    actor = SDNoiseActor(4, 3, zs_dim=5, hdim=8, head_seed=18).cuda()
    state = torch.randn(4, 4, device="cuda")
    zs = torch.randn(4, 5, device="cuda")
    force = torch.randn(4, 6, device="cuda")
    eager = TD7PCActorTrainer(actor, eager_args).settle_and_directions(
        state, zs, force
    )[0]
    compiled = TD7PCActorTrainer(actor, compiled_args).settle_and_directions(
        state, zs, force
    )[0]
    for actual, reference in zip(compiled, eager):
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)
