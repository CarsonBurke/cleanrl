from types import SimpleNamespace

import gymnasium as gym
import torch
import torch.nn.functional as F

import cleanrl.pc.ppo_continuous_action_pc_fisher_dreamer_retnorm_twohot_v13 as v13


def make_agent(**overrides):
    values = dict(
        hidden_size=8,
        pc_num_hidden_layers=2,
        pc_inference_steps=2,
        pc_inference_scale=1.0,
        compile=False,
    )
    values.update(overrides)
    args = v13.Args(**values)
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-1.0, 1.0, (4,), dtype=float),
        single_action_space=gym.spaces.Box(-1.0, 1.0, (2,), dtype=float),
    )
    return v13.Agent(envs, args), args


def test_exact_dreamer_raw_symexp_support_and_uniform_decode():
    bins = v13.dreamer_twohot_bins()
    assert bins.shape == (255,)
    torch.testing.assert_close(bins, -bins.flip(0), rtol=0, atol=0)
    assert bins[127].item() == 0.0
    torch.testing.assert_close(
        bins[0], -torch.expm1(torch.tensor(20.0)), rtol=0, atol=0
    )
    logits = torch.zeros(7, 255)
    assert torch.equal(v13.twohot_decode(logits, bins), torch.zeros(7))


def test_twohot_targets_reconstruct_raw_values_and_saturate_only_at_support():
    bins = v13.dreamer_twohot_bins()
    values = torch.tensor(
        [bins[0] * 2, -10_000.0, -1.0, 0.0, 1.0, 10_000.0, bins[-1] * 2]
    )
    targets = v13.twohot_encode(values, bins)
    torch.testing.assert_close(targets.sum(dim=-1), torch.ones(7))
    assert (targets >= 0).all()
    reconstruction = (targets * bins).sum(dim=-1)
    torch.testing.assert_close(
        reconstruction, values.clamp(bins[0], bins[-1]), rtol=2e-6, atol=1e-5
    )
    assert (targets > 0).sum(dim=-1).max() <= 2


def test_ce_force_is_exact_negative_cross_entropy_logit_gradient():
    torch.manual_seed(3)
    bins = v13.dreamer_twohot_bins()
    logits = torch.randn(5, 255, requires_grad=True)
    values = torch.tensor([-1000.0, -0.5, 0.0, 2.0, 20_000.0])
    force, targets, probabilities = v13.twohot_ce_force(logits, values, bins)
    objective = (targets.detach() * F.log_softmax(logits, dim=-1)).sum()
    exact = torch.autograd.grad(objective, logits)[0]
    torch.testing.assert_close(force, exact, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(probabilities, logits.softmax(dim=-1))
    torch.testing.assert_close(force.sum(dim=-1), torch.zeros(5), atol=2e-6, rtol=0)


def test_zero_initialized_critic_head_decodes_exactly_zero():
    agent, _ = make_agent()
    assert torch.count_nonzero(agent.critic_output.weight) == 0
    assert torch.count_nonzero(agent.critic_output.bias) == 0
    values = agent.get_value(torch.randn(6, 4))
    assert torch.equal(values, torch.zeros(6))


def test_critic_uses_one_identity_pc_settle_and_exact_free_head_direction(monkeypatch):
    torch.manual_seed(9)
    agent, args = make_agent()
    observations = torch.randn(5, 4)
    td_target = torch.tensor([-40.0, -1.0, 0.0, 3.0, 120.0])
    free_states = agent.critic_pc.initial_states(observations)
    free_logits = agent.critic_output(free_states[-1])
    force, _, _ = v13.twohot_ce_force(free_logits, td_target, agent.critic_bins)
    expected_head = force.unsqueeze(2) * agent.critic_output.augmented_features(
        free_states[-1]
    ).unsqueeze(1)

    calls = []
    original_settle = agent.critic_pc.settle

    def recording_settle(*settle_args, **settle_kwargs):
        calls.append(settle_args[3])
        return original_settle(*settle_args, **settle_kwargs)

    monkeypatch.setattr(agent.critic_pc, "settle", recording_settle)
    _, _, head_direction, _ = agent.settle_critic(
        observations, td_target, args, collect_diagnostics=False
    )

    assert calls == [None]
    torch.testing.assert_close(head_direction, expected_head, rtol=0, atol=0)


def test_identity_endpoint_uses_exact_w_transpose_w_hidden_curvature():
    torch.manual_seed(12)
    agent, args = make_agent()
    with torch.no_grad():
        agent.critic_output.weight.normal_(0.0, 0.02)
    states = agent.critic_pc.initial_states(torch.randn(5, 4))
    factors, _, _ = agent.critic_pc.curvature_factors(
        states, agent.critic_output, None, args, collect_diagnostics=False
    )
    actual = factors[-1] @ factors[-1].T
    eye = torch.eye(args.hidden_size)
    expected = (
        (1.0 + args.pc_curvature_damping) * eye
        + agent.critic_output.weight.T @ agent.critic_output.weight
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_immediate_critic_directions_have_no_trace_or_scalar_td_modulation():
    scores = [
        torch.arange(24, dtype=torch.float32).reshape(3, 2, 4),
        torch.arange(18, dtype=torch.float32).reshape(3, 2, 3),
    ]
    directions = v13.immediate_critic_directions(scores, vf_coef=0.5)
    for actual, score in zip(directions, scores):
        torch.testing.assert_close(actual, 0.5 * score.mean(dim=0), rtol=0, atol=0)


def test_actor_td_modulation_remains_raw_target_dreamer_range_scaling():
    norm = v13.RunningReturnRange(
        torch.device("cpu"), rate=1.0, limit=1.0, perclo=0.0, perchi=100.0
    )
    reward = torch.tensor([1.0, 4.0])
    terminated = torch.tensor([False, True])
    next_value = torch.tensor([10.0, 50.0])
    value = torch.tensor([2.0, 3.0])
    target, error, actor_delta, _, scale = v13.compute_td_modulations(
        reward, terminated, next_value, value, 0.9, norm, actor_clip=10.0
    )
    torch.testing.assert_close(target, torch.tensor([10.0, 4.0]))
    torch.testing.assert_close(error, torch.tensor([8.0, 1.0]))
    torch.testing.assert_close(scale, torch.tensor(6.0))
    torch.testing.assert_close(actor_delta, error / scale)


def test_terminal_target_ignores_nonfinite_bootstrap():
    norm = v13.RunningReturnRange(
        torch.device("cpu"), rate=1.0, limit=1.0, perclo=0.0, perchi=100.0
    )
    target, error, actor_delta, _, _ = v13.compute_td_modulations(
        reward=torch.tensor([3.0]),
        terminated=torch.tensor([True]),
        next_value=torch.tensor([float("nan")]),
        value=torch.tensor([1.0]),
        gamma=0.99,
        retnorm=norm,
        actor_clip=10.0,
    )
    torch.testing.assert_close(target, torch.tensor([3.0]))
    torch.testing.assert_close(error, torch.tensor([2.0]))
    torch.testing.assert_close(actor_delta, torch.tensor([2.0]))
