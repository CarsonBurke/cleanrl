import gymnasium as gym
import numpy as np
import torch

from cleanrl.ppo_continuous_action_streaming_bp_tdlam_control_v1 import (
    Agent,
    Args,
    EligibilityTrace,
    ExactPerEnvironmentJacobians,
    FlatParameterLayout,
    OnlineKLController,
    apply_actor_update_with_kl_limit,
    beta_log_prob,
    bootstrap_observations,
)


class DummyVecEnv:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


def make_small_system(num_envs=2):
    args = Args(
        num_envs=num_envs,
        hidden_size=4,
        num_hidden_layers=2,
        compile=False,
        cuda=False,
    )
    agent = Agent(DummyVecEnv(), args)
    actor_layout = FlatParameterLayout(agent.actor)
    critic_layout = FlatParameterLayout(agent.critic)
    jacobians = ExactPerEnvironmentJacobians(agent.actor, agent.critic, args)
    return agent, actor_layout, critic_layout, jacobians


def actor_logprob(agent, observation, action_z):
    alpha, beta = agent.actor(observation.unsqueeze(0))
    return beta_log_prob(alpha.squeeze(0), beta.squeeze(0), action_z).sum()


def test_each_environment_has_an_independent_exact_actor_trace():
    torch.manual_seed(0)
    agent, actor_layout, _, jacobians = make_small_system()
    observations = torch.randn(2, 3)
    actions_a = torch.tensor([[0.2, 0.7], [0.3, 0.6]])
    actions_b = actions_a.clone()
    actions_b[1] = torch.tensor([0.8, 0.1])

    gradients_a = actor_layout.flatten_batched(jacobians.actor_score(observations, actions_a))
    gradients_b = actor_layout.flatten_batched(jacobians.actor_score(observations, actions_b))

    assert torch.allclose(gradients_a[0], gradients_b[0])
    assert not torch.allclose(gradients_a[1], gradients_b[1])

    trace = EligibilityTrace(2, actor_layout.numel, torch.device("cpu"))
    trace.accumulate(gradients_a, decay=0.9)
    trace.accumulate(gradients_b, decay=0.9)
    expected = 0.9 * gradients_a + gradients_b
    assert torch.allclose(trace.value, expected)


def test_current_eligibility_is_added_before_td_modulation_and_reset_afterward():
    trace = EligibilityTrace(2, 3, torch.device("cpu"))
    trace.value.copy_(torch.tensor([[2.0, 2.0, 2.0], [7.0, 7.0, 7.0]]))
    current = torch.tensor([[3.0, 3.0, 3.0], [5.0, 5.0, 5.0]])

    trace.accumulate(current, decay=0.5)
    direction = trace.modulated_mean(torch.tensor([2.0, -1.0]))

    # e_0 = .5*2+3 = 4; e_1 = .5*7+5 = 8.5. Mean(delta_i*e_i) = -0.25.
    assert torch.allclose(trace.value[0], torch.full((3,), 4.0))
    assert torch.allclose(trace.value[1], torch.full((3,), 8.5))
    assert torch.allclose(direction, torch.full((3,), -0.25))

    trace.reset(torch.tensor([True, False]))
    assert torch.count_nonzero(trace.value[0]) == 0
    assert torch.allclose(trace.value[1], torch.full((3,), 8.5))


def test_truncation_bootstraps_final_observation_but_termination_mask_is_separate():
    next_obs = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], dtype=np.float32)
    truncations = np.array([False, True, False])
    infos = {
        "final_observation": np.array(
            [None, np.array([2.0, 3.0], dtype=np.float32), None],
            dtype=object,
        ),
        "_final_observation": np.array([False, True, False]),
    }

    bootstrap = bootstrap_observations(next_obs, truncations, infos)

    np.testing.assert_array_equal(bootstrap[0], next_obs[0])
    np.testing.assert_array_equal(bootstrap[1], np.array([2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(bootstrap[2], next_obs[2])


def test_actor_score_jacobian_matches_centered_finite_difference_and_ascent_sign():
    torch.manual_seed(1)
    agent, actor_layout, _, jacobians = make_small_system(num_envs=1)
    observation = torch.tensor([[0.2, -0.4, 0.7]])
    action_z = torch.tensor([[0.25, 0.8]])
    exact = actor_layout.flatten_batched(jacobians.actor_score(observation, action_z))[0]

    parameter = actor_layout.parameters[0]
    flat_index = actor_layout.slices[actor_layout.names[0]].start
    original = parameter.view(-1)[0].item()
    epsilon = 1e-3
    with torch.no_grad():
        parameter.view(-1)[0] = original + epsilon
    plus = actor_logprob(agent, observation[0], action_z[0]).item()
    with torch.no_grad():
        parameter.view(-1)[0] = original - epsilon
    minus = actor_logprob(agent, observation[0], action_z[0]).item()
    with torch.no_grad():
        parameter.view(-1)[0] = original
    finite_difference = (plus - minus) / (2.0 * epsilon)

    assert np.isclose(exact[flat_index].item(), finite_difference, rtol=2e-2, atol=2e-4)
    before = actor_logprob(agent, observation[0], action_z[0]).item()
    actor_layout.add_flat_(exact, step_size=1e-4)
    after = actor_logprob(agent, observation[0], action_z[0]).item()
    assert after > before


def test_critic_value_jacobian_matches_finite_difference_and_positive_td_increases_value():
    torch.manual_seed(2)
    agent, _, critic_layout, jacobians = make_small_system(num_envs=1)
    observation = torch.tensor([[0.1, 0.5, -0.3]])
    exact = critic_layout.flatten_batched(jacobians.critic_value(observation))[0]

    parameter = critic_layout.parameters[-1]
    parameter_name = critic_layout.names[-1]
    local_index = parameter.numel() - 1
    flat_index = critic_layout.slices[parameter_name].start + local_index
    original = parameter.view(-1)[local_index].item()
    epsilon = 1e-3
    with torch.no_grad():
        parameter.view(-1)[local_index] = original + epsilon
    plus = agent.critic(observation).item()
    with torch.no_grad():
        parameter.view(-1)[local_index] = original - epsilon
    minus = agent.critic(observation).item()
    with torch.no_grad():
        parameter.view(-1)[local_index] = original
    finite_difference = (plus - minus) / (2.0 * epsilon)

    assert np.isclose(exact[flat_index].item(), finite_difference, rtol=2e-2, atol=2e-4)
    before = agent.critic(observation).item()
    critic_layout.add_flat_(exact, step_size=1e-4)
    after = agent.critic(observation).item()
    assert after > before


def test_actor_update_bisection_enforces_hard_current_state_kl():
    torch.manual_seed(3)
    agent, actor_layout, _, jacobians = make_small_system(num_envs=2)
    observations = torch.tensor([[0.2, -0.4, 0.7], [-0.1, 0.6, 0.3]])
    action_z = torch.tensor([[0.05, 0.95], [0.9, 0.1]])
    with torch.no_grad():
        old_alpha, old_beta = agent.actor(observations)
    direction = actor_layout.flatten_batched(jacobians.actor_score(observations, action_z)).mean(0)
    before = actor_layout.flat_parameters().clone()

    accepted_kl, accepted_scale, was_limited = apply_actor_update_with_kl_limit(
        agent.actor,
        actor_layout,
        observations,
        old_alpha,
        old_beta,
        direction,
        base_step_size=1.0,
        proposed_scale=torch.tensor(1.0),
        max_kl=1e-4,
        bisection_steps=16,
    )

    assert was_limited
    assert 0.0 < accepted_scale < 1.0
    assert accepted_kl.item() <= 1e-4
    assert not torch.equal(before, actor_layout.flat_parameters())


def test_kl_controller_matches_pc_relative_error_rule():
    controller = OnlineKLController(
        target=0.003,
        adaptation_rate=0.05,
        scale_min=0.05,
        scale_max=2.0,
        device=torch.device("cpu"),
    )
    controller.observe(torch.tensor(0.0))
    assert torch.allclose(controller.scale, controller.scale.new_tensor(np.exp(0.05)), atol=1e-7)
    controller.observe(torch.tensor(0.009))
    assert torch.allclose(controller.scale, controller.scale.new_tensor(np.exp(-0.05)), atol=1e-7)
