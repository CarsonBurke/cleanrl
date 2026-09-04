import importlib.util
import inspect
import sys
from pathlib import Path

import gymnasium as gym
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl" / "embedding-optimization" / "ppo_continuous_action_intact_ppo_v17.py"
)
SPEC = importlib.util.spec_from_file_location("intact_ppo_v17", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DummyEnvs:
    single_observation_space = gym.spaces.Box(
        low=-float("inf"), high=float("inf"), shape=(17,)
    )
    single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,))


def make_agent():
    return MODULE.Agent(DummyEnvs(), MODULE.Args(actor_latent_dim=16))


def clear_gradients(agent):
    for parameter in agent.parameters():
        parameter.grad = None


def has_nonzero_gradient(module):
    return any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in module.parameters()
    )


def has_any_gradient(module):
    return any(parameter.grad is not None for parameter in module.parameters())


def test_defaults_match_requested_halfcheetah_benchmark():
    args = MODULE.Args()
    assert args.env_id == "HalfCheetah-v4"
    assert args.total_timesteps == 8_000_000
    assert args.num_envs == 16 and args.num_steps == 128
    assert args.cuda and args.seed == 1
    assert args.actor_latent_dim == 64


def test_actor_mean_uses_exact_intact_four_slot_grammar():
    actor = MODULE.INTACTActorMean(17, 6, 16)
    observation = torch.randn(5, 17)
    previous_action = torch.randn(5, 6)
    mean = actor(observation, previous_action)
    assert mean.shape == (5, 6)
    source = inspect.getsource(MODULE.INTACTActorMean.forward)
    assert "z = self.encoder(observation)" in source
    assert "intent = self.intent_prescriber(z)" in source
    assert "[z, intent, z * intent, embedded_previous_action]" in source


def test_actor_mean_is_state_and_previous_action_dependent():
    torch.manual_seed(2)
    actor = MODULE.INTACTActorMean(17, 6, 16)
    observation = torch.randn(5, 17)
    previous_action = torch.randn(5, 6)
    baseline = actor(observation, previous_action)
    changed_state = actor(observation.roll(1, dims=0), previous_action)
    changed_previous = actor(observation, torch.zeros_like(previous_action))
    assert (baseline - changed_state).abs().sum() > 0
    assert (baseline - changed_previous).abs().sum() > 0


def test_previous_action_is_zeroed_only_for_reset_environments():
    previous = torch.tensor(
        [[1.0, -1.0], [0.25, 0.5], [-0.5, 0.75]]
    )
    done = torch.tensor([False, True, False])
    actual = MODULE.reset_previous_actions(previous, done)
    expected = previous.clone()
    expected[1] = 0.0
    torch.testing.assert_close(actual, expected)


def test_action_api_uses_previous_action_and_standard_gaussian_outputs():
    agent = make_agent()
    observation = torch.randn(7, 17)
    previous = torch.randn(7, 6)
    fixed_action = torch.randn(7, 6)
    action, logprob, entropy, value = agent.get_action_and_value(
        observation, previous, fixed_action
    )
    torch.testing.assert_close(action, fixed_action)
    assert logprob.shape == entropy.shape == (7,)
    assert value.shape == (7, 1)
    assert torch.isfinite(logprob).all() and torch.isfinite(entropy).all()


def test_ppo_policy_loss_gradients_actor_but_not_separate_critic():
    agent = make_agent()
    observation = torch.randn(32, 17)
    previous = torch.randn(32, 6)
    fixed_action = torch.randn(32, 6)
    _, logprob, entropy, _ = agent.get_action_and_value(
        observation, previous, fixed_action
    )
    ppo_actor_loss = -logprob.mean() - 0.01 * entropy.mean()
    ppo_actor_loss.backward()
    assert has_nonzero_gradient(agent.actor_mean)
    assert has_nonzero_gradient(agent.actor_mean.encoder)
    assert has_nonzero_gradient(agent.actor_mean.intent_prescriber)
    assert has_nonzero_gradient(agent.actor_mean.previous_action_embedding)
    assert has_nonzero_gradient(agent.actor_mean.direct_action_operator)
    assert agent.actor_logstd.grad is not None
    assert not has_any_gradient(agent.critic)


def test_value_loss_gradients_critic_but_not_actor():
    agent = make_agent()
    value_loss = agent.get_value(torch.randn(32, 17)).square().mean()
    value_loss.backward()
    assert has_nonzero_gradient(agent.critic)
    assert not has_any_gradient(agent.actor_mean)
    assert agent.actor_logstd.grad is None


def test_rollout_stores_and_minibatches_previous_actions():
    source = SCRIPT.read_text()
    for required in (
        "previous_actions = torch.zeros_like(actions)",
        "previous_actions[step] = next_previous_action",
        "next_previous_action = action",
        "b_previous_actions = previous_actions.reshape(",
        "b_previous_actions[mb_inds]",
    ):
        assert required in source
    assert "reset_previous_actions(\n                next_previous_action, next_done" in source
    rollout_start = source.index("for step in range(0, args.num_steps):")
    action_start = source.index("# ALGO LOGIC: action logic", rollout_start)
    rollout_prefix = source[rollout_start:action_start]
    assert rollout_prefix.index("reset_previous_actions(") < rollout_prefix.index(
        "previous_actions[step] = next_previous_action"
    )


def test_training_source_is_standard_ppo_without_auxiliary_objectives():
    source = SCRIPT.read_text().lower()
    for required in (
        "pg_loss1 = -mb_advantages * ratio",
        "pg_loss2 = -mb_advantages * torch.clamp",
        "entropy_loss = entropy.mean()",
        "v_loss * args.vf_coef",
        "advantages[t] = lastgaelam",
    ):
        assert required in source
    for forbidden in (
        "world_model",
        "intact_nll",
        "auxiliary_loss",
        "physical_intent",
        "goal_intent",
        "planner",
        "sample_cem",
        "rti_action",
        "distillation",
    ):
        assert forbidden not in source


def test_compile_support_compiles_only_action_value_callables_with_graph_boundaries():
    source = SCRIPT.read_text()
    assert "action_value_function = torch.compile(" in source
    assert "value_function = torch.compile(" in source
    assert "torch.compiler.cudagraph_mark_step_begin()" in source
    assert "action = action.clone()" in source


def test_saved_policy_evaluation_preserves_previous_action_and_latent_width():
    source = inspect.getsource(MODULE.evaluate_intact_policy)
    assert "Args(actor_latent_dim=actor_latent_dim)" in source
    assert "previous_action = reset_previous_actions(previous_action, done)" in source
    assert "obs_tensor, previous_action" in source
    assert "previous_action = action" in source
    training_source = SCRIPT.read_text()
    assert "episodic_returns = evaluate_intact_policy(" in training_source
    assert "actor_latent_dim=args.actor_latent_dim" in training_source
