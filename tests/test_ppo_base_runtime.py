"""Baseline Beta PPO interface and fixed-work numerical contracts.

Model/loss/checkpoint tests require CUDA and must run through mlq. CPU tests
inspect interfaces or exercise scalar GAE/storage only; no simulator is used.
"""

import ast
import copy
import inspect
import io
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.distributions import Beta, Distribution

from cleanrl import ppo_continuous_action as ppo
from cleanrl.shared.ppo_loop import get_gae_fn
from cleanrl.shared.runtime import configure_runtime
from cleanrl.shared.sampling import sample_beta_actions


def _spaces(low=None, high=None):
    low = np.array([-3.0, -0.25, 1.0], dtype=np.float32) if low is None else np.asarray(low, dtype=np.float32)
    high = np.array([-1.0, 4.0, 9.0], dtype=np.float32) if high is None else np.asarray(high, dtype=np.float32)
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32),
        single_action_space=gym.spaces.Box(low, high, dtype=np.float32),
    )


def test_base_defaults_and_public_evaluation_interfaces():
    args = ppo.Args()
    assert (args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) == (1, 2048, 32, 10)
    assert (args.learning_rate, args.max_grad_norm, args.gamma, args.gae_lambda) == (3e-4, 0.5, 0.99, 0.95)
    assert args.staggered_starts is True  # applied only with parallel environments
    assert list(inspect.signature(ppo.make_env).parameters) == ["env_id", "idx", "capture_video", "run_name", "gamma"]
    assert list(inspect.signature(ppo.Agent).parameters) == ["envs"]
    assert list(inspect.signature(ppo.Agent.get_action_and_value).parameters) == ["self", "x", "action"]
    assert "rollout_tail_value" in inspect.signature(ppo.compute_gae).parameters


def test_base_kl_stop_uses_last_minibatch_only_after_complete_epoch():
    tree = ast.parse(inspect.getsource(ppo.main))
    epochs = [node for node in ast.walk(tree) if isinstance(node, ast.For)
              and isinstance(node.target, ast.Name) and node.target.id == "epoch"]
    assert len(epochs) == 1
    epoch = epochs[0]
    minibatches = [node for node in epoch.body if isinstance(node, ast.For)]
    checks = [node for node in epoch.body if isinstance(node, ast.If)
              and any(isinstance(value, ast.Attribute) and value.attr == "target_kl"
                      for value in ast.walk(node.test))]
    assert len(minibatches) == len(checks) == 1
    assert epoch.body.index(checks[0]) > epoch.body.index(minibatches[0])
    assert not any(isinstance(node, ast.Break) for node in ast.walk(minibatches[0]))
    assert len(checks[0].body) == 1 and isinstance(checks[0].body[0], ast.Break)
    condition = compile(ast.Expression(checks[0].test), "ppo epoch KL check", "eval")
    metrics = np.zeros((3, 6), dtype=np.float32)
    metrics[:2, 4] = 1000.0
    namespace = dict(args=SimpleNamespace(target_kl=0.03), update_metrics=metrics, updates=3)
    assert not eval(condition, namespace), "earlier minibatch KL must not replace the original last-minibatch check"
    metrics[2, 4] = 0.04
    assert eval(condition, namespace)
    namespace["args"].target_kl = None
    assert not eval(condition, namespace)


@pytest.mark.parametrize("num_envs, staggered, expected_horizon", [(1, True, 0), (16, True, 1000), (16, False, 0)])
def test_base_staggering_preserves_single_env_start_and_charges_parallel_warmup(num_envs, staggered, expected_horizon):
    tree = ast.parse(inspect.getsource(ppo.main))
    statements = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(
        (isinstance(target, ast.Name) and target.id == "horizon") or
        (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
         and target.value.id == "args" and target.attr == "num_iterations")
        for target in node.targets)]
    assert len(statements) == 2
    calls = []

    def horizon(env_id):
        calls.append(env_id)
        return 1000

    args = SimpleNamespace(num_envs=num_envs, staggered_starts=staggered, env_id="test-env",
                           total_timesteps=1_000_000, batch_size=num_envs * 2048)
    namespace = dict(args=args, episode_horizon=horizon)
    exec(compile(ast.Module(body=statements, type_ignores=[]), "ppo phase budget", "exec"), namespace)
    assert namespace["horizon"] == expected_horizon
    assert calls == (["test-env"] if expected_horizon else [])
    assert args.num_iterations == (args.total_timesteps - expected_horizon * num_envs) // args.batch_size


def test_public_gae_keywords_preserve_termination_and_truncation_semantics():
    rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    values = torch.full((2, 2), 0.5)
    terms = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    truncs = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    advantages, returns = ppo.compute_gae(
        rewards=rewards, values=values, terminations=terms, truncations=truncs,
        truncation_bootstrap_values=torch.full((2, 2), 10.0),
        rollout_tail_value=torch.zeros(2), gamma=0.99, gae_lambda=0.95,
    )
    expected = torch.tensor([[0.5, 10.4], [1.5, 1.5]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected + values)


def test_public_bootstrap_observations_preserve_input_and_require_valid_finals():
    observations = np.array([[1.0, 2.0], [3.0, 4.0]])
    before = observations.copy()
    finals = np.empty(2, dtype=object)
    finals[:] = [None, np.array([7.0, 8.0])]
    result = ppo.bootstrap_observations(observations, [False, True], {
        "final_observation": finals, "_final_observation": np.array([False, True]),
    })
    np.testing.assert_array_equal(result, [[1.0, 2.0], [7.0, 8.0]])
    np.testing.assert_array_equal(observations, before)
    assert not np.shares_memory(result, observations)
    with pytest.raises(RuntimeError, match="final_observation"):
        ppo.bootstrap_observations(observations, [False, True], {})


@pytest.mark.parametrize("failure", [None, "load", "reset", "policy", "step"])
def test_evaluator_closes_environment_and_disables_grad_without_changing_sampling_api(monkeypatch, failure):
    from cleanrl_utils.evals import ppo_eval

    class FakeEnv:
        closed = 0
        steps = 0

        def reset(self):
            if failure == "reset":
                raise RuntimeError("reset failure")
            return np.zeros((1, 11), dtype=np.float32), {}

        def step(self, actions):
            if failure == "step":
                raise RuntimeError("step failure")
            assert actions.shape == (1, 3)
            self.steps += 1
            return np.zeros((1, 11)), None, None, None, {
                "final_info": [None, {"episode": {"r": 7.0 + self.steps}}],
            }

        def close(self):
            self.closed += 1

    env = FakeEnv()
    checkpoint = {"beta_head": "stand-in weights"}
    factory_calls = []
    actions = torch.zeros((1, 3))

    class FakeModel:
        def __init__(self, actual_env):
            assert actual_env is env

        def to(self, device):
            assert device == "cpu"
            return self

        def load_state_dict(self, state):
            assert state is checkpoint

        def eval(self):
            return self

        def get_action_and_value(self, observations, action=None):
            assert observations.shape == (1, 11) and observations.dtype == torch.float32
            assert not torch.is_grad_enabled()
            assert action is None  # still stochastic policy sampling, not a mean action
            if failure == "policy":
                raise RuntimeError("policy failure")
            return actions, None, None, None

    def make_env(*args):
        factory_calls.append(args)
        return lambda: env

    def load(path, *, map_location):
        assert path == "weights.cleanrl_model" and map_location == "cpu"
        if failure == "load":
            raise RuntimeError("load failure")
        return checkpoint

    monkeypatch.setattr(ppo_eval.gym.vector, "SyncVectorEnv", lambda thunks: thunks[0]())
    monkeypatch.setattr(ppo_eval.torch, "load", load)
    kwargs = dict(model_path="weights.cleanrl_model", make_env=make_env, env_id="test-env",
                  eval_episodes=2, run_name="evaluation", Model=FakeModel, device="cpu",
                  capture_video=False, gamma=0.97)
    if failure is None:
        assert ppo_eval.evaluate(**kwargs) == [8.0, 9.0]
    else:
        with pytest.raises(RuntimeError, match=f"{failure} failure"):
            ppo_eval.evaluate(**kwargs)
    assert factory_calls == [("test-env", 0, False, "evaluation", 0.97)]
    assert env.closed == 1


def test_evaluator_cli_requires_explicit_local_beta_checkpoint_and_uses_cuda(monkeypatch):
    from cleanrl_utils.evals import ppo_eval

    monkeypatch.setattr(Path, "is_file", lambda path: str(path) == "beta.cleanrl_model")
    monkeypatch.setattr(ppo_eval.torch.cuda, "is_available", lambda: True)
    calls = []

    def evaluate(*args, **kwargs):
        calls.append((args, kwargs))
        return [17.0]

    monkeypatch.setattr(ppo_eval, "evaluate", evaluate)
    assert ppo_eval.main(["--model-path", "beta.cleanrl_model"]) == [17.0]
    args, kwargs = calls[0]
    assert args == ("beta.cleanrl_model", ppo.make_env, "HalfCheetah-v4")
    assert kwargs == dict(eval_episodes=10, run_name="ppo_beta_eval", Model=ppo.Agent,
                          device=torch.device("cuda"), capture_video=False, gamma=0.99)
    assert ppo_eval.main(["--model-path", "beta.cleanrl_model", "--env-id", "Hopper-v4",
                          "--eval-episodes", "3", "--run-name", "saved", "--gamma", "0.97",
                          "--capture-video"]) == [17.0]
    assert calls[1][0][-1] == "Hopper-v4"
    assert calls[1][1]["eval_episodes"] == 3 and calls[1][1]["capture_video"] is True


@pytest.mark.parametrize("argv", [[], ["--model-path", "missing"],
    ["--model-path", "beta.cleanrl_model", "--eval-episodes", "0"],
    ["--model-path", "beta.cleanrl_model", "--gamma", "nan"],
])
def test_evaluator_cli_rejects_invalid_inputs_before_any_cuda_query(monkeypatch, argv):
    from cleanrl_utils.evals import ppo_eval

    monkeypatch.setattr(Path, "is_file", lambda path: str(path) == "beta.cleanrl_model")

    def forbidden():
        pytest.fail("invalid command must fail before querying CUDA")

    monkeypatch.setattr(ppo_eval.torch.cuda, "is_available", forbidden)
    with pytest.raises(SystemExit) as error:
        ppo_eval.main(argv)
    assert error.value.code == 2


def test_evaluator_cli_rejects_cpu_fallback(monkeypatch):
    from cleanrl_utils.evals import ppo_eval

    monkeypatch.setattr(Path, "is_file", lambda path: True)
    monkeypatch.setattr(ppo_eval.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        ppo_eval.main(["--model-path", "beta.cleanrl_model"])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="queued CUDA test required")
@pytest.mark.parametrize("low, high", [([-np.inf], [1.0]), ([-1.0], [np.inf]), ([1.0], [1.0])])
def test_beta_policy_requires_finite_nonempty_action_intervals(low, high):
    with pytest.raises(ValueError, match="(?i)(finite|bound|positive|interval)"):
        ppo.Agent(_spaces(low, high)).cuda()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="queued CUDA test required")
def test_beta_policy_sampling_bounds_jacobians_rng_and_checkpoint_roundtrip():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    envs = _spaces()
    agent = ppo.Agent(envs).cuda()
    observations = torch.randn(64, 11, device="cuda")
    validation_before = Distribution._validate_args
    low, high = (torch.as_tensor(value, device="cuda") for value in
                 (envs.single_action_space.low, envs.single_action_space.high))
    scale = high - low
    with torch.no_grad():
        alpha, beta, value = agent.get_policy_and_value(observations)
        assert alpha.shape == beta.shape == (64, 3)
        assert (alpha >= 1).all() and (beta >= 1).all()
        assert torch.isfinite(alpha).all() and torch.isfinite(beta).all()
        distribution = Beta(alpha, beta, validate_args=False)
        rng = torch.cuda.get_rng_state()
        native = distribution.sample().clamp(1e-6, 1.0 - 1e-6)
        expected_actions = low + scale * native
        expected_rng = torch.cuda.get_rng_state()
        expected_logprob = (distribution.log_prob(native) - scale.log()).sum(1)
        expected_entropy = (distribution.entropy() + scale.log()).sum(1)
        torch.cuda.set_rng_state(rng)
        actions, logprob, entropy, actual_value = agent.get_action_and_value(observations)
        assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
        assert actions.shape == (64, 3) and logprob.shape == entropy.shape == (64,)
        assert ((actions >= low) & (actions <= high)).all()
        for actual, expected in ((actions, expected_actions), (logprob, expected_logprob),
                                 (entropy, expected_entropy), (actual_value, value)):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        # Supplied physical actions use the inverse affine map, without sampling.
        given = low + scale * torch.linspace(0.1, 0.9, 64, device="cuda")[:, None]
        before = torch.cuda.get_rng_state()
        actual_action, given_logprob, _, _ = agent.get_action_and_value(observations, given)
        expected = (distribution.log_prob((given - low) / scale) - scale.log()).sum(1)
        torch.testing.assert_close(actual_action, given, rtol=0, atol=0)
        torch.testing.assert_close(given_logprob, expected, rtol=1e-6, atol=1e-6)
        assert torch.equal(torch.cuda.get_rng_state(), before)
    assert Distribution._validate_args is validation_before

    # Keep the evaluator's bare-state-dict serialization and Model(envs) API.
    checkpoint = io.BytesIO()
    torch.save(agent.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = ppo.Agent(envs).cuda()
    restored.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    assert "actor_logstd" not in restored.state_dict()
    restored.eval()
    with torch.no_grad():
        for actual, expected in zip(restored.get_policy_and_value(observations), (alpha, beta, value)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="queued CUDA test required")
def test_compiled_beta_inference_keeps_sampling_outside_graph_and_preserves_rng():
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    envs = _spaces()
    agent = ppo.Agent(envs).cuda()
    observations = torch.randn(32, 11, device="cuda")
    low, high = (torch.as_tensor(value, device="cuda") for value in
                 (envs.single_action_space.low, envs.single_action_space.high))
    compiled = torch.compile(agent.get_policy_and_value, mode="reduce-overhead", fullgraph=True)
    with torch.no_grad():
        expected = agent.get_policy_and_value(observations)
        rng = torch.cuda.get_rng_state()
        for _ in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            actual = tuple(value.clone() for value in compiled(observations))
        assert torch.equal(torch.cuda.get_rng_state(), rng)
        for candidate, reference in zip(actual, expected):
            torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-6)
        expected_native, expected_physical = sample_beta_actions(*actual[:2], low, high)
        after = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(rng)
        # Same compiled parameters, same PyTorch sampler: compilation itself
        # must not consume draws or capture stochastic policy sampling.
        torch.compiler.cudagraph_mark_step_begin()
        alpha, beta, _ = compiled(observations)
        native, physical = sample_beta_actions(alpha, beta, low, high)
        torch.testing.assert_close(native, expected_native, rtol=0, atol=0)
        torch.testing.assert_close(physical, expected_physical, rtol=0, atol=0)
        assert torch.equal(torch.cuda.get_rng_state(), after)


def _reference_loss(agent, observations, native_actions, old_logprobs, advantages, returns, old_values, args, scale):
    alpha, beta, value = agent.get_policy_and_value(observations)
    distribution = Beta(alpha, beta, validate_args=False)
    new_logprobs = (distribution.log_prob(native_actions) - scale.log()).sum(1)
    entropy = (distribution.entropy() + scale.log()).sum(1)
    logratio = new_logprobs - old_logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        old_kl = (-logratio).mean()
        kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1).abs() > args.clip_coef).float().mean()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    policy_loss = torch.max(-advantages * ratio,
                            -advantages * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
    value = value.view(-1)
    if args.clip_vloss:
        clipped = old_values + (value - old_values).clamp(-args.clip_coef, args.clip_coef)
        value_loss = 0.5 * torch.max((value - returns).square(), (clipped - returns).square()).mean()
    else:
        value_loss = 0.5 * (value - returns).square().mean()
    entropy_loss = entropy.mean()
    loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * value_loss
    return loss, torch.stack((policy_loss, value_loss, entropy_loss, old_kl, kl, clipfrac))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="queued CUDA test required")
@pytest.mark.parametrize("norm_adv", [False, True])
@pytest.mark.parametrize("clip_vloss", [False, True])
def test_compiled_beta_ppo_loss_gradients_and_clipped_adam_match_reference(norm_adv, clip_vloss):
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    envs = _spaces()
    reference = ppo.Agent(envs).cuda()
    candidate = copy.deepcopy(reference)
    args = ppo.Args(norm_adv=norm_adv, clip_vloss=clip_vloss, ent_coef=0.03)
    observations = torch.randn(64, 11, device="cuda")
    native = 0.1 + 0.8 * torch.rand(64, 3, device="cuda")
    scale = torch.as_tensor(envs.single_action_space.high - envs.single_action_space.low, device="cuda")
    with torch.no_grad():
        alpha, beta, values = reference.get_policy_and_value(observations)
        logprobs = (Beta(alpha, beta, validate_args=False).log_prob(native) - scale.log()).sum(1)
        offsets = torch.linspace(-1.0, 1.0, 64, device="cuda")
        old_logprobs = logprobs - offsets
        advantages = 2.0 + 3.0 * offsets
        # A positive offset keeps a substantial critic-bias gradient in the
        # unclipped half even when both policy and value clipping are enabled.
        returns = values.view(-1) + 8.0 + 3.0 * offsets
        old_values = values.view(-1) - 0.8 * offsets
    inputs = (observations, native, old_logprobs, advantages, returns, old_values)
    expected_loss, expected_metrics = _reference_loss(reference, *inputs, args, scale)
    compiled = torch.compile(lambda *batch: ppo.ppo_loss(candidate, *batch, args), fullgraph=True,
                             options={"triton.cudagraphs": False})
    actual_loss, actual_metrics = compiled(*inputs)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_metrics, expected_metrics, rtol=2e-5, atol=2e-6)
    assert expected_metrics[-1] > 0.5, "fixture must exercise policy clipping"
    expected_loss.backward()
    actual_loss.backward()
    for expected, actual in zip(reference.parameters(), candidate.parameters()):
        assert expected.grad is not None and actual.grad is not None
        assert torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, expected.grad, rtol=4e-5, atol=3e-6)
    assert any(torch.count_nonzero(parameter.grad) for name, parameter in candidate.named_parameters()
               if "actor" in name)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=args.learning_rate, eps=1e-5, fused=False)
    candidate_optimizer = torch.optim.Adam(candidate.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
    for agent, optimizer in ((reference, reference_optimizer), (candidate, candidate_optimizer)):
        norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm, foreach=True)
        assert norm > args.max_grad_norm, "fixture must exercise gradient clipping"
        optimizer.step()
    for expected, actual in zip(reference.parameters(), candidate.parameters()):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
        for name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(candidate_optimizer.state[actual][name],
                                       reference_optimizer.state[expected][name], rtol=4e-5, atol=3e-7)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="queued CUDA test required")
def test_beta_production_compiled_graphs_interoperate_across_three_rollout_updates():
    """Fixed compute ownership replay, not a shortened training experiment.

Use the SAME model in all live compiled callables with the production compile
options. Copy rollout outputs before advancing graph generations. No simulator
or manual capture is involved; sampling stays outside the compiled policy.
    """
    configure_runtime(matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    args = ppo.Args(num_envs=16, num_steps=4, num_minibatches=1, update_epochs=2)
    agent = ppo.Agent(_spaces()).cuda()
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, fused=True)
    policy_model = torch.compile(agent.get_policy_and_value, mode="reduce-overhead", fullgraph=True, dynamic=False)
    logprob_model = torch.compile(agent.action_logprob, mode="reduce-overhead", fullgraph=True, dynamic=False)
    value_model = torch.compile(agent.get_value, fullgraph=True, dynamic=True,
                                options={"triton.cudagraphs": False})
    loss_model = torch.compile(lambda *batch: ppo.ppo_loss(agent, *batch, args),
                               mode="reduce-overhead", fullgraph=True, dynamic=False)
    gae_fn = get_gae_fn(compiled=True, mode="reduce-overhead")
    fixtures = torch.randn(3, 5, 16, 11, device="cuda")
    rewards = torch.randn(4, 16, device="cuda")
    terms = torch.zeros(4, 16, device="cuda")
    truncs = torch.zeros_like(terms)
    terms[1, 0] = 1
    truncs[2, 1:3] = 1
    observations = torch.empty(4, 16, 11, device="cuda")
    native_actions = torch.empty(4, 16, 3, device="cuda")
    values, old_logprobs = torch.empty_like(rewards), torch.empty_like(rewards)
    next_obs = torch.empty(16, 11, device="cuda")
    update_snapshots = torch.empty(2, 6, device="cuda")
    before = tuple(parameter.detach().clone() for parameter in agent.parameters())
    for round_index in range(3):
        with torch.no_grad():
            for step in range(4):
                next_obs.copy_(fixtures[round_index, step])
                torch.compiler.cudagraph_mark_step_begin()
                alpha, beta, value = policy_model(next_obs)
                native, physical = sample_beta_actions(alpha, beta, agent.action_low, agent.action_high)
                logprob = logprob_model(alpha, beta, native)
                observations[step].copy_(next_obs)
                native_actions[step].copy_(native)
                values[step].copy_(value.flatten())
                old_logprobs[step].copy_(logprob)
                assert torch.isfinite(physical).all()
            next_obs.copy_(fixtures[round_index, -1])
            torch.compiler.cudagraph_mark_step_begin()
            tail = value_model(next_obs).flatten()
            bootstrap = torch.zeros_like(rewards)
            bootstrap[2, 1:3] = value_model(fixtures[round_index, 3, 1:3]).flatten()
            advantages, returns = gae_fn(rewards, values, terms, truncs, bootstrap,
                                          tail, args.gamma, args.gae_lambda)
            b_advantages, b_returns = advantages.flatten().clone(), returns.flatten().clone()
        batch = (observations.flatten(0, 1), native_actions.flatten(0, 1), old_logprobs.flatten(),
                 b_advantages, b_returns, values.flatten())
        sampler_rng = torch.cuda.get_rng_state()
        for epoch in range(2):
            torch.compiler.cudagraph_mark_step_begin()
            loss, metrics = loss_model(*batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
            update_snapshots[epoch].copy_(metrics)
        # Attribute graph/ownership faults to this round, not another GPU test.
        torch.cuda.synchronize()
        assert torch.equal(torch.cuda.get_rng_state(), sampler_rng), "learner consumed policy sampling RNG"
        assert torch.isfinite(update_snapshots).all()
        assert all(torch.isfinite(parameter).all() for parameter in agent.parameters())
    assert any(not torch.equal(initial, final) for initial, final in zip(before, agent.parameters()))
