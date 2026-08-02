import importlib.util
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import torch


ROOT = Path(__file__).parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v4 = load_module(
    "spk_packed_coordinate_v4_for_v5_test",
    ROOT / "cleanrl" / "sparse-nn" / "ppo_continuous_action_spk_packed_coordinate_v4.py",
)
v5 = load_module(
    "spk_counterfactual_rewire_v5",
    ROOT
    / "cleanrl"
    / "sparse-nn"
    / "ppo_continuous_action_spk_counterfactual_rewire_v5.py",
)


def make_env():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-1.0, 1.0, shape=(17,)),
        single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(6,)),
    )


def make_agent(seed=1):
    args = v5.Args(pool="prior", rewire="counterfactual", compile=False)
    torch.manual_seed(seed)
    return v5.Agent(make_env(), args), args


def test_v5_initial_agent_matches_effective_v4_control():
    torch.manual_seed(10)
    control = v4.Agent(
        make_env(), v4.Args(pool="prior", rewire="none", weight_coordinate="effective")
    )
    torch.manual_seed(10)
    treatment = v5.Agent(
        make_env(), v5.Args(pool="prior", rewire="counterfactual", weight_coordinate="effective")
    )
    assert torch.equal(control.actor_trunk.layers[1].indices, treatment.actor_trunk.layers[1].indices)
    assert torch.equal(control.actor_trunk.layers[1].weight, treatment.actor_trunk.layers[1].weight)
    observations = torch.randn(19, 17)
    zs = torch.rand(19, 6).clamp(v5.SAMPLE_EPS, 1.0 - v5.SAMPLE_EPS)
    for expected, actual in zip(
        control.get_beta_action_and_value(observations, zs),
        treatment.get_beta_action_and_value(observations, zs),
    ):
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_actor_dormant_gemm_matches_live_effective_gradient():
    agent, args = make_agent(11)
    time_count, env_count = 12, 4
    obs = torch.randn(time_count, env_count, 17)
    zs = torch.rand(time_count, env_count, 6).clamp(v5.SAMPLE_EPS, 1.0 - v5.SAMPLE_EPS)
    with torch.no_grad():
        old_logprobs = agent._dist(obs.reshape(-1, 17)).log_prob(zs.reshape(-1, 6)).sum(1)
    old_logprobs = old_logprobs.reshape(time_count, env_count)
    advantages = torch.randn(time_count, env_count)
    env_ids = torch.arange(env_count)
    layer = agent.actor_trunk.layers[1]

    scores = v5.actor_dense_scores(
        agent, layer, obs, zs, old_logprobs, advantages, env_ids, 1, 0, args.clip_coef
    )
    agent.zero_grad()
    flat_logprobs = agent._dist(obs.reshape(-1, 17)).log_prob(zs.reshape(-1, 6)).sum(1)
    loss = (-advantages.reshape(-1) * flat_logprobs).mean()
    loss.backward()
    dense_live = scores.gradient.mean(0).gather(1, layer.indices)

    assert torch.allclose(dense_live, layer.weight.grad, atol=2e-6, rtol=2e-5)
    assert bool((scores.curvature >= 0).all())


def test_critic_dormant_gemm_matches_live_effective_gradient():
    agent, args = make_agent(12)
    time_count, env_count = 10, 4
    obs = torch.randn(time_count, env_count, 17)
    returns = torch.randn(time_count, env_count)
    env_ids = torch.arange(env_count)
    layer = agent.critic_trunk.layers[1]
    with torch.no_grad():
        old_values = agent.get_value(obs.reshape(-1, 17)).reshape(time_count, env_count)

    scores = v5.critic_dense_scores(
        agent,
        layer,
        obs,
        returns,
        old_values,
        env_ids,
        1,
        0,
        args.vf_coef,
        args.clip_coef,
    )
    agent.zero_grad()
    values = agent.get_value(obs.reshape(-1, 17)).flatten()
    loss = 0.5 * args.vf_coef * (values - returns.reshape(-1)).square().mean()
    loss.backward()
    dense_live = scores.gradient.mean(0).gather(1, layer.indices)

    assert torch.allclose(dense_live, layer.weight.grad, atol=2e-6, rtol=2e-5)
    assert bool((scores.curvature >= 0).all())


def test_actor_clipped_constant_branch_has_zero_dormant_gradient():
    agent, args = make_agent(121)
    time_count, env_count = 8, 4
    obs = torch.randn(time_count, env_count, 17)
    zs = torch.rand(time_count, env_count, 6).clamp(v5.SAMPLE_EPS, 1.0 - v5.SAMPLE_EPS)
    with torch.no_grad():
        current = agent._dist(obs.reshape(-1, 17)).log_prob(zs.reshape(-1, 6)).sum(1)
    old_logprobs = (current - torch.log(torch.tensor(2.0))).reshape(time_count, env_count)
    advantages = torch.ones(time_count, env_count)
    scores = v5.actor_dense_scores(
        agent,
        agent.actor_trunk.layers[1],
        obs,
        zs,
        old_logprobs,
        advantages,
        torch.arange(env_count),
        1,
        0,
        args.clip_coef,
    )
    assert torch.equal(scores.gradient, torch.zeros_like(scores.gradient))


def test_critic_clipped_constant_branch_has_zero_dormant_gradient():
    agent, args = make_agent(122)
    time_count, env_count = 8, 4
    obs = torch.randn(time_count, env_count, 17)
    with torch.no_grad():
        current = agent.get_value(obs.reshape(-1, 17)).reshape(time_count, env_count)
    old_values = current - 1.0
    returns = current + 10.0
    scores = v5.critic_dense_scores(
        agent,
        agent.critic_trunk.layers[1],
        obs,
        returns,
        old_values,
        torch.arange(env_count),
        1,
        0,
        args.vf_coef,
        args.clip_coef,
    )
    assert torch.equal(scores.gradient, torch.zeros_like(scores.gradient))


def test_screening_rollout_is_not_counted_as_validation_evidence():
    torch.manual_seed(13)
    layer = v5.SparseKLinear(9, 3, 3, rewire_mode="none", weight_coordinate="effective")
    cohort = v5.ChallengerCohort(layer, challengers=2)
    gradient = torch.zeros(4, 3, 9)
    curvature = torch.ones_like(gradient)
    active = torch.zeros(3, 9, dtype=torch.bool)
    active.scatter_(1, layer.indices, True)
    for row in range(3):
        dormant = (~active[row]).nonzero(as_tuple=False).flatten()
        gradient[:, row, dormant[0]] = 10.0
    scores = v5.DenseLayerScores(gradient, curvature)
    age = torch.full_like(layer.weight, 4, dtype=torch.long)

    cohort.screen(scores, age, min_edge_age=4, damping=0.01)
    assert cohort.cluster_count == 0
    assert cohort.validation_rollouts == 0
    cohort.validate(scores, damping=0.01)
    assert cohort.cluster_count == 4
    assert cohort.validation_rollouts == 1
    assert len(cohort.proposals(confidence_z=0.0)) == 3


def test_commit_preserves_survivors_and_seeds_new_adam_variance():
    agent, args = make_agent(14)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    observations = torch.randn(32, 17)
    zs = torch.rand(32, 6).clamp(v5.SAMPLE_EPS, 1.0 - v5.SAMPLE_EPS)
    _, _, logprob, _, value = agent.get_beta_action_and_value(observations, zs)
    (-(logprob.mean()) + value.square().mean()).backward()
    optimizer.step()
    controller = v5.CounterfactualRewirer(agent, optimizer, args, torch.device("cpu"))
    layer = controller.actor_layer
    row, slot = 0, 0
    old_source = int(layer.indices[row, slot])
    active = torch.zeros(layer.in_features, dtype=torch.bool)
    active[layer.indices[row]] = True
    new_source = int((~active).nonzero(as_tuple=False)[0])
    state = optimizer.state[layer.weight]
    weight_before = layer.weight.detach().clone()
    first_before = state["exp_avg"].detach().clone()
    second_before = state["exp_avg_sq"].detach().clone()
    expected_second = torch.cat((second_before[row, :slot], second_before[row, slot + 1 :])).median()
    swap = v5.CommittedSwap(
        row,
        slot,
        old_source,
        new_source,
        float(layer.weight[row, slot].detach()),
        1.0,
        0.0,
        0.0,
    )

    controller._commit(layer, controller.actor_age, [swap])
    survivor = torch.ones_like(layer.weight, dtype=torch.bool)
    survivor[row, slot] = False
    assert torch.equal(layer.weight[survivor], weight_before[survivor])
    assert torch.equal(state["exp_avg"][survivor], first_before[survivor])
    assert torch.equal(state["exp_avg_sq"][survivor], second_before[survivor])
    assert float(layer.weight[row, slot].detach()) == 0.0
    assert float(state["exp_avg"][row, slot]) == 0.0
    assert torch.equal(state["exp_avg_sq"][row, slot], expected_second.clamp_min(1e-16))
    assert int(layer.indices[row, slot]) == new_source
    assert torch.unique(layer.indices[row]).numel() == layer.k


def test_capture_hook_is_removed_after_exception():
    layer = v5.SparseKLinear(9, 3, 3, rewire_mode="none", weight_coordinate="effective")
    try:
        with v5.capture_layer_call(layer):
            layer(torch.randn(2, 9))
            raise RuntimeError("expected")
    except RuntimeError as error:
        assert str(error) == "expected"
    else:
        raise AssertionError("exception should propagate")
    assert not layer._forward_hooks


def test_temporary_swap_restores_topology_and_weights_after_exception():
    agent, _ = make_agent(141)
    layer = agent.actor_trunk.layers[1]
    row, slot = 0, 0
    active = torch.zeros(layer.in_features, dtype=torch.bool)
    active[layer.indices[row]] = True
    new_source = int((~active).nonzero(as_tuple=False)[0])
    swap = v5.CommittedSwap(
        row,
        slot,
        int(layer.indices[row, slot]),
        new_source,
        float(layer.weight[row, slot].detach()),
        1.0,
        0.0,
        0.1,
    )
    indices_before = layer.indices.clone()
    weights_before = layer.weight.detach().clone()
    try:
        with v5.temporary_swaps(layer, [swap], fitted=True):
            assert int(layer.indices[row, slot]) == new_source
            assert torch.isclose(layer.weight[row, slot], torch.tensor(0.1))
            raise RuntimeError("expected")
    except RuntimeError as error:
        assert str(error) == "expected"
    else:
        raise AssertionError("exception should propagate")
    assert torch.equal(layer.indices, indices_before)
    assert torch.equal(layer.weight, weights_before)


def test_random_control_generator_does_not_advance_global_rng():
    agent, args = make_agent(142)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    controller = v5.CounterfactualRewirer(agent, optimizer, args, torch.device("cpu"))
    layer = controller.actor_layer
    cohort = controller.actor_cohort
    cohort.candidates = torch.stack(
        [
            (~torch.zeros(layer.in_features, dtype=torch.bool).scatter_(0, layer.indices[row], True))
            .nonzero(as_tuple=False)
            .flatten()[: cohort.challengers]
            for row in range(layer.out_features)
        ]
    )
    cohort.last_benefit = torch.ones(layer.out_features, cohort.challengers)
    cohort.last_fitted_weight = torch.zeros_like(cohort.last_benefit)
    cohort.last_deletion_ucb = torch.zeros(layer.out_features)
    proposal = v5.SwapProposal(0, 0, int(cohort.candidates[0, 0]), 1.0, 1.0, 0.0, 1.0, 0.0)
    controller.args.rewire = "random"
    before = torch.random.get_rng_state()
    controller._randomize_sources(layer, cohort, [proposal])
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)


def test_controller_screening_step_runs_without_mutating_topology():
    args = v5.Args(
        pool="prior",
        rewire="counterfactual",
        compile=False,
        num_envs=4,
        num_steps=8,
        batch_size=32,
        rewire_score_stride=2,
    )
    torch.manual_seed(143)
    agent = v5.Agent(make_env(), args)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    controller = v5.CounterfactualRewirer(agent, optimizer, args, torch.device("cpu"))
    obs = torch.randn(args.num_steps, args.num_envs, 17)
    zs = torch.rand(args.num_steps, args.num_envs, 6).clamp(v5.SAMPLE_EPS, 1.0 - v5.SAMPLE_EPS)
    with torch.no_grad():
        flat_obs = obs.reshape(-1, 17)
        flat_zs = zs.reshape(-1, 6)
        old_logprobs = agent._dist(flat_obs).log_prob(flat_zs).sum(1).reshape(
            args.num_steps, args.num_envs
        )
        old_values = agent.get_value(flat_obs).reshape(args.num_steps, args.num_envs)
    advantages = torch.randn(args.num_steps, args.num_envs)
    returns = old_values + advantages
    actor_indices = controller.actor_layer.indices.clone()
    critic_indices = controller.critic_layer.indices.clone()

    class Writer:
        def add_scalar(self, *_args, **_kwargs):
            pass

    controller.step(
        obs,
        zs,
        old_logprobs,
        advantages,
        returns,
        old_values,
        0.01,
        Writer(),
        args.batch_size,
    )
    assert not controller.actor_cohort.screening
    assert not controller.critic_cohort.screening
    assert torch.equal(controller.actor_layer.indices, actor_indices)
    assert torch.equal(controller.critic_layer.indices, critic_indices)


def test_actor_gate_restores_full_weight_tensor_after_exception():
    agent, args = make_agent(144)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    controller = v5.CounterfactualRewirer(agent, optimizer, args, torch.device("cpu"))
    before = controller.actor_layer.weight.detach().clone()

    def fail(*_args):
        with torch.no_grad():
            controller.actor_layer.weight.zero_()
        raise RuntimeError("expected")

    controller._actor_gate_impl = fail
    try:
        controller._actor_gate([], None, None, None, None, None)
    except RuntimeError as error:
        assert str(error) == "expected"
    else:
        raise AssertionError("exception should propagate")
    assert torch.equal(controller.actor_layer.weight, before)


def run_cuda_candidate_check():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA candidate check requires CUDA")
    agent, args = make_agent(15)
    agent = agent.cuda()
    time_count, env_count = 128, 4
    obs = torch.randn(time_count, env_count, 17, device="cuda")
    zs = torch.rand(time_count, env_count, 6, device="cuda").clamp(
        v5.SAMPLE_EPS, 1.0 - v5.SAMPLE_EPS
    )
    with torch.no_grad():
        old_logprobs = agent._dist(obs.reshape(-1, 17)).log_prob(zs.reshape(-1, 6)).sum(1)
    old_logprobs = old_logprobs.reshape(time_count, env_count)
    advantages = torch.randn(time_count, env_count, device="cuda")
    layer = agent.actor_trunk.layers[1]
    scores = v5.actor_dense_scores(
        agent,
        layer,
        obs,
        zs,
        old_logprobs,
        advantages,
        torch.arange(env_count, device="cuda"),
        1,
        0,
        args.clip_coef,
    )
    agent.zero_grad()
    logprob = agent._dist(obs.reshape(-1, 17)).log_prob(zs.reshape(-1, 6)).sum(1)
    (-advantages.reshape(-1) * logprob).mean().backward()
    error = (scores.gradient.mean(0).gather(1, layer.indices) - layer.weight.grad).abs().max()
    print(f"candidate_active_gradient_error={float(error):.3g}")
    assert float(error) < 1e-5


if __name__ == "__main__":
    run_cuda_candidate_check()
