import copy
import importlib.util
import inspect
import sys
from pathlib import Path

import gymnasium as gym
import pytest
import torch
import torch.nn.functional as F
from torch.distributions import Beta


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "cleanrl/tpo/md/ppo_continuous_action_tpomd_thinktrunk_tdpc_gem_v3.py"
V5_SCRIPT = (
    ROOT
    / "cleanrl/iterthink/v24_d4hlgauss/rawret/ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_idxtransfer_tpomd_v5_dyntrust.py"
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PC = _load("tpomd_thinktrunk_tdpc_gem_v3", SCRIPT)
V5 = _load("pure_tpo_v5_reference_for_tdpc", V5_SCRIPT)


class _DummyEnvs:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(5,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,))


class _HalfCheetahShapes:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(17,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(6,))


def _args(*, hidden=4, blocks=3, experts=2, bins=7, candidates=4):
    return PC.Args(
        hidden=hidden,
        k_blocks=blocks,
        n_experts=experts,
        num_bins=bins,
        tpo_k=candidates,
        share_backbone=True,
        actor_dist="beta",
    )


def _agent(*, seed=193, hidden=4, blocks=3, experts=2, bins=7):
    torch.manual_seed(seed)
    return PC.Agent(
        _DummyEnvs(),
        _args(hidden=hidden, blocks=blocks, experts=experts, bins=bins),
    )


def _problem(agent, *, batch=24, candidates=4, actor_noise=0.01, critic_noise=0.03):
    generator = torch.Generator().manual_seed(277)
    observations = torch.randn(batch, 5, generator=generator)
    free = PC.free_dag_activities(agent, observations)
    candidate_zs = torch.rand(
        batch, candidates, 2, generator=generator
    ).clamp(0.03, 0.97)
    actor_logits = PC.beta_candidate_logits(
        free.actor_alpha_raw, free.actor_beta_raw, candidate_zs
    )
    q_targets = torch.softmax(
        actor_logits
        + actor_noise
        * torch.randn(actor_logits.shape, generator=generator),
        dim=-1,
    )
    value_targets = torch.softmax(
        free.critic_logits
        + critic_noise
        * torch.randn(free.critic_logits.shape, generator=generator),
        dim=-1,
    )
    distribution = Beta(
        1.0 + F.softplus(free.actor_alpha_raw),
        1.0 + F.softplus(free.actor_beta_raw),
    )
    with torch.random.fork_rng():
        torch.manual_seed(347)
        rollout_latents = distribution.sample().clamp(
            PC.SAMPLE_EPS, 1.0 - PC.SAMPLE_EPS
        )
    rollout_logprobs = distribution.log_prob(rollout_latents).sum(-1)
    return (
        observations,
        candidate_zs,
        q_targets,
        value_targets,
        rollout_latents,
        rollout_logprobs,
    )


def _settled_statistics(agent, problem, *, steps=10):
    observations, candidates, q_targets, value_targets, *_ = problem
    result = PC._settle_dag_core(
        agent, observations, candidates, q_targets, value_targets, steps
    )
    statistics = PC.empty_dag_statistics(agent, observations.device)
    PC.accumulate_dag_statistics(agent, statistics, observations, result.activities)
    return result, statistics


def test_defaults_freeze_the_ten_by_ten_shared_v5_intervention():
    args = PC.Args()
    assert args.env_id == "HalfCheetah-v4"
    assert args.total_timesteps == 8_000_000
    assert args.seed == 1 and args.cuda
    assert args.share_backbone and args.actor_dist == "beta"
    assert (args.hidden, args.k_blocks, args.n_experts, args.num_bins) == (
        64,
        3,
        16,
        511,
    )
    assert args.pc_inference_steps == PC.PC_INFERENCE_STEPS == 10
    assert args.outer_cycles == PC.OUTER_CYCLES == 10
    assert args.pc_chunk_size == 512


def test_default_halfcheetah_state_and_parameter_contract_is_exact():
    torch.manual_seed(101)
    agent = PC.Agent(_HalfCheetahShapes(), PC.Args())
    assert sum(parameter.numel() for parameter in agent.parameters()) == 510_012
    assert len(tuple(agent.named_parameters())) == 279
    assert len(agent.state_dict()) == 281
    assert agent.critic_head.bias is None
    assert torch.count_nonzero(agent.critic_head.weight) == 0
    assert not any(parameter.requires_grad for parameter in agent.parameters())
    assert tuple(agent.action_low.shape) == tuple(agent.action_high.shape) == (6,)


def test_v5_state_init_rng_and_free_trace_are_bitwise_exact():
    pc_args = _args()
    v5_args = V5.Args(
        hidden=4,
        k_blocks=3,
        n_experts=2,
        num_bins=7,
        share_backbone=True,
        actor_dist="beta",
    )
    torch.manual_seed(113)
    agent = PC.Agent(_DummyEnvs(), pc_args)
    pc_rng = torch.rand(9)
    torch.manual_seed(113)
    reference = V5.Agent(_DummyEnvs(), v5_args)
    v5_rng = torch.rand(9)
    assert tuple(agent.state_dict()) == tuple(reference.state_dict())
    for name, tensor in agent.state_dict().items():
        torch.testing.assert_close(tensor, reference.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(pc_rng, v5_rng, rtol=0, atol=0)

    observations = torch.randn(11, 5)
    free = PC.free_dag_activities(agent, observations)
    torch.testing.assert_close(free.trunk, reference.trunk(observations), rtol=0, atol=0)
    torch.testing.assert_close(
        free.critic_logits, reference.get_value(observations), rtol=0, atol=0
    )
    residuals = PC.dag_residuals(
        free, PC.dag_predictions(agent, observations, free)
    )
    assert all(torch.count_nonzero(residual) == 0 for residual in residuals)


def test_v5_action_value_candidate_api_is_bitwise_exact():
    torch.manual_seed(131)
    agent = PC.Agent(_DummyEnvs(), _args())
    torch.manual_seed(131)
    reference = V5.Agent(
        _DummyEnvs(),
        V5.Args(
            hidden=4,
            k_blocks=3,
            n_experts=2,
            num_bins=7,
            share_backbone=True,
            actor_dist="beta",
        ),
    )
    observations = torch.randn(8, 5)
    z = torch.rand(8, 2).clamp(0.02, 0.98)
    candidates = torch.rand(8, 5, 2).clamp(0.02, 0.98)
    actual = agent.get_action_and_value(observations, z, candidates)
    expected = reference.get_action_and_value(observations, z, candidates)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)

    torch.manual_seed(139)
    sampled_actual = agent.get_action_and_value(observations, return_dist=True)
    actual_rng = torch.rand(7)
    torch.manual_seed(139)
    sampled_expected = reference.get_action_and_value(observations, return_dist=True)
    expected_rng = torch.rand(7)
    for left, right in zip(sampled_actual[:5], sampled_expected[:5], strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)


def test_rmsnorm_vjp_and_row_gram_match_jacrev():
    torch.manual_seed(149)
    x = torch.randn(3, 6, dtype=torch.float64)
    cotangent = torch.randn_like(x)
    output_gram = torch.randn(6, 6, dtype=torch.float64)
    output_gram = output_gram.T @ output_gram
    epsilon = 1e-7
    actual_vjp = PC.rmsnorm_vjp(x, cotangent, epsilon)
    jacobians = torch.vmap(
        torch.func.jacrev(
            lambda row: row / (row.square().mean() + epsilon).sqrt()
        )
    )(x)
    expected_vjp = torch.einsum("bij,bj->bi", jacobians.transpose(-1, -2), cotangent)
    expected_gram = torch.einsum(
        "bij,jk,bkl->bil", jacobians.transpose(-1, -2), output_gram, jacobians
    )
    torch.testing.assert_close(actual_vjp, expected_vjp, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(
        PC.rmsnorm_row_gram(x, output_gram, epsilon),
        expected_gram,
        rtol=2e-11,
        atol=2e-11,
    )


def test_relu_squared_branch_message_and_exact_gn_match_jacrev():
    agent = _agent()
    block = agent.trunk.blocks[0]
    branch_count = 1 + len(block.experts)
    pre = torch.randn(5, branch_count, 4, dtype=torch.float64)
    own_error = torch.randn_like(pre)
    child_error = torch.randn_like(pre)
    block = copy.deepcopy(block).to(torch.float64)
    gradient, preconditioner = PC._branch_pre_gradient_and_preconditioner(
        block, pre, own_error, child_error
    )
    expected_gradients = []
    expected_grams = []
    for row in range(pre.shape[0]):
        row_gradients, row_grams = [], []
        for branch_index, branch in enumerate(PC._branch_modules(block)):
            jacobian = torch.func.jacrev(
                lambda value: branch.out_linear(torch.relu(value).square())
            )(pre[row, branch_index])
            row_gradients.append(
                own_error[row, branch_index] - jacobian.T @ child_error[row, branch_index]
            )
            row_grams.append(torch.eye(4, dtype=torch.float64) + jacobian.T @ jacobian)
        expected_gradients.append(torch.stack(row_gradients))
        expected_grams.append(torch.stack(row_grams))
    torch.testing.assert_close(gradient, torch.stack(expected_gradients), rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(preconditioner, torch.stack(expected_grams), rtol=2e-11, atol=2e-11)


def test_sigmoid_residual_mix_pair_message_matches_autograd():
    torch.manual_seed(167)
    projection = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    residual_logit = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    entry = torch.randn(4, 5, dtype=torch.float64)
    child = torch.randn(4, 5, dtype=torch.float64)
    projection_error = torch.randn(4, 5, dtype=torch.float64)
    residual_error = torch.randn(4, 5, dtype=torch.float64)
    gate = torch.sigmoid(residual_logit)
    prediction = gate * projection + (1.0 - gate) * entry
    child_error = child - prediction.detach()
    objective = (
        (projection * projection_error).sum()
        + (residual_logit * residual_error).sum()
        + 0.5 * (child - prediction).square().sum()
    )
    expected_projection, expected_residual = torch.autograd.grad(
        objective, (projection, residual_logit)
    )
    residual_direction = gate.detach() * (1.0 - gate.detach()) * (
        projection.detach() - entry
    )
    actual = torch.stack((projection_error, residual_error), dim=1) - torch.stack(
        (gate.detach(), residual_direction), dim=1
    ) * child_error.unsqueeze(1)
    torch.testing.assert_close(actual[:, 0], expected_projection, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(actual[:, 1], expected_residual, rtol=2e-12, atol=2e-12)


def test_softmax_mixture_message_and_gn_match_explicit_jacobian():
    torch.manual_seed(181)
    logits = torch.randn(4, 3, dtype=torch.float64)
    outputs = torch.randn(4, 3, 5, dtype=torch.float64)
    child_error = torch.randn(4, 5, dtype=torch.float64)
    message, gram = PC.softmax_mixture_message_and_gn(logits, outputs, child_error)
    expected_message, expected_gram = [], []
    for row in range(logits.shape[0]):
        jacobian = torch.func.jacrev(
            lambda value: (torch.softmax(value, -1).unsqueeze(-1) * outputs[row]).sum(0)
        )(logits[row])
        expected_message.append(jacobian.T @ child_error[row])
        expected_gram.append(jacobian.T @ jacobian)
    torch.testing.assert_close(message, torch.stack(expected_message), rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(gram, torch.stack(expected_gram), rtol=2e-12, atol=2e-12)


def test_indexed_history_design_uses_only_same_channel_and_joint_covariance():
    torch.manual_seed(197)
    current = torch.randn(6, 4, dtype=torch.float64)
    history = torch.randn(6, 3, 4, dtype=torch.float64)
    design = PC.joint_indexed_design(current, history)
    assert design.shape == (6, 4, 8)
    for channel in range(4):
        torch.testing.assert_close(design[:, channel, :4], current, rtol=0, atol=0)
        torch.testing.assert_close(
            design[:, channel, 4:7], history[:, :, channel], rtol=0, atol=0
        )
        assert torch.equal(design[:, channel, -1], torch.ones(6, dtype=torch.float64))


def test_concat_history_gradient_equals_sum_of_every_fanout_message():
    agent = _agent(hidden=4, blocks=3, experts=2)
    observations = torch.randn(5, 5)
    free = PC.free_dag_activities(agent, observations)
    perturbed = PC.DAGActivities(
        *(value + 0.01 * torch.randn_like(value) for value in free)
    )
    predictions = PC.dag_predictions(agent, observations, perturbed)
    residuals = PC.dag_residuals(perturbed, predictions)
    values = PC._activity_lists(perturbed)
    errors = PC._activity_lists(residuals)
    main = values[0]
    for main_index in range(4):
        actual = PC._history_gradient(
            agent,
            main,
            values[2],
            errors[0],
            errors[1],
            errors[3],
            errors[4],
            errors[7],
            main_index,
        )
        variable = main[main_index].detach().clone().requires_grad_(True)
        changed = list(main)
        changed[main_index] = variable
        objective = 0.5 * (variable - predictions.main[:, main_index]).square().sum()
        hidden = variable.shape[-1]
        for later in range(main_index, 3):
            block = agent.trunk.blocks[later]
            history = torch.cat(changed[: later + 1], dim=-1)
            objective = objective + 0.5 * (
                values[1][later] - block.in_proj(history)
            ).square().sum()
            objective = objective + 0.5 * (
                values[4][later]
                - PC._block_pre_predictions(block, values[3][later], history)
            ).square().sum()
        if main_index == 0:
            for block_index in range(3):
                gate = torch.sigmoid(values[2][block_index])
                mix = gate * values[1][block_index] + (1.0 - gate) * variable
                objective = objective + 0.5 * (values[3][block_index] - mix).square().sum()
        final_history = torch.cat(changed, dim=-1)
        objective = objective + 0.5 * (
            values[7]
            - agent.trunk.out_proj(agent.trunk.out_norm(final_history))
        ).square().sum()
        expected = torch.autograd.grad(objective, variable)[0]
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


def test_joint_indexed_and_biasless_critic_cplus_match_explicit_pseudoinverse():
    agent = _agent(hidden=4, blocks=1, experts=2, bins=7)
    block = agent.trunk.blocks[0]
    device = torch.device("cpu")
    joint = PC._empty_joint_branch_statistics(3, 4, 1, device)
    torch.manual_seed(211)
    current = torch.randn(19, 4)
    history = torch.randn(19, 1, 4)
    residual = torch.randn(19, 3, 4)
    joint = PC._accumulate_joint_branch_statistics(joint, current, history, residual)
    corrections, before, after, *_ = PC.joint_indexed_m_step(block, joint)
    design = PC.joint_indexed_design(current, history).double()
    stopped = residual.double().permute(0, 2, 1)
    for channel in range(4):
        centered_x = design[:, channel, :-1] - design[:, channel, :-1].mean(0)
        centered_y = stopped[:, channel] - stopped[:, channel].mean(0)
        coefficient = centered_y.T @ centered_x @ torch.linalg.pinv(
            centered_x.T @ centered_x, hermitian=True
        )
        for branch_index in range(3):
            name = "dense" if branch_index == 0 else f"experts.{branch_index - 1}"
            torch.testing.assert_close(
                corrections[f"{name}.current_linear.weight"][channel].double(),
                coefficient[branch_index, :4],
                rtol=2e-5,
                atol=2e-6,
            )
    assert torch.all(after <= before + 2e-6 * (1 + before.abs()))

    critic = torch.nn.Linear(4, 7, bias=False)
    stats = PC._empty_affine_statistics(critic, device)
    features = torch.randn(23, 4)
    critic_residual = torch.randn(23, 7)
    stats = PC._accumulate_affine_statistics(stats, features, critic_residual)
    correction, before, after, *_ = PC._affine_m_step(critic, stats)
    expected = critic_residual.double().T @ features.double() @ torch.linalg.pinv(
        features.double().T @ features.double(), hermitian=True
    )
    torch.testing.assert_close(correction["weight"].double(), expected, rtol=2e-5, atol=2e-6)
    assert after <= before + 2e-6 * (1 + before.abs())


def test_zero_task_is_exact_noop_for_activities_and_all_279_style_corrections():
    agent = _agent()
    observations = torch.randn(12, 5)
    candidates = torch.rand(12, 4, 2).clamp(0.03, 0.97)
    free = PC.free_dag_activities(agent, observations)
    q_targets = torch.softmax(
        PC.beta_candidate_logits(
            free.actor_alpha_raw, free.actor_beta_raw, candidates
        ),
        dim=-1,
    )
    value_targets = torch.softmax(free.critic_logits, dim=-1)
    result = PC._settle_dag_core(
        agent, observations, candidates, q_targets, value_targets, 10
    )
    for actual, expected in zip(result.activities, free, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    statistics = PC.empty_dag_statistics(agent, observations.device)
    PC.accumulate_dag_statistics(agent, statistics, observations, result.activities)
    m_step = PC.dag_m_step(agent, statistics)
    assert set(m_step.corrections) == set(dict(agent.named_parameters()))
    assert all(torch.count_nonzero(value) == 0 for value in m_step.corrections.values())


def test_mstep_covers_every_named_parameter_once_and_never_increases_local_sse():
    agent = _agent()
    problem = _problem(agent)
    _, statistics = _settled_statistics(agent, problem)
    result = PC.dag_m_step(agent, statistics)
    assert tuple(result.corrections) != ()
    assert set(result.corrections) == set(dict(agent.named_parameters()))
    assert len(result.corrections) == len(tuple(agent.named_parameters()))
    assert all(result.corrections[name].shape == parameter.shape for name, parameter in agent.named_parameters())
    tolerance = 2e-6 * (1.0 + result.sse_before.abs())
    assert torch.all(result.sse_after <= result.sse_before + tolerance)
    assert set(result.factor_rank_fraction) == set(result.factor_condition) == set(
        result.factor_correction_norm
    )
    torch.testing.assert_close(
        torch.stack(tuple(result.factor_rank_fraction.values())).min(),
        result.min_rank_fraction,
    )


def test_cycle_two_propagates_new_zero_critic_head_signal_into_shared_hidden_graph():
    agent = _agent()
    observations, candidates, q_targets, _, latents, logprobs = _problem(
        agent, actor_noise=0.0, critic_noise=0.0
    )
    free = PC.free_dag_activities(agent, observations)
    value_targets = F.one_hot(
        torch.arange(observations.shape[0]) % agent.num_bins,
        agent.num_bins,
    ).float()
    cycle_one = PC.propose_dag_cycle(
        agent,
        observations,
        candidates,
        q_targets,
        value_targets,
        latents,
        logprobs,
        observations.shape[0],
        lambda o, c, q, v: PC._settle_dag_core(agent, o, c, q, v, 10),
        0,
    )
    assert cycle_one.diagnostics.critic_terminal_message_rms == 0
    assert torch.count_nonzero(cycle_one.corrections["critic_head.weight"])
    PC.apply_atomic_dag_corrections(agent, cycle_one.corrections)
    cycle_two = PC.propose_dag_cycle(
        agent,
        observations,
        candidates,
        q_targets,
        value_targets,
        latents,
        logprobs,
        observations.shape[0],
        lambda o, c, q, v: PC._settle_dag_core(agent, o, c, q, v, 10),
        1,
    )
    assert cycle_two.diagnostics.critic_terminal_message_rms > 0
    assert cycle_two.diagnostics.factor_correction_norm["trunk.entry"] > 0
    PC.apply_atomic_dag_corrections(agent, cycle_two.corrections)
    assert not torch.equal(free.trunk, PC.free_dag_activities(agent, observations).trunk)


def test_hostile_k10_and_k50_exact_gn_remain_finite_without_spectral_divergence():
    args = _args(candidates=8)
    torch.manual_seed(222)
    agent = PC.Agent(_DummyEnvs(), args)
    torch.manual_seed(1901)
    observations = torch.randn(32, 5)
    free = PC.free_dag_activities(agent, observations)
    distribution = Beta(
        1 + F.softplus(free.actor_alpha_raw),
        1 + F.softplus(free.actor_beta_raw),
    )
    candidates = distribution.sample((8,)).permute(1, 0, 2).clamp(
        PC.SAMPLE_EPS, 1 - PC.SAMPLE_EPS
    )
    anchor = PC.beta_candidate_logits(
        free.actor_alpha_raw, free.actor_beta_raw, candidates
    )
    q_targets = PC.build_tpo_target(
        anchor, torch.randn(32, 8) * 3, torch.tensor(1.0), args
    ).probabilities
    value_targets = F.one_hot(torch.randint(7, (32,)), 7).float()
    result_10 = PC._settle_dag_core(
        agent, observations, candidates, q_targets, value_targets, 10
    )
    result_50 = PC._settle_dag_core(
        agent, observations, candidates, q_targets, value_targets, 50
    )
    for result in (result_10, result_50):
        assert all(torch.isfinite(value).all() for value in result.activities)
        assert torch.isfinite(result.energies).all()
        assert torch.isfinite(result.stationarity_rms).all()
        assert result.energies[-1] <= result.energies[0] * 1.001
    assert result_50.stationarity_rms[-1] < 2e-3


def test_fixed_ten_outer_cycles_use_immutable_targets_fresh_weights_and_match_explicit_helper(
    monkeypatch,
):
    run_agent = _agent()
    manual_agent = copy.deepcopy(run_agent)
    problem = _problem(
        run_agent,
        batch=48,
        candidates=4,
        actor_noise=0.02,
        critic_noise=0.0,
    )
    observations, candidates, q_targets, value_targets, latents, logprobs = problem
    q_copy = q_targets.clone()
    value_copy = value_targets.clone()
    target_versions = (q_targets._version, value_targets._version)
    settle_snapshots = []
    apply_calls = []
    original_apply = PC.apply_atomic_dag_corrections

    def settle(o, c, q, v):
        settle_snapshots.append(run_agent.actor_alpha_head.weight.detach().clone())
        return PC._settle_dag_core(run_agent, o, c, q, v, 10)

    def counted_apply(agent, corrections):
        apply_calls.append(tuple(corrections))
        original_apply(agent, corrections)

    monkeypatch.setattr(PC, "apply_atomic_dag_corrections", counted_apply)
    actual = PC.run_dag_outer_gem(
        run_agent,
        observations,
        candidates,
        q_targets,
        value_targets,
        latents,
        logprobs,
        24,
        settle,
    )
    assert len(actual) == len(apply_calls) == PC.OUTER_CYCLES == 10
    assert len(settle_snapshots) == 20
    for cycle in range(10):
        torch.testing.assert_close(
            settle_snapshots[2 * cycle], settle_snapshots[2 * cycle + 1], rtol=0, atol=0
        )
    assert all(
        not torch.equal(settle_snapshots[2 * cycle], settle_snapshots[2 * cycle + 2])
        for cycle in range(9)
    )
    assert (q_targets._version, value_targets._version) == target_versions
    torch.testing.assert_close(q_targets, q_copy, rtol=0, atol=0)
    torch.testing.assert_close(value_targets, value_copy, rtol=0, atol=0)
    assert len({id(cycle.statistics) for cycle in actual}) == 10
    assert all(
        actual[index].statistics is not actual[index + 1].statistics
        for index in range(9)
    )

    explicit = []
    for cycle_index in range(10):
        cycle = PC.propose_dag_cycle(
            manual_agent,
            observations,
            candidates,
            q_targets,
            value_targets,
            latents,
            logprobs,
            24,
            lambda o, c, q, v: PC._settle_dag_core(
                manual_agent, o, c, q, v, 10
            ),
            cycle_index,
        )
        original_apply(manual_agent, cycle.corrections)
        explicit.append(cycle)
    for (name, actual_parameter), (_, expected_parameter) in zip(
        run_agent.named_parameters(), manual_agent.named_parameters(), strict=True
    ):
        torch.testing.assert_close(
            actual_parameter, expected_parameter, rtol=0, atol=0, msg=lambda: name
        )
    actor_before = torch.stack(
        [cycle.diagnostics.actor_boundary_ce_before for cycle in actual]
    )
    actor_after = torch.stack(
        [cycle.diagnostics.actor_boundary_ce_after for cycle in actual]
    )
    assert torch.all(actor_after <= actor_before)
    torch.testing.assert_close(actor_after[:-1], actor_before[1:], rtol=0, atol=0)
    behavior_kl = torch.stack(
        [cycle.diagnostics.proposed_behavior_kl for cycle in actual]
    )
    assert behavior_kl[-1] > 20.0 * behavior_kl[0]
    assert torch.all(behavior_kl >= 0)


def test_canonical_h64_ten_cycle_realistic_targets_remain_finite_and_settle_each_cycle():
    agent = _agent(hidden=64, blocks=3, experts=16, bins=511)
    problem = list(
        _problem(
            agent,
            batch=16,
            candidates=8,
            actor_noise=0.02,
            critic_noise=0.0,
        )
    )
    problem[3] = F.one_hot(torch.arange(16) % 511, 511).float()
    results = PC.run_dag_outer_gem(agent, *problem, chunk_size=8)
    assert len(results) == 10
    for cycle in results:
        diagnostics = cycle.diagnostics
        assert torch.isfinite(diagnostics.energy_per_row).all()
        assert torch.isfinite(diagnostics.stationarity_rms).all()
        assert diagnostics.energy_per_row[-1] < diagnostics.energy_per_row[0]
        assert diagnostics.stationarity_rms[-1] < 1e-3
    assert results[0].diagnostics.critic_terminal_message_rms == 0
    assert results[1].diagnostics.critic_terminal_message_rms > 0


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_real_cpu_inductor_fullgraph_single_sweep_observes_live_parameter_mutation(
    monkeypatch,
):
    torch._dynamo.reset()
    monkeypatch.setattr(torch._inductor.config, "compile_threads", 1)
    agent = _agent(hidden=1, blocks=1, experts=1, bins=2)
    observations = torch.randn(1, 5)
    candidates = torch.rand(1, 2, 2).clamp(0.05, 0.95)
    free = PC.free_dag_activities(agent, observations)
    q_targets = torch.softmax(
        PC.beta_candidate_logits(
            free.actor_alpha_raw, free.actor_beta_raw, candidates
        ),
        dim=-1,
    )
    value_targets = torch.softmax(free.critic_logits, dim=-1)
    compiled = torch.compile(
        lambda o, c, q, v: PC._settle_dag_core(agent, o, c, q, v, 1),
        backend="inductor",
        fullgraph=True,
        dynamic=False,
    )
    first = compiled(observations, candidates, q_targets, value_targets)
    with torch.no_grad():
        agent.actor_alpha_head.bias.add_(0.05)
    second = compiled(observations, candidates, q_targets, value_targets)
    assert torch.isfinite(first.energies).all()
    assert torch.isfinite(second.energies).all()
    assert not torch.equal(
        first.activities.actor_alpha_raw, second.activities.actor_alpha_raw
    )


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_fixed_k10_settlement_captures_as_one_fullgraph_and_keeps_live_weights():
    torch._dynamo.reset()
    agent = _agent(hidden=1, blocks=1, experts=1, bins=2)
    observations = torch.randn(1, 5)
    candidates = torch.rand(1, 2, 2).clamp(0.05, 0.95)
    free = PC.free_dag_activities(agent, observations)
    q_targets = torch.softmax(
        PC.beta_candidate_logits(
            free.actor_alpha_raw, free.actor_beta_raw, candidates
        ),
        dim=-1,
    )
    value_targets = torch.softmax(free.critic_logits, dim=-1)
    compiled = torch.compile(
        lambda o, c, q, v: PC._settle_dag_core(
            agent, o, c, q, v, PC.PC_INFERENCE_STEPS
        ),
        backend="eager",
        fullgraph=True,
        dynamic=False,
    )
    first = compiled(observations, candidates, q_targets, value_targets)
    assert first.energies.shape == first.stationarity_rms.shape == (11,)
    with torch.no_grad():
        agent.actor_beta_head.bias.sub_(0.04)
    second = compiled(observations, candidates, q_targets, value_targets)
    assert not torch.equal(
        first.activities.actor_beta_raw, second.activities.actor_beta_raw
    )


def test_small_logratio_behavior_kl_is_nonnegative_and_quadratic():
    logratio = torch.tensor([-2e-6, -1e-6, 0.0, 1e-6, 2e-6], dtype=torch.float32)
    precise = logratio.double()
    actual = torch.expm1(precise) - precise
    expected = 0.5 * precise.square()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-18)
    assert torch.all(actual >= 0)
    assert actual[-1] / actual[-2] == pytest.approx(4.0, rel=2e-5)


def test_source_has_no_reference_import_optimizer_or_hidden_update_control():
    source = SCRIPT.read_text()
    assert V5_SCRIPT.name not in source
    assert "local_pcopt_outer_gem_v2" not in source
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "autograd.grad" not in source
    assert "clip_grad" not in source
    assert "tpo_kl_breaker" not in source
    for forbidden in (
        "target_network",
        "target_net",
        "rollback",
        "learning_rate",
        "weight_decay",
        "ExponentialMovingAverage",
    ):
        assert forbidden not in source
    assert 'torch.set_float32_matmul_precision("high")' in source
    atomic_source = inspect.getsource(PC.apply_atomic_dag_corrections)
    assert "parameter.add_(corrections[name])" in atomic_source


def test_v5_and_v2_are_off_the_production_import_path():
    assert Path(PC.__file__).resolve() == SCRIPT.resolve()
    source = SCRIPT.read_text()
    assert "importlib" not in source
    assert "runpy" not in source
    assert "pure_tpo_v5_reference_for_tdpc" not in sys.modules or PC is not V5
