import importlib.util
import inspect
import sys
from pathlib import Path

import gymnasium as gym
import pytest
import torch


ROOT = Path(__file__).parents[1]


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load(
    "tpomd_prednext_cuda_v16",
    "cleanrl/tpo/md/ppo_continuous_action_tpomd_prednext_cuda_v16.py",
)
V13 = _load(
    "tpomd_prednext_v13_reference_for_cuda_v16",
    "cleanrl/tpo/md/ppo_continuous_action_tpomd_prednext_v13.py",
)
BASE = _load(
    "tpomd_v5_dyntrust_reference",
    "cleanrl/iterthink/v24_d4hlgauss/rawret/ppo_continuous_action_iterthink_v24_beta_d4hlgauss_symlog_rawret_idxtransfer_tpomd_v5_dyntrust.py",
)


class _DummyEnvs:
    single_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(7,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))


def _args(module):
    return module.Args(hidden=8, k_blocks=2, n_experts=2, num_bins=7)


def test_auxiliary_initialization_preserves_tpomd_parameters_and_rng_stream():
    torch.manual_seed(287)
    base = BASE.Agent(_DummyEnvs(), _args(BASE))
    base_rng_after_init = torch.get_rng_state().clone()

    torch.manual_seed(287)
    composed = MODULE.Agent(_DummyEnvs(), _args(MODULE))
    composed_rng_after_init = torch.get_rng_state().clone()

    composed_state = composed.state_dict()
    for name, value in base.state_dict().items():
        torch.testing.assert_close(composed_state[name], value, rtol=0.0, atol=0.0)
    assert torch.equal(composed_rng_after_init, base_rng_after_init)


def test_default_policy_forward_is_exactly_tpomd_v5_at_initialization():
    torch.manual_seed(917)
    base = BASE.Agent(_DummyEnvs(), _args(BASE))
    torch.manual_seed(917)
    composed = MODULE.Agent(_DummyEnvs(), _args(MODULE))
    observations = torch.randn(11, 7)

    torch.manual_seed(121)
    expected = base.get_action_and_value(observations)
    torch.manual_seed(121)
    actual = composed.get_action_and_value(observations)

    assert len(actual) == len(expected)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0.0, atol=0.0)


def test_task_adam_step_is_exactly_tpomd_v5_when_auxiliary_is_not_admitted():
    torch.manual_seed(411)
    base = BASE.Agent(_DummyEnvs(), _args(BASE))
    torch.manual_seed(411)
    composed = MODULE.Agent(_DummyEnvs(), _args(MODULE))
    with torch.no_grad():
        critic_weights = torch.randn_like(base.critic_head.weight)
        base.critic_head.weight.copy_(critic_weights)
        composed.critic_head.weight.copy_(critic_weights)

    base_optimizer = torch.optim.Adam(base.parameters(), lr=3e-4, eps=1e-5)
    composed_optimizer = torch.optim.Adam(composed.task_parameters(), lr=3e-4, eps=1e-5)
    generator = torch.Generator().manual_seed(83)
    observations = torch.randn(16, 7, generator=generator)
    native_actions = torch.sigmoid(torch.randn(16, 3, generator=generator))
    candidates = torch.sigmoid(torch.randn(16, 8, 3, generator=generator))
    mirror_target = torch.softmax(torch.randn(16, 8, generator=generator), dim=-1)
    value_target = torch.softmax(torch.randn(16, 7, generator=generator), dim=-1)

    def task_step(agent, optimizer):
        actor_params = agent.actor_parameters()
        critic_params = agent.critic_parameters()
        _, _, _, entropy, value_logits, candidate_logprobs = agent.get_action_and_value(
            observations, native_actions, candidate_zs=candidates
        )
        actor_loss = (-(mirror_target * torch.log_softmax(candidate_logprobs, -1)).sum(-1)).mean()
        value_loss = (-(value_target * torch.log_softmax(value_logits, -1)).sum(-1)).mean()

        optimizer.zero_grad(set_to_none=True)
        value_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(critic_params, 0.25)
        value_grads = [
            (parameter, parameter.grad.detach().clone())
            for parameter in critic_params
            if parameter.grad is not None
        ]
        optimizer.zero_grad(set_to_none=True)
        (actor_loss - 0.0 * entropy.mean()).backward()
        torch.nn.utils.clip_grad_norm_(actor_params, 0.25)
        for parameter, gradient in value_grads:
            parameter.grad = (
                gradient if parameter.grad is None else parameter.grad + gradient
            )
        optimizer.step()

    task_step(base, base_optimizer)
    task_step(composed, composed_optimizer)

    composed_state = composed.state_dict()
    for name, value in base.state_dict().items():
        torch.testing.assert_close(composed_state[name], value, rtol=0.0, atol=0.0)


def test_tpo_target_and_natural_kl_keep_v5_semantics():
    generator = torch.Generator().manual_seed(19)
    anchor_logp = torch.log_softmax(torch.randn(23, 8, generator=generator), dim=-1)
    scores = torch.randn(23, 8, generator=generator)
    eta = 1.7

    target = MODULE.tpo_restricted_target(anchor_logp, scores, eta)
    expected = torch.softmax(anchor_logp + scores / eta, dim=-1)
    torch.testing.assert_close(target, expected)
    expected_kl = (
        anchor_logp.exp()
        * (anchor_logp - torch.log_softmax(anchor_logp + scores / eta, dim=-1))
    ).sum(-1).mean()
    torch.testing.assert_close(MODULE.tpo_reverse_kl(anchor_logp, scores, eta), expected_kl)
    assert MODULE.tpo_reverse_kl(anchor_logp, scores, 4.0) < expected_kl
    torch.testing.assert_close(
        MODULE.tpo_restricted_target(anchor_logp, torch.ones_like(scores), 0.2),
        anchor_logp.exp(),
    )


def test_nextlat_indices_use_outgoing_actions_and_same_env_future_states():
    # T-major rows: (t, env) -> t * N + env.
    source = torch.tensor([1, 5, 8]).numpy()
    actions, targets = MODULE.make_nextlat_indices(
        source, num_envs=4, batch_size=12, depth=2
    )

    assert actions.tolist() == [[1, 5, 8], [5, 9, 11]]
    assert targets.tolist() == [[5, 9, 11], [9, 11, 11]]


def test_nextlat_mask_excludes_boundaries_and_rollout_tail():
    boundaries = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    mask = MODULE.build_nextlat_mask(boundaries, depth=2)

    expected_h1 = 1.0 - boundaries
    expected_h1[-1] = 0.0  # no s_{t+1} exists past the rollout tail
    torch.testing.assert_close(mask[:, :, 0], expected_h1)
    expected_h2 = torch.tensor(
        [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    )
    torch.testing.assert_close(mask[:, :, 1], expected_h2)


def test_policy_space_auxiliary_freezes_decoder_weights():
    agent = MODULE.Agent(_DummyEnvs(), _args(MODULE))
    source = torch.randn(5, 8, requires_grad=True)
    target = torch.randn(5, 8)

    with torch.no_grad():
        teacher, _, _ = agent._actor_dist_frozen_head(target)
    student, _, _ = agent._actor_dist_frozen_head(source)
    torch.distributions.kl_divergence(teacher, student).sum().backward()

    assert source.grad is not None and source.grad.norm() > 0.0
    assert all(parameter.grad is None for parameter in agent.actor_alpha_head.parameters())
    assert all(parameter.grad is None for parameter in agent.actor_beta_head.parameters())


def test_optimizer_partitions_and_logical_blocks_are_exact():
    agent = MODULE.Agent(_DummyEnvs(), _args(MODULE))
    task_ids = {id(parameter) for parameter in agent.task_parameters()}
    predictor_ids = {id(parameter) for parameter in agent.nextlat_predictor_parameters()}
    all_ids = {id(parameter) for parameter in agent.parameters()}
    trunk_ids = {id(parameter) for parameter in agent.trunk.parameters()}
    policy_head_ids = {
        id(parameter)
        for head in (agent.actor_alpha_head, agent.actor_beta_head)
        for parameter in head.parameters()
    }
    blocks = agent.nextlat_trunk_blocks()
    flat_block_ids = [id(parameter) for block in blocks for parameter in block]

    assert task_ids.isdisjoint(predictor_ids)
    assert task_ids | predictor_ids == all_ids
    assert set(flat_block_ids) == trunk_ids
    assert len(flat_block_ids) == len(set(flat_block_ids))
    assert trunk_ids.isdisjoint(policy_head_ids)
    assert len(blocks) == 1 + _args(MODULE).k_blocks + 1


def test_logical_block_projection_protects_actor_and_critic_and_caps_locally():
    block_parameters = [
        [torch.nn.Parameter(torch.zeros(2))],
        [torch.nn.Parameter(torch.zeros(1))],
    ]
    _, layout = MODULE.make_logical_block_layout(block_parameters)
    task = [torch.tensor([1.0, 0.0]), torch.tensor([100.0])]
    predictive = [torch.tensor([1.0, -100.0]), torch.tensor([0.0])]
    actor = [torch.tensor([1.0, 0.0]), torch.tensor([0.0])]
    critic = [torch.tensor([0.0, 1.0]), torch.tensor([0.0])]

    admitted, stats = MODULE.admit_predictive_updates(
        task,
        predictive,
        0.05,
        actor_gradients=actor,
        critic_gradients=critic,
        layout=layout,
    )

    # Actor-conflicting x is removed. Safe -y cannot borrow the second block's
    # much larger task trust budget, so block 0 is capped at 0.05.
    torch.testing.assert_close(admitted[0], torch.tensor([0.0, -0.05]), atol=1e-7, rtol=0.0)
    assert torch.equal(admitted[1], torch.zeros_like(admitted[1]))
    assert torch.dot(actor[0], admitted[0]) <= 0.0
    assert torch.dot(critic[0], admitted[0]) <= 0.0
    assert stats["block_admitted_ratio"][0] <= 0.05 + 1e-7
    assert stats["admitted_norm"] <= 0.05 * stats["task_norm"] + 1e-7
    assert stats["block_actor_conflict"].tolist() == [True, False]
    assert stats["block_critic_conflict"].tolist() == [False, False]


def _explicit_two_halfspace_projection(proposal, actor, critic):
    """High-precision active-set oracle independent of the production algebra."""
    proposal = proposal.double()
    actor = actor.double()
    critic = critic.double()
    candidates = []

    def add_if_feasible(candidate):
        actor_tolerance = 8e-15 * (
            actor * candidate
        ).abs().sum() + torch.finfo(torch.float64).tiny
        critic_tolerance = 8e-15 * (
            critic * candidate
        ).abs().sum() + torch.finfo(torch.float64).tiny
        if (
            actor.dot(candidate) <= actor_tolerance
            and critic.dot(candidate) <= critic_tolerance
        ):
            candidates.append(candidate)

    add_if_feasible(proposal)
    add_if_feasible(torch.zeros_like(proposal))
    for gradient in (actor, critic):
        gradient_sq = gradient.dot(gradient)
        if gradient_sq > 0.0:
            multiplier = (gradient.dot(proposal) / gradient_sq).clamp_min(0.0)
            add_if_feasible(proposal - multiplier * gradient)

    actor_norm = actor.norm()
    critic_norm = critic.norm()
    if actor_norm > 0.0 and critic_norm > 0.0:
        actor_unit = actor / actor_norm
        critic_unit = critic / critic_norm
        correlation = actor_unit.dot(critic_unit)
        alignment = 1.0 if correlation >= 0.0 else -1.0
        angular_difference = critic_unit - alignment * actor_unit
        critic_orthogonal = angular_difference - actor_unit.dot(
            angular_difference
        ) * actor_unit
        critic_orthogonal_sq = critic_orthogonal.dot(critic_orthogonal)
        if critic_orthogonal_sq > 1e-14:
            actor_coordinate = actor_unit.dot(proposal)
            orthogonal_coordinate = critic_orthogonal.dot(
                proposal
            ) / critic_orthogonal_sq
            both_active = (
                proposal
                - actor_coordinate * actor_unit
                - orthogonal_coordinate * critic_orthogonal
            )
            critic_on_actor = alignment + actor_unit.dot(angular_difference)
            actor_multiplier = (
                actor_coordinate - orthogonal_coordinate * critic_on_actor
            ) / actor_norm
            critic_multiplier = orthogonal_coordinate / critic_norm
            if actor_multiplier >= -1e-9 and critic_multiplier >= -1e-9:
                add_if_feasible(both_active)
        elif alignment < 0.0:
            # Opposed rank-one constraints imply equality on their common normal.
            equality_projection = proposal - actor_unit.dot(proposal) * actor_unit
            candidates.append(equality_projection)

    assert candidates
    return min(candidates, key=lambda candidate: (candidate - proposal).square().sum())


@torch.no_grad()
def _production_projection(proposal, actor, critic):
    parameter = torch.nn.Parameter(torch.zeros_like(proposal))
    _, layout = MODULE.make_logical_block_layout([[parameter]])
    task = torch.full_like(proposal, 100.0)
    admitted, stats = MODULE.admit_predictive_updates(
        [task],
        [proposal],
        max_ratio=100.0,
        actor_gradients=[actor],
        critic_gradients=[critic],
        layout=layout,
    )
    return admitted[0], stats


@torch.no_grad()
def test_randomized_projection_matches_explicit_convex_oracle_fp32_and_fp64():
    for dtype, relative_tolerance in ((torch.float32, 3e-5), (torch.float64, 1e-11)):
        for seed in range(256):
            generator = torch.Generator().manual_seed(10_000 + seed)
            dimension = 3 + seed % 29
            scale = 10.0 ** ((seed % 9) - 4)
            proposal = torch.randn(dimension, generator=generator, dtype=dtype)
            actor = scale * torch.randn(dimension, generator=generator, dtype=dtype)
            critic = torch.randn(dimension, generator=generator, dtype=dtype) / scale

            projected, stats = _production_projection(proposal, actor, critic)
            oracle = _explicit_two_halfspace_projection(
                proposal, actor, critic
            ).to(dtype)
            torch.testing.assert_close(
                projected,
                oracle,
                rtol=relative_tolerance,
                atol=relative_tolerance * max(1.0, oracle.norm().item()),
            )
            assert actor.dot(projected) <= stats["block_actor_tolerance"][0] * 1.01
            assert critic.dot(projected) <= stats["block_critic_tolerance"][0] * 1.01


@torch.no_grad()
def test_ill_conditioned_and_collinear_projection_remains_optimal_and_safe():
    cases = []
    for dtype, angles in (
        (torch.float32, (1e-2, 1e-3)),
        (torch.float64, (1e-3, 1e-6)),
    ):
        for seed in range(48):
            generator = torch.Generator().manual_seed(20_000 + seed)
            actor_unit = torch.randn(31, generator=generator, dtype=dtype)
            actor_unit /= actor_unit.norm()
            orthogonal = torch.randn(31, generator=generator, dtype=dtype)
            orthogonal -= orthogonal.dot(actor_unit) * actor_unit
            orthogonal /= orthogonal.norm()
            proposal = torch.randn(31, generator=generator, dtype=dtype)
            for angle in angles:
                for alignment in (-1.0, 1.0):
                    critic_unit = alignment * actor_unit + angle * orthogonal
                    critic_unit /= critic_unit.norm()
                    # Extreme scale mismatch verifies scale-aware feasibility.
                    cases.append(
                        (
                            proposal,
                            actor_unit * (1e-4 if seed % 2 else 1e4),
                            critic_unit * (1e4 if seed % 2 else 1e-4),
                        )
                    )
            cases.extend(
                (
                    (proposal, actor_unit, 3.7 * actor_unit),
                    (proposal, actor_unit, -3.7 * actor_unit),
                )
            )

    for proposal, actor, critic in cases:
        projected, stats = _production_projection(proposal, actor, critic)
        oracle = _explicit_two_halfspace_projection(proposal, actor, critic).to(
            proposal.dtype
        )
        tolerance = 3e-4 if proposal.dtype == torch.float32 else 3e-8
        torch.testing.assert_close(
            projected,
            oracle,
            rtol=tolerance,
            atol=tolerance * max(1.0, oracle.norm().item()),
        )
        assert actor.dot(projected) <= stats["block_actor_tolerance"][0] * 1.01
        assert critic.dot(projected) <= stats["block_critic_tolerance"][0] * 1.01


def test_meaningful_positive_constraint_violation_is_still_vetoed():
    proposal = torch.tensor([1.0, 2.0])
    actor = torch.tensor([1.0, 0.0])
    critic = torch.tensor([0.0, -1.0])
    projected, stats = _production_projection(proposal, actor, critic)

    # The closest feasible update removes the actor-conflicting x component while
    # preserving the critic-compatible y component. A tolerance must not keep x.
    torch.testing.assert_close(projected, torch.tensor([0.0, 2.0]))
    assert actor.dot(projected) <= stats["block_actor_tolerance"][0]
    assert critic.dot(projected) <= stats["block_critic_tolerance"][0]
    assert stats["block_actor_tolerance"][0] < 1e-4
    assert stats["block_critic_tolerance"][0] < 1e-4


def test_predictive_transaction_advances_adam_but_replaces_unsafe_proposal():
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([parameter], lr=0.1, betas=(0.0, 0.0), eps=1e-8)
    _, layout = MODULE.make_logical_block_layout([[parameter]])

    raw, admitted, stats = MODULE.apply_predictive_trunk_transaction(
        [parameter],
        optimizer,
        [torch.tensor([-1.0, -1.0])],
        [torch.tensor([1.0, 1.0])],
        0.05,
        actor_gradients=[torch.tensor([1.0, 0.0])],
        critic_gradients=[torch.tensor([0.0, 1.0])],
        layout=layout,
    )

    assert raw[0].norm() > 0.0
    assert torch.equal(admitted[0], torch.zeros_like(admitted[0]))
    assert torch.equal(parameter, torch.zeros_like(parameter))
    assert optimizer.state[parameter]["step"].item() == 1
    assert stats["actor_first_order"] <= 0.0
    assert stats["critic_first_order"] <= 0.0


def test_nan_predictive_gradient_restores_exact_post_task_parameters():
    parameter = torch.nn.Parameter(torch.tensor([3.25, -7.5]))
    post_task = parameter.detach().clone()
    optimizer = torch.optim.Adam([parameter], lr=0.1, betas=(0.0, 0.0), eps=1e-8)
    _, layout = MODULE.make_logical_block_layout([[parameter]])

    raw, admitted, stats = MODULE.apply_predictive_trunk_transaction(
        [parameter],
        optimizer,
        [torch.tensor([float("nan"), 1.0])],
        [torch.tensor([0.02, -0.01])],
        0.05,
        actor_gradients=[torch.tensor([1.0, 0.0])],
        critic_gradients=[torch.tensor([0.0, 1.0])],
        layout=layout,
    )

    assert not torch.isfinite(raw[0]).all()
    assert torch.equal(admitted[0], torch.zeros_like(admitted[0]))
    assert torch.isfinite(admitted[0]).all()
    assert torch.equal(parameter, post_task)
    assert optimizer.state[parameter]["step"].item() == 1
    assert stats["nonfinite_veto"].item() == 1.0
    assert MODULE._admission_result_is_finite(admitted, stats)


def test_nonfinite_admission_result_is_vetoed_before_parameter_add(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([2.0, -4.0]))
    post_task = parameter.detach().clone()
    optimizer = torch.optim.Adam([parameter], lr=0.1, betas=(0.0, 0.0), eps=1e-8)
    _, layout = MODULE.make_logical_block_layout([[parameter]])
    original_admission = MODULE.admit_predictive_updates

    def corrupted_admission(*args, **kwargs):
        admitted, stats = original_admission(*args, **kwargs)
        admitted[0][0] = float("nan")
        stats["raw_cosine"] = stats["raw_cosine"].new_tensor(float("nan"))
        return admitted, stats

    monkeypatch.setattr(MODULE, "admit_predictive_updates", corrupted_admission)
    _, admitted, stats = MODULE.apply_predictive_trunk_transaction(
        [parameter],
        optimizer,
        [torch.tensor([-1.0, 1.0])],
        [torch.tensor([0.02, -0.01])],
        0.05,
        actor_gradients=[torch.tensor([-1.0, 0.0])],
        critic_gradients=[torch.tensor([0.0, 1.0])],
        layout=layout,
    )

    assert torch.equal(admitted[0], torch.zeros_like(admitted[0]))
    assert torch.equal(parameter, post_task)
    assert stats["nonfinite_veto"].item() == 1.0
    assert MODULE._admission_result_is_finite(admitted, stats)


def test_tpo_breaker_prevents_predictive_trunk_transaction_in_source():
    source = (ROOT / "cleanrl/tpo/md/ppo_continuous_action_tpomd_prednext_cuda_v16.py").read_text()
    private_step = source.index("predictor_step_norm = apply_private_optimizer_step(")
    guard = source.index("if actor_active:", private_step)
    transaction = source.index("apply_predictive_trunk_transaction(", guard)
    frozen_branch = source.index("The TPO breaker is a policy freeze", transaction)

    assert private_step < guard < transaction < frozen_branch


def test_cuda_v16_defaults_match_v13_except_filename_derived_name():
    expected = vars(V13.Args())
    actual = vars(MODULE.Args())
    expected.pop("exp_name")
    actual.pop("exp_name")
    assert actual == expected
    assert MODULE.Args().compile_mode == "reduce-overhead"


@pytest.mark.parametrize("actor_dist", ["beta", "gaussian"])
@pytest.mark.parametrize("sample_rollout", [False, True])
def test_policy_model_wrapper_preserves_outputs_gradients_and_rng(
    actor_dist,
    sample_rollout,
):
    args_v13 = V13.Args(
        actor_dist=actor_dist,
        hidden=8,
        k_blocks=2,
        n_experts=2,
        num_bins=7,
    )
    args_v16 = MODULE.Args(
        actor_dist=actor_dist,
        hidden=8,
        k_blocks=2,
        n_experts=2,
        num_bins=7,
    )
    torch.manual_seed(6201)
    reference = V13.Agent(_DummyEnvs(), args_v13)
    torch.manual_seed(6201)
    candidate = MODULE.Agent(_DummyEnvs(), args_v16)
    observations = torch.randn(13, 7)
    native_actions = None
    if not sample_rollout:
        native_actions = (
            torch.full((13, 3), 0.37)
            if actor_dist == "beta"
            else torch.linspace(-0.8, 0.8, 39).reshape(13, 3)
        )
    candidates = (
        torch.full((13, 8, 3), 0.43)
        if actor_dist == "beta"
        else torch.linspace(-1.1, 1.1, 13 * 8 * 3).reshape(13, 8, 3)
    )

    torch.manual_seed(3389)
    expected = reference.get_action_and_value(
        observations,
        native_actions,
        candidate_zs=candidates,
        return_actor_feat=True,
    )
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(3389)
    actual, _, _, _ = MODULE.action_value_from_policy_outputs(
        candidate,
        MODULE.policy_model_forward(candidate, observations),
        native_actions,
        candidates,
    )
    actual_rng = torch.get_rng_state().clone()

    assert len(actual) == len(expected)
    for expected_tensor, actual_tensor in zip(expected, actual):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0.0, atol=0.0)
    assert torch.equal(actual_rng, expected_rng)

    expected_loss = sum(tensor.float().sum() for tensor in expected[2:])
    actual_loss = sum(tensor.float().sum() for tensor in actual[2:])
    expected_loss.backward()
    actual_loss.backward()
    expected_grads = dict(reference.named_parameters())
    for name, parameter in candidate.named_parameters():
        expected_grad = expected_grads[name].grad
        if expected_grad is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(
                parameter.grad,
                expected_grad,
                rtol=0.0,
                atol=0.0,
            )


def test_graph_output_retention_is_alias_safe_only_when_compiled():
    graph_output = torch.randn(5, 7)
    eager = MODULE.retain_graph_output(graph_output, compiled=False)
    retained = MODULE.retain_graph_output(graph_output, compiled=True)
    expected = graph_output.clone()

    assert eager.data_ptr() == graph_output.data_ptr()
    assert retained.data_ptr() != graph_output.data_ptr()
    graph_output.fill_(123.0)  # model the next CUDA-graph replay overwriting storage
    assert torch.equal(eager, graph_output)
    assert torch.equal(retained, expected)


def test_policy_model_wrapper_is_fullgraph_compilable_without_rng_consumption():
    torch.manual_seed(211)
    eager_agent = MODULE.Agent(_DummyEnvs(), _args(MODULE))
    torch.manual_seed(211)
    compiled_agent = MODULE.Agent(_DummyEnvs(), _args(MODULE))
    observations = torch.randn(16, 7)
    compiled = torch.compile(
        lambda batch: MODULE.policy_model_forward(compiled_agent, batch),
        backend="eager",
        dynamic=False,
        fullgraph=True,
    )

    rng_before = torch.get_rng_state().clone()
    expected = MODULE.policy_model_forward(eager_agent, observations)
    rng_after_eager = torch.get_rng_state().clone()
    actual = compiled(observations)
    rng_after_compiled = torch.get_rng_state().clone()
    for expected_tensor, actual_tensor in zip(expected, actual):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0.0, atol=0.0)
    assert torch.equal(rng_before, rng_after_eager)
    assert torch.equal(rng_before, rng_after_compiled)

    sum(tensor.square().mean() for tensor in expected).backward()
    sum(tensor.square().mean() for tensor in actual).backward()
    expected_parameters = dict(eager_agent.named_parameters())
    for name, parameter in compiled_agent.named_parameters():
        expected_gradient = expected_parameters[name].grad
        if expected_gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(
                parameter.grad,
                expected_gradient,
                rtol=0.0,
                atol=0.0,
            )


def test_async_clip_matches_pytorch_and_fails_loudly():
    expected = [torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(2))]
    actual = [torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(2))]
    gradients = [torch.tensor([3.0, 4.0, -2.0]), torch.tensor([0.5, -1.5])]
    for parameter, gradient in zip(expected, gradients):
        parameter.grad = gradient.clone()
    for parameter, gradient in zip(actual, gradients):
        parameter.grad = gradient.clone()
    expected_norm = torch.nn.utils.clip_grad_norm_(expected, 0.25)
    actual_norm = MODULE.clip_grad_norm_async_fail_loud_(actual, 0.25)
    assert torch.equal(actual_norm, expected_norm)
    for expected_parameter, actual_parameter in zip(expected, actual):
        assert torch.equal(actual_parameter.grad, expected_parameter.grad)

    invalid = torch.nn.Parameter(torch.zeros(2))
    invalid.grad = torch.tensor([float("nan"), 1.0])
    with pytest.raises(RuntimeError, match="non-finite"):
        MODULE.clip_grad_norm_async_fail_loud_([invalid], 0.25)


def test_performance_wiring_is_static_cuda_resident_and_rng_safe():
    source = Path(MODULE.__file__).read_text()
    training = source[source.index('if __name__ == "__main__":') :]
    assert 'torch.set_float32_matmul_precision("high")' in training
    assert source.count("torch.compile(") >= 6
    assert "mode=args.compile_mode, dynamic=False" in source
    assert "fullgraph=True" in source
    assert source.count("torch.compiler.cudagraph_mark_step_begin()") >= 6
    assert "retain_graph_output(" in training
    assert "target_probs = retain_graph_output(" in training
    assert "epoch_inds = torch.as_tensor(b_inds, device=device)" in training
    assert "mb_inds = epoch_inds[start:end]" in training
    assert "action_indices = (" in training
    assert "nextlat_action_offsets" in training
    assert "clipfrac_sum.add_(" in training
    assert "epoch_kl_sum.add_(approx_kl)" in training
    assert "epoch_mean_kl.item()" in training
    assert "clipfracs" not in training
    # Sampling and persistent probe-RNG save/restore remain outside compiled wrappers.
    assert "dist.sample()" not in inspect.getsource(MODULE.policy_model_forward)
    assert "with torch.random.fork_rng(devices=[device]):" in training
    assert "probe_cpu_rng_state = torch.get_rng_state()" in training
    assert "probe_cuda_rng_state = torch.cuda.get_rng_state(device)" in training
    # HL-Gauss labels are created and consumed on CUDA without epoch transfers.
    assert "project_value_targets_fn(returns)" in training
    assert "b_target_probs[mb_inds]" in training
    assert "b_target_probs[mb_inds].to(" not in training
    assert ".cpu()" not in training[
        training.index("projected_targets = project_value_targets_fn") :
        training.index("b_target_probs = target_probs.reshape")
    ]


def test_scalar_telemetry_uses_one_explicit_transfer_and_no_minibatch_item():
    values = {
        "a": torch.tensor(1.25),
        "b": torch.tensor(-4.0, dtype=torch.float64),
    }
    assert MODULE.synchronize_scalar_telemetry(values) == {"a": 1.25, "b": -4.0}
    helper = inspect.getsource(MODULE.synchronize_scalar_telemetry)
    assert helper.count(".cpu()") == 1
    assert ".item()" not in helper
    source = Path(MODULE.__file__).read_text()
    minibatch = source[
        source.index("for start in range(0, args.batch_size, args.minibatch_size):") :
        source.index("epochs_completed = epoch + 1")
    ]
    assert ".item()" not in minibatch
    assert "float(prednext" not in minibatch
