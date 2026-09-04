import ast
import importlib.util
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl" / "nextlat" / "ppo_continuous_action_nextlat_successor_innovation_td_v10.py"
)
SPEC = importlib.util.spec_from_file_location("nextlat_successor_innovation_td_v10", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DummyVectorEnv:
    single_observation_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(5,), dtype=np.float32
    )
    single_action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


def test_defaults_are_the_full_halfcheetah_successor_experiment():
    args = MODULE.Args()

    assert args.env_id == "HalfCheetah-v4"
    assert args.total_timesteps == 8_000_000
    assert args.seed == 1
    assert args.cuda
    assert args.successor_td
    assert args.successor_gammas == (0.60, 0.90, 0.97, 0.99)
    assert args.successor_direct_horizons == (4, 8, 16, 64)
    assert args.successor_direct_coef > args.successor_policy_coef > 0
    assert args.successor_reward_coef > 0
    assert args.successor_value_coef > 0
    assert args.successor_critic_semantic_coef > 0
    assert args.successor_trunk_grad_clip < args.successor_predictor_grad_clip
    assert args.successor_ema_decay == 0.95
    assert args.successor_trunk_grad_clip == 0.03


def test_successor_target_is_direct_prefix_plus_one_td_tail():
    innovations = torch.arange(1.0, 6.0).view(5, 1, 1)
    tails = (10.0 * torch.arange(1.0, 6.0)).view(5, 1, 1, 1)
    next_state_tails = torch.full_like(tails, 10.0)
    result = MODULE.build_successor_targets(
        innovations,
        tails,
        next_state_tails,
        torch.zeros(5, 1),
        torch.zeros(5, 1),
        torch.ones(5, 1),
        gammas=(0.5,),
        direct_horizons=(2,),
    )

    # t=0: .5*eta_0 + .25*eta_1 + .25*target_Psi_2.
    torch.testing.assert_close(result.values[0, 0, 0, 0], torch.tensor(8.5))
    # Short prefixes at the artificial edge bootstrap from the actual final next state.
    torch.testing.assert_close(result.values[3, 0, 0, 0], torch.tensor(5.75))
    torch.testing.assert_close(result.values[4, 0, 0, 0], torch.tensor(7.5))
    assert result.bootstrap_masks[:, 0, 0].all()
    torch.testing.assert_close(result.observed_mass[0, 0, 0], torch.tensor(0.75))
    torch.testing.assert_close(result.observed_mass[4, 0, 0], torch.tensor(0.5))


def test_terminal_final_innovation_is_kept_but_reset_and_tail_are_masked():
    innovations = torch.tensor([1.0, 2.0, 100.0, 200.0]).view(4, 1, 1)
    tails = torch.full((4, 1, 1, 1), 1000.0)
    boundaries = torch.tensor([0.0, 1.0, 0.0, 0.0]).view(4, 1)
    result = MODULE.build_successor_targets(
        innovations,
        tails,
        tails,
        boundaries,
        boundaries,
        torch.ones_like(boundaries),
        gammas=(0.5,),
        direct_horizons=(3,),
    )

    # Source t=0 sees transition 0 and the terminal transition 1, then stops.
    torch.testing.assert_close(result.values[0, 0, 0, 0], torch.tensor(1.0))
    # Source t=1 still learns its observed terminal-state innovation.
    torch.testing.assert_close(result.values[1, 0, 0, 0], torch.tensor(1.0))
    assert not result.bootstrap_masks[0, 0, 0]
    assert not result.bootstrap_masks[1, 0, 0]
    assert result.masks[:, 0, 0].all()


def test_truncation_bootstraps_from_final_state_without_crossing_reset():
    innovations = torch.tensor([1.0, 2.0, 100.0, 200.0]).view(4, 1, 1)
    tails = torch.full((4, 1, 1, 1), 1000.0)
    next_state_tails = torch.full_like(tails, 20.0)
    boundaries = torch.tensor([0.0, 1.0, 0.0, 0.0]).view(4, 1)
    result = MODULE.build_successor_targets(
        innovations,
        tails,
        next_state_tails,
        torch.zeros_like(boundaries),
        boundaries,
        torch.ones_like(boundaries),
        gammas=(0.5,),
        direct_horizons=(3,),
    )

    # t=0 includes eta_0, eta_1, then Psi(final_obs); eta_2 belongs to reset.
    torch.testing.assert_close(result.values[0, 0, 0, 0], torch.tensor(6.0))
    # Starting on the truncating transition uses the correctly shorter prefix.
    torch.testing.assert_close(result.values[1, 0, 0, 0], torch.tensor(11.0))
    assert result.bootstrap_masks[0, 0, 0]
    assert result.bootstrap_masks[1, 0, 0]


def test_only_truncations_and_nonboundary_rollout_edge_need_new_actions():
    terminations = torch.tensor(
        [[False, False, False], [True, False, False], [False, True, False]]
    )
    boundaries = torch.tensor(
        [[False, True, False], [True, False, False], [False, True, False]]
    )
    valids = torch.ones_like(boundaries)

    rows = MODULE.next_state_bootstrap_rows(terminations, boundaries, valids)

    assert rows.tolist() == [
        [False, True, False],
        [False, False, False],
        [True, False, True],
    ]


def test_missing_final_observation_masks_the_source_entirely():
    innovations = torch.ones(3, 1, 2)
    tails = torch.ones(3, 1, 1, 2)
    boundaries = torch.tensor([[1.0], [0.0], [0.0]])
    valids = torch.tensor([[0.0], [1.0], [1.0]])

    result = MODULE.build_successor_targets(
        innovations,
        tails,
        tails,
        boundaries,
        boundaries,
        valids,
        (0.9,),
        (2,),
    )

    assert not result.masks[0, 0, 0]
    assert torch.equal(result.values[0], torch.zeros_like(result.values[0]))
    assert not result.bootstrap_masks[0, 0, 0]


def test_later_missing_final_observation_uses_known_pretransition_tail():
    innovations = torch.tensor([1.0, 999.0, 100.0]).view(3, 1, 1)
    tails = torch.tensor([3.0, 10.0, 30.0]).view(3, 1, 1, 1)
    boundaries = torch.tensor([[0.0], [1.0], [0.0]])
    valids = torch.tensor([[1.0], [0.0], [1.0]])

    result = MODULE.build_successor_targets(
        innovations,
        tails,
        torch.full_like(tails, 1000.0),
        boundaries,
        boundaries,
        valids,
        (0.5,),
        (3,),
    )

    # .5*eta_0 + .5*Psi(s_1,a_1); unknown eta_1 is neither used nor zero-filled.
    torch.testing.assert_close(result.values[0, 0, 0, 0], torch.tensor(5.5))
    assert result.masks[0, 0, 0]
    assert result.bootstrap_masks[0, 0, 0]


def test_successor_labels_are_always_stop_gradient():
    innovations = torch.randn(6, 2, 3, requires_grad=True)
    tails = torch.randn(6, 2, 2, 3, requires_grad=True)
    next_state_tails = torch.randn(6, 2, 2, 3, requires_grad=True)
    result = MODULE.build_successor_targets(
        innovations,
        tails,
        next_state_tails,
        torch.zeros(6, 2),
        torch.zeros(6, 2),
        torch.ones(6, 2),
        (0.6, 0.95),
        (2, 4),
    )

    assert not result.values.requires_grad
    assert result.values.grad_fn is None


def test_band_scaled_loss_is_invariant_to_target_units():
    torch.manual_seed(4)
    target = torch.randn(12, 2, 5)
    prediction = target + 0.2 * torch.randn_like(target)
    mask = torch.ones(12, 2, dtype=torch.bool)

    loss, scale = MODULE.masked_scaled_smooth_l1(prediction, target, mask)
    rescaled_loss, rescaled_scale = MODULE.masked_scaled_smooth_l1(
        17.0 * prediction, 17.0 * target, mask
    )

    torch.testing.assert_close(loss, rescaled_loss)
    torch.testing.assert_close(rescaled_scale, 17.0 * scale)

    feature_units = torch.tensor([2.0, 3.0, 5.0, 7.0, 11.0]).view(1, 1, -1)
    feature_loss, feature_scale = MODULE.masked_scaled_smooth_l1(
        feature_units * prediction, feature_units * target, mask
    )
    torch.testing.assert_close(loss, feature_loss)
    torch.testing.assert_close(feature_scale, feature_units.squeeze(0) * scale)
    assert scale.shape == (2, 5)

    _, reused_scale = MODULE.masked_scaled_smooth_l1(
        prediction[:4], target[:4], mask[:4], scale=scale
    )
    torch.testing.assert_close(reused_scale, scale)


def test_fp16_resident_critic_labels_preserve_hlgauss_cross_entropy():
    args = MODULE.Args()
    support = MODULE.Dreamer3BucketHLGaussSupport(
        args.num_bins,
        args.v_min,
        args.v_max,
        args.value_sigma_to_bin_ratio,
        torch.device("cpu"),
    )
    returns = torch.linspace(-20_000.0, 20_000.0, 1025).view(-1, 1)
    probabilities = support.project(returns)
    mask = torch.ones(probabilities.shape[:-1], dtype=torch.bool)

    stored, stored_mask = MODULE.store_critic_targets(
        probabilities, mask, torch.device("cpu")
    )
    logits = torch.randn_like(probabilities)
    reference_ce = -(probabilities * torch.log_softmax(logits, -1)).sum(-1)
    stored_ce = -(stored * torch.log_softmax(logits, -1)).sum(-1)

    assert stored.dtype == torch.float16
    assert stored_mask.dtype == torch.bool
    assert stored_ce.dtype == torch.float32
    assert stored.element_size() * 2 == probabilities.element_size()
    assert (stored.float().sum(-1) - 1.0).abs().max() < 5e-4
    torch.testing.assert_close(stored_ce, reference_ce, rtol=5e-4, atol=5e-3)


def test_critic_minibatches_do_not_transfer_labels_from_cpu():
    source = SCRIPT.read_text()

    assert "b_target_probs[mb_inds].to(" not in source
    assert "return_mtp_mask.reshape(-1, args.critic_mtp_horizon).cpu()" not in source
    assert "b_target_probs[..., 0].float()" in source
    assert "b_target_mask.to(dtype=torch.float32)" in source


def test_target_latent_capture_owns_outputs_from_reused_graph_buffers():
    shared_graph_output = torch.empty(6, 3)
    current_observations = torch.arange(18.0).reshape(6, 3)
    next_observations = 100.0 + current_observations
    marks = []

    def replay_into_shared_buffer(observations):
        shared_graph_output.copy_(observations)
        return shared_graph_output

    current_latents, next_latents = MODULE.capture_target_latent_tables(
        replay_into_shared_buffer,
        replay_into_shared_buffer,
        current_observations,
        next_observations,
        lambda: marks.append(len(marks)),
    )

    # A later compiled replay overwrites its reusable graph output. Both returned tables
    # must nevertheless remain stable because training retains them across many replays.
    replay_into_shared_buffer(torch.full_like(current_observations, -999.0))
    assert marks == [0, 1]
    assert current_latents.data_ptr() != shared_graph_output.data_ptr()
    assert next_latents.data_ptr() != shared_graph_output.data_ptr()
    torch.testing.assert_close(current_latents, current_observations)
    torch.testing.assert_close(next_latents, next_observations)


def test_training_path_uses_owned_target_latent_capture_before_innovations():
    source = SCRIPT.read_text()
    capture = source.index(
        "target_latents, target_next_latents = capture_target_latent_tables("
    )
    innovation = source.index(
        "raw_innovations = target_next_latents - target_latents"
    )
    helper_source = ast.get_source_segment(
        source,
        next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "capture_target_latent_tables"
        ),
    )

    assert capture < innovation
    assert helper_source is not None
    assert helper_source.count(".clone()") == 2
    assert "target_current_feature_fn(current_observations).clone()" in helper_source
    assert "target_next_feature_fn(next_observations).clone()" in helper_source


def test_conflicting_auxiliary_gradient_is_projected_and_task_ratio_capped():
    auxiliary = [torch.tensor([-1.0, 1.0])]
    task = [torch.tensor([1.0, 0.0])]

    delivered, diagnostics = MODULE.project_and_cap_auxiliary_gradients(
        auxiliary, task, absolute_cap=1.0, task_ratio_cap=0.25
    )

    torch.testing.assert_close(delivered[0], torch.tensor([0.0, 0.25]))
    torch.testing.assert_close(diagnostics["delivered_norm"], torch.tensor(0.25))
    assert diagnostics["cosine"] < 0
    assert diagnostics["conflict"] == 1
    assert torch.dot(delivered[0], task[0]) >= 0


def test_gradient_projection_prevents_cross_block_conflict_cancellation():
    auxiliary = [torch.tensor([-1.0]), torch.tensor([2.0])]
    task = [torch.tensor([1.0]), torch.tensor([1.0])]

    delivered, diagnostics = MODULE.project_and_cap_auxiliary_gradients(
        auxiliary,
        task,
        absolute_cap=10.0,
        task_ratio_cap=10.0,
        groups=[0, 1],
    )

    torch.testing.assert_close(delivered[0], torch.zeros(1))
    torch.testing.assert_close(delivered[1], torch.tensor([2.0]))
    assert diagnostics["conflict"] == 1
    torch.testing.assert_close(diagnostics["conflict_fraction"], torch.tensor(0.5))
    assert diagnostics["worst_group_cosine"] < 0


def test_gradient_projection_uses_only_auxiliary_support_and_holds_dormant_blocks():
    projected, _ = MODULE.project_and_cap_auxiliary_gradients(
        [torch.tensor([-1.0]), None],
        [torch.tensor([1.0]), torch.tensor([10.0])],
        absolute_cap=100.0,
        task_ratio_cap=100.0,
    )
    torch.testing.assert_close(projected[0], torch.zeros(1))

    delivered, _ = MODULE.project_and_cap_auxiliary_gradients(
        [torch.tensor([100.0]), torch.tensor([100.0])],
        [torch.tensor([1.0]), torch.tensor([0.0])],
        absolute_cap=100.0,
        task_ratio_cap=0.25,
        groups=[0, 1],
    )
    torch.testing.assert_close(delivered[0], torch.tensor([0.25]))
    torch.testing.assert_close(delivered[1], torch.zeros(1))


def test_actor_and_critic_conflicts_cannot_cancel_in_combined_task_direction():
    delivered, diagnostics = MODULE.project_and_cap_auxiliary_gradients(
        [torch.tensor([1.0])],
        [torch.tensor([0.0])],
        absolute_cap=10.0,
        task_ratio_cap=10.0,
        safety_tasks=(
            [torch.tensor([1.0])],
            [torch.tensor([-1.0])],
        ),
    )

    torch.testing.assert_close(delivered[0], torch.zeros(1))
    torch.testing.assert_close(
        diagnostics["objective_veto_fraction"], torch.tensor(1.0)
    )


def test_projection_fails_closed_for_nan_inf_and_finite_square_overflow():
    invalid_auxiliaries = (
        torch.tensor([float("nan"), 1.0]),
        torch.tensor([float("inf"), -1.0]),
        torch.tensor([1e30, -1e30]),
    )
    assert torch.isfinite(invalid_auxiliaries[-1]).all()
    assert not torch.isfinite(invalid_auxiliaries[-1].square()).all()

    for auxiliary in invalid_auxiliaries:
        delivered, diagnostics = MODULE.project_and_cap_auxiliary_gradients(
            [auxiliary],
            [torch.tensor([0.25, -0.5])],
            absolute_cap=0.0,
            task_ratio_cap=0.0,
            safety_tasks=([torch.tensor([0.1, -0.2])],),
        )

        assert torch.equal(delivered[0], torch.zeros_like(auxiliary))
        assert diagnostics["numeric_valid"].item() == 0.0
        assert diagnostics["aux_numeric_valid"].item() == 0.0
        assert all(torch.isfinite(value).all() for value in diagnostics.values())


def test_projection_diagnostics_stay_finite_when_valid_norm_reductions_overflow():
    huge = torch.full((8,), 1e19)
    assert torch.isfinite(huge.square()).all()
    assert not torch.isfinite(huge.square().sum())

    delivered, diagnostics = MODULE.project_and_cap_auxiliary_gradients(
        [torch.ones(8)],
        [huge],
        absolute_cap=1.0,
        task_ratio_cap=0.25,
    )

    assert torch.equal(delivered[0], torch.zeros(8))
    assert diagnostics["numeric_valid"].item() == 0.0
    assert diagnostics["task_numeric_valid"].item() == 0.0
    assert all(torch.isfinite(value).all() for value in diagnostics.values())


def test_zero_caps_are_finite_and_exactly_suppress_valid_shared_auxiliary():
    delivered, diagnostics = MODULE.project_and_cap_auxiliary_gradients(
        [torch.tensor([2.0, -3.0])],
        [torch.tensor([0.5, 0.25])],
        absolute_cap=0.0,
        task_ratio_cap=0.0,
    )

    assert torch.equal(delivered[0], torch.zeros(2))
    assert diagnostics["numeric_valid"].item() == 1.0
    assert diagnostics["delivered_norm"].item() == 0.0
    assert all(torch.isfinite(value).all() for value in diagnostics.values())


def test_roundoff_scale_negative_safety_dot_does_not_veto_whole_block():
    # The two task objectives cancel in aggregate.  One has a harmless -1e-8 dot
    # caused at float32 roundoff scale; an exact ``dot < 0`` veto dropped this block.
    delivered, diagnostics = MODULE.project_and_cap_auxiliary_gradients(
        [torch.tensor([1.0, 0.0])],
        [torch.tensor([0.0, 0.0])],
        absolute_cap=10.0,
        task_ratio_cap=10.0,
        safety_tasks=(
            [torch.tensor([-1e-8, 1.0])],
            [torch.tensor([1e-8, -1.0])],
        ),
    )

    torch.testing.assert_close(delivered[0], torch.tensor([1.0, 0.0]))
    assert diagnostics["objective_veto_fraction"].item() == 0.0


def test_aligned_objectives_never_false_veto_after_float32_pcgrad_stress():
    torch.manual_seed(8301)
    random.seed(8301)
    false_vetoes = 0
    # The full 50k deterministic audit is retained in the research log; this focused
    # regression keeps the exact generator and enough adversarial scales for CI speed.
    for _ in range(2_000):
        dimension = random.randint(2, 512)
        task = torch.randn(dimension)
        task = task / task.norm() * (10 ** random.uniform(-6, -0.3))
        orthogonal = torch.randn(dimension)
        orthogonal -= orthogonal.dot(task) / task.dot(task) * task
        orthogonal = (
            orthogonal
            / orthogonal.norm()
            * (10 ** random.uniform(-6, -0.3))
        )
        auxiliary = orthogonal - task * random.uniform(0.01, 10.0)

        _, diagnostics = MODULE.project_and_cap_auxiliary_gradients(
            [auxiliary],
            [task],
            absolute_cap=1.0,
            task_ratio_cap=1.0,
            groups=[0],
            safety_tasks=([0.5 * task], [0.5 * task]),
        )
        false_vetoes += int(diagnostics["objective_veto_fraction"].item() != 0.0)

    assert false_vetoes == 0


def test_invalid_or_zero_capped_auxiliary_matches_task_only_adam_with_momentum():
    cases = (
        (torch.tensor([float("nan"), 1.0]), 1.0, 0.25),
        (torch.tensor([float("inf"), -1.0]), 1.0, 0.25),
        (torch.tensor([1e30, -1e30]), 1.0, 0.25),
        (torch.tensor([4.0, -3.0]), 0.0, 0.0),
    )
    for auxiliary, absolute_cap, ratio_cap in cases:
        reference = torch.nn.Parameter(torch.tensor([0.4, -0.7]))
        candidate = torch.nn.Parameter(reference.detach().clone())
        reference_optimizer = torch.optim.Adam([reference], lr=0.03)
        candidate_optimizer = torch.optim.Adam([candidate], lr=0.03)

        # Establish nonzero first and second moments before the compared update.
        warmup = torch.tensor([0.6, -0.2])
        MODULE.install_gradient_transaction([reference], ({reference: warmup},), {})
        reference_optimizer.step()
        MODULE.install_gradient_transaction([candidate], ({candidate: warmup},), {})
        candidate_optimizer.step()

        task = torch.tensor([-0.3, 0.8])
        MODULE.install_gradient_transaction([reference], ({reference: task},), {})
        reference_optimizer.step()
        delivered, _ = MODULE.project_and_cap_auxiliary_gradients(
            [auxiliary],
            [task],
            absolute_cap=absolute_cap,
            task_ratio_cap=ratio_cap,
        )
        MODULE.install_gradient_transaction(
            [candidate],
            ({candidate: task},),
            {candidate: delivered[0]},
        )
        candidate_optimizer.step()

        assert torch.equal(candidate, reference)
        for name, reference_value in reference_optimizer.state[reference].items():
            candidate_value = candidate_optimizer.state[candidate][name]
            if torch.is_tensor(reference_value):
                assert torch.equal(candidate_value, reference_value)
            else:
                assert candidate_value == reference_value


def test_invalid_private_auxiliary_zero_advances_then_restores_existing_momentum():
    first = torch.nn.Parameter(torch.tensor([0.5, -0.25]))
    second = torch.nn.Parameter(torch.tensor([-0.1, 0.9]))
    optimizer = torch.optim.Adam([first, second], lr=0.02)
    initial = MODULE.apply_private_auxiliary_optimizer_transaction(
        optimizer,
        [first, second],
        [torch.tensor([0.4, -0.7]), torch.tensor([0.2, 0.3])],
        True,
    )
    assert initial["numeric_valid"].item() == 1.0

    for invalid in (
        torch.tensor([float("nan"), 1.0]),
        torch.tensor([float("inf"), -1.0]),
        torch.tensor([1e30, -1e30]),
    ):
        parameter_before = [first.detach().clone(), second.detach().clone()]
        state_before = {
            parameter: {
                name: value.detach().clone() if torch.is_tensor(value) else value
                for name, value in optimizer.state[parameter].items()
            }
            for parameter in (first, second)
        }
        diagnostics = MODULE.apply_private_auxiliary_optimizer_transaction(
            optimizer,
            [first, second],
            [invalid, torch.tensor([0.2, 0.3])],
            True,
        )

        assert diagnostics["numeric_valid"].item() == 0.0
        assert diagnostics["step_norm"].item() == 0.0
        assert all(torch.isfinite(value).all() for value in diagnostics.values())
        assert torch.equal(first, parameter_before[0])
        assert torch.equal(second, parameter_before[1])
        beta1, beta2 = optimizer.param_groups[0]["betas"]
        for parameter in (first, second):
            assert optimizer.state[parameter]["step"].item() == (
                state_before[parameter]["step"].item() + 1
            )
            torch.testing.assert_close(
                optimizer.state[parameter]["exp_avg"],
                beta1 * state_before[parameter]["exp_avg"],
            )
            torch.testing.assert_close(
                optimizer.state[parameter]["exp_avg_sq"],
                beta2 * state_before[parameter]["exp_avg_sq"],
            )
            assert all(
                torch.isfinite(value).all()
                for value in optimizer.state[parameter].values()
                if torch.is_tensor(value)
            )


def test_private_optimizer_repairs_bad_moment_and_rejects_parameter_proposal():
    parameter = torch.nn.Parameter(torch.tensor([0.5, -0.25]))
    optimizer = torch.optim.Adam([parameter], lr=0.02)
    MODULE.apply_private_auxiliary_optimizer_transaction(
        optimizer, [parameter], [torch.tensor([0.4, -0.7])], True
    )
    optimizer.state[parameter]["exp_avg"][0] = float("inf")
    parameter_before = parameter.detach().clone()

    diagnostics = MODULE.apply_private_auxiliary_optimizer_transaction(
        optimizer, [parameter], [torch.tensor([0.2, 0.3])], True
    )

    assert diagnostics["numeric_valid"].item() == 0.0
    assert torch.equal(parameter, parameter_before)
    assert all(
        torch.isfinite(value).all()
        for value in optimizer.state[parameter].values()
        if torch.is_tensor(value)
    )


def test_task_gradients_are_not_silently_sanitized_by_transaction_installer():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    MODULE.install_gradient_transaction(
        [parameter],
        ({parameter: torch.tensor([float("nan")])},),
        {parameter: torch.zeros(1)},
    )

    assert torch.isnan(parameter.grad).all()
    assert "error_if_nonfinite=True" in SCRIPT.read_text()


def test_training_loop_wires_fail_closed_shared_and_private_transactions():
    source = SCRIPT.read_text()

    assert "optimizer = optim.Adam(task_params" in source
    assert "successor_optimizer = (" in source
    assert "auxiliary_transaction_valid = auxiliary_gradients_are_adam_safe(" in source
    assert "transaction_valid=auxiliary_transaction_valid" in source
    assert source.count("install_gradient_transaction(") == 2
    assert source.count("apply_private_auxiliary_optimizer_transaction(") == 2
    assert "successor_optimizer.param_groups[0][\"lr\"] = lrnow" in source
    assert "nn.utils.clip_grad_norm_(\n                            successor_predictor_params" not in source


def test_adam_safety_check_packs_gradients_before_one_reduction():
    source = ast.get_source_segment(
        SCRIPT.read_text(),
        next(
            node
            for node in ast.walk(ast.parse(SCRIPT.read_text()))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "auxiliary_gradients_are_adam_safe"
        ),
    )

    assert source is not None
    assert "flat = torch.cat(present)" in source
    assert "torch.isfinite(flat.square().sum())" in source
    assert "for gradient in gradients" in source  # packing only
    assert "part_square" not in source


def test_projection_sanitizes_once_per_logical_block_not_per_parameter():
    source_text = SCRIPT.read_text()
    source = ast.get_source_segment(
        source_text,
        next(
            node
            for node in ast.walk(ast.parse(source_text))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "project_and_cap_auxiliary_gradients"
        ),
    )

    assert source is not None
    assert "raw_flat_auxiliary = torch.cat(" in source
    assert "safe_flat_auxiliary = torch.where(" in source
    assert "flat_delivered = torch.where(" in source
    assert source.count("torch.where(") <= 6
    assert "safe_float" not in source
    assert "delivered[index] = torch.where(" not in source


def test_ema_update_moves_parameters_and_copies_buffers_without_gradients():
    class WithBuffer(torch.nn.Module):
        def __init__(self, weight, marker):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([weight]))
            self.register_buffer("marker", torch.tensor([marker]))

    online = WithBuffer(2.0, 7.0)
    target = WithBuffer(0.0, -1.0).requires_grad_(False)

    MODULE.ema_update(target, online, decay=0.75)

    torch.testing.assert_close(target.weight, torch.tensor([0.5]))
    torch.testing.assert_close(target.marker, torch.tensor([7.0]))
    assert not target.weight.requires_grad
    assert target.weight.grad is None


def test_auxiliary_semantic_probes_freeze_policy_and_critic_decoders():
    torch.manual_seed(8)
    args = MODULE.Args(hidden=8, k_blocks=1, n_experts=2, num_bins=11)
    agent = MODULE.Agent(DummyVectorEnv(), args)
    observations = torch.randn(7, 5)
    target_observations = torch.randn(7, 5)
    actions = torch.randn(7, 2).tanh()

    feature = agent.get_actor_feat(observations)
    with torch.no_grad():
        target_feature = agent.get_target_feat(target_observations)
    direct, _, reward, value = agent.get_successor_predictions(feature, actions)
    agent.innovation_scale.fill_(7.0)
    predicted_next = agent.reconstruct_next_feature(feature, direct)
    torch.testing.assert_close(predicted_next, feature + direct)
    policy_loss = torch.distributions.kl_divergence(
        agent.frozen_actor_dist(target_feature),
        agent.frozen_actor_dist(predicted_next),
    ).sum(-1).mean()
    value_target = torch.softmax(agent.frozen_value_logits(target_feature)[:, 0], -1)
    value_loss = -(
        value_target.detach()
        * torch.log_softmax(agent.frozen_value_logits(predicted_next)[:, 0], -1)
    ).sum(-1).mean()
    (policy_loss + value_loss + reward.square().mean() + value.square().mean()).backward()

    policy_decoder_parameters = (
        list(agent.actor_alpha_head.parameters())
        + list(agent.actor_beta_head.parameters())
    )
    assert all(parameter.grad is None for parameter in policy_decoder_parameters)
    assert all(parameter.grad is None for parameter in agent.critic_head.parameters())
    assert any(parameter.grad is not None for parameter in agent.successor_model.parameters())
    assert any(parameter.grad is not None for parameter in agent.successor_trunk_parameters())
    assert all(not parameter.requires_grad for parameter in agent.target_encoder.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in agent.target_successor_model.parameters()
    )


def test_target_modules_are_excluded_from_the_optimizer_parameter_set():
    args = MODULE.Args(hidden=8, k_blocks=1, n_experts=2, num_bins=11)
    agent = MODULE.Agent(DummyVectorEnv(), args)
    online_ids = {id(parameter) for parameter in agent.online_parameters()}

    assert online_ids.isdisjoint(
        id(parameter) for parameter in agent.target_encoder.parameters()
    )
    assert online_ids.isdisjoint(
        id(parameter) for parameter in agent.target_successor_model.parameters()
    )
    groups = agent.successor_trunk_parameter_groups()
    assert len(groups) == len(agent.successor_trunk_parameters())
    assert set(groups) == set(range(args.k_blocks + 2))
    task_ids = {id(parameter) for parameter in agent.task_parameters()}
    predictor_ids = {id(parameter) for parameter in agent.successor_predictor_parameters()}
    assert task_ids.isdisjoint(predictor_ids)
    assert task_ids | predictor_ids == online_ids


def test_horizon_zero_value_path_matches_full_head_without_large_mtp_output():
    args = MODULE.Args(hidden=8, k_blocks=1, n_experts=2, num_bins=11)
    agent = MODULE.Agent(DummyVectorEnv(), args)
    observations = torch.randn(9, 5)

    torch.testing.assert_close(agent.get_value_h0(observations), agent.get_value(observations)[:, 0])


def test_frozen_policy_bootstrap_action_is_env_space_and_gradient_free():
    args = MODULE.Args(hidden=8, k_blocks=1, n_experts=2, num_bins=11)
    agent = MODULE.Agent(DummyVectorEnv(), args)
    observations = torch.randn(9, 5, requires_grad=True)

    action = agent.sample_frozen_policy_action(observations)

    assert action.shape == (9, 2)
    assert not action.requires_grad
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)
    assert observations.grad is None


def test_successor_ema_is_called_once_after_the_entire_epoch_loop():
    source = SCRIPT.read_text()
    update = "agent.update_successor_targets(args.successor_ema_decay)"
    tree = ast.parse(source)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    update_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update_successor_targets"
    ]

    assert source.count(update) == 1
    assert source.index(update) > source.index("for epoch in range(args.update_epochs):")
    assert source.index(update) < source.index("y_pred, y_true =")
    assert len(update_calls) == 1
    ancestors = []
    node = update_calls[0]
    while node in parents:
        node = parents[node]
        ancestors.append(node)
    assert not any(
        isinstance(node, ast.For)
        and "args.update_epochs" in ast.unparse(node.iter)
        for node in ancestors
    )
