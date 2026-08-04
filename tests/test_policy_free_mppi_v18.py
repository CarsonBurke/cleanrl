import importlib.util
import inspect
import sys
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "embedding-optimization"
    / "ppo_continuous_action_policy_free_mppi_v18.py"
)
SPEC = importlib.util.spec_from_file_location("policy_free_mppi_v18", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def small_args(**overrides):
    values = dict(
        latent_dim=4,
        hidden_dim=16,
        model_horizon=3,
        planner_horizon=3,
        sigreg_projections=8,
        sigreg_knots=5,
    )
    values.update(overrides)
    return MODULE.Args(**values)


def make_batch(batch_size=8, horizon=3, action_dim=2):
    return MODULE.ReplayBatch(
        observations=torch.randn(batch_size, horizon + 1, 17),
        actions=torch.randn(batch_size, horizon, action_dim).clamp(-1.0, 1.0),
        interval_velocities=torch.randn(batch_size, horizon),
        valid=torch.ones(batch_size, horizon, dtype=torch.bool),
    )


def test_world_model_objective_is_the_only_gradient_route():
    torch.manual_seed(1)
    args = small_args()
    agent = MODULE.Agent(17, 2, args)
    loss, metrics = MODULE.world_model_objective(
        agent,
        MODULE.SIGReg(args.sigreg_projections, args.sigreg_knots),
        make_batch(),
        args,
    )
    loss.backward()
    assert all(parameter.grad is not None for parameter in agent.parameters())
    assert any(parameter.grad.abs().sum() > 0 for parameter in agent.parameters())
    assert set(agent._modules) == {
        "encoder",
        "dynamics",
        "velocity_head",
        "interval_velocity_head",
    }
    assert set(metrics) == {
        "wm_loss",
        "wm_forward_loss",
        "wm_sigreg_loss",
        "wm_state_velocity_huber",
        "wm_factual_interval_huber",
        "wm_predicted_interval_huber",
    }


class ExactTimingModel(torch.nn.Module):
    latent_dim = 1
    action_dim = 1

    def predict_next(self, z, action):
        return z + action

    def predict_interval_velocity(self, successor_z):
        return 2.0 * successor_z.squeeze(-1)


def test_population_score_uses_successor_velocity_and_exact_action_cost():
    model = ExactTimingModel()
    z = torch.tensor([[1.0]])
    sequences = torch.tensor([[[[2.0], [3.0]]]])
    score, velocities = MODULE.action_sequence_population_score(
        model,
        z,
        sequences,
        action_cost_coef=0.1,
        return_velocities=True,
    )
    torch.testing.assert_close(velocities, torch.tensor([[[6.0, 12.0]]]))
    expected = (6.0 - 0.1 * 2.0**2) + (12.0 - 0.1 * 3.0**2)
    torch.testing.assert_close(score, torch.tensor([[expected]]))


def test_antithetic_perturbations_include_mean_and_are_seed_deterministic():
    first_generator = torch.Generator().manual_seed(17)
    second_generator = torch.Generator().manual_seed(17)
    different_generator = torch.Generator().manual_seed(18)
    arguments = dict(
        updates=2,
        batch_size=3,
        population=7,
        horizon=4,
        action_dim=2,
        device="cpu",
        dtype=torch.float32,
    )
    first = MODULE.antithetic_gaussian_perturbations(
        **arguments, generator=first_generator
    )
    second = MODULE.antithetic_gaussian_perturbations(
        **arguments, generator=second_generator
    )
    different = MODULE.antithetic_gaussian_perturbations(
        **arguments, generator=different_generator
    )
    assert first.shape == (2, 3, 7, 4, 2)
    torch.testing.assert_close(first[:, :, 0], torch.zeros_like(first[:, :, 0]))
    torch.testing.assert_close(first[:, :, 1:4], -first[:, :, 4:7])
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert not torch.equal(first, different)


def test_antithetic_population_contract_rejects_even_or_degenerate_sizes():
    for population in (1, 4):
        try:
            MODULE.antithetic_gaussian_perturbations(
                1,
                2,
                population,
                3,
                1,
                device="cpu",
                dtype=torch.float32,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid antithetic population was accepted")


def test_warm_start_shifts_repeats_tail_and_zeroes_resets():
    sequence = torch.tensor(
        [
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
        ]
    )
    shifted = MODULE.shift_action_sequence(
        sequence, torch.tensor([False, True])
    )
    torch.testing.assert_close(
        shifted[0], torch.tensor([[2.0], [3.0], [3.0]])
    )
    torch.testing.assert_close(shifted[1], torch.zeros_like(shifted[1]))


def test_mppi_is_bounded_deterministic_for_seed_and_gradient_free():
    torch.manual_seed(4)
    args = small_args()
    agent = MODULE.Agent(17, 2, args)
    z = torch.randn(3, args.latent_dim)
    warm_start = torch.full((3, args.planner_horizon, 2), 0.9)

    def plan(seed):
        perturbations = MODULE.antithetic_gaussian_perturbations(
            2,
            3,
            7,
            args.planner_horizon,
            2,
            device="cpu",
            dtype=torch.float32,
            generator=torch.Generator().manual_seed(seed),
        )
        return MODULE.mppi_plan(
            agent,
            z,
            warm_start,
            perturbations,
            noise_std=0.5,
            temperature=1.0,
            action_cost_coef=0.1,
        )

    first_plan, first_metrics = plan(23)
    second_plan, second_metrics = plan(23)
    different_plan, _ = plan(24)
    torch.testing.assert_close(first_plan, second_plan, rtol=0.0, atol=0.0)
    for key in first_metrics:
        torch.testing.assert_close(
            first_metrics[key], second_metrics[key], rtol=0.0, atol=0.0
        )
    assert not torch.equal(first_plan, different_plan)
    assert torch.all(first_plan.abs() <= 1.0)
    assert not first_plan.requires_grad
    assert set(first_metrics) == {
        "selected_score",
        "predicted_interval_velocity",
        "score_spread",
        "weight_entropy",
        "effective_sample_size",
        "action_change",
        "sequence_change",
    }


def test_mppi_uses_soft_weighted_update_without_selection_path():
    source = inspect.getsource(MODULE.mppi_plan).lower()
    assert ".softmax(" in source
    assert "weights[:, :, none, none] * candidates" in source
    for forbidden in ("topk", "argsort", "elite", "quantile"):
        assert forbidden not in source


def test_defaults_are_h12_lightweight_policy_free_mppi():
    args = MODULE.Args()
    assert args.env_id == "HalfCheetah-v4"
    assert args.total_timesteps == 8_000_000
    assert args.seed == 1 and args.cuda
    assert args.model_horizon == args.planner_horizon == 12
    assert args.mppi_population == 65
    assert args.mppi_updates == 2
    assert args.mppi_noise_std == 0.5
    assert args.mppi_temperature == 1.0


def test_executable_has_no_learned_control_or_postwarm_random_action_path():
    source = SCRIPT.read_text()
    lowered = source.lower()
    for forbidden in (
        "intact_action_law",
        "intent_prescriber",
        "terminal_value",
        "critic",
        "actor",
        "action_aux",
    ):
        assert forbidden not in lowered
    assert source.count("optim.AdamW(") == 1
    assert source.count(".uniform_(-1.0, 1.0)") == 1
    main = source[source.index('if __name__ == "__main__":') :]
    assert "if global_step < args.warmup_steps:" in main
    assert "planned_action_sequence, diagnostics = planner_function(" in main
    assert "action = planned_action_sequence[:, 0]" in main
    assert "planner_generator.manual_seed(args.seed)" in main


def test_planner_is_vectorized_over_population_and_compile_targeted():
    score_source = inspect.getsource(MODULE.action_sequence_population_score)
    assert "for step in range(horizon):" in score_source
    assert "for candidate" not in score_source
    assert "batch_size * population" in score_source
    source = SCRIPT.read_text()
    assert "planner_function = torch.compile(" in source
    assert "torch.compiler.cudagraph_mark_step_begin()" in source
    assert "key: value.clone() for key, value in diagnostics.items()" in source
