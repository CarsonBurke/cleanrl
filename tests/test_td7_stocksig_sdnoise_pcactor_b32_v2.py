from types import SimpleNamespace

import pytest
import torch

from cleanrl.td7_lesale_v1 import Args, LeSALEAgent, td7_pc_actor_batch


def pc_args(batch_size):
    return Args(
        hidden_dim=6,
        zs_dim=5,
        pc_actor=True,
        pc_actor_batch_size=batch_size,
        pc_actor_inference_steps=3,
        sd_noise=True,
        use_subsig=False,
        residual_predictor=False,
        buffer_size=64,
        batch_size=4,
        fused_adam=False,
        gpu_replay=False,
        torch_compile=False,
    )


def make_agent(batch_size, seed):
    torch.manual_seed(seed)
    writer = SimpleNamespace(add_scalar=lambda *args, **kwargs: None)
    return LeSALEAgent(
        4,
        3,
        1.0,
        pc_args(batch_size),
        torch.device("cpu"),
        writer,
    )


def test_default_batch_selection_is_the_exact_full_batch_without_views():
    state = torch.randn(7, 4)
    fixed_zs = torch.randn(7, 5)
    noise = torch.randn(7, 3)
    selected = td7_pc_actor_batch(state, fixed_zs, noise, 0)
    assert Args().pc_actor_batch_size == 0
    assert selected[0] is state
    assert selected[1] is fixed_zs
    assert selected[2] is noise


def test_configured_batch_selection_is_the_deterministic_leading_subset():
    state = torch.arange(28).view(7, 4)
    fixed_zs = torch.arange(35).view(7, 5)
    noise = torch.arange(21).view(7, 3)
    selected = td7_pc_actor_batch(state, fixed_zs, noise, 3)
    for actual, full in zip(selected, (state, fixed_zs, noise)):
        torch.testing.assert_close(actual, full[:3])
    capped = td7_pc_actor_batch(state, fixed_zs, noise, 100)
    for actual, full in zip(capped, (state, fixed_zs, noise)):
        torch.testing.assert_close(actual, full)
    with pytest.raises(ValueError, match="nonnegative"):
        td7_pc_actor_batch(state, fixed_zs, noise, -1)


@pytest.mark.parametrize("configured_size, expected_size", [(0, 4), (2, 2)])
def test_pc_actor_update_exactly_matches_manual_subset(configured_size, expected_size):
    automatic = make_agent(configured_size, seed=41)
    manual = make_agent(configured_size, seed=41)
    automatic.critic.requires_grad_(False)
    manual.critic.requires_grad_(False)
    generator = torch.Generator().manual_seed(412)
    state = torch.randn(4, 4, generator=generator)
    noise = torch.randn(4, 3, generator=generator)
    with torch.no_grad():
        automatic_zs = automatic.fixed_encoder.zs(state)
        manual_zs = manual.fixed_encoder.zs(state)
    torch.testing.assert_close(automatic_zs, manual_zs, rtol=0, atol=0)

    actor_loss, actor_log_pi, diagnostics = automatic._pc_actor_update(
        state, automatic_zs, noise
    )
    subset = slice(0, expected_size)
    terminal_force, expected_loss, expected_log_pi, raw_rms = (
        manual._pc_actor_terminal_force(
            state[subset], manual_zs[subset], noise[subset]
        )
    )
    expected_diagnostics = manual.pc_actor_trainer.step(
        state[subset],
        manual_zs[subset],
        terminal_force,
        manual.args.actor_lr,
    )
    expected_diagnostics["raw_terminal_force_rms"] = raw_rms
    expected_diagnostics["batch_size"] = terminal_force.new_tensor(expected_size)

    torch.testing.assert_close(actor_loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(actor_log_pi, expected_log_pi, rtol=0, atol=0)
    assert diagnostics.keys() == expected_diagnostics.keys()
    for name, expected in expected_diagnostics.items():
        torch.testing.assert_close(diagnostics[name], expected, rtol=0, atol=0)
    for name, expected in manual.actor.state_dict().items():
        torch.testing.assert_close(
            automatic.actor.state_dict()[name], expected, rtol=0, atol=0
        )


def test_full_shape_noise_draw_preserves_rng_cadence_before_subset_selection():
    shape = (256, 6)
    actual_generator = torch.Generator().manual_seed(73)
    full_noise = torch.randn(shape, generator=actual_generator)
    selected_noise = td7_pc_actor_batch(
        torch.empty(256, 17), torch.empty(256, 256), full_noise, 32
    )[2]
    actual_state = actual_generator.get_state()

    reference_generator = torch.Generator().manual_seed(73)
    expected_full_noise = torch.randn(shape, generator=reference_generator)
    torch.testing.assert_close(selected_noise, expected_full_noise[:32], rtol=0, atol=0)
    torch.testing.assert_close(
        actual_state, reference_generator.get_state(), rtol=0, atol=0
    )
