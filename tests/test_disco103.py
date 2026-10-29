"""Pin cleanrl.shared.disco103 to google-deepmind/disco_rl's JAX implementation.

The fixture comes from ``scripts/disco103_reference.py``, which runs the original
meta-network with the released Disco103 weights on random inputs.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from cleanrl.shared import disco103

REFERENCE = Path(__file__).with_name("fixtures") / "disco103_reference.npz"


@pytest.fixture(scope="module")
def reference():
    with np.load(REFERENCE) as archive:
        return {key: torch.as_tensor(archive[key]) for key in archive.files}


def taken_first(x, actions):
    """Swap each step's taken action into index 0 along the action axis (dim 2)."""
    steps, batch = actions.shape
    order = torch.arange(x.shape[2]).expand(steps, batch, -1).clone()
    rows = torch.arange(steps)[:, None], torch.arange(batch)[None]
    order[(*rows, torch.zeros_like(actions))] = actions
    order[(*rows, actions)] = 0
    index = order.reshape(steps, batch, -1, *([1] * (x.dim() - 3))).expand_as(x)
    return x.gather(2, index)


def meta_inputs(r):
    actions = r["actions"][:-1].long()
    current = lambda key: taken_first(r[key][:-1], actions)
    return disco103.MetaInputs(
        logits=current("logits"), next_logits=r["logits"][1:],
        behaviour_logits=current("behaviour_logits"), target_logits=current("target_logits"),
        rewards=r["rewards"], terminations=r["is_terminal"], resets=r["is_terminal"],
        values=r["v_scalar"][:-1], next_values=r["v_scalar"][1:],
        advantages=r["adv"], normalized_advantages=r["normalized_adv"],
        y=r["y"][:-1], next_y=r["y"][1:], target_y=r["target_y"][:-1], next_target_y=r["target_y"][1:],
        z=current("z"), next_z=r["z"][1:], target_z=current("target_z"), next_target_z=r["target_z"][1:],
        target_q=current("q"), qv_advantages=current("qv_adv"),
        normalized_qv_advantages=current("normalized_qv_adv"),
    ), actions


def test_meta_network_matches_jax(reference):
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs, actions = meta_inputs(reference)
    network = disco103.Disco103()
    with torch.no_grad():
        pi_hat, y_hat, z_hat, state = network(inputs, reference["lifetime_state"])
    torch.testing.assert_close(pi_hat, taken_first(reference["pi_hat"], actions), atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(y_hat, reference["y_hat"], atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(z_hat, reference["z_hat"], atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(state, reference["new_lifetime_state"], atol=2e-5, rtol=1e-4)


def test_meta_network_is_frozen_and_complete():
    network = disco103.Disco103()
    assert not list(network.parameters())
    assert sum(buffer.numel() for buffer in network.buffers()) == sum(
        array.size for array in np.load(disco103.WEIGHTS).values())


def test_categorical_value_matches_rlax(reference):
    value = disco103.TwoHotValue(601, 300.0)
    torch.testing.assert_close(value.loss(reference["value_logits"], reference["value_targets"]),
                               reference["value_ce"], atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(value.scalar(reference["value_logits"]), reference["value_scalars"],
                               atol=1e-3, rtol=1e-4)


def test_signed_parabolic_inverts_signed_hyperbolic():
    x = torch.tensor([-1e4, -3.0, -1e-3, 0.0, 2e-3, 7.5, 2e3], dtype=torch.float64)
    torch.testing.assert_close(disco103.signed_parabolic(disco103.signed_hyperbolic(x)), x)


def test_ema_normalization_matches_moving_average(reference):
    state = disco103.ema_init()
    for sample in reference["ema_samples"]:
        state = disco103.ema_update(state, sample)
    torch.testing.assert_close(disco103.ema_normalize(state, reference["ema_samples"][-1]),
                               reference["ema_normalized"], atol=1e-5, rtol=1e-5)
