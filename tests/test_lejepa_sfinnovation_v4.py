import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "ppo_continuous_action_lejepa_sfinnovation_v4.py"
)
SPEC = importlib.util.spec_from_file_location("lejepa_sfinnovation_v4", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_successor_innovation_bootstraps_truncation_but_not_termination():
    phi = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    psi_current = torch.tensor([[[0.5]], [[1.0]], [[1.5]]])
    psi_next = torch.tensor([[[10.0]], [[20.0]], [[30.0]]])
    terminations = torch.tensor([[0.0], [1.0], [0.0]])
    transition_valids = torch.tensor([[1.0], [1.0], [0.0]])

    target, innovation = MODULE.successor_innovation(
        phi,
        psi_current,
        psi_next,
        terminations,
        transition_valids,
        gamma=0.5,
    )

    torch.testing.assert_close(target[:, 0, 0], torch.tensor([6.0, 2.0, 3.0]))
    torch.testing.assert_close(
        innovation[:, 0, 0],
        torch.tensor([5.5, 1.0, 1.5]),
    )


def test_reward_projection_matches_scalar_td_when_reward_is_linear_in_phi():
    torch.manual_seed(4)
    time, envs, features = 7, 3, 5
    phi = torch.randn(time, envs, features)
    psi_current = torch.randn_like(phi)
    psi_next = torch.randn_like(phi)
    reward_weights = torch.randn(features)
    terminations = torch.zeros(time, envs)
    transition_valids = torch.ones(time, envs)

    _, innovation = MODULE.successor_innovation(
        phi,
        psi_current,
        psi_next,
        terminations,
        transition_valids,
        gamma=0.97,
    )
    projected = innovation @ reward_weights
    scalar_td = (
        phi @ reward_weights
        + 0.97 * (psi_next @ reward_weights)
        - psi_current @ reward_weights
    )
    torch.testing.assert_close(projected, scalar_td)


def test_innovation_has_no_trace_across_time():
    phi = torch.zeros(4, 1, 2)
    psi_current = torch.zeros_like(phi)
    psi_next = torch.zeros_like(phi)
    terminations = torch.zeros(4, 1)
    transition_valids = torch.ones(4, 1)

    _, original = MODULE.successor_innovation(
        phi,
        psi_current,
        psi_next,
        terminations,
        transition_valids,
        gamma=0.99,
    )
    phi[-1] = 1000.0
    _, changed = MODULE.successor_innovation(
        phi,
        psi_current,
        psi_next,
        terminations,
        transition_valids,
        gamma=0.99,
    )

    torch.testing.assert_close(changed[:-1], original[:-1])
    assert not torch.equal(changed[-1], original[-1])

