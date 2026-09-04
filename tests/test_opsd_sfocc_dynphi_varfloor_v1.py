import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "cleanrl" / "opsd" / "core" / "sfocc" / "ppo_continuous_action_opsd_sfocc_dynphi_varfloor_v1.py"
PARENT_SCRIPT = ROOT / "cleanrl" / "opsd" / "core" / "sfocc" / "ppo_continuous_action_opsd_sfocc_dynphi_v1.py"


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


varfloor = load_script("opsd_sfocc_dynphi_varfloor_v1", SCRIPT)
parent = load_script("opsd_sfocc_dynphi_v1_parent", PARENT_SCRIPT)


def test_variance_floor_is_a_per_coordinate_linear_std_hinge():
    phi = torch.tensor([[-0.1, -1.0], [0.1, 1.0]], requires_grad=True)

    loss = varfloor.phi_variance_floor(phi, std_floor=0.5)
    expected = (0.5 - torch.sqrt(torch.tensor(0.01 + varfloor.VAR_FLOOR_EPS))) / 2
    torch.testing.assert_close(loss, expected)

    loss.backward()
    assert phi.grad is not None
    assert phi.grad[0, 0] > 0
    assert phi.grad[1, 0] < 0
    torch.testing.assert_close(phi.grad[:, 1], torch.zeros(2))


def test_exact_coordinate_collapse_has_zero_restorative_gradient():
    phi = torch.zeros(8, 3, requires_grad=True)

    loss = varfloor.phi_variance_floor(phi, std_floor=0.5)
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(0.49))
    assert phi.grad is not None
    torch.testing.assert_close(phi.grad, torch.zeros_like(phi))


def test_variance_floor_gradient_isolated_to_phi_encoder():
    torch.manual_seed(7)
    sf = varfloor.SFCritic(obs_dim=5, act_dim=2, phi_dim=8, hidden=16, dynphi=True)
    obs = torch.randn(64, 5)
    action = torch.randn(64, 2)

    loss = varfloor.phi_variance_floor(sf.phi(obs, action), std_floor=1.0)
    loss.backward()

    phi_grads = [parameter.grad for parameter in sf.phi_net.parameters()]
    assert len(phi_grads) == 6
    assert all(grad is not None and torch.isfinite(grad).all() for grad in phi_grads)
    assert sum(grad.square().sum() for grad in phi_grads) > 0
    assert all(parameter.grad is None for parameter in sf.psi_net.parameters())
    assert all(parameter.grad is None for parameter in sf.w_head.parameters())
    assert all(parameter.grad is None for parameter in sf.dyn_net.parameters())


def test_disabled_floor_preserves_parent_loss_exactly():
    torch.manual_seed(11)
    sf = varfloor.SFCritic(obs_dim=5, act_dim=2, phi_dim=8, hidden=16, dynphi=True)
    args = varfloor.Args(
        phi_dim=8,
        hidden=16,
        dynphi=True,
        var_floor=False,
    )
    obs = torch.randn(32, 5)
    action = torch.randn(32, 2)
    next_obs = torch.randn(32, 5)
    a_bar_next = torch.randn(32, 2)
    psi_target = torch.randn(32, 8)
    reward = torch.randn(32)

    actual = varfloor.sf_losses(
        sf, obs, action, next_obs, a_bar_next, psi_target, reward, args
    )
    expected = parent.sf_losses(
        sf, obs, action, next_obs, a_bar_next, psi_target, reward, args
    )

    assert actual[-1] is None
    for actual_loss, expected_loss in zip(actual[:-1], expected, strict=True):
        if actual_loss is None:
            assert expected_loss is None
        else:
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)


def test_enabled_total_adds_only_the_weighted_floor():
    torch.manual_seed(19)
    sf = varfloor.SFCritic(obs_dim=5, act_dim=2, phi_dim=8, hidden=16, dynphi=True)
    args = varfloor.Args(
        phi_dim=8,
        hidden=16,
        dynphi=True,
        var_floor=True,
        phi_std_floor=0.75,
        var_floor_coef=1.7,
    )
    obs = torch.randn(32, 5)
    action = torch.randn(32, 2)
    next_obs = torch.randn(32, 5)
    a_bar_next = torch.randn(32, 2)
    psi_target = torch.randn(32, 8)
    reward = torch.randn(32)

    actual = varfloor.sf_losses(
        sf, obs, action, next_obs, a_bar_next, psi_target, reward, args
    )
    parent_losses = parent.sf_losses(
        sf, obs, action, next_obs, a_bar_next, psi_target, reward, args
    )

    assert actual[-1] is not None
    torch.testing.assert_close(
        actual[0],
        parent_losses[0] + args.var_floor_coef * actual[-1],
        rtol=0,
        atol=0,
    )
