"""CUDA contracts for direct two-level MTP; execute through mlq."""
import pytest
import torch

from cleanrl import ppo_continuous_action_jepa_parallel_hierarchy_v13 as model
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(autouse=True)
def cuda_runtime():
    assert torch.cuda.is_available()
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)


def branch(consistency="on"):
    return model.LeWMBranch(6, 16, 8, "mse", "stopped", (1, 4, 16), 4, (4, 16), consistency).cuda()


def data(rows=32):
    return (torch.randn(rows, 64, device="cuda", requires_grad=True),
            torch.randn(rows, 3, 64, device="cuda", requires_grad=True),
            torch.rand(rows, 16, 6, device="cuda", requires_grad=True),
            torch.ones(rows, 3, device="cuda", dtype=torch.bool))


def has_gradient(loss, parameters):
    gradients = torch.autograd.grad(loss, tuple(parameters), retain_graph=True, allow_unused=True)
    return any(gradient is not None and bool(gradient.any()) for gradient in gradients)


def test_each_level_predicts_independent_horizons_from_ordered_prefixes():
    network = branch()
    with torch.no_grad():
        for predictor in (network.predictor, network.coarse_predictor):
            predictor.modulation[-1].bias[128:].fill_(1.0)
            predictor.modulation[-1].weight.normal_(std=0.02)
    current, following, actions, valid = data()
    fine, coarse = network.rollouts(current, following[:, 0], actions, valid)
    changed_actions = actions.detach().clone()
    changed_actions[:, 4:] = 1.0 - changed_actions[:, 4:]
    changed_fine, changed_coarse = network.rollouts(current, following[:, 0], changed_actions, valid)
    for horizon in (1, 4):
        torch.testing.assert_close(fine[horizon], changed_fine[horizon], rtol=0, atol=0)
    torch.testing.assert_close(coarse[4], changed_coarse[4], rtol=0, atol=0)
    assert not torch.allclose(fine[16], changed_fine[16])
    assert not torch.allclose(coarse[16], changed_coarse[16])
    with torch.no_grad():
        for encoder in (network.action_encoder, network.coarse_action_encoder):
            for parameter in encoder.parameters():
                parameter.add_(0.25)
    changed_fine, changed_coarse = network.rollouts(current, following[:, 0], actions, valid)
    torch.testing.assert_close(fine[16], changed_fine[16], rtol=0, atol=0)
    torch.testing.assert_close(coarse[16], changed_coarse[16], rtol=0, atol=0)
    assert not torch.allclose(fine[1], changed_fine[1])
    assert not torch.allclose(coarse[4], changed_coarse[4])


def test_consistency_stops_coarse_teacher_but_observed_anchors_train_both_levels():
    network = branch()
    current, targets, actions, valid = data()
    pred, reg, consistency, _ = network(current, targets, actions, valid)
    assert has_gradient(consistency, network.predictor.parameters())
    assert has_gradient(consistency, network.coarse_projector.parameters())
    assert not has_gradient(consistency, network.coarse_predictor.parameters())
    assert not has_gradient(consistency, network.coarse_pred_proj.parameters())
    assert has_gradient(pred, network.coarse_predictor.parameters())
    assert not bool(torch.autograd.grad(pred, targets, retain_graph=True)[0].any())
    assert bool(torch.autograd.grad(reg, targets, retain_graph=True)[0][:, 0].any())
    assert torch.autograd.grad(pred + reg + consistency, actions, allow_unused=True)[0] is None


def test_off_on_keep_initialization_anchors_and_regularization_draws_identical():
    torch.manual_seed(19)
    off = branch("off")
    rng = torch.get_rng_state()
    torch.manual_seed(19)
    on = branch("on")
    torch.testing.assert_close(torch.get_rng_state(), rng)
    for name, value in off.state_dict().items():
        torch.testing.assert_close(value, on.state_dict()[name])
    inputs = data()
    torch.cuda.manual_seed(71)
    expected = off(*inputs)
    cuda_rng = torch.cuda.get_rng_state()
    torch.cuda.manual_seed(71)
    actual = on(*inputs)
    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng)
    torch.testing.assert_close(expected[0], actual[0])
    torch.testing.assert_close(expected[1], actual[1])
    assert expected[2] == 0
    assert actual[2] > 0


@pytest.mark.parametrize("consistency", ["off", "on"])
def test_compiled_two_level_objective_updates_both_forecasters(consistency):
    network = branch(consistency)
    optimizer = torch.optim.AdamW(network.parameters(), lr=5e-5)
    inputs = tuple(value.detach() for value in data(512))
    compiled = torch.compile(network, mode="reduce-overhead", fullgraph=True)
    initial = {name: value.detach().clone() for name, value in network.named_parameters()}
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        pred, reg, agreement, _ = compiled(*inputs)
        objective = pred + 0.09 * reg + agreement
        objective.backward()
        assert torch.isfinite(objective)
        optimizer.step()
    for prefix in ("fine_prefix_encoders.", "coarse_prefix_encoders."):
        assert any(not torch.equal(parameter, initial[name])
                   for name, parameter in network.named_parameters() if name.startswith(prefix))
