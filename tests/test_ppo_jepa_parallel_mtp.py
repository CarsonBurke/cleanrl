"""CUDA contracts for direct parallel MTP; execute through mlq."""
import pytest
import torch

from cleanrl import ppo_continuous_action_jepa_multistep_v9 as recurrent
from cleanrl import ppo_continuous_action_jepa_parallel_mtp_v12 as model
from cleanrl.shared.runtime import configure_runtime


@pytest.fixture(autouse=True)
def cuda_runtime():
    assert torch.cuda.is_available()
    configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)


def branch(horizons=(1, 4, 16), gradient="stopped"):
    return model.LeWMBranch(6, 16, 8, "mse", gradient, horizons).cuda()


def data():
    return (torch.randn(32, 64, device="cuda", requires_grad=True),
            torch.randn(32, 3, 64, device="cuda", requires_grad=True),
            torch.rand(32, 16, 6, device="cuda", requires_grad=True),
            torch.ones(32, 3, device="cuda", dtype=torch.bool))


def predictions(network, inputs):
    captured = []
    hook = network.pred_proj.register_forward_hook(lambda module, args, output: captured.append(output.detach().clone()))
    try:
        network(*inputs)
    finally:
        hook.remove()
    return captured[0].reshape(inputs[0].shape[0], -1, 64)


def test_horizons_use_current_state_and_only_their_ordered_action_prefix():
    network = branch()
    # AdaLN-zero intentionally ignores actions at initialization; activate the learned gate.
    with torch.no_grad():
        network.predictor.modulation[-1].bias[128:].fill_(1.0)
        network.predictor.modulation[-1].weight.normal_(std=0.02)
    inputs = data()
    base = predictions(network, inputs)
    changed_actions = inputs[2].detach().clone()
    changed_actions[:, 4:] = 1.0 - changed_actions[:, 4:]
    changed = predictions(network, (inputs[0], inputs[1], changed_actions, inputs[3]))
    torch.testing.assert_close(base[:, :2], changed[:, :2], rtol=0, atol=0)
    assert not torch.allclose(base[:, 2], changed[:, 2])
    permuted = inputs[2].detach().clone()
    permuted[:, :4] = permuted[:, :4].flip(1)
    assert not torch.allclose(base[:, 1], predictions(network, (inputs[0], inputs[1], permuted, inputs[3]))[:, 1])
    # Altering H1 conditioning cannot propagate through a predicted state into H4/H16.
    with torch.no_grad():
        for parameter in network.action_encoder.parameters():
            parameter.add_(0.25)
    changed = predictions(network, inputs)
    torch.testing.assert_close(base[:, 1:], changed[:, 1:], rtol=0, atol=0)
    assert not torch.allclose(base[:, 0], changed[:, 0])


@pytest.mark.parametrize("gradient", ["attached", "stopped"])
def test_target_stop_keeps_sigreg_attached_and_invalid_padding_inert(gradient):
    network = branch(gradient=gradient)
    current, targets, actions, valid = data()
    valid[-8:, 1:] = False
    pred, reg, _ = network(current, targets, actions, valid)
    target_grad = torch.autograd.grad(pred, targets, retain_graph=True)[0]
    assert bool(target_grad.any()) == (gradient == "attached")
    assert not bool(target_grad[-8:, 1:].any())
    assert bool(torch.autograd.grad(reg, targets, retain_graph=True)[0][:, 0].any())
    assert torch.autograd.grad(pred + reg, actions, allow_unused=True)[0] is None
    padded_targets, padded_actions = targets.detach().clone(), actions.detach().clone()
    padded_targets[-8:, 1:] = float("nan")
    padded_actions[-8:, 1:] = float("nan")
    torch.cuda.manual_seed(42)
    reference = network(current, targets, actions, valid)
    torch.cuda.manual_seed(42)
    padded = network(current, padded_targets, padded_actions, valid)
    torch.testing.assert_close(reference[0], padded[0])
    torch.testing.assert_close(reference[1], padded[1])
    network.zero_grad(set_to_none=True)
    (padded[0] + padded[1]).backward()
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)


def test_single_horizon_matches_frozen_recurrent_control_and_rng():
    torch.manual_seed(19)
    original = recurrent.LeWMBranch(6, 16, 8, "mse", "stopped", (1,)).cuda()
    original_rng = torch.get_rng_state()
    torch.manual_seed(19)
    direct = branch((1,))
    torch.testing.assert_close(torch.get_rng_state(), original_rng)
    current, targets, actions, valid = data()
    inputs = (current, targets[:, :1], actions[:, :1], valid[:, :1])
    torch.cuda.manual_seed(71)
    expected = original(*inputs)
    torch.cuda.manual_seed(71)
    actual = direct(*inputs)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    expected_grad = torch.autograd.grad(expected[0] + expected[1], tuple(original.parameters()))
    actual_grad = torch.autograd.grad(actual[0] + actual[1], tuple(direct.parameters()))
    for left, right in zip(expected_grad, actual_grad, strict=True):
        torch.testing.assert_close(left, right)


def test_compiled_parallel_objective_updates_all_prefix_forecasters():
    network = branch()
    optimizer = torch.optim.AdamW(network.parameters(), lr=5e-5)
    current, targets, actions, valid = data()
    # Exercise actual SSL512 batch shape, not a reduced training run.
    inputs = (current.detach().repeat(16, 1), targets.detach().repeat(16, 1, 1),
              actions.detach().repeat(16, 1, 1), valid.repeat(16, 1))
    compiled = torch.compile(network, mode="reduce-overhead", fullgraph=True)
    before = network.pred_proj.state_dict()
    before = {key: value.clone() for key, value in before.items()}
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        pred, reg, _ = compiled(*inputs)
        (pred + 0.09 * reg).backward()
        assert torch.isfinite(pred + reg)
        optimizer.step()
    assert any(not torch.equal(value, before[key]) for key, value in network.pred_proj.state_dict().items())
    assert all(any(p.grad is not None and bool(p.grad.any()) for p in encoder.parameters())
               for encoder in network.prefix_encoders)
