import copy
import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "streaming-posterior-filter"
    / "ppo_continuous_action_spf_v1.py"
)
spec = importlib.util.spec_from_file_location("spf_v1", SCRIPT)
assert spec is not None
module = importlib.util.module_from_spec(spec)
loader = spec.loader
assert loader is not None
loader.exec_module(module)
StreamingPosteriorFilter = module.StreamingPosteriorFilter


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "SPF contract tests require CUDA"
    return torch.device("cuda")


def test_nominal_gain_and_first_update_match_temperature(device):
    parameter = torch.nn.Parameter(torch.zeros(2, 4, device=device))
    optimizer = StreamingPosteriorFilter([parameter], lr=3e-4, beta1=0.9, student_df=5.0)
    parameter.grad = torch.ones_like(parameter)

    optimizer.step()

    state = optimizer.state[parameter]
    torch.testing.assert_close(
        state["last_gain"],
        torch.full((2,), 0.1, device=device, dtype=state["last_gain"].dtype),
    )
    torch.testing.assert_close(parameter, torch.full_like(parameter, -3e-4), rtol=1e-5, atol=1e-7)
    assert torch.all(state["posterior_var"] > 0)
    assert torch.all(state["observation_var"] > 0)


def test_student_measurement_rejects_only_surprising_row(device):
    parameter = torch.nn.Parameter(torch.zeros(2, 8, device=device))
    optimizer = StreamingPosteriorFilter([parameter], lr=3e-4, beta1=0.9, student_df=5.0)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    second = torch.ones_like(parameter)
    second[1].fill_(100.0)
    parameter.grad = second
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["last_student_weight"][0] == 1.0
    assert state["last_student_weight"][1] < 0.01
    assert state["last_gain"][1] < state["last_gain"][0]
    assert state["last_delta_per_dim"][1] > state["last_delta_per_dim"][0]


def test_independent_rows_are_not_competitive(device):
    parameter = torch.nn.Parameter(torch.zeros(2, 8, device=device))
    optimizer = StreamingPosteriorFilter([parameter], lr=3e-4, beta1=0.9, student_df=5.0)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    second = torch.ones_like(parameter)
    second[1].fill_(100.0)
    parameter.grad = second
    optimizer.step()
    gain_with_other_row_outlier = optimizer.state[parameter]["last_gain"][0].clone()

    control = torch.nn.Parameter(torch.zeros(1, 8, device=device))
    control_optimizer = StreamingPosteriorFilter([control], lr=3e-4, beta1=0.9, student_df=5.0)
    control.grad = torch.ones_like(control)
    control_optimizer.step()
    control.grad = torch.ones_like(control)
    control_optimizer.step()
    control_gain = control_optimizer.state[control]["last_gain"][0]

    torch.testing.assert_close(gain_with_other_row_outlier, control_gain, rtol=0, atol=0)


def test_gradient_rescaling_preserves_parameter_update(device):
    low = torch.nn.Parameter(torch.zeros(2, 4, device=device))
    high = torch.nn.Parameter(torch.zeros(2, 4, device=device))
    low_optimizer = StreamingPosteriorFilter([low], lr=3e-4, beta1=0.9, student_df=5.0)
    high_optimizer = StreamingPosteriorFilter([high], lr=3e-4, beta1=0.9, student_df=5.0)

    for value in (1e-12, 5e-13, -2.5e-13, 7.5e-13):
        low.grad = torch.full_like(low, value)
        high.grad = torch.full_like(high, 100.0 * value)
        low_optimizer.step()
        high_optimizer.step()

    torch.testing.assert_close(low, high, rtol=2e-4, atol=2e-7)
    torch.testing.assert_close(
        low_optimizer.state[low]["last_gain"],
        high_optimizer.state[high]["last_gain"],
        rtol=2e-5,
        atol=2e-7,
    )


def test_extreme_finite_outlier_keeps_every_state_finite(device):
    parameter = torch.nn.Parameter(torch.zeros(2, 8, device=device))
    optimizer = StreamingPosteriorFilter([parameter], lr=3e-4, beta1=0.9, student_df=5.0)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    outlier = torch.ones_like(parameter)
    outlier[1].fill_(1e30)
    parameter.grad = outlier
    optimizer.step()

    state = optimizer.state[parameter]
    assert torch.isfinite(parameter).all()
    for value in state.values():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all()
    assert state["last_student_weight"][1] < 1e-40
    assert state["last_gain"][1] < state["last_gain"][0]
    assert torch.all(state["posterior_var"] > 0)
    assert torch.all(state["observation_var"] > 0)


def test_scalar_diagnostics_and_closure_contract(device):
    parameter = torch.nn.Parameter(torch.tensor(0.0, device=device))
    optimizer = StreamingPosteriorFilter([parameter], diagnostic_interval=1)
    parameter.grad = torch.tensor(1.0, device=device)
    expected_loss = torch.tensor(7.0, device=device)

    returned_loss = optimizer.step(lambda: expected_loss)
    diagnostics = optimizer.diagnostics()

    assert returned_loss is expected_loss
    assert diagnostics["spf/gain_std"] == 0.0
    assert all(torch.isfinite(torch.tensor(value)) for value in diagnostics.values())


def test_state_dict_resume_is_exact(device):
    parameter = torch.nn.Parameter(torch.zeros(2, 4, device=device))
    optimizer = StreamingPosteriorFilter([parameter], diagnostic_interval=1)
    for value in (1.0, -0.5, 0.25):
        parameter.grad = torch.full_like(parameter, value)
        optimizer.step()

    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed_optimizer = StreamingPosteriorFilter([resumed_parameter], diagnostic_interval=1)
    resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))

    parameter.grad = torch.full_like(parameter, 0.75)
    resumed_parameter.grad = torch.full_like(resumed_parameter, 0.75)
    optimizer.step()
    resumed_optimizer.step()

    torch.testing.assert_close(parameter, resumed_parameter, rtol=0, atol=0)
    original_state = optimizer.state[parameter]
    resumed_state = resumed_optimizer.state[resumed_parameter]
    assert original_state.keys() == resumed_state.keys()
    for key in original_state:
        if torch.is_tensor(original_state[key]):
            torch.testing.assert_close(original_state[key], resumed_state[key], rtol=0, atol=0)
        else:
            assert original_state[key] == resumed_state[key]
