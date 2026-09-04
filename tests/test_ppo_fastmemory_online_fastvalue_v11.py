import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "streaming-posterior-filter"
    / "ppo_continuous_action_spf_online_fastvalue_v11.py"
)
spec = importlib.util.spec_from_file_location(
    "ppo_continuous_action_spf_online_fastvalue_v11",
    SCRIPT,
)
assert spec is not None
module = importlib.util.module_from_spec(spec)
loader = spec.loader
assert loader is not None
loader.exec_module(module)
Args = module.Args
FastValueResidual = module.FastValueResidual
validate_online_contract = module.validate_online_contract


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "fast-value tests require CUDA"
    return torch.device("cuda")


def test_default_configuration_is_single_use_v_critic():
    validate_online_contract(Args())


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("num_steps", 2),
        ("update_epochs", 2),
        ("num_minibatches", 2),
        ("norm_adv", True),
        ("ret_percnorm", True),
        ("adv_transform", "rankgauss"),
        ("critic_mtp_horizon", 2),
        ("target_kl", 0.03),
        ("ent_alpha", 0.1),
        ("ent_coef", 0.01),
        ("auto_entropy", True),
    ),
)
def test_contract_rejects_nonstreaming_or_confounded_settings(name, value):
    args = Args()
    setattr(args, name, value)

    with pytest.raises(ValueError, match=name):
        validate_online_contract(args)


def test_fast_value_reads_before_current_td_update(device):
    args = Args(num_envs=2, fast_value_leak=1.0)
    fast_value = FastValueResidual(2, 3, args, device)
    obs = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), device=device)

    before = fast_value.read(obs)
    fast_value.update(obs, torch.tensor((1.0, 0.0), device=device))
    after = fast_value.read(obs)

    torch.testing.assert_close(before, torch.zeros_like(before), rtol=0, atol=0)
    assert after[0] > 0
    torch.testing.assert_close(after[1], torch.zeros_like(after[1]), rtol=0, atol=0)


def test_update_is_robust_to_extreme_td_error(device):
    args = Args(num_envs=1, fast_value_leak=1.0)
    fast_value = FastValueResidual(1, 2, args, device)
    obs = torch.tensor(((1.0, 0.0),), device=device)
    td_error = torch.tensor((1_000.0,), device=device)
    expected_weight = (
        (args.fast_value_student_df + 1.0)
        / (args.fast_value_student_df + td_error.square())
    ).clamp_max(1.0)

    fast_value.update(obs, td_error)

    expected = args.fast_value_eta * expected_weight * td_error
    torch.testing.assert_close(fast_value.weight[0, 0], expected[0])
    assert fast_value.weight.abs().max() < 0.001


def test_actor_correction_uses_both_preupdate_endpoint_reads(device):
    args = Args(num_envs=1, gamma=0.99, fast_value_leak=1.0)
    fast_value = FastValueResidual(1, 2, args, device)
    fast_value.weight.copy_(torch.tensor(((2.0, -1.0),), device=device))
    obs = torch.tensor(((1.0, 0.0),), device=device)
    next_obs = torch.tensor(((0.0, 1.0),), device=device)
    raw_delta = torch.tensor((0.5,), device=device)
    current = fast_value.read(obs)
    successor = fast_value.read(next_obs)
    expected = raw_delta + args.gamma * successor - current

    fast_value.update(obs, expected)

    torch.testing.assert_close(expected, torch.tensor((-2.49,), device=device))


def test_normalized_state_key_bounds_update_geometry(device):
    args = Args(num_envs=1, fast_value_leak=1.0)
    first = FastValueResidual(1, 2, args, device)
    second = FastValueResidual(1, 2, args, device)

    first.update(torch.tensor(((1.0, 0.0),), device=device), torch.ones(1, device=device))
    second.update(torch.tensor(((10.0, 0.0),), device=device), torch.ones(1, device=device))

    torch.testing.assert_close(first.weight, second.weight)
    torch.testing.assert_close(F.normalize(first.weight), F.normalize(second.weight))
