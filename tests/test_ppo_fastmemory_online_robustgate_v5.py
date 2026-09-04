import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "streaming-posterior-filter"
    / "ppo_continuous_action_spf_online_robustgate_v5.py"
)
spec = importlib.util.spec_from_file_location(
    "ppo_continuous_action_spf_online_robustgate_v5",
    SCRIPT,
)
assert spec is not None
module = importlib.util.module_from_spec(spec)
loader = spec.loader
assert loader is not None
loader.exec_module(module)
Args = module.Args
CausalStudentAdvantage = module.CausalStudentAdvantage
validate_online_contract = module.validate_online_contract


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "robust online PPO tests require CUDA"
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
        ("ent_alpha", 0.1),
        ("ent_coef", 0.01),
        ("auto_entropy", True),
        ("target_kl", 0.03),
    ),
)
def test_contract_rejects_nonstreaming_or_confounded_settings(name, value):
    args = Args()
    setattr(args, name, value)

    with pytest.raises(ValueError, match=name):
        validate_online_contract(args)


def test_transform_uses_preupdate_scale_and_preserves_units(device):
    args = Args(
        num_envs=2,
        robust_adv_scale_rate=0.1,
        robust_adv_initial_scale=2.0,
    )
    transform = CausalStudentAdvantage(2, args, device)
    td_error = torch.tensor((1.0, 100.0), device=device)
    preupdate_scale = torch.full_like(td_error, 2.0)
    z = td_error / preupdate_scale
    expected = td_error * (
        args.robust_adv_student_df
        / (args.robust_adv_student_df + z.square())
    ).sqrt()

    actual = transform.transform(td_error)

    torch.testing.assert_close(actual, expected)
    assert actual[0] > 0.9 * td_error[0]
    assert actual[1] <= preupdate_scale[1] * args.robust_adv_student_df**0.5
    assert transform.log_scale[1].exp() > preupdate_scale[1]


def test_stream_scales_do_not_couple(device):
    args = Args(num_envs=2, robust_adv_scale_rate=0.1)
    first = CausalStudentAdvantage(2, args, device)
    second = CausalStudentAdvantage(2, args, device)

    first_output = first.transform(torch.tensor((2.0, 10.0), device=device))
    second_output = second.transform(torch.tensor((2.0, 1_000.0), device=device))

    torch.testing.assert_close(first_output[0], second_output[0], rtol=0, atol=0)
    torch.testing.assert_close(first.log_scale[0], second.log_scale[0], rtol=0, atol=0)
    assert not torch.equal(first.log_scale[1], second.log_scale[1])


def test_next_transition_reads_updated_scale(device):
    args = Args(num_envs=1, robust_adv_scale_rate=0.1)
    transform = CausalStudentAdvantage(1, args, device)
    td_error = torch.tensor((10.0,), device=device)

    first = transform.transform(td_error)
    second = transform.transform(td_error)

    assert second > first


def test_scale_respects_configured_clamps(device):
    args = Args(
        num_envs=1,
        robust_adv_scale_rate=1.0,
        robust_adv_min_scale=0.5,
        robust_adv_max_scale=2.0,
    )
    transform = CausalStudentAdvantage(1, args, device)

    transform.transform(torch.tensor((1_000.0,), device=device))
    torch.testing.assert_close(
        transform.log_scale.exp(),
        torch.tensor((2.0,), device=device),
    )
    transform.transform(torch.zeros(1, device=device))
    transform.transform(torch.zeros(1, device=device))
    torch.testing.assert_close(
        transform.log_scale.exp(),
        torch.tensor((0.5,), device=device),
    )
