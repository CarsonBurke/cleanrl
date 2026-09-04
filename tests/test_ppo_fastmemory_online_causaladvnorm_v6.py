import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "streaming-posterior-filter"
    / "ppo_continuous_action_spf_online_causaladvnorm_v6.py"
)
spec = importlib.util.spec_from_file_location(
    "ppo_continuous_action_spf_online_causaladvnorm_v6",
    SCRIPT,
)
assert spec is not None
module = importlib.util.module_from_spec(spec)
loader = spec.loader
assert loader is not None
loader.exec_module(module)
Args = module.Args
CausalAdvantageNormalizer = module.CausalAdvantageNormalizer
validate_online_contract = module.validate_online_contract


@pytest.fixture(scope="module")
def device():
    assert torch.cuda.is_available(), "causal advantage tests require CUDA"
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


def test_current_advantages_use_only_prior_statistics(device):
    args = Args(
        causal_norm_initial_scale=2.0,
        causal_norm_location_rate=0.1,
        causal_norm_scale_rate=0.1,
    )
    normalizer = CausalAdvantageNormalizer(args, device)
    td_error = torch.tensor((1.0, 3.0), device=device)

    actual = normalizer.transform(td_error)

    torch.testing.assert_close(actual, td_error / 2.0)
    assert normalizer.location > 0


def test_current_batch_does_not_center_itself(device):
    normalizer = CausalAdvantageNormalizer(Args(), device)

    actual = normalizer.transform(torch.full((64,), 10.0, device=device))

    torch.testing.assert_close(actual, torch.full_like(actual, 10.0))
    assert normalizer.location > 0


def test_shared_lagged_transform_preserves_relative_magnitudes(device):
    args = Args(causal_norm_initial_scale=2.0)
    normalizer = CausalAdvantageNormalizer(args, device)
    td_error = torch.tensor((-3.0, 1.0, 5.0), device=device)

    actual = normalizer.transform(td_error)

    expected_differences = (td_error[1:] - td_error[:-1]) / 2.0
    torch.testing.assert_close(actual[1:] - actual[:-1], expected_differences)


def test_statistic_update_is_permutation_invariant(device):
    args = Args(
        causal_norm_location_rate=0.1,
        causal_norm_scale_rate=0.1,
    )
    first = CausalAdvantageNormalizer(args, device)
    second = CausalAdvantageNormalizer(args, device)
    td_error = torch.tensor((-4.0, -1.0, 2.0, 8.0), device=device)
    permutation = torch.tensor((2, 0, 3, 1), device=device)

    first.transform(td_error)
    second.transform(td_error[permutation])

    torch.testing.assert_close(first.location, second.location)
    torch.testing.assert_close(first.log_scale, second.log_scale)
