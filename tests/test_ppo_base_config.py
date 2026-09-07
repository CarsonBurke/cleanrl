"""CPU-only execution configuration checks; no model or simulator construction."""

import pytest

from cleanrl import ppo_continuous_action as ppo


@pytest.mark.parametrize("kwargs", [
    {"num_envs": 0}, {"num_steps": 0}, {"num_minibatches": 0},
    {"update_epochs": 0}, {"env_threads": 0}, {"env_backend": "unknown"},
    {"num_steps": 8, "num_minibatches": 32},
    {"num_steps": 32, "num_minibatches": 32}, {"cuda": False},
])
def test_invalid_configuration_fails_before_cuda_or_physics(kwargs):
    with pytest.raises(ValueError):
        ppo.validate_args(ppo.Args(**kwargs))
