"""Statistical and CUDA simulator contracts; GPU tests MUST run through mlq."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from types import SimpleNamespace

PATH = Path(__file__).resolve().parents[1] / "scripts/reppo_latentroll_oracle.py"
spec = importlib.util.spec_from_file_location("latentroll_oracle_test", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pair_freezes_temperature_but_retains_both_judges():
    snapshot = {"actor_net": "new", "critic": "new_critic", "temperature": 0.2,
                "actor_net_before_update": "old", "critic_before_update": "old_critic",
                "temperature_before_update": 0.1}
    old, new = module.make_pair(snapshot, "update")
    assert old["actor_net"] == "old" and new["actor_net"] == "new"
    assert old["critic"] == "old_critic" and new["critic"] == "new_critic"
    assert old["temperature"] == new["temperature"] == 0.1
    assert snapshot["temperature"] == 0.2


def test_cluster_interval_does_not_treat_rollout_replicates_as_states():
    # Increasing identical within-state samples supplies no extra information
    # about the state distribution, and must not narrow its bootstrap interval.
    delta = np.arange(20, dtype=float)[:, None].repeat(2, axis=1)
    a = module.paired_statistics(delta)
    b = module.paired_statistics(delta.repeat(16, axis=1))
    assert a["ci95_low"] == b["ci95_low"]
    assert a["ci95_high"] == b["ci95_high"]
    assert a["within_state_mc_se_rms"] == b["within_state_mc_se_rms"] == 0


def test_identity_policy_has_exact_zero_paired_return_and_interval():
    result = module.paired_statistics(np.zeros((64, 8)), np.zeros((64, 8)))
    assert result["mean"] == result["ci95_low"] == result["ci95_high"] == 0
    assert result["prediction_error_mean"] == 0
    assert result["prediction_correlation"] is None



def test_prediction_error_intervals_pair_predictions_with_their_states():
    # The signal varies greatly across states, but the actual prediction error
    # is constant. Subtracting one fixed prediction mean would invent variance.
    prediction = np.arange(24, dtype=float)[:, None].repeat(4, axis=1)
    delta = prediction - 2.5
    clusters = np.repeat(np.arange(6), 4)
    result = module.paired_statistics(delta, prediction, clusters)
    for key in ("prediction_error_mean", "prediction_error_ci95_low", "prediction_error_ci95_high",
                "prediction_error_trajectory_ci95_low", "prediction_error_trajectory_ci95_high"):
        assert result[key] == -2.5
    assert result["ci95_high"] > result["ci95_low"]


def test_trainer_architecture_inference_does_not_construct_networks():
    for key, filename in (("dynamics.0.weight", "latentroll_v27"), ("predictor.0.weight", "refresh_v25")):
        trainer, path = module.load_trainer({"critic": {key: None}})
        assert filename in path
        assert callable(trainer.lambda_returns)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; queue through mlq")
def test_identical_compiled_policies_have_exactly_zero_simulator_deltas():
    # Exercise actual restored MuJoCo branches and compiled CUDA networks, not a
    # fabricated return array. Unequal episode ages verify no accidental reset.
    trainer, _ = module.load_trainer({"critic": {"dynamics.0.weight": None}})
    trainer.configure_runtime(cudnn_deterministic=True, matmul_precision="highest", allow_tf32=False)
    torch.manual_seed(1)
    args = trainer.Args(actor_hidden_dim=32, critic_hidden_dim=32, num_bins=31, num_envs=2)
    physics = module.legacy.base.Physics(2, 1)
    obs_dim = physics.nq - 1 + physics.nv
    actor = trainer.Actor(obs_dim, physics.nu, args).cuda()
    critic = trainer.Critic(obs_dim, physics.nu, args, "cuda").cuda()
    snapshot = {"args": vars(args), "actor_net": actor.net.state_dict(), "critic": critic.state_dict(),
                "temperature": 0.1, "obs_means": np.zeros((2, obs_dim)),
                "obs_variances": np.ones((2, obs_dim)), "obs_epsilon": 1e-8}
    cli = SimpleNamespace(value_samples=4, seed=1, rollouts=4, policy_horizon=20,
                          policy_checkpoints=[1, 10, 20])
    before = module.CompiledOracle(trainer, snapshot, physics, cli, torch.device("cuda"))
    after = module.CompiledOracle(trainer, snapshot, physics, cli, torch.device("cuda"))
    data = {"state": physics.reset(2), "row": np.array([0, 1]), "age": np.array([0, 995])}
    returns, predictions = module.paired_returns(before, after, cli, data, np.arange(2))
    for values in returns.values():
        for per_branch in values.values():
            assert np.isfinite(per_branch).all()
            np.testing.assert_array_equal(per_branch[:, 0], per_branch[:, 1])
            np.testing.assert_array_equal(per_branch[:, 0], per_branch[:, 2])
    for name, value in predictions.items():
        np.testing.assert_allclose(value, 0, atol=1e-7, err_msg=name)
