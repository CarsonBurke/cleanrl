import importlib.util
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_lejepa_geometric_full_actor_v8.py"
)
SPEC = importlib.util.spec_from_file_location("lejepa_geometric_full_actor_v8", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_full_credit_default_and_fixed_control_retain_long_model_horizon():
    args = MODULE.Args()
    assert args.credit_mode == "full"
    assert args.credit_horizon == 256
    assert args.gamma == MODULE.GEOMETRIC_GAMMA
    assert abs((1.0 - args.gamma**1000) - 0.95) < 1e-12
    assert 334.0 < 1.0 / (1.0 - args.gamma) < 335.0


def test_one_step_normalized_geometric_target():
    phi = torch.tensor([[[2.0, -1.0]], [[3.0, 4.0]]])
    next_mean = torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]])
    bootstrap = torch.tensor([[1.0], [0.0]])
    gamma = 0.75
    actual = MODULE.build_one_step_geometric_target(
        phi, next_mean, bootstrap, gamma
    )
    expected = (1.0 - gamma) * phi + gamma * bootstrap.unsqueeze(-1) * next_mean
    torch.testing.assert_close(actual, expected)


def test_full_suffix_uses_samples_then_bootstraps_only_at_data_edges():
    phi = torch.ones(4, 1, 1)
    next_mean = torch.full((4, 1, 1), 5.0)
    continuation = torch.tensor([[1.0], [0.0], [1.0], [1.0]])
    gamma = 0.5

    # t=1 is a truncation, so it uses M(final_observation)=5. The rollout tail
    # at t=3 also bootstraps. Sampled recursion is used at t=0 and t=2.
    truncation = MODULE.build_full_suffix_geometric_targets(
        phi,
        next_mean,
        torch.ones(4, 1),
        continuation,
        gamma,
    )
    expected = torch.tensor([2.0, 3.0, 2.0, 3.0]).view(4, 1, 1)
    torch.testing.assert_close(truncation, expected)

    # A true termination has a zero tail, and t=0 consumes that terminal sample.
    bootstrap = torch.ones(4, 1)
    bootstrap[1] = 0.0
    terminated = MODULE.build_full_suffix_geometric_targets(
        phi, next_mean, bootstrap, continuation, gamma
    )
    torch.testing.assert_close(
        terminated[:2], torch.tensor([0.75, 0.5]).view(2, 1, 1)
    )


def test_constant_feature_has_normalized_fixed_point_one():
    phi = torch.ones(128, 3, 2)
    targets = MODULE.build_full_suffix_geometric_targets(
        phi,
        torch.ones_like(phi),
        torch.ones(128, 3),
        torch.ones(128, 3),
        MODULE.GEOMETRIC_GAMMA,
    )
    torch.testing.assert_close(targets, torch.ones_like(targets))


def test_fixed_horizon_vector_innovations_telescope_to_reward_td():
    torch.manual_seed(3)
    steps, envs, base_dim = 10, 2, 4
    gamma, horizon = 0.91, 4
    base_phi = torch.randn(steps, envs, base_dim)
    reward = torch.randn(steps, envs)
    covector = torch.randn(base_dim)
    phi = MODULE.augment_reward_residual(base_phi, reward, covector)
    task = torch.cat([covector, torch.ones(1)])
    torch.testing.assert_close(phi @ task, reward)

    means = torch.randn(steps + 1, envs, base_dim + 1)
    delta = (
        (1.0 - gamma) * phi
        + gamma * means[1:]
        - means[:-1]
    )
    vector_credit = MODULE.fixed_horizon_sum(
        delta, torch.ones(steps, envs), gamma, horizon
    )
    actual = (vector_credit @ task) / (1.0 - gamma)
    for start in range(steps):
        length = min(horizon, steps - start)
        prefix = sum(
            gamma**offset * reward[start + offset]
            for offset in range(length)
        )
        value_start = (means[start] @ task) / (1.0 - gamma)
        value_end = (means[start + length] @ task) / (1.0 - gamma)
        expected = prefix + gamma**length * value_end - value_start
        torch.testing.assert_close(actual[start], expected, rtol=2e-5, atol=2e-5)


def test_zero_successor_mean_gives_exact_fixed256_reward_prefix():
    torch.manual_seed(5)
    steps, envs, base_dim = 300, 3, 5
    gamma = MODULE.GEOMETRIC_GAMMA
    reward = torch.randn(steps, envs)
    base_phi = torch.randn(steps, envs, base_dim)
    covector = torch.randn(base_dim)
    phi = MODULE.augment_reward_residual(base_phi, reward, covector)
    task = torch.cat([covector, torch.ones(1)])
    delta = (1.0 - gamma) * phi
    actual = (
        MODULE.fixed_horizon_sum(
            delta, torch.ones(steps, envs), gamma, horizon=256
        )
        @ task
    ) / (1.0 - gamma)
    expected = MODULE.fixed_horizon_sum(
        reward, torch.ones(steps, envs), gamma, horizon=256
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_fixed_control_scalar_diagnostic_equals_vector_then_contraction():
    torch.manual_seed(17)
    delta = torch.randn(20, 3, 7)
    task = torch.randn(7)
    continuation = (torch.rand(20, 3) > 0.15).float()
    gamma = 0.93
    vector_then_contract = (
        MODULE.fixed_horizon_sum(delta, continuation, gamma, horizon=9) @ task
    ) / (1.0 - gamma)
    contract_then_scalar = MODULE.fixed_horizon_sum(
        (delta @ task) / (1.0 - gamma),
        continuation,
        gamma,
        horizon=9,
    )
    torch.testing.assert_close(vector_then_contract, contract_then_scalar)


def test_full_successor_residual_is_return_minus_complete_current_baseline():
    gamma = 0.8
    reward = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    phi = reward.unsqueeze(-1)
    edge_successor = torch.tensor([[[9.0]], [[9.0]], [[9.0]], [[5.0]]])
    successor_cur = torch.tensor([[[2.0]], [[3.0]], [[4.0]], [[6.0]]])
    continuation = torch.ones(4, 1)
    target = MODULE.build_full_suffix_geometric_targets(
        phi,
        edge_successor,
        torch.ones(4, 1),
        continuation,
        gamma,
    )
    actual = (target - successor_cur).squeeze(-1) / (1.0 - gamma)
    for start in range(4):
        sampled = sum(
            gamma**offset * reward[start + offset]
            for offset in range(4 - start)
        )
        edge_value = edge_successor[-1, 0, 0] / (1.0 - gamma)
        current_value = successor_cur[start, 0, 0] / (1.0 - gamma)
        expected = (
            sampled
            + gamma ** (4 - start) * edge_value
            - current_value
        )
        torch.testing.assert_close(actual[start], expected)


def test_sampled_suffix_horizon_stops_at_boundaries_and_rollout_edge():
    continuation = torch.tensor(
        [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    actual = MODULE.same_episode_suffix_steps(continuation)
    expected = torch.tensor(
        [[3.0, 2.0], [2.0, 1.0], [1.0, 2.0], [1.0, 1.0]]
    )
    torch.testing.assert_close(actual, expected)


def test_reward_gauge_transport_preserves_values_and_resets_row_moments():
    torch.manual_seed(7)
    layer = torch.nn.Linear(6, 4)
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
    layer(torch.randn(8, 6)).square().mean().backward()
    optimizer.step()

    features = torch.randn(11, 6)
    before = layer(features).detach()
    old_covector = torch.randn(3)
    new_covector = torch.randn(3)
    old_task = torch.cat([old_covector, torch.ones(1)])
    expected = before @ old_task

    MODULE.transport_reward_gauge(layer, old_covector, new_covector)
    MODULE.reset_optimizer_output_row(optimizer, layer)
    new_task = torch.cat([new_covector, torch.ones(1)])
    torch.testing.assert_close(layer(features) @ new_task, expected)
    assert not torch.equal(
        optimizer.state[layer.weight]["exp_avg"][:-1],
        torch.zeros_like(optimizer.state[layer.weight]["exp_avg"][:-1]),
    )
    assert optimizer.state[layer.weight]["exp_avg"][-1].count_nonzero() == 0
    assert optimizer.state[layer.bias]["exp_avg"][-1] == 0


def test_agent_has_one_vector_critic_head_and_shared_embedding_input():
    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-1.0, 1.0, shape=(5,)),
        single_action_space=gym.spaces.Box(-1.0, 1.0, shape=(2,)),
    )
    args = SimpleNamespace(
        emb_dim=8,
        hidden=16,
        k_blocks=1,
        n_experts=2,
        share_backbone=True,
        actor_dist="beta",
    )
    agent = MODULE.Agent(envs, args)
    action, latent, log_prob, entropy, successor = agent.get_action_and_value(
        torch.randn(4, 5 + args.emb_dim)
    )
    assert action.shape == latent.shape == (4, 2)
    assert log_prob.shape == entropy.shape == (4,)
    assert successor.shape == (4, args.emb_dim + 5 + 2 * 2 + 2)
    assert agent.critic_head.out_features == successor.shape[-1]


def test_geometric_lejepa_predicts_all_distances_with_attached_targets():
    args = SimpleNamespace(
        emb_dim=8,
        ssl_hidden=16,
        seq_len=max(MODULE.LEJEPA_HORIZONS) + 1,
        pred_depth=1,
        pred_heads=2,
        pred_mlp_dim=16,
        pred_dim_head=4,
        sigreg_num_proj=8,
        sigreg_proj_chunk=4,
    )
    model = MODULE.LeJepaSSL(obs_dim=5, act_dim=2, args=args)
    obs = torch.randn(2, args.seq_len, 5)
    actions = torch.randn(2, args.seq_len, 2)
    continuation = torch.ones(2, args.seq_len)
    loss, _, _, horizon_losses = model(
        obs, actions, continuation, sigreg_weight=0.01
    )
    assert horizon_losses.shape == (len(MODULE.LEJEPA_HORIZONS),)
    loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in model.encoder.parameters()
    )
    for horizon in MODULE.LEJEPA_HORIZONS:
        assert all(
            parameter.grad is not None
            for parameter in model.pred_projs[str(horizon)].parameters()
        )


def test_obsolete_credit_and_target_mechanisms_are_absent():
    source = SCRIPT.read_text().lower()
    for token in (
        "gae_lambda",
        "per_dim_lambda",
        "critic_mtp_horizon",
        "td_shell",
        "popart",
        "contrastive",
        "target_ema",
        "sf_target_ema",
    ):
        assert token not in source
    assert "copy.deepcopy(ssl.encoder)" not in source
    assert "old_parameter.lerp" not in source
    assert source.index("sf_target = build_full_suffix_geometric_targets") < source.index(
        "for epoch in range(args.update_epochs)"
    )
    assert "rho_used" not in source
    assert "tail_coefficient" not in source


def test_credit_mode_selects_complete_full_residual_or_fixed_trace():
    source = SCRIPT.read_text()
    full_residual = source.index(
        "full_vector_advantage = sf_target - successor_cur"
    )
    selector = source.index('if args.credit_mode == "fixed"', full_residual)
    fixed_trace = source.index("vector_advantage = fixed_horizon_sum(", selector)
    contraction = source.index("vector_advantage @ reward_task", fixed_trace)
    actor_update = source.index("for epoch in range(args.update_epochs)")
    assert full_residual < selector < fixed_trace < contraction < actor_update
