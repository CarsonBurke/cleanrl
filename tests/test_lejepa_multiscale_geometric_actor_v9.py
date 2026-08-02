import importlib.util
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "encoder-value"
    / "ppo_continuous_action_lejepa_multiscale_geometric_actor_v9.py"
)
SPEC = importlib.util.spec_from_file_location("lejepa_multiscale_geometric_actor_v9", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_geometric_beta_grid_has_exact_final_gamma():
    args = MODULE.Args()
    assert args.advantage_horizon == 16
    assert args.ssl_control_mode == "line_search"
    assert args.gamma == MODULE.GEOMETRIC_GAMMA
    assert MODULE.GEOMETRIC_BETAS[-1] is MODULE.GEOMETRIC_GAMMA
    assert MODULE.GEOMETRIC_EFFECTIVE_HORIZONS[:-1] == (
        1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0
    )
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


def test_multiscale_targets_equal_independent_direct_targets():
    torch.manual_seed(2)
    phi = torch.randn(12, 3, 2)
    next_means = torch.randn(
        12, 3, len(MODULE.GEOMETRIC_BETAS), 2
    )
    bootstrap = (torch.rand(12, 3) > 0.2).float()
    continuation = (torch.rand(12, 3) > 0.2).float()
    actual = MODULE.build_multiscale_full_suffix_targets(
        phi,
        next_means,
        bootstrap,
        continuation,
        MODULE.GEOMETRIC_BETAS,
    )
    expected = torch.stack(
        [
            MODULE.build_full_suffix_geometric_targets(
                phi,
                next_means[:, :, index],
                bootstrap,
                continuation,
                beta,
            )
            for index, beta in enumerate(MODULE.GEOMETRIC_BETAS)
        ],
        dim=2,
    )
    torch.testing.assert_close(actual, expected)


def test_every_multiscale_constant_feature_fixed_point_is_one():
    phi = torch.ones(128, 3, 2)
    next_means = torch.ones(
        128, 3, len(MODULE.GEOMETRIC_BETAS), 2
    )
    targets = MODULE.build_multiscale_full_suffix_targets(
        phi,
        next_means,
        torch.ones(128, 3),
        torch.ones(128, 3),
        MODULE.GEOMETRIC_BETAS,
    )
    torch.testing.assert_close(targets, torch.ones_like(targets))
    torch.testing.assert_close(targets[:, :, 0], phi)


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


def test_zero_successor_mean_gives_exact_fixed16_reward_prefix():
    torch.manual_seed(5)
    steps, envs, base_dim = 24, 3, 5
    gamma = MODULE.GEOMETRIC_GAMMA
    reward = torch.randn(steps, envs)
    base_phi = torch.randn(steps, envs, base_dim)
    covector = torch.randn(base_dim)
    phi = MODULE.augment_reward_residual(base_phi, reward, covector)
    task = torch.cat([covector, torch.ones(1)])
    delta = (1.0 - gamma) * phi
    actual = (
        MODULE.fixed_horizon_sum(
            delta, torch.ones(steps, envs), gamma, horizon=16
        )
        @ task
    ) / (1.0 - gamma)
    expected = MODULE.fixed_horizon_sum(
        reward, torch.ones(steps, envs), gamma, horizon=16
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_reward_gauge_transport_preserves_all_heads_and_resets_all_rows():
    torch.manual_seed(7)
    num_heads, sf_dim = 4, 4
    layer = torch.nn.Linear(6, num_heads * sf_dim)
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
    layer(torch.randn(8, 6)).square().mean().backward()
    optimizer.step()

    features = torch.randn(11, 6)
    before = layer(features).detach().view(11, num_heads, sf_dim)
    old_covector = torch.randn(3)
    new_covector = torch.randn(3)
    old_task = torch.cat([old_covector, torch.ones(1)])
    expected = before @ old_task

    MODULE.transport_reward_gauge(
        layer, old_covector, new_covector, sf_dim
    )
    MODULE.reset_optimizer_output_rows(optimizer, layer, sf_dim)
    new_task = torch.cat([new_covector, torch.ones(1)])
    after = layer(features).view(11, num_heads, sf_dim)
    torch.testing.assert_close(after @ new_task, expected)
    residual_rows = torch.arange(sf_dim - 1, num_heads * sf_dim, sf_dim)
    nonresidual_rows = torch.tensor(
        [row for row in range(num_heads * sf_dim) if row not in residual_rows]
    )
    weight_moment = optimizer.state[layer.weight]["exp_avg"]
    bias_moment = optimizer.state[layer.bias]["exp_avg"]
    assert weight_moment[residual_rows].count_nonzero() == 0
    assert bias_moment[residual_rows].count_nonzero() == 0
    assert weight_moment[nonresidual_rows].count_nonzero() > 0


def test_agent_has_direct_multiscale_heads_and_shared_embedding_input():
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
    sf_dim = args.emb_dim + 5 + 2 * 2 + 2
    assert successor.shape == (4, len(MODULE.GEOMETRIC_BETAS), sf_dim)
    assert agent.critic_head.out_features == len(MODULE.GEOMETRIC_BETAS) * sf_dim


def test_parameter_interpolation_and_largest_acceptable_alpha():
    before = {"weight": torch.tensor([1.0, -2.0])}
    proposed = {"weight": torch.tensor([5.0, 6.0])}
    quarter = MODULE.interpolate_parameter_dict(before, proposed, 0.25)
    torch.testing.assert_close(
        quarter["weight"], torch.tensor([2.0, 0.0])
    )
    chosen = MODULE.select_line_search_alpha(
        MODULE.SSL_LINE_SEARCH_ALPHAS,
        lambda alpha: alpha <= 0.25,
    )
    assert chosen == 0.25
    assert MODULE.select_line_search_alpha((1.0, 0.5), lambda _: False) == 0.0


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
    assert source.index("sf_target = build_multiscale_full_suffix_targets") < source.index(
        "for epoch in range(args.update_epochs)"
    )


def test_actor_credit_and_scalar_value_use_only_exact_gamma_head():
    source = SCRIPT.read_text()
    rollout_value = source.index("value_sf[:, -1] @ reward_task")
    actor_next = source.index("successor_next[:, :, -1]")
    actor_current = source.index("successor_cur[:, :, -1]")
    actor_update = source.index("for epoch in range(args.update_epochs)")
    assert rollout_value < actor_next < actor_current < actor_update
    assert "0.5 * sf_mse_by_head[-1]" in source
    assert "sf_drift_by_head.max()" in source


def test_line_search_uses_disjoint_alignment_and_trust_probes_and_restores_atomically():
    source = SCRIPT.read_text()
    assert "trial_online[frame_fit]" in source
    assert "probe_online_before[frame_fit]" in source
    assert ").sum(-1)[trust_eval].mean()" in source
    assert "old_policy[trust_eval]" not in source
    assert "trial_policy[trust_eval]" not in source
    assert "trial_successors[trust_eval]" in source
    assert "old_successors[trust_eval]" in source
    evaluate = source.index("def evaluate_ssl_trial(alpha):")
    reset = source.index("ssl.load_state_dict(ssl_state_before)", evaluate)
    interpolate = source.index("interpolate_parameter_dict(", reset)
    assert evaluate < reset < interpolate
    rejection = source.index("else:", source.index("accepted = accepted_alpha"))
    assert source.index("ssl.load_state_dict(ssl_state_before)", rejection) > rejection
    assert source.index(
        "ssl_optimizer.load_state_dict(ssl_optimizer_state_before)", rejection
    ) > rejection
