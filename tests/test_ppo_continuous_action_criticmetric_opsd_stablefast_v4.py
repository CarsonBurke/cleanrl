import pytest
import torch

from cleanrl.critic import (
    ppo_continuous_action_criticmetric_opsd_stablefast_v4 as stablefast,
)


def test_metric_identity_floor_bounds_condition_and_preserves_ascent():
    torch.manual_seed(7)
    action_dim = 6
    identity_fraction = 0.25
    metric = stablefast.CriticGradientMetric(
        obs_dim=17,
        action_dim=action_dim,
        hidden=32,
        identity_fraction=identity_fraction,
    )
    with torch.no_grad():
        metric.output.bias.copy_(
            torch.linspace(-10.0, 10.0, action_dim * action_dim)
        )

    batch_size = 32
    observations = torch.randn(batch_size, 17)
    mean = torch.randn(batch_size, action_dim)
    logstd = torch.randn(batch_size, action_dim)
    action = torch.randn(batch_size, action_dim, requires_grad=True)
    normalized_action = torch.randn(batch_size, action_dim)
    critic_score = torch.randn(batch_size)
    critic_action_grad = torch.randn(batch_size, action_dim)

    actor_loss, outputs = metric(
        observations,
        mean,
        logstd,
        action,
        normalized_action,
        critic_score,
        critic_action_grad,
    )
    eigenvalues = torch.linalg.eigvalsh(outputs["metric"])
    condition = eigenvalues[:, -1] / eigenvalues[:, 0]
    maximum_condition = (
        identity_fraction + (1.0 - identity_fraction) * action_dim
    ) / identity_fraction

    torch.testing.assert_close(
        outputs["metric"].diagonal(dim1=-2, dim2=-1).sum(dim=-1),
        torch.full((batch_size,), float(action_dim)),
    )
    assert torch.all(eigenvalues[:, 0] >= identity_fraction - 1e-5)
    assert torch.all(condition <= maximum_condition + 1e-3)
    assert torch.all(outputs["ascent_dot"] > 0)

    actor_loss.backward()
    assert action.grad is not None and torch.isfinite(action.grad).all()
    assert all(parameter.grad is not None for parameter in metric.parameters())


def test_replay_retains_latest_transitions_across_wrap_and_oversized_add():
    replay = stablefast.ReplayBuffer(
        capacity=5,
        obs_shape=(1,),
        action_shape=(1,),
        seed=1,
        device=torch.device("cpu"),
    )

    def add_range(start, stop):
        values = torch.arange(start, stop, dtype=torch.float32)
        replay.add(
            values[:, None],
            (-values)[:, None],
            values + 0.5,
            (values + 1.0)[:, None],
            torch.full_like(values, 0.99),
        )

    add_range(0, 3)
    add_range(3, 7)
    torch.testing.assert_close(
        replay.observations.squeeze(-1).sort().values,
        torch.arange(2, 7, dtype=torch.float32),
    )
    assert replay.position == 2
    assert replay.size == 5

    add_range(7, 15)
    torch.testing.assert_close(
        replay.observations.squeeze(-1),
        torch.arange(10, 15, dtype=torch.float32),
    )
    assert replay.position == 0
    assert replay.size == 5

    sample = replay.sample(5)
    assert len(sample) == 5
    assert all(field.shape[0] == 5 for field in sample)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="StableFast CUDA replay requires a CUDA device",
)
def test_cuda_replay_samples_only_retained_data_reproducibly():
    device = torch.device("cuda")
    buffers = [
        stablefast.ReplayBuffer(
            capacity=5,
            obs_shape=(1,),
            action_shape=(1,),
            seed=13,
            device=device,
        )
        for _ in range(2)
    ]
    values = torch.arange(10, 18, dtype=torch.float32, device=device)
    for replay in buffers:
        replay.add(
            values[:, None],
            (-values)[:, None],
            values + 0.5,
            (values + 1.0)[:, None],
            torch.full_like(values, 0.99),
        )

    first_sample = buffers[0].sample(32)
    second_sample = buffers[1].sample(32)
    for first_field, second_field in zip(
        first_sample, second_sample, strict=True
    ):
        torch.testing.assert_close(first_field, second_field, rtol=0, atol=0)

    sampled_observations = first_sample[0].squeeze(-1)
    assert torch.all((sampled_observations >= 13) & (sampled_observations <= 17))
    torch.testing.assert_close(
        first_sample[1].squeeze(-1),
        -sampled_observations,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        first_sample[2],
        sampled_observations + 0.5,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="StableFast critic fit requires CUDA BF16",
)
def test_bf16_fused_fit_preserves_fp32_frozen_scoring_path():
    device = torch.device("cuda")
    critic = stablefast.QCritic(obs_dim=5, action_dim=2).to(device)
    optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=3e-4,
        eps=1e-5,
        fused=True,
    )
    observations = torch.randn(32, 5, device=device)
    actions = torch.randn(32, 2, device=device)
    targets = torch.randn(32, device=device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        low_precision_values = critic(observations, actions)
    assert low_precision_values.dtype == torch.bfloat16
    values = low_precision_values.float()
    loss = 0.5 * (values - targets).square().mean()
    assert loss.dtype == torch.float32
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    critic.requires_grad_(False)
    scoring_actions = actions.detach().clone().requires_grad_(True)
    scores = critic(observations, scoring_actions)
    assert scores.dtype == torch.float32
    (action_gradient,) = torch.autograd.grad(scores.sum(), scoring_actions)
    assert torch.isfinite(action_gradient).all()
    assert all(parameter.grad is None for parameter in critic.parameters())

    critic.requires_grad_(True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        next_values = critic(observations, actions).float()
    (0.5 * (next_values - targets).square().mean()).backward()
    assert all(parameter.grad is not None for parameter in critic.parameters())


def test_async_vector_env_matches_sync_rollout_and_bootstrap_observations():
    env_fns = [
        stablefast.make_env(
            "Pendulum-v1",
            index,
            capture_video=False,
            run_name="stablefast-vector-contract",
            gamma=0.99,
        )
        for index in range(2)
    ]
    sync_envs = stablefast.gym.vector.SyncVectorEnv(env_fns)
    async_envs = stablefast.gym.vector.AsyncVectorEnv(
        env_fns,
        shared_memory=True,
        context="spawn",
    )
    try:
        sync_obs, _ = sync_envs.reset(seed=23)
        async_obs, _ = async_envs.reset(seed=23)
        torch.testing.assert_close(
            torch.as_tensor(async_obs),
            torch.as_tensor(sync_obs),
            rtol=0,
            atol=0,
        )
        actions = torch.zeros((2, 1)).numpy()
        saw_truncation = False
        for _ in range(200):
            sync_step = sync_envs.step(actions)
            async_step = async_envs.step(actions)
            for sync_value, async_value in zip(
                sync_step[:4], async_step[:4], strict=True
            ):
                torch.testing.assert_close(
                    torch.as_tensor(async_value),
                    torch.as_tensor(sync_value),
                    rtol=1e-6,
                    atol=1e-6,
                )
            sync_bootstrap = stablefast.bootstrap_observations(
                sync_step[0], sync_step[3], sync_step[4]
            )
            async_bootstrap = stablefast.bootstrap_observations(
                async_step[0], async_step[3], async_step[4]
            )
            torch.testing.assert_close(
                torch.as_tensor(async_bootstrap),
                torch.as_tensor(sync_bootstrap),
                rtol=1e-6,
                atol=1e-6,
            )
            saw_truncation = saw_truncation or bool(sync_step[3].any())
        assert saw_truncation
    finally:
        sync_envs.close()
        async_envs.close()
