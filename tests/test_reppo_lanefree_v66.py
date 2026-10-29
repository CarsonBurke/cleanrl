"""Contracts of v66's two arms (--obs-norm shared, --kl-mode penalty); CPU only, no environments."""
import numpy as np
import pytest
import torch

from cleanrl.reppo.ppo_continuous_action_reppo_lanefree_v66 import (
    KL_POLICY_LOSSES, Args, SharedObsNorm, frozen_normalize, gated_policy_loss, gaussian_kl, penalized_policy_loss,
    pooled_moments, validate_args,
)
from cleanrl.shared.vector_norm import VectorObsNorm

LANES, OBSERVATION_DIM, CLIP = 5, 17, 10.0


def lane_samples(counts=(3, 40, 7, 1, 250), seed=0):
    """Lanes of unequal length whose means and scales differ (a tumbling lane next to steady ones)."""
    rng = np.random.default_rng(seed)
    return [rng.normal(loc=rng.normal(size=OBSERVATION_DIM), scale=rng.uniform(0.2, 3.0, OBSERVATION_DIM),
                       size=(count, OBSERVATION_DIM)) for count in counts]


def fed_accumulator(steps=60, seed=0, lanes=LANES):
    """A VectorObsNorm after whole-batch steps plus a few single-lane ones (resets): unequal counts."""
    rng = np.random.default_rng(seed)
    accumulator = VectorObsNorm(lanes, (OBSERVATION_DIM,), clip=CLIP)
    offsets, scales = rng.normal(size=(lanes, OBSERVATION_DIM)), rng.uniform(0.3, 2.0, (lanes, OBSERVATION_DIM))
    seen = [[] for _ in range(lanes)]
    for step in range(steps):
        batch = offsets + scales * rng.normal(size=(lanes, OBSERVATION_DIM))
        accumulator.normalize(batch)
        for lane in range(lanes):
            seen[lane].append(batch[lane])
        if step % 7 == 0:
            lane = step % lanes
            extra = offsets[lane] + scales[lane] * rng.normal(size=(1, OBSERVATION_DIM))
            accumulator.normalize(extra, rows=slice(lane, lane + 1))
            seen[lane].append(extra[0])
    return accumulator, [np.stack(rows) for rows in seen], rng


def update_normalize(raw, rows, obs_mean, obs_inv_std, num_envs, clip=CLIP):
    """The trainer's update-side `normalize` closure, verbatim (it lives inside main)."""
    env = rows % num_envs
    return ((raw - obs_mean[env]) * obs_inv_std[env]).clamp(-clip, clip)


def test_pooled_moments_equal_the_concatenated_datas():
    samples = lane_samples()
    means = np.stack([lane.mean(0) for lane in samples])
    variances = np.stack([lane.var(0) for lane in samples])  # population variance, as RunningMeanStd keeps
    counts = np.array([len(lane) for lane in samples], dtype=np.float64)
    mean, variance, total = pooled_moments(means, variances, counts)
    union = np.concatenate(samples)
    assert total == len(union)
    np.testing.assert_allclose(mean, union.mean(0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(variance, union.var(0), rtol=1e-12, atol=1e-12)
    # The between-lane term is not optional: the count-weighted mean of the lanes' variances misses it.
    within = (counts[:, None] * variances).sum(0) / counts.sum()
    assert (union.var(0) - within).min() > 1e-3


def test_pooled_moments_of_equal_counts_and_one_lane():
    samples = lane_samples(counts=(32, 32, 32))
    means, variances = np.stack([lane.mean(0) for lane in samples]), np.stack([lane.var(0) for lane in samples])
    mean, variance, _ = pooled_moments(means, variances, np.full(3, 32.0))
    union = np.concatenate(samples)
    np.testing.assert_allclose(mean, union.mean(0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(variance, union.var(0), rtol=1e-12, atol=1e-12)
    mean, variance, total = pooled_moments(means[:1], variances[:1], np.array([32.0]))
    np.testing.assert_array_equal(mean, means[0])
    np.testing.assert_array_equal(variance, variances[0])
    assert total == 32.0


def test_pooling_the_accumulator_matches_everything_it_saw():
    accumulator, seen, _ = fed_accumulator()
    assert len({len(rows) for rows in seen}) > 1  # the single-lane steps made the counts unequal
    mean, variance, total = SharedObsNorm(accumulator).pooled()
    union = np.concatenate(seen)
    # Each lane starts from RunningMeanStd's prior (1e-4 samples at mean 0, variance 1): pooled exactly too.
    prior = 1e-4 * LANES
    assert total == pytest.approx(len(union) + prior, rel=1e-12)
    expected_mean = union.sum(0) / total
    expected_variance = (np.square(union - expected_mean).sum(0) + prior * (1.0 + np.square(expected_mean))) / total
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(variance, expected_variance, rtol=1e-10, atol=1e-12)


def test_shared_statistics_give_one_input_per_physical_state():
    accumulator, _, rng = fed_accumulator()
    shared = SharedObsNorm(accumulator)
    shared.freeze()
    state = rng.normal(size=OBSERVATION_DIM).astype(np.float32)
    raw = np.tile(state, (LANES, 1))  # the same physical state met by every lane
    host = shared.normalize(raw)
    assert host.dtype == np.float32
    assert all(np.array_equal(host[0], row) for row in host)
    # The update's view: every lane row of the device tables holds the frozen statistic (copy_ broadcasts).
    obs_mean, obs_inv_std = torch.zeros((LANES, OBSERVATION_DIM)), torch.ones((LANES, OBSERVATION_DIM))
    obs_mean.copy_(torch.from_numpy(shared.mean))
    obs_inv_std.copy_(torch.from_numpy(shared.inv_std))
    rows = torch.arange(3 * LANES)  # (step, env) order: row % LANES is the lane
    updated = update_normalize(torch.from_numpy(np.tile(state, (len(rows), 1))), rows, obs_mean, obs_inv_std, LANES)
    assert all(torch.equal(updated[0], row) for row in updated)
    # ... and it is the host's input to the bit, so old_actor there is the behaviour policy.
    assert np.array_equal(updated[0].numpy(), host[0])
    # v60's lanes, by contrast, send that state to different inputs.
    lane_mean = torch.from_numpy(accumulator.means.astype(np.float32))
    lane_inv_std = torch.from_numpy((1.0 / np.sqrt(accumulator.variances + accumulator.epsilon)).astype(np.float32))
    by_lane = update_normalize(torch.from_numpy(raw), torch.arange(LANES), lane_mean, lane_inv_std, LANES)
    assert (by_lane - by_lane[0]).abs().max() > 0.1


def test_frozen_normalize_is_the_updates_fp32_arithmetic():
    rng = np.random.default_rng(3)
    mean = rng.normal(size=OBSERVATION_DIM).astype(np.float32)
    inv_std = (1.0 / rng.uniform(0.01, 3.0, OBSERVATION_DIM)).astype(np.float32)
    raw = rng.normal(scale=4.0, size=(64, OBSERVATION_DIM))  # float64, as environments return it
    host = frozen_normalize(raw, mean, inv_std, CLIP)
    stored = torch.from_numpy(raw.astype(np.float32))  # what the rollout buffer holds
    device = ((stored - torch.from_numpy(mean)) * torch.from_numpy(inv_std)).clamp(-CLIP, CLIP)
    assert host.dtype == np.float32 and np.array_equal(host, device.numpy())
    assert np.abs(host).max() == CLIP and (np.abs(host) < CLIP).any()  # both clipped and unclipped entries
    assert frozen_normalize(raw, mean, inv_std, CLIP) is not host  # rollout rows keep their own arrays
    np.testing.assert_array_equal(frozen_normalize(raw[0], mean, inv_std, CLIP), host[0])  # a single observation


def test_frozen_statistics_hold_through_a_rollout_and_move_at_the_next_freeze():
    accumulator, _, rng = fed_accumulator()
    shared = SharedObsNorm(accumulator)
    shared.freeze()
    mean, inv_std = shared.mean, shared.inv_std
    kept = mean.copy(), inv_std.copy()
    probe = rng.normal(size=(LANES, OBSERVATION_DIM))
    before = shared.normalize(probe)
    for step in range(200):  # a rollout: the accumulator counts every step, boundaries included
        batch = 3.0 + 5.0 * rng.normal(size=(LANES, OBSERVATION_DIM))
        ended = np.zeros(LANES, dtype=bool)
        ended[step % LANES] = step % 50 == 49
        finals = np.empty(LANES, dtype=object)
        for lane in np.flatnonzero(ended):
            finals[lane] = batch[lane] + 1.0
        accumulator.normalize_step(batch, ended, np.zeros(LANES, dtype=bool),
                                   {"final_observation": finals, "_final_observation": ended})
        assert shared.mean is mean and shared.inv_std is inv_std
    assert np.array_equal(mean, kept[0]) and np.array_equal(inv_std, kept[1])
    assert np.array_equal(shared.normalize(probe), before)
    live_mean, live_inv_std = shared.scale()  # what evaluation and saved statistics read: already moved
    assert not np.array_equal(live_mean, kept[0])
    shared.freeze()
    assert np.array_equal(shared.mean, live_mean) and np.array_equal(shared.inv_std, live_inv_std)
    assert np.array_equal(mean, kept[0])  # the previous freeze's arrays were not written in place
    assert not np.array_equal(shared.normalize(probe), before)


def test_saved_lane_rows_hold_the_pooled_moments_and_resume_exactly():
    accumulator, _, rng = fed_accumulator()
    shared = SharedObsNorm(accumulator)
    mean, variance, total = shared.pooled()
    means, variances, counts = shared.lane_state()
    assert means.shape == accumulator.means.shape and counts.shape == accumulator.counts.shape
    assert all(np.array_equal(row, mean) for row in means)  # any lane-indexed reader gets the shared statistic
    assert all(np.array_equal(row, variance) for row in variances)
    restored = pooled_moments(means, variances, counts)
    np.testing.assert_allclose(restored[0], mean, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(restored[1], variance, rtol=1e-14, atol=0.0)
    assert restored[2] == pytest.approx(total, rel=1e-14)
    # A resumed accumulator (loaded in place, as the trainer does) pools to the original's from then on.
    resumed = VectorObsNorm(LANES, (OBSERVATION_DIM,), clip=CLIP)
    np.copyto(resumed.means, means)
    np.copyto(resumed.variances, variances)
    np.copyto(resumed.counts, counts)
    for _ in range(50):
        batch = 2.0 * rng.normal(size=(LANES, OBSERVATION_DIM)) - 1.0
        accumulator.normalize(batch)
        resumed.normalize(batch)
    for original, again in zip(shared.pooled(), SharedObsNorm(resumed).pooled()):
        np.testing.assert_allclose(again, original, rtol=1e-10, atol=1e-12)


def test_a_lane_checkpoint_pools_on_resume():
    # A v60 checkpoint's per-lane rows, loaded in place: the first freeze is their exact pooled statistic.
    source, seen, _ = fed_accumulator(seed=4)
    resumed = VectorObsNorm(LANES, (OBSERVATION_DIM,), clip=CLIP)
    np.copyto(resumed.means, source.means.copy())
    np.copyto(resumed.variances, source.variances.copy())
    np.copyto(resumed.counts, source.counts.copy())
    shared = SharedObsNorm(resumed)
    shared.freeze()
    union = np.concatenate(seen)
    np.testing.assert_allclose(shared.mean, union.mean(0), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(1.0 / shared.inv_std, union.std(0), rtol=1e-4)


class TinyActor(torch.nn.Module):
    """State-independent diagonal Gaussian: small enough for the Lagrangian's gradient to be written out."""

    def __init__(self, action_dim):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.mean = torch.nn.Parameter(torch.randn(action_dim, generator=generator, dtype=torch.float64))
        self.log_std = torch.nn.Parameter(0.3 * torch.randn(action_dim, generator=generator, dtype=torch.float64))


def synthetic_actor_terms(states=64, action_dim=3, seed=1):
    """A linear pathwise objective -(c_s . a_s), a_s = mean + std eps_s, and KL(old_s || new) per state, with
    old policies spread so that states fall on both sides of the bound."""
    generator = torch.Generator().manual_seed(seed)
    actor = TinyActor(action_dim)
    slope = torch.randn((states, action_dim), generator=generator, dtype=torch.float64)
    noise = torch.randn((states, action_dim), generator=generator, dtype=torch.float64)
    spread = torch.linspace(0.0, 0.5, states, dtype=torch.float64)[:, None]
    old_mean = actor.mean.detach() + spread * torch.randn((states, action_dim), generator=generator,
                                                           dtype=torch.float64)
    old_std = actor.log_std.detach().exp() * torch.exp(
        0.3 * spread * torch.randn((states, action_dim), generator=generator, dtype=torch.float64))
    std = actor.log_std.exp().expand(states, -1)
    mean = actor.mean.expand(states, -1)
    objective = -(slope * (mean + std * noise)).sum(-1)
    kl = gaussian_kl(old_mean, old_std, mean, std)
    return actor, objective, kl, dict(slope=slope, noise=noise, old_mean=old_mean, old_std=old_std)


def test_penalty_gradient_is_the_lagrangians():
    bound, multiplier = 0.03, torch.tensor(0.7, dtype=torch.float64)
    actor, objective, kl, data = synthetic_actor_terms()
    outside = kl.detach() >= bound
    assert 0.2 < outside.double().mean() < 0.8  # a gate would have split these states
    loss = penalized_policy_loss(objective, kl, multiplier, bound)
    torch.testing.assert_close(loss, (objective + multiplier * kl).mean())
    mean_grad, log_std_grad = torch.autograd.grad(loss, (actor.mean, actor.log_std))
    # d/dmean:    -c + eta (mean - old_mean) / std^2
    # d/dlog_std: -c std eps + eta (1 - (old_std^2 + (old_mean - mean)^2) / std^2), averaged over states.
    std, mean = actor.log_std.detach().exp(), actor.mean.detach()
    offset = mean - data["old_mean"]
    expected_mean = (-data["slope"] + multiplier * offset / std.square()).mean(0)
    expected_log_std = (-data["slope"] * std * data["noise"]
                        + multiplier * (1.0 - (data["old_std"].square() + offset.square()) / std.square())).mean(0)
    torch.testing.assert_close(mean_grad, expected_mean, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(log_std_grad, expected_log_std, rtol=1e-10, atol=1e-12)


def test_penalty_gives_every_state_both_terms_and_the_gate_one():
    bound, multiplier = 0.03, torch.tensor(0.7, dtype=torch.float64)
    actor, objective, kl, data = synthetic_actor_terms()
    inside = kl.detach() < bound
    states = len(kl)

    def per_state_mean_gradient(loss_fn):
        """Row s: the gradient of state s's share of the loss with respect to the mean."""
        return torch.stack([torch.autograd.grad(
            loss_fn(objective[s:s + 1], kl[s:s + 1], multiplier, bound), actor.mean, retain_graph=True)[0]
            for s in range(states)])

    q_part = -data["slope"]
    kl_part = multiplier * (actor.mean.detach() - data["old_mean"]) / actor.log_std.detach().exp().square()
    torch.testing.assert_close(per_state_mean_gradient(penalized_policy_loss), q_part + kl_part)
    torch.testing.assert_close(per_state_mean_gradient(gated_policy_loss),
                               torch.where(inside[:, None], q_part, kl_part))


def test_gate_is_v65s_expression_in_value_and_gradient():
    bound, multiplier = 0.03, torch.tensor(0.7, dtype=torch.float64)
    actor, objective, kl, _ = synthetic_actor_terms()
    parameters = (actor.mean, actor.log_std)
    loss = gated_policy_loss(objective, kl, multiplier, bound)
    # v65 arm_actor_loss / v60 actor_loss, verbatim.
    inside = kl < bound
    reference = torch.where(inside, objective, kl * multiplier.detach()).mean()
    assert torch.equal(loss, reference)
    for ours, theirs in zip(torch.autograd.grad(loss, parameters, retain_graph=True),
                            torch.autograd.grad(reference, parameters, retain_graph=True)):
        assert torch.equal(ours, theirs)
    assert KL_POLICY_LOSSES == {"gate": gated_policy_loss, "penalty": penalized_policy_loss}
    assert not torch.equal(loss, penalized_policy_loss(objective, kl, multiplier, bound))


def test_arm_flags_are_validated():
    defaults = validate_args(Args())
    assert (defaults.obs_norm, defaults.kl_mode) == ("lane", "gate")
    validate_args(Args(obs_norm="shared", kl_mode="penalty", dqda_norm="clip", actor_critic="ema"))
    validate_args(Args(obs_norm="shared", staggered_starts=False, resume="checkpoint.pt"))
    for bad in (dict(obs_norm="pooled"), dict(kl_mode="soft"), dict(obs_norm="shared", staggered_starts=False),
                dict(obs_norm="shared", num_envs=1, minibatch_size=512)):
        with pytest.raises(ValueError):
            validate_args(Args(**bad))
    validate_args(Args(staggered_starts=False))  # the lane default keeps v65's freedom
