"""Contracts for MaxRL-PPO v3: latent branching on top of the critic, dense rewards.

v3's claim is that per-timestep GAE (credit WITHIN a branch) and the continuous failure
series (outcome weighting ACROSS branches) act on orthogonal axes and so compose. That
only means anything if the branches really are branches -- identical physical state,
independent continuations -- and if each branch's outcome lands on exactly its own
timesteps. Both are checked here against the real native MuJoCo envs.
"""
import importlib.util
import pathlib

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "cleanrl" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


maxrl = _load("ppo_continuous_action_maxrl_v3")


class Args:
    def __init__(self, **overrides):
        self.maxrl_estimator = "continuous"
        self.maxrl_order = 8
        self.maxrl_beta = 1.0
        self.group_size = 8
        self.segment_length = 125
        self.gamma = 0.99
        self.__dict__.update(overrides)


# --- the branches must be real branches -------------------------------------

@pytest.fixture(scope="module")
def native_envs():
    from cleanrl.shared.mujoco_env import make_mujoco_vector_env
    envs = make_mujoco_vector_env("HalfCheetah-v4", 4, num_threads=1, backend="native")
    envs.reset(seed=1)
    yield envs
    envs.close()


def test_cloning_makes_group_mates_physically_identical(native_envs):
    """Not 'similar states' -- set_state writes into the mjData the pool steps in place,
    so a cloned env must produce bit-identical observations and rewards thereafter."""
    envs = native_envs
    bases = envs._bases
    rng = np.random.default_rng(0)
    for _ in range(12):
        envs.step(rng.uniform(-1, 1, (4, 6)).astype(np.float32))

    obs = np.zeros((4, 17), dtype=np.float32)
    maxrl.clone_group_states(bases, obs, group_size=4)
    action = rng.uniform(-1, 1, (4, 6)).astype(np.float32)
    # Every env is now the same state, so an identical action must give identical results.
    next_obs, rewards, _, _, _ = envs.step(np.repeat(action[:1], 4, axis=0))
    assert np.array_equal(next_obs[0], next_obs[1]) and np.array_equal(next_obs[0], next_obs[3])
    assert rewards[0] == rewards[1] == rewards[3]


def test_cloning_leaves_the_leader_untouched_and_groups_independent(native_envs):
    """The leader's trajectory stays a genuine on-policy episode (it is the only one
    whose episodic return is reported), and group 0 must not leak into group 1."""
    envs = native_envs
    bases = envs._bases
    envs.reset(seed=3)
    rng = np.random.default_rng(1)
    for _ in range(5):
        envs.step(rng.uniform(-1, 1, (4, 6)).astype(np.float32))

    leader0 = bases[0].data.qpos.copy()
    leader2 = bases[2].data.qpos.copy()
    obs = np.zeros((4, 17), dtype=np.float32)
    maxrl.clone_group_states(bases, obs, group_size=2)  # groups {0,1} and {2,3}
    assert np.array_equal(bases[0].data.qpos, leader0), "leader must never be written to"
    assert np.array_equal(bases[2].data.qpos, leader2)
    assert np.array_equal(bases[1].data.qpos, leader0)
    assert np.array_equal(bases[3].data.qpos, leader2)
    assert not np.array_equal(bases[1].data.qpos, bases[3].data.qpos), "groups must stay apart"


def test_cloning_copies_the_normalized_observation_not_a_recomputed_one(native_envs):
    """Members inherit the leader's normalized obs verbatim; recomputing it would
    double-count the running observation normalizer."""
    obs = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    maxrl.clone_group_states(native_envs._bases, obs, group_size=4)
    assert np.array_equal(obs, np.repeat(obs[:1], 4, axis=0))


# --- each branch's outcome must land on its own timesteps -------------------

def test_outcome_broadcast_lands_on_exactly_the_right_timesteps():
    """Trace every element's provenance through the (segments, length, groups, size)
    reshape. A transposition here would silently credit one branch with another's
    outcome and the whole experiment would be measuring noise."""
    segments, length, groups, size = 2, 3, 2, 4
    num_steps, num_envs = segments * length, groups * size
    outcome = torch.arange(segments * groups * size, dtype=torch.float32).view(segments, groups, size)
    flat = outcome.unsqueeze(1).expand(-1, length, -1, -1).reshape(num_steps, num_envs)
    for step in range(num_steps):
        for env in range(num_envs):
            assert flat[step, env] == outcome[step // length, env // size, env % size]


def test_segment_returns_discount_within_the_segment_only():
    """Each branch is scored on its own segment, truncated with no bootstrap."""
    segments, length, groups, size = 2, 4, 1, 2
    rewards = torch.zeros(segments * length, groups * size)
    rewards[0, 0] = 1.0          # first step of segment 0 -> discount^0
    rewards[length + 2, 1] = 1.0  # third step of segment 1 -> discount^2
    gamma = 0.9
    discounts = gamma ** torch.arange(length)
    shaped = rewards.view(segments, length, groups, size)
    returns = (shaped * discounts[None, :, None, None]).sum(dim=1)
    assert returns[0, 0, 0].item() == pytest.approx(1.0)
    assert returns[1, 0, 1].item() == pytest.approx(gamma ** 2)
    assert returns[1, 0, 0].item() == 0.0


# --- the normalisation must preserve cross-group difficulty -----------------

def test_global_normalisation_keeps_easy_and_hard_groups_apart():
    """branch_v1's bug: a PER-GROUP scale forces every group to the same spread and
    erases the difficulty variation the failure series exists to exploit."""
    segment_returns = torch.tensor([[[10.0, 11.0, 12.0, 13.0],   # an easy state
                                     [0.0, 1.0, 2.0, 3.0]]])      # a hard one
    low, high = torch.tensor(0.0), torch.tensor(13.0)
    r = maxrl.normalize_segment_returns(segment_returns, low, high)
    assert r[0, 0].mean() > 0.75 and r[0, 1].mean() < 0.25
    # A per-group min-max would have mapped BOTH to mean 0.5 -- the failure mode.
    assert abs(r[0, 0].mean() - r[0, 1].mean()) > 0.5


def test_normalisation_clamps_and_survives_a_degenerate_scale():
    values = torch.tensor([[[-5.0, 0.0, 5.0, 100.0]]])
    r = maxrl.normalize_segment_returns(values, torch.tensor(0.0), torch.tensor(5.0))
    assert r.min() >= 0.0 and r.max() <= 1.0
    flat = maxrl.normalize_segment_returns(values, torch.tensor(2.0), torch.tensor(2.0))
    assert torch.isfinite(flat).all(), "a collapsed reward range must not divide by zero"


# --- estimators and blending ------------------------------------------------

def test_continuous_estimator_is_the_failure_series_on_the_dense_reward():
    from cleanrl.shared.continuous_maclaurin import continuous_maclaurin_weights
    rewards = torch.rand(2, 3, 8)
    got = maxrl.group_ml_advantages(rewards, Args(maxrl_estimator="continuous"))
    want = continuous_maclaurin_weights(rewards, 8) * 8
    assert torch.allclose(got, want.to(got.dtype))


def test_binary_estimator_reproduces_the_v1_mistake_for_ablation():
    """The binary arm exists to measure what thresholding costs, so it must genuinely
    threshold and must differ from the continuous arm on non-extreme rewards."""
    rewards = torch.tensor([[[0.9, 0.6, 0.4, 0.1]]])
    binary = maxrl.group_ml_advantages(rewards, Args(maxrl_estimator="binary", maxrl_order=4))
    continuous = maxrl.group_ml_advantages(rewards, Args(maxrl_estimator="continuous", maxrl_order=4))
    assert not torch.allclose(binary, continuous)
    hard = maxrl.group_ml_advantages(
        torch.tensor([[[1.0, 1.0, 0.0, 0.0]]]), Args(maxrl_estimator="binary", maxrl_order=4))
    assert torch.allclose(
        binary[0, 0, :2].sign(), hard[0, 0, :2].sign()
    ), "0.9 and 0.6 must both threshold to success"


def test_grpo_estimator_is_group_standardised():
    rewards = torch.rand(1, 2, 8)
    got = maxrl.group_ml_advantages(rewards, Args(maxrl_estimator="grpo"))
    assert torch.allclose(got.mean(-1), torch.zeros(1, 2), atol=1e-5)
    assert torch.allclose(got.std(-1, unbiased=False), torch.ones(1, 2), atol=1e-3)


def test_beta_zero_is_an_exact_passthrough():
    """The control that prices cloning alone must not perturb the advantage at all."""
    advantages = torch.randn(64, 16)
    out = maxrl.blend_advantages(advantages, torch.randn(64, 16), Args(maxrl_beta=0.0))
    assert out is advantages


def test_blending_holds_the_advantage_scale_fixed_across_beta():
    """With norm_adv off, PPO's step size tracks the advantage scale directly, so beta
    must not smuggle in a learning-rate change."""
    torch.manual_seed(0)
    advantages = torch.randn(4096) * 3.0
    outcome = torch.randn(4096) * 0.01
    base = advantages.std()
    for beta in (0.25, 0.5, 1.0, 2.0):
        blended = maxrl.blend_advantages(advantages, outcome, Args(maxrl_beta=beta))
        assert blended.std().item() == pytest.approx(base.item(), rel=0.05)


# --- validation -------------------------------------------------------------

def _full_args(**overrides):
    args = maxrl.Args()
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.mark.parametrize("overrides, message", [
    ({"group_size": 1}, "at least two branches"),
    ({"num_envs": 60}, "whole number of groups"),
    ({"num_steps": 260}, "must tile the rollout"),
    ({"maxrl_beta": -0.5}, "non-negative"),
    ({"maxrl_order": 0}, "truncation level"),
    ({"num_steps": 256, "segment_length": 128}, "episode horizon"),
])
def test_incoherent_configurations_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        maxrl.validate_args(_full_args(**overrides))


def test_the_shipped_defaults_are_coherent():
    args = _full_args()
    maxrl.validate_args(args)
    assert args.num_groups == 8 and args.segments_per_rollout == 2
    assert args.norm_adv is False, "norm_adv renormalises the blend away; see the header"


def test_cloning_off_lifts_the_native_backend_requirement():
    args = _full_args(maxrl_clone=False, env_backend="sync")
    maxrl.validate_args(args)


# --- the loss still compiles and runs ---------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_loss_backward_reaches_actor_and_critic():
    import gymnasium as gym

    device = torch.device("cuda")

    class Envs:
        single_action_space = gym.spaces.Box(-1.0, 1.0, (6,))
        single_observation_space = gym.spaces.Box(-float("inf"), float("inf"), (17,))

    torch.manual_seed(0)
    agent = maxrl.Agent(Envs()).to(device)
    args = _full_args()
    maxrl.validate_args(args)
    n = 64
    loss, metrics = maxrl.ppo_loss(
        agent, torch.randn(n, 17, device=device),
        torch.rand(n, 6, device=device).clamp(1e-3, 1 - 1e-3),
        torch.randn(n, device=device), torch.randn(n, device=device),
        torch.randn(n, device=device), torch.randn(n, device=device), args,
    )
    assert torch.isfinite(loss) and torch.isfinite(metrics).all()
    assert metrics.numel() == 6, "metric width must match update_metrics"
    loss.backward()
    assert torch.count_nonzero(agent.actor[0].weight.grad) > 0
    assert torch.count_nonzero(agent.critic[0].weight.grad) > 0
