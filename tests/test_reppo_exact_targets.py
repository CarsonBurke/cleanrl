"""REPPO v61 exact (Q(lambda)-corrected) critic targets on a deterministic MDP with an exactly solved soft Q^pi.

The claims under test (v61 header): with deterministic transitions the corrected target's error is only the
critic's own error, sum_{k>=1} gamma^k lam^(k-1) [e_V(s_{t+k}) - lam e_Q(s_{t+k}, a_{t+k})], so it equals Q^pi
exactly at the fixed point for every lambda, while REPPO's soft lambda-return keeps a noise floor from the
sampled future actions. Terminations, truncations (bootstrapping from the true final state) and rollout ends
follow the trainer's conventions.
"""

import numpy as np
import pytest
import torch

from cleanrl.reppo import ppo_continuous_action_reppo_exacttarget_v61 as v61
from cleanrl.shared.ppo_loop import get_gae_fn

GAMMA, TEMPERATURE = 0.97, 0.3
STATES, ACTIONS = 12, 3
GAE = get_gae_fn(compiled=False, explicit_next_values=True)


class Mdp:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.next = rng.integers(0, STATES, (STATES, ACTIONS))
        self.reward = rng.normal(size=(STATES, ACTIONS))
        self.terminal = rng.random((STATES, ACTIONS)) < 0.08
        logits = rng.normal(size=(STATES, ACTIONS))
        # Mixed with uniform so every (s, a) is visited: unvisited pairs are unconstrained by any target.
        self.pi = 0.5 * np.exp(logits) / np.exp(logits).sum(1, keepdims=True) + 0.5 / ACTIONS
        self.bonus = -TEMPERATURE * (self.pi * np.log(self.pi)).sum(1)  # T H(s)
        # Soft policy evaluation, solved exactly: Q = r + gamma (1 - term) P V, V = pi . Q + T H.
        size = STATES * ACTIONS
        transfer = np.zeros((size, size))
        for s in range(STATES):
            for a in range(ACTIONS):
                if not self.terminal[s, a]:
                    transfer[s * ACTIONS + a, self.next[s, a] * ACTIONS:(self.next[s, a] + 1) * ACTIONS] = \
                        GAMMA * self.pi[self.next[s, a]]
        constant = self.reward.flatten() + GAMMA * np.array(
            [0.0 if self.terminal[s, a] else self.bonus[self.next[s, a]]
             for s in range(STATES) for a in range(ACTIONS)])
        self.q = np.linalg.solve(np.eye(size) - transfer, constant).reshape(STATES, ACTIONS)

    def soft_value(self, q):
        return (self.pi * q).sum(1) + self.bonus

    def rollout(self, envs, steps, seed, truncate_p=0.03):
        """Time-major on-policy data with the trainer's boundary conventions: a termination resets the env;
        a truncation bootstraps from the true final state, then resets; the last step is a rollout end."""
        rng = np.random.default_rng(seed)
        state = rng.integers(0, STATES, envs)
        data = {k: np.zeros((steps, envs), dtype=int) for k in ("s", "a", "s_next")}
        data["term"], data["cut"] = np.zeros((steps, envs)), np.zeros((steps, envs))
        for t in range(steps):
            action = np.array([rng.choice(ACTIONS, p=self.pi[s]) for s in state])
            nxt = self.next[state, action]
            data["s"][t], data["a"][t], data["s_next"][t] = state, action, nxt
            data["term"][t] = self.terminal[state, action]
            data["cut"][t] = (rng.random(envs) < truncate_p) | (t == steps - 1)
            reset = (data["term"][t] > 0) | (data["cut"][t] > 0)
            state = np.where(reset, rng.integers(0, STATES, envs), nxt)
        data["r"] = self.reward[data["s"], data["a"]]
        return data


def tensors(*arrays):
    return [torch.as_tensor(np.asarray(x), dtype=torch.float64) for x in arrays]


def exact_targets(mdp, data, q, lam):
    rewards, taken, next_values, term, cut = tensors(
        data["r"], q[data["s"], data["a"]], mdp.soft_value(q)[data["s_next"]], data["term"], data["cut"])
    return v61.exact_lambda_returns(GAE, rewards, taken, next_values, term, cut, GAMMA, lam).numpy()


def reppo_targets(mdp, data, q, lam):
    """The trainer's soft branch: the expected bonus T H(s_{t+1}) folded into the reward where the chain
    continues, V' = E_pi Q + T H as next value."""
    continuing = 1.0 - data["term"]
    rewards = data["r"] + GAMMA * lam * continuing * (1.0 - data["cut"]) * mdp.bonus[data["s_next"]]
    rewards, next_values, term, cut = tensors(rewards, mdp.soft_value(q)[data["s_next"]], data["term"], data["cut"])
    return v61.lambda_returns(GAE, rewards, next_values, term, cut, GAMMA, lam).numpy()


@pytest.fixture(scope="module")
def mdp():
    return Mdp()


def test_solved_q_satisfies_soft_bellman(mdp):
    backup = mdp.reward + GAMMA * np.where(mdp.terminal, 0.0, mdp.soft_value(mdp.q)[mdp.next])
    np.testing.assert_allclose(mdp.q, backup, atol=1e-10)


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.8, 0.95, 1.0])
def test_exact_target_is_q_pi_at_the_fixed_point(mdp, lam):
    data = mdp.rollout(16, 64, seed=1)
    assert data["term"].any() and (data["cut"][:-1] > 0).any()
    np.testing.assert_allclose(exact_targets(mdp, data, mdp.q, lam), mdp.q[data["s"], data["a"]], atol=1e-10)


def test_reppo_target_is_unbiased_but_noisy_at_the_fixed_point(mdp):
    data = mdp.rollout(512, 64, seed=2)
    error = reppo_targets(mdp, data, mdp.q, 0.8) - mdp.q[data["s"], data["a"]]
    assert error.std() > 0.1  # the sampled future actions' true advantages
    assert abs(error.mean()) < 4 * error.std() / np.sqrt(error.size / 8)  # loose: errors correlate in time
    np.testing.assert_allclose(reppo_targets(mdp, data, mdp.q, 0.0), mdp.q[data["s"], data["a"]], atol=1e-10)


@pytest.mark.parametrize("lam", [0.0, 0.8, 1.0])
def test_exact_target_error_is_the_critic_error_series(mdp, lam):
    rng = np.random.default_rng(3)
    q_hat = mdp.q + rng.normal(scale=0.5, size=mdp.q.shape)
    data = mdp.rollout(8, 48, seed=4)
    error = exact_targets(mdp, data, q_hat, lam) - mdp.q[data["s"], data["a"]]
    e_v = (mdp.soft_value(q_hat) - mdp.soft_value(mdp.q))[data["s_next"]]  # error of V' at s_{t+1}
    e_q = (q_hat - mdp.q)[data["s"], data["a"]]  # error of Q' at (s_t, a_t)
    # D_t = gamma e_V(s_{t+1}) at a cut, 0 at a termination, else gamma (e_V - lam e_Q)_{t+1} + gamma lam D_{t+1}.
    expected = np.zeros_like(error)
    for t in reversed(range(len(error))):
        chained = GAMMA * (e_v[t] - lam * e_q[t + 1]) + GAMMA * lam * expected[t + 1] if t + 1 < len(error) else 0.0
        expected[t] = np.where(data["term"][t] > 0, 0.0, np.where(data["cut"][t] > 0, GAMMA * e_v[t], chained))
    np.testing.assert_allclose(error, expected, atol=1e-10)


def fit_on_fixed_data(mdp, data, target_fn, lam, iterations):
    """Tabular critic refit to the targets' per-(s, a) mean on one fixed on-policy dataset."""
    index = data["s"] * ACTIONS + data["a"]
    counts = np.bincount(index.ravel(), minlength=STATES * ACTIONS)
    assert counts.min() > 0
    q = np.zeros_like(mdp.q)
    for _ in range(iterations):
        q = (np.bincount(index.ravel(), target_fn(mdp, data, q, lam).ravel(), STATES * ACTIONS) / counts)
        q = q.reshape(STATES, ACTIONS)
    return np.abs(q - mdp.q).max()


@pytest.mark.parametrize("lam", [0.0, 0.8, 1.0])
def test_fitting_exact_targets_recovers_q_pi_from_finite_data(mdp, lam):
    """Every deterministic transition is an exact constraint: finite on-policy data pins Q^pi down."""
    data = mdp.rollout(256, 64, seed=5)
    assert fit_on_fixed_data(mdp, data, exact_targets, lam, 1500) < 1e-8


def test_fitting_reppo_targets_keeps_a_noise_floor(mdp):
    data = mdp.rollout(256, 64, seed=5)
    assert fit_on_fixed_data(mdp, data, reppo_targets, 0.8, 1500) > 1e-2
