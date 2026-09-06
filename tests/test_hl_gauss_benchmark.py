"""Guard the proxy against false winners, not particular benchmark scores."""

import torch

from scripts.hlgauss.factorial import (
    Candidate,
    Critics,
    Labels,
    make_mrp,
    regression_data,
    td_metrics,
    update,
)


def test_batched_critic_updates_are_independent_even_when_neighbor_is_clipped():
    one = Critics(1, 6, 7)
    two = Critics(2, 6, 7)
    optimizer_one = torch.optim.Adam(one.parameters(), lr=0.003)
    optimizer_two = torch.optim.Adam(two.parameters(), lr=0.003)
    g = torch.Generator().manual_seed(21)
    x = torch.randn(16, 6, generator=g)
    target = torch.randn(1, 16, 7, generator=g)
    # A high-gradient neighbor must not alter candidate one's step size.
    neighbors = torch.cat((target, target * 1000))
    for _ in range(3):
        update(one, optimizer_one, x, target, True)
        update(two, optimizer_two, x, neighbors, True)
    torch.testing.assert_close(one(x)[0], two(x)[0], atol=1e-6, rtol=1e-5)


def test_markov_truth_solves_bellman_and_perfect_critic_has_oracle_actor_utility():
    _, transition, reward, value, advantage, gamma = make_mrp(50)
    torch.testing.assert_close(transition.sum(-1), torch.ones(2, len(value)))
    solved = torch.linalg.solve(torch.eye(len(value)).double() - gamma * transition.double().mean(0), reward.double().mean(0))
    torch.testing.assert_close(solved, value.double(), atol=4e-5, rtol=1e-5)
    exact = td_metrics(value, transition, reward, value, advantage, gamma, 50)
    flat = td_metrics(torch.zeros_like(value), transition, reward, value, advantage, gamma, 50)
    assert exact["advantage_relative_mse"] < 1e-10
    assert exact["policy_improvement"] > 0
    assert abs(exact["policy_utility_ratio"] - 1) < 1e-6
    assert flat["advantage_relative_mse"] > 1
    assert flat["policy_utility_ratio"] < exact["policy_utility_ratio"]


def test_twohot_control_preserves_noisy_raw_mean_not_transformed_mean():
    scalar = Labels(Candidate(kind="twohot", transform="symlog", bins=101), 50)
    y = torch.tensor([1.0, 31.0])
    mixture = scalar.project(y).mean(0)
    torch.testing.assert_close(scalar.decode(mixture.log()), y.mean())


def test_confirmation_has_disjoint_inputs_and_nonstationary_targets():
    train_x, targets, test_x, truth, _ = regression_data(50, "moving", 0)
    other_x, _, other_test, _, _ = regression_data(50, "moving", 1)
    assert not torch.equal(train_x, other_x)
    assert not torch.equal(test_x, other_test)
    assert ((truth[1] - truth[0]) / 50).square().mean() > 0.1
    assert targets[0].shape == targets[1].shape


def test_moving_auc_keeps_the_discontinuous_switch_in_its_own_phase():
    from scripts.hlgauss.factorial import regression

    (row,) = regression([Candidate(kind="mse", bins=1)], 5, "moving", 0, 80, 0.003)
    curve = row["curve"]
    boundary = [point for point in curve if point["step"] == 40]
    assert [point["phase"] for point in boundary] == [0, 1]
    assert boundary[1]["nmse"] > boundary[0]["nmse"]
    areas = []
    for phase in (0, 1):
        points = [point for point in curve if point["phase"] == phase]
        areas.append(sum((a["nmse"] + b["nmse"]) * (b["step"] - a["step"]) / 2 for a, b in zip(points, points[1:])))
    assert abs(row["auc_nmse"] - sum(areas) / 80) < 1e-12
