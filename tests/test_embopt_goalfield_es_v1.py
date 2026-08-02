import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "cleanrl"
    / "embedding-optimization"
    / "ppo_continuous_action_embopt_goalfield_es_v1.py"
)
SPEC = importlib.util.spec_from_file_location("goalfield_es_v1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_paired_reset_seeds_are_interleaved_duplicates():
    assert MODULE.paired_reset_seeds(100, 2, 3) == [106, 106, 107, 107, 108, 108]


def test_orthogonal_directions_have_expected_covariance_scale():
    generator = torch.Generator().manual_seed(3)
    directions = MODULE.orthogonal_directions(32, 8, generator, torch.device("cpu"))
    gram = directions @ directions.T
    torch.testing.assert_close(gram, 32 * torch.eye(8), atol=1e-5, rtol=1e-5)


def test_mirrored_population_is_centered_on_theta():
    theta = torch.arange(32, dtype=torch.float32)
    directions = torch.eye(32)[:8]
    population = MODULE.mirrored_population(theta, directions, 0.2)
    torch.testing.assert_close(
        0.5 * (population[0::2] + population[1::2]),
        theta.expand(8, -1),
    )


def test_spherical_population_has_fixed_norm_and_tangent_directions():
    theta = torch.zeros(32)
    theta[0] = 1.0
    generator = torch.Generator().manual_seed(7)
    directions = MODULE.orthogonal_directions(
        32, 8, generator, torch.device("cpu"), tangent=theta
    )
    torch.testing.assert_close(directions @ theta, torch.zeros(8), atol=1e-5, rtol=0)
    population = MODULE.mirrored_population(
        theta, directions, sigma=0.1, target_norm=1.0
    )
    torch.testing.assert_close(
        population.norm(dim=-1), torch.ones(16), atol=1e-6, rtol=1e-6
    )


def test_es_gradient_follows_better_mirrored_directions():
    directions = torch.eye(4)
    returns = torch.tensor([3.0, 1.0, 1.0, 3.0, 2.0, 2.0, 5.0, 1.0])
    gradient, differences, _ = MODULE.normalized_es_gradient(
        returns, directions, sigma=0.1
    )
    assert gradient[0] > 0
    assert gradient[1] < 0
    assert gradient[2] == 0
    assert gradient[3] > gradient[0]
    torch.testing.assert_close(differences, torch.tensor([2.0, -2.0, 0.0, 4.0]))


def test_continuous_goal_is_inside_local_edge_convex_hull():
    y = torch.zeros(2, 2)
    theta = torch.zeros(2, 4)
    atlas_y = torch.tensor([[0.0, 0.0], [0.1, 0.0], [8.0, 8.0]])
    atlas_delta = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-5.0, -5.0]])
    desired, weights, indices, candidates = MODULE.continuous_local_goal(
        y,
        theta,
        atlas_y,
        atlas_delta,
        field_features_count=2,
        local_edges=2,
        temperature=0.2,
    )
    torch.testing.assert_close(weights.sum(-1), torch.ones(2))
    assert torch.all(weights >= 0)
    torch.testing.assert_close(desired, (candidates * weights[..., None]).sum(1))
    assert not torch.any(indices == 2)


def test_field_features_include_bias():
    y = torch.tensor([[2.0, -2.0, 5.0]])
    features = MODULE.field_features(y, 3)
    torch.testing.assert_close(features[:, 0], torch.ones(1))
    torch.testing.assert_close(features[:, 1:], torch.tanh(y[:, :2]))
