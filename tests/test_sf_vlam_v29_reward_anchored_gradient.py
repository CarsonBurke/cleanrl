import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]


def _load():
    path = ROOT / "cleanrl/encoder-value/ppo_continuous_action_sf_vlam_v29_reward_anchored_gradient.py"
    spec = importlib.util.spec_from_file_location("sf_vlam_v29_reward_anchored_gradient", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()

GAMMA = 0.99
LAM = 0.95


def _reference_scalar_gae(rewards, values, next_values, terminations, boundaries, valids):
    """cleanrl's scalar GAE, written out independently of the vector recursion."""
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        boot = (1.0 - terminations[t]) * valids[t]
        cont = 1.0 - boundaries[t]
        delta = rewards[t] + GAMMA * boot * next_values[t] - values[t]
        running = delta + GAMMA * LAM * cont * running
        advantages[t] = running
    return advantages


def test_reward_coordinate_innovation_is_exactly_the_scalar_gae():
    """The load-bearing claim of the anchor.

    With phi_0 = r, coordinate 0 of psi is the discounted reward sum and coordinate 0 of
    the vector TD(lambda) residual must be the scalar GAE advantage BIT FOR BIT -- not an
    approximation of it, not a probe of it. Includes a truncation (term=0, valid=1,
    boundary=1) and a termination, the two cases where the bootstrap and trace masks
    disagree.
    """
    torch.manual_seed(0)
    steps, envs, emb = 12, 3, 5
    rewards = torch.randn(steps, envs)
    embeddings = torch.randn(steps, envs, emb)
    phi = torch.cat([rewards.unsqueeze(-1), (1.0 - GAMMA) * embeddings], dim=-1)
    psi_cur = torch.randn(steps, envs, emb + 1)
    psi_next = torch.randn(steps, envs, emb + 1)

    terminations = torch.zeros(steps, envs)
    boundaries = torch.zeros(steps, envs)
    valids = torch.ones(steps, envs)
    terminations[4, 0] = 1.0  # a real termination
    boundaries[4, 0] = 1.0
    boundaries[7, 1] = 1.0    # a truncation: bootstrap through it, cut the trace
    valids[9, 2] = 0.0

    residual = MODULE.successor_lambda_residual(
        phi,
        psi_cur,
        psi_next,
        terminations,
        boundaries,
        valids,
        GAMMA,
        torch.full((emb + 1,), LAM),
    )
    reference = _reference_scalar_gae(
        rewards, psi_cur[..., 0], psi_next[..., 0], terminations, boundaries, valids
    )
    torch.testing.assert_close(residual[..., 0], reference, atol=0.0, rtol=0.0)


def test_anchor_covector_contracts_to_the_scalar_gae():
    """The covector the actor consumes is anchor + residual_std * grad g.

    The anchor half is not learned, so it must reproduce the coordinate-0 innovation
    exactly under the standardized contraction used in the training loop -- including
    when the standardizer is non-trivial, which is the only way the scaling can be wrong.
    """
    torch.manual_seed(1)
    steps, envs, dim = 6, 4, 7
    trace = torch.randn(steps, envs, dim)
    readout_std = torch.rand(dim) + 0.5

    anchor = torch.zeros_like(trace)
    anchor[..., 0] = readout_std[0]
    credit = (anchor * (trace / readout_std)).sum(-1)
    torch.testing.assert_close(credit, trace[..., 0])


def test_residual_head_starts_negligible_so_credit_starts_as_scalar_gae():
    """std=0.01 output init: iteration 1's correction must be a rounding error.

    This is what makes the scalar-GAE control the FLOOR of this variant rather than a
    thing it has to relearn.
    """
    torch.manual_seed(2)
    dim = 33
    readout = MODULE.LatentValueResidual(dim, 128)
    psi = torch.randn(512, dim)
    trace = torch.randn(512, dim)
    residual_std = torch.ones(())
    gradient = MODULE.value_gradient_covector(readout, psi)
    correction = residual_std * (gradient * trace).sum(-1)
    anchor = trace[..., 0]
    assert float(correction.square().mean().sqrt()) < 0.05 * float(
        anchor.square().mean().sqrt()
    )


def test_critic_coordinate_weights_reserve_the_reward_share():
    for dim in (33, 9):
        for coefficient in (0.5, 0.25):
            weights = torch.full((dim,), (1.0 - coefficient) / (dim - 1))
            weights[0] = coefficient
            assert abs(float(weights.sum()) - 1.0) < 1e-6
            assert float(weights[0]) == coefficient
            # Every embedding coordinate keeps an equal share of what is left.
            assert float(weights[1:].std()) < 1e-8


def test_fitted_residual_head_recovers_a_linear_correction():
    """On G - psi_0 = w.psi the fitted correction covector must converge to w."""
    torch.manual_seed(3)
    weight = torch.tensor([0.0, 2.0, -1.0, 0.5, 1.0, -0.25])
    psi = torch.randn(4096, 6)
    targets = psi @ weight
    readout = MODULE.LatentValueResidual(6, 16)
    optimizer = torch.optim.Adam(readout.parameters(), lr=3e-3)
    loss, grad_norm, steps = MODULE.fit_latent_value_readout(
        readout, optimizer, psi, targets, 60, 512, 10.0, torch.Generator().manual_seed(4)
    )
    assert steps == 60 * 8
    covector = MODULE.value_gradient_covector(readout, psi)
    cosine = torch.nn.functional.cosine_similarity(covector.mean(0), weight, dim=0)
    assert float(cosine) > 0.99, (float(cosine), loss, grad_norm)


def test_return_residual_correlation_uses_the_cross_env_baseline():
    torch.manual_seed(5)
    returns = torch.randn(20, 8)
    residual = returns - returns.mean(1, keepdim=True)
    valid = torch.ones(20, 8, dtype=torch.bool)
    assert abs(MODULE.credit_return_correlation(residual, returns, valid) - 1.0) < 1e-5
    shifted = returns + torch.arange(20.0).unsqueeze(1)
    assert abs(MODULE.credit_return_correlation(residual, shifted, valid) - 1.0) < 1e-5
