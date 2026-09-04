import torch

from cleanrl.iterthink.critic_variants.ppo_continuous_action_iterthink_v24_beta_d4c51_dgae_v3 import (
    Args,
    c51_project,
    distributional_lambda_target,
)


def _scalar_lambda_return(rewards, dones, next_done, values, next_value, gamma, lam):
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            nonterm = 1.0 - next_done
            nextval = next_value
        else:
            nonterm = 1.0 - dones[t + 1]
            nextval = values[t + 1]
        delta = rewards[t] + gamma * nextval * nonterm - values[t]
        adv[t] = last = delta + gamma * lam * nonterm * last
    return adv + values


def _atoms(n=41, lo=-20.0, hi=20.0):
    return torch.linspace(lo, hi, n)


def _point_mass(atoms, value):
    # nearest-atom one-hot
    idx = (atoms - value).abs().argmin()
    p = torch.zeros_like(atoms)
    p[idx] = 1.0
    return p


def test_c51_project_on_grid_stays_point_mass():
    atoms = _atoms()
    loc = torch.tensor([[2.0]])      # exactly on a grid atom (40 spacing/40 = 1.0)
    w = torch.tensor([[1.0]])
    out = c51_project(loc, w, atoms)
    assert torch.isclose(out.sum(), torch.tensor(1.0))
    assert out.max() > 0.999          # all mass on one bin -> no diffusion
    mean = (out * atoms).sum()
    assert torch.isclose(mean, torch.tensor(2.0), atol=1e-5)


def test_c51_project_preserves_mean_off_grid():
    atoms = _atoms()
    loc = torch.tensor([[2.3], [-7.6], [0.4]])
    w = torch.ones_like(loc)
    out = c51_project(loc, w, atoms)
    means = (out * atoms).sum(-1)
    assert torch.allclose(means, loc.squeeze(-1), atol=1e-5)
    assert torch.allclose(out.sum(-1), torch.ones(3), atol=1e-6)


def _peaked(atoms, centers, width=0.8):
    # concentrated categorical around `centers` (realistic critic output; ~no edge mass)
    logits = -((atoms - centers.unsqueeze(-1)) ** 2) / (2 * width ** 2)
    return torch.softmax(logits, -1)


def test_target_mean_matches_scalar_lambda_return():
    g = torch.Generator().manual_seed(0)
    T, B = 6, 3
    atoms = _atoms(n=81)
    n = atoms.numel()
    gamma, lam = 0.99, 0.95
    # Concentrated distributions in [-4,4] so the Bellman shift never hits the ±20 edge.
    value_probs = _peaked(atoms, 8.0 * torch.rand(T, B, generator=g) - 4.0)
    bootstrap_probs = _peaked(atoms, 8.0 * torch.rand(B, generator=g) - 4.0)
    rewards = torch.randn(T, B, generator=g)
    dones = torch.zeros(T, B)
    next_done = torch.zeros(B)

    lam_probs, _ = distributional_lambda_target(
        rewards, dones, next_done, value_probs, bootstrap_probs, atoms, gamma, lam
    )
    values = (value_probs * atoms).sum(-1)
    next_value = (bootstrap_probs * atoms).sum(-1)
    scalar_ret = _scalar_lambda_return(rewards, dones, next_done, values, next_value, gamma, lam)
    dist_mean = (lam_probs * atoms).sum(-1)
    assert torch.allclose(dist_mean, scalar_ret, atol=1e-4), (dist_mean - scalar_ret).abs().max()


def test_no_diffusion_consistent_value_stays_sharp():
    # THE v3 fix: a self-consistent (zero) value function must stay a (near) point
    # mass through the full TD(lambda) recursion -- v2's Gaussian projection diffused
    # it to ~2-bin width every step; C51 linear-interp keeps it sharp.
    T, B = 64, 2
    atoms = _atoms(n=81)
    n = atoms.numel()
    gamma, lam = 0.99, 0.95
    value_probs = torch.stack([torch.stack([_point_mass(atoms, 0.0) for _ in range(B)]) for _ in range(T)])
    bootstrap_probs = torch.stack([_point_mass(atoms, 0.0) for _ in range(B)])
    rewards = torch.zeros(T, B)
    dones = torch.zeros(T, B)
    next_done = torch.zeros(B)

    lam_probs, target_std = distributional_lambda_target(
        rewards, dones, next_done, value_probs, bootstrap_probs, atoms, gamma, lam
    )
    # Zero reward + zero value => exact point mass at atom 0; std must be ~0 (no diffusion).
    assert target_std.max() < 1e-5, target_std.max()
    assert (lam_probs.max(-1).values > 0.999).all()


def test_td0_is_one_step_bellman():
    # lambda=0 (td0) target must equal a single C51 projection of r + gamma*Z(s').
    g = torch.Generator().manual_seed(1)
    T, B = 4, 2
    atoms = _atoms(n=81)
    n = atoms.numel()
    gamma = 0.99
    value_probs = torch.softmax(torch.randn(T, B, n, generator=g), -1)
    bootstrap_probs = torch.softmax(torch.randn(B, n, generator=g), -1)
    rewards = torch.randn(T, B, generator=g)
    dones = torch.zeros(T, B)
    next_done = torch.zeros(B)

    lam_probs, _ = distributional_lambda_target(
        rewards, dones, next_done, value_probs, bootstrap_probs, atoms, gamma, 0.0
    )
    # check t=0 directly: target = C51proj(r_0 + gamma * atoms) weighted by Z(s_1)
    loc = rewards[0].unsqueeze(-1) + gamma * atoms.unsqueeze(0)
    expected = c51_project(loc, value_probs[1], atoms)
    assert torch.allclose(lam_probs[0], expected, atol=1e-5)


def test_terminal_collapses_to_reward():
    g = torch.Generator().manual_seed(2)
    T, B = 4, 2
    atoms = _atoms(n=81)
    n = atoms.numel()
    gamma, lam = 0.99, 0.95
    value_probs = torch.softmax(torch.randn(T, B, n, generator=g), -1)
    bootstrap_probs = torch.softmax(torch.randn(B, n, generator=g), -1)
    rewards = torch.randn(T, B, generator=g)
    dones = torch.zeros(T, B)
    dones[2] = 1.0
    next_done = torch.zeros(B)

    lam_probs, _ = distributional_lambda_target(
        rewards, dones, next_done, value_probs, bootstrap_probs, atoms, gamma, lam
    )
    dist_mean = (lam_probs * atoms).sum(-1)
    assert torch.allclose(dist_mean[1], rewards[1], atol=1e-4)


def test_v3_defaults():
    args = Args()
    assert args.dist_target_mode == "td_lambda"
    assert args.value_symlog is False
    assert args.v_min == -20.0 and args.v_max == 20.0
