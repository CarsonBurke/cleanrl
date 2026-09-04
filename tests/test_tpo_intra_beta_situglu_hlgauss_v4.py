import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).parents[1]
GUARD = 20.0


def _load(name, relpath):
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V4 = _load(
    "tpo_intra_beta_situglu_hlgauss_v4",
    "cleanrl/tpo/ppo_continuous_action_tpo_intra_beta_situglu_hlgauss_v4.py",
)
V2 = _load("tpo_intra_beta_v2_ref", "cleanrl/tpo/ppo_continuous_action_tpo_intra_beta_v2.py")


class _Args:
    gamma = 0.99
    reward_clip = 10.0
    num_bins = 511
    value_sigma_to_bin_ratio = 0.75


def _envs_stub():
    import gymnasium as gym

    class Stub:
        single_observation_space = gym.spaces.Box(-np.inf, np.inf, (5,), np.float32)
        single_action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)

    return Stub()


# --- the TPO objective must be byte-for-byte the same behaviour as v2 ---------


def test_tpo_loss_is_identical_to_v2():
    torch.manual_seed(0)
    logratio = torch.randn(64) * 0.5
    utility = torch.randn(64) * 1.5
    for eta in (2.0, 4.0):
        torch.testing.assert_close(
            V4.tpo_intra_loss(logratio, utility, eta, GUARD),
            V2.tpo_intra_loss(logratio, utility, eta, GUARD),
        )


def test_gradient_is_ratio_minus_target_and_zero_at_neutral_utility():
    utility = torch.tensor([-3.0, 0.0, 3.0])
    eta = 4.0
    logratio = torch.tensor([-0.4, 0.0, 0.3], requires_grad=True)
    V4.tpo_intra_loss(logratio, utility, eta, GUARD).sum().backward()
    torch.testing.assert_close(logratio.grad, logratio.detach().exp() - (utility / eta).exp())
    assert logratio.grad[1].abs().item() < 1e-7  # utility 0 at ratio 1


def test_suppressed_samples_keep_the_restoring_gradient():
    utility = torch.zeros(3)
    logratio = torch.tensor([-19.0, -21.0, -1000.0], requires_grad=True)
    V4.tpo_intra_loss(logratio, utility, eta=1.0, logratio_guard=GUARD).sum().backward()
    torch.testing.assert_close(logratio.grad, torch.full((3,), -1.0), atol=1e-6, rtol=0)


def test_variance_free_batch_is_actor_neutral():
    utility = V4.tpo_utility(torch.full((16,), 2.5), utility_clip=3.0)
    torch.testing.assert_close(utility, torch.zeros(16))


# --- HL-Gauss critic ---------------------------------------------------------


def test_support_covers_the_reachable_normalized_return_range():
    bound = V4.normalized_return_abs_bound(_Args())
    assert math.isclose(bound, 10.0 / 0.01)
    lo, hi = V4.hlgauss_coord_bounds(_Args())
    assert lo == -hi
    # The support must actually reach the worst-case discounted return.
    support = V4.make_hlgauss_support(_Args(), torch.device("cpu")).support
    assert support.min().item() <= -bound * 0.999
    assert support.max().item() >= bound * 0.999
    assert support.numel() == _Args.num_bins


def test_support_has_an_exact_zero_bucket():
    support = V4.make_hlgauss_support(_Args(), torch.device("cpu")).support
    assert torch.isclose(support[(_Args.num_bins - 1) // 2], torch.tensor(0.0), atol=1e-6)


def test_projection_is_a_normalised_distribution_centred_on_the_target():
    support = V4.make_hlgauss_support(_Args(), torch.device("cpu"))
    targets = torch.tensor([-250.0, -3.0, 0.0, 3.0, 250.0])
    probs = support.project(targets)
    torch.testing.assert_close(probs.sum(-1), torch.ones(5), atol=1e-5, rtol=0)
    assert torch.all(probs >= 0.0)
    # Decoding the projection recovers the target closely.
    torch.testing.assert_close(support.probs_to_scalar(probs), targets, rtol=0.02, atol=0.5)


def test_value_loss_is_minimised_by_predicting_the_projected_target():
    support = V4.make_hlgauss_support(_Args(), torch.device("cpu"))
    returns = torch.tensor([-12.0, 0.0, 7.5])
    perfect_logits = support.project(returns).clamp_min(1e-20).log()
    perfect = V4.hl_value_loss(perfect_logits, returns, support)
    for shift in (-5.0, 5.0, 40.0):
        wrong = V4.hl_value_loss(support.project(returns + shift).clamp_min(1e-20).log(), returns, support)
        assert wrong > perfect
    uniform = V4.hl_value_loss(torch.zeros(3, _Args.num_bins), returns, support)
    assert uniform > perfect


def test_value_loss_gradient_flows_to_the_logits():
    support = V4.make_hlgauss_support(_Args(), torch.device("cpu"))
    logits = torch.zeros(4, _Args.num_bins, requires_grad=True)
    V4.hl_value_loss(logits, torch.tensor([-1.0, 0.0, 2.0, 30.0]), support).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0.0


# --- SiTU-GLU ----------------------------------------------------------------


def test_situ_glu_cap_product_is_invariant_to_the_learned_allocation():
    branch = V4.SiTUGLUBranch(8, 8)
    baseline = (branch.beta_gate() * branch.beta_up()).item()
    for delta in (-1.3, -0.2, 0.7, 2.0):
        with torch.no_grad():
            branch.log_beta_delta.fill_(delta)
        torch.testing.assert_close(
            (branch.beta_gate() * branch.beta_up()).item(), baseline, rtol=1e-6, atol=0
        )


def test_situ_glu_default_allocation_matches_the_reference_caps():
    branch = V4.SiTUGLUBranch(8, 8)
    assert math.isclose(branch.beta_gate().item(), 4.0, rel_tol=1e-6)
    assert math.isclose(branch.beta_up().item(), 25.0, rel_tol=1e-6)


def test_situ_glu_is_bounded_and_learned_caps_receive_gradient():
    branch = V4.SiTUGLUBranch(16, 16)
    x = torch.randn(256, 16) * 5.0
    out = branch(x)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert branch.log_beta_delta.grad is not None
    assert branch.log_beta_delta.grad.abs().item() > 0.0


def test_situ_glu_output_scale_is_matched_to_the_relusq_stage_it_replaces():
    """The down_gain calibration should keep initial trunk output scale comparable."""
    torch.manual_seed(0)
    h, n = 64, 4096
    x = torch.randn(n, h)
    situ = V4.SiTUGLUBranch(h, h)(x).std().item()
    relusq_stage = V4.layer_init(torch.nn.Linear(h, h))(torch.relu(x).square()).std().item()
    assert 0.25 < situ / relusq_stage < 4.0, f"scale ratio {situ / relusq_stage:.3f}"


def test_gated_width_is_parameter_matched():
    for h in (32, 64, 128):
        assert V4.SiTUGLUBranch(h, h).gated_dim == round(2.0 * (h + 1) / 3.0)


def test_trunks_use_situ_glu_and_no_relusq_or_tanh_modules():
    agent = V4.Agent(_envs_stub(), num_bins=_Args.num_bins)
    branches = [m for m in agent.modules() if isinstance(m, V4.SiTUGLUBranch)]
    assert len(branches) == 4, "two stacked branches in each of actor and critic"
    assert not any(isinstance(m, torch.nn.Tanh) for m in agent.modules())
    # No two linear layers may be adjacent without a nonlinearity between them.
    for trunk in (agent.actor_trunk, agent.critic_trunk):
        assert all(isinstance(m, V4.SiTUGLUBranch) for m in trunk)


# --- agent wiring ------------------------------------------------------------


def test_actor_and_critic_are_separately_callable_networks():
    torch.manual_seed(0)
    agent = V4.Agent(_envs_stub(), num_bins=_Args.num_bins)
    obs = torch.randn(32, 5)
    alpha, beta = agent.policy_params(obs)
    assert alpha.shape == (32, 3) and beta.shape == (32, 3)
    assert torch.all(alpha >= 1.0) and torch.all(beta >= 1.0), "unimodality requires >= 1"
    logits = agent.critic_logits(obs)
    assert logits.shape == (32, _Args.num_bins)
    # The two networks share no parameters, so they can be compiled independently.
    actor_ids = {id(p) for p in agent.actor_trunk.parameters()}
    critic_ids = {id(p) for p in agent.critic_trunk.parameters()}
    assert actor_ids.isdisjoint(critic_ids)


def test_actions_stay_inside_the_action_box_and_replay_is_exact():
    torch.manual_seed(0)
    agent = V4.Agent(_envs_stub(), num_bins=_Args.num_bins)
    obs = torch.randn(64, 5)
    alpha, beta = agent.policy_params(obs)
    dist = torch.distributions.beta.Beta(alpha, beta)
    z = dist.sample().clamp(V4.SAMPLE_EPS, 1.0 - V4.SAMPLE_EPS)
    action = agent.z_to_action(z)
    assert torch.all(action >= agent.action_low) and torch.all(action <= agent.action_high)

    logprob = dist.log_prob(z).sum(1)
    alpha2, beta2 = agent.policy_params(obs)
    replayed = torch.distributions.beta.Beta(alpha2, beta2).log_prob(z).sum(1)
    torch.testing.assert_close(replayed, logprob)


def test_end_to_end_gradient_is_ratio_minus_target_times_dlogpi():
    torch.manual_seed(0)
    agent = V4.Agent(_envs_stub(), num_bins=_Args.num_bins)
    obs = torch.randn(8, 5)
    with torch.no_grad():
        alpha, beta = agent.policy_params(obs)
        dist = torch.distributions.beta.Beta(alpha, beta)
        z = dist.sample().clamp(V4.SAMPLE_EPS, 1.0 - V4.SAMPLE_EPS)
        old_logprob = dist.log_prob(z).sum(1)
    with torch.no_grad():
        agent.actor_alpha.weight.add_(0.05)

    utility, eta = torch.linspace(-2.0, 2.0, 8), 4.0
    a, b = agent.policy_params(obs)
    logprob = torch.distributions.beta.Beta(a, b).log_prob(z).sum(1)
    logratio = logprob - old_logprob
    V4.tpo_intra_loss(logratio, utility, eta, GUARD).sum().backward()
    got = agent.actor_alpha.weight.grad.clone()

    agent.zero_grad(set_to_none=True)
    coefficient = (logratio.detach().exp() - (utility / eta).exp()).detach()
    a2, b2 = agent.policy_params(obs)
    logprob2 = torch.distributions.beta.Beta(a2, b2).log_prob(z).sum(1)
    (coefficient * logprob2).sum().backward()
    torch.testing.assert_close(got, agent.actor_alpha.weight.grad, rtol=1e-4, atol=1e-6)


def test_critic_gradient_does_not_leak_into_the_actor():
    torch.manual_seed(0)
    agent = V4.Agent(_envs_stub(), num_bins=_Args.num_bins)
    support = V4.make_hlgauss_support(_Args(), torch.device("cpu"))
    obs = torch.randn(16, 5)
    V4.hl_value_loss(agent.critic_logits(obs), torch.randn(16) * 5.0, support).backward()
    assert all(p.grad is None for p in agent.actor_trunk.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.critic_trunk.parameters())
